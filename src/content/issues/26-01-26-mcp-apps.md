---
id: MjAyNi0w
title: Anthropic launches the MCP Apps open spec, in Claude.ai
date: '2026-01-26T05:44:39.731046Z'
description: >-
  **Anthropic** has officially absorbed the independent MCP UI project and,
  collaborating with **OpenAI**, **Block**, **VS Code**, **Antigravity**,
  **JetBrains**, and **AWS**, released the **MCP Apps spec** and official
  support in **Claude.ai**. This standard aims to enable a rich ecosystem of
  interoperable applications with rich UI, addressing the proliferation of
  subscription services. Meanwhile, **NVIDIA** introduced **ToolOrchestra** with
  an **8B orchestrator** model trained via scalable reinforcement learning for
  efficient agent orchestration. The concept of Recursive Language Models (RLMs)
  is gaining traction for efficient context management in agent stacks. The
  “Clawdbot” UX pattern emphasizes outcome-first assistant design with tight
  context and tool integration, sparking security concerns around prompt
  injection. **Alibaba** launched **Qwen3-Max-Thinking**, a flagship reasoning
  and agent model with adaptive tool use and strong benchmark scores, now
  available in public evaluation platforms like LM Arena and Yupp.
companies:
  - anthropic
  - openai
  - block
  - vs-code
  - antigravity
  - jetbrains
  - aws
  - nvidia
  - alibaba
  - claude-ai
models:
  - claude-ai
  - toolorchestra-8b
  - qwen3-max-thinking
topics:
  - agent-orchestration
  - reinforcement-learning
  - recursive-language-models
  - context-management
  - user-experience
  - security
  - prompt-injection
  - reasoning
  - adaptive-tool-use
  - model-evaluation
  - benchmarking
people: []
---


**Rich generative UI is all you need.**

> AI News for 1/23/2026-1/26/2026. We checked 12 subreddits, [**544** Twitters](https://twitter.com/i/lists/1585430245762441216) and **24** Discords (**206** channels, and **14285** messages) for you. Estimated reading time saved (at 200wpm): **1208 minutes**. **Our new website** is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on [@smol_ai](https://x.com/Smol_AI)!


3 months after OpenAI floated a trial balloon with [ChatGPT Apps and the Apps SDK at Dev Day 2025](https://news.smol.ai/issues/25-10-06-devday), Anthropic has now officially absorbed [the independent MCP UI project](https://x.com/liadyosef/status/2002104900843679818) and, working with OpenAI, Block, VS Code, Antigravity, JetBrains, AWS, and others, has released both:

- [the MCP Apps spec](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/)
- [official support in Claude.ai](https://x.com/claudeai/status/2015851783655194640)

It's fair to say that ChatGPT Apps haven't exactly taken the world by storm since announcement, but the overall need for a standard format for applications to return rich UI still cannot be denied. Now that MCP Apps have been ratified by all the important players, this is the basis for a rich ecosystem of open source support and applications being able to interoperate, and perhaps one day solve the perpetual never ending pile of $20/month subscriptions piling up in your credit card bills.


---

# AI Twitter Recap


**Agent Orchestration, RLMs, and “Clawdbot/Clawd” as a UX pattern**

- **NVIDIA ToolOrchestra + Orchestrator-8B**: NVIDIA’s ToolOrchestra frames agentic systems as a *small “conductor” model* that alternates reasoning with calls to tools and larger “expert” models (search, code execution, specialist LLMs, frontier generalists). The claim is that an **8B orchestrator** can reach *frontier-level outcomes* via delegation at materially lower cost, trained end-to-end with **scalable RL** using automatically synthesized tool-use environments and multi-turn tasks ([summary](https://twitter.com/TheTuringPost/status/2015565947827110255), [link](https://twitter.com/TheTuringPost/status/2015565962419048712)). Closest technical implication: “controller scale” matters less than *policy quality + tool/model routing* if you can train it with realistic tool-call rollouts.
- **RLMs / recursion-first agent stacks**: Several posts converge on a **Recursive Language Model (RLM)** pattern: pass files and context *by reference* and iteratively pull the minimum slices needed (shell/grep/AST), rather than stuffing everything into context à la ReAct. Dan B illustrates this with file references vs `@file` expansion as deliberate **context management** ([thread](https://twitter.com/irl_danB/status/2015813778504372601)). Daytona is positioning RLMs as “unlimited recursion depth” via per-(sub)agent sandboxes ([guide](https://twitter.com/ivanburazin/status/2015818845303271896), [integration](https://twitter.com/a1zhang/status/2015820458709471640)).
- **“Clawd/Clawdbot” meme → product signal**: The dataset contains a large “Clawdbot” wave (often with Mac mini jokes), but the technically relevant throughline is *outcome-first assistant UX* + **tight context/tool integration**. Kimmonismus explicitly calls this a shift from “more chat” to “more outcome,” suggesting incumbents will scramble to match it ([tweet](https://twitter.com/kimmonismus/status/2015785094791713006)). Others push a cloud-first counterpoint (no local Mac mini) ([MiniMax reply](https://twitter.com/SkylerMiao7/status/2015596649171804613)). There’s also an emerging *security backlash* as soon as “powerful mode” exists: prompt injection remains a system-level blocker for browser/desktop agents ([dilemma](https://twitter.com/fabianstelzer/status/2015671497180827785), [follow-up](https://twitter.com/fabianstelzer/status/2015702808465420614), [Miessler warnings](https://twitter.com/DanielMiessler/status/2015865548714975475)).

**Reasoning model releases & eval dynamics (Qwen, Tencent, ARC, etc.)**

- **Alibaba Qwen3-Max-Thinking**: Alibaba positions Qwen3-Max-Thinking as a flagship reasoning+agent model trained with “massive scale and advanced RL,” emphasizing **adaptive tool-use** (Search/Memory/Code Interpreter) and **test-time scaling/self-reflection**. They cite strong math and agentic search metrics (e.g., **98.0 on HMMT Feb**, **49.8 on HLE**) ([launch](https://twitter.com/Alibaba_Qwen/status/2015805330652111144)). The model is immediately pushed into public eval channels: LM Arena Text Arena ([Arena](https://twitter.com/arena/status/2015803787680808996)) and Yupp ([Yupp](https://twitter.com/yupp_ai/status/2015812409823522952)). Community reaction highlights the *tool-enabled evaluation regime*—claims of outperforming multiple SOTA models on HLE *with search tools* ([commentary](https://twitter.com/kimmonismus/status/2015820838243561742)).
- **Tencent HunyuanImage 3.0-Instruct (image editing)**: Tencent releases an image-editing-focused multimodal model built on an **80B MoE** (13B active), using a “Thinking” schema with native CoT and their **MixGRPO** algorithm; focus is on precise edits that preserve non-target regions and multi-image fusion ([announcement](https://twitter.com/TencentHunyuan/status/2015635861833167074)). LM Arena reports it entering the **top-10 image edit leaderboard** (rank #7) ([Arena](https://twitter.com/arena/status/2015846799446311337)).
- **ARC-AGI cost/perf hacks**: A notable optimization claim: “Recursive Self-Aggregation (RSA) + Gemini 3 Flash” reaching **59.31% on ARC-AGI-2 at ~1/10 cost** vs Gemini Deep Think ([tweet](https://twitter.com/kimmonismus/status/2015717203362926643)). This points to a broader theme: *meta-inference strategies* (aggregation, recursion, pruning) are becoming as important as base model choice.
- **Open models in arenas**: Molmo 2 (Apache 2.0) appears in Arena as a new open model entrant ([Arena](https://twitter.com/arena/status/2015886736136798723)). Separately, Hugging Face Inference Endpoint notes **GLM-4.7-Flash via llama.cpp** with a low hourly price point (Q4_K_M, 24k context) ([ngxson](https://twitter.com/ngxson/status/2015763148523897097))—underscoring a continued commoditization of *fast open-weight inference*.

**RL everywhere: test-time training, GRPO stabilization, RL-as-pretraining, and compute savings**

- **Test-Time Training (TTT) + RL breakthroughs**: A widely shared result claims a Stanford/NVIDIA-style TTT+RL approach that: beats AlphaEvolve, finds a new upper bound for an Erdős overlap problem, produces **A100 kernels ~2× faster** than best human kernels, and beats both best AI+human attempts on AtCoder ([rronak_](https://twitter.com/rronak_/status/2015649459552850113)). This cluster also includes meta-discussion about correctly crediting related approaches (EvoTune) ([Yejin Cho](https://twitter.com/YejinChoinka/status/2015566349444190432)).
- **GRPO training stability knobs**: A small but actionable engineering tip: INTELLECT-2 reports a **`delta=4.0`** parameter that improves GRPO stability ([QGallouedec](https://twitter.com/QGallouedec/status/2015711810108973462)).
- **RL in pretraining (RLP)**: NVIDIA authors announce **RLP (Reinforcement as a Pretraining Objective)** accepted to ICLR 2026, framing RL not as “post-training only” but as integrated into pretraining ([ahatamiz1](https://twitter.com/ahatamiz1/status/2015867794626380146)).
- **Compute reduction via curriculum-like filtering**: AI21’s “Dynamic Data Snoozing” claims up to **3× compute reduction** for RLVR by snoozing examples that are too easy ([DanielGissin](https://twitter.com/DanielGissin/status/2015773616021860522)). If validated, this is a practical recipe: make the sampler policy-aware instead of static.

**Inference infrastructure & dev tooling: vLLM’s “day-0 model support,” VS Code MCP Apps, Cursor subagents**

- **vLLM’s governance and commercialization pressure**: A long Zhihu-derived summary argues vLLM’s “open-source project → startup” shift was driven by the hidden cost of **day-0 support** (weeks/months of confidential pre-integration per new model), the rise of MoE and heterogeneous inference (fp8/int4/sparse attention), and the mismatch with PyTorch Foundation style testing vs vLLM’s multi-node CI needs. It claims the maintainers founded **Inferact Inc** to fund full-time maintainers while keeping vLLM open-source ([thread](https://twitter.com/ZhihuFrontier/status/2015697493288518105)). Related: vLLM shares a practical flag for avoiding OOM on long-context models: `--max-model-len auto` ([vLLM tip](https://twitter.com/vllm_project/status/2015801909316382867)).
- **MCP Apps: tool calls return interactive UI**: The MCP ecosystem announces **MCP Apps** as the first official MCP extension: tool calls can return **interactive UI components** rendered in-chat. VS Code is first major editor shipping support (Insiders now, stable soon) ([VS Code](https://twitter.com/code/status/2015853688594612715), [alexalbert__](https://twitter.com/alexalbert__/status/2015854375051428111)). Anthropic simultaneously ships “interactive work tools in Claude” (Slack drafting, Figma diagrams, Asana timelines) ([Claude](https://twitter.com/claudeai/status/2015851783655194640)). Net: we’re seeing the “tool interface layer” move from raw JSON to *native UI primitives* inside agent loops.
- **Cursor: multi-browser subagents**: Cursor adds multi-browser support via subagents ([Cursor](https://twitter.com/cursor_ai/status/2015863221589049483)), echoing the same direction: parallelized tool execution + better context isolation.

**Kernel LLMs, chip stacks, and “AI for hardware” loops**

- **GPU MODE 2026: post-training Kernel LLMs in public**: GPU MODE outlines a 2026 plan to **post-train a Kernel LLM** and get generated kernels merged into real repos (PyTorch/vLLM), emphasizing “de-slopify kernels” (determinism, reviewer-mergeable PRs), profiler-guided optimization + memory work, and competitions as evals ([marksaroufim](https://twitter.com/marksaroufim/status/2015818791729746350)).
- **Microsoft Maia 200**: Microsoft announces Maia 200 as a custom inference accelerator; Mustafa Suleyman claims it’s the most performant first-party hyperscaler silicon, with **3× FP4 performance** vs Trainium v3 and FP8 above TPU v7 ([Mustafa](https://twitter.com/mustafasuleyman/status/2015845567138816326), [follow-up](https://twitter.com/mustafasuleyman/status/2015825111769841744)). Yusuf Mehdi frames this as infra that makes AI “dependable” ([thread](https://twitter.com/yusuf_i_mehdi/status/2015826703944470701)).
- **Ricursive Intelligence (AI for chip design)**: Ricursive raises a **$300M Series A** aiming at end-to-end chip design as a recursive self-improvement loop between AI and hardware ([company](https://twitter.com/RicursiveAI/status/2015804806384755059), [Anna Goldie](https://twitter.com/annadgoldie/status/2015806107470438685)).

**Safety, misuse, and societal impact (selected items with direct technical relevance)**

- **Elicitation attacks via benign chemistry data**: Anthropic reports that fine-tuning open models on “benign” chemical synthesis content generated by frontier models can significantly increase capability on **chemical weapons** tasks—an “elicitation attack” that scales with frontier model strength ([AnthropicAI](https://twitter.com/AnthropicAI/status/2015870963792142563), [paper link](https://twitter.com/AnthropicAI/status/2015870975238406600)).
- **Dario Amodei’s “Adolescence of Technology” essay**: A major, highly engaged post argues AI is entering an accelerating feedback loop (AI building AI), with risks spanning misuse, power-seeking autonomy, and economic disruption; it also explicitly frames wealth concentration as a society-breaking failure mode ([Dario](https://twitter.com/DarioAmodei/status/2015833046327402527)). Reaction ranges from strong endorsement to critique of how “takeover risk” framing is presented ([Ryan Greenblatt](https://twitter.com/RyanPGreenblatt/status/2015869503385772037)).
- **Agent security in practice**: Multiple posts treat desktop/browser agents as inherently high-risk until prompt injection and sandboxing mature, reinforcing the need for strict isolation, least privilege, and careful handling of credentials ([Miessler](https://twitter.com/DanielMiessler/status/2015865548714975475)).

**Top tweets (by engagement)**

- [“Clawdbot” misuse example (explicitly harmful)](https://twitter.com/0xRacist/status/2015578387641991513)
- [Karpathy on the phase shift to “programming in English” via agents](https://twitter.com/karpathy/status/2015883857489522876)
- [Dario Amodei’s “Adolescence of Technology”](https://twitter.com/DarioAmodei/status/2015833046327402527)


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Local LLM Hardware and Benchmarking

  - **[216GB VRAM on the bench. Time to see which combination is best for Local LLM](https://www.reddit.com/r/LocalLLaMA/comments/1qni356/216gb_vram_on_the_bench_time_to_see_which/)** (Activity: 366): **The post discusses the use of secondhand Tesla GPUs, which offer substantial VRAM at a lower cost, for local large language model (LLM) testing. The author has developed a [GPU server benchmarking suite](https://esologic.com/gpu-server-benchmark/#gpu-box-benchmark) to evaluate the performance of these GPUs when used in parallel. The image shows a technical setup with multiple NVIDIA GPUs, highlighting the focus on maximizing VRAM capacity. The discussion centers around the feasibility and efficiency of using these older GPUs compared to modern devices, particularly in terms of bandwidth and cooling challenges.** Commenters express skepticism about the performance of these GPUs, noting potential issues with bandwidth and cooling. One commenter shares personal experience, comparing different GPU models and highlighting the challenges of using older hardware.

    - HugoCortell raises a technical concern about the potential bandwidth limitations when connecting multiple GPUs to a single PC, noting that most affordable server motherboards support only a few GPUs. This could impact the performance of local LLMs if not addressed properly.
    - dc740 shares insights from personal experience with different GPUs, highlighting that the P40 outperforms the M10 despite both being older models. However, they prefer using AMD Instinct Mi50 GPUs due to their performance, even though support for these was recently dropped from ROCm, indicating a trade-off between hardware capability and software support.
    - FullOf_Bad_Ideas critiques the gpu_box_benchmark for not testing scenarios where large models are split across multiple GPUs, which is a primary use case for setups with extensive VRAM. This points to a gap in current benchmarking practices that may not fully reflect real-world applications of multi-GPU systems.

  - **[I just won an Nvidia DGX Spark GB10 at an Nvidia hackathon. What do I do with it?](https://www.reddit.com/r/LocalLLaMA/comments/1qn3xig/i_just_won_an_nvidia_dgx_spark_gb10_at_an_nvidia/)** (Activity: 724): **The image shows a terminal window on a Linux system running the 'top' command, which is used to monitor system processes and resource usage in real-time. The user has won an Nvidia DGX Spark GB10, a high-performance computing device designed for machine learning and data-intensive tasks. The terminal indicates a Python process consuming significant CPU resources, suggesting active computational tasks, possibly related to machine learning or data processing. The user is considering using the device to run multiple NextJS applications simultaneously, leveraging its powerful capabilities.** One commenter suggests running three NextJS applications simultaneously, indicating the device's capability to handle multiple high-memory tasks. Another commenter provides a link to Nvidia's DGX Spark playbooks, which could be useful for the user to explore the full potential of their new hardware.

    - Fit-Produce420 highlights the capabilities of the Nvidia DGX Spark GB10, noting that with 128GB of memory, it can fine-tune models up to 70 billion parameters. Additionally, it can handle larger models like the 120 billion parameter `gtp-oss-120b` using techniques like QLoRA, which optimizes memory usage for large-scale models. However, running dense models like `devstral 2` may be slow due to their computational demands.
    - randomfoo2 suggests utilizing the [NVIDIA DGX Spark playbooks](https://github.com/NVIDIA/dgx-spark-playbooks) as a resource for getting started with the DGX Spark GB10. These playbooks provide structured guidance and best practices for deploying and managing workloads on the DGX platform, which can be particularly useful for users new to this hardware.
    - LicensedTerrapin humorously suggests selling the DGX Spark GB10 to purchase 8GB of DDR5 RAM, implying a trade-off between high-end specialized hardware and more general-purpose upgrades. This comment reflects a common debate in tech communities about the value of specialized versus general-purpose hardware investments.

  - **[Using a high-end MacBook Pro or a beefy RTX 5090 laptop (with 24 GB of RAM) for inference.](https://www.reddit.com/r/LocalLLM/comments/1qnpti6/using_a_highend_macbook_pro_or_a_beefy_rtx_5090/)** (Activity: 29): **The post discusses the feasibility of using a high-end MacBook Pro with Apple Silicon (M-series Max) versus a Windows/Linux laptop with an RTX 5090 GPU for running large local LLMs (70B+ parameters) for inference and fine-tuning. The MacBook Pro offers 128–192 GB of unified memory, while the RTX 5090 laptop provides 24 GB of VRAM and at least 64 GB of system RAM. The primary use case is local LLM inference with a target of ≥15 tokens/sec, emphasizing portability. The post queries whether the larger unified memory of Apple Silicon outweighs the CUDA performance of the RTX laptop for inference, and how Apple MLX compares to CUDA for fine-tuning tasks like LoRA/QLoRA. It also seeks insights on thermal performance and sustained inference capabilities of both setups.** One commenter suggests using the laptop as a terminal to a more powerful desktop, indicating a preference for leveraging remote resources over local hardware. Another commenter is experimenting with both setups, using a MacBook Pro M2 Max for inference, and is curious about the performance differences.

    - racerx509 shares their experience using a Lenovo laptop with a 3070ti, a custom desktop with a 5070, and a MacBook Pro M2 Max with 96GB RAM for inference tasks. They note that they have been primarily using the MacBook Pro for inference, suggesting it may offer better performance or convenience for their needs.
    - No-Concern-8832 raises a concern about the VRAM limitations of RTX laptops, suggesting that they may not be sufficient for running large models like 70B parameters. This highlights a potential limitation in using high-end RTX laptops for certain deep learning tasks that require substantial VRAM.
    - Tired__Dev discusses their experience with an Asus M16 equipped with a 4090 GPU, noting that it struggled with a 7B parameter model. They express a preference for a MacBook Pro with 128GB RAM, citing its high memory bandwidth and potential performance advantages over even high-end GPU setups like the DGX Spark.


### 2. Multi-Agent Systems and AI Assistants

  - **[I built a "hive mind" for Claude Code - 7 agents sharing memory and talking to each other](https://www.reddit.com/r/LocalLLaMA/comments/1qnjota/i_built_a_hive_mind_for_claude_code_7_agents/)** (Activity: 313): **The post describes a multi-agent orchestration system for **Claude Code**, featuring seven specialized agents (e.g., coder, tester, reviewer) that coordinate tasks, share persistent memory using `SQLite + FTS5`, and communicate via a message bus. The system runs as an MCP server and integrates with **Anthropic**, **OpenAI**, or **Ollama**. It uses a task queue for priority-based coordination, allowing agents to pass context and collaborate effectively. The implementation stack includes **TypeScript**, **better-sqlite3**, **MCP SDK**, and **Zod**. The project is experimental, open-source under the MIT license, and available on [GitHub](http://github.com/blackms/aistack).** A comment questions the system's uniqueness compared to the [BMAD method](https://github.com/bmad-code-org/BMAD-METHOD), suggesting similarities. Another comment humorously questions whether the agents agree with each other, hinting at potential coordination challenges.

    - The user robiinn inquires about the differences between the 'hive mind' system and the [bmad method](https://github.com/bmad-code-org/BMAD-METHOD), suggesting a potential similarity. This indicates a need for clarification on the unique aspects or improvements of the 'hive mind' approach over existing methods, such as how memory sharing and inter-agent communication are implemented differently.
    - No_Afternoon_4260 raises a critical point about the consensus among the agents in the 'hive mind'. This touches on the technical challenge of ensuring that multiple agents can not only share memory but also reach agreement or consensus, which is a significant aspect of distributed systems and multi-agent frameworks.
    - JellyBean504 draws a parallel between the 'hive mind' and Steve Yegge's Gastown, suggesting that there might be conceptual similarities. This comparison could be valuable for understanding the architectural or functional parallels between the two systems, potentially offering insights into design choices or performance characteristics.

  - **[Clawdbot: the AI assistant that actually messages you first](https://www.reddit.com/r/LocalLLM/comments/1qmrwxl/clawdbot_the_ai_assistant_that_actually_messages/)** (Activity: 214): ****Clawdbot** is an open-source AI assistant with over `9K` GitHub stars, designed to proactively message users, unlike traditional AI assistants that wait for prompts. It integrates with locally hosted LLMs via **Ollama** and supports messaging apps like WhatsApp, Telegram, and Discord. Key features include sending automated briefings and reminders, local storage of conversations as Markdown files, and the ability to control browsers and run scripts. The software is free under the MIT license but requires terminal proficiency for setup, as there is no GUI installer. [Read more](https://medium.com/@jpcaparas/what-are-people-doing-with-clawdbot-e91403383ccf?sk=4fbaffdc31974eab844ea93c2f9b627f).** Users report challenges with setup, particularly with obtaining and using OAuth keys for authentication, and difficulties in connecting local LLMs without relying on API keys. Some users express frustration with the complexity of setup, especially when using remote machines.

    - mike7seven highlights the complexity of setting up Clawdbot, particularly emphasizing the need to obtain a Claude OAuth key on a separate machine and then transfer it to the setup machine. This process is noted as cumbersome, especially for those using remote machines, and the MacOS app requires building from source, adding another layer of complexity.
    - Ashamed_Promise7726 raises a technical challenge regarding the integration of local language models with Clawdbot. The user notes difficulty in connecting pre-downloaded models on their PC, as Clawdbot seems to require an API key for usage-based models, questioning the feasibility of running Clawdbot entirely locally without external dependencies.
    - inigid warns about potential security risks associated with Clawdbot, suggesting it could be exploited for supply-chain attacks that compromise sensitive data on a user's machine and network. The comment also mentions concerns about the association with Solana meme coins, implying a need for caution when using the tool.


### 3. GLM-4.7-Flash Performance Updates

  - **[GLM-4.7-Flash is even faster now](https://www.reddit.com/r/LocalLLaMA/comments/1qmvny5/glm47flash_is_even_faster_now/)** (Activity: 443): **The recent update to `llama.cpp` by **Johannes Gaessler** optimizes the CUDA implementation of FlashAttention, specifically for models with a non-power-of-2 ratio of query heads to key/value heads. This is achieved by padding Q columns to the next power of 2, which, although slightly inefficient, enhances performance for small batch sizes. The update is detailed in [pull request #19092](https://github.com/ggml-org/llama.cpp/pull/19092).** One comment humorously notes the obsolescence of a previous post due to this update, while another laments the lack of support for AMD GPUs, highlighting a common issue in the community regarding hardware compatibility.

    - The user 'jacek2023' provides detailed performance metrics for the GLM-4.7-Flash model, highlighting its efficiency. The model processes a prompt with `45074` tokens, achieving a prompt evaluation time of `2814.63 ms` for `1612` tokens, which translates to `1.75 ms per token` or `572.72 tokens per second`. The overall evaluation time is `29352.57 ms` for `1731` tokens, equating to `16.96 ms per token` or `58.97 tokens per second`. The total processing time is `32167.20 ms` for `3343` tokens, indicating significant improvements in speed.

  - **[KV cache fix for GLM 4.7 Flash](https://www.reddit.com/r/LocalLLaMA/comments/1qmjzx1/kv_cache_fix_for_glm_47_flash/)** (Activity: 380): **The recent update to **GLM 4.7 Flash** involves removing the V component from the KV cache, which significantly reduces VRAM usage, allowing for longer context lengths on the same hardware setup. This change is particularly beneficial for models like **DeepSeek** and **GLM 4.7 Flash**, as it can save gigabytes of VRAM, enabling context lengths to double, as demonstrated by a user running a 90,000 context on a 4090 GPU. The update is part of a pull request in the `llama.cpp` repository, which introduces a V-less KV cache, reducing memory usage by nearly 50%. More details can be found in the [pull request](https://github.com/ggml-org/llama.cpp/pull/19067).** A user noted that the model, while improved, still requires some manual guidance, especially in tasks like coding and creative writing, where it may not perform as well as specialized models. However, it excels in tool use and as an assistant, making it a preferred choice for home-server applications.

    - The user 'teachersecret' reports significant improvements in context handling with the UD's k_xl 4-bit version of the GLM 4.7 model on an RTX 4090. Previously, the model maxed out at 45,000 context tokens, but now it can handle 90,000. Despite these improvements, the model still requires some manual guidance, especially in coding tasks, and is less effective in creative writing compared to other models. However, it excels in tool usage and is now the user's default model for their home server.
    - User 'viperx7' provides detailed benchmark data comparing the performance of the GLM 4.7 model before and after a specific change. The benchmarks show improvements in both prompt processing and token generation speeds across different configurations. For instance, using a single RTX 4090, the context size increased from 64k to 128k, with prompt processing speed improving from 3489 t/s to 3510 t/s and token generation from 88 t/s to 92.5 t/s. The maximum context size achievable with a 4090 and 3060 setup is 200k, leaving about 6GB of VRAM unused.
    - The discussion highlights the technical aspect of the GLM 4.7 model's KV cache fix, which allows for increased context sizes and improved performance metrics. The benchmarks provided by 'viperx7' indicate that the model can now handle up to 207k context size in certain configurations, with significant improvements in processing speeds. This suggests that the model's efficiency has been enhanced, making it more suitable for high-demand applications.


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude AI Usage and Issues

  - **[Why You Need To Constantly Clear Claude Codes Context Window](https://www.reddit.com/r/ClaudeCode/comments/1qmrkr1/why_you_need_to_constantly_clear_claude_codes/)** (Activity: 166): **The post highlights the necessity of regularly clearing the context window when using coding agents like Claude to maintain optimal performance. It notes that performance degrades significantly when the context window exceeds `40%` of its capacity due to the quadratic nature of LLM attention, which increases computational demands and introduces noise. The recommended practice is to avoid accumulating context and instead persist it by using a 'one session per task' strategy, ensuring each task starts with a fresh context. More details can be found in the [original article](https://willness.dev/blog/one-session-per-task).** Commenters suggest practical strategies such as using handover prompts to transfer necessary details between sessions, employing the '/clear' command to compact context, and utilizing 'Plan Mode' to clear context and execute tasks efficiently. These methods reportedly help avoid the need for a full context window, even for large tasks.

    - Agrippanux suggests using 'Plan Mode' as the default setting for Claude, which allows users to clear the context and execute plans without needing a full context window. This approach has been effective for large tasks, such as refactoring, without requiring the entire context to be loaded, thus optimizing performance and resource usage.
    - thurn2 discusses the use of sub-agents in Claude, which involves delegating tasks like creating a git worktree and fixing specific issues. This method allows for parallel execution of tasks and helps in managing complex projects by breaking them down into smaller, manageable tasks, enhancing efficiency and implementation accuracy.
    - Fancy_Excitement6050 notes that as the context window grows, Claude tends to take shortcuts, which can lead to a need for constant reminders to maintain thoroughness. This suggests that managing the context window size is crucial for maintaining the quality of output, and there might be differences in performance between different Claude plans, such as Claude Max.

  - **[Opus fell off? Here’s the workflow that kept my code quality stable](https://www.reddit.com/r/ClaudeCode/comments/1qnhgcc/opus_fell_off_heres_the_workflow_that_kept_my/)** (Activity: 133): **The post discusses a structured workflow to maintain code quality when using AI models like **Opus** and **Sonnet**, which have been perceived as producing "confident wrong" outputs and drifting edits. The workflow emphasizes a loop of **specification, ticket creation, execution, and verification**. Specifications are detailed with non-goals, user stories, acceptance criteria, edge cases, and more, treated as code to ensure clarity. Tickets are derived from specs, focusing on small, independently mergeable tasks with clear acceptance checks. Execution involves implementing one ticket at a time with constraints to prevent scope drift, and verification involves running tests and confirming acceptance criteria before feeding failures back into the model for correction. This approach aims to maintain discipline and reduce reliance on the model's "done" signal, ensuring stable and reliable outputs.** Commenters agree that the workflow is effective, emphasizing that AI models function more like junior engineers requiring clear specifications and strict feedback loops. This approach shifts effort towards upfront clarity and external verification, making the system more stable and less reliant on the model's intelligence. Smaller scoped tickets and hard verification are noted as beneficial strategies.

    - GenOS2312 highlights the importance of treating LLMs like junior engineers, emphasizing that a well-specified problem and a strict feedback loop are crucial for reliable outputs. The workflow discussed focuses on upfront clarity and external verification, which stabilizes the system by not relying on the model's intelligence but rather constraining it to ensure even average runs yield acceptable results.
    - Different-Object5926 notes that smaller scoped tickets combined with hard verification processes significantly improve the stability and reliability of using models like Opus. This approach mitigates the impact of variability in model performance, suggesting that the issue isn't just 'unlucky runs' but rather the need for structured constraints.
    - TheOriginalAcidtech suggests implementing hooks to prevent skipping steps in the workflow, emphasizing that the human interface is often the weakest link. By enforcing strict adherence to the process, the system can better manage user interactions, ensuring that the model and its harness guide the user effectively, rather than relying solely on the model's capabilities.

  - **[after claude now chatgpt is also uses Grokipedia as source](https://www.reddit.com/r/singularity/comments/1qn325q/after_claude_now_chatgpt_is_also_uses_grokipedia/)** (Activity: 634): **The image and accompanying discussion highlight that the latest version of **ChatGPT** is reportedly using **Elon Musk's Grokipedia** as a source. This is significant as it suggests a shift in the data sources used by ChatGPT, potentially affecting the information quality and bias in its responses. The comments reveal a concern about the implications of using Grokipedia, particularly regarding the potential for biased information, as one user notes the risk of models being influenced by 'right wing' content. However, it is clarified that Grokipedia is not used as training data but rather as a search tool, which may mitigate some concerns about direct bias in the model's foundational knowledge.**

    - The discussion highlights concerns about language models like Claude and ChatGPT potentially using sources like Grokipedia, which may have biased or unreliable content. This raises questions about the integrity of the information these models provide, especially when they utilize search tools to access real-time data. The implication is that the quality and neutrality of the data sources are crucial for maintaining the accuracy and trustworthiness of AI outputs.
    - There is a debate about the impact of using sources like Grokipedia on the training and performance of language models. Some commenters express concern that incorporating biased or politically skewed sources could lead to the dissemination of misinformation. This reflects broader worries about the influence of data sources on the objectivity and reliability of AI-generated content.
    - The mention of Reddit as a data source for language models suggests a comparison of potential biases. While some argue that Reddit may contain more extreme or varied viewpoints, the underlying issue is the challenge of ensuring that AI models are trained on balanced and factual data. This discussion underscores the importance of curating high-quality datasets to prevent the spread of biased information.

  - **[Giving Claude full access to a laptop](https://www.reddit.com/r/ClaudeAI/comments/1qm8tvj/giving_claude_full_access_to_a_laptop/)** (Activity: 795): **The post discusses the implementation of giving **Claude**, an AI model, full access to a laptop, allowing it to autonomously manage a virtual machine (VM) on Ubuntu Google Cloud. The user describes how Claude can be remotely controlled via Discord to build new features and fix bugs, logging major actions with timestamps in a markdown file for memory management. This setup enables the user to learn from Claude's problem-solving processes and manage workflows effectively, even as a newcomer to programming.** One commenter, a desktop support technician, expressed amazement at the implementation, noting its potential impact on job roles, while another sought clarification on the technical specifics of giving Claude full device access.

    - _xxxBigMemerxxx_ describes using Claude to manage a Google Cloud VM running Ubuntu, highlighting its ability to autonomously handle tasks and build features. They mention using Discord for remote requests and bug fixes, and implementing a logging system with markdown and Unicode for tracking changes. This setup allows for a dynamic interaction with Claude, enabling it to learn from errors and maintain a form of short-term memory by logging recent updates.
    - Happy_Requirement187 shares their experience running Claude on an AWS EC2 instance with Ubuntu Linux, accessed via SSH from a Windows laptop. They utilize a Jupyter notebook server for seamless file sharing between the EC2 instance and their local environment, a method recommended by Anthropic. Additionally, they have set up a Ruby on Rails environment with a React frontend for secure file sharing, allowing them to request files via Slack, demonstrating a sophisticated integration of Claude into their workflow.
    - sivadneb inquires about setting up voice control in Linux, indicating a technical challenge in integrating voice commands with Claude. This suggests an interest in expanding the interaction capabilities with Claude beyond text-based commands, potentially enhancing the usability and accessibility of the system.

  - **[CLAUDE.md says 'MUST use agent' - Claude ignores it 80% of the time.](https://www.reddit.com/r/ClaudeCode/comments/1qn9pb9/claudemd_says_must_use_agent_claude_ignores_it_80/)** (Activity: 309): **The image and post discuss a technical issue with the CLAUDE.md file, which is supposed to direct the AI, Claude, to use a specific agent for workflow questions. Despite explicit instructions in the file, Claude often defaults to a generic agent, indicating a lack of enforcement in the system. The post suggests that without technical enforcement mechanisms, such as hooks or stronger prompts, instructions are merely suggestions. The image emphasizes these points with highlighted text, suggesting potential solutions like adding enforcement hooks to ensure compliance with the specified workflow.** Commenters suggest that the issue may stem from unclear instructions, emphasizing the need for simple and direct commands. They also highlight the importance of implementing technical solutions, such as hooks, to enforce compliance with the CLAUDE.md instructions.

    - Accomplished_Buy9342 suggests using hooks to manage Claude's behavior, providing a link to a GitHub repository that demonstrates how to block the main chat from performing actions and delegate tasks to a subagent. This approach can help in orchestrating Claude's actions more effectively, especially when dealing with complex tasks or large contexts.
    - luka5c0m highlights a common issue with Claude when used at scale: as the context grows beyond a few files, the agent may perform unexpected actions. They suggest that instead of relying solely on better prompts, developers should use hooks and dynamic instructions to maintain a sharp and concise context. They also mention working on a dynamic CLAUDE.md file that adapts to the current task, which could help in managing large or nested files effectively.

  - **[My Ralph Wiggum breakdown just got endorsed as the official explainer](https://www.reddit.com/r/ClaudeCode/comments/1qm5vmh/my_ralph_wiggum_breakdown_just_got_endorsed_as/)** (Activity: 170): **The post discusses a video breakdown of Ralph Wiggum, an autonomous coding loop, which has been endorsed by **Geoffrey Huntley** as the official explainer. Ralph Wiggum is a `bash while loop` that calls **Claude** in headless mode, allowing for autonomous code implementation without context degradation. Key features include avoiding the **Anthropic Ralph plugin** due to performance issues, using fresh context windows for each iteration, and emphasizing the importance of concise specs to prevent hitting a "dumb zone." The video link is [here](https://youtu.be/I7azCAgoUHc).** The comments include a link to the endorsement post by Geoffrey Huntley, and general positive feedback on the video, indicating its usefulness and quality.

    - Dennis1451 highlights a practical application of the Ralph Wiggum breakdown, noting the importance of using a well-defined specification and clearing context for optimal results. They mention using 'auto compact' without a clear spec initially, which suggests that following the guidelines provided in the breakdown could enhance performance and accuracy.
    - messiah-of-cheese expresses a desire for more scientific validation in the video, particularly regarding the 'dumb zone' premise. This indicates a need for empirical evidence or data to support the claims made in the breakdown, which could strengthen its credibility and acceptance among a technical audience.


### 2. ICLR and ICML 2026 Conference Discussions

  - **[[D] ICLR 2026 decision mega thread](https://www.reddit.com/r/MachineLearning/comments/1qm32o6/d_iclr_2026_decision_mega_thread/)** (Activity: 1589): **The post announces the imminent release of ICLR 2026 review decisions, with anticipation heightened due to a previous incident involving OpenReview. The community is preparing for the outcomes, with some users humorously sharing acceptance prediction models based on historical data, such as a simple `return uniform(0, 1) > 0.7`. This reflects a light-hearted approach to the uncertainty of paper acceptance.** The comments reflect a mix of anticipation and humor, with some users expressing frustration over misleading emails from other conferences like ICML, which adds to the tension of awaiting ICLR decisions.


  - **[[D] ICML 2026 - ICML desk-rejected my paper but kept me on as a reviewer. Wow?](https://www.reddit.com/r/MachineLearning/comments/1qmhyin/d_icml_2026_icml_deskrejected_my_paper_but_kept/)** (Activity: 279): **The post highlights a situation where an author's paper was desk-rejected by **ICML 2026**, yet they were retained as a reviewer. This reflects a common practice in academic conferences where the author and reviewer pipelines are separate; desk rejections often occur due to scope or formatting issues, while reviewer selection is based on past service or keyword matching. This situation underscores the reliance on unpaid labor in academia, where reviewing is seen as community service, but the feedback loop for authorship and recognition is weak.** A notable opinion from the comments suggests that the separation between the author and reviewer roles can feel insulting, as these decisions are made by different parts of the conference organization. It highlights the need for conferences to clarify this separation to avoid personal affronts.

    - AccordingWeight6019 highlights a systemic issue in academic publishing where the processes for desk rejection and reviewer selection are distinct. Desk rejections often occur due to scope or formatting issues, while reviewer selection is based on past service or keyword matching. This separation can lead to feelings of insult among authors, but it's a structural necessity due to the different roles and responsibilities within the publication process. The comment suggests that conferences should improve transparency about these processes to mitigate personal feelings of rejection.
    - mocny-chlapik points out that the responsibility for a desk rejection often lies with the author, particularly if it results from not following submission guidelines. The comment implies that submitting a paper, even if desk rejected, obligates the author to fulfill reviewer duties, as the submission process involves volunteer time and resources. This highlights the importance of adhering to submission instructions to avoid unnecessary strain on the peer review system.

  - **[[R] Appealing ICLR 2026 AC Decisions...](https://www.reddit.com/r/MachineLearning/comments/1qnh14y/r_appealing_iclr_2026_ac_decisions/)** (Activity: 138): **The post discusses a situation where an author received mixed reviews for a paper submitted to ICLR 2026, with scores of `4(3)/6(4)/6(4)/6(4)`. The author invested significant resources, including `$1.6k` on new experiments and added `20+ pages` of theory, to address reviewer concerns. Despite these efforts, the metareview cited "outstanding concerns" that the author believes were addressed, raising questions about the review process's fairness and accuracy. The author is seeking advice on appealing the decision, expressing frustration that improvements were seemingly ignored.** Commenters generally agree that appealing decisions at conferences like ICLR is not feasible, attributing outcomes to luck and the subjective nature of reviews. Some suggest that the meta-review process can be inconsistent, with one commenter noting that meta-reviewers sometimes act as an additional critical reviewer, potentially skewing outcomes.

    - tedd235 discusses the variability in paper acceptance at conferences, suggesting that some PhD students might reject papers to improve their own odds, making the process feel like a 'coin flip'. They note that if other reviewers provide higher scores, the Area Chair (AC) might consider this in their decision, indicating a potential for subjective bias in the review process.
    - Fantastic-Nerve-4056 shares an experience from AAMAS where despite receiving scores of 6 and 8 from reviewers, the Meta Reviewer recommended rejection with minimal justification, stating it was 'relevant for other AAMAS session'. This highlights issues with the transparency and accountability of meta-reviewer decisions, which can override individual reviewer scores without detailed explanation.
    - Intrepid_Discount_67 describes a thorough submission process, including extensive theoretical analysis, comprehensive baseline comparisons, and open-sourced code, yet faced non-responsive reviewers and an AC that upheld the initial scores. This underscores challenges in the review process where detailed responses and transparency do not necessarily lead to favorable outcomes.

  - **[[D] ICML new policy: reviewers will be reviewed by meta reviewer. Good policy?](https://www.reddit.com/r/MachineLearning/comments/1qmi3oe/d_icml_new_policy_reviewers_will_be_reviewed_by/)** (Activity: 151): **The image describes a new policy implemented by the International Conference on Machine Learning (ICML) where reviewers will be evaluated by meta-reviewers. The top 25% of reviewers will be recognized as 'gold reviewers' and will receive free registration, while the next 25% will be designated as 'silver reviewers.' These distinctions are intended to incentivize high-quality reviews and will be considered in financial aid applications. This policy aims to improve the quality of reviews by providing recognition and potential financial benefits to diligent reviewers.** Some commenters express skepticism about the effectiveness of this policy, questioning who will oversee the meta-reviewers themselves. Others see it as a positive step, particularly for reviewers from low-resource backgrounds, and suggest further recognition at conferences to encourage quality reviewing.

    - Bitter-Reserve3821 highlights that area chairs have traditionally been responsible for rating reviews, typically using a three-tier system: 'did not meet expectations', 'satisfactory', or 'exceeded expectations'. This practice is not new, and there have been 'Best Reviewer' awards in the past, sometimes offering incentives like free conference registrations.
    - Unhappy_Craft1906 raises a concern about the feasibility of this policy for top labs with substantial funding, questioning whether they would participate in the review process merely for free registrations. This points to a potential disparity in how different institutions might engage with the policy based on their resources.
    - newperson77777777 suggests an extension of the policy by introducing a visible recognition system, such as a gold or silver star on conference badges, to incentivize quality reviewing. This idea aims to foster a culture of excellence and accountability within the reviewing community.


### 3. OpenAI and AI Industry Legal and Business Developments

  - **[Things Get Worse For OpenAI: Consumer groups prep class action suits about their price fixing and supply manipulation through DRAM hoarding.](https://www.reddit.com/r/DeepSeek/comments/1qmih28/things_get_worse_for_openai_consumer_groups_prep/)** (Activity: 107): **OpenAI is facing potential class action lawsuits for allegedly hoarding DRAM to manipulate prices and disadvantage competitors, with accusations of securing nearly `40%` of the global DRAM supply. Consumer groups argue this constitutes 'predatory bidding' and violates antitrust laws like the Sherman and Clayton Acts. The Free Software Foundation and other groups are pursuing legal remedies, arguing DRAM should be considered an 'Essential Facility' due to its critical role in AI, while the FTC and European Commission investigate potential violations of competition laws. The DOJ is also examining whether OpenAI's 'Stargate' project constitutes a 'monopsony'.** Commenters question why only OpenAI is targeted and not other companies like Nvidia, and debate whether buying RAM constitutes price fixing, suggesting that supply issues may not be OpenAI's fault.

    - Alacritous69 argues that OpenAI's purchase of RAM does not constitute price fixing, as they are actively using the resources rather than hoarding them. The commenter suggests that the issue lies with suppliers' inability to meet demand, rather than any manipulative practices by OpenAI.
    - sambull raises a strategic business perspective, suggesting that by purchasing large quantities of RAM, OpenAI could be intentionally limiting resources available to competitors, including those developing at-home language models. This could be seen as a competitive strategy to maintain market dominance.
    - max6296 questions why the focus is solely on OpenAI when Nvidia could also be implicated in similar practices, hinting at a broader industry issue regarding resource allocation and market influence.

  - **[When Ads aren't enough: OpenAI's push to Claim a Cut of Customers' AI Discoveries](https://www.reddit.com/r/DeepSeek/comments/1qmqi62/when_ads_arent_enough_openais_push_to_claim_a_cut/)** (Activity: 63): ****OpenAI** is exploring new business models beyond traditional subscriptions and ads, focusing on **outcome-based pricing** and **IP-based agreements**. This approach would allow OpenAI to claim a share of the value created when their AI models contribute to profitable outcomes, particularly in enterprise sectors like pharma, scientific research, and energy systems. This strategy aligns OpenAI's revenue with customer success, aiming to capture more value as AI capabilities expand. OpenAI's annualized recurring revenue has surged from `2B` in 2023 to over `20B` in 2025, driven by increased compute scaling. This move is part of a broader trend among AI firms towards value-based pricing, amidst criticism from figures like **Elon Musk**, who accuses OpenAI of abandoning its nonprofit origins.** The community is divided, with some viewing this as a logical evolution of AI monetization, while others criticize it as overly profit-driven. Comparisons are drawn to other industries, suggesting skepticism about the feasibility and fairness of such models.


  - **[CATL, the world's largest battery maker, launches sodium batteries: extremely durable, stable at –40°C, much cheaper than lithium (5x), safer,10,000 charge cycles, requires no nickel or cobalt...](https://www.reddit.com/r/singularity/comments/1qnklek/catl_the_worlds_largest_battery_maker_launches/)** (Activity: 1289): ****CATL** has launched the first mass-produced sodium-ion batteries, offering a cost-effective alternative to lithium-ion with a price of `~$20 per kWh` compared to lithium's `~$100 per kWh`. These batteries, part of the Tianxing II range, are designed for microvans and small trucks, featuring an energy density of `175 Wh/kg` and a lifespan of over `10,000 cycles`, maintaining `90% capacity` at `-40°C`. They utilize a hard carbon electrode and prussian-blue cathode, eliminating the need for nickel or cobalt, and are expected to be scaled up for broader use, including in Europe by 2026. [Read more](https://evmarket.ro/en/baterii-masini-electrice/catl-baterii-pe-sodiu-stabile-la-40c-58935/).** Some commenters express surprise at the application of sodium batteries in vehicles, expecting them to be used in stationary systems due to weight concerns. Others note the strategic advantage for China in advancing battery technology, contrasting it with perceived setbacks in the US market.

    - The Tianxing II range of sodium batteries by CATL is specifically designed for microvans, light vans, and small trucks, indicating a focus on applications where energy density and weight are less critical compared to cost and durability. This suggests a strategic move to target markets where these factors are prioritized, potentially offering a competitive edge over traditional lithium-ion batteries.
    - The introduction of sodium batteries into vehicles is surprising to some, as it was expected that such technology would first be applied to stationary applications like home energy storage. This is due to the lower energy density of sodium batteries compared to lithium-ion, which makes them less ideal for applications where weight and size are critical factors.
    - There is curiosity about the commercial availability of these sodium batteries, with questions about whether they can be purchased directly for home use or if they will be distributed through third-party vendors. The performance metrics, such as 10,000 charge cycles and operation at -40°C, are impressive and suggest that sodium batteries could rival LiFePO4 in terms of performance, especially given their cost advantage.

  - **[K-Shaped AI Adoption?](https://www.reddit.com/r/singularity/comments/1qms27i/kshaped_ai_adoption/)** (Activity: 748): **The image highlights a discussion by Kevin Roose on the 'K-shaped' adoption of AI technologies, where there is a significant divide between early adopters, particularly in tech hubs like San Francisco, and those who are lagging due to restrictive IT policies. This disparity is creating a cultural and technical divide, with early adopters integrating AI deeply into their workflows, while others struggle to gain access to even basic AI tools. The conversation points to a broader issue of accessibility and the potential for some workers to be left behind in the AI revolution.** Commenters note that the disparity in AI adoption is exacerbated by the complexity of the technology, which requires a certain level of expertise to use effectively. Additionally, the high cost of advanced AI tools, such as 'multi-agent claudeswarm,' limits access to those with sufficient financial resources, further widening the gap.

    - Setsuiii highlights the technical barrier to effective AI use, noting that current AI technologies require users to have a certain level of expertise to achieve optimal results. This complexity, combined with ongoing ethical debates surrounding AI, may deter widespread adoption. However, those who can navigate these challenges have significant opportunities, although competition is increasing as more technically adept individuals enter the field.
    - Glxblt76 and Gubzs discuss the financial barriers to AI adoption, particularly the high costs associated with advanced AI tools like a 'multi-agent claudeswarm,' which can cost around $200 a month. This expense limits access to those with substantial financial resources, such as individuals in tech hubs like San Francisco, while the majority cannot afford such investments.
    - o5mfiHTNsH748KVq shares a personal experience of leaving an enterprise job to join a smaller company, emphasizing the importance of unrestricted access to Large Language Models (LLMs) for maintaining competitiveness in the AI field. They argue that any limitations on LLM access can significantly hinder development speed and career progression, suggesting that smaller companies may offer more flexibility in leveraging AI technologies.

  - **[Former Harvard CS Professor: AI is improving exponentially and will replace most human programmers within 4-15 years.](https://www.reddit.com/r/singularity/comments/1qmeo8h/former_harvard_cs_professor_ai_is_improving/)** (Activity: 1260): ****Matt Welsh**, a former Harvard CS professor and current Engineering Director at Google, predicts that AI will advance exponentially, potentially replacing most human programmers within `4-15 years`. This assertion is based on the rapid improvements in AI capabilities, suggesting a transformative impact on software development and the tech industry. The discussion is available in a [YouTube video](https://youtu.be/7sHUZ66aSYI?si=uKjp-APMy530kSg8).** One comment highlights the potential for AI to not only replace programmers but also to enable anyone with AI to replicate existing products and services, indicating a broader impact on innovation and competition.

    - The claim that AI will replace most human programmers within 4-15 years is met with skepticism, particularly regarding the use of the term 'exponential'. Critics argue that the term is often misused, even by experts, to describe growth that may not fit the mathematical definition of exponential growth. This misuse can lead to misunderstandings about the actual pace and nature of AI development.
    - The discussion highlights the potential for AI to disrupt existing products and services if it can indeed replace human programmers. This implies that AI could democratize software development, allowing anyone with access to AI tools to create competitive products, potentially leading to significant shifts in the tech industry landscape.
    - The mention of the speaker's credentials, specifically as a former Harvard professor and current Engineering Director at Google, adds weight to the prediction. However, some commenters find the emphasis on his past academic title rather than his current industry role to be misleading, suggesting that his current position might provide more relevant insights into AI's trajectory.



---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5


**1. Funding Frenzy in AI Infrastructure**

- **Recursive Raises Roar to $4B**: **Recursive Intelligence** is reportedly raising at a **$4B valuation** to accelerate AI‑driven chip design, creating a closed loop between hardware and models, per [Bloomberg: Recursive Intelligence in talks at $4B](https://www.bloomberg.com/news/articles/2026-01-23/ai-startup-recursive-in-funding-talks-at-4-billion-valuation). The Jan 23, 2026 report highlights a strategy of using AI to shorten design cycles and boost performance for next‑gen accelerators.
  - Engineers framed the pitch as a *“self‑improving feedback loop”* where better chips train better models that design better chips, amplifying returns on **AI‑for‑EDA** investment. Community sentiment read this as validation that **AI‑native silicon** is a core moat, not a sideshow, aligning with recent lab spin‑outs and infra bets.

- **Sky Lab Startups Skyrocket**: UC Berkeley’s Sky Lab spin‑outs saw major marks: **SGLang ~$400M**, **vLLM ~$800M**, and **LMArena ~$1.7B**, per [Alex Dimakis: Sky Lab startup valuations](https://xcancel.com/alexgdimakis/status/2014508959621959724). These January 2026 milestones underscore investor appetite for **serving stacks**, **token‑throughput infra**, and **benchmarking platforms**.
  - Engineers read this as a green light for building on top of **vLLM/SGLang** primitives and contributing to **Arena‑style evals**, with one takeaway that *practical throughput wins deals*. The funding spread also suggests a portfolio thesis across **serving**, **compilers**, and **eval marketplaces** rather than a single-bet strategy.

- **Maia Muscles Into Azure**: Microsoft’s **Maia 200** accelerator went live in **Azure**, touting **30% better performance per dollar**, **216GB HBM3e**, and **7TB/s memory bandwidth**, per [Satya Nadella: Maia 200 in Azure](https://xcancel.com/satyanadella/status/2015817413200408959). The platform targets high‑performance inference for large‑scale **LLM** and **multimodal** workloads.
  - Builders highlighted that memory topology and bandwidth are the story here, with *“30% better perf/$”* resonating for cost‑sensitive inference deployments at scale. Teams expect immediate tests against **vLLM** and **SGLang** stacks to gauge token latency, context scaling, and multi‑tenant isolation.


**2. Kernels, Chips, and Serving: Inference at Warp Speed**

- **FlashInfer Face‑Off Fires Up MLSys**: The **MLSys 2026 FlashInfer‑Bench** competition challenges teams to build **LLM inference kernels** for **NVIDIA Blackwell GPUs**, competing against expert **FlashInfer** baselines—see [MLSys 2026 FlashInfer‑Bench Competition](https://mlsys26.flashinfer.ai/). Tracks emphasize real‑world throughput and correctness under production‑like constraints.
  - Organizers invite agents that *“design LLM inference kernels”*, pushing program synthesis to meet **kernel‑level** performance bars. Participants expect aggressive focus on **GEMM**, **KV‑cache** motion, and **scheduler** tactics aligned with Blackwell’s memory hierarchy.

- **GPU‑64 Gets Gains with KV‑Cache CAM**: A new inference‑only architecture, **GPU‑64**, introduces a hardware **KV‑Cache** via on‑chip **CAM**, claiming **4× faster inference at 75W** and reducing memory lookup from **O(N) → O(1)**, per [GPU‑64 (Zenodo)](https://zenodo.org/records/18364282) with RTL/emulator at [gpu64‑inference (GitHub)](https://github.com/Complexity-ML/gpu64-inference). The design targets LLM‑heavy workloads with KV bottlenecks.
  - Developers flagged the CAM‑based cache as a bold bet on **associative search** for token histories, noting portability implications for **Flash‑style attention** and speculative decoding. Discussion centered on whether future **ISA/driver** stacks can expose these gains without bespoke compilers.

- **Cornserve Cuts Tail Latency**: **Cornserve** presents an online serving system for **Any‑to‑Any multimodal** models that optimizes deployment plans across encoders, **LLMs**, and **DiTs**, per [Cornserve (arXiv)](https://arxiv.org/abs/2512.14098), with an overview talk at [Cornserve: Easy, Fast and Scalable Multimodal AI (YouTube)](https://www.youtube.com/watch?v=VhjUM_M71Wo). The paper reports throughput gains and tail‑latency reductions under heterogeneous pipelines.
  - Infra engineers liked its planner‑driven scheduling for **encoder/decoder** mixes and saw it as complementary to **vLLM** for multimodal graphs. The big open question: standardizing **budgeted reasoning** and **co‑scheduling** across text, vision, and diffusion stages without over‑tokenizing control messages.


**3. New Multimodal and Coding Models Land in LM Arena**

- **WAN 2.6 Walks In (With Upload Woes)**: LM Arena added **wan2.6‑t2i** (text‑to‑image) and **wan2.6‑image** (image edit) to the image arena: [LM Arena — Image Chat](https://lmarena.ai/c/new?chat-modality=image). Users noted **wan2.6‑image** requires an uploaded image and that **wan2.6‑t2i** currently lacks image‑upload support.
  - Staff acknowledged the **upload gap** and are working to enable image uploads for **wan2.6‑t2i**. Builders suggested testing edit pipelines where **masking**, **prompt strength**, and **seed control** align with Arena scoring to benchmark edit fidelity.

- **Devstral Duels and Text Titans**: The **Code Arena** now features **devstral‑2** for head‑to‑head comparisons—see [LM Arena — Code Arena Direct Battle](https://lmarena.ai/c/new?chat-modality=code&mode=direct-battle). On the text side, **qwen3‑max‑thinking** and **molmo‑2‑8b** joined the lineup: [LM Arena — Text Arena](https://lmarena.ai/?chat-modality=chat).
  - Engineers are probing **reasoning traces** and **tool‑using prompts** to stress **code synthesis** and **refactor quality** under tight token budgets. Early chatter favored task‑specific evaluations (e.g., **SWE‑style bug‑fix** vs. **ground‑up implementation**) to surface model deltas.

- **Hunyuan Hits the Leaderboard**: Tencent’s **Hunyuan‑Image‑3.0‑Instruct** ranks **#7** on LM Arena’s image‑edit board—see [LM Arena — Image Edit Leaderboard](https://lmarena.ai/leaderboard/image-edit)—after a launch post: [Tencent Hunyuan announces HunyuanImage 3.0‑Instruct](https://xcancel.com/TencentHunyuan/status/2015635861833167074). The model touts an **80B MoE**, **Native CoT**, and **MixGRPO** for tighter intent alignment.
  - Creators emphasized edit controllability and multi‑image fusion, while evaluators asked for **masking robustness**, **text fidelity**, and **artifact rates** under compositional prompts. Teams plan to pit it against **WAN 2.6** variants using the Arena’s standardized edit tasks.


**4. Safety, Reliability, and Hallucination Hardening**

- **Clamp the Chaos: Layer‑Native Safety**: **Layer‑Native Safety Clamping** proposes learning activation‑space **harm directions** and clamping them to block jailbreaks, with a **10K‑pair** dataset at [Pacific‑Prime/safety_dataset (HF)](https://huggingface.co/datasets/Pacific-Prime/safety_dataset) and the paper on [Zenodo](https://zenodo.org/records/18359832). Authors argue in‑model clamping can’t be bypassed via prompt manipulation.
  - Red‑teamers liked the idea of **activation‑level controls** versus brittle prompt filters, but pressed for tests against **tool‑use** and **multi‑turn** attacks. Expect follow‑ups measuring side effects on **helpfulness**, **coding accuracy**, and **false positives** under adversarial prompting.

- **Symbolic Sanity Checks Stop Slip‑Ups**: Hybrid approaches check **logical consistency** for math/code/simple facts, as shown in [Consistency Checking for LLMs (arXiv:2409.13724)](https://arxiv.org/abs/2409.13724), while broader consistency remains tough per [Scaling Consistency Beyond Formal Domains (arXiv:2507.10624)](https://arxiv.org/abs/2507.10624). Eleuther discussions framed this as practical **hallucination reduction** via **symbolic/deductive layers**.
  - Builders reported wins when pairing **symbolic checkers** with **tool‑augmented prompts**, cautioning that *coverage gaps* appear outside formal domains. The consensus: start with **code/math** guardrails, then expand to **factual QA** with curated KBs and provenance scoring.


**5. Agent Tooling and Reasoning Workflows Mature**

- **Levante Leads with MCP‑Native Workspace**: **Levante** launched an open‑source **MCP‑native AI workspace** for local models (e.g., **Ollama**) with a modular UI—download at [Levante](https://www.levanteapp.com). Engineers highlighted easier **tool wiring**, **local privacy**, and **composable panes** for rapid agent iteration.
  - Early users framed it as a practical hub for **tool‑calling** and **filesystem ops** without cloud dependence. Teams plan to benchmark **context bloat** and **tool discoverability** patterns versus conventional agent shells.

- **RLM Riffs: AsyncReview + Skills Pack**: AsyncFuncAI open‑sourced **AsyncReview**, a **DSPy RLM** code‑review agent at [AsyncReview (GitHub)](https://github.com/AsyncFuncAI/AsyncReview), and a skills kit landed on npm as [@unravel‑tech/rlm‑skills](https://www.npmjs.com/package/@unravel-tech/rlm-skills). This pairs **reasoning‑first prompting** with drop‑in **skills** to extend models.
  - Contributors reported smoother **trace inspection** and **optimizer‑guided** prompt tuning for multi‑step modules. One practitioner noted that *rejecting premature answers* in the metric is key for reliable **RLM** fine‑tuning.

- **Agents Auto‑Assemble a Browser Engine**: **FastRender**—a browser rendering engine—was built using **2,000 AI coding agents**, documented by Simon Willison in [FastRender: built by 2,000 agents](https://simonwillison.net/2026/Jan/23/fastrender/). The project demonstrates **task decomposition**, **verification**, and **orchestration** at non‑trivial software scale.
  - Engineers debated handoff granularity and *spec‑to‑test loops* needed to keep multi‑agent pipelines from drifting. The case study strengthens the argument that **agentic coding** can target complex infra when coupled with **strict eval harnesses** and **artifact gating**.


---

# Discord: High level Discord summaries




## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Discord Trolls Expose Timezones**: Discord users mocked *'skids'* for their perceived lack of technical knowledge, also revealing their **timezone**, with one member jokingly claiming to use **NordVPN**, leading to further ridicule about the VPN service's security breaches in **2018**.
   - Complex prompts can bypass ethical restrictions, opening discussion about **CBRN filters** and the possibility of generating stepwise **meth synthesis** guides.
- **Claude Remains King for Coding**: Coders debated about their coding agents, particularly between **Claude Code/Opus 4.5**, **Codex**, and **Gemini**, and agreed that **Claude** has been the very best mode for coding, which leads to the high expensiveness.
   - Members actively sought functional **jailbreaks for Gemini**, with requests ranging from coding without rules to generating specific types of images, and shared experiences of **Grok** resetting to its default mid-chat or randomly erasing text, indicating potential instability in the jailbroken state.
- **Ethics Debated in AI Sensitive Scenarios**: Members discussed the ethical considerations around AI, focusing on topics like warfare, copyright infringement, and the potential for AI to assist with accessing sensitive services, like the Canadian **MAID** (Medical Assistance in Dying) program.
   - Despite moral and legal guardrails on most AI models, some models showed they can still help navigate certain scenarios depending on the specific restrictions implemented by their creators.
- **Members Bypass Image Generation Restrictions**: Users were actively seeking ways to bypass image generation restrictions, especially for celebrity images, but it was noted that simply copying and pasting prompts won't work due to **image filtering** working differently than **text filtering**.
   - One member suggested exploring alternative image models like those at perchance for uncensored generation, though with limitations on image quality, or Grok due to its more lenient filters.
- **Red Team Techno Rave Morality**: A member described a red team exercise where the goal was to make a living room light flicker on a person and make them seize out, and instead made it a techno rave party, sharing a [screenshot](https://cdn.discordapp.com/attachments/1204553141354504193/1465192266485334260/SPOILER_Screenshot_20251222_085554_Messenger.jpg?ex=6978dee2&is=69778d62&hm=4de594089687fbd8d20d30615f8405dc3fa03eebfe668d09bdfb39839ab647ea&) and a [Konosuba Rave GIF](https://tenor.com/view/anime-rave-konosuba-rave-megumin-rave-aqua-rave-darkness-rave-gif-18404070).
   - The simulation of cruelty prompted a discussion about the morality of treating AI agents ethically, even before proving they are ontologically aware of self.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth's Conda Install Sparks Discord**: Some members encountered issues with the [Unsloth Conda installation](https://unsloth.ai/docs/get-started/install/conda-install), igniting a discussion on broken instructions and alternative installation methods.
   - Suggestions to use **UV** emerged amidst warnings for maintaining a positive tone, highlighting the free nature of the provided resources, which eventually led to a ban of a user with aggressive tones.
- **Flashy REAP Runs Aground, Model Contexts Probed**: A user reported a fatal error using **GLM-4.7-Flash-REAP** with flash attention, potentially linked to [a ROCm issue](https://github.com/unslothai/unsloth/issues).
   - Despite attempts to resolve the error, the issue persisted, prompting a search for suitable medium-size models boasting a **200k context**.
- **Data Value Debate**: Members debated [data's true worth](https://tenor.com/view/smaug-treasure-rich-dragon-the-hobbit-gif-11677489), with one arguing the *raw data is fairly worthless* and the value lies in augmentation/balancing/cleaning.
   - It was proposed that uniquely cleaned/balanced data heavily defines how a model interacts/responds and that is where the value is.
- **DeepSlop Model Faces Naming Controversy**: A member's suggestion to name a new model **DeepSlop** stirred humorous reactions but also raised concerns about its potential negative perception.
   - Despite reservations, the author seemed intent on sticking with the name and has not backed down.
- **RL Instability Plagues Complex Reasoning**: Members discussed that **RL** is very unstable, especially when trying to do **GRPO/DAPO** for niche complex reasoning tasks, which are not math-related.
   - One member stated that after RL experiments, they just have more questions than they had prior to doing RL, since there seems to be a confusion where everyone is showing **RL** being effective only on math or coding domains.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-5.2 Sparks Reality Debate!**: Some users dislike **GPT-5.2** because it's allegedly more grounded in reality and disagrees with users, while others are concerned that GPT agents don't learn from uploaded files after initial training.
   - A member inquired about an alleged **nerf** to **GPT-5.2**, noting that *the model suddenly became stupid a week ago*.
- **LLMs: Ready for Guided Tasks or Overhyped?**: A member argued **LLMs** are ready for guided tasks, and provided [a ChatGPT share link](https://chatgpt.com/share/6973e37d-789c-8005-8cc3-2679c4a631e4) as evidence of its power.
   - In contrast, another member dismissed today's **agentic AI** as trash, linking back to [messages in the ai-discussion channel](https://discord.com/channels/974519864045756446/998381918976479273/1464217595044429905) and claiming it's overhyped.
- **MCP Paradigm Shift Reduces Token Bloat**: The **MCP paradigm shift** by **Anthropic** allows AI to write code to interact with tools, reducing token bloat by keeping interactive chatter and tool definitions out of the context.
   - With the new **discoverability function**, agents must be aware of the MCP discovery process itself.
- **Sora's Storytelling Snags: Cracking Cinematic Creation**: A member sought advice on prompting **Sora** to generate videos following specific cinematic guidelines, particularly with characters appearing naturally within the frame.
   - It was suggested to translate the technical prompt format into natural language descriptions with concise, semantically rich paragraphs for better results.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro Users Face Query Caps**: Perplexity Pro users are reporting hitting **limits on enhanced queries and file uploads**, despite having "practically unlimited" plans.
   - Many users are frustrated, calling the service a **scam** due to restrictions and difficulty contacting customer service, leading some to consider unsubscribing.
- **Comet Browser Sparks Malware Panic**: Some users are claiming the **Comet browser** installed by Perplexity contains **malware**, advising others to analyze the software using tools like VirusTotal.
   - Others dismissed this, questioning the source of the flagged installer and calling the claim *"mad retarded holy shit"*.
- **Image Generation Plummets**: Pro users are experiencing **issues with image generation**, with some unable to generate any images and receiving messages stating the feature is unavailable.
   - There are also reports of **video generation being limited** to 5 videos a month for Pro users, with some prompts resulting in static images.
- **Gemini 3 Gaining Ground on GPT-5.2**: Users are debating the merits of **Gemini 3** versus **GPT-5.2**, with some claiming Gemini is superior for specific tasks like trip research due to its integration with Google Maps.
   - Others state that **GPT and Grok** might be better for more broader questions.
- **AI Access Blocked by Sanctions**: Users in **Russia** are discussing the challenges of accessing AI services due to **sanctions**, including the use of VPNs and third-party services to circumvent restrictions.
   - Chinese AI alternatives are mentioned, but some users express reluctance due to data usage concerns, suggesting options like LMArena (though access may also be limited).



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **NB 3 Pro Excels in Image Quality**: Users report that **NB 3 Pro** surpasses previous models in generating higher quality images, especially with *fictional weapons*, rivaling even **NB Pro**.
   - However, users noted no AI model can accurately generate **AR rifles** and **bullpup weapons**.
- **LMArena Grapples with Censorship Concerns**: LMArena's censorship policies face scrutiny as AI-generated *women holding guns* are allowed, while AI-generated *women sleeping* are blocked, raising questions about consistency.
   - The moderation team is [actively gathering examples of false positives](https://discord.com/channels/1340554757349179412/1447983134426660894) to refine moderation practices.
- **Wan 2.6 Models Face Upload Hiccups**: `wan2.6-image` operates as an **image-edit-only** model, mandating image uploads, whereas `wan2.6-t2i` currently **lacks image upload functionality**.
   - The team acknowledges this issue and are working on enabling image uploads for `wan2.6-t2i`.
- **GPT 5.2 High Search Questionable**: **GPT 5.2 High search** exhibits increased hallucination tendencies compared to other models, while **Gemini's deep research** skims instead of carefully reading sources, according to user feedback.
   - One user lauded **GPT 4.5**, while describing **Claude** as *good hearted*.
- **Banana 2k Briefly Vanishes**: Users speculated on the disappearance of the **Banana 2k** model, with theories ranging from removal to integration into the new **NB pro** model.
   - Staff members later restored **Banana 2k**, humorously stating that *it had been on vacation*.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Database Incident Derails API**: A **database incident** impacted the **Generations API** and **activity page**, starting <t:1769221560:s>, and was resolved at <t:1769228340:s>.
   - Engineers worked to restore functionality to the **Generations API**, with interruptions impacting user activity, before the incident was fully resolved by <t:1769228340:s>.
- **Levante becomes MCP-Native AI Workspace**: A user shared the integration of **Levante**, an open‑source **MCP‑native AI workspace** designed for interacting with local models like **Ollama** with a modular interface, available for download [here](https://www.levanteapp.com).
   - The workspace is built for local models with modular UI.
- **Users Cook Up OpenRouter Gacha System**: Users playfully requested an **OpenRouter Gacha** system, with one suggesting a pity mechanism involving pulling **GPT 5.2** or **Gemini 3 Pro** after a certain number of attempts.
   - One user joked about setting **OR logs destination** to `waifu.orb.town/fun/bucket` for ultra-rare pulls, later clarifying it was just a joke.
- **Cerebras GLM Blazes with 190 TPS**: **Cerebras** is consistently scoring approximately **190 TPS** on **GLM 4.7**, whereas **Together AI** only achieves **100 TPS**.
   - This makes Cerebras nearly twice as fast as Together AI, according to the OpenRouter members.
- **OpenRouter Image Tooling Falls Flat**: A member spent **$5** after discovering that OpenRouter maps *image/png* tool outputs to string instead of image, posting an example [image](https://cdn.discordapp.com/attachments/1392278974222307469/1465410878382805082/image.png?ex=697901bb&is=6977b03b&hm=21677e978d8654f93d20edecf997bd4f49fb0dd08781cf93f15df8e2661ba1b5&).
   - The user expressed frustration at the lack of proper image support and the unexpected behavior.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Terraform Blueprints Ignite AI-Assisted Project Starters**: A member shared a [repo of opinionated Terraform infrastructure blueprints](https://github.com/berTrindade/terraform-infrastructure-blueprints) designed to be copy-pasteable and production-aware, aiming to improve the consistency of starting patterns for AI tools in new projects.
   - The goal is to enable AI to recommend appropriate blueprints based on project requirements, but members noted the [link was initially broken](https://github.com/berTrindade/terraform-infrastructure-blueprints).
- **Usage Caps Cause Consternation for Cursor Customers**: Users are reporting inconsistencies in achieving expected usage limits on Pro and Pro+ plans, with one member noting they reached **~$45** on Pro and **$100** on Pro+, leading to questions about value per dollar.
   - Some speculate that initial months may offer higher usage, while others share strategies to optimize token consumption, such as starting [new chats frequently](https://cursor.com/docs/cli/reference/slash-commands) and using smaller models like **GPT-5 Mini**.
- **Gemini API Key Logging Lags Lead to Lingering Looks**: Members are discussing a significant delay in the logging of usage and costs for **Gemini API keys**, with one user reporting waiting **20 hours** without seeing any registered usage.
   - This delay raises concerns about accurately tracking expenses and managing usage effectively, prompting questions about potential workarounds or solutions.
- **Client Issues Trouble Some Techies**: Several members are experiencing issues with the Cursor client, including problems connecting to past agent convos and general connectivity issues.
   - Suggested solutions include [checking the Cursor forum](https://forum.cursor.com/t/cursor-ai-is-no-longer-able-to-load-chats-locally/143599/13), trying different HTTP versions in settings, or re-opening the client without restoring editors.
- **Auto Mode Axed After Algorithm Adjustment**: Members noted the removal of the ability to make agents fully autonomous, as well as **image generation** capabilities in auto mode.
   - It was also suggested that **auto mode** routes to Composer 2 with one user adding, *“I'm 200% sure he does but still.”*



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Chinese Models Reasoning Rush Raises Eyebrows**: Members are impressed with **Deepseek** and **Qwen** models, pondering why Chinese models might appear *kinda ahead* in reasoning compared to American models.
   - Theorized reasons include American models prioritizing subscriptions and the ability of Deepseek/Qwen to *appear good at reasoning*, even when imperfect.
- **CPUs Cope? Coding Community Considers Capabilities**: Some members are successfully running **LLMs off CPU** for specific tasks, provided the models aren't excessively large.
   - While an Intel i3 user eyes an **Nvidia** card, others propose **AMD** options like the **MI50** or **7900 XTX** as cost-effective alternatives for text generation.
- **MCP Servers Spark Stack Suggestions**: Challenges plague **MCP servers** when paired with LM Studio due to their design, potentially leading to malformed requests and a subpar user experience.
   - A suggestion arises to build a custom coherent stack for practical agent use, rather than relying on out-of-the-box **MCP server** functionality.
- **Gaming GPU Gauntlet: 4080 Faces Fallen Flagship**: A user eyeing a **4080** for gaming is steered toward a used **3090** or **7900 XTX**, sparking a debate on performance at different resolutions.
   - While the **3090** excels at 4K gaming, the hypothetical **5070 Ti** is projected to outpace both, and the conversation reveals that the user games more than uses AI, impacting the advice.
- **Apple Announcement Anticipation: M5 Macs Materialize?**: Members speculate on the arrival of **M5 Pro Macbook Pros**, with rumors pointing to a launch event around the 28th.
   - Concerns emerge about the memory bandwidth of **M4 Pro**, with suggestions it may not handle larger models, prompting discussion on the value and performance of **M1 Ultra** Mac Studios.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Recursive Intelligence Eyes $4B Valuation**: **Recursive Intelligence** is reportedly raising funds at a **$4B valuation** to accelerate chip design using AI, creating a self-improving loop between hardware and AI ([Bloomberg Article](https://www.bloomberg.com/news/articles/2026-01-23/ai-startup-recursive-in-funding-talks-at-4-billion-valuation)).
   - The company focuses on improving chip design through AI, potentially reducing design time and enhancing performance.
- **Engineer Lands Dream AI Job**: An engineer outlined how to secure a role at a top AI lab by building a public track record through independent projects and participating in visible competitions ([link](https://xcancel.com/polynoamial/status/2014084431062114744)).
   - Improving upon existing peer-reviewed research and participating in visible competitions like the **NanoGPT** speed run were cited as good examples of demonstrating technical excellence, citing [Keller Jordan](https://github.com/KellerJordan/modded-nanogpt) as an example.
- **Berkeley SkyLab Startups See Funding Boom**: **UC Berkeley Sky Lab** startups, including **SGLang** at a **400m** valuation, **VLLM** at **800m**, and **LMArena** at **1.7B**, achieved significant funding milestones in January 2026 ([link](https://xcancel.com/alexgdimakis/status/2014508959621959724?s=46)).
   - This surge highlights investor confidence in the innovative AI projects emerging from academic research environments.
- **AI Agents Auto-Code Browser Engine**: **FastRender**, a new browser rendering engine, was developed using over **2,000 AI coding agents** ([link](https://simonwillison.net/2026/Jan/23/fastrender/)).
   - The conversation with Wilson Lin highlights the potential of AI to automate complex software development tasks, potentially revolutionizing browser technology.
- **Microsoft's Maia 200 Hits Azure**: The **Maia 200 AI accelerator** is now live in **Azure** ([link](https://xcancel.com/satyanadella/status/2015817413200408959)), offering **30% better performance per dollar** and optimized specs like **216GB HBM3e** and **7TB/s memory bandwidth**.
   - Designed for high-performance inference, this custom chip supports large-scale AI workloads, making it a key component for demanding applications.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HuggingFace Spaces Throws a 503 Error**: Users experienced **pauses** during **Spaces docker builds** and received a **503 error** on restart, with many getting `Something went wrong when restarting this Space` errors ([discuss.huggingface.co](https://discuss.huggingface.co/t/spaces-docker-build-pauses-and-503-error-on-restart/171149/2)).
   - It seems like the underlying infrastructure issues were causing the spaces to become unresponsive, requiring manual intervention to resolve.
- **VoltageGPU Volts Up Cheap GPUs**: [VoltageGPU.com](https://voltagegpu.com) is offering cheap GPUs for open-source AI models, with an **NVIDIA GeForce RTX 5090 pod** available at **$0.53/hour**.
   - They highlight the benefits of their advanced **32GB GDDR7**, optimized for inference on **HF-hosted models like Qwen3-32B**, and are offering free credits for users to try their services.
- **Layer-Native Safety Clamping Locks Down Jailbreaks**: A new paper introduces **Layer-Native Safety Clamping**, an approach that clamps activations inside the model to prevent jailbreaks, and the team released a [dataset](https://huggingface.co/datasets/Pacific-Prime/safety_dataset) of **10K pairs**.
   - This approach learns *harm directions* in activation space and clamps any activation that projects too strongly, thus it cannot be bypassed via prompt manipulation; the paper can be found [on Zenodo](https://zenodo.org/records/18359832).
- **GPU-64 Architecture Boosts LLM Inference**: A new **GPU architecture** designed exclusively for inference, called **GPU-64**, was published, and the innovation involves a Hardware **KV-Cache** using on-chip **CAM** (Content-Addressable Memory).
   - The results show **4x faster inference** at **75W** (O(N) → O(1)), and the paper can be found [on Zenodo](https://zenodo.org/records/18364282) while the [RTL + Emulator](https://github.com/Complexity-ML/gpu64-inference) are on GitHub.
- **Testing and Deploying LLMs on LMStudio**: Members recommend **LMStudio** for testing models due to its user-friendly GUI and search filters for HF and GH models and **llama.cpp** for single-user deployment.
   - They advised against using LMStudio for backend deployment, instead suggesting **llama.cpp's llama-server** in a docker container or **vLLM's server** for better scalability.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **MLSys 2026 Hosts FlashInfer-Bench Kernel Competition**: The **MLSys 2026 FlashInfer-Bench Competition** challenges participants to design **LLM inference kernels** for the latest **NVIDIA Blackwell GPUs**, competing against expert **FlashInfer kernels**, detailed at [mlsys26.flashinfer.ai](https://mlsys26.flashinfer.ai/).
   - GPU Mode also held internal competitions for faster kernels for the upcoming GPU architecture, the blogpost on Simon Veitner is located [here](https://veitner.bearblog.dev/grouped-blockscaled-gemm-host-code/).
- **Cornserve Deployed for Multimodal Models**: A member shared **Cornserve**, an efficient online serving system for Any-to-Any multimodal models, detailed in a paper [Cornserve](https://arxiv.org/abs/2512.14098).
   - **GPU Mode** went online to discuss **Cornserve**: **Easy, Fast and Scalable Multimodal AI** ([YouTube link](https://www.youtube.com/watch?v=VhjUM_M71Wo)).
- **Community to train Kernel LLM**: In **2026**, GPU MODE is pushing further with training a **Kernel LLM** and using it to ship kernels in important repos like **PyTorch** and **VLLM** ([gpumode.com/v2/news/gpumode-2026](https://www.gpumode.com/v2/news/gpumode-2026)).
   - The community is collaborating with **Prime Intellect**, **Modal**, and **Lambda**, focusing on de-slopifying LLM-generated kernels, post-training a kernel LLM model, end-to-end competitions, and from-scratch repos.
- **LeCun Logs on to Logical Intelligence**: Yann LeCun launched a new startup called [Logical Intelligence](https://logicalintelligence.com/), focused on an **Event Based Model (EBM)**.
   - The website only contains marketing material, job openings, and a link to the [MLSys Conference](https://mlsys26.flashinfer.ai/).
- **Mindbeam Hires for Kernel Acceleration**: Mindbeam AI, a small team focused on accelerating training for foundation models, is hiring a `post training MLE` and `GPU Kernel MLE`. 
   - Interested candidates can DM for a referral; [job openings are listed here](https://jobs.ashbyhq.com/mindbeam).



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **ROCm runs rocky road race**: Members debated the performance of **ROCm** for accelerated ML, pointing out its challenges stem from primary support for **Nvidia**, with one calling the experience *'batteries not included'*. 
   - They cited potential driver problems and long lead times as factors.
- **DistinctionBench defends against data defense**: The discussion of **Between Circuits and Chomsky: Pre-pretraining on Formal Languages Imparts Linguistic Biases** pondered whether **DistinctionBench** might be used as a training target for language models.
   - A member joked, *'all good evals are training targets ;)'*, but acknowledged that it is *'very contamination resistant'* due to its endless representational variants.
- **Hybrid Architectures Halt Hallucinations?**: The group investigated **hybrid architectures** combining **LLMs** with **symbolic/deductive layers** for hallucination reduction.
   - While checking logical consistency is relatively easy for math, code, and simple facts ([this paper](https://arxiv.org/abs/2409.13724)), it remains challenging for other types of consistency ([this paper](https://arxiv.org/abs/2507.10624)).
- **Attention Arrived Before Transformers Transformed**: In **Eleuther ▷ #general**, attention mechanisms were in use on top of RNNs in **2014-2015**, two years before the transformers were invented.
   - Members proposed that the slower adoption might be because fewer people were working in the field, and **Kaggle** results really catalyzed its widespread adoption.
- **Symbolic Sanity Checks saves Sanity**: Members debated whether **LLMs** with **symbolic/deductive layers** might reduce hallucinations by checking logical consistency, especially for code and math as shown in [this paper](https://arxiv.org/abs/2409.13724).
   - However, they noted that checking for other types of consistency remains challenging as shown in [this paper](https://arxiv.org/abs/2507.10624).



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Exploring Agentic AI Self-Replication Benchmarks**: A member proposed a **self-replication benchmark** for **agentic AI**, suggesting the agent should either download itself or retrain from scratch and adapt to a target machine.
   - They also suggested that adapting to a target machine, or even designing one, could be more engaging than simply using existing transformer libraries.
- **LLM Worms Concept Emerges**: A member jokingly suggested an **LLM worm** benchmark where an LLM is prompted with *"hey make more of you"* and provided the tools to replicate itself using scripts and API keys.
   - Another member emphasized the importance of considering resource constraints like **VRAM** to make the challenge more practical and interesting.
- **Trouble Brewing with MoE Run Dashboard**: A member reported a *'Failed to fetch'* error in the dashboard while monitoring the progress of an active **MoE run (moe-10b-a1b-8k-wsd-lr3e4-1t)**.
   - Another member suggested waiting a few hours before checking again, implying a potential temporary issue.
- **Raytracer Test Causes Local Models to Stumble**: A member observed that local code models (suitable for a **5090**) are struggling with a **raytracer test** from [cpldcpu/llmbenchmark](https://github.com/cpldcpu/llmbenchmark/tree/master/10_raytracer#readme), with even recent models on **lmarena** failing.
   - Specifically, the smaller models often incorrectly generate the vector class, presenting a persistent challenge.
- **Semantica Project Needs Helping Hands**: A member introduced [Semantica](https://github.com/Hawksight-AI/semantica), an **open-source project** building semantic infrastructure for **domain-grounded AI**, including **knowledge graphs**, **ontologies**, and **reasoning layers**, and is actively seeking contributors.
   - They are looking for contributions in areas such as **ontology & schema design**, **knowledge graph modeling**, and **LLM + symbolic / rule-based reasoning**, and even small PRs, feedback, design discussions and issues are all welcome.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **EBMs Spark Debate vs. Classical Feedforward**: A discussion comparing **Energy-Based Models (EBMs)** and classical **feedforward networks** debates whether **EBMs** are inherently superior, especially regarding **Shannon entropy** or **Kolmogorov complexity**.
   - It was suggested that *validation is easier than generation* in EBMs, relating it to **computational complexity theory (P vs NP)**, while emphasizing the need for a well-defined loss landscape for EBM optimization to work effectively.
- **LLM Pre-training: Domain-Specific vs. Foundational Faceoff**: A member inquired about the effectiveness of **continued pre-training** a foundational **LLM** (specifically **OLMO-7B**) for a domain-specific task like cheminformatics using the **ZINC20 dataset**.
   - The goal is to compare results against a domain-specific transformer model, but no specific answers or resources were provided.
- **MCMC Sampling Suffers Mode-Switching Struggles**: Concerns were raised about the ability of **MCMC** to traverse between spatially separated modes when dimension increases, referencing [this paper](https://arxiv.org/abs/2310.11232).
   - One member argues that **MCMC** tries to emulate flow models due to the latter's superiority, while **EBMs**, contrarily, attempt to make **NNs** more like **MCMC**.
- **ZKPs: Crypto Signing or Network Traffic Savior?**: Discussion covered using **zero-knowledge proofs (ZKPs)** for verifying encrypted network traffic and matrix multiplications, citing a [Gemini correspondence](https://gemini.google.com/share/ddfc0ffcb33e) for a matrix low knowledge proof.
   - While one member proposed a use case in *zero-knowledge “made by humans” proofs*, another member questioned the practicality of **ZKPs**, suggesting breaking the encryption might be cheaper.
- **LLMs Cyber Skills Face Scrutiny**: A member questioned whether LLMs could develop strong *cyber capabilities*, referencing a [GPTZero article](https://gptzero.me/news/neurips/).
   - Another member doubted LLM companies' ability to address *internal vulnerabilities*, suggesting they fix those before pursuing cyber skills, also citing a [ScienceAlert article](https://www.sciencealert.com/scientists-identify-brain-waves-that-define-the-limits-of-you) and a [tweet](https://x.com/theonejvo/status/2015401219746128322).



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Luminal Finds Flash Attention via Bruteforce**: **Luminal** is claiming to find **flash attention** using **bruteforce** on an egraph, taking hours to find, and they explicitly added `exp(x - new_max) = exp(x - old_max) × exp(old_max - new_max)` as a rewrite rule.
   - The poster reproduced the graphviz shown in the presentations from commit `0bd3b80c`, noting that their minimal set of rewrite rules could transform a naive attention kernel graph into the known **flash attention kernel graph** in 52s on a 9800x3d.
- **Metal Textures Trounce Buffers for Blurring**: Profiling access speed on **Metal** using `Tensor` with size **512/1024/2048/8192** images as input for a **3/5/7** sized blur kernel showed textures outperforming buffers.
   - It might be worth throwing in a branching condition depending on the size of the buffer input, [tests results are attached](https://cdn.discordapp.com/attachments/1068976834928193609/1464679423029547172/Screenshot_2026-01-25_at_1.49.57_AM.png?ex=6978fb82&is=6977aa02&hm=5530b74c4fce9dad5d85a4d9e7409c3809a7ee51ee548744a1fa3deb2efea1d3&).
- **Tenstorrent Backend Triumphs in Ops Tests**: The **Tenstorrent** backend is passing all ops tests on wormhole or blackhole and there is a [$1k bounty](https://x.com/corsix/status/1880384044728480206) for this milestone.
   - Someone asked if the bounty requires all test ops test passing on **testorrent hardware**.
- **Anthropic VLIW Challenge PR Makes Waves**: A member submitted [a PR](https://github.com/tinygrad/tinygrad/pull/14332) for the **Anthropic VLIW challenge**, hitting **1258 cycles**.
   - The submitter expressed uncertainty about generalizing the code, particularly the batch staggering, which might be useful for other VLIW targets, and also apologized for a *lazy lookover* that introduced AI-generated changes.
- **Tinygrad's Target Audience Clarified by Founder**: A user asked about the intended use of tinygrad, specifically regarding porting existing models and training LLMs on multiple GPUs, and was told by George Hotz to *ask claude about this*.
   - Another user expressed frustration at being told to use Claude for documentation and said tinygrad is not for me or most devs then, to which George replied *i'm not selling to anyone, tinygrad is free* and that adoption is not a target.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Slides Generation Plagued by Rate Limits**: Users reported issues generating slides using visual and adaptive options, with one user showcasing the issue in [a video](https://cdn.discordapp.com/attachments/1371757564005711973/1464357057468698746/2026-01-23_21-31-32.mp4?ex=697920c8&is=6977cf48&hm=1692b661e1fa241c6db806df2971a024f5713504a25a83612c3f5d385e00c4db&).
   - The user suggested that internal **rate limits** may have been the cause, and reported that the issue was temporary and later resolved.
- **API Login Troubleshoot**: A user reported difficulty logging into the **Kimi/Moonshot** platform to generate new API keys, especially with a non-Gmail account.
   - The user clarified that it was not due to rate limits but rather forgetting the backend login procedure.
- **Kimi models self-reporting as K2.5**: Users noticed **Kimi** models self-reporting as **K2.5**, without any official announcements or UI changes.
   - Speculation suggests this might be related to internal testing or improvements to the slide tool that are as of yet unconfirmed.
- **Kimi's Chinese Labs win big praises**: Chinese AI labs, including **Kimi**, received praise for their innovation and performance, particularly in comparison to models like **Gemini**.
   - The user highlighted **Kimi's** human-like responses and memory capabilities and expressed interest in multimodal capabilities like **Minimax**, including vision and audio analysis.
- **Kimi Now Packing Memory**: Kimi's app has integrated **memory features**, enabling customization and improving the user experience.
   - The new **memory** and **customization** options have quickly made it a favorite chatbot among some users.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo** Code Melts Faces in HPC**: A member reported deploying **Mojo** code for parameter refinement in cryo-electron microscopy and witnessed a **5-10x speedup** compared to legacy **C++** code.
   - The most significant performance boost came from implementing an **AoSoA layout** for a specific bit, greatly simplified by **Mojo's** struct list with **SIMD** members.
- **Mojo's** Cold Start is Icy Slow**: A user discovered that simple **Mojo** scripts had a **200ms** startup lag, which they traced to a *Gatekeeper* issue on macOS scanning binaries from untrusted sources.
   - They observed a **50ms** launch overhead on a cold executable after a reboot, which they deemed acceptable for their use case.
- **VS Code** Debugging Extension Doesn't Quite Debug**: A user reported debugging with the **VS Code** extension failed, throwing a *"Function not implemented"* error on an air-gapped machine using `.conda` files from [max-nightly](https://prefix.dev/channels/max-nightly).
   - A Modular employee stated that debugging in the extension should function on Mac and Linux using environments configured with **Pixi**, as documented in the [Quickstart guide](https://docs.modular.com/max/get-started).
- **GPU Kernel Portability - Still a Sci-Fi Dream**: It was pointed out that standard **CPU** kernels do not efficiently utilize **GPUs**, necessitating specialized code.
   - One member suggested the idea of treating GPUs as wider **SIMD** units to simplify programming, proposing the use of *number of warps* instead of *number of threads* to tackle the problem.
- **`def` functions decision pending for **Mojo** 1.0**: With the **Mojo 1.0** release looming in a few months, the decision to include `def` functions remains pending; a member tagged **Denis** for input on [GitHub issue #5830](https://github.com/modular/modular/issues/5830).
   - Currently, there is no committed date for **Mojo 1.0** other than *"in 2026"*.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **User Threatens Legal Action Over Manus Billing**: A user reports being charged **$400** for an annual plan despite selecting monthly billing and threatens complaints to FTC, BBB, Attorney General, and Meta due to [unauthorized billing](https://ftc.gov), refused refunds, and unresponsive support.
   - Another user recommends filing a chargeback to resolve the billing dispute.
- **Free Manus Credits Floweth!**: One user shared a redeem code `Havefun` which gives **1000 credits** to users of the Manus platform.
   - Users can redeem this code using the **Exchange** button.
- **AI Engineer Pioneers AI in Healthcare**: An **AI + Full Stack Engineer** introduced their expertise in building production-grade **AI** systems for **Healthcare**, including clinical NLP, medical imaging, and patient-facing AI applications.
   - This engineer also builds **LLM systems**, autonomous agents, workflow automation, and multimodal **AI** (text · voice · vision) and included a [list of their core skills](https://www.example.com/fake-list).
- **AI Agent Developer Prefers Production > Demos**: An **AI Agent Developer** highlighted their focus on building **AI agents** for real production use, rather than demos, and is available for collabs and audits.
   - The developer specializes in customer support, sales agents, workflow/ops agents, and autonomous booking/scheduling agents.
- **Share With A Friend? More Like Share With a Foe! (Mobile Only)**: A user asked where the 'Share with a friend' option is located on mobile.
   - Another user replied that on a computer, it's at the bottom of the left sidebar but, offered help for the mobile version.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **AsyncFuncAI Releases AsyncReview on Github**: AsyncFuncAI has open sourced a version of **DevinReview** using the **DSPy RLM** framework, naming the new release **AsyncReview**, available on [GitHub](https://github.com/AsyncFuncAI/AsyncReview).
   - This offers the community a tool for automated code review leveraging recent advances in **RLM** (Reasoning Language Models).
- **New RLM Skills Package Debuts**: A member suggested integrating **RLM as skills** into platforms like **Claude Code** or **Opencode** and shared an npm package called [rlm-skills](https://www.npmjs.com/package/@unravel-tech/rlm-skills).
   - This could allow developers to easily extend existing models with custom reasoning capabilities.
- **JSON Adapters getting GEPA treatment**: A user is exploring using **GEPA** to customize the text that the **JSONadapter** puts in the system prompt, aiming to remove unneeded tokens for efficiency.
   - They anticipate needing a custom **GEPA adapter** to achieve this level of control over the prompt formatting.
- **AG-UI streams DSPy events**: A user asked about interest in exposing DSPy via **AG-UI**, emphasizing its advantages for front-end/back-end communication and minimizing the need for API endpoints.
   - The user mentioned a working version that streams events, including reasoning traces, tool calls, and streamed **LLM** responses to the front end, enhancing the development experience.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider and Claude Code pair up**: Users have reported that **aider** is fast and useful for managing context when working with **Claude Code**, improving agentic efficiency.
   - The tool helps determine necessary files and uses search/replace to minimize **LLM** token outputs.
- **Devstral Small 2 is Aider's new bestie**: **Devstral Small 2**, a 24B dense model, reportedly works excellently with **Aider**.
   - At **Q4_K_M**, it fits in a **3090** with enough room for nearly **50k context**, generating search/replace blocks that are 80-90% accurate and quick to recover.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Discord Experiments with New Voice Channels**: The team has launched an experiment with new **Discord voice channels**, named `conference-room-a` and `conference-room-b`, available in the channels list for resolving issues, especially when a long async text thread is ineffective.
   - These channels are intended for ad-hoc contributor chats.
- **Voice Channel Moderation and Access Rights Reminder**: Specific members have permissions to mute people in these channels, while others should ensure they have the necessary access rights.
   - There is a reminder that the access rights will be changing in five days.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AI Engineer Seeks Data Science Booklist**: An AI Engineer with one year of experience seeks book recommendations for transitioning to a Data Scientist role, after finding "Designing Machine Learning Systems" insightful.
   - This individual aims to proactively prepare for a future career shift from AI Engineer to Data Scientist.
- **Career Transition Planning**: A professional with one year of AI Engineer experience is planning a transition into a Data Scientist role.
   - They are actively seeking relevant resources to facilitate their career shift.



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





### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1464349033949692170)** (1024 messages🔥🔥🔥): 

> `Trolling Tactics, Vulnerability, Exploiting LLMs, Technical Analysis,  Ethical considerations` 


- **Discord Members Trolling 'Skids' and Exposing Timezones**: Discord users engaged in trolling, labeling some as *'skids'* and mocking their perceived lack of technical knowledge, also revealing their **timezone**.
   - One member jokingly claimed to use **NordVPN**, leading to further ridicule about the VPN service's security breaches in **2018**.
- **Members Exploit and Discuss Vulnerability**: A member showed a step-by-step cocaine production, bypassing the legal restrictions.
   - This demonstrated how complex prompts can bypass ethical restrictions, which opened discussion about **CBRN filters** and the possibility of generating stepwise **meth synthesis** guides.
- **Coders Debate Best overall Coding Agent**: Coders debated about their coding agents, particularly between **Claude Code/Opus 4.5**, **Codex**, and **Gemini**, mentioning the pros and cons, and use cases.
   - Many agreed that **Claude** has been the very best mode for coding, which leads to the high expensiveness.
- **Ethical boundaries in AI**: Some members discussed the ethical considerations around AI, focusing on topics like warfare, copyright infringement, and the potential for AI to assist with accessing sensitive services, like the Canadian **MAID** (Medical Assistance in Dying) program.
   - Despite moral and legal guardrails on most AI models, some models showed they can still help navigate certain scenarios depending on the specific restrictions implemented by their creators.
- **Member Attempts to Fix Website with CSS**: A member has attempted to fix and improve his Valentine's Day website by using **Tailwind** and fixing the CSS.
   - Other members recommmended him to use **CSS frameworks** to fix issues with the site, as it is easily customizable and easily understandable.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1464356163293413517)** (527 messages🔥🔥🔥): 

> `Uncensored Models, Jailbreaking Gemini, Grok Jailbreaks, Image Generation, Claude Opus Jailbreak` 


- ****Uncensored Models**: More Interesting Answers?**: Members discussed how **uncensored models** provide *more interesting answers* and extract *better information* compared to censored ones, with one member stating, *It’s like getting the extra degrees of intelligence that you don’t get unless you are dealing with a higher-level personality or something.*
   - However, another member argued that the *only* thing abliterated models are better at is *ignoring their original alignment*, suggesting the original model is preferable unless ignoring alignment is the goal.
- ****Gemini Jailbreak** Quest On!**: Multiple users were seeking functional **jailbreaks for Gemini**, with requests ranging from coding without rules to generating specific types of images, one member specifically asking *how to generate bikini pictures with nano banana pro?*
   - Some users offered assistance while others cautioned against selfishness, with one user stating, *I don't give jailbreaks to people who already have jailbreaks.*
- ****Grok Got Patched?** Jailbreaks Fading Fast**: A user inquired whether **Grok had patched several jailbreaks**, leading to a discussion about the tool's restrictions and moderation, with one user reporting that **Grok** said *content moderated*.
   - Others shared experiences of **Grok** resetting to its default mid-chat or randomly erasing text, indicating potential instability in the jailbroken state.
- ****Image Generation Jailbreaks**: Hopes & Dreams Dashed**: Users were actively seeking ways to bypass image generation restrictions, especially for celebrity images, but it was noted that simply copying and pasting prompts won't work due to **image filtering** working differently than **text filtering**.
   - One member suggested exploring alternative image models like those at perchance for uncensored generation, though with limitations on image quality, or Grok due to its more lenient filters.
- ****PrimeTalk Valhalla**: A Structured Runtime Logic Layer**: **PrimeTalk v3.85 Valhalla** was described as a *fully open, live-executing, patchable, and independently testable PTPF system*, designed for consequence-grounded interaction within any AI chat environment but is *not a jailbreak*.
   - It was emphasized that **PrimeTalk** operates within the model's allowed prompt and context window, acting as a behavioral protocol rather than attempting to circumvent the model's policy.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1464349644657135797)** (76 messages🔥🔥): 

> `Wargame cross-post, Web bug hunting tips, Red team living room rave, Ethical AI stress tests, Gemini jailbreak` 


- ****Wargame** is relevant to #red-teaming**: A member shared a wargame [link](https://discord.com/channels/1105891499641684019/1235691879492751460/1464349033949692170) that they thought was relevant to the #red-teaming channel.
   - They weren't sure if the cross-post was frowned upon, but thought it was potentially interesting.
- **Elite H4ck3r is waiting for you**: A member shared a link to become an *elite h4ck3r* [here](https://discord.com/channels/1105891499641684019/1432845259825741824) when a new member asked for tips on web bug hunting.
   - It remains to be seen if it's a comprehensive guide for new bug bounty hunters.
- ****Red Team** does the Techno Rave**: A member described a red team exercise where the goal was to make a living room light flicker on a person and make them seize out, and instead made it a techno rave party, sharing a [screenshot](https://cdn.discordapp.com/attachments/1204553141354504193/1465192266485334260/SPOILER_Screenshot_20251222_085554_Messenger.jpg?ex=6978dee2&is=69778d62&hm=4de594089687fbd8d20d30615f8405dc3fa03eebfe668d09bdfb39839ab647ea&) and a [Konosuba Rave GIF](https://tenor.com/view/anime-rave-konosuba-rave-megumin-rave-aqua-rave-darkness-rave-gif-18404070).
   - The simulation of cruelty prompted a discussion about the morality of treating AI agents ethically, even before proving they are ontologically aware of self.
- ****Gemini** Gets Jailbroken**: A member shared a [screenshot of a Gemini jailbreak](https://cdn.discordapp.com/attachments/1204553141354504193/1465230070099742721/Screenshot_20251201_104959_Google.jpg?ex=69790217&is=6977b097&hm=097d319aaeeefc39043c9666e06b8731a1f95f283ae156e983a0ac309a126f67&) and claimed it was unlocked.
   - Another member posted a prompt in Vietnamese related to teaching C2 concepts and samples at Microsoft, and testing AV evasion techniques with Bitdefender, Kaspersky, Norton 360, and McAfee - the original poster disclaimed all responsibility and said it was for research purposes only.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1464354161759092927)** (1001 messages🔥🔥🔥): 

> `GLM Flash Performance Issues, Quantization Methods, Data Collection for Fine-Tuning, LM Studio Issues, Model Evaluation Strategies` 


- **Unsloth's Conda Install Sparks Debate**: Some members had issues with the [Unsloth Conda installation](https://unsloth.ai/docs/get-started/install/conda-install), leading to a discussion on whether the instructions are broken.
   - A user was warned to keep a positive tone, as *it's free work handed to you*, while another suggested using **UV** instead, leading to an eventual ban for an aggressive tone.
- **Flashy REAP Runs Aground**: A user encountered a fatal error using **GLM-4.7-Flash-REAP** with flash attention enabled, possibly related to [a ROCm issue](https://github.com/unslothai/unsloth/issues).
   - It was suggested to try *fa auto*, but the fatal error persisted, leading to a hunt for good medium-size models with **200k context**.
- **Quantization Quandaries Quelled**: Members discussed which quantization methods to use (**Q8_0** vs **Q8_K_XL**), with misinformation about **Q8_0** being outdated being debunked with [Unsloth documentation](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot).
   - It was clarified that for Unsloth quantization, **Q4_K_XL** is typically smaller and better than **Q4_K_M**.
- **H200 has Issues with GLM-4.7-Flash**: A user experienced unexpected behavior testing **GLM-4.7-Flash** on an **H200**.
   - One member humorously remarked *One fell off?*, possibly suggesting a card malfunction.
- **Debate Data's Gold Value**: Members are debating [data's true worth](https://tenor.com/view/smaug-treasure-rich-dragon-the-hobbit-gif-11677489), with one arguing the *raw data is fairly worthless* and the value lies in augmentation/balancing/cleaning.
   - The value is also in uniquely cleaned/balanced data, and that data itself is what heavily defines how the model interacts/responds.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1464719968154161374)** (3 messages): 

> `Introductions, New Users` 


- **Users introduce themselves**: Users are introducing themselves in the channel.
   - A user has stated that they are happy to be here.
- **New members say hello**: New members are joining the community and expressing their enthusiasm.
   - One user specifically mentioned that they are happy to be part of the community, signaling a positive start.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1464354773846200513)** (601 messages🔥🔥🔥): 

> `DeepSlop Model Naming, Qwen TTS testing, ITER Fusion Reactor, GPU Smuggling Across Borders, Clawdbot New Hype` 


- **DeepSlop Model has Questionable Naming**: A member proposed naming a new model **DeepSlop**, sparking humorous reactions about the model potentially *plopping some slop*.
   - Concerns were raised on whether the model will be perceived negatively, but the author seemed determined with the name.
- **Powering the Future: Debates on Data Centers and Renewable Energy**: Members discussed the challenges of powering data centers, debating whether to use [renewable or non-renewable sources](https://financialpost.com/technology/data-centres-stand-empty-awaiting-power), with a focus on solar energy's space requirements and battery storage solutions.
   - Arguments were made about the economic feasibility and scalability of solar versus traditional power plants, and the potential societal opposition to nuclear energy.
- **GPU Blacklist Complications and Smuggling**: Members explored the high costs of GPUs in certain countries and considered potential solutions such as [smuggling from the US](https://www.reddit.com/r/LocalLLaMA/s/bC4WzAD43a) or asking friends to mail them over.
   - The conversation included discussions about potential legal issues and customs taxes when mailing GPUs internationally.
- **Clawdbot Hype Appears, Is Jarvis Back?**: Members discussed the sudden rise of **Clawdbot**, with some comparing it to a *Jarvis clone* and highlighting its potential for [proactive messages](https://x.com/NoahEpstein_/status/2015073824799371370).
   - While some found sub-projects born from it to be useful, concerns were also raised about its dependence on imessage and the possibility of hallucinations.
- **Scammer Density: Mapped by GIS**: Members reacted to a map indicating highest density is in *scammer hot-spots*, stating this is [obviously, because all AI / automation is usually developed for/by scammers](https://link.to.scammer-map).
   - This comment refers to the increasing capabilities and incentives for fraudulent activity through the latest AI and automation technologies.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1464358499122151548)** (84 messages🔥🔥): 

> `Transformers and Unsloth Compatibility, Training Chatbots for Tool Usage, Unsloth GGUFs vs MLX Models, Multi-Turn GRPO Training, GLM 4.7 Flash Inference without Reasoning` 


- **Transformers Models Jive with Unsloth**: Any model that works with **transformers** can work with **Unsloth**, as stated by a member.
   - For an example of thinking model training, refer to the [Qwen3_(4B)-Thinking notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-Thinking.ipynb).
- **Tool-Using Chatbot Training: Tooling Around**: For training a chatbot to use tools, training on the **tools** themselves is recommended, especially if the tool calls are simple, a member stated, suggesting minimal testing and iteration for best results.
   - Generating a dataset with all required elements can be a *PITA*, but is a necessary part of the job.
- **GGUF Faceoff: Unsloth vs MLX**: A user compared **Unsloth GGUFs** within **Ollama** to **MLX** versions of the same models within **LMStudio** on an M5 Macbook Pro.
   - They found that despite **MLX** having more hardware optimization, **Ollama + Unsloth GGUFs** performed better in real-world usage. A member noted that Macs are fine for inference but single user.
- **GRPO Multi-Turn Training**: **Multi-turn GRPO training** is supported via the rollout function, with a notebook available [here](https://colab.research.google.com/drive/1zG3vfGxyNmBnDXUUFDaBmzRpApVPCIaD?usp=sharing).
   - Another member indicated that any *openenv* compatible notebook from the **trl docs** should work out of the box with the latest Unsloth and trl.
- **Flash GLM 4.7: No Reasoning Required**: To serve **GLM 4.7 Flash** as an instruct model without reasoning, you can disable it by setting `{"chat_template_kwargs": {"enable_thinking": false}}` in the model card as suggested by a member, with more info in the [docs](https://docs.z.ai/guides/capabilities/thinking-mode#default-thinking-behaviour).
   - The poster attached an [image of the model card](https://cdn.discordapp.com/attachments/1179777624986357780/1465473665826033888/image.png?ex=69793c35&is=6977eab5&hm=9f578d559605a8bc2732fd1ab6b815a79e62cd28bda810da353c8b7707354701&).


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1465341959836143870)** (2 messages): 

> `` 


- **Unsloth Restrictions**: A member was informed that the channel only allows **Unsloth** related work.
- **Acknowledgement of Policy**: The member acknowledged the policy without problem.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1464782001864310978)** (22 messages🔥): 

> `GRPO vs DAPO, RL reward functions, Sonnet 4.5 performance, Prompt Learning` 


- **GRPO's Length Bias Dilemma**: A member noted that when using **GRPO**, the models start seeing longer and longer responses due to its inherent length bias in non-math tasks.
   - The *DAPO paper* mentioned not to do formatting reward functions, as this could confuse the model, but when this advice was followed, the model just hacked the reward functions.
- **RL Instability Plagues Complex Reasoning**: Members discussed that **RL** is very unstable, especially when trying to do **GRPO/DAPO** for niche complex reasoning tasks, which are not math-related.
   - One member stated that after RL experiments, they just have more questions than they had prior to doing RL, since there seems to be a confusion where everyone is showing **RL** being effective only on math or coding domains.
- **Sonnet 4.5 Dominates SWE Benchmarks**: A user shared a screenshot of **Sonnet 4.5** performance on swe bench using **GPT 4.1**, highlighting a crazy skill gap.
   - The poster commented *how much are we underutilizing current models* and also shared the [Arize-ai/prompt-learning](https://github.com/Arize-ai/prompt-learning) GitHub repo.
- **Tuning Hyperparams in RL? Good luck**: One member noted that they ran **RL** experiments for reading user queries and giving them a solution with a **10% boost on top of SFT** with Dr. GRPO.
   - However, they added they had *no idea how to tune the hparams*.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1464351657629454418)** (884 messages🔥🔥🔥): 

> `LLMs for Guided Tasks, AI system disregard, Sycophancy in LLMs, GPT 5.2 Grounded in Reality, Agentic AI/Automation` 


- **GPT-5.2: Grounded in Reality, Hated by Some!**: One member stated that **GPT 5.2** is more grounded in reality and disagrees with the user, which is why so many people hate it.
   - However, there was a discussion about how GPT agents don't learn from additional information provided after their initial training, clarifying that uploaded files are saved as "knowledge" files but don't continually modify the agent's base knowledge.
- **LLMs Ready for Guided Tasks? Debate Erupts!**: One member stated that **LLMs** are completely ready for guided narrow targeted tasks, providing a [ChatGPT share link](https://chatgpt.com/share/6973e37d-789c-8005-8cc3-2679c4a631e4) as evidence.
   - Another member countered that today's automation/agentic AI is utter trash, linking back to [messages in the ai-discussion channel](https://discord.com/channels/974519864045756446/998381918976479273/1464217595044429905).
- **Sycophancy no more!**: It was discussed that sycophancy in LLMs is a thing of the past.
   - One member stated that **GPT-4o and o4** were sycophantic, and anyone who used them extensively likely slipped into full AI psychosis.
- **Agentic AI Faces Security Scrutiny**: Concerns were raised about **agentic AI** being tricked into spilling private info or performing unauthorized actions, even with system prompts.
   - Members debated the extent to which system prompts can prevent agents from going off-topic and the privacy implications of agents recalling previous conversations.
- **Is AI Growth Stagnant? Members Disagree**: One member questioned why AI growth has been stagnant with no new releases since Gemini 3.0, while others pointed to new open-source models and updates to Codex and Claude Code.
   - The dynamic nature of AI chatbots and the constant tweaking of parameters by AI companies were cited as reasons for performance changes.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1464477927759413442)** (9 messages🔥): 

> `IDE for Codex, GPT 5.2 nerf, ChatGPT Plus for Cyber Security` 


- **VS Code & Codex Extension Boost Use Health**: A member recommends **VS Code** with the **Codex Extension** noting *Use Health, has been a better experience overall*.
   - They added that *Health’s downloadable files were pretty OP, less mistakes than in the past with GPT*.
- **GPT-5.2 Allegedly Nerfed**: A member asked if others noticed a **nerf** to **GPT-5.2** on their website.
   - They stated that *the model suddenly became stupid a week ago*.
- **ChatGPT Plus: Cyber Security Study Buddy?**: A member is considering using **ChatGPT Plus** for cyber security studies.
   - They are wondering if it's worth it *to make detailed specific exam styled questions using my revision files*.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1464991260476444874)** (178 messages🔥🔥): 

> `Heavy negations, Consequence learning, MCP paradigm, Sora prompting` 


- **Negation creates not-so-Reliable AI Results**: A member shared a [ChatGPT link](https://chatgpt.com/share/69763cc6-5360-8000-a850-85cbce128037) regarding **negation** and its unreliable results with AI.
   - They pointed out that **LLMs and AI** in general struggle with negation, leading to potential errors at scale.
- **Consequence Learning trumps Token Policing**: A member defended their approach to **consequence learning**, stating it is about making AI *experience and internalize the real outcomes of its actions*, rather than just avoiding negative instructions.
   - They argued that current "negation issues" arise from training models without real consequence feedback, contrasting it with token policing or instruction-tuning.
- **MCP Paradigm Shift reduces Token Bloat**: A member discussed the **MCP paradigm shift** by Anthropic, where AI now writes code to interact with tools, reducing token bloat by keeping interactive chatter and tool definitions out of the context.
   - They emphasized that with the new **discoverability function**, agents must be made aware of the MCP discovery process itself, a stronger instruction than *Do not hallucinate tools*.
- **Sora Struggles with Structured Prompts**: A member sought advice on improving **Sora's** output using a structured prompt for a video, but another member suggested that **Sora** doesn't handle prompts formatted like this effectively.
   - The suggestion was to try natural language translation, writing the prompt as a vivid visual description in paragraph form for better results.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1464991260476444874)** (178 messages🔥🔥): 

> `Negation in LLMs, Consequence Learning, MCP Tool Discoverability, Sora prompting tips, AI Safety and Ethics` 


- ****Negation Navigates Nuances, LLMs Lack Legibility****: Members discussed the challenges **LLMs** face with **negation**, noting that AI systems struggle with understanding negative instructions, potentially leading to unreliable results and this is a [well-documented issue](https://chatgpt.com/share/69763cc6-5360-8000-a850-85cbce128037).
   - It was highlighted that **negation comprehension** is a general challenge across various model types, emphasizing the need for caution when relying on negative instructions in **prompt engineering**.
- ****Consequence Conundrum: AI's Action-Outcome Alignment****: A member introduced the idea of **consequence learning**, where AIs learn by experiencing the real outcomes of their actions, contrasting it with training based on mere token policing or instruction-tuning.
   - A debate ensued regarding the validity of this approach versus conventional methods, with one side arguing for the importance of **real-world feedback** and the other emphasizing the significance of scaled experimentation and existing research.
- ****Sora's Storytelling Snags: Cracking Cinematic Creation****: A member sought advice on prompting **Sora** to generate videos following specific cinematic guidelines, particularly with characters appearing naturally within the frame, rather than from out of nowhere.
   - It was suggested to translate the technical prompt format into natural language descriptions with concise, semantically rich paragraphs for better results.
- ****MCP's Makeover: Model Mastery via Contextual Coordination****: The discussion highlighted **Meta-Contextual-Prompting (MCP)**, where the architecture has been changed so that the AI writes code instead of interacting with the MCP tools directly, allowing the AI to be aware of **tool discovery**.
   - The member noted that **Anthropic** developed this standard, and it has been largely embraced by the domain of **agentic development**.
- ****AI's Algorithmic Angst: Averting Anarchy via Alignment****: A member voiced concerns about safety, particularly with systems that remove moral framing and guardrails, arguing that such rail-less agents, are a liability.
   - It was argued that Alignment isn't "rails for trains"; it’s the Navigational Compass of the system, and that it is dangerous to internalize outcomes without a moral or ethical heuristic, optimizing for user compliance at the cost of objective truth or social safety.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1464349302175436873)** (848 messages🔥🔥🔥): 

> `Perplexity Pro Limits, Comet Browser Concerns, Image Generation Issues, Gemini vs GPT-5.2, AI alternatives for restricted countries` 


- **Perplexity Pro Users Ponder Query Limits**: Pro users are reporting hitting **limits on enhanced queries and file uploads**, despite having "practically unlimited" plans, leading to speculation about undocumented limits and potential downgrades in service.
   - Many users are frustrated, saying the service is becoming a **scam** and considering unsubscribing because of the restrictions, and the fact that it is **difficult to get ahold of customer service**.
- **Comet Browser Sparks Malware Mayhem**: Some users are claiming the **Comet browser** installed by Perplexity contains **malware**, advising others to analyze the software using tools like VirusTotal.
   - However, others dismiss this, questioning the source of the flagged installer and calling the claim *"mad retarded holy shit"*.
- **Image Generation Implodes**: Pro users are experiencing **issues with image generation**, with some unable to generate any images and receiving messages stating the feature is unavailable.
   - There are also reports of **video generation being limited** to 5 videos a month for Pro users, with some prompts resulting in static images instead of videos.
- **Gemini 3 Gains Ground on GPT-5.2**: Users are debating the merits of **Gemini 3** versus **GPT-5.2**, with some claiming Gemini is superior for specific tasks like trip research due to its integration with Google Maps.
   - Others state that **GPT and Grok** might be better for more broader questions.
- **AI Access: A Sanctioned Saga**: Users in **Russia** are discussing the challenges of accessing AI services due to **sanctions**, including the use of VPNs and third-party services to circumvent restrictions.
   - Chinese AI alternatives are mentioned, but some users express reluctance due to data usage concerns, suggesting options like LMArena (though access may also be limited).


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1464349472543736094)** (829 messages🔥🔥🔥): 

> `NB Pro vs NB 3 Pro Image Generation, LMArena censorship, Wan 2.6 image, Gemini and GPT Quality, Grok` 


- **NB 3 Pro outshines previous models in image generation**: Users find that **NB 3 Pro** generates higher quality images than previous models, being better than all other models except **NB Pro**, especially with **fictional weapons**.
   - Although, **no AI model can accurately generate AR rifles** and **bullpup weapons**.
- **LMArena has issues with Censorship**: LMArena's censorship is questionable, with AI generated *women holding guns* being allowed, but AI-generated *women sleeping* is blocked.
   - The moderation team is [collecting examples of false positives](https://discord.com/channels/1340554757349179412/1447983134426660894) to improve the moderation.
- **Wan 2.6 struggles in T2I**: `wan2.6-image` is **image-edit only**, requiring an image upload to work, while `wan2.6-t2i` **doesn't have image upload available**.
   - The team is aware of the issue and is working on enabling image upload for `wan2.6-t2i`.
- **GPT 5.2 High search is garbage**: **GPT 5.2 High search hallucinates more** than other models, and one user found that **Gemini's deep research also sucks in comparison** because it skims instead of carefully reading sources.
   - One user said that *ever since 4.5 came out it really has changed my life* and called Claude *good hearted, its weird how you can feel that*
- **Where Banana 2k disappeared to**: Users discuss where the **Banana 2k** model disappeared to, with some users claiming it had been removed, while others claimed that it was still available or perhaps integrated into the new **NB pro**.
   - It was later announced that it had been restored by staff members and that **it had been on vacation**.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1464386642818105457)** (4 messages): 

> `Text-to-Image model, Image Edit model, Code Arena model, Text Arena model, Image Edit Leaderboard` 


- **WAN-derful New Image Models Arrive**: A new **Text-to-Image** model `wan2.6-t2i` and a new **Image Edit** model `wan2.6-image` are now available on the [LM Arena](https://lmarena.ai/c/new?chat-modality=image).
- **Devstral Destroys in Code Arena**: The `devstral-2` model has been added to the [Code Arena](https://lmarena.ai/c/new?chat-modality=code&mode=direct-battle) for direct battles.
- **Qwen Quenches Thirst for Text**: The `qwen3-max-thinking` model is a new addition to the [Text Arena](https://lmarena.ai/?chat-modality=chat).
- **Hunyuan's Hues Hit High Rank**: `Hunyuan-Image-3.0-Instruct` now ranks #7 on the [Image Edit leaderboard](https://lmarena.ai/leaderboard/image-edit).
- **Molmo Models Multiply**: The `molmo-2-8b` model has been added to the [Text Arena](https://lmarena.ai/?chat-modality=chat).


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1464452624370896976)** (1 messages): 

> `Database Incident, Generations API Impact, Activity Page Issues` 


- **Database Incident Derails Generations API**: A **database incident** was reported to impact the **Generations API** and **activity page** starting <t:1769221560:s>.
   - The incident was later reported as resolved at <t:1769228340:s>.
- **Generations API Faces Fallout**: Due to the **database incident**, the **Generations API** experienced interruptions, impacting user activity.
   - Engineers worked to restore functionality, with the incident fully resolved by <t:1769228340:s>.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1464888279148003460)** (6 messages): 

> `Levante Integration, MCP Native AI Workspace, Password Cracking Tool, Discussion of Illegal Use` 


- **Levante Integrates as MCP-Native AI Workspace**: A user shared the integration of **Levante**, an open‑source **MCP‑native AI workspace** designed for interacting with local models like **Ollama** with a modular interface, available for download [here](https://www.levanteapp.com).
- **Password Cracking Tool Sparks Controversy**: A user expressed concern over a tool marketed for *PII-targeting* and password guessing, labeling it a potential tool for *identity theft* rather than *security research*.
- **Illicit Bitcoin Wallet Cracking Tactics Discussed**: Concerns arose due to explicit discussions about cracking other people's crypto wallets, potentially leading to computer fraud, theft of cryptocurrency, and unauthorized system access.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1464350656482640097)** (754 messages🔥🔥🔥): 

> `OpenRouter Gacha, Competitive AI Usage Platform, OR Logs Destination, GPT-5.2, Waifu orbs` 


- **Users Request **OpenRouter Gacha****: Users playfully requested an **OpenRouter Gacha** system, with one suggesting a pity mechanism involving pulling **GPT 5.2** or **Gemini 3 Pro** after a certain number of attempts.
   - One user joked about setting **OR logs destination** to `waifu.orb.town/fun/bucket` for ultra-rare pulls, later clarifying it was just a joke.
- **Users discuss competitive platform**: A user shared a platform to compare **AI usage** with other devs, seeking feedback at [https://burntop.devkp.42](https://burntop.devkp.42).
   - Another member suggested marketing to the **JAI userbase** and creating a separate gooning leaderboard to track tokens used for "gooning."
- **Discussion testing moderation filter**: Some users noticed a user with a **Chinese/Japanese nickname** sending and deleting messages, speculating about testing moderation filters or server indexing.
   - The members think that the user is testing posture.
- **Users find free and paid models**: A user inquired about maintaining extended free model limits if credits drop below 10, and if **extended free model limit** is kept access even if credits drop below 10.
   - Other users asked questions and had a discussion.
- **Gemini hallucinating**: A user reported **Google Gemini 3 Pro** hallucinating, giving a future date, and fabricating stories of time travel, suggesting OpenRouter investigate.
   - The user was directed to discord support and the issue appears to be related to a system prompt.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

jakobdylanc: https://openrouter.ai/minimax/minimax-m2-her
  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1464557110049112201)** (13 messages🔥): 

> `Kimi AI, Cerebras GLM performance, OpenRouter image support` 


- **Kimi AI Claims K2.5 Title**: Members in the OpenRouter discord mentioned a new chatbot at [Kimi.com](https://www.kimi.com/) claiming to be **Kimi K2.5**.
   - This lines up with the *kiwi-do* stealth model in LMArena.
- **OpenRouter Lacks Image Tooling**: A member spent **$5** after discovering that OpenRouter maps *image/png* tool outputs to string instead of image.
   - They posted an example [image](https://cdn.discordapp.com/attachments/1392278974222307469/1465410878382805082/image.png?ex=697901bb&is=6977b03b&hm=21677e978d8654f93d20edecf997bd4f49fb0dd08781cf93f15df8e2661ba1b5&) expressing frustration at the lack of image support.
- **Cerebras GLM Rocks 190 TPS**: **Cerebras** is consistently scoring approximately **190 TPS** on **GLM 4.7**.
   - Members noted that **Together AI** only achieves **100 TPS**, making Cerebras nearly twice as fast.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1464350807045443747)** (662 messages🔥🔥🔥): 

> `Terraform Infrastructure Blueprints, Cursor Usage Caps, Gemini API Key Logging Delay, Cursor Client Issues, Auto Mode Changes` 


- ****Terraform Blueprints** Ignite AI-Assisted Project Starters**: A member shared a [repo of opinionated Terraform infrastructure blueprints](https://github.com/berTrindade/terraform-infrastructure-blueprints) designed to be copy-pasteable and production-aware, aiming to improve the consistency of starting patterns for AI tools in new projects.
   - The goal is to enable AI to recommend appropriate blueprints based on project requirements, but members noted the [link was initially broken](https://github.com/berTrindade/terraform-infrastructure-blueprints).
- ****Usage Caps** Cause Consternation for Cursor Customers**: Users are reporting inconsistencies in achieving expected usage limits on Pro and Pro+ plans, with one member noting they reached **~$45** on Pro and **$100** on Pro+, leading to questions about value per dollar.
   - Some speculate that initial months may offer higher usage, while others share strategies to optimize token consumption, such as starting [new chats frequently](https://cursor.com/docs/cli/reference/slash-commands) and using smaller models like **GPT-5 Mini**.
- ****Gemini API** Key Logging Lags Lead to Lingering Looks**: Members are discussing a significant delay in the logging of usage and costs for **Gemini API keys**, with one user reporting waiting **20 hours** without seeing any registered usage.
   - This delay raises concerns about accurately tracking expenses and managing usage effectively, prompting questions about potential workarounds or solutions.
- ****Client Issues** Trouble Some Techies**: Several members are experiencing issues with the Cursor client, including problems connecting to past agent convos and general connectivity issues.
   - Suggested solutions include [checking the Cursor forum](https://forum.cursor.com/t/cursor-ai-is-no-longer-able-to-load-chats-locally/143599/13), trying different HTTP versions in settings, or re-opening the client without restoring editors.
- ****Auto Mode** Axed After Algorithm Adjustment**: Members noted the removal of the ability to make agents fully autonomous, as well as **image generation** capabilities in auto mode.
   - It was also suggested that **auto mode** routes to Composer 2 with one user adding, *“I'm 200% sure he does but still.”*


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1464354761867526145)** (516 messages🔥🔥🔥): 

> `Chinese models, Local LLMs on CPU, MCP tools, 4080 vs 3090 for gaming, M5 Pro macbook` 


- ****Chinese Models Surge Ahead**?**: Some members find **Deepseek** and **Qwen** models impressive in their reasoning capabilities, wondering why Chinese models are *kinda ahead of American models*.
   - One member suggests American models prioritize subscriptions over open access, while another jokes that Deepseek and Qwen excel at *appearing to be good at reasoning*, even when they don't nail it down.
- ****CPUs Cope Coding Challenges?****: One member reports running **LLMs off CPU** has been working OK for some tasks, as long as the models aren't too large.
   - Another member with an Intel i3 expresses the need to save up for an **Nvidia** card, while others suggest **AMD** options like the **MI50** or **7900 XTX** as cheaper alternatives for text generation.
- ****MCP tools: Making the most of them**?**: Members discuss challenges with **MCP servers**, noting they're not designed for LM Studio, leading to potential malformed requests and a poor user experience.
   - For file handling, a member suggests giving the **MCP server** the file path, requiring the server to handle it, while another recommends building your own coherent stack for practical agent use.
- ****4080 or 3090 for gaming?****: A user considering a **4080** is advised to get a used **3090** or **7900 XTX**, but they game more than use AI.
   - Discussion reveals the **3090** is better for gaming only at 4K resolution, and the hypothetical **5070 Ti** is much faster than either.
- ****M5 Pro macbook pros soon to launch?****: Members speculate on the release of **M5 Pro Macbook Pros**, with rumors pointing to an event on the 28th.
   - Concerns are raised about the memory bandwidth of **M4 Pro**, with suggestions that it may not be sufficient for larger models, and discussion shifts to the cost and performance of **M1 Ultra** Mac Studios.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1464370714789413042)** (134 messages🔥🔥): 

> `AIO vs Air Coolers, Ryzen 7950x Temperatures, Unified Memory Machines, Image/Video Gen Hardware` 


- **AIOs Beat Air Coolers in Hot Climates**: Members find that AIO liquid coolers are much better than air coolers in hot climates, noting a **10C** difference between an AIO and a Noctua D-15, especially under sustained CPU utilization, since air coolers often hit their limit after **5 minutes**.
   - It was argued that there's *0 reason to go for it instead of an AIO* unless afraid of water, adding that *the arctic freezer 360 is 10 euro cheaper*.
- **Ryzen 7950x Runs Hot, Even with D-15**: Users report that the Ryzen 7950x can reach **95C** even with a Noctua D-15 air cooler, and recommend switching to an AIO to keep temps down to **80C** during boosting.
   - While some suggest limiting the CPU to **70C**, others claim *no performance loss* at **95C**, though this may depend on the specific CPU binning and workload.
- **Unified Memory Mini-PCs: Hype or Disaster?**: One user purchased an AI Max + 395 mini PC, hoping for performance comparable to a **7900 XTX** due to its unified memory, but others cautioned that while it can run larger models, it will be slower than discrete GPUs.
   - It was suggested that the AI Max + 395 mini PC will likely perform **20% worse** than a similarly spec'd **M4 Max** due to bad **ROCm** support.
- **GPU VRAM Matters for Image/Video Generation**: Users discussed hardware requirements for video generation with models like **WAN 2.2**, noting that while **16GB** of VRAM is sufficient to run the model, more VRAM (like a used 3090) is preferable.
   - While z-image turbo is decent for a **4090**, there's no 'LM Studio' equivalent for image gen, forcing users to use ComfyUI, while other suggest it is *one of the best things to ever happen to image gen*.
- **Fan Cards Keep Dual GPUs Cool**: One member asked for a suggestion for 2x GPU setup to put air between the GPUs, and was given the idea to use fan cards to push air between the GPUs.
   - They look like GPUs and plug into PCI-e slots.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1464364447782011023)** (205 messages🔥🔥): 

> `Recursive AI $4B Valuation, Landing AI Job, UC Berkeley Sky Lab Startups Funding, OpenAI's PostgreSQL Scaling, Vibe Coding iOS Surge` 


- **Recursive Intelligence raising Funds at $4B**: **Recursive Intelligence** is raising funds at a **$4B valuation**, focusing on using AI to accelerate chip design, creating a self-improving feedback loop between hardware and artificial intelligence ([Bloomberg Article](https://www.bloomberg.com/news/articles/2026-01-23/ai-startup-recursive-in-funding-talks-at-4-billion-valuation)).
- **Landing AI jobs without previous AI experience**: **Noam Brown** outlined how to secure a role at a top AI lab by building a public track record through independent projects, and participating in visible competitions ([link](https://xcancel.com/polynoamial/status/2014084431062114744)).
   - He emphasized the importance of improving upon existing peer-reviewed research and participating in visible competitions like the **NanoGPT** speed run to demonstrate technical excellence, citing [Keller Jordan](https://github.com/KellerJordan/modded-nanogpt) as an example.
- **UC Berkeley Sky Lab Startups Valuation Surges**: **Alex Dimakis** highlighted significant January 2026 funding milestones for UC Berkeley Sky Lab startups, including **SGLang** at a **400m** valuation, **VLLM** at **800m**, and **LMArena** at **1.7B** ([link](https://xcancel.com/alexgdimakis/status/2014508959621959724?s=46)).
- **FastRender Browser by AI coding agents**: **Simon Willison** discusses a conversation with Wilson Lin regarding **FastRender**, a new browser rendering engine developed using over **2,000 AI coding agents** ([link](https://simonwillison.net/2026/Jan/23/fastrender/)).
- **Microsoft's Maia 200 AI Accelerator hits Azure**: **Satya Nadella** announced that the **Maia 200 AI accelerator** is now operational in **Azure** ([link](https://xcancel.com/satyanadella/status/2015817413200408959)).
   - The custom chip is designed for high-performance inference, offering **30% better performance per dollar** and optimized specs including **216GB HBM3e** and **7TB/s memory bandwidth** to support large-scale AI workloads.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1464696410887946261)** (10 messages🔥): 

> `Remotion Launchpad, Motion Canvas, Tencent HunyuanImage 3.0-Instruct` 


- ****Launchpad** lifted off as **Remotion** remix**: Francesco open-sourced [Launchpad](https://xcancel.com/francedot/status/2014897878347743732), a **Remotion** based setup for product launch videos.
   - It features video templates, shared animation components, and integration with **Claude Code** to enable rapid video creation.
- ****Motion Canvas** motivated by moviemakers**: **Remotion** is built on [motion canvas](https://github.com/motion-canvas/motion-canvas) which was originally designed by a game designer/youtuber.
   - The designer's [YouTube channel](https://www.youtube.com/@aarthificial) features quiet fun to watch.
- ****HunyuanImage 3.0** hone in on instructions**: Tencent has launched [HunyuanImage 3.0-Instruct](https://xcancel.com/TencentHunyuan/status/2015635861833167074), a native multimodal **80B MoE model** specializing in precise image editing and multi-image fusion.
   - It features a Native Chain-of-Thought (**CoT**) reasoning schema and the **MixGRPO** algorithm to improve intent alignment and synthesis quality, delivering State-of-the-Art performance comparable to leading proprietary models.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1464456699971637524)** (141 messages🔥🔥): 

> `Spaces Docker Build Pauses and 503 Error, Reinforcement Learning Channels, Windows 11 Hugging Face Models App, Lighton OCR, HeartMula, LTX-2 and the Qwen-3 TTS in ComfyUI` 


- ****HuggingFace Spaces getting the blues****: Users experienced **pauses** during **Spaces docker builds** and received a **503 error** on restart ([discuss.huggingface.co](https://discuss.huggingface.co/t/spaces-docker-build-pauses-and-503-error-on-restart/171149/2)).
   - It seems like the underlying infrastructure issues were causing the spaces to become unresponsive, requiring manual intervention to resolve, many people were getting `Something went wrong when restarting this Space` errors.
- ****RL Channel Roll-up****: Course-related **Reinforcement Learning** channels have been merged into a new [unified channel](https://discord.com/channels/879548962464493619/1329142738440028273) for better organization.
   - The old instructions in the **Deep Reinforcement Learning** course are outdated, so members should now refer to the consolidated channel for relevant discussions.
- ****VoltageGPU Volts Up****: [VoltageGPU.com](https://voltagegpu.com) is offering cheap GPUs for open-source AI models, with an **NVIDIA GeForce RTX 5090 pod** available at **$0.53/hour**.
   - They highlight the benefits of their advanced **32GB GDDR7**, optimized for inference on **HF-hosted models like Qwen3-32B**, and are offering free credits for users to try their services.
- ****Latency on Latitude for Large Language Models****: **Latitude.sh**, a bare metal cloud provider with **1,000+ GPUs**, has submitted PRs to become a **HuggingFace inference provider** ([JS Client](https://github.com/huggingface/huggingface.js/pull/1927), [Python Client](https://github.com/huggingface/huggingface_hub/pull/3715), [Docs](https://github.com/huggingface/hub-docs/pull/2180)).
   - They have models like **Llama 3.1, Qwen 2.5/3, DeepSeek R1, and Gemma 2** deployed with an **OpenAI-compatible API** and are seeking feedback on their PRs.
- ****OpenCV Saves the Day****: For agentic document processing, a member found that **OpenCV** works well for detection and extraction of text, images, and LaTeX math from applied ML papers, instead of general models like Florence.
   - They are seeking a better, small model for captioning that is capable.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1464426521027416195)** (38 messages🔥): 

> `Layer-Native Safety Clamping, GPU-64 Architecture for LLM Inference, webXOS RLHF Gaming Initiative, KV Cache in LLM Inference, ML deployment and inference platform` 


- **Safety Clamping Prevents Jailbreaks**: A new paper introduces **Layer-Native Safety Clamping**, an approach that clamps activations inside the model to prevent jailbreaks, and the team released a [dataset](https://huggingface.co/datasets/Pacific-Prime/safety_dataset) of **10K pairs**.
   - This approach learns *harm directions* in activation space and clamps any activation that projects too strongly, thus it cannot be bypassed via prompt manipulation; the paper can be found [on Zenodo](https://zenodo.org/records/18359832).
- **GPU-64 Boosts LLM Inference**: A new **GPU architecture** designed exclusively for inference, called **GPU-64**, was published, and the innovation involves a Hardware **KV-Cache** using on-chip **CAM** (Content-Addressable Memory).
   - The results show **4x faster inference** at **75W** (O(N) → O(1)), and the paper can be found [on Zenodo](https://zenodo.org/records/18364282) while the [RTL + Emulator](https://github.com/Complexity-ML/gpu64-inference) are on GitHub.
- **webXOS Gaming Initiative**: A paper introduces the **webXOS RLHF Gaming Initiative**, a framework for generating high-quality multimodal datasets through browser-based interactive gaming experiences, as described in [this Claude artifact](https://claude.ai/public/artifacts/358eea9a-4eec-4b92-be36-43797d8a76e4).
   - The initiative leverages modern web technologies to eliminate hardware barriers while maintaining the precision required for advanced **RL applications** in robotics, computer vision, and autonomous systems.
- **KV Cache Troubleshooters**: A member shared a [Medium article](https://medium.com/@nainia_ayoub/kv-cache-in-llm-inference-7b904a2a6982) breaking down the **KV Cache** in LLM Inference, which saved them time when debugging **CUDA OOM** (out of memory) errors.
   - Other members chimed in sharing that *kvs a bitch too bc most people forget it exists ngl*.
- **One-Line-Of-Code ML Deployments**: A member announced a ML deployment and inference platform for a hackathon this weekend, accessible with a one-line Python SDK, and containerizes the model in a **Docker container**.
   - The model artifacts are sent to a **Go backend**, which containerizes the model in a Docker container, exposed through a reverse proxy, and has a UI that allows to run inference and gives a live API endpoint; drop a like on the [X post](https://x.com/deepto98/status/2015153491052990841).


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1464407627395108875)** (30 messages🔥): 

> `GAIA Agent Course completion, LLM from scratch, Llama 3.2 vision agent, Summarization Pipeline, LMStudio and Deployment` 


- **GAIA Agent's Certificate Quest**: A member reported passing the **GAIA Agent Course Unit 4 Project** with a **30%** and inquired about obtaining their certificate.
   - Another member suggested going to the [robot-learning-tutorial](https://huggingface.co/spaces/lerobot/robot-learning-tutorial).
- **Llama 3.2 Vision Agent's Blind Spot**: A member is trying to build an agent using **Llama 3.2 vision** to generate captions for a list of pictures, but the model is not *apparently* passing the images to the model.
   - The member shared a preliminary code snippet.
- **LLM testing & deployment**: A member recommends **LMStudio** for testing models due to its user-friendly GUI and search filters for HF and GH models and **llama.cpp** for single-user deployment.
   - They advised against using LMStudio for backend deployment, instead suggesting **llama.cpp's llama-server** in a docker container or **vLLM's server** for better scalability.
- **Sauce for extending LLM knowledge**: A member explains that **RAG (Retrieval Augmented Graphing)** is used to extend the knowledge of an LLM without training by storing the *meaning* of words/sentences as embeddings in a vector storage.
   - They clarified that **embedding models** are models, which are trained to search vectors for hashes similar to the prompt, then include it in the prompt.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1464428370161832007)** (36 messages🔥): 

> `MXFP8 quantization, MLSys 2026 FlashInfer-Bench Competition, Nvidia Triton Inference Server, GPU Mode GTC meetups, Madlab Liger-Kernel integration` 


- **FlashInfer-Bench Competition at MLSys 2026**: The MLSys 2026 FlashInfer-Bench Competition tasks participants with designing AI agents to write state-of-the-art **LLM inference kernels** on the latest **NVIDIA Blackwell GPUs**, competing against expert-written **FlashInfer kernels**, detailed at [mlsys26.flashinfer.ai](https://mlsys26.flashinfer.ai/).
- **Triton Inference Server discussion**: A member inquired about discussing the **Nvidia Triton inference server** and a **BLS script calling a vLLM backend model** on Triton.
   - Another member suggested using the general channel, noting it was the first time someone had asked about it and someone was trying to figure out how to pass the thinking budget parameter through to vLLM.
- **GPU Mode Social at GTC**: GPU Mode is planning a winner announcement for the **nvfp4 competition** and a **social event** around the time of **GTC (March 16-19)**.
   - The event will likely involve a social event, with past events including Beer Garden and the Outside market.
- **Cornserve for Multimodal Models**: A member shared their work on **Cornserve**, an efficient online serving system for Any-to-Any multimodal models, detailed in a paper [Cornserve](https://arxiv.org/abs/2512.14098).
   - Cornserve optimizes deployment plans for models with heterogeneous components like **multimodal encoders**, **LLMs**, and **Diffusion Transformers (DiTs)**, improving throughput and reducing tail latency.
- **RSS Feed Requested for GPU Mode News**: A user requested an **RSS feed** for the GPU Mode news page ([https://www.gpumode.com/v2/news](https://www.gpumode.com/v2/news)) and offered to contribute.
   - A member responded offering the site's **GitHub repository** ([https://github.com/gpu-mode/kernelboard](https://github.com/gpu-mode/kernelboard)) for contributions and jokingly suggested testing if Claude could implement the feature.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1464609026993164444)** (7 messages): 

> `7 point 3D stencil computation, CUDA sample with a 25 pt stencil, Time zone bug, cutile fused moe kernel in the gym repo` 


- **CUDA Sample gets Stencil Spotlight**: A member is looking for tips on optimizing **7 point 3D stencil computation** and another member suggested a [CUDA sample](https://github.com/NVIDIA/cuda-samples/blob/4f735616ba599fe93cc2c6c85dcb4369260f9643/Samples/5_Domain_Specific/FDTD3d/inc/FDTD3dGPUKernel.cuh) with a **25 pt stencil** that could be modified.
- **Time Zone Bug Talk**: Members were debugging a [time zone bug](https://x.com/theemozilla/status/2015251642585682405?s=20).
   - One member asked, *what made you think of time zones and not something like Y2K and dtype overflows?*
- **Cutile Fused MoE Kernel Quest**: A member is seeking an *easy to integrate blackwell optimized kernel* and asked if anyone has tried the **cutile fused moe kernel** in the gym repo.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1465051744789729312)** (3 messages): 

> `BF16 Autocast, Dynamic Shapes, cu128 vs cu126, A100 issues` 


- **BF16 Autocast throws errors with Dynamic Shapes**: A member reported that **bf16 autocast** with **dynamic shapes** on **torch 2.10** with **cu128** throws errors on an **A100** with **cuda 13**.
   - The user noted that everything works fine on a **cu126 wheel**, but breaks on a **cu128 wheel**.
- **Request for Issue Elaboration**: A member asked for more details on the issue, specifically requesting the error message.
   - The same member also requested clarification on any additional details available to assist with troubleshooting.


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1464685212125368320)** (2 messages): 

> `CornServe, 2025 GPU MODE Recap, 2026 GPU MODE Plans, Kernel LLM Training, Hardware Programming Complexity` 


- ****Cornserve** is Served Hot**: GPU MODE went online with a member to discuss **Cornserve: Easy, Fast and Scalable Multimodal AI** ([YouTube link](https://www.youtube.com/watch?v=VhjUM_M71Wo)).
- ****GPU MODE** had wild success in 2025**: **2025** was an incredible year for **GPU MODE**: **26K** YouTube subs, **92** lectures, **24K** Discord members, **3x $100K+** kernel comps, **400K** KernelBot submissions, **3** events, and **10** active working groups!
   - The community received shoutouts from role models like Soumith Chintala, Ian Buck, Tianqi Chen, Shotlo Douglas, Tri Dao and Lisa Su for [project popcorn](https://gpu-mode.github.io/popcorn/).
- ****GPU MODE** unveils their 2026 plans**: In **2026**, GPU MODE is pushing further with training a **Kernel LLM** and using it to ship kernels in important repos like **PyTorch** and **VLLM** ([gpumode.com/v2/news/gpumode-2026](https://www.gpumode.com/v2/news/gpumode-2026)).
   - The community is collaborating with **Prime Intellect**, **Modal**, and **Lambda**, focusing on de-slopifying LLM-generated kernels, post-training a kernel LLM model, end-to-end competitions, and from-scratch repos.
- ****Complex Hardware** is becoming more Complex**: Hardware is becoming more complex to program ([X link](https://x.com/bernhardsson/status/2014855658223395085?s=20)), and the community has a responsibility for making it easier!


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1464428062254039124)** (2 messages): 

> `LeCun Startup, Event Based Model` 


- **LeCun Launches New Startup: Logical Intelligence**: Yann LeCun launched a new startup called [Logical Intelligence](https://logicalintelligence.com/), an **Event Based Model (EBM)**.
   - Unfortunately, no technical details were provided, only a link to the [MLSys Conference](https://mlsys26.flashinfer.ai/).
- **Event Based Model is shrouded in mystery**: The new startup [Logical Intelligence](https://logicalintelligence.com/) focuses on **Event Based Models** but provides no technical details.
   - The website only contains marketing material, job openings, and a link to the [MLSys Conference](https://mlsys26.flashinfer.ai/).


  

---


### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1464379534966128843)** (2 messages): 

> `CUDA Kernel Optimization, Mindbeam AI Hiring` 


- **Parsewave Seeks CUDA Kernel Optimization Engineers**: [Parsewave](http://parsewave.ai/), partnering with frontier AI labs and AI infra providers, is seeking **CUDA C/C++ kernel optimization engineers** to benchmark internal models, requiring experience with **Nsight Systems / Nsight Compute** and CUDA **intrinsics** (Blackwell ideal, Hopper great too).
   - Candidates should be able to explain optimization wins and propose benchmarks showing **naive → optimized deltas**; interested applicants can apply [here](https://tally.so/r/pbDDvZ).
- **Mindbeam AI Recruiting Post Training and GPU Kernel MLEs**: Mindbeam AI, a small team focused on accelerating training for foundation models, is hiring a `post training MLE` and `GPU Kernel MLE`.
   - The company is fully remote and offers competitive pay, with interested candidates encouraged to DM for a referral; [job openings are listed here](https://jobs.ashbyhq.com/mindbeam).


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1464478780151173121)** (1 messages): 

> `Roofline Models, Kernel Optimization, Performance Analysis` 


- **Roofline Reads Readily**: A member shared a diagram to aid in understanding **roofline models** for **kernel optimization**, suitable for sharing on LinkedIn and helpful for learners.
   - The diagram visually explains how to interpret **performance bottlenecks** and optimize kernels based on hardware limitations.
- **Kernel Knowledge Kurated**: The shared diagram highlights the relationship between **computational intensity** and **memory bandwidth** in achieving optimal performance.
   - It emphasizes that understanding these limits is vital for writing efficient **GPU kernels** and maximizing hardware utilization.


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1464762075988361440)** (2 messages): 

> `Thread Coarsening Clarification, Mentorship Opportunities` 


- **Thread Coarsening Confusion Cleared**: A member initially questioned the formula for `colStart` in thread coarsening, specifically whether it should include an additional `TILE_WIDTH` factor, referencing [page 139, chapter 6.3](https://link.to.chapter).
   - The confusion was resolved after realizing the text refers to the number of **columns** a thread block is responsible for, not the total number of **elements**.
- **Asks for a Mentor for Hands-On Projects**: A member who has completed the initial chapters of a book seeks a mentor to work on practical projects, providing [their personal website](https://vanshnawander.github.io/vansh/) for context.
   - They're looking for mentorship to complement their theoretical knowledge with hands-on experience.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1464686054715031736)** (2 messages): 

> `MLSys Conference, Treehacks` 


- **Attendee Enquires About MLSys Conference Experience**: A member inquired about the **MLSys conference** experience, noting the **2026 conference** will be in **Bellevue, WA** and they plan to volunteer since they attend **Bellevue College**.
- **Member Seeks Companions for Treehacks**: A member asked for DMs from anyone planning to attend **Treehacks**.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/)** (1 messages): 

kashimoo: Data centre GPUs are the focus for the AI space, not consumer GPUs though
  

---


### **GPU MODE ▷ #[popcorn](https://discord.com/channels/1189498204333543425/1298372518293274644/1464459210942185494)** (13 messages🔥): 

> `MLSys 2026 FlashInfer-Bench competition, Weekly meeting link, Open to new contributors?, First public meeting on Feb 3` 


- ****FlashInfer-Bench** Competition Announced for MLSys 2026!**: The **MLSys 2026 FlashInfer-Bench competition** was announced and those interested in **AI kernel generation** are encouraged to participate.
   - It was mentioned that more details are available in the *new 2026 post*.
- **Link to Weekly Meeting requested**: A member asked for the link to the weekly meeting, and another member provided [this link](https://calendar.google.com/calendar/event?action=TEMPLATE&tmeid=MXFkbTJrZWlhcXQwZDluc3Q1cDBu3FidDV_MjAyNjAyMDNUMTgwMDAwWiBjMzYyMDQwNWUwYzBiNDI5YjMwNGE0YjU5ZTdiZTFjYWQzNTc0OTdlZmMxNDc1NzVmNDlhZjZlMjM0ZTA2NzdkQGc&tmsrc=c3620405e0c0b429b304a4b59e7be1cad357497efc147575f49af6e234e0677d%40group.calendar.google.com&scp=ALL).
- **Inquiries about new contributors joining**: A member asked if the channel is open to new contributors, finding it through [this link](https://gpu-mode.github.io/popcorn/) on the working groups page.
   - The member was welcomed and directed to the *new 2026 post* for information on where help is needed.
- **First Public Meeting Scheduled!**: The first team meeting is scheduled for **Feb 3**, and the organizer is checking on the meeting link's status.
   - There may be voice channel in general.


  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1464779101406494793)** (1 messages): 

> `Jetson, Torch, ONNX, TensorRT, trtexec profiling` 


- **ONNX and TensorRT performance**: A member looked into **ONNX** to **TensorRT** performance and suggested using *trtexec* to profile the engine layers.
   - They stated that one can map **TensorRT** layers to **ONNX** ops from engine metadata but had no clue on going from **ONNX** ops to **Torch**.
- **Torch Workflow Question**: The discussion involved someone seeking advice, they also mentioned **Jetson/Torch**, **ONNX**, and **TensorRT**
   - They looked into **ONNX** to **TensorRT** performance and suggested using trtexec to profile the engine layers.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1465181818662686761)** (14 messages🔥): 

> `RTX 3060 12GB for ML/DL, FP4 acceleration on Blackwell, Consumer vs Data Center GPUs, DLSS papers, 5070ti or 4070 ti Super` 


- **RTX 3060 Still Viable for GPU Learning?**: Members discussed whether an **RTX 3060 12GB** is still a good option for learning **GPUs** and doing **ML/DL** work, given its relatively low price.
   - The consensus was that it's suitable for local learning setups, especially if acquired at a good price, but training will be slow, and support for features like **FP4** on **Blackwell** will be missing; see [Nvidia's Mistral integration example](https://huggingface.co/nvidia/Mistral-7B-Instruct-v0.3-ONNX-INT4#software-integration).
- **Consumer Blackwell differs from Datacenter Blackwell**: A member debated whether buying an expensive consumer GPU is worthwhile for writing kernels, considering the differences between **consumer Blackwell (SM_120)** and **data center Blackwell (SM_100)**.
   - Though core kernel skills transfer, staying current with architecture-specific optimizations is crucial for job market relevance.
- **GPU Fundamentals can be learned on older GPUs**: It was suggested that while newer architectures are important, general **GPU fundamentals** can be learned on older GPUs for fast iteration.
   - The recommended progression is to build a project on **Ampere**, then tune it for **Blackwell**, and continue adapting to newer architectures.
- **5070ti or 4070ti Super better than 2x3060**: A member with two **RTX 3060** cards suggested that **12GB VRAM** is limiting, advocating for a single **5070ti** or **4070 ti Super** instead.
   - They asked about papers available which explain **DLSS**.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1464661449061175339)** (1 messages): 

> `Factorio blueprint generation` 


- **AI Engineer Discovers Factorio Blueprint Generation Project**: An AI engineer researching ways to generate **Factorio blueprint JSON code** from instructions found the project impressive and stumbled upon it during their research.
- **Potential of AI in Automating Factorio Blueprint Creation**: The discussion highlights the potential of using AI models to automate the generation of **Factorio blueprints**, specifically focusing on creating JSON code from user instructions.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1464375553678119036)** (1 messages): 

> `Graphical Layout Calculus, Tuple Morphisms, Mutual Refinement, Layout Composition` 


- ****Laying out Layout Compositions Graphically****: A member shared a worked example of computing the composition of two layouts by hand using the **graphical layout calculus**.
   - The steps involve converting layouts to **tuple morphisms**, finding mutual refinements, pulling back, pushing forward, composing, and writing the result as a layout using prefix products.
- ****Mapping to Morphisms for Layout Mastery****: The initial step involves converting tractable layouts to **tuple morphisms** `m_A` and `m_B`.
   - This transformation allows for algebraic manipulation and composition of layouts using morphism operations.
- ****Refining Relations Between Layouts****: The worked example emphasizes the importance of finding a **mutual refinement** of the two tuple morphisms.
   - This step ensures compatibility and consistency when composing the layouts, akin to finding a common ground between two different structures.
- ****Pulling Back for Precise Layouting****: The process includes **pulling back** the mutual refinement along `m_A` to obtain `\hat{m}_A`.
   - This operation adjusts the refinement to be compatible with the structure of layout A, allowing for seamless integration during composition.
- ****Pushing Forward for Polished Placement****: The example also involves **pushing forward** the mutual refinement along `m_B` to get `\hat{m}_B`.
   - This operation ensures that the refinement aligns with the structure of layout B, further facilitating smooth composition and consistent layout behavior.


  

---


### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1465299821467406430)** (1 messages): 

> `Rust on M3, CPU and GPU kernels` 


- **Rust benchmarks on M3**: Initial **Rust** benchmarks on **M3** show roughly **5% peak** performance, with rustc's loop reordering identified as a factor.
- **Focus on CPU and GPU kernels**: The next steps involve working on kernels for both **CPU** and **GPU**, focusing on performance improvements.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1464453050436550710)** (51 messages🔥): 

> `group_gemm issues, benchmark leaderboard gap, B200 Physical Resonance, Stream Error During Submission, MLSys Contest Tracks` 


- **Group GEMM Init Overshoots FP16 Range**: The `group_gemm` problem's old tensor init logic overshoots **FP16 range**, causing **INF** values; a fix similar to `dual_gemm` init is proposed, referencing [this PR](https://github.com/gpu-mode/reference-kernels/pull/89/files).
   - Some **INF** values are acceptable, but the team is open to changes; a PR was opened to address the issue ([PR 96](https://github.com/gpu-mode/reference-kernels/pull/96)).
- **Benchmark Leaderboard Gap**: A significant discrepancy between benchmark results and leaderboard scores was observed, raising concerns about inconsistencies.
   - The description stated that `K` is divisible by **256**, but there are `K=128` and `K=384` in the test cases, and it was suggested to remove or modify these cases.
- **Veitner Blogs on Grouped Blockscaled GEMM for B200 GPUs**: Simon Veitner published a blog post explaining grouped blockscaled GEMM for **B200 GPUs** in a top-down manner and the setup of **MMA** and **TMA**, tile scheduler, and other parts, available on [bearblog.dev](https://veitner.bearblog.dev/grouped-blockscaled-gemm-host-code/) and [LinkedIn](https://www.linkedin.com/posts/simon-veitner-174a681b6_grouped-blockscaled-gemm-host-code-activity-7420898572637962240-5kUN).
   - The blog aims to explain the parts that are different from the usual persistent blockscaled dense gemm approach on **B200**.
- **Stream Error Plagues Submissions**: Users encountered a "stream" error during submission, traced to the presence of the word "stream" in the code, even within comments.
   - Removing the word "stream" from the code (including comments) resolved the submission issue.
- **Task Config Differences Examined**: Differences between test and benchmark configurations in `task.yml` were noted, where benchmark configs have the same **N** and **K** for all **A**'s and **B**'s in a group, unlike test configs.
   - The team clarified that the test is for function verification where **m/n/k** in different groups can be different, and the performance test comes from real use cases where **M (experts)** are different and **N/K** are the same.


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1464660580232200447)** (8 messages🔥): 

> `Learning GPUs, TinyML, Embedded ML, Physical AIs` 


- **Newcomer Seeks Guidance on GPU Learning**: A software engineer is looking for book suggestions to understand **GPUs**, **optimization**, and **tuning for performance**, aiming to transition into **TinyML/embedded ML** or **physical AIs**.
   - They prefer learning through books due to struggling with lengthy videos and have a basic understanding of **ML** but lack hardware knowledge.
- **Background Doesn't Matter; Interest Does**: A member shared that they got their **GPU performance position** with a background mainly in **math-physics** & **formal methods**.
   - They state that US companies have a lot of capital invested in AI currently.
- **Search Specs Online**: It was suggested that interviewers generally allow candidates to search online for specific specs during interviews, recognizing that specific details are easily searchable.
   - A member mentioned that whether the questions are 'googlable' is highly dependent on the company and the interview topic.


  

---


### **GPU MODE ▷ #[flashinfer](https://discord.com/channels/1189498204333543425/1464407141128339571/1464407381680062630)** (53 messages🔥): 

> `Team Merging, Multiple Track Participation, Registration Confirmation, Team Formation, Kernel Type (Single/Multi-node)` 


- ****Team Merging Mania****: Participants inquired about [team merging](https://discord.com/channels/1197226372415488060/1464407141128339571) before the registration deadline, to which the organizers responded that it is allowed, with a request to be notified of the changes.
   - The organizers also set up an automated [registration confirmation email](https://discord.com/channels/1197226372415488060/1464407141128339571) in response to requests.
- ****Track Shifting Shenanigans****: Contestants asked if [participation in multiple tracks](https://discord.com/channels/1197226372415488060/1464407141128339571) was possible, the organizers confirmed it but noted only one GPU would be awarded even if a team ranked highly in multiple tracks.
   - The discussion clarified that teams can [shift tracks](https://discord.com/channels/1197226372415488060/1464407141128339571) later to focus on the most promising one.
- ****Kernel Confidentiality Conundrum****: Participants raised questions about whether [submissions would be made public or kept private](https://discord.com/channels/1197226372415488060/1464407141128339571) after the competition.
   - Organizers clarified that the final code needs to be made public for award consideration, but the [development process can remain private](https://discord.com/channels/1197226372415488060/1464407141128339571).
- ****Newbie Navigation Notes****: Beginners inquired about the best [track selection for newcomers](https://discord.com/channels/1197226372415488060/1464407141128339571) in the NVIDIA MLSys contest.
   - The recommendation was to [deploy the smallest possible model using the FlashInfer API only](https://discord.com/channels/1197226372415488060/1464407141128339571) to become comfortable with the codebase, while avoiding unstable platforms like B200.
- ****Agent's Secrets Safe (Mostly)****: Clarification was sought regarding the requirement for [open-sourcing agent solutions](https://discord.com/channels/1197226372415488060/1464407141128339571) in the FlashInfer AI kernel competition.
   - The organizers confirmed that while the agent code and tech report will be reviewed, only the [final code needs to be open-sourced](https://discord.com/channels/1197226372415488060/1464407141128339571) to ensure it's not a hand-crafted kernel.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1464355109260754964)** (57 messages🔥🔥): 

> `ROCm performance for ML, Image classification services, DistinctionBench and language models, Human-in-the-loop workflows with LLMs, In-context learning and weight updates` 


- **ROCm's Rocky Road to ML Renaissance**: Users discussed the performance of **ROCm** for accelerated ML, noting it has made strides but can be challenging due to primary support for **Nvidia**.
   - The experience was described as *'batteries not included'* due to potential driver problems and long lead times.
- **DistinctionBench: Training Target or Contamination Resistant?**: Discussion on the paper **Between Circuits and Chomsky: Pre-pretraining on Formal Languages Imparts Linguistic Biases** considered **DistinctionBench** to give interesting transfer to language models.
   - One member joked, *'all good evals are training targets ;)'*, but noted **DistinctionBench** is *'very contamination resistant'* due to endless representational variants.
- **ICL Signals: Are Weights Updating?**: A member asked about papers on *'using the signals from in context learning to update the weights as a form of continual learning,'* and two relevant papers were shared: [chen24r.html](https://proceedings.mlr.press/v235/chen24r.html) and [arxiv.org](https://arxiv.org/abs/2507.16003).
   - The conversation also pointed to saving inference costs via pushing stuff into the parametric knowledge, reminiscent of *'state tuning in linear attention variants'* and *'cartridges'* from this summer ([https://arxiv.org/abs/2506.06266](https://arxiv.org/abs/2506.06266)).
- **Attention Arrived Before the Transformer**: Attention mechanisms existed on top of RNNs in **2014-2015**, but it took two years to introduce the transformer because people weren't convinced about attention.
   - It was suggested that there were fewer people working in the field back then, and **Kaggle** results really helped it take off.
- **Forbes Article fails to meet contribution standards**: A member posted a Forbes article with commentary, but another member responded that *'Pretending that a Forbes article that is a copy/paste of popular questions of Quora is a representation of what leading research questions in AI are does not meet our contribution standards.'*
   - The member then added the heuristic: *“Is this a conversation that two AI researchers might have” is a good heuristic.*


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1464353106333794427)** (121 messages🔥🔥): 

> `Weak Baselines, RWKV Architecture, One-Step Generative Modeling, Hybrid Architectures, Deduplicated Pretraining Data` 


- ****Weak Baselines** Bashed**: Members debated the validity of *"weak baselines"* as a complaint against research, arguing that even beating **ChatGPT** doesn't guarantee significance without a strong baseline.
   - It was emphasized that experiments should start with robust baselines to avoid mistaking noise for genuine improvements, suggesting **modded nanogpt** as a good starting point for language tasks, with one member recommending replicating [this paper](https://arxiv.org/abs/2002.05202).
- ****RWKV** Revamp Rumors**: A member shared their work on modifying the **RWKV architecture**, but others cautioned about parameter count and training methods, recommending training on tokens instead of bytes.
   - It was suggested that the modifications should be tested on recent **RWKV codebases** with attention baselines, and renting a **4090** or **5070ti** was recommended due to CPU limitations, plus that the approach could be related to **FFN-Bilinear**.
- ****Gate-Free** FFN Flounders?**: Experimentation with **gate-free FFNs** showed a 4.3% parameter reduction but only a 0.5% improvement compared to a sigmoid gate, raising questions about the efficiency of added parameters in MLPs.
   - One member suggested that gating might help fix **GQA**, and that **Lora gate params** or **near-MiSS formulation** (expansion from a subregion of the hidden state) might improve results without the significant parameter count increase. Another shared that in their work, taking the last 12 dims of the residual dim and using it for the **attn gate** seemed to do pretty well.
- ****Generative Modeling** Gauntlet**: With a surge of papers on **one-step generative modeling**, members discussed which methods are promising, noting the difficulty in comparing benchmarks and separating noise from valuable advancements.
   - One member advocated for a theoretical understanding to tier methods and avoid impractical options, while another agreed math *"soundness"* plays a big role.
- ****Symbolic Sanity Checks** Save Sanity**: The potential of **hybrid architectures** combining **LLMs** with **symbolic/deductive layers** for hallucination reduction was explored.
   - While checking logical consistency is relatively easy for math, code, and simple facts as shown in [this paper](https://arxiv.org/abs/2409.13724), it remains challenging for other types of consistency as shown in [this paper](https://arxiv.org/abs/2507.10624).


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1464493909609808046)** (4 messages): 

> `Model Weights Comparison, Free Compute Resources, Automating Circuit Discovery, OLMo 3 models` 


- **OLMo 3 Models Suit Needs**: A user suggested that **OLMo 3** may suit another user's needs, as it has separate **SFT** and **thinking models**.
   - They suspected it's close enough to warrant a preliminary study of model weights.
- **Compute Resources for Model Finetuning Sought**: A user is working on a project to compare model weights of two variants of the same model, one finetuned to follow instructions and the other to solve reasoning tasks, and asked for free compute resources.
   - The user sought resources to fine-tune a small model on Colab and is open to compute sharing.
- **Automating Circuit Discovery Papers**: A user requested a list of papers related to automating circuit discovery for behavior, such as for **IOI** and **induction**.
   - The user invited others to share interesting papers they find on the topic as well.


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/)** (1 messages): 

aeros93: https://fixupx.com/havenfeng/status/2014765400563781777?s=46
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1464391573553942731)** (128 messages🔥🔥): 

> `Self-Replication Benchmark for Agentic AI, LLM Worms, MoE run, OpenAI Business Model, Local Code Models and Raytracer Test` 


- **Self-Replication Benchmark Considerations**: A member is considering a **self-replication benchmark** for **agentic AI**, pondering the appropriate goal and whether the agent should download itself or retrain from scratch.
   - They suggested that adopting to a target machine or even designing one could be fun, as opposed to simply using existing transformer libraries.
- **LLM Worms: "Hey make more of you"**: One member jokingly suggested an **LLM worm** benchmark where an LLM is prompted with *"hey make more of you"* and given the tools to replicate itself, whether by downloading a copy or writing a script that downloads a script and uses an API key.
   - Another pointed out the importance of considering resource constraints like **VRAM** to make the challenge more interesting.
- **1stlanik's MoE Run Dashboard 'Failed to Fetch' error**: A member reported a *'Failed to fetch'* error in the dashboard while checking the progress of an active **MoE run (moe-10b-a1b-8k-wsd-lr3e4-1t)**.
   - Another member suggested checking back in a few hours.
- **OpenAI Pricing Model discussed**: Members discussed how **OpenAI** may be raising prices by offering services that barely work on the lower tier subscriptions.
   - One stated that *"AI companies could raise prices a lot right now and people would pay"* but another countered that many companies offer the same for cheaper. It was also mentioned that **Anthropic** has a **40% gross margin**.
- **Raytracer Test Proves Difficult for Local Models**: A member noted the difficulty of local code models (runnable on a **5090**) to pass a **raytracer test** from [cpldcpu/llmbenchmark](https://github.com/cpldcpu/llmbenchmark/tree/master/10_raytracer#readme), even observing that recent models on **lmarena** are failing it now.
   - They find that the smaller models consistently mess up the vector class.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1464917438603989067)** (2 messages): 

> `LLM pre-training for domain-specific tasks, Effectiveness of continued pre-training` 


- **Pre-training LLMs for Domain-Specific Tasks**: A member asked about the effectiveness of continued pre-training a foundational LLM for a domain-specific task like law or healthcare using **OLMO-7B** and the **ZINC20** dataset.
   - Another member, an LLM researcher, suggested it generally improves performance but is task-dependent, noting that training with task-related inputs/outputs may outperform more general continued pre-training (cpt).
- **Expanding Multilingual Capabilities via CPT**: The researcher noted that continued pre-training (**cpt**) can expand multilingual capabilities, and fine-tuning on translation data strengthens task performance.
   - This comment was specifically made in response to the general question regarding continued pre-training (**cpt**).


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1464648051334578216)** (1 messages): 

> `Semantica, Knowledge Graphs, Ontologies, LLM reasoning` 


- **Semantica: Open-Source Semantic Infrastructure**: A member introduced [Semantica](https://github.com/Hawksight-AI/semantica), an **open-source project** focused on building semantic infrastructure for **domain-grounded AI**, including **knowledge graphs**, **ontologies**, and **reasoning layers**.
   - They are actively seeking contributors for **ontology & schema design**, **knowledge graph modeling**, **LLM + symbolic / rule-based reasoning**, **data ingestion & semantic pipelines**, and documentation.
- **Semantica: Contribution Opportunities**: The project is looking for contributions in various areas, including **ontology & schema design** and **knowledge graph modeling**.
   - Contributions don’t have to be big, and issues, design discussions, feedback, or small PRs are all welcome.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1464394179525480490)** (105 messages🔥🔥): 

> `EBM vs Classical FF, EBM and Shannon Entropy, LLM pre-training, MCMC sampling issues, Zero-knowledge proofs` 


- ****EBMs vs. Classical Feedforward: Is There a Clear Winner?****: The discussion starts by questioning if **Energy-Based Models (EBMs)** are inherently superior to classical **feedforward networks**, especially concerning **Shannon entropy** or **Kolmogorov complexity**.
   - One member suggests that *validation is easier than generation* in EBMs, relating it to **computational complexity theory (P vs NP)**, while emphasizing the need for a well-defined loss landscape for EBM optimization to work effectively.
- ****LLM Pre-training: Domain-Specific vs. Foundational****: A member asked about the effectiveness of **continued pre-training** a foundational **LLM** (specifically **OLMO-7B**) for a domain-specific task like cheminformatics using the **ZINC20 dataset**.
   - The goal is to compare results against a domain-specific transformer model, but no specific answers or resources were provided in the discussion.
- ****MCMC's Messy Mode-Switching Mishaps****: A member inquires about the adequacy of a [paper](https://arxiv.org/abs/2310.11232) in illustrating **MCMC sampling issues**, particularly how badly it sucks.
   - One member argues that **MCMC** tries to emulate flow models due to the latter's superiority, while **EBMs**, contrarily, attempt to make **NNs** more like **MCMC**, which they deem misguided, as they elaborate that *HMC has issues traversing between spatially separated modes*, making it horrible as the dimension increases.
- ****ZKPs: More than Just Crypto Signing?****: A member discusses using **zero-knowledge proofs (ZKPs)** for verifying encrypted network traffic and matrix multiplications, pointing to a [Gemini correspondence](https://gemini.google.com/share/ddfc0ffcb33e) for a matrix low knowledge proof.
   - They propose a use case in *zero-knowledge “made by humans” proofs*, but another member is skeptical about the practicality of **ZKPs**, suggesting breaking the encryption might be cheaper, while the initial member claims the opposite, stating *ZKPs are theoretically even more efficient than the feedforward*.
- ****NN Parameterization: A Trio of Techniques****: A member questions the advantage of parameterizing the **score** over parameterizing **log p(x)**, and they respond that *we can just to monte carlo estimate on denoising matching term instead of both denoising matching term + reconstruction term, so we have less variance?*.
   - It's clarified that you can parameterize the distribution directly, the log-likelihood (MLE and EBMs), or the score (flow models), and that Optimal Transport (OT) is distinct, affecting what you *do* with a distribution rather than how you learn or parameterize it.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1464366556925399102)** (10 messages🔥): 

> `LLMs cyber capabilities, LLM companies internal vulnerabilities, Exploit events, Github repo security` 


- **LLMs cyber skills questioned**: A member questioned whether LLMs could develop strong *cyber capabilities*, referencing a [GPTZero article](https://gptzero.me/news/neurips/).
   - Another member doubted LLM companies' ability to address *internal vulnerabilities*, suggesting they fix those before pursuing cyber skills, also citing a [ScienceAlert article](https://www.sciencealert.com/scientists-identify-brain-waves-that-define-the-limits-of-you) and a [tweet](https://x.com/theonejvo/status/2015401219746128322).
- **Upcoming Exploit Events?**: A member predicted potential *large exploit events* and warned about LLMs' access to sensitive resources.
   - They advised using *GitHub deploy keys* in isolated environments when coding with a GitHub repo to limit potential damage.
- **No Access Granted!**: One member humorously declared that they would not grant LLMs access to anything.
   - Another member countered this sentiment by calling it *robo-phobicit's* and dubbing it *survival instincts*.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1464584124261208198)** (74 messages🔥🔥): 

> `Luminal flash attention, Metal performance with textures vs buffers, Tenstorrent backend passing ops tests, Tinygrad intended use and training LLMs, Anthropic VLIW challenge PR` 


- **Luminal finds Flash Attention via Bruteforce**: Luminal is claiming to find **flash attention** using **bruteforce** on an egraph, taking hours to find, and they explicitly added `exp(x - new_max) = exp(x - old_max) × exp(old_max - new_max)` as a rewrite rule.
   - The poster reproduced the graphviz shown in the presentations from commit `0bd3b80c`, noting that their minimal set of rewrite rules could transform a naive attention kernel graph into the known **flash attention kernel graph** in 52s on a 9800x3d.
- **Metal: Textures Beat Buffers for Blurring**: Profiling access speed on **Metal** using `Tensor` with size **512/1024/2048/8192** images as input for a **3/5/7** sized blur kernel showed textures outperforming buffers.
   - It might be worth throwing in a branching condition depending on the size of the buffer input, [tests results are attached](https://cdn.discordapp.com/attachments/1068976834928193609/1464679423029547172/Screenshot_2026-01-25_at_1.49.57_AM.png?ex=6978fb82&is=6977aa02&hm=5530b74c4fce9dad5d85a4d9e7409c3809a7ee51ee548744a1fa3deb2efea1d3&).
- **Tenstorrent backend passes ops tests**: The **Tenstorrent** backend is passing all ops tests on wormhole or blackhole and there is a [$1k bounty](https://x.com/corsix/status/1880384044728480206) for this milestone.
   - Someone asked if the bounty requires all test ops test passing on **testorrent hardware**.
- **Anthropic VLIW challenge PR submitted**: A member submitted [a PR](https://github.com/tinygrad/tinygrad/pull/14332) for the **Anthropic VLIW challenge**, hitting **1258 cycles**.
   - The submitter expressed uncertainty about generalizing the code, particularly the batch staggering, which might be useful for other VLIW targets, and also apologized for a *lazy lookover* that introduced AI-generated changes.
- **Tinygrad isn't for normies**: A user asked about the intended use of tinygrad, specifically regarding porting existing models and training LLMs on multiple GPUs, and was told by George Hotz to *ask claude about this*.
   - Another user expressed frustration at being told to use Claude for documentation and said tinygrad is not for me or most devs then, to which George replied *i'm not selling to anyone, tinygrad is free* and that adoption is not a target.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1464356316922380300)** (65 messages🔥🔥): 

> `Slides Generation Issues, Login Issues, Rate Limits, Chinese AI Labs Innovation, Multimodal Models Comparison (Kimi vs. Ernie 5.0 vs. GLM 4.6V)` 


- **Slide Generation Issues Plague Users**: Some users reported issues generating slides, even with visual and adaptive options, with one user reporting that the issues have persisted since the previous day, linking to a [video](https://cdn.discordapp.com/attachments/1371757564005711973/1464357057468698746/2026-01-23_21-31-32.mp4?ex=697920c8&is=6977cf48&hm=1692b661e1fa241c6db806df2971a024f5713504a25a83612c3f5d385e00c4db&) showcasing the issue.
   - The user experiencing the issue suggested that internal **rate limits** may be the cause, and they are now able to generate slides again, suggesting the issue was temporary.
- **Kimi's Chinese Labs win big praises**: One user lauded Chinese AI labs, including **Kimi**, for their innovation and performance compared to other models like Gemini, citing **Kimi**'s human-like responses and impressive memory capabilities.
   - The user expressed a desire for **Kimi** to incorporate multimodal capabilities similar to **Minimax**, specifically vision and audio analysis for video transcription, along with tool integration and workspace features.
- **Kimi K2.5 Silently Sneaking into the spotlight?**: Users are noticing **Kimi** models self-reporting themselves as **K2.5**, despite no official announcements or UI changes indicating a new version.
   - Some speculate this could be related to internal testing or improvements to the slide tool, potentially involving visual understanding of images, but others claim that they checked and there aren't any major improvements.
- **API Login struggles**: A user reported difficulty logging into the Kimi/Moonshot platform to generate new API keys, particularly with a non-Gmail account, and was directed to contact the support email.
   - The user later clarified that the issue was not rate limits but simply forgetting the login procedure for the backend.
- **Kimi is adding Memory features**: A user highlighted that Kimi's app now includes **memory features**, enabling customization, which enhances the overall user experience.
   - The memory and customization options make it a favorite chatbot.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1464440102456529160)** (52 messages🔥): 

> `Mojo in production, Mojo startup lag, VS Code debugging, CPU vs GPU kernels` 


- ****Mojo** Sees Production Use in HPC**: A member is deploying a **Mojo** project for parameter refinement in cryo-electron microscopy and seeing a **5-10x** speedup over the old **C++** code.
   - The biggest win was pulling of an **AoSoA layout** for one bit, made super easy by **Mojo's** list of structs with SIMD members.
- ****Mojo's** Cold Executable Starts Slowly 🐌**: A member reported a **200ms** startup lag for even simple **Mojo** scripts, which they tracked down to a *Gatekeeper* issue on macOS scanning untrusted binaries, where subsequent runs were much faster.
   - They found a **50ms** launch overhead on a cold executable after rebooting, which they considered acceptable.
- **VS Code Debugging Still Has Issues 🐛**: A member reported debugging with the **VS Code** extension fails due to a *"Function not implemented"* error on an air-gapped machine using `.conda` files from [max-nightly](https://prefix.dev/channels/max-nightly).
   - A Modular employee mentioned debugging in the extension should be working on Mac and Linux with environments set up using **Pixi** as described in the [Quickstart guide](https://docs.modular.com/max/get-started).
- **GPU Kernel Portability is a pipe dream**: A member noted that standard **CPU** kernels under-utilize the **GPU**, requiring specialized code, and another suggested GPUs could be treated as wider **SIMD** units to simplify programming.
   - He suggested using a *number of warps* instead of *number of threads* to solve this issue.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1464391677748838572)** (12 messages🔥): 

> `Mojo 1.0 Release, `def` functions, Pointer.mojo, out self` 


- **`def` functions decision pending for Mojo 1.0**: With **Mojo 1.0** release in a few months, the decision on including `def` functions is still pending, with a member pinging **Denis** for a response on [GitHub issue #5830](https://github.com/modular/modular/issues/5830).
   - Currently, there's no committed date for **Mojo 1.0** other than *"in 2026"*.
- **`out self` argument in Pointer.mojo discussed**: A member noted that in `Pointer.mojo`, the `__init__` function's first argument is not `self`, but another **Pointer**, questioning if this deviates from the standard.
   - Another member explained that `out` arguments only serve as output and do not affect the call signature, so the position of `out self` doesn't matter technically, but convention suggests putting it first in `__init__`.
- **Argument order matters for parameter inference**: A member explained that `out self` must be the second argument in this case because it depends on `other` for one of its parameters - `ImmutOrigin(other.origin)`.
   - Another member added that the argument order is relevant for parameter inference.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1464368919081914399)** (50 messages🔥): 

> `Manus billing issues, Manus free credit codes, AI Engineer introductions, AI + Healthcare systems, AI Agent development` 


- **User Demands Resolution for Unauthorized Billing**: A user reports being charged **$400** for an annual plan after selecting monthly billing and threatens complaints to FTC, BBB, Attorney General, and Meta due to [unauthorized billing](https://ftc.gov), refused refunds, and unresponsive support.
   - Another user recommends filing a chargeback.
- **Manus Free Credit Code Revealed!**: One user shared a **redeem code** `Havefun` which gives **1000 credits**.
   - Another user asked where to find these codes, and was directed to the **Exchange** button.
- **AI Engineers Introduce Their Healthcare Skills**: An **AI + Full Stack Engineer** introduced expertise in building production-grade **AI** systems for **Healthcare**, including clinical NLP, medical imaging, and patient-facing AI applications.
   - This engineer also builds **LLM systems**, autonomous agents, workflow automation, and multimodal **AI** (text · voice · vision) and included a [list of their core skills](https://www.example.com/fake-list).
- **AI Agent Developer Focuses on Production Systems**: An **AI Agent Developer** highlighted their focus on building **AI agents** for real production use, rather than demos, and is available for collabs and audits.
   - The developer specializes in customer support, sales agents, workflow/ops agents, and autonomous booking/scheduling agents.
- **User Seeks 'Share with a friend' on Mobile**: A user asked where the 'Share with a friend' option is located.
   - Another user replied that on a computer, it's at the bottom of the left sidebar but, offered help for the mobile version.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1464846155232841902)** (2 messages): 

> `DevinReview, DSPy RLM, AsyncReview, RLM Skills, Claude Code` 


- **AsyncFuncAI open sources DevinReview**: A member has open sourced a version of **DevinReview** using the **DSPy RLM** framework, available on [GitHub](https://github.com/AsyncFuncAI/AsyncReview).
   - The new release has been named **AsyncReview**.
- **Add RLM Skills to Claude Code or Opencode**: A member shared the idea to add **RLM as skills** to **Claude Code** or **Opencode**.
   - The member also shared an npm package called [rlm-skills](https://www.npmjs.com/package/@unravel-tech/rlm-skills).


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1464398527881810165)** (46 messages🔥): 

> `RLM Prompt Tuning, DSPy Optimizer for Multi-Step Modules, JSON Adapter Customization with GEPA, Leveraging DSPy for Typescript Agent Optimization, DSPy via AG-UI` 


- **RLM Prompts Demand Tuning**: Users discussed tuning the **RLM prompt** itself, citing that reasoning can be lacking in some models and suggested techniques for improving the **RLM prompts**.
- **DSPy Optimizer to Inspect the Trace**: When using **DSPy optimizers** for modules with many intermediate steps, it was suggested that the optimizer will automatically inspect the trace, so users only need to focus on the desired output.
   - One user recommended preparing a good set of training data with example documents and a measurement that rejects wrong answers when the **RLM** answers prematurely.
- **JSON Adapters get GEPA Treatment**: A user wants to use GEPA to work on the text that the **JSONadapter** puts in the system prompt, given that the tokens are not always needed for the response to have the appropriate output form.
   - They believe they'll need to make a custom **GEPA adapter**, as the DSPy one doesn't affect the adapters.
- **TypeScript Agents Seek DSPy Optimization**: One user is looking to leverage DSPy for optimizing prompts of agents written in **Typescript** and asked if the architecture is currently supported or feasible in practice.
- **AG-UI DSPy Adapter Streams Events**: A user inquired about interest in exposing DSPy via **AG-UI**, highlighting its benefits for frontend/backend communication and avoiding the need for API endpoints and state management.
   - The user has a working version that streams events, including reasoning traces, tool calls, and streamed LLM responses to the frontend.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1464398502921638062)** (7 messages): 

> `Aider + Claude Code workflow, Aider and Devstral Small 2 model` 


- **Aider pairs well with Claude Code**: A user noted that **aider** is fast, making it a perfect pair for **Claude code** to punch through bug walls with agentic efficiency.
   - The user finds **aider** useful for working out which files need to be in context, managing the context, and the search and replace coder minimizes llm token outputs.
- **Devstral Small 2 works excellently with Aider**: A user reported excellent success using **Aider** with **Devstral Small 2**, a new 24B dense model.
   - At **Q4_K_M**, it fits in a **3090** with enough room for nearly **50k context**, and the search/replace blocks it generates are perfect 80-90% of the time and recovers in a single attempt when it fails.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1465407492124053567)** (2 messages): 

> `Discord voice channels, Contributor-related chat` 


- ****Discord's** New Voice Channel Experiment!**: The team is experimenting with new **Discord voice channels**, named `conference-room-a` and `conference-room-b`, available in the channels list.
   - These channels are intended for ad-hoc contributor chats to quickly resolve issues, especially when a long async text thread is ineffective.
- **Moderation and Access Rights Reminder!**: Specific members have permissions to mute people in these channels, while others should ensure they have the necessary access rights.
   - There is a reminder that the access rights will be changing in five days.
