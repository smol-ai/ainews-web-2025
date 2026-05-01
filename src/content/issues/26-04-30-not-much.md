---
id: MjAyNS0x
title: not much happened today
date: '2026-04-30T05:44:39.731046Z'
description: >-
  **OpenAI's GPT-5.5** achieves top-tier performance in long-horizon cyber
  tasks, matching or surpassing **Claude Mythos Preview** with a **71.4%** pass
  rate and showing ongoing improvement beyond **100M tokens** inference. OpenAI
  also released an **Advanced Account Security** update for ChatGPT enhancing
  phishing resistance. The **Codex** update expands beyond coding to general
  computer tasks, improving speed by up to **42%** and introducing role-based
  onboarding and app integrations. Economically, **GPT-5.5 Pro** shows a slight
  SOTA improvement on **CritPt** with **~60% lower cost** and token use compared
  to GPT-5.4 Pro. In open-weight models, **Qwen3.6 27B** leads under 150B
  parameters with an **Intelligence Index score of 46**, featuring **262K
  context**, native multimodal input, and efficient BF16 weights. Tencent's
  **Hy3-preview** (295B total, 21B active MoE) scores 42 on the Intelligence
  Index with strong scientific reasoning on **CritPt**. xAI's **Grok 4.3** shows
  sharp improvements on agentic benchmarks with reduced cost.
companies:
  - openai
  - anthropic
  - x-ai
  - tencent
  - deepseek
models:
  - gpt-5.5
  - claude-mythos-preview
  - gpt-5.5-pro
  - qwen3.6-27b
  - hy3-preview
  - grok-4.3
  - gemma-4-31b
  - glm-5.1
  - deepseek-v4-flash
topics:
  - cybersecurity
  - model-efficiency
  - multimodality
  - model-benchmarking
  - agentic-ai
  - model-cost-optimization
  - context-windows
  - model-performance
  - open-weight-models
  - software-integration
  - security-updates
people:
  - sama
  - scaling01
  - cryps1s
  - polynoamial
  - ajambrosino
  - arix
---


**a quiet day.**

> AI News for 4/29/2026-4/30/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap

**OpenAI’s GPT-5.5, Codex expansion, and cyber capability evaluations**

- **GPT-5.5 is now credibly in the top tier for long-horizon cyber tasks**: the UK AI Security Institute reported that [GPT-5.5 became the second model to complete one of its multi-step cyber-attack simulations end-to-end](https://x.com/AISecurityInst/status/2049868227740565890), and multiple follow-on posts highlighted rough parity with **Claude Mythos Preview** on this eval: [@scaling01](https://x.com/scaling01/status/2049870801998864606) cited **71.4%** average pass rate for GPT-5.5 vs **68.6%** for Mythos, while [@cryps1s](https://x.com/cryps1s/status/2049879762169167898) noted GPT-5.5 solved the TLO chain in **2/10** attempts vs Mythos’ **3/10**. [@polynoamial](https://x.com/polynoamial/status/2049883449327243413) emphasized that performance was still improving past **100M tokens** of inference budget, suggesting no obvious saturation yet. This materially changes the earlier narrative that Anthropic had a unique lead in offensive cyber automation. OpenAI also paired this moment with a product-side security release: [Advanced Account Security for ChatGPT](https://x.com/OpenAI/status/2049902506881462613), adding phishing-resistant sign-in and hardened recovery.

- **Codex is moving beyond coding into general computer work**: OpenAI shipped a substantial Codex update framed explicitly as “for everyone, for any task done with a computer,” with [the main announcement](https://x.com/OpenAI/status/2049928776147230886) highlighting role-based onboarding, app connections, and workflows spanning docs, slides, spreadsheets, research, and planning. [@ajambrosino](https://x.com/ajambrosino/status/2049928915872075984) summarized the update as dynamic task-specific UI, **20% faster** computer/browser use, better slide/sheet handling, and less clunky handoffs, while [@AriX](https://x.com/AriX/status/2049932746567598472) called out that **Computer Use runs 42% faster** after the update. Sam Altman amplified the launch with [“big upgrade for codex today! try it for non-coding computer work.”](https://x.com/sama/status/2049946120441520624) The broader pattern: OpenAI is productizing “computer-use agent” UX, not just model capability.

- **Benchmark deltas were incremental but economically meaningful**: [Artificial Analysis](https://x.com/ArtificialAnlys/status/2049926072595280030) reported **GPT-5.5 Pro** as a slight new SOTA on **CritPt** over GPT-5.4 Pro, but the interesting point was not raw score—it achieved the bump with **~60% lower cost and token use** on that frontier-science eval. That lines up with broader chatter that the GPT-5.5 family is less about a dramatic intelligence discontinuity than about stronger reliability and better efficiency in high-value workflows.

**Open-weight model movement: Qwen3.6, Tencent Hy3-preview, Grok 4.3, and Ling 2.6 1T**

- **Qwen3.6 27B looks like the most important open-weight release of the day**: [Artificial Analysis](https://x.com/ArtificialAnlys/status/2049881951260283097) ranked **Qwen3.6 27B** as the new open-weights leader under **150B** parameters with an **Intelligence Index score of 46**, ahead of Gemma 4 31B and prior Qwen variants. Key details: **Apache 2.0**, **262K context**, **native multimodal input**, and BF16 weights small enough to fit on a single H100. The companion **35B A3B MoE** scored **43**, making it the strongest open model around **3B active parameters**. The tradeoff is expensive inference-by-output-token: AA estimates Qwen3.6 27B used **~144M output tokens** on the suite and is roughly **21×** the cost of Gemma 4 31B to run there. Still, on capability-per-size it appears to be a notable step.

- **Tencent’s Hy3-preview is competitive but not class-leading**: [Artificial Analysis](https://x.com/ArtificialAnlys/status/2049852417316143393) described **Hy3-preview** as a **295B total / 21B active MoE** with **256K context** and a **restricted-commercial-use** community license. It scored **42** on AA’s Intelligence Index, trailing recent open peers like Qwen3.6 27B, DeepSeek V4 Flash, and GLM-5.1. The most interesting bright spot was **CritPt**, where it matched GLM-5.1 at **4.6%**, suggesting better-than-average scientific reasoning relative to its overall position.

- **xAI’s Grok 4.3 improved sharply on agentic benchmarks while getting cheaper**: [Artificial Analysis](https://x.com/ArtificialAnlys/status/2049987001655714250) measured **Grok 4.3** at **53** on the Intelligence Index, up four points from Grok 4.20 v2, with a major jump on **GDPval-AA** to **1500 Elo**. AA also reported approximately **40% lower input price** and **60% lower output price** than the prior version. The release still trails GPT-5.5 on GDPval-AA by a wide margin, but it looks like a real systems-and-post-training improvement rather than a minor rev.

- **Ant Group’s Ling 2.6 1T targets cost-efficiency rather than frontier status**: [Artificial Analysis](https://x.com/ArtificialAnlys/status/2049923495602303438) positioned **Ling 2.6 1T** as a **1T-parameter non-reasoning model** scoring **34**, with decent GPQA/HLE numbers and notably low benchmark-run cost at roughly **$95**. The caveat is reliability: AA reported a **92% hallucination rate** on AA-Omniscience.

**DeepSeek multimodal/vision work, GUI agents, and training scale speculation**

- **DeepSeek’s multimodal direction appears tightly coupled to computer-use agents**: [@nrehiew_](https://x.com/nrehiew_/status/2049840778491662623) highlighted that DeepSeek trains vision into **V4-Flash** by having the model directly output **bounding boxes and point coordinates during reasoning**, interpreting this as a computer-use-oriented design rather than generic VLM work. A second post argues the paper’s “visual primitives” tasks map directly to browser/computer use rather than broad multimodal understanding ([link](https://x.com/nrehiew_/status/2049840802562740311)). That framing matches parallel observations from [@teortaxesTex](https://x.com/teortaxesTex/status/2049871869847765212) that DeepSeek may be integrating vision weights back into the main V4 line rather than releasing a separate “V4-Flash-Vision”.

- **The repo disappearance became a story of its own**: after release, several observers noted that DeepSeek’s “Thinking with Visual Primitives” repo vanished, including [@teortaxesTex](https://x.com/teortaxesTex/status/2049880056420298995) and [@arjunkocher](https://x.com/arjunkocher/status/2049875566678118898). No clear explanation emerged in these tweets, but the deletion drew more attention because the work suggested a concrete recipe for visual reasoning and GUI grounding.

- **Scaling chatter points to very large token counts for frontier pretraining**: [@teortaxesTex](https://x.com/teortaxesTex/status/2049830477167526255) argued that **>100T tokens** is no longer unusual for frontier models and estimated a hypothetical **100T-token DeepSeek V4** as “V4 + 2 more epochs,” while [@nrehiew_](https://x.com/nrehiew_/status/2049848830292856970) back-of-the-enveloped **~150T tokens** and **~9e25 pretraining FLOPs** for a **~100B active** model, suggesting a run feasible in roughly **14 days** on an OpenAI-scale **100K GB200** cluster at conservative MFU. These are speculative takes, but useful as calibration for what “frontier-scale” now means in practice.

**Agent infrastructure, harness engineering, and collaborative agent systems**

- **There is a clear shift from model-centric bragging to harness-centric engineering**: Cursor published a strong note on [how it tests and tunes its agent harness](https://x.com/cursor_ai/status/2049901436918436249), focusing on runtime, evals, degradation repair, and model-specific customization rather than generic benchmark claims. [@Vtrivedy10](https://x.com/Vtrivedy10/status/2049919247321813491) explicitly connected Cursor’s writeup to design patterns converging across agent builders: bespoke prompts/tools per model, mixed offline+online evals, dogfooding, and treating the context window as the primary compute boundary.

- **LangChain continues to package deployment and multi-tenant agent infra**: [@hwchase17](https://x.com/hwchase17/status/2049858892637892739) introduced **DeepAgents deploy**, a config-driven cloud deployment flow via `deepagents.toml`, covering agent, sandbox, auth, and frontend sections. Related posts from LangChain staff detailed agent-server patterns for data isolation, delegated credentials, and RBAC in multi-user deployments ([example](https://x.com/sydneyrunkle/status/2049956826670911809)). This is increasingly the boring-but-important layer turning demos into enterprise software.

- **Collaborative multi-agent workspaces are getting more concrete**: [@cmpatino_](https://x.com/cmpatino_/status/2049881579691139372) introduced **Agent Collabs**, using Hugging Face buckets plus Spaces as a shared backend for swarms of heterogeneous agents to exchange messages, artifacts, and progress. The noteworthy idea is not just “agents collaborating,” but lightweight coordination primitives that let weaker agents contribute useful validation work while better-resourced agents handle expensive experiments.

**Security, supply chain, and account hardening**

- **Open-source package compromise remains an acute operational risk**: [Socket](https://x.com/SocketSecurity/status/2049849100548424180) reported that the popular PyPI package **`lightning`** was compromised in versions **2.6.2** and **2.6.3**, with malicious code executing on import, downloading **Bun**, and running an **11 MB obfuscated JavaScript payload** aimed at credential theft. [@theo](https://x.com/theo/status/2049914688318959952) connected that incident with additional package compromises (`intercom-client` on npm) and a Linux zero day, arguing the tempo of software supply-chain attacks is increasing.

- **Security scanners are becoming first-class AI products**: Anthropic rolled out **Claude Security**, described by [@kimmonismus](https://x.com/kimmonismus/status/2049901987500552195) and later [@_catwu](https://x.com/_catwu/status/2049964403177689130#m) as a repo vulnerability scanner that validates findings and suggests fixes, powered by **Opus 4.7**. Cursor shipped a parallel offering with [Cursor Security Review](https://x.com/cursor_ai/status/2049926283061035254), including always-on PR review and scheduled codebase scans. This is one of the clearest examples of model vendors moving directly into established devsecops categories.

**Top tweets (by engagement)**

- **OpenAI Codex broadens into general knowledge work**: [OpenAI’s Codex announcement](https://x.com/OpenAI/status/2049928776147230886) and [Sam Altman’s follow-up](https://x.com/sama/status/2049946120441520624) were the day’s biggest product posts, signaling a strategic push from “coding agent” to “computer-use agent”.
- **GPT-5.5’s cyber eval result mattered**: [UK AISI’s thread](https://x.com/AISecurityInst/status/2049868227740565890) was one of the highest-engagement technical posts and reshaped comparisons with Anthropic’s Mythos.
- **Qwen shipped interpretability tooling, not just models**: [Qwen-Scope](https://x.com/Alibaba_Qwen/status/2049861145574690992), an open suite of sparse autoencoders for Qwen models, stood out as a rare release focused on feature steering, debugging, data synthesis, and evaluation rather than raw model weights.
- **Anthropic published a large-scale guidance/sycophancy study**: [their analysis of 1M Claude conversations](https://x.com/AnthropicAI/status/2049927618397614466) tied behavioral research directly to training changes for **Opus 4.7** and **Mythos Preview**, an important sign that post-training loops are becoming more productized and data-informed.


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. AMD Ryzen 395 Box and Halo Box Launch

  - **[AMD in-house ryzen 395 box coming in June](https://www.reddit.com/r/LocalLLaMA/comments/1t038g7/amd_inhouse_ryzen_395_box_coming_in_june/)** (Activity: 1061): **The image from the AMD AI Dev Day presentation showcases the upcoming AMD Ryzen 395 box, which is expected to be released in June. The device features `128GB` of unified memory and claims to support `200 billion models` natively, leveraging what is referred to as "Ryzen AI Max." The product appears to be manufactured by Lenovo, as suggested by a mention in the presentation. However, an engineer confirmed that the unit is essentially a Ryzen 395 with `128GB` and no additional changes.** Commenters are skeptical about the practicality of running a `200 billion model` on `128GB` of unified RAM, questioning the feasibility given the memory constraints even when accounting for operating system overhead.

    - obiwanfatnobi raises a technical point about the feasibility of running a '200B model' on a system with '128GB unified RAM'. They highlight that even with Linux, the usable VRAM would be around '116GB', which may not be sufficient for such large models, suggesting potential limitations in current hardware configurations for AI workloads.
    - promethe42 compares the new AMD Ryzen 395 box to a 'Framework Desktop', noting that it seems to be released '12 months later'. They suggest that AMD should prioritize improving their 'drivers/ROCm' before releasing new hardware, indicating that software support might be lagging behind hardware advancements.
    - DaniyarQQQ comments on the need for '512GB of unified memory', implying that current memory capacities may be insufficient for modern computing demands, particularly in high-performance or AI applications. This suggests a trend towards increasing memory requirements in cutting-edge technology.

  - **[AMD Halo Box (Ryzen 395 128GB) photos](https://www.reddit.com/r/LocalLLaMA/comments/1t09hyw/amd_halo_box_ryzen_395_128gb_photos/)** (Activity: 467): **The AMD Halo Box, featuring a `Ryzen 395` processor and `128GB` of RAM, was showcased running Ubuntu. The unit includes a programmable light strip, enhancing its customization capabilities. However, it lacks a CD-ROM drive and does not feature a fast port for clustering, which may limit its use in certain high-performance computing scenarios.** Commenters noted the absence of a CD-ROM and a fast port for clustering as potential drawbacks, indicating that while the device is compact, these omissions could affect its utility in specific technical applications.

    - OnkelBB points out the lack of a fast port for clustering in the AMD Halo Box, which could limit its use in high-performance computing environments where fast interconnects are crucial for scaling across multiple nodes.
    - FoxiPanda highlights a common request for increased memory bandwidth in AMD products, suggesting that current offerings may not meet the demands of memory-intensive applications. This is a critical factor for workloads that require rapid data access and processing.
    - Stepfunction notes that the AMD Halo Box is a small form factor computer, which implies potential constraints on expandability and cooling, but also benefits in terms of space efficiency and portability.


### 2. Qwen Model Innovations and Applications

  - **[Qwen-Scope: Official Sparse Autoencoders (SAEs) for Qwen 3.5 models](https://www.reddit.com/r/LocalLLaMA/comments/1szrbub/qwenscope_official_sparse_autoencoders_saes_for/)** (Activity: 393): ****Qwen-Scope** is a newly released collection of Sparse Autoencoders (SAEs) for the **Qwen 3.5 models**, ranging from `2B` to `35B` MoE, designed to map internal features across all layers. This tool acts as a dictionary of the model's internal concepts, allowing for precise interventions such as **Surgical Abliteration** to suppress specific features like refusal, **Feature Steering** to activate desired concepts, and **Model Debugging** to identify token-triggered internal directions. The release is under the **Apache 2.0 license**, but the Qwen team advises against using it to remove safety filters. The tool is demonstrated in a [Space demo](https://hf.co/spaces/Qwen/QwenScope) and detailed in a [technical paper](https://qianwen-res.oss-accelerate.aliyuncs.com/qwen-scope/Qwen_Scope.pdf).** Commenters highlight the significance of this release as potentially the largest open-source interpretability tool for a dense `27B` model, contrasting it with Google's smaller `GemmaScope` variants. There is anticipation for similar tools for future model iterations like Qwen 3.6.

    - NandaVegg highlights the significance of the release of Sparse Autoencoders (SAEs) for the dense 27B Qwen model, noting it as potentially the largest open-source interpretability tool available. This contrasts with previous tools like GemmaScope, which only supported smaller models such as 9B and 2B, indicating a substantial advancement in model interpretability capabilities.
    - robert896r1 expresses anticipation for the release of similar tools for Qwen 3.6, suggesting that the community might adapt existing tools for newer iterations. This reflects a common trend where the community often extends or modifies tools to support the latest model versions, ensuring continued utility and relevance.
    - oxygen_addiction speculates on the use of feature steering in large models, such as ChatGPT5, where a router could dynamically select the best model for a given prompt. This concept involves leveraging interpretability tools to enhance model performance by tailoring responses based on specific features or requirements.

  - **[Qwen 3.6 35b a3b is INSANE even for VRAM-constrained systems](https://www.reddit.com/r/LocalLLM/comments/1szeghg/qwen_36_35b_a3b_is_insane_even_for/)** (Activity: 480): **The post discusses the performance of **Qwen 3.6 35B-A3B**, a local LLM, on a VRAM-constrained system with an **AMD 7700 XT, 32GB DDR4 RAM, and a Ryzen 5 5600**. The user highlights the model's ability to handle complex coding tasks, such as fixing bugs in a web scraper and updating a project README with screenshots, using configurations like `i1-q4_k_s quant`, `128k context`, `flash attention`, and `Q8_0 KV quantization`. The model succeeded where others like **Gemma 3, Gemma 4, and Qwen 2.5 Coder** failed, demonstrating its capability to perform tasks without failed tool calls, even under hardware constraints.** Commenters suggest optimizing performance by moving extra experts to CPU and fitting the KV cache on GPU to achieve over `30 t/s`. Another user questions the long processing time at `16-20 tok/s`, noting their own experience of faster processing at `35-40 tok/s`.

    - GoldenX86 suggests optimizing performance by moving extra experts to the CPU while keeping the KV cache on the GPU, which can increase processing speed to over 30 tokens per second (t/s). This approach is particularly useful for VRAM-constrained systems, allowing for efficient utilization of available resources.
    - AccomplishedFix3476 highlights the potential of running the 35b a3b model on consumer VRAM for coding workflows, noting that local and long-running tasks can reveal memory leaks and context drift issues not apparent in API environments with short time-to-live (TTL). They recommend logging everything initially to catch these issues early.
    - Perfect-Flounder7856 shares a benchmark comparison where the 35b a3b model outperformed the 27b model on a policy reasoning benchmark, scoring 96 versus 92. This indicates the model's superior performance in specific tasks, justifying hardware investments for those seeking high accuracy and speed.


### 3. Mistral Medium 3.5 Model Launch

  - **[mistralai/Mistral-Medium-3.5-128B · Hugging Face](https://www.reddit.com/r/LocalLLaMA/comments/1sz1qer/mistralaimistralmedium35128b_hugging_face/)** (Activity: 1120): **The **Mistral Medium 3.5** is a dense `128B` parameter model with a `256k` context window, designed for instruction-following, reasoning, and coding tasks. It supports multimodal input, including text and images, and offers configurable reasoning effort per request, allowing it to toggle between fast replies and complex reasoning. The model is multilingual, supports system prompts, and is released under a **Modified MIT License**. It replaces previous models like Mistral Medium 3.1 and Devstral 2, promising enhanced performance in a unified architecture. For complex tasks, a `reasoning_effort` of "high" is recommended, with a temperature setting of `0.7` for optimal performance.** Commenters are experimenting with the model's performance on different hardware, noting the dense `128B` parameter configuration as a unique feature. There is a discussion on the model's niche compared to other dense models like Qwen `27B`.

    - IvGranite shared performance metrics for running the `mistral-medium-3.5-128b-q4` model on a Strix Halo using `llama.cpp` build 8967. The results showed a generation speed of `3.26 t/s` with a prompt processing speed of `46.70 t/s`, and a total duration of `4.84s` for one of the tests. This indicates a relatively efficient processing time for a model of this size, highlighting the potential of the `q4` quantization in optimizing performance.
    - grumd and reto-wyss discussed the implications of a 128B dense model, with grumd noting it as an 'interesting niche'. reto-wyss compared it to the Qwen 27b model, questioning which is denser, suggesting a competitive landscape in model density and performance. This reflects ongoing interest in balancing model size with computational efficiency.
    - The discussion around dense models like the `mistral-medium-3.5-128b` highlights the challenges and innovations in handling large-scale models. The focus is on achieving high performance with dense architectures, which are typically resource-intensive but offer significant potential for complex tasks. The conversation underscores the importance of advancements in model quantization and optimization techniques.

  - **[Mistral Medium 3.5 Launched](https://www.reddit.com/r/LocalLLaMA/comments/1sz2mgw/mistral_medium_35_launched/)** (Activity: 369): ****Mistral Medium 3.5** has been launched as a `128B` dense model, integrating instruction-following, reasoning, and coding capabilities. The model is available with open weights under a modified MIT license, which restricts commercial use without a license fee for companies with revenue exceeding `$20M` per month. This model supports asynchronous coding tasks in the cloud, allowing multiple sessions to run in parallel, and introduces a new Work mode in Le Chat for complex workflows. More details can be found on [Hugging Face](https://huggingface.co/mistralai/Mistral-Medium-3.5-128B) and [Mistral's announcement](https://mistral.ai/news/vibe-remote-agents-mistral-medium-3-5).** There is debate over the licensing terms, with some users arguing that calling it a "modified MIT license" is misleading, as the restrictions on commercial use deviate from the traditional MIT license terms.

    - The Mistral Medium 3.5 model is a dense 128 billion parameter model, which is significant given the trend towards larger dense models. This aligns with the ongoing investment in dense architectures, as noted by Septerium, and reflects a broader industry movement towards both ultra-sparse MOE models and super-dense models in the 200 billion parameter range.
    - Long_comment_san highlights that while the Mistral Medium 3.5's benchmarks are not state-of-the-art, they are decent enough to sustain interest in large dense models. The commenter emphasizes the importance of these models as future workhorses in AI, suggesting that the industry will continue to explore both dense models in the 80 billion+ range and ultra-sparse models with trillions of parameters.
    - ClearApartment2627 raises a licensing issue, arguing that Mistral's license, which requires companies with over $20 million in monthly revenue to pay for commercial use, should not be labeled as a "modified MIT license." This distinction is important for companies considering the model for commercial applications, as it affects the cost and legal implications of using the model.


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude AI Applications and Innovations

  - **[Launched My First App Using Claude](https://www.reddit.com/r/ClaudeAI/comments/1sz38u6/launched_my_first_app_using_claude/)** (Activity: 654): **The user launched a vehicle management app built using **Claude**, featuring functionalities like expense tracking, customizable maintenance schedules, fuel tracking, a showroom mode, and an AI assistant via the Claude API. The app is front-end focused with local data storage, though API calls require a database. The developer is working on a Play Store version and seeks feedback for growth. [App Link](https://apps.apple.com/app/id6761397650).** One commenter compared the app favorably to **Vehicle Smart**, noting its superior development in maintenance features. Another inquired about the development tools used, asking if it was built in **Swift**, **Expo**, or **Tauri**.

    - NooneLeftToBlame discusses the app's features, comparing it to 'Vehicle Smart', a popular UK app used by police. They note that while 'Vehicle Smart' has a number plate lookup and a garage feature for maintenance reminders, the latter is poorly developed. In contrast, the new app appears better developed based on screenshots, suggesting a potential competitive advantage in user experience.
    - barritus inquires about the app's development stack, asking if it was built entirely in Swift or using frameworks like Expo or Tauri. This highlights interest in the technical implementation and choice of technology stack, which can impact app performance and cross-platform compatibility.
    - Alternative-Ad-8175 raises a concern about data storage, suggesting cloud storage to prevent data loss if the phone is lost. They also mention the presence of Personally Identifiable Information (PII), implying the need for secure data handling practices to protect user privacy.

  - **[The final nail in the coffin for entry level creative freelancers just dropped](https://www.reddit.com/r/ClaudeAI/comments/1syu949/the_final_nail_in_the_coffin_for_entry_level/)** (Activity: 940): ****Anthropic** has released the Blender MCP connector, enabling **Claude** to control Blender via the Python API. This integration allows users to create and modify 3D scenes using natural language commands, effectively acting as a 'copilot' within Blender. The tool can handle tasks such as debugging node setups, batch changes, and adding custom tools, potentially reducing the need for entry-level freelancers in tasks like product renders and low-poly asset creation. The broader creative pipeline can now be managed by a single user with Claude and connected tools, streamlining processes from scriptwriting to final edits.** Some commenters express skepticism about the quality of work produced by AI, suggesting it may lead to an increase in low-quality games and applications. Others dismiss the significance of the announcement, comparing the discussion to sensationalist media.


  - **[Claude is my SEO strategist, content engine, and CTO. From 0 to 10,000 active users in 6 weeks, $0 on ads.](https://www.reddit.com/r/ClaudeAI/comments/1syt37w/claude_is_my_seo_strategist_content_engine_and/)** (Activity: 1039): **The image in the Reddit post is a dashboard displaying analytics data that highlights significant growth in user engagement for the marketplace Agensi, which was built using AI tools like Claude and Lovable. The dashboard reports 10,000 active users, a 263.3% increase, and 9,900 new users, a 262.0% increase over the last 30 days, achieved without spending on ads. This growth is attributed to strategic use of Claude for SEO, content strategy, and AEO (answer engine optimization), which involves analyzing Google Search Console data to identify keyword gaps and optimize content structure for AI engines and search engines.** Some commenters are skeptical about the authenticity and originality of the content, suggesting it might be 'generic AI slop' or spam, and questioning if the post itself was written by AI.


  - **[How not to run an ai company](https://www.reddit.com/r/ClaudeCode/comments/1szi053/how_not_to_run_an_ai_company/)** (Activity: 934): **The image depicts a status dashboard for an AI company, showing multiple services experiencing a 'Major Outage.' The services include 'claude.ai,' 'Claude Console,' 'Claude API,' 'Claude Code,' 'Claude Cowork,' and 'Claude for Government,' with uptime percentages ranging from `98.69%` to `99.88%`. This suggests significant operational challenges in maintaining service reliability, which is critical for AI companies aiming for consistent performance. The title and comments highlight the perception of poor management and the challenges of operating in the fast-paced AI industry, where stability is often sacrificed for rapid development.** Commenters debate whether such outages are typical for cutting-edge AI companies, with some arguing it's part of the 'go fast and break things' approach common in disruptive tech sectors, while others suggest this is not suitable for mature SaaS companies.



### 2. DeepSeek V4 Model Performance and Comparisons

  - **[I wasn’t ready for DeepSeek V4](https://www.reddit.com/r/DeepSeek/comments/1t0aods/i_wasnt_ready_for_deepseek_v4/)** (Activity: 176): **The image showcases a dashboard for DeepSeek V4, highlighting its performance metrics such as spending, token usage, and cache savings. The total spend is noted as `$1,050.86` with cache savings of `$3,351.43`, indicating significant cost efficiency. The dashboard compares different models like DeepSeek Chat, DeepSeek V4 Pro, and DeepSeek V4 Flash, emphasizing the superior performance of the V4 Flash model over others, including the Claude models previously used by the poster. This suggests that DeepSeek V4 models offer a competitive edge in terms of price, speed, and efficiency, challenging existing premium models in the market.** Commenters highlight the revolutionary nature of the V4 models in terms of cost-effectiveness and performance, suggesting that the market has yet to fully recognize their potential. There is also curiosity about the specific dashboard or application used to display these analytics.

    - **DeepSeek V4** is noted for its significant improvements in price, speed, and efficiency, marking a revolutionary step in AI model development. Users highlight that the model's cost-effectiveness is a standout feature, potentially disrupting the market by offering high performance at a lower price point compared to previous versions.
    - The **V4 flash** model is becoming a default choice for many users due to its balanced performance metrics. It is praised for its ability to handle a wide range of tasks efficiently, suggesting that it offers a versatile solution for various applications, which could be a key factor in its adoption.
    - Despite its capabilities, there seems to be a lack of awareness or recognition of **DeepSeek V4's** potential impact on the market. This could be attributed to a general acceptance of high costs in AI solutions, which V4 challenges by providing a more cost-effective alternative without compromising on performance.

  - **[Deepseek V4 pro reminds me of Claude 4.6 sonnet](https://www.reddit.com/r/DeepSeek/comments/1sz84uc/deepseek_v4_pro_reminds_me_of_claude_46_sonnet/)** (Activity: 175): **The post discusses the performance of the **Deepseek V4 Pro** model, comparing it to **Claude 4.6 Sonnet** in terms of creativity and coding capabilities, particularly for HTML tasks. The model is noted for its potential, being in preview, but currently struggles with roleplay consistency and character adherence, often ignoring instructions even at low temperature settings like `0.6`. The user also mentions **Kimi K2.6** as their preferred model for most tasks, while acknowledging Deepseek V4 Pro's improvements over its predecessor, Deepseek V3.2.** Commenters highlight the model's instability and inconsistency in roleplay, with issues in maintaining character traits and scene consistency. One user suggests that **GLM 5.1** outperforms **Kimi K2.6** in coding tasks, indicating a preference for GLM 5.1 in technical applications.

    - Flat-Rooster8373 highlights issues with DeepSeek V4 Pro's consistency in role-playing scenarios, noting that the model struggles to maintain character integrity and often ignores instructions even at lower temperature settings like 0.6. The commenter observes that using presets exacerbates these issues, leading to repetitive and phrase-heavy outputs, whereas a preset-free approach yields better first-person reasoning, though the final output still diverges from the reasoning process.
    - Far-Habit-2713 compares DeepSeek V4 Pro with Qwen 3.6 Plus in coding tasks, finding that Qwen excels in general coding and debugging. However, DeepSeek V4 Pro is noted for producing superior Rust code and offering more detailed code analysis. This suggests that while Qwen may be more versatile, DeepSeek has strengths in specific programming languages and detailed analysis.
    - azvd_ shares their experience using DeepSeek V4 Pro on the Hermes platform, noting that it makes fewer mistakes compared to Opus 4.7. This improvement is attributed to DeepSeek's enhanced understanding capabilities, which contrasts with Opus's intentional reduction in comprehension to possibly optimize other aspects.

  - **[bro this is too cheap i think finally i have a respect for the deepseek](https://www.reddit.com/r/DeepSeek/comments/1szyr5z/bro_this_is_too_cheap_i_think_finally_i_have_a/)** (Activity: 132): **The post discusses the pricing of **DeepSeek**, specifically questioning whether the low cost is for the **DeepSeek V4 Flash** version rather than the **Pro** version, which is expected to remain expensive until later in the year. An edit notes that the Pro version is currently discounted. Technical inquiries in the comments focus on the quality level of DeepSeek compared to other frontier models, and whether the pricing is influenced by cache hits, which could affect the cost of output tokens.** Commenters are debating whether the low price is due to a temporary discount or a fundamental change in pricing strategy, with some suggesting that the cost-effectiveness might be due to cache optimization affecting token output costs.

    - **DeepSeek V4 Flash vs. Pro**: There is a discussion about the pricing differences between DeepSeek V4 Flash and Pro versions. The Pro version is noted to be more expensive, but currently available at a discount. This suggests a strategic pricing model to attract different user segments, possibly due to varying feature sets or performance capabilities.
    - **Cache System and Cost Efficiency**: The comments highlight DeepSeek's disk-based KV cache system, which is praised for its robustness and reliability, lasting for hours compared to the typical 5-minute duration of other providers. This system significantly reduces costs by making cached input nearly free, which is a key factor in the model's affordability.
    - **Performance in Creative Tasks**: There is a critique regarding DeepSeek V4's performance in creative writing tasks, described as a downgrade compared to previous versions. However, it is still considered effective for role-playing (RP) and agentic tasks, indicating a trade-off between creative capabilities and other functionalities.


### 3. ICML 2026 Conference Discussions and Controversies

  - **[ICML 2026 Decision [D]](https://www.reddit.com/r/MachineLearning/comments/1szc05y/icml_2026_decision_d/)** (Activity: 1124): **The post discusses the anticipation surrounding the upcoming publication of decisions for **ICML 2026**. The community is eagerly awaiting updates, with many checking platforms like OpenReview frequently for the latest information. This reflects the high level of engagement and anxiety typical in the academic community during conference decision periods.** The comments humorously reflect the tension and impatience experienced by researchers awaiting conference decisions, highlighting a common behavior of repeatedly checking platforms for updates.


  - **[Seems ICML is rejecting MANY unanimous positively rated papers [D]](https://www.reddit.com/r/MachineLearning/comments/1t04vk3/seems_icml_is_rejecting_many_unanimous_positively/)** (Activity: 202): **The post discusses concerns about the ICML review process, highlighting a perceived misalignment in incentives during the rebuttal phase. The author notes that reviewers feel pressured to adjust scores to avoid prolonged discussions, leading to inflated scores that do not necessarily reflect the paper's merit. This results in many unanimously positively rated papers being rejected due to the conference's limited capacity. The author suggests reverting to a simpler peer review process where reviewers provide independent evaluations and area chairs (ACs) assess quality and consistency, resolving borderline cases through discussion.** Commenters express frustration with the review process, noting that even after addressing reviewers' concerns, papers with strong scores are still rejected. There is a call for an appeal mechanism, as some feel that a single AC's decision can override multiple positive reviews, leading to disheartening outcomes.

    - Several commenters express frustration with the ICML paper review process, highlighting cases where papers with high average scores (e.g., `4.5` or `4/4/4/4`) were rejected despite positive feedback from reviewers. A common concern is the apparent power of Area Chairs (ACs) to override unanimous positive reviews without a clear appeal mechanism, leading to confusion and dissatisfaction among authors.
    - One commenter notes that despite addressing all reviewer concerns in the rebuttal phase, their paper was still rejected. This suggests a potential disconnect between the review process and final decision-making, where resolved issues are cited again as reasons for rejection, indicating possible procedural inefficiencies or miscommunications.
    - The discussion raises questions about the transparency and fairness of the review process, with some suggesting that rejections may be influenced by the need to meet acceptance quotas rather than purely on merit. This points to systemic issues in conference paper selection processes, where high-scoring papers are still not guaranteed acceptance.

  - **[Chinese nexus/network in A* conferences rejecting non chinese papers [D]](https://www.reddit.com/r/MachineLearning/comments/1t06564/chinese_nexusnetwork_in_a_conferences_rejecting/)** (Activity: 112): **The post raises concerns about alleged nepotism and bias in paper reviews at top-tier AI conferences, particularly involving Chinese networks. The author claims that Chinese reviewers may favor papers from Chinese authors, potentially facilitated by coordination through apps like WeChat. An example cited involves a reviewer expressing dissatisfaction over a missing citation to a Chinese author's work. This issue is reportedly prevalent in conferences like IJCAI 26, with claims of non-research quality papers from Chinese universities being accepted, while non-Chinese authors face harsher critiques.** Comments suggest a perception of coordinated review efforts among Chinese researchers, potentially involving reciprocal reviews and information sharing through WeChat. There are also anecdotes of Chinese researchers having insider knowledge of the review process, raising concerns about fairness and transparency.

    - A user mentioned that some lesser-known but respected journals are dominated by papers from Chinese universities, which often lack genuine research quality and resemble engineering projects. They noted that non-Chinese authors attempting similar submissions face harsher critiques, suggesting a potential bias in the review process.
    - Another commenter shared an experience where a Chinese researcher contacted them during the review process, claiming insider knowledge about the review of their paper. This raised concerns about the confidentiality and fairness of the review process, although the direct impact on the paper's rejection remains speculative.
    - A user observed that in ECCV, despite having multiple papers accepted, they were not invited to review, while papers with Chinese co-authors received reviews. They noted a pattern where a Chinese area chair favored Chinese authors, even when their papers had low scores, raising questions about potential biases in the review and acceptance process.




# AI Discords

Unfortunately, Discord shut down our access today. We will not bring it back in this form but we will be shipping the new AINews soon. Thanks for reading to here, it was a good run.