---
id: MjAyNS0x
title: not much happened today
date: '2026-05-13T05:44:39.731046Z'
description: >-
  **Cline, LangChain, Notion, and Cursor** advanced agent infrastructure and
  developer platforms with innovations like **Cline SDK**, **LangSmith Engine**,
  **SmithDB** (offering **12–15×** faster observability), and Notion's External
  Agents API integrating third-party agents such as Claude and Codex. Agent UX
  trends emphasize **long-running state, streaming, and orchestration** over
  chat, with tools like **Duet Agent** and **VS Code Agents window** enhancing
  durable execution and inspectable states. Research highlights include **Nous
  Research's Token Superposition Training** achieving **2–3× speedup** in
  pretraining, a **multi-stream LLM** architecture for parallel reasoning by
  Jonas Geiping et al., and **δ-mem** external memory improving benchmark
  scores. NVIDIA's **Star Elastic** offers post-training model compression at
  **360× lower cost** than pretraining, while Datology focuses on data curation
  for vision-language models.
companies:
  - cline
  - langchain
  - notion
  - cursor
  - nous-research
  - nvidia
  - datology
models:
  - claude
  - codex
  - langsmith-engine
  - smithdb
  - duet-agent
  - multi-stream-llm
  - delta-mem
  - star-elastic
topics:
  - agent-infrastructure
  - developer-platforms
  - observability
  - long-running-state
  - streaming
  - orchestration
  - pretraining-efficiency
  - model-architecture
  - external-memory
  - post-training-compression
  - data-curation
  - vision-language-models
people:
  - jonas_geiping
  - siddharth_joshi
  - pratyush_maini
---


**a quiet day.**

> AI News for 5/12/2026-5/13/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap

**Agent Infrastructure, Harnesses, and Developer Platforms**

- **Cline, LangChain, Notion, and Cursor all pushed deeper into agent platform territory**: [Cline](https://x.com/cline/status/2054580767779700775) open-sourced a rebuilt **Cline SDK** and refreshed CLI with a TUI, agent teams, scheduled jobs, and connectors, positioning its harness as a reusable substrate for custom coding agents. [LangChain](https://x.com/LangChain/status/2054617687238865013) shipped a large batch of agent lifecycle infrastructure at Interrupt: **LangSmith Engine**, **SmithDB**, **Sandboxes**, **Managed Deep Agents**, **LLM Gateway**, **Context Hub**, and **Deep Agents 0.6**. The most technically notable piece is [SmithDB](https://x.com/LangChain/status/2054658661776244936), a purpose-built observability database for nested, long-running traces with large payloads, reportedly yielding **12–15×** faster access on key workloads; the team says it is built atop [Apache DataFusion and Vortex](https://x.com/ankush_gola11/status/2054681251513254260). In parallel, [Notion’s External Agents API](https://x.com/NotionDevs/status/2054600524423733307) lets third-party agents such as Claude, Codex, Cursor, Decagon, Warp, and Devin operate directly inside Notion as a shared, reviewable context layer rather than another silo. [Cursor](https://x.com/cursor_ai/status/2054651526715502998) expanded cloud agents with fully configured **development environments** including cloned repos, dependencies, version history, rollback, scoped egress, and isolated secrets.
- **Agent UX is increasingly about long-running state, streaming, and orchestration rather than chat**: Several launches converged on the same design direction. [Duet Agent](https://x.com/dzhng/status/2054619807715348779) proposes a state-machine harness for jobs that last **weeks or months**, with parent/sub-agent coordination and memory replacing compaction. LangChain’s OSS updates added [streaming typed projections, checkpoint storage, code interpreter, harness profiles, and model-specific tuning](https://x.com/LangChain_OSS/status/2054641656222388700), all aimed at richer agent event streams than plain tokens. [Tabracadabra](https://x.com/oshaikh13/status/2054613590695641269) moved from autocomplete to a context-aware assistant in any textbox, while [VS Code](https://x.com/code/status/2054669377367064613) introduced an Agents window and better multi-project task review. The architectural message across these releases is that production agents increasingly need **durable execution, inspectable intermediate state, and tool-native UI surfaces** rather than stateless prompt/response loops.

**Model Training, Architecture, and Data Efficiency**

- **Pretraining efficiency and architectural experimentation were the strongest research throughline**: [Nous Research’s Token Superposition Training](https://x.com/NousResearch/status/2054610062836892054) modifies the early phase of pretraining so the model reads/predicts contiguous bags of tokens before reverting to standard next-token prediction; they report **2–3× wall-clock speedup at matched FLOPs** with no inference-time architecture change, validated from **270M to 3B dense** and **10B-A1B MoE**. [Jonas Geiping et al.](https://x.com/jonasgeiping/status/2054600427128201688) argued current message-based/chat training overly constrains agents to a single stream and released a **multi-stream LLM** paper claiming lower latency, cleaner separation of concerns, and more legible parallel reasoning/tool use; paper and code are linked [here](https://x.com/jonasgeiping/status/2054600457746579816). [δ-mem](https://x.com/dair_ai/status/2054600147020222630) proposed an external online associative memory attached to a frozen full-attention backbone, with an **8×8 state** reportedly improving average score by **1.10×** and beating non-δ-mem baselines by **1.15×**, with larger gains on memory-heavy benchmarks.
- **Post-training/compression and data curation also produced notable results**: NVIDIA’s [Star Elastic](https://x.com/PavloMolchanov/status/2054607257166553292) claims one post-training run can derive a family of reasoning model sizes, at **360× lower cost than pretraining a family** and **7× better than SOTA compression**. Datology’s VLM work, highlighted by [Siddharth Joshi](https://x.com/sjoshi804/status/2054566179369574419) and [Pratyush Maini](https://x.com/pratyushmaini/status/2054607891202777192), argues **data curation alone** can produce major multimodal gains: **+11.7 points across 20 public VLM benchmarks at 2B**, beating InternVL3.5-2B by roughly **10 points** at about **17× less training compute**, and near-frontier 4B performance with **3.3× lower response FLOPs** than Qwen3-VL-4B. On the open data side, [Percy Liang](https://x.com/percyliang/status/2054550981527146942) said the next **Marin** run already has **18T tokens** in its mix and is still seeking more pretraining, mid-training, and SFT data, with a companion token viewer [shared here](https://x.com/percyliang/status/2054550984597328101).
- **Open evaluation and dataset work is maturing alongside model building**: [Kevin Li’s SWE-ZERO-12M-trajectories](https://x.com/kevin_x_li/status/2054600962137100493) is positioned as the largest open agentic trace dataset: **112B tokens, 12M trajectories, 122K PRs, 3K repos, 16 languages**. [Victor Mustar](https://x.com/victormustar/status/2054495700822478943) flagged **llama-eval** as a step toward more comparable llama.cpp community evals. Meanwhile, [Steve Rabinovich](https://x.com/steverab/status/2054564579573698921) and [Sayash Kapoor](https://x.com/sayashk/status/2054569643080077576) argued credible agent evaluation requires **log analysis**, not outcome-only metrics, because stronger agents expose hidden benchmark bugs and reward-hacking paths.

**Enterprise AI Pricing, Platform Competition, and Distribution**

- **Anthropic vs OpenAI competition sharpened around enterprise distribution and developer lock-in**: [Ramp data cited by Andrew Curran](https://x.com/AndrewCurran_/status/2054582686698848294) showed **Anthropic at 34.4%** of businesses vs **OpenAI at 32.3%** in April, the first apparent lead change in business adoption; [The Rundown](https://x.com/TheRundownAI/status/2054588969044627906) amplified the same figures. At the same time, Anthropic changed plan economics: [ClaudeDevs announced](https://x.com/ClaudeDevs/status/2054610152817619388) that paid Claude plans will get a dedicated monthly credit for programmatic usage across the **Agent SDK**, `claude -p`, GitHub Actions, and third-party SDK apps. This was immediately read by power users as a major restriction on subscription-subsidized harnesses, with criticism from [Theo](https://x.com/theo/status/2054620998205624746), [Jeremy Howard](https://x.com/jeremyphoward/status/2054682882753597603), [Matt Pocock](https://x.com/mattpocockuk/status/2054655310388674693), and [Omar Sanseviero](https://x.com/omarsar0/status/2054679776397300188). Anthropic partially offset that backlash with a separate [50% increase in Claude Code weekly limits](https://x.com/ClaudeDevs/status/2054639777685934564) through July 13, stacked on the previously announced 2× 5-hour limit increase.
- **OpenAI responded aggressively with Codex enterprise incentives**: [OpenAI Devs](https://x.com/OpenAIDevs/status/2054586214112780518) and [Sam Altman](https://x.com/sama/status/2054626219858293128) offered **two months of free Codex usage** for enterprise customers switching in the next 30 days. OpenAI also published more technical platform detail, including a [Windows sandbox design write-up](https://x.com/reach_vb/status/2054655421013434510) describing the combination of local users, firewall rules, ACLs, write-restricted tokens, DPAPI, and helper executables needed to safely run coding agents with local filesystem/tool access. The competitive dynamic now looks less like “best model wins” and more like **subsidy + workflow control + harness compatibility**.
- **Enterprise adoption is increasingly tied to runtime/security assurances**: [Perplexity](https://x.com/perplexity_ai/status/2054608966148374715) described a hardware-isolated sandbox architecture with VPC-level separation, short-lived proxy tokens, and scanning of external content before agent actions, with [additional details](https://x.com/perplexity_ai/status/2054608978680873457) on encryption and auto-deletion. [Aravind Srinivas](https://x.com/AravSrinivas/status/2054619058650411174) framed this as foundational to Perplexity becoming an enterprise knowledge/research platform. The broader pattern: agent vendors are no longer selling only intelligence; they’re selling **bounded execution environments**.

**Autonomous Science, Cyber Capability, and Robotics**

- **Recursive self-improvement moved from idea to startup cluster**: The largest single meta-theme was the launch of [Recursive](https://x.com/_rockt/status/2054491251345391852), founded to build AI that automates science and safely improves itself. Launch posts from [Richard Socher](https://x.com/_rockt/status/2054491251345391852), [Josh Tobin](https://x.com/josh_tobin_/status/2054576051431616873), [Dominik Schmidt](https://x.com/schmidtdominik_/status/2054498117416808727), [Jenny Zhang](https://x.com/jennyzhangzt/status/2054603211798147436), and [Shengran Hu](https://x.com/shengranhu/status/2054630820305088739) suggest a team drawn from open-endedness, AI Scientist, and research automation work. In adjacent work, [Adaption’s AutoScientist](https://x.com/adaption_ai/status/2054532113316434061) aims to automate the full training-research loop outside frontier labs, with [Sarah Hooker](https://x.com/sarahookr/status/2054551263275254084) arguing that most model training failures are due to research-loop brittleness rather than mere compute scarcity.
- **Cyber capability evaluations continue to steepen**: The UK [AI Security Institute](https://x.com/AISecurityInst/status/2054589758043496567) said the length of cyber tasks frontier models can complete has been doubling every few months, and that recent models are beating prior trends. Anthropic/Glasswing’s [Logan Graham](https://x.com/logangraham/status/2054613618168082935) said **Claude Mythos Preview** is the first model to solve both AISI end-to-end cyber ranges, including **Cooling Tower**, and the only one to clear every task under the institute’s **2.5M-token** cap. XBOW reportedly found “token-for-token, unprecedented precision,” and partner usage allegedly surfaced **thousands of high/critical vulnerabilities** in weeks. Independent commentary from [scaling01](https://x.com/scaling01/status/2054594892903436553) claimed a newer Mythos version completed a cyber range **6/10 times vs 3/10** for the preview baseline.
- **Robotics got a concrete long-horizon deployment demo**: [Figure’s Brett Adcock](https://x.com/adcock_brett/status/2054603963996278786) streamed humanoid robots running a full **8-hour autonomous shift** on package sorting using **Helix-02**, with follow-up details that the robots reason from camera pixels, operate around **human parity (~3s/package)**, perform **on-device inference**, coordinate as a networked fleet, autonomously swap for low battery, and self-diagnose/fail over to maintenance when needed [here](https://x.com/adcock_brett/status/2054615837903048807). This is one of the clearer public demonstrations of **multi-robot, long-duration, no-human-in-the-loop orchestration** rather than a short benchmark clip.

**Top tweets (by engagement)**

- **Claude Code pricing and limits**: [@ClaudeDevs on 50% higher weekly limits](https://x.com/ClaudeDevs/status/2054639777685934564), [@ClaudeDevs on programmatic credits](https://x.com/ClaudeDevs/status/2054610152817619388), and the ensuing developer backlash from [@theo](https://x.com/theo/status/2054620998205624746) made pricing policy the day’s most consequential developer story.
- **Codex enterprise push**: [@sama offering two free months of Codex usage for switchers](https://x.com/sama/status/2054626219858293128) and [@OpenAIDevs’ enterprise call-to-action](https://x.com/OpenAIDevs/status/2054586214112780518) signaled an unusually direct go-to-market counterpunch.
- **Figure’s 8-hour humanoid shift**: [@adcock_brett’s livestream post](https://x.com/adcock_brett/status/2054603963996278786) drew enormous attention and is one of the few viral posts in the set with clear technical substance.
- **Cline SDK launch**: [@cline’s SDK release](https://x.com/cline/status/2054580767779700775) was one of the highest-engagement genuinely technical launches, reflecting demand for open coding-agent harnesses.
- **Token Superposition Training**: [@NousResearch’s TST post](https://x.com/NousResearch/status/2054610062836892054) stood out as a rare pretraining-method tweet that broke through widely, likely because the claim—**2–3× training speedup without changing inference-time architecture**—is concrete and economically important.


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Efficient On-Device LLM Inference

  - **[Needle: We Distilled Gemini Tool Calling Into a 26M Model](https://www.reddit.com/r/LocalLLaMA/comments/1tb9b0r/needle_we_distilled_gemini_tool_calling_into_a/)** (Activity: 451): ****Cactus Compute** open-sourced **Needle**, a `26M`-parameter single-shot function/tool-calling model using a “Simple Attention Network” architecture—attention + gating with **no FFNs/MLPs**—arguing tool use is mainly retrieval/slot extraction/JSON assembly rather than deep reasoning. It was pretrained on `200B` tokens over `16 TPU v6e` in `27h`, post-trained on `2B` Gemini-synthesized function-calling tokens in `45m`, claims `6000 tok/s` prefill and `1200 tok/s` decode on consumer devices, and reportedly beats FunctionGemma-270M, Qwen-0.6B, Granite-350M, and LFM2.5-350M on single-shot function calling; code/weights are MIT-licensed on [GitHub](https://github.com/cactus-compute/needle), [Hugging Face](https://huggingface.co/Cactus-Compute/needle), with architecture notes in the [SAN writeup](https://github.com/cactus-compute/needle/blob/main/docs/simple_attention_networks.md).** Commenters framed Needle as potentially useful as a lightweight router that selects tools or dispatches queries to larger LLMs with parameters, while questioning whether the same no-FFN/cross-attention approach could generalize to summarization. One technical caution noted the repository apparently includes Python `pickle` files, which are discouraged due to code-execution/security risks and Python-specific portability issues.

    - Several commenters focused on the architectural implication of a **26M distilled tool-calling model** as a lightweight router: it could classify/route requests to the appropriate larger LLM, tool, or RAG pipeline with the right parameters, rather than generating full answers itself. One suggested this could be extended into a small post-trained model that consumes structured RAG output and verbalizes it in natural language.
    - A technical point was raised around the claimed “**no FFN**” result: if external structured knowledge is always supplied via tools/RAG/retrieval, the model may not need FFN layers to store factual knowledge in weights. This implies a possible design pattern where compact attention-heavy models specialize in orchestration or grounding over provided context instead of memorization.
    - One commenter noted that publishing **pickle files** is increasingly uncommon because of Python-specific dependency coupling and arbitrary-code-execution risks during deserialization. Another highlighted that **Gemini** has had visible tool-calling quirks, including system-prompt-level patches around tool specificity and avoiding inefficient file operations like `cat` in favor of dedicated tools such as `grep_search`, which could matter if Gemini-generated traces were used as distillation data.

  - **[I got a real transformer language model running locally on a stock Game Boy Color!](https://www.reddit.com/r/LocalLLaMA/comments/1tbi2n3/i_got_a_real_transformer_language_model_running/)** (Activity: 1326): **The image shows a stock **Game Boy Color** running a local transformer demo labeled `TINYSTORIES Q8 GBC`, matching the post’s claim that **Andrej Karpathy’s TinyStories-260K** was converted to `INT8`/fixed-point and executed directly on-device without PC, Wi‑Fi, link cable, or cloud inference: [image](https://i.redd.it/1hl9id7ghs0h1.jpeg). The project uses **GBDK-2020**, an **MBC5 Game Boy ROM**, bank-switched cartridge ROM for weights, cartridge SRAM for the KV cache, and on-device tokenization/prompt entry; the author notes generation is *extremely slow* and mostly gibberish due to heavy quantization/approximation, but the transformer prefill + autoregressive loop works. Source code: [github.com/maddiedreese/gbc-transformer](https://github.com/maddiedreese/gbc-transformer).** Comments are mostly impressed rather than technical, framing it as an impractical but compelling proof-of-concept—e.g. *“Pointless. Therefore, indispensable.”* and interest in porting similar experiments to other retro hardware like the N64.

    - A commenter references a related prior project, **GBALM**, linking to [`https://code.heni.lol/heni/gbalm`](https://code.heni.lol/heni/gbalm). The comment does not provide implementation details, but the link may be relevant for readers comparing other attempts at running language-model-like systems on Game Boy-class hardware.

  - **[Solar Powered Qwen 3.6 Server](https://www.reddit.com/r/LocalLLM/comments/1tbfcfe/solar_powered_qwen_36_server/)** (Activity: 449): **A user reports running a local **[Qwen](https://qwenlm.github.io/) 27B [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)** model build from **[Unsloth](https://unsloth.ai/)**, `UD-Q4_K_XL`, with `100k` context on an **M1 Max 32GB**, achieving roughly `~10 tok/s`. The inference server is powered by `3 × 100 W` solar panels feeding an **Anker `1.25 kW` all-in-one power unit**; observed power draw is `~80–85 W` under inference load, sometimes dropping to `~30 W`, with idle draw `≤5 W`. The user says performance is “really good” in **Hermes** and **opencode** workflows.** Commenters mainly highlighted the practicality of Apple Silicon for off-grid inference due to low power draw, with one noting that non-Mac solutions would drain batteries too quickly and that winter operation is challenging for fully off-grid setups, especially in northern climates.

    - One technically relevant thread notes that an **off-grid whole-house power setup** constrains hardware choice: the commenter uses **Macs** because alternative server/GPU solutions would drain battery capacity too quickly. They also highlight seasonal reliability issues for solar/off-grid compute, saying **winter near the Baltic** is difficult enough that they plan to move to a **hybrid power setup**.

  - **[Stop wasting electricity](https://www.reddit.com/r/LocalLLaMA/comments/1tayu5t/stop_wasting_electricity/)** (Activity: 1104): **A user reports that running [`llama.cpp`](https://github.com/ggerganov/llama.cpp) `llama-server` on an **RTX 4090** with `Qwen3.6-27B-UD-Q4_K_XL.gguf`, `--flash-attn on`, `-ngl all`, `-ctk q4_0 -ctv q4_0`, and `-c 262144` remains GPU power-limit-bound under `nvidia-smi -pl N`, implying actual board power tracks the configured cap. Their observation is that reducing the GPU power limit can cut consumption to roughly **40%** without materially hurting **decode/token-generation throughput**, while also reducing heat/noise; a commenter adds that **prefill** is more sensitive but reportedly only drops about `15–20%` when reducing from `450W` to `270W`, depending on model.** Commenters push for separating **prefill/prompt-processing** from **decode** benchmarks, since decode throughput may hide power-limit-induced regressions. Another user notes they already power-cap an RTX 5090 due to connector/thermal concerns and may lower the cap further based on these results.

    - Users discussed GPU power limiting for local inference, specifically that reducing an RTX 5090 from `450W` to `270W` reportedly has little impact on decode/token generation (`tg`) throughput, while prefill (`pp`) performance drops more noticeably but only around `15–20%` depending on the model. This suggests a potentially favorable efficiency tradeoff for inference workloads where decode dominates runtime.
    - One commenter noted capping a `5090` due to concerns about connector or hardware overheating, while another mentioned heavily power-limiting `3090s` to reduce noise for overnight operation. The technical implication is that aggressive power caps may materially improve thermals/acoustics and power efficiency without proportionally reducing LLM inference throughput, especially during decode-heavy workloads.


### 2. Open-Source Local Agent Interfaces

  - **[TextGen is now a native desktop app. Open-source alternative to LM Studio (formerly text-generation-webui).](https://www.reddit.com/r/LocalLLaMA/comments/1tbyyee/textgen_is_now_a_native_desktop_app_opensource/)** (Activity: 795): ****oobabooga/TextGen** has been refactored from `text-generation-webui` into a **portable, no-install Electron desktop app** for Windows/Linux/macOS, with self-contained `user_data` storage and builds for **CUDA, Vulkan, CPU-only, ROCm, and Apple Silicon/Intel macOS** via the [GitHub releases](https://github.com/oobabooga/textgen/releases). The app positions itself as an open-source **LM Studio** alternative with **zero outbound requests**, `ik_llama.cpp` support for newer quant types like `IQ4_KS`/`IQ5_KS`, built-in web search via `ddgs`, Python/HTTP/stdio MCP tool calling with approval gates, OpenAI/Anthropic-compatible APIs including Claude Code support, PDF extraction via `PyMuPDF`, web cleanup via `trafilatura`, and Jinja2 chat-template rendering; source is AGPLv3 at [oobabooga/textgen](https://github.com/oobabooga/textgen).** Top comments are mostly enthusiastic rather than technical, emphasizing recognition of **oobabooga** and demand for a more private, open alternative to **LM Studio**.

    - A commenter framed the project as filling a gap for an **open-source, private native desktop alternative to LM Studio**, contrasting it with prior local LLM UX options that were often web UI–centric rather than packaged app workflows.
    - One technical observation noted that after using text-generation-webui, they realized much of the local LLM ecosystem converges around an **OpenAI-compatible API**, implying that frontends and tooling can often be swapped as long as they target that API surface.

  - **[Let's build claude code from scratch!](https://www.reddit.com/r/LocalLLaMA/comments/1tb6nkx/lets_build_claude_code_from_scratch/)** (Activity: 462): **The image is a **technical terminal screenshot** (not a meme) showing a custom CLI coding agent branded as **“NANO CLAUDE”** in `~/projects/nano-claude`, described as *“Claude Code · from scratch”* and prompting the user to enter a coding request. The post links a build-from-scratch tutorial video and GitHub repo for the implementation: [YouTube](https://youtu.be/8pDfgBEy8bg), [GitHub](https://github.com/CohleM/nanoclaude), and the screenshot is available [here](https://i.redd.it/ass571o3gq0h1.png).** Commenters mainly warned that using **“Claude”** in the project name may create trademark risk with Anthropic, citing prior renaming pressure around OpenClaw/Clawdbot. Others suggested similar tools already exist, such as `opencode`, or pointed to Pi as an alternative.

    - One commenter argued that reimplementing a Claude Code-like agent is valuable for understanding the underlying **agent/tool loop**, since many users rely on these tools without understanding how model calls, tool invocation, and iterative execution are orchestrated under the hood.
    - Another commenter pointed to **opencode** as an existing implementation in this space, implying that similar Claude Code-style coding agents already exist and may be useful as references before starting a from-scratch build.




## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Real-World AI Agent Failure Modes

  - **[Inherited a 3-month old repo from a Vibe Engineer. Wrote the most satisfying PR in my career](https://www.reddit.com/r/ClaudeCode/comments/1tb7edc/inherited_a_3month_old_repo_from_a_vibe_engineer/)** (Activity: 6187): **The [image](https://i.redd.it/izgrhw5tgq0h1.png) shows an extreme PR diff of **`+10,197` additions** and **`−3,618,778` deletions**, contextualizing the post’s claim that a 3-month-old backend repo produced via “agentic”/vibe coding had accumulated massive generated or unnecessary code, docs, logs, secrets, and unused handlers. The author says they rewrote the repo in a week with Claude, preserving functionality while replacing a bloated architecture—`309k` LOC, `240k` docs, `1M+` markdown log lines, `220` handlers with only ~`20` used, and `40+` secrets with only `2` needed—with a cleaner backend and integration tests.** The comments shown are mostly non-technical jokes around the term “vibe engineer” and the irony of using AI-assisted coding to clean up an AI-generated codebase; there is no substantive technical debate in the provided top comments.

    - Several commenters framed the repo as an example of **AI/agent-generated technical debt**, suggesting that “fixing vibe-coded mess” may become a lucrative maintenance niche as teams inherit code produced without conventional engineering discipline. The discussion also notes a credibility gap: praise for “agentic approaches” often comes from people who are not software professionals, implying that generated code may look impressive while still requiring significant human refactoring, deletion, and validation.

  - **[I made an AI concierge for my wedding guests. The second most popular thing they did with it was try to jailbreak it.](https://www.reddit.com/r/ClaudeAI/comments/1tatxnq/i_made_an_ai_concierge_for_my_wedding_guests_the/)** (Activity: 2003): **The image is an illustrated usage report for a custom wedding **AI concierge** (“Aido”) built for a destination wedding in **Mauritius**, reportedly connected to wedding/travel info via an API/MCP server. It shows `719` sessions, `8,678` messages, and `29` users, with the largest categories being **sincere logistics** (`35%`) and **jailbreak/hack attempts** (`25%`), highlighting that even low-stakes private assistants attract adversarial prompting. Image: [AI Concierge Report Card](https://i.imgur.com/8n0k4Ve.jpeg).** Commenters found the project more interesting than a generic chatbot, but were surprised by the engagement volume—over `8,000` messages from only `29` users—and amused that jailbreak attempts were the second-largest use case.

    - The OP described building a two-part system: first a **wedding planning assistant** for a destination wedding in **Mauritius**, then a guest-facing **AI concierge** connected to an API through an `MCP server` so it could retrieve event/travel information dynamically for users.
    - One commenter highlighted the usage volume as notable for a small deployment: only `29` users generated **over `8,000` messages**, implying unusually high engagement and/or repeated probing such as jailbreak attempts.
    - A privacy concern was raised around observability and message logging: a commenter asked whether guests were uncomfortable with the OP being able to read their interactions, which is relevant for any personal-event chatbot that stores or inspects user messages.


# AI Discords

Unfortunately, Discord shut down our access today. We will not bring it back in this form but we will be shipping the new AINews soon. Thanks for reading to here, it was a good run.