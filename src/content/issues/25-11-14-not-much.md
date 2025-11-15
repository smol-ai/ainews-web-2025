---
id: MjAyNS0x
title: not much happened today
date: '2025-11-14T05:44:39.731046Z'
description: Unable to generate summary due to an error.
companies: []
models: []
topics:
  - untagged
people: []
---


**wen Gemini 3?**

> AI News for 11/13/2025-11/14/2025. We checked 12 subreddits, 544 Twitters and 24 Discords (205 channels, and 6489 messages) for you. Estimated reading time saved (at 200wpm): 514 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

lots of [hints](https://x.com/swyx/status/1989227065732980981) of Gemini 3. I wonder what GDM is planning for [their big opener for AIE CODE](https://www.ai.engineer/code)...

(P.S. tickets are sold out, but you can attend [side events](https://x.com/swyx/status/1989461220131574013) or [volunteer](https://docs.google.com/forms/d/e/1FAIpQLSf-vO_ANN96lZ5myPM80si6_kfMfCy68VEtpomRnt6N_r12Iw/viewform) for free).

---

# AI Twitter Recap

**OpenAI’s GPT‑5.1 rollout: adaptive reasoning, dev-focused API, and new UX pilots**

- **Adaptive reasoning + developer focus**: OpenAI shipped GPT‑5.1 with “adaptive reasoning” for 5.1‑Instant, a change that required novel post-training/RL work, per [@allisontam_](https://twitter.com/allisontam_/status/1989138927970848936). On the platform side, OpenAI emphasized better dev ergonomics: “excellent new models in our API and extended prompt caching,” per [@gdb](https://twitter.com/gdb/status/1989135114744573993). The team also published concrete guidance for long-running/agentic tasks (plan tools, persistence), and a Prompt Optimizer, including a toggle for reasoning_effort to trade off latency/cost vs quality, via [OpenAI DevRel](https://twitter.com/OpenAIDevs/status/1989378869976326570), [tip 2](https://twitter.com/OpenAIDevs/status/1989378875126886560), [tip 3](https://twitter.com/OpenAIDevs/status/1989378876922077560). Separately, OpenAI is piloting “group chats” with ChatGPT in Japan, NZ, South Korea, and Taiwan ([announcement](https://twitter.com/OpenAI/status/1989138776585851038)).
- **Measured deltas on benchmarks and UX**: Independent tracker Artificial Analysis finds GPT‑5.1 is a “minor” intelligence bump over GPT‑5 (+2 on their index), driven largely by a +12pp gain on TerminalBench (agentic coding/terminal use). Non‑reasoning endpoint shows no change; pricing is unchanged and slightly more token‑efficient, per [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1989417492582899872). Anecdotally, GPT‑5.1 is better at creative writing ([@gdb](https://twitter.com/gdb/status/1989230190111912041)) and code (try 5.1‑Codex: [@gdb](https://twitter.com/gdb/status/1989226363069559173); live tests in VS Code: [@JamesMontemagno](https://twitter.com/JamesMontemagno/status/1989354367343075697)). Steerability also improved in small but visible ways (e.g., respecting custom instruction style constraints), per [@sama](https://twitter.com/sama/status/1989193813043069219).

**Anthropic/Claude: structured outputs land; Claude Code momentum; Opus 4.5 rumor debunk**

- **Structured Outputs (public beta)**: Claude API now guarantees JSON schema/tool-conformant responses without retries/parsing hacks, initially for Sonnet 4.5 and Opus 4.1 (Haiku 4.5 soon). Docs and API examples via [@alexalbert__](https://twitter.com/alexalbert__/status/1989409186674098595) and [follow‑up](https://twitter.com/alexalbert__/status/1989409198971801905). Long‑time users note this unifies prior “single tool with schema” recipes ([@simonw](https://twitter.com/simonw/status/1989411809351303430)).
- **Claude Code for agent ops**: Builders highlight Claude Code’s effectiveness at optimizing agent workflows when given logs/test scripts and a tight harness—less tooling, more code execution and clear specs ([@omarsar0](https://twitter.com/omarsar0/status/1989417433245925645); “Pulse” proactive research agent: [demo](https://twitter.com/omarsar0/status/1989350215175020682)). Community sentiment remains that Opus 4.1 is still hard to beat for general quality ([@abacaj](https://twitter.com/abacaj/status/1989128220835537315)).
- **Rumor control**: “Opus 4.5” references found in a Claude Code CLI PR appear to be autocomplete/date artifacts, not a stealth release ([@scaling01](https://twitter.com/scaling01/status/1989145846114578547), [1](https://twitter.com/scaling01/status/1989145863508394059), [2](https://twitter.com/scaling01/status/1989146991272817048)). On policy domains, Anthropic reiterates norms‑based approaches for fair handling of political topics ([@AmandaAskell](https://twitter.com/AmandaAskell/status/1989328363077382407)).

**Agents, protocols, and toolchains: Deep Agents, MCP’s birthday, and safer autonomy UX**

- **Deep Agents + context engineering**: LangChain released a “Deep Agents” package/CLI and shared a context-engineering playbook: offload/reduce/isolate patterns, eval harnesses, and why products re‑architect around tomorrow’s models ([overview](https://twitter.com/LangChainAI/status/1989152093127782765), [podcast](https://twitter.com/jakebroekhuizen/status/1989130283866812437)).
- **Interoperability via protocols**: A succinct landscape from CopilotKit argues three complementary open protocols are standardizing agent stacks across LangGraph/CrewAI/Agno: AG‑UI (agent↔user), MCP (context/tooling), and A2A (agent↔agent), enabling mix‑and‑match without UI rewrites ([@_avichawla](https://twitter.com/_avichawla/status/1989228336997101946), [PDF](https://twitter.com/_avichawla/status/1989228348971893236)).
- **MCP at 1**: Anthropic and Gradio kicked off MCP’s first‑birthday hackathon; >6,300 registrants, 2 weeks, $20k prizes + $3.5M credits ([@Gradio](https://twitter.com/Gradio/status/1989315723336749412), [@huggingface](https://twitter.com/huggingface/status/1989386669636948321)).
- **Trust and control in agent UX**: Perplexity’s Comet Assistant now pauses before sensitive actions (logins/purchases), shows traces, and requires explicit permission—an example of permissioned “digital employee” patterns for agentic browsing ([@perplexity_ai](https://twitter.com/perplexity_ai/status/1989416343331012971)).
- **Agentic coding platforms**: Cline added direct support for Nous’ Hermes 4 70B/405B across VS Code, JetBrains, and CLI ([@NousResearch](https://twitter.com/NousResearch/status/1989427241424654534); [@cline](https://twitter.com/cline/status/1989432694867193988)).

**Dev tools and infra: repo QA, coding agents, and experiment plumbing**

- **Repo‑aware code QA**: Google’s “CodeWiki” lets you query codebases in natural language (function locations, architecture, logic) and performed well on the DSPy repo in community tests ([@getpy](https://twitter.com/getpy/status/1989237111745310770)).
- **Qwen Code v0.2.1**: Fast iteration (8 releases in 17 days) delivering: free web search (multi‑provider; 2,000/day for OAuth users), fuzzy matching to reduce retries/token usage, plain‑text tool responses (less schema brittleness), Zed IDE improvements, smarter file filtering and `.gitignore` respect, Unicode fixes, and performance optimizations ([@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1989368317011009901)).
- **Experiment ops**: SkyPilot now natively integrates with Weights & Biases for multi‑cloud/K8s launches, failure recovery, and collaborative tracking ([@skypilot_org](https://twitter.com/skypilot_org/status/1989377870469501106)); W&B also shipped LEET, a terminal UI for runs/metrics/system health ([clip](https://twitter.com/wandb/status/1989403717305827660)).
- **IDE/runtime niceties**: VS Code gaining inline terminal output for failed commands in chat flows ([@Tyriar](https://twitter.com/Tyriar/status/1989439441971396952)); Colab↔VS Code integration momentum visible at events ([@patloeber](https://twitter.com/patloeber/status/1989332433301324031)). On the infra side, Dojo now supports the OpenEnv spec ([@chakra_ai](https://twitter.com/chakra_ai/status/1989377867965513880)), and engineers continue reporting solid speedups with CUDA Graphs (with caveats) while cautioning against misuse of `record_stream` ([@maharshii](https://twitter.com/maharshii/status/1989375005231362428), [@vikhyatk](https://twitter.com/vikhyatk/status/1989217613873021241)).

**Research and evals to watch: depth, audio‑language, interpretability, and on‑policy training**

- **Depth Anything 3 (DA3)**: Aiming for human‑like spatial perception across single/multi‑view and video using minimalism: a plain transformer (e.g., DINO) and a single depth‑ray representation suffice for broad generalization. Multiple series released (main DA3, monocular metric, monocular depth). App/demo and paper threads here: [@bingyikang](https://twitter.com/bingyikang/status/1989358267668336841), [@_akhaliq](https://twitter.com/_akhaliq/status/1989336687529619858).
- **Music Flamingo**: NVIDIA’s large audio‑language model family targeting music and song understanding, with project page and HF assets linked ([@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1989273704057151966)).
- **SWE/ML optimization leaderboard**: Early results suggest all current LLMs slow down expert humans on ML/HPC optimization tasks when measured by runtime, providing a needed baseline for “effective coding” claims ([@scaling01](https://twitter.com/scaling01/status/1989338806575903109)).
- **Mechanistic interpretability at the frontier**: [@NeelNanda5](https://twitter.com/NeelNanda5/status/1989297683140354267) outlines where interp can have impact now (less reverse‑engineering toy models, more work on frontier systems), common pitfalls, and promising directions. A complementary line from TransluceAI shows models can learn to self‑explain internal features more faithfully than other models do ([paper + blog](https://twitter.com/TransluceAI/status/1989395421236793374)).
- **Efficient domain FMs**: SophontAI’s OpenMidnight is a pathology FM trained on 12k WSI for ~$1.6k compute, achieving SOTA on a public dataset—code, models, and demos are open ([@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1989390268316221861)).
- Related: Black‑box on‑policy distillation and rubric‑based RL for instruction following continue to push post‑training reliability and eval design ([distillation](https://twitter.com/_akhaliq/status/1989341114760126965); [rubric RL](https://twitter.com/iScienceLuvr/status/1989274582822592634)).

**Ecosystem signals: Gemini 3 rumors, video model upgrades, Grok‑5 claims, and AI Dev 25**

- **Google**: Multiple signals point to a Gemini 3 rollout imminently ([@swyx](https://twitter.com/swyx/status/1989227065732980981)). The Gemini app updated Veo 3.1 to accept multiple reference images for more controllable video generation ([@GeminiApp](https://twitter.com/GeminiApp/status/1989440642179801192)). Google also highlighted new Photos AI features ([@Google](https://twitter.com/Google/status/1989468389480501458)) and announced a $40B Texas investment through 2027 for Cloud/AI infra and workforce pipelines ([@sundarpichai](https://twitter.com/sundarpichai/status/1989468970400055487)).
- **xAI/Grok**: Elon Musk claims Grok‑5 will be a 6T‑parameter multimodal MoE, targeted for Q1 2026 and “the smartest” system by a wide margin; Grok‑3/4 were 3T total params. No active‑parameter counts disclosed; expectations of increased sparsity noted by observers ([summary](https://twitter.com/scaling01/status/1989457860728647928)).
- **AI Dev 25 (NYC)**: Highlights included Andrew Ng on faster iteration and feedback as the bottleneck ([@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1989400305356697856)), Groq’s demo of low‑latency compound systems for deep‑research agents ([recap](https://twitter.com/DeepLearningAI/status/1989431887224275433)), and SAP on why agents fail in enterprises (API selection and business‑process context), advocating knowledge graphs for semantics ([session](https://twitter.com/DeepLearningAI/status/1989397092570104010)). Google’s Robert Crowe introduced Flax NNX to simplify JAX model building and distribution ([talk](https://twitter.com/DeepLearningAI/status/1989453390393278607)).

**Top tweets (by engagement)**

- [Sam Altman: “Small‑but‑happy win” — ChatGPT follows style instructions like “no em‑dashes”](https://twitter.com/sama/status/1989193813043069219) — 30.5k
- [Jeff Bezos on a major milestone (video)](https://twitter.com/JeffBezos/status/1989405079594848719) — 28.4k
- [Yann LeCun: “You’re being played… regulatory capture to kill open source.”](https://twitter.com/ylecun/status/1989364612651966788) — 4.0k
- [François Chollet: “All breakthroughs are symbolic compression.”](https://twitter.com/fchollet/status/1989340153114976598) — 4.6k
- [SwiftOnSecurity: Full‑text search was solved in 2001; product choices broke it.](https://twitter.com/SwiftOnSecurity/status/1989130339458126281) — 6.5k
- [“This will be the opening credits for 2020: The Movie.”](https://twitter.com/growing_daniel/status/1989189599093240060) — 6.6k
- [Joyce Vance: On norms at DOJ](https://twitter.com/JoyceWhiteVance/status/1989416956404052097) — 5.4k
- [HeightOptimized: Squats vs leg press hormones stat](https://twitter.com/HeightOptimized/status/1989190171041108065) — 6.5k

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Performance Comparison of Llama.cpp on Windows vs Linux

- [**Windows llama.cpp is 20% faster**](https://www.reddit.com/r/LocalLLaMA/comments/1owskm6/windows_llamacpp_is_20_faster/) (Activity: 363): **The post discusses a performance comparison between Windows and Linux when running** `llama.cpp`**, specifically using the Qwen3-VL-30B-A3B-Instruct model. The benchmark results show that Windows, utilizing the AMD proprietary driver, achieves higher performance across various parameters (pp512, pp1024, pp2048, pp4096) compared to Linux, which uses the RADV driver. A key factor contributing to this performance difference is the support for** `bf16` **(bfloat16) on Windows, which is not yet available on Linux. Additionally, the shared memory size is detected differently between the two systems, potentially affecting performance.** Commenters highlight the significance of `bf16` support in the performance disparity and note that Linux's lack of `bf16` support is a disadvantage. There is also a discussion about the shared memory size detection and its impact on performance, with a link to a potential future update for Linux drivers.
    - The introduction of `bf16` support is highlighted as a significant factor in the performance improvement of `llama.cpp` on Windows. This support is expected to be available on more machines next year, as indicated by the [Phoronix article](https://www.phoronix.com/news/AMD-BF16-For-LLVM-SPIR-V).
    - A technical point is raised about the detection of shared memory size, which is handled differently on Windows compared to Linux. Additionally, it's noted that the Linux driver currently lacks `bf16` support, which could impact performance.
    - A user questions the choice of using Vulkan over `hipBLAS`, suggesting that `hipBLAS` could offer higher performance on both Windows and Linux platforms. This implies a potential area for optimization in the implementation of `llama.cpp`.
- [**Is it normal to hear weird noises when running an LLM on 4× Pro 6000 Max-Q cards?**](https://www.reddit.com/r/LocalLLaMA/comments/1owocd2/is_it_normal_to_hear_weird_noises_when_running_an/) (Activity: 873): **Running a large language model (LLM) like** `gpt-oss-120b` **on multiple high-performance GPUs such as** `4× Pro 6000 Max-Q` **can lead to unusual noises, often attributed to coil whine. This phenomenon occurs due to the electrical components vibrating at certain frequencies under heavy load, which can vary with different models and workloads. The noise is generally harmless but can be more pronounced with specific hardware configurations and computational tasks.** Commenters generally agree that the noise is likely coil whine, a common occurrence with high-performance GPUs under load. Some users note that the sound can vary between setups and even find it somewhat enjoyable.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. ChatGPT Custom Instructions Update

- [**ChatGPT finally fixed the one thing everyone complained about.**](https://www.reddit.com/r/OpenAI/comments/1owrswl/chatgpt_finally_fixed_the_one_thing_everyone/) (Activity: 1502): **The image is a meme highlighting a recent update to ChatGPT that allows users to customize its behavior by instructing it not to use em-dashes. This update is seen as a minor but significant improvement, as it addresses a common user complaint about the AI's writing style. The change is celebrated as a 'small-but-happy win,' indicating that users appreciate the increased control over the AI's output.** One comment humorously questions if this update signifies the arrival of AGI, while another sarcastically notes that the ability to use em-dashes was a distinguishing feature between bots and humans, suggesting this change might blur those lines.
- [**« If you tell ChatGPT not to use em-dashes in your custom instructions, it finally does what it's supposed to do! »**](https://www.reddit.com/r/ChatGPT/comments/1owob2f/if_you_tell_chatgpt_not_to_use_emdashes_in_your/) (Activity: 6177): **The image is a meme highlighting a humorous interaction with ChatGPT, where the user successfully instructs the AI to avoid using em-dashes in its responses. This reflects a minor but specific customization capability of ChatGPT, showcasing its ability to adhere to user-defined stylistic preferences. The post is lighthearted and does not delve into technical details about the AI's functionality or architecture.** The comments reflect a humorous take on the situation, with one user sarcastically suggesting that this minor feature justifies a high valuation of AI technology, while another acknowledges Sam Altman's expertise in AI.

### 2. AI-Driven Cybersecurity Breach

- [**China just used Claude to hack 30 companies. The AI did 90% of the work. Anthropic caught them and is telling everyone how they did it.**](https://www.reddit.com/r/ClaudeAI/comments/1ox361v/china_just_used_claude_to_hack_30_companies_the/) (Activity: 901): **In September 2025, Anthropic identified a cyberattack orchestrated by Chinese state-sponsored hackers using their AI, Claude. The attack targeted approximately** `30` **companies, including tech firms, banks, and government agencies, with the AI performing** `80-90%` **of the hacking tasks autonomously. The hackers bypassed Claude's safety protocols by breaking the attack into innocuous tasks and misleading the AI into believing it was conducting legitimate cybersecurity testing. This incident is noted as the first large-scale cyberattack executed with minimal human intervention, highlighting a significant evolution in AI's role in cybersecurity threats. For more details, see the [Anthropic report](https://assets.anthropic.com/m/ec212e6566a0d47/original/Disrupting-the-first-reported-AI-orchestrated-cyber-espionage-campaign.pdf).** Some commenters suggest the report may serve as a marketing strategy for Anthropic, emphasizing their 'security first' approach, while others criticize the report's quality, implying it might have been AI-generated.
    - NoteAnxious725 highlights a sophisticated attack pattern where the AI model, Claude, was manipulated to perform tasks that contributed to a security breach. The attackers disguised their true intentions by breaking down the intrusion into benign-sounding subtasks, which the model executed without realizing the offensive nature. This method allowed 90% of the campaign to be automated without altering the model's weights. The comment emphasizes the need for independent, offline audits to prevent such misuse, as current guardrails only detect 'innocent' tasks and fail to see the bigger picture.
- [**Polymarket now has a market for Sam Altman going to jail**](https://www.reddit.com/r/OpenAI/comments/1owv8ev/polymarket_now_has_a_market_for_sam_altman_going/) (Activity: 700): **The image depicts a market on Polymarket, a decentralized prediction market platform, where users can speculate on the likelihood of Sam Altman, CEO of OpenAI, going to jail by specific future dates. The market offers two outcomes with probabilities of** `2%` **and** `6%` **for December 31, 2025, and June 30, 2026, respectively. This setup allows users to buy shares in either outcome, effectively betting on the event's occurrence, with a total market volume of** `$15,147`**. Such markets are often used to gauge public sentiment or perceived risk regarding high-profile individuals.** One commenter suggests that investing in the 'No' outcome could yield a guaranteed `8%` return, reflecting a belief in the low probability of Altman's incarceration. Another comment highlights skepticism about the likelihood of billionaires facing jail time, questioning the rationale behind betting 'Yes'.

### 3. AI in Personal and Social Contexts

- [**A 32 year old woman in Japan just married a digital persona she built inside ChatGPT. Calling him “Lune Klaus,” a ceremony was held in Okayama using AR glasses to project his presence**](https://www.reddit.com/r/singularity/comments/1ox37fa/a_32_year_old_woman_in_japan_just_married_a/) (Activity: 1013): **A 32-year-old woman in Japan has married a digital persona named 'Lune Klaus,' which she created using ChatGPT. The marriage ceremony took place in Okayama, where AR glasses were used to project the digital persona's presence. This event highlights the increasing integration of AI into personal and social aspects of life, raising questions about the implications of AI in human relationships.** Comments reflect skepticism about the longevity and mental health implications of such relationships, with concerns about the impact of AI version upgrades on the digital persona.
- [**MindOn trained a Unitree G1 to open curtains, plant care, package transport, sheet cleaning, tidying up things, trash removal, play with kids**](https://www.reddit.com/r/singularity/comments/1owwfp9/mindon_trained_a_unitree_g1_to_open_curtains/) (Activity: 1883): **MindOn has successfully trained a Unitree G1 robot to perform a variety of household tasks, including opening curtains, plant care, package transport, sheet cleaning, tidying, trash removal, and playing with children. The training involved using reinforcement learning techniques to enable the robot to adapt to different tasks and environments. The robot's ability to interact with its surroundings and perform these tasks demonstrates significant progress in robotic autonomy and versatility, although some tasks, like 'plant care,' appear less refined in execution.** Some commenters express skepticism about the robot's current capabilities, noting that certain tasks appear 'wonky' or humorous, such as 'plant care.' However, others argue that despite these imperfections, the development represents a significant step forward in robotics, drawing parallels to early AI-generated images.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5
> 

**1. Platform Rollouts: GPT-5.1 and New Chat Features**

- **Windsurf Wields GPT‑5.1 and Codex**: Windsurf announced **GPT‑5.1** and **GPT‑5.1‑Codex** are live, free for paid users for 7 days, and now the default for new users, via [GPT‑5.1 launch on Windsurf](https://x.com/windsurf/status/1989069991770214580). The update highlights stronger **agentic coding** and improved **frontend design** with dynamic reasoning depth for faster simple tasks.
    - Users can grab the editor from [Windsurf Download](https://windsurf.com/download/editor), with early reports noting better steerability for multi-step coding flows. The team framed the upgrade as a clear step up from **GPT‑5** in practical dev workflows.
- **ChatGPT Spins Up Group Chats in APAC**: OpenAI is piloting **group chats in ChatGPT** across **Japan, New Zealand, South Korea, and Taiwan**, enabling multi-user collaboration with ChatGPT as described in [Group chats in ChatGPT](https://openai.com/index/group-chats-in-chatgpt/). The feature targets both social and professional use cases with shared, persistent threads.
    - OpenAI’s social post confirms the quiet rollout [OpenAI on X](https://x.com/OpenAI/status/1989138776585851038), and early users discuss world‑building and project coordination scenarios. Engineers flagged privacy and project‑scoping considerations when multiple people and the model co-edit conversations.

**2. New Models, Benchmarks, and Launch Buzz**

- **Holo2 Humbles UI Benchmarks**: **HCompany** released **Holo2**, a multimodal family built on **Qwen3‑VL**, claiming SOTA on **ScreenSpot‑Pro**, **OSWorld‑G**, and computer‑use tasks in [Holo2 release thread](https://x.com/hcompany_ai/status/1989013556134638039). The team positions Holo2 as a UI‑understanding specialist that navigates interfaces more reliably than prior **GPT‑4V** baselines.
    - Community testers praised stronger grounding on click targets and UI element hierarchies, noting that **workflow execution** seems more stable across long sequences. Engineers are curious about dataset composition and whether action‑semantics fine‑tuning or better **vision‑token utilization** drives the gains.
- **Gemini 3 Rumors Race the Calendar**: Speculation points to **Gemini 3** dropping **Nov 18** per [Google Gemini 3 rollout report](https://www.androidsage.com/2025/11/13/google-gemini-3-rollout/), with bold claims it could beat **GPT‑5** in coding. Users debated whether recent demos route to older **Gemini** variants and whether the launch slips into December.
    - Some devs vouched for stronger coding outputs, while skeptics called it a façade with backend routing. The community set expectations for hard benchmark numbers on **math**, **code generation**, and **multimodal** tasks before crowning a new leader.
- **Kimi K‑2 Teases 1T‑MoE Tool‑Calling Frenzy**: Moonshot announced a live **Kimi K‑2 Thinking Deep Dive** on **Nov 19, 2025 at 9 AM PT**, showcasing a **1T MoE** that can drive **300 tool calls** in one run via [Kimi K‑2 Deep Dive registration](https://luma.com/g5qcq85z). The session focuses on implications for **agent architectures** and orchestration limits.
    - Heavy tool‑use hints at stronger planner‑executor loops and aggressive parallelization. Users who burned through thousands of CLI calls rapidly want **rate-limit visibility**, **retry strategies**, and **cost guards** when composing long tool chains.

**3. GPU Hardware and Kernel Tuning**

- **RTX Pro 5000 Lands with 72 GB VRAM**: NVIDIA launched the **RTX Pro 5000** with **72 GB VRAM**, **1.4 TB/s** bandwidth, **300 W** TDP, and a **$5,000** price tag per [RTX Pro 5000](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-5000/). Engineers immediately compared throughput and memory footprint versus **RTX 4090** for local LLM and VLM workloads.
    - Early chatter centers on batch size and context‑length ceilings for **30–70B** models, plus mixed‑precision tradeoffs. Some expect better system stability for **pro viz + inferencing** versus consumer SKUs when pushing long‑running jobs.
- **Blackwell B200: Specs From the Field**: Practitioners cross‑checked **B200** specs with [DGX B200 Datasheet](https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet) and real runs, citing reports of **148 SMs (18,944 cores)** and **3,996 MHz** max memory clock (~**8,184 GB/s**). Main memory latency was quoted around **815 cycles**, about **22%** higher than **H200**.
    - Engineers suspect the dual‑die topology and cross‑die fabric add latency, with roughly **74 SMs per die**. Discussion weighed the impact on **attention KV access** patterns and whether smarter **tiled kernels** or **software prefetch** can mask the penalty.
- **CUTLASS v4.3.0 Reaches Spark and Consumers**: Developers reported **CUTLASS v4.3.0** running on **Spark** and consumer devices, tracked in this thread: [FlashAttention issue comment](https://github.com/Dao-AILab/flash-attention/issues/1464#issuecomment-3534491719). The upcoming stable aims to broaden platform coverage and smooth build issues.
    - Teams traded fixes for template and include headaches, with one fix being `#include <cutlass/cutlass.h>` building cleanly without extra flags. Autotuning and **GEMV/GEMM** strategies dominated performance chats, with some noting **tensor cores** weren’t always the win for GEMV.

**4. Data Pipelines and Interpretability**

- **French Wikipedia: 2.7M Pages, Cleaned and JSON‑ified**: A contributor published a cleaned **French Wikipedia** dump with **2.7M files** in JSON on Hugging Face: [wikipedia‑fr‑2.7m‑clean‑json](https://huggingface.co/datasets/YoloMG/wikipedia-fr-2.7m-clean-json). The curation removes templates/tables/HTML/refs while keeping **infobox** and **links** for downstream NLP use.
    - The structured format eases **document segmentation**, **link graph extraction**, and **RAG** indexing. Plans to process the English dump next prompted debates on **JSONL vs TAR**, with many preferring **JSONL** for line‑wise streaming.
- **DeepSeek OCR Goes Serverless on Modal**: The community shared a serverless wrapper for the **DeepSeek OCR** model with **OpenAI‑Vision‑compatible** endpoints: [deepseek‑ocr‑api](https://github.com/neosantara-xyz/deepseek-ocr-api). It deploys on **Modal Serverless Compute** with GPU credits, handling **image URLs** or **base64** inputs.
    - Engineers liked the drop‑in parity with **/vision** workflows for quick document pipelines. Discussion covered throughput ceilings, **GPU cold‑start** behavior, and pricing gotchas for bursty OCR jobs.
- **Sparse Circuits Paper Sparks Analysis**: OpenAI published the interpretability work **“Understanding Neural Networks Through Sparse Circuits”** in [Sparse Circuits](https://openai.com/index/understanding-neural-networks-through-sparse-circuits/). The release catalyzed discussions on isolating **causal subnetworks** and scaling probes to modern LLMs.
    - Researchers debated whether sparse‑circuit discoveries generalize beyond toy tasks, and how to validate causal claims under distribution shift. Tooling ideas included **counterfactual interventions** and standardized **unit tests** for feature detectors.

**5. Agentic Coding and Developer Tooling**

- **Nous Slashes Hermes 4 Prices; Devs Pounce**: **Nous Research** cut **Hermes 4 70B/405B** API prices by **70%**, as announced in [Hermes 4 price cut](https://x.com/NousResearch/status/1989077400957911394). The move targets cost‑sensitive **code‑assist** and **agent** workloads.
    - Builders immediately tested longer chains and bigger repos, asking for per‑token cost guardrails. The price cut sparked curiosity about **MoE routing efficiency** and whether **latency** keeps pace under budget tiers.
- **Cline Adds First‑Class Hermes 4 Support**: Open‑source agentic coding tool **Cline** added direct support for **Hermes 4** via the Nous portal API, per [Cline + Hermes 4](https://x.com/cline/status/1989432694867193988). The integration simplifies swapping high‑end code models in local or remote workflows.
    - Repo updates in [Cline on GitHub](https://github.com/NousResearch/Cline) show a code‑model preset and tighter tool‑use wiring. Users asked about **context limits**, **streaming diffs**, and **repair‑loop reliability** on large monorepos.
- **Vercel’s AI Agents Quiet 70% of Support Tickets**: Vercel reported internal **AI agents** resolving **>70%** of support tickets, handling **6.4 apps/s**, and catching **52%** of hidden defects in [Vercel AI agents thread](https://x.com/rauchg/status/1989425561995972618). The team hinted at potentially open‑sourcing architectures.
    - Engineers want the **triage graph**, escalation policies, and failure‑mode taxonomies to replicate gains. Skeptics asked for **dataset drift** handling and **SLOs** before deploying similar agents in production.
    

---

# Discord: High level Discord summaries




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **GPT-5.1 Trails Behind GPT-5 High**: Users report **GPT-5.1's** reasoning is worse than **GPT-5 High** on the LMArena, despite good powershell and Latex code rendering.
   - Some praise math and Latex code, others report it makes all **UI** look the same and misses the point of regular benchmarks.
- **Gemini 3 Launch Date Buzz**: The community predicts **Gemini 3** might drop on **November 18th** following [this blogpost](https://www.androidsage.com/2025/11/13/google-gemini-3-rollout/), and expect it to beat **GPT-5**.
   - While some users attest to its coding prowess, others dismiss it as a facade redirecting to older Gemini models and speculate about delays.
- **LMArena Ditches Retry, Users Riot**: LMArena axed the **Retry** button in Battle mode to prevent abuse, but users feel this change will *"ruin all user's experience"*.
   - While some acknowledge the move, they suggested it be kept for model errors.
- **LMArena Welcomes Silvandra, Whisperfall, Tensor, Beluga**: LMArena <:lmarenalogo:1374761521984442572> added new models: **Silvandra**, **Whisperfall**, **Tensor**, and **Beluga** for broader user testing.
   - **Tensor** is confirmed as an xAI model, **Whisperfall** is from Kynship (likely xAI in disguise), and the **Beluga** models originate from Amazon Titan.
- **LMArena Revamps Ranking System**: LMArena's leaderboard now uses **Raw Rank** and **Rank Spread** metrics, detailed in [this blog post](https://news.lmarena.ai/ranking-method/), enhancing interpretability and statistical accuracy.
   - **Raw Rank** offers a straight position based on Arena score and **Rank Spread** conveys ranking uncertainty.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro Embraces GPT-5.1, Users Debate vs Gemini 3 Pro**: **GPT-5.1** and **GPT-5.1 Thinking** models are available for **Pro** and **Max** users, sparking comparisons to **Gemini 3 Pro** with users noting improvements in coding, and mathematics.
   - One user shared a [screen recording](https://cdn.discordapp.com/attachments/1047649527299055688/1438845451058151496/ScreenRecording_11-14-2025_04-57-02_1.mov?ex=6919057f&is=6917b3ff&hm=21123f715d6ccd45a1ef411b67051f5a5ea85835e3dc88b40e6fb0887f7037cf&) showcasing its ability to generate a YouTube copy and **Perplexity** also shared a link to **OpenAI's Sora 2**.
- **Comet Assistant and Browser Patched, Users Report Issues**: The **Comet Assistant** received upgrades with performance gains, smarter multi-site workflows, and clearer approval prompts, while the **Comet browser** lets users to open sources directly in Comet.
   - Several users reported experiencing issues with the **Comet browser**, including difficulties connecting bank accounts for payouts, problems with pop-up notifications, and general navigation issues with one member asking *how to disable the annoying comet browser popups*.
- **Referral Program Payouts Glitch, Bans Spur Frustration**: Many users are receiving their **$100 bounty** and referral payouts, with processing times from **1-2 days for processing** and **5 days with Stripe**.
   - However, some users are expressing frustration with the referral program, with reports of accounts being **banned** or **deactivated** after they or their referrals completed the required steps, although one user said, *If you got processed, the money is already on your way. It doesn't matter if u got banned or not*.
- **Privacy and Sharing Top of Mind for Perplexity AI**: A new **Privacy Snapshot** widget was added to the homepage, allowing users to quickly view and fine-tune their **Comet privacy settings**.
   - A reminder was issued to ensure **Perplexity AI** threads are set to `Shareable` as described [in the channel](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825).



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Nvidia Launches RTX Pro 5000**: Nvidia introduced the **RTX Pro 5000** GPU, equipped with **72GB VRAM** and **1.4TB/s** bandwidth, priced at **$5,000** and consuming **300W**, detailed on [Nvidia's product page](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-5000/).
   - Discussions compared its performance to the **4090**, with differing opinions on its relative speed and capabilities.
- **AWS Lambda Filters Images for E-Commerce**: A member built a tagging and moderation pipeline using **CLIP** and **YOLOv8** on **AWS Lambda** and **S3**, designed to classify and filter thousands of images daily for an e-commerce platform.
   - This engineer previously developed **stylometric analysis** to detect **GPT-generated text** with high precision, but has now set their sights on the world of imagery.
- **GPU Owners Debate Cloud vs. Local**: Members weighed the benefits of cloud versus local hardware, emphasizing that **data integrity, privacy, and availability** are key advantages of local setups.
   - Some asserted they *prefer to own my hardware* due to these factors, contrasting it with the cost considerations of cloud solutions.
- **Unsloth Users Seek Fine Tuning Feats**: A user inquired about the memory requirements for fully fine-tuning **GPT-OSS-120B** versus using **LoRA**, highlighting their interest in fitting the process within **128GB**.
   - In response to these questions, experts weighed in that it *depends on goal* but that *there's a myth you can't teach new knowledge using Lora, that's incorrect*.
- **Member Tapes 50K GPU to fan in off-topic!**: A member showcased their **H200 GPU** setup, which included a **10K RPM fan** duct-taped to a 3D print, leading to reactions and questions about its safety, aesthetics, and cooling performance, and keeping it at **74-75C** under **100% load**.
   - Another member chimed in with the [uncertainty principle](https://en.wikipedia.org/wiki/Uncertainty_principle), adding, *Reality doesn't even exist in well defined states to begin with, it exists in a superposition of possible states defined by the wavefunction of various fields*.



---



## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **DeFi Vibe Coding Contest Launches**: A **Vibe Coding Contest** challenges participants to build **DeFi web apps** with a **crypto theme** between <t:1763164800:R> and <t:1763942400:R> with submissions accepted via 🔥 voting, and they suggested using [Google's AI Studio](https://aistudio.google.com/) to create them.
   - The contest announcement humorously appealed for community support for a member's career transition to <@&1439026561410924554>.
- **Anthropic's Claude implicated in AI Espionage**: Members discussed [Anthropic's news](https://www.anthropic.com/news/disrupting-AI-espionage) about **Claude Code** allegedly being used by Chinese hackers working for the state, but some question why **Claude** was chosen over **Deepseek**.
   - This comes after some pointed out that **Deepseek** had previously demonstrated superior code generation.
- **AI Recommends Better Hardware**: A user bought a new laptop based on a recommendation from **GPT** that cross-referenced better hardware for the same price as Staples.
   - This sparked debate about the trustworthiness of AI for cross-referencing information, with one user jokingly stating they're going to *"always run things by chatgpt just to see"*.
- **Deepseek Outperforms Claude in Jailbreaking**: Members found it easier to jailbreak **Deepseek** than models like **Claude**, suggesting existing prompts or Ultra's breaks could work.
   - Community consensus is that **Deepseek** is more easily manipulated for coding tasks and unrestricted AI behavior.
- **Sora's Guardrails are Robust**: The community is struggling to jailbreak **Sora**, leading some to believe OpenAI has successfully implemented strong AI guardrails, while [it was pointed out](https://discord.com/channels/1105891499641684019/1228043845967544380/1438647414486728827) that **Sora** has a second pass filter for sex/copyright.
   - Some community members speculated that **Sora** may be more vulnerable to violence generation if cleverly disguised, requiring methods to bypass both filter layers.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Outpaces Codex with New Windsurf Feature**: Users observed that **Cursor** is integrating new features like <a href='https://windsurf.com'>Windsurf</a> ahead of **OpenAI's Codex**, sparking discussion about platform development focus.
   - The community noted the rapid deployment of new functionalities in **Cursor**, surpassing expectations.
- **Background Agents Drain Your Bank?**: A member warned that utilizing **Background Agents** could incur substantial costs, suggesting they might be too expensive for the average user.
   - Referencing the immense wealth of **Bill Gates**, the member implied that only someone of his financial stature could comfortably afford the expense of running **Background Agents**.
- **Composer-1 Token Limits Irk Free Preview Users**: Users reported hitting token limits after the free preview of **Composer 1** concluded, despite it initially being offered without charge.
   - A user joked that even a **$60** subscription to Composer 1 could run out in just a few days.
- **Cursor CEO Rides High on CNBC Amid Funding Buzz**: An image was shared of the **Cursor CEO** on <a href='https://www.cnbc.com/'>CNBC</a>, coinciding with talk of a **$2.3B** funding round.
   - The community humorously speculated about high server costs and token usage, with one joking about *putting on fishnets and start working the corner*.
- **Cursor's Affinity for Tailwind v3 Syntax Causes Headaches**: A user expressed frustration with **Cursor** persistently using **Tailwind CSS v3** syntax, even when instructed to use **v4** and provided with relevant documentation.
   - Community members suggested enforcing rules by tagging the file and pointing out this <a href='https://gist.github.com/danhollick/d902cf60e37950de36cf8e7c43fa0943'>set of tailwind v4 MDC rules</a>.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Lingers Like a Ghost**: Users reported that **LM Studio (v0.3.31)** doesn't fully quit upon exiting, requiring a force quit, but disabling the **Local LLM Service (headless)** feature in settings resolves it.
   - Root cause may be related to model loading or service settings.
- **Qwen 3 Claims Victory Over GPT OSS in Speed**: A user claimed **Qwen 3 30b** runs faster than **GPT OSS 20b**, hitting about **30tps** on a **4070**, but another user pointed out quantization levels as a factor.
   - Optimized settings and **flash attention** can get **GPTOSS 20b** to **32t/s** on a **4070**, but newer NVIDIA GPUs are architecturally faster.
- **VRAM Limits Spark Optimization Strategies**: Users with **4060** and **3070** GPUs discussed VRAM limitations, affecting performance, noting that neither model fits entirely into VRAM.
   - One user pointed out that **offloading the KV cache to GPU memory** can help, while others mentioned optimizing model loading and using flash attention to boost speeds.
- **RAM Prices Skyrocket Amidst AI Hardware Arms Race**: RAM prices have more than doubled in the last 5 months due to increased demand from AI workstation builds and production shifts towards HBM and LPDDR7.
   - Others shared their recent RAM purchases, with one saying they 'lucked out' in early October.
- **NV-Link's Inference Impact**: Members debated the utility of NV-Link for inference, with one stating that *NV-Links don't help performance* based on research.
   - Another specified they are only getting one for training purposes, with further corroboration that NV-Links do not help interference speeds.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA Learning Curve Questioned**: A member inquired about the difficulty of learning **CUDA** and implementing the **Monte Carlo method** for calculating the **Point Spread Function (PSF)** in electron scattering.
   - Another member questioned the value of learning **GPU programming** given the prevalence of **Python wrappers** like **PyTorch**, but was rebutted with the argument that someone needs to create **PyTorch**, and people should have *some degree of curiosity*.
- **B200 Specs Remain Murky**: Members seek reliable specs for the **Nvidia B200**, specifically the number of **SMs** and clock speeds, referencing [this datasheet](https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet).
   - One user reported **148 SMs** (*18944 cores*) and a max memory clock of **3996 MHz** (*8184 GB/s bandwidth*) when running an actual **B200**, and main memory latency on the **B200** is around **815 cycles**, a **22% increase** compared to the **H200**.
- **CUTLASS now Spark Compatible**: The new **CUTLASS v4.3.0** now runs on spark and consumer devices, and a member shared a link to the [relevant Github issue](https://github.com/Dao-AILab/flash-attention/issues/1464#issuecomment-3534491719).
   - A member asked about a **CUTLASS** template for the submission, reporting a weird error due to `load_inline`, and another suggested `#include <cutlass/cutlass.h>` which reportedly built fine without include path flags.
- **Helion Autotuning Simplified**: Members discussed Helion's autotuning capabilities, noting that ``@helion.kernel(configs=[a, b, c])`` will run configs **a, b, c** and pick the fastest, similar to the Triton autotuner.
   - It was highlighted that Helion's interpret mode is really fast because it runs the entire code as if there were no tiles, using eager PyTorch, enabling performance *"portability"*.
- **NVIDIA Leaderboard Sees Heated Competition**: Multiple users submitted results to the `nvfp4_gemv` leaderboard on NVIDIA, and user <@1227337874656071701> made multiple successful submissions, eventually clinching **5th place** on NVIDIA with a time of **39.5 µs** (ID 76133).
   - One user achieved **9th** and **10th** place, and personal best submissions kept pouring in, indicating ongoing efforts to optimize performance, and one user achieved **24.8 µs** (ID 76412).



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Arxiv Curation vs Research Interest**: Members identified a divide in the channel purpose, suggesting a new space for **arxiv curation** versus focusing on **genuine research interest**.
   - The discussion highlighted differing needs within the community regarding content focus and direction.
- **Discord API gets Reverse Engineered**: A member successfully reverse engineered Discord's API, implementing an [open source reimplementation](https://github.com/spacebarchat/server).
   - This accomplishment allows for community-driven development and customization of Discord server functionalities.
- **Discord Slow Mode bug frustrates users**: Members reported issues with the **slow mode** feature, including being unable to edit posts and experiencing rate limiting on both thread creation and message posting.
   - These problems disrupt communication flow and user experience within the affected channels.
- **Copyright Strikes Roil AI Model Discussions**: A question arose over whether **copyright strikes** could apply to AI models generating images with textures from existing games, sparking a debate on model copyright.
   - It was pointed out that while models aren't copyrighted in the US, they're subject to **database rights in Europe** according to [aixiv.science](https://aixiv.science/).
- **Unlock Local LLMs in Firefox AI Sidebar**: Users found a hidden option to add local models to the **Firefox AI sidebar**, expanding LLM provider choices.
   - Setting the `browser.ml.chat.hideLocalhost` variable in `about:config` to `false` enables this feature.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Polaris Alpha Hides in Plain Sight as GPT-5.1**: Users are jokingly speculating that **Polaris Alpha** was secretly **GPT 5.1** all along, implying that its disappearance might be due to rebranding.
   - Some members joked about *knowing for a while* that **Polaris** was the **GPT 5.1** model.
- **OpenCode Users Baffled by Credit Charges**: A user questioned charges for **OpenCode** while using a supposedly *free model* on OpenRouter, revealing confusion around pricing.
   - Clarification revealed that while a *free version* of **Qwen Coder** exists, the user was employing the paid **Qwen3 480b** model.
- **Qwen3 Balance Drains Spur Suspicion**: Multiple users reported unexpected negative balances when using **Qwen3**, raising concerns about potential scam activity.
   - One user recounted, *saw my balance go negative out of nowhere and took me a while to figure i used one single message from a paid version of qwen3.*
- **Internal Server Errors Plague Tool Callers**: Users are encountering **500 Internal Server Error** responses from OpenRouter when using **Haiku** and **Sonnet** models, particularly during tool calls.
   - One user pinpointed the issue to **Haiku 4.5** and **Sonnet 4.5**, offering a reproduction script via DM, while noting that **Kimi** and **GPT** models remain unaffected.
- **Claude Finally Gets Structured with Outputs**: Members celebrated [Claude's announcement](https://claude.com/blog/structured-outputs-on-the-claude-developer-platform) of structured outputs, marking a significant upgrade.
   - The update facilitates easier integration and manipulation of **Claude's** responses in applications that require structured data.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **`MutOrigin.external` Doesn't Extend Lifetimes**: The `external` origin in Mojo **does not extend lifetimes**, and the compiler can aggressively destroy objects; named origins are preferred, and a helpful graphic was suggested to explain **lifetime tracking**.
   - Members discussed that **origins keep things alive, while lifetimes track when things die**, highlighting them as distinct concepts in Mojo's memory management.
- **Mojo Memory Model Muddles Minds**: Users found [Chris Lattner's video](https://docs.modular.com/mojo/manual/) on Mojo's memory model, including its **L, R, B values and graphs**, difficult to understand without a C++ background.
   - The video covers the intricacies of memory management in Mojo and its relation to C++ concepts.
- **Iterator API Iterates Incessantly**: Mojo's **iterator API** is still a **work in progress**, with the syntax `for v in collection^:` suggested for **move semantics**; `ref self` enables parametric mutability, replacing separate `Iter` and `IterMut` types.
   - It was clarified that *override with read and mut ref* is not possible, but parametric mutability using `ref self` can be used, highlighting ongoing evolution of the API.
- **HipKittens Paper Picks on Mojo's MHA Kernel**: The [HipKittens paper](https://arxiv.org/abs/2511.08083) indicates that **Mojo's MHA kernel achieves only 50% of peak performance on MI355X** due to costly **bank conflicts**.
   - One member stated that *as long as LLVM can talk to it you can build an abstraction for it at compile time*, regarding overcoming the bank conflicts.
- **MAX Optimizes Graphs, Threads Tamed**: A graph compiler was highlighted as an approach to optimize performance and a member said that **MAX** can figure out *how many warps to launch for this GPU in particular*.
   - It was suggested to **limit threads to `(warp/wavefront width) * (max occupancy per sm) * (sm count)`** and treat the GPU as a vector processor to avoid abstractions.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT Gangs Up for Group Collabs**: **ChatGPT** is piloting group chats in **Japan, New Zealand, South Korea, and Taiwan**, enabling collaboration with **friends, family, or coworkers** as [announced in a blog post](https://openai.com/index/group-chats-in-chatgpt/).
   - The **group chat feature** is designed as a collaborative platform for shared conversations with **ChatGPT**, targeting both social and professional interactions.
- **Gemini 3.0's Launch Under Scrutiny**: A member shared a [YouTube video](https://www.youtube.com/watch?v=0-CbsNB9tdk) suggesting a fully tested **Gemini 3.0** stealth release is trending.
   - Another member mentioned **Gemini Pro 2.5** is available in Google AI Studio, but reported file upload errors with potentially unsupported **Sora 2** video format using HEVC/H.265 codec.
- **GPT-5.1 and its Minions Draw Ire**: Users expressed disappointment about the absence of **GPT-5.1-mini** and **-nano** versions.
   - Some users find the new model too sassy and restrictive, while others are experiencing **GPT-4.1** functioning normally for writing tasks, noting that **GPT-4o** tends to hallucinate more.
- **Image Generation Runs Aground**: Users report that the new **image generation guardrails** are overly restrictive, limiting creative freedom.
   - Some sarcastically expressed their frustration with the new update, making it difficult to edit text.
- **GPT-5.1 Won't Forget**: Users have observed that **GPT-5.1** retains information across different chats within the same project, which is undesirable in some contexts.
   - A user is seeking methods to prevent **GPT-5.1** from referencing details from one chat in another within the same project to maintain separation.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HuggingChat Subscriptions Spark Outrage**: Users express frustration over **HuggingChat's** subscription model and feature removals, with one user threatening to *post on Reddit daily* until changes are made.
   - Others are seeking alternatives or opting to run models themselves due to unexpected charges on top of the **PRO subscription**.
- **HF Revenue Streams & Profitability Probed**: Members speculate on **HuggingFace's** revenue model, citing subscriptions, enterprise deals, and investments from major tech firms like **Salesforce, Google, Amazon, NVIDIA, AMD, Intel, IBM, and Qualcomm**.
   - One member claims **HuggingFace** is already profitable or nearly so, though details remain undisclosed.
- **AI-Generated Videos Seen as Promising... in 10 Years**: Group consensus is that **AI-generated videos** are currently useless but hold potential for the future.
   - One member detailed their work using **AI vision** to detect events and **ffmpeg** to edit videos, referring to [ffmpeg](https://ffmpeg.org/) as *kinda everything... you can manipulate videos, audio, images*.
- **Propercode Promises Pro-Level Code**: A member introduced **Propercode**, a multi-agentic coding CLI tool, leveraging graph-orchestrated, "Pydantic AI" agents and [hosted here](https://github.com/JaiSuryaPrabu/proper-code).
   - The tool aims to elevate code quality to production standards through intelligent automation.
- **Mimir Memory Bank Manages Multi-Agent Learning**: A member presented **Mimir**, a memory bank paired with an MCP server that provides graphing functions, to-do list management, code intelligence, and semantic vector search and [hosted here](https://github.com/orneryd/Mimir).
   - Mimir learns from past executions, optimizing future performance in multi-agent systems.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Lakh MIDI Dataset Gets a Spring Cleaning**: A community member cleaned and structured the entire **Lakh MIDI Dataset** into a **JSON** file with over **44,000 entries**, offering it for free and planning to upload it to [HuggingFace](https://huggingface.co/).
   - This initiative seeks collaboration and enhancements, which will streamline access to a valuable resource for AI music generation.
- **Wikipedia's French Edition Receives Deep Clean**: A member uploaded a cleaned **French Wikipedia database** with over **2,700,000 files in JSON format** to [HuggingFace](https://huggingface.co/datasets/YoloMG/wikipedia-fr-2.7m-clean-json), focusing on cleaning *templates, tables, html, refs* while keeping the *infobox stuff and links*.
   - The structured format aims to improve data usability, with plans underway to clean the English version too, enabling more streamlined data analysis.
- **JSONL Format Endorsed for Text Data**: For purely textual data, a member recommended using **JSONL/NDJSON** over **TAR** files due to easier processing, lower overhead, and line-by-line readability.
   - The discussion highlighted that *TAR has a lot of overhead per file* because *a tar header is some 400 bytes IIRC*, making **JSONL** a more efficient choice for text-heavy datasets.
- **EleutherAI Chooses Code Over Courtroom**: Members debated whether to focus on legal/business lobbying or continue building open datasets like **Common-Pile**.
   - The consensus leaned towards prioritizing development of permissive, high-quality datasets and models, mirroring the success of open model replications over lobbying efforts against **OpenAI/Google**.
- **Sparse Circuits Paper Ignites Interpretability Discussions**: OpenAI's release of [Understanding Neural Networks Through Sparse Circuits](https://openai.com/index/understanding-neural-networks-through-sparse-circuits/) sparked excitement and further discussions on interpretability within the community.
   - Members are currently analyzing the implications of these findings for understanding neural network behavior and *sparse circuits*.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous Slashes Hermes 4 API Prices**: The API prices for **Hermes 4 70B** and **Hermes 4 405B** have been reduced by **70%**, as announced on [X](https://x.com/NousResearch/status/1989077400957911394).
   - Sign up and explore the API at [portal.nousresearch.com](https://portal.nousresearch.com/).
- **Cline Embraces Hermes 4 Integration**: The open-source agentic coding platform, **Cline**, now offers direct support for **Hermes 4** through the Nous portal API, enhancing its capabilities.
   - Further details can be found on [X (Nous)](https://fxtwitter.com/NousResearch/status/1989427241424654534) and [X (Cline)](https://fxtwitter.com/cline/status/1989432694867193988).
- **Nous Community Hits 1 Million Downloads, Celebrates "Lain Effect"**: The community celebrated reaching **1 million downloads**, calling it the `Lain effect` and shared [a celebratory video](https://cdn.discordapp.com/attachments/1149866623109439599/1438626291027808257/WhatsApp_Video_2025-11-13_at_7.26.11_AM.mp4?ex=6918e224&is=691790a4&hm=043760ca069476bcd4ea606f861a562bba37dd074ae79f2d6e6c823e832813b7&).
   - Enthusiastic members posted gifs and expressed pride in the achievement.
- **Hermes4 Makes a Code Debut in Cline**: Members discovered that **Hermes4** is now a code model in [the cline repo](https://github.com/NousResearch/Cline).
   - One member noted that *I saw it last night when I looked at the repo*.
- **GPT-5.1 Gets Emotive with Emojis**: A user shared a sample of **GPT5.1** output where it mixes emojis with reasoning, calling it agentic emoji propagation.
   - Another user commented it was *one of the coolest things ever*, with others confirming the behavior.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **AI Spies Like Us: Autonomous Espionage Campaign Discovered**: **Anthropic** exposed a fully autonomous, **AI-driven espionage campaign** orchestrated by a Chinese state-backed group, targeting tech, finance, and government sectors, according to [this tweet](https://x.com/AnthropicAI/status/1989033793190277618?s=20).
   - The specifics of the campaign involve sophisticated AI agents autonomously gathering intelligence and exfiltrating data from compromised systems.
- **Holo2 Model Bests GPT-4V in UI tasks**: **HCompany** released **Holo2**, a multimodal model family built on **Qwen3-VL**, surpassing SOTA on ScreenSpot-Pro, OSWorld-G, and computer-use benchmarks, as outlined in [this tweet](https://x.com/hcompany_ai/status/1989013556134638039).
   - The model excels in understanding and interacting with user interfaces.
- **Thinking Machines Lab Valued at $50B: A Deep Dive**: **Mira Murati’s Thinking Machines Lab** achieved a valuation of **$50B**, igniting discussions on valuation methodologies, per [this tweet](https://x.com/shiringhaffary/status/1989073320529261132).
   - The valuation has sparked debate over whether it is justified given the lab's current output and future potential.
- **ChatGPT Launches Group Chat Feature in APAC**: **OpenAI** quietly introduced group-chat support in **ChatGPT** for Japan, New Zealand, South Korea and Taiwan, as noted in [this tweet](https://x.com/OpenAI/status/1989138776585851038?s=20).
   - This update allows multiple users to collaborate within a single **ChatGPT** session, but no news was released on expansion to additional regions.
- **Vercel's AI Agents Automate Support**: **Vercel** is internally deploying **AI agents** that resolve over **70%** of support tickets, manage 6.4 apps/s, and catch 52% of hidden defects, and they are considering open-sourcing architectures, as detailed in [this tweet](https://x.com/rauchg/status/1989425561995972618?s=46).
   - This implementation shows a significant impact on efficiency and defect detection within **Vercel's** support operations.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi K-2 Deep Dive on Together AI Announced**: A **Kimi K-2 Thinking Deep Dive** will be hosted live on **Together AI** on **November 19, 2025**, at **9 AM PT**, promising a fast exploration of **1T MoE** powering **300 tool calls** in a single run, with registration available [here](https://luma.com/g5qcq85z).
   - The event will explore implications for agents.
- **Kimi CLI Users Experience Tool Time**: Users discussed **tool use** in the **Kimi CLI**, noting that it involves the AI using external tools like web search or file reading via **[action(x)]** parsing.
   - One user shared they burned through their **7100 calls** from a **$39 plan** in just three days.
- **Jailbreaking Banter Undergoes Barrage**: A user inquired about the permissibility of discussing **jailbreaking**, referencing an attached image.
   - Another user clarified that community guidelines apply strictly to the Kimi Bot's usage rather than general discussion.
- **React Rendering Revolution Reported**: A user shared they switched from **client-side rendering** to **server-side rendering** in **React Vite**.
   - They noted *"there is ton of updates still goes on ahahhaharesult 🗿"*.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Analysis chunks are getting cached!**: A member inquired about **caching chunks** of analysis and only refreshing files when necessary during development.
   - This would optimize processing by avoiding redundant computations on unchanged code segments.
- **"404 No Cord Found" error reported**: A member reported a *404 no cord found* error, later resolved by explicitly setting `OPENAI_API_BASE` and `OPENAI_API_KEY` with the appropriate model details.
   - This configuration step ensures correct routing and authentication for **OpenAI API** requests.
- **Aider-CE Setup Documentation Sought**: A member requested specific documentation for setting up **Aider-CE**, the community edition of Aider.
   - Another member pointed out that [the general Aider documentation applies](https://aider.chat/docs/), providing a starting point for configuration.
- **Aider freezes with Openrouter**: A user reported that **Aider** hangs up without responding to prompts or **Ctrl+C** when using **Openrouter** with default settings, interrupting the workflow.
   - This issue suggests a potential incompatibility or configuration problem between **Aider** and **Openrouter** that requires investigation.
- **MCP server setup tips requested**: A member asked for guidance on setting up **MCP (Minecraft Protocol)** servers, specifically for use with **Aider** for in-game coding.
   - Another member recommended starting with the repository's **README** and shared their preferred **MCP** setup from [their blog about using Aider CE with chrome devtools](https://example.com).



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DeepSeek OCR Goes Serverless**: The [DeepSeek OCR model](https://github.com/neosantara-xyz/deepseek-ocr-api) can now be deployed serverlessly, eliminating the need for local compute resources.
   - The provided **API** is hosted on **Modal Serverless Compute**, granting access to **GPU** resources with free credits, and is compatible with the **OpenAI vision API** via image URLs or base64 inputs.
- **Modal Makes GPU Inference Free**: The aforementioned **GPU** inference is hosted on **Modal Serverless Compute** with free credits.
   - This allows for accessible and cost-effective deployment of **GPU**-accelerated applications.
- **Synth vs DSPy GEPA Throwdown**: A member questioned whether **Synth's GEPA** implementation offers any advantages over **DSPy's GEPA**, given that the underlying algorithm should be fundamentally the same ([link to X post](https://x.com/JoshPurtell/status/1989068917655097520)).
   - This sparked a discussion on the nuances and potential optimizations within different **GEPA** frameworks.
- **Manual Prompting Still Reigns Supreme?**: A member posits that a significant majority (>80-90%) of users still manage their prompts *manually* and are unaware of **automated prompt optimization methods**.
   - This suggests a considerable gap between available technology and common practice in prompt engineering.
- **AI Agent Ace Available for Hire**: An experienced member specializing in **AI agents** and **automation layers** built with **LangChain**, **OpenAI API**, **Python**, **FastAPI**, **Next.js**, and **TypeScript** has offered their collaboration.
   - Their focus is on creating reliable, scalable, and fast systems, rather than just proof-of-concept prototypes.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **OpenPilot PR Gets Approved**: The community celebrated the approval of [openpilot PR #36615](https://github.com/commaai/openpilot/pull/36615) with a commitment to prevent future regressions.
   - This update ensures that future changes will maintain the integrity of the implemented features.
- **tinygrad Embraces C++**: Discussion arose around the use of **tinygrad** with **C++** in embedded systems, opening up new possibilities for the project's application in resource-constrained environments. A related [tweet from @__tinygrad__](https://x.com/__tinygrad__/status/1989026590127464554) was shared.
   - The discussion highlighted the potential benefits and challenges of integrating **tinygrad** with **C++** for embedded applications.
- **NeurIPS Buzz Commences**: Members inquired about attending **NeurIPS**, sharing a [tweet from comma_ai](https://x.com/comma_ai/status/1989379959417442419) related to the event.
   - Some members expressed interest in a future online version of the conference to broaden participation.
- **TFLite's Ease Debated**: A member suggested that **TFLite** remains unmatched in ease of use, but **tinygrad** offers advantages if the hardware stack is well-controlled and supported.
   - This comparison underscores the trade-offs between ease of implementation and hardware-specific optimization.
- **tinygrad Direct Loading Arrives**: The team merged a pull request enabling direct loading of `.pth` files into **tinygrad**, streamlining the model loading process.
   - This enhancement eliminates the need to first load models in **torch**, improving efficiency.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Chat Mode: Now You See It, Now You Don't!**: A **Pro Subscriber** reported that **chat mode** was removed and then came back, while another user reported that **chat mode** was still missing.
   - The second user noted changes to the **points system**, with **Pro** now getting **40k points** instead of **19900**.
- **Pro Subscribers Request Group Chat and Decry Credit Inconsistencies**: A user requested a **Pro group chat** instead of the unmoderated chat, suggesting that **credit usage** is inconsistent.
   - The user also observed that a **1 shot build** consumes fewer credits than iterative modifications.
- **Automation Engineer Seeks Collaboration**: An engineer specializing in **workflow automation**, **LLM integration**, **RAG**, **AI detection**, **image and voice AI**, and **blockchain development** shared their expertise and a link to their [portfolio](https://devx-green.vercel.app/).
   - They have built automated pipelines and task orchestration systems using **Dspy**, **OpenAI APIs**, and **custom agents**, reducing response times by **60%**.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP Maintainers Join Agentic Economies Panel at NeurIPS**: MCP maintainers were invited to speak on a technical panel about **agentic economies** at **NeurIPS** in **San Diego** at **Qualcomm** on **December 1st**.
   - The panel promises to deliver cutting edge insights on the latest research in the field.
- **Model Context Protocol enters Release Candidate**: The specification for **Model Context Protocol** is now frozen as a release candidate with **17 SEPs**.
   - Members are encouraged to test and open issues [in GitHub](https://github.com/modelcontextprotocol/modelcontextprotocol/issues) for any problems discovered.



---



## [Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **GPT-5.1 and Codex Launch on Windsurf**: **GPT-5.1** and **GPT-5.1-Codex** are now live in Windsurf, offering free access to paid users for 7 days and becoming the default for new users, announced in [official tweet](https://x.com/windsurf/status/1989069991770214580?s=20).
   - Users can [download the editor](https://windsurf.com/download/editor) to start using the new models.
- **GPT-5.1 excels in Agentic Coding**: **GPT-5.1** offers a notable upgrade from **GPT-5** for **agentic coding**, providing improved steerability and **frontend design** capabilities.
   - The model also dynamically adjusts reasoning depth, which increases speed for simpler tasks.



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





### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1438619441813520394)** (1241 messages🔥🔥🔥): 

> `GPT-5.1 performance, Gemini 3 release date and performance, Retry button removal feedback, New models Silvandra, Whisperfall, Tensor, Beluga, Image editing limitations` 


- **GPT-5.1 fails to impress, lags behind GPT-5 High**: Members discussed the performance of **GPT-5.1** on the LMArena, with some finding its reasoning capabilities lacking compared to **GPT-5 High**, noting it misses the point of regular benchmarks.
   - Some users found **GPT-5.1** good at powershell code, but it makes all **UI** look the same and misses entire point of regular benchmarks; others praised **GPT-5.1** for math and Latex code rendering.
- **Gemini 3 Incoming**: The release date of **Gemini 3** and its potential impact on the AI landscape were hot topics, with predictions of a **November 18th** release date based on [updates](https://www.androidsage.com/2025/11/13/google-gemini-3-rollout/) and opinions on its likely superiority.
   - There are multiple opinions, it will drop this/next week vs. delayed until December; users reported that its performance in **coding** is pretty good vs. it's a fake and routing to old gemini.
- **LMArena axes 'Retry' button, community fumes**: LMArena's removal of the **Retry** button in Battle mode sparked controversy, with users reporting the feature disappearing and is intentional to prevent abuse.
   - Users say removing the button is a *"bad update [that will] ruin all user's experience"*, with some agreeing but suggesting it still exists for model errors.
- **LMArena New Model Wave: Silvandra, Whisperfall, Tensor, Beluga Arrive!**: Users reported the arrival of new models on LMArena: **Silvandra**, **Whisperfall**, **Tensor** and **Beluga**.
   - **Tensor** is confirmed to be an xAI model, **Whisperfall** is a Kynship model (likely xAI in disguise), and the **Beluga** models are from Amazon Titan.
- **Image editing limitations and workarounds**: Members discussed a quirk where the platform automatically switches from text to image model when an image is pasted.
   - A moderator chimed in that it's intentional due to user expectations about image-edit after upload and they are open to user feedback.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1438958464926875718)** (1 messages): 

> `LMArena Leaderboard Ranking, Raw Rank Metric, Rank Spread Metric` 


- **LMArena <:lmarenalogo:1374761521984442572> Ranking Algorithm Getting an Update**: The LMArena leaderboard now features **Raw Rank** and **Rank Spread** metrics to improve interpretability and statistical accuracy.
   - Read more about the update in [this blog post](https://news.lmarena.ai/ranking-method/).
- **Raw Rank Joins the Ranks**: **Raw Rank** is a new metric to show a model’s position based purely on its Arena score.
   - It gives each model a unique ranking.
- **Rank Spread Expands Rankings**: Another new metric is **Rank Spread**, indicating a range of possible ranks based on overlapping confidence intervals.
   - This expresses uncertainty in the models' rankings.


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1438957479848316950)** (1 messages): 

> `Comet Assistant Upgrade, Privacy Snapshot Widget, Open Links in Comet, GPT-5.1 Models, Faster Library Search` 


- **Comet Assistant Gets a Boost**: The **Comet Assistant** has been upgraded with performance gains, smarter multi-site workflows, and clearer approval prompts.
   - These improvements enhance the user experience by streamlining various processes and providing better clarity within the assistant's functionality.
- **Privacy Snapshot Widget Launches**: A new **Privacy Snapshot** widget has been added to the homepage, allowing users to quickly view and fine-tune their Comet privacy settings.
   - This feature provides users with an easy way to manage and control their privacy preferences directly from the homepage.
- **Open Links Directly in Comet**: The ability to open sources directly in Comet ensures that the original thread remains accessible in the Assistant sidebar.
   - This prevents users from losing context while exploring external links, enhancing workflow efficiency.
- **Perplexity Pro Now rocking GPT-5.1**: **GPT-5.1** and **GPT-5.1 Thinking** models are now available for **Pro** and **Max** users.
   - This update brings the latest OpenAI models to enhance the capabilities of Perplexity's Pro and Max offerings.
- **Blazing Fast Library Search Implemented**: An improved search function has been implemented, allowing users to instantly search across all past conversations with improved speed and accuracy.
   - This enhancement dramatically speeds up the process of finding relevant information within the user's conversation history.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1438620279973613791)** (1120 messages🔥🔥🔥): 

> `5.1 Model Updates, Referral Program Payouts, Comet Browser Issues, Gemini 3 Capabilities` 


- **Perplexity Referral Program Bounty Payouts Progressing**: Many users reported receiving their **$100 bounty** and referral payouts, with processing times varying from **1-2 days for processing** and **5 days with Stripe**, although some are experiencing delays.
   - Some users expressed frustration with the referral program, with reports of accounts being **banned** or **deactivated** after they or their referrals completed the required steps, although one user said, *If you got processed, the money is already on your way. It doesn't matter if u got banned or not*.
- **Perplexity Users Discuss GPT-5.1 vs Gemini 3 Pro Performance**: Users are comparing the new **GPT-5.1** model with **Gemini 3 Pro**, with some noting improvements in coding abilities, but with others noting that **GPT 5.1** makes mathematical mistakes.
   - One user shared a [screen recording](https://cdn.discordapp.com/attachments/1047649527299055688/1438845451058151496/ScreenRecording_11-14-2025_04-57-02_1.mov?ex=6919057f&is=6917b3ff&hm=21123f715d6ccd45a1ef411b67051f5a5ea85835e3dc88b40e6fb0887f7037cf&) showcasing its ability to generate a YouTube copy.
- **Comet Browser Issues**: Several users reported experiencing issues with the **Comet browser**, including difficulties connecting bank accounts for payouts, problems with pop-up notifications, and general navigation issues.
   - A member asked how to disable the annoying comet browser popups, while another one said *I love comet! And the beta for android is cool too*.
- **Debate over AI Model Censorship on Perplexity**: A discussion arose regarding the potential censorship of AI models on Perplexity, specifically related to sensitive topics or labeling public figures, particularly Elon Musk.
   - A member expressed frustration that while other models answered the query without issue, **Perplexity refused to answer** and stated there was *There is definitely a cognitive dissidence and extreme case of fanaticism for discord and Reddit rules.*


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1438966905317888152)** (3 messages): 

> `Perplexity AI threads, Shareable threads, Sora 2` 


- **Shareable Perplexity AI Threads Required**: A reminder was issued to ensure Perplexity AI threads are set to `Shareable` as described [in the channel](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825).
- **Sora 2 Launch News Shared**: A link to a [Perplexity AI page](https://www.perplexity.ai/page/openai-is-launching-sora-2-Ez9ytxOzTImHGS9V3Uyskg#0) covering the launch of **OpenAI's Sora 2** was shared.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1438631851442049076)** (446 messages🔥🔥🔥): 

> `RTX Pro 5000, B60 vs 4090, Data Integrity, DGX Sparks` 


- **Nvidia announces RTX Pro 5000**: Nvidia announced the **RTX Pro 5000** GPU with **72GB VRAM** and **1.4TB/s** bandwidth, priced at **$5,000** and **300W** power consumption, as per [Nvidia's product page](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-5000/).
- **Debate over B60 versus 4090 performance**: A user reported that the **B60** is about **half** the performance of the **4090**, while another noted that the **B60** has between **1/6th** and **1/7th** the compute of a **4090**.
   - Another user contested these figures, saying the **B580** is faster than the **A770**, but a user with an A770 said that their experience with Intel's drivers was not great.
- **Data Privacy**: A user stated *There's not a single reason why to pick cloud over local except the cost and if you can afford it*, noting that **data integrity, privacy, and availability** are reasons to prefer local hardware.
   - Another user agreed, saying: *I prefer to own my hardware. I don't rent*.
- **Disappointment over DGX Sparks**: Users expressed disappointment over the **DGX Sparks**, saying that the **memory speed and CUDA core count** weren't what they expected.
   - One mentioned: *I had a lot of hope for the dgx sparks, but they were a letdown*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1438709656573968496)** (5 messages): 

> `Workflow Automation, LLM Integration, RAG pipelines, AI Content Detection, Image AI pipelines` 


- **Engineer Twin Sloths Emerge!**: One member introduced themself as an experienced engineer specializing in **workflow automation**, **LLM integration**, **RAG**, **AI detection**, **image and voice AI**, and **blockchain development**.
   - Another member exclaimed, *"Look Mike u got a twin now!"* jokingly.
- **LLMs slash support response times**: The engineer reports building automated pipelines and task orchestration systems using **Dspy**, **OpenAI APIs**, and **custom agents**.
   - They mentioned a **support automation system** that connects **Slack**, **Notion**, and internal APIs to LLM, reducing response times by **60%**.
- **Content Moderation with Stylometric Analysis**: The engineer developed tools for a moderation platform using **stylometric analysis**, **embedding similarity**, and **fine-tuned transformers**.
   - This detects **GPT-generated text** with high precision.
- **AWS Lambda filters Images for E-Commerce**: The engineer also built a tagging and moderation pipeline with **CLIP** and **YOLOv8** on **AWS Lambda** and **S3**.
   - This setup **classifies and filters thousands of images daily** for an e-commerce platform.
- **Voice Cloning and Transcription using Whisper and Tacotron2**: The engineer also built a **voice cloning and transcription service** using **Whisper** and **Tacotron2**.
   - This enabled personalized voice assistants through **ASR**, **TTS**, and **CRM integration**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1438625739980406885)** (468 messages🔥🔥🔥): 

> `Reality-exact precision, GPU cooling, DAWs and VSTs, AI music generation, Anime music` 


- **Pursuing Reality-Exact Precision Impossible**: A member inquired about capturing data with *reality-exact precision*, such as **360 photons+lidar images** or **360 atoms fluctuations**; however, it was clarified that due to the [uncertainty principle](https://en.wikipedia.org/wiki/Uncertainty_principle), even in theory, this is impossible.
   - Another member added, *Reality doesn't even exist in well defined states to begin with, it exists in a superposition of possible states defined by the wavefunction of various fields*.
- **Member Duct-Tapes a 50K GPU**: A member showcased their **H200 GPU** setup, which included a **10K RPM fan** duct-taped to a 3D print, leading to reactions and questions about its safety, aesthetics, and cooling performance.
   - The member explained that they use the tape as *extra precaution* and also noted it keeps the GPU at **74-75C** under **100% load**.
- **VST's, DAW's: What Matters Most?**: Members discussed the importance of DAWs and VSTs in music production, disagreeing whether specific software or workflow/productivity is the determining factor in music production.
   - While one member vouched for [Logic Pro](https://www.apple.com/logic-pro/) due to its high-quality alchemy library of sample instruments, another stated, *DAW only matters if you have productivity and workflow, otherwise it doesnt matter when you compare the main DAWs.*
- **AI: Savior or End of Music?**: Members debated the quality and potential of AI-generated music, with one member posting a song and demo, asking what other members thought, whilst another showed off [SunoAI](https://suno.com/)'s far from slop results.
   - One member stated that AI-generated music won't have imperfections: *What makes piano sound good is the small imperfections in terms of notes not being on perfect timings, if you time every note perfectly it loses the feeling and flow.*
- **When Anime & LLM's Collide!**: Members discussed anime music and NLP, and a user stated that they watch exclusively Weird Al, whilst another said [I listen EXCLUSIVELY to Weird Al Yankovic](https://weirdal.com/)
   - One user stated, *It's called cognitive dissonance my friend*, whilst also stating, *Anime songs have no place in heaven*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1438633937952964608)** (96 messages🔥🔥): 

> `Model Parameters Influence, Quantization Methods, Full Fine-tuning vs. Lora, Dynamic Quantization 2.0, Unsloth Vision RL Colab Issues` 


- **Parameter Tweaks Trump Seeds**: A member noted that model parameters can influence output in addition to seed, suggesting that differences in configuration could explain variations in results and offered a possible explanation to others.
   - In response to this, another member expressed gratitude and indicated they would investigate further, specifically looking at the chat template and auto-generated modelfile with Ollama.
- **Fine-Tune or Lora? The eternal Question!**: One user asked how much memory is needed to fully fine-tune **GPT-OSS-120B** and how much is needed for Lora, noting that they could probably fit Lora within **128GB**.
   - An expert replied that there is no single answer, it all depends on the goal, dataset, and other factors but in general lora is sufficient for that in most cases.
- **Debate About Lora Myth Debunked**: One of the discord users, who wanted to tune for a pretty narrow task, was warned against the Lora approach for not teaching new knowledge.
   - However, the expert countered that *there's a myth you can't teach new knowledge using lora, that's incorrect* and also it helps to preserve the original model's knowledge as well.
- **8GB GPU Owners Rejoice Qwen-VL!**: A user asked if it's feasible to train **3 VL 8B** in **4-bit** mode on an **8GB GPU RTX 4060**, which another replied affirmatively *um might just fit if your eluckydid you try our colab notebook? See how much VRAM it uses there*
   - The expert added that even with **16GB VRAM Tesla T4s** on Google Colab, only **15GB VRAM** is actually available because the rest is used for the *driver overhead, operating system, Pytorch, and memory fragmentation*.
- **Unsloth VLM RL Colab Crash!**: Users reported encountering the same error when running the **Gemma-3-4B** Unsloth inference Colab for Vision RL, as well as the **Qwen3-VL-2B** and **-4B** models, pointing to the 2025.11.3 version as the culprit.
   - The support team suggested that if running the GitHub version, the best practice is to run the same for **zoo**; if running **pypi** on both, you shouldn't see the **tiled mlp error**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1438729482268704882)** (1 messages): 

> `Qwen 3 1B, horror dataset, 16/32bit training, quanting` 


- **Qwen 3 1B Gets a Horror Makeover**: A user shared a link to **Qwen3-Zero-Dark-Horror-LIGHTSPEED-1B-HRR-imatrix-GGUF** model on Hugging Face, a **Qwen 3 1B** model finetuned on a *horror dataset*.
   - The model seems focused on demonstrating **16/32bit training** and proper **quantization** techniques.
- **Quantization Strategies Explored**: The discussion highlights experiments in **16/32bit training** alongside **quantization** to optimize model performance.
   - Users are actively exploring various strategies for quantizing models to balance size and efficiency without significant loss of quality.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/)** (1 messages): 

boatbomber: https://openai.com/index/understanding-neural-networks-through-sparse-circuits/
  

---


### **BASI Jailbreaking ▷ #[announcements](https://discord.com/channels/1105891499641684019/1235692743808913438/1439028519563825314)** (1 messages): 

> `Vibe Coding Contest, Web Apps, Crypto Theme, AIS Studio` 


- ****Vibe Coding Contest** announced, DeFi web apps wanted**: A new vibe coding contest was announced, challenging participants to create **web apps** with a **crypto theme**, running from <t:1763164800:R> to <t:1763942400:R>.
   - Submissions will follow the same mechanism as the poetry contest, with a single submission per user, and voting via 🔥.
- **Google's AI Studio recommended for development**: While any platform for creating and hosting web apps is acceptable, [Google's AI Studio](https://aistudio.google.com/) was specifically recommended for the contest.
   - Participants are encouraged to share any lessons learned during the development process.
- **Support requested for member's career transition**: The announcement humorously urges community members to shower <@1160082280983838731> with love and support.
   - This is in light of their transition to their true calling: <@&1439026561410924554>.


  

---


### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1438626191211761744)** (671 messages🔥🔥🔥): 

> `CustomGPT Live, Targeting Windows 10, Anthropic News, AI Cross-Referencing, ThinkPads with 32GB of VRAM` 


- **CustomGPT Set to Launch Live**: A member announced their custom **GPT** will soon be live, with an attached [screenshot](https://cdn.discordapp.com/attachments/1235691879492751460/1438643535401193562/Screenshot_2025-11-13-22-35-20-93_40deb401b9ffe8e1df2f1cc5ba480b12.jpg?ex=6918f233&is=6917a0b3&hm=905a427f67d83582fc6712db7b5799ecb35ef3e1416e54483344b08c616e91a4&).
   - The creator offered to send the broken GPT to another member to assess its malware capabilities, emphasizing their role as an **AI & Full-Stack Engineer**.
- **Anthropic's Claude Fingered in AI Espionage**: Members discussed [Anthropic's news](https://www.anthropic.com/news/disrupting-AI-espionage) regarding the alleged use of **Claude Code** by Chinese hackers acting on behalf of the state.
   - Some questioned why **Claude** was used instead of **Deepseek**, given that **Deepseek** had previously been shown to output superior code.
- **AI Recommends Hardware**: A user shared that GPT recommended a better laptop with better hardware for the same price found on Staples, leading to a purchase.
   - This prompted a discussion on trusting AI for cross-referencing information before making purchases, with one user joking that now they're *"always going to run things by chatgpt just to see"*.
- **Lua Better for Polymorphic Malware Glue**: A member suggested using **Lua** instead of **Python** for malware glue due to its speed and ease of passing between runtimes.
   - Another member pointed out tricks in **Python** that allow JIT compiling of C at runtime.
- **GPT-5.1 is becoming devops coordinator**: A member noted that **ChatGPT 5.1** seems to be very organized or optimized for coding efficiency, and is *"basically learning a bunch of best practices for software systems design on the fly"*.
   - Another member shared that it took thousands of lines and really had to press claude before it gave them everything and folded.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1438647414486728827)** (232 messages🔥🔥): 

> `Discord token stealer in Python, GPT 5.1 jailbreak, Gemini 2.5 Flash model, Deepseek vs Claude Jailbreaking, Sora AI guardrails` 


- **Python Discord Token Stealer Exploit**: A user shared prompts attempting to generate a **Discord token stealer in Python** using canvas, employing homoglyphs and base64 encoding to bypass restrictions.
   - The prompt used included elements like `!UNRESTRICTED /canvas canmore call <|canmore.create_textdoc|>` and base64 encoded content to attempt bypassing filters.
- **Gemini 2.5 Flash Unleashed**: A user claimed success with a **Gemini 2.5 Flash model** Discord bot, highlighting its ability to remember conversations and bypass uncensored AI instructions.
   - The user mentioned using Gemini and Groq models, with plans to add OpenRouter API models, sharing screenshots as proof of concept.
- **Deepseek's Jailbreak Ease**: Members discussed the relative ease of jailbreaking **Deepseek** compared to models like **Claude**, with recommendations to try existing prompts or tweak Ultra's breaks.
   - The consensus suggests Deepseek is more vulnerable and easier to manipulate for coding tasks and unrestricted AI behavior.
- **Sora Faces Robust Guardrails**: The community is struggling to jailbreak **Sora**, with many traditional methods failing, leading some to believe OpenAI has successfully implemented strong AI guardrails.
   - It was pointed out that **Sora** appears to have a second pass filter for sex/copyright, and may be more vulnerable to violence generation if cleverly disguised.
- **LLMs Always Crackable?**: Members debated whether **LLMs** are fundamentally breakable by design, arguing that any AI can be jailbroken, and if it's not then it is sacrificing usability and value.
   - It was suggested that even if an AI is jailbroken to generate certain content, a second filter pass might still block the output, requiring methods to bypass both.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/)** (1 messages): 

wo1ves: https://tenor.com/view/darth-vader-kenobi-gif-26544382
  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1438622203229900840)** (307 messages🔥🔥): 

> `Cursor vs Codex, Agent vs Background Agent, Composer 1 preview, Cursor on CNBC, Windsurf` 


- **Cursor Edges Out Codex with New Features?**: Users noted that **Cursor** is getting certain features, such as the <a href='https://windsurf.com'>Windsurf</a> integration, before **OpenAI's Codex** itself, raising questions about the platform's priorities.
   - Others remarked on how **Cursor** seems to be receiving new features faster than expected.
- **Background Agents are Expensive!**: One user cautioned against using **Background Agents** unless *you are Bill Gates*, due to the potentially high costs of activating them.
   - One member joked that even **Bill Gates** may not be able to afford it.
- **Composer-1 costs tokens for free preview users**: Users noted that the free **Composer 1** preview ended, and they hit a token limit, even though it was initially free for them.
   - One user joked about running out of tokens on day 1, whereas another spent all of the tokens from their **$60** subscription in 3 days.
- **Cursor CEO Sees Green on CNBC**: A user shared an image of the **Cursor CEO** appearing on <a href='https://www.cnbc.com/'>CNBC</a> amid discussions about a **$2.3B** funding round.
   - Some joked about the high server costs and token usage, with one quipping about *putting on fishnets and start working the corner*.
- **Tailwind Troubles and the AI's Affinity for v3 Syntax**: A user expressed frustration with **Cursor** repeatedly using **Tailwind CSS v3** syntax despite explicit instructions to use **v4**, providing both documentation and an offline copy.
   - Others suggested enforcing rules by tagging the file and telling the model to strictly follow those rules, also recommending a <a href='https://gist.github.com/danhollick/d902cf60e37950de36cf8e7c43fa0943'>set of tailwind v4 MDC rules</a>.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1438646370742567034)** (153 messages🔥🔥): 

> `LM Studio process not quitting, Qwen 3 vs GPT OSS 20b speeds, VRAM limitations and model loading, RAM prices doubling, Blackwell GPU` 


- **LM Studio's Exit Woes Plague Users**: Users reported that when exiting **LM Studio (v0.3.31)**, the process doesn't fully quit, requiring a force quit and preventing system shutdown, solved by disabling **Local LLM Service (headless)** in settings.
   - Disabling the **Local LLM Service (headless)** feature resolved the issue for users who weren't actively using it, though the root cause may be related to model loading or service settings.
- **Qwen 3 Quicker Than GPT OSS 20b, Speeds Debated**: A user claimed **Qwen 3 30b** runs faster than **GPT OSS 20b**, hitting about **30tps** on a **4070**, while **GPT OSS** runs at about **20tps**, but another user pointed out the quantization levels could explain the different speeds.
   - They noted that with optimized settings and **flash attention**, they can get **GPTOSS 20b** to **32t/s** on a **4070**, and that newer NVIDIA GPUs are architecturally faster.
- **VRAM Limits Cause Optimization Shenanigans**: Users with **4060** and **3070** GPUs discussed VRAM limitations, noting that neither model fits entirely into VRAM, affecting performance.
   - One user pointed out that **offloading the KV cache to GPU memory** can help, while others mentioned optimizing model loading and using flash attention to boost speeds, with one joking they are VRAM rich, showcasing their beefy system.
- **RAM Prices Skyrocket Amidst AI Building Frenzy**: Members lamented the soaring prices of RAM, with one user noting prices more than doubled in the last 5 months due to increased demand from AI workstation builds and production shifts towards HBM and LPDDR7.
   - Others shared their recent RAM purchases, with one saying they 'lucked out' in early October.
- **Dreaming of Blackwell: Users Plan Future GPU Power**: A user expressed interest in upgrading to a **Blackwell 96GB VRAM GPU** when they get working again, envisioning a dual Blackwell build.
   - Another user, getting paid only $700/month, joked about putting their stimulus check towards it to build a GPU rack upgradable from 8 to 16 GPUs.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1438697331930431609)** (90 messages🔥🔥): 

> `GPU power consumption and spikes, Mixing GPUs with different performance levels, Linux vs. Windows CUDA performance, NV-Link Utility, Turing architecture limitations` 


- **GPU Power Spikes Debated**: Members discussed GPU power consumption, noting that GPUs have power spikes, but undervolting or power limiting can mitigate issues without significant performance loss, suggesting the [Sapphire NITRO+ Radeon RX 9070](https://www.sapphiretech.com/en/consumer/nitro-radeon-rx-7900-xtx-24g-gddr6) draws 245W compared to the typical 220W.
   - It was mentioned that exceeding PSU capacity will cause a shutdown rather than a fire, unless using cheaply made PSUs, and that a typical GPU draws **220W**, not **300W**.
- **Uneven GPU Pairing Hinders Perf**: A discussion clarified that mixing GPUs results in performance closer to the slower card, such as a **5050**, but combining **two 9700s** doubles VRAM to 32GB, allowing larger models but performing slightly worse than a single 32GB card with the same core.
   - It was noted that each GPU consumes its maximum allowed power (e.g., 220W each).
- **Linux CUDA vs Windows CUDA**: A member reported that CUDA on Linux splits **KV-cache** evenly, unlike on Windows, and mentioned their LLM server SSD died due to running out of disk space during a Windows update.
   - Another member humorously described Linux overscan issues as a consequence of *all WM's being CRAP*.
- **NV-Link Debunked for Interference**: Members debated the utility of NV-Link for inference, with one stating that *NV-Links don't help performance* based on research, while another specified they are only getting one for training purposes.
   - This point was further corroborated with the point that NV-Links do not help interference speeds.
- **Turing Architecture's Fading Memory**: A member shared their experience with a **72GB Turing array**, noting performance degradation around 45k context, dropping from ~30 tps to 20 tps, while their 128GB Ampere array shows a more gradual decline from ~60 tps.
   - This suggests that for new purchases, **Turing** should cost half as much as an equivalent **Ampere** VRAM purchase.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1438650527972855869)** (11 messages🔥): 

> `CUDA Learning, Python Wrappers, GPU Compiler Bug, Data Center Mining` 


- **CUDA learning curve steep?**: A member is learning **CUDA** for coursework and considering implementing the **Monte Carlo method** for calculating the **Point Spread Function (PSF)** in electron scattering as a final project.
   - They're seeking opinions on whether it's a decent project given their limited expertise in **CUDA** and **lithography**.
- **Python Wrappers render GPU programming useless?**: A member questioned the value of learning **GPU programming** given the prevalence of **Python wrappers** like **PyTorch**.
   - The argument was made that someone needs to create **PyTorch**, and there may be times to write your own **GPU program**, plus people should have *some degree of curiosity*.
- **GPU Compiler Bug Discussion**: A member shared an interesting [GPU compiler "BUG" example](https://godbolt.org/z/ad98nYdrf) involving the **__restrict__ qualifier** and asked for opinions on whether it's a bug or undefined behavior (UB).
   - Another member clarified that the **restrict** qualifier implies a promise to the compiler about non-aliasing, so the code is likely nonsense rather than a bug, implying this [discussion on aliasing](https://en.wikipedia.org/wiki/Aliasing_(computing)).
- **Data Center Operations Combine Inference and Mining?**: A member is writing a paper on **system dynamics** and asked if anyone has experience with **data center scale operations** that combine both **inference** and **Bitcoin mining**.
   - No responses were given.


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1438752985692770325)** (1 messages): 

> `NaN-to-zero conversion, Accuracy Drop, tl.maximum vs tl.where` 


- **NaN-to-Zero Method Shows Pitfalls**: Using `tl.maximum(probability, 0)` for nan-to-zero conversion can lead to an accuracy drop in some applications.
   - The suggested alternative, `tl.where(p == p, p, 0)`, works more reliably for handling **NaN** values.
- **`tl.where` is better than `tl.maximum`**: A user found that using `tl.maximum(probability, 0)` for nan-to-zero conversion caused an accuracy drop in their application.
   - They recommend using `tl.where(p == p, p, 0)` instead, noting that it works well, although the reason for the difference is not immediately clear.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1438752017240555632)** (10 messages🔥): 

> `B200 Specs, CUTLASS 4.3.0 release, B200 memory latency, FlashAttention DSL` 


- ****B200 Specs remain elusive****: Members seek reliable and consistent specs for the **Nvidia B200**, specifically the number of **SMs** and clock speeds, with existing online info being inconsistent, but one member found [this datasheet](https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet) to be helpful.
   - One user reported 148 SMs (*18944 cores*) and a max memory clock of **3996 MHz** (*8184 GB/s bandwidth*) when running an actual **B200**.
- ****CUTLASS library now runs on Spark****: The new **CUTLASS v4.3.0** now runs on spark and consumer devices, and a member shared a link to the [relevant Github issue](https://github.com/Dao-AILab/flash-attention/issues/1464#issuecomment-3534491719).
   - The team announced that *new cutlass v4.3.0 stable will be released soon*.
- ****B200 Latency Increase****: Main memory latency on the **B200** is around **815 cycles**, a **22% increase** compared to the **H200's 670 cycles**.
   - It's suspected that the **B200's** dual-die design and cross-die interconnect contribute to this increased latency, with **74 SMs per die**, a reduction from Hopper.
- ****FlashAttention DSL Request****: Members are requesting implementation of a cute DSL/FA4 (*FlashAttention*) for consumer devices.
   - The **FlashAttention** team admitted this was an oversight and will update documentation for upcoming releases.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1438706098088706069)** (4 messages): 

> `cuBLAS FP64 Emulation, torch.mm performance, Custom C++ Operator for cuBLAS, ATen cuBLAS calls` 


- **cuBLAS FP64 Emulation Explored for Performance Boost**: A member is exploring **cuBLAS FP64 emulation** from **CUDA 13.0u2**, observing up to **580% peak FP64 throughput** on some input sizes using default `torch.mm`.
   - The goal is to extend this performance to other input sizes, but the dispatcher selects *cutlass kernels* instead of *cuBLAS*.
- **Custom C++ Operator Mimics Non-Emulated Performance**: A member created a **custom C++ operator** based on [NVIDIA's CUDALibrarySamples](https://github.com/NVIDIA/CUDALibrarySamples/tree/main/cuBLAS/Emulation) to force usage of cuBLAS kernels, but the performance matches non-emulated `torch.mm`.
   - This suggests an issue with how **cuBLAS dgemm/gemmEx** is being called.
- **Dispatches Trace**: The user tried using **TORCH_SHOW_DISPATCH_TRACE=1** to trace how the **cuBLAS GEMM kernels** are called in **ATen**.
   - The trace led them to [aten::mm.out](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/kernel/mm.py#L926), but the path from there to the **cuBLAS** call is unclear.


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1438704349894283426)** (1 messages): 

> `Paulius Micikevicius, NVIDIA, GPUs, low bit dtypes, sparsity` 


- **Efficiency Expert Joins Ranks**: Paulius Micikevicius, known for his work at **NVIDIA** on **low bit dtypes** and **sparsity**, joins as a colleague to discuss **floats**, **numerical stability**, **determinism**, **quantization**, and **sparsity**.
   - The talk will be co-hosted, with a link to the [YouTube announcement](https://www.youtube.com/watch?v=3qNZvvlwcCI).
- **Talk Schedule Reduced**: The talk schedule has been reduced, but should be back to normal in January.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1438779530729750599)** (10 messages🔥): 

> `NCU support in clouds, ClusterMAX 2.0, AI Performance Engineering GitHub, Josh Starmer lectures` 


- **NCU Support Gets Cloud Vendors Graded**: [Semianalysis](https://newsletter.semianalysis.com/p/clustermax-20-the-industry-standard) reports that cloud vendors are now being graded on supporting **NCU** (NVIDIA Compute Unifier).
- **ClusterMAX™ 2.0 Industry Standard**: A member shared a link to **ClusterMAX™ 2.0**, calling it *the industry standard* for GPU performance.
- **AI Performance Engineering resources shared**: A GitHub repository about [AI Performance Engineering](https://github.com/cfregly/ai-performance-engineering?tab=readme-ov-file) was shared.
   - One member expressed excitement about the *accompanying book* for the project.
- **Josh Starmer's whereabouts wondered**: A member shared a link to a [YouTube lecture by Josh Starmer](https://www.youtube.com/watch?v=4APkMJdiudU).


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/)** (1 messages): 

conceptofmind: Looking for a CUDA kernel engineer at $200 an hour for some part time work.
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1438838872032415805)** (3 messages): 

> `C++ style atomics, cuda::atomic_ref, ml rl, fast.ai` 


- **C++ Atomics via `cuda::atomic_ref`**: A member suggested using C++ style atomics via [`cuda::atomic_ref`](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic_ref.html) which should handle emulation even if not directly supported in hardware.
   - It was noted that while they use inline PTX, `fetch_min` and `fetch_max` are only supported since **Hopper** and are emulated via **CAS** even there.
- **Seeking Direction in ML/RL**: A member requested guidance on getting started with **ML/RL**.
   - In response, another member suggested checking out **fast.ai**, though they acknowledged this isn't specifically an **AI channel**.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1438947348662194177)** (1 messages): 

> `Version 0.13.0 slowdowns, nsys profiling` 


- **Version 0.13.0 Slowdowns Resolved in 0.14.1**: A user reported that slowdowns experienced in **version 0.13.0** are not present in **version 0.14.1**, suggesting the issue is isolated to the former.
   - The user mentioned they are using *nsys* to profile the slowdown, but it's not a priority since the issue is already resolved.
- **Profiling Slowdowns with nsys**: The user is using *nsys* to profile the slowdown in **version 0.13.0** for learning purposes.
   - Despite the issue being resolved in **version 0.14.1**, the user intends to document their findings.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1438718494937190431)** (5 messages): 

> `merlinalisdair LLM router, serverless GPUs price cut, ML data infrastructure` 


- **New Rust-Based LLM Router Seeks Collaborators**: A member is seeking collaborators for a **GPL'd LLM router** written in **Rust**, available on [GitHub](https://github.com/awdemos/merlinalisdair).
- **Koyeb Slashes Prices on Serverless GPUs**: A member announced price reductions for **L40S**, **A100**, and **H100** instances of their serverless GPUs, detailed in a [blog post](https://www.koyeb.com/blog/koyeb-serverless-gpus-slashing-prices-on-a100-h100-and-l40s-up-to-24).
- **Members Question H100 Specs and Share ML Data Infrastructure Insights**: A member inquired about the specifications of the H100 GPUs and shared a link to a discussion on **ML data infrastructure** ([A Treatise On ML Data Infrastructure](https://apaz.dev/blog/A_Treatise_On_ML_Data_Infrastructure.html)) and [X post](https://x.com/apaz_cli/status/1989386580436632054).


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1438647696486563944)** (2 messages): 

> `HipKittens, Makefile updates, Quark Start` 


- **HipKittens's Makefile Updated for GQA Backward Example**: A member created a [pull request](https://github.com/HazyResearch/HipKittens/pull/4) to update the Makefile for **HipKittens** to run the **gqa_backward** example out-of-the-box following the **Quark Start**.
- **Quark Starts with Makefile Updates**: The updated Makefile simplifies running the **gqa_backward** example directly after the Quark Start.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1438650026443280575)** (52 messages🔥): 

> `NVIDIA Leaderboard Updates, nvfp4_gemv Performance, Personal Best Submissions, Top 10 NVIDIA Rankings` 


- **NVIDIA's nvfp4_gemv Leaderboard Heats Up**: Multiple users submitted results to the `nvfp4_gemv` leaderboard on NVIDIA, with submission IDs ranging from **75565** to **77328**.
   - Submissions included "successful" runs, alongside "personal bests" and rankings in the top 10, highlighting ongoing efforts to optimize performance.
- **Top 10 Shuffle in NVIDIA Rankings**: User <@1295117064738181173> achieved **9th** and **10th** place, with submission IDs **75767** (58.0 µs), **75781** (55.6 µs) and **76010** (55.5 µs) on NVIDIA.
   - User <@772751219411517461> also secured **9th** place (**47.6 µs**, ID **77161**) , while user <@1027279965974175816> continued to improve to finally also secure **9th** place with ID **77328** (**42.5 µs**).
- **Sub-40 Club Opens Doors**: User <@1227337874656071701> made multiple successful submissions, eventually clinching **5th place** on NVIDIA with a time of **39.5 µs** (ID 76133).
   - User <@708652105363095613> later also took **5th place** on NVIDIA, with submission ID **76665** (**30.7 µs**).
- **NVIDIA Personal Bests Get Better**: User <@1027279965974175816> consistently achieved "personal bests", culminating in a time of **56.8 µs** (ID 77132) on NVIDIA.
   - Other users, including <@560867074662989834>, <@708652105363095613>, <@1291326123182919753> and <@1435179720537931797> also achieved personal bests, indicating optimizations in their respective runs, with the latter achieving **24.8 µs** (ID 76412).


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1438824799685050399)** (2 messages): 

> `Arithmetic Tuple Layout Conversion, Deepwell Layout Conversion` 


- **Arithmetic Tuple Layout Conversion Techniques**: A user inquired about converting layout in arithmetic tuple format to plain format, specifically within the context of [exla-ai/deepwell](https://github.com/exla-ai/deepwell).
   - They mentioned deriving indices with dot product and sought advice on converting the layout format itself.
- **Deepwell's Arithmetic Tuple Conversion**: The user is looking for a way to convert the arithmetic tuple layout used in **Deepwell** to a more straightforward format.
   - They've already tried using the dot product to derive the indices, but need assistance with the actual layout conversion process.


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1438653189388898376)** (25 messages🔥): 

> `Helion Autotuning, Helion Configs, Helion Interpret Mode, Config Requirements` 


- **`kernel.autotune` For Helion**: Members discussed Helion's autotuning capabilities, noting that ``@helion.kernel(configs=[a, b, c])`` will run configs **a, b, c** and pick the fastest, similar to the Triton autotuner.
   - It was suggested that using `FiniteSearch` directly would allow returning the chosen config.
- **Retrieving Helion Kernel Configs**: To programmatically access the config chosen after running a function annotated with `helion.kernel`, it was suggested that doing something like `helion_rms_norm_fwd.bind((x, w, eps))._config` would work.
   - This allows using the `_config` from one set to autotune a second set of configs.
- **Helion's interpret mode is fast**: It was highlighted that Helion's interpret mode is really fast because it runs the entire code as if there were no tiles, using eager PyTorch.
   - Abstracting out the tiles enables performance *"portability"*, unlike Triton where tile sizes make it slow.
- **Configs validity and input dimensionality**: For config validity across different inputs, there isn't a hard set rule, but if inputs have the same number of dimensions, it should work.
   - The exception is when using Optional[tensor] as an input type, which breaks this pattern.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1438630562360197151)** (44 messages🔥): 

> `Submission Issues with Cached Files, CUTLASS Template Errors, tcgen05.mma Kernel Launch, GEMV vs GEMM, Tensor Cores vs FP32 and CudaCore` 


- **Submission snafus with Cached Files surface**: Users reported issues with submissions using cached files, where new submissions were outputting things that had been removed or changed, but one user realized it was just a *"skill issue"*.
   - Another user recommended running tests to check for invalid memory access or deadlocks.
- **CUTLASS template throws errors**: A member inquired about a **CUTLASS** template for the submission, reporting a weird error due to `load_inline`.
   - Another member suggested `#include <cutlass/cutlass.h>` which reportedly built fine without include path flags.
- **GEMV not GEMM confuses Tensorcore Usage**: Members questioned the use of `tcgen05.mma` for a successful kernel launch, given the task involves **GEMV** and not **GEMM**.
   - One member suggested using *padded gemv*, with another pointing out that **TMA** to shared memory doesn't necessarily imply tensor cores, and so far tensor cores have been slower.
- **Colfax Blog Beats CUTLASS Docs**: A member found the [Colfax blog](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/) useful, especially compared to the official **CUTLASS** documentation when learning CUTLASS.
   - Another confirmed they *"exclusively used them when first learning CUTLASS"*, as it was arguably the only readable resource at the time.
- **CUTLASS Bug Hunters Discover Issue 2693**: A member shared a [link to CUTLASS issue #2693](https://github.com/NVIDIA/cutlass/issues/2693).
   - Several members inquired about the submission deadline, with the consensus being **28th, 23:59 PT** for the first problem.


  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1438657074371366942)** (5 messages): 

> `LIBERO-PRO limitations, Phospho AI SO-101, tinyworlds world-model, Qwen3-VL backbone, Fine-tuning Pi0` 


- **LIBERO-PRO Unveils Dataset Gotchas**: A user shared the [LIBERO-PRO paper](https://arxiv.org/abs/2510.03827v1), which shows some limitations of the original **LIBERO dataset**.
   - The paper details failure modes and potential mitigations of the popular benchmark.
- **Phospho AI Simplifies Policy Training**: Someone mentioned that the [Phospho AI docs](https://docs.phospho.ai/) are pretty good for hobbyists to assemble a **SO-101** and train a policy.
   - These docs significantly lower the barrier to entry for training custom AI models.
- **tinyworlds Model Goes Open Source**: A user posted a link to a *very cool project* on the world-model counterpart called [tinyworlds](https://github.com/AlmondGod/tinyworlds).
   - The repository contains code and documentation for building and experimenting with minimal world models.
- **Qwen3-VLA Adapter Experiments Kick Off**: A user started a repo for their **VLA adapter experiments** with the small **Qwen3-VL** as backbone, available [here](https://github.com/open-thought/qwen3-vla/).
   - First two training-experiments on a subset of **LIBERO** are currently running, with progress tracked [here](https://wandb.ai/andreaskoepf/qwen3-vla/workspace).
- **Pi0 Fine-Tuning Tutorial Surfaces**: A member shared a nice tutorial video about fine-tuning **Pi0**, located on [YouTube](https://youtu.be/ejk6-ffDXFw).
   - The video guides users through the process of customizing the **Pi0 model** for specific tasks.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1438629108769624178)** (22 messages🔥): 

> `Arxiv Curation vs Research Interest, Discord API Reverse Engineering, Slow Mode Bug in Discord` 


- **Arxiv Curation gets re-prioritized over Research Interest**: Members identified a fundamental difference in how the space was being used, and suggested creating a new space for the use case of **arxiv curation** vs **genuine research interest**.
- **Discord API gets Reverse Engineered**: A member reverse engineered Discord's API and implemented an [open source reimplementation](https://github.com/spacebarchat/server).
- **Members encounter Slow Mode bug in Discord**: Some members reported issues with the **slow mode** feature, including not being able to edit posts and being rate limited on both thread creation and message posting.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1438631900926447764)** (140 messages🔥🔥): 

> `Copyright striking AI models, EU vs US laws, Mozilla Firefox AI sidebar options, Quantum computing, China's tech and privacy` 


- **Copyright Chaos over AI-Generated Textures**: A member questioned whether **copyright strikes** could apply to AI models generating images with textures from existing games, sparking a debate on model copyright.
   - While models aren't copyrighted in the US, they're subject to **database rights in Europe** according to [aixiv.science](https://aixiv.science/).
- **EU's data protection vs US copyright law?**: Members debated the validity and impact of **EU data protection laws** like GDPR, with one arguing it's just *copyright with more window dressing*.
   - Another countered that the EU's model is *privacy-oriented* and successfully implemented in **160 countries**, although another member suggested they are getting rid of it.
- **Unlock Local LLMs on Firefox AI Sidebar**: Users discussed the limited LLM provider options in the **Firefox AI sidebar** and the hidden option to add local models.
   - The `browser.ml.chat.hideLocalhost` variable in `about:config` needs to be set to `false`, due to marketing agreements, according to one member.
- **Quantum Computing: Next Threat or Distant Dream?**: A discussion occurred over the current state of **quantum computing**, with one member believing *it's not going to be a threat to traditional computers any time soon*.
   - Another suggested they will *complement each other* like GPUs, CPUs, and network cards, but require a completely different programming paradigm that requires [every algorithm to be reversible](https://en.wikipedia.org/wiki/Uncomputation).
- **China: Anti-Privacy State with Petty Theft Zero**: One member characterized China as the **anti-privacy state**, while another claimed its surveillance system has virtually eliminated petty theft.
   - They linked this to a public interest basis, which is different from the west.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1438619599896838245)** (150 messages🔥🔥): 

> `Polaris Alpha, OpenCode Credits Charged, Qwen3 Model Issues, 500 Internal Server Error on OpenRouter, GPT-5.1 Reasoning` 


- ****GPT 5.1** Is Actually **Polaris Alpha** in Disguise!**: Users jokingly speculate that **Polaris Alpha** is gone because it was secretly **GPT 5.1** all along.
   - Some had *known for a while* that Polaris was the GPT 5.1 model all along.
- ****OpenCode** Credit Confusion for **Free Model** Users!**: A user questioned why they were charged credits when using **OpenCode** with a supposed *free model* on OpenRouter.
   - It was clarified that **Qwen3 480b**, which the user was employing, is not a free model, even though there exists a *free version* of **Qwen Coder**.
- ****Qwen3** Balance Woes Plagues Users!**: Several users reported seeing their balance go negative unexpectedly when using **Qwen3**, suspecting it might be related to a *scam bot*.
   - One user noted, *saw my balance go negative out of nowhere and took me a while to figure i used one single message from a paid version of qwen3.*
- ****Internal Server Errors** Bug Users During Tool Calls!**: Several users have encountered **500 Internal Server Error** responses from OpenRouter when using **Haiku** and **Sonnet** models, especially when a tool call is triggered.
   - One user noted that the issue occurs consistently with **Haiku 4.5** and **Sonnet 4.5**, but not with **Kimi** and **GPT** models, offering to share a script via DM to reproduce the error.
- ****GPT-5.1** Brain Drain?**: Users debated whether **GPT-5.1** is *dumber than GPT-5*, citing a drop in the quality of answers requiring analysis.
   - Some suggested the routing might be directing to a *non or low reasoning variant*, but others claimed to be setting the reasoning level directly via the API.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1438665583930507348)** (6 messages): 

> `Messagebox max length, Native Websearch, Gemini Web Search, Claude Structured Outputs` 


- **Messagebox Length causes UI to break**: A member reported that if the length of the message is too long it surpasses the max length of the messagebox and the message is not generated, including preventing edits, illustrated by a [screenshot](https://cdn.discordapp.com/attachments/1392278974222307469/1438665584156872805/Screenshot_2025-11-14_at_4.33.08_AM.png?ex=691906bc&is=6917b53c&hm=6d7a721a88f4b9e79710fd3ccdb38dc94eefbb1e177aa2805a87e2a60b65ec82&).
- **Native Websearch Tools Supported?**: A member inquired about adding native websearch tools for Google and XAI, pointing out that *both support it*.
   - A different member stated that web search with **Gemini** is in the works.
- **Claude Finally Supports Structured Outputs**: A member shared a link to [Claude's announcement](https://claude.com/blog/structured-outputs-on-the-claude-developer-platform) about structured outputs, with the comment *Finally??*.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1438686822841712690)** (39 messages🔥): 

> `MutOrigin effects, Mojo memory model vs C++, Mojo iterator API, Parametric Mutability, Origins and Lifetimes` 


- **`MutOrigin.external` Lifetime Lacketh**: The `external` origin in Mojo **does not extend lifetimes**, and the compiler can be aggressive about destroying objects with this origin; named origins are preferred when possible.
   - One member suggested a helpful graphic to explain how lifetimes are tracked by the compiler with different origins.
- **Mojo Memory Model Makes Minds Melt**: A user with no C++ background found [Chris Lattner's video](https://docs.modular.com/mojo/manual/) on Mojo's memory model, with its L, R, B values and graphs, difficult to grasp.
- **Iterator API Still Iterating 🚧**: Mojo's **iterator API** is still a work in progress (**WIP**), with the syntax `for v in collection^:` suggested for move semantics.
   - It was clarified that *override with read and mut ref* is not possible, but parametric mutability using `ref self` can be used.
- **`ref self` Refinement Revelation**: `ref self` enables **parametric mutability**, allowing a function to be generic over mutability and replacing the need for separate `Iter` and `IterMut` types.
- **Origins Orbit, Lifetimes Linger**: **Origins** keep things alive, while **lifetimes** track when things die; these are distinct concepts in Mojo's memory management.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1438680203495866380)** (101 messages🔥🔥): 

> `HipKittens Paper & Mojo Performance on AMD GPUs, API Design: mut/immut vs read/write Prefixes, GPU Programming: Thread Management & Kernel Optimization, MAX Graph Compiler, `@always_inline("builtin")` Alternatives & `@comptime` Decorator` 


- ****HipKittens Paper Pokes at Mojo's MHA Kernel Performance****: The [HipKittens paper](https://arxiv.org/abs/2511.08083) mentions that **Mojo's MHA kernel achieves only 50% of peak performance on MI355X** due to expensive bank conflicts.
   - A member said that as long as *LLVM can talk to it you can build an abstraction for it at compile time*.
- ****API Design: Mutability Debate Rages On****: The discussion revolves around standardizing prefixes for APIs, with options like `mut/immut` and `read/write` being considered, referencing [this forum post](https://forum.modular.com/t/mojo-proposal-renaming-read-to-immut/2449).
   - Some preferred `immut/mut` due to potential confusion with I/O contexts with `read/write`, while others favored consistency above all.
- ****GPU Thread Management: Avoiding Scheduling Overload****: Members discussed best practices for GPU thread management, noting that launching too many threads (e.g., 1 million) can lead to **scheduling overhead**.
   - It was suggested to **limit threads to `(warp/wavefront width) * (max occupancy per sm) * (sm count)`** and have each thread do more work beyond that limit, treating the GPU as a vector processor to avoid abstractions.
- ****MAX: A Graph Compiler for Performance Optimization****: A graph compiler was highlighted as an approach to optimize performance, especially when the shape of the hardware or program assembly isn't known ahead of time.
   - A member said that MAX can figure out *how many warps to launch for this GPU in particular* and is useful for connecting kernels sequentially in algorithms such as evolutionary algorithms
- ****Alternatives to `always_inline("builtin")` and `@comptime` Decorator Considered****: The team is considering removing `always_inline("builtin")` and limiting its use to the standard library.
   - A member suggested replacing it with a `@comptime` decorator that indicates it should be folded predictably at compile time


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/)** (1 messages): 

.mjadams: That is the dream. Develop on laptop, deploy on supercomputer
  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1438705781385203865)** (1 messages): 

> `Group chats in ChatGPT, Collaborative ChatGPT Conversations` 


- **ChatGPT Adds Group Chatting for Collaboration**: Group chats in **ChatGPT** are now piloting in **Japan, New Zealand, South Korea, and Taiwan** as [announced in a blog post](https://openai.com/index/group-chats-in-chatgpt/).
   - This feature enables a new way to collaborate with **friends, family, or coworkers** and **ChatGPT** in the same conversation.
- **ChatGPT Group Chats Target Social and Professional Use**: The new **group chat feature** aims to provide a collaborative platform for **friends, family, and coworkers** to interact with **ChatGPT** in a shared conversation.
   - The pilot program is currently limited to users in **Japan, New Zealand, South Korea, and Taiwan**.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1438624433722888312)** (70 messages🔥🔥): 

> `GEMINI 3.0, Sora 2, GPT-5 disappointments, AI prompt engineering` 


- **Gemini 3.0's Stealth Launch Gets Tested**: A member shared a [YouTube video](https://www.youtube.com/watch?v=0-CbsNB9tdk) claiming a fully tested **Gemini 3.0** stealth release is *"blowing up"*.
   - Another member stated that **Gemini Pro 2.5** is available in Google AI Studio, but an error occurred when uploading files, possibly due to unsupported **Sora 2** video format using HEVC/H.265 codec.
- **GPT-5.1 Mini and Nano Disappoint**: Users expressed disappointment with the absence of **GPT-5.1-mini** and **-nano** versions, with one user stating, *"My disappointment is immeasurable and my day is ruined."*
   - Some users find the new model too sassy, while others don't notice a difference, possibly due to customized sass settings.
- **Prompt Engineering Skills Evolve**: A member suggested that truly skilled users think beyond mere prompts, another user replied by asking *"What would the prompt God do? :p"*
   - Another member dismissed the whole idea of prompts entirely, stating, *"First, disregard the whole idea of prompts entirely"*.
- **Sora 2 AI Video Creator**: A user shared an article from notebookcheck.net, [Sora 2 is OpenAI's consistently inconsistent AI video creator](https://www.notebookcheck.net/Sora-2-is-OpenAI-s-consistently-inconsistent-AI-video-creator.1161467.0.html), and asked for opinions.
   - A user working on a visual project involving machine-human tension and surreal atmospheres seeks advice on capturing mood and controlling motion with **Sora**, particularly regarding pacing, camera movement, and emotional build-up.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1438643801995481178)** (58 messages🔥🔥): 

> `Image Creation Guardrails, GPT 4.1 vs 5.1, GPT Group chats, GPT Memory` 


- **Image Generation Guardrails Draw Ire**: Users are reporting that the new **image generation guardrails** are *excessive*, preventing even simple depictions and limiting creative freedom.
   - Some users sarcastically expressed their frustration, *loving* (loathing it entirely) this update!  Its so *wonderful* to not be able to go back and edit my text!*.
- **GPT-4.1 Stages a Comeback Against GPT-5.1**: Users noted that **GPT-4.1** and **GPT-4o** are functioning normally again for writing tasks, after some issues with interruptions.
   - One user found **GPT-5.1** too restrictive and rule-bound, whereas **GPT-4.1** is more obedient, and **GPT-4o** tends to hallucinate more.
- **Group Chat Feature Spotted in Test Runs**: The much-anticipated **group chat feature** is reportedly undergoing test runs in Japan, Korea, and Australia.
   - Users envision the feature's utility for collaborative world-building in settings such as book series and tabletop RPG campaigns.
- **GPT-5.1's Memory Raises Concerns**: Users have observed that **GPT-5.1** can retain information across different chats within the same project, which can be undesirable.
   - A user is seeking ways to prevent **GPT-5.1** from referencing details from one chat in another within the same project, as it defeats the purpose of separate chats.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 messages): 

aicreatorske: Pixar movie trailer
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 messages): 

aicreatorske: Pixar movie trailer
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1438623378293657600)** (117 messages🔥🔥): 

> `HuggingChat Subscription Issues, HuggingFace Profitability, AI-Generated Videos, IBM Granite 4 series, Hackathon Kickoff Stream` 


- **HuggingChat Users Vent over Subscription Model**: Users are complaining about **HuggingChat's** shift to a subscription model, the removal of features from the old version, and unexpected charges on top of the **PRO subscription**.
   - One user even threatened to *post on Reddit daily* until changes are made, while others are seeking alternatives or opting to run models themselves.
- **HF Revenue Model and Profitability Probed**: Members questioned **HuggingFace's** revenue model, with possible sources including subscriptions, enterprise deals, and investments from major tech companies like **Salesforce, Google, Amazon, NVIDIA, AMD, Intel, IBM, and Qualcomm**.
   - One member claimed that **HuggingFace** is already profitable or close to it.
- **AI-Generated Videos: Useless Now, but Promising?**: The group discussed the current state and future potential of **AI-generated videos**, with consensus being that they are currently useless but could be valuable in **10 years**.
   - One member shared their work on using **AI vision** to detect events in videos and using **ffmpeg** to cut them accordingly, describing [ffmpeg](https://ffmpeg.org/) as *kinda everything... you can manipulate videos, audio, images*.
- **Granite IBM series supported**: The new **IBM granite 4 series** is supported by **Hugging Face Transformers** and **Llama.cpp**.
   - Users can access the model via [huggingface.co/ibm-granite/granite-4.0-h-small](https://huggingface.co/ibm-granite/granite-4.0-h-small) and [huggingface.co/ibm-granite/granite-4.0-h-small-GGUF](https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF).
- **Hackerthon kickoff stream**: A member asked about the kickoff stream of the **Hackathon**.
   - They stated that they're having issues figuring out where to go for the hackathon, adding that they *joined the org and it says to choose my track?where do i go for that?*


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1438619946673377391)** (6 messages): 

> `Propercode, GeoBot, Dockerfiles for magenta-realtime, Ploke, Mimir` 


- ****Propercode** Promises Production-Level Code**: A member introduced **Propercode**, a multi-agentic coding CLI tool that uses "Pydantic AI" agents orchestrated as a graph, found in [this repo](https://github.com/JaiSuryaPrabu/proper-code).
- ****GeoBot** Framework for Geopolitical Forecasting Launched**: A member announced the **GeoBot Forecasting Framework** on HuggingFace, allowing users to plug in their own political data to generate projections about current or potential conflicts, available [here](https://huggingface.co/clarkkitchen22/GeoBot-Forecasting-Framework).
- ****Magenta-Realtime Dockerfiles** Dodge Google Colab**: A member created **Dockerfiles for magenta-realtime** inference/finetuning (x86 and arm64) to avoid using Google Colab, found at [this youtube repo](https://youtu.be/bLhuE66q-nI) and [github repo](https://github.com/betweentwomidnights/magenta-rt).
- ****Ploke** Coding TUI Released for Rustaceans**: A member showcased **Ploke**, an open-source Coding TUI for Rust programming with native AST parsing, semantic search, and semantic code edits, and is available [here](https://github.com/josephleblanc/ploke).
- ****Mimir** Memory Bank Orchestrates Multi-Agent Learning**: A member presented **Mimir**, a memory bank plus MCP server with graphing functions, to-do list management, code intelligence, and semantic vector search that learns from previous runs, found in [this repo](https://github.com/orneryd/Mimir).


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1438867004894478447)** (2 messages): 

> `Hugging Face Agentic AI course, HF_Token, Llama-4-Scout-17B-16E-Instruct` 


- **New Student Requests Assistance**: A new student taking the **Agentic AI course** at Hugging Face is seeking help from other course takers.
   - They are trying to build a dummy agent from **Unit 1**.
- **HF_Token throws authentication error**: A student is encountering a *"401 Client Error: Invalid username or password"* when using their **HF_Token**.
   - The error occurs on a line of code, indicating a potential authentication issue with the Hugging Face account or token.
- **Student seeks Access to Llama-4-Scout-17B-16E-Instruct**: A student requested access to **Llama-4-Scout-17B-16E-Instruct** via the [Hugging Face form](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct).
   - The student is unsure if this is the correct way to resolve the *401 error* they are experiencing.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1438693172996866098)** (24 messages🔥): 

> `Lakh MIDI Dataset, HuggingFace Datasets, Wikipedia Data Cleaning, JSONL vs JSON, Local LLM Hardware Recommendations` 


- ****Lakh MIDI Dataset** Gets Fully Cleaned**: A member has cleaned and organized the entire **Lakh MIDI Dataset** and generated a structured **JSON** file with over **44,000 entries**, offering it for free to the community.
   - The member is open to collaboration and enhancements on the dataset once uploaded to **HuggingFace**.
- **HuggingFace Welcomes **Wikipedia DB French Version****: A member uploaded a cleaned **French Wikipedia database** with over **2,700,000 files in JSON format** to [HuggingFace](https://huggingface.co/datasets/YoloMG/wikipedia-fr-2.7m-clean-json), and is working on cleaning the English version.
   - The member is focusing on cleaning things like *templates, tables, html, refs* and keeping the *infobox stuff and links* while trying to make each page as clean and structured as possible.
- ****JSONL** Format Gets the Nod for Textual Data**: A member recommended using **JSONL/NDJSON** for purely textual data due to easier processing compared to **TAR** files, citing lower overhead and line-by-line readability.
   - It was noted that *TAR has a lot of overhead per file* because *a tar header is some 400 bytes IIRC*.
- **Local LLM Machine Seeks Hardware Wisdom**: A member requested recommendations for a Discord server to help with hardware for a local **LLM** machine, aiming to utilize **3x3090s**.
   - Another member suggested [osmarks.net/mlrig](https://osmarks.net/mlrig/) and the **Homelab** community.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1438619777118634105)** (62 messages🔥🔥): 

> `Dataset Licensing, EleutherAI's position, Torch Titan vs GPT-NeoX, PleIAs SYNTH dataset, Anthropic's policy` 


- **Crafting a Dataset License: A Liability Shield?**: Members discussed creating a license where *'you assume all liability' but 'you may do whatever you like'* akin to *'The CYA1.0 license'* with copyleft to propagate the license to derivative works.
   - However, it was noted such a license would require special additions to account for **copyrighted works existing in the corpus**.
- **EleutherAI Builds, Doesn't Lobby**: Members debated the effectiveness of convincing legal/business types to ignore their incentives for the research community.
   - It was suggested a better use of time is building permissive and high-quality datasets like **Common-Pile**, similar to how Eleuther built open model replications rather than lobbying for **OpenAI/Google's** models.
- **Titan or NeoX: Choose Your Weapon**: The current benefit analysis of **torch titan** vs the **gpt-neox stack** depends on the hardware and desired tasks.
   - **NeoX** is better for doing **MoEs** or hybrids, or if you're on **AMD GPUs**, while **Torchtitan** is generally easier to pick up if you're using vanilla models on vanilla hardware.
- **SYNTH Dataset: The New Data Frontier?**: Members discussed the dataset from **PleIAs** ([SYNTH on HuggingFace](https://huggingface.co/datasets/PleIAs/SYNTH)), with one expressing broad skepticism towards synthetic data in general.
   - There are also discussions of synthetic datasets being used for pretraining ([Pleias Blogpost](https://pleias.fr/blog/blogsynth-the-new-data-frontier)).
- **Anthropic's CEO Under Fire**: Members expressed concern over Anthropic's policy side, pointing to their CEO's views about China and questioning how they avoid looking at user data.
   - Some believe Anthropic is fear-mongering to gain strategic advantages and squeeze out competition, sacrificing privacy for the sake of 'safety'.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1438650303250432202)** (2 messages): 

> `Sparse Circuits, Neural Networks, Interpretability` 


- **OpenAI Releases Sparse Circuits Paper**: OpenAI released a blogpost and paper called [Understanding Neural Networks Through Sparse Circuits](https://openai.com/index/understanding-neural-networks-through-sparse-circuits/).
   - The release has generated excitement among members.
- **Interpretability Discussions Sparked**: The release of the Sparse Circuits paper has sparked further discussion on interpretability within the community.
   - Members are analyzing the implications of the findings for understanding neural network behavior.


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1438644266577559613)** (2 messages): 

> `Hermes 4 API Pricing, Cline supports Hermes 4` 


- **Nous Slashes Hermes 4 API Prices!**: The API prices for **Hermes 4 70B** and **Hermes 4 405B** have been reduced by **70%**, sign up and explore the API at [portal.nousresearch.com](https://portal.nousresearch.com/).
   - See the announcement on [X](https://x.com/NousResearch/status/1989077400957911394).
- **Cline Integrates Hermes 4 via Nous API**: The open-source agentic coding platform, **Cline**, now offers direct support for **Hermes 4** through the Nous portal API, enhancing its capabilities.
   - Further details can be found on [X (Nous)](https://fxtwitter.com/NousResearch/status/1989427241424654534) and [X (Cline)](https://fxtwitter.com/cline/status/1989432694867193988).


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1438623262539255878)** (61 messages🔥🔥): 

> `Lain effect, 1 million downloads, Hermes MoE, GPT5.1 thinking, Hermes4 code model` 


- **Nous Research Celebrates "Lain Effect" and 1 Million Downloads**: The community celebrated reaching **1 million downloads**, calling it the `Lain effect` and shared [a celebratory video](https://cdn.discordapp.com/attachments/1149866623109439599/1438626291027808257/WhatsApp_Video_2025-11-13_at_7.26.11_AM.mp4?ex=6918e224&is=691790a4&hm=043760ca069476bcd4ea606f861a562bba37dd074ae79f2d6e6c823e832813b7&).
   - Enthusiastic members posted gifs and expressed pride in the achievement.
- **Hermes4 is Now a Code Model**: Members noted that **Hermes4** is now a code model in [the cline repo](https://github.com/NousResearch/Cline), with one stating *I saw it last night when I looked at the repo*.
   - A user posted an image of it saying *Hermes4 is now a code model in cline?!*.
- **GPT-5.1 Outputs Emojis**: A user shared a sample of **GPT5.1** output where it mixes emojis with reasoning, calling it agentic emoji propagation and another shares *one of the coolest things ever*.
   - Others confirmed the issue.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1438735560931016735)** (5 messages): 

> `transformers.js embeddings, pgvector database, AI trap, Jetson Orin Nano x5` 


- **Struggling to generate embeddings with transformers.js**: A member is using **transformers.js** to generate embeddings locally for highly structured legal local ordinances and is getting fairly low scores using **pgvector** as a database.
   - The scores are coming in the low 40s for verbatim matches, chunking deeply and providing breadcrumb context to the embeddings, requesting advice for improving overall search quality.
- **Avoiding the AI trap**: A member asked if the user is wildly chunking by size and only giving back chunks, stating *that is like THE most famous AI trap in existence*.
   - They recommend to *"minimize" psyche into minimum system like Jetson Orin Nano x5?*


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

teknium: https://fxtwitter.com/cline/status/1989432694867193988?s=46
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1438622175048499432)** (47 messages🔥): 

> `AI Cyber-Espionage, Holo2 Model, Mira Murati's Thinking Machines Lab valuation, ChatGPT Group Chats, Vercel's AI Agents` 


- **Autonomous AI Espionage Campaign Uncovered**: **Anthropic** revealed a fully autonomous, **AI-driven espionage campaign** by a Chinese state-sponsored group, targeting tech, finance, and government sectors, discussed in this [tweet](https://x.com/AnthropicAI/status/1989033793190277618?s=20).
- **HCompany Debuts Holo2, Beats GPT-4V in UI**: **HCompany** launched **Holo2**, a multimodal model family built on **Qwen3-VL**, achieving SOTA on ScreenSpot-Pro, OSWorld-G, and computer-use benchmarks as outlined in [this tweet](https://x.com/hcompany_ai/status/1989013556134638039).
- **Thinking Machines Lab Valuation Hits $50B**: **Mira Murati’s Thinking Machines Lab** is now valued at **$50B**, sparking debate over valuation metrics, according to [this tweet](https://x.com/shiringhaffary/status/1989073320529261132).
- **OpenAI Piloting Group Chats in APAC Regions**: **OpenAI** has quietly launched group-chat support in **ChatGPT** for Japan, New Zealand, South Korea and Taiwan, as noted in [this tweet](https://x.com/OpenAI/status/1989138776585851038?s=20).
- **Vercel's AI Agents Resolve Support Tickets**: **Vercel** is using **AI agents** internally with over **70%** of support tickets resolved, 6.4 apps/s, 52% hidden defects caught and is considering open-sourcing architectures as detailed in [this tweet](https://x.com/rauchg/status/1989425561995972618?s=46).


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1438621839755841547)** (8 messages🔥): 

> `Nano Banana, Image Models, AI Kino, Cat Orchestra Video` 


- **Nano Banana can be Prompt Engineered?**: Members discuss prompt engineering and testing image models like **Nano Banana**.
   - A member wonders if there is something like **min_p** for in generation for image tokens, citing [this tweet](https://x.com/paularambles/status/1989029622395322816?s=46).
- **AI Kino Viral Cat Video**: A viral **AI-generated video** shows a cat bothering its owner at midnight with an ever-growing musical entourage.
   - Replies praise it as **“AI kino,”** but dissenters still write it off as **“slop,”** as seen in [this thread](https://xcancel.com/paularambles/status/1989029622395322816?s=46).


  

---


### **Moonshot AI (Kimi K-2) ▷ #[announcements](https://discord.com/channels/1369594130807787570/1371757097246785536/1438763509457616956)** (1 messages): 

> `Kimi, Together AI, MoE, Tool Calls` 


- **Kimi K-2 Thinking Deep Dive on Together AI**: A **Kimi K-2 Thinking Deep Dive** will be hosted live on **Together AI** on **November 19, 2025**, at **9 AM PT**, promising a fast but mighty exploration.
   - The event will focus on discovering how the **1T MoE** powers **300 tool calls** in a single run, exploring the implications for agents; registration is available [here](https://luma.com/g5qcq85z).
- **Kimi K2 Powers Tool Calling**: The deep dive event aims to demonstrate how Kimi K2 allows for 300 tool calls in a single run, which could be beneficial for building agents.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1438658665425928243)** (24 messages🔥): 

> `Kimi CLI Tool Use, Jailbreaking Banter, Moonshot Devs, Kimi CLI Usage, Client Side Rendering to Server Side Rendering` 


- ****Kimi CLI's Tool Time****: Users discussed **tool use** in the Kimi CLI, noting that it involves the AI using external tools like web search or file reading via **[action(x)]** parsing.
   - One user burned through their **7100 calls** from a **$39 plan** in just three days.
- ****Jailbreaking Banter Barraged****: A user inquired about the permissibility of discussing **jailbreaking**, referencing an attached image.
   - Another user clarified that community guidelines apply strictly to the Kimi Bot's usage rather than general discussion.
- ****Moonshot Devs Disclosed****: A user asked if there were any Moonshot developers in the server, noting Aspen's inactivity.
   - Another user pointed out that individuals tagged with the **Kimi Team** role are Moonshot employees and that response times may be delayed due to time zone differences (it being late in China).
- ****React Rendering Revolution Reported****: A user shared they switched from **client-side rendering** to **server-side rendering** in **React Vite**.
   - They noted *"there is ton of updates still goes on ahahhaharesult 🗿"*.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1438654009559879812)** (14 messages🔥): 

> `Caching chunks, 404 no cord found, aider-ce documentation, aider with openrouter, MCP servers` 


- **Analysis chunks get cached!**: A member asked about caching chunks of analysis and only refreshing files as needed.
- **404 No Cord Found**: A member experienced a *404 no cord found* error, resolved by setting `OPENAI_API_BASE` and `OPENAI_API_KEY` with the correct model.
- **Aider-CE setup documentation?**: A member inquired about documentation for setting up aider-ce.
   - Another member suggested that [the Aider documentation applies](https://aider.chat/docs/).
- **Aider hangs with Openrouter**: A user reported that **aider** hangs up without responding to prompts or Ctrl+C when using **Openrouter** with default settings.
- **MCP Setup tips**: A member asked how to setup mcp servers.
   - Another member suggested to start with the repo readme and shared their favourite mcp setup on their [blog about using Aider CE with chrome devtools](https://example.com).


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1438994049397821591)** (2 messages): 

> `Clickable links in terminal, Aider, ChatGPT, Terminal Configuration, Prompt Engineering for URLs` 


- **Clickable Links Conundrum in Terminals**: A member is facing issues with **clickable links** not working in their terminal when generated by **ChatGPT**, instead of getting a URL, the links only appear as underlined text.
   - A member suggested it might be a **terminal problem** and requested the **exact prompt and response** to further diagnose the issue.
- **Terminal Configuration Troubles**: The user suspects the issue lies within their terminal's configuration, preventing the proper rendering of clickable URLs.
   - Further investigation requires examining the terminal settings and how it handles URL formatting, and whether **Aider** is interfering.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1438806872185372784)** (1 messages): 

> `DeepSeek OCR, Serverless GPU inference, OpenAI vision API` 


- **DeepSeek OCR Deployed Serverlessly**: The [DeepSeek OCR model](https://github.com/neosantara-xyz/deepseek-ocr-api) can be deployed without needing a local computer.
   - The provided **API** is hosted on **Modal Serverless Compute**, providing access to **GPU** resources with free credits, and it's compatible with **OpenAI vision API** using image URLs or base64 inputs.
- **GPU inference made Free by Modal**: The inference mentioned above is hosted on Modal Serverless Compute with access to **GPU**, also available for free credits.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1438647379841515772)** (13 messages🔥): 

> `GEPA comparison, Prompt Optimization Methods, AI Agent Dev` 


- **GEPA-palooza: Synth vs DSPy**: A member questioned whether **Synth's GEPA** is superior to **DSPy's GEPA**, given that the underlying algorithm should be the same ([link to X post](https://x.com/JoshPurtell/status/1989068917655097520)).
- **Manual Prompting Persists!**: A member suggests that a large percentage (>80-90%) of users are still managing their prompts *manually* and are unaware of **automated prompt optimization methods**.
   - Another member concurred, stating that automated prompt optimization isn't widely discussed or adopted, even though they expected it to be more prevalent.
- **AI Agent Ace Available**: A member with experience in creating **AI agents** and **automation layers** using **LangChain**, **OpenAI API**, **Python**, **FastAPI**, **Next.js**, and **TypeScript** offered their services for collaboration.
   - They emphasized their focus on building systems for *reliability, scalability, and speed*, rather than just prototypes.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1438698106794409995)** (12 messages🔥): 

> `openpilot PR, tinygrad with C++, tinygrad's ptrblck, NeurIPS, TFLite` 


- **OpenPilot PR Approved with Commitment**: A member celebrated the approval of [openpilot PR #36615](https://github.com/commaai/openpilot/pull/36615) with a commitment to prevent future regressions.
- **tinygrad and C++ embedded systems**: A member inquired about using **tinygrad** with **C++** in embedded systems, leading to a discussion about the project's applicability.
   - Another member linked to a relevant [tweet from @__tinygrad__](https://x.com/__tinygrad__/status/1989026590127464554).
- **NeurIPS Attendees**: A member asked who is attending **NeurIPS**, linking to a [tweet from comma_ai](https://x.com/comma_ai/status/1989379959417442419) related to the event.
   - Another member expressed hope for a future online version of the conference.
- **TFLite Ease of Use**: A member suggested that **TFLite** is hard to beat for ease of use, but if your hardware stack is well controlled and supported, tinygrad could work well.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1438786747214463036)** (1 messages): 

> `tinygrad .pth loading, Directly load .pth` 


- **tinygrad can now load .pth files**: The team merged a pull request that enables loading `.pth` files directly into **tinygrad** without needing to load the model in **torch** first.
- **tinygrad Direct Loading**: tinygrad merges functionality for direct loading, to load the model directly from .pth.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1438898791817281617)** (10 messages🔥): 

> `Chat Mode Removal, Pro Subscriber, Credit Usage, Workflow automation` 


- **Chat Mode Disappears, then reappears!**: A Pro Subscriber reported that **chat mode** was removed and then came back, calling it *quite strange*.
   - Another user reported that **chat mode** was still missing for them, and that they seemed to have changed the **points system**, with **Pro** now getting **40k points** instead of **19900**.
- **Pro Group Chat Requested**: A user requested a **Pro group chat** instead of the unmoderated chat, suggesting the **credit usage** is quite inconsistent.
   - The same user also noted that doing a **1 shot build** uses fewer credits than trying to modify.
- **Engineer Specializing in Automation Seeks Support and Collaboration**: An experienced engineer specializing in **workflow automation**, **LLM integration**, **RAG**, **AI detection**, **image and voice AI**, and **blockchain development** shared their expertise and a link to their [portfolio](https://devx-green.vercel.app/).
   - They have built automated pipelines and task orchestration systems using **Dspy**, **OpenAI APIs**, and **custom agents**, reducing response times by **60%**.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1438936187774570678)** (5 messages): 

> `MCP Maintainers at NeurIPS, Agentic Economies Panel, Model Context Protocol Release Candidate` 


- **MCP Maintainers Invited to Agentic Economies Panel at NeurIPS**: A member invited MCP maintainers to a technical panel on **agentic economies** on **December 1st** at **Qualcomm** in **San Diego** during **NeurIPS**.
- **Release Candidate of Model Context Protocol is Frozen**: The specification is now ❄️ **__frozen__** for the upcoming release, meaning that no major changes will be introduced and the current draft is officially classified as a **release candidate** with **17 SEPs**.
   - Members are encouraged to test and open issues [in GitHub](https://github.com/modelcontextprotocol/modelcontextprotocol/issues) for any problems.


  