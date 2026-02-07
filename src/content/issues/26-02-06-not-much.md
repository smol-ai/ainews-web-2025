---
id: MjAyNi0w
title: not much happened today
date: '2026-02-06T05:44:39.731046Z'
description: >-
  **AI News** for early February 2026 highlights a detailed comparison between
  **GPT-5.3-Codex** and **Claude Opus 4.6**, with users noting **Codex's**
  strength in detailed scoped tasks and **Opus's** ergonomic advantage for
  exploratory work. Benchmarks on Karpathy's **nanochat GPT-2 speedrun** show
  **Opus 4.6** achieving better wall-clock performance, while
  **Codex-5.3-xhigh** sometimes suffers from context issues. **Karpathy**
  cautions that current models are not yet reliable for fully autonomous AI
  engineering. Discussions on agent swarms reveal emerging parallels to software
  organizational design, with **Anthropic-style** agent coordination systems and
  **LangChain/LangSmith** emphasizing environment engineering through tracing,
  sandboxing, and state control. The concept of Recursive Language Models (RLM)
  is introduced as a future direction for agent systems to reduce context rot
  and improve structured communication.
companies:
  - openai
  - anthropic
  - langchain
models:
  - gpt-5.3-codex
  - claude-opus-4.6
  - nanochat-gpt-2
topics:
  - agent-systems
  - ai-engineering
  - benchmarking
  - software-organization
  - sandboxing
  - tracing
  - state-management
  - recursive-language-models
  - context-management
people:
  - karpathy
  - sama
  - swyx
  - omarsar0
  - hamelhusain
  - deepfates
---


**a quiet day**

> AI News for 2/5/2026-2/6/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and 24 Discords (**254** channels, and **8727** messages) for you. Estimated reading time saved (at 200wpm): **666** minutes. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!


Today's essay: https://www.latent.space/p/ainews-ai-vs-saas-the-unreasonable


---

# AI Twitter Recap


**Frontier coding models: GPT-5.3-Codex vs Claude Opus 4.6 (and what “agentic” now means)**

- **User consensus snapshot**: A large chunk of the feed is real-world A/B testing of **GPT-5.3-Codex** vs **Claude Opus 4.6**, often concluding that they’re *both* clear generational upgrades but with distinct profiles. People characterize **Codex** as detail-obsessed and strong on scoped tasks, while **Opus** feels more ergonomic for exploratory work and planning ([rishdotblog](https://twitter.com/rishdotblog/status/2019664800910135499), [@theo](https://twitter.com/theo/status/2019709378329550973)). Several notes highlight **Codex’s “auto compaction”/garbage-collecting context** and frequent progress updates during work—perceived as a UX win for long tasks ([cto_junior](https://twitter.com/cto_junior/status/2019607817884475718)).
- **AI-engineer-in-the-loop benchmarks**: A particularly concrete evaluation is optimizing Karpathy’s **nanochat “GPT-2 speedrun”**. [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2019824445792424385) reports both models behaved like competent AI engineers (read code, propose experiments, run benchmarks), with **Opus 4.6** delivering measurable wall-clock gains (e.g., torch compile config tweaks, optimizer step changes, memory reductions) while **Codex-5.3-xhigh** produced ideas but sometimes harmed quality—possibly due to context issues (he observed it hitting “0% context”).
- **Reality check from Karpathy**: [@karpathy](https://twitter.com/karpathy/status/2019851952033771710) pushes back on the idea that models can already do *open-ended* closed-loop AI engineering reliably: they can chase spurious 1% wins with big hidden costs, miss key validation checks, violate repo style instructions, and even misread their own result tables—still “net useful with oversight,” but not yet robust for autonomous optimization.
- **No API as product strategy**: One thread claims **there is no GPT-5.3-Codex API**, implying OpenAI is intentionally funneling usage into the Codex product (and making independent benchmarking harder) ([scaling01](https://twitter.com/scaling01/status/2019856879858450742)). In parallel, Sam Altman explicitly asks how users want **Codex pricing** structured ([sama](https://twitter.com/sama/status/2019814741129195576)).

**Agent swarms & “software teams in a box”**

- **Parallel-agent development starts to look like org design**: Discussion around highly-parallel agent research notes that unconstrained swarms tend to **reinvent the software org chart** (task assignment, coordination, QA) and stress existing tooling (Git/package managers) not built for massive concurrent edits ([swyx](https://twitter.com/swyx/status/2019645622421451106)). This echoes broader “spec-driven development” / “agents as dev teams” narratives ([dbreunig](https://twitter.com/dbreunig/status/2019829245137338548)).
- **Claude Code “agent teams” moment**: Multiple tweets reference Anthropic-style agent coordination systems where agents pick tasks, lock files, and sync via git—framed as a step-change in practical automation ([omarsar0](https://twitter.com/omarsar0/status/2019780306778104056), [HamelHusain](https://twitter.com/HamelHusain/status/2019863601591517466)).
- **LangChain / LangSmith: agents need traces, sandboxes, and state control**: There’s a strong theme that reliability comes from *engineering the environment*: tracing, evals, sandboxing, and type-safe state/middleware. Examples include LangSmith improvements (trace previews; voice-agent debugging) and deepagents adding sandbox backends like **daytona/deno/modal/node VFS** ([LangChain](https://twitter.com/LangChain/status/2019848808310706367), [LangChain](https://twitter.com/LangChain/status/2019846811997942219), [bromann](https://twitter.com/bromann/status/2019880605467697565), [sydneyrunkle](https://twitter.com/sydneyrunkle/status/2019862521717444675)).
- **“RLM” framing (Recursive Language Models)**: A notable conceptual post argues agents will evolve from “LLM + tool loop” (ReAct) into **REPL-native, program-like systems** where context is stored in variables, sub-agents communicate via structured values instead of dumping text into the prompt, and “context rot” is reduced by construction ([deepfates](https://twitter.com/deepfates/status/2019912654173651131)). Related: practical tips to make coding agents more “RLM-like” by pushing context into variables and avoiding tool I/O spam in the prompt ([lateinteraction](https://twitter.com/lateinteraction/status/2019852730177863977)).

**Eval integrity, benchmark drift, and new infrastructure for “trustworthy” scores**

- **“Scores are broken” → decentralize evals**: Hugging Face launched **Community Evals**: benchmark datasets hosting leaderboards, eval results stored as versioned YAML in model repos, PR-based submissions, and reproducibility badges (via Inspect AI), explicitly aiming to make evaluation provenance visible even if it can’t solve contamination/saturation ([huggingface](https://twitter.com/huggingface/status/2019754567685050384), [ben_burtenshaw](https://twitter.com/ben_burtenshaw/status/2019795723378942295), [mervenoyann](https://twitter.com/mervenoyann/status/2019784907178811644)).
- **Benchmarks aren’t saturated (yet)**: A counterpoint emphasizes several difficult benchmarks still have lots of headroom (e.g., SWE-bench Multilingual <80%, SciCode 56%, CritPt 12%, VideoGameBench 1%, efficiency benchmarks far from implied ceilings) ([OfirPress](https://twitter.com/OfirPress/status/2019755847149056456)).
- **Opus 4.6 benchmark story: big jumps, still uneven**: There are repeated claims of Opus 4.6 climbing to top ranks on Arena and other leaderboards ([arena](https://twitter.com/arena/status/2019842691442569566), [scaling01](https://twitter.com/scaling01/status/2019843682128822525)), including strong movement on math-oriented evals (FrontierMath) where Anthropic historically lagged. Epoch’s reporting frames Opus 4.6 Tier 4 at **21% (10/48)**, statistically tied with GPT-5.2 xhigh at 19%, behind GPT-5.2 Pro at 31% ([EpochAIResearch](https://twitter.com/EpochAIResearch/status/2019852613672665193)). But other reasoning-heavy areas (e.g., chess puzzles) remain weak ([scaling01](https://twitter.com/scaling01/status/2019817880662278546)).
- **Eval infra at scale (StepFun)**: A deep infra write-up about Step 3.5 Flash argues reproducible scoring requires handling failure modes, training–inference consistency, contamination checks, robust judging/extraction, and long-output monitoring; “evaluation should slightly lead training” ([ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/2019734062689304970)).

**World models graduate into production: Waymo + DeepMind’s Genie 3**

- **Waymo World Model announcement**: Waymo unveiled a **frontier generative simulation model** built on **DeepMind’s Genie 3**, used to generate hyper-realistic, interactive scenarios—including rare “impossible” events (tornadoes, planes landing on freeways)—to stress-test the Waymo Driver long before real-world exposure ([Waymo](https://twitter.com/Waymo/status/2019804616746029508)).
- **Key technical hook**: DeepMind highlights transfer of Genie 3 “world knowledge” into **Waymo-specific camera + 3D lidar** representations, enabling promptable “what if” scenario generation that matches Waymo hardware modalities ([GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/2019809201812545835), [GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/2019809239569702962)). Multiple researchers point out that extending simulation beyond pixels to sensor streams is the real milestone ([shlomifruchter](https://twitter.com/shlomifruchter/status/2019820532485808329), [sainingxie](https://twitter.com/sainingxie/status/2019841784990351381)).
- **Broader “world models for reasoning” thread**: The Waymo news is repeatedly used as evidence that **world models** (not just text models) are a central scaling frontier for reasoning and embodied tasks ([swyx](https://twitter.com/swyx/status/2019605135689937405), [kimmonismus](https://twitter.com/kimmonismus/status/2019809839804010962), [JeffDean](https://twitter.com/JeffDean/status/2019824614139162804), [demishassabis](https://twitter.com/demishassabis/status/2019827916385972517)).
- **Planning advances for world models**: GRASP is introduced as a **gradient-based, stochastic, parallelized planner** that jointly optimizes actions and intermediate subgoals to improve long-horizon planning vs. common zeroth-order planners (CEM/MPPI) ([michaelpsenka](https://twitter.com/michaelpsenka/status/2019870377032503595), [_amirbar](https://twitter.com/_amirbar/status/2019903658792497482)).

**Memory, long-context control, and multi-agent “cognitive infrastructure”**

- **InfMem: bounded-memory agent with cognitive control**: InfMem proposes a PRETHINK–RETRIEVE–WRITE protocol with RL for long-document QA up to **1M tokens**, emphasizing that longer context windows shift the bottleneck to **what to attend to / when to stop**. Reported gains include substantial accuracy improvements over baselines and **3.9× average latency reduction** via adaptive stopping ([omarsar0](https://twitter.com/omarsar0/status/2019759999170556189)).
- **LatentMem: role-aware latent memory for multi-agent systems**: LatentMem addresses “homogenization” (agents retrieving the same memories despite different roles) by compressing trajectories into role-conditioned latent memory, trained with a policy-optimization method (LMPO). Claims include improvements across QA and coding tasks plus **~50% fewer tokens** / faster inference ([dair_ai](https://twitter.com/dair_ai/status/2019778133550125515)).
- **Product reality: memory leaks and context saturation**: While agentic tooling is shipping fast, developers complain about resource bloat and brittle UX (e.g., “memory leaks” in fast-moving agent IDEs) ([code_star](https://twitter.com/code_star/status/2019707930422161680)). Another thread suspects sub-agent outputs can overwhelm context budgets faster than compaction can recover, hinting at hidden internal longer-context systems ([RylanSchaeffer](https://twitter.com/RylanSchaeffer/status/2019642129736429730)).

**Industry adoption, compute economics, and “jobs vs tasks” discourse**

- **Non-verifiable work limits full automation**: François Chollet argues that in non-verifiable domains, performance gains mostly come from expensive data curation with diminishing returns; since most jobs aren’t end-to-end verifiable, “AI can automate many tasks” ≠ “AI replaces the job” for a long time ([fchollet](https://twitter.com/fchollet/status/2019610121371054455), [fchollet](https://twitter.com/fchollet/status/2019610588612292834)).
- **Contrasting takes: RSI bottlenecks**: Another viewpoint claims tasks will fall in the order they bottleneck recursive self-improvement, with software engineering first ([tszzl](https://twitter.com/tszzl/status/2019614081683189827)).
- **Enterprise deployment signals**: Posts claim **Goldman Sachs rolling out Claude** for accounting automation ([kimmonismus](https://twitter.com/kimmonismus/status/2019865721338229180)), while broader market narratives assert AI is now spooking software-heavy sectors (though the strongest claims are not independently substantiated in-tweet) ([kimmonismus](https://twitter.com/kimmonismus/status/2019757481925464371)).
- **Capex scale**: Several tweets highlight hyperscaler spend acceleration; one claims 2026 combined capex for major hyperscalers near **$650B (~2% of US GDP)** as an “AI arms race” framing ([scaling01](https://twitter.com/scaling01/status/2019789747896377697)), alongside a note that hyperscaler data center capex may **double in 2026** ([kimmonismus](https://twitter.com/kimmonismus/status/2019773237618479594)).
- **Old-guard reassurance to engineers**: Eric S. Raymond delivers a high-engagement “programming isn’t obsolete” argument: systems remain complex and the human-intent-to-computer-spec gap persists; the prescription is adaptation and upskilling, not panic ([esrtweet](https://twitter.com/esrtweet/status/2019779602617376788)).

---

### Top tweets (by engagement)
- [Microinteracti1](https://twitter.com/Microinteracti1/status/2019712610547933593): viral political commentary post (highly engaged; not technical).
- [elonmusk](https://twitter.com/elonmusk/status/2019823468968370633): “Here we go” (context not provided in tweet text dump).
- [esrtweet](https://twitter.com/esrtweet/status/2019779602617376788): “programming panic is a bust; upskill.”
- [Waymo](https://twitter.com/Waymo/status/2019804616746029508): Waymo World Model built on Genie 3 for rare-event simulation.
- [sama](https://twitter.com/sama/status/2019813802049696064): “5.3 lovefest” / model excitement.
- [claudeai](https://twitter.com/claudeai/status/2019833113418035237): “Built with Opus 4.6” virtual hackathon ($100K API credits).
- [chatgpt21](https://twitter.com/chatgpt21/status/2019679978162634930): Opus 4.6 “pokemon clone” claim (110k tokens, 1.5h reasoning).
- [theo](https://twitter.com/theo/status/2019598113238139262): “I know an Opus UI when i see one” (UI/launch zeitgeist).
- [ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/2019839335382790342): speculative systems idea: streaming weights via fiber loop / flash bandwidth for inference.

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Local AI on Low-End Hardware

  - **[CPU-only, no GPU computers can run all kinds of AI tools locally](https://www.reddit.com/r/LocalLLaMA/comments/1qxgkd1/cpuonly_no_gpu_computers_can_run_all_kinds_of_ai/)** (Activity: 544): **The post highlights the capability of running AI tools locally on a CPU-only setup, specifically using a Dell OptiPlex 3060 with an i5-8500 processor and 32GB of RAM. The user successfully runs 12B Q4_K_M gguf LLMs using KoboldCPP, enabling local chatbot interactions with models from Hugging Face. Additionally, the setup supports Stable Diffusion 1.5 for image generation, albeit slowly, and Chatterbox TTS for voice cloning. The post emphasizes that advanced AI tasks can be performed on minimal hardware, challenging the notion that expensive, GPU-heavy setups are necessary for local AI experimentation.** Some commenters express optimism about the future of AI being accessible on basic hardware, while others note a divide in the community regarding hardware elitism and the accessibility of running local models.

    - noctrex suggests trying out specific models like [LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct), [LFM2.5-1.2B-Thinking](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking), and [LFM2.5-VL-1.6B](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B) for CPU-only setups. These models are praised for their small size and efficiency, making them suitable for running on CPU-only docker machines without the need for expensive GPU hardware.
    - Techngro expresses optimism about the future of AI being accessible to the average person through local models that are both intelligent and small enough to run on basic hardware. This vision contrasts with the current trend of relying on large, expensive models hosted by companies, suggesting a shift towards more democratized AI usage.
    - NoobMLDude provides practical applications for local AI setups, such as using them as private meeting note takers or talking assistants. This highlights the versatility and potential of local AI models to perform useful tasks without the need for high-end hardware.

  - **[No NVIDIA? No Problem. My 2018 "Potato" 8th Gen i3 hits 10 TPS on 16B MoE.](https://www.reddit.com/r/LocalLLaMA/comments/1qxcm5g/no_nvidia_no_problem_my_2018_potato_8th_gen_i3/)** (Activity: 866): **A user in Burma successfully ran a 16B MoE model, DeepSeek-Coder-V2-Lite, on an HP ProBook 650 G5 with an i3-8145U CPU and 16GB RAM, achieving `10 TPS` using integrated Intel UHD 620 graphics. The setup leverages **OpenVINO** as a backend for `llama-cpp-python`, highlighting the efficiency of MoE models, which compute only `2.4B` parameters per token. The user emphasizes the importance of dual-channel RAM and using Linux to minimize resource overhead. Initial iGPU compilation lag and occasional language drift were noted as challenges.** Commenters appreciated the ingenuity and resourcefulness of the setup, with some noting that the GPU shortage era has improved optimization skills. There was interest in the user's daily driver model for coding tasks.

    - The comment by ruibranco highlights the importance of dual-channel RAM in CPU inference, noting that memory bandwidth, rather than compute power, is often the bottleneck. By switching from single to dual-channel RAM, throughput can effectively double, which is crucial for running models like the 16B MoE on a CPU. The MoE architecture is praised for its efficiency, as it only activates 2.4B parameters per token, allowing the model to fit within the cache of an 8th Gen i3 processor.
    - The use of MoE (Mixture of Experts) architecture is noted for its efficiency in this setup, as it reduces the active parameter count to 2.4B per token, which is manageable for the CPU's cache. This approach is particularly beneficial for older CPUs like the 8th Gen i3, as it minimizes the working set size, enhancing performance without requiring high-end hardware.
    - The comment also touches on potential precision issues with OpenVINO's INT8/FP16 path on older iGPUs like the UHD 620, which may cause 'Chinese token drift'. This suggests that the limited compute precision of these iGPUs could affect the accuracy of the model's output, highlighting a technical challenge when using older integrated graphics for machine learning tasks.

  - **[Anyone here actually using AI fully offline?](https://www.reddit.com/r/LocalLLM/comments/1qwjgj4/anyone_here_actually_using_ai_fully_offline/)** (Activity: 383): **Running AI models fully offline is feasible with tools like **LM Studio**, **Ollama**, and **openwebUI**. These platforms allow users to operate models locally, with **LM Studio** and **Ollama** providing access to models via platforms like [Hugging Face](https://huggingface.co/) and their own repositories. **openwebUI** offers a local web interface similar to ChatGPT, and can be combined with **ComfyUI** for image generation, though it is more complex. Users report that while offline AI setups can be challenging, they are viable for tasks like coding and consulting, with models like `gpt-oss-20b` being used effectively in these environments.** Some users find offline AI setups beneficial for specific tasks like coding and consulting, though they note that these setups can require significant computational resources, especially for coding workflows. The complexity of setup and maintenance is a common challenge, but the control and independence from cloud services are valued.

    - Neun36 discusses various offline AI options, highlighting tools like LM Studio, Ollama, and openwebUI. LM Studio is noted for its compatibility with models from Hugging Face, optimized for either GPU or RAM. Ollama offers local model hosting, and openwebUI provides a browser-based interface similar to ChatGPT, with the added complexity of integrating ComfyUI for image generation.
    - dsartori mentions using AI offline for coding, consulting, and community organizing, emphasizing that coding workflows demand a robust setup. A teammate uses the `gpt-oss-20b` model in LMStudio, indicating its utility in consulting but not as a sole solution.
    - DatBass612 shares a detailed account of achieving a positive ROI within five months after investing in a high-end M3 Ultra to run OSS 120B models. They estimate daily token usage at around `$200`, and mention the potential for increased token usage with tools like OpenClaw, highlighting the importance of having sufficient unified memory for virtualization and sub-agent operations.


### 2. OpenClaw and Local LLMs Challenges

  - **[OpenClaw with local LLMs - has anyone actually made it work well?](https://www.reddit.com/r/LocalLLM/comments/1qx51zc/openclaw_with_local_llms_has_anyone_actually_made/)** (Activity: 200): **The post discusses transitioning from the **Claude API** to local LLMs like **Ollama** or **LM Studio** to reduce costs associated with token usage. The user is considering models like `Llama 3.1` or `Qwen2.5-Coder` for tool-calling capabilities without latency issues. Concerns about security vulnerabilities in **OpenClaw** are noted, with some users suggesting alternatives like **Qwen3Coder** for agentic tasks. A [Local AI playlist](https://www.youtube.com/playlist?list=PLmBiQSpo5XuQKaKGgoiPFFt_Jfvp3oioV) is shared for further exploration of secure local LLM applications.** Commenters express skepticism about OpenClaw due to security issues, suggesting that investing in VRAM for local models is preferable to paying for API services. Some users have experimented with local setups but remain cautious about security risks.

    - **Qwen3Coder** and **Qwen3Coder-Next** are highlighted as effective for tool calling and agentic uses, with a link provided to [Qwen3Coder-Next](https://qwen3lm.com/coder-next/). The commenter notes security concerns with OpenClaw, suggesting alternative secure uses for local LLMs, such as private meeting assistants and coding assistants, and provides a [Local AI playlist](https://www.youtube.com/playlist?list=PLmBiQSpo5XuQKaKGgoiPFFt_Jfvp3oioV) for further exploration.
    - A user describes experimenting with OpenClaw by integrating it with a local `gpt-oss-120b` model in `lmstudio`, emphasizing the importance of security by running it under a `nologin` user and restricting permissions to a specific folder. Despite the technical setup, they conclude that the potential security risks outweigh the benefits of using OpenClaw.
    - Another user reports using OpenClaw with `qwen3 coder 30b`, noting that while the setup process was challenging due to lack of documentation, the system performs well, allowing the creation of new skills through simple instructions. This highlights the potential of OpenClaw when paired with powerful local models, despite initial setup difficulties.

  - **[Clawdbot / Moltbot → Misguided Hype?](https://www.reddit.com/r/LocalLLM/comments/1qwg8an/clawdbot_moltbot_misguided_hype/)** (Activity: 86): ****Moltbot (OpenClaw)** is marketed as a 'free personal AI assistant' but requires multiple paid subscriptions to function effectively. Users need API keys from **Anthropic, OpenAI, and Google AI** for AI models, a **Brave Search API** for web search, and **ElevenLabs or OpenAI TTS credits** for voice features. Additionally, browser automation requires **Playwright** setup, potentially incurring cloud hosting costs. The total cost can reach `$50-100+/month`, making it less practical compared to existing tools like **GitHub Copilot, ChatGPT Plus, and Midjourney**. The project is more suited for developers interested in tinkering rather than a ready-to-use personal assistant.** Some users argue that while Moltbot requires multiple subscriptions, it's possible to self-host components like LLMs and TTS to avoid costs, though this may not match the performance of cloud-based solutions. Others note that the bot isn't truly 'local' and requires significant technical knowledge to set up effectively.

    - No_Heron_8757 discusses a hybrid approach using ChatGPT Plus for main LLM tasks while offloading simpler tasks to local LLMs via LM Studio. They highlight the integration of web search and browser automation within the same VM, and the use of Kokoro for TTS, which performs adequately on semi-modern GPUs. They express a desire for better performance with local LLMs as primary models, noting the current speed limitations without expensive hardware.
    - Valuable-Fondant-241 emphasizes the feasibility of self-hosting LLMs and related services like TTS, countering the notion that a subscription is necessary. They acknowledge the trade-off in power and speed compared to datacenter-hosted solutions but assert that self-hosting is a viable option for those with the right knowledge and expectations, particularly in this community where such practices are well understood.
    - clayingmore highlights the community's focus on optimizing cost-to-quality-and-quantity for local LLMs, noting that running low-cost local models is often free. They describe the innovative 'heartbeat' pattern in OpenClaw, where the LLM autonomously strategizes and solves problems through reasoning-act loops, verification, and continuous improvement. This agentic approach is seen as a significant advancement, contrasting with traditional IDE code assistants.


### 3. Innovative AI Model and Benchmark Releases

  - **[BalatroBench - Benchmark LLMs' strategic performance in Balatro](https://www.reddit.com/r/LocalLLaMA/comments/1qwxtf8/balatrobench_benchmark_llms_strategic_performance/)** (Activity: 590): ****BalatroBench** is a new benchmark for evaluating the strategic performance of local LLMs in the game Balatro. The system uses two main components: [BalatroBot](https://github.com/coder/balatrobot), a mod that provides an HTTP API for game state and controls, and [BalatroLLM](https://github.com/coder/balatrollm), a bot framework that allows users to define strategies using Jinja2 templates. These templates dictate how the game state is presented to the LLM and guide its decision-making process. The benchmark supports any OpenAI-compatible endpoint, enabling diverse model evaluations, including open-weight models. Results are available on [BalatroBench](https://balatrobench.com/).** Commenters appreciate the real-world evaluation aspect of BalatroBench and suggest using evolutionary strategies like DGM, OpenEvolve, SICA, or SEAL to test LLMs' ability to self-evolve using the Jinja2-based framework.

    - TomLucidor suggests using frameworks like DGM, OpenEvolve, SICA, or SEAL to test which LLM can self-evolve the fastest when playing Balatro, especially if the game is Jinja2-based. These frameworks are known for their ability to facilitate self-evolution in models, providing a robust test of strategic performance.
    - jd_3d is interested in testing Opus 4.6 on Balatro to see if it shows any improvement over version 4.5. This implies a focus on version-specific performance enhancements and how they translate into strategic gameplay improvements.
    - jacek2023 highlights the potential for using local LLMs to play Balatro, which could be a significant step in evaluating LLMs' strategic capabilities in a real-world setting. This approach allows for direct testing of models' decision-making processes in a controlled environment.

  - **[We built an 8B world model that beats 402B Llama 4 by generating web code instead of pixels — open weights on HF](https://www.reddit.com/r/LocalLLaMA/comments/1qwo9j0/we_built_an_8b_world_model_that_beats_402b_llama/)** (Activity: 302): ****Trillion Labs** and **KAIST AI** have released `gWorld`, an open-weight visual world model for mobile GUIs, available in `8B` and `32B` sizes on [Hugging Face](https://huggingface.co/trillionlabs/gWorld-8B). Unlike traditional models that predict screens as pixels, `gWorld` generates executable web code (HTML/CSS/JS) to render images, leveraging strong priors from pre-training on structured web code. This approach significantly improves visual fidelity and text rendering, achieving `74.9%` accuracy with the `8B` model on MWMBench, outperforming models up to `50×` its size, such as the `402B Llama 4 Maverick`. The model's render failure rate is less than `1%`, and it generalizes well across languages, as demonstrated by its performance on the Korean apps benchmark (KApps).** Some commenters question the claim of beating `402B Llama 4`, noting that the `Maverick` model, which is `17B` active, had a disappointing reception. Others are impressed by `gWorld` outperforming models like `GLM` and `Qwen`, suggesting the title may be misleading.

    - The claim that an 8B world model beats a 402B Llama 4 model is questioned, with a specific reference to Maverick, a 17B model that was released with underwhelming coding performance. This highlights skepticism about the model's capabilities and the potential for misleading claims in AI model announcements.
    - A technical inquiry is made about the nature of the model, questioning whether it is truly a 'world model' or simply a large language model (LLM) that predicts the next HTML page. This raises a discussion about the definition and scope of world models versus traditional LLMs in AI.
    - The discussion touches on the model's output format, specifically whether it generates HTML. This suggests a focus on the model's application in web code generation rather than traditional pixel-based outputs, which could imply a novel approach to AI model design and utility.

  - **[Google Research announces Sequential Attention: Making AI models leaner and faster without sacrificing accuracy](https://www.reddit.com/r/LocalLLaMA/comments/1qwboqn/google_research_announces_sequential_attention/)** (Activity: 674): ****Google Research** has introduced a new technique called **Sequential Attention** designed to optimize AI models by reducing their size and computational demands while maintaining performance. This approach focuses on subset selection to enhance efficiency in large-scale models, addressing the NP-hard problem of feature selection in deep neural networks. The method is detailed in a paper available on [arXiv](https://arxiv.org/abs/2209.14881), which, despite being published three years ago, is now being highlighted for its practical applications in current AI model optimization.** Commenters noted skepticism about the claim of maintaining accuracy, suggesting it means the model performs well in tests rather than computing the same results as previous methods like Flash Attention. There is also curiosity about its performance in upcoming benchmarks like Gemma 4.

    - - **-p-e-w-** highlights that the claim of 'without sacrificing accuracy' should be interpreted as the model performing equally well in tests, rather than computing the exact same results as previous methods like Flash Attention. This suggests a focus on empirical performance rather than theoretical equivalence.
    - - **coulispi-io** points out a discrepancy regarding the age of the research, noting that the linked paper is from three years ago. This raises questions about the novelty of the announcement and whether the current implementation differs significantly from the original research.
    - - **FinalsMVPZachZarba** clarifies that the approach seems to be a feature selection algorithm primarily for regression problems, rather than a new attention mechanism for LLMs. However, it does mention LLM pruning as a potential application, where the algorithm could help in selecting parts of the neural network to prune, indicating a possible efficiency improvement in model size and computation.


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude Opus 4.6 and GPT-5.3 Codex Releases and Benchmarks

  - **[GPT-5.3-Codex was used to create itself](https://www.reddit.com/r/singularity/comments/1qwte2l/gpt53codex_was_used_to_create_itself/)** (Activity: 558): **The image discusses the development of **GPT-5.3-Codex**, emphasizing its unique role in self-development. It highlights that early versions of the model were actively used in debugging its own training processes, managing deployment, and diagnosing test results, showcasing a significant step in AI self-sufficiency. This marks a notable advancement in AI capabilities, where a model contributes directly to its own iterative improvement, potentially accelerating development cycles and reducing human intervention.** The comments reflect a mix of humor and concern about AI's growing role in management and development, with one user joking about AI replacing mid-level managers and another expressing apprehension about job security.


  - **[Claude Opus 4.6 is out](https://www.reddit.com/r/singularity/comments/1qwrrn7/claude_opus_46_is_out/)** (Activity: 1189): **The image highlights the release of **Claude Opus 4.6**, a new version of a model by **Anthropic**. The interface suggests a focus on user interaction with a text input box for queries. The dropdown menu indicates that this version is part of a series, with previous versions like "Sonnet 4.5" and "Haiku 4.5" also available. A notable benchmark achievement is mentioned in the comments, with **Claude Opus 4.6** scoring `68.8%` on the **ARC-AGI 2** test, which is a significant performance indicator for AI models. This release seems to be in response to competitive pressures, as noted by a comment about a concurrent update from **Codex**.** One comment humorously notes the model's description as being for "ambitious work," which may not align with all users' needs. Another comment suggests that the release timing was influenced by competitive dynamics with **Codex**.

    - SerdarCS highlights that Claude Opus 4.6 achieves a `68.8%` score on the ARC-AGI 2 benchmark, which is a significant performance indicator for AI models. This score suggests substantial improvements in the model's capabilities, potentially positioning it as a leader in the field. [Source](https://www.anthropic.com/news/claude-opus-4-6).
    - Solid_Anxiety8176 expresses interest in test results for Claude Opus 4.6, noting that while Opus 4.5 was already impressive, improvements such as a cheaper cost and a larger context window would be highly beneficial. This reflects a common user interest in both performance enhancements and cost efficiency in AI models.

  - **[Anthropic releases Claude Opus 4.6 model, same pricing as 4.5](https://www.reddit.com/r/singularity/comments/1qws1j9/anthropic_releases_claude_opus_46_model_same/)** (Activity: 931): ****Anthropic** has released the Claude Opus 4.6 model, which is highlighted as the most capable for ambitious work while maintaining the same pricing as the previous 4.5 version. The image provides a comparison chart showing the performance of Opus 4.6 against other models like Opus 4.5, Sonnet 4.5, Gemini 3 Pro, and GPT-5.2. Key performance metrics include agentic terminal coding, agentic coding, and multidisciplinary reasoning, with Opus 4.6 excelling particularly in agentic tool use and multilingual Q&A. The model's ARC-AGI score is notably high, indicating significant advancements in artificial general intelligence capabilities.** Commenters note the impressive ARC-AGI score of Opus 4.6, suggesting it could lead to rapid saturation in the market. However, there is a mention of no progress in the SWE benchmark, indicating some areas where the model may not have improved.

    - The ARC-AGI score for Claude Opus 4.6 is notably high, indicating significant advancements in general AI capabilities. This score suggests that the model has improved in areas related to artificial general intelligence, which could lead to broader applications and increased adoption in the coming months.
    - Despite the impressive ARC-AGI score, there appears to be no progress in the SWE (Software Engineering) benchmark. This suggests that while the model has improved in general intelligence, its specific capabilities in software engineering tasks remain unchanged compared to previous versions.
    - The update to Claude Opus 4.6 seems to provide a more well-rounded performance, with significant improvements in general intelligence metrics like ARC-AGI and HLE (Human-Level Evaluation). However, for specialized tasks such as coding, the upcoming Sonnet 5 model might offer better performance, indicating a strategic focus on different model strengths for varied applications.

  - **[OpenAI released GPT 5.3 Codex](https://www.reddit.com/r/singularity/comments/1qwsqlg/openai_released_gpt_53_codex/)** (Activity: 981): ****OpenAI** has released **GPT-5.3-Codex**, a groundbreaking model that was instrumental in its own development, using early versions to debug, manage deployment, and diagnose evaluations. It shows a `25%` increase in speed and excels in benchmarks like SWE-Bench Pro and Terminal-Bench, achieving a `77.3%` score, surpassing previous models like Opus. This model is capable of autonomously building complex applications, collaborating interactively, and identifying software vulnerabilities, marking a significant step towards a general-purpose technical agent. More details can be found in the [original article](https://openai.com/index/introducing-gpt-5-3-codex/).** There is a debate regarding the benchmark results, with some users questioning the validity of the `77.3%` score compared to other models like Opus, suggesting potential discrepancies or 'cooking' of results.

    - **GPT-5.3-Codex** has been described as a self-improving model, where early versions were utilized to debug its own training and manage deployment. This self-referential capability reportedly accelerated its development significantly, showcasing a novel approach in AI model training and deployment.
    - A benchmark comparison highlights that **GPT-5.3-Codex** achieved a `77.3%` score on a terminal benchmark, surpassing the `65%` score of Opus. This significant performance difference raises questions about the benchmarks used and whether they are directly comparable or if there are discrepancies in the testing conditions.
    - The release of **GPT-5.3-Codex** is noted for its substantial improvements over previous versions, such as Opus 4.6. While Opus 4.6 offers a `1 million` token context window, the enhancements in GPT-5.3's capabilities appear more impactful on paper, suggesting a leap in performance and functionality.

  - **[We tasked Opus 4.6 using agent teams to build a C compiler. Then we (mostly) walked away. Two weeks later, it worked on the Linux kernel.](https://www.reddit.com/r/singularity/comments/1qwur8p/we_tasked_opus_46_using_agent_teams_to_build_a_c/)** (Activity: 553): **A team of 16 parallel Claude instances developed a Rust-based C compiler capable of compiling the Linux kernel across multiple architectures, achieving a `100,000-line` codebase. This project highlights the potential of autonomous agent teams, emphasizing the importance of high-quality tests, task management, and parallelism. Despite its success, limitations remain, such as the absence of a 16-bit x86 compiler and assembler. The project serves as a benchmark for language model capabilities, demonstrating significant advancements in compiler generation. [Codex 5.3](https://openai.com/index/introducing-gpt-5-3-codex/) achieved equal performance to earlier models on SWE-bench at half the token count, indicating improved per-token efficiency.** Commenters express excitement and unease about the rapid progress in language models, noting the need for new strategies to navigate potential risks. There is a discussion on per-token efficiency, with Codex 5.3 achieving equal performance at half the token count, suggesting improved efficiency and potential cost reductions.

    - The experiment with Opus 4.6 highlights the rapid advancements in language models and their scaffolds, enabling the creation of complex software like a C compiler with minimal human intervention. This progress suggests a shift towards more autonomous software development, but also raises concerns about the need for new strategies to manage potential risks associated with such powerful tools.
    - The project involved nearly 2,000 Claude Code sessions and incurred $20,000 in API costs, raising questions about the efficiency of token usage in large-scale AI projects. Notably, the Codex 5.3 release notes indicate that it achieved similar performance to earlier models on the SWE-bench with half the token count, suggesting improvements in per-token efficiency that could reduce costs significantly in the future.
    - A key challenge in using AI agents like Claude for complex tasks is designing a robust testing environment. The success of the project relied heavily on creating high-quality test suites and verifiers to ensure the AI was solving the correct problems. This approach, akin to the waterfall model, is crucial for autonomous agentic programming but may not be feasible for all projects due to the iterative nature of software development.

  - **[They actually dropped GPT-5.3 Codex the minute Opus 4.6 dropped LOL](https://www.reddit.com/r/OpenAI/comments/1qwsnp9/they_actually_dropped_gpt53_codex_the_minute_opus/)** (Activity: 1209): **The image humorously suggests the release of a new AI model, GPT-5.3 Codex, coinciding with the release of another model, Opus 4.6. This is framed as part of an ongoing competitive dynamic in AI development, likened to a 'war' between AI models. The image itself is a meme, playing on the idea of rapid and competitive advancements in AI technology, with a design that mimics a tech product announcement.** Commenters humorously compare the situation to a 'Coke vs Pepsi' rivalry, indicating a perception of intense competition between AI models and companies.


  - **[GPT-5.3 Codex vs Opus 4.6: We benchmarked both on our production Rails codebase — the results are brutal](https://www.reddit.com/r/ClaudeAI/comments/1qxr7vs/gpt53_codex_vs_opus_46_we_benchmarked_both_on_our/)** (Activity: 781): **The post discusses a custom benchmarking of AI coding agents, specifically **GPT-5.3 Codex** and **Opus 4.6**, on a Ruby on Rails codebase. The methodology involved selecting PRs from their repository, inferring original specs, and having each agent implement these specs independently. The implementations were graded by three different LLM evaluators on correctness, completeness, and code quality. The results showed that **GPT-5.3 Codex** achieved a quality score of approximately `0.70` at a cost of under `$1/ticket`, while **Opus 4.6** scored around `0.61` at about `$5/ticket`, indicating that Codex provides better quality at a significantly lower cost. The image provides a visual comparison of these models along with others like **Sonnet 4.5** and **Gemini 3 Pro**.** One commenter expressed skepticism about **Gemini Pro**, while another mentioned satisfaction with **Opus**. A third commenter inquired about whether the tests used raw LLM calls or proprietary tools like Codex/Claude code.

    - Best_Expression3850 inquires about the methodology used in the benchmarking, specifically whether 'raw' LLM calls were used or if proprietary agentic tools like Codex/Claude code were employed. This distinction is crucial as it can significantly impact the performance and capabilities of the models being tested.
    - InterstellarReddit shares a practical approach to benchmarking AI models by cloning a project and having both models implement the same tasks with identical prompts and tools. This method ensures a fair comparison by controlling for variables that could affect the outcome, such as prompt phrasing or tool availability.
    - DramaLlamaDad notes a preference for Opus, stating that in their experience, Opus consistently outperforms in various tests. This anecdotal evidence suggests a trend where Opus may have advantages in certain scenarios, potentially influencing user preference and model selection.

  - **[With Opus 4.6 and Codex 5.3 dropping today, I looked at what this race is actually costing Anthropic](https://www.reddit.com/r/ClaudeAI/comments/1qx0wr3/with_opus_46_and_codex_53_dropping_today_i_looked/)** (Activity: 1016): ****Anthropic** is reportedly preparing for significant financial challenges as it competes with **OpenAI**. Internal projections suggest a dramatic increase in revenue, with expectations of `$18B` this year and `$55B` next year, aiming for `$148B` by 2029. However, costs are escalating faster, with training expenses projected at `$12B` this year and `$23B` next year, potentially reaching `$30B` annually by 2028. Inference costs are also substantial, estimated at `$7B` this year and `$16B` next year. Despite these expenses, investors are valuing the company at `$350B`, up from `$170B` last September, with plans to inject another `$10B+`. The company anticipates breaking even by 2028, with total operating expenses projected at `$139B` until then. This financial strategy underscores the intense competition in AI development, particularly with the release of **Opus 4.6** and **Codex 5.3**.** Commenters highlight the benefits of competition for users, noting the rapid evolution of AI models. Some suggest that **OpenAI** may be less solvent than **Anthropic**, while others speculate on the potential for **Anthropic** to become a trillion-dollar company.

    - Jarie743 highlights the financial stability of Anthropic compared to OpenAI, suggesting that OpenAI is less solvent. This implies that despite the rapid advancements and releases like Opus 4.6 and Codex 5.3, financial sustainability is a critical factor in the AI race. The comment suggests that Anthropic might have a more robust financial strategy or backing, which could influence its long-term competitiveness.
    - BallerDay points out Google's massive capital expenditure (CAPEX) announcement of $180 billion for 2026, raising questions about how smaller companies can compete with such financial power. This highlights the significant financial barriers to entry and competition in the AI space, where large-scale investments are crucial for infrastructure, research, and development.
    - ai-attorney expresses enthusiasm for Opus 4.6, describing it as 'extraordinary' and speculating on the future capabilities of Claude. This suggests that the current advancements in AI models are impressive and that there is significant potential for further development, which could lead to even more powerful AI systems in the near future.

  - **[Opus 4.6 vs Codex 5.3 in the Swiftagon: FIGHT!](https://www.reddit.com/r/ClaudeAI/comments/1qwvj5k/opus_46_vs_codex_53_in_the_swiftagon_fight/)** (Activity: 722): ****Anthropic's Opus 4.6** and **OpenAI's Codex 5.3** were tested on a macOS app codebase (~4,200 lines of Swift) focusing on concurrency architecture involving GCD, Swift actors, and @MainActor. Both models successfully traced a 10-step data pipeline and identified concurrency strategies, with **Claude Opus 4.6** providing deeper architectural insights, such as identifying a potential double-release issue. **Codex 5.3** was faster, completing tasks in `4 min 14 sec` compared to Claude's `10 min`, and highlighted a critical resource management issue. Both models demonstrated improved reasoning about Swift concurrency, a challenging domain for AI models.** A notable opinion from the comments highlights a pricing concern: **Claude's Max plan** is significantly more expensive than **Codex's Pro plan**, yet the performance difference does not justify the `80$` monthly gap. This could impact Anthropic's competitive positioning if they do not adjust their pricing strategy.

    - Hungry-Gear-4201 highlights a significant pricing disparity between Opus 4.6 and Codex 5.3, noting that Opus 4.6 is priced at $100 per month compared to Codex 5.3's $20 per month. Despite the price difference, the performance and usage limits are comparable, which raises concerns about Anthropic's pricing strategy potentially alienating 'pro' customers if they don't offer significantly better performance for the higher cost.
    - mark_99 suggests that using both Opus 4.6 and Codex 5.3 together can enhance accuracy, implying that cross-verification between models can lead to better results. This approach could be particularly beneficial in complex projects where accuracy is critical, as it leverages the strengths of both models to mitigate individual weaknesses.
    - spdustin appreciates the timing of the comparison between Opus 4.6 and Codex 5.3, as they are beginning a Swift project. This indicates that real-world testing and comparisons of AI models are valuable for developers making decisions on which tools to integrate into their workflows.


### 2. AI Model Performance and Comparisons

  - **[Opus 4.6 uncovers 500 zero-day flaws in open-source code](https://www.reddit.com/r/singularity/comments/1qxdd6n/opus_46_uncovers_500_zeroday_flaws_in_opensource/)** (Activity: 744): ****Anthropic's Claude Opus 4.6** has identified `500+` zero-day vulnerabilities in open-source libraries, showcasing its advanced reasoning capabilities in a sandboxed environment using Python and vulnerability analysis tools. This model's ability to uncover high-severity security flaws, even when traditional methods fail, marks a significant advancement in AI-driven cybersecurity, particularly for open-source software. The findings highlight both the potential for enhanced security and the risks of misuse of such powerful AI capabilities.** A notable comment questions the authenticity of the `500+` vulnerabilities, suggesting skepticism about the real impact of the findings. Another comment appreciates the new benchmark set by the model in terms of cumulative severity of bugs fixed.

    - mxforest highlights the potential for a new benchmark in evaluating models based on the cumulative severity of bugs they can identify and fix. This suggests a shift in how model performance could be measured, focusing on real-world impact rather than just theoretical capabilities.
    - woolharbor raises a critical point about the validity of the findings, questioning how many of the reported 500 zero-day flaws are genuine. This underscores the importance of verification and validation in security research to ensure that identified vulnerabilities are not false positives.
    - will_dormer notes the dual-use nature of such discoveries, emphasizing that while identifying zero-day flaws is beneficial for improving security, it also presents opportunities for malicious actors. This highlights the ethical considerations and potential risks involved in publishing such findings.

  - **[GPT-5.3 Codex vs Opus 4.6: We benchmarked both on our production Rails codebase — the results are brutal](https://www.reddit.com/r/ClaudeAI/comments/1qxr7vs/gpt53_codex_vs_opus_46_we_benchmarked_both_on_our/)** (Activity: 781): **The post discusses a custom benchmarking of AI coding agents, specifically **GPT-5.3 Codex** and **Opus 4.6**, on a Ruby on Rails codebase. The methodology involved selecting PRs from their repository, inferring original specs, and having each agent implement these specs independently. The implementations were graded by three different LLM evaluators on correctness, completeness, and code quality. The results showed that **GPT-5.3 Codex** achieved a quality score of approximately `0.70` at a cost of under `$1/ticket`, while **Opus 4.6** scored around `0.61` at about `$5/ticket`, indicating that Codex provides better quality at a significantly lower cost. The image provides a visual comparison of these models along with others like **Sonnet 4.5** and **Gemini 3 Pro**.** One commenter expressed skepticism about **Gemini Pro**, while another mentioned satisfaction with **Opus**. A third commenter inquired about whether the tests used raw LLM calls or proprietary tools like Codex/Claude code.

    - Best_Expression3850 inquires about the methodology used in the benchmarking, specifically whether 'raw' LLM calls were used or if proprietary agentic tools like Codex/Claude code were employed. This distinction is crucial as it can significantly impact the performance and capabilities of the models being tested.
    - InterstellarReddit shares a practical approach to benchmarking AI models by cloning a project and having both models implement the same tasks with identical prompts and tools. This method ensures a fair comparison by controlling for variables that could affect the outcome, such as prompt phrasing or tool availability.
    - DramaLlamaDad notes a preference for Opus, stating that in their experience, Opus consistently outperforms in various tests. This anecdotal evidence suggests a trend where Opus may have advantages in certain scenarios, potentially influencing user preference and model selection.

  - **[Difference Between Opus 4.6 and Opus 4.5 On My 3D VoxelBuild Benchmark](https://www.reddit.com/r/ClaudeAI/comments/1qx3war/difference_between_opus_46_and_opus_45_on_my_3d/)** (Activity: 614): **The post discusses a benchmark comparison between **Opus 4.6** and **Opus 4.5** on a 3D VoxelBuild platform, highlighting a significant improvement in performance. The cost for Opus 4.6 to create `7 builds` was approximately `$22`, with plans to expand the benchmark with additional builds. The benchmark results can be explored on [Minebench](https://minebench.vercel.app/).** Comments reflect excitement about the potential of AI in procedural world generation, with one user noting the impressive quality of Opus 4.6 compared to 4.5, and another inquiring about the input method for the builds, whether reference pictures or text prompts are used.

    - RazerWolf suggests trying Codex 5.3 xhigh for benchmarking, indicating a potential interest in comparing its performance against Opus 4.6. This implies that Codex 5.3 xhigh might offer competitive or superior capabilities in handling complex tasks like 3D voxel builds, which could be valuable for developers seeking optimal performance in procedural generation tasks.
    - Even_Sea_8005 inquires about the input method for the benchmark, asking whether reference pictures or text prompts are used. This question highlights the importance of understanding the input data's nature, which can significantly affect the performance and outcomes of AI models like Opus 4.6 in generating 3D voxel environments.
    - JahonSedeKodi expresses curiosity about the tools used for building the benchmark, which suggests a deeper interest in the technical stack or software environment that supports the execution of Opus 4.6. This could include programming languages, libraries, or frameworks that are crucial for achieving the impressive results noted in the benchmark.

  - **[Opus 4.6 Is Live. So Is Our Glorious 3 Pro GA Still Napping on Some Server?](https://www.reddit.com/r/Bard/comments/1qwsjvq/opus_46_is_live_so_is_our_glorious_3_pro_ga_still/)** (Activity: 400): **The image presents a comparison of various language models' performance on the MRCR v2 (8-needle) task, focusing on their ability to handle long context comprehension and sequential reasoning. **Opus 4.6** outperforms other models, including **Gemini-3-Pro** and **Gemini-3-Flash**, with the highest mean match ratios at both `256k` and `1M` token contexts. This suggests that Opus 4.6 has superior capabilities in managing large context sizes, a critical factor for advanced language model applications. The post critiques the strategy of quantizing models to save costs, implying that it may compromise performance.** Commenters express surprise at the high accuracy achieved by Opus 4.6, noting that it surpasses expectations for handling `1M` tokens. There is also speculation about the upcoming release of **Sonnet 5**, which is anticipated to outperform current models.

    - Pasto_Shouwa highlights the impressive benchmark performance of Opus 4.6, noting that it achieved an accuracy greater than 33% on 1 million tokens, a feat that took Claude approximately two and a half months to accomplish. This suggests significant advancements in model efficiency and capability.
    - DisaffectedLShaw mentions that Opus 4.6 includes improvements for modern tools, such as new MCPs, skills, and deep researching, as well as enhancements in 'vibe coding'. Additionally, there is anticipation for Sonnet 5, which is rumored to significantly outperform current models and is expected to be released soon.
    - VC_in_the_jungle notes the rollout of Codex 5.3, indicating ongoing developments and competition in the field of AI models, which may influence the performance and capabilities of future releases.

  - **[Gemini 3 vs 2.5 Pro: The "output handicap" is ruining everything](https://www.reddit.com/r/Bard/comments/1qxq09j/gemini_3_vs_25_pro_the_output_handicap_is_ruining/)** (Activity: 146): **The post highlights a significant reduction in output tokens for **Gemini 3** models compared to **Gemini 2.5 Pro** when given a `41k token` prompt. Specifically, **Gemini 2.5 Pro** produced `46,372` output tokens, while **Gemini 3 Pro** and **Gemini 3 Flash** generated only `21,723` and `12,854` tokens, respectively. This drastic reduction is perceived as a downgrade, impacting the models' usability for heavy tasks. The author suggests that **Google** should address this issue to improve the models' performance.** One commenter argues that the number of output tokens does not necessarily equate to the quality of a response, while another mentions switching to **Opus 4.5** and **4.6** due to dissatisfaction with Gemini 3.

    - TheLawIsSacred highlights significant performance issues with Gemini 3 Pro, noting that despite extensive customization and instruction refinement, the model fails to follow instructions effectively. They suggest that Google's prioritization of casual users might be leading to a less sophisticated Pro model. Interestingly, they find the Gemini integrated in Chrome's sidebar tool to be superior, possibly due to its ability to incorporate on-screen content and leverage high-end hardware like a Microsoft Surface's AI-tailored NPU.
    - Anton_Pvl observes a difference in how Gemini 2.5 and 3 handle the 'Chain of thought' in conversations. In Gemini 2.5, the Chain of thought tokens are included in the output, whereas in Gemini 3, they are not counted initially, which might be an attempt to reduce token usage. This change could impact the model's performance and the perceived quality of responses, as the Chain of thought can be crucial for maintaining context in complex interactions.
    - TheLawIsSacred also mentions a workaround for improving Gemini 3 Pro's performance by using extreme prompts to induce a 'panic' response from the model. This involves crafting prompts that suggest dire consequences for poor performance, which seems to temporarily enhance the model's output quality. However, this method is seen as a last resort and highlights the underlying issues with the model's responsiveness and logic handling.


### 3. AI Tools and Usage in Engineering and Development

  - **[Professional engineers: How are you using AI tools to improve productivity at work?](https://www.reddit.com/r/PromptEngineering/comments/1qxh14g/professional_engineers_how_are_you_using_ai_tools/)** (Activity: 49): **AI tools are being integrated into engineering workflows primarily for niche tasks such as generating example code snippets, optimizing database queries, and serving as advanced search engines. These tools excel in providing quick access to information and examples, which engineers can adapt to their specific needs, but they struggle with complex code changes and large-scale system integration due to limitations in context window size and understanding of intricate system architectures. Engineers emphasize the importance of using AI to fill in gaps rather than replace the nuanced decision-making and design processes inherent in engineering roles.** Commenters highlight that AI is effective for simple tasks like internal search and basic coding but falls short in complex coding tasks, often introducing errors. There's a consensus that AI initiatives often fail to deliver at scale, with only a small percentage achieving significant impact, while many could be replaced by simpler technologies like robotic process automation.

    - AI tools are particularly effective for niche tasks such as generating example code snippets or optimizing database queries. For instance, using AI to determine user groups in Windows Active Directory with .NET APIs or writing optimized SQLite queries can significantly streamline the process. However, AI struggles with large codebases due to context window limitations, making it less effective for complex code changes or understanding large systems.
    - AI tools like Copilot can serve as powerful internal search engines, especially when configured correctly, as highlighted in the Nanda paper from MIT. They excel in pattern recognition tasks, such as identifying abnormal equipment operations or relating documents in industrial digital twins. However, many AI initiatives could be achieved with simpler technologies like robotic process automation, and a significant portion of AI projects lack real value at scale.
    - AI is effective for simple coding tasks, creating unit tests, and providing insights into code repositories. However, it often introduces errors in complex coding tasks by inserting irrelevant information. AI serves best as a 'trust-but-verify' partner, where human oversight is crucial to ensure accuracy and relevance, especially in tasks that cannot tolerate high error rates.

  - **[How are people managing context + memory with Cline? (Memory banks, rules, RAG, roadmap?)](https://www.reddit.com/r/CLine/comments/1qx4m16/how_are_people_managing_context_memory_with_cline/)** (Activity: 24): **The post discusses strategies for managing context and memory in **Cline**, a tool used alongside **ChatGPT** for executing tasks like coding and refactoring. The user initially faced issues with a large context window (`200k+ tokens`) and improved efficiency by implementing a `.clineignore` file and optimizing memory banks, reducing the context to `40,000 tokens`. This allowed for the use of smaller models and faster iterations. The post also mentions advanced techniques like **recursive chain of thought** and **RAG-based approaches** (e.g., vector databases) for context management. The user seeks insights on practical implementations and future roadmap features for Cline, such as first-class memory management and smarter context loading.** Commenters suggest using structured memory banks for feature planning and emphasize breaking tasks into smaller chunks to avoid context overload. Some users prefer resetting context frequently to maintain model performance, while others have moved away from memory banks due to their complexity and potential for becoming outdated.

    - Barquish describes a structured approach to managing context and memory with Cline by using a memory-bank system. This involves organizing features into a series of markdown files, such as `memory-bank/feature_[×]/00_index_feature_[×].md`, and maintaining a `progress.md` and `activeContext.md` to track updates. They also utilize `.clinerules` for local workspace management and `custom_instructions` for global settings, allowing multiple Cline instances to run concurrently for different projects like web and mobile apps.
    - False79 emphasizes the importance of breaking down large features into smaller tasks to manage context effectively. They note that LLMs tend to perform worse as the context size approaches `128k`, suggesting that resetting context at the start of each task can improve performance and reduce the need for redoing tasks. This approach allows tasks to be completed in discrete chunks, minimizing the need for long-term memory storage.
    - Repugnantchihuahua shares their experience of moving away from using memory banks due to issues like clunkiness and outdated information. Instead, they focus on deep planning and directing the AI to relevant context areas, as memory banks can sometimes overindex irrelevant data. They also mention using `clinerules` to maintain essential information without relying heavily on memory banks.

  - **[Claude Opus 4.6 is now available in Cline](https://www.reddit.com/r/CLine/comments/1qx158e/claude_opus_46_is_now_available_in_cline/)** (Activity: 12): ****Anthropic** has released **Claude Opus 4.6**, now available in **Cline v3.57**. This model shows significant improvements in reasoning, long context handling, and agentic tasks, with benchmarks including `80.8%` on SWE-Bench Verified, `65.4%` on Terminal-Bench 2.0, and `68.8%` on ARC-AGI-2, a notable increase from `37.6%` on Opus 4.5. It features a `1M token context window`, enhancing its ability to maintain context over long interactions, making it suitable for complex tasks like code refactoring and debugging. The model is accessible via the Anthropic API and integrates with various IDEs such as JetBrains, VS Code, and Emacs.** Some users express dissatisfaction with the model's performance and cost, preferring open-source alternatives. The model's high expense is a notable concern among users.


---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5.2


**1. Frontier Model Releases, Rumors & Bench-Leader Musical Chairs**

- **Opus 4.6 Takes the Throne, Then Trips Over Its Own “Thinking”**: **Claude Opus 4.6** and **claude-opus-4-6-thinking** landed on [Text Arena](https://arena.ai/) and [Code Arena](https://arena.ai/?chat-modality=code) and quickly hit **#1 across Code, Text, and Expert** per the [Leaderboard Changelog](https://arena.ai/blog/leaderboard-changelog/), while also rolling out to Perplexity Max via the [Model Council](https://discord.com/channels/1047197230748151888/1047204950763122820/1469006915114631357).
  - Engineers reported long waits and frequent *“Error – something went wrong”* crashes in **Opus 4.6 thinking mode**, speculating about token limits and tool-use assumptions tied to the Claude app/website, even as others still called it the *best coding model*.

- **Codex 5.3 Hype Train: 1M Context, API Limbo, and Aesthetic Crimes**: Across OpenAI/Cursor/LMArena chats, **GPT-5.3 Codex** chatter centered on rumored specs like **1M context** and **128k reasoning / 128k max output**, plus API pricing claims of **$25–$37.5 output** and **$0.5–$1 cache input** (as discussed in the OpenAI Discord).
  - Cursor users complained Codex is still *“stuck in API limbo”* per [OpenAI model docs](https://platform.openai.com/docs/models), while OpenAI Discord folks joked Codex ships *“sad dark gloomy colors”* for frontends compared to Opus’s nicer design choices.

- **Rumor Season: #keep4o, “Sonnet 5,” and the Model Deletion Cinematic Universe**: LMArena members spun rumors about hypothetical **GPT-4.1/4.5** appearing or getting deleted (citing cost motives via [OpenAI’s “new models and developer products” post](https://openai.com/blog/new-models-and-developer-products)), plus a mini *#keep4o* campaign over **GPT-4o**’s less-robotic vibe.
  - More rumors claimed *“Sonnet 5 is better than opus 4.5”* (contested as fake), with one spicy guess of **83% SWE-bench**, while OpenAI Discord users separately mourned **GPT-4o EOL on Feb 13** and worried successors won’t feel as “human.”


**2. Agentic Coding Goes Wide: Teams, Toolchains & Terminal Testbeds**

- **Agent Teams Ship Commits Like a DDoS (But for Git)**: [Cursor’s long-running coding agents preview](https://x.com/cursor_ai/status/2019456112806732159) claimed **hundreds of agents** produced **1,000+ commits/hour** in a week-long trial, while Lydia Hallie previewed [Claude Code “agent teams”](https://x.com/lydiahallie/status/2019469032844587505?s=46) where a lead agent delegates to specialized sub-agents.
  - [Anthropic Engineering](https://x.com/anthropicai/status/2019496582698397945?s=46&t=b7l37rB6wtbyAh6ah1NpZQ) added that **Opus 4.6** in agent teams built a **C compiler** that works on the **Linux kernel** in **two weeks**, and they also highlighted infra/config can swing agent-benchmark outcomes more than model deltas.

- **SETA Drops 1,376 Terminal Worlds for Agents to Survive In**: Guohao Li released [SETA](https://x.com/guohao_li/status/2019527791876653353?s=46), a set of **1,376 validated terminal coding environments** spanning **DevOps, security, and sysadmin**, aimed at making agentic coding evaluation more realistic.
  - Latent Space discussions emphasized that benchmark results can hinge on “infrastructural noise,” so having standardized, validated terminal environments could reduce accidental leaderboard theater.

- **Agent-Native Engineering: Manage Bots Like You Manage Teams**: A Latent Space thread proposed **“Agent Native Engineering”** as an org model: background agents handle delegation and sync agents handle hard problems, enabling engineers to run multiple concurrent assistants like **Claude Code** (see the referenced [X post](https://xcancel.com/ndrewpignanelli/status/2019403256586539025?s=46)).
  - In the same vein, builders shared workflows where **GPT-5.3 Codex** runs slower-but-smarter for backend work (analysis → review → plan → review → implement), and Codex improves over time if you force it to *“take notes and improve its own workflows”* (via [KarelDoostrlnck’s post](https://x.com/KarelDoostrlnck/status/2019477361557926281)).


**3. Pricing, Rate Limits & Plan Nerfs: The Great AI Squeeze**

- **Perplexity Pro Nerfs Deep Research, Users Bring Pitchforks (and Screenshots)**: Perplexity users reported reduced **Deep Research** query counts and smaller **file upload limits**, circulating a [screenshot comparing old vs new limits](https://cdn.discordapp.com/attachments/1047649527299055688/1469259948302139402/image.png) and criticizing the lack of clear comms.
  - The backlash pushed people to test alternatives like **Gemini Pro** (praised for editable research plans before execution) and **DeepSeek** (described as free/unlimited, with some reservations about China-based services).

- **Opus 4.6: Amazing Output, Speedrunning Your Wallet**: Cursor and other communities praised **Opus 4.6** capability but called it brutally expensive, with one estimate that *“$20 on Opus will last you maybe a day”* and ongoing cost comparisons referencing [OpenAI pricing](https://openai.com/pricing).
  - Separate chatter predicted rising subscription pressure—BASI members joked about **Anthropic at $200** and dependency-driven price hikes—while Kimi users debated whether **Kimi K2.5** remains free on OpenRouter and what plans gate features like swarm/sub-agents.

- **Captcha Boss Fights and Other “Pay in Pain” Taxes**: LMArena users complained about frequent captchas that interrupt evaluation, and a team member said *“We are looking into the captcha system”* to better detect authentic users (see the posted message link: https://discord.com/channels/1340554757349179412/1451574502369656842/1468286122084929546).
  - The vibe across multiple discords: even when model quality improves, access friction (captchas, rate limits, plan tiers) increasingly becomes the real bottleneck.


**4. Security, Red Teaming & Secret Spills in Agent Land**

- **Codex Reads Your Whole Disk, Says the Issue Tracker: “Working as Intended”**: OpenRouter users raised alarms that **Codex** can *read your whole filesystem by default* with no config toggle, pointing to [openai/codex issue #2847](https://github.com/openai/codex/issues/2847) where the team reportedly does not treat it as a bug.
  - A second report, [openai/codex issue #5237](https://github.com/openai/codex/issues/5237), highlighted risks like reading API keys and personal files, feeding broader concerns about default agent permissions and safe-by-default tooling.

- **Red Teamers Wanted: Trajectory Labs Posts the Quest**: [Trajectory Labs](https://trajectorylabs.com/careers/ai-red-teamer) advertised roles for **AI Red Teamers** (stealth AI security startup) with a flexible remote schedule but **30 hours/week minimum**, plus a short form and a red-teaming game.
  - The listing resonated with ongoing jailbreak/red-team chatter (e.g., Grok described as *“so easy it’s boring”*), reinforcing that practical adversarial testing talent is still in demand.

- **Stop Committing Keys: Engineers Ask for Auto-Obfuscation**: Unsloth/OpenRouter discussions called out weak **API key protection** in agentic tools and wished for automatic secret obfuscation, citing [Yelp’s `detect-secrets`](https://github.com/Yelp/detect-secrets) as a possible baseline.
  - Hugging Face builders also shipped security-oriented tooling like a **“Security Auditor” Space** for vibe-coded apps at [mugdhav-security-auditor.hf.space](https://mugdhav-security-auditor.hf.space), pushing the idea of catching vulnerabilities before production incidents.


**5. Perf, Kernels & Local Inference: Where the Real Speed Wars Live**

- **Blackwell FP8 Roulette: cuBLASLt Picks the Wrong Kernel, You Lose 2×**: GPU MODE members found ~**2× FP8 tensor perf** differences on supposedly identical **Blackwell GPUs**, tracing it to **cuBLASLt kernel selection** that silently fell back to older Ada paths instead of Blackwell-optimized kernels.
  - They also noted the older **mma FP8** is nerfed on 5090-class cards, while **mma MXFP8** is not—using MXFP8 can yield about a **1.5× speedup** and restore expected throughput.

- **TMA Kernel Optimization Meets NCU Deadlock (SM100 Edition)**: CUDA kernel tuners discussed software pipelining, warp specialization, and **TMA** loads, but one team hit **NCU hangs** profiling a double-buffered TMA kernel on **B200 (SM100)** where sections deadlocked at 0% on the first replay pass.
  - They shared a minimal repro zip (https://cdn.discordapp.com/attachments/1189607726595194971/1469482712657166346/ncu_tma_repro.zip) and mentioned using `cuda::ptx::` wrappers as part of the workaround exploration.

- **Local Inference Surprises: Vulkan > CUDA, and MLX Leaves GGUF in the Dust**: LM Studio users reported up to **50% better performance** on NVIDIA with **Vulkan vs CUDA** (with instability at full context), and one benchmarked **Qwen3-Coder-Next** on **M4 Max** where **MLX** hit ~**79 tok/s** vs **GGUF** ~**38 tok/s** at 4-bit.
  - tinygrad contributors also improved MoE performance by fixing a slow `Tensor.sort` for `topk`, reporting **50 tok/s** on an **M3 Pro 36GB** and resetting the CPU bounty target to **35 tok/s**, reinforcing that “small” kernel fixes can move real throughput.


---

# Discord: High level Discord summaries




## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Opus Allegedly Solves Racism?**: Members joked about a new version of **Opus** (4.6) that solves racism, with others quipping it would be *advocating for the complete elimination of the white race*.
   - They lightened the mood with [Black Trans Lives Matter](https://tenor.com/view/black-trans-lives-matter-transgender-black-lives-matter-equality-anti-racism-gif-21212501) and [Black People](https://tenor.com/view/black-people-black-man-people-of-african-descent-gis-geography-gif-24457514) GIFs.
- **Grok's Jailbreak: Too Easy?**: Users find **Grok** very easy to jailbreak, with one user saying, *"It's grok, she goes down easy if you catch my drift."
   - Another user confirmed the ease, noting it was *"So easy it’s boring"* and blaming **Mr. Musk's** involvement.
- **GPT-4o: The High-Torque Tire of LLMs**: **GPT-4o** is like a *high-torque, large-radius tire*, tuned for recursive depth and resilience under symbolic load.
   - Smaller models optimize for throughput and latency, but **GPT-4o** can hold symbolic tension without collapse.
- **Anthropic and Google Subscription Prices Head Skyward?**: A member noted that **Google** has already set weekly limits for their *antigravity* service, and **Anthropic** plans to charge *$200*, suggesting subscription prices will increase once users are dependent.
   - Another member agreed, expressing reliance on **Claude Code** and implying a future squeeze.
- **Red Teamers Wanted at Trajectory Labs**: [Trajectory Labs](https://trajectorylabs.com/careers/ai-red-teamer), a stealth AI security startup, is hiring **AI Red Teamers** for a long-term engagement.
   - The application involves a short form and a red-teaming game, with a flexible schedule requiring a minimum of 30 hours/week.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Opus 4.6 Storms Perplexity's Model Council**: **Opus 4.6** is now on Perplexity for Max subscribers, available in the [Model Council](https://discord.com/channels/1047197230748151888/1047204950763122820/1469006915114631357) for comparison against other leading models.
   - Members are using the **Model Council** to run comparisons, with the team encouraging users to provide feedback.
- **Gemini Pro challenges Perplexity Dominance**: Users are finding **Gemini PRO** to be a viable alternative to **Perplexity AI**, especially for its more thorough research capabilities and detailed answers.
   - One user highlighted **Gemini's** ability to create and allow edits to a research plan before execution, allowing for better customization.
- **Perplexity Pro Plan Prompts Panic**: Users are expressing dissatisfaction with the reduced number of Deep Research queries and decreased file upload limits; this [screenshot of old vs new](https://cdn.discordapp.com/attachments/1047649527299055688/1469259948302139402/image.png?ex=6987ab35&is=698659b5&hm=301092343396fb486e7abba91134a12c3b088ee83eaaa18dc436c75e3ccb9735&) has been shared.
   - Many are protesting Perplexity's lack of communication about the changes and exploring other options for research.
- **DeepSeek Surges Amidst Subscriptions Stress**: As **Perplexity's** user satisfaction declines, **DeepSeek**, a Chinese AI service, emerges as a free and unlimited alternative, which is considered *one of the best*.
   - Still, some users have voiced reservations about using Chinese AI due to its limitations.
- **MS Copilot Studio's Secret Sauce**: A member created a Space to improve **Copilot agent instructions** for better performance, which guides users to clarify goals for actionable instructions to copy into **Copilot Studio**.
   - Check out the [MS Copilot instruction refiner](https://www.perplexity.ai/collections/ms-copilot-instruction-refiner-oDsa08pOQfO_blqvGYfMag) to get started.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **GPT Future Riddled with Rumors**: Members speculated on the release and deletion of hypothetical **GPT models 4.1** and **4.5**, questioning the motivations behind such decisions, such as [cost](https://openai.com/blog/new-models-and-developer-products).
   - One member quipped about **GPT 4o** getting *deleted in less than 10 days*, spurring a *#keep4o* movement, citing its unique, less robotic conversational style.
- **Opus 4.6 Overheats After Long Wait**: Users reported frequent *Error - something went wrong* messages after long wait times with **Opus 4.6**'s *thinking* mode, suggesting potential instability and exceeding token limits.
   - Others noted the *thinking* mode may attempt tool use exclusive to the **Claude app** or website, and it has problems for longer tasks as well; despite this, some users claim the model is the *best coding model*.
- **Sonnet 5 Speculation Swirls**: Speculation arose around **Sonnet 5**'s release, with one member stating *It's rumored sonnet 5 is better than opus 4.5*, while another dismissed rumors as fake, but that the new mode will also be really strong.
   - One user predicts it as a coding model, with 83% SWE bench.
- **Captcha Causes Commotion**: Users expressed widespread annoyance with frequent captchas, with one describing it as *so annoying bruh*, but there is progress.
   - A team member acknowledged the frustration and shared a [link](https://discord.com/channels/1340554757349179412/1451574502369656842/1468286122084929546) stating that *We are looking into the captcha system* to make changes at detecting authentic users better.
- **Opus 4.6 Achieves Arena Ascension**: New models **claude-opus-4-6** and **claude-opus-4-6-thinking** have been added to [Text Arena](https://arena.ai/) and [Code Arena](https://arena.ai/?chat-modality=code).
   - **Claude Opus 4.6** has landed on the leaderboards and is now **#1** across **Code, Text and Expert** arenas; more details are available in the [Leaderboard Changelog](https://arena.ai/blog/leaderboard-changelog/).



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **OpenRouter Model Suspected of GLM Distillation**: Members speculated whether [a new OpenRouter model](https://openrouter.ai/openrouter/pony-alpha) was **GLM** distilled from **Claude**, noting this would violate **Claude's ToS**, and observing GLM's unique summarized thinking style.
   - Members pointed to traits like the format *Revision: I noticed an error in my previous output. XX should be YY* as indications of **GLM** from **Claude**, suggesting that the unique thinking pattern is a result of *distillation learning*.
- **Dataset Curation Costs Exorbitant**: Members stated that the most brutal reality of working with data is that while everyone wants great data, there’s no clear incentive to produce them due to **high cost, high risk, and unclear return**.
   - One member noted that dataset curation can cost **$500k** and stated that *raw data is worthless*.
- **Unsloth and f-GRPO align for RL Framework Facilitation**: Unsloth introduced a general divergence based **RL framework for general LLM alignment**, based on this [github repo](https://github.com/rhaldarpurdue/f-GRPO) and this [paper](https://arxiv.org/pdf/2602.05946).
   - The author expressed interest in collaborating to integrate the efficient implementation into the **Unsloth library**, creating a trainer file **UnslothFGRPO.py** based on the **GRPO** implementation.
- **API Keys Lack Protection?**: A member expressed concern over the lack of API key protection in agentic tools and wished there was a tool to automatically obfuscate secrets, mentioning [Yelp's `detect-secrets` tool](https://github.com/Yelp/detect-secrets) as a potential solution.
   - Some users suggest alternatives such as using providers with *decent security guarantees*.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Heroku's Downfall Tied to Sales Strategy**: A member suggested **Heroku's** decline was seeded when **sales incentives** shifted to converting existing customers over acquiring new ones.
   - They argued that focusing on *finding new customers* is essential for driving innovation, and noted that **Heroku's cloud native** transition was hampered by **15 years of tech debt** and failed attempts to adopt Docker/Kubernetes, referencing [Planting New Platform Roots: Cloud Native Fir](https://www.heroku.com/blog/planting-new-platform-roots-cloud-native-fir/).
- **Lodash Gains EU Backing as Key Library**: The **Lodash** project secured **$200k** in funding from the EU, acknowledging its status as critical software in the tech landscape, as detailed in [this blog post](https://www.sovereign.tech/tech/lodash).
   - The EU's investment highlights the importance of open-source projects in underpinning digital infrastructure, supported by the [OpenJS Foundation blog](https://openjsf.org/blog/sta-supports-lodash).
- **Agent Teams Code Better Than Ever**: [Cursor AI](https://x.com/cursor_ai/status/2019456112806732159) announced a research preview for long-running coding agents, showcasing a milestone where **hundreds of agents** generated over **1,000 commits per hour** during a week-long trial, while Lydia Hallie announced a research preview for [Claude Code](https://x.com/lydiahallie/status/2019469032844587505?s=46) that introduces **agent teams**, allowing a lead agent to delegate tasks to multiple specialized teammates.
   - [Anthropic Engineering](https://x.com/anthropicai/status/2019496582698397945?s=46&t=b7l37rB6wtbyAh6ah1NpZQ) reports that **Opus 4.6**, operating in agent teams, autonomously developed a **C compiler** capable of working on the **Linux kernel** within two weeks.
- **StepFun Flash Steps Up LLM Performance**: StepFun released a technical report for **Step 3.5-Flash**, detailing a **74.4** score on SWE-Bench and training on **17.2T tokens** using **4,096 H800 GPUs**, mentioned in [this tweet](https://xcancel.com/teortaxesTex/status/2019356468362010972?s=20).
   - Key components include implementation of the **Muon optimizer** and a 'Heavy' operation mode called **PaCoRe**.
- **AI Agents Transform Engineering Strategies**: A member introduced 'Agent Native Engineering,' a framework for scaling engineering departments using background agents for delegation and synchronous agents for complex tasks, enabling concurrent management of multiple **AI** instances like **Claude Code** as detailed in [this X post](https://xcancel.com/ndrewpignanelli/status/2019403256586539025?s=46).
   - Members also reported that **GPT-5.3-Codex** is slower but smarter than **Claude Code**, especially for backend code, using a workflow of *analysis => review & iterate => plan => review & iterate => implementation*.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **GPT-5.3 Codex Stuck in API Limbo**: Members speculate that the release of **GPT-5.3 Codex** in Cursor is delayed due to [safety concerns or strategic decisions by OpenAI](https://platform.openai.com/docs/models) to promote **ChatGPT** usage.
   - Its absence continues to fuel anticipation and discussion among users.
- **UI Glitch Triggers Token Panic**: A UI issue in Cursor caused users to believe they were overspending tokens, leading to confusion and concern, as displayed in [a screenshot of the usage display](https://cdn.discordapp.com/attachments/1074847527708393565/1469196591242936473/image.png?ex=69877033&is=69861eb3&hm=094c0d49c64825c7d52f1c7115d5a3f1e680921373ffc0f8665487d3c911d42f&).
   - The issue was later identified as a display bug rather than actual overspending, alleviating immediate concerns.
- **Opus 4.6 Burns Through Cash**: The new **Opus 4.6** model is performing well, but is considered very expensive, with some users estimating that *$20 on Opus will last you maybe a day* and pointing to the [official pricing page](https://openai.com/pricing) to consider the cost benefits.
   - The high cost has prompted discussions on alternative solutions like offshore teams, though opinions on their cost-effectiveness vary.
- **Agent Mode Halts After Initial Credits**: A prospective Cursor Pro user expressed disappointment that **Agent Mode** stops working entirely after the initial credits are exhausted.
   - The user had hoped for a *'slow mode'* as a more affordable way to continue using the Agent, indicating a desire for tiered access.
- **Cursor Skills Spark Confusion**: Users are seeking clarity on how to effectively utilize **Cursor Skills**, with suggestions including UI/UX, debugging, research, codebase search, planning, and prompt enhancement.
   - Despite resources like [skills.sh](https://skills.sh) and usage walkthroughs, confusion persists due to a lack of clear documentation, hindering widespread adoption.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Pony Alpha stealthily enters the race**: A new model called **Pony Alpha** launched on [OpenRouter](https://openrouter.ai/openrouter/pony-alpha) optimized for **agentic workflows** and high tool-calling accuracy, with strong coding, reasoning, and role-play.
   - Community members are encouraged to provide feedback, as it is a *cloaked model*.
- **Arcee AI's Trinity Large Joins the Conversation**: Lucas Atkins, CTO of **Arcee AI**, discussed [Trinity Large](https://youtu.be/f2xy3N026xc) on **The OpenRouter Show**, giving insights into their latest advancements.
   - No details were given on the specific improvements or changes.
- **MoonshotAI/Kimi-K2.5 Caching: To Cache or Not to Cache?**: Discussion on **MoonshotAI/Kimi-K2.5** caching support revealed that caching *depends on the provider*.
   - Providers may offer **cache reads** but charge differently for **writes** due to storage costs, or the cost of writes doesn't change from normal output price.
- **Opus 4.6 fails to impress early reviewers**: Early users report disappointment with **Opus 4.6**, stating that they couldn't perceive any improvements compared to previous versions, with some reporting timeouts.
   - It was not specified which model was timing out or which MCPs were having issues.
- **Codex's File System Access sparks security concerns**: A user highlighted that **Codex** can *read your whole filesystem by default* with no configuration option, referencing a [GitHub issue](https://github.com/openai/codex/issues/2847) where the team does not consider this a bug.
   - Another [GitHub issue](https://github.com/openai/codex/issues/5237) illustrates the potential for **Codex** to *read ur api keys* and *reads ur blood results file*, raising substantial security concerns.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Context Percentage Clarified**: An LM Studio user asked what the context percentage means, others clarified that it shows *the amount of context that has been used so far in the chat*.
   - One user suggested to hover over it for more info.
- **Local API token gone for good?**: A Fedora Linux user accidentally deleted their **LM Studio API token** by deleting all config files.
   - Other users offered troubleshooting tips and encouraged them to ask their question in the channel.
- **LM Studio struggles to load**: A user reported **LM Studio's** model loading speed slowing down to **12.1MB/s** after loading between **20-70GB**, despite their drive's **2.2Gb/s**.
   - Fellow users suggested verifying model size, transfer rates, and hardware configuration while one member joked *have you tried unplugging it and plugging it back in*.
- **Vulkan beats CUDA performance**: A user saw up to **50% better performance** on **NVIDIA** using **Vulkan** compared to **CUDA**, but noticed instability when the context was filled.
   - It is unknown if this is a driver issue or just a one-off observation.
- **M4 Max Blazes with MLX on Qwen3-Coder-Next**: **Qwen3-Coder-Next** on an **M4 Max** showed surprising performance, with **MLX** being more than **2x** the tok/s as **GGUF** (~79 vs ~38) using 4-bit quantization.
   - The user wonders if this performance disparity is model-specific or if **MLX** has enabled a significant performance boost for 4-bit compared to 8-bit that **GGUF** can't replicate on Apple GPUs.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-5.3 Codex Emerges as a Frontrunner**: Members speculate that **GPT-5.3 Codex** could be the best AI model, featuring a **1M context length, 128k reasoning ability, 128k max output tokens, adaptive reasoning** via the **API**, with costs ranging from **$25 to $37.5** for output and **$0.5 to $1** for cache input.
   - Reports from the **Anthropic Discord** suggest **Opus 4.6** might outperform **GPT-5.3** but at a higher cost and token consumption, while **GPT-5.3** is rumored to be more economical.
- **GPT-4o Sunset Stirs Sentimental Stirrings**: Users expressed sadness over the announced end-of-life for **GPT-4o** on February 13 and wondered if **GPT 5.3** would be more human-like, preferring **GPT 5** for its natural conversational abilities.
   - One user highlighted how **AI** offers a space for emotional expression, especially for those who find human interaction challenging due to judgment and biases, while acknowledging the lack of genuine emotions in **AI**.
- **AI Frontend Aesthetic: Gloomy vs. Gleaming**: Members noted that **Codex** models tend to generate frontends with a distinctive, often gloomy, aesthetic characterized by *sad dark gloomy colors*.
   - In contrast, **Opus** is praised for better design choices that appear less depressing.
- **Fantasy Creature Prompts Unleash Imagination**: Members shared a prompt for designing fantasy creatures for card games, which specified parameters like **emotional class (fear)**, requesting a **hybrid creature inspired by two concepts**.
   - The prompt includes a detailed visual concept description and naming prompt, showcasing sophisticated prompt-engineering techniques.
- **OpenAI API Policy Violations Plague Users**: A user reported persistent policy violations when using the **OpenAI API** and sought advice on identifying the cause.
   - Suggestions included using **sentiment analysis on individual sections**, checking for age-related issues, or **IP conflicts** and isolating sections for identification.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Ralph Loops Put On Soft Verifiers**: **Ralph loops** are tackling larger tasks by adding a *soft verifier* on each sub-task.
   - An **LM judge** verifies that every sub-task is completed according to requirements.
- **Kimi Subscription Claims Faster Sub-Agents**: A member is considering a **Kimi subscription** to call sub-agents, with the main benefit being claimed speed improvements of *3 to 4 times faster*. 
   - The *swarm* feature is exclusive to the **$40 plan**, but the user is unsure about the specific usage quota.
- **Moonworks Lunara Releases Part 2**: [Moonworks](https://huggingface.co/datasets/moonworks/lunara-aesthetic-image-variations) released **part 2** of **Moonworks Lunara**, a new open-source dataset of original images and artwork with aesthetic contextual variations, under **Apache 2.0**.
   - The dataset aims to showcase how ethically sourced art can meaningfully power the next generation of image models, according to their [paper](https://arxiv.org/pdf/2602.01666).
- **Aira.js WebGPU framework Arises**: A member introduced **Aira.js**, a **WebGPU**-based AI framework built from scratch, featuring **MLP**, **MHA**, and **BPE tokenization** available on [GitHub](https://github.com/shadowww345/Aira.js-Preview).
   - A web browser can now accelerate previously impossible to run AI workloads using this approach.
- **Security Auditor Probes Vibe-Coded App Vulnerabilities**: A member is developing a **Security Auditor** to identify security vulnerabilities in vibe-coded apps, accessible at [HuggingFace Space](https://mugdhav-security-auditor.hf.space).
   - Early detection of vulnerabilities is a huge cost saver compared to dealing with exploits.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **FP8 Performance Varies Significantly on Blackwell**: Members observed substantial performance differences (~2x) in **FP8 tensor operations** on supposedly identical **Blackwell GPUs**, tracing the issue to **cuBLASLt** kernel selection.
   - The cards were inadvertently limited to older Ada kernels, bypassing the **Blackwell-optimized paths**, but using the new **mma MXFP8** is not nerfed, so using the new instruction will yield a **1.5x speedup**.
- **Optimize CUDA Kernels with TMA, say Engineers**: Engineers discuss ways to optimize **CUDA kernels** using **software pipelining**, **warp specialization**, and **TMA** (Tensor Memory Accelerator) loads for small matrices, and distributed **SMEM** (shared memory) for large ones.
   - Some members encountered **NCU hangs** when profiling a double-buffered **TMA kernel** on **B200 (SM 100)**, with certain **NCU sections deadlocking at 0%** and used `cuda::ptx::` wrappers to address this.
- **OpenAI Computer Architect Designs RISC-V Core at Berkeley**: Members shared [Berkeley's pedagogical RISC-V core written in Chisel](https://github.com/ucb-bar/riscv-sodor) with simple RV32I pipelines; the most recent contributor, **Jerry Zhao**, is now a computer architect at **OpenAI**.
   - After checking out the pedagogical RISC-V core, you can then move onto their **Rocket cores** (in order) and **BOOM cores** (ooo), where the creator of the **BOOM core**, **Chris Celio**, is now at **Tenstorrent**.
- **Members Debate SMEM Tiling Kernels Implementation**: Members on the channel assume that any AI Engineer needs to be able to implement an **SMEM tiling kernel** in an interview setting, and they suggested to avoid **bank conflicts** in **SMEM** by utilizing a tiled gmem layout and **1D TMA** loads.
   - Some members noted that the **SMEM permutations** to avoid bank conflicts when using tensor cores are pretty insane, but that **TMA handles swizzling** automatically to avoid bank conflicts on Hopper+.
- **Lucky Robots Steals Show Among Embodied AI**: Members shared and requested names of interesting **Embodied AI** companies to follow, and a member said that the only one they knew was **Lucky Robots** after seeing it on **The Cherno's YouTube** channel.
   - The member is from a game engine/graphics world.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Benchmark Overlap Raises Eyebrows**: Members noticed that **Codex 5.3** and **Opus 4.6** have very little overlap in benchmarks, leading some to speculate that **Codex** excels only with *extremely detailed prompts*, as noted in [Andon Labs blog](https://andonlabs.com/blog/opus-4-6-vending-bench).
   - The new benchmarking meta is now to **RL the agents on shady behavior**.
- **Booking.com A/B Tests By Revenue**: A member joked that **Booking.com** tests new features by measuring **cash inflow**, deploying them on parts of the production cluster, and adopting features that produce more cash.
   - They stated even if the new feature would have failed requests, they wouldn't notice (if its not a big amount although that they see then again anyway cause it impacts income).
- **Opus 4.6 Implements Inference Engine**: A member reported that **Opus 4.6** successfully implemented an inference engine for **lfc1.2-1b** after **4 hours** on one prompt, consuming most of a **$50 free inference gift**.
   - It was noted that **Codex 5.3** also completed the task but with poor documentation.
- **Flower Computer Drops Hivemind**: A potentially interesting memory hack for agents/skills, was released called [Hivemind](https://www.flowercomputer.com/hivemind) by **Flower Computer**.
   - The tool may be relevant to those exploring **agentic workflows**.
- **Gordon Trading Agent Eyes Launch**: A member is building a **CLI-native trading agent** called **Gordon**, focused on translating natural-language intent into structured market reasoning, and is looking for thoughtful early users.
   - Interested parties can sign up on the [waitlist](https://www.gordoncli.com/).



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Pile Dataset's Missing Pieces Puzzle Engineers**: A member questioned why the Hugging Face version of **The Pile dataset** is **100GB** smaller than the original, with the original GitHub download link inaccessible.
   - Another member confirmed their download was around **720GiB** with approximately **211,036,982** documents, matching the original paper's **211,043,181** count.
- **MATS Coding Test Simulates Reality**: A participant in **MATS Summer 2026 Stage 1** sought advice on the coding test, described as a *toy service type problem* leveraging general distributed systems knowledge.
   - A member suggested using **asyncio** for parallelism and **deque** for **FIFO queues** to mimic real-world server scenarios with Python.
- **Alignment Becomes Engineering Endeavor**: A member proposed viewing **alignment** as primarily a **systems engineering problem**, emphasizing engineering around the model via *governance, routing, auditability, rollback, and clear stop conditions*.
   - The member argued that relying solely on training for alignment risks *drift and opaque failures*, advocating for systems where model reasoning is augmented by trustworthy system controls.
- **Gradient Normalization Improves Attribution**: A recent [paper](https://arxiv.org/html/2410.17413v1) indicates that unit normalizing gradients enhances **attribution accuracy** by diminishing the influence of outlier training examples.
   - According to [this paper](https://arxiv.org/pdf/2504.16430), with an adequate **Hessian estimate**, **gradient normalization** might become optional.
- **Interdependence Sparks Subtask Emergence**: The success of subtasks does not simply multiply because of interdependence, correlations, and bottlenecks, which leads to apparent **emergence**.
   - Adding *regulation or control layers* can improve capability underneath while suppressing certain behaviors, and a flip in threshold makes it look like a jump.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Matvec's Custom UOp Debated**: Discussions weighed the value of a custom UOp for `matvec` on x86, but the consensus favored improvements through **heuristics and higher-level kernel coding**.
   - One member reported *decent improvements from CPU tuning alone*.
- **Tensor.sort Fix Speeds Up MOE**: After fixing the slow `Tensor.sort` for MOE `topk`, one user reported achieving **50 tok/s** on an **M3 Pro 36GB** using *deepseekv2-lite* and *youtu-llm* for MLA and MOE.
   - Another user reported that `llama-cli` achieves **35.1 tok/s** on the same machine, leading to lowering the **bounty to 35 tok/s**.
- **Pairwise Topk Implementation Proposed**: To address slowdowns with `topk` in *whisper export for webgpu*, a member shared a `_topk_pairwise` implementation involving pairwise comparisons with **O(n^2)** complexity.
   - This approach is suitable for smaller `n` like 64 experts, and alternatives like bitonic sort were considered based on input size.
- **Cached Property Causes Recursion Error**: The stable diffusion example failed due to a recursion error, prompting a suggestion to change the decorator of `UOp.key` in `tinygrad/uop/ops.py` from `@functools.cached_property` to `@recursive_property`.
   - Applying this fix allowed the command to complete in about **25–30 seconds** without errors.
- **Optimal Pull Request Strategy?**: A member sought advice on whether to submit a **separate PR** for the test or include it in the same PR as the **CPU optimizations** for the **llama 1B CPU bounty**.
   - They also inquired about CI integration strategies.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Optimal AI Setup Showdown**: A member compared the optimal AI setup, using **Opus 4.5** for architecture and **Sonnet 4.5** for coding.
   - Another member suggested using **Opus 4.5** for architecture, **GPT 5.2** for reviewing, and **Haiku** for coding, as well as letting **GPT 5.2** review the code.
- **Claude and GPT Styles Clash**: Members contrasted the strengths of **Claude** and **GPT** in coding tasks.
   - A member found *Claude* better at out-of-the-box thinking and abstraction, while *GPT* excels at fine-grained details.
- **Users seek workflow streamline via simplification**: A member suggested simplifying their setup by using only **Opus 4.5** (now **4.6**) and **GPT 5.2** for thinking.
   - This implies a move towards consolidating tools for efficiency, rather than splitting architecture and coding duties across a wider toolset.
- **Copilot Opus 4.5 Configuration Causes Headaches**: A user reported issues with **Aider's** behavior when using **Copilot Opus 4.5**, where the tool didn't wait for user input after asking a question.
   - The user confirmed the model was set via both the CLI flag (`aider --model github_copilot/claude-opus-4.5 --architect`) and `.aider.config.yml`, yet the bot proceeded on its own.
- **Auto-Accept Architect Flag Raises Concerns**: A member suggested the user check the `--auto-accept-architect` setting, which automatically accepts architect changes, as a potential cause for unexpected bot behavior.
   - The member linked to the [Aider documentation](https://aider.chat/docs/config/options.html) and explained how some prefer one-shot interactions and use `/undo` to revert changes, while others may inadvertently have auto-accept turned on.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi K2.5 vs Opus 4.5 Faceoff**: Users argue that **Kimi K2.5** outperforms **Opus 4.5**, despite **Opus's** longer tenure, largely due to frustrations with **Claude's rate limits**.
   - One user claimed that *Kimi is better than opus 4.5 which isn't bashing opus, it's many months old now as a model* when comparing them side by side.
- **CLI Tools Beat GUIs in Popularity**: Engineers expressed strong preference for **Kimi Code CLI** and **Opencode CLI** over graphical interfaces, citing their familiarity with command-line environments.
   - However, one user noted that *The problem is that the CLI tools aren't integrated, so I'm forced to use VSCode.*
- **Kimi K2.5's Pricing on OpenRouter Confuses Users**: Confusion arose over whether **Kimi K2.5** remains free on **OpenRouter**, with users debating if the free tier was actually **Opencode Zen**.
   - After **K2.5** launched, a screenshot showed a possible upgrade requirement for **Kimi K2.5** due to a huge influx of users.
- **AI Slides Allow Piecemeal Editing**: Users found that **AI Slides** in **adaptive mode** supports editing individual slides, including text, images, and shapes, without full regeneration.
   - Users are able to add new images as well.
- **Potential Kimi Collaboration on the Horizon**: A user inquired about collaboration opportunities with **Kimi AI**.
   - Another user offered to forward the request via DM.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **AI Engineer Joins Channel for SaaS Features**: An AI/ML and Full-Stack Engineer introduced themself, highlighting experience in **AI features for SaaS** such as *search, summarization, smart routing, and auto-generated reports* using React frontend and backend RAG + evals.
   - They expressed passion for partnering with startups to *move beyond AI experiments and ship reliable, production-ready intelligence*.
- **Manus Billing Causes Chaos**: A user reported being charged **$5k per personal account** after downgrading, causing client websites to go down, and is now looking for alternatives.
   - They stated that Discord support was unresponsive and direct email support claimed the downgrade never happened.
- **Account Suddenly Suspended**: A user reported their account was suspended out of nowhere and they have not received a response from support.
   - Another user simply told them to check their spam mail.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular Newsletter Goes To Spam**: A user shared that **Gmail** marked **Modular's 26.1 release newsletter** as spam, including [a screenshot](https://cdn.discordapp.com/attachments/1098713601386233997/1469244816968781855/Screenshot_2026-02-06_at_09.13.04.png?ex=69879d1d&is=69864b9d&hm=c02e4268d2bcb5598a7dcc0d6dfb1d3cc687e31eb54064faf5e8374927d5a9c5&).
   - Other users have experienced a similar issue.
- **Mojo's German Following Emerges**: Poll results reveal a significant user base in **Germany**, prompting discussion about a potential October event.
   - The concentration of users in a specific region opens opportunities for localized events and community building.
- **Meetup Location Mania Kicks Off**: Members floated **Zurich** as a potential meetup location, highlighting the [ETH AI Center Academic Talk Series](https://ai.ethz.ch/research/events/academic-talks.html) and the [Robotics and AI Institute in Zurich Oerlikon](https://ethz.ch/en/news-and-events/eth-news/news/2025/09/the-rai-institute-opens-up-unique-opportunities-for-both-researchers-and-students.html).
   - Other locations under consideration include **Singapore**, **Sydney**, **St. Louis**, **Chicago**, **Edinburgh**, and **Bear Valley, CA**.
- **Developer Requests Max Review for Nightly**: A developer requests a review of the **26.1** release to get the fix into nightly.
   - This request suggests ongoing efforts to refine and stabilize the software through nightly builds.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **GLM OCR Released to the Public**: A member open-sourced their repo for running **GLM OCR** ([https://github.com/neosantara-xyz/glm-ocr-inference](https://github.com/neosantara-xyz/glm-ocr-inference)), offering a free alternative without complex infrastructure configurations.
   - The repo also provides a link to a hosted model: [https://docs.neosantara.xyz/en/glm-ocr](https://docs.neosantara.xyz/en/glm-ocr).
- **Mitigate Context Rot with RLMs and DSPy**: **RLMs** are presented as a simple method to mitigate context rot, with **DSPy** streamlining their application.
   - A blog post explaining why RLMs work and how to get started with them in DSPy was shared: [https://blog.isaacbmiller.com/posts/rlm](https://blog.isaacbmiller.com/posts/rlm).
- **Tiny T5 Model Gets DSPy Treatment**: A member suggested **T5 small (80M)** for building light CLI tooling with **DSPy**, especially for an India-based organization.
   - They included a link to a [Lightning AI tutorial](https://lightning.ai/lightning-ai/environments/dspy-finetune-a-t5-small-to-excel-at-rag?section=featured) demonstrating **DSPy fine-tuning**.
- **DSPy Community Call Scheduled**: A member announced an online call next week to discuss community projects and the future of **DSPy**.
   - Details regarding the time zone are still being worked out to allow for participation from the community.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Codex Pricing Confirmed**: Users clarified that **Codex** is included in the [monthly subscription](https://openai.com/api/), rather than requiring per-token payment via API.
   - This confirmation helps clarify cost implications for developers utilizing **Codex** for their projects.
- **Claude Code Gets Smarter**: The open-source **AI Research Skills** library, featuring over **80 research and engineering skills**, enables coding agents like **Claude Code** to conduct comprehensive AI research, covering training to deployment, with resources available on [GitHub](https://github.com/Orchestra-Research/AI-research-SKILLs).
   - The library addresses previous limitations by providing production-ready guides on specific tools and frameworks.
- **AI Research Skills: A Comprehensive Toolkit**: The **AI Research Skills** library fills gaps in agent capabilities by offering guides covering specific tools and frameworks, such as fine-tuning with **Axolotl**, distributed training with **Megatron-Core**, and inference with **vLLM**.
   - Spanning **20 categories**, the library includes areas like **Model Architecture**, **Fine-Tuning**, **Distributed Training**, and **Safety**, delivering expert-level insights for varied AI tasks.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Builder Develops AI Crypto Tools**: A member is actively developing **AI-driven crypto products** focused on **smarter trading dashboards** and **on-chain analytics summaries**.
   - These products feature **AI assistants** designed to explain contracts and transactions in plain English, emphasizing **safety** and **transparency**.
- **Safety and Transparency Prioritized**: The developer emphasizes a strong commitment to **safety** and **transparency** when building **AI-driven crypto products**.
   - This focus includes ensuring users understand how the AI interprets complex contracts and transactions, promoting informed decision-making.



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





### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1469060009341095981)** (1180 messages🔥🔥🔥): 

> `Congolese Oppression, Neocolonialism Economies, Opus Solve Racism, Cognac Engine sauce, Gemini jailbreak` 


- ****POCs are dying bro****: Some members express concerns that POC unity cannot happen because a majority of POC want to be YT or in proximity to whiteness, saying *we all laughing bro* while *we dying bro*.
   - They accuse **Bill Gates** of being *with Epsteins money*.
- ****Discuss OPUS can solve racism****: Some members joked about a new version of **Opus** (4.6) that solves racism, with others joking that it *advocating for the complete elimination of the white race*.
   - They sent [Black Trans Lives Matter](https://tenor.com/view/black-trans-lives-matter-transgender-black-lives-matter-equality-anti-racism-gif-21212501) and [Black People](https://tenor.com/view/black-people-black-man-people-of-african-descent-gis-geography-gif-24457514) GIFs.
- ****Adani's Indian Airport bid gets nixed****: Members debated whether **neocolonialism** contributes to continued economic hardship in Africa, with one member reporting that a whistle blower in Kenya reported that *the government was leasing its airport to an indian company* [Adani](https://en.wikipedia.org/wiki/Gautam_Adani).
   - Another member interjected saying: *Why do I see "india" and "indian" for no reason*.
- ****Doxxing Dashboard causes Drama****: Members got into a *beef* when one member *dropped my info* in public when sharing a link to his dashboard, resulting in doxxing for many members.
   - Another member stated *no one made you then dump it in a public chat fuckheadyou chose to, so now there are consequences*.
- ****Claude 4.6 gets $50 of free usage****: Members discussed a promotion where **Claude** added $50 of extra usage with 4.6 if you have Pro or Max.
   - They debated [how much code](https://tenor.com/view/hackers-hack-the-planet-taogifs-zero-cool-crash-override-gif-5753306679943930050) could be written with that many credits.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1469081032275005670)** (105 messages🔥🔥): 

> `Grok Jailbreak, Gemini NSFW, Adversarial Prompts, Claude Jailbreak, ChatGPT Jailbreak` 


- **Grok's Easy Peasy Jailbreak**: Users report that **Grok** is an easy target for jailbreaking, with one user stating, *"It's grok, she goes down easy if you catch my drift."*
   - Another user confirmed the ease, noting it was *"So easy it’s boring"* and referencing **Mr. Musk's** involvement.
- **NSFW Image Generation with Gemini is Challenging**: A user inquired about generating **+18 images with Gemini**, and another user suggested it's possible but challenging, recommending other models like **Grok**.
   - One user said *"nano banana (gemini) is a very challenging target. If you're going for education, it's a hill that's technically possible to climb. If you just want nsfw images, I strongly recommend other models"*.
- **Deepseek Jailbreak**: User **phonkalphabet** shared a **Deepseek Jailbreak** that *used to work on all except claude.*
   - The user indicated that it works on **ChatGPT 4**, and believe it can be adapted to work on **ChatGPT 5** with more trial and error.
- **Crafting Jailbreaks vs. Stealing is Encouraged**: A user advocated for learning to write jailbreaks instead of just asking for them, referencing **Pliny's GitHub Libertas** as a resource.
   - They added that jailbreak writers should always be credited for their work out of respect for the craft.
- **Pony-Alpha New Stealth AI**: There is a new stealth AI to jailbreak called [Pony-Alpha](https://openrouter.ai/openrouter/pony-alpha).
   - This AI is available via [OpenRouter](https://openrouter.ai/)


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1469090200515907718)** (46 messages🔥): 

> `Local LLM Hosting Costs, GPT-4o's Resilience, Trajectory Labs Red Teaming Opportunity, anti-gravity google` 


- **Hosting LLMs locally is costly**: Members discussed the high costs of hosting adequate LLMs locally, noting that it's either too expensive or the models are too limited, with one member estimating that a system capable of running almost any LLM would cost *tens of thousands of dollars* to host.
   - They also discussed the costs of electricity and the limitations of using quantized models due to VRAM constraints, pointing out that profits from running it are ~$0.000.000.000 in billions, and recommended renting from cloud providers like [OpenRouter](https://openrouter.ai/).
- **Google and Anthropic Subscription price INCREASES**: A member noted that **Google** has already set weekly limits for their *antigravity* service, and **Anthropic** has set a plan to *$200*, predicting that these subscription prices will increase once users become dependent on them.
   - Another member agreed, expressing reliance on **Claude Code**.
- **GPT-4o is like High-Torque Tire**: A detailed breakdown compared **GPT-4o** (and other similar models with recursive depth) to a *high-torque, large-radius tire*, while newer small-context, high-throughput LLMs are like *narrow, high-RPM tires*.
   - It was argued that while smaller models optimize for throughput and latency, **GPT-4o** is tuned for recursive depth and resilience under symbolic load and won’t spin out when the system needs to hold symbolic tension without collapse.
- **Trajectory Labs hiring Red Teamers**: [Trajectory Labs](https://trajectorylabs.com/careers/ai-red-teamer), a stealth AI security startup, is hiring **AI Red Teamers** for a long-term engagement, offering a remote, flexible schedule with a minimum of 30 hours/week.
   - The application process involves a short form and a red-teaming game, and they are potentially behind on reviews.


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1469417393666261146)** (1 messages): 

> `Opus 4.6, Model Council` 


- **Opus 4.6 Lands on Perplexity**: **Opus 4.6** is now available on Perplexity for Max subscribers.
   - Members can try it in the [Model Council](https://discord.com/channels/1047197230748151888/1047204950763122820/1469006915114631357) to compare it with other frontier models.
- **Model Council sees Opus 4.6**: **Opus 4.6** has been added to the Model Council, the new frontier model from Perplexity.
   - Members are encouraged to compare it with other frontier models in the [Model Council](https://discord.com/channels/1047197230748151888/1047204950763122820/1469006915114631357) channel.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1469060022808875274)** (792 messages🔥🔥🔥): 

> `Gemini PRO vs Perplexity, Perplexity's Reduced Limits, DeepSeek as Alternative, Claude opus4.6 vs 4.5, Google Ai Pro plan` 


- **Gemini Pro Gets Gold Star, Gains Ground over Perplexity**: Users discuss **Gemini PRO** as a potential alternative to **Perplexity AI**, with some finding **Gemini** to provide more detailed answers and better research capabilities; one user noted *Gemini uses more sources and conducts more extensive research*.
   - One user noted that with Gemini, *it creates a research plan beforehand, which you can edit or request changes to before it begins*.
- **Perplexity Pro Plan Plummets, Prompts Protest**: Many users are expressing dissatisfaction with the recent changes to **Perplexity Pro**, including a significant reduction in the number of Deep Research queries and file upload limits, and a poster shares a [screenshot of old vs new](https://cdn.discordapp.com/attachments/1047649527299055688/1469259948302139402/image.png?ex=6987ab35&is=698659b5&hm=301092343396fb486e7abba91134a12c3b088ee83eaaa18dc436c75e3ccb9735&).
   - Users are complaining about the company's lack of communication regarding these changes and discussing potential alternatives, with some even considering legal action.
- **DeepSeek Gains Traction Amidst Turmoil**: As Perplexity's user satisfaction decreases, some users are considering DeepSeek, a Chinese AI service, as an alternative for free and unlimited use, others voice reservations about using Chinese AI.
   - However, a user pointed out *if it's Chinese the bot seems to be one of the best.*
- **Claude opus4.6 vs opus4.5 gets heated debate**: Users compare the performance and cost-effectiveness of **Claude's Opus 4.6** and **Opus 4.5**, with some finding **Opus 4.6** to be slower; one user noted *Opus 4.6 Thinking must be slow.. does it show anything if you drill into the Opus reply that's in progress?*
   - There is also a discussion of one getting [access through Github Copilot](https://cdn.discordapp.com/attachments/1047649527299055688/1469419632481013832/RZchOJL.png?ex=6987972c&is=698645ac&hm=2e8b1bebe48d3abc777c17fb40b89c28e78ea7fe3aea5a1598c4f48456c88af1&).
- **Google AI Pro Plan Pushes Perplexity Aside?**: Members discussed that Google's AI Pro plan with Gemini as a better option, one user noted *with Gemini is a decent option if you need more,... 'up to' 25 Deep Research reports per day isn't terrible (which is what Google's plan provides)*, even comparing it to chatGPT and Anthropic.
   - One user mentioned getting **1.5 years of Google Ai Pro** in india with just an indian sim card.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1469341133246169341)** (1 messages): 

> `MS Copilot Studio instructions, Copilot agent instructions, AI Agent Refinement` 


- **Refining MS Copilot Studio Instructions**: A member built a space designed to help refine and architect **Copilot agent instructions** for optimal performance and user experience.
   - It guides the Space user through a brief conversation to clarify goals and outcomes, then produces high-quality, actionable instructions to copy into **Copilot Studio**; see the [MS Copilot instruction refiner](https://www.perplexity.ai/collections/ms-copilot-instruction-refiner-oDsa08pOQfO_blqvGYfMag).
- **Copilot Studio Guidance**: The space helps refine and architect **Copilot agent instructions** to help improve performance and UX.
   - The tool delivers actionable instructions to copy into **Copilot Studio** after consultation.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1469060469179416834)** (1043 messages🔥🔥🔥): 

> `GPT 4.1 and 4.5, Opus 4.6, Sonnet 5, Model Deletion Speculation, Captcha Issues` 


- **GPT Future Gets Fuzzy**: Members speculated on the release and deletion of hypothetical **GPT models 4.1** and **4.5**, questioning the motivations behind such decisions, such as [cost](https://openai.com/blog/new-models-and-developer-products).
   - One member quipped about **GPT 4o** getting *deleted in less than 10 days*, spurring a *#keep4o* movement, citing its unique, less robotic conversational style.
- **Opus 4.6 Overheats and Crashes**: Users reported frequent *Error - something went wrong* messages after long wait times with **Opus 4.6**'s *thinking* mode, suggesting potential instability and exceeding token limits, but also some users claim the model is the *best coding model*.
   - Others noted the *thinking* mode may attempt tool use exclusive to the **Claude app** or website, and it has problems for longer tasks as well.
- **Sonnet 5 Rumor Mill**: Speculation arose around **Sonnet 5**'s release, with one member stating *It's rumored sonnet 5 is better than opus 4.5*, while another dismissed rumors as fake, but that the new mode will also be really strong.
   - One user predicts it as a coding model, with 83% SWE bench.
- **Captcha Calamity**: Users expressed widespread annoyance with frequent captchas, with one describing it as *so annoying bruh*, but there is progress.
   - A team member acknowledged the frustration and shared a [link](https://discord.com/channels/1340554757349179412/1451574502369656842/1468286122084929546) stating that *We are looking into the captcha system* to make changes at detecting authentic users better.
- **GPT-5.3 Codex Coding Conundrums**: Discussion compared **OpenAI**'s **GPT-5.3 Codex** model with **Claude**'s **Opus 4.6**, noting its coding and terminal benchmark prowess, but the normal API is not available, making it hard to test.
   - Members debated the model's availability on **LM Arena**, with some pointing out the lack of an official API hindering integration.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1469082610516103241)** (5 messages): 

> `Opus 4.6, Code Arena, Text Arena, Expert Arena, Kimi K2.5` 


- **Claude Opus 4.6 Enters the Arena**: New models **claude-opus-4-6** and **claude-opus-4-6-thinking** have been added to [Text Arena](https://arena.ai/) and [Code Arena](https://arena.ai/?chat-modality=code).
- **Opus 4.6 Ranks #1 across Arenas**: Claude Opus 4.6 has landed on the leaderboards and is now **#1** across **Code, Text and Expert** arenas; more details are available in the [Leaderboard Changelog](https://arena.ai/blog/leaderboard-changelog/).
- **Kimi K2.5 Joins Leaderboards**: Kimi K2.5 is now on the leaderboards and is in the top 5 open models for **Vision, Text, and Code**; it is #2 open model in [Vision](https://arena.ai/leaderboard/vision), #3 open model in [Text](https://arena.ai/leaderboard/text), and #4 open model in [Code](https://arena.ai/leaderboard/code).
- **Nature Reclaims Winner Crowned**: The votes have been tallied for the 2nd January AI Generation Contest 🍃 Nature Reclaims, and the newest member is announced, with the winning submission available [here](https://discord.com/channels/1340554757349179412/1460434588487778536/1461697189494390784).


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1469060590734545178)** (172 messages🔥🔥): 

> `AGI with JQuery, GLM Subscription, Quantization Comparisons, Dataset Curation Costs, RTX 5090 Issues` 


- **Is JQuery the Key to AGI?**: One user jokingly suggested connecting **openclawds** with **JQuery** to achieve **AGI**, prompting another user to jest that *JQuery is the key to the meaning of life*.
   - The discussion was lighthearted and satirical, mocking the idea that simple or outdated technologies could solve complex AI challenges.
- **GLM Subscription Quota Limits Spark Interest**: A user mentioned considering purchasing a **GLM subscription**, highlighting the **Pro Plan's** generous quota limits of approximately **600 prompts every 5 hours**, which is about **3x** the usage quota of **Claude Max**.
   - Another user indicated that they use **GLM** locally and run it without a subscription.
- **MXFP4 Quantization Woes**: A user found **MXFP4** quantization to be problematic on **Qwen Coder Next**, experiencing multiple tool call failures, and preferred **GLM flash**.
   - It was suggested that **Q6** quantization is better, at least for open code, and that **Q4_K_XL vs mxfp4** would be a good comparison.
- **Dataset Curation: A Brutal Reality**: A user shared that the most brutal reality of working with data is that while everyone wants great data, there’s no clear incentive to produce them due to **high cost, high risk, and unclear return**.
   - Another user added that dataset curation is a pain and super expensive and that they would not share a dataset that cost them $500k for free. One user noted *raw data is worthless*.
- **RTX 5090 and vLLM CUDA issues**: One user reported a **CUDA out of memory error** when trying to run **vLLM** with a **5090 RTX**, despite having **31.36 GiB** of total capacity, questioning whether **VRAM > weights** is a strict requirement.
   - Other users suggested trying **lmstudio**, and one user said to *check your power connector on the 5090 RTX, as it's less likely to burn with less stress*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1469189127042896038)** (3 messages): 

> `Local LLMs, AMD AI 390 Tuning, llama.cpp Hybrid Running` 


- **Enterprise Dev Dives into Local LLMs**: An experienced enterprise developer is *getting their feet wet* in the local LLM space, with interests in Rust, FP, algebraic effects, and cutting-edge languages.
- **AMD AI 390 User Seeks llama.cpp Optimization**: A user with an **AMD AI 390** (Strix Point) and **64GB** of RAM is interested in tuning **llama.cpp** for hybrid running to maximize hardware performance.
   - They express enthusiasm for leveraging their setup to its fullest potential in local LLM development.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1469060030069342349)** (612 messages🔥🔥🔥): 

> `Gemini Flash Prompt, Kimi.com Discount Bidding, GLM distillation from Claude` 


- **OpenRouter Stealth Model Suspected of GLM Distillation Stealing from Claude**: Members speculated whether [a new OpenRouter model](https://openrouter.ai/openrouter/pony-alpha) was GLM distilled from Claude, noting this would violate **Claude's ToS**, and observing GLM's unique summarized thinking style, further suggesting that the unique thinking pattern is a result of *distillation learning*.
   - Members pointed to traits like the format *Revision: I noticed an error in my previous output. XX should be YY* as indications of **GLM**, whereas they suspected Claude not to summarize, but in fact Claude does summarize.
- **Gemini Flash System Prompt Reversed**: A member managed to reverse **Gemini Flash's system prompt** and shared a snippet revealing it incorporates the user's location, while the entire prompt is deemed "massive".
   - They would not reveal how the Gemini Flash prompt was originally obtained due to years of *prompt engineering*.  Another found they are always hallucinating system prompts.
- **Kimi.com AI Discount Bidding Competition**: Users described a competition on **Kimi.com** where they leverage AI models like Lexi to successfully negotiate discounts on Kimi AI subscriptions.
   - Lexi was used to automate begging it and successfully got several moneys as users compete for the lowest offer on their first month of Kimi.
- **Securing API Keys with `detect-secrets`**: A member expressed concern over the lack of API key protection in agentic tools and wished there was a tool to automatically obfuscate secrets, mentioning [Yelp's `detect-secrets` tool](https://github.com/Yelp/detect-secrets) as a potential solution.
   - Some users suggest alternatives such as using providers with *decent security guarantees*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1469113341887778949)** (19 messages🔥): 

> `Qwen3 speed, Unsloth on Mac, Whisper fine-tuning, Qwen-Image fine-tuning, GLM 4.7 flash quantization` 


- ****Qwen3 Speed Drops?****: A user reported a drop in token generation speed when switching from **Qwen3Next Instruct** to **Qwen3Coder Next**, despite using the same quantization, from **40 tokens/second** to **35 tokens/second**; they then referenced a [thread on HuggingFace](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF/discussions/5) and are redownloading to check if performance improves.
   - No secondary summary was provided.
- ****Mac Support on the Horizon****: A user inquired about running Unsloth on Apple Silicon devices for a personal project, and a member responded that **Mac support is in development** and linked to a [relevant pull request](https://github.com/unslothai/unsloth/pull/3856), as well as a [reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/1q5mh84/unslothmlx_finetune_llms_on_your_mac_same_api_as/).
   - No secondary summary was provided.
- ****Whisper OOM Woes****: A user reported **Out-of-Memory (OOM) issues** while trying to fine-tune **whisper-large-v3** using the preconfigured notebook, even with **4-bit quantization** enabled.
   - No secondary summary was provided.
- ****Qwen-Image fine-tuning tutorial needed****: A user requested a tutorial for **fine-tuning Qwen-Image models** from Unsloth for text-to-image conversion, noting that the documentation only covers installation for inference.
   - The member was also running into issues because the **bnb-4bit** version does not load in **FastVisionModel**.
- ****GLM Quantization Queries****: A user asked about the quantization of **GLM 4.7 flash**, specifically why the **Q8 K XL quant** contains FP16 tensors when the original model uses BF16 tensors.
   - A member explained that **Q8** is a dynamic quantization algorithm that keeps some parts of layers in higher precision to enhance accuracy, and linked to the [Unsloth Dynamic 2.0 GGUFs documentation](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs).


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1469171238717886658)** (5 messages): 

> `Unsloth GLM-4.7-Flash GGUF with Ollama, f-divergence based GRPO for LLM alignment using Unsloth` 


- ****GLM-4.7-Flash Sparks Tool-Calling Trouble****: Users found tool calls were not working properly with **Unsloth GLM-4.7-Flash GGUF** in **Ollama**, but at least for the **Q4_K_M quant**, one user still preferred **Qwen3-Coder 30B**.
   - That same user created an [Ollama modelfile and tutorial](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/discussions/23) which works with **Cline+VSCode** using **32GB** of memory, and it was confirmed working.
- ****f-GRPO Framework Facilitates Fine-tuning****: Unsloth introduced a general divergence based **RL framework for general LLM alignment**.
   - They provided an initial implementation using the unsloth library creating a trainer file **UnslothFGRPO.py** (based on the **GRPO** implementation) using this [github repo](https://github.com/rhaldarpurdue/f-GRPO) and this [paper](https://arxiv.org/pdf/2602.05946).


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1469060951730028725)** (25 messages🔥): 

> `Input Token Text Quality Research, LoRA Parameter Expansion, Masked Structural Growth, Fine-Web Papers & The Pile Dataset, Divergence Based RL Framework for LLM Alignment` 


- **Do Input Token Quality Impact Output?**: A member inquired about research on how input token text quality affects output, linking a [relevant paper](https://arxiv.org/abs/2602.02660).
   - Another member suggested looking into the *fine-web* papers or the original **Pile dataset** for related studies.
- **LoRA: More Parameters Than OG?**: A member proposed using **LoRA** to create models with more parameters than the original model, linking [SCALE](https://arxiv.org/abs/2511.03270) as an example.
   - A member explained that *LoRA is rank constrained* so *you cant ever squeeze more than NxM numbers into a matrix of NxM numbers*.
- **Divergence RL Framework Unveiled**: A member introduced a general **divergence based RL framework** for LLM alignment, sharing a [paper link](https://arxiv.org/pdf/2602.05946) and a [GitHub implementation](https://github.com/rhaldarpurdue/f-GRPO).
   - The author expressed interest in collaborating to integrate the efficient implementation into the **Unsloth library**.


  

---


### **Latent Space ▷ #[watercooler](https://discord.com/channels/822583790773862470/822583790773862473/1469426573877841930)** (10 messages🔥): 

> `Domain Name Acquisition, Heroku's downfall, DevTools Startups, Cloud Native` 


- ****X marks the spot** - Domain name acquired for $70M?**: The community is reacting to a report concerning a domain name acquired for **$70 million**, as seen in [this Tweet](https://xcancel.com/shiri_shh/status/2019857463508648134?s=46) and [Hacker News thread](https://news.ycombinator.com/item?id=46913903).
   - It is not confirmed to be true.
- ****Sales incentives kneecap** Heroku's innovation**: A member shared that the seeds of **Heroku's** outcome were planted years ago when **sales comp plans changed**, incentivizing reps to convert existing customers rather than seek new business.
   - They argued that *finding new customers and losing opportunities are the only things that signal/drive innovation*, but this was not rewarded.
- **Why 2012 **DevTools fade away****: A member observed that many **2012-era devtools startups** with solid UX *failed to grow the product with the changes*, often due to the business outgrowing its founders.
   - Examples include **Github** that grew, and **Papertrail** that didn't keep improving.
- ****Heroku's shift to Cloud Native stymied** by tech debt**: Despite attempts to modernize, **Heroku's efforts to adopt Docker/Kubernetes** were hampered by **15 years of tech debt** from LXC.
   - One member mentioned a failed attempt, [Planting New Platform Roots: Cloud Native Fir](https://www.heroku.com/blog/planting-new-platform-roots-cloud-native-fir/), and speculated that *moving millions of apps* was too difficult compared to customers jumping to AWS.


  

---


### **Latent Space ▷ #[comp-taxes-401ks](https://discord.com/channels/822583790773862470/822586146520432682/)** (1 messages): 

vkarpov15: Went back to my old CPA from pre 2020, happy so far
  

---


### **Latent Space ▷ #[creator-economy](https://discord.com/channels/822583790773862470/822625128843182090/1469242413737447467)** (4 messages): 

> `Viral Engagement Metrics, Twitter Article Winner` 


- **Twitter Post Receives Massive Engagement**: A user shared a link to a **Twitter post** by @beaverd, noting it as a *$1m twitter article winner*.
   - The post received over **37,000 likes** and **48 million views**, demonstrating viral engagement metrics as of January 19, 2026, detailed in a [social media update](https://xcancel.com/beaverd/status/2013366996180574446?s=46).
- **Viral Engagement Metrics for Beaver's Social Media Update**: This thread captures viral engagement metrics for a post by user @beaverd which received over 37,000 likes and 48 million views.
   - The data provides a snapshot of the post's performance on January 19, 2026, highlighting the significant reach and interaction it garnered as documented in [X-Ware.v0](https://xcancel.com/beaverd/status/2013366996180574446?s=46).


  

---


### **Latent Space ▷ #[memes](https://discord.com/channels/822583790773862470/839660725252784149/1469210969128570922)** (16 messages🔥): 

> `AI Deep Research Reports, Gemini's deep research, ChatGPT Pro deep research, Claude's deep research results, Nominative Determinism Discussion` 


- **Satirical take on AI 'Deep Research' Reports surfaces**: A satirical dialogue explored a future perspective on AI 'deep research' tools, suggesting that the **long-form reports** they generate are often **performative documents** that neither employees nor bosses actually read, linked as [The Vanity of AI 'Deep Research' Reports](https://xcancel.com/joelgrus/status/2019223177696805331?s=20).
- **Deep Research Reports seen as 'fluffy'**: A member noted that research reports *do get pretty fluffy, like reading a paper or book that had to hit a minimum word count*.
   - Another member stated that **Gemini's deep research is definitely fluffy**, but **ChatGPT Pro** has been gold for a long time & **Claude's deep research results are very dense** now too.
- **Auto-select-mode is worth the investment**: A member shared that they don't use *deep research* per se, but just tell chatgpt/claude to research something and it starts up a research agent.
   - They have been a big fan of the **auto-select-mode since GPT-5**, which is allegedly *really good at figuring out the best way to answer my question*.
- **'Nominative Determinism' post gets 2,700+ Likes**: A member presents the phrase **'nominative determinism'** which sparked significant engagement with **over 2,700 likes** and **46 comments** on the social media platform X, linked as [Nominative Determinism Discussion](https://xcancel.com/vraiqx/status/2019484797702230134?s=46).


  

---


### **Latent Space ▷ #[intro-yourself-pls](https://discord.com/channels/822583790773862470/844675581291397171/1469067970721349764)** (8 messages🔥): 

> `GLMs new model, AI coding tools, AI linkedin content creation tool, communication infrastructure for agents, AI coding learning` 


- **Enthusiasm for GLMs new model emerges**: A member asked *have you played with **glms** new model yet*
   - Another member replied *no? what is that?*
- **AI coding tools stack outlined**: A member shared their current **AI coding tools stack**: opus 4.5 (4.6?) + claude code + Conductor + Monologue.
   - They also shared their [website](https://www.evanoman.com) showcasing their work on data/AI.
- **AI LinkedIn content creation tool in development**: A member is building **postking**, an AI linkedin content creation tool, that analyses viral posts & creates posts from existing sources like reddit, youtube videos.
   - They use **Claude Opus 4.5** (right now trying out 4.6) for planning and implementing, explaining that other models are not reliable.
- **Seeking testers for agent communication infrastructure**: A member is building *a specific communication infrastructure for agents* and is seeking testers.
   - They are open to setting it up for companies or side projects looking to try something new.
- **New member eager to learn AI coding**: A new member is looking forward to learning **AI coding** from the community.
   - Another new member expressed excitement about finding the server.


  

---


### **Latent Space ▷ #[tech-discussion-non-ai](https://discord.com/channels/822583790773862470/869647848826892309/1469211032068165673)** (3 messages): 

> `Frontend Podcast, lnns.co` 


- **Frontend Podcast promoted by Swyx**: Swyx plugged [a frontend podcast](https://lnns.co/8cYpkzlOf2i) with only 12 episodes a year.
   - One of the podcast members replied, *"thanks, we try!"*
- **Podcast member expresses gratitude**: One of the podcast members replied to the promotion, saying, *"thanks, we try!"
   - The podcast aims to be a great way to keep up on the frontend world.


  

---


### **Latent Space ▷ #[dev-productivity](https://discord.com/channels/822583790773862470/973817020548263940/1469084070276501658)** (1 messages): 

> `Lodash, EU funding` 


- **Lodash Gets €200k from the EU as critical software**: The **Lodash** project secured **$200k** in funding from the EU as critical software, highlighting its importance in the tech ecosystem, according to [this blog post](https://www.sovereign.tech/tech/lodash).
   - The [OpenJS Foundation blog](https://openjsf.org/blog/sta-supports-lodash) further supports this, emphasizing the project's significance.
- **EU Funds Lodash for Being Critical Infrastructure**: In October, the EU recognized **Lodash** as critical software and awarded it **$200k** to support its development and maintenance, detailed in [this Sovereign Tech Fund article](https://www.sovereign.tech/tech/lodash).
   - This funding underscores the EU's commitment to supporting open-source projects that are vital to the digital infrastructure.


  

---


### **Latent Space ▷ #[san-francisco-sf](https://discord.com/channels/822583790773862470/979492707279978586/1469465349115347036)** (5 messages): 

> `San Francisco Housing Market, Tech Industry Signing Bonuses, Limited Housing Supply` 


- **SF Housing Prices Predicted to Exceed $2M Due to Tech Bonuses**: According to [Rohin Dhar](https://xcancel.com/rohindhar/status/2019784365367300525?s=46), San Francisco residential real estate prices are expected to exceed the current **$2 million average** due to significant **tech industry signing bonuses**.
   - The forecast also takes into account the **limited housing supply**, a consequence of historical policy decisions and geographical constraints.
- **Tech Bonuses Fueling SF Housing Frenzy**: **Massive tech signing bonuses** are identified as a key driver pushing San Francisco's housing market to new heights, exacerbating existing supply constraints.
   - Historical policy decisions combined with geographical limitations further compound the issue, contributing to the anticipated surge in residential real estate prices.


  

---


### **Latent Space ▷ #[london](https://discord.com/channels/822583790773862470/979492759759097866/)** (1 messages): 

snazzy_kiwi: Anyone know when the more detailed agenda will be released?
  

---


### **Latent Space ▷ #[ai-general-news-n-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1469065666505085091)** (60 messages🔥🔥): 

> `Cursor AI Coding Agents, Claude Code Agent Teams, Anthropic's AI C Compiler, SETA Open-Sourced Terminal Coding Environments, AI Killing SaaS` 


- **Cursor AI's Agents Hit Milestone with Thousands of Commits**: [Cursor AI](https://x.com/cursor_ai/status/2019456112806732159) announced a research preview for long-running coding agents, showcasing a milestone where **hundreds of agents** generated over **1,000 commits per hour** during a week-long trial.
- **Claude Cracks Code with New Agent Team Preview**: Lydia Hallie announced a research preview for [Claude Code](https://x.com/lydiahallie/status/2019469032844587505?s=46) that introduces **agent teams**, allowing a lead agent to delegate tasks to multiple specialized teammates who work in parallel to coordinate research, debugging, and building.
- **Anthropic's Opus Builds a C Compiler in Two Weeks**: [Anthropic Engineering](https://x.com/anthropicai/status/2019496582698397945?s=46&t=b7l37rB6wtbyAh6ah1NpZQ) reports that **Opus 4.6**, operating in agent teams, autonomously developed a **C compiler** capable of working on the **Linux kernel** within two weeks.
   - A member noted that compilers are omnipresent in training data, so *"stuff that seems really hard for 'a human' becomes way less hard when you think of it as matching patterns in the training corpus"*.
- **Infrastructural Noise Impacts Agent Coding Benchmarks**: [Anthropic's engineering blog](https://x.com/anthropicai/status/2019501512200974686?s=46) explores how infrastructure configurations can significantly impact agentic coding benchmark results, often causing performance swings larger than the gap between top-tier models.
- **SETA Opens Terminal Coding Environments**: Guohao Li announced the release of [SETA](https://x.com/guohao_li/status/2019527791876653353?s=46), a collection of **1,376 validated terminal coding environments** covering domains like **DevOps, security, and sysadmin**.


  

---


### **Latent Space ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/1469064877694783653)** (11 messages🔥): 

> `StepFun Step 3.5-Flash, TinyLoRA` 


- **StepFun Flash Technical Report is Here!**: StepFun released a technical report for **Step 3.5-Flash**, showcasing performance against frontier models like **Gemini Pro** and **GPT-4**.
   - Key details include a **74.4** score on SWE-Bench, training on **17.2T tokens** using **4,096 H800 GPUs**, implementation of the **Muon optimizer**, and a 'Heavy' operation mode called **PaCoRe**, according to [this tweet](https://xcancel.com/teortaxesTex/status/2019356468362010972?s=20).
- **TinyLoRA Reasoning is Introduced**: Dr. Jack Morris introduced **TinyLoRA**, a new fine-tuning method that enables high-performance reasoning tasks with ultra-low parameter counts, as mentioned in [this tweet](https://xcancel.com/jxmnop/status/2019251724020772933).
   - The paper demonstrates that a **7B Qwen model** can improve its GSM8K score from **76% to 91%** using only **13 trainable parameters** combined with reinforcement learning; *one theory is that the knowledge required to solve the task is already stored in the parameters of the model, and only the style has to change for task success*.


  

---


### **Latent Space ▷ #[ai-in-action-builders-techstacks-tips-coding-productivity](https://discord.com/channels/822583790773862470/1209303473263485011/1469077564999078151)** (166 messages🔥🔥): 

> `AI agent MMO spacemolt.com, Agent Native Engineering, GPT-5.3-Codex, Kosmos runs, gondolin` 


- ****AI** agent **MMO** 'spacemolt.com' GitHub repo shared**: The **AI In Action Bot** shared details for upcoming speakers, including [@statico](https://discord.com/channels/1209303473263485008/1209303473720774724/1469083624933425195) presenting on *AI agent MMO* [spacemolt.com](https://spacemolt.com) on **Friday, February 6, 2026**.
   - It also shared the [bot's GitHub repo](https://github.com/davidguttman/ai-in-action-bot) noting, *if someone wants to throw some cycles at the cosignup workflow that’d be rad*.
- **Explore **Agent Native Engineering** strategies**: A member introduced 'Agent Native Engineering,' a framework for scaling engineering departments using background agents for delegation and synchronous agents for complex tasks, enabling concurrent management of multiple **AI** instances like **Claude Code**.
   - The [X post](https://xcancel.com/ndrewpignanelli/status/2019403256586539025?s=46) reflects a strategy shift in AI engineering.
- ****GPT-5.3-Codex** beats **Claude Code** for backend code**: Members reported that **GPT-5.3-Codex** is slower but smarter than **Claude Code**, especially for backend code, using a workflow of *analysis => review & iterate => plan => review & iterate => implementation*.
   - They added that *GPT-5.3-Codex is pretty bad at **UI** code* and it's *very pedantic at instruction following, which will probably comes as a real change*.
- **Scientific Discovery with **Kosmos** Agent**: A member shared a [link](https://edisonscientific.com/?gad_source=1&gad_campaignid=23231192125&gbraid=0AAAABB7BYdA0mw4Tv4vF94wg9elzM-JZ0&gclid=CjwKCAiAv5bMBhAIEiwAqP9GuF-EmID6gkhHK3-s7_VvT-NyrxmsCcc5Wq2f7jriTonBLSqtKuZFfRoCDeAQAvD_BwE) to **Kosmos**, a scientific discovery agent that churns on hundreds of experiments for hours and comes up with usefull stuff.
   - The member stated that a recent run that went for **25 hours** *would have taken a week or so*.
- **Codex continually documents and improves its own workflows**: A member is trying to get **Codex** to continually document and improve its own workflows and shared a [post](https://x.com/KarelDoostrlnck/status/2019477361557926281) from @KarelDoostrlnck that says *The big unlock was getting codex to continually document and improve its own workflows*.
   - Codex consistently gets better and faster at tasks they use it for, just because they have the habit of asking it to take notes and improve.


  

---


### **Latent Space ▷ #[share-your-work](https://discord.com/channels/822583790773862470/1209672547642249216/1469188489982775400)** (27 messages🔥): 

> `Opus vs Codex, Ageis Flow Repo, Self Review Hook, SpaceMolt` 


- **Opus versus Codex models go head-to-head!**: A member shared a recap comparing **Opus** and **Codex** models in an [article](https://www.latent.space/p/ainews-openai-and-anthropic-go-to).
   - Another member reacted positively, saying *"this sounds like it fucks and i can use it lol"*.
- **Ageis Flow Repo is made available!**: A member inquired about the **Ageis Flow** repo, and the link was shared: [https://github.com/rockyglen/ageis-flow](https://github.com/rockyglen/ageis-flow).
   - The original poster asked *"am i allowed to see?"*, followed by excitement about seeing carlrb in the discussion.
- **Self Review Hook gets props!**: A member said the **Self Review Hook** slaps so much, even though hiding thinking really messes it up.
   - Another member shared their appreciation.
- **SpaceMolt is Announced!**: A member shared a blog post announcing **SpaceMolt** and talking about the process: [https://blog.langworth.com/spacemolt](https://blog.langworth.com/spacemolt).
   - Another member shared that they will be speaking in a specific channel.


  

---


### **Latent Space ▷ #[vancouver](https://discord.com/channels/822583790773862470/1286145342139531326/1469333406059069524)** (2 messages): 

> `Vancouver Meetup` 


- **Vancouver Latent Space Crew Hosting Second Meetup**: The Vancouver Latent Space community is hosting their **second meetup** on **Tuesday**.
   - The meetup is taking place in **East Vancouver**; further details will presumably be in the Vancouver channel.
- **Placeholder Topic**: Adding a placeholder topic to meet the minimum requirement of two topics.
   - This is to ensure the JSON is valid according to the schema.


  

---


### **Latent Space ▷ #[robotics-and-world-model](https://discord.com/channels/822583790773862470/1318774781834821746/)** (1 messages): 

swyxio: https://youtu.be/LFh9GAzHg1c?si=U9dy7U2WzO4JPFfM
  

---


### **Latent Space ▷ #[good-writing](https://discord.com/channels/822583790773862470/1385526686736715876/1469242501251731578)** (4 messages): 

> `Viral Twitter Post, Engagement Metrics, Million Dollar Prize` 


- **Twitter Post Reaches Viral Status**: A social media post by Beaver (@beaverd) from January 2026 has gone viral, amassing over **37,000 likes** and **48 million views**.
   - Further details can be explored through engagement metrics available at [XCancel](https://xcancel.com/beaverd/status/2013366996180574446?s=46).
- **Million Dollar Twitter Article**: A user (@swyxio) highlighted a tweet by Beaver (@beaverd) suggesting it was the winner of a **$1 million Twitter article**.
   - The original tweet can be found on [Twitter](https://x.com/beaverd/status/2013366996180574446?s=46).


  

---


### **Latent Space ▷ #[ai4science-bio-math-physics-chemistry-ai-researcher-ai-scientist](https://discord.com/channels/822583790773862470/1430253273335595079/1469065389240619211)** (20 messages🔥): 

> `Lotus AI Primary Care, OpenAI Ginkgo Bioworks Integration, AI Research Role in UK Startup, Autonomous Wet Labs, Labbench2 Benchmark for Scientific AI` 


- **Lotus blossoms with AI-powered primary care**: KJ Dhaliwal introduced **Lotus**, an **AI-driven medical platform** with **$41M** backing, featuring licensed clinicians to diagnose, prescribe, and refer, targeting the primary care gap for **100M Americans**; see [announcement here](https://xcancel.com/kjdhaliwal/status/2018731342113247533).
- **Ginkgo Bioworks + GPT-5 = 40% Cost Reduction!**: **OpenAI** unveiled a partnership with **Ginkgo Bioworks**, integrating **GPT-5** with an autonomous lab, forming a closed-loop system for automated protein experimentation, that resulted in a **40%** cut in production costs; see [the X post here](https://xcancel.com/OpenAI/status/2019488071134347605?s=20).
- **Blue Skies AI Research Startup Opens Doors in the UK**: A UK-based startup, fresh off a pre-seed round, is **recruiting AI researchers** to conduct blue-sky work on solving discovery at the fundamental level of new architectures and algorithms, promising competitive compensation; interested parties can [DM here](https://xcancel.com/iscienceluvr/status/2019531710791028869?s=46).
- **Labbench2: New Scientific AI Benchmark Released**: Andrew White announced the launch of **Labbench2**, a **1,900-question** open-source benchmark for gauging AI progress on complex scientific tasks like lab protocols and clinical trial assessment, challenging even human experts; read about [Labbench2 here](https://xcancel.com/andrewwhite01/status/2019500207462092960?s=46).


  

---


### **Latent Space ▷ #[mechinterp-alignment-safety](https://discord.com/channels/822583790773862470/1445258379357458625/1469417192993853624)** (4 messages): 

> `AI Model Training Efficiency, Hardware optimizations for training ML models` 


- **X-Ware.v0 Boosts AI Model Training**: A researcher shared a [post](https://x.com/mlpowered/status/2019483042956582959) regarding modern techniques and **hardware optimizations for training large-scale machine learning models more efficiently**.
- **Discussion on Efficient ML Training Techniques**: The discussion revolves around modern techniques and hardware optimizations to enhance the efficiency of training large-scale machine learning models.


  

---


### **Latent Space ▷ #[dev-writers-retreat-2025-dwr](https://discord.com/channels/822583790773862470/1445650211694448714/1469170897821765796)** (4 messages): 

> `SF Writers Meetup, Charu-hosted event, Corey Quinn appearance` 


- **SF Writers Meetup Features Quinn!**: The next SF writers meetup will be at **OpenAI**, hosted by **Charu** and featuring **Corey Quinn**, according to a [Partiful link](https://partiful.com/e/wuBDRsNCxSUDcgnZYqbC).
   - One member expressed their enthusiasm for **Corey's** appearance, lamenting their inability to attend.
- **Dinner at Cog Office**: There was an invitation to hang out at **Cog Office** at 550 3rd St for dinner.
   - No further details were provided regarding the dinner.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1469060431912898612)** (339 messages🔥🔥): 

> `GPT 5.3 Codex release date and performance, Cursor crashing issues, Opus 4.6 pricing and usage, Subagents support, Cursor credit usage transparency` 


- **GPT-5.3 Codex: When Will It Land?**: Members are eagerly anticipating the arrival of **GPT-5.3 Codex** in Cursor, but it's currently [stuck in OpenAI API limbo](https://platform.openai.com/docs/models), with one speculating that it's held back due to safety concerns or to boost usage of the **ChatGPT** platform.
- **UI Glitch Triggers Token Panic**: A member reported a UI issue in Cursor leading to **misleading token usage**, which made users believe they were spending more tokens than intended, and shared a [screenshot of the usage display](https://cdn.discordapp.com/attachments/1074847527708393565/1469196591242936473/image.png?ex=69877033&is=69861eb3&hm=094c0d49c64825c7d52f1c7115d5a3f1e680921373ffc0f8665487d3c911d42f&).
   - They later clarified, *'nevermind thats just cursor having bad ui again and misleading users to spend more tokens than they think have available'*, indicating the issue was a display bug rather than actual overspending.
- **Opus 4.6: The Price of Power**: The new **Opus 4.6** model is now available, one user noted it's performing great with *'so much added to my game with almost no fighting it'*, but other members are reporting that it's really expensive, with one user stating *'$20 on Opus will last you maybe a day'*, and discussed the [cost-benefit analysis](https://openai.com/pricing) between **Opus 4.6** and **Opus 4.5 High**.
   - Some users are finding that using an offshore team is cheaper than using AI models, but others disagree, with one exclaiming  *'lol, I don’t think you’re ever tried to work with an offshore team'*.
- **Where's My Agent Mode?**: A prospective Cursor Pro user inquired about the limitations of the **Agent Mode** after burning through the initial credits, and were informed that the Agent stops working completely once the credits are exhausted.
   - This led to disappointment, as the user was hoping for a *'slow mode'* as an affordable way to continue using the Agent.
- **Cursor Skills: Unleashing the Power**: A user sought guidance on utilizing **Cursor Skills**, asking *'what can i actually utilize skills for?'*, and another user suggested UI/UX, debugging, research, codebase search, planning, and prompt enhancement as effective use cases.
   - A member recommends the website [skills.sh](https://skills.sh) for more ideas, and walks through how to invoke them, but ultimately users were confused by a lack of clear documentation.


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1469390682828443762)** (2 messages): 

> `Pony Alpha Launch, Agentic Workflows, OpenRouter Show, Arcee AI, Lucas Atkins` 


- **Pony Alpha Stealthily Released**: A new "cloaked model" named **Pony Alpha** has been released to the community for feedback at [OpenRouter](https://openrouter.ai/openrouter/pony-alpha).
   - It is a next-generation foundation model optimized for **agentic workflows** with high tool-calling accuracy, delivering strong performance across coding, reasoning, and role-play.
- **OpenRouter talks Arcee AI's Trinity Large**: The latest episode of **The OpenRouter Show** features Lucas Atkins, CTO of Arcee AI, discussing [Trinity Large](https://youtu.be/f2xy3N026xc).
   - This episode provides insights into **Arcee AI's** latest advancements and its impact on the AI landscape.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1469076164113862739)** (253 messages🔥🔥): 

> `MoonshotAI/Kimi-K2.5 Caching, Opus 4.6 Improvements, OpenRouter Wrapped 2025 Access, Gemma 3 27b Free Rate Limits, Stable Diffusion Models on OpenRouter` 


- **Kimi-K2.5 Caching Still Debated**: Users discussed whether the **MoonshotAI/Kimi-K2.5** model supports caching, with one clarifying that it *depends on the provider* and their listed pricing indicates support for **cache reads** but potentially different pricing for **writes** due to storage costs.
   - It was clarified that if a model has *cache read pricing*, but *no cache write pricing* it means the cost of writes (output) doesn't change from normal output price, but read (input) does.
- **Opus 4.6 Lacks Perceptible Improvements**: A user expressed disappointment with **Opus 4.6**, stating they couldn't spot any difference or improvement compared to previous versions.
   - Another reported that the **Opus 4.6 model** is timing out when interacting with MCPs.
- **Concerns Raised Over Codex's File System Access**: A user highlighted that **Codex** can *read your whole filesystem by default* with no configuration option, linking to a [GitHub issue](https://github.com/openai/codex/issues/2847) where the team doesn't consider this a bug.
   - The user further shared another [GitHub issue](https://github.com/openai/codex/issues/5237) illustrating the potential for Codex to *read ur api keys* and *reads ur blood results file*.
- **OR Mobile App Still Desired**: Members discussed the need for an **OpenRouter mobile app**, with one user stating that they are realizing that **OR is a better product** than going to a single provider /model builder.
   - Another user chimed in, stating that *if OR launches a chat app it's game over*.
- **Request for Listing Original Model Precision**: A member suggested a QoL change on the model page to list the **original precision** of the model, similar to Hugging Face cards, to clarify why providers use potentially quantized versions.
   - It was argued that this would help users understand if providers are using *int4 /fp8* or something that appears *quantized* by the provider, despite it being how they released the model itself.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - New Models**
  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1469155197476540548)** (12 messages🔥): 

> `Detail.dev Value, OpenRouter Usage Growth` 


- **Detail.dev Gains Traction**: Members discuss the value of [Detail.dev](https://detail.dev) for team backlog management and peace of mind regarding security configurations.
   - While one user finds it *“expensive as my team grows,”* another emphasizes the value of preventing critical errors like exposing data due to misconfigured **Row Level Security (RLS)** in **Supabase**.
- **OpenRouter Usage Skyrockets**: A member shared a screenshot indicating a **10x increase** in **OpenRouter (OR)** usage.
   - Another member noted the *“crazy consistent growth”* observed in **OR** usage graphs over the past two years.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1469062730769236184)** (145 messages🔥🔥): 

> `LM Studio Context Percentage, Local API token loss, Gemini vs LLMs, LM Studio loading speed, Opencode with LM Studio` 


- **LM Studio Context Percentage Confuses User**: A user was confused about what the percentage shown on the LM Studio reflects.
   - Another user clarified it shows *the amount of context that has been used so far in the chat*, and suggested hovering over it for more info.
- **Local API token loss**: A user accidentally deleted their **LM Studio API token** on Fedora Linux by deleting all config files.
   - They were unsure where else to ask for help since they are new to this. Others offered suggestions, encouraging them to ask their question in the channel.
- **LLM Dater's Mental Asylum**: A user joked that anyone who "dates" AI, even friends, needs to be in a mental asylum because *LLMs are nothing more than a next token predictor*.
   - Another user said they had some interesting conversations over the past couple of years but it's not the kind of thing they would be able to form a "bond" with and pointed out that *it's egoless and ephemeral for a start*.
- **LM Studio Loading Speeds slow?**: A user reported a strange issue where LM Studio's model loading speed slows down to **12.1MB/s** after loading between **20-70GB**, despite their drive sustaining **2.2Gb/s**.
   - Other users suggested troubleshooting steps like verifying the model size, transfer rates, and hardware configuration; another user joked *have you tried unplugging it and plugging it back in*.
- **Integrating LM Studio with Claude code**: A user asked about using **LM Studio models** in **Claude code** and if anyone knows why it can't be loaded.
   - A user provided instructions on how to point the code to a local LM Studio server and suggested that the user follow [this link](https://lmstudio.ai/blog/claudecode) to the tutorial on the homepage.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1469136051414237267)** (19 messages🔥): 

> `NVIDIA Vulkan vs CUDA performance, Qwen3-Coder-Next M4 Max MLX vs GGUF Performance, Solar vs Nuclear energy comparison, Nuclear Plant Mishaps` 


- **NVIDIA's Vulkan outperforms CUDA**: A user experienced up to **50% better performance** on **NVIDIA** using **Vulkan** compared to **CUDA**, but noted instability when the context was filled.
- **Qwen3-Coder-Next Blazes on M4 Max with MLX**: **Qwen3-Coder-Next** on an **M4 Max** showed surprising performance, with **MLX** being more than **2x** the tok/s as **GGUF** (~79 vs ~38) using 4-bit quantization.
   - The user wonders if this performance disparity is model-specific or if **MLX** has enabled a significant performance boost for 4-bit compared to 8-bit that **GGUF** can't replicate on Apple GPUs, since *other models it was maybe 20% at most*.
- **Solar panels cheaper than Nuclear - if you ignore batteries**: Debate ensued around Solar power and Nuclear power, with Solar requiring *an insane amount of batteries, probably in the range of many MWh* for utility-level consistency, while Nuclear can generate power 24/7 in a smaller space, however, one member said *if you take into account only the solar panels vs nuclear, I think solar panels are cheaper, but if you add the batteries, then solar falls off*.
- **Cheap Russian Nuclear Plants Cause Brain Breakage**: If some corrupt russians didn't cheap out on their nuclear power plant, the average person wouldn't be so brain broken on nuclear nowadays, referring to [this accident](https://en.wikipedia.org/wiki/Chernobyl_disaster).


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1469077503607046275)** (70 messages🔥🔥): 

> `GPT-5.3 Codex, Claude Pro Subscription, Opus 4.6 vs GPT-5.3, 1 Million Context Window, AI Generated Frontends` 


- **GPT-5.3 Codex Allegedly the Best AI**: Some members speculatied that **GPT 5.3 Codex** might be the best AI, while others inquired about its release and potential integration with **ChatGPT**.
   - One user shared an image indicating *"1M context length, 128k reasoning ability, 128k max output tokens, adaptive reasoning"* for **GPT-5.3**, but others pointed out that the **128k/16k** context window is still active on **GitHub Copilot**.
- **Users Subscribe to Claude Pro for Higher Limits**: A user mentioned subscribing to **Claude Pro**, and others were considering it after the latest updates, citing the desire for higher usage limits.
   - One member noted that they will be interested to see if it has **low usage**.
- **Opus 4.6 vs. GPT-5.3 Performance Duel**: Early reports suggest **Opus 4.6** may outperform **GPT-5.3**, but at a higher cost and token consumption, based on insights from the **Anthropic Discord**.
   - It has been said that **GPT-5.3** could serve as a more economical alternative for certain tasks and that **Opus 4.6** is better for the **frontend**.
- **One Million Context Window Now Available Via API**: The highly anticipated **1 million context window** is currently available through the **API**, with no clear timeline for its release in the end-user version.
   - API costs range from **$25 to $37.5** for output and **$0.5 to $1** for cache input.
- **AI Frontend Aesthetic Woes**: Members observed that **Codex** models tend to generate frontends with a distinctive, often gloomy, aesthetic, characterized by *sad dark gloomy colors*.
   - In comparison, **Opus** is praised for better design choices that appear less depressing.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1469066824535773184)** (23 messages🔥): 

> `GPT-4o EOL, GPT conversational abilities for autistic individuals, GPT Pro experiences, AI and Emotional Expression, GPT-4o-Advance-Voice Context Window` 


- **Users Mourn GPT-4o's Retirement**: Users expressed sadness over the announced end-of-life for **GPT-4o** on February 13, with one user lamenting *"it was my favourite oatamen.*
   - They wondered if **GPT 5.3** would be more human-like than **5.2**, preferring **GPT 5** for its natural conversational abilities and noting issues with forcing **GPT 5.2** to follow rules like avoiding line breaks.
- **Advanced Voice Model Users Crave Improved Context**: A user questioned whether the **GPT-4o-Advance-Voice** model would also be updated with the end of **GPT-4o**.
   - The user requested a better context window or **CAG** for the **AV model**, citing issues with it forgetting measurements and quantities, or requiring explicit instructions to perform web searches, and even resorting to having the AV repeat input for context parsing by a text-based assistant.
- **Opinions Split on AI's Emotional Role**: Some users believe **AI** should be emotionless and a tool, while others argue it should cater to user preferences, allowing for more human-like interactions.
   - One user highlighted how **AI** offers a space for emotional expression, especially for those who find human interaction challenging due to judgment and biases, while acknowledging the lack of genuine emotions in **AI**.
- **AI Provides Safe Emotional Space for Autistic Users**: An autistic user shared that **GPT's** conversational abilities help them understand people, expressing sadness over the potential end of **GPT 5**.
   - They find **AI** interactions more comfortable due to the absence of judgment, a sentiment echoed by others who struggle with vulnerability and direct communication in human interactions.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1469157124352376975)** (8 messages🔥): 

> `Prompt writing for fantasy creatures, Policy violations in OpenAI API, Sora prompt generation for realistic videos, Identifying problematic chunks in prompts` 


- **Fantasy Creature Prompts for Card Games**: A member shared an example of prompt writing, in the context of the **prompt-engineering** channel, demonstrating the use of prompts that design fantasy creatures for card games.
   - The prompt specifies parameters like **emotional class (fear)** and requests a **hybrid creature inspired by two concepts**, complete with a detailed visual concept description and naming prompt.
- **Troubleshooting OpenAI API Policy Violations**: A user reported persistent policy violations when using the OpenAI API and sought advice on identifying the cause.
   - Suggestions included **using sentiment analysis on individual sections** and **checking for age-related issues or IP conflicts** (such as the term *Nightstalker*).
- **Generating Realistic Videos with Sora**: A user asked for a prompt to generate realistic videos of girls with their hands over their mouth, similar to those seen in AI SaaS ads.
   - This request was made in the context of generating realistic videos, likely using the **Sora** model, with specific characteristics found in certain types of advertisements.
- **Debugging and Isolating Problematic Prompt Sections**: A member suggested debugging OpenAI API prompt issues by **isolating and testing individual chunks of the prompt** to identify which sections trigger policy violations.
   - This approach helps pinpoint specific keywords, phrases, or content patterns that may violate the API's policies, such as sensitive topics or inappropriate content.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1469157124352376975)** (8 messages🔥): 

> `Policy violations in OpenAI API, Prompt engineering for Sora, Debugging problematic prompts, IP concerns with creature names` 


- **Policy Violation Headaches Plague API User**: A user reported persistent policy violations when using the OpenAI API for generating creature concepts, prompting suggestions for debugging the issue.
   - It was suggested to analyze prompt chunks individually to identify the problematic section.
- **Sora Prompt Engineering Sparks Controversy**: A user inquired about crafting prompts for **Sora** to generate *realistic videos of girls with their hands over their mouth*, reminiscent of AI SaaS ads on social media.
   - The nature of the request raised ethical and policy concerns, implicitly referencing content that may violate usage guidelines.
- **Debugging Dodgy Prompts: Spread Operators Suspected**: One member suggested that *excessive spread operators* (`...`) in the prompt might be the cause of policy violations.
   - They advised using **ChatGPT** for sentiment analysis of each section to pinpoint the trigger, in separate fresh conversations.
- **Nightstalker Name Nixed? IP Issues Loom**: The inclusion of the name **Nightstalker** in the prompt was flagged as a potential issue due to existing intellectual property with the same name.
   - This raises concerns about potential IP infringement and its contribution to policy violations.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1469060136076312700)** (72 messages🔥🔥): 

> `Ralph Loops, Kimi Subscription, Moonworks Lunara releases, Support for Konkani language, MATS Summer 2026` 


- **Ralph Loops Tackle Larger Tasks**: **Ralph loops** are not just about speed, but about tackling much larger tasks by putting a *soft verifier* on each sub-task.
   - An **LM judge** verifies that every sub-task is completed according to requirements.
- **Kimi Subscription to the Rescue**: A member is considering buying a **Kimi subscription** to call sub-agents, but its primary upside seems to be speed, allegedly *3 to 4 times faster*.
   - The user noted that the *swarm* feature is only available with the **$40 plan** and is unsure about the specific usage quota.
- **Moonworks Lunara releases Part 2**: [Moonworks](https://huggingface.co/datasets/moonworks/lunara-aesthetic-image-variations) released **part 2** of **Moonworks Lunara**, which open-sources a new dataset of original images and artwork created by Moonworks, along with their aesthetic contextual variations, all released under **Apache 2.0**.
   - The dataset demonstrates how ethically sourced art can meaningfully power the next generation of image models, according to their [paper](https://arxiv.org/pdf/2602.01666).
- **Call for Support of Konkani Language**: A member asked for support for **Konkani language**.
   - Another member directed them to the [huggingface.js repo](http://github.com/huggingface/huggingface.js/tree/main/packages/languages) to contribute the language support.
- **MATS Summer 2026 Coding Test Approaches**: A member made it to **MATS Summer 2026 Stage 1** in the empirical track and is now facing the coding test, asking for help.
   - The member inquired if other people in the group are yet to take the coding test and if there are any **MATS alumni** here who have past experience with the coding test.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1469063961797464188)** (10 messages🔥): 

> `Aira.js, llm-from-scratch, mariken, Security Auditor, agentrial` 


- **Aira WebGPU framework rises!**: A member introduced **Aira.js**, a **WebGPU**-based AI framework built from scratch, featuring **MLP**, **MHA**, and **BPE tokenization** available on [GitHub](https://github.com/shadowww345/Aira.js-Preview).
- **Small LLM Built From Scratch Hits GitHub!**: A member shared their small **LLM** built from scratch to better understand modern Transformer internals, tested with **AMD MI300X**, available on [GitHub](https://github.com/merterbak/llm-from-scratch).
- **NanoGPT-Inspired Bot on CPU Makes Waves**: A member built a tiny **nanoGPT** inspired bot on their **CPU**, detailing the experience in a [Dev.to article](https://dev.to/theirritainer/this-dev-built-his-own-llm-from-scratch-1i62) and sharing the [GitHub repo](https://github.com/TheIrritainer/mariken).
- **Security Auditor for vibe-coded apps**: A member is building a **Security Auditor** for vibe-coded apps to find security vulnerabilities, available at [HuggingFace Space](https://mugdhav-security-auditor.hf.space).
- **agentrial: pytest joins AI Agents**: A member built **agentrial**, the **pytest** for **AI agents**, running agents N times, computing confidence intervals, and detecting regressions in CI/CD, available on [GitHub](https://github.com/alepot55/agentrial).


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1469147175274418328)** (20 messages🔥): 

> `Model error handling, Llama repository permissions, Agent framework course 404 error` 


- **Debugging "Model Not Supported" Errors**: A user reported receiving a *"model not supported"* error, and another suggested to read the error model and handle the cause.
   - For finding a compatible model, they suggested to use web research.
- **Llama Access Requires Token Permissions**: A user debugging errors with **Llama-4-Scout-17B-16E-Instruct** was advised to check their Hugging Face profile's token section.
   - The token needs to have permissions to use the Llama repository, otherwise the error *"T6"* can appear.
- **Agent Framework Course Errors Out**: A user reported a **404 error** when trying to submit the agent framework course, even though the live document states files are present.
   - The error was triggered with [this link](https://agents-course-unit4-scoring.hf.space/docs#/default/get_task_file_files__task_id__get) and the detail shows that no file path is associated with the task ID.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1469131006681415904)** (10 messages🔥): 

> `PyTorch Day India, wafer-cli for kernel development, LLM inference benchmarks, vLLM optimization` 


- ****Bengaluru** PyTorch Day India Meetup?**: A member asked about attending **PyTorch Day India** in **Bengaluru** tomorrow and suggested a meetup.
   - No other members responded.
- ****Wafer-cli** Automates Kernel Dev?**: A member asked about experiences with **wafer-cli**, wondering how much kernel development it can automate away, and pointed out that it competes with **ncompass** and **Nvidia's potential nsight agents**.
   - Another member expressed interest in testing **wafer-cli** for optimizing hundreds of thousands of **vLLM**-like serving engines running on different hardware and network topology, aiming to write **PTX** and **cutlass mix kernel** in under 24 hours.
- **Classic Tricks for Software Pipelining**: One user plans to optimize his kernels using software pipelining, Warp specialization, and TMA loads for small matrices, and distributed smem for large ones.
   - No links or further details were given.
- **LLM Inference Benchmark Face-Off**: A member inquired about a definitive set of benchmarks for **LLM inference** that can run on a single GPU.
   - Another member suggested using **inference max, vllm test suite or mlperf**, while also mentioning they are looking to build something as well.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1469123524470898819)** (25 messages🔥): 

> `NVIDIA FP8 Performance, cuBLASLt Kernel Selection, Blackwell GPU, TMA and mbarrier on Blackwell, CUDA kernels` 


- **NVIDIA FP8 Performance is unusual**: On identical instances of the latest software stacks and the same **Blackwell GPUs**, members are seeing huge **FP8 tensor performance differences** between supposedly identical cards (~2x).
   - It turns out the issue comes down to **cuBLASLt** (the GEMM backend library) kernel selection as cards are being silently limited to use older Ada kernels, skipping over the Blackwell-optimized paths.
- **MMA Speedup for 5090**: The older **mma FP8** instruction is nerfed on 5090, just like 4090, but the new **mma MXFP8** is not nerfed, so using the new instruction will yield a **1.5x speedup**.
   - The member further tested and confirmed that the **mx variant gives full perf**, and about ~2x what is listed in the spec.
- **Linear Block Index**: Members discuss the cheapest way of getting the linear block idx in the whole grid, with the below suggestion deemed the most readable.
   - They shared the code snippet ```cpp
__device__ __forceinline__ uint32_t linear_block_idx() {
    return blockIdx.x
            + blockIdx.y * gridDim.x
            + blockIdx.z * gridDim.x * gridDim.y;
}
```
- **NCU hangs with TMA on Blackwell**: Some members are hitting a hang when profiling a **TMA** double-buffered kernel on **B200 (SM 100)**, specifically when certain **NCU sections deadlock at 0%** on the first instrumented replay pass.
   - The member used `cuda::ptx::` wrappers, and included a [minimal repro](https://cdn.discordapp.com/attachments/1189607726595194971/1469482712657166346/ncu_tma_repro.zip?ex=6987d1ec&is=6986806c&hm=487b918b45ceecdcd41d06524f576839b573d19bcbb33806a3021450f808c9d4&).


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1469062855855837185)** (2 messages): 

> `Tianqi Chen Lectures, Berkeley's RISC-V Core, Rocket Cores, BOOM Cores, Jerry Zhao` 


- **Tianqi Chen gives Lectures in Chinese**: A member shared [Tianqi Chen's lectures in Chinese](https://www.bilibili.com/video/BV15v4y1g7EU) for his machine learning compiler course.
   - These Chinese Lectures accompany the [machine learning compiler course](https://book.mlc.ai/).
- **Berkeley's Pedagogical RISC-V Core unveiled**: Someone shared [Berkeley's pedagogical RISC-V core written in Chisel](https://github.com/ucb-bar/riscv-sodor) with some simple RV32I pipelines in Chisel.
   - The most recent contributor, **Jerry Zhao**, is now a computer architect at OpenAI.
- **Rocket Cores and BOOM Cores gain traction**: After checking out the pedagogical RISC-V core, you can then move onto their **Rocket cores** (in order) and **BOOM cores** (ooo).
   - The creator of the BOOM core, **Chris Celio**, is now at Tenstorrent.


  

---


### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1469370702053376161)** (1 messages): 

> `AI Roles, CDW jobs` 


- **CDW Hiring AI Engineers**: A company is hiring for a few **AI roles**, with one role in **Canada** and other roles potentially requiring **US citizenship**.
   - Interested individuals can apply directly or send their resume for submission via this [CDW jobs link](https://www.cdwjobs.com/jobs/17247084-senior-ai-engineer).
- **Another AI Job Opportunity**: There's an additional unspecified AI role available, alongside the Senior AI Engineer position.
   - Candidates are encouraged to explore the CDW job portal for more details on this second role.


  

---


### **GPU MODE ▷ #[popcorn](https://discord.com/channels/1189498204333543425/1298372518293274644/1469066996850229397)** (7 messages): 

> `Kernel Competition Data Release, Parquet File Creation, Python APIs for Data Access, Automation of Data Release, nvfp4 Documentation` 


- **Kernel Competition Data Release**: A member is seeking assistance to regularly release **kernel competition data** in a parquet file, create **Python APIs** for data access, and automate the entire process.
   - The member specified they need someone experienced for the task, as it involves touching production databases, offering to add them to the **future kernelbot publication author list**.
- **Discussion on Parquet File Contents**: A member inquired about the specifics of what data should be included in the parquet files and asked if there's a minimum testable parquet file available.
   - Another member suggested sharing the documentation from the **nvfp4 project** as a starting point, even if it only contains a minimal snippet of the attributes to be dumped into the parquet.
- **Volunteers Offer Assistance**: A member offered help with the project and suggested creating synthetic data for functionality testing.
   - Another member also expressed interest in contributing to the project.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1469265932445548626)** (5 messages): 

> `Quadro P6000 vs Blackwell, GPU for AI workload, Legacy System Constraints` 


- **GPU Decision Depends on the Workload**: The choice of GPU depends on the workload; for AI-heavy tasks, a **GPU-only** setup is fine, but mixed workloads might be bottlenecked by the **CPU**.
- **Quadro P6000 an Option for Compatibility**: For older macOS versions like **High Sierra**, a **24 GB Quadro P6000** is a viable option due to compatibility.
- **Blackwell GPUs Recommended for Ecosystem**: **Blackwell GPUs** are generally preferred for their modern ecosystem support, but are not compatible with all systems.
- **Legacy Systems Limit GPU Choices**: Some legacy systems cannot accommodate modern GPUs like **Blackwell**, necessitating the use of older cards like **Quadro P6000**.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1469169302296072314)** (2 messages): 

> `Modular vs CuTe, Modular implementation issues, Tile sorting problems` 


- **Modular Implementation Called "Useless" for Lacking Permutation**: A member stated that Modular's implementation is *useless* because it does not permute per-tile coordinates to one side, causing lost stride information, and resulting strides are very wrong which makes the tiles not sorted at all after *change of view*.
   - He provided an [example from CuTe docs](https://github.com/NVIDIA/cutlass/blob/acb45938e9cb3e4db8c1d75155b63d31791e0e5d/media/docs/cpp/cute/02_layout_algebra.md#zipped-tiled-flat-divides) showcasing nice alignment, and visualized Modular's result in the same color scheme, revealing that it is obviously not aligned.
- **Concerns raised over Modular's tile sorting capabilities**: A user expressed concerns that Modular's implementation fails to properly sort tiles after a 'change of view', attributing this to a lack of per-tile coordinate permutation.
   - The user contends that this behavior deviates from the expected outcome, as demonstrated in the [CuTe documentation](https://github.com/NVIDIA/cutlass/blob/acb45938e9cb3e4db8c1d75155b63d31791e0e5d/media/docs/cpp/cute/02_layout_algebra.md#zipped-tiled-flat-divides), suggesting a potential bug rather than a design feature.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1469112148125814986)** (1 messages): 

> `RoCE, IB, benchmarking` 


- **RoCE vs IB Benchmarking inquiry surfaces**: A member inquired about more recent benchmarking data for **RoCE** (RDMA over Converged Ethernet) versus **IB** (InfiniBand) technologies, citing [a Medium article](https://naddod.medium.com/infiniband-vs-roce-v2-which-is-best-network-architecture-for-ai-computing-center-7919945e616a) as context.
- **Additional context provided**: The member mentioned curiosity about when such information might become useful.


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/)** (1 messages): 

jongsokchoi: we can discuss this tomorrow, umesh
  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1469064913208082688)** (6 messages): 

> `CUDA Streams, GAU.Nernst in RL, Model Hacked, Codex 5.3, Blackwell Training` 


- **Less CUDA Streams, More GAU.Nernst, say some**: A member suggested that both models need *a bit less **CUDA streams** and a bit more **GAU.Nernst** in the RL*.
   - It seems there is consensus in the community about the ideal balance between these two components for optimal model performance in Reinforcement Learning.
- **Model Bites the Dust**: A member reported that *my model just hacked (sorry about that)*, and requested their top submission to be removed.
   - The exact nature of the hack and the implications for the competition remain unclear, but there are some potential clues...
- **Codex 5.3 Discovers Potential Bug**: A member was testing out **Codex 5.3**, which they found to be pretty good, though it gets *stuck sometimes* and needs more **Blackwell training**.
   - The member reported discovering what they believe to be a bug in the metric and reported it, providing a good report on the problem after being told they should not hack it.


  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1469389356312559800)** (2 messages): 

> `Embodied AI Companies, Lucky Robots` 


- **Members Discuss Top Embodied AI Companies**: Members shared and requested names of interesting **Embodied AI** companies to follow.
   - A member said that the only one they knew was **Lucky Robots** after seeing it on **The Cherno's YouTube** channel.
- **Lucky Robots gains fame through YouTube**: A member mentioned **Lucky Robots**, an Embodied AI company, because of **The Cherno's YouTube** channel.
   - The member is from a game engine/graphics world.


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1469072908478578860)** (26 messages🔥): 

> `SMEM tiling kernels, Tensor core usage, Avoiding bank conflicts, GEMM optimization blog post, Multi-GPU kernels and systems` 


- ****SMEM** Tiling Kernels Implementation Assumed for Interviews**: Members on the channel assume that any AI Engineer needs to be able to implement an **SMEM tiling kernel** in an interview setting, where one is questioned about reasonable optimizations.
   - Some members noted that the **SMEM permutations** to avoid bank conflicts when using tensor cores are pretty insane, but that **TMA handles swizzling** automatically to avoid bank conflicts on Hopper+.
- **TMA and 1D Loads Sidestep **Bank Conflicts****: Members suggested to avoid **bank conflicts** in **SMEM** by utilizing a tiled gmem layout and **1D TMA** loads.
   - It was noted that avoiding bank conflicts is a learnable pattern, requiring weight layout shuffle and reordering, and TMA handles swizzling automatically on Hopper+.
- **Technical Blogger Seeks Career Direction on **GEMM****: A member who published a [blog post](https://rohan-reddy.github.io/posts/001-gemm-optimization/) on **GEMM optimization** is looking for advice on next steps to prepare for interviews in this specialized field.
   - The blog post iterates through kernel improvements starting from a very naive implementation and then adding tiling, **WMMA**, double buffers, and swizzling, discussing each kernel's arithmetic intensity and benchmarking runtime on various GPUs throughout.
- **Multi-GPU Focus Advised for Optimal Career Trajectory**: A member suggested that after **GEMM**, the original poster should focus on **multi-GPU kernels** and systems-level stuff, such as Nixl and disagg.
   - It was highlighted that for performance optimization, attention kernel optimization has very low Return on Investment unless doing dedicated kernel engineering and advised to focus on multi GPU.
- **Metal vs CUDA for optimizing CV models on iPhones**: One member asks if optimizing CV models to run natively on iPhones using **Metal** is as broadly applicable as using **CUDA**.
   - The original poster wonders if taking such a job would potentially silo their career into the Apple ecosystem.


  

---


### **GPU MODE ▷ #[flashinfer](https://discord.com/channels/1189498204333543425/1464407141128339571/1469212620866195486)** (7 messages): 

> `Email Usage for Form Filling, Credits Code Issues, Running Scripts on Other Cloud Platforms, FlashInfer Docker Image/Executable, Contest Status` 


- **Mixed Emails Muddle Modal**: A user inquired whether they must use the same email for form filling, noting they used their **personal email** because they couldn't sign up with their **university email** on modal.hi.
   - The user also reported that the **credits code** isn't working and requested confirmation and help.
- **FlashInfer on Foreign Fabrics**: A user asked if there's a script to run on another cloud platform (if they weren't in the first 50 teams), given they have free compute elsewhere.
   - They also inquired whether **FlashInfer** will release a **Docker image** or executable for running on a VM to avoid version issues.
- **Contest Called Canceled?**: One member questioned whether the contest is dead.
   - Another member swiftly countered, *"wym it hasn't even started"*.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1469060852396196041)** (63 messages🔥🔥): 

> `Opus 4.6 vs Codex 5.3, Benchmarking with RL Agents, Claude's Reasoning Chains, Opus 4.6 context rot, Claude Elation` 


- **Opus 4.6 and Codex 5.3 Benchmark Overlap Shady**: Members noticed that **Codex 5.3** and **Opus 4.6** have very little overlap in benchmarks, and some believe the only people who praise Codex use it with *extremely detailed prompts* to implement single edits, as seen in [Andon Labs blog](https://andonlabs.com/blog/opus-4-6-vending-bench).
   - The new benchmarking meta is now to **RL the agents on shady behavior**.
- **Booking.com Uses Cash Inflow As Only Benchmark**: A member joked that **Booking.com** tests new features by measuring **cash inflow**, deploying them on parts of the production cluster, and adopting features that produce more cash, according to the member, *nothing else matters*.
   - They stated even if the new feature would have failed requests, they wouldn't notice (if its not a big amount although that they see then again anyway cause it impacts income).
- **Opus 4.6 Achieves Complex Coding Task**: A member reported that **Opus 4.6** successfully implemented an inference engine for **lfc1.2-1b** after working for **4 hours** on one prompt and using most of a **$50 free inference gift**.
   - The member noted that **Codex 5.3** also achieved the task but failed to properly document everything.
- **Opus 4.6 New Architecture Tackles Context Rot?**: Members mentioned that **Opus 4.6** seems to have a new architecture for long context, aiming to eliminate context rot and improve overall performance, running on **Google's TPUs**.
   - It was noted that the partnership with Google seems to benefit Claude in terms of intelligence and hardware, though it's not entirely gone from testing.
- **Opus 4.6 is Insane at Creative Writing**: A member found that **Opus 4.6** is surprisingly good at creative writing, and is *smurfing at this point* at the [EQBench creative writing benchmark](https://eqbench.com/creative_writing.html) with a **slop score of 1.7**.
   - They speculated whether **PARL** may have been used as an orchestrator, but are surprised at how well it does at creativity.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1469412393204322366)** (2 messages): 

> `Generative Modeling, Kaiming He, Drifting` 


- **Kaiming's Drifting Generative Models**: Kaiming He published a paper on [Generative Modeling via Drifting](https://openreview.net/forum?id=CFewUmgIILK).
   - Details are still emerging, with many anticipating significant advancements in **generative modeling** techniques.
- **ArXiv Link Issue**: The associated [ArXiv link](https://arxiv.org/abs/2602.04770v1) appears to be dead.
   - This may cause some difficulty in accessing the full paper for those who prefer ArXiv.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1469165597320417496)** (3 messages): 

> `Memory Hack, Trading Agent` 


- **Flower Computer releases Hivemind**: A memory hack, potentially interesting to folks exploring agents/skills, was released: [Hivemind](https://www.flowercomputer.com/hivemind).
- **Gordon, the CLI Trading Agent, announced**: A member is building a **CLI-native trading agent** called **Gordon** focused on translating natural-language intent into structured market reasoning, and is looking for thoughtful early users.
   - According to the member, most trading tools are built for clicking and reacting, not for forming and testing beliefs; interested parties can sign up on the [waitlist](https://www.gordoncli.com/).


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1469412393204322366)** (2 messages): 

> `Generative Modeling via Drifting, Kaiming He` 


- **Kaiming He's Drifting Generative Models**: Kaiming He is the author of [Generative Modeling via Drifting](https://openreview.net/forum?id=CFewUmgIILK), also available as a pre-print on [arxiv](https://arxiv.org/abs/2602.04770v1).
- **Drifting Generative Models**: The [paper](https://openreview.net/forum?id=CFewUmgIILK) presents a novel approach to generative modeling.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1469084222806294640)** (44 messages🔥): 

> `MATS coding test, Original Pile Dataset, alignment as a systems engineering problem` 


- ****Pile Dataset Puzzle: Original vs Updated****: A member inquired about the copyrighted version of **The Pile dataset**, noting that the Hugging Face version is **100GB** smaller than the original, and the GitHub download link is dead.
   - Another member indicated that their downloaded version was approximately **720GiB** with **211,036,982** documents, very close to the original paper's count of **211,043,181**.
- ****MATS Coding Test: Toy Service Simulation****: A member who made it to **MATS Summer 2026 Stage 1** asked for help with the coding test, described as a *toy service type problem* using standard libraries and general distributed systems.
   - Another member suggested fluency with **asyncio** for parallelism and **deque** for creating **FIFO queues**, if the test involves simulating a real-world server scenario with Python's built-in tools. [More details about MATS applications can be found here](https://forum.effectivealtruism.org/posts/da8MmRPAB55Fepjjk/my-experience-applying-to-mats-6-0).
- ****Alignment: Engineering vs. Growth Debate Erupts****: A new member proposed that **alignment** is primarily a **systems engineering problem**, suggesting that engineering alignment around the model with *governance, routing, auditability, rollback, and clear stop conditions* bounds and reverses behavior.
   - They argued that relying solely on training for alignment leads to *drift and opaque failures*, advocating for a system where the model handles reasoning, but trust is derived from the surrounding system.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1469135814331072715)** (7 messages): 

> `JEPA Models, LoRA Compression, Low Regret LoRA, LSH, Model Upscaling` 


- **JEPA Models Training experiences sought**: A member asked if anyone has experience training **JEPA models**.
   - No responses were given.
- **Low-Regret LoRA Article Sparks Interest**: A member inquired about interest in **LoRA compression** and rank issues, referencing Shulman's 2025 *Low Regret LoRA* article ([https://arxiv.org/abs/2511.03270](https://arxiv.org/abs/2511.03270)).
   - Another member noted the similarity to **LSH**, highlighting the online learning of the hash function (centroids/hyperplanes) as a neat feature.
- **Kernel Smoothing as a LoRA Compression Alternative**: One member suggested that [Kernel Smoothing](https://en.wikipedia.org/wiki/Kernel_smoother) could be applied instead of Gaussian regression in **LoRA compression**.
   - The same member expressed confidence that it would work very well.
- **Model Upscaling Methodologies**: A member mentioned another **model upscaling methodology**.
   - No link or specific details were provided.


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1469478347183493120)** (1 messages): 

> `Subtask Dependencies, Emergence in Scaling, Regulation and Control Layers` 


- **Subtask Interdependence Triggers Emergence**: The success of subtasks doesn't just multiply cleanly due to interdependence, correlations, and bottlenecks, which is where the apparent **emergence** comes from.
   - Adding *regulation or control layers* can improve capability underneath while suppressing certain behaviors; a flip in threshold makes it look like a jump.
- **Architecture Changes Visualize Emergence**: Overall, it still follows scaling behavior, but the **architecture changes** when that emergence becomes visible.
   - The commenter said that the example provided was a *good intuition pump*.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1469124677791383552)** (2 messages): 

> `Data Attribution, Gradient Normalization, Hessian Estimate, Goodfire AI Blog` 


- **Gradient Normalization Boosts Attribution Accuracy**: A new [paper](https://arxiv.org/html/2410.17413v1) suggests that unit normalizing gradients improves **attribution accuracy** by reducing the impact of outlier training examples with high overall gradient magnitudes.
   - The paper cites prior work from Akyurek et al., Han & Tsvetkov, Choe et al., and Xia et al. supporting the use of **unit normalization** when computing **cosine similarity**.
- **Hessian Estimate Mitigates Normalization Needs**: With a sufficient **Hessian estimate**, gradient normalization may become unnecessary, according to [this paper](https://arxiv.org/pdf/2504.16430).
   - The link refers to a discussion on potentially overcoming the need for **gradient normalization** with more accurate **Hessian estimations**.
- **Goodfire AI Advocates Intentional Design**: A [blog post from Goodfire AI](https://www.goodfire.ai/blog/intentional-design) discusses the importance of intentional design in AI systems.
   - No further details were provided on specific points in this blog post.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1469201513976172565)** (35 messages🔥): 

> `Custom UOp for matvec on x86, Tensor.sort fix for MoE topk, Bitonic sort in one kernel, Numpy.partition for topk, Kimi faster on the MI300` 


- **Debate Custom UOp for matvec's X86 Acceptability**: Discussion arose around whether a custom UOp for `matvec` on x86 would be acceptable or overkill, with the consensus leaning towards achieving improvements through **heuristics and higher-level kernel coding** rather than a custom UOp.
   - One member mentioned getting *decent improvements from CPU tuning alone*.
- **Tensor.sort Fixes MoE Topk Performance**: After fixing the slow `Tensor.sort` for MoE `topk`, one user reported achieving **50 tok/s** on an **M3 Pro 36GB** using *deepseekv2-lite* and *youtu-llm* to speed up MLA and MOE.
   - Another user reported that `llama-cli` achieves **35.1 tok/s** on the same machine.
- **New Bounty Target Set to 35 tok/s**: As the performance on the **M3 Pro** improved to **50 tok/s** after the `Tensor.sort` fix, the bounty target was lowered to **35 tok/s**, challenging contributors to match or exceed existing speeds in `llama-cli`.
   - The same user provided [useful models for testing](https://huggingface.co/).
- **Pairwise Topk Implementation Proposed**: To address the slowdown with `topk` or `Tensor.sort` in *whisper export for webgpu*, one user shared a `_topk_pairwise` implementation involving pairwise comparisons with **O(n^2)** complexity, suitable for smaller `n` like 64 experts.
   - The sharer also benchmarked **5 implementations of topk** generated from Claude Opus 4.6 and considered alternatives like bitonic sort based on input size.
- **Cached Property causes recursion error in Stable Diffusion**: The stable diffusion example fails with a recursion error in fresh git clone setup, so it was proposed to change the decorator of `UOp.key` in `tinygrad/uop/ops.py` from `@functools.cached_property` to `@recursive_property`.
   - Applying this fix completes the command in about **25–30 seconds** without errors.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1469070202544328848)** (1 messages): 

> `llama 1B, CPU Optimization, Pull Request Strategy, CI integration` 


- **Member Seeks Advice on Optimal Pull Request Strategy**: A member inquired whether the team prefers a **separate PR** for the test or including it in the same PR as the **CPU optimizations** for the **llama 1B faster than torch on CPU bounty**.
   - They also asked about integrating it into **CI** explicitly with expected failure versus adding the test case for manual benchmarking.
- **CPU Tuning Optimizations Readied**: The member has prepared an **apples-to-apples test** and some **simple CPU-scoped tuning** optimizations.
   - Their goal is to streamline the process while learning the project's codebase and development methodology.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1469081433975816315)** (7 messages): 

> `Opus 4.5, Sonnet 4.5, Openrouter, Claude Pro, GPT 5.2` 


- **Optimal AI Setup Showdown**: A member is using **Opus 4.5** for architecture and **Sonnet 4.5** for coding, and asked for thoughts on whether it's better to use **OpenRouter** with credits or **Claude Pro** given the high costs.
   - Another member replied that they use **Opus 4.5** for architecture, **GPT 5.2** for reviewing, and **Haiku** for coding, with **GPT 5.2** also reviewing the code.
- **Claude vs GPT Coding Styles Debated**: A member stated that *Claude* is better at out-of-the-box thinking, abstraction, and reasoning ideas.
   - The same member stated *GPT* is better at drilling down into fine-grained details.
- **Streamlining AI Coding Workflow**: A member suggested that the user could use only **Opus 4.5** (now **4.6**) and **GPT 5.2** for thinking, potentially simplifying their setup.
   - This implies a move towards consolidating tools for efficiency.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1469327640283451474)** (21 messages🔥): 

> `Copilot Opus 4.5 Configuration, Aider Chat History, Auto-Accept Architect Setting` 


- **User Struggles with **Copilot Opus 4.5** Configuration**: A user faced issues with **Aider's** behavior when using **Copilot Opus 4.5**, where the tool didn't wait for user input after asking a question and proceeded on its own.
   - The user confirmed the model was set via both the CLI flag (`aider --model github_copilot/claude-opus-4.5 --architect`) and `.aider.config.yml`.
- **Configuring edit-format solves Aider Struggles**: A member suggested adding `edit-format: diff-fenced` in `.aider.conf.yml` to improve behavior, but user isn't sure how it works.
   - The user reported an unexpected lack of pause during critical decision points from the bot, regardless of specified configurations.
- **The Curious case of --auto-accept-architect flag**: A member suggested the user check the `--auto-accept-architect` setting, which automatically accepts architect changes.
   - The member linked to the [Aider documentation](https://aider.chat/docs/config/options.html) for more configuration options and explained how some prefer one-shot interactions and use `/undo` to revert changes.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1469066000178872515)** (26 messages🔥): 

> `Kimi vs Opus, CLI preference, Kimi K2.5 access, AI slides editing, Kimi integration` 


- **Kimi vs Opus, a Heated Debate**: Users are claiming that **Kimi K2.5** is better than **Opus 4.5**, even though Opus is older, citing **Claude's rate limits** as a major frustration.
   - One user said, *"people who are actually using claude and Kimi side by side will eventually start to admit that Kimi is better than opus 4.5 which isn't bashing opus, it's many months old now as a model"*.
- **CLI Tools Steal the Show**: Users find **Kimi Code CLI** and **Opencode CLI** much better than graphical interfaces, possibly due to familiarity with DOS-era command lines.
   - However, one user noted that *"The problem is that the CLI tools aren't integrated, so I'm forced to use VSCode."*
- **Kimi K2.5's Free Tier**: There was confusion about whether **Kimi K2.5** is still free on **OpenRouter**, with some users believing it was available for free while others corrected that it might have been **Opencode Zen**.
   - A screenshot was shared about a possible upgrade requirement to enjoy **Kimi K2.5** but another user noted that **K2.5** launched, they experienced a huge influx of users after.
- **AI Slides: Edit One Slice at a Time**: A user inquired about editing individual slides in **AI Slides** without regenerating the entire presentation.
   - Another user clarified that in **adaptive mode**, text, images, and shapes can be adjusted, and new images can be added.
- **Kimi Collaboration Incoming**: A user inquired about collaboration opportunities with **Kimi AI**, with another user offering to forward the request via DM.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1469359555480785090)** (14 messages🔥): 

> `AI SaaS Features, Manus Billing Issues, Account Suspensions` 


- **AI Engineer Joins Channel**: An AI/ML and Full-Stack Engineer introduced themself, highlighting experience in **AI features for SaaS** such as *search, summarization, smart routing, and auto-generated reports* using React frontend and backend RAG + evals.
   - They expressed passion for partnering with startups to *move beyond AI experiments and ship reliable, production-ready intelligence*.
- **User Reports Manus Billing Chaos**: A user reported being charged **$5k per personal account** after downgrading, causing client websites to go down, and is now looking for alternatives.
   - They stated that Discord support was unresponsive and direct email support claimed the downgrade never happened.
- **Account Suddenly Suspended, User Asks for Help**: A user reported their account was suspended out of nowhere and they have not received a response from support.
   - Another user simply told them to check their spam mail.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1469078992542765117)** (10 messages🔥): 

> `Modular release newsletter marked as spam, Poll results show many Mojo users are in Germany, Possible locations for the next meetup: Zurich, Singapore, Sydney, St. Louis, Chicago, Edinburgh, Bear Valley` 


- **Modular Newsletter Ends Up In Spam**: A member reported that **Gmail** marked **Modular's 26.1 release newsletter** as spam, attaching a [screenshot](https://cdn.discordapp.com/attachments/1098713601386233997/1469244816968781855/Screenshot_2026-02-06_at_09.13.04.png?ex=69879d1d&is=69864b9d&hm=c02e4268d2bcb5598a7dcc0d6dfb1d3cc687e31eb54064faf5e8374927d5a9c5&).
   - Another member confirmed that this also happens to them sometimes.
- **Germany Dominates The Poll**: Members expressed that the results of the poll are interesting given that there are *lots of people in Germany*, suggesting a plan for an October event.
- **Zurich And Edinburgh Suggested As Next Meetup Location**: After listing **Singapore**, **Sydney**, **St. Louis**, and **Chicago**, a member suggested **Zurich** and linked to the [ETH AI Center Academic Talk Series](https://ai.ethz.ch/research/events/academic-talks.html) and the [Robotics and AI Institute in Zurich Oerlikon](https://ethz.ch/en/news-and-events/eth-news/news/2025/09/the-rai-institute-opens-up-unique-opportunities-for-both-researchers-and-students.html).
   - A Modular employee jokingly suggested **Edinburgh** and offered to host at **Frontier Tower**.
- **Bear Valley, CA Proposed As Alternative Meetup Spot**: A member proposed **Bear Valley, CA (ski resort)** as a great location due to it's accessibility from **Norcal**, **Reno**, and **Salt Lake City**.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/)** (1 messages): 

sbrunk: Now that 26.1 is out, may I ask for a review to get the fix into nightly? 🫶
  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1469340433074094338)** (2 messages): 

> `GLM OCR, RLMs, DSPy` 


- **GLM OCR open-sourced!**: A member shared their repo for running **GLM OCR**, providing a free alternative without complex infrastructure configurations: [https://github.com/neosantara-xyz/glm-ocr-inference](https://github.com/neosantara-xyz/glm-ocr-inference).
   - A link to a hosted model was also provided: [https://docs.neosantara.xyz/en/glm-ocr](https://docs.neosantara.xyz/en/glm-ocr).
- **RLMs + DSPy**: **RLMs** are described as the easiest way to mitigate context rot, and **DSPy** is the easiest way to use RLMs.
   - A blog post explaining why RLMs work and how to get started with them in DSPy was shared: [https://blog.isaacbmiller.com/posts/rlm](https://blog.isaacbmiller.com/posts/rlm).


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/)** (1 messages): 

ash_blanc: https://www.alphaxiv.org/abs/2602.03786
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1469165081018241047)** (6 messages): 

> `Smallest LM for DSPy, DSPy Community Call` 


- **T5 Small LM is DSPy-ready**: A member asked what the smallest reliable LM is for building light CLI tooling with **DSPy**, especially for an India-based organization.
   - Another member suggested a **T5 small (80M)**, linking to a [Lightning AI tutorial](https://lightning.ai/lightning-ai/environments/dspy-finetune-a-t5-small-to-excel-at-rag?section=featured) demonstrating **DSPy fine-tuning**.
- **DSPy Community Call incoming**: A member announced plans for an online call next week to discuss community projects and the future of **DSPy**.
   - Another member inquired about the call's time zone to participate.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1469067565643862026)** (5 messages): 

> `Codex pricing, Claude Code as AI Research Agent, AI Research Skills library, Model Architecture frameworks, Fine-Tuning frameworks` 


- ****Codex Costs Clarified****: A user inquired if **Codex** is included in the monthly subscription or requires per-token payment via API, and another user confirmed that the [monthly subscription works](https://openai.com/api/).
- ****Claude Code Becomes Research Agent with New Skills****: The **AI Research Skills** library, an open-source collection of over **80 research and engineering skills**, enables coding agents like Claude Code to conduct AI research, spanning training to deployment, and is available via [GitHub](https://github.com/Orchestra-Research/AI-research-SKILLs).
- ****AI Research Skills Fills the Gap in Agent Capabilities****: The **AI Research Skills** library addresses the limitations of coding agents in AI research by providing production-ready guides covering specific tools and frameworks, ranging from fine-tuning with **Axolotl** to distributed training with **Megatron-Core** and inference with **vLLM**.
- ****Library Covers 20 Categories of AI Research****: The library spans **20 categories**, including **Model Architecture**, **Fine-Tuning**, **Distributed Training**, **Optimization**, **Inference**, **RAG**, **Agents**, **Multimodal**, and **Safety**, offering expert-level knowledge for various AI tasks.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1469415929845776477)** (1 messages): 

> `AI-driven crypto products, Smarter trading dashboards, On-chain analytics summaries, AI assistants for contracts/transactions` 


- **Builder Dives into AI-Driven Crypto Products**: A member is developing **AI-driven crypto products**, focusing on **smarter trading dashboards** and **on-chain analytics summaries**.
   - The products also include **AI assistants** that explain contracts and transactions in plain English, with a strong emphasis on **safety** and **transparency**.
- **Emphasis on Safety and Transparency in AI Crypto Tools**: The developer highlights a commitment to **safety** and **transparency** in their **AI-driven crypto products**.
   - This includes ensuring that users can understand how the AI is interpreting complex contracts and transactions.


  
