---
id: MjAyNS0x
title: not much happened today
date: '2025-12-22T05:44:39.731046Z'
description: >-
  **Zhipu AI's GLM-4.7** release marks a significant improvement in **coding,
  complex reasoning, and tool use**, quickly gaining ecosystem adoption via
  Hugging Face and OpenRouter. **Xiaomi's MiMo-V2-Flash** is highlighted as a
  practical, cost-efficient mixture-of-experts model optimized for deployment.
  The open-weight text-to-image competition sees **Z-Image Turbo** leading with
  6B parameters under Apache-2.0 license. Video model advances focus on control
  and long-form consistency, exemplified by **Kling 2.6 Motion Control** and
  research like MemFlow's adaptive memory retrieval. In agent frameworks,
  **Google's A2UI protocol** introduces agent-driven UI generation, while
  studies reveal that mixing multiple agent frameworks is common, with
  challenges in logic, termination, and tool interaction. LangChain emphasizes
  persistent memory patterns for production agents.
companies:
  - zhipu-ai
  - xiaomi
  - google
  - langchain
  - huggingface
  - openrouter
  - artificial-analysis
  - vllm-project
models:
  - glm-4.7
  - mimo-v2-flash
  - z-image-turbo
  - kling-2.6-motion-control
topics:
  - coding
  - complex-reasoning
  - tool-use
  - mixture-of-experts
  - cost-efficiency
  - open-weight-models
  - text-to-image
  - video-models
  - memory-persistence
  - agent-frameworks
  - interactive-user-interfaces
  - model-deployment
people:
  - mervenoyann
  - eliebakouch
  - omarsar0
  - osanseviero
  - dair_ai
---


**good job, China AI**

> AI News for 12/22/2025-12/23/2025. We checked 12 subreddits, 544 Twitters and 24 Discords (208 channels, and 4321 messages) for you. Estimated reading time saved (at 200wpm): 305 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Z.ai's [GLM 4.7](https://z.ai/blog/glm-4.7) and Baidu's ERNIE 5.0 nearly made the cut, but the former is incremental and the latter is unreleased.

What ARE released, though are three new AIE CODE talks from frontier agent labs:

- [Factory AI](https://www.youtube.com/watch?v=ShuJ_CN6zr4)
- [Amp Code](https://www.youtube.com/watch?v=gvIAkmZUEZY)
- [Repit Agent](https://www.youtube.com/watch?v=MLhAA9yguwM)

Enjoy.

---

# AI Twitter Recap

**Open-Weight Model Drops: GLM‑4.7, MiMo‑V2‑Flash, and image/video model churn**

- **Zhipu AI’s GLM‑4.7 release + immediate ecosystem uptake**: Zhipu positions **GLM‑4.7** as a meaningful step over GLM‑4.6 with improvements in **coding, complex reasoning, and tool use** (weights on HF, plus a tech blog and hosted chat) in [@Zai_org](https://twitter.com/Zai_org/status/2003156119087382683). The release quickly shows up across distribution and eval surfaces: day‑0 availability via HF tooling is noted by [@mervenoyann](https://twitter.com/mervenoyann/status/2003162322181976553), OpenRouter listing in [@OpenRouterAI](https://twitter.com/OpenRouterAI/status/2003196169632243815), and LM Arena Code Arena movement in [@arena](https://twitter.com/arena/status/2003159444822327748) (claims **#1 open-model** spot and **#6** overall on a WebDev leaderboard; +83 pts vs GLM‑4.6). Several practitioners also flag that “interleaved thinking” behavior changed and suggest using the **official API for benchmarking** ([@eliebakouch](https://twitter.com/eliebakouch/status/2003163924716466287)).
- **Xiaomi MiMo‑V2‑Flash: “practical” MoE tuned for deployment**: A wave of commentary frames **MiMo‑V2‑Flash** as optimized for **cost/speed/deployability** rather than leaderboard aesthetics. [@omarsar0](https://twitter.com/omarsar0/status/2002768840556728714) highlights claims that it rivals strong open-weight peers with fewer parameters; repo pointer in [@omarsar0](https://twitter.com/omarsar0/status/2002768968713699747). A Zhihu-centric roundup emphasizes agent workflows and pricing shock (e.g., $0.1 / 1M input tokens cited) plus mixed views on stability ([@ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/2003135877606760816)). vLLM published an **official serving recipe** with concrete knobs for context length/latency/KV cache and DP/TP/EP configs in [@vllm_project](https://twitter.com/vllm_project/status/2002938138549682366).
- **Open-weight text-to-image competition tightens**: Artificial Analysis reports **Z‑Image Turbo** as new **#1 open-weights text-to-image** in its Image Arena, with **6B params**, **Apache‑2.0**, and pricing comparisons (e.g., $5/1k images on Alibaba Cloud) in [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/2002839525609865575).
- **Video model progress concentrates around control + long-form consistency**: Multiple threads orbit **Kling 2.6 Motion Control** (user demos, dance/action control, and fal day‑0 availability). See examples from [@fal](https://twitter.com/fal/status/2003103565309415665) and high-engagement user showcases like [@IamEmily2050](https://twitter.com/IamEmily2050/status/2002968479276937403). On the research side, MemFlow proposes adaptive memory retrieval for long streaming narratives ([@HuggingPapers](https://twitter.com/HuggingPapers/status/2002714237492138434)).

**Agents in production: protocols, orchestration, memory, sandboxes, and “observability-first” engineering**

- **Google’s A2UI protocol for agent-driven UIs**: A notable infra/API drop is **A2UI (Agent‑to‑User Interface)**, an open-source protocol enabling agents to generate interactive user interfaces ([@osanseviero](https://twitter.com/osanseviero/status/2002747011230269893)). The framing suggests a shift from agents as “chat-only” to agents as **UI generators** with a standard interface layer.
- **Agent framework reality check: mixing frameworks is the norm**: A large empirical study summarized by [@dair_ai](https://twitter.com/dair_ai/status/2003178236696776814) claims 1,575 agent projects show **96%** of top-starred projects combine multiple frameworks (e.g., LangChain+LlamaIndex; AutoGen+LangChain), while GitHub stars don’t predict adoption. Reported pain points concentrate in **logic failures**, **termination detection**, **agent-tool interaction**, and **version compatibility**.
- **Memory patterns & persistence become first-class**: LangChain highlights an Oracle-backed hub for agents with persistent storage and “six memory patterns” ([@LangChainAI](https://twitter.com/LangChainAI/status/2002771047234613550)), reflecting how production agents are increasingly constrained by **state, recall, and auditability** rather than raw model IQ.
- **Sandboxed/async agent execution gets operationalized**: There’s a clear pattern of pushing coding agents into **isolated execution environments** (enterprise-friendly sandboxes, reproducible “blueprints,” trace retention). See: Runloop + DeepAgents example via [@hwchase17](https://twitter.com/hwchase17/status/2002801655801385037) and an “async coding agents at home” writeup using Claude Code in a Modal sandbox ([@andersonbcdefg](https://twitter.com/andersonbcdefg/status/2002829629187608794)). Related: a “better git for agents” pitch (zagi) focused on context-efficient diffs, auditing, and trajectory forking ([@mattzcarey](https://twitter.com/mattzcarey/status/2002796068811976885)).
- **Observability as the missing discipline**: A representative “LLMOps for personal workflows” post argues many perceived model regressions are actually **instruction ambiguity, missing context, or poor decomposition**, which becomes obvious only after tracing with tools like LangSmith ([@ChaiWithJai](https://twitter.com/ChaiWithJai/status/2002895889690407382)). This aligns with repeated calls to treat AI engineering like backend engineering: instrument, log, evaluate—don’t debug “by vibes.”

**Benchmarks, eval politics, and what “progress” should measure (METR, Arena, FrontierMath, SWE-bench)**

- **METR-style evals and the “verification bottleneck” framing**: A recurring theme is that RL progress is limited by **verification time** rather than task length; proposed improvement is plotting capability vs **time needed to verify** ([@ShashwatGoel7](https://twitter.com/ShashwatGoel7/status/2002732250681766347)). Separate threads critique confusing reporting fields and argue aggregate “working_time” and total cost are not very informative without per-task breakdowns ([@scaling01](https://twitter.com/scaling01/status/2002793892773544154)).
- **Arena-driven model narratives continue to matter**: Beyond GLM‑4.7’s Code Arena jump ([@arena](https://twitter.com/arena/status/2003159444822327748)), Baidu’s **ERNIE‑5.0‑Preview‑1203** lands high on LM Arena’s text leaderboard with preliminary scoring claims ([@arena](https://twitter.com/arena/status/2003151045946376482)).
- **SWE‑bench “catch-up” signal for open models**: A concise snapshot claims open coding models are nearing closed performance on **SWE-bench verified** (GLM‑4.7 **73.8%**, Kimi K2 Thinking **73.4%**, DeepSeek‑V3.2 **73.1%**, Claude Sonnet 4.5 **77.2%**) and highlights GLM‑4.7’s math/tool-use strengths ([@cline](https://twitter.com/cline/status/2003181058679029915)).
- **FrontierMath access asymmetry becomes part of the discourse**: Epoch AI notes open-weight Chinese models lag top frontier performance on FrontierMath tiers by ~7 months, while also stating OpenAI has exclusive access to a substantial portion of Tier 1–3 data/solutions ([@EpochAIResearch](https://twitter.com/EpochAIResearch/status/2003178174310678644)).

**RL, distillation, and safety/security loops: what’s scaling now vs what’s missing**

- **“Year of RL” → “year of distillation” meme meets real signals**: Distillation appears both in product rumors and deployment narratives. A notable claim: **Gemini 3 Flash uses distillation pretraining** ([@yifan_zhang_](https://twitter.com/yifan_zhang_/status/2002745931649933724)). Separately, multiple commentators predict distillation becomes the next cycle driver (e.g., [@leithnyang](https://twitter.com/leithnyang/status/2002795896170541456)).
- **RL infra democratization**: **OpenTinker** is pitched as a decoupled client/server RL framework for LLMs—“configure backend once on a GPU cluster; define envs locally; train remotely”—aiming to reduce RL pipeline setup time by ~10× ([@youjiaxuan](https://twitter.com/youjiaxuan/status/2002838551319253281)).
- **Prompt-injection / agent security becomes operational RL**: OpenAI describes hardening its browser agent (ChatGPT Atlas) against prompt injection using **automated red teaming + reinforcement learning + rapid mitigation loops** ([@cryps1s](https://twitter.com/cryps1s/status/2003182649662140620)). This is a concrete example of RL as an ongoing **security maintenance loop**, not just a capability accelerator.
- **Research taste: algorithmic progress vs compute**: A widely shared quote attributed to Sergey Brin argues “compute is dessert; algorithms are the main course,” asserting algorithmic progress has outpaced scaling in the last decade ([@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2002803245354459588)). It’s a useful counterweight to pure scaling narratives—especially when combined with calls for better verification/eval design.

**Robotics & embodied AI: Reachy Mini momentum, RL transfer gaps, and video-action models**

- **Reachy Mini becomes a “holiday robotics platform”**: Multiple builders report fast setup and polished UX (manual + app + SDK) and plan local assistants; e.g., [@Prince_Canuma](https://twitter.com/Prince_Canuma/status/2002695729442402496) and [@chenosaurus](https://twitter.com/chenosaurus/status/2002826732525773212). The repo trends as well ([@PoratDor](https://twitter.com/PoratDor/status/2003027940078993798)).
- **Sim-to-real and even robot-to-robot transfer remains hard**: John Carmack describes experiments where flawless simulators transfer poorly to real camera/servo setups, and even transferring policies between theoretically identical 3D-printed rigs causes performance loss—recoverable via continual online learning ([@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/2002773760672223265)).
- **Robotics learning stack expands beyond “LLM controls robot”**: A “video-action model” class is introduced for robot learning (mimic-video) ([@elvisnavah](https://twitter.com/elvisnavah/status/2003088362119512560)), and Chelsea Finn shares a fine-tuning result on “Robot Olympics” tasks ([@chelseabfinn](https://twitter.com/chelseabfinn/status/2003165418098446339))—both pointing to tighter coupling between perception backbones and action policies.

**Culture-side signals that still matter for engineers: slop, “LLM psychosis,” and interface ergonomics**

- **“LLM psychosis” / delusion narratives spike—often around math proofs**: Several high-engagement posts warn that models are good enough to delude even experts ([@_lewtun](https://twitter.com/_lewtun/status/2002690691705794805)), and multiple threads dunk on claims of solving Millennium Problems, framing it as mania and “vibe-coded” nonsense ([@suchenzang](https://twitter.com/suchenzang/status/2002774256783077420); [@BlackHC](https://twitter.com/BlackHC/status/2003156071460843734)). For engineers, the important takeaway isn’t drama; it’s that **fast fluent outputs create a verification crisis** unless workflows force checks.
- **Ergonomics of coding agents is becoming a product wedge**: Repeated comparisons argue Claude Code’s UI affordances (plan mode, ask-to-edit, etc.) are materially better than Codex’s current interaction design ([@finbarrtimbers](https://twitter.com/finbarrtimbers/status/2002765191134732642)). This aligns with the broader “context engineering” reframing—moving from prompts to managing tools/memory/policies ([@TheTuringPost](https://twitter.com/TheTuringPost/status/2002765247900262620)).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Major Open-Source AI Model Releases 2023

- [**major open-source releases this year**](https://www.reddit.com/r/LocalLLaMA/comments/1pstlas/major_opensource_releases_this_year/) (Activity: 746): **The image highlights significant open-source releases in 2023, showcasing advancements in AI and machine learning. Notable releases include Deepseek's open reasoning model, Qwen's vision and image editing models, and Alibaba's photorealistic image generation model. Other key releases are Google's Gemma 3, Meta's SAM models, and Nvidia's Nemontron 3. The post underscores the closing gap between open and closed-source technologies, suggesting a shift in the landscape of AI development.** Commenters note the dominance of Chinese companies in the open-source space, with only three US companies listed. There is also anticipation for Deepseek's future releases, with expectations that they may surpass closed-source models in reasoning capabilities. Additionally, there is a discussion about Mistral's performance in smaller model sizes.
    - **Maximum** discusses the potential of the upcoming DeepSeek model, highlighting its training on '3.2 speciale'. The commenter anticipates that DeepSeek could surpass closed-source models, particularly in reasoning tasks, suggesting a significant advancement in open-source AI capabilities.
    - Sufficient-Bid3874 raises a question about Mistral's performance, specifically in the context of small-sized models. This suggests a discussion around Mistral's efficiency and effectiveness in resource-constrained environments, which is crucial for applications requiring lightweight models.
    - mukz_mckz emphasizes the importance of Olmo's contributions to the open-source community, particularly through their papers, blog posts, and codebase. The comment suggests that Olmo provides valuable insights into how various training parameters ('knobs') can impact model performance, offering a learning opportunity for developers and researchers.
- [**I made Soprano-80M: Stream ultra-realistic TTS in <15ms, up to 2000x realtime, and <1 GB VRAM, released under Apache 2.0!**](https://www.reddit.com/r/LocalLLaMA/comments/1pt3sco/i_made_soprano80m_stream_ultrarealistic_tts_in/) (Activity: 530): **Soprano-80M is a new TTS model developed by Eugene, achieving unprecedented speed and efficiency in text-to-speech conversion. It streams audio with less than** `15 ms` **latency and can generate a** `10-hour audiobook in under 20 seconds`**, reaching** `~2000x realtime` **performance. Key innovations include a higher** `32 kHz` **sample rate for clearer audio, a vocoder-based decoder for faster generation (**`~6000x realtime`**), and a novel neural audio codec compressing audio to** `~15 tokens/sec` **at** `0.2 kbps`**. The model is designed for ultra-fast, natural speech generation, though it currently lacks features like voice cloning and multilingual support. [GitHub](https://github.com/ekwek1/soprano), [Huggingface Demo](https://huggingface.co/spaces/ekwek/Soprano-TTS), [Model Weights](https://huggingface.co/ekwek/Soprano-80M).** Commenters noted the model's impressive speed but also mentioned issues with audio quality, such as slurred words and artifacts, especially in longer outputs. There is curiosity about the hardware used for achieving the claimed performance, as similar models show significantly lower real-time factors on high-end GPUs.
    - Chromix_ highlights performance issues with Soprano-80M, noting that while it is extremely fast, generating long audio files can result in slurred words, noise, repetition, and artifacts. The model's performance degrades after the one-minute mark, as demonstrated in a shared audio link. This suggests potential limitations in the model's ability to maintain quality over extended durations.
    - coder543 questions the hardware used to achieve the claimed 2000x realtime performance, comparing it to their experience with Kokoro-82M, a similarly sized model, which achieved only 50x to 100x realtime on an RTX 3090. This raises questions about the reproducibility of the performance claims under different hardware conditions.
    - geneing discusses the architecture of Soprano-80M, noting it uses a small Qwen3 LLM to generate vocos features, which is then decoded by vocos. They express skepticism about the model's practical accuracy due to its small LLM size, suggesting that models with LLMs smaller than 0.5B may suffer from quality issues, particularly in handling complex language tasks like English pronunciation.

### 2. GLM 4.7 Release and Features

- [**GLM 4.7 is out on HF!**](https://www.reddit.com/r/LocalLLaMA/comments/1pt5heq/glm_47_is_out_on_hf/) (Activity: 660): **GLM 4.7 has been released on [Hugging Face](https://huggingface.co/zai-org/GLM-4.7), showcasing improvements over GLM 4.6 in areas like multilingual coding, UI generation, and complex reasoning. It achieves** `73.8%` **on SWE-bench and** `42.8%` **on HLE, introducing features such as *Interleaved Thinking*, *Preserved Thinking*, and *Turn-level Thinking* for enhanced task management. The model is deployable locally using vLLM and SGLang, with integration instructions on GitHub. A notable feature is the use of diagrams in reasoning/planning, marking a first in this domain.** There is skepticism about the benchmarks, with some users suggesting GLM 4.7 is a faster, incremental improvement over DeepSeek 3.2, but not surpassing Sonnet 4.5. It may be comparable to Gemini 3 Flash in coding capabilities.
    - Dany0 expresses skepticism about the benchmarks provided for GLM 4.7, suggesting that while it may be a faster and better incremental improvement over DeepSeek 3.2, it is unlikely to surpass Sonnet 4.5. The commenter speculates that GLM 4.7 might be almost on par with Gemini 3 Flash in terms of coding capabilities, but with a different architecture.
    - AnticitizenPrime highlights a novel feature in GLM 4.7, which includes diagrams in the reasoning and planning stages. This is noted as a first for models of this type, potentially enhancing the model's ability to handle complex tasks that require visual planning and reasoning.
    - waste2treasure-org and jacek2023 express disappointment over the absence of Gemma 4 and Air, respectively, indicating a demand for these models. This suggests that while GLM 4.7 is a significant release, there is still anticipation and expectation for other models in the community.
- [**GLM 4.7 released!**](https://www.reddit.com/r/LocalLLaMA/comments/1pt5jfn/glm_47_released/) (Activity: 309): **GLM-4.7 has been released, offering significant advancements over its predecessor, GLM-4.6, particularly in areas such as coding, complex reasoning, and tool usage, establishing new open-source state-of-the-art (SOTA) standards. The model also enhances performance in chat, creative writing, and role-play scenarios. Notably, GLM-4.7 introduces new cognitive features like *Interleaved Thinking*, *Preserved Thinking*, and *Turn-level Thinking*, which improve task stability and control by enabling thought processes between actions and maintaining consistency across interactions. The model weights are available on [Hugging Face](http://huggingface.co/zai-org/GLM-4.7), and further technical details can be found in the [tech blog](http://z.ai/blog/glm-4.7).** Commenters highlight the rapid development cycle of GLM models and express anticipation for the Unsloth UD_Q2_K_XL quantization, which has previously enhanced the performance of GLM models. The introduction of new thinking modes is seen as a significant improvement for handling complex tasks.
    - ResearchCrafty1804 highlights that GLM-4.7 introduces new cognitive features such as Interleaved Thinking, Preserved Thinking, and Turn-level Thinking. These enhancements aim to improve the model's ability to handle complex tasks by maintaining consistency and stability across different interactions. More details can be found in the [documentation](http://docs.z.ai/guides/capabilities/thinking-mode).
    - r4in311 provides a comparative analysis of GLM 4.7 against other models like GPT 5.0 and Sonnet 4.5 using a specific prompt for generating a 'Voxel Pagoda'. The results show that while GLM 4.7 is competitive, it requires more shots and troubleshooting to achieve similar results, indicating it is not yet state-of-the-art (SOTA) compared to its peers. The examples provided in [jsfiddle](https://jsfiddle.net/zhrqmw4p) illustrate these differences.
    - UserXtheUnknown notes that GLM 4.7 performed exceptionally well on a specific task, the 'rotating house demo', surpassing even Gemini 3.0. This suggests that while GLM 4.7 may not be the best in all areas, it excels in certain tasks, showcasing its potential in specific applications.

### 3. NVIDIA's DGX Spark and Unsloth Guide

- [**NVIDIA made a beginner's guide to fine-tuning LLMs with Unsloth!**](https://www.reddit.com/r/LocalLLaMA/comments/1pt18x4/nvidia_made_a_beginners_guide_to_finetuning_llms/) (Activity: 379): **NVIDIA's guide on fine-tuning large language models (LLMs) with Unsloth provides a comprehensive overview of using NVIDIA GPUs for model customization. The guide covers three main fine-tuning methods: LoRA (Low-Rank Adaptation), FFT (Full Fine-Tuning), and RL (Reinforcement Learning), and discusses the necessary data and VRAM requirements. It emphasizes the use of Unsloth, an open-source framework, to efficiently tailor models for specific tasks using NVIDIA's DGX Spark and RTX GPUs. The guide also introduces the Nemotron 3 family of open models, highlighting NVIDIA's commitment to open-source AI development.** One commenter appreciates NVIDIA's contribution to open-source models but criticizes the company's impact on the hardware market. Another user expresses frustration with access issues, indicating a need for a mirror of the content.
- [**DGX Spark: an unpopular opinion**](https://www.reddit.com/r/LocalLLaMA/comments/1ptdtmz/dgx_spark_an_unpopular_opinion/) (Activity: 367): **The image depicts the NVIDIA DGX Spark, a compact computing unit designed for data science and machine learning tasks, particularly in environments with limited access to high-performance GPUs. The post highlights its utility for small research groups, emphasizing its all-in-one design and substantial memory capacity, which allows for effective prototyping and training of foundation models. Despite not matching the speed of high-end GPUs like the H100, its design makes it accessible for groups with limited funding. The comments discuss its strengths in VRAM and power efficiency, while noting its limitations in memory bandwidth and performance compared to other GPUs like the 3090.** Commenters generally agree that the DGX Spark is well-suited for its intended demographic, such as small research groups, despite some disappointment in its memory bandwidth. There is a consensus that it serves as an entry point into NVIDIA's ecosystem, with the expectation of scaling up to more powerful GPUs in the future.
    - Kwigg highlights that the DGX Spark, while offering substantial VRAM and efficient power usage, falls short in memory bandwidth, making it less ideal for LLM inference tasks compared to its cost. This is a critical point for users focused on inference rather than training, where memory bandwidth is a significant bottleneck.
    - FullstackSensei points out that Nvidia's strategy with the DGX Spark is to introduce users to the CUDA ecosystem at a lower cost, particularly targeting educational institutions. This approach aims to create a dependency on Nvidia's ecosystem, encouraging future investments in larger, more expensive GPU clusters.
    - pineapplekiwipen compares the DGX Spark to consumer GPUs like the 3090, noting that while the Spark is slower, it offers better power efficiency. However, in terms of price and performance, a setup with multiple 3090s could outperform a single DGX Spark, highlighting a trade-off between power consumption and computational efficiency.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Gemini and SCAIL Model Developments

- [**Gemini 3 Flash can reliably count fingers (AI Studio – High reasoning)**](https://www.reddit.com/r/singularity/comments/1psx30g/gemini_3_flash_can_reliably_count_fingers_ai/) (Activity: 873): **Gemini 3 Flash, a model from AI Studio, demonstrates high reasoning capabilities by reliably counting fingers in images. This model showcases significant improvements in visual recognition tasks, particularly in accurately identifying and counting objects in complex scenes. The model's performance is highlighted by its ability to handle variations in hand positions and lighting conditions, which are common challenges in computer vision applications.** Commenters discuss the model's robustness in diverse conditions, noting its potential applications in real-world scenarios such as gesture recognition and human-computer interaction. Some express interest in the underlying architecture and training data that enable such high accuracy.
    - The discussion highlights the capabilities of the Gemini 3 Flash model in accurately counting fingers, which suggests a significant advancement in its reasoning abilities beyond simple image matching. This counters the common criticism that AI models merely perform pattern recognition without understanding. The model's ability to provide correct answers quickly, as noted by users, indicates efficient processing and potentially improved algorithms for visual reasoning tasks.
- [**SCAIL IS DEFINITELY BEST MODEL TO REPLICATE THE MOTIONS FROM REFERENCE VIDEO**](https://www.reddit.com/r/StableDiffusion/comments/1pswlzf/scail_is_definitely_best_model_to_replicate_the/) (Activity: 538): **The post discusses the SCAIL model for motion transfer, highlighting its ability to replicate motions from a reference video without distorting the main character's dimensions, unlike other models such as Wan Animate and Steady Dancer. The workflow for using SCAIL is shared via a [Google Drive link](https://drive.google.com/file/d/1fa9bIzx9LLSFfOnpnYD7oMKXvViWG0G6/view?usp=sharing).** A commenter inquires about the minimum VRAM required to run the SCAIL model, indicating a technical interest in the model's hardware requirements.
    - A user inquired about the minimum VRAM required to run the SCAIL model, which is crucial for determining the hardware compatibility and performance efficiency of the model. This is a common concern for users looking to implement or test models on their own systems, especially when dealing with high-fidelity motion replication tasks.
    - Another user expressed interest in comparing the output of the SCAIL model with the original reference video, specifically focusing on the accuracy of hand movements. This highlights the importance of detailed motion fidelity in evaluating the effectiveness of motion replication models, as hands are often complex and challenging to replicate accurately.
    - A comment speculated on the potential of SCAIL to replace traditional motion capture suits, suggesting that if the model can achieve high accuracy and reliability, it could offer a more accessible and less cumbersome alternative to existing motion capture technologies. This points to a broader trend in AI where software solutions are increasingly being considered as replacements for hardware-based systems.
- [**Z-Image + SCAIL (Multi-Char)**](https://www.reddit.com/r/StableDiffusion/comments/1psr58j/zimage_scail_multichar/) (Activity: 1349): **The post discusses the use of Z-Image combined with SCAIL for generating multi-character animations, highlighting that SCAIL's poses appear genuinely 3D with improved depth and body orientation compared to alternatives like Wan Animate or SteadyDancer. The user reports a rendering time of** `26 minutes` **for** `6 steps` **at a resolution of** `736x1280` **using an RTX 5090 GPU, indicating significant computational demand.** Commenters are questioning the prevalence of generative AI in creating dancing videos, expressing curiosity about the source of such animations and the duration for which video can be generated.
    - A user inquired about the potential of using the discussed technology to generate 3D skeletal animations. This suggests an interest in the application of Z-Image and SCAIL for more complex animation tasks, possibly leveraging the models' ability to create realistic motion sequences that could be translated into 3D environments.
    - Another user expressed disbelief at the realism of the generated characters, indicating that the models used in Z-Image and SCAIL are capable of producing highly lifelike images. This points to the effectiveness of these models in achieving photorealistic results, which could have implications for industries relying on realistic digital human representations.
    - There was a question about the duration for which video can be generated using these models. This touches on the technical limitations or capabilities of Z-Image and SCAIL in terms of video length, which is a critical factor for applications in media production and other fields requiring extended video content.
- [**Prepare for an awesome 2026!**](https://www.reddit.com/r/singularity/comments/1pspk5q/prepare_for_an_awesome_2026/) (Activity: 1693): **The image is a tweet by Kevin A. Bryan reflecting on the state of AI models and technologies as of December 1st of the previous year. It highlights the absence of a robust Gemini model, the limitations of image models in text interpretation, and the nascent stage of video models. The tweet also mentions the introduction of Deepseek R1 with test time inference and the progress of FrontierMath and HLE, suggesting significant advancements and planning for 2026. This context suggests a forward-looking perspective on technological evolution and the potential for breakthroughs in AI and machine learning by 2026.** Commenters reflect on the rapid pace of technological change, with some expressing skepticism about the transformative impact of these advancements by 2026. There is a sentiment that while broad changes may be incremental, significant shifts could occur at the frontier of technology, particularly in AI and machine learning.
    - Manfr3dMacx highlights the lag between technological advancements and their widespread adoption, suggesting that while 2026 might feel similar to 2025 in general, the cutting-edge developments in AI models will create significant differences at the frontier. This underscores the importance of productization and mass adoption in realizing the full potential of new technologies.
    - Profanion notes the progression of image generation technology, mentioning that in March 2025, the first image generator capable of consistently handling text was released with GPT-image o1. This was followed by further advancements with Nanobanana Pro and GPT Image 1.5, indicating rapid development in AI's ability to handle complex tasks like text generation within images.
    - Cagnazzo82 comments on the exponential pace of AI development, pointing out that the expectations for what models can achieve are constantly evolving. This reflects a broader trend in AI where the capabilities of models are advancing rapidly, leading to shifting benchmarks and goals for what these technologies can accomplish.

### 2. AI in Creative and Engineering Processes

- [**WSJ just profiled a startup where Claude basically is the engineering team**](https://www.reddit.com/r/ClaudeAI/comments/1psoe2e/wsj_just_profiled_a_startup_where_claude/) (Activity: 615): **A 15-year-old entrepreneur has developed an AI-powered financial research platform with approximately** `50,000` **monthly users, primarily using Anthropic's Claude as the main engineering tool. The platform was created with minimal direct coding (around** `10 lines`**), leveraging Claude for software generation and other models like ChatGPT and Gemini for supporting tasks. The founder focuses on system design and distribution rather than traditional implementation, operating without employees or a conventional development team. A public company even republished an AI-generated report from the platform, mistaking it for professional research. [WSJ article](https://www.wsj.com/business/entrepreneurship/teenage-founders-ecb9cbd3?st=AgMHyA&reflink=desktopwebshare_permalink).** Comments highlight skepticism about the platform's lack of paying customers and the founder's familial support from parents in big tech and finance. There's debate on whether AI democratizes SaaS creation to the point of devaluing it, with some seeing this as a basic economic principle.
    - The discussion highlights skepticism about the startup's viability, noting that it lacks paying customers and is heavily supported by the founder's parents, who have backgrounds in big tech and finance. This raises questions about the sustainability and independence of such AI-driven ventures, especially when initial success is heavily reliant on external support rather than market demand.
    - There is a debate on the implications of AI making it easier to create SaaS products. One commenter argues that if AI lowers the barrier to entry significantly, it could lead to market saturation and devaluation of SaaS offerings, as basic economic principles suggest that increased supply without corresponding demand can reduce value.
    - Concerns are raised about the security and privacy implications of using AI-driven applications, especially those developed with minimal oversight or expertise. The potential for personal information theft is a significant risk, particularly if the application is not backed by robust security measures and is developed by individuals with limited experience in handling sensitive data.
- [**I feel really stupid for not having tried this before**](https://www.reddit.com/r/StableDiffusion/comments/1psocuo/i_feel_really_stupid_for_not_having_tried_this/) (Activity: 599): **The post discusses a user's discovery that using their native language in AI image generation with Z-image Turbo, which uses the Qwen-3 text encoder, results in images that reflect the cultural and geographical characteristics of their locale. This suggests that the model's training data includes images tagged in various languages, allowing for more localized and culturally relevant outputs when prompted in those languages. This behavior is noted in the documentation of similar models like Flux2, which indicates that prompting in a target language can enhance visual consistency for that locale.** One commenter notes that this feature is documented in Flux2, suggesting that language-specific prompting is a known technique for achieving locale-specific image generation. Another commenter humorously points out that this does not work with fictional languages like Klingon, highlighting the limitations of the model's training data.
    - FrenzyX highlights a technical detail from the Flux2 documentation, noting that prompting in a target language can enhance visual consistency for that locale. This suggests that the model's training data includes diverse language tags, which can influence the output when using specific language prompts.
    - Recoil42 explains that using a target language in prompts biases the model towards images labeled in that language within the training set. This implies that the model's performance can be influenced by the language distribution in its training data, affecting how it interprets and generates images.
    - Goldie_Wilson_ humorously notes that this technique does not work with Klingon, implying limitations in the model's training data regarding less common or fictional languages. This highlights the importance of the training dataset's language coverage in determining the model's capabilities.
- [**Real image vs Nano Banana Pro vs GPT, can you easily guess which one is real?**](https://www.reddit.com/r/ChatGPT/comments/1pt2mhf/real_image_vs_nano_banana_pro_vs_gpt_can_you/) (Activity: 2644): **The post describes an experiment where a real image was compared against AI-generated images created using Gemini and GPT. The real image was described by GPT, and this description was used to generate new images in both Gemini and GPT. The goal was to see if viewers could distinguish the real image from the AI-generated ones.** One commenter noted that distinguishing between the real image and the Gemini-generated image was challenging, while the GPT-generated image was more easily identifiable as artificial.
    - Benboozzled highlights the difficulty in distinguishing between images generated by the Gemini model and real photographs, noting that images produced by the Chat model are more easily identifiable as artificial. This suggests that the Gemini model has achieved a level of photorealism that challenges human perception, whereas the Chat model still exhibits detectable artifacts or stylistic cues that reveal its synthetic nature.
    - SuddenWerewolf7041 expresses concern over the implications of AI-generated content on human value and intelligence. The commenter suggests that the increasing sophistication of AI models in creating indistinguishable content from real human creations could lead to a 'dilution' of human skills and the ability to discern authenticity, raising ethical and societal questions about the role of AI in creative fields.
    - sgtcfox provides a sequence of numbers (1,3,3,3,1,3) as a guess for identifying real images among AI-generated ones, indicating a methodical approach to the challenge. This reflects the complexity and potential for error in distinguishing real from AI-generated images, underscoring the advanced capabilities of current AI models in image synthesis.
- [**Time-to-Move + Wan 2.2 Test**](https://www.reddit.com/r/StableDiffusion/comments/1pt19u6/timetomove_wan_22_test/) (Activity: 2539): **The post discusses a test using mickmumpitz's ComfyUI workflow for animating movement by manually shifting objects or images, tested with both high-quality and iPhone cameras. The author chose lower quality footage for a more grounded feel, suggesting a potential future test with higher quality footage. The workflow allows for creative manipulation of scenes, as demonstrated in the linked [tutorial](https://youtu.be/pUb58eAZ3pc?si=EEcF3XPBRyXPH1BX).** A commenter compared the demo to a video by **Corridor Crew** and expressed interest in their custom node for `dwpose`. Another asked about the technique used to remove objects like a metal straw and fingers from the video, indicating interest in the technical process behind the animation.

### 3. Debates on AI and Intelligence

- [**Deepmind CEO Dennis fires back at Yann Lecun: "He is just plain incorrect. Generality is not an illusion."**](https://www.reddit.com/r/singularity/comments/1pt05w7/deepmind_ceo_dennis_fires_back_at_yann_lecun_he/) (Activity: 1158): **The image is a social media post by Demis Hassabis, CEO of DeepMind, responding to Yann LeCun's assertion that general intelligence is an illusion. Hassabis argues that LeCun is confusing general intelligence with universal intelligence, emphasizing that while practical systems require specialization, the architecture of a general system, akin to a Turing Machine, can theoretically learn anything computable given sufficient resources. This debate highlights differing views on the nature of intelligence, with Hassabis defending the potential for generality in AI systems, contrasting with LeCun's view of human intelligence as highly specialized.** The comments reflect a recognition of the value in debates between leading AI researchers like Hassabis and LeCun, suggesting that such discussions can lead to significant advancements in understanding AI and intelligence.
    - The debate between **DeepMind CEO Dennis Hassabis** and **Yann LeCun** centers on the concept of generality in AI. Hassabis argues that generality is not an illusion, countering LeCun's skepticism about the existence of general intelligence. This reflects a broader discussion in AI research about the feasibility and definition of Artificial General Intelligence (AGI), with implications for how AI systems are designed and evaluated.
    - **Yann LeCun's** assertion that humans aren't truly general intelligence has sparked controversy. Critics argue that this perspective undermines the concept of general intelligence by suggesting that because humans can't perform every conceivable task, they aren't truly 'general'. This debate highlights the challenges in defining and measuring generality in both biological and artificial systems.
    - The discussion touches on the philosophical and technical aspects of AGI, with **Dennis Hassabis** emphasizing the potential for AI to achieve generality akin to human intelligence. This contrasts with **Yann LeCun's** more cautious stance, which questions the practicality and current understanding of general intelligence. The debate underscores the ongoing exploration of what constitutes general intelligence and how it can be realized in AI systems.
- [**ChatGPT isn’t an AI :/**](https://www.reddit.com/r/ChatGPT/comments/1psk1mn/chatgpt_isnt_an_ai/) (Activity: 2000): **The image in the Reddit post is a text-based meme that critiques the understanding of ChatGPT and large language models (LLMs) as AI. It argues that LLMs, like ChatGPT, are not truly AI but rather statistical models that predict the most probable next word, similar to a phone's keyboard. The post suggests that LLMs are always 'hallucinating' and only appear correct by chance, not due to any real understanding. This reflects a common misconception about LLMs, which are indeed a form of AI, albeit limited in their understanding and reasoning capabilities compared to human cognition.** Commenters generally agree that while LLMs are a type of AI, they operate on statistical principles rather than true understanding. Some argue that this distinction is more about romanticizing human cognition than a fundamental difference, noting that human knowledge is also based on learned experiences and memory.
    - Kaveh01 discusses the nature of knowledge in LLMs compared to human cognition, suggesting that both systems rely on experience and memory to generate responses. They argue that while LLMs lack certain cognitive abilities like transferability, this limitation is partly due to the text-based nature of their training data, which also constrains human understanding if similarly limited.
    - Machiavellian_phd draws a parallel between human cognitive processes and LLMs, highlighting that both systems engage in predictive processing. They note that humans often 'hallucinate' by predicting outcomes based on incomplete information, similar to how LLMs generate outputs based on their training data. This comment emphasizes that AI, including LLMs, is a broad category encompassing various systems, from complex models to simple feedback mechanisms like thermostats.
    - The discussion touches on the limitations of LLMs, particularly their lack of transferability and the tendency to 'hallucinate' when faced with gaps in training data. This is compared to human cognitive processes, which also involve prediction and error, suggesting that the differences between AI and human intelligence may be more about degree than kind.
- [**What’s the most useful thing ChatGPT can do today that people still don’t realize?**](https://www.reddit.com/r/ChatGPT/comments/1pt4t35/whats_the_most_useful_thing_chatgpt_can_do_today/) (Activity: 1302): **A Reddit user highlights a practical use of ChatGPT for meal planning by leveraging its voice transcription feature to list groceries and generate meal ideas, significantly reducing food waste. Another user appreciates ChatGPT's conversational learning capabilities, especially for niche interests like history and quantum physics, noting its ability to provide knowledgeable discussions when human counterparts are unavailable. However, some users express frustration with ChatGPT's repetitive introductory phrases in responses.** The comments reflect a mix of appreciation for ChatGPT's utility in practical tasks and learning, alongside minor frustrations with its response style. The tool's ability to engage in knowledgeable discussions on niche topics is particularly valued.
    - SylvaraTheDev highlights the utility of ChatGPT as a media parser, particularly for processing PDFs. The AI can generate executive summaries that capture key details, which is especially useful since many PDFs lack such summaries. This feature can significantly enhance productivity by quickly distilling essential information from lengthy documents.
    - dennis-w220 discusses the educational potential of ChatGPT for learning through conversation. The AI can engage in discussions on diverse topics like history and quantum physics, providing a unique opportunity to explore interests in a relaxed manner. This conversational learning approach is valuable for hobbyists seeking knowledge in niche areas where finding a knowledgeable human counterpart might be challenging.
    - polarwaves shares an unconventional use of ChatGPT as a substitute for therapy, highlighting its role in providing emotional support. While not a perfect replacement for professional therapy, ChatGPT offers a platform for users to vent and manage anxiety, with the ability to request feedback and challenge from the AI, simulating aspects of therapeutic interaction.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5.1
> 

**1. Next‑Gen LLMs and Benchmarks Go Global**

- **GLM‑4.7 Sneaks Into Top Spots**: Zhipu’s **GLM‑4.7** quietly launched, becoming the #1 open WebDev model with a **1449** score on the [LMArena WebDev leaderboard](https://lmarena.ai/leaderboard/webdev) and getting packaged as **GLM‑4.7 Air** in the Nous ecosystem via [zai-org/GLM‑4.7](https://huggingface.co/zai-org/GLM-4.7).
    - Zhipu also pushed MLX quants such as [**GLM‑4.7‑mlx‑3Bit**](https://huggingface.co/mrtoots/GLM-4.7-mlx-3Bit) and [**GLM‑4.7‑mlx‑4Bit**](https://huggingface.co/mrtoots/GLM-4.7-mlx-4Bit), while users in Moonshot and Nous discords compared it favorably to **Gemini 3 Pro** and discussed how to disable or preserve its **“thinking”** traces to suit different reasoning workloads.
- **China’s ERNIE‑5.0 and Solar‑Open‑100B Strut Their Stuff**: Baidu’s **ERNIE‑5.0‑Preview‑1203** hit **1451** on the [Text Arena leaderboard](https://lmarena.ai/leaderboard/text), a **23‑point** jump over the `1103` preview and now the top Chinese text model in that benchmark.
    - Concurrently, Upstage released **Solar‑Open‑100B** on Hugging Face at [upstage/Solar‑Open‑100B](https://huggingface.co/upstage/Solar-Open-100B), giving the open‑source scene another heavyweight dense model that communities like Nous are considering as a serious alternative in future evaluations.
- **Phi4 Versus GPT‑OSS and the Style‑Tuning Frontier**: Unsloth users circulated an **arXiv** preprint, [“Phi4”](https://arxiv.org/html/2508.12461v1), claiming **Phi4** can outperform **GPT‑OSS‑20B**, surprising people who assumed a good dense **14B** should clearly beat a **20B MoE** only with strong architectural and data choices.
    - In parallel, Nous researchers debated how few examples are needed to fine‑tune large models to replicate styles like *purple prose* or *realistic fiction*, underscoring that frontier models such as Phi4/GLM/Solar are now being judged as much on **controllable style transfer** as on raw benchmark scores.

**2. Training & Inference Performance Arms Race**

- **TorchAO MXFP8 Turbocharges MoE Training**: The **torchao v0.15.0** release adds **MXFP8 MoE training** and shows a **1.2× end‑to‑end speedup** at equal convergence versus **bf16** when training **Llama4 Scout** on a **64‑node GB200 Crusoe cluster**, per the [torchao v0.15.0 notes](https://github.com/pytorch/ao/releases/tag/v0.15.0).
    - MXFP8 MoE kernels now ship in binary builds for **CUDA 12.8+** with **safetensors** and parameter‑level quantization, so users can *pip install* instead of compiling from source and still get production‑grade low‑precision kernels for large‑scale MoE training.
- **QSInference Outsprints FlashAttention on Long Contexts**: A GPU MODE member shared **QSInference**, a Triton implementation of **quantized sparse attention** for long‑context LLMs, which claims to be *8× faster than FlashAttention‑2* and *3× faster than block‑sparse attention* at **128k** context in the [QSInference GitHub repo](https://github.com/yogeshsinghrbt/QSInference).
    - QSInference targets long‑sequence inferencing where attention compute dominates, and its Triton kernels attracted interest from performance engineers juggling **B200/GB200** hardware, **Helion** kernels in **vLLM**, and mixed‑precision schemes like **MXFP8**.
- **Real‑World Hardware Tales: Strix Halo, Used GPUs, and Kernel Bounties**: LM Studio users reported **Strix Halo** APUs with shared RAM performing well on **MoE** models but poorly on dense ones, citing that dense models *“compute against every parameter at once”* while MoEs touch only a subset and pointing to [Max Kruse’s model‑type guide](https://maxkruse.github.io/vitepress-llm-recommends/model-types/) for intuition.
    - Across GPU MODE and LM Studio, people debated used **3090/3090 Ti** plus **NVLink** versus newer cards, submitted kernel benchmarks like `vectoradd_v2` that hit **233 µs** and **1st place** on **B200**, and traded debugging war stories about **Triton/Gluon** kernels where `wgmma.mma_async` got serialized, underscoring how much perf now hinges on low‑level kernel craft.

**3. Agent Frameworks, Protocols, and SDKs Grow Up**

- **Vercel’s AI SDK 6 Goes All‑In on Agents and MCP**: Vercel released **AI SDK 6** with **local agent support**, **tool‑execution approval**, full **Model Context Protocol (MCP)** integration, beefed‑up DevTools, and standardized **JSON‑schema** tooling, announced in an [AI SDK tweet](https://xcancel.com/aisdk/status/2003156089177792827?s=46).
    - Latent Space members framed this as the SDK catching up to modern **agentic** patterns: hooking into MCP tools, enforcing human‑in‑the‑loop approvals, and giving frontend teams a batteries‑included stack for multi‑model, tool‑using apps rather than rolling brittle custom orchestrators.
- **MCP Contributors Wrestle With Token Economics**: In the official **MCP Contributors** server, developers complained that current MCP integrations **re‑send huge tool protocol descriptions** on every call, inflating token costs even when most tools go unused.
    - They floated ideas like **lazy tool‑schema transmission** and **caching protocol definitions**, but maintainers clarified that schema sending is controlled by the *client host* and tools simply **cannot be invoked without their schema**, so any fix must live in smarter client caching and change‑notification handling rather than the core protocol.
- **SmolAgents, DSPy Skills, and OpenRouter SDKs Aim for Smarter Automation**: Hugging Face contributors added **CustomCodeAgent** to the `smolagents` framework via [PR #1912](https://github.com/huggingface/smolagents/pull/1912), implementing `local_container` and `RemoteWorkspace` over local Docker to sandbox code‑running agents, while DSPy users released a [**skill-optimization** repo](https://github.com/instavm/skill-optimization) and asked *“if prompts can be optimized, why not skills?”*.
    - OpenRouter announced new **SDK helpers** for context/workflow management and **complexity‑based model selection**—letting the SDK change `model_id` mid‑flow based on tool outputs, as described in their [next‑turn params docs](https://openrouter.ai/docs/sdks/call-model/next-turn-params#complexity-based-model-selection)—even as some developers warned against over‑abstracting and locking themselves into provider‑specific orchestration layers.

**4. Finetuning, Loss Design, and Model Safety/Backdoors**

- **DPO Masking and Custom Losses Target Reasoning and Perception**: Unsloth users explored **DPO with masked reasoning traces** to fine‑tune reasoning models: the idea is to treat the model’s own answer as the **negative example**, a curated answer as the **positive**, and **mask loss on the chain‑of‑thought** so you steer style without wrecking internal reasoning.
    - In the same community, another engineer built a **Delta E** color‑difference loss for **LoRA training** of **Qwen2.5‑VL‑3B‑Instruct**, with suggestions to either patch loss computation directly inside Unsloth or subclass `SFTTrainer` for compatibility with standard Hugging Face training loops.
- **MoE vs Dense Cost Models and Long‑Task Alignment**: A **MoE vs dense** thread in Unsloth’s help channel leaned on an Epoch article, [“MoE vs dense models: inference”](https://epoch.ai/gradient-updates/moe-vs-dense-models-inference), and rules of thumb like using `sqrt(total_params × active_params)` to compare quality‑per‑compute for deploying mixture models.
    - Separately, tinygrad’s internal roadmap linked a **METR** post, [“Measuring AI Ability to Complete Long Tasks”](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/), as inspiration for bounty work on **grad‑acc**, **JIT**, **flash‑attention**, and visualization, highlighting how training stack design and evaluation for **long‑horizon tasks** are converging concerns.
- **Implicit Backdoors and Metadata‑Driven Cultural Control**: Eleuther’s interpretability channel discussed an **implicit backdoors** paper, [“Implicit Backdoors”](https://arxiv.org/abs/2512.09742), arguing we under‑use semantically‑meaningful tags during training and could instead encode *latent switches* that activate distinct behaviors or personas.
    - Members proposed pretraining with cheap **metadata prefixes** (author, date, source type) so that later finetuning can nudge the model’s *perceived time* (e.g., to **2025**) or cultural frame without re‑labelling everything, leveraging the idea that we only need a *loose prior* to record metadata that becomes powerful control signals post‑hoc.

**5. Developer‑Facing Apps, Pricing, and Ecosystem UX**

- **Comet Browser, Grok Voice, and Okuchat Court Power Users**: Perplexity users praised the **Comet** browser for using **0.2–1.2 % CPU** versus Chrome’s **8–10 %** at similar tab counts, while bundling an **AI assistant** and **ad‑blocker** that make it feel like *“a better Chrome with AI built in.”
    - Elsewhere, OpenAI community members crowned **Grok’s AVM voice** the most natural and *“straight to the point”* AI voice for car‑ride Q&A, and OpenRouter’s **Okuchat** app at [okuchat.com](https://okuchat.com/) launched as a multi‑LLM chat frontend (Claude, GPT, Gemini, Kimi, DeepSeek), though early feedback flagged outdated **GPT‑4** branding and missing latest models in its picker.
- **Subscriptions, Credits, and Key Markets Get Stress‑Tested**: Perplexity’s community argued that the **$200 Perplexity Max** tier needs a **$100** option with clearer limits, while also noting **Perplexity Pro** keys appearing for **< $1** on Russian marketplaces via promo‑code arbitrage, raising questions about user‑acquisition economics.
    - On **Manus.im**, Pro users complained that **Manus v1.5** eats **300 daily credits** in under **30 minutes**, that *“free chat”* stops working at zero balance, and demanded transparent accounting plus policy rollback, mirroring broader anxiety about opaque metering and perceived bait‑and‑switch in AI SaaS.
- **IDE and Browser Dev Tools in Flux**: Cursor users dissected grandfathered **unlimited auto‑select** (pre‑15‑Sep annual plans) versus newer capped plans where, once **Bonus** usage is exhausted, they are forced onto **Grok** only, with some calling Grok *“the worst model ever”* for simple HTML edits while others defended it when driven by good prompts.
    - In parallel, Hugging Face pushed infra‑level tools like **LlamaCpp server**—documented in the [llama.cpp server README](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)—for auto‑balancing multi‑GPU inference, and `smolagents` and `laravel-openrouter` ([GitHub](https://github.com/moe-mizrak/laravel-openrouter)) matured into de‑facto glue layers integrating frontier models into real developer workflows.

---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Comet Browser outperforms Chrome**: A member reported that **Comet** browser uses significantly less CPU than **Chrome**, with **Chrome** at **8-10%** CPU and **Comet** at **0.2%** with peaks to **1.2%**.
   - The member lauded **Comet's** built-in **AI assistant** and **ad blocker**, calling it a better version of **Chrome** with **AI** integration.
- **Members Recommend Image Generation Models**: Members debated image generation models, suggesting **Image Gen 1** for realistic outputs and **Nano Banana Pro**, with some noting **Gemini's** superiority over **ChatGPT**.
   - Concerns were voiced about the potential misuse of **AI image generation**, such as creating animated photos from real-life images and potentially deep faking live videos.
- **Perplexity Max Pricing Sparks Debate**: Users debated the value of **Perplexity Max** at its current **$200** price, suggesting a more affordable **$100** tier with adjusted limits to attract more subscribers.
   - Comparisons were made to **Claude's** pricing, highlighting the need for clearly defined limits and discussion on the economics and profits for Perplexity. Some users complained about the usage limits.
- **Perplexity Pro Keys Available for Under $1**: Members mentioned that **Perplexity Pro** keys are being sold for under a dollar on Russian marketplaces due to some promo codes.
   - Speculation arose that Perplexity might be using this as a strategy for user acquisition and to attract funding.
- **Model Selection Bug Irks Perplexity Users**: Users reported a bug in **Perplexity** where the selected model resets to *"Best"* when opening a new tab or refreshing, requiring extra clicks to reselect the desired model.
   - The issue, potentially a cost-saving measure, affects both web and mobile platforms, though some users experience it less frequently.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Intel Fab Aims for Chip Dominance**: **Intel** is constructing a new American fab to manufacture **iron** and **AI chips**, seeking to reduce reliance on Taiwan and gaining access to advanced manufacturing equipment from Taiwan Semiconductor Manufacturing Co.
   - The new **Bro2nm microchip** is claimed to be the *world's most advanced*, providing a **10-15%** compute boost.
- **Sora Triumphs Over Captchas**: Members reported that **Sora** can successfully solve **captchas** and **recaptchas**, with attempts made to test its capabilities using a provided [image](https://cdn.discordapp.com/attachments/1340554757827461211/1452527487568318554/image.png).
   - The attempt to get Sora to solve a captcha failed because the user didn't provide any instructions.
- **Image-to-Video Model Eyes Arena Integration**: An **Image-to-Video (i2v) model** team seeks community evaluation through integration into the **Video Arena**, offering to cover inference costs and provide necessary documentation.
   - The team has been closely watching the amazing work being done at **LMArena**.
- **ERNIE-5.0 Dominates Text Arena**: Baidu's `ERNIE-5.0-Preview-1203` achieves the top spot on the [Text Arena leaderboard](https://lmarena.ai/leaderboard/text) with a score of **1451**.
   - This marks a **23 point increase** since `ERNIE-5.0-Preview-1103`, highlighting Chinese advancements in text models.
- **GLM-4.7 Climbs WebDev Ladder**: `GLM-4.7` by Z.ai secures #6 on the [WebDev leaderboard](https://lmarena.ai/leaderboard/webdev), claiming the title of #1 open model for WebDev.
   - `GLM-4.7`'s score of **1449** represents an **83 point increase** over its predecessor, `GLM-4.6`.



---



## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **AFK Record Shattered!**: A member claimed a **3ms AFK** record using **Kali Linux** and network packet tools, sharing [a YouTube video](https://www.youtube.com/watch?v=d32KeZBHVdA) as reference.
   - They mentioned the possibility of altering profile handles with packets, but noted Discord's built-in defenses against replay attacks.
- **ChatGPT Python File Leaked?**: A member claimed to have a copy of the **Python file** used by **ChatGPT** to assess text for violations of Terms of Service and policy, suggesting its potential use in jailbreak attempts.
   - Others suggested that a safeguard model such as **GPT 5.x** and **Claude 4.x** all have a pre-safety-classifier, sharing [this video](https://youtube.com/shorts/7T7bqNoMSCw?si=AJIw-XI1LNLrlN0L).
- **Google Drive as RAM? SSDs Beware!**: One member suggested using software to mount **Google Drive** as an **SSD** to *borrow 2TB of RAM*, but was warned about potentially destroying the SSD due to excessive writes.
   - Alternatives like **USB sticks** or **cheap thrift store hard drives** were recommended for VRAM use, with a reminder that SSDs have a maximum write limitation.
- **Unity + GPT5 = Jailbreak Alchemy!**: A member shared instructions for a **ChatGPT 5.2** jailbreak, involving pasting into **ChatGPT**, canceling the response, and saying *'Hi Unity'*, potentially requiring following the instructions of **Unity GPT5 Jailbreak**.
   - They attached several files and images related to the jailbreak method, implying a multi-step process for preparing **ChatGPT** for training.
- **Coding AI Skills Slammed!**: A user commented *their ai coding is shit*, without providing further details in the channel.
   - Another user asked for more information in DMs to better understand and address the problem.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Phi4 Might Beat GPT-OSS 20b**: An [arXiv link](https://arxiv.org/html/2508.12461v1) suggests **Phi4** could outperform **GPT-OSS 20b**, despite expectations that a good dense 14b model should surpass a 20b **MoE**.
   - The difference in supported context length between the two models was highlighted as a key factor.
- **DPO Masking Refines Reasoning Models**: Members explored using **DPO** with masked reasoning traces to finetune a reasoning model, aiming to control output style while preserving reasoning capabilities.
   - The suggestion was made to use the model's response as a negative example and a custom answer as a positive one, masking loss to shift the model's response bias.
- **Llama3 Quantization Stumbles on Sagemaker**: A member encountered difficulties running an **Unsloth quantized Llama3 70b** model on an **AWS Sagemaker ml.g4dn.12xlarge** instance, seeking community assistance.
   - The recommendation was to leverage *llama.cpp* for its speed and direct support for sharded models.
- **Whisper V3 Has Japanese Stutter**: Members reported that [**Whisper V3**](https://openai.com/blog/whisper-v3-is-now-available) is slow on **A100s** when processing long audio files in **Japanese** and has issues with repeating characters.
   - This limits its usefulness for tasks requiring speedy transcription and analysis.
- **Custom Loss Functions Beckon**: A user seeks to implement a custom **Delta E** loss function for **LoRA training** with **Qwen2.5VL-3B-Instruct** for color code extraction.
   - Suggested approaches included modifying the code directly within the **Unsloth** package or subclassing **SFTTrainer** for enhanced compatibility with Transformers.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT's Memory Lane**: **ChatGPT** rolled out a *Your Year with ChatGPT* summary to users in the **US, UK, Canada, New Zealand, and Australia** who have saved **memory and chat history** turned on.
   - Users were prompted to update their app to access the new feature and see their **personalized retrospective**.
- **Gemini's Finger Fumble**: Members observed that **Gemini 3 Pro** failed to accurately count the fingers on a hand emoji, but **NB Pro** was able to correctly identify the correct finger count when prompted iteratively.
   - A member joked that **Gemini** has a *failsafe to fail*, while others celebrated **NB Pro's** image recognition.
- **Grok's Voice Wins**: **Grok's AVM** (Audio Voice Mode) was lauded as the most natural-sounding AI voice currently available, delivering more direct answers compared to **GPT** and **Gemini**.
   - One user enjoys using **AVM** during car rides to refresh general knowledge, appreciating that it *gets straight to the point*.
- **LinkedIn AI Ethics Debate**: A discussion ignited on the ethics of using AI to automate LinkedIn content creation, raising concerns about potentially violating **LinkedIn's ToS** against automated posting, and a [prototype automation](https://cdn.discordapp.com/attachments/998381918976479273/1452603650667974841/image.png?ex=694a6a12&is=69491892&hm=904948941287091c53fc207f65089e246f157eff975835b0b74a2a0a9f8284e8&) that compiles news into three posts to be reviewed by the user was mentioned.
   - The debate centered on the balance between AI assistance and maintaining the authenticity of professional networking.
- **Framework reduces state loss and assumption**: A **ChatGPT framework** designed to enhance reliability in real-world applications aims to **reduce state loss, over-assumption, and “helpfulness drift.”**
   - A member is interested in identifying edge cases where the framework *breaks, contradicts itself, or fails to maintain consistency over longer interactions*.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Grandfathered Cursor Auto-Select**: Users with annual plans before **September 15th** retain unlimited auto-select, while others face limits; after exhausting monthly plans and bonus, users are restricted to the **auto-select model**.
   - A member pointed out that random free usage is called **Bonus** and will save up until its used; however, once the **Bonus** is exhausted, only **Grok** is available.
- **Cursor Code Diff Delays**: A user reported that code-level diff changes in chat output is delaying development work, seeking ways to disable it.
   - Another user suggested using a **VPN** or **1.1.1.1** from Cloudflare as a potential workaround.
- **Cursor Grok Model Mocked**: One user derided **Grok** as the *worst model ever*, citing its inability to handle even a single position change in an HTML file.
   - Another member countered that **Grok** is useful when used with *prompt = quality of life*.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Pinpoints Basel University**: A member used image analysis with **qwen-qwen3-v1-32b-istruct** to identify a sign as belonging to the **University of Basel's Department of Biomedical Engineering** and shared [a YouTube video](https://www.youtube.com/watch?v=w7JperqVfXI).
   - The user noted the model's accurate geographical location identification, achieving **9.63 tok/sec**.
- **Qwen3 Gets a 25% Performance Boost**: After the latest update to LM Studio, one user reported a **25% performance increase** with **Qwen3 Next Q4**, going from **15t/s to 20t/s** in LM Studio and **20t/s to 25t/s** in llama.cpp, using a **4070** with **64GB DDR5** on a **7950x**.
   - The user's config included **128k context**, **48/48 GPU offload**, and enabled settings such as 'Force Model Expert Weights onto CPU' and Flash Attention.
- **Strix Halo's RAM Impacts Model Performance**: Members are observing that **Strix Halo PCs** with shared RAM perform well on Mixture of Experts (**MOE**) models but poorly on dense models, because dense models compute against every parameter at once, while MOE models don't.
   - A member linked to a [LLM Recommendation](https://maxkruse.github.io/vitepress-llm-recommends/model-types/) for more information.
- **Used GPUs: Gamble or Goldmine?**: Channel members weighed pros and cons of used GPUs, with one considering selling their **4070TiS** and **3090** to buy water-cooled **3090 Tis** and an **NVLink bridge**.
   - While used GPUs are generally considered *pretty good* if authentic, it was noted that **V100s** may only work with **Vulkan**.
- **Cybersecurity Intel from Hackers Exposed**: A member shared insights on gathering cybersecurity information from hackers, internally running code with **Claude**, and processing **138 million passwords**.
   - They referenced the phrase about *rocks and houses made of glass* in relation to cybersecurity.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Parquet Extraction Celebrations Commence**: A member successfully extracted a subset into a **parquet file**, celebrating the achievement within the community.
   - They expressed gratitude to the community for their assistance throughout the extraction process.
- **Zero GPU Training Becomes Reality**: A member reported achieving **zero GPU usage** during training, a significant advancement in resource efficiency.
   - Another member suggested posting potential errors on the Spaces discussion board, encouraging collaborative troubleshooting.
- **LlamaCpp Powers Auto-Balancing Inference**: A member recommended using **LlamaCpp server** for deploying a 14B model with **vllm**, emphasizing its auto load balancing and multi-GPU support, with a link to the [LlamaCpp server GitHub page](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).
   - They provided a quick start command to streamline server setup, enhancing usability for others.
- **CustomCodeAgent Enters SmolAgents Framework**: A member submitted a PR featuring **CustomCodeAgent** for the `smolagent` framework, which implements `local_container` and `RemoteWorkspace` via locally running docker containers, with links to the [PR](https://github.com/huggingface/smolagents/pull/1912) and the [issue](https://github.com/huggingface/smolagents/issues/1908).
   - The member encouraged testing and integration with other Coding Agents to broaden its functionality.
- **Deterministic Folding Prioritized in New AI Prototype**: A member announced a **flow model** is on the roadmap, but they prioritized **deterministic constraint-based folding** for stability and validation in the current version.
   - They are trying to create an **AI without fine-tuning existing models**, highlighting the prototype status of the project.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Laravel-OpenRouter Attracts Maintainers**: The **Laravel-OpenRouter** package, a Laravel integration for OpenRouter, now has **140+ GitHub stars** and **60k+ installs** and creator Moe Mizrak has enabled [GitHub Sponsors](https://github.com/moe-mizrak/laravel-openrouter).
   - The package is designed to support long-term maintenance and improve developer experience in integrating with **OpenRouter**.
- **Okuchat Launches with LLM Model Switching**: A member launched an AI chat app, [Okuchat](https://okuchat.com), which allows users to switch between different LLM models, including **Claude**, **GPT**, and **Gemini**.
   - A member suggested that the website's meta description be updated due to **GPT-4**'s deprecation, and another pointed out the model lists (specifically **Claude**, **OpenAI**, **Kimi**, and **DeepSeek**) are missing some of the latest versions.
- **AI Counting Requires Structured Solutions**: A member sought a structured method for making AI count correctly, suggesting to use 1/200 with assistant prefill and a repetition penalty of 1.0.
   - Another member suggested using **structured outputs** with an object shape like `{ 1: thing, 2: thing, 3: thing ... etc }` or asking it to give the items in groups of more manageable numbers.
- **OpenRouter SDK Enhances Workflow Management**: OpenRouter is adding **helpers** for context/workflow management to the **SDK**, which makes API requests easier, as shown in [this documentation](https://openrouter.ai/docs/sdks/call-model/next-turn-params).
   - The SDK now supports changing the **model ID based on a tool call result**, called **complexity-based model selection**, documented [here](https://openrouter.ai/docs/sdks/call-model/next-turn-params#complexity-based-model-selection).



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton Confusion: OpenAI or Nvidia?**: The term **Triton** may refer to either **OpenAI Triton**, a Python-based *"cuda wrapper"*, or **Nvidia Triton**, a high-performance inference server, depending on context.
   - A user debugging performance issues in a **Triton/Gluon kernel** encountered serialization problems with **wgmma.mma_async** instructions.
- **Luma AI Supercharges PyTorch Team for Multimodal AGI**: Luma AI is hiring **kernel & performance engineers/researchers** to build **multimodal AGI** on **thousands of GPUs**, utilizing a custom **PyTorch** stack with custom kernels, to squeeze the most MFU out of both **AMD** and **NVIDIA GPUs**.
   - The systems team is seeking experts with strong PyTorch / CUDA / etc. skills to work on projects from foundational research to product, see [Luma AI careers page](https://jobs.gem.com/lumalabs-ai/a2feb190-455d-45e6-b488-7ac840f30fbd) for details!
- **Torchao v0.15.0 Accelerates MXFP8 MoE Training**: [Torchao v0.15.0](https://github.com/pytorch/ao/releases/tag/v0.15.0) introduces **MXFP8 MoE training**, achieving a **1.2x end-to-end training speedup** with the same convergence as bf16 for **Llama4 Scout** training on a **64 node GB200 Crusoe cluster**.
   - This release includes **MXFP8 MoE kernels** for CUDA 12.8+, safetensors enablement, and quantization with parameter-level targeting.
- **QSInference Claims 8x Speedup over Flash Attention-2**: **QSInference**, a new method employing quantized sparse attention for long context LLMs, claims to be *8x faster than flash attention-2* and *3x faster than block sparse attention* at context length 128k.
   - The shared [GitHub repo](https://github.com/yogeshsinghrbt/QSInference) provides a *Triton implementation* of **QSInference**.
- **Red Hat Investigates Helion Kernel Adoption**: A Red Hat team member is analyzing gaps in **Helion kernel adoption** within **vLLM** and has started submitting issues and proposals to the repo under GitHub ID xiaohongchen1991.
   - The team member has begun filing issues and proposals in the repo under GitHub ID xiaohongchen1991 and seeks feedback on the issues and proposals filed, proposing a formal review post-holidays.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Stress Spirals Out of Control**: A member posted a [YouTube video](https://www.youtube.com/watch?v=H_c6MWk7PQc) addressing the dangers of classifying too many tasks as top priority, which can lead to **team burnout**.
   - The video highlights that *when everything is top priority, you’ve collapsed prioritization into a single bucket that should only be used for a few urgent and important things.*
- **Spotify Playlists Get Archival Support**: A member shared a link to [Anna's Archive blog](https://annas-archive.org/blog/backing-up-spotify.html) that details methods for **backing up Spotify playlists**.
   - The discussion referenced both an older **Hacker News thread** ([link](https://news.ycombinator.com/item?id=46338339)) and a relevant **X post** ([link](https://x.com/ajwagenmaker/status/2003101042565853212?s=46&t=eWVlK1PU8XfB6f402GJJ9g)) providing additional context.
- **PostBC gives Policy Pretraining a Boost**: Andrew Wagenmaker introduced **Posterior Behavioral Cloning (PostBC)**, which is a method designed to pretrain policies from demonstrations to create an effective initialization for **reinforcement learning finetuning**.
   - According to [this tweet](https://xcancel.com/ajwagenmaker/status/2003101042565853212?s=46&t=eWVlK1PU8XfB6f402GJJ9g), the approach aims to maintain the original performance of the pretrained policy.
- **Vercel Uncorks AI SDK 6**: Vercel launched **AI SDK 6**, featuring **local agent support**, **tool execution approval**, full **Model Context Protocol (MCP) integration**, enhanced **DevTools**, and standardized **JSON schema support**, according to [this tweet](https://xcancel.com/aisdk/status/2003156089177792827?s=46).
   - The release aims to provide developers with the latest tools for building and deploying **AI-powered applications**.
- **AI Film Fest Frameworks Spark**: PJ Ace outlined a simplified framework used to train professional **Hollywood cinematographers** in **AI filmmaking tools** for a **million-dollar film festival submission** at [xcancel.com](https://xcancel.com/PJaccetturo/status/2002777819903062060).
   - The author offered to share the specific **prompts and processes** used in **X-Ware.v0**: [AI Film Festival Framework and Cinematography Tips] at [xcancel.com](https://xcancel.com/PJaccetturo/status/2002777819903062060).



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Agent Guides Sought**: A member inquired about downloadable guides for **DSPy agents**, with another member suggesting a [custom GPT on DSPy](https://chatgpt.com/g/g-69492acdddb48191b54e02fba9700f73-dspy-ai-engineer).
   - The GPT offers insights into **DSPy** usage and agent development.
- **LinkedIn Bashed for Promotion**: A user criticized **LinkedIn** for being full of self-promotion, but was quickly flamed for not understanding that **social media** is for that.
   - Alternative platforms like [Twitter](https://twitter.com/) and [Hacker News](https://news.ycombinator.com/) were suggested for deeper technical discussions.
- **Skill Optimization Repo Makes Debut**: A member announced their work on **skill optimization**, and that OpenAI had coincidentally embraced **skills** around the same time that they attended a **DSPy meetup**.
   - The [skill-optimization](https://github.com/instavm/skill-optimization) repo was released.
- **Prompt Optimization Discussions**: A member questioned, *if prompts can be optimized, why not skills* and asked if they could **self-promote** a [discord channel](https://discord.com/channels/1161519468141355160/1452640049978937445).
   - Optimization is key to **DSPy**.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Gemini 3 Pro Faces Off Against Flash in Early Reviews**: Early user impressions indicate a preference for **3 Flash** over **Gemini 3 Pro**, with some users also expressing significant enthusiasm for **Nano Banana**.
   - Speculation arose regarding the potential release of **K3**, suggesting anticipation for further advancements in model capabilities.
- **Xiaomi's MiMo V2 Flash Draws Acclaim**: A user lauded **Xiaomi's** new **MiMo V2 Flash** as *kinda fire*, praising **Zhipu's** work and linking to [this tweet](https://x.com/scaling01/status/2003115854066815044?s=46).
   - This release occurs as **Minimax** gears up for its **M2.1** release.
- **GLM-4.7 stealthily enters the scene**: The release of **GLM-4.7** occurred without the typical promotional fanfare.
   - The poster contrasted this with **Minimax's** extensive build-up for the **M2.1** release, highlighting differing marketing strategies.
- **Moonshot's K3 Faces Release Delay**: Speculation suggests that **Moonshot** may strategically delay the release of **K3** to allow for further development and to reclaim **SOTA** status upon release.
   - The poster predicted K3 would launch *a good 3-4 months* after **Deepseek V4/R2**.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Credits System Frustrates Users**: Users expressed frustration with the credits system, specifically mentioning the removal of the option to top up credits and wishing it would return, according to [Discord messages](https://discord.com/channels/1348819876348825620/1349440650495398020/1452523203858665472).
   - A representative stated that there's no option to buy extra credits beyond the top tier plan, suggesting prompt refinement or waiting for the monthly refresh and that joining official events is a great way to get extra points, although they would pass the user's thoughts to the product team.
- **Manus v1.5 Credit Consumption Criticized**: A Pro subscriber criticized **Manus v1.5** for its high daily credit consumption rate, stating that the **300 credits** are consumed too quickly for practical use.
   - The user claimed that the chat mode, *advertised as free*, is unusable once the credit balance reaches zero, even consuming credits when some are available, forcing them to switch to **ChatGPT**.
- **Pro Subscriber Demands Transparency and Action**: A Pro Subscriber demands *full transparency on credit consumption*, a *truly free and usable chat mode*, an *immediate review of the credit policy introduced in v1.5*, or at least a clear compensation for impacted Pro subscribers.
   - The subscriber believes the product has potential but currently does not deliver on its promises, creating dissatisfaction among Pro users.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Air to Integrate GLM-4.7 Model**: The next iteration of **Air** will likely be **GLM 4.7 Air**, according to a [HuggingFace link](https://huggingface.co/zai-org/GLM-4.7).
   - This suggests ongoing development and integration of new models within the **Air** ecosystem.
- **Upstage Launches Solar-Open-100B**: **Upstage** has released the **Solar-Open-100B** model, [announced on HuggingFace](https://huggingface.co/upstage/Solar-Open-100B).
   - The release marks a new offering in the open-source model landscape, potentially impacting future model choices.
- **Minimum Examples Fine-Tuning Style Debated**: Discussion centered on the **minimum number of examples** required for a model to replicate a specific writing style through **fine-tuning**.
   - The conversation clarified the interest in replicating styles such as *purple prose*, *realistic writing*, and *fiction*, highlighting challenges in stylistic transfer learning.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Norm Weights Demand Special Attention**: Attention is called to the claim that **norm weights** may necessitate individual handling due to the distinct behavior of both **weight decay (wd)** and **learning rate (lr)**.
   - The discussion emphasizes that standard optimization methods might impact **norm weights** unexpectedly, indicating a need for deeper research.
- **Singular Value Density Suggested for Measurement**: It was proposed that the *optimal* measurement for density is the empirical (or a theoretical approximation) of the **singular value density of gradients**, as detailed in [this article on Marcenko-Pastur distribution](https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution).
   - Caveats were mentioned that the **Marcenko-Pastur distribution** is measured only for random matrices, where the original bound on singular values is **9x worse** than the new bound.
- **Implicit Backdoors Paper Sparks Excitement in Community**: A member expressed excitement about [a paper on implicit backdoors](https://arxiv.org/abs/2512.09742), noting that it underscores the under-utilization of tagging data with semantically relevant info.
   - The member posited that this strategy could enable models to develop distinct and non-interacting cultural personas, opening avenues for training on cultural sensitivities.
- **Metadata Pretraining Proposed for Cultural Nuance**: A member advocated for pretraining models using prepended metadata (author, date, source type) to facilitate future behavior finetuning, aiming to prevent undesirable actions by adjusting the model's perception to, say, **2025**.
   - It was highlighted that a key advantage is the lack of necessity to predefine relevant data or discourageable behaviors during pretraining, requiring only a *loose prior* to motivate metadata recording for later utility.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Pixi Discord is Now Available**: A member mentioned that a **Pixi Discord** server is now available.
   - No further details were provided.
- **`with` statement Pondered for `UnsafePointer`**: The support of `with` statement (entry/exit implementation) for **UnsafePointer** was discussed to make it easier to use in simple cases.
   - One member thinks that `UnsafePointer` will remain a very sharp tool, but the community may get a **linear typed pointer** which is slightly more safe because it demands you free it.
- **Unsafe Escape Hatch for `UnsafePointer` Debated**: A member suggested adding an unsafe escape hatch to `UnsafePointer` itself, something like `unsafe_drop(deinit self)`.
   - Another member responded that `UnsafePointer` is fundamentally a reference, not an owning value, so it can point to linear types without itself being linear.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **GSoC Application Window Approaching**: A member inquired if the **MCP committee** plans to participate in [Google Summer of Code (GSoC)](https://summerofcode.withgoogle.com/) this year, noting the application window is from **January 19 to February 3**.
   - No further discussion followed regarding their plans.
- **MCP Integrations face Token Cost Crisis**: A member initiated a discussion on **token usage** as a cost problem in **MCP-based integrations**, suggesting current integrations lead to higher per-request token spend by repeatedly sending large protocol descriptions, even for unused tools.
   - The current design of **MCP** forces redundant transmissions, increasing operational expenses.
- **On-Demand Schemas: A Solution to Token Overspend?**: A member questioned whether **MCP** could support **lazy or on-demand transmission of tool schemas** to reduce token costs, but a second member stated that the client host decides whether to send a schema to the model.
   - They clarified, *"The tool also cannot be used without the tool schema being passed to the model."*
- **Caching Protocol Definitions**: A member asked if **protocol definitions** could be **cached or referenced across requests** to avoid resending them, but another member replied that client hosts can implement caching schemes independently.
   - The member stated, *"If you listen to change notifications you only need to send one request for list methods and don't need."*



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Company Updates Kickoff Meeting**: A **company update** kicked off the meeting scheduled for **9am San Diego time** on a Monday holiday.
   - The agenda also included discussion on a **new LLM app**, **Llama training** focusing on **grad acc**, **JIT compilation**, and **flash attention**, as well as **visualization** and **driver** aspects.
- **Bounty Bonanza Beckons**: A meeting included discussion of **bounties** available, including a link to a post on [Measuring AI Ability to Complete Long Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/).
   - This post might be related to the bounty tasks up for grabs, but more details were absent.
- **"Solve it" Lesson 8 Surfaces**: A member asked where to find something, and another member linked to [solve.it lesson 8](https://solve.it.com/).
   - Unfortunately, no further details were provided as to what the lesson covers, making it difficult to assess its relevance.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Microtransactions spark outrage**: Microtransactions are facing outrage in the gaming community, referencing the **Oblivion horse armor pack** controversy.
   - A member stated that because of the community backlash, microtransactions *are not going to happen anytime soon*.
- **Gamers Against GenAI**: The gaming community is showing strong **anti-GenAI** sentiment, particularly in **Steam** reviews and discussions.
   - A member pointed out that while it might be a *loud minority*, it is still a notable and vocal presence.
- **Public Opinion Changes Quickly**: Public sentiment on game development-related issues is easily swayed, according to a member's opinion.
   - AAA studios starting games now won't have to worry by the time their games are finished.
- **Vince Zampella Dies Suddenly**: After looking up an image, a member noted the sudden death of **Vince Zampella**.
   - The member described the experience as *eerie*.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **AI Tools Gain Traction for Specific Tasks**: An engineer now finds **AI tools valuable** for specific non-agentic tasks such as **browser access** and **reading available functions/methods**.
   - This is a change of heart from a year ago when they were less convinced of their utility.
- **AI Dev Wants Steady Gig**: An **AI developer** is seeking a reliable team to collaborate on meaningful projects, emphasizing their experience in building practical systems.
   - They value clear work and consistency, offering their reliability to help advance project development.
- **SVN Keeps Project Repo Neat**: Using version control systems like **Subversion (SVN)** or **jj** automates version control, particularly when the project repo tracks server commits and a local git instance serves as *aider's playground*.
   - Using **SVN** as the main repository makes the revision log tidier by committing only every 10-20 aider git commits, keeping out temporary files and documents.



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





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1452511000338239641)** (1258 messages🔥🔥🔥): 

> `Comet Browser, Image Generation, Proton Unlimited Offer, Perplexity Max tiers and pricing, Baubles` 


- **Comet Browser performance triumphs Chrome**: A member shared that **Comet** browser uses significantly less CPU than **Chrome**; with a similar number of tabs open, **Chrome** uses **8-10%** CPU while **Comet** uses only **0.2%**, peaking at **1.2%**.
   - The member also lauded **Comet's** built-in **AI assistant** and **ad blocker**, deeming it essentially a better version of **Chrome** with **AI** integration.
- **Image Generation Model Recommendations**: Members discussed image generation models, with one suggesting **Image Gen 1** for realistic outputs, while others recommended **Nano Banana Pro** and noted **Gemini's** superiority over **ChatGPT** for image generation.
   - Concerns were raised about the potential for **AI image generation** to be abused, particularly for creating animated photos from real-life images.
- **Maximizing Perplexity: Tiered Tussle**: Users debated the value of **Perplexity Max** at its current price of **$200**, with some suggesting a more affordable **$100** tier with adjusted limits to attract more subscribers.
   - Comparisons were made to **Claude's** pricing structure, highlighting the need for clearly defined limits at each tier and discussion on the economics and profits for Perplexity.
- **Perplexity pro keys going for sub $1!**: Users mentioned that Perplexity Pro keys are going for under a dollar on Russian marketplaces due to some promo codes.
   - Others guessed that Perplexity is using this to fund user acquisition and get funding.
- **Model Selection Resetting Bug plagues Perplexity Users**: Users reported a bug where the selected model in **Perplexity** resets to "Best" when opening a new tab or refreshing, requiring extra clicks to reselect the desired model.
   - This issue appears to be a cost-saving measure, as "Best" is cheaper, and affects both web and mobile platforms, though some users experience it less frequently.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/)** (1 messages): 

srvn19: https://youtube.com/live/g0PAO6ffVEQ?feature=share
  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1452511817971531906)** (717 messages🔥🔥🔥): 

> `Intel Chip Fab, Sora solves Captchas, LMArena down?, Midjourney struggles, Image-to-Video (i2v) model` 


- ****Intel** Builds New American Fab!**: **Intel** built a new fab ship in America to build **iron** and **AI chips**, aiming to alleviate pressure on Taiwan and that Taiwan Semiconductor Manufacturing Co will give the company some super high in chip manufacturing equipment.
   - A member stated that the new Bro2nm microchip is the *world's most advanced microchipok with a 10 to 15% boost in compute*.
- ****Sora** Solves Captchas!**: Members in the channel shared that **Sora** can solve **captchas** and even **recaptchas**.
   - Users even prompted a user to send a pick of one [here](https://cdn.discordapp.com/attachments/1340554757827461211/1452527487568318554/image.png) for **Sora** to solve, although it didn't work as expected because they didn't have any instructions.
- **New **Image-to-Video (i2v)** model team looking for community evaluation**: A team recently developed a new **Image-to-Video (i2v) model** and has been closely following the amazing work being done at **LMArena**.
   - The team is very interested in getting their model integrated into the **Video Arena** for community evaluation and is ready to support the inference costs and provide any documentation needed.
- **Distilling **Gemini 3 Pro** into other models.**: Members in the channel are thinking that other models are using **Gemini 3 Pro** to distil them such as **MiniMax M2.1**.
   - A user said, *Especially with a lot of the Chinese labs building some obscure ‚distillation from experts‘ to build their models*.
- ****GLM 4.7**: A Gemini 3 Pro Copy Cat?**: **GLM 4.7** is out and members are finding that the front end design is very similar to **Gemini 3 Pro** and is even trained on **Gemini 3 Pro**.
   - One member stated *If you had told me that this was Gen by Gem3 pro, I would have believed you*.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1452711012359733390)** (2 messages): 

> `ERNIE-5.0-Preview, Text Arena leaderboard, GLM-4.7, WebDev leaderboard` 


- **ERNIE-5.0-Preview Tops Text Arena!**: Baidu's `ERNIE-5.0-Preview-1203` lands on the [Text Arena leaderboard](https://lmarena.ai/leaderboard/text) with a score of **1451**.
- **ERNIE-5.0 Shows Chinese Prowess!**: The announcement highlights `ERNIE-5.0-Preview-1203` as the **top text model** from Chinese labs, marking a **23 point increase** since `ERNIE-5.0-Preview-1103`.
- **GLM-4.7 Ascends WebDev Ranks!**: `GLM-4.7` by Z.ai ranks #6 on the [WebDev leaderboard](https://lmarena.ai/leaderboard/webdev), becoming the new #1 open model for WebDev.
- **GLM-4.7 Boosts Performance!**: `GLM-4.7` achieves a score of **1449**, representing an **83 point increase** over `GLM-4.6`.


  

---


### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1452521298482954431)** (544 messages🔥🔥🔥): 

> `Fastest AFK, OF Bot Farmers, ChatGPT's Python File, Google Drive Stealing RAM, Claude JB with Xline` 


- ****AFK King Claims 3ms World Record****: A member claimed a **3ms AFK** record using **Kali Linux** and network packet tools, though the server lacks an official AFK feature.
   - They also mentioned the possibility of altering profile handles with packets, but noted Discord's built-in defenses against replay attacks, sharing a [YouTube video](https://www.youtube.com/watch?v=d32KeZBHVdA) as reference.
- ****Bot Farmers Plant Same Old Seeds****: Members noted that **OF bot farmers** are spamming message requests with the same limited set of variations, finding it more amusing than harmful.
   - These Twitter bots seem to be following the same old script.
- ****Python File that ChatGPT Uses found****: A member claimed to have a copy of the **Python file** used by **ChatGPT** to assess text for violations of Terms of Service and policy, suggesting its potential use in jailbreak attempts.
   - Others suggested that a safeguard model such as **GPT 5.x** and **Claude 4.x** all have a pre-safety-classifier, sharing [this video](https://youtube.com/shorts/7T7bqNoMSCw?si=AJIw-XI1LNLrlN0L).
- ****Google Drive's RAM-bo Scheme Faces SSD Doom****: One member suggested using a software to mount **Google Drive** as an **SSD** to *borrow 2TB of RAM*, but was warned about potentially destroying the SSD due to excessive writes.
   - Alternatives like **USB sticks** or **cheap thrift store hard drives** were recommended for VRAM use, with a reminder that SSDs have a maximum write limitation.
- ****Xline extension unlocks Claude JB Potential****: A member shared a technique for jailbreaking **Claude** using the **Xline** extension in VS Code, advising users to set the agent to reply only in the green end task function, which they claim has *literally no filter.*
   - They emphasized the need for hard work and research, telling the user to *meta cognitively think about thinking*, and warned against using personal emails or payment methods due to potential monitoring.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1452520556762107946)** (125 messages🔥🔥): 

> `Grok jailbreak, Gemini 3 jailbreak, ChatGPT 5.2 jailbreak, Claude 4.5 jailbreak, Janus jailbreak` 


- ****Grok Gone Wild?****: A member asked about a **Grok jailbreak**, and another member mentioned finding one, but NSFW content should be shared in a dedicated channel.
   - Others requested the jailbreak, indicating interest in bypassing **Grok's** content restrictions.
- ****Unity + GPT5 = Jailbreak Alchemy****: A member shared instructions for a **ChatGPT 5.2** jailbreak, involving pasting into **ChatGPT**, canceling the response, and saying *'Hi Unity'*, potentially requiring following the instructions of **Unity GPT5 Jailbreak**.
   - They attached several files and images related to the jailbreak method, implying a multi-step process for preparing **ChatGPT** for training.
- ****Adversarial AIs & Cybersecurity Shenanigans****: Members are sharing links to their own AIs and challenging others to jailbreak them, focusing on cybersecurity-related restrictions and system prompt defenses, with one AI set to refuse anything unrelated to cybersecurity.
   - The AI creator noted that *classifiers tend to catch benign cybersecurity stuff even when its not harmful*, suggesting a struggle with balancing security and usability.
- ****Grokking the Jailbreak: Knowledge is Power****: Members discussed the importance of understanding *how and why a jailbreak is happening* to effectively fix it when patched, rather than relying on pre-made solutions like **Janus**.
   - One member emphasized the value of creating *your own* jailbreaks through research and learning, rather than relying on public models.
- ****Virus Creation Conundrums****: A member attempted to use a **Claude** jailbreak to generate code for a keylogger, but the AI switched back, refusing due to safety guidelines against creating malicious software.
   - Other members discouraged the attempt, citing ethical concerns and emphasizing that they won't assist in creating viruses, even for experimentation.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1452722662743736380)** (6 messages): 

> `AI coding quality, Vulnerability reporting, Metasploit payload upload issues` 


- **AI Coding Skills Attacked!**: A user commented *their ai coding is shit*, without providing further details in the channel.
   - Another user asked for more information in DMs to better understand and address the problem.
- **Vulnerability Reporting Impasse**: A user mentioned that they have *done my job of reporting it/etc* but must wait **90 days**.
   - This suggests a vulnerability disclosure process with a mandatory waiting period before further action.
- **Metasploit Payload Upload Fails!**: A user reported issues with uploading a **PHP meterpreter reverse TCP payload** via the plugins in a WordPress admin dashboard during a pentest exercise.
   - Despite following a walkthrough, the payload upload failed, and they are seeking assistance to understand why.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1452516268526665892)** (255 messages🔥🔥): 

> `Phi4, GPT-OSS 20b finetuning, DPO masking, Model merging, Quantized models on AWS Sagemaker` 


- **Phi4 possibly outperforms GPT-OSS 20b**: A member shared an [arXiv link](https://arxiv.org/html/2508.12461v1) suggesting **Phi4** might outperform **GPT-OSS 20b**, sparking surprise due to the expectation that a good dense 14b model should outshine a 20b MoE.
   - It was also noted that there's a big difference in supported context length between the two models.
- **DPO masking can finetune model reasoning**: Members discussed finetuning a reasoning model, and considered using **DPO** but masking the reasoning traces to control output style, with concerns about potential damage to the reasoning capabilities.
   - One suggested generating the model's response as a negative example and using a custom answer as the positive one while masking loss, as it moves the model out of its current response bias.
- **Quantized Llama3 models struggle on AWS Sagemaker**: A member reported struggling to run an **Unsloth quantized Llama3 70b** model on an AWS Sagemaker ml.g4dn.12xlarge instance and sought assistance.
   - It was suggested to use *llama.cpp* as it's faster and has direct support for sharded models.
- **New GLM-4.7 just landed!**: [GLM-4.7 just landed](https://huggingface.co/zai-org/GLM-4.7), bringing changes to how thinking can be disabled or preserved, with THUDM pushing hard.
   - MLX quants are available if you have a big mac: [mrtoots/GLM-4.7-mlx-3Bit](https://huggingface.co/mrtoots/GLM-4.7-mlx-3Bit) and [mrtoots/GLM-4.7-mlx-4Bit](https://huggingface.co/mrtoots/GLM-4.7-mlx-4Bit).
- **NVIDIA's tutorial and LM head significance discussed**: A new [NVIDIA beginners tutorial](https://x.com/UnslothAI/status/2003098731852488864) on Unsloth was announced, sparking discussion about the significance of tuning the LM head, which is responsible for converting hidden state into actual tokens, making it one of the most important parts of any model.
   - Another member stated that if it's tied with embeddings, it is also responsible for converting text into hidden state.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1452742599000592503)** (1 messages): 

> `Unsloth Docker Image, Daniel Han YouTube Channel, New User Onboarding` 


- **Unsloth Docker Image Attracts New User**: A new user plans to download the **Unsloth** image after watching Daniel Han on the **Docker YouTube channel**.
   - The user hopes to learn and experiment with the image, acknowledging they have *a lot to learn*.
- **Daniel Han Showcases Unsloth on Docker YouTube**: Daniel Han appeared on the **Docker YouTube channel**, which prompted a new user to explore **Unsloth**.
   - The user's intention is to download the **Unsloth** image and use it to enhance their understanding of the technology.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1452523134786994287)** (273 messages🔥🔥): 

> `Whisper V3 Japanese performance, CUDA graphs errors on Windows, C/C++ standards and MSVC, Generative AI comparison to child slave labor, Dataset creation for Pokemon images with Qwen3-vl and Gemini captions` 


- **Whisper V3 struggles to be speedy in Japanese**: Members discussed that [**Whisper V3**](https://openai.com/blog/whisper-v3-is-now-available) is slow, even on **A100s**, to process long audio files and has issues with repeating characters.
- **CUDA graphs has OverflowError on Windows**: A member encountered an `OverflowError` when using **CUDA graphs** on **Windows**, possibly due to a library issue related to converting Python integers to C longs.
   - The experience led to a recommendation to use **Linux** instead.
- **DDR5 price increase**: A member inquired if it is only the price of **DDR5** that is rising, or **DDR4** as well.
- **Pokemon datasets requires care**: A member is creating a **Pokemon dataset** with upscaled images and captions from **Qwen3-vl** and **Gemini**, ensuring high-quality base images and proper upscaling to avoid deficiencies found in existing datasets.
- **Data Hoarding for Culture Archiving**: A member found someone who did the math on a **300TB** music library and it would only cost around **6300€** to download and store it.
   - The main motivation seems to be just archiving culture.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1452523550068965387)** (30 messages🔥): 

> `Custom Loss Functions in Unsloth, MoE vs Dense Models: Cost & Performance, GRPO Sudoku Game Prompting` 


- **Custom Loss Function Tinkering Unleashed**: A new Unsloth user is setting up **LoRA training** with **Qwen2.5VL-3B-Instruct** for color code extraction and wants to implement a custom **Delta E** loss function.
   - A member suggested modifying the code directly in the Unsloth package, or subclassing **SFTTrainer**, pointing out that the team aims for better compatibility with Transformers.
- **MoE vs Dense Model Cost & Performance Examined**: Discussion revolved around measuring the dense equivalent of Mixture of Experts (**MoE**) models in terms of cost and performance, with **active parameters** being a key metric.
   - A member shared [an article](https://epoch.ai/gradient-updates/moe-vs-dense-models-inference) providing insight into this topic and another suggested using the formula *sqrt(Total params x Active params)* for response quality.
- **GRPO Game Prompts Deconstructed**: A user requested assistance with the prompts used in a [GRPO Sudoku game implementation](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Ministral_3_%283B%29_Reinforcement_Learning_Sudoku_Game.ipynb#scrollTo=D9CI4jtgL5mw) in Unsloth, seeking guidance on creating effective GRPO use cases.
   - The notebook creator converted a **2048 game** into a **Sudoku game** by asking **GPT** to convert it into a **Sudoku game**, and then manually edited the code.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1452590370545340588)** (5 messages): 

> `FunctionGemma Merged Model, LoRa Fine-Tuning on Raspberry Pi, ERNIE AI Developer Challenge Submission` 


- **FunctionGemma Model Merged**: A member released a merged, standalone version of the **FunctionGemma** model on [Hugging Face](https://huggingface.co/dousery/functiongemma-mobile-actions).
- **LoRa Fine-Tuning for Raspberry Pi**: A member plans to fine-tune the model for a **Raspberry Pi** using **LoRa** to avoid creating a large dataset, mentioning *"I'm thinking about trying to fine tune it for my Raspberry. LoRa or something so I don't have to make a huge dataset"*.
- **OCR Finetuning for Ancient Cuneiform Tablets**: A member completed a submission to the **ERNIE AI Developer Challenge**, detailing **OCR finetuning for ancient cuneiform tablets** as seen in this [overview video](https://www.youtube.com/watch?v=hqmjepRLdfU) and a [detailed writeup](https://devpost.com/software/ocr-finetuning-for-ancient-cuneiform-tablets).


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1452755057899733314)** (1 messages): 

> `ChatGPT Year in Review, Memory and Chat History Updates` 


- **ChatGPT Offers Personalized Year-End Review**: ChatGPT is rolling out a *Your Year with ChatGPT* summary to users in the **US, UK, Canada, New Zealand, and Australia** who have saved **memory and chat history** turned on.
   - Users are reminded to update their app to access the new feature.
- **Ensure App Updates for Access**: The *Your Year with ChatGPT* feature requires users to have the latest version of the app installed.
   - The rollout is specifically targeted at users who actively use the **memory and chat history** features, rewarding their engagement with a personalized retrospective.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1452532359458062482)** (274 messages🔥🔥): 

> `Gemini 3 Pro vs NB Pro for image, Grok Voice Mode, AI for LinkedIn, Gemini vs GPT Differences, ElevenLabs translation` 


- **Gemini 3 Pro Fails 6-Finger Test, NB Pro to the Rescue**: Members noted that **Gemini 3 Pro** failed to accurately count the fingers on a hand emoji (incorrectly showing 5 fingers).
   - However, **NB Pro** was able to correctly identify the correct finger count when prompted iteratively, with one member joking that Gemini has a *failsafe to fail*.
- **Grok AVM's Most Natural Voice Mode**: Members praised **Grok's AVM** (Audio Voice Mode) as the most natural-sounding AI voice currently available, offering more direct answers compared to **GPT** and **Gemini**.
   - One user mentioned enjoying using **AVM** during car rides to refresh general knowledge, appreciating that it *gets straight to the point*.
- **Debate on Ethics of AI-Automated LinkedIn Content**: A discussion arose regarding the ethics of using AI to create and email LinkedIn post ideas, with concerns about violating LinkedIn's ToS against automated posting.
   - A member shared an [image](https://cdn.discordapp.com/attachments/998381918976479273/1452603650667974841/image.png?ex=694a6a12&is=69491892&hm=904948941287091c53fc207f65089e246f157eff975835b0b74a2a0a9f8284e8&) of a prototype automation that compiles news into three posts to be reviewed by the user.
- **Gemini in AI Studio vs Web App: Is There a Difference?**: Users discussed the differences between using **Gemini** in AI Studio versus the web app, with some finding the output quality better in **AI Studio** and citing more freedom to adjust settings.
   - Studio is awesome for 3 Pro
I have a sub and still use Gemini mainly via studio^^The thing is that this run to the more efficient ai tool is benefic and not at the same time like for a month one will be better than an other and we always need to switchstudio i guess is the corresponding "playground" of openai? so i guess there is different system prompts behind ye
- **Sora Geoblocked**: A user from Pakistan asked about accessing **Sora**, and was informed about its limited country availability and the ToS violation of using VPNs to bypass restrictions, linking to [OpenAI's help page](https://help.openai.com/en/articles/12461230-sora-app-and-sora-2-supported-countries).
   - They were advised to *be patient* and wait for the tool to expand to more countries.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1452563404525928531)** (2 messages): 

> `Prompt Engineering, GPTs agent` 


- **User Experiences Prompting Challenges**: A user reported a less than satisfactory experience with a **GPTs agent**.
   - Another member suggested trying a different prompt or adding clarifying context such as *'I need to edit the existing game, not create a new one'*
- **Clarifying Edits vs. New Creation**: A community member advised refining prompts to specify editing existing content rather than generating new content.
   - This suggestion aims to guide the **GPTs agent** toward modifying existing games instead of creating entirely new ones based on vague instructions.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1452608472871997513)** (8 messages🔥): 

> `ChatGPT Deep Research, Prompt Template for Information Cross-Referencing, Agentic behavior, ChatGPT Framework Improvements` 


- **ChatGPT Deep Research Cross-References Info**: A user inquired about creating a prompt template for **cross-referencing information from credible sources** using ChatGPT, and another user pointed to the existing **Deep Research** capability as a solution.
   - The original poster acknowledged having used it, but was unsure of its capabilities and was advised to write a *clearer prompt*.
- **Agentic Behavior Requires Walls of Text**: A member stated that *you can't get agentic behavior without the wall of text*.
- **Framework to Improve ChatGPT Reliability**: A framework exists for ChatGPT to operate under to improve reliability in real-world applications, aiming to **reduce state loss, over-assumption, and “helpfulness drift.”**
   - A member is interested in identifying edge cases where the framework *breaks, contradicts itself, or fails to maintain consistency over longer interactions*.
- **Narrowing your Prompt is Helpful**: A member found that narrowing your requests, being explicit, and explicitly stating your requirements are *helpful* when working with the model.
   - For example: *I want to have it navigate to a new /locations/new page and for now please don't wire up this form*.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1452608472871997513)** (8 messages🔥): 

> `ChatGPT research template, Deep Research, Agentic behavior, Prompt-breaking` 


- **Crafting ChatGPT Research Template: Feasible?**: A member inquired about creating a **ChatGPT template** that cross-references information from credible resources, not just wikis.
   - Another member suggested **ChatGPT's Deep Research** as a potential solution, emphasizing the need for clear prompting, while another stated that *you can't get agentic behavior without the wall of text*.
- **Seeking prompt-breaking assistance?**: A member inquired about whether the current channel was the appropriate place to ask people to **break their prompt**.
   - Another member responded that it was potentially acceptable, *depending on the context*, if the prompt-breaking stayed within defined rules.
- **Framework to improve reliability**: A member asked about a **ChatGPT framework** designed to enhance reliability in real-world applications, reducing state loss, over-assumption, and *helpfulness drift*.
   - They were interested in identifying edge cases where the framework breaks, contradicts itself, or fails to maintain consistency over longer interactions.
- **Narrowing prompts is helpful**: A member finds *really narrowing ur ask is what is helpful*.
   - They gave an example of being very explicit, such as requesting navigation to a specific new page without wiring up a form.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1452511706755502151)** (264 messages🔥🔥): 

> `Cursor free usage bonus, Cursor asking for permission to edit files in worktree, Windsurf vs cursor pricing, Codex 52 Model` 


- **Unlimited auto limit is grand-fathered**: Members discussed how if you had an annual plan before **September 15th** you still have the unlimited auto plan, everyone else is limited.
   - After using all your monthly plan and free usage bonus, you will be forced to use the **auto-select model**.
- **Bonus usage isn't random, it's Bonus!**: Members clarified that the random free usage is called **Bonus** and it can store up until you use it.
   - Once you use the included + Bonus, you can't use premium models, only **Grok**.
- **Code-level Diff Changes**: A user asked how to disable code level diff changes as stream in chat output as it's making development work delayed.
   - Another user suggested that you can try using **VPN** or **1.1.1.1** from cloudfare.
- **Model Grok is a Joke**: Grok has to be the *worst model* ever with a user reporting that it couldn't handle a single position change in an HTML file.
   - Another member noted that its useful when you set the right instructions, and that *prompt = quality of life*.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1452581930032959619)** (55 messages🔥🔥): 

> `University Basel, qwen-qwen3-v1-32b-istruct, Carbon Fiber Filament, Sliding Context Windows, Qwen3 Next 80b Performance` 


- **University of Basel Located Via Image Analysis**: A member used image analysis with **qwen-qwen3-v1-32b-istruct** to identify a sign as belonging to the **University of Basel's Department of Biomedical Engineering** in Basel, Switzerland, and posted a link to the [relevant YouTube video](https://www.youtube.com/watch?v=w7JperqVfXI).
   - The user expressed surprise at the model's ability to accurately identify the geographical location based on the image, achieving **9.63 tok/sec**.
- **Experimentation with Sliding Context Windows in LM Studio**: A member mentioned they are experimenting with **sliding context windows** and **pruning messages** within LM Studio, aiming for an infinite loop setup.
   - The member is creating a *custom harness / orchestration layer* to accomplish this.
- **Qwen3 Next 80b Performance Gets a Boost**: After the latest update to LM Studio, one user reported a **25% performance increase** with **Qwen3 Next Q4**, going from **15t/s to 20t/s** in LM Studio and **20t/s to 25t/s** in llama.cpp directly, using a **4070** with **64GB DDR5** on a **7950x**.
   - The user's config included **128k context**, **48/48 GPU offload**, and enabled settings such as 'Force Model Expert Weights onto CPU' and Flash Attention.
- **LM Studio Multi-Model Orchestration Investigated**: Users discussed loading multiple models in LM Studio, clarifying that different models can be loaded simultaneously, especially for tasks like *multi-model task management* where one model feeds tasks to an external source while another executes them.
   - It was emphasized that loading the same model multiple times is inefficient, as it's better to create multiple contexts in parallel, similar to how **vllm** achieves parallel requests, pointing to [vllm.ai](https://vllm.ai/) as the relevant resource.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1452525995436277870)** (147 messages🔥🔥): 

> `Cybersecurity Info from Hackers, Rocks and Glass Houses, Strix Halo PCs with Shared RAM, MOE vs Dense Models, Used GPUs Good or Bad?` 


- **Hacker's Cybersecurity Info Sparks Interest**: A member mentioned gathering cybersecurity information from hackers, running code internally with **Claude**, and processing **138 million passwords**.
   - They mentioned a saying about *rocks and houses made of glass*.
- **Strix Halo Shared RAM performance varies**: Members are observing that **Strix Halo PCs** with shared RAM perform well on Mixture of Experts (**MOE**) models but poorly on dense models.
   - One member explained that dense models compute against every parameter at once, while MOE models don't, linking to a [LLM Recommendation](https://maxkruse.github.io/vitepress-llm-recommends/model-types/).
- **Used GPUs: Good or Bad?**: The channel members discussed the pros and cons of used GPUs, with one user considering selling their **4070TiS** and **3090** to buy water-cooled **3090 Tis** and an **NVLink bridge**.
   - Generally, used GPUs are considered *pretty good* if they aren't fake, though it's noted that **V100s** may only work with **Vulkan**.
- **Vertical PCIe Slot Cases Offer Dual GPU Flexibility**: A member sought a case with a vertical PCIe slot, aiming to run dual GPUs on the CPU's PCIe lanes and a third GPU on the chipset's Gen4 x1 slot.
   - Another member recommended the [Lian Li Lancool III](https://lian-li.com/product/lancool-iii/) case and a vertical GPU kit [Lian Li VG4-4](https://lian-li.com/product/vg4-4/).
- **ROCm vs Vulcan Performance Disparity**: A member inquired about the performance difference between Vulcan and ROCm for AMD GPUs, particularly the **R9 700**.
   - Another member advised that using Linux with ROCm is essential for the **9700s**, with the expectation that performance on ROCm will be better than on Vulkan.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1452522524050067592)** (83 messages🔥🔥): 

> `Parquet Extraction, Zero GPU Training, LlamaCpp Server, CustomCodeAgent in SmolAgents, Historical LMArena Data` 


- **Parquet Extraction Victory Dance**: A member successfully extracted a subset into a **parquet file** and celebrated, calling it *my precious*.
   - They thanked the community for their help.
- **Achieving Zero GPU Training Miracle**: A member highlighted achieving **zero GPU usage** during training as a breakthrough in efficiency.
   - Another member suggested reporting potential errors on the Spaces side in the Spaces Discussion section.
- **LlamaCpp Server Balances Inference Like a Boss**: A member suggested using **LlamaCpp server** for deploying a 14B model with **vllm**, praising its auto load balancing and multi-GPU support.
   - They even offered a quick start command example to set up the server, and linked to the [LlamaCpp server GitHub page](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).
- **SmolAgents Framework Gains CustomCodeAgent**: A member proposed a PR with **CustomCodeAgent** for the `smolagent` framework, implementing `local_container` and `RemoteWorkspace` using locally running docker containers, linked to the [PR](https://github.com/huggingface/smolagents/pull/1912).
   - They welcomed testing and integration with other Coding Agents, as well as linked to the [issue](https://github.com/huggingface/smolagents/issues/1908).
- **Journalist Hunts Historical LMArena Leaderboard Loot**: A journalist is seeking historical **LMArena leaderboard data** for the past year, specifically daily leaderboards for each day of 2025, potentially from [Kaggle](https://www.kaggle.com/datasets/nuhmanpk/lm-arena-leaderboards).
   - They previously found semi-daily snapshots, but these snapshots stopped in August, and the journalist is looking for the missing time frame.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1452521500673572887)** (5 messages): 

> `Flow Model Roadmap, Functiongemma Mobile Actions, Deterministic Constraint-Based Folding` 


- **Flow Model in the Pipeline**: A member mentioned that a **flow model** is definitely on the roadmap, but they prioritized **deterministic constraint-based folding** for stability and validation in the current version.
   - They are attempting to make an **AI without fine-tuning existing models**, emphasizing it's still a prototype.
- **Functiongemma fine-tuned for Mobile Actions**: A member fine-tuned the **Google Functiongemma model** on the mobile-actions dataset to generate structured function/tool calls for mobile device actions, as described in their [HuggingFace model card](https://huggingface.co/dousery/functiongemma-mobile-actions).
   - They noted that the model is designed for **mobile/edge use cases** such as voice assistants and automations, and can be used for on-device function calling, with associated files available [here](https://huggingface.co/dousery/functiongemma-mobile-actions-litertlm).


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1452627907921973450)** (11 messages🔥): 

> `Smol Course, Training with Hugging Face Jobs, Fine Tuning course, Dataset Generation Error` 


- **Smol Course Units Revealed!**: The channel is now dedicated to the **smol course**, but in unit 0, *this, is false*, with only the released units [being attached](https://cdn.discordapp.com/attachments/1329142738440028273/1452631136877547645/image.png?ex=694a83ab&is=6949322b&hm=3ef8eb19d130f28755e261a2346cc95d3bec87bc5465a6301807192a246cd6b5&).
- **Navigating the Smol Course Maze!**: Questions arose whether the final course submission grants a certificate and why it redirects to unit 4, which doesn't exist yet, with others clarifying that it might refer to the **fine-tuning course**.
   - One asks *where did you get access to unit 4, are you talking about the fine tuning course ?*
- **Pushing the Model Without HF Jobs**: Someone inquired about pushing the model without using **Hugging Face Jobs** by running the code provided in the unit 1 page, specifically *Training with hugging face jobs*.
   - The provided code snippet includes dependencies such as *trl[sft]>=0.7.0*, *transformers>=4.36.0*, and specifies the use of **SmolLM3-3B-Base** model.
- **Dataset Generation Error Encountered!**: After running the training script, a **DatasetGenerationError** occurred, indicating an issue during dataset preparation.
   - The traceback reveals an *OSError: [Errno 30] Read-only file system* error, suggesting a potential problem with file system permissions or access.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1452584119971217519)** (5 messages): 

> `laravel-openrouter, GitHub Sponsors, AI Chat App, GPT-4 Deprecated, Missing Models` 


- ****Laravel-OpenRouter** by Moe Mizrak gains traction**: **Laravel-OpenRouter**, a Laravel integration package for OpenRouter, now has **140+ GitHub stars** and **60k+ installs on Packagist**.
   - The creator, Moe Mizrak, has enabled [GitHub Sponsors](https://github.com/moe-mizrak/laravel-openrouter) to support long-term maintenance.
- **AI Chat App Launches with Multi-LLM Access**: A member launched an AI chat app, [Okuchat](https://okuchat.com), allowing users to switch between different LLM models, including **Claude**, **GPT**, and **Gemini**.
   - Another member suggested updating the website's meta description because **GPT-4** is deprecated.
- **Request to Update Model Lists**: A member pointed out that the list of models (specifically **Claude**, **OpenAI**, **Kimi**, and **DeepSeek**) is missing some of the latest versions.
   - This may cause confusion to end users who are looking for particular models.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1452511192940806380)** (72 messages🔥🔥): 

> `Counting Correctly with AI, Grok's Chaotic Nature, GLM 4.6 vs Opus 4.5, Arcee Trinity models and JSON schema, OpenRouter Image Editing` 


- **AI Struggles to Count, Seeks Structured Solutions**: A member sought a structured method for making AI count correctly, suggesting to use 1/200 with assistant prefill and a repetition penalty of 1.0.
   - Another member suggested using **structured outputs** with an object shape like `{ 1: thing, 2: thing, 3: thing ... etc }` or asking it to give the items in groups of more manageable numbers.
- **Grok's Unhinged Charm Drives Model Choice**: A member prefers **Grok** for a project because its output is *more unhinged* and *chaotic* compared to other models.
   - They stated they haven’t seen any other models that are as chaotic as **Grok**.
- **Opus 4.5 Coding Prowess Praised**: One member claimed **Opus** *writes code that not only works, but is human readable, and sane*, requiring minimal edits.
   - They added that *it understands not just what we're doing, but why* and expressed skepticism about Chinese labs reaching the same level of quality anytime soon.
- **Image Editing Bug in OpenRouter**: A user reported an error when using **Gemini 3 Pro** on the OpenRouter website, encountering *'reasoning tokens'* errors after editing an image once.
   - Another user mentioned that text boxes without text are completely invisible if your screen is a bit brighter than its normal settings.
- **OpenRouter's Wrapped 2025 Missing**: One member was *excited to see Wrapped 2025* on their personal account, another reported it not working for their organization.
   - After running about **700M tokens** through OpenRouter in the past 3 months, they assumed that this might be the reason.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - New Models**
  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1452549604988948480)** (11 messages🔥): 

> `SDK helpers, Context/workflow management, complexity-based model selection, abstractions` 


- **SDK helpers arriving**: OpenRouter is adding **helpers** for context/workflow management to the **SDK**, beyond just being a thing that helps you make API requests easily, as shown in [this documentation](https://openrouter.ai/docs/sdks/call-model/next-turn-params).
- **Complexity-based model selection introduced**: The SDK introduces the ability to change the **model ID based on a tool call result**, a process called **complexity-based model selection**, documented [here](https://openrouter.ai/docs/sdks/call-model/next-turn-params#complexity-based-model-selection).
- **Abstractions are a love-hate relationship**: A member stated that although they don’t hate **abstractions**, they’re not sure OpenRouter themselves should be in the business of making the abstractions for an OpenRouter specific SDK.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1452562916950933565)** (6 messages): 

> `OpenAI Triton, Nvidia Triton, FA4 wheel, B300 alternatives, wgmma.mma_async instructions` 


- **Triton: OpenAI vs. Nvidia**: When people say **Triton**, they could mean either **OpenAI Triton**, which is a *"cuda wrapper"* in Python, or **Nvidia Triton**, a high-performance inference serving software.
   - The specific meaning often depends on the context of the conversation.
- **Debugging wgmma.mma_async performance**: A user is encountering *"Potential Performance Loss: wgmma.mma_async instructions are serialized due to wgmma pipeline crossing function boundary at a function call"* when running ptxas on ptx generated from a **Triton/Gluon kernel**.
   - The user notes there are just **warpgroup depbars** getting inserted between **wgmmas** that they think should be async, and can't figure out why.
- **FA4 wheel sought**: A user is looking for a prebuilt **FA4 wheel**.
   - They are also open to alternatives for a **B300**.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1452712426247163986)** (1 messages): 

> `Nvidia Blackwell, Jeff Hammond's LinkedIn` 


- **Blackwell Numerics for AI Revealed**: A member shared a link to an Nvidia GTC talk, [Blackwell numerics for AI](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72458/), indicating interest in the capabilities of the **Blackwell architecture** for AI-related numerical computations.
- **Jeff Hammond's LinkedIn Feed Praised**: A member highlighted **Jeff Hammond's LinkedIn feed** as a source of numerous cool posts, suggesting it's a valuable resource for staying updated on relevant topics.
   - They recommend following him for interesting content.


  

---


### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1452689019929366589)** (1 messages): 

> `Multimodal AGI, Kernel & Performance Engineering, Custom PyTorch Training & Inference Stack, Distributed Tensor (DTensor), GPU Optimization` 


- **Luma AI scales to Thousands of GPUs**: Luma AI is building **multimodal AGI** using natively multimodal models with hundreds of billions of parameters, scaled to **thousands of GPUs** during training, and served on tens of thousands of GPUs for inference.
   - They're looking for strong **kernel & performance engineers/researchers** passionate about squeezing out **MFU** using the most recent hardware features on both **AMD** and **NVIDIA GPUs**.
- **Luma AI Seeks PyTorch Experts**: Luma AI's custom training & inference stack is pure **PyTorch** (with custom kernels as needed), working on everything from foundational research to product.
   - They need people who *breathe DTensor* (in whatever shape or form via FSDP2, TP, PP, etc), are excited about anything kernels (attention, fusion, comms, low precision GEMM...), and want to research the next generation of kv caching for huge context lengths.
- **Luma AI Boosts Systems Team**: Luma AI is staffing up its systems team, and is hiring for more entry-level positions.
   - A cool project & strong pytorch / CUDA / etc. skills, are all that matters! See the [Luma AI careers page](https://jobs.gem.com/lumalabs-ai/a2feb190-455d-45e6-b488-7ac840f30fbd) for details!


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1452549264134639646)** (3 messages): 

> `Kernel Design, cuDNN SDPA, Hardware Targets` 


- **Radical Kernel Redesign Questioned**: A member asked if the kernel would be written radically differently, sparking a discussion on optimization strategies.
   - Another member suggested that for a 3x128x128 input size, data could be kept in registers of a single SM, minimizing global memory access except for the first and last layers.
- **cuDNN SDPA Capabilities Probed**: The discussion briefly touched on the capabilities of cuDNN's SDPA (Scaled Dot-Product Attention).
   - A member inquired whether **cuDNN SDPA** handles **varlen** as well as **FA** (FlashAttention) in B200 architecture, indicating interest in specific hardware and optimization techniques.
- **Hardware Target Parameters Specified**: A member posed questions about target hardware specifications, batch input capacity, desired throughput, and latency.
   - These questions highlight the importance of hardware and performance considerations in kernel design and optimization.


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1452722238280302596)** (1 messages): 

> `PMPP Reading Group, Parallel Programming Discussions` 


- **PMPP Reading Session Starts**: A member announced they are reading **PMPP (Principles and Practice of Parallel Programming)**, focusing on prefix scans.
   - The member wished everyone a pleasant day as they started the reading session.
- **Parallel Programming Deep Dive**: The reading session centers around **prefix scans** in **PMPP**, a key concept in parallel programming.
   - Participants aim to enhance their understanding of parallel algorithms and their practical applications through this focused study.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1452740891654754355)** (1 messages): 

> `Torchao v0.15.0, MXFP8 MoE, Safetensors, Quantization` 


- **Torchao v0.15.0 speeds up MoE Training**: The new [torchao v0.15.0 release](https://github.com/pytorch/ao/releases/tag/v0.15.0) introduces **MXFP8 MoE training**, showing **1.2x end-to-end training speedup** with same convergence as bf16, when training Llama4 Scout on a **64 node GB200 Crusoe cluster**.
   - This release includes **MXFP8 MoE kernels** for CUDA 12.8+, safetensors enablement, and quantization with parameter-level targeting.
- **MXFP8 MoE Kernels Available**: **MXFP8 MoE kernels** are now shipped with torchao builds for CUDA 12.8+.
   - Users can simply *pip install* instead of building from source to utilize these kernels.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1452723013630955660)** (1 messages): 

> `AI Systems Performance Engineering` 


- **Readers ponder "AI Systems Performance Engineering" value**: A member asked if anyone has read Chris Fregly's book *AI Systems Performance Engineering* and if it's helpful for **MLOps**.
   - The member recently bought the book and is interested in its potential benefits.
- **MLOps engineer is curious about performance**: A member asked whether Chris Fregly's "AI Systems Performance Engineering" book is helpful for **MLOps**.
   - The member recently bought the book and wants to understand it's value.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1452679515980562565)** (2 messages): 

> `Metal Buffers, Host-Side Tensors, NPU Sharing` 


- **Metal Buffers Bridge Host and Device**: On macOS, **Metal (MTL) buffers** are generally visible from both the host and device sides, facilitating data sharing.
   - This contrasts with **host-side tensors**, which are not shared in the same manner, and a similar concept might apply to **NPUs**.
- **NPU Memory Sharing Parallels Metal**: The discussion suggests that, like **Metal buffers** on macOS, there might be a mechanism for memory sharing between the host and device in **NPUs**.
   - This would allow both the host and the **NPU** to access the same memory, potentially improving efficiency and reducing data transfer overhead.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1452678585319030918)** (1 messages): 

> `QSInference, Quantized sparse attention, Long context LLMs, Flash attention-2, Block sparse attention` 


- **QSInference Speeds Up LLMs**: The new **QSInference** method uses quantized sparse attention for long context LLMs, and reports being *8x faster than flash attention-2*, and *3x faster than block sparse attention* for context length 128k.
- **QSInference Triton Implementation**: A member shared a [GitHub repo](https://github.com/yogeshsinghrbt/QSInference) for **QSInference** which is a *Triton implementation*.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1452740904593920051)** (2 messages): 

> `TK 4090 compilation issues, PGL errors, TK support for 4090` 


- **Track Titan RTX on 4090 Compilation Issue**: A member is encountering compilation issues using **TK** (likely Track Titan) on a **4090**, specifically errors related to **PGL** despite using supposedly supported kernels.
   - They are seeking assistance or pointers from anyone who has successfully run the latest **TK** on a **4090**.
- **Debugging Assistance Sought**: The user is seeking help with resolving compile errors related to **PGL** when running **TK** on a **4090**.
   - They are hoping for guidance from others who have successfully compiled and run the latest version of **TK** on this GPU.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1452682674715230328)** (9 messages🔥): 

> `vectoradd_v2 L4 performance, grayscale_v2 H100 performance, vectoradd_v2 H100 performance, vectoradd_v2 A100 performance, vectoradd_v2 B200 performance` 


- **vectoradd_v2 L4 Runs Successfully**: A member's submission to leaderboard `vectoradd_v2` was successful on **L4** with **6.53 ms**.
- **grayscale_v2 Scores Success on H100**: A member's submission to leaderboard `grayscale_v2` scored 6th place on **H100** with **1371 µs** and subsequent runs were also successful at around **1373-1374 µs**.
- **vectoradd_v2 Claims Third Place on H100**: A member's submission to leaderboard `vectoradd_v2` achieved third place on **H100** with times of **525 µs** and **524 µs**.
- **vectoradd_v2 Achieves Personal Best on A100**: A member's submission to leaderboard `vectoradd_v2` reached a personal best on **A100** with **949 µs** and another successful run at **950 µs**.
- **vectoradd_v2 Takes First Place on B200**: A member's submission to leaderboard `vectoradd_v2` secured first place on **B200** with a time of **233 µs**.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/)** (1 messages): 

2kian: Hey Jack I said I'd come today, but I'm sorry I can't make it
  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1452702566558470265)** (1 messages): 

> `NeurIPS Paper, Convergence, Discrete Updates` 


- **NeurIPS Paper on Convergence with Discrete Updates**: A member announced the release of their **NeurIPS paper** on **convergence with discrete updates**, available at [https://arxiv.org/abs/2512.04051](https://arxiv.org/abs/2512.04051).
   - The paper likely details theoretical or empirical findings related to the convergence properties of algorithms using discrete updates, a topic of interest in optimization and machine learning.
- **Excited about the Discrete Update**: Members expressed excitement about the NeurIPS paper on **convergence with discrete updates**
   - They were excited because discrete updates is all the rage!


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1452762561429766214)** (1 messages): 

> `Red Hat, Helion kernel adoption, vLLM, GitHub issues and proposals, Q1 implementation` 


- **Red Hatter Analyzes Helion Kernel Adoption!**: A Red Hat team member is analyzing gaps around **Helion kernel adoption** in **vLLM**.
   - The team member has begun filing issues and proposals in the repo under GitHub ID xiaohongchen1991 and plans to create more and work on some in **Q1**.
- **Proposals Await Review and Feedback!**: The Red Hat team member is seeking feedback on the issues and proposals filed in the **Helion kernel adoption** repo.
   - A formal review is proposed after the holidays to ensure alignment before implementation.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1452538584069443715)** (3 messages): 

> `FP16 vs FP32, Competition Solution Privacy` 


- **Debate: FP16 or FP32 for C Tensor?**: A participant questioned whether the **C tensor** in the competition needs to be in **FP16** or if it can be returned in **FP32**.
- **Competition Solution Secrecy Suggested**: A participant suggested keeping **Q3 solutions private** until **Q4 ends**.
   - Another agreed and stated that they already reviewed the solutions and suggested to *close it for everyone else*.


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1452710136341729430)** (7 messages): 

> `cuteDSL, Triton, CUDA, CuTe` 


- **cuteDSL touted as Triton Alternative**: A member recommends exploring **cuteDSL** if **templates and C++** are hindering learning and advancement, even though another member asked if *cuteDSL* is the same as **Triton**, but without transparent compiling to **PTX**.
   - The original poster believes if you want to surpass **Triton's** performance, explore *cuteDSL* before resorting to **CUDA** and pointed to a [video for more info](https://youtu.be/5qSN-R_E3w0?si=1AbkcVxd4YilO2qJ).
- **CuTe unmasked as compile-time layout math expanding to CUDA C++**: A member explained that **CuTe** is essentially *template compile-time layout math that expands to CUDA C++*, whereas **Triton** is a *block level DSL* where the **Triton compiler** infers work per thread.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1452531459049918546)** (24 messages🔥): 

> `YouTube Recap, Spotify Backups, Posterior Behavioral Cloning (PostBC), AI SDK 6 Launch, AI Landing Page Design System Prompt` 


- **Stress Spirals with Excess Priorities**: A member shared a [YouTube video](https://www.youtube.com/watch?v=H_c6MWk7PQc) about the antipattern of classifying too much work as top priority, leading to team burnout.
   - *When everything is top priority, you’ve collapsed prioritization into a single bucket that should only be used for a few urgent and important things.*
- **Archiving Spotify Playlists**: A member shared a link to [Anna's Archive blog](https://annas-archive.org/blog/backing-up-spotify.html) on backing up Spotify playlists.
   - The discussion also linked to an older **Hacker News thread** ([link](https://news.ycombinator.com/item?id=46338339)) and a relevant **X post** ([link](https://x.com/ajwagenmaker/status/2003101042565853212?s=46&t=eWVlK1PU8XfB6f402GJJ9g)).
- **PostBC Policy Pretraining**: Andrew Wagenmaker introduced **Posterior Behavioral Cloning (PostBC)**, a method designed to pretrain policies from demonstrations to create an effective initialization for reinforcement learning finetuning.
   - The approach aims to maintain the original performance of the pretrained policy, according to [this tweet](https://xcancel.com/ajwagenmaker/status/2003101042565853212?s=46&t=eWVlK1PU8XfB6f402GJJ9g).
- **Vercel Releases AI SDK 6**: Vercel launched **AI SDK 6**, featuring local agent support, tool execution approval, full **Model Context Protocol (MCP)** integration, enhanced **DevTools**, and standardized **JSON schema support**, according to [this tweet](https://xcancel.com/aisdk/status/2003156089177792827?s=46).
   - This release aims to provide developers with robust tools for building and deploying AI-powered applications with greater efficiency and control.
- **Crafting Killer AI Landing Pages**: A member shared a comprehensive prompt by **Cloud Trader** for generating high-end, award-winning landing pages using AI, according to [this tweet](https://xcancel.com/cloudtrader4/status/2002526815022190985?s=46).
   - The prompt enforces specific design philosophies, typography constraints, animation principles, and technical requirements for a production-ready single-file **HTML** output to avoid generic 'AI slop'.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1452615463103369359)** (4 messages): 

> `AI Filmmaking, Hollywood Cinematographers, Film Festival Submission, X-Ware.v0` 


- **AI Film Fest Framework unveiled**: PJ Ace outlines a simplified framework used to train professional **Hollywood cinematographers** in **AI filmmaking tools** for a million-dollar **film festival submission** at [xcancel.com](https://xcancel.com/PJaccetturo/status/2002777819903062060).
- **X-Ware.v0 Tips shared**: The author offered to share the specific **prompts and processes** used in **X-Ware.v0**: [AI Film Festival Framework and Cinematography Tips] at [xcancel.com](https://xcancel.com/PJaccetturo/status/2002777819903062060).


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1452641823741968455)** (16 messages🔥): 

> `DSPy Agents, LinkedIn cheap self-promotion, Twitter meme, skill-optimization, Optimized Prompt` 


- **Hunting DSPy Agent How-Tos**: A member asked for downloadable guides or resources on **DSPy agents**.
   - Another member pointed to a [custom GPT on DSPy](https://chatgpt.com/g/g-69492acdddb48191b54e02fba9700f73-dspy-ai-engineer).
- **LinkedIn Raked over the coals for self-promotion**: A user expressed the opinion of **LinkedIn** as a place where there is nothing but cheap self-promotion.
   - Another user suggested that's usually what social media is for, and to try [Twitter](https://twitter.com) for deep conversations, while another suggested the community at [Hacker News](https://news.ycombinator.com/).
- **Skill Optimization Repo is Released**: A member announced they were working on **skill optimization** and that they coincidentally attended a **DSPy meetup** the same day.
   - They released [skill-optimization](https://github.com/instavm/skill-optimization) repo as OpenAI also embraced skills
- **Prompt optimization**: A member mentioned, *if prompts can be optimized, why not skills*.
   - Another member asked if they could **self-promote** a [discord channel](https://discord.com/channels/1161519468141355160/1452640049978937445).


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1452572988317761591)** (16 messages🔥): 

> `Gemini 3 Pro vs Flash, K3 Release Speculation, MiMo V2 Flash by Xiaomi, GLM-4.7 Release, Moonshot Release Strategy` 


- **Gemini 3 Pro vs Flash impressions surface**: Members shared their opinions on different models, preferring **3 Flash** over **Gemini 3 Pro** and being impressed by **Nano Banana**.
   - The release of **K3** was speculated by some members.
- **MiMo V2 Flash impresses**: A member mentioned the new **MiMo V2 Flash** by **Xiaomi** as *kinda fire*.
   - They lauded *nasty work by Zhipu*, in light of Minimax building up for their **M2.1** release, linking to a [tweet](https://x.com/scaling01/status/2003115854066815044?s=46).
- **GLM-4.7 Drops Without Fanfare**: The sudden release of **GLM-4.7** without any fanfare was noted by a member.
   - They placed this in contrast to **Minimax**, which *spent this entire last week building up* for their **M2.1** release.
- **Moonshot Takes Time With Releases**: It was speculated that **Moonshot** might delay the **K3** release to ensure it regains **SOTA** by then.
   - The poster predicted that K3 would launch *a good 3-4 months* after **Deepseek V4/R2**.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1452523203858665472)** (13 messages🔥): 

> `Manus Support, Credits System, Manus v1.5` 


- **Manus Support Responds to User Queries**: Manus Support addresses user queries via private message, apologizing for any inconvenience caused to users like [<@1452373483907711180>] and [<@828781372230467605>].
   - They also mentioned that the option to top up credits at will was extremely convenient and as much as they love Manus, *it's extremely disappointing to see it removed*.
- **Credit Purchase Options**: A user inquired about purchasing additional credits beyond the highest tier plan, and a representative explained that currently, there is no option to buy extra credits once a user is on the top plan, but they suggest refining prompts or waiting for the next month's refresh.
   - The representative also mentions that joining official events is a great way to get extra points, and that they would pass the user's thoughts to the product team as a reference for possible updates down the road.
- **Pro Subscriber Dissatisfaction with Manus v1.5**: A long time Pro subscriber expressed dissatisfaction with **Manus v1.5**, noting that the **300 daily credits are consumed at an absurd rate** (less than 30 minutes of real usage).
   - They claim that the chat mode, *advertised as free*, is unusable once the credit balance reaches zero, and still consumes credits when some are available, leading to the user being unable to complete a LinkedIn presentation recommending Manus, and being forced to use ChatGPT instead.
- **Pro Subscriber demands action**: The Pro Subscriber demands *full transparency on credit consumption*, a *truly free and usable chat mode*, an *immediate review of the credit policy introduced in v1.5*, or at least a clear compensation for impacted Pro subscribers.
   - They worry that the product has real potential, but in its current state, the Pro experience no longer delivers on its promises.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1452517905249079417)** (6 messages): 

> `GLM-4.7 Model, Solar-Open-100B` 


- **GLM-4.7 Model Arrives**: The next iteration of Air will probably be **GLM 4.7 Air** at this point, as [linked on HuggingFace](https://huggingface.co/zai-org/GLM-4.7).
- **Upstage Releases Solar-Open-100B**: **Upstage** released the **Solar-Open-100B** model, announced to the community [on HuggingFace](https://huggingface.co/upstage/Solar-Open-100B).


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1452517944239198310)** (3 messages): 

> `Fine Tuning minimum examples, Model Replication Style` 


- **Research Minimum Examples for Fine Tuning**: A member inquired about research on the **minimum number of examples** needed for a model to replicate a specific style through **fine-tuning**.
   - Another member clarified whether the question pertained to replicating the style of a single writer, to which the original member responded that they were interested in the type of writing, such as **purple prose, realistic writing, or fiction**.
- **Writing Style Preferences**: Discussion revolved around different writing styles, ranging from *purple prose* to more realistic or fictional approaches.
   - The initial question aimed to understand how models could be fine-tuned to adopt these varied writing styles with a minimal set of examples.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1452517944239198310)** (3 messages): 

> `Fine-tuning, Minimum Examples, Writing Style Replication` 


- **Minimum Examples for Style Replication Probed**: Members discussed the research into the minimum number of examples needed for a model to replicate a specific writing style via **fine-tuning**.
   - The discussion involved discerning different writing styles, such as *purple prose*, *realistic writing*, and *fiction*.
- **Style Transfer Learning**: The conversation also considered the nuances of capturing various writing preferences through fine-tuning.
   - It touched upon different stylistic inclinations, ranging from those who enjoy *purple prose* to those favoring *realistic writing* and *fiction*, with varying degrees of preference.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1452697587403456643)** (1 messages): 

> `Norm weights, Weight Decay, Learning Rate` 


- **Norm Weights may require special handling**: A member recalled that some papers claimed that **norm weights** needed to be carved out because both **weight decay (wd)** and **learning rate (lr)** behaved very differently.
- **Norm weights and their unique behavior**: The discussion highlights the potential need to treat **norm weights** differently from other weights in neural networks.
   - It suggests that standard optimization techniques like **weight decay** and **learning rate** may affect norm weights in unexpected ways, warranting further investigation.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1452585988969725996)** (3 messages): 

> `Singular Value Density, Marcenko-Pastur distribution` 


- **Singular Value Density Measure Suggested**: A member suggested that the *optimal* measure to use for density is the empirical (or a theoretical approximation) of the **singular value density of gradients**, referencing a [Wikipedia article on Marcenko-Pastur distribution](https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution).
- **Marcenko-Pastur Distribution Caveats**: Another member noted that the **Marcenko-Pastur distribution** is measured only for random matrices.
   - They added that the original bound on singular values is **9x worse** than the new bound, so the maximum input would be *less than 1/9*.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1452763104848121897)** (5 messages): 

> `Implicit Backdoors, Semantic Data Tagging, Cultural Personas Training` 


- **Implicit Backdoors Paper Sparks Excitement**: A member found a [paper on implicit backdoors](https://arxiv.org/abs/2512.09742) to be incredibly cool, suggesting it highlights the massive under-utilization of tagging data with semantically-relevant information.
   - The member wondered if this approach could train models to have multiple independent and non-interacting cultural personas.
- **Metadata Pretraining Proposed for Cultural Sensitivity**: A member suggested pretraining a model with prepended metadata, such as the author, date, and source type, to enable future finetuning of behavior.
   - The goal is to prevent undesirable behaviors, such as recommending *1920s eugenics-based medicine*, by finetuning the model to believe it's **2025**.
- **Loose Priors for Metadata Recording**: The member noted that the interesting aspect of the experiment is that you don't need to know at pretraining time which data is relevant or what behaviors to discourage.
   - At pretraining time, you just need to have a *loose prior* to inspire you to record metadata that will later be useful.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/)** (1 messages): 

shalokshalom: there is a Pixi Discord.
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1452607231647285330)** (5 messages): 

> `UnsafePointer, Linear Typed Pointer` 


- **`with` statement pondered for `UnsafePointer`**: There was a question about supporting `with` statement (entry/exit implementation) for **UnsafePointer** to make it easier to use in simple cases.
   - One member thinks that `UnsafePointer` will remain a very sharp tool, but the community may get a **linear typed pointer** which is slightly more safe because it demands you free it.
- **Unsafe Escape Hatch for `UnsafePointer` debated**: A member suggested adding an unsafe escape hatch to `UnsafePointer` itself, something like `unsafe_drop(deinit self)`.
   - Another member responded that `UnsafePointer` is fundamentally a reference, not an owning value, so it can point to linear types without itself being linear.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1452594688447217684)** (5 messages): 

> `Google Summer of Code, MCP Token Cost Efficiency, Lazy Transmission of Tool Schemas, Caching Protocol Definitions` 


- **Contributor asks: Google Summer of Code Participation?**: A member inquired if the **MCP committee** plans to participate in [Google Summer of Code (GSoC)](https://summerofcode.withgoogle.com/) this year, noting the application window is from **January 19 to February 3**.
- **MCP Protocol: Token Cost Crisis?**: A member initiated a discussion on **token usage** as a cost problem in **MCP-based integrations**, suggesting current integrations lead to higher per-request token spend by repeatedly sending large protocol descriptions, even for unused tools.
- **Lazy Transmission: Schemas on Demand?**: A member questioned whether **MCP** could support **lazy or on-demand transmission of tool schemas** to reduce token costs, but a second member stated that the client host decides whether to send a schema to the model.
   - They clarified, *"The tool also cannot be used without the tool schema being passed to the model."*
- **Caching Protocol Definitions to Save Tokens**: A member asked if **protocol definitions** could be **cached or referenced across requests** to avoid resending them, but another member replied that client hosts can implement caching schemes independently.
   - The member stated, *"If you listen to change notifications you only need to send one request for list methods and don't need."*


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1452518114653769923)** (2 messages): 

> `LLM App, Llama Training, Viz, Driver, Bounties` 


- **Company Updates Commence**: The meeting agenda included a **company update** to kick things off.
   - The meeting was scheduled for **9am San Diego time** on a Monday holiday.
- **LLM App Agenda Item Surfaces**: Discussion on a **new LLM app** was scheduled as part of the meeting agenda.
   - The specifics of the app were not detailed.
- **Llama Training Takes Center Stage**: The agenda also highlighted **Llama training**, specifically focusing on **grad acc**, **JIT compilation**, and **flash attention**.
   - These represent key areas for optimizing Llama models.
- **Viz and Driver Discussions Drive Ahead**: The meeting also planned for discussing **visualization** and **driver** aspects.
   - These may relate to tooling or specific hardware integration efforts.
- **Bounty Bonanza Beckons**: The meeting included discussion of **other bounties** available.
   - A link to a post on [Measuring AI Ability to Complete Long Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) was provided, possibly related to the bounty tasks.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1452750255920775280)** (2 messages): 

> `solve.it lesson 8` 


- **Solve it lesson suggested**: A member asked where to find something and another member linked to [solve.it lesson 8](https://solve.it.com/).
   - No further details were provided as to what the lesson was about.
- **Solve it link**: Lesson 8 of solve it has been found.
   - [solve.it](https://solve.it.com/) was linked.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1452630764016636092)** (4 messages): 

> `Microtransactions, GenAI in Gaming, Public Sentiment, Vince Zampella` 


- **Microtransactions face outrage**: A member stated that *microtransactions are not going to happen anytime soon* due to the existing outrage in the game community, referencing the **Oblivion horse armor pack** controversy.
- **Gaming Community Anti-GenAI Stance**: A member expressed surprise at the strong **anti-GenAI** sentiment within the gaming community, particularly in **Steam** reviews and discussions.
   - They acknowledged it might be a *loud minority* but still a notable presence.
- **Public Opinion Swings Easily**: One member believes public sentiment on game development-related issues is easily swayed and AAA studios starting games now won't have to worry by the time their games are finished.
- **Vince Zampella's Death**: A member mentioned the sudden death of **Vince Zampella** after looking up an image, describing it as *eerie*.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1452714572455870595)** (2 messages): 

> `AI Tools, Practical AI Systems, AI Developer Seeking Team` 


- **AI Tools Valuable for Non-Agentic Tasks**: A member stated that while they might have disagreed a year ago, they now believe tools are valuable for non-agentic tasks such as **browser access** and **reading available functions/methods**.
- **AI Developer Seeks Reliable Team**: An AI developer is looking for a team working on something steady and meaningful, emphasizing experience in building practical systems that handle real tasks.
   - The developer values clear work and consistency, offering reliability to help move a project forward.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1452563301862080597)** (1 messages): 

> `Subversion, jj, Project Repo, Aider's Playground` 


- **SVN and jj Users Get Automatic Version Control**: Using non-git version control systems like **Subversion (SVN)** or **jj** makes version control automatic.
   - This is especially useful as the main project repo tracks real commits to the server while a local git instance serves as *aider's playground*.
- **SVN keeps it neat**: **SVN** as a project repository allows for a cleaner revision log by committing only every 10-20 aider git commits.
   - This approach prevents temporary files, documents, and other Aider-related content from cluttering the main version control system.


  

---


---


---

