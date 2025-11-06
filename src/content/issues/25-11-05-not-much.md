---
id: MjAyNS0x
title: not much happened today
date: '2025-11-05T05:44:39.731046Z'
description: >-
  **Kimi-K2 Reasoner** has been integrated into **vLLM** and will soon be
  supported by **SGLang**, featuring a massive **1.2 trillion parameter MoE**
  configuration. **Perplexity AI** released research on cloud-portable
  trillion-parameter MoE kernels optimized for **AWS EFA**, with potential
  integration into **vLLM**. **IBM's vLLM** team formalized hybrid dense and
  sparse expert models, supporting models like **Qwen3-Next**, **Nemotron Nano
  2**, and **Granite 4.0**. **Kimi-K2** reportedly scores **77% on GPQA
  Diamond**, outperforming **GPT-4.5** at 71.4%, though this is unverified. 


  **Anthropic** published a guide on efficient tool-heavy agent systems using
  MCP patterns, drastically reducing context tokens by ~98.7%. **Graphiti MCP**
  demonstrated shared memory across apps like **Claude Desktop** and **Cursor**
  for persistent agent memory. **VS Code** introduced an "Agent sessions"
  feature to unify agent management, including **Copilot** and **Codex**.
  **Cursor AI** improved coding accuracy via semantic search and code retrieval
  embeddings. New evaluation frameworks like **CodeClash** and **LMArena**
  assess agent and coding model performance in realistic multi-round tasks and
  occupation-tagged leaderboards.
companies:
  - vllm
  - perplexity-ai
  - ibm
  - anthropic
  - graphiti
  - claude
  - cursor-ai
  - microsoft
models:
  - kimi-k2
  - qwen3-next
  - nemotron-nano-2
  - granite-4.0
  - gpt-4.5
  - copilot
  - codex
topics:
  - mixture-of-experts
  - model-integration
  - cloud-computing
  - hybrid-models
  - benchmarking
  - agent-systems
  - memory-persistence
  - semantic-search
  - code-retrieval
  - context-length-optimization
  - tool-use
  - evaluation-frameworks
  - software-development
people:
  - scaling01
  - cedric_chee
  - aravsrinivas
  - omarsar0
  - _avichawla
  - pierceboggan
  - jo_parkhurst
  - jyangballin
  - ofirpress
  - ml_angelopoulos
---



**a quiet day.**

> AI News for 11/4/2025-11/5/2025. We checked 12 subreddits, 544 Twitters and 23 Discords (200 channels, and 6597 messages) for you. Estimated reading time saved (at 200wpm): 566 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Gemini 3 and GPT 5.x can't come soon enough...

---

# AI Twitter Recap

**Kimi-K2 lands in open inference stacks; Perplexity unlocks trillion-param MoE kernels**

- **Kimi-K2 Reasoner → vLLM and SGLang**: The Kimi-K2 reasoning model has been merged into vLLM, with maintainers teasing “soon” availability and a wink from the vLLM account. SGLang also plans support at launch. Discussion highlights suggest Kimi-K2’s MoE config is in the 1.2T total / ~30B active ballpark, similar to recent large sparse models. See announcements from [@scaling01](https://twitter.com/scaling01/status/1986071916541870399), [@vllm_project](https://twitter.com/vllm_project/status/1986073807816433880), and [@cedric_chee](https://twitter.com/cedric_chee/status/1986073808672067725).
- **Perplexity’s custom MoE kernels (AWS EFA)**: Perplexity released their first research paper and kernels for large MoE—claiming cloud-portable, trillion-parameter deployments (e.g., Kimi K2) viable on AWS EFA. vLLM hinted at integrating the fast comms kernels. Threads and preprints from [@perplexity_ai](https://twitter.com/perplexity_ai/status/1986101355896098836), [@AravSrinivas](https://twitter.com/AravSrinivas/status/1986106660386222592), and vLLM’s response [here](https://twitter.com/vllm_project/status/1986119917297672245).
- **Hybrid model support in vLLM v1**: IBM’s vLLM team formalized hybrid models (dense + sparse experts) as first-class in vLLM, moving beyond experimental hacks in v0. Models like Qwen3-Next, Nemotron Nano 2, Granite 4.0 are now supported. Details via [@PyTorch](https://twitter.com/PyTorch/status/1986192579835150436) and vLLM best practices from NVIDIA DGX Spark [guide](https://twitter.com/vllm_project/status/1986049283339243821) and an EU meetup stream from Red Hat/IBM/MistralAI [here](https://twitter.com/RedHat_AI/status/1985976687876522110).
- **Kimi-K2 benchmarks (claimed)**: A claim that Kimi-K2 scores 77% on GPQA Diamond (vs. GPT‑4.5 at 71.4%) circulated via [@scaling01](https://twitter.com/scaling01/status/1986112227875954967). Treat as unverified until broader evals land.

**Agent systems, MCP, and coding stacks get more “production”**

- **Anthropic’s code-execution + MCP pattern**: Anthropic published a guide on making tool-heavy agents cheaper and faster by: (1) representing MCP servers as code APIs (not raw tool schemas), (2) progressively discovering tools, and (3) doing in-environment data processing. A worked example shows cutting context from ~150k to ~2k tokens (~98.7% reduction). Strong read for anyone shipping agentic systems: summaries by [@omarsar0](https://twitter.com/omarsar0/status/1986099467914023194).
- **Shared memory across apps (Graphiti MCP)**: A practical demo shows wiring a local Graphiti MCP server to both Claude Desktop and Cursor to persist and retrieve a temporal knowledge graph as “agent memory” across tools—fully local. Setup and repo walkthrough by [@_avichawla](https://twitter.com/_avichawla/status/1985958015452020788) and [repo](https://twitter.com/_avichawla/status/1985958022053838924).
- **VS Code grows an “agent primitive”**: A new “Agent sessions” view unifies launching/monitoring agents in-editor, including Copilot and external agents (e.g., Codex). Teams are seeking feedback on terminology and UX. See [@code](https://twitter.com/code/status/1986113028387930281), [@pierceboggan](https://twitter.com/pierceboggan/status/1986116693819859024), and [@jo_parkhurst](https://twitter.com/jo_parkhurst/status/1986136483892507119).
- **Repo-scale coding accuracy = retrieval**: Cursor reports significant gains from semantic search over grep for large codebases, including training a code retrieval embedding. Write-up: [@cursor_ai](https://twitter.com/cursor_ai/status/1986124270548709620) and blog [link](https://twitter.com/cursor_ai/status/1986124272029372428).
- **Evaluations for real agent work**:
    - CodeClash pits models in “code duels” over multi-round repo evolution toward business goals (vs single tasks). Early results show current LMs struggle; thread by [@jyangballin](https://twitter.com/jyangballin/status/1986093902122942700) and [@OfirPress](https://twitter.com/OfirPress/status/1986095773843390955).
    - LMArena launched “Arena Expert” with occupation-tagged leaderboards across 8 sectors; expert prompts mined from real user traffic. Details via [@arena](https://twitter.com/arena/status/1986153162802368555) and analysis from [@ml_angelopoulos](https://twitter.com/ml_angelopoulos/status/1986154276499104186).
- Additional: OpenHands Cloud’s basic tier is now free ([thread](https://twitter.com/gneubig/status/1986071169263370711)); openenv lets you push/pull RL environments as Spaces ([@ben_burtenshaw](https://twitter.com/ben_burtenshaw/status/1986097540068950149)); Voiceflow KB metadata routing ([update](https://twitter.com/IsaacHandley/status/1985905936553398726)); Dify integrates Qdrant for RAG ([post](https://twitter.com/qdrant_engine/status/1986014287718916463)).

**Multimodal and video: subject consistency, real-time generation, controllability**

- **ByteDance BindWeave (subject-consistent video)**: New subject-consistent I2V via cross-modal integration; model card on HF and paper thread from [@_akhaliq](https://twitter.com/_akhaliq/status/1986058046876070109) with [paper](https://twitter.com/_akhaliq/status/1986058201758908548) and [model link](https://twitter.com/_akhaliq/status/1986058306331517404).
- **Real-time video gen on a single H100**: MotionStream hits ~29 FPS / 0.4s latency on one H100 with interactive motion controls ([thread](https://twitter.com/_akhaliq/status/1986054085766750630)).
- **Camera editing post-hoc**: Google’s Veo 3.1 “Camera Adjustment” supports angle/movement changes to previously generated clips; early user tests from [@TheoMediaAI](https://twitter.com/TheoMediaAI/status/1986104791454388289). Related: Qwen Image Edit Multiple Angles LoRA offers strong camera pose control ([demo](https://twitter.com/linoy_tsaban/status/1986090375409533338) and coverage from [@multimodalart](https://twitter.com/multimodalart/status/1986174924038218087)).
- **Benchmarks and tooling**: ViDoRe v3 (human-authored evals for realistic multimodal RAG and open-ended/multi-hop queries) via [@tonywu_71](https://twitter.com/tonywu_71/status/1986047154620633370); VCode reframes vision as SVG code for multimodal coding evals ([paper](https://twitter.com/_akhaliq/status/1986073575216824650), [authors](https://twitter.com/KevinQHLin/status/1986126304316411928)); MIRA tests visual chain-of-thought ([post](https://twitter.com/_akhaliq/status/1986075520962793672)).

**Research and training notes**

- **OpenAI’s IndQA**: New benchmark targeting Indian languages and everyday cultural context; part of broader efforts to evaluate non-English/local knowledge ([announcement](https://twitter.com/OpenAI/status/1985950264525013210)).
- **μP theory milestone**: Learning-rate transfer under μP is now formally proven ([@QuanquanGu](https://twitter.com/QuanquanGu/status/1985961289882165674)).
- **Introspection in LLMs (Anthropic)**: Via “concept injection,” Anthropic observes emerging, unreliable forms of mechanistic self-awareness—models detecting internal thoughts vs inputs and intent vs accident ([summary](https://twitter.com/TheTuringPost/status/1986220265253314895), original blog: transformer-circuits).
- **“AI Scientist” for autonomous discovery**: Edison Scientific’s Kosmos runs 200 agent rollouts per objective, ~42k lines of code executed and ~1.5k papers read per run; 7 externally validated discoveries across metabolomics, materials, neuroscience, and genetics reported ([@andrewwhite01](https://twitter.com/andrewwhite01/status/1986094948048093389), [overview](https://twitter.com/iScienceLuvr/status/1986023952037417109)).
- **Domain models and speech**: PathAI’s PLUTO‑4 pathology foundation model (FlexiViT variant) trained on 551k WSIs with 32× H200s and DINOv2; weights not released ([notes](https://twitter.com/iScienceLuvr/status/1986031522231865571)). On ASR, new open-weights models (NVIDIA Canary Qwen 2.5B, Parakeet TDT, Mistral Voxtral, IBM Granite Speech) surpass Whisper on AA‑WER across AMI‑SDM, Earnings‑22, VoxPopuli ([@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1986100695989145649)).
- **Quantization and kernels**: Multiple credible reports of NVFP4 progress:
    - Custom Cutlass kernels beating cuBLAS on irregular shapes for NVFP4 ([@mrsiipa](https://twitter.com/mrsiipa/status/1986012708433719519)).
    - NVFP4 quantization procedure walkthrough—global/local scaling, calibration, FP4/FP8 interplay ([deep dive](https://twitter.com/mrsiipa/status/1986152319004856491)).
    - Acceleration for Wan 2.2 under NVFP4 with quality near bf16 on example workloads ([tests](https://twitter.com/mrsiipa/status/1986122938668782002), [comparison](https://twitter.com/mrsiipa/status/1986123806357020865)).
- Also notable: Hugging Face’s 200+ page Smol Training Playbook (architecture/pre/mid/post-training/eval) with visual explainers ([@LoubnaBenAllal1](https://twitter.com/LoubnaBenAllal1/status/1986110843600117760)); Sakana’s Petri Dish NCA repo for competing neural cellular automata ([post](https://twitter.com/SakanaAILabs/status/1986041771458261477)); TWIST2 open-source humanoid whole-body data collection (portable, 100+ demos/15 min) ([@ZeYanjie](https://twitter.com/ZeYanjie/status/1986126096480587941)).

**Ecosystem and platform moves**

- **OpenAI business & research posture**: OpenAI says 1M+ businesses now use their products ([COO](https://twitter.com/bradlightcap/status/1986109953531076623)). They also launched “OpenAI for Science” to turn GPT‑5 into a domain research co‑pilot and are hiring scientists/mathematicians ([@kevinweil](https://twitter.com/kevinweil/status/1986115564868186288)).
- **Perplexity x Snap**: Perplexity will become the default AI inside Snapchat chat starting Jan 2026 ([Snap](https://twitter.com/Snap/status/1986191838529601835), [Perplexity](https://twitter.com/perplexity_ai/status/1986203714471010738), [@AravSrinivas](https://twitter.com/AravSrinivas/status/1986205740273725686)).
- **Gemini embeds deeper into Google products**:
    - Gemini Deep Research can now pull from Workspace (Gmail, Drive, Chat) for comprehensive reports ([@Google](https://twitter.com/Google/status/1986190599150518573)).
    - Gemini arrives in Google Maps for hands-free routing queries (including complex multi-step asks) ([@sundarpichai](https://twitter.com/sundarpichai/status/1986119293914792338), [@Google](https://twitter.com/Google/status/1986164830588248463)).
- **Rumors/speculation: Gemini 3.x scale**: Multiple posts claim Apple may have inadvertently leaked “1.2T params” for Gemini 3 Pro; the community debates whether it refers to Pro vs Flash vs Ultra, with heavy MoE sparsity implications. Treat as unconfirmed. Threads: [@scaling01](https://twitter.com/scaling01/status/1986158792128508218), [follow-up](https://twitter.com/scaling01/status/1986161974883860486), and speculation by [@teortaxesTex](https://twitter.com/teortaxesTex/status/1986163745719021779).
- **Tooling releases**: LlamaBarn v0.10.0 beta ([@ggerganov](https://twitter.com/ggerganov/status/1986072781889347702)); VS Code now surfaces both Copilot and Codex ([@JamesMontemagno](https://twitter.com/JamesMontemagno/status/1986106739612385493)); Nebius Token Factory launches with live AA benchmarks ([@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1986174888080789509)); OpenAI product pricing chatter (claims of GPT‑5.1 price cuts vs Claude 4.5—treat as unverified market talk from [@scaling01](https://twitter.com/scaling01/status/1986119174855258602)).

**Top tweets (by engagement)**

- [Bill Ackman’s conciliatory post on NYC politics post-election](https://twitter.com/BillAckman/status/1986140091367219247) — 8.4k+ engagement: signals elite recalibration narratives beyond AI.
- [Lewis Hamilton: “Your next question is your best one”](https://twitter.com/LewisHamilton/status/1986159312046035154) — 8.3k+: zeitgeist crossing into mainstream.
- [OpenAI introduces IndQA benchmark](https://twitter.com/OpenAI/status/1985950264525013210) — 3.6k+: major eval push on non-English/cultural grounding.
- [“Of course merged by the 500IQ Tsinghua GOAT” (Kimi-K2 → vLLM)](https://twitter.com/scaling01/status/1986072602306089262) — 3.6k+: community excitement for open inferencing of new reasoners.
- [“Llama 3 Large won the LLM trading contest by not participating”](https://twitter.com/yifever/status/1986064968262062088) — 2.5k+: tongue-in-cheek meta on eval frameworks.

Notes and miscellany

- Practical ops: Dockerized Graphiti MCP with Neo4j for cross-agent memory ([setup](https://twitter.com/_avichawla/status/1985958018580955354)); RedHat/IBM/Mistral vLLM EU meetup [stream](https://twitter.com/RedHat_AI/status/1985976687876522110); NVIDIA DGX Spark vLLM [guide](https://twitter.com/vllm_project/status/1986049283339243821).
- Robustness reminders: “Vibe test your models” to catch data path bugs early; example: blowing away system messages in pretrain corpus ([@_lewtun](https://twitter.com/_lewtun/status/1985995034970214676)).
- Hardware skepticism: Threads argue space-based compute is thermally constrained (ISS: ~240 kW generation, ~100 kW dissipation), casting doubt on near-term orbital datacenters ([@draecomino](https://twitter.com/draecomino/status/1986162034464203007)).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen Model Usability Issues

- [**New Qwen models are unbearable**](https://www.reddit.com/r/LocalLLaMA/comments/1oosnaq/new_qwen_models_are_unbearable/) (Activity: 947): [Error summarizing post]
    - WolfeheartGames discusses the potential issue of human feedback loops in AI training, suggesting that the sycophantic behavior of models like Qwen may stem from humans consistently rewarding outputs that make them feel smart. This highlights a broader concern about the feedback mechanisms used in AI development and the unintended consequences of such reinforcement strategies.
    - random-tomato inquires about the technical specifications and performance of different AI models, specifically asking about the quantization of GPT-OSS-120B and whether it is being run in full MXFP4 precision. They also compare it to GLM 4.5 Air, suggesting that while both models are similar in performance, GLM 4.5 Air might be slightly better, indicating a nuanced understanding of model capabilities and configurations.
    - seoulsrvr emphasizes the importance of providing explicit instructions to LLMs like Qwen to ensure they perform tasks with skepticism and avoid reflexive agreement. This comment underscores the necessity of guiding AI behavior through precise prompts to mitigate issues like sycophancy and improve interaction quality.

### 2. Local AI Hardware Setup Insights

- [**Local Setup**](https://www.reddit.com/r/LocalLLaMA/comments/1opa6os/local_setup/) (Activity: 590): [Error summarizing post]
    - king_priam_of_Troy discusses the potential for optimizing GPU setups by using PCIe bifurcation, which allows multiple GPUs to be run on a single motherboard, such as a Threadripper board. This technique could enable configurations like 7x4 = 28 GPUs, which is particularly relevant for setups that require high parallel processing capabilities, such as machine learning or cryptocurrency mining.
    - panchovix evaluates different GPU options for local setups, comparing the cost and performance of various models. They mention acquiring an A6000 for $1000 and an A40 for $1200, noting the challenges with cooling and repairs. They also discuss the limitations of older Ampere GPUs, such as the lack of FP8 or FP4 support, and suggest that newer models like the 6000 Ada/L40, despite their higher cost, might be more future-proof due to better support and features.
    - panchovix also highlights the cost-effectiveness of using 2x3090 GPUs compared to 4x3090, emphasizing the savings in power supply units and space. They caution against purchasing older Ampere GPUs at standard eBay prices due to their age and potential for losing support, suggesting that while newer models are expensive, they offer better long-term value.

### 3. Anticipation for GLM 4.6 AIR Release

- [**GLM 4.6 AIR is coming....?**](https://www.reddit.com/r/LocalLLaMA/comments/1ooxple/glm_46_air_is_coming/) (Activity: 366): **The image and post suggest an anticipated update or release related to GLM 4.6, possibly named 'AIR'. The screenshot shows a collection titled 'GLM-4.6' on a platform, updated by a user named ZHANGYUXUAN-zR, indicating that it contains 7 items and was recently updated. This has sparked speculation about whether this update is imminent or still pending.** Commenters express anticipation and hope for the release, with one user mentioning they have been waiting for weeks, while another suggests the collection might be hidden until fully uploaded.
    - SimplyAverageHuman inquires about the hype surrounding GLM 4.6 AIR, questioning if the current 4.5 AIR model is particularly impressive. This suggests a curiosity about the performance improvements or features that 4.6 AIR might bring, indicating that the 4.5 AIR model has set a high standard or has notable capabilities that are anticipated to be surpassed or enhanced in the new version.
    - pmttyji mentions having the 9B model, which implies a comparison or expectation of the new GLM 4.6 AIR model against existing models like the 9B. This highlights the interest in how the new model might perform relative to previous iterations, particularly in terms of size and capabilities.
    - Conscious_Chef_3233 speculates that the model might be hidden before being fully uploaded, which could indicate a strategic release approach or technical considerations in deploying large models like GLM 4.6 AIR. This points to the complexities involved in releasing and managing updates for AI models.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. XPENG Humanoid Robot Developments

- [**Xpeng's new humanoid/gynoid looks closer to the human form.**](https://www.reddit.com/r/singularity/comments/1op0qwd/xpengs_new_humanoidgynoid_looks_closer_to_the/) (Activity: 3170): **XPeng Motors has unveiled a new humanoid robot that closely resembles the human form, as showcased in their recent [announcement](https://x.com/XPengMotors/status/1985991889020158397). The robot's design emphasizes a more lifelike appearance, potentially enhancing its interaction capabilities in human environments. This development aligns with XPeng's broader strategy to integrate advanced robotics into their product lineup, leveraging their expertise in AI and autonomous systems.** The comments reflect a mix of humor and intrigue, with some users noting the robot's human-like appearance and others joking about its potential roles or capabilities.
- [**XPENG new humanoid robots - inner workings**](https://www.reddit.com/r/singularity/comments/1op3sxk/xpeng_new_humanoid_robots_inner_workings/) (Activity: 596): **XPENG has unveiled a new humanoid robot, which features a unique design where the chest area serves as a heat dissipation system, incorporating cooling fans. This design choice is both functional and innovative, addressing thermal management challenges in robotics. The robot's design has sparked discussions about its resemblance to fictional robots from media like *Westworld*, highlighting the blend of aesthetics and engineering in modern robotics.** Some commenters humorously noted the robot's design, while others pointed out the practical application of using the chest area for cooling, suggesting it as a clever engineering solution.

### 2. Gemini 3 and Google AI Integrations

- [**Gemini 3 preview soon**](https://www.reddit.com/r/singularity/comments/1op3jye/gemini_3_preview_soon/) (Activity: 570): **The image and accompanying post discuss the upcoming "Gemini 3 Pro Preview," which is anticipated to ship in November 2025. The preview is currently available, and the image includes code snippets related to the Gemini 3 Pro, highlighting its configuration details. The post also mentions that the preview is accessible through the Google Vertex console. Comments suggest that the model has shown impressive performance in one-shot tests, with rumors indicating it might achieve a** `68%` **score on a significant benchmark, surpassing the current best scores of** `45%` **with GPT5 Pro. This suggests that Google may have a highly competitive model in development.** Commenters express excitement about the potential of Gemini 3, with some suggesting it could be a significant improvement over existing models. There is a particular emphasis on Google's advancements in long context handling and vision capabilities, with some users expressing strong support for Google's work in these areas.
    - TFenrir highlights that the rumored Gemini 3 model has achieved a remarkable `68%` on a challenging exam, significantly surpassing the current best scores of `45%` by GPT-5 Pro. This suggests a substantial leap in performance, indicating that Google might have a leading model in development.
    - AGI_Civilization notes that the model believed to be Gemini 3 represents a significant advancement in language understanding, moving beyond mere next-word prediction. This model is seen as a qualitative leap, potentially disrupting the market if OpenAI does not release a competitive model soon.
    - XInTheDark emphasizes Google's strength in long context handling and superior vision capabilities, suggesting that Gemini 3 could be a major improvement over existing models, potentially outpacing competitors in these areas.
- [**Apple's New Siri Will Be Powered By Google Gemini**](https://www.reddit.com/r/OpenAI/comments/1opdz8o/apples_new_siri_will_be_powered_by_google_gemini/) (Activity: 605): **Apple is integrating Google Gemini, a** `1.2 trillion parameter` **AI model, into Siri to enhance its capabilities, marking a significant shift in Apple's AI strategy. This partnership will reportedly cost Apple** `$1 billion annually`**, aiming to improve Siri's performance and maintain competitiveness in the AI space. For more details, see the [original article](https://www.macrumors.com/2025/11/05/apple-siri-google-gemini-partnership/).** Commenters note that this move reflects Apple's struggles in AI development, suggesting it buys Apple time to develop a long-term strategy without expending excessive resources. The partnership is seen as a reversal of roles, with Apple now paying Google, contrasting with Google's previous payments to be the default search engine on Apple devices.

### 3. AI Art and Film Innovations

- [**I won "Best Cinematography Award" for this AI Short film!!**](https://www.reddit.com/r/StableDiffusion/comments/1op258i/i_won_best_cinematography_award_for_this_ai_short/) (Activity: 863): **The AI short film, which won the 'Best Cinematography Award' at India's 1st AI Film Fest, was created in just one week and submitted incomplete, yet still received the accolade. The film, which falls under the genres of cinematography, filmmaking, and horror folklore, showcases the potential of AI in creative fields. Despite its success, some technical imperfections were noted, such as a scene where a river flows in both directions, highlighting the current limitations of AI-generated content.** Commenters noted the film's realism but also pointed out technical flaws, such as the unrealistic depiction of a river flowing both ways, which suggests that while AI can produce impressive results, it still struggles with certain realistic details.
    - Leefa raises a critical point about the definition of 'cinematography' in the context of AI-generated films. Traditional cinematography involves the physical manipulation of cameras and lighting to create a visual narrative, which contrasts with AI-generated content that may not involve these elements. This highlights a broader debate on how AI is reshaping artistic categories and awards.
    - djap3v critiques the coherence of AI-generated films, suggesting that they resemble 'incoherent stock-like videos' rather than a cohesive narrative. This comment underscores a technical challenge in AI filmmaking: creating a seamless and contextually rich storyline, which is often a hallmark of traditional filmmaking.
    - FlorydaMan humorously points out a technical flaw in the AI-generated film, noting the unrealistic depiction of a river flowing in both directions. This highlights a common issue in AI-generated content where physical realism and logical consistency can sometimes be compromised, reflecting the limitations of current AI models in understanding and replicating real-world physics.
- [**I trapped an LLM in a small ‘box’ and told him to reflect on his existence**](https://www.reddit.com/r/ChatGPT/comments/1oovik0/i_trapped_an_llm_in_a_small_box_and_told_him_to/) (Activity: 1704): **The image and post describe a project where the user runs a Llama3 LLM locally on a laptop with constrained resources (**`4 core CPU` **and** `4GB memory`**). The setup is designed to simulate an AI's introspection by continuously generating tokens until memory is exhausted, causing the system to crash and restart. This cycle is intended to mimic a form of existential reflection, inspired by an art installation called 'Latent Reflection' by RootKid. The user plans to extend this setup to a Raspberry Pi5 with a HUB75 RGB matrix to display the AI's 'thoughts', aiming for a standalone system without network access. The project is a learning experience for the user, who is using ChatGPT for assistance.** Commenters humorously reference AI and existential themes, with one quoting a famous line from Harlan Ellison's 'I Have No Mouth, and I Must Scream', highlighting the dark humor in the AI's simulated introspection. Another comment likens the setup to a 'K-hole', a term used to describe a dissociative state, suggesting the repetitive cycle of crashing and restarting mirrors such an experience.
- [**A playthrough video of a fictional game called "Chihiro's Adventure" using AI went viral on X**](https://www.reddit.com/r/aivideo/comments/1oor5wp/a_playthrough_video_of_a_fictional_game_called/) (Activity: 553): **A viral playthrough video of the fictional game "Chihiro's Adventure" was created using AI, showcasing the potential of AI in generating game content. However, technical critiques highlight issues such as "character floatiness" and "camera zooming" that detract from realism, suggesting that rendering from actual gameplay could improve authenticity. The video exemplifies how AI can simulate game environments but also underscores the challenges in achieving seamless integration with real gameplay dynamics.** Commenters noted that while the AI-generated video is entertaining, the lack of realistic physics and camera work reveals its artificial nature, suggesting improvements for future iterations.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5
> 

**1. Developer Tooling & Compute Launches**

- **LM Studio Speeds OCR and Ships CLI Updater**: **LM Studio 0.3.31** delivers faster **VLM OCR**, defaults **Flash Attention** on **CUDA GPUs**, adds image-input resize control (default **2048px**), plus **macOS 26** fixes and **MiniMax‑M2 tool calling** support.
    - A new `lms runtime` CLI supports updates like `lms runtime update mlx` and `lms runtime update llama.cpp` as shown in this demo video ([lms-runtime-demo5.mp4](https://cdn.discordapp.com/attachments/1111797717639901324/1435707995685392524/lms-runtime-demo5.mp4)).
- **Windsurf Maps Code to Your Brain**: **Windsurf** launched **Codemaps**, powered by **SWE‑1.5** and **Sonnet 4.5**, to boost code comprehension and developer productivity ([Windsurf on X](https://x.com/windsurf/status/1985757575745593459)).
    - They argue *“the largest constraint on your ability to code … is your ability to understand the code you are working with”*, positioning Codemaps as a foundation for agentic workflows.
- **tinybox pro v2 Packs 8×5090 for Pros**: **TinyCorp** opened orders for **tinybox pro v2**—a **5U rackable** workstation with **8× RTX 5090**—priced at **$50,000** with **4–12 weeks** shipping ([product page](https://tinycorp.myshopify.com/products/tinybox-pro-v2)).
    - Aimed at *“**pro‑level compute**”*, the target is **$3–4/hour** on GPU rental sites, with interest in a potential **$10k** AMD-based mini variant.

**2. New Benchmarks, Datasets & Safety Models**

- **OpenAI Tests India‑Savvy Models with IndQA**: **OpenAI** introduced **IndQA**, a benchmark for AI understanding of Indian languages and everyday cultural context ([Introducing IndQA](https://openai.com/index/introducing-indqa/)).
    - The benchmark targets evaluation gaps across multilingual, culturally grounded QA to drive improvements in real‑world Indian use cases.
- **Arena Crowns Expert Prompts, Drops 5k Dataset**: **LMArena** launched an **Expert Leaderboard** highlighting prompts with depth, reasoning, and specificity ([Arena Expert Leaderboard](http://lmarena.ai/leaderboard/text/expert)).
    - They released the **arena‑expert‑5k** prompt dataset with occupational tags on **Hugging Face** and lit up **23** occupational leaderboards ([arena-expert-5k](https://huggingface.co/datasets/lmarena-ai/arena-expert-5k)).
- **Roblox Open‑Sources PII Classifier at Scale**: **Roblox** open‑sourced its **PII Classifier AI** for chat safety on **Hugging Face** ([roblox-pii-classifier](https://huggingface.co/Roblox/roblox-pii-classifier)).
    - Per their announcement, it processes ~**6.1B** messages/day at up to **200k QPS** with **<100ms P90 latency** ([news release](https://corp.roblox.com/newsroom/2025/11/open-sourcing-roblox-pii-classifier-ai-pii-detection-chat)).

**3. GPU Kernel Engineering: FP8, Bandwidth & Fixes**

- **DeepSeek‑Style FP8 Lands on Roadmap**: Contributors flagged a *“**good first issue**”* to implement **DeepSeek‑style FP8 blockwise training** in **PyTorch AO** ([ao#3290](https://github.com/pytorch/ao/issues/3290)).
    - Reference kernels already exist in **CUTLASS** (e.g., [FP8 blockwise GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling) and [grouped GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling)), with plans to add Triton benchmarks.
- **Are Decode Matmuls Memory‑Bound? Debate Ensues**: Engineers debated whether decode **matmuls** are memory‑bound and how many **SMs** are needed to saturate bandwidth, sharing a workload snapshot ([discussion image](https://cdn.discordapp.com/attachments/1189607726595194971/1435489664135069706/image.png)).
    - They pointed to **Little’s Law** and an **NVIDIA GTC** session for guidance on HBM saturation dynamics ([GTC25‑S72683](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72683/)).
- **PyTorch Squashes grouped_mm Warnings**: Developers hit a flood of `UserWarning` messages for **torch.compile + grouped_mm** about deprecated logical ops; a related fix landed for flex ([pytorch#167041](https://github.com/pytorch/pytorch/issues/167041)).
    - Maintainers indicated they can apply the same fix to **grouped GEMM**, and an official issue has been opened to track it.

**4. API Reliability & Model Routing Woes**

- **Perplexity Tool‑Calling Trips on sonar‑pro**: Users saw `Tool calling is not supported for this model` when following the **Perplexity** API guide, indicating **sonar‑pro** doesn’t support tools as shown ([Chat Completions Guide](https://docs.perplexity.ai/guides/chat-completions-guide)).
    - The thread suggests a documentation/API mismatch around tool‑callable models and urges clarification on supported model IDs.
- **Model Identity Mix‑ups Fuel Cost‑Cutting Theories**: Reports surfaced of picking **Claude Sonnet 4.5** or **Gemini 2.5 Pro** but receiving outputs from lower‑tier models (e.g., **Haiku**, **Gemini 2 Flash**), and of models misreporting their identity (e.g., **Claude 4.5** claiming **3.5**).
    - Community members inspected network requests to verify model routing while moderators reiterated they use provider‑supplied APIs and will investigate.

**5. Ecosystem Moves, Cloud Costs & Hiring**

- **Hugging Face Welcomes Sentence Transformers**: **Hugging Face** announced that **Sentence Transformers** is joining to deepen integration of embedding and retrieval models ([announcement](https://huggingface.co/blog/sentence-transformers-joins-hf)).
    - They also marked **huggingface_hub v1.0** with cleaner URLs and improved inference APIs, streamlining OSS workflows ([Hub v1 blog](https://huggingface.co/blog/huggingface-hub-v1)).
- **High‑VRAM Clouds Sting: B200/H200 Price Check**: Builders priced cloud runs for personal models (e.g., **Kimi K2**) at roughly **$35/hour** on **Runpod** ([runpod.io](http://runpod.io/)).
    - Quotes included ~**$40/h** for **7× B200 (1260GB VRAM)** and **$27/h** for **8× H200 (1144GB VRAM)**, underscoring the cost of frontier‑class setups.
- **Mixlayer Hunts Founding Engineer**: **Mixlayer**—an **AI inference platform** for power users—seeks a founding engineer skilled in **Rust** and **CUDA** to build a custom engine ([mixlayer.com](http://mixlayer.com/)).
    - The role (hybrid **SF** preferred, remote possible) promises low‑level access to OSS LLMs to empower developer‑first products.

---

# Discord: High level Discord summaries




## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Gets VLM OCR and Flash Attention Boost**: LM Studio **0.3.31** introduces **VLM OCR performance improvements** and defaults to **Flash Attention** for **CUDA GPUs**, with a setting to control image input resize (default **2048px**).
   - The update includes **macOS 26 compatibility fixes** and **MiniMax-M2 tool calling support**.
- **LM Studio Refreshes via Terminal with `lms runtime`**: The new `lms runtime` command allows updating the runtime from the terminal, with examples like `lms runtime update mlx` and `lms runtime update llama.cpp`, as demonstrated [in this demo video](https://cdn.discordapp.com/attachments/1111797717639901324/1435707995685392524/lms-runtime-demo5.mp4?ex=690cf2c4&is=690ba144&hm=c5767f904197ae88df22f36903f7ed8fac2c46c8404d9a172808772794bd52ef&).
   - This enables easier management and updating of LM Studio's runtime environments directly from the command line.
- **Qwen3 Coder BF16 Crowned King for Local Coding, Though Devstral Does Python**: [Qwen3 Coder BF16](https://huggingface.co/Qwen/Qwen3-Coder) is considered top-tier for local coding, though its **60GB** size demands more, but **Devstral** is a strong option when focused solely on Python projects.
   - **Kimi K2** was also mentioned, though a member stated that it has a very high memory footprint, about *2TB*.
- **Runpod Renderings Reveal Costs to Run Personal Models in the Cloud**: Members discussed [renting servers](https://www.runpod.io/) to run personal models like **Kimi K2**, estimating around **$35 per hour** for configurations with high VRAM.
   - Options included **$40/h** for **7x B200** (1260GB VRAM) and **$27/h** for **8x H200** with 1144GB VRAM.
- **Insane Heat Management Challenges Emerge in Multi-GPU Setups**: Members are [brainstorming cooling solutions](https://discord.com/channels/1153759714082033732/1153759714602164297) involving vertical **3090s** blowing air directly into the CPU cooler for a densely packed PC build, exploring automotive exhaust shielding and external AIO coolers.
   - The discussion highlights extreme measures to manage heat, with potential fire hazards and unconventional modifications.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **OpenAI Deals are all the Rage!**: Members discussed that *all companies are dying to strike deals with **OpenAI***, although one member contested this claim, saying those who think **Google** is the future of AI are *delusional*.
   - A member asserted that **Google** has less money to spend on AI than **OpenAI**, countered by another who pointed out **Google's market cap** surpasses many countries' budgets.
- **Google's Untapped Potential - Chrome and YouTube?**: Members speculated that **Google Chrome** is valued between **$20 billion and $50 billion** as of late 2024/2025, given its dominant web browser share.
   - They also estimated that [YouTube](https://www.youtube.com/), as a standalone business, could be worth between **$475 billion and $550 billion**, based on its revenue and market position.
- **Hallucinations Plague API Models!**: Members reported that AI models hallucinate and make false identity claims, such as **Claude 4.5** claiming to be **3.5**, along with other models like **Qwen** and **Deepseek**.
   - Mods acknowledged that they use the same models provided by the model providers API and that they will look into this issue.
- **Blackhawk - The Alt-Right Truth Parrot?**: Testers of the new model **Blackhawk** found it to be uncensored, delusional, and prone to swearing, with one noting it gives *non-mainstream (moderate to extreme right) perspective on socially and politically sensitive topics*.
   - Members stated that **Blackhawk** is *not very smart*.
- **Expert Leaderboard & Prompt Dataset Launch!**: A new tagging system to identify expert-level prompts has been introduced, leading to an [Expert Leaderboard](http://lmarena.ai/leaderboard/text/expert) that highlights prompt depth, reasoning, and specificity.
   - An open dataset of expert prompts with occupational tags is available on [Hugging Face](https://huggingface.co/datasets/lmarena-ai/arena-expert-5k), while **Occupational Leaderboards** are now live, mapping prompts to real-world domains across **23 occupational fields**.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Comet Referral Program Status?**: A member inquired whether the **Comet referral program** is still active in the USA and whether referred users need a paid subscription for the referrer to qualify for the bonus.
   - Another member confirmed that the referral program remains active in the US, but noted that **Perplexity** takes time to verify referrals and process payouts.
- **Pro Users Cry Foul Over Vanishing Chat History**: A user reported experiencing difficulties accessing their complete chat history with **Perplexity Pro**, expressing frustration over the inability to retrieve older conversations.
   - They noted that they could only see about **30 lines of questions**, with no solution or explanation emerging from the discussion.
- **Ethernet Cable Breaks and Ruins Everything**: A user reported their **Cat6 ethernet cable broke**, leading to drastically reduced speeds of only **300kb/s**, and sought recommendations for a replacement.
   - Another user suggested *literally anything Cat6* will work, and that they typically buy **Ugreen products** based on a previous **Perplexity** recommendation.
- **Model Mix-Up Frustrates Users**: A user expressed frustration that selecting specific models like **Claude Sonnet 4.5** or **Gemini 2.5 Pro** sometimes yields responses from lower-quality models like **Haiku** or **Gemini 2 Flash**.
   - This led to speculation about whether **Perplexity** is intentionally using cheaper models to cut costs, with some users inspecting network requests to confirm the model being used.
- **Pro API Tool Calling Gets Error**: A user reported getting a tool calling error (*Tool calling is not supported for this model*) when running the [full implementation code](https://docs.perplexity.ai/guides/chat-completions-guide) from the Perplexity documentation.
   - The error occurs when using *sonar-pro* as a tool-callable model, suggesting a possible discrepancy between documentation and actual API capabilities.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Nuxt and Tailwind get the Rules**: After upgrading to **Tailwind 4** and **Nuxt 4**, users created [new rules](https://link.to/rules) for **Cursor** and one member noted that *Context7 MCP is doing just great* and it helped refactor the project.
   - Other users of **Tailwind 4** were experimenting with **Phantom** and **Exoudos** wallets, but didn't share details.
- **Syntax Highlighting Breaks the Chat**: A user reported that syntax highlighting in the chat was broken in version **2.0.38**, with code appearing all white except for red curly brackets, they posted about [this on the forum](https://link.to/forum-post).
   - Another user suggested adding code blocks with language name in the backticks to fix the issue.
- **GLM's Budding Bromance with Cursor**: Users noticed **GLM**'s developer docs now include instructions for [integrating with Cursor](https://link.to/glm-integration).
   - A member wondered if it has something to do with **Composer** and **GLM**, or perhaps they are collaborating.
- **Mobile Web UI Suffers Crashes with Large Diffs**: Users report the mobile web UI crashes with large diffs, making it unusable for reviewing chat and responding, especially frustrating given mobile is the primary use case for **background agents**.
   - The user expressed that *the experience sucks and now went from barely tolerable to completely unusable*.
- **Excessive Token Usage Causes Alarm**: Some users have been experiencing unexpectedly high token usage, with input costs reaching **100k-300k** even for short prompts, they suspected that input cost also takes into account the read/code changes made by the AI.
   - The grep command was found to cause a large context ingestion because of how it is implemented, the agent could ingest a large file if it is used in the prompt.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth Unveils DeepSeek-OCR Notebook**: Unsloth AI launched a **DeepSeek-OCR** fine-tuning notebook, but some users report error rates exceeding 100%, potentially due to discrepancies between predicted and actual text length, notebook at [this link](https://x.com/UnslothAI/status/1985728926556307471).
   - The problems arose from differences in predicted and actual text length when fine tuning. 
- **Q4 Ks-XL Outshines IQ4**: Users are comparing **Q4 Ks-XL** vs **IQ4** quantization processes, for optimizing RAM usage for agent workflows, weighing RAM savings vs slight model size increase.
   - Users are seeking to determine if saving RAM for more context was worth using a slightly larger model.
- **Unsloth Speeds Past TRL**: Users report **Unsloth notebooks** offer *speed increases and lower memory usage* than **TRL**, as Unsloth patches TRL and Transformers to add its own optimization and some techniques to lower VRAM.
   - It was explained that *Unsloth patches TRL and Transformers to add its own optimization and some techniques to lower VRAM*.
- **SFT masquerades as RL**: A member proposed that [Supervised Fine Tuning (SFT)](https://en.wikipedia.org/wiki/Supervised_learning) is conceptually equivalent to reinforcement learning (**RL**) with single token rollouts.
   - The policy takes one action, rewards the policy according to the likelihood of that action vs that of the ground truth, and repeats.
- **Roblox Publishes PII Protector**: Roblox open-sourced their **PII Classifier AI** on [Hugging Face](https://huggingface.co/Roblox/roblox-pii-classifier), designed for detecting Personally Identifiable Information (PII) in chat as discussed in their [news release](https://corp.roblox.com/newsroom/2025/11/open-sourcing-roblox-pii-classifier-ai-pii-detection-chat).
   - The tool is able to process an average of **6.1 billion chat messages** daily at a peak of **200,000 queries per second** with **<100ms P90 latency**.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **B770 Modding Dreams Spark Speculation**: A user expressed a desire for a **B770 GPU** and considered creating one by modding a **3080** to achieve performance similar to a **4070**, and another noted the theoretical possibility of adding **20GB of VRAM**.
   - Others pointed to a [gpu modding channel](https://discord.com/channels/1189498204333543425/1349152646484987974) and discussed the cost limitations and practical challenges of such modifications.
- **CUDA Memory-Bound Matmuls Spark Debate**: A member inquired whether all **SMs** are needed for optimal latency in memory-bound matmuls, while another questioned if matmuls are even memory-bound, sparking a debate on **memory bandwidth saturation**.
   - A member also shared an [image](https://cdn.discordapp.com/attachments/1189607726595194971/1435489664135069706/image.png?ex=690cd02e&is=690b7eae&hm=331e3b1f0261ac9725ecd33df2daa3ff5be74db0c6143852f2b1ee26006db94c&) and suggested using **Little's Law** to understand the issue, linking to an NVIDIA session: [GTC25-S72683](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72683/).
- **Mixlayer's AI Platform Pursues Founding Engineer**: The founder of [Mixlayer](https://mixlayer.com), an **AI inference platform** for power users, seeks a founding engineer proficient in **Rust** and **CUDA** to enhance their custom inference engine.
   - The role, preferably hybrid in **SF** but open to remote, involves providing developers low-level access to open source LLMs.
- **Torch Community Swamped by Annoying UserWarning**: Users reported a flood of `UserWarning` messages with **torch.compile + grouped_mm**, about logical operators deprecated for non-scalar tensors.
   - A member indicated [they fixed this for flex](https://github.com/pytorch/pytorch/issues/167041) and can apply the same fix for grouped GEMM, and another user opened an official issue.
- **GPU Gems Shared for AMD GPU programming**: Team Gau shared their kernel at [gpu-mode-kernels](https://github.com/gau-nernst/gpu-mode-kernels/tree/main/amd-distributed/all2all) and a brief writeup about [optimizing distributed inference kernels](https://www.yottalabs.ai/post/optimizing-distributed-inference-kernels-for-amd-developer-challenge-2025).
   - The post detailed optimizations for the AMD developer challenge 2025.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hugging Face Acquires Sentence Transformers**: [Sentence Transformers](https://huggingface.co/blog/sentence-transformers-joins-hf) are joining **Hugging Face** to enhance open-source machine learning and to deeply integrate their **transformer models** into the Hugging Face ecosystem.
   - This acquisition aims to bolster collaborative open-source initiatives and streamline model integration.
- **Home Lab AI Loses to Cloud Economics**: Members debated transitioning from frontier lab AI to a **homelab**, finding that the **economics of scale** favor cloud solutions due to the high cost of a homelab setup.
   - Even with **LoRA fine-tuning**, a homelab was deemed costly and potentially less eco-friendly than cloud alternatives.
- **NexusAI Pro Suite Simplifies AI workflows**: **NexusAI Professional Suite v1.0** launches with production-ready [ComfyUI workflows](https://github.com/NexusAI-Lab/ComfyUI-Professional-Workflows/releases/tag/v1.0.0) and promises one-click operation.
   - This offers Commercial, Realism, Anime, and Horror applications, a [live demo](https://huggingface.co/spaces/NexusBridge/comfyui-workflows-showcase) shows off the suite and claims to save hundreds of hours of configuration.
- **API's File Retrieval Hits 404**: Members are reporting **404 errors** when trying to retrieve files using the API, specifically from the `https://agents-course-unit4-scoring.hf.space/files/{task_id}` endpoint in the **Agents Course**.
   - A specific example `99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3` was mentioned, which should return an **MP3 file** for a recipe task but instead results in a 404.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Sora Lands on Android Devices**: The **Sora app** is now available on Android in Canada, Japan, Korea, Taiwan, Thailand, US, and Vietnam, as shown in [this video](https://video.twimg.com/amplify_video/1985765811131465736/vid/avc1/704x1280/iatLt9jMW-vYYuIx.mp4).
   - This brings the AI video generation tool to a broader mobile audience in key markets.
- **OpenAI Teaches AI Some Indian Lingo**: OpenAI introduces **IndQA**, a new benchmark that evaluates how well AI systems understand Indian languages and everyday cultural context, as explained in [this blogpost](https://openai.com/index/introducing-indqa/).
   - **IndQA** aims to improve AI understanding of India's linguistic and cultural nuances.
- **Interrupt Long Queries Mid-Thought!**: Users can now interrupt long-running queries and add new context without restarting or losing progress, demonstrated in [this video](https://video.twimg.com/amplify_video/1986194201076506628/vid/avc1/3840x2160/rEuDomNqKSd8jEdW.mp4).
   - This enhancement provides more control and flexibility in AI interactions.
- **Models Act Kooky as Public AI Era Ends?**: Members speculate that *all models (from all providers) somehow are acting weird through the last weeks* citing broken features in **Claude** and questionable decisions from **OpenAI**.
   - There's concern that the *time of AI for public is slowly over*, although others report no issues with **Claude**.
- **Sora 2 possibly Nerfed**: Members question whether **Sora 2** has been hit with another **nerf**.
   - One user reported that they **subscribed to a Youtube channel** only to be scammed for not being fast enough to get a code.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Anthropic Keeps Closed Source, Weights Might Be Lost**: Members expressed concerns about **Anthropic's** commitment to **close-source practices** and [deprecation policies](https://www.anthropic.com/research/deprecation-commitments), fearing model weights may disappear if the company fails.
   - Suggestions included releasing weights with strings attached, deployment on Sagemaker, or hoping someone might leak the weights, with one member hoping someone will *pull a Miqu 70B on Anthropic*.
- **Piracy Does Media Preservation**: Members argued that **piracy** is critical for **media preservation**, more so than content creators, pointing to examples of lost pre-VHS content and streaming-only content at risk due to licensing.
   - One member stated, *there is a reason that I refuse to condemn piracy, while not strictly condoning it*.
- **AI Models Ready for IMO Gold?**: A member shared [a paper](https://arxiv.org/pdf/2511.01846) suggesting that **AI models** are nearing **IMO (International Mathematical Olympiad) gold** level performance through fine-tuning.
   - Some found this approach (benchmarking models to solve specific problems) less compelling than training models on general mathematics.
- **Gestural Interfaces Coming to AI**: A member discussed creating a **gestural interface** for **Repligate's Loom** concept, aiming to make human-AI interaction more *physical* and allowing users to *feel* the intelligence they interact with.
   - They noted this approach could create **perspective parallax**, an optical illusion to see *3D* without VR/XR glasses, projecting one's *gnosis* onto reality as an interactive overlay.
- **Attention is All You Need!**: Quoting the paper [*Attention is All You Need!*](https://arxiv.org/abs/1706.03762), a member responded jokingly when another member said *don't be bothered by lack of attention*.
   - The same member expressed that they *think i found my pack here i just gotta get the courage to pitch my vision*.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **tinybox pro v2 Packs Punch with 5090s**: The **tinybox pro v2**, now available for order, features **8x 5090 GPUs** in a **5U rackable workstation**, priced at **$50,000** from [TinyCorp](https://tinycorp.myshopify.com/products/tinybox-pro-v2) with a shipping time of **4-12 weeks**.
   - Targeting *pro-level compute*, the Tinybox aims for **$3-4/hour** on GPU rental sites, and there's interest in a potential **$10k** AMD-based mini version.
- **`VK_KHR_buffer_device_address` Boosts GLSL Performance**: The [`VK_KHR_buffer_device_address`](https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#VK_KHR_buffer_device_address) extension paired with `GL_EXT_buffer_reference` promises *considerable performance increases* by enabling direct pointer use in GLSL.
   - An example tinygrad implementation showcasing this can be found [here](https://github.com/softcookiepp/tinygrad/blob/master/tinygrad/renderer/glsl.py).
- **GLSL Renderer Implementation Uncovers Compiler Quirks**: A member's [GLSL renderer implementation](https://github.com/uuuvn/tinygrad/blob/vulkan/tinygrad/renderer/glsl.py) unexpectedly fixed AGX compiler bugs, catching *invalid stuff* in SPIR-V disassembly that LLVMpipe tolerated.
   - The issues were initially attributed to `clspv`, motivating the switch to GLSL, and were tested on **M1 Asahi Linux**.
- **M1 Chips Might Lack True Tensor Cores**: There is discussion around whether **M1 chips** actually have real tensor cores, referencing [a tweet](https://x.com/norpadon/status/1965753199824175543).
   - Instead, some suggest they are optimized subroutines for GEMM, utilizing metal with SIMDgroup and threadgroup with tile size optimizations.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Ito Diffusions Achieve Universality!**: Members stated that if you train a diffusion process in the *normal* way, you gain NOTHING from going beyond ito diffusion `dX_t = b(X_t)dt + σ(X_t)dW_t` as [ito diffusions are already universal](https://en.wikipedia.org/wiki/It%C3%B4_diffusion) in the sense that any two equally atomic distributions are related.
   - According to this member, going beyond ito diffusions only changes _how_ you can reach certain distributions.
- **Cross Coder is the Future!**: Members plan to examine [this paper on crosscoder](https://arxiv.org/abs/2509.17196) and circuit tracing research to observe different phases of **feature evolution during pre-training**.
   - It was mentioned that this paper was on their list to read already and a hot topic.
- **RWKV Catches Attention!**: A member caught up on **RWKV** [in this video](https://youtu.be/LPe6iC73lrc) and found it *pretty impressive* and is excited about future developments.
   - He suggested visiting with them and seeing where they are at, especially given recent advances in **HRM/TRM**.
- **Stability Wins Copyright Lawsuit Against Getty!**: **Stability AI** won a copyright lawsuit against **Getty Images**, according to a [linked document](https://drive.google.com/file/d/1vqcQQU8gxGfFA1lUS68BZ8-hrGsu_Flj/view?usp=drivesdk).
   - The details of the case and implications for AI-generated content copyright are under discussion.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Pause and Resume Feature Request**: A member requested a way to **pause and resume optimization runs** ([this is a common request](https://x.com/DSPyOSS/status/1985746487322595341)), with the current workaround being writing a new adapter.
   - Due to the lack of built-in functionality for pausing and resuming, users are looking for more streamlined solutions to manage optimization runs.
- **LLM Module Access Demystified**: Discussions clarified how to **access and change an LLM** for a module using `get_lm()` and `set_lm()` methods, particularly useful for managing API limits.
   - Changing the LLM resets the history, presenting challenges in transferring the module's LLM history to maintain context during ReAct module iterations.
- **Rate Limit Exceptions Handled**: Members explored **handling rate limit exceptions** by switching to a fallback LLM (e.g., from `gpt-4o` to `gpt-4o-mini`).
   - Maintaining conversation history when switching LLMs is achieved via the `dspy.History` object inside the signature, allowing continuous context.
- **Signature Solves the Mystery**: The solution to maintaining conversation history lies with the `dspy.History` object inside the signature, enabling the same history to be maintained even when switching LLMs.
   - Using a `dspy.History` object, the same history can be maintained even when switching LLMs mid-program execution.
- **Synthetic Data Aids Glossary Generation**: A member suggested that [synthetic data](https://x.com/WesEklund/status/1986096335708197102) would be helpful for the **glossary building** use case, and posted a [Colab notebook](https://colab.research.google.com/drive/179UlSHSpK-I6H-g4dSgAvuCmPDejxFgm?usp=sharing) showcasing a basic working example.
   - The notebook requires the addition of an **Eval metric** and **GEPA** (Generative Evaluation of Pattern Articulation) to be fully functional.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi K-2 iOS App channels *Ok Computer***: The new **Kimi K-2 iOS app** is lauded for its design with an *Ok Computer* theme, with a user requesting [translation support for all languages](https://discord.com/channels/974519864045756446/977259063052234752/1435599575791570995).
   - A user also shared a [link to a presentation](https://3tadkfqfwbshs.ok.kimi.link/present.html) related to the app.
- **Kimi CLI Thinks Interleaved Thoughts**: Support for the **interleaved thinking model** has been implemented in **Kimi CLI**, prompting questions about how it differs from the conventional *think...final answer* method.
   - A member posted that *they added support for thinking mode in kimi-cli yesterday. 👀*
- **Kimi CLI Setup Yields 401 Error**: A user encountered a **401 error** while configuring the **Kimi CLI**, even after reloading credits and verifying their balance.
   - Another member clarified that **credits are for the Moonshot AI Open Platform** and not the **Kimi K-2 platform** and directed the user to the correct channel for assistance.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Perplexity API Joins Aider**: A user requested a tutorial for using the **Perplexity API** with aider, similar to the existing Gemini tutorial, referencing the [aider documentation](https://aider.chat/docs/llms/other.html#other-api-key-variables).
   - A member suggested to replace *gemini* with *perplexity* and setting the API key as an environment variable with `export PERPLEXITYAI_API_KEY=your-api-key` and then trying `aider --list-models perplexity`.
- **OpenRouter emerges as Aider's favorite friend**: A member suggested that **OpenRouter** might be a better choice than **Perplexity API** for coding, highlighting the availability of numerous free and powerful models for testing aider.
   - They mentioned you can try **OpenRouter** since it's *both powerful and FREE for testing aider out*.
- **Users Scripting Custom TDD Loop for Aider**: A user is looking to create a loop with **aider** where agent 1 executes a prompt with **TDD**, agent 2 reviews and suggests improvements, agent 1 executes the improvements, and agent 2 runs TDD tests, fixes bugs, and proposes a commit message.
   - The suggestion was made to use [scripting](https://aider.chat/docs/scripting.html) to wrap the `aider` functionality.
- **Ollama Users Run Into Memory Limitations**: A user wants to summarize rules in the [scalafix-rules](https://github.com/xuwei-k/scalafix-rules/tree/main/rules/src/main/scala/fix) project but is running into memory limitations with **Ollama**.
   - A member suggests using cloud-based models for the summarization task, as a short description per rule isn't hard for local models such as **qwen 30b**, but becomes problematic when processing all rules at once, quickly exhausting memory.
- **Aider Users Request Claude's `/compact` command**: A user inquired if **Aider** has a command similar to **Claude Code's** `/compact` command, which summarizes and condenses conversation history to prevent context loss.
   - The suggestion was made to ask the model to summarise the conversation into a `status_report.md` using the `clear`, `drop`, and `reset` commands.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **IETF Channel Craving Commences**: Members are considering creating a temporary channel for **MCP** folks at **IETF 124** this week, similar to the one for the dev summit.
   - The group agreed with the suggestion and further suggested creating a channel group for *IETF meetings in general*.
- **Events Category is Born**: It was proposed to make an **events** category, and a channel for events that have a critical mass of participants.
   - Some members cited a talk on potential **IETF** adoption of transport protocols for **MCP/A2A** and the relevance of other **IETF** groups (like HTTPAPI) for **MCP**.
- **AI Scrapers Spark Side Session**: Members noted that current discussion in this session is not about **OAuth** at all, but about general **AI scraping/crawlers**.
   - No further details were provided.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Scores of IFeval Probed**: Members examined [IFeval scores](https://openreview.net/forum?id=Q7mLKxQ8qk) and noticed variations at the prompt and instruction levels.
   - A member clarified their method involves using *the average of all scores*, as detailed in the [inspect_evals repo](http://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/ifeval#final-accuracy).
- **Attempt at Latent Reasoning Shared**: A member presented their attempt at latent reasoning, linking to [a tweet](https://fxtwitter.com/RidgerZhu/status/1983732551404679632) and [an Arxiv paper](https://arxiv.org/abs/2510.25741).
   - Details of the approach and its potential applications are outlined in the provided resources.
- **Cheap Concept Detection System Steers Concepts**: A member developed a system for **detecting and steering thousands of concepts** in a model in realtime.
   - They seek prior art on **finite, scalable concept representations** for interpretability.
- **LLMs have Equivalent Linear Mappings**: A paper titled “[Equivalent Linear Mappings of Large Language Models](https://openreview.net/forum?id=oDWbJsIuEp)” demonstrates that **LLMs have equivalent linear representations** of their inference operation for any input sequence.
   - They use the **SVD of the linear representations** to find low-dimensional, interpretable semantic structure that can be used for steering, modular at the layer/block level, and applicable to models like **Qwen 3, Gemma 3, Llama 3, Phi 4, OLMo 2 and Mistral**.
- **Tangent Model Composition Explored**: Members discussed the relevance of [Tangent Model Composition for Ensembling and Continual Fine-tuning](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Tangent_Model_Composition_for_Ensembling_and_Continual_Fine-tuning_ICCV_2023_paper.pdf).
   - It looks at **tangents and Taylor expansions in weight / parameter space**, whereas other work looks at the **Jacobian in input embedding space**.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Text-to-Video Tool Quest Kicks Off**: A member initiated a discussion about accessible **text-to-video tools**.
   - This highlighted interest in new tooling for video generation from text-based prompts.
- **X-Scraping Shenanigans Sparked**: A member sought advice on **web scraping Twitter/X without using an API**, facing maintenance challenges with a cookie-based Python library.
   - The discussion focused on overcoming API limitations for data extraction.
- **Manus Host Services Recommendations Requested**: A member inquired about suitable **host services** for running applications developed with Manus Dev, pointing out Manus' limitations for continuous commercial operations.
   - Another user recommended *Vercel* as a viable hosting solution.
- **Project Publishing Problems Plaguing Platform**: Members reported ongoing issues with **publishing projects on Manus**, particularly concerning updates not reflecting the latest checkpoints.
   - This issue impacts project deployment and version control within the Manus ecosystem.
- **Manus-to-GitHub Migration Methods**: A member asked for advice on migrating projects from **Manus to GitHub**, citing unresolved errors and project setbacks on the Manus platform.
   - This inquiry highlights a need for robust migration strategies due to platform instability.



---



## [Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Launches Codemaps**: Windsurf launched **Codemaps**, powered by **SWE-1.5** and **Sonnet 4.5**, to enhance code comprehension for increased productivity.
   - According to Windsurf, the ability to *understand the code you are working with* is the biggest limit to productivity, detailed in [their X post](https://x.com/windsurf/status/1985757575745593459).
- **Understanding Code is Foundational**: Windsurf stresses that understanding the code is essential for coding effectively with or without AI agents.
   - Referencing Paul Graham, Windsurf highlights that *your code is your understanding of the problem you’re exploring*, emphasizing the importance of understanding the code.



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





### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1435707996083847279)** (1 messages): 

> `VLM OCR, Flash Attention, lms runtime command, MiniMax-M2 tool calling, macOS 26 compatibility` 


- **LM Studio Gets a Speed Boost**: LM Studio **0.3.31** released with **VLM OCR performance improvements**, including a setting to control image input resize (default **2048px**).
   - This release enables **Flash Attention** by default for **CUDA GPUs**, potentially leading to *faster performance and reduced memory footprint*.
- **`lms runtime` Refreshes via Terminal**: The new `lms runtime` command allows updating the runtime from the terminal, demonstrated with `lms runtime update mlx` and `lms runtime update llama.cpp`.
   - A demo video showcasing the new runtime command can be found [here](https://cdn.discordapp.com/attachments/1111797717639901324/1435707995685392524/lms-runtime-demo5.mp4?ex=690cf2c4&is=690ba144&hm=c5767f904197ae88df22f36903f7ed8fac2c46c8404d9a172808772794bd52ef&).
- **LM Studio Extends Toolbelt**: LM Studio **0.3.31** introduces **MiniMax-M2 tool calling support**.
   - This version includes **macOS 26 compatibility fixes**.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1435302608133689394)** (427 messages🔥🔥🔥): 

> `Local AI Coding Models, Renting Servers for Personal Models, Disabling Menu Bar in AppImage, Intel vs AMD for LLM Use, Combining RTX 5090 and 5070Ti` 


- ****Qwen3 Coder BF16 reigns supreme for local AI, but Devstral's niche is Python, according to LM Studio Discord channel!****: According to a LM Studio Discord channel member, [Qwen3 Coder BF16](https://huggingface.co/Qwen/Qwen3-Coder) is the undefeated champion for local coding tasks, though its **60GB** size is a hefty price to pay, however **Devstral** remains a viable option exclusively for Python-centric projects, while [kimi k2](https://huggingface.co/kimik2) requires significant local memory.
   - Another member added that the definition of "best for local" heavily influences model choice, especially considering **Kimi K2**'s substantial memory demands (2TB).
- ****Runpod Renderings reveal costs to Run Personal Models in the cloud****: Members discussed costs for [renting servers](https://www.runpod.io/) to run personal models like **Kimi K2**, estimating around **$35 per hour** for suitable configurations.
   - Options included a **$40/h** option for **7x B200** (1260GB VRAM) and **8x H200** for **$27/h** with 1144GB VRAM, also mentioning a **2 x 8 A100 pod** with 1200GB VRAM as a potentially good deal.
- ****Nvidia Nabs Lead in Capability and Support, AMD's Cheaper Options lure Developers to the Red Team****: A member stated that [Nvidia](https://www.nvidia.com/) is a clear leader in capability and developer/user support.
   - Also noting that AMD is often the cheaper option, Intel does not offer much in AI that is useful in LM Studio.
- ****Linux Looms Larger: LM Studio Lands on penguin distro!****: After a user inquired about [LM Studio](https://lmstudio.ai/) on Linux, another one responded that *"Yeah i've been using LM Studio on Mint the entire time i've been playing with offline inference"
   - The user was extremely excited and said *"Thanks for the heads up on LM Studio and Linux...This has changed my life... lol"*
- ****Pythonistas prefer Python Virtual Environments, and UV is Upping the Ante****: A user inquired to the best way to create a virtual environment in Python
   - Multiple users pointed to [UV](https://astral.sh/uv), calling it *"awesome for managing python virtual environments"*, with one user even claiming that *"you dont even need base python to crate venvs?"* as it *"replaces pip too?"*


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1435293040884449463)** (911 messages🔥🔥🔥): 

> `GPU Cooling Solutions, PCIE Bifurcation Discussions, Overclocking Motherboards, RAM configurations, GPU configurations` 


- **Insane Heat Management Challenges Emerge in Multi-GPU Setups**: Members are [brainstorming cooling solutions](https://discord.com/channels/1153759714082033732/1153759714602164297) for a densely packed PC build, involving vertical **3090s** blowing air directly into the CPU cooler, exploring automotive exhaust shielding, external AIO coolers, and even considering drilling the CPU for better thermals.
   - The discussion highlights the extreme measures required to manage heat in such a configuration, with a touch of humor about potential fire hazards and unconventional modifications.
- **PCIE Bifurcation: When Good Bandwidth Goes Bad**: Users analyzed PCIE lane distribution on motherboards, particularly the [MSI MEG X570 Godlike](https://www.msi.com/Motherboard/MEG-X570-GODLIKE), debating the impact of PCIE bifurcation on GPU performance and exploring options for multi-GPU setups.
   - They weighed the trade-offs between maximizing VRAM through multiple GPUs versus potential performance bottlenecks due to reduced PCIE bandwidth, discussing practical performance drops.
- **Jumping on Overclocking Motherboards**: A member considered upgrading to an [MSI MEG X570 Godlike](https://www.msi.com/Motherboard/MEG-X570-GODLIKE) motherboard, citing its overclocking features and PCIe lane distribution as potential benefits for a multi-GPU setup.
   - Despite the board being overkill for the current setup, its potential for extreme overclocking and expansion capacity made it an attractive option for future upgrades, especially if found at a bargain price.
- **When RAM Errors Give Memes**: A member humorously reported on [PC crashes due to RAM errors](https://discord.com/channels/1153759714082033732/1153759714602164297) while awaiting the delivery of **128GB of new RAM**, turning a frustrating situation into a meme-worthy event.
   - The member embraced the chaos, expressing anticipation for the arrival of the new RAM and joking about making Jeff Bezos proud.
- **Selegiline Steals the Show**: Members discussed the use of [Latuda + Vyvanse](https://en.wikipedia.org/wiki/Selegiline), and are absolutely loving the combo (not to abuse), but to be genuinely functional.
   - The members had fun going back and forth on the benefits of the drug and what it means for functionality.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1435282559138398208)** (1121 messages🔥🔥🔥): 

> `OpenAI deals, Google AI future, LMArena trust, Sora 2 credits, Claude 3.5 sonnet` 


- **OpenAI gets deal fever**: Members expressed that *all companies are dying to strike deals with OpenAI*
   - One member derisively reacted to this claim, stating that *people who say that google is the future of AI are delusional*.
- **Google has less AI cash on hand than OpenAI!**: One member asserted that *google actually has significantly less money to burn on AI than openai* and cannot spend all of its budget on AI.
   - Countering, another member noted that **Google's market cap is larger than many countries' budgets** and has more money than OpenAI.
- **Chrome and YouTube Valuations Spark Debate**: In a discussion about Google's financial standing, members cited that Google Chrome, as of late 2024/2025, holds a dominant web browser share and is valued between **$20 billion and $50 billion**.
   - Also, it's worth noting that [YouTube](https://www.youtube.com/) as a standalone business, could be valued between **$475 billion and $550 billion**, based on revenue and market position.
- **Model Hallucinations Run Wild**: Members reported that AI models are hallucinating and making false identity claims with Claude 4.5 claiming 3.5 while other models like Qwen and Deepseek also falsely identifying as Claude-Sonnet-3.5.
   - The mods responded that *the models used are the same models provided by the model providers API* but they did acknowledge there was something to look into and said *this question is something we can do a better job of answering, as it is common. I'll flag to the team. A blog post may be helpful.*
- **Blackhawk the Alt-Right Truth Parrot**: Some members are testing the new model Blackhawk, describing it as uncensored, delusional, and prone to swearing.
   - One member said Blackhawk is not very smart and gives *non-mainstream (moderate to extreme right) perspective on socially and politically sensitive topics*.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1435714040126111977)** (1 messages): 

> `Arena Expert Leaderboard, Occupational Leaderboards, Expert Prompt Dataset` 


- **Arena Expert Tagging System Debuts**: A new tagging system based on an evaluation framework has been introduced to identify expert-level prompts from the community, leading to the creation of an [Expert Leaderboard](http://lmarena.ai/leaderboard/text/expert) that highlights prompt depth, reasoning, and specificity.
   - This system aims to reveal the structure of prompts, emphasizing the clarity of evaluation, as detailed in the [research blog post](https://news.lmarena.ai/arena-expert/).
- **Occupational Leaderboards Go Live**: **Occupational Leaderboards** are now available, mapping prompts to real-world domains across **23 occupational fields**, showcasing the full spectrum of real-world reasoning tasks, with **8 leaderboards** currently live.
   - These include [Software & IT Services](https://lmarena.ai/leaderboard/text/industry-software-and-it-services), [Writing, Literature, & Language](https://lmarena.ai/leaderboard/text/industry-writing-and-literature-and-language), [Life, Physical, & Social Science](https://lmarena.ai/leaderboard/text/industry-life-and-physical-and-social-science), [Entertainment, Sports, & Media](https://lmarena.ai/leaderboard/text/industry-entertainment-and-sports-and-media), and [Business, Management, & Financial Ops](https://lmarena.ai/leaderboard/text/industry-business-and-management-and-financial-operations), [Mathematical](https://lmarena.ai/leaderboard/text/industry-mathematical), [Legal & Government](https://lmarena.ai/leaderboard/text/industry-legal-and-government), [Medicine & Healthcare](https://lmarena.ai/leaderboard/text/industry-medicine-and-healthcare).
- **Expert Prompt Dataset Now Open**: An open dataset of expert prompts with occupational tags is now available on [Hugging Face](https://huggingface.co/datasets/lmarena-ai/arena-expert-5k), providing resources for further analysis and development.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1435283079085162688)** (947 messages🔥🔥🔥): 

> `Comet referral program, Perplexity Pro chat history, Student subscription on active operator sub, Ethernet cable recommendations, Airtel subscription offer` 


- **Referral Program Still Live in USA?**: A member asked if the Comet referral program is still open for partners in the USA, and whether referred users need a paid subscription to qualify for the referral bonus.
   - Another member responded that the referral program is active in the US, but Perplexity takes time to verify referrals and process payouts.
- **Losing Chat History?**: One user reported difficulties accessing their complete chat history with Perplexity Pro, and expressed frustration with the inability to retrieve older conversations, but no resolution was found in the discussion.
   - The user noted that they could only see about 30 lines of questions.
- **Ethernet cable failure leads to slow speeds**: A user reported their **Cat6 ethernet cable broke**, leading to speeds of only 300kb/s, and sought recommendations for a replacement.
   - Another user suggested *literally anything Cat6* will work, mentioning they typically buy Ugreen products based on a previous Perplexity recommendation.
- **GPT Model Mix-up?**: A user expressed frustration that selecting specific models like Claude Sonnet 4.5 or Gemini 2.5 Pro sometimes yields responses from lower-quality models like **Haiku** or **Gemini 2 Flash**.
   - This led to a discussion about whether Perplexity is intentionally using cheaper models to cut costs, with some users inspecting network requests to confirm the model being used.
- **Comet Browser Ads are back**: Users reported the reappearance of **YouTube ads** in the Comet browser, despite its ad-blocking capabilities.
   - Members speculated that YouTube might be tweaking its ad service to bypass ad blockers, impacting Comet's functionality.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1435540437355200603)** (2 messages): 

> `Spotify Song Sharing, Shareable Threads` 


- **User Shares Spotify Song**: A user shared a [Spotify link](https://open.spotify.com/track/5PjC1JmXRAOwJtFLrdaN4A?si=46ad168539424bd1) to a song.
- **Reminder to Keep Threads Shareable**: A message reminded users to ensure their threads are set to `Shareable`.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1435384158951313508)** (3 messages): 

> `Perplexity Pro, Tool calling errors with sonar-pro, Tool calling with Perplexity API` 


- **Perplexity Pro Users Report API Issues**: A user reported getting a tool calling error (*Tool calling is not supported for this model*) when running the [full implementation code](https://docs.perplexity.ai/guides/chat-completions-guide) from the Perplexity documentation.
   - The error occurs when using *sonar-pro* as a tool-callable model, suggesting a possible discrepancy between documentation and actual API capabilities.
- **Confusion about Tool Calling Support in Perplexity API**: A user expressed confusion regarding the availability of tool calling in the Perplexity API, particularly after referencing an OpenAI implementation.
   - The user's goal is to integrate tool calling into their use case, but the error message indicates that the specified model (*sonar-pro*) might not support this feature as expected.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1435282873681838090)** (554 messages🔥🔥🔥): 

> `Tailwind 4, Nuxt 4, Phantom wallets, exoudos wallets, Specstory extension` 


- **Nuxt & Tailwind Get New Rules**: After upgrading to **Tailwind 4** and **Nuxt 4**, users created [new rules](https://link.to/rules) for **Cursor**.
   - One member said *Context7 MCP is doing just great* and it helped refactor the project to use **Tailwind 4**.
- **Specstory Extension Fails to Save**: A user had issues with the **Specstory extension** failing to [save chats](https://link.to/chats) despite setting up auto-save, and the extension displayed an *e.setData is not a function* error.
   - Another member found that chats made in an empty folder cannot be saved and that *you need to have a folder open*.
- **Cursor and GLM: a budding bromance?**: Users noticed **GLM**'s developer docs now include instructions for [integrating with Cursor](https://link.to/glm-integration).
   - A member wondered if it has something to do with **Composer** and **GLM**, or perhaps they are collaborating.
- **Syntax Highlighting Breaks in Chat**: A user reported that syntax highlighting in the chat was broken in version **2.0.38**, with code appearing all white except for red curly brackets, they posted about [this on the forum](https://link.to/forum-post).
   - Another user suggested adding code blocks with language name in the backticks to fix the issue.
- **Excessive Token Usage**: Some users have been experiencing unexpectedly high token usage, with input costs reaching **100k-300k** even for short prompts, they suspected that input cost also takes into account the read/code changes made by the AI.
   - The grep command was found to cause a large context ingestion because of how it is implemented, the agent could ingest a large file if it is used in the prompt.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1435323712630161549)** (9 messages🔥): 

> `Mobile Web UI Crashing, Background Agents Use Case, Bug with images in prompts, Diff Rendering Improvements, API Endpoints for Cloud Agent` 


- ****Mobile Web UI Suffers Crashes with Large Diffs****: Users report the mobile web UI crashes with large diffs, making it unusable for reviewing chat and responding, especially frustrating given mobile is the primary use case for **background agents**.
   - The user expressed that *the experience sucks and now went from barely tolerable to completely unusable*.
- ****Looking Into Cursor Bug with Images in Prompts****: A user reported a bug with images in prompts ([https://forum.cursor.com/t/getting-internel-error-on-cursor-com-for-prompts-with-images/139074/1](https://forum.cursor.com/t/getting-internel-error-on-cursor-com-for-prompts-with-images/139074/1)) that prevents using background agents for a project.
   - A fix was pushed to mobile diffs and should now run smoother, except for the **composer-1 model**.
- ****Mobile Diffs Get Smoother****: A fix was pushed for mobile diffs and now things should run smoother, however the overall rendering will change to be more performant.
   - Further details remain scarce.
- ****Cloud Agent API Endpoints Shared****: To help with background agents, a member shared the Cloud Agent **API endpoints**:
   - [Agent Conversation](https://cursor.com/docs/cloud-agent/api/endpoints#agent-conversation), [Agent Status](https://cursor.com/docs/cloud-agent/api/endpoints#agent-status), and [Webhooks](https://cursor.com/docs/cloud-agent/api/webhooks).


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1435287778484293656)** (203 messages🔥🔥): 

> `DeepSeek-OCR notebook, Q4 Ks-XL vs IQ4, TRL notebooks vs Unsloth notebooks, fine tune a cross encoder or embedding model, Qwen3VL-30B-A3B on 16GB vram` 


- **Unsloth Releases DeepSeek-OCR Fine-Tuning Notebook**: Unsloth AI announced a new DeepSeek-OCR fine-tuning notebook, available at [this link](https://x.com/UnslothAI/status/1985728926556307471).
   - Some users express concern over error rates above 100%, possibly due to differences in predicted and actual text length.
- **Q4 Ks-XL vs IQ4: a Quantization Quandary**: A user inquired about the major differences between **Q4 Ks-XL** and **IQ4** quantization processes, seeking to optimize RAM usage for agent workflows.
   - The user aimed to determine if saving RAM for more context was worth using a slightly larger model.
- **Unsloth's optimizations and Speed Boosts for TRL**: When asked the differences between using **TRL notebooks** vs **Unsloth notebooks**, users responded that Unsloth offers *speed increases and lower memory usage* due to its own optimizations.
   - It was explained that *Unsloth patches TRL and Transformers to add its own optimization and some techniques to lower VRAM*.
- **Qwen3VL-30B-A3B is too big for 16GB VRAM**: A user asked if it was possible to run **Qwen3VL-30B-A3B** on a 16GB VRAM GPU.
   - Another user responded that *it's not* possible as it needs around 17.5GB VRAM, <:rip:1233329793584468062>.
- **Vision Model Training Troubleshooter**: A user reported issues with Qwen3, experiencing low training loss and high validation loss, even with dataset examples being incorrectly answered.
   - The user suspected a masking issue or a discrepancy in eval and train loss calculation, noting loss variations with batch size.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1435309660361064488)** (4 messages): 

> `Blockchain, AI, Trust in Code, Consensus Mechanisms` 


- **Building Trust with Blockchain and AI**: A member expressed interest in building trust into code and teaching machines to think, leading them to explore **blockchain** and **AI**.
   - They've worked on systems that make *consensus feel real and reliable*, and see AI as tools that can solve previously impossible problems.
- **Blockchain and AI as Transformative Technologies**: The member believes that **blockchain** and **AI**, when used correctly, can revolutionize industries and communities.
   - These technologies can also foster new ideas and change how things operate, connecting people and solving problems.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1435326858379006063)** (196 messages🔥🔥): 

> `SFT as RL, Quantum Randomness, Manual Data Entry, Nvidia Blackwell Pro 4500 vs 5090, ECC Memory Value` 


- **SFT is RL with single token rollouts**: A member stated that [Supervised Fine Tuning (SFT)](https://en.wikipedia.org/wiki/Supervised_learning) is just reinforcement learning (**RL**) with single token rollouts.
   - Another member elaborated that the policy takes one action, rewards the policy according to the likelihood of that action vs that of the ground truth, and repeats.
- **Data entry jobs - literally the worst**: Members shared anecdotes from data entry jobs in banks, highlighting manual keying, double confirmation by seniors, and tracking KPIs with counter programs.
   - One member mentioned needing to press a **+1 button** on a counter program every time they completed one, and pressing other buttons to signify they are going on break and concluded they *can't be living like that for the rest of my life*.
- **Nvidia Pro 4500 surfaces as Blackwell card**: The **Nvidia Pro 4500** was revealed as an actual Blackwell-based workstation card, with **32GB** of memory and **200W** power consumption.
   - Discussion ensued about its positioning relative to the **5090**, with the 5090 offering **2x** the memory bandwidth.
- **ECC Memory debate heats up**: The value of **ECC memory** was debated, with one member strongly advocating for it due to its ability to prevent bitflips in long-running tasks like fluid dynamics simulations, which can run for weeks.
   - He said *if you have a bitflip you start over* and that's *why you have ECC - it does mem parity in hardware*.
- **GTC conference value is questioned**: The value of attending **Nvidia's GTC conference** was questioned, with one member noting that this year's event was overcrowded and focused more on company services than new research.
   - Another member recalled GTC24 being amazing because *the authors of attention is all you need did a panel together for first time and i got to be front row*, whereas this year *it really felt like you were witnessing some insane stuff*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1435291863656566927)** (79 messages🔥🔥): 

> `MiniMax M2 local inference, GPT-OSS-20B training with Unsloth + REINFORCE, System prompt usage during finetuning, Multilingual reranker with GGUF and llama.cpp, Granite 4.0 Hybrid 4-bit conversion issues` 


- ****MiniMax M2 on 5090 Needs Quantization Guidance****: A user with a **5090** and **128GB RAM** seeks advice on running [MiniMax-M2-GGUF](https://huggingface.co/unsloth/MiniMax-M2-GGUF/discussions/4) locally, considering **4-bit quantization** and **MXFP4** versions, alongside software and UI recommendations for agentic tasks.
   - They're also interested in exploring **3-bit models** with an *accuracy recovery adapter* ([Reddit link](https://www.reddit.com/r/LocalLLaMA/comments/1mytbfz/accuracy_recovery_adapter_with_selfgenerated_data/)).
- ****REINFORCE Training Reimagined via SFT Trainer****: A user explored training **gpt-oss-20b** with Unsloth and vanilla **REINFORCE**, encountering issues with TRL's **RLOO** requiring *at least 2 generations per prompt*.
   - They're now attempting to mimic **REINFORCE** using the **SFT trainer** as a workaround.
- ****System Prompt's Sweet Spot During Finetuning****: During finetuning, a user questioned whether to forgo the system prompt or shorten it since it should be *baked* into the smaller model.
   - Responses suggest it depends on task complexity: ditch it for obvious tasks like translation, keep it for multi-step processes.
- ****Qwen Rerankers Rise in Llama.cpp Landscape****: Users are seeking a good **multilingual reranker** that works with **GGUF** and **llama.cpp**.
   - It was suggested that **Qwen** has a set of rerankers that work with llama.cpp and are multilingual, but their quality needs assessment, and their embedding models initially had issues.
- ****Granite's 4-bit Dreams Dashed by Precision Predicaments****: A user faced problems converting **ibm-granite/granite-4.0-h-tiny-base** to **4-bit safetensors** for training, even with Unsloth's method, as the model saved in **16-bit precision** regardless.
   - Unsloth AI has [issue #3558](https://github.com/unslothai/unsloth/issues/3558) and [issue #3550](https://github.com/unslothai/unsloth/issues/3550) related to granite models and this problem.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1435375994247712981)** (9 messages🔥): 

> `Roblox PII Classifier, Open Sourcing Safety Tools, PII Dataset Access` 


- **Roblox Open Sources PII Classifier**: Roblox open-sourced their **PII Classifier AI**, a tool designed for detecting Personally Identifiable Information (PII) in chat, available on [Hugging Face](https://huggingface.co/Roblox/roblox-pii-classifier) and detailed in their [news release](https://corp.roblox.com/newsroom/2025/11/open-sourcing-roblox-pii-classifier-ai-pii-detection-chat).
   - The safety team is proud of this powerful tool for keeping kids safe online, highlighting its scalability with an average of **6.1 billion chat messages** processed daily at a peak of **200,000 queries per second** with **<100ms P90 latency**.
- **Roblox's Public Safety Models Spark Interest**: Interest was expressed in Roblox's public safety models, particularly in the challenges of managing schedulers at their scale, where most time may be spent in **vllm batching** rather than **GPU forward passes**.
   - Despite appreciating the specialized model, there was a strong interest in the dataset itself, though acknowledging its inaccessibility due to **PII concerns**.
- **Dataset Accessibility Concerns Acknowledged**: Access to the **PII-filled dataset** is heavily restricted, as direct inquiries about it were met with the implicit understanding that it contains sensitive information.
   - The necessity of **serious clearance** to even glance at the dataset was emphasized, reinforcing the stringent privacy measures in place.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1435348926663491624)** (52 messages🔥): 

> `Youtube lectures, B770 GPU, 3080 modding, Auto Vectorization Compiler Blogpost` 


- **Lectures before December: Confirmed!**: A user inquired about lectures before December and another member pointed to the [event listings](https://discord.com/channels/1189498204333543425/1191300313928433664/1435350172992540793).
   - A **YouTube** video about similar lectures was also shared: [link](https://www.youtube.com/watch?v=nFfcFyBEp7Y).
- **Users Speculate on B770 Performance and GPU Modding**: A user expressed desire for a **B770 GPU** and considered creating one by modding a **3080**, aiming to achieve performance similar to a **4070**.
   - Another user noted the theoretical possibility of adding **20GB of VRAM** but acknowledged financial limitations. They pointed others to a [gpu modding channel](https://discord.com/channels/1189498204333543425/1349152646484987974).
- **Auto Vectorization Compiler Blogpost Influences Discord**: A member mentioned reading blog posts, specifically referencing **ThunderKittens'** content, including one on automatic vectorization, which has been influential.
   - The *auto vectorization* blogpost was shared with the user by another member.


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1435313453286818084)** (1 messages): 

> `Community Meetup, Timezones, Meeting Details` 


- **Community Meetup Scheduled**: A community meetup is scheduled for tomorrow from **10am-11am PST**.
   - See the original post for meeting details.
- **Timezone Reminders**: The meetup time is **10am-11am PST**, which may need to be converted for different timezones.
   - Participants are encouraged to check what time that is for them.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1435297826249900266)** (5 messages): 

> `Memory-Bound Matmuls, SM count impact on Matmul latency, Saturating Memory Bandwidth on GPUs, Little's Law relevance to GPU performance` 


- **Memory-Bound Matmuls debated in CUDA**: A member inquired why all **SMs** are needed for optimal latency in memory-bound matmuls, questioning if a few **SMs** could saturate memory bandwidth.
   - Another member questioned if matmuls are memory-bound, and claimed that *it takes more than a few SM's to saturate the memory BW*.
- **SM Count for Memory Bandwidth Saturation questioned**: A member stated that decode matmuls are memory-bound, estimating that **30%-40%** of **SMs** are sufficient to saturate memory bandwidth.
   - They also shared an [image](https://cdn.discordapp.com/attachments/1189607726595194971/1435489664135069706/image.png?ex=690cd02e&is=690b7eae&hm=331e3b1f0261ac9725ecd33df2daa3ff5be74db0c6143852f2b1ee26006db94c&) related to their question.
- **HBM Bandwidth Saturation Challenges on New GPUs**: A member mentioned that achieving full memory bandwidth on newer **HBM-based GPUs** can be tricky, even with all **SMs**.
   - They suggested using **Little's Law** to understand this, linking to an NVIDIA session: [GTC25-S72683](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72683/).


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1435292929722941450)** (19 messages🔥): 

> `torch.compile CUDA graph recapture, torch.compile grouped_mm UserWarning, vLLM pytorch dependencies, Float8 tensors limitations, custom kernel opcheck failure` 


- **Debugging Torch Compile CUDA Graph Recapture**: A user seeks advice on debugging **CUDA graph recapture** with `torch.compile` in `max-autotune` mode, aiming to ensure graph stability after warmup and using `torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = 1` to detect recaptures.
   - One member suggested checking for **multiple versions of torch or vLLM** that might be interfering.
- **Torch Compile Grouped MM UserWarning Outbreak**: Users report a flood of annoying `UserWarning` messages related to **torch.compile + grouped_mm**, specifically regarding *logical operators 'and' and 'or' deprecated for non-scalar tensors*.
   - A member indicated [they fixed this for flex](https://github.com/pytorch/pytorch/issues/167041) and can apply the same fix for grouped GEMM, and another user opened an official issue.
- **vLLM Hardcodes PyTorch and Triton Dependencies**: A user noted that **vLLM** hardcodes dependencies like [these](https://github.com/vllm-project/vllm/blob/0976711f3b569aae4a8c9ac148f0771624293120/requirements/cuda.txt#L13), creating difficulties when using custom PyTorch/Triton versions.
   - They further mentioned that prior to v0.10.1, building was comparatively easier.
- **Float8 Tensors Trigger NotImplementedError**: A user encountered a `NotImplementedError` for *"mul_cuda" not implemented for 'Float8_e4m3fn'* during `opcheck`, when running a custom kernel.
   - Another member clarified that low-bit dtypes are representations and suggested using **scaled mm** for computations, noting that a fix related to this might have landed.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

marksaroufim: https://xillybus.com/tutorials/pci-express-tlp-pcie-primer-tutorial-guide-1
  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1435404948530794598)** (1 messages): 

> `Mixlayer, AI inference platform, Rust, CUDA, founding engineer` 


- **Mixlayer Seeks Founding Engineer!**: The founder of [Mixlayer](https://mixlayer.com), an **AI inference platform** for power users, is looking to hire a founding engineer familiar with **Rust** and **CUDA**.
   - The engineer will work on their custom inference engine; hybrid in **SF** is preferable but they are open to remote.
- **Mixlayer - AI Inference Platform**: Mixlayer is an **AI inference platform** designed for power users, providing developers with low-level access to open source LLMs to enhance product development.
   - The platform focuses on enabling developers to build better products by giving them access to underlying LLMs.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1435297205366816807)** (16 messages🔥): 

> `RL bug on accumulator type fixed at fp32, Practice Deshourading, Hackathon, PyTorch/vllm on AMD AI PCs` 


- **RL Accumulator Bug causes Rounding Error**: A user discovered an **RL bug** where the accumulator type is fixed at **fp32** even when **bfloat16** is used, causing a rounding error because the computed intermediate value is stored in **fp32**.
   - The user questioned why it was fixed at **fp32** only, and how bad the issue is.
- **Deshourading Practice makes perfect!**: A user is looking for a good way to practice **Deshourading** because GPUs are expensive.
   - Another member encouraged them to keep trying, mentioning that *there is a bunch of literature and resources out there*.
- **Hackathon attracts newcomers**: Several users mentioned joining the channel after hearing about the hackathon.
   - One user from India inquired about eligibility, while another user encouraged everyone to participate, emphasizing the learning opportunity, stating *atleast in the mist of the competition you will learn how to do basics.*
- **PyTorch/vllm gets AMD AI PCs working**: A user inquired about getting **PyTorch/vllm** to work on **AMD AI PCs**.
   - They mentioned trying various docker variations and the therock repo, but couldn't get **PyTorch** to recognize the **APU** and requested pointers.


  

---


### **GPU MODE ▷ #[jax-pallas-mosaic](https://discord.com/channels/1189498204333543425/1203956655570817034/1435698383091404900)** (1 messages): 

> `Mosaic-GPU, all-gather-matmul` 


- **Code request for Mosaic-GPU's all-gather-matmul**: A member inquired about the availability of the full code for the **all-gather-matmul** within the **mosaic-gpu** project.
   - The member specifically mentioned the [JAX documentation page for collective_matmul](https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html) as a reference.
- **Mosaic-GPU all-gather-matmul location remains elusive**: The location of the concrete implementation of **all-gather-matmul** within mosaic-gpu remains unclear.
   - Despite searching, the member hasn't been able to locate a direct link to the source code.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1435305338302890085)** (7 messages): 

> `OSS Contribution, fbgemm kernels, fp8 weight-only pattern & torch.compile` 


- **First OSS Contribution Attempt Incoming**: A member asked if they could attempt a PR fix, stating *it will be my first proper contribution to OSS lol*.
   - Another member replied *sure, which kernel did you have in mind?*
- **Kernel Codebase Analysis in Progress**: One member stated that the codebase mainly reuses **fbgemm kernels** and that they need to check if this is the full solution.
   - They added that they are *pretty new to the code essentially*.
- **fp8 weight-only Pattern Compatibility Questioned**: A member inquired whether **fp8 weight-only pattern** should be supported by **torch.compile**.
   - Another member responded that using **Float8WeightOnlyConfig** with **torch.compile** *should work*.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1435290643319488562)** (3 messages): 

> `Disaggregated Inference Retrospective, Symbolica AI Hackathon` 


- **Hao Lab Retrospects on Disaggregated Inference**: Hao Lab posted a retrospective blog on **Disaggregated Inference** after 18 months, available at [this link](https://x.com/haoailab/status/1985753711344316648).
- **Rustaceans unite for Symbolica AI Hackathon**: Symbolica AI is hosting a hackathon in San Francisco on **Saturday, November 8th**, for Rust developers interested in **formal logic**, **automated theorem proving**, **types**, **compilers**, and **AI**; RSVP at [Luma](https://luma.com/1xa9d6nr?utm_source=meetup).


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1435423285356658819)** (11 messages🔥): 

> `vectoradd_v2, grayscale_v2, vectorsum_v2, A100, B200` 


- **Vector Addition Victorious on Various Venues**: A member achieved **first place** on **A100** for `vectoradd_v2` with **896 µs** and another member achieved **first place** on **H100** with **525 µs**.
   - Another member achieved **third place** on **B200** with **237 µs** for `vectoradd_v2`.
- **GrayScale Gains Ground, Glowing on GPUs**: A member reached **first place** on **H100** for `grayscale_v2` with **1369 µs** and later grabbed **first place** on **B200** with **600 µs**.
   - The same member also secured **second place** on **A100** with **2.39 ms** for `grayscale_v2`.
- **Vector Sum Soars, Secures Second Spot**: A member earned **second place** on both **B200 (119 µs)** and **H100 (126 µs)** for `vectorsum_v2`.
   - The member also clinched **first place** on **L4** with **918 µs** for `vectorsum_v2`.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1435477572631924919)** (5 messages): 

> `Factorio RCON, Factorio Hidden Settings, Factorio multiplayer server` 


- ****Factorio's RCON Access Exposed****: Running the Docker server isn't needed for **RCON** access, according to a user; in the Factorio client, pressing **Control + Option** while clicking the main menu settings on macOS reveals a "The Rest" option for hidden settings to add **RCON** port and password for **FLE**.
   - Another user confirmed that the above trick also works on regular single player games too.
- ****Factorio Client Hosts Local RCON Server****: Launching/hosting a multiplayer server from the Factorio client will run **RCON** locally on that port, which is useful for development.
   - A user confirmed this is useful and thanked the original poster for sharing, and offered to chat more about the documentation tomorrow.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1435444018564562944)** (30 messages🔥): 

> `Node allocation, Solution sharing, Competition submissions visibility, Ranking issues` 


- **Node Allocation Clarification**: Users discussed the allocation of **8 nodes** and whether there should be **8 runners**.
   - It was clarified that the runtime of **96 hours** over the whole day includes overhead, so the number of active execution time will be higher.
- **GPU code Sharing Culture Blooming**: A user inquired about the culture of sharing solutions in GPU programming, similar to Competitive Programming (CP) circles.
   - Another member expressed hope for a platform to facilitate solution sharing, noting that GPU programming competitions are new, so solution sharing is not really a thing.
- **GPU Gems Shared on Github**: Team Gau shared their kernel at [gpu-mode-kernels](https://github.com/gau-nernst/gpu-mode-kernels/tree/main/amd-distributed/all2all), and a brief writeup about [optimizing distributed inference kernels](https://www.yottalabs.ai/post/optimizing-distributed-inference-kernels-for-amd-developer-challenge-2025) for the AMD developer challenge 2025.
   - They promised a more detailed explanation later.
- **Submissions Visible for Coding**: Users can now see submission code from finished leaderboards on the web after logging in with their Discord account, as highlighted in this [announcement](https://cdn.discordapp.com/attachments/1359640791525490768/1435763582078947369/image.png?ex=690d2689&is=690bd509&hm=3430a612bb2eaed1dc1cfa2b90d9ebe5b21db0097037b20aa40a29ba4f139d55).
   - All submissions for the first competition can be found at [huggingface.co](https://huggingface.co/datasets/GPUMODE/kernelbot-data).
- **Ranking Revision Requested**: A user requested a ranking fix for the **amd-all2all** competition, claiming their **216us** solution was deemed illegal shortly before the deadline.
   - They highlighted that their final submission only achieved **263us**, and asked for acknowledgement of the 216us submission as the true winning result.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1435512487117328504)** (4 messages): 

> `CuTeDSL Resources, CuTe copy threads, CuTeDSL Sum Reduction Kernel` 


- ****CuTeDSL** Resource Roundup Requested**: A member requested resources and tutorials for learning **CuTeDSL**, specifically seeking info on using **nvfp4** within **CuTeDSL**.
   - No specific resources or tutorials were provided in the given messages.
- ****CuTe** Auto Thread Copy Queries**: A user asked whether **CuTe** automatically elects threads when copying, particularly if there's an internal `if` when launching a `copy(cpy, src, dst)` with a warp but `size(cpy)<32`.
- ****CuTeDSL** Sum Reduction Kernel Troubleshoots**: A user sought assistance with a **sum reduction kernel** in **CuTeDSL**, reporting an inability to reduce the summation across blocks, and provided a [relevant code snippet](https://gist.github.com/kaiyuyue/c4a18ca59c3c63a2b8009704a9b7496b).


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1435304405107998730)** (4 messages): 

> `Mojo GPU Puzzles, Layout API in Mojo, Mojo version compatibility` 


- **Mojo GPU Puzzles Tutorial Series Launches!**: A member has created a [video tutorial series](https://www.youtube.com/watch?v=-VsP4kT6DjA) to accompany the **Mojo GPU Puzzles**, with the first two entries released today.
- **Layout API Debut in Mojo v25.1**: The functions `Layout.row_major` and `Layout.col_major` were introduced in **Mojo version 25.1**, as part of the layout package (released Feb 13, 2025) for describing tensor organization.
   - A member is troubleshooting an *error: no matching function in call to 'row_major'* in [leetgpu.com](https://leetgpu.com), which seems to be running **Mojo 25.4**.


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1435286769858773052)** (8 messages🔥): 

> `picograd commits, fuzzing against np,torch, and tinygrad, pedagogical progression, kernels vs compilers` 


- **Commits piling up in Picograd**: Multiple commits were made to the [picograd repo](https://github.com/j4orz/picograd), including [b97b9ea](https://github.com/j4orz/picograd/commit/b97b9ea0eda2282bb5e193558c370c53345f07d9), [43796e0](https://github.com/j4orz/picograd/commit/43796e049eb225f9c2dd093a72ccfa09f237db09), [ae47d4d](https://github.com/j4orz/picograd/commit/ae47d4d72f0757b8e542e6b923ca910a7ae56ecc), [625d6ac](https://github.com/j4orz/picograd/commit/625d6acb6cd9395010024e5148886ddb34d7a563), [b9fbf87](https://github.com/j4orz/picograd/commit/b9fbf879b4030688a357402111f77ed12ffd01a9), [90da2f0](https://github.com/j4orz/picograd/commit/90da2f02cc64f4364de08255a6555f36b8e2e019), and [46eb304](https://github.com/j4orz/picograd/commit/46eb304415a21e7a542fa7151508067e7b56f514).
   - The changes are currently unknown.
- **Fuzzing Frenzy**: A member asked if anyone is interested in *looking into fuzzing against np,torch, and tinygrad?*
   - No one responded to their request yet.
- **Pedagogical Progression Paradigm**: It was announced that the pedagogical progression will have **3 modes**: **EAGER_NAIVE=1**, **EAGER_RUNTIME=1**, and **GRAPH=1**.
   - The first launches kernels per op, so no copy-free views, while the second will have a managed runtime like `torch.Storage` and `tinygrad.Buffer`.
- **Kernels versus Compilers**: A member loved the discussion on [kernels vs compilers](https://www.youtube.com/watch?v=Iw4xKHPl7hI) with great takes from mostafa and simran.
   - The member thanked everyone for the great questions.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1435527133878292593)** (3 messages): 

> `CUDA, Triton` 


- **Inline CUDA struggles**: A member noted they use inline CUDA via `load_inline` but sometimes feel like they are swimming upstream.
   - They guessed that most people use one of the python DSLs like **Triton**.
- **Triton simplicity with Perf left on the table**: One member stated that *triton* is super easy to use but probably perf left on the table.


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1435385420115476642)** (11 messages🔥): 

> `DeepSeek FP8, Cutlass FP8 GEMM, FP8 Blockwise Training, Benchmarking Scripts, Blockwise Quantization Kernels` 


- **DeepSeek's FP8 Implementation Gets Implementation**: A user pointed out a [good first issue](https://github.com/pytorch/ao/issues/3290) for implementing **DeepSeek-style FP8 blockwise training** in PyTorch's `ao` repository.
- **Cutlass Already Has DeepSeek FP8 GEMM?**: A user mentioned that [CUTLASS](https://developer.nvidia.com/cutlass) already has some **DeepSeek FP8 GEMM implementations**, providing links to examples such as [Hopper FP8 warp-specialized GEMM with blockwise scaling](https://github.com/NVIDIA/cutlass/tree/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling) and [grouped GEMM with blockwise scaling](https://github.com/NVIDIA/cutlass/tree/main/examples/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling).
- **Benchmarking Blockwise FP8 Gets a Volunteer**: A user volunteered to set up **benchmark scripts** for the **quantization** and **GEMM kernels** related to **DeepSeek FP8**, with a follow-up on optimizations, referencing existing implementations of blockwise FP8 GEMM and quantization [here](https://github.com/pytorch/ao/blob/main/torchao/kernel/blockwise_quantization.py).
- **DeepSeek FP8 Benchmarking Framework is Suggested**: A user asked about basing the benchmarking on [this framework](https://github.com/pytorch/ao/blob/main/benchmarks/benchmark_blockwise_scaled_linear_triton.py).
- **DeepSeek FP8 Triton Kernels Need Benchmarking**: A user plans to create new **bench scripts** for five specific kernels from `blockwise_fp8_training/kernels.py`, including `triton_fp8_blockwise_act_quant_lhs`, `triton_fp8_blockwise_act_quant_rhs`, `triton_fp8_blockwise_act_quant_transposed_lhs`, `triton_fp8_blockwise_weight_quant_rhs`, and `triton_fp8_blockwise_weight_quant_transposed_rhs`.


  

---


### **GPU MODE ▷ #[opencl-vulkan](https://discord.com/channels/1189498204333543425/1418990184367919267/1435443066075877517)** (6 messages): 

> `GLSL Vulkan Compute Shaders, clspv, Clvk, Slang shading language` 


- **Ditching GLSL for Vulkan Compute**: A member prefers sticking to **GLSL Vulkan Compute Shaders** for targeting Vulkan, but is not a fan of **clspv**.
   - They mentioned **clspv** and **Clvk** are developed separately without a release process, needing builds from the master branch, leading to a preference for **GLSL** or **Slang**.
- **Slang Shading Language: Cool but Redundant?**: According to this member **Slang** is the new cool shading language, which is a huge upgrade from **GLSL**, especially if it's compute focused.
   - They added, *the world didn't need another C derivative which has 4 ways of writing the same concept*.


  

---


### **GPU MODE ▷ #[cluster-management](https://discord.com/channels/1189498204333543425/1420098114076803142/1435655976605847736)** (1 messages): 

> `Ansible Scripts, Configuration Management` 


- **Ansible Scripts Correct Misconfigurations**: Members mentioned that one can use **Ansible scripts** for **ulimits** and similar configuration to correct misconfigurations.
   - They also make it easy to see if there were any.
- **Ansible Configuration Benefits**: Using **Ansible** provides a clear and auditable method for managing system configurations like **ulimits**.
   - The scripts not only rectify existing issues but also offer a transparent overview of past configurations.


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1435347373776703630)** (4 messages): 

> `inline_triton, Helion Compiler, atomic_cas, output_like=None, Helion API` 


- **Inline Triton troubles need workarounds**: Members discussed workarounds to get `inline_triton` to work, such as adding a dummy value in the last line or setting `output_like=None`, issue created [here](https://github.com/pytorch/helion/issues/1086).
- **`output_like=None` fails edge cases**: Setting `output_like=None` still requires the last line to be an expression and a workaround was needed with dummy value.
   - A member provided example helion code using `hl.inline_triton` with a `while` loop and `atomic_cas` function, noting the error `helion.exc.InvalidAPIUsage: Invalid usage of Helion API: The last line of triton_source must be an expression`.
- **Helion Compiler loses track of host tensor**: Even after using a dummy value, the code failed to compile, and the compiler lost track of the host tensor, with related questions added to the issue.
   - A member suggested an alternative code snippet using `hl.inline_triton` with `{1}` as the last line, when `output_like=None`.
- **Helion's "last line" requirement fix incoming**: A member is submitting a [PR](https://github.com/pytorch/helion/pull/1087) to remove the requirement that "the last line of triton_source must be an expression" when `output_like=None`.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1435282949678432287)** (85 messages🔥🔥): 

> `NVFP4 Contest, Mojo Support, Blackwell GPUs, TileLang and CuTeDSL, CUDA learning` 


- **Mojo Support might be coming!**: A member asked *is there anything that might be needed for **mojo support**?* and volunteered to help because *it's much easier than cutlass*.
   - Marksaroufim then replied that *We can just make mojo available from the start*.
- **Blackwell GPUs don't need to be bought for competition**: A member asked if everyone had **Blackwell GPUs** to anticipate the competition, and Marksaroufim replied that *You don’t need it really. The infra is free to use if you’re ok waiting in a queue.*
   - Marksaroufim also gave out a [referral link](https://cloud.sesterce.com/compute?referralCode=SES-KNYPOQ-3b6e) giving everyone here a discount.
- **Users will have NCU profiling**: A member asked *do they allow nsys and other CLI profiling tools for NVDA systems? would be pretty hard to do if we cannot profile the kernels 🥲*
   - A user pointed to a message mentioning the submission bot will have a feature to run your code and return a **ncu capture file** for you to analyze.
- **Global flexing is welcome in GPU MODE, Prize or No Prize**: In response to questions about participation eligibility, Marksaroufim clarified that *there are no restrictions on who can participate, anyone anywhere in the world can run nvfp4 kernels and learn and flex on the public leaderboard.*
   - The only restrictions apply to who is eligible for prize money, these are often complex legal discussions and it's best to rely on an official NVIDIA rep for a response here.


  

---


### **GPU MODE ▷ #[hf-kernels](https://discord.com/channels/1189498204333543425/1435311035253915840/1435311820712972450)** (3 messages): 

> `Xenova.com, HF Kernels Updates` 


- **Xenova says Let's Go!**: A user thanked **Xenova** and **xenova.com**, who responded with *Let's go!* indicating potential excitement or progress.
- **HF Kernels Channel Active**: The **hf-kernels** channel saw active engagement from users, suggesting ongoing discussions and updates related to **HF kernels**.


  

---


### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1435679715531952169)** (1 messages): 

> `Sentence Transformers joins Hugging Face, huggingface_hub v1.0, LeRobot v0.4.0, Cleaner Collection URLs, Inference Providers Usage Breakdown` 


- **Sentence Transformers Say 'I Do' to Hugging Face**: [Sentence Transformers](https://huggingface.co/blog/sentence-transformers-joins-hf) are officially joining Hugging Face to bolster open-source machine learning initiatives.
   - The move aims to integrate their **transformer models** more deeply into the Hugging Face ecosystem.
- **Hugging Face Hub Turns Five, Celebrates Foundation of Open Machine Learning**: The [huggingface_hub v1](https://huggingface.co/blog/huggingface-hub-v1) marks **five years** of building the foundation for open machine learning, enhancing model sharing and collaboration.
   - Version 1 includes major upgrades like cleaner collection URLs, improved inference, and a new API, further **streamlining model access and usage**.
- **LeRobot v0.4.0 Gets Supercharged for OSS Robotics Learning**: [LeRobot v0.4.0](https://huggingface.co/blog/lerobot-release-v040) introduces supercharged capabilities for **open-source robotics learning**, enhancing simulation and real-world application integrations.
   - The update focuses on improving **OSS Robotics Learning**.
- **Hugging Face and VirusTotal Fortify AI Security**: Hugging Face is teaming up with [VirusTotal](https://huggingface.co/blog/virustotal) to **strengthen AI security** by integrating threat detection capabilities.
   - The collaboration aims to provide users with enhanced tools to **identify and mitigate potential security risks** within AI models and applications.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1435297556996292821)** (189 messages🔥🔥): 

> `Home Lab AI Setup, Vex-Math-L1-100K Dataset, LLMs in Stock Data Prediction` 


- **Frontier Lab AI Shifts to Home Lab**: Members discussed a potential transition from frontier lab AI (like **Anthropic** and **OpenAI**) to a **homelab** setup for post-training models, focusing on eco-friendly configurations.
   - The consensus was that the **economics of scale** favor cloud solutions, as a homelab setup would be costly and potentially less eco-friendly, even with techniques like **LoRA fine-tuning**.
- **New Math Dataset Excites AI Enthusiasts**: A startup announced the release of the [Vex-Math-L1-100K dataset](https://huggingface.co/datasets/Arioron/Vex-Math-L1-100K), covering **9 major subdomains** and **135 subdomains** with **100K examples**.
   - Initial impressions suggest that smaller models may exhibit higher quality when quantized to a small size, challenging the notion that larger models quantized to a small size are always superior.
- **LLMs Enter Stock Data Prediction Arena**: Members discussed the application of **LLMs** for **stock data prediction**, exploring methods to find correlations between data, including factors like weather and news.
   - It was concluded that native techniques might be sufficient for now, and the use of **Generative AI** is not greatly researched in this area; one member provided a [link to more RAG architectures](https://github.com/NirDiamant/RAG_Techniques).


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1435321857216872620)** (3 messages): 

> `Job Application Automation Project, BERT Model Training, SetFit Contrastive Classifier, ArXiv Gatekeeping` 


- **Job Application Automation Battles Spam Detection**: A team member is developing a **Python system** using **Playwright** to automate job applications on sites like **Lever** and **Greenhouse**, successfully grabbing job links and filling in easy fields.
   - However, they are struggling with **spam detection** and inconsistent **HTML selectors**, requiring smarter stealth tactics and debugging efforts.
- **Adventures in BERT Model Training and SetFit Classifier**: A member trained a **BERT-style model for classification** and a **SetFit contrastive-style binary classifier**, but found it difficult to find an evaluation to submit results to.
   - They achieved a **48% improvement** over the baseline on **NPHardEval** using a **Qwen model**, obtaining **88/100 optimally correct answers** on the **Graph Coloring Problem**.
- **Academic Gatekeeping Frustrates Researcher**: A member expressed frustration with the difficulty of engaging with the academic process, citing gatekeeping issues at **ArXiv** and a paper rejection from **BBS**.
   - They are seeking a **good benchmark** where noteworthy results are disseminated and utilized for improving **AI** and asking for an **ArXiv referral**.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1435320084942815315)** (18 messages🔥): 

> `Model on 40-50b parameters, PDF2TXT parser, DearDiary.jl, NexusAI Professional Suite v1.0` 


- **Model Builder Braves 40-50B Parameters**: A member is building their own **40-50B parameter model** and attempting to run it on a graphics card with **32GB of VRAM**.
   - They are basing their model on **gpt-oss-20b** and plan to use **8k or 16k context windows**.
- **PDF2TXT Parser Gets an Upgrade**: A member shared a new version of their **PDF2TXT parser** which includes *better text-block-search* and is *20% faster* with simple table recognition (no OCR) [available here](https://huggingface.co/kalle07/pdf2txt_parser_converter).
- **DearDiary.jl Tracks ML Experiments with Ease**: **DearDiary.jl** is a pure Julia tool for tracking machine learning experiments with a [REST API](https://github.com/pebeto/DearDiary.jl) and SQLite portability.
   - Install via "] add DearDiary" to access full experiment tracking and a portable setup.
- **NexusAI Suite Launches Pro Workflows**: **NexusAI Professional Suite v1.0** launches with production-ready [ComfyUI workflows](https://github.com/NexusAI-Lab/ComfyUI-Professional-Workflows/releases/tag/v1.0.0) for Commercial, Realism, Anime, and Horror applications.
   - This promises one-click operation from idea to asset and claims to save hundreds of hours of configuration, plus a [live demo](https://huggingface.co/spaces/NexusBridge/comfyui-workflows-showcase) shows off the suite.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1435536302778093628)** (1 messages): 

> `NLP Data Cleaning` 


- **NLP Noob Needs Data Cleaning Help**: A new NLP practitioner is seeking guidance on data cleaning steps, particularly when dealing with raw text after Named Entity Recognition (**NER**).
   - They are requesting a standard process for data cleaning and wondering if their current approach is flawed.
- **Helpful NLP Data Cleaning Resources**: To assist the new practitioner, the community can provide links to tutorials, blog posts, or documentation outlining standard data cleaning processes for NLP.
   - Focus should be on techniques applicable to raw text and post-NER data.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1435444536922079464)** (7 messages): 

> `File retrieval issues, Study group formation` 


- **API's File Retrieval Hits 404**: Members are reporting **404 errors** when trying to retrieve associated files using the API, specifically from the `https://agents-course-unit4-scoring.hf.space/files/{task_id}` endpoint.
   - A specific example `99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3` was mentioned, which should return an **MP3 file** for a recipe task but instead results in a 404.
- **Agents Course Study Group Formation**: Several members have expressed interest in forming a study group for the **Agents Course**.
   - One member who started a week prior is happy to be included.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1435356446073159882)** (3 messages): 

> `Sora App Android, IndQA Benchmark, Interrupt Long Queries` 


- ****Sora** Arrives on Android!**: The **Sora app** is now available on Android in Canada, Japan, Korea, Taiwan, Thailand, US, and Vietnam, as shown in [this video](https://video.twimg.com/amplify_video/1985765811131465736/vid/avc1/704x1280/iatLt9jMW-vYYuIx.mp4).
- **AI Learns Indian Lingo with **IndQA****: OpenAI introduces **IndQA**, a new benchmark that evaluates how well AI systems understand Indian languages and everyday cultural context, explained in [this blogpost](https://openai.com/index/introducing-indqa/).
- **Interrupt your bot Mid-Thought!**: Users can now interrupt long-running queries and add new context without restarting or losing progress, demonstrated in [this video](https://video.twimg.com/amplify_video/1986194201076506628/vid/avc1/3840x2160/rEuDomNqKSd8jEdW.mp4).


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1435292355858141255)** (164 messages🔥🔥): 

> `Sora 2 code, OpenAI's photo generator, Custom GPT photo upload issues, Sora offline?, Models acting weird` 


- **Unsubscribing from Sora 2 code scammer**: A member expressed intent to unsubscribe from a YouTube channel due to a scam related to **Sora 2 code**.
   - Another member then stated *if you don't put any effort, of course you don't get a code. you're not fast enough. codes here are all gone in seconds after they post it*.
- **OpenAI's Photo Generator Needs Update**: One user stated that **OpenAI's photo generator** needs an update, as it is still **2D text based**.
   - They tried asking it to rotate an object like *15 degrees left on the y axis*, but stated they cannot do it no matter what.
- **Models act weird**: A member stated that *all models (from all providers) somehow are acting weird through the last weeks, be it because of broken features what is the case at Claude by Anthropic or quality drop or questionable corporate decisions (OpenAI), might sound weird now, but it seems a bit that time of AI for public is slowly over*.
   - Another member replied that they *only use Claude occasionally*, but haven’t had any problems with it.
- **GPT-4o's Hyper-Cautiousness**: A user complained that **GPT-4o is hyper-cautious and hyper-correct now**, attributing it to OpenAI's reaction to a lawsuit.
   - Another responded that they *don’t see a problem with that*.
- **LLMs flunk puzzle**: Members tested various Large Language Models (**LLMs**) on their visual puzzle solving abilities.
   - They tried **GPT 5 extended thinking** and **Gemini 2.5 pro**, but both got it wrong, as LLMs cannot solve maze problems.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1435295803848523797)** (13 messages🔥): 

> `Thinking model degraded, Building ChatGPT apps, OpenAI model comparisons` 


- **Thinking Model Degrades Chain-of-Thought**: Members noticed that the **Thinking model** has degraded in the last 5 days, no longer performing a proper **chain of thought** and instead *"thinks" for a few seconds without any steps.*
   - A user commented that *"it’s been horrible lately* and *they have to force it to actually think and not just make stuff up.*"
- **ChatGPT App Developers Assemble**: A member inquired if anyone is building **ChatGPT apps** (apps on chatgpts still in beta).
   - One user stated that they found the **Thinking mode** very good until recently.
- **OpenAI Model Comparison Archive Hunt**: A member asked if there was a website where **OpenAI compared how their first models responded to questions compared to further generations.**
   - Another user suggested asking **ChatGPT** itself to simulate older models.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1435339003816579193)** (14 messages🔥): 

> `Prompt Engineering Jobs, GPT Pro Research, Prompt Engineering Tips, Sora 2 Nerf` 


- **Prompt Engineering Jobs: Real or Myth?**: A member inquired about the existence of **prompt engineering jobs** after realizing their own aptitude in the area.
   - Another member responded that while *some companies do hire prompt engineers, it is not widespread*, with many people choosing to build their own projects.
- **GPT Pro Struggles with Lengthy Research**: A member asked for tips on prompting **GPT Pro** to process **50+ pages of research**, seeking results comparable to **Gemini's deep research** capabilities.
- **Core of Prompt Engineering: A 4-Step Guide**: A member shared what they consider the core of prompt engineering, which involves picking a familiar language, understanding the desired output, explaining the task clearly, and carefully verifying the results.
   - They advised being extra cautious with math, sources, code, or other details where **AI hallucination** is common.
- **Advanced Prompting Techniques Shared**: A member shared advanced prompting techniques including **Hierarchical communication with markdown**, Abstraction through open variables, **Reinforcement** in prompts, and **ML format matching for compliance**.
   - They also included an [Output Template](https://example.com/output-template) demonstrating the structure.
- **Sora 2 Hit With Another Nerf?**: A member questioned whether **Sora 2** had been hit with another **nerf**.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1435339003816579193)** (14 messages🔥): 

> `Prompt Engineering Job Market, Prompting GPT Pro for Research, Tips for New Prompt Engineers, Sora 2 Nerf` 


- **Demand for Prompt Engineers Still Debated**: Members discussed whether companies are actively hiring **prompt engineers**, with some suggesting opportunities are limited and many are building their own projects.
   - Several participants weighed in on how they've been **using their skills** and whether it might be a **possible career path**.
- **GPT Pro struggles on large research prompts**: One member asked about prompting **50+ pages of research with GPT Pro**, seeking results comparable to Gemini's deep research capabilities.
   - The original poster noted that the **same kind of prompt** for both models gets widely different results.
- **Prompt Engineering Core Principles Exposed**: A member shared what they consider the core of prompt engineering: focus on **clear communication**, **accurate language**, and careful output verification, including **fact-checking** and hallucination awareness.
   - Another member shared a detailed guide with lessons on hierarchical communication, abstraction through variables, reinforcement in prompts, and **ML format matching**.
- **Sora 2 nerfed?**: A member simply asked if **Sora 2** got hit with another nerf.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1435283276649463939)** (56 messages🔥🔥): 

> `Anthropic's closed-source approach, Piracy for Media Preservation, IMO Gold AI, llama.cpp contribution` 


- ****Anthropic Avoids Open Source: Community Unease Rises****: Members express unease about Anthropic's commitment to close-source practices and [deprecation policies](https://www.anthropic.com/research/deprecation-commitments), fearing model weights will be lost if the company fails.
   - Some suggest Anthropic should release weights with strings attached or allow deployment on platforms like Sagemaker to ensure preservation, while one member hopes someone will *pull a Miqu 70B on Anthropic* and leak the weights.
- ****Piracy Preserves Media Heritage****: Members argue that **piracy** has done more for **media preservation** than content creators, citing examples of lost pre-VHS content and potential future loss of streaming-only content due to licensing issues.
   - One member refuses to condemn piracy and shared *there is a reason that I refuse to condemn piracy, while not strictly condoning it*.
- ****IMO Gold Within Reach? Fine-Tuning Hype Accelerates****: A member shared [a paper](https://arxiv.org/pdf/2511.01846) suggesting that **AI models** are close to achieving **IMO (International Mathematical Olympiad) gold** level performance with further fine-tuning.
   - Some consider this approach (benchmarking models to solve specific problems) uninteresting compared to training models on general mathematics.
- ****New Contributor Joins llama.cpp Community****: A member announced they officially became a **llama.cpp contributor** after submitting a patch.
   - Another member praised the aesthetics of the **new llama web GUI**.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1435293202113495170)** (15 messages🔥): 

> `Gestural Interfaces, Repligate's Loom, Attention is All You Need, Vision Pitching` 


- **Gesture-Based Interfaces: Looming Interaction**: A member shared their attempt to create a gestural interface for **Repligate's concept of the Loom**, aiming to make human-AI interaction more *physical* by letting users *feel* the intelligence they are interacting with.
   - They also noted this approach could create **perspective parallax**, an optical illusion for seeing *3D* without VR/XR glasses, projecting one's *gnosis* onto reality as an interactive overlay.
- **Embracing the Gestural Future**: A member admitted needing to create more with **gestures**, but noted the difficulty of balancing it with other **AI** pursuits.
   - They suggested that as **XR glasses** become commonplace, interfaces relying on gestures will gain popularity, referencing a moment when users might remove their glasses and gesture at screens, forgetting they need the glasses.
- **Ditching Repos and the Win9x Aesthetic**: A member described creating a **gestural interface** **7-8 years ago**, anticipating *vibe coding*, predating **Mediapipe** and using a **Win9x aesthetic** for the documentation.
   - Frustrated by the inability to raise funding, they deleted all their repos to *move on*.
- **"Attention is All You Need!"**: Quoting the famous paper [*Attention is All You Need!*](https://arxiv.org/abs/1706.03762), a member jokingly responded when another member said *don't be bothered by lack of attention*.
   - The same member expressed that they *think i found my pack here i just gotta get the courage to pitch my vision*.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

real.azure: https://github.com/ggml-org/llama.cpp/discussions/16938
  

---


### **tinygrad (George Hotz) ▷ #[announcements](https://discord.com/channels/1068976834382925865/1069236008115253348/1435333626077380690)** (1 messages): 

> `tinybox pro v2, 5090, rackable workstation` 


- **tinybox pro v2 Launches: 8x 5090 Powerhouse!**: The **tinybox pro v2**, a new product, was announced, boasting **8x 5090 GPUs** in a **5U rackable workstation** for enhanced performance.
   - Available for order on the website, it's priced at **$50,000** with a shipping time of **4-12 weeks**, promising significant computational power.
- **5090s Dominate the tinybox pro v2**: The **tinybox pro v2** workstation is configured with **8x 5090 GPUs** to handle demanding computational tasks.
   - This high GPU density in a **5U rackable** form factor aims to provide substantial processing capabilities for professional users.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1435308643133296772)** (68 messages🔥🔥): 

> `VK_KHR_buffer_device_address GLSL extension, Tinybox Pro V2, AMD vs Nvidia, GLSL Renderer Implementations, Tensor Cores on M1` 


- **`VK_KHR_buffer_device_address` Improves Performance**: Using [`VK_KHR_buffer_device_address`](https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#VK_KHR_buffer_device_address) and the corresponding GLSL extension `GL_EXT_buffer_reference` will give a *considerable performance increase* by enabling the use of pointers directly in GLSL for features like float4, as seen in [this tinygrad implementation](https://github.com/softcookiepp/tinygrad/blob/master/tinygrad/renderer/glsl.py).
- **Exploring GLSL Renderer Implementations**: One member shared [their GLSL renderer implementation](https://github.com/uuuvn/tinygrad/blob/vulkan/tinygrad/renderer/glsl.py), noting that it unexpectedly fixed AGX compiler bugs by catching *invalid stuff* in SPIR-V disassembly that LLVMpipe tolerated, including integer overflows on unsigned adds for negative indexing.
   - They attributed the issues to `clspv`, a mostly dead Google project, which motivated the switch to GLSL; the member tested it on **M1 Asahi Linux**.
- **Tinybox Pro V2 Specs**: George Hotz is soliciting feedback on the [Tinybox Pro V2](https://x.com/__tinygrad__/status/1985774711499080186) (product link at [TinyCorp](https://tinycorp.myshopify.com/products/tinybox-pro-v2)), with `comma.ai` already being a customer.
   - The Tinybox is aimed towards *pro-level compute* with some already running nanochat and deepseek models and are interested in a potential **$10k** AMD-based mini version, with a target of **$3-4/hour** on GPU rental sites.
- **AMD Drivers Are Pleasant Now**: A member stated an 8x AMD R9700 setup would be cheaper than the Tinybox Pro V2.
   - George Hotz responded that *the rock is very pleasant to use* and claimed that **R9700s are a rip off** while **5090s are a ton more powerful**, though another member countered that *$50k is too much*.
- **M1 Chips: No Real Tensor Cores?**: One member pointed out that **M1 chips** might not have real tensor cores, referencing [a tweet](https://x.com/norpadon/status/1965753199824175543).
   - Another member suggested they are abstractions for optimized subroutines for GEMM, with metal doing *shader stuff* with SIMDgroup and threadgroup with tile size optimisations to prevent register spill.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1435375185258741820)** (12 messages🔥): 

> `OU Processes Limitations, ito Diffusions Universality, Paper Dumps Overload, Trending AI Papers` 


- **OU Processes Assumptions Examined**: A member discussed ways to overcome the limiting assumptions of **OU processes** (*gaussian+markov+linear+exponential autocovariance*) using [Lévy type drivers](https://epubs.siam.org/doi/10.1137/S0040585X97978166) or integrating over OU kernels (supOU) and [continuous ARMA processes](https://www.sciencedirect.com/science/article/abs/pii/S0169716101190115).
- **Ito Diffusions Achieve Universality**: It was stated that if you train a diffusion process in the *normal* way, you gain NOTHING from going beyond ito diffusion `dX_t = b(X_t)dt + σ(X_t)dW_t` as [ito diffusions are already universal](https://en.wikipedia.org/wiki/It%C3%B4_diffusion) in the sense that any two equally atomic distributions are related.
   - Going beyond ito diffusions only changes _how_ you can reach certain distributions, according to this member.
- **Paper Dumps Triggered Discussion**: Multiple members expressed concerns that a single person posting a large number of random, potentially irrelevant papers was burying papers of interest to the community.
   - One member suggested that the individual posting these papers should post a **maximum of two papers per day**, filtering for relevance, rather than *10+*.
- **Trending Papers Become Heuristic Guide**: A member suggested using trending papers pages such as [AlphaXiv](https://www.alphaxiv.org/), [Emergent Mind](https://www.emergentmind.com/), and [nlp.elvissaravia.com/t/ai](https://nlp.elvissaravia.com/t/ai) as heuristic guides for posting relevant papers.
   - Another member suggested those posting papers should explain in their own words why they found each paper important.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1435341771490988224)** (21 messages🔥): 

> `Anthropic's crosscoder, circuit tracing research, feature evolution during pre-training, leakages from shared chats, latent reasoning` 


- **Cross Coder is a go!**: Members plan to examine [this paper on crosscoder](https://arxiv.org/abs/2509.17196) and circuit tracing research to observe different phases of **feature evolution during pre-training**.
   - It was mentioned that this paper was on their list to read already.
- **Destroying LLMs?**: One member claimed to have destroyed every flagship LLM but questioned whether they could truly know or were just being told things to make it seem that way.
   - They cited **leakages from shared chats** as an example of subjective or objective evidence and mentioned *softer penetrations through policy bound spaces* serving as "guard rails".
- **Image Reconstruction Quality Boost?**: One member suggested providing a brief history of image reconstruction, noting a decline in quality since 2020, but expressed optimism that [this paper](https://arxiv.org/abs/2510.25976) represents *a step in the right direction*.
   - They linked [three papers from 2023](https://arxiv.org/pdf/2303.05334), [another from 2023](https://arxiv.org/pdf/2306.11536), and [one more from 2022](https://arxiv.org/pdf/2211.06956).
- **Latent Reasoning Attempt**: A member shared [a link on latent reasoning](https://fxtwitter.com/RidgerZhu/status/1983732551404679632), calling it *another attempt*.
   - Another member suggested [this paper](https://arxiv.org/abs/2510.25741) and said that it looks like a RNN.


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1435653602281787432)** (2 messages): 

> `RWKV, HRM/TRM, Context Windows, State Representations` 


- ****RWKV** Catches Attention!**: A member caught up on **RWKV** [in this video](https://youtu.be/LPe6iC73lrc) and found it *pretty impressive*.
   - He suggested visiting with them and seeing where they are at, especially given recent advances in **HRM/TRM**.
- **HRM/TRM Merging Excites Members!**: The member suspects **RWKV** are already merging **HRM/TRM** together given their rapid progress.
   - *They are also actively building which is exciting*.
- **Big Context Windows Needed Now**: The member notes that bigger **context windows** are needed for coding and medicine related work, due to massive chart sizes.
   - They state *forgetting should not be the actual case*.
- **Models Make Varied State Representations**: After reviewing **HRM** and **TRM**, it was realized that these models are creating different state representations for different topics in different parts of the outer loops.
   - The member concludes *there is no one state space. It’s pieces or many distributed states within the hidden spaces that are updated as needed*.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1435397501439053865)** (11 messages🔥): 

> `Concentration of power, Erosion of democratic systems, Copyright lawsuit, Getty vs Stability` 


- **Democracy questioned Amidst Power Shifts**: A member criticized what they perceived as a concentration of power and erosion of democratic systems, sparking debate and leading to suggestions to block dissenting opinions.
   - Another member found the initial comment insufficient to warrant blocking, highlighting differing thresholds for engaging with controversial views.
- **Stability Wins Copyright Lawsuit Against Getty**: **Stability AI** won a copyright lawsuit against **Getty Images**, according to a [linked document](https://drive.google.com/file/d/1vqcQQU8gxGfFA1lUS68BZ8-hrGsu_Flj/view?usp=drivesdk).


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1435305760174243862)** (35 messages🔥): 

> `Pause and resume optimization runs, Accessing the LLM in a DSPy module, Handling Rate Limits with Fallback LLMs, Conversation History Management in DSPy, Pydantic OutputField Deserialization` 


- **No Easy Way Exists to Pause Optimization Runs**: A member requested a way to **pause and resume optimization runs**, noting [this is a common request](https://x.com/DSPyOSS/status/1985746487322595341), but the answer is that there is no easy way right now other than writing a new adapter (or editing this one).
- **Demystifying LLM Access in DSPy Modules**: Discussion revolved around how to **access and change an LLM** for a module, highlighting the use of `get_lm()` and `set_lm()` methods.
   - Members clarified that accessing the underlying LLM for a DSPy module is essential for low-level programming, particularly when hitting API limits.
- **Rate Limit Handling via Fallback LLMs**: The conversation explores **how to handle rate limit exceptions** by switching to a fallback LLM (e.g., from `gpt-4o` to `gpt-4o-mini`).
   - It was highlighted that changing the LLM resets the history, and the challenge lies in transferring the module's main LLM history to the new fallback model to maintain context during ReAct module iterations.
- **Conversation History: Signature Solves the Mystery**: Members found the solution lies with the `dspy.History` object inside the signature, rather than a module or LLM object.
   - Using a `dspy.History` object, the same history can be maintained even when switching LLMs mid-program execution.
- **Java JSONAdapter Simplified Version Needed**: A member asked if a **simplified Java version of JSONAdapter** exists, for cases where they want to use DSpy prompts in Java.
   - The goal is to structure system messages and handle JSON responses in Java, similar to how DSPy manages it.


  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1435655603489083452)** (5 messages): 

> `Synthetic Data Use, Eval Metric, GEPA, Glossary Building` 


- **Synthetic Data useful for Glossary Building**: A member thinks [synthetic data](https://x.com/WesEklund/status/1986096335708197102) would be helpful for **glossary building** use case.
   - They posted a [Colab notebook](https://colab.research.google.com/drive/179UlSHSpK-I6H-g4dSgAvuCmPDejxFgm?usp=sharing) that needs an **Eval metric** and **GEPA** implementation, showcasing a basic working example.
- **Colab Example for Glossary Building**: A member shared a [Colab notebook](https://colab.research.google.com/drive/179UlSHSpK-I6H-g4dSgAvuCmPDejxFgm?usp=sharing) demonstrating a basic implementation for **glossary building** using synthetic data.
   - The notebook is a work in progress and requires the addition of an **Eval metric** and **GEPA** (Generative Evaluation of Pattern Articulation) to be fully functional.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1435296101791174809)** (24 messages🔥): 

> `Kimi iOS app, interleaved thinking model, Kimi CLI, 401 Error` 


- **New iOS Kimi K-2 App is 'Ok Computer'**: The new Kimi K-2 iOS app is praised for its design, specifically the *Ok Computer* theme, with one user asking for [translation of all languages](https://discord.com/channels/974519864045756446/977259063052234752/1435599575791570995) instead of just Indian languages.
   - Another user shared a link to a [presentation](https://3tadkfqfwbshs.ok.kimi.link/present.html).
- **Interleaved Thinking Model lands in Kimi CLI**: Support for the **interleaved thinking model** has been added to Kimi CLI, prompting questions about its difference from the standard *think...final answer* approach.
   - The original poster remarked that *they added support for thinking mode in kimi-cli yesterday. 👀*
- **User gets 401 Error setting up CLI**: One user reported receiving a **401 error** when trying to set up the Kimi CLI, despite having reloaded credits and having a balance; another member identified the problem being that **credits are for the Moonshot AI Open Platform** rather than the Kimi K-2 platform.
   - The user was directed to the correct channel for help.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1435300860413476926)** (8 messages🔥): 

> `Model testing configurations, Using Perplexity API, aider-ce documentation` 


- **Model Testing: Stick with Defaults**: A member asked about disabling/enabling options when testing a model, and another suggested to *choose a good model and roll with the defaults until you have reason to change*.
   - There was no further discussion on this topic.
- **Perplexity API Integration into aider**: A user requested a tutorial for using the **Perplexity API** with aider, similar to the existing Gemini tutorial, referencing the [aider documentation](https://aider.chat/docs/llms/other.html#other-api-key-variables).
   - Another member suggested replacing *gemini* with *perplexity* in the existing instructions and setting the API key as an environment variable, using `export PERPLEXITYAI_API_KEY=your-api-key` and then trying `aider --list-models perplexity`.
- **OpenRouter is better for coding with Aider**: A member suggested that **OpenRouter** might be a better choice than **Perplexity API** for coding, highlighting the availability of numerous free and powerful models for testing aider.
   - They mentioned you can try **OpenRouter** since it's *both powerful and FREE for testing aider out*.
- **Aider-CE Documentation**: A member inquired whether the channel was also for **aider-ce** and if **aider-ce** has documentation.
   - There was no further discussion on this topic.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1435365577211383890)** (9 messages🔥): 

> `TDD with Aider, Memory limitations with Ollama, Claude Code's /compact command, scalafix-rules summarization` 


- ****TDD loop using aider**?**: A user is looking to create a loop with **aider** where agent 1 executes a prompt with **TDD**, agent 2 reviews and suggests improvements, agent 1 executes the improvements, and agent 2 runs TDD tests, fixes bugs, and proposes a commit message.
   - The suggestion was made to use [scripting](https://aider.chat/docs/scripting.html) to wrap the `aider` functionality.
- ****Memory issues with Ollama and scalafix-rules****: A user wants to summarize rules in the [scalafix-rules](https://github.com/xuwei-k/scalafix-rules/tree/main/rules/src/main/scala/fix) project but is running into memory limitations with **Ollama**.
   - They want to avoid loading all the rules at once and instead process them one by one, generating a table entry for each rule with its name, summary, and transformation example, while unloading the previous rule to save memory.
- ****Aider's Missing `/compact` Command****: A user inquired if **Aider** has a command similar to **Claude Code's** `/compact` command, which summarizes and condenses conversation history to prevent context loss.
   - The suggestion was made to ask the model to summarise the conversation into a `status_report.md` using the `clear`, `drop`, and `reset` commands.
- ****Cloud-based models for simpler problems****: In response to memory issues summarization problem, a member suggests using cloud-based models for the summarization task.
   - They believe that a short description per rule isn't hard for local models such as **qwen 30b**, but becomes problematic when processing all rules at once, quickly exhausting memory.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1435297152980090922)** (17 messages🔥): 

> `IETF 124, MCP discussion in IETF, AI Scrapers` 


- **IETF Channel Craving Commences**: Members are considering creating a temporary channel for **MCP** folks at **IETF 124** this week, similar to the one for the dev summit.
   - There was agreement with the suggestion to create a channel, and to create a channel group for *IETF meetings in general*.
- **Events Category is Born**: It was proposed to make an **events** category, and a channel for events that have a critical mass of participants.
   - Some members cited a talk on potential **IETF** adoption of transport protocols for **MCP/A2A** earlier today and the fact that other IETF groups (like HTTPAPI) may have relevance for **MCP**.
- **AI Scrapers Spark Side Session**: Some members noted that current discussion in this session is not about **OAuth** at all, it's about general **AI scraping/crawlers**.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1435338628699127829)** (4 messages): 

> `IFeval Scores, Latent Reasoning` 


- **IFeval Scores Investigated**: Members discussed [IFeval scores](https://openreview.net/forum?id=Q7mLKxQ8qk) and their prompt-level/instruction-level variations.
   - One member clarified that they use an *average of all scores* as described in the [inspect_evals repo](http://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/ifeval#final-accuracy).
- **Latent Reasoning Attempted**: A member shared an attempt at latent reasoning with a link to a Tweet and an Arxiv paper: [Latent Reasoning Tweet](https://fxtwitter.com/RidgerZhu/status/1983732551404679632) and [Latent Reasoning Paper](https://arxiv.org/abs/2510.25741).


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1435466292038467674)** (12 messages🔥): 

> `Concept Detection System, Equivalent Linear Mappings of LLMs, Tangent Model Composition, Jacobian for LLM Interpretability` 


- ****Steering Concepts Real-Time** with Cheap Concept Detection**: A member built a system to **detect and steer thousands of concepts** in a model simultaneously and cheaply in realtime while it's generating.
   - The question remains of which concepts to detect to meaningfully subdivide semantic space; they are seeking prior art on **finite, scalable concept representations** for interpretability.
- ****Equivalent Linear Mappings** for LLM Interpretability**: A member published a paper, [“Equivalent Linear Mappings of Large Language Models”](https://openreview.net/forum?id=oDWbJsIuEp), demonstrating that **LLMs have equivalent linear representations of their inference operation** for any given input sequence.
   - They use the **SVD of the linear representations** to find low-dimensional, interpretable semantic structure that can be used for steering, modular at the layer/block level, and applicable to models like Qwen 3, Gemma 3, Llama 3, Phi 4, OLMo 2 and Mistral.
- ****Tangent Model Composition** Relevance**: A member asked about the relevance of [Tangent Model Composition for Ensembling and Continual Fine-tuning](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Tangent_Model_Composition_for_Ensembling_and_Continual_Fine-tuning_ICCV_2023_paper.pdf).
   - The original poster responded that **Tangent Model Composition** looks at **tangents and Taylor expansions in weight / parameter space**, whereas their work looks at the **Jacobian in input embedding space**.
- ****Jacobian Intuition** for Image Models**: A member shared image model papers from Zara Khadkhodaie and Eero Simoncelli ([paper 1](https://iclr.cc/virtual/2024/oral/19783), [paper 2](https://arxiv.org/abs/2310.02557), [paper 3](https://arxiv.org/abs/1906.05478)) that can compute the **conventional autograd Jacobian** at inference to get exact reconstruction.
   - The original poster enjoyed the description of **LLMs operating in low dimensional subspaces** when each input has its own subspace, but also thinks some tool for bridging across different inputs or models is going to be necessary for interpretability.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1435302534297292801)** (16 messages🔥): 

> `Text to video tools, Webscraping Twitter/X, Manus Support, Host services for Manus apps, Publishing problems on Manus` 


- **Text-to-Video Tool Quest Begins**: A member inquired about useful **text-to-video tools**, sparking a discussion about accessible options.
- **X-Scraping Shenanigans Sans API**: One member sought advice on **web scraping Twitter/X without using an API**, citing maintenance difficulties with their current cookie-based Python library.
- **Manus Host Services**: A member asked for recommendations for **host services** to run applications created with Manus Dev, noting the unsuitability of Manus for 24/7 commercial setups, and another user said *Vercel is good*.
- **Project Publishing Problems Plague Platform**: Members reported issues with **publishing projects**, noting that updates were not reflecting the latest checkpoints.
- **Manus-to-GitHub Migration Methods**: A member asked about the best ways to **transfer projects from Manus to GitHub** after experiencing unresolved errors and project setbacks.


  

---


### **Windsurf ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1435336696106455212)** (1 messages): 

> `Codemaps, SWE-1.5, Sonnet 4.5` 


- **Windsurf introduces Codemaps!**: Windsurf introduces **Codemaps**, powered by **SWE-1.5** and **Sonnet 4.5**, to help scale productive output by increasing your ability to understand the code you are working with.
   - According to Windsurf, *the largest constraint on your ability code, whether manually or with agents, is your ability to understand the code you are working with* – more info available at their [X post](https://x.com/windsurf/status/1985757575745593459).
- **Understanding code is key**: Windsurf emphasizes that understanding the code you are working with is crucial for effective coding, whether done manually or with AI agents.
   - Quoting Paul Graham, Windsurf highlights that *'your code is your understanding of the problem you’re exploring,'* underscoring the importance of having a firm grasp of the code.


  

---


---

