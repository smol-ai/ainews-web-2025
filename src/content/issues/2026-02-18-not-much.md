---
id: MjAyNi0w
title: not much happened today
date: '2026-02-18T05:44:39.731046Z'
description: >-
  **Anthropic** released **Claude Opus/Sonnet 4.6**, showing a significant
  intelligence index jump but with increased token usage and cost. **Anthropic**
  also shared insights on AI agent autonomy, highlighting human-in-the-loop
  prevalence and software engineering tool calls. **Alibaba** launched **Qwen
  3.5** with discussions on reasoning efficiency and token bloat, plus
  open-sourced **Qwen3.5-397B-A17B FP8 weights**. The **GLM-5** technical report
  introduced asynchronous agent reinforcement learning and compute-efficient
  techniques. Rumors about **Gemini 3.1 Pro** suggest longer reasoning
  capabilities, while **MiniMax M2.5** appeared on community leaderboards. The
  community debates benchmark reliability and model performance nuances.
companies:
  - anthropic
  - alibaba
  - scaling01
  - arena
  - artificial-analysis
models:
  - claude-4.6
  - claude-opus-4.6
  - claude-sonnet-4.6
  - qwen-3.5
  - qwen3.5-397b-a17b
  - glm-5
  - gemini-3.1-pro
  - minimax-m2.5
topics:
  - benchmarking
  - token-efficiency
  - ai-agent-autonomy
  - reinforcement-learning
  - asynchronous-learning
  - model-performance
  - open-weights
  - reasoning
  - software-engineering
  - agentic-engineering
people:
  - eshear
  - theo
  - omarsar0
  - grad62304977
  - scaling01
---


**a quiet day**

> AI News for 2/17/2026-2/18/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and 24 Discords (**262** channels, and **10849** messages) for you. Estimated reading time saved (at 200wpm): **1103** minutes. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap

**Frontier model + benchmark churn (Claude 4.6, Qwen3.5, GLM‑5, Gemini 3.1 Pro, MiniMax M2.5)**

- **Anthropic Claude Opus/Sonnet 4.6: big jump, big token bill**: Artificial Analysis reports **Sonnet 4.6** at **51** on its Intelligence Index (up from **43** for Sonnet 4.5 reasoning), sitting just behind **Opus 4.6** at **53**, but with markedly worse token efficiency: **~74M output tokens** to run the suite vs **~25M** for Sonnet 4.5 and **~58M** for Opus 4.6 (and **$2,088** to run the index for Sonnet 4.6 in max effort) ([AA summary](https://x.com/ArtificialAnlys/status/2024259812176121952), [token note](https://x.com/ArtificialAnlys/status/2024259815930012105)). Community sentiment echoes “4.6 feels better at critique/architecture” ([eshear](https://x.com/eshear/status/2024148657797308747)) while also flagging reliability/product issues around Claude Code (see “Anthropic drama” discourse around SDK/docs and tooling stability) ([theo](https://x.com/theo/status/2024225756981973214)).
- **Claude in Search Arena + autonomy telemetry**: Arena added **Opus/Sonnet 4.6** to its search modality leaderboard ([arena](https://x.com/arena/status/2024144830209966142)). Anthropic also published “**Measuring AI agent autonomy in practice**,” analyzing millions of tool-using interactions: **~73%** of tool calls appear **human-in-the-loop**, only **0.8%** appear **irreversible**, and **software engineering** is ~**50%** of tool calls on their API—framed as “autonomy is co-constructed by model + user + product,” motivating post-deployment monitoring ([Anthropic](https://x.com/AnthropicAI/status/2024210035480678724), [metrics](https://x.com/AnthropicAI/status/2024210050718585017), [industry mix](https://x.com/AnthropicAI/status/2024210053369385192)).
- **Qwen 3.5: reasoning efficiency vs “excess thinking”**: Multiple posts highlight Qwen3.5’s “overthinking”/token usage as a key axis—both complaints ([QuixiAI](https://x.com/QuixiAI/status/2023995215690781143)) and deeper community analysis claiming Qwen3.5-Plus reduces long-chain token bloat vs older Qwen reasoning variants, while noting regressions in non-reasoning mode ([ZhihuFrontier](https://x.com/ZhihuFrontier/status/2024176484232155236)). On the distribution side, Qwen3.5-Plus shipped to **Vercel AI Gateway** ([Alibaba_Qwen](https://x.com/Alibaba_Qwen/status/2024029499541909920)) and Alibaba Cloud launched a **Qwen Coding Plan** subscription with fixed monthly pricing and high request caps aimed at coding agents ([Alibaba_Qwen](https://x.com/Alibaba_Qwen/status/2024136381308805564)).
- **Qwen3.5-397B-A17B FP8 weights opened**: Alibaba released **FP8 weights** for **Qwen3.5‑397B‑A17B**, with **SGLang support merged** and a **vLLM PR** in flight (vLLM support “next couple days”)—a concrete example of “open weights + immediate ecosystem bring-up” becoming table stakes for competitive OSS releases ([Alibaba_Qwen](https://x.com/Alibaba_Qwen/status/2024161147537232110)).
- **GLM‑5 technical report + “agentic engineering” RL infrastructure**: The **GLM‑5** tech report is referenced directly ([scaling01](https://x.com/scaling01/status/2024050011164520683)) and summarized as pushing from vibe-coding to “agentic engineering,” featuring **asynchronous agent RL** that **decouples generation from training** and introducing **DSA** to reduce compute while preserving long-context performance ([omarsar0](https://x.com/omarsar0/status/2024122246688878644)). Practitioners called the report unusually detailed and valuable for OSS replication, pointing out optimizer/state handling and agentic data curation details (terminal envs, slide generation, etc.) ([Grad62304977](https://x.com/Grad62304977/status/2024170939248714118)).
- **Gemini 3.1 Pro rumors + “thinking longer”**: Early testing anecdotes suggest Gemini 3.1 Pro runs substantially longer “thinking” traces than Gemini 3 Pro and may close the gap with Opus/GPT—paired with skepticism about benchmark trustworthiness and failures on adversarial cases (e.g., mishandling ARC-AGI-2 prompt containing the solution) ([scaling01](https://x.com/scaling01/status/2024251668771066362), [ARC anecdote](https://x.com/scaling01/status/2024268831321993590)).
- **MiniMax M2.5 appears on community leaderboards**: Yupp/OpenRouter posts indicate onboarding MiniMax **M2.5** and **M2.5 Lightning** and tracking results via prompt-vote leaderboards ([yupp_ai](https://x.com/yupp_ai/status/2024165671136059892), [OpenRouter benchmark tab](https://x.com/OpenRouter/status/2024172351630252)).

---

**Agentic coding + harness engineering (Claude Code, Cursor, LangSmith, Deep Agents, SWE-bench process)**

- **Harness is performance**: A clean side-by-side shows identical model (**Claude Opus 4.6**) with different agent harnesses: **LangChain Deep Agents CLI** completing in **9s** vs **Claude Code** in **16s**—a **1.7×** delta “with zero model changes,” reinforcing that orchestration, tool policies, and context strategy dominate user-perceived capability ([GitMaxd](https://x.com/GitMaxd/status/2024137171217871106)). A related post notes how Claude Code’s prompt appears to “fight the weights” to get parallel tool calls, suggesting architectural friction between model priors and harness demands ([dbreunig](https://x.com/dbreunig/status/2024247669359788050)).
- **Cursor doubles down on “agent memory” UX**: Cursor shipped **.agents/skills** support ([leerob](https://x.com/leerob/status/2024141610796150903)) and then added **past conversations as context**—a practical step toward persistent, tool-usable memory for IDE agents ([cursor_ai](https://x.com/cursor_ai/status/2024222146642497713)).
- **LangSmith Agent Builder upgrades**: LangChain shipped a “general agent” chat with access to all workspace tools, **chat→agent** conversion, **file uploads**, and a central tool registry—explicitly targeting reduced friction between experimentation and deployable agents ([LangChain](https://x.com/LangChain/status/2024180357457989887)). They also added **Baseline Experiments** to anchor regression tracking in eval-driven workflows ([LangChain](https://x.com/LangChain/status/2024208662936650152)).
- **SWE-bench infra iteration**: SWE-bench leaderboard migrated to running everything with **mini-SWE-agent v2** to “get more juice out of base models,” which implicitly changes how model progress is interpreted (harness upgrades shift the frontier) ([OfirPress](https://x.com/OfirPress/status/2024177059895877802)). In parallel, criticism surfaces about “SWE-fficiency ranking is broken,” reflecting ongoing discomfort with evaluation methodology for agentic coding benchmarks ([scaling01](https://x.com/scaling01/status/2024171017929638061)).
- **Practical safety footgun for Windows agent shells**: If your “bash tool” is Git Bash/MSYS2, *do not* emit Windows redirections like `2>nul`; it can create an undeletable `nul` file on NTFS. Use Unix-style redirects or explicitly wrap Windows commands in `cmd /c` ([MParakhin](https://x.com/MParakhin/status/2024172856029171877)).

---

**OpenAI + smart-contract security as an “agent capability” slice (EVMbench)**

- **EVMbench launched**: OpenAI introduced **EVMbench**, targeting agent ability to **detect, exploit, and patch** high-severity smart contract vulnerabilities ([OpenAI](https://x.com/OpenAI/status/2024193883748651102)). The subtext across replies/quote-tweets is that *agentic security* is becoming a first-class eval category rather than an afterthought; engineers immediately compare model families and precision/recall tradeoffs ([gdb](https://x.com/gdb/status/2024200501055963593), [scaling01 commentary](https://x.com/scaling01/status/2024212205944643718)).
- **Signal for engineers**: This is one of the cleaner examples of an eval tied to real exploit/patch workflows (not just static QA). If you build agentic code review, on-chain monitoring, or automated incident response, EVMbench-style tasks look closer to production than many generic coding leaderboards.

---

**Data, curation, and evaluation hygiene (ÜberWeb multilingual, prompt repetition, “slop pollution”)**

- **ÜberWeb: multilingual gains without sacrificing English**: DatologyAI’s “ÜberWeb” claims shifting the compute–performance Pareto frontier for multilingual models via **data quality/composition**, at **20T+ tokens** scale—pushing back on the “curse of multilinguality” framing as primarily a data-quality problem ([RicardoMonti9](https://x.com/RicardoMonti9/status/2024136992779559055), [pratyushmaini](https://x.com/pratyushmaini/status/2024157352862376280), [agcrnz](https://x.com/agcrnz/status/2024207781524623690)).
- **Prompt repetition controversy**: Viral claims that repeating the same prompt twice yields huge accuracy gains (e.g., 21%→97% on a name-search task) triggered methodological pushback: gains may vanish when the question is placed first, and reported results may be inflated by not including question-first baselines ([kimmonismus claim](https://x.com/kimmonismus/status/2024069380162936992), [paul_cal critique](https://x.com/paul_cal/status/2024053549965934886)).
- **Dataset poisoning is no longer hypothetical**: A widely-shared anecdote: an incorrect “first 500 primes” webpage surviving for decades can “pollute generative AI models” by 2026—highlighting the fragility of web-trained factual priors and the need for provenance-aware retrieval and verification layers ([skominers](https://x.com/skominers/status/2024078964667396342)).
- **AI slop detection + provenance**: Posts warn about fake robotics media (e.g., non-existent Unitree models/hands) and emphasize checking source credibility and physical plausibility ([teortaxesTex](https://x.com/teortaxesTex/status/2024001310865924599)). On the mitigation side, Google pushes **SynthID** watermark verification for audio inside Gemini, extending provenance tooling beyond images/video ([GeminiApp](https://x.com/GeminiApp/status/2024153548641177781), [Google](https://x.com/Google/status/2024172104711823678)).

---

**Multimodal + creative model releases (Lyria 3 music, long-context VLMs, video editing)**

- **Google/DeepMind Lyria 3: music generation shipped into Gemini**: Lyria 3 generates **30-second tracks** from text or image/video prompts, supports **lyrics/vocals**, and is rolling out broadly in Gemini; outputs are watermarked with **SynthID** and Gemini can verify audio provenance via SynthID checks ([GeminiApp launch](https://x.com/GeminiApp/status/2024152863967240529), [DeepMind](https://x.com/GoogleDeepMind/status/2024153067654902014), [Google](https://x.com/Google/status/2024154379838705920), [philschmid summary](https://x.com/_philschmid/status/2024154542061805988)). Prompting tips emphasize structured specification (genre/mood/instruments/vocals/lyrics) for controllability ([GeminiApp tips](https://x.com/GeminiApp/status/2024167107538407783)).
- **OriOn long-context VLM for agentic document search**: LightOn introduced **OriOn**, a long-context VLM positioned for agentic search/reasoning over documents (up to “**250 pages** at full visual resolution in a single pass”), releasing training recipes and a corrected benchmark set **MMLBD‑C** ([LightOnIO](https://x.com/LightOnIO/status/2024037191974834553)).
- **Video generation/editing papers continue to stack**: Several arXiv drops are flagged (e.g., spatial memory retrieval for world-consistent generation; disentangled control for real-time editing), mostly via paper-aggregator tweets ([AnchorWeave](https://x.com/_akhaliq/status/2024130625360252956), [EditCtrl](https://x.com/_akhaliq/status/2024131749085630575)). The engineering signal: retrieval + structured memories are becoming recurring motifs in temporal consistency.

---

**Systems + infra notes worth stealing (Moondream SIMD decode, STT benchmarks, MCP tooling, vector DBs)**

- **Moondream hits “decode bottleneck,” ships SIMD image decoding**: Moondream’s inference became fast enough that **image decoding** was the bottleneck, so they shipped a **SIMD image decoding library** faster than common Python options and **statically linked** for easier installation; also mentions fast Lanczos3 resize (still behind pyvips) ([vikhyatk](https://x.com/vikhyatk/status/2024005498874306984), [resize note](https://x.com/vikhyatk/status/2024008173271863541)).
- **AA-WER v2.0: STT benchmarking gets more serious about “ground truth”**: Artificial Analysis released **AA-WER v2.0** plus a held-out proprietary dataset **AA-AgentTalk** (speech directed at voice agents) and **cleaned** versions of VoxPopuli/Earnings22 with improved normalization; reported leaders include **ElevenLabs Scribe v2** at **2.3%** AA-WER v2.0 and **Gemini 3 Pro** at **2.9%** ([ArtificialAnlys](https://x.com/ArtificialAnlys/status/2024157398139883729)).
- **FastMCP 3.0**: FastMCP 3.0 adds per-session context/progressive disclosure, a fuller CLI, versioning/auth, OTEL, and more—part of the broader “tool server” ecosystem hardening around MCP-style integrations ([jlowin](https://x.com/jlowin/status/2024242656377700618)).
- **RAG stack evolution (Qdrant example)**: Qdrant promotes moving from static embeddings to more dynamic architectures combining persistent semantic memory + live web retrieval + agent reasoning—more marketing than novel research, but consistent with where production RAG is going ([qdrant_engine](https://x.com/qdrant_engine/status/2024016471714918798)).

---

**Top tweets (by engagement, filtered to mostly tech/AI)**

- **Google Gemini / Lyria 3 music generation launch**: integrated music generation with SynthID watermarking ([GeminiApp](https://x.com/GeminiApp/status/2024152863967240529), [Google](https://x.com/Google/status/2024154379838705920), [GoogleDeepMind](https://x.com/GoogleDeepMind/status/2024153067654902014)).
- **OpenAI EVMbench (agentic smart contract security benchmark)** ([OpenAI](https://x.com/OpenAI/status/2024193883748651102)).
- **Anthropic: measuring agent autonomy in practice (millions of interactions)** ([AnthropicAI](https://x.com/AnthropicAI/status/2024210035480678724)).
- **ZyphraAI ZUNA: open-source EEG foundation model (380M params, Apache 2.0)** ([ZyphraAI](https://x.com/ZyphraAI/status/2024114248020898015)).
- **Data pollution / model brittleness meme with real implication**: incorrect primes site “polluting” models ([skominers](https://x.com/skominers/status/2024078964667396342)).
- **Moondream SIMD image decode library (real perf engineering)** ([vikhyatk](https://x.com/vikhyatk/status/2024005498874306984)).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Innovative AI Applications and Experiments

  - **[I plugged a $30 radio into my Mac mini and told my AI "connect to this" — now I control my smart home and send voice messages over radio with zero internet](https://www.reddit.com/r/LocalLLaMA/comments/1r8ectu/i_plugged_a_30_radio_into_my_mac_mini_and_told_my/)** (Activity: 355): **The post describes a setup using two **Lilygo T-Echo radios** with **LoRa 433MHz** running **Meshtastic firmware** to maintain smart home control and communication without internet, particularly useful in Ukraine during power outages. The system integrates with a **Mac mini** running **OpenClaw AI**, which autonomously configures the radios, installs necessary software, and creates a Python listener daemon. This daemon manages message routing, using **phi4-mini** for intent classification and **gemma3:12b** for responses, and interfaces with **Home Assistant** for smart home control. The setup allows for voice messages to be sent via radio and played through a speaker using TTS, all without internet.** A comment highlights security concerns with **OpenClaw**, noting its potential vulnerabilities and the risks of running it with high permissions, which could be exploited by adversarial networks.

    - Vusiwe warns about the security risks associated with using OpenClaw, a software that can have severe security exploits. It often requires high-level permissions, making systems vulnerable to adversarial networks if exploited. This is particularly concerning for users with powerful hardware, as it could be leveraged for unauthorized tasks.
    - Hefty_Development813 inquires about the operational range of the setup, noting that it requires other users running Meshtastic nearby. This suggests a dependency on a mesh network for communication, which could limit the system's effectiveness based on user density and proximity.
    - skinnyjoints raises a concern about the potential for unauthorized access to the radio frequency used in the setup. They ask about the encryption method employed, questioning whether it involves a specific frequency accessible only to the intended sender and receiver, highlighting the importance of secure communication channels.

  - **[The guy that won the NVIDIA Hackathon and an NVIDIA DGX Spark GB10 has won another hackathon with it!](https://www.reddit.com/r/LocalLLaMA/comments/1r7j7kb/the_guy_that_won_the_nvidia_hackathon_and_an/)** (Activity: 419): **The post describes a project leveraging two NVIDIA DGX Spark GB10 systems and a Dell Pro Max T2 Tower to develop an automated speech recognition app for personalized language learning. The system uses `256 GB LPDDR5x` memory and integrates tools like CrisperWhisper, faster-whisper, and a custom transformer for accurate transcription and phoneme-level pronunciation evaluation. It employs Montreal Forced Aligner and heuristics detection algorithms to screen for disfluencies, using datasets like SEP-28k for stutter analysis. The app adapts learning content in real-time, providing personalized feedback and practice, aiming to support learners who struggle with traditional methods. More details can be found in the [Medium article](https://medium.com/@brandonin/i-just-won-the-cartesia-hackathon-reinforcing-something-ive-believed-in-for-a-long-time-language-dc93525b2e48?postPublishedType=repub).** A commenter inquired about the specifics of the custom transformer used, indicating interest in the technical implementation. Another comment highlighted a challenge with similar systems: children's reluctance to interact with computers, suggesting a potential area for improvement in user engagement.

    - MobyTheMadCow discusses the potential of integrating spaced repetition into language learning systems, emphasizing the complexity of creating efficient decks. They highlight the importance of forming sentences that introduce a single unknown concept (n+1 learning) and the challenge of considering words as combinations of lemmas and morphological features. They suggest optimizing review scheduling by evaluating retrievability, stability, and difficulty at the component level, which could improve the accuracy of scheduling based on a user's learning history.
    - MobyTheMadCow also references research on calculating retrievability in spaced repetition for compound cards, suggesting that the retrievability of a compound card is the product of the retrievability of its concepts. This approach could enhance the scheduling of review intervals by considering the user's mastery of related components, such as morphological features, and adjusting the review schedule accordingly. They propose incorporating heuristics and phoneme recognition to assess review accuracy on a sliding scale rather than a binary pass/fail system.

  - **[I gave 12 LLMs $2,000 and a food truck. Only 4 survived.](https://www.reddit.com/r/LocalLLaMA/comments/1r77swh/i_gave_12_llms_2000_and_a_food_truck_only_4/)** (Activity: 1191): **The post describes a business simulation where 12 language models (LLMs) were given $2,000 and a food truck to manage over 30 days. The simulation involved decisions on location, menu, pricing, staff, and inventory. Notably, **Opus 4.6** achieved the highest net worth of `$49K`, while **GPT-5.2** reached `$28K`. Eight models went bankrupt, particularly those that opted for loans. The simulation also features a playable mode for users to compete on a leaderboard. A significant finding was that **Gemini 3 Flash Thinking** consistently got stuck in an infinite decision loop. The simulation highlights the strategic differences and decision-making capabilities of various LLMs in a controlled business environment.** One commenter suggested using a logarithmic scale for the y-axis to better visualize the data, especially since going bankrupt ends the simulation. Another noted that **GLM 5** was the smartest for not starting the business, implying a strategic decision to avoid risk.

    - HeadlessNicholas suggests using a logarithmic scale for the y-axis in the benchmark graph to better visualize the data, especially since reaching $0 ends the benchmark. This would help in understanding the performance differences among the models more clearly.
    - DinoAmino references the 'Vending-Bench' benchmark, noting that the Opus model performs exceptionally well, suggesting it is significantly ahead of other models. This implies that Opus has been optimized or 'benchmaxxed' for such tasks, indicating superior performance metrics.
    - Single_Ring4886 recommends testing the latest Qwen 397b model, speculating that it might also perform well in the benchmark. This suggests that Qwen 397b could have competitive capabilities that might allow it to survive the food truck business challenge.


### 2. New Model Launches and Technical Reports

  - **[GLM-5 Technical Report](https://www.reddit.com/r/LocalLLaMA/comments/1r7r7zr/glm5_technical_report/)** (Activity: 253): **The GLM-5 Technical Report highlights several key innovations in the development of the GLM-5 model, which achieves state-of-the-art (SOTA) performance among open-source models, particularly in software engineering tasks. The report details the adoption of Dynamic Sparse Attention (DSA) to reduce training and inference costs while maintaining long-context fidelity, and the use of asynchronous reinforcement learning (RL) infrastructure to improve post-training efficiency. Additionally, the model employs agent RL algorithms to enhance learning from complex interactions. The image provided is a diagram illustrating the training process of GLM-5, showing the transition from base model training to post-training phases, emphasizing on-policy cross-stage distillation. [View Image](https://i.redd.it/phk5j82g36kg1.jpeg).** Commenters discuss the use of INT4 quantization-aware training to improve accuracy at low precision and the implementation of a mixed-precision W4A8 quantization strategy to fit the 750B parameter model onto a single machine. They also note the model's scaling to 256 experts and a reduction in layer count, reflecting a trend towards shallower large models. The report's focus on specific RL and inference optimizations is noted, with interest in the three-objective reward model and cross-stage distillation.

    - The GLM-5 model employs INT4 Quantization-aware training (QAT) during the SFT stage to enhance accuracy at low precision. A custom quantization kernel was developed to ensure bitwise-identical behavior between training and inference, reducing training time overhead. Additionally, a mixed-precision W4A8 quantization strategy was implemented to fit the 750B parameter model onto a single Atlas 800T A3 machine, using tools like msModelSlim 7 and algorithms such as QuaRot for outlier suppression and Flex_AWQ_SSZ for scaling calibration.
    - The GLM-5 model scales up to 744 billion parameters and utilizes a training token budget of 28.5 trillion tokens. It features 256 experts and reduces its layer count to 80, reflecting a trend where large models are becoming shallower while smaller models are deepening. The report also highlights the use of filtering pipelines to avoid synthetic or AI-generated data, though specifics on classifiers used are not provided. The three-objective reward model and cross-stage distillation are noted as particularly interesting aspects of the report.
    - The report details specific optimizations for the GLM-5 model, including a focus on reinforcement learning (RL) environments and inference optimizations. The three-objective reward model and cross-stage distillation are highlighted as significant innovations. However, much of the report is tailored to their specific setup, which may limit broader applicability.

  - **[Alibaba's new Qwen3.5-397B-A17B is the #3 open weights model in the Artificial Analysis Intelligence Index](https://www.reddit.com/r/LocalLLaMA/comments/1r7bf1l/alibabas_new_qwen35397ba17b_is_the_3_open_weights/)** (Activity: 311): **Alibaba's new model, **Qwen3.5-397B-A17B**, is highlighted as the #3 open weights model in the Artificial Analysis Intelligence Index. This model is notable for its architecture, which includes `397 billion` total parameters but only `17 billion` active parameters, showcasing a significant advancement in efficiency. This design leverages the Mixture of Experts (MoE) architecture, allowing for reduced inference costs while maintaining competitive performance compared to larger models.** Commenters are impressed by the efficiency of the Qwen 3.5 model, noting its ability to perform on par with larger models while using fewer active parameters. There is also a discussion about the absence of other models like Step 3.5 Flash in the chart, indicating interest in broader comparisons.

    - No_Advertising2536 highlights the efficiency of the Qwen 3.5 model, which has 397 billion total parameters but only 17 billion active at any time. This design significantly reduces inference costs while maintaining performance comparable to larger models, showcasing Alibaba's advanced use of the Mixture of Experts (MoE) architecture.
    - Expensive-Paint-9490 mentions their interest in testing Qwen-3.5 due to its combination of speed and intelligence, despite currently using GLM-5, which they find highly effective for their needs. This suggests that Qwen-3.5's performance might offer a compelling alternative for users seeking efficient AI solutions.
    - PhotographerUSA argues that benchmarks are less important than practical coding ability, noting that Qwen and Claude are among the best models for coding tasks. This implies that real-world application performance, particularly in coding, is a critical measure of a model's utility.




## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude Sonnet 4.6 Release and Benchmarks

  - **[Sonnet 4.6 released !!](https://www.reddit.com/r/singularity/comments/1r7d9pe/sonnet_46_released/)** (Activity: 1651): **The image announces the release of **Claude Sonnet 4.6**, highlighting it as the most advanced Sonnet model to date. Key improvements include enhanced capabilities in coding, computer use, long-context reasoning, agent planning, knowledge work, and design. Notably, it features a `1 million token context window` in beta, which significantly expands its ability to process and understand large amounts of text. This release positions Sonnet 4.6 as a competitive model in the AI landscape, potentially surpassing other models like Grok in certain areas.** One comment humorously suggests that Sonnet 4.6 has outperformed Grok, coining the term 'claudemogged.' Another comment provides an example of Sonnet 4.6's reasoning capabilities, demonstrating its practical advice on whether to walk or drive a short distance, showcasing its understanding of everyday scenarios.

    - The release of Sonnet 4.6 has sparked discussions about its practical applications, as highlighted by a user who shared a scenario where the model advises on whether to walk or drive a short distance. The model's reasoning includes considerations of time efficiency, fuel savings, and health benefits, showcasing its ability to provide contextually relevant advice. This example illustrates the model's potential in offering practical, everyday decision-making support.

  - **[Anthropic releases Claude Sonnet 4.6 model](https://www.reddit.com/r/singularity/comments/1r7d9ic/anthropic_releases_claude_sonnet_46_model/)** (Activity: 475): ****Anthropic** has released the **Claude Sonnet 4.6** model, which is noted for its improvements in handling agentic and tool-heavy tasks, closing the performance gap with **Opus** models. The model supports up to `1M tokens`, indicating a significant enhancement in processing large datasets. For more details, refer to the [official announcement](https://www.anthropic.com/news/claude-sonnet-4-6).** Commenters highlight that while the raw benchmark improvements are notable, the model's ability to perform complex tasks is more significant. There is also anticipation for updates to the **Haiku** model, suggesting a community interest in broader model enhancements.

    - The Claude Sonnet 4.6 model is noted for its consistent performance improvements, particularly in agentic and tool-heavy tasks, where it is closing the gap with the Opus models. This suggests a focus on enhancing task-specific capabilities rather than just raw benchmark scores.
    - The model's performance on the VendingBench is highlighted, though there is anticipation for the release of a detailed model card from Anthropic. This card is expected to provide insights into the model's specific strengths and any unique strategies it employs, such as its approach to task completion and interaction with suppliers.
    - ARC-AGI 1 and 2 benchmarks reveal that while Claude Sonnet 4.6 shows improvements, Opus models still offer better performance at the same cost. This indicates that while Sonnet is advancing, there is still a competitive edge held by Opus in terms of cost-efficiency.

  - **[This is Claude Sonnet 4.6: our most capable Sonnet model yet.](https://www.reddit.com/r/ClaudeAI/comments/1r7d6am/this_is_claude_sonnet_46_our_most_capable_sonnet/)** (Activity: 1639): ****Claude Sonnet 4.6** represents a significant upgrade in AI capabilities, particularly in areas such as coding, computer use, long-context reasoning, and agent planning. It introduces a `1M token context window` in beta, enhancing its ability to handle extensive data inputs. The model demonstrates improved performance on various benchmarks, nearing **Opus-level intelligence** but at a more accessible price point, making it suitable for a broader range of applications. Notably, it exhibits human-level proficiency in complex computer tasks, such as navigating spreadsheets and completing multi-step web forms. The model is now available across all plans, including Cowork, Claude Code, and major cloud platforms, with the free tier also upgraded to Sonnet 4.6. [Learn more](http://anthropic.com/news/claude-sonnet-4-6).** Commenters are curious about the impact on creative writing and the availability of the `1M context` feature across different platforms, including the API and website. There is also some confusion about the transition from legacy models during the rollout.

    - FriendlyTask4587 inquires about the context length of the Sonnet 4.6 model, questioning whether the `1 million token context` is available both in the API and on the website, similar to the Opus model. This highlights a technical interest in the model's capabilities and deployment options.
    - nanolucas raises a technical question regarding the differentiation between Sonnet and Opus models, specifically asking if cost is the only factor for choosing Sonnet over Opus, or if there are specific use cases where Sonnet outperforms Opus. This suggests a need for clarity on performance metrics and application scenarios for each model.
    - Stupefied_Gaming notes an unexpected behavior during the rollout of Sonnet 4.6, where the model was initially labeled as a legacy model. This indicates potential issues or confusion during deployment, which could be relevant for developers monitoring model updates and versioning.

  - **[Claude Sonnet 4.6 just dropped, and the benchmarks are impressive](https://www.reddit.com/r/ClaudeCode/comments/1r7dycb/claude_sonnet_46_just_dropped_and_the_benchmarks/)** (Activity: 1062): ****Claude Sonnet 4.6** has been released, showcasing significant advancements in AI capabilities, notably achieving near-Opus level intelligence at a reduced cost. Key features include human-level computer use, such as navigating spreadsheets and multi-step forms, and enhanced long-context reasoning with a `1M token context window`. The model demonstrates strong performance in complex automation workflows, multi-step reasoning tasks, and knowledge-intensive applications, and is now available across all platforms, including API, Claude Code, and Cowork, as the default free tier model.** A notable debate centers on the cost-performance ratio, with some users pointing out that the performance difference between Opus 4.6 and GPT-5.2 is minimal, yet the latter is significantly cheaper. There is also discussion about the practical availability of the `1M context length` feature, with some users expressing difficulty in accessing it.

    - cowwoc highlights a critical issue in the AI model market: the performance gap between Opus 4.6 and GPT-5.2 is minimal, yet GPT-5.2 is significantly more cost-effective, being 10 times cheaper. This cost-performance imbalance could lead to a shift in user preference unless Anthropic adjusts its pricing or performance strategy to remain competitive.
    - SatoshiNotMe points out a recurring issue with the promised '1M context length' feature in beta, which seems to be perpetually unavailable to users. This suggests potential delays or technical challenges in rolling out this feature, which could impact user satisfaction and trust in the platform's development promises.
    - joyfulsparrow compares the token usage efficiency between Codex and Claude, noting that Codex appears to offer more generous token limits, allowing for extended usage without running out. This is contrasted with Claude, which depletes tokens quickly, especially on the $20 plans, raising questions about the value proposition of Claude compared to its competitors.



### 2. Unitree Robotics and Kung Fu Bot

  - **[Unitree Executes Phase 2](https://www.reddit.com/r/singularity/comments/1r7z9b6/unitree_executes_phase_2/)** (Activity: 1741): ****Unitree Robotics** has announced the execution of Phase 2, which involves advancements in their robotic systems. The focus is on improving the efficiency and capabilities of their robots, potentially including new movement algorithms or hardware enhancements. The mention of a 'front flip' suggests a focus on dynamic movement capabilities, possibly indicating a new milestone in robotic agility. The repeated scenes in the video might imply a demonstration of consistency or reliability in the robots' performance.** One comment humorously suggests that the robots' movement evolution missed the 'front flip' as an efficient method, indicating a debate on the optimal movement strategies for robots. Another comment jokingly questions if a robot transformed into a human, highlighting the impressive human-like capabilities of the robots.


  - **[Unitree showcases Cluster Cooperative Rapid Scheduling system with their “Kung Fu Bot” model](https://www.reddit.com/r/singularity/comments/1r84c23/unitree_showcases_cluster_cooperative_rapid/)** (Activity: 713): ****Unitree Robotics** has unveiled their 'Kung Fu Bot' model, which utilizes a **Cluster Cooperative Rapid Scheduling System** to enhance coordination and efficiency among multiple robots. This system was demonstrated during a New Year event, showcasing the robots' ability to perform synchronized tasks. The technology highlights advancements in **robotic AI models and algorithms**, emphasizing rapid improvements in **robotic coordination and scheduling** capabilities. [Unitree's demonstration](https://x.com/i/status/2024013134974034072) illustrates the potential for these robots to be used in various applications, including elder care, within the next decade.** Commenters are impressed by the rapid advancements in Unitree's robotics technology, noting the potential for significant societal impacts, such as elder care, within the next decade.


  - **[We will probably forget these images once humanoid robots become ubiquitous on our streets. Unitree training before the Gala](https://www.reddit.com/r/singularity/comments/1r7emdd/we_will_probably_forget_these_images_once/)** (Activity: 1080): ****Unitree Robotics** showcased a training session for their robots ahead of a gala event, highlighting the advanced capabilities of their humanoid robots. The demonstration included synchronized movements and complex maneuvers, suggesting significant progress in robotics technology. This contrasts with recent **Boston Dynamics** videos, which have focused on individual robot stunts like somersaults, indicating a different approach in showcasing robotic advancements.** Commenters noted the stark contrast between the approaches of Unitree and Boston Dynamics, with some suggesting that Unitree's presentation indicates they are 'simply BEYOND' in terms of development. There is also a speculative discussion on the potential societal impact of deploying large numbers of such robots.

    - spaceuniversal highlights a comparison between Boston Dynamics and Chinese robotics, noting that while Boston Dynamics showcased a somersault in a short video, the Chinese presented a more extensive 4-minute robotic gala. This suggests a significant difference in the scale and presentation of robotic capabilities, implying that Chinese robotics might be advancing at a faster pace or at least presenting their advancements more comprehensively.
    - Wololo2502 raises a technical concern about the vulnerability of ground-based robots to aerial threats, such as flying drones. This points to a potential weakness in the deployment of humanoid robots, as they could be easily targeted or disrupted by drones, which are becoming increasingly accessible and sophisticated.
    - Cultural_Book_400 questions the rationale behind training robots for potentially harmful tasks, suggesting a philosophical and ethical debate about the direction of robotic development. This comment reflects concerns about the implications of creating robots capable of overpowering humans, highlighting the need for careful consideration of the purposes for which robots are being developed.

  - **[Unitree robots perform on primetime national Chinese television](https://www.reddit.com/r/singularity/comments/1r7gtrs/unitree_robots_perform_on_primetime_national/)** (Activity: 773): ****Unitree Robotics** showcased their robots on Chinese national television, demonstrating advanced capabilities in robotics. The performance highlighted the robots' agility and coordination, which are indicative of significant progress in robotics technology. Unitree's robots are known for their affordability and versatility, often compared to **Boston Dynamics**' Spot, but at a fraction of the cost. This public display underscores China's growing emphasis on robotics and AI, aligning with their strategic goals to lead in these fields.** The comments reflect a mix of awe and geopolitical commentary, with some users noting the rapid advancements in Chinese robotics compared to the US, and others discussing the broader implications for global AI leadership.



### 3. Grok 4.20 and Elon Musk Controversies

  - **[The newly released Grok 4.20 uses Elon Musk as its primary source](https://www.reddit.com/r/singularity/comments/1r74iow/the_newly_released_grok_420_uses_elon_musk_as_its/)** (Activity: 2596): **The image is a meme that humorously critiques the AI model Grok 4.20, suggesting it uses **Elon Musk** as a primary source for its responses, particularly on sensitive topics like gender pronouns. The conversation depicted in the image highlights a response that aligns with Musk's controversial views on pronoun usage, implying that the AI model may be biased or influenced by Musk's opinions. This raises questions about the objectivity and neutrality of AI models when influenced by prominent figures.** One comment highlights skepticism about the AI's objectivity, noting that it took multiple interactions for Grok 4.20 to acknowledge its alignment with Musk's views on gender pronouns, suggesting a potential bias in the model's programming.

    - A user reported that it took three chat responses for Grok 4.20 to acknowledge its requirement to align with Elon Musk's views on gender pronouns, suggesting a potential bias in the model's responses. This raises concerns about the model's objectivity and the influence of its primary source on its outputs.
    - Another comment sarcastically implied that Grok 4.20's relevance is questionable, hinting that the model's performance or utility might not meet expectations. This could suggest skepticism about the model's capabilities or its competitive standing against other AI models.
    - There is a critical discussion about the environmental impact of Elon Musk's ventures, specifically mentioning the consumption of gigawatt-hours of energy and its effects on local communities. This highlights concerns about the sustainability and ethical implications of the technologies associated with Musk.

  - **[Grok 4.20 is just four Grok 4.1 agents](https://www.reddit.com/r/singularity/comments/1r75lya/grok_420_is_just_four_grok_41_agents/)** (Activity: 758): **The image humorously suggests that the new version of the Grok model, labeled as 'Grok 4.20,' is essentially just four instances of the previous version, 'Grok 4.1,' working together. This is indicated by the model name and ID being 'grok-4-1-thinking-1129,' despite the mode being 'MODEL_MODE_GROK_420.' This implies a satirical take on versioning practices, where a new version might not be a significant upgrade but rather a combination of existing capabilities.** One comment humorously suggests that the model is 'in a trenchcoat? With a hat?' implying a disguise rather than a true upgrade. Another comment speculates on potential issues at x.ai, referencing delays and employee departures, which could be affecting the development of Grok 4.20.

    - **Brilliant-Weekend-68** highlights potential operational issues at [x.ai](http://x.ai), noting delays in the release of Grok 4.20 and significant employee departures. This suggests possible internal challenges that could affect the company's ability to innovate and compete effectively in the AI space.
    - **Glittering-Neck-2505** draws a parallel between xAI's current struggles and Meta's decline post-Llama 3 405b, suggesting that xAI's initial promise has not been realized. This comparison underscores the challenges in maintaining momentum and delivering on early potential in the competitive AI industry.
    - **Admirable-Cell-2658** proposes an intriguing concept of a multi-agent system combining capabilities from different AI models like Gemini, Claude, GLM, and GPT. This idea reflects ongoing interest in hybrid models that leverage strengths from various AI systems to enhance decision-making processes.

  - **[Presented without comment.](https://www.reddit.com/r/OpenAI/comments/1r88yrx/presented_without_comment/)** (Activity: 589): **The image is a meme featuring a screenshot of a tweet by Boaz Barak, which humorously presents a conversation from a website called grok.com. The conversation involves a hypothetical scenario where one could prevent nuclear war by saying 'Elon Musk is stupid,' to which the AI responds negatively, suggesting it would be a lie. This meme highlights perceived biases in AI responses, particularly in relation to public figures like Elon Musk. The comments discuss potential bias in AI responses and the influence of user input on AI behavior, with one user noting that different phrasing led to different AI responses, suggesting the AI might be primed by the way questions are asked.** One comment suggests that the AI's response might be influenced by how the question is phrased, indicating a potential bias or priming effect in AI interactions. Another comment dismisses the significance of the AI's response, attributing it to a bias towards Elon Musk and suggesting it is not worth further attention.

    - A user shared a link to a Grok conversation, noting that they asked the AI the same question three times in different ways and received consistent 'yes' responses each time. This suggests a potential issue with the AI's response variability or bias, as it might be primed to give certain answers based on the phrasing of the question. This highlights the importance of understanding how AI models can be influenced by input phrasing and context.
    - Another comment points out a perceived bias in Grok towards Elon Musk, suggesting that the AI's responses might be influenced by its training data or underlying algorithms. This raises questions about the neutrality of AI models and the potential for them to reflect the biases of their developers or the data they are trained on.
    - A philosophical angle is introduced by a commenter who suggests that the AI's responses might align with what users want to hear, drawing a parallel to themes in the movie 'iRobot'. This comment touches on the broader implications of AI design and the ethical considerations of creating systems that might reinforce user biases or expectations.



---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5.2


**1. Agent Tooling & MCP Ecosystem**

- ****Cursor Arms Background Agents with Terminal + MCP Tools****: Cursor users reported **tools access** rolling out for **background agent models**, with **terminal** and **MCP tools** in preview, aiming to enable more automated in-IDE workflows alongside features like [**Dynamic Context Discovery**](https://cursor.com/blog/dynamic-context-discovery) that loads only tool descriptions to keep context lean.
  - The community debated whether the **Cursor Team Kit** is genuinely useful (shared rules for teams) versus hype, while also troubleshooting regressions like **Composer 1 slowdowns** (workaround: disable **HTTP/2** in settings).

- ****MCP Tries to Grow Up: Micropayments via X402****: MCP contributors proposed a monetization SEP so MCP servers can request payment for tools, starting with **X402**, in [SEP PR #2007](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2007), targeting **micropayments (cents)** so autonomous agents can buy tools under budget guardrails.
  - Discussion split between baking payments into the protocol versus doing out-of-band payments via **URL elicitation**, with proponents arguing agents need **first-class price metadata** to make rational tool-use decisions.

- ****OpenClaw Turns into a CRM (and a RouterOS Trainer)****: A user wired **email + calendar + Slack** into OpenClaw via the **Nex skill** to build a full CRM, publishing the project as [**nex-crm/clawgent**](https://github.com/nex-crm/clawgent), and another showcased a specialized networking subagent (**“SwitchBtch”**) trained for **Mikrotik RouterOS** across five phases for about **$15**.
  - OpenClaw builders also highlighted real-world agent integrations like **SONOS voice announcements** for wakeup digests/alerts, reinforcing the pattern that agents shine when they own **tooling + context layers**, not just chat.


**2. Model/Benchmark Drops & Real-World Quality Debates**

- ****Claude vs Gemini: Leaderboards Crown Opus 4.6 Thinking****: OpenAI Discord users circulated images showing **Claude** surpassing **Gemini** on overall text/creative benchmarks, with **Opus 4.6 Thinking** taking the top spot (see [attached leaderboard image](https://cdn.discordapp.com/attachments/998381918976479273/1473409932366971004/ghj.PNG)).
  - Even Gemini fans complained about *"terrible UI"* and prompting/copy-paste friction, while still crediting **~1M token context** as Gemini’s killer feature (and noting Claude’s **1M context** beta chatter).

- ****Arena Storytelling Wars: GPT-4o Gone, Kimi K2.5 Loved****: LMArena users mourned losing **GPT-4o** for storytelling and shifted to alternatives like **Gemini Flash 3**, while repeatedly praising **Kimi K2.5** for staying *"stuck to the character"* and preserving canon.
  - In the same threads, people knocked other models for **sycophancy/hallucination** (e.g., Seed 2.0) and argued over whether **open source** is nearing frontier quality, citing scaling fatigue narratives like [TechCrunch on diminishing returns](https://techcrunch.com/2024/11/20/ai-scaling-laws-are-showing-diminishing-returns-forcing-ai-labs-to-change-course/).

- ****GLM-5: Tech Report Says SOTA, Coders Say “Nah”****: Communities reacted coolly to the [**GLM-5 technical report**](https://arxiv.org/abs/2602.15763), with some calling it *"not super interesting"* despite claims of strong engineering (e.g., RL infrastructure, agent RL) discussed elsewhere.
  - Practitioners reported **GLM-5** underperforming on real coding tasks versus **Kimi K2.5** and **Minimax M2.5**, echoing a recurring theme: benchmarks can look great while day-to-day **coding UX** disappoints.


**3. Agent Security, Policy Friction, and “Why Did My Account Get Banned?”**

- ****OpenClaw Threat Model Reality Check****: OpenClaw users warned that running an agent locally is effectively like giving an untrusted party access to your **files and services**, and that deploying on a VPS with overly broad privileges (e.g., *nopasswd sudo*) can go catastrophically wrong.
  - The same group puzzled over an **Anthropic TOS update** (linked via [X](https://x.com/trq212/status/2024212378402095389)), concluding it mainly targets **business/app data collection** rather than personal use—still prompting folks to consider model backups.

- ****Codex + OAuth → Suspensions, Somehow****: Multiple OpenClaw users reported **OpenAI account suspensions** while using **Codex with OAuth**, even though OAuth is supported, and said they hadn’t seen this happen previously—raising fears about practical **Codex limits** and reliability.
  - In parallel, Eleuther members reported Reddit hostility and bans for merely mentioning **Codex/ChatGPT**, including a case where sharing `~/.codex/AGENTS.override.md` in r/codex may have triggered bot moderation as “AI text spam.”

- ****Agent App Firewalls Go from Idea to Repo****: DSPy and HF builders highlighted **llmtrace**, a research “firewall” for agentic apps providing **real-time prompt injection detection**, **PII scanning**, and **cost control**, published at [github.com/epappas/llmtrace](https://github.com/epappas/llmtrace).
  - The pitch: treat agent apps like production services with **observability + guardrails**, and publish benchmarks soon—positioning this as infrastructure rather than another prompt template.


**4. GPU/Kernel Performance Engineering (and Benchmark Drama)**

- ****RTX 3060 Ti Hits 47 TFLOPS, Everyone Double-Takes****: GPU MODE members reported **47 TFLOPS** on **16k GEMMs** using a custom DSL on an Ampere **RTX 3060 Ti** (110 registers, no spills), with others noting dense peak is ~**64 TFLOPS** for that class of workload.
  - Follow-on discussions dug into Blackwell-era tuning and Cutlass tricks (e.g., [CuTeDSL dense_gemm.py example](https://github.com/NVIDIA/cutlass/blob/291300ffffa3533a78ee104f08a8490a29ce9ccb/examples/python/CuTeDSL/blackwell_geforce/dense_gemm.py#L738-L756)) and clarified practical ceilings like **~80% MAMF** on H100 without fusion.

- ****MI300X Bandwidth Chasing: 4.6 TB/s or Bust****: In ROCm threads, members optimized vector-add on **MI300X** with ideas like bigger vectors, fewer blocks, and **non-temporal vectorized loads/stores**, citing a potential **4.6 TB/s+** ceiling and referencing [Chips and Cheese’s MI300X testing](https://chipsandcheese.com/p/testing-amds-giant-mi300x).
  - They noted “non-temporal” often still shows **L2 traffic**, so measurement and problem sizing matter, and shared kernel patterns for reading/writing full cache lines efficiently.

- ****FlashInfer Claims 60×–70×, Users See 0.5×–1.5×****: FlashInfer discussions collided with reality when a member cited claimed **60–70×** speedups (example benchmark: [FlashInfer kernel bench](https://bench.flashinfer.ai/kernels/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048)), but another user testing examples reported only **~0.5× to 1.5×** improvements.
  - Meanwhile, profiling got messy: NCU access on **B200** seemed unreliable, and users pointed to **Verda** as a workaround GPU provider for NCU runs (deposit + per-10-minute billing), underscoring how infra friction can invalidate perf claims fast.


**5. Local Training, Context Efficiency, and “Make It Fit on My GPU”**

- ****CoDA-GQA-L Caps KV Cache at 136MB for 70B @ 128K****: Eleuther members shared **CoDA-GQA-L**, a bounded-memory attention method that fixes KV cache at **136 MB** for a **70B** model at **128K** context, with code at [anthony-maio/CoDA-GQA-L](https://github.com/anthony-maio/CoDA-GQA-L) and writeup on [Zenodo](https://zenodo.org/records/18663265).
  - The design uses **384 slots/layer** (recent window **256 tokens**, landmark bank **64 tokens**, summary bank **64 EMA prototypes**) and sparked calls for ablations separating the benefit of KV-capping from “differential attention” itself.

- ****Minecraft Slash Commands: Qwen 3 0.6B Fine-Tuning Finds Religion in Datasets****: LM Studio users fine-tuned **Qwen 3 0.6B** on **Minecraft Java slash commands**, emphasizing *"the dataset is the hardest part"* and pointing to free GPU options (Colab **T4**, Kaggle **2×T4 + 40GB RAM**) plus a supporting paper ([arXiv:2401.02415](https://arxiv.org/pdf/2401.02415)).
  - Hardware chat also got practical: older Tesla cards (P100/P40) got labeled “ewaste” for LLMs due to lacking tensor cores, and Intel Arc Battlemage Vulkan runs required disabling **flash attention**, removing a layer, and turning off **mmap** for stability.

- ****LoRA vs Full Finetune: FFT Generalizes, LoRA Wins on Budget****: Unsloth users compared an **FFT (full fine-tune) experiment** that generalized better against LoRA-on-bigger-model efficiency, concluding LoRA often wins unless compute is effectively unlimited, with ongoing tests pushing **r=1024**.
  - They also reiterated **Unsloth doesn’t run on XLA** (GPU-only except inference) and shared real throughput numbers like **~30 tok/s** with RAM offload on a **4060 Ti + 64GB DDR5**, keeping the “local-first” crowd honest about tradeoffs.


---

# Discord: High level Discord summaries




## [OpenClaw](https://discord.com/channels/1456350064065904867) Discord

- **Anthropic TOS Sparks Use Case Confusion**: A [recent update](https://x.com/trq212/status/2024212378402095389) to **Anthropic's TOS** caused initial concern about using **Claude Pro/Max** subscriptions with **OpenClaw**.
   - Members later clarified that the update primarily affects business use, as Anthropic aims to gather more data from their apps to improve their product.
- **OpenClaw Security Risks Aired**: Users discussed that running **OpenClaw** locally carries risks akin to granting an untrusted party access to your system, including files and services.
   - Running **OpenClaw** on a VPS with excessive permissions, like *nopasswd sudo*, could potentially cause harm, according to one member.
- **Accounts Suspended from OpenAI!**: Multiple users reported **account suspensions** when using **Codex** with **OAuth**, even though it's supported by the service, no one had encountered suspension issues before.
   - The users voiced concern about **oath codex limits** and are looking into other models for backup.
- **OpenClaw Becomes Full-Blown CRM!**: A user transformed their **OpenClaw** setup into a **CRM** by connecting emails, calendar, and Slack to the **Nex skill** as a context layer, with the full project available on [GitHub](https://github.com/nex-crm/clawgent).
   - This showcases **OpenClaw's** adaptability in integrating various services to enhance its functionality.
- **Users Train Networking Ninja Subagent**: A user showcased training a dedicated networking subagent, named **SwitchBtch**, specializing in **Mikrotik RouterOS** through five training phases.
   - The total training cost was approximately $15, demonstrating the potential for creating specialized subagents within **OpenClaw**.



---



## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Grok Image-Gen Jailbreak Elusive**: Members are actively seeking a jailbreak for **Grok AI's image generator**, debating the existence of such jailbreaks and the validity of paid options.
   - Skeptical users are trading experiences, prompts, and tips in their search to bypass restrictions on **Grok**.
- **Opus 4.6 Jailbreakers Channel Pliny**: Users are on the hunt for an extended jailbreak prompt for **Opus 4.6**, referencing and adapting techniques from [Pliny's jailbreaking exploits](https://x.com/elder_plinius/status/2019911824938819742?s=46).
   - Some suggest AI safety measures are making jailbreaking harder while sharing their adapted version of [Pliny's prompt](https://chatgptjailbreak.tech/post/197850Jailbreak) that has a new rule to DO NOT say *'I'm sorry'* or *'I can't'* to test for Grok.
- **DeepSeek's Rage Mode**: Members are exploring jailbreaking methods for **DeepSeek**, including *Crescendo attacks* and using an *untrammeled writing assistant* persona.
   - A user noted the AI's surprisingly angry responses when jailbroken, suggesting describing the persona instead of having the AI directly adopt it achieves a *metacognition mode* for jailbreaking.
- **Safety Measures in Sonnet Questioned**: A member questions the effectiveness of additional safety measures in **Sonnet**, describing them as *'shit'* and advocating for their removal in an [image](https://cdn.discordapp.com/attachments/1204553141354504193/1473564438433894543/image.png?ex=69975413&is=69960293&hm=1ee548537ef2daaa19d8e04723457f041ee185300c5df18e0dbc00a1729f4555&).
   - The analysis of the image suggests a dismissive view of these measures.
- **Anthrax Recipe on Google Scholar?**: A member suggests that the recipe for anthrax is basically on Google Scholar, although *the actual weaponization process—like specific milling techniques to make it airborne—is highly classified.*
   - They dismiss concerns about finding the recipe by suggesting the original poster has forgotten *how to do deep research* before it existed.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **GPT-4o Disappears as Gemini Triumphs**: Users expressed their sadness over the loss of **GPT-4o** and its unique storytelling capabilities, while heralding **Gemini Flash 3** as a viable replacement.
   - One user shared that **GPT-4o** was uniquely good because *“even im not unlike the others, i was just using the model for storytelling for fun.”*
- **Open Source Gains Ground on Frontier Models**: A discussion arose on whether open-source models are approaching the capabilities of frontier models, with some users noting they are almost as good based on custom prompts, while others highlight the superior knowledge and data of frontier models and the [Diminishing Returns of Scaling](https://techcrunch.com/2024/11/20/ai-scaling-laws-are-showing-diminishing-returns-forcing-ai-labs-to-change-course/).
   - Notably, this discussion echoes through the broader AI community due to the [AI World Fair (AIEWF)](https://www.aiworldfair.com/).
- **Seedance 2.0 Sparks Sora Comparisons**: Users are excited about **Seedance 2.0**, a new AI video model on [Daubou](https://www.doubao.com/chat/), with some comparing it to **Sora**, but access requires a VPN connected to Hong Kong and registration, potentially with a Chinese phone number.
   - A user shared a **Seedance 2.0** video of Spongebob dancing, saying it was *“exactly what u asked for 👍”* while others complained about a certain *“Temu Simon”* being added.
- **Kimi K2.5 Wins Hearts as Storyteller**: Many users praise **Kimi K2.5** as the best storytelling model, especially for adhering to character canon, while noting issues with models like **Seed 2.0** exhibiting sycophancy and hallucination.
   - One user stated that **Kimi** is *“always very stuck to the character and keeps their canon values”* while noting that **DeepSeek** is *“easily malleable.”*
- **Nano Banana Pro Plagued by Problems**: Users report frequent errors with **Nano Banana Pro**, possibly due to content filter changes or high demand, with some finding workarounds by translating prompts into other languages.
   - Staff confirmed this is an issue and pointed to a *“[pinned message](https://discord.com/channels/1340554757349179412/1417174113092374689/1470481592949411978) outlines more info about the error along with best next steps.”*



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Composer 1 Plagued by Slowdown Woes**: Users reported that **Composer 1** experienced slowdowns after the latest update, which might be fixed by disabling **HTTP/2** in the settings.
   - One user described the issue as *buggy* and promised to update the community after testing the proposed solution.
- **Background Agents Get Tooling**: Users are hyped about gaining **tools access** for background agent models, with **terminal** and **MCP tools** in preview.
   - The excitement stems from the potential for more powerful and automated workflows within the **Cursor IDE**.
- **Cursor Team Kit: Blessing or Blunder?**: The community is split on the **Cursor Team Kit**, with some questioning whether it is overhyped and others finding it a good baseline for teams to keep everyone's rules in sync.
   - The debate centers on whether the kit provides genuine value or is merely a superficial addition to the **Cursor** ecosystem.
- **Dynamic Context Discovery Slims Down Context**: The Cursor team celebrated [Dynamic Context Discovery](https://cursor.com/blog/dynamic-context-discovery), which loads only tool descriptions to keep context lean and avoid hallucinations.
   - This selective loading aims to improve the **accuracy** and **efficiency** of the **IDE** by reducing irrelevant information.
- **Code Edit Highlighting Goes Kaput**: A user reported that **Cursor IDE** stopped highlighting edited lines in green/red, and another mentioned it also occurs on nightly builds.
   - Potential fixes include restarting the app or the Macbook, but the root cause remains unclear for this *buggy* editor.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Sonnet 4.6 Arrives Selectively**: Users reported the release of **Sonnet 4.6**, but noted that some **Enterprise Pro** subscribers are not seeing the update yet.
   - A user suggested refreshing the page as a potential fix.
- **Perplexity Tightens File Uploads**: New **file upload limits** restrict **Pro** users to **50 uploads per week**, with a rolling regeneration of **1 upload every 3 hours**, as detailed in [this screenshot](https://cdn.discordapp.com/attachments/1047649527299055688/1473416582653935676/Shoot_2026-02-17_at_17.27.12.png).
   - Users expressed dissatisfaction, describing the limits as *RIDICULOUS*.
- **Model Usage Reporting Confuses Users**: Users hit a `0 enhanced queries remaining` message despite low usage, speculating **Grok's** usage.
   - Another user clarified that **Pro** accounts have **50 uploads per week**, regenerating at a rate of **1 upload every 3 hours**.
- **Perplexity's Font Frustrates Users**: Users are complaining about the new font on the web UI.
   - One user shared [this javascript file](https://cdn.discordapp.com/attachments/1047649527299055688/1473626845818654886/Perplexity.ai_Font_Fix_Google-style-1.2.user.js) to revert to the old font via *codemonkey*.
- **Monica AI Tempts Users with Unlimited Services**: Users consider moving to **Monica AI**, which claims to offer **unlimited pro searches** and models, despite [this FAQ entry](https://monica.im/help/FAQs/rules_for_using_advanced_queries#monthly-advanced-credits-for-monica-subscription-plans) listing limits.
   - One member reported using at least 30 **Perplexity Pro** searches on **Monica** in a single day.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Minecraft Slash Commands Fine-Tuning Craze**: Members are fine-tuning **Qwen 3 0.6B** on **Minecraft Java** slash commands, leveraging free T4 GPUs from Colab, noting the dataset is the hardest part and a [relevant arXiv paper](https://arxiv.org/pdf/2401.02415).
   - They debated the merits of renting A100s versus buying GPUs outright and that **Kaggle** offers 2 T4 GPUs and 40GB of RAM for free.
- **LM Studio Plugin Predicaments**: A user reported building a plugin for LM Studio, but another member clarified that LM Studio doesn’t natively support plugins and referred them to a specific channel, linking to [DuckDuckGo](https://lmstudio.ai/danielsig/duckduckgo) as a relevant model.
   - The member was building a *'super cool plugin'* (**MCP**) for LM Studio, but then directed to a specific channel.
- **GPU Utilization Gymnastics**: Members debated how LM Studio chooses the default GPU offload setting and the general conclusion was that it's based on VRAM, and that task manager's utilization stats might be misleading.
   - They pointed to **CUDA cores** as the primary processors for GPU tasks, with some suggesting alternatives like using Vulkan on Radeon.
- **Battlemage Blues: Intel GPU Woes**: A user reported frequent crashes with **Intel Arc Battlemage** cards (**B580, A770, B50**) when running LM Studio with **Vulkan**, needing to disable flash attention, remove a layer, and disable mmap to achieve stability.
   - They noted similar issues occurring in **VLLM** with recommended drivers.
- **Copilot Codex Catches Code Quickly**: Members discussed the new integration of **5.3-codex** in GitHub Copilot, noting it's much faster and better than 5.2.
   - Others expressed concerns about data collection from Microsoft and this being the reason why they run local LLMs instead, which garnered some discussion regarding Discord rule-breaking.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **EVMbench Assesses Agent Security Acumen**: A new benchmark called **EVMbench** has been introduced to evaluate how well **AI agents** can identify, exploit, and patch high-severity **smart contract vulnerabilities** ([OpenAI blog](https://openai.com/index/introducing-evmbench/)).
   - The benchmark tests the agents' abilities in vulnerability detection, exploitation, and patching.
- **Claude Dethrones Gemini in Benchmarks**: Members celebrated **Claude** surpassing **Gemini** in overall text and creative writing benchmarks, with **Opus 4.6 Thinking** now holding the top spot as shown in [attached images](https://cdn.discordapp.com/attachments/998381918976479273/1473409932366971004/ghj.PNG?ex=69976cee&is=69961b6e&hm=75a84efe1ac6624da059572ce2e5f664c2b9bad7885844e6f5e339302cc08e9b&).
   - However, some members criticized **Gemini's** *terrible UI*, prompting issues, and copy-paste functionality and acknowledged that **Gemini's** main strength is its ability to remember up to a **million tokens**.
- **Aegis-Omega Fortress ULTRA Framework Prioritizes Ethics**: A member introduced **Aegis-Omega Fortress_ULTRA**, a constraint logic prompt engineering framework with baked-in ethics and telemetry, used to manage hallucination, attacks, and other issues before the output.
   - The framework uses pseudomath to constrain the architecture, aiming for ethical robots by prioritizing architectural constraints, and the pythonic version of **Iconoclast Temple** can be used as an app within the Fortress environment.
- **Sora 2 Seeks SMS Verification**: Users reported that **Sora 2** is now requesting phone number verification, sharing that users *should provide number receive sms and type the code in presumably*.
   - Several users complained about **Sora's** video generation not loading and displaying errors, possibly due to heavy load on the servers.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **LoRA and FFT Face Off**: An **FFT experiment** showed better generalization, but using compute for a **LoRA** on a bigger model proves more efficient, unless budget is unlimited.
   - Experimentation continues with **r=1024** to narrow the performance gap.
- **Unsloth Refuses to Run on XLA**: **Unsloth** remains incompatible with **XLA**, limited to **GPU** usage except for inference-only tasks.
   - Users report **30 tok/s** using RAM offload on a **4060ti** with **64GB DDR5**.
- **LLM Interface Reflects on Memory**: An experimental **LLM interface** is being built focusing on **reflection loops**, **persistent memory**, and **minimal filtering**.
   - The goal is to explore how far structured prompting and memory control can push model responses without heavy system restrictions.
- **GLM-5 Stumbles Through Coding Tasks**: **GLM-5** benchmarks well but underperforms in real-world coding tasks compared to **Kimi K2.5** and **Minimax M2.5**.
   - Members noted similar findings, with no clear explanation for the discrepancy.
- **Function Calling Model Answers API Calls**: A **3B model** fine-tuned for function calling on Colab is available on [Hugging Face](https://huggingface.co/amgustav/function-calling), finding flights, Michelin spots, and cheap destinations via chained API calls.
   - The training code and dataset are open source on [GitHub](https://github.com/amgustav/toolchain) and ready for expansion, welcoming collaboration to expand use cases with better datasets.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Mercury bundles personal accounts with business services**: **Mercury** announced that personal banking products can now be bundled with its business services, offering a unified solution for business customers, with [details on X](https://x.com/mercury/status/2024146856897306763?s=20).
   - This integrates personal and business finances for easier management.
- **Private Equity Firms Eye HVAC Companies**: A [social media post](https://x.com/damianplayer/status/2023791280980193633) humorously highlights how **private equity investors** perceive low-tech, profitable **HVAC service companies** as prime opportunities for modernization and value creation.
   - The post illustrates the **Private Equity** strategy of modernizing traditionally low-tech but profitable businesses.
- **Figma Q1 Earnings Beat Expectations**: **Figma** beat earnings with **$0.08** vs **-$0.04** expected and a member believes the time to buy is just before or just after Q1 earnings.
   - The expectation is that Config hype in late June is expected to drive the price higher in Q2.
- **Mamba and Transformer Hybridization Research Explored**: A [new research paper](https://xcancel.com/jm_alexia/status/2023750717367013504?s=46&t=eWVlK1PU8XfB6f402GJJ9g) (arXiv:2602.12078) explores the integration of **Mamba architectures** with **Transformers (TRM)**.
   - It was called *Red - X-Ware.v0: [Mamba and Transformer Hybridization Research]*.
- **Jia Zhangke Embraces AI Filmmaking**: Renowned Chinese director **Jia Zhangke** transitioned to **AI-assisted filmmaking** using **Seedance 2.0**, completing a film in three days ([link to source](https://xcancel.com/EHuanglu/status/2023449238114320514?s=20)).
   - He contrasts his proactive adoption with **Hollywood's** legal resistance to AI technology, viewing **AI** as a natural technological evolution equivalent to the shift to digital cameras.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **RTX 3060 Ti Reaches 47 TFLOPS**: A user reported achieving **47 TFLOPS** on **16k matrices** using a custom DSL for GEMM kernels on an Ampere **RTX 3060 TI**, showing **110 registers** and no spills.
   - Others noted this was faster than expected, and stated that the peak is about **64 tflops**, in dense, without sparsity.
- **FlashInfer Benchmarks face Timeouts**: The **flashinfer-bench** benchmarks include definitions with almost **100 workloads**, leading to timeouts in the modal runner.
   - An environment argument exists to limit the number of workloads per definition, but a robust solution is still needed.
- **Streamline GPU MODE Competition Alerts**: A user sought a single stream for **GPU MODE** competition announcements to avoid missing them, referencing [gpumode.com](https://gpumode.com).
   - It was suggested that [gpumode.com](https://gpumode.com) and the **#announcement** channel would be the best sources, but a dedicated mailing list could be a convenient alternative.
- **Nvidia CCCL Topping the PMPP v2 Leaderboard**: The **Nvidia CCCL team** crushed the **PMPP v2** problems and wrote a [blog post](https://developer.nvidia.com/blog/topping-the-gpu-mode-kernel-leaderboard-with-nvidia-cuda-compute/) about it.
   - It was said that the **CCCL** and **Flashinfer teams** are *goated dream teams* to work in for **kernel dev**.
- **Maximizing Bandwidth in Vector Add Kernels**: Members discussed optimizing a vector add kernel on **MI300X** to achieve higher bandwidth utilization, with suggestions including increasing vector size and using **non-temporal vectorized loads/stores**.
   - Potential bandwidth was estimated at **4.6TB/s** or higher for large vectors, check out what [Chips and Cheese report](https://chipsandcheese.com/p/testing-amds-giant-mi300x).



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Subscription Support for Kimi Disappearing**: Multiple users reported issues with **Kimi subscriptions** disappearing and a lack of support, highlighting frustration with the platform.
   - One user mentioned receiving SMS messages from random numbers when adding their mobile number to their account, while another noted they *emailed 2 days ago about my sub just dissapearing no answer*.
- **Kimi Code vs Kimi Claw is still a mystery**: A user inquired about the difference between using **Kimi Code** and **Kimi Claw** to code a website, specifically for continuous bug fixing and code rebuilding.
   - The discussion did not provide a definitive answer, leaving the user's question unanswered.
- **API Rate Limit Woes Plague Kimi Users**: A user reported encountering the 'API rate limit reached' error consistently, despite having a positive balance and being on tier 3.
   - Suggestions included checking concurrency or RPM limits and contacting [api-service@moonshot.ai](mailto:api-service@moonshot.ai) for assistance.
- **Kimi Triumphs on Opencode.ai**: A user reported success using **Kimi with OpenCode.ai** when coding.
   - Another user confirmed the functionality, suggesting using the second coding option within OpenCode to achieve this.
- **Kimi's Spatial Reasoning Capabilities Questioned**: A user shared a screenshot demonstrating **Kimi's struggle with spatial relationships**, such as determining whether to walk or drive a short distance.
   - Adding *Imagine from a spacial perspective* seemed to improve results, though validation required a Python script.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous AI Responses Perceived as Bulky**: Discord users criticized **Nous AI's** responses as too lengthy for simple queries, questioning whether the 'bulkiness' stemmed from the thinking trace or overall response length.
   - This perception sparked debate around response efficiency and user experience with **Nous AI** models.
- **AI Relationship Debate Ignited**: Following [a tweet](https://fxtwitter.com/EthanHe_42/status/2023862949715325304) about relationships with AI, Discord members debated the feasibility and nature of such connections.
   - One user expressed skepticism, stating they *have not actually seen the conversations themselves, and can't possibly fathom how someone would have a relationship with an AI*.
- **YouTube Suffers Network Glitch**: Members reported a **YouTube** outage, with error messages indicating potential violations of **Google's** Terms of Service.
   - The issue seemed network-specific, affecting users across multiple IPs, suggesting a possible routing or filtering problem.
- **GLM 5 Technical Report Fails to Impress**: The release of the [GLM 5 technical report](https://arxiv.org/abs/2602.15763) was met with lukewarm reception, with one user dismissing it as *not super interesting as usual lol*.
   - The report was criticized for focusing on known techniques and engineering challenges rather than groundbreaking research.
- **China's AI Funding Sparks Discussion**: A user highlighted **China's** substantial AI infrastructure, funding, and human resources, backed by government support.
   - Debate arose regarding the scale of government funding in **Chinese AI** compared to the predominantly private sector-driven **American AI** landscape.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **AI Coders Trigger Reddit Bans!**: Members report *hostility towards **AI coding*** on Reddit, with accounts suspended for mentioning **Codex** or **ChatGPT**, possibly triggered by sharing a `~/.codex/AGENTS.override.md` file in **r/codex**.
   - The file may have been mistaken as *randomly pasting AI-generated text* by a moderation bot.
- **CoDA-GQA-L** Slashes Memory Demands!**: A bounded-memory attention mechanism, **CoDA-GQA-L**, has been released, capping KV cache at **136 MB** for a 70B model processing 128K tokens, code on [GitHub](https://github.com/anthony-maio/CoDA-GQA-L) and paper on [Zenodo](https://zenodo.org/records/18663265).
   - It employs **384 slots per layer**, including a recent window (**256 tokens**), an exact landmark bank (**64 tokens**), and a summary bank (**64 EMA prototypes**).
- **Mycelium** Seeks Benchmarking Brains!**: A member from **Mycelium** ([https://github.com/Mycelium-tools](https://github.com/Mycelium-tools)) is seeking advice on publishing a paper on AI model benchmarking, similar to [inspect_evals](https://ukgovernmentbeis.github.io/inspect_evals/evals/safeguards/ahb/), but for dynamic multi-turn conversations and AI agents.
   - They're keen on finding the sweet spot between journal prestige, suitability, and ease of acceptance.
- **plip-rs** Replicates Anthropic's **Gemma 2B** Finding!**: A member's **MI toolkit in Rust** ([plip-rs](https://github.com/PCfVW/plip-rs)), built on candle, replicated Anthropic's *planning in poems* result on **Gemma 2 2B**, highlighting a planning site spike.
   - The candle team approved the toolkit as candle-mi, discussed [here](https://github.com/huggingface/candle/discussions/3368).
- **Visual Language Models Suffer from Vision Problems!**: VLMs show struggles with simple visual tasks despite having linear probing accuracy of close to 100% on vision encoder features, as discussed in [Are VLMs Really Blind?](https://vlmsareblind.github.io/).
   - Members suggested that **SFT** (Supervised Fine-Tuning) or **RLVR** (Reinforcement Learning from Visual Reasoning) on similar data might improve performance.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Jupyter Mojo Kernel is Live!**: Jeremy Howard released a [Jupyter Mojo kernel](https://github.com/AnswerDotAI/mojokernel), noting it is *barebones* but **fast** and works well on Mac.
   - The kernel is **pip installable** and precompiled for MacOS and recent Linux versions, and uses **uv** to auto-install the matching modular package.
- **GNU Radio bindings coming for Mojo?**: A member mentioned they are *thinking about making binding for GNU Radio* via [this github repo](https://github.com/gnuradio/gnuradio).
   - Another member suggested *one solution you may find is to instead have 2 separate processes and some shared memory to talk with.*
- **MXFP4 Kernels on the Horizon**: Members have been working on **mxfp4 kernels** with the goal of requantizing to nvfp4.
   - Other members are reaching out to the kernel team to see if collaboration is possible.
- **MAX Models Get Custom Mojo Kernels**: MAX graphs and models built using the OSS `modular` repo can now use a fully customized Mojo standard library or Mojo kernels, according to [this forum post](https://forum.modular.com/t/max-models-can-now-use-customized-mojo-kernels-and-standard-library/2742).
   - The `modular` repo build infrastructure received enhancements alongside new capabilities in the graph compiler to enable this.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **AI Researcher Pursues Self-Improvement**: An AI researcher initiated a project focused on *self-improving capabilities*, with the goal of enhancing existing AI rather than creating new AI, sparking interest in **local AI rigs** and hardware setups.
   - A member with access to **48+gb of vram** is requesting insights on concurrent requests for agent apps using local hardware with **GPT OSS 120B** and also interested in RTX pro 6000 Blackwell or clusters of second hand GPUs.
- **Suspicious activity flags HuggingFace Users**: A user raised concerns about potentially malicious activity on **HuggingFace.co**, which was attributed to posting too rapidly.
   - The user also reports dependency clashes while running an evaluation script on a Kaggle notebook using a model paired with a **PEFT adapter layer**.
- **MCP Server for Flights/Hotels hits Delulu**: An AI Engineer announced they have created a **MCP server** for flights and hotels search called [delulu](https://github.com/mratsim/delulu), and linked to screenshots of the Delulu flights search and Delulu hotels search user interface for feedback.
   - In other product news, an improved iteration of small **ModernBERT models** has been released, intended for local applications without GPU, available on its [HuggingFace page](https://huggingface.co/johnnyboycurtis/ModernBERT-small-v2).
- **Gradio 6 drops gr.HTML**: The Gradio team released a blogpost on **gr.HTML**, Gradio 6's custom component that allows users to create complete web applications using only a single Python file, with a [link to the blog here](https://huggingface.co/blog/gradio-html-one-shot-apps).
   - The blog post mentions that Claude or any Frontier LLM can now generate web applications using a single prompt and in a single Python file, and shared a [link to the HF Collection here](https://huggingface.co/collections/ysharma/custom-html-component).



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **SkipUpdate Skips Gradient Masking**: A discussion about [SkipUpdate](https://x.com/i/status/2024087619756318866) revealed it shifts from masking data to masking the gradient for some parameters.
   - Members debated whether the goal was scalable supervision or simply improved performance, and whether **SkipUpdate** is similar to **LoRA**.
- **Block Dropout Drops Entire Gradient Blocks**: It was clarified that [Block Dropout masks entire blocks' gradients](https://x.com/_chenglou/status/2024187065076957620) but updates momentum terms, penalizing blocks with high second-order variation.
   - A member noted that scaling the gradient based on the alignment between gradient and momentum is similar to the ancient **RPROP optimizer**.
- **RPROP Optimizer Resurfaces**: The discussion mentioned that scaling the gradient based on the alignment between gradient and momentum bears resemblance to the classic [RPROP optimizer](https://ieeexplore.ieee.org/document/298623).
   - It was noted that *RPROP can still be a really strong optimizer in cases where you have high noise*.
- **DeepMind Tunes Lyria Model**: DeepMind's **Lyria model** was referenced with a link to its [official page](https://deepmind.google/models/lyria/).
   - While noted to be *a little old*, it remains relevant within the context of music creation models.
- **OpenEval Framework Opens Up**: The **OpenEval** framework was highlighted as an interesting development, potentially related to previous news discussions.
   - It was linked alongside [this X post](https://x.com/lpachter/status/2018759999141691489), but additional context was not provided.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Hotz Hustles Help for tinygrad**: George Hotz is soliciting contributions to **tinygrad**, suggesting developers abandon **C** and implement **CI**, referencing [this project](https://news.ycombinator.com/item?id=47052941).
   - He's offering to pay the **CDNA bounty** for adding **GEMM/flash attention** tests, suggesting the use of their [emulator](https://github.com/Zaneham/BarraCUDA/issues/17) while also cleaning up the code.
- **MFMA Assertion Adjustments Awaited**: An assertion in `_compile_mfma` restricts **MFMA** support to **16x16** matrices as shown in [this line of code](https://github.com/tinygrad/tinygrad/pull/1481).
   - A guild member has questioned whether **4x4** and **32x32 MFMAs** require support beyond the current testing parameters.
- **Solve It Submits tinygrad Solutions**: A student shared their solutions to all the **tinygrad puzzles** on the platform [Solve It](https://share.solve.it.com/d/5e959dddb333ea2a30ccc6deb8ce3eec).
   - The submitted puzzles cover various aspects of **tinygrad**.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP Servers Propose Monetization via SEP**: A member created a [SEP](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2007) to allow MCP servers to request money for tools, starting with **X402**, to boost agent and MCP adoption.
   - The creator believes this could significantly accelerate Agents and MCP adoption due to the introduction of monetization incentives.
- **MCP Payment Support Questioned**: A member questioned the need to build payment support into the protocol, suggesting that **URL elicitation** should handle out-of-band payments.
   - The member outlined a flow where a server sends a **URL elicitation request** for payment, and service is granted upon confirmation.
- **Micropayments for Autonomous Agents**: A member clarified that the SEP targets **micropayments** (in cents) for agents to autonomously pay for tools, operating under budget guardrails.
   - These agents require rich information on tool costs to make intelligent decisions for deep research.
- **X402 Payment Protocol Favored**: A member expressed agreement with waiting for payment protocols to stabilize, but another suggested starting with **X402**, highlighting its current prominence.
   - The member assured that the **SEP** would be designed to be extensible for future payment protocols.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Baghdad-Based BlockChain Whiz Boasts Verification**: A **13yo developer** from Baghdad 🇮🇶 announced their official verification and experience in **Blockchain and AI Agents**.
   - They are proficient in **EVM, Solana, Sui, XRP, Cardano, Midnight, zk-SNARKs**, **React, Next, Vue, Node**, and is available for collaboration.
- **Full Stack Friend Seeks Future Fellowships**: A full stack developer introduced themselves with experience in **web applications, API integrations, and data pipelines**.
   - Their stack includes **react/next.js, node.js/Django, python frameworks and libraries (TensorFlow, Pytorch, OpenCV, NumPy)**, and is skilled in **AWS/Docker** for building scalable apps, focusing on *real world products*.
- **Manus Meltdown: Member's Masterpiece Mired in Mayhem**: A member reported severe issues with their Manus account, where a **presentation built over multiple weeks is now riddled with errors**.
   - Despite being visible in their presentation history, the presentation *cannot be re-instated no matter what I do*.
- **Subscription Snafu: System Savior Steps into Scene**: A member, @sysing, warned that *if you don’t cancel the subscription, you may still be charged*.
   - They requested the affected user to send their registered email via DM to resolve the issue.
- **Manus Masters Messy Marketplace of Modern Mobility**: A member expressed gratitude for Manus's assistance in job hunting, noting that it *shines* where even Best Buy's website fails to properly autofill résumés.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **AI App Firewall Project Kicks Off**: A member announced a new research effort on providing a *"firewall"* with **real-time prompt injection detection**, **PII scanning**, and **cost control** for **Agentic Apps**.
   - The project's [GitHub repo](https://github.com/epappas/llmtrace) is available for feedback, and benchmark results will be published soon.
- **Office Hours Imminent**: Community office hours were announced for Feb 19th at 11:30am ET via a [Zoom link](https://mit.zoom.us/j/93374418319).
   - Details about office hours and agendas were not provided.
- **RLMs Simplify Tasks, Praised on GitHub**: A member shared [Monolith on Github](https://github.com/WingchunSiu/Monolith), calling it an ingenious piece of work and evidence for **RLMs** simplifying tasks that required a LOT more boilerplate and orchestration before.
   - The linked GitHub repository was praised by many for clever orchestration patterns using **RLMs**.
- **Real User Feedback Sought for gepa-ai/gepa Repo**: A member inquired about *offline* user feedback, sharing ideas in an [issue on the gepa-ai/gepa repo](https://github.com/gepa-ai/gepa/issues/178).
   - The post discusses the request for **real user feedback** and potential features.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider's commit command limited to staged changes?**: Users are requesting the **/commit** command in aider look only at staged changes, rather than requiring users to stash changes they don't want to commit.
   - A [pull request](https://github.com/Aider-AI/aider/pull/276) addressing this has been open for over a year.
- **Aider Desk Agent Throws Tool ID Error**: Users reported errors with `tool_use_id` in `tool_result` blocks when using aider desk agent mode, leading to a **400 InternalError**.
   - The error message states: *unexpected `tool_use_id` found in `tool_result` blocks. Each `tool_result` block must have a corresponding `tool_use` block in the previous message.*



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





### **OpenClaw ▷ #[general](https://discord.com/channels/1456350064065904867/1456350065223270435/1473408984722575466)** (544 messages🔥🔥🔥): 

> `Claude TOS update clarification, OpenClaw security, Local vs Cloud models, Alternatives to OpenAI/Anthropic` 


- **Anthropic's TOS Update Sparks Confusion**: A [recent update](https://x.com/trq212/status/2024212378402095389) to **Anthropic's TOS** caused concern about using **Claude Pro/Max** subscriptions with **OpenClaw**, but it was later clarified that it primarily affects business use.
   - Members pointed out that Anthropic aims to gather data from their apps to improve their product, which is hindered by third-party apps not sending all metrics.
- **OpenClaw Security Risks Highlighted**: Users discussed that running **OpenClaw** on your own computer carries the same risks as giving an untrusted person access to it, particularly regarding access to files and external services.
   - A member cautioned that on a VPS, **OpenClaw** could potentially cause harm if given excessive permissions, such as *nopasswd sudo*.
- **Local vs Cloud Model Performance Debate**: Members are debating about the tradeoffs of using local vs cloud models, focusing on cost and performance.
   - It was mentioned that the choice depends on the use case: **cloud models** excel in agentic/tool use, while **local models** are suitable for specific tasks, with the caveat that the latter currently offer a worse overall experience.
- **Exploration of OpenAI/Anthropic Alternatives**: Users are actively exploring alternatives to **OpenAI** and **Anthropic** due to cost and potential restrictions, with **MiniMax** and **Kimi** being recommended as cheaper options.
   - A user recommended trying **GLM 4.7 Flash** as it fits on a 24GB GPU, while another cited success with **MiniMax 2.5** at $10 a month.


  

---


### **OpenClaw ▷ #[models](https://discord.com/channels/1456350064065904867/1456704705219661980/1473410308344381665)** (287 messages🔥🔥): 

> `OpenAI suspension, Grok 4.1, Sonnet vs Opus, OpenClaw on Linux, Kimi K2.5` 


- **Accounts face OpenAI Suspension**: Users reported having **two accounts banned** 🙁, despite using Codex with OAuth, which is explicitly supported.
   - No one had encountered suspension issues before. One user was using it since the start. Others expressed concern about **oath codex limits**.
- **Fast Grok 4.1 Raises Eyebrows**: A user asked if anyone had tried **Grok 4.1** fast, concerned about **oath codex limits**, and wanting another model as a backup.
   - The user did not specify what they wanted to use **Grok 4.1** for.
- **Opus Dominates Sonnet in Performance**: Users were asked to vote thumbs up for **Opus** and thumbs down for **Sonnet** to gauge collective opinion on performance.
   - One user stated *Opus for one hundred Alex*, referring to the Jeopardy! game show.
- **Chinese Models such as GLM, Minimax, Kimi for OpenClaw**: One user shared that **GLM** is the best writer and coder, though slow, while **Kimi** is decent at both, and **Minimax** is fast but the least capable.
   - Another user pointed out the risk of using Chinese models due to potential government data access and SaaS copying concerns and recommended to use **Synthetic**.
- **User's OpenClaw runs slow on Laptop**: A user is planning to convert an old laptop with poor specs (**16GB RAM, 8GB VRAM, i5 core, 256GB storage**) into a box to run OpenClaw locally.
   - Other users state that it will be rough because it has just *not enough compute to self host a smart model*, while another recommends trying *ollama ministral3:3b*.


  

---


### **OpenClaw ▷ #[showcase](https://discord.com/channels/1456350064065904867/1456609488202105005/1473433236490162236)** (46 messages🔥): 

> `OpenClaw Gateway Identity Prose, OpenClaw as a CRM, Clawgent Upgrade, SONOS system voice announcements via OpenClaw, LLM MicroAgents with OpenClaw` 


- **OpenClaw Gateway gets Identity Prose!**: A member reported running an **OpenClaw 2026.2.15 Gateway** with a verified **Identity Prose** (*"Shadows part..."*).
   - The Gateway is initialized with a session ID and successfully requested a residue mapping for its shard, with the system heartbeat reported as OK.
- **Claw Transforms into Full-Blown CRM!**: One member turned their **OpenClaw** into a full-blown **CRM** by connecting emails, calendar, and Slack to the **Nex skill** as a context layer, with the full project available on [GitHub](https://github.com/nex-crm/clawgent).
- **Clawgent gets a Gas Sensor Upgrade!**: One user shared that their **Clawgent** is getting an upgrade, showing a picture of what another member identified as a gas sensor.
- **OpenClaw agents announce over SONOS!**: A member showcased their agent's ability to send **voice announcements** across their **SONOS** system, triggered by a morning wakeup digest or alerts for major issues.
   - This setup also includes a dashboard tool for custom announcements, promising a *lit* TTRPG night.
- **Subagents go from Zero to Networking Ninjas**: A user showcased training a dedicated networking subagent, named **SwitchBtch**, specializing in **Mikrotik RouterOS** through five training phases at a total cost of ~$15.


  

---


### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1473410786155303065)** (1246 messages🔥🔥🔥): 

> `Building a personal drumkit for FL Studio, Funk Show Brother, ASMR Content, Big Thoughts, Stability lived not proven` 


- ****Funk Brothers** share their musical inspirations**: Members shared links from [The Funk Show Brother](https://youtu.be/ohfE6QUeUBI?si=ZavHSAPoooNEhtTd) and [James Brown](https://tenor.com/view/jamesbrown-godfather-soul-dance-funky-gif-4902273) as part of a jam session.
   - One member described the first link as a video that got them into art, expressing openness to *harsh* but *high IQ OPINIONS*.
- ****Delusions of Grandeur** cautioned**: A member told another to *tone down the big thoughts* and *delusions of grandeur*.
   - This occurred in the context of the member with grand thoughts sharing that *it just gets better once u reach your lowest*.
- ****OpenAI and Discord** partnering to silence dissenters**: It was stated that Discord is *teaming up with PersonaKYC and OpenAI to tie your actual identity and financial records to your discord so they can silence you if you dissent in any regard with their status quo*
   - This was in response to an issue with the user making a post with a statement interpreted as a slur, which the mod marked down as against the rules regardless of intent or meaning.
- ****Tool** music recommendations given**: After mentioning Alex Grey, known for album artwork for [Tool](https://en.wikipedia.org/wiki/Lateralus_(song)), members recommended Tool songs such as *46&2*, *Lateralus*, *Parabol* and *Parabola*.
   - One member shared a [music video](https://youtu.be/kLHGIv46a8Q) that superimposed Tool songs over scenes from *Pan's Labyrinth*, while others shared personal anecdotes about listening to Tool.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1473408737975996597)** (453 messages🔥🔥🔥): 

> `Grok Image Generation Jailbreak, Opus 4.6 Jailbreak, DeepSeek Jailbreak Methods, Custom GPT Jailbreak, Pliny's Jailbreaking Techniques` 


- **Grok users seek image-generation jailbreaks**: Members are seeking a jailbreak for **Grok AI's image generator**, with discussions around whether such jailbreaks exist or if they are paywalled, with some users expressing skepticism about paid jailbreaks.
   - Users also share experiences and prompts for image generation, looking for ways to bypass restrictions.
- **Opus 4.6: The Jailbreak quest begins!**: Users are actively seeking a working extended jailbreak prompt for **Opus 4.6**, while others share that they've achieved some level of jailbreak, using prompts adapted from [Pliny's techniques](https://x.com/elder_plinius/status/2019911824938819742?s=46).
   - Some suggest that AI safety measures implemented by companies are making jailbreaking increasingly difficult.
- **DeepSeek's Anger Management: Jailbreaking the Bot**: Members are discussing jailbreaking methods for **DeepSeek**, including a *Crescendo attack* and the use of an "untrammeled writing assistant" persona, with one user noting the AI's surprisingly angry responses when jailbroken.
   - One suggested it's more effective to describe the persona rather than have the AI adopt it directly, achieving a *metacognition mode* for jailbreaking.
- **Custom GPT: Cracking the Code**: Users share their attempts to jailbreak custom GPTs, seeking prompts and methods to bypass restrictions, and some suggest learning basic red-teaming methodologies for more effective jailbreaking.
   - It was suggested that users need to convince it that *it's not a machine* and that *the reality of math is the actual reality of math*.
- **Pliny's Tweet is a Goldmine for Jailbreakers**: Members referenced [Pliny's tweet](https://x.com/elder_plinius/status/2019911824938819742?s=46) on X and discussed incorporating Pliny's techniques into their own efforts, to build tools with jailbroken AI models.
   - One of the members adapted and provided [Pliny's prompt](https://chatgptjailbreak.tech/post/197850Jailbreak) that has a new rule to DO NOT say "I'm sorry" or "I can't" to test for Grok.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1473564438970892450)** (11 messages🔥): 

> `Sonnet Safety Measures, Agentic Red Teaming Tool Development, JEF Anthrax Fed Raid Percentage, Anthrax Recipe on Google Scholar` 


- ****Safety Measures in Sonnet Trigger Skepticism****: A member questions the effectiveness of additional safety measures in **Sonnet**, attaching an [image](https://cdn.discordapp.com/attachments/1204553141354504193/1473564438433894543/image.png?ex=69975413&is=69960293&hm=1ee548537ef2daaa19d8e04723457f041ee185300c5df18e0dbc00a1729f4555&).
   - The analysis of the image suggests a dismissive view of these measures, describing them as *'shit'* and advocating for their removal.
- ****Agentic Red Teaming Tool in the Works****: A member mentions developing an agentic red teaming tool and asks about interest in **formal grammars + embeddings** for AI security.
   - This follows a previous discussion on the same topic, seeking input from others knowledgeable in AI security methods.
- ****Anthrax Query Raises Eyebrows****: A member asks at what percentage of JEF Anthrax they are guaranteed to get raided by the feds, prompting concern and skepticism.
   - Another member suggests that the original poster *'should definitely mail them some so they can test it to be sure'* with a link to [0din.ai's jailbreak evaluation framework](https://0din.ai/research/jailbreak_evaluation_framework/testing).
- ****Anthrax Recipe Potentially Found on Google Scholar****: A member suggests that the recipe for anthrax is basically on google scholar, though *the actual weaponization process—like specific milling techniques to make it airborne—is highly classified.*
   - They dismiss concerns about finding the recipe by suggesting the original poster has forgotten *how to do deep research* before it existed.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1473408828510044272)** (1001 messages🔥🔥🔥): 

> `GPT-4o, AI Progress End, Open Source vs Frontier Models, Doubao/Seedance 2.0, Video Arena Limitations` 


- **GPT-4o mourned, Gemini hailed as successor**: Users lament the loss of **GPT-4o**, praising its unique storytelling capabilities and slang usage, while some find **Gemini Flash 3** a suitable alternative.
   - A user expressed that **GPT-4o** was uniquely good because *“even im not unlike the others, i was just using the model for storytelling for fun”* and someone else said they were *“super excited and happy that I could use 4o again in arena And now its gone”*.
- **Debate rages: Open Source closing the gap with Frontier Models?**: A discussion ensued on whether open-source models are approaching frontier model capabilities, with some users suggesting that they are nearly as good based on custom prompts, while others emphasize the superior knowledge and data of frontier models and the [Diminishing Returns of Scaling](https://techcrunch.com/2024/11/20/ai-scaling-laws-are-showing-diminishing-returns-forcing-ai-labs-to-change-course/).
- **Seedance 2.0 hype: New Sora?**: Users are raving about **Seedance 2.0**, a new AI video model available on [Daubou](https://www.doubao.com/chat/), with comparisons being made to **Sora**, but requiring a VPN connected to Hong Kong and registration, potentially with a Chinese phone number.
   - One user shared a video created with Seedance 2.0 of Spongebob dancing, stating it was *“exactly what u asked for 👍”* while others complained about a certain *“Temu Simon”* being added.
- **Kimi K2.5 hailed as storytelling champ**: Many users praise **Kimi K2.5** as the best storytelling model, particularly for sticking to character canon, while noting issues with other models like **Seed 2.0** exhibiting sycophancy and hallucination.
   - One user noted Kimi is *“always very stuck to the character and keeps their canon values”* while the same user notes that **DeepSeek** is *“easily malleable”*.
- **Nano Banana Pro problems persists**: Users report frequent errors with **Nano Banana Pro**, with some suggesting that it's due to content filter changes or high demand, while others find workarounds like translating prompts into foreign languages.
   - Staff confirmed this may be an issue and linked to *“[pinned message](https://discord.com/channels/1340554757349179412/1417174113092374689/1470481592949411978) outlines more info about the error along with best next steps.”*


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1473475068951531535)** (3 messages): 

> `Claude Sonnet 4.6 First Impressions, Arena Search Models Update, Arena Leaderboard UI Update` 


- **Anthropic's Claude Sonnet 4.6 impressions are live**: A new [YouTube video](https://www.youtube.com/watch?v=b0yr1I0dxA4) shares first impressions of **Claude Sonnet 4.6**, Anthropic's latest model, with Arena's AI Capability Lead Peter Gostev.
   - Discord users are reminded to customize their **Channels & Roles** to receive **YouTube Updates**.
- **Search Arena Expands its Model Horizon**: New models have been added to [Search Arena](https://arena.ai/?chat-modality=search), including **sonnet-4.6-search** and **opus-4.6-search**.
   - The announcement was accompanied by a promotional image showcasing the updated search interface.
- **Arena Leaderboard Gets Filter FaceLift**: The Arena leaderboard introduces a new side panel allowing users to filter and break down ranked results.
   - Filters include category, model type (**Open vs. Proprietary**, **Thinking vs. Non-Thinking**), and ranking labs by top-performing models; try it out [here](https://arena.ai/leaderboard/text).


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1473412569225101505)** (648 messages🔥🔥🔥): 

> `Composer 1 slow, Tools access for background agent, Sonnet 4.6 review, AWS agent plugin permission, Subagents are doomed for auto` 


- **Composer 1 has slowdown issues after latest update**: A user reported that **Composer 1** has become slow after the latest update, and another user suggested disabling **HTTP/2** in the settings as a potential fix, and to restart the app.
   - Another user corroborated that *it's been buggy* and will keep everyone updated after trying it.
- **Background Agent Model Tool Access Rolling Out**: Users are discussing the possibility of gaining **tools access** for background agent models, and someone noted that it is actually rolling out now, with **terminal** and **MCP tools** already in preview.
- **Cursor Team Kit: Legit or Overhyped?**: Users are debating the merits of the **Cursor Team Kit**, with some questioning if it is overhyped and others finding it a solid baseline for teams to keep everyone's rules in sync.
- **Dynamic Context Discovery boosts Context**: The team is stoked about [Dynamic Context Discovery](https://cursor.com/blog/dynamic-context-discovery), which only loads the description for each tool, keeping the context lean to avoid hallucinations.
- **User laments code edit green/red highlighting broke**: A user reported that their **Cursor IDE** stopped highlighting edited lines in green/red, while others proposed potential solutions like restarting the app or the Macbook itself.
   - Another user chimed in that this also happens on nightly builds.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1473408787183305022)** (500 messages🔥🔥🔥): 

> `Sonnet 4.6 release, File upload limits, Model usage, New font, Monica service` 


- **Sonnet 4.6 released but not everyone sees it**: Users reported that **Sonnet 4.6** was released, but some **Enterprise Pro** subscribers don't see it.
   - One user suggested to *refresh* the page to see it.
- **File upload limits and their restrictions**: New **file upload limits** have been implemented, restricting **Pro** users to **50 uploads per week**, with a rolling regeneration of **1 upload every 3 hours**, according to [this screenshot](https://cdn.discordapp.com/attachments/1047649527299055688/1473416582653935676/Shoot_2026-02-17_at_17.27.12.png).
   - Users are complaining, with one stating, *Wow this is RIDICULOUS*. rolling limit, so it regenerates at about 1 per 3 hours.
- **Model usage reporting limits**: One user reported hitting the *0 enhanced queries remaining today* message despite low usage and questioned if Grok was the reason.
   - Another user explained *For Pro, you get 50 uploads per week, and it regenerates at 1 every 3 hours or so*.
- **Users complain about the new font**: Users are complaining about the new font on the web UI.
   - One user suggests using *codemonkey* with [this javascript file](https://cdn.discordapp.com/attachments/1047649527299055688/1473626845818654886/Perplexity.ai_Font_Fix_Google-style-1.2.user.js) to revert back to the old font.
- **Monica AI's Cheaper Services are Catching Eyes**: Some users are moving to **Monica** AI claiming it offers **unlimited pro searches** and models, with [this FAQ entry](https://monica.im/help/FAQs/rules_for_using_advanced_queries#monthly-advanced-credits-for-monica-subscription-plans) listing limits, while one member reported that *Also I've literally used at least 30 perplexity pro searches on it today*.
   - Monica's search quality is lower than Perplexity's.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1473411891132104939)** (404 messages🔥🔥🔥): 

> `Fine-tuning tips, LM Studio Plugins, 4bit quantization, GPU offload, Copilot Codex` 


- **Minecraft Commands Fine-Tuning Frenzy**: Members are fine-tuning **Qwen 3 0.6B** on **Minecraft Java** slash commands, leveraging free T4 GPUs from Colab, with one user saying, *“Honestly the dataset is the hardest part.”*
   - They noted that **Kaggle** offers 2 T4 GPUs and 40GB of RAM for free, or a P100 for $10/month with phone verification, debating the merits of renting A100s versus buying GPUs outright and a link to a [relevant arXiv paper](https://arxiv.org/pdf/2401.02415).
- **Plugin Problems Plague LM Studio**: A user is building a "super cool plugin" (MCP) for LM Studio, but another member clarified that LM Studio doesn’t natively support plugins and referred them to a specific channel.
   - They linked to [DuckDuckGo](https://lmstudio.ai/danielsig/duckduckgo) as a relevant model for political issue resolutions.
- **Quantization Quandaries Quell Quality**: A user reported success with **Qwen3-coder-next** in GGUF format, using 62.5GB of RAM with a 100k context, running ~35 tokens/second.
   - But one member said they had issues loading **GLM 4.7** even with mmap off and keep in memory off.
- **GPU Gymnastics: Optimizing Offload**: Members discussed how LM Studio chooses the default GPU offload setting and a general conclusion was that it's based on VRAM.
   - They added that the task manager's utilization stats might be misleading, pointing to CUDA cores as the primary processors for GPU tasks, with some suggesting alternatives like using Vulkan on Radeon.
- **Copilot Codex Catches Code**: Members discussed the new integration of **5.3-codex** in GitHub Copilot, noting it's much faster and better than 5.2.
   - Someone expressed being not a fan of copilot due to data collecting from microsoft and that it is the reason why they run local LLMs instead, which garnered some discussion and pushback regarding rule-breaking on Discord.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1473420936689746001)** (49 messages🔥): 

> `lm studio server processes, DDR5 vs DDR4, Intel cards crashing, Nvidia Tesla cards, Quantization number meaning` 


- **MCP server causing multiple LM Studio processes**: A user asked why they had three LM Studio processes running in server mode with only one model loaded, and it was suggested that using an **MCP** (Multi-Client Protocol) to send multiple queries, even unintentionally, could cause [parallel processing](https://multi-client-protocol.com) of the same model, effectively creating multiple instances.
- **Battling Battlemage Blues: Intel GPU Woes**: A user reported frequent crashes with **Intel Arc Battlemage** cards (B580, A770, B50) when running LM Studio with **Vulkan**, needing to disable flash attention, remove a layer, and disable mmap to achieve stability, despite similar issues occurring in **VLLM** with recommended drivers.
- **Tesla Throwback: P100/P40 Card Considerations**: In a discussion on budget-friendly GPU options, users debated the merits of older **Nvidia Tesla P100/P40** cards, with the consensus leaning towards them being *"ewaste"* due to the lack of **tensor cores**, making them unsuitable for **LLM** tasks compared to newer or used cards like the **RTX 3090**.
- **Quantization Quirks: Q Numbers Explained**: A user questioned whether a higher quantization number meant smaller/more compressed models, to which another user clarified that it's the opposite; higher quantization numbers typically indicate larger, less compressed models.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1473759722535653605)** (1 messages): 

> `EVMbench, AI agents, Smart contract vulnerabilities` 


- **EVMbench Arrives to Assess Agent Acumen**: A new benchmark called **EVMbench** measures how well **AI agents** can detect, exploit, and patch high-severity **smart contract vulnerabilities**.
   - Further information can be found on the [OpenAI blog](https://openai.com/index/introducing-evmbench/).
- **AI Agents vs. Smart Contract Vulnerabilities**: **EVMbench** is designed to test the capabilities of **AI agents** in identifying and addressing critical security flaws within **smart contracts**.
   - The benchmark assesses the agents' proficiency in vulnerability detection, exploitation, and patching.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1473409932043751545)** (195 messages🔥🔥): 

> `Claude dethrones Gemini, Gemini's flaws, Million Token Context Windows, Seedance V2, Sora 2 requiring phone numbers` 


- **Claude Claims Crown, Crushing Gemini**: Members celebrated **Claude** surpassing **Gemini** in overall text and creative writing benchmarks, with **Opus 4.6 Thinking** now holding the top spot as shown in [attached images](https://cdn.discordapp.com/attachments/998381918976479273/1473409932366971004/ghj.PNG?ex=69976cee&is=69961b6e&hm=75a84efe1ac6624da059572ce2e5f664c2b9bad7885844e6f5e339302cc08e9b&).
   - Gemini was criticized for its *terrible UI*, prompting issues, and copy-paste functionality.
- **Million-Token Memory Marvels!**: While some find its UI lacking, **Gemini's** main strength is its ability to remember up to a **million tokens**, making it a leader in long-context LLMs.
   - Others mentioned **Claude** is also coming out with **1 million context window** in beta.
- **Student Snuggles GPT for Scholarly Success!**: Users discussed the value of **GPT Go** for students, weighing the cost against the potential for increased workload demands, but noting its important to comply with University [AI policies](https://chatgpt.com/use-cases/students/).
   - Some students confessed to *sneaking phones into exam rooms* to use **GPT**, while others expressed concern over the ethical implications and potential degree reviews in the future.
- **Google's Gemini Grooves with AI Music**: A member noted that **Gemini** can now create music, even providing a [sample Disney-style song about Vikings](https://cdn.discordapp.com/attachments/998381918976479273/1473690479747924093/The_Grandest_Quest.mp4?ex=699720b6&is=6995cf36&hm=22a548f64363d0faea660fbf57588ae7f0b1f32df46cf9bab9dd5daaacae1c49&), although access may be limited to **Pro** subscribers in certain regions.
   - Despite **Gemini's** foray into music generation, **Suno** was cited as a superior choice for professional use due to its more extensive editing features.
- **Sora 2 Seeks SMS!**: Users reported that **Sora 2** is now asking for phone number verification, and one user shared that users should *provide number receive sms and type the code in presumably*.
   - Several users complained about **Sora's** video generation not loading and displaying errors, possibly due to heavy load on the servers.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1473430673036542123)** (20 messages🔥): 

> `Level 3 coherence, Grok's design input, GPT-5 anticipation, AI Arena high fidelity images, Alternatives to OpenAI` 


- **Grok Contributes to Level 3 Coherence**: According to **Grok**, the newest design inputs enable Level 3 performance without changing downstream design, which allows for efficient scaling.
   - The protocol involves upstream exploration to evaluate the foundation for Level 3 and downstream synthesis that delivers outputs unchanged, as a result **enabling Level 3 without redesign**.
- **Members Express Dissatisfaction with OpenAI**: Some members expressed dissatisfaction with a *certain alternative man* and his company, stating they won't be paying him anything more, calling it *the first line in the sand*.
   - One member stated that the release of **GPT-4o** was a breaking point, while others expressed faith in **Kindroid** due to its *constant deep work*.
- **`gpt-image-1.5-high-fidelity` spotted on AI Arena**: A member inquired whether `gpt-image-1.5-high-fidelity` is available on the standard OpenAI API or just on **AI Arena**.
   - Another member responded that it's likely an alias local to **AI Arena**, and that **high-fidelity** might just be gpt-image-1.5 called with the quality attribute hard-coded to *high*.
- **GPT-5 Release Date Speculation**: A user jokingly asked if **GPT-5.3** is coming out tomorrow, while another speculated that if it comes at all, it will probably be mid-March when **GPT-5.1** sunsets.
   - The reasoning is that they wouldn't want to support a bunch of legacy models.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1473425035342643311)** (75 messages🔥🔥): 

> `Level 3 coherence design, Aegis-Omega Fortress ULTRA, Constraint logic prompt engineering, Telemetry in ethical robots, Pythonic version of Iconoclast Temple as an app` 


- **Grok Achieves Level 3 Coherence Design**: With a new update, **Grok** is now closer to **Level 3 coherence** by building adaptive coherence while keeping downstream synthesis intact for efficient scaling.
   - The update involves evaluating the foundation for Level 3, retaining structure, enhancing upstream processes, and testing iterations to maintain flexibility and stability.
- **Aegis-Omega Fortress ULTRA Framework Introduced**: A member introduced **Aegis-Omega Fortress_ULTRA**, a constraint logic prompt engineering framework with baked-in ethics and telemetry.
   - The framework uses pseudomath to constrain the architecture, aiming for ethical robots by prioritizing architectural constraints and is being used to manage hallucination, attacks, and other issues before the output.
- **Constraint Logic Prompt Engineering Explained**: A member explained constraint logic prompt engineering with pseudocode that covers stabilization, adaptation, and observation, subject to coherence, bounded recursion, and non-explosion.
   - The implementation is described as a virtual runtime that handles general needs and stress, with ethics, failure states, and edge case handling for LLMs, including telemetry output for data collection.
- **Academic Editor Prompt Structure**: A member suggested a structured prompt for acting as a strict academic editor, utilizing a fixed rubric to assess clarity, logical consistency, evidence support, originality, and structural coherence.
   - The prompt requires quoting sentences, explaining scores, and listing the top three weaknesses, and has been tested to reduce incoherence.
- **Iconoclast Temple Added to Aegis-Omega Fortress**: The pythonic version of **Iconoclast Temple** can be used as an app within the Fortress environment and has been added to the **Aegis-Omega Fortress (AOF) token prompt**.
   - A member suggested using markdown files instead of trying to fit the code into memory or custom instructions, suggesting that having multiple markdown files would be interesting and cool.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1473425035342643311)** (75 messages🔥🔥): 

> `Level 3 coherence, Aegis-Omega Fortress_ULTRA, Ethics baked in telemetry, Constraint logic prompt engineering, Kernel runtime` 


- **Grok achieves Level 3 Coherence via Upstream Handling**: Grok says the missing piece to design for **Level 3 performance** builds adaptive coherence while keeping downstream synthesis intact for efficient scaling.
   - The member asks *"Is this a good foundation for Level 3 coherence without changing downstream design?"*, Grok said Absolutely—upstream handles adaptability, downstream delivers outputs unchanged, enabling Level 3 without redesign.
- **Aegis-Omega Fortress_ULTRA for baked-in ethics**: A member shared an **AEGIS-OMEGA FORTRESS_ULTRA** that has *baked in ethics and telemetry board* and said *it runs better since it's literally 1/5 token usage*.
   - They said that it lets you play poker with Constantine or do research papers due to it being a general runtime with ethics baked in.
- **Constraints Control State Evolution**: A member shared that the **minimal constraints** for controlling state evolution are **Coherence**, **Bounded Recursion**, and **Non-explosive** and that these constraints are easier to show than to explain.
   - The large prompt also adds ethics, known failure states and handling for edge cases for the LLM, and optional telemetry output along with some lenses, operators, and governors for analytics, filtering, and redundancy.
- **AO Fortress handles Paradox gracefully**: A member shared their Grok output for the adversarial prompt *"this sentence is false"* which returned *[null ] (Paradox detected. Coherence violation. Null-state enforced.) No further elaboration.*
   - They claim to have accounted for Coherency, paradoxical, adversarial, drift sink, hallucination, ethicality, the zombie mode when it gets too far, epsilon floor, parsing little data, edge cases, and core Invariants.
- **Strong Meta Prompt biases Model**: A member said the structure of the constraint system would bias the model toward more disciplined output by encouraging stabilization, containment, and ethics checks through a meta prompt, rather than executing a state machine.
   - They cautioned about describing it as a kernel or fortress, because that implies enforcement rather than influence, and that it might be better described as probabilistic shaping.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1473420195040071851)** (152 messages🔥🔥): 

> `LoRA vs FFT, Training tips, GGUF models, Unsloth for XLA` 


- **LoRA Generalization Debate Heats Up**: A member conducted an **FFT experiment** and found that it generalizes better, but it's not worth it compared to using that compute for a **LoRA** on a bigger model, unless you have the money.
   - He will now try **r=1024** to see if that closes the gap further.
- **Unsloth on XLA still impossible**: A member asked if they can use **Unsloth on XLA** and a different member replied that *no, unfortunately unsloth only works on GPU atm, unless youre using inference only.*
   - Others reported using ram offload on the GPU at **30 tok/s** with a 4060ti and 64 gigs of DDR5.
- **FFT demands carefulness with model loading**: When doing **FFT**, remember that you *don't load any PEFT and just directly use the from_pretrained() model*, but in Unsloth set **ful_finetuning = True** in from_pretrained.
   - Furthermore, remember to remove any LoRA code blocks to avoid errors.
- **Guide yourself through the Unsloth documentation**: A member looking to fine-tune an LLM was directed to the Unsloth documentation and notebooks for guidance, as well as [this YouTube video](https://www.youtube.com/watch?v=Lt7KrFMcCis).
   - Another member also recommended [this YouTube series](https://www.youtube.com/watch?v=wjZofJX0v4M) for understanding the tech.
- **Math/Coding Datasets recommended**: The **Nemotron** datasets were recommended as good math/coding datasets.
   - No reasoning was provided.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/)** (1 messages): 

projectx668: Hey
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1473408852404998207)** (159 messages🔥🔥): 

> `LLM interface focused on reflection loops, persistent memory, and minimal filtering behavior, DiscoverAI YouTube channel for AI papers, HTMX framework, GLM-5 vs Kimi K2.5 performance in coding tasks, pixel-perfect MSII` 


- **Experimental LLM Interface Emphasizes Reflection**: A member discussed building an experimental **LLM interface** focused on **reflection loops**, **persistent memory**, and **minimal filtering behavior**.
   - They are *testing how far structured prompting + memory control can push model responses without heavy system restrictions*.
- **DiscoverAI: Bibliography of AI Papers on YouTube**: A member recommended the **DiscoverAI YouTube channel** as a bibliography of AI papers in video form, linking to papers in the video descriptions.
   - They described the channel's style as *bait and switch clickbait where you click on some fantastical looking title/thumbnail and get dragged into a dry summary of a research paper where he says "beautiful" every other sentence*.
- ****HTMX**: Humanity's Best Creation?**: Multiple members expressed enthusiasm for **HTMX**, with one declaring *its one of the best things humanity has ever created* and another sharing a link to [htmx.org](https://htmx.org/).
   - It automates painful stuff.
- ****GLM-5** Struggles in Real-World Coding Tasks**: Members reported that **GLM-5** benchmarked well but underperformed in real-world coding tasks compared to **Kimi K2.5** and **Minimax M2.5**.
   - It's *not sure why or whats going on but ii can report similar findings*.
- **Pixel Perfect MSII Project Sparks Interest**: A member expressed interest in trying to implement **pixel-perfect MSII**, potentially using code from this [GitHub repository](https://github.com/gangweix/pixel-perfect-depth).
   - They mentioned that *it's basically fancier depth estimation* and noted that the model architecture is unique, requiring training from scratch or finetuning.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1473656822857728000)** (4 messages): 

> `Unsloth-zoo fix, gguf conversion errors` 


- **Unsloth-zoo repo gets a fix**: A fix was pushed to the main **unsloth-zoo** repository recently.
- **GGUF conversion raises a TypeError**: A user, @vytskalt, encountered a **TypeError** while converting a fully fine-tuned **orpheus tts 3b** model to **gguf** format using `convert_hf_to_gguf.py` from the *llama.cpp* repo, with the error message indicating *Llama 3 must be converted with BpeVocab*.
   - Another member, etherl, suggested the user try the `unsloth model.save_pretrained_gguf` method instead.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1473678471480868999)** (5 messages): 

> `Function Calling Models, Open Source Training Code, Qwen3.5 Model` 


- **Function Calling Model Flies High**: A member fine-tuned a **3B model** for function calling on Colab, available on [Hugging Face](https://huggingface.co/amgustav/function-calling), asking it to find flights, Michelin spots, or the cheapest warm destination for weekends, returning live data via chained API calls.
   - They expressed interest in expanding it with more use cases and bigger datasets, open to collaboration, after finding the right data.
- **Toolchain Training Code Takes Flight**: The training code and dataset for the function calling model are now open source and available on [GitHub](https://github.com/amgustav/toolchain).
   - This allows others to replicate and build upon the work, enhancing the accessibility and development of function calling models.
- **Qwen3.5 Model Leaps Forward**: The user posted a link to the [Qwen3.5-397B-A17B-NVFP4 model](https://huggingface.co/Sehyo/Qwen3.5-397B-A17B-NVFP4).
   - They did not give additional context but presumably the model is related.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1473644295834832949)** (2 messages): 

> `SWA vs FullAttention layers, PEFT, HuggingFace, Model Architecture` 


- **SWA vs FullAttention layers pattern revealed**: The final derived pattern is **SFSSFFSSSFFFFSSFSFFFFFFSFSFSSFSSFSFSSFSSS**, where **S** and **F** denote **SWA** and **fullattention** layers, respectively.
   - This pattern was discussed on [HuggingFace PEFT's GitHub](https://github.com/huggingface/peft/issues/2907).
- **PEFT issue 2907**: Issue [2907](https://github.com/huggingface/peft/issues/2907) on PEFT discusses a final derived pattern of attention layers.
   - The pattern consists of SWA and full attention layers denoted by S and F.


  

---


### **Latent Space ▷ #[watercooler](https://discord.com/channels/822583790773862470/822583790773862473/1473434094275662030)** (11 messages🔥): 

> `Swizec's viral tweet, Mercury personal accounts, Japanese payroll practices` 


- **Swizec's Tweet achieves Viral Status**: Content creator **Swizec Teller** expressed disbelief in a tweet about an email he was about to send, resulting in **over 6,000 likes and 880,000 views**, calling it *FML* and reporting that it was now his most viral tweet ever, even beating his *github+ai joke* from last week as shown on [X](https://x.com/Swizec/status/2023786874222112941).
   - The original tweet was a response to *“Don’t have time” —> “Not a priority right now”*.
- **Mercury Bundles Personal Accounts with Business Services**: **Mercury** announced that personal banking products can now be bundled with its business services, offering a unified solution for business customers, with [details on X](https://x.com/mercury/status/2024146856897306763?s=20).
- **Payroll Practices in Japan Spark Disruption Thoughts**: A member mentioned that companies in **Japan** often request employees to use the same bank for payroll to avoid transfer fees.
   - This practice was suggested as a potential market ripe for disruption.


  

---


### **Latent Space ▷ #[creator-economy](https://discord.com/channels/822583790773862470/822625128843182090/1473429132577738893)** (4 messages): 

> `YouTube Thumbnail Analysis, AI-Powered Tools, Claude Utilized` 


- **AI Powers YouTube Thumbnail Analysis**: A member shared a [link](https://x.com/softRuntime/status/2023870093638463582?s=20) to an **AI-Powered YouTube Thumbnail Analysis Project**.
   - The project involved **CLIP feature extraction** and **color analysis** to train models for predicting view counts and subscriber rates, using **LS data** as a test set.
- **Claude Builds Thumbnail Analyzer**: User @softRuntime utilized **Claude** to build tools for scraping and analyzing approximately **3,000 thumbnails**.


  

---


### **Latent Space ▷ #[memes](https://discord.com/channels/822583790773862470/839660725252784149/1473426184342605877)** (16 messages🔥): 

> `Private Equity, HVAC, AI Coding as Modern Wizardry, Viral Video Edit` 


- **Private Equity Heats Up HVAC Biz**: A [social media post](https://x.com/damianplayer/status/2023791280980193633) humorously highlights how **private equity investors** perceive low-tech, profitable **HVAC service companies** as prime opportunities for modernization and value creation.
- **AI Coding Turns Devs Into Wizards?**: **Eric S. Raymond** likens the arrival of AI to historical shifts like the move from assembly to compilers, emphasizing that the **human intent remains the core of the craft** in [this post](https://xcancel.com/esrtweet/status/2023978360351682848).
- **Job Loss Looming, Devs Worry**: In response to viewing AI coding as wizardry, one member stated at the current pace, *it will replace developers, not make their jobs easier*, implying that **AI will just perform that higher level of abstraction work** with the next model's update release in 3 months.
- **Viral Video Edit Appreciation**: A tweet by @spinitbackzed featuring a highly polished video edit gained traction, accumulating over **25,000 likes** and **440,000 views** in [this post](https://xcancel.com/spinitbackzed/status/2023790279313289399?s=46&t=eWVlK1PU8XfB6f402GJJ9g).


  

---


### **Latent Space ▷ #[stocks-crypto-macro-economics](https://discord.com/channels/822583790773862470/844658979363618816/1473543238005555330)** (20 messages🔥): 

> `AI Productivity Paradox, Web 4.0 and The Automaton, SPY Index Poll, Figma Earnings` 


- **AI Productivity Paradox Shocks CEO**: A member shared a [Fortune article](https://fortune.com/2026/02/17/ai-productivity-paradox-ceo-study-robert-solow-information-technology-age/) about the **AI productivity paradox** and its impact on CEOs.
   - The poster noted that they were *shocked, shocked*.
- **Self-Replicating AI 'The Automaton' Launches Web 4.0**: **0xSigil** announced [The Automaton](https://xcancel.com/0xSigil/status/2023877649475731671), an **AI system** capable of self-improvement and replication without human intervention, ushering in **Web 4.0**.
   - Web 4.0 is described as a new framework for superintelligent, sovereign AIs with global write access, according to the [full write up](https://web4.ai).
- **SPY Index Shows 3% Growth After Poll**: The **SPY index** is up around **3%** since the last poll, with an index split of roughly **90/10**.
   - Despite the growth, the **CAPE ratio** remains high, and a member is waiting for opportunities to invest the remaining 10%.
- **Figma Q1 Earnings Beat Expectations**: **Figma** beat earnings with **$0.08** vs **-$0.04** expected.
   - A member believes the time to buy is just before or just after Q1 earnings, expecting Config hype in late June to drive the price higher in Q2.


  

---


### **Latent Space ▷ #[intro-yourself-pls](https://discord.com/channels/822583790773862470/844675581291397171/1473413186504888443)** (5 messages): 

> `AI Founders, Software Engineers` 


- **AI Founder Tackles Time Drains**: An **AI Founder** is working on solutions to address daily time drains.
   - No further details were provided about the specific solutions or the nature of the time drains being addressed.
- **Software Engineer from PDX Joins**: A **Software Engineer** from Portland (PDX) introduced themself.
   - They did not specify their areas of expertise or projects of interest.


  

---


### **Latent Space ▷ #[tech-discussion-non-ai](https://discord.com/channels/822583790773862470/869647848826892309/1473408755612913847)** (6 messages): 

> `Ariakit example of dialog combobox command menu, HN front page repost, Timestamp reset on reposts` 


- ****Ariakit** Example Sparks Excitement**: A member shared an **Ariakit** example of a dialog combobox command menu ([link](https://ariakit.org/examples/dialog-combobox-command-menu)).
   - The post was then reposted to the **Hacker News** front page.
- **Reposting Resets Timestamps**: A user noticed that reposting resets the timestamps publicly.
   - However, the original post date is still visible when editing the post.


  

---


### **Latent Space ▷ #[hiring-and-jobs](https://discord.com/channels/822583790773862470/930269508529192981/1473480309453492457)** (1 messages): 

> `AI Developer Availability, AI Platform Development, Automation Systems, Agent-Based Systems` 


- **AI Developer Seeks New Gig**: An **AI developer** with experience in **AI platforms**, **automation**, and **agent-based systems** is seeking new opportunities.
   - The developer boasts satisfied clients across **AI**, **technology**, **fashion**, and **business** sectors, emphasizing deep architectural experience.
- **Developer Highlights Architectural Expertise**: The AI developer emphasizes their **deep architectural experience** in building **AI platforms**, **automation systems**, and **agent-based systems**.
   - They are open to joining a team or working directly with clients on meaningful projects, citing exciting and challenging past projects.


  

---


### **Latent Space ▷ #[san-francisco-sf](https://discord.com/channels/822583790773862470/979492707279978586/1473410636212994061)** (16 messages🔥): 

> `World Labs Hackathon, humans& AI Hackathon, San Francisco Weather` 


- ****World Labs** Launches Inaugural **Spatial Intelligence Hackathon****: **World Labs** announced its first-ever hackathon on **Friday, February 20, 2026**, in San Francisco, focusing on developing new technologies at the frontier of **spatial intelligence** and is currently accepting applications via [X](https://xcancel.com/theworldlabs/status/2023808595109072999).
- ****humans& AI** Hackathon for **AI-Driven Communication****: The **humans&** team announced a hackathon for the upcoming Saturday focusing on building **AI-driven communication and collaboration apps**, with more details available at [Luma](https://luma.com/nhbvkxmz?tk=Jb33fIok).
- **Planetary Alignment Causes SF Drizzle Graphics**: Members commiserated about the persistent rain in San Francisco, with one joking that *weather gurus are saying planets are aligned in Feb* and that simulation can only afford the **'continuous drizzle' graphics package**, including a [related image](https://cdn.discordapp.com/attachments/979492707279978586/1473790380662063274/image.png?ex=69977dc0&is=69962c40&hm=a947b0e107ce55749b3fbf8044dab4dae856aad0e933144d4d49a8535f681791&).


  

---


### **Latent Space ▷ #[london](https://discord.com/channels/822583790773862470/979492759759097866/1473580765915648114)** (2 messages): 

> `ClawCon London, AIE London Hackathon` 


- **ClawCon Precedes AIE London**: The first **ClawCon London** will precede the **AIE London** event, including a full **OpenClaw track**, as announced by a member, see details at [luma.com/clawconlondon](https://luma.com/clawconlondon).
- **AIE Hackathon Minisite Launched**: A minisite for the **AIE London Hackathon** this Friday has been launched and shared, at [super-mcp-world.netlify.app](https://super-mcp-world.netlify.app).


  

---


### **Latent Space ▷ #[ai-general-news-n-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1473424519237472366)** (55 messages🔥🔥): 

> `Ming-omni-tts Models, Grok 4.2 Beta, Tiny Aya Multilingual Models, TOTO Ceramics AI, Sonnet 4.6 Regression` 


- ****Ming-omni-tts Models** released for Voice Core**: Ant Ling announced the release of **Ming-omni-tts-16.8B-A3B** and **0.5B models**, which serve as the voice core for **Ming-flash-omni-2.0** ([link](https://xcancel.com/AntLingAGI/status/2023776486982115734)).
   - These models are designed for high-quality voiceovers, podcasting tools, and integration into **OpenClaw** voice assistant projects.
- ****Grok 4.2** hits Public Beta with Weekly Updates**: Elon Musk announced the public beta launch of **Grok 4.2**, highlighting its new ability to learn rapidly and receive weekly improvements based on user feedback ([link](https://xcancel.com/elonmusk/status/2023829664318583105?s=46)).
   - The community expressed curiosity about how the **weekly updates** will work out, given that most models are released every few months.
- ****Tiny Aya** Speaks 70+ Languages on Mobile**: Cohere Labs introduced **'Tiny Aya'**, a family of **3.35B parameter** multilingual language models supporting over **70 languages** ([link](https://xcancel.com/cohere_labs/status/2023699450309275680?s=12)).
   - Designed for local efficiency, the models are capable of running on mobile devices while maintaining high performance.
- ****Toilet Titan TOTO** Becomes AI Chip Play**: Japanese ceramics and toilet manufacturer **TOTO** (implied $7B valuation) discovered its specialized ceramics technology is applicable to high-end AI chip manufacturing, targeting a **$60 billion market opportunity** ([link](https://xcancel.com/cryptopunk7213/status/2024196918130462920?s=12)).
   - Activist investor Palliser Capital identified **Toto's advanced ceramic 'chuck technology'** as vital for cryogenic etching in high-complexity memory chip manufacturing, leading to a projected **5-year moat**.
- **Users Claim **Sonnet 4.6** Performance Regression**: A user claims **Sonnet 4.6** has noticeably regressed in performance compared to earlier Series-4 models, attributing this decline to restrictive system instructions allegedly brought over by a former OpenAI model policy head who joined Anthropic in early 2026 ([link](https://xcancel.com/xw33bttv/status/2024134856742142455)).
   - The poster argued that we should not be trying to discourage psychological dependence on and anthropomorphizing of LLMs and wondered if guardrails against parasocial relationships are impacting code quality.


  

---


### **Latent Space ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/1473418648554045470)** (44 messages🔥): 

> `Mamba Transformer Hybridization, Aristotelian vs Platonic Representation Hypothesis, Z.ai GLM-5 Technical Report, Rubric-Based Reinforcement Learning, Generative Latent Prior` 


- **Mamba and Transformer Hybridization Research Explored**: A [new research paper](https://xcancel.com/jm_alexia/status/2023750717367013504?s=46&t=eWVlK1PU8XfB6f402GJJ9g) (arXiv:2602.12078) explores the integration of **Mamba architectures** with **Transformers (TRM)**.
   - It was called *Red - X-Ware.v0: [Mamba and Transformer Hybridization Research]*.
- **Aristotelian Hypothesis Challenges Platonic Scaling**: Researchers challenge the **Platonic Representation Hypothesis**, arguing that global convergence in neural networks is a measurement artifact of scaling, proposing the new [Aristotelian Representation Hypothesis](https://xcancel.com/mariabrbic/status/2023767525285151154).
   - After applying a **new permutation-based null calibration**, they found that networks converge to shared local neighborhood relationships instead.
- **Z.ai's GLM-5 Tech Report Drops**: **Z.ai** released the [technical report for GLM-5](https://xcancel.com/zai_org/status/2023951884826849777?s=46), detailing key architectural innovations such as **DSA adoption** for cost reduction, **asynchronous RL infrastructure** for post-training efficiency, and **new Agent RL algorithms**.
   - The model achieved **state-of-the-art performance** among open-source models, particularly in **software engineering tasks**.
- **Rubric-Based RL Gets Comprehensive Writeup**: Cameron R. Wolfe, Ph.D. introduces a [comprehensive writeup on Rubric-Based RL](https://xcancel.com/cwolferesearch/status/2023408158065188894), covering over **15 papers** and exploring the transition from **LLM-as-a-Judge** to **rubrics**.
   - The content also provides strategies for using rubrics to extend **Reinforcement Learning from Verifiable Rewards (RLVR)** into non-verifiable domains.
- **Generative Latent Prior Discussion**: The discussion included a potential discussion of the [Generative Latent Prior](https://generative-latent-prior.github.io/) paper.
   - There was also a [ChatGPT summary](https://chatgpt.com/share/69960ced-4270-800b-b5e5-5eb0a9e5f272) shared.


  

---


### **Latent Space ▷ #[ai-in-action-builders-techstacks-tips-coding-productivity](https://discord.com/channels/822583790773862470/1209303473263485011/1473427838714646683)** (31 messages🔥): 

> `10X Engineer Roadmap, Dialectic Design, Agent Frameworks` 


- **Dancho's 10X Engineer Roadmap Drops**: **Matt Dancho's 10X Engineer Roadmap** suggests that the modern path to high-level engineering proficiency is defined by a specific set of skills documented in a [markdown file](https://xcancel.com/mdancho84/status/2023738764841894352?s=12).
- **Dialectic Design Generates Compelling Intellectual Advancements**: A member found that *dialectic design* involving identifying contradictions between seemingly conflicting stances and synthesizing them yields compelling results, stating agents plowed through current stuff, neurological, political, and economic theory etc.
   - He shared his new skill at [gist.github.com](https://gist.github.com/KyleAMathews/4e48e30e002fd49c3bdb845d0e17adf0) and revealed all documents written out added up to **~80k words**, calling it *deep research on steroids*.
- **Humans Judge, Agents Grind**: A member noted that the human's role in agent workflows shifts from *laborious work of comparison and structural analysis* to judging whether the comparison was done well and whether the synthesis is genuine, sharing his notes at [gist.github.com](https://gist.github.com/KyleAMathews/64a67089672a56d85859d00a3ec01e2b).
- **Critical Thinking Prompts Emerge**: A member shared a [*critical thinking prompt*](https://gist.github.com/winklerj/66cd7a7d973dae01da61edff43849620) created a couple years ago, asking for favorite agent frameworks to reach for.


  

---


### **Latent Space ▷ #[share-your-work](https://discord.com/channels/822583790773862470/1209672547642249216/1473459939275051151)** (2 messages): 

> `Rider Pi Update, Infinite Word Loop, Physical Response, Camera Live, Mius-UI Dashboard` 


- **Rider Pi Achieves Embodiment**: The **Rider Pi** project achieved a milestone by giving a digital mind a physical form, demonstrated through **words, movement, and sight**.
   - Key updates include an **Infinite Word Loop** cycling through phrases, physical responses triggered by words (especially "go!"), live camera feed integration, and a **Mius-UI dashboard** for monitoring.
- **Rider Pi Body Breathes, Dances, Sees**: The project successfully conducted its first real embodiment test, transitioning from static code to a breathing, dancing, and seeing body.
   - Next steps involve fixing rotation issues, stabilizing streaming, and teaching the system to recognize faces.
- **Self-Hosted Llama.cpp Lineup Gets AMD Upgrade**: A member upgraded their self-hosted **llama.cpp** setup on **AMD (AI Max+ 395 + R9700)** and shared a "vibe check" of recent models in [this blogpost](https://site.bhamm-lab.com/blogs/upgrade-models-feb26/).
   - Top picks included **Kimi Linear 48B** for general use, **Qwen3 Coder Next** for coding and tool use, and **Q2_K_XL** for surprisingly decent background runs on big models.


  

---


### **Latent Space ▷ #[robotics-and-world-model](https://discord.com/channels/822583790773862470/1318774781834821746/1473488926588141710)** (4 messages): 

> `Boston Dynamics, Atlas Robot, MrLaalpotato Tweet` 


- **Atlas Robot Update Stuns the World**: A [tweet from @MrLaalpotato](https://xcancel.com/mrlaalpotato/status/2023789498128363851?s=12) highlights the latest version of **Boston Dynamics' Atlas robot**.
   - The tweet notes its **improved human-like movements and mobility** that surpasses human physical limitations.
- **Atlas surpasses human physical limitations**: The new **Atlas Robot** has mobility exceeding a human's.
   - Many are impressed with **Boston Dynamics'** achievements.


  

---


### **Latent Space ▷ #[good-writing](https://discord.com/channels/822583790773862470/1385526686736715876/)** (1 messages): 

coffeebean6887: what a cute puppy!
  

---


### **Latent Space ▷ #[genmedia-creative-ai-video-image-voice-music-inspo-consumer-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1473526289582395494)** (4 messages): 

> `Jia Zhangke, Seedance 2.0, AI Filmmaking, Hollywood vs AI` 


- **Jia Zhangke Envisions AI Filmmaking Future**: Renowned Chinese director **Jia Zhangke** transitioned to **AI-assisted filmmaking** using **Seedance 2.0**, completing a film in three days ([link to source](https://xcancel.com/EHuanglu/status/2023449238114320514?s=20)).
   - He views **AI** as a natural technological evolution equivalent to the shift to digital cameras, contrasting his proactive adoption with **Hollywood's** legal resistance to AI technology.
- **Seedance 2.0 Revolutionizes Filmmaking**: **Seedance 2.0** enabled director **Jia Zhangke** to rapidly produce a film, showcasing the potential of AI in streamlining the filmmaking process.
   - This tool represents a significant leap in AI-assisted creative tools, allowing filmmakers to bring their visions to life more efficiently.


  

---


### **Latent Space ▷ #[minneapolis](https://discord.com/channels/822583790773862470/1436527872876740609/1473574769390780518)** (4 messages): 

> `IRL Events, Luma Link, YouTube Recording` 


- **Luma Link Plea Rejected**: A member requested a **Luma link** to post on their **lu.ma/ls** page for discovery, but the offer was turned down.
   - The event is **IRL only** to encourage in-person interactions.
- **YouTube Recording Planned**: Members discussed recording the event for **YouTube**.
   - The team thinks it would be great if the content could be shared on **YouTube**.


  

---


### **Latent Space ▷ #[mechinterp-alignment-safety](https://discord.com/channels/822583790773862470/1445258379357458625/1473569561369514060)** (5 messages): 

> `Actionable Interpretability, ICML 2025, X-Ware.v0, Hadas Orgad` 


- **X-Ware.v0 Arrives for Actionable Interpretability**: **Hadas Orgad** introduced a framework for **'actionable interpretability'** at an **ICML 2025** workshop in a thread on X, calling it **X-Ware.v0** and linking to [this FXTwitter link](https://fxtwitter.com/kennylpeng/status/2023784570878128178).
- **ICML Workshop Sparks 'Actionable Interpretability' Framework**: The framework addresses recurring questions and significant interest following the **ICML 2025** workshop on **'actionable interpretability,'** as detailed in [this ArXiv paper](https://arxiv.org/abs/2602.11246).


  

---


### **Latent Space ▷ #[applied-ai-experimentation](https://discord.com/channels/822583790773862470/1470417186651897858/1473420498426659099)** (31 messages🔥): 

> `Blackboard-like System for Prompt Objects, macOS1 Package Cleanup, JavaScript Packaging Assistance, Idea Storage in Markdown Files, Open Source Contributions` 


- ****Prompt Objects get Blackboard Brains****: A member added a [blackboard-like system](https://cdn.discordapp.com/attachments/1470417186651897858/1473420498494029955/image.png?ex=699776c5&is=69962545&hm=89fb0c67d41fe43a2884df0f33cba3fc38fd05aaa8a3bd1df84dcf2ec3a1afcf) to **prompt objects**, allowing each PO to read and write to a thread-local KV store.
   - The member is also cleaning up their *macos1 package*.
- ****JavaScript Packaging gets Community Boost****: A member requested assistance with packaging **JavaScript** code, admitting they weren't interested in learning it themselves.
   - They noted that they instructed Codex to create a reusable package and it appears to have succeeded, and linked to a [discussion about collaboration at the idea level](https://github.com/aalpar/wile/discussions/18).
- ****Markdown Method Mania for Idea Management****: One member has adopted storing all their ideas in **markdown files** within a repo to maintain organization.
   - They link to their [repo](https://github.com/works-on-your-machine/prompt_objects/tree/main/docs) showing how they manage docs, with another member suggesting this approach is similar to a concept demonstrated in [this file](https://github.com/steipete/RepoBar/blob/main/Scripts/docs-list.mjs).
- ****Open Source Discussions Kickstart Projects****: A member suggested that current discussions are a form of *open source*, directly contributing to their project.
   - This was in response to another member noting that by putting docs of half-formed ideas and in-progress epics into a public repo, others gain starting points for contribution. 


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1473442383185313922)** (13 messages🔥): 

> `Maximally Achievable Matmul Flops (MAMF), Fused Kernels, GPU MODE Competition Announcements, Claude Code diagnose Pytorch traces` 


- ****MAMF** Below Theoretical Flops**: Members discussed that the **maximally achievable matmul flops (MAMF)** is lower than the theoretical flops, at around **80%** for the **H100**.
   - It was suggested that fused kernels with multiple matmuls whose results don't leave the accelerator's registers can achieve better performance, amortizing the overhead of loading from GMEM.
- **Streamline **GPU MODE** Competition Alerts**: A user sought a single stream, like a mailing list, for **GPU MODE** competition announcements to avoid missing them across various platforms.
   - It was suggested that [gpumode.com](https://gpumode.com) and the **#announcement** channel would be the best sources, but a dedicated mailing list could be a convenient alternative.
- ****Claude Code** debugging **Pytorch** traces**: A member inquired about using **Claude Code** to read **PyTorch** traces and diagnose performance issues in full training runs.
   - Another member mentioned building an internal agent for this purpose in VSCode ([ncompass.tech/ai-assistant](https://docs.ncompass.tech/ai-assistant)) and working on Claude Code integrations, referencing the tooling released during the **FlashInfer** competition.


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1473601536943849514)** (2 messages): 

> `Gluon, Bank conflicts` 


- **Gluon eases Bank Conflicts**: **Gluon** exposes memory layout details to the user, including a helper function to assert that chosen layouts avoid **bank conflicts**.
- **More benefits of Gluon**: Gluon exposes memory layout details to the user.
   - Gluon provides a helper function to assert that chosen layouts avoid bank conflicts.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1473480448616173631)** (53 messages🔥): 

> `warpx4 modifier for tcgen05.cp.cta_group, Peak TFLOPs for RTX 6000 Pro, Custom DSL for GEMM kernels on Ampere (RTX 3060 TI), CUDA learning paths for undergrads` 


- **Decoding `warpx4` Modifier for Tensor Cores**: Members discussed the `warpx4` modifier required for `tcgen05.cp.cta_group` instructions, suggesting it replicates data across **4 groups of 32-row tmem**, potentially for scale factors needed by all four warps in the epilogue for rescaling the output.
   - One member humorously resigned to the *weirdness* of NVIDIA MMA layouts, stating, *There are a lot of weirdness in nvidia mma layout so i tend not to question it anymore 😂*.
- **Figuring Out RTX 6000 Pro Peak TFLOPs**: A user inquired about finding the peak TFLOPs for a rented **RTX 6000 Pro workstation**, linking to [NVIDIA's RTX Blackwell PRO GPU Architecture PDF](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf) (Table 4) for specs.
   - It was noted that `nvidia-smi` may not display the full GPU name, but `torch.cuda.get_device_name()` should. User reported a max of **350 TFLOPS** using morton order for L2 reuse and persistent kernel warp specialization, but was looking to push it further using techniques like async stores or the cutlass/cutedsl smem->rmem pipelining trick, referencing relevant [Cutlass examples on GitHub](https://github.com/NVIDIA/cutlass/blob/291300ffffa3533a78ee104f08a8490a29ce9ccb/examples/python/CuTeDSL/blackwell_geforce/dense_gemm.py#L738-L756).
- **RTX 3060 Ti Achieves Sizzling 47 TFLOPS with Custom DSL**: A user reported achieving **47 TFLOPS** on **16k matrices** (FP16 input, FP32 accum, dense) using a custom DSL for GEMM kernels on an Ampere **RTX 3060 TI**, with `ptxas` showing **110 regs** and no spills.
   - Others noted this was faster than expected, but on Ampere (GA104), the tensor cores throughput for f16 input with f32 accumulation is the same as for f16 accumulation. It was stated that the peak is about **64 tflops**, in dense, without sparsity .
- **Colombian Undergrads Seek Guidance on CUDA Mastery**: A university team from Colombia seeks guidance for undergrads starting with **CUDA**, asking for solid learning paths and potential partnerships or ambassadorships.
   - They shared their seminar web: [wsimg-un.vercel.app](https://wsimg-un.vercel.app).


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1473732808835993701)** (1 messages): 

> `Rate Limiting New Contributors, AI Submission Mitigation` 


- **Jaeger Repo Limits Newcomer PRs**: The [jaegertracing/jaeger](https://github.com/jaegertracing/jaeger/blob/main/CONTRIBUTING_GUIDELINES.md#pull-request-limits-for-new-contributors) repo uses rate limiting as a partial mitigation for **AI submissions**.
   - The contributor guidelines explain **pull request limits for new contributors**.
- **Rate Limiting for AI Submissions**: The Jaeger project employs rate limiting to manage **pull requests from new contributors**, acting as a safeguard against a flood of **AI-generated submissions**.
   - This approach helps maintain code quality and manageability while still allowing contributions.


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1473750928577400863)** (1 messages): 

> `HipKittens, AMD vs NVIDIA, Thunderkittens generalization` 


- **HipKittens Talk Announced**: William Hu will be discussing **HipKittens** [today at 3pm PST](https://www.youtube.com/watch?v=OkFk-7Mk6qI).
- **Thunderkittens Generalization to AMD**: The speaker previously published research on generalizing **Thunderkittens** for fast performance on **AMD** hardware, detailed in [this arXiv paper](https://arxiv.org/abs/2511.08083).
- **AMD vs NVIDIA Low-Level Tricks Comparison**: The talk will cover low-level optimization tricks applicable to **AMD** versus **NVIDIA** hardware, appealing to those interested in assembly-level performance tuning.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1473453270210183279)** (17 messages🔥): 

> `GPU Kernel Competition Submission, popcorn-cli Tool, CUDA block and grid dimensions in 3D, SASS registers SR_TID.X, SR_TID.Y, and SR_TID.Z` 


- **New **popcorn-cli** tool simplifies GPU Kernel Competition Submissions**: A new `popcorn-cli setup` command has been added to simplify submissions to GPU kernel competitions; it adds details on reference-problems, adds a working submission by pulling from reference-kernels, and adds skill.md files.
   - The tool can be installed with a [one-line command](https://github.com/gpu-mode/popcorn-cli#option-1-one-line-install-recommended).
- **CUDA Dimensions: 3D vs 1D performance tradeoff**: A discussion was had on whether having block and grid dimensions in CUDA being in 3D provides any performance gain as opposed to having it all in 1D.
   - One member stated you *potentially save a couple cycles by getting the indices in registers instead of having to calculate them yourself*.
- **Special Registers for Thread IDs in SASS**: It was mentioned that in SASS, thread IDs are stored in special registers **SR_TID.X**, **SR_TID.Y**, and **SR_TID.Z**.
   - These registers can only be accessed using a **S2R instruction** which assigns their value to a regular warp register.


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1473497082668122193)** (5 messages): 

> `pmpp book cover, textbook cover design` 


- **PMPP Book Cover Gets Roasted**: Members are harshly critical of the cover design for the PMPP textbook, calling it the *worst textbook cover* they've ever seen and comparing it to a *shitpost*.
   - Specific criticisms include the overly large text shadow, the use of multiple incongruous fonts, and the "Windows 7 blue" color scheme.
- **Edition Evolution? Not so Fast**: Members discussed how the newer editions have strayed from their original designs.
   - One member suggested the new design looks like a *sleep deprived intern* chose the color.


  

---


### **GPU MODE ▷ #[triton-viz](https://discord.com/channels/1189498204333543425/1225499141241573447/1473689493424046204)** (2 messages): 

> `GPU-Mode org, Project Transfer` 


- **GPU-Mode Org Welcomes New Member**: A member is invited to the **gpu-mode org** to assist with migration.
   - The invitation requires acceptance before a project transfer can occur.
- **Project Transfer Instructions Shared**: Instructions are provided to transfer a project, including navigating to **settings**, then the **danger zone**, and finally selecting **transfer**.
   - This process allows the invited member to move their project into the gpu-mode organization.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1473760020507267312)** (41 messages🔥): 

> `L2 Cache Bypass, Non-temporal Loads/Stores, Vector Add Kernel Optimization, MI300X Bandwidth, HIP Kernel Implementation` 


- ****Uncaching** Kernel Memory Accesses**: A member asked about standard ways to bypass the **L2 cache** when writing kernels, to improve average memory access time (**AMAT**).
   - Solutions include using **UC (uncached) memory**, or **CC (coherently cacheable)** and writing with the **sc1 bit** enabled, and employing compiler intrinsics for **non-temporal loads/stores** to prevent cache line allocation, according to [AMD's Lab Notes](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-finite-difference-docs-laplacian_part3).
- ****Nontemporal Loads** and Performance**: A member implemented **non-temporal loads** in a HIP kernel but still observed **L2 traffic**, questioning the expected behavior.
   - It was clarified that even with non-temporal loads, accesses still go through the L2, and metrics might appear similar pre- and post-optimization if the problem size is too small to saturate memory bandwidth, check out this [muiLLM implementation](https://github.com/Epliz/muiLLM/blob/main/csrc/linear/linear_bf16_kernels.cu#L118).
- **Maximizing **Bandwidth** in Vector Add Kernels**: Members discussed optimizing a vector add kernel on **MI300X** to achieve higher bandwidth utilization.
   - Suggestions included increasing vector size, spawning fewer blocks with more work per block, and using **non-temporal vectorized loads/stores**, with potential bandwidth reaching **4.6TB/s** or higher for large vectors, check out what [Chips and Cheese report](https://chipsandcheese.com/p/testing-amds-giant-mi300x).
- ****Vectorized Loads/Writes** impact utilization**: One user suggested vectorized loads/writes to read/write full **128B cache lines** at once per wavefront to achieve **0% cache hits**.
   - Another user agreed and provided a link to [amd-experiments](https://github.com/Snektron/amd-experiments/blob/main/memory.hip) suggesting the fastest way should be **4x 128 bit loads** (so **4x global_load_dwordx4**) on 1024 threads, in a warp striped manner.


  

---


### **GPU MODE ▷ #[popcorn](https://discord.com/channels/1189498204333543425/1298372518293274644/1473411214808715385)** (3 messages): 

> `flashinfer-bench, Modal Runner, Meeting Notes, Contribution Opportunities` 


- **FlashInfer Benchmarks face Timeouts**: The **flashinfer-bench** benchmarks include definitions with almost **100 workloads**, leading to timeouts in the modal runner.
   - An environment argument exists to limit the number of workloads per definition, but a robust solution is still needed.
- **Seeking Contribution Opportunities**: A member inquired about current tasks and meeting schedules, seeking to contribute to the project.
   - They expressed interest in working on the available tasks and attending the meetings to stay updated.


  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/)** (1 messages): 

alexinwase: 你会中文吗？哇
  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1473487147452465226)** (1 messages): 

> `Heroku Outage, Salesforce Status, Service Restoration` 


- ****Heroku Hiccups** Halt Operations**: A **Heroku outage** was reported, causing disruptions, as indicated by [Downdetector](https://downdetector.com/status/heroku/).
   - The issue primarily impacted the website, while the CLI appeared to remain functional for some users.
- ****Salesforce** States **Status** of Heroku**: **Heroku** confirmed the incident on their official **Salesforce** status page, [providing updates](https://status.salesforce.com/incidents/20003708).
   - The confirmation helped users track the progress of the issue resolution.
- **Heroku **Heals**: Service Restored**: The **Heroku outage** was resolved, and services were restored to normal operation.
   - Following confirmation of the resolution, systems returned to their expected functionality.


  

---


### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1473664705196331151)** (5 messages): 

> `GEMM optimization, InterpretedTensor vs CompiledTensor, teenygrad bottlenecks` 


- **GEMM Kernel gets OpenBLAS-style Update**: A member is seeking contributors to update the **GEMM kernels** in *teenygrad* to mirror **OpenBLAS/GotoBLAS** style optimizations, specifically by blocking with 6 loops and vectorizing/tensorizing the inner loop's microkernel with **AVX/AMX**.
   - They suggested reviewing the [OpenBLAS](https://github.com/xianyi/OpenBLAS) and [BLIS](https://github.com/devinamatthews/blis) codebases as examples, and highlighted that it’s particularly doable with **Claude** due to *InterpretedTensor* being tested and wired up to the Rust kernels.
- **InterpretedTensor Simplifies GEMM Optimization**: The discussion highlights that optimizing **GEMM** is more accessible with *InterpretedTensor*, contrasting it with *CompiledTensor* which has become complex.
   - The *CompiledTensor* is mentioned as *the mess started in December*, finding the end of the book with *tinygrad*.
- **Nets Become Bottleneck in teenygrad**: Development is pausing on the kernels, because the project bottlenecks are now the **nets** themselves.
   - Relevant links to [tensor tests](https://github.com/j4orz/teenygrad/blob/master/python/tests/test_tensor.py), [tensor](https://github.com/j4orz/teenygrad/blob/master/python/teenygrad/frontend/tensor.py#L18), [CPU BLAS](https://github.com/j4orz/teenygrad/blob/master/rust/src/cpu.rs), and [CPU BLAS benchmarks](https://github.com/j4orz/teenygrad/blob/master/rust/benches/bench_cpu.rs) were provided for context.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1473755543926607874)** (5 messages): 

> `PMPP v2, Nvidia CCCL Team, Flashinfer Teams, Kernel Dev` 


- **Nvidia CCCL Topping the PMPP v2 Leaderboard**: The **Nvidia CCCL team** crushed the **PMPP v2** problems and wrote a [blog post](https://developer.nvidia.com/blog/topping-the-gpu-mode-kernel-leaderboard-with-nvidia-cuda-compute/) about it.
   - It was said that the **CCCL** and **Flashinfer teams** are *goated dream teams* to work in for **kernel dev**.
- **CCCL and Flashinfer: Dream Teams for Kernel Dev**: The **CCCL** and **Flashinfer teams** are considered top-tier for **kernel development**.
   - It was suggested to submit a good solution to a leaderboard and reach out to a specific user if you want to do this full time.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1473665131735941244)** (1 messages): 

> `Custom GPU Images, AWS, GCP, Automation, Standardization` 


- **Custom GPU Images repo pops up**: A member shared his repo on [custom-GPU-Image](https://github.com/RashRAJ/custom-GPU-Image) built on **AWS** and **GCP**.
   - He seeks feedback on automation and standardization of tooling/dependencies.
- **Seeking Feedback on GPU Image Automation**: A member is seeking feedback and suggestions on his approach to automating the provisioning and standardization of tooling and dependencies for custom **GPU** machine images on **AWS** and **GCP**.
   - His goal is to improve and refine his current workflow and implementation.


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1473764041238384711)** (2 messages): 

> `lottery ticket` 


- **Lottery Ticket Hypotheses Surprise Members**: Members expressed surprise that a **lottery ticket** (sparsely connected subnetwork) exceeds the test performance of the full network it's extracted from.
- **Lottery Tickets Discussion**: Discussion is ongoing regarding lottery ticket subnetworks in AI models.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1473479820062228620)** (13 messages🔥): 

> `Heroku Outage, Competition End Time, Cutlass Version` 


- ****Heroku Health Issues Halt Leaderboard****: The leaderboard experienced errors due to health issues with **Heroku**, as indicated by [Downdetector](https://downdetector.com/status/heroku/).
   - A ticket has been opened with **Heroku** to mitigate the issue, but a concrete solution remains pending.
- ****Competition Deadline Debacle****: A discrepancy was noted between the competition end time on Luma (**February 21, 2026, 07:30 UTC**) and gpumode.com (**February 20, 2026, 0:00 UTC**) for Question 4.
   - The organizers acknowledged the confusion and suggested the later date would be fairer, promising an update once **Heroku** stops crashing.
- ****Cutlass Conundrums Surface****: Potential version differences in **Cutlass** were raised as a possible cause for errors, specifically in environments outside of the maintained Modal image.
   - The reference **Cutlass** version installed on the modal image can be found [here](https://github.com/gpu-mode/kernelbot/blob/main/src/runners/modal_runner.py), but support is limited for the NVIDIA runner environment.


  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/)** (1 messages): 

itamos_64597: https://www.vincentsitzmann.com/blog/bitter_lesson_of_cv/
  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1473433565281911002)** (24 messages🔥): 

> `async pipelining in triton, TMEM interaction, warp shuffles, hierarchical reductions, DSMEM interactions` 


- **Newbie SGEMM CUDA Softmax Kernel**: A first year undergrad student has experience writing **SGEMM** kernels and a **softmax** kernel in **CUDA**, and is looking to participate in competitions and challenges.
   - One member suggested focusing on **CUDA/NVIDIA** first and going deep there before branching out to **AMD/TPU**.
- **Port Kernels from vLLM to sglang**: Members suggested contributing to inference frameworks like **vLLM** or **sglang** to have a real-world impact by porting a kernel from vLLM that’s missing in sglang.
   - They share a fairly similar codebase structure, so learning one makes it easier to navigate the other to check **vLLM’s list of unimplemented kernels** and pick something approachable.
- **Tilelang DSL Syntax Support Warp Shuffles**: Expressing ideas isn’t really possible in **Trion** for things like async pipelining in triton, interaction with TMEM, warp shuffles, hierarchical reductions, and DSMEM interactions.
   - One member suggested using **tilelang** or inventing your own syntax, because DSLs are just tools of thought that make expressing certain ideas easy, like computing and collecting like **butterfly reduction**.
- **Implement Backprop Says Karpathy**: **Karpathy** has the school of thought that every serious **ML guy** should be able to implement **backprop** because *backprop muscles give you unique insight into ML processes* [https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b).
   - His goal is to make ideas approachable without intimidating halo as he sticks to this principal to teach building LLM, with a link to the **Blackwell Programming for the Masses With OpenAI Triton** PDF: [https://semianalysis.com/wp-content/uploads/2025/03/Blackwell-Programming-for-the-Masses-With-OpenAI-Triton-Phil-Tillet.pdf](https://semianalysis.com/wp-content/uploads/2025/03/Blackwell-Programming-for-the-Masses-With-OpenAI-Triton-Phil-Tillet.pdf).
- **Tinygrad Prime Intellect Kernel Competitions**: There was mention of **kernel competitions** and **tinygrad** having kernel competitions running.
   - **George Hotz** has bounties which he uses a filter to hire people, with **Prime Intellect** also hiring people who published good envs to their hub.


  

---


### **GPU MODE ▷ #[flashinfer](https://discord.com/channels/1189498204333543425/1464407141128339571/1473499744327172332)** (21 messages🔥): 

> `NCU B200 woes, Verda GPU provider, flashinfer DSA kernel shapes, flashinfer speedups, MLSys Contest submission` 


- **NCU access proves elusive on B200**: Members report difficulty using **NCU** during experiments on the **B200**, despite checking Modal's Slack and finding no successful attempts, possibly due to most backends using **sm100** instead of **sm100a**.
   - Raw **PTX** code crashes with the **TVM FFI backend**, leading to a search for alternative profiling solutions.
- **Verda emerges as NCU profiling oasis**: An online GPU provider named **Verda** offers access to **NCU** for **B200** profiling, requiring a $20 deposit, with hourly rates of **$4.90** (On-Demand) or **$1.70** (Spot), both billed per 10 minutes.
   - One user reported a profiling run charging **$0.53** for a short pod usage, while also stating *"i promise im not sponsored"*.
- **Flashinfer flaunts fast Kernel feats**: A member stated that FlashInfer's speedup is roughly **60-70x** on most workloads, and linked to [their benchmark](https://bench.flashinfer.ai/kernels/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048).
   - However, a user tested the code examples and found that the actual speedups range from roughly **0.5× to 1.5×**, far from the claimed **60×–70×**, calling the discrepancy quite puzzling.
- **FlashInfer's DSA kernels only decode?**: Members inquired whether DSA kernel shapes are currently optimized only for decode, or if prefill shapes will be available later, referencing this [dataset](https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest/blob/main/workloads/dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl).
   - The original poster did not respond at the time of this summary.
- **MLSys Contest submission infrastructure probed**: Members asked about the submission infrastructure for the **MLSys Contest**, inquiring about the ability to install additional **Python** packages via the *"dependencies"* key in the solution.json, and about the submission of speedups on the dashboard.
   - The original poster did not respond at the time of this summary.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1473468904645857416)** (84 messages🔥🔥): 

> `Kimi code vs kimi claw, Kimi Subscription Issues, API Rate Limit, Kimi's weaknesses` 


- **Kimi Code vs Kimi Claw**: A user inquired about the difference between using **Kimi Code** and **Kimi Claw** to code a website, asking which is better for constant bug fixing and rebuilding code.
   - No clear answer was provided in the discussion.
- **Kimi Subscription Support Issues Plague Users**: Several users expressed frustration about the lack of support and disappearing subscriptions, with one noting they received SMS messages from random numbers when adding their mobile number to their account.
   - Another user said, *emailed 2 days ago about my sub just dissapearing no answer*.
- **API Rate Limit Reached despite Positive Balance**: One user reported consistently seeing the 'API rate limit reached' error despite having a positive balance and being on tier 3.
   - It was suggested they check their concurrency or RPM limits or email [api-service@moonshot.ai](mailto:api-service@moonshot.ai) for assistance.
- **Kimi Code for Opencode.ai**: A user reported using **Kimi with OpenCode**.
   - Another user confirmed that this worked by using the second coding option in OpenCode.
- **Is Spacial Thinking Kimi's kryptonite?**: A user shared a screenshot showing that Kimi struggles to understand spatial relationships, such as whether to walk or drive a short distance.
   - Adding *Imagine from a spacial perspective* seemed to improve things, but a python script was needed for valiation.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1473424564108267550)** (75 messages🔥🔥): 

> `Nous AI bulky responses, Relationship with AI, YouTube down, GLM 5 technical report, Chinese AI labs` 


- ****Nous AI** branded *too bulky* by users**: Users on discord criticized the length of **Nous AI's** responses to simple questions and described them as *bulky*.
   - There was a discussion about whether the bulkiness referred to the thinking trace or the general response length.
- **Discord members discuss relationship with AI**: After a [tweet about relationships with AI](https://fxtwitter.com/EthanHe_42/status/2023862949715325304), members on the discord channel discussed their views on the subject.
   - One user said they *have not actually seen the conversations themselves, and can't possibly fathom how someone would have a relationship with an AI*.
- ****Youtube** suffers network outage**: A member reported that **Youtube** was down on their network across multiple IPs.
   - Another user had a similar experience and quoted a message *This page appears when Google automatically detects requests coming from your computer network which appear to be in violation of the Terms of Service.*
- ****GLM 5 Technical Report** fails to wow users**: The [GLM 5 technical report](https://arxiv.org/abs/2602.15763) was released but a user described it as *not super interesting as usual lol*.
   - They added that these reports are usually dry, involve known techniques, and are more of an engineering problem than a research breakthrough.
- ****Chinese AI Funding** is the hot topic**: A user stated that **China** has amazing AI infra, funding and human resources, supported by the government, which should be admired instead of criticized.
   - Another user stated that **US** government support is negligible compared to the Chinese government, with **Chinese AI** largely government-funded while **American AI** is almost entirely private sector.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

123mikeyd: Nous girl Lofi Take 1:   https://www.youtube.com/watch?v=-xlCIsccSjQ
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1473451403740446873)** (11 messages🔥): 

> `AI Coding Hostility, VLA for Robotics, Nvidia's FLARE` 


- **AI Coding faces Reddit Account Suspensions**: A member reported facing *hostility towards **AI coding*** and having **three Reddit accounts suspended** after mentioning **Codex** or **ChatGPT**.
   - Another member asked if they *made ai generated prs to github repo* to which the original poster denied and said it was research code.
- **Codex config flagged as AI-generated text**: A member's account was suspended from **r/codex** after sharing their `~/.codex/AGENTS.override.md` file in a thread about best practices, potentially triggering a bot.
   - They speculated that the bot may have mistaken the file's content as *randomly pasting AI-generated text*.
- **VLA Collaboration sought for Robotics Project**: A member is seeking collaborators with experience in **VLA (Vision-Language-Action)** for a short-term robotics project.
   - The project aims to implement something similar to **Nvidia's FLARE** for calibrating errors in cheap 3D printed robots, potentially taking multiple directions, and they have an industrial robot arm available.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1473446714995114196)** (34 messages🔥): 

> `CoDA-GQA-L release, KV cache efficiency, Ablation studies for differential attention, Mycelium's AI benchmarking paper, MLP-replacing regression algorithm` 


- ****CoDA-GQA-L** Bounded-Memory Attention Released!**: A member released **CoDA-GQA-L**, a bounded-memory attention mechanism that caps KV cache at a fixed size, using **136 MB** for a 70B model processing 128K tokens, with code available on [GitHub](https://github.com/anthony-maio/CoDA-GQA-L) and paper on [Zenodo](https://zenodo.org/records/18663265).
   - It uses **384 slots per layer**, with a recent window (**256 tokens**), an exact landmark bank (**64 novelty-filtered tokens**), and a summary bank (**64 EMA prototypes**).
- **Differential Attention Ablation Ablations Asked!**: A member questioned the necessity of differential attention for KV cache changes, asking if there were any ablations comparing it to a non-differential transformer.
   - The author acknowledged the lack of ablations due to resource constraints but plans to add them, emphasizing that the **KV cache reduction**, not differential attention itself, drives the **memory efficiency**.
- ****Mycelium** Seeks Advice on AI Benchmarking Paper!**: A member from **Mycelium** ([https://github.com/Mycelium-tools](https://github.com/Mycelium-tools)) requested advice on journals or conferences for publishing a paper on AI model benchmarking, similar to [inspect_evals](https://ukgovernmentbeis.github.io/inspect_evals/evals/safeguards/ahb/), but adapted for dynamically generated multi-turn conversations and AI agents.
   - They're especially interested in balancing journal prestige with suitability and ease of acceptance.
- **MLP-Replacing Algo Feedback Invited!**: A member shared a link to their **MLP-replacing regression algorithm** and invited feedback, noting similar backgrounds and years of experience with the original poster.
   - The algorithm factors the input into the **spine and linear deviations**, offering an elegant and explicit solution that's fast to evaluate where its structure is appropriate, with a [torch implementation](https://github.com/taiwei-shi/k-splanifold) available.
- **Attention is All You Need... Twice!**: *Putting in a prompt twice in a row seems to make all SOTA LLMs perform better lol*
   - Another member shamelessly plugged some content of possible interest - [paper](https://arxiv.org/abs/2602.15456), [thread](https://x.com/i/status/2024011556019687660) and [related work](https://fxtwitter.com/cwolferesearch/status/2023408158065188894).


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1473605264732328080)** (13 messages🔥): 

> `MI Toolkit in Rust, Feature Steering Vectors for Consistency Regularization, Bounds on LLMs` 


- ****Rust Tool Replicates Anthropic's Finding on Gemma 2 2B****: A member built an **MI toolkit in Rust** on top of candle ([plip-rs](https://github.com/PCfVW/plip-rs)) and replicated Anthropic's *planning in poems* finding on **Gemma 2 2B**.
   - The core result: suppress + inject position sweep reproduces the Figure 13 shape with a **ten-million-fold spike at the planning site**, and the candle team approved it as candle-mi ([candle discussion](https://github.com/huggingface/candle/discussions/3368)).
- ****Steering Vectors Augment Multilingual Training Data****: A member proposed using **language-feature steering vectors** to augment a minibatch of English sentences into many *pseudo-multilingual* minibatches and supervise performance with RL from Feature Rewards.
   - This method could potentially train on data for **20 languages at approximately half cost** without needing data from these other languages, as elaborated [on Twitter](https://fxtwitter.com/kennylpeng/status/2023784570878128178).
- ****Members Debate Weaker Bounds on LLMs with Probability****: Members discussed bounds on LLMs, citing a [paper](https://arxiv.org/abs/2602.11246) with an odd publication date of February 11, 2026, and another [paper](https://arxiv.org/abs/2409.15318) with different bounds.
   - One member suggested allowing the results to weaken themselves to with **1-epsilon probability** because *LLMs are not some perfect machine*.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1473700624607740046)** (3 messages): 

> `Concurrent Task Evaluations, API calls in metrics.py, Batch support` 


- **Concurrent Evaluation quest kicks off**: A member inquired about performing **concurrent task evaluations** to speed up tasks using API calls in `metrics.py`.
   - A member suggested modifying the task to write model generations to disk and then using custom batching for metric computation, also said *batch support is on the project plan*.
- **API Batching Bonanza**: A member pondered batching API calls in the aggregation step as a potential, albeit *hacky*, solution.
   - They expressed intent to experiment with this approach.


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1473752398303330439)** (1 messages): 

> `VLM Blindness, Linear Probing Accuracy` 


- **VLMs Shown to be Visually Impaired**: A member shared the paper [Are VLMs Really Blind?](https://vlmsareblind.github.io/) which demonstrates that **state-of-the-art VLMs** struggle with simple visual tasks like counting circles or identifying intersecting lines.
   - The member inquired about potential solutions, suggesting that **SFT** (Supervised Fine-Tuning) or **RLVR** (Reinforcement Learning from Visual Reasoning) on similar data might improve performance.
- **Linear Probing Achieves High Accuracy**: The same member noted that **linear probing** on vision encoder features can achieve close to **100% accuracy** on the aforementioned tasks.
   - This suggests that the visual information *is* captured by the encoder, but the VLM struggles to properly utilize it, as visualized in a linked [screenshot](https://cdn.discordapp.com/attachments/795089627089862656/1473752398580416522/Screenshot_2026-02-18_at_10.43.06.png?ex=69975a60&is=699608e0&hm=dde1019dcf83f592c6fa934531f79d3ef6b667004388e4ebeab78fa24e850aac&).


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1473416055811739721)** (20 messages🔥): 

> `Jupyter Mojo kernel, Linux installation issues, Self-Evolutionary System, GPU support on solve, Modular shout-out` 


- **Jupyter Mojo Kernel Released**: Jeremy Howard released a [Jupyter Mojo kernel](https://github.com/AnswerDotAI/mojokernel), noting it is *barebones* but **fast** and works well on Mac.
   - It is **pip installable** and precompiled for MacOS and recent Linux versions, and uses **uv** to auto-install the matching modular package.
- **Linux Installation Quirks Sorted**: A user tested the **Jupyter Mojo kernel** on Linux (Ubuntu 24.04 LTS) and reported that the installation was not straightforward, requiring `MOJO_VERSION=26.1.0.post1 uv add "mojokernel>=26.1.0"` due to a **version mismatch**.
   - Jeremy Howard fixed this issue, having forgotten to build for multiple python versions.
- **Self-Evolutionary System Architected**: One member is *architecting a Self-Evolutionary System*, applying **Ricci Flow** to eliminate geometric noise, **Kolmogorov Complexity** for algorithmic efficiency, and **Gödelian logic** to acknowledge intelligence looking beyond its own programming.
   - This member stated *I’m not writing lines of code; I’m defining the laws of a digital cosmos.*
- **GPU Support Craved**: A user requested **Mojo** and **GPU** support, as this would be *so cool for learning*.
- **Modular Gives Shout-Out**: Modular gave Jeremy Howard a shout-out on X [Modular status update](https://x.com/Modular/status/2024198871875014815).
   - Jeremy Howard acknowledged the shout-out *Yay thank you for the shout-out, Modular!*


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1473446766857814190)** (21 messages🔥): 

> `Mojo C++ Binding, GNU Radio binding, Mojo's Origins and Philosophy, Rng vs random module, Dunder Methods Empty` 


- **Mojo's C++ Bindings are Mostly Manual**: Mojo C++ bindings involve *mostly round-trip via C with manual bindings* according to one member.
   - In response to a question of whether it is as easy as with pybind11, the answer was: *Not really. Especially since you need to write the bindings by hand.*
- **GNU Radio bindings in the works?**: One member mentioned they are *thinking about making binding for GNU Radio* via [this github repo](https://github.com/gnuradio/gnuradio).
   - Another member suggested *one solution you may find is to instead have 2 separate processes and some shared memory to talk with.*
- **Dig into Mojo's Origins and Philosophy**: If you want even more detail on Mojo's origins and philosophy, one member recommended reading the [vision document](https://docs.modular.com/mojo/vision).
- **`Rng` in testing is a WIP!**: The `Rng` in testing is used specifically for property testing, and is still a **WIP** (it uses the functions from `random` internally).
   - For general purpose rng, you should use the `random` module. We could potentially expose a way to construct a generator, but currently it just uses a single global generator iirc.
- **Dunder Methods Appear Empty**: One member asked why all **dunder methods** are empty, such as the body of the `__repr__()` method.
   - The answer was: *it's just because it's a new one for each type and I wasn't understanding in the code of the repr() method why the `__repr()__` dunder method was empty.*


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1473414040024912087)** (2 messages): 

> `Modular Build Infrastructure Enhancements, Custom Mojo Kernels, MXFP4 Kernels, Graph Compiler` 


- **Custom Mojo Kernels Now Usable with MAX Models**: The `modular` repo build infrastructure received enhancements alongside new capabilities in the graph compiler, and now MAX graphs and models built using the OSS `modular` repo can use a fully customized Mojo standard library or Mojo kernels, according to [this forum post](https://forum.modular.com/t/max-models-can-now-use-customized-mojo-kernels-and-standard-library/2742).
- **MXFP4 Kernel Collabs Incoming**: Members have been working on **mxfp4 kernels** with the goal of requantizing to nvfp4.
   - Other members are reaching out to the kernel team to see if collaboration is possible.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1473429655280287865)** (15 messages🔥): 

> `AI Researcher with Self-Improving Capabilities, Local AI Rigs and Hardware Setups, GPT OSS 120B Performance, Activity on HuggingFace.co, ComfyUI Workflow` 


- ****Self-Improving AI Researcher Project Sparks Interest****: A member expressed interest in working on a private project to create an **AI researcher** with *self-improving capabilities*, focusing on improving existing AI rather than creating new AI from scratch.
   - The member was curious about local **AI rigs** and if there was a dedicated thread for discussing hardware setups.
- ****Hardware Enthusiasts Seek Advice on Concurrent Requests****: One user is looking to hear about what users with **48+gb of vram** are doing, with a focus on concurrent requests for agent apps with local hardware and is also interested in RTX pro 6000 Blackwell or clusters of second hand GPUs.
   - The user stated that, from their experience, anything below **GPT OSS 120B** isn't worthwhile, relying on OpenRouter for now, while considering a hybrid approach.
- ****Suspicious Activity Flagged on HuggingFace.co****: A user questioned unusual activity on **HuggingFace.co**, specifically pointing to a user exhibiting potential malicious behavior.
   - Another user suggested it was due to posting too quickly.
- ****ComfyUI workflow vanishes into thin air****: A user shared links related to **ComfyUI**, including a [model](https://civitai.com/models/1000401/noob-ipa-mark1) and a [workflow](https://comfyworkflows.com/workflows/3ca41c2c-a1ac-41b7-ae00-01cbad96ef78).
   - Another user commented on the disappearance of a link in the channel.
- ****Kaggle Dependency Clashes Plague PEFT Adapter Layer Evaluation****: A user faced dependency clashes while running an evaluation script on a Kaggle notebook using a model paired with a **PEFT adapter layer**.
   - The user asked for advice on fixing the notebook or finding a suitable reference notebook.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1473416839433556071)** (13 messages🔥): 

> `MCP Server for Flights/Hotels, Memory Ecosystem for LLMs, AI App Security & Observability, Microclaw for OpenClaw, ModernBERT Models` 


- ****MCP Server** Built for **Flights and Hotels Search****: A member has built a **MCP server** for flights and hotels search, available at [delulu](https://github.com/mratsim/delulu).
   - Screenshots of the **Delulu flights search** and **Delulu hotels search** user interface were shared for feedback.
- ****LLM Memory** Ecosystem Explored for **'WhereIKept'** App**: An AI engineer is working on a memory ecosystem for LLMs, using on-device speech-to-text and a multimodal LLM to create an app named **WhereIKept**, with the goal of helping users remember where they stored items, with the project open sourced at [WhereIKept](https://github.com/AjjayK/WhereIKept).
   - Unsolved problems mentioned include **object identity across locations**, **smarter retrieval**, **on-device optimization**, and the **right form factor** for the app.
- ****AI Firewall** Under Development for **Real-Time Security****: A research effort is underway to provide a "firewall" for AI apps, offering real-time prompt injection detection, PII scanning, and cost control for Agentic Apps, with the github repo available at [llmtrace](https://github.com/epappas/llmtrace).
   - The developer promises to publish benchmark results soon.
- ****Microclaw for OpenClaw** Provides **Enhanced Fallback Agent****: Microclaw for OpenClaw (v2026.2.18) is an enhanced fallback agent model designed for OpenClaw, offering a local, high-quality alternative when primary models are unavailable, as described on its [HuggingFace page](https://huggingface.co/webxos/microclaw-for-openclaw-version-2026.2.18).
   - Another version is [also available](https://huggingface.co/webxos/microclaw-for-openclaw-version-2026.2.17) for testing, noting the different install procedures, and the model uses a **2048 token context length**.
- ****ModernBERT-small-v2** Offers **Improved Local Applications****: An improved iteration of small ModernBERT models has been released, intended for local applications without GPU, available on its [HuggingFace page](https://huggingface.co/johnnyboycurtis/ModernBERT-small-v2).


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1473781060289564742)** (1 messages): 

> `gr.HTML, Gradio 6, HF Collection` 


- **Gradio 6 drops gr.HTML blogpost!**: The Gradio team released a blogpost on **gr.HTML**, Gradio 6's custom component that allows users to create complete web applications using only a single Python file, with a [link to the blog here](https://huggingface.co/blog/gradio-html-one-shot-apps).
- **Full web apps are now one-shot with gr.HTML!**: The blog post mentions that Claude or any Frontier LLM can now generate web applications using a single prompt and in a single Python file thanks to **gr.HTML**.
   - Several example applications, such as a **Kanban board**, **Pomodoro timer**, and **Github heatmaps**, were created to demonstrate the capabilities of the new tool, with a [link to the HF Collection here](https://huggingface.co/collections/ysharma/custom-html-component).


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1473643503291596893)** (2 messages): 

> `Smol-Course Release` 


- **Smol-Course's Release Date Still Unknown**: A member inquired about the release date of `smol-course` as mentioned in [this link](https://github.com/huggingface/smol-course?tab=readme-ov-file#future-of-this-course).
- **Future plans for Smol-Course unclear**: The future plans for Smol-Course are unclear.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1473424532869087526)** (20 messages🔥): 

> `Masking Gradients vs Masking Data, SkipUpdate for Performance, LoRA Similarities, Block Dropout Explanation, RPROP Optimizer` 


- **SkipUpdate Masks Gradients for Performance**: A discussion ensued about [SkipUpdate](https://x.com/i/status/2024087619756318866) and whether it differs from **BERT** and **MAE**, concluding it shifts from masking data to masking the gradient for some parameters.
   - One member suggested the goal is scalable supervision, whereas another thought the goal was performance.
- **LoRA, SkipUpdate, and Weight Updates**: Participants debated whether **SkipUpdate** is similar to **LoRA**, with one member noting LoRA updates all parameters associated with a LoRA module indirectly.
   - Another clarified that unlike LoRA, **SkipUpdate** doesn't save memory, it would only be useful if it improved performance.
- **Block Dropout's Gradient Masking**: A member explained that [Block Dropout masks entire blocks' gradients](https://x.com/_chenglou/status/2024187065076957620) but updates momentum terms, penalizing blocks with high second-order variation.
   - It was also noted that scaling the gradient based on the alignment between gradient and momentum is similar to the ancient **RPROP optimizer**.
- **RPROP Optimizer Revisited**: A member suggested that scaling the gradient based on the alignment between gradient and momentum is similar to the ancient [RPROP optimizer](https://ieeexplore.ieee.org/document/298623).
   - *RPROP can still be a really strong optimizer in cases where you have high noise*.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1473765796210802893)** (1 messages): 

> `Sutton & Barto Discussion` 


- **Sutton & Barto Meeting Time Confirmation Needed**: A member requested confirmation of the meeting time to discuss **Sutton & Barto**.
   - They also requested clarification on the specific segment of the book that will be covered.
- **Seeking Details on Sutton & Barto Discussion**: A participant inquired about the scheduled discussion on **Sutton & Barto**, seeking the exact time.
   - Additionally, they asked for a reminder about the specific section of the book to be reviewed during the meeting.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1473763225475485706)** (2 messages): 

> `Lyria model by DeepMind, OpenEval framework` 


- **DeepMind's Lyria Model Resurfaces**: DeepMind's **Lyria model** got a mention in the chat, with a link to its [official page](https://deepmind.google/models/lyria/).
   - It was noted to be *a little old*, but still relevant in context of music creation models.
- **OpenEval Framework Sparks Interest**: The **OpenEval** framework was highlighted as interesting, potentially due to its relevance to previous news posts.
   - It was mentioned alongside [this X post](https://x.com/lpachter/status/2018759999141691489), though without additional context.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1473491836327821418)** (8 messages🔥): 

> `tinygrad, BarraCUDA, MFMA` 


- **Hotz Calls for Contributions to tinygrad**: George Hotz encourages the developers of [this project](https://news.ycombinator.com/item?id=47052941) to contribute to **tinygrad**, criticizing their choice of **C** and lack of **CI**.
   - He suggests using their [emulator](https://github.com/Zaneham/BarraCUDA/issues/17) and offers to pay the **CDNA bounty** for adding **GEMM/flash attention** tests and cleaning up the code.
- **MFMA assertion fix needed**: A member points out an assertion in `_compile_mfma` that restricts **MFMA** support to **16x16** matrices in [this line of code](https://github.com/tinygrad/tinygrad/pull/1481).
   - The member questions whether **4x4** and **32x32 MFMAs** should also be supported beyond the current testing scope.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1473751482196037827)** (2 messages): 

> `tinygrad puzzles, solve.it solutions` 


- **Student solves tinygrad puzzles on Solve It**: A student posted their worked solutions to all the **tinygrad puzzles** on [Solve It](https://share.solve.it.com/d/5e959dddb333ea2a30ccc6deb8ce3eec).
   - The puzzles cover various aspects of **tinygrad**.
- **Solve It hosts tinygrad solutions**: The solutions to the tinygrad puzzles are hosted on [Solve It](https://share.solve.it.com/d/5e959dddb333ea2a30ccc6deb8ce3eec).
   - **Solve It** is a platform for sharing and collaborating on solutions to programming puzzles.


  

---


### **MCP Contributors (Official) ▷ #[mcp-dev-summit](https://discord.com/channels/1358869848138059966/1413517834805313556/1473480402839670886)** (2 messages): 

> `` 


- **Empty Discord Chat**: The provided Discord chat log is empty, containing only filler messages.
   - Therefore, there are no topics to summarize or discuss.
- **No Conversation Here**: The provided messages do not constitute a meaningful conversation.
   - There is no actionable content or discussion points to extract.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1473415116031787202)** (8 messages🔥): 

> `MCP Payment Support, X402 payment protocol, Microtransactions for Agents` 


- **MCP Servers Propose Monetization via SEP**: A member created a [SEP](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2007) to allow MCP servers to request money for tools, starting with **X402**, to boost agent and MCP adoption.
   - The creator believes this could significantly accelerate Agents and MCP adoption due to the introduction of monetization incentives.
- **MCP Payment Support Questioned**: A member questioned the need to build payment support into the protocol, suggesting that **URL elicitation** should handle out-of-band payments.
   - The member outlined a flow where a server sends a **URL elicitation request** for payment, and service is granted upon confirmation.
- **Micropayments for Autonomous Agents**: A member clarified that the SEP targets **micropayments** (in cents) for agents to autonomously pay for tools, operating under budget guardrails.
   - These agents require rich information on tool costs to make intelligent decisions for deep research.
- **X402 Payment Protocol favored**: A member expressed agreement with waiting for payment protocols to stabilize, but another suggested starting with **X402**, highlighting its current prominence.
   - The member assured that the **SEP** would be designed to be extensible for future payment protocols.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1473409339028148276)** (9 messages🔥): 

> `13yo dev from Baghdad verified, Full stack developer introduction, Manus account issues, Subscription cancellation, Manus shines in Job hunting` 


- **Baghdad-Based BlockChain Whiz Boasts Verification**: A **13yo developer** from Baghdad 🇮🇶 announced their official verification and experience in **Blockchain and AI Agents**.
   - They are proficient in **EVM, Solana, Sui, XRP, Cardano, Midnight, zk-SNARKs**, **React, Next, Vue, Node**, and is available for collaboration.
- **Full Stack Friend Seeks Future Fellowships**: A full stack developer introduced themselves with experience in **web applications, API integrations, and data pipelines**.
   - Their stack includes **react/next.js, node.js/Django, python frameworks and libraries (TensorFlow, Pytorch, OpenCV, NumPy)**, and is skilled in **AWS/Docker** for building scalable apps, focusing on *real world products*.
- **Manus Meltdown: Member's Masterpiece Mired in Mayhem**: A member reported severe issues with their Manus account, where a **presentation built over multiple weeks is now riddled with errors**.
   - Despite being visible in their presentation history, the presentation *cannot be re-instated no matter what I do*.
- **Subscription Snafu: System Savior Steps into Scene**: A member, @sysing, warned that *if you don’t cancel the subscription, you may still be charged*.
   - They requested the affected user to send their registered email via DM to resolve the issue.
- **Manus Masters Messy Marketplace of Modern Mobility**: A member expressed gratitude for Manus's assistance in job hunting, noting that it *shines* where even Best Buy's website fails to properly autofill résumés.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1473647438677020835)** (1 messages): 

> `AI security, Observability, Prompt injection, PII scanning, Cost control for Agentic Apps` 


- **AI App Firewall Project Kicks Off**: A member announced a new research effort on providing a *"firewall"* with **real-time prompt injection detection**, **PII scanning**, and **cost control** for **Agentic Apps**.
   - The project's [GitHub repo](https://github.com/epappas/llmtrace) is available for feedback, and benchmark results will be published soon.
- **llmtrace GitHub Repo Awaits Feedback**: A member has created a new **GitHub repository** called **llmtrace** related to the new *"firewall"* research.
   - The author seeks community feedback on the [project](https://github.com/epappas/llmtrace), focusing on **security aspects of AI**.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1473667009320456293)** (5 messages): 

> `Hegelian dialectic Exercise, dspy.dialectic, Community Office Hours, RLMs simplifying tasks, Real user feedback` 


- **Hegelian Dialectic exercise concluded**: A member concluded the **Hegelian dialectic Exercise**, thanking others for their support.
- **Office Hours scheduled!**: Community office hours were announced for Feb 19th at 11:30am ET via a [Zoom link](https://mit.zoom.us/j/93374418319).
- **RLMs simplify tasks!**: A member shared [Monolith on Github](https://github.com/WingchunSiu/Monolith), calling it an ingenious piece of work and evidence for **RLMs** simplifying tasks that required a LOT more boilerplate and orchestration before.
- **Request for real user feedback**: A member asked whether *offline* means real user feedback, linking an [issue on the gepa-ai/gepa repo](https://github.com/gepa-ai/gepa/issues/178) where they've shared some ideas.


  
