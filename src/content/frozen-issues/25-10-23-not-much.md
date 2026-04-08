---
id: MjAyNS0x
title: not much happened today
date: '2025-10-23T05:44:39.731046Z'
description: >-
  **LangSmith** launched the **Insights Agent** with multi-turn evaluation for
  agent ops and observability, improving failure detection and user intent
  clustering. **Meta PyTorch** and **Hugging Face** introduced **OpenEnv**, a
  Gymnasium-style API and hub for reproducible agentic environments supporting
  distributed training. Discussions highlighted the importance of provider
  fidelity in agent coding, with **OpenRouter**'s exacto filter improving
  stability. Builder UX updates include **Google AI Studio**'s Annotation mode
  for Gemini code changes, **Microsoft**'s Copilot Mode enhancements in Edge,
  and **OpenAI**'s Shared Projects and Company Knowledge features for ChatGPT
  Business. **Claude** added project-scoped Memory. In reinforcement learning,
  **Meta**'s ScaleRL proposes a methodology to predict RL scaling outcomes for
  LLMs with improved efficiency and stability.
companies:
  - langchain
  - meta-ai-fair
  - hugging-face
  - openrouter
  - google-ai
  - microsoft
  - openai
  - anthropic
models:
  - gemini-1.5-pro
  - claude-3
  - chatgpt
topics:
  - agent-ops
  - observability
  - multi-turn-evaluation
  - reinforcement-learning
  - distributed-training
  - api
  - model-stability
  - user-intent-clustering
  - software-development
  - project-management
  - code-generation
people:
  - hwchase17
  - ankush_gola11
  - whinthorn
  - koylanai
  - _lewtun
  - bhutanisanyam1
  - thom_wolf
  - danielhanchen
  - cline
  - canvrno
  - pashmerepat
  - mustafasuleyman
  - yusuf_i_mehdi
  - jordirib1
  - fidjissimo
  - bradlightcap
  - mikeyk
  - alexalbert__
---


**a quiet day**

> AI News for 10/22/2025-10/23/2025. We checked 12 subreddits, 544 Twitters and 23 Discords (198 channels, and 8784 messages) for you. Estimated reading time saved (at 200wpm): 592 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

a quiet day.

---

# AI Twitter Recap

**Agent ops, observability, and real-world envs**

- **LangSmith ships “Insights Agent” + multi-turn evals**: LangChain introduced an in-product agent that scans traces to auto-cluster usage patterns and failure modes, plus multi-turn evals to assess goal completion across full conversations. Teams report near-immediate visibility into silent failure classes and user intent clusters without manual triage. See launch threads and details from [@LangChainAI](https://twitter.com/LangChainAI/status/1981390300502487370), [@hwchase17](https://twitter.com/hwchase17/status/1981390508841980332), and engineering notes from [@Hacubu](https://twitter.com/Hacubu/status/1981396190077043162), [@ankush_gola11](https://twitter.com/ankush_gola11/status/1981408009097265344), [@WHinthorn](https://twitter.com/WHinthorn/status/1981403256598192451), and a hands-on analysis by [@koylanai](https://twitter.com/koylanai/status/1981444604869087624).
- **OpenEnv: a shared spec and hub for agent/RL environments**: Meta PyTorch and Hugging Face launched OpenEnv, a Gymnasium-style API (reset/step/state), built for container/server execution with simple HTTP, and a Hub for reproducible “agentic environments” (tools, credentials, sandboxing). Early integrations span TRL, Unsloth, Atari, and community examples (e.g., Poker), aiming to standardize env packaging and scale distributed training. See [@_lewtun](https://twitter.com/_lewtun/status/1981380372748521929), [@bhutanisanyam1](https://twitter.com/bhutanisanyam1/status/1981377720157351938), [@Thom_Wolf](https://twitter.com/Thom_Wolf/status/1981396028117901401), and [@danielhanchen](https://twitter.com/danielhanchen/status/1981428184215363956).
- **Agent coding in the wild: provider fidelity matters**: Cline highlights how identical open-weight models behave radically differently across inference endpoints (quantization, tool-call formatting, “thinking” tags), often causing users to blame the model instead of the infra. Their fix combined aggressive system prompt reduction and provider filtering (e.g., OpenRouter’s :exacto) to restore stability. They’re also releasing ClineBench with real-world, interruptible tasks. See [@cline](https://twitter.com/cline/status/1981370535176286355), the breakdown by [@canvrno](https://twitter.com/canvrno/status/1981403534471119330), and [@pashmerepat](https://twitter.com/pashmerepat/status/1981431374386233840).
- **Builder UX updates, briefly**: Google AI Studio’s new Annotation mode lets you “mark up” the live app UI and have Gemini apply the code changes ([announcement](https://twitter.com/GoogleAIStudio/status/1981375306423554490), [demo](https://twitter.com/patloeber/status/1981375563685384430)). Microsoft rolled out Copilot Mode in Edge (Journeys, Actions), Mico voice UI, and upgraded search grounding inside Copilot ([@mustafasuleyman](https://twitter.com/mustafasuleyman/status/1981390345578697199), [@yusuf_i_mehdi](https://twitter.com/yusuf_i_mehdi/status/1981426387958583717), [@JordiRib1](https://twitter.com/JordiRib1/status/1981399255576174657)). OpenAI added Shared Projects and “Company knowledge” (Slack, Drive, GitHub, etc.) for ChatGPT Business/Enterprise/Edu ([@OpenAI](https://twitter.com/OpenAI/status/1981432799212249119), [@fidjissimo](https://twitter.com/fidjissimo/status/1981437695915413947), [@bradlightcap](https://twitter.com/bradlightcap/status/1981454865454027007)), while Claude shipped project-scoped Memory ([@mikeyk](https://twitter.com/mikeyk/status/1981415275695394852), [@alexalbert__](https://twitter.com/alexalbert__/status/1981421146886328778)). Also worth a look: Firecrawl’s integration guides across LangChain, n8n, and MCP ([@firecrawl_dev](https://twitter.com/firecrawl_dev/status/1981390679462072766)), and Vercel’s “useworkflow” for durable async tasks in TypeScript ([@cramforce](https://twitter.com/cramforce/status/1981399119559348290), [@rauchg](https://twitter.com/rauchg/status/1981426366982824387)).

**RL for LLMs: scaling laws, stability, and off-policy**

- **ScaleRL (Meta): toward predictable RL scaling**: New work proposes a recipe and methodology to predict LLM RL outcomes from small runs, with design choices like PipelineRL-8 (async), CISPO loss, FP32 compute, prompt-average loss, batch-level norm, zero-variance filtering, and No-Positive-Resampling. Claims accurate extrapolation up to 100k GPU-hours and better efficiency vs GRPO/DAPO/Magistral. Summaries: [@TheTuringPost](https://twitter.com/TheTuringPost/status/1981487666714800356) and paper link therein.
- **Avoiding RL collapse via train–inference alignment**: A deep post-mortem shows minor framework/precision divergences (KV cache precision, softmax/norm in FP32, RoPE deltas, attention backend differences, MoE routing stability) accumulate across layers/tokens—especially in MoE and long rollouts—causing collapse. The prescription: per-layer activation logging and alignment across prefill/decode, consistent numerics, and high-precision routing. Read the technical checklist via [@ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/1981337266523164694).
- **Memory-based continual learning and off-policy RL**: Memento reframes agent improvement as memory-based online RL over a memory-augmented MDP (case-based reasoning + executor over MCP tools), no weight updates needed ([thread](https://twitter.com/_avichawla/status/1981246733322768780) + [repo](https://twitter.com/_avichawla/status/1981246746497077492)). BAPO targets off-policy RL for LLMs in partial rollouts and experience reuse settings ([@Be1ong1](https://twitter.com/Be1ong1/status/1981297924564046007)). OpenEnv’s standardization (above) plus Unsloth/TRT/Llama ecosystems are converging on shared, reproducible envs for large-scale training ([@danielhanchen](https://twitter.com/danielhanchen/status/1981428184215363956)).

**Generative media, OCR/VLM surge, and robotics**

- **Open creative/video engines**: LTX released LTX-2, an open AI creative engine with synchronized audio+video, native 4K, up to 50 fps and 10s sequences, API-first design, efficient on consumer GPUs; weights coming later this year ([@ltx_model](https://twitter.com/ltx_model/status/1981346235194683497), [@LTXStudio](https://twitter.com/LTXStudio/status/1981371951894667279)). Argil announced Atom, emphasizing controllability and temporal consistency with no duration limits, plus a “style Tinder” for look selection ([launch](https://twitter.com/BrivaelLp/status/1981343140196778270), [try it](https://twitter.com/BrivaelLp/status/1981344149862314183)).
- **Robotics foundation models and OCR/VLMs**: NVIDIA’s Gr00t N1.5 (via LeRobot) is a cross-embodiment action model with vision/language/proprioception inputs and a flow-matching action transformer, trained on real/synthetic/internet-scale data; evaluated on Libero and real hardware ([@LeRobotHF](https://twitter.com/LeRobotHF/status/1981334159801929947)). OCR/VLMs are spiking: LightOnOCR-1B (end-to-end VLM) focuses on speed/throughput ([@staghado](https://twitter.com/staghado/status/1981379888301867299)), OlmOCR-2 leans on RLVR + binary unit tests for fast iteration ([@kylelostat](https://twitter.com/kylelostat/status/1981380820658180310)), and model comparisons are being updated rapidly ([summary](https://twitter.com/mervenoyann/status/1981396054634615280); VLM/OCR releases trending on HF per [@MaziyarPanahi](https://twitter.com/MaziyarPanahi/status/1981421331053760775)). Also notable: Runway’s “Apps for Advertising” collection to productize common video/image workflows without complex prompting ([announcement](https://twitter.com/runwayml/status/1981380360249159783)).

**Infrastructure and model platforms**

- **Anthropic x Google TPU mega-deal**: Anthropic plans to expand onto “approximately one million” TPUs and “well over” 1 GW capacity in 2026—tens of billions of dollars in compute—sharply expanding training/inference headroom ([@AnthropicAI](https://twitter.com/AnthropicAI/status/1981460118354219180), [follow-up](https://twitter.com/AnthropicAI/status/1981460119742533848)).
- **Serving stacks**: vLLM now serves NVIDIA’s Nemotron Nano 2 (9B hybrid Transformer–Mamba reasoning model, open weights, >9T tokens) with a tunable “thinking budget” for predictable cost/latency; vLLM claims up to 6× faster “thinking” token throughput vs similar open dense models, improving agent search/reflection ([@vllm_project](https://twitter.com/vllm_project/status/1981553870599049286)). Cerebras released REAP-pruned GLM‑4.6 MoE checkpoints at 25/30/40% compression (FP8, A32B) targeting efficiency with preserved generation quality ([@vithursant19](https://twitter.com/vithursant19/status/1981476324045967785)). Ollama published perf tests on NVIDIA Spark firmware + new builds ([@ollama](https://twitter.com/ollama/status/1981486870963114121)). Also: Qdrant launched a Vector Search “Academy” ([@qdrant_engine](https://twitter.com/qdrant_engine/status/1981319267749679599)), and Modular released Mojo GPU “puzzles” for hands-on CUDA/Metal learning ([@Modular](https://twitter.com/Modular/status/1981455872137318556)).

**Routing and serving fidelity**

- **Lookahead routing for multi-LLM systems**: Proposed “Lookahead” predicts latent representations of potential responses to get a cheap “peek” at what each model would say, enabling response-aware routing without full decode. Reported +7.7% average over SOTA routing across 7 benchmarks, with strong data efficiency (16% of data to reach full perf) and generalization across causal/masked LMs ([@omarsar0](https://twitter.com/omarsar0/status/1981360482813710384)).
- **Provider variance is a first-class risk**: Cline’s analysis shows silent provider-side changes (quantization, tool-call formatting) can flip outcomes from “works” to “fails,” eroding trust in open-source models. Their mitigation: a 57% system prompt trim (56,499 → 24,111 chars), strict provider filtering (e.g., OpenRouter’s :exacto), and workflow enforcement. Recommendation: require transparent reporting of quantization/impl differences, and test providers as part of model evals ([@cline](https://twitter.com/cline/status/1981420111815987494), [@canvrno](https://twitter.com/canvrno/status/1981403534471119330)).

**Research highlights**

- **Instruction-following during reasoning**: Together’s ReasonIF benchmark finds large reasoning models often violate user constraints mid-chain-of-thought (multilingual formatting, length control), stressing the need for instruction-fidelity checks during generation ([@togethercompute](https://twitter.com/togethercompute/status/1981441935303975059)).
- **Pretraining “coverage profile” over cross-entropy**: A new preprint argues success comes from coverage metrics—what distributions the model internalizes—rather than loss alone ([@canondetortugas](https://twitter.com/canondetortugas/status/1981481591177105740)).
- **Invertibility/injectivity of LMs**: A paper claims provable injectivity/invertibility of model mappings (inputs → reps) across large empirical tests, suggesting lossless representation properties with implications for interpretability ([@HuggingPapers](https://twitter.com/HuggingPapers/status/1981452722495787286)).
- **Optimization dynamics**: New work on muP-based weight decay scaling (independent scaling for hyperparameter transfer) plus empirical commentary on early vs late-phase effects ([paper](https://twitter.com/tonysilveti/status/1981406663086391588), discussion by [@giffmana](https://twitter.com/giffmana/status/1981483376604565969)). Also: explorations of representational flow via TunedLens ([@neuranna](https://twitter.com/neuranna/status/1981357907170959799)) and notes on linear attention precision from practice ([@francoisfleuret](https://twitter.com/francoisfleuret/status/1981487811489317175)).

---

**Top tweets (by engagement)**

- [Anthropic plans ~1M TPUs and >1 GW capacity in 2026](https://twitter.com/AnthropicAI/status/1981460118354219180) (3.4k+)
- [OpenAI: Shared Projects in ChatGPT](https://twitter.com/OpenAI/status/1981432799212249119) (3.1k+)
- [Yann LeCun: “You can’t prove turbojets safe before building them; same for AI.”](https://twitter.com/ylecun/status/1981360519442321451) (2.9k+)
- [LTX-2: open-source AI creative engine (4K/50fps, API-first)](https://twitter.com/ltx_model/status/1981346235194683497) (2.5k+)
- [Argil Atom: controllable video with strong consistency](https://twitter.com/BrivaelLp/status/1981343140196778270) (2.3k+)
- [Microsoft: “Clippy is back!!”](https://twitter.com/satyanadella/status/1981466897557196837) (5.7k+)
- [EU “AI factories” vs industry GPU scale – compute comparison](https://twitter.com/levelsio/status/1981351393513615813) (4.2k+)

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. AI Agent Fundamentals Tutorial

- [**I spent months struggling to understand AI agents. Built a from scratch tutorial so you don't have to.**](https://www.reddit.com/r/LocalLLaMA/comments/1oee1ie/i_spent_months_struggling_to_understand_ai_agents/) (Activity: 345): **The Reddit post introduces a comprehensive tutorial for building AI agents from scratch, focusing on fundamental understanding rather than relying on frameworks like LangChain or CrewAI. The tutorial, available on [GitHub](https://github.com/pguso/ai-agents-from-scratch), includes 8 progressive examples using plain JavaScript and local LLMs such as Qwen and Llama. It covers key concepts like system prompts, streaming, token control, function calling, memory systems, and the ReAct pattern, aiming to demystify the underlying mechanics of AI agents for developers who prefer a hands-on approach.** One commenter appreciated the clarity and suggested the tutorial be featured in relevant forums like r/LocalLLaMA. Another shared a similar learning experience, emphasizing the importance of understanding tool use and function calling in AI agents, as illustrated in the Mistral documentation.
    - mobileJay77 discusses a debugging approach for AI agents, referencing Agno Agi and Mistral documentation. They highlight a method where instead of directly asking the LLM a question, you format the query into a JSON with function names and parameters. This structured approach allows parsing the result, executing the function, and then having the LLM convert it into a complete sentence, emphasizing a systematic way to handle tool use in AI agents.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. AI and Job Replacement Concerns

- [**Fair question**](https://www.reddit.com/r/OpenAI/comments/1oe42et/fair_question/) (Activity: 1322): **The image is a meme featuring a tweet from Sen. Bernie Sanders that raises concerns about the potential for AI and robots to replace all jobs, a viewpoint associated with Elon Musk. Sanders questions the implications for workers who may be left without jobs and income, highlighting a significant issue regarding the future of employment in the face of technological advancements. The discussion reflects broader societal concerns about the impact of AI on labor markets and the need for economic systems to adapt to technological changes.** Commenters express skepticism about the complete replacement of jobs by AI and robots, suggesting that if such a scenario were to occur, it would necessitate a fundamental change in economic systems. There is a debate about whether AI will create new jobs or lead to a future where traditional labor is obsolete, requiring new societal structures.
- [**Elon Musk says AI will replace all jobs and make work optional. Do you think that’s a dream or a disaster?**](https://www.reddit.com/r/ChatGPT/comments/1ody5w4/elon_musk_says_ai_will_replace_all_jobs_and_make/) (Activity: 6231): **The image is a tweet from Elon Musk suggesting that AI and robots will replace all jobs, making work optional. This concept implies a future where employment is not necessary for survival, akin to choosing to grow your own vegetables instead of buying them. The idea is that AI could handle mundane tasks, allowing humans to focus on hobbies or creativity. However, this raises concerns about losing purpose, identity, and motivation if traditional jobs disappear.** Commenters express skepticism, questioning if AI will replace high-level positions like CEOs and how the economy would function if people no longer earn money through work. Some dismiss Musk's statement as unrealistic or misleading.
- [**Fair question**](https://www.reddit.com/r/ChatGPT/comments/1oe42v0/fair_question/) (Activity: 790): **The image is a meme featuring a tweet from Sen. Bernie Sanders that highlights concerns about the impact of AI and robotics on employment, echoing sentiments attributed to Elon Musk. Sanders questions the future of workers in a world where AI could potentially replace all jobs, raising issues about income and societal structure. This reflects ongoing debates about the implications of AI on the labor market and the need for potential societal changes to address these challenges.** One comment suggests that a capitalist society cannot function without jobs, implying a need for a massive societal shift where traditional economic structures are replaced. Another comment cynically refers to a dystopian future where humans are seen as surplus, while a third comment suggests redefining 'workers' as simply 'humans' in a post-job world.

### 2. Claude AI Memory Feature Launch

- [**Claude now has memory for Pro and Max plan users**](https://www.reddit.com/r/ClaudeAI/comments/1oe8td4/claude_now_has_memory_for_pro_and_max_plan_users/) (Activity: 617): **Claude has introduced a memory feature for its Pro and Max plan users, allowing the AI to learn and retain users' workflow patterns, including tool usage, key collaborators, and problem-solving preferences. This feature enables ideas to build over time across conversations, with users having control over what is remembered, the ability to edit or reset memory, and the option to toggle memory on or off. The feature is currently available for Max users and will be rolled out to Pro users over the next two weeks. More details can be found on [Anthropic's news page](https://www.anthropic.com/news/memory).** Some users express skepticism about the utility of AI memory, suggesting the need for standardized tests to evaluate its impact on output quality. Others recommend features like an 'ignore memory' flag for individual chats, and some have experienced issues with inaccurate memory entries affecting chat accuracy.
    - A user expressed skepticism about the utility of AI 'Memory', suggesting that it might not improve output quality. They proposed a comparative analysis using standardized tests on accounts with and without memory enabled to evaluate its impact on performance.
    - Another user highlighted a potential feature improvement by suggesting an 'ignore memory' flag for individual chats. This would allow users to toggle memory usage on or off, potentially even mid-conversation, to better manage when past interactions should influence current outputs.
    - A user reported disabling the memory feature after experiencing issues where Claude referenced incorrect memory entries as factual information, rather than relying on the current chat context. This suggests potential reliability issues with the memory feature, particularly in maintaining accurate and relevant information.
- [**Genie's experimental launch is imminent**](https://www.reddit.com/r/singularity/comments/1oe6twb/genies_experimental_launch_is_imminent/) (Activity: 664): **The image appears to be a promotional or conceptual graphic for a new feature or tool called 'Genie,' which is likely related to interactive or creative processes, possibly involving AI. The text 'Let's start by sketching your world' suggests a focus on user-driven content creation, potentially allowing users to describe environments and characters in a text-based format. This aligns with the comment speculation that the feature might initially support 'text to world' capabilities, indicating a possible AI-driven world-building tool. The mention of 'Genie's experimental launch' implies that this is an upcoming or beta feature, possibly linked to Google's AI initiatives, as inferred from the comments referencing 'gemini 3 and genie.'** Commenters express excitement about the potential of 'Genie,' with some hoping for future capabilities like image uploads to bring their worlds to life. There is also a humorous anticipation of AI-generated content surpassing existing entertainment, as seen in the comment about AI GTA 6.

### 3. OpenAI Controversy and Legal Issues

- [**OpenAI going full Evil Corp**](https://www.reddit.com/r/OpenAI/comments/1oe48qe/openai_going_full_evil_corp/) (Activity: 3168): **The image highlights a controversial legal request by OpenAI in a lawsuit involving the family of Adam Raine, a teenager who died by suicide after using ChatGPT. OpenAI's request for documents related to Raine's memorial, including attendee lists and eulogies, has been criticized as 'intentional harassment' by the family's lawyer. This request is likely part of the discovery process in a wrongful death lawsuit, where OpenAI may be seeking to verify interactions Raine had with ChatGPT, potentially to corroborate chat logs with personal testimonies from the memorial.** Commenters note that while the request may seem intrusive, it is a standard part of legal discovery in a lawsuit. Some suggest that OpenAI is trying to verify the context of Raine's interactions with ChatGPT, especially since he had jailbroken the system.
- [**ChatGPT saved my Moms life.**](https://www.reddit.com/r/ChatGPT/comments/1odxuy6/chatgpt_saved_my_moms_life/) (Activity: 1547): **The post describes how ChatGPT was used to identify serious medical conditions, prompting immediate medical intervention. In one case, it correctly suggested an infection that required emergency care, and in another, it identified a blood clot. These instances highlight the potential of AI tools in providing preliminary medical advice, although they should not replace professional medical consultation. The post emphasizes the importance of using AI as a supplementary tool for urgent health assessments.** Comments reflect a mix of personal anecdotes where ChatGPT provided critical health insights, leading to life-saving interventions. Users shared experiences where ChatGPT's suggestions prompted them to seek further medical advice, which was later validated by healthcare professionals, underscoring the tool's potential utility in emergency scenarios.
    - ChuCHuPALX shared a case where ChatGPT was used to identify a misdiagnosis of anxiety in their father, who had a major stroke. By inputting symptoms and medications into ChatGPT, they discovered the error and moved him to another hospital where he received appropriate care. This highlights the potential of AI tools in cross-verifying medical treatments and advocating for patient care.
    - touchofmal described how ChatGPT helped identify the cause of involuntary tremors as a side effect of high dosage anti-psychotic drugs, potentially leading to tardive dyskinesia. This prompted a consultation with another psychiatrist, confirming ChatGPT's suggestion. This case underscores the utility of AI in recognizing medication side effects and prompting further medical consultation.
    - Single-Intention-535 recounted an experience where ChatGPT advised immediate hospital admission for sepsis symptoms, which were initially dismissed by a doctor. The AI's recommendation led to a timely intervention, preventing a potentially life-threatening situation. This illustrates the role of AI in providing critical second opinions in urgent medical scenarios.

---

# AI Discord Recap

> A summary of Summaries of Summaries by X.ai Grok-4
> 

**Theme 1. AI Models Buzz with Rumors and Releases**

- **Gemini 3 Teases Imminent Launch**: Users speculate on **Google's Gemini 3** release after spotting Easter eggs on the Gemini site, with debates on readiness due to hallucination issues. Some predict a 2024 rollout, while others doubt it amid resource demands.
- **Sora Evolves with Cameos and Edits**: OpenAI's **Sora** adds [character cameos](https://video.twimg.com/amplify_video/1981118365541310464/vid/avc1/896x512/hoABFvQ_q78wMp_h.mp4) starting with animals and toys, plus basic editing tools and Android support. Community excitement focuses on trending UI and social sharing for enhanced video generation.
- **LTX-2 Lights Up Local Video Magic**: Lightricks drops [open-source LTX-2](https://xcancel.com/ltx_model/status/1981346235194683497) for synced **4K/50 fps** videos with audio, running 10-15 seconds on consumer GPUs. Weights arrive later this year, sparking buzz for pro-grade local AI tools.

**Theme 2. Hardware Hustles and GPU Grips**

- **Anthropic Grabs Gigawatt TPUs**: Anthropic secures ~**1 million Google Cloud TPUs** and >**1 GW** capacity by 2026 in a tens-of-billions deal, per [CNBC report](https://www.cnbc.com/2025/10/23/anthropic-google-cloud-deal-tpu.html). Speculation swirls on lower API costs and extended context windows from this massive compute boost.
- **Mojo Tackles GPU Kernels**: Modular's [Mojo workloads](https://arxiv.org/abs/2509.21039) deliver MLIR-based HPC kernels on **NVIDIA H100** and **AMD MI300A** GPUs, matching vendor baselines in four scientific tasks. GitHub repo at [Mojo-workloads](https://github.com/tdehoff/Mojo-workloads) fuels portable performance talks.
- **Cloud Rentals Trump Local Buys**: Experts advise renting cloud GPUs for dozens of hours before local hardware, estimating breakeven at half a year's rental costs including power and management pains.

**Theme 3. Tools Tangle with Bugs and Boosts**

- **Unsloth QAT Scores NVIDIA Nod**: Unsloth announces [NVIDIA support](https://x.com/NVIDIAAIDev/status/1981510959048106262) for **QAT** release, sparking fine-tuning chats on **Qwen Next 80B**. Dependency fixes include pip sequencing for **transformers 4.57.1** to dodge numpy conflicts.
- **DSPy Dodges Async Pitfalls**: Users debug **DSPy async ReAct** modules running synchronously, suggesting `await program.acall` fixes amid gripes on confusing docs. Token tracking via [LiteLLM snippets](https://discord.com/channels/1161519468141355160/1161519469319946286/1431029715511232512) calculates costs from program.history.
- **Cursor Auto-Routing Riles Coders**: Cursor's model roulette frustrates with opaque switches between LLMs, prompting **/summarize** hacks to cut context and costs. Background agents lag with half-day bootups, blamed on runaway dev processes.

**Theme 4. Research Rumbles on Scaling and Myths**

- **Scaling Hits Diminishing Returns**: Paper on [pretraining limits](https://arxiv.org/abs/2506.06266) questions if scaling caps out with RL and test-time compute, but interfaces like **ChatGPT** unlock zeitgeists per debates. Critics argue unscaled HCI research, citing [Minecraft Voyager](https://x.com/DrJimFan/status/1662117799974809603) as pioneer.
- **MythWorx Boasts ARC-AGI Win**: MythWorx claims **100% ARC-AGI** in 4 hours without pre-training during **$100M** fundraise, but lacks validation from organizers. Skeptics draw Theranos parallels, with [Greg Kamradt](https://latent.space/) offering official tests.
- **Meta-Learning Trumps Atari Rules**: [Disco RL paper](https://www.nature.com/articles/s41586-025-09761-x) uses agent experiences to meta-learn rules beating Atari benchmarks, codebase at [GitHub disco_rl](https://github.com/google-deepmind/disco_rl). Sparks talks on cumulative RL evolution.

**Theme 5. Community Gripes on Pricing and Perks**

- **Perplexity Slashes Referral Rewards**: Users fume over **Perplexity AI** referrals dropping from **$20** to **$5** in the USA, with some denied Pro access entirely. Free Pro hacks via **PayPal** and **Airtel** surface amid debates on speed versus **GPT-4**.
- **OpenRouter Fees Fuel Frustrations**: Top-ups hit with **80-cent** service fees, turning **$5.83** payments into **$5** credits, plus **DeepSeek v3.2** outages from depleted balances. Exacto models spark pricing rows, though **glm-4.6** rates hold steady.
- **Hugging Face Spaces Charge Credits**: Spaces like [Sora-2](https://huggingface.co/spaces/akhaliq/sora-2) now bill user credits, shifting monetization. Updated [RAG bases](https://huggingface.co/kalle07/embedder_collection) aid retrieval systems amid gripes on **6GB VRAM** tweaks for **1.1B** models.



---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Referral Program Rewards Reduced**: Users are reporting confusion and disappointment regarding the **Perplexity AI referral program**, with some only receiving **$5** instead of the advertised **$20** reward.
   - One user noted that the reduced reward may be specific to users in the USA, while others have reported not receiving **Pro** access at all.
- **Perplexity Debated Against GPT Models**: Users are actively debating the relative strengths of **Perplexity AI** compared to **GPT models** such as **GPT-3.5** and **GPT-4**, with comparisons of both paid and free versions.
   - One user asserted that *Perplexity is 10x better and faster than chatgpt*, while another jokingly suggested acquiring *super grok*.
- **Free Perplexity Pro Access Explored**: Members are sharing methods to obtain **Perplexity Pro** subscriptions for free, including promotions via **PayPal** and subscriptions through **Airtel**.
   - Additionally, some users are seeking guidance on using included perks and requesting refunds for their subscriptions.
- **Startup Idea: Automated Content Generation for Gooning**: A user has proposed a startup focused on automating content generation specifically for *gooning*, suggesting it could generate *ni hao fine shyts*.
   - The proposed startup aims to produce both *male and female* content, with a particular emphasis on Asian preferences, sparking further discussion among members.
- **Seeking the Safest Family Vehicle**: A member inquired about the [safest family vehicle](https://www.perplexity.ai/search/what-s-the-safest-family-vehic-D4glhRttT3WnhFuyInsfvQ#0) using Perplexity AI.
   - No additional details or context were provided regarding the specific requirements or intended use of the vehicle.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Google Gemini 3 release rumored**: Members are speculating about a potential **Gemini 3** release this year after discovering clues and Easter eggs on the Google Gemini website.
   - Some users think its launch could be imminent, while others believe it may not be ready for release due to issues like hallucination.
- **LithiumFlow Model Mysteriously Removed**: **LithiumFlow**, a popular model in the arenas, appears to have been removed, leading to speculation about its potential integration into AI Studio or a forthcoming release.
   - Some users expressed disappointment, while others noted it could be prompt-sensitive and produce inconsistent results.
- **Code Arena's HTML Upgrade Sparks Debate**: The Code Arena received updates featuring a new prompt, and file creations are now in **.HTML** format.
   - Members suggest refining or removing the explicit tailwind CSS part from the system prompt, as it may not always be necessary and could lead to issues where models create UI-heavy versions of prompts.
- **Sora 2 Pro Videos Debunked**: A user posted videos claimed to be generated by **Sora 2 Pro**, sparking a debate about their authenticity.
   - Other members quickly debunked this claim, suggesting they are likely from a movie or use Minecraft with RTX mod instead.
- **NimbleBean Becomes Top Video Generator**: Users are buzzing about the new **NimbleBean** video model, also known as **Kling 2.5 Turbo Standard**, now the top video generator in the arena.
   - One user stated that it *produces a perfect run* and that many wouldn't be able to tell if it's AI or not, noting their key ingredient is the *labelling*.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor's Auto Model Routing Feels Like Roulette**: Users express frustration with **Cursor's auto-routing** to different models, describing it as a *roulette* that inconsistently picks between smart and dumb LLMs depending on prompt and **context**.
   - Users note that the lack of transparency makes it difficult to determine which models are being routed to, with some suggesting **cluster benchmark performance** to estimate performance.
- **Cursor's Billing Triggers Usage Anxiety**: Users report **Cursor's new plan** pushes them to use the **Auto model**, warning when premium model usage decreases days left, but one user stated they will *use premium models to the max as long as I can use premium models unlimited with a delay*.
   - Members note the **Auto model** uses **Claude 4.5 Sonnet Thinking** and is cheaper, so the suggestion is to use **/summarize** often to lower the context window and reduce costs.
- **Background Agents Suffer Half-Day Bootups**: Users complain about long **Background Agent** boot times, reporting waits up to *half a day* for git cloning, repo scanning, and initial linting.
   - Members humorously speculate a developer might have accidentally left a **Background Agent** running indefinitely, hinting at inefficient resource management.
- **Parallel Agent Paralyzed by Usability Issues**: A user struggles to find a viable use case for the **parallel agent**, while another encounters issues enabling MCP to run `cursor-agent` in a container, citing *✗ MCP login failed*.
   - The error indicates the **MCP server "atlassian"** requires approval before loading, suggesting a potential configuration or permission issue.
- **UI Error Message Causes Debugging Headaches**: A user suggests clarifying a **UI error message** that caused confusion when a local branch wasn't pushed to the origin, by providing a more explicit message like *"The branch {branch name} does not exist in the remote repository."*
   - The user initially suspected an issue with the *environment.json* file, highlighting the importance of clear error messaging for efficient debugging.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **NVIDIA Navigates into Unsloth QAT**: Unsloth [announced NVIDIA's support](https://x.com/NVIDIAAIDev/status/1981510959048106262) for their **QAT release**, indicating growing industry interest in quantization-aware training.
   - The announcement sparked discussions around fine-tuning **Qwen** next 80b, although some responses were job promotion attempts.
- **Unsloth Users Unravel Dependency Disasters**: Users installing **Unsloth** reported dependency conflicts, particularly with `transformers` and `numpy` versions, causing installation failures.
   - Workarounds included specific installation sequences (`pip install unsloth` then `pip install transformers==4.57.1`) and using Docker to bypass version incompatibilities.
- **Karpathy Kudos to Key Token Kickstarts**: Karpathy confessed [on X](https://x.com/karpathy/status/1980397031542989305) about almost not using **tmux**.
   - He highlighted that **good token initialization** really matters, especially without a long teaching phase.
- **QAT to GGUF Conversion Coming Soon**: A team member confirmed that converting **QAT (Quantization Aware Training)** models to **GGUF** format isn't currently supported but is planned for the end of the year.
   - The new feature will allow low-powered devices to leverage **QAT GGUF** models.
- **Docker's Llama.cpp Compilation Crash Course**: A user reported that **llama-cpp compilation** failed within the official Unsloth Docker image due to a missing CUDA compiler **nvcc**.
   - Troubleshooting suggestions included checking the Unsloth version, CUDA setup, ensuring the NVIDIA container toolkit is correctly installed, and running the container with the `--gpus all` flag.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Deepseek OCR faces scrutiny**: Members discussed that the brand new [**Deepseek OCR**](https://example.com/deepseek-ocr) requires scrutiny before deployment.
   - Members suggested testing and reviewing thoroughly before relying on its performance.
- **Zero3 config goes boom!**: A member reported exploding **GPU RAM** when training a **1b Gemma 3** model with max seq size of 8k using **LoRA** with r of 8, flash attention, deepspeed zero 3 and bf16 precision with batch size of 1.
   - Another member suggested that their **Zero3 config** might be busted, implying a configuration error.
- **Tiny LLM gets roasted**: One member shared a [GitHub repo](https://github.com/ker2x/MytinyLLM) for their **Tiny LLM** built from scratch, designed as a learning exercise.
   - Another member dismissed the contribution, stating that a **400k parameter model** is just a toy and can't learn anything meaningful out of the box.
- **Hugging Face Spaces Wants Your Credits Now**: A member noted that some **Hugging Face Spaces**, including those for **Sora** and **Veo 3.1**, now require payment to use *your* credits, see [Sora space](https://huggingface.co/spaces/akhaliq/sora-2).
   - They linked to an example of a **Sora space** with a warning indicating that usage now costs money, noting a shift in the platform's monetization strategy.
- **RAG Base Updated on Hugging Face**: A member shared an updated **RAG** base for everyone who works with **RAG** at [huggingface.co/kalle07/embedder_collection](https://huggingface.co/kalle07/embedder_collection) and [huggingface.co/Pacific-Prime/pacific-prime](https://huggingface.co/Pacific-Prime/pacific-prime).
   - The update aims to provide a better starting point for building **Retrieval-Augmented Generation** systems.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT Extension Sees All**: OpenAI launched a [**ChatGPT Chrome extension**](https://video.twimg.com/amplify_video/1981054579648368640/vid/avc1/1280x720/YmADR8pumfl1Zwin.mp4) enabling **ChatGPT** to view the current page and provide instant, relevant answers without switching tabs.
   - This integration aims to streamline information access and enhance user experience by contextualizing **ChatGPT's** responses.
- **Sora to Premiere Character Cameos**: **Sora** is set to introduce [character cameos](https://video.twimg.com/amplify_video/1981118365541310464/vid/avc1/896x512/hoABFvQ_q78wMp_h.mp4), starting with animals and toys, allowing users to feature familiar characters in generated videos.
   - Upcoming features include a trending cameos UI, basic video editing tools, social sharing options, feed quality improvements, and an **Android** version of **Sora**.
- **Calculators Get Chatty with GPT**: A **Wired** article and [YouTube video](https://www.youtube.com/watch?v=olcZdTRdnQg) highlighted the integration of **GPT on a TI-84 calculator** using an **ESP32**, opening doors for AI-assisted problem-solving.
   - The community humorously debated whether using such a setup to cheat on math tests deserves credit for technical ingenuity.
- **Unofficial ChatGPT Extension Fixes Bugs**: A user shared a link to a [new unofficial ChatGPT extension](https://www.reddit.com/r/ChatGPT/) called **ChatGPT LightSession**, reporting that it fixed the *red text errors* and sped up their work.
   - Another user warned to *be careful with this extension*, recommending to stick with publicly accessible content to keep your private information safe.
- **AI Voice Cloning Goes DIY**: A member shared a [DIY voice AI project](https://www.instructables.com/Create-Your-Own-AI-Voice-Agent-Using-EchoKit-ESP32/) that allows users to clone voices using **EchoKit ESP32**.
   - This open-source project facilitates training AI on custom voices for various applications, including video generation, with ongoing discussions about its potential uses and ethical implications.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Gemini Fails at Structured Outputs in DSPy**: A user reported that **Gemini models** with the `responses` parameter in **DSPy** are throwing warnings and errors when used with structured output adapters, specifically a *"Stub file not found"* error.
   - Ongoing discussions focus on enabling structured outputs for **Gemini models** to ensure type safety in Python using **DSPy**.
- **Refine Loophole: Premature Termination**: A user discovered that `dspy.Refine` does not necessarily run for all N iterations, but stops early if the `reward_fn` exceeds the set `threshold`, contrary to initial assumptions from the [documentation](https://dspy.ai/tutorials/output-refinement/best-of-n-and-refine/?h=refine#refine).
   - The user provided a [demo](https://github.com/stanfordnlp/dspy/blob/main/dspy/predict/refine.py#L142) illustrating that the refine loop breaks if the `reward_fn` surpasses a threshold, noting that `Refine` performs a deep copy of the module each time.
- **DSPy's Async ReAct: Not So Async?**: A user is facing issues with running two `ReAct` modules asynchronously using `dspy.asyncify`, noting that they appear to be executing synchronously, and sought guidance on proper implementation.
   - A member suggested using `await program.acall` instead of `await program.aforward` as well as implementing `async def aforward(...)` within the module, noting **DSPy's** confusing documentation.
- **DSPy Tracks Tokens and Costs**: Users are discussing how to track token usage and costs in **DSPy**, with one member sharing a [code snippet](https://discord.com/channels/1161519468141355160/1161519469319946286/1431029715511232512) to calculate costs using `program.history` and **LiteLLM** model pricing.
   - They clarified that the `if "usage"` condition ensures cost calculation is based on actual model usage, accounting for caching and the member noted that `usage` in `program.history` is similar to `result.get_lm_usage()`.
- **Customization Causes Conniptions**: A user expressed frustration with **DSPy's** complexity, especially regarding accessing `ReAct` module outputs, suggesting that it is easier to implement LLM calls in a custom loop for better control and UI integration.
   - Users explored alternative approaches, like subclassing and reimplementing the `aforward` method in a `ReAct` module as shown in [this code snippet](https://cdn.discordapp.com/attachments/1431066467897577492/1431067652763156666/message.txt?ex=68fc111c&is=68fabf9c&hm=956afd7b5a72fb6ea1288e1d2656952b4e64d31baf63e1feccda13f151437ba5&).



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo ❤️ Rust: C ABI Bridges the Gap**: Mojo can communicate over the **C ABI** with Rust, C++, Zig, or any other language that speaks **C ABI**, similar to how a host C/C++/Rust process embeds a python interpreter.
   - However, using the **C ABI** requires too much manual intervention, motivating the search for solutions that make packages available like they are native to the platform.
- **Python Power: Mojo's Runtime Secrets 🤫**: Mojo interacts with **CPython** similarly to how a host C/C++/Rust process embeds a python interpreter, allowing it to leverage the Python ecosystem; the goal is to have **zero runtime overhead** if possible.
   - This is especially important when proving out things for making hot loop code go faster.
- **Type System Talk: Mojo Flexes its Power 💪**: Mojo's type system is powerful enough that Dependent Haskell is probably the most popular language that can express everything Mojo can, while a language with a more powerful type system can consume code from a language with a less powerful type system.
   - To make calling Mojo functions feel native from another language, the other language needs a type system more powerful than or equally as powerful as Mojo's, potentially including dependent and linear types.
- **Interop Insights: A Dynamism vs. Statism Duel ⚖️**: Using a compiled language provides both a measure of dynamism and the ability for the Mojo code to get type system guarantees back from your code, which massively cuts down on the type introspection.
   - If a host language is static and compiled, a new language implemented on top of it should also be static and compiled to minimize overhead, as static and compiled languages can be compiled to emit dynamic code if the dynamic language is more expressive.
- **Mojo Misses Out on Pipeline Operator (for Now)**: A member noted that Mojo lacks a pipeline character or function composition like `|>` or `>>` in other languages, and since Mojo isn't stable, the team likely won't consider it soon, but pointed to [this github issue](https://github.com/modular/modular/issues/213) as a related feature request.
   - Another member noted that Python lacks it.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **DeepSeek Experiences Credit Crunch**: The **DeepSeek/deepseek-v3.2-exp** model faced availability issues on OpenRouter due to depleted credits, resulting in a **402 Proxy Error** for users.
   - Some users initially mistook the error as account-specific, highlighting confusion around error messaging.
- **Top-Up Timing Troubles**: Users reported delays in OpenRouter credit top-ups and the depletion of **DeepSeek** provider balances.
   - One user wryly commented that resolution would occur *"until Toven reads it."*
- **OpenRouter Top-Up Fees Exposed**: Users scrutinized service fees for adding credits to OpenRouter, noting that a $5.83 USD payment yielded only $5 in credits.
   - The additional $0.83 covers service fees, approximately **80c (US)**.
- **Exacto Models Spark Pricing Debate**: The implementation of **exacto** models as separate API variants raised concerns about potential pricing increases and fragmented token/app statistics.
   - Despite worries about higher costs, **glm-4.6** prices remain consistent for **exacto** and **non-exacto** endpoints; however, a member argued that this approach *splits token and app stats*.
- **OpenRouter API Hit With Access Problems**: Users encountered login issues and **401 unauthorized errors** on the OpenRouter completions endpoint.
   - The problem may be tool-specific, with cURL requests succeeding while other tools failed.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Linear Shifts to Self-Driving SaaS**: Linear is pivoting from reactive chatbot UIs to **proactive AI**, termed *"self-driving SaaS"*, to automatically advance tasks.
   - Commenters lauded the bold strategy, drawing parallels to **Level-5 autonomy**, where software handles tasks autonomously.
- **Airbnb Applauds Alibaba’s Qwen**: **Airbnb’s CEO** highlighted **Alibaba’s Qwen models** for being faster and more economical than **OpenAI’s latest**, showcasing **Cerebras’ 2,500-tokens/sec inference**.
   - Discussions revolved around the diminishing significance of marginal accuracy gains compared to cost reduction, with security, sovereignty, and hardware evolutions emerging as pivotal considerations.
- **MythWorx's ARC-AGI Claim Faces Scrutiny**: Organizers from the **ARC Prize** have not confirmed **MythWorx’s** claim of achieving **100%** on **ARC-AGI**, which was announced during a **$100M** fundraise.
   - **Greg Kamradt** offered to conduct an official test if MythWorx submits accurately, yet skepticism prevails due to lacking validation and parallels with **Theranos-like publicity**.
- **Anthropic Secures Massive TPU Capacity**: Anthropic revealed a deal to incorporate **~1 million Google Cloud TPUs** and **>1 GW** capacity by **2026**, generating excitement about computational capabilities.
   - Reactions involved speculation on whether this will reduce API costs, increase usage, or extend context windows; the deal is valued in the tens of billions and will bring over a gigawatt of AI compute online in 2026 ([CNBC Article](https://www.cnbc.com/2025/10/23/anthropic-google-cloud-deal-tpu.html), [Anthropic Announcement](https://www.anthropic.com/news/expanding-our-use-of-google-cloud-tpus-and-services)).
- **Lightricks Lights Up Local Latent LTX-2**: **Lightricks** released **LTX-2**, an open-source creative engine that creates [synchronized **4K/50 fps** video with audio](https://xcancel.com/ltx_model/status/1981346235194683497) for ~10–15 s on consumer GPUs.
   - The community is energized by prospects for democratizing pro-grade AI video tools, with weights expected later this year and an API Playground currently live.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Mojos HPC Kernels Hit GPUs**: [Mojo](https://arxiv.org/abs/2509.21039) and [Mojo Workloads](https://github.com/tdehoff/Mojo-workloads) is now being used for **MLIR-Based Performance-Portable HPC Science Kernels on GPUs**.
   - The paper targets four scientific workloads comparing their performance against vendor baselines on **NVIDIA H100** and **AMD MI300A GPUs**.
- **SMEM Descriptor Decoded for WGMMA**: A user shared a custom implementation of shared memory (**SMEM**) descriptors for use with **WGMMA** instructions, including `matrix_descriptor_encode` and `make_smem_descriptor` functions and is derived from the [PTX guide](https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-descriptor-format).
   - Another member suggested looking into DeepSeek-AI's DeepGEMM [implementation](https://github.com/deepseek-ai/DeepGEMM/blob/c9f8b34dcdacc20aa746b786f983492c51072870/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh#L275) for handling similar issues with memory fences and WGMMA.
- **Rent from the Cloud before you commit**: A member suggests to *rent from the cloud* until you've spent at least a few dozen hours doing serious work to evaluate the costs and benefits.
   - They estimate that the *breakeven point* for investing in local hardware is *at least half a year of cloud rental time*.
- **HQQ+ Moves to Dropbox**: The **MobiusML** domain is migrating to **Dropbox** today, causing the original links to break, where users should now replace `mobiusml` with `dropbox` for both blog posts and GitHub repositories.
   - A member was looking for a working link to the **HQQ+ blog post** after noticing that [https://mobiusml.github.io/1bit_blog/](https://mobiusml.github.io/1bit_blog/) was down.
- **Torch/XLA Takes a New Turn, Still Relevant**: Despite a *new direction* with [torch/xla](https://github.com/pytorch/xla/issues/9684), the **Lazy Tensor Paper** remains relevant for picograd in shoehorning an eager mode into tinygrad to lower the ramp.
   - The [Lazy Tensor Paper](https://arxiv.org/pdf/2102.13267) explores integrating an eager mode into tinygrad using picograd.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Discord Introduces 'Pending' Member Status**: Discord introduced a 'pending' status for new members joining the server, activated after the *'join'* option was added to the server tag.
   - Members confirmed being 'pending' before moderator approval; speculation arose about whether this manual approval process is still in place.
- **AI Network like Internet Proposed**: A member proposed an **AI network** similar to the internet, where thousands of models communicate via a protocol to answer questions.
   - Another member requested concrete examples of such a system's implementation and potential architectures.
- **New Model Claims Incredible Performance**: A member claimed their **50M** parameter model achieved a **0.223** loss, while their **1B** parameter model has a validation loss of **~0.197**, resulting in a perplexity of **~1.22**, attaching an [image](https://cdn.discordapp.com/attachments/747850033994662000/1431083629748027402/image.png?ex=68fc1ffd&is=68face7d&hm=c347fd3fa1d4dae87e30579f0723253fd5c83a7197c57f6583d66e7d2ba5ca67&).
   - Other members expressed skepticism, suggesting that the model has a bug or that the author was seeking attention rather than help and suggested they debug via inference.
- **Meta-Learning RL Discovers Novel Rules**: A [paper](https://www.nature.com/articles/s41586-025-09761-x) explored **meta-learning** from cumulative experiences of agents across environments, yielding a rule that surpasses existing ones on the **Atari benchmark**.
   - The codebase for this project is available on [GitHub](https://github.com/google-deepmind/disco_rl).
- **Causal Abstraction Craze Calms**: Members inquired about the lack of recent discussions on **Causal Abstractions** (see [paper](https://arxiv.org/abs/2301.04709)), a popular topic in 2023.
   - It was noted that the framework is still useful but suffers from implicit linearity assumptions ([paper](https://arxiv.org/abs/2507.08802)), difficulty in choosing the right level of abstraction ([survey](https://arxiv.org/abs/2408.01416)), and multiple valid abstractions for a single behavior ([paper](https://arxiv.org/abs/2502.20914)).



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Django Splits into Microservices**: A member is considering splitting a Django app that uses **bi-encoders, cross-encoders, YOLO models**, into a separate microservice to offload inference and deal with potential request queuing with limited VRAM (32GB).
   - In response, another member suggested scheduling requests sequentially or running them in parallel, encouraging them to leverage parallel RPC calls to maximize the GPU's capabilities.
- **Meta's Weight Hardness Hack**: Inspired by real neuron connections, a member linked [Meta's implementation](https://x.com/realJessyLin/status/1980662516285075762) to use an additional scalar on each weight to store its *hardness* or *softness*.
   - The concept involves adjusting weights based on the *fruit* of the computation, strengthening beneficial connections and weakening detrimental ones, effectively learning without modifying irreversible connections.
- **MythWox Mysteriously Masters ARC-AGI**: A member shared a link to [MythWorx.ai](https://mythworx.ai/capabilities/), claiming **100%** on **ARC-AGI 1** with no pre-training within 4 hours.
   - The member expressed skepticism, questioning the legitimacy of the capabilities, with another reacting with confused disbelief.
- **Jsonnet Juggling for Functional Configs**: Members debated whether [Jsonnet](https://jsonnet.org/) is overengineered, while another appreciated its tailorability and usage by **DeepSeek** for configurations.
   - One member likened it to **Nix**, defining a hash table using functional programming, questioning its readability for JSON. Another offered to discuss **Jsonnet**/**VinePPO** and then provided links to [ICML](https://icml.cc/virtual/2025/poster/45526), [OpenReview](https://openreview.net/forum?id=5mJrGtXVwz), [ArXiv](https://arxiv.org/abs/2410.01679), and [GitHub](https://github.com/McGill-NLP/VinePPO) related to it.
- **Brave Browses 'Unseeable' Injections**: A member shared a [Brave Browser blog post](https://brave.com/blog/unseeable-prompt-injections/) discussing **unseeable prompt injections**.
   - No further discussion was added.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Models do Jamaican accent**: A user reported that the **4o model** still does a **Jamaican accent** in voice mode, while **model 5** claims to do it but doesn't change the voice at all.
   - The user specified that this is *one of the few metrics I care about*.
- **Debate of Open Sourcing with Bugs**: A member is considering open-sourcing their software but is debating whether to delay the release to fix bugs or ship it with annoying garbage.
   - Their app is designed to let anyone train AI models locally without SaaS, and they plan to include a **Weeb Sam finetune** consisting of **40k images**.
- **ChatGPT Gets Seahorse Scare**: A user shared a link to a [tweet](https://fxtwitter.com/ltx_model/status/1981346235194683497) about a **ChatGPT bug** that occurs when sending the prompt *is there an emoji of a seahorse?*.
   - One commenter speculated that it looks like an **Unreal Engine/Runway/Wan competitor** in the making.
- **New Paper Tests Scaling Limit**: A member shared a [paper](https://arxiv.org/abs/2506.06266) and questioned if **scaling** has hit its limit, considering diminishing returns from **pretraining**, **test-time compute**, and **RL** scaling.
   - Other members disagreed, saying no, *RL, test time compute, etc are only one component of scaling AI*.
- **Interface Plays Key Role in Scaling**: A member argued that the **interface**, **workflows**, and **infra** are extremely important components of scaling AI, as the **ChatGPT interface** unlocked a research zeitgeist.
   - They pointed out that *Claude is calling their new prompt system 'Skills' but really this was pioneered (and coined) by Minecraft Voyager years ago* and cited [Jim Fan's tweet](https://x.com/DrJimFan/status/1662117799974809603).



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **AlphaXiv Maps AI Paper Origins**: A member shared [AlphaXiv](https://www.alphaxiv.org/labs/geo-trends), a tool that tracks the geographic origins of **AI research papers**.
   - Another member was timed out for advertising, with repeated violations.
- **Kimi K2 Cheaper with Questionable Chutes**: A member asked about **Chutes** compared to **Moonshot AI** for **Kimi K2**, citing the quality and data policy.
   - Another member claimed that **Chutes** trains on user data without a data policy and has less reliable uptime; its tool call accuracy is half that of the official API, memed on [Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1da7j6g/did_you_know_you_can_ban_chutes_openrouter_go_to/).
- **Home Server Versions Controlled via Git?**: A member inquired about using **Git** to manage home server versions to aid with resetting after breakages.
   - The community ignored this user's concerns.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Hybrid Architecture Debated**: A user proposed a *hybrid architecture* using local compute and storage to avoid wasting available resources.
   - The suggestion involved building large native apps, processing datasets locally, and running resource-intensive AI models on local hardware.
- **Pro Plan Users Allege Bait and Switch**: A user voiced discontent over the **Pro plan** now featuring credit limits, claiming it was initially advertised as *unlimited*.
   - The user, feeling it unfair for the paid month, inquired about others wanting to test the pro features.
- **Manus Launches Logic Arenas App**: A user noted that Manus created a full-stack **AI Arenas app** ([image](https://cdn.discordapp.com/attachments/1349440650495398020/1430878322073796628/2025-10-23_12-15.jpg?ex=68fc0988&is=68fab808&hm=efef49879866f78ee43ea3c281eca345ab2bcf800110c561cc4e81bc723f5219&)) (*not yet tested*) within an hour using minimal credits.
   - The **Logic Arenas** app comprises *Heart vs Mind*, *Dual Parents*, and *Moral Court* arenas and utilizes **Kotlin**, **Ktor**, **Redis**, **Jetpack Compose**, and **Material 3**, amounting to ~7k LOC across 28 files.
- **Manus Code Blasted for Deprecated Modules**: A user observed that **Manus** could be improved by using non-deprecated code, modules, and plugins, though this requires a lot of work.
   - Despite **Manus** touting a beautiful Material 3 dark theme ([image1](https://cdn.discordapp.com/attachments/1349440650495398020/1430917287363350728/2025-10-23_14-51.jpg?ex=68fc2dd2&is=68fadc52&hm=4e68706e60e030a96d99d157b3abdacc9d2b99077639e85f545cb96a0620d614&), [image2](https://cdn.discordapp.com/attachments/1349440650495398020/1430917287707279391/2025-10-23_14-47.jpg?ex=68fc2dd2&is=68fadc52&hm=04f22a7c26e41abcb491ebfcbdcb2c4ec8b05ad504eeb504de7aa4c2bc80f3f4&)), the initial Android app design seemed outdated; however, Claude Code improved it with one prompt ([image](https://cdn.discordapp.com/attachments/1349440650495398020/1430971270219956234/2025-10-23_18-26_1.jpg?ex=68fbb758&is=68fa65d8&hm=d73323d08753d88efdd123cd9be451622fbb989859f6077c67b76d45598ebfcd&)).
- **Manus Botches Homework**: A user claimed that **Manus** failed to correctly solve a homework assignment, despite the user providing similar examples.
   - The user mentioned that two notebooks were supplied but the generated PDF was disorganized, underscoring **Manus'** inability to complete this task.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider API Keys Cheaper?**: A user inquired about using **aider** with API keys to match **gemini-cli's** cost-effectiveness, using `/tokens` to manage keys, clear history, and watch context size.
   - They noted **gemini-cli's** plan at around **$20 USD** monthly offers nominally **1500** requests daily with a **1M token** context window, outperforming **aider's** interface with file system operations like `grep` but lacking a repo map.
- **Playwright struggles on Fedora**: A user asked if anyone got **Playwright** working on **Fedora**.
   - No workarounds or solutions were provided.
- **Community eyes Aider's Future**: Members want to know the future direction of **Aider** and are waiting to hear from its creator, **Paul Gauthier**.
   - The community looks forward to upcoming developments and strategic decisions for the project.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **tinygrad Slides into pytorchconf!**: A **tinygrad** [slide](https://cdn.discordapp.com/attachments/1068976834928193609/1430758414744686655/HDjLLqQ.png?ex=68fc429c&is=68faf11c&hm=13bea9c27e9b130accbe594e495dbd0e6813cb720aaabb5bb6f2375495a385b5&) appeared at **pytorchconf**, showcasing **tinygrad**'s significance in the PyTorch ecosystem.
   - This highlights **tinygrad**'s growing adoption and relevance in the broader machine learning community.
- **tinygrad Boosts Dev Team**: A member inquired about documentation for becoming a **tinygrad** developer, signaling interest in contributing to the project.
   - This suggests potential growth and expansion of the **tinygrad** contributor base.
- **JITBEAM Focuses Jitted Kernels**: A member noted that **JITBEAM** only beams the *jitted kernels*.
   - Further context on the specifics of "beaming" and the implications for **tinygrad**'s performance were not provided.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Block Compares MCP Tools to Ogres**: A member shared a [blog post from Block](https://engineering.block.xyz/blog/build-mcp-tools-like-ogres-with-layers) comparing building **MCP tools** to building ogres *with layers*.
   - The comparison alludes to the layered approach for building robust and complex systems.
- **MCP Website Requests Meeting Schedules**: A reminder was posted for **working group leads** to post their upcoming meeting times on the [Model Context Protocol website](https://meet.modelcontextprotocol.io/).
   - Meeting organizers can clone a previous event from the *Past Events* section to speed up the scheduling process.



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





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1430677331239899186)** (1422 messages🔥🔥🔥): 

> `Referral Program, GPT models, Free Pro Accounts, Gooning startup` 


- **Referral Rewards Cause Confusion**: Users are expressing confusion about the referral program, with some reporting they only received **$5** instead of the advertised **$20** and some not getting Pro at all.
   - One user stated that it used to be $20 before, but *now they are only giving $5 from USA*.
- **Users Debate Perplexity vs GPT**: Users are debating the capabilities of **GPT models** (3.5 vs 4) and discussing the differences between paid and free versions.
   - One user pointed out that *Perplexity is 10x better and faster then chatgpt*, while another joked they *should have bought super grok*.
- **Pro Perks & Free Accounts**: Users discuss ways to get **Perplexity Pro** for free, like through a **PayPal** promotion or an **Airtel** subscription.
   - There are also some users requesting refunds and asking how to use perks.
- **Enthusiasts Announce Gooning Startup**: A user proposed a **gooning machine startup** that automates content generation, *which will show ni hao fine shyts for gooning*.
   - Members discussed the startup which aims to create both *male and female* content with a focus on Asian preferences.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1430984118543450173)** (2 messages): 

> `safest family vehicle, computational evidence` 


- **Members seek safest family vehicle**: A member asked Perplexity about the [safest family vehicle](https://www.perplexity.ai/search/what-s-the-safest-family-vehic-D4glhRttT3WnhFuyInsfvQ#0).
   - No additional context or discussion was provided.
- **Evidence found computationally**: A member linked to a Perplexity page about [computational evidence](https://www.perplexity.ai/page/computational-evidence-for-rec-MZ.AjbR6SlGMwJpoCK7cCA).
   - No additional context or discussion was provided.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1430681312255148144)** (1333 messages🔥🔥🔥): 

> `Gemini 3 Release, LithiumFlow Removal, Code Arena System Prompt, Sora 2 Pro, NimbleBean Kling 2.5` 


- **Google Gemini 3 rumored to release soon!**: Members are speculating about a potential **Gemini 3** release this year after discovering clues and Easter eggs on the Google Gemini website, although Logan hasn't tweeted yet.
   - Some users think its launch could be imminent, while others believe it may not be ready for release due to issues like hallucination and the need for extensive resources to run the world simulator.
- **LithiumFlow vanishes from Arena**: **LithiumFlow**, a popular model in the arenas, appears to have been removed, leading to speculation about its potential integration into AI Studio or a forthcoming release.
   - Some users expressed disappointment, while others noted it could be prompt-sensitive and produce inconsistent results.
- **Code Arena revamps with new prompt & .HTML**: The Code Arena got new updates with new prompt and file creations being in **.HTML**.
   - Members suggest refining or removing the explicit tailwind CSS part from the system prompt in the new Code Arena, as it may not always be necessary and could lead to issues where models create UI-heavy versions of prompts.
- **Debate sparks over New Sora 2 Pro videos**: A user posted some videos and claimed them to be generated by **Sora 2 Pro**, sparking a debate about their authenticity.
   - However, other members quickly debunked this claim and noted that these are more likely from some movie or use Minecraft with RTX mod.
- **NimbleBean now the best video generator**: Users discussed about the new **NimbleBean** video model on the arena, AKA **Kling 2.5 Turbo Standard** model, which is now the top video generator.
   - One user stated that it *produces a perfect run* and that many wouldn't be able to tell if it's AI or not, noting their key ingredient is the *labelling*.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1430677736669843496)** (411 messages🔥🔥🔥): 

> `Cursor Auto Model Routing, Cursor Usage and Billing, Version Control in Cursor, Background Agents gone wild, GCP deployment` 


- **Auto Model Roulette?**: Users discuss tricks to avoid auto-routing to less capable models, noting it feels like a *roulette* whether a smart or dumb model is chosen, with suspicion that the choice depends on prompt and **context**.
   - Members suggest that it's hard to know which LLMs are being routed to and that the only option is to *cluster benchmark performance* to estimate the performance.
- **Usage and Billing gotchas**: Cursor's new plan is trying to force users to use the Auto model, and warn you if you use it too much, decreasing the days left for usage, forcing you to use the terrible Auto model, but one member says that *as long as I can use premium models unlimited with a delay I am gonna keep using em to the max*.
   - Members warn that **Auto model** uses **Claude 4.5 Sonnet Thinking** and it charges much less, so it would be a great way to save money, but if that is not an option members suggest using **/summarize** every 2 prompts to lower the context.
- **Background Agents Cooked!**: Users complain that background agents can take *half a day* to boot up, git clone the repo, scan the entire repo, make changes, lint, etc etc etc.
   - Members joke that *most likely one of the devs used background agents and forgot about it*.
- **Parallel Agent, more like Paralyzed Agent**: A member reports struggling to find a use case for the parallel agent, and another has issues enabling MCP to run `cursor-agent` in a container.
   - When running `cursor-agent mcp login atlassian` they get: *✗ MCP login failed. Failed to load MCP 'atlassian': MCP server "atlassian" has not been approved. Please approve it before loading.*
- **When Gemini 3.0?**: With Gemini 3.0 release in the horizon, it may devalue Cursor
   - Some members speculate that *Gemini won’t beat Sonnet 4.5 thinking*


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1430708078936064150)** (5 messages): 

> `Debugging Cursor's UI Errors, Cursor PRs in review mode, Background Agent API Key` 


- **UI Error Leads to Head-Scratching**: A user suggested updating a **UI error message** that caused confusion when a local branch wasn't pushed to the origin, suggesting a more explicit message like *"The branch {branch name} does not exist in the remote repository."*
   - The user spent time debugging because they thought *environment.json* wasn't checked in, and a coffee would have helped.
- **Cursor's Remote Branch Logic Questioned**: A user found the error message clear, noting that the **remote branch** lacking an *environment.json* file implies local changes must be merged.
   - The user stated *"Remote branch does not have an environment.json ... won't work unless you merge your local changes into it"*.
- **Ready-to-Review PRs Sought**: A user inquired if it's possible to configure **Cursor** to open **Pull Requests** directly in *ready-to-review mode* instead of draft mode.
   - No answers were provided.
- **Background Agent API Key Origin**: A user asked about the source of the **API key** used for **Background Agent status reports**.
   - No answers were provided.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1430725021923410010)** (264 messages🔥🔥): 

> `UnslothTrainer parameters, LoRA in Unsloth, QAT model conversion, GRPO Training, Unsloth VLLM` 


- **Tweaking UnslothTrainer For Overfitting**: A user asked if increasing `max_grad_norm` would improve overfitting with [UnslothTrainer](https://github.com/unslothai/unsloth), seeking to memorize training data and remove watermarks, with initial attempts showing loss decreasing from **2.0 to 0.2**.
   - Suggestions included setting `weight_decay` to 0, using higher learning rates (**5e-4**), and adjusting other parameters to encourage memorization, such as reducing or eliminating warmup and using a constant learning rate schedule.
- **Applying LoRA to lm_head in Unsloth**: A user inquired about applying **LoRA** to the `lm_head` layer in **Unsloth**, noting that their initial attempts resulted in the layer being wrapped as a `ModulesToSaveWrapper` instead of injecting a **LoRA** module.
   - The user sought clarification on whether **lm_head LoRA** is supported in the latest **Unsloth** version and if there's a recommended approach for fine-tuning newly added tokens via **LoRA**.
- **Converting QAT Models to GGUF**: A user inquired about converting **QAT (Quantization Aware Training)** models to **GGUF** format for use on low-power devices.
   - A team member responded that **QAT to GGUF** conversion isn't currently supported but is planned for the end of the year, highlighting the potential benefits of running **QAT GGUF** models on low-power devices.
- **Dependency Issues Plague Unsloth Installations**: Multiple users reported encountering dependency conflicts when installing **Unsloth**, specifically with `transformers` and `numpy` versions.
   - Suggested workarounds included specific installation sequences (`pip install unsloth` then `pip install transformers==4.57.1`) and alternative installation methods like Docker due to version incompatibilities.
- **NVIDIA Backs Unsloth QAT**: Unsloth [announced NVIDIA's support](https://x.com/NVIDIAAIDev/status/1981510959048106262) for their **QAT release**.
   - The tweet was met with job promotion attempts and follow up questions regarding finetuning qwen next 80b.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1430928966801424404)** (6 messages): 

> `New member introductions, LLM Training Interest` 


- **New Members Join Community**: Several new members introduced themselves, with interests ranging from training **tiny models** to experimenting with existing ones.
   - One member admitted to joining because they see **Unsloth AI** *"everywhere"*.
- **LLM Training Sparks Interest**: A new member named Daryl with **no prior experience** in training LLMs expressed interest in learning from the community.
   - He stated that it's *"something i'm interested in and would love to learn around like-minded people"*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1430696763047804949)** (54 messages🔥): 

> `Karpathy No Tmux, Voice generated LLM prosody tokens, Humanoid Robot Temperature, GPU Credits from Alts` 


- **Karpathy Confesses No Tmux Temptation**: Karpathy confessed [on X](https://x.com/karpathy/status/1980397031542989305) about nearly falling for the temptation of not using **tmux** again.
   - He emphasized that **good token initialization** really matters, especially without a long teaching phase.
- **LLM's Voice Prosody Token Tricks**: Members discussed **finetuning an LLM** with specific **prosody tokens** before voice generation to control the tone.
   - For example, using tokens like `<p3> Hello! </p3>` to denote different levels of emphasis and tone in the generated speech.
- **Roboticists Debate Humanoid's Heat Management**: A member asked if maintaining a constant temperature of **36.6°C** in a humanoid robot would hurt internal hardware, considering CPUs/GPUs often run around **60°C**.
   - Another member responded that electronics are generally much more **temperature-tolerant** than biological systems, humorously noting that meatbags start failing at 42.5°C.
- **GPU Credit Alt Account Antics**: One user joked about making **alt accounts** to get extra **GPU credits**.
   - Another user responded with a [Chris Pratt mind-blown GIF](https://tenor.com/view/chris-pratt-mind-blown-parks-and-rec-surprised-shookt-gif-9771169), likening it to MMO's 101.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1430678328297263176)** (72 messages🔥🔥): 

> `Qwen3 finetuning issues, LoRA on lm_head layer, Gemma 4b model training instability, llama-cpp compilation failure in Docker, Fine-tuning Qwen/Qwen3-VL-30B-A3B-Instruct` 


- **German Dialect Qwen3 Model Stumbles**: A user reported difficulties fine-tuning a **Qwen3-4B-Instruct-2507** model to translate German into a specific dialect, facing issues with the model either losing its German capabilities or getting stuck in repetitive loops.
   - Reducing the **learning rate** helped mitigate the language corruption, but the model still struggled to learn the dialect, indicating potential data or training configuration issues.
- **LoRA on lm_head Layer Leaves User Lost**: A user experimenting with applying **LoRA** to the **lm_head** layer in Unsloth found that it only wrapped the layer instead of injecting LoRA.
   - Another member suggested that **lm_head LoRA** is unnecessary.
- **Gemma 4b Model Grows Glitchy**: A user encountered instability in their **Gemma 4b** model after attempting to fix code to avoid repetition during SFT training with Unsloth.
   - Despite resetting the kernel and starting with a fresh model and dataset, the model continued to produce implausible answers, suggesting potential persistent interference or data contamination.
- **Docker Llama.cpp Derails**: A user reported that **llama-cpp compilation** failed within the official Unsloth Docker image due to a missing CUDA compiler **nvcc**.
   - Troubleshooting suggestions included checking the Unsloth version, CUDA setup, and ensuring the NVIDIA container toolkit is correctly installed and the container is run with the `--gpus all` flag.
- **Vision Model Finetuning Frustration**: A user encountered an error when trying to fine-tune **Qwen/Qwen3-VL-30B-A3B-Instruct** using the Unsloth notebook for Vision models, specifically during the trainer.train() step.
   - It was suggested that disabling compilation might resolve the issue by setting `os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"`.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1431050839975526471)** (2 messages): 

> `AGI Definition, Cattel-Horn-Carroll test` 


- **AGI Definition Paper Posted**: A member shared a link to the [AGI Definition paper](https://www.agidefinition.ai/paper.pdf).
- **Cattel-Horn-Carroll test on LLMs**: A member mentioned that it was *pretty smart* to apply the **Cattel-Horn-Carroll test** to LLMs.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1430682469912416407)** (287 messages🔥🔥): 

> `OCR Models, DeepSpeed Zero 3 Configuration, Tiny LLM Training, Hugging Face Spaces Pricing, AI Prompt Standardization` 


- **Deepseek OCR is shiny and new**: Members discussed that [**Deepseek OCR**](https://example.com/deepseek-ocr) is brand new and suggested to go through its reviews before using.
- **Zero3 config might be busted!**: A member was experiencing exploding **GPU RAM** when training a **1b Gemma 3** model with max seq size of 8k using **LoRA** with r of 8, flash attention, deepspeed zero 3 and bf16 precision with batch size of 1.
   - Another member suggested that their **Zero3 config** might be busted.
- **Tiny LLM for Great Justice**: One member shared a [GitHub repo](https://github.com/ker2x/MytinyLLM) for their **Tiny LLM** built from scratch, designed as a learning exercise.
   - Another member pointed out that a **400k parameter model** is just a toy and can't learn anything meaningful out of the box.
- **Hugging Face Spaces now cost money?!**: A member noted that some **Hugging Face Spaces**, including those for **Sora** and **Veo 3.1**, now require payment to use, using *your* credits.
   - They linked to an example of a [Sora space](https://huggingface.co/spaces/akhaliq/sora-2) with a warning indicating that usage now costs money.
- **Standardizing Prompts Like Pseudocode**: A member wondered if standardizing **AI prompts** into a **pseudocode** that's more efficient than JSON could improve AI performance.
   - Another member shared [GitHub repositories](https://huggingface.co/datasets/John6666/forum2/blob/main/ai_prompt_standardization_1.md) with similar purposes.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 messages): 

waffles1: Ah yes this is totally legit
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1430893934212550756)** (2 messages): 

> `RAG, raw-text-searcher, embedder_collection, pacific-prime, 6GB VRAM optimization` 


- **RAG base gets an Update**: A member shared an updated, good base for everyone who works with **RAG**, found at [huggingface.co/kalle07/embedder_collection](https://huggingface.co/kalle07/embedder_collection) and [huggingface.co/Pacific-Prime/pacific-prime](https://huggingface.co/Pacific-Prime/pacific-prime).
- **Raw Text Searcher for Buzzwords Announced**: A member is working on a smart **raw-text-searcher** for buzzwords including **AND/OR, wildcard** to cut out snippets around matches like embedders do but not similarity and asked who likes testing?
- **6GB VRAM optimized for 1.1B Model**: A member reported a **10% gain** pushing the limit for **6GB VRAM**, now able to run a **1.1B** model from scratch with a **10% win**.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/)** (1 messages): 

its_nmt05: I used vannilla clip just switched to general prompts
  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1431018915139555471)** (3 messages): 

> `NanoChat Course, Karpathy Educational Material, nanochat-students org` 


- **Members ponder NanoChat Course viability**: A member asked if a **Hugging Face course on NanoChat** would be useful.
   - Another member expressed confusion about the goal of it, as well as noting the absence of a new channel in the Discord server.
- **Community Shares Models on the Hub**: A member shared that they started a community on the Hub so that people could share their models, and they are working on porting the architecture to transformers so people can use them more broadly.
   - The same member also mentioned that they think **Karpathy** is going to release more educational material, so it makes sense to follow that.
- **Clarifying the Goal of nanochat-students org**: A member inquired about whether another member meant the Karpathy server or HF, and the goal of **NanoChat** or the **nanochat-students org** on the Hub.
   - Another member said they believed that most people are GPU poor, so using **8xH100** is overkill.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1430763200608993481)** (5 messages): 

> `HuggingFace Learn Agents Course, Quiz Errors, Qwen Model Issues` 


- **HuggingFace Agents Course throws Quiz Errors**: Users report errors in the final quiz section of the [HuggingFace Learn Agents Course](https://huggingface.co/learn/agents-course/unit2/smolagents/final_quiz).
   - The error message indicates a **404 Client Error: Not Found** when generating feedback, specifically with the **Qwen/Qwen2.5-Coder-32B-Instruct** model.
- **Unit 4 Questions Lead to 404 Error**: One user reports getting a **404 error** when trying to download **Unit 4 questions** in the HuggingFace Learn Agents Course.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1430860721717248143)** (2 messages): 

> `ChatGPT Chrome Extension, Sora Roadmap Updates, Sora Character Cameos, Sora Video Editing, Sora Social Experience` 


- **ChatGPT Extension gives Instant Answers**: OpenAI released a [ChatGPT Chrome extension](https://video.twimg.com/amplify_video/1981054579648368640/vid/avc1/1280x720/YmADR8pumfl1Zwin.mp4) that allows **ChatGPT** to see the page you're on to give instant, accurate answers without tab-switching.
- **Sora Roadmap Update: Cameos, Editing, Social, Android**: **Sora** is adding [character cameos](https://video.twimg.com/amplify_video/1981118365541310464/vid/avc1/896x512/hoABFvQ_q78wMp_h.mp4), starting with dogs, guinea pigs, and favorite stuffed toys, and cameos of generated characters from **Sora** videos.
   - They are updating the generation UI to show the latest trending cameos in real time, adding basic video editing capabilities like stitching multiple clips, exploring new ways to use **Sora** with friends, improving the feed quality and moderation, and working on the **Android** version of **Sora**.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1430679139626647572)** (216 messages🔥🔥): 

> `Sora access, AI Education Software, Meta Glasses, AI Assisted calculators, Voice AI cloning` 


- **Users Inquire on Sora Generation**: Members discussed how to access **Sora AI** for video generation, mentioning the need for **text prompts** and suggesting tools like **Comet** or **Atlas** to assist with the process.
- **Discussion on AI Assisted Education Software**: A member mentioned working on **AI-assisted university education software**, while others discussed the potential pitfalls and successes of such ventures, including a $5M investment by **LAUSD** in an AI project during COVID.
- **Meta Glasses Spark Interest Despite Hurdles**: Some members expressed interest in the **new Meta glasses**, noting the difficulty in purchasing them due to the appointment requirement, with one member also mentioning the illegality of concealed recording devices in their country.
   - The discussion touched on the legality of recording devices, with references to **two-party consent laws** and the potential for fines and imprisonment for illegal concealed recording in places like **Australia**.
- **AI Powers TI-84 Calculator**: A member linked a **Wired** article and a [YouTube video](https://www.youtube.com/watch?v=olcZdTRdnQg) showcasing **GPT running on a TI-84 calculator** via an **ESP32**, suggesting that hacking a calculator with AI to cheat on math should warrant credit and a high grade.
   - The article was [ChatGPT on a TI-84 Graphing Calculator Is a Cheating Device](https://www.wired.com/story/chatgpt-on-a-ti-84-graphing-calculator-cheating-device/)
- **Voice Cloning DIY Project**: A member shared a [DIY voice AI project](https://www.instructables.com/Create-Your-Own-AI-Voice-Agent-Using-EchoKit-ESP32/) that allows users to clone voices using **EchoKit ESP32**, with discussions about training AI on custom voices for video generation.
   - The project is open source.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1430713328703115425)** (26 messages🔥): 

> `ChatGPT LightSession Extension, Muon as Adam replacement, OpenAI Support Issues, 503 Error on OpenAI` 


- **ChatGPT LightSession Extension Shared**: A user shared a [new unofficial ChatGPT extension](https://www.reddit.com/r/ChatGPT/) called **ChatGPT LightSession**, noting it is still in beta and accessible via a shared link.
   - Another user warned to *be careful with this extension*, recommending to stick with publicly accessible content to keep your private information safe.
- **LightSession Extension fixes red text errors**: A user reported that the **LightSession extension** fixed the *red text errors* and sped up their work.
   - They said that the best configuration statement they have found for chat is *ask clarifying questions*.
- **Muon versus Adam Optimizers**: A user asked about **Muon** as a potential replacement for the **Adam** optimizer.
   - No one responded to the question.
- **OpenAI support MIA, user faces persistent 503 error**: A user has been unable to send or receive any text messages due to a consistent **503 error** for the past 16 days, and has received no response from OpenAI support.
   - The user, a **ChatGPT Plus subscriber**, has tried multiple troubleshooting steps and is seeking advice on how to reach a real person at OpenAI support.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1430738072815009852)** (14 messages🔥): 

> `Creating CCTV footage from Sora, Good prompt engineering source for ChatGPT, LLM's performance and prompting differences, Gemini's translation vocabulary issues, Creating video of ball bouncing and falling` 


- **Creating CCTV footage with Sora**: A member asked for help on how to create **CCTV footage videos** from **Sora**.
   - No solutions were given.
- **Seeking a Prime Prompting Primer**: A member inquired about a good source for **prompt engineering** for **ChatGPT** and whether each **LLM** has its own performance characteristics and prompting methods.
   - No solutions were given.
- **Gemini's Grammar Goofs**: A member reported that **Gemini's** translations of video scripts into unfamiliar languages resulted in out-of-context vocabulary and seeks advice on prompting **ChatGPT** for better results.
   - No solutions were given.
- **Crafting Catchy Compositions for Custom Creations**: A member shared a *mad semi reusable prompt* distilled from a custom GPT lyricist/sound engineer/album visualiser, for making music artists usable in music generation AIs as a custom instruction/prompt drop.
   - The member attached a [text file](https://cdn.discordapp.com/attachments/1046317269069864970/1431083530695217233/drop_in_as_is_knowledge_initial_instruct.txt?ex=68fc1fe5&is=68face65&hm=5f027ddb6661f87ccf71a3d202e90d685e1ec64cb814fa2c5da0a6791c3725cc&) intended as knowledge to be *rehydrated* prior to use.
- **Pinpointing Precise Prompting Parameters**: A member asked how to find the information needed to build a precise prompt, particularly for areas outside their expertise, such as lighting and photography.
   - Another member suggested showing the example image to **ChatGPT** and asking it to make a nearly identical image and clearly describing the image, focusing on the areas of interest.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1430738072815009852)** (14 messages🔥): 

> `Sora for CCTV, Prompt engineering sources, LLM translation, Image prompting, GPTs for prompts` 


- **Sora's CCTV Debut Falls Flat**: A member inquired about using **Sora** to create **CCTV footage videos**, but received no immediate responses or solutions.
   - It remains to be seen how well Sora can simulate the grainy, low-fidelity aesthetic of surveillance cameras.
- **Prompt Engineers Hunt the Holy Grail**: A user requested recommendations for a good source for **prompt engineering for ChatGPT**, questioning whether each **LLM** has unique performance characteristics that require tailored prompts.
   - The community didn't immediately offer resources, highlighting the ongoing quest for effective prompt engineering strategies.
- **Gemini's Linguistic Stumble**: A member shared their experience using **Gemini** to **translate video scripts**, noting that the vocabulary sometimes lacked context despite Google's translation expertise.
   - They sought advice on improving prompts for **ChatGPT** to achieve better translation accuracy.
- **Image Analysis Inspires Precise Prompts**: A member sought advice on creating precise prompts for image generation, asking *how does a person atually go about finding the info to build an actually percise prompt?*.
   - The member attached an example image, seeking guidance on how to translate visual elements into detailed prompt instructions.
- **GPTs Unleash Prompting Power**: A member suggested developing personal **GPTs** to tackle specific prompt requests, arguing that specialized GPTs can hone in on specifics better than generalized ones.
   - This strategy aims to leverage the specialization of custom GPTs to produce targeted and refined results in prompt generation.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1430874121679405068)** (214 messages🔥🔥): 

> `Gemini and structured outputs, DSPy Refine logic, Async ReAct modules, DSPy token usage and cost tracking, DSPy customization pain points` 


- **Gemini can't handle structured outputs in DSPy**: A user reported that Gemini models with `responses` parameter in DSPy are throwing warnings and errors when used with structured output adapters, specifically experiencing a "Stub file not found" error.
   - There is ongoing discussion on enabling structured outputs for **Gemini models** to ensure type safety in Python using **DSPy**.
- **Refine stops before N is reached**: A user discovered that `dspy.Refine` does not necessarily run for all N iterations, but stops early if the `reward_fn` exceeds the set `threshold`, contrary to initial assumptions from the [documentation](https://dspy.ai/tutorials/output_refinement/best-of-n-and-refine/?h=refine#refine).
   - The user provided a [demo](https://github.com/stanfordnlp/dspy/blob/main/dspy/predict/refine.py#L142) illustrating that the refine loop breaks if the `reward_fn` surpasses a threshold, and suggested that `Refine` performs a deep copy of the module each time, which prevents an accurate count of iterations.
- **DSPy's async ReAct runs in sync**: A user is facing issues with running two `ReAct` modules asynchronously using `dspy.asyncify`, noting that they appear to be executing synchronously, and sought guidance on proper implementation.
   - A member suggested using `await program.acall` instead of `await program.aforward` as well as implementing `async def aforward(...)` within the module, while noting DSPy's confusing documentation.
- **DSPy's token usage tracks costs**: Users are discussing how to track token usage and costs in DSPy, with one member sharing a [code snippet](https://discord.com/channels/1161519468141355160/1161519469319946286/1431029715511232512) to calculate costs using `program.history` and **LiteLLM** model pricing.
   - They clarified that the `if "usage"` condition ensures cost calculation is based on actual model usage, accounting for caching and the member noted that `usage` in `program.history` is similar to `result.get_lm_usage()`.
- **Users struggle with DSPy customization**: A user expressed frustration with DSPy's complexity, especially regarding accessing `ReAct` module outputs, suggesting that it is easier to implement LLM calls in a custom loop for better control and UI integration.
   - Users explored alternative approaches, like subclassing and reimplementing the `aforward` method in a `ReAct` module as shown in [this code snippet](https://cdn.discordapp.com/attachments/1431066467897577492/1431067652763156666/message.txt?ex=68fc111c&is=68fabf9c&hm=956afd7b5a72fb6ea1288e1d2656952b4e64d31baf63e1feccda13f151437ba5&).


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1430752336048816139)** (181 messages🔥🔥): 

> `Mojo and Rust interoperability, CPython runtime interaction with Mojo, PyO3 and JuliaCall usage with Mojo, Extensible traits in Mojo, Effect systems and logical programming in Mojo` 


- ****Mojo** ❤️ **Rust**: C ABI as Common Ground**: Mojo, being a systems language, can communicate over the **C ABI** with Rust, C++, Zig, or any other language that speaks **C ABI**, similar to how a host C/C++/Rust process embeds a python interpreter.
   - However, using the **C ABI** requires too much manual intervention, motivating the search for solutions that make packages available like they are native to the platform.
- ****Python Power**: Mojo's Runtime Secrets 🤫**: Mojo interacts with **CPython** similarly to how a host C/C++/Rust process embeds a python interpreter, allowing it to leverage the Python ecosystem, however this requires work on the part of the runtime and the ability to express concepts from the host language.
   - The goal is to have **zero runtime overhead** if possible, especially when proving out things for making hot loop code go faster.
- ****PyO3** and **JuliaCall**: The Interoperability Dream Team ✨**: **PyO3** and **JuliaCall** can be used from a language built on Mojo to access Rust and Julia ecosystems, as they create Python modules, but this approach may introduce performance penalties due to round-tripping through Python.
   - It was suggested to just have the **Julia lib export a C ABI**, instead of roundtripping through python, however this has to be done for every single project individually, and its less safe as type safety is lost.
- ****Type System Talk**: Mojo's Power Play 💪**: Mojo's type system is powerful enough that Dependent Haskell is probably the most popular language that can express everything Mojo can, while a language with a more powerful type system can consume code from a language with a less powerful type system.
   - To make calling Mojo functions feel native from another language, the other language needs a type system more powerful than or equally as powerful as Mojo's, potentially including dependent and linear types.
- ****Interop Insights**: Dynamism vs. Statism ⚖️**: Using a compiled language means both a measure of dynamism and the ability for the Mojo code to get type system guarantees back from your code, which massively cuts down on the type introspection.
   - If a host language is static and compiled, a new language implemented on top of it should also be static and compiled to minimize overhead, as static and compiled languages can be compiled to emit dynamic code if the dynamic language is more expressive.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1430827604964933672)** (8 messages🔥): 

> `Mojo pipelines, LeetGPU, GPU Puzzles` 


- **Mojo Lacks Pipeline Operator (for Now)**: A member inquired about Mojo having a pipeline character or function composition like `|>` or `>>` in other languages.
   - Another member responded that Python lacks it, and since Mojo isn't stable, the team likely won't consider it soon, but pointed to [this github issue](https://github.com/modular/modular/issues/213) as a related feature request.
- **LeetGPU supports Mojo**: A member shared a link to [LeetGPU](https://leetgpu.com/), noting its support for Mojo.
   - A Modular employee responded saying that Modular found it important to develop training that aligns with the Modular platform, and suggested [GPU Puzzles](https://puzzles.modular.com/introduction.html) as an alternative.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/)** (1 messages): 

brockelmore: https://github.com/modular/modular/issues/5496
  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1430682766122680442)** (141 messages🔥🔥): 

> `Claude Haiku 4.5 web search, OpenRouter Models in Cursor, Exacto Evaluation, NovitaAI and GPT-OSS-120B, DeepSeek v3.2 exp Moment` 


- **DeepSeek has a Moment due to Balance**: **DeepSeek/deepseek-v3.2-exp** experienced issues due to OpenRouter running out of credits, leading to a **402 Proxy Error**.
   - Users mistook the error for account-specific problems, while others noted that the error was directed at the users, implying something was wrong with their account.
- **Top-up Timing Troubles Triggered**: Users reported delays in topping up OpenRouter credits and the **DeepSeek** provider running out of balance.
   - One user humorously noted that the delay lasted *until Toven reads it*.
- **OpenRouter Top-Up Fee Facts**: Users discussed the service fees associated with adding credits to OpenRouter, where a $5.83 USD payment resulted in only $5 in credits.
   - It was clarified that the additional $0.83 covers service fees, which is approximately **80c (US)**.
- **Exacto Model Variants: A Costly Conundrum?**: Users debated the implementation of **exacto** models as separate variants in the API, with concerns about pricing and token/app stats splitting.
   - Despite concerns that OpenRouter will charge more, examination of **glm-4.6** showed prices are the same for **exacto** and **non-exacto** endpoints, but one member argued that implementing them as model variants rather than an extra parameter *splits the token and app stats for these models*.
- **OpenRouter API Access Problems**: Users reported issues logging into OpenRouter and receiving a **401 unauthorized error** on the completions endpoint.
   - It was suggested that the problem might be tool-specific, with cURL requests working while other tools failed.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - New Models**
  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1430702602978463794)** (6 messages): 

> `Model statistics visibility, OpenAI data preservation, Krea AI's video model, OpenRouter Docs` 


- **Model Stats Visibility Plea**: Members requested that model statistics be publicly displayed, or indicated with a badge to show **"exacto quality"**.
   - Staff confirmed that this feature *"is coming"*.
- **OpenAI no longer has to preserve all ChatGPT data**: A member shared an [Engadget article](https://www.engadget.com/ai/openai-no-longer-has-to-preserve-all-of-its-chatgpt-data-with-some-exceptions-192422093.html) stating that **OpenAI** no longer has to preserve all of its **ChatGPT** data, with some exceptions.
- **Krea AI's 14B video model is here**: A member shared a link to [Krea AI's tweet](https://x.com/krea_ai/status/1980358158376988747) about their **14B video model**.
- **OpenRouter Docs Status?**: A member reported that **OpenRouter docs** were still down.
   - Staff replied that they were *"looking into it"*.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1430680098100609114)** (115 messages🔥🔥): 

> `OpenClip TS Implementation, Linear's Self-Driving SaaS, Alibaba’s Qwen vs OpenAI, MythWorx’s ARC-AGI Claim, SOTA RL Algorithms` 


- **TypeScript OpenClip Implementation Discovered!**: A member was looking for a TypeScript implementation of **OpenClip** and found their own solution at [Marqo's fashionCLIP](https://huggingface.co/Marqo/marqo-fashionCLIP).
   - They shared code snippets demonstrating how to use the `CLIPTextModelWithProjection`, `CLIPVisionModelWithProjection`, `AutoTokenizer`, and `AutoProcessor` from `@huggingface/transformers` to compute similarity scores between text and images.
- **Linear's Vision: Self-Driving SaaS Takes the Wheel!**: Linear is teasing a shift from reactive chatbot UIs to **proactive AI** that automatically moves work forward, calling it *"self-driving SaaS"*.
   - Followers have praised the bold framing and likened it to **Level-5 autonomy**, where software completes work while users sit back.
- **Qwen's Charm Offensive: Airbnb Picks Cheaper LLMs!**: **Airbnb’s CEO** publicly praised **Alibaba’s open-source Qwen models** as faster and far cheaper than **OpenAI’s latest**.
   - Commenters highlight **Cerebras’ 2,500-tokens/sec inference**, multilingual quality, and the race-to-the-bottom on cost over marginal accuracy gains, noting security, sovereignty, and future hardware shifts as looming concerns.
- **ARC-AGI Skepticism: MythWorx's Bold 100% Claim!**: **ARC Prize** organizers state they have not yet validated **MythWorx’s** claim in a **$100M** fundraise press release that its model achieved **100%** on **ARC-AGI**.
   - **Greg Kamradt** offers to run the official test if MythWorx submits properly, while commenters question the claim’s legitimacy due to lack of verification, minimal team visibility, and parallels to **Theranos-level hype**.
- **Anthropic's Compute Conquest: 1M TPUs by 2026!**: Anthropic announced a multi-billion-dollar deal to add **~1 million Google Cloud TPUs** and **>1 GW** of capacity by **2026**.
   - The community reacted with excitement over the compute scale, jokes about ‘empire-building’, and questions about whether this will translate to lower API prices, higher usage limits, or longer context windows; the deal is worth tens of billions of dollars and is expected to bring well over a gigawatt of AI compute capacity online in 2026 ([CNBC Article](https://www.cnbc.com/2025/10/23/anthropic-google-cloud-deal-tpu.html), [Anthropic Announcement](https://www.anthropic.com/news/expanding-our-use-of-google-cloud-tpus-and-services)).


  

---


### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1430727704147136524)** (7 messages): 

> `AMD Strix Halo ROCm, GPT-OSS on ROCm, Local AI app for multi-page scanned PDF QA` 


- ****Lisa Su** Signs **Strix Halo**!**: A user shares a [link](https://xcancel.com/AnushElangovan/status/1981031660209770802) to **Anush Elangovan** boasting a **Lisa-Su**-signed **Strix Halo (Ryzen AI Max+ PRO 395)** laptop.
   - The laptop is running **GPT-OSS** locally on **ROCm**, prompting users to share benchmark results, bug reports, purchase tips and wish-list items for higher memory/next-gen parts.
- **Seeking Local AI for Scanned PDF QA**: A user asks for a local AI app that supports **QA directly on a multi-page scanned PDF** with a **VLM** like **Qwen3-VL-4B**.
   - They note that most apps either only support images or do **RAG** when you upload files, like **LM Studio**.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1430702568006357062)** (15 messages🔥): 

> `Sora Roadmap Update, Teaching Sand to Think - Runtime 2025, LTX-2 Launch, Character AI Ovi` 


- **Peebles Previews Promised Parade of Sora Progress**: Bill Peebles from **OpenAI** outlined upcoming **Sora** app updates, including [custom character cameos](https://xcancel.com/billpeeb/status/1981118483607032050) launching within days, a basic clip-stitching editor, upcoming social channels, and the long-awaited **Android** release.
   - Users requested cameo search, storyboard saving, longer clips, UK/global access, and better moderation feedback.
- **A16Z's Abstract AI Ad Astonishes All**: **a16z** unveiled a teaser for ["Runtime 2025"](https://xcancel.com/a16z/status/1981074692443427321?s=46) featuring the line *"Teaching sand to think, or being left in the dust."*
   - The **A24-style video** sparked enthusiasm and debate across crypto, AI, and hardware communities, with viewers praising its cinematic quality while questioning whether it’s visionary hype or shallow gloss.
- **Lightricks Launches Local Latent LTX-2**: **Lightricks** launched **LTX-2**, an open-source creative engine that generates [synchronized **4K/50 fps** video with audio](https://xcancel.com/ltx_model/status/1981346235194683497) for up to ~10–15 s on consumer GPUs (RTX 5-series realistic when optimized).
   - Community buzz centers on democratizing pro-grade AI video tools and upcoming local installs, with weights dropping later this year and an API Playground live now.
- **Ovi Open-Source Offering Overlooked?**: Character AI open-sourced [**Ovi on Github**](https://github.com/character-ai/Ovi).
   - Some community members wondered why it didn't *"make a splash at all"*.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1430688885809090672)** (21 messages🔥): 

> `FA4's correctness optimization, Deterministic inference blogs, Graphics programming Discord server, Kernel programming DSLs for non-ML workloads, Mojo for HPC` 


- **Numerical Stability Resources Sought**: A member sought resources on numerical stability after reading about **FA4's correctness optimization** and **deterministic inference blogs**, and was pointed to a book on the topic: [Numerical Solution of Stochastic Differential Equations](https://epubs.siam.org/doi/book/10.1137/1.9781611971491).
- **Discussing graphics programming outside GPGPU**: A member inquired about resources for graphics programming, specifically for scientific visualization using **OpenGL** or **Vulkan**, rather than **GPGPU**-focused content.
   - Another member suggested the [Graphics Programming Discord server](https://graphics-programming.org/) with 20k members, known for expertise in both **OpenGL** and **Vulkan**.
- **DSL kernels for non-ML workloads**: A member asked about the application of **kernel programming DSLs** (TileLang, Triton, Gluon, TLX, Helion) for non-ML workloads, proposing extensions for **sparsity**, **stencils**, and **tiled hardware architectures**.
   - Another member referenced [Mojo](https://github.com/tdehoff/Mojo-workloads) work for HPC and [Triton](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=w2kMs8IAAAAJ&citation_for_view=w2kMs8IAAAAJ:u-x6o8ySG0sC) papers.
- **Mojo's HPC kernels on GPUs**: A member shared a link to a paper and GitHub repo about [Mojo](https://arxiv.org/abs/2509.21039) and [Mojo Workloads](https://github.com/tdehoff/Mojo-workloads) being used for **MLIR-Based Performance-Portable HPC Science Kernels on GPUs**.
   - The paper targets four scientific workloads: a seven-point stencil, BabelStream, miniBUDE, and Hartree-Fock, and compares their performance against vendor baselines on **NVIDIA H100** and **AMD MI300A GPUs**.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1430758919269257246)** (18 messages🔥): 

> `wgmma template, smem descriptors, Pinned portable memory, PTX memory consistency model, Compiler issued barriers` 


- **Warpgroup woes with WGMMA**: A user is facing issues with `wgmma` template and compiler-injected barriers, even after using `wgmma.fence.sync.aligned` via the `warpgroup_arrive` function.
   - The user is using the `wgmma` instruction with shared memory descriptors and suspects the problem lies in incorrect fences or memory consistency within PTX; [code snippets are provided](https://cdn.discordapp.com/attachments/1189607726595194971/1430858333014986862/image.png?ex=68fbf6ea&is=68faa56a&hm=c2d221fd9b6ecca17e72af6f77398d4cb88ef8c8049c472ee765ed5250b771e4).
- **SMEM descriptor decoded**: A user provided their custom implementation of shared memory (SMEM) descriptors for use with WGMMA instructions, including `matrix_descriptor_encode` and `make_smem_descriptor` functions.
   - The implementation is derived from the [PTX guide](https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-descriptor-format), encoding the address and memory layout information into a 64-bit descriptor.
- **DeepSeek DeepDive for GEMM**: A community member suggested looking into DeepSeek-AI's DeepGEMM [implementation](https://github.com/deepseek-ai/DeepGEMM/blob/c9f8b34dcdacc20aa746b786f983492c51072870/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh#L275) for handling similar issues with memory fences and WGMMA.
   - Additional links to NVIDIA's CUTLASS [library](https://github.com/NVIDIA/cutlass/blob/b2ca083d2bb96c41d9b3c5a930637c641f6669bf/include/cute/atom/mma_traits_sm90_gmma.hpp#L49), [mma_sm90_gmma.hpp](https://github.com/NVIDIA/cutlass/blob/b2ca083d2bb96c41d9b3c5a930637c641f6669bf/include/cute/arch/mma_sm90_gmma.hpp#L88) and [discussions](https://github.com/NVIDIA/cutlass/discussions/1375) were provided.
- **FP8 precision accumulation?**: A user inquired about the accumulation precision of `m16n8k32 fp8xfp8` operations, wondering if it accumulates in FP32 before converting back to FP16.
   - This question was inspired by this [documentation link](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma) mentioning that accumulation of intermediate values is performed with at least single precision.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1430838297424298005)** (2 messages): 

> `Pruna.ai Hiring, Vectorware Hiring, Rust GPU software` 


- ****Pruna.ai** Hunts Applied ML Engineer**: [Pruna.ai](https://careers.pruna.ai/jobs/6569302-applied-ml-engineer) is hiring an **Applied ML Engineer** with experience optimizing **Diffusion Models**; the position is remote and based in either Paris or Munich.
- ****Vectorware** Ventures into Hiring**: A company building advanced GPU software in **Rust**, [Vectorware](https://www.vectorware.com/jobs), is hiring, where candidates are expected to learn Rust.
   - More details about the company are available in [this blog post](https://www.vectorware.com/blog/announcing-vectorware/).


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1430745347440906274)** (1 messages): 

> `Cloud vs Local, Cost Analysis, Serious Work Consideration` 


- **Cloud Beats Local For a Trial Run**: A member suggests to *rent from the cloud* until you've spent at least a few dozen hours doing serious work to evaluate the costs and benefits.
- **Local Breakeven Requires Half a Year**: They estimate that the *breakeven point* for investing in local hardware is *at least half a year of cloud rental time*.
- **Don't forget Management and Power Costs**: The cost analysis should consider *management pains and power costs* when evaluating the cloud vs local tradeoff.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1431050313552494713)** (2 messages): 

> `HQQ+ blog post, mobiusml github down, dropbox github` 


- **HQQ+ Blog Post Link Puzzle Solved!**: A member was looking for a working link to the **HQQ+ blog post** after noticing that [https://mobiusml.github.io/1bit_blog/](https://mobiusml.github.io/1bit_blog/) was down.
   - Another member pointed out that *`mobiusml` should be replaced with `dropbox` for both the blog post and the GitHub link*, due to a recent change announced today.
- **MobiusML to Dropbox Migration**: The **MobiusML** domain is migrating to **Dropbox** today, causing the original links to break.
   - Users should now replace `mobiusml` with `dropbox` for both blog posts and GitHub repositories.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1430954293346697258)** (4 messages): 

> `Mobius Labs acquisition, Austin Huang news` 


- **Adept Strategy Shift & Co-Founders Joining Amazon**: The co-founders of [Mobius Labs](https://x.com/Mobius_Labs/status/1981391562836721786) share *personal news* that it *was a good ride*.
   - One member congratulated them, hoping they were treated well, and said *you did great work*.
- **Austin Huang Announces Personal News**: [Austin Huang](https://x.com/austinvhuang/status/1981393212003521017) announces *some personal news*, with a link to X.
   - He also shared an image of **salmon on the electric grill**, along with a tomato, cucumber, sea salt, coffee, milk cream, and stevia.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

melnimr: I dm'ed you
  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1430995659586142319)** (4 messages): 

> `Lunar Lake, MAMF Script, roofline, vk_cooperative_matrix_perf, CUDA events` 


- **Lunar Lake Draft Roofline Debuts**: A member shared a draft **roofline** created with data points from [Stas's MAMF script](https://github.com/stas00/ml-engineering/tree/master/compute/accelerator/benchmarks) for **Lunar Lake** under PyTorch 2.9.0+xpu.
   - They noted constant overhead at small sizes, making them unrepresentative of the GPU's capabilities, but highlighting the test's consistency.
- **Measuring Time for short Kernels**: A member suggested ways to improve time measurements for short kernels by running multiple kernels on disjoint inputs/outputs to invalidate the L2 cache.
   - It was also suggested that if you're using a **CUDA event** (or XPU equivalent), you can add a small matmul before `event.record()` to reduce overhead, and they linked to a [Speechmatics article about timing operations in PyTorch](https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch).
- **vk_cooperative_matrix_perf Patched**: The **vk_cooperative_matrix_perf** has been improved with patched.
   - No further details were given.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1430981414530318448)** (6 messages): 

> `Vectorware, GPU in Rust, ML on GPU, GPU rendering` 


- **Vectorware ventures into GPU software in Rust**: A new company called [Vectorware](https://www.vectorware.com/blog/announcing-vectorware/) has launched, focusing on writing advanced software for the GPU in **Rust**.
   - The company is explicitly not launching a Discord server yet to maintain focus, but is initially focusing on **ML applications** and has some **CPU-based use cases running on the GPU**.
- **ML takes center stage on GPUs**: Vectorware is prioritizing **Machine Learning (ML)** applications as their initial focus for **GPU-based software development**.
   - In the coming weeks, Vectorware plans to showcase demos that leverage the GPU for CPU-based use cases, hinting at a broader long-term vision beyond just ML.
- **GPU rendering takes center stage**: One of the members expressed curiosity of apps like **Zed** that heavily rely on the **GPU** for rendering, which Vectorware may tackle.
   - Members speculated that there are other **non-rendering**, **non-ML applications** that would greatly benefit.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1430950569538883696)** (2 messages): 

> `PGL Support for sm_80, Single-GPU kernels for 4090/L4s` 


- **Ampere GPUs phased out for development**: A question was raised about future **PGL support for sm_80** (Ampere architecture).
   - A member responded that they *no longer have Ampere GPUs* which probably means no more support.
- **Single-GPU kernels for 4090/L4s gets a PR**: A member is working on a **PR to unblock single-GPU kernels for 4090/L4s** under the old global layouts.
   - The compilation was broken because of the new required namespaces + this, but this would apply to A100s as well because PGLs are now only tested for Hopper/Blackwell.


  

---


### **GPU MODE ▷ #[ppc](https://discord.com/channels/1189498204333543425/1343373905783554048/1430917059931537488)** (1 messages): 

> `Code Snippets Policy, Shared Memory Speed` 


- **Discourage lengthy code snippets**: A member asked others to avoid posting longer code snippets and instead share a [link to their submission in the ppc system](https://ppc.system).
   - The goal is to not spoil things for other people.
- **Shared Memory not always performant**: A member suggested that load addresses should be calculated like `(something) + threadIdx.x`.
   - They added that while **shared memory is fast**, it's important to ensure data in registers gets reused well, especially when numbers are loaded and used exactly once; it may be more important than even using shared memory.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1431020837695262821)** (3 messages): 

> `Thread Fragment Visualization, Code Formatting in Discord` 


- **Thread Fragment Visualization Questioned**: A user asked if their method to visualize the layout of a **thread fragment** with a given shape and stride is correct, noting that it prints 128 elements with repeated indices between **0** and **79**.
   - The user provided code using `Shape` and `Stride` templates, iterating through 128 elements to print the calculated index using `crd2idx`.
- **Discord Code Formatting Assistance**: A user suggested reformatting the code provided by another user, advising to use triple backticks (`````) for proper code formatting in Discord.
   - Proper formatting is needed to avoid that Discord duplicates the message.


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1430849056074371144)** (4 messages): 

> `Torch/XLA new direction, Lazy Tensor Paper relevance, eDSLs frontends and embeddedness` 


- **Torch/XLA Takes New Turn**: Despite a *new direction* with [torch/xla](https://github.com/pytorch/xla/issues/9684), the **Lazy Tensor Paper** remains relevant for picograd in shoehorning an eager mode into tinygrad to lower the ramp.
   - The [Lazy Tensor Paper](https://arxiv.org/pdf/2102.13267) explores integrating an eager mode into tinygrad using picograd.
- **eDSLs Embeddedness Depth**: It's realized that eDSLs have *non-trivial frontends* depending on the depth of their embeddedness with the host language, using [picograd <- tinygrad](https://pytorch.org/assets/pytorch2-2.pdf) as a case study.
   - Previously it was believed that eDSLs were able to duck parsing, but this depends on whether the eDSL has a *shallow or deep embedding*.


  

---


### **GPU MODE ▷ #[irl-accel-hackathon](https://discord.com/channels/1189498204333543425/1416087968933740688/1430677295458287881)** (12 messages🔥): 

> `Blackwell B200, H100 availability, Nebius Offerings, PyTorch Open Source Week in SF, IRL Event streaming` 


- **Blackwell B200s served a la carte**: Attendees will likely receive a single **Blackwell B200 GPU** upon arrival by scanning a **QR code**.
   - *If you're rich feel free to bring your own compute*.
- **Nebius might not have H100s in stock**: A member inquired about the availability of **H100 GPUs** from Nebius.
   - No direct answer was given, but the inquiry implies that **H100s** might not be a primary offering at the event.
- **Hackathon completely booked, no more tickets**: The IRL accel-hackathon event is overbooked by approximately **6x**, which means it's completely sold out.
   - A member currently in **SF** for **PyTorch/Open Source Week** hoped to attend but is on the waiting list and inquired whether any portions of the event would be streamed online.


  

---


### **GPU MODE ▷ #[cluster-management](https://discord.com/channels/1189498204333543425/1420098114076803142/1430810943251808366)** (2 messages): 

> `Replica Based Recovery, Automated Fault Detection, AI-Based Fault Handling` 


- **Nodes get Redundant Replicas**: Depending on the number of nodes and budget, running redundant replicas actively can yield lower downtime for replica-based recovery.
   - If redundancy is not an option, **checkpointing** is the realistic recovery strategy.
- **Automated Recovery and Orchestration Discussed**: Automated recovery and orchestration are being developed for longer training runs using a layer that automates fault detection and recovery across nodes, GPUs, and interconnects.
   - The system plugs into existing stacks, adding observability and **AI-based fault handling** to prevent losing runs or time due to hardware issues.


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1430792657747312702)** (7 messages): 

> `Phi nodes, cudagraphs, tilelang, Helion compiler improvements` 


- **Phi Nodes Chosen for Compiler Simplicity**: The choice of **phi nodes** in Device IR over block arguments was motivated by implementation simplicity in the Helion compiler, as they primarily preserve the user's control flow.
   - The phi node becomes something like *assign these to the same variable in the output*, which gives output code that looks a lot like the input code for control flow.
- **Helion Kernels Support CUDA Graphs**: **CUDA graphs** are supported in Helion kernels, subject to the same restrictions as other languages, enabling the use of `torch.cuda.make_graphed_callables`.
   - As long as you don't *do something in your kernel that is not cudagraphable*, you should be fine.
- **TileLang Improves Mamba-2 Benchmark**: [TileLang](https://github.com/tile-ai/tilelang/blob/main/benchmark/mamba2/README.md) recently updated their benchmark on **mamba-2-chunk-scan**, showcasing improvements.
   - These improvements to their compiler were in response to published performance numbers, potentially impacting comparisons with Helion and Triton.
- **Helion Compiler Sees Improvements**: TileLang has made some improvements to their compiler in response to published performance numbers.
   - Members pointed out they might have just tuned their **kernel hyperparams**.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1430690040400314388)** (19 messages🔥): 

> `Chat datasets with system messages, AI network like the internet, Discord 'pending' member status` 


- **User Seeks Chat Datasets with System Messages**: A member inquired about recommendations for chat datasets that include **system messages**, finding that [Open Orca](https://huggingface.co/datasets/OpenOrca/OpenOrca) *did the trick*.
   - The member was seeking more recent datasets; another user followed up to ask for a link.
- **Envisioning an AI 'Internet'**: A member proposed the idea of an **AI network** akin to the modern internet, where thousands of models communicate via a specific protocol to answer questions.
   - Another member asked for examples of what that might look like.
- **Discord Introduces 'Pending' Member Status**: Users noticed a new 'pending' status for members joining the Discord server after the *'join'* option was added to the server tag.
   - Members confirmed being 'pending' before approval, with moderators manually accepting them, and speculate that this may no longer be the case.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1430780880175104061)** (59 messages🔥🔥): 

> `Encoder-Decoder Training, Grokking Theory, Meta-Learning RL, New Model Architecture` 


- ****Encoder-Decoder Training**: Is it Better?**: A member questioned if a new paper ([https://arxiv.org/abs/2510.17558v1](https://arxiv.org/abs/2510.17558v1)) is any better than simply training such that the encoder is used for prefill and the decoder for generation, referencing [another paper](https://arxiv.org/abs/2412.09810).
- ****Grokking Theory** gets Interesting Results**: Discussion arose around an interesting **grokking theory** paper with results on classic modular operations tasks, linking to the [paper](https://arxiv.org/abs/2412.09810) and its [code](https://github.com/brantondemoss/GrokkingComplexity).
- ****Meta-Learning RL** Discovers Rules Surpassing Existing Ones**: A paper ([https://www.nature.com/articles/s41586-025-09761-x](https://www.nature.com/articles/s41586-025-09761-x)) on **meta-learning** from cumulative experiences of agents across environments, leading to a discovered rule surpassing existing ones on the **Atari benchmark**, with code available [here](https://github.com/google-deepmind/disco_rl).
- ****New Model Claims** Incredible Performance**: A new member claimed their **50M** model achieves a **0.223** loss compared to a **2.73** loss for a vanilla transformer, with a validation loss of **~0.197** for their **1B** model, resulting in a perplexity of **~1.22**, attaching an [image](https://cdn.discordapp.com/attachments/747850033994662000/1431083629748027402/image.png?ex=68fc1ffd&is=68face7d&hm=c347fd3fa1d4dae87e30579f0723253fd5c83a7197c57f6583d66e7d2ba5ca67&).
- ****Inference** and Bug-Hunting**: Members challenged the new member's claims, suggesting that the model likely has a bug and suggesting debugging via inference, requesting to see the code.
   - The new member declined to share the code for IP reasons, leading to skepticism and suggestions that they were seeking attention rather than genuine help.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1430833790175150143)** (4 messages): 

> `Induction Circuits, Causal Abstractions` 


- **Interpretable Circuits Induce Discussion**: Members discussed the presence of **induction circuits** and structures for bracket matching and long spaces in a given model.
   - According to one member, this model doesn’t seem to contain a lot else (that we can see anyway).
- **Causal Abstraction Craze Calms**: A member inquired about the lack of recent discussion on **Causal Abstractions** ([paper here](https://arxiv.org/abs/2301.04709)), a topic actively discussed in 2023.
   - Another member stated that the framework is still useful but suffers from implicit linearity assumptions ([paper here](https://arxiv.org/abs/2507.08802)), difficulty in choosing the right level of abstraction ([survey here](https://arxiv.org/abs/2408.01416)), and multiple valid abstractions for a single behavior ([paper here](https://arxiv.org/abs/2502.20914)).


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1430853450048213082)** (30 messages🔥): 

> `Model Deployment Microservice, GPU Memory Management, Weight Hardness in Neural Nets, ARC-AGI 100% MythWox, Docker vs Colab` 


- ****Django App Asks: Microservice or Not?****: A member is considering splitting a Django app using **bi-encoders, cross-encoders, YOLO models**, and other GPU-intensive models into a separate microservice to offload inference.
   - They are concerned about potential request queuing on a single GPU and seek advice on deployment strategies and handling multiple parallel RPC calls with limited VRAM (32GB).
- ****GPU Memory Woes and Parallel RPC Prayers****: In response to concerns about GPU overload, a member suggested scheduling requests sequentially if they exhaust VRAM, or running them in parallel if memory permits.
   - They encouraged the original poster to try parallel RPC calls with a **32GB GPU**, to take advantage of parallel processing.
- ****Weight Hardness: Meta Does Dendrites****: A member shared an idea to use an additional scalar on each weight to store their *hardness* or *softness*, inspired by real neuron connections, and linked [Meta's implementation](https://x.com/realJessyLin/status/1980662516285075762).
   - This concept involves adjusting weights based on the *fruit* of the computation, strengthening beneficial connections and weakening detrimental ones, effectively learning new things without modifying irreversible connections.
- ****MythWox Claims ARC-AGI 100%****: A member shared a link to [MythWorx.ai](https://mythworx.ai/capabilities/), claiming **100%** on **ARC-AGI 1** with no pre-training within 4 hours.
   - The member expressed skepticism, questioning the legitimacy of the capabilities, with another reacting with confused disbelief.
- ****Colab can't Docker, Docker can't Colab****: Members discussed the pros and cons of Google Colab versus Docker containers for DL experiments, agreeing Colab is great for quick experiments but lacks Docker container and SSH support.
   - One member emphasized that reproducibility on Google Colab with 1 CPU is a nice headline, but has limitations as it's not always hassle-free.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1430710513293328494)** (26 messages🔥): 

> `Jsonnet, VinePPO, Configuration Management, Tiny Recursive Model, MLP Mixer` 


- **Jsonnet: Overengineered or Tailored?**: Members debated whether [Jsonnet](https://jsonnet.org/) is overengineered, while another appreciated its tailorability and usage by **DeepSeek**.
   - One member found a directory of config files *forbidding*, even for a simple experiment like **VinePPO**.
- **Functional Configs with Jsonnet**: A member suggested using Jsonnet for configurations, especially for applying a functional programming scheme to models to allow easy swapping.
   - Another likened it to **Nix**, defining a hash table using functional programming, questioning its readability for JSON.
- **Tiny Recursive Model Reproduction**: A member shared a link about [reproducing a Tiny Recursive Model with weights](https://www.alphaxiv.org/models/samsung/tiny-recursive-model), noting a 95% probability.
   - The poster highlighted that a **single network design** beats **dual networks** and **MLP attention** improves performance from 74.7% to 87.4%.
- **VinePPO Config Overview**: A member offered to discuss **Jsonnet**/**VinePPO** and then provided links to [ICML](https://icml.cc/virtual/2025/poster/45526), [OpenReview](https://openreview.net/forum?id=5mJrGtXVwz), [ArXiv](https://arxiv.org/abs/2410.01679), and [GitHub](https://github.com/McGill-NLP/VinePPO) related to it.
   - Later specified that *the configuration details for RLVR are more important than the jsonnet language itself*.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1430697332848066571)** (8 messages🔥): 

> `Unseeable Prompt Injections, Google's Genie 3 AI` 


- **AI Community Pans Paper**: Members criticized the quality of a linked paper, with one calling it *poor* despite an *interesting title* and the other agreeing that *this is just not the one*.
   - The first member said that he *thought oh this will be a fun and easy paper and was surprised at how poor it was*.
- **Brave Exposes 'Unseeable' Prompt Injections**: A member shared a [Brave Browser blog post](https://brave.com/blog/unseeable-prompt-injections/) discussing **unseeable prompt injections**.
   - No further discussion was added.
- **Google Gears Up Genie 3 Experiment**: A member linked to a report on [Google's upcoming **Genie 3** experiment](https://www.testingcatalog.com/google-prepares-genie-3-public-experiment-with-ai-generated-worlds/) for generating AI worlds.
   - No further discussion was added.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1430684999568457909)** (12 messages🔥): 

> `Voice mode accents in models, Open Sourcing decision, Building in Public, Weeb Sam finetune, ChatGPT bug` 


- **Models' Accent Adventures: Jamaican Me Crazy!**: A user reported that the **4o model** still does a **Jamaican accent** in voice mode, while **model 5** claims to do it but doesn't change the voice at all.
   - They specified this as *one of the few metrics I care about*.
- **To Open Source or Polish: A Dev's Dilemma**: A member is considering open-sourcing their software but is debating whether to delay the release to fix bugs or ship it with annoying garbage.
   - Their app is designed to let anyone train AI models locally without SaaS, but they have concerns about releasing it with imperfections.
- **Build in Public for Cred**: One member suggested that it's usually good to **build in public**, even if it's just in a personal blog that you don't tell anyone about.
   - They suggest it *helps build cred and even if it flops on social media you have a documented narrative that can help you get grants/funding*.
- **Weeb Sam Finetune for Anime**: The member mentions that they plan to include a **Weeb Sam finetune** in their release, which currently consists of **40k images**.
   - They claim it seems to semi outperform **SAM** when it comes to **Anime**.
- **ChatGPT has Seahorse Scare**: A user shared a link to a [tweet](https://fxtwitter.com/ltx_model/status/1981346235194683497) about a **ChatGPT bug** that occurs when sending the prompt *is there an emoji of a seahorse?*.
   - One commenter speculated that it looks like an **Unreal Engine/Runway/Wan competitor** in the making.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1430749921040007361)** (11 messages🔥): 

> `Scaling Limits, KBLaM Citation, RL Scaling, Interface Matters, Minecraft Voyager` 


- **New Paper Sparks Scaling Limits Debate**: A member shared a [paper](https://arxiv.org/abs/2506.06266) and questioned if **scaling** has hit its limit, considering diminishing returns from **pretraining**, **test-time compute**, and **RL** scaling.
   - Other members disagreed, saying no, *RL, test time compute, etc are only one component of scaling AI*.
- **Interface Design Drives Research Zeitgeist**: A member argued that the **interface**, **workflows**, and **infra** are extremely important components of scaling AI, as the **ChatGPT interface** unlocked a research zeitgeist.
   - They pointed out that *Claude is calling their new prompt system 'Skills' but really this was pioneered (and coined) by Minecraft Voyager years ago*.
- **Voyager's Interface Foresight Praised**: Members emphasized that there's a ton of research that hasn't been implemented at scale yet because the **Human-Computer Interaction component** hasn't caught up with the research.
   - One member praised **Voyager**, noting they *got the privilege of meeting Jim Fan at GTC after he presented it* ([tweet](https://x.com/DrJimFan/status/1662117799974809603)).


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1430749921040007361)** (11 messages🔥): 

> `Scaling Limits, RL Scaling, Minecraft Voyager, Claude Skills` 


- **Scaling Reaching Limit, Paper Claims**: A member shared a [paper](https://arxiv.org/abs/2506.06266) they think is a particularly promising evolution of the idea, and asked if others think **scaling** has hit its limit.
   - Another member simply answered *no*.
- **Interface and Infra Important for Scaling**: One member stated that **RL**, **test time compute**, etc. are only one component of scaling AI and that the interface, workflows, and infra are also extremely important.
   - They added that there's tooons of research that just hasn't been implemented at scale yet because the **Human-Computer Interaction** component hasn't caught up with the research yet, and cited [Jim Fan's tweet](https://x.com/DrJimFan/status/1662117799974809603).
- **Claude Skills vs Minecraft Voyager**: A member discussed that **Claude** is calling their new prompt system *Skills* but really this was pioneered (and coined) by **Minecraft Voyager** years ago it's just that the interface to actually implement it hadn't been formalized at scale (e.g. CLI agents).
   - Another member said they *love bigger Voyager*.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1430691461875236906)** (23 messages🔥): 

> `Kimi Partnership, AI Paper Tracking, Chutes vs Moonshot AI for Kimi K2, Home Server Version Control with Git, OpenRouter Chutes ban` 


- **AlphaXiv Tracks AI Papers Geographically**: A member shared a link to [AlphaXiv](https://www.alphaxiv.org/labs/geo-trends), a tool for tracking the geographic origins of **AI research papers**.
   - Another member asked not to advertise, and then was timed out for repeated violations.
- **Kimi K2 Cheaper Chutes Compromises Quality**: A member inquired about the quality of **Chutes** compared to **Moonshot AI** as a provider for **Kimi K2**.
   - Another member claimed that Chutes trains on user data without a data policy, has less reliable uptime, and its tool call accuracy is half that of the official API and then **Chutes** was memed on [Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1da7j6g/did_you_know_you_can_ban_chutes_openrouter_go_to/).
- **Home Servers Manage versions with Git?**: A member inquired about using **Git** to manage home server versions to aid with resetting after breakages.
   - No one responded to the user's question or concerns.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1430679223584034928)** (22 messages🔥): 

> `Manus sandbox limitations, Pro plan changes, Logic Arenas app, Manus code deprecation, Homework Solutions` 


- **Hybrid Architecture Brainstorming Session**: A user suggested a *hybrid architecture* leveraging local compute and storage, questioning why waste local resources when the user has them available.
   - The suggestion included enabling building large native apps, processing huge datasets locally, and running resource-intensive AI models on local hardware.
- **Pro Plan Users Claim Bait and Switch**: A user expressed frustration over the **Pro plan** now having credit limits, claiming that upon purchase, it was advertised as *unlimited*.
   - The user acknowledged that plans can change but felt it unfair for the month already paid for, asking if anyone wanted to test the pro features.
- **AI-Powered Arena App created by Manus**: A user reported that Manus created a full-stack **AI Arenas app** ([image](https://cdn.discordapp.com/attachments/1349440650495398020/1430878322073796628/2025-10-23_12-15.jpg?ex=68fc0988&is=68fab808&hm=efef49879866f78ee43ea3c281eca345ab2bcf800110c561cc4e81bc723f5219&)) (*not yet tested*) within an hour using minimal credits.
   - The **Logic Arenas** app features *Heart vs Mind*, *Dual Parents*, and *Moral Court* arenas, built with **Kotlin**, **Ktor**, **Redis**, **Jetpack Compose**, and **Material 3**, comprising ~7k LOC across 28 files.
- **Manus code uses Deprecated Modules, Plugins**: A user reported that **Manus** could improve by using non-deprecated code, modules, and plugins, requiring significant effort to update everything.
   - Although Manus claimed to create a beautiful Material 3 dark theme ([image1](https://cdn.discordapp.com/attachments/1349440650495398020/1430917287363350728/2025-10-23_14-51.jpg?ex=68fc2dd2&is=68fadc52&hm=4e68706e60e030a96d99d157b3abdacc9d2b99077639e85f545cb96a0620d614&), [image2](https://cdn.discordapp.com/attachments/1349440650495398020/1430917287707279391/2025-10-23_14-47.jpg?ex=68fc2dd2&is=68fadc52&hm=04f22a7c26e41abcb491ebfcbdcb2c4ec8b05ad504eeb504de7aa4c2bc80f3f4&)), the initial Android app design looked outdated, but Claude Code improved it with one prompt ([image](https://cdn.discordapp.com/attachments/1349440650495398020/1430971270219956234/2025-10-23_18-26_1.jpg?ex=68fbb758&is=68fa65d8&hm=d73323d08753d88efdd123cd9be451622fbb989859f6077c67b76d45598ebfcd&)).
- **Manus Flunks Homework Assignment**: A user reported that **Manus** failed to correctly solve their homework, even with similar examples provided in two notebooks, and the generated PDF was disorganized.
   - The user highlighted Manus' inability to complete this seemingly simple request.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1430893317289410560)** (7 messages): 

> `aider API keys, gemini-cli cost, playwright on Fedora` 


- **Aider API Keys: Cost-Competitive?**: A user asked about using **aider** with API keys to match **gemini-cli's** cost-effectiveness.
   - The user noted that they use `/tokens` to manage keys, clear history, and watch context size.
- **Gemini-cli's Generous Plans**: A user described **gemini-cli's** cost profile, mentioning a plan at around **$20 USD** a month that provides nominally **1500** requests a day with a **1M token** context window.
   - The user also mentioned that while the interface is mostly better than **aider's**, it lacks a repo map but relies a lot on file system operations, such as `grep`.
- **Playwright on Fedora: Anyone?**: A user asked if anyone has successfully gotten **Playwright** working on **Fedora**.
   - No solutions or workarounds were provided in the excerpt.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1430741210439618612)** (1 messages): 

> `Future of Aider, Paul Gauthier's Vision` 


- **Community Awaits Paul Gauthier's Vision for Aider's Future**: Members are curious about the future direction of **Aider** and are looking forward to hearing from its creator, **Paul Gauthier**.
   - The community anticipates insights into upcoming developments and strategic decisions for the project.
- **Speculation on Aider's Roadmap**: Discussion revolves around the potential evolution of **Aider**, with users eager to understand the roadmap.
   - Users seek clarity on planned features, enhancements, and the overall trajectory of the **Aider** project.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1430758415516569640)** (2 messages): 

> `tinygrad slide at pytorchconf, tinygrad dev onboarding` 


- **tinygrad Slides into pytorchconf!**: tinygrad got a [slide](https://cdn.discordapp.com/attachments/1068976834928193609/1430758414744686655/HDjLLqQ.png?ex=68fc429c&is=68faf11c&hm=13bea9c27e9b130accbe594e495dbd0e6813cb720aaabb5bb6f2375495a385b5&) at **pytorchconf**.
   - The slide showcases **tinygrad**'s integration and relevance within the broader PyTorch ecosystem.
- **New tinygrad Devs Wanted**: A member asked about documentation on becoming a **tinygrad developer**.
   - No specific resources were directly linked in the messages, but it implies an interest in expanding the **tinygrad** contributor base.


  