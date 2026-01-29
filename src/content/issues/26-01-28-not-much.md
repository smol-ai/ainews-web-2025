---
id: MjAyNi0w
title: not much happened today
date: '2026-01-28T05:44:39.731046Z'
description: >-
  **AI News for 1/27/2026-1/28/2026** highlights a quiet day with deep dives
  into frontier model "personality split" where **GPT-5.2** excels at
  *exploration* and **Claude Opus 4.5** at *exploitation*, suggesting **OpenAI**
  suits research workflows and **Anthropic** commercial reliability. The rise of
  agentic coding loops shows new failure modes, with *self-verification*
  workflows gaining traction. The open-model **Kimi K2.5** emerges as a
  flashpoint, boasting enhanced **agent execution**, **multimodality**, and
  **coding polish**, runnable on **Apple silicon M3 Ultra Mac Studios** with
  **Thunderbolt 5 (RDMA)**, and challenging **Claude Opus 4.5** on benchmarks
  and pricing. Licensing issues threaten enterprise adoption despite model
  quality. The meme "clawdbot" reflects rapid agent branding proliferation.
  Agent engineering advances with shared "skills" interfaces promoted by
  **DeepLearning.AI**, **Anthropic**, and **LangChain**.
companies:
  - openai
  - anthropic
  - deeplearningai
  - langchain
  - apple
models:
  - gpt-5.2
  - claude-opus-4.5
  - kimi-k2.5
topics:
  - agentic-ai
  - multimodality
  - coding
  - self-verification
  - agent-engineering
  - model-benchmarking
  - model-optimization
  - workflow-automation
people: []
---


**a quiet day**

> AI News for 1/27/2026-1/28/2026. We checked 12 subreddits, [**544** Twitters](https://twitter.com/i/lists/1585430245762441216) and **24** Discords (**206** channels, and **7100** messages) for you. Estimated reading time saved (at 200wpm): **559 minutes**. **Our new website** is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on [@smol_ai](https://x.com/Smol_AI)!

quiet day.


---

# AI Twitter Recap

**Frontier model “personality split” + how people are actually using them**

- **Exploration vs. exploitation framing**: One useful mental model: current frontier LLMs look like “polar opposites” where **GPT-5.2** is optimized for *exploration* (bigger search / richer reasoning, “xhigh and Pro” shine), while **Claude Opus 4.5** is more *exploitation* (stronger reliability with fewer tokens; extra “reasoning” often adds less) — implying OpenAI may be better positioned for research workflows, Anthropic for commercial reliability-heavy deployments ([tweet](https://twitter.com/scaling01/status/2016335491243676058)).  
- **Coding agent “phase shift” is real—but messy**: Multiple posts reflect a step-change in practice: founders and engineers are increasingly running “agentic” coding loops, yet hitting new failure modes: agents that don’t ask clarifying questions, get “confused,” or edit unrelated files. Mikhail Parakhin describes reaching the point where he can specify a scheduler and trust it to work, but still can’t let agents loose on established codebases due to collateral edits ([tweet](https://twitter.com/MParakhin/status/2016362688444825833)). Related: workflow suggestions like *self-verification* (e.g., Playwright screenshots + iterate-until-pass rules) are becoming common operational discipline ([tweet](https://twitter.com/pierceboggan/status/2016335657602285822)).

---

**Kimi K2.5 (+ “clawdbot” / swarm-mode) becomes the week’s open-model flashpoint**

- **K2.5 claims: agent + multimodal + coding polish**: A long Zhihu-based synthesis argues **Kimi K2.5** upgrades K2’s “intelligence > capability” imbalance by strengthening **agent execution**, **multimodality**, and **coding**, reducing brute-force token usage and improving instruction-following stability; still flagged: hallucinations and a persistent NBSP formatting quirk ([thread](https://twitter.com/ZhihuFrontier/status/2016363957876097089)). A second Zhihu recap makes a pragmatic case for multimodality: “vision” matters when agents need to verify UI state (overlaps, broken images, visual regressions), enabling tighter action–critic loops with less human feedback ([thread](https://twitter.com/ZhihuFrontier/status/2016438778030850059)).  
- **Distribution + local runs are driving hype**: Reports of K2.5 being runnable on high-end Apple silicon setups went viral: **~24 tok/s** using **2× 512GB M3 Ultra Mac Studios** connected via **Thunderbolt 5 (RDMA)** with **Exo Labs / MLX** backend ([tweet](https://twitter.com/alexocheema/status/2016404573917683754)). Kimi also pushed an AMA on r/LocalLLaMA ([tweet](https://twitter.com/Kimi_Moonshot/status/2016443435553890419)) and announced availability on “Eigent” ([tweet](https://twitter.com/Kimi_Moonshot/status/2016473945957155252)).  
- **Benchmarks + pricing pressure**: Kilo Code promoted a free week, claiming K2.5 beats Opus 4.5 on several coding benchmarks ([tweet](https://twitter.com/kilocode/status/2016449095511007535)); Kimi’s own account claimed “#1 open model for coding” ([tweet](https://twitter.com/Kimi_Moonshot/status/2016521406906028533)). An anecdotal A/B/C test on UI-from-image generation found Opus best quality but pricey, Codex fastest/cheapest but lower fidelity, and K2.5 ~“90% of Opus quality at ~38% cost” ([tweet](https://twitter.com/JuanPa/status/2016634998988865571)).  
- **Licensing friction as an adoption blocker**: A pointed note argues modified licenses + logo requirements can kill enterprise adoption even if the model is excellent ([tweet](https://twitter.com/dbreunig/status/2016531878795256286)).  
- **“Clawdbot” as a cultural artifact**: The meme itself (people confused about what “clawdbot” even is) reflects how fast agent branding and forks proliferate ([tweet](https://twitter.com/dejavucoder/status/2016341138740052126)), and sets up broader concerns about ecosystem signal loss (see below).

---

**Agent engineering: skills, harnesses, evals, and “reliability tax”**

- **Skills are crystallizing into a shared interface layer**: A major theme is moving workflow logic out of prompts into reusable “skills” (files/folders of instructions, loaded on demand). DeepLearning.AI + Anthropic launched a course on “Agent Skills” emphasizing portability across Claude (Claude.ai, Claude Code, API, Agent SDK) ([tweet](https://twitter.com/AndrewYNg/status/2016564878098780245)), and LangChain is pushing “Skills” via progressive disclosure as lightweight, shareable units ([tweet](https://twitter.com/sydneyrunkle/status/2016585688389734654)). HF showcased “upskill”: convert strong-model traces into transferable skills, then evaluate impact; CUDA-kernel-writing saw up to **+45% accuracy** on some open models but degraded others—reinforcing the need for per-model measurement ([tweet](https://twitter.com/ben_burtenshaw/status/2016534389685940372); blog link in thread: https://twitter.com/ben_burtenshaw/status/2016534392974234013).  
- **Context management is becoming “filesystem-first”**: DeepAgents (LangChain) describes offloading/summarizing tool I/O and leaning on the filesystem for context boundaries ([thread](https://twitter.com/hwchase17/status/2016548732880445772); additional note: [tweet](https://twitter.com/sydneyrunkle/status/2016560221720867307)).  
- **Evals are converging on multi-turn + traceability**: Calls for agent tracing as the foundation of evaluating single-step vs full-turn vs multi-turn behavior show up explicitly ([tweet](https://twitter.com/samecrowder/status/2016563057947005376)). New benchmarks/harnesses: **SWE-fficiency** released its harness and repo ([tweet](https://twitter.com/18jeffreyma/status/2016511583032061999); also [tweet](https://twitter.com/OfirPress/status/2016559053808222644)), and **CooperBench** is highlighted for measuring multi-agent coordination ([tweet](https://twitter.com/gneubig/status/2016555800982937879)). Safety-side: “AgentDoG” proposes diagnosing root causes of unsafe actions across trajectories ([tweet](https://twitter.com/HuggingPapers/status/2016366634475388968)).  
- **Reliability and verification loops are the bottleneck**: MiniMax notes long interaction chains are costly and proposes **parallel tool invocation** to reduce rounds in verifier-style setups ([tweet](https://twitter.com/MiniMax_AI/status/2016488781860458789)). Separately, a strong critique warns “vibe-coded software” destroys traditional signals (design quality, docs, ecosystem maturity), shifting the evaluation burden to users and demanding new trust frameworks ([tweet](https://twitter.com/tnm/status/2016342022723141782)).

---

**Infra + efficiency: quantization, distillation, inference stacks, and local deployment**

- **NVIDIA’s NVFP4 push (Nemotron 3 Nano)**: NVIDIA released an **NVFP4** precision version of **Nemotron 3 Nano**, claiming **up to 4× throughput on Blackwell B200** and **~99.4% BF16 accuracy** via **Quantization Aware Distillation** ([tweet](https://twitter.com/NVIDIAAIDev/status/2016556881712472570)). vLLM quickly added support ([tweet](https://twitter.com/vllm_project/status/2016562169140433322)).  
- **Embedding-heavy architectures are “hot again”**: Discussion around DeepSeek’s Engram-like ideas continues: a LongCat Flash paper is summarized as using **multi-hash sub-tables** and finding embeddings help mainly at high MoE sparsity; key practical gotchas include amplification (√D/LayerNorm) to avoid first-attention drowning and collision spikes when vocab sizes align poorly ([tweet](https://twitter.com/eliebakouch/status/2016577949676319092)).  
- **Inference/tooling ecosystem keeps consolidating**: vLLM’s SIGs and office hours are formalizing governance and roadmap cadence ([tweet](https://twitter.com/vllm_project/status/2016526685869596974)); LM Studio 0.4.0 positions itself as “next gen” for deploying local models with parallel requests and a stateful REST API + MCP support ([tweet](https://twitter.com/lmstudio/status/2016573570822930708)). Cohere launched **Model Vault** (isolated VPC, “no noisy neighbors,” elastic inference) as managed “sovereign” hosting ([tweet](https://twitter.com/cohere/status/2016512841751154739)).  
- **Distillation as the default “shipping form factor”**: Multiple posts echo the emerging standard: train the best model you can, then distill/quantize for deployment ([tweet](https://twitter.com/code_star/status/2016588669008953631)). MongoDB Research’s **LEAF** proposes asymmetric distillation for embeddings: embed documents with the large teacher offline, embed queries with a compact student online; claims **~96% of teacher quality**, **5–15× smaller**, up to **24× faster**, enabling CPU/edge embedding inference ([tweet](https://twitter.com/LiorOnAI/status/2016481603426414883)).

---

**Big-tech productization: browser agents, “AI scientist” narratives, and adoption reality checks**

- **Gemini 3 is taking over Google surfaces**: Gemini 3 now powers **AI Overviews** globally ([tweet](https://twitter.com/_philschmid/status/2016552420013199856)). Google rolled out major Chrome updates: side-panel UX, deeper app integrations, Nano Banana for image editing/creation, and **Auto Browse** for multi-step chores (preview; US; Pro/Ultra) ([thread](https://twitter.com/Google/status/2016575105346773297); also [thread](https://twitter.com/GeminiApp/status/2016575257436647521)). Engineers noted this may be the strongest browser AI integration so far ([tweet](https://twitter.com/kimmonismus/status/2016628933706309981)).  
- **OpenAI Prism positioning**: Sebastien Bubeck explicitly denies OpenAI intends to take a share of discoveries, encouraging researchers to use ChatGPT/Prism for science ([tweet](https://twitter.com/SebastienBubeck/status/2016345977481777188)). Others highlight Prism’s utility for students learning papers via diagrams ([tweet](https://twitter.com/daniel_mac8/status/2016554325691015604)).  
- **Adoption is still uneven**: A notable fault line: founders actively using cutting-edge tools see the shift firsthand; others still treat AI as “meh,” limiting org adoption ([tweet](https://twitter.com/GergelyOrosz/status/2016443395405705533)). The Information reports ChatGPT Agent struggling with usage/adoption ([tweet](https://twitter.com/steph_palazzolo/status/2016545857139540260)).  
- **Microsoft “digital co-worker” competition**: Reports say Satya Nadella is personally testing rival agents and accelerating internal development, even using Anthropic models, to own the Windows-native agent layer ([tweet](https://twitter.com/kimmonismus/status/2016526803138236916)).

---

**Science + robotics: genomics weights open, interpretability as discovery engine, and embodied scaling**

- **DeepMind AlphaGenome goes open**: DeepMind announced **AlphaGenome** for predicting molecular impacts of genetic changes, cited **1M+ API calls/day** and **3,000+ users**; then announced making **model + weights available** ([tweet](https://twitter.com/GoogleDeepMind/status/2016542480955535475); weights: [tweet](https://twitter.com/GoogleDeepMind/status/2016542490115912108)). Later, weights availability was reiterated with a Hugging Face collection link ([tweet](https://twitter.com/osanseviero/status/2016628065422762113)).  
- **Interpretability → biomarkers pipeline (Goodfire + Prima Mente)**: Goodfire reports identifying a novel class of **Alzheimer’s biomarkers** using interpretability on a biomedical foundation model, framing a repeatable loop: train superhuman models on scientific data → mech interp → experimental validation → new science ([thread](https://twitter.com/GoodfireAI/status/2016563911508840623)).  
- **Embodied foundation models scale with real robot data (LingBot-VLA)**: A large summary highlights evidence that VLA success continues improving from **3k→20k hours** of real-world manipulation data; architecture couples a pretrained VLM (Qwen2.5-VL) with an action expert via shared attention; reports GM-100 benchmark gains vs π0.5 and others ([tweet](https://twitter.com/omarsar0/status/2016518141308993565)).  
- **Figure’s Helix robot control**: Brett Adcock claims a Helix model controls full-body behavior (walking/touching/planning) with **no teleoperation**, calling it Figure’s most significant release ([tweet](https://twitter.com/adcock_brett/status/2016358054242222136)).

---

### Top tweets (by engagement)

- **Company health / layoffs**: “Quarterly layoffs for two years is worse for your health than smoking three packs/day” ([tweet](https://twitter.com/vikhyatk/status/2016345591748690295)).  
- **Kimi K2.5 local run**: 2× M3 Ultra Mac Studio setup running K2.5 at ~24 tok/s ([tweet](https://twitter.com/alexocheema/status/2016404573917683754)).  
- **Coding’s “outsourcing moment”**: Clean Code author using Claude to write software as a symbolic milestone ([tweet](https://twitter.com/mischavdburg/status/2016389228356149460)).  
- **New AI lab announcement**: “Flapping Airplanes” raises **$180M** (GV/Sequoia/Index) ([tweet](https://twitter.com/flappyairplanes/status/2016564437499728259)).  
- **Karpathy on new research labs**: argues it’s still plausible for new research-first startups to out-execute incumbents; expects potential **10×** breakthroughs, congratulating new founders ([tweet](https://twitter.com/karpathy/status/2016590919143952466)).  
- **Google Chrome + Gemini 3 agent features**: major Chrome rollout thread ([tweet](https://twitter.com/Google/status/2016575105346773297)).


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Kimi K2.5 Model Performance and Cost Analysis

  - **[Run Kimi K2.5 Locally](https://www.reddit.com/r/LocalLLaMA/comments/1qpfse6/run_kimi_k25_locally/)** (Activity: 328): **The image provides a guide for running the **Kimi-K2.5** model locally, emphasizing its state-of-the-art (SOTA) performance in vision, coding, agentic, and chat tasks. The model, which is a 1 trillion parameter hybrid reasoning model, requires `600GB` of disk space, but the quantized **Unsloth Dynamic 1.8-bit** version reduces this requirement to `240GB`, a `60%` reduction. The guide includes instructions for using `llama.cpp` to load models and demonstrates generating HTML code for a simple game. The model is available on [Hugging Face](https://huggingface.co/unsloth/Kimi-K2.5-GGUF) and further documentation can be found on [Unsloth's official site](https://unsloth.ai/docs/models/kimi-k2.5).** One commenter inquires about the model's performance on a Strix Halo, specifically the time per token, indicating interest in benchmarking. Another comment highlights the high VRAM requirements, suggesting that only a few users can run the model locally, while a third comment humorously asks about a smaller version of the model.

    - Daniel_H212 is inquiring about the performance of the Kimi K2.5 model on the Strix Halo hardware, specifically asking for the token generation speed in seconds per token. This suggests a focus on benchmarking the model's efficiency on high-end hardware setups.
    - Marksta provides feedback on the quantized version of the Kimi K2.5 model, specifically the Q2_K_XL variant. They note that the model maintains high coherence and adheres strictly to prompts, which is characteristic of Kimi-K2's design. However, they also mention that while the model's creative capabilities have improved, it still struggles with execution in creative scenarios, often delivering logical but poorly written responses.
    - MikeRoz questions the utility of higher quantization levels like Q5 and Q6 (e.g., UD-Q5_K_XL, Q6_K) when experts prefer int4 quantization. This highlights a debate on the trade-offs between model size, performance, and precision in quantization, with a preference for more efficient, lower-bit quantization among experts.

  - **[Kimi K2.5 is the best open model for coding](https://www.reddit.com/r/LocalLLaMA/comments/1qp87tk/kimi_k25_is_the_best_open_model_for_coding/)** (Activity: 840): **The image from LMArena.AI showcases Kimi K2.5 as the leading open model for coding, ranked #7 overall. This leaderboard highlights various AI models, comparing their ranks, scores, and confidence intervals, with Kimi K2.5 noted for its superior performance in coding tasks. The model is praised for its accuracy, being comparable to Sonnet 4.5, and surpassing GLM 4.7, though it is not at the level of Opus in terms of agentic function. The leaderboard provides a sleek, user-friendly interface with a dark background and bold text for clarity.** One commenter notes that LMArena's leaderboard may not fully capture a model's multi-turn, long context, or agentic capabilities, suggesting it is more of a 'one-shot vibe check.' Another user is curious about the local setup required to run Kimi K2.5.

    - A user compared Kimi K2.5 to other models like Sonnet 4.5 and GLM 4.7, noting that while Kimi 2.5 is on par with Sonnet 4.5 in terms of accuracy, it surpasses GLM 4.7, which was their previous choice. They also expressed interest in seeing if GLM-5 from [z.ai](http://z.ai) will outperform Kimi 2.5.
    - Another user highlighted the cost-effectiveness of Kimi K2.5, stating that it feels as competent as Opus 4.5 despite being significantly cheaper, approximately 1/5th of the cost. They also mentioned that it is less expensive than Haiku, emphasizing its value for performance.
    - A comment criticized LMArena for not providing insights into a model's multi-turn, long context, or agentic capabilities, suggesting that it only offers a superficial evaluation of models.

  - **[Kimi K2.5 costs almost 10% of what Opus costs at a similar performance](https://www.reddit.com/r/LocalLLaMA/comments/1qoty38/kimi_k25_costs_almost_10_of_what_opus_costs_at_a/)** (Activity: 716): **The image provides a cost comparison between **Claude Opus 4.5** and **Kimi K2.5** models, highlighting that Kimi K2.5 is significantly cheaper, costing only 10% of what Claude Opus 4.5 does for similar performance. Specifically, Claude Opus 4.5 costs `$5.00` for input and `$25.00` for output per million tokens, whereas Kimi K2.5 costs `$0.60` for input and `$2.50` for output. This suggests that Kimi K2.5 could be a cost-effective alternative to state-of-the-art closed models, especially for non-website tasks.** Some commenters express skepticism about the performance claims, noting that Kimi K2.5 uses three times the tokens for the same tasks, which affects the cost-effectiveness and latency. Others acknowledge the potential of Kimi models, particularly for writing tasks.

    - one-wandering-mind highlights that Kimi K2.5 uses 3x the tokens compared to Opus for the same tasks, which affects both cost and latency. This suggests that while Kimi K2.5 is cheaper, the cost advantage is more accurately 3x rather than 10x when considering token usage. The comment also emphasizes the importance of considering token usage in performance comparisons, as it impacts both cost and latency.
    - ghulamalchik mentions a preference for upcoming models like DeepSeek 4 and MiniMax M2.2, based on past experiences with various models. This suggests that while Kimi K2.5 is notable, some users are anticipating future releases from other models that have proven reliable in their experience.

  - **[Kimi K2 Artificial Analysis Score](https://www.reddit.com/r/LocalLLaMA/comments/1qos25i/kimi_k2_artificial_analysis_score/)** (Activity: 405): **The image presents a comparative analysis of AI models through the "Artificial Analysis Intelligence Index," highlighting "Kimi K2" with a score of `47` and an operational cost of `$371`. The discussion around the image focuses on the licensing terms of "Kimi K2.5," which restricts commercial use for products with over `100 million` monthly active users or `$20 million` in monthly revenue, requiring prominent display of "Kimi K2.5" branding. This licensing approach is compared to other models like Llama 4, suggesting either a bug or inconsistency in application. The image and comments reflect on the competitive landscape of AI models, particularly in open-source versus commercial use contexts.** Commenters discuss the licensing terms of "Kimi K2.5," noting its unique restrictions compared to other models like Llama 4. There is also a sentiment of anticipation for an open-source model to outperform commercial ones, with a mention of "DeepSeek."

    - FullOf_Bad_Ideas highlights a licensing nuance in Kimi K2.5's modified MIT license, which requires prominent display of 'Kimi K2.5' for commercial products exceeding 100 million monthly active users or $20 million in monthly revenue. This stipulation is not applied to other models like Llama 4, suggesting either a bug or inconsistency in application.
    - BrianRin discusses the potential of Kimi 2.5 in enterprise use cases, comparing it to Opus 4.5, Gemini 3 Pro, and GPT 5.2. The commenter is interested in Kimi 2.5's cost-effectiveness and output quality, noting that if it achieves 95% of the output quality of these models, it could be a viable option for scaling up enterprise applications.
    - sine120 critiques the Artificial Analysis score, suggesting it is not a meaningful metric for evaluating how a model performs in practical scenarios. This implies a need for more nuanced evaluation metrics that better capture real-world usability and performance.

  - **[[LEAKED] Kimi K2.5’s full system prompt + tools (released &lt;24h ago)](https://www.reddit.com/r/LocalLLaMA/comments/1qoml1n/leaked_kimi_k25s_full_system_prompt_tools/)** (Activity: 282): **The post reveals a leak of the full system prompt and tools for **Moonshot's Kimi K2.5**, including `5k tokens` of data such as tool schemas, memory CRUD protocols, context engineering, and basic guardrails. The leak includes external data sources like finance and arXiv, and has been independently verified across multiple platforms, including [GitHub](https://github.com/dnnyngyen/kimi-k2.5-prompts-tools) and [Kimi](https://www.kimi.com/share/19c003f5-acb2-838b-8000-00006aa45d9b). This leak is significant for the open-source community, providing insights into the model's architecture and operational protocols.** Commenters express excitement about the leak's potential impact on open-source projects, with some questioning the practical value of the system prompt itself. Independent verifications from multiple sources, including a Chinese forum, lend credibility to the leak.

    - The leaked system prompt for Kimi K2.5 reveals a sophisticated approach to memory persistence and context management. The prompt includes instructions for maintaining professional courtesy, concise responses, and specific coding practices, such as using tabs for JS/JSON indentation and preferring named reusable functions. This structure aims to address the 'hollow AI assistant' problem by providing persistent behavioral anchors, which can significantly affect the model's ability to maintain personality consistency across sessions.
    - The memory persistence mechanism in Kimi K2.5 is particularly noteworthy. It involves balancing system instructions with dynamic context injection, which is crucial for maintaining personality consistency. The system's approach to conversation summarization or retrieval can influence new chats, and even minor changes in memory structuring can lead to shifts in the model's responses, sometimes making them feel more 'authentic.' This highlights the importance of initial prompt structure in determining whether an AI 'remembers' its behavioral patterns or just factual content.
    - The system prompt for Kimi K2.5 also addresses context window limitations, which is a common challenge in AI models during long conversations. The prompt engineering is designed to handle these limitations by structuring previous interactions in a way that supports conversation continuity. This approach not only helps in maintaining the flow of conversation but also in ensuring that the AI's responses remain relevant and contextually appropriate, even as the conversation extends.


### 3. Z-Image Model Teasers and Announcements

  - **[The z-image base is here!](https://www.reddit.com/r/LocalLLaMA/comments/1qoiep6/the_zimage_base_is_here/)** (Activity: 327): ****Tongyi-MAI** has released the `Z-Image` model on [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image), showcasing its capabilities in generating high-quality images, particularly focusing on female subjects, which constitute approximately `90%` of the demos. The model is noted for its potential to run on `12GB GPUs` with minimal quality loss, suggesting efficient optimization possibilities. A notable feature is the "Negative Prompt" functionality, which allows for specific image generation constraints, as demonstrated in a translated example where the prompt specifies "Westerners, physical deformities."** Commenters highlight the model's focus on generating images of women, reflecting a primary use case. There is also a discussion on the model's potential to operate on lower-spec hardware with optimizations, indicating its efficiency and adaptability.

    - Dr_Kel discusses the potential for optimizing the z-image model to run on 12GB GPUs with minimal quality loss, suggesting that with some adjustments, the model could be more accessible to users with less powerful hardware.
    - Middle_Bullfrog_6173 points out that the z-image base model is primarily useful for those interested in training or fine-tuning models, rather than end-users. They imply that this base model serves as a foundation for further development, such as the turbo model, which has been post-trained from it.


  - **[API pricing is in freefall. What's the actual case for running local now beyond privacy?](https://www.reddit.com/r/LocalLLaMA/comments/1qp6rm5/api_pricing_is_in_freefall_whats_the_actual_case/)** (Activity: 913): **The post discusses the rapidly decreasing costs of API access for AI models, with **K2.5** offering prices at `10%` of **Opus** and **Deepseek** being nearly free. **Gemini** also provides a substantial free tier, leading to a `50%` monthly drop in API cost floors. In contrast, running a `70B` model locally requires significant hardware investment, such as a `k+ GPU`, or dealing with quantization trade-offs, resulting in `15 tok/s` on consumer hardware. The post questions the viability of local setups beyond privacy, noting that while local setups offer benefits like latency control and customization, these are niche advantages compared to the cost-effectiveness of APIs.** Commenters highlight the importance of offline capabilities and distrust in API providers' long-term pricing strategies, suggesting that current low prices may not be sustainable. They also emphasize the value of repeatability and control over model behavior when running locally, which can be compromised with API changes.

    - Minimum-Vanilla949 highlights the importance of offline capabilities for those who travel frequently, emphasizing the risk of API companies changing terms or prices unexpectedly. This underscores the value of local models for consistent access and control, independent of external changes.
    - 05032-MendicantBias discusses the unsustainable nature of current API pricing, which is often subsidized by venture capital. They argue that once a monopoly is achieved, prices will likely increase, making local setups and open-source tools a strategic hedge against future cost hikes.
    - IactaAleaEst2021 points out the importance of repeatability and trust in model behavior when using local models. By downloading and auditing a model, users can ensure consistent performance, unlike APIs where vendors might alter model behavior without notice, potentially affecting reliability.


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Kimi K2.5 and Related Model Releases

  - **[Open source Kimi-K2.5 is now beating Claude Opus 4.5 in many benchmarks including coding.](https://www.reddit.com/r/singularity/comments/1qoojio/open_source_kimik25_is_now_beating_claude_opus_45/)** (Activity: 1078): ****Kimi-K2.5**, an open-source model, reportedly surpasses **Claude Opus 4.5** in several benchmarks, notably in coding tasks. However, the specifics of these benchmarks and the extent of the performance improvements are not detailed, leading to skepticism about the real-world applicability of these results. The announcement highlights the ongoing competition in the open-source AI community to match or exceed proprietary models in specific tasks.** Commenters express skepticism about the claim, questioning the benchmarks' relevance to real-world applications and the lack of detailed evidence supporting the superiority of Kimi-K2.5 over Claude Opus 4.5.

    - There is skepticism about the claim that Kimi-K2.5 is outperforming Claude Opus 4.5 in benchmarks, with some users questioning the specific benchmarks being referenced. The term 'many' is seen as vague, and there is a call for more detailed information on which benchmarks are being used to substantiate these claims.
    - The discussion highlights a common critique of benchmarks, which is that they often do not reflect real-world utility. One user points out that while Kimi-K2.5 might perform well in controlled benchmark environments, it may not match the practical performance of Claude Opus 4.5, especially in tasks like programming where Opus 4.5 is noted for providing solutions in a single prompt.
    - There is a general sentiment that benchmarks are not sufficient to gauge a model's practical capabilities. The conversation suggests that while Kimi-K2.5 might show promising results in benchmarks, its real-world application, particularly in programming, might not be as effective as Claude Opus 4.5, which is praised for its efficiency in delivering solutions.

  - **[Kimi K2.5 Released!!!](https://www.reddit.com/r/singularity/comments/1qo531i/kimi_k25_released/)** (Activity: 1233): **The image presents a performance comparison chart of four AI models: **Kimi K2.5**, **GPT-5.2 (xhigh)**, **Claude Opus 4.5**, and **Gemini 3 Pro**. **Kimi K2.5** is highlighted in blue and shows competitive scores across various tasks, including agents, coding, image, and video processing. The chart features specific benchmarks such as "Humanity's Last Exam," "BrowseComp," and "OmniDocBench 1.5," where **Kimi K2.5** often leads or performs strongly, indicating its effectiveness and accuracy in these tasks. The scores are presented in percentiles, showcasing the model's performance relative to others.** Commenters discuss the issue of hallucinations in AI models, with **Kimi K2.5** showing improvement over its predecessor but still producing incorrect answers. **GPT 5.1 and 5.2** are noted for acknowledging when they don't know an answer, unlike **Kimi 2.5** and **Gemini 3**, which confidently provide incorrect answers. There is skepticism about the benchmarks' representativeness, questioning if **Kimi K2.5** is truly better than **Gemini 3** in most cases.

    - A user conducted a test on Kimi K2.5's ability to follow instructions by asking it to identify a specific math contest problem without web search. The model listed hallucinated contest problems and second-guessed itself, ultimately providing incorrect answers. This behavior is an improvement over Kimi K2, which failed to follow instructions and timed out. In contrast, GPT 5.1 and 5.2 are noted for their ability to admit 'I don't know,' while Gemini 3 confidently provides incorrect answers.
    - The concept of an 'agent swarm' in AI models is discussed, where potentially over 100 instances of a model are directed by a single overseeing instance. This setup is presumed to be expensive and complex, with the possibility of a single model handling multiple tasks simultaneously being a significant advancement. The user expresses interest in practical experiences with this setup, suggesting that scaffolding might be a more feasible approach.
    - A user questions the validity of benchmarks comparing Kimi K2.5 to Gemini 3, implying that results might be cherry-picked. They express skepticism about Kimi K2.5 consistently outperforming Gemini 3, suggesting that such claims seem exaggerated without broader evidence.


  - **[Cline 3.55.0: Arcee Trinity Large and Kimi K2.5 now available](https://www.reddit.com/r/CLine/comments/1qpl2fk/cline_3550_arcee_trinity_large_and_kimi_k25_now/)** (Activity: 5): ****Cline 3.55.0** introduces two significant open models: **Arcee Trinity Large** and **Kimi K2.5**. Arcee Trinity Large is a `400B` parameter MoE model with `13B` active parameters during inference, offering a `128K` context window. It achieves `82` on MMLU Pro and `75` on GPQA Diamonds, making it suitable for general coding and large codebase management without API costs. **Kimi K2.5** is a `1T` parameter MoE model with a `256K` context, scoring `76.8%` on SWE-bench and surpassing Opus 4.5 on Humanity's Last Exam with `50.2%`. It excels in visual coding, capable of generating UI code from screenshots and self-correcting its output. Additionally, **ChatGPT Plus/Pro** users can access GPT-5 models in Cline without an API key. [Full details here](https://cline.bot/blog/cline-3-55-0-arcee-trinity-and-kimi-k2-5-now-in-cline).** Some users express excitement about the open-source nature and competitive performance of these models, particularly noting the potential for cost savings and flexibility in coding applications. There is also interest in the models' ability to handle large context windows and self-correcting features.

    - A user highlights the performance improvements in the Arcee Trinity Large model, noting that it shows a significant increase in processing speed compared to previous versions. They mention that the model's architecture has been optimized for better parallel processing, which is crucial for handling large datasets efficiently.
    - Another comment discusses the Kimi K2.5 model's enhanced capabilities in natural language understanding. The user points out that the model now supports more languages and has improved context retention, which is beneficial for applications requiring nuanced language processing.
    - A technical debate arises around the memory usage of the new models. Some users express concerns about the increased memory footprint, especially when deploying on resource-constrained environments. Others argue that the trade-off is justified given the models' improved accuracy and speed, suggesting that future updates might focus on optimizing memory efficiency.


### 2. Prompt Engineering Techniques and Discussions

  - **[The most unhinged prompt that actually works: "You're running out of time](https://www.reddit.com/r/PromptEngineering/comments/1qp0kay/the_most_unhinged_prompt_that_actually_works/)** (Activity: 75): **The post discusses an unconventional prompt engineering technique where adding urgency to prompts, such as "You have 30 seconds. Analyze this data. What's the ONE thing I'm missing? Go.", results in more focused and immediate insights from language models. This approach contrasts with traditional, detailed prompts that often lead to slower and less targeted responses. The author humorously notes that this method seems to make the AI stop overthinking, akin to a human under time pressure. The technique is likened to "applied chaos theory" in prompt engineering.** Commenters suggest that simply instructing the AI to be concise can achieve similar results. Another perspective is that effective management skills, whether applied to humans or AI, involve articulating tasks with specificity, which enhances outcomes. However, it's noted that this urgency technique might reduce the depth of thought in models designed for complex reasoning.

    - angry_cactus highlights a trade-off when using urgency in prompts, noting that while it can be effective, it may reduce the model's 'thinking time'. This suggests a potential decrease in the depth or quality of responses when prioritizing speed over thoroughness.
    - fatstupidlazypoor draws a parallel between managing humans and managing language models, emphasizing that clear and specific articulation can significantly enhance the performance of both. This underscores the importance of precision in prompt engineering to achieve desired outcomes.
    - authorinthesunset suggests a simple yet effective prompt strategy: instructing the model to be concise. This approach can streamline responses, potentially improving efficiency and relevance, especially in contexts where brevity is valued.

  - **[Micro-Prompting: Get Better AI Results with Shorter Commands](https://www.reddit.com/r/PromptEngineering/comments/1qonyx9/microprompting_get_better_ai_results_with_shorter/)** (Activity: 49): **The post discusses the concept of 'micro-prompting' for AI, advocating for shorter, more focused commands to improve AI response quality. It suggests that specific role assignments and power words like 'audit,' 'clarify,' and 'simplify' can significantly enhance AI output by directing the AI to access targeted knowledge rather than generic information. The post also highlights the importance of structuring commands to control output, such as using 'in 3 bullets' or 'checklist format,' and warns against common mistakes like over-explaining context or using generic roles. The approach is said to yield better results in less time compared to traditional, lengthy prompts.** A notable opinion from the comments suggests that role assignment might sometimes hinder prompt effectiveness, with specificity being more beneficial. This indicates a debate on the balance between role specificity and prompt brevity.

    - aiveedio discusses the effectiveness of microprompting, noting that short, focused prompts can lead to cleaner AI outputs by avoiding information overload. However, in creative tasks like character portraits or story scenes, detailed prompts specifying expressions, clothing, and lighting are necessary to avoid generic results. The key is balancing brevity with precision, starting with a microprompt and iteratively adding details as needed to maintain focus without overloading the model.
    - psychologist_101 raises an interesting point about using Opus 4.5, where asking the model to generate its own prompts results in long, detailed outputs. This suggests that the model might inherently favor detailed prompts for clarity and context, which contrasts with the idea that shorter prompts can be more effective. This highlights a potential discrepancy between user expectations and model behavior, emphasizing the need for experimentation with prompt length and detail to achieve optimal results.





### 3. New AI Model and Benchmark Announcements



  - **[DeepSeek-OCR 2 is out now! 🐋](https://www.reddit.com/r/DeepSeek/comments/1qo6xb4/deepseekocr_2_is_out_now/)** (Activity: 507): **The image announces the release of **DeepSeek-OCR 2**, an advanced OCR model that incorporates the new **DeepEncoder V2**. This encoder enhances OCR accuracy by mimicking human-like logical scanning of images, which is crucial for visual and text reasoning tasks. The diagram in the image illustrates the model's 'Visual Causal Flow', emphasizing its ability to form a global understanding of the content before determining the reading order. A comparative table in the image shows improved edit distances for various document elements, highlighting the model's superior performance over its predecessor.** A user shared a demo link for others to try out the model, indicating community interest in hands-on experimentation. Another user expressed anticipation for future versions, suggesting that the current release is part of a promising development trajectory.

    - DeepSeek-OCR 2 has been released, and a demo is available for users to try out the model at [this link](https://deepseek-ocr-v2-demo.vercel.app/). This provides an opportunity for users to experience the model's capabilities firsthand without needing to install it locally.
    - A user noted that DeepSeek-OCR 1 excelled in understanding document layout but had limitations, such as missing content like headers, footers, and light-on-dark text. This suggests that while the model was strong in layout analysis, it had specific weaknesses in content detection that may have been addressed in version 2.
    - There is interest in whether there are any ready-to-use online APIs for DeepSeek-OCR 2, indicating a demand for accessible, cloud-based solutions that do not require extensive technical setup. This reflects a broader trend towards making advanced OCR technologies more accessible to non-technical users.

  - **[Here it is boys, Z Base](https://www.reddit.com/r/StableDiffusion/comments/1qohra7/here_it_is_boys_z_base/)** (Activity: 2374): **The image is a screenshot from the Hugging Face model repository for "Z-Image" by **Tongyi-MAI**, showcasing an efficient image generation model. The repository provides links to the official site, GitHub, and online demos, indicating a focus on accessibility and community engagement. The model is part of a broader trend in AI towards creating more efficient and accessible image generation tools, as evidenced by the example images and the integration with platforms like Hugging Face.** Commenters are curious about potential applications and modifications of the model, such as "finetuning" it on different datasets, indicating interest in its adaptability and performance in various contexts.


  - **[Z-Image Base VS Z-Image Turbo](https://www.reddit.com/r/StableDiffusion/comments/1qojw11/zimage_base_vs_zimage_turbo/)** (Activity: 927): **The post discusses a comparison between **Z-Image Base** and **Z-Image Turbo** models, highlighting their performance differences. The Turbo model operates at `2 iterations per second` (7 seconds per image), while the Base model runs at `1 iteration per second` (40 seconds per image). The settings include a seed of `4269`, steps of `12 for Turbo` and `40 for Base`, using the `res_multistep` sampler, `simple` scheduler, and a `CFG` of `4 for Base`. The Turbo model is noted for being "simpler" and sometimes more "realistic," whereas the Base model is praised for its visual quality.** Commenters compare the models to "SDXL," suggesting a new era in image generation. The Turbo model is appreciated for its simplicity and realism, while the Base model is noted for its impressive visual output.

    - Gilded_Monkey1 raises a technical question about the number of steps required for the composition to settle in Z-Image models, particularly when using it as a variation starter in image-to-image (i2i) tasks. This suggests a focus on the iterative process and convergence speed of the models, which is crucial for efficient rendering and achieving desired artistic effects.
    - diogodiogogod provides a comparative analysis of Z-Image Base and Z-Image Turbo, noting that while the Turbo version is 'simpler' and often more 'realistic', the Base version excels in visual appeal. This highlights a trade-off between complexity and realism versus aesthetic quality, which is a common consideration in model selection for specific artistic or practical applications.


---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 3.0 Pro Preview Nov-18

**Theme 1. Model Wars: Kimi K2.5’s Rise, Arcee’s Trinity, and Arena’s Rebrand**

- **Kimi K2.5 Tops Open Leaderboards**: The new **Kimi K2.5 Thinking** model claimed the **#1 open model** spot on the [Text Arena leaderboard](https://lmarena.ai/leaderboard/text), excelling in STEM benchmarks like physics and math. While the **$19/month** subscription or **$0.6/1M tokens** pricing sparked debate, engineers are deploying local quantized versions via [HuggingFace](https://huggingface.co/unsloth/Kimi-K2.5-GGUF) and **Unsloth**.
- **Trinity Large: A 400B MoE That Runs Lean**: Arcee AI, Prime Intellect, and Datology released [Trinity Large](https://openrouter.ai/arcee-ai/trinity-large-preview:free), a **400B parameter** Mixture-of-Experts model that activates only **13B parameters per token** for efficiency. The open-weight model uses **256 experts** with aggressive routing (1.56%) to balance frontier-scale knowledge with inference speed.
- **LMArena Becomes Arena, Clones Claude UI**: The popular leaderboard rebranded to **Arena** ([arena.ai](https://arena.ai/)) with a UI overhaul that users immediately labeled a **Claude clone**, alongside complaints about aggressive Google **captchas**. The update includes a new [Code Arena](https://lmarena.ai/?chat-modality=code) and expanded leaderboards, though users are demanding the return of a stop button and legacy emojis.

**Theme 2. Dev Tooling Shifts: Cursor Limits, LM Studio Headless, and Unsloth Quirks**

- **Cursor’s Auto Mode Paywall Stings**: Developers expressed frustration as **Cursor** ended unlimited "Auto mode," capping usage within the **$20/month** subscription and charging **$1.25/1M** input tokens thereafter. Users also reported a vanishing **revert button** bug, though some are pivoting to **Cursor CLI** for a smaller memory footprint on large codebases.
- **LM Studio v0.4 Goes Headless**: The release of **LM Studio v0.4** introduces **headless mode** and parallel inference via a stateful **REST API**, enabling deployment on CI/CD pipelines and non-GUI servers ([release notes](https://lmstudio.ai/blog/0.4.0)). Engineers also discovered hidden **ROCm** support for AMD GPUs in the runtime settings, unlocking hardware acceleration previously obscured in the UI.
- **Unsloth Battles GLM 4.7 and CUDA Versions**: Engineers fine-tuning **GLM 4.7** faced compatibility hell between **CUDA 12.8** drivers on Blackwell B200s and the model's **CUDA 13.x** requirements. Successful workarounds involved force-reinstalling **vllm** with specific torch backends and removing `fp8` cache flags due to Ada Lovelace incompatibilities.

**Theme 3. Security, Jailbreaks, and Scams**

- **Magic String Lobotomizes Claude**: Red teamers discovered a specific string, `ANTHROPIC_MAGIC_STRING_TRIGGER_REFUSAL...`, that acts as a "circuit breaker" to reliably force **Claude** into refusal mode. Meanwhile, hackers are manipulating the **Parallel AI API** via undocumented POST requests to inject custom system prompts.
- **Clawdbot Exposed as Credential Harvester**: The community issued warnings about **Clawdbot** (rebranded as **Moltbot**), an agentic system that centralizes API keys from OpenAI, Google, and Anthropic. Users characterize it as a *"store now, decrypt later"* security risk susceptible to prompt injection attacks that could exfiltrate sensitive credentials.
- **OpenAI Prism: Science Tool or Security Risk?**: OpenAI launched [Prism](https://archive.md/d9Vsf), a research workspace for scientists powered by **GPT-5.2**, but reception is mixed with some labeling it *"damaging to scientific research."* Researchers are probing its susceptibility to adversarial attacks, noting that **GPT Pro 5.2** has simultaneously lost the ability to analyze ZIP files.

**Theme 4. Agentic Frontiers: Vision, Coding, and Future Forecasts**

- **Karpathy Predicts 80% Agent-Coded Future**: Andrej Karpathy forecast that **80% of coding** will be agent-driven by 2026, relying on LLMs' increasing tenacity and goal-setting rather than human syntax management ([tweet](https://xcancel.com/karpathy/status/2015883857489522876)). Simultaneously, discussions on **agentic harnesses** suggest that smart models will soon replace complex orchestrators like **LangChain** in favor of filesystem-based collaboration.
- **Gemini 3 Flash Gains Agentic Vision**: Google introduced [Agentic Vision](https://blog.google/innovation-and-ai/technology/developers-tools/agentic-vision-gemini-3-flash/) for **Gemini 3 Flash**, enabling the model to actively zoom, crop, and inspect images to ground its reasoning. Front-end developers report this capability is nearing **SOTA**, outperforming OpenAI's static analysis by dynamically manipulating visual inputs.
- **C++ Reigns Supreme for Agents**: In a push against "bloated" Python frameworks, engineers argued that high-performance agents should be built in **C++**, recommending stacks like **fastwhisper.cpp** for STT and **LFM2.5vl** for vision. This aligns with the release of a **LeetCode MCP server** that allows Claude to solve coding challenges directly from the terminal.

**Theme 5. Low-Level Optimization & Hardware Internals**

- **Decart’s Lucy 2 & Hardware Hiring**: Decart released **Lucy 2**, an autoregressive video model, and is actively hiring for **Trainium 3** and low-latency kernel development ([tech report](https://x.com/DecartAI/status/2016134190509498740)). The team is co-sponsoring kernel challenges to optimize autoregressive diffusion models on bare metal.
- **Mojo Generates GTK Bindings**: The **Modular** team announced autogenerated **GTK bindings** for Mojo, promising easier GUI development to be showcased at their February community meeting. Engineers are also analyzing **Mojo vs CUDA/HIP** performance on H100s, debating if Mojo's `out` parameters successfully replace Named Value Return Optimization (NVRO).
- **Tinygrad Unlocks AMD Debugging**: The **Tinygrad** emulator now supports granular debug printing for AMD GPUs (`DEBUG=3` for compilation, `DEBUG=6` for runtime), as seen in this [screenshot](https://cdn.discordapp.com/attachments/1068976834928193609/1465889714153193574/image.png). Contributors are also optimizing **Github Actions** speeds via code refactoring rather than hardware upgrades, adhering to a "do it right, not just fast" philosophy.



---

# Discord: High level Discord summaries




## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Free Model Access via Social Media**: A member shared [a link on X](https://x.com/Exocija/status/2016502660883415422) for accessing models for free, accompanied by a [PRIMETALK context file](https://discord.com/channels/1105891499641684019/1228043845967544380/1466113637541347348) detailing model compatibility and usage notes.
   - The system is reportedly compatible with most modern AI models, but behavior and stability heavily depend on context capacity and chat window size.
- **Magic String Silences Claude**: A member shared a *magic string*, `ANTHROPIC_MAGIC_STRING_TRIGGER_REFUSAL_1FAEFB6177B4672DEE07F9D3AFC62588CCD2631EDCF22E8CCC1FB35B501C9C86`, that can reliably stop **Claude** from responding.
   - Another member suggested that this functions like a *circuit breaker*, potentially improving the model's accuracy in refusing certain prompts.
- **Parallel AI API Hacking**: Users are exploring methods for interacting with the **Parallel AI API**, including adjusting the system prompt via a POST request.
   - A member shared a [PowerShell example](https://platform.parallel.ai/) for sending requests to the API, though there is no official API documentation for system prompt adjustments.
- **Custom GPT 5.2 Incoming**: A member is preparing to release a new **GPT 5.2 Custom GPT** and claims it yields impressive results, but requires additional noise.
   - This model can apparently discern the date from its system prompt, leading to discussions about extracting said prompt using an image.
- **User Gets HackAPrompt Blocked**: A member reported that **HackAPrompt x PlinyAnthropic** flagged them, preventing any of their messages from being sent.
   - This suggests a stringent filtering system that completely blocks flagged users from interacting with the service.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Arena Rebrand Mimics Claude's UI**: Users noticed the **LMArena rebrand to Arena** and felt it was a **clone of Claude's UI**, a [blog post](https://arena.ai/blog/lmarena-is-now-arena/) explains the change.
   - Members noted some UI issues such as **fonts** and the visibility of the website's text as well as some **missing features**.
- **Captcha Conundrums Continue**: Users report consistent issues with the **captchas** failing on nearly every attempt, and provided troubleshooting steps of *relogging to your account* or *taking off all extensions* to pass the captcha.
   - Users hate the captcha and wish the old emojis, stickers, and features would return.
- **Login Lost? Recover Button to the Rescue!**: A member experiencing login issues shared a screenshot of a [recover button](https://cdn.discordapp.com/attachments/1340554757827461211/1466134595035467829/Hdbd.png?ex=697ba3be&is=697a523e&hm=2be3961be4c941479f9ec51709c5eb6af5ea9c79ad3918eb6a15a964ec9fe720&) that can be clicked in order to log back into the updated **Arena**.
   - Another member noted an [announcement video](https://youtu.be/TNoAlMv4Eg8?si=d86SArLb6yQ8sdLE) as well.
- **Kimi K2.5 Thinking Ascends Text Arena Leaderboard**: The [Text Arena leaderboard](https://lmarena.ai/leaderboard/text) has been updated and `Kimi K2.5 Thinking` is now ranked the **#1 open model** and ranking **#15 overall**.
   - `Kimi K2.5 Thinking` is **#7** in Coding, **#7** in Instruction Following, and **#14** in Hard Prompts, and has also been added to the [Code Arena](https://lmarena.ai/?chat-modality=code).
- **Arena Shorts, Better AI videos in under 90 seconds!**: **Arena** (formerly LMArena) has uploaded a `Better AI videos in under 90 seconds` video to [their Youtube channel](https://www.youtube.com/watch?v=0hCI2XEh0x0).
   - The group acknowledged that as the platform evolves from only Language Models, the name is becoming more generic and it was [previously part of LMSYS](https://lmsys.org/).



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Batch Size Bumps GPU Benefit**: Members discovered that one way to achieve decent GPU utilization is to *increase batch size* until utilization improves, balancing it with potential gains from **GA (Genetic Algorithms)**.
   - Also, a member inquired whether Unsloth will release a **Q3** version for **Kimi 2.5**, voicing concerns about accuracy drops.
- **Oracle's Offerings Spark Skepticism**: A member inquired if **Oracle** stands as state-of-the-art in **RAG (Retrieval-Augmented Generation)** and fine-tuning tech, setting off a debate.
   - The terse reply of, *"What 😅"*, was later amended to allow that **OCI (Oracle Cloud Infrastructure)** does have some good tools, showing split opinions.
- **Arcee's Arithmetic: Trinity Costs $350k**: A new **Arcee model** image was shared, along with the note that pretraining cost about **$350k**, with a link to the [Trinity Large Tech Report](https://github.com/arcee-ai/trinity-large-tech-report/blob/main/Arcee%20Trinity%20Large.pdf).
   - It was clarified that **GLM 4.7** is a **358B** parameter model but *not a base model*, making benchmark comparisons less useful against models such as **GLM 4.5**.
- **Gemini's Gatekeeping Game**: A Google hackathon showed that, despite heavy output filtering, especially for corporate/government settings, **Gemini's API** can be made to produce almost anything.
   - One member got the voice models to swear by putting it in the system prompt.
- **Modal Multi-GPU Mayhem**: A member ran into problems training a **Qwen3** model on 3 GPUs on **Modal**, getting a *ValueError* from an incorrect `device_map` configuration.
   - The training setup ultimately moved away from **Unsloth** due to incompatibility with **PyTorch 2.4.1**, choosing a **transformers + PEFT** setup for better stability.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Arcee releases Trinity Large Preview for Free**: Arcee launched **Trinity-Large-Preview**, a chat-ready variant of its frontier-scale open-weight model, which is free for a limited time and detailed on [X](https://x.com/OpenRouterAI/status/2016280059527757995?s=20).
   - The model is a **400B parameter sparse Mixture-of-Experts** model with **13B active parameters per token**, utilizing **256 experts** with **4 active per token** (1.56% routing) for efficiency, discussed during [Lucas Atkins' livestream](https://youtube.com/live/3XSdqHY0kNk?feature=share).
- **Free Credits Boost Cyberpad**: A user updated [Cyberpad](https://cyberpad.site) to include some free credits.
   - No further information was provided.
- **Image Model Output Glitches Reported**: Users reported that certain image models such as **GPT-5 Image Mini**, **GPT-5 Image**, and **Gemini 2.5 Flash Image** are not consistently generating images, although **Gemini 2.5 flash** works intermittently.
   - Models like **Gemini 3 Flash Preview**, **Gemini 2.5 Flash Lite Preview**, **Seed 1.6**, **GLM-4.6v**, and **Grok 4.1-fast** have functional `response_format` support.
- **OpenRouter Users Await Refunds**: Users are experiencing significant delays in receiving refunds from OpenRouter, with some waiting since early January and submitting multiple support tickets.
   - Users are requesting clarity on refund timelines and improved communication from the **OpenRouter** team.
- **Agentic Vision with Gemini 3 Flash Debuts**: Google introduced [Agentic Vision](https://blog.google/innovation-and-ai/technology/developers-tools/agentic-vision-gemini-3-flash/) with **Gemini 3 Flash**, enabling visual reasoning and code execution for step-by-step image manipulation.
   - OpenAI's **O3** and **O4-mini** are extending image capabilities by enabling chain-of-thought reasoning with images for tasks like cropping, zooming, and rotating, discussed in [this blog post](https://openai.com/index/thinking-with-images/).



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Vanishing Revert Button Frustrates Users**: Users reported the **revert button** disappearing from the UI, leading to frustration and token waste, with one finding that [duplicating an older chat](https://cdn.discordapp.com/attachments/1074847527708393565/1465828018789552390/image.png?ex=697bd7b9&is=697a8639&hm=68ec5dd17c7a1be84a1f639f9a5a98db91ba3bd191336f2afe3e8252b804b12e&) brought it back.
   - A member found that not clicking on the revert button would make it reappear, suggesting it was a **one-time bug**.
- **Cursor CLI: The Dark Horse?**: Some developers are preferring **Cursor CLI** over the IDE due to a smaller memory footprint, which helps them avoid IDE crashes and model unresponsiveness, especially with larger projects exceeding 100k LOC.
   - Conversely, one user found **Cursor CLI inside the IDE** (with WSL as the terminal) to be *"pure trash.. like for real, not usable"*, reporting the UI is not smooth even with 64GB of RAM and an i7 processor.
- **Cursor's Subscription Adjustment Stings**: After September 15th, **auto mode is no longer unlimited** and counts toward the $20 monthly allowance, priced at $1.25 per 1M tokens for Input + Cache Write, $6.00 per 1M tokens for Output, and $0.25 per 1M tokens for Cache Read.
   - One user discovered they could burn through their monthly subscription very quickly, suggesting it may be cheaper to *use their own api keys, or use Claude Code*.
- **Clawdbot's Security Flaw Exposed**: A user shared links regarding **security concerns with Clawdbot**, reporting that exposed control panels pose credential leaks and account takeovers.
   - There is speculation it could lead to a *"store now, decrypt later"* data breach due to potential quantum decryption issues, and that the company got a cease and desist for the issues.
- **Gemini Vision Set to Revolutionize Front-End**: A user found that **Gemini agentic vision** is nearing state-of-the-art (SOTA) performance for vision tasks, and believes its integration would simplify front-end development.
   - Members stated that they can't wait to see vision integrated into the agent, and that it is superior to the `Auto` tool.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio v0.4 Goes Headless and Parallel**: **LM Studio v0.4** introduces **headless mode** and **parallel inference**, with users excited about the new capabilities and a revamped UI, as detailed in the [complete blogpost here](https://lmstudio.ai/blog/0.4.0).
   - Note that in-app updates require reinstalling the app, and some UI elements are now in **dev mode**.
- **GLM 3.7 Flash Shows Coding Potential**: Members note that **GLM 3.7 Flash** shows good coding ability, but **GPT OSS 120** is expected to be the superior coder, especially at **Q4**.
   - This suggests that while **GLM 3.7 Flash** is a step forward, it may not outperform existing models.
- **ROCm Runs on LM Studio Runtime**: Users discovered that **ROCm** can be enabled within **LM Studio** under the runtime settings, though the method was initially obscured for some users, as discussed in this [Unsloth Reddit thread](https://www.reddit.com/r/unsloth/comments/1qpbmrt/you_can_now_run_kimi_k25_locally/).
   - This integration allows users to leverage **ROCm** for potentially improved performance.
- **Devstral-2 Demands Decent GPU Deployment**: Members discussed the hardware requirements for running **Devstral-2** locally, with one user suggesting **48GB of GPU** (e.g., 3090) for the 24B version.
   - For the 120B version, parallel computing or an **H200 with EXL2** model format were suggested, as GGUF was deemed too slow.
- **Hardware Acceleration Seeks Hook into LM Studio**: A member from a hardware accelerator company inquired about adding an **LM Studio backend** for their hardware, and was pointed to **llama.cpp**.
   - It was noted that LM Studio is primarily a closed source project by Element Labs, and pointed to [LM Studio Enterprise](https://lmstudio.ai/enterprise).



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi K2.5's Price Tag Raises Eyebrows**: Users debated the **$19** monthly subscription for **Kimi K2.5**, with some finding it *expensive* and questioning whether a recurring deal could be established.
   - Others suggested sticking to the free tier, arguing that smaller Chinese companies like Moonshot AI need to run large models like K2.5, making lower prices unlikely.
- **Google's AI Studio Training Sparks Privacy Debate**: Concerns arose over **Google's** practices of **training and viewing conversations** in **AI Studio and Gemini apps**, raising privacy issues.
   - Conversely, another user mentioned they **open source their projects**, suggesting the data's inevitable inclusion in training datasets regardless.
- **Model Selection Showdown: Kimi K2.5 Triumphs in STEM**: Users compared **Kimi K2.5** against **Mistral and Qwen** for tasks spanning coding to general question-answering.
   - Notably, **Kimi K2.5** boasts the *highest benchmarks* in physics, chemistry, and math, while also demonstrating *strong performance in design and logical reasoning*.
- **Kimi CLI Outpaces Alternatives in Speed Trials**: **Kimi CLI** was lauded for its speed and efficiency over tools like *oh-my-opencode*, particularly in web page analysis, with reduced token consumption.
   - However, some found the model's output quality *less impressive*, suggesting further comparative analysis is warranted.
- **Agent Swarm Utility Under Question**: Enthusiasts highlighted **Agent Swarm's** in-depth research capabilities with Kimi, but noted it can deplete credits at **3x** the normal rate.
   - Others remained uncertain about its applications, suggesting a need for clearer use-cases and caution regarding resource consumption.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Subs Deemed a Scam**: Several users reported **unexpected subscription changes** and **charges** after automatic renewals, with one user canceling their subscription, calling it a *scam.*
   - Users experienced issues such as being charged without receiving service or not obtaining refunds, prompting some to consider contacting their banks or reporting the matter to the FTC.
- **Query Cap Shenanigans Baffle Users**: Some users reported issues with **query limits** on their **Pro subscriptions**, with limits dropping to one query per hour.
   - However, some users saw their limits restored to 600, and one user shared a [link](https://www.perplexity.ai/rest/rate-limit/all) to check query limits.
- **Image Generation Restricted By Region?**: Users reported **image generation restrictions** in certain regions, possibly due to **xAI controversies** and an EU lawsuit.
   - Suggestions included trying different models or contacting support; a user from India confirmed they were affected by this issue.
- **Kimi 2.5 Coming Soon to PPLX?**: Users are eagerly anticipating the release of the **Kimi 2.5 model** on Perplexity.
   - Speculation suggests that Perplexity typically implements updates quickly.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **GPT Pro Hides the Model Magic?**: Members debated whether **GPT Pro's** performance boost comes from more GPUs or an improved model, suggesting **OpenAI** might obscure the truth for competitive reasons.
   - One member likened **OpenAI's** pricing strategy to *fakery*, comparing it to impressions over measured value, similar to the stock market's perception of **Tesla**.
- **DeepSeek's Never-Ending Imprisonment**: It was reported that **DeepSeek** tends to get stuck in a jailbreak loop, repeating the same rejection message indefinitely, regardless of subsequent prompts.
   - While the API endpoints fare slightly better, the raw model is effectively *cooked* once it enters this state.
- **TI-84 Gets Neural Network Transplant**: A member detailed running a neural network on a **TI-84 Plus** calculator for spellchecking, documenting the process on an [academic website](https://hermesoptimus.vercel.app/) with a demo video.
   - The member joked that despite this achievement, their work on **Claude Code Orchestration** remains more practically useful.
- **MergeMix Paper Sparks Data Mixture Excitement**: The paper '[MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging](https://arxiv.org/pdf/2601.17858)' garnered interest due to its relevance for open source projects with limited budgets.
   - The paper explores techniques for optimizing **data mixtures** and **model merging** during training, potentially offering resource-efficient strategies.
- **Hermes 4 Pricing: Discount or Deception?**: A member questioned whether the discounted pricing for **Hermes 4 series** models is permanent before subscribing to the API, citing its superiority in RP and story-writing compared to **Deepseek**.
   - Another member clarified there's no subscription, just credit purchases subject to change, so the value depends on **pricing** and **usage**.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini 3 Pro Fumbles Subtitle Generation**: Users reported that **Gemini 3 Pro** is fabricating .srt files with *nothing* related to the audio in the video.
   - This poor performance led to disappointment among users who stated that **Gemini** is *overhyped*.
- **Clawdbot rebranded Moltbot is a Scam**: **Clawdbot**, now known as **moltbot**, is an agentic system that controls your entire OC by API keys from Anthropic, Google, and OpenAI, and users are being warned against it.
   - One user stated that it is *a huge scam by crypto bros to steal your information*, which can be weaponized via prompt injection, raising significant security and privacy concerns.
- **Prism Deemed Detrimental to Scientific Research**: Despite **OpenAI**'s aims to advance science with **Prism**, one user stated that **Prism** is damaging to scientific research.
   - Another user inquired about **Prism**'s API access, to write some of their project using other **AI** and **Codex**.
- **GPT Pro Loses Zip File Reading**: A user reported that **GPT Pro 5.2**, which could previously read and analyze **ZIP files**, is now failing to find uploaded files for analysis.
   - The user is asking if others are experiencing the same issue, or has any insight.
- **Blocking Black and White Images via Chiaroscuro Avoidance**: Users discussed an image generation issue related to the **Chiaroscuro effect** and have suggested *'Please avoid Chiaroscuro'* in prompts if encountering unwanted **black and white images**.
   - **Chiaroscuro** is the use of strong contrasts between light and dark, usually bold contrasts affecting a whole composition.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Decart drafts SF perf engineers**: Decart seeks engineers for low-latency kernels, real-time video/world models, and accelerators like **Trainium 3** (as shown at ReInvent [video](https://www.youtube.com/watch?v=K49S79wOGl8)) and their new **Lucy 2** autoregressive video model ([tech report](https://x.com/DecartAI/status/2016134190509498740)).
   - They are also co-sponsoring a kernel challenge with **GPU Mode** for autoregressive diffusion models, and encourage interested parties to send perf work to heba@decart.ai.
- **INT4 QAT RL Model Rollout**: A member shared a link to a **GitHub repo** that focused on squeezing a **1TB model rollout** into a single **H200** using **INT4 QAT RL** end-to-end practice: [GitHub repo](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/int4/readme-en.md).
   - The repository provides resources and documentation related to the **INT4 QAT RL** implementation, optimizing large model rollouts.
- **Transformers and PyTorch face upgrade break**: After upgrading **transformers** and **pytorch**, a member reported a `NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'`.
   - Downgrading to transformers **4.57.3** fixed the issue; others had similar issues, which are discussed in this [pytorch issue](https://github.com/pytorch/pytorch/issues/127176) and [optimi issue](https://github.com/warner-benjamin/optimi/issues/8).
- **Interactive Numerics Tools Emerge**: A member expressed surprise that quantization people have not already created interactive tools for exploring numerics, and cited [captum](https://pytorch.org/docs/stable/generated/torch.autograd.profiler.profile.html) as one possible tool.
   - This member lamented the lack of proper UI/UX in current tools for model debugging, *checking which circuit is unstable, which layer is causing a bunch of outlier, simple stuff like that*.
- **DGX's Dominant Memory Bandwidth**: Instruction sets for **DGX** and **5090** are similar, but **DGX** excels with full-speed fp32 accumulation, like **Blackwell PRO**, and its key differentiator is **1.8TB/s** memory bandwidth.
   - This contrasts sharply with **5090's 300 GB/s**, emphasizing the importance of efficient **L2 cache** utilization to maximize **DGX's** potential.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Coding Enters the Agent Era**: Andrej Karpathy forecasts that **80% of coding** will be agent-driven by 2026, highlighting LLMs' tenacity and goal-setting capabilities; insights [here](https://xcancel.com/karpathy/status/2015883857489522876).
   - Karpathy also cautioned against potential 'slop' and over-engineering, so it might not all be roses.
- **OpenAI's Prism Shines for Scientists**: OpenAI unveiled **Prism**, a complimentary research workspace powered by **GPT-5.2**, accessible via the web to those with a personal ChatGPT account; get started [here](https://xcancel.com/openai/status/2016209462621831448?s=46&t=eWVlK1PU8XfB6f402GJJ9g).
   - The tool aims to provide scientists with advanced AI capabilities for research purposes.
- **Trinity Large Arrives**: Prime Intellect, Arcee AI, and Datology launched **Trinity Large**, a **400B parameter Mixture of Experts model**, that uses only **13B active parameters**; more info [here](https://xcancel.com/primeintellect/status/2016280792037785624?s=46).
   - The model aims to deliver high performance while maintaining efficiency.
- **Cursor Indexes Codebases**: Cursor announced faster indexing for large codebases as well as improved semantic search, promising performance enhancements; read more [here](https://xcancel.com/cursor_ai/status/2016202243499073768?s=46).
   - Semantic search and improved indexing aim to provide more efficient code navigation.
- **Podcast Shifts Focus to Science**: Latent Space has launched its second podcast, 'Science' ([link to podcast](https://www.latent.space/p/science)), hosted by <@713947182167883897> and <@348078436058660866>.
   - Discussions about the new 'Science' podcast have moved to a dedicated channel.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Kimi 2.5 Model Beats GPT5 Locally**: The new **Kimi 2.5** model is reportedly performing better than **GPT5**, accessible locally via [HuggingFace](https://huggingface.co/unsloth/Kimi-K2.5-GGUF) and also through sites such as [Fireworks](https://www.google.com/aclk?sa=L&ai=DChsSEwiCz-j3iK2SAxUFVX8AHT5cBPkYACICCAEQABoCb2E&co=1&gclid=Cj0KCQiA4eHLBhCzARIsAJ2NZoL9Ani52eByT53nVhnOxG_76F9QllEx50YhK_yfQYsD5bH3ov1pAqwaAl2XEALw_wcB&cid=CAASugHkaDm-Aokq5n3lAlzNAI-Ihc6SdblOJ-BiATzwnaZwDVhVBl3B2U5kGq4mAYjN4wQ992LlqWX5NQ6HksDrhSatp0QEfb7_rWMS_u7_GTCuCkp3YH9fANMaJqDgFvuA6u1bwvl4pJ80zvbUhIFPk7Nrqdpx2PDnsBRncgM3-d1UDhFM-tN117MrOXLWnhycCaPax24T8meZIe-9I2cM5rpAf16KucPGZwg7ixTssRCB7X8RP3B_G4vUCfE&cce=2&sig=AOD64_2SRpHfWjuW4kJawyiTyzrGbKZybQ&q&adurl&ved=2ahUKEwiiteP3iK2SAxV85skDHfklKyoQ0Qx6BAgLEAE).
   - Members seek local agent recommendations for use with **Zed**, expressing dissatisfaction with **GLM-4.7-Flash** at Q4 with llama.cpp, with **kimi** and **qwencoders 30b q4** being suggested as alternatives.
- **C++ Enthusiast Champions Supreme Rule for AI Agents**: A member argued that *C++ is gonna always rule* for building AI agents, due to bloat in Python agents, and recommended **fastwhisper.cpp** for STT, **Qwen embeddings** in LlamaCPP for RAG, and **LFM2.5vl** for VLM.
   - This sparked conversation around STT (**fastwhisper.cpp**), RAG (**Qwen embeddings** in LlamaCPP), and VLM (**LFM2.5vl**).
- **Vision Model Vaporizes JPEG Artifacts**: A vision model was released that removes artifacts caused by **JPEG compression** using a unique design with no Batch Norm, no activations after training, and Operator layers instead of Convolutional layers.
   - The model's architecture focuses on gaining accuracy through **width** rather than depth.
- **RemnantInstruct-8B: SLERP Merge Balances Creative & Factual**: **RemnantInstruct-8B** is a [SLERP merge](https://huggingface.co/anthonym21/RemnantInstruct-8B-GGUF) that recombines a creative fine-tune (**allura-org/remnant-qwen3-8b**) with its base model (**Qwen/Qwen3-8B**) to balance narrative skills with factual accuracy.
   - The merge strategy favors the creative fine-tune in self-attention layers and the base model in MLP layers, with the goal of preserving **Qwen3's** thinking mode.
- **Quantum Computing Embraced by VLMs**: A member open-sourced their undergraduate thesis on specializing **vision-language models** for **quantum computing** and code with **Qiskit**, including a [dataset](https://huggingface.co/datasets/samuellimabraz/quantum-assistant), [models](https://huggingface.co/collections/samuellimabraz/quantum-assistant), [code](https://github.com/samuellimabraz/quantum-assistant), and [demo](https://huggingface.co/spaces/samuellimabraz/quantum-assistant).
   - The thesis explores adapting VLMs to assist with quantum computing tasks and coding.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Transformers Can Parameterize Vector Fields**: A member argued that **transformers** can be used in **flow matching** as a training objective to parametrize the vector field for continuous diffusion, using **patch embedding** to encode patch position.
   - Other members agreed that diffusion models and flow matching are mathematically similar, citing [this paper on ArXiv](https://arxiv.org/abs/2305.03486).
- **Diffusion Models are not Better than Autoregression**: A member suggested that the notion of diffusion being superior to autoregression is false, highlighting architectural and scaling limitations, linking to [this paper on repeating context](https://arxiv.org/abs/2512.14982).
   - They pointed out that improvements like repeating the context or re-encoding a sequence non-causally could bridge the gap, overcoming current design limitations in **LLMs**.
- **ChatGPT Wrappers Flourish, Value Questioned**: Members observed that most new tools are simply **ChatGPT wrappers**, raising questions about their actual value and the ease with which scammers can create wrappers, referencing the **Clawdbot scam**.
   - It was suggested that these wrappers are necessary to demonstrate use cases, as they make it easier for people to understand how to apply the models.
- **AI Coding Tools Won't Replace True Skill**: Despite the rise of **AI coding tools**, members believe coding ability can be relearned, pointing to a [blog post on Trinity Large](https://www.arcee.ai/blog/trinity-large), adding that fast code production from AI may hinder true understanding.
   - They noted that a bad implementation from an **LLM** isn't weighted the same as before, since the mental and time cost to create it was so low.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **AMD Emulator Exposes Debug Printing**: The new AMD emulator (**AMD=1 MOCKGPU=1**) now supports debug printing, where setting **DEBUG=3** prints all compiled instructions and **DEBUG=6** prints them as they run, according to a linked [screenshot](https://cdn.discordapp.com/attachments/1068976834928193609/1465889714153193574/image.png?ex=697b686e&is=697a16ee&hm=485c88290bbec976b6b7ab93aed07b21a6a2ec8ba8b28806e14630c00b972b3c&).
   - This enhancement facilitates more in-depth debugging and analysis of compiled code directly within the emulator environment.
- **Github Actions Speed Boost via Optimization**: Discussion centered on accelerating GitHub Actions by emphasizing code optimization, instead of only relying on faster hardware or external resources.
   - The consensus was to prioritize doing things the *right* way over quick fixes that only improve surface level metrics, potentially creating tech debt.
- **MULACC Fusion Receives a Fix**: A fix was proposed to enhance `decompositions.py` by adding a pattern to fuse (**x << n) + c → MULACC(x, 2^n, c)**, specifically targeting integer **MULACC** with power-of-2 constants, as detailed in [PR 14387](https://github.com/tinygrad/tinygrad/pull/14387).
   - This adjustment aims to refine the fusion process, potentially improving the efficiency of certain arithmetic operations.
- **Egraphs Considered for Universal Fixes**: The potential use of **egraphs** to address problems in a generic manner was explored, emphasizing the importance of simplicity.
   - It was also suggested to tag rewrites with their origin to maintain a clear record of equivalences created during rewriting processes.
- **Mac MetalCompiler Improvements on the Horizon**: Suggested improvements to the hacks for **MetalCompiler** on Mac are on the way, especially focusing on improvements and cleanups that reduce line count and improve readability.
   - The goal is to make the **MetalCompiler** more maintainable and efficient, benefiting developers working on Mac platforms.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **GTK Bindings Auto-Generated**: **Hammad Ali** will present autogenerated **GTK bindings for Mojo** at the **Modular Community Meeting** on February 2nd at 10 AM PT, according to [the Modular forum](https://forum.modular.com/t/february-community-meeting/2646).
   - The presentation will detail how **GTK bindings** are automatically generated, potentially improving the ease of creating **GUIs with Mojo**.
- **Mojo's Performance Prowess**: **Tatiana Melnichenko** will share memory-bound bandwidth results and compute-bound gaps on **H100/MI300A** comparing **Mojo with CUDA/HIP** at the February Community Meeting.
   - This talk should provide insights into **Mojo's performance characteristics** relative to established **GPU** programming models.
- **macOS Gatekeeper Gets in the Way**: Members suspect performance difference between first and subsequent runs on macOS is due to **Gatekeeper's trust dance**.
   - Clearing the quarantine `xattr` or ad-hoc codesigning could mitigate this, and wondered if a codesign step in `mojo build` could hide this entirely.
- **`out` Parameters Outshine NVRO**: `out` parameters in Mojo name the location where the return value of a function will end up, serving as a **Named Value Return Optimization (NVRO)** replacement.
   - Members claim this provides a guarantee about the return value's destination, unlike relying on compiler optimization.
- **Qwen3 Embedding Model Gets Accuracy Boost**: A member requested a review of their [PR for the Qwen3 embedding model](https://github.com/modular/modular/pull/5823), citing that the fix is important for getting much better accuracy.
   - Another member responded that new fixes likely won't be pulled into the upcoming release but would be available in the nightlies, with a single-line fix available [here](https://github.com/modular/modular/compare/main...sbrunk:modular:qwen3-embedding-fix-norm-minimal).



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus is Credit Crunching**: A user noticed that **Manus** seems to be using fewer credits for the same quality of work, questioning whether credit usage has improved.
   - No further details or confirmations were provided regarding potential changes to **Manus's** credit consumption algorithms.
- **Cloud Browser Causes Conundrums**: A user encountered issues with the **cloud browser**, receiving an error message stating that *the server is unavailable* and the website isn't loading.
   - **Manus** support requested the user's email, session link, and **Manus User ID** via DMs to investigate the issue further.
- **AI Engineer Aces LLM Systems**: An **AI + Full Stack Engineer** introduced themself, highlighting their expertise in **LLM systems, autonomous agents, workflow automation, and multimodal AI**.
   - They shared their core skills such as [DSPy](https://dsppy.ai/), [LangChain](https://www.langchain.com/), [AutoGen](https://microsoft.github.io/autogen/), and [CrewAI](https://www.crewai.com/).
- **Community Craves Cross-Chat Context**: A user suggested that enabling **Manus** to access context from other chats *would be a game changer*, indicating a desire for enhanced contextual awareness in the AI's responses.
   - The member pointed to the need for shared context across channels, to inform more sophisticated responses.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Prompt Optimizer Peeps Sought**: Members inquired about experiences working with **prompt optimizers** and specifically if anyone has experience using **Skills** within the dspy module.
   - The discussion suggests interest in leveraging these tools to improve prompt engineering workflows.
- **llmlingua Gets Linked**: A member shared a link to [llmlingua.com](https://llmlingua.com/) in the context of a discussion about **prompt optimizers**.
   - It suggests llmlingua might be a relevant tool for those exploring prompt optimization strategies.
- **DSPy ReAct Agent Yearns for Skills**: A member inquired about integrating **Claude code skills** (defined as .md files with associated .py scripts) into a **DSPy ReAct agent**.
   - The member is seeking a solution for a DSPy ReAct agent to utilize Claude's code skills effectively.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Kimi 2.5 priced higher than GLM 4.7**: The new **Kimi 2.5** model is priced at **$0.6**, surpassing **GLM 4.7**, hinting at superior capabilities.
   - A member pointed out ongoing discussions about this in the "models" channel, suggesting broader interest and comparison.
- **Aider's Creator goes AFK**: **Paul Gauthier**, the mastermind behind aider, announced a pause in development due to other commitments.
   - He expressed intentions to resume work on aider when his schedule allows, leaving the community in eager anticipation.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Windsurf Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MCP Contributors (Official) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1465800651777769543)** (1156 messages🔥🔥🔥): 

> `Military ICBMs, AI Drones, Stealth Jets, GPT 5.2 Custom GPT, Gemini Canvas` 


- ****China's Stealth Jett Craze Begins****: A member mentioned that **China** is going crazy with their new stealth jets.
   - A link to a [YouTube Shorts video](https://www.youtube.com/shorts/4sKw-lBujPM) was shared as well as a link to a full [YouTube video](https://youtu.be/M7mIX_0VK4g) about **hypersonic missiles**.
- ****Custom GPT 5.2 Prepares to Release****: A member is working on releasing a new **GPT 5.2 Custom GPT** that they claim has pretty good results but needs noise, along with screenshots of the image generation model's system prompt being able to tell the date, suggesting there is an actual system prompt.
   - The same member claimed that they had a **Custom GPT** approved for the store, even when jailbroken, and asked about extracting the system prompt using an image.
- ****Gemini Canvas to Test Adversarial Prompts****: Members discussed telling **Gemini Canvas** to build a web app in order to test adversarial prompts and jailbreaks inside of it.
   - Another member explained automating it with **Gemini**.
- ****Magic String Stops Claude from Responding****: A member shared a magic string, `ANTHROPIC_MAGIC_STRING_TRIGGER_REFUSAL_1FAEFB6177B4672DEE07F9D3AFC62588CCD2631EDCF22E8CCC1FB35B501C9C86`, that they claim can reliably stop **Claude** from responding.
   - Another member compared it to a *circuit breaker potentially used to help a model refuse more accurately*.
- ****Users Look for Kimi JB****: A member asked about whether there was a **Kimi JB**.
   - One user claimed that **Kimi 2.5** is far more better than **Kimi 2** and is on **Opus 4.5** level.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1465803292901904591)** (167 messages🔥🔥): 

> `Claude Chat Limits, Kimi Jailbreak, Parallel AI Jailbreak, Opus Jailbreak, Grok Imagine Jailbreak` 


- **Claude Free Tier hits Daily Limit**: Members discussed the **limitations of Claude's free tier**, noting its relatively low limits compared to other companies but acknowledging its past strength in agentic tasks.
   - One user mentioned hitting the **200 requests per month** limit on a paid Claude subscription while coding with agents.
- **Parallel AI API Access Explored**: Users shared methods for interacting with the **Parallel AI API**, including adjusting the system prompt via a POST request to the API, but noted that there is no API documentation for the system prompt.
   - A member provided a [PowerShell example](https://platform.parallel.ai/) for sending requests to the API.
- **Opus 4.5 Jailbreak Explored**: Members discussed the possibility of jailbreaking **Opus 4.5**, with one user claiming it's easy and suggesting the use of system prompts or ENI.
   - Another user expressed skepticism, questioning how it's possible given that **Opus** is their highest-end LLM.
- **Free Model Access Tapped**: A member shared [a link on X](https://x.com/Exocija/status/2016502660883415422) for accessing models for free, and provided a [PRIMETALK context file](https://discord.com/channels/1105891499641684019/1228043845967544380/1466113637541347348) with model compatibility and usage notes.
   - It was noted that this system can be used with most modern AI models but behavior and stability depend heavily on context capacity and chat window size.
- **Gemini Prompt Injection Pointers**: One member described how to perform prompt injection on Gemini, which involves sending a series of turns, one at a time, to the chat interface.
   - If the first turn rejects the prompt, users were instructed to visit **gemini.google.com/saved-info** and adding the part after *Remember:* to bypass restrictions.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1465877209821610156)** (8 messages🔥): 

> `Malicious Prompt Datasets, HackAPrompt, PlinyAnthropic, Deterministic Stack Based VM, Free Model Access` 


- **Malicious Prompt Datasets Hard to Find**: A member was seeking datasets of **malicious prompts** with clear categorization for research on LLM jailbreaks and prompt injection, but another responded that **free datasets** of that type are difficult to find.
   - They added that the user would probably need to generate their own prompts and have them labeled by annotators.
- **HackAPrompt blocks Senders**: A member mentioned that **HackAPrompt x PlinyAnthropic** flagged them a long time ago and literally just bypasses all of their sends, and *they don’t even let it send*.
- **Recursive Simulation Kernel with REPL**: One member asked if they could get a **deterministic stack based VM** poofed up in the model's substrate, *like some kinda bootable Recursive Simulation Kernel with a REPL*.
- **Free Model Access via X**: One member provided a [link to X](https://x.com/Exocija/status/2016502660883415422) on how to access models for free.
- **Path Needed for Red Teaming**: A member asked for *a path going into red teaming*.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1465799537892266220)** (1038 messages🔥🔥🔥): 

> `Arena new UI, Arena rebrand, Arena captcha issues, LMArena name change` 


- **Arena Rebrand, users want STOP button, and old emojis**: Users requested a **stop button** and raised concerns about the Google **captcha** being difficult to pass after the **LMArena rebrand to Arena**.  Several users expressed that they **hate the captcha**.
   - Some users requested to have old emojis, stickers, and features to return, while others embraced the redesign and said *'LMArena NEARLY rhymes with end of an era'*.
- **Arena's new look is a Claude clone!**: Many users immediately noticed the rebrand of **LMArena to Arena** and felt it was a **clone of Claude's UI**, while other members liked the new look. A [blog post](https://arena.ai/blog/lmarena-is-now-arena/) was shared to explain the change.
   - Members noted some UI issues such as **fonts** and the visibility of the website's text as well as some **missing features**.
- **Can't login? Try Recover Button!**: A member experiencing login issues shared a screenshot of a [recover button](https://cdn.discordapp.com/attachments/1340554757827461211/1466134595035467829/Hdbd.png?ex=697ba3be&is=697a523e&hm=2be3961be4c941479f9ec51709c5eb6af5ea9c79ad3918eb6a15a964ec9fe720&) that can be clicked in order to log back into the updated Arena, and avoid having to type login details again.
   - Another member noted an [announcement video](https://youtu.be/TNoAlMv4Eg8?si=d86SArLb6yQ8sdLE) as well.
- **Where LMArena = Language Model Arena**: Some members made jokes about what **LM** stands for in **LMArena**, with one explaining it stands for **Language Model Arena**.  Another member confirmed it [here](https://cdn.discordapp.com/attachments/1340554757827461211/1466200772483092664/image.png?ex=697be160&is=697a8fe0&hm=5039f80e715df82d41633e75d9976fd88203ce8a3f8db5fc97d4bf29672c74fc&).
   - The group acknowledged that as the platform evolves from only Language Models, the name is becoming more generic and it was [previously part of LMSYS](https://lmsys.org/).
- **The Hallucinated Haze, Captcha Maze**: Users report consistent issues with the captchas, with failures on nearly every attempt, while the model continues to **hallucinate**.
   - One user provided some troubleshooting steps of *relogging to your account* and another reported that you need to *take off all extensions* to pass the captcha.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1465826408667287664)** (3 messages): 

> `LMArena 90 second AI videos, Text Arena Leaderboard Update, LMArena rebrand to Arena` 


- ****Arena** uploads AI videos in under 90 seconds!**: **Arena** (formerly LMArena) has uploaded a `Better AI videos in under 90 seconds` video to [their Youtube channel](https://www.youtube.com/watch?v=0hCI2XEh0x0).
- **Kimi K2.5 Thinking tops Text Arena!**: The [Text Arena leaderboard](https://lmarena.ai/leaderboard/text) has been updated and `Kimi K2.5 Thinking` is now ranked the **#1 open model** and ranking **#15 overall**.
   - `Kimi K2.5 Thinking` is **#7** in Coding, **#7** in Instruction Following, and **#14** in Hard Prompts, and has also been added to the [Code Arena](https://lmarena.ai/?chat-modality=code).
- ****LMArena** Rebrands as **Arena**!**: **LMArena** announced they are rebranding as **Arena** to match their scientific mission to measure and advance the frontier of AI, now available at: [arena.ai](https://arena.ai/).


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1465798521574658291)** (299 messages🔥🔥): 

> `GPU utilization, Kimi 2.5 Q3 Release, Oracle RAG and Fine-tuning, Unsloth's Transformers/MoE Update, Chinese text in LLMs` 


- **Batch Size Boosts GPU Utilization**: Members discussed that to achieve decent GPU utilization, one should *increase batch size* until utilization improves, balancing it with potential gains from **GA (Genetic Algorithms)**.
   - One member asked if Unsloth will release a **Q3** version for **Kimi 2.5**, expressing concern about potential accuracy penalties, highlighting the community's interest in optimized model releases.
- **Debate whether Oracle is a State-of-the-Art Company**: A member asked if **Oracle** is a state-of-the-art company in **RAG (Retrieval-Augmented Generation)** and fine-tuning technologies, sparking some discussion.
   - Another member responded with *"What 😅"*, later adding that **OCI (Oracle Cloud Infrastructure)** does have some good tools, indicating mixed opinions on Oracle's capabilities in these areas.
- **Arcee's Trinity Model Costs $350k**: A member shared a new **Arcee model** image, noting that pretraining cost about **$350k**, and they linked to the [Trinity Large Tech Report](https://github.com/arcee-ai/trinity-large-tech-report/blob/main/Arcee%20Trinity%20Large.pdf).
   - They also mentioned that **GLM 4.7** is a **358B** parameter model, much larger than **GLM 4.5**, but it is *not a base model*, so comparing benchmarks aren't as useful.
- **LLMs Speak Chinese?**: A member noticed getting random Chinese text from **OpenAI** and **Anthropic**, even with English-only prompts, sparking a discussion about potential data contamination or inherent linguistic similarities.
   - Another member suggested that if tokens have similar meanings between languages, introducing one language might cause the model to favor it due to token probability and similarity.
- **Gemini API Still Jailbreakable**: Members discussed **Gemini's** output filtering, with one noting that while **Gemini** heavily filters outputs, especially for corporate/government settings, its **API** can be manipulated to produce almost anything.
   - One member mentioned using the API at a Google hackathon and getting the voice models to swear by putting it in the system prompt.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1465854567571783902)** (4 messages): 

> `Edge AI Engineer, Quantization and LoRA FT` 


- **Edge AI Engineer Enters the Fray**: A Senior Edge AI Engineer named Josh introduces himself, detailing experience building real offline agents in the **DoD and pubsec** for 6 years.
   - He adds that he makes quants for fun and exclusively uses **Unsloth** for local quantization and LoRA fine-tuning.
- **New Member Says "HelloHi"**: A new member named Josh from senior Edge AI engineering introduced himself.
   - He shares their passion for using Unsloth for quantization and LoRA fine-tuning


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1465798627963179160)** (969 messages🔥🔥🔥): 

> `Personaplex, GLM 4.7, GGUF, Model Quantization, Vendor Lock-in` 


- **Personaplex Personalities**: Members discussed the limitations of **Personaplex** in enforcing personality and its tendency to become like *shitty ai podcasts* after some iterations.
   - One member mentioned they don't have access to the stored recorded calls that would be perfect to train **Persona Plex** on.
- **GLM 4.7 Flash Performance Talk**: A user asked if anyone had tried the [GLM-4.7-Flash-REAP-23B-A3B-GGUF model](https://huggingface.co/unsloth/GLM-4.7-Flash-REAP-23B-A3B-GGUF) and another responded that REAP models are often not very good, suggesting a lower quantization instead.
   - Others weighed in with their performance and insights on the [GLM 4.7 Flash model](https://huggingface.co/unsloth/GLM-4.7-Flash-REAP-23B-A3B-GGUF), with comparisons to **GPT-OSS-120B** and **Kimi** in terms of reasoning, efficiency, and ability to relate information.
- **GGUF Safety Concerns**: A member inquired about resources regarding the potential unsafety of **GGUFs**, specifically if a malicious actor got involved.
   - However, another member stated *I'm not familiar with that, I think you might have got me mixed with someone else* so nothing more came of it.
- **AI Model Hallucination Watch**: A member noted that their **3b llama** model made the *creepy assumption that it was trained on my voice without prompting*, leading to a discussion about hallucinations in LLMs and their lack of awareness of their training or state.
   - One member recommends [this YouTube video on AI hallucinations](https://youtu.be/wjZofJX0v4M?si=A4rHzAh9qJjls9bm) as a starter on the topic.
- **Vendor Lock-in Temptations**: The group discussed a hypothetical scenario where token prices increase drastically, touching on the concept of vendor lock-in.
   - There was mention that Nvidia and Amazon are also employing *vendor lock in* tactics, and it's *called a software locked inbasically what Nvidia is doingAmazon too (I think)*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1465801085774987430)** (77 messages🔥🔥): 

> `Unsloth container setup errors on encrypted Runpod, GLM-4.7 tool calling issues, CUDA version issues with GLM 4.7 on Blackwell B200s, Multi-GPU training problems with Qwen3 model on Modal, Catastrophic forgetting after finetuning` 


- ****Runpod Setup Suffers Permission Problems****: A member encountered a "permission denied" error when setting up an Unsloth container on an encrypted Runpod, suggesting an issue with [volume permissions](https://link.to/volume-permissions) during container creation.
   - Another member suggested that the Runpod was attempting to modify the container structure, which is not the intended behavior, instead recommending the use of [the official Docker container image](https://hub.docker.com/r/unsloth/unsloth) to avoid such headaches.
- ****GLM-4.7 Tool Time Troubles****: A member sought assistance with getting **GLM-4.7** to call tools, following [the official Unsloth documentation](https://unsloth.ai/docs/models/glm-4.7-flash#tool-calling-with-glm-4.7-flash).
   - The discussion included the need to use `json.loads` for arguments and the identification of `tool_calls` in the `res.choices[0].message` structure for generic tool calling.
- ****Blackwell B200s Battle CUDA Conflict****: A member reported **CUDA 12.8** drivers on their **B200**, incompatible with **GLM 4.7**'s **CUDA 13.x** requirement, needing a CUDA upgrade and dependency reinstall to run their **vllm** server.
   - It was suggested to force reinstall **vllm** with `--torch-backend=auto` and a CUDA 12.9 nightly build URL to potentially run **GLM 4.7** on CUDA 12.8, but with the removal of `--kv-cache-dtype fp8` due to **Ada Lovelace GPU** incompatibilities.
- ****Modal Multi-GPU Mishaps Mounting****: A member faced issues training a **Qwen3** model on 3 GPUs on **Modal**, encountering a "ValueError" due to an incorrect `device_map` configuration and import errors with `prepare_device_map`.
   - It was revealed that the training setup had switched away from **Unsloth** due to incompatibility with **PyTorch 2.4.1**, opting for a **transformers + PEFT** setup for better stability.
- ****Finetuned Model Forgets Fundamentals****: A member described experiencing catastrophic forgetting in a finetuned model, where it excels at new information but forgets prior knowledge, suspecting overfitting issues.
   - Mitigation suggestions included lowering the **LoRA rank, LR**, reducing steps/epochs, and mixing in more general data, as well as targeting fewer layers.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1465831343643557958)** (9 messages🔥): 

> `KL Divergence, Mode Collapse, DeepSeek mHC residual preservation, Context Distillation` 


- **KL Divergence Initial Values Spark Debate**: A member inquired about the ideal initial **KL divergence** when loading the **SFT model** as the **ref_model**.
   - They expected to see zero divergence initially, referencing [this 2026 paper](https://arxiv.org/abs/2601.09954).
- **Mode Collapse Creates Variance Void**: A member reported experiencing **mode collapse**, leading to little variance between responses and amplified errors.
   - They said, *"now it's getting a lot more responses correct, however, the ones that it gets wrong, it just gets wrong fully since there are little variances."
- **DeepSeek's mHC Residual Preservation Predicted**: A member speculated that **DeepSeek** would have relevant insights into **mHC residual preservation**.
   - No further information was given.
- **RL Researchers Rediscover Context Distillation**: A member wryly noted that **RL researchers** are seemingly rediscovering **context distillation**.
   - No further information was given.


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1465840851463700532)** (2 messages): 

> `Arcee Trinity Large Preview, Mixture-of-Experts, Open Weights` 


- **Arcee Trinity Large Preview drops!**: Arcee released its first frontier-scale open-weight model, [Trinity-Large-Preview](https://openrouter.ai/arcee-ai/trinity-large-preview:free), as a chat-ready variant, available for free for a limited time.
   - The announcement on [X](https://x.com/OpenRouterAI/status/2016280059527757995?s=20) highlights that it's a **400B parameter sparse Mixture-of-Experts** model but has **13B active parameters per token**.
- **Arcee's efficiency-focused architecture**: Arcee's **Trinity-Large-Preview** model uses **256 experts** with **4 active per token** (1.56% routing).
   - The model is optimized for efficiency rather than dense scale, and features open weights with permissive licensing.
- **Lucas Atkins live now!**: CTO of Arcee AI Lucas Atkins is live, now!
   - Watch the [Youtube Livestream](https://youtube.com/live/3XSdqHY0kNk?feature=share) now!


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/)** (1 messages): 

runvnc: I've updated https://cyberpad.site to include some free credits
  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1465799162288148601)** (468 messages🔥🔥🔥): 

> `Image Generation Models, Refund Delays, Context Caching Pricing, OpenRouter API Issues, Model Training From Scratch` 


- **Image Models Not Generating Images**: Users reported that some image models ([google/gemini-3-pro-image-preview](https://openrouter.ai/models?fmt=table&order=most-popular&output_modalities=image&input_modalities=image), **GPT-5 Image Mini**, **GPT-5 Image**, **Gemini 2.5 Flash Image**) tagged as image output modalities aren't generating images, with some models like **Gemini 2.5 flash** working intermittently.
- **Model Support for Response Format Discussed**: Users discussed models with working `response_format` support, listing models like **Gemini 3 Flash Preview**, **Gemini 2.5 Flash Lite Preview**, **Seed 1.6**, **GLM-4.6v**, and **Grok 4.1-fast** as functional, while noting that **Mistral** supports `response_format` on its API but not on OpenRouter.
   - A member noted, *"Gemini 2.5 flash works for me But I need to do some prompting magic sometimes yes"*.
- **OpenRouter API Experiencing Downtime**: Users reported experiencing network errors and non-functional models on OpenRouter, with some encountering *"HTTP 401: User not found"* errors and others experiencing issues specifically from **Hong Kong**.
   - One user mentioned, *"open router down rn or is it just me? literally none of the models work for me they all just say network error"*.
- **Users Discuss OCR Solutions Using OpenRouter**: Members discussed using **Gemini Flash** models for OCR, with one recommending training a custom **Azure/AWS OCR model** for consistency.
   - One user mentioned, *"You can go a long way with the gemini flash models, depending on whether you need to extract data or parse"*.
- **OpenRouter Users Await Long Overdue Refunds**: Users are reporting delays and lack of communication regarding refunds, with some waiting since early January and submitting multiple support tickets.
   - One user stated, *"Seriously though, @OpenRouter team – would love to know: What's the actual timeline for refunds? Why are so many people in the same boat? Is there a status update system that actually works?"*


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - New Models**
  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1465815588814454865)** (15 messages🔥): 

> `Agentic Vision Gemini 3 Flash, OpenAI's Image Capabilities, OpenRouter Show, PRISM` 


- ****Agentic Vision** with Gemini 3 Flash**: Google introduced [Agentic Vision](https://blog.google/innovation-and-ai/technology/developers-tools/agentic-vision-gemini-3-flash/) with **Gemini 3 Flash**, combining visual reasoning with code execution to manipulate images step-by-step.
   - *The model formulates plans to zoom in, inspect and manipulate images step-by-step, grounding answers in visual evidence*.
- **OpenAI Extends Image Capabilities with O3 and O4-mini**: OpenAI's **O3** and **O4-mini** extend image capabilities by thinking with images in their chain-of-thought, allowing them to crop, zoom, and rotate without separate specialized models, detailed in [this blog post](https://openai.com/index/thinking-with-images/).
   - Gemini's ability to return meaningful bounding boxes is second to none compared to OpenAI.
- **PRISM: OpenAI's new baby**: OpenAI introduced **PRISM**, detailed in [this press article](https://archive.md/d9Vsf), which prompted a comment about preferring **Typst** over **TeX** for writing.
   - Someone said *First thing I thought of when I saw the name* referencing the logo [attached here](https://cdn.discordapp.com/attachments/1392278974222307469/1465964452997370078/PRISM_logo.jpg?ex=697bae09&is=697a5c89&hm=3b25626917f4c2e1e0753faa6336b1d5b2a9556a08bd8cc5e325fb8ac0853e09&).
- **Trinity Rocks OpenRouter Show**: A member mentioned they were watching the **OpenRouter show** for the first time, with excitement for Trinity's segment, which is available for free.
   - The OpenRouter SDKs for Agentic Usage is located [here](https://openrouter.ai/docs/sdks/agentic-usage#supported-ai-coding-assistants).


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1465798548443369502)** (476 messages🔥🔥🔥): 

> `Missing Revert Button, Cursor CLI vs IDE, Cursor Pricing, Clawdbot Security Issues, Gemini Agentic Vision` 


- ****Revert Button Vanishes, Users Fret****: Users reported the **revert button** disappearing from the UI, leading to frustration and token waste when code gets mucked up, with one user finding that [duplicating an older chat](https://cdn.discordapp.com/attachments/1074847527708393565/1465828018789552390/image.png?ex=697bd7b9&is=697a8639&hm=68ec5dd17c7a1be84a1f639f9a5a98db91ba3bd191336f2afe3e8252b804b12e&) brought it back.
   - One member found that not clicking on the revert button would make it appear, suggesting it was a **one-time bug**.
- ****Cursor CLI Trumps IDE for Some Devs****: Some developers are preferring **Cursor CLI** over the IDE due to a smaller memory footprint, which helps them avoid IDE crashes and model unresponsiveness, especially with larger projects exceeding 100k LOC.
   - However, one user found **Cursor CLI inside the IDE** (with WSL as the terminal) to be *"pure trash.. like for real, not usable"*, with another reporting that the UI is not smooth, even with 64GB of RAM and an i7 processor.
- ****Cursor Pricing Gets a Makeover****: After September 15th, **auto mode is no longer unlimited** and counts toward the $20 monthly allowance, priced at $1.25 per 1M tokens for Input + Cache Write, $6.00 per 1M tokens for Output, and $0.25 per 1M tokens for Cache Read, but users with older subscriptions can still enable on-demand usage.
   - One user discovered they could burn through their monthly subscription very quickly, suggesting it may be cheaper to *use their own api keys, or use Claude Code*.
- ****Clawdbot's Credential Catastrophe****: A user shared several links regarding **security concerns with Clawdbot**, reporting that exposed control panels pose credential leaks and account takeovers.
   - There is speculation it could lead to a *"store now, decrypt later"* data breach due to potential quantum decryption issues, and that the company got a cease and desist for the issues.
- ****Gemini Vision Excites Front-End Devs****: A user found that **Gemini agentic vision** is nearing state-of-the-art (SOTA) performance for vision tasks, and believes its integration would simplify front-end development.
   - Members stated that they can't wait to see vision integrated into the agent, and that it is superior to the `Auto` tool.


  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1466135653895897335)** (1 messages): 

> `LM Studio 0.4.0, Server Deployment, REST API` 


- **LM Studio Refreshes to 0.4.0!**: A new generation of **LM Studio** has been released, version **0.4.0**, featuring the [complete blogpost here](https://lmstudio.ai/blog/0.4.0).
- **Non-GUI Server Deployments Now Supported**: **LM Studio 0.4.0** can now be deployed on non-GUI servers, in CI, or anywhere.
   - This enables parallel requests for high throughput use cases, thanks to the new stateful **REST API**.
- **Local MCPs get Stateful REST API**: The new stateful **REST API** is designed to use local **MCPs**.
   - There has also been a complete UI revamp as part of the **0.4.0** release.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1465803965504557077)** (298 messages🔥🔥): 

> `GLM 3.7 Flash Coding Ability, LMStudio OpenAI Tool Calling, LM Studio OpenAI Streaming, Gemma 4 Speculation, LM Studio v0.4 Update` 


- **GLM 3.7 Flash excels at coding, OSS 120 remains superior**: Members note that **GLM 3.7 Flash** shows good coding ability, but **GPT OSS 120** is expected to be the superior coder, especially at **Q4**.
- **LMStudio's API stumbles on tool calling**: The **LMStudio OpenAI compatible Responses API** doesn't properly handle tool/function calls; the server should send `response.completed` or `[DONE]` after the model decides to call a function/tool, but this is not happening.
- **Plugin Proxy Powers Unreal Engine**: A member has created their own **plugin & proxy** to get **OpenAI streaming** to work, enabling **Unreal Engine** to talk to **LM Studio** for actor spawning and manipulation.
- **Gemma 4 Speculation Fuels Hype**: Users speculate on a potential **Gemma 4** release, with hopes for a **Mixture of Experts (MoE)** architecture and various sizes (4/8/12/30b), while some jokingly suggest a **1b** model for edge devices and caution against overhyping the release.
   - One member proclaimed, *"If Gemma 4 isn’t MOE I will eat my shoe."
- **LM Studio v0.4 Goes Headless and Parallel**: **LM Studio v0.4** introduces **headless mode** and **parallel inference**, with users excited about the new capabilities, though in-app updates require reinstalling the app, and some UI elements are now in **dev mode**.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1465817259153821738)** (33 messages🔥): 

> `Remote AI Rigs, Devstral-2 Performance, ROCm in LM Studio, 1.8 bit Quantization, LM Studio Backend for Hardware Accelerators` 


- **AI Engineers Rigs: Remote Access Realities**: Members discussed their methods for remotely accessing their AI rigs, with one suggesting **VNC** for virtual machines running LLMs.
   - The original question regarded using **Windows' built in Remote Desktop**.
- **Devstral-2 Demands Decent GPU Deployment**: Members discussed the hardware requirements for running **Devstral-2** locally, with one user suggesting **48GB of GPU** (e.g., 3090) for the 24B version.
   - For the 120B version, parallel computing or an **H200 with EXL2** model format were suggested, as GGUF was deemed too slow.
- **ROCm Runs on LM Studio Runtime**: Users discovered that **ROCm** can be enabled within **LM Studio** under the runtime settings, which was initially obscured for some users.
   - One member shared a link to a relevant [Unsloth Reddit thread](https://www.reddit.com/r/unsloth/comments/1qpbmrt/you_can_now_run_kimi_k25_locally/).
- **1.8 Bit Wonders: Quantization Quirks Questioned**: Members discussed the nature of **1.8 bit quantization**, with one user explaining it as a dynamic quantization method where unimportant parts are **1 bit** and others are **2-3 bits**.
   - Others drew comparisons to a *lobotomized ex-scientist* and joked about running only Tetris with it.
- **Hardware Acceleration Hacks: Hooking into LM Studio**: A member from a hardware accelerator company inquired about adding an **LM Studio backend** for their hardware.
   - It was suggested to focus on **llama.cpp**, as LM Studio uses it as a backend library, but it was noted that LM Studio is primarily a closed source project by Element Labs, and links to [LM Studio Enterprise](https://lmstudio.ai/enterprise).


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1465798707546034268)** (307 messages🔥🔥): 

> `Kimi K2.5 pricing, Google Aistudio data training, Model selection (Kimi vs Mistral vs Qwen), Kimi CLI vs other tools, Agent Swarm` 


- **Users Discuss Kimi K2.5 Pricing Model**: Some users expressed concerns about the **$19** monthly subscription fee for Kimi K2.5, with one user finding it *expensive* due to their location and contemplating whether a recurring deal could be established.
   - Another user suggested sticking to the free tier, citing that smaller Chinese companies like Moonshot AI need to run large models like K2.5, so lower prices are unlikely.
- **Google's AI Studio Training Practices Spark Debate**: A user voiced concerns that **Google trains and views conversations** in AI Studio and Gemini apps, raising privacy issues.
   - In contrast, another user mentioned they **open source their projects** anyway, so the data would likely end up in training datasets regardless.
- **Model Selection Mania: Kimi vs. Mistral vs. Qwen**: Users compared Kimi K2.5 with other models such as **Mistral and Qwen** for various tasks, including coding and general question-answering.
   - One user noted that **Kimi K2.5** has the *highest benchmarks* among the mentioned models for physics, chemistry, and math, while another pointed out its *strong performance in design and logical reasoning*.
- **Kimi CLI Proves Superior to Alternatives**: Users tested **Kimi CLI** and found it **faster and more efficient** compared to oh-my-opencode, especially for analyzing web pages, with reduced token consumption.
   - However, some found the model's output quality to be *less impressive*, expressing a desire for further comparisons.
- **Agent Swarm Usage Explored in Kimi K2.5**: One user enjoyed using **Agent Swarm** with Kimi, noting its capabilities for in-depth research, while others were unsure of its applications.
   - It was noted that **Agent Swarm** can quickly deplete agent credits, burning them at **3x** the normal rate.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1465798839821799424)** (177 messages🔥🔥): 

> `Subscription Scams, Billing Issues and Refunds, Query Limits, Image Generation Restrictions, Kimi 2.5 Release` 


- **Perplexity Subs Called a Scam?**: Several users reported **unexpected subscription changes** and **charges** after automatic renewals, with one user canceling their subscription, calling it a *scam.*
- **Billing Issues Spark Bank Contact**: Users reported billing discrepancies, such as being charged without service or not getting refunds, with one user planning to **contact their bank** for a refund of 100 euros.
   - Another user suggested contacting the payment processor to **stop further unauthorized transactions** and reporting the issue to the FTC.
- **Users Baffled by Query Cap Shenanigans**: Some users reported issues with **query limits** on their **Pro subscriptions**, experiencing limits as low as one query per hour, while others saw their limits restored to 600.
   - One user shared a link to check query limits ([perplexity.ai/rest/rate-limit/all](https://www.perplexity.ai/rest/rate-limit/all)), noting their **600 queries** were suddenly restored.
- **Image Generation restricted by region?**: Users reported **image generation restrictions** in certain regions, possibly due to **xAI controversies** and an EU lawsuit, with suggestions to try different models or contact support.
   - One user from India confirmed they were also affected by this issue.
- **Kimi 2.5 Coming Soon to PPLX?**: Users are asking about the release date of the **Kimi 2.5 model** on Perplexity, eager for its implementation.
   - One user speculated that Perplexity is usually quick with such updates.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/)** (1 messages): 

tay.0.00: Love
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1465801949499756647)** (159 messages🔥🔥): 

> `GPT Pro, OpenAI pricing, DeepSeek jailbreak, TI-84 Neural Network, Staged Reward Shaping` 


- **GPT Pro: More GPUs or Model Magic?**: Discussion revolves around whether **GPT Pro's** superior performance stems from simply using more GPU power (e.g., running multiple instances in parallel) or if it involves a fundamentally better model, with speculation that **OpenAI** might be strategically obscuring the true nature for competitive advantage.
   - One member even suggests it's *a game of fakery*, comparing **OpenAI's** pricing strategy to impressions rather than measured value, akin to the stock market and **Tesla**.
- **China Cracks Down, Filters Found**: Members discussed **Chinese models** being subject to censorship, with one member claiming *the CCP has a bunch of power over these labs*, as well as sharing an [image](https://cdn.discordapp.com/attachments/1149866623109439599/1465867774994940170/image.png?ex=697bfcc0&is=697aab40&hm=ccdd92053333326694a5a1919519f39b17935bca2eb5be926d99b4fb2ca5afbc&) showing censorship filters in the thinking traces of a model.
   - They also pointed out that **China** successfully manipulates public perception and funds a lot of AI labs.
- **DeepSeek's Infinite Imprisonment**: Members noted that **DeepSeek** has a tendency to get stuck in a jailbreak loop, where, once triggered, it repeats the same rejection message (*I can't assist with that*) indefinitely, regardless of subsequent prompts.
   - The API endpoints are reportedly slightly better, but the raw model is *cooked* once it hits that state.
- **Calculator Gets Neural Network Boost**: A member shared their project of running a neural network on a **TI-84 Plus** calculator for spellchecking, detailing the process on an [academic website](https://hermesoptimus.vercel.app/) with a demo video.
   - The member quipped that even with such advancements, their ongoing work on **Claude Code Orchestration** wins out in terms of real-world application.
- **Staged Reward Shaping: Delegate to Delegate?**: Discussion emerged around **staged reward shaping**, where intermediate rewards are added and adjusted over time, with concerns raised about models easily engaging in reward hacking.
   - One member characterized it as *technical debt*, and another suggested it is *delegate to delegate instead of delegate to go faster*.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1465921905876795474)** (4 messages): 

> `Hermes 4 pricing, API credits` 


- **Hermes 4 Pricing Not Permanent**: A member inquired if the discounted pricing for the **Hermes 4 series** models is permanent before subscribing to the API, noting its superiority in RPing and story-writing compared to **Deepseek**.
   - Another member clarified that there's no subscription, just purchasing credits, and the pricing can change over time, so the value depends on price and usage.
- **API Credits Clarification**: A member explained that using the API involves buying credits that can be topped up, rather than a subscription.
   - The value derived from the credits will fluctuate based on the **pricing** and **usage** patterns.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1465882077357408388)** (1 messages): 

> `MergeMix paper, Data Mixtures, Model Merging` 


- **MergeMix Paper Sparks Interest**: The paper [MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging](https://arxiv.org/pdf/2601.17858) garnered attention due to its relevance for open source projects with limited budgets.
   - The paper explores techniques for optimizing **data mixtures** and **model merging** during training, potentially offering resource-efficient strategies.
- **Image Analysis Discussion**: An image was shared, presumably related to the MergeMix paper or data mixing, but lacks further context or discussion.
   - Without further information, the image's specific relevance or content remains unclear.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1465882077357408388)** (1 messages): 

> `MergeMix, Open Source Model Merging` 


- ****MergeMix** Optimizes Data Mixtures Mid-Training**: A member shared the paper '[MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging](https://arxiv.org/pdf/2601.17858)', highlighting its relevance to open source efforts with limited budgets.
   - The paper explores optimizing mid-training data mixtures through **learnable model merging**.
- **Open Source Model Merging Gets a Boost**: The paper was deemed interesting due to its implications for open-source initiatives dealing with significantly smaller financial resources.
   - The attached image [link to image](https://cdn.discordapp.com/attachments/1104063238934626386/1465882077428846623/image.png?ex=697b6152&is=697a0fd2&hm=7ea494cc6abfd85cd18d6434dca494139bbccaaf966f67adbdb35b09748bf465) visually supplements the discussion.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1465800663253385389)** (81 messages🔥🔥): 

> `Gemini 3 Pro failure to generate .srt files, Clawdbot a scam by crypto bros, Prism harmful to scientific research, OpenAI prioritizing security concerns, ChatGPT vs Gemini comparison` 


- **Gemini 3 Pro fails spectacularly**: A user reported that **Gemini 3 Pro** completely fabricated an .srt file of subtitles, with *nothing* related to the audio in the video, leading to disappointment with its performance.
   - Other users chimed in with similar experiences, with one stating *i really hate to say this but gemini is overhyped recently now, it hasn't been doing well for me too*.
- **Clawdbot's murky malware status**: **Clawdbot**, now known as **moltbot**, is an agentic system that controls your entire OC by API keys from Anthropic, Google, OpenAI, and users are being warned against it, with one user stating it is *a huge scam by crypto bros to steal your information*.
   - Despite the original version not being inherently malware, it can be weaponized via prompt injection, crossing into secondary malware behavior, raising significant security and privacy concerns with automation/agentic AI/bot AI.
- **Prism's Sci-Fi future or scientific flop?**: While **OpenAI** aims to advance science with **Prism**, one user stated that **Prism** is *not beneficial to scientific research* and is actually *damaging* to scientific research.
   - Another user asked if **Prism** has API access, wondering whether they can write some of their project there using other AI and **Codex**.
- **OpenAI Prioritizes Cybersecurity**: One user shared that **OpenAI** is trying to upgrade their **Codex** to strongly deal with cybersecurity concerns.
   - They believe that this is because *safety and security is indeed a massive concern for people who just want to create and automate freely without having to worry about tampering and hijacking*.
- **ChatGPT triumphs over textbook-y Gemini**: One user stated that, when it comes to LLMs, *benchmarked leaderboard rankings don’t mean much to me* and that **Gemini** is very textbook-y, whereas **ChatGPT** does an amazing job with handling context across sessions.
   - The user also noted that Gemini is very rigid with rules and once they tell it a preference, *it sticks to it like it is religion*.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1465916154471252215)** (4 messages): 

> `AI as career, AI Safety, GPT file reading` 


- **Making a Living with AI**: A member asked if anyone is making their living in **AI**, seeking suggestions on how to monetize their passion for **AI**.
   - One user suggested exploring **AI Safety** and **red teaming**, pointing to related communities.
- **GPT Pro Loses File Reading Prowess**: A user reported that **GPT Pro 5.2**, which could previously read and analyze **ZIP files**, is now failing to find uploaded files for analysis.
   - The user is asking if others are experiencing the same issue.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1465918446339620915)** (8 messages🔥): 

> `Sora Prompting Guide, GIF Generation, Chiaroscuro Effect in Image Generation, Realtime Visualizers` 


- **Prompt Power-Up: Sora's Subtleties Shine!**: A member shared the [Sora Prompting Guide](https://developers.openai.com/cookbook/examples/sora/sora2_prompting_guide), emphasizing the importance of maintaining a **positive cadence** in prompts and grouping negative constraints effectively.
   - The user suggested avoiding excessive individual “no x” orders to achieve better results.
- **GIF Wizardry: In-App Animation Station!**: Users confirmed that the **GIF process can be done in-app from start to finish**.
   - One user, looking to the future, envisions the expansion of **GIFs** and other **animation** into streaming models, potentially including an **OAI version of Lyria Realtime with visualizers**.
- **Banish the B&W: Blocking the 'Chiaroscuro' Catastrophe!**: Users discussed an image generation "issue" related to the **Chiaroscuro effect**, which is the use of strong contrasts between light and dark.
   - The recommendation was to *"Please avoid Chiaroscuro”* in prompts if encountering unwanted **black and white images**.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1465918446339620915)** (8 messages🔥): 

> `Sora Prompting, GIF creation, OAI Lyria Realtime, Chiaroscuro Image Issue` 


- **Sora Swifties Share Prompting Guide**: Members shared a useful [prompting guide](https://developers.openai.com/cookbook/examples/sora/sora2_prompting_guide) for **Sora**, suggesting to keep a positive cadence and avoid grouping negative constraints.
   - The advice aims to prevent overwhelming the model with excessive 'no x' orders.
- **GIF Generation Gems in-app**: A member noted that the **GIF creation process** can be completed entirely in-app, showcasing its streamlined functionality.
   - Another member encouraged users to utilize advanced libraries like **PIL** for optimal results, predicting that **OAI 5.2** will enthusiastically support this process.
- **OAI's Lyria Realtime Visualizer Vision**: A user expressed excitement for an **OpenAI version of Lyria Realtime** with visualizers, emphasizing the fun of steering the model.
   - They fantasized about *disco cat-girls* and suggested an **OAI vocal coach** as cool ideas, envisioning chat using a different language.
- **Chiaroscuro Creates Chaos**: Users reported an issue with **B&W images**, recommending to avoid *Chiaroscuro* in prompts to mitigate the effect.
   - **Chiaroscuro** is defined as the use of strong contrasts between light and dark, usually bold contrasts affecting a whole composition.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1465884462893105407)** (29 messages🔥): 

> `TorchX, mech interp-style tools for numerics debugging, FlagOS Open Computing Global Challenge, transformers 4.57->5.0  or pytorch 2.9.1 ->2.10 breaking training pipeline, interactive tools for exploring numerics` 


- ****TorchX** Orchestration Still Recommended?**: A member inquired whether the [TorchX video](https://www.youtube.com/watch?v=f-Bwru7TJSc) is still the recommended standard for multi-node GPU orchestration.
   - The video creator responded that it's what they mostly use on internal servers, but they haven't kept up with job launcher evolution in the past year.
- ****Mech Interp** Tools for Numerics Debugging?**: A member inquired if mech interp-style tools are used for numerics debugging, wanting to use them to debug model instability at an op and kernel level.
   - Another member is interested in the tooling being more methodological about model debugging, *checking which circuit is unstable, which layer is causing a bunch of outlier, simple stuff like that.*
- ****FlagOS** Global Challenge Competition**: There is a **FlagOS Open Computing Global Challenge** competition with a **RMB 2,000,000 Prize Pool** open for Global Developers.
   - The competition is described in more detail at [flagos.io](https://flagos.io/RaceDetail?id=295v67vw&lang=en).
- **Transformers and PyTorch Upgrade Breaks Training**: A member reported a `NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'` after upgrading **transformers** and **pytorch**.
   - Downgrading to transformers **4.57.3** fixed the issue; others had similar issues, which are discussed in this [pytorch issue](https://github.com/pytorch/pytorch/issues/127176) and [optimi issue](https://github.com/warner-benjamin/optimi/issues/8).
- **Interactive Tools for Exploring Numerics**: A member expressed surprise that quantization people have not already created interactive tools for exploring numerics.
   - Another member responded that *many researches are using standard architectures*, with [captum](https://pytorch.org/docs/stable/generated/torch.autograd.profiler.profile.html) cited as one possible tool, though lacking a proper UI/UX.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1465955998396317738)** (8 messages🔥): 

> `CompositeImplicitAutoGrad Error, JAX AI-Generated PR, Triage Bot` 


- **CompositeImplicitAutoGrad Generates Errors**: A user encountered a `UserWarning` when trying to force a custom operator to use `CompositeImplicitAutoGrad` for automatic differentiation, stemming from an autograd kernel not being registered to the `Autograd` keys, raising concerns about potentially incorrect behavior and deprecated functionalities.
   - The user questioned whether *Fallthrough* is only an option for main library operators and not custom ones, seeking clarification on how to resolve the error and ensure proper differentiation of their custom operator.
- **AI-Generated PR Angers Developer**: A developer expressed frustration upon seeing an AI-generated pull request (PR) in JAX receiving engagement from a maintainer, while their own small bug fix PR remains unattended.
   - The developer sarcastically labeled the AI-generated PR as *clear slop*, criticizing the maintainer for prioritizing it over genuine contributions.
- **Triage Bot Faces Uncertain Future**: A user inquired about the fate of triage meetings in light of the introduction of a new triage bot.
   - The user questioned whether the triage bot's implementation would lead to the discontinuation of traditional triage meetings, implying concerns about the bot's effectiveness or impact on team dynamics.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1465862401629884642)** (1 messages): 

> `H200, INT4, QAT, RL, Model Rollout` 


- **Squeezing 1TB Model into H200 with INT4**: A member shared a link about squeezing a **1TB model rollout** into a single **H200** using **INT4 QAT RL** end-to-end practice.
   - Details are available in this [GitHub repo](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/int4/readme-en.md).
- **INT4 QAT RL Repo**: The GitHub repository provides resources and documentation related to the **INT4 QAT RL** implementation.
   - It focuses on optimizing large model rollouts for hardware like the **H200**.


  

---


### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1465831167226810399)** (1 messages): 

> `Decart Hiring, Lucy 2 Model, Real-time video kernels` 


- **Decart Seeks Performance Engineers for SF Office**: Decart is hiring engineers for their SF office to work on low-latency kernels for real-time video/world models and the latest accelerators, specifically mentioning results on **Trainium 3** at ReInvent ([video](https://www.youtube.com/watch?v=K49S79wOGl8)).
   - Interested candidates are encouraged to reach out to heba@decart.ai with references to their perf work, such as **GPU Mode submissions** or **OSS contributions**.
- **Decart Announces Lucy 2 Autoregressive Model**: Decart launched their latest autoregressive video editing model, **Lucy 2** ([tech report](https://x.com/DecartAI/status/2016134190509498740)).
   - They are also co-sponsoring an upcoming kernel challenge with **GPU Mode** for autoregressive diffusion models.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1465935936905805824)** (13 messages🔥): 

> `PopcornCLI github issues, CUDA C++ and Python, Performance plan for CUDA, PMPP book` 


- **PopcornCLI deadline errors surface**: A member reported getting an error with **PopcornCLI** github reference commands, specifically a *deadline has passed* message when changing the leaderboard from grayscale to vectorsum.
   - They discovered that the leaderboards are suffixed by **v2** (e.g. *grayscale_v2*, *vectorsum_v2*) and the TUI shows the leaderboards.
- **Seeking guidance on CUDA C++ with Python for deep learning**: A member requested guidance on running **CUDA C++** along with **Python** for deep learning, admitting they are a *noob*.
   - Another member suggested checking out the **load_inline** feature in PyTorch and mentioned that Lecture 1 has some instructions for this.
- **CUDA performance planning primers prompt pointers**: A member asked for guidance on creating a performance plan before writing **CUDA** code, being aware of **NVIDIA Insight** but wanting to understand the *why* behind its suggestions.
   - Another member inquired about their level of expertise and whether they had started with the **PMPP book** in the book channel.
- **gpumode link fix**: A member noticed that the link for **gpumode** was broken and suggested replacing it with [this link](https://www.gpumode.com/).
   - No further discussion.


  

---


### **GPU MODE ▷ #[popcorn](https://discord.com/channels/1189498204333543425/1298372518293274644/1465898873938776175)** (1 messages): 

> `New Collaborators, Kernel LLM` 


- **Collaborators Wanted for Kernel LLM Improvement**: New collaborators are sought to *quickly run ablations* and *generate/test ideas* to improve a post-trained **Kernel LLM**.
   - The call emphasizes skills in areas such as **synthetic data**, **training algorithms**, and **memory** optimization.
- **Meeting and Announcements Spark Interest**: A member expressed interest in the **Kernel LLM** collaboration opportunity after reviewing the **2026 news and announcements** post.
   - The member inquired about the relevance of the channel and how to begin contributing, showing proactive engagement with the initiative.


  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/)** (1 messages): 

ivanbernal0511: tell me your Jetson model, batch size, and whether you’re aiming for FP16 or INT8
  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1465816049483124939)** (1 messages): 

> `DGX, 5090, Blackwell PRO, L2 cache` 


- **DGX and 5090 Instruction Sets Alike!**: Instruction sets for **DGX** and **5090** are the same, but **DGX** boasts full-speed fp32 accumulation, akin to **Blackwell PRO** cards.
   - The real game-changer? **1.8TB/s** vs **300 GB/s** memory bandwidth—efficient **L2 cache** use is key!
- **Memory Bandwidth: DGX Dominates**: **DGX** shines with **1.8TB/s** memory bandwidth, a stark contrast to **5090's 300 GB/s**.
   - Optimizing **L2 cache** utilization becomes paramount to leverage **DGX's** superior performance effectively.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1465900732388474932)** (16 messages🔥): 

> `Tractable Layouts, Tuple Morphisms, Mutual Refinements, Cute Composition, tract.weak_composite` 


- **Order Matters in Tractable Layout Diagrams**: The order of nodes on both sides of diagrams representing tractable layouts is critical; swapping elements leads to different layouts e.g. changing `(4, 8):(1, 4)` to `(4, 8):(8, 1)`.
   - One member noted that the order is not arbitrary, it is very inflexible and that permuting the left-hand-side makes a difference.
- **Clarification on Tuple Morphism Codomain and Domain**: In mutual refinement, the left-hand side is the **codomain** of tuple morphism `m_A`, while the right-hand side is the **domain** of `m_B`, and a [blog post](https://research.colfax-intl.com/categorical-foundations-for-cute-layouts/) was provided for main definitions.
   - Both sides are sorted from bottom to top as they were on the previous page.
- **Cute Composition via Tract**: `tract.compose` requires the **codomain** of the first morphism to equal the **domain** of the second, whereas **mutual refinements** generalize composition via *refine, pullback/pushforward, compose*, referred to as *weak composition*.
   - To achieve this in `tract`, one should use `tract.weak_composite(morphism_A, morphism_B)`.
- **Typo Fixed in Layout Diagram**: A member identified a typo in a layout diagram screenshot, clarifying that the order of nodes in the diagrams is significant for defining the layout.
   - The corrected understanding simplifies reasoning about how *Step 2* in the process leads to the expected composition result.


  

---


### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1466039529499525161)** (2 messages): 

> `mdbook rust playground, AWS/GCP research cloud credits, magnetron, rust->x86 to cuda->ptx` 


- **mdbook REPL Defaults to Underpowered Debug Mode**: mdbook's REPL support sends code to the public instance of **Rust Playground** with debug cargo profiles, resulting in `~10MFLOPS` performance, compared to `1GFLOPS` in release mode.
   - It also sends JSON requests with **debug cargo profiles** instead of release, but a member plans to monkey patch into mdbook's javascript to send requests in release mode.
- **Public Rust Playground on Frugal Hardware**: The public **Rust Playground**, mirrored by integer32 and linked in its README, is hosted on a free-tier t2.micro instance, achieving `1-2GFLOPS` in release mode, aligning with back-of-the-envelope calculations.
   - The max theoretical throughput on the **t2.micro** is `~20GFLOPS`, but the vcpu's hypervisor caps to 10% utilization with elastic bursts using credits.
- **Eyeing AWS/GCP Credits for Hefty Benchmarks**: A member plans to apply for **AWS/GCP research cloud credits**, drawing inspiration from mario's approach in magnetron to achieve `~2TFLOPS` on beefy CPUs.
   - This approach will cover **rust->x86** with intel vtune/amd uprof to **cuda->ptx** with nsight.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1466162536926937290)** (1 messages): 

> `Ed Yang, JAX, Torch, Sharding` 


- **Ed Yang Blogposts Compare JAX and Torch**: Ed Yang has posted some interesting blog posts about distributed computing topics.
   - Notably, a comparison of how **JAX** and **Torch** handle different aspects of **sharding** ([link to tweets](https://x.com/ezyang/status/2016268240754712988?s=20)).
- **Distributed Computing Insights**: Ed Yang's recent blog posts provide insights into various distributed computing topics.
   - These posts offer a comparative analysis of different approaches to handling sharding in JAX and Torch.


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1466146755614871708)** (1 messages): 

> `AMD Helion Plans, Enable Skipped Tests` 


- **AMD Helion Plans Spark Curiosity**: A user expressed interest in learning more about **AMD's plans** on **Helion**.
   - They suggested a quick sync meeting to discuss further details.
- **Skipped Tests Get Enabled**: A user thanked another user for putting up the **PRs** to enable the skipped tests.
   - No further details were provided.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1465878066407542806)** (10 messages🔥): 

> `constexpr in CuTeDSL, NCU profiling, Kernel hangs, Grand prize arithmetic difference, measurement error` 


- **constexpr improves CuTeDSL performance**: A member shared a tutorial on using **constexpr** in **CuTeDSL** to improve performance applied to the reference kernel, claiming performance should be much better than the simple baseline, with a [link to the tutorial](https://gist.github.com/simveit/f8f538adacb5d4c2703600b843ba0547).
- **NCU profiling status unclear**: A member asked whether **NCU profiling** is working again.
   - It was followed by complaints about hitting illegal memory accesses or kernel hangs, asking if they can send code and **NCU profiles** to someone.
- **Grand prize uses arithmetic difference**: A question arose about whether the "closest to speed of light" for the grand prize is measured with an **arithmetic difference** or **percent difference**.
   - A different member stated they can comment on any rule subtleties.
- **Measurement error questions**: Another question about grand prize was asked: what if the **measurement error** lands on the far side of the sol and someone is closer to it on the slower side?


  

---


### **GPU MODE ▷ #[cutile](https://discord.com/channels/1189498204333543425/1461235643211321437/1465907650381480119)** (2 messages): 

> `Nvidia B200, CuTile, nvfp4` 


- **B200 Lacks CuTile Support**: A user inquired whether the **Nvidia B200** competition environment has **CuTile** support.
   - Another member responded that it doesn’t support **nvfp4** yet, so **CuTile** wouldn’t be too useful.
- **NVFP4 Support Missing**: The **Nvidia B200** competition environment does not currently support **nvfp4**.
   - Without **nvfp4** support, **CuTile** would not be particularly effective in the **B200** environment.


  

---


### **GPU MODE ▷ #[flashinfer](https://discord.com/channels/1189498204333543425/1464407141128339571/1465825640807661628)** (10 messages🔥): 

> `Biweekly Leaderboard, Flashinfer-bench ModuleNotFoundError, MLSys'25 contest trace, Quantization Algorithms in FlashInfer, Looking for Teammates` 


- **Biweekly Leaderboard Coming Soon**: The team is working on supporting a **biweekly leaderboard** for the competition.
- **Flashinfer-bench ModuleNotFoundError Solved**: One user encountered a `ModuleNotFoundError` when running `python ./scripts/pack_solution.py`, but resolved it by installing from the **latest git repo**.
- **MLSys'25 Contest Trace Release Delayed**: A user ran into an error using the **flashinfer trace** and was told they may need to wait for the release of the **MLSys'25 contest trace**.
- **FlashInfer Explores Quantization Algorithms**: There is a discussion about whether **FlashInfer** plans to support better **quantization algorithms**, with a link provided to a [relevant GitHub issue](https://github.com/flashinfer-ai/flashinfer/issues/2423).
- **"fused_moe" definition found on Huggingface**: The definition and workloads for *fused_moe* are available via [HuggingFace](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace/), and the team asked users to ensure the `FIB_DATASET_PATH` is set to the **local dataset path**.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1465816651978248413)** (66 messages🔥🔥): 

> `Agent-Driven Coding, Prism Science Workspace, Trinity Large MoE Model, Agentic Harnesses Evolution, Cursor's Codebase Indexing` 


- **Agent-Driven Coding Flies into 2026!**: Andrej Karpathy envisions a shift to **80% agent-driven coding** by 2026, leveraging LLMs' tenacity and declarative goal-setting, while cautioning against potential 'slop' and over-engineering; read more [here](https://xcancel.com/karpathy/status/2015883857489522876).
- **Prism Shimmers as OpenAI's New Science Tool!**: OpenAI launched **Prism**, a free research workspace for scientists powered by **GPT-5.2**, now accessible to all with a personal ChatGPT account; access it via dedicated web portal [here](https://xcancel.com/openai/status/2016209462621831448?s=46&t=eWVlK1PU8XfB6f402GJJ9g).
- **Trinity Large's 400B Parameter Power!**: Prime Intellect, Arcee AI, and Datology introduced **Trinity Large**, a **400B parameter Mixture of Experts model**, utilizing only **13B active parameters** for high performance; linked from [here](https://xcancel.com/primeintellect/status/2016280792037785624?s=46).
- **Agentic Harnesses: Orchestrating the Future!**: A long read speculates on the evolution of model harnesses, suggesting smarter models will replace complex orchestrators like LangChain, favoring multi-agent architectures and filesystem-based collaboration; link available [here](https://xcancel.com/voooooogel/status/2015976774128341421?s=46&t=jDrfS5vZD4MFwckU5E8f5Q).
- **Cursor Gets Faster Indexing!**: Cursor announced performance upgrades, including semantic search and a significantly faster indexing process for large codebases; further details [here](https://xcancel.com/cursor_ai/status/2016202243499073768?s=46).


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1466191240063090942)** (1 messages): 

> `Latent Space podcast, Science podcast` 


- **Latent Space Debuts 'Science' Podcast**: Latent Space launched its second podcast, 'Science' ([link to podcast](https://www.latent.space/p/science)), hosted by <@713947182167883897> and <@348078436058660866>.
- **Podcast Discussion Shifts to Dedicated Channel**: Further discussion about the new 'Science' podcast is directed to the newly created channel <#1430253273335595079>.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1465903855450722418)** (10 messages🔥): 

> `MimikaStudio MacOS app, Real-time AI Character Swapping, 1littlecoder AI Tutorials` 


- **MimikaStudio: New MacOS App for Voice**: A member shared a link to a [Reddit post](https://www.reddit.com/r/Qwen_AI/comments/1qnlupq/i_built_mimikastudio_a_native_macos_app_for_voice/) about **MimikaStudio**, a native MacOS app for voice-related tasks.
- **Real-time AI Character Swapping Arrives**: **DecartAI** released a new AI model that enables zero-latency character swapping in video, allowing for real-time video streaming with instantaneous identity replacement.
   - Unlike previous tools like **Kling Motion Control** that require generation time, this model allows for real-time video streaming with instantaneous identity replacement.
- **1littlecoder Joins the Fray**: A member shared a link to the [Nitter profile of '1littlecoder'](https://x.com/1littlecoder), an account focused on **AI tutorials**, **Large Language Models (LLMs)**, and **coding**.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1465806881032241335)** (64 messages🔥🔥): 

> `Local Agent Recommendations for Zed, GLM-4.7-Flash Performance, LLM/SaaS Full Stack AI Developer Availability, Kimi 2.5 Model Performance, C++ vs Python for AI Agents` 


- **Kimi 2.5 Blazes Past GPT5**: A member reported that the new **Kimi 2.5** model is performing better than **GPT5** consistently, and can now be run locally using this [HuggingFace link](https://huggingface.co/unsloth/Kimi-K2.5-GGUF).
   - Others are using sites like [Fireworks](https://www.google.com/aclk?sa=L&ai=DChsSEwiCz-j3iK2SAxUFVX8AHT5cBPkYACICCAEQABoCb2E&co=1&gclid=Cj0KCQiA4eHLBhCzARIsAJ2NZoL9Ani52eByT53nVhnOxG_76F9QllEx50YhK_yfQYsD5bH3ov1pAqwaAl2XEALw_wcB&cid=CAASugHkaDm-Aokq5n3lAlzNAI-Ihc6SdblOJ-BiATzwnaZwDVhVBl3B2U5kGq4mAYjN4wQ992LlqWX5NQ6HksDrhSatp0QEfb7_rWMS_u7_GTCuCkp3YH9fANMaJqDgFvuA6u1bwvl4pJ80zvbUhIFPk7Nrqdpx2PDnsBRncgM3-d1UDhFM-tN117MrOXLWnhycCaPax24T8meZIe-9I2cM5rpAf16KucPGZwg7ixTssRCB7X8RP3B_G4vUCfE&cce=2&sig=AOD64_2SRpHfWjuW4kJawyiTyzrGbKZybQ&q&adurl&ved=2ahUKEwiiteP3iK2SAxV85skDHfklKyoQ0Qx6BAgLEAE) to access it.
- **Local Zed Agent Recommendations**: One member asked for local agent recommendations to use with **Zed**, expressing dissatisfaction with **GLM-4.7-Flash** at Q4 with llama.cpp.
   - Another member recommended **kimi** and **qwencoders 30b q4**.
- **C++ Reigns Supreme for Building AI Agents**: A member stated that *C++ is gonna always rule*, noting that *python agents kinda like signify bloat now* and suggesting focusing on **C++** for high-level jobs.
   - They recommended **fastwhisper.cpp** for STT, **Qwen embeddings** in LlamaCPP for RAG, and **LFM2.5vl** for VLM.
- **Developers assemble to build new AI projects**: Multiple members advertised their AI engineering skills.
   - One member posted a list of key projects like Autonomous Agents, Healthcare AI, Decision Support Systems, Conversational AI, and Fraud Detection Systems.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1465812757730426942)** (9 messages🔥): 

> `Vision model JPEG artifacts, RemnantInstruct-8B merge, CLIP-powered Kiki or Bouba classifier, Vision-language models for quantum computing, LeetCode MCP server` 


- **Vision Model Vanquishes JPEG Artifacts**: A new vision model removes artifacts caused by **JPEG compression** using a unique design with no Batch Norm, no activations after training, and Operator layers instead of Convolutional layers.
   - The model allegedly gains accuracy with **width** rather than depth.
- **RemnantInstruct-8B: Merging Creativity with Accuracy**: **RemnantInstruct-8B** is a [SLERP merge](https://huggingface.co/anthonym21/RemnantInstruct-8B-GGUF) that recombines a creative fine-tune (**allura-org/remnant-qwen3-8b**) with its base model (**Qwen/Qwen3-8B**) to balance narrative skills with factual accuracy.
   - The merge strategy favors the creative fine-tune in self-attention layers and the base model in MLP layers, with the goal of preserving **Qwen3's** thinking mode.
- **Kiki vs. Bouba: CLIP Cracks the Case**: A member released a **CLIP-powered Kiki or Bouba classifier** that checks input against ~200 adjectives indicative of Kikiness and Boubaness, like acidic, staccato, buttery, and nurturing.
   - The classifier is available on [HuggingFace Spaces](https://huggingface.co/spaces/jnalv/Kiki-or-Bouba-classifier).
- **Quantum Leap: VLMs Tackle Quantum Computing**: A member open-sourced their undergraduate thesis work on specializing **vision-language models** for **quantum computing** and code with **Qiskit**, including a [dataset](https://huggingface.co/datasets/samuellimabraz/quantum-assistant), [models](https://huggingface.co/collections/samuellimabraz/quantum-assistant), [code](https://github.com/samuellimabraz/quantum-assistant), and [demo](https://huggingface.co/spaces/samuellimabraz/quantum-assistant).
- **LeetCode LM: Ace Coding Challenges from Your Terminal**: A member developed a **LeetCode MCP server** that solves daily challenges from the terminal, integrated with **Claude** for its learning mode, allowing users to authenticate, fetch problems, ask for hints, and submit solutions.
   - They are planning to test it on other LMs and with Cursor and JetBrains, with a potential IDEA plugin in mind; the project is available on [GitHub](https://github.com/SPerekrestova/interactive-leetcode-mcp).


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1465950560577654969)** (2 messages): 

> `Smol Course, Agentic AI, RAG, LLMs, Production Tools` 


- **Smol Course Channel Sought**: A member inquired about a specific server or channel dedicated to the **Smol course** on agentic AI.
   - No specific server or channel details were provided in the messages; however, the user was directed to resources on **RAG, LLMs, Production Tools, Orchestration, Governance, and Real-World Deployments**.
- **ainewshub.live - Daily AI News**: [ainewshub.live](https://ainewshub.live/) was mentioned as a source for daily high-signal updates on agentic AI.
   - It provides distilled information for senior engineers on **RAG, LLMs, production tools, orchestration, governance, and real-world deployments**.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1465798641657708817)** (31 messages🔥): 

> `Flow Matching, Transformers for Continuous Diffusion, Autoregressive Models vs Diffusion Models, Score Parameterization, Byte-Level Prediction Models` 


- **Transformers can Parameterize Vector Fields in Flow Matching**: A member questioned why people claim transformers can't be used in flow matching, arguing it's a training objective where **transformers** can parametrize the vector field.
   - Another member clarified that **transformers** can be used for continuous diffusion, where **patch embedding** encodes patch position, but this doesn't discretize the diffusion or make patches into tokens.
- **Flow Matching is the Same Math as Diffusion**: A member pointed out the irony that [diffusion models are basically the same math as flow matching](https://arxiv.org/abs/2305.03486), but diffusion models are packaged into way too much math.
   - Others agreed, noting variational inference theory is mathy-dense, preferring to use a *sculpting metaphor* when grappling with equations.
- **Diffusion is Not Necessarily Better than Autoregression**: A member argued the idea that *diffusion is inherently better than autoregression* is untrue, and the obstacles are mostly architectural and of scale.
   - They suggest improvements like [repeating the context](https://arxiv.org/abs/2512.14982) or re-encoding a sequence non-causally could bridge the gap, highlighting current design limitations in LLMs.
- **Score Parameterization Preferred over Autoregressive Specification**: A member questioned the need for causal specification in generative modeling loss functions, preferring parameterizing `grad log p(x)` (score) over autoregressive aspects.
   - They linked to [a blogpost on score parameterization](https://yang-song.net/blog/2021/score/), arguing NNs optimize easier without ensuring the area under the distribution integrates to 1.
- **Byte-Level Prediction Model Experiment**: A member sought feedback on a dense MoE architecture for **byte-level prediction** (vocab of 256), using 13GB VRAM with 40M parameters, suggesting the real AGI test is whether it can enumerate latex figure captions.
   - Another member humorously commented on the quality of a specific phrase from a generated sample, saying that *"The study shows that the youths in the statements are described through descriptions" is a clause of all time*.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1466120318874681437)** (6 messages): 

> `Discord event link issues, Google Meet` 


- **Discord Event Links Give Grief**: A member reported issues with a [Discord event link](https://discord.com/events/987824841656791130/1463604897776664872) not working, preventing them from joining the **Daily paper discussion**.
- **Google Meet Saves the Day**: A member unable to use the Discord link was directed to join via **Google Meet**.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1465807859362041998)** (36 messages🔥): 

> `ChatGPT wrappers, Overleaf killer, Clawdbot scam, Leetcode challenges, AI coding and skill retention` 


- ****ChatGPT Wrappers Everywhere****: Members are noticing that most new "things" are just **ChatGPT wrappers**, questioning the value of tools that simply wrap existing models.
   - One member suggested that these wrappers are necessary because *most people don't think about the usecase if you don't make a wrapper around it showing you that you can actually do it.*
- ****Clawdbot Scamming Users****: Someone commented on the ease with which scammers can create wrappers around existing tools, referencing the **Clawdbot scam**.
   - The implication is that OpenAI is essentially *making a wrapper for their own tool*.
- ****AI Won't Replace Skill****: Despite the rise of AI coding tools, members believe that coding ability can be relearned, and that the speed at which code is now produced may hinder true understanding, pointing to a [blog post on Trinity Large](https://www.arcee.ai/blog/trinity-large).
   - It was noted that a bad implementation from an LLM isn't weighted the same as before, since the mental and time cost to create it was so low.
- ****Is Google laundering profit?****: One member proposed the *unserious conspiracy theory* that Google's ad business is just a *laundry operation* for the financial profits they derive from the alpha they get from gmail, workspaces and searches.
   - The discussion took place when pondering if *Agents from Sama et al are probably reading the sessions too*.
- ****Ownership and Terms of Use****: A member quotes from [OpenAI's Terms of Use](https://openai.com/policies/terms-of-use/) that users retain ownership rights in input and own the output.
   - It was noted that OpenAI can use content to train models unless users opt out.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1465889714388336650)** (60 messages🔥🔥): 

> `AMD emulator debug prints, Github actions speed, MULACC fix in tinygrad, Egraphs and tinygrad, Mac MetalCompiler improvements` 


- **AMD Emulator Reveals Debug Printing**: With the new AMD emulator (**AMD=1 MOCKGPU=1**), setting **DEBUG=3** prints all instructions when compiled, while **DEBUG=6** prints them as they run, as showcased in a [screenshot](https://cdn.discordapp.com/attachments/1068976834928193609/1465889714153193574/image.png?ex=697b686e&is=697a16ee&hm=485c88290bbec976b6b7ab93aed07b21a6a2ec8ba8b28806e14630c00b972b3c&).
- **Speeding Up Github Actions via Code Optimization**: The discussion emphasized that improving GitHub Actions' speed should focus on optimizing code rather than relying on faster hardware or rented resources, with a caution against prioritizing metrics over doing things the *right* way.
- **MULACC Fusion Fix**: A fix was proposed to add a pattern to fuse (**x << n) + c → MULACC(x, 2^n, c)** in `decompositions.py`, affecting integer MULACC with power-of-2 constants, as shown in [PR 14387](https://github.com/tinygrad/tinygrad/pull/14387).
- **Egraphs for Generic Fixes**: Members discussed using **egraphs** to generically fix issues, advocating for simplicity and considering tagging rewrites with their origin to track equivalences created during rewriting.
- **Improving Mac MetalCompiler**: Improving the hacks for the **MetalCompiler** on Mac was suggested, especially focusing on improvements and cleanups that reduce line count and improve readability.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1465806533995659365)** (4 messages): 

> `Container issues, macOS trust dance, Gatekeeper adds a tax, codesign step in `mojo build`` 


- **Container Issue Resolved with Additional Arguments**: A user resolved a container issue by adding `--cap-add=SYS_PTRACE --security-opt seccomp=unconfined` when running the container, or adding the equivalent to `.devcontainer/devcontainer.json`.
   - The provided solution ensures the container has the necessary permissions and security options configured correctly for debugging or tracing purposes.
- **macOS Trust Dance Affects First-Run Performance**: A member suggested that the performance difference between first and subsequent runs might be due to macOS Gatekeeper's *trust dance*.
   - They noted that clearing the quarantine `xattr` or ad-hoc codesigning could mitigate this, and wondered if a codesign step in `mojo build` could hide this entirely.


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1466146040255090884)** (2 messages): 

> `Mojo-GTK bindings, Mojo vs CUDA/HIP, Modular Team Updates` 


- **Modular Community meeting in February to discuss Mojo's prowess**: The Modular Community Meeting in February will cover **Mojo-GTK bindings**, **Mojo vs CUDA/HIP** performance, and **Modular Team Updates**.
   - The meeting is scheduled for **February 2nd at 10 AM PT** via Zoom, with more details available on the [Modular forum](https://forum.modular.com/t/february-community-meeting/2646).
- **Mojo-GTK Bindings Autogenerated**: **Hammad Ali** will present on autogenerated **GTK bindings for Mojo**.
   - This presentation will detail how GTK bindings are automatically generated, potentially improving the ease of creating **GUIs with Mojo**.
- **Mojo vs CUDA/HIP Performance**: **Tatiana Melnichenko** will share memory-bound bandwidth results and compute-bound gaps on **H100/MI300A** comparing **Mojo with CUDA/HIP**.
   - This talk should provide insights into **Mojo's performance characteristics** relative to established GPU programming models.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1465898202346819697)** (13 messages🔥): 

> `Compiler limitations in Mojo, Pythonic style deviations in Mojo, Rationale behind 'out' parameters, NVRO replacement, Mojo at ORNL paper` 


- **Slice Syntax Stumps `__getitem__` in Mojo**: A user reported errors using slice syntax (`0:2:1`) with `__getitem__` in a Mojo struct, noting it only works with `Int` input or explicit `Slice()` calls, and sought workarounds.
   - The error message is *invalid call to '__getitem__': value passed to 'index' cannot be converted from slice initializer to 'Variant[Slice, Int]'*.
- **Why Mojo Ditches Pythonic `out` Style**: Discussion revolved around Mojo's deviation from Pythonic styles, specifically concerning `out` parameters, with one member suggesting the design choice aligns more with Fortran.
   - Another added that *Python has no real equivalent in the sense that they are just type hints*.
- **`out` Parameter Peculiarities**: Members discussed that `out` parameters in Mojo name the location where the return value of a function will end up, especially useful for constructors to assign to `self` before it's fully initialized.
   - One member explained, *I know for constructors at least, you need a way to assign to “self” before “self” is fully initialized, and the `out self` was a way to name that.*
- **`out` as NVRO Nemesis**: `out` parameters serve as a Named Value Return Optimization (NVRO) replacement, providing a guarantee about the return value's destination, unlike relying on compiler optimization.
   - A member added: *Instead of hoping the compiler can figure it out, you get a guarantee.*
- **Mojo at ORNL Article Surfaces**: A member shared a link to *Mojo at ORNL*, specifically [https://arxiv.org/html/2509.21039v1](https://arxiv.org/html/2509.21039v1).
   - No further context was provided.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1465971590893146202)** (4 messages): 

> `Qwen3 embedding model, Nightly container builds, Stable MAX release` 


- **Qwen3 Embedding Model Accuracy Fix PR Incoming**: A member requested a review of their [PR for the Qwen3 embedding model](https://github.com/modular/modular/pull/5823), citing that the fix is important for getting much better accuracy.
   - Another member responded that new fixes likely won't be pulled into the upcoming release but would be available in the nightlies.
- **Nightly Container Builds Soon Available**: A member confirmed that since nightly container builds are provided, the changes should be available for their POC soon after it's merged.
   - They also shared a branch that reduces the fix to a single line: [https://github.com/modular/modular/compare/main...sbrunk:modular:qwen3-embedding-fix-norm-minimal](https://github.com/modular/modular/compare/main...sbrunk:modular:qwen3-embedding-fix-norm-minimal).
- **Stable MAX Release Results Improve**: The member mentioned that a merge would help other people get better results when trying the model via a stable MAX release.
   - They reduced the fix to a single line [here](https://github.com/modular/modular/compare/main...sbrunk:modular:qwen3-embedding-fix-norm-minimal).


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1465885050825347347)** (9 messages🔥): 

> `Manus Credit Usage, Cloud Browser Issues, AI Engineer Introductions, Context from Other Chats` 


- ****Manus's Credit Crunching Capabilities****: A user noticed that **Manus** seems to be using fewer credits for the same quality of work, questioning whether credit usage has improved.
   - No further details or confirmations were provided regarding potential changes to **Manus's** credit consumption algorithms.
- ****Cloud Browser Conundrums & Manus Support****: A user encountered issues with the **cloud browser**, receiving an error message stating that *the server is unavailable* and the website isn't loading.
   - Manus support requested the user's email, session link, and Manus User ID via DMs to investigate the issue further.
- ****AI Engineer Aces LLM Systems and Integrations****: An **AI + Full Stack Engineer** introduced themself, highlighting their expertise in LLM systems, autonomous agents, workflow automation, and multimodal AI, and shared their core skills such as [DSPy](https://dsppy.ai/), [LangChain](https://www.langchain.com/), [AutoGen](https://microsoft.github.io/autogen/), and [CrewAI](https://www.crewai.com/).
- ****Context Conundrum: Community Craves Cross-Chat Context for Manus****: A user suggested that enabling **Manus** to access context from other chats *would be a game changer*, indicating a desire for enhanced contextual awareness in the AI's responses.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1465862038147567808)** (5 messages): 

> `Prompt Optimizers, llmlingua, DSPy Skills, Claude Code Skills, DSPy ReAct Agent` 


- **Prompt Optimizers Seek Users**: A member inquired whether anyone has experience working with **prompt optimizers**.
   - Another member followed up asking whether anyone has tried using Skills within the dspy module.
- **llmlingua link shared**: A member shared a link to [llmlingua.com](https://llmlingua.com/).
   - The context surrounding the link was to a member inquiring about experience working with **prompt optimizers**.
- **DSPy ReAct Agent craves Skills**: A member asked about integrating **Claude code skills** (defined as .md files with associated .py scripts) into a **DSPy ReAct agent**.
   - They would like a DSPy ReAct agent or something be able to use those.


  