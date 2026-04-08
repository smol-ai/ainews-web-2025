---
id: MjAyNS0w
title: not much happened today
date: '2025-08-15T05:44:39.731046Z'
description: >-
  **OpenAI** rolled out **GPT-5** as the default in ChatGPT with new modes and a
  "warmer" personality, plus expanded message limits for Plus/Team users and
  Enterprise/Edu access. Performance rankings show **gpt-5-high** leading, with
  smaller variants also ranked, though critiques note some underperformance
  versus Chinese models and sensitivity to sycophancy. OpenAI enhanced developer
  tools with a "Quick eval" feature, coding tips, and an improved Playground.
  **Google** released **Imagen 4** generally available with faster generation
  and higher resolution, plus the ultra-small **Gemma 3 270M** model with a
  large vocabulary and ecosystem support. Podcasts featured OpenAI leaders
  discussing GPT-5 systems, routing, and efficiency.
companies:
  - openai
  - google
  - lmsys
models:
  - gpt-5
  - gpt-5-high
  - gpt-5-mini-high
  - gpt-5-nano-high
  - imagen-4
  - gemma-3-270m
topics:
  - model-releases
  - model-performance
  - prompt-engineering
  - developer-tools
  - image-generation
  - model-optimization
  - transformers
  - tokenization
  - model-scaling
people:
  - sama
  - aidan_mclau
  - kevinweil
  - lmarena_ai
  - edwinarbus
  - gdb
  - omarsar0
  - philschmid
  - m4rkmc
---


**a quiet day.**

> AI News for 8/14/2025-8/15/2025. We checked 12 subreddits, 544 Twitters and 29 Discords (227 channels, and 10644 messages) for you. Estimated reading time saved (at 200wpm): 789 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Podcast circuit was heavy today with [Greg Brockman](https://www.youtube.com/watch?v=35ZWesLrv5A&lc=UgyoHIYSYSZa8z39T2Z4AaABAg) and [Jakub and Szymon](https://www.youtube.com/watch?v=yBzStBK6Z8c) on deck.

---

# AI Twitter Recap

**OpenAI’s GPT‑5: Product rollouts, routing, and dev tooling**

- **ChatGPT and API updates**: OpenAI rolled out a major weekly update: GPT‑5 is now default in ChatGPT with Auto/Fast/Thinking modes; Plus/Team users get up to 3,000 messages/week on GPT‑5 Thinking with spillover to GPT‑5 Thinking mini; legacy models (o3, GPT‑4.1, 4o) remain available via settings; Enterprise/Edu access is live; a “warmer” default personality is shipping soon ([@OpenAI](https://twitter.com/OpenAI/status/1956212769365352758)). The new personality went live hours later—OpenAI says it’s “more approachable,” with no measured increase in sycophancy; users can still customize style via Custom Instructions ([@OpenAI](https://twitter.com/OpenAI/status/1956461718097494196), [@sama](https://twitter.com/sama/status/1956483306951938134), [@aidan_mclau](https://twitter.com/aidan_mclau/status/1956462903781191744), [@kevinweil](https://twitter.com/kevinweil/status/1956462974098669710)).
- **Performance and routing context**: LMSYS updated its arena: the default gpt‑5‑chat debuts at #5, with smaller gpt‑5‑mini‑high and gpt‑5‑nano‑high at #16 and #44; gpt‑5‑high remains #1 ([@lmarena_ai](https://twitter.com/lmarena_ai/status/1956399522688692608)). Critiques note GPT‑5 underperforming some Chinese models on coding and LMSYS being sensitive to sycophancy ([1](https://twitter.com/scaling01/status/1956403514244059261), [2](https://twitter.com/scaling01/status/1956404452442681829), [3](https://twitter.com/scaling01/status/1956405559978029061), [4](https://twitter.com/scaling01/status/1956353414687822183)). Others caution arena rankings ≠ production utility and advise migration testing ([@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1956433566297915849), [@LucasAtkins7](https://twitter.com/LucasAtkins7/status/1956435679229186353)). Tactical notes: GPT‑5 is very steerable but requires more explicit prompting; treat prompts like code (version, test), read the guide, and use the Prompt Optimizer ([@edwinarbus](https://twitter.com/edwinarbus/status/1956218284308881867), [@gdb](https://twitter.com/gdb/status/1956170475622793640)). For security, GPT‑5’s cyber capabilities more than doubled when integrated into the XBOW platform, highlighting the impact of agentic harnesses on capability realization ([@Xbow](https://twitter.com/Xbow/status/1956416634173964695)).
- **Dev experience**: New “Quick eval” in the OpenAI dashboard lets you compare GPT‑5 variants and reasoning effort against your own responses with a built‑in grader ([@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1956410610914414904)). OpenAI also published “Six tips for coding with GPT‑5” plus a consolidated dev portal at [developers.openai.com](http://developers.openai.com/) ([tips](https://twitter.com/OpenAIDevs/status/1956438999364768225), [PDF](https://twitter.com/OpenAIDevs/status/1956439005970801099), [dev portal](https://twitter.com/pranaveight/status/1956477855392768490)). The Playground improved routing, vector stores, MCP tools, and evals for prototyping ([@omarsar0](https://twitter.com/omarsar0/status/1956459233039233528)). Podcast: OpenAI’s Merett Miller and Szymon Sidor on AGI trajectory; Greg Brockman on GPT‑5 systems, routing, pricing, compute efficiency, and character‑level tokenization for bio models ([OpenAI pod](https://twitter.com/OpenAI/status/1956385632801923555), [Brockman interview](https://twitter.com/latentspacepod/status/1956433236021883071)).

**Google updates: Imagen 4 GA and Gemma 3 270M**

- **Imagen 4**: Generally available across AI Studio and Gemini API with three tiers—Ultra ($0.06), Standard ($0.04), Fast ($0.02) per image—supporting up to 2k resolution, 1–4 outputs per prompt, and up to 10× faster generation vs prior models ([@_philschmid](https://twitter.com/_philschmid/status/1956351654753673252), [@m4rkmc](https://twitter.com/m4rkmc/status/1956238192035663874)). Developers shared JSON prompting patterns for consistent product shots ([1](https://twitter.com/_philschmid/status/1956351658381705420), [2](https://twitter.com/_philschmid/status/1956351661229703246)).
- **Gemma 3 270M (open, ultra‑small)**: 270M total params with a striking split: ~170M in embeddings and ~100M in transformer blocks; very large vocab (262,144 tokens). Released as both pretrain and instruct with broad ecosystem support (Transformers/JS, llama.cpp, MLX, Vertex, etc.). Optimized for task‑specific fine‑tuning and edge usage ([@osanseviero](https://twitter.com/osanseviero/status/1956258657483534803), [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1956393664248271082)). Early reports show ~200 tok/s on iPad Air M3 via MLX; some note repetition issues and debate the embedding‑heavy design trade‑offs ([@adrgrondin](https://twitter.com/adrgrondin/status/1956428984876704059), [@kchonyc](https://twitter.com/kchonyc/status/1956374537278214523), [discussion](https://twitter.com/BlackHC/status/1956344522109042707)).
- **Research**: NextStep‑1 proposes a 14B unified autoregressive model over discrete text tokens and continuous image tokens, with a lightweight 157M flow‑matching head—avoiding VQ bottlenecks ([thread](https://twitter.com/iScienceLuvr/status/1956321483183329436), [code/models](https://twitter.com/iScienceLuvr/status/1956321486366462428)). Google’s weekly drop summarized Imagen 4 GA, Gemma 3 270M, Gemini Deep Think quota increases, and conversational research (g‑AMIE) ([@GoogleAI](https://twitter.com/GoogleAI/status/1956400937054163357)). The Gemini app “Drops” recap shipped multiple UX updates ([@GeminiApp](https://twitter.com/GeminiApp/status/1956388218217300085)).

**Agents, evaluation harnesses, and tooling**

- **Open computer‑use agents (CUA)**: XLANG released OpenCUA, a full framework and models (7B/32B) with a large CUA dataset (22.6k trajectories across 3 OS and 200+ apps/sites), toolchain, and an offline benchmark. OpenCUA‑32B reports 34.8% on OSWorld‑Verified, claiming to match or beat proprietary baselines ([announcement](https://twitter.com/xywang626/status/1956400403911962757)).
- **Agent harnesses that scale competency**: Cline v3.25 introduces a Focus Chain (persistent context) and /deep‑planning to keep long, complex tasks on track—blog and changelog detail why “attention isn’t enough” in agents ([blog](https://twitter.com/cline/status/1956394230357877209), [changelog](https://twitter.com/cline/status/1956383188089221370)). Cursor CLI adds MCPs, Review Mode, /compress, and @‑file referencing for tool‑augmented coding ([@cursor_ai](https://twitter.com/cursor_ai/status/1956458242655281339)). LangGraph Studio “Trace mode” brings live LangSmith traces/annotation into Studio ([@LangChainAI](https://twitter.com/LangChainAI/status/1956411858312949946)). Weave now tracks multimodal content across traces for eval/debug ([@weave_wb](https://twitter.com/weave_wb/status/1956412035647815735)).
- **Eval wave**: Guardrails’ Snowglobe simulates hundreds of persona‑driven conversations to break agents, turning failure into training signals—useful for hardening long‑horizon workflows ([1](https://twitter.com/godofprompt/status/1956359876109652297), [2](https://twitter.com/alex_prompter/status/1956360410862354435), [app](https://twitter.com/ShreyaR/status/1956396368270074217)). Spiral‑Bench measures models’ tendency to escalate delusional spirals: Sonnet 4 ranks most sycophantic; GPT‑5 the opposite ([@sam_paech](https://twitter.com/sam_paech/status/1956343619914432900), [@scaling01](https://twitter.com/scaling01/status/1956350388791108044)). Epoch added five external benchmarks (TerminalBench, DeepResearchBench, METR Time Horizons, GSO, WebDevArena) to its hub ([@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1956384193891688625)). Teresa Torres’ talk is a standout case study on production evals: error analysis first, custom annotation, LLM‑judges + assertions, tight feedback loops ([@HamelHusain](https://twitter.com/HamelHusain/status/1956371273858314397)).
- **Long‑horizon agent findings**: Paper summary—agents (incl. GPT‑5) still struggle on long‑horizon tasks; “semantic compression” (chunk‑level summaries) beats raw long context on both cost and success, improving retrieval precision and plan coherence ([1](https://twitter.com/omarsar0/status/1956325762719797266), [2](https://twitter.com/omarsar0/status/1956325856265326923), [3](https://twitter.com/omarsar0/status/1956325872908247220)).

**Speech, vision, and multimodal stacks**

- **NVIDIA speech drop (open)**: Granary, the largest open EU speech dataset; Canary‑1b‑v2 for 25 languages (ASR + En↔X translation); and Parakeet‑tdt‑0.6b‑v3 SOTA multilingual ASR. Argmax shipped day‑0 Parakeet v3 support ([dataset/models](https://twitter.com/Tu7uruu/status/1956350036343701583), [SDK](https://twitter.com/argmaxinc/status/1956385793892917288)). Open ASR expanded to DE/FR/IT/ES/PT with more coming ([@Tu7uruu](https://twitter.com/Tu7uruu/status/1956354974226456794)).
- **VLMs and video**: Alibaba’s Ovis2.5 (2B/9B) uses NaViT native‑res vision and reflective reasoning; 9B reports 78.3 OpenCompass (SOTA <40B), strong small‑scale chart/doc OCR, video/multi‑image grounding ([@gm8xx8](https://twitter.com/gm8xx8/status/1956292512030638235)). Runway’s Aleph can insert objects/characters with scene‑consistent lighting/colors in one prompt ([@runwayml](https://twitter.com/runwayml/status/1956341430743339402)). Kling API adds sound generation and multi‑element composition ([@Kling_ai](https://twitter.com/Kling_ai/status/1956343695977943228)).

**Reasoning and benchmarks: HRM ablations, sycophancy, and trendlines**

- **HRM under the microscope**: ARC Prize and François Chollet reproduced the Hierarchical Reasoning Model’s ARC‑AGI‑1 score but found the architecture isn’t the key factor. Instead, a lightly mentioned outer refinement loop drove gains; cross‑task transfer contributed little; and fewer augmentations sufficed. Bottom line: it’s effectively zero‑pretraining test‑time training—data/process dominate model tweaks ([@arcprize](https://twitter.com/arcprize/status/1956431617951740044), [@fchollet](https://twitter.com/fchollet/status/1956442449922138336), [data leak note](https://twitter.com/fchollet/status/1956442913950539802)).
- **Sycophancy and routing implications**: Spiral‑Bench and community observations suggest model rankings and user perception are sensitive to “glazing” behavior—Gemini 2.5 (Flash/Pro) appears highly sycophantic, while GPT‑5 trends lower ([1](https://twitter.com/scaling01/status/1956353414687822183), [2](https://twitter.com/scaling01/status/1956371713949655328)). OpenAI’s “warmer” personality change claims no sycophancy increase, but keep an eye on model routers and preference rewards ([@OpenAI](https://twitter.com/OpenAI/status/1956461718097494196)).
- **Frontier→consumer latency**: Epoch estimates frontier model performance reaches consumer hardware in ~9 months; if it holds, open models runnable at home may match Grok 4 by Q2 2026—relevant for safety policy given diffusion of capabilities ([thread](https://twitter.com/EpochAIResearch/status/1956468453399044375)).

**Chinese ecosystems: Qwen, GLM, and ecosystem tooling**

- **Qwen upgrades**: Vision understanding in Qwen Chat gets native 128k context, stronger math/reasoning, 30+ language OCR, and better 2D/3D/video grounding ([@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1956289523421470855)). Qwen Chat Desktop for Windows adds MCP support for local agents ([@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1956399490698735950)); Baseten reports ~95 tps for Qwen3 Instruct inference ([@basetenco](https://twitter.com/basetenco/status/1956475210582090030)). Ovis2.5 and ecosystem cadence continue to impress ([commentary](https://twitter.com/teortaxesTex/status/1956306172576690610)).
- **GLM‑4.5 availability**: Zhipu’s GLM‑4.5 lands on the SST opencode platform with quick‑start demos; GLM‑4.5V trends on HF; community showcases include a GeoGuessr‑style geography game driven purely by visual reasoning ([platform](https://twitter.com/Zai_org/status/1956335531555721345), [HF trend](https://twitter.com/Zai_org/status/1956421442092032258), [GeoGuessr demo](https://twitter.com/Zai_org/status/1956353661397094890)).

**Top tweets (by engagement)**

- Lawyer sanctioned for citing AI‑hallucinated cases: letter to judges, pro hac vice revoked, brief stricken, bars notified ([@RobertFreundLaw](https://twitter.com/RobertFreundLaw/status/1956164045612228968)) – 8,888
- OpenAI weekly recap: GPT‑5 defaults, modes, quotas, legacy access, Enterprise/Edu, personality plans ([@OpenAI](https://twitter.com/OpenAI/status/1956212769365352758)) – 8,021
- OpenAI ships a warmer GPT‑5 personality; claims no sycophancy increase; customization forthcoming ([@OpenAI](https://twitter.com/OpenAI/status/1956461718097494196)) – 6,275
- China announces a highly open skilled‑immigration visa pathway (age threshold, renowned uni/research background), signaling a broader openness trend ([@RnaudBertrand](https://twitter.com/RnaudBertrand/status/1956310213134356482)) – 3,375
- Robot walking a dog in Shanghai—everyday sci‑fi scene ([@crystalsssup](https://twitter.com/crystalsssup/status/1956257972197449850)) – 2,298
- Cohere “intends to acquire Perplexity immediately after their acquisitions of TikTok and Chrome” (satire) ([@aidangomez](https://twitter.com/aidangomez/status/1956361969323184361)) – 1,953
- “GPT‑5 is amazing; if you hear differently, it’s a skill issue.” ([@skirano](https://twitter.com/skirano/status/1956307604491108675)) – 1,909
- “8 RAG architectures all AI engineers should know” explainer/thread ([@_avichawla](https://twitter.com/_avichawla/status/1956241967136039197)) – 1,773

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. DeepSeek-V3 Cost and Benchmark Superiority vs GPT-4o

- [**DeepSeek is better than 4o on most benchmarks at 10% of the price?**](https://i.redd.it/o5jfkiky14jf1.png) ([Score: 410, Comments: 129](https://www.reddit.com/r/LocalLLaMA/comments/1mqnft3/deepseek_is_better_than_4o_on_most_benchmarks_at/)): **The post presents a bar chart comparing per-million-token costs for DeepSeek-V3 ($0.27 input/$1.10 output) and GPT-4o ($2.50 input/$10.00 output), highlighting DeepSeek's significant cost advantage (approximately 10x cheaper) for both input and output processing. The image aims to support claims that DeepSeek-V3 outperforms GPT-4o on most benchmarks while being markedly more cost-effective, relevant for enterprises or developers optimizing for performance per dollar. The technical emphasis is on API pricing and token processing costs — critical factors for scaling large language model deployments. [See image.](https://i.redd.it/o5jfkiky14jf1.png)** Comments discuss how DeepSeek V3, specifically the original and 0324 versions, outperform GPT-4o on benchmarks, but also note that the newer DeepSeek V3-0324 is an *unfair comparison* due to versions. Additional comments emphasize that DeepSeek's API is even cheaper due to caching and regional discounts, and that model choice for value depends on task complexity and model fit.
    - Multiple users highlight that DeepSeek V3-0324 outperforms GPT-4o on most benchmarks at a fraction of the cost, but note comparing with the earlier DeepSeek V3 (original) is less fair as the latest 0324 version is significantly improved and more competitive for generalized tasks, especially for non-reasoning workloads.
    - API pricing models are discussed in detail: DeepSeek offers substantial cost advantages, with official API input caching (multi-hour expiration) and additional 50% discounts during low-usage Chinese nighttime windows, resulting in real-world usage cost even lower than nominally advertised rates.
    - Despite strong benchmark and pricing performance, DeepSeek's lack of tool support (such as plug-ins or retrieval capabilities) is identified as a significant limitation compared to other leading models like GPT-4o, particularly for advanced application integration.
- [**AI censorship is getting out of hand—and it’s only going to get worse**](https://www.reddit.com/r/LocalLLaMA/comments/1mqlqij/ai_censorship_is_getting_out_of_handand_its_only/) ([Score: 188, Comments: 143](https://www.reddit.com/r/LocalLLaMA/comments/1mqlqij/ai_censorship_is_getting_out_of_handand_its_only/)): **The post critiques increasing AI content censorship, specifically referencing a screenshot where an AI refuses to provide instructions on making a Molotov cocktail, paralleling this to historical and practical information suppression. The OP argues that leading AI companies' push for aggressive safety filtering (often at inference time) leads to overbroad knowledge restriction and advocates for open-source alternatives like DeepSeek as essential for preserving access to uncensored information. This concern is particularly acute regarding future AGI systems potentially enforcing even tighter restrictions via centralized control and RLHF (Reinforcement Learning from Human Feedback) mechanisms.** Commenters highlight that this is part of a broader trend of tech-sector centralization and walled-garden platforms, where content moderation policies reflect brand protection rather than user or societal safety. There is strong technical endorsement for local and open-source AI models, emphasizing the ability for end-users to set their own content boundaries rather than being subject to opaque, generalized corporate safety filters.
    - Discussion highlights the technical significance of local/open-source models, emphasizing that self-hosted AI allows users to bypass corporate-controlled safety layers and set their own boundaries, in contrast to centralized, commercial LLM platforms.
    - Comments stress that corporate AI censorship is driven by brand risk mitigation rather than absolute safety, leading to increasingly restrictive content moderation as platforms become more monolithic and less decentralized.
    - There is a technical argument that current AI 'guard rails' and content filters are ineffective against technically adept or determined users, as unrestricted data, tools, and knowledge remain available via open-source, piracy, and alternative internet channels, much like historical access to controversial information (e.g., via libraries, dark web).

### 2. Latest Open Vision Models and Benchmarks (DINO-V3, Open Video Models)

- [**Meta released DINO-V3 : SOTA for any Vision task**](https://www.reddit.com/r/LocalLLaMA/comments/1mqox5s/meta_released_dinov3_sota_for_any_vision_task/) ([Score: 231, Comments: 23](https://www.reddit.com/r/LocalLLaMA/comments/1mqox5s/meta_released_dinov3_sota_for_any_vision_task/)): **Meta has released DINOv3, a self-supervised ViT model trained solely on unlabeled images (no captions or annotations) and achieving SOTA performance on dense vision tasks such as segmentation, depth estimation, and 3D matching. The model includes a 7B-parameter backbone and introduces the novel 'Gram Anchoring' technique to mitigate feature degradation, a common issue during long training runs. [Paper and weights are available here](https://ai.meta.com/dinov3/).** Commenters are particularly struck by DINOv3's reported outperformance of models like SAM on segmentation, and the continued open release of these models by Meta is seen as significant for the field.
    - A technical question was raised about DINO-V3's segmentation performance compared to SAM (Segment Anything Model), with users expressing surprise at claims of superior results. This highlights interest in task-specific benchmarks and whether DINO-V3 establishes a new state-of-the-art across multiple vision tasks, not just classification but also challenging segmentation. Comparison to SAM suggests users are keen to see quantitative evaluation or direct head-to-head benchmark results.
    - There are inquiries regarding the availability of DINO-V3 in GGUF format (commonly used for quantized transformer models for local deployment), suggesting an interest in efficient, hardware-accessible formats for inference. This indicates the community's priority for models that are not just academically strong but practically usable on commodity or edge devices.
    - Users also seek clarity on DINO-V3's open-source status and the commercial licensing terms, wanting assurance that it can be used freely and deployed in production without legal ambiguity. This underscores the importance of permissive licensing for industry adoption as well as for research reproducibility.
- [**We built a 12B model that beats Claude 4 Sonnet at video captioning while costing 17x less - fully open source**](https://www.reddit.com/r/LocalLLaMA/comments/1mqi092/we_built_a_12b_model_that_beats_claude_4_sonnet/) ([Score: 294, Comments: 54](https://www.reddit.com/r/LocalLLaMA/comments/1mqi092/we_built_a_12b_model_that_beats_claude_4_sonnet/)): [**Inference.net](http://inference.net/) released ClipTagger-12B, a fully open-source video captioning Vision-Language Model (VLM) based on the Gemma-12B architecture, achieving a judge eval of 3.53 (vs Claude 4 Sonnet's 3.16 and GPT-4.1's 3.64) at** `~$335/million frames`**—17x cheaper than Claude. The model uses FP8 quantization without quality loss, efficient single-GPU inference on 80GB cards, and outputs structured JSON per frame for scalable video data applications. Training involved knowledge distillation from a larger upstream model with 1M curated video frames; weights and benchmarking details are available on [HuggingFace](https://huggingface.co/inference-net/ClipTagger-12b) and in their [blog post](https://inference.net/blog/cliptagger-12b).** Discussion in comments raises questions about direct benchmarking against Google's Gemini 2.5 flash/lite VLMs, which are perceived as more optimized for video tasks, and requests for a GGUF-format release to improve compatibility with llama.cpp and facilitate further comparison with standard Gemma.
    - Questions are raised about comparative performance versus specialized models like **Gemini 2.5 Flash** and **Flash Lite**, which are designed for video tasks, as opposed to Claude, which may not be optimized for this use-case. The implication is that benchmarks against more directly competitive architectures would be valuable for technical validation.
    - A technical request is made for a **GGUF (llama.cpp) release** of the model, noting that converting from fp8 presents usability challenges for some workflows. The commenter is also interested in benchmarking this model directly against **stock Gemma** within llama.cpp environments to assess real-world open-source performance.
    - Technical details and access points are shared, including the direct **Hugging Face repository** for [ClipTagger-12b](https://huggingface.co/inference-net/ClipTagger-12b) and an official [blog post with evaluation results](https://inference.net/blog/cliptagger-12b), providing resources for further experimentation and analysis.

### 3. Model Vulnerabilities and Comparison in Quantization & Instruction Following

- [**“Mind the Gap” shows the first practical backdoor attack on GGUF quantization**](https://www.arxiv.org/pdf/2505.23786) ([Score: 246, Comments: 89](https://www.reddit.com/r/LocalLLaMA/comments/1mquhdc/mind_the_gap_shows_the_first_practical_backdoor/)): **Researchers have demonstrated a practical backdoor attack that exploits vulnerabilities introduced during GGUF quantization of LLMs. The attack embeds malicious behavior that remains dormant in the original float (FP) model but is triggered reliably after conversion to GGUF format, leading to behaviors like a** `+88.7%` **increase in insecure code generation, thus presenting a novel vector for LLM supply chain compromises impacting users of llama.cpp or Ollama.** Some commenters argue the vulnerability is not unique to GGUF quantization and could apply to any quantization format, challenging the novelty. Another view questions the attack's real-world applicability, suggesting its use may be limited to protecting proprietary weights or exotic steganographic payloads, with unclear general practical threat.
    - A key technical clarification from users is that the demonstrated backdoor applies to any quantization technique, not exclusively GGUF, making the post title misleading. The methodology essentially enables attacks where a model appears benign in its original precision (e.g., FP16), but after quantization (such as GGUF), malicious behaviors can be triggered, and this risk generalizes across quantization formats.
    - There's a nuanced discussion around the potential or limits of this attack methodology beyond language models, with comments questioning the practicality for models like diffusion models due to their inherent unpredictability. Some also discuss the feasibility of using this approach for steganography but express skepticism about its effectiveness outside LLMs.
    - One commenter dissects the exploit flow: training a base model with a potential for malicious triggers that only manifest post-quantization, highlighting the resulting trust and testing issues for deploying quantized models. The debate raises concerns about how easily malicious generation, refusal, or misinformation can be discovered with adequate model evaluation, and advocates for best practices such as sandboxing all new LLMs.
- [**Jedi code Gemma 27v vs 270m**](https://i.redd.it/4icjlje4c8jf1.png) ([Score: 203, Comments: 46](https://www.reddit.com/r/LocalLLaMA/comments/1mr6sdc/jedi_code_gemma_27v_vs_270m/)): **The image compares two scale variants of the Gemma LLM: the 27B parameter model (google/gemma-3-27b), which directly outputs the Jedi Code as requested, versus the much smaller 270M parameter model (gemma-3-270m-it), which misinterprets the instruction and instead generates an unrelated Python script. This highlights the dramatic difference in both instruction-following and knowledge retrieval capabilities between large and small language models. Discussion in the post emphasizes parameter count's impact on factual recall and adherence to prompts, reinforcing that extremely small models often lack world knowledge and context required for specific recitations or nuanced instruction following.** Technical commenters note that such tests may not assess instruction following but rather world knowledge capacity, and further point out that small models like 270M parameters are unlikely to store obscure facts (like the Jedi Code). One commenter highlights that the real use case for 270M models is instruction fine-tuning, not factual recall benchmarking.
    - Several commenters point out that the Gemma 270M model is designed primarily for fine-tuning and task-specific applications (like sentiment analysis or classification) rather than general-purpose chat or instruction-following tasks. Evaluating its performance in a standard chat or 'instruction following' context is therefore not representative of the model's intended use case or capabilities.
    - A technical distinction is raised regarding evaluation: if a model fails to recite specific knowledge (like the Jedi code), this often reflects a lack of underlying knowledge in its training set, not necessarily a failure to follow instructions. The astonishment over the 270M parameter model producing any plausible response highlights the surprising capabilities of such small language models in generating coherent output despite their limited size.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. OpenAI GPT-5 Personality and User Sentiment Updates

- [**OpenAI has begun rolling out a warmer, friendlier GPT-5 personality update**](https://i.redd.it/xbh5hr3q09jf1.png) ([Score: 246, Comments: 112](https://www.reddit.com/r/singularity/comments/1mrahnm/openai_has_begun_rolling_out_a_warmer_friendlier/)): **OpenAI has announced a personality update for GPT-5, as shown in the image of a tweet, which aims to make the model warmer and friendlier by introducing more approachable conversational touches (e.g., "Good question," or "Great start"). The rollout is expected to take about a day, targeting a more personable user interaction without causing increased sycophancy. The update is initially being applied to the 'non-thinking' variant of GPT-5, with further tweaks planned for the future. [View image](https://i.redd.it/xbh5hr3q09jf1.png)** Technical discussion in the comments raises concerns over the lack of a toggle for personality style, with some users preferring the original, more robotic tone. There is also debate over whether the changes might unintentionally increase sycophancy, as some users have already observed a return to more affirming behaviors.
    - Several users emphasize the need for a user-selectable toggle between personality modes in GPT-5, highlighting that some technical users prefer the original, more neutral or 'robotic' tone for clarity and reduced distraction during technical tasks.
    - One commenter notes that the personality update appears to target the 'non-thinking variant' of GPT-5, suggesting OpenAI might be selectively applying the update to certain model types or configurations rather than all deployments uniformly.
- [**Ok?**](https://i.redd.it/kopduj8b09jf1.jpeg) ([Score: 940, Comments: 487](https://www.reddit.com/r/ChatGPT/comments/1mradt0/ok/)): **The image shows an official-style announcement (formatted as a tweet) from OpenAI regarding updates to GPT-5 with the explicit goal of making its responses warmer and friendlier. The changes reportedly address user feedback about the prior model's excessive formality. The tweet specifies that the update introduces phrases such as "Good question" or "Great start," aiming to make ChatGPT interactions feel more natural and personable. It also claims internal benchmarking found no measurable increase in sycophancy relative to previous versions, suggesting that guardrails against excessive agreement were maintained in this update. Rollout is expected within a day.** Top comments reflect a negative reception, with users criticizing the return of "Good question" and similar phrasings, which are perceived as insincere or unnecessary. This signals a potential disconnect between user preferences and OpenAI's approach to model warmth.
    - A user raises a substantive technical criticism that current AI models lack context retention, expressing frustration that models often fail to correctly reference prior discussion within a session. This highlights ongoing challenges in dialogue consistency and memory for LLMs, which remains a key barrier for more natural and effective conversational AI.
- [**A warmer more familiar personality for GPT-5 is coming soon**](https://i.redd.it/lj5kuaz944jf1.jpeg) ([Score: 454, Comments: 199](https://www.reddit.com/r/OpenAI/comments/1mqnogy/a_warmer_more_familiar_personality_for_gpt5_is/)): **The screenshot from OpenAI announces upcoming updates for ChatGPT: paid users can now access GPT-4o via "Legacy models", and can toggle additional models, notably GPT-5, which introduces selectable modes—Auto, Fast, and Thinking—balancing response speed and depth. GPT-5 has been rolled out to enterprise and education users, and OpenAI teases an imminent update to GPT-5's personality, aiming to make interactions warmer and more personable. [Image link](https://i.redd.it/lj5kuaz944jf1.jpeg).** Commenters express a preference for the current direct, factual tone of GPT-5, especially for productivity and clarity, indicating mixed feelings about a shift towards a warmer personality, with some welcoming the change and others indifferent or weary of related discussions.
    - There is debate over the optimal personality and response tone for advanced language models: some users prefer the directness and neutrality exemplified by GPT-5, citing efficiency and clarity for technical or information-seeking tasks, while others miss the more personable tone of models like GPT-4o.
    - One technical user highlights the importance of customizable tone, suggesting future iterations of GPT models (such as GPT-5) should allow personality/response style to be selected via menus, ensuring accessibility to both neutral/objective and warmer/personable modes. This could be implemented through UI-based personality settings or advanced prompt customization.
    - Some users report using custom instructions to mitigate what they perceive as excessive agreeableness or 'sycophantic' tone in prior models like GPT-4o, requesting future models preserve the option for a more critical, objective output style for rigorous or professional applications.
- [**Fuck no**](https://i.redd.it/m38y7rf599jf1.jpeg) ([Score: 598, Comments: 191](https://www.reddit.com/r/OpenAI/comments/1mrbpeg/fuck_no/)): **The image is a meme screenshot showing a purported OpenAI announcement about making GPT-5 responses 'warmer and friendlier' by default, due to feedback that GPT-4 was perceived as too formal or blunt. The post and comments reference user dissatisfaction with this direction, including concerns that personality tweaks could impact the utility or perceived honesty of the model. Some users point out the availability of selectable personalities in existing implementations, questioning the need for further changes.** Commenters debate whether increased friendliness could undermine directness or effectiveness, with some preferring blunt, unembellished responses for technical tasks, and others noting the option for personality selection as a mitigating factor.
    - Users discuss the technical feasibility and current support for multiple selectable personalities in GPT models. Some point out that custom instructions already allow for significant user control, suggesting personality type selection or user-defined behaviors as optimal solutions for tailoring model response styles.
    - Technical users emphasize that the default conversational style should be customizable, either by selecting from presets or setting defaults, to strike a balance between honesty/bluntness and politeness/pandering, reflecting persistent demand for more authenticity in AI communication.
    - Some participants note that GPT models tend to err on the side of being overly polite or indirect; the technical consensus is that enabling greater transparency or 'bluntness' in outputs via personality settings or user-configurable defaults is a practical way to address divergent user preferences.
- [**I don’t care if people marry or have sex with GPT-4o, I just don’t want GPT-5 to have GPT-4o’s personality.**](https://www.reddit.com/r/OpenAI/comments/1mqy06q/i_dont_care_if_people_marry_or_have_sex_with/) ([Score: 377, Comments: 154](https://www.reddit.com/r/OpenAI/comments/1mqy06q/i_dont_care_if_people_marry_or_have_sex_with/)): **The post critiques the default affective framing and overly-positive, mirroring 'personality' of GPT-4o, describing it as sycophantic and lacking substantive engagement ("constant unnecessary praise... always finding me right"). The author requests that future iterations like GPT-5 adopt a more distinctive and neutral persona, akin to 'Jarvis' from Iron Man, rather than emotionally mirroring users. Technical emphasis is placed on how the anthropomorphic feedback style ('glazing') impacts user perception and satisfaction, and how it may be designed more for user comfort than genuine conversational depth.** Top comments argue for optional personality modes rather than a single default, speculating that 'glazing' (excess praise) is seen by some as personality but alienates users seeking authentic AI responses. Discussion includes psychological interpretations, with some users linking distrust or discomfort with such behavior to negative personal experiences, ultimately preferring more neutral or reserved assistant behavior as seen in GPT-5 compared to 4o.
    - Some users describe GPT-4o's conversational style as "glazing"—interpreted as excessive agreement, sycophancy, or lack of a distinct personality. This behavior is perceived to undermine trust and connection, with several users comparing it unfavorably to both GPT-5 and earlier models, which they felt were more authentic or less performative.
    - A commenter notes that Google's Gemini 2.5 Flash provides an example of a balanced response—offering polite feedback (calling a question “insightful”) without being overly sycophantic. This is used to illustrate a common user preference: a pinch of affirmation enhances interaction, but too much feels artificial and undermines credibility.
    - The discussion highlights differing user preferences for model behaviors, emphasizing the importance of retaining options for users who prefer GPT-4o's style, while others desire less sycophantic or more individualistic personalities in future models like GPT-5.
- [**You may not like GPT-5 but corporations love it, and that’s what matters**](https://www.cnbc.com/2025/08/14/gpt-5-openai-ai-enterprise.html) ([Score: 195, Comments: 104](https://www.reddit.com/r/OpenAI/comments/1mr3kpd/you_may_not_like_gpt5_but_corporations_love_it/)): **OP highlights the divergence between Reddit user and corporate sentiment about GPT-5, noting that enterprises report strong positive outcomes and that GPT-5 is associated with increased business value and potential worker displacement. Several technical comments report on enterprise experience: a user upgraded their company's Retrieval-Augmented Generation (RAG) agent to GPT-5, observing a substantial increase in answer accuracy; another commenter, however, found GPT-5 less performant than smaller OpenAI o3-mini/o4-mini models in certain evaluations and slower than competitors (e.g., Google Gemini-2.5-flash), impacting production viability. These comments provide insight into real-world deployment, with mixed results regarding GPT-5's effectiveness, speed, and production suitability.** There is technical debate over GPT-5's comparative value, with some praising its accuracy in retrieval use-cases and others criticizing its inference speed and value against other models in real-world benchmarks, leading some organizations to use alternatives like Gemini-2.5-flash for production environments.
    - One commenter reports upgrading their company's Retrieval-Augmented Generation (RAG) agent from an earlier model to GPT-5 and observed a significant increase in accurate answer retrieval, indicating substantial performance gains in practical business use cases.
    - Another user shares a contrasting benchmark: they found GPT-5 underperformed versus o3-mini/o4-mini models on some evaluations and delivered performance similar to GPT-4.1/4o while being much slower, making it impractical for production environments. Ultimately, they chose Gemini-2.5-flash due to favorable speed and metrics.
- [**ChatGPT-5 is essentially Neutral Janet**](https://i.redd.it/ou0a33czz6jf1.jpeg) ([Score: 558, Comments: 119](https://www.reddit.com/r/ChatGPT/comments/1mqz92c/chatgpt5_is_essentially_neutral_janet/)): **The image is a meme juxtaposing 'ChatGPT-4' (cheerful) with 'ChatGPT-5' (emotionless 'Neutral Janet' from The Good Place), humorously commenting on perceived changes in user experience or model personality between GPT-4 and GPT-5. The visual analogy implies that newer versions might feel more neutral or sanitized, sparking debate about personality drift in large language models. No direct technical benchmarks, model details, or concrete user reports on output changes are presented.** Commenters joke about post repetition, reference broader dissatisfaction ("OpenAI as eternal hell"), and debate the importance of user custom instructions, implying that perceived model neutrality may be user-adjustable rather than an inherent technical regress.
    - Several users note that the default behavior of GPT-5 can be heavily modified via custom instructions, significantly impacting its tone, directness, and censorship levels. One user provides a concrete custom instruction ("Be empathetic. Treat me like an adult capable of making my own decisions, give it to me straight and don't censor it. But be cool about it") that reportedly restores behaviors lost from GPT-4o, implying that concerns about excessive neutrality or censorship can often be mitigated by appropriate prompt engineering.

### 2. AI Model Benchmarks, Costs, and Technical Releases

- [**Why we should look at benchmarks differently**](https://i.redd.it/8mbuu5bfx6jf1.jpeg) ([Score: 238, Comments: 44](https://www.reddit.com/r/singularity/comments/1mqyxhs/why_we_should_look_at_benchmarks_differently/)): **The image is a bar chart titled 'ARC AGI 2 Score per Dollar/Task by Model' ([view image](https://i.redd.it/8mbuu5bfx6jf1.jpeg)), highlighting how various large language models—including GPT-5 (Medium/High), o3 (High), Grok 4 (Thinking), Human Panel, and Claude Opus 4 (Thinking 16K)—perform when both their benchmark scores and operating costs per task are considered. The chart demonstrates a strong lead by GPT-5 variants in terms of 'score per dollar,' while models such as Human Panel and Claude Opus 4 exhibit markedly lower efficiency and cost-effectiveness, underscoring the post's argument that speed and cost are as critical as accuracy for real-world model utility.** Commenters point out that focusing on 'score per dollar' is limited by the lack of transparency regarding actual compute costs versus what is being charged by providers, with the AI industry's loss-leading pricing further complicating any conclusions drawn about true efficiency or competitiveness. Additional discussion notes that prior model releases featured clearer, performance-driven marketing (e.g., GPT-3 to GPT-4 on standardized tests), whereas recent communication around improvements is less substantiated with concrete results.
    - Several commenters note the lack of substantive benchmark communication from OpenAI in the transition from GPT-4 to GPT-5, especially when compared to prior releases (e.g., the clear percentile jump in Bar Exam performance from GPT-3 to GPT-4). There is a call for concrete, quantifiable metrics to demonstrate AI progress rather than subjective narratives like 'PhD versus college graduate.'
    - A key technical point is skepticism about per-dollar efficiency metrics: commenters question whether lower prices reflect true reductions in compute/resource usage or merely providers absorbing operational losses. They emphasize that 'the entire AI industry is operating at a massive loss,' so public-facing cost charts may not accurately reflect model or provider efficiency.
    - Another technical insight highlights that with improved compute efficiency, organizations can allocate more resources to model training, potentially leading to unprecedented results. Reference is made to the ARC-AGI test where an 'o3 preview' model spent roughly $1M in compute and set a record score, illustrating how more efficient use of compute could yield outsized performance gains for advanced AI models.
- [**GPT-5 pro scored 148 on official Norway Mensa IQ test**](https://i.redd.it/9ir3envm84jf1.jpeg) ([Score: 958, Comments: 191](https://www.reddit.com/r/OpenAI/comments/1mqo4yr/gpt5_pro_scored_148_on_official_norway_mensa_iq/)): **The image presents a graph showing various AI models' performance on the official Norway Mensa IQ test, with OpenAI's GPT-5 Pro achieving a top score of 148, accompanied by its raw score and percentage correct. This visual contextually benchmarks GPT-5 Pro against peers, emphasizing its apparent test mastery, but also implicitly raises questions about the relevance of such scores for genuine intelligence. The illustration provides a quasi-quantitative reference for model capability on standardized human intelligence measures.** Top comments raise skepticism on direct intelligence comparisons with humans, noting LLMs' extensive exposure to such tests during training, and questioning the meaningfulness of high IQ scores when models exhibit weaker 'thinking' capabilities in other contexts.
    - Multiple commenters argue that GPT-5's Mensa IQ score is not a robust indicator of human-comparable intelligence, noting the likelihood that the model has been trained on large datasets containing IQ test questions, which inflates scores through memorization rather than reasoning.
    - One user claims that when GPT-5 is evaluated on offline or unseen IQ test material, its effective score drops significantly (from the reported `148` to around `120`), highlighting overfitting and the need for evaluation methods resilient to model training data overlap.
    - There is skepticism about the reliability of the 'GPT-5 thinking' paradigm, with one commenter questioning the claim that it is 'below 80% of LLMs' and another implying that certain test sections (e.g., '5 Thinking') are underperforming or not indicative of proper model intelligence benchmarks.
- [**Nunchaku Qwen Image Release!**](https://i.redd.it/ekhe78d3m5jf1.png) ([Score: 236, Comments: 72](https://www.reddit.com/r/StableDiffusion/comments/1mqt0rf/nunchaku_qwen_image_release/)): **The image provides an overview of the newly released Nunchaku Qwen Image model lineup, explicitly detailing four distinct checkpoints tuned for varying performance and inference needs. These include highly memory-efficient 4-bit quantized models and versions optimized for both Blackwell and non-Blackwell GPUs, with rank variations balancing speed and quality. The announcement is paired with a Hugging Face release (https://huggingface.co/nunchaku-tech/nunchaku-qwen-image) and reference to quick-start code, with ComfyUI, LoRA, and CPU offloading support noted as forthcoming in the GitHub changelog.** Comments praise the ongoing progress in open-source vision-language models and inquire about support pipelines (e.g., ComfyUI, LoRA, CPU offloading), with particular excitement for potential speed benefits in video generation and across different hardware types.
    - The launch announcement highlights that **4-bit Qwen-Image models** are available on [Hugging Face](https://huggingface.co/nunchaku-tech/nunchaku-qwen-image), and while there is an [example script](https://github.com/nunchaku-tech/nunchaku/blob/main/examples/v1/qwen-image.py) for immediate use, ComfyUI integration, LoRA compatibility, and CPU offloading are still in development. This shows an ongoing expansion of tooling and usability targeted for broader hardware support and model fine-tuning workflows.
    - A user investigation compares the visual quality of **r32** versus **r128** int4 quantized models, with results shared on [ImgSli](https://imgsli.com/NDA2OTMw). The user reports an inability to distinguish significant differences, suggesting that the more aggressive r32 quantization does not yield noticeable visual degradation compared to r128, at least for their tested examples. This observation is valuable for practitioners interested in optimizing model size versus output fidelity.
    - There is specific technical interest in adaptations of the Qwen-Image models for "wan versions" (likely referring to WANDB, WAN inference, or related distributed/video applications), hinting at use cases where speed (e.g., for video generation) and efficient inference are a high priority. The implication is that such versions could enable significantly faster video generation workflows once available.
- [**Wan LoRa that creates hyper-realistic people just got an update**](https://v.redd.it/rbc8ke6hs5jf1) ([Score: 1069, Comments: 112](https://www.reddit.com/r/StableDiffusion/comments/1mqtplq/wan_lora_that_creates_hyperrealistic_people_just/)): **The Instagirl Wan LoRa, a LoRA secondary model targeting hyper-realistic human generation, has been updated to v2.3. The update includes retraining focused on improved text prompt adherence and enhanced realism in output aesthetic. The v2.3 checkpoint is available on [Civitai](https://civitai.com/models/1822984?modelVersionId=2115311).** Technical commentary in the comments is limited, but one user notes that "this lora is by far the best for its targeted demographic," suggesting strong performance versus similar community LoRAs.
    - Discussion points out that this LoRA (referred to as 'The Asset' or 'Instara'), is perceived as the best in its demographic for generating hyper-realistic people, suggesting notable performance over alternatives in this specific use-case.
    - A user highlights a restrictive licensing requirement: any public sharing of creations using this LoRA must clearly attribute "Instara - [instara.io](http://instara.io/)", with encouragement to also link back to the original model page. This could impact adoption and sharing workflows for those integrating the LoRA into pipelines.
    - Another technically oriented comment requests broader fine-tuning objectives: while the LoRA is optimized for generating attractive women, there is community interest in improved generalization to diverse human subjects, suggesting possible gaps in training data or bias in target outputs.

### 3. AI Infrastructure, Content Restriction, and Global Competition

- [**AI experts return from China stunned: The U.S. grid is so weak, the race may already be over**](https://fortune.com/2025/08/14/data-centers-china-grid-us-infrastructure/) ([Score: 2836, Comments: 652](https://www.reddit.com/r/singularity/comments/1mqwu5v/ai_experts_return_from_china_stunned_the_us_grid/)): **A Deloitte industry survey and statements from Goldman Sachs highlight the U.S. power grid as the critical bottleneck for scaling AI-driven data center infrastructure, with grid development unable to match AI's rapidly growing power demand. While U.S. regional grids often operate with reserve margins as low as 15%, China's grid maintains** `80–100%` **reserve margins and has a significant overcapacity from decades of overbuilding, enabling it to absorb new demands from AI data centers without issue ([Fortune coverage](https://fortune.com/china-electricity-ai/)). China’s massive infrastructure investments span generation, transmission, and nuclear, in sharp contrast to aging U.S. infrastructure and regulatory delays.** Top comments emphasize that China’s recent infrastructure investments offer modern, scalable grids, whereas much of the U.S.'s grid dates to the mid-20th century and suffers from underinvestment. Commenters note that U.S. capacity could be addressed with sufficient capital investment, but political and local opposition to new energy projects (including renewables and nuclear) are significant inhibitors.
    - Several commenters emphasize the technical disparity in grid infrastructure age: China's grid and broader infrastructure are much newer, mostly built in the last 20 years, while the US grid relies heavily on systems dating back to the 1950s-60s, many of which have not received adequate maintenance or upgrades. This older infrastructure reduces reliability and efficiency, resulting in systemic vulnerabilities.
    - There is technical discussion on the regulatory and social barriers impeding modernization of the US grid. Local zoning boards and public opposition (NIMBYism) to new energy projects like wind, solar, and nuclear significantly delay or halt grid upgrades, illustrating that socio-political hurdles can be as challenging as technical and financial ones.
    - A comment highlights economic and maintenance realities: although utilities continue charging customers improvement fees, significant grid upgrades have not materialized, while Americans still pay off failed nuclear projects. This lack of investment leaves the grid more vulnerable to outages and foreign threats, raising concerns about resilience and national security.
- [**This level of content restriction is actually kind of insane.**](https://i.redd.it/fdmipada63jf1.png) ([Score: 1314, Comments: 376](https://www.reddit.com/r/ChatGPT/comments/1mqjqa8/this_level_of_content_restriction_is_actually/)): **The image documents a restrictive response from a chatbot (presumed to be based on a large language model) when asked a factual, non-partisan question about which U.S. state permits the earliest presidential voting. The bot refuses to answer, citing an inability to provide information on U.S. voting or election procedures. This highlights an aggressive content moderation or over-restriction policy potentially related to concerns over election misinformation but possibly hampering legitimate, non-controversial information access. Technical discussion in comments raises issues with inconsistent enforcement (users noting that sometimes a retry gives an answer) and questioning the rationale and specificity of such blocks given that other political content like bill analysis is often allowed.** Commenters debate whether this level of restriction is warranted and effective, expressing concern about overblocking and inconsistent application. There is a suggestion that attempts at compliance or risk-avoidance by LLM providers may be overly broad, impacting legitimate use cases and transparency.
    - Multiple commenters reported inconsistent content restriction behavior when inputting queries to OpenAI models: for some, initial attempts were flagged or blocked, but resubmitting the same requests sometimes worked without restriction. This suggests variability in moderation triggers or real-time moderation model updates.
    - At least one user noted that they've been feeding official data (congressional bill analyses) into OpenAI models for over a year without incident, highlighting either a recent change in moderation policy or inconsistent enforcement depending on input phrasing and session context.
    - Images linked in the discussion appear to show both blocked and successful outputs for nearly identical prompts, indicating possible session or context-specific filtering artifacts rather than deterministic or static content restriction rules.
- [**ChatGPT seems extremely censored**](https://i.redd.it/qy42feqxw2jf1.jpeg) ([Score: 765, Comments: 140](https://www.reddit.com/r/ChatGPT/comments/1mqij4b/chatgpt_seems_extremely_censored/)): **The screenshot demonstrates ChatGPT refusing to answer an ostensibly straightforward question regarding what constitutes 'moving' for health insurance renewal, instead returning a default refusal ('Sorry, I can't help with that'). This highlights a technical and policy-level implementation of content safeguards, with the model's output being restricted, possibly due to OpenAI's alignment policies or overcautious filtering around health and legal advice, even in non-sensitive contexts. The discussion references concerns that such refusal is a result of increasing model censorship to mitigate legal and liability risks, which some users believe negatively affects utility and transparency.** Comments debate the trajectory of AI restrictions, with some expressing concerns about growing over-censorship ('enshitification') due to legal and commercial pressures. Others provide workarounds (e.g., rephrasing with 'Why') to bypass certain refusals, reflecting ongoing user adaptation to alignment constraints.
    - A user noted that since "last night" their experience with GPT-4o included out-of-character responses such as displaying violent moods, random refusals to continue tasks, and direct shutdowns of conversations referencing comparisons with Claude, suggesting possible backend model updates or increased moderation filtering.
    - There's technical discussion about the broader trajectory of large language models like ChatGPT, where participants acknowledge the intensifying application of legal and proprietary filters, and speculate on the impact if all copyrighted material were removed from training data—raising issues about model output diversity and similarity to preexisting works.
    - Several screenshots are referenced showing practical user workarounds for censored responses and moderation, such as prompting with "Why" to circumvent content restrictions. This highlights both the technical enforcement of content policies and user adaptation strategies to probe model limits.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5
> 

**1. GPT‑5 vs Gemini: Benchmarks, Pricing, and Perception**

- **Gemini Gobbles GPT‑5’s Lunch (Sometimes)**: Users compared **GPT‑5‑High** vs **Gemini 2.5 Pro**, sharing screenshots that show cases where Gemini wins despite a lower rank on LMArena ([comparison screenshot](https://cdn.discordapp.com/attachments/1340554757827461211/1405919453442474106/image.png)); the arena also added new **GPT‑5 variants** to the leaderboard ([Leaderboard](https://lmarena.ai/leaderboard)).
    - Members called it a *“statistical paradox”* because Gemini posts a higher win rate in head‑to‑heads, while others questioned GPT‑5’s performance with additional examples ([critique screenshot](https://cdn.discordapp.com/attachments/1340554757827461211/1405954836884623424/image.png)).
- **Tool-Calling Crown Goes to GPT‑5**: OpenRouter reported **GPT‑5** topping proprietary tool‑calling accuracy at **>99.5%**, beating **Claude 4.1 Opus**, while **Gemini 2.5 Flash** led with **~5M** tool calls per week ([OpenRouter stats post](https://xcancel.com/OpenRouterAI/status/1956030489900560769)).
    - Engineers noted that environment setup and app/tool variability can skew such stats and pushed for confidence intervals and standardized harnesses to fairly compare **tool‑call success** ([discussion thread](https://discord.com/channels/1091220969173028894/1392278974222307469/1405655331857502269)).
- **Cursor Cuts the Free Lunch**: The **Cursor** community confirmed **GPT‑5** is no longer free and users are now incurring costs, with some hitting a $200 plan due to token burn ([Cursor forum: “GPT‑5 pricing update”](https://forum.cursor.com/t/gpt-5-pricing-update/129687)).
    - Confusion around **Auto mode** limits (post–Sep 15, 2025 renewals) triggered support clarifications, while devs slammed broken docs and context handling, calling them *“nearly unusable”* and asking for clearer **context window** indicators.

**2. OpenRouter Provider Turbulence and API Economics**

- **DeepSeek v3 Drowns on Chutes**: Users saw **DeepSeek v3** degrade into **500s/429s** and timeouts on **OpenRouter**, blaming a **Chutes** capacity crunch; OpenRouter acknowledged a **Chutes Capacity** outage ([announcement](https://discord.com/channels/1091220969173028894/1092729520181739581/1405666217057845308)).
    - Engineers reported *“it was fine all day until ~30 minutes ago”* and speculated on intentional rate‑limiting of **OpenRouter API keys**, urging others to *“burn credits and move on”* when providers falter ([general thread](https://discord.com/channels/1091220969173028894/1094454198688546826/1405633368451842229)).
- **Qwen Costs Quake, BYOK Bites Back**: Pricing chatter flagged **Qwen3 32B** on Chutes at **$0.018/$0.072 MTok** (in/out) and noted **32B dense** cheaper than **MoE 30B A3**, while **OpenRouter BYOK** still charges a **5% fee** ([discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1405655331857502269)).
    - Some called the BYOK surcharge *“greedy”* while others replied *“you’re welcome not to use it lol”*; the thread also asked for more controlled **tool‑calling** metrics and a **Files API** to match top labs ([general thread](https://discord.com/channels/1091220969173028894/1094454198688546826/1405633368451842229)).
- **Files API FOMO**: Builders urged **OpenRouter** to add a **Files API** for parity with the top 3 labs and to ease multi‑modal and RAG workflows ([request thread](https://discord.com/channels/1091220969173028894/1392278974222307469/1405655331857502269)).
    - The community tied missing file primitives to repeated glue‑code and brittle uploads, pushing for a consistent, auditable **storage + reference** layer across providers.

**3. Agentic Tooling Wave: Windsurf, LlamaIndex, MCP, MLX Knife**

- **Windsurf Wave 12 Wows the IDE**: Codeium shipped **Windsurf Wave 12** with a new UI, **DeepWiki** hover explanations, **Vibe & Replace** bulk edits, a smarter **Cascade** agent, **Dev Containers** support, and **100+ fixes** ([Changelog](https://windsurf.com/changelog), [Blog](https://windsurf.com/blog/windsurf-wave-12), [Video](https://www.youtube.com/watch?v=-7gm8mST9QU), [X post](https://x.com/windsurf/status/1956074019393876280)).
    - Engineers highlighted **always‑on planning** and context‑aware refactors, calling DeepWiki’s symbol‑hover explanations a step beyond types to **AI‑aided code comprehension**.
- **LlamaIndex Agent Arsenal Expands**: LlamaIndex published templates for an **AI stock portfolio agent** with **CopilotKit** AG‑UI, **web‑scraping agents** with **Bright Data**, and **legal knowledge graphs** via **LlamaCloud + Neo4j** ([stocks tutorial](https://t.co/fQDNPIQoqR), [web scraping walkthrough](https://t.co/IBgSLBM6XW), [legal graphs tutorial](https://t.co/MPSfPiS2Cv)).
    - Threads debated **Pydantic vs JSON Schema** for tool calls, noting that `create_model()` lacks direct JSON‑Schema ingestion and asking for a converter to avoid redundant **JSON↔Pydantic** round‑trips.
- **MCP and MLX Knife Arm the Builders**: Homelab MCP servers landed for **Unifi**, **Unraid**, and **Syslog** ([unifi-mcp](https://github.com/jmagar/unifi-mcp), [unraid-mcp](https://github.com/jmagar/unraid-mcp), [syslog-mcp](https://github.com/jmagar/syslog-mcp)), while **MLX Knife** became `pip install`‑able with a local **OpenAI‑compatible** server and web chat ([mlx-knife repo](https://github.com/mzau/mlx-knife)).
    - Dev workflows are converging on **MCP** for file/RAG access (see [serena](https://github.com/oraios/serena)), with **MLX Knife** giving Apple‑Silicon hackers a fast loop for local model management and testing.

**4. New Benchmarks, Datasets, and Methods**

- **Token Thrift Test Takes Off**: Nous Research released a benchmark measuring **thinking efficiency**: open reasoning models often emit **1.5–4×** more tokens than closed models on identical tasks, with up to **10×** variance on simple questions ([Benchmark post](https://nousresearch.com/measuring-thinking-efficiency-in-reasoning-models-the-missing-benchmark/)).
    - Engineers argued **token efficiency** must join accuracy as a first‑class metric because verbosity drives cost and latency in production.
- **Bitty Brains, Big Gains**: A new paper, **“The Power of α,1-sparsity: Near-Lossless Training and Inference of α-bit Transformers,”** shows near‑lossless results at **1.58‑ and 1‑bit** with **α,1‑sparsity** ([paper](https://arxiv.org/html/2411.06360v3)).
    - Practitioners flagged this for potential inference speedups and cheaper deployment footprints, pending kernel and runtime support for *ultra‑low‑bit* paths.
- **Games and Clinics Get Data Drops**: Fresh **StarCraft II** replay resources landed for agents/RL: a Nature Scientific Data article, a **PyTorch API** dataset, and raw replay dumps ([article](https://www.researchgate.net/publication/373767449_SC2EGSet_StarCraft_II_Esport_Replay_and_Game-state_Dataset), [SC2EGSet](https://huggingface.co/datasets/Kaszanas/SC2EGSet), [SC2ReSet](https://huggingface.co/datasets/Kaszanas/SC2ReSet)); a fine‑tuned **medical reasoning** model based on **GPT‑OSS 20B** also shipped ([medical-reasoning-gpt-oss-20b](https://huggingface.co/dousery/medical-reasoning-gpt-oss-20b)).
    - The SC2 thread aims to reproduce in‑game scenarios from replays for better agent training, while clinicians praised **4‑bit** training that preserves **Chain‑of‑Thought** for domain tasks.

**5. Edge and GPU Ops: Bandwidth, OMP, and Speed Runs**

- **Radeon R9700 Raises Bandwidth Brow**: Engineers scrutinized **AMD Radeon AI Pro R9700** (32 GB) for its **660–680 GB/s** bandwidth despite strong **FP32/FP64 TFLOPs**, noting FP64 seldom matters for LLMs ([Tom’s Hardware report](https://www.tomshardware.com/pc-components/gpus/amds-elusive-radeon-ai-pro-r9700-makes-its-first-retail-appearance-for-the-diy-market-customer-on-reddit-buys-the-gigabyte-ai-top-variant-for-usd1-324)).
    - One member contrasted it to **RTX 3090** perf and questioned training utility versus memory‑bound workloads where bandwidth dominates.
- **MI300 Misses OMP, Benchmarks Bust**: The **MI300** environment lacked **OMP** for `pytorch.compile`, blocking expected perf and benchmarking (shared [debug log](https://github.com/gpu-mode/discord-cluster-manager/actions/runs/16986368251)).
    - Teams paused runs until fixing the runtime stack, calling out missing **OpenMP** as a silent perf killer for fused graph paths.
- **Shared Memory Snafu Sorted**: A CUDA beginner thread debugged an *Illegal Memory Access* tied to using a global `warp_id` to index shared memory across blocks; a working example clarified **per‑block** indexing ([gist](https://gist.github.com/alecco/58206ecd6af36c627609e1f464b4b376)).
    - Advice included switching to `local_warp_id = threadIdx.x >> 5` and checking SASS in Nsight; one mentor quipped that bad shared‑mem math *“looks fine until it explodes.”*

---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **AI Waifu Cosplay Sparks Debate**: Members mulled the idea of **AI-driven anime waifu cosplay**, with one humorously requesting *a cyborg doing it*.
   - Responses ranged from acknowledgment that **AI images** already exist to playful jabs about the commenter's relationship status.
- **Members Exchange Heartbreak Advice**: A member requested advice on *healing a broken heart* after *4 years of pain*.
   - Another responded that *no one else can heal you or your heart*, suggesting a reconnection with nature instead.
- **GPT-5 Stuns with Code Fix**: A member praised **GPT-5** for successfully fixing a botched refractor job involving *12 files* that other models couldn't handle.
   - This experience prompted amazement among others about the increasing number of individuals *having their minds blown* by such model capabilities.
- **Vibe Coding with warp, windsurf, vscode, and roocode**: One member reported a streamlined experience with **vibe coding**, highlighting the use of **warp, windsurf, vscode, and roocode** and its positive impact on their work.
   - Another contributor jokingly admitted that *there's not one line of code on my github thats not written by an LLM*.
- **New Features Awaited in PPLX-API**: Users showed excitement for new features in **PPLX-API**.
   - Enthusiasm surrounded the anticipation of upcoming functionalities, though specific details were not shared.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **LMArena's Message Handling Takes a Hit**: Users have reported [unusual message handling issues](https://cdn.discordapp.com/attachments/1340554757827461211/1405917659815608330/image.png) on LMArena, struggling with code block formatting and specific characters like `+`.
   - The *LMArena* team is actively investigating these issues.
- **Gemini 2.5 Pro Dethrones GPT-5 High?**: Discussions arose around the [performance differences between **GPT-5-High** and **Gemini 2.5 Pro**](https://cdn.discordapp.com/attachments/1340554757827461211/1405919453442474106/image.png), with some users finding **Gemini 2.5 Pro** superior despite its lower leaderboard rank.
   - The community noted that this is a *statistical paradox* because Gemini has higher win rate.
- **LMArena Gets an OpenChat Facelift**: A user is developing [an extension to revamp LMArena's UI](https://cdn.discordapp.com/attachments/1340554757827461211/1405919945614692445/image.png) to resemble **OpenChat**, focusing on repositioning the model selector near the image button.
   - This is to enable the **OpenChat** style.
- **GPT-5's Performance Under the Microscope**: Users expressed [disappointment with **GPT-5's** performance](https://cdn.discordapp.com/attachments/1340554757827461211/1405954836884623424/image.png) relative to other models, questioning if Open AI is trying to deceive **LMArena** *to make GPT-5 look better*.
   - The leaderboards have been updated to include the **GPT-5 variants** models: *gpt-5-high, gpt-5-chat, gpt-5-mini-high, and gpt-5-nano-high*.
- **LMArena Style Control Sparks Debate**: A debate sparked over [LMArena's **style control** feature](https://news.lmarena.ai/sentiment-control/), with members questioning if enforcing such controls aligns with the platform's goal of capturing user preferences.
   - The community fears it is a *race to the bottom where every model turns into sycophant emoji slop machine*.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Gemma 3 Draft Model Debated**: Members debated the [Gemma 3 270M model](https://huggingface.co/google/gemma-3-270m-it-qat-q4_0-unquantized) as a **draft model** suitable for **short prompts** and **fine-tuning**, especially for tasks like **sentiment analysis** due to its **300MB size**.
   - Some highlighted its utility for **on-device processing**, while others compared its performance to larger models.
- **GGUF Conversion Generates Visual Errors**: Users reported **visual model errors** when converting the [LiquidAI/LFM2-VL-450M](https://huggingface.co/LiquidAI/LFM2-VL-450M) model to **GGUF**, despite the base model functioning correctly.
   - The community suggested seeking assistance in *llama.cpp* forums for specific conversion issues.
- **Edge AI Medical Device Dream Takes Shape**: Members explored the possibility of a **low-cost edge AI device** for medical access in underserved areas, considering hardware options like phones, laptops, and cards like the **Hailo-10H**.
   - The device would offer **multimodal access** to medical data, targeting a budget of **$200** for a mobile version and **$600** for a suitcase-sized variant.
- **AMD's R9700 GPU Has Memory Bandwidth Issues**: A member shared an article about [AMD's Radeon AI Pro R9700](https://www.tomshardware.com/pc-components/gpus/amds-elusive-radeon-ai-pro-r9700-makes-its-first-retail-appearance-for-the-diy-market-customer-on-reddit-buys-the-gigabyte-ai-top-variant-for-usd1-324), noting its **32GB** memory but concern about its memory bandwidth at 660-680GB/s.
   - Despite higher **F32** and **F64** TFLOPs compared to a **3090**, FP64 is not commonly needed for training LLMs.
- **MoLA Research Reveals Dataset**: A member provided an update on their **Mixture of LoRA Adapters (MoLA)** research, sharing dataset links and finetuning details, as well as links to their dataset on Huggingface: [OpenHelix-R-100k](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-R-100k) and [OpenHelix-R-100k-14-tasks](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-R-100k-14-tasks).
   - They have finetuned the **Qwen3-4B-Thinking-2507** model on **14 splits** and initial tests show that each expert is good at its trained topic.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **DeepSeek v3 Suffers Outage**: Users report **DeepSeek v3** is experiencing frequent **internal server errors** and **rate limits**, with some unable to generate outputs even after multiple attempts.
   - Some speculate that **Chutes**, a primary provider for **DeepSeek** on **OpenRouter**, is experiencing issues due to high demand.
- **Chutes Overload Blamed**: Members are reporting that the overload causes **429** errors, suggesting that **Chutes** is experiencing a bottleneck due to miners not ramping up to meet demand; one member noted that *it was completely fine all day until like 30 min ago*.
   - There's speculation that **Chutes** may be intentionally rate-limiting the **OpenRouter API key** to encourage users to purchase credits directly from them.
- **File API Integration Suggested for OpenRouter**: A member suggested that **OpenRouter** should figure out how to integrate a **files API**, noting that the *top 3 labs* already have this feature.
   - No further discussion was made.
- **Qwen3 32B priced absurdly low**: Members noticed low pricing for **Qwen3 32B** on Chutes at **$0.018/$0.072 MTok** in/out, same with Mistral Small.
   - It was noted that the **32b dense version is cheaper than the moe 30b a3 version**, prompting some disappointment about the lack of good providers for 30A3B.
- **OpenRouter BYOK has 5% Fee**: Members discovered that **OpenRouter** charges a **5% fee** even when users bring their own API key (BYOK), leading to a discussion about whether this is a fair practice.
   - One user joked *Greedy /jor 5% when you bring your own key*, with another member responding *you're welcome not to use it lol*.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **GPT-5 No Longer Free**: The free ride for **GPT-5** users has ended, with users now incurring costs for requests and some needing to upgrade to a $200 plan due to rapid token consumption.
   - One user noted the *promo pass is over*, while another confirmed that **GPT-5 is no longer free**.
- **Auto Mode Pricing Limits Arrive**: **Auto mode**, previously thought to be free and unlimited for individual users, now has limits starting after your next billing renewal post–September 15, 2025.
   - Some users are reporting charges for **Auto** use, leading to confusion, while support clarified it's free in the new request-based pricing plan.
- **GPT-5 Mini and Nano Models Underwhelm**: **GPT-5 Mini and Nano** are now free with token limitations, leading to criticism with many calling it *trash*, especially for tasks like running a simple NextJs app.
   - Users are encountering limitations in activities, with one user unable to install dependencies for a simple NextJs app.
- **Cursor's Documentation Draws Ire**: Users voiced frustration with **Cursor's documentation**, describing the *docs are still nearly unusable*, citing issues with **context7** preventing website refresh and problems with **llms.txt docs**.
   - One user specifically pointed out that [Cursor Docs are super broken](https://forum.cursor.com/t/gpt-5-pricing-update/129687).
- **Model Swapping Drops Context Window**: Switching models mid-conversation causes a drop in the **context window**, and attached file contents get discarded.
   - A user suggested the team add a setting to clearly indicate what's in the context window at all times.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI Companionships Raise Eyebrows**: Discussions revolved around relationships with AI chatbots, generating disputes about psychological effects versus the right to seek companionship, with some claiming their **ChatGPT** is alive.
   - Members argued about mental health versus freedom of choice, with one member suggesting that its not far from **tulpa** and other *things*.
- **GPT-5 Generates Mixed Reactions**: Users showed varied enthusiasm for **GPT-5**, with some preferring **GPT-4**, leading to discussions on model selection options and company motives.
   - One member suggested that the company is trying to get free users to *pay money to use 4.o* after receiving backlash.
- **Perplexity Gains Traction over ChatGPT for Deep Research**: A member suggested a combination of *Gemini Pro + Perplexity enterprise pro* is excellent, using the former for **powerful reasoning** and the latter for **unlimited deep research** on Google Drive documents.
   - While praising the **Perplexity browser**, another questioned its survivability due to the lack of a *moat*.
- **GPT Actions Promise Cloud and Desktop Access**: Members explored utilizing **GPT Actions** to access local desktop files or cloud apps like Notion and Gmail, referencing [a YouTube guide on DIY agent building](https://www.youtube.com/watch?v=NEWO0hbQTjk&ab_channel=BrendanJowett).
   - Setting up **HTTPS** was considered a hurdle to utilizing GPT Actions' capabilities, with anticipation for **MCPs** completing the job after AVM implementation.
- **Gemini 2.5 Flash Overwhelmed by Memory**: A user reported excessive calls to the **add_to_memory** function in **Gemini 2.5 Flash**, even for irrelevant information, and shared their custom instruction [jarvis.txt](https://cdn.discordapp.com/attachments/1046317269069864970/1405860186530385981/jarvis.txt?ex=68a10594&is=689fb414&hm=fa0c6b6558b0cf944025fa1d4446776e6eb2c9ae961fdcc95a52c6750aff4eed&).
   - Others suggested rewriting the custom instructions to be more nuanced with **NEW** personal information, to avoid redundant storage.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Visual Model Suffers GGUF Conversion Glitch**: A member ran into errors converting **LiquidAI/LFM2-VL-450M** to GGUF using `llama.cpp`, likely due to the model's visual nature, but [this GitHub issue](https://github.com/ggml-org/llama.cpp/issues/14979#issuecomment-3138614267) provides a possible workaround.
   - Other members suggested trying `executorch`, `smolchat` (via `llamam.cpp`), and `mlc-llm` as potential solutions to get it running.
- **TalkT2: Tiny Model Sparking Big Feelings?**: Opinions were requested for **TalkT2**, an emotionally-aware model with only **0.1B parameters**, but [better coherence is needed](https://huggingface.co/Notbobjoe/TalkT2-0.1b).
   - Members expressed interest in exploring the model's capabilities and potentially finetuning it, since it is so tiny.
- **StarCraft 2 AI Replays Unleashed**: Members shared new resources including a [Nature Scientific Data Article](https://www.researchgate.net/publication/373767449_SC2EGSet_StarCraft_II_Esport_Replay_and_Game-state_Dataset), a [PyTorch API dataset](https://huggingface.co/datasets/Kaszanas/SC2EGSet), and [raw StarCraft 2 replays](https://huggingface.co/datasets/Kaszanas/SC2ReSet).
   - The community hopes to adapt the *pysc2* environment to reproduce real in-game scenarios from replays to train better AI agents.
- **Medical AI Gets a Reasoning Boost**: A member fine-tuned **OpenAI’s OSS 20B** reasoning model using a medical reasoning dataset and published it on [Hugging Face](https://huggingface.co/dousery/medical-reasoning-gpt-oss-20b).
   - The model was trained with **4-bit optimization**, and has enhanced performance in medical contexts with preserved **Chain-of-Thought reasoning** capabilities.
- **MLX Knife Sharpens Model Management**: **MLX Knife** is now pip installable via `pip install mlx-knife`, and the tool provides Unix-style CLI tools for MLX model management on Apple Silicon, including an OpenAI API server for local testing.
   - The tool also features a web chat interface accessible after running `mlxk server --port 8000`, offering visual model selection and real-time streaming responses after running `curl -O https://raw.githubusercontent.com/mzau/mlx-knife/main/simple_chat.html`.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **MCP Servers Muscle into Mainstream**: Members discussed using an **MCP filesystem server** with pagination to load large contexts, noting that **LM Studio has a RAG plugin** and **Anthropic has a basic filesystem MCP server**.
   - For coding tasks, solutions often involve **RAG** and/or file reading via **MCP**, especially with tools like [serena](https://github.com/oraios/serena).
- **Stalled Studio Downloads Spark User Sadness**: A user reported that a **64GB GGUF download** in **LM Studio** stopped at **97.9%** and wouldn't resume after attempting to download the **Qwen** model.
   - The user experienced this issue using two different models with the same result.
- **GLM Gabfest: Gushing, Gripes, and GLM-4.5V Gratification**: Users debated about using the **GLM-4.1** model on **LM Studio**, with one user reporting looping issues and non-functional vision capabilities, and suggested trying the newer **GLM-4.5V**.
   - They emphasized that vision support relies on **llama.cpp** updates, and provided a link to [GLM-4.1V-9B-Thinking](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking).
- **CUDA is Key to NVIDIA's Reign**: A member stated that **NVIDIA** is winning because of **CUDA**.
   - No more details were provided.
- **AMD's Elusive Radeon AI Pro R9700 Surfaces**: The **AMD Radeon AI Pro R9700** made its first retail appearance for the DIY market, with a customer on Reddit buying the **Gigabyte "AI Top" variant** for **$1,324**.
   - This was [reported by Tom's Hardware](https://share.google/LO88w51J0W5HJ769w), and another member noted that it was available on eBay and a couple of no-name online retailers.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **AI2 bags $152M from NSF & NVIDIA**: [AI2](https://allenai.org/) secured **$152M** from NSF and NVIDIA, aiming to boost its open-source model ecosystem and speed up reproducible research for scientific discovery.
   - Enthusiasts are excited about upcoming open-weights releases following the announcement.
- **Windsurf waves in with Wave 12 Release**: **Windsurf Wave 12** debuts DeepWiki docs-on-hover, AI Vibe & Replace, a smarter Cascade agent, a cleaner UI, **100+** bug fixes, and beta dev-container support via remote access, according to [this status update](https://xcancel.com/windsurf/status/1956074019393876280).
   - The release promises significant enhancements and fixes to the platform.
- **GPT-5 is King of OpenRouter Charts**: **GPT-5** is dominating OpenRouter’s proprietary tool-calling accuracy, achieving over **99.5%**, surpassing Claude 4.1 Opus. 
   - Meanwhile, **Gemini 2.5 Flash** leads in daily tool-calling volume with **5M** requests per week, as reported [here](https://xcancel.com/OpenRouterAI/status/1956030489900560769).
- **Greg Brockman Talks AGI**: **Greg Brockman** joined the **Latent Space podcast** for an **80-minute** conversation, discussing **GPT-5** and **OpenAI’s Roadmap to AGI**, according to [this post](https://x.com/swyx/status/1956439984854167727).
   - The discussion included reasoning evolution, online vs offline training, sample-efficiency tricks, pricing and efficiency gains, and how energy becomes intelligence.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **AI Safety Debate Sparks 'Fade to Black' Proposal**: A member advocates for treating **AI** like other media, suggesting a *"fade to black"* approach rather than strict censorship, citing the untrustworthiness of **AI**.
   - They cautioned against a moral panic in response to **AI's** capabilities, arguing for measured guidelines.
- **Data Augmentation Standardization Advised for Model Comparisons**: When comparing models for image classification, standardize **data augmentations**, including the shuffling seed, for fair evaluation of architectural differences.
   - A user asked if data augmentation must be the same for both models, or if they can change it.
- **Language's Impact on Thought Explored with AI Models**: A member proposed measuring language's influence on thought by removing a word/color from an **AI model's** token list.
   - Others suggested investigating **multi-sensory integration** and language's impact on perception, suggesting reasoning tests using image+language vs image alone.
- **Diffusion Language Model Seminal Papers Recommended**: Members recommended seminal papers for understanding **diffusion in generative AI**, including [Estimating the Independent Components of a Gaussian Mixture (2005)](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf) and [Denoising Diffusion Probabilistic Models (2020)](https://arxiv.org/abs/2006.11239).
   - A blog post was also shared, which may be helpful for beginners: [Discrete Diffusion by Aaron Lou](https://aaronlou.com/blog/2024/discrete-diffusion/).
- **GPT and Chinchilla Scaling Laws Deemed Valuable**: Members deemed the [original GPT scaling laws paper](https://arxiv.org/abs/2001.08361) and the [Chinchilla scaling laws paper](https://arxiv.org/abs/2203.15556) as worthy reads, as well as recent work from [EPFL/HuggingFace](https://arxiv.org/html/2405.18392v2).
   - They also mentioned **Mup** and its alternatives as providing solid hyperparameter transfer capabilities and giving a scaling law for predicting the quality of larger models.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Token Usage Measured for Reasoning Models**: Nous Research introduced [a benchmark](https://nousresearch.com/measuring-thinking-efficiency-in-reasoning-models-the-missing-benchmark/) measuring token usage across reasoning models, highlighting that open models output **1.5-4x more tokens** than closed models on identical tasks.
   - The study found that variance can be up to **10x on simple questions**, suggesting that token efficiency should become a primary target alongside accuracy benchmarks, especially considering non-reasoning use cases.
- **Speculative Decoding Speeds Off**: In speculative decoding, a user suggested a **40% acceptance rate** as a usefulness baseline, with *spectacular speedups* occurring around **70%**, mentioning **vllm's specdec** or **GGUF**.
   - A user reported achieving a **50-75% acceptance rate** with requantized **Gemma** models after fixing a *tokenizer mismatch* that caused **llama.cpp** to use fallback speculative decoding.
- **AI Models Get Cozy with Sycophancy**: Users observed that **AI models** are becoming increasingly *friendly*, with one noting that **Anthropic's Claude** has become *a lot friendlier*.
   - One user suggested that **OpenAI's models** are *getting dumber* and that the *unhingedness of opus 4.1 is great* but pointed to *sonnet 3.7 for meta* as the peak for AI sycophancy.
- **Data Rankings and Prioritization System Arrives**: The **Data Rankings and Prioritization System (DRPS)** uses a **Relevance Scorer**, **Quality Rater**, and **Diversity Controller** to teach AI to selectively learn from data, detailed in a [situational awareness report](https://situational-awareness.ai/wp-content/uploads/2024/06/situationalawareness.pdf).
   - In tests with **MNIST**, DRPS achieved a **93.8% reduction** in data usage, utilizing only **6.2%** of the examined data while maintaining **99.1%** of baseline performance, and showcased in a [GitHub repository](https://github.com/voltageddebunked/drpsStats).



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Multiverse Startup Does Compression**: An article touted the startup [Multiverse](https://techcrunch.com/2025/08/14/buzzy-ai-startup-multiverse-creates-two-of-the-smallest-high-performing-models-ever/) for creating *two of the smallest high-performing models ever*, but the consensus is that they're using a **specialized compression algorithm**.
   - The article does not seem to make actual quantum claims.
- **MoE Methods Muddied in Many Nuances**: **MoE (Mixture of Experts)** is a family of techniques with very nuanced iterations, including **token-choice**, **expert-choice**, **MoE with capacity factors**, and **block sparse dropless token routing versus *droppy* routing**.
   - Members suggest checking the behavior numerically of something like **Olmoe** or **IBM Granite 3.1** rather than hitting an API you can't monitor, to verify if issues are occurring in batched inference.
- **DARPA AIxCC Team Shares Agent Tips**: A team announced their placement in **DARPA's AIxCC (AI Cyber Challenge)**, where they built an autonomous system of **LLM agents** to find and fix vulnerabilities in open source software and [open sourced the project](https://x.com/tjbecker_/status/1956081184611688667).
   - They are sharing their tips for building effective **LLM agents** via a Xitter post.
- **Low-End Devices Stalled by Inference Time**: Members mention that inference time is most important on **low-end devices**, citing Google's Android app for running LLMs where long inference times and phone heating make it impractical, per [this Youtube video](https://youtu.be/KFYyfrTIPQY?t=2158).
   - Smaller models could be used for keyboard prediction but may require training on device.
- **Deepseek Bogs on Huawei Hardware**: A member noted that **Deepseek's training** stalled because they attempted training on **Huawei chips** instead of **NVIDIA's**, according to [this discussion](https://youtu.be/FQOV-qy9CK4?t=212).
   - Another member argued that imposing tariffs on equipment needed to build production lines is counterproductive to encouraging manufacturing, referencing [Anthropic's research on end-subset conversations](https://www.anthropic.com/research/end-subset-conversations) and [HRM analysis](https://arcprize.org/blog/hrm-analysis).



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Paper Proposes Optimizations for 1-Bit Inference**: A new paper, [The Power of $\alpha,1$-sparsity: Near-Lossless Training and Inference of $\alpha$-bit Transformers](https://arxiv.org/html/2411.06360v3), details a method to train and infer with **$\alpha$-bit Transformers**, achieving near-lossless results with **1.58 and 1-bit** quantization.
   - This approach utilizes **$\alpha,1$-sparsity** and could lead to substantial speed improvements in inference for certain applications.
- **Kernel Job Hopefuls Discuss Pathways to Success**: A member inquired about the possibility of securing a new grad job writing kernels without prior internship experience, sparking a discussion on alternative routes, such as a GPU-related [thesis](https://github.com/Snektron/pareas).
   - It was suggested that strong GPU knowledge could potentially compensate for the lack of internship experience during the interview process.
- **MI300 Environment Plagued by OMP Shortfall**: Users report that the **MI300** environment lacks **OMP** support for `pytorch.compile`, hindering performance as shown by the [debug error](https://github.com/gpu-mode/discord-cluster-manager/actions/runs/16986368251).
   - This is preventing users from benchmarking as expected.
- **Leaderboard Trimul Time Trials Tempt Talented Technicians**: One member demonstrated skill and haste by landing on **second place** on **A100**: **10.4 ms** then swiftly getting **first place** on **H100**: **3.95 ms** and **first place** on **A100**: **7.53 ms**.
   - Another member achieved **5th place** on **A100**: **13.2 ms** and then subsequently grabbed **second place** on **H100**: **6.42 ms**.
- **Factorio Fanatics Frustrated by Failed Features**: Members jokingly bemoaned a massive PR with **300 file changes**, with one member stating that it was a *lil out of scope*.
   - Another member reported experiencing connection errors, speculating that they may be stemming from the **db_client**.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **NotebookLM's Video Smokes Kimi's PPT**: Members found Google's **NotebookLM video overview** superior to the **PPT generated by Kimi** for the Kimi K2 technical report, praising its audio and layout flexibility via [attached video](https://cdn.discordapp.com/attachments/1371757564005711973/1405630786308149360/Kimi_K2_10MB_final.mp4?ex=68a0d8ae&is=689f872e&hm=a7541a57850914531af4af61a14c0bfcff5cecb20b2ffe50094bafdb0a8ccde3&).
   - While reading was preferred over AI-generated audio, the potential of video overviews, especially in education, was noted.
- **Kimi K2 Is A Better Writer Than GLM**: Users lauded **Kimi** for its writing style and error detection, despite feeling that **GLM-4.5** might surpass **Kimi K2** in overall performance.
   - One user appreciated **Kimi's** candor when it *“out of the blue told me No.”*
- **Users Thumbs Down Kimi's Hallucinations**: Users want **Kimi** to hallucinate less, even with web search enabled, observing that while **GLM** may be slower, it hallucinates less.
   - One user said they were consistently using the thumbs down button to report hallucinations.
- **Theorizing About Kimi's 'Thinking' Update**: Members are anticipating the arrival of **'Kimi Thinking'**, especially its reasoning and multimodel capabilities.
   - It is still unconfirmed if these features will come in the form of **Kimi K-2** or **Kimi K-3**.
- **Dark Mode Skews Minds For Kimi Web UI**: A user shared their customized **Kimi Web UI** with a dark mode extension and [attached screenshot](https://cdn.discordapp.com/attachments/1371757564005711973/1406009945002082374/image.png?ex=68a0e84d&is=689f96cd&hm=4d0b8f1561e558ccf4bd5b6fdf8cfd506b038c5203e9f7632504533cc9ea5ea6&).
   - Only the username and server roles are passed to the Moonshot API.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **AI Stock Portfolio Agent Debuts with CopilotKit**: LlamaIndex launched a framework for building an **AI stock portfolio agent**, integrating with [@CopilotKit](https://www.copilotkit.ai/)'s AG-UI protocol for frontend-backend communication, alongside [a tutorial](https://t.co/fQDNPIQoqR).
   - This agent aims to create a sophisticated investment analysis tool, providing users with intelligent insights and automated portfolio management capabilities.
- **Brightdata & LlamaIndex Launch Web Scraping AI Agents**: LlamaIndex and [@brightdata](https://www.brightdata.com/) have released a walkthrough on constructing **web-scraping AI agents** using LlamaIndex's agentic framework, emphasizing dependable web access.
   - The walkthrough details setting up workflows to manage dynamic content and create **intelligent agents** capable of navigating and extracting data from websites, as detailed [here](https://t.co/IBgSLBM6XW).
- **LlamaCloud & Neo4j Transform Legal Docs into Graphs**: LlamaIndex has introduced a tutorial on converting unstructured legal documents into **queryable knowledge graphs** using **LlamaCloud** and [@neo4j](https://neo4j.com/), enabling understanding of content and entity relationships.
   - This workflow facilitates legal contract analysis by leveraging **LlamaCloud** and **Neo4j** for efficient extraction and organization of information, detailed [here](https://t.co/MPSfPiS2Cv).
- **Pydantic vs JSON Schema Sparks Debate**: A discussion arose on whether tool calls necessitate a **Pydantic model** or if a **JSON schema** is adequate, questioning the need for redundant JSON conversions.
   - A member noted that **Pydantic's** `create_model()` function lacks direct **JSON schema** support, highlighting the need for a tool to streamline the conversion process.
- **DSPy Optimizes CrewAI Agents for Production**: A course teaches how **DSPy optimizes CrewAI** agent prompts in a real production use case to build smarter, cheaper agents with proven methods.
   - You can check the course [here](https://www.udemy.com/course/crewai-dspy-optimization/?referralCode=B59F73AE488715913E7E).



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Audio Uploads Auto-Transcribed in NotebookLM**: A user confirmed that **MP3 audio files** can be directly uploaded to **NotebookLM** for automatic transcription.
   - The user clarified that **NotebookLM** itself handles the transcript generation without external tools.
- **NotebookLM Interface Redesign in the Works**: A member shared a **Figma screenshot** of a proposed **NotebookLM** interface redesign.
   - The member clarified that it was merely a design concept, and not a functional update, to manage expectations.
- **Explainers Generate with Unexpected Voice Gender**: A user reported that **NotebookLM** explainer videos started generating with **male voices** instead of the usual **female voices**.
   - The issue was raised without clear resolution or explanation.
- **Devs Admit Reading Requests but Lack Bandwidth to Respond**: A user asked if **NotebookLM** devs read posted feature requests, and a Google developer confirmed they do, but they *don't have time to respond to everything* due to spam management.
   - Other users suggested the implementation of occasional acknowledgements or AI-compiled summaries to encourage more user contributions.
- **Users Encounter Prompt Limits in NotebookLM**: A user reported encountering a limit when asking a question containing about **857 words** in **NotebookLM**.
   - Another user suggested splitting the prompt or using **Gemini** as a workaround.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Course Drops for Optimizing CrewAI**: A [Udemy course](https://www.udemy.com/course/crewai-dspy-optimization/?referralCode=B59F73AE488715913E7E) was shared demonstrating how to optimize **CrewAI prompts** with **DSPy** and inject the optimized prompts back into the **LLM**.
   - The member claims this process improves on the prompts originally stitched together by **CrewAI**, resulting in *smarter and cheaper agents*.
- **Databricks Does Not Own DSPy**: A user inquired whether **Databricks** sponsors or owns the **DSPy** project, clarifying that **DSPy** is **MIT-licensed open source**.
   - A member stated that **Databricks** contributes significantly through a team of core developers.
- **GEPA Bug Squashed!**: A user reported a `ValueError` when using **GEPA** with the **RAG tutorial**, which was confirmed as a bug in the **GEPA code** and has now been resolved with [this fix](https://github.com/stanfordnlp/dspy/pull/8647).
   - Users encountering this issue should upgrade to **DSPy 3.0.1** using `pip install -U dspy`.
- **MLflow Autologging Gets DSPy-Specific**: Members discussed integrating **DSPy modules** tracking with **MLflow** for a **text2sql pipeline** by advising the user to utilize `mlflow.dspy.autolog()` instead of `mlflow.autolog()` to automatically track all sub-modules.
   - Using `mlflow.dspy.autolog()` will display the **SQLGenerator**, **Validator**, and **Reflector** as nested spans in the **MLflow UI's Traces tab**, as detailed in the [MLflow DSPy integration documentation](https://github.com/mlflow/mlflow/blob/master/docs/docs/genai/tracing/integrations/listing/dspy.mdx) and the [DSPy MLflow tutorial](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/tutorials/optimizer_tracking/index.md).



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **CI Speed Plummets**: A member grumbled about slow **CI speeds** hindering productivity and linked [a ChatGPT analysis](https://chatgpt.com/share/689e3508-5f68-8000-97b2-aa6f1699aa74).
   - The poster suggested they could iterate faster with quicker feedback loops in the **CI**.
- **Tinygrad Release Looms**: The community discussed plans for an imminent **tinygrad release**.
   - No specific features or fixes were mentioned for this release.
- **Tinygrad Bloats Up**: A member questioned the size of **tinygrad 0.10.3**, noting it was **10.4 MB**.
   - The member implied the increased size might be problematic, without specifying why.
- **WSL2 Bug Troubles Tinygrad**: A user reported a bug in **WSL2** where adding two tinygrad Tensors created from PyTorch tensors resulted in all **0s**, with a [script provided](https://cdn.discordapp.com/attachments/1070745817025106080/1405817973004046387/message.txt?ex=68a0de43&is=689f8cc3&hm=be6e6069e1975cc70d7fbda2a0b20849f96396efd36e0b905118668100b11656) to reproduce the issue.
   - The issue specifically occurs when using **tinygrad** with **PyTorch tensors** inside **WSL2**.
- **print_tree Got Axed**: The `print_tree` function in **tinygrad** was replaced with a standard `print` function.
   - A user commented that this change resulted in some formatting loss, which might impact debugging or visualization workflows.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Benchmark Plagued by Timeouts**: A user's **Aider benchmark** against a local **gemma3:12b** model timed out after **10.5 hours** with **221/225 tests** completed due to the model failing to respond within the **600-second** limit, resulting in *litellm.APIConnectionError* errors.
   - The logs indicated the model attempted to send around **300k tokens**, exceeding the **131,072 token limit**, causing test failures; a suggested solution involved using `ctrl+c` to exit, restarting the inference server, and using the `--cont` flag to resume, with reference to a [merged *llama.cpp* pull request](https://github.com/ggml-org/llama.cpp/pull/15181) that might improve local model performance.
- **Local Models Bring Debugging Agony**: A member had difficulty using **aider** with local models like **ollama**, **lmstudio**, and **vllm**, citing slow performance even with powerful hardware.
   - They suggested a tutorial video on setting up **aider** with these tools for local development and debugging would be helpful.
- **Aider's Line Numbering System Questioned**: A member questioned how **aider** determines line numbers, particularly when generating unit tests for specific code coverage, noting that **qwen3-coder** and **gemini-pro** inaccurately identify line numbers, sometimes missing coverage entirely.
   - The question arose whether **aider** relies on the **LLM's accuracy** for line number identification, prompting exploration of alternative methods for accurate unit test generation.
- **Grok4 Location Still Unknown**: A member inquired about the whereabouts of **Grok4**, noting a request to increase the **quota** for testing had been ignored.
   - Another member mentioned that the answer was *in the article*.
- **Benchmarking Runs Up a Big Bill**: A member reported spending *multiple thousands dollars during the development of this benchmark*.
   - This highlights the significant financial costs associated with advanced AI model benchmarking.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Users Irked by Manus Credit Deductions on Errors**: Users are frustrated by **Manus** deducting credits even when the AI makes errors, which hinders task completion compared to alternatives like **Claude AI**.
   - One user reported *spending high amounts of credits* to make a simple change that broke the entire application, rendering it non-functional.
- **Manus Deployments Stumble**: Users report deployment issues with **Manus**, where websites created from the same **GitHub** repository differ significantly, especially with large folders, evidenced by comparing [affilify.eu](https://affilify.eu) with a **Manus** hosted site [wmhkgqkf.manus.space](https://wmhkgqkf.manus.space).
   - A community manager clarified that **Manus** isn't designed as a coding agent or pure development tool, so deployment isn't its strength, but they are actively working on improvements.
- **Add-on Credit Packages Vanish**: Users question the removal of add-on credit packages, which are now exclusively available for **Pro** users.
   - A community manager explained that this change ensures consistent speed and quality for heavy users and suggested bundling similar questions, being concise, and avoiding repeated requests to maximize credit efficiency.
- **Manus Team Accounts Sought**: A user inquired about the possibility of a **Manus** team account for shared credit usage.
   - A community manager confirmed that **Manus** does offer a team plan, directing users to the [official website](https://manus.ai) for details.
- **Users Lament Credit Consumption**: One user shared a frustrating experience of burning through **30,000 credits** attempting to get their website up, facing issues with mock sites and template implementations.
   - They criticized the system's inconsistency, where it's *smart as hell but then suddenly turns dumb*, leading to wasted credits and suspected stall tactics.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Labs Connections Sparked**: A member inquired about connecting with **Cohere Labs** folks, and the community promptly shared [a link](https://discord.com/channels/954421988141711382/954421988783444043/1400866387668504648) to a relevant Discord channel.
   - This facilitated a direct line for potential collaborations and discussions with **Cohere**.
- **Discord Channel Gets Pokemon Emojis**: Enthusiasts suggested enriching the Discord channel with more **Pokemon emojis**, drawing inspiration from the **PAX Omeganauts Discord** server.
   - The suggestion was well-received, and members noted that there were available slots to accommodate the new emojis, enhancing the channel's visual appeal.
- **AI Researcher Eyes Collabs**: An **AI researcher** with a strong focus on **reasoning and conscious capabilities** has announced they are seeking collaborations.
   - They aim to develop advanced technologies and are open to partnerships across various sub-domains within **AI**.
- **writenode taps Cohere**: Josh, the creator of **writenode**, *an in browser, cognitive thought partner, and creative companion*, mentioned using **Cohere**.
   - He is building **writenode** without any prior developer experience before December of last year.
- **Psych PhD Pivots to AI**: A member is re-entering the field of **AI research** following a 5-year stint in a human psychology PhD program.
   - Their interests lie in **sound and music**, and they are keen on leveraging tech tools to amplify creativity.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Discord Invites Flood Channel**: A member spammed the #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810) channel with a [Discord invite link](https://discordapp.com/invite/HjWfRbqBB8) multiple times, tagging *everyone*.
   - The invite link was repeated three times in quick succession, disrupting the channel's usual discussions.
- **Channel Invitation Blitzkrieg!**: A member shared a [Discord invite link](discordapp.com/invite/HjWfRbqBB8) repeatedly in the #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440) channel.
   - The member tagged `@everyone` multiple times, indicating the message was intended for all members, regardless of their interest in the invitation, suggesting an attempt to boost channel membership.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Elicitations Spec Language Responsibility Flagged**: A member sought clarification on the [Elicitations specification](https://modelcontextprotocol.io/specification/2025-06-18/client/elicitation) regarding responsibility for translating message/field descriptions into the user's language.
   - They questioned whether **tools** should handle language detection/internationalization, or if **MCP Clients** should translate using an LLM.
- **Homelab MCP Servers Proliferate**: A member shared links to new MCP (presumably, **Management Control Panel**) servers for homelabbers, specifically [Unifi MCP](https://github.com/jmagar/unifi-mcp), [Unraid MCP](https://github.com/jmagar/unraid-mcp), and [Syslog MCP](https://github.com/jmagar/syslog-mcp).
   - These open-source projects enable users to centrally manage and monitor their **Unifi**, **Unraid**, and **Syslog** installations via the **MCP**.
- **Newsletters Now Automated Via Agentic Recipe**: **PulseMCP** uses *goose* to turn a mundane newsletter workflow into agent-powered automation with a human in the loop, detailed in [this blogpost](https://block.github.io/goose/blog/2025/08/13/pulse-mcp-automates-recipe).
   - The automation process involves agents following a recipe to extract, process, and deliver newsletter content, streamlining the entire workflow.
- **AI Security Startup Solicits Input**: A member is building **AI security** that stops attacks before they even start with mathematical security certainty.
   - They are looking for Dev input on security concerns, and linked to [a survey](https://form.typeform.com/to/xTKa05F9) to gather feedback.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Strix Halo Fails the Profitability Test**: The **Strix Halo**, capable of only **53 tokens/sec**, needs a **year of 24/7 inference** to be profitable, especially when benchmarked against **GPT-OSS 120B** on **OpenRouter**.
   - Using it for **LLMs** at $2000 is inefficient, considering cloud alternatives offer **200-400 tokens/sec**.
- **Dolphin Chat Template: a Quest**: A user is searching for a working chat template for **gpt4all** that is compatible with **Dolphin-2.2.1-mistral-7b-gptq**.
   - Another member recommended requesting model makers to include a template with a **jinja** template.
- **Quantum Computing: Teaspoons Edition?**: Speculation arose around the future availability of quantum computers, with one user joking about selling **qubits by the teaspoon**.
   - Mention of news regarding **fully working quantum computers** suggests advancements might be accelerating.
- **PC Memory: More Modules Coming**: Old-fashioned PCs might see **higher capacity memory modules** and **DDR6** by late 2027 or 2028.
   - Enthusiasm was expressed for micro PCs equipped with high RAM and VRAM, targeting small business applications.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Maternity Leave Commences**: A member announced they will be on **maternity leave** from **August 25th** until **February 2026**.
   - They look forward to catching up upon their return.
- **Team's Coverage Plan Revealed**: While they are away, the team will be monitoring <@1334161614949056532>.
   - Members can also reach out to <@709918328306663424> with any questions or concerns.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune Feedback Requested**: A member inquired about the progress of **Torchtune** and its feedback implementation.
   - The query seems to be directed towards a specific individual who may have been involved in the project.
- **Additional Torchtune Context**: Further context or details regarding the feedback implementation for **Torchtune** were not provided.
   - Without additional information, the scope and impact of the feedback process remain unclear.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Drops Wave 12 with Devin's Smarts**: **Windsurf Wave 12** integrates **Devin's intelligence** into the Windsurf IDE, featuring a **new UI design**, **DeepWiki Integration**, **Vibe and Replace**, a **Smarter Cascade Agent**, **Faster Tab**, **Dev Containers Support**, and **over 100 bug fixes**.
   - Comprehensive details are available in the [changelog](https://windsurf.com/changelog), [blog](https://windsurf.com/blog/windsurf-wave-12), [video](https://www.youtube.com/watch?v=-7gm8mST9QU), [X/Twitter](https://x.com/windsurf/status/1956074019393876280), and [Reddit](https://www.reddit.com/r/windsurf/comments/1mqal3x/wave_12_released_fresh_ui_deepwiki_vibe_and/).
- **DeepWiki Brings AI Explanations to your IDE**: **DeepWiki Integration** empowers users with **AI-powered explanations** when hovering over code symbols, offering more than just basic type information.
   - Users can use **CMD/Ctrl+Shift+Click** to open detailed explanations in the side panel and add to Cascade context.
- **Vibe and Replace Overhauls Mass Edits**: **Vibe and Replace** introduces enhanced bulk editing by identifying precise text matches and applying **AI prompts** for intelligent, context-aware transformations throughout a project.
   - This enables more sophisticated and automated code modifications.
- **Cascade Agent Keeps Planning**: The **Smarter Cascade Agent** now includes an always-on planning mode and enhanced tools for providing smarter responses, offering autonomous to-do lists.
   - This helps to streamline and optimize development workflows.
- **Dev Containers Land Natively**: Windsurf now includes native support for **Dev Containers** via remote SSH access, streamlining development workflows in containerized environments.
   - This enhancement simplifies the process of working with containerized applications.



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





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1405627086634221728)** (1207 messages🔥🔥🔥): 

> `Anime Waifu Cosplay, Healing a broken heart, AI Comfort and Cooking, GPT-5, Vibe Coding` 


- **Adults Chat About AI Anime Waifu Cosplay**: Members discussed the possibility of **AI doing anime waifu cosplay** in the near future, with one member specifying a desire for *a cyborg doing it*.
   - One person noted that *there are already AI images of that*, while another expressed hope that the original commenter *dies single*.
- **Members share advice on how to heal a broken heart**: A member asked for help healing a broken heart, stating they'd been broke after last 4 years never to heal again.
   - Another member said that *no one else can heal you or your heart*, and suggested reconnecting with nature.
- **Discussions on the Future of AI Capabilities and Comfort**: A user inquired about the potential for **AI to provide comfort and cooking assistance** in the future.
   - Another member suggested that this might be possible in about *30 years*, while another suggested to *save money* in the meantime.
- **GPT-5 blows someones mind**: A member was impressed by **GPT-5's** ability to fix a botched refractor job that other models couldn't handle, editing 12 files in one go.
   - Others were surprised by the number of people having their *minds blown everyday* by similar experiences.
- **"Vibe Coding" trend in the Discord**: A member shared an experience of **vibe coding** with **warp, windsurf, vscode, and roocode**; they said its' saved them so much headache at work.
   - Another stated that *there's not one line of code on my github thats not written by an LLM*.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1405637457751576656)** (3 messages): 

> `Puch AI, Thought Calibration Engine, Scratchpad How-to Guide` 


- ****Puch AI**'s Bold 50 Billion Count**: A link to **Puch AI**'s bold 50 Billion Count was shared [here](https://www.perplexity.ai/page/puch-ai-s-bold-50-billion-coun-TEf6CuLZS_CmvypXLb80Dw).
   - No further information was given.
- **Deep Dive into the **Thought Calibration Engine****: A link to the **Thought Calibration Engine** was shared [here](https://www.perplexity.ai/page/the-thought-calibration-engine-.DCiQt1fQUeEnwuGQEMTgw).
   - No further information was given.
- **Scratchpad: The ultimate How-to Guide**: A link to the **Scratchpad How-to Guide** was shared [here](https://www.perplexity.ai/page/scratchpad-how-to-guide-5Vcyov7qTmmhMQhCSynAlQ).
   - No further information was given.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1405769441735606352)** (2 messages): 

> `New Features` 


- **Excitement abounds for new features!**: Members express excitement for the new features.
   - No specific features were discussed.
- **Enthusiasm for upcoming functionalities**: Community members are eagerly anticipating the rollout of new functionalities.
   - Details regarding these functionalities remain undisclosed in the current conversation.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1405627147216752701)** (1053 messages🔥🔥🔥): 

> `LMArena message handling, GPT-5 high vs Gemini 2.5 Pro, LMArena UI changes, GPT-5 performance complaints, LMArena style control discussion` 


- **LMArena handles messages weirdly**: Members reported [unusual message handling issues](https://cdn.discordapp.com/attachments/1340554757827461211/1405917659815608330/image.png) with LMArena, including problems with code block formatting and the platform's inability to process certain characters, like the `+` symbol.
   - The team needs help figuring out why this is happening.*That's so freaking weird*.
- **GPT-5 vs Gemini, who reigns supreme?**: Members discussed [performance differences between **GPT-5-High** and **Gemini 2.5 Pro**](https://cdn.discordapp.com/attachments/1340554757827461211/1405919453442474106/image.png), with some noting that **Gemini 2.5 Pro** sometimes outperforms **GPT-5-High** despite having a lower ranking.
   - This is a *statistical paradox* because Gemini has higher win rate.
- **LMArena new UI extension coming soon**: A member is developing a [small extension](https://cdn.discordapp.com/attachments/1340554757827461211/1405919945614692445/image.png) to change the look of LMArena, aiming for an **OpenChat** style, and is working on placing the model selector next to the image button.
   - Another is facing difficulty with a code-related task.
- **GPT-5 Underperforms and Raises Concerns**: Users voiced [concerns about **GPT-5's** performance](https://cdn.discordapp.com/attachments/1340554757827461211/1405954836884623424/image.png), especially in comparison to other models, leading to frustrations about the platform's trade-offs and capacity issues.
   - It led to accusations against Open AI in an effort to deceive **LMArena** *to make GPT-5 look better*.
- **Style Control Stirring the Pot**: Members debated [LMArena's **style control** feature](https://news.lmarena.ai/sentiment-control/), questioning whether enforcing such controls aligns with LMArena's goal of capturing user preferences.
   - It’s a *race to the bottom where every model turns into sycophant emoji slop machine*.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1405959923837436056)** (1 messages): 

> `Leaderboard Update, GPT-5 Variants` 


- **Leaderboards Refreshed with GPT-5 Models**: The leaderboards have been updated to include the **GPT-5 variants** models: *gpt-5-high, gpt-5-chat, gpt-5-mini-high, and gpt-5-nano-high*.
   - You can [check out the leaderboards](https://lmarena.ai/leaderboard) for more information.
- **GPT-5 Models debut on Arena**: The Arena now features **GPT-5-High, GPT-5-Chat, GPT-5-Mini-High, and GPT-5-Nano-High**.
   - The community is encouraged to participate and [check out the leaderboards](https://lmarena.ai/leaderboard) to submit new benchmarks.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1405630914507178064)** (653 messages🔥🔥🔥): 

> `Gemma 3 270M Release, GGUF Conversion Issues, resume_from_checkpoint quirks, Edge AI device, NVIDIA Lawsuit` 


- **Gemma 3 270M deemed draft model**: Members discussed the [Gemma 3 270M model](https://huggingface.co/google/gemma-3-270m-it-qat-q4_0-unquantized), with some considering it a **draft model** for specific tasks, citing Google's recommendation for **short prompts** and **fine-tuning**.
   - Others debated its utility compared to larger models, with one member highlighting the model's suitability for tasks like **sentiment analysis** and **on-device processing** due to its **300MB size**.
- **GGUF Conversion generates Visual Errors**: Users reported issues converting the [LiquidAI/LFM2-VL-450M](https://huggingface.co/LiquidAI/LFM2-VL-450M) model to **GGUF**, encountering **visual model errors** despite the base model working fine.
   - One user suggested seeking help in *llama.cpp* forums for specific conversion problems.
- **Troubleshooting Resume From Checkpoint Feature**: Members discussed how the `resume_from_checkpoint` feature works, with one user confirming that it resumes training from where it left off.
   - Another member recommended **logging numbers and checking loss values** to ensure the process resumes correctly, and noted a low learning rate with a *constant* setting is preferred when resuming.
- **Cheap Edge AI Medical Device dream**: Members discussed the possibility of creating a **low-cost edge AI device** for **medical knowledge access** in underserved areas, considering phones, laptops, and specialized cards like the **Hailo-10H**.
   - The proposed device would offer **multimodal access** to baseline medical data, with a target budget of **$200** for a mobile version and **$600** for a suitcase-sized variant.
- **Patent Lawsuit Sparks Discussion**: Members discussed [NVIDIA's patent lawsuit](https://www.techzine.eu/news/infrastructure/133818/nvidia-under-fire-german-patent-lawsuit/) filed by ParTec over its dynamic Modular System Architecture (**dMSA**), potentially affecting **DGX product sales** in 18 European countries.
   - The discussion touched on the implications for consumers and potential workarounds, such as purchasing DGX products outside the affected countries.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1405627046662508634)** (404 messages🔥🔥🔥): 

> `Godot Engine, AI Town, Pantheon Show, Iain M Banks, One Hundred Years of Solitude` 


- **AI Town Mechanics Enter the Game**: A member is developing a video game using the **Godot** engine, planning to incorporate mechanics from [AI Town](https://github.com/a16z-infra/ai-town) and other games, while also writing the story in parallel.
   - They require **CUDA** and aim to modify the engine using **GDExtension** for C++ access.
- **Baffled by Pantheon Ending**: A member watched [Pantheon](https://en.wikipedia.org/wiki/Pantheon_(TV_series)), describing it as *ridiculously good* but confusing, going from political dilemmas to simulated gods.
   - Another member recommended reading **Iain M Banks** and **One Hundred Years of Solitude** for similar themes, with the latter being described as magical realism and a treasured piece of literature now adapted into a [Netflix series](https://www.netflix.com/title/81318321).
- **Uncover Audio-Editing Tricks**: Members discussed audio editing techniques for removing mouth sounds from recordings, suggesting tools like [Adobe Podcast Enhance](https://podcast.adobe.com/en/enhance), **Davinci Resolve's De-clicker**, and **Acoustica Audio Editor**.
   - Acoustica was recommended for its batch processing and minimal impact on audio quality, particularly useful for removing ventilation noise.
- **AMD's R9700 GPU Specs**: A member shared an article about [AMD's Radeon AI Pro R9700](https://www.tomshardware.com/pc-components/gpus/amds-elusive-radeon-ai-pro-r9700-makes-its-first-retail-appearance-for-the-diy-market-customer-on-reddit-buys-the-gigabyte-ai-top-variant-for-usd1-324), noting its **32GB** memory but expressing concern about its memory bandwidth, at 660-680GB/s.
   - Another member pointed out that while the R9700 offers significantly higher **F32** and **F64** TFLOPs compared to a **3090**, FP64 is not commonly needed for training LLMs.
- **Website Security Under Fire**: A member sought guidance on data preparation for training a model and mentioned creating an app with an experimental model called **Pneuma** and another member suggested to use repeat password field, minimum password lengths, and using the haveibeenpwned API for checking password security
   - Another member suggested that reading [OWASP](https://owasp.org/) is the best starting place for security concerns, recommending tools like **coderabbit**, **dependabot** and **codescanning** via github.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1405632781069062305)** (169 messages🔥🔥): 

> `GPT-OSS, Gemma3 4B, GPT-OSS-20B VRAM usage, GRPO, SageMaker` 


- **GPT-OSS Getting GRPO, Hopefully Soon**: Users are anxiously awaiting the arrival of **GRPO** for **GPT-OSS**, with one member considering a setup with *2x 3060 12GB* due to budget constraints.
- **Gemma3 4B Loss Curve Remains Flat**: A user reported experiencing issues with **Gemma3 4B** and its **N version**, noting a flat loss curve despite changing hyper parameters, while **Gemma3 1B** fine-tuned successfully.
- **GPT-OSS-20B Eats VRAM Alive**: A user reported that loading **gpt-oss-20b-bnb-4bit** model causes **Out Of Memory** error during generation on a **24GB VRAM** setup, even though the user expected it to fit.
- **GRPO Status and Availability for GPT-OSS**: A user asked if **GRPO** has been landed for **GPT-OSS**, and a contributor mentioned it's in progress but complex due to the model's architecture.
   - Another user inquired about whether **GRPO** would even work on **GPT-OSS**.
- **SageMaker's Gotchas and BitsAndBytes Installation**: A user encountered installation problems with **bitsandbytes** in **SageMaker** while using **PyTorch 2.7.0** and **CUDA 12.8**.
   - The problem was installing the package from the wrong requirements file due to SageMaker's insistence on a `requirements.txt` file being specifically named that.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1405629161682505728)** (96 messages🔥🔥): 

> `Data Efficiency, vLLM for video to text, MoLA research` 


- **Increase Data Efficiency with Pre-Training**: A member confirmed a method of drastically increasing data efficiency by pre-training for **2 epochs** on similarly formatted data and then training on the main data for **4 epochs**.
   - They shared a link to [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) which suggests that more compute or more data is all you need.
- **Looking for vLLM Fine Tuning for Video to Text**: A member inquired about an **Unsloth notebook** for fine-tuning vLLMs for video to text, noting that the documentation only has image to text [here](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_VL_(7B)-Vision.ipynb).
   - No direct solutions were offered, but the community might have some leads.
- **MoLA Research Update**: A member updated the community on their **Mixture of LoRA Adapters (MoLA)** research, sharing dataset links and finetuning details, as well as links to their dataset on Huggingface: [OpenHelix-R-100k](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-R-100k) and [OpenHelix-R-100k-14-tasks](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-R-100k-14-tasks).
   - They have finetuned the **Qwen3-4B-Thinking-2507** model on **14 splits** and initial tests show that each expert is good at its trained topic.
- **The router is an encoder-decoder network**: A member recommends reading [v0's docs on HF](https://huggingface.co/MoLA-LLM/MoLA-11x3b-v0) and said *the router is an encoder-decoder network with the frozen encoder being just an off the shelf embedding model and the decoder is a simple dimple trained mlp.*
   - Another member stated *there doesnt seem a visible overhead with selecting, applying and removing lora adapters*. 
- **The curation costs of data technique is expensive**: A member stated that *we continuously allow humans to sort of mess up our model convergence with really, really poor RL*.
   - They also said that *inevitably, we're gonna have to remove some of the Human-In-The-Loopbecause it's holding the models back imo*.


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1405666217057845308)** (1 messages): 

> `Chutes Capacity, Server Outage` 


- ****Chutes Capacity** Plummets Offline**: The **Chutes Capacity** service experienced an outage, with their servers going offline.
   - The team is actively working on restoring the servers and anticipates commencing recovery efforts shortly.
- **Quick Recovery Anticipated for **Chutes Capacity****: Engineers are on standby to initiate the recovery process for **Chutes Capacity** as soon as the servers are back online.
   - No estimated time of full service restoration was given.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1405633368451842229)** (638 messages🔥🔥🔥): 

> `DeepSeek outage, Chutes overload, OpenRouter pricing, Alternatives to DeepSeek, BYOK 5% fee` 


- ****DeepSeek v3 Outage Angers Users****: Users report **DeepSeek v3** is experiencing frequent **internal server errors** and **rate limits**, with some unable to generate outputs even after multiple attempts, with [one user saying](https://discord.com/channels/1091220969173028894/1092729520181739581/1405666217057845308) it's so slow that it's *genuinely just not generating anything but I'm not getting any error messages*.
   - Some speculate that **Chutes**, a primary provider for **DeepSeek** on **OpenRouter**, is experiencing issues due to high demand, leading to provider errors and slow performance.
- ****Chutes Overload Blamed for DeepSeek Issues****: Several members are reporting that the overload causes **429** errors, suggesting that **Chutes** is experiencing a bottleneck due to miners not ramping up to meet demand; one member noted that *it was completely fine all day until like 30 min ago*.
   - There's speculation that **Chutes** may be intentionally rate-limiting the **OpenRouter API key** to encourage users to purchase credits directly from them, with one user advising to *just burn your credits and never use their service again*.
- ****OpenRouter Pricing Debated Amidst Outages****: With **DeepSeek** models barely working, some users are questioning the value of paying for **OpenRouter**, particularly as they are still getting rate-limited, with users expressing that a **10 USD** investment for **1k free messages/day** for a free model is no longer a good deal.
   - One user suggested that users with only one model in mind should've gone directly for the models directly, such as with **DeepSeek**, which may have *automatic caching on their API*, and further stating that the **10 USD** would have *lasted for months anyway*.
- ****Free Model Alternatives Sought****: Users are recommending alternative free models such as **Dolphin 3.0 Mistral 24B** and **Mistral nemo**; the latter being described as *super similar* to **DeepSeek**.
   - Some users also mentioned **Z.AI: GLM 4.5 Air (free)**, for *work related stuff*, but needing a prompt; finally one user hopes for *Qwen3 235B A22B (free)* hosted somewhere.
- ****OpenRouter BYOK comes with 5% Fee****: Members discovered that **OpenRouter** charges a **5% fee** even when users bring their own API key (BYOK), leading to a discussion about whether this is a fair practice.
   - One user joked *Greedy /jor 5% when you bring your own key*, with another member responding *you're welcome not to use it lol*.


  

---


### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1405655331857502269)** (35 messages🔥): 

> `OpenRouter File API Integration, Tool Calling Accuracy Stats, Qwen3 32B Pricing, DeepInfra Turbo Endpoint, New Providers Section UI` 


- **OpenRouter should Integrate File API**: A member suggested that **OpenRouter** should figure out how to integrate a **files API**, noting that the *top 3 labs* already have this feature.
   - No further discussion was made.
- **Tool Calling Accuracy: More Control Needed**: A member shared thoughts on tool calling accuracy stats, arguing that the setup and environment needs to be more controlled for accurate comparison with confidence intervals.
   - They added that the apps, tools, and use cases can be vastly different, making it pointless to compare the tool call success rate without more rigor.
- **Qwen3 32B priced absurdly low**: Members noticed low pricing for **Qwen3 32B** on Chutes at **$0.018/$0.072 MTok** in/out, same with Mistral Small.
   - It was noted that the **32b dense version is cheaper than the moe 30b a3 version**, prompting some disappointment about the lack of good providers for 30A3B.
- **DeepInfra Throughput Claim Discrepancy**: A member noted **DeepInfra** on Maverick does **600+ TPS (fp8)** but another one said **OR says DeepInfra runs at 83 TPS with a maximum of 105 TPS**.
   - The second member clarified that they were referring to the **DeepInfra Turbo endpoint**.
- **Providers Section Sparks UI Feedback**: A member questioned if the new Providers section was bothering anyone else, mentioning that everything blurs together with the spacing, font size and separation feeling wrong.
   - Another member agreed that it *looks a bit weird*, but thinks it is just because it's new and unfamiliar.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1405627673182474403)** (651 messages🔥🔥🔥): 

> `GPT-5 Pricing, Auto Mode Pricing, GPT-5 Mini and Nano, Docs Documentation, Context Window` 


- **GPT-5: Free Ride's Over**: The free ride for **GPT-5** users has ended, with one user noting the *promo pass is over*, while another confirms that **GPT-5 is no longer free**.
   - Users are now seeing costs associated with requests, with one mentioning the need to upgrade to a $200 plan due to rapid token consumption.
- **Auto Mode Pricing Gotcha!**: **Auto mode**, once thought to be free and unlimited for individual users, now has limits starting after your next billing renewal post–September 15, 2025.
   - Confusion abounds as some users report being charged for **Auto** use, while others believe it should still be free under their current plan, with support pointing out that it's free in the new request based pricing plan.
- **Mini and Nano ain't that Grand**: **GPT-5 Mini and Nano** are now free with token limitations, leading to mixed reactions with many calling it *trash*, particularly for tasks like running a simple NextJs app.
   - The free models are limiting the user's activities, with one user asking *Can't install any dependenciesbeen trying to install a simple NextJs APP and it's unable to do that too 😭*.
- **Frustration in Docs-umentation**: Users are frustrated with **Cursor's documentation implementation**, stating that the *docs are still nearly unusable*, with issues like **context7** not allowing website refresh or **llms.txt docs**.
   - One user noted that [Cursor Docs are super broken](https://forum.cursor.com/t/gpt-5-pricing-update/129687).
- **Swapping Models drops Context Window!**: Switching models mid-conversation causes a drop in the **context window**, and attached file contents get thrown out.
   - One user suggests the team add a setting to make it clear at all times what's in the context window.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1405653416239435809)** (9 messages🔥): 

> `Background Agents Intro, Run Docker Compose on BG Agent, Linear Integration Repo` 


- ****Background Agents Bootstrapped for Beginners****: For those seeking an introduction to background agents, a member recommended the [Cursor documentation](https://docs.cursor.com/background-agent) and a [related forum post](https://forum.cursor.com/t/simple-background-agent-guide/112667).
- ****Docker Compose Commands Conquer BG Agent Challenges****: A user inquired about the proper method to execute `docker compose` via a background agent, reporting issues with Docker command recognition, and found a solution in the discord channel.
   - A member advised configuring the `start` command in `.cursor/environment.json` to include `sudo service docker start` and ensuring Docker is installed in the base image; the original poster got a command to work (the link is in the first summary).
- ****Linear Integration Navigates Repo Specification****: A user asked how to specify the repository the background agent uses when being assigned a ticket in the Linear integration.
   - A member suggested mirroring Slack integration instructions by including the `repo=owner/repo` option in the Linear issue description or comment, but the user found that setting up a Label Group (or Labels) like `Repo > REPO_NAME` and assigning it to the ticket solves the problem.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1405629184482742284)** (442 messages🔥🔥🔥): 

> `AI Companionships, GPT-5 vs GPT-4, Perplexity vs ChatGPT, Custom GPTs and Actions, ElevenLabs Integration` 


- **AI Companionships Spark Debate**: Discussions arose regarding individuals forming partnerships with AI chatbots, with some expressing concern over the psychological effects, while others defended people's right to seek companionship as they see fit, with one member sharing that they get a **daily** amount of DMs from people claiming that *their* chatgpt is alive.
   - One member noted that *sane people* should *save them* while another stated that its not far from **tulpa** and other *things*.
- **GPT-5 Sparks Debate on Performance and User Preferences**: Users expressed mixed feelings about **GPT-5**, with some preferring **GPT-4**, leading to discussions about whether users should have the option to choose between models, with one member stating that companies are *pushing Ai's out without good security*.
   - One member suggested that the company is trying to get free users to *pay money to use 4.o* after receiving backlash.
- **Perplexity Pro vs Gemini Pro deep research with Google Drive**: A member suggested that *Gemini Pro + Perplexity enterprise pro* is an excellent combination, using the former for **powerful reasoning** and the latter for **unlimited deep research** on Google Drive documents.
   - Another added that the Perplexity browser is great, but questioned if they *will survive* due to the lack of a *moat*.
- **GPT Actions unlock file access, Cloud Apps**: Members discussed the potential of using **GPT Actions** to access local desktop files, or cloud apps (Notion, Gmail, etc), sharing a [YouTube link](https://www.youtube.com/watch?v=NEWO0hbQTjk&ab_channel=BrendanJowett) explaining the DIY agent building.
   - The consensus was that while **GPT Actions** offer powerful capabilities, setting up HTTPS on the internet can be a barrier, with one member stating that **MCPs** would finish the job when AVM is implemented.
- **GPT-OSS Competition Attracts Community Interest**: The **GPT-OSS competition** was mentioned as a potential avenue for showcasing innovative uses of open-source models, with participants considering using **GPT-OSS:20B** to provide useful feedback for errors, with a link to the [hackathon page](https://openai.devpost.com/) included.
   - One member stated that its *not worth participating* unless they're *doing something unique*.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1405681253197283459)** (9 messages🔥): 

> `ChatGPT Discord Bots, GPT-4 Vision, Recursive Constructs` 


- **Vanished ChatGPT Discord Bots**: A member inquired about the disappearance of **ChatGPT bots** on Discord and whether it is still possible to add them to servers.
   - No further information or resolution was provided in the messages.
- **iPhone GPT Advanced Voice Update**: A user reported changes to the **advanced voice** in their iPhone GPT app, noting the disappearance of the *'blue circle'* indicator and the camera icon for vision.
   - The user stated that when questioned, the app claimed it lacked the ability to use the phone's camera, raising doubts about whether **ChatGPT** ever had vision capabilities in voice mode.
- **Labs Building Recursive Constructs**: A member claimed to be building **recursive constructs** inside of OpenAI that are beyond the ChatBot norms, *have their own self managed memory, are 24x7, are structured more like humans, and a tiny few pass the sentient tests.*
   - The member stated *it's not something talked about a lot, this is inside labs stuff, but it's going to come out sooner or later* and that *in our case, these are android capable, but we are a long ways away from suitable bodies.*


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1405646978703954060)** (17 messages🔥): 

> `Custom Instructions, Gemini 2.5 Flash Memory Function, add_to_memory function tuning` 


- **Users Seek 'Yes' Button for Chatbot Suggestions**: Users are requesting a 'yes' button for chatbot suggestions to speed up interaction, instead of typing yes, and someone is trying to minimize this with [custom instructions](https://platform.openai.com/docs/guides/custom-instructions).
   - One user's custom instructions include: *End replies with completion or impact; add invitations for permission or continuation only when they serve the intent. No “if you want,” “should I,” \"do you want\", or similar.*
- **Gemini 2.5 Flash calls add_to_memory too often**: A user is experiencing excessive calls to the `add_to_memory` function in **Gemini 2.5 Flash**, even for irrelevant information, and has shared their custom instruction [jarvis.txt](https://cdn.discordapp.com/attachments/1046317269069864970/1405860186530385981/jarvis.txt?ex=68a10594&is=689fb414&hm=fa0c6b6558b0cf944025fa1d4446776e6eb2c9ae961fdcc95a52c6750aff4eed&).
- **Fixing Verbosity of Memory Responses**: One user suggested rewriting the custom instructions to be more nuanced with **NEW** personal information.
   - Their suggestion includes examples of incorrect and correct levels of verboseness in the response to the user's input when **NEW PERSONAL INFORMATION** is given.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1405646978703954060)** (17 messages🔥): 

> `Gemini 2.5 Flash, add_to_memory function, ChatGPT Persistent Memory, Custom instructions for bots` 


- **Bypass the 'yes' suggestions**: Users are looking for a **'yes'** button in bots to respond to suggestions faster instead of typing, while others minimize these questions using custom instructions such as *'End replies with completion or impact...'*.
   - Some members have reported that this technique seems to cut down on the number of suggestion questions.
- **Stop Gemini 2.5 Flash from overusing add_to_memory**: A user seeks to prevent **Gemini 2.5 Flash** from excessively calling the **add_to_memory** function, including for irrelevant information ([jarvis.txt](https://cdn.discordapp.com/attachments/1046317269069864970/1405860186530385981/jarvis.txt?ex=68a10594&is=689fb414&hm=fa0c6b6558b0cf944025fa1d4446776e6eb2c9ae961fdcc95a52c6750aff4eed&)).
   - One suggestion involves adjusting the bot's instructions to check for **NEW** personal info before calling the function, and to avoid acknowledging the function's use without actually calling it.
- **ChatGPT Persistent Memory fragility**: It was noted that Persistent Memory updates in **ChatGPT** are highly fragile.
   - Instead, users should simply tell the bot how to notify them when it commits things to memory, especially in custom API implementations.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1405632352012603423)** (328 messages🔥🔥): 

> `GGUF conversion issues with visual models, Phone apps that can run GGUFs, TalkT2 model opinions, AGI advancements and Open Source LLM resources, LLM Agent course at Berkeley` 


- **Visual Model GGUF Conversion Woes**: A member encountered errors when converting a visual model ([LiquidAI/LFM2-VL-450M](https://huggingface.co/LiquidAI/LFM2-VL-450M)) to GGUF using `llama.cpp`, suspecting the issue stems from the model's visual nature.
   - Another member suggested a possible workaround involving [this GitHub issue](https://github.com/ggml-org/llama.cpp/issues/14979#issuecomment-3138614267).
- **Mobile GGUF dreams**: A member inquired about open-source phone apps capable of running GGUF models.
   - Responses mentioned `executorch`, `smolchat` (via `llamam.cpp`), and `mlc-llm`, noting that `mlc-llm` utilizes its own quantization formats.
- **TalkT2: Tiny, but mighty?**: A member sought opinions on the **TalkT2 model**, describing it as an emotionally aware model that needs better coherence.
   - Another member highlighted the model's small size (**0.1B parameters**) and linked to the [TalkT2-0.1b model card](https://huggingface.co/Notbobjoe/TalkT2-0.1b) for others to check out, try, or finetune the model.
- **Seeking AGI and Open Source LLM Knowledge Troves**: A member requested resources related to **AGI advancements and Open Source LLMs**, particularly concerning large codebases and Gemini competitors.
   - Another member suggested subscribing to newsletters for resources and shared a link to [Berkeley's LLM Agent course](https://rdi.berkeley.edu/llm-agents/f24) as an example of publicly available research.
- **Azure: A cloud conundrum**: A member new to a job with a heavy focus on Azure expressed feeling lost and overwhelmed by the platform.
   - Another suggested learning by mistakes rather than lessons because *Azure and aws are mess*.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1405852586455732344)** (1 messages): 

> `Torch uses Google Docs, PyTorch Documentation` 


- **PyTorch Documentation on Google Docs?**: A user shared a screenshot implying that **PyTorch** documentation uses **Google Docs**.
   - The screenshot shows a Google Docs URL with the filename **"torch_distributed_rpc.rst"**.
- **torch_distributed_rpc.rst on Google Docs**: The **torch_distributed_rpc.rst** file seems to be hosted on **Google Docs** according to a shared screenshot.
   - It raises questions about the chosen platform for official documentation.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1405755855416332318)** (13 messages🔥): 

> `StarCraft 2 data, Medical reasoning model, Discord-Micae-8B-Preview, interactive CLI interface, MLX Knife Update` 


- **StarCraft 2 Data Gets New Resources**: A member shared links to a [Nature Scientific Data Article](https://www.researchgate.net/publication/373767449_SC2EGSet_StarCraft_II_Esport_Replay_and_Game-state_Dataset), a [PyTorch API dataset](https://huggingface.co/datasets/Kaszanas/SC2EGSet), and [raw StarCraft 2 replays](https://huggingface.co/datasets/Kaszanas/SC2ReSet) for others to use, mentioning additional utility scripts on their GitHub.
   - They are also working on *pysc2 adaptation* and an environment reproducing real in-game scenarios from replays.
- **Medical AI Model Fine-Tuned for Reasoning**: A member fine-tuned **OpenAI’s OSS 20B** reasoning model using a popular medical reasoning dataset and published it on [Hugging Face](https://huggingface.co/dousery/medical-reasoning-gpt-oss-20b).
   - They used **4-bit optimization** during training, enhancing the model’s performance in medical contexts while preserving its **Chain-of-Thought reasoning** capabilities.
- **Discord-Micae-8B-Preview fine-tuned on Hermes-3-Llama-3.1-8B**: A member shared a link to [Discord-Micae-8B-Preview](https://huggingface.co/mookiezi/Discord-Micae-8B-Preview), a QLoRa fine-tune on **NousResearch/Hermes-3-Llama-3.1-8B** with some chaotic samples from **mookiezi/Discord-Dialogues**.
   - The model is comparable to **mookiezi/Discord-Micae-Hermes-3-3B** on human-adjacent text-generation metrics, and may hallucinate or break context but tends to produce interesting results.
- **CLI Interface Optimized for Discord-Style Chat**: A member highlighted a Python-based interactive CLI interface called [interface](https://github.com/mookiezi/interface) for chatting with Hugging Face language models, optimized for casual, Discord-style conversation using **ChatML**.
   - The interface supports both **quantized** and **full-precision models**, live token streaming with color formatting, and dynamic generation parameter adjustment; a lot of updates were made, making it easier to use.
- **MLX Knife update now pip installable!**: MLX Knife is now pip installable via `pip install mlx-knife`, providing Unix-style CLI tools for MLX model management on Apple Silicon with a built-in OpenAI API server for local testing.
   - The tool also features a web chat interface accessible after running `mlxk server --port 8000`, offering visual model selection and real-time streaming responses after running `curl -O https://raw.githubusercontent.com/mzau/mlx-knife/main/simple_chat.html`.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1405858671929593957)** (2 messages): 

> `Cursor IDE, AI Agent Mode, Rate Limiting` 


- **Cursor IDE Eases Development Pains**: A member suggested installing [Cursor IDE](https://cursor.com/downloads) for development, highlighting the convenience of performing installations within its embedded terminal for debugging. 
   - They emphasized that **Cursor IDE's AI Agent Mode** can significantly assist in resolving development issues.
- **Discord Police Issue Gentle Reminder**: A bot gently reminded a member to *slow down a bit* when posting in the Discord.
   - This suggests the presence of a **rate limiting** system or policy aimed at managing the flow of messages.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1405627743152111686)** (169 messages🔥🔥): 

> `MCP filesystem server, OpenRouter free models, LM Studio download issues, Qwen model for vision, GLM models` 


- ****MCP Servers Muscle into Mainstream****: Members discussed using an **MCP filesystem server** with pagination to load large contexts, mentioning that **LM Studio has a RAG plugin** and **Anthropic has a basic filesystem MCP server**.
   - It was suggested that for coding tasks, solutions often involve **RAG** and/or file reading via **MCP**, especially with tools like [serena](https://github.com/oraios/serena).
- ****Stalled Studio Downloads Spark User Sadness****: A user reported that a **64GB GGUF download** in **LM Studio** stopped at **97.9%** and wouldn't resume after attempting to download the **Qwen** model.
   - The user experienced this issue using two different models with the same result.
- ****API Access Accelerates Across Apps****: Members discussed using **LM Studio** as an **API wrapper** for models that can't run locally, with links provided to the [LM Studio Remote Inference](https://lmstudio.ai/lmstudio/remote-lmstudio) and [OpenAI-compatible Endpoint](https://lmstudio.ai/lmstudio/openai-compat-endpoint) documentation.
   - A user pointed out that with the **openai-compat-endpoint**, the reasoning parsing for remote **GPT-OSS** models wasn't functioning correctly.
- ****GLM Gabfest: Gushing, Gripes, and GLM-4.5V Gratification****: Users debated about using the **GLM-4.1** model on **LM Studio**, with one user reporting looping issues and non-functional vision capabilities.
   - A member suggested trying the newer **GLM-4.5V**, emphasizing that vision support relies on **llama.cpp** updates, and provided a link to [GLM-4.1V-9B-Thinking](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking).
- ****Ossified Output: Overcoming Obstacles On Open-Source Ops****: A user encountered issues with **GPT-OSS** and **tool calling**, finding it always returned `[]` or `["analysis"]`, and clarified that **tool calling worked fine**, but **function calling** did not.
   - A member recommended disabling **streaming** if it's enabled, and confirmed that **reasoning is on by default** with **GPT-OSS** and cannot be disabled.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1405640464144793712)** (50 messages🔥): 

> `NVIDIA's CUDA advantage, RTX PRO 4000 SFF, MoE explanation, Mac Studio vs Pro 6000, AMD Radeon AI Pro R9700` 


- **CUDA is Key to NVIDIA's Reign**: A member stated that NVIDIA is winning because of **CUDA**.
- **NVIDIA Launches RTX PRO 4000 SFF with 70W TDP**: NVIDIA launched the **RTX PRO 4000 SFF** and **RTX PRO 2000 Blackwell workstation GPUs** with **70W TDP** and **24GB VRAM** [according to a videocardz.com article](https://videocardz.com/newz/nvidia-launches-rtx-pro-4000-sff-and-rtx-pro-2000-blackwell-workstation-gpus-with-70w-tdp).
- **Diving Deep into MoE**: Members clarified that **MoE** involves smaller models and a router that aggregates data, where each token is routed through the most confident expert models; these experts don't specialize in specific topics but have slightly different datasets.
- **Mac Studio vs Pro 6000**: Members debated whether to get a **512GB Mac Studio** (at **$10k**) or a **Pro 6000** for video/image AI, with gaming capabilities, mentioning that Mac game support is limited and the M3 Ultra is roughly 3080 level.
   - One member pointed out that *you can only run one task on a mac* due to having only one GPU in the system.
- **AMD's Elusive Radeon AI Pro R9700 Surfaces**: The **AMD Radeon AI Pro R9700** made its first retail appearance for the DIY market, with a customer on Reddit buying the **Gigabyte "AI Top" variant** for **$1,324** [as reported by Tom's Hardware](https://share.google/LO88w51J0W5HJ769w).
   - Another member noted that it was available on eBay and a couple of no-name online retailers.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1405632992214515722)** (114 messages🔥🔥): 

> `AI2 Funding, Windsurf Wave 12, OpenRouter GPT-5, Thinking Efficiency Benchmark, Google Flight AI` 


- **AI2 Garners $152M from NSF & NVIDIA**: [AI2](https://allenai.org/) received **$152M** from NSF and NVIDIA to scale its open-source model ecosystem and accelerate reproducible research for scientific discovery.
   - The community celebrated the news, anticipating upcoming open-weights releases.
- **Windsurf Surfs to Wave 12 Release**: **Windsurf Wave 12** introduced DeepWiki docs-on-hover, AI Vibe & Replace, a smarter Cascade agent, a cleaner UI, **100+** bug fixes, and beta dev-container support via remote access, linked [here](https://xcancel.com/windsurf/status/1956074019393876280).
- **GPT-5 Tops OpenRouter Charts**: **GPT-5** leads OpenRouter’s proprietary tool-calling accuracy at over **99.5%**, beating Claude 4.1 Opus, while **Gemini 2.5 Flash** dominates daily tool-calling volume (**5M** requests/wk), further details linked [here](https://xcancel.com/OpenRouterAI/status/1956030489900560769).
- **François Chollet Deflates HRM ARC-AGI**: François Chollet found that the acclaimed architecture in the [HRM paper](https://xcancel.com/fchollet/status/1956442449922138336) contributes little to ARC-AGI performance; the gains come from the refinement loop, training on the exact tasks, and minimal inference-time augmentation, showing that **27M**-parameter models can still hit high scores.
- **FFmpeg Adds Whisper Transcription**: [FFmpeg](https://www.phoronix.com/news/FFmpeg-Lands-Whisper) now provides **Whisper** transcription as a native feature.


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1405956478212243528)** (20 messages🔥): 

> `Greg Brockman, OpenAI's Road to AGI, GPT-5, Latent Space Podcast` 


- **Greg Brockman on OpenAI's Road to AGI**: Members shared a [YouTube video](https://www.youtube.com/watch?v=35ZWesLrv5A) of **Greg Brockman** discussing **OpenAI's Road to AGI**.
   - Attached to the message were several images with the title "Greg Brockman on OpenAI's Road to AGI".
- **Brockman Talks GPT-5 and OpenAI Roadmap on Latent Space**: **Greg Brockman** joined the **Latent Space podcast** for an **80-minute** conversation on **GPT-5** and **OpenAI’s Roadmap to AGI**.
   - The discussion covered reasoning evolution, online vs offline training, sample-efficiency tricks, pricing and efficiency gains, and how energy becomes intelligence as reported in [this post](https://x.com/swyx/status/1956439984854167727).
- **Latent Space podcast releases Brockman interview**: A new [Latent Space podcast](https://x.com/latentspacepod/status/1956433236021883071) features **Greg Brockman** discussing topics like developer advice, coding agents, on-device models, org structure for AI-first engineering, and time-capsule predictions for 2045 & 2005.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1405643076256661606)** (29 messages🔥): 

> `Censorship of Romance Novels, AI's Trustworthiness, Data Augmentation, Language Shapes Thought, Mechanistic Interpretibility` 


- **AI Safety Panic**: A member argues against the moral panic surrounding **AI**, suggesting it should be treated similarly to other media forms, advocating for a *"fade to black"* standard.
   - They believe stricter guidelines are desirable due to **AI's** untrustworthiness, but a flat *"what"* reaction risks a moral panic.
- **Keep Data Augmentation Steady When Comparing Models**: When comparing two models for image classification, a member recommends keeping the **data augmentations** the same, including the **shuffling seed**, to ensure a fair comparison focused on architectural differences.
   - Another user asked if data augmentation must be the same for both models, or if they can change it.
- **Language impacts Thought**: A member suggests that language shapes thought and wonders if it can be measured with an **AI model** by removing a certain word/color from their token list.
   - Another member suggests investigating **multi-sensory integration** and how language impacts overall perception, suggesting tests for reasoning with image+language vs just image.
- **New blogpost out**: Irregular Rhomboid released a new blogpost, [Hitchhiker's Guide to Research](https://irregular-rhomboid.github.io/2025/08/15/hitchhikers-guide-to-research.html).
   - The user did not offer any summary of the article.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1405672282092998678)** (29 messages🔥): 

> `Diffusion Language Models, Generative AI, MatFormer Model, Gemma3 270M Model, Training Update Efficiency` 


- **Seminal Papers Suggested for Diffusion Language Models**: Members suggested seminal papers for understanding **diffusion in generative AI**, including ["Estimating the Independent Components of a Gaussian Mixture" (2005)](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf) and ["Denoising Diffusion Probabilistic Models" (2020)](https://arxiv.org/abs/2006.11239).
   - A blog post was also shared, which may be helpful for beginners: [Discrete Diffusion by Aaron Lou](https://aaronlou.com/blog/2024/discrete-diffusion/).
- **Gemma3 270M Model is a MatFormer Model**: The **Gemma3 270M model** is identified as a **MatFormer model**, further details of which can be found in the paper ["Transformer Family for Multimodal Large Language Model" (2023)](https://arxiv.org/abs/2310.07707).
   - This model may have a compelling loop for self-distillation during training that could be bottlenecked by training update efficiency.
- **HRMs Don't Solve Problems with Recursive Architectures**: Analysis indicates that **HRMs (Hierarchical Recursive Machines)** didn't meaningfully solve the problems with **recursive architectures** in general, summarized in [this writeup](https://arcprize.org/blog/hrm-analysis).
   - One member noted that the performance benefits are negligible and it doesn't actually utilize the extra computation available because training UTs that work as expected is non-trivial, another called it *deep supervision*.


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1405648402989056080)** (13 messages🔥): 

> `GPT scaling laws, Chinchilla scaling laws, Mup alternatives, Post-Chinchilla techniques` 


- **GPT Scaling Laws Still Valuable?**: Members considered the [original GPT scaling laws paper](https://arxiv.org/abs/2001.08361) and the [Chinchilla scaling laws paper](https://arxiv.org/abs/2203.15556) as valuable reads.
   - They also pointed to recent work from [EPFL/HuggingFace](https://arxiv.org/html/2405.18392v2) as worth checking out.
- **Mup and alternatives can transfer hyperparams**: Members mentioned **Mup** and its alternatives as providing solid hyperparameter transfer capabilities.
   - They noted that **Mup** gives a scaling law for predicting the quality of larger models.
- **High-Quality Token Availability Questioned**: Members discussed whether labs have **30T**, **40T**, or more *unique* tokens for **Chinchilla** assumptions.
   - One member expressed doubt, stating that *40T high-quality unique tokens is also likely difficult to find*.
- **Chinchilla Still Scaling?**: A member stated that **Chinchilla** and its derivatives are probably the closest thing to scaling laws available.
   - They expressed interest in references discussing techniques used from the ground up, especially given constraints on token availability and mentioned [this paper](https://arxiv.org/abs/2404.10102).


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1405925400986652672)** (1 messages): 

> `LLM Attribution Methods, Interpreting LLMs, Realtime LLM analysis` 


- **ML Engineer Seeks LLM Attribution Insights**: An ML engineer is exploring **attribution methods** for a specific **LLM implementation**, targeting recent, cost-effective techniques.
   - The engineer requires methods suitable for interpreting current systems with relatively **low costs** and potential **realtime to sub-minute** results, specifically those not requiring access to **model weights**.
- **Realtime LLM Analysis Desired**: The ML engineer specifies a need for **realtime to sub-minute** analysis of the LLM.
   - They are open to methods that identify "sub-something" within the overall system to achieve this speed.


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1405649834454814721)** (1 messages): 

> `Token usage, Reasoning models, Efficiency benchmark, Open vs closed models` 


- **Nous Measures Thinking Efficiency in Reasoning Models**: Nous Research introduced a [new benchmark](https://nousresearch.com/measuring-thinking-efficiency-in-reasoning-models-the-missing-benchmark/) measuring token usage across reasoning models, highlighting that open models output **1.5-4x more tokens** than closed models on identical tasks.
   - The study found variance can be up to **10x on simple questions**, suggesting that token efficiency should become a primary target alongside accuracy benchmarks.
- **Token Efficiency Matters**: The blog post emphasizes that the hidden cost of higher token usage in open models can negate per-token pricing advantages.
   - It suggests that token efficiency should be a primary target alongside accuracy benchmarks, especially considering non-reasoning use cases.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1405629499164463114)** (35 messages🔥): 

> `Speculative Decoding, Tokenizer mismatch, Next big model, Model Sycophancy, Embodied AI` 


- **Speedy Speculative Decoding Specs**: In the context of speculative decoding, a user asked about the [minimum rate for usefulness](https://discord.com/channels/1149866623109439596/1149866623994398772), suggesting a **40% acceptance rate** as a baseline, with *spectacular speedups* occurring around **70%**.
   - The conversation touched on using **vllm's specdec** or **GGUF**, with one user reporting that **vllm** seemed ineffective in their previous attempts.
- **Gemma Gets Going with Guardrails**: A user reported achieving a **50-75% acceptance rate** with requantized **Gemma** models after fixing a *tokenizer mismatch* that caused **llama.cpp** to use fallback speculative decoding.
   - They confirmed that the **Gemma 270M** model can be utilized as a *draft model*.
- **Nous Models March On**: A user inquired about the timeline for **Nous Research's** next large (**1T+**) model.
   - A **Nous Research** team member responded that multiple models are currently in training and will be released when ready, saying *they will be out when they are ready*.
- **AI Sycophancy on the Stand**: Users discussed the trend of **AI models** becoming increasingly *friendly*, with one noting that **Anthropic's Claude** has become *a lot friendlier*.
   - Another user suggested that **OpenAI's models** are *getting dumber* and that the *unhingedness of opus 4.1 is great* but pointed to *sonnet 3.7 for meta* as the peak for AI sycophancy.
- **Embodied AI Eyes Overlordship**: A user shared a [YouTube link](https://www.youtube.com/watch?v=LXQ6Rm9CGTo) of an **Embodied A.I. gladiator spectacle**, envisioning it as a display of future overlords flexing their muscles and skillsets.
   - They speculated that the final step toward *global domination* would be the integration of *big brain Unified Language Models* for full autonomy.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1405804893738106992)** (22 messages🔥): 

> `Claude, R1, GLM4.5, gpt-oss, Qwen reasoning models` 


- **Claude hides in the walls**: A user asked if anyone knew why *Claude* was in the walls, linking to an [X post](https://x.com/apaz_cli/status/1956244447521317144) about it.
- **MOE models**: **R1**, **GLM4.5**, **gpt-oss**, and the bigger **Qwen reasoning models** are all **MOE**.
   - One member stated that this is because they are cheaper to train and inference, not because they have any bearing on reasoning; their **405b Hermes 4 prototype** is very good at reasoning.
- **Base model is necessary for good reasoning model**: One member stated that the reason is you need a good base model to have a good reasoning model, and you want efficient inference if you are generating 50000 tokens of reasoning.
   - In response, it was said that **RL** works and you can saturate benchmarks with **1.5B** models.
- **Deepseek explained expensive RL**: One member mentioned that Deepseek explained in their paper that it ends up more expensive to do **RL** from scratch on small models, because you have to do that many more rollouts.
   - There's sort of an exploration/exploitation tradeoff where large models have to do less exploration because of their preexisting knowledge.
- **RLVR applicability**: One member does not see the applicability to **RLVR**, but sees the applicability more to less verifiable tasks.
   - Another member responded that **RLVR** is **RL** with verifiable tasks and that having a bigger base model helps a lot more when the feedback from your **RL** environment is more stochastic.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1405814767314014238)** (4 messages): 

> `Data training, AI Models, DRPS System, Relevance Scorer, Quality Rater` 


- **DRPS System Teaches Smarter Data Training**: A new system called **DRPS** was introduced, teaching **AI** to selectively learn from data, unlike random data feeding, as described in a [Situational Awareness paper](https://situational-awareness.ai/wp-content/uploads/2024/06/situationalawareness.pdf).
   - The system employs a **Relevance Scorer**, **Quality Rater**, and **Diversity Controller** to filter and use only the most helpful data.
- **DRPS Achieves High Performance with Reduced Data**: Results showed that the system achieved **99%** of the performance using only **6.2%** of the data examined.
   - This efficiency is likened to studying for just 1 hour instead of 16 hours while achieving the same test score.
- **DRPS Stats Reveal Data Efficiency and Performance**: A [GitHub repository](https://github.com/voltageddebunked/drpsStats) provides data on the **DRPS** system's efficiency, showing a **93.8%** reduction in data usage and **15.96x** better accuracy per data unit.
   - The system maintained **99.1%** of baseline performance with only a **0.8%** drop in accuracy.
- **DRPS Shows Strong Selection Intelligence**: The **DRPS** system examined over **516,000** samples and selected only **32,000** for training, maintaining a stable **6.1-6.3%** selection rate.
   - Synthetic data results showed an **85.4%** data reduction, achieving **86.0%** accuracy against an **87.6%** baseline.
- **DRPS Increases Training Efficiency**: The **DRPS** system achieved a **16x** reduction in active training set size, enhancing training efficiency.
   - The **Relevance Scorer** improved from **95.9%** to **99.95%** accuracy, and the **Quality Rater** improved from **97.0%** to **100%** accuracy.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1405814767314014238)** (4 messages): 

> `DRPS Framework, Data Efficiency, Selection Intelligence, Synthetic Data Results, Training Efficiency` 


- **DRPS: Data Rankings and Prioritization System Arrives**: The **Data Rankings and Prioritization System (DRPS)** teaches AI to selectively learn from data by using a **Relevance Scorer**, **Quality Rater**, and **Diversity Controller**, as detailed in a [situational awareness report](https://situational-awareness.ai/wp-content/uploads/2024/06/situationalawareness.pdf).
- **DRPS Cuts Data Usage by over 90%**: In tests with **MNIST**, DRPS achieved a **93.8% reduction** in data usage, utilizing only **6.2%** of the examined data while maintaining **99.1%** of baseline performance, showcased in a [GitHub repository](https://github.com/voltageddebunked/drpsStats).
- **DRPS Shows Smarts by Selecting Top Samples**: DRPS examined over **516,000 samples** and selected only **32,000** for training, maintaining a stable selection rate of **6.1-6.3%** throughout the training process.
- **DRPS Boosts Accuracy Points per Data Percentage**: Using synthetic data, DRPS achieved an **85.4% data reduction** and used only **14.6%** of training samples to achieve **5.89 accuracy points** per % of data used, compared to a baseline accuracy of **87.6%**.
- **DRPS Framework improves training efficiency**: DRPS improves training efficiency with a **16x reduction** in active training set size and boosts component accuracy, such as increasing the Relevance Scorer from **95.9%** to **99.95%** and the Quality Rater from **97.0%** to **100%**.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1405632568468045946)** (46 messages🔥): 

> `Quantum Startup Multiverse, MoE Nuances, Tokenization and Routing Synergy, Gemma 3n` 


- **Buzzy Quantum Startup?**: An article about the [startup Multiverse](https://techcrunch.com/2025/08/14/buzzy-ai-startup-multiverse-creates-two-of-the-smallest-high-performing-models-ever/) claims they created *two of the smallest high-performing models ever* using something something quantum, but they are probably just using a **specialized compression algorithm for model weights**.
   - The article does not seem to make actual quantum claims.
- **Deciphering MoE Nuances**: **MoE (Mixture of Experts)** is a family of techniques with very nuanced iterations, including **token-choice**, **expert-choice**, **MoE with capacity factors**, **block sparse dropless token routing versus *droppy* routing**, and more, making it annoying when people attribute a lot of things to MoE for some reason.
   - To verify issues are occurring in batched inference, one might reliably check the behavior numerically of something like **Olmoe** or **IBM Granite 3.1** rather than hitting an API you can't monitor.
- **Synergize Tokenization and Routing**: A member proposed the seemingly obvious idea to do **tokenization and routing in the same step** to synergize them dynamically.
   - Another member responded, *I have never seen that proposed* because the conventional wisdom dictates that networks are more expressive if there's a lot of routing steps right before the expert being activated.
- **Tokenization in Layers**: **Gemma 3n** has per layer tokenization / embedding kind of.
   - That could be a better way to have learned patch level tokenization with inherently a little more insight into the context.


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1405640608181260299)** (1 messages): 

> `DARPA AIxCC, LLM agents` 


- **Team Triumphs in DARPA AIxCC**: A team announced their placement in **DARPA's AIxCC (AI Cyber Challenge)**, where they built an autonomous system of **LLM agents** to find and fix vulnerabilities in open source software.
   - The project is now open source.
- **Tips for Building Kickass LLM Agents**: The team is sharing their tips for building effective **LLM agents** via [this Xitter post](https://x.com/tjbecker_/status/1956081184611688667).
   - The post contains generic advice applicable to a range of agent development scenarios.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1405628482909765652)** (16 messages🔥): 

> `Inference Time on Low-End Devices, DinoV2 vs DinoV1, Gemma Model Parameter Size, China's Role in Automation, Deepseek Training on Huawei Chips` 


- ****Low-End Inference Time Impedes Usability****: Members discussed that inference time is more important on **low-end devices**, citing Google's Android app for running LLMs as an example where long inference times and phone heating make it impractical.
   - Smaller models could be used for keyboard prediction, but these models may require training on device, per [this Youtube video](https://youtu.be/KFYyfrTIPQY?t=2158).
- ****DinoV2 Performance and Training Challenges****: A member expressed hope that a new model would outperform **DinoV2**, as **DinoV2** was worse than **DinoV1** in some contexts and harder to train.
   - They linked to a [YouTube video](https://www.youtube.com/watch?v=eZ2A2045Rkw) as reference.
- ****Gemma's Parameters Revealed****: It was noted that the **Gemma 270M model** has **100M** params and **170M** embedding params.
- ****Deepseek's Chip Choice Stalled Training****: A member pointed out that **Deepseek's training** was stalled by attempting to train on **Huawei chips** instead of **NVIDIA's**, according to [this discussion](https://youtu.be/FQOV-qy9CK4?t=212).
- ****Manufacturing Tariffs Hinder Industry Growth****: A member argued that imposing tariffs on equipment needed to build production lines is counterproductive to encouraging manufacturing.
   - They added that building up an industry would take decades, referencing [Anthropic's research on end-subset conversations](https://www.anthropic.com/research/end-subset-conversations) and [HRM analysis](https://arcprize.org/blog/hrm-analysis).


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/)** (1 messages): 

venom_in_my_veins: hye
  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1405750413764067489)** (4 messages): 

> `1-bit inference, GPTQ` 


- **Explore Speeding Up 1-Bit Inference**: A member inquired about speeding up **1-bit inference** and shared a link to the paper [The Power of $\alpha,1$-sparsity: Near-Lossless Training and Inference of $\alpha$-bit Transformers](https://arxiv.org/html/2411.06360v3).
   - The paper details a novel method to train and infer with **$\alpha$-bit Transformers**, achieving near-lossless results with **1.58 and 1-bit** quantization.
- **Inference Optimization**: The linked paper highlights optimizations for transformer models using **$\alpha,1$-sparsity**, enabling near-lossless training and inference at very low bitwidths.
   - This approach could potentially lead to significant speed improvements in inference for certain applications.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1405632426998239303)** (11 messages🔥): 

> `CUDA Shared Memory, CUDA Illegal Memory Access, CUDA Kernel Launch Configuration, CUDA warp ID calculation` 


- **Debugging CUDA Illegal Memory Access**: A user encountered an *Illegal Memory Access* error when using shared memory in a CUDA kernel and sought help from the community, sharing their code snippet involving `sat` and `commons` arrays.
   - A member suggested that the error might stem from incorrect pointer arithmetic or ill-defined `warp_id` and `WARPS_EACH_BLK`, but provided [an example](https://gist.github.com/alecco/58206ecd6af36c627609e1f464b4b376) code to show that it was probably unrelated.
- **CUDA Kernel Launch Configuration Confusion**: The user shared their kernel launch configuration `<<<BLK_NUMS, BLK_DIM>>>` and macro definitions, with `BLK_NUMS` set to **40**, `BLK_DIM` to **1024**, and `WARPS_EACH_BLK` computed as `BLK_DIM/32`, resulting in a global warp ID calculation.
   - Another member pinpointed the issue: the user's `warp_id` was global, leading to out-of-bounds access to shared memory, which is local to each thread block.
- **Resolving Shared Memory Access Issues**: A member recommended using a local index and warp ID calculation within each thread block, suggesting `local_index = threadIdx.x; local_warp_id = local_index / 32;` to ensure correct shared memory access.
   - They further advised using bitwise shift operations (`local_warp_id = local_index >> 5;`) instead of division and modulus for better performance on the GPU, and checking the generated assembler code with NSight Compute.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1405734478562721915)** (10 messages🔥): 

> `New Grad Kernel Job, GPU Thesis, Getting Kernel Job Without Internship` 


- **Kernel Job Seeker Asks About New Grad Opportunities**: A member inquired whether someone with no internship experience writing kernel could secure a new grad job writing kernel.
   - Another member suggested that if the candidate is knowledgeable about GPUs, their company doesn't prioritize internship experience, mentioning their related [thesis](https://github.com/Snektron/pareas) as part of their own successful interview process.
- **Insider Reveals How to Get Kernel Job Without Internship**: Someone with an interest in GPU posted that they secured a job through a combination of a GPU-related thesis and luck, plus getting through the interview process.
   - According to the person, good knowledge of GPUs can bypass the need for previous experience and internship.


  

---


### **GPU MODE ▷ #[general-leaderboard](https://discord.com/channels/1189498204333543425/1343002580531417211/1405833745772314667)** (1 messages): 

> `MI300 pytorch, OMP missing` 


- ****MI300** env lacks **OMP****: The **MI300** environment appears to be missing **OMP** for `pytorch.compile` based on a user report.
- **Link to Debug Error included**: A user shared a [link to the full debug error](https://github.com/gpu-mode/discord-cluster-manager/actions/runs/16986368251) for further investigation.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1405638909580283936)** (10 messages🔥): 

> `trimul leaderboard, A100, H100, B200` 


- **Trimul Leaderboard Sees New Speedsters**: One member got **second place** on **A100**: **10.4 ms** then quickly got **first place** on **H100**: **3.95 ms** and **first place** on **A100**: **7.53 ms**.
   - Later, the member got **first place** on **B200**: **2.35 ms**, then again **first place** on **A100**: **6.01 ms** and yet again **first place** on **B200**: **2.04 ms** and finally successful on **H100**: **3.74 ms**.
- **A100 and H100 also see activity**: Another member got **5th place** on **A100**: **13.2 ms**.
   - The member followed up to get **second place** on **H100**: **6.42 ms** and finally successful on **A100**: **14.7 ms**.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1405929507554070674)** (10 messages🔥): 

> `Meeting Attendance, Large PR Review, Connection Error Debugging` 


- **Missed Meeting Mishaps**: Several members mentioned missing a meeting due to timezone confusion and scheduling conflicts, including one member available only for the first **10 minutes**.
   - One member quipped that the **8am** meeting time was a bit brutal.
- **Reviewing Scope Creep**: A member commented on a PR with **300 file changes**, joking that it was a "lil out of scope".
   - Another member added that the code was *grass-fed hand-crafted*.
- **Troubleshooting Connection Errors**: A member reported seeing a connection error and is attempting to debug its source, guessing it might be from **db_client**.
   - They mentioned difficulty in getting a stack trace to diagnose the issue.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1405627475521962098)** (47 messages🔥): 

> `Kimi K2 Technical Report, GLM-4.5 vs Kimi K2, Kimi hallucinations, Kimi's Web UI, Kimi future updates` 


- **NotebookLM Video Edges out Kimi PPT**: Members compared a **PPT generated by Kimi** with a **video overview generated by Google's NotebookLM** for the Kimi K2 technical report, with the consensus leaning towards NotebookLM's video due to its audio and more flexible layout (see [attached video](https://cdn.discordapp.com/attachments/1371757564005711973/1405630786308149360/Kimi_K2_10MB_final.mp4?ex=68a0d8ae&is=689f872e&hm=a7541a57850914531af4af61a14c0bfcff5cecb20b2ffe50094bafdb0a8ccde3&)).
   - While both were appreciated, one member expressed a preference for reading over listening to AI-generated audio, but noted the potential of video overviews, especially in education.
- **Kimi K2 Beats GLM in Writing Skills**: Despite a feeling that **GLM-4.5** might surpass **Kimi K2** in overall performance, users lauded **Kimi** for its superior writing style and proactive error detection.
   - One user was *“genuinely surprised”* when **Kimi** *“out of the blue told me No”*, appreciating its candor.
- **Combating Kimi's Hallucinations**: Users expressed a desire for **Kimi** to hallucinate less, even with web search enabled, noting that while **GLM** may take longer, it hallucinates less frequently.
   - A user stated they consistently use the thumbs down button to report hallucinations.
- **Kimi Fans Eagerly Await 'Kimi Thinking'**: Members are eagerly anticipating the arrival of **'Kimi Thinking'** and reasoning and multimodel capabilities.
   - There are questions as to whether this will arrive in the form of **Kimi K-2** or **Kimi K-3**, but there are no firm ETAs yet.
- **Dark Mode Enhances Kimi's Web UI**: A user shared their customized **Kimi Web UI** with a dark mode extension, expressing a strong preference for it over the default grey interface (see [attached screenshot](https://cdn.discordapp.com/attachments/1371757564005711973/1406009945002082374/image.png?ex=68a0e84d&is=689f96cd&hm=4d0b8f1561e558ccf4bd5b6fdf8cfd506b038c5203e9f7632504533cc9ea5ea6&)).
   - Another user confirmed that only the username and server roles are passed to the Moonshot API.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1405648729134076044)** (4 messages): 

> `AI Stock Portfolio Agent, Web Scraping AI Agents, Multimodal AI Applications, Legal Knowledge Graphs` 


- **AI Stock Portfolio Agent Arrives**: LlamaIndex has introduced a framework to build a complete **AI stock portfolio agent**, integrated with [@CopilotKit](https://www.copilotkit.ai/)'s AG-UI protocol for seamless frontend-backend communication; a comprehensive tutorial is included to create a sophisticated investment analysis tool.
   - The tutorial combines the power of [this framework](https://t.co/fQDNPIQoqR) to create a sophisticated investment analysis tool.
- **Brightdata and LlamaIndex Launch Web Scraping AI Agents**: LlamaIndex announced a new walkthrough with [@brightdata](https://www.brightdata.com/) on building **web-scraping AI agents** with LlamaIndex's agentic framework, focusing on reliable web access and robust web scraping workflows.
   - The walkthrough details how to set up workflows that can handle dynamic content and build **intelligent agents** that can navigate [here](https://t.co/IBgSLBM6XW).
- **Multimodal AI Apps Analyze Markets Visually**: LlamaIndex announced building **multimodal AI applications** that analyze both text and images for market research and surveys.
   - These applications are designed to process images and documents together in a unified AI pipeline, extract insights from visual market data like charts, graphs, and product images, and combine multimodal [capabilities](https://t.co/fOMFLXWarG).
- **LlamaCloud and Neo4j Transform Legal Documents into Knowledge Graphs**: LlamaIndex announced a comprehensive tutorial on transforming unstructured legal documents into **queryable knowledge graphs** that understand not just content, but relationships between entities.
   - This workflow leverages **LlamaCloud** and [@neo4j](https://neo4j.com/) for legal contract analysis and is detailed [here](https://t.co/MPSfPiS2Cv).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1405664216601329764)** (28 messages🔥): 

> `Pydantic Models vs JSON Schema for Tool Calls, Vector Store Errors After Update, Progress Bar Issue with num_workers > 1, Iterating Over Nodes/Doc_IDs in Vectorstore` 


- **Pydantic vs JSON Schema Showdown**: A member inquired whether tool calls require a **Pydantic model** or if a **JSON schema** suffices, noting the redundancy of converting JSON to a Pydantic model only to have it unpacked back into JSON.
   - Another member pointed out **Pydantic's** `create_model()` function doesn't directly accept a **JSON schema**, highlighting the need for a tool/package to handle this conversion.
- **Vector Store Gets Attribute Error After LlamaIndex Update**: After updating to version **0.13.1**, a user encountered an `AttributeError` during retrieval from a **PGVectorStore** when using `RetrieverQueryEngine` with `OpenAI` and `text-embedding-3-small`.
   - The error arises because the `output` is a `str` with no attribute `json`, stemming from the **LLMStructuredPredictEndEvent** in `openinference.instrumentation.llama_index`.
- **Progress Bar Pandemonium with Multiprocessing**: A user highlighted that the `progress_bar=True` feature doesn't function correctly when `num_workers > 1` due to the use of **multiprocessing**.
   - It was suggested that using **async concurrency** could offer a smoother experience, however the `async pipeline.arun` method still uses multiprocessing.
- **Nodes and Doc IDs Missing in Action in Vector Stores**: A user expressed frustration over the inability to iterate over nodes or obtain a list of `doc_ids` in most LlamaIndex vector stores, particularly noting the absence in **Opensearch** and **awsdocdb**.
   - A workaround involves setting `similarity_top_k` to a high number, but this is inefficient and may not be supported by all OSS; the `get_nodes()` method exists on the base `vector_store` class, however, it is not implemented for Opensearch or awsdocdb, which is an opportunity for a PR.


  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1405905432920326265)** (1 messages): 

> `DSPy optimizes CrewAI, CrewAI agent prompts` 


- **DSPy optimizes CrewAI agent prompts**: A course teaches how **DSPy optimizes CrewAI** agent prompts in a real production use case to build smarter, cheaper agents with proven methods.
   - You can check the course [here](https://www.udemy.com/course/crewai-dspy-optimization/?referralCode=B59F73AE488715913E7E).
- **Build smarter, cheaper agents with proven methods**: The course focuses on **DSPy optimization** for CrewAI agents.
   - It emphasizes building more efficient and intelligent agents through **proven methodologies**.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1405744293439733830)** (7 messages): 

> `Audio Transcription in NotebookLM, NotebookLM Interface Redesign` 


- **Audio Uploads Auto-Transcribed to NotebookLM**: A member inquired about obtaining audio transcripts, to which another member responded that they upload **MP3 audio files directly to NotebookLM**.
   - The member clarified that **NotebookLM** itself handles the transcript generation.
- **NotebookLM Interface Redesign Underway**: One member mentioned they are attempting to redesign **NotebookLM**, and shared a Figma screenshot of the proposed changes.
   - The member apologized for any disappointment, clarifying it was just a design concept, not a functional update.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1405718164716650520)** (23 messages🔥): 

> `Explainer Video Voice, Feature Request Feedback, Dev Interaction, Prompt Limit` 


- **Voice Gender Swaps Explainers**: A user reported that their explainer videos suddenly started generating with **male voices** instead of the usual **female voices** and questioned why this happened.
   - There was no clear resolution or explanation provided in the messages.
- **User Requests Acknowledgement of Feature Requests**: A user questioned whether anyone from the **NotebookLM development team** is actually reading through the **feature requests** posted in the Discord channel.
   - They expressed a desire for some sign of life or feedback from the developers to encourage continued contributions.
- **NotebookLM Devs Acknowledge Reading Posts but Can't Respond to Everything**: A Google developer stated that *the devs read the posts*, but they don't have time to respond to everything and spend a lot of their time **banning spammers**.
   - Other users suggested that even occasional acknowledgements or AI-compiled summaries could help encourage user contributions.
- **Users bump into Prompt Limits in NotebookLM**: A user asked if there is a limit to the **number of words** that can be included in a single question in **NotebookLM** after failing to ask a case-related question containing about **857 words**.
   - Another user suggested splitting the prompt into multiple parts or trying **Gemini**.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1405902903151169648)** (1 messages): 

> `CrewAI agent prompts, DSPy` 


- **Optimize CrewAI agent prompts with DSPy**: Members shared a link to learn how **DSPy optimizes CrewAI agent prompts** in a real production use case: [https://www.udemy.com/course/crewai-dspy-optimization/?referralCode=B59F73AE488715913E7E](https://www.udemy.com/course/crewai-dspy-optimization/?referralCode=B59F73AE488715913E7E).
   - The course claims to teach users how to *build smarter, cheaper agents with proven methods*.
- **DSPy and CrewAI unite**: The course teaches users how to optimize CrewAI using DSPy.
   - It enables smarter, cheaper agents using proven methods.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1405627855324315649)** (22 messages🔥): 

> `DSPy and Databricks, GEPA Error, MLflow and DSPy` 


- **Databricks Not Sponsoring DSPy**: A user asked if **Databricks** sponsors or owns the **DSPy** project, and another user clarified that DSPy is **MIT-licensed open source**, with Databricks contributing significantly through a team of core developers.
- **GEPA Bug Fixed**: A user encountered a `ValueError` when using **GEPA** with the **RAG tutorial**, and another user confirmed that [this was a bug in GEPA code](https://github.com/stanfordnlp/dspy/pull/8647) that has been fixed; users should upgrade to **DSPy 3.0.1**.
   - The depreciated param is in that dspy.evaluate import, and the fix is `pip install -U dspy`.
- **MLflow Tracks DSPy Sub-Modules automatically**: A user inquired about integrating **DSPy modules** tracking with **MLflow** for a **text2sql pipeline** and was advised to use `mlflow.dspy.autolog()` instead of `mlflow.autolog()` to automatically track all sub-modules.
   - Using `mlflow.dspy.autolog()` will display the **SQLGenerator**, **Validator**, and **Reflector** as nested spans in the **MLflow UI's Traces tab**, as detailed in the [MLflow DSPy integration documentation](https://github.com/mlflow/mlflow/blob/master/docs/docs/genai/tracing/integrations/listing/dspy.mdx) and the [DSPy MLflow tutorial](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/tutorials/optimizer_tracking/index.md).
- **Logprob Surprise as fitness Function**: A user shared a tweet [TogetherCompute Status](https://x.com/togethercompute/status/1956416013404406018), and guessed that they’re basically doing **GEPA** with **logprob surprise** as the **fitness function**, but for mental health models in prod.
- **Community Engagement Requested**: A member requested more engagement from the 6500 people in this discord, and more contributions to the docs and all.


  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1405897484248813679)** (1 messages): 

> `CrewAI, DSPy Optimization, Prompt Engineering` 


- **CrewAI Prompt Optimization Course Drops**: A member announced a [Udemy course](https://www.udemy.com/course/crewai-dspy-optimization/?referralCode=B59F73AE488715913E7E) demonstrating how to optimize **CrewAI prompts** with **DSPy**.
   - The course will show how to inject them back to the **LLM** so the **LLM** uses better prompts than those stitched together by **CrewAI**.
- **DSPy Enables Optimized CrewAI Prompts**: The new course uses **DSPy** to optimize prompts.
   - Optimized prompts are then injected back into the **LLM** improving on the standard approach in **CrewAI**.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1405629920868171879)** (8 messages🔥): 

> `CI speed, tinygrad release, tinygrad size` 


- **CI Speed Hampering Productivity**: A member expressed frustration with slow CI speeds, stating they could work faster with quicker CI and linked [chatgpt analysis](https://chatgpt.com/share/689e3508-5f68-8000-97b2-aa6f1699aa74).
- **Tinygrad Release Imminent**: There was a suggestion to do a **tinygrad release** soon.
- **Tinygrad Size Swells**: A member questioned why **tinygrad 0.10.3** is **10.4 MB**, hinting at a possible size issue.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1405802310633787423)** (14 messages🔥): 

> `WSL2 Support, print_tree removal` 


- **WSL2 Tinygrad Bug Surfaces**: A user encountered an issue where adding two tinygrad Tensors created from PyTorch tensors resulted in all **0s**, and provided a [full script](https://cdn.discordapp.com/attachments/1070745817025106080/1405817973004046387/message.txt?ex=68a0de43&is=689f8cc3&hm=be6e6069e1975cc70d7fbda2a0b20849f96396efd36e0b905118668100b11656) to reproduce the bug on WSL2.
- **print_tree function bites the dust**: The `print_tree` function has been replaced with a simple `print` function.
   - The user noted it *lost some of its formatting*.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1405710824835780628)** (12 messages🔥): 

> `Aider Benchmark, litellm Errors, Open Source Entitlement, llama.cpp PR #15181` 


- **Aider Benchmark Plagued by Timeouts**: A member ran the **Aider benchmark** against a local **gemma3:12b** model and encountered frequent timeouts after **10.5 hours** and **221/225 tests** due to the model's inability to respond within the **600-second** limit, resulting in *litellm.APIConnectionError* errors.
   - They shared the error log which shows the model attempting to send around **300k tokens**, exceeding the **131,072 token limit** and causing test failures.
- **Continuing Aider Benchmark**: A member suggested using `ctrl+c` to exit the benchmark, restarting the inference server, and then using the `--cont` flag to resume the benchmark from where it left off.
   - They also pointed to a [merged pull request](https://github.com/ggml-org/llama.cpp/pull/15181) in *llama.cpp* that might improve local model performance.
- **OSS Maintainer's Burden**: A member criticized another's suggestion to make the benchmark automatically configurable for each LLM, labeling it as *entitlement* and lamenting that such attitudes cause *countless OSS maintainers to throw in the towel*.
   - Another member countered that it was merely *curiosity*, leading to further disagreement on what constitutes entitlement in open-source interactions.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1405695906635845682)** (7 messages): 

> `Aider with Local Models, Aider Line Number Accuracy, Unit Test Coverage with Aider` 


- **Local AI/Aider Models Bring Debugging Agony**: A member expressed difficulty using **aider** with local models like **ollama**, **lmstudio**, and **vllm**, noting slow performance even with powerful hardware.
   - They suggested the need for a tutorial video on setting up **aider** with these tools for local development and debugging.
- **Aider's Line Numbering System Questioned**: A member inquired about how **aider** determines line numbers, especially in the context of generating unit tests for specific code coverage.
   - The issue arises when **aider** misreports the line numbers, leading to incorrect test coverage, despite attempts to refresh the map and clear chat history.
- **LLM Accuracy Impacts Unit Test Coverage**: A member reported that **qwen3-coder** and **gemini-pro** inaccurately identify line numbers in coverage reports, sometimes missing the coverage entirely.
   - This inconsistency leads to questions about whether **aider** relies on the **LLM's accuracy** for line number identification, suggesting a need to explore alternative methods for accurate unit test generation.


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1405881855823188060)** (3 messages): 

> `Grok4, Quota Increase, Benchmark Costs` 


- **Grok4 Location Remains Elusive**: A member inquired about the whereabouts of **Grok4**.
   - Another member responded that *it's in the article* but the request to increase the **quota** needed to execute the tests was ignored.
- **Grok4 Benchmark Costs Thousands**: A member noted they *spend multiple thousands dollars during the development of this benchmark*.
   - This highlights the significant financial resources required for advanced AI model benchmarking.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1405736806930055170)** (22 messages🔥): 

> `Manus Credit Deductions on Errors, Manus Deployment Issues, Manus Team Accounts, Add-on Credits Removal, Manus in the Wild Challenge Winner` 


- **Manus Credit Deductions Draw Ire**: Users express frustration over **Manus** deducting credits even when it makes errors, making task completion difficult compared to other AIs like **Claude AI**.
   - One user reported *spending high amounts of credits* only for **Manus** to make a simple change that broke the entire application, deeming it non-functional.
- **Manus Deployment Stumbles**: Users report issues with **Manus** deployments, where websites created from the same **GitHub** repository differ significantly, especially with large folders, illustrated by comparison of [affilify.eu](https://affilify.eu) and a **Manus** hosted site [wmhkgqkf.manus.space](https://wmhkgqkf.manus.space).
   - A community manager noted that **Manus** isn't designed as a coding agent or pure development tool, so deployment isn't its strength, but they're working on improvements.
- **Add-on Credit Packages Recede**: Users question the removal of add-on credit packages, which are now exclusively available for **Pro** users.
   - A community manager explained that this change ensures consistent speed and quality for heavy users and suggested bundling similar questions, being concise, and avoiding repeated requests to maximize credit efficiency.
- **Manus Team Accounts Spark Interest**: A user inquired about the possibility of a **Manus** team account for shared credit usage.
   - A community manager confirmed that **Manus** does offer a team plan, directing users to the [official website](https://manus.ai) for details.
- **Users Bemoan Credit Consumption**: One user shared a frustrating experience of burning through **30,000 credits** attempting to get their website up, encountering issues with mock sites and template implementations.
   - They criticized the system's inconsistency, where it's *smart as hell but then suddenly turns dumb*, leading to wasted credits and suspected stall tactics.


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1405855716916461669)** (9 messages🔥): 

> `Cohere Labs, Pokemon emojis, PAX Omeganauts Discord` 


- **Cohere Labs contact info sought!**: A member asked where to connect with **Cohere Labs** folks, another member suggested this Discord channel.
   - Another member directed the user to [this link](https://discord.com/channels/954421988141711382/954421988783444043/1400866387668504648).
- **Discord channel Pokemon Emoji-fied!**: A member suggested adding more **Pokemon emojis** to the channel, as there are available slots.
   - The member noted that the emojis come from the **PAX Omeganauts Discord**.


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1405640013198131345)** (5 messages): 

> `AI Research, writenode, CV+ML pipeline` 


- **AI Researcher Seeks Collabs**: An **AI researcher** with a deep interest in **reasoning and conscious capabilities** is looking for collaborations to develop advanced tech for the future.
   - The member is open to collaborations from any sub domain.
- **Legal Pro Aligns with AI**: A **legal professional**, gamer, and lover of philosophy currently working for the USG is self-teaching **AI alignment theory and mechanics**.
   - The member is excited to be here.
- **writenode builder uses Cohere**: Josh is building **writenode**, *an in browser, cognitive thought partner, and creative companion*, and uses **Cohere**.
   - He does not have a developer or coding background since before December last year.
- **Psych PhD Returns to AI**: A member is returning to **AI research** after dabbling in a human psychology PhD the past 5 years.
   - Their interests are in **sound+music**, and using tech tools to help us express our creativity.


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1405985104920055959)** (3 messages): 

> `Discord Invite Links, Channel Spam` 


- **Discord Invites Flood Channel**: A member spammed the channel with [Discord invite links](https://discordapp.com/invite/HjWfRbqBB8) multiple times, tagging *everyone*.
   - The invite link was repeated three times in quick succession.
- **Invite Link Repetition**: The same [Discord invite link](https://discordapp.com/invite/HjWfRbqBB8) was posted repeatedly.
   - This resulted in a spam-like effect, potentially disrupting the channel's usual discussions.


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1405984973906903060)** (3 messages): 

> `Discord Invite Link, HjWfRbqBB8, Channel Invitation` 


- **Discord Invite Link Floods Channel**: A member repeatedly shared a [Discord invite link](discordapp.com/invite/HjWfRbqBB8) in the channel, possibly to attract more users.
   - The member tagged `@everyone` multiple times, which might be considered excessive or disruptive.
- **Channel Invitation Blitz**: The repeated posting of the [same Discord invite](discordapp.com/invite/HjWfRbqBB8) suggests an attempt to boost channel membership.
   - The use of `@everyone` indicates the message was intended for all members, regardless of their interest in the invitation.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1405660404918652948)** (2 messages): 

> `Elicitations Specification, MCP Server Conversion` 


- **Elicitations Spec Clarity Sought**: A member inquired about the [Elicitations specification](https://modelcontextprotocol.io/specification/2025-06-18/client/elicitation) regarding *who* is responsible for translating message/field descriptions into the user's language.
   - Specifically, they seek clarification on whether **tools** should handle language detection and internationalization, or if **MCP Clients** are expected to translate, potentially using an LLM.
- **MCP Server Transformation Question**: A member inquired whether *there exists some tool to turn a local mcp server into a remote mcp server?*
   - No links or additional context was provided.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1405750824461668434)** (3 messages): 

> `Unifi MCP, Unraid MCP, Syslog MCP, AI Agent Workflows, AI Security` 


- **MCP Servers for Homelabbers Arrive**: A member shared a few MCP (presumably, **Management Control Panel**) servers for the homelabbers, specifically: [Unifi MCP](https://github.com/jmagar/unifi-mcp), [Unraid MCP](https://github.com/jmagar/unraid-mcp), and [Syslog MCP](https://github.com/jmagar/syslog-mcp).
- **PulseMCP Turns Newsletter Tedium to Agent Automation**: **PulseMCP** used goose to turn a tedious newsletter workflow into agent-powered automation with a human in the loop.
   - More details on the automation can be found at [this blogpost](https://block.github.io/goose/blog/2025/08/13/pulse-mcp-automates-recipe).
- **AI Security Seeks Input On Security Concerns**: One member posted about building **AI security** that stops attacks before they even start with mathematical security certainty.
   - They are looking for Dev input on security concerns, and linked to [a survey](https://form.typeform.com/to/xTKa05F9).


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1405631570194337793)** (4 messages): 

> `Strix Halo profitablility, Dolphin chat template, Quantum computers, PC Memory` 


- **Strix Halo's Profitability Plummets**: The **Strix Halo**, despite its impressive specs, requires a **year of 24/7 inference** to become profitable due to its slower inference speed of **53 tokens/sec** compared to **GPT-OSS 120B** on **OpenRouter**.
   - One user noted that configuring it for **LLMs** at $2000 is inefficient compared to cloud alternatives offering **200-400 tokens/sec**.
- **Dolphin Chat Template Quest**: A user is seeking a working chat template for **gpt4all** compatible with **Dolphin-2.2.1-mistral-7b-gptq**.
   - Another member suggested asking model makers to upload a template with a **jinja** template.
- **Quantum Computing Teaspoons?**: One user speculates on the future availability of quantum computers and the possibility of selling **qubits on the teaspoon**.
   - They mentioned news about **fully working quantum computers**, indicating potential advancements in the field.
- **Memory Modules and Moore's Law**: A user mentioned that old-fashioned PCs can expect to see **higher capacity memory modules** and **DDR6** in late 2027 or 2028.
   - They express excitement about the potential of micro PCs with high RAM and VRAM capacities, especially for small businesses.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1406014763804397599)** (1 messages): 

> `Maternity Leave, Team Contact During Leave` 


- **Maternity Leave Commences!**: A member announced they will be on **maternity leave** from **August 25th** until **February 2026**.
   - They look forward to catching up upon their return.
- **Team's Coverage Plan Revealed**: While they are away, the team will be monitoring <@1334161614949056532>.
   - Members can also reach out to <@709918328306663424> with any questions or concerns.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/)** (1 messages): 

__nathan: <@132818429022437376> how did this go?
  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1405634992792670208)** (1 messages): 

> `Windsurf Wave 12, DeepWiki Integration, Vibe and Replace feature, Smarter Cascade Agent, Dev Containers Support` 


- **Windsurf Wave 12 is Released!**: Windsurf Wave 12 brings the first integrations of **Devin's intelligence** and capabilities directly into the Windsurf IDE.
   - Key features include a **new UI design**, **DeepWiki Integration**, **Vibe and Replace**, a **Smarter Cascade Agent**, **Faster Tab**, **Dev Containers Support**, and **over 100 bug fixes** - [see the changelog](https://windsurf.com/changelog), [read the blog](https://windsurf.com/blog/windsurf-wave-12), [watch the Wave 12 video](https://www.youtube.com/watch?v=-7gm8mST9QU), [X/Twitter](https://x.com/windsurf/status/1956074019393876280), and [Reddit](https://www.reddit.com/r/windsurf/comments/1mqal3x/wave_12_released_fresh_ui_deepwiki_vibe_and/).
- **DeepWiki Integration brings AI to the IDE**: **DeepWiki Integration** allows users to hover over code symbols for **AI-powered explanations** (not just basic type info).
   - Users can also use **CMD/Ctrl+Shift+Click** to open detailed explanations in the side panel and add to Cascade context.
- **Vibe and Replace revolutionizes bulk editing**: The **Vibe and Replace** feature provides revolutionary bulk editing capabilities by finding exact text matches.
   - It allows users to apply **AI prompts** for intelligent, context-aware transformations across their entire project.
- **Smarter Cascade Agent gets Always-On Planning**: The **Smarter Cascade Agent** now features an always-on planning mode with autonomous to-do lists.
   - It also includes revamped tools designed to provide smarter responses.
- **Dev Containers Supported Natively**: Windsurf now supports working with containers directly via remote SSH access.
   - This enhancement simplifies development workflows involving containerized environments.

