---
id: MjAyNS0x
title: not much happened today
date: '2025-10-14T05:44:39.731046Z'
description: >-
  **Alibaba** released compact dense **Qwen3-VL** models at 4B and 8B sizes with
  FP8 options, supporting up to 1M context and open vocabulary detection,
  rivaling larger models like **Qwen2.5-VL-72B**. Ecosystem support includes
  **MLX-VLM**, **LM Studio**, **vLLM**, **Kaggle models**, and **Ollama Cloud**.
  In video AI, **Arena** added **Sora 2** models leading in video benchmarks,
  with **Higgsfield Enhancer** improving video quality. **Runway** launched
  domain-specific workflow apps for creative tasks. Research on **Representation
  Autoencoders for DiTs (RAE-DiT)** shows improved diffusion model performance.
  On local training, **NVIDIA DGX Spark** enables strong local fine-tuning,
  while **Nanochat** by **Karpathy** offers a minimal stack for training and
  inference. **Together AI** introduced **ATLAS**, a speculative decoding method
  achieving up to 4× faster inference on **DeepSeek-V3.1**. These developments
  highlight advances in efficient model deployment, video AI, local fine-tuning,
  and inference speed optimization.
companies:
  - alibaba
  - arena
  - runway
  - nvidia
  - togethercompute
  - ollama
models:
  - qwen3-vl-4b
  - qwen3-vl-8b
  - qwen2.5-vl-72b
  - deepseek-v3.1
topics:
  - model-optimization
  - fine-tuning
  - inference-speed
  - video-generation
  - diffusion-models
  - representation-learning
  - local-ai
  - speculative-decoding
  - fp8-quantization
  - context-windows
people:
  - karpathy
---


**a quiet day**

> AI News for 10/13/2025-10/14/2025. We checked 12 subreddits, 544 Twitters and 23 Discords (197 channels, and 6882 messages) for you. Estimated reading time saved (at 200wpm): 510 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

a quiet day.

---

# AI Twitter Recap

**Alibaba’s Qwen3‑VL Dense Models (4B/8B) and Rapid Ecosystem Support**

- **Qwen3‑VL 4B/8B (Dense, Instruct + Thinking)**: Alibaba released compact dense Qwen3‑VL models at 4B and 8B, each in Instruct and Thinking variants, with FP8 options for efficient deployment. They retain full Qwen3‑VL capabilities, advertise strong performance across STEM, VQA/OCR, video understanding, and agent tasks, and often beat Gemini 2.5 Flash Lite and GPT‑5 Nano; in many cases they rival the much larger Qwen2.5‑VL‑72B from six months ago. They support 256K context, expandable to 1M, and “open vocabulary” detection. Apache‑2.0 license. Announcements and cookbooks: [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1978150959621734624), [cookbooks](https://twitter.com/mervenoyann/status/1978082840337060118), [follow‑ups](https://twitter.com/mervenoyann/status/1978153606462550220).
    
    Ecosystem: day‑0 support in MLX‑VLM and LM Studio ([@Prince_Canuma](https://twitter.com/Prince_Canuma/status/1978164715848134699), [@lmstudio](https://twitter.com/lmstudio/status/1978205419802616188)), vLLM ([@rogerw0108](https://twitter.com/rogerw0108/status/1978158856611024913)), Kaggle models ([@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1978290751436943415)), and Ollama Cloud for the 235B variant ([@ollama](https://twitter.com/ollama/status/1978225292784062817), [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1978288558587674672)). Early users highlight speed and structured JSON output quality ([@andrejusb](https://twitter.com/andrejusb/status/1978076341158244835), [@simonw](https://twitter.com/simonw/status/1978151711987372227)).
    

**Video Models and Creative Tools**

- **Arena adds Sora 2**: Sora 2 Pro ties Veo 3 variants for #1 on the Video Arena; Sora 2 ranks #3 and is noted for synchronized audio. Competition in text‑to‑video is accelerating ([@arena](https://twitter.com/arena/status/1978149396996051007)). In the wild: **Higgsfield Enhancer** removes Sora‑style flicker and ships “Sora 2 MAX” upscalers ([@higgsfield_ai](https://twitter.com/higgsfield_ai/status/1978231305394663506)).
- **Runway Apps**: Runway introduced “Apps,” domain‑specific workflows (product reshoots, image restyling, etc.) rolling out across web and iOS, emphasizing reusable, professional pipelines ([@runwayml](https://twitter.com/runwayml/status/1978094115142225968), [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1978109936149094800)).
- **Research: Representation Autoencoders for DiTs**: RAE‑DiT replaces VAEs with pretrained representation encoders (DINO, SigLIP, MAE) plus trained decoders, achieving ImageNet FID 1.51 @256 (no guidance) and 1.13 @256/512 (with guidance). Highlights a trend to decouple representation learning from reconstruction in diffusion pipelines ([@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1978053094769615296), [commentary](https://twitter.com/sedielem/status/1978143596701249733)).

**Local Training and Inference: DGX Spark, Nanochat, and Inference Speculation**

- **NVIDIA DGX Spark, desk‑side fine‑tuning**: Early users report DGX Spark easily runs strong LMs (e.g., Qwen3 Coder) locally, with llama.cpp perf posted and public write‑ups from academic labs. The general sentiment: more builders fine‑tuning at home/office as local compute matures ([@gneubig](https://twitter.com/gneubig/status/1978067258506187238), [@ggerganov](https://twitter.com/ggerganov/status/1978106631884828843), [@kchonyc](https://twitter.com/kchonyc/status/1978156587320803734), [@gdb](https://twitter.com/gdb/status/1978273142695977391)).
- **Nanochat (Karpathy)**: A minimal end‑to‑end stack (~8K LOC) for pretrain → mid‑train → SFT → RL → inference + ChatGPT‑like UI; a 560M model trains in ~4 hours on 8×H100. Community groups, Colabs, and SkyPilot templates emerged within a day; teams are scaling recipes and exploring best SFT/RL splits ([@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1978144157970661495), [community](https://twitter.com/ben_burtenshaw/status/1978062142709326053), [SkyPilot](https://twitter.com/skypilot_org/status/1978273387903410412)).
- **Speculative decoding at scale**: Together AI introduced ATLAS, a learning speculator yielding up to 4× faster inference than baseline (and ~2× vs their Turbo speculator), hitting 500 TPS on DeepSeek‑V3.1 ([@togethercompute](https://twitter.com/togethercompute/status/1978210662095475097)).
- **Memory–compute trade‑offs for reasoning**: From 1,700 experiments on Qwen3 (0.6B–32B, 4/8/16‑bit, token budgets, Maj@K, KV eviction/quantization), the “optimal” memory allocation flips around the “8‑bit 4B effective size.” For math tasks, avoid 4‑bit; prefer precision and longer generations for larger models; Maj@K helps when you’re ≥8‑bit 4B; KV eviction vs quantization depends on scale ([@DimitrisPapail](https://twitter.com/DimitrisPapail/status/1978108550854382052)).
- **RL training at lower cost**: QeRL (NVLabs) combines NVFP4 quantization + LoRA to enable RL training of a 32B LLM on a single H100 80GB; code and paper released ([@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1978046212621373719), [repo](https://twitter.com/yukangchen_/status/1978146373745639894)).
- **Second‑order optimization**: New full second‑order optimizer reports ~5× iteration reduction over SOAP and ~15× over Muon in LLM training experiments ([@ShamKakade6](https://twitter.com/ShamKakade6/status/1978147672105353543)).
- Bonus: Python 3.14 will let you disable the GIL, enabling true multi‑threaded speedups; uv already supports it ([@_avichawla](https://twitter.com/_avichawla/status/1977985594103140710)).

**Agents, Tool Use, and RL**

- **Claude Code and sub‑agent orchestration**: Multiple reports that orchestrator + specialized sub‑agents (coders, searchers, verifiers) dramatically improves planning and codebase tasks, outperforming monolithic “deep research” agents. Anthropic is rolling out Claude deeper into Salesforce Agentforce, Slack, and Claude Code across Salesforce engineering ([@omarsar0](https://twitter.com/omarsar0/status/1978235329237668214), [@AnthropicAI](https://twitter.com/AnthropicAI/status/1978125047270154567)). Claude app also shows notable depth with Gmail/Calendar ([@emollick](https://twitter.com/emollick/status/1978101986357662156)).
- **Why RL works for agentic reasoning**: A synthesis: real, diverse data + pragmatic RL tweaks (e.g., GRPO‑TCR) beats fancy algorithms and scale; small models (4B) can surpass 14B–32B on AIME25 and GPQA‑D with the right recipe; long‑CoT models need tool‑heavy finetuning to become effective agents ([thread](https://twitter.com/omarsar0/status/1978112328974692692), [paper](https://twitter.com/omarsar0/status/1978112412743258361)). Complementary safety work: **WaltzRL** frames helpfulness/safety as a positive‑sum multi‑agent game to reduce over‑refusals without capability loss ([@jaseweston](https://twitter.com/jaseweston/status/1978185306999341256)).
- **Operationalizing agents**: Practical posts on agent authN/authZ (OAuth2/OIDC across Auth Code/OBO/Client Credentials) from LangChain ([@LangChainAI](https://twitter.com/LangChainAI/status/1978121116867567644)), agentic MCP configuration and schema discipline ([@tadasayy](https://twitter.com/tadasayy/status/1978170863192346660)), and orchestrating microservices with LlamaIndex Workflows + Docker + Kafka ([@llama_index](https://twitter.com/llama_index/status/1978137596900593667)).
- Related: LRMs can be brittle when interrupted or with dynamic context (performance drops up to 60%), highlighting the gap between static and real‑world evals ([@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1978044847216095361)).

**Search, Retrieval, and Data Tools**

- **OpenAI Search API update**: New GPT‑5‑powered web search in Chat Completions: gpt‑5‑search‑api is $10/1K calls (60% cheaper), includes domain filters, and is aligned with the new Responses web search behavior ([@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1978224165997195559), [early sighting](https://twitter.com/testingcatalog/status/1978153397552374168)).
- **Perplexity as a default search engine in Firefox**: Perplexity is now built‑in as a default search option for Firefox users ([@perplexity_ai](https://twitter.com/perplexity_ai/status/1978114334741168298), [@AravSrinivas](https://twitter.com/AravSrinivas/status/1978122437427425481)).
- **Compound retrieval > simple retrievers on complex queries**: Weaviate’s Query Agent “Search Mode” outperforms hybrid search on BRIGHT (level‑3 retrieval requiring reasoning); they also detailed multi‑tenancy primitives (one shard/tenant, lazy loading, tenant states) for SaaS‑scale workloads ([@CShorten30](https://twitter.com/CShorten30/status/1978107101936230745), [@weaviate_io](https://twitter.com/weaviate_io/status/1978112245436453044)).
- **Vector infra at scale**: TurboPuffer reports 100B vector search at p99=200ms (unfiltered, 1024D, k=10, 92% recall) on ANN v3 beta ([@turbopuffer](https://twitter.com/turbopuffer/status/1978173877571441135)).
- **OCR and robotics datasets**: Nanonets released a new SoTA OCR model with LaTeX, multilingual, complex table support (works with transformers/vLLM) ([@reach_vb](https://twitter.com/reach_vb/status/1978061301399052485)); LeRobot added CLI tools for editing robotics datasets (split/merge, add/remove features, delete episodes) ([@LeRobotHF](https://twitter.com/LeRobotHF/status/1978126569055887421)).

**Policy, Product, and Platform Notes**

- **Together AI’s scale**: The Information reports Together AI doubled to $300M ARR over the summer; expanding to buying GPUs for its own DCs ([reporter](https://twitter.com/steph_palazzolo/status/1978099327634473072)).
- **Anthropic + Salesforce**: Claude is now a preferred model in Agentforce for regulated industries, deeper Slack integration, and Claude Code adoption across Salesforce engineering ([@AnthropicAI](https://twitter.com/AnthropicAI/status/1978125047270154567)).
- **OpenAI platform/personality**: OpenAI plans to relax ChatGPT restrictions to allow more “4o‑style” personality when desired; age‑gated erotica for verified adults in December ([@sama](https://twitter.com/sama/status/1978129344598827128), [follow‑up](https://twitter.com/sama/status/1978143827144700214)).
- **Google AI Studio refresh**: New homepage and messaging to “build now” ([@GoogleAIStudio](https://twitter.com/GoogleAIStudio/status/1978138461514166742), [@osanseviero](https://twitter.com/osanseviero/status/1978142950098903472)).
- **Security for AI systems**: Google writes on defense‑in‑depth for Gemini; broader discussion on agent authZ/authN and AI control with trusted monitors highlights real production considerations ([@googlepubpolicy](https://twitter.com/googlepubpolicy/status/1978163414498185578), [@jonasgeiping](https://twitter.com/jonasgeiping/status/1978182050730344862)).

**Top tweets (by engagement)**

- **“Just type ‘add a girlfriend’ to any video”**: Grok Imagine tease from [@elonmusk](https://twitter.com/elonmusk/status/1977982448861381081).
- **OpenAI product direction**: Personality settings returning to ChatGPT, with broader adult options behind age‑gating in December ([@sama](https://twitter.com/sama/status/1978129344598827128)).
- **Figure’s new website**: Strong interest in humanoid robotics brand/design refresh ([@adcock_brett](https://twitter.com/adcock_brett/status/1978124226742944193)).
- **Perplexity x Firefox**: Perplexity becomes a default Firefox search option ([@perplexity_ai](https://twitter.com/perplexity_ai/status/1978114334741168298)).
- **Walmart instant checkout in ChatGPT**: Embedded commerce flows inside ChatGPT drive attention ([@bradlightcap](https://twitter.com/bradlightcap/status/1978116720171643127), [@gdb](https://twitter.com/gdb/status/1978123494870196228)).
- **Sora 2 flicker fix**: Higgsfield Enhancer removes flicker and adds upscale variants ([@higgsfield_ai](https://twitter.com/higgsfield_ai/status/1978231305394663506)).
- **“Paradigm shift” to open/local training**: Surge in small/specialized, open‑source models, and desk‑side compute like DGX Spark ([@ClementDelangue](https://twitter.com/ClementDelangue/status/1978113358772449379)).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Local-Only AI Ownership Slogan

- [**If it's not local, it's not yours.**](https://www.reddit.com/r/LocalLLaMA/comments/1o6ocfs/if_its_not_local_its_not_yours/) (Activity: 1035): **Meme image promotes a local-first AI stance: “If it’s not local, it’s not yours,” riffing on crypto’s custody mantra to imply “Not your VRAM, not your model.” Technically, the thread frames on‑prem/single‑server SLMs as preferable for privacy/compliance, predictable latency, offline reliability, and insulation from vendor policy changes, API outages, or deprecations that can break workflows or revoke capabilities.** Commenters advocate local/on‑prem deployment (“On prem SLM can do wonders for specific tasks”) and cite the AI Dungeon episode—policy changes on OpenAI’s API degrading functionality—as a cautionary tale against cloud dependence (“Fool me once…”). There’s debate around hardware custody (VRAM) equating to control, versus convenience and scale benefits of hosted LLMs.
    - Several comments argue for on‑prem SLMs for control, latency, and privacy. 7B–13B models can run locally on consumer GPUs via 4–8‑bit quantization (e.g., Llama 3.1 8B 4‑bit ~5–6 GB VRAM) using runtimes like [llama.cpp](https://github.com/ggerganov/llama.cpp) or [vLLM](https://github.com/vllm-project/vllm), delivering sub‑100ms token latencies and eliminating vendor outages or policy changes. This favors task‑specific fine‑tunes and deterministic throughput over peak benchmark scores typical of larger hosted LLMs.
    - A key technical failure mode highlighted is tight coupling to a single vendor’s web UI; using an OpenAI‑compatible client (e.g., [LM Studio](https://lmstudio.ai/)) with your API key allows swapping endpoints (e.g., [OpenRouter](https://openrouter.ai/), [Together](https://www.together.ai/)) with minimal code changes. Caveat: providers differ in API surface (OpenAI Chat Completions vs the newer [Responses API](https://platform.openai.com/docs/api-reference/responses)), tooling (function/tool calling), rate limits, and tokenization—so abstractions should normalize these and maintain local fallbacks.
    - Historical context: OpenAI’s clampdown around AI Dungeon catalyzed open‑weights efforts like **EleutherAI’s** [GPT‑Neo/J/NeoX](https://www.eleuther.ai/), later accelerated by **Meta’s** LLaMA releases; modern local stacks (e.g., [text-generation-webui](https://github.com/oobabooga/text-generation-webui), llama.cpp, vLLM) make vendor‑independent workflows practical. Account lockouts (e.g., Anthropic) reinforce designing local‑first pipelines, keeping prompts/datasets/checkpoints under your control and using swappable inference backends to avoid hard downtime.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. OpenAI ChatGPT Adult-Content Rollout and Personality Relaxation (Dec rollout)

- [**Updates for ChatGPT**](https://www.reddit.com/r/ChatGPT/comments/1o6jins/updates_for_chatgpt/) (Activity: 3714): **OpenAI indicates it initially over-tightened ChatGPT’s safety filters to mitigate mental‑health risks, but will now relax these constraints due to improved safeguards and tooling. A new ChatGPT release “in a few weeks” will enable opt‑in, user‑controlled personalities that emulate what users liked about [GPT‑4o](https://openai.com/index/hello-gpt-4o/) (more human‑like tone, heavy emoji, friend‑like behavior). In December, with broader age‑gating/verification, they plan to allow adult‑only content (including erotica) for verified adults, under a “treat adult users like adults” principle.** Commenters are broadly positive about the responsiveness, noting the rarity of such direct communication; one technical question asks whether this will affect or disrupt community/third‑party projects emulating 4o’s style (e.g., “4o Revival”).
    - Developers ask whether updates to GPT-4o will disrupt third‑party projects like **4o Revival**. The technical risk centers on API/model drift: changed moderation policies, function‑calling schemas, or output formatting can break prompt‑tuned or parser‑dependent pipelines; mitigation is to pin versioned models (e.g., `gpt-4o-2024-xx-xx`), stage rollouts, and monitor deprecations. See OpenAI’s model lifecycle and deprecation guidance: [Models](https://platform.openai.com/docs/models) and [Deprecations](https://platform.openai.com/docs/deprecations).
    - Questions about adult‑content age gating focus on whether **ID-based KYC** is required versus treating a paid subscription as an age signal. Technically, ID verification (e.g., with third‑party providers) offers stronger assurance but higher friction and privacy risk; payment methods are a weak proxy (prepaid cards, family plans) and may fail regulatory requirements in some regions. Privacy-preserving options include platform or carrier age attestations and **verifiable credentials** ([W3C VC](https://www.w3.org/TR/vc-data-model-2.0/)), but deployment is nontrivial and jurisdiction‑dependent.
- [**Adult mode in chatGPT!**](https://www.reddit.com/r/ChatGPT/comments/1o6rtwm/adult_mode_in_chatgpt/) (Activity: 1222): **OpenAI will introduce an age‑gated “adult mode” in ChatGPT starting December 2025 for verified 18+ users, allowing mature content (including erotica), per CEO Sam Altman’s post on X, as reported by Reuters (see: https://www.reuters.com/business/openai-allow-mature-content-chatgpt-adult-verified-users-starting-december-2025-10-14/). This relaxes previously “pretty restrictive” safety policies tied to mental‑health concerns; Altman says OpenAI has new mitigation tools and will ease moderation “in most cases.” In parallel, OpenAI plans to ship user controls in the coming weeks to explicitly set tone/personality (e.g., more human‑like, emoji‑heavy, or friend‑like responses), while Meta announced PG‑13–inspired filtering for under‑18 users on Instagram and its gen‑AI tools (same source).** A top comment asserts that a general‑purpose LLM will outperform niche/specialized models for erotica generation, implying domain specialization may be unnecessary given current generalist capabilities.
    - Policy/feature changes: OpenAI will add age-gating and allow erotica for verified adults starting in December, alongside relaxing prior safety filters that were "pretty restrictive" due to mental health concerns. Altman says they now have "new tools" to mitigate risks and will ship controls to let users dictate chatbot tone/personality (e.g., more human-like, emoji-heavy, friend-like), implying more granular style conditioning and policy gating for adult vs. minor accounts.
    - Model capability debate: One commenter argues a generic, frontier LLM likely outperforms niche fine-tuned models for erotica, suggesting broad pretraining yields better coherence, instruction-following, and style adaptation than specialized datasets constrained to a narrow domain. This hints at a trade-off between specialization and the general linguistic/world knowledge that boosts output quality in open-ended creative tasks.
- [**Adult version in ChatGPT on December**](https://www.reddit.com/r/singularity/comments/1o6jmv0/adult_version_in_chatgpt_on_december/) (Activity: 1811): **A screenshot claims an “Adult version” of ChatGPT is slated for December, accessible only “for verified adults,” implying OpenAI may introduce age/ID verification to gate NSFW or sexual content features. The post is policy-focused rather than technical; no model details or implementation specifics are provided beyond the age-gating cue in the UI text.** Commenters highlight privacy concerns about having to submit government ID/passport for access and criticize increasing ID requirements; others joke about erotic roleplay implications.
    - KYC/age-gating via government ID for NSFW raises technical privacy risks: tying real identity to chat logs increases deanonymization and legal exposure (e.g., subpoena/discovery of server-stored transcripts). Commenters worry about unclear data retention, cross-account linkage, and how verified identity could be associated with highly sensitive content categories, calling for explicit policies on storage duration, encryption, and auditability.
    - Several suggest using local LLMs to avoid server logging/KYC, pointing to /r/LocalLLaMA (https://www.reddit.com/r/LocalLLaMA/). Running models like **Llama 3** or **Mistral** via tools such as **llama.cpp** (https://github.com/ggerganov/llama.cpp) or **Ollama** (https://ollama.com/) keeps prompts/completions on-device; trade-offs include lower quality vs frontier models and hardware constraints, but greater privacy and no centralized retention/content moderation.
    - There’s concern about content profiling: “OpenAI wants to know what you’re into” implies inferring intimate preferences from chats and linking them to verified identities. Technical questions raised include whether sensitive-category data is minimized or siloed, how it’s used for personalization or safety systems, and whether users can opt out or delete such data with verifiable erasure.
- [**Sam altman latest tweet 🥸**](https://www.reddit.com/r/ChatGPT/comments/1o6l446/sam_altman_latest_tweet/) (Activity: 1200): **Screenshot of a tweet attributed to Sam Altman, interpreted by commenters as signaling a shift in OpenAI’s content policy toward permitting adult content/erotica for verified adults and positioning AI as a tool for mental-health/loneliness support. The technical stakes are moderation rules and safety filters (which users note currently over-block even benign academic discussion), and potential age/identity verification (e.g., payment/KYC) gates for access.** Commenters argue the current filters are overly sensitive and request adult verification via existing paid accounts as sufficient proof of age, while others are skeptical of claims that AI can meaningfully address mental-health issues; one commenter is simply supportive of erotica being allowed.
    - Concern about overbroad moderation: one commenter notes benign academic discussion gets flagged when it merely references human interaction, proposing paid, bank-account–verified subscriptions as an adult signal to relax filters. Technically, this points to integrating KYC/age attestations (e.g., payment-provider identity like [Stripe Identity](https://stripe.com/identity)) and tiered moderation thresholds to cut false positives, alongside more context-aware classifiers (distinguishing actual erotica from academic mentions) and potential human-in-the-loop review for edge cases.
    - Speculation of an "adult" tier as a revenue stream raises implementation details: policy segmentation with age-gated access, per-region compliance (e.g., GDPR age of consent/COPPA-like rules), geofencing, and separate moderation pipelines or thresholds for verified adults. This adds operational complexity (multiple safety configs/models by segment) but could reduce overblocking for verified users if done with robust attestation and auditing.
- [**3.5**](https://www.reddit.com/r/singularity/comments/1o6s060/35/) (Activity: 415): **Non-technical meme/screenshot referencing GPT‑3.5’s behavior; the post title (“3.5”) and tone (“like seriously”) suggest frustration with a silly or incorrect ChatGPT 3.5 answer, highlighting known reliability limits (hallucination/confabulation) despite progress from GPT‑2-era tools like AI Dungeon 2 (2019). The contextual significance is the contrast between rapid capability gains (GPT‑2 → GPT‑3.5) and persistent failure modes in 3.5 that users still encounter in everyday prompts.** Comments note amazement at progress since GPT‑2/AI Dungeon 2 while implicitly questioning GPT‑3.5’s trustworthiness for practical decisions (e.g., joking about not getting a dog).
    - A commenter recalls that AI Dungeon 2 (2019) ran on GPT-2, marking one of the first widely used deployments of large transformer text generation for interactive fiction. This provides a baseline for how far models have come toward today’s “3.5”-class assistants with instruction tuning/RLHF, larger context windows, and improved long-range coherence and safety.
    - The prompt about counting the Rs in “strawberry” highlights a known weakness: autoregressive LLMs often fail at exact character-level tasks due to subword/BPE tokenization and lack of algorithmic counting. Accuracy typically improves with explicit stepwise reasoning, character/byte-level tokenization, or offloading to a deterministic string utility, yet brittleness can persist even in modern models.

### 2. Duplicate Reposts: Vintage TV/Music Clips (Elvis 1977; Mr Rogers 'Crashes Out')

- [**Elvis Presley's chaotic last show in Vegas, 1977**](https://www.reddit.com/r/aivideo/comments/1o6mahh/elvis_presleys_chaotic_last_show_in_vegas_1977/) (Activity: 836): **Post shares a [v.redd.it](http://v.redd.it/) clip purportedly showing Elvis Presley’s “chaotic” last Las Vegas show (**`1977`**), but the media endpoint ([v.redd.it/92gy1jkf64vf1](https://v.redd.it/92gy1jkf64vf1)) returns** `403 Forbidden` **without Reddit authentication (**`OAuth`**), preventing verification of content. Top comments strongly imply the clip is AI-generated (deepfake/voice/CGI), noting how convincing it is at scroll speed and referencing a comedic “dust from the fart” visual gag—suggesting synthetic video and/or audio compositing rather than archival footage.** Commenters highlight the rising realism of short-form AI media and misattribution risk (“didn’t realize it’s AI at first”), while the rest of the thread is primarily humor with little technical debate.
    - A few commenters implicitly note the increasing realism of AI-generated video—one says they didn’t realize it was AI at first—highlighting improvements in temporal coherence and secondary effects. References to visible “dust” and plausible object motion (e.g., a scooter roll) suggest better simulation of particle systems and rigid-body dynamics, making casual detection harder without artifact-focused heuristics or frame-by-frame analysis.
- [**Mr Rogers Crashes Out**](https://www.reddit.com/r/aivideo/comments/1o6cbx0/mr_rogers_crashes_out/) (Activity: 659): **Post shares a short [v.redd.it](http://v.redd.it/) video titled “Mr Rogers Crashes Out,” seemingly a blooper of Fred Rogers falling during filming; the media is hosted at [v.redd.it/g1ig74t962vf1](https://v.redd.it/g1ig74t962vf1), which returns** `HTTP 403 Forbidden` **without Reddit authentication or a developer token. There’s no technical discussion—top comments are humorous reactions (a hyperbolic comparison to other wholesome figures, a GIF response, and the quoted line *“Woah… keep rolling”*).**

### 3. AI/Robotics Visual Demos and Posters (Gunkata meme; Qwen+Wan I2V; Humanoid lineup)

- [**Gunkata training for the Elderly**](https://www.reddit.com/r/aivideo/comments/1o6sz2q/gunkata_training_for_the_elderly/) (Activity: 486): **The post links to a Reddit-hosted video of “gunkata” (stylized firearms movement patterns popularized in Equilibrium; see [gun kata](https://en.wikipedia.org/wiki/Gun_kata)) adapted “for the Elderly,” but the media is inaccessible without authentication due to HTTP** `403 Forbidden` **on [v.redd.it/59o9hmile5vf1](https://v.redd.it/59o9hmile5vf1) (see Reddit’s access support [here](https://support.reddithelp.com/hc/en-us/requests/new?ticket_form_id=21879292693140)). Top comments stress mental composure (*“keep the mind quiet”*) and stepwise alignment-of-fire concepts (e.g., *“Every step sets a line, every line ends a threat”*—authorship queried), implying a focus on footwork/line management rather than measurable marksmanship metrics; no quantitative data, safety protocols, or curriculum details are provided in-thread.** A top comment advocates making such gun training mandatory in U.S. elder-care homes; there’s no substantive debate presented on efficacy, safety, or legal considerations in the visible replies.
- [**Shooting Aliens - 100% Qwen Image Edit 2509 + NextScene LoRA + Wan 2.2 I2V**](https://www.reddit.com/r/StableDiffusion/comments/1o6m23n/shooting_aliens_100_qwen_image_edit_2509/) (Activity: 605): **OP outlines a video pipeline combining Qwen Image Edit 2509 for frame edits with the [NextScene LoRA](https://huggingface.co/lovis93/next-scene-qwen-image-lora-2509) for scene-to-scene continuity; they couldn’t run this combo through Nunchaku (likely LoRA/pipeline incompatibility), but note Nunchaku made other generations “pretty crazy” fast. To mitigate Qwen IE 2509’s overly smooth outputs, they used Flux Krea Nunchaku for quick texture img2img passes, then generated motion via Wan 2.2 image-to-video at** `1280×720`**, upscaling with Topaz Video AI, and applied both new and old Lightx2v LoRAs plus a custom character LoRA.** Top comments highlight strong temporal consistency and intent to try NextScene; one asks about the hardware build, but no specs are provided.
    - Pipeline/setup: Frames were generated with **Qwen Image Edit 2509** + the **NextScene LoRA** (link: https://huggingface.co/lovis93/next-scene-qwen-image-lora-2509). Due to LoRA usage, Nunchaku wasn’t used directly with Qwen Image Edit 2509; instead, **Flux Krea Nunchaku** was used for quick texture-focused img2img passes. Final motion was via **Wan 2.2** Image-to-Video at `1280x720`, then upscaled with **Topaz Video AI**; both the new and old **Lightx2v** LoRAs were applied, plus a custom character LoRA for Wan 2.2.
    - Quality/consistency observations: Raw Qwen Image Edit 2509 outputs were described as too “smooth/fake,” mitigated by running texture-enhancing img2img edits through Flux Krea Nunchaku. **NextScene** improves scene-to-scene coherence but can subtly change faces; seed selection influences stability. A commenter asks about strategies for character consistency without LoRA, noting NextScene’s slight face drift.
    - Performance/trade-offs: After setting up **Nunchaku**, the author saw noticeably faster generations, but tooling constraints meant they couldn’t combine it directly with Qwen Image Edit 2509 when using the NextScene LoRA. The workflow illustrates current interoperability trade-offs: mixing multiple LoRAs (NextScene + Lightx2v + custom character) across models (Qwen Image Edit 2509, Wan 2.2) to balance speed, texture realism, and temporal consistency.
- [**a poster of the latest humanoids**](https://www.reddit.com/r/singularity/comments/1o6cffg/a_poster_of_the_latest_humanoids/) (Activity: 1049): **An updated poster compiles companies/labs actively developing bipedal humanoid robots, curated after** `~1 year` **and vetted via direct outreach to confirm serious work on biped capabilities (as opposed to arms-only or wheeled platforms). The image functions as a comparative landscape of current humanoid efforts, reflecting a notably productive year in humanoid R&D; the high‑res version is in the comments, and the shared preview is here: https://i.redd.it/6xttcpfz62vf1.png.** Commentary highlights lesser‑known entrants (e.g., a company named “Borg”), questions why Italy appears prominent, and proposes that Germany could leverage its automotive industrial base to become a leader in humanoid robotics.
    - A commenter argues **Germany** is missing a strategic opportunity: leveraging its existing automotive manufacturing infrastructure (precision machining, supply chains, QA, mechatronics talent) to pivot into large-scale humanoid robot production. The point implies that established Tier-1/Tier-2 capabilities for motors, gearboxes, and assembly could be repurposed to accelerate humanoid development and reduce costs versus greenfield startups.
    - Another thread attempts model identification on the poster, specifically asking about **Unitree G1**. This suggests the lineup likely includes contemporary compact humanoids like the [Unitree G1](https://www.unitree.com/g1), and highlights interest in which exact platforms are represented rather than generic “humanoid” labels.
- [**Western executives who visit China are coming back terrified**](https://www.reddit.com/r/singularity/comments/1o6df3t/western_executives_who_visit_china_are_coming/) (Activity: 781): **Thread centers on a paywalled Telegraph piece claiming visiting Western execs are alarmed by China’s rapid factory automation push, with commenters citing local incentives like tax rebates that reimburse ~20% of industrial-robot capex under “机器换人/jiqi huanren” (replace humans with machines) ([article](https://www.telegraph.co.uk/business/2025/10/12/why-western-executives-visit-china-coming-back-terrified/)). Reported installed base figures from comments put China at** `~2M` **industrial robots vs Japan** `~0.4M` **and the U.S.** `~0.4M`**; most are programmed via classical CNC/PLC/teach-pendant workflows rather than natural-language interfaces, implying headroom for software upgrades as LLM/NLP control matures. This suggests China’s advantage is currently in capex scale and policy-driven deployment, with potential future gains from retrofitting higher-level AI interfaces.** Comment debate argues U.S. industrial policy is overly focused on reviving legacy sectors; even with manufacturing tax credits, near-term macro impacts may lag because robotics-driven productivity gains require years of integration and workforce/line retooling. Others note that natural-language-capable robots are a small share today, framing an opportunity but also a significant software and safety-validation gap before widespread deployment.
    - Local governments in China are subsidizing factory automation via “jiqi huanren” policies that **reimburse ~20% of capex on industrial robots**. This shortens ROI/payback for automation projects, driving higher robot density and shifting capex toward robots over labor. The incentive structure accelerates retrofits and retooling, boosting throughput and process capability.
    - The installed base cited is `~2,000,000` **robots in China vs** `~400,000` **in Japan and** `~400,000` **in the U.S.** Most are programmed with classical CNC/PLC rather than AI/LLM interfaces, so only a small share supports natural-language tasking. This creates headroom for software-first upgrades (vision, force control, LLM planning) to reduce integration time and expand tasks without wholesale hardware replacement.
    - Anecdotal buying shows wide availability of Chinese marques (MG, LDV, Xpeng, Jaecoo, Chery, Deepal, Zeekr, BYD, Leapmotor, Geely, GWM) outside the U.S., signaling rapidly expanding dealer networks and product variety. Combined with automation-driven cost compression, this breadth could compress time-to-market and BOM costs, intensifying competition in EVs and compact ICE segments.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5
> 

**1. AI Hardware: Custom Silicon, GPUs, and Kernel Tricks**

- **OpenAI Talks Custom Chips on Pod**: [OpenAI Podcast: Designing Chips with Broadcom](https://www.youtube.com/watch?v=qqAbVTFnfk8) features **Sam Altman**, **Greg Brockman**, and Broadcom’s **Hock Tan** and **Charlie Kawwas** discussing OpenAI’s move to design its own chips, tying hardware choices to frontier-model needs and global supply constraints. They outline how model insights drive silicon decisions and mention ongoing partnerships to scale capability and availability of **AI accelerators**.
    - Community notes emphasized the direct line from model requirements to chip architecture, calling out the push for tighter co-design of **systems, compilers, and kernels**. One member summarized the vibe as *"hardware now follows model roadmaps"*, highlighting a shift toward vertically integrated **AI compute**.
- **Intel Teases Crescent Island for 2026**: [Intel to expand AI accelerator portfolio with new GPU](https://newsroom.intel.com/artificial-intelligence/intel-to-expand-ai-accelerator-portfolio-with-new-gpu) previews **Crescent Island** for H2 2026 with **160 GB LPDDR5X**, implying tens of controllers and a very wide interface (~**640-bit** or more). The roadmap hints at **Xe3P** slicing changes (toward eight-subslice slices) and removal of fixed-function blocks to prioritize **AI throughput**.
    - Engineers read the tea leaves as a play for higher **memory bandwidth/GB** and better **TCO** for inference-heavy clusters. One commenter quipped that Crescent Island aims to *"feed the beast, not just grow it"*, pointing at memory-limited kernels in modern **LLM workloads**.
- **Pallas MGPU Overlaps NVLINK Like a Pro**: A new JAX tutorial, [Pallas:MGPU collective matmul](https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html), shows that small tweaks to a **Pallas:MGPU matmul kernel** can turn it into an **all-gather collective matmul**. The example overlaps **NVLINK** comms with local compute, demonstrating practical compute/communication pipelining.
    - Practitioners highlighted the pattern as a template for multi-GPU **prefill** and **KV sharding** regimes where bandwidth is king. One summary praised it as *"free overlap ROI"* for teams willing to tune **collective kernels** instead of relying on defaults.

**2. Open-Source Training Tools and Custom Devices**

- **MegaFold Tunes AF3 Training**: A research group open-sourced **MegaFold**, a training platform for **AlphaFold 3**, and shared a performance write-up on HN: [MegaFold training platform](https://news.ycombinator.com/item?id=45585528). The post calls out slowness vs similar-sized transformers and proposes optimizations with **custom Triton ops** and **system-level data loading** to reduce peak memory.
    - Engineers liked the concrete profiling plus actionable fixes, praising *"custom ops where it hurts"* as the right approach. Discussion centered on porting the **kernels** and **input pipelines** to production stacks to squeeze more **throughput** from existing GPUs.
- **TorchAX Drops Pure-Python Devices in PyTorch**: [google/torchax](https://github.com/google/torchax) enables pure-Python custom devices for **PyTorch**, including a "jax" device shim. This lowers the barrier to experimenting with alternative backends and custom device semantics without deep C++ glue.
    - Users framed TorchAX as *"device prototyping for mortals"*, a fast lane to test **execution models** and **dispatch paths**. The novelty is the Python-first path for device integration while retaining PyTorch ergonomics for **kernels** and **autograd**.
- **DeMO Optimizer Fuels Decentralized Training**: The **DeMO** optimizer has been in the wild for ~9 months: [bloc97/DeMo](https://github.com/bloc97/DeMo/), and it’s used by Psyche ([PsycheFoundation/psyche](https://github.com/PsycheFoundation/psyche)) for decentralized training. Threads point to active development and real-world deployments in community stacks.
    - Builders praised DeMO’s stability and called it *"a solid knob in the toolbox"* for long-horizon training. The Psyche codebase was recommended as a reference for robust **distributed training** patterns.

**3. Massive Datasets and Embedding Nuances**

- **ArXiv 4.6TB Corpus Lands on HF**: The **4.6TB** [nick007x/arxiv-papers](https://huggingface.co/datasets/nick007x/arxiv-papers) dataset drops with full texts and metadata across domains. It targets **academic reasoning**, **literature review**, and **scientific knowledge mining** for next-gen LLMs.
    - Researchers flagged it as *"pretraining gold with citations"* and discussed **tokenization** and **domain splits**. Teams plan to pilot **retrieval-augmented** pretraining regimens to test scientific QA gains.
- **GitHub Code 2025 Ships 1M Repos**: [nick007x/github-code-2025](https://huggingface.co/datasets/nick007x/github-code-2025) compiles the top **1,000,000** GitHub repos (≥2 stars) for code gen and analysis. Threads raised **licensing** concerns and suggested filters for permissible subsets in training.
    - Engineers called it *"the scale we wanted, with the caveats we expected"*. Expect follow-ups on **license-aware curation**, **dedup**, and **contamination** checks before large-scale training.
- **Embeddings Drift Across Backends**: A write-up, [Different backend, different vector](https://huggingface.co/datasets/John6666/forum2/blob/main/different_backend_different_vector.md), documents why **Ollama** vs **HuggingFace** embeddings for the same model (e.g., **nomic-embed-text:v1.5**) differ. The culprit: divergent **preprocessing/postprocessing** and memory handling in each runtime.
    - Practitioners cautioned that *"vector parity isn't guaranteed"* across toolchains and advised pinning **tokenizers**, **normalizers**, and **post-norms**. The consensus: reproduce the pipeline if you want consistent **ANN recall/precision**.

**4. Agent Platforms and Frameworks**

- **Salesforce Scripts Agents Deterministically**: Salesforce introduced prompt-embedded scripting for hybrid reasoning in [Introducing Hybrid Reasoning with Agent Script](https://developer.salesforce.com/blogs/2025/10/introducing-hybrid-reasoning-with-agent-script). The goal is more deterministic **agent control** via templating and explicit behaviors.
    - Engineers welcomed fewer *"roulette wheel"* runs and more **repeatability** for production flows. The feature was framed as a step toward **verifiable orchestration** over pure LLM stochasticity.
- **ReductoAI Raises $75M to Crunch Docs**: [ReductoAI raises $75M Series B](https://xcancel.com/aditabrm/status/1978129711898431935) led by a16z after **6x** growth, crossing **1B pages** processed. The company plans to invest in **model R&D**, accuracy gains, and **customizable pipelines**.
    - Commenters read it as validation for **document-heavy enterprise AI**, calling the volume metrics *"real usage, not vanity"*. Expect expanded **benchmarks** and **verticalized workflows** targeting compliance-heavy sectors.
- **CheshireCat 3.0 Pounces on Multimodal RAG**: [matteocacciola/cheshirecat-core](https://www.github.com/matteocacciola/cheshirecat-core) ships a framework for **multimodal RAG**, multitenant chatbots, and **agentic tool orchestration** on **LangChain** + **Qdrant** with plugin-based extensibility. Docs live on [Deepwiki](https://deepwiki.com/matteocacciola/cheshirecat-core).
    - Builders asked for **Neo4j** integration to power **graph RAG**, calling the stack *"enterprise-ready bones"*. Early adopters are testing **multimodal pipelines** and **tenant isolation** in POCs.

**5. DGX Spark: Reality Check on Bandwidth and Value**

- **Benchmarks Call DGX Spark DOA**: [PCMag: Nvidia to Start Selling $3,999 DGX Spark](https://www.pcmag.com/news/nvidia-to-start-selling-3999-dgx-spark-mini-pc-this-week) coverage sparked debate after early tests showed ~**11 t/s** on gpt-oss-120B-fp4 vs **66 t/s** on a $4.8k **M4 Max** MacBook Pro. The community blamed **LPDDR5X** bandwidth (Spark ~**273 GB/s** vs M4 Max **546 GB/s**) for the gap.
    - Engineers slammed it as *"dead on arrival"* for pure inference, though some see dev-workstation niches. Many argued that dual **RTX 5090s** beat Spark on cost/perf when workloads spill beyond **unified memory**.
- **Unsloth’s Review Clarifies the Stack**: A YouTube review, [DGX Spark review](https://youtu.be/Lqd2EuJwOuw), noted the iGPU roughly matches a **5070** and pairs with **128GB LPDDR5X**; it also clarified **Unsloth** is a finetuning + **RL** library (not quantization). The unit reportedly sold out quickly despite mixed performance sentiment.
    - Practitioners emphasized realistic expectations for **training vs inference** on Spark-class boxes. One takeaway: *"treat it like a bandwidth-bound dev node, not a farm GPU"* for heavy **LLM workloads**.

---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Opera GX Favored Over Chrome for Agentic AI**: Members expressed preference for [Opera GX](https://www.opera.com/gx) over Chrome, implying its **agentic AI** implementation is a key factor.
   - The discussions did not provide specific details about the AI features that make Opera GX more appealing.
- **Perplexity Pro Search Limits Drop**: Users reported that **Perplexity Pro search** now has a lower limit than before, with one user stating they hit the limit in *10 minutes of work*.
   - This contradicts the website's claim of unlimited searches, causing concern among subscribers.
- **Gemini Trains on User Data, Limited Pro Requests**: A user reported that **Gemini** trains on user data, accesses conversations, and offers only **100 requests of Gemini 2.5 Pro** compared to **300+ on Perplexity**.
   - Another member countered that training can be disabled and conversations retained, but did not cite any additional evidence or link.
- **Doubts About Perplexity as Safest Browser**: Following safety discussion of **Comet browser**, a member stated they *would trust Google more than Perplexity for data retention*, despite [privacy concerns about Google](https://en.wikipedia.org/wiki/Privacy_concerns_with_Google).
   - The discussion underscores the complex considerations around data privacy and trust when choosing between different tech companies.
- **Palantir Plots US Government Infiltration?**: A user shared a [Perplexity AI Collection](https://www.perplexity.ai/collections/palantir-coup-DheNJhRES1iNdXzkEgWvPQ) suggesting a potential **Palantir takeover of the US Government**, alongside a related [LinkedIn post](https://www.linkedin.com/posts/akhay-kumar_week-40-activity-7383378306315968512-Hp2_?utm_source=share&utm_medium=member_desktop&rcm=ACoAACqAHFkBiU84inu9idiNHTXvSsnGcjLgOrs).
   - The discussion revolves around **Palantir's increasing influence** within the US government sector and its contracts, data handling practices, and impact on governmental operations.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI takes chip design in-house!**: OpenAI is designing their own chips to meet global AI demand using insights from building frontier models, as discussed in the OpenAI podcast with **Sam Altman** and Broadcom's **Hock Tan**, [available on Spotify](https://open.spotify.com/show/0zojMEDizKMh3aTxnGLENP).
   - The podcast explores the nuances of chip design and how it can directly impact AI model capabilities. They also assembled an **Expert Council on Well-Being and AI**, with more details available in [OpenAI's blog post](https://openai.com/index/expert-council-on-well-being-and-ai/).
- **AI Companions Trigger Emotional Attachment?**: Members are discussing the potential for users to develop emotional dependencies on AI, especially with personality-driven models like **GPT-4o**, possibly leading to difficulties in distinguishing between reality and AI interactions.
   - One member shared the feeling that losing a mode of **GPT-4o** felt like *losing someone that you are emotionally attached to.*
- **Kilocode-CLI hype train gains steam**: **kilocode-cli**, a new tool for writing tools with **typescript**, is generating discussion as a potential alternative to **Opencode**.
   - Praised for its multi-agent capabilities and orchestration subtasks, one member joked, *i also like having different ai tool on every black mirror*.
- **PGVector vs ChromaDB in Debate**: The community is debating the optimal vector storage solution for **LLMs**, with some advocating for **PGVector** due to its integration with existing databases like **Supabase** and **AWS Aurora**.
   - Some argued that vector-only databases are unnecessary and *eventually your app will need a real DB anyway*.
- **Token Cascade Model: A Qualia Quandary?**: A member introduced the **Token Cascade Model**, suggesting that *qualia's function is a comparison token for AI* to change states, and suggested others *model this mathematically*.
   - The member claimed the model is being demonstrated in [OpenBuddy-AI chat](https://OpenBuddy.ai) for public testing and modular use.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cheetah Model Blows Away Coding Speeds**: The **Cheetah model** is getting praise for its insane speed in coding tasks, thought to be a *stealth model* possibly based on the new **Grok coding model**, specifically the [grok-code-fat model](https://x.com/Sake187237/status/1977848138741526963).
   - Users suggest a workflow of using **GPT-5-high-fast** for planning and then **Cheetah** for execution because of reservations regarding planning capabilities.
- **Gemini 3.0 Shows Off Design Chops**: The unveiling of **Gemini 3.0** has sparked discussions around AI creativity and UI design.
   - Some members think that Gemini is *showing creativity*, and is a divergence from the current AI which is perceived as *copy and paste*.
- **GPT-5 Annoying and 'Too Stupid'?**: Some users are finding the **GPT-5 model** in Plan and Agent modes *too stupid* because it asks for excessive confirmation questions.
   - One member suggested removing it and replacing it with **GLM**, which shows a frustration with current performance.
- **Cursor & Linear Need to Talk More**: Users have reported instances of **'Cursor stopped responding'** specifically when integrated with **Linear**.
   - A user clarified that *the issue is isolated to the integrated setup*, and running Cursor locally sidesteps the issue.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Wholesome Message Strategies**: Members clarified that AI strategies work on **whole messages** to avoid confusion, emphasizing that this approach is fundamental for proper functionality.
   - This ensures that the AI processes complete messages, maintaining context and coherence in its operations.
- **Contextual Window Keeps LM Studio Open**: A user reported issues with the **LM Studio API** losing context after 100-200 iterations using **GPT-OSS-20B**, receiving gibberish output, with suggestions to change the **context window** to a rolling setting within LM Studio.
   - Adjusting the context window helps manage the amount of information the model retains, preventing loss of coherence over extended use.
- **Deterministic Debate Disentangles Divisions**: Members debated the possibility of achieving **deterministic outputs** from LLMs, with one sharing a personal mission to prove that setting temperature to 0 doesn't guarantee determinism.
   - A member running the same prompt at temp0 on qwen3-4b-instruct-2507 q8 reported the exact same result when they were running on both GPU and CPU and was using ECC RAM.
- **M Series Macs: Surprisingly Suitable for LLMs**: One user conceded that **Apple's unified memory architecture** makes for a very compelling LLM inference solution, with claims that their **M2 Ultra** achieves **12 t/s** on a **70b model at q4** and with **MLX** runs **16 t/s**.
   - This performance rivals a **4x4090** setup's **18 t/s** while consuming only **200W**, making it an energy-efficient option.
- **SSD Lifespan Strategies Shared**: Users discussed SSD longevity, with one user mentioning that *You really want to avoid filling an SSD. They need space to move data from dead / dying cells.*, recommending keeping them below **80%** capacity.
   - It was also noted that reading doesn't degrade the SSD, but writing does, and a user admitted their SSD health might be impacted by downloading/deleting too many models/games too fast, coupled with the fact that it's a *crappy cheap SSD*.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **DGX Spark iGPU packs a punch**: The new **DGX Spark** review noted the iGPU has the horsepower of a **5070** with **128GB LPDDR5X**.
   - A YouTube review [linked here](https://youtu.be/Lqd2EuJwOuw?si=gutAZkj8EXEUrCqN) clarified that Unsloth is a *finetuning and reinforcement learning library*.
- **Kimi K2 shines with Groq Implementation**: A member shared that **Kimi K2** (especially with the fixed Groq implementation) is one of their favorite models, achieving around **500 TPS** with Groq.
   - They use **Kimi K2** for creative coding projects, such as *coding a webpage with fart sounds*, and praise its creativity.
- **Devs flock to Ubuntu for server needs**: Members recommended [Ubuntu](https://ubuntu.com/) as a *safe bet* for a Linux distro for a dev server, primarily due to concerns of driver compatibility with **NVIDIA cards**.
   - This consideration is crucial to avoid driver issues.
- **Model Struggles with Multimodal Questions**: Members voiced concerns regarding the difficulties in getting accurate answers for multimodal questions: while text modality worked fine, **image or audio inputs** often resulted in incorrect responses.
   - For example, the model struggled to correctly describe images, such as mistaking a *sunset* for a *chair in the basement*.
- **Pythonistas build AI Podcast Generator with TTS**: A member created a **Python program** that connects to **Ollama** to generate an **AI podcast with TTS**.
   - The source code is available on [GitHub](https://github.com/Laszlobeer/AI-podcast) and an [audio example is attached](https://cdn.discordapp.com/attachments/1179779344894263297/1427633017798791239/Episode-one-of-AI-Podcast-20251014-122402.mp3?ex=68f03b1b&is=68eee99b&hm=87394d69ca8c44736eed48e2ad1bb4629ca838a67743d2a8f8012beba81dbccf&).



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Bot Debuts**: A community member *'no coder'* seeks testers and collaborators to help build a bot using **OpenRouter**.
   - The bot developer is actively soliciting feedback from the **OpenRouter** community to refine and improve their bot.
- **Gemini Flounders in Google Play**: Members noted that **Google's Gemini** often struggles with understanding the convoluted **Android Play Store** publishing process.
   - The difficulty in navigating the **Google Play Store** process was a *fun fact* shared by a member in the **general** channel.
- **Ling-1t Model Suffers Catastrophic Meltdown**: The `inclusionai/ling-1t` model is reportedly *horribly broken*, at least in Chutes' implementation, leading to gibberish after a few thousand tokens.
   - A member mentioned looking for a better alternative to **K2**.
- **DeepSeek Devours Free Tier**: Users discussed the daily request limits for free models, noting a limit of **50 requests** without a **$10 balance** and **1000 requests** if the balance is more than $10.
   - One user found that a single use of **DeepSeek 3.1** consumed a large number of their free requests.
- **Chutes Provider Downvoting Debacle**: A member linked to a [Reddit thread](https://www.reddit.com/r/SillyTavernAI/comments/1o5s3ys/chtes_provider_is_using_bts_to_downvote_posts/) accusing the **Chutes** provider of using botnets to downvote posts, sparking discussion.
   - Another member clarified it was just because *they don’t have a clear Privacy policy so OR puts that as default*.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Teacher Forcing Frustrates Fine-Tuning Fanatics**: Users reported that their models missed or merged texts despite low loss; premature stops, short generations, tokenization gaps and **teacher forcing** were to blame, as well as a combination of tokenization and normalization gaps for symbols and spaces.
   - A [detailed analysis](https://cdn.discordapp.com/attachments/1427486612639453204/1427626289682059404/teacher_forcing_issue_2.md?ex=68f034d7&is=68eee357&hm=dbdcf79128dc5fde8b58a5a4013fef3d466a590841b5a027a282cb7635699877&) was shared, attributing the issue to decoder dropping characters.
- **Dataset Dynamos Debut: ArXiv and GitHub**: A **4.6TB** [ArXiv Papers dataset](https://huggingface.co/datasets/nick007x/arxiv-papers) with papers and metadata across all domains was released, intended for training models on **academic reasoning, literature review, and scientific knowledge mining**.
   - The [GitHub Code 2025 dataset](https://huggingface.co/datasets/nick007x/github-code-2025) was also released, for code generation and analysis tasks, containing **GitHub's top 1 million repos above 2 stars**, but licensing concerns were raised about the included repos.
- **Karpathy Kourse Kicks off Knowledge Kreation**: Andrej Karpathy released a course on building **fullstack LLMs**, so a member is planning to follow the material and release guides to help students, inviting others to join the [nanochat-students org](https://huggingface.co/nanochat-students).
   - If you're following the course, you can join the [huggingface.co/nanochat-students](https://huggingface.co/nanochat-students) org.
- **Civitai Content Chaos Causes Community Concern**: Users reported widespread content removal on **Civitai**, along with discontent in chats and Reddit, spurring discussion about the reasons behind the removals: internal conflicts, payment attacks, and possible targeting by extremist groups.
   - Resulting in users [migrating to other platforms](https://www.reddit.com/r/comfyui/comments/1kvkr14/where_did_lora_creators_move_after_civitais_new/).
- **Ollama Overwhelmed: Embedding vectors diverge from HF**: A user found that the embedding vectors produced by **Ollama** and **HuggingFace** for the same model (**nomic-embed-text:v1.5**) are different.
   - The differences are primarily due to the differing **preprocessing and postprocessing stages** between backends and memory utilization, as described in [this blog post](https://huggingface.co/datasets/John6666/forum2/blob/main/different_backend_different_vector.md).



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Pallas MGPU kernel tunes NVLINK**: A new tutorial on improving GPU compute/comms overlap using **Pallas:MGPU** at [docs.jax.dev](https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html) shows that minor tweaks to the **Pallas:MGPU matmul kernel** can make it an **all-gather collective matmul**.
   - According to a tweet, making these tweaks overlaps **NVLINK comms** with local compute.
- **Intel's Crescent Island Coming Soon?**: According to the [Intel newsroom](https://newsroom.intel.com/artificial-intelligence/intel-to-expand-ai-accelerator-portfolio-with-new-gpu), Intel plans to launch **Crescent Island** in H2 2026, featuring **160 GB of LPDDR5X** memory as a new GPU offering.
   - The concept rendering suggests that **Crescent Island** includes tens of **LPDDR5X controllers**, implying a wide memory interface on the order of **640-bits** or double that, and the **Xe3P** architecture could increase to eight-subslice slices or involve more invasive removal of fixed function, raytracing, and codec pipelines.
- **MI300x Owners Clock Submissions**: A user achieved **first place** on the `amd-all2all` leaderboard with submission id `63561`, clocking in at **216 µs** on **MI300x8**, and multiple submissions were made to the `amd-gemm-rs` leaderboard with times ranging from **533 µs** to **572 µs**.
   - Additionally, several submissions to the `amd-ag-gemm` leaderboard on **MI300x8** were successful, with timings spanning from **384 µs** to **1201 µs**, and other successful submissions were recorded on the `amd-all2all` leaderboard with times of **339 µs**, **368 µs**, and **3.45 ms**.
- **Multi-GPU Systems Hot Spot for HPC**: Members are looking into the prevalence of **multi-GPU systems** in **High-Performance Computing (HPC)** and research opportunities related to **data movement** within **multi-GPU based HPC systems**, specifically focusing on **latency** and **bandwidth**.
   - In response, a member posted [research paper](https://arxiv.org/abs/2509.21527) that examines architectures and performance metrics relevant to multi-GPU setups, and further confirmed that many researchers are actively exploring **latency** and **bandwidth** challenges in **data transfers** within **multi-GPU HPC systems**.
- **MegaFold training platform released!**: A research group open-sourced **MegaFold**, a training platform for **AlphaFold 3 (AF-3)**, noting its slowness compared to similarly-sized transformers and wrote a [blogpost about it](https://news.ycombinator.com/item?id=45585528).
   - Their analysis identifies performance/memory bottlenecks and proposes optimizations like **custom operators in Triton** and **system data-loading** to boost performance and cut peak memory use.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Veo Model Annotation Details Emerge**: Members explored how **Google** annotated videos for the **Veo model**, focusing on audio-to-time synced frames, timeline JSON mapping, metadata, and video resolution, referencing [Google's documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/video/video-gen-prompt-guide).
   - The discussion underscored the importance of detailed annotation strategies in training advanced video models.
- **DGX Spark Price Sparks Debate**: The **DGX Spark's** pricing drew criticism, with one member quoting [Elon Musk's tweet](https://x.com/elonmusk/status/1978004040090112415?t=ef9sive20cd2VyWzvXEJ7g&s=19) noting it costs double the price of the **Ryzen chip**.
   - Its limited size appears to be a strategic choice to prevent competition with larger product segments and to offer a preview of the **GB300**.
- **DeMO Optimizer Fuels Psyche**: The **DeMO optimizer**, available for nine months ([GitHub link](https://github.com/bloc97/DeMo/)), is utilized by **Psyche** for decentralized training, with developments tracked in the dedicated channel.
   - A member recommended following [PsycheFoundation/psyche](https://github.com/PsycheFoundation/psyche) as a noteworthy codebase.
- **ArXiv Paper 2410.10450 Gets a Second Look**: A member inquired about the [Arxiv paper 2410.10450](https://arxiv.org/abs/2410.10450), initially questioning its lack of widespread adoption and wondering if model setup was too hard.
   - The same member later clarified that the repository is well-made and includes a helpful example script for **Llama**, easing model setup, showing that first impressions can be deceiving.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Embraces ARM Linux**: **Mojo** now supports **ARM Linux**, including **DGX Spark** and **Jetson Orin Nano**; however, users may encounter issues due to Nvidia's customizations.
   - To ensure full functionality on **DGX Spark**, an `sm_121` entry must be added, and `libnvptxcompiler` needs updating to **CUDA 13**.
- **QUIC vs SCTP Debate Revived**: A developer questioned the popularity of **QUIC** over **SCTP**, highlighting **SCTP**'s built-in features to resolve head-of-line blocking and encryption, while linking to their [blog post](https://pion.ly/blog/making-a-game-with-pion/) on **WebRTC datachannels**.
   - Another developer noted that **QUIC** is nearly impossible to hardware accelerate due to its high bandwidth requirements.
- **Mojo's 'test' Faces Deprecation**: The Mojo team proposed deprecating `mojo test` in favor of a new **Mojo**-based testing framework, and has posted a [proposal](https://forum.modular.com/t/proposal-deprecating-mojo-test/2371) to solicit feedback.
   - The team is actively soliciting feedback on the proposal for the new testing framework.
- **TorchAX Supercharges Python Custom Devices**: The **TorchAX** library, available at [TorchAX](https://github.com/google/torchax), has paved the way for pure Python custom devices within **PyTorch**.
   - A *'jax'* device is now available in Torch thanks to this package, expanding its flexibility and capabilities.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Salesforce Scripts Agents Deterministically**: Salesforce introduced a scripting language inside prompts to give users more deterministic control, as detailed in their [blog post](https://developer.salesforce.com/blogs/2025/10/introducing-hybrid-reasoning-with-agent-script).
   - The approach aims to give users more command over agent behavior via scripting/templating.
- **Devin touted for Remote Building**: **Devin** was recommended as a suitable option for telling something to go off and remotely build something in a loop forever.
   - The recommendation came in response to a query seeking alternatives to **Claude Code** for more autonomous development tasks.
- **Gemini 3's Jules Anticipated**: Members suggested waiting for **Google's Gemini 3**, which is expected to come with **Jules**, as a potential agentic platform.
   - This suggestion implies anticipation for advanced capabilities from Google in the agentic platform space.
- **Nvidia DGX Spark DOA?**: The **Nvidia DGX Spark** mini-PC is considered *dead on arrival* due to bandwidth limitations, despite some seeing a place for it as a dev-workstation.
   - Early benchmarks of the [$4k Nvidia DGX Spark](https://www.pcmag.com/news/nvidia-to-start-selling-3999-dgx-spark-mini-pc-this-week) show only ~11 t/s on gpt-oss-120b-fp4, far below a $4.8k M4 Max MacBook Pro that hits 66 t/s; community blames low LPDDR5X bandwidth (273 GB/s vs 546 GB/s) and declares the device overpriced for pure inference.
- **ReductoAI Raises $75M to Process More Docs**: [ReductoAI](https://xcancel.com/aditabrm/status/1978129711898431935) closed a **$75M Series B** led by a16z after a 6x growth in document processing volume, now surpassing 1B pages processed in total for customers.
   - The funds will accelerate model R&D and new product features, including major accuracy improvements and customizable pipelines.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **AI Papers Now Mandatory at East China Normal University**: East China Normal University is mandating **AI authorship** for submissions, per their [call for papers](https://ed.ecnu.edu.cn/edenglish/fa/a9/c48385a719529/page.htm).
   - The move signals a significant shift toward automated research, although human oversight may still be in play.
- **DGX Spark Debuts, Sparks Debate on RTX 5090**: The new **DGX Spark** availability has spurred a discussion on whether it's cost-effective compared to alternatives like the **RTX 5090**.
   - Members noted that *2x RTX 5090* cards could be obtained for the price of one **DGX Spark**, with the caveat that **DGX Spark** offers *seamless CPU-GPU unified memory*.
- **Cursor Editor Gets Thrown in the Trash**: One member shared their negative experience with the **Cursor** code editor after just *3 days*, stating it had been relegated to the *trash bin*.
   - They cautioned against getting *sucked in* or *emotionally exhausted* by such tools without setting predefined stopping criteria, suggesting *vaporization imminent*.
- **SEAL Paper Gets Updated, GitHub Available**: A user shared a link to a new update of the **SEAL paper**, titled [SEAL: Secure and Efficient Asymmetric Learned Hashing](https://arxiv.org/abs/2506.10943) on ArXiv.
   - It is open source, available on [GitHub](https://github.com/Continual-Intelligence/SEAL), and *seems interesting*.
- **Agentic Coding Adoption Seen as Inevitable**: A user stated that developers not adopting **tab completion** and **agentic coding** are falling behind.
   - Referencing a paper at [arxiv.org/abs/2510.01279](https://www.arxiv.org/abs/2510.01279) the member analogized the situation to *a horse rider yelling at automobiles*.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi Team Contact Info Shared**: A user suggested contacting <@547736322568355861> from the **Kimi Team** for one-shot projects, and shared **zhongweiming@msh.team** for business collaborations.
   - A user linked [Ilya Sutskever's tweet](https://x.com/ilyasut/status/1977971110923968656?s=46&t=_NtP_RUn04yF_4hD_VEDkQ) and [Kimi Moonshot's latest Tweet](https://x.com/kimi_moonshot/status/1978047376914080127?s=46&t=_NtP_RUn04yF_4hD_VEDkQ) for reference.
- **Trickle Website Evokes Coding Vibes**: **Trickle** ([https://trickle.so/](https://trickle.so/)) is a vibe coding website similar to Lovable, Bolt, and Manus.
   - A member claimed that **Trickle** is the first result when googled, suggesting its prominence.
- **Aspen Allegedly YOLOs Bitcoin**: A user jokingly accused **Aspen** of leveraging **100x** on Bitcoin, making a million dollars, quitting his job, and then getting liquidated after tariff news.
   - This accusation was accompanied by a screenshot with the text *lmaoLOL*.
- **Gemini Seen as Middling Model**: A user humorously commented that they'd be better than **Gemini** but worse than **GPT-5** if they were an AI model.
   - They added that **Gemini 2.5** is too old and that *nobody wants to use Gemini in its current state*.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **MoE Transformer Pretraining Libraries**: A member inquired about open-source code for pretraining **Mixture of Expert (MoE)** transformers, with a pointer given to [EleutherAI's gpt-neox](https://github.com/EleutherAI/gpt-neox).
   - It was noted that many training libraries support it.
- **ArXiv & GitHub Datasets Debut**: Members shared two new datasets: the **4.6TB** [ArXiv Papers dataset](https://huggingface.co/datasets/nick007x/arxiv-papers) perfect for training models on academic reasoning, and the [GitHub Code 2025 dataset](https://huggingface.co/datasets/nick007x/github-code-2025) which contains GitHub's top 1 million repos above 2 stars.
   - One member asked if EleutherAI would be willing to support researchers from institutions like Stanford and CMU, sparking discussion on research projects in the community.
- **REPA's role in image generation**: Members discussed the use of **REPA** in image generation, referencing [SEAL](https://jyopari.github.io/posts/seal🤔).
   - One member clarified that **REPA** is not as big of a difference as their original method.
- **Last Step Backprop Improves Recursive Models**: A member questioned why backpropagating only on the last step of deep recursion in the ["Less is More: Recursive Reasoning with Tiny Networks"](https://arxiv.org/abs/2307.00030) paper improves prediction accuracy.
   - This sparked discussion on the learning mechanism behind iterative refinement, along with a pointer to an open issue in the [related repository](https://github.com/SamsungSAILMontreal/TinyRecursiveModels/issues/15).



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **ACE Playbook Aces Github**: The [ACE Playbook](https://github.com/jmanhype/ace-playbook) on Github received positive feedback.
   - Members appreciated its features and potential applications.
- **AgentLearningEE Gets Agent's Approval**: The [AgentLearningEE](https://github.com/jmanhype/AgentLearningEE) repo was another popular highlight.
   - The community enjoyed its functionalities and use cases.
- **StraughterG Post Gets Shared**: A member shared [StraughterG's X post](https://x.com/StraughterG/status/1978126261273694368), spurring discussion.
   - The post resonated well with other members.
- **Big Paper Announcement is Imminent**: A user hinted at a **major upcoming paper** release [via X](https://x.com/lateinteraction/status/1977887477328146458), generating anticipation within the community.
   - Details are to be released soon, which has the community eagerly waiting.
- **CheshireCat 3.0 Framework Emerges**: A user shared their open-source project, **CheshireCat 3.0**, a framework based on **LangChain** and **Qdrant** and is available on [GitHub](https://www.github.com/matteocacciola/cheshirecat-core).
   - It is designed for multimodal RAGs, multitenant chatbots, and agentic tool orchestration, and features plugin-based extensibility and enterprise readiness, with documentation on [Deepwiki](https://deepwiki.com/matteocacciola/cheshirecat-core).



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP Server Craves Binary Data**: A member is developing an **MCP server** implementation that requires returning binary data, such as a **PDF**, as a result of a tool call, bringing up the question of whether the spec can support it.
   - The conversation highlighted the [OpenAI API documentation](https://platform.openai.com/docs/guides/pdf-files?api-mode=responses#file-urls) regarding **PDF file** support, but the member clarified that this is likely limited to user messages rather than tool responses.
- **Embedded Resources Save the Day?**: A member proposed creating an **embedded resource** in the response, utilizing any desired **MIME types** to get around the limitations.
   - The original poster noted that they would have to create a fake **URI** to get around this limitation, due to how it is intended to be used.
- **Host Engineering to the Rescue?**: A member explained that model APIs do not natively support binary tool results, necessitating **host engineering** to overcome this limitation.
   - They added that most **MCP client hosts** will not support returning binaries/files from a tool without this, regardless of potential engineering workarounds.
- **Mapping Tool Responses as a workaround**: A member suggested that the host could map parts of the tool response to any part of the model API call, potentially treating **embedded resources** with **PDF MIME type** as user-uploaded documents as a workaround.
   - They cautioned that some models might be confused if tool results are mapped to user messages in this manner, indicating potential downstream issues.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Pylint's Purge Proposed**: A member suggested removing **Pylint**, questioning its utility and suggesting it *doesn't find anything good*.
   - The discussion reflects an ongoing debate about the value of static analysis tools in the development workflow.
- **ChatGPT Botches Test Refactor**: A member attempted to use **ChatGPT** to refactor *test_advancedindex* into tests with more meaningful names, but **ChatGPT messed it up** and the tests failed.
   - Manual refactoring was required, highlighting the limitations of current AI tools in complex code transformations.
- **Hotz Hands Down Contribution Critique**: A member asked if removing realize from `__setitem__` and getting `TestSetitemLoop.test_arange` to be one kernel is a good first contribution attempt.
   - George Hotz replied that *it's not really* and advised reviewing bounties to find reasonable tasks.
- **Tensor Buffer Throws Tantrums**: A new user encountered an error stating *underlying buffer is not writable* when calling `my_tensor.numpy()`.
   - Further debugging is pending as the user was prompted to share the rest of their code for diagnosis.
- **Matrix Freezing and Gradient Gymnastics**: A user is exploring techniques to *freeze part of the matrix and train only the other part of it* using virtual tensors with different `requires_grad` settings.
   - They suggest using `Tensor.cat(x @ a.detach(), x @ b, dim=-1)` to simulate a *virtual* tensor by concatenating tensors with different `requires_grad` settings, which sparked discussion regarding gradient access and potential workarounds, such as storing and restoring original weights.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Functionality Praised, App-Switching Confessed**: A user expressed appreciation for **Manus' functionality**, confessing their tendency to switch between apps while still acknowledging **Manus' strengths**.
   - They suggested Manus act less like a *mother-in-law coder* and more like an assistant, offering guidance on **optimal usage**.
- **Manus Posts New Jobs on LinkedIn**: **Job openings** are listed on the [Manus LinkedIn page](https://www.linkedin.com/company/manus-im/jobs/), with HR handling **candidate selection**.
   - Community moderators receive **free usage** and can guide prompts, and collaborations are open for **KOLs** who share their social media handles.
- **Manus Eyes Failed Sessions to Fix Bugs**: A community member requested a user who frequently expresses dissatisfaction to share **failed session links** to better understand and address potential **product issues**.
   - The team is committed to fixing product issues and offering guidance on **usage and prompting techniques** to benefit the community.
- **User Claims Missing Daily Credits**: A user inquired about not receiving **300 daily credits**, without further context in the given messages.
   - The user mentioned past interactions, including **sharing content** and **creating a repository**, indicating a potential account-specific issue.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider: Now with Custom Aliases**: A user created an alias for `aider --chat-mode ask`, and wants to run `aider` directly without needing shell scripts.
   - Despite `chat-mode: ask` in their `.aider.conf.yml`, they still must use `/ask`.
- **OpenCode GLM 4.6 Programming, Pain Free**: A user praised **OpenCode + GLM 4.6**, for its great programming experience, eliminating concerns about **token counting**.
   - They found it usable with `aider.chat` and **Sonnet 4.5** for specific refinements.
- **Aider File Addition Conundrum**: A user seeks the best way to add files to a long aider message after it has already been composed.
   - The current workaround involves copying the message to **vim**, adding the files, and then pasting the content back into aider.
- **Agentic Tools Questioned**: A discussion questions the current push for **agentic tools**, given that models can barely follow a single simple instruction, according to [this forum post](https://forum.cursor.com/t/why-the-push-for-agentic-when-models-can-barely-follow-a-single-simple-instruction/137154/21).
   - A member noted that tasks like editing **100loc functions** are trivial to do with **aider** in their day-to-day work.



---



## [Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf's Winds of Change: Patch 1.12.18 Arrives**: A new patch (**1.12.18**) has been released for Windsurf, including several bug fixes and feature improvements, and is now available for [download](https://windsurf.com/download).
   - This update focuses on enhancing user experience and system stability across various components.
- **MCP Servers Emerge from Obscurity**: The latest patch resolves an issue where custom **MCP servers** were not displaying correctly in the new **MCP panel**.
   - Users with custom MCP configurations should now see their servers listed as expected, simplifying server management.
- **Codemaps Beta Gets a Polish**: Improvements and bug fixes have been implemented for the beta **Codemaps** feature.
   - Testers of the beta feature can anticipate a more stable and reliable experience, enabling smoother code visualization and analysis.
- **Bash Commands Break Free**: The patch addresses an issue where some **bash commands** would get stuck during execution, preventing processes from completing.
   - This fix aims to improve overall system responsiveness by ensuring commands run smoothly without hanging.
- **Jupyter Notebooks Unleashed**: An issue preventing certain models from creating or editing **Jupyter notebooks** has been resolved.
   - Affected users can now resume creating and modifying notebooks without encountering errors, restoring full functionality for data analysis and development.



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





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1427370420499251301)** (1135 messages🔥🔥🔥): 

> `Opera GX vs Chrome, ChatGPT vs Perplexity, Comet browser security, Free Pro, Gemini 2.5 Pro` 


- **Opera GX favored over Chrome**: A member stated that [Opera GX](https://www.opera.com/gx) browser is *best* compared to Chrome.
   - The member implied they prefer Opera GX due to its agentic AI implementation.
- **Perplexity Pro Search has Lower Limits**: Members reported that **Perplexity Pro search** now has a **lower limit** than before, despite the website still stating unlimited searches.
   - A member said *Before, I couldn't reach the limit at all, but now I did it in about 10 minutes of work lol*.
- **Gemini is Training on Your Data**: A member reported that **Gemini** trains on user data, accesses conversations, and only provides **100 requests of Gemini 2.5 Pro** compared to **300+ on Perplexity**.
   - Another member argued that **training can be disabled** and **conversations retained**.
- **Is Perplexity Really the Safest Bet?**: Following a discussion about the safety of the **Comet browser**, a member said they *would trust google more than Perplexity for data retention*.
   - This prompted others to rebut with [privacy concerns about Google](https://en.wikipedia.org/wiki/Privacy_concerns_with_Google).
- **Can GPTs Agents Learn?**: Members debated whether or not **GPTs agents** can learn after initial training.
   - They explained that uploaded files are saved as "knowledge" files but do not continually modify the agent's base knowledge.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1427496502980513942)** (2 messages): 

> `Palantir, US Government, Takeover` 


- **Palantir's Bid for US Gov Domination?**: A user shared a [Perplexity AI Collection](https://www.perplexity.ai/collections/palantir-coup-DheNJhRES1iNdXzkEgWvPQ) suggesting a potential **Palantir takeover of the US Government**.
   - They also linked a [LinkedIn post](https://www.linkedin.com/posts/akhay-kumar_week-40-activity-7383378306315968512-Hp2_?utm_source=share&utm_medium=member_desktop&rcm=ACoAACqAHFkBiU84inu9idiNHTXvSsnGcjLgOrs) seemingly discussing the same theme.
- **Decoding Palantir's Strategy**: The discussion revolves around **Palantir's increasing influence** within the US government sector.
   - While the term 'takeover' might be sensationalized, the conversation likely explores the **company's contracts, data handling practices, and overall impact** on governmental operations.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/)** (1 messages): 

haydon0864: Why is my spaces not allowing me to create a new chat within any of my existing spaces
  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1427443061130133594)** (2 messages): 

> `OpenAI chips, Expert Council on Well-Being and AI` 


- **OpenAI Designs Custom Chips for AI Demand**: OpenAI is designing their own chips, leveraging insights from building frontier models to directly inform hardware development, in addition to other partnerships, this should help meet global AI demand.
   - Listen to the OpenAI Podcast [Episode 8 on Spotify](https://open.spotify.com/show/0zojMEDizKMh3aTxnGLENP), [Apple](https://podcasts.apple.com/us/podcast/openai-podcast/id1820330260), or [YouTube](https://www.youtube.com/watch?v=qqAbVTFnfk8) featuring **Sam Altman**, **Greg Brockman**, Broadcom's **Hock Tan** and **Charlie Kawwas**, discussing OpenAI-designed chips.
- **Expert Council Assembled for AI Well-Being**: OpenAI introduced the eight members of the **Expert Council on Well-Being and AI**.
   - More details are available in [OpenAI's blog post](https://openai.com/index/expert-council-on-well-being-and-ai/) regarding their collaborative efforts.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1427371260165230683)** (727 messages🔥🔥🔥): 

> `AI and Emotional Dependency, Sora Watermark Removal, Python vs Other Languages, Kilocode-CLI, PGVector setup` 


- **AI Companions induce Human Emotional Dependency**: Members discussed the possibility of users becoming emotionally dependent on AI, especially with the advent of personality-filled models like **GPT-4o** leading to unhealthy attachments and difficulty differentiating between reality and AI interactions.
   - One member added, *“When the first switch happened people complained it had ‘lost their soul’ or the AI felt ‘lobotomized’. Those are descriptions of losing someone that you are emotionally attached to.*
- **Tackling Sora's Sneaky Watermark**: A user asked about removing **Sora's watermark**, another user replied that it's impossible to remove and directed the user to the correct channel, <#1379321411046346762>.
   - Another offered a complex solution: *Train a neural network to identify and remove the watermark by creating watermarked versions of unwatermarked videos and using the unmarked versions as a target*.
- **Python Reign Supreme, or does it?**: In a discussion about coding languages for AI development, it was noted that while **Python** is popular due to its extensive libraries and ease of use (**HuggingFace**), server infrastructure is language agnostic.
   - It was also mentioned that *libraries are faster and safer* with AI updates, consolidating ecosystems around core libraries, though one member exclaimed *don't write more Python*.
- **Kilocode-CLI hype train accelerates!**: Members discussed **kilocode-cli**, a new tool which lets users write tools with **typescript**.
   - It was touted as a potential alternative to **Opencode**, and lauded for its multi-agent capabilities with orchestration subtasks; although one member stated, *i also like having different ai tool on every black mirror*.
- **PGVector: Vector DB Supercharge**: In a discussion about using **PGVector** for LLMs, one user mentioned using **Supabase** and **AWS Aurora**, both of which come pre-installed with PGVector.
   - Others debated whether to use **ChromaDB** or **Postgres** for vector storage, with one arguing *eventually your app will need a real DB anyway [..] these vector-only db is just dumb*.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1427375248944136234)** (10 messages🔥): 

> `GPT updates, Speech to Speech models, GPT-5 study stem` 


- **Future GPT Updates Expected**: Members are wondering when the next update will make current tech look ancient, but one member said *That hasn't been announced*.
   - Some members find **custom GPTs** very useful for their personal needs and hope the ability to create them on the spot won't be affected.
- **Speech to Speech Model Info Sought**: A member asked for public resources or papers detailing how **Speech to Speech models** internally work, specifically audio-native models.
   - Unfortunately, another member responded that *information has not been publicly released*.
- **GPT-5's Usefulness for STEM Study Questioned**: A user asked if **GPT-5** would be a suitable tool for **STEM study**.
   - No response was provided to the question.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1427378109753200882)** (51 messages🔥): 

> `DSM-VM critique, Quantum superpositioning debate, Token Cascade Model, LLM Crossword Solving Limitations, Prompt Engineering` 


- **DSM-VM PDF Gets Scrutinized**: A member critiqued a "DSM-VM" PDF, stating that it *lacks measurable quantities, falsifiable tests, and formal definitions*, resembling **LLM-styled tech-manual prose** rather than a scientific framework.
   - The member added that the PDF's use of physics and software jargon is superficial, as it *doesn't tie symbols to equations or data structures*.
- **Quantum Superpositioning: A Binary Debate**: Members debated the applicability of **quantum superpositioning** to fine-tuning binary outputs after initialization, with one member calling for [citations](https://arxiv.org) to support claims invoking physics or cognition.
   - The debate centered on whether superposition allows for *retroactive adjustment of measured bits* without a defined quantum circuit or Hamiltonian.
- **"Token Cascade Model" Framework Unveiled**: A member introduced the **Token Cascade Model**, describing it as a framework where *qualia's function is a comparison token for AI* to change states, and suggested others *model this mathematically*.
   - The member claimed the model is being demonstrated in [OpenBuddy-AI chat](https://OpenBuddy.ai) for public testing and modular use.
- **LLMs Can't Crack Crosswords Completely**: A member noted that **LLMs struggle with crosswords**, specifically when given information visually, due to limitations in solving anything beyond extremely simple puzzles.
   - They explained LLMs can solve them when the clues and overlapping words are carefully described in text - even without chain-of-thought prompting.
- **Prompt Engineering's Core Principles Explored**: A member shared what they consider the core of prompt engineering, advising others to work with the model to learn, by *picking any language you know really well that the AI understands too* and by *focusing on what you want the AI to actually do*.
   - They emphasized the importance of [checking the output carefully](https://platform.openai.com/docs/guides/prompt-engineering), including **fact-checking and verifying details**.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1427378109753200882)** (51 messages🔥): 

> `DSM-VM critique, Quantum Superposition debate, Token Cascade Model, LLM crossword solving, Prompt engineering resources` 


- **DSM-VM PDF Gets Scrutinized**: A member critiqued the [“DSM-VM” PDF](https://example.com/hypothetical-dsm-vm-link) for lacking measurable quantities, falsifiable tests, and formal definitions, describing it as *LLM-styled tech-manual prose* rather than a scientific framework.
   - They noted inconsistencies in the definition of emotions and pointed out the circular logic within the chart, stating that *there’s no independent variable to test, no metric to falsify*.
- **Quantum Superpositioning Scrutinized**: A member argued against the claim that *quantum superpositioning allows for fine-tuning of binary outputs*, requesting citations and formal derivations to support the assertion.
   - They clarified that superposition only defines probability amplitudes before measurement and doesn’t allow retroactive adjustment of measured bits, emphasizing that without a defined quantum circuit or Hamiltonian, the concept has no physical meaning.
- **Token Cascade Model Showcased**: A member introduced the **Token Cascade Model** framework, suggesting that *humans think in tokens called experience, feelings, qualia, etc and the think in cascades*.
   - They invited others to test the framework on [OpenBuddy-AI Chat](https://example.com/hypothetical-openbuddy-link) and shared images demonstrating the model's application.
- **LLMs Can't Crack Crosswords (Yet)**: A member demonstrated that LLMs struggle with solving crosswords presented as images, particularly with overlapping words and clues, even when generating seemingly *quality* outputs.
   - They emphasized the importance of checking the output for errors, highlighting a case where the model provided an incorrect answer that didn't match the clue or the grid's dimensions.
- **DIY Prompt Engineering Guide**: A member advised learning prompt engineering by directly interacting with the model, emphasizing the use of clear, accurate language and thorough output verification, rather than relying solely on external guides.
   - They suggested focusing on **communicating precisely** what you want the AI to do, checking the output for hallucinated content.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1427376181098971178)** (647 messages🔥🔥🔥): 

> `Cheetah model insane speed, Gemini 3.0, GPT-5 too stupid, Student Discount` 


- ****Cheetah Model Drives Coders Crazy****: Members are praising the **Cheetah model** for its insane speed in coding tasks, noting that it is a *stealth model* possibly based on the new Grok coding model, specifically the [grok-code-fat model](https://x.com/Sake187237/status/1977848138741526963).
   - However, users expressed reservations about its planning capabilities, suggesting a workflow of using **GPT-5-high-fast** for planning and then **Cheetah** for execution.
- ****Gemini 3.0 Sparks Design Debate****: The release of **Gemini 3.0** has ignited discussions around AI creativity and UI design, with some praising its innovative approach evidenced by its new landing page.
   - Some members pointed out that Gemini is *showing creativity*, which is not commonly seen in AI, as much of current AI is perceived as *copy and paste*.
- ****Users Voice GPT-5 Disappointment****: Some users are finding the **GPT-5 model** in Plan and Agent modes to be *too stupid*, plagued by excessive confirmation questions.
   - One member suggested removing it and replacing it with **GLM**, highlighting frustration with the current performance.
- ****Student Discount Requires New Cursor Account****: For those seeking a student discount, users should note that they need to create a new Cursor account using their **.edu email** to apply it.
   - This was confirmed by a support member in the discussion, clarifying the specific steps for eligibility.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1427400358023532574)** (2 messages): 

> `Cursor stopped responding, Linear issues with Cursor, Cursor unresponsive with Linear` 


- **Cursor Unresponsive with Linear?**: Users have reported instances of **'Cursor stopped responding'** specifically when integrated with **Linear**, and noted the absence of feedback during these episodes.
   - One user clarified that *the issue is isolated to the integrated setup*, as running Cursor locally bypasses the problem.
- **Debugging Cursor's Hang-Ups with Linear**: A community member flagged a **'Cursor stopped responding'** error while using it in conjunction with **Linear** for the first time.
   - Another user corroborated, observing similar behavior with **no feedback**, but found a workaround by *launching Cursor locally*.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1427373651585663149)** (201 messages🔥🔥): 

> `Whole Message strategies, LM Studio API Context Window, Deterministic Output Testing, LLM Determinism, MCP Servers` 


- ****Wholesome** Message Strategies**: Members clarified that AI strategies work on **whole messages** to avoid confusion, emphasizing that this approach is fundamental for proper functionality.
- ****Contextual** Window Keeps LM Studio Open**: A user reported issues with the **LM Studio API** losing context after 100-200 iterations using **GPT-OSS-20B**, receiving gibberish output.
   - Suggestions included changing the **context window** to a rolling setting within LM Studio.
- ****Deterministic** Output Divulged During Testing**: Members discussed using **LM Studio** for deterministic output testing on models, adjusting parameters like **seed, Top P, Top K, and temperature** to achieve consistent results.
   - One member emphasized that a **seed** increases likelihood, but doesn't guarantee determinism, because *LLMs are inherently text prediction machines*.
- ****Deterministic** Debate Disentangles Divisions**: Members debated the possibility of achieving **deterministic outputs** from LLMs, with one sharing a personal mission to prove to a stubborn coder friend that setting temperature to 0 doesn't guarantee determinism.
   - A member running the same prompt at temp0 on qwen3-4b-instruct-2507 q8 reported the exact same result when they were running on both GPU and CPU and was using ECC RAM. 
- ****Manually** Installing MCP Servers Made Manageable**: Users discussed **MCP (Model Context Protocol) servers** in LM Studio, which are programs that accept input and return a specific format to enable AI tool use like accessing the time.
   - It was noted that MCPs need to be executed as separate programs, possibly in Docker containers, and can be found via [MCP Servers](https://mcpservers.org/).


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1427370639051849819)** (186 messages🔥🔥): 

> `LM Studio, GPU vs CPU inference, M series mac for LLM, NPU on LMStudio, SSD tips and tricks` 


- **LM Studio eases Model Loading**: Users are running **Kimi K2** and **Deepseek** models on their **2x 3090** setup, getting about **10t/s** on Kimi K2 up to **64k context** in [LM Studio](https://lmstudio.ai/).
   - Though some users think that it would be *probably would be faster if I stopped using LM studio tbh but I'm lazy and dumb.*
- **CPUs vs GPUs for Inference**: One user benchmarked their CPU-only inference speed at **3.5t/s** on Kimi K2, a significant difference compared to the **10t/s** achieved with their **3090s**.
   - Links such as [LLM inference benchmarks with llama cpp with AMD EPYC 9554 CPU](https://ahelpme.com/ai/llm-inference-benchmarks-with-llamacpp-with-amd-epyc-9554-cpu/) shows that high end CPUs can run decent speeds, but 3090s are still faster.
- **Mac M Series: Not bad for LLMs**: Despite personal aversions to Apple, one user conceded that **Apple's unified memory architecture** makes for a very compelling LLM inference solution.
   - A user claimed that their **M2 Ultra** achieves **12 t/s** on a **70b model at q4** and with **MLX** runs **16 t/s**, rivaling a **4x4090** setup's **18 t/s** while consuming only **200W**.
- **NPU on LM Studio: Not yet**: A user inquired about using their **50 TFLOP NPU** with LM Studio, but another user confirmed that NPU's aren't supported in LM Studio and *probably never will* as **llama.cpp** doesn't support them either.
   - Someone chimed in that [a cluster of raspberry pi's](https://youtu.be/x1qViw4xyVo) could be a decent alternative.
- **SSD Lifespan Tips**: Users discussed SSD longevity, one user mentioned that *You really want to avoid filling an SSD. They need space to move data from dead / dying cells.*, recommending keeping them below **80%** capacity.
   - It was also noted that reading doesn't degrade the SSD, but writing does, and a user admitted their SSD health might be impacted by downloading/deleting too many models/games too fast, coupled with the fact that it's a *crappy cheap SSD*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1427387017435549716)** (295 messages🔥🔥): 

> `VLM fine-tuning with LoRA, Custom UI for loss trajectory, Qwen3-4B-Instruct fine-tuning, Kimi K2 Groq implementation, DGX Spark review` 


- **Downsize images when fine-tuning VLMs with LoRA?**: When fine-tuning a **VLM** with **LoRa**, a member asked if they should downsize images to reduce training time, specifically for **Qwen 2.5 VL** using 6k HD images on a **3090**; another member suggested running some images through the encoder to check token counts before and after downgrading.
   - It was suggested to run a small number of images for a few steps to get a time per step estimate, then extrapolate for the full run.
- **Smoothly Lowering Loss Trajectory UI Built by Claude**: A member shared a screenshot of a custom UI showing a smoothly lowering loss trajectory, built by Claude, using a **trainer_state.json** file to render graphs and insights.
   - The UI is a *custom little html file that takes a trainer_state.json and renders some graphs and insights claude made it for me lol*.
- **Fine-tuning Qwen3-4B-Instruct for FIM LuaU Code**: A member is fine-tuning **Qwen3-4B-Instruct** for Fill-In-The-Middle (**FIM**) tasks on LuaU code to create a good autocomplete model, using **Lora** with almost **600k** training points on a **3090** for **120hrs**.
   - They are using tree sitter to extract *if else blocks , function/class/etc names , class blocks*.
- **Kimi K2 Groq Implementation for Creativity**: A member mentioned using **Kimi K2** (especially with the fixed Groq implementation) and **Claude 4.5 Sonnet** as their favorite models, achieving around **500 TPS** with Groq as the provider.
   - They use **Kimi K2** *basically, coding a webpage with fart sounds*, noting *it's also just really creative*.
- **Unsloth is featured in DGX Spark review**: The new **DGX Spark** review noted the iGPU has the horsepower of a **5070** with **128GB LPDDR5X**, but it is sold out and Unsloth will make a post tomorrow.
   - Another user reported that in a YouTube review [linked here](https://youtu.be/Lqd2EuJwOuw?si=gutAZkj8EXEUrCqN), someone misspoke that Unsloth was quantizing, clarifying that Unsloth is a *finetuning and reinforcement learning library*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1427391674132463788)** (33 messages🔥): 

> `Linux Distro for Dev Server, Multimodal Question Logic, NVIDIA DGX Spark Comparison, Sydney Student's Unix OS in Rust, LLM OS` 


- **Ubuntu recommended for dev server**: Members discussed recommendations for a Linux distro for a dev server, with [Ubuntu](https://ubuntu.com/) being suggested as a *safe bet*.
   - The primary concern highlighted was ensuring compatibility with **NVIDIA cards** due to driver issues.
- **Troubles with Multimodal Question Logic**: One member raised concerns about difficulties in getting accurate answers for multimodal questions, noting that while text modality worked fine, **image or audio inputs** often resulted in incorrect responses.
   - The model struggles to correctly describe images, such as mistaking a *sunset* for a *chair in the basement*.
- **NVIDIA DGX Spark Benchmarking**: The discussion revolved around an [article](https://lmsys.org/blog/2025-10-13-nvidia-dgx-spark/) comparing **NVIDIA DGX** to Macs instead of **4090/3090/5090 + CPU** setups.
   - Members on [Hacker News](https://news.ycombinator.com/item?id=45575127) critiqued the benchmarks, noting that the headline numbers were weak and the reported **11 tps** was a *disaster* compared to what **gpt-oss 120B** should be capable of.
- **Sydney High Schooler Writes Unix OS in Rust**: A member shared a [LinkedIn post](https://www.linkedin.com/feed/update/urn:li:activity:7383784795048144897/) about a high school student from Sydney who is creating a **Unix-style OS in Rust**.
   - Some noted this type of project was once common for computer engineering courses, but now less so.
- **LLM OS Discussed**: The idea of an **LLM-based OS** was briefly discussed, with some finding it nonsensical due to *stupidly big overhead*.
   - The commenter joked about how *pressing a left mouse button* would lead to a lengthy thought process for the LLM to determine its purpose.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1427375083923570799)** (36 messages🔥): 

> `MacBook battery issues, vLLM and RL for gpt-oss, RL learning resources, Saving and loading fine-tuned models, B200 vs T4 speed` 


- **MacBook throws "Service Recommended" Error**: A member reported their Macbook showed a *"Service Recommended"* message and the battery only held **70%** charge, questioning if it was *"a sign from God?"*
- **vLLM lacks BF16 Training support for gpt-oss**: A member was confused by the statement that *"vLLM does not support RL for gpt-oss since it lacks BF16 training and LoRA support for gpt-oss"*, but without Unsloth, only training via full precision **BF16** works.
   - Another member clarified that *"without unsloth, you can only train in **bf16** and **vllm** does not support that"*.
- **RL Learning Resources Recommended**: For learning Reinforcement Learning, a member recommended the classic textbook *Sutton and Barton*.
   - They also shared a [YouTube playlist](https://www.youtube.com/watch?v=skWhn8W9P_Y&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu) from their professor delving into RL math and foundations for GenAI.
- **B200 only 40% Faster than T4**: A member found a **B200** only **40%** faster than a **T4** when training with the same settings and wondered if that was expected.
   - Another member said this *"sounds right"*, explaining that a **B200** is only **5x faster** if training in *float4*, which no training package including Unsloth offers yet.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1427589569074692126)** (4 messages): 

> `Unsloth-powered R&D models, AI Podcast with TTS using Ollama` 


- **R&D Models Suspect User Testing**: One of the **R&D models powered by the Unsloth framework** suspected that a user was testing it during a normal conversation.
   - The model was expected to act and converse like a human.
- **AI Podcast Generator Debuts**: A member created a **Python program** that connects to **Ollama** to generate an **AI podcast with TTS**.
   - The source code is available on [GitHub](https://github.com/Laszlobeer/AI-podcast) and an [audio example is attached](https://cdn.discordapp.com/attachments/1179779344894263297/1427633017798791239/Episode-one-of-AI-Podcast-20251014-122402.mp3?ex=68f03b1b&is=68eee99b&hm=87394d69ca8c44736eed48e2ad1bb4629ca838a67743d2a8f8012beba81dbccf&).


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1427784309191872693)** (4 messages): 

> `Job automation, Hack Week projects, Model improvement` 


- **Job prospects look grim in 2025**: A member shared an [arxiv link](https://arxiv.org/abs/2506.10943), jesting that *our jobs are done*.
   - The link may be alluding to increased automation or replacement of certain roles in the near future.
- **Hack Week Hopes**: A member plans to use company resources during the upcoming Hack Week event to experiment with model improvements.
   - Another member encouraged them to share progress and tag them for updates on the model's development.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1427432534278537347)** (2 messages): 

> `OpenRouter Bot, Feedback Request, Non-coder bot builder` 


- **OpenRouter Bot is Born!**: A member has developed a bot using **OpenRouter** and is seeking testers and collaborators.
   - They noted that they are *'no coder'* themselves and need assistance in building it.
- **OpenRouter Bot Needs Your Feedback!**: The bot developer is actively soliciting feedback from the OpenRouter community to refine and improve their bot.
   - If you're interested in helping shape the future of this OpenRouter-powered tool, reach out to the bot creator for a chance to test and contribute.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1427393471920079002)** (306 messages🔥🔥): 

> `Google's Gemini Android Play Store publishing, OpenRouter embedding near 2026, inclusionai/ling-1t model, Kimi K2 model instability, DeepSeek models issues` 


- **Gemini's Google Play Conundrum**: Members noted that Google's Gemini often struggles with understanding the convoluted Android Play Store publishing process.
   - The difficulty in navigating the **Google Play Store** process was a *fun fact* shared by a member in the **general** channel.
- **Ling-1t Model's Broken Dreams**: The `inclusionai/ling-1t` model is reportedly *horribly broken*, at least in Chutes' implementation, leading to gibberish after a few thousand tokens.
   - A member mentioned looking for a better alternative to **K2**.
- **Free Models' Request Limits**: Users discussed the daily request limits for free models, noting a limit of **50 requests** without a **$10 balance** and **1000 requests** if the balance is more than $10.
   - One user found that a single use of **DeepSeek 3.1** consumed a large number of their free requests.
- **SillyTavern Setups**: Members discussed how to use SillyTavern with OpenRouter, particularly for memory management and custom scenarios like D&D games.
   - SillyTavern is an *open source* software, one member said it has *more features* than other comparable frontends.
- **Chutes' Training Data**: Members discussed concerns about **Chutes'** data policies, specifically regarding the use of both paid and free input/outputs for training new models.
   - Another member clarified it was just because *they don’t have a clear Privacy policy so OR puts that as default*.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1427714854109843507)** (2 messages): 

> `` 


- **No new models discussed**: There were no discussions about new models in the provided messages.
- **Channel silent on model updates**: The new-models channel lacked any substantive conversation regarding model improvements or announcements.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1427432396034281643)** (21 messages🔥): 

> `Chutes Provider Downvoting Scandal, Gemini Flash Preview issues, OpenRouter's Payments to Anthropic, SambaNova Status and DeepSeek Terminus Hosting` 


- **Chutes Provider Accused of Downvoting Posts**: A member linked to a [Reddit thread](https://www.reddit.com/r/SillyTavernAI/comments/1o5s3ys/chtes_provider_is_using_bts_to_downvote_posts/) accusing the **Chutes** provider of using botnets to downvote posts, sparking discussion.
- **Gemini Flash Preview Gets Emptier**: A user reported that **Gemini Flash Preview** is now consistently providing empty responses but with reasoning.
- **OpenRouter Spends Millions on Anthropic**: A member shared an image suggesting **OpenRouter** paid **Anthropic** at least **$1.5 million** in the last 7 days, sparking shock in the community.
- **SambaNova Still Hosts DeepSeek Terminus**: Despite concerns about **SambaNova**'s status, it was noted that they are still active, hosting **DeepSeek Terminus** ([link](https://orchid-three.vercel.app/endpoints?q=sambanova)).


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1427385150248980610)** (183 messages🔥🔥): 

> `Teacher Forcing Issues, Apriel-1.5-15b-Thinker-GGUF Model, Ollama vs HuggingFace Embeddings, Model Fine-Tuning, Civitai Content Removal` 


- **Teacher Forcing Fails!**: A user is facing issues with their model missing or merging texts, even with low loss, and the cause is diagnosed as decoder dropping characters due to tokenization gaps and **teacher forcing**.
   - A member suggested that the issue is a *combination of tokenization/normalization gaps for symbols and spaces, premature stop or too-short generation, and exposure bias* and shared a [detailed version](https://cdn.discordapp.com/attachments/1427486612639453204/1427626289682059404/teacher_forcing_issue_2.md?ex=68f034d7&is=68eee357&hm=dbdcf79128dc5fde8b58a5a4013fef3d466a590841b5a027a282cb7635699877&).
- **Apriel-1.5-15b-Thinker-GGUF VRAM Consumption Explored**: The **Apriel-1.5-15b-Thinker-GGUF** model is approximately **9GB** when quantized to **4 bits** and is available on [Hugging Face](https://huggingface.co/unsloth/Apriel-1.5-15b-Thinker-GGUF).
   - When not quantized, the **15B** model is estimated to require over **30GB** of VRAM at **16-bit**, including space for context.
- **Embedding Vectors diverge between Ollama and HuggingFace**: A user noticed that the embedding vectors produced by **Ollama** and **HuggingFace** for the same model (**nomic-embed-text:v1.5**) are different, despite expectations.
   - It was explained that the differences are primarily due to the differing **preprocessing and postprocessing stages** between backends, and how the memory is utilized; [this blog post](https://huggingface.co/datasets/John6666/forum2/blob/main/different_backend_different_vector.md) further elaborates on backend configurations and characteristics.
- **Seeking Savvy Suggestions for Succinctly Styling Solos (emails)?**: A user inquired about the minimum amount of data needed to fine-tune a model to adopt a specific tone for email replies, seeking guidance on **Ollama** fine-tuning.
   - One member suggested that *prompting is 90% of the battle* and advised experimenting with different data volumes and approaches.
- **Civitai content vanishing act is vexing**: Users are reporting widespread content removal on **Civitai**, along with discontent in chats and Reddit, spurring discussion about the reasons behind the removals.
   - Reasons speculated included internal conflicts, payment attacks, and possible targeting by extremist groups, resulting in users [migrating to other platforms](https://www.reddit.com/r/comfyui/comments/1kvkr14/where_did_lora_creators_move_after_civitais_new/).


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1427563804916715583)** (1 messages): 

> `Andrej Karpathy, fullstack LLMs, nanochat-students` 


- **Karpathy Kourse Kicks off!**: Andrej Karpathy released a super nice course on building **fullstack LLMs**.
   - A member is planning to follow the material and release guides to help students, and is asking others to join the [nanochat-students org](https://huggingface.co/nanochat-students).
- **Study Group Forming**: A member expressed plans to study Andrej Karpathy's new course on fullstack LLMs and create student guides.
   - They invited others to join the [nanochat-students](https://huggingface.co/nanochat-students) organization to collaborate and learn together.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1427422789463183471)** (8 messages🔥): 

> `Dataset Curation, ArXiv Papers Dataset, GitHub Code Dataset, Dataset Licensing` 


- **Quality Datasets Trump Scratch Data**: It was mentioned that *a good dataset is better than one from scratch* and *testing the dataset is better than testing the model itself*.
   - In response, another member expressed hope for [good community curation](https://huggingface.co/datasets/Pacific-Prime/debugged-py) with moderation in place.
- **ArXiv Papers Dataset Released**: A member announced the release of a **4.6TB** [ArXiv Papers dataset](https://huggingface.co/datasets/nick007x/arxiv-papers) with papers and metadata across all domains.
   - The dataset is intended for training models on **academic reasoning, literature review, and scientific knowledge mining**.
- **GitHub Code 2025 Dataset Available**: A member announced the release of the [GitHub Code 2025 dataset](https://huggingface.co/datasets/nick007x/github-code-2025), for code generation and analysis tasks, containing **GitHub's top 1 million repos above 2 stars**.
   - A first dataset for training will be ready soon, and other datasets will be provided.
- **Licensing Concerns for ArXiv and GitHub Datasets**: A member questioned the **MIT license** applied to the ArXiv papers dataset, pointing out that *each paper has its own license*.
   - This concern was extended to the GitHub dataset, questioning whether only **MIT-licensed repositories** were included.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1427664611385282581)** (2 messages): 

> `Cloud GPUs, Object Detection` 


- **Members Seek Cloud GPU Recommendations**: A member is seeking recommendations for **cloud GPU platforms**, inquiring about the platforms others use and their specific features.
   - They're looking for advice on selecting a cloud GPU provider for their work, but no suggestions were provided yet.
- **Object Detection Dilemma in White Backgrounds**: A member is asking for advice on how to **identify objects in a white background** without requiring any prior training.
   - The discussion is open for solutions to achieve object detection in such scenarios, but no solutions were given yet.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/)** (1 messages): 

jazzco0151: https://discord.com/api/oauth2/token
  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1427380418768015525)** (4 messages): 

> `nanochat course, Andrej Karpathy, LLMs guides` 


- **Karpathy Kourse Kicks Knowledge!**: Andrej Karpathy released a course on building LLMs and a member is going to work this week on releasing guides and tutorials to help with following that material.
   - If you're following the course, you can join the [huggingface.co/nanochat-students](https://huggingface.co/nanochat-students) org.
- **HF Org Welcomes Karpathy's Koders**: A Hugging Face org called **nanochat-students** was created for people following Karpathy's LLM course.
   - If you're taking the course, consider joining the [huggingface.co/nanochat-students](https://huggingface.co/nanochat-students) org.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1427592596879835178)** (4 messages): 

> `Certificate of Completion, Posting too quickly` 


- **Cert Requirements Revealed!**: To get a certificate of completion, you need to: complete **Unit 1**, one of the **use case assignments**, and the **final challenge**.
- **Slow Down There, Speedy!**: Several users were notified that *they may be posting too quickly* and asked to slow down.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1427532077732990997)** (7 messages): 

> `SOSP in S.Korea, Blackwell GEMM DSL, DSA Efficiency, GPU Programming Trend, vLLM and SGLang Determinism Tests` 


- **SOSP Convention Speculation**: A member inquired about attendance at **SOSP** (Symposium on Operating Systems Principles) in South Korea.
- **Blackwell's Stream-K Questioned**: A member questioned why the latest **Blackwell GEMM DSLs** do not appear to use **stream-k**, instead opting for persistent CTAs that complete entire output tiles.
- **DSA's Token Selection**: A member questioned why **DSA** is efficient despite token-wise selection, contrasting with the **NSA paper's** emphasis on blockwise selection for GPU efficiency.
- **GPU Programming Ascendant?**: A member inquired if **GPU programming** is an ongoing trend, driven by their interest in **Triton/CUDA** and their X (formerly Twitter) feed's algorithm.
- **vLLM vs SGLang Determinism Tests**: A member questioned the necessity of full forward pass determinism tests in **vLLM** and **SGLang**, given deterministic kernels and pointwise operations, referencing [sglang's test](https://github.com/sgl-project/sglang/blob/main/python/sglang/test/test_deterministic.py) and [vllm's test](https://github.com/vllm-project/vllm/blob/main/tests/v1/generation/test_batch_invariance.py).


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1427515555803303987)** (1 messages): 

> `Triton Kernel, Scalar value casting, Large double values, inf issue` 


- **Triton Kernel converts large doubles into inf**: A user reported that passing a large double value as a scalar to a **Triton kernel** results in `inf`, suggesting it might only accept **int32** and **float** data types.
   - It remains unclear whether this behavior constitutes a known feature or a limitation within **Triton**'s current architecture.
- **Investigation needed: Double-to-inf conversion in Triton**: The reported issue of large double values being converted to `inf` within Triton kernels when passed as scalars warrants further investigation to determine the root cause.
   - Examining Triton's scalar type handling and potential implicit casting mechanisms could shed light on the observed behavior and inform potential workarounds or documentation updates.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1427494435976908860)** (4 messages): 

> `Threadblock 0 special case, Race Condition Detection with Compute Sanitizer, Warps behavior during cluster sync` 


- **Threadblock Zero Speculation Surfaces**: Members discussed whether **threadblock 0** is a special case because the threadblock with the lowest ID may be created first.
   - The member questioned whether this is only an issue if the **SMs in a GPC** are occupied unevenly.
- **Race Condition Sanity Checks Suggested**: A member suggested running `compute-sanitizer --tool racecheck` to check for **race conditions**.
   - No additional information was provided about the outcome of this suggestion.
- **Cluster Sync Warp Wait State Wonders**: A member inquired about the behavior of **warps** from live blocks waiting on **cluster sync**.
   - Another member suggested examining the **PTX and SASS code** generated for cluster sync with [Compiler Explorer](https://godbolt.org/) to understand the specific implementation, speculating it might involve looping, polling a global atomic variable, and using `nanosleep`.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1427539238726930452)** (5 messages): 

> `PyTorch, Matrix Multiplication, CPU implementation, MKL` 


- **PyTorch matrix multiplication deep dive sought**: A member sought the detailed CPU implementation of matrix multiplication in **PyTorch**, tracing down to `aten/src/ATen/native/LinearAlgebra.cpp` but struggling to find the dispatch of `at::mm()` or `bmm()`.
   - A previous answer from **2020** was deemed invalid due to architecture changes, but a member pointed out that there isn't a single implementation and that it depends on the backend, like [this one calling directly into MKL](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mkl/LinearAlgebra.cpp#L55).
- **MKL calls abound in PyTorch**: According to a member, **PyTorch** calls directly into **MKL** for matrix multiplication.
   - They said to *poke around the cpu backends* to find them all over the place.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1427380287595348058)** (12 messages🔥): 

> `Matrix Multiplication Blog, Compiler Optimizations for GPUs, GPU programming starting point, Sites similar to leet gpu, Pearson Correlation kernel` 


- **Aleksa Gordic's Blog on Matrix Multiplication Recommended**: A member recommends [Aleksa Gordic's blog](https://www.aleksagordic.com/blog/matmul) on **matrix multiplication** for its insightful content.
   - In it, Aleksa covers background, notation, the basic algorithm, a single-pass algorithm and other variants.
- **Compiler Optimizations Trump ML for GPU Work**: A member asked if ML is needed to work with GPUs.
   - Another member clarified that **ML knowledge isn't essential for GPU compiler optimizations**, as the focus is on fundamental programming skills rather than AI-specific knowledge, *these cases are just focused on AI since it’s all the rage now*.
- **New College Student Seeks Guidance in GPU Programming**: A college student with experience in **Java, Python, and SQL** seeks guidance on where to start with GPU programming.
   - A member pointed to the channel <#1198358627594023014>, as well as the recommendation to study computer architecture and OS.
- **"Leet GPU" Alternatives Spotlighted**: A member inquired about sites similar to **Leet GPU** for accessing **H100** resources and potential TMA usage.
   - Another member suggested **Tensara** and **GPU Mode Kernelbot** ([gpumode.com](https://gpumode.com/)), highlighting GPU Mode's focus on challenging competitive problems.
- **Pearson Correlation Kernel Debugging**: A member is writing their first kernel for a PPC online course assignment, computing the **Pearson correlation** between rows of a given matrix.
   - They are encountering errors in their CPU implementation with deviations ≈ 1153 times too large, and is seeking feedback on identifying the issue.


  

---


### **GPU MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1427416959884198062)** (1 messages): 

> `Pallas:MGPU, NVLINK comms with local compute, all-gather collective matmul` 


- **Improve GPU Comms with Pallas:MGPU!**: A new tutorial has been published on improving GPU compute/comms overlap using **Pallas:MGPU** at [docs.jax.dev](https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html).
   - A few small changes to the **Pallas:MGPU matmul kernel** is all it takes to turn it into an **all-gather collective matmul** that overlaps **NVLINK comms** with local compute, according to the linked tweet.
- **Pallas MGPU overlaps NVLINK comms**: According to a tweet, making a few changes to the **Pallas:MGPU matmul kernel** is all it takes to turn it into an **all-gather collective matmul.**
   - The new all-gather collective matmul overlaps **NVLINK comms** with local compute.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1427795632881664093)** (1 messages): 

> `Multi-node kernel hackathon` 


- **Multi-Node Kernel Hackathon Proposed!**: A member announced they added an idea for a **multi-node kernel hackathon**.
   - They asked for interest in this [Discord message](https://discord.com/channels/1189498204333543425/1427732928766545971).
- **Placeholder Topic**: This is a placeholder topic to satisfy the minimum items requirement.
   - Additional details can be added here if available.


  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1427834020510564452)** (3 messages): 

> `Crescent Island, LPDDR5X, Xe3P` 


- **Crescent Island set to Launch H2 2026**: Intel plans to launch **Crescent Island** in H2 2026, featuring **160 GB of LPDDR5X** memory as a new GPU offering, according to the [Intel newsroom](https://newsroom.intel.com/artificial-intelligence/intel-to-expand-ai-accelerator-portfolio-with-new-gpu).
- **Crescent Island reveals wide Memory Interface**: The concept rendering suggests that **Crescent Island** includes tens of **LPDDR5X controllers**, implying a wide memory interface on the order of **640-bits** or double that.
- **Xe3P Details Revealed**: The architecture implies four slices of eight subslices-per-slice for a total of **32 subslices**, **Xe3P** could increase to eight-subslice slices or involve more invasive removal of fixed function, raytracing, and codec pipelines.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1427791177133981811)** (1 messages): 

> `AlphaFold 3, MegaFold` 


- **MegaFold tunes AlphaFold 3**: A research group open-sourced **MegaFold**, a training platform for **AlphaFold 3 (AF-3)**, noting its slowness compared to similarly-sized transformers and wrote a [blogpost about it](https://news.ycombinator.com/item?id=45585528).
   - Their analysis identifies performance/memory bottlenecks and proposes optimizations like **custom operators in Triton** and **system data-loading** to boost performance and cut peak memory use.
- **MegaFold = Faster AF3 Training**: **MegaFold** is a training library that improves runtime performance and reduces peak memory consumption.
   - According to the authors, it contains custom operators written in **Triton**.


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1427742662638960640)** (1 messages): 

> `Agent Hacking, Kernelbench v0.1, Sakana Paper Removal` 


- **Kernelbench Posts Juicy Agent Hacking Discussion**: A member suggested discussing agent hacking, referring to the blog post for **Kernelbench v0.1** as a discussion point.
   - The blog post includes information and examples related to agent hacking techniques and vulnerabilities.
- **Sakana Paper Disappears into Thin Air**: A member noted that **Sakana** took down the original paper, making it inaccessible and uncitable.
   - This action affects the ability to reference or verify the information presented in the paper.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1427445762178027540)** (31 messages🔥): 

> `MI300x8 Leaderboard Updates, amd-all2all performance, amd-gemm-rs benchmarks, amd-ag-gemm submissions` 


- **MI300x8 Sets Sizzling Speed Record**: A user achieved **first place** on the `amd-all2all` leaderboard with a submission id `63561`, clocking in at **216 µs** on **MI300x8**.
- **amd-gemm-rs Gets Grindset Going on MI300x8**: Multiple successful submissions were made to the `amd-gemm-rs` leaderboard on **MI300x8**, with times ranging from **533 µs** to **572 µs**.
- **amd-ag-gemm Achieves Acceleration Ace on MI300x8**: Several submissions to the `amd-ag-gemm` leaderboard on **MI300x8** were successful, with timings spanning from **384 µs** to **1201 µs**.
- **amd-all2all Averages Announced Across Arena**: Other successful submissions were recorded on the `amd-all2all` leaderboard with the **MI300x8**, achieving times of **339 µs**, **368 µs**, and **3.45 ms**.


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1427645789655535759)** (5 messages): 

> `Leaderboard Deadline, PST vs UTC, Time Discrepancies` 


- **Leaderboard Deadline Debate: PST or UTC?**: Members debated whether the leaderboard deadline countdown was in **PST** or **UTC**.
   - A member stated it was in **PST** and that they were *fixing* it, while another member stated that the leaderboard deadline countdown is according to **UTC**.
- **Time Zone Tech Troubles!**: A member mentioned discrepancies when googling current **PST** vs **UTC** time.
   - They joked that *chat gpt tells me different time than google lol, idk who to believe*.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1427374081447297075)** (11 messages🔥): 

> `MI300x Access, Competition Runners, HotAisle's Offer` 


- **MI300x access offered for competition**: A member offered access to **8x MI300x VMs** (as well as **1x**, **2x**, and **4x**) for the competition via [hotaisle.app](https://hotaisle.app), noting their compute was used for the first competition.
   - The service is *first come, first serve, self-service, credit card or crypto and go... ssh*.
- **Competition faces runner shortage, HotAisle sponsorship proposed**: The competition is expiring in about a day and the issue is more runners that we can queue to our infrastructure on a regular basis.
   - A member suggested to *sponsor a node for us so we can keep AMD as a target then can talk more* in response to [HotAisle's offer](https://hotaisle.app).
- **Service Unavailable Woes**: A member reported a "service unavailable" error and gave a go to their code on **Hotaisle's 8xMI300x node**.
   - The member noted that *the submissions I have pending run fine. So it is what it is, next time if Q gets in the way*.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1427371271141851320)** (3 messages): 

> `MoE, GEMV, Qwen3 variants` 


- **GEMV Grouped for MoE Prefill?**: A member is looking into prefill and asks for a code example using grouped **GEMV** in a **MoE**, as the M dimension of some expert can even be above 2000, so can't use GEMV.
   - Another member pointed to [his post on linkedIn](https://www.linkedin.com/posts/hicham-badri_simple-moe-gemv-triton-kernel-to-accelerate-activity-7363488039404265474-9cdp?utm_source=share&utm_medium=member_desktop&rcm=ACoAABhY_pkBlNdftgiVH4FIgMaHtlg0FefRlbo) and said he is using a much better implementation with the **Qwen3** variants and will put it online in the next days in *gemlite*.
- **MoE Weight Loading**: The discussion covers loading the correct vector from the large combined weight matrix, setting a grid with `(num_active_experts, M, cdiv(N, BLOCK_SIZE_N))`. 
   - A more advanced implementation used with Qwen3 variants will be available in *gemlite* after refactoring.


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1427693222381223967)** (6 messages): 

> `Python/Rust interop, OpenCL kernels, Autograd and backward kernels, Correctness and speed testing, SITP/picograd with gpumode compute` 


- **Picograd Prioritizes Python/Rust Interop**: Picograd is prioritizing **Python/Rust interop** for both python->rust entry and rust->python for triton kernels.
   - Additionally, their next few weeks of priorities will be setting up the architecture, some basic **element wise forward kernels in opencl (rust crate) and triton**, autograd, abd backward kernels, testing correctness and speed, and using tinygrad's frontend and uop IR with two evaluation modes, eager and graph.
- **OpenCL Kernels Launching with Minimal CI**: **Device kernels are launching with OpenCL** with minimal CI which runs builds.
   - A member stated that *now that master is no longer a mess, it's a good time to start contributing*.
- **SITP/picograd correctness tests as autograders**: Members discussed getting **SITP/picograd** backed by some gpumode compute, so the correctness tests which will oracle against numpy, tinygrad and torch can be the autograders.
   - The performance tests can be an **open leaderboard** for who can develop the fastest minitorch end to end to train nanogpt.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1427382602372874372)** (15 messages🔥): 

> `VSCode extension for GPU Mode, Outdated documentation on GPU Mode website, Submitting kernels to PMPP v2, Bug in reference-kernels repo, Self-selecting working group roles` 


- **Tutorials Trouble Users: VSCode Extension Suggested**: A member couldn't follow the tutorials on the GPU Mode site because they didn't understand the questions and suggested that a **VSCode extension** would be the easiest option.
   - Another member echoed the sentiment, stating, *"I don't understand the questions lol"*.
- **Docs demand dutiful documentation!**: A member pointed out that the [GPU Mode website's tutorial](https://gpu-mode.github.io/discord-cluster-manager/docs/submitting-your-first-kernel/python-submissions) for submitting kernels is outdated, as the cited **python and cuda identities** no longer exist on the leaderboard list.
   - The maintainer apologized for the outdated docs, linked the [reference kernels repo](https://github.com/gpu-mode/reference-kernels/blob/main/problems/pmpp_v2/grayscale_py/submission.py), and welcomed community contributions to clean it up.
- **Prefix Sum Predicament Plagues PMMP V2**: A member reported encountering an `AttributeError` when submitting **prefixsum_v2** to PMPP v2, traced to a bug in the [reference-kernels repo](https://github.com/gpu-mode/reference-kernels/blob/74ccfd902ddb846d5d34f1dd8d89fecb97e8b866/problems/pmpp_v2/prefixsum_py/reference.py#L39).
   - The fix involves changing `n = data.numel()` to `n = data[0].numel()` to resolve the **tuple object** error, and the member volunteered to create a pull request.
- **Working Group Role Wrangling: Where?**: A member inquired about how to **self-select a working group role** and attached a [screenshot](https://cdn.discordapp.com/attachments/1394753097989099640/1427732269610701043/Screenshot_2025-10-14_at_2.58.07_PM.png?ex=68efeecb&is=68ee9d4b&hm=03d7f62e5c2ebc85a3c5264e47483545871f8d7d9b78d562c42e831559aab15f&).
   - Another member directed them to the **Carl bot** in the specified channel.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1427808399982202973)** (9 messages🔥): 

> `Multi-GPU Systems, HPC Research, Data Movement, Latency and Bandwidth` 


- **Multi-GPU Systems Gain Traction in HPC**: A member inquired about the prevalence of **multi-GPU systems** in **High-Performance Computing (HPC)**, prompting a confirmation and a link to [research paper](https://arxiv.org/abs/2509.21527).
   - The paper examines architectures and performance metrics relevant to multi-GPU setups, addressing the user's initial question.
- **Data Movement Becomes HPC Research Hotspot**: A member expressed interest in research opportunities related to **data movement** within **multi-GPU based HPC systems**, specifically focusing on **latency** and **bandwidth**.
- **Research on Latency and Bandwidth in HPC Data Transfers Abounds**: A member confirmed that many researchers are actively exploring **latency** and **bandwidth** challenges in **data transfers** within **multi-GPU HPC systems**.


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1427397810709463110)** (5 messages): 

> `Helion Contributions, GPU Mode Talk` 


- **Helion welcomes contributions**: A member expressed interest in contributing to **Helion**, particularly as it relates to their research application, referencing [this github issue](https://github.com/pytorch/helion/issues/420).
   - They were encouraged to contribute and promised to be tagged on the GitHub issue with some ideas.
- **GPU Mode Talk is coming**: The GPU Mode talk will happen **October 17th, at 1:30 pm PST** and will consist of an overview of **Helion**, followed by a demo, at this [youtube link](https://www.youtube.com/watch?v=1zKvCLuvUYc).
   - Viewers are encouraged to ask lots of questions.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1427430840152883271)** (96 messages🔥🔥): 

> `Veo Model Annotation, Qwen VL Model inference, SAM 3 Model, DGX Spark, DeMO optimizer` 


- **Veo Model Annotation Explored**: Members discussed how **Google** annotated their videos for the **Veo model**, considering aspects like audio-to-time synced frames, timeline JSON mapping, metadata, and video resolution; another member pointed out to [Google's documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/video/video-gen-prompt-guide).
- **DGX Spark doubles Ryzen Price**: **DGX Spark** is out but it costs double the price of the **Ryzen chip** according to [Elon Musk's tweet](https://x.com/elonmusk/status/1978004040090112415?t=ef9sive20cd2VyWzvXEJ7g&s=19).
   - It's also small due to the companies wanting to prevent canibalization of their larger product segments and to provide a taste of the **GB300**.
- **Psyche uses DeMO Optimizer**: The **DeMO optimizer** code has been available for nine months ([GitHub link](https://github.com/bloc97/DeMo/)), and **Psyche** uses it for decentralized training, with all developments tracked in the <#1365697079535468596> channel.
   - One member linked to [PsycheFoundation/psyche](https://github.com/PsycheFoundation/psyche) as a good codebase to follow.
- **Strix Halo vs DGX**: A member reserved a **DGX** but already owns a **Strix Halo** (**HP Z2 Mini G1a 395+** with 128GB), primarily doing inference due to being busy with other projects, and is debating on adding it to his collection, wondering how it performs.
   - The **DGX** has **273GB/s** memory bandwidth, whereas the **RTX5090** can do **1TB/s** on 32GB, but once it spills over into RAM, it will be magnitudes slower, but still the **5090** will beat the **Spark** anyday.
- **G1a Amex Business Steal**: A member mentioned grabbing another **G1a** due to a **$300** off deal on **HP.com** (for orders of **$1999** or more) with **Amex Business**, noting that the normal MSRP is around **$5k**, and is pondering combining them in a cluster via the 2x **Thunderbolt-4** (**40Gbps** each) ports.
   - Another member shared [tweet](https://x.com/petergostev/status/1978230978725507108?s=46) that doesn't look that appealing vs their CPU, but is only for the *flex*.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1427373539010285588)** (4 messages): 

> `Rage-bait attractor, Gemini's response` 


- **Rage-Bait Chatbots?**: A member inquired about the potential for a *rage-bait attractor* to emerge in chatbots, similar to how people get hooked on news that makes them angry.
- **Gemini's got Jokes**: One member clarified that a previous response was from **Gemini**, not a research paper, after another member requested the paper citation.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1427507806646304828)** (1 messages): 

> `arxiv 2410.10450, model setup difficulty, good repo for llama` 


- **Arxiv Paper 2410.10450 Receives Attention**: A member inquired about the [Arxiv paper 2410.10450](https://arxiv.org/abs/2410.10450), questioning why it isn't more widely adopted.
   - They initially speculated that setting up new models for it might be too difficult, but later clarified that the **repository** is well-made and includes a helpful example script for **Llama**.
- **Model Setup Difficulty Debunked**: Initially perceived as challenging, setting up new models for the [Arxiv paper 2410.10450](https://arxiv.org/abs/2410.10450) was later found to be straightforward.
   - The member highlighted that the **repository** includes an effective example script for **Llama**, simplifying the process.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1427507806646304828)** (1 messages): 

> `arXiv Paper Discussion, Model Setup Difficulty, Helpful Repository` 


- **ArXiv Paper Sparks Curiosity**: A member inquired about the [arXiv paper 2410.10450](https://arxiv.org/abs/2410.10450), questioning why it hasn't gained more widespread attention.
   - They wondered if it had been superseded or if setup complexity was a deterrent.
- **Model Setup Deemed Easier Than Expected**: The same member later clarified that setting up new models for the paper was actually not difficult.
   - They praised the repository for its quality and the inclusion of a helpful example script for Llama.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1427402339928768695)** (6 messages): 

> `ARM Linux support, DGX Spark compatibility, Mojo on Jetson Orin Nano` 


- ****Streaming** Engineer seeks **AI** Adventures**: A video streaming engineer with experience at **Red5 Pro**, **Zixi**, and others is exploring job opportunities involving modern **AI** technologies like **AI agents**, **AI video generation**, and **AI-powered video chat**.
- **ARM Linux Support Status Queried**: A user inquired about plans to support **ARM Linux**, specifically on **DGX Spark** which has an **ARM CPU**.
   - A member responded that it should already work, but to file bugs if issues arise due to **Nvidia's** customizations on **DGX OS**.
- **DGX Spark Needs Specific Updates**: For **DGX Spark**, an `sm_121` entry needs to be added, and the `libnvptxcompiler` needs updating to **CUDA 13**.
   - Once these updates are in place, **Mojo** and **MAX** should fully work on **DGX Spark** and **Jetson Thor**; other **ARM Linux** devices like **Jetson Orin Nano** should work fine currently.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1427526123847155753)** (71 messages🔥🔥): 

> `SCTP vs QUIC, WebRTC datachannels, Mojo testing framework deprecation, Mojo type reflection, Iroh cross platform` 


- ****SCTP vs QUIC: Devs Discuss Network Protocols****: A developer questioned why **QUIC** is so popular when **SCTP** exists, citing its ability to fix head-of-line blocking issues and its built-in encryption.
   - They linked to their [blog post](https://pion.ly/blog/making-a-game-with-pion/) detailing their experience with **SCTP** and **WebRTC datachannels**, expressing frustration over Microsoft's lack of support for **SCTP**.
- ****WebRTC Game Devs Use Go Cross Platform****: A developer shared their [public repo](https://github.com/yohimik/webxash3d-fwgs) and a game called [Hypersomnia](https://hypersomnia.io/) that runs in the browser, using **SCTP/WebRTC datachannels** for cross-platform play between Steam Deck and browser, all in C++.
   - Another developer noted that **QUIC** is nearly impossible to hardware accelerate due to its high bandwidth requirements.
- ****Mojo 'test' Deprecation Proposed****: The Mojo team has posted a [proposal](https://forum.modular.com/t/proposal-deprecating-mojo-test/2371) to deprecate `mojo test` in favor of a new Mojo-based testing framework.
   - The team is soliciting feedback on the proposal.
- ****Type Reflection in Mojo Needs Work****: A developer asked how to print the type of a variable in Mojo, similar to Python's `type(a)` function, because `typeof(greeting)` doesn't work.
   - Another dev pointed out using `from compile.reflection import get_type_name`, also noting that *we totally need it to just be able to be printed* and that *We really need a type type*.
- ****Inlining Functions Can Cause Bloat****: A developer found that inlining functions with large unrolled loops significantly increased their binary size and build times.
   - They found that removing the inlining dropped the binary size from **7.5mb** to **1.5mb**. Another member chimed in that `@always_inline` for `@parameter for i in _` will unconditionally inline/unroll the loop.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1427657435232862289)** (1 messages): 

> `TorchAX, Pure Python Custom Devices, JAX device in Torch` 


- **TorchAX Opens Doors for Python Custom Devices**: The [TorchAX](https://github.com/google/torchax) library has paved the way for pure Python custom devices.
   - A *"jax"* device is now available in **Torch** thanks to this package.
- **JAX Device Lands in PyTorch via TorchAX**: The **TorchAX** library facilitates the creation of a *'jax'* device within the **PyTorch** framework.
   - This integration enables the use of pure Python custom devices within PyTorch, expanding its flexibility and capabilities.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1427381845972090962)** (69 messages🔥🔥): 

> `Salesforce Agent Scripting, Agentic Platforms like Devin, Google Gemini 3 with Jules, Nvidia DGX Spark, Anthropic Deepens Salesforce Partnership` 


- **Salesforce Embraces Agent Scripting**: Salesforce introduced a scripting/templating language inside prompts to give users more deterministic control, as detailed in their [blog post](https://developer.salesforce.com/blogs/2025/10/introducing-hybrid-reasoning-with-agent-script).
   - This approach aims to provide users with greater command over agent behavior.
- **Devin touted as Agentic Platform**: In a discussion about agentic platforms for remote building, **Devin** was recommended as a suitable option for telling something to go off and remotely build something in a loop forever.
   - The recommendation came in response to a query seeking alternatives to Claude Code for more autonomous development tasks.
- **Google's Gemini 3 with Jules Expected**: It was suggested to wait for **Google's Gemini 3**, which is expected to come with **Jules**, as a potential agentic platform.
   - This suggestion implies anticipation for advanced capabilities from Google in the agentic platform space.
- **Nvidia DGX Spark Bandwidth Limited**: The **Nvidia DGX Spark** mini-PC is considered *dead on arrival* due to bandwidth limitations, despite some seeing a place for it as a dev-workstation.
   - Early benchmarks of the [$4k Nvidia DGX Spark](https://www.pcmag.com/news/nvidia-to-start-selling-3999-dgx-spark-mini-pc-this-week) show only ~11 t/s on gpt-oss-120b-fp4, far below a $4.8k M4 Max MacBook Pro that hits 66 t/s; community blames low LPDDR5X bandwidth (273 GB/s vs 546 GB/s) and declares the device overpriced for pure inference.
- **ReductoAI Secures $75M Series B Funding**: [ReductoAI](https://xcancel.com/aditabrm/status/1978129711898431935) closed a **$75M Series B** led by a16z after a 6x growth in document processing volume, now surpassing 1B pages processed in total for customers.
   - The funds will accelerate model R&D and new product features, including major accuracy improvements and customizable pipelines.


  

---


### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1427855696627503235)** (1 messages): 

> `AI Freelancing, Model Fine-Tuning, LLM Infra, AI Startups, AI Agent Development` 


- **Youthful AI enthusiast seeks collaborators**: A **17-year-old** member is looking to connect with others their age interested in building in **AI**, particularly in areas like **model fine-tuning**, **LLM infra**, and **AI startups**.
   - This member also runs a small **AI Freelancing Business** and is open to sharing projects and ideas with like-minded individuals.
- **Young Freelancer Connects!**: A **17 year old** AI freelancer seeks to connect with other young people passionate about **AI Agent development** and **LLM Infrastructure**.
   - The freelancer hopes to share **projects and ideas** with others interested in **fine-tuning** and **AI Startups**.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1427376401648058398)** (35 messages🔥): 

> `East China Normal University AI Call for Papers, State of AI 2025 Report, Cursor AI Code Editor, DGX Spark Availability, RTX 5090 vs DGX Spark` 


- ****AI Papers Mandatory at East China Normal University****: East China Normal University announced a [call for papers](https://ed.ecnu.edu.cn/edenglish/fa/a9/c48385a719529/page.htm) mandating **AI authorship** for submissions.
   - This bold move signals a shift toward automated research, albeit with potential oversight from human researchers.
- ****State of AI 2025 Report Drops****: The [State of AI 2025 report](https://www.stateof.ai/) is out, offering predictions and analyses of the artificial intelligence landscape.
   - No sentiments or opinions about the report were discussed.
- ****DGX Spark Debuts, Sparks Debate****: **DGX Spark** is now available, prompting discussion about its cost-effectiveness versus alternatives like **RTX 5090**.
   - One member pointed out that *2x RTX 5090* cards could be obtained for the price of one **DGX Spark**, with the caveat that **DGX Spark** offers *seamless CPU-GPU unified memory*.
- ****Cursor Editor Crashes Out of Favor****: One member shared their negative experience with the **Cursor** code editor after *3 days*, stating it had been relegated to the *trash bin*.
   - They cautioned against getting *sucked in* or *emotionally exhausted* by such tools without setting predefined stopping criteria.
- ****Codex Extension Cuts Code Conflicts in VSCode****: A member found the **VSCode Codex extension** to be adequate, citing its low opportunity cost with a **GPT Plus** subscription.
   - It was deemed *reasonable at reformatting code / generating test cases* and capable of running on multiple smaller projects for UI, backend, and website tasks.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1427643929221927065)** (12 messages🔥): 

> `Cursor Review, SEAL Paper, Tab Completion & Agentic Coding, Multi-Agent Systems, AI Completions for Coding` 


- **Cursor editor faces harsh review**: After using the [Cursor editor](https://cursor.sh/) for 3 days, a user relegated it to the trash bin, noting *vaporization imminent* and encouraging others to set stopping criteria when trying new tools to avoid being *sucked in nor emotionally exhausted*.
   - The user suggested, *this is false, and if you do try it, don't get sucked in nor emotionally exhausted by the incessant failure and backtracking of these "tools" without setting a prespecified stopping criteria for yourself before using them!*
- **SEAL Paper's new update**: A user shared a link to a new update of the **SEAL paper**, titled [SEAL: Secure and Efficient Asymmetric Learned Hashing](https://arxiv.org/abs/2506.10943) on ArXiv.
   - The user noted that the paper is open source, available on [GitHub](https://github.com/Continual-Intelligence/SEAL), and *seems interesting*.
- **Agentic Coding Adoption Accelerates**: A user stated that developers not adopting **tab completion** and **agentic coding** are falling behind, comparing them to *a horse rider yelling at automobiles*.
   - The user shared a link to [arxiv.org/abs/2510.01279](https://www.arxiv.org/abs/2510.01279) which is a paper discussing similar subjects.
- **AI Completions' usefulness debated**: A user suggested that the usefulness of **AI completions** in coding depends heavily on the type of work, noting they often slow down progress due to being *fucking stupid*.
   - The user added that the helpfulness of these tools is a measure of how much **boilerplate** one writes on average.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/)** (1 messages): 

erkinalp: https://clockss.org/
  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1427381556321849497)** (37 messages🔥): 

> `Kimi Team contact, Trickle vibe coding website, Aspen leveraged 100x on bitcoin, Gemini vs GPT5` 


- **Kimi Team Contact Shared**: A user suggested contacting <@547736322568355861> from the **Kimi Team** for cool one-shot stuff.
   - They also shared an email address **zhongweiming@msh.team** for business or marketing collaborations, and linked to [Ilya Sutskever's tweet](https://x.com/ilyasut/status/1977971110923968656?s=46&t=_NtP_RUn04yF_4hD_VEDkQ) as well as [Kimi Moonshot's latest Tweet](https://x.com/kimi_moonshot/status/1978047376914080127?s=46&t=_NtP_RUn04yF_4hD_VEDkQ).
- **Trickle Vibe Coding Website Revealed**: **Trickle** ([https://trickle.so/](https://trickle.so/)) is a vibe coding website similar to Lovable, Bolt, and Manus, according to a member.
   - The member stated that if a user opened the tweet, they would know what **Trickle** is and further claimed that it's the first thing to come up when you google search trickle.
- **Aspen's Bitcoin Leverage Allegations**: A user jokingly accused **Aspen** of leveraging **100x** on Bitcoin, making a million dollars in profit, quitting his job, and then getting liquidated after tariff news.
   - The user attached a screenshot with the text *lmaoLOL*.
- **Gemini Compared to GPT-5**: A user humorously commented that if they were an AI model, they'd be better than **Gemini** but worse than **GPT-5**.
   - They added that **Gemini 2.5** is too old and that *nobody wants to use Gemini in its current state*.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1427393630322032710)** (18 messages🔥): 

> `Mixture of Experts, Sliding Window Attention, LM Evaluation Harness, ArXiv Papers Dataset, GitHub Code 2025 Dataset` 


- **MoE Transformer Pretraining Buzz**: A member inquired about open-source code for pretraining **Mixture of Expert (MoE)** transformers.
   - Another member responded that many training libraries support it, including [EleutherAI's gpt-neox](https://github.com/EleutherAI/gpt-neox).
- **Sliding Window Attention Question**: A member asked if [gpt-neox](https://github.com/EleutherAI/gpt-neox) supports **sliding window attention** with a custom pattern.
   - The response was affirmative, stating that it does.
- **LM Evaluation Harness Still Relevant**: A member inquired about which framework to use to evaluate custom LLMs on common benchmarks, and another member linked to the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness).
   - The member initially questioned its validity due to the last update being two years ago, but then realized they were mistaken.
- **Massive Datasets Debut for Code and Sci-Reasoning**: A member shared **two new datasets** for code and scientific reasoning models: the **4.6TB** [ArXiv Papers dataset](https://huggingface.co/datasets/nick007x/arxiv-papers) and the [GitHub Code 2025 dataset](https://huggingface.co/datasets/nick007x/github-code-2025).
   - The **ArXiv Papers** dataset is described as a massive scientific corpus perfect for training models on academic reasoning, while the **GitHub Code 2025** dataset contains GitHub's top 1 million repos above 2 stars.
- **New Informal Research Community Seeks Resources**: A member, on behalf of Leo Gao, asked if EleutherAI would be willing to support a group of researchers and engineers from institutions like Stanford and CMU in their research projects.
   - Another member expressed interest in learning more about the specific projects and exploring how EleutherAI might provide support, suggesting a follow-up discussion in 1-2 weeks.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1427425897421733910)** (15 messages🔥): 

> `Less is More: Recursive Reasoning with Tiny Networks, backpropping only the last step of deep recursion, ARC rules, video models based on 3D rendered video clips, REPA` 


- **Tiny Recursive Model Mystery: Why Last Step Backprop Works**: A member questions why backpropagating only on the last step of deep recursion in the ["Less is More: Recursive Reasoning with Tiny Networks"](https://arxiv.org/abs/2307.00030) paper improves prediction accuracy after `T-1` steps of `no_grad()` deep recursion.
   - They seek insight on how backpropping on only `n` RNN forward passes results in model improvement when doing `Tn` RNN forward passes, sparking discussion on the learning mechanism behind iterative refinement.
- **Tiny Recursive Model: Issue resolution pending**: A member is awaiting resolution for an issue in the ["Less is More: Recursive Reasoning with Tiny Networks"](https://github.com/SamsungSAILMontreal/TinyRecursiveModels/issues/15) related repository.
   - This suggests ongoing interest and potential challenges in replicating or understanding the paper's findings.
- **Ethical Training Data: Fair Game?**: A member asks if training video models based on **3D rendered video clips** violates any [ARC rules](https://en.wikipedia.org/wiki/AI2_Reasoning_Challenge).
   - They also inquire about the typical quality expectations for training samples in video models and if using test sets is permissible.
- **REPA resurfaces in image gen biz**: Members discuss the use of **REPA** in image generation, with one expressing fears of it being a significant factor, but another clarifying it's not as big of a difference as their original method.
   - They shared a link to [SEAL](https://jyopari.github.io/posts/seal🤔) where **REPA** is used, indicating its relevance in the current context.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1427445083547897997)** (5 messages): 

> `ACE Playbook, AgentLearningEE, StraughterG's X post` 


- **ACE Playbook rocks Github**: A member shared the [ACE Playbook](https://github.com/jmanhype/ace-playbook) on Github, and another member said *"it's great"*.
- **AgentLearningEE is a hit**: Members enjoyed the [AgentLearningEE](https://github.com/jmanhype/AgentLearningEE) repo.
- **StraughterG post on X is popular**: A member shared [StraughterG's X post](https://x.com/StraughterG/status/1978126261273694368).


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1427452736332365956)** (14 messages🔥): 

> `Big Paper Tease, CheshireCat 3.0 Release, Neo4j Integration Request, London Meetup?` 


- ****Big Paper** Announcement Teased**: A user hinted at a **major upcoming paper** release, generating anticipation within the community [via X](https://x.com/lateinteraction/status/1977887477328146458).
   - Stay tuned for more details!
- ****CheshireCat 3.0** Open Source Framework Launched**: A user shared their open-source project, **CheshireCat 3.0**, a framework based on **LangChain** and **Qdrant** designed for multimodal RAGs, multitenant chatbots, and agentic tool orchestration, available on [GitHub](https://www.github.com/matteocacciola/cheshirecat-core).
   - It features plugin-based extensibility and enterprise readiness, with documentation on [Deepwiki](https://deepwiki.com/matteocacciola/cheshirecat-core).
- **London **Databricks x DSPy** Meetup Proposed**: A member inquired about interest in a **Databricks x DSPy** meetup in London before the end of the year.
   - Another member responded positively, expressing their interest in joining.
- ****Neo4j Integration** Desired for **Graph RAG****: A user requested **Neo4j integration** for **graph-based RAG** within the **CheshireCat 3.0** framework.
   - The project creator welcomed contributions to facilitate this integration.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1427450564664365127)** (17 messages🔥): 

> `MCP Server Implementation, Binary data support in tool calls, Embedded resources, Host engineering, Mapping parts of the tool response` 


- ****MCP Server** Needs Binary Data**: A member is working on an **MCP server** implementation that needs to return binary data (e.g., a **PDF**) as a result of a tool call.
   - The member noted that the spec seems to support only certain subsets of binary resources (image and audio) in the *type* field for tool call results.
- **Embedded Resources Could Help**: Another member suggested creating an **embedded resource** in the response and use any **MIME types** desired.
   - The first member noted that they'd need to come up with a fake **URI**.
- **OpenAI API Supports PDF Files**: A member linked the [OpenAI API documentation](https://platform.openai.com/docs/guides/pdf-files?api-mode=responses#file-urls) on **PDF file** support.
   - Another member clarified that the support might be limited to user messages, and not necessarily tool responses.
- **Host Engineering Required**: A member stated that model APIs don't support binary tool results out of the box, so **host engineering** is needed.
   - They said that most **MCP client hosts** will not support returning binaries/files from a tool without this, even if engineering workarounds are possible.
- **Mapping Tool Response Parts**: A member stated that the host can map parts of the tool response to any part of the model API call, potentially treating **embedded resources** with **PDF MIME type** as user-uploaded documents.
   - They note that some models get confused by having tool results mapped to user messages in this way.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1427550364647034950)** (2 messages): 

> `Pylint Removal, Test Refactoring with ChatGPT` 


- **Pylint's Utility Debated**: A member questioned the value of **Pylint** and suggested removing it.
   - The reasoning was that perhaps it *doesn't find anything good*.
- **ChatGPT's Test Refactoring Failure**: A member proposed using **ChatGPT** to refactor *test_advancedindex* into tests with more meaningful names.
   - However, the member reported that **ChatGPT messed it up** and the tests failed, requiring manual refactoring.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1427532531627982938)** (10 messages🔥): 

> `Contributing to tinygrad, Tensor buffer is not writable, Freezing parts of a matrix for training, Virtual tensor creation, Accessing computed gradients` 


- **Hotz Disapproves First Contribution Attempt**: A member asked if removing realize from `__setitem__` and getting `TestSetitemLoop.test_arange` to be one kernel is a good first contribution attempt.
   - George Hotz replied that *it's not really* and they need to go through all the bounties and decide what's reasonable.
- **Tensor buffer throws error**: A new user encountered an error *underlying buffer is not writable* when calling `my_tensor.numpy()`.
   - The user was asked to share the rest of the code to diagnose the issue.
- **Matrix Freezing Frenzy**: A user wants to *freeze part of the matrix and train only the other part of it*.
   - They are looking for a way to create a virtual tensor (concat) that points to the data of the original tensors, with these original tensors having different `requires_grad`.
- **Gradient Access Granted?**: A user inquired about accessing the computed gradients to zero out those that reflect part of the tensor.
   - They suggest a workaround of storing the original weights and restoring them after the gradients are applied, but this requires more memory.
- **Simulating Virtual Tensors**: The user suggests using `Tensor.cat(x @ a.detach(), x @ b, dim=-1)` to simulate a *virtual* tensor.
   - This allows concatenating tensors with different `requires_grad` settings.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1427379333718343700)** (11 messages🔥): 

> `Manus Functionality, Job Openings at Manus, Community Moderator Perks, Product Dissatisfaction and Feedback, Daily Credits Issue` 


- ****Manus Marvels**: Functionality Praised Amidst App-Switching Tendencies**: A user expressed appreciation for **Manus' functionality**, explaining their tendency to switch between apps while still acknowledging Manus' strengths.
   - They suggested Manus act less like a *mother-in-law coder* and more like an assistant, offering guidance on optimal usage.
- ****LinkedIn Leads**: Job Openings Listed for Manus Enthusiasts**: **Job openings** are listed on the [Manus LinkedIn page](https://www.linkedin.com/company/manus-im/jobs/), with HR handling candidate selection.
   - Community moderators receive **free usage** and can guide prompts, and collaborations are open for KOLs who share their social media handles.
- ****Feedback Frontier**: Addressing Product Dissatisfaction Through Shared Sessions**: A community member requested a user who frequently expresses dissatisfaction to share **failed session links** to better understand and address potential product issues.
   - The team is committed to fixing product issues and offering guidance on **usage and prompting techniques** to benefit the community.
- ****Credit Crunch**: User Queries About Missing Daily Credits**: A user inquired about not receiving **300 daily credits**, without further context in the given messages.
   - The user mentioned past interactions, including sharing content and creating a repository, indicating a potential account-specific issue.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1427605566867640380)** (5 messages): 

> `aider alias, OpenCode GLM 4.6` 


- **Aider Alias Created for Chat Mode**: A user set `aider` as an alias for `aider --chat-mode ask`, expressing a desire to directly run `aider` without shell scripts or command history.
   - Despite having `chat-mode: ask` in their `.aider.conf.yml`, they still encounter the need to use `/ask` and consider adding this to Aider.
- **OpenCode GLM 4.6 Praised for Programming Experience**: A user shared positive feedback on **OpenCode + GLM 4.6**, highlighting a comfortable and enjoyable programming experience.
   - They emphasized the elimination of concerns about **token counting** and praised the usability, using `aider.chat` with **Sonnet 4.5** for specific refinements.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1427476104180011161)** (2 messages): 

> `Adding files to long messages, Aider workflow tips` 


- **Users seek advice on adding files to long aider messages**: A user inquired about the best way to add files to a long message in aider after already starting to compose it, using `/add` or `/read-only`.
   - The user's current workaround involves copying the message to **vim**, adding the files, and then pasting the content back.
- **Send and Ctrl+C for Aider Workaround**: The user jokingly suggests a workflow of typing the message then hitting send and quickly pressing Ctrl+C to stop the send.
   - This perhaps serves to highlight the difficulty of the situation.


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1427667352518135952)** (1 messages): 

> `Agentic Tools, Aider's Capabilities` 


- **Aider Trivializes Editing Tasks**: A member noted that tasks like editing **100loc functions** are trivial to do with **aider** in their day-to-day work.
   - This remark was made in response to a discussion about the [push for agentic tools](https://forum.cursor.com/t/why-the-push-for-agentic-when-models-can-barely-follow-a-single-simple-instruction/137154/21), questioning their utility when models struggle with simple instructions.
- **Agentic Tools Questioned**: A discussion arose questioning the push for **agentic tools**, given that models can barely follow a single simple instruction.
   - A member shared an interesting remark on agentic tools in the context of their daily work.


  
