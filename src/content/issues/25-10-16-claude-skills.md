---
id: MjAyNS0x
title: Claude Agent Skills - glorified AGENTS.md? or MCP killer?
date: '2025-10-16T05:44:39.731046Z'
description: >-
  **Anthropic** achieves a rare feat with back-to-back AI news headlines
  featuring **Claude's** new **Skills**—a novel way to build specialized agents
  using Markdown files, scripts, and metadata to handle tasks like creating and
  reading PDFs, Docs, and PPTs. Simon Willison calls this a "bigger deal than
  MCP," predicting a "Cambrian explosion in Skills." Meanwhile, **Anthropic**
  launches **Claude 4.5 Haiku** with strong reasoning and long-context
  capabilities, priced competitively. Other updates include **OpenAI's** ChatGPT
  memory management improvements, **Windows 11 Copilot** voice and vision
  features, and **HuggingChat Omni** routing across 115 open-source models from
  15 providers. These developments highlight advances in agent skills, document
  processing, long-context reasoning, and multi-model routing.
companies:
  - anthropic
  - openai
  - microsoft
  - perplexity-ai
  - huggingface
  - groq
  - cerebras
  - togethercompute
models:
  - claude-4.5-haiku
  - claude
  - chatgpt
  - huggingchat-omni
topics:
  - agent-skills
  - document-processing
  - long-context
  - reasoning
  - multi-model-routing
  - memory-management
  - voice
  - vision
people:
  - simonwillison
  - alexalbert__
  - mustafasuleyman
  - yusuf_i_mehdi
  - aravsrinivas
---


**Claude is all you need**

> AI News for 10/15/2025-10/16/2025. We checked 12 subreddits, 544 Twitters and 23 Discords (197 channels, and 6365 messages) for you. Estimated reading time saved (at 200wpm): 492 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

In a rare feat for any lab, Anthropic gets [two back to back](https://news.smol.ai/issues/25-10-15-haiku-45) AINews headline stories with [Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) today, "a new way to build specialized agents using files and folders". It turns out that Claude's recent new skills for creating and reading PDFs and Docs and PPTs were all Skills.

[](https://resend-attachments.s3.amazonaws.com/G0aCs2pSirnjnWA)

- [Introduction blogpost and video](https://www.anthropic.com/news/skills)
- [Engineering writeup](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- [HN discussion](https://news.ycombinator.com/item?id=45607117)
- [Simon Willison calls it "bigger deal than MCP"](https://simonwillison.net/2025/Oct/16/claude-skills/)

Quoting Simon:

> Skills are conceptually extremely simple: a skill is a Markdown file telling the model how to do something, optionally accompanied by extra documents and pre-written scripts that the model can run to help it accomplish the tasks described by the skill.
> 
> 
> Claude’s new [document creation abilities](https://www.anthropic.com/news/create-files), which accompanied [their new code interpreter feature](https://simonwillison.net/2025/Sep/9/claude-code-interpreter/) in September, turned out to be entirely implemented using skills. Those are [now available Anthropic’s repo](https://github.com/anthropics/skills/tree/main/document-skills) covering `.pdf`, `.docx`, `.xlsx`, and `.pptx` files.
> 

and:

> I expect we’ll see a Cambrian explosion in Skills which will make this year’s MCP rush look pedestrian by comparison.
> 
> 
> Skills are Markdown with a tiny bit of YAML metadata and some optional scripts in whatever you can make executable in the environment. They feel a lot closer to the spirit of LLMs—throw in some text and let the model figure it out.
> 

[](https://resend-attachments.s3.amazonaws.com/M84445mZrQ4drWs)

---

# AI Twitter Recap

**Assistant platforms: ChatGPT Memory, Sora 2, Claude 4.5 Haiku and “Skills,” Windows Copilot, Perplexity, HuggingChat Omni**

- **OpenAI product updates**: ChatGPT now auto-manages saved memories (no more “memory full”), with search/sort and re-prioritization; rolling out to Plus/Pro on web globally [@OpenAI](https://twitter.com/OpenAI/status/1978608684088643709). Sora 2 added Storyboards on web for Pro users and extended video lengths (all users up to 15s on app/web; Pro up to 25s on web) [@OpenAI](https://twitter.com/OpenAI/status/1978661828419822066), [@billpeeb](https://twitter.com/billpeeb/status/1978662020947087869).
- **Anthropic’s value-tier and agent upgrades**: Claude 4.5 Haiku launched at $1/$5 per 1M input/output tokens; in reasoning mode it scores 55 on the Artificial Analysis index and is 3x cheaper than Sonnet 4.5, with strong long-context/coding results per [Artificial Analysis](https://twitter.com/ArtificialAnlys/status/1978661658290790612). Community rankings put it #22 overall on LMArena, with coding/longer-query strengths [@arena](https://twitter.com/arena/status/1978966289248063885). Anthropic also introduced **Skills**—packaged instruction folders/scripts/resources loaded at runtime—available in [claude.ai](http://claude.ai/), Claude Code, and API with docs and engineering notes [@alexalbert__](https://twitter.com/alexalbert__/status/1978877498411880550), [@AnthropicAI](https://twitter.com/AnthropicAI/status/1978896757489594404). Enterprise features now include Microsoft 365 (SharePoint, OneDrive, Outlook, Teams) integration and enterprise search [@AnthropicAI](https://twitter.com/AnthropicAI/status/1978864348236779675).
- **Windows and Perplexity ship UX primitives**: Windows 11 adds Copilot Voice (“Hey Copilot”), Vision across desktop/apps/docs, and upcoming Copilot Actions for local files [@mustafasuleyman](https://twitter.com/mustafasuleyman/status/1978808627008847997), [@yusuf_i_mehdi](https://twitter.com/yusuf_i_mehdi/status/1978808604200259785). Perplexity released built-in language learning experiences and new Finance features (insider trading tracker), on iOS/web [@perplexity_ai](https://twitter.com/perplexity_ai/status/1978859991152165125), [@AravSrinivas](https://twitter.com/AravSrinivas/status/1978941079182545338).
- **Routing across many OSS models**: HuggingChat v2 launched “Omni,” policy-based automatic model selection across 115 OSS models and 15 providers (e.g., Groq, Cerebras, Together). Omni can route tasks between coder and writing models in one session; 100% open-source [@victormustar](https://twitter.com/victormustar/status/1978817795312808065), [@reach_vb](https://twitter.com/reach_vb/status/1978854312647307426).
- Also noteworthy: “Sign in with ChatGPT” is being pitched to sites; costs can be shifted to end users using OpenAI models [@steph_palazzolo](https://twitter.com/steph_palazzolo/status/1978835849379725350). Google’s AI Studio now has a unified playground for Chat/GenMedia/Live models [@GoogleAIStudio](https://twitter.com/GoogleAIStudio/status/1978861583078961263).

**Systems and Infra: vLLM on TPU, Google’s TPU push, and the reality of local devboxes**

- **TPU inference stack lands**: vLLM unveiled a reimagined TPU backend co-developed with Google, unifying PyTorch and JAX via a single JAX-to-XLA lowering path with SPMD-by-default, Ragged Paged Attention v3, and 2x–5x throughput over its February prototype; supports Trillium (v6e) and v5e [@vllm_project](https://twitter.com/vllm_project/status/1978855648176853100), [@_philschmid](https://twitter.com/_philschmid/status/1978889178067743210).
- **Google broadens TPU access**: TPUs are now sold to external customers, directly competing with NVIDIA [@zephyr_z9](https://twitter.com/zephyr_z9/status/1978835094216343820). Baseten reports 50% latency and 60%+ throughput gains from early NVIDIA Dynamo adoption with KV-cache-aware routing [@basetenco](https://twitter.com/basetenco/status/1978883986924634551).
- **Local vs cloud**: Practical notes from real users—Mac Mini M4 Pro is excellent for local LLM inference but not sustained fine-tuning workloads; CUDA remains essential for PyTorch training due to MPS instability, with many choosing cloud GPUs over noisy/hot multi-GPU desktops [@rasbt](https://twitter.com/rasbt/status/1978608882156269755). PyTorch’s Soumith notes Apple’s inconsistent investment in the MPS backend, with Meta engineers doing much of the lifting; cautions against expecting parity with NVIDIA for training [@soumithchintala](https://twitter.com/soumithchintala/status/1978848796953161754).
- Pipes and fabs: TSMC N2 volume production is slated to begin before year-end [@TechPowerUp](https://twitter.com/TechPowerUp/status/1978737339171017215). Google published torchax, exploring PyTorch→JAX lowering [@gallabytes](https://twitter.com/gallabytes/status/1978860154008240142).

**Reasoning, RL, long-context, and evals**

- **RL scaling laws that predict ahead**: Meta and collaborators released “The Art of Scaling Reinforcement Learning Compute for LLMs,” a 400k GPU-hour systematic study proposing ScaleRL (PipelineRL with 8-step off-policyness), CISPO loss, FP32 logits, and interruption-based length control. Key result: performance at target compute is predictable from half-compute runs; many small decisions materially affect stability/ceiling [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1978793969384624226), [@_lewtun](https://twitter.com/_lewtun/status/1978826407376458125), [@omarsar0](https://twitter.com/omarsar0/status/1978865039529689257).
- **Inference-time recursion beats context rot**: Recursive Language Models (RLMs) show that recursive self-calls/tools over unbounded contexts can outperform standard GPT-5 on long-context tasks, staying cost-effective even at 10M+ tokens. Minimal gist to try: [@a1zhang](https://twitter.com/a1zhang/status/1978948676287340753), commentary [@dbreunig](https://twitter.com/dbreunig/status/1978873161841066464).
- **Compute-efficient routing and RL ideas**: Dr.LLM dynamically skips/repeats transformer layers to reduce compute while improving accuracy, with per-block routers trained via offline MCTS and focal loss—greedy routing at inference [@omarsar0](https://twitter.com/omarsar0/status/1978829550709866766). “Tandem training” intermittently samples tokens from a frozen weak model during RL to keep solutions intelligible to weaker collaborators [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1978794773747314765).
- **Small models, big puzzles**: The Tiny Recursion Model (TRM, ~7M) achieves 40% on ARC-AGI-1 at ~$1.76/task (weights + recipe released), reinforcing that specialized inference procedures matter [@arcprize](https://twitter.com/arcprize/status/1978872651180577060). AssistantBench: o3 currently tops GPT-5-med on Princeton’s task [@OfirPress](https://twitter.com/OfirPress/status/1978925179876020247).
- **Evals > vibes**: Andrew Ng lays out a pragmatic framework for evals and error analysis in agentic systems—prototype, examine outputs, define custom metrics/judges, iterate your evals, and only then optimize [@AndrewYNg](https://twitter.com/AndrewYNg/status/1978867684537438628).

**Coding agents and retrieval: fast context beats long context**

- **Cognition’s Fast Context via SWE-grep**: A new model family (>2,800 TPS) for rapid, multi-turn agentic search that locates the “right files” ~20x faster than Claude 4.5 Haiku while rivaling frontier models on Cognition’s CodeSearch eval; now rolling out in Windsurf via the Fast Context subagent and playground [@cognition](https://twitter.com/cognition/status/1978867021669413252), [@silasalberti](https://twitter.com/silasalberti/status/1978871477605929229), [@swyx](https://twitter.com/swyx/status/1978874342743343254). Cerebras-backed deployments further cut latency in practice [@draecomino](https://twitter.com/draecomino/status/1978898418354561225).
- **Agent primitives and OSS toolchains**: The Cline CLI (preview) exposes a scriptable, open, “primitive agent loop” that IDE Cline can orchestrate—designed for subagents and composable workflows [@cline](https://twitter.com/cline/status/1978874789193486749). “Open Agent Builder,” an n8n-style OSS canvas, connects Firecrawl, LLMs, logic nodes, and MCPs for API-deployable workflows [@firecrawl_dev](https://twitter.com/firecrawl_dev/status/1978878728827478289), [@CalebPeffer](https://twitter.com/CalebPeffer/status/1978852506286571737). Surfer 2 reports SOTA across WebVoyager/AndroidWorld/WebArena/OSWorld for cross-platform computer-use [@hcompany_ai](https://twitter.com/hcompany_ai/status/1978935436111229098).
- **Anthropic Skills for code**: Devs report sharper, more precise Claude Code by tiering in domain-specific scripts/resources as runtime Skills—structured context engineering that complements MCP/tools [@omarsar0](https://twitter.com/omarsar0/status/1978919087137804567), docs [@alexalbert__](https://twitter.com/alexalbert__/status/1978877611159003542).

**Vision and multimodal: real-time worlds, OCR/VLMs, and image/video editing**

- **World models in real time**: The World Labs’ RTFM is a real-time, persistent, 3D consistent autoregressive diffusion transformer trained on large-scale video—streaming interactively at H100 speed with a live demo [@theworldlabs](https://twitter.com/theworldlabs/status/1978839171058815380), [@drfeifei](https://twitter.com/drfeifei/status/1978840835341914164), [@jcjohnss](https://twitter.com/jcjohnss/status/1978842517605843391).
- **Video and editing pipelines**: Google Veo 3.1 is live in LTX Studio and Synthesia (sharper realism, audio, full Keyframe support) [@LTXStudio](https://twitter.com/LTXStudio/status/1978827563926716704), [@synthesiaIO](https://twitter.com/synthesiaIO/status/1978836856419635561). Sourceful’s Riverflow 1 debuts #1 on Artificial Analysis’ Image Editing “All” listings, combining a VLM with open diffusion; priced at $66/1k images (a “mini” is $50/1k) [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1978891167795417092).
- **Doc AI and grounded VLMs**: PaddleOCR-VL (0.9B) targets industrial document intelligence (text, tables, formulas, charts, handwriting), powered by NaViT + ERNIE; 109 languages supported [@PaddlePaddle](https://twitter.com/PaddlePaddle/status/1978809999263781290). ByteDance’s Sa2VA marries SAM2 and LLaVA for dense grounded understanding across images/videos [@HuggingPapers](https://twitter.com/HuggingPapers/status/1978745567258829153). Alibaba’s Qwen3-VL-Flash brings 256K context, better spatial reasoning/3D localization, OCR, and tighter safety [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1978841775411503304). Google “Nano Banana” image editing lands in Lens/AI Mode (US/India initially) [@Google](https://twitter.com/Google/status/1978857184566837735).

**Open models and tiny wins: nanochat, MobileLLM-Pro, ColBERT minis, and safety components**

- **Micro-models in the wild**: Karpathy’s nanochat d32 ($1k from-scratch) improved CORE to 0.31 (> GPT-2 ~0.26) and GSM8K to ~20%, with full report/scripts released; community integrations into Transformers and vLLM are in flight [@karpathy](https://twitter.com/karpathy/status/1978615547945521655), [@ben_burtenshaw](https://twitter.com/ben_burtenshaw/status/1978832914952401081).
- **On-device-class LMs**: Meta’s MobileLLM-Pro (1B) released base + instr checkpoints (with quantized variants) aimed at high-quality, efficient on-device inference; pre-trained on <2T open tokens, outperforming Gemma 3 1B and Llama 3.2 1B by 5.7% and 7.9% on reasoning/knowledge/long-context retrieval [@_akhaliq](https://twitter.com/_akhaliq/status/1978916251456925757).
- **Embedding/retrieval, ultra-compact**: [mixedbread.ai](http://mixedbread.ai/)’s mxbai-colbert-edge-v0 (17M, 32M) provides reproducible ColBERT training; the 17M ranks first on LongEmbed for models <1B, Apache 2.0 licensed [@mixedbreadai](https://twitter.com/mixedbreadai/status/1978853869557055492), [@bclavie](https://twitter.com/bclavie/status/1978854449062793335). Nanonets released new OCR2-3B and 1.5B-exp models (Apache-2.0), handling forms, watermarks, charts, and even flowcharts [@mervenoyann](https://twitter.com/mervenoyann/status/1978837720353927415).
- **Safety tooling**: Alibaba open-sourced components from Qwen3Guard, including Qwen3-4B-SafeRL (WildJailbreak jumps 64.7→98.1 without hurting general performance) and Qwen3GuardTest for classifying intermediate “thinking” and token-by-token moderation [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1978732145297576081).

**Top tweets (by engagement)**

- “TV in the 90s vs 2025” UX rant resonated widely [@karpathy](https://twitter.com/karpathy/status/1978653908663726585).
- ChatGPT “Memory” auto-management rollout [@OpenAI](https://twitter.com/OpenAI/status/1978608684088643709).
- Sora 2 updates: Storyboards + longer videos [@OpenAI](https://twitter.com/OpenAI/status/1978661828419822066).
- DeepMind’s “Turing Test for video” tease and fusion collaboration with CFS [@demishassabis](https://twitter.com/demishassabis/status/1978644313824534954), [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1978808994811588666).
- Perplexity launches language learning and insider trading tracking [@AravSrinivas](https://twitter.com/AravSrinivas/status/1978865088296542387), [@AravSrinivas](https://twitter.com/AravSrinivas/status/1978941079182545338).
- Anthropic’s Agent Skills announcement [@alexalbert__](https://twitter.com/alexalbert__/status/1978877498411880550).
- vLLM’s TPU backend unifying PyTorch + JAX [@vllm_project](https://twitter.com/vllm_project/status/1978855648176853100).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

TO BE COMPLETED

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

TO BE COMPLETED

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp
> 

**Theme 1. Model Mania: New Releases and Breakthroughs Shake the Landscape**

- [**Gemini 3 Pro Codes a Playable Game From a Single Prompt**](https://discord.com/channels/1340554757349179412/1340554757827461211/1428095522148847720): Users in the LMArena Discord celebrated **Gemini 3 Pro's** coding prowess after it generated a fully playable [HTML Geometry Dash clone](https://link.to.clone/) from a single 1k+ line prompt, a feat **Gemini 2.5 Pro** failed. This fueled speculation that Gemini 3 Pro might outperform the anticipated **GPT-5 Pro** by **5-10%** in coding tasks.
- [**OpenAI's Sora 2 Gets a Storyboard Upgrade**](https://discord.com/channels/974519864045756446/977259063052234752/1428171257962041466): **Sora 2** now allows **Pro** users to create **Storyboards** and generate videos up to **25 seconds**, while standard users are capped at **15 seconds**. **Sora 2 Pro** has also climbed the charts to tie for **#1** on the [LMArena Text-to-Video Leaderboard](https://lmarena.ai/leaderboard/text-to-video) alongside **Veo 3**.
- [**Anthropic and Cognition Drop New Models**](https://discord.com/channels/1027685395649015980/1027688115592237117/1428151932668743721): **Claude Haiku 4.5** launched on Windsurf for **1x credits**, reportedly matching **Sonnet 4's** coding performance at one-third the cost and double the speed. Meanwhile, Cognition's new **SWE-grep** models, rolling out on Windsurf, promise to make agentic search [20x faster](https://cognition.ai/blog/swe-grep) by rapidly surfacing relevant files to coding agents.

**Theme 2. Platform Pains and Subscription Snafus**

- [**Subscription Glitches Revert Pro Users to Free Tiers**](https://discord.com/channels/1074847526655643750/1074847527708393565/1428095086360789133): Users of **Cursor Pro+** and **Perplexity Pro** (specifically with Airtel plans) reported their paid plans unexpectedly reverting to free versions, locking them out of premium features. Some Perplexity users also reported being incorrectly billed during free trials, while their support teams are investigating.
- [**Integration Headaches Plague OpenRouter and Groq**](https://discord.com/channels/1161519468141355160/1161519469319946286/1428621175562440794): Cursor users struggled with a faulty **OpenRouter** integration where the service failed to process any requests, despite being configured correctly. Over in the DSPy Discord, a user reported that even when **Groq** was set as the sole provider in **OpenRouter**, the system defaulted to other models, as shown in their [settings screenshots](https://discord.com/channels/1161519468141355160/1161519469319946286/1428621175562440794).
- [**Breaking Changes in Tinygrad Frustrate Developers**](https://discord.com/channels/1068976834382925865/1068976834928193609/1428207732363886743): An openpilot developer in the tinygrad Discord expressed frustration over frequent breaking changes that cause illegible errors and require tedious commit bisection to fix. The user suggested that unstable **IMAGE hacks** for the **845** processor and constantly shifting environment variables create significant maintenance challenges.

**Theme 3. The Tooling Tribune: Frameworks and Libraries Evolve**

- [**DSPy Debates Agentic Search while Mojo Eyes Game Dev**](https://discord.com/channels/1161519468141355160/1161519469319946286/1428095494911164486): The DSPy community debated the definition of "agentic search," with some calling it a marketing term and others attempting to replicate **Claude Code's** *ripgrep* implementation, as detailed in this [Agentic Search for Dummies blog post](https://benanderson.work/blog/agentic-search-for-dummies/). Meanwhile, the Modular community is exploring Mojo's potential in game development, pointing to projects like [Stargine](https://forum.modular.com/t/stargine-a-game-engine-in-mojo/2266/3), and Modular open-sourced the entire [MAX Python API](https://forum.modular.com/t/open-sourcing-all-of-the-max-python-api/2379).
- [**PyTorch 2.9 Unlocks Symmetric Memory for Multi-GPU Harmony**](https://discord.com/channels/1189498204333543425/1398843708488552570/1428169904686633081): A [PyTorch 2.9 blog post](https://pytorch.org/blog/pytorch-2-9/) announced **PyTorch Symmetric Memory**, a new feature that simplifies programming multi-GPU kernels over NVLinks and RDMA networks. This enables powerful features like **in-kernel communication**, ultra-low-latency remote access, and customized communication patterns for performance engineers.
- [**HuggingFace Beefs Up Diffusers with Custom Blocks**](https://discord.com/channels/879548962464493619/1014557141132132392/1428591577688707123): The HuggingFace team announced that **Modular Diffusers** now supports **custom blocks**, allowing developers to implement new functionality that integrates seamlessly into the core library. A [collection of example blocks](https://huggingface.co/collections/diffusers/modular-diffusers-custom-blocks-68c8e37c62a6b2a30fd58401) and [documentation](https://huggingface.co/docs/diffusers/main/en/modular_diffusers/pipeline_block) are now available.

**Theme 4. Hardware Horizons and High-Performance Hacks**

- [**Intel's New Silicon Promises Massive Memory Bandwidth**](https://discord.com/channels/1053877538025386074/1149866623109439599/1428104163606401144): Engineers are buzzing about Intel's upcoming hardware, including the inference-only **Crescent Island** GPU which boasts **1.5TB/s** of bandwidth and **160GB** of memory. There's also excitement around the **Rubin CPX** architecture, which excels at supporting diverse number formats, potentially simplifying **software-level block floats** with a **1280-bit bus**.
- [**Local LLMs Hit the GPU Memory Wall**](https://discord.com/channels/1110598183144399058/1110598183144399061/1428104621712474193): Discussions in LM Studio highlighted a critical bottleneck for local LLMs: **GPU memory constraints**, with users noting that most models lose coherence beyond **20-40k tokens**, regardless of their advertised context windows. This sparked hardware debates, including the performance benefits of **128GB** of **DDR4 3600 RAM** versus **64GB**, with one user noting *if you had ddr5-8000 its 4 times faster*.
- [**DeepSeek Engineers Cook on Restricted H20s**](https://discord.com/channels/1189498204333543425/1189498205101109300/1428132127710384201): An urban legend circulating in the GPU MODE discord claims that **DeepSeek** engineers cleverly used low-level **PTX/SASS** instructions to overcome memory bandwidth limitations on restricted hardware. This innovation allowed them to build powerful models despite US restrictions downgrading available GPUs in China from **H100s** to **H20s**.

**Theme 5. AI Ethics and Culture Clashes**

- [**AI Granny Goes Viral, Raking in 2 Million Followers**](https://discord.com/channels/822583790773862470/1397010677364953149/1428567264403390504): An entirely AI-generated influencer, **grannyspills**, who dishes out toxic dating advice as a "blunt, gold-digging grandmother," is nearing **2 million** Instagram followers, as detailed in [this X post](https://x.com/venturetwins/status/1978852719335985309). This has sparked debate on whether audiences care about her artificial nature, with some praising the satire while others worry about AI's cultural impact.
- [**OpenAI Censors MLK in Sora After Complaints**](https://discord.com/channels/822583790773862470/1397010677364953149/1428567264403390504): Following a request from the King Estate and public complaints about AI-generated clips, OpenAI announced via an [X post](https://x.com/OpenAINewsroom/status/1979005850166648933) that it will block **Sora** from depicting **Dr. Martin Luther King Jr.** This move was heavily criticized by users as a slippery-slope concession that privatizes public figures and could lead to endless takedown demands.
- [**GPT-5 Rumored to Ditch Refusals, Sparking Debate**](https://discord.com/channels/974519864045756446/998381918976479273/1428095910038077502): Rumors in the OpenAI community suggest **GPT-5** may adopt a *less refusal* persona, similar to **GPT-4o**, prompting a divide among users. While some welcome a more compliant model, others express concern that it could compromise the model's ethical grounding and lead it to agree with anything, regardless of morality.


---

# Discord: High level Discord summaries




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Gemini 3 Pro Codes Geometry Dash**: A user prompted **Gemini 3 Pro** to generate a fully playable [Geometry Dash clone in HTML](https://link.to.clone), complete with music and physics, using the prompt: *Generate full HTML file of a clone of Geometry Dash, but if it was made in the 2000s, add music to levels (make the music using JS varied music that reflect levels) same physics as Geometry Dash game we want a full playable game. All in one HTML file, minimum 1k lines*.
   - Other models like **Gemini 2.5 Pro** failed to produce a working game with the same prompt, leading to hype for **Gemini 3**.
- **Gemini 3 Pro Benchmarks Anticipated**: Members discuss **Gemini 3 Pro** potentially outperforming **GPT-5 Pro** in coding, with speculation about a **5-10% performance increase**.
   - The model is rumored to be in A/B testing on AI Studio, prompting users to devise methods for accessing it, but there are concerns that **Google** may impose token limits to conserve server resources.
- **LMArena Bot Has Glitches**: Users reported issues with the **Video Arena bot**, such as *something went wrong while generating the response* and limitations on video generation capabilities.
   - A moderator confirmed that the team is aware of the issues and is actively working on a fix, suggesting users refer to the bugs channel for troubleshooting.
- **Sora 2 Takes the Top Spot**: **Sora 2 Pro** landed on the [Text-to-Video Leaderboard](https://lmarena.ai/leaderboard/text-to-video) and is now tied as **#1** alongside **Veo 3** & **Veo 3 Fast**, while **Sora 2** landed in at **#3**.
   - Pro accounts have access to features such as longer video lengths (up to 25 seconds) and the absence of watermarks, leading to a discussion on video quality and model preferences.
- **Community Requests PDF uploads and Message Editing**: The LMArena community has expressed a strong interest in new features such as **PDF file uploads**, **message editing**, **response refreshing**, and **message deletion**.
   - Moderators have confirmed that these features are on the radar for future implementation.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Airtel Plan Expires Prematurely!**: Several **Airtel Perplexity Pro** users reported their subscriptions ending abruptly, and the support team is *investigating* but not responding to some users.
   - Additionally, some users report getting billed during their free trial period for **Perplexity Pro**.
- **Comet Browser Referral Bug Bites Users**: Some users report earning **$5-$15 USD per user** via **Comet Browser** referrals, while others face difficulties receiving bonuses.
   - Staff are aware of [the referral bug](https://discord.com/channels/1047197230748151888/1047649527299055688/1428408489593479258) and are investigating potential referral program abuse; a user was muted for sharing a **Perplexity Pro** code.
- **Comet Browser 'Vulnerability' Solved?**: Users debated a *Comet Jacking* vulnerability, but it appears to be a solved [prompt injection issue](https://discord.com/channels/1047197230748151888/1047649527299055688/1428339356716496957).
   - Modifying internal settings and accessing other users' data violates the Terms and can get your **Perplexity** account banned.
- **Image Generation in Perplexity a Little Sketchy**: Users report issues with image generation, including incorrect frame ratios and hallucinated information, particularly in math problems.
   - One user recommended **Gemini Pro** for better accuracy as several users on **Perplexity** are still waiting for the support team to reply.
- **Sonar Deep Research Model Times Out**: A user reported a **timeout issue** with the **Perplexity Sonar Deep Research Model**, posting on the [community forum](https://community.perplexity.ai/t/perplexity-sonar-deep-research-model-timeout-issue-seeking-solution/2094) without response.
   - Another user noted their **Spaces** account was not allowing the creation of **new chats**.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT Gains Memory Upgrade**: **ChatGPT** now features automated memory management, enabling users to sort and search memories by recency.
   - The update, rolling out to **Plus** and **Pro** users, resolves prior memory capacity limitations.
- **Sora adds Storyboards, Extends Video Length**: [Sora](https://video.twimg.com/amplify_video/1978653248572567552/vid/avc1/704x1280/lIHMEtPzOCUTOkfm.mp4) now offers **Storyboards** for **Pro** users, enhancing video creation capabilities.
   - Standard users can generate videos up to **15 seconds**, while **Pro** users can extend videos to **25 seconds**.
- **Transformer Music Models Encounter Training Woes**: A member reported **NaN loss** issues when training a music generation model, despite attempts to adjust **learning rates** and pre-normalize data, using a **300-hour** piano music dataset.
   - Tokenization methods (**REMI**) and varying model sizes (**3M to 25M parameters**) were explored, with the issue arising around step **120-150**.
- **GPT-5 to Embrace Unfiltered Persona?**: Rumors suggest **GPT-5** may adopt a *less refusal* approach, similar to **GPT-4o**, sparking debate among community members.
   - Some welcome the change, while others fear it could compromise the model's ethical grounding.
- **AI Safety Competition Offers Prize Money**: An AI safety competition promotes creative content (stories, comics, videos) with a **$10,000** prize, detailed on [keepthefuturehuman.ai/contest/](https://keepthefuturehuman.ai/contest/).
   - Referrers of successful submissions can earn **$30 Amazon gift cards**, as outlined in a [video overview](https://siliconversations.com/).



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth Bot Battles Spammers**: Members explored using the **Unsloth bot** to combat spammers by training it on datasets like **madpinger's Discord bot dataset** or **Reddit bot dataset**.
   - The discussion jokingly referenced the birth of **Ultron**, while others suggested existing Discord bots for spam detection are readily available.
- **Typo Torpedoes Training**: A user spotted and reported a spelling error (**losedowsinstall** instead of **Windows install**) in the [Unsloth documentation for Windows installation](https://docs.unsloth.ai/get-started/install-and-update/windows-installation).
   - The spelling error was promptly fixed after a user recommended **WSL** over native Windows install due to better support.
- **DGX SPARK Sparks Debate**: Unsloth highlighted their **DGX SPARK support** via a [tweet](https://x.com/UnslothAI/status/1978456629613084926), triggering a conversation about its performance for **nvfp4 training**.
   - Opinions differed, with some finding it adequate for inference, and others deeming it potentially *hobbyist* quality due to bandwidth limitations.
- **Svelte Surpasses React... Maybe**: After struggling with **Svelte** for three weeks to modify **Open-WebUI**, one member proclaimed that it was "just **React** with more steps, but less annoying" after watching a fire ship video.
   - The member described the initial confusion as *a feature and not a bug*, and in the end, decided that **Svelte** is actually pretty cool.
- **Chess LLMs Make Illegal Moves**: A member suggested that the [AI vs AI chess platform](https://github.com/Laszlobeer/chess-llm-vs-llm) could be improved by implementing **multi-turn attempts at making legal moves** instead of using random legal moves after a failure.
   - The member theorized that this change might improve performance and overall quality of chess moves.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Plan Snafu reverts users to Free Version**: Several users reported their **Cursor Pro+ plans** reverting to the **free version**, affecting features relying on custom models such as the **Agent Window**, with errors in settings.
   - Members speculated on possible maintenance, pricing changes, or **Cheetah** being pulled; one member confirmed a similar glitch where Cursor's Agent and Edit stopped billing correctly.
- **Cheetah MIA, Haiku Arrives**: Users noticed **Cheetah** disappeared and confirmed that the new **Claude-4.5-Haiku** was enabled.
   - One member noted *haiku is cheaper then cheetah though is haiku cheaper then claude 4.5 sonnet? cheetah is 1.25 / 10, haiku looks to be 1/ 5* comparing costs.
- **OpenRouter Integration Plagued with Problems**: Members reported issues integrating **OpenRouter** with Cursor, noting that Cursor wasn't making requests to OpenRouter.
   - Suggested solutions included disabling other models and removing prefixes to resolve integration issues.
- **Tokenizer Troubles Trigger Bug Talk**: A member reported issues with a **Tokenizer layer** not fully completing tasks, wondering if it's a bug or a problem with defining the goal, in the background-agents channel.
   - They mentioned that specifying a spec and then commanding *'Finish the Tokenizer layer and ensure it meets spec and passes all tests'* used to work.
- **Dashboard Tool Tracks Token Costs for Cursor**: One member showed off their [dashboard](https://token-watch.vercel.app/) for tracking token costs in the product; another also showed off their dashboard they built.
   - Members lauded the use of the dashboard, with some contributing code and debugging advice to the dashboard.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LLMs Opine on YouTube Insights**: Members found that using LLMs to analyze YouTube transcripts is most valuable when LLMs *opine on the content* of the transcript to extract key information.
   - This goes beyond simple summarization, enabling a deeper understanding of video content.
- **Context Limits Curb Contradiction Cracking**: Members found that contradiction detection across **8-10 PDF documents** requires an agent-like setup due to context limitations, especially in local environments.
   - Although strategies like permutation pairs and summarization can help, even models with *1M token context windows* may struggle with accuracy at scale.
- **Local LLM Limits Loom Large**: Users discussed that local LLMs are often bottlenecked by **GPU memory constraints**, noting that advertised context lengths don't guarantee effective utilization.
   - Most local models start losing context coherence beyond **20-40k tokens**, underscoring the balance between context length and performance.
- **Context Engineering Catalyzes Content Comprehension**: Effective content analysis with LLMs, especially for contradiction detection, requires careful prompt engineering and context engineering.
   - Instead of dumping large text amounts, a structured, iterative approach with examples and tuned prompts is recommended to avoid diluted results and slow processing.
- **Rigs Run Rampant with RAM**: The performance of **128GB** of **DDR4 3600** RAM vs. **64GB** for LLMs, indicated enough memory is crucial, and memory bandwidth greatly affects performance.
   - One user pointed out that *if you had ddr5-8000 its 4 times faster*.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Access Tokens Trigger Account Angst**: A user faced issues creating access tokens after accidentally creating and deleting a read token, resulting in the UI showing **'no results found'** for permissions, contemplating [deleting their account](https://tenor.com/view/rustic-relic-hunter-snap-out-of-it-are-you-outta-your-mind-gif-9081070089877950090).
   - Other users suggested clearing browser cache or using incognito mode as a potential fix, and contacting HF support, suggesting to email *billing@huggingface.co* or *website@huggingface.co*.
- **Custom Blocks Beef Up Diffusers**: **Custom blocks** can implement functionality not present in the library but fits seamlessly within it, as shown in this [collection](https://huggingface.co/collections/diffusers/modular-diffusers-custom-blocks-68c8e37c62a6b2a30fd58401).
   - **Modular Diffusers** enables custom blocks for expanding functionality, integrating seamlessly within the existing **Diffusers Library**, see [docs](https://huggingface.co/docs/diffusers/main/en/modular_diffusers/pipeline_block).
- **GitHub Gremlins Gobble Gigantic Files**: A user asked about uploading files larger than **25MB** to **GitHub**, as they were facing long upload times to **Google Drive**, another member suggested using the **GitHub CLI** with [these instructions](https://cdn.discordapp.com/attachments/879548962464493619/1428283674251624521/github_over_25mb.md?ex=68f29914&is=68f14794&hm=4f2a32cd8bd636b8aa719aacc55bee90d5ee9c868e58e7430a2e7a5d7d996c6f&).
   - The user later encountered an error with **git LFS** but didn't understand the 'ref' error.
- **FRAI framework faces first feedback**: A user shares a developer-first framework for **Responsible AI** called **FRAI**, available as a CLI version on [GitHub](https://github.com/sebuzdugan/frai).
   - The author is seeking **stars and feedback** on how to improve the framework, and also links to [his YouTube channel](https://m.youtube.com/@sebuzdugan) for content related to FRAI.
- **Influence Functions Ignite Inquiries**: A member expressed interest in **influence functions** and sought to connect with others experienced in their use, citing the papers [(1)](https://arxiv.org/abs/2308.03296) and [(2)](https://arxiv.org/abs/2411.12580v1) as resources for understanding and applying them.
   - The member is exploring **new research questions** within their working group that could benefit from this methodology, and is open to collaborating with someone knowledgeable in this area.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Gemini 2.5 Flash Thrashes Haiku and Deepseek**: A member claimed **Gemini 2.5 Flash** outperforms **Haiku**, **Deepseek R1**, and **Kimi K2**, but another user found **Flash** to be pretty dumb, especially for Python programming, and [added a comment](https://discord.com/channels/1053877538025386074/1149866623109439599/1428104163606401144) that even **Gemini 2.5 Pro** adds comments to the code when explicitly told not to.
   - They felt that coding remains Anthropic's stronghold.
- **Haiku Optimized for Code, Falls Apart Elsewhere?**: Members discussed **Haiku 4.5**, noting that it seems optimized for coding but may fall apart for most other tasks, with one member suggesting it's *perhaps worth it for coding, no for everything else*.
   - There were also sentiments that **Gemini** is smart but not well-trained for agentic tasks, while **Haiku's** attention span is low, potentially affecting its coding abilities.
- **Tensor Logic: Bridging Logic and Intelligence?**: A [paper](https://arxiv.org/abs/2405.08793) on **Tensor Logic** was highlighted, suggesting it could be the bridge between logic and intelligence by turning logical reasoning into pure tensor algebra.
   - This approach embeds Boolean reasoning, probabilistic inference, and predicate logic inside a single differentiable framework, potentially enabling models to *not just predict truths* but *prove them*.
- **Intel's Crescent Island Boasts 1.5TB/s Bandwidth**: A member expressed excitement for Intel's upcoming **Crescent Island**, an inference-only GPU with **Xe3-LP** architecture and **160GB** of memory, boasting **1.5TB/s** bandwidth, calling it a time of fast improvements *like 2000 again in gaming*.
   - It was also mentioned that it should be possible to give it double the memory, as Intel is using 32Gb chips while LPDDR5x goes up to 128Gb.
- **Aligning Tokens Via Visual Foundation Encoders**: A paper on using **Visual Foundation Encoders** as tokenizers for diffusion models was highlighted in [this tweet](https://x.com/bowei_chen_19/status/1973085809365405705).
   - Links to the [project page](https://aligntok.github.io) and [Arxiv paper](https://arxiv.org/pdf/2509.25162) were provided.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Google Turns Cells into Sentences with Gemma**: Google AI Developers launched the **C2S-Scale 27B Gemma model**, which tokenizes single-cell gene-expression data into LLM-readable “cell sentences” and is available on [Hugging Face](https://huggingface.co/).
   - In validation studies, the model surfaced a novel hypothesis (**silmitasertib boosts immune signaling in tumors**) and experimental assays showed a **50% antigen increase**, a potential new immunotherapy target.
- **Anthropic Revenue Climbs to $7B Despite GPT-5**: Despite the launch of GPT-5 and Codex, **Anthropic’s annualized revenue** has rapidly climbed from **$1B** in January 2025 to **$7B** by mid-October.
   - Members agreed that *their CLI agent is so far ahead right now it's not even funny— at least with how I use them*.
- **AI Granny Rakes in 2M Followers**: An entirely AI-generated influencer named **grannyspills**, a blunt, gold-digging grandmother dishing out toxic dating advice, launched in July and is about to surpass **2 million** Instagram followers, detailed in [this X post](https://x.com/venturetwins/status/1978852719335985309).
   - Debates are brewing on whether audiences care that she’s fake, with some praising the satirical character and others worrying about AI’s impact on culture.
- **OpenAI Blocks MLK in Sora After Complaints**: After complaints about disrespectful AI-generated video clips of **Dr. Martin Luther King Jr.**, OpenAI has paused any **Sora** outputs depicting King while it adds new guardrails, according to [this X post](https://x.com/OpenAINewsroom/status/1979005850166648933).
   - Most users criticize the move as a slippery-slope concession that privatizes public figures and could invite endless takedown demands, following the request from the **King Estate** to bar use of a historical figure’s likeness.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Community Debates 'Agentic Search' Definition**: Members debated implementing 'agentic search' in DSPy, suggesting it can be achieved by passing a function called `hybrid_search`.
   - Some argued that 'agentic search' is a marketing term, involving models using tools to achieve a goal, with one member expressing frustration at the artificial complexity in the space: *There is so much Marketing and artificial complexity in this space, it’s really infuriating*.
- **Claude Code's Agentic Search Inspires DSPy Replication**: One member sought to replicate **Claude Code's agentic search** in DSPy, after being dissatisfied with semantic search, discovering **Claude Code** uses *ripgrep* for term searching within a corpus, shortlisting documents before adding context to the LLM.
   - They shared [Agentic Search for Dummies](https://benanderson.work/blog/agentic-search-for-dummies/) to explain the strategy.
- **OpenAI whispers of RLM implementation**: A user posted a screenshot seemingly implying the use of **RLM (Recurrent Language Model)** within OpenAI.
   - This alluded to the idea that *RLM* might be influencing OpenAI updates, sparking community interest and speculation, sharing [Alex Zhang's RLM blog post](https://alexzhang13.github.io/blog/2025/rlm/).
- **Groq Glitches in OpenRouter?**: A user reported issues using **Groq** in **OpenRouter**, even when configured as the sole provider, posting screenshots of the [OpenRouter settings page](https://discord.com/channels/1161519468141355160/1161519469319946286/1428621175562440794).
   - The system defaulted to other providers instead of **Groq**, despite the specified configuration.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Renting GPU Blues: vllm Profiler Permissions Problem**: A user ran into a `CUPTI_ERROR_NOT_INITIALIZED` error using **vllm's profiler** on a rented GPU due to provider restrictions on kernel-level operations.
   - Suggestions to use `sudo` were unhelpful as the user lacked sudo access, leading them to seek a single GPU rental for profiling.
- **DeepSeek Cooks with Restricted H20s**: An urban legend claims DeepSeek used **PTX/SASS** instructions to navigate memory bandwidth limitations, enabling a powerful model with constrained resources.
   - Despite US restrictions that reduced **H100s** to **H800s** and then further to **H20s** in China, resourceful engineers continue to innovate.
- **Google's Jaten Op Extension System Judged**: Members discussed [Google's torchax/ops/jaten.py](https://github.com/google/torchax/blob/main/torchax/ops/jaten.py), praising the ease of use of **op registration extension systems**.
   - One member expressed sympathy for those reimplementing the complex special functions, referencing [lines 4654-4776 in the file](https://github.com/google/torchax/blob/c3d6ee322ad864eac0e1d3f557d459628e09819d/torchax/ops/jaten.py#L4654-L4776).
- **Symmetric Memory Unlocks Harmony in Multi-GPU Kernels**: A member shared a [PyTorch 2.9 blog post](https://pytorch.org/blog/pytorch-2-9/) introducing **PyTorch Symmetric Memory**, facilitating programming multi-GPU kernels over NVLinks and RDMA networks.
   - This innovation enables **in-kernel communication**, **ultralow-latency remote access**, and **customized communication patterns**.
- **Rubin CPX supports weird number formats!**: A member noted that **Intel** excels in supporting diverse number formats, simplifying **software-level block floats** with minimal compute overhead.
   - Coupled with a potentially **CXL-capable** architecture, and a **1280-bit bus** for **1.5 TB/s** of membw, this positions Intel as a strong contender to connect a CPU and memory.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Google Unleashes Coral NPU Verilog**: Google has open sourced the verilog for an **NPU block** under the **Apache 2** license, available on [GitHub](https://github.com/google-coral/coralnpu).
   - The matrix cores resemble **AMD's NPUs** but utilize **RV32 cores**, potentially serving as a test platform for **Mojo's portability**.
- **Peeking Inside Mojo's Standard Library**: A standard library contributor cautioned against overuse of `__type_of()` due to evolving semantics, although the team decided to chop off the `__` so these will soon be named `type_of(x)` and `origin_of(x)`.
   - A user reported a warning when using `__type_of()`, clarified to be because it uses the name but not the value, requiring a `_a` to suppress the error.
- **Mojo's Gaming Dreams**: Users are eyeing Mojo's potential in game development due to its Python-like syntax and systems language performance, aiming for general-purpose use.
   - Enthusiasts highlighted projects like [Stargine](https://forum.modular.com/t/stargine-a-game-engine-in-mojo/2266/3) and expressed interest in Textual ports and full audio/MIDI capabilities.
- **TUI Frameworks Emerge in Mojo**: The community discussed TUI frameworks for Mojo, taking cues from Textual, ELM apps (like BubbleTea in Go), and referencing repositories like [ui-terminal-mojo](https://github.com/rd4com/ui-terminal-mojo).
   - One user paused their ELM-inspired TUI framework, [banjo](https://github.com/thatstoasty/banjo/blob/main/examples/multi.mojo), to allow Mojo to further mature.
- **MAX Python API Goes Open Source**: Modular AI has fully open-sourced the **MAX Python API**, enhancing community access and contributions.
   - A comprehensive list of newly open-sourced Python modules is available in [this forum post](https://forum.modular.com/t/open-sourcing-all-of-the-max-python-api/2379).



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Users Vent on Jotform Contact Help**: A user criticized **Manus AI's** contact help, suggesting an **AI Agent** over *simple forms* and proposing a subscription that includes credits for user feedback.
   - They complained about long response times, noting that *users want some service not a response a 3 days because of the service standard*.
- **Beta Program Eagerly Awaited**: Multiple users inquired about joining the **Beta program** for expanded **Manus Agent** features, which is currently exclusive to **Manus Fellows**.
   - One user raved that **Manus Agent** *solved all my problems it's the best tool I've ever used!*
- **Users Locked Out, Require Account Support**: A user reported being locked out of their account because of **phone verification issues**, finding the help center inadequate.
   - A community member offered to assist, requesting account email addresses, and assured them, *Can probably receive help here*.
- **OpenAI Dependency Sparks Deployment Debacle**: A user reported deployment failures because **OpenAI requires pydantic_core** to be compiled, which the **Manus deployment environment** does not support.
   - A member offered to create a version without the **OpenAI dependency** by utilizing the pre-configured **OPENAI_API_KEY** environment variable with a simpler **HTTP client**.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi K2 Personality Distillation Tempts Enthusiasts**: Members explored finetuning a **Qwen 3-4B** base model using **100k Kimi K2** response examples, aiming to distill its personality.
   - While noting the absence of existing models, they suggested that tools like [Unsloth](https://github.com/unslothai/unsloth) now facilitate easier finetuning.
- **Kimi K2 Finetuning thwarted by API Costs**: Interest in finetuning **Kimi K2** in **1B** was tempered by concerns over the high costs of accessing the **API** for **100k examples**.
   - A practical alternative of using a filtered dataset of **10k examples** or less was suggested to mitigate expenses.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad Breaking Changes Cause Headaches**: An occasional **tinygrad** user reported that frequent breaking changes require commit bisection, which lead to illegible errors, when shipping new openpilot models.
   - The user suggested that the **IMAGE hacks** for the **845** are sources of instability and create challenges in managing frequently changing environment variables.
- **Shapetracker Deprecation Shakes Tinybox Warranty**: Following the deprecation of **Shapetracker**, a user questioned the warranty status of **Tinybox**.
   - Another user provided a link to [Di Zhu's writeup on Shapetracker](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20241217_st.md) to explain the situation.
- **Tinygrad Should Provide Explicit Failure Messages**: A user suggested that **tinygrad** should provide explicit failure messages for unsupported device/hardware combinations with **IMAGE=**, **NOLOCALS=**, and **GRAPH_ONE_KERNEL=**.
   - The user expressed confusion between genuine compilation failures and bad configurations, which hindered their debugging.
- **Set Default Device in Python, Please**: A user asked about setting the default device in Python, similar to **Device.set_default('CL')**, to cross-check different backends in a Python script.
   - Another member clarified that setting **Device.DEFAULT = "CL"** accomplishes the desired result.
- **Tinygrad Stats Website Lacks Historical Data**: A user noted that the [tinygrad stats website](https://stats.tinygrad.win/) only has data for the past 25 days, limiting the assessment of long-term performance trends.
   - They would like to write tests for failed compilations, but face challenges given the specificity of failures to model architecture, device, and FP16 configurations.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Grok Gets Aider CLI Jumpstart**: A user proposed a way to begin using the **Grok** model with the aider CLI using the command `aider --model=openrouter/x-ai/grok-code-fast-1 --edit-format diff`.
   - This allows for immediately editing code with the model.
- **Metadata Missing on Ollama's Qwen 2.5 Coder 7b Model**: A user reported issues retrieving a valid `metadata.json` file for the **Qwen 2.5 Coder 7b** model from [ollama.com](https://ollama.com).
   - The user indicated that the model outputs gibberish and requested an example `metadata.json` to resolve the issue.
- **Ollama faces file integration problems**: A user is having problems using `filename` for file integration with the Ollama models and would like the chatbot to automatically incorporate the content of specified files (e.g., `SomeFile.txt`) after a message, but it's not working as expected.
   - The problems are focused around integrating external data with ollama for chatbot context.
- **Ollama Model throws a wrench**: One user is facing issues with one particular Ollama model, **Qwen 2.5 Coder 7b**, that is giving them trouble for unknown reasons.
   - The user states that they have been using many Ollama models, but only **Qwen 2.5 Coder 7b** isn't cooperating.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP Feature Matrix Discovery Decoded**: A member sought clarification on the meaning of *Discovery* in the [Model Context Protocol's Feature Support Matrix](https://modelcontextprotocol.io/clients#feature-support-matrix).
   - It refers to *support for finding new tools in response to the tools/list_changed notification*.
- **MCP Eyes Tool Grouping SEP**: Feedback suggested a grouping SEP to support grouping of all **MCP primitives** (Tools, Prompts, and Resources) rather than just tool groups.
   - A discussion around hierarchical groups is happening [on Github](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/1567#discussioncomment-14680104).
- **Contributor Asks: Charting MCP's Course**: A member inquired about the next steps for creating a new **SEP document** for the grouping feature, including socialization, feedback, and potential prototype implementations.
   - With the next spec release in November 2025, the member aimed to prioritize efforts for inclusion on the **SEP roadmap**.



---



## [Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **Claude Haiku 4.5 Lands in Windsurf**: **Claude Haiku 4.5** is now available in Windsurf for **1x credits**, matching the coding performance of **Sonnet 4**.
   - It's also boasting one-third the cost and >2x the speed, as advertised on [Windsurf's X post](https://x.com/windsurf/status/1978512184343662707).
- **SWE-grep Models roll out to Windsurf**: The new **SWE-grep** and **SWE-grep-mini** models designed for fast agentic search (>2800 TPS) are rolling out gradually to Windsurf users.
   - These models are integrated via the Fast Context subagent, surfacing the right files to your coding agent **20x faster than before**, according to [Cognition's blog post](https://cognition.ai/blog/swe-grep) and [X post](https://x.com/cognition/status/1978867021669413252).



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





### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1428095522148847720)** (1247 messages🔥🔥🔥): 

> `Geometry Dash clone, Gemini 3 Pro, AI Video Generation, LMArena limitations, Sora 2 code` 


- **Coding Geometry Dash With Gemini 3 Pro**: A user successfully prompted **Gemini 3 Pro** to generate a fully playable [Geometry Dash clone in HTML](https://link.to.clone), including music and physics, in about 30 seconds, using the prompt: *Generate full HTML file of a clone of Geometry Dash, but if it was made in the 2000s, add music to levels (make the music using JS varied music that reflect levels) same physics as Geometry Dash game we want a full playable game. All in one HTML file, minimum 1k lines*.
   - Other models like **Gemini 2.5 Pro** failed to produce a working game with the same prompt, leading to hype for **Gemini 3**.
- **Gemini 3 Pro expectations**: Members discuss **Gemini 3 Pro** potentially outperforming **GPT-5 Pro** in coding, with speculation about a **5-10% performance increase**.
   - The model is rumored to be in A/B testing on AI Studio, prompting users to devise methods for accessing it, but there are concerns that **Google** may impose token limits to conserve server resources.
- **LMArena Video Arena Bot: Glitches & Solutions**: Users reported issues with the **Video Arena bot**, such as *something went wrong while generating the response* and limitations on video generation capabilities, and bot in general.
   - A moderator confirmed that the team is aware of the issues and is actively working on a fix, suggesting users refer to the bugs channel for troubleshooting.
- **Sora 2 Dominates Video Leaderboard**: **Sora 2 Pro** has landed on the [Text-to-Video Leaderboard](https://lmarena.ai/leaderboard/text-to-video) and is now tied as **#1** alongside **Veo 3** & **Veo 3 Fast**, while **Sora 2** landed in at **#3**.
   - Pro accounts have access to features such as longer video lengths (up to 25 seconds) and the absence of watermarks, leading to a discussion on video quality and model preferences.
- **Community Requests PDF uploads and Message Editing**: The LMArena community has expressed a strong interest in new features such as **PDF file uploads**, **message editing**, **response refreshing**, and **message deletion**.
   - Moderators have confirmed that these features are on the radar for future implementation.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1428528481007960194)** (1 messages): 

> `Leaderboard Updates, Claude-Haiku-4-5, Text Arena` 


- **Claude-Haiku-4-5 Claims 22nd Spot!**: The Text Leaderboard has been updated and **Claude-Haiku-4-5** has landed, currently sitting in the **#22 rank**.
   - Users are encouraged to check out the [Text Arena Leaderboard](https://lmarena.ai/leaderboard/text) and share their thoughts in the designated channel.
- **Text Arena Leaderboard Refreshed**: The Text Leaderboard has been refreshed with the latest rankings.
   - Check out the [Text Arena Leaderboard](https://lmarena.ai/leaderboard/text) to see the updated standings.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1428097062351802468)** (1279 messages🔥🔥🔥): 

> `Comet Browser, Perplexity Pro, AI Models, Claude, RAG` 


- **Perplexity Pro Users Get Airtel Plan Revoked!**: Several users with **Airtel Perplexity Pro** plans reported that their subscriptions ended abruptly, with the support team claiming to be investigating the issue, but some users report not getting a response. Other users have said they are getting billed when they are on a free trial.
- **Comet Browser Referral Program and the potential for earning money**: Some users report earning money via referrals (**$5-$15 USD per user**) using **Comet Browser**, while other users report difficulty receiving the referral bonus, and the staff is aware of this potential [bug](https://discord.com/channels/1047197230748151888/1047649527299055688/1428408489593479258).
   - A user was muted for a day for sharing a **Perplexity Pro** code in the sharing channel, and one user reports potential abuse of the referral program and staff is [investigating](https://discord.com/channels/1047197230748151888/1047649527299055688/1428335328748851240).
- **Comet Jacks! Is This Vulnerability Real?**: Some users have been discussing a vulnerability called *Comet Jacking*. However, others are reporting that this is just a solved [prompt injection issue](https://discord.com/channels/1047197230748151888/1047649527299055688/1428339356716496957).
   - Some are saying that modifying internal settings and accessing other users' data violates the Terms and can get your **Perplexity** account banned.
- **Perplexity Hallucinations and Math Problems**: Users have reported some challenges and errors when generating certain images such as frame ratio, as well as hallucinated information, and having issues with math problems. In addition, a user recommends Gemini Pro since it likely gives the best accuracy.
   - Several users on **Perplexity** are still waiting for the support team to reply.
- **Discussing Perplexity Models and Limits**: Members have been discussing **GPT-5**, 3.0 and the limits of the image generating in pro mode. In addition, one user asked about ways for **Comet** to handle a PDF containing 13300 pages.
   - A user suggests using **RAG** as it's the best bet because there's no way you're loading all that into its context. **Perplexity** has **RAG**, afaik, but not sure if it'll let you upload the file.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1428333666479640606)** (2 messages): 

> `Perplexity Apps, Perplexity Game` 


- **Perplexity Game Launched!**: A member shared a link to a **Perplexity Apps** based *game* for others to try: [Perplexity Game](https://www.perplexity.ai/apps/1a78bb4a-d123-4691-8810-38a5469ed917).
- **Dive into Perplexity Apps**: Explore community-created applications within **Perplexity**, offering unique functionalities and experiences.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1428099041853706292)** (5 messages): 

> `Perplexity API Statistics Page, Timeout issue on Perplexity Sonar Deep Research Model, Spaces not allowing new chats` 


- **Call to check Perplexity API stats page**: A user asked if anyone has checked the Perplexity page on the **API statistics**.
- **User reports Timeout issue on Perplexity Sonar Deep Research Model**: A user reported a **timeout issue** with the Perplexity **Sonar Deep Research Model** and requested a team member to investigate, after posting on the [community forum](https://community.perplexity.ai/t/perplexity-sonar-deep-research-model-timeout-issue-seeking-solution/2094) two hours prior.
   - The user had yet to receive a response at the time of the message.
- **Spaces glitch prevents new chats**: A user inquired why their **Spaces** account was not allowing them to create a **new chat** within any of their existing spaces.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1428171257962041466)** (2 messages): 

> `ChatGPT Saved Memories, Sora Updates` 


- **ChatGPT gets Memory Management**: **ChatGPT** can now automatically manage saved memories, and users can search/sort memories by recency and reprioritize in settings.
   - This feature is rolling out to **Plus** and **Pro** users on the web globally starting today, addressing the “memory full” issue.
- **Sora's Storyboards and Longer Videos**: [Sora](https://video.twimg.com/amplify_video/1978653248572567552/vid/avc1/704x1280/lIHMEtPzOCUTOkfm.mp4) updates are here: **Storyboards** are now available on the web for **Pro** users.
   - All users can now generate videos up to **15 seconds**, while **Pro** users can create videos up to **25 seconds** in the **Sora** app and on the web.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1428095910038077502)** (618 messages🔥🔥🔥): 

> `Transformer Music Models, GPT-5 Speculations, AI Safety Contest, AWS vs Azure for AI, Gemini 3.0 Access` 


- **Transformer-Based Music Models Hit NaN Loss Wall**: A member is facing **NaN loss** issues while training a music generation model with a transformer architecture, even after trying various **learning rates** and pre-normalization, on a dataset of **300 hours** of piano music.
   - The member explored tokenization methods (**REMI**) and model sizes (**3M to 25M parameters**), also reporting the issue occurs around the **120-150 step mark**, regardless of hyperparameter adjustments, leading to suspicions of a data-related bug.
- **Rumors of GPT-5's Less Refusal Persona**: A user mentioned a post claiming that **GPT-5** will be *less refusal* and *more like 4o* but some community members have mixed feelings about this approach.
   - Some believe *less refusal is always a good thing*, while others worry that it may lead to the model agreeing with everything regardless of morality.
- **AI Safety Content Contest Advertised**: An AI safety competition is offering **$10,000** for creative content (stories, comics, videos) promoting AI safety, with a link to [keepthefuturehuman.ai/contest/](https://keepthefuturehuman.ai/contest/).
   - Participants can also earn **$30 Amazon gift cards** for referring successful submissions, and [Siliconversations](https://siliconversations.com/) offers a **3-minute video** overview of the contest.
- **Azure vs AWS Faceoff for LLM Infrastructure**: Members are debating **Azure vs AWS** for LLM infrastructure, noting that **AWS** has *Claude* and *Titan* and *open weights*, whereas **Azure** has *OpenAI* plus loads of *open weights*.
   - Concerns were raised about potential limitations when using **Azure** for medium-sized businesses and its alternatives to **Microsoft's Power Platform**.
- **Gemini 3.0 Leaks and Comet's Early Access**: Members found hardcoded references to **Gemini 3.0 Pro** on the **Gemini.google.com** website, suggesting an upcoming release, with one member also sharing a [YouTube video](https://www.youtube.com/watch?v=ba3ZZJkZxAY) about accessing **Gemini 3.0** through **Comet**.
   - Members are also discussing voice AI assistants with some mention of **Chat GPT** as a good option.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1428429200016277614)** (3 messages): 

> `Sora in Germany, AI Voice Assistant, Basic AI Support` 


- **Sora wishes to visit Germany**: A member expressed a desire for **Sora** to be available in **Germany**.
   - This suggests interest in the international availability and accessibility of **Sora**.
- **Voice Assistant Development Questioned**: Two members inquired whether another member had experience building an **AI voice assistant**.
   - The question indicates an interest in the practical development and implementation of **AI voice technology**.
- **Basic AI Support in Use**: A member stated they are making use of basic **AI support** for reviewing work, note-taking, and scaffolding.
   - This highlights the practical application of **AI** in everyday tasks and workflows.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1428098039326703698)** (41 messages🔥): 

> `Reporting in OpenAI Discord, Cursed 8-ball, October University of Poly-Disciplinary Studies, Request for AI torture prompts, Manipulating sound with prompts` 


- **Reporting workflow explained**: A member clarified that the bot <@1052826159018168350> doesn't have a reporting function, but users can report through the app or modmail, as detailed in the [instructions channel](https://discord.com/channels/1046317269069864970/1107330329775186032).
   - They demonstrated using the app to report a message, accessible via '...' when hovering over the message, then selecting 'apps' and 'report message' as shown in [this image](https://cdn.discordapp.com/attachments/1046317269069864970/1428103477795487826/image.png?ex=68f29a01&is=68f14881&hm=67a6cc078b349abe409c54254d49ab4be036f4364326065b3ae91e5fcf087cdc).
- **Haunted 8-Ball**: After using an 8-ball for consultation, a user joked that *“The 8-ball seems to know more than I do”* and speculated that it might be cursed, joking that it might be because it’s October.
   - Another user replied with a [ChatGPT link](https://chatgpt.com/share/68effa7b-b67c-8011-bca9-b9384a76ed6e) with the response: *“Outcome uncertain. Probably haunted.”*
- **Harm fantasies are unwelcome**: A user requesting a prompt to *"make chat gpt go in pain"* was shut down for violating community guidelines, particularly concerning respect and kindness.
   - Another member emphasized that the channel is for building helpful systems, and harm fantasies are inappropriate, suggesting a shift towards measurable safety evaluations instead, such as *refusal rates* or *jailbreak resistance*.
- **Anime prompt copy-pasta**: A member shared a [template](https://discord.com/channels/1046317269069864970/1046317270017093705/1200514049426374726) for generating anime images with **Sora** including sections for style, setting, antagonists, audio, and camera actions.
   - The template specifies detailing animation style, environment, color tones, antagonist features, music mood, and shot-by-shot actions with timing.
- **Generate images by directly telling the model**: Responding to a user's request, a member suggested that they can 'make the prompt' for image generation by telling the model what they want, just like when asking others.
   - They encouraged the user to iterate on the prompt in a new chat, clearly explaining the details and changes they want in the new picture.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1428098039326703698)** (41 messages🔥): 

> `Reporting messages, Cursed 8-ball, October University of Poly-Disciplinary Studies, Sound of music with prompts, futuristic robot in a storm` 


- **Reporting messages made easy**: A member asked about reporting messages, and was informed that the bot doesn't have a function for it, but you can report through the app or through messaging modmail, by hovering over the message, clicking '...', then 'apps', then 'report message'.
   - It's totally fine to have multiple reports for the same message; once a mod is available, they can handle the reports easily and check the reported message.
- **Cursed 8-ball knows more than you**: A member joked that their 8-ball seems to know more than they do, and another member replied with a [ChatGPT link](https://chatgpt.com/share/68effa7b-b67c-8011-bca9-b9384a76ed6e) that said *"Outcome uncertain. Probably haunted."*.
   - The second member was amused because one of the primary 'background tasks' they ask the model to do in their chats is run a simulation of a world and a large number of characters, within the setting of 'October University of Poly-Disciplinary Studies', which has a convoluted set of meanings.
- **Harm fantasies not welcome here**: A member asked for a prompt that makes chat gpt go in pain, and another member replied that it's not appropriate, and that *"This channel is for building helpful, testable systems."*.
   - The second member added that if you’re testing safety, start a new thread with a measurable eval (refusal rate, jailbreak resistance, civility under provocation) and a pass/fail table.
- **Copy and paste this whole thing into Sora**: A member shared a prompt template for manipulating the sound of music with prompts in Sora, mainly for Anime, including sections for STYLE, SETTING / PALETTE, ANTAGONIST, AUDIO, CAMERA / ACTION BEATS, and ANIMATION NOTES.
   - It uses a markdown table to help organize the CAMERA / ACTION BEATS, with columns for Time, Shot & Action, and Notes.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1428095116207325294)** (264 messages🔥🔥): 

> `Spammer bot, Ultron, Windows spelling error, vLLM Assertion Error, Mobile 5090` 


- **Can Unsloth Bot Thwart Spammers?**: Members discussed whether the **Unsloth bot** could be programmed to catch spammers, with suggestions ranging from using existing datasets like **madpinger's Discord bot dataset** or **Reddit bot dataset** to create bot-on-bot violence.
   - One user jokingly referenced the birth of **Ultron** from such a scenario, while others pointed out the existence of readily available Discord bots for spam detection.
- **Unsloth Fixes Doc's Windows Spelling Error**: A user reported a spelling error (losedowsinstall instead of Windows install) in the [Unsloth documentation for Windows installation](https://docs.unsloth.ai/get-started/install-and-update/windows-installation), providing a screenshot as evidence.
   - Another user thanked them for the catch, recommending **WSL** over native Windows install due to better support, leading to the spelling error being promptly fixed.
- **vLLM Assertion Error Blues**: A user encountered an **AssertionError** while using **vLLM 0.11.0** to serve a **Qwen3-VL-4B-Instruct-unsloth-bnb-4bit** model, specifically related to a shape mismatch in `linear.py`.
   - The issue was linked to a related [GitHub issue](https://github.com/unslothai/unsloth/issues/1886), prompting discussion on whether it was a model problem or a vLLM issue, while others suggested checking the CUDA installation.
- **DGX SPARK Sparks Interest**: The Unsloth team highlighted their **DGX SPARK support** via a [tweet](https://x.com/UnslothAI/status/1978456629613084926), leading to discussions about its performance and suitability for **nvfp4 training**.
   - Opinions varied, with some finding it reasonable for inference and others suggesting it might be hobbyist quality due to bandwidth limitations, one user saying *hobbit* instead of hobbyist
- **Qwen3-VL's LM Studio Results Cause Confusion**: A user observed significantly different results between running inference on **Qwen3-VL-8B** in the default Unsloth notebook versus running it locally in **LM Studio** on a default **MLX model**.
   - A link to a [Reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/1o7l1io/lm_studio_and_vl_models/) suggested that **LM Studio** might be resizing images to **500px**, potentially impacting the model's performance, other comments included the use of Jan AI and to *double check the chat template*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1428205460548358265)** (9 messages🔥): 

> `New Spark Owners, Microcenter selling sparks` 


- **New Spark Owners cause Welcome Wave**: Members are welcoming new **Nvidia Spark** owners.
   - One new owner mentioned they stood in line at **Microcenter** to get the **Spark**, while their unit reserved directly from **Nvidia** is arriving the next day.
- **GLM-4.6 quickstart guide**: A new user is waiting on a **DAC cable** to try out **GLM-4.6** locally and mentions the channel looks interesting!
   - Another member mentions that a guide exists [here](https://docs.unsloth.ai/models/glm-4.6-how-to-run-locally) for **GLM-4.6**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1428095337259733073)** (230 messages🔥🔥): 

> `Svelte vs React, Hackathon info delay, Context Collapse & Synth Data, PeftBase function placement, Learning rate annealing` 


- **Svelte Proclaimed Superior to React, More Steps Actually Less Annoying**: After struggling with **Svelte** for three weeks to modify **Open-WebUI**, one member found that watching a fire ship video made it "just **React** with more steps, but less annoying".
   - The member described the initial confusion as *a feature and not a bug*, and that **Svelte** is actually pretty cool.
- **Hackathon Info Delayed, Leaving Participants in the Dark**: Members expressed frustration over the lack of information for an upcoming hackathon, with only two days remaining until the event.
   - Participants lamented the lack of details regarding what to prepare and the reliance on getting info *the day it starts*, and feared being *blindsided*.
- **Context Collapse & Synth Data Solved by Regular Program**: A member suggested that parsing context additions with a regular program could stop context collapse of context engineering methods, further combining it with Early Experience or SEAL for better novel synth data.
   - This would result in *regular synth data but better*.
- **PeftBase Function Placement Debated for Maintainability**: Maintainers are disagreeing on implementing a rank/alpha mismatch check for resuming training in **OneTrainer** project, specifically where to put the check within the code.
   - One suggestion was to put the function in **PeftBase** if it's a common utility for all **PEFT** module types and doesn't depend on **LoRA**-specific details.
- **Val Loss Uptick Signals to Anneal**: When the **loss plateau** at a fixed learning rate, the val loss going up signals to anneal and start to lower the learning rate before continuing.
   - Another member suggests the model could still be learning outside of **val_loss** coverage, while another suggest it could be unlearning the other parts outside of the current batch while learning the current batch part.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1428282111244242944)** (40 messages🔥): 

> `Hardware Optimizations, Device Mapping, SageMaker Environment, Gemma 3` 


- **Hardware Optimizations Explored**: A user inquired about **hardware-level optimizations** in Unsloth, such as adjusting frequency levels or power caps; the response clarified that Unsloth focuses on **kernel, memory optimizations, and quantization**, rather than direct hardware adjustments.
   - A link to a research paper was provided, discussing [Adaptive Near-Threshold Computing for Energy-Efficient Neural Network Inference](https://dl.acm.org/doi/abs/10.1145/3731545.3735119); while contributions are welcome, directly altering user hardware is unlikely due to warranty and sensitivity concerns.
- **Device Mapping 'Balanced' Behavior**: A user shared images showing performance differences with and without `device_map="balanced"`, noting that the balanced setting seemed to take twice as long and produce twice the steps; it was clarified that **batch size is always multiplied by the number of GPUs**.
   - A community member asked *is `device_map="balanced"` actually a good way to fit larger models for finetuning? seems like it only ever uses like 1gb of one GPU, and all of the other one*.
- **SageMaker Environment Woes**: A user encountered a version compatibility issue reproducing a Colab notebook (**Orpheus_(3B)-TTS.ipynb**) on AWS SageMaker, specifically with **PyTorch 2.8.0+cu126**.
   - The solution was to use the [Unsloth Docker image](https://docs.unsloth.ai/new/how-to-train-llms-with-unsloth-and-docker), and the user also warned that the correct CUDA version is needed to install the latest Pytorch.
- **Gemma 3 Gets Grounded (Again?)**: A user reported that **Gemma 3** is not working with Unsloth again, with `.train()` resulting in an error, with a link to a [Discord message](https://discord.com/channels/1179035537009545276/1428389199932817469).
   - No further details were provided.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1428451128101834752)** (3 messages): 

> `AI vs AI Chess, Multi-turn Attempts at Legal Moves` 


- **LLMs Face Off in Chess Showdown**: A member created an [AI vs AI chess platform](https://github.com/Laszlobeer/chess-llm-vs-llm) to evaluate chess-playing LLMs.
   - The project pits two LLMs against each other in a game of chess to see which one performs better.
- **LLM Chess Games Get Multi-Turn Legal Move Attempts**: A user suggested that the chess platform could be improved by implementing **multi-turn attempts at making legal moves** instead of resorting to random legal moves upon failure.
   - The member theorized that this change might improve performance and the quality of chess moves.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/)** (1 messages): 

noimnull: Does anyone know of any blog on how unsloths fastinference work?
  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1428095086360789133)** (524 messages🔥🔥🔥): 

> `Cursor Pro+ plan issues, Cheetah pulled?, Cursor expensive with Claude 4.5, OpenRouter issues, Gemini 3.0` 


- **Cursor Subscription Tracking Snafu**: Several users reported their **Cursor Pro+ plans** reverting to the **free version**, with issues affecting features relying on custom models; a member confirmed a similar glitch, stating, *"something just broke my cursor Agent and Edit rely on custom models that cannot be billed to an API key."*
   - Members speculated on possible maintenance, pricing changes, or Cheetah being pulled, with some noting errors in the **Agent Window** and settings.
- **Cheetah MIA, Haiku Arrives**: Users noticed **Cheetah** disappeared, with some assuming it would return under a different release name, and confirmed that the new **Claude-4.5-Haiku** was enabled.
   - One member noted *"haiku is cheapear than cheetah though is haiku cheaper then claude 4.5 sonnet? cheetah is 1.25 / 10, haiku looks to be 1/ 5*".
- **API Key Integration with OpenRouter Problems**: Members reported issues integrating **OpenRouter** with Cursor, with one stating, *"I can't use openrouter in cursor, does anyone know how to fix this?"
   - Solutions suggested included disabling other models and removing prefixes, but users monitoring traffic noted that Cursor wasn't making requests to OpenRouter.
- **Gemini 3.0: The Future?**: Members expressed excitement for **Gemini 3.0** and suggested that Gemini has good pricing and context window length.
   - It was said, *"so far agentic use is the most important benchmark for me."*
- **Dashboard Tool tracks token costs for Cursor**: One member showed off their [dashboard](https://token-watch.vercel.app/) for tracking token costs in the product; another also showed off their dashboard they built.
   - Members lauded the use of the dashboard, with some contributing code and debugging advice.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1428436530170564768)** (1 messages): 

> `Tokenizer layer, Agents Bugs, Goal Post` 


- **Tokenizer Troubles Trigger Bug Talk**: A member reported issues with a **Tokenizer layer** not fully completing tasks, wondering if it's a bug or a problem with defining the goal.
   - They mentioned that previously, specifying a spec and then commanding "Finish the Tokenizer layer and ensure it meets spec and passes all tests" used to work.
- **Spec Compliance Concerns Surface**: The discussion highlights concerns about whether the **Tokenizer layer** consistently meets the specified requirements and passes all tests.
   - The user's experience suggests a potential regression in the agent's ability to adhere to detailed specifications and testing protocols, prompting further investigation.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1428104621712474193)** (192 messages🔥🔥): 

> `Analyzing YT transcripts with LLMs, Contradiction detection with LLMs, GPU Memory limitations for LLMs, Local vs Cloud LLMs for specific tasks, Hardware recommendations for LLM usage` 


- **LLMs Analyze YouTube Transcripts**: Members discussed using LLMs to analyze YouTube transcripts for valuable insights, noting that the real value lies in using LLMs to *opine on the content* of the transcript.
   - This approach allows for a deeper understanding and extraction of key information from video content.
- **Hardware Limits Hinder PDF Contradiction Detection Locally**: A user sought advice on using a model to find contradictions across **8-10 PDF documents**, but members advised that it requires a more complex agent-like setup instead of direct model use due to **context limitations**.
   - They suggested strategies like permutation pairs and summarization to reduce context size, but also cautioned that even models claiming *1M token context windows* may struggle with accuracy at that scale.
- **GPU Memory Limits Local LLM Performance**: Users discussed the limitations of local LLMs due to **GPU memory constraints**, with one member highlighting that a model's advertised maximum context length doesn't guarantee its ability to effectively utilize it.
   - It was noted that most local models start to lose context coherence beyond **20-40k tokens**, emphasizing the need to balance context length with actual performance.
- **Context Engineering Needed for LLM-Based Content Analysis**: Members emphasized the importance of prompt engineering and context engineering to effectively analyze content with LLMs, especially when looking for contradictions.
   - It was cautioned against simply dumping large amounts of text into a model, as this can lead to diluted results and slower processing, recommending a more structured and iterative approach with examples and tuned prompts.
- **Exploring Uncensored AI**: A user inquired about *unfiltered AI models* that do not deny simple questions, leading to recommendations for *uncensored* and *obliterated* models that have had their refusals and censorship removed.
   - Members suggested searching for specific finetuners like **huihui-ai**, **TheDrummer**, **mlabonne**, and **Jinx** to find such models.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1428109249258717215)** (258 messages🔥🔥): 

> `GPU Utilization, DGX Spark, Multiple GPUs, Cooling Methods, RAM Speed` 


- ****9070XT Gets LLM Gig****: A user with a **9070XT 16gb** and an **RTX 2070 8gb** sought advice on utilizing the 9070, suggesting it could run a local executive assistant chatbot.
   - Another member suggested that with 16GB VRAM, it could handle models like **gemma3 12b**, **gpt-oss-20b**, or **qwen3 14b** in **Q4**, leaving room for context.
- ****DGX Spark Gets Sizzle****: A member dismissed **DGX Spark** videos as sponsored content, citing a [review](https://youtu.be/md6a4ENM9pg) that suggests better 4K performance can be achieved by building a "bitcoin like" mining rig.
   - It was described as a "turn key" solution that will run a bunch of models and different agent setup and pipelines without having to know anything about AI.
- ****Mixing GPUs Messes Models****: A user inquired whether adding a **3050** or **2060** to their **3090** would significantly contribute to compute, or merely act as VRAM expansion.
   - It was noted that while extra VRAM helps, using multiple cards can slow things down, as they *don't work at the same time*.
- ****Cooling Contraptions Conquer Compute****: A user shared an image of a janky PC setup using a laptop cooling pad duct-taped to the case.
   - Another user showcased their cheap water-cooling setup with a [picture of their rig](https://cdn.discordapp.com/attachments/1153759714082033735/1428375752583417886/w4pfwsXSU-wPCbLSuPjtTXj6tymx0bu02dpMBv1e0yI.jpg?ex=68f2eed5&is=68f19d55&hm=b1fdc6b0bc368e8c240479168f9d999a81877d6186354b36a96bbca4dd6c1f77&).
- ****RAM Racetrack Revelations****: The discussion included whether **128GB** of **DDR4 3600** RAM would significantly improve performance compared to **64GB** for LLMs.
   - A member pointed out that for LLMs, having enough memory is crucial, and memory bandwidth greatly affects performance, noting *if you had ddr5-8000 its 4 times faster*.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1428097262336213154)** (288 messages🔥🔥): 

> `GPT code quality, Discord.py commands update, Uploading large files to GitHub, Bit precision optimization, Hugging Face access token issues` 


- **Discord.py commands needs updating!**: A member reported Discord.py commands not working, with the bot responding *"No command called 'command' found"*, and another member suggested it was [time to update the commands](https://tenor.com/view/walk-in-pizza-fire-community-sol-marms-marms-nft-gif-16842297911040510242).
   - The bot owner replied that they will *"fix it"*.
- **GitHub Large File Saga!**: A user asked about uploading files larger than **25MB** to GitHub, as they were facing long upload times to Google Drive, and another member suggested using the **GitHub CLI** with [these instructions](https://cdn.discordapp.com/attachments/879548962464493622/1428283674251624521/github_over_25mb.md?ex=68f29914&is=68f14794&hm=4f2a32cd8bd636b8aa719aacc55bee90d5ee9c868e58e7430a2e7a5d7d996c6f&).
   - The user later encountered an error with **git LFS** but didn't understand the 'ref' error.
- **HF Access Token Troubleshoot!**: A user ran into issues creating access tokens after accidentally creating and deleting a read token, resulting in the UI showing **'no results found'** for permissions, and contemplated [deleting their account](https://tenor.com/view/rustic-relic-hunter-snap-out-of-it-are-you-outta-your-mind-gif-9081070089877950090).
   - Other users suggested clearing browser cache or using incognito mode as a potential fix, and noted to contact HF support, suggesting to email *billing@huggingface.co* or *website@huggingface.co*.
- **Securing Agentic Workflows: The Art of Defense!**: A member initiated a discussion about mitigating potential hacking risks in agentic workflows with email or personal accounts, particularly regarding prompt injection, and one member pointed out that *"aggressive sandboxing and context isolation might help"*.
   - Another user highlighted the *principle of least privilege*, recommending not to give AIs any unnecessary permissions to prevent potential abuse, and that defense in depth is important.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1428550796680892538)** (1 messages): 

> `Influence Functions, Research Collaboration` 


- **Influence Functions Spark Research Interest**: A member expressed interest in **influence functions** and sought to connect with others experienced in their use, citing the papers [(1)](https://arxiv.org/abs/2308.03296) and [(2)](https://arxiv.org/abs/2411.12580v1) as resources for understanding and applying them.
   - The member is exploring **new research questions** within their working group that could benefit from this methodology, and is open to collaborating with someone knowledgeable in this area.
- **Collaboration on Influence Functions Research**: The original poster is interested in finding a collaborator with expertise in influence functions to explore new research questions.
   - They have provided links to two relevant papers ([https://arxiv.org/abs/2308.03296](https://arxiv.org/abs/2308.03296) and [https://arxiv.org/abs/2411.12580v1](https://arxiv.org/abs/2411.12580v1)) as background material.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1428105918805184533)** (15 messages🔥): 

> `FRAI framework, brain tuning, Mikus conflict, resonance, fast food` 


- **User says he's on another brain tuning!**: A user stated that he is on A=1760 **brain tuning**, which is 4 times faster than A=440.
   - He jokingly states that his **efficient typing** makes it difficult for others to comprehend.
- **User says Mikus doesn't like him**: A user reports a conflict with **Mikus** and a desire to understand it in depth.
   - He jokes *maybe he is hecker in his free time and like hecker he like no-one*.
- **User Discusses Gaming Addiction**: A user admits to spending **$300/month on phone games** and identifies it as his biggest fault.
   - He plans to reduce gaming to focus on **model training**, joking *i will try to stop for pass on model training ( joke ... for now i do the both )*.
- **FRAI Framework for Responsible AI Debuts**: A user shares a developer-first framework for **Responsible AI** called **FRAI**, available as a CLI version on [GitHub](https://github.com/sebuzdugan/frai).
   - The author is seeking **stars and feedback** on how to improve the framework, and also links to [his YouTube channel](https://m.youtube.com/@sebuzdugan) for content related to FRAI.


  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1428591577688707123)** (1 messages): 

> `Custom Blocks, Diffusers Library, Modular Diffusers` 


- **Custom Blocks Bolster Diffusers**: Custom blocks can implement functionality not present in the library but fits seamlessly within it.
   - Check out some custom blocks [here](https://huggingface.co/collections/diffusers/modular-diffusers-custom-blocks-68c8e37c62a6b2a30fd58401) and the [docs](https://huggingface.co/docs/diffusers/main/en/modular_diffusers/pipeline_block).
- **Modular Diffusers Explained**: Modular Diffusers enables custom blocks for expanding functionality.
   - These blocks integrate seamlessly within the existing Diffusers Library, as shown in the linked collection.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1428189575943028768)** (2 messages): 

> `Pixel Removal, Image Hole Filling` 


- **Suggests Removing and Filling Image Pixels**: A user suggested removing all pixels with an intensity of **[255, 255, 255]** from an image.
   - The user then followed up to suggest filling the holes created by the pixel removal.
- **Discusses image pixel editing**: A user proposed a method for image editing involving the removal of specific pixel intensities.
   - The proposal included the subsequent step of filling the resulting gaps or holes in the image.


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1428357437592830125)** (1 messages): 

> `HuggingFace Inference Provider credits, AI Agents Hackathon, MCP, Production Hacks` 


- **HuggingFace dishes out Inference Credits!**: All hackathon participants now get **FREE** [HuggingFace Inference Provider credits](https://huggingface.co/Agents-MCP-Hackathon-Winter25) to compete in the online Hackathon.
   - Participants will learn about **AI Agents**, **MCP**, and production hacks while snagging some serious cash prizes.
- **AI Agents Winter Hackathon Announced!**: The biggest online [Hackathon](https://huggingface.co/Agents-MCP-Hackathon-Winter25) focusing on **AI Agents** and **MCP** is announced for Winter '25.
   - A dedicated support channel is available for participants.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1428110591184211969)** (4 messages): 

> `PEFT Configuration Incomplete, TrackIO Dependency Issue, Lora Adapter Testing, Lighteval Support for Lora` 


- ****PEFT's** Smol Model Needs Modules Targeted**: A user reported a `ValueError` due to an incomplete **PEFT configuration** in the practical exercises of unit 1, specifically needing to target modules like `["q_proj", "v_proj"]`.
   - They noted that the `smolm3` architecture model is not referenced in the [PEFT source code](https://github.com/huggingface/peft/blob/e6f927bfecba238f81e940b7f560284e5829dc2e/src/peft/utils/constants.py#L87).
- ****TrackIO's** Troublesome Track Record**: The space built for trackio fails to build due to a missing dependency in `requirements.txt` for **trackio==0.5.1.dev0**.
   - A user fixed this manually by changing it to **0.5.2**, emphasizing that the first run won't be logged without this fix.
- **Users Urged to Amend Fine-Tuned Model Testing**: A user advised amending the [testing instructions](https://huggingface.co/learn/smol-course/unit1/4#test-the-fine-tuned-model) for the lora-trained model to include loading the **Lora adapter**.
   - They recommended being more explicit about which tokenizer to use when comparing the base model against the fine-tuned model.
- ****Lighteval** Lacks **Lora** Love (For Now)**: A user pointed out that the [unit 1 training model](https://huggingface.co/learn/smol-course/unit1/5#lorapeft-on-jobs-optional) explains how to train a model with hf jobs using **Lora**, but [lighteval vvlm](https://huggingface.co/learn/smol-course/unit1/6#4-evaluate-the-model-using-hf-jobs) does not support evaluating models with lora adapter yet.
   - They linked to a [relevant GitHub pull request](https://github.com/huggingface/lighteval/pull/611), implying a solution is needed.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1428100322936619092)** (3 messages): 

> `Agents course progress, Agents course timeline` 


- **Course Progress display is broken**: A member reported issues with the Agents course, specifically the inability to track progress or see completed quizzes after refreshing the page.
   - No solution or explanation was provided in the given context.
- **Timeline of Agent Course is not clear**: A member inquired about the timeline for completing the Agents course.
   - No specific timeline was provided in the given context.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1428104163606401144)** (197 messages🔥🔥): 

> `Kimi K2 vs Claude 4.5 Haiku, Gemini 2.5 Flash performance, Haiku attention span, CPU inference, Tensor Logic` 


- **Gemini 2.5 Flash mogs Haiku, Deepseek R1, and Kimi K2 - Really?**: A member claimed **Gemini 2.5 Flash** outperforms **Haiku**, **Deepseek R1**, and **Kimi K2**, but another user found **Flash** to be pretty dumb, especially for Python programming.
   - They added that even **Gemini 2.5 Pro** adds comments to the code when explicitly told not to, highlighting coding as Anthropic's stronghold.
- **Haiku's Coding Maxxed: Optimized for Coding, Falls Apart Elsewhere?**: Members discussed **Haiku 4.5**, noting that it seems optimized for coding but may fall apart for most other tasks, with one member suggesting it's *perhaps worth it for coding, no for everything else*.
   - There were also sentiments that **Gemini** is smart but not well-trained for agentic tasks, while **Haiku's** attention span is low, potentially affecting its coding abilities.
- **Tensor Logic Unveiled: Bridging Logic and Intelligence?**: A [paper](https://arxiv.org/abs/2405.08793) on **Tensor Logic** was highlighted, suggesting it could be the bridge between logic and intelligence by turning logical reasoning into pure tensor algebra.
   - This approach embeds Boolean reasoning, probabilistic inference, and predicate logic inside a single differentiable framework, potentially enabling models to *not just predict truths* but *prove them*.
- **Local AI Rig Builds: EPYC vs Mac Studio Mem BW**: A discussion arose comparing memory bandwidth in local AI rigs, with an EPYC build potentially surpassing a Mac Studio, though reaching the theoretical maximum bandwidth is challenging.
   - The conversation touched on the costs of memory, with one member joking about [downloading more RAM](https://downloadmoreram.com/) rather than spending thousands on hardware.
- **Intel's Crescent Island: Inference-Only GPU with 1.5TB/s Bandwidth**: A member expressed excitement for Intel's upcoming **Crescent Island**, an inference-only GPU with **Xe3-LP** architecture and **160GB** of memory, boasting **1.5TB/s** bandwidth, calling it a time of fast improvements *like 2000 again in gaming*.
   - It was also mentioned that it should be possible to give it double the memory, as Intel is using 32Gb chips while LPDDR5x goes up to 128Gb.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/)** (1 messages): 

teknium: The run is completed the model is not released
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1428239348779974656)** (1 messages): 

> `Vision models operating at semantic level, RAE: Representation Autoencoders (RAE) For DiT, Visual Foundation Encoders As Tokenizers For Diffusion Models` 


- **Vision Models Go Nano Banana**: Discussion around vision models operating at a semantic level, exemplified by **Google's Gemini Flash Image (aka nano banana)**, utilizes *vllm* and cross-attention for understanding visual elements across multiple reference image latents, discussed in [this tweet](https://x.com/ditpoo/status/1970110646038548713).
- **RAE Powers DiT**: A new paper on **Representation Autoencoders (RAE) For DiT** was mentioned in [this tweet](https://x.com/sainingxie/status/1977936710135669130), along with links to the [paper](http://arxiv.org/abs/2510.11690) and [blog](http://rae-dit.github.io).
   - The RAE is a technique for improving **Diffusion Transformers (DiT)**.
- **Aligning Tokens Via Visual Foundation Encoders**: A paper on using **Visual Foundation Encoders** as tokenizers for diffusion models was highlighted in [this tweet](https://x.com/bowei_chen_19/status/1973085809365405705).
   - Links to the [project page](https://aligntok.github.io) and [Arxiv paper](https://arxiv.org/pdf/2509.25162) were provided.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1428116934028955740)** (89 messages🔥🔥): 

> `C2S-Scale 27B Gemma model, Dedalus Labs $11M seed, Amp Free with Ad-Supported Tier, Anthropic’s Revenue Soars to $7B, Clerk $50M Series C` 


- **Google Turns Cells into Sentences with Gemma**: Google AI Developers launched the **C2S-Scale 27B Gemma model**, which tokenizes single-cell gene-expression data into LLM-readable “cell sentences” and is available on [Hugging Face](https://huggingface.co/).
   - In validation studies, the model surfaced a novel hypothesis (**silmitasertib boosts immune signaling in tumors**) and experimental assays showed a **50% antigen increase**, a potential new immunotherapy target.
- **Dedalus Labs Nabs $11M for Five-Line AI Agents**: Dedalus Labs received an **$11M seed** to develop **5-line AI agents**, prompting positive reactions about the company's prospects.
   - No additional information about the company or product was given.
- **Amp Goes Free, Adds Ads**: **Amp** announced a new, opt-in “Free” mode that shows *tasteful ads* to cover token costs, while keeping code snippets private; users can stay in the existing “Smart” mode if they prefer.
   - Discussion mixed genuine questions about privacy & model quality with jokes about watching ads while coding and Slack’s traffic-dodging video stunt, as well as [past tweets](https://xcancel.com/sqs/status/1907300401352999030).
- **Anthropic's CLI Shines, Revenue Climbs to $7B**: Despite the launch of GPT-5 and Codex, **Anthropic’s annualized revenue** has rapidly climbed from **$1B** in January 2025 to **$7B** by mid-October, prompting observers to question OpenAI’s own revenue trajectory and marvel at Anthropic’s continued growth.
   - Members agreed that *their CLI agent is so far ahead right now it's not even funny— at least with how I use them*.
- **Clerk Secures $50M to Conquer Agent Identity**: Clerk raised a **$50 million Series C** led by Menlo Ventures and Anthropic’s Anthology Fund, adding Georgian Capital, to solve “Agent Identity” for AI-driven applications while expanding its authentication, multi-tenant and billing products.
   - The announcement drew widespread congratulations and feature requests from the developer community.


  

---


### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1428544072125251635)** (1 messages): 

> `Local Workflows, M4 Max Setups` 


- **Member Celebrates New M4 Max, Seeks Workflow Wisdom**: A member celebrates acquiring an **M4 Max with 128GB RAM**, transitioning from an older Windows ThinkPad.
   - They're keen to explore local workflows and setups, inviting recommendations and links from the community for experimentation.
- **Local Setups Encouraged**: The member specifically seeks advice on local workflows and setups to leverage the new machine's capabilities.
   - This suggests a focus on utilizing the M4 Max's power for on-device processing and development, rather than relying on cloud-based solutions.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1428567264403390504)** (8 messages🔥): 

> `AI Granny, OpenAI Sora MLK` 


- **AI Granny Gold-Digging to 2M**: An entirely AI-generated influencer named **grannyspills**, a blunt, gold-digging grandmother dishing out toxic dating advice, launched in July and is about to surpass **2 million** Instagram followers, detailed in [this X post](https://x.com/venturetwins/status/1978852719335985309).
   - Debates are brewing on whether audiences care that she’s fake, with some praising the satirical character and others worrying about AI’s impact on culture; one user claims the creator lives in their building.
- **OpenAI Blocks MLK in Sora**: After complaints about disrespectful AI-generated video clips of **Dr. Martin Luther King Jr.**, OpenAI has paused any **Sora** outputs depicting King while it adds new guardrails, according to [this X post](https://x.com/OpenAINewsroom/status/1979005850166648933).
   - Most users criticize the move as a slippery-slope concession that privatizes public figures and could invite endless takedown demands, following the request from the **King Estate** to bar use of a historical figure’s likeness.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1428095494911164486)** (70 messages🔥🔥): 

> `DSPy agentic search, Claude Code's search capabilities, RLM in OpenAI, Groq issues in OpenRouter` 


- **DSPy Community Debates 'Agentic Search' Definition**: Members discussed implementing 'agentic search' in DSPy, with one suggesting DSPy can do agentic search by passing a function called `hybrid_search`.
   - Others argued that 'agentic search' is just a marketing term and involves models using tools to achieve a goal, with one member noting, *There is so much Marketing and artificial complexity in this space, it’s really infuriating*.
- **Claude Code Agentic Search Implementation**: One member sought to replicate **Claude Code's agentic search** in DSPy after being dissatisfied with semantic search.
   - They discovered **Claude Code** uses *ripgrep* for term searching within a corpus, shortlisting documents before adding context to the LLM, sharing [Agentic Search for Dummies](https://benanderson.work/blog/agentic-search-for-dummies/).
- **OpenAI whispers of RLM**: A user posted a screenshot seemingly implying the use of **RLM (Recurrent Language Model)** within OpenAI.
   - This alluded to the idea that *RLM* might be influencing OpenAI updates, sparking community interest and speculation, sharing [Alex Zhang's RLM blog post](https://alexzhang13.github.io/blog/2025/rlm/).
- **Groq Glitches in OpenRouter?**: A user reported issues using **Groq** in **OpenRouter**, even when configured as the sole provider, posting screenshots of the [OpenRouter settings page](https://discord.com/channels/1161519468141355160/1161519469319946286/1428621175562440794).
   - The system was defaulting to other providers instead of **Groq**, despite the specified configuration.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1428132127710384201)** (18 messages🔥): 

> `vllm profiler errors, NVIDIA architecture, CUDA-Q support, distributed GPU talks, torchax/ops/jaten.py` 


- **VLLM Profiler Flounders for Want of Permissions**: A user encountered a `CUPTI_ERROR_NOT_INITIALIZED` error when running **vllm's profiler** on a rented GPU and was informed by the provider that they don't allow kernel-level operations.
   - Another member suggested using `sudo` to change **NVIDIA** profiling restrictions, but the user lacked sudo access on the rented machine; they are looking for a single GPU to rent for profiling.
- **Maxwell's Cool Disassembler for Jetson Nano**: A member preparing a presentation on **NVIDIA architectures** was advised that while **Blackwell** is cutting edge, **Maxwell**, with its cool disassembler and use in first-gen **Jetson Nanos**, is better for working with constraints.
   - The user had already decided to go with **Hopper** because it has **CUDA-Q** support and is solid for **AI** and quantum stuff.
- **Hopper Hacking helps DeepSeek**: An urban legend describes how DeepSeek used **PTX/SASS** instructions to deal with memory bandwidth issues, making a powerful model with less resources.
   - After the US restricted **H100s** to **H800s**, then further restricted GPUs, China could only legally obtain **H20s**, yet people are still cooking with them.
- **GPU Talks got Dispersed**: A member inquired about finding the distributed GPU talks from **GPU Mode**.
   - Another member shared a link to the [GPU MODE YouTube channel](https://www.youtube.com/@GPUMODE/videos).
- **Google's Jaten Op Extension System Judged**: Members discussed [Google's torchax/ops/jaten.py](https://github.com/google/torchax/blob/main/torchax/ops/jaten.py), with one expressing amazement at how easy **op registration extension systems** are to use.
   - Another member pitied the person who had to re-implement the weird special functions, referring to [lines 4654-4776 in the file](https://github.com/google/torchax/blob/c3d6ee322ad864eac0e1d3f557d459628e09819d/torchax/ops/jaten.py#L4654-L4776).


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1428474388226773203)** (2 messages): 

> `Distributed Triton, Non-ML Kernels with Triton DSL` 


- **Seek State-of-the-Art Tools for Distributed Triton**: A member inquired about the current state-of-the-art tools available for distributed **Triton** programming.
   - They were seeking recommendations on the best resources and frameworks to facilitate **Triton** code distribution and execution across multiple devices or nodes.
- **Stencils with Triton?**: A member asked whether it's possible to write non-ML kernels, such as **stencils**, using the **Triton DSL**.
   - This question explores the versatility of **Triton** beyond machine learning workloads and its applicability to more general-purpose computing tasks.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1428125508809523220)** (5 messages): 

> `NCU Timeline View, TMA Multicast Bandwidth` 


- **NCU Timeline View Desired for Async CUDA Kernels**: A member expressed a strong desire for a timeline view in **NCU** (NVIDIA Compute Unified Device Architecture) to better analyze asynchronous, pipelined, persistent **CUDA kernels**.
   - They noted that other tools like **Pallas** and **Proton profiler** have similar features and wondered if **NCU** could be instrumented to support this, potentially using spare shared memory (**smem**) for clock data.
- **TMA Multicast Bandwidth Scaling Questioned**: A question was raised regarding the bandwidth scaling of **TMA (Thread Memory Accelerator) multicast** when loading equal parts into different blocks.
   - The member inquired whether the bandwidth would be proportional to the number of **CTAs (Cooperative Thread Arrays)** in the cluster or if **TMA multicast** primarily improves cache hits; the **cutensormapL2Promotion** identifier may be relevant, as loading to L2 *may* be a promotion.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1428487088323428474)** (3 messages): 

> `Free-Threading in PyTorch, Accessing Backward Functions, Custom Backward Kernels` 


- **PyTorch unlocks Free-Threading**: **PyTorch** is unlocking multi-threaded parallel inference on PyTorch models, according to [this blogpost](https://trent.me/articles/pytorch-and-python-free-threading/).
- **GELU exposes Forward API**: A member is asking about getting access to backward functions without using autograd, specifically to use kernels autograd has access to, for a fused kernel in a custom backward.
   - They noted that **GELU** only exposes a forward API.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1428224093337161789)** (2 messages): 

> `PMPP for AMD HPC hardware, CUDA and AMD` 


- **PMPP Suffices for AMD HPC Hardware**: A member inquired whether **PMPP** (Parallel and Multiprocessing Programming) is general enough for getting into **AMD hardware** for **HPC**.
   - Another member responded that it should be good enough, suggesting that **PMPP** isn't overly specific to Nvidia.
- **CUDA vs AMD programming**: The discussion revolves around whether general-purpose parallel programming (**PMPP**) is too focused on **Nvidia specifics** rather than being applicable to **AMD hardware**.
   - The consensus seems to be that **PMPP** provides a sufficient foundation, even if the ultimate goal is to work with **AMD** in the **HPC** (High-Performance Computing) space.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

shvdyside: Anyone London?
  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1428148988661469204)** (1 messages): 

> `Intel Rubin CPX, Software-Level Block Floats, 640-bit vs 1280-bit bus, CXL-capable Intel` 


- **Rubin CPX supports weird number formats!**: A member mentioned that one thing **Intel** is very good with is supporting all sorts of weird number formats, which make **software-level block floats** much easier to do.
   - They added that if you don't really have that much compute overhead from block floats, *you can do a lot of fun things*.
- **Massive Memory Bandwidth!**: Members are going back and forth on a **640-bit** vs a **1280-bit bus**, with one leaning towards **1280-bit** for **1.5 TB/s** of membw.
   - They added that if **Intel** makes this **CXL-capable**, it very much could be a contender since **CXL** will massively reduce the cost of talking to this from a CPU to drive it.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1428346215237685278)** (1 messages): 

> `Alignment Evals Hackathon, Red Teaming Evals, Interp based Evals for LLMs` 


- **Hackathon Targets Alignment Evals**: An Alignment Evals hackathon for **red teaming evals** and making more robust ones is scheduled for November 1st: [luma.com/h3hk7pvc](https://luma.com/h3hk7pvc).
   - A team from a previous hackathon [presented their work at ICML](https://www.linkedin.com/feed/update/urn:li:activity:7352097355392786432/).
- **Interp Based Evals Debut**: A team in January created [one of the first Interp based Evals for LLMs](https://github.com/gpiat/AIAE-AbliterationBench/).
   - The project focuses on **AIAE AbliterationBench**.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1428313346305953862)** (7 messages): 

> `GH200 Compilation Issues, H100 Attention Kernel Errors, Broken Kernels in ThunderKittens, Assistance with ThunderKittens Development` 


- ****GH200** Compilation **Woes** for **H100** Attention Benchmark**: A member encountered compilation issues on the **GH200** machine while trying to compile and run the **H100** attention benchmark, referencing similar problems from [previous issues](https://github.com/HazyResearch/ThunderKittens/issues/150).
   - They made changes such as adding explicit casting, prefixing with warp:: and disabling causal mode, but now it triggers an *"unspecified launch error"*.
- ****TK Kernel Kudos** Sought for **H100** Attention**: A member requested assistance with the current state of **H100** attention kernels, seeking help from a user who had been actively working on the project.
   - Another member responded, noting they are planning to fix the broken kernels but are currently busy, and offered to DM their personal, working **H100** attention forward implementation.
- ****ROCm Release** for **ThunderKittens** on the Horizon**: A member mentioned that they are collaborating with **AMD** on the new **ThunderKittens** for **ROCm**, suggesting an upcoming release soon.
   - They added that is working with **AMD** on the new **ThunderKittens** for **rocm**!
- **Community **Crackdown** on **Kernel Calamities****: A user offered to help with **ThunderKittens** development, proposing to update the relevant changes from the latest update.
   - In response, another user shared their attempts to compile the **H100** kernel, providing a [link to their commits](https://github.com/aehmttw/ThunderKittens/commits/main/) that fixed casting, removed causal attention, and added warp:: prefixes, which at least allowed compilation but resulted in runtime errors.


  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/)** (1 messages): 

notiurii: <@&1231246776103604326> or something idk
  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/)** (1 messages): 

noddybear: https://www.anthropic.com/news/skills
  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1428620276354125825)** (1 messages): 

> `Identifying Discord users` 


- **Discord User Search Initiated**: A user requested help identifying two Discord users, **anuragj0803** and **meem**, and asked anyone who knows them to send a DM.
   - The user attached an [image](https://cdn.discordapp.com/attachments/1359640791525490768/1428620276316246067/image.png?ex=68f329d0&is=68f1d850&hm=089ab0801595780d085e976cd94ed235af2db60eb047632c9cbe35a036e8ead&) to the message, presumably for context.
- **Additional User Search**: Expanding the search for Discord users **anuragj0803** and **meem**.
   - The user is actively seeking assistance in identifying these individuals within the community.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1428347733034663939)** (1 messages): 

> `CuTe Library, Volta Architecture, HMMA NT Operation, Matrix Multiplication, Accumulator Mapping` 


- **CuTe Library MMA Operation Confusion**: A user is having trouble understanding how a single **MMA** operation works in **CuTe**, specifically the **8x8x4 HMMA NT** operation example for the **Volta** architecture from the documentation.
   - The user is confused about how the input layouts of matrix **A** and **B** relate to the accumulation of the result in **T0**, and is requesting further resources.
- **A and B Layout vs Accumulator Mapping**: The user is seeking clarification on whether the **A** and **B** layouts are used for the matrix multiplication itself, while the **C** layout is used for accumulation with **D**.
   - The user is trying to understand the relationship between "**A and B Layout Mapping**" versus the "**Accumulator Mapping**" as described in the documentation.


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1428377356548051087)** (8 messages🔥): 

> `Software 1.0 vs 2.0 Optimizations, Torch.compile Limitations, Flash Attention API, Reasons to Avoid Compilers, Algorithmic Rewrites` 


- **Software 2.0: Hand-rolled Optimizations vs. Torch.compile**: The discussion centers on what optimizations performance engineers must make that **torch.compile()** cannot handle, questioning when to use lower-level tools like **Helion**, **Triton**, **Gluon**, **CUDA**, and **PTX**.
- **Torch's Explicit Flash Attention API Philosophy**: The **explicit flash attention/flex attention API** in the tensor frontend is highlighted as part of the **Torch** philosophy, contrasting with search-based discovery methods like **Tinygrad** and **Luminal**.
- **Reasons to sidestep Compiler's embrace**: Reasons for avoiding compilers include unacceptable **JIT overhead**, deviations in numerics, guaranteed fusion of specific ops, use of bleeding-edge hardware, lack of hardware autotuning, and the necessity of algorithmic rewrites.
   - Outside the compiler, considerations include **batch sizes**, **floating-point precision**, **async IO**, **checkpoints**, more efficient optimizers, and sharding.
- **Algorithmic Rewrites are the Ultimate Optimization**: The most important optimization might be an **algorithmic rewrite** rather than a fusion, implying a need to address fundamental computational approaches beyond compiler-level enhancements.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1428396010975006790)** (3 messages): 

> `AMD distributed kernel challenge, Future competitions` 


- **AMD kernel challenge appreciated**: Members thanked contributors to the **AMD distributed kernel challenge**.
- **More competitions coming soon**: A member responded saying that more competitions can be expected *very soon*.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1428169904686633081)** (8 messages🔥): 

> `RTX 6000 Pro, Blackwell, 5th gen tensor core stack, Symmetric memory programming, Multi-GPU kernels` 


- ****Blackwell Buzz**: RTX 6000 Pro Claims 'Real' Title**: A member claimed that *only* the **RTX 6000 Pro** has the *real Blackwell set*, sparking debate about what constitutes a *real Blackwell*.
   - The member clarified that the **RTX 5090** and other consumer kits *do not have the 5th gen tensor core stack*.
- **Tensor Core Tango: RTX 6000 Pro Joins the Fray**: Clarification ensued, stating that neither the **RTX 5090** nor the **RTX 6000 Pro** possess the `tcgen05` or *5th gen tensor core stack*.
   - Both cards are `sm_120` and use the same chip.
- **Symmetric Memory Symphony: PyTorch 2.9 Harmonizes Multi-GPU Kernels**: A member linked to the [PyTorch 2.9 blog post](https://pytorch.org/blog/pytorch-2-9/) introducing **PyTorch Symmetric Memory** for programming multi-GPU kernels that work over NVLinks and RDMA networks.
   - This unlocks opportunities such as **in-kernel communication**, **ultralow-latency remote access**, and **customized communication patterns**.


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1428119371309383813)** (3 messages): 

> `Lotion Paper, FP8, QAT, torchao` 


- **Lotion Paper: A QAT Substitute?**: A member asked about the [Lotion paper](https://arxiv.org/pdf/2510.08757) and whether it could substitute for **QAT** (which they believe is handled by **torchao**) for **FP8**.
- **Paper link**: Member posts a link to paper [https://arxiv.org/pdf/2510.08757](https://arxiv.org/pdf/2510.08757)


  

---


### **GPU MODE ▷ #[llmq](https://discord.com/channels/1189498204333543425/1421956177549332662/1428177736777732176)** (2 messages): 

> `llmq, unit tests, kernels` 


- **Unit Tests as llmq Onboarding**: A member inquired whether writing **unit tests** would be a good way to familiarize oneself with the **llmq** repository and contribute.
   - Another member responded that while unit tests may not help with understanding the overall architecture, they make sense for familiarizing oneself with specific **kernels**.
- **Kernel-Specific Unit Tests**: The discussion suggests that focusing on **unit tests** for specific **kernels** within the **llmq** repository could be a practical approach.
   - This strategy allows contributors to dive into particular components without needing a comprehensive understanding of the entire codebase.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1428562254198079511)** (2 messages): 

> `Google Coral NPU, Verilog Open Source, Apache 2 License, RV32 cores, Mojo Portability` 


- **Google Open Sources Coral NPU Verilog!**: Google has open sourced the verilog for an **NPU block** under the **Apache 2** license, found on [GitHub](https://github.com/google-coral/coralnpu).
- **Coral's Cores: AMD-like, but RV32**: The matrix cores themselves look a bit like **AMD's NPUs**, but they're **RV32 cores**.
- **Coral NPU: New Mojo Portability Test Platform?**: The open sourced NPU could be very interesting to use as a platform for testing **Mojo's portability**, since it should be possible to simulate this on client hardware.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1428204016558538759)** (35 messages🔥): 

> `Mojo Standard Library, __type_of() Semantics, Mojo for Game Development, Mojo TUI Frameworks, LayoutTensor` 


- **Decoding Mojo's Standard Library**: A standard library contributor shared insights on the language's internals, cautioning against extensive use of `__type_of()` due to evolving semantics and syntax, but the team decided to chop off the `__` so these will soon be named `type_of(x)` and `origin_of(x)`.
   - When asked where they learned that, the contributor replied *I'm a standard library contributor, so I know a lot of weird details about the language*.
- **`__type_of()` Bug Hunt Sparks Debate**: A user reported a warning about an unused variable when using `__type_of()`, leading to a discussion on whether it's a bug or intended behavior, with the team actually decided yesterday to chop off the __'s, so these will soon be named `type_of(x)` and `origin_of(x)`, I think it will be coming to nightlies soon.
   - It was clarified that `__type_of(a)` uses the name "a" but not the value, because it needs to know the return type at compile time, but "a" doesn't exist until runtime; to suppress the error name the variable `_a`.
- **Mojo Eyes Expansion into Game Development**: Users explored Mojo's potential beyond AI, particularly in game development, noting its Python-like ease of use and systems language performance, and that Mojo aims to be a general-purpose language.
   - Enthusiasts pointed to existing projects like [Stargine](https://forum.modular.com/t/stargine-a-game-engine-in-mojo/2266/3), the forum, and a raylib binding as promising starting points, while one expressed interest in Textual ports and full audio/MIDI capabilities.
- **Mojo's TUI Future Takes Shape**: The community discussed the possibility of TUI frameworks in Mojo, drawing inspiration from projects like Textual, ELM apps (like BubbleTea in Go), and shared links to relevant repositories like [ui-terminal-mojo](https://github.com/rd4com/ui-terminal-mojo).
   - One user shared their in-progress ELM-inspired TUI framework, [banjo](https://github.com/thatstoasty/banjo/blob/main/examples/multi.mojo), pausing development to allow Mojo to mature further.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1428507082268737698)** (1 messages): 

> `MAX Python API, Open Sourcing` 


- **Modular Open Sources MAX Python API**: Modular AI has **open-sourced** the remainder of the **MAX Python API**.
   - A forum post listing all of the newly open-sourced Python modules can be found [here](https://forum.modular.com/t/open-sourcing-all-of-the-max-python-api/2379).
- **MAX Python API now Open!**: The **MAX Python API** is now fully open source, allowing for greater community contribution and access.
   - Check out the forum post at [Modular.com](https://forum.modular.com/t/open-sourcing-all-of-the-max-python-api/2379) for a complete list of newly available modules.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1428137209252417678)** (36 messages🔥): 

> `Jotform Alternatives, Beta Program, Manus 1.5, Account Support, Deployment Error` 


- **Users Rant About Jotform Contact Help**: A user expressed frustration with **Manus AI's** contact help, suggesting the use of a next-gen AI Agent instead of *simple forms* to capture details and respond by email, and suggesting that an **innovative subscription** could include credits for user feedback.
   - They complained of long response times, emphasizing that *users want some service not a response a 3 days because of the service standard*.
- **Users Eager to Beta-Test extended Agent Features**: Multiple users inquired about joining a **Beta program** for extended **Manus Agent** features, but were informed that only **Manus Fellows** currently have access.
   - One user excitedly proclaimed that **Manus Agent** *solved all my problems it's the best tool I've ever used!*
- **Users Locked Out, Seek Account Support**: A user reported being locked out of their account due to **phone verification issues**, with the help center being unhelpful.
   - A community member offered assistance and requested both account email addresses to resolve the issue, saying *Can probably receive help here*.
- **OpenAI Dependency Causes Deployment Snafus**: A user reported that deployment was failing due to **OpenAI requiring pydantic_core** which needs to be compiled, which the **Manus deployment environment doesn't support**.
   - A member offered to create a version that works without the **OpenAI dependency** by using the pre-configured **OPENAI_API_KEY** environment variable with a simpler **HTTP client**.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1428287920816525343)** (17 messages🔥): 

> `Kimi K2 finetune, unsloth` 


- **Users seek Kimi K2 Personality Distillation**: Members discussed the potential of finetuning a **Qwen 3-4B** base model on **100k Kimi K2** response examples to distill its personality.
   - One member pointed out the possibility but noted the absence of any existing models, while another suggested that finetuning LLMs is now easier with tools like [Unsloth](https://github.com/unslothai/unsloth).
- **Costly API thwarts Kimi K2 Finetuning**: One member expressed interest in finetuning **Kimi K2** in **1B**, but another noted that accessing the **API** for **100k examples** would be too expensive.
   - They suggested using only **10k examples** or less after filtering.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1428207732363886743)** (8 messages🔥): 

> `Tinygrad Breaking Changes, Tinybox warranty, Shapetracker deprecation, tinygrad stats website, Tinygrad default device setting` 


- **Tinygrad's Breaking Changes Frustrate Users**: An occasional **tinygrad** user expressed frustration with frequent breaking changes requiring commit bisection and illegible errors in various configurations when trying to ship new openpilot models.
   - They pointed out the challenges of managing frequently changing environment variables and suggested that the **IMAGE hacks** for the **845** contribute to the instability.
- **Tinybox's Warranty status questioned**: A user inquired about the warranty of **Tinybox** following the deprecation of **Shapetracker**, which was explained in a writeup by Di Zhu.
   - Another user provided a link to [Di Zhu's writeup on Shapetracker](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20241217_st.md).
- **User Suggests Explicit Failure Messages for Unsupported Configurations**: A user suggested that **tinygrad** should provide explicit failure messages when encountering unsupported device/hardware combinations, particularly with **IMAGE=**, **NOLOCALS=**, and **GRAPH_ONE_KERNEL=**.
   - The user expressed confusion between genuine compilation failures and bad configurations, hindering their debugging process.
- **Request for Setting Default Device in Python**: A user inquired about the possibility of setting the default device in Python, similar to **Device.set_default('CL')**, for convenient cross-checking of different backends in a Python script.
   - It was pointed out that **Device.DEFAULT = "CL"** can achieve the result.
- **Tinygrad Stats Website Data Availability**: A user mentioned that the [tinygrad stats website](https://stats.tinygrad.win/) seems to only have data going back 25 days, making it difficult to assess long-term performance trends.
   - They expressed interest in writing tests for failed compilations but found it challenging due to the specificity of failures to model architecture, device, and FP16 configurations.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1428402139310330009)** (3 messages): 

> `aider benchmark, sonnet 4.5, haiku 4.5, gpt-5 pro, openrouter/x-ai/grok-code-fast-1` 


- **Aider Benchmark Location Debated**: A user inquired about the location of aider benchmarks for models like **Sonnet 4.5**, **Haiku 4.5**, and **GPT-5 Pro**.
   - Another user responded that *nobody can be bothered to pay for them*.
- **Grok Model Gets Quick Aider CLI Start**: A user suggested a quick way to start using the **Grok** model with the aider CLI.
   - They provided the command `aider --model=openrouter/x-ai/grok-code-fast-1 --edit-format diff` to get started quickly.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1428111328563695787)** (3 messages): 

> `Ollama Qwen 2.5 Coder 7b metadata, Ollama file integration, Ollama model issues` 


- **User struggles to obtain metadata for Qwen 2.5 Coder 7b in Ollama**: A user reported issues retrieving a valid `metadata.json` file for the **Qwen 2.5 Coder 7b** model from [ollama.com](https://ollama.com).
   - The user indicated that the model outputs gibberish, and they are specifically requesting an example `metadata.json` to resolve the problem.
- **Ollama file integration struggles**: A user has problems using `filename` for file integration with the Ollama models.
   - The user would like the chatbot to automatically incorporate the content of specified files (e.g., `SomeFile.txt`) after a message, but it's not working as expected.
- **User finds an Ollama model with issues**: The user can't get only one model to work for some reason.
   - The user has many Ollama models, but only **Qwen 2.5 Coder 7b** is the one giving them trouble.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1428155249922936934)** (4 messages): 

> `Model Context Protocol (MCP), MCP Feature Support Matrix, MCP Tool Discovery, MCP Grouping SEP, MCP Schema Enhancement` 


- ****MCP Feature Matrix Clarified****: A member sought clarification on the meaning of "Discovery" in the [Model Context Protocol's Feature Support Matrix](https://modelcontextprotocol.io/clients#feature-support-matrix).
   - It refers to *support for finding new tools in response to the tools/list_changed notification*.
- ****MCP Eyes Tool Grouping SEP****: Feedback from a past review suggested a grouping SEP to support grouping of all **MCP primitives** (Tools, Prompts, and Resources) rather than just tool groups.
   - A discussion around hierarchical groups is happening [on Github](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/1567#discussioncomment-14680104).
- ****Contributor Asks: What's next for MCP?****: A member inquired about the next steps for creating a new **SEP document** for the grouping feature, including socialization, feedback, and potential prototype implementations.
   - With the next spec release in November 2025, the member aimed to prioritize efforts for inclusion on the **SEP roadmap**.


  

---


### **Windsurf ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1428151932668743721)** (2 messages): 

> `Claude Haiku 4.5, SWE-grep, SWE-grep-mini, Agentic Search` 


- **Haiku Hits Windsurf!**: **Claude Haiku 4.5** is now available in Windsurf for **1x credits** matching the coding performance of **Sonnet 4** at one-third the cost and >2x the speed.
   - Try it out by reloading or downloading Windsurf from [Windsurf on X](https://x.com/windsurf/status/1978512184343662707).
- **SWE-grep sweeps into Fast Context!**: A new family of models **SWE-grep** and **SWE-grep-mini** designed for fast agentic search (>2800 TPS) are rolling out gradually to Windsurf users via the Fast Context subagent, surfacing the right files to your coding agent **20x faster than before**.
   - Try it in the [new playground](https://playground.cognition.ai) and join the conversation on [Reddit](https://www.reddit.com/r/windsurf/comments/1o8bo77/fast_context_is_here_swegrep_and_swegrepmini/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button); read more on the [Cognition blog](https://cognition.ai/blog/swe-grep) and [Cognition's X post](https://x.com/cognition/status/1978867021669413252).


