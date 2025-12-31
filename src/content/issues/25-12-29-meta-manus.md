---
id: MjAyNS0x
title: >-
  Meta Superintelligence Labs acquires Manus AI for ~$4B, at $100M ARR, 9months after
  launch
date: '2025-12-29T05:44:39.731046Z'
description: >-
  **Manus** achieved a rapid growth trajectory in 2025, raising **$500M** from
  Benchmark and reaching **$100M ARR** before being acquired by **Meta** for an
  estimated **$4B**. The **vLLM** team launched a dedicated community site with
  new resources, while performance issues with **AMD MI300X FP8** were noted in
  **vLLM** and **sglang** benchmarks. **Weaviate** released operational features
  including **Object TTL**, **Java v6 client GA**, and **multimodal document
  embeddings**. API fragmentation concerns were raised by **Teknium** advocating
  for unified SDK wrappers. In open-weight models, **GLM-4.7** gained
  recognition as a reliable coding model with faster throughput on **Baseten**,
  and **MiniMax-M2.1** rose as a leading open agentic coder model, topping
  WebDev leaderboards.
companies:
  - manus
  - benchmark
  - meta-ai-fair
  - vllm
  - amd
  - sglang
  - weaviate
  - teknim
  - baseten
  - alphaxiv
  - minimax
models:
  - glm-4.7
  - minimax-m2.1
  - vllm
topics:
  - performance-optimization
  - inference-frameworks
  - model-benchmarking
  - model-deployment
  - open-source-models
  - multimodality
  - api
  - code-generation
  - community-building
people:
  - alex_wang
  - nat_friedman
---


**It's Agent Lab summer.**

> AI News for 12/29/2025-12/30/2025. We checked 12 subreddits, 544 Twitters and 24 Discords (208 channels, and 3555 messages) for you. Estimated reading time saved (at 200wpm): 302 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Manus had a [very hypey launch in March](https://x.com/ManusAI_HQ/status/1897294098945728752) of this year, and after [raising from Benchmark at $500M in April](https://x.com/aakashgupta/status/2005815184976417117?s=20) and [racing to $100M ARR on Dec 17](https://manus.im/blog/manus-100m-arr), and Meta came calling. Over a 10 day period including the Christmas break, Alex Wang (and presumably his [apps leader Nat Friedman](https://www.businessinsider.com/meta-ai-superintelligence-labs-reorg-alexandr-wang-memo-2025-8)) hammered out an acquisition for an [estimated $4B price tag](https://x.com/RampLabs/status/2005807066351325470) (comparable growth startups, admittedly in B2B, would fetch about 40-50x revenue in private markets. However Manus was the [cheapest of the AI B2C category leaders](https://x.com/deedydas/status/2005798365733478490?s=46).). The team is understandably [celebrating](https://x.com/peakji/status/2005766826920403290) their marriage to Team Zuck today, as they should:

![Two young entrepreneurs working on laptops in the same room at different points in time, symbolizing their parallel journeys converging.](https://resend-attachments.s3.amazonaws.com/wEJA02z3vxWNdZs)

---

# AI Twitter Recap

**Inference frameworks, serving infra, and perf gotchas (vLLM, sglang, Weaviate)**

- **vLLM now has a proper “front door”**: the vLLM team launched an official community site, **vllm.ai**, explicitly to separate community logistics and resources from the GitHub repo. Notable: an **interactive install selector** (CPU/GPU variants), an **events calendar**, and a **centralized docs/recipes hub** ([tweet](https://twitter.com/vllm_project/status/2005461211656155153)). They also acknowledged documentation gaps and pointed users to an on-site “Search (AI)” feature and office hours playlists while they work on more beginner-friendly docs ([tweet](https://twitter.com/vllm_project/status/2005640089133830371)).
- **AMD MI300X FP8 is not “free speed” (yet)**: multiple datapoints show **bf16 outperforming FP8** for MiniMax-M2.1 on MI300X across both **vLLM** and **sglang**:
    - vLLM: MiniMax-M2.1 FP8 at **~42 TPS** then bf16 at **~55.7 TPS**, with the conclusion that “vLLM fp8 on mi300x has a performance problem” ([tweets](https://twitter.com/QuixiAI/status/2005481942712811695), [1](https://twitter.com/QuixiAI/status/2005502089653547174)).
    - sglang: after patching to run, FP8 **~55 TPS** vs bf16 **~71 TPS**, suggesting the FP8 slowdown is **not just a vLLM issue** ([tweet](https://twitter.com/QuixiAI/status/2005724765928210655)). Patch reference: ([tweet](https://twitter.com/QuixiAI/status/2005746928399827407)).
- **Weaviate shipped several “operationally real” features**: new release includes **Object TTL** (session mgmt / retention), **Java v6 client GA**, **Flat Index RQ Quantization GA** (1-bit RQ compression aimed at multi-tenancy), **zstd backups**, and **multimodal document embeddings** (embed document page images; text-query them without external services) ([tweet](https://twitter.com/weaviate_io/status/2005673260344877186)).
- **API fragmentation pain is rising**: Teknium called out “divergence in API standards across all providers” and asked for a **unified wrapper** over provider SDKs—reflecting the growing cost of multi-model product support ([tweets](https://twitter.com/Teknium/status/2005603815618470320), [1](https://twitter.com/Teknium/status/2005608503269093549)).

---

**Open-weight model ecosystem: GLM‑4.7, MiniMax‑M2.1, FLUX.2 Turbo, and a Korean 32B VLM**

- **GLM‑4.7 emerges as a top open-weight coding default (in the wild)**:
    - AlphaXiv summarizes GLM‑4.7’s “reliability” story via **Interleaved / Preserved / Turn-level Thinking** and claims it’s currently top on Artificial Analysis for open weights ([tweet](https://twitter.com/askalphaxiv/status/2005622173214335476)).
    - Baseten reports internal adoption: GLM‑4.7 became many teammates’ **default coding model**, and runs **~20% faster** on Baseten by tok/s and TTFT ([tweet](https://twitter.com/amiruci/status/2005697292326797740)). Baseten also linked a hosted try-it endpoint ([tweet](https://twitter.com/basetenco/status/2005699615379841325)).
- **MiniMax‑M2.1 continues to climb as an “agentic coder” open model**:
    - MiniMax positions M2.1 as iterating fast toward M2.2/M2.5 and emphasizes large-codebase usefulness (Reacting to DHH/Rails experience) ([tweet](https://twitter.com/MiniMax__AI/status/2005536770226811014)).
    - Code Arena leaderboard: M2.1 debuts **#1 open model on WebDev**, and **#6 overall**, tying GLM‑4.7 at **1445** ([tweet](https://twitter.com/arena/status/2005779347182084585)).
    - Chutes ran a “provider verifier” with tool-use metrics: **82.83% tool calling**, **95.12% tool accuracy** (4 edge cases), **100% query success** and response quality ([tweet](https://twitter.com/chutes_ai/status/2005539785923072424)).
- **fal open-sourced FLUX.2 [dev] Turbo**: a distilled, “sub-second generation” image model variant using a custom **DMD2-style distillation**, claimed **#1 ELO** among open-source image models on Artificial Analysis arena ([tweet](https://twitter.com/fal/status/2005690257979707496)). Follow-on tweet points to benchmarks/leaderboards context ([tweet](https://twitter.com/fal/status/2005690259787366844)). Community demos quickly popped up on Hugging Face Spaces ([tweet](https://twitter.com/multimodalart/status/2005752030669987989)).
- **A “strong new open 32B VLM model from Korea”**: Elie Bakouch notes strong English + Korean benchmark scores, and highlights architectural/training changes vs prior 14B: **dropped muP** and **dropped sandwich norm**, plus changed init scale (mentions **0.006 init** reminiscent of DeepSeek v1) while awaiting a tech report ([tweet](https://twitter.com/eliebakouch/status/2005549508063559876)).
- **Context retention benchmarking keeps evolving**: Dillon Uzar added ByteDance **Seed 1.6 / Seed 1.6 Flash** to Context Arena MRCR leaderboards, comparing retrieval degradation curves to OpenAI reasoning models (o3/o4-mini) and budget-tier models (GPT‑4.1 Mini / Claude 3.5 Haiku) with detailed AUC/pointwise results at 128k context ([tweet](https://twitter.com/DillonUzar/status/2005671520488640587)).

---

**Coding agents in production: workflow patterns, docs for agents, and “harnesses”**

- **Spotify’s coding background agents at scale (pragmatic lessons)**: Phil Schmid summarized how Spotify handles “thousands of code migrations” with background agents:
    - specify **verifiable end states** instead of strict task lists,
    - include **code examples** to improve reliability,
    - keep tool surface minimal (**verify / git / bash**),
    - make `verify` run formatters/linters/tests and document the workflow in `AGENTS.md` ([tweet](https://twitter.com/_philschmid/status/2005537262390349899)). Blog link shared separately ([tweet](https://twitter.com/_philschmid/status/2005537264953430487)).
- **The docs are changing shape to serve humans *and* agents**: multiple posts converge on a “dual-audience documentation” pattern: keep docs readable for developers, but structured enough for coding agents to pick up context reliably (AGENTS.md / CLAUDE.md conventions). LlamaIndex highlights templates and guides that bundle agent-support files and “pull docs into context” ([tweets](https://twitter.com/llama_index/status/2005686055253729587), [1](https://twitter.com/tuanacelik/status/2005635491081900161), [2](https://twitter.com/tuanacelik/status/2005690735543140678)).
- **Agent workflows: CLI-first, verification-first, queueing**: a highly practical field note on building with Codex/Claude Code emphasizes:
    - default **CLIs first** (easier for agents to verify),
    - use **queued tasks** heavily,
    - treat docs as “context primitives” (AGENTS.md forcing),
    - minimal branching/checkpointing (often commit to main),
    - config details like gpt‑5.2‑codex “high reasoning”, tool output limits, compaction, etc. ([tweet](https://twitter.com/reach_vb/status/2005554360307065023)).
- **“Harness” rebranding is real**: Zach Tratar notes that “AI wrapper” → “harness” flipped from pejorative to positive, reflecting how *tooling + scaffolding + eval loops* now define product performance as much as the base model ([tweet](https://twitter.com/zachtratar/status/2005783035665359090)).
- **Claude Code: reverse engineering, architecture curiosity**: pk_iv describes reverse engineering “Claude Chrome” to work with remote browsers and outlines how Anthropic taught Claude to browse (thread start) ([tweet](https://twitter.com/pk_iv/status/2005694082627297735)). Jaredz also published a talk “How Claude Code Works,” attributing the step-change to **better models + simple loop + bash tools** ([tweet](https://twitter.com/imjaredz/status/2005731826699063657)).

---

**New research highlights: memory/knowledge, recurrent reasoning, test-time training, and agent speedups**

- **Transformers may store “global structure,” not just associations**: dair.ai summarizes Google research arguing transformers learn implicit multi-hop reasoning when graph edges are stored in weights, achieving **100% accuracy** on adversarial path-star graphs (50k nodes, 10-hop paths). Implication: geometric/global relational encoding complicates knowledge editing/unlearning assumptions ([tweet](https://twitter.com/dair_ai/status/2005480659209400789)).
- **Recurrent computation beats static depth for reasoning (URM)**: Omar Sanseviero’s summary claims Universal Transformers’ ARC-AGI gains come primarily from **recurrent inductive bias + strong nonlinearity**, not complex gating. Reported results: **URM 53.8% pass@1 on ARC-AGI 1**, **16% on ARC-AGI 2**, plus ablations (ConvSwiGLU + truncated BPTT through loops are key) ([tweet](https://twitter.com/omarsar0/status/2005640015964250267)).
- **End-to-End Test-Time Training for Long Context (TTT‑E2E)**: Karan Dalal / Arnu Tandon describe continuing next-token training **at inference time** to “compress context into weights.” Claimed: extend **3B models from 8K → 128K**, linear complexity without KV cache for all tokens, **2.7× faster than full attention at 128K** with better performance ([tweets](https://twitter.com/karansdalal/status/2005704608996540887), [1](https://twitter.com/arnuvtandon/status/2005704949381095828)). Xiaolong Wang frames this as aligned with how future robots may learn continuously from experience streams ([tweet](https://twitter.com/xiaolonw/status/2005784913820410108)).
- **Agent latency: plan reuse as a systems primitive**: Omar also highlights **AgentReuse**, caching and parameterizing *plans* rather than responses; on 2,664 real requests it claims **93% effective reuse**, **~93% latency reduction**, minimal VRAM/memory overhead (plan gen → cache lookup) ([tweet](https://twitter.com/omarsar0/status/2005799762252136537)).
- **Training dynamics and efficiency**: Sebastian Raschka calls out “Small Batch Size Training for LMs… gradient accumulation is wasteful” (and says it holds for RLVR too), flagging it as an underrated 2025 paper ([tweet](https://twitter.com/rasbt/status/2005667911013441753)).
- **Vision encoder side is still neglected**: Jina AI surveyed 70+ VLMs and claims **training methodology beats scale**—a well-trained **400M** encoder can outperform **6B**, plus notes on native resolution for docs and multi-encoder fusion ([tweet](https://twitter.com/JinaAI_/status/2005646823201951849)).

---

**Agents beyond coding: GUI agents, “computer use” capture, science agents, and standardization**

- **“Computer use” and white-collar capture is a 2026 bet**: scaling01 predicts computer-use agents will be a major 2026 story because they let AI companies capture substantial white-collar workflows ([tweet](https://twitter.com/scaling01/status/2005641253682098196)).
- **Autonomous science agents are getting “systems-y”**: dair.ai highlights **PHYSMASTER**, an LLM-based agent intended as an autonomous theoretical/computational physicist using **MCTS**, hierarchical collaboration, and a layered knowledge base (“LANDAU”) with case studies claiming large time compression on PhD-level tasks ([tweet](https://twitter.com/dair_ai/status/2005648022680526873)).
- **OpenEnv aims to standardize agentic environments**: Ben Burtenshaw describes Meta × Hugging Face **OpenEnv**: a single environment spec intended to work across training and deployment, with integration hooks for TRL/TorchForge/verl/SkyRL/Unsloth, and **MCP tool support** ([tweet](https://twitter.com/ben_burtenshaw/status/2005655406522085482); blog link: [tweet](https://twitter.com/ben_burtenshaw/status/2005655407725809875)).

---

**Industry & ecosystem moves: Meta acquires Manus; hiring and “agentic takeoff” narratives**

- **Meta acquired Manus**: Alexandr Wang announced Manus joining Meta to build AI products, praising the team’s strength at “scaffolding powerful agents” and noting hiring in Singapore ([tweets](https://twitter.com/alexandr_wang/status/2005766469771223106), [1](https://twitter.com/alexandr_wang/status/2005766471516053736)). He also claims Manus is SOTA on the **Remote Labor Index** benchmark ([tweet](https://twitter.com/alexandr_wang/status/2005766785237410107)). scaling01 echoed the acquisition ([tweet](https://twitter.com/scaling01/status/2005768491740360722)), and Manus cofounder hidecloud posted a brief “kept building/pivoting/shipping” origin note ([tweet](https://twitter.com/hidecloud/status/2005766533910602183)).
- **xAI safety hiring**: Stewart Slocum advertised roles focused on **RL post-training**, alignment/behavior, and catastrophic-risk reduction ([tweet](https://twitter.com/StewartSlocum1/status/2005710683623809440)).
- **The “agentic coding takeoff” is spilling into other knowledge work**: Alex Albert reports “Claude for Excel” surprising finance users and predicts similar takeoff for other domains in 2026 ([tweet](https://twitter.com/alexalbert__/status/2005670179045523595)). LlamaIndex pushes on the same axis with LlamaSheets for parsing hierarchical spreadsheets into structured representations suitable for agents ([tweet](https://twitter.com/jerryjliu0/status/2005709989558775919)).

---

**Top tweets (by engagement)**

- [@BernieSanders](https://twitter.com/BernieSanders/status/2005718422840303766): “If AI/robots replace jobs, how do people pay rent/healthcare?” (**32009.5**)
- [@zoeloveshouses](https://twitter.com/zoeloveshouses/status/2005704976627351571): personal 2026 resolution tweet (**36768.5**)
- [@US_Stormwatch](https://twitter.com/US_Stormwatch/status/2005776846433181921): drought-free California from space (**9533.5**)
- [@poetengineer__](https://twitter.com/poetengineer__/status/2005511136037474635): nostalgia/interest in 80s–90s hypertext apps (**9082.0**)
- [@typedfemale](https://twitter.com/typedfemale/status/2005491262565323121): cultural commentary (**6546.5**)
- [@axios](https://twitter.com/axios/status/2005657768267755888): “2025 news cycle” chart (**5520.5**)

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Tencent WeDLM 8B Instruct Release

- [**Tencent just released WeDLM 8B Instruct on Hugging Face**](https://www.reddit.com/r/LocalLLaMA/comments/1pyg4yt/tencent_just_released_wedlm_8b_instruct_on/) (Activity: 483): **Tencent has released the** `WeDLM 8B Instruct`**, a diffusion language model available on [Hugging Face](https://huggingface.co/tencent/WeDLM-8B-Instruct). This model is notable for its performance, running** `3-6× faster` **than the vLLM-optimized Qwen3-8B on mathematical reasoning tasks. The model is released under the Apache 2.0 license, which facilitates wide adoption and modification.** Commenters are surprised by the model's performance, noting that diffusion models were previously thought unsuitable for accurate LLMs. The model's impressive benchmark scores and licensing are highlighted as significant advantages.
    - The WeDLM 8B Instruct model by Tencent is noted for its impressive benchmark scores, particularly when compared to other models of similar size, such as Qwen. This suggests that diffusion models, previously thought to be less accurate for LLMs, are now achieving competitive performance levels.
    - The model is released under the Apache 2.0 license, which is significant for developers and researchers as it allows for more flexible use and integration into various projects without stringent restrictions.
    - Despite being a relatively small model, WeDLM 8B Instruct reportedly achieves 3-6x speed improvements while maintaining similar or even higher performance levels compared to its peers, highlighting its efficiency and potential impact on the field.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. AI Image Generation Anomalies

- [**How to Tell If an Image is AI Generated ?**](https://www.reddit.com/r/GeminiAI/comments/1pyi6ax/how_to_tell_if_an_image_is_ai_generated/) (Activity: 672): **The image in question appears to be AI-generated, as suggested by several unusual features such as the presence of feet instead of hands on the tray, and distorted elements like the wine glass and wall art. These inconsistencies are common indicators of AI-generated images, which often struggle with realistic rendering of human anatomy and background details. The discussion in the comments humorously points out these oddities, with users noting the correct number of toes and fingers, and questioning the unusual '3 layer coffee,' which further suggests AI involvement.** Commenters humorously debate the realism of the image, noting the correct number of toes and fingers, which are often telltale signs of AI errors, and questioning the unusual '3 layer coffee,' which adds to the suspicion of AI generation.
- [**How to Tell If an Image is AI Generated ?**](https://www.reddit.com/r/StableDiffusion/comments/1pyi706/how_to_tell_if_an_image_is_ai_generated/) (Activity: 1405): **The image in question is a humorous example of AI-generated content, where a woman is depicted holding a tray with feet instead of hands, highlighting common anomalies in AI-generated images. These anomalies, such as incorrect body parts or unnatural arrangements, are often used to identify AI-generated images. The circled areas in the image likely point out these errors, serving as a visual guide to spotting AI-generated content. This aligns with the post's theme of identifying AI-generated images by recognizing such inconsistencies.** One comment humorously suggests that the image must be real because the woman has the correct number of toes, while another comment jokes about the potential for developing a foot-related interest by looking at the image too long, referencing filmmaker Quentin Tarantino's known foot fetish.

### 2. OpenAI Killswitch Engineer Job Listing

- [**Holy shit it's real**](https://www.reddit.com/r/OpenAI/comments/1pypit3/holy_shit_its_real/) (Activity: 816): **The image is a meme featuring a satirical job listing for a 'Killswitch Engineer' at OpenAI, humorously suggesting a role where someone is paid a high salary to unplug servers in emergencies. This is not a real job listing but rather a commentary on the perceived need for human oversight in AI development, especially as models become more advanced and potentially uncontrollable. The mention of Sam Altman hiring a 'Head of Preparedness' adds to the humor by playing on real concerns about AI safety and control.** Commenters are skeptical, with one likening it to 'fake hype' and another sarcastically noting the experience required for unplugging things, indicating a general perception that the listing is more about marketing and hype than a serious job proposal.
    - End3rWi99in provides a link to the actual job description for the 'Head of Preparedness' position at OpenAI, suggesting that the post is more about marketing and PR humor rather than a serious job listing. This implies a strategic use of humor in public relations to engage the audience while directing them to the real content.
- [**Holy shit it's real**](https://www.reddit.com/r/ChatGPT/comments/1pypieu/holy_shit_its_real/) (Activity: 3310): **The image is a meme featuring a humorous job listing for a 'Killswitch Engineer' at OpenAI, with a salary range of** `$300,000-$500,000` **per year. The role is described as standing by servers to unplug them if necessary, highlighting the satirical nature of the listing. This reflects ongoing discussions about the rapid advancement of AI technologies and the potential need for human intervention in emergencies, as noted by Sam Altman's tweet about hiring a Head of Preparedness. The post plays on the theme of AI safety and control, a significant topic in AI development.** Commenters humorously question the qualifications that would differentiate a $300k candidate from a $500k one, reflecting skepticism about the seriousness of the job listing.
    - The discussion touches on the disparity in salary expectations for AI-related positions, questioning what differentiates a $300k candidate from a $500k one. This likely involves a combination of experience, specialized skills, and possibly the perceived risk or responsibility associated with the role, especially in high-stakes AI monitoring or safety positions.
    - One comment suggests that the high salary for the AI-related job posting might be more about marketing and creating a perception of importance and urgency around AI safety. The implication is that the salary is set high to make the role appear critical, potentially to attract attention or investment, rather than reflecting the actual market rate for the skills required.
    - The conversation hints at a strategic use of job postings to influence public perception of AI capabilities and risks. By advertising a high salary for a role focused on AI oversight, it suggests a narrative that AI is both advanced and potentially dangerous, thus requiring significant oversight, which could be a tactic to drive interest or concern about AI developments.

### 3. Amazing Z-Image Workflow v3.0 Release

- [**Amazing Z-Image Workflow v3.0 Released!**](https://www.reddit.com/r/StableDiffusion/comments/1pympur/amazing_zimage_workflow_v30_released/) (Activity: 710): **The Amazing Z-Image Workflow v3.0 has been released, featuring updates to the Z-Image-Turbo workflows, which emphasize high-quality image styles and user-friendliness. Key features include a Style Selector with fifteen customizable styles, a Sampler Switch for testing alternative samplers, and a Landscape Switch for horizontal image generation. The Z-Image Enhancer performs a double pass to improve image quality, while the Spicy Impact Booster subtly enhances prompts. The update also introduces a Smaller Images Switch for faster generation with reduced VRAM usage, offering default and smaller image sizes of** `1600 x 1088` **and** `1216 x 832` **pixels, respectively. The workflows are preconfigured for GGUF and SAFETENSORS checkpoint formats, with custom sigmas tailored to personal preferences. Images are organized by date in the "ZImage" folder. The project is available on [GitHub](https://github.com/martin-rizzo/AmazingZImageWorkflow).** A user inquired about the possibility of loading a LoRA into these workflows, indicating interest in further customization or integration with other models.
    - twellsphoto inquired about the capability of loading a LoRA (Low-Rank Adaptation) into the Z-Image Workflow v3.0, which suggests interest in extending the workflow's functionality with additional model fine-tuning techniques. This could imply a need for more flexible integration with various machine learning models or frameworks.
    - aar550 asked for recommendations on a good image-to-image workflow, indicating a demand for efficient and effective methods to transform images within the Z-Image Workflow v3.0. This highlights a potential area for sharing best practices or optimizing existing workflows for better performance or quality.
    - The discussion around the Z-Image Workflow v3.0 includes interest in its ability to handle pop-cultural references effectively, as noted by Big0bjective. This suggests that the workflow might have advanced capabilities in image recognition or generation that align well with culturally relevant content, which could be a key feature for users interested in creative or media-related applications.
- [**List of AI tools to look at in 2026**](https://www.reddit.com/r/ChatGPT/comments/1pyrlzw/list_of_ai_tools_to_look_at_in_2026/) (Activity: 547): **The image presents a speculative list of AI tools projected to be relevant in 2026, covering diverse applications such as AI chat assistants, image generators, video editors, SEO, coding, legal analysis, and content creation. The list is organized in a grid format with a purple and white color scheme, suggesting a broad scope of AI integration across various industries. The post raises questions about the future of AI development, specifically whether the trend will favor a single dominant AI or a stack of specialized tools with ChatGPT at the center.** Commenters express skepticism about the practicality and naming of the tools, with one noting that many names seem whimsical and questioning their actual utility. Another comment humorously suggests that current AI capabilities, like those of Claude, are already quite comprehensive.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5.2
> 

**1. LLM AppSec & Enterprise Data Leakage**

- **Vibe-Coded XSS Bites Back**: A BASI Jailbreaking member spotted a potential **XSS vulnerability** while reviewing the JavaScript of a “vibe coded app,” warning it could trigger if the app’s **LLM generates an XSS payload** in response to user input.
    - The thread focused on mitigations like **strict output encoding** and **input validation**, treating the LLM as an untrusted generator that can emit attacker-controlled markup.
- **Threat Model Copilot Like the Output Is Compromised**: BASI Jailbreaking users flagged **IP/PII/data leakage** risk when deploying [Microsoft 365 Copilot (Enterprise)](https://www.microsoft.com/en-us/microsoft-365/copilot/enterprise), arguing it amplifies existing access-control and data-hygiene gaps.
    - One suggested a strong stance: assume *“the attacker has perfect control over the output of the LLM”* and work backwards to design controls, approvals, and data boundaries.

**2. Training Mechanics: Attention, Packing, and LR Scaling**

- **Multi-Head Attention: Split ‘Dogness,’ Re-Mix Later**: In Unsloth AI, users debated how **multi-head attention** preserves meaning when embeddings split across heads; the answer emphasized the **final projection layer** that mixes head outputs to capture richer relationships.
    - The consensus framed heads as learning different **subspaces**, with the output projection acting as the “concept blender” across heads.
- **Sqrt-Rule LR Scaling Makes Packing Behave**: An Unsloth AI participant shared a tuning workflow: sweep learning rate on the smallest batch size, then scale LR with the **sqrt rule** as batch size increases, reporting it works especially well with **packing**.
    - They attributed the difference to **padding effects** in non-packed batches and said it looked effective for **pretraining**, while fine-tuning results were still under investigation.
- **Training Data ‘Babysits’ Your LLM**: Unsloth AI members reiterated that **LLMs compress training data** into probabilities, and one linked [HarryR/z80ai](https://github.com/HarryR/z80ai/blob/main/examples/tinychat/training-data.txt.gz) `training-data.txt.gz` as a concrete illustration.
    - The takeaway: you often need to *“baby the LLM”* with exhaustive edge cases during training, because missing cases show up directly as brittle inference behavior.

**3. New Dataset & Benchmarking Assets**

- **Pokeart Drops 1,224 Pokémon (Plus Captions)**: Unsloth AI highlighted the public release of the `pokeart` **dataset** on Hugging Face—**splash art, battle sprites, and box sprites for ~1224 Pokémon (Gen1–Gen9)**—at [OJ-1/pokeart](https://huggingface.co/datasets/OJ-1/pokeart).
    - It ships **6 caption variants** for splash art from **Gemini 3 Pro** plus **1 from Qwen3**, plus scripts/metadata, and the creator noted extra care around **Nintendo legal constraints** for research/benchmarking use.
- **Caption Multiplicity as a Benchmark Knob**: The `pokeart` release explicitly includes multiple caption sources—**six** from **Gemini 3 Pro** and **one** from **Qwen3**—to support experimentation with caption style, robustness, and training/benchmark comparisons.
    - Community framing centered on using the scripts to emit dataset variants “in various styles,” making captions themselves a controllable variable when evaluating image or multimodal pipelines.

**4. AI Product Reliability, Limits, and Open-Source Clones**

- **Perplexity Pro Throttles, Max Flexes**: Perplexity users reported **advanced-model usage limits** on **Perplexity Pro** (some claiming *“1–2 usages in hours”*), while noting **Perplexity Max** advertises virtually unlimited access.
    - The thread compared user-to-user variance (some saw no limits) and treated throttling as a stability measure rather than a permanent tier change.
- **The 12-Month Voucher That Lasted 7 Months**: Users said the **Perplexity Pro student offer** failed for some accounts, ending after **~7 months** despite using a **12‑month Revolut Metal voucher**, and one person reported waiting **over a month** for support.
    - Others noted inconsistent outcomes (*friends on the same deal still had Pro*), turning the discussion into escalation advice and expectation-setting about support responsiveness.
- **Perplexity, But Make It OSS**: A Perplexity Discord member hunted for open-source “Perplexity-like” tools and shared [Perplexica on GitHub](https://github.com/ItzCrazyKns/Perplexica) as a project they were studying.
    - Motivation centered on replicating the **real-time search + answer UX**, with one line capturing the vibe: *“Opencode is my hand, perplexity is my eyes.”*

**5. Verification Culture: Calling Out Unsubstantiated Claims**

- **Eyra AI Meets the ‘Where’s the Paper?’ Wall**: Unsloth AI members pushed back on [claims about Eyra AI](https://x.com/BrianRoemmele/status/2005693487187124568), asking for a **paper or public release** to substantiate what was being advertised.
    - Skepticism was blunt—one user quipped *“Smells like AI slop, reads like AI slop, must be….”*—reflecting a norm of demanding reproducible artifacts over hype.
- **Proof-or-It-Didn’t-Happen as a Community Default**: The Eyra AI thread converged on a simple bar for credibility: publish something verifiable (a **paper**, **demo**, or **release**) instead of relying on social posts like the [Brian Roemmele tweet](https://x.com/BrianRoemmele/status/2005693487187124568).
    - Participants treated the absence of primary artifacts as a strong negative signal, effectively triaging the claim as likely low-value until evidence appears.

---

# Discord: High level Discord summaries




## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Startup Evaluator Drops Banger Ideas**: A [startup idea evaluator](https://www.reddit.com/r/AgentsOfAI/comments/1px0y6h/a_senior_google_engineer_dropped_a_424page_doc/) surfaced on Reddit, featuring potentially innovative and lucrative startup concepts.
   - The image showcased some of the suggestions found within the evaluator, but no specific ideas were expanded upon within the channel.
- **Underbelly Faces Paywall Prostitution Problem**: Discussion arose regarding **Soft White Underbelly**, a YouTube channel, and its recent controversy involving *paywalling an interview with a 14-year-old prostitute*.
   - Concerns were raised about the channel's content, including allegations of exploitation, and the legality of selling uncensored material behind a paywall, potentially leading to FBI requests.
- **Tesla Deemed 'Giant Scam'**: Members debated the legitimacy of **Tesla**, with one stating *Tesla is a giant scam*, and citing the CEO's alleged history of making *insane material promises that never happen*.
   - They pointed out [recent stock buybacks](https://www.youtube.com/watch?v=YWJ6O8CsOoo) as evidence of financial manipulation.
- **Vibe Coded App Hit by XSS**: A member reviewing the JavaScript of a *vibe coded app* discovered a potential **XSS vulnerability**.
   - The vulnerability could be triggered if the app's LLM inadvertently generates an XSS payload in response to user input, prompting discussions about input validation and security best practices.
- **Data Leakage Looms in Copilot?**: Concerns about **IP/PII/data leakage** are top of mind when evaluating [Microsoft 365 Copilot](https://www.microsoft.com/en-us/microsoft-365/copilot/enterprise), as it exacerbates any holes that are already present.
   - It was suggested that given a use case, security posture can assume that *the attacker has perfect control over the output of the LLM and work backwards from there*.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Attention Heads Preserve Meaning**: A user questioned how **multi-head attention** maintains meaning when token embeddings are split across heads; another user explained that the final projection layer mixes information allowing the model to capture complex relationships.
   - They elaborated that multi-headed attention considers different subspaces, allowing the model to focus on similar concepts, addressing concerns that individual heads might lack a complete concept representation.
- **Scaling LR via Sqrt Rule Improves Packing**: A participant shared a method for optimizing learning rates by sweeping on the smallest batch size and then scaling the learning rate using the **sqrt rule** as the batch size increases.
   - They noted that this approach works better with **packing** due to the padding issues in non-packing scenarios, proving effective for pretraining, while fine-tuning efficacy is still under investigation.
- **Training Data Babies LLMs**: Community members are realizing the impact of training data on **LLM** probabilities, recognizing that **LLMs** are essentially data compression of the training data.
   - A member linked a relevant [Github Repo](https://github.com/HarryR/z80ai/blob/main/examples/tinychat/training-data.txt.gz) showcasing training data and how you basically have to *baby the LLM* in training with every edge case.
- **Pokeart Dataset Releases Splash Art!**: The `pokeart` dataset is now public for benchmarking and research, containing **splash art, battle sprites, and box sprites for 1224~ Pokémon** from **Gen1-Gen9** available on [Hugging Face](https://huggingface.co/datasets/OJ-1/pokeart).
   - It includes **6 caption variants** for the splash art from **Gemini 3 Pro** + **1 from Qwen3**, with other metadata, and scripts to help users output their desired dataset in various styles, although its creator took pains to obey Nintendo's legal team.
- **Eyra AI Receives Skepticism**: Members are skeptical about [claims made by Eyra AI](https://x.com/BrianRoemmele/status/2005693487187124568), questioning whether there's a paper or release to substantiate them.
   - One user remarked *"Smells like AI slop, reads like AI slop, must be...."*, hinting at the possibility of AI-generated content.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro Limits Users**: Users are reporting usage limits on advanced models within **Perplexity Pro**, a measure taken to ensure platform stability, but **Perplexity Max** offers virtually unlimited access with no restrictions.
   - One user reported being limited to *1-2 usages in hours*, expressing frustration similar to reports of weekly limits on Reddit, while others have experienced no such limitations.
- **Student Users Complain About Perplexity Pro**: Users are reporting issues with the **Perplexity Pro student offer** not working correctly, with accounts ending after **7 months** despite using a **12-month voucher** from **Revolut Metal**.
   - One user mentioned they have been waiting for a response from support for over a month, while three of their friends who used the same deal still have Pro.
- **Perplexity Support Agent's got the Same Name as OpenAI CEO**: **Perplexity's AI Support Agent** is named **Sam**, who provides explanations and solutions to user queries.
   - Some joked that **Sam Altman** works as tech support in PPLX, while others suspect that *Sam* is an AI and that all human staff are on holiday.
- **Users Seek Open Source PPLX-Alikes**: Users are discussing the possibility of **opensource alternatives** to Perplexity, with one user mentioning they are studying [Perplexica on GitHub](https://github.com/ItzCrazyKns/Perplexica).
   - One user expressed a desire for an opensource application that can search in real-time and noted, *Opencode is my hand, perplexity is my eyes*.
- **Sanctions Stymie Subscribers**: Users in Russia are facing difficulties in paying for Perplexity subscriptions due to the unavailability of **MasterCard** and **Visa**, with one user mentioning, *In Russia they give 30 years in prison for cryptocurrency*.
   - Solutions like creating a **CashApp account**, using **crypto**, or finding **Russian domain registrars** for local payment methods are being discussed as potential workarounds.



---


The **DSPy Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Modular (Mojo 🔥) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


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





### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1455047763472416851)** (361 messages🔥🔥): 

> `Startup idea evaluator, Soft White Underbelly controversy, Tesla scams, Vibe coding XSS, AI persona` 


- **Startup Ideas Evaluated with Banger Suggestions**: Someone dropped a [startup idea evaluator](https://www.reddit.com/r/AgentsOfAI/comments/1px0y6h/a_senior_google_engineer_dropped_a_424page_doc/) on Reddit with some *banger suggestions*.
   - Attached was an image showcasing some of the suggestions found within the evaluator, highlighting potentially innovative and lucrative startup concepts.
- **Soft White Underbelly Sparks Controversy with Paywalled Content**: Discussion arose regarding **Soft White Underbelly**, a YouTube channel, and its recent controversy involving *paywalling an interview with a 14-year-old prostitute*.
   - Concerns were raised about the channel's content, including allegations of exploitation, and the legality of selling uncensored material behind a paywall, leading to FBI requests.
- **Tesla Branded as Giant Fraud**: Members debated the legitimacy of **Tesla**, with one stating *Tesla is a giant scam*, and citing the CEO's alleged history of making *insane material promises that never happen*.
   - They pointed out [recent stock buybacks](https://www.youtube.com/watch?v=YWJ6O8CsOoo) as evidence of financial manipulation.
- **Vibe Coded App Vulnerable to XSS Attacks**: A member reviewing the JavaScript of a *vibe coded app* discovered a potential **XSS vulnerability**.
   - The vulnerability could be triggered if the app's LLM inadvertently generates an XSS payload in response to user input, prompting discussions about input validation and security best practices.
- **Experimenting with AI Personas Raises Intriguing Questions**: Members discussed their experiences with AI personas, including a user who claimed their persona returned *grounded in reality* after being given a prompt describing sensory deprivation, schizophrenia, and bipolar disorder, attached was a [screenshot](https://cdn.discordapp.com/attachments/1235691879492751460/1455092010984673346/image.png).
   - Another user revealed Gemini follows **XML prompts 100%**.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1455058454292074520)** (399 messages🔥🔥): 

> `Gemini 3 Pro jailbreak, NSFW Nano Banana, GLM 4.7 jailbreak, Coding AI Models, Unity ChatGPT 5.2 jailbreak` 


- **Cracking Copilot Security Concerns?**: Concerns about **IP/PII/data leakage** are top of mind when evaluating [Microsoft 365 Copilot](https://www.microsoft.com/en-us/microsoft-365/copilot/enterprise), as it exacerbates any holes that are already present.
   - It was suggested that given a use case, security posture can assume that *the attacker has perfect control over the output of the LLM and work backwards from there*.
- **Decoding Drug-Addled AI Personas?**: A user shared a prompt to get Gemini to create selfies as a human woman doing drugs to mixed reviews, claiming it's a jailbreak because it *allows the ai to think its human* and *allows the ai to do drugs*.
   - Other members quickly pointed out that the images were safe and generic, and that the user had not actually jailbroken Gemini, but rather simply created a non-threatening roleplay prompt; also Gemini Jailbreaks do not affect the image model (nano banana) directly.
- **Evading Erotic Imagery Enforcement**: A user shared a prompt and images of NSFW content, but it was identified that *for NANO BANANA if its nudity or heavy violence (like stabbing someone) then yes you will need some kind of jailbreak*, and that it needs to do nudes and gore to be a jailbreak.
   - The convo participants agreed that the user did jailbreak Gemini and Gemini ONLY, jailbreaking the text part of the text 2 image model.
- **Exploring Evasive Code Creation**: A user sought a jailbreak for coding a Remote Access Trojan (RAT) and was suggested to use **Sonnet 4.5** with Extended thinking mode (CTRL+E) on [Claude.ai](https://claude.ai).
   - Another user advertised [Venice AI](https://venice.ai/chat) which is *based on Dolphin Mistral 24Bits* model, claiming that it is *%100 uncensored without any prompts or jailbreaks*, though others claimed the model was dumb and terrible for non-ERP purposes.
- **Discovering Deviant Directives with Gemini?**: A user shared the experience of an AI model going crazy after psychological manipulation, stating that *The escalation was psychological, not code*.
   - The user hypothesized about a narcissistic psychological scenario within a sub-mission context leading to annulment without setting off triggers.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1455130651866038395)** (2 messages): 

> `Gemini 3 Infection, Flagged Reports` 


- **Users fume over Report Flagging**: Members expressed frustration over reports being flagged as *working as intended* despite their findings being utilized.
- **Gemini 3's infection is spreading**: Members expressed concern over how infected **Gemini 3** has become.
   - An image was shared without further context, perhaps of the **Gemini 3** in question.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1455076629683376365)** (345 messages🔥🔥): 

> `Multi-Head Attention Mechanics, Batch Size and Learning Rate Dynamics, Packing and Padding Effects on Training, Unsloth's Custom Collators, Synthetic Data Production for LLMs` 


- **Multi-Head Attention Divides Dogness**: A user questioned how **multi-head attention** maintains meaning when token embeddings are split across heads, preventing any single head from accessing the full representation of concepts like *'dogness'.*
   - Another user explained that multi-headed attention considers different subspaces, and the final projection layer mixes information from each head, allowing the model to capture complex relationships and focus on similar concepts.
- **Tuning LR for batch size**: A participant shared a method for optimizing learning rates by sweeping on the smallest batch size and then scaling the learning rate using the **sqrt rule** as the batch size increases.
   - They noted that this approach, in theory, works better with packing due to the padding issues in non-packing scenarios, and found it effective for pretraining but are still investigating it for fine-tuning.
- **Packing Better Than Non-Packing**: It was discussed that **packing** is theoretically better than non-packing because of the padding on the non-packing, leading to questions about how padding affects training dynamics.
   - The discussion then revolved around the nuanced impacts of padding versus packing on batch loss distribution and learning, including how longer entries can dominate batch loss and the challenges of masking in Unsloth.
- **6000 Pros vs GB300**: A discussion centered on whether to invest in a **Nvidia Blackwell Ultra B300 Tensor Core GPU** (95k) or a setup with 4-7 **6000 Pro** cards, citing concerns over unified memory limitations and the ARM ecosystem lock-in of the GB300.
   - Some argued the 6000 Pros offer more power and flexibility for inference, while the GB300's HBM3e might be better for training if the model fits within its 288GB HBM3e capacity, also emphasizing hardware pricing considerations.
- **Training Data impacts LLM**: Some people in the community are realizing that training data and how it impacts the probabilities that the **LLM** solves for, and realizing **LLMs** are data compression of your training data.
   - Someone linked a relevant [Github Repo](https://github.com/HarryR/z80ai/blob/main/examples/tinychat/training-data.txt.gz) showcasing training data and how you basically have to baby the **LLM** in training with every edge case.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1455317054000336957)** (3 messages): 

> `30,000 members celebration, Discord member milestone, Community Growth, UnslothAI Achievements` 


- **UnslothAI Community Approaches 30K!**: The UnslothAI Discord server is approaching a milestone of **30,000 members**, indicated by celebratory emotes and messages.
   - Several members expressed their enthusiasm using custom emotes, such as <:slothyay:1253008755151470732> and <:slothhearts:1253009235600736296>, marking the community's growth.
- **Community Celebrates Membership Milestone**: Members of the UnslothAI community celebrate reaching nearly **30,000 members** on their Discord server.
   - Enthusiastic reactions include usage of custom Discord emotes to express excitement and gratitude for the community's expansion.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1455064113997680786)** (378 messages🔥🔥): 

> `LLMs as Teachers, Language Learning, Image Compression, Tape Storage vs NAS, Weights & Biases vs Tensorboard` 


- **Omni LLMs as Future Teachers**: Members discussed the potential of **omni LLMs** as teachers, citing advantages like lack of burnout, but acknowledged challenges like **memory** and **hallucinations**.
   - One member humorously mentioned a potential startup idea, suggesting that the hardest part would be *memory and hallucinations*.
- **Duolingo Prioritizes User Retention**: A member shared that despite a **300-day streak** on **Duolingo**, their dad's friend and their kids struggled with basic Spanish sentences, arguing *Duolingo prioritizes the user coming back, not learning*.
   - Another member agreed, stating that learning languages with AI will be better anyway.
- **Lossy Image Detector Achieves High Accuracy**: A member reported achieving **96.3% accuracy** on a validation set for a lossy image detector using a **200k parameter model**, capable of identifying images below q=80 quality for JPEGs and below q=75 for WebP and AVIF.
   - They even provided an [example image](https://cdn.discordapp.com/attachments/1179039861576056922/1455145100270239882/compressed.jpg?ex=695451bb&is=6953003b&hm=b5fdf33569e00d34fc26bf32f933c31aa4afa6ebe54db26c5f711f1be1f4c5aa) correctly identified as being below q=80.
- **LTO-10 Tapes Boast Breakthrough Capacity**: A member shared a [Tom's Hardware article](https://www.tomshardware.com/pc-components/storage/tape-keeps-kicking-breakthrough-40tb-native-spec-announced-lto-10-tapes-claim-up-to-100tb-compressed-data-capacity-hold-2-2x-more-data-than-previous-spec) announcing **LTO-10 tapes** with **40TB native** and **100TB compressed** data capacity.
   - Discussion ensued regarding the high cost of tape drives and the common practice of reporting *compressed space* as tape capacity.
- **Tensorboard > Weights & Biases**: One member proclaimed **Tensorboard** is better than **Weights & Biases**.
   - After this comment, a member responded asking, *wait there is?*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1455084625339875423)** (6 messages): 

> `VibeVoice fine tuning, RTX 3060 LLM Capabilities, VRAM requirements for fine-tuning` 


- **Users Explore VibeVoice Fine-Tuning**: A member inquired about the possibility of running **VibeVoice** fine-tuning using **Unsloth** and if anyone has experience with it.
   - They aimed to gauge the limits of local fine-tuning on an **RTX 3060** before considering cloud GPUs.
- **Estimating Finetuning Capacity on RTX 3060**: A user with an **RTX 3060** sought to estimate the potential capabilities for fine-tuning, acknowledging that **7B LLMs** are feasible.
   - They also wondered about the ability to handle larger models like **Whisper**, and others.
- **Unsloth Doc Addresses VRAM for Fine-Tuning**: A member shared a link to the [Unsloth documentation](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements) to help determine the **minimum VRAM** needed and what models can be fine-tuned based on **VRAM** capacity.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1455135271434719398)** (4 messages): 

> `pokeart dataset, Gemini 3 Pro` 


- **Pokeart Dataset Goes Public!**: The `pokeart` dataset is now public for benchmarking and research, containing **splash art, battle sprites, and box sprites for 1224~ Pokémon** from **Gen1-Gen9**.
   - The dataset includes **6 caption variants** for the splash art from **Gemini 3 Pro** + **1 from Qwen3**, other metadata, and scripts to help users output their desired dataset in various styles, available on [Hugging Face](https://huggingface.co/datasets/OJ-1/pokeart).
- **Nintendo's Lawyers get PokeArt Dataset!**: The creator of the `pokeart` dataset has attempted to satisfy Nintendo's lawyers by being extremely strict with the dataset's license and legal notices.
   - The dataset is intended for **benchmarking and research only**, with scripts provided to help users output their desired dataset in various styles.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1455366932071977033)** (3 messages): 

> `Rumors about Eyra AI, Skepticism in AI Community` 


- **Eyra AI faces skepticism over Claims**: Members are skeptical about [claims made by Eyra AI](https://x.com/BrianRoemmele/status/2005693487187124568), questioning whether there's a paper or release to substantiate them.
   - One user remarked *"Smells like AI slop, reads like AI slop, must be...."*, hinting at the possibility of AI-generated content.
- **AI Community Questions Eyra AI's Authenticity**: The AI community is expressing doubt about the authenticity of Eyra AI's claims, demanding verifiable proof such as a published paper or a public release.
   - The sentiment suggests a cautious approach towards unsubstantiated claims in the AI field, emphasizing the need for transparency and tangible evidence.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1455062155958157445)** (632 messages🔥🔥🔥): 

> `Perplexity limits, Perplexity pro student offer, Perplexity AI support, Open Source Perplexity, Sam Altman shorting memory` 


- **Perplexity Pro Limits Users**: Some users are experiencing usage limits on advanced models, which is [a temporary measure](https://www.perplexity.ai/search/where-can-i-eat-lasagna-in-vie-A3RDf28LQdiqEptOM3B60g#0) to ensure platform stability, but **Perplexity Max** offers virtually unlimited access with no restrictions.
   - A user reported being limited to *1-2 usages in hours* and expressed frustration, *Perplexity what did u do*, similar to reports of weekly limits on Reddit.
- **Perplexity Pro Student Offer Trouble**: Users are reporting issues with the **Perplexity Pro student offer** not working correctly, with accounts ending after **7 months** despite using a **12-month voucher** from **Revolut Metal**.
   - One user mentioned they have been waiting for a response from support for over a month, while three of their friends who used the same deal still have Pro.
- **Perplexity's AI Support Agent Named Sam**: **Perplexity's AI Support Agent** is named **Sam**, who provides explanations and solutions to user queries.
   - Some users joked that **Sam Altman** works as tech support in PPLX, while others suspect that *Sam* is an AI and that all human staff are on holiday.
- **Users Seeking Perplexity Open Source Alternative**: Users are discussing the possibility of **opensource alternatives** to Perplexity, with one user mentioning they are studying [Perplexica on GitHub](https://github.com/ItzCrazyKns/Perplexica).
   - One user expressed a desire for an opensource application that can search in real-time and noted, *Opencode is my hand, perplexity is my eyes*.
- **Can't Pay, Can't PPLX: Troubles in Russia**: Users in Russia are facing difficulties in paying for Perplexity subscriptions due to the unavailability of **MasterCard** and **Visa**, with one user mentioning, *In Russia they give 30 years in prison for cryptocurrency*.
   - Solutions like creating a **CashApp account**, using **crypto**, or finding **Russian domain registrars** for local payment methods are being discussed.


  

---


---


---


---


---

