---
id: MjAyNS0x
title: not much happened today
date: '2026-06-24T05:44:39.731046Z'
description: >-
  **OpenAI** announced **Jalapeño**, its first custom AI chip for LLM inference,
  built with **Broadcom**, aiming to control more of the AI stack and improve
  compute economics with a fast 9-month design cycle. Community analysis
  suggests Jalapeño features **216GB HBM3E**, **~7.1–7.4 TB/s bandwidth**, and
  **~10 PFLOPS FP4** performance, signaling hyperscaler-style inference silicon
  as a new standard. Meanwhile, **Qualcomm** is acquiring **Modular**, with
  **Mojo** open-sourcing on track, indicating rising competition in vertically
  integrated inference stacks beyond **NVIDIA/CUDA**. On infrastructure,
  **NVIDIA**'s **NeMo AutoModel** boosts training throughput for MoE models by
  3.4–3.7x, and startups like **SkyPilot** and **Modal** advance unified and
  open-source inference solutions. Custom training of **DFLASH** models yields
  30–50% decode gains. In UX, **Anthropic**'s Slack-native **Claude** agent
  shifts agent interaction from tools to coworkers, raising new security and
  cost concerns around identity, permissions, and lock-in, with debates on
  capability-based security and attribution. **Hugging Face** responded with its
  self-hosted Slack coding agent **Moon Bot**.
companies:
  - openai
  - broadcom
  - qualcomm
  - modular
  - nvidia
  - skypilot
  - modal
  - anthropic
  - hugging-face
models:
  - dflash
  - nemo-automodel
  - claude
topics:
  - hardware
  - inference
  - performance-optimization
  - model-training
  - agent-ux
  - security
  - capability-based-security
  - open-source
  - fine-tuning
  - infrastructure
  - model-optimization
people:
  - gdb
  - kimmonismus
  - scaling01
  - clattner_llvm
  - karpathy
  - gallabytes
  - dabit3
  - kentonvarda
  - random_walker
  - jubbaonjeans
  - victormustar
---


**a quiet day.**

> AI News for 6/23/2026-6/24/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap


**OpenAI’s Jalapeño Chip and the Race Toward Full-Stack AI Infrastructure**

- **OpenAI goes deeper into hardware**: [OpenAI](https://x.com/OpenAI/status/2069770172802773292) announced **Jalapeño**, its first custom AI chip for LLM inference, built with **Broadcom** and intended for ChatGPT, Codex, API traffic, and future agent products. The strategic message is straightforward: own more of the stack—chips, kernels, memory, networking, scheduling, deployment—so compute economics and product behavior become less dependent on merchant GPU supply. [@gdb](https://x.com/gdb/status/2069809298612621629) emphasized strong **performance-per-watt**, while [@kimmonismus](https://x.com/kimmonismus/status/2069795647956373632) highlighted the reported **9-month design-to-tapeout cycle**, unusually fast for a high-performance ASIC and reportedly accelerated by OpenAI’s own models.
- **Technical read-through and ecosystem implications**: Community reverse-engineering suggests Jalapeño looks TPU-like: [@scaling01](https://x.com/scaling01/status/2069867464716939413) estimated a near-reticle die, roughly **216GB HBM3E**, **~7.1–7.4 TB/s bandwidth**, and **~10 PFLOPS FP4**. Even if those numbers remain unofficial, the signal is that hyperscaler-style inference silicon is now table stakes for frontier labs. The same day also reshaped the compiler/runtime landscape: [Chris Lattner announced](https://x.com/clattner_llvm/status/2069769232477192354) **Qualcomm is acquiring Modular**, while [Modular said](https://x.com/Modular/status/2069787078032834635) **Mojo open-sourcing remains on track**. That combination points to more serious competition around vertically integrated inference stacks beyond NVIDIA/CUDA.
- **Serving and throughput remain active fronts**: On the infra side, [NVIDIA](https://x.com/NVIDIAAI/status/2069813582825418828) said **NeMo AutoModel** delivers **3.4–3.7x higher training throughput** for MoE models via Expert Parallelism, DeepEP, and TransformerEngine kernels. [SkyPilot](https://x.com/skypilot_org/status/2069815107891388477) launched **Endpoints** for unified inference across owned clusters, and [Modal](https://x.com/modal/status/2069818060991762809) claimed open-source inference setups outperforming proprietary providers on latency. For local optimization, [@jon_durbin](https://x.com/jon_durbin/status/2069876870628155397) reported **30–50% real-world decode gains** from training custom **DFLASH** draft/speculator models.

**Agent UX Shifts From “Tool” to “Coworker,” Raising New Security and Cost Questions**

- **Anthropic’s Slack-native agent model is the big UI story**: Several tweets converged on the significance of Claude embedded into Slack/team workflows. [@karpathy](https://x.com/karpathy/status/2069822834160124091) argued people are underrating it because it is not “just a feature” or Slack bot, but an **org-level harness**. [@gallabytes](https://x.com/gallabytes/status/2069808735212716225) described the experiential jump from Claude Code as a “pairing partner” to Tags as “managing a team.” [@dabit3](https://x.com/dabit3/status/2069785904206508241) pushed the idea further: eventually, you may not even need to explicitly tag agents.
- **The hard part is identity, permissions, and lock-in**: Anthropic detailed its **agent identity** model in [this thread](https://x.com/ClaudeDevs/status/2069895377080443271): Claude gets its own credentials, actions are auditable under that identity, and access can be revoked centrally. That design drew both praise and concern. [@KentonVarda](https://x.com/KentonVarda/status/2069765917018382568) argued explicit per-agent permissioning does not scale and advocated **capability-based security** with fine-grained, task-scoped access. [@random_walker](https://x.com/random_walker/status/2069760540709208306) framed Claude Tag as “a coworker that remembers everything and bills by the thought,” warning of tacit-knowledge lock-in, prompt-injection risk, and budget opacity once one shared agent becomes deeply embedded in org workflows. [@JubbaOnJeans](https://x.com/JubbaOnJeans/status/2069798018879238517) similarly flagged attribution ambiguity for write actions and future access-control complexity outside clean Slack-like boundaries.
- **The open/DIY response is immediate**: Hugging Face described its internal Slack-based coding agent **Moon Bot** in [a blog tweet](https://x.com/victormustar/status/2069696147526947290), emphasizing self-hosting, custom tools, auditable sessions, and zero lock-in. A follow-up from [@calebfahlgren](https://x.com/calebfahlgren/status/2069768499510013978) listed production integrations spanning GitHub, Athena, analytics, MongoDB, Elasticsearch, and HF Buckets. The larger pattern: teams increasingly want agent-native UX, but many would rather own the harness and memory layer than outsource organizational intelligence to a vendor.

**Qwen-AgentWorld, OpenThoughts-Agent, and Memory as the Next Agent Scaling Axis**

- **Qwen-AgentWorld pushes “language world models” for agents**: Alibaba Qwen introduced [Qwen-AgentWorld](https://x.com/Alibaba_Qwen/status/2069720365442719867), positioning it as a native **language world model** that simulates **7 environments**—MCP, Search, Terminal, SWE, Web, OS, Android—inside a single model. Qwen claims two paths: build the simulator itself, and use world modeling as agent pretraining. They open-sourced [Qwen-AgentWorld-35B-A3B and AgentWorldBench](https://x.com/Alibaba_Qwen/status/2069720412481888400), with a **35B MoE / 3B active**, **256K context** model. One notable result: single-turn environment prediction transfers to multi-turn agent tasks with gains across both in-domain and out-of-domain benchmarks, as summarized in [this follow-up](https://x.com/Alibaba_Qwen/status/2069720397747220493).
- **OpenThoughts-Agent contributes a serious open data recipe**: [@iScienceLuvr](https://x.com/iScienceLuvr/status/2069643721155793114) and [@RichardZ412](https://x.com/RichardZ412/status/2069827815403557287) highlighted **OpenThoughts-Agent**, an open curation/training pipeline for agentic models with **100+ controlled ablations**. The team builds a **100K-example** training set and fine-tunes **Qwen3-32B**, reaching **44.8% average accuracy across seven agentic benchmarks**. The key findings are useful for practitioners: instruction choice matters disproportionately, strongest benchmark teacher ≠ best teacher, longer execution traces help, and source diversity beats over-repetition at scale.
- **Memory is turning into a first-class systems layer**: A lot of high-signal discussion centered on memory as the unresolved problem in agents. [Weaviate’s Engram GA](https://x.com/victorialslocum/status/2069722431460168171) frames memory as asynchronous infrastructure that extracts, deduplicates, reconciles, and scopes memories rather than dumping everything into context. [@hwchase17](https://x.com/hwchase17/status/2069857129272627626) showed a LangSmith/Context Hub workflow for “sleep-time compute,” where traces are analyzed offline and written back as memory. [@dair_ai](https://x.com/dair_ai/status/2069846777977880769) pointed to a paper arguing agent memory should be evaluated as a full **data-management layer**—storage, retrieval, update, consolidation, lifecycle—not a black box judged only by end-task success. This is increasingly where agent differentiation appears to be moving.

**Chinese Open Models Keep Closing the Gap: GLM-5.2, Kimi Distribution, and Compute Scale**

- **GLM-5.2 continues to dominate the open-model conversation**: Multiple tweets positioned **GLM-5.2** as the strongest open-weight contender right now. [CoreWeave](https://x.com/CoreWeave/status/2069874833576321150) said it tops open-model rankings on Artificial Analysis and Agent Arena, while [Baseten](https://x.com/baseten/status/2069832610289709156) and [Cursor availability](https://x.com/ZixuanLi_/status/2069921339817795869) showed rapid serving/distribution uptake. [@nutlope](https://x.com/nutlope/status/2069827178569638243) compared GLM 5.2 against Opus 4.8 on web tasks, reporting **similar quality**, **~2x token output**, but still **faster** and roughly **3x cheaper**. [Arena](https://x.com/arena/status/2069885722333769963) also said GLM-5.2 Max leads Code Arena: Frontend against a strong field.
- **Benchmark nuance matters**: GLM-5.2 also showed up on ARC-AGI-2. [@fchollet](https://x.com/fchollet/status/2069858556552298519) called it the **strongest ARC-AGI-2 result to date by an open-source model**, while others debated what its **22.8%** really implies relative to frontier Western models. The broader takeaway is less about any single benchmark and more about open Chinese models being consistently “in the room” across coding, agents, and knowledge work.
- **Commercialization and infrastructure acceleration**: [Moonshot’s Kimi API](https://x.com/Kimi_Moonshot/status/2069718757338202140) is now on **AWS Marketplace**, easing enterprise procurement via consolidated billing and EDP drawdown. Meanwhile, Chinese domestic compute remains a major theme: [@teortaxesTex](https://x.com/teortaxesTex/status/2069760099925524864) flagged reports that Huawei may demo a **950 SuperPOD** scale system, implying production of large domestic NPU clusters at meaningful scale. If true, that would materially improve the economics and resilience of China’s model-serving ecosystem.

**Policy, Talent, and Frontier-Lab Strategy Are Reshaping the Competitive Landscape**

- **Anthropic remains at the center of policy disputes**: [@kimmonismus](https://x.com/kimmonismus/status/2069704003311567045) reported the first major legal challenge to Trump-era AI export controls, with Legion arguing hosted model access is not equivalent to exporting weights or technical data. In parallel, the much-discussed Mythos story gained context: [Reuters/AP details summarized here](https://x.com/kimmonismus/status/2069692592250360126) suggest Anthropic’s model found vulnerabilities in sensitive U.S. systems during a restricted testing exercise, though some commenters warned earlier coverage had been overstated.
- **Distillation and access control are becoming geopolitical issues**: [@kimmonismus](https://x.com/kimmonismus/status/2069879640835961277) also reported Anthropic’s accusation that Alibaba-linked operators used **~25,000 fraudulent accounts** and **28.8 million Claude exchanges** to distill frontier capabilities into Qwen-class systems. If accurate, that escalates the “adversarial distillation” debate from rumor to something closer to enforcement and statecraft.
- **Talent and new labs**: The day also brought talent movement and new institutional formation. [Arthur Conmy joining Anthropic](https://x.com/ArthurConmy/status/2069820098890674334) is notable on the alignment side. [Mirendil AI launched](https://x.com/bneyshabur/status/2069860934148079800) with a **$200M seed round** and a thesis around self-accelerating AI R&D for science. In the UK, [BOLD Lab and SOFAIR](https://x.com/KanishkaNarayan/status/2069777169551671420) received **£60M** in seed funding across two new national fundamental AI labs, with [UCL DARK merging into BOLD](https://x.com/_rockt/status/2069713868918587399). And on the commercial side, [Bloomberg-reported departures from Google DeepMind toward Anthropic](https://x.com/kimmonismus/status/2069870513283871203) underscore how startup upside is continuing to pull frontier talent.

**Top Tweets (by engagement)**

- **OpenAI Jalapeño**: [OpenAI announces its first custom inference chip](https://x.com/OpenAI/status/2069770172802773292) — the most consequential product/infra launch in the set.
- **GPT-5.5 Instant update**: [OpenAI rolls out a revised GPT-5.5 Instant](https://x.com/OpenAI/status/2069843083701915755) with improved intent understanding, constraint handling, and conversational style.
- **Qwen-AgentWorld**: [Alibaba Qwen launches and open-sources language world models for agents](https://x.com/Alibaba_Qwen/status/2069720365442719867).
- **Anthropic’s agent identity model**: [Claude in Slack now uses its own credentials and audit trail](https://x.com/ClaudeDevs/status/2069895377080443271), clarifying one of the thorniest enterprise-agent design questions.
- **Cursor x Notion**: [Cursor tasks can now be delegated directly from Notion](https://x.com/cursor_ai/status/2069872515548340407), another sign that agent workflows are moving into existing team software rather than living in standalone chat apps.


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. China AI Chip Ecosystem and Controls

  - **[7 Chinese companies are already shipping H100/H200-class AI chips, most IPO'd in the last 6 months. I mapped all of them.](https://www.reddit.com/r/LocalLLaMA/comments/1udkxde/7_chinese_companies_are_already_shipping/)** (Activity: 1423): **The post maps **seven Chinese AI accelerator vendors**—Huawei Ascend, Alibaba T-Head, Baidu Kunlunxin, MetaX, Moore Threads, Biren, and Iluvatar CoreX—claiming current parts are roughly **H100-class** and next-gen parts target **H200-class**, based largely on a CHITEX/Dmitry Shilov deck and the author’s linked [X thread](https://x.com/superalesha/status/2069415581237813437). Key cited specs include **Huawei Ascend 910C/910D/950** roadmaps with domestic HBM, Alibaba’s `16×96GB` PG1 server totaling `1.536TB` VRAM, MetaX C600 with `144GB HBM3e`, Moore Threads S5000 with `80GB` and `1 PFLOPS`, and Biren/Iluvatar roadmaps adding FP8/FP4 and edge-inference modules. The larger claim is that Chinese AI infrastructure is moving from NVIDIA/CUDA dependence toward a domestic stack: OAM-like modules, proprietary interconnects, SMIC production, near-100% utilization, and Chinese open-weight models such as Qwen/DeepSeek/GLM increasingly being tuned first for non-NVIDIA accelerators.** Top comments were skeptical about practical access and deployment: users asked whether these systems would be available in Europe or even via AliExpress, while the most substantive concern was that *“the software stack”*—CUDA compatibility, drivers, compiler/runtime maturity, and framework integration—will be the main bottleneck regardless of raw hardware specs.

    - A technically detailed critique argues that the post overstates real deployability: `1,536 GB` of aggregate VRAM is not sufficient to run a `~1,510 GB` BF16 model once runtime overhead, KV cache, activations, fragmentation, and distributed execution requirements are included. The commenter also challenges the “H100/H200-class” framing by noting Huawei Ascend 950PR reportedly has `128GB` VRAM at `1.6TB/s` and `1 PFLOPS FP8`, versus NVIDIA H200’s `144GB`, `4.8TB/s`, and `2 PFLOPS dense FP8`, making memory bandwidth and compute materially lower despite vendor claims.
    - Several claims are called out as “shipping soon” rather than currently shipping. For example, the commenter says Kunlun M100 lacks publicly findable core specs such as memory size, bandwidth, or TFLOPS, while existing `vLLM` support appears to target older Kunlun chips rather than the M100.
    - The Moore Threads / C-series claims are questioned: the commenter says current shipments appear to be C500/C550-class parts with less impressive specs, likely `64GB` GDDR6, while the C600’s advertised `144GB HBM3e` and H200 positioning are still future mass-production claims. They emphasize that moving from GDDR6 products to HBM3e at scale is a major unproven manufacturing and integration jump.

  - **[Seems this community might have missed it: Bill that would mandate AI chip location tracking gains industry support | Half a dozen companies have come out in support of the Chip Security Act, which would require location-tracking mechanisms for America’s most advanced computing chips.](https://www.reddit.com/r/LocalLLaMA/comments/1ue2fd7/seems_this_community_might_have_missed_it_bill/)** (Activity: 440): **The post points to several-day-old coverage (also discussed on [r/politics](https://www.reddit.com/r/politics/comments/1uahgcs/bill_that_would_mandate_ai_chip_location_tracking/) and [r/LocalLLM](https://www.reddit.com/r/LocalLLM/comments/1ubz5xh/us_to_require_location_tracking_for_ai_and/)) that the proposed **Chip Security Act** would require location-tracking mechanisms for the most advanced U.S. AI accelerators. Technically, this implies adding some form of hardware/firmware-level geolocation, attestation, or reporting capability to export-controlled compute devices, with the stated goal of preventing diversion of high-end AI chips to restricted jurisdictions.** Top comments were broadly hostile, arguing the mandate could weaken U.S. competitiveness versus China and introduce new security/privacy risks; one commenter mocked the idea as *“the best most secure location tracking mechanism”* with *“no security issues.”*



### 2. Open Model Releases for OCR and Agent Simulation

  - **[Unlimited-OCR is now on ModelScope! A 3.3B multilingual OCR model for one-shot parsing across single images, multi-page documents, and PDFs. License: MIT](https://www.reddit.com/r/LocalLLaMA/comments/1ue51uk/unlimitedocr_is_now_on_modelscope_a_33b/)** (Activity: 948): ****Baidu’s Unlimited-OCR** is announced on [ModelScope](https://x.com/ModelScope2022/status/2069335055965491525) as an **MIT-licensed `3.3B` multilingual OCR/document-parsing model** for one-shot parsing of single images, multi-page documents, and PDFs, with up to **`32K` output tokens** for long OCR sequences. The [GitHub repo](https://github.com/baidu/Unlimited-OCR) advertises Transformers inference plus **SGLang serving** with OpenAI-compatible streaming, and two image/layout modes: `base` and `gundam`.** Technical commenters asked how it compares with **PaddleOCR-VL-1.6**, how many pages fit within the `32K` output limit, what `gundam`/“gundan” mode means, and whether **Paddle** support is missing.

    - Commenters asked for concrete comparative evaluation against **PaddleOCR-VL-1.6**, specifically throughput/accuracy tradeoffs and how many document pages can fit within the model’s `32k` context limit during multi-page/PDF parsing.
    - Several users questioned unclear terminology in the release, especially *“gundam mode”*, suggesting the ModelScope/Hugging Face documentation needs to define this mode and its effect on OCR behavior or document parsing. The Hugging Face model card was linked here: https://huggingface.co/baidu/Unlimited-OCR

  - **[Qwen-AgentWorld-35B-A3B: a 3B-active MoE trained to simulate MCP, terminal, SWE, Android, web and OS environments](https://www.reddit.com/r/LocalLLaMA/comments/1ue5149/qwenagentworld35ba3b_a_3bactive_moe_trained_to/)** (Activity: 292): ****Qwen** released [`Qwen-AgentWorld-35B-A3B`](https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B), a `35B`-parameter MoE with roughly `3B` active parameters per token, positioned as a **language world model** rather than a chat/instruction model or autonomous agent. It is trained to predict environment observations after agent actions across **MCP/tool calling, search, terminal, SWE, Android, web, and OS GUI** domains, enabling mocked/simulated agent loops for offline evaluation, synthetic trajectory generation, tool-use workflow testing, and sandbox-like training without invoking real tools.** Comments were mostly light, but one technical reaction noted it could be useful for evals by mocking actions such as predicting terminal output for `ls -la`; others joked/skeptically suggested the training may resemble swapping user/assistant roles or prompting *“You are an MCP server now.”*

    - One commenter highlights a concrete use case: training a model to predict environment responses, e.g. given a user command like `ls -la`, generate the corresponding terminal output. They suggest this could be useful for **evaluation harnesses or mock environments**, where agent actions can be simulated without invoking a real terminal or external tool.
    - Another technically relevant thread frames Qwen-AgentWorld-35B-A3B as a possible **world-model-style component for LLM agents**, comparing it conceptually to Yann LeCun’s world-model work. The commenter notes that applying environment simulation directly to LLM reasoning/training across MCP, SWE, Android, web, OS, and terminal settings could improve agent capabilities if the benchmark gains generalize.




## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Krea 2 Open-Source Image Model and GenAI Fidelity

  - **[We are the team behind Krea 2. Ask us anything!](https://www.reddit.com/r/StableDiffusion/comments/1udnm0a/we_are_the_team_behind_krea_2_ask_us_anything/)** (Activity: 1017): ****Krea** announced that **Krea 2**, an in-house-trained open-source text-to-image model, is available with code/weights via [krea.ai/krea-2-open-source](http://krea.ai/krea-2-open-source), [GitHub](http://github.com/krea-ai/krea-2), and Hugging Face checkpoints for [`Krea-2-Raw`](http://huggingface.co/krea/Krea-2-Raw) and [`Krea-2-Turbo`](http://huggingface.co/krea/Krea-2-Turbo). The head of research said this is their first fully in-house open-source model release and that they are considering releasing additional artifacts such as a **Turbo checkpoint without guidance/step distillation**, **5B variants**, and capability-focused improvements including **image references, editing, bounding boxes, better text rendering, and realism**.** Commenters focused on roadmap/architecture questions: whether an **image editing** version will be released, and why Krea chose the **Qwen VAE** instead of the **Flux 2 VAE**.

    - A Krea 2 researcher noted this is their first fully in-house trained open-source model release and that both **raw** and **turbo** checkpoints were released. They are considering additional open releases based on community feedback, including a **Turbo checkpoint without guidance / step distillation**, **5B checkpoint variants**, and capability-focused improvements such as image references, editing, bounding boxes, improved text rendering, and realism.
    - Several commenters focused on model-component and training-choice transparency, especially why Krea 2 uses a **Qwen VAE** rather than a **FLUX.2 VAE**. Another technically relevant request was for Krea to release its **aesthetic reward model**, with the commenter arguing that open-source image generation currently lacks strong reward models for preference/aesthetic optimization.
    - Feature requests centered on downstream controllability: users asked whether Krea 2 will get an **image editing** variant and whether there will be support for **style transfer**. These requests align with the researcher’s listed possible future capability expansions around image references and editing workflows.

  - **[I aged and restored a photo of myself](https://www.reddit.com/r/ChatGPT/comments/1ud6wuy/i_aged_and_restored_a_photo_of_myself/)** (Activity: 3288): **The image ([link](https://i.redd.it/rqbz1fkqhy8h1.png)) is a controlled four-panel test of **ChatGPT image restoration/colorization**: the poster starts from a known original portrait, artificially ages/damages it, then asks ChatGPT to restore it. The result demonstrates a key limitation of generative restoration: instead of recovering the original face, the model **hallucinates plausible facial details**, making the subject look like a different, older person with altered beard/face structure and sharpened invented features.** Commenters largely read it as evidence that AI “restoration” is not faithful reconstruction but generation conditioned on a degraded input. One commenter connected this to risks in face recognition/security systems, while others joked that the restored version resembles Jack Black.

    - A commenter argues the result demonstrates a core limitation of AI age-transformation/restoration workflows: the output can become *“a completely different person”* rather than preserving identity. They explicitly connect this identity drift to potential failure modes in AI-based face recognition and security systems.
    - One user compares restoration workflows by taking the **“Aged by Gemini”** output, cropping it back to the original framing, and running it through **NanoBananaPro**, claiming it is *“still way better for restorations”* and produced a better result on the first attempt. They note that the Gemini-aged image appeared to zoom out, so framing/cropping materially affected the restoration pipeline and that the second image was *“doing a LOT”* of reconstruction.

  - **[Japanese animator using Seedance to render anime from simple 3D models](https://www.reddit.com/r/singularity/comments/1ue6yoh/japanese_animator_using_seedance_to_render_anime/)** (Activity: 2674): **A Reddit post highlights a Japanese animator reportedly using **Seedance** to generate/render anime footage from simple **3D models**, suggesting a workflow where coarse 3D scene/blocking provides spatial and temporal consistency for AI video generation. The linked Reddit video is inaccessible via the provided URL due to **HTTP 403 Forbidden**, but commenters identify the animator as [**Tetsurou**](https://x.com/craftcapitallab), reportedly an anime-industry veteran with credits on **TRIGUN STAMPEDE** and **TRIGUN STARGAZE**.** Commenters frame this as a plausible path toward long-format AI video with a consistent world model, and debate whether the animator’s use of 3D control/input provides enough intentionality to qualify as art. One commenter argues the result looks better than typical CGI in anime, while dismissing anti-AI-art objections as gatekeeping.

    - Commenters identify the workflow as a plausible path for **long-format video consistency**: using simple 3D models/layouts as a stable scene/pose/world representation, then having **Seedance** render the final anime look. One user notes this could allow style-swapping via prompt changes, e.g. anime to **photoreal** or **retro comic**, while preserving the underlying staged motion and composition.
    - A technically relevant production point is that AI could target animation labor such as **inbetweening**—generating the intermediate frames between keyframes—which one commenter describes as a major cost driver that contributes less directly to perceived creative quality than layout, acting, or key animation. This frames the Seedance-style pipeline as potentially useful for reducing production cost while retaining human-authored direction through 3D blocking and prompts.
    - The creator is credited as [**Tetsurou**](https://x.com/craftcapitallab), reportedly an anime-industry veteran of over `10 years` with recent work on **TRIGUN STAMPEDE** and **TRIGUN STARGAZE**. That context matters technically because the demo appears less like raw text-to-video generation and more like an experienced animator using AI as a renderer/compositor over intentional 3D staging.


### 2. AI Datacenter Backlash and Defense

  - **[Data center noise irks Virginia neighbors: ‘You just want to curse’, Neighbors have put mattresses and plexiglass up in their windows to block the noise from this data center in Virginia. It's a high pitched whine from the natural gas turbines that power it. The noise never stops 24/7. - NewsNation](https://www.reddit.com/r/singularity/comments/1ue6sio/data_center_noise_irks_virginia_neighbors_you/)** (Activity: 2474): **A Virginia data center is reportedly generating continuous `24/7` high-pitched noise from on-site **natural-gas turbines**, severe enough that nearby residents have added mattresses and plexiglass to windows for sound attenuation. Commenters focused on the technical/siting issue: if the facility is not grid-connected and primarily needs fiber/network access plus power generation, placing turbine-powered infrastructure near residential neighborhoods suggests a zoning/permitting failure rather than an inherent data-center requirement.** Top comments questioned how this was permitted under U.S. suburban zoning, argued that more data centers are needed but not in residential areas, and broadly characterized the site selection as indefensible.

    - Commenters framed the problem as primarily a **power-generation siting issue**, not inherent data-center noise: the facility reportedly uses **on-site natural-gas turbines** rather than grid power, producing a continuous high-pitched whine that would normally be inappropriate near residential zoning. One technical takeaway was that data centers can be sited flexibly because they mainly need **power, cooling, and network connectivity**, so commenters argued there is little engineering need to place turbine-backed infrastructure in a neighborhood.
    - Several comments questioned the regulatory/planning failure: users contrasted this Virginia case with **EU/UK planning regimes**, where industrial noise sources such as gas turbines would typically face stricter permitting, environmental-noise review, and separation from residential areas. The discussion emphasized that stronger zoning or permitting could require grid interconnection, acoustic mitigation, or relocation rather than allowing `24/7` turbine operation next to homes.

  - **[John Carmack weighs in on datacenters](https://www.reddit.com/r/singularity/comments/1ue1sya/john_carmack_weighs_in_on_datacenters/)** (Activity: 2034): **The image is a [screenshot of an X/Twitter exchange](https://i.redd.it/mius3v4nc59h1.png) where **John Carmack** argues that public opposition to data centers could become analogous to U.S. anti-nuclear sentiment, potentially slowing AI infrastructure deployment. He frames data-center demand as evidence of *“real value and progress”* tied to a major AI-driven transition, while **Markus “notch” Persson** challenges him with a simple *“Why?”*** Comments push back on Carmack’s framing, arguing for a middle ground: data centers should be allowed where they do not create local nuisances and should supply their own power/water. Others note that anti-nuclear sentiment was partly influenced by fossil-fuel interests and suggest those same interests may now benefit from AI data-center energy demand.

    - Several commenters framed datacenter expansion as primarily an **infrastructure siting and resource-provisioning problem**: build freely only where facilities do not create residential nuisances, and require operators to bring or secure their own **power and water** rather than burdening local grids or municipal utilities. Noise and waste heat were specifically called out as siting constraints, with opposition to placing large facilities near towns where cooling exhaust and acoustic load affect residents.
    - A recurring technical-policy theme was that large AI datacenter growth should be paired with **new reliable generation**, especially nuclear power, before scaling further. Commenters argued that “safe nuclear power” would better match datacenters’ high, continuous load profiles than simply expanding fossil-fuel-backed capacity, while also noting concerns that oil and coal interests benefit from AI load growth if new clean baseload is not built.


### 3. Gemini and Fable Model-Launch Rumors

  - **[3.5 pro Coming this week](https://www.reddit.com/r/GeminiAI/comments/1uei7js/35_pro_coming_this_week/)** (Activity: 1211): **The image is a screenshot of an unverified tweet claiming a **“Gemini 3.5 Pro”** release “this week,” with rumored upgrades including stronger vision/multimodal reasoning, improved memory/context retention, agent workflows, SVG/frontend generation, a native image model, a “Gemini Super App,” and a claimed `2.5M` token context window ([image](https://i.redd.it/kxh47zuxa99h1.png)). Technically, the post is speculative rather than an announcement: commenters note the lack of coding benchmark claims and question whether it would outperform existing Gemini 3.x/2.5 Pro previews or compete with GPT/Claude/Fable-tier coding models.** Comments are skeptical, with users saying Google should “release it first” and avoid regressions, while others argue the `2.5M` context-window claim sounds fake and expect `1M` again.

    - Commenters questioned whether **3.5 Pro** would actually improve over **3.1 Pro Preview**, with one noting that if the leak were credible and the model were strong, the announcement would likely emphasize **leading coding benchmarks**; the absence of such claims was interpreted as a possible sign it may not beat current top coding models.
    - A claimed **`2.5M` context window** was treated skeptically, with users arguing that **`1M` tokens** is more plausible and that the inflated context-size claim makes the leak look fake.
    - One technically relevant concern was model routing under load: users joked/complained that even paid **Pro** subscribers might receive responses from a fallback model during “intense usage,” which would make real-world quality inconsistent despite access to the advertised model.

  - **[Fable 5 return RUMORED with some hints in CC](https://www.reddit.com/r/ClaudeAI/comments/1uehr3a/fable_5_return_rumored_with_some_hints_in_cc/)** (Activity: 845): **A rumor based on **Claude Code `v2.1.190` string changes** claims Anthropic may be preparing a permanent subscription inclusion for **Fable 5**, with a weekly quota: the added string *“You've used your Fable 5 usage for this week”* and removal of *“purchased separately from your plan”* are cited as evidence ([source](https://x.com/synthwavedd/status/2069813760622043483)). If accurate, this would suggest Fable 5 access may move from time-limited/separately purchased availability to recurring capped usage within subscriptions.** Comments were mostly hype/speculation; the only substantive preference was that a low weekly cap would be better than a short temporary subscription window, since it preserves ongoing access.

    - One commenter raised a concrete product-access concern: they would prefer a **low weekly usage cap** for Fable over a subscription model that only grants access for a limited `two-week` window, arguing that capped ongoing access is more useful than time-limited availability.




# AI Discords

Unfortunately, Discord shut down our access today. We will not bring it back in this form but we will be shipping the new AINews soon. Thanks for reading to here, it was a good run.