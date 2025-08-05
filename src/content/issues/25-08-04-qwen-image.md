---
id: MjAyNS0w
title: 'Qwen-Image: SOTA text rendering + 4o-imagegen-level Editing Open Weights MMDiT'
date: '2025-08-04T05:44:39.731046Z'
description: >-
  **Alibaba** surprised with the release of **Qwen-Image**, a **20B MMDiT**
  model excelling at bilingual text rendering and image editing, with open
  weights and detailed transparency including synthetic data usage. **Google
  DeepMind** launched **Gemini 2.5 Deep Think** to Ultra subscribers, showing
  significant reasoning improvements and benchmark gains close to **OpenAI's o3
  Pro**. ByteDance's **SeedProver** achieved record scores on math benchmarks,
  surpassing DeepMind's **AlphaGeometry2**, while OpenAI is developing a
  "universal verifier" for math and coding. Competitive reasoning benchmarks and
  game arenas by Google and Kaggle highlight a meta-shift in reasoning model
  efficiency, with models like **Grok 4** leading instruction-following
  benchmarks. Other open-weight models gaining momentum include **GLM-4.5**,
  **XBai o4**, and **Tencent Hunyuan**. *"Reasoning models represent a
  compute-equivalent gain on the order of 10×,"* per community analysis.
companies:
  - alibaba
  - google-deepmind
  - openai
  - bytedance
  - kaggle
  - tencent
models:
  - qwen-image
  - mmdit-20b
  - gemini-2.5
  - o3-pro
  - seedprover
  - alphageometry2
  - grok-4
  - glm-4.5
  - xbai-o4
  - hunyuan
topics:
  - bilingual-text-rendering
  - image-generation
  - image-editing
  - synthetic-data
  - reasoning
  - math-proving
  - benchmarking
  - instruction-following
  - model-transparency
  - open-weight-models
  - competitive-evaluation
people:
  - swyx
  - demishassabis
  - tulseedoshi
  - mparakhin
  - teortaxestex
  - cgeorgiaw
  - dorialexander
  - steph_palazzolo
  - corbtt
  - synthwavedd
  - epochairesearch
---


**Qwen is all you need.**

> AI News for 8/1/2025-8/4/2025. We checked 12 subreddits, 544 Twitters and 29 Discords (227 channels, and 14214 messages) for you. Estimated reading time saved (at 200wpm): 1248 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

In a surprise model drop, the Alibaba Qwen team [announced](https://x.com/Alibaba_Qwen/status/1952398250121756992) a 20B MMDiT model "especially strong at creating stunning graphic posters with native text" ([blog](https://qwenlm.github.io/blog/qwen-image/), [paper](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/Qwen_Image.pdf#page=14.57)).

[](https://resend-attachments.s3.amazonaws.com/aZx2Cus84j7nUjl)

Ask your friendly neighborhood Sinophone if they've ever seen this level of non-Arabic text rendering:

[](https://resend-attachments.s3.amazonaws.com/eYc92vDYGrg3osA)

[](https://resend-attachments.s3.amazonaws.com/WCJt5u4A5z6B7zH)

but of course they do well on English too:

[](https://resend-attachments.s3.amazonaws.com/LuRphbToh17jqHN)

other than pure image generation they are also shockingly good at image editing, favorably comparing to [Flux Kontext](https://flux-ai.io/flux-kontext/):

[](https://resend-attachments.s3.amazonaws.com/oVJ3VWbD5nbZn7b)

The 46 page tech report exhibits a level of transparency rare in western labs: https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/Qwen_Image.pdf#page=14.57

[](https://resend-attachments.s3.amazonaws.com/FUBv1l4QqkNW6Lc)

and offers some (but not complete) insight into usage of synthetic data to achieve their text rendering results:

[](https://resend-attachments.s3.amazonaws.com/GeGPPO0kIWDiRxb)

---

# AI Twitter Recap

**Frontier reasoning: Gemini 2.5 Deep Think, new math/proving systems, and head-to-head evals**

- **Google’s Gemini 2.5 Deep Think ships to Ultra subscribers**: DeepMind touts SOTA across hard benchmarks, with early users reporting major uplifts vs prior Gemini and near-parity with OpenAI’s o3 Pro on some tasks. Quantified deltas from community testing: +11.2% AIME (2025), +13.2% HLE (knowledge), +13.4% LiveCodeBench (coding) compared to o3 pro’s smaller gains in analogous tasks, per [@swyx](https://twitter.com/swyx/status/1951460518293807241). Demo threads from [@demishassabis](https://twitter.com/demishassabis/status/1951468051578142848), [@tulseedoshi](https://twitter.com/tulseedoshi/status/1952059171727437859), and [@MParakhin](https://twitter.com/MParakhin/status/1952028947153371631) show improved reasoning and terser outputs, albeit with current usage limits.
- **Math & theorem proving jump**:
    - ByteDance’s SeedProver reports 331/657 on PutnamBench (≈4× prior SOTA), 201/657 under “light” inference, and 100% on OpenAI’s miniF2F, surpassing DeepMind’s AlphaGeometry2 per [@teortaxesTex](https://twitter.com/teortaxesTex/status/1951875052967739787) and summaries by [@cgeorgiaw](https://twitter.com/cgeorgiaw/status/1952301113446699347). Paper thread: [@Dorialexander](https://twitter.com/Dorialexander/status/1952094475725238479).
    - OpenAI is reportedly developing a “universal verifier” to transfer math/coding gains to subjective domains, per [@steph_palazzolo](https://twitter.com/steph_palazzolo/status/1952375778361954801); related open efforts include RULER universal rewards from [@corbtt](https://twitter.com/corbtt/status/1952437149544144984).
    - The “Hieroglyph” benchmark probes lateral reasoning (Only Connect-style): models score <50% on the hardest 20 Qs, per [@synthwavedd](https://twitter.com/synthwavedd/status/1951645151203324099).
- **Benchmarking the meta-shift**: Reasoning models represent a compute-equivalent gain on the order of 10× on tasks amenable to reasoning—comparable to the original Transformer jump—according to [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1951734757483487450). Kaggle and Google launched the Game Arena to pressure-test models in competitive games (starting with text chess), with live commentary by Magnus Carlsen and Hikaru Nakamura; details from [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1952406075996533077) and [@demishassabis](https://twitter.com/demishassabis/status/1952436066524299432). Artificial Analysis updated its model index adding IFBench (instruction following); Grok 4 stays top while o3/o4-mini move ahead of Gemini 2.5 Pro in their methodology ([thread](https://twitter.com/ArtificialAnlys/status/1952302030812483982)).

---

**Open-weight model wave: Qwen-Image, GLM-4.5 momentum, XBai o4, Tencent Hunyuan, and efficient training**

- **Qwen-Image (20B MMDiT) released under Apache-2.0**: Strong bilingual text rendering (rivals GPT‑4o in English; best-in-class Chinese) with in-pixel text synthesis, plus broad image styles; open weights, code, and demos via [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1952398250121756992). Community notes it leverages a fine-tuned Wan 2.1 VAE and Qwen VL text encoder components ([@multimodalart](https://twitter.com/multimodalart/status/1952409238413684901)); the public demo was quickly saturated ([@victormustar](https://twitter.com/victormustar/status/1952416615351366033)).
- **Zhipu AI’s GLM‑4.5 rises in leaderboards**: Now #5 in LMSYS Arena overall with 4K+ votes and strong agent/tool-use showing ([@lmarena_ai](https://twitter.com/lmarena_ai/status/1952402506497020330); [@Zai_org](https://twitter.com/Zai_org/status/1952404744225349799)). Terminal-Bench corroborates top-tier performance among reasoning/code assistants ([link](https://twitter.com/Zai_org/status/1952411485742760324)). Demand temporarily filled [Z.ai](http://z.ai/) Chat’s feature storage ([update](https://twitter.com/Zai_org/status/1951494857039454250)).
- **XBai o4 (parallel test-time scaling)**: Open weights under Apache‑2.0; authors claim it outperforms OpenAI o3‑mini in “medium mode” ([announcement](https://twitter.com/theMetaStoneAI/status/1951486506562101656)).
- **Tencent Hunyuan small family (0.5B/1.8B/4B/7B)**: Edge-ready models (single-card deploy) with 256K context, tool/agent skills, and multi-framework support (SGLang, vLLM, TensorRT-LLM). Repos and HF weights linked by [@TencentHunyuan](https://twitter.com/TencentHunyuan/status/1952262079051940322).
- **Large yet affordable VLMs**: StepFun’s Step‑3 (321B MoE) targets the decoding cost Pareto frontier ([@HuggingPapers](https://twitter.com/HuggingPapers/status/1952038716488208409)).
- **Training/optimization**:
    - GSPO (RL alignment from Qwen) is trending; TRL v0.20 added first‑class support and example scripts ([@SergioPaniego](https://twitter.com/SergioPaniego/status/1952305247411691871)).
    - Microsoft released Dion (distributed optimizer with Muon/MuP options and Triton kernels). Good code quality and infra notes; comms optimizations and FSDP/all-to-all tips discussed by [@jxbz](https://twitter.com/jxbz/status/1951806916440854982) and [@JingyuanLiu123](https://twitter.com/JingyuanLiu123/status/1951885788855345221).
    - Hugging Face’s Ultra‑Scale Playbook (200+ pages, 4,000+ scaling experiments) covers 5D parallelism, ZeRO, FlashAttn, overlap, and bottlenecks; free for HF Pro ([@Thom_Wolf](https://twitter.com/Thom_Wolf/status/1951581743607070851), [@ClementDelangue](https://twitter.com/ClementDelangue/status/1952048356710039700)).
- **Coding-specialist ecosystem**:
    - Qwen3‑Coder runs “17× faster” on Cerebras and is free to try; hackathon ran this weekend ([@SarahChieng](https://twitter.com/SarahChieng/status/1951453803905163693)). Educational MoE notebook (128 experts, 8 active; single A100) by [@rasbt](https://twitter.com/rasbt/status/1951635208375034191).
    - Smaller fast variants available on Fireworks (Qwen3‑Coder‑Flash, GLM‑4.5‑Air) with competitive tool-use quality vs larger siblings for simple, low-latency tasks ([@dzhulgakov](https://twitter.com/dzhulgakov/status/1952049826067050735)).

---

**Agent systems and coding: Claude Code evolves, infra matures, and “deep agents” patterns**

- **Claude Code updates**: Microcompact (auto-clears old tool calls to extend sessions), subagents with @‑mention and per-agent model selection, and native PDF ingestion landed ([@_catwu](https://twitter.com/_catwu/status/1952488684579873195)). Context/pruning remains a common tuning pain point; multiple users highlight verbosity vs concision tradeoffs ([e.g.](https://twitter.com/giffmana/status/1952434564472644016)).
- **Ecosystem**:
    - Cline × Cerebras hackathon drew 800+ developers for instant “vibe coding” ([@CerebrasSystems](https://twitter.com/CerebrasSystems/status/1952511328964509794)). Opencode added Together’s model suite ([@togethercompute](https://twitter.com/togethercompute/status/1952495692557046141)), and Kilo integrates GLM‑4.5 ([Z.ai](http://z.ai/) [providers](https://twitter.com/Zai_org/status/1952390223742255504)).
    - Amp vs Claude Code head‑to‑head deep dive incoming ([@isaac_flath](https://twitter.com/isaac_flath/status/1952399160579366957)); Jules (Google) now opens PRs in‑loop ([@julesagent](https://twitter.com/julesagent/status/1952446750167310456)); Lindy 3.0 ships Agent Builder, Autopilot, and team collaboration ([@Altimor](https://twitter.com/Altimor/status/1952414217187086441)).
- **Design patterns**:
    - “Deep agents” (LangChain) formalize multi‑step subagents with virtual filesystem state; code walkthrough by [@hwchase17](https://twitter.com/hwchase17/status/1952408450878918834).
    - Reflective prompt evolution can rival RL in compound systems (GEPA), per [@CShorten30](https://twitter.com/CShorten30/status/1952376642283708788); OpenPipe’s RULER offers relative universal rewards ([link](https://twitter.com/corbtt/status/1952437149544144984)).
    - Memory is emerging as critical infra for agentic personalization and efficiency—taxonomy and strategies by [@_philschmid](https://twitter.com/_philschmid/status/1952370348600533000).
- **Policy friction**: Anthropic stated it restricted OpenAI’s Claude Code access for ToS violations and heavy internal OAI usage, while keeping API access for safety evals/benchmarks ([@sammcallister](https://twitter.com/sammcallister/status/1951642025381511608)).

---

**Multimodal generation and video: Grok Imagine, Runway Aleph, Veo 3, and toward real-time**

- **Grok Imagine rollout**: xAI’s image/video generation is now in the app (initially via waitlist, then Premium+ and Premium), with rapid-generation demos and broad endorsements ([@tetsuoai](https://twitter.com/tetsuoai/status/1951444393065586840), [@tobi](https://twitter.com/tobi/status/1951789462268391749), [@chaitualuru](https://twitter.com/chaitualuru/status/1952174534142067092), [@obeydulX](https://twitter.com/obeydulX/status/1951724900198367515)). Elon Musk reports 6s clips rendering in 15s (down from 60s) and targets real-time in 3–6 months ([progress](https://twitter.com/elonmusk/status/1951883927582552547); “ideas as fast as Imagine” [tweet](https://twitter.com/elonmusk/status/1951516837906202782)).
- **Runway Aleph**: General release (web + API), with strong classroom adoption (USC/UPenn) and rapidly improving controllability/extensibility ([@runwayml](https://twitter.com/runwayml/status/1951634909501575659), [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1951663311503688018), [customers](https://twitter.com/c_valenzuelab/status/1951568696155017286)). Community experiments show multi-step composites and “infinite UIs” control paradigms (e.g., Blender + Aleph workflows [example](https://twitter.com/c_valenzuelab/status/1952419024291188794)).
- **Veo 3 image-to-video**: Available in the Video Arena Discord for side-by-side tests ([@lmarena_ai](https://twitter.com/lmarena_ai/status/1952052092719517729)). Video arena invites broad model comparisons and voting (Discord link in thread).

---

**Open tooling, infra, and the “open models as national priority” push**

- **Open infra maturity**: Hugging Face Inference is pushing “open weights infra” toward parity with proprietary APIs ([@ClementDelangue](https://twitter.com/ClementDelangue/status/1951668724848599143)), Jan adds HF as a remote provider ([@jandotai](https://twitter.com/jandotai/status/1952248389531570333)), and Qdrant Edge enters private beta for embedded vector search ([@qdrant_engine](https://twitter.com/qdrant_engine/status/1951631317939990765)). Modal reiterates it’s a general-purpose compute platform, not just inference ([@bernhardsson](https://twitter.com/bernhardsson/status/1951729049866514508)).
- **ATOM project (US open models)**: Calls for US investment to reclaim open-model leadership (after a summer surge from China) gathered support from researchers across labs ([@natolambert](https://twitter.com/natolambert/status/1952370970762871102), endorsements by [@Miles_Brundage](https://twitter.com/Miles_Brundage/status/1952400404668657966) and [@finbarrtimbers](https://twitter.com/finbarrtimbers/status/1952401883391520794)). VentureBeat op-eds argue “open is critical” ([@bgurley](https://twitter.com/bgurley/status/1952031129143591234)).
- **Data/ops**: Google’s AlphaEvolve shows LLM-driven, test‑loop code evolution yielding novel kernels and infra wins (1% training time cut) ([@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1952112235196678274)). RAG hygiene advances include hierarchical reranking of internal vs external sources to reduce hallucinations ([@qdrant_engine](https://twitter.com/qdrant_engine/status/1951978606617326011)).

---

**Top tweets (by engagement)**

- [@nearcyan](https://twitter.com/nearcyan/status/1951926555934073147): “you wont believe what happens next” (viral meta-commentary on the feed)
- [@sama](https://twitter.com/sama/status/1951695003157426645): “ton of stuff to launch over the next couple of months”
- [@elonmusk](https://twitter.com/elonmusk/status/1951516837906202782) and [render update](https://twitter.com/elonmusk/status/1951883927582552547): on Grok Imagine and near real-time video
- [@karpathy](https://twitter.com/karpathy/status/1951577221753094399): “2024: everyone releases Chat; 2025: everyone releases Code” (plus [PayoutChallenge](https://twitter.com/karpathy/status/1952076108565991588))
- [@gdb](https://twitter.com/gdb/status/1951882297172779336): “more fun than ever to be a software engineer”
- [@LHSummers](https://twitter.com/LHSummers/status/1951998034973163940): on politicization of statistics as authoritarian drift
- [@balajis](https://twitter.com/balajis/status/1951515516939673996): “Prompt your AI like old Twitter” (limit to 140 chars/words/lines)
- [@demishassabis](https://twitter.com/demishassabis/status/1951468051578142848): Gemini 2.5 Deep Think announcement
- [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1951677714173477031): on Thinky’s $1.5B turn-down and MVP retention
- [@OpenAI](https://twitter.com/OpenAI/status/1952414411131671025): optimizing ChatGPT for “un-regrettable time” (break reminders, advice improvements)
- [@naval](https://twitter.com/naval/status/1951900029389820253): “Good teams throw away far more product than they keep”
- [@paulg](https://twitter.com/paulg/status/1952155863864733750): link deprioritization hides the web’s best content

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen-Image 20B Model Release and Benchmarks

- [**QWEN-IMAGE is released!**](https://huggingface.co/Qwen/Qwen-Image) ([Score: 711, Comments: 166](https://www.reddit.com/r/LocalLLaMA/comments/1mhhdig/qwenimage_is_released/)): **QWEN-IMAGE, a newly released vision model, demonstrates superior performance to Flux Kontext Pro in internal benchmarks. According to technical discussion, QWEN-IMAGE supports a broad range of image understanding tasks including object detection, semantic segmentation, depth/edge (Canny) estimation, novel view synthesis, and super-resolution. Qualitative tests, such as complex prompt adherence and text rendering, suggest high fidelity results, although minor font style anomalies are noted in generated text.** Commenters note that prompt fidelity and multi-task capabilities are particularly impressive, with special attention to QWEN-IMAGE's execution of sophisticated image-text combinations surpassing prior open models.
    - QWEN-IMAGE supports a broad suite of image understanding tasks, notably including object detection, semantic segmentation, depth/edge (Canny) estimation, novel view synthesis, and super-resolution, which signals substantial capability in both generative and analytical vision applications.
    - Early user tests demonstrate strong text rendering and semantic understanding, with accurate text placement even in complex, multi-modal prompts (e.g., anthropomorphic characters with environment-based signage). However, nuances like font style and decal clarity may present edge cases.
    - There is notable criticism regarding the model's evaluation plots/visualizations, with some users flagging issues in data presentation quality—suggesting potential concerns when interpreting benchmark performance or training diagnostics.
- [**🚀 Meet Qwen-Image**](https://i.redd.it/7a463it8z0hf1.jpeg) ([Score: 460, Comments: 65](https://www.reddit.com/r/LocalLLaMA/comments/1mhhctd/meet_qwenimage/)): **The provided image is referenced in a post announcing Qwen-Image, a 20B parameter MMDiT model for text-to-image generation, notable for its strong native text rendering abilities, bilingual support (English and Chinese), fully integrated in-pixel text generation, and versatility across image styles. Technical benchmarks and image editing examples are shared in the comments, highlighting competitive performance in text rendering (rivaling GPT-4o in English, best-in-class in Chinese) and diverse image synthesis, but the linked image itself is not described explicitly in technical terms in the post or comments. The main significance lies in the model's technical advances rather than the specific image content.** Technical discussions in the comments highlight Qwen-Image's benchmark comparisons, especially for text rendering and image editing capabilities, with some skepticism or tongue-in-cheek remarks about its use cases, but otherwise focus on its strengths in bilingual generation and layout handling.
    - A [benchmark screenshot](https://preview.redd.it/a3o2wim001hf1.png?width=3036&format=png&auto=webp&s=fe8173646c7ea177041e2c110861a373b01356a6) is shared, suggesting Qwen-Image has competitive or leading performance compared to other models in multi-modal or image generation tasks, though exact numerical results would require analysis of the image itself for details.
    - Links are provided to the [blog announcement](https://qwenlm.github.io/blog/qwen-image/), [Hugging Face model card](https://huggingface.co/Qwen/Qwen-Image), [Model Scope summary](https://modelscope.cn/models/Qwen/Qwen-Image/summary), [GitHub repository](https://github.com/QwenLM/Qwen-Image), [technical report PDF](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/Qwen_Image.pdf), and two live demo endpoints ([Wavespeed](https://wavespeed.ai/models/wavespeed-ai/qwen-image/text-to-image) and [Modelscope](https://modelscope.cn/aigc/imageGeneration?tab=advanced)), facilitating deep technical exploration, model validation, and reproducibility.
    - A technical comparison is made regarding Qwen-Image achieving good *text generation* in its diffusion-based pipeline, which is notable since ChatGPT-4o's autoregressive model also featured strong text handling within generated images—indicating Qwen-Image may be addressing a commonly challenging aspect for diffusion models.
- [**Qwen-Image is out**](https://v.redd.it/4077mfg081hf1) ([Score: 431, Comments: 35](https://www.reddit.com/r/LocalLLaMA/comments/1mhiqqn/qwenimage_is_out/)): **Alibaba has released Qwen-Image, a new multimodal model announced on Twitter ([link](https://x.com/Alibaba_Qwen/status/1952398250121756992)). The post claims that Qwen-Image outperforms Flux Kontext and achieves performance near "GPT-image" level, implying technical parity or superiority in visual understanding and reasoning benchmarks. No in-depth model architecture, benchmarks, or open weights details are provided in the original post.** Discussion in the comments centers on appreciation for Alibaba releasing free assets and a willingness to adopt their API, but no specific technical debate or comparison details are present.
    - There is mention of a "20.B" parameter, which likely refers to the model's size—20 billion parameters—suggesting that Qwen-Image is a very large-scale multimodal (vision-language) model. This size puts Qwen in direct competition with leading models like GPT-4V and Gemini in terms of capacity, which is significant for researchers tracking the scaling trends and capabilities of open-source vision-language models.
    - A linked screenshot (https://preview.redd.it/p49ocex2p2hf1.png?width=1328&format=png&auto=webp&s=84f30442e738efa1c07f79ce4508e89baadad3fb) likely contains evidence of Qwen-Image's interface or demo outputs, which can provide technical readers with early insights into the feature set, output quality, and prompt handling capabilities of the model.
- [**Qwen image 20B is coming!**](https://www.reddit.com/r/LocalLLaMA/comments/1mhf0kl/qwen_image_20b_is_coming/) ([Score: 303, Comments: 60](https://www.reddit.com/r/LocalLLaMA/comments/1mhf0kl/qwen_image_20b_is_coming/)): **Alibaba's Qwen team is preparing to release Qwen Image 20B, a** `20B parameter` **diffusion image generation model, with imminent support being added to the Hugging Face** `diffusers` **library ([relevant PR](https://github.com/huggingface/diffusers/pull/12055)). This model will likely require** `40-44GB VRAM` **for FP16 inference, highlighting its resource intensiveness compared to LLMs that can more gracefully degrade under lower precision (FP8).** Commenters discuss the practical challenges of running such large vision models, including the lack of user-friendly software ecosystems akin to LM studio, as well as the steep VRAM requirements which make recent consumer GPUs like the RTX 5090 insufficient for FP16 inference at this scale.
    - A commenter highlights the high memory requirements for running a 20B diffusion image model, estimating 40-44GB of VRAM needed at FP16 precision. They note that, unlike LLMs, diffusion models' performance degrades significantly even when using lower-precision formats like FP8, stressing the increasing hardware barrier for local inference.
    - There is a technical discussion around the practical limitations for running large, open-weight models for various tasks (chat, code, image generation, OCR, RAG) in the U.S., noting that many high-performing models remain inaccessible due to commercial restrictions, in contrast to the open availability of models like Qwen3 235B in other regions.
- [**New Qwen Models Today!!!**](https://i.redd.it/qemmgysvuzgf1.png) ([Score: 677, Comments: 103](https://www.reddit.com/r/LocalLLaMA/comments/1mhbpmo/new_qwen_models_today/)): **The post announces the imminent release of new Qwen models, likely from Alibaba's Qwen series known for their open-source LLMs. Commenters are speculating about the potential models to be released, such as 'Qwen 3 VL' and a new 'Qwen 3 Coder 14B', and are particularly excited about the possibility of a new multimodal model (i.e., one supporting both text and vision) that could bolster open-source alternatives in this space. The image is likely a teaser or promo from the developers, indicating multiple models to be unveiled soon.** Comments reflect anticipation for an open-source multimodal model, with sentiment that more such models are needed in the ecosystem, and curiosity about coding-specialized models. There is also conjecture about the capabilities and specifications of the upcoming releases.
    - Multiple users express keen interest in the possible release of "Qwen 3 VL" and "Qwen 3 Coder 14b," indicating demand for both open-source multimodal models and larger code-specialized variants. The anticipation for Qwen3VL highlights a gap in freely available, high-performance models capable of handling both text and vision tasks, with comparisons implied to recent multimodal efforts like Llama-3 and open Flamingo.

### 2. Major Chinese LLM Releases: Pangu Ultra and Hunyuan

- [**Huawei released weights of Pangu Ultra,a 718B model.**](https://ai.gitcode.com/ascend-tribe/openpangu-ultra-moe-718b-model/blob/main/README_EN.md) ([Score: 269, Comments: 50](https://www.reddit.com/r/LocalLLaMA/comments/1mhctvk/huawei_released_weights_of_pangu_ultraa_718b_model/)): **Huawei has released the weights for Pangu Ultra, a** `718B`**parameter Mixture-of-Experts (MoE) model, notable for being trained entirely on Huawei Ascend NPUs, thereby not utilizing Nvidia hardware. The model is distributed under a custom license requiring attribution (e.g., "Powered by openPangu"), but is otherwise permissive; see their [license file](https://ai.gitcode.com/ascend-tribe/openpangu-ultra-moe-718b-model/blob/main/LICENSE) for specifics.** Commenters highlight the significance of a fully China-developed large model stack (hardware and software), suggesting technological independence from Nvidia/US restrictions, and note the massive parameter count as noteworthy. There is early discussion of unverified claims about the model's real-world performance and transparency, as referenced in external threads.
    - The Pangu Ultra 718B model is notable for being trained exclusively on Huawei's Ascend NPU hardware, without using any Nvidia GPUs. This makes it a fully Chinese-developed model both in terms of software and hardware stack, highlighting China's growing self-reliance in AI infrastructure.
    - The released weights use a custom license that, while relatively permissive, mandates attribution requirements such as including statements like "Powered by openPangu" and acknowledging "openPangu is a trademark of Huawei Technologies Co., Ltd." in derived products.
    - There are open questions about inference support and hosting, specifically whether usage is restricted to Ascend devices or if there are broader deployment options. Technical details regarding model compatibility with other hardware are not yet clarified in initial documentation.
- [**new Hunyuan Instruct 7B/4B/1.8B/0.5B models**](https://www.reddit.com/r/LocalLLaMA/comments/1mh3s7q/new_hunyuan_instruct_7b4b18b05b_models/) ([Score: 255, Comments: 51](https://www.reddit.com/r/LocalLLaMA/comments/1mh3s7q/new_hunyuan_instruct_7b4b18b05b_models/)): **Tencent released the Hunyuan-Instruct series of open-source language models in parameter sizes 0.5B, 1.8B, 4B, and 7B ([Hugging Face links](https://huggingface.co/tencent)), supporting both pre-trained and instruct-tuned variants as well as the GGUF format for llama.cpp compatibility. Key technical features include a native** `256K` **context window, Grouped Query Attention (GQA) for efficient inference, advanced quantization, and strong performance on agent benchmarks (BFCL-v3, τ-Bench, C3-Bench), with training inheritance from Hunyuan-A13B. This model family is positioned for scalability from edge to high-throughput environments, emphasizing efficient memory usage and deployment flexibility.** Comments highlight the value of small-scale models (0.5B-4B) for low-VRAM environments, the importance of verifying the claimed long-context capability, and comparisons to Qwen for diversity in small-model offerings.
    - Commenters highlight the technical significance of Hunyuan Instruct's release of multiple small LLM variants (7B, 4B, 1.8B, 0.5B), noting its direct competition with Qwen in serving users with limited VRAM. This makes the models particularly relevant for edge deployment, personal devices, or researchers with constrained hardware.
    - Attention is drawn to the importance of evaluating these models' performance specifically in long-context scenarios, as context length capabilities can sharply influence usability for tasks involving large input windows. The release of smaller models (such as 0.5B) is considered noteworthy for their potential in memory- and compute-constrained environments, emphasizing the demand for efficient, lightweight architectures.

### 3. Meta-Discussion and Memes: Qwen Model Drops and Community Reactions

- [**Sam Altman watching Qwen drop model after model**](https://i.redd.it/g7t8cmgrv0hf1.jpeg) ([Score: 607, Comments: 22](https://www.reddit.com/r/LocalLLaMA/comments/1mhgu6t/sam_altman_watching_qwen_drop_model_after_model/)): **The image is a meme referencing Sam Altman, CEO of OpenAI, 'watching' as Qwen (Alibaba's AI model) rapidly releases a series of new language models. Context from post title and technical comments indicate concerns about increased competition from Chinese models like Qwen and speculation about regulatory actions. A key technical concern raised is that impending model releases might introduce novel safety mechanisms potentially subject to patents and future regulation, which could restrict local, open models lacking such safety systems.** Commentary reflects apprehension about possible industry-wide moves to enforce proprietary model 'safety' features via regulation, with speculation that this could benefit large companies at the expense of openness and local deployment. There is also a discussion of geopolitical dynamics between US and Chinese AI model innovation.
    - A commenter speculates that upcoming Qwen models from Alibaba may introduce a new form of model-level 'safety' protection intended to be notably robust (potentially 'unbreakable' except with severe performance degradation). The discussion suggests this could be the basis for patents and subsequent lobbying to make such protections a legal requirement, which could limit the release of local or open-source models lacking similar mechanisms.
    - There is an undercurrent of discussion about the strategic implications of Chinese companies, especially Alibaba, accelerating foundation model releases. This could put pressure on Western firms and potentially drive policy debates around model safety, regulation, and international competition.
- [**r/LocalLLaMA right now**](https://i.redd.it/f0xr7mshc0hf1.png) ([Score: 510, Comments: 65](https://www.reddit.com/r/LocalLLaMA/comments/1mhe1rl/rlocalllama_right_now/)): **The image is a meme depicting the competitive landscape of open-source LLMs, specifically referencing OpenAI’s rumored open-source model and its struggle to remain relevant amidst strong releases from other organizations (like Meta and Chinese models). The discussion centers on skepticism about OpenAI's contribution to open-source, contrasting it with other more active organizations in the space.** Commenters debate whether OpenAI deserves recognition for an unreleased model, with some suggesting Meta (Zuckerberg/LLaMA) is more deserving due to tangible releases and others noting the general hype around Chinese LLMs. There's a general sentiment that praise should align with actually released, useful models.
    - There is a critique regarding the term “OpenAI’s open-source model,” highlighting that OpenAI hasn’t genuinely released a truly open-source model, and contrasting it with Meta (Zuckerberg) which has released models like LLaMA that have had substantial impact on the open-source community. The comment implies that technical praise should be reserved for organizations that have made concrete model and weight releases.
    - A discussion arises about the benchmarking and comparison of regionally-developed models, specifically mentioning 'Qwen' as representative of a broader slate of technically ambitious Chinese AI models. The suggestion is that 'Qwen' stands out not just individually, but as a symbol of increasing global competition in open LLM quality and innovation.
- [**Horizon Beta is OpenAI (Another Evidence)**](https://www.reddit.com/r/LocalLLaMA/comments/1mh2v1h/horizon_beta_is_openai_another_evidence/) ([Score: 269, Comments: 58](https://www.reddit.com/r/LocalLLaMA/comments/1mh2v1h/horizon_beta_is_openai_another_evidence/)): **The post presents evidence that the "Horizon Beta" model is based on OpenAI technology, indicated by its handling of the Chinese token '_天天啪' as a single token, a distinctive quirk of OpenAI's tokenizer (not observed in Anthropic, Google Gemini, or Qwen). Screenshots and testing confirm Horizon Beta's tokenization and translation failure on prompts containing 'ketøy', aligning its behavior with GPT-4o rather than competing models. This technique of identification by tokenizer behavior is based on previous evidence discussed in the LLaMA community linking similar tokenizer bugs to OpenAI-origin models.** Commenters note practical effectiveness and speed of the model, though some express indifference unless the model is open source or local, and another highlights impressive generation capabilities (e.g., high-quality game demos). There is speculation in the comments that it could be related to OpenAI's open-source (oss) efforts, but consensus points to its OpenAI origin.
    - Several commenters note that the Horizon Beta model appears optimized for creative writing rather than multimodal or code-heavy use, referencing public hints from Sam Altman suggesting a focus on story generation or text creativity. Comparisons are made between the '-beta' and '-alpha' versions, with observations that '-beta' enforces substantially more content filtering and censorship, indicating deliberate moderation adjustments during development.
    - Some users speculate whether Horizon Beta represents an open-source, local-LM release from OpenAI (possibly related to their recently open-sourced model), but other commenters push back, underscoring a distinction between online/proprietary deployments and the much-anticipated local, open models. There’s a consensus that unless a model can be run locally/offline, technical value for certain use cases is limited.
    - Performance reviews indicate the model is fast, delivers high-quality creative writing, and can generate impressive retro-style game assets (e.g., detailed sidescroller demos), but it's described as 'not super stunning great'—implying solid competence but not a breakthrough leap in generative capabilities.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Qwen-Image Model Release and Benchmarks

- [**Qwen-Image has been released**](https://huggingface.co/Qwen/Qwen-Image) ([Score: 407, Comments: 177](https://www.reddit.com/r/StableDiffusion/comments/1mhh7nr/qwenimage_has_been_released/)): **Alibaba has released Qwen-Image, an advanced vision-language model claimed to support tasks such as image editing, multimodal understanding, and image-to-text generation. Technical previews shared include interface screenshots demonstrating its editing capabilities, indicating functionality similar to models like Kontext. Qwen-Image appears to be part of the broader Qwen series of models, with anticipated future quantized versions for resource-constrained deployment.** Expert commentary notes excitement over image editing functions akin to Kontext and expresses high interest in upcoming quantized variants, signaling expectations around resource efficiency and wider accessibility.
    - Discussion touches on the model's size, with commenters noting the weight of Qwen-Image is reported as `40GB`, making it inaccessible for users with consumer GPUs under `48GB` VRAM (such as an RTX 3060 with 12GB VRAM). This highlights significant hardware requirements for on-premise inference.
    - A technical user expresses interest in quantized versions ("Can't wait for the Quants"), indicating the current release may be full-precision and less accessible, and that quantization could lower requirements, making it more usable on consumer-grade hardware.
    - Qwen-Image is compared to Kontext in terms of editing capabilities, suggesting multi-modal or image editing functionalities. This implicitly raises expectations about the feature set and technical parity with other advanced multi-modal models.
- [**Qwen image is coming!**](https://www.reddit.com/r/StableDiffusion/comments/1mhe9jb/qwen_image_is_coming/) ([Score: 141, Comments: 66](https://www.reddit.com/r/StableDiffusion/comments/1mhe9jb/qwen_image_is_coming/)): **Alibaba's Qwen team is preparing to release the Qwen Image 20B model, a** `20B` **parameter image-generation model, positioned as an image-centric counterpart to the highly regarded Wan video model. Early support for the model has already been integrated into HuggingFace's [Diffusers library](https://github.com/huggingface/diffusers/pull/12055), signaling imminent public availability. The Qwen image model release is anticipated to advance state-of-the-art (SOTA) vision capabilities as prior releases from Qwen have for LLMs and multimodal models.** Commenters emphasize the rapid progression and scaling in open-source vision models, noting Qwen's frequent SOTA achievements, the importance of hardware accessibility for large models (notably 20B parameter scale and GGUF quantization), and expressing anticipation for increased VRAM availability at competitive prices from Chinese manufacturers.
    - The new Qwen image model has already received support in the Hugging Face Diffusers library ([commit link](https://github.com/huggingface/diffusers/pull/12055)), indicating imminent release and suggesting rapid integration into existing generative image model pipelines.
    - Technical discussion compares anticipated model parameter sizes: Qwen's upcoming image model is expected to be larger than the Hidream (17B parameters) model, and may reach '20 billion params' with expectations of GGUF quantization, supporting efficient local use.
    - Discussion acknowledges Qwen's consistent state-of-the-art (SOTA) performance in both LLMs and vision models, specifically mentioning the impressive abilities of Wan (video model) and VLMs, with community anticipation for competitive VRAM utilization given the increasing model scale.
- [**Qwen Image is even better than Flux Kontext Pro in Image editing.**](https://www.reddit.com/gallery/1mhikh2) ([Score: 244, Comments: 51](https://www.reddit.com/r/StableDiffusion/comments/1mhikh2/qwen_image_is_even_better_than_flux_kontext_pro/)): **Alibaba's Qwen-Image model demonstrates state-of-the-art performance in both image generation and editing tasks, surpassing competing systems like Flux Kontext Pro and other open/closed models per benchmarks summarized on their official blog (https://qwenlm.github.io/blog/qwen-image/). However, the advanced editing model has not yet been publicly released, with Alibaba noting potential future availability. The model's operational requirements are high—current versions reportedly require significant GPU memory (80GB+), making local/private experimentation with unquantized models currently impractical for most researchers.** Commenters highlight concerns about accessibility due to high resource requirements and expect community-driven quantized versions soon. There is skepticism about model generalization and real-world fidelity—especially post-quantization—and some note the characteristic 'cartoonized' output style.
    - Qwen Image's currently unreleased model draws attention for its image editing capability, but there is skepticism over accessibility: users note that if it requires an 80GB GPU, running it locally is infeasible for most; however, there are expectations that quantized versions (smaller, lower memory requirement models) will be available soon, potentially increasing usability.
    - In direct visual comparisons, technical users acknowledge that while Qwen Image shows impressive results, especially for an open-source offering, models like Imagen 4 Ultra are still perceived as superior for photorealism, particularly in challenging compositional details (e.g., shallow depth of field, lighting, bokeh effects, and cinematic grading).
    - There is some caution among technical users regarding the common issue with open models—quantization usually leads to reduced performance, and "cartoonized” outputs tend to be exaggerated. Nonetheless, the growing quality of open-source competitors is seen as a positive development, driving rapid iteration and improvement in the field.
- [**Warning: pickle virus detected in recent Qwen-Image NF4**](https://www.reddit.com/r/StableDiffusion/comments/1mhkmsa/warning_pickle_virus_detected_in_recent_qwenimage/) ([Score: 146, Comments: 76](https://www.reddit.com/r/StableDiffusion/comments/1mhkmsa/warning_pickle_virus_detected_in_recent_qwenimage/)): **A HuggingFace model repository ([lrzjason/qwen_image_nf4](https://huggingface.co/lrzjason/qwen_image_nf4)) was flagged for containing a potential 'pickle virus', prompting a warning to avoid downloading. Although the repo had released several models previously, the alleged file in question was a** `.safetensors` **file (which is designed to avoid the code execution vulnerabilities of Pickle serialization); the repository has since been taken down.** Top comments raise skepticism, noting that `.safetensors` files are supposed to be safe from Pickle exploits and suggesting the issue might be a mislabeling or false positive. Some also point out that the uploader has a history of prior, seemingly reputable model releases, advocating caution but not definitive alarm.
    - There's a discussion about the relative safety of `.safetensors` files versus `.pkl` (pickle) model files: `.safetensors` format was designed to mitigate risks inherent in pickle serialization, specifically arbitrary code execution, making `.safetensors` considered inherently safer for model distribution.
    - One comment points out that the flagged file is incorrectly identified as a pickle-based virus: it's actually a `.safetensors`, which does not support the Python pickle mechanism and thus can't execute arbitrary Python code. This emphasizes the importance of distinguishing between file formats for security concerns.
    - Another technically relevant point is the user history on HuggingFace: the uploader has an established track record, suggesting the likelihood of a false positive. Still, users are advised to exercise caution until the file's safety is confirmed.

### 2. Claude 4.1 & Opus Next-Gen Model Launch Hype

- [**Looks like Claude 4.1 Opus is also coming soon**](https://i.redd.it/a2kwuoauf0hf1.png) ([Score: 266, Comments: 58](https://www.reddit.com/r/singularity/comments/1mhehrb/looks_like_claude_41_opus_is_also_coming_soon/)): **The image appears to show a pre-release or announcement hint that Claude 4.1 Opus, likely an updated version of Anthropic's flagship LLM (Large Language Model), is coming soon. Commenters speculate on version access limits ("4.1 messages before hitting the weekly limit"), hope for Sonnet 4.1 counterpart, and note recent Anthropic user experience surveys and A/B testing, indicating broader platform and model updates.** The discussion centers on anticipated model improvements and changes to access/pricing, with some users criticizing the frequency of usage limits and others focusing on recent user experience research as evidence of significant changes ahead.
    - A user notes that Anthropic has begun requesting reviews from users engaging with Claude Code, suggesting that A/B testing on new features or user experience changes is underway, which could be indicative of upcoming improvements or adjustments for future Claude 4.1 releases.
    - Concerns are mentioned regarding the Opus tier of Claude, noting that it is perceived as both 'way too censored and expensive', reflecting ongoing community debate over Anthropic's pricing and content-moderation policies for premium models.
    - There is anticipation for Claude Sonnet 4.1, indicating user interest in feature or performance parity across different Claude model variants, not just the flagship Opus.
- [**Opus 4.1 on the way?**](https://i.redd.it/a2kwuoauf0hf1.png) ([Score: 162, Comments: 51](https://www.reddit.com/r/ClaudeAI/comments/1mhf8mo/opus_41_on_the_way/)): **The post speculates about the imminent release of OpenAI's rumored "Opus 4.1" model, as hinted by the image (which appears to show backend or dashboard evidence), and contextualizes timing around potential model drops—possibly to pre-empt the hype around a "GPT-5". Commenters are discussing expectations of new model limits or pricing (comparing to Sonnet 4), and correlating performance complaints with model retraining periods, suggesting possible resource constraints due to heavy GPU utilization ahead of model launches. [See image here.](https://i.redd.it/a2kwuoauf0hf1.png)** Commenters broadly agree that new model launches often coincide with observed dips in performance and speculate this is due to GPU resources being dedicated to training. There's also uncertainty about whether Opus 4.1 will introduce better access limits, which would affect practical use.
    - A technical hypothesis suggests that the upcoming "Opus 4.1" may utilize a new checkpoint of Opus 4, then undergo pruning to be more efficient—potentially achieving better performance at approximately 60% of the original model's size. This would allow Anthropic to maintain pricing while reducing compute costs in response to increased context-window (CC) demand.
    - One user notes a perceptible decline in present model quality, reporting that performance feels regressed to the level of "3.7", reflecting ongoing fluctuations possibly caused by major training activities or upcoming model refreshes.
    - There is speculative discussion on whether Opus 4.1 will inherit Sonnet 4's usage limits, implying that technical and infrastructural constraints (such as request throttling or quota enforcement) could limit practical improvements for end users regardless of model advances.
- [**This has to be one of the craziest one shots I've seen - Claude Opus 4**](https://v.redd.it/i9aehe15axgf1) ([Score: 146, Comments: 20](https://www.reddit.com/r/ClaudeAI/comments/1mh2ybk/this_has_to_be_one_of_the_craziest_one_shots_ive/)): **A user claims that Claude Opus 4 generated, in a single inference ("one shot"), a self-contained single-page HTML drone simulator using ThreeJS per the following prompt: 'Create an autonomous drone simulator (drone flies by itself, isometric god like view, optionally interactive. With a custom environment (optionally creative), using ThreeJS, output a single-page self-contained HTML.' The technical implication is that Opus 4 can write substantial and non-trivial application logic, integrating 3D rendering and basic simulation, all in one contiguous output.** Top comments express skepticism regarding reproducibility (asking for the prompt), suggest similar challenging tasks (e.g., zombie outbreak simulator), and note surprise at the model's apparent output quality. There is implicit debate on whether this is generally achievable or an outlier prompt/model interaction.
    - A commenter suggests using horizon-beta as an alternative, claiming it not only one-shot the same prompt but also generated more advanced features such as multiple camera modes, environmental effects (water, wind textures), propeller animations, mission system, dynamic weather, and even contoured terrain, implying a significant difference in output complexity and completeness compared to the referenced Claude Opus 4 result.
    - One commenter requests specific details about the prompt used to achieve the result, expressing skepticism about being able to replicate such an advanced one-shot outcome, which highlights ongoing concerns about reproducibility and prompt engineering's impact on large model outputs.
    - Another technical inquiry relates to output latency, with a user asking about the time it took for the Claude Opus 4 one-shot to complete, reflecting interest not just in content generation quality, but inference speed and throughput for complex prompts.

### 3. Upcoming GPT-5 Release Signals and OpenAI Announcements

- [**Looks like Thursday will be the day for GPT-5 (at least according to Jimmy, who's been reliable)**](https://i.redd.it/2d0vdv8spzgf1.png) ([Score: 185, Comments: 66](https://www.reddit.com/r/singularity/comments/1mhb58g/looks_like_thursday_will_be_the_day_for_gpt5_at/)): **The post's image (https://i.redd.it/2d0vdv8spzgf1.png) appears to be a screenshot or visual reference suggesting GPT-5's release date may be Thursday, attributed to a leaker nicknamed 'Jimmy.' The discussion in the title and top comments references this individual's prior reliability in predicting OpenAI events, but notes previous inaccuracies such as claims about AGI. The context also mentions alignment with circulating rumors and recent hints from Sam Altman showing off GPT-5, supporting speculation about an imminent release.** Commenters question the track record of 'Jimmy' as a reliable source, and express skepticism about rumored release dates as they keep shifting (e.g., "First Monday now Thursday?"). There is also a tongue-in-cheek remark downplaying expectations for the demo's utility.
    - Discussion centers on the credibility of Jimmy's leaks, referencing his past claim that OpenAI achieved AGI internally in 2023, which has not been substantiated. Users note that recent speculation around a Thursday GPT-5 release aligns with separate rumors and Sam Altman’s public hints on Twitter, giving current rumors slightly more weight compared to previous, less substantiated claims.
- [**OpenAI VP of ChatGPT: "Big Week Ahead"**](https://i.redd.it/bja36ao2w0hf1.jpeg) ([Score: 226, Comments: 30](https://www.reddit.com/r/OpenAI/comments/1mhgwwg/openai_vp_of_chatgpt_big_week_ahead/)): **The post references a statement from the OpenAI VP of ChatGPT indicating an upcoming significant announcement or event, captioned as 'Big Week Ahead.' The image itself is not described in detail due to failed analysis, but context suggests it is likely promotional or teaser content tied to anticipated updates or releases for ChatGPT. Discussion points in the comments highlight large-scale usage metrics (over a billion users monthly for generative AI platforms like ChatGPT and Gemini) and the marketing tactics used to generate hype in the AI community.** Commenters debate the effectiveness and sincerity of OpenAI's marketing strategy, with skepticism regarding the buildup of hype versus the substance of forthcoming announcements, and some contrasting this with the broad adoption and real-world usage figures for ChatGPT and Gemini.
    - A user notes the contrast between perceived public skepticism about AI and actual usage, referencing data that ChatGPT and Gemini collectively serve "well over a billion people" on a weekly or monthly basis. This highlights the significant adoption rates of these AI tools despite vocal criticism or doubts about mainstream use.
- [**GPT-5 Easter Egg**](https://www.reddit.com/r/singularity/comments/1mhkahr/gpt5_easter_egg/) ([Score: 125, Comments: 71](https://www.reddit.com/r/singularity/comments/1mhkahr/gpt5_easter_egg/)): **A Reddit user points out a potential Easter egg: OpenAI employee Boris Power tweeted about GPT-5 at precisely 8:05 am PDT, speculating that 8/5 (August 5th) is the intended release date for GPT-5. The thread references the 1440 minutes in a day to highlight the intentionality of the timing (see the [tweet](https://x.com/BorisMPower/status/1952385313546146238)), but provides no direct technical detail or confirmation regarding GPT-5 capabilities, architecture, or benchmarks.** Technically-minded commenters are skeptical, challenging the statistical significance and suggesting that correlation here is weak, and instead recommend leveraging such speculation in prediction markets rather than treating it as meaningful evidence.
    - One commenter notes that major model releases such as GPT-5 are typically announced in advance via official events rather than revealed suddenly or through hidden signals, implying that significant launches are accompanied by substantial marketing and communication efforts by organizations like OpenAI. This is supported by historical patterns of prior major AI releases.
    - Another technically-minded reply challenges the interpretation of numeric coincidences (1440 minutes in a day) as signals for a release date and points out that given a range of plausible dates (8/5-8/31), pattern matching in date numbers can be misleading and is not a statistically rigorous prediction method.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp
> 

**Theme 1. The New Model Frontier: GPT-5 Hype, Horizon Alpha's Debut, and Qwen's Rise**

- [**Engineers Brace for Impending GPT-5 Drop**](https://discord.com/channels/974519864045756446/998381918976479273/1400554785911865466): Speculation is rampant that **GPT-5's** release is imminent, with some calling it a *"panic drop"* due to scaling limitations and diminishing returns from **Chain of Thought (CoT)**, which some developers call a *"complete dead end"*. While rumors from the **OpenAI** discord suggest a delay due to an inability to surpass **Grok4**, others report a brief sighting of **GPT-5** in the API ([source](https://x.com/chetaslua/status/1951301385292493259)) and anticipate a unified, omnimodal model that [combines multiple products](https://link.to/openais-next-foundational-model).
- [**Mysterious Horizon Alpha Model Outperforms Paid LLMs**](https://discord.com/channels/1091220969173028894/1094454198688546826/1400553644956323921): A new model, **Horizon Alpha**, is impressing developers by outperforming paid LLMs on **OpenRouter**, delivering [perfect one-shot code in custom programming languages](https://openrouter.ai/) and demonstrating superior shell use. Speculation on **Nous Research AI** suggests it could be a **120B MoE** or **20B** model from **OpenAI**, based on a [tweet](https://x.com/apples_jimmy/status/1951180954208444758) and [Reddit thread](https://www.rxddit.com/r/LocalLLaMA/comments/1mepeqh/the_openai_open_weight_model_might_be_120b/), while others believe it *could always be something turbo weird we’re not thinking of like codex-2*.
- [**Qwen3-Coder Smashes Speed Records and Poses Challenges**](https://discord.com/channels/1027685395649015980/1027688115592237117/1400899899402489856): The new **Qwen3-Coder** model is now available in **Windsurf** at a blazing **2000 tokens/sec**, hosted on US servers. Meanwhile, developers in the **Unsloth AI** discord are debating the best quantization methods, with **Q4_K_M gguf** being slow in **Ollama**, while others run the **Qwen3-30b-a3b** model at **40k** context in **vllm** on a **3090**; however, users in **LM Studio** are hitting a *Cannot read properties of null (reading '1')* error when loading the model.

**Theme 2. Technical Trenches: Quantization Woes, RAG Debates, and API Spats**

- [**Quantization Puzzles Plague New Models**](https://discord.com/channels/1053877538025386074/1149866623109439599/1400597993442513128): Engineers are grappling with quantization for new models, as unexpected refusals in the **Hermes-3 dataset** complicated *imatrix* computation, prompting a [deeper investigation of the dataset](https://huggingface.co/datasets/NousResearch/Hermes-3). The leaked config for a rumored **OpenAI MoE** model indicates its hidden size of **2880** will make it impossible to quantize using K or I quants, a limitation also seen in **SmolLM** and **Qwen2.5**.
- [**Developers Question Large Context Windows, Refine RAG**](https://discord.com/channels/974519864045756446/998381918976479273/1400554785911865466): A debate is brewing over whether [large context windows are overrated](https://nealgoogs.website/), with some finding **Claude** and **ChatGPT** better for legacy codebases than **Gemini's** 1M window, while others argue large context is crucial for agentic applications to *remember and weave in far‑back details automatically*. To improve retrieval, engineers in the **Yannick Kilcher** discord are using [query expansion techniques](https://www.promptingguide.ai/techniques/query_expansion) in **RAG** systems by generating multiple questions from a single query.
- [**Anthropic Blocks OpenAI's Access to Claude API**](https://discord.com/channels/822583790773862470/1075282825051385876/1400565376567480373): In a significant competitive move, **Anthropic** revoked **OpenAI's** API access to its models, including **Claude**, citing a violation of its terms of service. **OpenAI** expressed disappointment, highlighting that its API remains open to **Anthropic**, sparking community discussion about the escalating rivalry and the blurring lines of model training.

**Theme 3. Rise of the Agents and Specialized Tools**

- [**Coding Agents Battle for Developer Dominance**](https://discord.com/channels/1131200896827654144/1131200896827654149/1400559583528747048): **Aider** continues to win praise for its effectiveness, with one user claiming it completed *"one week of programming work in a single day for just $2"* using **DeepSeek**. In the funding arena, the open-source AI coding agent **Cline** secured **$32 million** in a funding round led by **Emergence Capital** and **Pace Capital**, aiming to serve its **2.7 million** developers with transparent tools.
- [**Embodied AI and Creative Tools Push Boundaries**](https://discord.com/channels/1053877538025386074/1149866623109439599/1400597993442513128): The **Unitree R1 foundational robot model**, priced at **$5,900** and featuring an open SDK, is democratizing embodied AI development, as shown in [this YouTube video](https://www.youtube.com/watch?v=ljo7TjOqRzs). In creative AI, the [**TheCinema AI**](https://thecinema.ai/) research project is tackling the challenge of generating cohesive movie scenes, detailed in its [arxiv paper](https://arxiv.org/html/2507.18634v1), while a developer training a **VITS** model discovered it can learn subtleties like breathing at commas.
- [**New Protocols and Frameworks Expand the Ecosystem**](https://discord.com/channels/1312302100125843476/1312302100125843479/1400559945786589275): A new payment layer for the Model Context Protocol, **PayMCP**, is in development with [Python](https://github.com/blustAI/paymcp) and [TypeScript](https://github.com/blustAI/paymcp-ts) implementations available for early adopters. In the GPU space, progress on the [picocuda](https://github.com/j4orz/picocuda) compiler is advancing, with plans to follow the [GPUCC paper](https://dl.acm.org/doi/pdf/10.1145/2854038.2854041) to build up to compiling **Karpathy's llm.c**.

**Theme 4. Clash of the Titans: Gemini's Stumbles and Kimi's Surge**

- [**Google's "Deepthink" Plan Draws Scorn for High Price, Low Limits**](https://discord.com/channels/1340554757349179412/1340554757827461211/1400554342616010903): The release of **Gemini 2.5 Deepthink** for Ultra members was met with ridicule across Discords for its **$250/month** price tag and a **10 queries-per-day limit**. Members in **LMArena** and **Moonshot AI** called it a *"scam"* and *"very funny and very scummy"*, viewing it as a rushed release ahead of GPT-5.
- [**Kimi K2 Turbo Launches with 4x Speed and a Discount**](https://discord.com/channels/1369594130807787570/1371757097246785536/1400719197838770217): **Moonshot AI** announced **Kimi K2 Turbo**, a faster version of its model that boasts a **4x speed increase** to **40 tokens/sec**, available via the official API at [platform.moonshot.ai](http://platform.moonshot.ai/). The new model comes with a **50% discount** on tokens until September 1st, positioning **Kimi K2** as the first model a user felt could replace **Claude**, leading them to drop **Gemini 2.5 Pro**.
- [**Gemini Models Exhibit Glitches and Biases**](https://discord.com/channels/1340554757349179412/1340554757827461211/1400554342616010903): Users in **LMArena** noted that **Gemini** models were exhibiting repetitive, glitchy behavior. More systematically, developers in the **Yannick Kilcher** and **Eleuther** Discords observed that **Gemini-2.5-flash** consistently ranked **Gemma** models higher than others, sparking concerns about a potential *family bias* in its evaluation capabilities.

**Theme 5. User Experience Battleground: Freezing Bugs, API Errors, and Freebie Deals**

- [**Cursor Freezing Bug and Pricing Debates Frustrate Users**](https://discord.com/channels/1074847526655643750/1074847527708393565/1400555426764034119): A persistent **Cursor freezing bug** is causing machines to freeze every **30-60 seconds** after an hour of chat use, with the team directing users to report on the [Cursor forum](https://forum.cursor.com/c/bug-report/6). This is compounded by debates over escalating costs, with one user reporting spending **$600 in 3 months**, while another prefers **Claude's $200 plan**.
- [**Platform APIs Suffer from Errors and Outages**](https://discord.com/channels/1091220969173028894/1094454198688546826/1400553644956323921): **OpenRouter** users are plagued by API issues, including internal errors, timeouts, and empty responses from the **Deepseek v3 free model**, which is described as *"completely overloaded"*. At the same time, **Perplexity Pro** subscribers on **iOS** report that image generation fails to incorporate attached images, and the `search_domain_filter` is not functioning correctly.
- [**Airtel Subscribers in India Score Free Perplexity Pro**](https://discord.com/channels/1047197230748151888/1047649527299055688/1400553929611280499): In a major promotional deal, **Airtel** subscribers in India, a user base of over **300 million people**, are receiving **Perplexity Pro** for free for **12 months**. This promotion is exclusive to Airtel subscribers located within India, offering a significant user acquisition channel for the platform.

---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Comet Browser Invites Trickle Out**: **Perplexity** is slowly distributing **Comet Browser** invites, prioritizing **Pro users**.
   - Users report varied wait times, suggesting Pro users can share up to **2 invites** to speed up the process.
- **Perplexity Pro Image Generation Fails on iOS**: Users are reporting that **Perplexity Pro on iOS** fails to incorporate attached images during image generation, creating recurring issues.
   - The model summarizes requests without generating images from the attachments, even after starting new chats.
- **Airtel India Subscribers Score Free Perplexity Pro**: **Airtel** subscribers in India (over **300 million people**) are receiving **Perplexity Pro** for free for **12 months**.
   - The promotion is exclusive to Airtel subscribers located in India.
- **GPT-5 Release Date: Still Shrouded in Mystery**: Speculation surrounds the release of **GPT-5**, with conflicting views on whether it will be a full release or a smaller, more focused model.
   - One user claimed to have briefly seen **GPT-5 in the API** ([source](https://x.com/chetaslua/status/1951301385292493259)), but it was quickly removed, fueling further speculation.
- **Search Domain Filter Confounded**: A **Perplexity Pro** subscriber reported that the **search_domain_filter** is not functioning as expected despite the feature not being in beta.
   - Another member requested a copy of the user's request for further investigation and assistance.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **GPT-5: Panic Drop or Polite Improvement?**: Members are speculating if **GPT-5** will be a *panic drop* due to **OpenAI's** limitations in scaling, along with diminishing returns from **Chain of Thought (CoT)**.
   - Claims suggest **CoT** is a *complete dead end*, proposing direct network feedback of the model's vector output instead of using tokens for thinking.
- **Qwen3 tests quantization limits**: Discussions revolve around the best quantization for **Qwen3 Coder 30B**, with reports on **Q4_K_M gguf** being slow in **Ollama**, while others prefer **UD q3 XL** for VRAM savings.
   - One member runs the April **Qwen3-30b-a3b** model at **40k** in **vllm** on a **3090** 24/7, awaiting a 4-bit AWQ version for the coder model.
- **Unsloth now supports GSPO**: After Qwen proposed **GSPO** as an update to **GRPO**, members clarified that **GSPO** already works in **Unsloth** and it is a wrapper that will auto-support **TRL** updates.
   - Although **GSPO** is slightly more efficient, members did not note any significant updates to performance.
- **VITS Learns to Breathe**: A member training a **VITS checkpoint overnight** shared that **model quality depends on epochs and dataset quality**, and **VITS excels at speaker disentanglement**.
   - Furthermore, they discovered **VITS encodes raw audio into latent space** for realistic recreation and can learn subtleties like breathing at commas with annotation and ran into memory issues on iOS.
- **Dynamic Quantization gets Quant Clone**: A member created [a small application](https://github.com/electroglyph/quant_clone) to quantize finetunes the same way as Unsloth's dynamic quantization, wanting to replicate it on their own finetunes.
   - A user reported high refusal rates in their **Gemini** finetunes, and found *Gemini to be quite obnoxious* in that regard.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Arena Enhancements Aim to Assist**: Members suggested adding buttons for **Search, Image, Video, and Webdev Arena** to boost visibility, and also suggested adding tooltips to the leaderboard explaining how **Rank, CI, and Elo** are determined, sharing a [concept image](https://cdn.discordapp.com/attachments/1340554757827461211/1400554342167089222/uzzzSHh.png).
   - The goal is to assist users in navigating the platforms and understand ranking metrics.
- **Data Concerns: Personal Info Peril**: A user raised concerns about accidentally including **personal information** in published prompts and asked for ability to remove prompts.
   - A member responded that such examples should be DM'd to them for escalation, and acknowledged [sharing these concerns with the team](https://www.deepcogito.com/research/cogito-v2-preview).
- **Gemini's Generation Gets Glitchy**: Some members noted **Gemini** exhibited repetitive behavior, while another questioned if **Gemini 2.5 Flash** fixed the issue and one user noted video limits dropping from **10 to 8**, urging others to use the video generation arena quickly.
   - The community's sentiment is split between experiencing glitches and consistent performance.
- **DeepThink Debut Disappoints?**: With the release of **Gemini 2.5 Deepthink** for Ultra members, members are wondering if it is worth it after seeing **10 RPD limit**.
   - Members called it a **scam** and a daylight robbery, saying it's just a rushed version because of the imminent **GPT-5** release.
- **Veo 3 Visuals Victory**: **Veo 3 Fast & Veo 3** are out with new **Image-to-Video with audio capabilities** within the [Video Arena](https://discord.com/channels/your_server_id/1397655695150682194).
   - The community can now create videos from images using the new `/image-to-video` command in the video-arena channels, with voting open for the best videos.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Vibe Coding Sparks GitHub Needs**: A member inquired about the necessity of **GitHub** for background agents, exclaiming *this thing is sick* alongside an attached image, sparking curiosity about **vibe coding** setups.
   - Another user, having spent **$40** on prompts, sought advice on optimizing their **Cursor** setup, reflecting a common interest in efficient configuration.
- **Cursor Freezing Bug Creates Frustration**: A user reported frequent machine freezes every **30-60 seconds** after an hour of chat use, indicating a persistent **Cursor freezing bug**.
   - A **Cursor** team member recommended posting the issue on the [Cursor forum](https://forum.cursor.com/c/bug-report/6), highlighting the official channels for bug reporting and assistance.
- **Model Spending Compared to Claude Pro**: Users debated the pricing of **Cursor** versus **Claude Pro**, with one stating their preference for the cheapest plans and best models, favoring Claude's **$200** plan.
   - Another user cautioned about escalating costs, reporting spending **$600** in 3 months, emphasizing the need for cost management.
- **Horizon Alpha Experience Divides Users**: One user described their personal experience with **Horizon-Alpha** as *a bit underwhelming*, suggesting mixed reactions to the new feature.
   - Conversely, another user lauded *cursor is the best app i have ever seen*, underscoring the subjective nature of user experiences.
- **Referral Program Requested for Cursor**: Members have inquired about a referral program for **Cursor**, with one user claiming to have onboarded *at least 200+ people by now sitting in discords lmao*, indicating significant community-driven adoption.
   - A link to the [Cursor Ambassador program](https://cursor.com/ambassador) was shared, providing an alternative avenue for rewarding community contributions.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Function Calling APIs Trump XML Workarounds**: Function Calling APIs have **inherent value** over structured XMLs, which are often used as a workaround when models like **Qwen** don't support native tool calling.
   - Inline tool calls maximize interoperability for coding models like **Qwen**, even with minor inefficiencies.
- **Zuckerberg's AI Sparks Bio-Weapon Concerns**: **Mark Zuckerberg's** AI superintelligence initiative raised concerns about potential bio-weapon creation, and one member warned against releasing superintelligence to the public.
   - Members also expressed concern that *controlling minds with fake users and carefully crafted language* could be more dangerous than bio-weapons.
- **GPT-5 Faces Delay, Grok4 Takes the Crown?**: Rumors suggest **GPT-5's** delay is due to an inability to surpass **Grok4**, but [OpenAI plans to combine multiple products into GPT-5](https://link.to/openais-next-foundational-model).
   - Clarification was given that **GPT-5** will be a single, unified, omnimodal model.
- **Horizon Alpha Outshines Paid LLMs**: **Horizon Alpha** is outperforming paid LLMs via the OpenRouter API, delivering [perfect one-shot code in custom programming languages](https://openrouter.ai/).
   - Its shell use and task list creation in orchestrator mode are superior to other models, though some speculate it *could always be something turbo weird we’re not thinking of like codex-2*.
- **Large Context Windows Spark Debate**: Despite **Gemini's** 1 million context window, legacy codebase issues were better solved with **Claude** and **ChatGPT**, sparking debate on whether [large context windows are overrated](https://nealgoogs.website).
   - Some prefer models with smaller context windows and better output, while others insist larger windows are crucial for agentic applications to *remember and weave in far‑back details automatically*.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Image-to-Video Prompt Generation Dreams in LM Studio**: Members are anticipating future **image-to-video prompt generation** and **image attachment** features in **LM Studio**, favoring offline capabilities over cloud-based alternatives like **ChatGPT**.
   - As an alternative, one member mentioned **ComfyUI**, noting it might not be optimized for **AMD** cards.
- **LM Studio's Roadmap: A Mystery**: The community discussed the absence of a **public roadmap** for **LM Studio**, with speculation that development plans might be unstructured and unpredictable.
   - A member stated, *no public roadmap so noone knows*.
- **LM Studio API Security Considerations**: Users debated connecting to the **LM Studio API** across a network, highlighting potential security vulnerabilities.
   - Concerns were raised about **LM Studio's** unverified security, cautioning against exposing it without proper risk assessment and network protection.
- **Qwen3 Coder Model Faces Loading Glitches**: Users encountered difficulties when loading the **Qwen3 Coder 30B** model, triggering a *Cannot read properties of null (reading '1')* error.
   - A fellow member suggested an update to version **0.3.21 b2** which claims to have resolved the issue, along with enabling **recommended settings**.
- **Nvidia Bursts Out a Driver**: **Nvidia** released driver **580.88** quickly after **577.00**, a **9-day-old driver** with a fix for a potential issue with GPU video memory speed after enabling **NVIDIA Smooth Motion** [5370796].
   - The user runs the drivers from the cuda toolkit, and doesn't use the fancy control panel or GFE (GeForce Experience).



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **API Errors Plague OpenRouter**: Users reported experiencing **API errors** when using models via the **OpenRouter API**, with one user suggesting checking the **model ID prefix** and **base URL** to resolve the issue.
   - Errors include *no endpoint found* which members suggested was caused by potential misconfiguration.
- **Deepseek v3 Free Model Plagued by Outages**: Users experienced issues with the **Deepseek v3 0324 free** model, including *internal errors*, *empty responses*, and **timeouts**, leading some to switch to the paid version.
   - One member pointed out *free is completely overloaded. paid has none of these issue, and the actual content quality is better.*
- **Horizon Alpha Hailed as Effective**: Users praised the **Horizon Alpha** model for its effective reasoning and good performance.
   - While the model claimed it was developed by **OpenAI**, community members clarified that it was likely a distilled model.
- **Personality.gg Leverages OpenRouter for Roleplay**: [Personality.gg](https://personality.gg) launched a roleplay site using **OpenRouter** for most models, providing access to all 400 models through **OpenRouter PKCE** completely free/cheap.
   - This integration lets users engage in role-playing scenarios with a wide variety of **AI models**.
- **PyrenzAI's UX Wins Praise**: A user complimented the **UI/UX** of [PyrenzAI](https://pyrenzai.com), appreciating its unique look and style, and distinctive sidebar design compared to other apps.
   - Despite speed and security critiques, the application's user interface received positive feedback.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi K2 Goes Ludicrous Speed with Turbo!**: The Moonshot team announced **Kimi K2 Turbo**, touting **4x the speed** at **40 tokens/sec**, with a **50% discount** on input and output tokens until **Sept 1** at [platform.moonshot.ai](https://platform.moonshot.ai/).
   - Users can now experience significantly faster performance thanks to faster hosting of the same model, available via official API.
- **Moonshot AI Launches New Hangout Spot**: Moonshot AI launched the ***Moonshot AI Forum*** ([https://forum.moonshot.ai/](https://forum.moonshot.ai/)) for technical discussions, API help, model behavior, debugging, and dev tips.
   - While *Discord’s still vibin for memes* and chill convos, the forum aims to be the go-to spot for serious builds and tech discussions.
- **Kimi K2 Challenges Claude's Reign**: One user reported **Kimi K2** as the first model they can use instead of **Claude**, prompting them to drop **Gemini 2.5 Pro** due to coding, as a kind of information, becoming freer.
   - The user also added that they expect most AIs will converge in terms of knowledge, so the differences between them will start to blur.
- **Kimi K2 Turbo Pricing Details Exposed**: The speedy **Kimi K2 Turbo** is priced at **$0.30/1M** input tokens (cached), **$1.20/1M** input tokens (non-cached), and **$5.00/1M** output tokens with a special promo until Sept 1.
   - This equates to roughly *4x faster for 2x the price* during the discount, tailored for users requiring swift processing.
- **Gemini Ultra's Deep Thinking Costs a Pretty Penny**: Members ridiculed Google Gemini Ultra's plan imposing a **10 queries a day limit for $250/month**, with one user saying it was *very funny and very scummy*.
   - Comparisons were made to **ChatGPT pro** at $200/month which gives unlimited **Office 365 Pro**, and **Claude Max**, seen as more reasonably priced.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes-3 Dataset Refusals Raise Eyebrows**: Members investigated unexpected refusals in the **Hermes-3 dataset** while computing the *imatrix* for quantization, leading to [further dataset investigation](https://huggingface.co/datasets/NousResearch/Hermes-3) to confirm the dataset is devoid of refusals.
   - The team is hoping to confirm that the dataset is devoid of refusals by ensuring the dataset is fully vetted.
- **Unitree's R1 Robot Democratizes Embodied A.I.**: The community explored the **Unitree R1 foundational robot model**, priced at **$5,900**, providing a fully open software development kit (**Python**, **C++**, or **ROS**) for A.I. development, showcased in [this YouTube video](https://www.youtube.com/watch?v=ljo7TjOqRzs).
   - Users stated it is an ideal tool for research teams transitioning to the next evolution of A.I.
- **Horizon Alpha Model Sparks OpenAI Speculation**: Members debated whether the **OpenAI Horizon Alpha model** resembles **OpenAI's** style, speculating it could be a **120B MoE** model with low activation or possibly a **20B** model, noted in [this tweet](https://x.com/apples_jimmy/status/1951180954208444758).
   - Some suggested on [this Reddit thread](https://www.rxddit.com/r/LocalLLaMA/comments/1mepeqh/the_openai_open_weight_model_might_be_120b/) that quantization would be impossible if it is **FP4** only.
- **AnythingLLM Advocates for Data Sovereignty**: A user shared a [link to a tweet](https://x.com/AnythingLLM/status/1755265335803212271?t=ypxd85gvodugP-ksZP6Nvg&s=19) about **AnythingLLM** and declared it the future for **data sovereignty**.
   - The user also shared links to **Neuronpedia** and other tweets relating to **data sovereignty** from [Jack_W_Lindsey's tweet](https://x.com/Jack_W_Lindsey/status/1950952346990862502?t=JGcHUqVwZF8_GBoWV5JPcg&s=19) and [heyshrutimishra's tweet](https://x.com/heyshrutimishra/status/1950801664379953468?t=ywRLWQRNGMsoXD8eOMPV-g&s=19).
- **OSS Model Training Script Bootstrapped**: A public research engineer has begun developing an **OSS model training script** to help fill the lack of good OSS models for natural cursor navigation.
   - The engineer acknowledged the possibility that websites that block crawling bots may be scraped by new "clones" using this technology.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Cline bags $32M for Open-Source AI Coding Agent**: Cline, an AI coding agent, secured **$32 million** in Seed and Series A funding led by **Emergence Capital** and **Pace Capital**, aiming to empower developers with transparent, open-source AI tools, serving **2.7 million** developers with transparent pricing and no upcharging.
   - A **Latent.Space Podcast** episode features **Cline**, discussing its origin, the 'Plan + Act' paradigm, community tools, and future directions with Saoud Rizwan and Pash, available on their [website](https://xcancel.com/latentspacepod/status/1951008883163668522) and [YouTube](https://www.youtube.com/watch?v=dQw4w9WgXcQ).
- **OpenAI's OS Model Details YOFO Leaked**: Details about **OpenAI**'s upcoming OS model, **YOFO**, surfaced after its config was briefly accessible, sparking excitement around rumored **120B** and **20B** parameter variants.
   - A member noted that Jimmy Apples was reluctant to share all configuration details.
- **Anthropic's Claude Generates 22,000-Line Code Update**: Anthropic merged a **22,000-line** change to their production reinforcement learning codebase, largely written by **Claude**, sparking skepticism about the reliability of such a large AI-generated code change, which was largely a **json dsl**.
   - Discussions touched on human review processes and concerns about the reliability of large AI-driven code merges; Sauers confirmed the change was real.
- **Anthropic Blocks OpenAI's Claude API Access**: Anthropic revoked OpenAI's API access to its models, including **Claude**, citing a violation of terms of service.
   - **OpenAI** expressed disappointment, noting that its API remains available to **Anthropic**, leading to community discussions about competitive moves and blurring lines of model training.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Query Expansion Boosts RAG**: Discussion revolved around using [query expansion techniques](https://www.promptingguide.ai/techniques/query_expansion) in **RAG** systems by generating multiple questions from a single user query to improve information retrieval.
   - For the query *'what is the name of the customer'*, expanding it to *'What is the name?'* and *'Who is the customer?'* was suggested.
- **Cross-Encoders Flop at Ranking**: Experimenting with a cross-encoder on **MS MARCO** data for ranking results related to the question *'What is the name of the customer?'* yielded poor outcomes.
   - The expected top hit (*Customer Name*) was ranked lower than (*Definition of Customer*), scoring **-0.67** vs **-1.67**.
- **Fine-Tuning is Key for Retrieval**: Directly training on a retrieval task is essential to control ranking quality, according to [this paper](https://arxiv.org/abs/2212.01349).
   - Members suggested that the optimal similarity metric is task-dependent, implying that general-purpose embeddings may not be sufficient for specialized retrieval scenarios.
- **Gemini 2.5 Flash has Gemma Favortism**: **Gemini-2.5-flash** consistently ranked **Gemma models** higher than other models, even some 70B models.
   - The suspected reason is that the response tone of Gemma models might be more plausible to both humans and LLMs, affecting the ranking.
- **Cinema AI generates cohesive movie scenes**: [TheCinema AI](https://thecinema.ai/) research project focuses on generating movie scenes that maintain **cohesion** with each other, according to the [arxiv paper](https://arxiv.org/html/2507.18634v1).
   - The project explores methods for generating cohesive movie scenes and is detailed in the project's website and paper.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM Warriors Demand Offline Access**: Users are seeking ways to save **NotebookLM studio material** for offline access during travel without constant internet connection.
   - One user mentioned downloading audio to an iPad and adding it to PowerPoint slides with family photos.
- **Pro Users Ponder Missing Preview Perks**: Several **Pro account users** report not having access to the **video overview feature**, despite upgrading and others with free accounts having access.
   - A user who briefly had video access lost it after refreshing the page, suggesting ongoing rollout issues.
- **User Dreams of Custom NotebookLM with Gemini**: A user is considering using **Gemini embedding 001** and **Gemini 2.5 models API** to create a custom multi-hop, multi-step reasoning **RAG pipeline** for documents.
   - They aim to surpass **NotebookLM's** capabilities, citing limitations such as the **300-file limit**, lack of transparency in workflow, and limited system instructions.
- **Comet Extension Catapults NBLM into Orbit**: Users discussed **Comet**, a browser extension that can access tabs/history/bookmarks and control the browser, and its potential integration with **NotebookLM** for source finding.
   - The suggestion was raised that **Comet** could potentially code an extension to dynamically add sources to **NotebookLM**.
- **Spanish Audio Overviews Still Short and Sweet?**: A user inquired about why **Audio Overviews** in Spanish remain short in duration, noting a workaround: *switch it to English, change the duration, then prompt it to do it in Spanish*.
   - Another user confirmed that while Portuguese isn't officially supported for explainer videos, they were able to force it to work.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Attention Probes' Performance Proves Polarizing**: EleutherAI's experiments with **attention probes**—tiny neural networks trained to classify transformer hidden states—yielded mixed results, sometimes underperforming standard **linear probes** due to **overfitting** and **optimization issues**, as detailed in their [blog post](https://blog.eleuther.ai/attention-probes/).
   - The code for these experiments has been open-sourced on [GitHub](https://github.com/EleutherAI/attention-probes/), inviting community exploration and refinement to uncover potential improvements.
- **Low-Power LLMs Brave Seabed Scenarios**: A member is deploying **LLMs** on low-power edge devices offshore for seabed mapping, environmental monitoring, and autonomous systems, focusing on **mission planning**, **anomaly detection**, and **smart data compression**.
   - Scientific modeling is currently limited by latency and bandwidth constraints, but the team is actively exploring ways to overcome these **challenges**.
- **Gemini-2.5-flash Judges Gemma Generation**: A member observed that **Gemini-2.5-flash** consistently ranked **Gemma** responses higher when comparing various LLMs, suggesting a potential *family bias* or superior performance of **Gemma3** models.
   - This observation has sparked discussion around the fairness and objectivity of LLM evaluation metrics, as well as the competitive landscape of open-source models.
- **Weight Tying Whips Up Worry**: A member argued that *weight tying is a universally bad practice*, causing inefficiency and instability, and *doesn't even make mathematical sense*, suggesting its detrimental effects on model performance.
   - This assertion sparked debate around the validity of **weight tying** in the broader research community.
- **HF Transformers Tweaks Trigger Tussles**: With **HuggingFace transformers 4.54**, **Llama & Qwen layers** now return residual streams directly (not tuple), which may affect users of `nnsight layer.output[0]`.
   - A member warned that using `nnsight layer.output[0]` will get you the 1st batch element only, not full residual stream, a bug spotted thanks to [nnterp tests](https://butanium.github.io/nnterp).



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider** Still Dominates Code Editing**: Users expressed strong appreciation for **Aider**, citing its superior blend of control and freedom compared to alternatives, with one user estimating **Aider** completed *one week of programming work in a single day for just $2* using **DeepSeek**.
   - Another user emphatically stated, *"Aider rules so hard"*, underscoring its effectiveness in code editing tasks.
- **SGLang** and **Qwen** Break Speed Barrier**: One user reported achieving speeds of **472 tokens/s** using **sglang** and **Qwen 0.6B Q8** on LM Studio with an **RTX 4090**, contrasting with the **330 tokens/s** achieved on regular LM Studio.
   - Another user expressed interest in replicating this local-only setup, particularly since **vllm** performed slower on their **4090** compared to Ollama, showing curiosity in trying *llama.cpp*.
- **Debating Motherboards for Multi-GPU**: Discussion covered hardware configurations, with one member recommending [this MSI motherboard](https://www.bhphotovideo.com/c/product/1864084-REG/msi_meg_x870e_godlike_e_atx.html) for dual **3090s** inside a Fractal North XL case.
   - Others shared their own setups, including servers with **3 L4s** and **T40s**, and diverse case options like the **Meshify2**.
- **Claude Code** Suffers from High Token Count**: Members compared **Claude Code** to other frontier models, noting that its performance degrades significantly beyond **64k tokens**, especially compared to **o3** and **Gemini 2.5 Pro**.
   - It was also mentioned that *the system prompt consumes a substantial portion of the available context window*.
- **Benchmarking **Qwen3 30B** locally**: One member sought an easy way to benchmark 8 different quants of **Qwen3 30B A3B Coder** locally using **LM Studio**.
   - Another member suggested utilizing *llama.cpp server + docker aider benchmark on the same computer* and referenced a writeup on getting **Gemini 2.5 Pro** working.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Security MCP Checker Seeks Feedback**: A member shared a [GitHub repo](https://github.com/minte-app/security-mcp-check) for a **security MCP check tool**, requesting community feedback.
   - This tool aims to assist users in identifying potential vulnerabilities in their **MCP** servers.
- **PayMCP Payment Layer Enters the Ring**: A new **payment layer** for **MCP**, dubbed **PayMCP**, is under development, with [Python](https://github.com/blustAI/paymcp) and [TypeScript](https://github.com/blustAI/paymcp-ts) implementations available.
   - The creator seeks collaborators and early adopters to explore its capabilities in facilitating payment acceptance on **MCP** servers.
- **PageRank for MCP Servers Quest Begins**: A member inquired about **PageRank** implementations for **MCP** servers, with the goal of ranking servers based on utility.
   - Suggestions included a [repository of MCP tools](https://github.com/YogiSotho/mcp-tools-collection) and the [MCP registry](https://github.com/modelcontextprotocol/registry) as valuable resources.
- **JSON MCP Server Cleans House**: A **JSON MCP Server** emerged to aid **LLMs** in efficiently parsing large and complex **JSON** files like **Excalidraw exports**, documented in this [GitHub repo](https://github.com/kehvinbehvin/json-mcp-filter).
   - The solution employs **schema generation** to understand the **JSON** structure and extract necessary data, cutting down on **tokens** and **context**.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Hylo Language Draws 'Heterogenous PL' Parallels**: The **Hylo** programming language ([https://www.hylo-lang.org/](https://www.hylo-lang.org/)) gains attention for its approach to memory safety via **value semantics** and scheduling, compared to **Halide** and **Mojo**.
   - Members reported that the person responsible for **Hylo** is currently working on **Scala 3/Scala Native**, noting that the leads come from **cpp** and **Swift**
- **AMD Drops Kernel AI Agent & GEAK Benchmarks**: AMD introduced the **GEAK benchmarks** and **Triton Kernel AI Agent** in their paper [GEAK: INTRODUCING TRITON KERNEL AI AGENT & EVALUATION BENCHMARKS](https://arxiv.org/abs/2507.23194).
   - Explore AMD's novel approach to **AI-driven kernel optimization** using their new **Triton Kernel AI Agent** for kernel optimization.
- **__launch_bounds__ setting launches CUDA fix**: A user fixed an issue where the compiler couldn't determine register count at entry by passing `minBlocksPerMultiprocessor` to `__launch_bounds__`, setting `maxThreadsPerBlock=128*3` and `minBlocksPerMultiprocessor=1`.
   - The `setmaxnreg` setting is still being ignored, now due to a different problem related to compatibility with an `'extern'` call.
- **MI300X Benchmarks Leave H200 Behind**: A user inquired about experiences with new [MI300X FP8 benchmarks](https://eliovp.com/mi300x-fp8-data%E2%80%91parallel-benchmarks-8-64-gpus-h200-left-behind-b200-within-reach/) on AMD hardware.
   - The benchmarks compare **AMD's MI300X** with **NVIDIA's H200** and suggest the MI300X outperforms the H200 in certain FP8 data-parallel tasks, with performance approaching **NVIDIA's B200**.
- **picocuda compiler Makes Strides Toward GPU Land**: Progress is being made on the [picocuda](https://github.com/j4orz/picocuda) compiler and [elements](https://github.com/j4orz/elements) graph data structures projects, according to members in the singularity-systems channel.
   - The textbook will roughly follow the [GPUCC paper](https://dl.acm.org/doi/pdf/10.1145/2854038.2854041) from CGO '16.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Flux Krea is Out, NSFW is Not**: The new **Flux Krea** model has been released, [available here](https://huggingface.co/Clybius/FLUX.1-Krea-dev-scaled-fp8) promising *much more detail* and compatibility with most lora on base.dev.
   - Early reports indicate that **NSFW** content generation is *not possible*.
- **Emergence AI Emerges Victorious**: **Emergence AI**'s architecture achieved [SOTA](https://www.emergence.ai/blog/emergence-is-the-new-new-state-of-the-art-in-agent-memory) on the **LongMemEval benchmark**, which evaluates long-term memory in AI agents.
   - This positions **Emergence AI** as a leader in memory benchmarks.
- **Smolagents Goes JavaScript**: A member has released **smolagents.js**, a **TypeScript** port of **smolagents**, available on [GitHub](https://github.com/yusuf-eren/smolagents.js) and [npm](https://www.npmjs.com/package/smolagents.js).
   - This port allows developers to use **smolagents** in **JavaScript** environments.
- **Discriminator Learning Rates Fine-Tuned**: Members discussed **debugging GANs** by lowering the **discriminator learning rate** to identify issues, suggesting observing loss changes at very low values like **1e-5**.
   - The goal is to determine if the discriminator's loss collapsing to **0** stems from a learning rate imbalance.
- **Qwen and DeepSeek-R1 Step Up**: Faced with blocked access to **Llama 4**, use **Qwen** or **DeepSeek-R1** as a replacement while running *dummy_agent_library.ipynb* on Colab.
   - These models are considered viable alternatives when access to **Llama 4** is restricted.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Context Window Size: 128k In, 8k Out!**: A user noticed a context window discrepancy, with the **Hugging Face model card** stating **32k context** while **API docs** claim **128k**. The team clarified that it's **128k in** and **8k out**.
   - Cohere team members promised to update the Hugging Face model card.
- **Rate Limits Thwart Hackathon Hopes!**: **Team Patriots**, participating in the **HackRx 6.0 AI hackathon**, faced rate limit issues with the **10 calls/minute trial key limit**.
   - A Cohere team member granted permission to create multiple accounts and cycle the keys to overcome the limit, suggesting rate limits are a known hurdle.
- **Startup Sweet on Cohere's Reranker Seeks Enterprise!**: A startup, enthusiastic about Cohere's **Reranker implementation**, expressed interest in an **Enterprise plan** due to exceeding the **1000/min limit** for the production API.
   - Cohere directed them to email details about their use case to support@cohere.com and varun@cohere.com for secure assistance.
- **Samsung's AI Architect Enters the Chat!**: An AI architect from **Samsung Biologics** introduced themself, focusing on integrating **AI methods and tools** and running a private **LLM service with RAG** for internal use.
   - They are looking to discuss **biopharmaceutical or biological challenges**.
- **Cohere API Hit with Timeouts!**: A user in #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/) reported receiving multiple timeout errors when querying the API.
   - The user was not given any feedback within the chat.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Spammer still spams**: A member reported receiving DM spam and requested an admin to perma-ban the user who is still active.
   - No action was taken during the period, and the spammer continues to spam.
- **Wide Research, is it wide?**: A member inquired about initial takes on using **Wide Research**.
   - No reviews of **Wide Research** were given.
- **Cloudflare config stuck, help needed**: A member is experiencing issues configuring a virtual environment within **Cloudflare**.
   - The setup keeps getting stuck on **Cloudflare**, preventing them from completing the virtual environment configuration.
- **Credits crash, users lash**: A member reported that daily refresh credits are no longer working, indicating issues with the platform's credit system.
   - Another user mentioned having their account suspended despite not breaking any rules, indicating possible issues with account management.
- **Layoffs likely lose refunds**: A member pointed out recent layoffs and suggested the user probably won't get their money back.
   - The comment implies that recent layoffs at the company may impact the ability to process refunds or resolve financial issues.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Joins Forces with Novita Labs**: The [LlamaIndex tweet](https://twitter.com/llama_index/status/1951315242904068483) announces the integration of **LlamaIndex** with **Novita Labs** model inference capabilities.
   - This integration provides diverse data source connections and transformation into vector embeddings.
- **Gemini Speaks TypeScript Fluently**: The [LlamaIndex tweet](https://twitter.com/llama_index/status/1951342252346974431) announces **Gemini Live integration** now available in **TypeScript**.
   - A demo is provided showcasing how to set up and run a basic terminal chat.
- **Engineer Crafts AI On-Chain**: A Senior AI & Blockchain Engineer is building **on-chain AI agents** for trading, media automation, and autonomous governance using **Eliza OS**, **LangGraph**, and custom toolchains.
   - This engineer has worked extensively across **Base**, **Solana**, **Berachain**, **Sui**, **Aptos**, **HBAR**, **EVM chains**, and cross-chain systems.
- **Git-Style Branching for LLM Conversations**: A member is experimenting with a system where each message is a node, enabling branching off at any point in the conversation to create new context paths, as detailed in [their blogpost](https://gupta-aniket.github.io/Mobile-developer/hire/#projects#branched-llm-mvp).
   - The system currently uses **Gemini API**, with plans to include **GPT-4**, **Claude**, and local **LLaMA** models, seeking testers for feedback.
- **Llama Parsers take fare share of time to parse**: Members discussed the performance of **LlamaIndex parsers** for **.doc**, **.pdf**, and **.ppt** files, particularly when dealing with text embedded in images.
   - Solutions proposed include using **LlamaParse** in premium mode, converting PPTs to PDFs for improved speed, or implementing **ThreadPoolExecutor()** for asynchronous document parsing.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSpill verb is coined for Yaron Minsky**: Members discussed who would *give it a second try to **DSpill Yaron Minsky / quant bros***, leading to a new verb '**DSpill**'.
   - The term '**DSpill**' was proposed to describe action against **Yaron Minsky** and the **quant bros**.
- **DSPy is now RL!**: A member shared [a blogpost](https://www.dbreunig.com/2025/07/31/how-kimi-rl-ed-qualitative-data-to-write-better.html) about using Reinforcement Learning in DSPy to improve writing quality.
   - No discussion happened, but could be interesting for those looking to optimize their generations.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Install Issues Merit GitHub Attention**: A member faced **Mojo** installation difficulties and contemplated opening a **GitHub issue** to report the problem.
   - Another member advised them to create a **GitHub issue** with detailed logs to assist developers in diagnosing and resolving the installation problem efficiently.
- **Logs are a Developer's Best Friend**: The discussion highlights the importance of including detailed logs when reporting **Mojo** installation issues on **GitHub**.
   - Providing comprehensive logs enables developers to diagnose and resolve the problem more efficiently by providing necessary information for debugging.
- **Print Statements Inhibit Tail Call Optimization?!**: A member observed that adding basic **print/log statements** to functions prevents **tail call elimination**.
   - The discussion is about how the addition of **print/log statements** affects **tail call elimination** in minimal **Mojo** examples and seeks to understand the underlying reasons for this behavior.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **OpenAI's Model Leaks with 128 Experts**: A rumored **OpenAI** model with **128 experts** and **120B parameters** has potentially leaked.
   - The model's weights are reportedly in **FP4** format, suggesting a compressed state.
- **Deep Dive into Mixture of Experts**: **Mixture of Experts (MoE)** models use multiple sub-networks (experts) with a gating network to route inputs.
   - This architecture enables scaling model size without a proportional increase in compute costs, making it an active area of research.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **MOOC Quizzes with Answer Keys Now Available**: An archive of the **quizzes with answer keys** is now accessible in the *"Quizzes"* section of the course website.
   - This gives students a resource to review course material and assess their understanding.
- **Google Forms to Remain Closed**: Course staff announced that they cannot reopen the **Google Forms** used for quizzes.
   - Students who missed taking quizzes via **Google Forms** should use the available archive for review.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Qwen3-Coder Surfs into Windsurf at Breakneck Speed**: **Qwen3-Coder** is now available in Windsurf, operating at approximately **2000 tokens/sec**.
   - Announced via [X](https://x.com/windsurf/status/1951340259192742063) and [Reddit](https://www.reddit.com/r/windsurf/comments/1mf3e5s/qwen3coder_at_2000_tokenssec_is_now_live_in/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button), the model is fully hosted on US servers.
- **Windsurf's Newest Resident: Qwen3-Coder**: Windsurf now houses **Qwen3-Coder**, boasting a blazing speed of **2000 tokens per second**.
   - The implications of this new model are being discussed on [Reddit](https://www.reddit.com/r/windsurf/comments/1mf3e5s/qwen3coder_at_2000_tokenssec_is_now_live_in/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button).



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Developer Seeks Opportunities**: alex_sdk4 inquired whether anyone is seeking a developer.
   - No further details regarding specific skills, projects, or expectations were provided.
- **Follow up: Developer Seeks Opportunities**: Since alex_sdk4 reached out, this may be a good opportunity for smaller tasks.
   - Potential clients can reach out directly to alex_sdk4.



---


The **tinygrad (George Hotz) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1400553929611280499)** (1048 messages🔥🔥🔥): 

> `Comet Browser Invites, Image Generation Issues on Perplexity Pro, Free Perplexity Pro for Airtel Subscribers in India, GPT-5 Release Speculation, Model Performance Comparison` 


- **Comet Browser Invites Rolling Out Gradually**: Perplexity is rolling out **Comet Browser** invites almost daily, prioritizing **Pro users**, but the wait time may vary.
   - Some users suggest that if your daughter has a Pro account, she can send you up to **2 invites**.
- **Image Generation Glitches Plague Perplexity Pro**: A user reports that image generation on **Perplexity Pro for iOS** fails to incorporate attached images, and another user confirms this is a recurring issue.
   - The model summarizes the request but doesn't generate an image based on the attached file, and starting a new chat does not consistently resolve the problem.
- **Airtel Subscribers in India Snag Free Perplexity Pro**: A user mentioned that **300 million people in India** get Perplexity Pro for free for **12 months** if they are Airtel subscribers.
   - To use the promo you have to be located in India and be an Airtel subscriber.
- **GPT-5 Release Date Remains a Mystery**: Users speculate about the release of **GPT-5**, with one suggesting it could be next week, but another member insists that it will probably be some kind of mini model lol.
   - One user had briefly seen **GPT-5 in the API**, but it was quickly removed ([source](https://x.com/chetaslua/status/1951301385292493259)).
- **Model Performance Sparks Debate: Sonnet 4 Reigns Supreme, O3 Holds Its Own**: Users discuss their experiences with various models, with **Sonnet 4** being praised for coding and value, while **O3** is recommended for reasoning ([cplx.app](https://www.cplx.app/)).
   - The discussion touches on tool call issues and the tendency of Anthropic models to *hold back information unless specifically asked*.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1400597657667244112)** (7 messages): 

> `Shareable threads, RAG without embeddings, Trump-Medvedev` 


- **Thread Sharing Settings Clarified**: A Perplexity AI staff clarified with a user that the thread should be set to `Shareable`.
   - A link was shared about *how to make threads shareable*.
- **OpenAI RAG without Embeddings**: A member shared a [Medium article](https://levelup.gitconnected.com/rag-without-embeddings-heres-how-openai-is-doing-this-45866cd5ddc6) about **RAG without embeddings** and how **OpenAI** is doing this.
   - It was written by **Gaurav Shrivastav**.
- **Trump-Medvedev drama with 2 nuke subs**: A member shared a [Perplexity search result](https://www.perplexity.ai/search/find-information-about-trump-p-g67iddgiQSe1WR4x6GKNjg#2) about the **Trump-Medvedev drama with 2 nuke subs being positioned near Russia** for a new Human Benchmark Report for August 1st.
   - They shared a [Gemini Canvas infographic](https://g.co/gemini/share/c43c0a891af3) made up for the report itself.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1400582585968496640)** (14 messages🔥): 

> `search_domain_filter, Moderator Bot Usage, Image Uploading via API` 


- **Troubleshoot Search Domain Filter!**: A user reported that the **search_domain_filter** is not being honored, even as a Pro subscriber, requesting insight on enabling the feature.
   - Another member responded saying that it should be working (not in beta), and requested a copy of the request for assistance.
- **Moderator Bot Pricing Questions?**: A student inquired about the usage and pricing for a moderator bot using **Perplexity AI**, anticipating around **200 requests** with less than **100 words** of data each.
   - The user is trying to make a moderator bot using perplexity AI.
- **Image Uploading gives Internal Server Error!**: A user encountered an internal server error (**code 500**) when uploading images as base64 via the API.
   - They then shared their [B4J code](https://www.b4x.com) to demonstrate their method, while a member asked for the request and the model being used.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1400560721535959271)** (1099 messages🔥🔥🔥): 

> `GPT-5 speculation, Qwen3 model, Cogito V2, Unsloth GRPO and TRL, H100 and batch sizes` 


- **GPT-5 Panic Drop Speculation Arises**: Members are speculating whether **GPT-5** will be a *panic drop* or a middle-of-the-road improvement due to **OpenAI's** exhaustion of scaling model size and diminishing returns from Chain of Thought (**CoT**).
   - There are claims CoT is a *complete dead end* and it's possible to achieve the same thing by feeding the model's vector output back through the network directly instead of using tokens for thinking.
- **Qwen3 Quantization and Performance Tests**: There's discussion on the ideal quantization for **Qwen3 Coder 30B**, with some finding the **Q4_K_M gguf** slow when adding context in **Ollama**, while others prefer **UD q3 XL** for VRAM savings.
   - One member reported running the April **Qwen3-30b-a3b** model at **40k** in **vllm** on a **3090** 24/7, while others eagerly await a 4-bit AWQ version for the coder model.
- **Cogito V2 Reinforcement Learning Discussed**: Members discussed the release of **Cogito-v2 GGUFs** and their reinforcement learning approach, with some viewing it as an iteration on existing techniques rather than a novel breakthrough.
   - A member shared an article covering process reward models in 2024 ([synthesis.ai](https://synthesis.ai/2025/02/25/large-reasoning-models-how-o1-replications-turned-into-real-competition/)), and another member shared a **Deepmind** paper from 2022 exploring similar concepts ([arxiv.org](https://arxiv.org/abs/2211.14275)).
- **Unsloth GRPO already supports GSPO**: A member asked about updating **Unsloth** to support **GSPO** learning, after Qwen proposed it as an update to **GRPO**.
   - Another member clarified that **GSPO** is slightly more efficient but that it already works in **Unsloth**, and that Unsloth will auto-support **TRL** updates because it is a wrapper.
- **Rumored New OpenAI Model Sparks Excitement**: Rumors of a new **OpenAI** model are circulating, with some speculating it could be the best Operating System (**OS**) model and beat **SOTA K2** in evaluations.
   - Many are hyped for a potentially dense **20B** base model, which could pair well with existing recipes, while others are curious if it will be dense or another mixture of experts (**MoE**).


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1400858572593565747)** (4 messages): 

> `New member introduction, Community assistance` 


- **New member joins, admits ignorance**: A new member, cyber.n0de, introduced themselves and humorously admitted to being completely clueless.
   - They expressed a need for guidance, signaling a potential opportunity for community assistance and onboarding.
- **Community offers helping hand**: A member, theyruinedelise, promptly responded to the new member's admission of ignorance by offering assistance.
   - This illustrates the community's willingness to support newcomers and provide guidance.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1400780930255163402)** (74 messages🔥🔥): 

> `VITS checkpoint training insights, On-device VITS system on iOS, Children voices recording, Avocodo and iSTFTNet for audio fidelity, Universal vocoder for Speech LLM` 


- **VITS Training Yields Eureka Moments**: After training a **VITS checkpoint overnight**, a member shared insights: **model quality depends on the number of epochs and dataset quality**, and **VITS excels at speaker disentanglement** for creating models with distinct voices.
   - They noted **VITS encodes raw audio into latent space** for realistic recreation and emphasized that it depends on specific needs compared to RVC.
- **VITS Runs into iOS Memory Mayhem**: A member reported that using **VITS for on-device system voice on iOS** faces memory consumption challenges with the **Hifi-GAN decoder**, requiring chunk-wise decoding.
   - They also found **VITS can learn subtleties like breathing at commas** and different styles for quoted text with proper annotation.
- **To Child Voice, Schedule Recording Hours Carefully**: A member expressed uncertainty about the number of hours needed to **record children's voices** for fine-tuning light woman-voices for a better baseline.
   - Another member suggested that 24 hours per speaker is overkill, emphasizing the need for quality data over quantity.
- **Avocodo's Fidelity Facelift Forefronts**: Members discussed **Avocodo** as a means for quick fidelity boosts without significant speed increase, noting reduced artifacts limited to dataset quality, with a link to an unofficial [Avocodo-pytorch implementation](https://github.com/rishikksh20/Avocodo-pytorch).
   - They pointed out that the linked implementation uses **Hi-Fi GAN** but requires training a model yourself.
- **Universal Vocoder Quest Kickstarts**: A member expressed a need for a **universal vocoder** for plugging **VITS into a Speech LLM**, requiring fast speed, low GPU usage, and the ability to train from scratch.
   - One suggestion was [BigVGAN](https://github.com/NVIDIA/BigVGAN), though the original poster wants to train from scratch; others considered the impact of lightweight LLM architecture.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1400554633470152867)** (207 messages🔥🔥): 

> `Circular Import Error, RuntimeError with Merged Model Loading, UV venv performance, Qwen3 tool calling problems, Qwen3-Coder-30B-A3B-Instruct-1M-Q8_0.gguf on vLLM` 


- **Circular Import Causes Grief**: One member reported an `ImportError: cannot import name 'convert_lora_modules' from partially initialized module 'unsloth_zoo.vllm_utils'` arising from a **circular import** when using `use_async=True` with `unsloth.FastLanguageModel.from_pretrained`.
- **Special Tokens Trigger Runtime Error**: A member encountered a `RuntimeError` related to **size mismatch** when loading a merged model after fine-tuning and adding **2 special tokens** to the tokenizer and model's embedder.
   - Another member suggested that adding new tokens isn't fully resolved, and the system might still attempt to load the base model's tokenizer; also, using `resize_model_vocab = 128258` may partially solve the issue, but not consistently for merged models, as it may load the base model's tokenizer.
- **UV venv causes performance decrease**: A user experienced a **20x performance slowdown** when using Unsloth within a **UV venv**, which led to extremely slow initialization during cuda graph shape capture.
   - It was suggested that UV might be downloading all xformers versions, causing the slowdown, but a member pointed out that they used mamba instead, to avoid using UV altogether.
- **Tool Calling troubles with Qwen3**: A user reported issues with **Qwen3 30B variants** not reliably performing **tool calling** in their **Langchain app**, unlike previous experiences with Qwen3 4B and larger models, despite using the latest Unsloth versions with Ollama.
   - It was suggested to check `fast_inference=True`, but the user confirmed it was already enabled, then it was suggested to check [this vLLM issue](https://github.com/vllm-project/vllm/issues/12324) related to vLLM and UV.
- **vLLM struggles with GGUF model**: A user encountered a `ValueError: GGUF model with architecture qwen3moe is not supported yet` when attempting to run **Qwen3-Coder-30B-A3B-Instruct-1M-Q8_0.gguf** on **vLLM**.
   - Members suggested that gguf format should rather be run on *llama.cpp* and noted the model architecture may not be supported, prompting a suggestion to install Transformers from source to potentially resolve the issue.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1400764067383083079)** (8 messages🔥): 

> `Unsloth Dynamic Quantization, Qwen3 30B-A3B, Space Invaders refined, Roleplay AI finetuning, Gemini Refusals` 


- **Dynamic Quantization gets Quant Clone**: A member created [a small application](https://github.com/electroglyph/quant_clone) to quantize finetunes the same way as Unsloth's dynamic quantization.
   - They wanted to replicate Unsloth's dynamic quantization on their own finetunes.
- **Unsloth's Qwen3 Coder Model builds Space Invaders**: Using a **Q4_M unsloth Qwen3 30B-A3B coder model** and Cline in VS Code, a member created and refined a Space Invaders style game.
   - The game was completed in about ten minutes without touching a single line of code, and is available [here](https://invaders.smolit.us/space_invaders/).
- **Roleplay AI finetunes with Unsloth**: A member announced an easy way to finetune with Unsloth and provide more data with their [roleplay-ai project](https://github.com/bjoern-buettner/roleplay-ai/tree/the-one/beam-llm-training).
   - The model is available on Hugging Face.
- **Gemini faces High Refusal Rates**: A member asked if others have experienced a higher level of refusal with their finetunes, comparing it to **Gemini**.
   - The member finds *Gemini to be quite obnoxious* in that regard.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1400731806977753219)** (4 messages): 

> `Gemma 3 1B garbage, finetuning project, continuous training of loras` 


- **Gemma 3 1B Flops**: A user trained **Gemma 3 1B** and found it to be *absolute garbage*, and a waste of compute, and is sticking to benchmark-crashing **4B** models.
   - They did not mention the training dataset or training methodology.
- **Fineteuning project in the works**: A user is looking to collaborate on a **finetuning project** using open-source LLMs, with compute available on GCP.
   - They are keen to work on anything from **code models** to domain-specific applications.
- **Continuous LoRA Training Revisited?**: A user inquired about recent work on continually updating the weights of a model, referencing some research from Amazon on **continuous training of LoRAs** from a few years ago.
   - Another user, suresh.b, confirmed the existence of such work, though didn't provide further details or links.


  

---


### **Unsloth AI (Daniel Han) ▷ #[unsloth-bot](https://discord.com/channels/1179035537009545276/1390899684834410536/1400636791173677151)** (114 messages🔥🔥): 

> `GRO Trainer dataset mapping, Chat template cut off, GRPOTrainer config, Sequence dictionary (seq-dict), Unsloth shape dynamically changes` 


- **Troubleshoot Permutation Errors in GRPO Trainer**: Users are facing permutation errors with the GRPO trainer when using a **Qwen 2.5** base model due to dataset feature issues like `Question` and `Answer`.
   - The error arises from the `shuffle_sequence_dict` function, particularly with `ref_per_token_logps`, indicating potential source code problems.
- **Can't configure Unsloth's Output Embeddings**: Users are struggling to configure the offloading location for `output_embeddings` in Unsloth, which defaults to storing in the `{model}/output_embeddings.pt` path.
   - It was raised as a concern that this *behavior* will be problematic if the user does not have write permissions to the `{model}` path.
- **Gemma's Image Format for Fine-Tuning**: Users are debugging the correct format for using multiple images and system prompts when fine-tuning **Gemma-3-it-4B**, encountering `ValueError: Invalid input type`.
   - The correct format involves structuring the input data with `type` keys for both text and image content, accommodating a mix of images with or without system prompts but requiring consistent image numbers per sample.
- **Leveraging AI for Fine-Tuning Data Generation**: Users are exploring methods to convert **0.5 million tokens** of raw text into fine-tuning data using AI, specifically considering models with long contexts or RAG.
   - The discussion included whether to use a **Phi-14B** model with RAG to create training data, although chunking was dismissed as an option.
- **VRAM Swells Up During SFT Training**: Users are curious about why **VRAM** increases during **SFT** training, presuming memory pre-allocation should prevent this.
   - It was mentioned that *training would be amenable to pre-allocating memory*.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1400554342616010903)** (968 messages🔥🔥🔥): 

> `Arena Visibility, Leaderboard Tooltips, Personal Info in Datasets, Gemini's Repetitive Tendencies, Gemini 2.5 Deepthink` 


- ****Arena Buttons Boost Browsing****: A member suggested adding three major buttons for **Search, Image, Video, and Webdev Arena** to increase visibility, sharing a [concept image](https://cdn.discordapp.com/attachments/1340554757827461211/1400554342167089222/uzzzSHh.png).
   - Another member recommended adding a **webdev arena** button since it's on a separate platform, and also adding tooltips to the leaderboard explaining how **Rank, CI, and Elo** are determined.
- ****Dataset Delves Deliver Dangerous Data****: A user voiced concerns about accidentally including **personal information** (emails, passwords, etc.) in published prompts, and suggested a way for users to remove prompts before public release.
   - A member responded that such examples should be DM'd to them for escalation, and acknowledged [sharing these concerns with the team](https://www.deepcogito.com/research/cogito-v2-preview).
- ****Gemini Gabs Get Glitchy****: A member asked if others noticed **Gemini** repeating itself, but another member found it consistent and questioned if **Gemini 2.5 Flash** improved.
   - One user noted video limits dropping from **10 to 8**, urging others to use the video generation arena quickly.
- ****DeepThink's Debut: Disappointment Delivered?****: **Gemini 2.5 Deepthink** is out for Ultra members, and members are wondering if it is worth it after seeing **10 RPD limit**.
   - Members called it a **scam** and a daylight robbery, with some saying it's just a rushed version because of the imminent **GPT-5** release.
- ****GPT-5 Gossip Generates Great Expectations****: Discussion revolved around **GPT-5's** potential release, with some anticipating a paradigm shift while others expect incremental improvements and members discuss various performance benchmark data.
   - A member stated the view that *we're moving away pretty rapidly from "the best" model* as routing to a really strong model might be a really strong model for some stuff but not use it all the time.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1400888347160739932)** (1 messages): 

> `Veo 3, Image-to-Video, Audio capabilities` 


- **Veo 3 Unleashes Image-to-Video & Audio**: **Veo 3 Fast & Veo 3** now boast **Image-to-Video with audio capabilities** within the [Video Arena](https://discord.com/channels/your_server_id/1397655695150682194).
- **Create Videos with Images in Discord**: A new `/image-to-video` command has been added to the video-arena channels: allowing users to create videos from images.
   - Users are encouraged to vote on the best videos created using the new command.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1400555426764034119)** (580 messages🔥🔥🔥): 

> `Background agents, Improving cursor setup, Cursor freezing issues, YOLO mode activation, Vibe coding strategy` 


- ****Vibe Coding Github Needed****: One member said that *for background agents you need github? this thing is sick* with attached image.
   - Another member had spent **$40** on prompts, and needed advice on improving their **Cursor** setup.
- ****Cursor Freezing Bug Frustrates Users****: A user reported that their machine freezes every **30-60 seconds** after being in a chat for more than an hour.
   - A Cursor team member suggested posting the issue on the [Cursor forum](https://forum.cursor.com/c/bug-report/6) for better visibility and assistance.
- ****Navigating the Murky Waters of Model Spending****: Users are comparing **Cursor** and **Claude Pro** pricing, with one user saying *I go where the cheapest plans and best models are to be honest, and the $200 plan with Claude is one of the best deals currently still for me even with their new weekly hour limits*.
   - Another user expressed the cost can quickly balloon, spending *$600 in 3 months*.
- ****Horizon Alpha Experience Underwhelming****: One user found their personal experience with **Horizon-Alpha** to be *a bit underwhelming*.
   - In contrast, another user said *cursor is the best app i have ever seen*.
- ****Cursor Users Request Referral Program****: Members are asking if there is a referral program for **Cursor**, as one member mentioned having onboarded *at least 200+ people by now sitting in discords lmao*.
   - A link to the [Cursor Ambassador program](https://cursor.com/ambassador) was shared.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/)** (1 messages): 

lintaffy: oh, my ba is still loading for the easy command....
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1400554785911865466)** (410 messages🔥🔥🔥): 

> `Function Calling vs XML, AI Superintelligence Bio-Weapons, Grok4 vs GPT5, Horizon Alpha Performance, Large Context Windows` 


- ****Function Calling APIs**: Inherent Value?**: Function Calling APIs are seen as having **inherent value** compared to using structured XMLs for function calls, but one member noted that [XML is often used as a workaround](https://drinkoblog.weebly.com/) when the model doesn't support tool calling.
   - Some coding models like **Qwen** don't support function calling, so inline tool calls maximize interoperability despite minor inefficiencies.
- ****Zuckerberg's AI Superintelligence**: A Bio-Weapon Threat?**: **Mark Zuckerberg's** AI superintelligence initiative sparked concern over potential bio-weapon creation, with one member stating that *you cant just release superintelligence to the public like that*.
   - Concerns were raised that *controlling minds with fake users and carefully crafted language* is even more dangerous than bio-weapons.
- ****GPT-5 Delayed**: Grok4's Victory?**: Rumors suggest **GPT-5** is delayed due to inability to surpass **Grok4**, but another member stated that [OpenAI is planning to combine multiple products into GPT-5](https://link.to/openais-next-foundational-model).
   - A member also clarified that **GPT-5** will be a single, unified, omnimodal model.
- ****Horizon Alpha Shines**: A Free Reasoning Model?**: **Horizon Alpha** appears to outperform paid LLMs via the OpenRouter API, delivering [perfect one-shot code in custom programming languages](https://openrouter.ai/), with one user claiming, *it was like 3-4 times more useful than o3o3's multi turn is so bad*.
   - Its advanced shell use and task list creation in orchestrator mode prove superior to other models, though some believe it *could always be something turbo weird we’re not thinking of like codex-2*.
- ****Context Windows**: Overrated or Essential?**: Despite **Gemini's** 1 million context window, legacy codebase issues were better solved with **Claude** and **ChatGPT**, sparking debate on whether [large context windows are overrated](https://nealgoogs.website).
   - Some believe models with smaller context windows and better output are preferable, whereas others assert that larger context windows are essential for agentic applications to *remember and weave in far‑back details automatically*.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1400657438746738861)** (11 messages🔥): 

> `Agent Mode Confusion, ChatGPT Agents vs Regular GPT, GPT-4o auto reasoning, Missing Chat History` 


- **Agent Mode Causes Confusion**: Users are experiencing confusion around the term **Agent Mode**, with some believing it to be a new feature when it's essentially referring to existing advanced modes like **Code Interpreter**/**Advanced Data Analysis**.
   - Some members attribute initial hiccups to basic growing pains, suggesting it might get confused, give wrong answers, or simply stop working but is *awesome* when it works.
- **ChatGPT Agents vs Regular GPT**: A member points out that [ChatGPT models are unaware of recent developments](https://openai.com/index/introducing-chatgpt-agent/), including new products like **ChatGPT Agent**.
   - Another member reported using **Agent Mode** to work within **GitHub** to resolve an issue, finding it *quite interesting to watch what its doing*.
- **GPT-4o Auto Reasoning**: Users noticed that **GPT-4o** auto-switches to *Thinking*, even when not tagged as **Deep Research** or **Study mode**.
   - The switch to **o3** for technical or coding-related questions results in big reasoned replies, which some users find undesirable, preferring concise responses.
- **Chat History Goes Missing**: A member reported that their **chat history** (not in a folder) progressively disappeared throughout the week on both the web and mobile app.
   - Another member mentioned that *it should be fixed tho* and that *they fixed it as of yesterday*.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1400645347990311045)** (1 messages): 

> `` 


- **No significant discussion**: There was no meaningful discussion to summarize from the provided content.
- **No noteworthy insights**: The provided screen recording did not contain any noteworthy insights or topics for summarization.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1400645347990311045)** (1 messages): 

> `` 


- **No Topics Discussed**: No relevant topics were discussed in the provided messages.
   - The content appears to be a screen recording without specific details for summarization.
- **Insufficient Data for Summary**: The provided image analysis lacks textual content suitable for generating meaningful summaries.
   - Further information or message details are required to create relevant topic summaries.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1400554223522811936)** (325 messages🔥🔥): 

> `Image-to-video prompt generation in LM Studio, LM Studio's lack of roadmap, LM Studio's Plugin System, Connecting to LM Studio API from other computers on the network, Qwen3 Coder model support on LM Studio` 


- **Image-to-Video LM Studio When?**: Members are wondering about future **image-to-video prompt generation** and **image attachment** features in **LM Studio**, expressing a preference for offline solutions over relying on **ChatGPT**.
   - A member suggested **ComfyUI** as an alternative, but noted it's *not as good on AMD cards*.
- **Roadmap Unknown, So Noone Knows**: Members discussed the lack of a **public roadmap** for **LM Studio**, with one suggesting the roadmap is *just a big bucket with random papers*.
   - Another member confirmed there's *noone* who knows what the plan is and stated *no public roadmap so noone knows*.
- **Securing LM Studio on the Network**: Members discussed connecting to the **LM Studio API** from other computers on the network, with concerns raised about security.
   - It was suggested that **LM Studio's security is not proven** and should not be exposed without understanding the risks and securing your own network.
- **Qwen Crash Course: Load Model!**: Members discussed issues with loading the **Qwen3 Coder 30B** model, with one user experiencing a *Cannot read properties of null (reading '1')* error.
   - A member pointed out the user should update the app version to **0.3.21 b2** which supposedly fixed the issue, and mentioned to click the **recommended settings**.
- **Speculative Decoding: Not Worth It, Says Fabguy**: A member inquired about using **speculative decoding** with **Qwen3 MoE** models, which leads to a crashing error.
   - Another member pointed out that *draft model and primary model may pick very different experts for the task [of speculative decoding]. Not worth it.*


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1400555314864197673)** (69 messages🔥🔥): 

> `Nvidia Driver 580.88, Second-hand servers, Partial KV Cache Offload, Mac mini M4 vs RTX 3070, Next-gen GPUs` 


- **Nvidia's Jumps in Driver Versions**: Nvidia released driver **580.88** shortly after **577.00**, a **9-day-old driver** with a potential fix for GPU video memory speed after enabling NVIDIA Smooth Motion [5370796].
   - The user runs the drivers from the cuda toolkit, and doesn't use the fancy control panel or GFE (GeForce Experience).
- **Pondering Partial KV Cache Offload**: There was a question raised about whether it is possible to do a partial KV Cache offload in LM Studio, for example with a **40GB model**, where **KV Cache needs 20GB**, and **GPUs have 48GB total**.
   - The user was wondering if it was possible to split, with 8 of 20gb of the cache would be in the gpu, the rest offloaded.
- **Mac mini M4 Sizing up against RTX 3070**: A user wondered if a **Mac mini M4 ten core 32GB** would outperform an **RTX 3070**.
   - It was stated that CUDA is generally quicker than silicon if the models can fit in VRAM.
- **Rambling about RAM Recommendations**: One user suggested saving money for a used **3090**, which they claim is the best bang for buck card for AI use cases.
   - They cost around **700 euros** and for LLMs it would probably be the best solution, but there might be issues since they might've been used in mining.
- **5070 TiS release is immanent!**: A user speculates the **5070TiS** will be released soon with **24 gigabytes** of ram, where the **5070ti & 5080 have 16 gigs of ram**.
   - Another user points out that for cheap inference, right now 5060Ti 16gigs are the best option, at 450€/each, and you can put 3 or 4 in a board.


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1400592183010263101)** (11 messages🔥): 

> `PyrenzAI launch, Personality.gg, OpenRouter PKCE, PyrenzAI feedback` 


- **Personality.gg enables Roleplay via OpenRouter**: [Personality.gg](https://personality.gg) launched a roleplay site using **OpenRouter** for most models, providing access to all 400 models through **OpenRouter PKCE** (Proof Key for Code Exchange) completely free/cheap.
- **PyrenzAI Launches a Free AI Chat Website**: A developer announced the launch of [PyrenzAI](https://pyrenzai.com), an **AI chat website** with a clean UI, models, a memory system and **free RAG** (Retrieval-Augmented Generation) for all tiers, using OpenRouter as the main AI generation backend.
- **PyrenzAI app faces speed and security critiques**: A user critiqued the newly launched PyrenzAI app, noting it's *cooked in terms of both speed and security*, with *laggy* performance and excessive fetching of user preferences (over 200+ times on every load).
- **UI and UX lauded on PyrenzAI release**: A member complimented the **UI/UX** of [PyrenzAI](https://pyrenzai.com), appreciating its unique look and style, and distinctive sidebar design compared to other apps.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1400553644956323921)** (242 messages🔥🔥): 

> `API Errors, Deepseek r1, Free Models, Horizon Alpha, API Key credit limit` 


- **API Errors plague OpenRouter Users**: Some users reported experiencing **API errors** when trying to use models via the OpenRouter API, including *no endpoint found* errors and other issues.
   - A member suggested checking the **model ID prefix** and the **base URL** for potential misconfiguration.
- **Deepseek v3 Outage Strikes Users**: Users reported issues with the **Deepseek v3 0324 free** model, including *internal errors*, *empty responses*, and **timeouts**.
   - One member noted that switching to the paid version of the model resolved the issues, suggesting the free version was overloaded: *free is completely overloaded. paid has none of these issue, and the actual content quality is better.*
- **Free Model Limits Frustrate OpenRouter Users**: Several users inquired about **free models** with higher message limits, with one user asking if there was any free model that *wont stop at 50 messages?*
   - Members clarified that topping up with **$10** provides a **1000 requests/day** limit and referenced [OpenRouter documentation](https://openrouter.ai/docs/api-reference/limits#rate-limits-and-credits-remaining) detailing the limits.
- **Horizon Alpha Raves Gain Momentum**: Users discussed the **Horizon Alpha** model, with some reporting that it was reasoning effectively and offering good performance.
   - The model itself reported that it was developed by OpenAI, though other members clarified that it was likely a distilled model.
- **Budget Overruns Baffle API users**: A user reported being charged significantly over their **API key credit limit**, suspecting that running **API calls in parallel** with Python threads might be the cause.
   - Other users shared similar experiences, suggesting that the credit limit updates might not be real-time, leading to occasional overcharges.


  

---


### **OpenRouter (Alex Atallah) ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - New Models**
  

---


### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1400586072773103901)** (23 messages🔥): 

> `Groq OpenBench, Provider Benchmarks, GPQA Evals, Inspect.ai, Prompt Caching for Kimi K2 and GLM 4.5` 


- ****OpenBench Groqs** for Provider Benchmarks**: Members discussed the [Groq OpenBench](https://github.com/groq/openbench) repository and how many times it has been posted regarding **provider benchmarks**.
   - One member mentioned they are *already working on evals (recently got prioritized)*, such as **GPQA** per provider, and expanding to other things.
- ****Inspect.ai** Discovery Praised**: A member expressed happiness in discovering [inspect.ai](https://inspect.ai) through the **OpenBench** link, noting it's *exactly what I've been looking for*.
   - This same user noted concerns about the chat UI using their full name from their account without control over it, leading to potential doxxing.
- ****Prompt Caching** Questioned for Kimi K2 and GLM 4.5**: A user inquired whether **OpenRouter** supports **prompt caching** for **Kimi K2** and **GLM 4.5**, noting that **Moonshot**'s platform directly supports it.
   - They stated it somewhat looks like it on [z.ai](https://z.ai).
- **Bypassing 20MB Limit: **Bigger PDFs** are now sendable**: Members questioned whether new feature would bypass the **20MB limit**, and they mentioned that they *recently added a way to send bigger pdfs*.
   - The new limit is the **upstream provider limit**.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[announcements](https://discord.com/channels/1369594130807787570/1371757097246785536/1400719197838770217)** (2 messages): 

> `Kimi K2 Turbo, Moonshot AI Forum` 


- **Kimi K2 Goes Ludicrous Speed!**: The Moonshot team announced **Kimi K2 Turbo**, a faster version of the Kimi K2 model, boasting **4x the speed** at **40 tokens/sec** from **10 tokens/sec**.
   - Until **Sept 1**, users get a **50% discount** on input and output tokens ([platform.moonshot.ai](https://platform.moonshot.ai/).
- **Moonshot AI Launches Official Forum**: The Moonshot AI team announced the launch of the ***Moonshot AI Forum*** ([https://forum.moonshot.ai/](https://forum.moonshot.ai/)) as a new hub for technical discussions, API help, model quirks, debugging, and dev tips.
   - *Discord’s still vibin for memes*, chill convos, and messin with ***Kimi Bot*** but if u tryna get serious with builds and tech stuff? forum’s the new spot fr 🔥


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1400679315850526800)** (126 messages🔥🔥): 

> `Kimi vs Claude, Kimi K2 Turbo pricing and speed, Using Kimi K2 Turbo in Claude code, Chinese companies video generation, Kimi K2's prompt format similar to ChatGPT` 


- **Kimi K2 challenges Claude throne**: After testing, a user finds that **Kimi K2** is the first model they feel they can use instead of **Claude**, ditching **Gemini 2.5 Pro** completely.
   - They add that coding, as a kind of information, is becoming freer and it's happening way faster than expected, eventually, most AIs will converge in terms of knowledge, and the differences between them will start to fade.
- **Kimi K2 Turbo goes 4x faster**: Kimi K2 Turbo is the **same model but with faster hosting**, now available with a special promo until Sept 1: **$0.30/1M** input tokens (cached), **$1.20/1M** input tokens (non-cached), and **$5.00/1M** output tokens.
   - This pricing implies it's *4x faster for 2x the price* during the discount, intended for users with speed requirements, and its official API helps keeps things steady.
- **Kimi K2 Turbo environment variable settings**: To use `kimi-k2-turbo-preview` in Claude code, set the following environment variable configurations: `export ANTHROPIC_SMALL_FAST_MODEL=kimi-k2-turbo-preview` and `export ANTHROPIC_MODEL=kimi-k2-turbo-preview`.
- **Kimi K2's prompt design mimics ChatGPT's**: Users noticed Kimi's prompt format is very similar to **ChatGPT**, with one user canceling subscriptions with **Gemini** ($250/month) and **OpenAI ChatGPT Pro** ($200/month) and **Grok 4 Heavy** ($3000/year).
   - One member joked that all it takes to get similar results from other chatbots is to *add a system prompt to tell it to act like an unhinged degen Discord mod, and tell it to “go and express yourself” haha.*
- **Google Gemini's daily deep think limit**: Members ridiculed Google Gemini Ultra's plan imposing **10 queries a day for $250/month**, one member calling it *very funny and very scummy*.
   - One added that even **ChatGPT pro** at $200/month gives unlimited **Office 365 Pro**, while **Claude Max** is more reasonable.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1400597993442513128)** (110 messages🔥🔥): 

> `Hermes-3 dataset, Unitree R1 robot, OpenAI's Horizon Alpha model, Quantization challenges, SmolLM and Qwen2.5` 


- **Hermes-3 Dataset Refusals Ruffle Quantization**: Members discussed whether refusals in the **Hermes-3 dataset** were purposeful or artifacts of censored models, with one member using it to compute the *imatrix* for quantization and finding unexpected refusals leading to [further dataset investigation](https://huggingface.co/datasets/NousResearch/Hermes-3).
   - The main intention was to confirm the dataset is devoid of refusals.
- **Unitree's R1 Robot Democratizes Embodied A.I.**: The community discussed the **Unitree R1 foundational robot model**, priced at **$5,900**, which offers a fully open software development kit (**Python**, **C++**, or **ROS**) for A.I. development, as showcased in [this YouTube video](https://www.youtube.com/watch?v=ljo7TjOqRzs).
   - It is an ideal tool for research teams transitioning to the next evolution of A.I.
- **Horizon Alpha Model sparks OpenAI Base Model Release Rumors**: Members discussed **OpenAI's Horizon Alpha model**, with speculation it resembles **OpenAI's** style and could be a **120B MoE** model with low activation, or possibly a **20B** model, as suggested in [this tweet](https://x.com/apples_jimmy/status/1951180954208444758).
   - There is speculation on Reddit, with [this thread](https://www.rxddit.com/r/LocalLLaMA/comments/1mepeqh/the_openai_open_weight_model_might_be_120b/) suggesting if it is **FP4** only, proper quantization would be impossible.
- **Quantization Quandaries for OpenAI's Leaked Model**: The community analyzed leaked config files indicating **OpenAI's model** is a **116.8B/5.7B MoE** model, which, when padded for GGUF, pushes it to **132.7B/6.3B**, making it difficult to quantize using methods other than **Q4_0**, **Q5_0**, **Q8_0**, and **IQ4_NL** due to the architecture's hidden size.
   - Because the hidden size of 2880 does not allow quantization to K or I quants.
- **SmolLM & Qwen2.5 Quantization Gotchas**: Discussions revealed that **SmolLM (135B/360B)** and **Qwen2.5 0.5B** have dimensions that cannot be made into K or I quants.
   - The members reported that only *o_proj* (from attention) can be quantized to K or I quants for the alleged **GPT model**.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1400563468649762909)** (4 messages): 

> `Input Tokens per Second, Prefill, Gemma, Time to First Token` 


- **Peeking into Input Token Processing**: A user inquired about resources for reasoning about **input tokens per second**.
   - Another member clarified this meant the *prefill* (just the context your using, not generating).
- **Profiling Gemma on a Laptop**: A user reported a **~50 second Time To First Token** for both 4500 and 9000 token prompts, using **Gemma** on a laptop.
   - The user is seeking comprehensive overview of that process for profiling purposes, and noted that the output tokens per second was the same across different input token sizes.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1400851409561194598)** (3 messages): 

> `OSS Model Training Script, Metaprogramming and DAG->HRM->code automation` 


- **OSS Model Training Script: Raizoken Builds!**: A public research engineer is writing a model training script with the intention of making it **OSS** right away.
   - They are trying to create good **OSS models** for natural cursor navigation but are worried about potential misuse of the model, such as scraping websites that block crawling bots.
- **Raizoken Seeks Metaprogramming Automation Advice**: A member is seeking advice on **metaprogramming** and **DAG->HRM->code automation**, noting they're already using it in their stack but are facing scaling bottlenecks.
   - They've implemented **Terraform** and **Helm** to offset this, but are struggling with cloned slaves in **Ray nodes** when they form clusters, lacking a mechanism to control the self-spawn outside of cooldowns.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1400575460483535091)** (5 messages): 

> `AnythingLLM, Neuronpedia, Data Sovereignty` 


- **AnythingLLM heralds Data Sovereignty future**: A user shared a [link to a tweet](https://x.com/AnythingLLM/status/1755265335803212271?t=ypxd85gvodugP-ksZP6Nvg&s=19) about **AnythingLLM** and declared it the future for **data sovereignty**.
- **Neuronpedia and Data Sovereignty gain traction**: The user also shared links to **Neuronpedia** and other tweets relating to **data sovereignty** from [Jack_W_Lindsey's tweet](https://x.com/Jack_W_Lindsey/status/1950952346990862502?t=JGcHUqVwZF8_GBoWV5JPcg&s=19) and [heyshrutimishra's tweet](https://x.com/heyshrutimishra/status/1950801664379953468?t=ywRLWQRNGMsoXD8eOMPV-g&s=19).


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1400851409561194598)** (3 messages): 

> `OSS model training script, Metaprogramming and DAG->HRM->code automation, Federated cycles between clones in ray nodes` 


- **OSS Model Training Script Emerges**: A public research engineer is developing an **OSS model training script** to address the lack of good OSS models for natural cursor navigation.
   - The engineer notes that websites blocking crawling bots may be scraped by new "clones" using this technology.
- **Metaprogramming Automation Bottleneck Surfaces**: A member is seeking advice on scaling issues with **metaprogramming** and **DAG->HRM->code automation**, despite using Terraform and Helm.
   - They are facing problems with federated cycles between clones in ray nodes, particularly with uncontrolled self-spawning outside of cooldown periods.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1400565376567480373)** (112 messages🔥🔥): 

> `Cline's $32M seed funding, CLI orchestration layer, Subagents and Claude Code Office Hours, Bytedance's Seed Diffusion LLM for Code, Open-License Hybrid Reasoning Models` 


- **Cline Closes $32M Funding Round**: Cline, an AI coding agent, announced a **$32 million** Seed and Series A funding round led by **Emergence Capital** and **Pace Capital** to support transparent, open-source AI tools for developers; serving **2.7 million** developers and transparent pricing with no upcharging.
   - Cline aims to empower developers by avoiding 'nerfed' products, focusing on enterprise features like access controls and centralized billing.
- **OpenAI's OS Model Leaks**: Details leaked about **OpenAI**'s upcoming OS model, **YOFO**, shortly after its config was briefly available, igniting excitement about rumored **120B** and **20B** variants.
   - A member noted that Jimmy Apples was reluctant to share all configuration details.
- **Anthropic's Production Reinforcement Learning Codebase Updated by Claude**: Anthropic merged a **22,000-line** change to their production reinforcement learning codebase, heavily written by **Claude**, sparking skepticism and discussion among users about the authenticity and safety of such a large AI-generated code change; it was largely a **json dsl**.
   - Sauers confirmed the change was real, and discussions touched on human review processes and concerns about the reliability of large AI-driven code merges.
- **Anthropic Cuts off OpenAI's API Access**: Anthropic revoked OpenAI's API access to its models, including **Claude**, citing a violation of terms of service.
   - A member noted that **OpenAI** expressed disappointment, mentioning that its API remains available to **Anthropic**, and the community discussed implications of competitive moves and blurring lines of model training.


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1400567742054011033)** (4 messages): 

> `Cline pod writeup, Latent Space Podcast, Open Source Code Agent` 


- ****Cline Podcast** Writeup Released!**: The writeup for the **Cline podcast** is now out, linked on [X](https://x.com/latentspacepod/status/1951008883163668522).
- ****Latent.Space Podcast** Features **Cline**!**: **Latent.Space Podcast** announces a new episode with **Cline**, an open-source VSCode extension that recently raised **$32 million**.
   - The episode discusses Cline's origin, the 'Plan + Act' paradigm, top community tools, and future directions, featuring guests Saoud Rizwan and Pash. The podcast is available on their [website](https://xcancel.com/latentspacepod/status/1951008883163668522) and [YouTube](https://www.youtube.com/watch?v=dQw4w9WgXcQ).


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1400554550129070171)** (86 messages🔥🔥): 

> `RAG query expansion techniques, Sentence embeddings vs. token embeddings, Cross-encoders for semantic similarity, Knowledge Graphs for information retrieval, LLMs and question-answer co-occurrence` 


- **Query Expansion Boosts RAG Performance**: Members discussed [query expansion](https://www.promptingguide.ai/techniques/query_expansion) for RAG systems, suggesting generating multiple questions from a single query.
   - Specifically, for *'what is the name of the customer'*, it was proposed to create the questions *'What is the name?'* and *'Who is the customer?'* to improve retrieval.
- **Cross-Encoders Fail Ranking Task**: Experiment using a cross-encoder with **MS MARCO** data to rank results for the question *'What is the name of the customer?'* showed poor results.
   - The expected top hit (*Customer Name*) was ranked lower than (*Definition of Customer*), with scores of -0.67 vs -1.67.
- **Fine-Tuning Retrieval Task is Key**: To control ranking quality, directly training on a retrieval task is essential, according to [this paper](https://arxiv.org/abs/2212.01349).
   - It was suggested that the optimal similarity metric is task-dependent, meaning general-purpose embeddings may not suffice for specific retrieval scenarios.
- **Gemini 2.5 Flash favors Gemma Models**: Members found that Gemini-2.5-flash consistently ranked **Gemma models** higher than other models, even some 70B models.
   - It's suspected that the **response tone** of Gemma models might be more plausible to both humans and LLMs, influencing the ranking.
- **LLMs Parallel Thinking Debated**: Discussion around [Google's Gemini 2.5](https://blog.google/products/gemini/gemini-2-5-deep-think/) and its *'Deep Think'* feature, which uses parallel thinking to deliver more detailed and thoughtful responses.
   - Some suggested the model generates multiple ideas in parallel, with parallel COT, while others believe it's higher-level orchestration of basic models and context management.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1400573062881214524)** (3 messages): 

> `The Cinema AI, Generating Movie Scenes` 


- **Generating Cohesive Movie Scenes with TheCinema AI**: The channel will be reviewing [TheCinema AI](https://thecinema.ai/), an interesting research project focused on generating movie scenes that maintain **cohesion** with each other, according to the [arxiv paper](https://arxiv.org/html/2507.18634v1).
- **TheCinema AI: Generating Movie Scenes**: This research explores methods for generating movie scenes that are cohesive, as detailed in the [TheCinema AI project](https://thecinema.ai/) and its corresponding [arXiv paper](https://arxiv.org/html/2507.18634v1).


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1400557372002271293)** (4 messages): 

> `NVIDIA Chips, Nintendo Switch` 


- **Experts Expose NVIDIA Chip Capabilities**: Experts in the American AI sector allegedly revealed that **NVIDIA's computing chips** have technologies for *tracking and geolocation* and *remote shutdown*.
   - A member called for [citation](https://citation.needed) since the source was the *State Internet Information Office of the PRC*, calling it an *absurd and feeble leverage attempt*.
- **Government restrictions are like Nintendo Switch**: A member said that government-imposed restrictions are just like the **Nintendo Switch**.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1400575531170402304)** (27 messages🔥): 

> `Audio pause timing in slide changes, Portuguese language support for explainer videos, NotebookLM for personalized podcasts, Canvas infographics from Perplexity Deep Research` 


- **Delay slide changes for smoother audio**: Users suggested adding an extra half-second pause before each slide change to avoid abrupt audio truncation in explainer videos.
   - This small adjustment could significantly *improve the viewing experience* by allowing audio to fade out naturally.
- **Portuguese Explainer Videos: Unofficial Support Available**: A user confirmed that while Portuguese isn't officially supported for explainer videos, they were able to force it to work.
   - Another user reported *mixed results*, with audio in Portuguese but slides sometimes remaining in English, while another suggested tweaking the prompt to specify both audio and video tracks.
- **NotebookLM + Gemini: Podcast Powerhouse?**: A user shared a workflow of asking Gemini a question and then feeding the answer into NotebookLM to create personalized podcasts.
   - They posted links to demonstrate the process: [NotebookLM](https://notebooklm.google.com/notebook/aa55ef62-9230-4b15-be5e-a6954247470c/audio) and [Gemini Share](https://g.co/gemini/share/11437d9da04c).
- **Canvas Infographics from Perplexity via NotebookLM?**: A user shared a process of creating canvas infographics directly from a **Perplexity Deep Research** report.
   - While not directly related to NotebookLM, they suggested it as a potential step to *leverage NotebookLM's power* with detailed outputs from other models, also adding that *Google can and SHOULD do better* than current video overviews, noting a current AI output.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1400554423864000664)** (65 messages🔥🔥): 

> `Offline access to NotebookLM studio material, Video overview rollout issues, NotebookLM and Gemini API for custom RAG pipeline, Comet browser extension for NotebookLM, Audio Overviews language and duration limitations` 


- ****NotebookLM Goes Offline for Road Warriors****: Users are seeking ways to save **NotebookLM studio material** for offline access during travel without constant internet connection.
   - One user mentioned downloading audio to an iPad and adding it to PowerPoint slides with family photos.
- ****Video Overview Vexation: Pro Users Ponder Missing Preview Perks****: Several **Pro account users** report not having access to the **video overview feature**, despite upgrading and others with free accounts having access.
   - A user who briefly had video access lost it after refreshing the page, suggesting ongoing rollout issues.
- ****RAG Dreams: User Schemes Custom NotebookLM with Gemini Power****: A user is considering using **Gemini embedding 001** and **Gemini 2.5 models API** to create a custom multi-hop, multi-step reasoning **RAG pipeline** for documents.
   - They aim to surpass **NotebookLM's** capabilities, citing limitations such as the **300-file limit**, lack of transparency in workflow, and limited system instructions, hoping to *plagiarize their work*.
- ****Comet Extension Could Catapult NBLM into Orbit****: Users discussed **Comet**, a browser extension that can access tabs/history/bookmarks and control the browser, and its potential integration with **NotebookLM** for source finding.
   - The suggestion was raised that **Comet** could potentially code an extension to dynamically add sources to **NotebookLM**.
- ****Spanish Audio Overviews Still Short and Sweet?****: A user inquired about why **Audio Overviews** in Spanish remain short in duration.
   - A workaround was suggested: *switch it to English, change the duration, then prompt it to do it in Spanish*.


  

---


### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1400876073763213554)** (1 messages): 

> `Attention probes, Linear probes, Overfitting, Optimization issues` 


- **Attention Probes: A New Way to Classify Hidden States**: EleutherAI conducted experiments with **attention probes**, tiny neural networks with attention trained to classify the hidden states of transformers.
   - Despite expectations, their performance was mixed, sometimes underperforming standard **linear probes** due to **overfitting** and **optimization issues**, as detailed in their [blog post](https://blog.eleuther.ai/attention-probes/).
- **Attention Probe code open sourced**: EleutherAI has open-sourced the code for their attention probes experiments, inviting others to explore and refine the approach.
   - The repository is available on [GitHub](https://github.com/EleutherAI/attention-probes/), with the hope that further investigation may uncover potential improvements.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1400692396144070698)** (11 messages🔥): 

> `LLMs on low-power edge devices offshore, Gemini-2.5-flash biased ranks for gemma responses, OpenAI open source model config, MLA vs MHA generalization` 


- **Low-Power LLMs Brave Offshore Deployment**: A member is running **LLMs** on low-power edge devices offshore, focusing on seabed mapping, environmental monitoring, and autonomous systems.
   - The current use-cases involve **mission planning**, **anomaly detection**, and **smart data compression**, rather than scientific modeling due to latency and bandwidth challenges.
- **Gemini-2.5-flash Shows Favoritism for Gemma Models**: A member using **Gemini-2.5-flash** to rank responses from various LLMs noted consistently biased ranks for **Gemma** responses.
   - The member speculates about *family bias* or the possibility that **Gemma3** models are simply superior.
- **OpenAI's Forthcoming Open Source Model Config Leaked!**: A member shared a [config](https://gemini.google.com/share/3b63a193539c) for the forthcoming **OpenAI open source model**, including specs like **36 hidden layers**, **128 experts**, and a **201088 vocab size**.
   - Other members congratulated those whose work was adopted by **OpenAI** in this model.
- **MLA Triumphs Over MHA in Generalization Debate**: A member asked whether **MLA** or **MHA** is better in terms of generalization, while pretraining a **300m parameter model** on textbook quality data, using **RoPE**.
   - Another member recommended using **MLA** (Multi-level Attention) as the preferred architecture.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1400589521535373434)** (41 messages🔥): 

> `RoPE is near optimal, Weight tying is bad, semantic search and RAG` 


- ****NovelAI** Reveals RoPE Research**: NovelAI research has been published [here](https://research.novelai.net/rope/), experimenting with golden ratio in RoPE as an optimization target.
   - The punchline is *some math and experiments that are only interesting to theorists and have no practical applications*.
- ****RoPE's** Optimality and General Form**: A blog post [here](https://nor-blog.pages.dev/posts/2025-07-28-deriving-rope/) argues that RoPE is near optimal if one tries to derive it.
   - The general form for **N dimensions** requires projecting positions along incoherent and uniform directions, though this *doesn't have much practical significance*.
- ****Weight Tying** Bashed as Bad Practice**: A member stated that *weight tying is a universally bad practice that being said* and also *a terrible inductive bias!*.
   - They argued that **weight tying** is the cause of a lot of inefficiency and instability and *doesn't even make mathematical sense*.
- **Semantic Search Troubles and RAG Alternatives**: A member is struggling with semantic search and raised a question about the liability cap.
   - Another member suggested to use **RAG** like approach rather than semantic search, and also said that a lot of *domain specific engineering needs to go into semantic search to work properly*.


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1400583667998527540)** (1 messages): 

> `EleutherAI Website PR, Tensor Program papers, Yang et al paper` 


- **EleutherAI Website Gets a Facelift**: A member thanked another for their article and opened a [PR](https://github.com/EleutherAI/website/pull/145) with some fixes to the EleutherAI website.
   - The member requested careful review, mentioning they hadn't read the **Tensor Program papers** yet and may have made mistakes, especially in the math appendix around equations 15-18.
- **Seeking Clarity on Tensor Program Equations**: A member who submitted a PR is seeking guidance on locating specific equations (**15/17**) within the **Yang et al paper**, indicating a need for clarification on the mathematical underpinnings of the Tensor Program.
   - This suggests a collaborative effort to ensure the accuracy and validity of the website's content concerning the Tensor Program.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1400837578130981006)** (5 messages): 

> `HF transformers update, Llama & Qwen residual streams, Attention Probes Work, NIAH datasets` 


- **HF Transformers' Llama Layers Launch Residual Streams**: In **HuggingFace transformers 4.54**, **Llama & Qwen layers** now return residual stream directly (not tuple), which may affect users of `nnsight layer.output[0]`.
   - A member warned that using `nnsight layer.output[0]` will get you the 1st batch element only, not full residual stream, a bug spotted thanks to [nnterp tests](https://butanium.github.io/nnterp).
- **Attention Probes Produce Promising Probing Progress**: Members discussed promising attention probes, but were surprised by the mixed results, based on [attention probes work](https://link-to-attention-probes-work).
   - One member suggested probing with a suffix to consider what you're trying to probe for, asking the LM to consider what you're trying to probe for (e.g. *Is the above statement true?*).
- **NIAH Datasets' Last-Token's Talent**: Members stated that the underperformance of attention probes is mainly coming from the **NIAH datasets**, which are constructed so that the thing being classified comes right at the end of the sequence.
   - This would explain why last-token probing works well there; in that case, one should train both a linear probe and an attention probe.
- **McKenzie Probing Papers Promote Prompting Progress**: The probing paper [McKenzie et al. 2025](https://arxiv.org/abs/2506.10805v1) considers prompting the model to give an answer as a baseline (with results lower than for probes), but not prompting to improve probing.
   - It's possible this would be an improvement on the datasets we considered where mean probes outperform last-token probes, and worth investigating.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1400755680994136134)** (1 messages): 

> `` 


- **User Finds Potential Solution**: A user expressed they may have found a way to solve their problem and will send a message if it doesn't work out.
- **Awaiting User Feedback**: The conversation is currently pending further updates from the user regarding the success of their solution.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1400600571442106388)** (14 messages🔥): 

> `MIT Collaboration on LLM Training, Containerization Issues, CUDA Issues, DeepSpeed checkpoint inspection` 


- **MIT Collabs on OLMo2 & DCLM Training**: MIT and EAI are collaborating on LLM training, starting with **OLMo2 1B** or **DCLM 1B** to familiarize themselves with the pipeline, initially focusing on pretraining, but with plans to incorporate **SFT** and safety alignment later on.
- **Container install faces tricky Permissions Error**: A user encountered permission errors during containerized installation using Apptainer, specifically related to `setgroups` failures, and was advised to try `apptainer exec --fakeroot your_image.sif ...` as a potential workaround.
   - Another member suggested using conda environments directly on the host if the container issue persists, based on their experience with Slurm-based HPC clusters.
- **CUDA configuration challenges in Conda env**: After switching to a conda environment, the user encountered **CUDA** issues, which they believe have been resolved, and they are now working on installing **flash-attention** and **TE**.
   - The user asked for specific test commands to verify the environment setup after installing **flash-attention** and **TE**.
- **DeepSpeed Checkpoint Inspection Woes**: A user reported that `inspect_ds_checkpoint` from the experimental branch doesn't support `pipe_parallel_size=0`, causing validation checks to fail due to the absence of `layer_*` files in the checkpoint directory.
   - They also inquired whether it's fundamentally impossible to scale from **(4 nodes x 8 GPUs)** to **(8 nodes x 8 GPUs)** with `pipe_parallel_size=0`, `model_parallel_size=1`, and zero stage 1.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1400559583528747048)** (61 messages🔥🔥): 

> `Aider Appreciation, SGLang and Qwen Speed, 4090 Mobo and Case, Aider vs Other Tools, Claude Code Context Limits` 


- ****Aider** Still Reigns Supreme**: One member expressed their appreciation for **Aider**, noting its superior combination of control and freedom compared to other tools, estimating **one week of programming work** was done in a single day for **$2** with DeepSeek.
   - Another user echoed this sentiment, saying, *"Aider rules so hard"*.
- ****SGLang** and **Qwen** Hit Ludicrous Speed**: A member reported achieving **472 tokens/s** with **sglang** and **Qwen 0.6B Q8** on LM Studio using an **RTX 4090**, whereas on regular lmstudio it only goes **330 t/s**.
   - Another user expressed interest in replicating this local-only flow, noting **vllm's** comparatively slow performance on their **4090** versus Ollama and was very interested in trying llama.cpp.
- **Mobos for Multi-GPU Setups Explored**: The discussion pivoted to hardware setups, with one member recommending this [MSI motherboard](https://www.bhphotovideo.com/c/product/1864084-REG/msi_meg_x870e_godlike_e_atx.html) for dual **3090s**, housed in a Fractal North XL case.
   - Others chimed in with their setups, including servers with **3 L4s** and **T40s**, and different cases like the **Meshify2**.
- **Aider versus Windsurf versus Cursor**: One user expressed disappointment with **Aider**, **OpenHands**, and **Chode-Pilot**, preferring **Windsurf** and **Cursor**.
   - They speculated the "sauce" might be in giant closed models running on beefy hardware, expressing a need to try **QWEN3** after unsatisfactory experiences with **Devstral** and **Codelamma**.
- ****Claude Code's** Context Window Caveats**: Members discussed the performance of **Claude Code** with one mentioning that it works well without RAG and mentioning that Claude, unlike other frontier models, suffers greatly from high context token count.
   - It was noted that quality noticeably degrades beyond **64k tokens**, an issue less pronounced in **o3**, and best handled by **Gemini 2.5 Pro**. Others pointed out *the system prompt alone eats a significant portion of the context window*.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1400608522361770119)** (10 messages🔥): 

> `Qwen3 30B A3B Coder Benchmarking, LM Studio Usage, llama.cpp server + docker aider benchmark, aider + claude-code max subscription integration, Gemini 2.5 Pro` 


- **Benchmarking Qwen3 30B locally in LM Studio**: A member wants to benchmark 8 different quants of **Qwen3 30B A3B Coder** locally using **LM Studio** in an easy way.
   - Another member suggested using *llama.cpp server + docker aider benchmark on the same computer*, and referred to a writeup involving **Gemini 2.5 Pro** that details the steps to get it working.
- **Aider integrates with Claude-Code Max Subscription**: A member inquired whether *aider* can be used with **claude-code max subscription integration** to tap into the new thinking model.
   - They also asked if the command *aider --model gemini/gemini-2.5-pro-preview-06-05 --thinking-tokens 32k* is an old way of thinking and if anyone had success running aider with Claude code.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1400559945786589275)** (43 messages🔥): 

> `Security MCP Check Tool, PayMCP Payment Layer, PageRank for MCP Servers, MCP Eval Platforms, Gateway for Agent Tool Search` 


- ****Security MCP Check Tool** unveiled**: A member shared a [GitHub repo](https://github.com/minte-app/security-mcp-check) for a **security MCP check tool**, seeking feedback.
   - This could provide a way to check your own server for vulnerabilities, but note that no further explanation was given.
- ****PayMCP** Payment Layer emerges**: A member announced the development of **PayMCP**, a payment layer for **MCP**, and is looking for collaborators and early users, providing [Python](https://github.com/blustAI/paymcp) and [TypeScript](https://github.com/blustAI/paymcp-ts) implementations.
   - This new tool promises to allow MCP servers to easily accept payments, though it is unclear what payment options it supports.
- ****PageRank for MCP Servers**: A new Search Tool**: A member inquired about the existence of **PageRank** implementations for **MCP** servers or tools, aiming to rank servers by utility rather than just name or description.
   - Another member shared a [repository of MCP tools](https://github.com/YogiSotho/mcp-tools-collection), and mentioned the [MCP registry](https://github.com/modelcontextprotocol/registry) as potentially helpful resources.
- **MCP Eval Platforms sought**: A member sought information on **MCP eval platforms** that generate different agents in various situations to test **MCP** servers.
   - Another member indicated they are developing a gateway for agents to search for tools and plan to have something available by Sunday.
- **Guidance for grasping MCPs**: A member requested assistance in understanding and using **MCPs** in their workflow, offering to pay for someone's time to help.
   - This highlights the complexity and learning curve associated with adopting **MCPs** for new users.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1400893140394578104)** (1 messages): 

> `JSON MCP Server, LLM Efficiency with JSON, Schema Generation for JSON, Token Savings` 


- ****JSON MCP Server** for LLMs Launched**: A new **JSON MCP Server** has been created to aid **LLMs** in efficiently parsing large and complex **JSON** files, such as **Excalidraw exports**; see the [GitHub repo](https://github.com/kehvinbehvin/json-mcp-filter).
   - The tool uses **schema generation** to first understand the structure of the **JSON** and then extract only the necessary data, saving tokens and context.
- **LLMs parse JSON files more efficiently**: The main goal of this tool is to help **LLMs** parse large and tangled JSON files more efficiently.
   - It saves tokens and context by extracting only the data you need.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1400575926592344246)** (8 messages🔥): 

> `Hylo Programming Language, Value Semantics, Halide, Scala 3/Scala Native, Heterogenous Programming` 


- ****Hylo** Language Heats Up**: A member inquired about the **Hylo** programming language ([https://www.hylo-lang.org/](https://www.hylo-lang.org/)), highlighting its approach to memory safety through **value semantics** and scheduling, drawing parallels with **Halide**.
   - It was noted that the team sits in the same "heterogenous pl for the 21st century" bucket as **Mojo**.
- **Hylo's value semantics and concurrency**: Members stated that the **Hylo** team is still hammering their **value semantics** and **concurrency** down, the hope and roadmap though is that value semantics squares nicely with scheduling, tiling, vectorizing.
   - The **Hylo** team is from Adobe STL and has experience hacking on **Halide**.
- ****Scala** team member is on Hylo?**: A member mentioned that the person responsible for **Hylo** is currently working on **Scala 3/Scala Native**.
   - Other members stated the leads come from **cpp** and **Swift**


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1400862107377074339)** (1 messages): 

> `Triton Kernel AI Agent, GEAK benchmarks` 


- **AMD Introduces GEAK & Triton Kernel AI Agent**: AMD introduced the **GEAK benchmarks** and **Triton Kernel AI Agent** in their paper [GEAK: INTRODUCING TRITON KERNEL AI AGENT & EVALUATION BENCHMARKS](https://arxiv.org/abs/2507.23194).
- **Dive into AMD's Kernel AI Agent**: Explore AMD's novel approach to **AI-driven kernel optimization** using their new **Triton Kernel AI Agent**.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1400653536185942016)** (4 messages): 

> `Profiling Copilot, __launch_bounds__ fix for register count issue, setmaxnreg ignored due to extern call` 


- **__launch_bounds__ setting launches CUDA fix**: A user fixed an issue where the compiler couldn't determine register count at entry by passing `minBlocksPerMultiprocessor` to `__launch_bounds__`, setting `maxThreadsPerBlock=128*3` and `minBlocksPerMultiprocessor=1`.
   - They noted they're *not sure how that fixes the problem exactly*, but are *happy to move forward*.
- **`setmaxnreg` meets incompatibility issues**: The `setmaxnreg` setting is still being ignored, now due to a different problem related to compatibility with an `extern` call, as indicated by the message: `ptxas info : (C7506) Potential Performance Loss: 'setmaxnreg' ignored to maintain compatibility into 'extern' call.`
   - A member asked if the kernel is calling an `'extern'` function defined in a separate compilation unit.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1400611087044579428)** (1 messages): 

> `CheckpointPolicy with Custom Kernels, Functorch API` 


- **CheckpointPolicy for Custom Kernels**: A member inquired about documentation on implementing **CheckpointPolicy** with custom kernels in Torch, specifically for fused **MLP**.
   - They asked if it's feasible to use it within the **Functorch API**.
- **Functorch and Custom Kernels**: The user wants to integrate a custom kernel, such as a fused **MLP**, into the **Functorch API** while using **CheckpointPolicy**.
   - They are seeking guidance or documentation on how to achieve this integration effectively.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1400600521634480232)** (1 messages): 

> `MI300X FP8 benchmarks on AMD, AMD MI300X vs H200 vs B200, FP8 Data Parallel Benchmarks` 


- **MI300X Benchmarks Leave H200 Behind**: A user inquired about experiences with new [MI300X FP8 benchmarks](https://eliovp.com/mi300x-fp8-data%E2%80%91parallel-benchmarks-8-64-gpus-h200-left-behind-b200-within-reach/) on AMD hardware.
- **FP8 Performance on MI300X**: The benchmarks compare **AMD's MI300X** with **NVIDIA's H200** and suggest the MI300X outperforms the H200 in certain FP8 data-parallel tasks.
   - The results indicate **MI300X** performance is getting close to **NVIDIA's B200**.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/)** (1 messages): 

celis1702: thank you both so much for your clear explanations and for sharing these details!
  

---


### **GPU MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1400694013803106367)** (2 messages): 

> `JIT function, JAXPR printing, Static arguments` 


- **JAXPR printing trouble**: A user encountered trace-time errors when attempting to print the **JAXPR** for a **JIT** function using static arguments.
   - The user was attempting to use `jax.make_jaxpr(jit_func)(1, 2)` but was running into errors.
- **Static Arguments and JIT Compilation**: The user's problem revolves around using `static_argnames` with `jax.jit` and then trying to inspect the resulting JAXPR.
   - Understanding how static arguments affect tracing and compilation is key to resolving the trace-time errors.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1400861221586210836)** (2 messages): 

> `Agreement, Acknowledgement` 


- **Affirmative Confirmation**: User @sshkr16 stated *"I am yeah"*, signalling agreement or confirmation within the conversation.
   - Another user, ali_8366, responded with *"Nice !"*, acknowledging and positively affirming the initial statement.
- **Positive Acknowledgment Received**: ali_8366's response of "Nice !" indicates a positive reception to @sshkr16's affirmation.
   - This simple exchange highlights mutual understanding and agreement within the channel.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1400585395363643573)** (2 messages): 

> `Profiling llama.cpp with rocprofilerv3, AMD machine for GGUF` 


- **rocprofilerv3 Profiling Woes with Llama.cpp**: A member inquired about using **rocprofilerv3** to profile **llama.cpp**, noting successful profiling of PyTorch code but issues with llama.cpp on **MI50s** with **ROCm 6.3.3**.
   - They were curious if the issue was specific to their setup.
- **AMD Hardware Inquiry for GGUF Execution**: Another member responded, expressing that they hadn't tried profiling **llama.cpp** and inquired about the specific AMD machine being used for running **GGUF** models.
   - They wanted to know the hardware setup for GGUF inference.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1400869779694419998)** (1 messages): 

> `C/ua Hiring, AI Agents Infrastructure, Founding Engineer Roles` 


- **C/ua Seeks Talent in SF and Spain**: **C/ua** is hiring Founding Engineers in San Francisco and Spain (Remote or Madrid hybrid) to build the infrastructure for general-purpose AI agents.
   - They are backed by **Y Combinator** and are developing open-source tools used by thousands of developers.
- **C/ua Building AI Agent Infrastructure**: **C/ua** focuses on infrastructure for AI agents to safely use computers and applications at scale.
   - The roles involve building secure runtimes, container orchestration, developer APIs, and OS-level virtualization.
- **Founding Engineer Roles at C/ua**: **C/ua** is looking for Founding Engineers passionate about system safety, reproducibility, and dev experience to shape how agents run at scale.
   - Interested candidates can find more details in the [San Francisco job post](https://ycombinator.com/companies/cua/jobs/dIskIB1-founding-engineer-infra-agent-systems).


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/)** (1 messages): 

tonic_1: really glad i was nosey enough to check this convo out 🙂 super excited about this 🙂
  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1400557751641313300)** (7 messages): 

> `README update on Resource vs Prototype, RCON client disconnects, Blueprint VQA pipelines` 


- ****Resource or Prototype** in README?**: A member inquired whether the README is up-to-date regarding the use of **Resource** vs **Prototype** for finding patches, specifically questioning whether `position=nearest(Prototype.IronOre))` should be `Resource.IronOre`.
   - Another member confirmed the likelihood, noting that *"That part of the README was made by claude in cursor"*.
- ****RCON client disconnects**, limiting testing**: Testing is being throttled because the **RCON client** is disconnecting, as demonstrated by the error *"The RCON client is currently not connected to the server"*.
   - This issue prevents complete trajectories.
- ****VQA Pipelines** for Blueprints Completed!**: A member reported the completion of **VQA pipelines for blueprints** and is now focusing on data augmentation.
   - The augmentation methods include **rotations**, **flips**, and **sub-section chunks**, aiming to multiply the available blueprints by 10-15x.


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1400572352168984789)** (6 messages): 

> `picocuda compiler, elements graph data structures, scalar compilation, GPU compilation, tinygrad's AMD GPU driver` 


- **Picocuda & Elements Projects Gain Momentum**: Progress is being made on the [picocuda](https://github.com/j4orz/picocuda) compiler and [elements](https://github.com/j4orz/elements) graph data structures projects.
   - The focus is now on diving into GPUs, following wrapping up scalar compilation for the [Zero to Hero](https://j4orz.ai/zero-to-hero/) textbook.
- **GPU Compilation Textbook to Follow GPUCC Paper**: The textbook will roughly follow the [GPUCC paper](https://dl.acm.org/doi/pdf/10.1145/2854038.2854041) from CGO '16, extending the big red intermediate language (BRIL) from sampsons cs6120, which is a mini LLVM ([BRIL webpage](https://www.cs.cornell.edu/~asampson/blog/bril.html)).
   - The author suggests building up both scalar and vector compilation incrementally with a small layer of runtime orchestrating the host and device code.
- **AMD GPU for Open-Source Textbook**: A **7900xtx** or **9070xt** will be purchased for development, using **tinygrad's AMD GPU driver** over USB.
   - AMD was chosen because it's open source, aligning with the textbook's target audience of hackers and tinkerers.
- **Porting llm.c to AMD's HIP**: The goal is to build up to **Karpathy's llm.c** (forked and modified to **AMD's HIP**).
   - Contributors are welcome, particularly with the C compiler at [picocuda](https://github.com/j4orz/picocuda) and the graph data structures at [elements](https://github.com/j4orz/elements).
- **Graph Algorithms Needed for Host Code**: The two main graph algorithms needed for host code are dominators for the middle (`opto`) and graph coloring for the backend (`cgen`)'s register allocator.
   - The author recommends lengauer-tarjan for dominators (like rustc) and briggs-chaitin-click for the register allocator (like hotspot's C2).


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1400849332994838631)** (4 messages): 

> `DTensor, Basic Parallelism Schemas, Shape Rotation, DTensor Problems, Marksaroufim visualizations` 


- **DTensor schema continuation planned**: Members are planning to continue working on **DTensor** and **basic parallelism schemas**.
   - The session is scheduled for Sunday around **8 PM CEST**, with the possibility of extending it if necessary.
- **Shape Rotation task in progress**: One of the members plans to focus on **shape rotation**.
   - The goal is to explore and implement techniques for efficiently manipulating the shapes of tensors.
- **Marksaroufim visualizations inspire DTensor problems**: Members will be exploring new **DTensor problems** by using [Marksaroufim's visualizations](https://www.youtube.com/@marksaroufim).
   - The aim is to leverage these visualizations for insights into potential challenges and solutions in **DTensor** development.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1400589363934527620)** (26 messages🔥): 

> `Flux Krea model, Synthetic Datasets with HF jobs, AMD GPU for EM image segmentation, Llama CP model path, Gemini-2.5-flash bias` 


- ****Flux Krea** new model released!**: A new **Flux Krea** model is out with *much more detail*, works with most lora on base.dev, [available here](https://huggingface.co/Clybius/FLUX.1-Krea-dev-scaled-fp8).
   - According to initial reports, **NSFW** is *not possible*.
- ****Gemini 2.5 Flash** possibly favors **Gemma3****: A member has been trying to use **Gemini-2.5-flash** to rank responses from various LLMs, and has been consistently seeing **Gemma3** models ranked higher than others even some **70B** models.
   - Another member thinks that there is some bias, and **Gemma 3** is one of the better models and *the default weights are also well done*.
- ****HuggingFace Ultrascale** book mirrors blogpost?**: A new member asked if the contents of the **HF ultrascale book** are the same as the blog, requiring **HF pro subscription**.
   - Another member confirmed that *it's 246 pages*, possibly the same as the blog post with lots of images, linking to [Julien Chaumond's tweet](https://x.com/julien_c/status/1951277984532279794).
- **Synthetic Datasets with **HF jobs** documented**: A member was asking how to create synthetic datasets with **HF jobs**.
   - Another member offered [hf jobs docs](https://huggingface.co/docs/huggingface_hub/en/guides/jobs), [script](https://ray.so/O8JjQ6X), [dataset](https://huggingface.co/datasets/dvilasuero/nemotron-kimi) and [config](https://huggingface.co/datasets/dvilasuero/nemotron-personas-kimi-questions/raw/main/config.yml) as an example.
- **Volume Seg Tool built on **AMD****: A member released one of the **SOTA tools for EM image segmentation** using a **10 years old GCN AMD GPU** that has no tensorcore and not even supported on lastest ROCm, [available here](https://github.com/fgdfgfthgr-fox/Volume_Seg_Tool).
   - They mentioned that achieved nearly a **5x-10x reduction** from other neural models.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1400825233887465542)** (2 messages): 

> `Note-taking tools, Remnote` 


- **Note-Taking App Disclosed: Remnote**: A user inquired about the note-taking tool being used, and the response pointed to [Remnote](https://www.remnote.com/).
   - Remnote is a **knowledge management tool** that integrates note-taking with spaced repetition learning.
- **Remnote: More Than Just Notes**: Discussion highlighted [Remnote](https://www.remnote.com/) as a **versatile platform**.
   - It combines traditional note-taking with features like **spaced repetition** to enhance learning and retention.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1400811422899896330)** (2 messages): 

> `AgentUp, Emergence AI, LongMemEval Benchmark` 


- ****AgentUp** Rockets onto the Scene!**: The [AgentUp](https://github.com/RedDotRocket/AgentUp) project was highlighted.
   - It seems to be gaining traction as a noteworthy agent framework.
- ****Emergence AI** Claims SOTA in Memory!**: **Emergence AI**'s new architecture achieved [SOTA](https://www.emergence.ai/blog/emergence-is-the-new-new-state-of-the-art-in-agent-memory) on the **LongMemEval benchmark**.
   - The benchmark is used for evaluating long-term memory in AI agents.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1400846134766862366)** (3 messages): 

> `smolagents.js, CodeBoarding, Qwen3-30B-A3B-Instruct-2507` 


- **Smolagents ported to JavaScript!**: A member released a **TypeScript** port of **smolagents** called **smolagents.js**, available on [GitHub](https://github.com/yusuf-eren/smolagents.js) and [npm](https://www.npmjs.com/package/smolagents.js).
- **CodeBoarding released!**: A member released **CodeBoarding**, an open-source project that uses static analysis + LLMs to generate interactive diagrams of **Python** codebases, available on [GitHub](https://github.com/CodeBoarding/CodeBoarding).
- **Qwen3 refuses questions no more!**: A member posted about tweaking **Qwen3-30B-A3B-Instruct-2507** to stop refusing even blatant questions, available on [HuggingFace](https://huggingface.co/pszemraj/Qwen3-30B-A3B-Instruct-2507-abliterated).


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/)** (1 messages): 

cakiki: <@570737726991761409> please don't promote paid content in the server
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1400694296440733776)** (2 messages): 

> `Discriminator Learning Rate, GAN Loss Issues, Debugging GANs` 


- **Lowering Discriminator Rate Debugs GANs**: A member suggested lowering the **discriminator learning rate** to a very low value to observe loss changes, which can help pinpoint issues in **GAN** training.
   - Another member inquired about how much lower they should go, noting their current rate is at **1e-5**.
- **Fine-Tuning GAN Learning Rates**: The discussion centered around techniques to debug **Generative Adversarial Networks (GANs)** by manipulating the discriminator learning rate.
   - The goal is to identify whether the discriminator's loss collapsing to **0** is due to an imbalance in learning rates.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1400700923663093811)** (2 messages): 

> `Llama 4 Access, Qwen Model, DeepSeek-R1` 


- **Llama 4 Access Blocked!**: A member reported being **unable to access Llama 4** while attempting to run *dummy_agent_library.ipynb* on Colab.
   - Another member suggested substituting with a **Qwen model** or **DeepSeek-R1** as viable alternatives.
- **Substitute Models to the Rescue!**: Since **Llama 4** access requests are getting rejected, use **Qwen** or **DeepSeek-R1** as a replacement.
   - These models should work OK as a substitute.


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1400583118104039715)** (21 messages🔥): 

> `Cohere API context window size discrepancy, HackRx 6.0 AI hackathon Rate Limit, Cohere Enterprise Plan, Cohere website login error, Cohere Support Team introduction` 


- **Context Window Size Debate: 32k or 128k?**: A user pointed out a discrepancy between the **Hugging Face model card (32k context)** and **API docs (128k context)**, leading to clarification that it's **128k in** and **8k out**.
   - The team acknowledged the issue and promised to update the Hugging Face model card soon.
- **Team Patriots Seek Rate Limit Relief**: A student team, **Team Patriots**, requested a temporary rate limit increase for the **HackRx 6.0 AI hackathon** due to being blocked by the **10 calls/minute trial key limit**.
   - A Cohere team member granted them permission to create multiple accounts and cycle the keys to overcome the limit.
- **Startup Eyes Cohere Enterprise**: A startup, loving Cohere's Reranker implementation, inquired about an **Enterprise plan** to handle exceeding the **1000/min limit** for the production API.
   - They were directed to email details about their use case and request profile to support@cohere.com and varun@cohere.com for secure assistance and connection with the right folks.
- **Login Error Causes Headaches**: A user reported an error when signing in on the **Cohere website**, specifically related to a **CORS policy** blocking access during the onboarding process.
   - No immediate solution was provided in the chat.
- **Cohere Support Team Gives Warm Welcome**: Varun, a **Technical Support Engineer** at Cohere, introduced himself and provided guidance on where to post for general support and API-specific discussions.
   - Newcomers were encouraged to join **Cohere Labs 🧪** a dedicated Discord community for research, at [https://cohere.com/research](https://cohere.com/research).


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/)** (1 messages): 

kaludi: Is there something going on with the API? We are getting multiple timeouts for our queries
  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1400730205450014751)** (6 messages): 

> `Samsung Biologics AI Architect, AI Developer with LLM Workflows, Dell's Engineering Technologist, Mobile & JS-fullstack AI Application Developer` 


- **Samsung's AI Architect arrives!**: An AI architect from **Samsung Biologics** introduced themself, focusing on integrating **AI methods and tools** to address business needs and highlighting a private **LLM service with RAG** for internal use.
   - They are eager to engage in conversations related to **biopharmaceutical or biological challenges**.
- **LLM-focused AI Developer Joins**: An AI developer specializing in **LLM workflows, agent-based tools, and MCP integration** introduced themself, noting experience in building **AI sales assistants and RAG pipelines** using **LangChain and FastAPI**.
   - Their primary tech stack includes **Python and Node.js**, and they are open to collaborations and contract work.
- **Mobile AI Application Developer says What's Up!**: An **AI application developer** with mobile & js-fullstack experience introduced themself.
   - No additional information was provided.
- **Dell's AI Research at Cohere's Door**: An Engineering Technologist from **Dell** working mostly with **AI research** introduced themself from Brazil.
   - They are here to connect and learn.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1400565757355753552)** (17 messages🔥): 

> `DM spam, Wide research, Cloudflare issues, Manus AI, Daily refresh credits` 


- **User complains of DM spam**: A member reported receiving DM spam and requested an admin to perma-ban the user.
   - No action was taken during the period, and the user who sent the spam remained unaddressed.
- **Users test out Wide Research Platform**: A member inquired about initial takes on using **Wide Research**.
   - No reviews of **Wide Research** were given.
- **User unable to setup Cloudflare virtual environment**: A member is experiencing issues configuring a virtual environment within **Cloudflare**.
   - The setup keeps getting stuck on **Cloudflare**, preventing them from completing the virtual environment configuration.
- **Daily refresh credits cease functioning**: A member reported that daily refresh credits are no longer working.
   - Another user mentioned having their account suspended despite not breaking any rules, indicating possible issues with the platform's credit and account management.
- **Layoffs possibly impact refunds**: A member pointed out recent layoffs and suggested the user probably won't get their money back.
   - The comment implies that recent layoffs at the company may impact the ability to process refunds or resolve financial issues.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1400874884271313083)** (2 messages): 

> `LlamaIndex, Novita Labs, Gemini Live` 


- **LlamaIndex & Novita Labs Unite!**: The [LlamaIndex tweet](https://twitter.com/llama_index/status/1951315242904068483) announces the use of **LlamaIndex** with **Novita Labs** model inference capabilities.
   - They provide diverse data source connections and data transformation into vector embeddings.
- **Gemini Live Now Speaking TypeScript**: The [LlamaIndex tweet](https://twitter.com/llama_index/status/1951342252346974431) announces **Gemini Live integration** available in **TypeScript**.
   - A demo shows how to set up and run a simple terminal chat.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1400596216693129216)** (13 messages🔥): 

> `Agentic AI Code Assistance, Git-Style Branching for LLM Conversations, LlamaIndex Parsers for PDFs and PPTs, AI+Blockchain for on-chain AI agents` 


- **LLM Web3 Engineer Availabile for Hire**: A Senior AI & Blockchain Engineer shared his experience building **on-chain AI agents** for trading, media automation, and autonomous governance using **Eliza OS**, **LangGraph**, and custom toolchains.
   - He has deep experience across **Base**, **Solana**, **Berachain**, **Sui**, **Aptos**, **HBAR**, **EVM chains**, and cross-chain systems.
- **Craving a Local Agentic AI Code Assistant**: A member inquired about local agentic AI code assistance tools, similar to **Cursor editor**, that can run locally.
   - Other members suggested that there are many options on GitHub, but the original poster expressed that **most options have dependency issues** or lack agentic features.
- **Git-Style Branching makes conversation trees**: A member is testing a system where every message is a node, enabling branching off anywhere in the conversation tree to create a new context path, detailed in [their blogpost](https://gupta-aniket.github.io/Mobile-developer/hire/#projects#branched-llm-mvp).
   - The system is tested with **Gemini API** so far, with plans to try **GPT-4**, **Claude**, and local **LLaMA** models, and the poster is looking for testers.
- **Llama Parsers take fare share of time to parse**: Members discussed the use of LlamaIndex parsers for **.doc**, **.pdf**, and **.ppt** files, especially when text is on images.
   - A member suggested using **LlamaParse** in premium mode, while another suggested converting PPTs to PDFs for better speed or using ThreadPoolExecutor() to parse documents asynchronously.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/)** (1 messages): 

dbreunig: https://www.dbreunig.com/2025/07/31/how-kimi-rl-ed-qualitative-data-to-write-better.html
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1400619842368962560)** (2 messages): 

> `DSpill, Yaron Minsky, Quant Bros` 


- **Coining new verbs: DSpill is here!**: A member asked who would *give it a second try to **DSpill Yaron Minsky / quant bros***.
   - Another member replied *Wow new verb: to **DSpill***.
- **The quant bros get DSpilled?**: A member proposed the idea of 'DSpilling' **Yaron Minsky** and the **quant bros**.
   - This sparked the coining of a new verb, '**DSpill**,' to describe the action.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1400588919791161475)** (2 messages): 

> `Mojo installation issues, GitHub issue reporting, Detailed logs for debugging` 


- **Mojo Install Woes Prompt GitHub Issue?**: A member reported difficulties installing **Mojo** for three days and inquired about opening a **GitHub issue**.
   - Another member encouraged them to open an issue and include detailed logs to aid in troubleshooting.
- **Detailed Logs Recommended for GitHub Issue**: When submitting a **GitHub issue** for **Mojo** installation problems, including detailed logs can significantly help.
   - This provides developers with the necessary information to diagnose and resolve the installation issue more efficiently.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1400972756421443615)** (1 messages): 

> `Tail Call Elimination, Print/Log Statements, Minimal Examples` 


- **Tail Call Elimination Triggers**: A member is creating a minimal example and noticed that **tail call elimination** doesn't trigger if basic **print/log statements** are added to the functions.
   - The member is asking why that might be the case.
- **Print/Log Statements Impact Tail Call Elimination**: The discussion centers on how adding **print/log statements** can prevent **tail call elimination** in minimal examples.
   - The member seeks to understand the underlying reasons for this behavior, specifically when creating minimal examples.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1400781766913949827)** (3 messages): 

> `OpenAI Model Leak, Mixture of Experts, FP4 weights` 


- **OpenAI's Alleged Model Leak**: It is rumored that **OpenAI** has a **leaked model** with **128 experts** and **120B parameters**.
   - The model's weights are allegedly in **FP4** format, indicating a highly compressed or quantized state.
- **Deep dive into MoE**: **Mixture of Experts** models are composed of multiple sub-networks, known as *experts*, with a gating network that learns to route each input to the most relevant experts.
   - This is an area of active research as this enables scaling model size without a proportional increase in compute costs.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1400911694011699361)** (1 messages): 

> `Course Quizzes Availability, Google Forms Reopening` 


- **Quizzes with Answer Keys Now Available Online**: An archive of the **quizzes (with answer key)** is available in the *"Quizzes"* section of the course website.
   - This provides students with a valuable resource for reviewing course material and assessing their understanding.
- **Google Forms for Quizzes Will Not Be Reopened**: The course staff has announced that they will not be able to reopen the **Google Forms** used for quizzes.
   - Students who missed the opportunity to take the quizzes through **Google Forms** should utilize the available archive for review.


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1400899899402489856)** (1 messages): 

> `Qwen3-Coder, Token Speed, US Servers` 


- **Qwen3-Coder Lands on Windsurf with Lightning Speed**: **Qwen3-Coder** is now live in Windsurf, clocking in at approximately **2000 tokens/sec**.
   - The launch was announced on [X](https://x.com/windsurf/status/1951340259192742063) and [Reddit](https://www.reddit.com/r/windsurf/comments/1mf3e5s/qwen3coder_at_2000_tokenssec_is_now_live_in/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button), and is fully hosted on US servers.
- **Windsurf houses Qwen3-Coder**: A new blazing fast model named **Qwen3-Coder** is in Windsurf.
   - Running at 2000 tokens per second, discussions are being had on [Reddit](https://www.reddit.com/r/windsurf/comments/1mf3e5s/qwen3coder_at_2000_tokenssec_is_now_live_in/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) about it's implications.

