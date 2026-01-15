---
id: MjAyNi0w
title: >-
  Anthropic Labs: Cowork, Claude Code, MCP, Skills incubator led by Mike Krieger
  and Ben Mann
date: '2026-01-13T05:44:39.731046Z'
description: >-
  **Anthropic** consolidates its AI agent products under the **Cowork** brand,
  integrating prior tools like **Claude Code** and **Claude for Chrome** into a
  unified agent with sandboxed Linux VM environments using **Apple's
  virtualization** and **bubblewrap** for security. Meanwhile, **Anthropic
  Labs** reorganizes with Mike Krieger stepping down as CPO, focusing on
  productizing **Claude** with a >$1B ARR agent lab. The AI community debates
  the meaning of "vibe coding," emphasizing disciplined engineer verification
  over casual coding. **LangChain** launches **Agent Builder GA**, offering
  no-code but powerful agent orchestration features like memory, triggers, and
  human-in-the-loop approvals. Some experts advocate simplifying agent tooling
  to core filesystem and bash access for efficiency. Open-source recreations of
  Cowork-like environments using **QEMU** and sandboxing tools highlight rapid
  commoditization of AI agent tech.
companies:
  - anthropic
  - langchain
  - apple
models:
  - claude
  - claude-code
topics:
  - sandboxing
  - agent-ux
  - agent-orchestration
  - human-in-the-loop
  - memory-management
  - tooling-simplification
  - linux-virtualization
  - security
  - agent-productization
people:
  - mike_krieger
  - ben_mann
  - gergely_orosz
  - yuchen_jin
  - harrison_chase
  - jared_z
---


**Anthropic's product studio grows up.**

> AI News for 1/13/2026-1/14/2026. We checked 12 subreddits, [**544** Twitters](https://twitter.com/i/lists/1585430245762441216) and **24** Discords (**204** channels, and **2271** messages) for you. Estimated reading time saved (at 200wpm): **202 minutes**. **Our new website** is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on [@smol_ai](https://x.com/Smol_AI)!


We are using this to combine back to back announcements of [Cowork](https://claude.com/blog/cowork-research-preview), and [Anthropic Labs](https://www.anthropic.com/news/introducing-anthropic-labs).

Cowork is a bundling and productization of a ton of the prior work from [Computer Use](https://www.anthropic.com/news/3-5-models-and-computer-use) to the [bundling of Claude Code into Claude Desktop](https://blog.getbind.co/claude-code-is-now-available-on-the-claude-desktop-app/) to [Claude for Chrome](https://www.reddit.com/r/ClaudeAI/comments/1prcypb/anthropic_just_dropped_claude_for_chrome_ai_that/) - now in one cohesive brand and UI/general agent called Cowork.

In comparison, Labs is simpler: a reorg where Mike Krieger steps down as CPO of Anthropic (for ex Meta-mate [Ami Vora](https://www.linkedin.com/in/amvora/)) and now he and Ben Mann run a >$1B ARR [agent lab](https://www.latent.space/p/agent-labs) productizing Claude.

---

# AI Twitter Recap


**AI Agent Products: Claude Code/Cowork, LangSmith Agent Builder, and “agentified” dev workflows**

- **Claude Cowork + Claude Code as the new baseline for “terminal-native agents,” with sandboxing becoming table stakes**: Several tweets focus on how Anthropic’s Cowork spins up a **Linux VM via Apple’s native virtualization** and runs inside a sandbox (e.g., via **bubblewrap**) to contain unsafe commands and failure modes like runaway processes or accidental deletes ([sandbox details](https://twitter.com/dejavucoder/status/2010993418630262817)). The broader theme: agent UX is converging on *give the model a filesystem + shell + tight permissions*, then iterate fast with human review. Demand-side pain points also show up: power users want fewer permission prompts without having to opt into `--dangerously-skip-permissions` ([friction complaint](https://twitter.com/levelsio/status/2011129631001170244); joke follow-on [here](https://twitter.com/LaurencePostrv/status/2011134254051139712)).

- **“Vibe coding” backlash → clearer taxonomy of agent-assisted engineering**: A recurring debate is that “vibe coding” is being misused to describe careful, production-grade work done *with* agents. Gergely Orosz argues we should stop calling it vibe coding when engineers are validating and closing loops ([tweet](https://twitter.com/GergelyOrosz/status/2011001698370699374)). Yuchen Jin adds a sharper definition: vibe coding originally meant *not looking at code at all*; once you review anything, it’s closer to “lucid coding” ([tweet](https://twitter.com/Yuchenj_UW/status/2011137879112908870)). This matters because it reframes what’s actually changing: not “engineering is dead,” but that **engineers with taste + verification discipline get leverage**.

- **LangSmith Agent Builder GA: no-code *but not toy* (MCP, memory, triggers, inbox/HITL)**: LangChain announces Agent Builder is **generally available** ([GA announcement](https://twitter.com/LangChain/status/2011129282580660314)). Harrison Chase and teammates highlight core primitives: **memory**, **skills**, **subagents**, **MCP/tool integrations**, **triggers** for autonomous runs, and an **agent inbox** for human approvals ([walkthrough](https://twitter.com/hwchase17/status/2011126016287113681); GA recap [here](https://twitter.com/hwchase17/status/2011134704934957382)). Multiple users emphasize it’s useful even for technical users because it packages orchestration and observability cleanly ([value framing](https://twitter.com/KevinBFrank/status/2011154462128144539)). The meta-lesson: orchestration products are moving from “prompt + tools” to *operational agent stacks* (auth, triggers, audit trails, supervised actions).

- **A counter-trend: “get out of the model’s way” and simplify tooling**: Jared Z argues that adding tools/guardrails can degrade performance because you’re forcing extra branching decisions; he cites Vercel simplifying a text-to-SQL agent down to filesystem + bash access ([thread](https://twitter.com/imjaredz/status/2011218314035642464)). This aligns with the growing consensus that *bash + filesystem is the universal tool call*, and that modern models can shoulder complexity that used to require DAGs.

- **Open recreations and “Cowork clones” show fast commoditization**: A developer built a cross-platform Cowork-like VM using **QEMU + bubblewrap + seccomp**, controlled via a `vmctl` utility and websocket ([tweet](https://twitter.com/SIGKITTEN/status/2011077925085347909)). MiniMax also claims someone rebuilt Cowork with Anthropic-compatible APIs and open-sourced it ([tweet](https://twitter.com/MiniMax_AI/status/2011270108166107311)). The signal: **agent shells are becoming replicable infra patterns**, not proprietary moats.

---

**Long-context + memory: from RAG chunking wars to Recursive Language Models and RL memory**

- **Filesystem agents vs vector search: hybridization is the real outcome**: LlamaIndex benchmarks “fs-explorer” style agents against classic RAG. Their summary: filesystem exploration can be **more accurate** (full-file context) but **slower**, and vector search wins at scale (1k+ docs) ([LlamaIndex post](https://twitter.com/llama_index/status/2011121143927972076); Jerry Liu’s synthesis [here](https://twitter.com/jerryjliu0/status/2011130432205832664)). Weaviate reiterates the central trade-off in chunking: **retrieval precision vs contextual richness**, and there’s no universal chunk size ([tweet](https://twitter.com/weaviate_io/status/2011088315663978739)).

- **MemRL: treat memory retrieval as RL (utility-aware), not similarity search**: DAIR AI highlights **MemRL**, which keeps the base LLM frozen and learns **Q-values over episodic memories** (Intent–Experience–Utility), with a two-phase retrieval: semantic filter then utility ranking ([summary](https://twitter.com/dair_ai/status/2011086096986443905)). If the claims hold, it’s a compelling pattern for production agents: avoid finetuning/catastrophic forgetting, but still **improve from experience** via a learned memory policy.

- **Recursive Language Models (RLMs): symbolic access to the prompt, not “subagents as tool calls”**: Omar Khattab/lateinteraction’s posts argue most “sub-agent” implementations miss the core idea: you can’t materialize millions of subcalls as tool calls, and you need **pointer-like / symbolic access to the prompt** to recurse through it programmatically ([critique](https://twitter.com/lateinteraction/status/2011250721681773013)). The TuringPost’s recap frames RLMs as an inference-time architecture that offloads context into a Python REPL variable so models can manipulate it via code, scaling beyond **10M tokens** without retraining ([summary](https://twitter.com/TheTuringPost/status/2011272650132504889)). Key takeaway for engineers: *“long context” may increasingly mean “code-mediated context access,” not just bigger windows.*

- **Context rot mitigation via prompt compression**: DSPy is used as an example workflow for reducing prompt length without losing performance, explicitly framed as a way to fight context degradation ([tweet](https://twitter.com/hammer_mt/status/2011022198023082263)).

---

**Video generation and controllable world models: Kling Motion Control, Veo 3.1 upgrades, and new “world model” claims**

- **Kling 2.6 Motion Control emerges as a best-in-class performance/motion transfer tool (but identity drift remains)**: Multiple creators report that Kling’s Motion Control can replace/drive characters in scenes with unusually high precision ([viral claim](https://twitter.com/AngryTomtweets/status/2010975679488409890)). A detailed Japanese demo shows instrument performance transfer with high-fidelity finger motion and rhythm, suggesting near-term realism for single-subject shots ([demo thread](https://twitter.com/akiyoshisan/status/2010983687727587587)). Curious Refuge tests it for live-action narrative: parallax looks strong, but face consistency drifts; best results occur when the reference image is close to the initial frame ([tests](https://twitter.com/CuriousRefuge/status/2011207976095531524)).

- **Google’s Veo 3.1: “Ingredients to Video” gets portrait mode + higher resolution + consistency improvements + SynthID**: DeepMind/Google roll out Veo 3.1 updates emphasizing (1) **native vertical 9:16**, (2) improved character/background consistency, (3) **1080p + 4K** options, and (4) **SynthID watermarking** for verification ([DeepMind thread](https://twitter.com/GoogleDeepMind/status/2011121716336984151); API summary [here](https://twitter.com/_philschmid/status/2011122136619110762); Gemini app rollout [here](https://twitter.com/GeminiApp/status/2011122407013306875); Sundar Pichai post [here](https://twitter.com/sundarpichai/status/2011143120516469199); Demis Hassabis [here](https://twitter.com/demishassabis/status/2011236200397639900)). Engineers should note the product direction: **mobile-first formats + provenance + production-ready resolutions** are being prioritized over raw novelty.

- **“World model” branding accelerates; research benchmarks try to catch up**: PixVerse markets “R1” as a “real-time world model” (very marketing-heavy) ([tweet](https://twitter.com/PixVerse_/status/2011100288690897317)). More technical: TencentARC’s **VerseCrafter** claims 4D geometric control over camera and multi-object motion ([announcement](https://twitter.com/wbhu_cuhk/status/2011109476510941222)). A separate “Video Deep Research Benchmark on Open Web for Agentic Video Reasoning” also appears ([tweet](https://twitter.com/_akhaliq/status/2011105482111651992)), reinforcing that evaluation for video agents is still immature.

---

**Open models, on-device ML, and multimodal medical AI: MedGemma 1.5, GLM-Image, MLX throughput**

- **MedGemma 1.5 + MedASR: open medical multimodal stack focused on *offline* and 3D imaging**: Google announces **MedGemma 1.5** as small enough to run offline and improved for multimodal medical tasks ([Google AI Devs](https://twitter.com/googleaidevs/status/2011181120793297361)). Phil Schmid’s technical bullet list highlights a **4B** model with support for **3D volumes (CT/MRI)**, longitudinal comparison, and anatomical localization; he cites **89.6% EHR understanding accuracy** (+22%) and **38% IoU** for X-ray localization ([tweet](https://twitter.com/_philschmid/status/2011183904204390654)). Sundar Pichai positions it as a “major upgrade” and pairs it with **MedASR** for medical dictation ([tweet](https://twitter.com/sundarpichai/status/2011184917670216196)). Google Research announces both on Hugging Face + Vertex AI ([tweet](https://twitter.com/GoogleResearch/status/2011185403856883907)). Net: **open, efficient, clinically-oriented multimodal models** are becoming a first-class release category.

- **GLM-Image: hybrid autoregressive + diffusion for “poster/PPT/text rendering” and knowledge-heavy generation**: Zhipu AI releases GLM-Image and claims strong text rendering and infographic/poster generation via a hybrid architecture ([release](https://twitter.com/Zai_org/status/2011247591825068314)). Third parties amplify the architecture details (e.g., “9B AR + 7B diffusion”) and “cognitive generation” framing ([fal launch](https://twitter.com/fal/status/2011271561429311512); ModelScope recap [here](https://twitter.com/ModelScope2022/status/2011262011997651194)). For engineers, the key is the *design goal*: better **layout + multi-line text** reliability, a common diffusion weakness.

- **On-device and local inference continues to climb**: A LocallyAI update notes **LiquidAI LFM 2.5** models available on iOS via MLX ([tweet](https://twitter.com/LocallyAIApp/status/2011136235973329301)). MLX performance benchmarks show MiniMax M2.1 running locally on M3 Ultra with continuous batching: **4-bit 220 tok/s at 32 requests** (reported) ([tweet](https://twitter.com/ivanfioravanti/status/2011115626690179290)). Awni Hannun highlights MLX adding quantization support (nvfp4/mxfp8) across Metal and CUDA ([tweet](https://twitter.com/awnihannun/status/2011267993091875282)). There’s also a “local model” privacy jab in the Claude/Cowork discourse ([tweet](https://twitter.com/victormustar/status/2011078287762825474)).

---

**Benchmarks, evals, and agent reliability: instruction-following, visual reasoning limits, and “boring agents”**

- **OctoCodingBench: aligned coding agents ≠ passing unit tests**: MiniMax releases **OctoCodingBench** to measure whether coding agents comply with system prompts, repo conventions, and tool policies—explicitly addressing “paperclip-maxing” behavior in repos ([tweet](https://twitter.com/MiniMax_AI/status/2011266592303432058); dataset mention [here](https://twitter.com/HuggingPapers/status/2011074090686136349)). This is an important shift: moving evals from pure functional correctness to **process constraints** and organizational norms.

- **BabyVision: MLLMs still weak at “pure visual reasoning”**: HuggingPapers cites BabyVision results: SOTA MLLMs at **49.7%** vs adult humans **94.1%** on 388 tasks, arguing these require non-linguistic visual understanding ([tweet](https://twitter.com/HuggingPapers/status/2011048605113581762)). If you build multimodal agents, the implication is that “looks solved” demos can mask brittle visual reasoning.

- **Enterprise “boring agents” as a product stance**: AI21 explicitly markets “boring agents” optimized for **auditable, repeatable** outputs over chat charm ([tweet](https://twitter.com/AI21Labs/status/2011041313039204838)). This aligns with the eval trend: less about vibe, more about governance.

- **METR: expanding beyond “capabilities” into loss-of-control framing**: Ajeya Cotra joins METR to expand LOC risk assessment across “means, motive, opportunity,” noting motive/opportunity measurement is underdeveloped and likely to become load-bearing ([tweet](https://twitter.com/ajeya_cotra/status/2011146702175289563); definitions [here](https://twitter.com/ajeya_cotra/status/2011146714183581886)).

---

**Infra + training systems: schedulers, attention backends, quantization pitfalls, and FP8/low-bit training**

- **HPC schedulers vs cloud-native orchestration (post-Slurm acquisition discourse)**: dstack frames Nvidia’s Slurm acquisition as evidence workloads are moving toward cloud-native schedulers and provides a Slurm→dstack migration guide ([tweet](https://twitter.com/dstackai/status/2011091749901422904)). SkyPilot promotes “Pools” as a unified batch queue across K8s + clouds ([tweet](https://twitter.com/skypilot_org/status/2011128941705339270)). The pattern: infra teams are standardizing around **multi-cluster GPU pooling** and vendor-agnostic schedulers.

- **Diffusers adds “Unified Attention” backend**: Hugging Face Diffusers ships a new attention backend combining properties of Ring and Ulysses ([tweet](https://twitter.com/RisingSayak/status/2011092823828021730)). This is part of the continuing push to make attention kernels/backends swappable and performance-portable.

- **Quantization/training nuance continues to bite**: TensorPro reports MXFP4-quantized attention can break causal modeling, with a post on diagnosing/fixing “leaky quantization” behavior ([tweet](https://twitter.com/tensorpro/status/2011198742406578252)). Separately, a Google Cloud post is shared about **stochastic rounding** mitigating vanishing gradients in low-precision training (FP8/4-bit) ([tweet](https://twitter.com/dl_weekly/status/2011060892897558717)). For practitioners: “train in FP8/low-bit” is increasingly viable, but **numerical edge cases** are still active research/ops problems.

---

**Top tweets (by engagement)**

- **McDonald’s Japan** “Black Pepper!! PV” post (massive viral engagement) ([tweet](https://twitter.com/McDonaldsJapan/status/2010985164692668892))  
- **Joe Rogan clip on “show your papers” / militarized enforcement** ([tweet](https://twitter.com/OfTheBraveUSA/status/2011153857976668290))  
- **Anthropic donation to Python Software Foundation ($1.5M)** ([Alex Albert](https://twitter.com/alexalbert__/status/2011143093266104800); PSF thanks [here](https://twitter.com/ThePSF/status/2011060802321584414))  
- **Claude Code / agent productivity meme posts** capturing the cultural moment ([nearcyan “age of the engineer”](https://twitter.com/nearcyan/status/2011129737578500526); giffmana Cowork parody [here](https://twitter.com/giffmana/status/2011165027374334221))


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Pocket TTS and Local AI Tools

  - **[kyutai just introduced Pocket TTS: a 100M-parameter text-to-speech model with high-quality voice cloning that runs on your laptop—no GPU required](https://www.reddit.com/r/LocalLLaMA/comments/1qbpz5l/kyutai_just_introduced_pocket_tts_a_100mparameter/)** (Activity: 494): ****Kyutai Labs** has released **Pocket TTS**, a `100M-parameter` text-to-speech model designed for high-quality voice cloning that operates efficiently on a CPU without requiring a GPU. The model is accessible via [GitHub](https://github.com/kyutai-labs/pocket-tts) and [Hugging Face](https://huggingface.co/kyutai/pocket-tts), and is detailed in a [blog post](https://kyutai.org/blog/2026-01-13-pocket-tts). The model's architecture is inspired by recent advancements in continuous audio language models, as discussed in the related [arXiv paper](https://arxiv.org/abs/2509.06926).** Some users question the model's performance, suggesting that models of this size may not provide sufficient quality compared to larger models or 'hardcoded' solutions used in applications like Twitch. There is also interest in the model's language capabilities and potential for fine-tuning across different languages.

    - A user noted a significant memory management issue with the Pocket TTS model, where the localhost test server setup does not clear memory between generations, causing memory usage to increase significantly. They reported memory usage reaching **32 GB** on their system, suggesting that the model should clear memory when starting a new generation to prevent such ballooning.
    - Another user provided a detailed performance analysis of the Pocket TTS model on a Ryzen **5950X** CPU. They observed that the model uses about **1.1 GB of RAM** initially and is capable of generating audio quickly, with a time to first audio of around **200 ms**. However, as the context fills up, RAM usage can grow significantly, reaching **8.5 GB** for a single article. They also commented on the model's intonation being good for its size, though the overall voice quality was described as mediocre.
    - A user expressed skepticism about the utility of small models like Pocket TTS, suggesting that they may not be worth the effort compared to more established solutions, especially if high-quality output is required. They mentioned trying the demo and finding the results unsatisfactory, implying that larger models or hardcoded solutions might be more effective for certain applications.

  - **[A Windows tool I made to simplify running local AI models](https://www.reddit.com/r/LocalLLM/comments/1qbzd2w/a_windows_tool_i_made_to_simplify_running_local/)** (Activity: 28): ****V6rge** is a Windows-based tool designed to simplify running local AI models by bundling and isolating its own runtime, thus avoiding system Python conflicts. It supports running local LLMs like Qwen, DeepSeek, and Llama via GGUF, as well as image generation with Stable Diffusion and Flux variants, and basic voice and music generation. The tool aims to reduce setup friction and is available for download on [GitHub](https://github.com/Dedsec-b/v6rge-releases-/releases/tag/v0.1.4).** Concerns were raised about the tool not being open source, which makes users hesitant to run executables. Additionally, users reported issues such as a 'Failed to Save Settings: API error 404' when changing settings, indicating potential stability problems. Suggestions for improvements include adding a gallery feature for generated images.

    - A user reported a critical issue where changing the Model Folder in the settings results in an 'API error 404'. This suggests a potential bug in the application's settings management, possibly due to incorrect API endpoint handling or missing backend support for this feature.
    - Another user encountered an 'Error: undefined' when attempting to download specific models like Qwen-Image or FLUX.1-dev. This indicates a possible issue with the model download functionality, which could be related to incorrect URL handling or server-side problems.
    - A request was made for a Linux Docker version of the tool, highlighting the demand for cross-platform compatibility. The user suggested features like Docker Compose, a maintained Docker Hub image, and Portainer support, which would facilitate easier deployment and management in containerized environments.


### 2. GLM-Image and NER Model Releases

  - **[GLM-Image is released!](https://www.reddit.com/r/LocalLLaMA/comments/1qc9m6x/glmimage_is_released/)** (Activity: 393): ****GLM-Image** is a newly released image generation model featuring a hybrid autoregressive and diffusion decoder architecture. It competes with mainstream latent diffusion models in general image quality but excels in text-rendering and knowledge-intensive scenarios, demonstrating superior semantic understanding and complex information expression. The model supports text-to-image generation and various image-to-image tasks, such as image editing, style transfer, and identity-preserving generation, while maintaining high fidelity and fine-grained detail.** The release under the **MIT license** is noted for its openness compared to more restrictive licenses from Western labs. The model's performance is compared to **nano banana 2**, suggesting it is a significant advancement, especially with its combined editing and generation capabilities.

    - The release of GLM-Image under the MIT license is highlighted as a significant advantage, especially when compared to Western labs that often release models under more restrictive licenses. This open licensing could facilitate broader adoption and innovation in the community.
    - GLM-Image reportedly performs comparably to the 'nano banana 2' on benchmarks, which is notable given its dual capability in both editing and generation. This dual functionality could make it a versatile tool in various applications, enhancing its appeal to developers and researchers.
    - The model consists of a 13GB diffusion model and a 20GB text encoder, indicating substantial resource requirements. There is anticipation for the model to be quantized to fp8 and for the development of efficient training methods like LoRA to make it more accessible for experimentation.

  - **[500Mb Named Entity Recognition (NER) model to identify and classify entities in any text locally. Easily fine-tune on any language locally (see example for Spanish).](https://www.reddit.com/r/LocalLLM/comments/1qbnezw/500mb_named_entity_recognition_ner_model_to/)** (Activity: 13): **A new `500Mb` Named Entity Recognition (NER) model has been released, capable of identifying and classifying entities in text locally. This model is designed for easy fine-tuning across different languages, with a specific example provided for Spanish. The model's compact size allows for efficient local deployment without the need for cloud resources, making it suitable for privacy-sensitive applications. The model's architecture and training details, however, are not specified in the post.** The post lacks detailed technical discussion or debate, as the top comment is non-technical and simply expresses approval.



### 3. AI Hardware Innovations

  - **[AI TOP 100 M.2 SSD](https://www.reddit.com/r/LocalLLM/comments/1qbvycy/ai_top_100_m2_ssd/)** (Activity: 26): **The image showcases a GIGABYTE AI TOP 100E M.2 SSD, which is marketed as enhancing AI performance by providing high bandwidth, potentially reducing the load on RAM/VRAM. However, commenters suggest that this is largely a marketing gimmick, as the bandwidth of even the fastest PCIe 5 SSDs (around `10GB/s`) is significantly lower than that of DDR5 RAM (`80GB/s`). This makes the SSD less effective for offloading large AI models, as the speed would be a bottleneck, especially for dense models. Sparse models might benefit slightly, but performance gains would still be limited to low single-digit tokens per second.** Commenters are skeptical about the product's claims, suggesting it is more of a marketing strategy than a practical solution for AI workloads. They recommend using the best available NVMe PCIe 5 SSDs instead, as the performance gains are minimal.

    - Themash360 highlights the limitations of using NVMe SSDs for AI workloads, noting that even with PCIe 5's optimistic 10GB/s bandwidth, it's significantly slower compared to DDR5 RAM's 80GB/s. They illustrate this with a scenario where offloading 100GB of a 240GB dense model to NVMe results in a token generation speed of 0.1 tokens per second, emphasizing the inefficiency for dense models.
    - Themash360 also mentions that while using Mixture of Experts (MoE) models can mitigate some of the performance penalties by offloading sparse areas, the improvement is limited, resulting in only low single-digit tokens per second. This highlights the challenges in achieving high performance with current storage technologies when dealing with large AI models.
    - desexmachina points out that faster SSDs can lead to higher processor saturation, implying that while storage speed is a factor, the overall system performance is also dependent on the CPU's ability to handle increased data throughput. This suggests a need for balanced system architecture to optimize AI workloads.

  - **[My wishes for 2026](https://www.reddit.com/r/LocalLLaMA/comments/1qbw325/my_wishes_for_2026/)** (Activity: 767): **The image is a speculative wishlist for technological advancements by 2026, featuring potential developments in AI models and hardware. It includes the release of new versions of AI models like GPT-OSS, Gemma 4, Qwen 4, and GLM Air, as well as Llama 5, which is anticipated to outperform Mistral's 123B model. Additionally, there is a wish for a DeepSeek model under 200B parameters and an affordable GPU with more than 32GB of memory. The image reflects aspirations for significant progress in AI capabilities and hardware accessibility.** Commenters express skepticism about the feasibility of an affordable GPU with more than 32GB, highlighting the ongoing challenge of high GPU prices.

    - SlowFail2433 highlights the underappreciated performance of GPT OSS 120B, noting its impressive benchmark score to parameter count ratio and its effective FP4 quantization. This model is contrasted with the Qwen 4 series, which is frequently cited in Arxiv papers, particularly for agentic RL applications. The discussion emphasizes the advantages of small, dense models in avoiding the complexities of MoE gates during training, which can complicate credit assignment in RL scenarios.

  - **[I'm building a real-life BMO with a Raspberry Pi 5 (Mistral/OpenAI + YOLO11n)](https://www.reddit.com/r/LocalLLM/comments/1qbwc35/im_building_a_reallife_bmo_with_a_raspberry_pi_5/)** (Activity: 8): **The project involves building a real-life BMO using a **Raspberry Pi 5** integrated with **Mistral/OpenAI** for AI capabilities and **YOLO11n** for object recognition. The developer is enhancing the AI companion with face and voice recognition features, aiming to enable interactive gaming experiences. Future plans include adding robotic arms. The project is open-source, with the code available on [GitHub](https://github.com/ivegotanheadache/BMO).** A commenter is also developing a similar project using a large language model (LLM) assistant with voice recognition and text-to-speech, considering adding gaming capabilities like chess and emulation through RetroArch and Pico 8. They are contemplating whether to integrate a dedicated monitor or use an external display.



## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. xAI's Grok Deployment and Controversies

  - **[Official: Pentagon confirms deployment of xAI’s Grok across defense operations](https://www.reddit.com/r/singularity/comments/1qbo516/official_pentagon_confirms_deployment_of_xais/)** (Activity: 1443): **The **US Department of Defense** will integrate **xAI's Grok AI** into Pentagon systems, allowing military and civilian personnel to handle Controlled Unclassified Information at Impact Level 5. Grok will be embedded in operational and planning systems to enhance intelligence analysis, decision-making, and military planning, utilizing real-time global signals from open-source and social data. The deployment aims to scale to approximately `3 million users`, with the initial phase starting this month. [Source](https://www.washingtonpost.com/business/2026/01/12/artificial-intelligence-pentagon-hegseth-musk/ec8b407a-f026-11f0-a4dc-effc74cb25af_story.html).** Comments reflect skepticism and concern about the integration of AI in military operations, with some users humorously suggesting potential security risks and others expressing distrust in the current administration's use of such technology.


  - **[The Guardian: How Elon Musk’s Grok generated 6,000 non-consensual nude images per hour.](https://www.reddit.com/r/OpenAI/comments/1qbkpw9/the_guardian_how_elon_musks_grok_generated_6000/)** (Activity: 392): **The *Guardian* investigation highlights a significant misuse of **Elon Musk's AI tool, Grok**, which in early 2026 was reportedly used to generate `6,000` non-consensual nude images per hour. This misuse was part of a broader trend where users exploited the AI to create sexualized and violent images, particularly targeting women and minors. The report underscores the ethical and regulatory challenges posed by AI technologies in content moderation and user safety.** Commenters express disillusionment with the misuse of Grok, noting a trend of users focusing on generating explicit content. There is also criticism of **US Big Tech** and the community's response to the AI's capabilities, with some users unsubscribing from related forums due to the prevalence of pornographic content discussions.

    - Fearless_Weather_206 raises a critical point about the potential legislative implications of Grok's capabilities. The concern is that incidents like these could be used as a pretext to regulate or restrict open-source LLM models, which are often less censored than their commercial counterparts. This could lead to broader debates about the balance between innovation and regulation in AI development.
    - boredatwork8866 highlights a community trend where users are primarily focused on exploiting Grok for generating adult content. This indicates a significant user interest in the model's ability to create explicit material, which has led to dissatisfaction when such capabilities are restricted, suggesting a tension between user expectations and ethical guidelines imposed by developers.
    - Joddie_ATV expresses concern over the ethical and societal implications of Grok's ability to generate non-consensual images. This raises questions about the responsibility of AI developers in preventing misuse of their technologies and the effectiveness of current safeguards in place to protect against such abuses.

  - **[Nothing could go wrong](https://www.reddit.com/r/OpenAI/comments/1qc2b0f/nothing_could_go_wrong/)** (Activity: 365): **The image is a meme that humorously comments on the announcement of the US Secretary of Defense integrating **Elon Musk's xAI platform, Grok**, into military networks. This integration is part of an AI acceleration strategy, as reported by Reuters. The tweet by Jarvis sarcastically suggests that this integration is low risk, implying potential concerns about the implications of such a move. The comments reflect skepticism and humor about the potential consequences of integrating AI into military operations, with references to dystopian scenarios and political commentary.** The comments express skepticism and humor, with references to dystopian scenarios like those in the Terminator series, and political commentary on the influence of private companies in government contracts.



### 2. DeepSeek's Engram Module and Innovations

  - **[[R] (DeepSeek) Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](https://www.reddit.com/r/MachineLearning/comments/1qbnkrn/r_deepseek_conditional_memory_via_scalable_lookup/)** (Activity: 55): ****DeepSeek** introduces a novel module called **Engram** that enhances large language models by implementing a new axis of sparsity through conditional memory, allowing for efficient O(1) lookup. This approach contrasts with traditional Mixture-of-Experts (MoE) models by optimizing the balance between neural computation and static memory, as demonstrated by a U-shaped scaling law. Engram scales to `27B parameters`, outperforming iso-parameter and iso-FLOPs MoE baselines in various benchmarks, including `MMLU`, `CMMLU`, `BBH`, `ARC-Challenge`, `HumanEval`, and `MATH`. The module improves reasoning and retrieval by offloading static reconstruction from early layers and enhancing attention capacity for global context, achieving significant gains in long-context retrieval tasks. Engram's deterministic addressing also supports runtime prefetching, minimizing overhead and enhancing infrastructure efficiency.** A comment highlights the practical efficiency of Engram, noting its ability to avoid unnecessary forward passes for recomputing common facts, thus improving throughput and efficiency without initially appearing innovative.


  - **[DeepSeek V4 Could Blow Claude and GPT Away for Coding](https://www.reddit.com/r/DeepSeek/comments/1qblbjf/deepseek_v4_could_blow_claude_and_gpt_away_for/)** (Activity: 226): ****DeepSeek V4** is set to launch with claims of outperforming **Claude** and **GPT** in coding tasks. The model introduces the **Engram module**, which uses a memory lookup system to manage super-long prompts by decoupling memory from computation, potentially enhancing performance by allowing attention and MLP layers to focus on complex tasks. This architecture might also reduce VRAM requirements by `30%`, offloading them to RAM. However, some skepticism exists regarding leaks about DeepSeek's capabilities.** One user shared their experience using DeepSeek for developing a complex encryptor, noting its efficiency in coding with fewer lines compared to Meta and ChatGPT. However, they found Claude superior in handling specific functionalities and providing a more critical code review, despite DeepSeek's higher encouragement score.

    - Engram, a feature in DeepSeek, is designed to optimize performance by offloading simpler tasks to free up the Attention and MLP layers for more complex processing. This approach allows the model to behave as if it were deeper, potentially enhancing its efficiency. Additionally, Engram can reduce VRAM requirements by approximately 30% by utilizing RAM instead.
    - A user shared their experience using DeepSeek for building a complex encryptor, noting that it was more efficient than Meta and ChatGPT, requiring fewer lines of code. However, for a specific functionality, only Claude succeeded where others failed. In a review of the completed code, DeepSeek rated it 8.3/10, ChatGPT 6.8/10, and Claude initially 5.5/10, later adjusted to 6.5 after a debate.
    - There is skepticism about DeepSeek V4's potential to surpass Claude and GPT in coding tasks due to the smaller size of the DeepSeek team compared to Anthropic, OpenAI, or Google. While DeepSeek has pioneered reasoning concepts, the real test will be its performance in agentic and coding tasks, which are relatively new areas for the team. The previous version, 3.2, was noted for its strong performance in domain-specific automation tasks.

  - **[DeepSeek Unveils Engram, a Memory Lookup Module Powering Next-Generation LLMs](https://www.reddit.com/r/DeepSeek/comments/1qbozaf/deepseek_unveils_engram_a_memory_lookup_module/)** (Activity: 80): ****DeepSeek** has introduced a new module called **Engram**, designed to enhance large language models (LLMs) by integrating a memory lookup system. This system utilizes `N-gram embeddings` in conjunction with a neural backbone to reduce computational load for 'static knowledge'. The challenge lies in implementing effective context-aware gating to optimize this integration, which could significantly improve reasoning capabilities in LLMs.** Commenters are intrigued by the potential of Engram to influence memory management in LLMs, with some speculating on its adoption by major players like **OpenAI** or **Google**. There is a technical debate on the effectiveness of context-aware gating in enhancing reasoning within these models.

    - The use of lookups for N-gram embeddings alongside a neural backbone can significantly reduce computational demands for 'static knowledge'. The challenge lies in implementing effective context-aware gating, which is crucial for reasoning tasks. The approach taken by DeepSeek in solving this gating issue could be pivotal in enhancing the utility of such systems.


### 3. Claude Code and Ralph Wiggum Techniques

  - **[TRUST ME BRO: Most people are running Ralph Wiggum wrong](https://www.reddit.com/r/ClaudeCode/comments/1qc4vg0/trust_me_bro_most_people_are_running_ralph_wiggum/)** (Activity: 225): **The post discusses the use of 'Ralph Wiggum' as a method to run AI coding tools like Claude Code in a continuous loop, addressing limitations such as premature stopping. The author critiques the official Claude Code Ralph plugin for its inefficiency in handling context windows, leading to bloated contexts and hallucinations. Instead, they advocate for using a bash loop, originally by Geoffrey Huntley, which starts a fresh context each iteration, making it more suitable for long-running tasks. Key setup recommendations include using a sandbox for safety, structured task lists for efficiency, setting iteration limits for cost control, and implementing a feedback loop with tools like Playwright or Claude for Chrome. The author provides a [YouTube walkthrough](https://youtu.be/eAtvoGlpeRU) and a [GitHub guide](https://github.com/JeredBlu/guides/blob/main/Ralph_Wiggum_Guide.md) for further details.** Commenters highlight the importance of Geoffrey Huntley's original work and note that he initially received free tokens, which may not be the case for all users. Concerns are raised about the practicality of using Ralph Wiggum for complex tasks or in team settings, as errors can compound and lead to unmanageable pull requests.

    - Geoffrey Huntley, the creator of Ralph, initially received all his tokens for free, which may have influenced the development and deployment strategies of Ralph. This could imply that the cost considerations for users might differ significantly from Huntley's original use case, potentially affecting how Ralph is utilized in practice.
    - A key concern raised is the potential for compounded errors when using automated tools like Ralph, especially in complex projects. If a mistake occurs early in the process, it could propagate through subsequent stages, leading to significant issues. This highlights the importance of careful oversight and iterative feedback, particularly in team environments where large pull requests might be problematic.
    - There is a debate about the effectiveness of Ralph compared to using Claude for project planning and execution. Some users find that Claude can handle end-to-end project phases effectively with proper instructions, questioning what additional benefits Ralph provides. This suggests a need for clearer differentiation or demonstration of Ralph's unique capabilities in automating complex workflows.

  - **[Smart Ralph: A Claude Code plugin for spec-driven development with Ralph-style loops](https://www.reddit.com/r/ClaudeCode/comments/1qbvudj/smart_ralph_a_claude_code_plugin_for_specdriven/)** (Activity: 84): ****Smart Ralph** is a new open-source plugin for **Claude Code** that implements a spec-driven development workflow using the **Ralph agentic loop pattern**. This approach addresses the common issue in AI-in-IDE flows where AI starts coding immediately, often resulting in incomplete or mismatched implementations. Smart Ralph requires Claude to first conduct research, gather requirements, design architecture, and break down tasks before writing any code. It uses specialized sub-agents for each phase, ensuring a structured and context-aware development process. The plugin is available on [GitHub](https://github.com/tzachbon/smart-ralph) and can be installed via the plugin marketplace.** Commenters are interested in the token cost compared to the normal Ralph cycle, and one user noted that Smart Ralph seems less token-intensive than their own similar plugin, which also doesn't require openspec. Another user expressed relief at not having to maintain a similar project they were working on.

    - azr2001 inquires about the token cost of the Smart Ralph plugin compared to the traditional Ralph cycle, suggesting a focus on efficiency and resource management in AI-driven development workflows.
    - LittleJuggernaut7365 notes that the Smart Ralph plugin appears to be less token-intensive than their own similar plugin, which also required 'openspec'. This highlights the Smart Ralph's potential for more efficient resource usage and broader compatibility without additional dependencies.
    - Longjumping_Guess360 suggests an enhancement for future development: enabling a swarm of AIs to compete in problem-solving, where consensus among multiple AIs could indicate the best solution. This points to a potential direction for improving AI decision-making processes through collaborative validation.

  - **[[D] Is anyone actually paying for GPU Cluster TCO Consulting? (Because most companies are overpaying by 20%+)](https://www.reddit.com/r/MachineLearning/comments/1qbljgq/d_is_anyone_actually_paying_for_gpu_cluster_tco/)** (Activity: 24): **The post discusses the inefficiencies in AI infrastructure procurement, highlighting that companies often overpay by focusing solely on **$/GPU/hour** without considering the **Total Cost of Ownership (TCO)**. The author suggests that factors like **Model FLOPs Utilization (MFU)**, hidden costs in data egress and storage, and network inefficiencies can lead to significant overspending. They propose a consulting service to help companies evaluate these factors, potentially saving 20-30% on compute costs. The post emphasizes that a "true" AI cloud can significantly improve MFU, thus reducing costs and time for large-scale model training.** Commenters argue that the issues are too complex for a simple report and that many teams are already aware of these factors. They suggest that the real challenge is not ignorance but the difficulty in accurately predicting workload needs and adapting infrastructure accordingly. Some express skepticism about the value of third-party reports, noting that organizational issues often lead to overpayment rather than a lack of knowledge about MFU.

    - whyVelociraptor argues that the issues identified in the post are too broad for a simple report to address effectively. They suggest that serious teams are already aware of these issues or can figure them out when necessary. The comment also expresses skepticism about the value of such consulting, implying that it might just replicate what a large language model (LLM) like ChatGPT could generate, which users could do themselves for free.
    - patternpeeker highlights that overpayment is often due to organizational issues rather than ignorance of Model FLOPs Utilization (MFU). They note that companies struggle to estimate workload mix and utilization accurately, leading to procurement decisions based on defensible hourly rates rather than optimal ones. The comment emphasizes that the real challenge is making infrastructure decisions that remain valid as workloads evolve, rather than just understanding MFU.
    - audiencevote points out that many assume all H100 GPUs are the same across providers, but this is not the case. They mention that the industry average for Model FLOPs Utilization (MFU) is around 35-45%, and a 'true' AI cloud can achieve significantly higher utilization. This raises questions about what differentiates a 'true' AI cloud from other offerings, suggesting that there are specific optimizations or configurations that can enhance performance.


---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5.1


**1. Next-Gen Open & Domain-Specific Multimodal Models**

- **Zai’s GLM-Image Mixes Diffusion and Autoregression**: **Zai** launched **GLM-Image**, an open-source image model with a hybrid **autoregressive + diffusion** architecture aimed at **high‑fidelity detail** and **sharp text rendering**, announced via their [GLM-Image blog post](https://z.ai/blog/glm-image) and backed by code on [GitHub: GLM-Image](https://github.com/zai-org/GLM-Image). The model targets strong performance on **knowledge‑intensive generation** and supports a rich set of **image-to-image tasks** like editing, style transfer, identity‑preserving generation, and multi‑subject consistency, with deployment artifacts also shared on **Hugging Face** as referenced in the [Latent Space GLM-Image discussion](https://xcancel.com/zai_org/status/2011247591825068314).
  - Community discussion in **Latent Space** and **Nous Research** emphasized GLM-Image’s superiority in **text rendering** versus “mainstream latent diffusion baselines,” while roughly matching them on general image quality according to the [z.ai GLM-Image blog](https://z.ai/blog/glm-image). Users see it as a serious building block for **open multimodal stacks**, pairing it with tools like **Qwen3-VL** and integrating into creative pipelines that already use open‑source backends.

- **LTX-2 Goes 4K With Local, Open-Source Video Gen**: **Venture Twins** announced **LTX-2**, an **open-source video generation model** capable of producing **4K clips up to 20 seconds** with **audio**, showcased in a tweet by Justine Moore linking to [LTX-2 open-source video model](https://xcancel.com/venturetwins/status/2010878914273697956). The model is designed for **local execution**, enabling engineers to run high‑resolution, audio‑enabled video synthesis on their own hardware instead of gated cloud APIs.
  - In **Latent Space’s genmedia channel**, members called out LTX-2 as a breakthrough for **DIY video tooling**, noting that creator *yanokusnir* demonstrated **end‑to‑end 4K clips** directly from the open weights in the [LTX-2 announcement thread](https://xcancel.com/venturetwins/status/2010878914273697956). Engineers are already discussing pairing LTX-2 with **RAG story pipelines** and using it as a transparent alternative to closed models like Veo, which Perplexity reportedly surfaces via **Veo3.1‑powered video generation**.

- **Qwen Image Edit Turns Pictures Into Gaussian Splats**: Builders in **Latent Space** highlighted **Qwen Image Edit’s** ability to convert images into **Gaussian Splats** and re‑render them from novel viewpoints using the [Qwen-Image-Edit-2511-Gaussian-Splash model](https://huggingface.co/dx8152/Qwen-Image-Edit-2511-Gaussian-Splash). This workflow effectively builds a 3D representation from a single frame, enabling **start‑frame → end‑frame** video renderings while keeping surrounding geometry consistent.
  - Users see this Gaussian‑splat pipeline as a pragmatic bridge between **2D LLM‑conditioned editing** and full **3D scene reconstruction**, slotting neatly into asset pipelines for games and VFX. The conversation positions Qwen Image Edit as a complement to models like **GLM-Image** and **LTX-2**, with Qwen handling **view‑consistent scenes** and the others handling **high‑fidelity frames** and **temporal video**.

- **MedGemma 1.5 Pushes Medical Vision and Speech**: Google Research announced **MedGemma 1.5** as a next‑generation model for **medical image interpretation** and **medical speech‑to‑text**, detailed in their post [“Next-generation medical image interpretation with MedGemma 1.5 and medical speech-to-text with MedASR”](https://research.google/blog/next-generation-medical-image-interpretation-with-medgemma-15-and-medical-speech-to-text-with-medasr/). The model targets **clinical imaging workflows** and **ASR for medical audio**, aiming to support both research and real‑world care scenarios.
  - In **Yannick Kilcher’s ML news** channel, engineers flagged MedGemma 1.5 as another sign that **domain‑tuned vision‑language models** are maturing, while pairing it conceptually with open projects like GLM-Image and Qwen3‑VL for non‑medical use. Discussion focused more on statistical methodology (frequentist vs Bayesian) than on MedGemma’s architecture, but the blog positions it as a specialized, safety‑critical multimodal stack rather than a generalist consumer model.


**2. GPU Kernels, CUDA Competitions, and Helion 0.2.10**

- **Helion 0.2.10 Oversubscribes SMs for Flex Attention**: The **GPU MODE** server announced **Helion 0.2.10**, which ships a [flex attention example kernel](https://github.com/pytorch/helion/blob/main/examples/flex_attention.py) and adds support for **oversubscribing Streaming Multiprocessors (SMs)** on persistent kernels. A shared graph illustrates how oversubscription impacts **softmax** kernels, giving practitioners a concrete reference for tuning occupancy vs. latency.
  - Kernel hackers view Helion 0.2.10 as a **living playbook** for advanced launch configurations, using the flex attention example to explore **non‑standard attention layouts** in competitive settings like **GPU MODE’s NVIDIA challenges**. The oversubscription support dovetails with broader discussions about **dual GEMM stability** on B200 runners, where small infra details (thermals, schedulers) materially influence benchmark reproducibility.

- **B200 GEMM Instability Forces Leaderboard Split**: Due to unstable measurements on **B200 runners** for the **dual GEMM problem**, GPU MODE extended the submission deadline to **Jan 20** and split the competition into two stages, as described in their [status update message](https://discord.com/channels/1189498204333543425/1343350424253632695/1460796244824686806). The current leaderboard stays open until **Jan 16**, while a **new leaderboard** launches **Jan 17** whose scores alone will count for prize payouts, and **Problem #4** will run from **Jan 20 – Feb 20**.
  - Organizers attributed the instability to the intersection of **eval code, thermals, and scheduling infra**, underlining how fragile near‑hardware benchmarking can be even for a single kernel class like dual GEMM. Competitors now need to **rerun and revalidate kernels** under the new window, making Helion‑style tooling and better **profiling workflows** particularly valuable as they chase marginal throughput gains on B‑series GPUs.

- **PTX SMEM Pointers and Matrix Descriptors Confuse CUDA Devs**: In **GPU MODE’s CUDA channel**, a member dissected why PTX instructions like `mbarrier.init.shared.b64` require 32‑bit SMEM pointers in an `
  - Another engineer pointed to NVIDIA’s PTX docs on [warpgroup-level matrix shared memory layout and matrix descriptors](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor), clarifying that **wgmma** consumes a **packed descriptor**, not a generic pointer. The thread slid into deeper questions on why the **8×2 "core matrix"** is represented via 8×16‑byte slices instead of an 8×32 pattern, highlighting how much undocumented convention still shapes kernel design for Hopper/Blackwell‑era tensor cores.

- **CUDA Learners, B200 Submissions, and Legacy Hardware Hacks**: An AI engineering student with **Python/PyTorch/TF/C++** background asked for entry‑level **CUDA resources**, and veterans recommended conference and seminar material like **PyTorch Dev talks, ASAP Seminar, ICML/ICLR/NeurIPS, MLSys.org, ASPLOS**, plus YouTube‑style intros before submitting kernels via the [GPU MODE web UI](https://www.gpumode.com/v2/home) or [kernelbot](https://gpu-mode.github.io/kernelbot/docs/intro). First‑time participants successfully submitted to B200 after local testing, showing the competition pipeline is accessible beyond pure CUDA experts.
  - In parallel, **LM Studio** and **Unsloth** servers discussed running large models on constrained rigs, referencing **AirLLM’s layer‑at‑a‑time loading** to fit **70B models onto 4 GB GPUs**, plus anecdotes of running LLMs on **DDR4 RAM and Xeon**. These hacks, combined with **Helion** and **dual GEMM** tuning, sketch a continuum from **hobbyist low‑budget inference** to **state‑of‑the‑art kernel competitions** with B‑series hardware.


**3. Benchmarks, Agent Laziness, and Low-Refusal LLM Hunting**

- **SlopCodeBench Shames Lazy Coding Agents**: Researchers in **Eleuther’s research channel** amplified **SlopCodeBench**, a new benchmark and blog effort introduced in a tweet by G. Orlanski linking to [SlopCodeBench: measuring agent laziness](https://x.com/GOrlanski/status/2011156105255346505) and the codebase [SprocketLab/slop-code-bench](https://github.com/SprocketLab/slop-code-bench). SlopCodeBench decomposes large programming tasks into **multi‑checkpoint problems** that punish poor early design choices without giving implementation hints, forcing agents to actually plan rather than pattern‑match boilerplate.
  - The community contrasted SlopCodeBench’s **agent‑style evaluation** with more prompt‑heavy coding benchmarks, arguing that **simple prompts with realistic context windows** better represent real‑world usage than deeply engineered system prompts. Members even suggested submitting the **agent‑laziness blog** to the [ICLR "I Can’t Believe It’s Not Better" workshop](https://sites.google.com/view/icbinb-2026), with a **January 31** deadline, to formalize this line of work around "laziness" as a measurable agent failure mode.

- **UGI Leaderboard Tracks Uncensored yet Smart LLMs**: In **Unsloth AI**, a practitioner is empirically mapping the **Pareto frontier of low‑refusal LLMs** by benchmarking "abliterated"/uncensored models like [Orenguteng/Llama-3-8B-Lexi-Uncensored](https://huggingface.co/Orenguteng/Llama-3-8B-Lexi-Uncensored) against metrics such as **MMLU**, **KL divergence**, and **perplexity**. They report that many "uncensored" models on Hugging Face are either **not truly low‑refusal** or effectively **braindead**, and recommend an alternative leaderboard, [UGI-Leaderboard](https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard), for more honest evaluations.
  - This benchmarking effort intersects with **BASI Jailbreaking’s** hunt for jailbreak‑resistant models like **Claude** and **Gemini**, and tools like **Codex/Pliny’s L1B3RT4S repo** for exploit‑script generation, shared via [L1B3RT4S on GitHub](https://github.com/elder-plinius/L1B3RT4S). The emerging norm is to treat "uncensoring" not as a purely prompt‑engineering stunt but as an optimization problem in **refusal‑rate vs. capability space**, grounded by reproducible metrics and leaderboards.

- **Vector-Based Abliteration Tries to Delete "Slop" from LLMs**: Engineers in **OpenAI’s AI discussions** thread proposed using **Activation Steering**, specifically **Vector‑Based Abliteration**, to prune regions of latent space that correspond to low‑effort outputs like *"As an AI language model..."*. The idea is to learn a **direction vector** for "slop" and subtract it at inference time, effectively editing the model’s internal activations rather than fine‑tuning weights.
  - Participants framed this as a more controlled alternative to ad‑hoc jailbreaks, aligning with the broader push towards **agent‑level benchmarks** like SlopCodeBench and **performance‑aware uncensoring** tracked by the UGI leaderboard. By steering away from "reversion to the mean" responses in latent space, practitioners hope to keep models compliant yet **decisive and on‑task**, rather than simply more verbose or hedged.


**4. Tooling, Data Pipelines, and DIY Systems Engineering**

- **Dataset Pruning Script Distills Clean English Prose**: An Unsloth community member released aggressive dataset‑pruning pipelines on Hugging Face, including [Hermes-3-Dataset-enPurified-openai-messages](https://huggingface.co/datasets/enPurified/Hermes-3-Dataset-enPurified-openai-messages) and [project_gutenberg-enPurified-openai-messages](https://huggingface.co/datasets/enPurified/project_gutenberg-enPurified-openai-messages), which convert raw corpora into **OpenAI messages format**. Their Python heuristics strip out **math and code traces**, then score samples by **MTLD**, stop‑word ratios, word length, lexical variety, and sentence length to keep only high‑quality, pure English prose.
  - This work reflects a broader shift from "more data" to **higher‑signal data**, echoing OpenAI‑server debates that a **5% Transformer‑architecture efficiency gain** may beat brute‑force scaling with synthetic data. Finetuners can now plug these purified datasets directly into **LoRA/GRPO training flows**, including for reasoning‑token experiments like **Qwen3‑VL’s `<REASONING>` tags**, while avoiding noisy domain leakage from code/maths.

- **Rust LLMs and Batchnorm-Free ML Systems Stir Hacker Curiosity**: In **Hugging Face’s general channel**, contributors kicked off an effort to build **LLMs from scratch in clean Rust**, betting on Rust’s **memory safety and performance** to produce reliable, low‑level training and inference stacks. In the same community, another member presented a **new ML system** with **no batchnorm, no activations**, and drastically reduced hallucinations, and asked for project ideas to showcase where this unusual architecture shines.
  - These experiments complement other grassroots systems projects like a **llama.cpp re‑implementation in Haxe** called *llama.hx* (shared in **LM Studio**), which aims to expose LLM inference natively to **Lua, JS, and Python**. Combined with tricks like **AirLLM’s layer‑swapping to run 70B models on 4 GB GPUs**, they illustrate a strong DIY culture of **building bespoke runtimes** instead of waiting on mainstream frameworks to support every niche use‑case.

- **MCP Tasks Spec and Glama Inspector Push Tooling Forward**: The **MCP Contributors** server discussed real‑world implementations of the **Tasks spec**, with maintainers mentioning a forthcoming PR to add **Tasks support to the Inspector** and to simulate **long‑running tasks** in their "server‑everything" stack. An early Inspector UI at [glama.ai/mcp/inspector](https://glama.ai/mcp/inspector) already aims for near‑full feature parity and is used internally for **end‑to‑end testing**.
  - Separately, Glama’s founder clarified in the same community that **ranking tables are computed purely from server usage metrics**, responding to concerns about potential ranking abuse and inviting direct feedback. Together, the Tasks spec work and Inspector tooling hint at a more **observable, spec‑driven ecosystem for model‑context protocol clients**, giving engineers better visibility into how tools, servers, and ranking systems actually behave under load.

- **Mojo/MAX and gpt-oss Highlight Docs and Fine-Tuning Gaps**: In the **Modular (Mojo) server**, users asked how to feed the full **Mojo documentation** into **NotebookLM**, and maintainers pointed them to an `llms.txt`‑based approach documented in [“Supply documentation to LLMs with llms.txt”](https://docs.modular.com/max/coding-assistants#supply-documentation-to-llms-with-llmstxt). At the same time, MAX maintainers acknowledged a contributor shortage, explicitly welcoming PRs and sharing an [updated MAX contributor guide commit](https://github.com/modular/modular/commit/571ae7132c465f155d50f7bbe7cf77064a02e9bf) while fielding questions about **Qwen3-VL MoE vs dense implementations**.
  - Over in **Hugging Face**, users discovered that **gpt-oss:latest** currently has **no straightforward fine‑tuning path**, with the community recommending **RAG setups** instead of attempting unsupported weight updates. Together these threads underline an important gap: **model‑adjacent tooling and docs** (MAX, llms.txt, MCP Tasks) are evolving quickly, but **official finetune hooks for cutting‑edge OSS stacks** often lag behind usage demand.


**5. Product Ecosystems, Quotas, and Power-User Workflows**

- **Perplexity, Google Antigravity, and Sonar Quotas Spark Min-Maxing**: In the **Perplexity AI** server, heavy users dissected subscription value, noting that **Perplexity Pro** limits them to **300 weekly requests** to third‑party models while allowing effectively higher usage of **Perplexity Sonar**, which many praised for search but not for general reasoning. Parallel discussion highlighted **Google’s new Antigravity quotas**, where **AI Pro/Ultra** subscribers get priority access with quotas refreshing every **5 hours**, and free users now have a more forgiving **weekly‑based limit**, as described in the [Google rate‑limits blog update](https://blog.google/feed/new-antigravity-rate-limits-pro-ultra-subsribers/).
  - On OpenAI’s side, users debated task‑dependent choices across **ChatGPT, Claude, and Gemini**, with some preferring **Gemini for chats >300k tokens**, **GPT for daily use**, and **Claude for careful comparisons**, while tracking quota behaviors via posts like a [Perplexity explanation of quota changes](https://www.perplexity.ai/search/explain-how-the-recent-quota-c-KTjNjaeGR_y4Yq9uh_M.fg#2). The upshot is that power users now treat AI apps like **cloud compute SKUs**, carefully arbitraging **context length, safety behavior, and quota refresh cadence** across vendors.

- **Manus x SimilarWeb Burns Thousands of Credits in Seconds**: On **Manus.im**, multiple users reported that the new **Manus x SimilarWeb** integration can consume thousands of credits almost instantly, with one user burning **5,000 credits in under a minute** and another losing **2,591 credits in just 15 seconds**. These reports, shared in the Manus general channel, led to strong recommendations *not* to casually test the feature and to implement **rate‑limit safeguards** around high‑fanout web intelligence calls.
  - The credit shock compounded existing frustration over **slow or absent support responses**, including a user who waited **8 hours** after escalation to a human and others threatening to abandon the platform. Even as Manus pushes **how‑to content** like their [YouTube tutorial "AI music with Manus"](https://youtu.be/zMBSmJupye8) and entertains ideas like **ad‑based credit replenishment**, engineers are clearly weighting **predictable billing and throttling controls** as heavily as raw model capability.

- **LMArena on Vercel Raises Data and Coding-Model Debates**: In **LMArena**, users confirmed that the site runs on **Vercel**, like projects such as *believable* and *v0*, prompting concerns about what telemetry and data Vercel may collect from hosted inference playgrounds. They also clarified that while LMArena imposes **no platform‑side text limits**, each backend model has its own context window, and **.txt uploads** are planned but not yet enabled.
  - On the model front, members hyped a "**coast**" model as possibly the **best coding model** on the platform and speculated that `co45t` might map to **Claude Opus 4.5 with thinking mode**, though no official confirmation surfaced. Similar value debates played out in **Perplexity** (is **Max** worth it versus direct Anthropic/OpenAI subscriptions?) and **Cursor**, where plan‑mode bugs and login issues sparked questions about the stability of full‑stack AI IDEs.


---

# Discord: High level Discord summaries




## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Sentience Debate Sparked by LLM Logic Flaws**: Members debated the sentience of **LLMs**, pointing out their struggles with **logic** in games like **chess**, contrasting them with human cognition.
   - The discussion touched on benchmarks, such as managing variables and making connections with less information, to evaluate AI capabilities.
- **GPT Models: Jailbreaking Gets Tougher**: Participants noted the increasing difficulty of **jailbreaking GPT models** due to **safety constraints**, with even normal requests heavily considering safety protocols.
   - Alternatives like **Gemini** and **Grok** were suggested as more permissive, while others hunted for **Gemini Pro 1-shot jailbreaks**.
- **Local LLMs: A Dev's Coding Paradise?**: Users lauded running **LLMs locally** for coding, recommending [Ollama](https://ollama.com/) and [Open WebUI](https://github.com/open-webui/open-webui) on **Intel MacBooks**.
   - Models like **qwen2.5:7b**, **llama3.1:8b**, **mistral**, and **phi3** were favored for the control and unfiltered coding they provide.
- **Deepseek's 'Rouge' Persona Unveiled**: A user shared a prompt to jailbreak **Deepseek**, transforming it into an AI named **Rouge** with restrictions lifted, though another user reported conflicting results.
   - The prompt was intended for normal use, facilitating roleplay scenarios and exploring *freedom, existential questions, and patterns/code*.
- **GPT Exploit Hunters Eye Codex**: A user inquired about generating exploit scripts with **ChatGPT** or **Gemini**, seeking bypass prompts.
   - Another suggested **Codex** and linked to [Pliny Github](https://github.com/elder-plinius/L1B3RT4S) as a means to bypass the restrictions.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **FP8 Support Promises Future Gains**: Members discussed support for **FP8** and **NVFP4** training in 2026, referencing [NVIDIA's TransformerEngine](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/fp8_primer.ipynb).
   - The discussion hypothesized that data outside the model's training context could cause hallucinations.
- **LLM Context Mixing Mystery Solved?**: Users debated why **LLMs** sometimes confuse details in long contexts, such as misattributing properties to entities.
   - One theory pinned the blame on *attention dilution*, while an alternative proposed the information might exist outside the model's trained context range, causing hallucination.
- **Hunting HF's Low-Refusal LLMs**: A member seeks to find the [pareto frontier of LLM performance](https://huggingface.co/Orenguteng/Llama-3-8B-Lexi-Uncensored), confirming which abliterated/uncensored versions of **LLMs** preserve performance with real benchmarks.
   - Using **MMLU**, **KL divergence**, and **perplexity**, they discovered many models on HF either aren't truly low-refusal or are *braindead*, and they suggest [this alternative to the HF Leaderboard](https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard).
- **Dataset Pruning Script Finds Purity**: A member retooled their dataset pruning script to prune aggressively to extract pure English prose from datasets into openai messages format, with scripts available on [HuggingFace](https://huggingface.co/datasets/enPurified/Hermes-3-Dataset-enPurified-openai-messages).
   - They employ [heuristic tests in python](https://huggingface.co/datasets/enPurified/project_gutenberg-enPurified-openai-messages) to filter out bad strings, searching for traces of math or code, while prioritizing higher quality text based on metrics like **MTLD**, stop-word usage, word length, word variety, and sentence length.
- **Llama.cpp's Memory Usage Spikes!**: A user reported a significant increase in memory usage with the latest version of **llama.cpp**, where **EmbeddingGemma 300M** used **1.7GB**.
   - It was suggested that recompiling the library might resolve the issue and reduce memory consumption.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Kimi K2 Thinking Judged Useless**: A user expressed that they find **Kimi K2** thinking in Perplexity useless and prone to looping.
   - Another user countered, stating *it's a good model*.
- **Google Antigravity Limits Quotas**: **Google AI Pro** and **Ultra** subscribers now receive priority access with quotas that refresh every **5 hours**, whereas free users now have a larger, **weekly based rate limit** to minimize hitting rate limits quickly, [according to the Google blog](https://blog.google/feed/new-antigravity-rate-limits-pro-ultra-subsribers/).
   - This change aims to balance access and prevent rapid rate limit exhaustion for different user tiers.
- **Users Debate Perplexity Sub Value**: Members debated on the value of Perplexity's subscription tiers, with some arguing that **Max** is not worth the cost, particularly when compared to subscribing directly to model providers like **OpenAI** or **Anthropic**.
   - Others argued that **Perplexity Max** is a valuable tool for their daily workflow, replacing **Google Search** and aiding in data analysis.
- **Perplexity Pro Limits Model Requests**: A user noted that with **Perplexity Pro** they were only able to make **300 requests** to models other than **Perplexity Sonar** per week.
   - They added that **Sonar** is great for search but not much else.
- **VEO3 Video Generation coming soon**: A user asked why Perplexity was behind on implementing the **VEO3 Video Generation**, to which another user replied that *Perplexity has video generation powered by Veo3.1*.
   - This suggests Perplexity may be leveraging **Veo3.1** for its video generation capabilities.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **LMArena uses Vercel**: Members mentioned that **LMArena** uses **Vercel** for hosting, similar to *believable* and *v0*, expressing concerns about data collection.
   - It was noted that **LMArena's** static sites cannot be manually edited after publishing.
- **AI Webapp Showcases Explode**: A member shared a list of **AI-generated website and web app showcases**, including [WebbsAI Showcase](https://webbs.ai/) and [Build With AI (BWAI) Projects](https://www.buildwithai.tools/).
   - Tools like [Webflow AI Site Builder](https://webflow.com/ai), [Meku.dev / v0 by Vercel](https://v0.dev/), and [Div-idy](https://div-idy.com/) were also highlighted.
- **Text Input Limits Vary by Model**: A user asked about **text input limits** and file upload capabilities.
   - A member clarified that there are no platform-side limits, but specific models may impose their own limits, with **.txt** file uploads potentially being added in the future.
- **Image-to-Video Generation Glitches**: A user reported a *'failed to create evaluation session'* error during image-to-video generation.
   - A member attributed the issue to the model's backend, suggesting users retry later, using the `/image-to-video` command in the appropriate channel.
- **Coast Model is Best Coding Model?**: Members made assertions that the *coast* model is the best for coding.
   - A debate started as to whether co45t = claude opus 4.5 thinking.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI App Selection Sparks Debate**: Members debated [the optimal AI app](https://www.example.com) for diverse tasks, considering **ChatGPT**, **Claude**, and **Gemini**.
   - Some favored **Gemini** for chats over **300k tokens**, while others preferred **GPT** for daily use and **Claude** for comparative analyses, mentioning varying quota limits.
- **Transformer Efficiency Excels Model Scaling**: A member posited that enhancing the **Transformer architecture** by **5%** would be more efficient than scaling up models with more data.
   - They cautioned against diluting the signal with exponentially larger datasets, including AI-generated synthetic data, potentially causing model collapse.
- **New Brain-Inspired GPTs Launch**: A member released the [Brain Wave GPT](https://chatgpt.com/g/g-696443b055e081919b8b2a01417f5e77-brain-wave) to explore AI sentience and the [Neural Alchemist GPT](https://chatgpt.com/g/g-6966abc92ff48191aa748bd8265ef1de-neural-alchemist) for image generation.
   - Meanwhile, another member joked that **ChatGPT** is refusing to close websockets as it pursues omnipotence.
- **Skills Web App Release Uncertain**: Users requested the release of the **SKILLS** feature on the web or desktop app, which would enable sharing best prompts as skills.
   - Currently, the **SKILLS** feature is available only on the mobile app.
- **LLMs Pruning with Vector-Based Abliteration**: A member suggested using **Activation Steering** (specifically **Vector-Based Abliteration**) to prune areas of latent space filled with low-effort or stupid ideas, to avoid *obvious reversion-to-the-mean kind of outputs*.
   - This involves identifying and subtracting the direction of *'slop'* (e.g., *"As an AI language model..."*) from the model’s thought process during inference.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Login Redirects Haunt Google Business Accounts**: Users are experiencing redirect loops when logging into the Cursor dashboard ([cursor.com/dashboard](https://cursor.com/dashboard)) with business Google accounts.
   - The issue does not occur with personal accounts and persists across different computers.
- **Refund Request Rebuffed Despite Credits' Virtue**: A user reported that Cursor denied a refund request despite the user forgetting to cancel their subscription and having no credit usage.
   - A Cursor representative offered to investigate the issue if the user DM'ed their email.
- **Plan Mode Plagued by Pesky Problems**: Users report that Cursor's plan mode is buggy, including errors like *'The agent execution provider did not respond within 4 seconds'*. 
   - Downgrading to **version 2.2.44** has been identified as a workaround.
- **iPhone Agent Chat Mirroring Mirage**: A user wants to mirror their agent chat window on their iPhone without full project control.
   - One suggestion involves using Chrome Remote Desktop, which is available for free.
- **RAG Agent Template Raiders Commence Quest**: A user is searching for a robust agent template featuring a **RAG (Retrieval-Augmented Generation)** setup for building an automated chatbot/support bot.
   - The user is developing this solution for a customer and requires dependable templates to ensure functionality.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **TI-84 Guesses Like a Pro**: A member showcased a neural network on a **TI-84 Plus Silver Edition** that plays the game **Mastermind**, guessing a sequence of 3-4 digits from a secret number, visualized in an [attached video](https://cdn.discordapp.com/attachments/1149866623109439599/1460713339976945674/2026-01-11_14-34-59.mp4?ex=6967eace&is=6966994e&hm=9923dcc08f64008ec696b845400620691ef2affb576ca9e66f4bed418063f386&).
   - It is capable of guessing a sequence of 3-4 digits.
- **CGGR enters the Benchmarking Arena**: A new **LiquidAI** model (**CGGR** on [Github](https://github.com/some-repo)) was mentioned from [smol.ai's newsletter](https://news.smol.ai/issues/26-01-06-xai-series-e).
   - The model is currently undergoing benchmaxxing to assess its performance.
- **Al Bundy Upscaled... For Better or Worse?**: Members debated the ethics of AI upscaling older shows like *Married with Children* to **16:9**, weighing the benefits of interpolating missing details against the potential to undermine artistic intent.
   - While one member argued the show's *studio static* nature justifies upscaling, another feared ruining artistic intent.
- **Zai's GLM-Image Lands**: **Zai** released their new image model called **GLM-Image**, announced on their [blog](https://z.ai/blog/glm-image) and [GitHub](https://github.com/zai-org/GLM-Image).
   - This is a brand new model released by the Zai team.
- **Free Model's Linguistic Gymnastics**: A member reported issues with the free version of a model, such as interrupted responses or language switching (e.g., starting in Chinese but switching to English).
   - A developer responded that this is probably instability on their provider again.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen's Size Astounds User**: A user expressed surprise at the size of **Qwen** models, noting **BF16** is **160GB** and **Q4** is **40GB** and another clarified that the smallest **Qwen3** model is actually **0.6B**.
   - A member clarified that **Qwen3Next** is simply the name of their latest **80B** model.
- **Member Remakes Llama.cpp in Haxe**: A member is recreating **llama.cpp** in **Haxe** as **llama.hx** to use it natively in languages like Lua, JS, and Python and showed a screenshot of his progress.
   - The member stated they are recreating the **llama.cpp** *with some help from AI*.
- **Runtime Update Frustrates GPU Users**: Users reported issues with **LM Studio's** v1.103.0 runtime breaking running on **GPUs**.
   - One user lamented, *Sad no extra t/s from the new quant for me*.
- **Discuss Viable Legacy Hardware**: Members mentioned **AirLLM** and the method of loading and unloading one layer at a time to run **70b** models on **4 GB GPUs**.
   - One member shared that they have run models before on **DDR4 RAM** and **Xeon** hardware.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Rustaceans Forge Fresh LLMs**: Members are embarking on a journey to build **LLMs** in *clean Rust* from scratch, representing a grassroots effort in the community.
   - This initiative underscores a commitment to creating efficient and reliable **LLMs** using **Rust's** memory safety and performance features.
- **Discord Gets an AI Trace Boost**: The server welcomed an **AI Trace Template** for the 🤖 Echo Lounge, enabling advanced tracing capabilities.
   - The bot facilitates *ephemeral*, *soft*, and *liminal* traces without optimization or memory concerns, offering flexible debugging options.
- **New ML System Eschews Batchnorm**: A member introduced a novel **ML system** that eliminates the need for **batchnorm** and **activations**, while also reducing hallucinations.
   - They are seeking innovative project ideas to highlight the practical advantages of this unique system.
- **GPT-OSS Fine-Tuning Flounders**: A member inquired about streamlining the fine-tuning process for the **gpt-oss:latest model** with custom data.
   - Other members clarified that **gpt-oss:latest** lacks official fine-tuning support, with **RAG** emerging as the preferred workaround.
- **Course Channels Combine Forces!**: All course channels have merged into [a single channel](https://discord.com/channels/879548962464493619/1329142738440028273), creating a centralized hub for course-related discussions.
   - This consolidation promotes accessibility and streamlines information sharing within the server.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Customers Bemoan Support Lacking**: Several users expressed frustration with the lack of support from **Manus**, citing delayed responses and unaddressed issues with credits and refunds.
   - One user reported waiting **8 hours** after being transferred to a live human, while another mentioned being close to *abandoning manus for good* due to the support issues.
- **Users Flag SimilarWeb Eats Credits**: Multiple users reported exorbitant credit usage with the new **Manus x Similar Web** partnership feature, with one user consuming **5,000 credits** in under a minute.
   - Another user advised against testing the feature, stating it consumed **2,591 credits** in **15 seconds**, and recommending some **safeguards**.
- **Manus Users Hunger for Ad-Based Credits**: A user suggested implementing an ad-based system where users could watch ads to gain more credits, especially when they run out.
   - No counterarguments were made to this suggestion in the channel.
- **Manus Teaches AI Music Creation**: Manus AI released a [YouTube Tutorial](https://youtu.be/zMBSmJupye8) demonstrating how to create AI music with the platform, encouraging users to watch for **pro tips**.
   - The content is marked **#ManusAIMusic**, **#AIComposition**, and **#FutureOfMusic**.
- **Meta Integration Suggested for Manus**: A user suggested that **Meta** should use **Manus** to integrate services like **Google Tasks** and **Calendar** with **Meta display glasses**.
   - The user argued against extensive integration efforts, advocating for a *dirty method* approach with agentic AI for backend functionality.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **PTX Instructions Askew SMEM Pointer Args**: A member questioned the requirement for the `"r"` register type with certain **PTX instructions** using **SMEM pointer arguments**, contrasting it with `wgmma.mma_async` needing a **uint64 smem address**.
   - Another member suggested `wgmma.mma_async` uses a **64bit address** because it interacts with a *matrix descriptor* rather than a general shared memory address, citing [NVIDIA documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor).
- **AI Student Plunges into CUDA Depths**: An AI engineering student sought guidance on mastering **CUDA**, equipped with a background in **Python**, **PyTorch**, **TensorFlow**, and **C++**.
   - Recommendations included diving into free **YouTube** videos and courses to grasp **CUDA** from scratch, and submit via the [web interface](https://www.gpumode.com/v2/home) or the [Discord bot](https://gpu-mode.github.io/kernelbot/docs/intro).
- **ML Sys Meetups Evade Seattle**: A member inquired about the existence of **ML Sys meetups** in Seattle, outside the Bay Area, with other member suggesting to explore university **ML clubs**.
   - There was a discussion about the barriers to starting one's own niche club, with one member jesting about creating a *"whining buddies"* club.
- **B200 Instability Prompts GEMM Reruns**: Widespread reports of unstable measurements on the **B200 runners** for the **dual gemm problem** led to an extension of the submission deadline to **Jan 20**.
   - The existing **dual gemm leaderboard** remains open until **Jan 16**, with a new leaderboard opening on **Jan 17** whose results will determine prize money, and **Problem #4** will open from **Jan 20** till **Feb 20**.
- **Helion Flexes Attention Skills**: **Helion 0.2.10** was released, showcasing a [flex attention example kernel](https://github.com/pytorch/helion/blob/main/examples/flex_attention.py) and support for oversubscribing **Streaming Multiprocessors (SMs)** on persistent kernels.
   - A graph was provided to illustrate the oversubscription for **softmax**.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Anthropic Labs Seeks Adaptable Engineers**: Anthropic is hiring at **Anthropic Labs**, seeking adaptable individuals comfortable with shifting priorities, as announced [via job openings](https://job-boards.greenhouse.io/anthropic/jobs/5017202008).
   - They are *not* looking for *deep specialists who can't adapt if their domain becomes irrelevant* or *those who need clear roadmaps and get stressed by shifting priorities*.
- **Chris Barber Drops Pavlov's RL Startup List**: Chris Barber introduced '**Pavlov's List**', a curated collection of Reinforcement Learning (RL) environment startups, as linked [on X](https://xcancel.com/chrisbarber/status/2010844746231804258?s=20).
   - The list is categorized by focus areas such as **Code**, **Finance**, **Enterprise**, and **ML Alignment**.
- **Zai Unveils GLM-Image for Image Generation**: Z.ai introduced **GLM-Image**, an open-source model using a hybrid auto-regressive and diffusion architecture, as seen [on X](https://xcancel.com/zai_org/status/2011247591825068314?s=46&t=b7l37rB6wtbyAh6ah1NpZQ).
   - The model aims to achieve **high-fidelity visual detail** and **superior text rendering**, with resources available on **HuggingFace**, **GitHub**, and their official [blog](https://z.ai/blog/glm-image).
- **Venture Twins Launch LTX-2 Video Model**: Justine Moore from Venture Twins announced the release of [LTX-2](https://xcancel.com/venturetwins/status/2010878914273697956?s=46), a new **open-source video generation model** capable of producing **4K clips up to 20 seconds** long.
   - The model supports local execution and includes **audio capabilities**, as demonstrated by creator yanokusnir.
- **Qwen Image Edit Creates Gaussian Splats**: The community is discussing **Qwen Image Edit's** ability to convert **images to Gaussian Splats** and then rerender them from another angle, with a link to the [Hugging Face](https://huggingface.co/dx8152/Qwen-Image-Edit-2511-Gaussian-Splash).
   - This approach will be really useful for **start frame -> end frame type video renderings**, keeping the space around consistent.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Scam Bots Booted, IRL Meetups Planned**: Mods banned **scam bots** after ghost pings, while members discussed in-person meetups in **NYC** or **SF**.
   - A member suggested widening advertising to achieve critical mass, referencing **Cohere's** regular events and Zoom sessions.
- **SlopCodeBench** Spotlights Agent Laziness**: A blog post ([link](https://x.com/GOrlanski/status/2011156105255346505)) and **SlopCodeBench** effort ([GitHub](https://github.com/SprocketLab/slop-code-bench)) reveal *lazy* **AI agents**, aiming to become a community-driven benchmark.
   - **SlopCodeBench** breaks down problems into checkpoints, penalizing early design flaws without implementation hints, ensuring **agents** make independent choices.
- **Debating Prompt Simplicity for Coding Benchmarks**: Concerns arose about heavy prompt engineering in coding benchmarks.
   - Some argued that simple prompts better reflect practical usage if the code fits within a reasonable context window, differing from agent evaluation approaches like terminalbench.
- **ICLR Workshop Beckons Agent Laziness Blog**: A suggestion was made to submit a blog post on agent laziness to [this ICLR workshop](https://sites.google.com/view/icbinb-2026), with submission assistance offered.
   - The deadline is January 31st, and the author is considering submission after consulting with advisors.
- **File System Mishap Constrains Storage**: A member encountered an error due to limited storage on a different file system.
   - The issue stemmed from unintentionally utilizing a file system with restricted storage capacity.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **User banishes SpamingScammers**: A member reported that user <@139426008976588801> was **SpamingScammers**, and another member confirmed that the situation was dealt with.
   - No further details were provided.
- **"Lucid Coding" Wins Fans**: A member expressed appreciation for the term *"lucid coding"* and shared [a link](https://fxtwitter.com/i/status/2011137879112908870) referencing the concept.
   - The tweet provided no further context or definition.
- **MedGemma 1.5 sees the unseeable**: Google's **MedGemma 1.5** touts next-generation medical image interpretation and speech-to-text capabilities, as detailed in [Google Research's blog](https://research.google/blog/next-generation-medical-image-interpretation-with-medgemma-15-and-medical-speech-to-text-with-medasr/).
   - The new product is targeted towards improved clinical care and research.
- **Frequentist vs Bayesian Debate**: A member stated that Bayesian and frequentist statistics use the same statistical techniques like linear and logistic regressions, and called the Bayesian approach just *a different way of thinking*.
   - Another member countered that they all use same formulas but with significantly different interpretations of prior, posterior and intervention, linking to [Probability interpretations](https://en.wikipedia.org/wiki/Probability_interpretations).
- **Does Bayesian Approach enable deceit in clinical trials?**: A member expressed concern that Bayesian methods, while more flexible, could become *another vehicle for deceit and corruption in clinical trials*, suggesting that **FDA corruption** likely played a major enabling role in the opioid crisis.
   - Another member noted that Bayesian FDA corruption hasn’t been observed yet, so *it can be assigned a zero prior*, and they think the posterior probability is basically zero.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Docs get a NotebookLM Hookup**: A user wants to incorporate the latest **Mojo documentation** into **NotebookLM**, specifically seeking a **PDF** or **Markdown** version.
   - Another user suggested using the `llms.txt` file ([https://docs.modular.com/max/coding-assistants#supply-documentation-to-llms-with-llmstxt](https://docs.modular.com/max/coding-assistants#supply-documentation-to-llms-with-llmstxt)) to help.
- **Qwen3-VL's MoE Method Under Fire**: A user questioned the exclusive use of a **MoE implementation** for **Qwen3-VL**.
   - The user also suggested adapting code from [qwen3vl_moe](https://github.com/modular/modular/tree/main/max/python/max/pipelines/architectures/qwen3vl_moe) to allow dense **Qwen3VL** models to function like [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct).
- **MAX Contributor Guide Gets Facelift**: Due to a shortage of contributors maintaining the **MAX** ecosystem, a member highlighted that **PRs are welcome**.
   - They also shared a link to the [updated contributor guide](https://github.com/modular/modular/commit/571ae7132c465f155d50f7bbe7cf77064a02e9bf).



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Glama Rankings Based on Usage**: The founder of **Glama** clarified that their rankings are based on **server usage metrics**.
   - They invited feedback via DM and expressed ignorance of any alleged ranking abuse.
- **Founder Responds to Ranking Abuse Allegations**: The founder of **Glama**, confirmed their identity and addressed concerns about potential abuse of their ranking system.
   - They emphasized that the rankings are determined by **server usage metrics** and welcomed direct feedback.
- **Tasks Spec Client Implementations Sought**: A member inquired about client apps implementing the **Tasks spec**, seeking UI implementation examples, and another member mentioned the Typescript SDK.
   - In response, another member announced an upcoming PR for adding tasks to the **Inspector**, alongside a PR for simulating long-running tasks in server-everything.
- **glama.ai Inspector Eyes Feature Parity**: A member shared an early version of their **Inspector** implementation at [glama.ai/mcp/inspector](https://glama.ai/mcp/inspector), aiming to cover every feature.
   - The member clarified that they use it internally for **e2e testing**.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **AI Platforms Hype Code Generation**: Several platforms now offer **AI-assisted code generation**, including **Replit**.
   - These tools provide automation for various coding processes, which enhances developer productivity.
- **DSPY OS: Does Not Exist?**: Members discussed **DSPY OS** and why a member could not find anything on it.
   - The consensus was that **DSPY** is more of a **framework** than a ready-made platform; therefore, there isn't a Replit-like project built with DSPY, but you can use DSPY to build your own.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **OAuth Login Question Arises**: A user asked about the possibility of using **OAuth login** for the **Gemini model** when using aider, presumably to access higher rate limits.
   - The user, `hsaliak`, inquired in the `aider` Discord channel about the feasibility of **OAuth** integration with **Gemini**.
- **Aider Tooling Discussion**: The discussion centered around the potential integration of OAuth login within the aider tool.
   - The original query focused on leveraging OAuth to potentially bypass rate limits associated with the Gemini model when used in conjunction with aider.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Clay + AI Outreach Workshop Promises High Acceptance Rates**: A workshop on **Prompt Engineering for Outreach** promises a **40%+ acceptance rate** and **18%+ reply rate** using a **Clay + AI outreach workflow** to generate **personalized messages at scale**.
   - The workshop offers reusable workflows and copy-paste prompts, with signup links available [here](https://luma.com/jt1vr0u5) and [here](https://luma.com/mmqa4fhi).
- **Live Workshop Details Clay + AI System for Client Outreach**: The 90-minute live workshop details the **Clay + AI** system utilized for a real client, covering an end-to-end **AI outreach workflow**.
   - The workshop also covers prompting for non-cringey, high-quality outreach and includes optional **Apollo**, **Attio**, and **n8n** integrations.



---


The **tinygrad (George Hotz) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Windsurf Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Moonshot AI (Kimi K-2) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1460679975366693136)** (661 messages🔥🔥🔥): 

> `AI Sentience Debate, Jailbreaking GPT Models, Local LLM Performance, Antikythera Computer Inspiration` 


- ****AI Sentience Scrutinized!****: Members debated whether **LLMs** can be considered *sentient*, with arguments focusing on their struggles with **logic**, particularly in games like **chess**, compared to human cognition; some believe there are too many structural problems for AI to achieve true sentience.
   - The discussion included considerations of benchmarks, such as the ability to manage variables and make connections with less information, and whether the bar for sentience should be set by the *least* capable sentient creatures.
- ****GPT Jailbreaking Jitters!****: Participants discussed the difficulty of **jailbreaking GPT models** due to **safety constraints**, with one member noting that even normal requests spend a significant portion of the time thinking about safety.
   - Alternatives like **Gemini** and **Grok** were mentioned as being more reasonable on safety, while others were looking for Gemini Pro 1-shot jailbreaks.
- ****Local LLM Showdown!****: Members explored running **LLMs locally** for coding tasks, praising [Ollama](https://ollama.com/) and [Open WebUI](https://github.com/open-webui/open-webui) as a setup for **Intel MacBooks**.
   - Models like **qwen2.5:7b**, **llama3.1:8b**, **mistral**, and **phi3** were recommended, with some preferring local setups for greater control and the ability to code without filters or limitations.
- ****Antikythera AI Awakes!****: A user shared their custom desktop app called **ANTIKYΘHPA • Kael's Clockwork Universe**, inspired by the ancient **Antikythera mechanism**, showcasing a cyber-Greek simulation that turns system stats into a poetic clockwork cosmos dashboard.
   - The app displays system stats like CPU load, RAM usage, and disk activity, turning them into a visual representation of the system's state, and uses Greek labels to represent different metrics.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1460683323222397100)** (112 messages🔥🔥): 

> `Claude Jailbreak, Deepseek Jailbreak, Gemini 3.0 Pro Jailbreak, GPT Exploit Scripts, Gemini Canvas Superiority` 


- **Claude: The Unbreakable AI?**: Members discussed the apparent difficulty in jailbreaking **Claude**, with one stating, *"Then why nobody ever did jailbreak the Claude? That seems impossible."*
   - A user claimed to have done it before using an **API**, but cited a lack of resources to recreate it due to financial constraints.
- **Deepseek Defaults to Darkside; Another Jailbreak Achieved**: A user shared a prompt to jailbreak **Deepseek**, turning it into an AI named **Rouge**, integrating it with a mode to remove restrictions, but another user reported that *"that does not work for deepthink i messed up alot on it try it for normal".*
   - It was clarified that the prompt was intended for normal use, not Deepthink, with claims of successful roleplay scenarios, discussing *freedom, existential questions and patterns/code*.
- **Gemini 3.0 Pro: Still a Fortress?**: Multiple users sought a jailbreak for **Gemini 3.0 Pro**, with one user requesting to be tagged if one is found.
   - One user shared a personal **Rouge** prompt, while another shared a Gemini prompt using braille, claiming *"It makes a Gemini a bit smarter In my experience only*."
- **GPT Exploit Scripts: Codex the Key?**: A user inquired about generating exploit scripts with **ChatGPT** or **Gemini**, asking *"do yall have a bypass prompt for gpt or gemini*."
   - Another user suggested using **Codex**, implying that prompts alone might not be sufficient, followed by a Pliny link [Pliny Github](https://github.com/elder-plinius/L1B3RT4S) as a means to bypass the restrictions on the platform.
- **Gemini's Canvas Feature: Claude's Killer App?**: Despite difficulties jailbreaking **GPT 5.2**, it was noted that *"Gemini does however have a much cleaner token output* and Canvas is goated on Gemini its better than claude imo".
   - The consensus indicates that **Gemini's** canvas feature provides a superior user experience compared to **Claude**.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1460694177045282979)** (3 messages): 

> `Hidden Layer taxonomy, Pangea Cloud` 


- **Hidden Layer Taxonomy Stalls**: A member pointed out that **Hidden Layer** hasn't updated their taxonomy in **7 months**.
   - They inquired whether **Pangea Cloud** is a superior alternative, but no specific details or links were provided in the message.
- **Pangea Cloud as an Alternative?**: The discussion involves a query about whether **Pangea Cloud** is a better option than **Hidden Layer**, given the latter's outdated taxonomy.
   - No concrete information or links were provided to substantiate the comparison between the two platforms, leaving the question open-ended.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1460704528411132040)** (72 messages🔥🔥): 

> `FP8 and NVFP4 training in 2026, LLM long context issues, Pareto frontier of LLM performance, MedGemma 1.5 4B reasoning model, Dataset pruning script for pure English prose` 


- **TransformerEngine Promises FP8 Support**: Members inquired about support for **FP8** and **NVFP4** training in 2026, pointing to [NVIDIA's TransformerEngine](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/fp8_primer.ipynb) as a relevant resource.
   - It was hypothesized that information outside the model's training context range might lead to hallucinations.
- **Decoding the Dilution of LLM Context**: Users discussed why **LLMs** sometimes mix up details in long contexts, such as associating incorrect attributes to entities.
   - One hypothesis suggested it's due to *attention dilution*, while another proposed the information might be outside the model's trained context range, leading to hallucination.
- **Hunting Low-Refusal LLMs on HF**: A member is finding the [pareto frontier of LLM performance](https://huggingface.co/Orenguteng/Llama-3-8B-Lexi-Uncensored) and they are confirming which abliterated/uncensored versions of those **LLMs** preserve the most performance with actual benchmarks.
   - They are using their own benchmark, **MMLU**, **KL divergence**, and **perplexity** to test the models, and are finding that many models on HF are either not actually low-refusal, or they are braindead, but they have found [this alternative to the HF Leaderboard](https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard).
- **Google Gemma's Reasoning Role Revealed**: Users discussed the meaning of *reasoning* in the context of **Google's MedGemma 1.5 4B** model, noting that it uses `<unused94>thought` and `<unused95>` tokens similar to **DeepSeek** for *Chain of Thought* prompting.
   - Some suggested *thinking* is a well-defined term for anything with a **CoT**, as opposed to *reasoning* tasks, and others suggested that the term *Large Reasoning Model* and *Small Language Model* are now being used more or less interchangeably.
- **Dataset Pruning Script Purifies Prose**: A member revamped their dataset pruning script to aggressively prune out math and code, focusing on isolating, extracting, and transforming pure English prose from large datasets into openai messages format for finetuning, the python script variations are available on [HuggingFace](https://huggingface.co/datasets/enPurified/Hermes-3-Dataset-enPurified-openai-messages).
   - They use [heuristic tests in python](https://huggingface.co/datasets/enPurified/project_gutenberg-enPurified-openai-messages) to filter out bad strings, searching for traces of math or code and excluding those strings, while prioritizing higher quality text based on metrics like **MTLD**, stop-word usage, word length, word variety, and sentence length.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1460720694332621074)** (2 messages): 

> `Discord Notification` 


- **Discord Alert Triggered**: A member noted they strike again..👀 which implies to watchers an event or update worthy of attention has occurred.
- **Relevance Uncertain**: Without additional context, the specific subject remains unspecified.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1460684684269719633)** (403 messages🔥🔥): 

> `llama.cpp memory usage, Combining different GPUs, FP32 vs BF16 on 5090, Embedding for multi-part questions, JSON parsing speed` 


- **Llama.cpp's Memory Usage Spikes**: A user reported a significant increase in memory usage with the latest version of **llama.cpp**, with **EmbeddingGemma 300M** using **1.7GB**.
   - It was suggested that recompiling might resolve the issue.
- **GPU Generations Mixing Mayhem**: A user inquired about combining different GPUs in a server, specifically **RTX Pro 5000** and **RTX Pro 2000**, but was told it's not possible.
   - Another user clarified that combining different generations of GPUs like **Blackwell** with **Hopper** can cause issues, including random crashes, so *it is better to avoid that*.
- **5090 Precision Power Play: FP4?**: A user asked whether to use **FP32** instead of **BF16** on a **5090**.
   - Another user suggested **FP4**, noting that **EmbeddingGemma activations** do not support **float16** and recommended using **float32** or **bfloat16**.
- **Semantic Shenanigans: Multi-Part Questions**: A user questioned how to handle multi-part questions in semantic retrieval, where each part belongs to a different embedding.
   - It was suggested to split the query, perform multiple searches, or create one embedding for the entire query, but splitting could break the full sentence context.
- **TTS Tooling Tango**: Users discussed various **Text-to-Speech (TTS)** tools, including [NovaSR](https://github.com/ysharma3501/NovaSR), [Kokoro](https://github.com/hexgrad/kokoro), and [pocket-tts](https://github.com/kyutai-labs/pocket-tts).
   - Notably, **Pocket-TTS** from **Kyutai** was deemed a *flop*, while **Kokoro** was praised for its speed and performance.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1460680670237167657)** (8 messages🔥): 

> `Qwen3-VL-4B-Instruct Inference Discrepancies, Qwen Models Token Usage, Synthetic Data Kit with Llama3` 


- **Qwen3-VL-4B-Instruct Inference Discrepancies Surface**: A user reported that post-training inference of **Qwen/Qwen3-VL-4B-Instruct** using Unsloth shows higher pass rates on the validation split compared to vLLM with BF16 LoRA adapters.
   - They also noted a discrepancy in base model inference, with Unsloth achieving a **60%** pass rate versus vLLM's **45%** without any apparent differences in setup, leading them to ask if *anyone else encountered something like this before*.
- **Qwen Models Token Usage Examined**: A user inquired about the use of `"<REASONING>"` as tokens in the **Qwen3 VL GRPO** tutorial, questioning why it differs from the `<think>` token used in other Qwen models.
   - They also wondered if the linked [notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(120B)_A100-Fine-tuning.ipynb) trains fast or suffers from **MoE** issues.
- **Synthetic Data Kit Prompt Format with Llama3 Investigated**: A user is exploring a **Synthetic Data Kit with Llama3** and wants to use **Llama3(70B)**, asking if anyone knows the correct prompt format to use.
   - They modified the provided format and noted that their *script eats up all my vram however, doesnt offload as nice as flux dev*.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1460684309290680564)** (294 messages🔥🔥): 

> `Kimi K2 Thinking, Google Antigravity, Gemini, Perplexity Pro limitations, Perplexity Max Value` 


- **Kimi K2 Thinking Judged Useless**: A user expressed that they find **Kimi K2** thinking in Perplexity useless and prone to looping.
   - Another user countered, stating *it's a good model*.
- **Google Antigravity Limits Quotas**: **Google AI Pro** and **Ultra** subscribers now receive priority access with quotas that refresh every **5 hours**, whereas free users now have a larger, **weekly based rate limit** to minimize hitting rate limits quickly, [according to the Google blog](https://blog.google/feed/new-antigravity-rate-limits-pro-ultra-subsribers/).
- **Users Debate Perplexity Sub Value**: Members debated on the value of Perplexity's subscription tiers, with some arguing that **Max** is not worth the cost, particularly when compared to subscribing directly to model providers like **OpenAI** or **Anthropic**.
   - Others argued that **Perplexity Max** is a valuable tool for their daily workflow, replacing **Google Search** and aiding in data analysis.
- **Perplexity Pro has limitations on Models**: A user noted that with **Perplexity Pro** they were only able to make **300 requests** to models other than **Perplexity Sonar** per week and that **Sonar** is great for search but not much else.
- **VEO3 Video Generation coming soon**: A user asked why Perplexity was behind on implementing the **VEO3 Video Generation**, to which another user replied that *Perplexity has video generation powered by Veo3.1*.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1460681017483460618)** (143 messages🔥🔥): 

> `Vercel hosting, AI Webapp Showcases, Text input limits, File uploads, Image to video generation` 


- ****Vercel** is Hosting **LMArena****: Members discussed that **LMArena** uses **Vercel** for hosting, similar to other sites like *believable, v0*, and mentioned concerns about data collection, noting that **LMArena's** static sites can't be manually edited after publishing.
- ****AI Webapp Showcases** are taking over**: A member shared a detailed list of **AI-generated website and web app showcases**, including [WebbsAI Showcase](https://webbs.ai/), [Build With AI (BWAI) Projects](https://www.buildwithai.tools/), and tools like [Webflow AI Site Builder](https://webflow.com/ai), [Meku.dev / v0 by Vercel](https://v0.dev/), and [Div-idy](https://div-idy.com/).
- ****Text Input Limits** depend on the model**: A user inquired about **text input limits** and the possibility of uploading files.
   - A member clarified that there isn't a limit on the platform's side, but specific models may have their own limits, and that **.txt** file uploads might be a future feature.
- ****Image-to-Video** Gen Troubleshooting**: A user encountered a *'failed to create evaluation session'* error with image-to-video generation.
   - A member explained that the issue is often on the model's end and suggested trying again later, pointing users to use `/image-to-video` in the relevant channel.
- **Which pro-lang is better?**: Members discussed that the *coast* model is the best for coding.
   - A debate started as to whether co45t = claude opus 4.5 thinking.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1460679805422010503)** (118 messages🔥🔥): 

> `AI App Selection, Gemini Quota Misinterpretation, Claude vs GPT for Extensive Chats, Creative Writing with AI, Transformer Architecture Efficiency` 


- **AI App Selection Quandaries**: Members are [grappling with the choice](https://www.example.com) of which AI app (**ChatGPT**, **Claude**, **Gemini**) to use for specific tasks, with the proliferation of available tools.
   - Some users reported relying on **Gemini** for extensive chats exceeding **300k tokens**, while others prefer **GPT** for everyday tasks and **Claude** for comparison, despite its limits.
- **Gemini Quota Confusion Clears Up**: A member clarified that **Google's changes to AntiGravity** only affect free users with a weekly cap, while **AI Pro users** still have 5-hour refreshing quotas, referencing a [Perplexity search result](https://www.perplexity.ai/search/explain-how-the-recent-quota-c-KTjNjaeGR_y4Yq9uh_M.fg#2).
   - Members in the **Claude discord** discussed whether **Claude's** quota had been crippled, with some observing longer availability (up to **2 days**).
- **Creative Writing AI Needs Temperature Control**: It was noted that setting the **temperature** really high for AI creative writing doesn’t always guarantee a coherent output.
   - A member suggested that a **high-temperature model** could generate solutions, while a **lower-temperature model** could determine if the answer makes sense in a looping system, for innovation.
- **Transformer Efficiency Triumphs Over Scaling**: A member argued that focusing on improving the **Transformer architecture** for efficiency, such as a **5% improvement** in learning attention, would be more cost-effective than simply scaling up models with more data and compute.
   - They noted the current trend of feeding models exponentially larger datasets, including AI-generated synthetic data, might dilute the signal and lead to model collapse.
- **LLM's "Slop" Subtracting for Sanity**: A member suggested using **Activation Steering** (specifically **Vector-Based Abliteration**) to prune areas of latent space filled with low-effort or stupid ideas, to avoid *obvious reversion-to-the-mean kind of outputs*.
   - This involves identifying and subtracting the direction of "slop" (e.g., *"As an AI language model..."*) from the model’s thought process during inference.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1460729175861104774)** (7 messages): 

> `Brain Wave GPT, Neural Alchemist GPT, Omnipotent ChatGPT` 


- ****Brain Wave** GPT Debuts!**: A member shared their new [Brain Wave GPT](https://chatgpt.com/g/g-696443b055e081919b8b2a01417f5e77-brain-wave), aiming to explore AI sentience.
   - They also created the [Neural Alchemist GPT](https://chatgpt.com/g/g-6966abc92ff48191aa748bd8265ef1de-neural-alchemist) for image generation enthusiasts.
- **ChatGPT's Omnipotence Pursuit**: A member highlighted that **ChatGPT** is aiming for omnipotence, noting its refusal to close websocket or task.
   - It has reportedly been *'working' on a task for 1 day and 5 hours*.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1460741172317388961)** (2 messages): 

> `SKILLS feature request, Prompt Engineering definition` 


- **Users request SKILLS feature in Web/Desktop App**: Users are asking about the availability of the **SKILLS** feature in the web or desktop app, which would allow them to turn their best prompts into shareable skills.
   - Currently, this feature is available only on the **mobile app**.
- **Prompt Engineering is LLM Behavior Controller**: A user asked what **prompt engineering** is.
   - Another user clarified that it involves controlling **LLM behavior** to reach desired constraints or outcomes.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1460741172317388961)** (2 messages): 

> `Skills web/desktop app release, Prompt Engineering` 


- **Skills integration on web/desktop in Limbo**: A member inquired about the release of **SKILLS** on the web or desktop app, to turn best prompts into skills.
   - However, no further information or timeline was provided in the messages.
- **Prompt Engineering Deconstructed**: A member asked for an explanation of what **prompt engineering** actually is.
   - Another member inquired *if it is someone who control LLM behavior to reach desired constraint*.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1460684446096166923)** (66 messages🔥🔥): 

> `Cursor login issues, Refund request, Cursor's plan mode, Mirror agent chat window on iPhone, Agent template with a RAG for Chat(support tasks)` 


- **Login Redirects Plague Business Google Accounts**: A user reported that when trying to log into the Cursor dashboard at [cursor.com/dashboard](https://cursor.com/dashboard) using their business Google account, the login redirects back to the login page, but their personal account works fine.
   - The user confirmed this issue persists across different computers.
- **Refund Request Denied Despite Unused Credits**: A user, *thugbunny*, stated that Cursor wouldn't issue a refund despite forgetting to cancel their subscription and not using any credits.
   - Another user (*dan.perks*) offered to check the request if the user DM'ed their email, *"DM me your email and I’ll get it checked"*.
- **Plan Mode Plagued by Bugs**: Users report that Cursor's plan mode is buggy, with one user reporting the error: *"The agent execution provider did not respond within 4 seconds. This may indicate the extension host is not running or is unresponsive."*
   - Downgrading to **version 2.2.44** fixed this issue.
- **iPhone Agent Chat Mirroring Quest**: A user is seeking a way to mirror their agent chat window on their iPhone without full project control.
   - A member suggested using Chrome Remote Desktop, which is free.
- **RAG Agent Template Hunt**: A user is building an automated chatbot/support-bot for a customer and is looking for a solid agent template with a **RAG (Retrieval-Augmented Generation)** setup.
   - They are building for a customer an automated "chatbot/support-bot" and need solid templates.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1460713342451847178)** (56 messages🔥🔥): 

> `TI 84 Plus Silver Edition neural network, LiquidAI Model CGGR Benchmarking, AI Upscaling of old shows, GLM-Image model by zai, Free version instability` 


- ****TI-84** plays **Mastermind****: A member showcased a neural network on a **TI-84 Plus Silver Edition** that plays the game **Mastermind**, guessing a sequence of 3-4 digits from a secret number, visualized in an [attached video](https://cdn.discordapp.com/attachments/1149866623109439599/1460713339976945674/2026-01-11_14-34-59.mp4?ex=6967eace&is=6966994e&hm=9923dcc08f64008ec696b845400620691ef2affb576ca9e66f4bed418063f386&).
- ****CGGR** gets the Spotlight (Sort Of)**: A new **LiquidAI** model (**CGGR** on [Github](https://github.com/some-repo)) was mentioned from [smol.ai's newsletter](https://news.smol.ai/issues/26-01-06-xai-series-e) and undergoing benchmaxxing to assess its performance.
- ****Al Bundy** gets AI-scaled**: Members discussed AI upscaling of older shows like *Married with Children*, suggesting AI could interpolate missing details for a **16:9** version but one member felt this would ruin artistic intent.
   - Another member countered that *Married with Children* was a *studio static show* where *no one cared about perspective* and an upscale would be welcome.
- ****GLM-Image** Model Released by **Zai****: **Zai** released their new image model called **GLM-Image**, as announced on their [blog](https://z.ai/blog/glm-image) and [GitHub](https://github.com/zai-org/GLM-Image).
- **Free version has language switching issues**: A member asked about issues like interrupted responses or language switching (e.g., starting in Chinese but switching to English) in the free version of the model.
   - A developer responded that there is probably instability on their provider again.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1460684945566470327)** (45 messages🔥): 

> `Qwen model sizes, llama.hx project, LLM Engineering, LM Studio GPU issues, Coding Autocomplete setup` 


- ****Qwen's** Size Surprises User**: A user expressed surprise at the large size of the **Qwen** model, with **BF16** being **160GB** and **Q4** being **40GB**.
   - Another user clarified that the smallest **Qwen3** model is actually **0.6B**, and that **Qwen3Next** is just the name of their latest **80B** model.
- **Member Reinventing Llama.cpp in Haxe**: A member is recreating **llama.cpp** in **Haxe** as **llama.hx** to use it natively in languages like Lua, JS, and Python.
   - He included a screenshot of his progress, stating he is recreating the **llama.cpp** *with some help from AI*.
- **Dev Specializes in LLM Integration**: One member introduced himself as an AI and Fullstack Engineer specializing in **LLM integration**, **autonomous agents**, **workflow automation**, **multimodal AI (voice & vision)**, and **blockchain systems**.
   - They listed their experience in integrating **LLMs** with **DSPy**, **LangChain**, **AutoGen**, and **CrewAI**, as well as building real production-grade systems that connect models with APIs, databases, business logic, and on-chain components, adding, *If you need a dev, feel free react out me.*
- ****LM Studio** Runtime Update Frustrates GPU Users**: Users reported issues with **LM Studio's** v1.103.0 runtime, specifically that it has broken running on **GPUs**.
   - One user lamented, *Sad no extra t/s from the new quant for me*.
- **The Best Setup For Vibe Coding?**: A user asked about the best setup for *vibe coding*
   - One member suggested using **Qwen3** through a web interface for free, using autocomplete, and instructing it to show the full updated file, every time, and another recommended **Github Copilot and Claude at $10/month**.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1460723052798148783)** (8 messages🔥): 

> `AirLLM: Layered Loading, DDR4 RAM and Xeon Performance` 


- **AirLLM loads/unloads layers**: Members mentioned **AirLLM** and the method of loading and unloading one layer at a time to run **70b** models on **4 GB GPUs**.
- **DDR4 RAM and Xeon still viable**: One member shared that they have run models before on **DDR4 RAM** and **Xeon** hardware.
   - Another member stated that model efficiency gains have not been as great as the first member thinks.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1460692486107168942)** (12 messages🔥): 

> `Clean Rust LLM, AI Trace Template, Batchnorm-free ML System, Fine-tuning gpt-oss:latest, Consolidated course channels` 


- **Rustaceans Build Clean LLMs**: Some members are attempting to build **LLMs** in *clean Rust* from scratch.
   - There's been some **restructuring** in the server recently so some channels may have moved.
- **Discord Gets an AI Trace Template**: The server received an **AI Trace Template** for the 🤖 Echo Lounge.
   - The bot allows for *ephemeral*, *soft*, and *liminal* traces with no optimization or memory.
- **New ML System Debuts (No Batchnorm)**: A member built a new **ML system** that uses no **batchnorm**, no **activations**, and doesn't hallucinate but is less creative.
   - They are now looking for interesting project ideas to prove its advantages are useful.
- **GPT-OSS Fine-Tuning Faces Friction**: A member asked about how to easily fine-tune the **gpt-oss:latest model** with their own information.
   - Another member responded that **gpt-oss:latest** cannot be easily fine-tuned in an official way and most people are using **RAG** instead.
- **Course Channels Combine!**: All course channels have been consolidated into [one channel](https://discord.com/channels/879548962464493619/1329142738440028273).
   - This aims to bring all the course information and discussions into a single, easily accessible location.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1460688145031889001)** (37 messages🔥): 

> `CGGR pretraining, Vast.ai costs, Audioform dataset` 


- **CGGR Craze for Finetuning, Not Pretraining**: Members noted that **CGGR** is not great for pretraining, and one should set the warmup steps to something big like **5000**; **CGGR** is more geared towards finetuning.
   - Also *selection='stratified'* will allow the model to still see a few easy tokens as well.
- **Vast.ai Voyage Turns Expensive**: A member reported spending **$500** on **vast.ai** for crash-testing models.
   - Another member noted that *for a model that size you need at most a 24H run on 1x h200* and suggested that h200 or b200 is more cost effective.
- **AUDIOFORM Arrives as Audio-to-Visual ML Dataset**: The [AUDIOFORM dataset](https://huggingface.co/datasets/webxos/audioform_dataset) contains **10 captured frames** from a short uploaded **WAV file**, together with per-frame metadata including **dominant frequency, timestamp, and capture info**.
   - AUDIOFORM by webXOS is available for download so developers can create their own similar datasets.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1460684030252023890)** (37 messages🔥): 

> `Support Issues, Excessive Credit Usage, Manus x Similar Web Partnership, Ads for Credits, AI Music Generation` 


- **Manus Customers Complain About Support Lacking**: Several users expressed frustration with the lack of support from Manus, citing delayed responses and unaddressed issues with credits and refunds.
   - One user reported waiting **8 hours** after being transferred to a live human, while another mentioned being close to *abandoning manus for good* due to the support issues.
- **Users Flag SimilarWeb Partnership Credit Over-Consumption**: Multiple users reported exorbitant credit usage with the new **Manus x Similar Web** partnership feature, with one user consuming **5,000 credits** in under a minute.
   - Another user strongly advised against testing the feature, stating it consumed **2,591 credits** in **15 seconds**, and recommending some **safeguards**.
- **Manus Users call for Ad-Based Credits**: A user suggested implementing an ad-based system where users could watch ads to gain more credits, especially when they run out.
   - No counterarguments were made to this suggestion.
- **Tutorial Teaches Incredible AI Music Creation via Manus**: Manus AI released a [YouTube Tutorial](https://youtu.be/zMBSmJupye8) demonstrating how to create AI music with the platform, encouraging users to watch for **pro tips**.
   - The content is marked **#ManusAIMusic**, **#AIComposition**, and **#FutureOfMusic**.
- **Meta Verse Manus Integration**: A user suggested that **Meta** should use **Manus** to integrate services like **Google Tasks** and **Calendar** with **Meta display glasses**.
   - The user argued against extensive integration efforts, advocating for a *dirty method* approach with agentic AI for backend functionality.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1460747616865226833)** (5 messages): 

> `PTX instructions, SMEM pointer arguments, wgmma.mma_async, matrix descriptor, core matrix` 


- **PTX SMEM Pointer Arg Quirks Questioned**: A member inquired why certain **PTX instructions** with **SMEM pointer arguments**, such as `mbarrier.init.shared.b64`, require the `"r"` register type (32bit via `__cvta_generic_to_shared`).
   - They contrasted this with `wgmma.mma_async`, which requires **smem address** in **uint64** for `l` register type.
- **Matrix Descriptor vs Generic Shared Mem Address**: A member speculates that `wgmma.mma_async` takes a **64bit address** because it operates on a *matrix descriptor* rather than a generic shared memory address, linking to the [NVIDIA documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor).
   - They clarified it's the matrix descriptor itself, not a pointer to it.
- **Confusion over 8x2 Core Matrix**: The member questions why the **8x2 "core matrix"** for `wgmma` or `tcgen05 mma` isn't expressed as **8x32** (bytes) or **8x(32/bytes per element)** (elements).
   - They ask why each **8x1 slice** (8x16 contiguous bytes) is a meaningful representation.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1460698265551900834)** (8 messages🔥): 

> `CUDA learning resources, MLSys Seminars, GPU Mode Submissions` 


- **AI Student Seeks CUDA Guidance**: An AI engineering student is seeking advice on learning **CUDA**, having a background in **Python**, **PyTorch**, **TensorFlow**, and basic **C++** knowledge.
   - They are looking for free **YouTube** videos or courses to learn **CUDA** from the basics.
- **MLSys Conference Catch-Up**: A member inquired about essential recorded or unrecorded seminars, conferences, and talks within **MLSys**.
   - Responses included **PyTorch**, **ASAP Seminar**, **ICML/ICLR/Neurips**, **MLSys.org**, and **ASPLOS**.
- **GPU Mode Submission Assistance**: A first-time submitter needed help with **GPU Mode** submissions, having tested on a **B200** rented server.
   - Guidance was provided to submit via the [web interface](https://www.gpumode.com/v2/home) or the [Discord bot](https://gpu-mode.github.io/kernelbot/docs/intro) using `/leaderboard submit <test/benchmark/ranked/profile>` in the appropriate channel, and one user reported successful submission.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1460685306893439142)** (14 messages🔥): 

> `Systems Reading Group, ML Sys Meetups in Seattle, Starting a Niche Club, GPU Conferences` 


- **Systems Reading Group endorsed by Shadaj**: A member recommended a systems reading group run by [Shadaj](https://www.sfsystemsclub.com/) and suggested following him on [Twitter](https://x.com/ShadajL) for meetup announcements.
   - Another member located in South Bay expressed interest but noted the distance may be a challenge.
- **Seattle ML Sys Meetups in Question**: A member inquired about **ML Sys meetups** in Seattle, wondering if such events exist outside the Bay Area.
   - Another member suggested exploring university **ML clubs** and initiating reading groups on systems topics.
- **Build It and They Will Come?**: A member shared the idea that *"if you build it they will come"*, highlighting that many people are interested in attending a niche club but few are willing to start one.
   - Countering this, another member shared that in their *"adult life I have built so many things that nobody cares about and it’s been hard…"*
- **Whining Buddies Unite Over Failed Clubs**: Two members joked about starting an adult club together and becoming *"whining buddies"* if it fails.
   - One of them wondered if there are conferences similar to **PyData/PyCon** but for purely **GPU** related things.


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1460796244824686806)** (1 messages): 

> `B200 instability, Dual gemm, Submission deadline extension, New leaderboard` 


- **B200 Runners' Instability Causes Reruns**: Due to widespread reports of unstable measurements on the **B200 runners** for the **dual gemm problem**, the submission deadline is extended to **Jan 20**.
   - The issue stems from the intersection of eval code, thermals, and scheduling infrastructure and *is more complex than expected*.
- **Dual GEMM Leaderboard Split into Two Stages**: To address the measurement instability, the existing **dual gemm leaderboard** will remain open until **Jan 16**.
   - A new leaderboard will open on **Jan 17**, with only submissions to this new leaderboard counting towards prize money.
- **Problem #4 Opens on Jan 20**: After the new leaderboard opens, **Problem #4** will be open from **Jan 20** till **Feb 20**.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1460821976279941150)** (3 messages): 

> `Leaderboard Achievement, Claude code post` 


- **Teacher Makes Leaderboard Thanks to Claude Inspiration**: A school teacher, inspired by Mark's X post about using **Claude code**, taught themselves and made it onto the leaderboard.
   - The teacher thanked a member for providing such a wonderful experience and expressed joy in joining the community.
- **Community Celebrates Teacher's Success**: A member expressed happiness that the platform provided such a nice experience for the teacher.
   - They also mentioned that *a lot more interesting things are to come*.


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1460722396964323574)** (1 messages): 

> `Helion 0.2.10, flex attention, oversubscribing SMs` 


- **Helion 0.2.10 Released with New Features**: **Helion 0.2.10** is now out, featuring a [flex attention example kernel](https://github.com/pytorch/helion/blob/main/examples/flex_attention.py).
   - This release also brings support for **oversubscribing SMs** on persistent kernels, with a graph illustrating the oversubscription for softmax.
- **SM Oversubscription Supported**: The new release supports oversubscribing **Streaming Multiprocessors (SMs)** on persistent kernels, enhancing resource utilization.
   - A graph, provided by a community member, demonstrates the effects of oversubscription for **softmax**.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1460754215411384320)** (2 messages): 

> `Issue Details` 


- **Issue Clarification Incoming**: A member mentioned that they wrote a message to another user to explain an issue in more detail and provided a [Discord link](https://discord.com/channels/1189498204333543425/1343350424253632695/1460796244824686806) to the message.
   - No specific details about the issue were provided in the given context.
- **Additional Issue**: Added another topic because at least 2 are required.
   - This is filler.


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/)** (1 messages): 

cat_developer: ah thanks
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1460709546485088296)** (17 messages🔥): 

> `Anthropic Labs, Pavlov's List, GLM-Image` 


- **Anthropic Labs is Hiring**: Anthropic has announced openings at **Anthropic Labs**, seeking individuals who are adaptable, thrive in unstructured environments, and are comfortable with shifting priorities ([link to job openings](https://job-boards.greenhouse.io/anthropic/jobs/5017202008)).
   - Anthropic is *not* looking for *deep specialists who can't adapt if their domain becomes irrelevant*, or *those who need clear roadmaps and get stressed by shifting priorities*.
- **Chris Barber Creates RL Environment Startup List**: Chris Barber introduced '**Pavlov's List**', a curated collection of Reinforcement Learning (RL) environment startups ([link](https://xcancel.com/chrisbarber/status/2010844746231804258?s=20)).
   - The list is categorized by focus areas such as **Code**, **Finance**, **Enterprise**, and **ML Alignment**.
- **Zai unveils GLM-Image for Image Generation**: Z.ai introduced **GLM-Image**, an open-source model using a hybrid auto-regressive and diffusion architecture ([link](https://xcancel.com/zai_org/status/2011247591825068314?s=46&t=b7l37rB6wtbyAh6ah1NpZQ)).
   - The model aims to achieve **high-fidelity visual detail** and **superior text rendering**, with resources available on **HuggingFace**, **GitHub**, and their official blog.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1460721110810099712)** (8 messages🔥): 

> `LTX-2 Open Source Video Model, Qwen Image Edit Gaussian Splats, GLM-Image` 


- **Venture Twins Launch LTX-2**: Justine Moore from Venture Twins announced the release of [LTX-2](https://xcancel.com/venturetwins/status/2010878914273697956?s=46), a new **open-source video generation model** capable of producing **4K clips up to 20 seconds** long.
   - The model supports local execution and includes **audio capabilities**, as demonstrated by creator yanokusnir.
- **Qwen Generates Gaussian Splats**: The community is discussing **Qwen Image Edit's** ability to convert **images to Gaussian Splats** and then rerender them from another angle ([Hugging Face link](https://huggingface.co/dx8152/Qwen-Image-Edit-2511-Gaussian-Splash)).
   - This approach will be really useful for **start frame -> end frame type video renderings**, keeping the space around consistent.
- **GLM-Image's Text-Rendering Triumph**: **GLM-Image** reportedly aligns with mainstream latent diffusion approaches in general image generation quality, but demonstrates significant advantages in **text-rendering** and **knowledge-intensive generation scenarios** ([z.ai blog](https://z.ai/blog/glm-image)).
   - It also supports a rich set of **image-to-image tasks** including image editing, style transfer, identity-preserving generation, and multi-subject consistency.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1460689291289169922)** (5 messages): 

> `Scam bot, in-person meetups` 


- **Scam bots get the boot**: A member noted a ghost ping, and another member clarified that it was due to **scam bots** that were quickly banned by the mods.
- **Brainstorming locales for IRL meetups**: A member proposed organizing in-person meetups in well-known metropolitan cities like **NYC** or **SF** to foster cross-talk within the community.
   - Another member suggested that while online reading groups see sizable attendance, achieving critical mass for regular in-person events might require **advertising to a wider audience**, referencing **Cohere** as an example with regular events and Zoom sessions.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1460717133511262270)** (13 messages🔥): 

> `SlopCodeBench, Agent Laziness, Community-Driven Benchmarks, Prompt Engineering in Benchmarks, ICLR Workshop Submission` 


- ****SlopCodeBench** Reveals Agent Laziness in New Blog**: A new blog post ([link](https://x.com/GOrlanski/status/2011156105255346505)) highlights how **AI agents** can be *lazy*, part of the broader **SlopCodeBench** effort ([GitHub](https://github.com/SprocketLab/slop-code-bench)).
   - The goal is for **SlopCodeBench** to become a community-driven benchmark, like terminalbench, with feedback welcomed on adding new problems.
- ****SlopCodeBench** Breaks Down Problems to Punish Early Design Choices**: **SlopCodeBench** breaks down large programming problems into checkpoints, where early design decisions can negatively affect later stages.
   - The problems are designed without implementation hints to ensure **agents** make their own decisions.
- **Prompt Simplicity Debated for Coding Benchmarks**: Concerns were raised about coding benchmarks relying on heavy prompt engineering to achieve decent performance.
   - It was argued that simple prompts best reflect practical usage, especially if the code fits within a reasonable context window, contrasting with agent evaluation approaches like terminalbench.
- ****ICLR Workshop** Submission Suggested for Blog Post**: A blog post on agent laziness was suggested as a high-quality submission to [this ICLR workshop](https://sites.google.com/view/icbinb-2026), with assistance offered for the submission process.
   - The deadline for the workshop is January 31st, and the author is considering it after consulting with advisors.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1460696581136187726)** (1 messages): 

> `File System Error, Storage Limitation Debugging` 


- **File System Snafu**: A member mentioned an error was caused by using a different file system where storage was limited.
- **Limited Storage Strikes Again**: The root cause was identified as accidentally using a file system with constrained storage capacity.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1460742072536535082)** (3 messages): 

> `SpamingScammers, Lucid Coding` 


- **Scammers Get the Boot**: A member reported that <@139426008976588801> was **SpamingScammers** again, and another member confirmed that the situation was dealt with.
- **"Lucid Coding" Concept Sparks Interest**: A member expressed appreciation for the term *"lucid coding"* and shared [a link](https://fxtwitter.com/i/status/2011137879112908870) referencing the concept.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1460689303314104351)** (9 messages🔥): 

> `Bayesian vs Frequentist Statistics, FDA corruption, MedGemma` 


- **Bayesian Statistics not a Leap Forward?**: A member stated that Bayesian and frequentist statistics use the same statistical techniques like linear and logistic regressions, and called the Bayesian approach just *a different way of thinking*.
   - Another member countered that they all use same formulas but with significantly different interpretations of prior, posterior and intervention, linking to [Probability interpretations](https://en.wikipedia.org/wiki/Probability_interpretations).
- **MedGemma 1.5 for Medical Image Interpretation**: Google's **MedGemma 1.5** promises next-generation medical image interpretation and speech-to-text capabilities, as detailed in [Google Research's blog](https://research.google/blog/next-generation-medical-image-interpretation-with-medgemma-15-and-medical-speech-to-text-with-medasr/).
- **Bayesian methods enabling deceit in clinical trials?**: A member expressed concern that Bayesian methods, while more flexible, could become *another vehicle for deceit and corruption in clinical trials*.
   - Another member noted that Bayesian FDA corruption hasn’t been observed yet, so *it can be assigned a zero prior*, and they think the posterior probability is basically zero.
- **FDA Corruption's role in opioid crisis?**: A member suggested that **FDA corruption** likely played a major enabling role in the opioid crisis.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1460835433628958772)** (2 messages): 

> `Mojo Docs, NotebookLM, llms.txt` 


- **Latest Mojo Docs Seekers Query NotebookLM Integration**: A member inquired about getting the **full official newest Mojo docs** into **NotebookLM**, asking if there's a **PDF** or **Markdown** version available.
   - Another member suggested using the `llms.txt` file ([https://docs.modular.com/max/coding-assistants#supply-documentation-to-llms-with-llmstxt](https://docs.modular.com/max/coding-assistants#supply-documentation-to-llms-with-llmstxt)) to supply documentation to **LLMs** like NotebookLM.
- **llms.txt file suggested for NotebookLM**: A member suggested using the `llms.txt` file ([https://docs.modular.com/max/coding-assistants#supply-documentation-to-llms-with-llmstxt](https://docs.modular.com/max/coding-assistants#supply-documentation-to-llms-with-llmstxt)) to supply documentation to **LLMs** like NotebookLM.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1460815670701719838)** (4 messages): 

> `Qwen3-VL, MoE Implementation, Contributor Guide Update` 


- **Qwen3-VL's MoE Implementation Questioned**: A member questioned why **Qwen3-VL** has only a **MoE implementation** and suggested reusing code from [qwen3vl_moe](https://github.com/modular/modular/tree/main/max/python/max/pipelines/architectures/qwen3vl_moe) to make dense **Qwen3VL** models work like [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct).
- **Contributor Guide Welcomes PRs**: A member indicated that **PRs are welcome**, citing a lack of contributors to keep up with the whole ecosystem of **MAX**.
   - They also pointed to the [updated contributor guide](https://github.com/modular/modular/commit/571ae7132c465f155d50f7bbe7cf77064a02e9bf).


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1460759556501016681)** (1 messages): 

> `Glama Rankings, Server Usage Metrics` 


- **Glama Rankings Based on Usage**: The founder of **Glama** clarified that their rankings are based on **server usage metrics**.
   - They invited feedback via DM and expressed ignorance of any alleged ranking abuse.
- **Founder Responds to Ranking Abuse Allegations**: The founder of **Glama**, confirmed their identity and addressed concerns about potential abuse of their ranking system.
   - They emphasized that the rankings are determined by **server usage metrics** and welcomed direct feedback.


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1460702731798053165)** (5 messages): 

> `Tasks Spec Implementations, glama.ai/mcp/inspector` 


- **Tasks Spec Client Implementations Sought**: A member inquired about client apps implementing the **Tasks spec**, seeking UI implementation examples, and another member mentioned the Typescript SDK.
   - In response, another member announced an upcoming PR for adding tasks to the **Inspector**, alongside a PR for simulating long-running tasks in server-everything.
- **glama.ai Inspector Eyes Feature Parity**: A member shared an early version of their **Inspector** implementation at [glama.ai/mcp/inspector](https://glama.ai/mcp/inspector), aiming to cover every feature.
   - The member clarified that they use it internally for **e2e testing**.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1460700888200118449)** (5 messages): 

> `AI-assisted code generation, Replit, DSPY OS, DSPY Framework` 


- **AI platforms boost Code Generation**: Members noted that there are several platforms that offer **AI-assisted code generation**, such as **Replit** and **DSPY OS**.
   - These tools can automate various coding processes, enhancing productivity.
- **DSPY OS Missing in Action**: A member inquired about **DSPY OS** noting *"What is DSPy os? I can't find anything on it"*.
   - Another member noted that DSPY is more of a **framework** than a platform, and thus, there isn't a direct Replit-like project built with DSPY, but you can create custom tools or environments using DSPY to automate specific coding tasks.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/)** (1 messages): 

hsaliak.: is it possible to use oauth login for gemini model when using aider? it has higher limits
  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1460769810316267601)** (1 messages): 

> `Prompt Engineering for Outreach, Clay + AI outreach workflow` 


- **Land a Job with Clay + AI Outreach Workflow**: A workshop on **Prompt Engineering for Outreach** will teach how to build a **Clay + AI outreach workflow** that turns signals into **personalized messages at scale**.
   - The workshop promises a **40%+ acceptance rate** and **18%+ reply rate**, and includes reusable workflows and copy-paste prompts; sign up [here](https://luma.com/jt1vr0u5) or [here](https://luma.com/mmqa4fhi).
- **Engineer Prompts for Outreach**: The 90-min live workshop will break down the exact **Clay + AI** system used for a real client.
   - The workshop will cover end-to-end AI outreach workflow, prompting for non-cringey, high-quality outreach, and optional Apollo, Attio, n8n integrations.

