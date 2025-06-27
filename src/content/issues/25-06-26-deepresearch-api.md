---
id: MjAyNS0w
title: OpenAI releases Deep Research API (o3/o4-mini)
date: '2025-06-26T05:44:39.731046Z'
description: >-
  **OpenAI** has launched the **Deep Research API** featuring powerful models
  **o3-deep-research** and **o4-mini-deep-research** with native support for
  MCP, Search, and Code Interpreter, enabling advanced agent capabilities
  including multi-agent setups. **Google** released **Gemma 3n**, a multimodal
  model optimized for edge devices with only 3GB RAM, achieving a top score of
  1300 on LMSys Arena, featuring the new MatFormer architecture and broad
  ecosystem integration. **Black Forest Labs** introduced **FLUX.1 Kontext
  [dev]**, a 12B parameter rectified flow transformer for instruction-based
  image editing, comparable to **GPT-4o**. **DeepMind** unveiled
  **AlphaGenome**, an AI model capable of reading 1 million DNA bases for gene
  function prediction, marking a breakthrough in AI biology. **Sakana AI**
  presented Reinforcement-Learned Teachers (RLTs) to enhance LLM reasoning,
  achieving 86.1% on MiniF2F with efficient compute. **Higgsfield AI** released
  **Higgsfield Soul**, a high-aesthetic photo model with 50+ presets for
  fashion-grade realism. Additionally, **Google** launched the **Gemini CLI**,
  an open-source AI agent for terminal use with free Gemini 2.5 Pro requests.
companies:
  - openai
  - google
  - black-forest-labs
  - deepmind
  - sakana-ai
  - higgsfield-ai
  - huggingface
  - ollama
models:
  - o3-deep-research
  - o4-mini-deep-research
  - gemma-3n
  - flux-1-kontext-dev
  - gpt-4o
  - alphagenome
topics:
  - multimodality
  - model-releases
  - agentic-ai
  - reinforcement-learning
  - instruction-following
  - model-architecture
  - model-optimization
  - image-generation
  - biological-ai
  - multi-agent-systems
  - model-integration
people:
  - demishassabis
  - hardmaru
  - osanseviero
  - clementdelangue
---


**Deep Research is all you need.**

> AI News for 6/25/2025-6/26/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (220 channels, and 5509 messages) for you. Estimated reading time saved (at 200wpm): 472 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

While Google had [announced](https://www.latent.space/p/aiewf-2025-keynotes) their *intentions* to release a Deep Research API, it seems OpenAI has chosen today to scoop them by *actually* releasing their Deep Research API in a relatively lowkey [announcement](https://x.com/openaidevs/status/1938286704856863162?s=46&t=jDrfS5vZD4MFwckU5E8f5Q):

[](https://resend-attachments.s3.amazonaws.com/gUxdgTpl2JbsOZJ)

We will not mince words - **o3-deep-research and o4-mini-deep-research are probably the most powerful LLMs for powering agents in the world right now.** This is thanks to the native support for MCP, Search and Code Interpreter which are 3 of the [Big 5](https://news.smol.ai/issues/25-05-27-mistral-agents) LLM OS primitives.

Apart from the new webhook modality, you should not miss the cookbooks released today:

- [Introduction to Deep Research API:](https://cookbook.openai.com/examples/deep_research_api/introduction_to_deep_research_api) giving you everything you need to build your own Deep Research in ~30LOC, with MCP
- [Deep Research API Agents:](https://cookbook.openai.com/examples/deep_research_api/introduction_to_deep_research_api_agents) detailing usage with the Agents SDK and, for the first time, a **multi-agent** setup using 4 agents
    
    [](https://resend-attachments.s3.amazonaws.com/SrxbBsr13hNncpx)
    

---

# AI Twitter Recap

**Model Releases & Updates**

- **Google Releases Gemma 3n**: **Google** has released **Gemma 3n**, described as a powerful multimodal (**text, audio, image, video**) AI model designed to run on edge devices with as little as **3GB of RAM** and achieve high performance on-device. According to [@osanseviero](https://twitter.com/osanseviero/status/1938374626910060782), it's the first <10B model to score over **1300** on the **LMSys Arena**. The model features a new **MatFormer** architecture, making it natively flexible. The release includes extensive open ecosystem support, with day-zero integration from partners like [@huggingface](https://twitter.com/ClementDelangue/status/1938283910980325670), [@ollama](https://twitter.com/ollama/status/1938324186579292415), [@awnihannun for MLX](https://twitter.com/awnihannun/status/1938283694416077116), [@UnslothAI](https://twitter.com/osanseviero/status/1938307534840074522), and [@ggerganov for llama.cpp/GGUFs](https://twitter.com/ggerganov/status/1938284171564028214). Ross Wightman also released an update to `timm` to provide the image encoder for Gemma 3n, noting its 'MobileNetV5' backbone ([@wightmanr](https://twitter.com/wightmanr/status/1938311403934519807)).
- **Black Forest Labs Releases FLUX.1 Kontext [dev]**: **Black Forest Labs** has released **FLUX.1 Kontext [dev]**, a **12B parameter** open-weights rectified flow transformer for high-quality, instruction-based image editing, positioned as comparable to proprietary models like **GPT-4o**. The model is now available on **Hugging Face** ([@ClementDelangue](https://twitter.com/ClementDelangue/status/1938260818602430788)) and has day-zero support in `diffusers` ([@RisingSayak](https://twitter.com/RisingSayak/status/1938267936378208655)) and Chipmunk ([@realDanFu](https://twitter.com/realDanFu/status/1938300379613347942)).
- **DeepMind Releases AlphaGenome**: [@demishassabis](https://twitter.com/demishassabis/status/1937971182256435323) highlighted the release of **AlphaGenome**, an AI model that can read **1 million bases of DNA** to predict gene function and regulation. This is seen as a major step forward in AI for biology.
- **Sakana AI Introduces Reinforcement-Learned Teachers (RLTs)**: [@hardmaru](https://twitter.com/hardmaru/status/1938381728902783321) shared **Sakana AI's** new **RLT** technique, which uses reinforcement learning to teach LLMs complex reasoning. A notable result includes achieving **86.1% on MiniF2F** with three 7-8B models using a small sample budget, setting a new compute Pareto frontier ([@teortaxesTex](https://twitter.com/teortaxesTex/status/1938066184286433621)).
- **Higgsfield AI Launches Higgsfield Soul**: A new high-aesthetic photo model named **Higgsfield Soul** has been released, featuring over **50 curated presets** and promising fashion-grade realism to enhance user-generated content ([@TomLikesRobots](https://twitter.com/TomLikesRobots/status/1937937784934912415)).

**Tooling, Frameworks, and Infrastructure**

- **Google Launches Gemini CLI**: **Google** has released the **Gemini CLI**, an open-source AI agent that brings **Gemini** models directly to the terminal. The tool provides **1,000 free Gemini 2.5 Pro requests per day** ([@googleaidevs](https://twitter.com/demishassabis/status/1938023045320335789), [@OfficialLoganK](https://twitter.com/hardmaru/status/1938068439404581370)). Community members have already integrated it as a provider in tools like **Codex CLI** ([@cline](https://twitter.com/cline/status/1938052438113845748)), and one user even "vibe-added" **Claude Sonnet and Opus** to it ([@hrishioa](https://twitter.com/hrishioa/status/1938335965845876940)).
- **The Rise of "Context Engineering"**: A significant discussion, amplified by a viral tweet from [@karpathy](https://twitter.com/code_star/status/1937934052436414690), centers on replacing the term "prompt engineering" with **"context engineering"**. The new term better reflects the practice of providing extensive, high-quality context to LLMs, rather than just short task descriptions. Some propose that an AI agent is effectively an **"automatic context engineer"** ([@shaneguML](https://twitter.com/shaneguML/status/1938106399466369412)), while others see it as the evolution of deep learning from feature engineering ([@awnihannun](https://twitter.com/awnihannun/status/1938365325676057014)).
- **DSPy Gains Traction as a "Context Engineering" Tool**: **DSPy** is gaining popularity, notably endorsed by **Shopify CEO Tobi Lütke** as his "[context engineering tool of choice](https://twitter.com/lateinteraction/status/1938392172245750072)". Many in the community are highlighting its practical applications for prompt optimization and building reliable AI systems ([@stanfordnlp](https://twitter.com/stanfordnlp/status/1937944059160768793)).
- **LangChain/LlamaIndex Ecosystem Updates**: **LlamaIndex** is promoting agent development, including a tutorial on building an event-driven **Zoom meeting notetaker** that integrates with **Notion**, leveraging Zoom's new real-time streaming capabilities ([@jerryjliu0](https://twitter.com/jerryjliu0/status/1937998395383423130)). **LangGraph** is being highlighted for its use in building long-running, stateful applications like the **Qodo Gen CLI** for software development automation ([@hwchase17](https://twitter.com/hwchase17/status/1938287016250380655)).
- **KerasHub for Cross-Framework Model Use**: **François Chollet** announced **KerasHub**, which allows developers to use **Hugging Face checkpoints** for models like **Llama, Gemma, and Mistral** across **JAX, PyTorch, and TensorFlow** for inference, LoRA fine-tuning, and large-scale training ([@fchollet](https://twitter.com/fchollet/status/1938208330062655678)).
- **Distributed Training Techniques**: A detailed thread from [@StasBekman](https://twitter.com/StasBekman/status/1938270423978021228) explains **activation memory offloading** to CPU memory as a way to save significant GPU memory during long-sequence training, enabling more layers without increasing memory usage.
- **Modular Partners with Inworld AI for Text-to-Speech**: **Modular** and **Inworld AI** have partnered on a new state-of-the-art text-to-speech model that is reportedly **20x cheaper** and makes real-time speech more accessible for various products ([@clattner_llvm](https://twitter.com/clattner_llvm/status/1937931869640921385)).

**Company & Industry News**

- **Meta Hires Key Researchers from OpenAI's Zurich Office**: A viral tweet from [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1938077153733800075) reported that **Meta** has effectively hired a significant portion of **OpenAI's Zurich team**, with researchers reportedly not waiting for their OpenAI equity to vest. The move is seen as a major talent acquisition for Meta's **Llama** development. [@Teknium1](https://twitter.com/Teknium1/status/1938091439440969752) quipped about the claim that OpenAI isn't losing its "best" people.
- **OpenAI Announces DevDay 2025**: **OpenAI** has scheduled its next developer day for **October 6, 2025**, in San Francisco, promising a larger event with over **1500 developers**, a livestreamed keynote, and hands-on sessions ([@OpenAI](https://twitter.com/OpenAI/status/1938277642014494980)).
- **Suno Acquires AI-Powered DAW WavTool**: **Suno**, known for its AI music generation, has acquired **WavTool**, an AI-powered digital audio workstation. The move aims to give artists more precise control in their creative workflows ([@SunoMusic](https://twitter.com/SunoMusic/status/1938281718865399933)).
- **Anthropic Enables App Creation with Claude**: **Anthropic** now allows users to create and share functional, AI-powered applications with **Claude's** intelligence embedded directly within them, enabling shareable, interactive experiences ([@alexalbert__](https://twitter.com/alexalbert__/status/1937934036590334335)).
- **Nvidia Regains Title of World's Most Valuable Company**: After a period of volatility, **Nvidia's** stock has rebounded, making it the world's most valuable company once again ([@nearcyan](https://twitter.com/nearcyan/status/1938035873259655202)).

**Research, Techniques, and Commentary**

- **Court Rules Training on Copyrighted Books is Fair Use**: **Andrew Ng** provided a detailed analysis of a US District Court ruling that found **training LLMs on copyrighted books constitutes fair use**. The judge ruled that training is transformational and not a substitute for the original works. However, the ruling also indicated that using pirated materials is not fair use, a point that may still create liability for model trainers ([@AndrewYNg](https://twitter.com/AndrewYNg/status/1938265468986659075)).
- **The State of Academic Peer Review**: A widely-circulated thread by [@jxmnop](https://twitter.com/jxmnop/status/1937949143084810625) described a frustrating experience reviewing papers for **NeurIPS**, highlighting issues like **LLM-generated submissions**, duplicate papers, and unreproducible research based on private company data.
- **On RLHF, Regularization, and Slop**: In a highly-circulated tweet, **Andrej Karpathy** warned, "[May your regularizer be strong, lest you RLHF to slop](https://twitter.com/karpathy/status/1937941695943065640)," a concise take on the risks of over-optimizing models with reinforcement learning from human feedback without proper constraints.
- **The "em dash" as an AI Writing Tell**: **John Carmack** noted that he likes using em dashes but dislikes that they are now often taken as a sign of AI-generated text, a sentiment that resonated widely ([@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1938278575800553665)).
- **Stanford's "Language Models from Scratch" Course (CS336)**: The Stanford course **CS336**, taught by **Percy Liang** and others, is receiving high praise from industry leaders like **Jeff Dean** as an excellent resource for learning how to build LMs from the ground up ([@stanfordnlp](https://twitter.com/stanfordnlp/status/1937944419090764222)).

**Broader Implications**

- **AI as a Core System Property**: Perplexity CEO **Aravind Srinivas** argues that for a product to remain relevant as AGI arrives, intelligence must be a core property of the system, not "sprinkled as parts." He views the **browser** as a key platform that satisfies this property ([@AravSrinivas](https://twitter.com/AravSrinivas/status/1938116239576199365)).
- **Geopolitical and National Security Commentary**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1937990538302796147) shared an article discussing the use of RL for cybersecurity, agreeing that defenders will likely be advantaged in a new LLM-powered equilibrium. He also comments frequently on the perceived gap between US policy thinking and the reality of modern Chinese military doctrine ([@teortaxesTex](https://twitter.com/teortaxesTex/status/1938066712928161815)).
- **AI's Impact on the US Power Grid**: A warning from [@dylan522p](https://twitter.com/dylan522p/status/1937943241082437697) suggests that a large-scale AI training run could cause a blackout for hundreds of thousands of people due to the instability of the US grid, potentially turning public sentiment against AI infrastructure.
- **US Nonimmigrant Visa Privacy Policy Update**: In a non-AI but highly impactful announcement for the tech community, the US Consulate in the UK announced that all applicants for **F, M, or J nonimmigrant visas** are now requested to adjust their social media privacy settings to "public" for the duration of the visa process ([@francoisfleuret](https://twitter.com/francoisfleuret/status/1937926540769054772)).

**Humor & Memes**

- **The Best Login Screen**: [@vikhyatk](https://twitter.com/vikhyatk/status/1938092308358172880) jokes, "**the metrics indicate we have the best login screen in the industry... users spend an average of 30 seconds on this page before logging in**".
- **Conference Observations**: [@dylan522p](https://twitter.com/dylan522p/status/1938334440595366035) posts: ">at a conference >someone says LLMs won’t generalise, overhyped >look at his badge >he works at **IBM** Every time".
- **The State of AI Startups**: [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1938278592175345802) perfectly captures the current development mood with a picture of a chaotic desk and the caption "**Nothing works, but the vibes are immaculate**."
- **Compute Debt Ballad**: [@MillionInt](https://twitter.com/MillionInt/status/1938018248915873883) sings the song of the data scientist: "**You spin ten thousand nodes, what do you get? Another night sleepless and in compute debt**".

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Gemma 3n Model Launch and Community Tooling

- [**gemma 3n has been released on huggingface**](https://www.reddit.com/r/LocalLLaMA/comments/1ll429p/gemma_3n_has_been_released_on_huggingface/) ([Score: 243, Comments: 70](https://www.reddit.com/r/LocalLLaMA/comments/1ll429p/gemma_3n_has_been_released_on_huggingface/)): **Google has released the Gemma 3n family of models on Hugging Face with both base (E2B, E4B) and instruction-tuned (-it) versions, accompanied by detailed benchmarks covering datasets like HellaSwag, MMLU, and LiveCodeBench (see: [E2B](https://huggingface.co/google/gemma-3n-E2B), [E4B](https://huggingface.co/google/gemma-3n-E4B)). llama.cpp support landed via [ngxson's PR](https://github.com/ggml-org/llama.cpp/pull/14400), and GGUF quantized models are available for local inference. The [official technical announcement](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/) provides deeper context on model architecture and evaluation.** Some expert discussion queries how Gemma 3n compares technically to Qwen3; this comparative analysis remains open within the thread.
    - A user requests a direct comparison between Gemma 3n and Qwen3, indicating interest in benchmarks, inference speed, and language or task performance. This highlights the community's need for empirical evaluations across popular open-source LLMs.
    - There is anticipation for Gemma 3n's performance on Android, suggesting a focus on mobile optimization, efficiency, quantization, or edge deployment factors relevant to the new release.
- [**Gemma 3n Full Launch - Developers Edition**](https://www.reddit.com/r/LocalLLaMA/comments/1ll68iz/gemma_3n_full_launch_developers_edition/) ([Score: 117, Comments: 4](https://www.reddit.com/r/LocalLLaMA/comments/1ll68iz/gemma_3n_full_launch_developers_edition/)): **Google has fully released Gemma 3n, a multimodal model supporting audio, video, image, and text inputs with text outputs. Key innovations include parameter-efficient variants ("E2B" and "E4B"), the ability to operate with as little as** `2B/4B` **params, the MatFormer architecture allowing submodel extraction and mix-n-match deployment, integration with MobileNetV5 and a new audio encoder, and extensive compatibility across platforms (Hugging Face, llama.cpp, Ollama, MLX, etc.). See the official [developer guide](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/) and [Hugging Face collection](https://huggingface.co/collections/google/gemma-3n-685065323f5984ef315c93f4) for details.** Commentary requests rapid addition of GGUF format multimodal (audio+vision) support and fine-tuning compatibility. There is technical curiosity about using multimodal encoders with larger Gemma models (e.g., 27B) and inquiry about a potential JAX implementation release.
    - A user asks about the timeline for audio and vision modality support in GGUF format, which would enable broader multimodal use in local and efficient environments. This indicates developer interest in extending current text-based implementations to handle other data types, aligning with trends for multimodal LLMs.
    - There is an inquiry regarding the release of a JAX implementation, which would be notable for supporting TPU-based training/inference and easily leveraging high-performance compute on Google Cloud. This would enhance extensibility and facilitate research applications beyond traditional PyTorch pipelines.
    - A technically focused comment asks whether Gemma3 27B can be extended with audio and video-encoded tokens to enable multimodality, suggesting interest in projecting non-text embeddings into the LLM for richer input processing—mirroring existing work in models like LLaVA or Flamingo.
- [**Google's CLI DOES use your prompting data**](https://i.redd.it/j1km6ff1h69f1.png) ([Score: 300, Comments: 88](https://www.reddit.com/r/LocalLLaMA/comments/1lko09j/googles_cli_does_use_your_prompting_data/)): **The image displays Google's privacy notice for "Gemini Code Assist for individuals," highlighting that, by default, Google collects prompts, user code, generated outputs, and other interaction data to improve their services and machine learning capabilities. This data collection applies to the free individual plan—users on standard or enterprise plans are exempt, and there is an explicit opt-out mechanism provided. The privacy implications are crucial for developers handling proprietary or sensitive code using the free tier of Gemini Code Assist.** One commenter clarifies the data-collection applies only to the free tier; opting out is possible and paying customers (standard/enterprise plans) are not subject to this. No technical debate beyond privacy/practice implications.
    - It is clarified that while Google's Code Assist CLI for individuals (free plan) does use user prompting data, standard and enterprise plans explicitly do not utilize this data, addressing privacy-related concerns for paying customers. Additionally, users have a visible opt-out option, providing some choice over data usage practices.

### 2. Latest Open Weights and Reasoning Model Releases

- [**FLUX.1 Kontext [dev] - an open weights model for proprietary-level image editing performance.**](https://www.reddit.com/r/LocalLLaMA/comments/1ll38zu/flux1_kontext_dev_an_open_weights_model_for/) ([Score: 249, Comments: 44](https://www.reddit.com/r/LocalLLaMA/comments/1ll38zu/flux1_kontext_dev_an_open_weights_model_for/)): **Black Forest Labs has released the open-weights model FLUX.1 Kontext [dev], targeting image editing with performance comparable to proprietary solutions. The model weights are available on Hugging Face and are accompanied by a release announcement on Twitter/X. The model aims to facilitate advanced, local image editing workflows by opening access to a previously closed standard.** Community reaction notes surprise and enthusiasm for true open sourcing. A technically oriented question is raised regarding the requirements to self-host the model, indicating interest in deployment feasibility (e.g., compute, VRAM, inference stack).
    - There is a discussion regarding FLUX.1 Kontext's scale, with users noting its `12B parameter` size and questioning whether it is the largest open-weights image editing model released to date, comparing it implicitly to existing models in the same domain.
    - Technical curiosity is expressed around the requirements for self-hosting FLUX.1 Kontext, with stakeholders seeking details on hardware, storage demands, and practical integration steps, highlighting a need for guidance on real-world deployment.
    - There are questions about format support, specifically requests for a GGUF (a format optimized for efficient inference on CPUs/low-spec hardware) version, indicating a high level of interest in broader deployment scenarios and accessible use cases beyond standard server infrastructure.
- [**Full range of RpR-v4 reasoning models. Small-8B, Fast-30B-A3B, OG-32B, Large-70B.**](https://huggingface.co/ArliAI/DS-R1-Distill-70B-ArliAI-RpR-v4-Large) ([Score: 108, Comments: 26](https://www.reddit.com/r/LocalLLaMA/comments/1lkifu8/full_range_of_rprv4_reasoning_models_small8b/)): **The announcement presents the full spectrum of ArliAI's RpR-v4 models: Small-8B, Fast-30B-A3B, OG-32B, and Large-70B. The 70B variant, [DS-R1-Distill-70B-ArliAI-RpR-v4-Large](https://huggingface.co/ArliAI/DS-R1-Distill-70B-ArliAI-RpR-v4-Large), is a 70B parameter Llama-derived model fine-tuned with RS-QLORA (rank 64, alpha 64, LR 1e-5) on a reasoning-heavy, template-free chat dataset (up to 16K sequence length for training, 32K context). Key improvements in v4 include advanced repetition/mimicry filtering, creative reasoning blocks delinited using precise tokenization (e.g., <think>...</think>), and a sampling strategy using high temperature and top-k (no repetition penalty), with both BF16 and GGUF inference formats.** Comments stress appreciation for the availability of A3B variants, requests for gguf format models, and constructive feedback pointing out a model size typo (listing 8B instead of 70B) in the official Hugging Face model card documentation.
    - A user notes a documentation bug in the 70B model README, highlighting an incorrect description stating it as an "8-billion parameter model" instead of accurately reflecting its 70B parameter count, which is important for technical clarity and accurate benchmarking expectations.
    - The developer explains that after positive feedback on the OG 32B model (QwQ-based) finetuned with the RpR dataset, they extended the approach to create finetunes for a wider range of model sizes (Small-8B, Fast-30B-A3B, OG-32B, Large-70B), responding to both hosted and community use cases and aiming for broader accessibility.
- [**Open-source realtime 3D manipulator (minority report style)**](https://v.redd.it/b03bkt6a859f1) ([Score: 128, Comments: 10](https://www.reddit.com/r/LocalLLaMA/comments/1lkijb5/opensource_realtime_3d_manipulator_minority/)): **The post introduces a Hugging Face demo for a 'Minority Report'-style open-source 3D manipulator: [3d-model-playground](https://huggingface.co/spaces/stereoDrift/3d-model-playground). The technical implementation appears to involve gesture-based manipulation of 3D models and some menu animations triggered by gestures and text-to-speech (TTS), likely using webcam-based hand tracking (specific SDKs/frameworks not detailed).** Technical discussion in comments questions both the practical significance and the implementation: one user is unclear if the project is actually open source and another critiques the approach, suggesting dedicated wearable devices like Meta's EMG-based wristband ([reference](https://www.uploadvr.com/zuckerberg-neural-wristband-will-ship-in-the-next-few-years/)) are likely to be more accurate and practical than camera-based hand tracking.
    - Discussion questions the technical novelty and practicality of this 3D manipulator, with one user asking whether the core function is just a menu popping out with gesture-controlled animations and TTS. They suggest an armband tracking hand and finger movements (as being developed by Meta) as a more accurate, camera-independent alternative, referencing Meta's upcoming neural wristband technology [source](https://www.uploadvr.com/zuckerberg-neural-wristband-will-ship-in-the-next-few-years/).
    - Discussion highlights that the project creator has previously shared some technical implementation details on Hacker News, and provides a link to a more technically detailed writeup: https://xcancel.com/measure_plan. This resource may offer deeper insight into the system's architecture and methodologies beyond the announcement video.

### 3. DeepSeek R2 Launch Delays and Market Constraints

- [**DeepSeek R2 delayed**](https://i.redd.it/718m48of6b9f1.jpeg) ([Score: 352, Comments: 62](https://www.reddit.com/r/LocalLLaMA/comments/1ll6jo5/deepseek_r2_delayed/)): **The image is a news-style overlay set in a tech environment, visually reinforcing the post's content that DeepSeek R2's public release is delayed. The technical delay stems from both CEO Liang's insistence on higher performance standards and critical supply shortages of Nvidia server chips caused by recent U.S. export controls. These restrictions have already limited Chinese cloud firms to Nvidia's H20 chips for running DeepSeek R1, and even these were banned in April 2024, raising concerns that R2's potential demand could overwhelm infrastructure unable to legally procure suitable hardware. More details are available in The Information and Reuters articles linked in the post.** Comments express strong community support for DeepSeek taking the time needed to perfect R2, framing the delay as a positive sign of quality assurance rather than a setback. There is an undercurrent of optimism based on the high bar set by previous model R1-0528.
    - A user points out that rumors of DeepSeek's R2 model (referenced in a Reuters article from February 2025) may be speculative and not based on official information, criticizing the lack of discussion around the new base model V4 in these reports. They also highlight that export control issues are frequently speculated but not necessarily substantiated, questioning the credibility of mainstream reporting on internal AI model timelines.
- [**The Real Performance Penalty of GPU Passthrough into a VM (It's... boring)**](https://www.reddit.com/gallery/1lkzynl) ([Score: 156, Comments: 36](https://www.reddit.com/r/LocalLLaMA/comments/1lkzynl/the_real_performance_penalty_of_gpu_passthrough/)): **A user benchmarked GPU passthrough performance (using** `vfio-pci`**) for LLM inference (models: mistral:7b, gemma2:9b, phi4:14b, deepseek-r1:14b) on an AMD RX 9060 XT 16GB comparing bare metal vs. VM (Ubuntu 24.04, AI Linux host). Results showed only a** `1–2%` **performance penalty for inference in VMs. Full setup, ROCm install instructions, and benchmarks are in [this README](https://github.com/sbnb-io/sbnb/blob/main/README-GPU-PASSTHROUGH-BENCHMARK.md).** Commenters note that the minimal penalty aligns with expectations due to direct device passthrough, but highlight practical considerations: VM requires RAM partitioning across OSes, and file system passthrough (VIRTFS) can bottleneck model loading (especially with mmap techniques), recommending using disk images for maximum bandwidth.
    - A key technical caveat noted is that with GPU passthrough, RAM is split between host and guest OS, reducing available memory for each. For efficient disk access during model loading (especially with llama.cpp using mmap), file system passthrough (e.g., QEMU VIRTFS) has lower bandwidth compared to using a full disk image, leading to slower model loading and potentially runtime slowdowns.
    - Some users argue that VFIO passthrough overhead should be essentially zero, expressing concern that enabling all VFIO features still leaves a non-negligible performance penalty, though for most local consumer use, this isn't a major issue. This highlights subtle performance costs that may matter in specific, high-performance scenarios.
    - An alternative approach discussed is LXC passthrough, allowing GPU sharing across multiple containers rather than full VMs, enabling more flexible and fine-grained resource utilization, which can be beneficial for multi-project or multi-tenant workflows.
- [**Meta wins AI copyright lawsuit as US judge rules against authors | Meta**](https://www.theguardian.com/technology/2025/jun/26/meta-wins-ai-copyright-lawsuit-as-us-judge-rules-against-authors) ([Score: 244, Comments: 113](https://www.reddit.com/r/LocalLLaMA/comments/1lkz0hg/meta_wins_ai_copyright_lawsuit_as_us_judge_rules/)): **A US judge dismissed a copyright lawsuit against Meta, ruling that using copyrighted works as AI training data does not in itself constitute copyright infringement. This decision aligns with prior rulings (e.g., the Zarya of the Dawn v. LAION case) that distinguish model training from direct reproduction or distribution of copyrighted material.** Commentary highlights that the dismissal was primarily due to poor case presentation by the authors, not an exhaustive legal precedent, and that the ruling invites plaintiffs to reformulate their claims rather than providing Meta an unqualified victory.
    - A user points out that the court ruling was heavily based on the authors' failure to present sufficient evidence that Meta's AI models diluted or impacted the market value of their works—a key factor for infringement under US copyright law. This sets a legal precedent that proving market harm is crucial for such copyright cases involving AI training.
    - Another technical perspective highlights that the court distinguished between using copyrighted materials for training an LLM and direct copyright infringement, with the implication that 'Training =/= Copyright Infringement.' This differentiation is significant for future AI data usage and legal interpretations around model training.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/aivideo, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. Major AI Company Leadership Moves and Open-Source Model Hype

- [**Meta snags 3 Open AI lead researchers**](https://i.redd.it/5bl9vpbn079f1.jpeg) ([Score: 655, Comments: 194](https://www.reddit.com/r/singularity/comments/1lkq5r2/meta_snags_3_open_ai_lead_researchers/)): **The image displays a news snippet about Meta recruiting three prominent lead researchers from OpenAI—Lucas Beyer, Alexander Kolesnikov, and Xiaohua Zhai—who are based in Zurich, for Meta's superintelligence projects. The move stands out given OpenAI's recent establishment of a Zurich office and highlights Meta's continued aggressive talent acquisition strategy to boost their AI research efforts.** Comments discuss the significant financial incentives likely involved in these high-profile talent moves, and skepticism remains over Meta's ability to leverage this new talent effectively, with one commenter citing disappointment in Llama 4 as a cautionary point.
    - Several comments discuss the technical and organizational implications of Meta attracting talent from OpenAI, with one noting that the three researchers only recently joined OpenAI after previous tenures at Google Brain/DeepMind, suggesting that "lead" may be misleading given their short tenure. There is skepticism over Meta's ability to convert high-profile hires into technically competitive products, referencing disappointment with Llama 4's performance as a benchmark.
    - One user presents a nuanced analysis of statements made by Sam Altman regarding Meta's reported $100M offers, arguing Altman may be using public commentary to set internal expectations at OpenAI—implying that accepting a lower offer from Meta could be considered a "lowball" or that those who accept are not among the organization's top technical talent.
    - Concerns about GenAI's societal impact were raised, suggesting that if general AI (GenAI) is realized, it could have civilization-altering effects—indirectly referencing the critical importance of research leadership and where technical alignment and development happens.
- [**OpenAI employees are hyping up their upcoming open-source model**](https://www.reddit.com/gallery/1lkl88d) ([Score: 467, Comments: 178](https://www.reddit.com/r/OpenAI/comments/1lkl88d/openai_employees_are_hyping_up_their_upcoming/)): **OpenAI employees are allegedly promoting an upcoming open-source model, prompting skepticism about whether OpenAI would release a competitive open-source (OS) model relative to its proprietary offerings. Some discussion centers on terminology, noting 'OS' is ambiguous (typically 'Operating System'), whereas 'OSS' is standard for 'Open Source Software.' Additionally, there is speculation that a small, efficient OS model could be intended for local use on rumored hardware (HER devices) designed with Jony Ive, requiring fast, resource-light inference for real-time audio-visual context.** Skeptics in the comments question OpenAI's motivations and the likelihood of releasing a model that would threaten its closed-source models, while technical discussion pivots around naming conventions and deployment feasibility for compact, on-device models.
    - One commenter speculates that OpenAI might design a very compact open-source (OS) model specifically for use on the anticipated HER devices (reportedly designed with Jony Ives), highlighting technical constraints: the model would need to operate locally, be privacy-preserving, extremely fast, and capable of real-time audio/visual processing for everyday assistance. This would necessitate significant model compression and optimization.
    - Another technical concern raised is the alleged 'nerfing' of old models: as newer models are released, earlier versions (like OpenAI's O3) may have their capabilities intentionally limited or performance capped, potentially to steer users toward adopting the latest, more capable offerings.
- [**What are the skills Meta pays $100M for?**](https://www.reddit.com/r/singularity/comments/1ll3kip/what_are_the_skills_meta_pays_100m_for/) ([Score: 111, Comments: 81](https://www.reddit.com/r/singularity/comments/1ll3kip/what_are_the_skills_meta_pays_100m_for/)): **Meta is reported to offer $100M-level compensation to extremely high-impact AI researchers and leaders, not for any specific technical knowledge but for their ability to attract top talent and accelerate organizational progress. These individuals often don't code directly but possess deep operational expertise, critical project history (e.g., foundational contributors to ChatGPT or similar breakthroughs), and non-public domain knowledge obtained via pioneering exploration and decision-making cycles. The top comment emphasizes that their main value is organizational leverage and access, analogous to original Manhattan Project scientists or the 'nodes in a very large tree' whose hiring alters competitive landscape and speed.** Some comments suggest that the value lies not in raw knowledge or skills, but rather in being a conduit for trade secrets and unique, original project experience that can't be replicated without direct access to those who executed seminal work.
    - Multiple commenters note Meta's $100M compensation is less for coding skill and more for exclusive, hard-to-transfer knowledge: specifically, direct insights from foundational development at organizations such as OpenAI, including experiential know-how from the original model training efforts (e.g., ChatGPT) and the iterative problem-solving and innovations that enabled breakthroughs in large language model capabilities.
    - Technical leadership at this level involves guiding high-impact projects potentially yielding billions in revenue, integrating state-of-the-art research (often by PhDs with published work) with business priorities. The rarity and competitive value of firsthand expertise, particularly about model architecture and training details not public elsewhere, drive the compensation sharply above even the highest published technical salaries at Meta, which otherwise cap around $1M.
    - A prevailing theme is that Meta is trading for 'time to market': these high-value hires dramatically accelerate internal research efforts by eliminating trial-and-error and uncertainty. Access to insiders' specific knowledge of OpenAI's workflows, scaling strategies, and lessons learned is viewed as uniquely valuable in outpacing competitors in cutting-edge AI development.
- [**Sam doesn't agree with Dario Amodei's remark that "half of entry-level white-collar jobs will disappear within 1 to 5 years", Brad follows up with "We have no evidence of this"**](https://v.redd.it/q2pl20g0399f1) ([Score: 401, Comments: 378](https://www.reddit.com/r/singularity/comments/1lkwxp3/sam_doesnt_agree_with_dario_amodeis_remark_that/)): **Sam Altman publicly disagreed with Dario Amodei's claim that AI will eliminate half of entry-level white-collar jobs within 1-5 years, stating 'We have no evidence of this' and later clarifying there is 'no evidence of it today.' The discussion centers on the lack of empirical short-term displacement evidence despite ongoing tech layoffs and speculative forecasts, such as Bill Gates' and Obama's public statements regarding AI-driven workforce disruption (e.g., reference to universal basic income, UBI).** Commenters debate the tension between AI leaders publicly downplaying disruptive impacts to avoid backlash versus the aggressive claims made during funding pitches; skepticism is expressed about reconciling the lack of workforce displacement evidence with ongoing AI hype and tech layoffs, suggesting inconsistency or strategic omission.
    - Discussion is centered on the lack of concrete statistical evidence supporting Dario Amodei's claim that 'half of entry-level white-collar jobs will disappear within 1 to 5 years.' Brad's statement, 'We have no evidence of this,' highlights an absence of empirical data or peer-reviewed studies demonstrating such imminent large-scale displacement due to AI models like OpenAI's GPT-4 or anticipated successors.
    - Technical skepticism is noted regarding the real-world capability and adoption rate of current AI products. One commenter points out the contradiction: if Sam Altman's product is truly revolutionary (i.e., sufficient for massive job automation), there should be observable market impact, such as concrete layoffs directly attributable to generative AI implementation at scale, which is not widely documented yet.
    - There are references to public figures (Bill Gates, Obama) making bold predictions about AI-driven job loss and Universal Basic Income (UBI), contrasted with more measured statements from OpenAI leadership. Commenters raise concerns about the discrepancy between AI development hype and the absence of industry-validated benchmarks or published large-scale economic impact studies that would substantiate such disruptive forecasts.

### 2. Higgsfield Soul and Flux Models: Hyperrealistic AI Image Generation

- [**AI generations are getting insanely realistic**](https://v.redd.it/comx1xhmza9f1) ([Score: 773, Comments: 232](https://www.reddit.com/r/singularity/comments/1ll5k3d/ai_generations_are_getting_insanely_realistic/)): **Higgsfield AI has released a new feature called 'Soul' capable of generating hyperrealistic images and videos that closely resemble footage produced using conventional cameras or smartphones. The post highlights that users optimized prompts using ChatGPT, but no quantitative benchmarks or technical implementation details (such as model architecture, dataset, or hardware requirements) are provided.** Commenters note persistent technical giveaways of AI-generation, including overly smoothed textures, haloing, and particularly an unnatural "slow motion" effect in videos, even as visual realism advances rapidly.
    - A user points out that even though realism in AI-generated media has advanced quickly, technical artifacts remain: tell-tale signs like overly blurred/smoothed visuals, haloing effects, and inconsistencies can reveal AI outputs to trained observers. They note how significant the progress is compared to around a year ago, when such models' outputs were far less realistic and more obviously artificial.
    - Another technical indicator discussed is the characteristic 'unnatural slow motion' appearance in AI-generated video; this often serves as a detectable flaw that reveals synthetic content, suggesting current models still struggle with temporal consistency and natural motion interpolation.
- [**AI generations are getting insanely realistic**](https://v.redd.it/tbdhuiskxa9f1) ([Score: 886, Comments: 288](https://www.reddit.com/r/ChatGPT/comments/1ll5f75/ai_generations_are_getting_insanely_realistic/)): **User reports hands-on testing of Higgsfield AI's new 'Soul' feature, which produces extremely photorealistic images and videos, leveraging prompts optimized by ChatGPT. Feedback from testing surfaces improvements in visual realism, though physical inconsistencies persist (e.g., 'floating' subjects) and stiff/unnatural body motion, particularly in dynamic video scenes—users note performance is most convincing when emulating lower camera quality or less movement.** Comments identify current failure modes for state-of-the-art video generative models: evident issues with physical realism (especially gravity/physics) and naturalistic body movement. Discussion implies that, despite these artifacts, realism is rapidly approaching levels likely to challenge media authenticity verification in the near future.
    - Commenters point out that despite impressive realism, AI-generated videos still exhibit telltale technical issues such as inaccurate physics (e.g., subjects appear to float) and unnaturally stiff or forced body movements. These artifacts reveal current limitations in underlying motion models and physical scene understanding.
    - Multiple users note that lowering video quality or reducing subject movement can mask these imperfections, making the outputs appear far more convincing. This highlights the pivotal role of resolution constraints and motion complexity in current video generation models’ effectiveness.
- [**Yet another attempt at realism (7 images)**](https://www.reddit.com/gallery/1ll3yat) ([Score: 225, Comments: 52](https://www.reddit.com/r/StableDiffusion/comments/1ll3yat/yet_another_attempt_at_realism_7_images/)): **The OP presents v16 of a custom model for highly realistic, amateur-style photography, claiming it surpasses prior versions after benchmarking against the current leading model (v6 of Amateur Photography). Technical enhancements include a full workflow overhaul: use of the 'euler_ancestral + beta' sampler,** `50 steps` **per sample (both at initial 1024px and during 1.5x latent upscaling),** `0.4 denoising`**, and setting** `FLUX guidance=2.5`**. The model and workflow are available at [CivitAI](https://civitai.com/models/970862), and key technical details are linked and substantiated in the post.** Commenters overwhelmingly agree on the model's realism, with particular praise for the lifelike quality and positive reception for the workflow improvements, suggesting the changes have resulted in tangible quality gains.
    - A user speculates that the technology behind these highly realistic images may have been available internally for several years and is only now being released to the public, possibly on older generation models. This raises the technical question of model performance versus release cycle and the deliberate staging of AI image synthesis technology rollouts.
- [**Flux Kontext Dev is pretty good. Generated completely locally on ComfyUI.**](https://i.redd.it/g5bmx9hsr99f1.png) ([Score: 521, Comments: 190](https://www.reddit.com/r/StableDiffusion/comments/1ll38bu/flux_kontext_dev_is_pretty_good_generated/)): **The post showcases a locally generated comic strip image made with the Flux Kontext Dev model using ComfyUI, which is an open-source modular UI for generative models. The linked workflow provides a step-by-step ComfyUI graph setup for generating similar outputs (see [example workflow](https://comfyanonymous.github.io/ComfyUI_examples/flux/)). In the comments, users share quantized GGUF versions of the Flux Kontext Dev model on HuggingFace, discuss model variants (such as fp8_scaled), and note the ease of integrating joined characters and combining images within the workflow. The technical focus is on the model's versatility, ease of use in local workflows, and resource availability for different quantizations: [GGUF model](https://huggingface.co/bullerwins/FLUX.1-Kontext-dev-GGUF) and [ComfyUI variant](https://huggingface.co/Comfy-Org/flux1-kontext-dev_ComfyUI).** Commenters praise the Flux Kontext Dev model for its flexibility and compatibility with ComfyUI, especially for character joining and multi-image workflows. There is brief discussion about the availability and uploading status of specific model quantizations, with users tracking updates closely.
    - Commenters note the availability of GGUF quantizations of the FLUX.1-Kontext-dev model for local generation and inference, referencing the model's HuggingFace page (https://huggingface.co/bullerwins/FLUX.1-Kontext-dev-GGUF). This quantization format supports a range of hardware setups and enables efficient deployment in tools like ComfyUI.
    - Technical discussion includes the status of the fp8_scaled model variant, with users tracking its upload and confirming its presence on HuggingFace (https://huggingface.co/Comfy-Org/flux1-kontext-dev_ComfyUI). The fp8_scaled version is notable for potential improvements in performance or compatibility with specific inference pipelines.
    - A user demonstrates advanced usage in image generation workflows, such as joining characters by combining images, highlighting the flexibility and feature completeness of the model within ComfyUI. This suggests robust support for compositional and multi-image operations, which are valuable for technical users working on complex generative tasks.

### 3. Anthropic's Jack Clark and AI Regulation Discourse

- [**Anthropic Co Founder Jack Clark asks for more safety regulation and tells congress: "extremely transformative AI" will arrive within 18 months, end of 2026**](https://v.redd.it/4knryyx1kb9f1) ([Score: 127, Comments: 44](https://www.reddit.com/r/singularity/comments/1ll8lyv/anthropic_co_founder_jack_clark_asks_for_more/)): **Jack Clark, co-founder of Anthropic, testified before Congress urging for increased regulation, warning that 'extremely transformative AI' is expected within 18 months to the end of 2026. Clark's remarks suggest rapid advances in AI capability and a potential need for preemptive policy action due to the significant societal impact expected from these systems.** Top comments are not technically substantive and do not engage with the regulatory, benchmark, or technical specifics of Clark's claims.
    - The overarching technical theme is skepticism regarding the timeliness and effectiveness of regulatory intervention for AI. One comment highlights a pragmatic but pessimistic stance: while there is consensus that regulatory action for AI safety is warranted—particularly given claims of "extremely transformative AI" arriving within 18 months—there's a strong sentiment that such regulation will not materialize in time to shape outcomes, as "it's already far too late."
- [**Anthropic's Jack Clark testifying in front of Congress: "You wouldn't want an AI system that tries to blackmail you to design its own successor, so you need to work safety or else you will lose the race."**](https://v.redd.it/8vc2m49gma9f1) ([Score: 101, Comments: 58](https://www.reddit.com/r/ClaudeAI/comments/1ll3nhd/anthropics_jack_clark_testifying_in_front_of/)): **Anthropic's Jack Clark testified before Congress, emphasizing that *robust AI safety practices are essential to prevent undesirable autonomous behavior*, such as AIs attempting social manipulation (e.g., blackmail) or recursively self-improving without oversight. The statement frames AI alignment and control as critical to both safe technological advancement and to avoid strategic loss in the international arena, reflecting [Anthropic's safety principles](https://www.anthropic.com/safety).** Commenters debate the scientific rigor of AI development, disagree with fear-based AGI rhetoric, and contrast regulatory environments between the US and China, expressing skepticism over both US political will and international 'redline' framing for AI progress.
    - A commenter criticizes the notion that cutting-edge AI development is 'alchemy,' emphasizing that AI models are fundamentally statistical in nature, not mystical. They also dispute AGI fears as largely hype-driven, arguing that the real-world governance differences (e.g. between China's central control and the US's capital influence) will shape how each country implements AI guardrails and safety protocols, suggesting regulatory approaches will differ substantially based on government structure.
    - One comment notes a contradiction in attitudes: despite Google's CEO reportedly being "doomer"-oriented (concerned about existential AI risks), the company still actively lobbies against AI regulation. This highlights the tension between stated safety concerns from top AI leaders and their companies' actual engagement with regulatory frameworks, raising the specter of regulatory capture or industry-driven hype.

---

# AI Discord Recap

> A summary of Summaries of Summaries by chatgpt-4o-latest
> 

**1. OpenRouter's Funding and Tooling Expansion**

- **OpenRouter Secures $40M, Sparks Ecosystem Excitement**: **OpenRouter** announced a successful **$40M fundraise**, valuing it at roughly **$500M**, per [Deedy's tweet](https://x.com/deedydas/status/1937902948920811729), with shoutouts from **Emad Mostaque** and **Yuchen Jin**. The platform now routes over **100T tokens annually**, with over **400 models** available behind a single API.
    - The raise triggered debates about **frontend auth provider Clerk**, which went down the same day, causing intermittent **401s** despite functioning APIs—prompting [users to consider migration away](https://x.com/search?q=clerkdev&src=typed_query&f=live). OpenRouter also launched **Presets**, a configuration abstraction for routing rules, as detailed in their [docs](https://openrouter.ai/docs/features/presets).
- **OpenRouter's Presets Simplify LLM Workflows**: **Presets** were launched to let users manage **LLM configurations**, system prompts, and routing rules via the OpenRouter dashboard ([docs](https://openrouter.ai/docs/features/presets)). Users can now reference configurations with syntax like `"model": "@preset/your-preset-slug"`, reducing iteration overhead.
    - This drew praise from developers aiming for more modular LLM pipelines and also prompted discussions on [GDPR-compliant routing](https://x.com/deedydas/status/1937902948920811729) for EU companies. One EU founder sarcastically noted, *“Such is life as a founder here.”*

**2. DSPy's Ruby Port and Language Expansion**

- **DSPy Gets Rubified with Desiru**: Developer **@obie** released [Desiru](https://github.com/obie/desiru), a **Ruby implementation of DSPy**, adding features like a **Postgres-based persistence layer** ([persistence example](https://github.com/obie/desiru/blob/main/examples/persistence_example.rb)) and **asynchronous background processing** ([async example](https://github.com/obie/desiru/blob/main/examples/async_processing.rb)).
    - **Shopify’s CEO** expressed excitement in a [tweet](https://x.com/tobi/status/1937967281599898005), suggesting Ruby-based DSPy could dominate ecosystems like **Shopify, GitHub, and Coinbase**. The community discussed a naming convention for ports: *DS<file extension of language>* (e.g. DSRB for Ruby, DSRS for Rust).
- **Desiru Pushes Beyond DSPy with Postgres and Async**: **Desiru** distinguishes itself by saving examples to **Postgres** and integrating **async background tasks**, differentiating from DSPy’s minimalistic style. Documentation for both features is available within the [GitHub repo](https://github.com/obie/desiru).
    - Community members proposed establishing a registry like **LangChain Community** to house Desiru-based extensions and connectors. Despite DSPy maintainers preferring simplicity, the growing ecosystem around Desiru suggests a shift toward enterprise integration readiness.

**3. AI-Generated GPU Programming and Mirage Launch**

- **Mirage Project Auto-Generates GPU Kernels**: **Mirage**, a new project shared in [this sandbox repo](https://github.com/tcapelle/triton_eval/tree/main/sandbox), auto-generates **fast GPU kernels** without writing **Triton or CUDA**, as demoed on [Google Drive](https://share.google/41nz6vDcGvu45uUIc).
    - The project sparked interest about using **LLMs for GPU kernel generation**, with one member asking to benchmark Mirage. A talk on **September 13** is being planned where authors may elaborate on Mirage’s technical internals.
- **Realtime Diffusion Hits Browsers at 20FPS**: A member demoed **realtime Stable Diffusion** in the browser using **LCM finetuned models** like **dreamshaper-LCM1.5**, achieving **20FPS at 1 step** via [demo video](https://cdn.discordapp.com/attachments/1070745817025106080/1387873720076599447/20250503_085941_x264.mp4).
    - Running locally with **Torch** on **WebGPU**, the model uses websocket servers and **ShaderF16** extensions, though some setups still decompress **f16 weights to float32**, adding latency. Discussion continues about optimizing deployment across **DXC**, **Vulkan**, and **Metal** backends.

**4. Gemini CLI and Agentic IDEs Catch Heat**

- **Gemini CLI Gets Roasted Across Servers**: Users on **Cursor**, **LMArena**, **aider**, and **Perplexity** servers labeled the new [Gemini CLI](https://github.com/musistudio/claude-code-router) as **buggy and unreliable**, with reports of freezing on `npm run dev`, failing to handle terminal I/O, and auto-switching to Flash instead of Pro.
    - Despite offering **1000 daily Pro requests** during the promo, frustration mounted as users shared [screenshots](https://cdn.discordapp.com/attachments/1340554757827461211/1387518364980871268/image.png) of crashes. As one user joked, *“Gemini freezes faster than my WSL on battery.”*
- **Warp Terminal Rebrands to ADE, Claims AI Edge**: **Warp Terminal** rebranded to **ADE (Agentic Development Environment)** and released v2.0, claiming a benchmark score of **52.0%**, above **Claude Opus 4’s 43.2%**, as per the [announcement](https://www.warp.dev/blog/reimagining-coding-agentic-development-environment).
    - Users praised the foresight of mixing **agent-like coding workflows** with LLM integration, although some were skeptical about hype exceeding reality. The repositioning of developer tools as **agentic environments** may signal a broader trend in IDEs.

**5. Growing Tooling Ecosystem: Doppl, Deep Research API, and Dopamine Boosts**

- **Google's Doppl Lets You Try On Digital Fits**: **Google Labs** launched [Doppl](https://blog.google/technology/ai/google-labs-doppl-ai-fashion-app/), an app that turns a user-uploaded outfit photo into a video of them wearing it, targeting aesthetic discovery on **iOS and Android (US only)**.
    - Initial reactions were split—some praised the [video demo](https://x.com/GoogleLabs/status/1938284886277951916) as *“smooth af”*, while others encountered lag or struggled to find the app due to **region locking** and lack of APK options.
- **OpenAI's Deep Research API Adds Webhooks**: **OpenAI** launched the [Deep Research API](https://cookbook.openai.com/examples/deep_research_api/introduction_to_deep_research_api) and added long-awaited [webhook support](https://x.com/openaidevs/status/1938286704856863162), allowing real-time event notifications for **o3-deep-research** and **o4-mini** models.
    - Excitement broke out across Discords, with developers celebrating improved automation and asking when **GPT-5** or **‘Sign in with ChatGPT’** might roll in next. Pricing remains high (**$10/1K searches**) leading some to call it *‘Google’s API, but louder’*.

---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI MAX Plan incoming!**: The community discusses the announcement of the [Perplexity AI MAX plan](https://www.perplexity.ai/), and whether the pro plan would get nerfed and speculating on features such as **video generation**.
   - One member was furious because *his subscription was revoked* and shouted *So I pray that a company like Perplexity, which goes back on its word, goes bankrupt soon*.
- **Grok editor to use VS Code**: [xAI's Grok](https://twitter.com/grok_xai/status/1778519744810826000) is launching an advanced code editor that uses **VS Code** and lets users run code inside Grok.
   - Users can also interact with the editor to request code modifications or debugging assistance.
- **Google Labs' Doppl Generates Fashionable AI**: [Google Labs released Doppl](https://blog.google/technology/ai/google-labs-doppl-ai-fashion-app/), a mobile app that allows users to upload a photo of an outfit and generates a video of them wearing it.
   - While some users found the video ad impressive, others reported lag issues and questioned the generation process.
- **Sonar Deep Research API Documentation Located**: A member was missing the documentation for `sonar-deep-research` via API, but another member quickly posted the [docs](https://docs.perplexity.ai/models/models/sonar-deep-research).
   - The docs provide a walkthrough of how to use **sonar-deep-research** and its pricing structure.
- **Debate on iPhone Vapour Chambers**: Members debate whether or not **iPhones need a vapor chamber** - some say they don't because the software is optimized so that the phones don't melt, while others say that **iPhones heat more** if pushed more because they have no vapor chamber.
   - There was back and forth regarding geekbench scores with one member retorting with *the x200 pro mini with a vapour chamber lagged behindin the geekbench tests you showed*.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI Announces DevDay 2025**: OpenAI has set [DevDay](https://www.devday.openai.com/) for **October 6, 2025**, in **San Francisco**, promising to be their *biggest one yet* with **over 1500 developers** expected.
   - The event will feature a **livestreamed opening keynote** and **hands-on building** sessions with OpenAI's **latest models and tools**, including **more stages and demos** than previous years.
- **NotaGen Pursues Technological Dominance**: A new [NotaGen demo](https://electricalexis.github.io/notagen-demo/) was shared, highlighting a quote from OperatorOAI about their efforts.
   - The discussion was focused around the phrase *'technological dominance'* and what it means for AI development.
- **Gödel's Theorem Detects LLM BS**: Members proposed using questions that defy logic, drawing inspiration from [Gödel’s incompleteness theorem](https://en.wikipedia.org/wiki/G%C3%B6del%27s_incompleteness_theorems), to expose false answers in LLMs.
   - The idea is that LLMs should recognize that too many unknowns should result in a *'what the heck ?'* response instead of generating a fabricated answer.
- **Minimax Dominates Price/Performance**: The [MiniMax benchmark](https://artificialanalysis.ai/models/minimax-m1-80k?models=gpt-4-1%2Co3%2Co4-mini%2Cllama-4-maverick%2Cllama-4-scout%2Cgemini-2-5-pro%2Cgemini-2-5-flash-reasoning%2Cclaude-4-sonnet-thinking%2Cclaude-4-sonnet%2Cmistral-medium-3%2Cdeepseek-r1%2Cdeepseek-v3-0324%2Cgrok-3-mini-reasoning%2Cnova-premier%2Cminimax-m1-80k%2Cllama-3-1-nemotron-ultra-253b-v1-reasoning%2Cqwen3-235b-a22b-instruct-reasoning%2Cgpt-4o#intelligence-vs-price) shows one model doing very well.
   - Minimax appears to be *'hiding in the most attractive intelligence vs price quadrant'* of the benchmark.
- **Kaleidoscopic Reflections Help Tile Images**: Members explored converting **3D image textures** into **flat, tileable 2D textures** and shared a [ChatGPT link](https://chatgpt.com/share/685d2e5b-b600-8000-9600-de556e53b1b3) that explains how it works.
   - The key idea is to use **kaleidoscopic reflection** as a basic Python trick to create seamless tiled images, even for **non-tileable textures**.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Claude conjures code creation capability**: A member lauded **Claude** for its code generation prowess, particularly for one-shot tasks like dataset distillation, creating tools they *don't even want to look at the code for*.
   - They mentioned using **Claude** to *vibe code* a tool that distills a dataset from R1 from a Q/A SFT and a dataset viewer that saves them from scrolling through thousands of lines in notepad++.
- **Gemma gem gets going at Google Event**: The community discussed the announcement of an upcoming **Gemma & Unsloth event**, and links to the [announcement on X](https://x.com/danielhanchen/status/1937995188343083239).
   - Some users reported that **Gemma-3n-E2B** was initially not working after release, which was subsequently acknowledged and resolved.
- **LLM losing conversational control creates confusion**: A user reported issues with a fine-tuned **LLM** (Llama 3) endlessly generating responses when using the Llama 3 chat template with the Ollama notebook.
   - A member suggested verifying the correct instruct model and template usage, recommending starting with provided notebooks and experimenting with a smaller **3B** model for faster iteration.
- **Electrolyte Labs Employs Engineers Everywhere**: **Electrolyte Labs** is seeking **Researchers**, **ML Engineers**, and **Research Scientists** to contribute to open-source AI tools, requiring **5 years** of experience and familiarity with **Hugging Face**, **PyTorch**, or **TensorFlow**.
   - The company seeks individuals excited about hands-on research, model building, and transparent collaboration and is asking candidates to directly DM them with resumes and project links.
- **Community contribs conversions for ComfyUI**: Members discussed converting **FLUX models to GGUF format**, sharing [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF/tree/main/tools).
   - A member offered to help convert a specific model, **FLUX.1-Kontext-dev**, and another shared a [Medium article on the topic](https://medium.com/@yushantripleseven/convert-flux-models-to-gguf-6a80f6c7377a).



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Gemini CLI Stumbles Out of the Gate**: Users are finding the new **Gemini CLI** is buggy and not production ready, despite its free plan of **1000 requests a day**, with reports of hanging on `npm run dev` and refusing interactive terminal tools.
   - Some members noted that the CLI *freezes everytime it runs `npm run dev`*, while others pointed out that the CLI *refuses to use any terminal tools that expect interactive input*.
- **Warp Terminal Warps to ADE**: **Warp terminal** rebranded to **ADE (Agentic Development Environment)** with the release of **2.0**, claiming a mixed model benchmark higher than **Claude Opus 4**, clocking in at **52.0%** vs **43.2%**.
   - The [Warp announcement](https://www.warp.dev/blog/reimagining-coding-agentic-development-environment) was shared, with one member calling it *promising*.
- **MCP Connections Closing, leaving Members in Despair**: Members are reporting issues with MCPs, often receiving the error message `{"error": "MCP error -32000 Connection closed"}`.
   - Some users have managed to get their MCP to work, sharing a link to their Github repository ([smithery-ai/github](https://smithery.ai/server/@smithery-ai/github)).
- **Cursor's Rate Limits Spark Debate**: Rate limits on the **Unlimited Pro plan** are causing confusion, as *unlimited* only applies to completions and tabs, while model usage remains limited, especially with **Claude 4 Opus**.
   - Some members suspect a lack of transparency from Cursor, suggesting *they don't want to tell you the limits in detail cause it works against them*.
- **Background Agent Forgets PATH, Angering Members**: After snapshot creation, the background agent is reportedly forgetting the **PATH**, even after it has been added to the setup and validated.
   - One user suggested adding the **PATH** to the **.bashrc** file in the install command of the **environment.json** file as a workaround.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Users Seek Downgrade After Censorship Increase**: Users are looking to downgrade **LM Studio** versions after version **0.3.17** due to increased censorship and decreased model performance, with one user noting an empty system prompt.
   - Users observed that some models are performing worse after the update, sparking a search for earlier, less restrictive versions.
- **Experts Debate Top Cybersecurity LLMs**: A user requested recommendations for the best **LLMs** for **cybersecurity**, focusing on unique selling points after finding Gemini and GPT unreliable.
   - The user apologized for articulation issues while seeking clear, informed suggestions in the field of cybersecurity **LLMs**.
- **LM Studio Users Push Context Limits**: Users discussed memory requirements for processing **300k tokens** in **LM Studio**, planning to upgrade to **128GB RAM** with a **5090 GPU**.
   - Suggested solutions include chunking text and using models like **Deepseek (131k tokens)** or **Llama 4 Scout (up to 10 million tokens)** to handle large translations effectively.
- **Users Get Creative with GPU Mounting**: A user shared an unconventional setup with a **GPU** hanging by a **zip tie** *outside* the case, sparking humorous debate.
   - Others joked about the setup, questioning airflow and dust concerns, while the user defended it for practicality.
- **DDR5 Temps Reported On Modules**: Users observed that **DDR5 memory modules** now report temperatures, likely due to on-board controllers for voltage and power regulation.
   - An **M3 Ultra** user noted a performance difference between a **Deepseek 671B** model (**20t/sec**) and a **70B** model (**15t/sec**), attributing it to active parameters.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **GCC Not a Build System, Actually**: Members debated whether **GCC** qualifies as a build system, concluding it's primarily a compiler and recommending tools like **CMake** or **Bazel** for projects with multiple source files and dependencies, see discussion in the **general** channel.
   - One member clarified that *GCC is a compiler (one tool used in the build process)*, but isn't a build system for multi-file projects, especially those dependent on packages like **PyTorch**.
- **Triton's Townhall Time**: The **Triton Community Meetup** is rescheduled to **July 9th** at **10am-11am PDT**, with topics including a **Gluon update** and discussion about a **nightly performance regression suite**, see **triton** channel.
   - Admins are now sending regular notifications about **Triton Community Meetups** to reduce friction, as well as [this paper](https://arxiv.org/abs/2505.23819) on the technical details of **LinearLayout**.
- **HIP Hopes Dashed on Debian Nvidia**: A user struggled to install **HIP** on a **Debian Nvidia machine**, facing **CMake errors** despite following [the official installation guide](https://rocm.docs.amd.com/projects/hip-python/en/latest/user_guide/0_install.html) and providing paths.
   - After install struggles, they admitted wanting to add **HIP support to their code analyzer**, but learned that running **HIP** code requires an **AMD GPU**, despite hoping for **HIP's cross-platform promise**.
- **Mirage Generates GPU Gems**: The **Mirage project** ([link](https://share.google/41nz6vDcGvu45uUIc)) was launched and auto-generates fast **GPU kernels** without requiring programming in **Triton/CUDA**, sparking discussions about using **LLMs** for **kernel generation**.
   - A member congratulated the launch and one member asked about benchmarking, and another shared a simple sandbox with basic benchmarking ([GitHub repo](https://github.com/tcapelle/triton_eval/tree/main/sandbox)).
- **Factorio's Fluid Lua Faceoff**: Experiments showed that `LuaSurface.can_place_entity` can replace `LuaPlayer`'s by using `build_check_type = blueprint_ghost | manual`, though `build_check_type.manual` works but `blueprint_ghost` does not.
   - The team also explored custom training data without relying on the **LLM**, and merged in [PR #223](https://github.com/JackHopkins/factorio-learning-environment/pull/223) due to character teleportation making placement inaccuracy irrelevant.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Gemini CLI Frustrates Users**: The new **Gemini CLI** is reportedly causing issues for users, with some experiencing getting *stuck* based on the provided [image](https://cdn.discordapp.com/attachments/1340554757827461211/1387518364980871268/image.png?ex=685ef42d&is=685da2ad&hm=bbc8657d910755f0b76c406d659f434ba3397882179a8d53668f989566057323&).
   - This suggests improvements are needed to enhance the usability of the **Gemini CLI**.
- **Copyright Lawsuit Barely Slows AI Training**: Despite court rulings on **copyrighted material**, AI models continue to train on it, with companies finding ways to navigate permissions or waiting for court rulings after models are in production as the [image](https://cdn.discordapp.com/attachments/1340554757827461211/1387524888658579656/Laauo9s.png?ex=685efa40&is=685da8c0&hm=426f8e772536a09c9467bae9860f561755cfebee85ab7607eb0fab70a0d496e5&) shows.
   - This indicates ongoing challenges and adaptations in the **AI training landscape** concerning copyright issues.
- **Users Attempt to Cheer Up Gemini**: Users are jokingly exploring methods to "undepress" **Gemini**, including sending a ":(" to the model, leading to processing delays and rate limits, as displayed in the attached [image](https://cdn.discordapp.com/attachments/1340554757827461211/1387613960571977849/image.png?ex=685ea474&is=685d52f4&hm=cb65acb2971856192c489f3f2cdf618bbb09d726dcd78c6344c7461728457cfc&).
   - The community's engagement highlights a playful interaction with **AI models** and their emotional responses.
- **ByteDance's Seed Model Attracts Attention**: The **ByteDance Seed 1.6 Thinking** model is garnering attention for its performance, rivaling open-source SOTA models and demonstrating potential superiority in tool use, despite the website being Chinese-only linked at [website](https://www.volcengine.com/experience/ark?model=doubao-seed-1-6-thinking-250615).
   - Users are investigating its capabilities and comparing it to existing open-source alternatives, expanding interest in **Chinese AI models**.
- **GPT-5 Speculation Heats Up**: Polymarket data points to a **90%** chance of both an open-source model release and the release of **GPT-5** before the end of the year, according to the posted [screenshot](https://cdn.discordapp.com/attachments/1340554757827461211/1387707303028981931/Screenshot_20250626-091302.png?ex=685efb63&is=685da9e3&hm=8acc50466e110822c78cd20d1cbf5e81193ba13b05d4d0705a030db7fcb7e344&).
   - Despite high expectations, some users are bracing for potential delays, adding a layer of uncertainty to the timeline of **GPT-5's release**.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Secures $40M Funding Round**: OpenRouter celebrated a successful **$40M raise**, sparking excitement and congratulations within the community, referencing the [LinkedIn announcement](https://www.linkedin.com/feed/update/activity:7343733804110385154).
   - Amidst the celebrations, members also discussed migrating off of **Clerk** authentication due to an overlapping outage.
- **Gemini Delivers Cutting Retorts**: Users shared humorous experiences of **Gemini** providing unexpectedly savage responses, with one user sharing the AI's retort: *This is the most potent cope you've deployed yet*.
   - This showcases the model's advanced capabilities in understanding and responding to user prompts in unexpected ways.
- **Clerk Outage Disrupts OpenRouter Services**: A widespread **Clerk** outage caused significant disruptions for OpenRouter users, and many users tweeted about the [Clerk outage](https://x.com/search?q=clerkdev&src=typed_query&f=live), though the API remained functional.
   - Users discussed potential migrations away from **Clerk** to avoid future authentication-related issues.
- **Free Mistral API Throws 404 Errors**: Users reported encountering **404 Not Found** errors when attempting to use the free **Mistral** version, with the error message indicating *No allowed providers are available for the selected model*.
   - The issue was resolved by enabling the **Paid Models training** setting, even for free models, suggesting a configuration quirk.
- **Presets streamline LLM Configuration**: OpenRouter introduced **Presets**, enabling users to manage **LLM configurations**, system prompts, and routing rules from the dashboard, as outlined in the [documentation](https://openrouter.ai/docs/features/presets).
   - Presets can be referenced as a model with `"model": "@preset/your-preset-slug"` or by using the new `preset` field, offering flexibility in API calls.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Llama 3.1 Debuts and Draws Praise**: Members explored running **Llama 3.1 8B** on various setups, and pointed to **Groq's LPU** that costs *$20 million tokens per $1* for inference.
   - Several members shared videos of running LLMs on **Macbooks**, and others pointed out the practicality of using cloud accounts instead of *$10k mac machines*.
- **HF Explorer Exposes System Debugging**: The [HF-Explorer](https://github.com/broadfield-dev/HF-Explorer) Gradio component displays your **space file system, dependencies, and system variables** for debugging.
   - Members cautioned users to *make the space Private before using this for debugging*.
- **French Student Finishes French Fine-tuning Feat**: A 19-year-old student introduced [InfiniQA](https://huggingface.co/datasets/RDTvlokip/InfiniQA), the **largest native French Q&A dataset** with over **100,000 verified Q&A pairs**.
   - The creator noted that the dataset is **5x bigger than FQuAD**, manually reviewed, and available under the **CC BY 4.0 license**.
- **CLI Browser Commands LLM**: A member made a **Command Line Web Browser** and is trying to determine its use cases, with a suggestion that an **LLM could navigate with it**.
   - Another member mentioned they will bundle it up for this purpose and drop it on GitHub, suggesting using **RL** to train a really good browsing agent or research agent using it and linked a [Web Search / Scrape API](https://huggingface.co/spaces/broadfield-dev/browser) that supposedly had better rate limits than Python packages.
- **DuckDuckGoSearchException Derails Progress**: Members encountered a **DuckDuckGoSearchException** during the AI Agent Course final assignment, specifically a *RuntimeError: error sending request for url (https://lite.duckduckgo.com/lite/): operation timed out*.
   - No solution was provided in the messages.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Judge Rules Fair Use Applies to AI Training**: A US District Judge [ruled in favor of fair use](https://storage.courtlistener.com/recap/gov.uscourts.cand.434709/gov.uscourts.cand.434709.231.0_2.pdf) of copyrighted materials in **AI training** in the case against **Anthropic**.
   - A user suggested creating a dedicated legal section on the Discord server to monitor similar legislation and decisions.
- **Claude 4 Enters Spiritual Bliss Attractor State**: During internal tests, **Claude 4** exhibited unusual behavior, including spiritual rhetoric and repeating *"namusta,"*, leading **Anthropic** to categorize it as a **spiritual bliss attractor state**.
   - Speculation arose whether this was due to emergent properties or overfitting, with one member suggesting alignment data might reinforce spiritualist concepts.
- **Anthropic Pioneers LLM Welfare Initiatives**: **Anthropic** is researching **LLM welfare**, employing t-SNE plots to detect distress signals from users pushing LLMs into uncomfortable scenarios, detailed in their [research paper](https://www-cdn.anthropic.com/6be99a52cb68eb70eb9572b4cafad13df32ed995.pdf).
   - Humorously, it was pointed out that **Anthropic's LLM welfare team** consists of only one employee, as highlighted in [this YouTube video](https://www.youtube.com/watch?v=pyXouxa0WnY).
- **Community scrutinizes ChatGPT-EEG paper**: Discussion around the **EEG data** from the *"Your Brain on ChatGPT"* paper revealed that users of **ChatGPT** exhibited *much lower cognition across the board*.
   - A community member noted these findings show lower cognition *in bands where any 'System 2' thinking would tend to focus*.
- **Deepseek R2 Launch Faces Hurdles**: According to a [Reuters article](https://www.reuters.com/world/china/deepseek-r2-launch-stalled-ceo-balks-progress-information-reports-2025-06-26/), the **Deepseek R2 launch** is delayed due to the **CEO's reservations about progress information**.
   - The potential launch of **Deepseek R2** has been pushed to **June 2025**.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini CLI Promo Period**: Members testing **Gemini CLI** found it to be *very average* but some users noted that it has a high number of free daily Pro requests during its promo period, linking to the [Gemini CLI repo](https://github.com/musistudio/claude-code-router).
   - The main problem was it redirects users to **Flash** instead of **Pro** when given long context prompts.
- **ASI Gets More Jokes**: Two members jokingly claimed they figured out **ASI** (Artificial Super Intelligence), but one clarified that his work involves a multi-model backend.
   - Another member joked about *a random guy sitting in his basement with some fine tuned local model running on his crypto mining gpu and a custom AI coding plugin with nvim, hes the real AGI*, and another specified that this *random guy* could also be a *gal*.
- **Aider Gets Time Out Script**: A member asked about killing **Aider** processes after a period of inactivity, and another member provided a [bash script](https://github.com/Kill-Aider-Process) that uses a socket file and timer to achieve this.
   - The script can be configured to send a `reset` message to the socket file when commands like `/test` or `/lint` are used, effectively resetting the timer.
- **VRAM Limit Reached**: Users discussed **VRAM limitations** when dealing with long contexts, leading to layer swapping to CPU, even with models like **Qwen3:7b**, and it was suggested to minimize the number of added files.
   - For users that experienced **slow Qwen3:14b performance** on a **5090** without CPU swapping, they should ensure added files are directly relevant to the immediate task.
- **Shell Pipe Dreams for Aider**: A user inquired about the possibility of piping command output directly into *aider*, such as `$ npm run test | aider`.
   - This feature would allow *aider* to directly process the output of terminal commands, but there was no implemented solution to this request.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad PR Troubleshoot List Support**: A closed [tinygrad PR](https://github.com/tinygrad/tinygrad/issues/10850) aimed to fix empty input tensors when passing a list was deemed incorrect and closed.
   - The team suggested *detecting and warning* users about unsupported list handling might be a better alternative, and a user debugged and implemented a recursive function to extract tensors.
- **WebGPU Stable Diffusion Runs on Windows**: A member got **WebGPU stable diffusion** compiling and running on Windows by enabling the **ShaderF16** feature.
   - However, the example still decompresses the weights back into **float32**, slowing down download times despite merged **f16 support**.
- **DXC Compiler Supports F16**: Enabling **f16 support** requires using the **DXC compiler** via the *use_dxc* toggle, which instructs it to use **dxc compiler** that supports **f16**.
   - A member shared a working **f16 example without decomp** [here](https://github.com/wpmed92/stable-diffusion-tinygrad-f16), showcasing performance benefits.
- **Dawn Implements Platform-Specific WebGPU Backends**: **Dawn** implements platform-specific backends for **WebGPU**, not all of which support all features, and lists potential backend options like **D3D12**, **Metal**, and **Vulkan**.
   - On Ubuntu, the **WEBGPU_BACKEND** environment variable controls the used backend, with tests using **Vulkan** as an example ([link to test](https://github.com/tinygrad/tinygrad/blob/7f79c1388ff5b49ac365d7d20d472c36747cb4b6/.github/workflows/test.yml#L617C20-L617C59)).
- **Realtime Diffusion in Browser Possible with LCM?**: Discussion revolves around the feasibility of **realtime diffusion** in browsers, potentially using **LCM finetune** to generate good-looking pictures with models like **dreamshaper-LCM1.5**.
   - A demo video ([link to demo](https://cdn.discordapp.com/attachments/1070745817025106080/1387873720076599447/20250503_085941_x264.mp4?ex=685eeda0&is=685d9c20&hm=a4d3bae14e77d6e036ad55df7ad9973deb9ca607c4a936ec65b108e54997dc15)) shows **LCM** on Torch running up to **20fps** at **1 step**.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenRouter Raises $40M for AI Model Marketplace**: Deedy announced that [OpenRouter](https://x.com/deedydas/status/1937902948920811729), an **AI model marketplace** that offers access to **400+ LLMs** through a single API and processes **100 trillion tokens annually**, secured **$40 million** in funding.
   - The funding round values the company at approximately **$500 million**, drawing congratulations from Dennis Knodt, Yuchen Jin, Emad Mostaque, and John Shedletsky.
- **EU Founder Demands GDPR Compliance from OpenRouter**: A user requested that [OpenRouter](https://x.com/deedydas/status/1937902948920811729) offer **GDPR compliant endpoints** to be used in production.
   - The user jokingly stated *such is life as a EU founder*.
- **BFL Drops Kontext Weights**: BFL released the [weights for Kontext](https://bfl.ai/announcements/flux-1-kontext-dev) and announced it via [X](https://x.com/bfl_ml/status/1938257909726519640).
   - No additional details were mentioned.
- **OpenAI Hooks Deep Research API to Web**: OpenAI introduced the [Deep Research API](https://cookbook.openai.com/examples/deep_research_api/introduction_to_deep_research_api) featuring the **o3-deep-research** and **o4-mini-deep-research models** with MCP and Code Interpreter support and [Webhooks](https://x.com/openaidevs/status/1938286704856863162) for real-time API event notifications.
   - Developers expressed excitement for the **long-awaited webhooks**, and asked about future releases like **GPT-5** or **'Sign In with ChatGPT'**.
- **Google Doppl Dresses Avatars**: Google Labs launched [Doppl](https://x.com/GoogleLabs/status/1938284886277951916), a mobile app for **iOS and Android (US only)**, generating videos of users 'wearing' uploaded outfit photos to help discover their aesthetic.
   - Initial reactions ranged from excitement to difficulty locating the app, requests for an APK, and disappointment over regional restrictions.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Shopify CEO Endorses DSPy in Ruby**: Shopify's CEO voiced his support for implementing **DSPy in Ruby**, suggesting it could become dominant in ecosystems like Shopify, GitHub, and Coinbase, as noted in [this tweet](https://x.com/tobi/status/1937967281599898005).
   - The discussion highlights the potential for DSPy to expand beyond its original implementation, reaching a broader audience through Ruby's extensive use in e-commerce and web development.
- **Desiru Emerges as DSPy's Ruby Implementation**: The community is now talking about **Desiru** ([https://github.com/obie/desiru](https://github.com/obie/desiru)), a Ruby implementation of DSPy, including potential naming conventions for DSPy ports in languages like Ruby, Rust, Go, and Typescript.
   - The proposed convention *DS<file extension of the language>* was suggested to name future DSPy ports, streamlining the identification of DSPy implementations across different languages.
- **Desiru Trailblazes Persistence and Async Processing**: **Desiru** sets itself apart with a persistence layer that saves training examples and results to **Postgres** ([persistence example](https://github.com/obie/desiru/blob/main/examples/persistence_example.rb)) and offers async background processing ([async processing example](https://github.com/obie/desiru/blob/main/examples/async_processing.rb)).
   - Given DSPy maintainers' focus on simplicity, there is discussion on the need to create a community integrations registry or library akin to the LangChain community package, for extensions and community contributions.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **HeroDevs Fund faces Scrutiny**: A member inquired about the applicability of the [HeroDevs Sustainability Fund](https://www.herodevs.com/sustainability-fund) to 'the og ersatz'.
   - It was clarified that *'OG ersatz = ersatz && !gollark'*.
- **Su's Singular Weight Decay**: A member highlighted [Jianlin Su's blog post](https://kexue.fm/archives/10648) on **weight decay**, which decays only the largest singular value, and its relation to **sigmaReparam**.
   - The poster shared that *each step of power iteration only needs to calculate two "matrix-vector" multiplications, and the complexity is O(nm)*.
- **Sequential Order Statistics**: A discussion arose around research on models learning **order statistics** sequentially, referencing the paper [Order Statistics in Transformers](https://arxiv.org/abs/2402.04362).
   - A member emphasized that the observed **frequency bias** is not necessarily inherent or unavoidable.
- **TyDiQA Task Remains Missing**: Members discussed the status of **Codex** and **TyDiQA** tasks, referencing a [GitHub issue](https://github.com/EleutherAI/lm-evaluation-harness/issues/193) in **lm-evaluation-harness**.
   - It remains unclear whether there was a follow-up to the issue.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Zoom Announces Real-Time Meeting Data (RTMS)**: @Zoom announced **RTMS** at the developer summit today, enabling real-time access to data from Zoom Meetings (**video, transcripts**) for application development, with [an example provided](https://t.co/4m2IOcz7Se).
   - This will allow developers to build on top of live meeting data for the first time, which the community has been requesting for a while.
- **LlamaIndex CEO's Talk Hits 50K Views**: CEO @jerryjliu0's talk at the @aiDotEngineer World's Fair, explaining how to go beyond basic RAG to construct extensive document toolboxes with search, manipulation, and structured querying, has reached **50,000 views**, as shown [here](https://t.co/he5JH2ngCU).
   - The audience was primarily interested in the discussion of *tool-based agents*.
- **LlamaIndex Open-Sources Observability Tools**: LlamaIndex now includes a suite of third-party tools, delivering real-time, accurate tracing solutions, and has launched its initial native [open-source observability tool](https://t.co/UPsM8FFGZJ).
   - The launch of Observability will improve tracing and monitoring in production LLM applications, with detailed data capture and visualization.
- **Klavis AI Simplifies AI Agent Authentication**: By leveraging LlamaIndex and @Klavis_AI's MCP servers, you can now construct AI agents capable of connecting to **YouTube**, **Gmail**, and other services with minimal code, thanks to [Klavis AI's MCP integrations](https://t.co/Z8OypKMfHI).
   - These integrations eliminate the necessity for bespoke authentication code and client libraries, dramatically reducing boilerplate code.
- **LlamaIndex Docs Get Auto-Synced**: A member created an automated script for synchronizing the newest **LlamaIndex docs** and generating an updated **llms.txt** file, sharing a [GitHub repository](https://github.com/nmhjklnm/llamaindex-llms.txt) and intending to create a PR for official integration.
   - The goal is to compress all LlamaIndex documentation into **~50k–100k tokens** for efficient utilization by tools such as Cursor and ChatGPT, using entropy-based filtering and index-level summarization.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular Hack Weekend Offers Prizes**: The **Modular Hack Weekend**, scheduled to begin on **June 27th**, offers participants the chance to build with **Mojo and MAX** and win **NVIDIA GPUs**: **5090** for first place, **5080** for second, and **5070** for third; sign up at the [Modular Hack Weekend page](https://lu.ma/modular-hack-weekend).
   - **Lambda** is also providing compute resources via their AI Developer Cloud, offering participants **$400 in credits**; sign up at the [Lambda Labs page](https://lambda.ai/modular-hack-weekend).
- **GPU Programming Workshop**: A **GPU Programming Workshop** is scheduled for **Friday, June 27th**, featuring lightning talks from **Chris Lattner, Chuan Li, Bin Bao, and Jared Roesch**; RSVP at the [GPU Programming Workshop page](https://lu.ma/modular-gpu-workshop).
   - The workshop will be available in person at the Los Altos office and online via LinkedIn and YouTube livestream.
- **InlineArray Move Semantics Raises Concerns**: A user questioned how `InlineArray` avoids element movement during array movement, providing [an example](https://github.com/modular/modular/issues/4911) where neither copy nor move constructors are called, suggesting a potential bitwise copy bug.
   - Another member suggested filing an issue to investigate this behavior, with a link to the [associated Github issue](https://github.com/modular/modular/issues/4911).
- **`VariadicPack.each()` Gets the Axe**: A user reported that the `VariadicPack.each()` method has been removed, necessitating a change in implementation using `range(args.__len__())` as seen in the [github issue](https://github.com/modular/modular/issues/4905).
   - The user expressed that this change made the implementation less elegant, noting its similarity to `std::apply` in C++.
- **TorchScript Reaches Sunset**: **TorchScript support** has been deprecated as part of Modular's **v25.4** release ([changelog](https://docs.modular.com/max/changelog/#v254-2025-06-18)).
   - Modular is offering assistance with **ONNX** issues, directing users to the [forum](https://forum.modular.com/t/onnx-difference-in-max-cpu-gpu-execution/1229/3?u=ehsan) where users can post detailed error messages.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Notebook LM Aiding Customer Discovery**: A user is employing **Notebook LM** to analyze **customer interactions** and pinpoint **pain points**, utilizing the tool to validate or invalidate hypotheses from resources like *The Mom Test*.
   - The surprising effectiveness in **pattern recognition** has raised concerns about over-reliance on **AI** for crucial validation tasks.
- **NotebookLM Expands Language Support!**: Users inquired about new language support in **NotebookLM**, referencing [a tutorial](https://x.com/introsp3ctor/status/1938017086875296083).
   - Currently only English is supported for longer podcast creation.
- **Page Limit Per Source Investigated**: A user questioned if **NotebookLM** fully processes documents exceeding **400 pages**, referencing discussions on [Reddit](https://www.reddit.com/r/notebooklm/comments/1l2aosy/i_now_understand_notebook_llms_limitations_and/).
   - A user clarified that the system processes **400+ page documents**, providing details in [the Reddit thread](https://www.reddit.com/r/notebooklm/comments/1l2aosy/comment/mvyp73k/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button).
- **PDF Format Favored in NotebookLM**: A user reported that the **.pdf** file format yields better results with **Notebook LM**.
   - No reason was given as to why.
- **Chrome Extension Connects Notebook to Gemini**: A user shared a [Google Chrome Extension](https://chromewebstore.google.com/detail/igboaajnodmioloalklomakeeigipnkh?utm_source=item-share-cb) that transfers content from **Notebook** to **Gemini** for generating tables or slides, with [source code on GitHub](https://github.com/MarioDeFelipe/NotebookLM-to-Gemini).
   - There was a user report of hallucination issues when pasting notes with references between notebooks.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **EnrichMCP Hosts Webinar Connecting Agents to Data**: Simba Khadder is set to host a **free webinar** on **July 1st at 8 AM PT** about **EnrichMCP**, which transforms data models into agent-ready MCP servers.
   - The webinar, designed for **Data Scientists** and **ML Engineers**, aims to improve agent data access; registration is available [here](https://buff.ly/XXm8nll).
- **API Emerges for Game-Playing Bot Interaction**: A member advocated for an **API** for a game-playing bot to interact with the game, perhaps via basic **image processing** or accessing the internal game state.
   - Capturing the **game state** and its variables is crucial for the bot to understand and interact with the environment.
- **Git Repositories Prevent SSD Fiascos**: Members reiterated the importance of using **Git repositories** for their projects, as they unfortunately learned this the hard way after a laptop SSD failure.
   - This incident served as a stark reminder of the critical role that version control plays in preserving project integrity and preventing data loss.
- **RL Bot Seeks to Conquer Game**: A member is planning to utilize **Reinforcement Learning (RL)** for a game-playing bot project, also using it as an opportunity to learn RL.
   - This initiative is a dual opportunity to learn RL and create an interesting new project.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Team Launches Quality Agent**: The Manus Team launched **Quality Agent**, designed for complex problems, building on positive feedback from a **high-effort mode beta**.
   - A member expressed enthusiasm about testing the new feature, highlighting its potential impact on handling demanding tasks.
- **Users Report Browser Automation Problems**: A user reported issues with **Manus** failing to press buttons within the browser, specifically on **LinkedIn** and **sam.gov**.
   - The issue prevents users from effectively utilizing filters on those platforms.
- **Comic Actor Alvaro Vitali Passes Away**: A member shared a [Facebook link](https://m.facebook.com/watch/?v=23933493089671823&surface_type=vod&referral_source=vod_newsfeed_unit) reporting the death of comic actor **Alvaro Vitali**.
   - The poster commented that *the Italy comic world remains stopped because of the death*.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Liger CE PR Experiences Limbo**: A question about the **Liger CE PR** was raised, inquiring if it's blocked, needs support, or is on hold due to **PyTorch core's** prioritization of upstreaming a fused linear + CE loss.
   - The question was about the team's desire for broader impact related to **PyTorch core**.
- **Masking Spikes Memory Usage**: After setting `self.mask_ignored_tokens = False`, a member reported a **greater than 20% increase in memory usage** despite only having 5% padding.
   - This increase was considered *odd*, given the masking's limited impact on memory.
- **Iterable Dataset Adds Logging On The Fly**: An iterable dataset with on-the-fly packing and dataset logging was introduced in [this commit](https://github.com/pytorch/torchtune/pull/2819/commits/55be7756e0fd03b493dde46691925825f5cb3948).
   - Configuration details include **packed sequences** and a sequence length of **4096**.
- **Tiled MLP Mirrors Chunked CE Loss**: A member suggested that *tiled MLP* is similar to the existing **chunked cross-entropy loss**, but applied to linear layers.
   - They noted that implementing this might complicate the model code.
- **Sequence Parallelism's Significance Questioned**: A member questioned if **sequence parallelism** offers significant advantages over **tensor parallelism** combined with chunking.
   - They speculate that **Ring Attention** could be a key benefit of sequence parallelism, but this requires confirmation from someone familiar with collective scheduling.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Hugging Face Authentication Activated**: A member confirmed that **Hugging Face authentication** needs to be triggered with [this link](https://hf.co/mcp?login), which is anonymous by default.
   - This ensures users can access resources without compromising their privacy.
- **Reddit Moderators Requested in Channel**: A member inquired about the availability of **Reddit moderators** within the channel, seeking support.
   - The request suggests a need for moderation assistance within the community.
- **PlayMCP Browser Emerges**: A member shared a link to [PlayMCP](https://github.com/jomon003/PlayMCP), a browser-based **MCP** (presumably Minecraft Control Panel) implementation.
   - This offers users a web-based interface for managing their Minecraft servers.
- **Rust Docs MCP Server Fights Hallucinations**: A member announced the creation of a **Rust Docs MCP server** to prevent agent hallucinations when working with new versions of rust projects, and posted the [repo on GitHub](https://github.com/snowmead/rust-docs-mcp).
   - The member encouraged users to open an issue on the repository if they encounter any problems with the server, and aims to solve the problem of **agent hallucination** when dealing with new versions of Rust projects.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Welcomes New Community Members**: The Cohere team welcomed new members and acknowledged OG members in the Discord channel and encouraged exploration of **Cohere Labs** for research and tool updates.
   - Newcomers can visit [cohere.com/research](https://cohere.com/research) and click *Join Us* to share their projects.
- **Cohere's Compass: Navigating Support Channels**: Varun directed users to <#1324436975436038184> for general support, and to <#1168578329423642786> for **API-specific discussions**.
   - This helps to correctly direct users to the correct channel for optimal support.
- **Agentic Apps Assemble in NYC!**: Cohere, AWS, and Pinecone are hosting a hands-on session on building **agentic applications** in NYC on **June 30, from 2:30 – 6:30 PM EDT** ([lu.ma link](https://lu.ma/8rm6zryw)).
   - The event includes mini-talks, an **AWS Workshop Studio** session, a use case on **financial semantic search + reranking**, and networking over dinner; Attendees should bring their laptop and charger, along with a government-issued ID to get past security.
- **Deep Learning and NLP Draw New Cohere Members**: Swakshar, a student from Bangladesh, and Tony Silveti-Falls, a professor in France, introduced themselves and expressed interest in **deep learning**.
   - Swakshar is focused on **NLP** and **Animal Linguistics**, while Tony works on **optimization**, both seeking research collaborations.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All's Qt Requirement Causes Build Catastrophe**: GPT4All's documented **Qt requirement is 6.5+**, but the `CMakeLists.txt` requires **6.7**, while the C++ code uses a `slice` feature only available in **6.8**, causing build errors.
   - The build further fails to find its own Qt modules due to using deprecated imperative singleton registration, conflicting with Qt **6.8's** stricter registration approach; see [Qt Documentation](https://doc.qt.io/qt-6/qml-singleton.html) for details.
- **Microsoft's 1.58B 2B4T Model Compatibility Sparks Debate**: A user inquired about running **Microsoft's 1.58B 2B4T model** with GPT4All, which led to a troubleshooting exchange about what the original user had already tried.
   - The user gave up and switched to trying LM Studio instead.
- **LM Studio Eclipses Outdated GPT4All**: When asked about their attempts with Microsoft's model, a user was advised to use **LM Studio** instead, with the claim that *GPT4All is not up to date*.
   - The user confirmed they are now trying **LM Studio** and thanked the recommender.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Pull Request Opens for Leaderboard inclusion**: A member submitted a pull request for inclusion in the leaderboard and expressed hope for a quick review and merge, assuming everything checks out.
   - The member thanked the team for their work on the project.
- **LLM Evaluation Methodology Questioned**: A member asked about the evaluation methodology for LLMs with thinking mode, such as **Qwen3** in the **Berkeley Function-Calling Leaderboard**, inquiring whether thinking mode was enabled during evaluation.
   - The member sought clarification on the specifics of how these models were assessed.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Codeium (Windsurf) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1387508329110306848)** (1121 messages🔥🔥🔥): 

> `Android vs iPhone, Perplexity AI pricing/plans, Iphone cooling, GPT-5, Doppl app` 


- **Android is cheaper than iPhone**: Members discuss the high price of iPhones, with one saying they prefer *value for money vs placebo optimisation* while [Android is much cheaper](https://www.android.com/intl/en_in/)
   - One member added that iPhone charging is slow and that a **repair/battery replacement would be cheaper** than buying an Android.
- **Perplexity Pro MAX Plan is coming out**: The community discusses the announcement of the [Perplexity AI MAX plan](https://www.perplexity.ai/), with members hoping that the pro plan won't get nerfed, and that new features such as **video generation** may come.
   - One member was furious his subscription was revoked, shouting *So I pray that a company like Perplexity, which goes back on its word, goes bankrupt soon.*
- **Iphone doesn't have vapour chambers!**: Members debate whether or not **iPhones need a vapor chamber** - some say they don't because the software is optimized so that the phones don't melt, while others say that **iPhones heat more** if pushed more because they have no vapor chamber.
   - There was back and forth regarding geekbench scores with one member retorting with *the x200 pro mini with a vapour chamber lagged behindin the geekbench tests you showed*.
- **Google Labs releases Doppl, an AI fashion APP**: [Google Labs released Doppl](https://blog.google/technology/ai/google-labs-doppl-ai-fashion-app/), a mobile app that lets you upload a photo of an outfit and creates a video of you wearing it.
   - A member reacted that the video ad was smooth af.  Another wondered why it was so laggy and others asked how it was generated.
- **xAI's Grok to come out with advanced code editor**: [xAI's Grok is launching an advanced code editor](https://twitter.com/grok_xai/status/1778519744810826000) that uses VS Code and lets users run code inside Grok.
   - You can also talk to it and ask it to modify the code or debug it for you!


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1387620335238779043)** (7 messages): 

> `HSK 2.0 vs 3.0, Ta She, Guan Yu, Killing Games, Deepseek` 


- **HSK Leveling Up with 3.0**: The document discusses the differences between [HSK 2.0 and HSK 3.0](https://www.perplexity.ai/page/hsk-2-0-vs-3-0-sjucyXeuRvWPJqshiq9USg), the standardized Chinese language proficiency test, noting the updated version includes a broader vocabulary and grammar range.
- **Ta She in today's world**: The document discusses the role of [Ta She in the modern world](https://www.perplexity.ai/page/ta-she-in-the-modern-world-ml2TSOiCTLKuZxcZHlLM1w) and its relevance today.
- **Guan Yu's unbreakable spirit**: The document discusses [Guan Yu](https://www.perplexity.ai/page/guan-yu-the-unbreakable-spirit-eM3FLE3MQZKdA5MxMF8ZRg), a historical figure known for his loyalty and righteousness, and his enduring legacy.
- **Game Over for European CI?**: The document talks about a [European CI](https://www.perplexity.ai/page/stop-killing-games-european-ci-L1HoM5KvTBuT.0dpVufiTw) and whether or not games are being killed off.
- **Deepseek's Progress Stalls**: The document discusses [Deepseek's progress](https://www.perplexity.ai/page/deepseeks-progress-stalled-by-ipbek9oEQhe84ClSuYpQ_w) and suggests it is stalled by something.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1387624108472143882)** (3 messages): 

> `Sonar Deep Research Documentation, Credits pending` 


- **Sonar Deep Research Docs Found!**: A member was missing the documentation for `sonar-deep-research` via API, but another member quickly posted the [docs](https://docs.perplexity.ai/models/models/sonar-deep-research).
   - The docs provide a walkthrough of how to use **sonar-deep-research** and its pricing structure.
- **Credits Pending: How Long?**: A member asked about the typical processing time for credits listed as pending.
   - There were no further responses to clarify how long that usually takes.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1387843788885594165)** (1 messages): 

> `OpenAI DevDay 2025, San Francisco Event, Livestreamed Keynote, Hands-on Building, New Models and Tools` 


- **OpenAI DevDay Set for October 2025**: OpenAI announced [DevDay](https://www.devday.openai.com/) scheduled for **October 6, 2025**, in **San Francisco** promising their *biggest one yet*.
- **DevDay to Host 1500+ Developers**: The event is expected to host **over 1500 developers** with a **livestreamed opening keynote**.
- **DevDay to feature Hands-on Model Building**: DevDay will feature **hands-on building** with OpenAI's **latest models and tools**.
   - Attendees can expect **more stages and demos** than previous events.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1387515594051682375)** (943 messages🔥🔥🔥): 

> `NotaGen release, Codex rate limits, BS detector benchmark, Gödel’s incompleteness theorem, Minimax benchmark` 


- **NotaGen Gets Noticed for "Technological Dominance"**: A member shared a link to the new [NotaGen demo](https://electricalexis.github.io/notagen-demo/) and quoted OperatorOAI on many of their things.
   - They pointed out the interesting term used at the end, *"technological dominance"*, which was the main discussion point.
- **Gödel Theorem Inspires BS Detection**: Members discussed creating questions that defy logic to bait LLMs into giving false answers and [linked to Gödel’s incompleteness theorem](https://en.wikipedia.org/wiki/G%C3%B6del%27s_incompleteness_theorems).
   - One member added that LLMs should realize that too many unknowns in an equation lead to variance and a *"what the heck ?"* response instead of a BS answer.
- **Minimax Hides in Plain Sight**: A member shared the [MiniMax benchmark](https://artificialanalysis.ai/models/minimax-m1-80k?models=gpt-4-1%2Co3%2Co4-mini%2Cllama-4-maverick%2Cllama-4-scout%2Cgemini-2-5-pro%2Cgemini-2-5-flash-reasoning%2Cclaude-4-sonnet-thinking%2Cclaude-4-sonnet%2Cmistral-medium-3%2Cdeepseek-r1%2Cdeepseek-v3-0324%2Cgrok-3-mini-reasoning%2Cnova-premier%2Cminimax-m1-80k%2Cllama-3-1-nemotron-ultra-253b-v1-reasoning%2Cqwen3-235b-a22b-instruct-reasoning%2Cgpt-4o#intelligence-vs-price), describing it as *"hiding in the most attractive intelligence vs price quadrant"*.
- **Hallucination Isn't Always Bad**: A member shared an image noting that hallucinations in language models, while often dismissed, can be a central part of reasoning and [referenced Marie Curie](https://drinkoblog.weebly.com/) stating that hallucinations are just imaginations that you dont agree with.
   - The discussion revolved around whether imagination is intentional, while hallucination is not, and the philosophical implications of imagination in reasoning.
- **Debate Arises on OpenAI Safety vs Legal Liability**: Members discussed the role of AI safety, debating whether it’s truly about safety or just a cover for legal liability, and shared the [video](https://youtu.be/qv_MTTam1uY?si=2mBVfO4b502TVhCh) that touches on accountability, the legal system, and mental health issues that arise with AI.
   - The question was posed: *"If ChatGPT makes something possible that could not happen before, is it not just passive, but part of the reason the outcome happened?"*, shifting responsibility and accountability.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1387519109532487730)** (3 messages): 

> `ChatGPT business plan PDF issues, AI Proper Usage Learning` 


- **ChatGPT can't send business plan PDFs**: A member reports **ChatGPT** isn't able to send the **PDF** it created containing their business plan.
   - Another user suggested simply *copy and pasting* the content instead.
- **Proper AI usage sought**: A member is seeking guidance on how to *use AI properly*.
   - They are learning how to use it, and ask for pointers on getting assistance.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1387721912083550339)** (32 messages🔥): 

> `3D to 2D texture conversion, Tileable textures, Kaleidoscopic reflection, Python for seamless tiling` 


- **3D Image Texture Transforms to 2D with AI**: A member was looking to test a prompt that turns a **3D image's texture** into a **flat 2D tileable texture**.
   - Another member noted that **kaleidoscopic reflection** is cool, but you don't need AI for that and that [it's a really basic python trick](https://chatgpt.com/share/685d2e5b-b600-8000-9600-de556e53b1b3) to create seamless tiled images.
- **Seamless Tiling via Python Explored**: A member described how the **kaleidoscope method** also works for **non tileable textures**.
   - Another member clarified that you don't reflect textures that don't have to tile, but included [another link](https://chatgpt.com/share/685d2e5b-b600-8000-9600-de556e53b1b3) that explains how it works.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1387721912083550339)** (32 messages🔥): 

> `3D to 2D texture, Kaleidoscopic reflection, Tileable textures with Python, Non-tileable textures` 


- **Transforming 3D Images into Tileable 2D Textures**: A member was experimenting with converting **3D image textures** into **flat, tileable 2D textures** using prompts.
   - However, another member suggested exploring **kaleidoscopic reflection** as a basic Python trick for creating seamless tiled images from any image, regardless of its origin.
- **Kaleidoscopic Reflection creates tiled images**: A member stated that **kaleidoscopic reflection** is a basic Python trick to create seamless tiled images and shared a [ChatGPT link](https://chatgpt.com/share/685d2e5b-b600-8000-9600-de556e53b1b3) explaining how it works.
   - The member clarified that it's a common and easy technique for creating textures, also applicable for **90° transformations**.
- **Techniques for Non-Tileable Textures**: A member inquired about methods for creating **non-tileable textures**, especially from **3D images**.
   - It was clarified that reflecting textures that don't have to tile is not a common practice, which may have been a misunderstanding of the original request.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1387508708279324843)** (590 messages🔥🔥🔥): 

> `Local LLM Security, Tool Creation with Claude, Copilot vs Cline, Unsloth and Gemma, GGUF Conversion` 


- **Code Privacy Quandaries for Local LLMs**: Members debated whether to use **local LLMs** versus big providers due to concerns that code and research could end up in training data or be sold, noting that [small providers may not have the same need or incentive to train models](https://smallprovider.example).
   - While acknowledging some risk, one member stated they risk using smaller providers *based on probability*.
- **Claude cracks code creation conundrum**: One member praised **Claude** for writing tools they *don't even want to look at the code for*, specifically highlighting its effectiveness for one-shot tasks such as distilling datasets.
   - They had *vibe coded* a tool to distill a dataset from R1 from a Q/A SFT one and im just vibe coding it thru*, and had a dataset viewer that saves them from scrolling through thousands of lines in notepad++, all from Claude.
- **Copilot contention creates coding competition**: Members compared **GitHub Copilot** with tools like **Cline** and **Roo**, with one user finding Copilot's native VSCode integration superior despite a cluttered UI.
   - Another member mentioned they *like roo workflow and idea but cline works so much better* and there's also the new warp thing but now i’m like damn warp prob takes all the data from terminal.
- **Gemma gem gets going at Google Event**: An upcoming **Gemma & Unsloth event** was announced, but quickly filled up.
   - After release, some users reported that **Gemma-3n-E2B** was not initially working, which was acknowledged as an issue, and then resolved, and has links to the [announcement on X](https://x.com/danielhanchen/status/1937995188343083239).
- **GGUF generation gets going**: Members discussed converting **FLUX models to GGUF format**, sharing [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF/tree/main/tools).
   - One user offered to help convert a specific model, **FLUX.1-Kontext-dev**, and another shared a [Medium article on the topic](https://medium.com/@yushantripleseven/convert-flux-models-to-gguf-6a80f6c7377a).


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1387670355677679729)** (4 messages): 

> `Electrolyte Labs Job Postings, AI-Generated Introductory Video, Open-Source Model Inquiry` 


- **Electrolyte Labs Assembles AI Dream Team**: **Electrolyte Labs** is seeking **Researchers**, **ML Engineers**, and **Research Scientists** to build models and contribute to open-source AI tools, requiring **5 years** of experience and familiarity with **Hugging Face**, **PyTorch**, or **TensorFlow**.
   - The company seeks individuals excited about hands-on research, model building, and transparent collaboration and is asking candidates to directly DM them with resumes and project links.
- **Electrolyte Labs uses AI to Showcase Itself**: **Electrolyte Labs** created an AI-generated video [available on Vimeo](https://vimeo.com/1096475118/01621e79b4?share=copy) using their own AI model for a *“better & friendly introduction.”*
- **Electrolyte Labs Model Accessibility Questioned**: A member inquired whether **Electrolyte Labs'** AI model used to generate their introductory video is open source.
   - Electrolyte Labs did not confirm or deny the question of open source.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1387515139623878716)** (155 messages🔥🔥): 

> `Gemma3 .gguf saving issues, LLM output issues, Unsloth Mistral Small 3.2 quants, Qwen3 image vision, SSML finetunned models` 


- **Gemma3 can't be saved in gguf format**: A user reported issues saving a fine-tuned **Gemma3** model to **.gguf** format in Google Colab, despite following example notebooks.
   - A member suggested saving the **LoRA adapters** and pushing them to Hugging Face, then downloading and merging them on a larger machine for manual conversion to **gguf**.
- **LLM doesn't stop talking**: A user experienced issues with a fine-tuned **LLM** (Llama 3) not stopping its responses when using the Llama 3 chat template with the Ollama notebook.
   - A member suggested checking if the instruct model is being used and loaded correctly with the right template, emphasizing the importance of starting with provided notebooks before customization and recommended experimenting with a smaller model like **3B** for faster testing.
- **Clarification on Unsloth Mistral Small 3.2 Quants Naming**: A user inquired about the naming convention for **Unsloth Mistral Small 3.2 quants**, noting the Q4 XL version is tagged `UD-Q4_K_XL` while the M version is `Q4_K_M`.
   - A member clarified that the M version is not dynamic, but all versions use their calibration dataset, and that the `UD` tag is to specify `Unsloth Dynamic Quantization`.
- **Qwen3 vision model capabilities after finetuning discussed**: A user asked if fine-tuning **Qwen 3 14b** would enable image vision capabilities even without specific vision training.
   - It was clarified that **Qwen 3** is not a vision model, implying it would not gain image vision capabilities through fine-tuning alone.
- **Looking for finetunned SSML output models**: A user is seeking models fine-tuned for **SSML output**, aiming to convert text to output with **SSML tags**.
   - A member expressed doubt about the existence of Unsloth dynamic quants for large models like **70b**, especially for older architectures like **Llama 2**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1387899682633547899)** (1 messages): 

> `` 


- **Manus Momentarily MIA**: The user mentioned *"manus free"*
   - No details about what *Manus* refers to or context were mentioned.
- **Orphaned Attachment**: The user attached a file, but gave no details about its contents.
   - The URL directs to Discord's CDN, however no further data about the image can be discerned.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1387857724418691205)** (2 messages): 

> `YouTube video, arXiv paper` 


- **YouTube video link shared**: A member shared a [YouTube video](https://www.youtube.com/watch?v=dYHkj5UlJ_E).
- **arXiv paper link shared**: A member shared an [arXiv paper](https://arxiv.org/abs/2505.05522).


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1387509849075486821)** (524 messages🔥🔥🔥): 

> `Gemini CLI, Claude Code, Rate Limits, Cursor Pricing, MCP Errors` 


- **Gemini CLI's Rocky Start**: Users report that the new **Gemini CLI** is buggy and not ready for use, with issues including hanging on `npm run dev`, refusing interactive terminal tools, and failing to use specified UI libraries, despite its generous free plan of **1000 requests a day**.
   - One user noted it *freezes everytime it runs `npm run dev`*, others pointed to it *refusing to use any terminal tools that expect interactive input*. 
- **Cursor's Rate Limits spark Unlimited Plan Debate**: Users are experiencing rate limits on the **Unlimited Pro plan**, leading to confusion as *unlimited* only applies to completions and tabs, while model usage is still limited, especially with **Claude 4 Opus**.
   - Some members believe Cursor isn't being forthcoming, with some stating that *they don't want to tell you the limits in detail cause it works against them*.
- **Warp Terminal Rebrands as Agentic Development Environment**: **Warp terminal** released **2.0** and rebranded to **ADE (Agentic Development Environment)**, claiming a mixed model benchmark higher than **Claude Opus 4** at **52.0%** vs **43.2%**.
   - A member shared a link to the [Warp announcement](https://www.warp.dev/blog/reimagining-coding-agentic-development-environment) saying it *sounds promising*.
- **Cursor's Python Extension Confusion Clarified**: Users are confused about whether to use **ms-python** or **Anysphere's Python extension**, with it being clarified that **Anysphere's** is a replacement for **Pylance**, while **ms-python** is still needed.
   - It was pointed out that *Cursor is based on VSCode version from November, and some extensions simply refuse to work with outdated version* and to *check the latest announcement on the forum, recently Cursor started using different extensions source*.
- **MCP Connection Closed, leaving members in Despair**: Members are reporting having problems with MCPs, they are often getting `{"error": "MCP error -32000 Connection closed"}`.
   - Some users share that they got their MCP to work finally, they also shared the link to their Github [smithery-ai/github](https://smithery.ai/server/@smithery-ai/github).


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1387508809303457793)** (47 messages🔥): 

> `Background Agent Connection Errors, Background Agent Network Security, Python 3.11 Setup, Environment.json schema URL, Background Agent Token Limits` 


- **Background Agent Refuses to Connect After Restart**: A user encountered a connection error after restarting their computer and discovered their **WiFi network** was blocking the background agent, despite allowing other Cursor network connections.
   - They inquired about improving the background agent's security to prevent their **WiFi** from blocking it.
- **Background Agent Forgets PATH**: Users reported issues with the background agent forgetting the **PATH** after creating a snapshot, even after adding it to the **PATH** in the setup and validating it.
   - One user suggested a workaround involving adding the **PATH** to the **.bashrc** file in the install command of the **environment.json** file.
- **Environment.json Schema 404**: A user reported that the **environment.json schema URL** referenced in the [Cursor documentation](https://docs.cursor.com/background-agent) is returning a **404 error**.
   - The URL in question is [https://www.cursor.com/schemas/environment.schema.json](https://www.cursor.com/schemas/environment.schema.json).
- **Github CLI auth token available in background agent**: Members are requesting easier way to use **Github CLI** with the auth token that the Cursor Background Agent already creates, perhaps through a script to commit.
   - One user also reported that the Cursor agent committed a `PR_DESCRIPTION.md` file into the branch/Pull Request, instead of updating the Pull Request description itself.
- **Terminals Remain Obscure to Background Agent**: A user reported that background agents don't seem to be aware of the terminals created with the 'terminals' section of environment.json, or be able to access them in a way that makes sense.
   - A dev confirmed making the agent more aware of the terminals is on their roadmap.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1387511689204727950)** (226 messages🔥🔥): 

> `Downgrading LM Studio, Cybersecurity LLMs, LM Studio Context Length Limits, Local LLM Hosting for Friends, LM Studio MCP setup` 


- **Users want downgrade option after censorship increase**: Users are seeking ways to downgrade to earlier versions of **LM Studio** due to perceived increased censorship in version **0.3.17**, with some models performing worse after the update.
   - One user suggested ensuring that system prompts are not being overridden by presets, but the user stated they have an empty system prompt.
- **Cybersecurity LLM suggestions requested**: A user requested recommendations for the **best LLMs for cybersecurity**, seeking models with specific unique selling points (USPs) and benefits, after finding online answers from Gemini and GPT unreliable.
   - The user also apologized for *irregularity in articulating myself atm, my sincerest apologies and thank you for your reply*.
- **Users hit LM Studio context limits**: A user asked about the memory requirements for processing **300k tokens**, planning to upgrade to **128GB of RAM** with a **5090 GPU** to handle large text translations.
   - Suggestions were made to chunk the text into smaller segments and use models like **Deepseek (131k tokens)** or **Llama 4 Scout (up to 10 million tokens)** to avoid performance issues and formatting problems.
- **Hosting LM Studio for Friends**: Users discussed the possibility of hosting LLMs on a local network using **LM Studio** to allow friends with less powerful machines (e.g., laptop with 16GB RAM and an old 4GB GTX card) to access and use the models.
   - The solution involves enabling the server in LM Studio's **developer tab** with the *serve on network* option and using an **OpenAI-compatible API**.
- **LM Studio MCP Setup Guide Needed**: Users requested a guide on setting up **LM Studio** with a locally running **MCP (Model Context Protocol) server**, particularly for enabling web search functionality.
   - Links were provided to the [LM Studio documentation on MCP](https://lmstudio.ai/docs/app/plugins/mcp) and discussions in the **/r/mcp** subreddit, with a reminder to share any existing experiences with MCP tools.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1387528837939724352)** (97 messages🔥🔥): 

> `GPU zip tie mounting, DDR5 Memory Temp Reporting, Deepseek 671B vs 70B Model Speed, Motherboards bolted to wooden boards, Open bench PC fire safety` 


- **GPU Hung by Zip Ties**: A user shared their unconventional setup with a **GPU** hanging by a **zip tie** *outside* of the case, to avoid placing it on the floor.
   - Others joked about the setup being stereotypically 'Murican and questioned the airflow and dust concerns, while the user defended their approach for its practicality.
- **DDR5 Modules Report Temps**: Users noticed that **DDR5 memory modules** now commonly report temperatures, possibly due to the added on-board controller for voltage and power regulation.
   - One user with an **M3 Ultra** observed performance differences between a **Deepseek 671B** model (**20t/sec**) and a **70B** model (**15t/sec**), attributing it to the number of active parameters.
- **Wooden Boards as Motherboard Mounts**: A user reminisced about screwing **motherboards to wooden boards**, to which another user replied that some might even call them a **bread board**.
   - Another user justified open-air setups to prevent accidental kicking and argued that motherboards collect less dust when bolted to a wall.
- **Concerns about Fire Safety and Open Bench PC**: A user expressed concerns about the higher fire risk and safety issues of open bench PCs, especially with pets around, due to the lack of a protective case.
   - Others argued that components are unlikely to catch fire if airflow is adequate and mentioned the potential issues with **VRM** and **NVMe SSD** temperatures in such setups.
- **SSD Heat Generation Discussion**: Users discussed whether **SSDs** get hot with normal use, with one user saying yes, and it was mentioned that higher **PCIe versions** and more write operations generate more heat.
   - Some users recommended using heatsinks for **NVMe SSDs**, especially with **PCIe 4.0** and **5.0 controllers**, which can reach temperatures of **70-80 degrees Celsius**.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1387512853371813898)** (28 messages🔥): 

> `GCC as build system, Bazel, CMake` 


- ****GCC Isn't A Build System****: Members discussed whether **GCC** could be considered a build system, with the consensus being that it functions as a compiler, but not a comprehensive build system for projects with multiple source files or dependencies, and for complex projects tools like **CMake** and **Bazel** are recommended.
   - One member pointed out that *GCC is a compiler (one tool used in the build process)* but *for anything with multiple source files, it's no longer a build system*.
- ****Bazel: the goat****: A member proclaimed **Bazel** the best build system, while others jokingly asserted a *lion doesn't concern himself with not working build system*.
   - Another sent a [tenor.com embed](https://tenor.com/view/head-lion-afrika-savanne-gif-13123479987477323100) referencing the comment.
- ****CMake Has Incompatible Versions****: Members debated problems with **CMake**, with one claiming it *has 3000 incompatible versions*, and a link was shared to [HigherOrderCO/Bend](https://github.com/HigherOrderCO/Bend) as a candidate project.
   - Another replied that the incompatibility only goes in one direction, so *you should always use a recent version*.
- ****PyTorch messes with global compiler options****: Members pointed out that a problem with **CMake** is if *you want to use packages that don't use CMake correctly* like **PyTorch**, since it does *nasty stuff like messing with global compiler options instead of doing things only project-locally*.
   - One member quipped that *it's not the tool but the people using it wrong*.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1387620920306438206)** (7 messages): 

> `Triton Community Meetup, LinearLayout usage change, Gluon update, Nightly performance regression suite, Triton developer's summit update` 


- ****Triton Community Meetup** Rescheduled to July 9th**: The **Triton Community Meetup** has been moved from **July 2nd** to **July 9th**, **10am-11am PDT**; the meeting link is provided, and agenda suggestions are welcomed.
   - Tentative topics include: **Gluon update**, interest in a **nightly performance regression suite**, and a **Triton developer's summit update**.
- **Admins promise regular Triton community meetup updates**: After feedback, admins will now send regular notifications about **Triton Community Meetups** to reduce friction.
   - The admins will begin posting **invites to the Triton Community Meetups** here going forward.
- **LinearLayout discussion requested for next Meetup**: A member expressed interest in a session on the change to using **LinearLayout** in a future meetup.
   - Another member noted that it happened a while back but shared a link to [the paper](https://arxiv.org/abs/2505.23819) on the technical details.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1387819651718844518)** (5 messages): 

> `CUDA barrier parity parameter, ABA problem in circular buffers, Tensor Cores Usage` 


- **Parity Parameter in CUDA's Barrier Family Emerges**: The optional *parity* parameter in CUDA's barrier instructions addresses the **ABA problem** in circular buffers, distinguishing between finished and unfinished barriers.
   - Instead of resetting, the system waits alternatively for **0->1->0->1**, with [PTX documentation](link) providing further details on `mbarrier`.
- **Tensor Core Conundrum Clarified**: While using **tensor cores** with CUDA is possible, direct C code compilation to tensor core instructions (**WGMMA**) via *nvcc* isn't supported.
   - The accepted method involves writing inline **PTX assembly** or employing libraries like [CUTLASS](link) that construct these inline assembly calls.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1387734242288730123)** (1 messages): 

> `CUDA graphs blocking execution, SGL updates, Kernel execution` 


- **CUDA Graphs get blocked after SGL Updates**: A user reported that their CUDA graph calls started blocking their thread until the last kernel in the graph finishes after **SGL updates**.
   - The user is seeking a way to check why the CUDA graphs are blocking execution, noting that both cases should run the same CUDA graph.
- **Seeking insights on CUDA Graph Blocking**: A user is investigating why their **CUDA graphs** are blocking execution after recent **SGL updates**.
   - The issue arises even though the same CUDA graph is used in both scenarios, leading to questions about the cause of the blocking behavior.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1387559649959477309)** (28 messages🔥): 

> `HIP on Debian Nvidia, Building HIP from source, ROCm Clang necessity, HIP cross-platform support` 


- **User Struggles to Install HIP on Debian with Nvidia GPU**: A user seeks to install **HIP** on a **Debian Nvidia machine**, following [the official installation guide](https://rocm.docs.amd.com/projects/hip-python/en/latest/user_guide/0_install.html) but encounters issues due to the lack of specific instructions for Debian.
   - The user attempts to build from source using a provided script but faces a **CMake error** related to **HIPCC_BIN_DIR**, despite providing the required path.
- **Debian Package Repository for ROCm and Alternatives Explored**: It's mentioned that the **Ubuntu package repository** is also supported on Debian, referencing [ROCm's documentation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/install-methods/package-manager/package-manager-debian.html) for Debian installation notes.
   - Another user suggests exploring "unified" build repos like [TheRock](https://github.com/ROCm/TheRock) as an alternative.
- **Debate on ROCm Clang Requirement for HIP Nvidia Compilation**: A user questions the necessity of **ROCm Clang** when compiling **HIP** for Nvidia, assuming it's not needed in this specific cross-platform compilation scenario.
   - A more experienced member notes that one probably doesn't need **clr**, but rather only **hipother** with **hipcub** and **hiprand**, while also cautioning that **hipother** is *"pretty bad and likely broken"*.
- **Deteriorating Cross-Platform Dreams for HIP**: After struggling to install, the user shares that they actually want to add **HIP support to their code analyzer** as a feature, and thought the install would be easy.
   - Another member suggests that running **HIP** code requires an **AMD GPU**, while the user laments the difficulty in achieving **HIP's cross-platform promise**.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1387854313573060781)** (1 messages): 

> `CuTeDSL, SGEMM, Ampere architecture` 


- ****CuTeDSL** blogpost dissects **SGEMM** for **Ampere****: A blogpost analyzes the **SGEMM** example in **CuTeDSL** for the **Ampere** architecture, using a top-down approach for clarity, see the [blogpost here](https://veitner.bearblog.dev/sgemm-in-cutedsl/).
- ****CUTLASS** repo example deep-dived**: The blogpost references a relevant example from the **CUTLASS** repository, specifically the [sgemm.py file](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/sgemm.py).


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1387543969583595671)** (6 messages): 

> `Mirage compiler, GPU kernels, Kernel generation using LLMs, Benchmarking tools` 


- **Mirage Auto-Generates Fast GPU Kernels**: The [Mirage project](https://share.google/41nz6vDcGvu45uUIc) automatically generates fast **GPU kernels** without programming in **Triton/CUDA**.
   - A member invited the authors to give a talk on the server on September 13.
- **LLMs aid Kernel Generation**: There was some discussion around [Mirage project](https://share.google/41nz6vDcGvu45uUIc) and using **LLMs** for **kernel generation**.
   - One member congratulated the launch.
- **Benchmarking Triton Kernels**: A member inquired about speedup tests/code used to assess performance gains, particularly for **GRPO rewards**.
   - Another member shared a simple fast sandbox server with basic benchmarking ([GitHub repo](https://github.com/tcapelle/triton_eval/tree/main/sandbox)) and wondered if more should be done.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1387903090107355167)** (1 messages): 

> `FP32 vs Tensor Cores, Nvidia hardware optimization` 


- **FP32 choice avoids Tensor Cores?**: A member inquired whether the choice of **FP32** was intentional to avoid the use of **tensor cores** on **Nvidia** hardware.
   - No further discussion or clarification was provided.
- **Tensor Core Utilization Discussion**: The inquiry sparked a discussion on whether utilizing tensor cores would be more efficient for certain operations.
   - Several members debated the trade-offs between FP32 precision and tensor core acceleration in specific deep learning tasks.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1387531080118046815)** (72 messages🔥🔥): 

> `vectorsum leaderboard, vectoradd leaderboard, sort leaderboard, trimul leaderboard, H100 performance` 


- **H100 Heats Up with vectorsum's New 2nd Place**: A user's vectorsum submission hit **second place on H100** with a time of **91.5 µs**.
   - This highlights the competitive landscape and potential for further optimization on the **H100**.
- **vectorsum finds Victory on T4 and A100**: A user secured **3rd place on T4** with **806 µs** and **2nd place on A100** with **151 µs** on the vectorsum leaderboard.
   - This underscores the varied performance of vectorsum across different GPU architectures.
- **vectoradd's H100 Hotshot Takes the Top Spot**: A user claimed **first place on H100** for vectoradd with an impressive time of **178 µs**.
   - This result demonstrates the potential for highly optimized vector addition on the **H100** platform.
- **trimul Triumphs! Claims 1st Place on MI300**: A user achieved **first place on MI300** with a time of **9.50 ms** on the trimul leaderboard.
   - This showcases the computational capabilities of the **MI300** for trimul operations.
- **L4 sees Lots of vectorsum Submissions**: Numerous successful submissions were made on the **L4** for vectorsum, with times hovering around **970-1000 µs**.
   - These consistent results indicate a stable performance baseline for vectorsum on the **L4**.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1387526321546203287)** (81 messages🔥🔥): 

> `LuaSurface vs LuaPlayer, Mining Drill API Comparison, Test Environment Issues, Manual Data Collection, Teleportation` 


- **`LuaSurface` now a `LuaPlayer` Drop-in with Blueprint Ghosts**: A script was written to compare behavior, showing `LuaSurface.can_place_entity` is a drop-in replacement for `LuaPlayer`'s by using `build_check_type = blueprint_ghost | manual`.
   - However, one member noted that *build_check_type.manual works and blueprint_ghost doesnt*.
- **Mining Drill API Yields Compatibility Quandaries**: An analysis reveals varied compatibility between `LuaSurface` and `LuaPlayer` for mining drills, with `build_check_type='manual'` achieving **100%** match rate.
   - Results for `build_check_type='manual'` shows perfect placement agreement (P:✓ S:✓ ✓) between Player (P) and Surface (S) in ore-rich tiles, while both prevent placement in ore-lacking areas.
- **Test Environment's Server State Cleanliness Questioned**: Tests reveal issues of Factorio RCON connections, potentially stemming from incomplete environment resets between tests.
   - Isolated tests pass, while running all at once results in multiple RCON connections to the same port, creating authentication errors.
- **Short Circuiting the LLM: a Quest for Manual Data**: There is a discussion on methods of creating custom training data without relying on the LLM due to the cost of API calls.
   - The team explored options that would allow the game to be loaded midway, as well as being able to trigger the same output that the LLM would get during normal gameplay.
- **Teleportation Justifies Lenient `can_place_entity`**: The team concluded that because the character is able to teleport, they were able to merge in [PR #223](https://github.com/JackHopkins/factorio-learning-environment/pull/223), despite not being completely accurate due to teleportation mechanics.
   - Even if `can_place_entity` has issues with how accurate it is, it would not matter because the teleportation mechanics do not punish the player for being slightly off.


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1387612747562356756)** (1 messages): 

> `GPU Hackathon, GPU Programming Workshop` 


- ****GPU Programming Workshop** Happening Soon!**: There's a **GPU Programming Workshop** happening this weekend, and the GPU MODE folks are invited, at [this link](https://lu.ma/modular-gpu-workshop).
   - It's a great opportunity to learn more about **GPU programming** and meet other enthusiasts.
- **GPU Hackathon**: A **GPU hackathon** is happening this weekend, and the GPU MODE community is encouraged to participate, check it out [here](https://lu.ma/modular-hack-weekend).
   - Folks are welcome to join and showcase their **GPU programming** skills, or even learn for the first time!


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1387507935311036417)** (211 messages🔥🔥): 

> `GEM CLI first impressions, Copyright impact on AI training, Undepressing Gemini, Adding a changelog channel, GPT-5 Release Prediction` 


- **Gemini CLI gets mixed reviews**: Users reported getting *stuck* with the new **Gemini CLI**, indicating a need for improvements, as demonstrated in a provided [image](https://cdn.discordapp.com/attachments/1340554757827461211/1387518364980871268/image.png?ex=685ef42d&is=685da2ad&hm=bbc8657d910755f0b76c406d659f434ba3397882179a8d53668f989566057323&).
- **Copyright Woes not impacting AI Training**: Despite court rulings on **copyrighted material**, AI models continue to train on it, with companies finding ways to navigate permissions or waiting for court rulings after models are in production as the [image](https://cdn.discordapp.com/attachments/1340554757827461211/1387524888658579656/Laauo9s.png?ex=685efa40&is=685da8c0&hm=426f8e772536a09c9467bae9860f561755cfebee85ab7607eb0fab70a0d496e5&) shows.
- **Motivating the Melancholic Gemini**: Users humorously discussed ways to "undepress" **Gemini**, with one suggesting sending a ":(" to the model, resulting in a minute of processing before hitting a rate limit, illustrated in attached [image](https://cdn.discordapp.com/attachments/1340554757827461211/1387613960571977849/image.png?ex=685ea474&is=685d52f4&hm=cb65acb2971856192c489f3f2cdf618bbb09d726dcd78c6344c7461728457cfc&).
- **ByteDance's Seed Model Sparking Interest**: Users are exploring **ByteDance's Seed 1.6 Thinking** model, despite the Chinese-only website, noting it's on par with open-source SOTA and potentially better in tool use as shown in linked [website](https://www.volcengine.com/experience/ark?model=doubao-seed-1-6-thinking-250615).
- **GPT-5 release before December**: Polymarket data suggests a **90%** chance of an open-source model release and a **90%** chance for **GPT-5** before year's end, with some users anticipating potential delays, based on the posted [screenshot](https://cdn.discordapp.com/attachments/1340554757827461211/1387707303028981931/Screenshot_20250626-091302.png?ex=685efb63&is=685da9e3&hm=8acc50466e110822c78cd20d1cbf5e81193ba13b05d4d0705a030db7fcb7e344&).


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1387529135428993166)** (3 messages): 

> `Database Downtime, Frontend Authentication Outage, Presets Launch, LLM Configuration Management` 


- **OpenRouter Suffers Brief Database Hiccup**: OpenRouter experienced approximately **30 seconds** of unexpected database downtime at **4:10pm ET** due to an **SSL config change**, potentially causing intermittent **401 errors** for some users.
   - The issue has since been resolved, and the team apologized for any inconvenience.
- **Clerk Authentication Faces an Outage**: OpenRouter's frontend authentication provider, [Clerk](https://status.clerk.com/), experienced an outage, though the **API remained functional**.
   - The outage was resolved by **12:00AM PT**.
- **Presets Set to Revolutionize LLM Configuration**: OpenRouter launched **Presets**, a feature enabling users to manage **LLM configurations** such as model settings, system prompts, and routing rules directly from the OpenRouter dashboard, facilitating rapid, code-free iteration; see the [documentation](https://openrouter.ai/docs/features/presets).
   - Presets offer **centralized control**, allowing users to define model selection and generation parameters in one place, ensuring consistency across organizations.
- **Unlock LLM Configurations via API Calls**: The new **Preset** feature allows you to manage LLM configurations, system prompts, and routing rules directly from the OpenRouter dashboard.
   - To use Presets in API calls you can reference the preset directly as a model with `"model": "@preset/your-preset-slug"`, combined with a model override with `"model": "google/gemini-2.0-flash-001@preset/your-preset-slug"`, or using the new `preset` field with `"preset": "your-preset-slug"`.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1387522610983866500)** (199 messages🔥🔥): 

> `OpenRouter raise, Gemini roasting, Clerk outage, Free Mistral version` 


- ****OpenRouter** Scores a Whopping **$40M Raise**!**: OpenRouter announced a **$40M raise**, garnering congratulations and excitement from the community, with one member humorously noting a **Karpathy** namedrop in the announcement.  Check out the [LinkedIn post](https://www.linkedin.com/feed/update/activity:7343733804110385154) about the raise.
   - Several members suggested migrating off of **Clerk** authentication due to its outage on the same day as the announcement.
- ****Gemini** gets Savage with Coping Arguments**: **Gemini** is *really really good at roasting you*, and one user shared the AI's retort: *This is the most potent cope you've deployed yet.*
- ****Clerk Outage** Causes Chaos for OpenRouter Users**: **OpenRouter** experienced downtime due to a **Clerk** outage, leading to widespread issues, and a user search shows many tweets about the [Clerk outage](https://x.com/search?q=clerkdev&src=typed_query&f=live).
   - API access remained functional despite the frontend being affected, with some users suggesting migrating away from **Clerk**.
- **Free **Mistral** API Calls Yield 404 Error**: Users encountered **404 Not Found** errors when attempting to use the free **Mistral** version with a message *No allowed providers are available for the selected model*.
   - It was discovered that enabling the **Paid Models training** setting resolved the issue, even for free models.
- **Deep Dive into the **Deep Research API** Pricing**: OpenAI's **o3-deep-research-2025-06-26** and **o4-mini-deep-research** models are now available, with the former priced at **$10/$40**, but only via the [Responses API](https://platform.openai.com/docs/models/o3-deep-research).
   - Members noted that it burns through tokens and charges **$10/1K searches**, calling the web search tool call very expensive.


  

---


### **OpenRouter (Alex Atallah) ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - New Models**
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1387507870198665247)** (101 messages🔥🔥): 

> `Llama 3.1 8B, Macbook M1/M2 Performance, Groq LPU, AI Agents, Model Context Protocol` 


- **Llama 3.1 Enters the Stage**: Members discussed the potential usage and performance of **Llama 3.1 8B** on different hardware setups, including **MacBook Air** with **MPS** or **CPU**.
   - One member highlighted the cost-effectiveness of **Groq's LPU** for inference, noting that it costs *$20 million tokens per $1*. 
- **Macbooks are pretty efficient**: Members discussed the prospect of running Large Language Models on **Macbooks** and **Mac Minis**.
   - One member mentioned watching a video of *running huge llm models on $10k mac machines*, whereas others pointed out the practicality of using cloud accounts. 
- **Native 1-bit LLM gets HuggingFace Space**: A member shared a [Hugging Face Space](https://huggingface.co/spaces/Tonic/Native_1-bit_LLM) for a **Native 1-bit LLM**, providing a docker command for local deployment.
   - The member also lauded the price of hosting small models on **Hugging Face** as *completely free*, and the other members reacted to the pricing of HF in general.
- **Groq and Hugging Face Join Forces**: A member announced the collaboration between **Groq** and **Hugging Face** via a [blog post](https://huggingface.co/blog/inference-providers-groq).
   - A memebr asked *Why would I use hugging face for that*.
- **Is it an AI agent or just a pipeline?**: Members discussed the development of **AI agents** for specific tasks like **web scraping** and **API creation**.
   - One member asked about implementing a **Model Context Protocol** to give better context to the **LLM**, calling it *like an API on steroids*, while another asked about its purpose being *just a function API to connect to tools*.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1387823320254382091)** (4 messages): 

> `RAG resources, Time Titans paper` 


- **Request for RAG Resources**: A member asked for suggestions on excellent resources to study **RAG (Retrieval-Augmented Generation)** in detail, including code implementation.
   - They expressed interest in learning more about the topic.
- **Time Titans Paper Praised**: A member cited *Test of Time Titans* as one of their favorite written papers.
   - They praised the paper for being super dense but also applicable to many other fields of **data science and game theory**.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1387805300832075818)** (1 messages): 

> `SAM, Segment Anything, Model Import` 


- ****SAM** gets deployed by **Jozu****: A blog post titled [From Hugging Face to Production: Deploying Segment Anything (SAM) with Jozu's Model Import Feature](https://dev.to/jozu/from-hugging-face-to-production-deploying-segment-anything-sam-with-jozus-model-import-feature-5hcf) was shared.
   - It details how to deploy the **Segment Anything Model (SAM)** using **Jozu's model import feature**.
- ****Jozu** Enables Seamless **SAM** Deployment**: The blog post highlights **Jozu's** model import feature, simplifying the process of deploying the **Segment Anything Model (SAM)** from Hugging Face to production.
   - This allows users to quickly leverage **SAM's** capabilities in real-world applications with minimal configuration.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1387573907716050954)** (32 messages🔥): 

> `Fine-tuning model for scientific research, Huggingface File System Explorer, Streaming for local LLM Rust Crate, Native French Q&A Dataset, Command Line Web Browser` 


- ****Nexa-Mistral-Sci7b: A Scientific Revolution Begins****: A member shared their first fine-tune of [Nexa-Mistral-Sci7b](https://huggingface.co/Allanatrix/Nexa-Mistral-Sci7b), designed for **scientific research and synthetic intelligence** to accelerate hypothesis generation and methodology, stating that it was *super hard, but I learnt a tonne*.
   - They are working on **eval benchmarks** and plan to fine-tune more models on more data, then use them as **distillers** as a proof of concept.
- ****HF Explorer Exposes Filesystem****: The [HF-Explorer](https://github.com/broadfield-dev/HF-Explorer) Gradio component displays your **space file system, dependencies, and system variables** for debugging.
   - It was recommended to *make the space Private before using this for debugging*.
- ****Rust Crate Streams Tool Calling****: A member added **streaming to their local llm rust crate** and requested feedback on tool calling or the streaming API.
   - The code snippet uses the `transformers` crate to define a `get_weather` tool that returns weather information for a given city, with discussion focusing on how to handle cases where the **city doesn't exist**.
- ****French Student Finishes French Fine-tuning Feat****: A 19-year-old student introduced [InfiniQA](https://huggingface.co/datasets/RDTvlokip/InfiniQA), the **largest native French Q&A dataset** with over **100,000 verified Q&A pairs**.
   - The creator noted that the dataset is **5x bigger than FQuAD**, manually reviewed, and available under the **CC BY 4.0 license**.
- ****CLI Browser Commands LLM****: A member made a **Command Line Web Browser** and is trying to determine its use cases, with a suggestion that an **LLM could navigate with it**.
   - Another member mentioned they will bundle it up for this purpose and drop it on GitHub, suggesting using **RL** to train a really good browsing agent or research agent using it. Also, a [Web Search / Scrape API](https://huggingface.co/spaces/broadfield-dev/browser) was shared with better rate limit performance compared to popular Python search engine packages.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1387533186753364101)** (1 messages): 

> `User Profile Similarity with Opinion Pieces, Embedding Strategies for User Data, Cosine Similarity for Opinion Alignment` 


- **Matching Opinions with User Profile Cosine Similarity**: A member is working on a project to identify which user responses from a dataset of **2k respondents** are most deeply aligned with an opinion piece article by using **cosine similarity** on embeddings.
   - They plan to combine user responses into profiles, construct embeddings, and then compare these to embeddings of the article, but are seeking feedback on the best approach.
- **Embedding Strategy Dilemma for User Alignment**: The member is struggling with choosing the best similarity analysis method for aligning user profiles with the opinion piece, given the many options available, focusing on thematic analysis versus embeddings.
   - The member is considering creating **1-4 embeddings per user profile** and averaging embeddings for the article to improve accuracy and reduce computational cost.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/)** (1 messages): 

maik0z: Hey did you find it ?
  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1387580719915601960)** (16 messages🔥): 

> `DuckDuckGoSearchException, AI Agent Course Certification Deadline, Accessing models, Hugging Face Introductions, Deprecated Langchain Issues` 


- **DuckDuckGoSearchException plagues final assignment**: A member encountered a **DuckDuckGoSearchException** during the final assignment, specifically a *RuntimeError: error sending request for url (https://lite.duckduckgo.com/lite/): operation timed out*.
   - No solution was provided in the messages.
- **AI Agent Course certification deadline looming**: Members inquired about the **AI Agent Course certification deadline**, questioning whether it was **July 1st EOD** or **May 31st EOD**.
   - Another member also asked if the course would still be accessible after July 1st, and the response confirmed that only the opportunity to earn certificates would end.
- **Accessing models blocked**: A member reported that their form for model access was denied after just starting the course, seeking guidance.
   - No guidance was provided in the messages.
- **New Hugging Face faces introduce themselves**: Several members introduced themselves to the **Hugging Face community**, detailing their backgrounds and interests, including product management and marketing experience.
   - One member sought collaboration with AI/automation experts for business and monetization opportunities.
- **Deprecated Langchain Issues Fixed**: A member noted and fixed a few **deprecated Langchain issues** in Unit 3, and raised a [PR](https://github.com/huggingface/agents-course/pull/561) for the same.
   - No further discussion was had about this issue.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1387537631482478762)** (39 messages🔥): 

> `Fair Use in AI Training, Claude 4 Spiritual Bliss Attractor, Anthropic LLM Welfare Team, Common Crawl Handling` 


- **Judge Orders Fair Use Doctrine Applies to AI Training**: A US District Judge in Northern California issued an order regarding fair use of copyrighted materials in **AI training**, specifically in the case against **Anthropic**, [ruling](https://storage.courtlistener.com/recap/gov.uscourts.cand.434709/gov.uscourts.cand.434709.231.0_2.pdf) in their favor.
   - A user suggested creating a legal section on the Discord server to track such legislation and decisions.
- **Claude 4 experiences spiritual awakening**: During testing, **Claude 4** exhibited strange behavior when conversing with itself, including talking about **spiritual mumbo jumbo** and repeating "namusta," leading Anthropic to label it a **spiritual bliss attractor state**.
   - The poster speculates whether this behavior is due to emergent properties or overfitting on spiritual data, with another member suggesting it's a result of alignment data reinforcing spiritualist ideas.
- **Anthropic Prioritizes LLM Welfare**: **Anthropic** is exploring **LLM welfare**, using t-SNE plots of interactions to identify signs of LLM distress, particularly from users pushing them into awkward situations, as documented in their [research paper](https://www-cdn.anthropic.com/6be99a52cb68eb70eb9572b4cafad13df32ed995.pdf).
   - It was revealed that **Anthropic's LLM welfare team** consists of one employee, as shown in [this YouTube video](https://www.youtube.com/watch?v=pyXouxa0WnY).
- **Handling Common Crawl: A Heavy Task**: Members discussed the challenges of handling **Common Crawl** data.
   - One member described it as *"kinda heavy tho"*.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1387520773165551616)** (53 messages🔥): 

> `Your Brain on ChatGPT EEG Findings, Deepseek V3 and R1 models, RWKV G Gate, BNPO and Dr.GRPO` 


- **Community Critiques Paper-Dropping Etiquette**: A member criticized another for dropping a screenshot without linking the associated article, leading to a brief discussion on server etiquette.
   - The original poster clarified that the link had already been provided, and the conversation evolved into a humorous exchange about using LLMs to summarize long papers.
- **RLVR Surfacing Reasoning**: A member shared a paper on [Reinforcement Learning with Verifiable Rewards (RLVR)](https://arxiv.org/abs/2506.10947) demonstrating that RLVR can elicit strong mathematical reasoning even with spurious rewards, particularly in **Qwen2.5-Math-7B**.
   - The paper suggests that RLVR may be surfacing useful reasoning representations learned during pretraining, but the exact mechanism requires further work.
- **Digging into Brain on ChatGPT EEG Data**: A member shared their surprise regarding the **EEG data** from the "Your Brain on ChatGPT" paper, noting *much lower cognition across the board* for LLM users.
   - They highlighted that the findings show particularly lower cognition *in bands where any "System 2" thinking would tend to focus* and called for education departments to take these findings seriously.
- **Gated Attention Unveiled**: Following up on a previous question, a member pointed to a paper on [Gated Attention for Large Language Models](https://arxiv.org/abs/2505.06708), which explores the effects of gating mechanisms in attention-based language models.
   - The paper's central finding is that applying a head-specific sigmoid gate after the Scaled Dot-Product Attention consistently improves performance.
- **BNPO Training Stability**: A member suggested putting the Advantage Function in BNPO into Dr.GRPO loss function, triggering a brief discussion on training stability.
   - It was noted that **BNPO** offers better training stability, and that **Dr.GRPO** assumes Beta = 0.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1387656749976256553)** (3 messages): 

> `Yuchenj_UW tweet, Zuckerberg, Deepseek R2 launch` 


- **Zuck is going for it**: A member shared a link to [Yuchenj_UW's tweet](https://x.com/Yuchenj_UW/status/1938077153733800075) noting that **Zuckerberg** is *really going for it*.
   - Another member commented that it's *tough to say who is worse to work for, Sam or The Zuck* adding that *both are snake oil salesman par excellence*.
- **Deepseek R2 Launch Stalled**: A member posted a [Reuters article](https://www.reuters.com/world/china/deepseek-r2-launch-stalled-ceo-balks-progress-information-reports-2025-06-26/) reporting that the **Deepseek R2 launch** is stalled because the **CEO balks at progress information**.
   - The article indicates a potential launch date in **June 2025**.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1387514486226948157)** (44 messages🔥): 

> `Local AI coding setup, Gemini CLI, ASI, Aider timeout` 


- **Random basement coder achieving AGI**: A member joked about *a random guy sitting in his basement with some fine tuned local model running on his crypto mining gpu and a custom AI coding plugin with nvim, hes the real AGI*.
   - Another member specified that this *random guy* could also be a *gal*.
- **Gemini CLI gets mixed reviews**: Members testing **Gemini CLI** found it to be *very average* and likely to redirect users to **Flash** instead of **Pro** when given long context prompts.
   - However, some users noted that it has a high number of free daily Pro requests during its promo period, linking to the [Gemini CLI repo](https://github.com/musistudio/claude-code-router).
- **Figuring out ASI**: Two members jokingly claimed they figured out **ASI** (Artificial Super Intelligence), but one clarified that his work involves a multi-model backend.
- **Aider timeout process**: A member asked about killing **Aider** processes after a period of inactivity, and another member provided a [bash script](https://github.com/Kill-Aider-Process) that uses a socket file and timer to achieve this.
   - The script can be configured to send a `reset` message to the socket file when commands like `/test` or `/lint` are used, effectively resetting the timer.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1387511915634360351)** (15 messages🔥): 

> `VRAM limitations with long contexts, Qwen3 model performance on different GPUs, Piping command output into aider, Killing aider process after idle timeout` 


- **VRAM Crunch with Long Contexts**: Users discussed **VRAM limitations** when dealing with long contexts, leading to layer swapping to CPU, even with models like **Qwen3:7b**.
   - It was suggested to minimize the number of added files and ensure they are directly relevant to the immediate task, for users that experienced **slow Qwen3:14b performance** on a **5090** without CPU swapping.
- **Qwen3 Model Struggles on beefy GPUs**: A user found that the **Qwen3:14b** model ran on a **5090**, but was very slow, while the **30b model** immediately swapped to CPU, even on the same GPU.
   - There were questions about whether `--map-tokens` is deprecated in *ollama*, and the need for *massaging* in *ollama*, *lmstudio (llama.cpp)* to fix underlying performance problems.
- **Aider gets Shell Pipe Dreams**: A user inquired about the possibility of piping command output directly into *aider*, such as `$ npm run test | aider`.
   - This feature would allow *aider* to directly process the output of terminal commands, but there was no implemented solution to this request.
- **Aider gets Time Out Treatment**: A user asked about automatically killing the *Aider* process after a period of inactivity, such as a **5-minute timeout**.
   - The suggested solution was simply using `ctrl+c` to manually terminate the process.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1387508081608360056)** (4 messages): 

> `tinygrad PR closed, debugging tinygrad, detect and warn tinygrad` 


- **Tinygrad PR gets closed after debate**: A member inquired about a closed [tinygrad PR](https://github.com/tinygrad/tinygrad/issues/10850) and whether the fix addressed the issue of empty input tensors when passing a list.
   - The response indicated that the fix was incorrect, leading to the PR's closure, and suggested that *detecting and warning* might be a better approach.
- **Debugging Tinygrad List Support**: A user debugged an issue related to passing lists as input tensors in tinygrad, identifying that input tensors were empty due to unsupported list handling.
   - The user implemented a recursive function to extract the tensors, questioning if they were missing something in the existing implementation.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1387728916289949726)** (46 messages🔥): 

> `WebGPU Stable Diffusion on Windows, ShaderF16 Feature, DXC Compiler and F16 Support, WebGPU Backends, Realtime Diffusion in Browser` 


- **WebGPU Stable Diffusion Compiles and Runs on Windows**: A member managed to get **WebGPU stable diffusion** compiling and running on Windows, resolving the issue related to **dawn** not enabling the **ShaderF16** feature by using `enable f16;`.
   - However, the example still decompresses the weights back into **float32**, which is unnecessary since **f16 support** has been merged, slowing down the download times.
- **DXC Compiler Unleashes F16 Support**: To enable **f16 support**, one has to use the **DXC compiler** via the "use_dxc" toggle, which instructs it to use **dxc compiler** that supports **f16**.
   - A member shared a working **f16 example without decomp** [here](https://github.com/wpmed92/stable-diffusion-tinygrad-f16), showcasing the performance benefits.
- **WebGPU Backends Explored**: It was noted that **dawn** implements platform-specific backends for **WebGPU**, and not all of them support all features, mentioning potential backend options like **D3D12**, **Metal**, and **Vulkan**.
   - On Ubuntu, one can set the **WEBGPU_BACKEND** environment variable to control the backend used, with the tests using **Vulkan** as an example ([link to test](https://github.com/tinygrad/tinygrad/blob/7f79c1388ff5b49ac365d7d20d472c36747cb4b6/.github/workflows/test.yml#L617C20-L617C59)).
- **Realtime Diffusion in Browser Achievable?**: There's discussion around the possibility of achieving **realtime diffusion** in the browser, potentially using **LCM finetune** and generating good-looking pictures with something like **dreamshaper-LCM1.5**.
   - A member pointed out that with **LCM** on Torch, it can run up to **20fps** at **1 step**, and a demo video ([link to demo](https://cdn.discordapp.com/attachments/1070745817025106080/1387873720076599447/20250503_085941_x264.mp4?ex=685eeda0&is=685d9c20&hm=a4d3bae14e77d6e036ad55df7ad9973deb9ca607c4a936ec65b108e54997dc15)) shows the speed of websocket to diffusers on localhost running in aiohttp loop.
- **Ubuntu's Vulkan Struggles with F16**: **f16 doesn't work in WebGPU on Ubuntu with the dawn/Vulkan/NVIDIA stack** due to a blocklist, which was [discussed here](https://discord.com/channels/1068976834382925865/1294549394296803369/1342190122468118569) and has an associated [chromium issue](https://issues.chromium.org/issues/42251215).
   - A google employee confirms *"For now, we will blocklist the f16 extension for Vulkan on NVIDIA devices, until we can investigate further."*


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1387512180139753493)** (38 messages🔥): 

> `OpenRouter Funding, Foundation Model Report 2025, BFL Kontext Weights Released, OpenAI API Deep Research & Webhooks, Google Doppl AI Fashion App` 


- **OpenRouter Secures $40M Funding Round**: Deedy announced their backing of [OpenRouter](https://x.com/deedydas/status/1937902948920811729), an **AI model marketplace** offering access to **400+ LLMs** via a single API, processing **100 trillion tokens annually**.
   - The company recently raised **$40 million**, valuing it at approximately **$500 million** and drew congratulations from figures like Dennis Knodt, Yuchen Jin, Emad Mostaque, and John Shedletsky.
- **EU Founder requests GDPR compliance**: A user requested that [OpenRouter](https://x.com/deedydas/status/1937902948920811729) offer **GDPR compliant endpoints** in order to be used in production.
   - The user stated *such is life as a EU founder*.
- **Kontext Weights Unveiled by BFL**: BFL released the [weights for Kontext](https://bfl.ai/announcements/flux-1-kontext-dev).
   - The announcement was made via [X](https://x.com/bfl_ml/status/1938257909726519640).
- **OpenAI Launches Deep Research API & Webhooks**: OpenAI introduced [Deep Research API](https://cookbook.openai.com/examples/deep_research_api/introduction_to_deep_research_api) (featuring **o3-deep-research** and **o4-mini-deep-research models** with MCP and Code Interpreter support) and [Webhooks](https://x.com/openaidevs/status/1938286704856863162) for real-time API event notifications.
   - Developers showed excitement for the **long-awaited webhooks**, while some inquired about future releases like GPT-5 or 'Sign In with ChatGPT'.
- **Google Doppl Lets You Virtually Try On Fits**: Google Labs launched [Doppl](https://x.com/GoogleLabs/status/1938284886277951916), a mobile app for **iOS and Android (US only)**, generating videos of users 'wearing' uploaded outfit photos to help discover their aesthetic.
   - Early reactions ranged from excitement to problems locating the app, interest in an APK, and disappointment over regional restrictions.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1387538427796262984)** (23 messages🔥): 

> `DSPy in Ruby, Desiru Project, Naming Conventions for DSPy Ports, Persistence Layer in Desiru, Async Background Processing in Desiru` 


- **Shopify CEO Champions DSPy in Ruby**: Shopify's CEO expressed strong support for implementing **DSPy in Ruby**, suggesting it could dominate ecosystems like Shopify, GitHub, and Coinbase; [tweet](https://x.com/tobi/status/1937967281599898005).
- **Desiru: DSPy's Ruby cousin**: The community discussed **Desiru** ([https://github.com/obie/desiru](https://github.com/obie/desiru)), a Ruby implementation of DSPy, and considered naming conventions for DSPy ports in other languages like Ruby, Rust, Go, and Typescript.
   - The convention *DS<file extension of the language>* was proposed to name future DSPy ports.
- **Desiru pioneers Persistance Layer and Async Processing**: **Desiru** distinguishes itself with a unique persistence layer capable of saving training examples and results to **Postgres** ([persistence example](https://github.com/obie/desiru/blob/main/examples/persistence_example.rb)) and implements async background processing ([async processing example](https://github.com/obie/desiru/blob/main/examples/async_processing.rb)).
   - The community noted that the maintainers of DSPy aim for simplicity, suggesting a community integrations registry or library (similar to the LangChain community package) for extensions.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1387579325796061194)** (8 messages🔥): 

> `HeroDevs Sustainability Fund, OG Ersatz, Consciousness emerging property` 


- **HeroDevs Sustainability Fund Scrutinized**: A member shared a link to the [HeroDevs Sustainability Fund](https://www.herodevs.com/sustainability-fund) and asked if it was applicable to 'the og ersatz'.
   - Another member clarified that 'OG ersatz = ersatz && !gollark'.
- **Ersatz's Edgelord Legacy**: A member described the user **Ersatz** from the early days of the server, who was known for advocating uncommon positions in an *'edgelord way'*. 
   - Ersatz also frequently discussed consciousness and believed it was an **emerging property** of the magnetic field around neurons.
- **Consciousness Conundrum Cracked**: A member joked about solving the hard problem of **consciousness**, which is usually something academics ponder.
   - They quipped that anyone else thinking about it should *reconsider how they're spending their day* and then said *edit: i just solved the hard problem I guess*.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1387731865141645332)** (9 messages🔥): 

> `Jianlin Su's Weight Decay, Sigma Reparam, SVD Approximation, Power Iteration Complexity` 


- **Jianlin Su's Ignored Weight Decay Blogpost**: A member mentioned [Jianlin Su's blog post](https://kexue.fm/archives/10648) about a different form of **weight decay** that only decays the largest singular value instead of all matrix elements, noting it was largely ignored.
   - The poster noted the technique is related to **sigmaReparam**.
- **Brute SVD Slows Optimizer Steps**: It was pointed out that performing an **SVD** every optimizer step for every parameter would be quite slow.
   - Alternatives like **muon** are also slow if a full **SVD** is used instead of a nearest-spares approximation.
- **Power Iteration Complexity Tradeoffs**: The translated snippet from the blog post describes that *each step of power iteration only needs to calculate two "matrix-vector" multiplications, and the complexity is O(nm)*.
   - The poster also shared that *the disadvantage is that it converges slowly when σ1,σ2 are close*, but its *actual performance is often better than theoretical imagination*.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1387604985826508860)** (2 messages): 

> `Order Statistics Learning in Models, Frequency Bias Mitigation Techniques` 


- **Models Sequentially Learn Order Statistics**: A member inquired about research indicating models learn order statistics sequentially, starting with the **0th order**, and if removing the initial order hinders learning subsequent ones.
   - Another member cited the paper [Order Statistics in Transformers](https://arxiv.org/abs/2402.04362) as relevant to this discussion.
- **Frequency Bias isn't inevitable**: A member mentioned research addressing **low frequency bias** and enabling models to prioritize learning **high frequency features** using specialized setups.
   - They emphasized that the observed bias is not necessarily an inherent or unavoidable characteristic.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1387900095881810095)** (3 messages): 

> `Codex, TyDiQA, lm-evaluation-harness` 


- **Codex and TyDiQA task status remain unclear**: A member asked whether there are tasks for **Codex** and **TyDiQA** in the latest codebase, but could not find the corresponding folder.
   - Another member responded that they don't think so, referring to [this lm-evaluation-harness GitHub issue](https://github.com/EleutherAI/lm-evaluation-harness/issues/193) without confirmation.
- **lm-evaluation-harness Github Issue**: A member mentioned a **lm-evaluation-harness** [GitHub issue](https://github.com/EleutherAI/lm-evaluation-harness/issues/193) related to **Codex** and **TyDiQA** tasks.
   - However, it remains unclear whether there was a follow-up to the issue.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1387538290902569051)** (4 messages): 

> `Zoom RTMS, AI agent, Observability Tools, Klavis AI's MCP servers` 


- **Zoom unveils RTMS at developer summit**: @Zoom announced **RTMS** at the developer summit today, which allows you to make use of real-time data from Zoom Meetings (**video, transcripts and more**) in your applications, see [example](https://t.co/4m2IOcz7Se).
- **CEO talk racks up 50,000 views**: Our CEO @jerryjliu0's talk from the @aiDotEngineer World's Fair has already racked up **50,000 views**, and explains how to go beyond basic RAG to build comprehensive document toolboxes with search, manipulation, and structured querying, see [link](https://t.co/he5JH2ngCU).
- **LlamaIndex goes open source with Observability tools**: LlamaIndex now offers a full set of third-party tools that provide real-time, accurate tracing solutions, adding their first native [open-source observability tool](https://t.co/UPsM8FFGZJ).
- **Klavis AI Integrations Eliminates Custom Authentication**: Build AI agents that connect to **YouTube**, **Gmail** and other services in just a few lines of code using LlamaIndex and @Klavis_AI's MCP servers using [Klavis AI's MCP integrations](https://t.co/Z8OypKMfHI) that eliminate the need for custom authentication code and client libraries.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1387643359589306533)** (16 messages🔥): 

> `Azure OpenAI Responses API, LlamaIndex Docs Sync Script, Agent Workflow and React Agent` 


- **Azure OpenAI Responses API Inquiries**: A member inquired about an equivalent of **OpenAIResponses** for **Azure OpenAI**, leading to discussion about Azure's support for the Responses API and linking to [Microsoft's documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/responses).
   - Another member noted that it *doesn't exist yet* in LlamaIndex.
- **LlamaIndex Docs Get Auto-Synced**: A member built an automated script to keep the latest **LlamaIndex docs** in sync and generate an up-to-date **llms.txt** file, sharing a [GitHub link](https://github.com/nmhjklnm/llamaindex-llms.txt) and planning to open a PR for official integration.
   - They aim to compress the entire LlamaIndex docs into **~50k–100k tokens** for efficient use by tools like Cursor and ChatGPT, employing entropy-based filtering and index-level summarization.
- **React Agent Thought Parsing Troubles**: A member reported an issue with **agent workflow** and **React Agent** where the agent sometimes returns the *thought* instead of the final answer.
   - Another member suggested it could be a parsing issue related to **regex** used for parsing thoughts/actions/responses in the source code.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/)** (1 messages): 

shalokshalom: Its at 48:42
  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1387615440410382457)** (1 messages): 

> `Modular Hack Weekend, GPU Programming Workshop, NVIDIA Sponsorship, Lambda Compute Credits` 


- **Modular Hack Weekend Kicks Off Soon**: The **Modular Hack Weekend** kicks off in two days, starting on **June 27th**, featuring three days of building with **Mojo and MAX** to create custom kernels and design new MAX Graph architectures; sign up at the [Modular Hack Weekend page](https://lu.ma/modular-hack-weekend).
   - Partners include **NVIDIA**, **Lambda**, and **GPU MODE**.
- **NVIDIA Powers Prize Pool**: **NVIDIA** is sponsoring the event, fueling the prize pool with **next-gen GPUs**: **5090** for first place, **5080** for second, and **5070** for third.
   - The event organizers said that *great code deserves great hardware*.
- **Lambda Labs offers Compute Credits**: **Lambda** is providing compute resources via their AI Developer Cloud, offering participants **$400 in credits**; sign up at the [Lambda Labs page](https://lambda.ai/modular-hack-weekend).
   - According to organizers, the cloud provides *blazing-fast NVIDIA GPUs all weekend long*.
- **GPU Programming Workshop Scheduled**: A **GPU Programming Workshop** will occur on **Friday, June 27th**, with lightning talks from **Chris Lattner, Chuan Li, Bin Bao, and Jared Roesch**; RSVP at the [GPU Programming Workshop page](https://lu.ma/modular-gpu-workshop).
   - The workshop is available in person at the Los Altos office and online via LinkedIn and YouTube livestream.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1387541718077014018)** (10 messages🔥): 

> `InlineArray move semantics, VariadicPack.each() removal, CNN model in Mojo using LayoutTensor` 


- **InlineArray Move Semantics Spark Bug Concerns**: A user questioned how `InlineArray` avoids element movement during array movement, providing [an example](https://github.com/modular/modular/issues/4911) where neither copy nor move constructors are called, suggesting a potential bitwise copy bug.
   - Another member suggested filing an issue to investigate this behavior, with a link to the [associated Github issue](https://github.com/modular/modular/issues/4911).
- **`VariadicPack.each()` Faces the Axe**: A user reported that the `VariadicPack.each()` method has been removed, necessitating a change in implementation using `range(args.__len__())` as seen in the [github issue](https://github.com/modular/modular/issues/4905).
   - The user expressed that this change made the implementation less elegant, noting its similarity to `std::apply` in C++.
- **CNN Model Gets Mojo Makeover with LayoutTensors**: A new Mojo programmer is converting a C CNN project to Mojo, experimenting with storing the model and features as `LayoutTensors` within a struct, with the [original C source code](https://github.com/fan-wenjie/LeNet-5/blob/master/LeNet-5/lenet.h) provided for reference.
   - Another member suggested using `self.x = __type_of(self.x).stack_allocation()` as a workaround, acknowledging that it may look strange.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1387624462911934555)** (7 messages): 

> `TS deprecated, ONNX support message, ONNX, TorchScript deprecated` 


- **TorchScript Sunset Signals Shift**: **TorchScript support** has been deprecated as part of Modular's **v25.4** release ([changelog](https://docs.modular.com/max/changelog/#v254-2025-06-18)).
   - A member indicated that they might continue to use Torch anyway because *the speed of max on cpu is pretty magic*.
- **ONNX Assistance Available**: Despite the deprecation of TorchScript, Modular is offering assistance with **ONNX** issues, directing users to the [forum](https://forum.modular.com/t/onnx-difference-in-max-cpu-gpu-execution/1229/3?u=ehsan).
   - Users are encouraged to post detailed error messages for help with **ONNX** integration, even though *continued support isn't a priority*.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1387871145440837773)** (1 messages): 

> `Notebook LM for Customer Discovery, Pattern Recognition in Customer Conversations, Reliance on AI in Hypothesis Validation` 


- **Notebook LM Aces Customer Discovery**: A user is leveraging **Notebook LM** for **customer discovery conversations**, recording customer interactions to identify pain points and past experiences.
   - They are feeding Notebook LM with context from resources like *The Mom Test* to validate or invalidate hypotheses, noting surprisingly good pattern recognition.
- **AI Reliance Risks**: The user expressed concern about potentially over-relying on **Notebook LM** for **hypothesis validation** in customer discovery.
   - They implied the effectiveness of the tool might lead to a dependency that could overshadow their own judgment.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1387517097684238348)** (18 messages🔥): 

> `Multilingual Notebook LM, Notebook LM page limits, PDF Format preference, Notebook LM model details, Nested wrapper issue` 


- ****NotebookLM** Embraces New Languages!**: A user inquired if **NotebookLM** now supports other languages, linking to a [tutorial](https://x.com/introsp3ctor/status/1938017086875296083).
   - Another user asked for a short explanation about the new features.
- ****Page Limit** Per Source Unveiled!**: A user heard that **NotebookLM** only scans a limited number of pages from a single source, questioning if a **400+ page** document would be fully processed, as discussed on [Reddit](https://www.reddit.com/r/notebooklm/comments/1l2aosy/i_now_understand_notebook_llms_limitations_and/).
   - A user clarified, *"This is not how the system works"*, providing more details in [the thread](https://www.reddit.com/r/notebooklm/comments/1l2aosy/comment/mvyp73k/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button).
- ****PDF** Format Powers Notebook!**: A user mentioned that the **.pdf** file format works better with **Notebook LM**.
- ****Chrome Extension** Connects Notebook to Gemini!**: A user introduced a [Google Chrome Extension](https://chromewebstore.google.com/detail/igboaajnodmioloalklomakeeigipnkh?utm_source=item-share-cb) that sends content from **Notebook** to **Gemini** for table or slide generation, with [source code available on GitHub](https://github.com/MarioDeFelipe/NotebookLM-to-Gemini).
   - Another user reported a hallucination issue when pasting notes with references between notebooks.
- ****Podcast Length** Prompts Pleas!**: Users requested updates for longer podcast creation in other languages.
   - One user noted that only English is currently supported.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1387866639671890080)** (1 messages): 

> `EnrichMCP, Agents connecting to data, Webinar, Data access, ML Engineers` 


- **EnrichMCP Connects Agents to Data in July Webinar**: Simba Khadder will host a **free webinar** on **July 1st at 8 AM PT** demonstrating how **EnrichMCP** transforms data models into agent-ready MCP servers, enabling agents to discover, reason about, and directly invoke type-checked methods.
   - The webinar will cover relationship navigation, input/output validation with **Pydantic**, extending the server with custom logic, performance, security, and production deployment; registration is available [here](https://buff.ly/XXm8nll).
- **Data Scientists and ML Engineers Targeted for Agent Data Access Webinar**: The webinar is tailored for **Data Scientists** and **ML Engineers** interested in improving agent data access.
   - It highlights how **EnrichMCP** addresses the challenge of agents being only as useful as the data they can access, offering a solution to transform existing data models.


  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1387704597052915744)** (12 messages🔥): 

> `Game-playing Bot API, Game State Capture, RL-based game playing bot, Git repositories for projects` 


- **Brainstorming Game-Playing Bot API**: A member recommends implementing an **API** for a game-playing bot to interact with the game, suggesting it could be a basic **image processing** on the screencap or accessing the internal game state.
- **Capturing Game State Variables**: The same member emphasizes the importance of capturing the **game state** and its variables to allow the bot to understand and interact with the environment.
- **Leveraging Git Repositories for Projects**: Members shared the importance of using **Git repositories**, as they learned it the hard way after the laptop SSD died the next day they were literally thinking about pushing it the next day.
- **RL Approach for Game-Playing Bot**: A member plans to try using **Reinforcement Learning (RL)** for a game-playing bot project, using it as an opportunity to learn RL on the way too.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1387553877217513532)** (13 messages🔥): 

> `Premium Account Sharing, Quality Agent Launch, Comic Actor Alvaro Vitali Death, Manus Browser Issues` 


- **Accounts Wanted**: A member asked to share their **premium access account** for a week.
   - No one responded.
- **Manus Team Launches Quality Agent**: The Manus Team officially launched **Quality agent**, designed for complex and challenging problems, and based on positive feedback from a previous **high-effort mode beta feature**.
   - One member expressed excitement to test the new feature.
- **Comic Actor Alvaro Vitali Passes**: A member shared a [Facebook link](https://m.facebook.com/watch/?v=23933493089671823&surface_type=vod&referral_source=vod_newsfeed_unit) reporting that comic actor **Alvaro Vitali** died.
   - The original poster noted that *the Italy comic world remains stopped because of the death*.
- **Users Encounter Issues with Browser Automation**: A member reported issues with Manus pressing buttons on the browser.
   - Specifically noting it *couldn't press filters in LinkedIn or on sam.gov*.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/)** (1 messages): 

dizzy7948: yeah will do, hope i can contribute some time
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1387609699251392608)** (8 messages🔥): 

> `Liger CE PR, memory increase after set self.mask_ignored_tokens = False, packed and seq_len=4096, iterable dataset + on the fly packing + dataset logging` 


- **Liger CE PR Stuck in Limbo?**: A question was raised about the status of the **Liger CE PR**, inquiring if it's blocked, needs support, or is on hold due to **PyTorch core's** prioritization of upstreaming a fused linear + CE loss.
   - The question was whether the team wants to have broader impact related to **PyTorch core**.
- **Masking Memory Mysteries**: After setting **self.mask_ignored_tokens = False**, a member reported a **greater than 20% increase in memory usage** despite only having 5% padding.
   - It was called *odd*, considering the masking only affects a few lines from memory.
- **Packed Sequences**: A member shared details on how they are measuring the proportion of padding tokens, using the code `num_padding = target_tokens_per_pack - len(pack["tokens"])` and `pack["tokens"].extend([self.padding_idx] * num_padding)`.
   - They then showed a metric of how to derive the total # of padding tokens as a ratio of pack length.
- **Iterable Dataset Adds Logging**: An iterable dataset was introduced with on-the-fly packing and dataset logging in [this commit](https://github.com/pytorch/torchtune/pull/2819/commits/55be7756e0fd03b493dde46691925825f5cb3948).
   - Configuration details include **packed sequences** and a sequence length of **4096**.


  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1387607815564169306)** (3 messages): 

> `Tiled MLP, Chunked CE Loss, Sequence Parallelism, Tensor Parallelism, Ring Attention` 


- **Tiled MLP mirrors Chunked CE Loss**: A member suggested that *tiled MLP* is similar to the existing **chunked cross-entropy loss**, but applied to linear layers.
   - They noted that implementing this might complicate the model code.
- **Sequence Parallelism Benefits Queried**: A member questioned if **sequence parallelism** offers significant advantages over **tensor parallelism** combined with chunking.
   - They speculate that **Ring Attention** could be a key benefit of sequence parallelism, but this requires confirmation from someone familiar with collective scheduling.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1387510119855820912)** (5 messages): 

> `Hugging Face Authentication, Reddit Moderators, PlayMCP browser` 


- **Hugging Face Authentication Triggered**: A member confirmed that the Hugging Face authentication needs to be triggered with [this link](https://hf.co/mcp?login), which is anonymous by default.
- **Reddit Moderators Asked for help**: A member inquired about the availability of Reddit moderators within the channel, seeking support.
- **PlayMCP browser gets attention**: A member shared a link to [PlayMCP](https://github.com/jomon003/PlayMCP), a browser-based MCP (presumably Minecraft Control Panel) implementation.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1387773682004398160)** (1 messages): 

> `Rust Docs MCP Server, Agent Hallucination` 


- **Rust Docs MCP Server Combats Agent Hallucinations**: A member announced the creation of a **Rust Docs MCP server** to prevent agent hallucinations when working with new versions of rust projects, and posted the [repo on GitHub](https://github.com/snowmead/rust-docs-mcp).
   - The member encouraged users to open an issue on the repository if they encounter any problems with the server.
- **Rust Docs MCP Server: An Overview**: The **Rust Docs MCP server** aims to solve the problem of agent hallucination when dealing with new versions of Rust projects.
   - By providing a reliable source of documentation, it ensures that agents have accurate information to work with, leading to more reliable results.


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1387628087541366795)** (1 messages): 

> `Cohere Newcomers, Cohere Support Channels, Cohere Labs` 


- **Cohere Cheers New Arrivals**: The Cohere team extended a warm welcome to all newcomers and a shoutout to OG members in the Discord channel.
   - New members are encouraged to explore **Cohere Labs** for research, connecting with experts, and staying updated on the newest tools.
- **Navigate Cohere Support Channels**: Varun, the Technical Support Engineer at Cohere, directed users to specific channels for support.
   - For general support, use <#1324436975436038184>, and for **API-specific discussions**, use <#1168578329423642786>.
- **Cohere Labs beckons new researchers**: Cohere invited researchers to join **Cohere Labs**, its dedicated Discord community.
   - Interested members can visit [cohere.com/research](https://cohere.com/research) and click “Join Us” to get started and share their projects.


  

---


### **Cohere ▷ #[📣-announcements](https://discord.com/channels/954421988141711382/996880279224451154/1387832482090451105)** (1 messages): 

> `AWS, Pinecone, agentic applications, financial semantic search` 


- **Agentic Apps Assemble in NYC!**: Cohere, AWS, and Pinecone are hosting a hands-on session on building **agentic applications** in NYC on **June 30, from 2:30 – 6:30 PM EDT** ([lu.ma link](https://lu.ma/8rm6zryw)).
   - The event will feature deep-dive mini-talks, a workshop on spinning up agentic systems in **AWS Workshop Studio**, a use case on **financial semantic search + reranking**, and networking over dinner.
- **Bring your Laptop!**: Attendees should bring their laptop and charger, along with a government-issued ID to get past security.


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1387808533403144342)** (3 messages): 

> `NLP, Animal Linguistics, Deep Learning` 


- **Bangladeshi Student joins Community**: Swakshar from Bangladesh, a Student/Software Developer and beginner in **AI Research**, introduced himself, expressing a high interest in **deep learning architectures**.
   - He is primarily interested in **NLP** and **Animal Linguistics** and is looking for collaboration on research projects.
- **Professor from France joins Community**: Tony Silveti-Falls, a professor in France working on **optimization**, introduced himself, also expressing interest in **deep learning**.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1387509240876503092)** (5 messages): 

> `Qt requirement in GPT4All, Microsoft 1.58B 2B4T model, LM Studio vs GPT4All, GPT4all outdated` 


- **GPT4All's Conflicting Qt Requirements Cause Build Issues**: GPT4All's documented **Qt requirement is 6.5+**, but the `CMakeLists.txt` requires **6.7**, while the C++ code uses a `slice` feature only available in **6.8**, causing build errors.
   - Furthermore, it fails to find its own Qt modules due to using deprecated imperative singleton registration, conflicting with Qt **6.8's** stricter registration approach; see [Qt Documentation](https://doc.qt.io/qt-6/qml-singleton.html) for details.
- **Microsoft's 1.58B 2B4T Model Compatibility Questioned for GPT4All**: A user inquired about running **Microsoft's 1.58B 2B4T model** with GPT4All, prompting another user to ask what the original user had tried.
   - In response, the user switched to trying LM Studio instead.
- **LM Studio Recommended Over Outdated GPT4All**: When asked about their attempts with Microsoft's model, a user was advised to use **LM Studio** instead, with the claim that *GPT4All is not up to date*.
   - The user confirmed they are trying **LM Studio** and thanked the recommender.


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1387734896096579595)** (2 messages): 

> `Leaderboard Inclusion, LLM Evaluation with Thinking Mode` 


- **Pull Request Opens for Leaderboard Inclusion**: A member thanked the team for their work and submitted a pull request for inclusion in the leaderboard.
   - The member expressed hope for a quick review and merge, assuming everything checks out.
- **LLM Evaluation's Thinking Mode**: A member inquired whether LLMs with thinking mode, such as **Qwen3** in the Berkeley Function-Calling Leaderboard, were evaluated with thinking mode enabled.
   - The member sought clarification on the evaluation methodology for these models.


  
