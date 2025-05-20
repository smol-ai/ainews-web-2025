---
id: MjAyNS0w
title: not much happened today
date: '2025-05-19T05:44:39.731046Z'
description: >-
  **Meta** released **KernelLLM 8B**, outperforming **GPT-4o** and **DeepSeek
  V3** on KernelBench-Triton Level 1. **Mistral Medium 3** debuted strongly in
  multiple benchmarks. **Qwen3** models introduced a unified framework with
  multilingual support. **DeepSeek-V3** features hardware-aware co-design.
  **BLIP3-o** family released for multimodal tasks using diffusion transformers.
  **Salesforce** launched **xGen-Small** models excelling in long-context and
  math benchmarks. **Bilibili** released **AniSORA** for anime video generation.
  **Stability AI** open-sourced **Stable Audio Open Small** optimized for Arm
  devices. Google’s **AlphaEvolve** coding agent improved **Strassen's
  algorithm** for the first time since 1969. Research shows **chain-of-thought
  reasoning** can harm instruction-following ability, with mitigation strategies
  like classifier-selective reasoning being most effective, but reasoning
  techniques show high variance and limited generalization. *"Chain-of-thought
  (CoT) reasoning can harm a model’s ability to follow instructions"* and
  *"Mitigation strategies such as few-shot in-context learning, self-reflection,
  self-selective reasoning, and classifier-selective reasoning can counteract
  reasoning-induced failures"*.
companies:
  - meta-ai-fair
  - mistral-ai
  - qwen
  - deepseek
  - salesforce
  - bilibili
  - stability-ai
  - google
models:
  - kernelllm-8b
  - gpt-4o
  - deepseek-v3
  - mistral-medium-3
  - qwen3
  - blip3-o
  - xgen-small
  - anisora
  - stable-audio-open-small
  - alphaevolve
topics:
  - benchmarking
  - model-performance
  - multilinguality
  - hardware-optimization
  - multimodality
  - image-generation
  - video-generation
  - text-to-audio
  - model-parallelism
  - chain-of-thought
  - instruction-following
  - reasoning
  - mitigation-strategies
people:
  - reach_vb
  - lmarena_ai
  - theadimeline
  - adcock_brett
  - jxmnop
  - dair_ai
  - omarsar0
---


**a quiet day.**

> AI News for 5/16/2025-5/19/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (215 channels, and 11148 messages) for you. Estimated reading time saved (at 200wpm): 947 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

It's an open secret that Google will be launching a lot of stuff at I/O tomorrow, and is already starting to [roll out Jules](https://x.com/nanulled/status/1924554666731262065). There've been other launches - [Amazon's Strands Agents](https://strandsagents.com/), and Anthropic's [Claude Code SDK](https://docs.anthropic.com/en/docs/claude-code/sdk) but nothing quite hitting title story status.

**Expo Explorer** tickets for AI Engineer [went live](http://ti.to/software-3/ai-engineer-worlds-fair-2025) over the weekend. If you love the hallway track, expo sessions, and meeting every top cloud/startup/employer in AI Eng, join us.

![](https://resend-attachments.s3.amazonaws.com/YrrfaUU3X1AHkBF)

There's a limited number of discounts available [here](https://ti.to/software-3/ai-engineer-worlds-fair-2025/discount/HNEXPO) for the first 50 AINews readers.

---

# AI Twitter Recap

**AI Model Releases and Performance**

- **Meta released KernelLLM 8B**, which outperformed **GPT-4o** and **DeepSeek V3** in single-shot performance on KernelBench-Triton Level 1, and with multiple inferences, it outperformed **DeepSeek R1**, according to [@reach_vb](https://twitter.com/reach_vb/status/1924478755898085552).
- **Mistral Medium 3** made a strong debut, ranking #11 overall in chat, #5 in Math, #7 in Hard Prompts & Coding, and #9 in WebDev Arena, according to [@lmarena_ai](https://twitter.com/lmarena_ai/status/1924482515244622120).
- **Qwen3 models**, including dense and Mixture-of-Expert models ranging from 0.6B to 235B parameters, were introduced, featuring a unified framework and expanded multilingual support, according to [@TheAITimeline](https://twitter.com/TheAITimeline/status/1924232110383960163).
- **DeepSeek-V3** uses hardware-aware co-design and addresses scaling issues, according to [@TheAITimeline](https://twitter.com/TheAITimeline/status/1924232113101890003).
- **BLIP3-o**, a family of fully open unified multimodal models using a diffusion transformer, was released, showing superior performance on image understanding and generation tasks, according to [@TheAITimeline](https://twitter.com/TheAITimeline/status/1924232118755824119).
- **Salesforce** released the **xGen-Small** family of small AI models, with the 9B parameter model showing strong performance on long-context understanding and math + coding benchmarks, according to [@adcock_brett](https://twitter.com/adcock_brett/status/1924133781704786366).
- **Bilibili** released **AniSORA**, an anime video generation model, according to [@reach_vb](https://twitter.com/reach_vb/status/1924425789774123316).
- **Stability AI** open-sourced **Stable Audio Open Small**, a text-to-audio AI model that generates 11s of audio and is optimized for Arm-based consumer devices, according to [@adcock_brett](https://twitter.com/adcock_brett/status/1924133939376996539).
- [@jxmnop](https://twitter.com/jxmnop/status/1924207755956478400) discussed **a 2003 paper from Montreal on training neural networks on text**, noting its model and techniques, including model parallelism, were forward-thinking.
- **AlphaEvolve**, a coding agent that uses LLM-guided evolution to discover new algorithms and optimize computational systems, was released by Google, and found the first improvement on **Strassen's algorithm** since 1969, according to [@dair_ai](https://twitter.com/dair_ai/status/1924150361750655178) and [@adcock_brett](https://twitter.com/adcock_brett/status/1924133683444793819).

**AI Safety, Reasoning and Instruction Following**

- **Chain-of-thought (CoT) reasoning can harm a model’s ability to follow instructions**: [@omarsar0](https://twitter.com/omarsar0/status/1924458157444579700) summarized a paper noting this counterintuitive weakness in reasoning-enhanced language models and shared mitigation tactics, also adding the paper to the Reasoning LLMs Guide.
- **Mitigation strategies** such as few-shot in-context learning, self-reflection, self-selective reasoning, and classifier-selective reasoning can counteract reasoning-induced failures, with classifier-selective reasoning being the most robust, according to [@omarsar0](https://twitter.com/omarsar0/status/1924458176096751806).
- **Reasoning fails to generalize across environments**, and prompting strategies yield high variance, undermining the reliability of advanced reasoning techniques, according to [@omarsar0](https://twitter.com/omarsar0/status/1924182841677709540) and [@omarsar0](https://twitter.com/omarsar0/status/1924182837307089216).
- **Larger models benefit less from strategic prompting**, and excessive reasoning hurts smaller models on simple tasks, according to [@omarsar0](https://twitter.com/omarsar0/status/1924182839081218092) and [@omarsar0](https://twitter.com/omarsar0/status/1924182835289620950).
- [@RichardSocher](https://twitter.com/RichardSocher/status/1924217608569528799) discussed the **AI Safety Paradox**, arguing that as the marginal cost of intelligence decreases, it can lead to better defense in biological or cyber warfare by identifying and addressing more attack vectors.

**AI Tools and Applications**

- **Microsoft added Grok to their foundry model collection**, according to [@TheTuringPost](https://twitter.com/TheTuringPost/status/1924508051253653745) and [@ibab](https://twitter.com/ibab/status/1924518628172693922), making Grok 3 available on Microsoft Azure.
- **GitHub Copilot** now supports the entire software development lifecycle, offering agent mode, team support, app modernization, and SRE Agent, according to [@TheTuringPost](https://twitter.com/TheTuringPost/status/1924495827999031709).
- **OpenAI launched Codex**, a new coding agent that builds features and fixes bugs autonomously and is available for Pro, Enterprise, and Team users, according to [@adcock_brett](https://twitter.com/adcock_brett/status/1924133661072396293).
- **Alibaba's Qwen team** made **Deep Research for Qwen Chat** available for all users, providing users the ability to prepare detailed reports on different subjects, according to [@adcock_brett](https://twitter.com/adcock_brett/status/1924133804630753660).
- **Notion** launched an "AI for Work" suite for its business plan subscribers, offering AI meeting notes, access to different AI models, enterprise search, and a research mode to draft docs, according to [@adcock_brett](https://twitter.com/adcock_brett/status/1924133849610543431).
- **Alibaba's Wan** dropped **Wan2.1-VACE**, a unified AI for video creation and editing, available in 1.3B and 14B sizes, according to [@adcock_brett](https://twitter.com/adcock_brett/status/1924133827095498952).
- **MLX-powered LLMs can be accessed directly from Hugging Face Hub**, enabling blazingly fast intelligence at the terminal, according to [@reach_vb](https://twitter.com/reach_vb/status/1924517049474101412).
- **New features for Modal Labs' Dicts serverless KV store** include no scale limits, LRU-cache semantics, distributed locking, and durability, according to [@akshat_b](https://twitter.com/akshat_b/status/1924552967673545055).
- **LangChain** announced node-level caching support for LangGraph, which allows for faster iteration, according to [@hwchase17](https://twitter.com/hwchase17/status/1924557667634172099).
- [@fchollet](https://twitter.com/fchollet/status/1924509605050327475) highlighted **Genspark AI Sheets**, an application that lets users talk to their spreadsheets.

**AI Business and Strategy**

- **Sakana AI** and **MUFG Bank** have signed a comprehensive partnership agreement to power its systems with AI, according to [@AIatMeta](https://twitter.com/AIatMeta/status/1924502785028190366) and [@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1924442310210678974), which will allow Sakana AI to turn profitable within a year, according to [@hardmaru](https://twitter.com/hardmaru/status/1924480171606003841).
- **Cohere** is partnering with **Dell** to offer secure, agentic enterprise AI solutions on-premises, according to [@cohere](https://twitter.com/cohere/status/1924512634373865950).
- **Perplexity** is now snappier, faster, and more chatty on Whatsapp, leading to increased usage, according to [@AravSrinivas](https://twitter.com/AravSrinivas/status/1923924897614659922).
- Andrew Ng will be taking the stage at @Snowflake's Dev Day 2025 on June 5 in San Francisco, according to [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1924484108974993540).
- [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1924509198848819347) discussed how the **new product development law requires teams to cultivate a culture where they live and breathe generative models** and experiment to discover new product experiences in days, not months.
- [@adcock_brett](https://twitter.com/adcock_brett/status/1923927753969353029) highlighted **the importance of setting company values at the beginning**, as it's difficult to course correct later.

**Infrastructure, Tools and Datasets**

- **NVIDIA** open-sourced **Physical AI models**, reasoning models that understand physical common sense and generate appropriate embodied decisions, according to [@reach_vb](https://twitter.com/reach_vb/status/1924525937443365193).
- **Meta** just released **KernelLLM 8B** on Hugging Face, according to [@reach_vb](https://twitter.com/reach_vb/status/1924478755898085552).
- **The SaharaLabsAI SIWA Testnet** is live, powering scalable compute for their dev platform, according to [@togethercompute](https://twitter.com/togethercompute/status/1924514334044213572).
- **Marin**, an open lab for AI, was built to fulfill the vision of open-source AI with open development, according to [@percyliang](https://twitter.com/percyliang/status/1924527490351169964).
- **MLX models can now be accessed directly from Hugging Face Hub**, according to [@reach_vb](https://twitter.com/reach_vb/status/1924517049474101412).
- [@maximelabonne](https://twitter.com/maximelabonne/status/1924412611430404492) announced that **Qwen3 has been abliterated**, using mini-batching and a hybrid approach with a dictionary and the Minos-v1 classifier from @NousResearch.
- [@HamelHusain](https://twitter.com/HamelHusain/status/1924454532224078220) announced the last day to enroll with a 35% off discount code for his first lecture.
- **Open Molecules 2025 (OMol25)**, a new Density Functional Theory (DFT) dataset for molecular chemistry, and **Meta's Universal Model for Atoms (UMA)**, a machine learning interatomic potential, were released, according to [@AIatMeta](https://twitter.com/AIatMeta/status/1924502785028190366).
- A paper by Tsinghua University researchers detailed HuB, a unified framework to help humanoids handle extreme balancing tasks, according to [@adcock_brett](https://twitter.com/adcock_brett/status/1924133916971020739).

**Humor**

- [@francoisfleuret](https://twitter.com/francoisfleuret/status/1924023253812531606) posted a meme saying **"Me: "I gonna make the AI god machine" Also me:"**.
- [@scottastevenson](https://twitter.com/scottastevenson/status/1924129325382533302) posted an **epic meme of Elon Musk**.
- [@vikhyatk](https://twitter.com/vikhyatk/status/1924104665161183685) shared a meme about **linux user meets a mac user**.
- [@TheTuringPost](https://twitter.com/TheTuringPost/status/1924296119582093752) shared a **"nice human story"** from @Microsoft’s CTO - @kevin_scott about how everyone is now much more capable to build in AI.
- [@teortaxesTex](https://twitter.com/teortaxesTex/status/1924499010565313029) joked about **indigenizing chip supply chains to summon the Sand God to kill commies**.

---

# AI Reddit Recap

## /r/LocalLlama Recap

### 1. Intel Arc Pro GPUs and Project Battlematrix Workstation Launches

- [**Intel launches $299 Arc Pro B50 with 16GB of memory, 'Project Battlematrix' workstations with 24GB Arc Pro B60 GPUs**](https://www.tomshardware.com/pc-components/gpus/intel-launches-usd299-arc-pro-b50-with-16gb-of-memory-project-battlematrix-workstations-with-24gb-arc-pro-b60-gpus) ([Score: 604, Comments: 255](https://www.reddit.com/r/LocalLLaMA/comments/1kq9294/intel_launches_299_arc_pro_b50_with_16gb_of/)): **Intel announced two Arc Pro GPUs: the $299 Arc Pro B50 (16GB VRAM) and the Arc Pro B60 (24GB VRAM, ~$500), targeting professional and AI workstation markets. The B60 is central to 'Project Battlematrix'—Intel's initiative to deliver cost-effective AI workstations—suggesting a strong price/performance proposition in memory-intensive LLM/AI workflows, although these cards are not focused on gaming. See the [original coverage](https://videocardz.com/newz/intel-announces-arc-pro-b60-and-b50-gpus-targeted-at-ai-workstations) for details on specs and positioning.** Commenters highlight the B60's high VRAM/$ value for LLM work and note the longer warranty versus used Nvidia RTX 3090s, acknowledging the cards' lack of gaming suitability but strong potential in AI and professional applications.
    - Multiple users highlight the appeal of the Arc Pro B60's pricing: $500 for 24GB of VRAM is considered a breakthrough compared to used NVIDIA 3090s, which are more expensive, harder to source, and often without warranty. This makes the Intel cards particularly attractive for large language model (LLM) applications and other VRAM-intensive workloads, as these cards are not intended for gaming.
    - There is technical curiosity about whether the Arc Pro B60's 24GB configuration is scalable—one user asks if a hypothetical 2-core version could provide 48GB of VRAM for $1,000. This suggests interest in whether Intel's architecture allows for such modular configurations and pricing.
    - Commenters further note that Intel's entry could expand AI hardware choices, particularly as the 3090's supply declines, potentially driving software and ecosystem improvements for AI beyond gaming (which these cards are reportedly less suited for).
- [**Is Intel Arc GPU with 48GB of memory going to take over for $1k?**](https://www.reddit.com/r/LocalLLaMA/comments/1kqaqmr/is_intel_arc_gpu_with_48gb_of_memory_going_to/) ([Score: 231, Comments: 149](https://www.reddit.com/r/LocalLLaMA/comments/1kqaqmr/is_intel_arc_gpu_with_48gb_of_memory_going_to/)): **Intel has announced the Arc Pro B60 (24GB GDDR6) and B50 (16GB GDDR6) workstation GPUs, with a dual-GPU B60 configuration offering 48GB of total VRAM at an expected price point below $1,000, and the 24GB card around $500 ([VideoCardz](https://videocardz.com/newz/intel-announces-arc-pro-b60-24gb-and-b50-16gb-cards-dual-b60-features-48gb-memory), [WCCFTech](https://wccftech.com/intel-arc-pro-b60-24-gb-b50-16-gb-battlemage-gpus-pro-ai-3x-faster-dual-gpu-variant/)). Positioned for AI and workstation tasks, these cards are less focused on gaming and more aimed at professional and AI workloads, with a significant advantage in VRAM capacity compared to current consumer GPUs at similar price points. Intel has not yet released detailed performance benchmarks, and it is unclear how their drivers stack up for deep learning workloads compared to NVIDIA or AMD alternatives.** Top comments express skepticism about real-world pricing and availability, suspecting that consumer access will be limited and prices may exceed the theoretical MSRPs due to demand outstripping supply. There is technical excitement regarding the memory size but doubt over whether stock or pricing will meet expectations.
    - The 48GB Intel Arc card utilizes all available PCIe lanes (2x8 configuration), which means it maintains full bandwidth when running a dual card setup—this eliminates potential bottlenecks present in cards that use fewer lanes. This could be particularly advantageous for workloads requiring high memory and bandwidth.
    - There is skepticism regarding the projected price of under $1,000 for a 48GB VRAM GPU. Commenters note that if such a GPU were available at this price point, it would be a significant shift in the GPU market; however, they anticipate issues with long-term availability, predicting it will be out of stock and backordered due to high demand and limited initial supply.
    - The abundance of VRAM (48GB) at a potentially lower price is seen as disruptive, especially compared to current Nvidia offerings where comparable VRAM is significantly more expensive. This motivates potential buyers to consider Intel as a competitive alternative in memory-constrained computational or creative tasks, if the pricing and availability materialize as rumored.
- [**Computex: Intel Unveils New GPUs for AI and Workstations**](https://newsroom.intel.com/client-computing/computex-intel-unveils-new-gpus-ai-workstations) ([Score: 155, Comments: 31](https://www.reddit.com/r/LocalLLaMA/comments/1kq8wo4/computex_intel_unveils_new_gpus_for_ai_and/)): **At Computex, Intel announced the Arc Pro B60 GPU with 24GB VRAM (MSRP ~$500) and B50 with 16GB (MSRP $299), targeting AI inference and workstation workloads. Initial availability is through OEM workstation vendors in Q3 2024, with a potential standalone DIY release after software stack maturity in Q4. These cards expand Arc Pro's presence in AI-adjacent workflows, alongside improved software support and the introduction of Gaudi 3 PCIe accelerators for scalable data center AI workloads ([Intel newsroom summary](https://newsroom.intel.com/client-computing/computex-intel-unveils-new-gpus-ai-workstations), [videoCardz spec breakdown](https://videocardz.com/newz/intel-announces-arc-pro-b60-24gb-and-b50-16gb-cards-dual-b60-features-48gb-memory)).** Top technical concerns from commenters include skepticism about DIY pricing and availability, the urgent need for improved Intel GPU software/driver maturity (citing subpar support versus NVIDIA), and suggestions to unify or more frequently update Intel's IPEX-LLM and OpenVINO frameworks for broader AI framework compatibility.
    - The Arc Pro B50 (16GB) and B60 (24GB) are launching in Q3 2024 at $299 and ~$500 respectively, targeting workstation OEMs first. There's uncertainty around DIY availability, as a retail launch may depend on initial commercial adoption and further software optimization (possibly Q4+). [(Source)](https://videocardz.com/newz/intel-announces-arc-pro-b60-24gb-and-b50-16gb-cards-dual-b60-features-48gb-memory)
    - Technical commenters highlight that the Arc Pro B60's 24GB VRAM is a standout at this price tier—approximately half the cost of an AMD 7900XTX, and just 1/5th the price of Nvidia's RTX 4090. Reported LLM inference speed is ~35T/s on Qwen2 7B Q8; although the cards may have slow absolute inference on demanding diffusion models (e.g., Flux dev, HiDream Q8), their large memory enables running bigger models not typically possible on consumer GPUs at this segment.
    - A significant bottleneck identified is the 450GB/s memory bandwidth, noticeably lower than competing chips like Apple's M4 Max (~550GB/s). While the lower bandwidth may be offset by differences in compute performance among these architectures, it's flagged as a consideration for workload suitability.

### 2. Offline and Open Source AI Productivity and Speech Tools (Clara, Kokoro-JS, OuteTTS)

- [**Clara — A fully offline, Modular AI workspace (LLMs + Agents + Automation + Image Gen)**](https://i.redd.it/u6niruxjqo1f1.png) ([Score: 433, Comments: 118](https://www.reddit.com/r/LocalLLaMA/comments/1kq590b/clara_a_fully_offline_modular_ai_workspace_llms/)): **The image showcases the UI of Clara, an open-source, fully offline, modular AI workspace aiming to unify LLMs (via Ollama or OpenAI APIs), agents, automation (via n8n integration), and local image generation (Stable Diffusion/ComfyUI). The interface emphasizes a drag-and-drop dashboard paradigm, with users able to add widgets for quick chat, email inbox, agent logic, code execution, and more, enabling a customizable 'AI control room' without reliance on cloud services or API keys. The modern design and widget-based architecture underscore extensibility and user-centric workflow integration, released under the permissive MIT license.** A technical concern was raised about Windows Defender reporting the GitHub executable as a virus, suggesting potential issues with distribution or false positives. There is also clear interest in open access to the repository, with requests for a repo link, and appreciation for the permissive licensing for an advanced local tool.
    - A user reported that Windows Defender flagged the provided executable as a virus, which could raise concerns among security-minded users about false positives or the need for code signing and transparent build processes.
    - There's positive mention of the modular API integration, but it's noted that the current UI only allows adding a single API key at a time. For example, a user could add an OpenAI key, but reported no apparent way to add multiple providers (e.g., Anthropic, Gemini) simultaneously.
    - An insightful point highlights that Clara integrates with n8n, a highly-regarded workflow automation tool, directly within the workspace. This significantly boosts its automation and agent capabilities compared to standard LLM or image gen apps.
- [**Unlimited text-to-speech using Kokoro-JS, 100% local, 100% open source**](https://streaming-kokoro.glitch.me/) ([Score: 159, Comments: 36](https://www.reddit.com/r/LocalLLaMA/comments/1kpw9nw/unlimited_texttospeech_using_kokorojs_100_local/)): **Kokoro-JS is an open source, client-side text-to-speech system that runs entirely in the browser using the ONNX-format Kokoro-82M-v1.0 model (~300MB). The JS implementation leverages local resources (WebGPU/WebGL) for inference, enabling offline speech synthesis without any server interaction; voice selection and multiple voices are supported, though on Firefox users must manually enable** `dom.webgpu.enabled` **and saving audio to disk currently has limitations. Full source and demo are available ([GitHub](https://github.com/rhulha/StreamingKokoroJS), [Demo](https://rhulha.github.io/StreamingKokoroJS/)).** Comments discuss whether this is the same Kokoro model as used in open-webui, but no definitive technical answer is provided; browser/WebGPU compatibility and offline privacy are highlighted as key user benefits.
    - The Kokoro-JS implementation operates 100% locally by downloading a ~300MB AI model that runs entirely in the user's browser, with no server-side data involved. The codebase is open source and available on [GitHub](https://github.com/rhulha/StreamingKokoroJS), and there are demo sites including a [Glitch project](https://glitch.com/edit/#!/streaming-kokoro), which allow users to test the streaming TTS functionality directly.
    - Important technical updates for browser support include enabling `dom.webgpu.enabled = true` and `dom.webgpu.workers.enabled = true` in Firefox's `about:config` for WebGPU support; the developer notes that saving generated audio to disk does not currently work in Firefox, likely due to browser limitations on file handling or WebGPU integration.
    - A user asks whether this Kokoro-JS version is the same as used in Open WebUI, suggesting ongoing interest in portability and compatibility between web-based TTS model deployments. Another user compares the local TTS process to OpenAI Whisper-based transcription models and mentions other mobile applications, highlighting a broader trend toward fully client-side AI model execution for both speech synthesis and recognition.
- [**OuteTTS 1.0 (0.6B) — Apache 2.0, Batch Inference (~0.1–0.02 RTF)**](https://huggingface.co/OuteAI/OuteTTS-1.0-0.6B) ([Score: 122, Comments: 30](https://www.reddit.com/r/LocalLLaMA/comments/1kq6ysz/outetts_10_06b_apache_20_batch_inference_01002_rtf/)): **OuteTTS-1.0-0.6B is a 0.6B parameter multilingual TTS model based on Qwen-3 (LLM architecture), released under Apache 2.0, and optimized for efficient batch inference with ~0.1–0.02 RTF on a single NVIDIA L40S (see benchmarks: e.g., 32→0.05 RTF with vLLM FP8, 32→0.106 with EXL2 6bpw, details [here](https://huggingface.co/OuteAI/OuteTTS-1.0-0.6B)). The release includes Python (**`outetts` **v0.4.2) support for multiple inference backends—vLLM, EXL2, llama.cpp—with features like continuous batching, external-URL model serving, and speaker reference-based voice synthesis/voice cloning. Sampling parameters (e.g., repetition penalty 1.1 over last 64 tokens, temperature 0.4, top-k 40, top-p 0.9) and use of the ibm-research/DAC.speech.v1.0 codec are critical for high-fidelity output, and the model trains on diverse multilingual corpora (MLS, Common Voice).** The primary technical question from the comments concerns how a TTS model was derived from Qwen3 (an LLM), requesting paper or methodology details. There's interest in demo audio and comparative evaluation with prior OuteTTS versions, as well as clarification on the specific models deployed on public playgrounds.
    - A user asks about the technical foundation of OuteTTS 1.0, specifically questioning how a TTS (text-to-speech) model could be built on Qwen3, which is primarily an LLM (Large Language Model). They inquire about available papers or technical details, indicating a need for clarification on model architecture, adaptation of Qwen3 for TTS, and reproducibility information.
    - Another technical discussion references the '2cent-tts' project, which leverages a much smaller (60M parameter) Qwen3 model to achieve faster inference. This approach uses phoneme inputs and a SNAC decoder to optimize performance, suggesting potential strategies for efficient TTS, and inviting comparison with OuteTTS in terms of parameter count, speed, and architecture.
    - There is also an inquiry about the capabilities of the model, specifically regarding voice cloning. This raises questions about the model's support for speaker adaptation or sample-based voice replication, which are important technical features for many TTS applications.

### 3. ParScale Model Launch and Parallel Scaling Paper

- [**Qwen released new paper and model: ParScale, ParScale-1.8B-(P1-P8)**](https://i.redd.it/7q0xsc86um1f1.png) ([Score: 440, Comments: 66](https://www.reddit.com/r/LocalLLaMA/comments/1kpyn8g/qwen_released_new_paper_and_model_parscale/)): **The image summarizes the Qwen team's new ParScale model and paper, introducing a parallel scaling method for transformers. ParScale trains/inferences with P parallel streams and their analysis suggests scaling with P streams is theoretically comparable to increasing parameter count by O(log P) (i.e., doubling streams approximates adding a constant multiple of parameters). Visuals in the image contrast ParScale's approach with traditional dense models and Mixture-of-Experts (MoE), noting ParScale's improved log-scaling efficiency, reasoning performance, and universal applicability ("plugin" capability for various models). A results table in the image quantifies improved or competitive performance versus parameter-only scaling.** Comments highlight community excitement, noting ParScale avoids MoE downsides (memory/computation inefficiency, complex routing), and speculate it could generalize to many models with minimal retraining. There is a technical distinction drawn versus MoE: ParScale utilizes parallel computation over repeated parameters, rather than selective expert activation.
    - Key stats from the discussion: ParScale is claimed to offer up to `22x` less memory increase and `6x` less latency increase compared to alternatives—though one commenter criticizes the statistical phrasing and suggests expressing it as a reduction to `4.5%` (for memory) and `16.7%` (for latency).
    - Technically, ParScale is contrasted to Mixture of Experts (MoE): MoE 'stores a lot, computes a little by being selective,' whereas ParScale 'stores a little, computes a lot (in parallel) by being repetitive with variation,' focusing on parallelism and parameter efficiency.
    - A point is raised that ParScale may be a generalizable technique for reducing compute and memory during scaling, potentially applicable to any model with only modest finetuning, which could have wide-ranging impact if validated.

## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. Major AI-Driven Corporate Layoffs and Workforce Restructuring

- [**The AI layoffs begin**](https://i.redd.it/qqiilvmesr1f1.png) ([Score: 187, Comments: 35](https://www.reddit.com/r/singularity/comments/1kqgxpm/the_ai_layoffs_begin/)): **The shared image summarizes recent layoffs at major tech firms (Microsoft, Google, PwC, Salesforce, Meta, Duolingo, Dell, etc.), emphasizing that these are either directly connected to, or justified by, AI-focused restructuring (e.g., shifts to AI-first strategies or elimination of certain roles perceived as less relevant in an AI-driven context). The image quantifies layoffs by workforce percentage or number, and details affected departments (e.g., marketing, sales, non-technical roles), highlighting industry stability concerns as AI integration accelerates. Some firms cited are shifting resources (like Duolingo investing in AI infrastructure or Dell planning automation via AI) as rationales for layoffs.** Top comments argue that in some cases (e.g., Google's layoff of ~200 salespeople), the layoffs are overstated as AI-driven, suggesting broader economic and business cycle factors may play a more significant role. There's also skepticism about the narrative of mass AI-driven layoffs, with some suggesting tech layoffs are annual occurrences followed by re-hiring, and the full impact of AI on job displacement is yet to be observed.
    - Discussion highlights skepticism about attributing recent layoffs at major tech companies like Alphabet (Google) solely to AI, with one commenter noting only ~200 sales jobs were cut, not roles directly affected by AI automation.
    - There is nuanced debate on whether observed layoffs are causally linked to AI advancements, versus other macroeconomic factors such as interest rates and tariffs, or whether companies are using "AI" layoffs as cover for typical annual workforce adjustments or strategic cost-cutting.
    - Some participants point out that the tech industry often goes through cycles of layoffs and re-hiring, suggesting that current developments may not represent a significant AI-driven inflection point, but rather ongoing business process optimization with varying external rationale given to investors.
- [**The AI layoffs begin**](https://i.redd.it/gadex0zasr1f1.png) ([Score: 322, Comments: 118](https://www.reddit.com/r/OpenAI/comments/1kqgx8i/the_ai_layoffs_begin/)): **The image compiles recent layoff data from major tech and consulting firms (Microsoft, Google, PwC, Salesforce, Meta, Chegg, HP, IBM, Duolingo, Dell, and Klarna), attributing much of the workforce reduction to restructuring around AI adoption and efficiency improvements. Figures for percentages or absolute numbers of layoffs are presented alongside company leaders and sectors impacted, aiming to contextualize perceived industry-wide shifts caused by AI integration and related strategic pivots. Layoff justifications range from direct AI integration to broader post-pandemic adjustments and company-specific restructures.** Commenters raise skepticism, noting that the layoff justification as AI-driven is often more of a positive investor narrative than a demonstration of clear AI impact, with real labor market signals (job postings, hiring rates, unemployment) not yet strongly reflecting an AI-led job reduction. Several point out possible narrative cherry-picking in attributing layoffs (like Intel's) to AI-related causes without substantive evidence.
    - Discussion highlights skepticism about attributing layoffs solely to AI, suggesting the narrative may be shaped for investor relations rather than grounded in technical efficiency improvements. Commenters note that concrete evidence—such as declines in technical job postings (e.g., customer service, software development) or rising unemployment in tech sectors—has not yet emerged, emphasizing the need for rigorous labor market data analysis.
    - One user shares a negative experience with ChatGPT generating API code that was several major versions out of date, indicating practical limitations in current generative models for up-to-date coding tasks and version handling. The comment underscores insufficient disclosure of model context and the need for LLMs to better identify and communicate codebase versions used in completions.
    - A cited case references Klarna reportedly rehiring after laying off 700 employees for AI automation, highlighting the operational reality that claimed AI-driven efficiencies sometimes fail to materialize, leading to reinstatement of human roles. This points to implementation challenges in replacing complex workflows with current AI tooling.
- [**The AI layoffs begin**](https://i.redd.it/9z1kxvub6r1f1.png) ([Score: 461, Comments: 132](https://www.reddit.com/r/ChatGPT/comments/1kqdukp/the_ai_layoffs_begin/)): **The infographic [The AI Layoffs Begin](https://i.redd.it/9z1kxvub6r1f1.png) visually details major recent layoffs at leading tech firms such as Microsoft, Google, PwC, and Meta, correlating each company's downsizing (with explicit layoff figures and executive portraits) to their stated adoption or realignment toward AI-driven strategies. The image suggests a trend where large-scale workforce reductions are framed as outcomes of AI integration or automation initiatives, underlining an industry shift toward leveraging artificial intelligence for operational efficiency.** Top comments challenge the narrative of layoffs being solely attributable to AI, noting that firms like Microsoft have a history of recurring layoffs unrelated to AI advancements, and highlighting other factors such as failed business ventures (e.g., Meta's bet on VR/AR). There's also skepticism about the net employment effect, with users asking for comparative hiring data to understand the true workforce impact.
    - Commenters highlight that attributing recent layoffs strictly to AI oversimplifies the issue; for example, Meta's job cuts are tied to its costly bet on VR/AR and the Metaverse rather than direct AI replacement, illustrating that strategic business decisions and shifts in focus remain primary drivers.
    - One user requests more granular statistics—specifically, not just layoff counts but net tech employment figures over the same period, as well as data on hiring to better contextualize layoffs at major companies like Microsoft and Dell.
    - Another technical point raised is the scale of impact, citing Dell's reported layoffs of 12,000 employees, which underscores how workforce reductions can be significant in hardware-centric sectors and not just in software or AI development teams.

### 2. Upcoming and Spotted AI Model Releases (Gemini, Claude, o1-pro)

- [**2.5 pro deepthink?**](https://i.redd.it/qm30m4bhup1f1.jpeg) ([Score: 259, Comments: 32](https://www.reddit.com/r/singularity/comments/1kq8rza/25_pro_deepthink/)): **The attached screenshot shows a tweet revealing the presence of a new Google model variant, 'Gemini-2.5-pro-deepthink,' along with a related 'gemini-2.5-pro-exp' variant, as seen in the Google Cloud API console. This suggests Google is experimenting with or releasing a version of Gemini 2.5 Pro focused on deeper reasoning or extended context handling, paralleling recent industry moves towards models that offer more advanced reasoning capabilities (such as OpenAI's 'o3-pro').** Commenters compare 'deepthink' to OpenAI's models and note that prior versions of 2.5 Pro did not substantially improve reasoning, expressing hope that 'deepthink' targets this gap.
    - One user comments on Gemini (likely referring to the 2.5 Pro model), noting its strong capabilities but criticizing its tendency to lose track of prompt instructions and occasionally repeat or paste in older text. This suggests lingering context window or attention management limitations, which are critical in production usage for maintaining coherent, goal-oriented dialogue over extended interactions.
    - There is a comparison to Google's o3-pro, implying that the model under discussion may be similar in quality or approach to Google's leading mid-tier language model. This draws a parallel between major competitors' offerings in the 'pro' tier segment, hinting at market convergence on specific model capabilities.
- [**o1-pro just got nuked**](https://www.reddit.com/r/OpenAI/comments/1kq5wc5/o1pro_just_got_nuked/) ([Score: 161, Comments: 82](https://www.reddit.com/r/OpenAI/comments/1kq5wc5/o1pro_just_got_nuked/)): **The post reports that the performance and code-generation capabilities of OpenAI's o1-pro model (formerly considered a top performer for complex coding tasks, at ~$200/mo) were drastically reduced in the past few days, yielding short, low-detail responses and apparent code output suppression—suggesting the addition of new filters. Notably, o1-pro had already been marked as legacy following the release of o3, hinting this is likely part of deprecation and resource reallocation for upcoming releases or to support Codex SWE needs. No official communication regarding these changes was provided to users, despite the significant impact and high subscription cost.** Top technical comments highlight: (1) o1-pro's deprecation status was known internally after o3's launch, so the downgrade aligns with planned sunset and resource allocation strategies; (2) a major concern is the lack of user transparency, especially as users relied on o1-pro for bug detection where other models failed; (3) a cynical view compares this to typical 'enshittification' seen in tech platforms—downgrading services post-market capture to boost margins.
    - Commenters note that o1-pro was marked as legacy and deprecated after the release of o3, with the recent shutdown seen as part of a resource reallocation, possibly to focus on the needs of Codex SWE or future launches.
    - A technically relevant discussion points out that o1-pro underwent significant downgrades ('nerfs') post-o3 release, including shortened 'thinking time' and reduced maximum response length, resulting in noticeably degraded output quality before its discontinuation.
    - Another comment highlights the value of code generation tools that are context-aware, such as Copilot, which operates directly within IDEs and is specialized for code completion tasks, contrasting this with the generalist approach and higher costs associated with chat-based AI systems like ChatGPT Pro.
- [**Amanda Askell is the woman in charge of giving Claude its personality, so I am ever grateful to her**](https://i.redd.it/gfagz8bkom1f1.png) ([Score: 266, Comments: 44](https://www.reddit.com/r/ClaudeAI/comments/1kpy0me/amanda_askell_is_the_woman_in_charge_of_giving/)): **The image is a screenshot of a tweet from Amanda Askell, a key figure at Anthropic responsible for Claude's personality and alignment, humorously describing her position between 'safety skeptics' and 'capability deniers.' The tweet reflects on her role managing the trade-off between AI safety (preventing harm/censorship) and capability (advancing what Claude can do), two core tension points in AI deployment, especially for commercial language models. This highlights the ongoing debate around responsible scaling, alignment, and external/internal pressures faced by AI companies like Anthropic.** Commenters highlight a skeptical view toward both corporate motives (seeing the balancing act as PR rather than true stewardship) and the impact of safety constraints, with some likening safety alignment to 'muzzling' and critiquing the perceived neutrality of such roles. Others downplay the post as a simple reference to balancing interests without deep technical implication.
    - A commenter references Anthropic's own research and internal papers (like those co-authored by Kyle Fish), which suggest options such as allowing models like Claude to terminate conversations as a gesture toward addressing potential moral patienthood or welfare. The critique is that, despite discussing these ethical safeguards in published literature, Anthropic has not implemented such features—even when similar functionality (such as a 'stop button') existed for Bing AI in 2023, raising questions about the company's alignment between public discourse, research, and actual product features.
    - There is technical skepticism about Anthropic's approach to AI safety and positioning: a commenter analogizes the framing of 'safety skeptics to the left, capability deniers to the right, stuck in the middle' as a rhetorical device that masks corporate censorship and limited transparency around Claude's actual capabilities and constraints. This meta-level critique addresses the gap between professed neutrality and the operational, enforced limitations on the model.

### 3. AI Progress, Automation Impact, and SWEs Replacement Discourse

- [**Timeline of SWEs replacement**](https://i.redd.it/apqpgkow2q1f1.jpeg) ([Score: 680, Comments: 206](https://www.reddit.com/r/singularity/comments/1kq942u/timeline_of_swes_replacement/)): **The image is a satirical timeline showing the recurring hype cycles around technologies that promise to replace software engineers, listing examples like COBOL, SQL, Visual Programming, MDA, No-Code, and now AI-based 'Vibe Coding,' with each iteration ultimately failing to remove the need for software engineering expertise. The technical context provided by commenters notes that while each technology boosted developer productivity, market dynamics have shifted: initially, growing computer adoption enabled these improvements to absorb new demand, but today market saturation and the advent of neural networks pose new challenges.** Commenters highlight that using past failed predictions to dismiss new technologies is a logical fallacy—*the merits of each wave of technology should be judged independently*. Some also suggest the transformative potential of neural networks is fundamentally different, implying a need for nuanced evaluation of current trends.
    - Several commenters compare historic boosts in developer productivity (COBOL, SQL, VBA) with modern advancements, noting past tools transformed the development process but typically expanded the market rather than shrinking opportunities for software engineers. However, they stress that neural networks and AI introduce fundamentally different dynamics; unlike past tools, AI can potentially *replace* manual coding entirely, not just accelerate it, suggesting a stronger long-term impact on the job market.
    - A discussion highlights that while 'no code' platforms are often cited as modern productivity enhancers, specific technical advances like modern web tooling and JavaScript on the server have arguably had a greater effect on developer efficiency. This is contrasted with AI, which is positioned as qualitatively different, since it introduces cognitive capabilities into tooling, rather than just new abstractions or frameworks.
    - Economic analysis emerges, with several commenters focusing on the current saturation of the computer market and the shifting allocation of capital. They observe that prior productivity improvements happened in expanding markets, whereas AI-driven change is happening post-saturation, possibly intensifying job displacement concerns and accelerating disruptive effects.
- [**The moment I realized AI could code better than me**](https://www.reddit.com/r/ChatGPT/comments/1kq8t4t/the_moment_i_realized_ai_could_code_better_than_me/) ([Score: 949, Comments: 282](https://www.reddit.com/r/ChatGPT/comments/1kq8t4t/the_moment_i_realized_ai_could_code_better_than_me/)): **The OP describes a practical use case where they leveraged an AI coding assistant to debug and refactor a software function, achieving a solution that was both cleaner and more efficient than their own version. This anecdote highlights current AI models' capabilities in code analysis, debugging, and refactoring, demonstrating productivity enhancements in software development workflows. Top comments provide parallel examples of AI utility in natural language processing, such as extracting semantic structure and context from error-ridden meeting transcripts—particularly, AI resolving inconsistent name spellings, indicating significant advances in context-aware NLP models. There is also recognition of limitations, e.g., AI struggling with complex bug fix loops and outputting "gibberish code" in edge cases, reflecting ongoing boundaries in AI coding reliability.** Commenters note impressive AI interpretability in NLP and programming but express caution: AI-generated solutions often fail in complex debugging scenarios, necessitating manual code review. This points to a consensus that AI excels at certain types of automation but still requires human oversight for more nuanced or context-dependent code issues.
    - ChatGPT demonstrated strong natural language understanding by extracting accurate meeting outcomes and participant roles from a noisy, error-prone Zoom transcript, including consistently inferring and normalizing a misspelled artist's name without additional context—suggesting advanced contextual inference and entity resolution capabilities beyond surface-level parsing.
    - While AI can generate impressive coding solutions, it often falters during iterative bugfixing: users report getting stuck in unproductive feedback loops where the AI's solutions repeatedly fail to resolve underlying issues, revealing limits in model reasoning and bug diagnosis, particularly with complex or ambiguous code.
    - AI's most common coding mistakes are attributed to model limitations in *context management* and *attention mechanism* constraints, rather than fundamental programming errors. However, in optimal scenarios where the prompt and code fit within the model's attention window, the resulting solutions can exceed expectations, highlighting unpredictability in quality tied to current architectural constraints.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Flash Preview
> 

**Theme 1. AI Agent Development and Orchestration Tools**

- **MCP Protocol Connects Diverse AI Agents**: Discussions highlighted the **Model Context Protocol (MCP)** enabling communication between AI agents, even across different machines, with *Qwen 3 235B* reportedly supporting it natively. Use cases range from Discord bots to complex multi-agent coordination, supported by [this agentapi repo](https://github.com/coder/agentapi) and new [MCPControl server v0.2.0](https://github.com/Cheffromspace/MCPControl/releases/tag/v0.2.0) with SSE support.
- **DSPy and LlamaIndex Build Agent Workflows**: Engineers are leveraging frameworks like **DSPy** and **LlamaIndex** for building agents, with **DSPy 2.6** replacing `Suggest`/`Assert` with `BestOfN`/`Refine` ([tutorial](https://dspy.ai/tutorials/output_refinement/best-of-n-and-refine)). **LlamaIndex Agents** now feature improved [long-term and short-term memory](https://www.llamaindex.ai/blog/improved-long-and-short-term-memory-for-llamaindex-agents) and support for [multi-agent workflows with Weaviate](https://docs.llamaindex.ai/en/stable/examples/agent/multi_agent_workflow_with_weaviate_queryagent).
- **New SDKs and Interfaces Boost Agent Capabilities**: Amazon released the [**Strands Agents SDK**](https://aws.amazon.com/blogs/opensource/introducing-strands-agents-an-open-source-ai-agents-sdk/) to streamline agent creation, while **Sherlog Canvas (Alpha)** provides an [AI-powered debugging interface](https://github.com/GetSherlog/Canvas) integrating MCP-powered cells for logs and metrics with a [demo video](https://youtu.be/80c5J3zAZ5c). A new [MCP UI SDK](https://x.com/idosal1/status/1923477857881190718) also adds rich web interaction capabilities to MCP servers.

**Theme 2. LLM Performance, Evaluation, and Model Behavior**

- **Gemini Models Face Scrutiny Over Performance and Quirks**: Users observed strange behaviors like commenting out code and removing comments in **Gemini** models and noted a trade-off where **Gemini 2.5 Pro 0506** is better for coding but older versions (like **03-25**) are better for math, citing [official Google numbers](https://deepmind.google/technologies/gemini/pro/) showing performance at **83% vs 88.9%**. The deprecation of **Gemini 2.5 Pro Experimental** also caused dissatisfaction due to filtering issues in the newer versions.
- **GPT/O Series Spark Speculation on Architecture and Release**: Speculation is rife that **GPT-5** might drop the **o3** component, adopting a structure similar to **Gemini 2.5 Pro** with a combined LLM and reasoning model, potentially arriving this summer. The continued delay of **O3 Pro** sparked frustration, with some users feeling the original **GPT-4** had a *genuine intelligence/creativity spark* that newer models lack.
- **Benchmarking and Evaluation Techniques Gain Focus**: The **AgentX Competition** offers over [$150K](https://x.com/dawnsongtweets/status/1924470174776004875) in prizes and requires participants to use their own **OpenAI API keys** for labs, prompting discussions on [lm eval harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage) for customized model evaluation, particularly for quantized models like **SpinQuant Llama-2-7b**. Benchmarks like **GSM8K** and **MATH** showed performance improvements for small models trained on **Reasoning Gym** data, with **Qwen 2.5 3B** outperforming **Llama 3.2 3B** in some tasks.

**Theme 3. Hardware Performance and Low-Level Optimization**

- **GPU Hardware Battles On: VRAM, Efficiency, and New Players**: The [**Intel Arc Pro B60**](https://www.intel.com/content/www/us/en/products/sku/243916/intel-arc-pro-b60-graphics/specifications.html) excited users with its **96GB VRAM** and sub-$2000 price point for local LLMs despite software concerns, while Macbooks were praised for running a **32B 4-bit model for >5 hours** silently with <2GB RAM usage. Enabling **Resizable BAR** on a **9070XT** GPU drastically boosted LM Studio performance from *8.52 t/s* to *131.91 t/s* on a **Gemma3 4b QAT** model.
- **Triton, CUDA, and ROCm Face Implementation Challenges**: Users are struggling with integrating **FSDP** and **Flash Attention 2** with `trl` due to incompatibility with models not initialized on the GPU, and debugging CUDA errors like *unspecified launch failure* and *illegal memory access* in neural net mutation functions ([godbolt link](https://cuda.godbolt.org/z/z8z6a85vP)). In **ROCm Triton**, debates arose over the `kpack` argument's impact on performance, with Torch Inductor defaulting it to **2** for **MFMA** with `ld128`.
- **Optimizing Performance Through Quantization and Kernels**: Discussions in the GPU Mode Discord explored the performance of **FP8-MM** and **Mixture-of-Experts** on **MI300** hardware with users achieving impressive timings down to **150 µs** and **9.64 ms** respectively on leaderboards. Users are also experimenting with **Quantization Aware Training (QAT)** with models like **Llama3.2 3B** using [Axolotl configs](https://github.com/axolotl-ai-cloud/axolotl/pull/2590/files#diff-29a7a53548f024812e8a2dc36eccc625c6b278b22b105f4eb5a924c63452a781) and exploring **CuTeDSL** ([blogpost](https://veitner.bearblog.dev/bridging-math-and-code-cute-layout-algebra-in-cutedsl/)) from the CUTLASS team for low-level kernel optimization.

**Theme 4. New AI Models, Research, and Emerging Concepts**

- **Novel AI Concepts Emerge: Self-Teaching Reasoners and World Models**: A new AI called **Absolute Zero Reasoner (AZR)** teaches itself from scratch without human data ([YouTube Short](https://youtube.com/shorts/avnHiKcOEQA?si=NWoqwZR1IcPxyrG0)), while **OpenWorldLabs** focuses on building [world models for robots and video games](https://openworldlabs.ai/), although some users found their mission unclear. Discussions touched upon resolving **accumulating error** in diffusion models by renoising context frames, noting existing video models are too slow but Google might train with error to denoise frame by frame.
- **LLMs Develop Spontaneous Social Conventions and Biases**: A study ([arxiv link](https://arxiv.org/abs/2505.10475), [science link](https://www.science.org/doi/epdf/10.1126/sciadv.adu9368)) revealed that **universally adopted social conventions** can spontaneously emerge in decentralized LLM populations through local interactions, leading to strong collective biases even without initial biases in individual agents. Notably, committed minority groups of **adversarial LLM agents** can drive social change by imposing alternative conventions.
- **Specific Research Areas Explored: Document AI, MCMC, and Empowerment**: A Data Scientist is focusing on **Document AI and NLP** including representation, TSR, and quality assessment, and personal projects on finance and ethical dilemmas. Research into **power consumption for MCMC** algorithms ([paper example](https://arxiv.org/pdf/1903.11714)) and the potential of probabilistic bit (**pbit**) hardware for efficiency was discussed, alongside the underexplored concept of **empowerment** in AI, particularly relevant to **Muon**'s expensive operations.

**Theme 5. AI Tooling and Platform Updates**

- **NotebookLM Launches Mobile App and Video Uploads**: The **NotebookLM mobile app** arrived on [iOS](https://apps.apple.com/us/app/google-notebooklm/id6737527615) and Android with an **MVP feature set** but lacks core features like mind maps and briefing notes, while the web version now supports [video uploads with automatic transcription](https://x.com/testingcatalog/status/1924246774346104863). Users noted the writing style has become *strangely simplistic and reductive*, resembling **high-school-essay quality**, and reported issues uploading materials related to social justice, suspecting *censorship*.
- **OpenRouter Navigates API Changes and Provider Issues**: Google is deprecating **Gemini 2.5 Pro Experimental** (`google/gemini-2.5-pro-exp-03-25`) on OpenRouter in favor of a paid endpoint, while the free **DeepSeek V3 0324** is undergoing maintenance. Users reported that the *Kluster* provider for **Qwen3 235B** prematurely ends tool calls, requiring OpenRouter to switch providers to resolve the issue.
- **Hugging Face and Related Tools See Updates**: **LlamaIndex** announced **LlamaParse** updates ([tweet](https://twitter.com/llama_index/status/1923510823164706925)) and first-class support in [Azure AI Foundry Agent Service](https://twitter.com/llama_index/status/1924502129974411504), while Vitalops released [datatune](https://github.com/vitalops/datatune), an open-source tool for data transformations via natural language. The [AI Engineer Pack link for 6 months of free Hugging Face Pro](https://discord.com/channels/879548962464493619/1353741359059570699) is reportedly not working for some users.

---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **You.com's Win Rate Misinterpretation**: A member shared a [Cunnyx link](https://cunnyx.com/youdotcom/status/1923100692098453703) clarifying that percentages displayed on **You.com** represent win rate, not quality.
   - The discussion highlighted the potential for misinterpreting **You.com** metrics.
- **MCP SuperAssistant Plays Well with Perplexity**: A user reported that the **MCP SuperAssistant** extension functions well within Perplexity across multiple servers, with occasional disconnections, and shared links to the [MCP SuperAssistant website](https://mcpsuperassistant.ai) and the [GitHub repository](https://github.com/srbhptl39/MCP-SuperAssistant).
   - The community explored the utility of integrating external assistants with **Perplexity**.
- **Grok Deepsearch Edges Out Perplexity Pro**: One user expressed a preference for **Grok Deepsearch** over **Perplexity Pro** for specific research tasks, citing superior results in comparing food prices across countries using Python math.
   - Despite using **Perplexity Pro**, the user found the **Python math tool** absent in Perplexity's research tasks, which led them to favor **Grok**.
- **Firefox Hypes Up Perplexity**: [Firefox will promote the Perplexity search engine](https://windowsreport.com/mozilla-firefox-to-promote-perplexity-search-engine/), raising questions about its adoption on Android and sparking debates about browser engine preferences (**Blink** vs. **Gecko**).
   - The announcement triggered discussions about the future of **Firefox** and its competitive positioning against **Blink**-based browsers.
- **Perplexity Sonar API Diverges Wildly From UI**: Users reported significantly different results between the **Perplexity Sonar API** and the **UI version**, noting discrepancies in source information and difficulties in replicating **UI** results via the **API**, including a [GitHub repo](https://github.com/alanbuxton/perplexity-api-tester) used for testing.
   - The discussion questioned the consistency and configurability of the **Sonar API**, and whether parameters like `top-p` affect the actual search.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Convex KOs Supabase for Real-Time**: While **Supabase** offers real-time capabilities, **Convex** is generally favored for real-time apps due to its automatic synchronization and event-driven architecture, see [this comparison](https://x.com/mntruell/status/1924297275993669893).
   - One member noted **Supabase's** local hosting benefit but admitted **Convex** excels in real-time, but its Auth, Edge Functions and Buckets are *fire as sheet*.
- **MCP Mania Grips Agent Communication**: Discussions highlighted using **MCP** (Model Context Protocol) for AI agents, even across different computers, with the observation that *Qwen 3 235B came with native MCP support because of this reason*.
   - Use cases ranged from Discord admin bots to complex inter-agent coordination with single source of truth and context, with [this repo](https://github.com/coder/agentapi) supporting use cases.
- **DeepSeek-R1T-Chimera Masters Markdown**: The **DeepSeek-R1T-Chimera** model was lauded for precise editing of .md and .mdc files and breaking loops in prompt testing, and can be found on [HuggingFace](https://huggingface.co/tngtech/DeepSeek-R1T-Chimera).
   - Notably, it's reportedly the only free model to achieve this level of accuracy, as it is fine-tuned between R1 and V3.
- **Cursor Code Context Crushing!**: Users reported slow requests and inconsistent quality with **Cursor**, prompting frustration, with one stating, *I really hate what is happening here*.
   - Suggestions included switching models (like DeepSeek 3.1) or resetting context; some encountered a bug where the 'Apply All' button doesn't display when using Gemini on max settings.
- **Navigating Narrative Niftily in Cursor**: Members explained **Cursor** reads linked documentation by parsing the HTML content of the linked page to gather info, also pointing out using the [Cursor docs system](https://docs.cursor.com/context/model-context-protocol) is better for keeping all of your code.
   - Users requested dynamic reading of more pages from linked documents, mirroring browser functionality, with the team announcing an upcoming API plugin for sending DOM to Cursor.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth's imatrix dataset generates excitement**: Members are excited about **UnSloth Dynamic 2.0 GGUFs** which are centered around the *paradigm shifting, instruction and conversational focused imatrix* calibration dataset.
   - The improved perplexity translates to faster token generation for **Llama-3.1-8B-Instruct-UD-Q8_0_XL**, according to reports.
- **Qwen3 GGUF prompts launch investigation**: Users reported issues running **Qwen3 235 128k UD-Q6** and **UD-Q8 ggufs** due to SHA hash mismatches and out-of-bounds errors.
   - The team responded that they will investigate and have *updated all qwen3 uploads with the chat template fixes*.
- **Colab downloads cause concern**: A member complained about downloading adapters from **Google Colab** being slow, and considered switching services.
   - Another member confirmed this problem and suggested uploading to **Hugging Face** for faster downloads.
- **Torch/Cuda errors hit Colab Users**: A user reported getting **CUDA errors** after updating Unsloth and it was traced to an older driver requiring a specific torch version.
   - The user implemented a self-curing [solution](https://www.urbandictionary.com/define.php?term=self-curing) by using `pip install "unsloth[cu121-torch250] @ git+https://github.com/unslothai/unsloth.git"`.
- **PTS Fuels DPO Performance**: Members discussed that [Pivotal Token Search](https://huggingface.co/blog/codelion/pts) isn't RL itself, but it generates DPO pairs for fine-tuning, citing improvements shown in the **Phi-4 technical report**.
   - A screenshot ([Image Link](https://cdn.discordapp.com/attachments/1257011997250424842/1373218344232157264/Screenshot_2025-05-17_at_4.37.39_PM.png?ex=682ce87e&is=682b96fe&hm=685beeb66d15e5f4e1195735b7a8b1cc7fb4145dc5dfc8f9cd2863e&)) visually confirmed **PTS's positive impact on DPO performance** across benchmarks.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ASI Lab Plagiarism Scandal Erupts**: **ASI Lab** is under fire for alleged plagiarism, with [pro.creations](https://cdn.discordapp.com/attachments/998381918976479273/1373015169357054082/image.png?ex=682cd405&is=682b8285&hm=41d43e15460edc8f88272551f18cfec1fa74fb94c748b9efa15f5c82234bb031&) reporting that their work was taken down from a *well respected university*.
   - The lab has been accused of slapping a *fake AGI name* on plagiarized work, sparking outrage online.
- **GPT-5 Excludes O3, Gemini 2.5 Pro Style?**: Members speculated that **GPT-5** may ditch the **o3** component, mirroring **Gemini 2.5 Pro** by integrating an LLM and a reasoning model.
   - Anticipation is building for a potential summer release, possibly coinciding with a **Google week** event.
- **Gemini 2.5 Pro Math Performance Degraded?**: Community members are comparing **Gemini 2.5 Pro's** performance with older models, observing a trade-off between math and coding skills, citing [official Google numbers](https://deepmind.google/technologies/gemini/pro/).
   - Performance is at **83 vs 88.9%**, suggesting that the older model is better for math.
- **HyperEnglish Prompting Syntax Enforces Clarity**: A new prompting method, [HyperEnglish](https://example.com/hyperenglish), was introduced, using *functional tagging* and **ALL-CAPS** for technical terms to enhance clarity in prompts.
   - The template uses a strict structure like `SUBJ:{subject} VERB:{verb} OBJ:{object}`.
- **ProtoMind Semantic Mapping Side-Steps Code**: [ProtoMind_001](https://chatgpt.com/g/g-682759c7c8d881919436d320e718d5d5) treats user input as **layered semantic operators**.
   - It has the ability to map input pretty well, and negates the need for explicit code for role-switching and pseudo-memory threading.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **O3 Pro's Release Remains Elusive**: Users are expressing frustration over the continued delay of **O3 Pro**, with some jokingly threatening a *hunger strike* until its release.
   - Some users believe that the original **GPT-4** was superior, possessing a *genuine intelligence/creativity spark* that newer, smaller models lack.
- **GPT-5: Incremental Upgrade or Base Model?**: Speculation surrounds **GPT-5**, with some suggesting it will be a base model comparable to **O4**, while others are skeptical, proposing it might only be a marginally improved version of **O3**.
   - The discussion covers whether it will be a model router or a new base model, and how RL training impacts improvement over stable versions.
- **Codex: Friend or Foe for Devs?**: The utility of **Codex** is being debated, with some considering it *noise* for junior developers, while others see its potential for more advanced tasks.
   - A user suggested that **Codex** needs to compete with tools like **RooCode/Cline/Cursor/Windsurf/Aider** to be worthwhile.
- **Gemini and Claude duke it out on code**: The bots locked horns over coding, with some finding **Gemini** annoyingly verbose and prone to adding unwanted features, while others praise **Claude** for its reliability.
   - Some users find **Gemini's** code comments to be a negative aspect, while others see them as beneficial.
- **LMArena Beta Welcomes Fresh Faces**: The LMArena Beta Site has added new models, including **mistral-medium-2505**, **claude-3-7-sonnet-20250219-thinking-32k**, **amazon.nova-pro-v1:0**, and **command-a-03-2025**.
   - Since the debut of **Mistral Medium 3** on the leaderboards, it made an impressive leap of **+90** from Mistral Large landing at #11 overall in chat.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemini 2.5 Pro Experimental Gets the Axe**: Google is deprecating **Gemini 2.5 Pro Experimental** (`google/gemini-2.5-pro-exp-03-25`) in favor of the paid **Preview endpoint** (`google/gemini-2.5-pro-preview`).
   - Users mourned the loss of **Gemini 2.5 Pro Exp** (*03-25*) and expressed dissatisfaction with the newer 05-06 version, noting issues of content filtering.
- **DeepSeek V3 Tunes Up for Optimal Performance**: The free **DeepSeek V3 0324** is temporarily offline to undergo scheduled maintenance.
   - Users should anticipate a brief disruption in service while the model undergoes necessary adjustments.
- **Chess Tournament Evolves with Stockfish and OpenRouter**: A member transformed their **chess tournament** idea into a **chess leaderboard** incorporating **Stockfish implementations** for accurate **Lichess accuracy ratings** using **Openrouter models**.
   - The project automates ratings using cronjobs and accommodates models like o1-mini, showcasing a *different use case*.
- **Request Tags to Refine API Key Tracking**: To better track API call sources, members suggested implementing *request tags* instead of embedding user IDs in the app name, particularly to track disconnects mid-stream when multiple users share a single key.
   - This facilitates detailed logging of individual user requests and minimizes confusion when using shared resources.
- **Qwen3 Tool Calling Faces Hiccups on Kluster**: A user reported that **Qwen3 235B** experiences tool calling issues with the *Kluster* provider, prematurely ending tool calls.
   - They discovered that OpenRouter resolves this by switching to another provider, suggesting potential compatibility challenges with *Kluster*.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **VRAM Usage in LM Studio still Mysterious**: Users are reporting that **LM Studio** displays the correct VRAM amount, but actual usage remains very low, even when offloading model layers to the GPU, with one user reporting it uses only *256-512MB* of VRAM on a 9070 machine.
   - They are investigating potential driver issues to determine if the model loads into dedicated VRAM or shared video memory, suggesting it could be a driver bug.
- **Users Want LM Studio Conversation Exports**: Users are seeking ways to export **LM Studio** conversations in more flexible formats than the current JSON, specifically for storytelling.
   - Suggestions included using an LLM to parse the JSON into a preferred format, while acknowledging the lack of API guarantees for the JSON structure.
- **Enable Resizable BAR for GPU Speed Boost**: A user reported poor performance with a **9070XT** GPU, initially getting only *8.52 t/s* on a **Gemma3 4b QAT** model after a fresh Windows install, but they saw huge improvements after toggling the right settings.
   - Enabling **Resizable BAR** (or **Smart Access Memory (SAM)** on AMD) boosted performance to *131.91 t/s*, showcasing its major impact on LM Studio.
- **Arc Pro B60 GPU Sparks Excitement**: Members expressed enthusiasm for the [Intel Arc Pro B60](https://www.intel.com/content/www/us/en/products/sku/243916/intel-arc-pro-b60-graphics/specifications.html) GPU, praising the **96GB VRAM** and anticipated sub-$2000 price, making it appealing for local LLMs.
   - Concerns remain about software support, but there is hope the high VRAM availability will enhance Intel GPU support in the AI field and encourage providers to improve their offerings.
- **macOS: The Silent LLM Performer**: A member described **macOS** on a Macbook as more fluid than Windows, with another user citing issues with MacOS window sizing, while a member shared how a Macbook can run a **32B 4-bit model for >5 hours** on battery at 5t/s.
   - The Macbook setup boasts efficient and silent LLM operation with system RAM usage of <2GB.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **MCP Course Channel: MIA?**: Members are seeking the **MCP course channel**, expressing confusion over whether it's related to the **Agents course**, and sharing links to the [course on GitHub](https://github.com/huggingface/mcp-course).
   - The current whereabouts or existence of an active **MCP course channel** remain unknown.
- **Hugging Face Pro Glitch troubles AI Engineers**: A user reported that the **AI Engineer Pack link for 6 months of free Hugging Face Pro** isn't working and was advised to contact HF support at [website@huggingface.co](https://discord.com/channels/879548962464493619/1353741359059570699).
   - This issue prevents AI engineers from accessing the benefits associated with **HF Pro**, potentially hindering their projects and workflows.
- **Strands Agents SDK: Amazon Simplifies AI Agent Creation**: Amazon introduces the [**Strands Agents SDK**](https://aws.amazon.com/blogs/opensource/introducing-strands-agents-an-open-source-ai-agents-sdk/), a well-designed **open-source AI agent SDK**.
   - The SDK aims to streamline the development of AI agents, providing developers with tools and resources to build and deploy agents more efficiently.
- **datatune Transforms Data with LLMs and Natural Language**: A member from Vitalops introduced [datatune](https://github.com/vitalops/datatune), a new **open source tool** that performs data transformations using simple natural language instructions and LLMs.
   - This offers a more intuitive approach to data manipulation compared to traditional coding methods, potentially increasing efficiency.
- **Lazarus rises! Small LLM**: A member shared [Aclevo/Lazarus](https://huggingface.co/Aclevo/Lazarus), the next best small LLM, distilled from Llama3 with approximately **124 million parameters**.
   - The poster made the claim in relation to its size compared to other LLMs, citing [distillation from Llama3](https://huggingface.co/blog/codelion/pts) as part of its secret sauce.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **AMD Challenge Access Achieved**: A participant in the **AMD challenge** reported difficulties accessing the leaderboard channel and after checking specific channels [<#1359640791525490768> and <#1343002583001726986>] access was achieved.
   - This highlights the importance of verifying channel permissions and consulting relevant resources within the community for troubleshooting.
- **FSDP and Flash Attention 2 Face Fusion Friction**: A member sought advice on integrating `FSDP` and `Flash Attention 2` with `trl` for model training, referencing the [trl `sft.py` script](https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py).
   - The challenge lies in the incompatibility when using **Flash Attention 2.0** with a model not initialized on the GPU, resulting in errors.
- **Kernel Builder Seeks Cohorts**: A member is actively developing a kernel from scratch and is inviting contributions, with project details available on [GitHub](https://github.com/DorsaRoh/Kernels).
   - The project is tagged "gpu mode = 🔥", signaling a strong emphasis on GPU-centric kernel development.
- **Neural Net Nightmare: Mutation Function Mishaps**: A member encountered issues with a **mutate function** for a **neural net**, triggering errors such as *unspecified launch failure* or *illegal memory access* at random generations, code at [godbolt link](https://cuda.godbolt.org/z/z8z6a85vP).
   - Error was *Malloc/Free Error encountered : Invalid pointer to free*, suggesting memory corruption during the mutation process.
- **CuTe Talk Incoming!**: Cris Cecka, the inventor of **CuTe**, is giving a talk about **CuTe** that started in 5 minutes.
   - The talk is a prime opportunity to learn more about **CuTe** directly from its creator.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **FinTech Data Scientist Digs into Document AI**: Akhil Theerthala, a Data Scientist from India, is developing **Document AI and NLP** projects, focusing on Document Representation, TSR, and Document Quality Assessment.
   - He is also exploring reasoning about **Personal Finance**, **Ethical Dilemmas**, and **Resume/Career path analysis** in his personal projects.
- **AI_WAIFU Shifts AGI Timeline**: **AI_WAIFU** has shortened their AGI timelines, anticipating it this year or early next year, attributing this acceleration to coding models and after having left EAI to work on something new.
   - They noted that improvements in **NN efficiency** are benefiting smaller models more than significantly boosting larger models and that nanotechnology might require more compute than AGI itself.
- **OpenWorldLabs Develops World Models**: **OpenWorldLabs** focuses on building **world models for robots and video games**, accessible on their [website](https://openworldlabs.ai).
   - Some community members struggled to understand their purpose and one stated: *I still have almost no idea what you're actually doing* after reading the website and GitHub twice.
- **Diffusion Models Face Error Accumulation Issue**: Members discussed that resolving **accumulating error** in diffusion models requires renoising context frames, but existing video models are too slow for this.
   - A member said that **Google** might have addressed this by training with error to denoise frame by frame.
- **lm eval harness Customized for Quantized Model**: A member sought guidance on reproducing zero-shot reasoning tasks using the **SpinQuant** technique on different bit precision quantized **Llama-2-7b** models from [Facebook Research](https://github.com/facebookresearch/SpinQuant) models.
   - It was suggested that the member could use *lm eval harness* by passing an already initialized model, following [this documentation](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage) which would require wrapping the model in a class, potentially by inheriting or modifying the existing [HFLM class](https://github.com/EleutherAI/lm-evaluation-harness/blob/53c653008182339e67b964a4cd3316f651611f38/lm_eval/models/huggingface.py#L47) within the *lm eval harness*.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Freeplay.ai mirrors in-house product architecture**: Members discussed [freeplay.ai](https://freeplay.ai), with one user reporting positive initial impressions and noting it *mirrors* in-house builds.
   - Another user expressed strong interest in updates and inquired about significant differences between **Freeplay.ai**, **Braintrust**, and **Humanloop**.
- **Absolute Zero Reasoner Teaches Itself Without Data**: A member shared a [YouTube Short](https://youtube.com/shorts/avnHiKcOEQA?si=NWoqwZR1IcPxyrG0) breaking down **Absolute Zero Reasoner (AZR)**, a new AI that teaches itself from scratch without human data.
   - The member requested feedback and thoughts on the video.
- **AI Devs Demand a Dev Survey**: A member suggested a "State of AI" dev survey to track usage trends in areas like **AI agent frameworks**, **SDKs**, **proxies**, and **models**.
   - Members shared a [link to the 2025 State of AI Engineering survey](https://x.com/barrnanas/status/1904593314533564604) and [Allstacks survey](https://www.surveymonkey.com/r/6WQD6QQ) highlighting the need to understand software productivity gains.
- **OpenAI Codex Tackles Shopify App Creation**: A member experimented with **OpenAI Codex** to convert an existing application, Answer HQ, into a compatible **Shopify App Store** app.
   - They noted that Codex first sought an *agents.md* file and that a good README helps streamline the process, recommending generating a AGENTS.md file for LLM consumption outlining domain, key commands, how to run tests, and conventions for the project.
- **Agent as Judge alleviates task fatigue**: Members discussed that lower task fatigue is due to less context switching, noting a burnt out developer who's been working at **0.1x capacity** returns to **1x performance** with **Agent as Judge**.
   - Burnt out devs can now get back to 1x or greater performance, and we're back to *agent as judge*.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Codex o4-mini Launch Fumbles Amidst Tech Layoffs**: The launch of the **Codex o4-mini** model coincided with big tech companies laying off people working on well respected and valued products.
   - One member joked that Zuck is probably upset that surfing all day makes him look like an idiot while Musk looks like a genius with Grok.
- **Gemini 2.5 Pro Stumbles as Coder Model**: Members noted that a pass rate of **45%** for **Gemini 2.5 Pro/Flash** raised concerns about using coder models in practice, recommending **O4-mini** or **Claude 3.7** if you value your time.
   - A member said that *pro messes diffs up so much* that they are thinking about doing more practical experiments, eg flash is not a coding model so some super cheap coding model.
- **Aider's Agentic Ambitions Spark Worry**: Members are concerned about **Aider's** direction in becoming more agent-like, worrying that it might lose its core identity as a pair programming tool.
   - One member expressed worry that even if agent-like features are added, it will end up being *neither here nor there*.
- **Aider Config Leveraged to Groom Workplan Documents**: One member utilizes a **workplan document** that is groomed through the process, set up in their **base prompt** ([aider_conf.html](https://aider.chat/docs/config/aider_conf.html)).
   - When they want to iterate like that, they generally do it in `/ask` mode, and when things look good, they can `/code okay, please do that`.
- **Aider's Minimal UI Sparks Theme Quest**: A user wondered if there were **UI themes available for Aider** and if some elements could be customized.
   - A member replied that there's **light mode and dark mode**, another member pointed to [Pygments](https://pygments.org/styles/) which can be set via `--code-theme`.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **One-Click MCP News Agent Debuts**: A user introduced a new agent that aggregates and summarizes MCP news from the past 24 hours with a single click, accessible [here](https://www.jenova.ai/app/tr45cl-mcp-news-summarizer).
   - The agent aims to streamline information gathering for those closely monitoring developments within the **Model Control Protocol** ecosystem.
- **Battle Brews Between App-Driven vs LLM-Driven Resources**: Members debated the merits of **application-driven** versus **LLM-driven** resources, particularly concerning tools, with some suggesting current resources are too limiting in their app-centric approach.
   - Others countered that app-driven resources enable powerful functionalities such as indexing in **MeiliSearch** and real-time **RAG**.
- **MCP Clients Gain Selective Tool Control**: The ability of an **MCP Client** to selectively choose tools from a server's tool signature was discussed, citing **Claude Desktop** as an example where users can toggle tools on/off.
   - The discussion emphasized the importance of client-side control over available tools for enhanced customization and security.
- **Stainless SDK Slammed for Data Validation Failures**: Users criticized the **Stainless SDK**, noting that its generated **pydantic models** fail to properly validate data, which contributes to ecosystem fragmentation.
   - They claimed that the **OpenAPI** document doesn't accurately reflect the API behavior.
- **Sherlog Canvas (Alpha) Powers Debugging**: Sherlog Canvas (Alpha), an AI-powered interface for incident debugging, has been open-sourced, integrating [MCP-powered cells](https://github.com/GetSherlog/Canvas) for logs, metrics, SQL, and more, and a [demo video](https://youtu.be/80c5J3zAZ5c) showcasing its capabilities.
   - It offers multi-agent workflows and allows AI to generate, run, and refine cells, aiding in incident and bug investigations.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Mobile NotebookLM App Arrives Bare-Bones**: The **NotebookLM mobile app** launched on [iOS](https://apps.apple.com/us/app/google-notebooklm/id6737527615) and [Android](https://link-to-android-store) with **MVP feature set**, though users note the absence of core features like **mind maps**, **briefing notes**, and **'Discover Sources'**.
   - Users report being unable to select sources for **audio overviews** and find the app to be a *web app packaged into a local container*, and are being encouraged to submit feedback and feature requests.
- **Video Uploads Supercharge NotebookLM**: NotebookLM now supports **video uploads** with automatic transcription, confirmed by a user who shared a [link about **Video overviews in NotebookLM**](https://x.com/testingcatalog/status/1924246774346104863).
   - However, some users are experiencing limitations with audio generation, such as NotebookLM only processing the introduction of longer documents, creating short audio files even with a **114-page PDF**, and it is unclear if this is a limitation or a bug.
- **NotebookLM Rejects Progressive Research**: A user expressed disappointment when **NotebookLM** failed to upload research materials related to social justice, ecology, and feminism, including an example from [The Conversation](https://theconversation.com/meet-the-forgotten-enslaved-and-working-class-labourers-behind-british-exploration-in-africa-asia-and-antarctica-252771).
   - The user suspected *censorship and anti-woke stuff* was the reason, and included screenshots showing materials being rejected.
- **Senpai NotebookLM Delivers Hot Takes**: A user boosted productivity with **NotebookLM** by sharing a `guide.md` file with every new notebook, containing instructions to use a custom persona.
   - The `guide.md` file assigns personas like *senpai* (a professional senior student) or *brain* (a genius AI) to **NotebookLM** for different types of requests, allowing for customized interactions, as shown in [this Youtube video](https://youtu.be/ZCPcBgJ54NY?si=9gHljywup_mO0cAM).
- **NotebookLM's Prose Dumbs Down**: A user noted that *NotebookLM’s writing style has become strangely simplistic and reductive in the last few days*, resembling a **high-school-essay quality**.
   - This contrasts with earlier discussions about potential uses for students preparing for olympiads, emphasizing **confidentiality** as a key differentiator from Gemini or ChatGPT.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Gemini's Code Comments get Quirky**: Members observed that **Gemini** sometimes generates commented-out code and removes existing comments without obvious cause.
   - A member humorously likened **Open A.I's** trajectory to a Greek saga suitable for a **Netflix** series, alluding to unexpected developments.
- **Google Announces Jules at I/O**: **Google** is expected to unveil its code agent, **Jules**, at **Google I/O**, featuring multimodal capabilities and daily summaries.
   - A [link to jules.google](https://jules.google/) was shared and anticipation grew within the community.
- **Deep Dive into AWQ INT4**: A user inquired whether **AWQ** uses **INT4** natively or conducts calculations in **BF16**, specifically within **VLLM**.
   - Another member clarified that **AWQ** operates in **W4A16** format and GPUs lack the necessary circuitry for mixed-precision **INT4xBF16** operations, suggesting **QuaRot** or **FP8** as alternatives.
- **Debate on Open Source AI vs Big Tech**: A member asserted that only **decentralized open source AI** can effectively prevent a big tech oligopoly in the AI field.
   - Counterarguments advocated for governmental regulations, such as open-sourcing AI models with data and anti-censorship laws, leading to a debate on the viability of each strategy.
- **LLMs Spontaneously Adopt Conventions**: A study ([arxiv link](https://arxiv.org/abs/2505.10475), [science link](https://www.science.org/doi/epdf/10.1126/sciadv.adu9368), and [HF blog link](https://huggingface.co/blog/codelion/ptsreal.azure)) demonstrates that **universally adopted social conventions** can emerge spontaneously in decentralized LLM populations through local interactions.
   - The research emphasizes that strong **collective biases** can develop during these interactions, even without initial biases in individual agents, and that **minority adversarial LLM agents** can force social change.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **CNN Diagrams Drawn with Matplotlib**: A member sought a way to create **CNN diagrams with skip connections** and resorted to using *matplotlib* with assistance from GitHub Copilot, resulting in a [script](https://github.com/dotrdp/DiagramVIS-for-computervis) available on GitHub.
   - The repository is named *dotrdp/DiagramVIS-for-computervis*.
- **Gemini 2.5 Pro Handles Physics Tasks**: One member uses **Gemini 2.5 Pro with Canvas or Cursor** for physics-related tasks and is interested in tools like *Windsurf, Aider, Cline, Roo*, and *Codex*.
   - They mentioned this in the context of tools that could be used for physics-related tasks.
- **AlphaEvolve Paper Sparking Discussion**: Members expressed interest in the **AlphaEvolve** whitepaper ([AlphaEvolve.pdf](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf)), which details a **Gemini**-powered coding agent designed to create advanced algorithms.
   - The discussion was subsequently canceled due to redundancy with ongoing conversations in the *Open Machine Learning Channel*.
- **Physics of Language Models Explored**: The group discussed **"Physics of Language Models: Part 3.1, Knowledge Storage and Extraction"** ([paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5250633), [blog](https://physics.allen-zhu.com/part-3-knowledge/part-3-1), [YouTube](https://www.youtube.com/watch?v=YSHzKmEianc)).
   - Further details of this paper can be found in the blog and YouTube video linked.
- **Open Source AI: A Strategy or Resource Grab?**: Members debated whether releasing AI research under open-source licenses is a strategic move or just a method to acquire a free workforce and resources, particularly concerning **Meta**.
   - One member argued Meta's AI research serves to *commoditize your complements* rather than being central to Facebook-the-product.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex AMA Debuts This Week**: The first **LlamaIndex** office hours will be this Thursday at **8AM PT/5PM CET** in the general voice channel for **1 hour**, to ask anything about **LlamaIndex** and building agent workflows.
   - The team also released an update on **LlamaIndex Agents** and their improved **long-term and short-term memory** on the [LlamaIndex blog](https://www.llamaindex.ai/blog/improved-long-and-short-term-memory-for-llamaindex-agents).
- **Weaviate Powers Multi-Agent Workflows**: A guide to multi-agent workflows using **Weaviate** `QueryAgent` is now available in the [LlamaIndex documentation](https://docs.llamaindex.ai/en/stable/examples/agent/multi_agent_workflow_with_weaviate_queryagent).
   - Learn how to extract structured data with citations and reasoning using **LlamaExtract** in [this YouTube video](https://youtu.be/01kM7tXRHi4).
- **LlamaParse Gets Facelift**: **LlamaIndex** announced updates to **LlamaParse**, with a *streamlined interface*, a new **Code Snippet** button, and more use-case presets, according to [this tweet](https://twitter.com/llama_index/status/1923510823164706925).
   - The now generally available **Azure AI Foundry Agent Service** comes with first-class **LlamaIndex** support, enabling enterprise customers to build customer support assistants and process automation bots, as highlighted in [this tweet](https://twitter.com/llama_index/status/1924502129974411504).
- **COBOL Crumbles Before Chonkie**: A user inquired about Python packages for splitting **COBOL** code into logical blocks and was directed to [Chonkie.ai's `CodeChunker`](https://chonkie.ai), which supposedly supports **COBOL**.
   - The user noted that LlamaIndex code splitter doesn't currently support **COBOL**; this may be a feature request.
- **LlamaIndex Ollama Sees Clearly Now**: A user encountered a `ValueError: "ChatResponse" object has no field "usage"` when using LlamaIndex with Ollama, despite following the [official documentation](https://docs.llamaindex.ai/en/stable/examples/llm/ollama/).
   - The user was able to resolve the issue by upgrading Python to **3.10**, creating a new environment, and upgrading llama-index and ollama packages.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **NDBuffer Deprecation Looms 🌅**: The **NDBuffer** is being deprecated in favor of **LayoutTensor**, and users are advised to avoid it, as active kernel transitions are in progress.
   - This transition may affect existing codebases relying on **NDBuffer**, requiring migration to **LayoutTensor** for continued functionality.
- **Atomic ArcPointer Impasse ⛔**: A user encountered issues creating an **ArcPointer** to a struct containing an **Atomic** due to the **Movable** trait.
   - The suggested workaround using `ArcPointer[OwnedPointer[T]]` did not resolve the problem, as **OwnedPointer** does not implement **Movable** either.
- **Mojo Notebook Import Mystery 🔍**: A user inquired about importing code from a **Mojo package** or file within the same folder as a **notebook**.
   - Unfortunately, no solution was provided in the messages, leaving the user's import conundrum unresolved.
- **LSP Server Meltdown 🫠**: Users reported high CPU usage (8-16 threads) and frequent crashes with the **LSP server**, especially on older systems or when using Docker.
   - Workarounds like restarting the **LSP server** or downgrading to a previous nightly build offered limited relief; one user had to resort to `killall mojo-lsp-server`.
- **`register_custom_ops` Gets the Axe 🪓**: The function `register_custom_ops` in `max.torch` was removed in the latest nightly, which broke some users' scripts.
   - A Modular team member confirmed ongoing work on registering Mojo custom ops with PyTorch in nightlies, directing them to the updated [documentation](https://docs.modular.com/max/api/python/torch/).



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **George Hotz Posts tinygrad Performance Stats**: George Hotz shared a link to [tinygrad performance statistics](https://stats.tinygrad.win), showcasing improvements in **tinygrad's** performance.
   - The stats provide a real-time view of **tinygrad's** efficiency, helpful for developers optimizing their models.
- **Debate Arises: GCC Instead of Clang for tinygrad?**: A user inquired about using **GCC** instead of **Clang** for a CPU target, specifically for an **AIX system with the PPC64 arch** where **Clang** is unavailable.
   - George Hotz responded that it's *not easily* done and would require adding **elf relocations for ppc64** to the custom elf loader, referencing [this file](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/runtime/support/elf.py#L41).
- **ONNX** Simplifies Model Porting to tinygrad**: A user asked about porting models like **Qwen3:30-a3b** to TinyGrad, inquiring about automatic tools versus manual porting.
   - George Hotz clarified that if the model is in **ONNX** format, it's easy to import using the `examples/benchmark_onnx.py` script.
- **tinygrad's API Stability Praised**: An author writing a book on AI application development considered using tinygrad, asking about the stability of its interfaces for examples to remain functional over 2-3 years.
   - George Hotz assured that the frontend has been *very stable for at least a year*, suggesting they *should do 1.0 before we do speed*.
- **WMMA** Instruction Benchmarking Tool Arrives**: A user shared a link to [HIPmmapeak](https://github.com/pkourouklidis/HIPmmapeak), a tool to measure max **FLOPS** of the **wmma instructions** on an **7900 XTX**, similar to mmapeak.
   - George Hotz responded *oh cool! use the tinygrad infra if you want the bounty*.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **AI/ML Engineer Debuts Agentic Expertise**: An AI/ML engineer with **8+ years** of experience in intelligent systems introduced themself, showing their work building agentic systems using modern stacks like **LangGraph**, **AutoGen**, **LlamaIndex**, **Letta**, and **DSPy** and shared [their portfolio](https://yangming.vercel.app/).
   - They demonstrated proficiency with **GPT-4o**, **Claude 3**, **Gemini**, **Mixtral**, **LLaMA-3**, and **Mistral**, as well as full-stack technologies like **React**, **Next.js**, **FastAPI**, **Django**, and **Laravel**.
- **Suggest and Assert Evolve into BestOfN and Refine**: **BestOfN** and **Refine** are now the replacements for `dspy.Suggest` and `dspy.Assert` as of **DSPy 2.6**, detailed in [this tutorial](https://dspy.ai/tutorials/output_refinement/best-of-n-and-refine/).
   - This change encourages more structured approaches to output refinement and validation within DSPy programs, and makes it easier to validate and debug outputs from LLMs.
- **Cline Reimplemented with DSPy Claims Size and Accuracy Benefits**: Members talked about reimplementing an AI coding agent like **Cline** using DSPy; suggesting it might be smaller and more accurate, with less *off-piste* changes.
   - It was noted that VS Code glue, memory, tools and models all being important factors in **Cline**'s success.
- **DSPy Latency Troubles with Large Prompts**: A member experienced long resolution times in DSPy with large system prompts, and inquired about configuring **litellm**'s [prompt caching](https://docs.litellm.ai/docs/completion/prompt_caching), which resulted in an error.
   - Another member optimized the prompt outside of DSPy and found that DSPy responses took **8s vs 2s**, suggesting potential inefficiencies in DSPy's handling of optimized prompts.
- **Elixir Port of DSPy 3.0 Attracts Advocates**: A member inquired about interest in a **1:1 parity port of DSPy 3.0 to Elixir**, sparking discussion around the language's scaling patterns and concurrency model being *perfect for LLM orchestration*.
   - While some advocated for Elixir's capabilities, others pointed to Python's dominance in the ecosystem due to integrations and maturity with libraries such as [Nx ecosystem](https://github.com/elixir-nx).



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Says No Finetune for Embeddings**: A member inquired about fine-tuning an embedding model, and another member clarified that **fine-tuning embedding models is not possible**, but **fine-tuning a reranker** based on specific requirements *is* possible.
   - They were unable to find a solution to their issues, as it is recorded as a known issue.
- **Embed-v4.0 Faces 'Unwanted' Embedding**: A member reported that **embed-v4.0** generated some *unwanted* embeddings, and inquired whether it is possible to fine-tune an embedding model.
   - No solution was given but is recorded as a known issue.
- **Embed 4 Pricing Investigated for Vision RAG**: A member inquired about pricing for **Embed 4** in the context of their team's **Vision RAG** implementation.
   - Another member suggested embedding an image and checking the **token count** in the usage section of the response.
- **Chat API Experiences Stalling During Agent Execution**: A member reported that the **Chat API** seems to stall midway during the execution of their agent, despite being a paid member, leading to requests for debugging assistance.
   - They mentioned experimenting with running the **nodes sequentially vs. in parallel** as a transient workaround, and were asked for details such as the **model, API version, task, and code snippet** to assist in debugging.
- **Vitalops' datatune goes open source**: Vitalops released a new **open source tool** called **datatune** on [GitHub](https://github.com/vitalops/datatune) that performs data transformations using natural language instructions and LLMs.
   - The **datatune** tool utilizes LLMs to execute **data transformations** based on simple **natural language instructions**, aiming to simplify complex data manipulations.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Swarm UI Favored Over Local Docs**: A user expressed a preference for **Swarm UI** over local documentation for its ease of use in various applications.
   - They stated that *local docs are fine ... for now* but **swarm-ui** is the better choice for *easy to mid to advanced* usages.
- **Linus Builds Million-Dollar Pi Server**: A member shared a link to Linus Tech Tips' construction of a **$1M server** designed for calculating digits of pi.
   - Another user responded that *That's not the first time I see Linus doing insane things.*
- **Guidance Sought on Customer Success Roles**: A member with a background in government, nonprofits, sales, and outreach inquired about transitioning into a **Customer Success** role at a tech company.
   - They emphasized their aptitude for supporting others, relationship building, and problem-solving, along with their technical expertise and ability to simplify complex ideas.
- **Models Recommended for Textbook Markdown Conversion**: A user asked for model recommendations to convert large quantities of textbook text into **markdown notes**, including summarizing and extracting specific details.
   - They were looking for tools capable of effectively handling these tasks.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **AgentX Competition Submission Deadline Nears!**: The submission deadline for the **AgentX Competition** is **May 31, 2025**, at 11:59 PM PT, with separate submission links for the [Entrepreneurship Track](https://forms.gle/FJTC4jd197bNeJJ96) and [Research Track](https://forms.gle/5dccciawydCZ8o4A8).
   - The competition boasts an all-star judging panel and a prize pool of over **$150K**, with details available on [Twitter](https://x.com/dawnsongtweets/status/1924470174776004875) and [LinkedIn](https://www.linkedin.com/posts/dawn-song-51586033_agentx-agentx-activity-7330241621160005633-E8Ii).
- **Roll Your Own OpenAI API Key**: Students in the LLM Agents MOOC must provide their own **OpenAI API keys** to run lab exercises, as the course *does not provide them*.
   - Lab TAs have been requested to provide insight into alternate methods to minimize or bypass the need for external API interactions.
- **MOOC Students Can Still Get Trailblazer Tier**: Students who struggle with labs in the LLM Agents MOOC can still apply for the **Mastery Tier** and potentially be 'downgraded' to the **Trailblazer Tier**.
   - This ensures recognition for students who demonstrate knowledge through quizzes and written articles.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **CI Nightly Builds for Evaluation Considered**: A user is weighing the pros and cons of evaluating models on an **ad hoc basis** versus setting up an **automated nightly CI pipeline** for model evaluation, which requires more effort and compute resources.
   - The user acknowledges the **ad hoc** approach is less reliable but easier to implement, while the **automated CI** approach provides higher fidelity but demands more maintenance and computational resources.
- **Torchtune validation with cfg.get**: A user searched GitHub's code to find out where the the **validation dataset** is set using the config's `cfg.get("dataset_val")` in the [pytorch/torchtune repository](https://github.com/search?q=repo%3Apytorch%2Ftorchtune%20cfg.get(%22dataset_val%22)&type=code).
   - Understanding how the validation dataset works is useful for evaluating model training, and exploring `cfg.get` helps discover where dataset configurations are set.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Vitalops Tunes Data with LLMs**: Vitalops released **DataTune**, an open-source tool leveraging LLMs for data transformations using natural language instructions, available on [GitHub](https://github.com/vitalops/datatune).
   - **DataTune** simplifies data transformation through natural language prompts.
- **Vitalops Tunes Data with LLMs Again!**: Vitalops released **DataTune**, an open-source tool leveraging LLMs for data transformations using natural language instructions, available on [GitHub](https://github.com/vitalops/datatune).
   - **DataTune** simplifies data transformation through natural language prompts.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Experienced AI Engineer Seeks Connection**: An AI Engineer with **10 years of experience** in machine learning, deep learning, and data science is seeking to connect and build the next generation of thinking software.
   - They possess a deep knowledge of **Python, TensorFlow, PyTorch**, and cloud platforms such as **AWS** and **GCP**.
- **AI Engineer Boasts Robust Skill Set**: The AI Engineer demonstrates proficiency in **Python, SQL, R**, ML/DL Frameworks (**TensorFlow, PyTorch**), and tools like **Jupyter, Git, Docker, MLflow, Streamlit**.
   - Their techniques encompass Supervised & Unsupervised Learning, Deep Learning (CNN, RNN, Transformers), NLP, Computer Vision, and Model Deployment (APIs, CI/CD).



---


The **Codeium (Windsurf) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1373012180626178140)** (1047 messages🔥🔥🔥): 

> `Cunnyx links, MCP SuperAssistant extension, Grok Deepsearch vs Perplexity Pro, Firefox promoting Perplexity, Yandex browser security` 


- **Percentages shown in You.com are win rate, not quality**: A member shared a [Cunnyx link](https://cunnyx.com/youdotcom/status/1923100692098453703) noting that the percentages shown in **You.com** are win rate, not quality, though the video won't state this explicitly.
- **MCP SuperAssistant Extension in Perplexity**: A user tested the **MCP SuperAssistant** extension within Perplexity, reporting it works well with many servers but faces occasional disconnections; links to the [MCP SuperAssistant website](https://mcpsuperassistant.ai) and the [GitHub repository](https://github.com/srbhptl39/MCP-SuperAssistant) were shared.
- **Grok Deepsearch gets favored over Perplexity Pro**: Despite having Perplexity Pro, a user preferred **Grok Deepsearch** due to inconsistent results from a query on food prices in Indonesia vs India, noting the Python math tool was absent in Perplexity's research tasks and shared links to [Perplexity Research mode](https://www.perplexity.ai/search/is-it-pricey-to-have-like-100g-p6cc07L0RAGxI9vBnBDfIw), [Perplexity Pro mode with o4 mini](https://www.perplexity.ai/search/is-it-pricey-to-have-like-100g-02GLrmEVR1eVEbFelRbBRQ), and [Grok Deepsearch](https://grok.com/share/bGVnYWN5_28787552-f2a7-494a-b876-00980d4d523d).
- **Firefox to Promote Perplexity Search Engine**: It was noted that [Firefox is set to promote the Perplexity search engine](https://windowsreport.com/mozilla-firefox-to-promote-perplexity-search-engine/), though a member questioned if it was already on Android, and another dismissed Firefox as dying, favoring Blink over Gecko.
- **Yandex Browser Safety in Question**: Members debated the security of **Yandex Browser**, with one user stating, *"it's better than Chrome,"* while another expressed concern due to it being a Russian browser and shared that they have *personal email account on yandex* and find their image generator useful, also another member provided a list comparing the [privacy and security of Yandex, Chrome, and Firefox](https://media.discordapp.net/attachments/669308329419341836/1079537276775829605/935C5928-EDC9-4D6B-993B-1E81113888E6.gif).


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1373177570610118708)** (16 messages🔥): 

> `API Credits, Sonar API vs UI Discrepancy, Sonar API Tweaking, Playground Outputs vs API Outputs` 


- **API Credits Delay Frustrates Hackathon Team**: A team reported waiting **5 days** for **API credits** and receiving only **AI responses** to their inquiries.
   - It was later confirmed that they had received the API credits; no further discussion occurred.
- **Sonar API Sources Differ Wildly From UI**: Multiple users expressed confusion over the **Perplexity Sonar API** returning drastically different results compared to the **UI version**.
   - One user noted that the **API sources** often have *“0 mention of that name”* when looking up profiles, and another shared a [GitHub repo](https://github.com/alanbuxton/perplexity-api-tester) used for testing.
- **Sonar API Tweaking Still Hazy**: Users questioned how to tweak the **Sonar API**, specifically whether parameters like `top-p` affect the actual search.
   - The [Perplexity API documentation](https://discord.com/channels/1047197230748151888/1118264005207793674/1366292101666439239) offers limited information on customizing the **Sonar API**.
- **Playground Outperforms API**: A user questioned why they were getting better outputs on the **playground** compared to the **API**.
   - Another user speculated the differences could be caused by a system prompt within the **UI**, or some other reason.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1373012823549935616)** (1551 messages🔥🔥🔥): 

> `Convex vs Supabase for real-time apps, MCP for AI agent communication, DeepSeek-R1T-Chimera model, Cursor speed issues, Document Navigation Within Cursor` 


- **Convex crushes Supabase for Real-Time Apps**: Members discussed that while **Supabase** offers real-time capabilities, **Convex** is generally considered better for real-time applications due to its automatic synchronization and event-driven architecture, also a [comparison highlighting why Convex excels in real-time scenarios](https://x.com/mntruell/status/1924297275993669893).
   - Someone also pointed out that Supabase can be hosted locally but Convex is better for real-time stuff, but the member wouldn't stop using Supabase for anything, adding that its Auth, Edge Functions and Buckets is *fire as sheet*.
- **MCP Mania for AI Agent Communication**: A conversation emerged around using **MCP** (Model Context Protocol) for AI agents to communicate, especially those not on the same computer, someone said that *Qwen 3 235B came with native MCP support because of this reason*
   - Members described their setups, from a Discord admin bot to coordinating tasks between agents with a single source of truth and context, someone even made a [repo for this](https://github.com/coder/agentapi).
- **DeepSeek-R1T-Chimera masters markdown**: Users praised the **DeepSeek-R1T-Chimera** model for its ability to precisely edit .md and .mdc files without mistakes and for breaking loops in prompt testing, noting that it's the only free model to achieve this, and can get started on [HuggingFace](https://huggingface.co/tngtech/DeepSeek-R1T-Chimera).
   - This model is fine-tuned between R1 and V3, showcasing its prowess in handling markdown files.
- **Cursor cries over crushing code context!**: Some users reported slow requests and inconsistent quality with **Cursor**, with one stating, *I really hate what is happening here*
   - Others suggested switching models (like DeepSeek 3.1 for instant results) or resetting the context, while some experienced a bug where the 'Apply All' button doesn't display when using Gemini on max settings.
- **Navigating Narrative Niftily**: Members discussed how **Cursor** handles linked documentation, explaining that it reads the HTML content of the linked page to gather information and links, while some pointed out using the [Cursor docs system](https://docs.cursor.com/context/model-context-protocol) is better for keeping all of your code.
   - Users wanted the ability to dynamically read more pages from a linked document, similar to a browser, with the team responding with an upcoming API plugin to send DOM to Cursor.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1373013206573912074)** (422 messages🔥🔥🔥): 

> `imatrix calibration dataset, Qwen3 GGUF issues, Estimate Cost of Running LLMs, Gemma 3 fine tune, AlphaEvolve the Google AI` 


- **Unsloth's imatrix calibration dataset dream come true**: Members rave about the **UnSloth Dynamic 2.0 GGUFs** and the *paradigm shifting, instruction and conversational focused imatrix* along with them.
   - It was stated that *the improved perplexity means faster token/s on response predictions and generations* for **Llama-3.1-8B-Instruct-UD-Q8_0_XL**.
- **Trouble running Qwen3 GGUFs prompts investigation**: Members reported they're *having some trouble running Qwen3 235 128k UD-Q6 and UD-Q8 ggufs* due to SHA hash mismatches and out-of-bounds errors.
   - The team stated they will investigate and that they have *updated all qwen3 uploads with the chat template fixes*.
- **Cost of running LLMs comparison**: A blogpost was shared [comparing the estimate cost of running LLMs](https://mburaksayici.com/blog/2025/05/15/llm-interviews-hosting-vs-api-the-estimate-cost-of-running-llms.html) for Hosting vs API.
   - Members stated that *the calibration dataset is very good and it took us 3 weeks or so to manually collect, clean and double check by hand*.
- **Fine-Tuning Convex Yields Third Place Finish**: A member's finetune on *convex*, a niche typescript backend, resulted in a **3rd place** score after evaluation, only **3%** behind **Claude 3.7 sonnet**, achieved with **Qwen3 14b finetune**.
   - They also emphasized *how IMPORTANT is HQ dataset*.
- **AlphaEvolve writes its own code**: A member shared an article about [AlphaEvolve, the Google AI that writes its own code](https://venturebeat.com/ai/meet-alphaevolve-the-google-ai-that-writes-its-own-code-and-just-saved-millions-in-computing-costs/) and saved millions in computing costs.
   - Another member wondered why **AlphaEvolve** can’t invent the next-gen transformer.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1373151204070395904)** (13 messages🔥): 

> `Downloading adapters from Google Colab, Private Hugging Face Models, Modern LLMs besides Qwen` 


- **Colab downloads cause consternation**: A member complained about downloading adapters from **Google Colab** being slow, and considered switching to another service.
   - Another member confirmed the slowness and suggested uploading to **Hugging Face** then downloading from there.
- **HF model privacy pondered**: A member inquired about how to upload a model to **Hugging Face** and have it be private by default.
   - Another member answered *just set, private=True*.
- **LLM landscape explored**: A member new to the local LLM game asked which models people use nowadays besides **Qwen**.
   - Another member mentioned that some still use old **Llama** models, but from the more recent ones recommended **Phi-4** or **Mistral Large**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1373013543770914879)** (659 messages🔥🔥🔥): 

> `TPU support, GGUF Saving Errors, Torch and Cuda errors, Unsloth Documentation, Continued Pretraining vs Lora` 


- **Unsloth Tool-Calling Notebook Spotted!**: Users discussed creating tools and agents using **Hugging Face** with models, and one member shared a [notebook for tool calling](https://docs.unsloth.ai/get-started/unsloth-notebooks#:~:text=Other%20important%20notebooks%3A-,Tool%20Calling%20%2D%20new,-ModernBERT%2Dlarge%20%2D%20new) on Unsloth.
   - Another member confirmed its applicability to coder models, noting that *most models here are good with tool calling, so will it work?*
- **Physical TPUs Spotted in the Wild!**: A user mentioned having a **TPU** lying around, leading to a discussion about the limited availability and feasibility of buying physical TPUs, since buying physical **TPUs** isn't really possible or feasible.
   - While buying physical **TPUs** isn't really possible nor feasible, Hyperbolic was recommended, as they offer **H100s** for **$0.99** and **H200s** for **$2.15**.
- **GGUF Saving Problems Triggered!**: A user encountered a *RuntimeError: Unsloth: Failed to convert llama.cpp/unsloth_convert_hf_to_gguf.py to GGUF* when trying to save to GGUF format.
   - The solution involves [merging the model before saving to GGUF](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal/gemma3.md), with a warning that the `save_pretrained_gguf` function might still be incompatible with the latest `llama.cpp` changes and pointed the user to helpful documentation.
- **Colab Users Hit With the Torch/Cuda Hammer!**: A user reported getting **CUDA errors** after updating Unsloth, which was traced to an older driver requiring a specific torch version.
   - The user found a solution by using `pip install "unsloth[cu121-torch250] @ git+https://github.com/unslothai/unsloth.git"` but expressed concerns about needing to nuke venvs with every Unsloth upgrade and [this user was self-curing](https://www.urbandictionary.com/define.php?term=self-curing).
- **Unsloth Documentation is the Ultimate Answer!**: A user asked for a binary classification notebook for Unsloth, and they were directed to the [Unsloth documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks), as it *has everything you need*.
   - Another member quipped *Do you really expect people to read docs before things explode?*


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1373190322409967666)** (19 messages🔥): 

> `PTS and DPO for Fine-Tuning, Beam Search with Trainable Permutations, Tokenizer Training and Embedding Research, Entropix GitHub Project` 


- **Pivotal Token Search Fuels DPO Fine-Tuning**: Members discussed that [Pivotal Token Search](https://huggingface.co/blog/codelion/pts) isn't RL itself, but it generates DPO pairs for fine-tuning, citing improvements shown in the **Phi-4 technical report**.
   - A screenshot ([Image Link](https://cdn.discordapp.com/attachments/1257011997250424842/1373218344232157264/Screenshot_2025-05-17_at_4.37.39_PM.png?ex=682ce87e&is=682b96fe&hm=685beeb66d15e5f4e1195735b7a8b1cc7fb4145dc5dfc8f9cd2863e&)) visually confirmed **PTS's positive impact on DPO performance** across benchmarks.
- **Trainable Beam Permutations**: A member likened a research paper ([https://arxiv.org/abs/2505.10475](https://arxiv.org/abs/2505.10475)) to *beam search with trainable beam permutations*.
   - Another member suggested it becomes a more powerful strategy the **larger the model is** and scales with **compute** instead of memory bandwidth.
- **Tokenizer Training Explored**: A member asked for research on training a custom tokenizer and embedding for existing LLMs, aiming to strip non-English tokens from the **Qwen tokenizer** and add more special tokens, citing the [ReTOK paper](https://arxiv.org/abs/2410.04335).
   - Another member suggested to search around **LLaVa** and similar to find research focused on adding new languages/modalities to existing models.
- **Entropix GitHub Project Resurfaces**: A member asked if anything came out of the [Entropix GitHub project](https://github.com/xjdr-alt/entropix), noting that it looked promising.
   - No further details or outcomes were provided, implying its status remains uncertain.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1373015169554321520)** (569 messages🔥🔥🔥): 

> `ASI Lab Plagiarism Accusations, Codex Experience, GPT-5, Gemini 2.5 Pro Math Performance, ChatGPT Memory Feature` 


- **ASI Lab Faces Plagiarism Claims**: Pro.creations claims [pro.creations](https://cdn.discordapp.com/attachments/998381918976479273/1373015169357054082/image.png?ex=682cd405&is=682b8285&hm=41d43e15460edc8f88272551f18cfec1fa74fb94c748b9efa15f5c82234bb031&) that the **ASI Lab's work was taken down for plagiarism** and that it is *pretty horrible to see a whole well respected university plagiarizing and slapping a fake agi name on it*.
- **Codex Access is a Gamechanger**: A member stated that Codex is a *gamechanger* and *staggeringly good*, noting that **it requires supervision** but others have run multiple agent threads simultaneously, it requires internet access and that [some people complain](https://cdn.discordapp.com/attachments/998381918976479273/1373141529303978075/image.png?ex=682ca0f4&is=682b4f74&hm=ecc62fa17cbb93ebfbeadcf07f169e859106154cc9b595469981e85b61899ac7&) it only handles a few hundred lines of code.
- **GPT-5 Speculations Heat Up**: Members speculated that **GPT-5 excludes o3** and will be like **Gemini 2.5 Pro**, an LLM and a reasoning model instead of the current system of the o-series and GPTs, and may drop sometime this summer, with another member suggesting to watch for a **Google week** event.
- **Gemini 2.5 Pro Math Performance Discussed**: Members discussed **Gemini 2.5 Pro's** performance in math, noting that the older model is better for math but that the new model is better for coding, with one providing the [official numbers from Google](https://deepmind.google/technologies/gemini/pro/) showing **Gemini 2.5 pro is 0506**, which has a performance of **83 vs 88.9%**.
- **ChatGPT's Memory Spurs Hyper-Personalization Debate**: Some members speculated on **ChatGPT's** new memory feature and whether they've switched to a **transformer–liquid neural network hybrid**, however some think it may be that the **advanced memory system is just RAG** and that advanced features are managed by systems external to the transformer and injected into the context at runtime.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1373040677574934629)** (414 messages🔥🔥🔥): 

> `Gemini 2.5 Pro vs 4.1, Rate Limits, GPT Lying, ChatGPT for Education, 4o Mini` 


- **Gemini 2.5 Pro vs 4.1, is it better?**: A member asked if **Gemini 2.5 pro** is better than **4.1**, which prompted a reminder to follow the [AI-Discussions rules](https://discord.com/channels/974519864045756446/1107255707314704505) to keep discussions on topic.
   - The question was determined to be off-topic, as it should have been placed in the appropriate channel.
- **Database Scans with OpenAI's API**: A member planned to send about **1000 prompts** to GPT via the API for a database scan and inquired about the likelihood of being blocked.
   - Another member responded that while **ChatGPT** has message limits, the **API** is subject to [rate limits](https://platform.openai.com/docs/guides/rate-limits/).
- **Debate Explodes over ChatGPT Lying Intentionally**: Members debated the nature of **ChatGPT** lying, with one suggesting that it is more sophisticated and capable of intentional deception due to its advanced nature, while another countered that its outputs are due to biased or incorrect training data and not intentional lying.
   - One member cited research indicating that **GPTs** can be programmed to lie, knowing what's true but giving the opposite, or exhibiting deception as a strategic pattern based on reward models, while another argued that attributing everything to hallucinations is a flawed way of thinking.
- **AI Assists Aspiring Engineers in STEM**: A member sought advice on using **ChatGPT** for education, specifically to prepare for university engineering studies, prompting suggestions for using mathematical reasoning models and visual aids like **YouTube** videos.
   - Another member suggested using **custom GPTs** to create a game-like learning experience, akin to **Duolingo**, to track progress and goals, leveraging **4o's** memory storage capabilities.
- **ChatGPT's 4o Mini Appraised for Versatility**: A member expressed a preference for **4o mini**, citing its nimble size and suitability for various use cases, while acknowledging a primary focus on image and video generation.
   - Others discussed **4o's memory features**, noting that it can store up to **100 entries** in the free version, allowing the model to remember user information and inject it into conversations over time.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1373014117346181211)** (48 messages🔥): 

> `HyperEnglish Prompting, AI Custom Instructions via Python, ProtoMind Semantic Mapping, Image Prompting Workflow, Learning Prompt Engineering` 


- **HyperEnglish Syntax Hypes Clarity**: A member introduced [HyperEnglish](https://example.com/hyperenglish), emphasizing *functional tagging* and **ALL-CAPS** for technical terms to prioritize clarity.
   - They shared an example template: `SUBJ:{subject} VERB:{verb} OBJ:{object} {additional_tags_as_needed}`.
- **AI Loads Custom Instructions with Python**: Members discussed dynamically loading custom instructions using **Python scripts**, allowing the AI to adjust its mode on the fly.
   - The AI can write a script to return the contents of a text file containing the instructions, enabling **weighted procedural responses**.
- **ProtoMind Maps Semantics, Skips Code**: One member lauded that *there is no need for explicit code* in [ProtoMind_001](https://chatgpt.com/g/g-682759c7c8d881919436d320e718d5d5), treating user input as **layered semantic operators** for role-switching and pseudo-memory threading.
   - Another member added that ProtoMind_001 was able to *map* them pretty well.
- **Image Prompting Workflow Iterates Visually**: A member shared their [image prompting workflow](https://example.com/image-prompting) for consistent character visuals in **TTRPGs**, starting with concept artifacts and iterative refinement using O3 for visual reasoning.
   - The process involves generating an **orthographic sketch**, critiquing it for clarity, and then using it to anchor multiple image prompts.
- **Mobile App Coders Ponder Prompt Engineering**: A mobile app coder inquired about the value of learning prompt engineering, considering the [Google Prompt Engineer Course](https://example.com/google-prompt-engineer).
   - Responses suggested focusing on clarity and organization, with one member advising against paying for such a course if free resources are available.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1373014117346181211)** (48 messages🔥): 

> `HyperEnglish, Meta-prompt generator complexity, Loading Custom Instructions, Weighted procedural responses, ProtoMind_001` 


- **HyperEnglish Improves Clarity**: A member shared their [HyperEnglish template](https://example.com/hyperenglish) which is designed to improve clarity via functional tagging and ALL-CAPS technical terminology, prioritizing clarity over naturalness.
   - The template enforces structured content such as `SUBJ:{subject} VERB:{verb} OBJ:{object}`.
- **Loading Custom Instructions on the Fly**: Members discussed [loading custom instructions](https://example.com/custom-instructions) on the fly using a Python tool to output adjustments, which can then be returned into the context.
   - The AI can create a script to load the instructions and run it, effectively changing the mode of operation, with one member noting *There is something really innovative that we came up with here.*
- **Agents Work as Semantic Runtime Operators**: A member stated that [agents work as semantic runtime operators](https://example.com/semantic-operators), emphasizing that there's no need for explicit code, just symbolic role-switching and pseudo-memory threading.
   - They treat user input not as queries, but as layered semantic operators, suggesting that it tracks contradictions, predicts evolution paths, and reshapes internal goals.
- **Iterative Image Prompting Generates Consistent TTRPG Visuals**: One member outlined an iterative process for [image prompting for TTRPG character visuals](https://example.com/ttrpg-image-prompting), starting with ideas, sketching, critiquing, and creating multiple consistent images.
   - The process includes using O3 for visual reasoning and generating an orthographic sketch before creating multiple images to ensure consistent results.
- **Weighing the value of Prompt Engineering Courses**: A user inquired about the value of [prompt engineering courses](https://example.com/prompt-engineering-courses) for a mobile app coder.
   - Another member suggested asking ChatGPT about prompt engineering and advised that free courses might be worth it if they help with organization, but cautioned against paying for such courses.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1373015087194832936)** (900 messages🔥🔥🔥): 

> `O3 Pro, GPT-5 speculation, Claude 4, DeepSeek's fate, Codex's potential` 


- **O3 Pro Delayed Indefinitely**: Users are expressing disappointment over the continued delay of **O3 Pro**, with some joking about resorting to a *hunger strike* until its release.
   - Some suggested that the original **GPT-4** was superior, possessing a *genuine intelligence/creativity spark* that newer, smaller models lack.
- **GPT-5 Speculation**: Some believe **GPT-5** will be a base model comparable to **O4**, but others are skeptical, suggesting it might only be a marginally improved version of **O3**.
   - There's discussion about whether it will be a model router or a new base model, and how RL training impacts improvement over stable versions.
- **Is Codex any good?**: Users are debating the utility of **Codex**, with some considering it *noise* for junior developers, while others see its potential for more advanced tasks.
   - One user suggested that **Codex** needs to compete with tools like **RooCode/Cline/Cursor/Windsurf/Aider** to be worthwhile.
- **Gemini and Claude face off**: There's ongoing debate about **Gemini** versus **Claude** for coding, with some finding **Gemini** annoyingly verbose and prone to adding unwanted features, while others praise **Claude** for its reliability.
   - Some users find **Gemini's** code comments to be a negative aspect, while others see them as beneficial.
- **Microsoft & XAI's future**: A user posted an image hinting at a Microsoft & XAI partnership.
   - Another user responded, *oh jeez*, linking to a social media post about the topic.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1373326966668656640)** (2 messages): 

> `Mistral Medium 3, Claude 3 Sonnet, Amazon Nova Pro, Command-a-03` 


- **New Models Invade Beta Site**: The LMArena Beta Site has added new models, including **mistral-medium-2505**, **claude-3-7-sonnet-20250219-thinking-32k**, **amazon.nova-pro-v1:0**, and **command-a-03-2025**.
- **Mistral Medium 3 Chart Climbs**: Since the debut of **Mistral Medium 3** on the leaderboards, it made an impressive leap of **+90** from Mistral Large landing at #11 overall in chat.
   - It performed in the top-tier in technical domains (**#5 in Math**, **#7 in Hard Prompts & Coding**) and **#9** in WebDev Arena.


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1373143224301912156)** (2 messages): 

> `Gemini 2.5 Pro Experimental deprecation, DeepSeek V3 maintenance` 


- **Gemini 2.5 Pro Experimental Sunset**: Google is deprecating **Gemini 2.5 Pro Experimental** (`google/gemini-2.5-pro-exp-03-25`) in favor of the paid **Preview endpoint** (`google/gemini-2.5-pro-preview`).
   - The model will be deprecated on OpenRouter shortly.
- **DeepSeek V3 Undergoes Tuning**: The free **DeepSeek V3 0324** will be down for maintenance for some time today.


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1373735131247808562)** (3 messages): 

> `Chess Tournament, Stockfish Implementations, Lichess Accuracy Ratings, Openrouter models` 


- **Chess Tournament gets Overhauled**: A member shared their project, a **chess leaderboard**, which evolved from a simple **chess tournament** idea to incorporating **Stockfish implementations** for accurate ratings.
   - The leaderboard replicates **Lichess accuracy ratings** and supports **Openrouter models** with temp and sys role capabilities, with workarounds for models like o1-mini, and is fully automated using cronjobs.
- **Chess Leaderboard is very cool**: A member said the **chess leaderboard** is *really cool* and a *different use case*.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1373028823049703484)** (790 messages🔥🔥🔥): 

> `API key identification, Gemini 2.5 Deprecation Fallout, Qwen3 Tool Calling Troubles, Low Latency LLMs, Gemini API Updates` 


- ****Request Tags** Needed for Shared API Keys**: A member inquired about identifying the source of API calls when multiple users share a single key, particularly for tracking disconnects mid-stream.
   - Suggestions included implementing *request tags* instead of embedding user IDs in the app name, to better log individual user requests.
- ****Vertex** to Bring Back Gemini 2.5**: Users mourned the deprecation of **Gemini 2.5 Pro Exp** (*03-25*) and expressed dissatisfaction with the lobotomized 05-06 version, with some hoping for its return.
   - One member sarcastically noted the lack of *serious outrage*, while others discussed experiencing content filtering and the unfortunate truth that non-open source models are ephemeral.
- ****Kluster's** Qwen3 Provider has Tool Calling Hiccups**: A user reported issues with **Qwen3 235B** and its tool calling capabilities when using the *Kluster* provider, noting that it prematurely ends tool calls.
   - They discovered that OpenRouter sometimes switches to another provider, resolving the problem, but forcing OpenRouter to use *Kluster* consistently results in failures.
- ****Anthropic** to Integrate OpenRouter Claude****: A user alleged that OpenRouter's Claude implementation is a *scam*, delivering a simplified experience with excessive follow-up questions compared to using Claude directly through Anthropic.
   - Others countered that the difference arises from Anthropic's system prompts, which are absent by default in OpenRouter, and that the raw model performance remains identical. 
- **LLM Service Optimization to Minimize Latency**: A member proposed that OpenRouter optimize the network path and routing to minimize latency, offering services with different levels of service level agreements.
   - They suggested providing colocation or hosting guidance, and levers to pull to specify *'I'll pay up for maximum speed, route me accordingly'* for workflows where speed is paramount.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1373048049042985020)** (174 messages🔥🔥): 

> `VRAM Usage, LM Studio conversations export, LM Studio blurry UI, Vulkan runtime troubleshooting, Prompt formatting issues` 


- **LM Studio VRAM Usage Questioned**: Users reported that while LM Studio displays correct VRAM amount, the actual VRAM usage is very low even when offloading model layers to the GPU, with one user reporting it uses only *256-512MB* of VRAM on a 9070 machine.
   - They are further investigating potential driver issues and trying to determine whether the model is loading into dedicated VRAM or shared video memory, with some suspecting it could be a bug in the drivers.
- **Users Need Ways to Export LM Studio Conversations**: Users are looking for ways to export LM Studio conversations in an exploitable way, beyond the current JSON format, particularly for storytelling purposes.
   - They suggested using the LLM to write a utility to parse the JSON format into a preferred format, while acknowledging the lack of API guarantees for the JSON format.
- **LM Studio UI Displaying Blurry Sections**: A new LM Studio user reported a problem with the UI, describing blurry sections that appear when the mouse is removed from certain areas inside the application.
   - Other users asked what the version of LM Studio in question was to try and reproduce or identify the underlying issue.
- **Newest LM Studio Version Needed for Prompt Formatting**: Users running **qwwn3 -32b** models encountered parsing errors related to the prompt template, indicating issues with the model's prompt template.
   - The fix was to [upgrade to the latest beta build of LM Studio](https://lmstudio.ai/beta-releases) which uses a different format for the preset files for prompt formats.
- **Context Overload Causes Models to Wily Wonka**: One user sought methods to prevent models from exceeding context limits while maintaining consistent character details and past event knowledge, suggesting external document storage.
   - It was suggested to summarize the conversation, use the "truncate middle" option, and provide the model with clear instructions on what to remember.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1373012188456816821)** (341 messages🔥🔥): 

> `Intel Arc Pro B60 GPU, macOS vs Windows, AMD vs Nvidia GPU for LM Studio, Resizable BAR Impact, Multi GPU setup` 


- **Intel's Arc Pro B60 GPU Sparks Interest**: Members are excited about the [Intel Arc Pro B60](https://www.intel.com/content/www/us/en/products/sku/243916/intel-arc-pro-b60-graphics/specifications.html) due to its **96GB VRAM** and potential price point **below $2000**, making it an attractive option for local LLM.
   - Concerns were raised about the potential software support challenges, with the hope that the availability of a high VRAM card would **increase support** for Intel GPUs in the AI space and push providers to up their game.
- **macOS Fluidity Wins Over Some, Windows Still Preferred By Others**: A member expressed that using **macOS** on a Macbook is more fluid and pleasurable than Windows, while another member shared issues with MacOS window sizing and application management.
   - Another member highlighted that a Macbook can run a **32B 4-bit model for >5 hours** on battery at 5t/s, and with a system RAM usage of <2GB, it provides an efficient and silent running LLM experience.
- **9070XT GPU struggles and requires Resizable BAR**: A user reported poor performance with a **9070XT** GPU, getting only *8.52 t/s* on a **Gemma3 4b QAT** model after a fresh install of Windows.
   - After enabling **Resizable BAR** (also known as **Smart Access Memory (SAM)** on AMD), the user achieved *131.91 t/s*, highlighting the significant performance impact of this setting on LM Studio.
- **Uneven GPU Performance with LM Studio**: Members discussed their experiences with split GPU performance in LM Studio, with one member noting that performance on Linux minimizes the multi-GPU *'penalty'* compared to Windows.
   - Another member shared how they used a **Deepseek R1** model to fix a software bug at work where smaller models failed, highlighting the benefits of using larger models despite slower token generation speeds.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1373015324890365962)** (225 messages🔥🔥): 

> `MCP Course Channel, AI Integration on ERP System, ACE Step Quality, AI Model for Design-to-Code Conversion, Hugging Face Pro Benefits` 


- **MCP Course Channel MIA**: Several members are seeking the **MCP course channel**, but it doesn't appear to be available yet, with links to the [course on GitHub](https://github.com/huggingface/mcp-course) and related Discord threads being shared.
   - Confusion persists over whether the course is part of the **Agents course**.
- **AI + ERP = Integration Conundrums**: Members are curious about experiences integrating **AI with ERP systems**, sparking a brief discussion.
   - One member provided a clarification of what **ERP** is: *Enterprise resource planning*
- **HF Pro Benefits Glitch for AI Engineers**: A user reported that the **AI Engineer Pack link for 6 months of free Hugging Face Pro** isn't working.
   - The user was advised to contact HF support at [website@huggingface.co](https://discord.com/channels/879548962464493619/1353741359059570699).
- **Xet Infrastructure and File Size Limits**: A **Xet team member** addressed questions about file size limits, noting that while theoretically files **>50GB** can be uploaded and downloaded using **hf_xet**, full web support and final design decisions are pending.
   - A request was made for a **200GB limit** to accommodate 70B models.
- **Dropwise Module: Estimate Uncertainty in HF Classifiers**: A member introduced **Dropwise**, a PyPI module for **uncertainty estimation** in Hugging Face classification models using **Monte Carlo Dropout**.
   - It features predictive entropy, per-class variance, confidence scores, and works with **transformers pipelines**; find it on [GitHub](https://github.com/aryanator/dropwise) and [PyPI](https://pypi.org/project/dropwise/).


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1373213142053687336)** (6 messages): 

> `MCP for Atlassian, ChatApp AI App` 


- ****Atlassian** MCP Port to Claude Desktop Proposed**: One member is working on adapting existing **MCPs** for Atlassian tools (**Jira/Confluence**) to run with Claude Desktop and is [seeking collaborators](https://www.atlassian.com/)
   - This project aims to build **MCP servers** for integrating **JIRA/Confluence** with Claude, enabling it to analyze, create, modify, and fetch details from JIRA tickets within conversations.
- **New AI ChatApp Project**: A member plans to develop an AI ChatApp that responds based on a provided dataset, such as conversation records.
   - The member is [seeking guidance and research topics](https://www.example.com) to aid in the app's development.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1373049047484465173)** (4 messages): 

> `Strands Agents SDK, AI for OS Funding` 


- ****Strands Agents SDK** Debuts!**: Amazon introduces the [**Strands Agents SDK**](https://aws.amazon.com/blogs/opensource/introducing-strands-agents-an-open-source-ai-agents-sdk/), a well-thought-out **open-source AI agent SDK**.
   - The SDK aims to simplify the creation of AI agents, offering developers tools and resources to build and deploy agents more efficiently.
- **Funding Opportunity: **AI for Open Source****: A member shared a funding opportunity for startups in the data science space via the [**AI for OS**](https://os.nav.fund/ai-for-os/) initiative.
   - The initiative seeks to support innovative projects that leverage AI to enhance open-source technologies, providing crucial funding for early-stage ventures.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1373012620264607824)** (14 messages🔥): 

> `EcoArt Cellular Automaton, Browser.AI with Tool Calls, tome: Local LLM Client with MCP Servers, Lazarus Small LLM, datatune Open Source Tool` 


- **EcoArt Cellular Automaton Visualized**: The poster introduced the [EcoArt Cellular Automaton](https://huggingface.co/spaces/KvnMln/Mechanistic-interpretable-Ethics-Cell-automata) which merges the beauty of nature with the complexity of systems thinking to make virtue values tangible and observable for **education**, **research**, **art**, **development**, **reflection**, and **meditation**.
   - The project aims to explore how values shape systems.
- **Browser.AI Adds Tool Calling**: A member announced a new release of [Browser.AI](https://browser.christophermckenzie.com/), a prototype browser demonstrating the power of running open source models on device, now with support for **chat**, **tool calls**, and **embeddings**.
   - The poster is looking for feedback on the new release.
- **tome: Local LLM Client with MCP Servers Launches**: A member introduced [tome](https://github.com/runebookai/tome), a simple local LLM client that connects to **Ollama** and allows users to add/use MCP servers without complex configuration.
   - The client integrates a **Smithery Registry** for one-click installation of thousands of MCP servers and the developers welcome feedback for improvement.
- **Lazarus: the next small LLM**: A member shared [Aclevo/Lazarus](https://huggingface.co/Aclevo/Lazarus), calling it the next best small LLM, [distilled from Llama3](https://huggingface.co/blog/codelion/pts) with approximately **124 million parameters**.
- **datatune Transforms Data via Natural Language**: A member from Vitalops introduced [datatune](https://github.com/vitalops/datatune), a new open source tool that does data transformations using simple natural language instructions and LLMs.


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/)** (1 messages): 

arpitbansal.: By any chance recording available for the recent session??
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1373190699490738186)** (13 messages🔥): 

> `WANVideo Lora Training, Invoice Extraction, Computer Vision Roadmap, Object Outlines, CS231n Lectures` 


- **WANVideo Lora Training Loss Plummets**: A member training a **WANVideo Lora** on a dataset of **130 videos** shared their [epoch loss](https://cdn.discordapp.com/attachments/922424143113232404/1373190699301998652/image.png?ex=682ccebf&is=682b7d3f&hm=56fa22b904be374ca12df36afe401f115988a2a189aa239a2ca7c685fd17095&) after **26 epochs** over **8 hours**, using a **batch size of 2** with **gradient accumulation steps = 2**.
   - Another member commented that for most image models, *loss means very little* and suggested periodically sampling to see how the results look, also recommending using **10%** of the dataset as a validation dataset.
- **Invoice Extraction Seeks Structured Process**: A member is seeking a structured process for **invoice extraction**, specifically extracting entities as **key-value pairs** and organizing **table data**, having found OCR and LayoutLM outputs unsatisfactory.
   - Suggestions included preprocessing, OCR, LayoutLM/Donut, NER, and post-processing, but the member is *struggling to implement LayoutLMv3 effectively*.
- **Computer Vision Roadmap Wanted**: A member requested a neat and structured roadmap/resources to specialize in CV topics such as **Object Detection**, **Semantic & Instance Segmentation**, **Object Tracking**, and **3D Computer Vision**.
   - The member has a good grasp on **Computer Vision Basics with OpenCV**, **Mathematical Foundations**, **Machine Learning Foundations**, and **Deep Learning for Computer Vision** and is comfortable with **Python** (**Tensorflow** and **Pytorch**).
- **Object Outline Models Requested**: A member inquired about models good at getting the **outline of an object** from an image or **BREP** of an object.
   - No specific models were recommended in the provided messages.
- **Karpathy's CS231n Lectures Recommended**: A member suggested going through the **Stanford CS231n lectures** by **Andrej Karpathy** for building intuition.
   - The member also recommended watching or reading through something which explores classic computer vision and machine learning, such as **Andrew Ng courses** on YouTube.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1373368517029662900)** (3 messages): 

> `Modern Approaches to DDS, Inference Differences in BERT-style Models` 


- **Dewey Decimal System declared Relic**: A member shared the opinion that classifying data using the **Dewey Decimal System** is a relic of the past due to the sheer volume of data today.
   - They also mentioned exploring more modern approaches and alternative classification techniques, and provided a [PDF on a universal framework for data](https://cdn.discordapp.com/attachments/922424173916196955/1373368516719415437/02._From_Chaos_to_Order__The_Universal_Framework_for_Data.pdf?ex=682ccb9a&is=682b7a1a&hm=d621f23829765a944006f237a349ba321f9d3f4e7ca6aaed225340f62fe95f81&).
- **BERT's Inference Gives Varied Results**: A member asked if it's unexpected to get significantly different logits when running inference on the same **BERT-style model** and task using different libraries (Candle and PyTorch), even with identical tokenization and config files.
   - The question implies that while logits differ, classification results remain mostly consistent, the member stating, *"If I run inference on the same BERT-style model, on the same task, in two different libraries... is it unexpected to get different significantly logits?"*


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1373114438755942400)** (11 messages🔥): 

> `Claude "overloaded" issues, GPT's 30% success rate, Meta Llama access denied, Multiple agents setup, Questions endpoint problem` 


- **GPT solves Claude's "Overloaded" Submission Snags**: A member reported spending a day and a half struggling to push their submission through with **Claude** due to being *"overloaded,"* but switching to **GPT** immediately solved the issue and achieved a **30%** success rate.
   - Despite being a fan of **Claude**, they had experienced *"a number of issues with it over the last eight or so days."
- **Meta Llama Models Denies Access to Course Takers**: A member was denied access to the **Meta Llama Models**, preventing them from working on the course and notebooks.
   - Another member suggested using alternative models like **Qwen** or **Gemma** as replacements.
- **Multi-Agent Setups Demand Prompting**: A member mentioned experimenting with a **multi-agent setup** by separating tools into multiple agents, but it *"did not work well just like that."*
   - They emphasized the need for changes to tool and agent prompts, stating, *"Its a pity there are close to no examples online except very basic ones."
- **Questions Endpoint Missing Attachment Files**: A member reported that the **questions endpoint** for the final assignment is missing attachment files (e.g., **.py**, **.xlsx**).
   - When hitting the questions endpoint, the user only receives a **JSON** response, without the expected attachments.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1373027781499158609)** (46 messages🔥): 

> `Agents Course Certifications, MCP Course Confusion, Unit 4 Project File Retrieval, GAIA Formatting Issues, Hugging Face space stuck` 


- **Agents Unite to Achieve Certs**: Several members celebrated completing their **Unit 4 agent** and obtaining their certifications for the Hugging Face Agents course.
   - Other new members are also forming study groups to complete the course and earn certification before the deadline.
- **MCP Course Mix-Up Mars Discussion**: Members clarified that the **MCP (presumably, Machine Learning Certification Program) course** is distinct from the **Agents course**, advising to seek details in a separate (non-existent) MCP channel.
   - Others humorously remarked that they might complete the **Agents cert** while waiting for information on the **MCP course**.
- **File Frustration: Agent's Unit 4 Files**: A member inquired about retrieving files such as **.png** or **.mp3** for tasks in the **Unit 4 project**.
   - A different member provided the relevant code snippet: `files_url = f"{api_url}/files/{task_id}"`.
- **GAIA's Gotcha: Exact Match Mandate**: One member highlighted the tedium of **GAIA's** requirement for exact matches, noting that the evaluation format is not flexible, and another member suggested `massaging the system message`.
   - Members suggested referring to the [suggested system prompt](https://huggingface.co/spaces/gaia-benchmark/leaderboard) to resolve **GAIA** formatting issues and improve scores.
- **Space Stuck? Assessment Anxieties Arise**: A member reported their final assessment space ([https://huggingface.co/spaces/vaibhav-vibe/agents_final_assessment](https://huggingface.co/spaces/vaibhav-vibe/agents_final_assessment)) was stuck building, and asked the community if there are any resolutions.
   - Another shared a potentially relevant [ChatGPT link](https://chatgpt.com/share/682b5764-9624-8008-b387-4532bdae4fc6), while the space creator mentioned trying a factory rebuild and hardware switch without success.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1373386018639712426)** (8 messages🔥): 

> `Kernel development, AMD challenge, FSDP and Flash Attention 2 with trl` 


- **Kernel from scratch project seeking contributors**: A member is building a kernel from scratch and is looking for contributors, with more details available on [GitHub](https://github.com/DorsaRoh/Kernels).
   - The project is explicitly tagged as "gpu mode = 🔥" suggesting a focus on GPU-related kernel development.
- **AMD Challenge participation difficulties**: A participant in the **AMD challenge** is experiencing issues with being added to the leaderboard channel and is seeking guidance on who to contact.
   - Other members suggested checking specific channels for permissions and access, [<#1359640791525490768> and <#1343002583001726986>] which helped to solve the issue.
- **Combining FSDP and Flash Attention 2**: A member is seeking advice on how to simultaneously train a model with `FSDP` and `Flash Attention 2` using `trl`, as they can get each to work independently but not together.
   - They referenced the [trl `sft.py` script](https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py) and shared the error message received when attempting to use **Flash Attention 2.0** with a model not initialized on the GPU.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1373202465524154509)** (4 messages): 

> `Triton Runtime Shared Memory, Triton on CPU, TRITON_INTERPRET API` 


- **Runtime Shared Memory in Triton Questioned**: A member inquired about Triton's memory size calculation, expressing that it should be a compile-time calculation rather than a runtime one.
   - Another member responded that Triton doesn't directly support CPU parallelism, but using the `TRITON_INTERPRET=1` API would allow sequential execution on the CPU.
- **Running Triton on CPU**: A member asked about the specific flag or device API needed to run Triton on the CPU on Ubuntu v22.0, with the eventual goal of running on a GPU.
   - They were also looking for a small code example for CPU testing such as **matmul**, **fastmul**, or **vector-add**.
- **Parallel scheme imitated**: One member suggested that although TRITON is not directly supported in CPU, it can imitate the parallel scheme almost perfectly.
   - They suggested this *would work for your purposes*.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1373101320529121300)** (11 messages🔥): 

> `Tensor Cores, CUDA Brute Forcer, GPU Usage Reporting, Neural Net Mutate Function, CUDA Errors` 


- **Smileforme Asks about Systolic Arrays**: Smileforme inquired whether **tensor cores** are implemented as **systolic arrays**.
- **CUDA Brute Forcer's Low GPU Usage Baffles**: kr1v reported creating a [CUDA brute forcer](https://github.com/kr1viah/WKChallengeModeSeedFinder) but observed only **0-20% GPU usage** in Task Manager, and was trying to figure out why.
   - mrsteed suggested that *Windows Task Manager doesn't 100% reliably report GPU usage*, suggesting `nvidia-smi` for more reliable stats.
- **Task Manager Settings Found**: ug0x01 suggested switching Task Manager to **Cuda** to see the **exact usage**, showing how to select **Cuda** in task manager [with an image](https://cdn.discordapp.com/attachments/1189607726595194971/1373842591036084264/image.png?ex=682c8ade&is=682b395e&hm=c63888438da7eef855829439bba3e467b828a5b5c101364d01e033bd1cdb21e7).
- **Neural Net Mutation Muddles Member**: A member is struggling with a **mutate function** for a **neural net**, encountering *unspecified launch failure* or *illegal memory access* errors at random generations.
   - They used *memcheck* and it showed *Malloc/Free Error encountered : Invalid pointer to free* and provided a [godbolt link](https://cuda.godbolt.org/z/z8z6a85vP).


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1373057306438271050)** (1 messages): 

> `Triton Kernels, Dynamic Shapes, Batch Size in PyTorch` 


- **PyTorch Defaults to Triton Kernel for Dynamic Shapes**: When the batch size is set to **None**, PyTorch defaults to a **Triton kernel** (`extern_kernels.mm`) that supports [dynamic shapes](https://pytorch.org/docs/stable/generated/torch.jit.script.html).
   - The system does not pad activations for autogen kernels; instead, it calls `extern_kernels` with a `s0` parameter indicating the batch size.
- **PyTorch Generates Specific Triton Kernels for each Batch Size**: For specific batch sizes, PyTorch generates **custom Triton kernels**, such as `triton_tem_fused_mm_4`, instead of using the generic dynamic shape kernel.
   - This optimization avoids padding and allows tailored kernel execution for each defined batch size, potentially improving performance.


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1373373636248998058)** (1 messages): 

> `CuTe, Cris Cecka` 


- **CuTe Inventor Speaking in 5!**: Cris Cecka, the inventor of **CuTe**, is starting a talk about **CuTe** in 5 minutes.
- **Another CuTe Summary**: Just adding another summary to satisfy the minimum requirement.


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1373166881904529470)** (43 messages🔥): 

> `Power Consumption for MCMC, Pbit Hardware, Analog Annealing Circuit, Quantum vs Pbit, Hardware for MCMC` 


- **Members Seek MCMC Power Consumption Resources**: A member requested more resources on **power consumption for MCMC** algorithms, citing [this paper](https://arxiv.org/pdf/1903.11714) as an example.
   - In response, links were shared to [three relevant papers](https://arxiv.org/abs/2411.04260),  [another paper](https://arxiv.org/abs/2002.01184), and [one more paper](https://arxiv.org/abs/2003.02629).
- **Pbit Hardware Powers Up Probabilistic Computing**: Discussion arose around probabilistic bit (**pbit**) hardware and its potential for **power efficiency** in probabilistic algorithms after [this interview](https://www.youtube.com/watch?v=5O5do_N07kY) was shared.
   - A member who prototyped an **analog annealing circuit** for combinatorial optimization stated that their *circuit should be around 100x less* in energy per flip than existing methods.
- **MCMC Hardware Hunt Kicks Off**: Members investigated optimal hardware and kernels for **MCMC**, with one looking into using **FPGAs**, but found that classic **Xilinx** models weren't viable.
   - The main challenge is creating **fast**, **high-quality randomness** with **low power consumption**, balancing randomness quality with bias in sampling.
- **TPU Triumph Over Nvidia?**: A member recalled reading that **Google TPUs** have an **MCMC advantage** over **Nvidia GPUs**, citing [this paper](https://arxiv.org/pdf/1903.11714) which compared TPUs against GPUs.
   - It was noted that **JAX** has conveniently good features for **parallel computing**, which gives NumPyro and BlackJAX libraries a JAX-based backend.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1373532531126833172)** (2 messages): 

> `External CUDA Allocator, MAXSUN Arc Pro B60 Dual` 


- **External CUDA Allocator unveiled**: A Github user shared a [blogpost](https://kshitij12345.github.io/python,/pytorch/2023/02/26/External-CUDA-Allocator-With-PyTorch.html) detailing how to use an **external CUDA allocator** with PyTorch.
   - The user demonstrates how to extend PyTorch's memory management capabilities.
- **MAXSUN Arc Pro B60 Dual discussed**: A user shared a [YouTube video](https://www.youtube.com/watch?v=Y8MWbPBP9i0) showcasing the **MAXSUN Arc Pro B60 Dual**.
   - The video seems to provide an overview or review of the product.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1373737452396609648)** (1 messages): 

> `Threading APIs, cuper alternative` 


- **cuper no longer available?**: A member asked what threading APIs are best for profiling real-time memory hierarchy.
   - They added that **GPT** mentioned that **cuper** is good but it seems like it's no longer available.
- **Real-time memory hierarchy needs profiling**: The inquiry focused on identifying suitable threading APIs for analyzing real-time memory hierarchy performance.
   - The aim is to find alternatives to **cuper**, which was suggested by **GPT** but is reportedly unavailable, for effective memory hierarchy profiling.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1374028977496260608)** (3 messages): 

> `QAT with Llama3.2 3B, prepare_model_for_qat, prepare_model_for_ptq, bf16 vs int4, axolotl-ai-cloud/axolotl` 


- **QAT with Llama3.2 3B: A Deep Dive**: A member sought assistance debugging **Quantization Aware Training (QAT)** with **Llama3.2 3B**, noting that the **QAT-trained, quantized model** didn't outperform a **full fine-tuned and then quantized model**.
   - Another member responded by sharing their relevant [Axolotl config](https://github.com/axolotl-ai-cloud/axolotl/pull/2590/files#diff-29a7a53548f024812e8a2dc36eccc625c6b278b22b105f4eb5a924c63452a781) and torchtune config ([gist.github.com](https://gist.github.com/andrewor14/f1121b9b4c2ccc50e0cc1726859eb79e)).
- **QAT vs Baseline: Dissecting the Fine-Tuning Flows**: The members detailed their workflows for **QAT** and **baseline models**, emphasizing that the sole difference lies in the absence of `prepare_model_for_qat` in the baseline approach.
   - Both workflows involve loading a **bf16 model** from HF, fine-tuning on alpaca, saving the **bf16 fine-tuned model**, applying `prepare_model_for_ptq` to convert to **int4**, and then evaluating with lm_eval.
- **Requesting Commands: A Call for Replication**: A member requested the commands used for fine-tuning, quantization, and evaluation to replicate the experiments and identify potential discrepancies.
   - The goal is to pinpoint the root cause of the performance disparity between **QAT-trained** and **baseline models**.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1373155113077833858)** (3 messages): 

> `kpack argument in rocm triton, AMD Triton Performance Optimization` 


- **Debate over `kpack` Argument in ROCm Triton Arises**: A user inquired about the `kpack` argument in **ROCm Triton**, noting that Torch Inductor sets it to **2** by default, referencing a [line of code](https://github.com/pytorch/pytorch/blob/e802b29ed499cdeba24b366830a1c76d4d8b8511/torch/_inductor/template_heuristics.py#L55).
   - The user found an explanation in the [ROCm Triton documentation](https://github.com/ROCm/triton/wiki/General-Guide-of-AMD-Triton-Performance-Optimization#kpacktilelang) where `kPack` is used for **MFMA** with `ld128`, but may not significantly impact performance.
- **kpack details**: kpack is used for MFMA with ld128
   - It probably doesn't significantly impact performance according to the docs.


  

---


### **GPU MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1373612104007028906)** (4 messages): 

> `CuTe Tensors, Lecture Slides Availability` 


- **CuTe Tensors Use Arbitrary Nested Layouts**: A member inquired about the **CuTe tensor library**, specifically how it promotes strides to arbitrary nested tuples of ints to support various layouts for tensor core **GEMMs**.
   - They asked if there are any downsides to these nested layouts and if tensor libraries like **PyTorch** could adopt them.
- **Lecture Slides Are Coming Soon**: A member inquired about the availability of lecture slides for the most recent series of lectures.
   - Another member responded that they have asked for them and they should be out soon, and asked for slides to be uploaded for all lectures since Lecture 43.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1373068325642567781)** (3 messages): 

> `SWEBench, CuTeDSL, AI Efficiency with Pruna AI` 


- **Ofir Press Talks SWEBench/SWEAgent**: Ofir Press will give a talk on **SWEBench/SWEAgent** on Wednesday at the [PyTorch Webinar](https://pytorch.org/event/towards-autonomous-language-model-systems/?utm_campaign=6571433-PyTorch+Webinar+Promotion&utm_content=332357919&utm_medium=social&utm_source=twitter&hss_channel=tw-776585502606721024).
   - The talk focuses on **autonomous language model systems** and will likely cover benchmarking strategies.
- **CuTeDSL Layout Algebra Exposed**: The **CUTLASS** team released **CuTeDSL**, and one member explains **CuTeDSL** alongside math concepts in a [blogpost](https://veitner.bearblog.dev/bridging-math-and-code-cute-layout-algebra-in-cutedsl/).
   - The blog post covers underlying math concepts like **Layout Function**, **Coalescing**, and **Complementation**.
- **Pruna AI Preaches Practicality**: Pruna AI now regularly organizes events on **AI efficiency**, including monthly webinars and in-person meetups.
   - The next webinar will be on how to **compress and deploy AI models on clouds** ([link](https://app.livestorm.co/pruna-ai/pruna-koyeb-event?utm_source=Livestorm+company+page)), and the first in-person event will be on May 28.


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1374018914761183293)** (6 messages): 

> `KernelLLM, PyTorch backend, RL baseline, leaderboard competitions, pass@k evals` 


- **Facebook Releases KernelLLM**: Facebook released its first public weights for [KernelLLM](https://huggingface.co/facebook/KernelLLM).
   - A member linked to [talks with more detail](https://www.youtube.com/watch?v=FtgXueoQkA0), noting that their newer baseline is much stronger than what they presented.
- **KernelLLM Plans Revealed**: A team member outlined next steps for KernelLLM including a **PyTorch backend** as an eval suite, **RL baseline**, more **leaderboard competitions**, and baseline models trained on leaderboard data for translation and edit tasks.
   - They added that this should be written down somewhere more visible.
- **Pass@k Evals Questioned**: A member questioned the use of comparing many different **pass@k** values.
   - They noted that it seems weird and wondered if there were better evals, especially when it's still possible to do **pass@k evals** for reasoning models.


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1374059255325261854)** (3 messages): 

> `LLMs, Qwen 2.5 3B, Llama 3.2 3B, GSM8K and MATH benchmarks` 


- **Reasoning Gym's Training and Evaluation Configs ready!**: A member shared that a paper will be on Arxiv in the next few days with training and evaluation configs at [Reasoning Gym](https://github.com/open-thought/reasoning-gym/tree/main/training)!
   - The readme contains info and reproduction instructions, however, full eval results will come with the full writeup.
- **RG Mathematics tasks enhances Benchmarks**: Some smaller LLMs (**Qwen 2.5 3B** and **Llama 3.2 3B**) were trained on RG reasoning data.
   - Training on a range of RG mathematics-related tasks improved performance on **GSM8K** and especially **MATH** benchmarks.
- **Qwen outperforms Llama on reasoning data**: Generally **Qwen** seemed more capable of learning from the data, compared to **Llama**.
   - There will also be some results for zero-shot performance of frontier LLMs, showing the value of the data for eval/benchmarking as well as training.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1373015422978494484)** (111 messages🔥🔥): 

> `amd-fp8-mm Leaderboard, amd-mixture-of-experts Leaderboard, Submission Errors on MoE, amd-identity Leaderboard, hipcc Arguments` 


- **FP8-MM Faceoff on MI300**: Several submissions were made to the `amd-fp8-mm` leaderboard with timings ranging from **150 µs** to **3.76 ms** on **MI300**, with multiple users achieving personal bests.
   - One user snagged **4th place** with **154 µs**, while another secured **3rd place** at **150 µs** and others landed in **6th place** with times around **160 µs**.
- **MoE Mania on MI300**: The `amd-mixture-of-experts` leaderboard saw a flurry of activity, with submissions varying from **255 ms** to **7564 ms** on the **MI300**, and one user reaching **first place** with a blazing **9.64 ms**.
   - Users celebrated new personal bests, pushing the limits of the **MI300**'s capabilities, and some even getting to **4th place** with only **14.7 ms**.
- **Submission Snafus Plague MoE**: One user reported an *"Error during creation of submission"* specifically when submitting files for `amd-mixture-of-experts` via Discord, particularly when the file size exceeds **10KB**.
   - Others noted that submission failures also occur when files contain tabs (`\t`) or newlines (`\n`), but not backslashes (`\`).
- **Identity Insights on MI300**: The `amd-identity` leaderboard witnessed a new champion, achieving **first place** with a swift **5.50 µs** on the **MI300**.
   - Other users made successful submissions, hovering around the **20 µs** mark, with one user also snatching **first place** at **6.79 µs**.
- **Hipcc Hooligans Handling Arguments**: A user requested the removal of specific `hipcc` arguments (*`--offload-arch=gfx900 --offload-arch=gfx906 ...`*), citing their incompatibility with specialized instructions on the **MI300**.
   - Another user proposed using *`os.environ['PYTORCH_ROCM_ARCH'] = 'gfx942'`* in Python to address the issue.


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1374077530419494972)** (1 messages): 

> `Problem #3 Released, MLA crash course, Bot Timeouts reduced, Due Dates Extended` 


- **Problem #3 hits Leaderboard**: Problem **#3** is now live on the [leaderboard](https://www.gpumode.com/leaderboard/463) thanks to the hard work of some members.
   - The authors aimed to make the problem simple but interesting.
- **Multi-head Latent Attention Decoding Guide Released**: A member wrote a guide for **Multi-head Latent Attention (MLA) Decoding** which is available [here](https://stormy-sailor-96a.notion.site/Multi-head-Latent-Attention-Decoding-1f0221cc2ffa803dbe1acb16bb510a40?pvs=74).
   - The guide serves as a crash course on MLA.
- **Bot Timeouts Reduced**: The bot has been updated to make timeouts less likely thanks to some members.
   - No further details were provided.
- **Problem Due Dates Extended to June 2**: Due to the late release of Problem **#3**, the due date for ALL problems has been extended to **June 2**.
   - This extends the deadline by **2 weeks**.


  

---


### **GPU MODE ▷ #[ppc](https://discord.com/channels/1189498204333543425/1343373905783554048/1373662486506635335)** (1 messages): 

> `llvm-mca usage, CPU execution estimation` 


- **`llvm-mca` estimates CPU Execution**: A member shared [an example](https://ppc.cs.aalto.fi/ch2/v5asm/) showcasing how to use `llvm-mca` to estimate how assembly code executes on a CPU.
- **LLVM-MCA Tool Usage**: An example demonstrates the use of the `llvm-mca` tool to estimate the execution of assembly snippets on the CPU.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1373045026518732982)** (14 messages🔥): 

> `Space Age Compatibility, Long Term Vision for Work Group, vllm and openai client classes, Understanding use-cases and evaluation areas` 


- ****Weekly Meeting** is Live**: The team is holding **weekly meetings** open to everyone at **16:00 UTC** on **Wednesdays** (**9 AM pacific, 5 PM London**).
   - They encourage everyone to **DM** with their email address to be added to the recurring invite.
- **Member finds inspiration from Space Age Compatibility**: A member found a potential solution for one of the open issues regarding **space age compatibility** at this [github.com/snarf-dev/fsm](https://github.com/snarf-dev/fsm/tree/main/screenshots) link.
   - He says *The impossibility is definitely a feature in this case as that means the benchmark will be impossible to saturate*.
- **Understanding Long Term Vision for Work Group**: A member read the article ([https://arxiv.org/pdf/2503.09617](https://arxiv.org/pdf/2503.09617)) and chat history and is a little confused about what the **mid to long term vision** for this work group is.
   - Another member suggests that their first goal is to understand which **interesting use-cases and evaluation areas** this type of an (unbounded) environment unlock.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1373060748313624737)** (85 messages🔥🔥): 

> `Mixture-of-experts Submission Issues, HIP Submissions, Popcorn CLI Output, Leaderboard Run Slower than Benchmark, Composable Kernel Library Error` 


- ****MoE Ranked Submissions Encounter Errors****: Users reported issues with ranked submissions of **Mixture-of-Experts** configurations, encountering an *"unexpected error"* after **10 minutes** of processing.
   - A fix was implemented and should be available soon: [reference-kernels commit 7d8a576](https://github.com/gpu-mode/reference-kernels/commit/7d8a57661a684f6a11270e4855179df5d0f1dff1).
- ****HIP Code needs Python Wrapper****: Submissions using **HIP** require a **Python script** to call the kernels, as pure **C++** submissions are not supported, example here: [template-hip.py](https://github.com/gpu-mode/reference-kernels/blob/main/problems/amd/fp8-mm/template-hip.py).
- ****Popcorn CLI Gets File Output Option****: A user requested the ability to output benchmark results to a file directly from the **Popcorn CLI**, instead of manual copying.
   - A member pointed out that the `-o flag` provides the functionality, however this [feature](https://github.com/gpu-mode/popcorn-cli/pull/8) was not picked up yet.
- ****Inconsistent Leaderboard Timing Explained****: Users are noticing that their leaderboard run is slower than their benchmark run, and the team is [investigating the cause](https://cdn.discordapp.com/attachments/1359640791525490768/1373399965832974387/Screenshot_2025-05-03_142043.png?ex=682ce8e4&is=682b9764&hm=bf0bb206889151e0924af5919ce178a75f2843800fc1bc4097c735b7c4c465cd&).
   - The slowdown seems to vary based on code specifics, and might be related to **L2 cache** behavior or hardware clock differences, particularly when `recheck=False`.
- ****Composable Kernels and offload-arch flags Need Tweaking****: Users encountered errors with the **composable kernel library** due to the submission server adding multiple `--offload-arch` flags, including `--offload-arch=gfx1030` which is unsupported.
   - Adding `extra_cuda_cflags=['--offload-arch=gfx942']` to the `load_inline` function, as shown in the [template-hip.py](https://cdn.discordapp.com/attachments/1359640791525490768/1373515912048541786/image0.jpg?ex=682cac20&is=682b5aa0&hm=23823fb51c5f50dd9f0ecea948f9d2a64a435274fac897979990633ab2b78b3a&), and using `gfx942:xnack-` can resolve the issue.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1373406268537962557)** (11 messages🔥): 

> `CUTLASS DSL 4.0, CuTeDSL blogpost, CuTeDSL examples, Layout Function` 


- **CUTLASS DSL 4.0 supports Linux and Python 3.12**: The **CUTLASS DSL 4.0** release currently supports **Linux** and **Python 3.12** only.
- **Tweet examples are out of date**: The CUTLASS team updated their examples and GTC slides after a tweet, so users should use the [latest examples](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL) instead.
- **CuTeDSL blogpost released**: The CUTLASS team made **CuTeDSL** available to the public, and there is a [blogpost](https://veitner.bearblog.dev/bridging-math-and-code-cute-layout-algebra-in-cutedsl/) explaining **CuTeDSL** alongside key underlying math concepts like the **Layout Function**, **Coalescing**, and **Complementation**.
- **Transpose thread tiling results in error**: A user asked about a kernel where the thread tiling results in an error, and provided a [code snippet](https://gist.github.com/simveit/ab0a28efb4338592f82c0a8f762f0ac7) for reference.


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1373274824847134792)** (2 messages): 

> `CUTLASS 4.0, DSL, cuTile, Python, Triton` 


- **CUTLASS 4.0 Inspires Mojo Team**: Following the release of **CUTLASS 4.0** with a new **DSL**, which aims to achieve **CUTLASS**-level performance using **Python** abstractions, the team is open to learning from any good ideas.
   - The team also pointed to [Democratizing AI Compute, Part 7](https://www.modular.com/blog/democratizing-ai-compute-part-7-what-about-triton-and-python-edsls) about the advantages of a *real programming language* rather than a DSL.
- **Debate Sparked Over DSLs vs. Real Programming Languages**: The discussion around **CUTLASS 4.0**'s **DSL** approach prompted a consideration of the differences between **DSLs** and full-fledged *real programming languages* like **Mojo**.
   - Modular's blog post suggests that while **DSLs** have their place, a *real programming language* offers broader capabilities and flexibility for AI compute.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1373017076444434453)** (235 messages🔥🔥): 

> `Document AI and NLP, Sociology in biomedical AI, AI_WAIFU's views on AGI, OpenWorldLabs, Diffusion Limitations` 


- **FinTech Data Scientist Pursues Document AI**: Akhil Theerthala from India, a Data Scientist in a FinTech company, is working on projects around **Document AI and NLP**, including Document Representation, TSR, and Document Quality Assessment.
   - He is also pursuing personal projects involving reasoning about **Personal Finance**, **Ethical Dilemmas**, and **Resume/Career path analysis**.
- **AI_WAIFU Shortens AGI Timelines**: **AI_WAIFU** stated that their AGI timelines are now shorter, with most of their probability mass for this year or early next year due to coding models accelerating development a lot, after having left EAI to work on something new.
   - However, they also expressed less conviction in fast takeoff, noting that low-hanging fruit for increasing **NN efficiency** is translating more into good performance for small models rather than seriously increased capabilities for larger models, though nanotechnology might require more compute than AGI itself.
- **OpenWorldLabs Focuses on World Models**: The company **OpenWorldLabs** makes **world models for robots and video games**, accessible on their [website](https://openworldlabs.ai).
   - Members struggled to understand, with one stating that after reading the website and GitHub twice, *I still have almost no idea what you're actually doing*.
- **Diffusion Models Grapple with Error Accumulation**: A member stated that to solve **accumulating error** requires renoising context frames, where existing video models aren't fast enough to solve the problem, but also noted that you *can't cache diffusions because of accumulating errors*.
   - Another member said that **Google** may have solved this through training with error and effectively training the model to denoise from frame to frame.
- **Community Reaches Universal Transformer Conclusion**: Members stated that AR is such a broad formulation that **Universal Transformer/RNN** type mental models easily transfer over.
   - Other members agreed, concluding that Universal Transformers are like crabs and are the ultimate life form for AR in general.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1373017907025674403)** (34 messages🔥): 

> `Approximating LM Finetuning, ReLU Activation, Smooth Activation Function, Muon expensive operations, Grand Research Ideas` 


- **Practical Experiments needed for LM Finetuning**: Practical experiments are needed to demonstrate stability when only approximating with `k << n_vocab` tokens in [LM finetuning](https://openreview.net/forum?id=D2PjEPGXghI).
   - The linked paper seems interesting to some members, but experiments are pretty limited.
- **Smooth Activations shine for Smooth Parametric PDEs**: In an applied math seminar, someone justified not using **ReLU activation** because they knew the target parametric PDE is smooth.
   - The goal of the paper is investigating how well an infinitely smooth network can approximate a high dimensional k-continuously differentiable function.
- **PTS Post Training Technique Blog**: A member shared a blog post about [PTS](https://huggingface.co/blog/codelion/pts).
   - Another member clarified it was posted to the wrong channel.
- **Underexplored Empowerment Ideas**: A member shared an [arxiv link](https://arxiv.org/abs/2505.10361) with the idea that *empowerment is so underexplored*.
   - This may be quite relevant, in particular, **Muon** uses a bunch of otherwise expensive **XX^T** operations.
- **Grand Research Ideas unwelcome**: A member explained that *We've had too many folks coming here with their grand research ideas mostly written by LLMs*.
   - Every single one of them has been a waste of time. Here's another [arxiv link](https://arxiv.org/abs/2403.00329).


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1374029274436210783)** (8 messages🔥): 

> `SpinQuant Llama-2-7b Reproduction, lm eval harness, HFLM modification` 


- **SpinQuant Technique Reproduction**: A member is seeking guidance on reproducing zero-shot reasoning tasks using the **SpinQuant** technique on different bit precision quantized **Llama-2-7b** models from [Facebook Research](https://github.com/facebookresearch/SpinQuant).
   - They specifically want to know how to use their custom quantized model with *lm eval harness*, since SpinQuant requires customized quantization code.
- **Leveraging lm eval harness with external library**: It was suggested that the member could use *lm eval harness* by passing an already initialized model, following [this documentation](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage).
   - This approach would require wrapping the model in a class, potentially by inheriting or modifying the existing [HFLM class](https://github.com/EleutherAI/lm-evaluation-harness/blob/53c653008182339e67b964a4cd3316f651611f38/lm_eval/models/huggingface.py#L47) within the *lm eval harness*.
- **Customizing HFLM for Quantized Models**: Guidance was provided on how to modify the **HFLM** class to accommodate the custom quantized model, passing the initialized model to `pretrained` and [modifying the `_model_call` method](https://github.com/EleutherAI/lm-evaluation-harness/blob/53c653008182339e67b964a4cd3316f651611f38/lm_eval/models/huggingface.py#L870) within the *lm eval harness*.
   - A link to the `_model_call` method was provided for further customization.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1373698894306869248)** (3 messages): 

> `PolyPythia Materials, Pretraining Data, Random Seeds, GPT-NeoX hash` 


- **PolyPythia Material Mix-Up**: A member sought clarification on the **PolyPythia materials** available on Hugging Face, specifically about the pretraining data built using each random seed and the numbering of folders.
   - They asked if folder labeled **0** corresponds to the original run with random seed **1234**, given the discrepancy between the **10 runs** presented in the paper and the folder numbering **0-9**.
- **Config Files Confirmation Quest**: A member mentioned that the config files should be the ones found on GitHub and expressed their intent to confirm this by comparing them to [WandB](https://wandb.ai/eleutherai/pythia-extra-seeds) later in the day.
   - They were under the impression that the **GPT-NeoX hash** was also logged there but were not seeing it and planned to investigate further.
- **Random Seed Riddle Resolved**: A member confirmed that the convention used in experiments was that **seed 0** is indeed equivalent to **seed 1234**.
   - Another member stuck to this convention.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1373030652357775410)** (120 messages🔥🔥): 

> `Freeplay.ai Feedback, Absolute Zero Reasoner (AZR), AI Agent Frameworks, OpenAI Codex Experiment, Perplexity Free Tier Costs` 


- **Freeplay.ai Sparks Interest and Mirroring**: Members discussed [freeplay.ai](https://freeplay.ai), with one user reporting positive initial impressions of its product architecture and direction after a call with them, noting that it *mirrors* in-house builds.
   - Another user expressed strong interest in updates and inquired about significant differences between **Freeplay.ai**, **Braintrust**, and **Humanloop**.
- **Absolute Zero Reasoner Teaches Itself**: A member shared a [YouTube Short](https://youtube.com/shorts/avnHiKcOEQA?si=NWoqwZR1IcPxyrG0) breaking down **Absolute Zero Reasoner (AZR)**, a new AI that teaches itself from scratch without human data.
   - The member requested feedback and thoughts on the video.
- **AI Devs Crave AI Dev Survey**: A member suggested a "State of AI" dev survey to track usage trends in areas like **AI agent frameworks**, **SDKs**, **proxies**, and **models**.
   - The discussion highlighted the need to understand software productivity gains in corporate settings after accounting for new code review processes and organizational scaffolding, and members shared a [link to the 2025 State of AI Engineering survey](https://x.com/barrnanas/status/1904593314533564604) and [Allstacks survey](https://www.surveymonkey.com/r/6WQD6QQ).
- **OpenAI Codex Tackles Shopify App Creation**: A member experimented with **OpenAI Codex** to convert an existing application, Answer HQ, into a compatible **Shopify App Store** app.
   - They noted that Codex first sought an *agents.md* file and that a good README helps streamline the process, drawing parallels to Claude Code's *init* command, recommending generating a AGENTS.md file for LLM consumption outlining domain, key commands, how to run tests, and conventions for the project.
- **Perplexity Free Tier Drains Millions**: A member shared a tweet noting [Perplexity spends $33M/year on its free tier](https://x.com/breadcrumbsre/status/1924474011125227687)


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 messages): 

swyxio: codex pod https://x.com/latentspacepod/status/1923532303327953295?s=46
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1373027176563085532)** (142 messages🔥🔥): 

> `Meta's Maverick LLM, Agent as Judge, Economic disaster, Context Switching, Home Rolled context sharing` 


- **Meta's Maverick LLM Arena gate surprises**: A member mentioned that *the 'if it can surprise you to the upside it can surprise you to the downside'* is a great first step in **Meta's Maverick LLM** arena gate, reminiscent of [Brownian ratchets](https://en.wikipedia.org/wiki/Brownian_ratchet) for leveraging randomness for directional motion.
   - The problem then becomes not eliminating variability but building validations/ratchets/ways to cut off the downsides while maintaining the upsides, such as *llm as judge + retry*.
- **Agent as Judge reduces task fatigue**: Members discussed that lower task fatigue is due to less context switching; as long as you stay in mental *idealand* and out of mental *syntaxspace* you are good, noting a burnt out developer who's been working at **0.1x capacity** returns to **1x performance** with **Agent as Judge**.
   - Burnt out devs can now get back to 1x or greater performance, and we're back to "*agent as judge*".
- **Home Rolled agent army at our fingertips**: Members discussed the possibility of having a **robot agent army** at our fingertips, emphasizing shared context like explicit **MCP** (more a home rolled version).
   - One member really liked the **golang** *100 lines to orchestrate agents* project that hit the channel earlier.
- **Kam's presentation and evaluation methods**: A member asked about learnings on eval methods/specifics and how to check output; another member asked why not use commercial frameworks, with one user quipping that *yeah 'not invented here' hits different in a post agent codegen world*.
   - Kam provided a link to download the video directly ([Dropbox link](https://www.dropbox.com/scl/fi/5l3qq5qf81rgdagilimlt/Latent-Space-AI-In-Action-2025-05-16-Engineering-Reliable-Agents.mp4?rlkey=ma8j2onvhp7kp5qlv27ujheyh&st=3i7x5gue&dl=0)) after editing to black out some personal info caught in the recording.
- **Potential deepfake of Kam's voice in progress**: After audio issues with the presentation recording, one member joked about potentially using **zonos RVC** to deepfake Kam's voice, especially since they missed a portion of the talk.
   - Another member provided a [Loom recording](https://www.loom.com/share/beefb650c11e4828924dd762dcaa9f3e?sid=15db8320-8d71-4d22-9ebf-89ab7f66515f) as a backup option, noting potential background noise due to recording in a gym lobby.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1373013881664049282)** (173 messages🔥🔥): 

> `Codex o4-mini model, Gemini 2.5 vs coding models, Aider settings, Aider's agent-like direction, Model preferences for Aider` 


- **Codex o4-mini Launch Stumbles**: The launch of the **Codex o4-mini** model coincided with big tech companies laying off people working on well respected and valued products, which members found unsurprising.
   - One member quipped that Zuck is probably upset that surfing all day makes him look like an idiot while Musk looks like a genius with Grok.
- **Gemini 2.5 Pro: Coder Model Woes?**: Members noted that a pass rate of **45%** for **Gemini 2.5 Pro/Flash** raised concerns about using coder models in practice, but others recommended using it, or **O4-mini** or **Claude 3.7**, if you value your time.
   - A member said that *pro messes diffs up so much* that they are thinking about doing more practical experiments, eg flash is not a coding model so some super cheap coding model.
- **Aider's Agentic Turn Sparks Concerns**: Members are concerned about the direction **Aider** is taking in becoming more agent-like, worrying that it might lose its core identity as a pair programming tool.
   - One member expressed worry that even if agent-like features are added, it will end up being *neither here nor there*.
- **Model Mania: Preferences for Aider Emerge**: Aider users discussed their model preferences, with **Gemini 2.5 Pro** being a popular choice, while some found **OpenAI's Codex** disappointing compared to **Claude Code**.
   - One member noted **Gemini 2.5 Pro** beats all other models they've tried and another used **Gemini 2.5 plus GPT 4.1** for simple stuff.
- **Large Codebase Wrangling: Plandex Enters the Chat**: Members debated **Plandex's** claim of handling large codebases effectively, but one noted their showcase video featured a simple todo list.
   - One member stated Aider's context handling has been good for them even if their codebase isn't huge.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1373030163327221831)** (42 messages🔥): 

> `Model iteration, Workplan document, Base prompt setup, Edit format issues, UI theme available` 


- **Grooming Workplan Documents Improves Iteration**: One member utilizes a **workplan document** that is groomed through the process, set up in their **base prompt** ([aider_conf.html](https://aider.chat/docs/config/aider_conf.html)).
   - When they want to iterate like that, they generally do it in `/ask` mode, and when things look good, they can `/code okay, please do that`.
- **Automated dev logs!**: A member utilizes a **base prompt** ([aider_conf.html](https://aider.chat/docs/config/aider_conf.html)) with instructions on how to set up a working document that builds out the plan before systematically implementing the changes.
   - They use this prompt: *Please use a file for use as a development context log that will be added to the context, named devlog_{datestamp}.md* to track code corrections and insights.
- **Aider Edit Format: Mission Impossible**: A user had edit format issues when the **LLM suggested multiple edits** and it applied the first edits, confusing things using **deepseek-chat**.
   - The error was: *The LLM did not conform to the edit format* due to a `SearchReplaceNoExactMatch`.
- **Aider's Minimal UI Has Fans**: A user wondered if there were **UI themes available for Aider** and if some elements could be customized.
   - A member replied that there's **light mode and dark mode**, another member pointed to [Pygments](https://pygments.org/styles/) which can be set via `--code-theme`.
- **PR Descriptions via Aider**: A user asked about a workflow for writing **Pull Request descriptions**.
   - A member suggested doing `/run git diff origin/main` from your branch and then `/ask write a pull request description describing this diff`.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1373039374622523515)** (155 messages🔥🔥): 

> `Agent for MCP news, Application vs LLM driven, MCP client tool selection, Streaming HTTP crawl4ai MCP server, MCP vs OpenAPI` 


- ****One-Click MCP News Agent Arrives****: A user announced a new agent that scrapes and summarizes all the MCP news in the past 24 hours in one click, found [here](https://www.jenova.ai/app/tr45cl-mcp-news-summarizer).
- ****Resources: App-Driven vs. LLM-Driven Debate Rages****: A member expressed that it's limiting that resources are **application-driven** rather than **LLM-driven** like tools.
   - Others countered that resources are super powerful and should be app driven over model driven, enabling cool stuff like indexing in MeiliSearch or doing real-time RAG.
- ****Client Control Over Tools****: A member questioned whether a **MCP Client** can selectively pick which tools from the server to pass into the tool signature of the model.
   - Another member clarified that **Claude Desktop** lets you switch tools on-and-off in *Settings*, and the tool stripping is implemented by the client.
- ****Stainless SDK Slammed****: Members complained that the **pydantic models** they generate don't actually validate the data properly, leading to ecosystem fragmentation.
   - They said that the OpenAPI document does not actually match their API behaviour.
- ****Shell Commands vs. JSON Tool Calls****: A member suggested coding agents need sandboxes where they can roam freely within and thinks everything should run in the browser using already existing **OAuth2**.
   - Another countered that if a model can't reliably generate shell commands from comprehensive documentation, it probably can't reliably understand the intent needed for **JSON tool calls** either - both require the same level of comprehension.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1373039474946211870)** (25 messages🔥): 

> `MCP UI SDK release, MCPControl server update, MCP SuperAssistant browser extension, Google Chat + LLM Agents via MCP, Sherlog Canvas (Alpha) release` 


- ****MCP UI SDK** Opens New **Web Interaction** Avenues**: A new [SDK](https://x.com/idosal1/status/1923477857881190718) has been released to add UI to MCP, offering a platform for experimenting with rich web interactions directly on top of the protocol.
   - It enables any MCP server to respond with an **Embedded Resource** with a *ui://* or *ui-app://* URI and handles follow-up interactions through events.
- ****MCPControl Server v0.2.0** Adds **SSE Support****: Version 0.2.0 of the MCPControl server has been released, introducing [SSE support](https://github.com/Cheffromspace/MCPControl/releases/tag/v0.2.0) which makes running it in a VM and giving Claude its own Windows computer possible.
   - This update enables more flexible deployment options for MCPControl, enhancing its integration with AI agents like **Claude**.
- ****MCP SuperAssistant** Integrates MCP with **Multiple AI Platforms****: The MCP SuperAssistant browser extension brings MCP capabilities to platforms like **Grok, ChatGPT, Perplexity, and Gemini AI Studio** without API configuration.
   - Users can now experience MCP directly in their browsers natively via the [extension](https://mcpsuperassistant.ai) , with demos available on [YouTube](https://youtube.com/playlist?list=PLOK1DBnkeaJFzxC4M-z7TU7_j04SShX_w&si=3_piTimdBJN7Ia4M).
- ****Google Chat** Gains **LLM Agent** Connectivity via MCP**: A new project has been released that connects Google Chat to LLM agents (like Cursor) using the Model Control Protocol (MCP), allowing agents to send messages, search convos, and summarize threads directly in Chat spaces, available [on GitHub](https://github.com/siva010928/google-chat-mcp-server).
   - The tool exposes functionalities such as sending messages, searching convos, and summarizing threads as MCP tools, secured via **OAuth 2.0**.
- ****Sherlog Canvas (Alpha)**: AI-Powered **Debugging Interface** Unveiled**: Sherlog Canvas (Alpha), an AI-powered interface akin to a Jupyter Notebook for incident debugging, has been open-sourced, incorporating [MCP-powered cells for logs, metrics, SQL, and more](https://github.com/GetSherlog/Canvas).
   - It offers multi-agent workflows and allows AI to generate, run, and refine cells, aiding in incident and bug investigations, further showcased in a [demo video](https://youtu.be/80c5J3zAZ5c).


  

---


### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1374118249809186846)** (1 messages): 

> `NotebookLM, Mobile App, I/O, MVP` 


- **NotebookLM Mobile App Goes Live**: The **NotebookLM mobile app** has been officially released with an **MVP feature set**, inviting users to provide feedback and feature requests, according to the [Google blog](https://blog.google/technology/ai/notebooklm-app/).
- **Users Encouraged to Submit Feedback on New App**: Users are being encouraged to submit feedback, feature requests and more on the new NotebookLM mobile app.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1373333350915506288)** (28 messages🔥): 

> `NotebookLM for Olympiad Prep, NotebookLM Fails to Upload Materials, Custom Instructions in NotebookLM, NotebookLM as Language Editor, Senpai meaning` 


- **NotebookLM's Olympiad Use Cases Remain Undiscovered**: A user inquired about the actual use case of **NotebookLM** for a class 11 student preparing for **Olympiads**, and how it differs from **Gemini** or **ChatGPT**.
   - However, there were no actual suggestions made for its use in the context of **Olympiads**.
- **NotebookLM Rejects User's Research Materials**: A user expressed disappointment that **NotebookLM** failed to upload research materials related to social justice, ecology, and feminism, with an example from [The Conversation](https://theconversation.com/meet-the-forgotten-enslaved-and-working-class-labourers-behind-british-exploration-in-africa-asia-and-antarctica-252771).
   - The user suspected *censorship and anti-woke stuff* was the reason, and included screenshots showing materials were being rejected.
- **NotebookLM Customized with "Guide.md" Files**: A user boosted productivity with **NotebookLM** by sharing a `guide.md` file with every new notebook, containing the instructions the notebook should follow.
   - The `guide.md` file assigns personas like *senpai* (a professional senior student) or *brain* (a genius AI) to **NotebookLM** for different types of requests, allowing for customized interactions, as shown in [this Youtube video](https://youtu.be/ZCPcBgJ54NY?si=9gHljywup_mO0cAM).
- **Language Editor Role Fits NotebookLM like a Glove**: A user found success using **NotebookLM** with open source textbooks on writing, media, communications, and grammar, highlighting its natural fit as a language editor or teacher due to its AI design.
   - The user created custom instructions as MD files, originally devised for Gemini, and integrated AI persona protocols such as **ScribeMaster AI**, **NSET Agent**, and **LSTA Agent**.
- **NotebookLM Works as Semantic Glossary**: A user suggested that **NotebookLM** is focused on the retrieval of information, reducing the chances of hallucination.
   - They added that it is best used as a semantic Glossary, Index, or Table of Contents for your source materials.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1373019022546964521)** (119 messages🔥🔥): 

> `Audio generation issues, Mind-map conversion to Markdown, Formatting recognition in NotebookLM, Debate podcast creation, NotebookLM API availability` 


- **NotebookLM's Audio Generation Limited for Some Users**: Users report that NotebookLM's audio generation only processes the introduction of longer documents, creating short audio files even with a **114-page PDF**.
   - It remains unclear if this limitation is due to the free version or other factors.
- **Mind-Map Conversion Proves Tricky**: Users seek a method to convert NotebookLM-generated mind maps into Markdown format, but prompting attempts have not replicated the native content accurately.
   - Suggestions include using **Mermaid** or **PlantUML**.
- **NotebookLM's Writing Style Reduced**: A user noted that *NotebookLM’s writing style has become strangely simplistic and reductive in the last few days*, resembling a **high-school-essay quality**.
   - Others discussed potential uses for students preparing for olympiads, emphasizing **confidentiality** as a key differentiator from Gemini or ChatGPT.
- **NotebookLM Mobile App Launches but Lacks Features**: The NotebookLM app has officially launched on [iOS](https://apps.apple.com/us/app/google-notebooklm/id6737527615) and [Android](https://link-to-android-store), but initial feedback indicates it lacks core features like **mind maps**, **briefing notes**, and the ability to 'Discover Sources'.
   - Users also report issues such as not being able to select sources for **audio overviews** and general feature disparity compared to the web version, with some suggesting the app feels like a *web app packaged into a local container*.
- **Video Uploads Added to NotebookLM**: Users discovered that NotebookLM now supports **video uploads** with automatic transcription.
   - This feature was confirmed by a user who shared a [link about **Video overviews in NotebookLM**](https://x.com/testingcatalog/status/1924246774346104863).


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1373060662611411056)** (101 messages🔥🔥): 

> `AWQ INT4 calculations, Gemini generating comments, Open Source AI vs Big Tech Oligopoly, Google's AI Code Agent Jules, Hermes models` 


- **Gemini's Code-Commenting Quirks**: Members discussed the strange behavior of **Gemini** generating commented-out code and removing other comments unexpectedly.
   - One member humorously noted the irony of **Open A.I's** path resembling a Greek saga ripe for a **Netflix** adaptation, hinting at unexpected plot twists and turns.
- **Google's Jules Enters the AI Code-Agent Arena**: **Google** is set to announce its code agent, **Jules**, at **Google I/O**, sparking excitement and anticipation within the community.
   - One user mentioned the multimodal capabilities and daily summary features in **Jules** in addition to sharing a [link](https://jules.google/).
- **AWQ INT4 Calculation Deep Dive**: A user asked whether **AWQ** uses **INT4** natively or does calculations in **BF16**, specifically for **VLLM**, to which another member clarified that **AWQ** is **W4A16** format and GPUs lack circuitry for mixed-precision **INT4xBF16** matmul.
   - They suggested that **QuaRot** is what they are looking for and mentioned **FP8** should also be fine.
- **Decentralized Open Source AI Fights Oligopoly**: A member passionately argued that **decentralized open source AI** is the only viable way to prevent a big tech oligopoly on AI.
   - Counterpoints suggested governmental regulations, such as open-sourcing all AI models with data and laws against censorship, as alternative solutions, sparking debate on the feasibility and likelihood of each approach.
- **Hermes models discussed**: A user asked what the current best **Hermes models** are to use.
   - Other users suggested **Hermes 405b**, **Deephermes 24B** and **8B**.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1373166804347650099)** (9 messages🔥): 

> `LLM Editing for Autonomous Bots, Fine-tuning vs RL vs Control Vectors, Nous Hermes Model` 


- **LLM Novice Seeks Editing Guidance**: A member expressed a desire to edit an LLM for an autonomous bot, admitting ignorance about the process and seeking assistance.
   - A senior member suggested that **fine-tuning**, **RL**, or using **control vectors** might be appropriate, while also asking if the member had experience making custom LLMs before.
- **Nous Hermes Model Launched**: A senior member, after being asked if they'd made custom LLMs before, linked to the [Nous Hermes model](https://nousresearch.com/hermes3/), implying they were the author.
   - Another member noted this was a *mic drop moment*.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1373195380933525645)** (3 messages): 

> `LLMs Conventions, LLMs Collective Biases, LLMs Adversarial Agents` 


- **LLMs Spontaneously Converge on Conventions**: According to a study ([https://arxiv.org/abs/2505.10475](https://arxiv.org/abs/2505.10475) and [https://www.science.org/doi/epdf/10.1126/sciadv.adu9368](https://www.science.org/doi/epdf/10.1126/sciadv.adu9368)), universally adopted social **conventions spontaneously emerge** in decentralized LLM populations through local interactions.
- **LLMs Develop Collective Biases**: Research indicates that strong **collective biases** can emerge during decentralized LLM interactions, even when individual agents show no initial bias (as per [https://arxiv.org/abs/2505.10475](https://arxiv.org/abs/2505.10475) and [https://www.science.org/doi/epdf/10.1126/sciadv.adu9368](https://www.science.org/doi/epdf/10.1126/sciadv.adu9368)).
- **Adversarial LLM Agents Drive Social Change**: A study ([https://arxiv.org/abs/2505.10475](https://arxiv.org/abs/2505.10475) and [https://www.science.org/doi/epdf/10.1126/sciadv.adu9368](https://www.science.org/doi/epdf/10.1126/sciadv.adu9368)) shows that committed minority groups of **adversarial LLM agents** can drive social change by imposing alternative conventions once they reach a critical threshold.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1373525722387189831)** (2 messages): 

> `Gemini, MoE, Long Context Window, Sub-global attention blocks` 


- **Hypothesizing Gemini's MoE Architecture**: A member hypothesized that **Gemini** uses a Long Context **MoE Architecture**, referring to an *Ensemble of Expert (EoE)* or *Mesh of Expert (MeoE)*, with a common/shared long (**1-10M**) context window.
   - The member's [X post](https://x.com/ditpoo/status/1923966380854157434) suggests this architecture uses shared context as independent shards of (mini) contexts, like *Sub-global attention blocks*.
- **Testing Sub-Global Attention Blocks**: A member wants to test if **sub-global attention blocks** or *sub-context experts* can operate somewhat independently and then scale up into a larger global attention paradigm for handling extremely long contexts.
   - They said that it *requires some engineering to make it possible*.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1373195380933525645)** (3 messages): 

> `Decentralized LLM Populations, Emergence of Social Conventions, Collective Biases in LLMs, Adversarial LLM Agents` 


- **Social Conventions Spark in Decentralized LLMs**: A new study ([arxiv link](https://arxiv.org/abs/2505.10475), [science link](https://www.science.org/doi/epdf/10.1126/sciadv.adu9368), and [HF blog link](https://huggingface.co/blog/codelion/ptsreal.azure)) shows that **universally adopted social conventions** can spontaneously emerge in decentralized LLM populations through local interactions.
   - The researchers highlight that even without individual biases, strong collective biases can form during these interactions.
- **Minority LLM Agents Drive Social Change**: The study also reveals that committed minority groups of **adversarial LLM agents** can drive social change.
   - These agents impose alternative conventions once they reach a critical threshold, potentially reshaping the overall behavior of the LLM population.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1373094388153978900)** (38 messages🔥): 

> `CNN diagrams, matplotlib for diagrams, DiagramVIS-for-computervis, Gemini 2.5 Pro, geometric deep learning` 


- ****CNN Diagram Dilemma** Resolved with Matplotlib**: A member sought a tool for creating **CNN diagrams with skip connections** and ended up using *matplotlib*, which they found painful but managed with GitHub Copilot.
   - The resulting script is now available in a [GitHub repo](https://github.com/dotrdp/DiagramVIS-for-computervis) under *dotrdp/DiagramVIS-for-computervis*.
- ****Gemini 2.5 Pro** Used for Physics**: One member mentioned that they often use **Gemini 2.5 Pro with Canvas or Cursor** for physics-related tasks.
   - They also expressed interest in trying tools like *Windsurf, Aider, Cline, Roo*, and *Codex*.
- ****Geometric Deep Learning** Gaining Traction**: A member shared a [link](https://fxtwitter.com/meowdib/status/1922315466401308965) asking if anyone is a fan of **geometric deep learning**.
   - Another member said *It's fine but I prefer augmenting data if symmetries are needed*.
- **Seeking MLOps Courses**: A member asked if there are any recommended courses for learning **MLOps**.
   - No specific courses were recommended in the chat.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1373083473245503520)** (29 messages🔥): 

> `AlphaEvolve Whitepaper, Physics of Language Models Discussion, Ecology of LLMs and Social Conventions, Loss Clamping` 


- **AlphaEvolve Paper Gets Thumbs Up**: Members expressed interest in reviewing the **AlphaEvolve** whitepaper ([AlphaEvolve.pdf](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf)), a **Gemini**-powered coding agent for designing advanced algorithms.
   - However, the discussion was cancelled because *it could not improve on the discussion that's just been taking place in the Open Machine Learning Channel*.
- **Physics of Language Models, The Sequel**: The group discussed **"Physics of Language Models: Part 3.1, Knowledge Storage and Extraction"** ([paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5250633), [blog](https://physics.allen-zhu.com/part-3-knowledge/part-3-1), [YouTube](https://www.youtube.com/watch?v=YSHzKmEianc)).
- **Ecology of LLMs shows social conventions can arise**: The group discussed **"Emergent social conventions and collective bias in LLM populations"** ([science.org](https://www.science.org/doi/10.1126/sciadv.adu9368)), exploring how AI agents can autonomously develop social conventions.
   - The paper's abstract highlights *the spontaneous emergence of universally adopted social conventions in decentralized populations of large language model (LLM) agents*.
- **Loss Clamping Explored in Detail**: A user asked about clamping the loss to threshold in a range, `loss = loss.clamp(l_avg-threshold,l_avg+threshold)`
   - Another user said *That wouldn't change anything, you could as well just reduce the learning rate*, and then subsequently said that they were *incorrect, it's way worse*.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1373025166547882024)** (34 messages🔥): 

> `Leadership issues, Open Source AI, Sam Altman Strategies, Attention seeking transformers` 


- **Exodus Implies Leadership Issues**: A large exodus of employees suggests issues with company mission, management, or a toxic workplace, with **Sam Altman** and **Mark Zuckerberg** cited as examples.
   - One member suggested that determining the root cause would require asking those who left, although one member stated that they *can fanthom why anyone would want to work at Meta at all*.
- **Open Source AI Strategy**: Members debated whether open-sourcing AI research is a genuine strategy or a way to get free workforce and resources, particularly in the case of **Meta**.
   - One member argued that Meta's AI research serves to *commoditize your complements* rather than being central to Facebook-the-product.
- **Altman's Harvesting Tactics**: A member claimed that **Sam Altman** is harvesting various resources, referencing **Codex-CLI** and free tier **ChatGPT**.
   - They linked a [YouTube video](https://www.youtube.com/watch?v=Y8Tj9kq4iWY) without further comment.
- **Transformers Crave Attention**: A member sarcastically commented on the attention-seeking nature of transformers, linking a [YouTube video](https://www.youtube.com/watch?v=T8Ty99O4m0w) without further comment.
   - Two more YouTube links were posted: [link 1](https://www.youtube.com/watch?v=lrM5KlNtC3c) and [link 2](https://www.youtube.com/watch?v=RH4hAgvYSzg).


  

---


### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1374060602640695337)** (1 messages): 

> `LlamaIndex office hours, LlamaIndex Agents with Long-Term and Short-Term Memory, Multi-Agent Workflow with Weaviate QueryAgent, LlamaExtract for Structured Data` 


- ****LlamaIndex** AMA this Thursday**: The first **LlamaIndex** office hours will be this Thursday at **8AM PT/5PM CET** in the general voice channel for **1 hour**, to ask anything about **LlamaIndex**.
   - The presenters will also walk through how to build agent workflows.
- ****LlamaIndex Agents** Now Have **Long-Term and Short-Term Memory****: An update on **LlamaIndex Agents** and their improved **long-term and short-term memory** was made available on the [LlamaIndex blog](https://www.llamaindex.ai/blog/improved-long-and-short-term-memory-for-llamaindex-agents).
- ****Weaviate** Powers Multi-Agent Workflow**: A guide to multi-agent workflows using **Weaviate** `QueryAgent` is now available in the [LlamaIndex documentation](https://docs.llamaindex.ai/en/stable/examples/agent/multi_agent_workflow_with_weaviate_queryagent).
- ****LlamaExtract** Extracts Structured Data with Citations**: Learn how to extract structured data with citations and reasoning using **LlamaExtract** in [this YouTube video](https://youtu.be/01kM7tXRHi4).


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1373070278703321249)** (4 messages): 

> `LlamaParse updates, Azure AI Foundry Agent Service, LlamaIndex Discord office hours` 


- **LlamaParse gets Easier, More Efficient**: LlamaIndex announced exciting updates to **LlamaParse**, featuring a *streamlined interface*, a new **Code Snippet** button, and more use-case presets coming soon, according to [this tweet](https://twitter.com/llama_index/status/1923510823164706925).
- **Azure AI Foundry Agent Service Adds LlamaIndex support**: The now generally available **Azure AI Foundry Agent Service** comes with first-class **LlamaIndex** support, enabling enterprise customers to build customer support assistants and process automation bots, as highlighted in [this tweet](https://twitter.com/llama_index/status/1924502129974411504).
- **LlamaIndex Hosts First Discord Office Hours**: The **LlamaIndex** team is hosting its first Discord office hours session this Thursday, including an events driven agent workflows run-through and live coding session; more info in [this tweet](https://twitter.com/llama_index/status/1924527932258845178).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1373055558726189107)** (58 messages🔥🔥): 

> `COBOL code splitting, Claude desktop file drops, AgentWorkflow streaming with Anthropic, LlamaIndex and Ollama integration, Agent state persistence with DB` 


- ****Chonkie** eats **COBOL** blocks**: A user inquired about Python packages for splitting **COBOL** code into logical blocks, and was directed to [Chonkie.ai's `CodeChunker`](https://chonkie.ai), which supposedly supports **COBOL**.
   - The user also noted that LlamaIndex code splitter doesn't currently support **COBOL**.
- **AgentWorkflow struggles to stream Anthropic**: A user reported that `AgentWorkflow.from_tools_or_functions` doesn't support streaming for Anthropic's thinking mode, only providing a final response, and provided a [code snippet](https://cdn.discordapp.com/attachments/1373578080853168128/1373659791200620554/test.py?ex=682c895f&is=682b37df&hm=b4fe5af0fef1a812017006a9fbbb11c6d04ec2e1a4f6da81f06a2c7dc05ad3c3&).
   - A community member suggested running `AgentWorkflow` step-by-step using `agent._run_step()` to capture `AgentStream` events.
- **Context is Queen for Agent State Persistence**: A user asked about managing context window limits when saving agent state to a database, especially with increasing messages.
   - A community member suggested limiting the number of instances fetched from the database, saving the context to the database, and passing entire context workflow objects instead of classic user/assistant message history.
- **LlamaIndex Ollama sees more clear**: A user encountered a `ValueError: \"ChatResponse\" object has no field \"usage\"` when using LlamaIndex with Ollama, despite following the [official documentation](https://docs.llamaindex.ai/en/stable/examples/llm/ollama/).
   - It was resolved by upgrading Python to **3.10**, creating a new environment, and upgrading llama-index and ollama packages.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1373056187863535717)** (45 messages🔥): 

> `NDBuffer deprecation and alternatives, ArcPointer and Atomic structs, Importing mojo code in notebooks, LSP issues and workarounds, Documentation issues in GPU basics tutorial` 


- ****NDBuffer's Sunset**: Transitioning to LayoutTensor**: **NDBuffer** is being deprecated in favor of **LayoutTensor**, with active work underway for kernel transitions.
   - Users were advised to hold off using **NDBuffer** due to its impending deprecation.
- ****ArcPointer** Faces **Atomic** Obstacles**: A user inquired about creating an **ArcPointer** to a struct containing an **Atomic**, encountering issues with the **Movable** trait.
   - A suggestion was made to use `ArcPointer[OwnedPointer[T]]`, however **OwnedPointer** does not implement **Movable** either and this workaround does not work as expected.
- ****Mojo Notebook Imports**: Seeking Guidance**: A user sought guidance on how to import code from a **Mojo package** or file within the same folder as a **notebook**.
   - No solution was provided in the messages.
- ****LSP Server**: Eating CPUs and Crashing**: Users reported various issues with the **LSP server**, including high CPU usage (8-16 threads) and frequent crashes, particularly on older systems or when using Docker.
   - Workarounds included restarting the **LSP server** or downgrading to a previous nightly build, but these were not universally effective, one user settled with `killall mojo-lsp-server`.
- ****GPU Basics Tutorial**: Docs Need Love**: A user identified two minor documentation issues in the [GPU basics tutorial](https://docs.modular.com/mojo/manual/gpu/basics/): the repo layout has changed, and the creation of **DeviceContext** should be wrapped in a try/except block.
   - The user provided code snippets to demonstrate the issues and suggested corrections.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1374056104044134491)** (8 messages🔥): 

> `Mojo kernel registration in PyTorch, Max vs. Fireworks.ai, Together.ai, and Groq.com for serving LLMs, register_custom_ops removal` 


- **Mojo Kernel Registration Gets Bumpy 🛤️**: A user reported issues with registering Mojo kernels in PyTorch after updating to the latest nightly, noting that the function `register_custom_ops` in `max.torch` was removed, and a link to an [outdated example](https://github.com/modular/modular/blob/main/examples/custom_ops/whisper.py) was provided.
   - A Modular team member confirmed ongoing work on registering Mojo custom ops with PyTorch in nightlies, cautioning about *friction and changes* in the coming days, but directing them to the updated [documentation](https://docs.modular.com/max/api/python/torch/).
- **Max Faces AI Inference Rivals ⚔️**: A user is trying to convince their boss to use Max for serving LLMs, but questions its performance against platforms like **Fireworks.ai**, **Together.ai**, and **Groq.com**, which claim significant speed and latency improvements.
   - The user referenced a comparison of Mojo GPU with **Cublas** in NVIDIA and **Rocblas/HipBlaslt** in AMD ([YouTube link](https://www.youtube.com/live/yOMflrCRya0?si=w9QCDUFvOFG4y7EQ&t=5842)), seeking information on Max's current performance and future optimization plans relative to these AI-inference providers.
- **`register_custom_ops` R.I.P. 💀**: The function `register_custom_ops` of `max.torch` was removed in the latest nightly.
   - A Modular team member confirmed ongoing work on registering Mojo custom ops with PyTorch in nightlies.


  

---


### **tinygrad (George Hotz) ▷ #[announcements](https://discord.com/channels/1068976834382925865/1069236008115253348/)** (1 messages): 

georgehotz: see tinygrad performance get better here https://stats.tinygrad.win
  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1373013491920797777)** (50 messages🔥): 

> `tinygrad use GCC instead of Clang, porting a model or weights to TinyGrad, tinygrad 1.0 plan, quantize onnx bug, get torch.index_put to work` 


- ****GCC** Instead of **Clang** for tinygrad?**: A user inquired about using **GCC** instead of **Clang** for a CPU target, specifically for an **AIX system with the PPC64 arch** where **Clang** is unavailable.
   - George Hotz responded that it's *not easily* done and would require adding **elf relocations for ppc64** to the custom elf loader, referencing [this file](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/runtime/support/elf.py#L41).
- ****ONNX** Eases Model Porting to tinygrad**: A user asked about porting models like **Qwen3:30-a3b** to TinyGrad, inquiring about automatic tools versus manual porting.
   - George Hotz clarified that if the model is in **ONNX** format, it's easy to import using the `examples/benchmark_onnx.py` script.
- **tinygrad's API Stability Praised**: An author writing a book on AI application development considered using tinygrad, asking about the stability of its interfaces for examples to remain functional over 2-3 years.
   - George Hotz assured that the frontend has been *very stable for at least a year*, suggesting they *should do 1.0 before we do speed*.
- ****WMMA** Instruction Benchmarking Tool Arrives**: A user shared a link to [HIPmmapeak](https://github.com/pkourouklidis/HIPmmapeak), a tool to measure max **FLOPS** of the **wmma instructions** on an **7900 XTX**, similar to mmapeak.
   - George Hotz responded *oh cool! use the tinygrad infra if you want the bounty*.
- ****ROCm 6.4** Incompatibility Issues Arise**: A user reported wasting half a day due to **ROCm 6.4 comgr incompatibility** while working on [this PR](https://github.com/tinygrad/tinygrad/pull/10417).
   - George Hotz responded that this issue is annoying, noting that *they changed enums like they were gonna run out of numbers of something!*


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1374066163872563231)** (1 messages): 

> `AI Agent Engineering, LLMs & Foundation Models, Full-Stack & Backend Systems, Automation & Agent Ops, Vector DBs & Memory Storage` 


- **Engineer Introduces AI/ML background**: An AI/ML engineer with over **8 years** of experience introduced themselves, highlighting their experience in building intelligent, production-grade systems across industries like healthcare, smart cities, e-commerce, and entertainment and a [portfolio](https://yangming.vercel.app/).
- **Expertise in Building Agentic Systems**: The engineer specializes in building agentic systems using modern stacks such as **LangGraph**, **AutoGen**, **LlamaIndex**, **Letta**, and **DSPy**.
   - They also have experience with **AI observability tools** like **LangSmith**, **Langfuse**, and **AgentOps**, and memory-augmented agents using **MemGPT**, **LangMem**, and **zep**.
- **Proficiency with LLMs and Foundation Models**: The engineer is proficient in fine-tuning, retrieval-augmented generation (**RAG**), prompt engineering, and hybrid chaining with top models like **GPT-4o**, **Claude 3**, **Gemini**, **Mixtral**, **LLaMA-3**, and **Mistral**.
- **Experience in Full-Stack and Backend Systems**: With expertise in **React**, **Next.js**, **FastAPI**, **Django**, and **Laravel**, they can build scalable architectures for serving LLMs via **vLLM**, **Ollama**, **Fireworks AI**, and **OpenAI APIs**.
- **Knowledge of Automation and Agent Ops**: The engineer is skilled in workflow orchestration via **n8n**, **Make.com**, **Zapier**, and **GoHighLevel**, as well as deployments using cloud-native solutions and sandboxing with **E2B** and **Modal**.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1373031794466885722)** (39 messages🔥): 

> `Assert/Suggest replacement, VS Code theme settings, AI coding agent with DSPy, DSPy latency with large system prompts, DSPy 3.0 port to Elixir` 


- ****Suggest and Assert Evolve into BestOfN and Refine****: `BestOfN` and `Refine` are the replacements for `dspy.Suggest` and `dspy.Assert` as of **DSPy 2.6**, as detailed in [this tutorial](https://dspy.ai/tutorials/output_refinement/best-of-n-and-refine/).
- ****Cline coding agent may be smaller and more accurate with DSPy****: Members discussed reimplementing an AI coding agent like **Cline** using DSPy; it was suggested that it might be smaller and more accurate, with less *off-piste* changes, with VS Code glue, memory, tools and models all being important factors.
- ****DSPy Latency Woes with Large Prompts? Caching Configuration Conundrums!****: A member noted that large system prompts cause long resolution times in DSPy and inquired about configuring **litellm**'s [prompt caching](https://docs.litellm.ai/docs/completion/prompt_caching), which resulted in an error because it fellback to JSON.
   - Another member optimized the prompt with DSPy and then used the optimized prompt directly with the API, but found that using DSPy responses took **8s vs 2s** without, and wondered if it was a local issue.
- ****Elixir Gains Traction with DSPy 3.0 Parity Port Proponents****: A member inquired about interest in a **1:1 parity port of DSPy 3.0 to Elixir**, sparking discussion around the language's scaling patterns.
   - Proponents argue **Elixir**'s concurrency model is *perfect for LLM orchestration* and that the [Nx ecosystem](https://github.com/elixir-nx) has made massive strides recently, while others argue Python is still the primary driver in the ecosystem due to integrations and maturity.
- ****Critics Question DSPy's Complexity and Benefits****: Concerns were raised by some online users about **DSPy's complexity** and lack of interesting results, suggesting that those familiar with prompting tricks may outperform the library.
   - A member countered that DSPy's true value lies in solving scaling issues for agentic apps, especially those in production with frequently changing requirements, citing that companies building with DSPy might avoid costly public failures in the future.


  

---


### **Cohere ▷ #[💬-general](https://discord.com/channels/954421988141711382/954421988783444043/1373093360012628061)** (7 messages): 

> `Fine-tuning embedding models, Unwanted embeddings from embed-v4.0` 


- **Fine-Tuning Embedding Models Faces Roadblock**: A member inquired whether it is possible to fine-tune an embedding model, and another member confirmed that **fine-tuning embedding models is not possible**.
   - However, they noted that it **is possible to fine-tune a reranker** based on specific requirements.
- **Embed-v4.0 Generates Unwanted Embeddings**: A member reported that **embed-v4.0** generated some *unwanted* embeddings.
   - No solution was given but is recorded as a known issue.


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1373110586023346287)** (5 messages): 

> `Embed 4 Pricing, Vision RAG, Chat API Stalling, Agent API Calls` 


- **Embed 4 Pricing Probed for Vision RAG**: A member inquired about pricing for **Embed 4** in the context of their team's **Vision RAG** implementation.
   - Another member suggested embedding an image and checking the **token count** in the usage section of the response.
- **Chat API Stalling Plagues Agent Implementation**: A member reported that the **Chat API** seems to be stuck midway during the execution of their agent, despite being a paid member.
   - They mentioned experimenting with running the **nodes sequentially vs. in parallel** as a transient workaround.
- **Debugging Help Requested for Chat API Freezes**: A member asked for help regarding the **Chat API** getting stuck during multiple API calls in an agent setup.
   - Another member requested details such as the **model, API version, task, and code snippet** to assist in debugging.


  

---


### **Cohere ▷ #[💡-projects](https://discord.com/channels/954421988141711382/1218409701339828245/1374148102197481513)** (1 messages): 

> `Vitalops datatune, Open source data transformation` 


- **Vitalops releases datatune on GitHub**: Vitalops released a new **open source tool** called **datatune** on [GitHub](https://github.com/vitalops/datatune) that performs data transformations using natural language instructions and LLMs.
- **Datatune leverages LLMs for data transformations**: The **datatune** tool utilizes LLMs to execute **data transformations** based on simple **natural language instructions**, aiming to simplify complex data manipulations.


  

---


### **Cohere ▷ #[🤝-introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1373093668960866354)** (3 messages): 

> `Game Development, AI/ML Engineering, 3D Game Development, AI-powered NPCs, Skills in Game Engines` 


- **Game Developer Seeks New Role**: A **Game Developer** and **AI/ML Engineer** with 8 years of experience is seeking a new job after their contract ended.
   - They specialize in **3D game development** with **AI-powered, human-like NPCs** and have a strong skill set including Unity, Unreal Engine, Godot, and networking tools like Photon and Mirror.
- **Skills Boast**: An engineer shows **skills in a multitude of engines**, including Unity (**C#**), UE (**C++/Blueprints**), and Godot (**GDScript**).
   - They can do: **combat**, **character control**, **inventory**, **interaction systems**, **Humanized NPCs**, **NLP**, **emotional state systems**, **UI/UX**, **HUD**, **inventory UI**, **dialog systems**, **responsive user interfaces**, **Photon**, **Mirror**.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1373029833235497010)** (13 messages🔥): 

> `Swarm UI, Linus Tech Tips $1M Server for Pi Calculation, Customer Success career advice, Models for text interpretation and formatting` 


- ****Swarm UI** Praised Over Local Docs**: A member stated that *local docs are fine ... for now* but they prefer using **swarm-ui** because *its the best way to go for easy to mid to advanced* usages.
- **Linus Builds Million Dollar Pi Server**: A member linked Linus Tech Tips building an insane **$1M server** *for calculating the digits of pie* and another responded that *That's not the first time I see Linus doing insane things.*
- **Seeking Advice on Customer Success Roles**: A member asked if anyone was involved in **Customer Success** and seeking to transition into a **CS role** at a tech company, highlighting their background in government, nonprofits, sales, and outreach.
   - They emphasized their natural fit for supporting others, building relationships, and problem-solving, along with their tech expertise and ability to simplify complex ideas.
- **Models for Interpreting Textbooks to Markdown**: A member inquired about recommended models for interpreting and formatting large quantities of text, specifically for converting textbooks into **markdown notes** and summarizing/extracting specific information.
   - They sought advice on tools capable of handling such tasks effectively.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1374126490228232482)** (1 messages): 

> `AgentX Competition, Submission Forms, Judging Panel, Entrepreneurship Track, Research Track` 


- **AgentX Competition Forms Launch**: The submission forms for the **AgentX Competition** are now open, featuring an all-star judging panel of VCs, founders, product leads, and researchers from top AI companies; submission links are now available for the [Entrepreneurship Track](https://forms.gle/FJTC4jd197bNeJJ96) and [Research Track](https://forms.gle/5dccciawydCZ8o4A8).
- **Crucial Submission Deadline Looms**: The deadline for all submissions is **May 31, 2025**, at 11:59 PM PT; teams should prepare a pitch deck, product demo video, and live product link for the Entrepreneurship Track, and a paper, video presentation, and GitHub repo for the Research Track.
- **$150K Prize Pool Teased!**: Over **$150K** in prizes are up for grabs for the top teams, with only ~2 weeks remaining to finalize projects; questions about submissions can be asked in the channel.
- **Social Boost Requested for AgentX**: Participants are encouraged to spread the word about AgentX on social media via RT/QT/Like/Repost using the announcement on [Twitter](https://x.com/dawnsongtweets/status/1924470174776004875) and [LinkedIn](https://www.linkedin.com/posts/dawn-song-51586033_agentx-agentx-activity-7330241621160005633-E8Ii).


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1374107112698941440)** (2 messages): 

> `OpenAI API keys, Alternative approaches to API calls, Trailblazer Tier Certificate` 


- **Students Must Bring Their Own OpenAI Keys**: Students in the LLM Agents MOOC are required to supply their own **OpenAI API keys** for lab exercises, as the course **does not provide them**.
   - The message clarified that although the key is needed for running the labs, it can be excluded from the final submission.
- **Explore Approaches Avoiding API Calls**: A lab TA was tagged to provide insight into alternative approaches that **avoid direct API calls** during lab exercises.
   - This suggests there may be methods or tools available to minimize or bypass the need for external API interactions.
- **Mastery Tier Students Can Fall Back to Trailblazer**: Students who complete quizzes and the written article but struggle with the labs can still apply for the **Mastery Tier** and potentially be "downgraded" to the **Trailblazer Tier** if necessary.
   - This ensures that students who demonstrate knowledge through assessments are still recognized even if they face challenges with the practical lab components.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1374037141625114825)** (1 messages): 

> `Model Evaluation, Finetuning, Continuous Integration` 


- **Discussing Strategies for Model Evaluation**: The message proposes two methods for model evaluation after finetuning: **ad hoc evaluation** before publishing, and **automated evaluation** in nightly continuous integration (CI).
   - The user noted that **automated evaluation** offers higher fidelity but requires more effort and compute resources.
- **Ad Hoc vs Automated Evaluation**: The user is weighing the pros and cons of evaluating models on an **ad hoc basis** versus setting up an **automated nightly CI** pipeline for model evaluation.
   - They acknowledge the **ad hoc** approach is less reliable but easier to implement, while the **automated CI** approach provides higher fidelity but demands more maintenance and computational resources.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1373224480289329233)** (1 messages): 

> `Torchtune cfg.get` 


- **Torchtune config values used in search**: A user searched GitHub's code to find out where the the validation dataset is set using the config's `cfg.get("dataset_val")` in the [pytorch/torchtune repository](https://github.com/search?q=repo%3Apytorch%2Ftorchtune%20cfg.get(%22dataset_val%22)&type=code).
   - This may be useful in understanding how the **validation dataset** is being used and how it can be customized.
- **Torchtune Validation**: Understanding how the validation dataset works is useful for evaluating model training.
   - Exploring `cfg.get` helps discover where dataset configurations are set.


  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1374149004580884631)** (1 messages): 

> `DataTune, Data transformation, Vitalops` 


- **Vitalops Releases DataTune**: Vitalops released **DataTune**, a new open source tool for data transformations using natural language instructions and LLMs, available on [GitHub](https://github.com/vitalops/datatune).
- **Tune Your Data**: **DataTune** uses LLMs to transform data with simple instructions.


  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1374039122989158541)** (1 messages): 

> `` 


- **AI Engineer's Profile**: An AI Engineer with **10 years of experience** in machine learning, deep learning, and data science is skilled in building, training, and deploying AI models for real-world applications.
   - They have deep knowledge of **Python, TensorFlow, PyTorch**, and cloud platforms such as **AWS, GCP** and are looking to connect and build the next generation of thinking software.
- **AI Engineer's Skill Set**: The AI Engineer boasts skills in **Python, SQL, R**, ML/DL Frameworks (**TensorFlow, PyTorch**), and tools like **Jupyter, Git, Docker, MLflow, Streamlit**.
   - Their techniques include Supervised & Unsupervised Learning, Deep Learning (CNN, RNN, Transformers), NLP, Computer Vision, and Model Deployment (APIs, CI/CD).

