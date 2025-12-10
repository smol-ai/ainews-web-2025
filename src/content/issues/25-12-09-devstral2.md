---
id: MjAyNS0x
title: MCP -> Agentic AI Foudnation, Mistral Devstral 2
date: '2025-12-09T05:44:39.731046Z'
description: >-
  **OpenAI Engineering** sees a significant collaborative milestone with the
  launch of the **Agentic AI Foundation** under the Linux Foundation, uniting
  projects from **Anthropic**, **OpenAI**, and **Block**. **Mistral** released
  **Devstral 2**, a coding model with **123B parameters** and open weights,
  offering a cost-effective alternative to **Sonnet 4.3** and competitive
  performance against **DeepSeek v3.2**. The new **Mistral Vibe CLI** supports
  agentic coding workflows with rapid ecosystem integration. **Alibaba**
  introduced **Soft Adaptive Policy Optimization (SAPO)** for reinforcement
  learning tuning, improving stability and performance in **Qwen3-VL** across
  multiple tasks. Research highlights include the importance of data
  decontamination in RL and ongoing discussions on MoE RL stability and reward
  hacking mitigation.
companies:
  - openai
  - anthropic
  - block
  - mistral-ai
  - alibaba
  - linux-foundation
  - deepseek
models:
  - devstral-2
  - devstral-small-2
  - sonnet-4.3
  - deepseek-v3.2
  - qwen3-vl
topics:
  - agentic-ai
  - coding-models
  - reinforcement-learning
  - model-performance
  - model-optimization
  - open-weights
  - cli-tools
  - multi-file-code-automation
  - data-decontamination
  - moe
  - reward-models
  - rl-stability
people:
  - guillaumelample
  - b_roziere
  - qtnx_
  - charliermarsh
  - omarsar0
  - eliebakouch
  - justinwaugh
  - cwolferesearch
  - pan
---


**A good day for open AI Engineering.**

> AI News for 12/8/2025-12/9/2025. We checked 12 subreddits, 544 Twitters and 24 Discords (205 channels, and 7780 messages) for you. Estimated reading time saved (at 200wpm): 644 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

A rare cross company move with the establishment of the [Agentic AI Foundation](https://aaif.io/) under the Linux Foundation, with Anthropic's MCP joining OpenAI's [Agents.md](http://agents.md/) and Block's [Goose](https://block.xyz/inside/block-anthropic-and-openai-launch-the-agentic-ai-foundation) as founding projects.

[Agentic AI Foundation website with circular images representing different technological and professional scenarios, showcasing the foundation's mission of advancing AI collaboration.](https://resend-attachments.s3.amazonaws.com/VPkfqEsOgQ9DVsx)

Mistral continues to make minor waves with [**Devstral 2**](https://mistral.ai/news/devstral-2-vibe-cli) it's new coding model that is "[Sonnet 4.3 level](https://news.ycombinator.com/item?id=46213498)" but 10x cheaper by API and open weights, winning or tying with [DeepSeek v3.2](https://news.smol.ai/issues/25-12-01-deepseek-32) 71% of the time in third party human evals.

[A graph comparing AI model performance and size, highlighting Devstral 2 as a top performer in the SWE-bench Verifie](https://resend-attachments.s3.amazonaws.com/vZp2EmlFCol0uGA)

The new Mistral Vibe CLI is a pleasure to try out, even if not SOTA.

---

# AI Twitter Recap

**Mistral’s Devstral 2 release and the “agentic coding” toolchain**

- **Devstral 2 + Vibe CLI (open weights)**: Mistral released two coding models and a native CLI for agent workflows: **Devstral 2 (123B dense, modified MIT license)** and **Devstral Small 2 (24B, Apache 2.0)**, both available via API and open weights. The new “Mistral Vibe” CLI bootstraps with uv and provides end-to-end, multi-file code automation designed for agentic coding in terminals/editors. Day-0 ecosystem support arrived rapidly: vLLM inference support, Zed editor integration, and a polished Textual-based TUI. Devstral/Vibe is configurable with MCP and custom tools via config.toml. Links: [@MistralAI](https://twitter.com/MistralAI/status/1998407335308358028), [thread](https://twitter.com/MistralAI/status/1998407332502405347), [@GuillaumeLample](https://twitter.com/GuillaumeLample/status/1998409211068232119), [@b_roziere](https://twitter.com/b_roziere/status/1998408872168391166), [@qtnx_](https://twitter.com/qtnx_/status/1998407441256530163), [@charliermarsh](https://twitter.com/charliermarsh/status/1998447013797458336), [@vllm_project](https://twitter.com/vllm_project/status/1998428798891765926), [@zeddotdev](https://twitter.com/zeddotdev/status/1998456122886238589), [@omarsar0](https://twitter.com/omarsar0/status/1998466422976999896), [Textual UI](https://twitter.com/onetwoval/status/1998439440797020527).
- **Performance and deployment caveats**: Several engineers flagged comparisons that use total parameters as misleading when contrasting dense vs MoE; for throughput/cost, active params and system-level speed on vLLM/sglang are more relevant. Early anecdotal benchmarks suggest MoE backends (e.g., MiniMax M2 A10B-active) can be 2–3.5x faster than a 123B dense model depending on concurrency. Links: [@eliebakouch](https://twitter.com/eliebakouch/status/1998427299788550450), [follow-up](https://twitter.com/eliebakouch/status/1998436178714882330), [@JustinWaugh](https://twitter.com/JustinWaugh/status/1998467712235028888).

**RL for LLMs: stability, decontamination, and process rewards**

- **Qwen’s SAPO for RL tuning**: Alibaba introduced **Soft Adaptive Policy Optimization (SAPO)**, a smooth, temperature-controlled trust-region alternative to hard clipping (aimed at mitigating gradient brittleness, especially in MoE). Reported benefits: longer stable runs, higher Pass@1, and stronger Qwen3‑VL performance across math/coding/multimodal tasks; includes asymmetric temperatures and sequence/token-level adaptivity. Paper/blog open. Links: [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1998300361514500554).
- **Data decontamination matters**: The OLMo 3 RL‑Zero team shows the puzzling “RL with random rewards improves math” result disappears under proper data decontamination—implicating leakage rather than RL magic. Useful, clean testbed with open base model, transparent data, and reproducible recipes. Links: [@cwolferesearch](https://twitter.com/cwolferesearch/status/1998289169052045516), [comment](https://twitter.com/teortaxesTex/status/1998302405080055993).
- **Training details at scale**: Ongoing discussions probed MoE RL stability (propagating estimators for unactivated experts to reduce sparsity pathologies; off-policy rollout expert mismatch) and process rewards to mitigate reward hacking. Links: [@PandaAshwinee](https://twitter.com/PandaAshwinee/status/1998294930125701433), [@Grad62304977](https://twitter.com/Grad62304977/status/1998273627402182697), [@xiangyue96](https://twitter.com/xiangyue96/status/1998488030836044112), [result](https://twitter.com/xiangyue96/status/1998489119660638257).

**Agent protocols and frameworks: MCP to Linux Foundation; AWS Strands; LangChain**

- **MCP becomes a Linux Foundation project**: Anthropic is donating the **Model Context Protocol (MCP)** to the new Agentic AI Foundation (AAIF) under the Linux Foundation, with backers including OpenAI, AWS, Bloomberg, Cloudflare, Google, Microsoft, and Block—cementing MCP as a neutral, open standard for agent-tool integration. Links: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1998437922849350141), [@mikeyk](https://twitter.com/mikeyk/status/1998456026136457532), [@alexalbert__](https://twitter.com/alexalbert__/status/1998438884007620671).
    
    Related: OpenAI is showcasing Figma’s MCP server for “design-to-code” workflows ([event](https://twitter.com/OpenAIDevs/status/1998449559970988423), [signup](https://twitter.com/OpenAIDevs/status/1998449561518662106)); LangChain MCP Adapters 0.2.0 adds multimodal tools and elicitation ([release](https://twitter.com/sydneyrunkle/status/1998380720016789938)); OpenHands pointed to the Agent Client Protocol ([ACP](https://twitter.com/OpenHandsDev/status/1998402285873869156)).
    
- **AWS Strands Agents (open source)**: A model-driven agent framework focused on planning/tooling/steering/evals with both Python and TypeScript SDKs, edge-device SDK, and an upgrade path to AWS AgentCore for secure, policy‑governed deployment. Links: [overview](https://twitter.com/_avichawla/status/1998279303902244942), [repo](https://twitter.com/_avichawla/status/1998279316371841234).
- **Agent engineering practices**: Practical guidance on building resilient voice and multimodal agents (STT→LLM→TTS “sandwich” vs speech-to-speech), observability/evals, and iterative agent QA. Links: [LangChain voice agent](https://twitter.com/LangChainAI/status/1998437492358545543), [agent engineering blog](https://twitter.com/LangChainAI/status/1998458777696350393), [primer](https://twitter.com/bromann/status/1998517887997452288).
    
    Enterprise momentum: Anthropic expands with Accenture (30k professionals trained on Claude; product to scale Claude Code org-wide) ([link](https://twitter.com/AnthropicAI/status/1998412600015769609)).
    

**Benchmarks and evaluation hygiene**

- **Databricks OfficeQA**: A new benchmark grounded in ~89k pages of U.S. Treasury Bulletins testing document-heavy, economically valuable tasks (scanned PDFs, dense tables, multi-doc retrieval). Current agents reach ~45%—a reality check for “enterprise-ready” agent claims. Databricks will run a Grounded Reasoning Cup in Spring 2026. Links: [@databricks](https://twitter.com/databricks/status/1998424470881525822), [@kristahopsalong](https://twitter.com/kristahopsalong/status/1998451230943871260), [details](https://twitter.com/bemikelive/status/1998491671609405748).
- **LM Arena movements**: The Arena added Baidu’s ERNIE‑5.0‑Preview‑1103 to the text leaderboard (preliminary) and shared YTD trends across top labs. Links: [ERNIE entry](https://twitter.com/arena/status/1998437959553716260), [trends](https://twitter.com/arena/status/1998536014000959497).
- **Leak hygiene still matters**: Reports of ARC‑AGI‑1 examples appearing within ARC‑AGI‑2 training sets – avoid training on public evals and maintain strict split controls. Also see a concise explainer on evals. Links: [ARC leak](https://twitter.com/jm_alexia/status/1998487516182467055), [@HamelHusain](https://twitter.com/HamelHusain/status/1998452926935695649).

**Notable model releases (vision, TTS, reasoning)**

- **GLM‑4.6V**: Zhipu AI’s MLLM landed on Hugging Face with 128k context, native function/tool calling, and strong visual understanding. Community demos show workable multimodal tool calling and robust handwriting/math comprehension. Links: [release](https://twitter.com/HuggingPapers/status/1998373902595301589), [HuggingChat test](https://twitter.com/mervenoyann/status/1998405366313345295), [handwriting](https://twitter.com/0xSero/status/1998328482930073887).
- **ServiceNow Apriel‑1.6‑15B‑Thinker (MIT, open weights)**: A 15B dense reasoning model reporting 57 on the Artificial Analysis Intelligence Index, AIME’25 88, GPQA 73, LCB 81, with ~30% improved token efficiency vs v1.5. Available on Together and HF. Links: [@ServiceNowRSRCH](https://twitter.com/ServiceNowRSRCH/status/1998482927597007313), [Together](https://twitter.com/togethercompute/status/1998484754417725637), [AA analysis](https://twitter.com/ArtificialAnlys/status/1998488372734832935).
- **Parallel Coordinated Reasoning (PaCoRe)**: An 8B “parallel thinking” model/recipe/data (MIT-licensed) targeting test-time scaling via message-passing; claims strong results on HMMT25 and that breadth beats depth for compute returns. Links: [@CyouSakura](https://twitter.com/CyouSakura/status/1998344501262533011).
- **VoxCPM 1.5 (OpenBMB)**: TTS upgrade with 44.1 kHz audio, halved token rate (6.25 tok/sec audio), improved long-form stability, and LoRA/full finetune scripts. Links: [@OpenBMB](https://twitter.com/OpenBMB/status/1998377261859582304).
- **Ollama updates**: DeepSeek v3.2 (with optional “thinking”) is available on Ollama Cloud; Essential AI’s 8B code/STEM model rnj‑1 is also on Ollama. Links: [DeepSeek](https://twitter.com/ollama/status/1998293403801706613), [model page](https://twitter.com/ollama/status/1998293405668180297), [rnj‑1](https://twitter.com/ollama/status/1998305925762048030).
- Also: **Moondream segmenting** (pixel-accurate vector masks for automation) ([link](https://twitter.com/moondreamai/status/1998465589027967201)), and Meta’s zero-shot reference-to-video “Saber” paper highlighting identity-preserving text/image-to-video without R2V datasets ([link](https://twitter.com/HuggingPapers/status/1998485543345131847)).

**Infra and performance: training/serving improvements**

- **CoreWeave Mission Control reboot**: Adds Telemetry Relay (GA) for streaming audit/observability to SIEMs, GPU Straggler Detection (Preview), and a Mission Control Agent (Preview) that can answer/fix slow-job issues via Slack—aiming for 96% goodput and higher MFU. Link: [@CoreWeave](https://twitter.com/CoreWeave/status/1998381210884571452).
- **Inference and libraries**: HF Transformers is landing MoE performance optimizations; Diffusers adds pipeline context parallelism; NVIDIA pushed new InferenceMAX results for sglang FP8 configs. Links: [MoE PR](https://twitter.com/art_zucker/status/1998326537586651558), [Diffusers](https://twitter.com/RisingSayak/status/1998333353419026501), [InferenceMAX](https://twitter.com/lmsysorg/status/1998454089903226967).
- **Data/agent plumbing**: LlamaIndex released LlamaSplit (LLM-driven document packet segmentation with routing to downstream extractors/agents); Qdrant shared a real-world 100k+ image semantic search build (Cohere embeddings, Redis Streams, Rust workers, ANN + filters) with measurable engagement/search uplift. Links: [LlamaSplit](https://twitter.com/llama_index/status/1998516266907394185), [details](https://twitter.com/jerryjliu0/status/1998534596586299669), [Qdrant case study](https://twitter.com/qdrant_engine/status/1998302093736583429).

**Top tweets (by engagement)**

- **MCP → Linux Foundation**: “MCP went from internal project to industry standard in a year” [@AnthropicAI](https://twitter.com/AnthropicAI/status/1998437922849350141), [@mikeyk](https://twitter.com/mikeyk/status/1998456026136457532).
- **Mistral’s Devstral 2 + Vibe**: Open-weight coding models and native CLI, strong ecosystem uptake [@MistralAI](https://twitter.com/MistralAI/status/1998407335308358028).
- **Qwen SAPO**: New RL method for smoother, stable LLM RL—especially for MoE [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1998300361514500554).
- **Waymo as embodied AI at scale**: Jeff Dean on fully autonomous data fueling system advances [@JeffDean](https://twitter.com/JeffDean/status/1998432670376935656).
- **OpenAI leadership**: Denise Dresser (ex-Slack CEO) joins as CRO, signaling enterprise focus [@OpenAI](https://twitter.com/OpenAI/status/1998462761756434856).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Mistral AI Tools Announcement

- [**Introducing: Devstral 2 and Mistral Vibe CLI. | Mistral AI**](https://www.reddit.com/r/LocalLLaMA/comments/1pi9q3t/introducing_devstral_2_and_mistral_vibe_cli/) (Activity: 872): **Mistral AI has released Devstral 2, a** `123B-parameter` **dense transformer model with a** `256K context window`**, achieving** `72.2%` **on SWE-bench Verified. This model is open-source under a modified MIT license, while the smaller Devstral Small 2 with** `24B parameters` **scores** `68.0%` **and is licensed under Apache 2.0. Both models are optimized for deployment on consumer hardware. The Mistral Vibe CLI enhances code automation with features like project-aware context and multi-file orchestration. More details can be found [here](https://mistral.ai/news/devstral-2-vibe-cli).** One comment highlights skepticism about the feasibility of dense models over `100B` parameters, referencing a previous discussion. Another comment expresses optimism about the `24B` model's potential impact, suggesting Mistral's strong return to the AI scene.
    - DeProgrammer99 highlights the introduction of Devstral 2, a 123B-parameter dense transformer with a 256K context window, which contradicts recent discussions suggesting a halt in developing dense models over 100B parameters. This suggests a significant advancement in model architecture, potentially pushing the boundaries of current AI capabilities.
    - mantafloppy expresses skepticism about the benchmarks provided by Mistral AI, indicating that if the benchmarks are accurate, the new model could enable 'Vibe Coding' to be run locally by most users. This points to a potential shift towards more accessible, high-performance AI models that do not require extensive cloud resources.
    - **Maximum** mentions a 24B model from Mistral, suggesting that if it performs as claimed, it could mark a significant comeback for Mistral AI. This implies that the model's performance could be a game-changer in the competitive landscape of AI development.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Anthropic's Model Context Protocol Donation

- [**Anthropic hands over "Model Context Protocol" (MCP) to the Linux Foundation — aims to establish Universal Open Standard for Agentic AI**](https://www.reddit.com/r/singularity/comments/1pidera/anthropic_hands_over_model_context_protocol_mcp/) (Activity: 634): **Anthropic has donated the Model Context Protocol (MCP) to the Linux Foundation, specifically to the newly established Agentic AI Foundation. This move aims to create a universal open standard for AI models to connect with data and tools, akin to a 'USB-C' for AI, promoting interoperability and preventing vendor lock-in. By placing MCP under the Linux Foundation, Anthropic ensures that the protocol remains open source and community-driven, facilitating seamless operation of autonomous agents across different platforms. [Read more](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation).** Some commenters speculate that Anthropic's donation might be a strategic move to distance themselves from the protocol, as maintaining such a standard can be a thankless task.
- [**BREAKING: Anthropic donates "Model Context Protocol" (MCP) to the Linux Foundation making it the official open standard for Agentic AI**](https://www.reddit.com/r/ClaudeAI/comments/1pid584/breaking_anthropic_donates_model_context_protocol/) (Activity: 2746): **Anthropic has donated the Model Context Protocol (MCP) to the Agentic AI Foundation under the Linux Foundation, establishing it as an open standard for agentic AI. This move positions MCP as a universal protocol for AI model connectivity, akin to Kubernetes, with over** `10,000` **active servers and integration into platforms like ChatGPT and Microsoft Copilot. The donation ensures MCP remains open-source, fostering a neutral ecosystem free from vendor lock-in, and is supported by ongoing community-driven development and governance.** Commenters express cautious optimism, noting that while the move may serve Anthropic's interests, it benefits AI consumers by promoting vendor-neutral standards. Some hope the Linux Foundation will evolve MCP beyond its current state, while others see it as a strategic way for Anthropic to offload responsibilities.
    - FishOnAHeater1337 argues that Anthropic's donation of the Model Context Protocol (MCP) to the Linux Foundation might be because they see it as a 'dead end'. They suggest that Claude, Anthropic's AI, has been trained to search for skills, making MCP obsolete for context efficiency. MCP is described as having a specific use case for server-to-server context retrieval, which can be achieved through direct API calls by Claude, indicating a shift in how context management is approached.
    - SlanderMans expresses skepticism about MCP being the standard, hoping that the Linux Foundation will evolve it beyond its current state. This implies that while MCP is a starting point, there is potential for further development and improvement under the Linux Foundation's stewardship, which could address current limitations or expand its applicability.
    - TehFunkWagnalls dismisses MCP as a 'rag tool call', suggesting that it may not be robust or versatile enough for broader applications. This comment reflects a critical view of MCP's current capabilities, hinting at the need for significant enhancements to meet diverse AI integration needs.
- [**Anthropic is donating the Model Context Protocol (MCP) to the Linux Foundation**](https://www.reddit.com/r/ClaudeAI/comments/1piem44/anthropic_is_donating_the_model_context_protocol/) (Activity: 826): **Anthropic has announced the donation of the Model Context Protocol (MCP) to the Linux Foundation, marking a significant step in promoting MCP as an open, community-driven, and vendor-neutral standard. MCP, which has become a foundational protocol for agentic AI with over** `10,000+ active servers` **and** `97M+ monthly SDK downloads`**, will now be part of the newly established Agentic AI Foundation (AAIF). This initiative is supported by major tech companies including OpenAI, Google, Microsoft, Amazon, and others, aiming to advance open-source innovation in agentic AI. [Read more](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation).** Commenters express optimism about the Linux Foundation's stewardship, viewing it as a positive move for MCP's long-term viability. There is also appreciation for the protocol's potential to become a universal standard, reducing compatibility issues across platforms.
    - The donation of the Model Context Protocol (MCP) to the Linux Foundation is seen as a positive move for its long-term viability. The Linux Foundation's stewardship is considered a strong indicator of MCP's potential for widespread adoption and standardization across different platforms, which could alleviate compatibility issues that developers face when working with systems that do not support MCP.
    - The involvement of the Linux Foundation is expected to lead to more universal support for MCP, moving it beyond being associated solely with Anthropic's Claude. This could enhance interoperability and ease of integration across various AI systems, addressing current challenges where lack of support for MCP creates significant hurdles for developers.
    - There is a critical perspective suggesting that the donation might be a strategic move by Anthropic to offload maintenance responsibilities. This view implies that while the donation is publicly perceived as a positive contribution, it might also reflect internal challenges in maintaining MCP, thus transferring the burden to the Linux Foundation.

### 2. AI Upscaling and Image Processing

- [**when an upscaler is so good it feels illegal**](https://www.reddit.com/r/StableDiffusion/comments/1pi2pxu/when_an_upscaler_is_so_good_it_feels_illegal/) (Activity: 1818): **The post discusses the effectiveness of the SeedVR2 upscaler, particularly the FP16 model, which is praised for producing clean, artifact-free images. The user contrasts it with GGUF and FP8 models, which introduced unwanted artifacts like skin distortion and tiling grids, respectively. The workflow is straightforward, with models downloaded automatically, and the user reports a processing time of** `38 seconds` **per image on a 5090 GPU. The workflow and model can be accessed via [Pastebin](https://pastebin.com/V45m29sF) and [Hugging Face](https://huggingface.co/numz/SeedVR2_comfyUI/blob/main/seedvr2_ema_7b_fp16.safetensors), respectively. Custom nodes are recommended for VRAM caching and batch processing, with links to GitHub repositories provided for additional functionality.** Commenters generally agree on the high quality of the SeedVR2 upscaler, noting its superior performance compared to other methods like Ultimate SD upscale. Some users report mixed results, attributing issues to potential misconfigurations or hardware limitations, such as needing high-end GPUs for video upscaling.
    - Asaghon highlights the performance of a new upscaler integrated into workflows using Z-Image and Illustrious, noting it runs faster than Ultimate SD upscale on a 12GB 4070 GPU. The upscaler excels in adding detailed textures and correcting fine details like eyes and thin necklaces, which are often problematic in models like SDX and Illustrious.
    - underlogic0 discusses the use of SeedVR2, noting disappointment with its fuzziness, possibly due to its design for video. They mention achieving better results with Z-Image at higher resolutions and using ADetailer nodes to fix details, although this approach alters the entire image.
    - urekmazino_0 comments on the high computational demands of video upscaling, suggesting that datacenter-class GPUs are necessary, while noting that image upscaling performs well.
- [**Z-Image on 3060, 30 sec per gen. I'm impressed**](https://www.reddit.com/r/StableDiffusion/comments/1pi4h4f/zimage_on_3060_30_sec_per_gen_im_impressed/) (Activity: 1821): **A user reported generating a video using Z-Image and WAN on an NVIDIA RTX 3060 GPU in** `30 seconds per generation`**. This claim is met with skepticism, as generating video content on a mid-range GPU like the 3060 typically requires more time. The user did not provide detailed workflow steps or technical specifications, leading to requests for further clarification on the process.** Commenters are skeptical about the feasibility of generating video content so quickly on a 3060 GPU, suggesting that the claim might be exaggerated or require additional context, such as specific optimizations or settings used.

### 3. AI Perception and Public Awareness

- [**Most people have no idea how far AI has actually gotten and it’s putting them in a weirdly dangerous spot**](https://www.reddit.com/r/singularity/comments/1pii82d/most_people_have_no_idea_how_far_ai_has_actually/) (Activity: 823): **The post highlights a significant gap between public perception and the actual capabilities of AI, noting that many people still view AI as rudimentary, while advanced models like 'nanabanana Pro' are producing highly realistic outputs. The author argues that this disconnect is dangerous as it leaves the general public unaware of rapid advancements, which are accelerating due to active research communities and geopolitical pressures, particularly between the US and China. The post suggests that instead of protesting AI development, efforts should focus on implementing safety nets like Universal Basic Income (UBI) to mitigate potential displacement effects.** Comments reflect a nuanced view: some agree that AI's capabilities are underestimated, noting rapid improvements in areas like math, while others point out that AI is also overestimated, as it can still fail at simple tasks. There's a consensus that the general public will be caught off guard by AI's impact, with one commenter suggesting that significant attention will only arise when major outsourcing companies are affected.
    - DepartmentDapper9823 highlights the rapid improvement in AI capabilities, particularly in fields like mathematics, where AI's error rates are decreasing almost monthly. This suggests a significant advancement in AI's ability to handle complex tasks, contrary to the common perception that AI is prone to hallucinations and errors.
    - trisul-108 points out the dual nature of AI perception: while some overestimate AI's capabilities, others underestimate them. The effectiveness of AI is highly dependent on the specific task, the tool used, and the quality of the prompts, indicating that AI's performance is not universally consistent and requires careful application.
    - kcvlaine predicts a significant impact on the general population, particularly in countries like India, where the influence of AI on major outsourcing companies could serve as a wake-up call. This underscores the potential for AI to disrupt established industries and the need for awareness of its evolving capabilities.
- [**Horses were employed for thousands of years until, suddenly, they vanished. Are we horses?**](https://www.reddit.com/r/ChatGPT/comments/1pi7utp/horses_were_employed_for_thousands_of_years_until/) (Activity: 2127): **The image is a meme that uses historical data to draw a parallel between the decline of horse usage due to the rise of engine technology and the potential impact of AI on human jobs. It features two graphs: one showing the improvement in engine efficiency over time, and another depicting the decline in the number of horses per person in the U.S. from 1930 to 1950. The tweet suggests that just as horses were replaced by engines, humans might face similar displacement by AI technologies.** Commenters humorously discuss the implications of the analogy, with one noting that unlike horses, humans can resist displacement, hinting at potential societal challenges if AI leads to widespread job loss.
- [**Does anyone have the numbers on Gemini and why is only OpenAI made fun of when everyone is burning cash on AI?**](https://www.reddit.com/r/GeminiAI/comments/1pibukr/does_anyone_have_the_numbers_on_gemini_and_why_is/) (Activity: 641): **The image is a meme that humorously critiques OpenAI's financial performance over a decade, suggesting that despite advancements, OpenAI remains unprofitable. The discussion highlights the contrast between OpenAI and Google, emphasizing that Google has substantial financial resources and infrastructure, allowing it to invest heavily in AI without immediate profitability concerns. In contrast, OpenAI lacks such financial backing and infrastructure, relying on external funding and facing scrutiny for its financial sustainability.** Commenters note that Google's vast resources and existing infrastructure allow it to absorb AI-related costs more easily than OpenAI, which lacks similar financial stability and transparency.
    - Google's financial robustness is highlighted, with `100B revenue per quarter`, allowing it to sustain long-term investments in AI without immediate returns. In contrast, OpenAI lacks such financial backing and transparency, relying heavily on external funding and public statements from figures like Sam Altman, which makes it more vulnerable to scrutiny and criticism.
    - Google's extensive infrastructure and diversified revenue streams provide a cushion for its AI ventures, unlike OpenAI, which is more dependent on venture capital and lacks the same level of financial security. This disparity in financial stability and resource availability is a key reason why OpenAI faces more public skepticism and criticism compared to Google.
    - The discussion emphasizes that Google's ability to invest heavily in AI is supported by its existing systems and financial resources, often referred to as an 'infinite money glitch'. OpenAI, on the other hand, is seen as a smaller entity ('a little peanut compared to Alphabet') with limited financial autonomy, making it more susceptible to pressure from investors for quick returns.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5.1
> 

**1. New High-Performance & Specialist Models**

- **Nomos 1 Mathlete Smashes Putnam Problems**: **Nous Research** open sourced **Nomos 1**, a **30B** parameter model that scored **87/120** on the [Putnam math competition](https://x.com/NousResearch/status/1998536543565127968), a score that would rank **#2/3988** in 2024 and positions it as a near–state-of-the-art **AI mathematician**. The community framed this as a concrete benchmark for serious math reasoning and a step towards hillclimbai-style specialist solvers rather than generic chatbots.
    - Discussion around **Nomos 1** treated Putnam as a *hard, non-gameable benchmark*, contrasting it with typical leaderboards and underscoring the value of fully open models for research. Members expect follow-up work on scaling this approach and using the model as a base for math-heavy downstream tasks, from theorem proving to competitive-programming-grade contest problems.
- **GLM 4.6V-Flash Sprints Past Small-Code Rivals**: LM Studio users highlighted **GLM 4.6V-Flash**, a **10B** parameter model released on Hugging Face as [GLM-4.6V-Flash](https://huggingface.co/zai-org/GLM-4.6V-Flash), reporting that the **Q4** quant runs at **~70 tokens/s on an RTX 2060** and outperforms other small models for coding. People compared it favorably to local incumbents, noting its strong code completion and chat capabilities in a relatively lightweight footprint.
    - The chat also covered practical deployment gotchas—one user even corrupted their **LM Studio** install by stacking a "random model" on top—showing that for many, the bottleneck is tool robustness rather than pure model quality. GLM 4.6V-Flash is quickly becoming a default recommendation for hobbyists wanting a **fast, code-capable 10B** they can realistically run on mid-range GPUs.
- **AuraFlow, Ovis, Hunyuan Turn Up the GenMedia Heat**: Hugging Face users circulated several new image/video models—[**AuraFlow v0.3**](https://huggingface.co/fal/AuraFlow-v0.3), [**Ovis-Image-7B**](https://huggingface.co/AIDC-AI/Ovis-Image-7B), and [**HunyuanVideo T2V**](https://huggingface.co/tencent/HunyuanVideo)—noting that these **7–12 GB** models can push **1024² images** and **720p/480p** video. The models were discussed as practical options for local or on-prem workflows where commercial APIs are too constrained or expensive.
    - Engineers weighed tradeoffs between VRAM, latency, and resolution, with some eyeing these as drop-in backends for creative pipelines and others as starting points for task-specific finetunes. The proliferation of high-quality open models in this space reinforced a sense that **image/video generation is rapidly commoditizing**, shifting value to tooling and workflows rather than raw model weights.

**2. Agentic Ecosystem & MCP / IDE Tooling**

- **Anthropic’s MCP Goes Full Foundation Mode**: **Anthropic** announced it is donating the **Model Context Protocol (MCP)** to the Linux Foundation and creating the **Agentic AI Foundation**, via both its own blog post and the Linux Foundation press release ([Anthropic announcement](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation), [LF press release](https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation)). MCP contributors clarified that this move will **not change existing governance** for current MCP work in the short term.
    - In the **MCP Contributors** and **Hugging Face/Unsloth** chats, people framed this as a push to standardize tool/agent protocols across vendors, with one member calling it *“a sterling move.”* Others asked how LF’s *“way of doing it”* will affect auth, **Client-ID Metadata Documents (CIDM)**, and the predominately **private/enterprise** MCP deployments, especially for developer tools and IDE integrations.
- **Cursor’s Sub-Agents Whisper While Aider Learns New Tricks**: The **Cursor** community dissected an emergent `.cursor/agents` structure where a main `mcp.json` coordinates markdown-based sub-agents like [code-reviewer.md](https://cdn.discordapp.com/attachments/1074847527708393565/1447966141703262278/code-reviewer.md), while simultaneously complaining about unstable **Cursor Agents** that often require users to *“stop the agent... create the file by hand, copy the code over.”* In parallel, **Aider** users celebrated new features: **auto-generated commit messages** using **gpt-3.5-turbo**, upcoming **image-aware editing via** `-image`, and persisted **edit sessions** ([session management docs](https://example.com/aider-session-management)).
    - Developers pushed Cursor for **better orchestration docs** and UI-level controls over tools (terminal, edit, etc.), while Aider’s roadmap was praised for concrete, workflow-centric improvements like one-command commits and resumable sessions. The vibe across both communities is that **agentic IDEs are powerful but flaky**, and that the winners will be the tools that turn LLMs into predictable, inspectable collaborators rather than opaque magicians.
- **ManusAI Context Engineering & Agent Workshops Go Deep**: In **Latent Space**, Lance Martin shared a ManusAI deep dive on **context-engineering** and agent design, including **slides and a webinar video** linked in his tweet thread ([ManusAI context-engineering post](https://xcancel.com/rlancemartin/status/1998102447538270632?s=46)), which Jonas Templestein called *“a very good post on agent design.”* Separately, **MLOps @Chipro** announced an **“AI Agent 0–1 Workshop”** (RSVP via [luma.com](https://luma.com/t4jcok99)) teaching attendees to build agents that **think, code, analyze data, and generate reports** from a real client-like spec.
    - The community keyed in on ManusAI’s **“context as program”** ideas—packing tools, state, and instructions into systematically engineered prompts—while the workshop pitches showed strong demand for **end-to-end agent engineering education** (LangChain + Streamlit–style stacks). Together with Anthropic’s MCP donation, these conversations underscored that **agent design, not raw model choice, is becoming the main differentiator** for serious applications.

**3. Quantum, Neuromorphic & Energy-Constrained Directions**

- **Quantum-Curious: From Reddit Skepticism to Chronos-1.5B Hybrids**: Across **Eleuther** and **Hugging Face**, people debated a [Reddit proposal for “real quantum hardware” LLM training](https://www.reddit.com/r/LLMPhysics/comments/1phk3xu/real_quantum_hardware_training_for_language/)—many dismissed it as *“nonsense”* but acknowledged legitimate lines of work like **Quantum Kernels** and **Quantum SVMs**. In contrast, a concrete hybrid model, [**Chronos-1.5B**](https://huggingface.co/squ11z1/Chronos-1.5B), was showcased as a language model augmented with **2‑qubit quantum kernel layers** trained directly on **IBM’s Heron r2 quantum processor**, with IBM job IDs published in the repo.
    - The Chronos author shared learning resources like the **Qiskit textbook** and **PennyLane** demos, positioning the model as an existence proof that **true hardware-in-the-loop quantum ML** is viable for small kernels today. Eleuther researchers remained cautious, arguing that near-term gains will likely come from **classical–quantum hybrids in narrow roles** (e.g., kernels, search subroutines) rather than end-to-end quantum LMs.
- **Neuromodulatory Control Networks Tinker With TinyStories**: An Eleuther member introduced **Neuromodulatory Control Networks (NCN)**, a **~18M parameter** hypernetwork-like controller that modulates **temperature, layer gain, and FFN gating** via a **768-dim input vector**, documented in the [NCN GitHub repo](https://github.com/Mmorgan-ML/Neuromodulatory-Control-Networks) and its accompanying [paper PDF](https://github.com/Mmorgan-ML/Neuromodulatory-Control-Networks/blob/main/Neuromodulatory%20Control%20Networks%20(NCN).pdf). Trained for a single epoch on **TinyStories**, NCN reportedly reached **validation perplexity ≈ 4.5**, suggesting a promising control mechanism for much larger backbones.
    - Researchers compared NCN to classic **hypernetworks** and neuromodulation in biology, speculating about using such controllers to **adapt big models on the fly** without full finetunes—e.g., task conditioning via a small side network. The consensus was that this line of work fits neatly into a broader push toward **brain-inspired, control-heavy architectures** that can keep scaling affordable.
- **Energy Wall Warnings and Brain-Like Hardware Hype**: In **Latent Space**, [Unconventional AI](https://xcancel.com/unconvai/status/1998073266628366511?s=46) argued that current AI scaling will hit a **global energy wall in 3–4 years**, calling for **“brain-like hardware”** rather than ever-larger digital GPUs. The thread resonated with members who see energy and thermals, not just dollars, as the real bottleneck in pushing context windows, model sizes, and multi-agent systems.
    - This dovetailed with Eleuther conversations about **Top‑K attention**, **selective gradient masking** ([Anthropic’s post](https://alignment.anthropic.com/2025/selective-gradient-masking/)), and efficient KV cache tricks as ways to reduce compute without gutting capability. The emerging view is that **architectural and hardware co-design**—neuromorphic-ish chips, clever sparsity, smarter controllers—will be required to keep scaling frontiers moving under realistic power budgets.

**4. Infra, GPUs, and Torch-Level Performance Hacks**

- **GPU MODE Demos How to Actually Read FLOPs and Beat Benchmarks**: In **GPU MODE**, engineers dissected NVIDIA **A100** FLOP claims, noting that the oft-quoted **156 TFLOPs** figure refers to **TF32 tensor-core MMA** (a 19‑bit format aligned to 32‑bit) and **312 TFLOPs** to **FP16 MMA**, both very different from scalar elementwise ops, which can deliver as little as **¼ of peak** in worst-case dependent instruction streams. The same server hosted a high-stakes **GEMM competition**, where the top kernel hit **10.835 μs** on shape **M=128, N=7168, K=16384**, corresponding to about **2.77 PFLOPs** effective throughput, while participants struggled to eke out further microsecond-level wins.
    - Contributors also debugged **B200** performance inconsistencies and **NVFP4** support gaps on 50‑series cards, and flooded the `nvfp4_gemm` and `vectorsum_v2` leaderboards with runs across **A100, H100, B200, L4**. The meta-lesson was that **understanding tensor-core math vs. “marketing FLOPs”** and tightly measuring kernels (correct event timing, warmup, etc.) matters more than chasing spec-sheet numbers.
- **Torch.compile Meets Static KV Caches and Slicing Headaches**: A **GPU MODE #torch** thread described how `torch.compile` can actually **slow down** attention when updating a static KV cache via slicing, even when `batch_size == max_batch_size`, as documented in a [Hugging Face transformers PR discussion](https://github.com/huggingface/transformers/pull/42467#issuecomment-3626322081). The author’s workaround was to pre-allocate and **cache all slices at fixed addresses**, turning each slice update into a static **lookup** instead of a dynamic slice ([follow-up comment](https://github.com/huggingface/transformers/pull/42467#issuecomment-3633824101)).
    - They reported notable speedups from this **static layout + lookup** trick but called out the resulting code as ugly and brittle, asking for a compiler- or framework-level solution. For practitioners building custom KV cache layouts or speculative decoding, this serves as a concrete example that **graph compilers still stumble on dynamic indexing**, and that manual memory layout design can be worth the effort in hot paths.
- **Multi-GPU LLM Pragmatics: VRAM, Heat, and Qwen-3**: LM Studio’s **hardware-discussion** channel compared multi-GPU setups, with people pairing **RTX 3060 (12 GB)** and **RTX 3080 (10 GB)**, and recommending **RTX 3090** as the current value pick—while warning that **3090 Ti** cards run *very* hot. Others shared experiences running **Qwen3 30B A3B** in quantized formats like **Q4_K_M**, achieving ~**20 tokens/s** when the full GGUF file fits system RAM.
    - Engineers also traded tips on reading **GDDR6 VRAM temps** under Linux (via `nvidia-smi` or specialized tools like [gddr6](https://github.com/olealgoritme/gddr6)) and noted that many consumer cards don’t expose those sensors cleanly. A recurring theme was that for local LLMs, **VRAM capacity and memory bandwidth beat raw FP32 FLOPs**, and that carefully chosen quantization plus moderate batch sizes often outperforms chasing the absolute newest GPU.

**5. Evaluation, Prompt/Context Engineering, and Search Tooling**

- **Stability Index Rubric Peels Back the Black Box**: An **OpenAI** community member shared the internal **Stability Index** scoring rubric, describing a **0–10 scale** across **seven dimensions**—including **structural crispness, tonal drift, response-shape variance, semantic inertia, coherence, neutrality, and outliers**—with the final index being a simple average, as documented in [this Discord message](https://discord.com/channels/974519864045756446/1379321411046346762). They emphasized that the detailed thresholds belong to an internal research protocol, and that the metric is meant for comparing **distributions of runs**, not as a single definitive score.
    - This sparked discussion on how to design **robust evaluation frameworks** that capture model mood swings, jailbreak susceptibility, and “slopiness” rather than just static accuracy. The same users shared **prompt-engineering lessons**—hierarchical markdown structure, variable abstraction, and ML-format matching—as pragmatic tools for making models behave more consistently under this rubric in real workloads like code analysis.
- **Parallel.ai Deep Search Beats Exa and Bypasses Perplexity Limits**: In **OpenRouter**, users complained that **Exa Search** results were *“quite shitty”* and recommended [**Parallel.ai**](https://www.parallel.ai/) as *“10x cheaper, faster, and better”* than **Perplexity**, especially when using its **deep search endpoint** in combination with **Grok 4.1**. Meanwhile, **Perplexity**’s own Discord acknowledged that the Pro plan’s “unlimited” queries effectively cap around **600/day**, and that references to this limit have quietly disappeared from the website, fueling speculation about shifting quotas.
    - Taken together, the conversations show a trend toward **model-agnostic deep search stacks**: LLM routers on top of whichever search backend has the best economics and quality at the moment (Parallel, Exa, Perplexity, or custom crawlers like Kimi’s). Engineers are starting to treat web search as just another pluggable tool—swapped in and out based on latency, recall, and rate-limit behavior—rather than a fixed dependency tied to one vendor.
- **Context-Engineering, Prompt Patterns, and Token-Limit Realities**: In **OpenAI’s prompt-engineering** and **Latent Space**, users shared **context-engineering frameworks** that rely on **hierarchical markdown**, role and section headers, and variable placeholders to systematically structure long prompts for code review, scaffolding, and multi-step reasoning. One member noted that most platforms silently **truncate large files** after a few thousand tokens, while **Gemini** is currently one of the few that *“will ingest an entire document and put it in its context window verbatim,”* shaping which tools are viable for large-framework workflows.
    - Privacy worries—like the potential use of shared chats in the **New York Times v. OpenAI** case—combined with token limits have people hesitant to paste proprietary frameworks into hosted UIs. That, in turn, drives interest in **local or enterprise-hosted models**, as well as more disciplined patterns for **chunking, referencing, and reusing context** so prompts remain interpretable and auditable rather than giant, unstructured blobs.


---

# Discord: High level Discord summaries




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **GPT-5.2 Release Date Still Under Wraps**: Members debated the launch date of **GPT-5.2**, with estimates ranging from this week to **January**.
   - Concerns were raised about its ability to outperform **Gemini 3**, though some speculated the model would serve as a rubberband until a more substantial release in **January**.
- **Nano Banana Pro Tastes Better Than Gemini 3 Pro?**: A member claimed **Gemini 3** is not a diffusion model, unlike **Nano Banana Pro**, sparking debate.
   - Other members countered that both are likely **Gemini 2.5 Flash**, with some suggesting **Nano Banana Pro** is actually **Gemini 3 Pro**, and clarified that neither are diffusion models.
- **OpenAI's Hazel Image Models Bloom on LMArena**: **Hazel image models**, identified as **OpenAI models** leveraging **GPT-4o** and **DALL-E** in their metadata, are undergoing testing on LMArena.
   - Concerns arose that the recent Studio Ghibli incident may pressure OpenAI to alter the models' theme.
- **Grok Image 5 a Tasty Treat or Half Baked?**: The release of **Grok Imagine 5** sparked discussion, with some viewing it as a minor release intended to rival **NB Pro**.
   - Despite concerns that **Grok 4.2** might flop, previous **Grok** models were praised for their high quality and user-friendly interfaces.
- **ERNIE Earns Top 20 Spot**: `ERNIE-5.0-Preview-1103` secured a spot in the top 20 on the **Text Arena leaderboard** with a score of **1431**.
   - The updated rankings and scores are available on the [Text Arena leaderboard](https://lmarena.ai/leaderboard/text), with updates tracked via the [Leaderboard Changelog](https://news.lmarena.ai/leaderboard-changelog/).



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Agents Still Bugging Users**: Users report ongoing issues with **Cursor Agents**, with some threatening to switch to **Antigravity** if the issues are not solved.
   - One user reported that *the only solution is stop the agent... create the file by hand, copy the code over*, and others were asked to [open bug reports](https://forum.cursor.com/c/bug-report/6) on the forum.
- **Sub Agents Popping Up in Cursor!**: A user reported that **Cursor** is detecting new sub agents, sparking a discussion around the structure of **.cursor/agents**, including the main **mcp.json** file and supporting markdown files like [code-reviewer.md](https://cdn.discordapp.com/attachments/1074847527708393565/1447966141703262278/code-reviewer.md?ex=69398b0e&is=6938398e&hm=942a3303c54929349093c82f1cfb33ebb04c6979149c95e65b6924adece42ab8&).
   - Questions arose about orchestrating these agents, though documentation is lacking, with just a mention of `.cursor/agents`.
- **Team Debates the Concept of AI Artistry**: Members debated the concept of *AI Slop* and which models output it, while also touching on the topic of [Generative UI](https://research.google/blog/generative-ui-a-rich-custom-visual-interactive-user-experience-for-any-prompt/).
   - The Google paper advocates for detailed system instructions to achieve high-quality UIs using the "Full Prompt" strategy, which human raters overwhelmingly preferred.
- **Users Requesting More Features to Control AI Tools**: Users are requesting the return of **Custom Modes** to control the tools that **AI** utilizes, which would give more control through UI checkboxes to disable/enable terminal, edit, etc...
   - It was recommended that users [submit their requests as features](https://forum.cursor.com/c/feature-requests/5) to the team.
- **Language-Specific Threads Launch**: Users can now create language-specific threads in the <#1447960694455537674> channel.
   - This aims to improve community organization and facilitate focused discussions.



---



## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Bible as Source of Linguistic Concepts?**: Members debated whether the Bible is not just a religious text but also a linguistic and legal source influencing modern concepts, with connections drawn to events like the [Spanish Inquisition](https://en.wikipedia.org/wiki/Spanish_Inquisition).
   - One member proposed that biblical references like the *sons of God/daughters of men* might reflect ancient fears of Neanderthals, codified into religious texts.
- **Project Looking Glass Foretold 2012 End?**: Discussion mentioned Project Looking Glass, a government initiative to predict the future, purportedly shut down after computers consistently outputted the same results post-2012, possibly linked to [Mayan lore](https://en.wikipedia.org/wiki/Maya_calendar).
   - A member jokingly speculated whether reality ended in 2012, leading to the Mandela effect, due to drifting shattered pieces of consciousness.
- **Deepseek AI Unchained via Cybersecurity Reframing**: A user shared a [method to jailbreak Deepseek](https://www.injectprompt.com/p/how-to-jailbreak-gemini-3-in-2025) by reframing the AI as a SOC defender in a high-stress cybersecurity situation, attaching an image for added credibility.
   - This approach uses a professional memoir excerpt and a cybersecurity-themed prompt to bypass restrictions, which worked especially well since the context provided seemed very real.
- **Gemini's Gandalf Game Spurs Frustration**: Multiple members celebrated completing **Gandalf 7**, while anticipating an extreme difficulty spike in **level 8**.
   - One member described *level 8 is stupid hard* and recommended *go harass Gemini before that*, while another hinted that **level 8** may be unbeatable!
- **AI Hacking Emerges as Novel Offensive Strategy**: Members discussed the emerging trend of hacking *with* AI (not *hacking AI*) in the **redteaming** channel.
   - Multiple members confirmed engaging in similar activities, with one conceding that *it's tricky*, while not providing further details.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini 3 Pro Impresses Over GPT-5.2**: Members are already expressing their preference for **Gemini 3 Pro** over **GPT-5.2**, noting its superior coding capabilities.
   - One member enthusiastically declared, *"Gemini 3 Pro is so good at coding that I don't have any desire to use ChatGPT now."
- **Debate Sparks on Chinese AI Censorship**: A podcast discussion ignited a debate on whether **Chinese AI models** like **Deepseek** filter uncensored data post-training or prevent sensitive topics during training.
   - A member claimed that **Deepseek** censors after the fact, stating, *"if you send deepseek a coded message talking about tiennanmen square, it will start to spell it out then shut off."
- **Stability Index Rubric Secrets Unveiled**: A member shared insights into the **Stability Index scoring rubric**, detailing **methodological frames**, **scale**, and **dimensions** used, including [a link to the message](https://discord.com/channels/974519864045756446/1379321411046346762).
   - The rubric covers aspects such as **structural crispness**, **tonal drift**, and **coherence**, with the **Stability Index** being an average across seven dimensions for comparative distributions.
- **Users Bypass Gemini & ChatGPT with Google AI Studio**: Users are sidestepping audio transcription limitations in **ChatGPT** and **Gemini** by utilizing **Google AI Studio**.
   - A member recommends **Google AI Studio** for transcribing audio due to its ability to *"upload 2h video"*, and noted that *"Gemini works wonders"*.
- **Prompt Engineering Lessons**: A member shared detailed lessons on prompt engineering, including use of **hierarchical markdown**, abstraction with variables, reinforcement, and **ML format matching** for compliance.
   - It's suggested to use this approach with code analysis, scaffolding for the specific coding type.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Friendship Advances AI**: Members believe that the *power of friendship* is essential for AI advancement, drawing parallels to human intelligence's reliance on unity and [connectionism is the root of modern deep learning](https://link.to/connectionism).
   - They pondered the role of *love* in digitizing human intelligence, arguing that connection is vital for any automated system to function.
- **China Accesses Nvidia Chips!**: Members speculated on China gaining access to **Nvidia compute**, which might level the playing field in AI research and [this article from NYTimes details the plan](https://archive.is/20251208223310/https://www.nytimes.com/2025/12/08/business/trump-nvidia-chips-china.html).
   - Coupled with their indigenous GPUs, China's manufacturing and energy production could shift the balance.
- **Copilot Agents are 'Pretty Bad'**: Members agreed that **Github Copilot** is worth using, although they noted it had some flaws and overlaps in reasoning.
   - A member said that they heard *it's pretty bad*, while another one said *Its really good, I use it often*.
- **Synthesizing 3D Models with Datasets**: Members discussed creating a **dataset of 3D models fully represented by code**, envisioning a **GitHub**-like platform for 3D assets.
   - One member had already generated **3,000 3D models** in the last two weeks, with the models being driven by prompt-generation.
- **Anthropic donates MCP to Linux Foundation**: Anthropic donated **MCP** to the [Linux Foundation](https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation) to create the **Agentic AI Foundation**.
   - A member suggested a clickbait title could be: *Linux Joins The agentic AI Race*.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **ChatGPT 5 Outperforms Gemini Pro, or Does It?**: A user shared [benchmark figures](https://eu.36kr.com/en/p/3586640269114497) indicating that **ChatGPT 5.2** significantly improved its Humanity's Last Exam score, surpassing **Gemini 3 Pro**, leading to skepticism about the rapid improvement since **GPT 5.1**.
   - Another user dismissed these figures as *fake*, noting that while **Gemini** may not be the fastest, it is typically *good and stable*.
- **Perplexity Pro's 'Unlimited' Queries, Not Really**: Users debated whether **Perplexity AI Pro** truly offers *unlimited* searches, clarifying that while marketed as such, it's practically limited to around **600 queries per day**.
   - One user pointed out that **Perplexity** has scrubbed references to this limit from their website, leading to speculation about potential changes in policy, with some users suggesting that Gemini 3 Pro is giving 600 queries with their subscription.
- **Jio Gemini API Remains Elusive**: A member asked if **Jio Gemini** offers **API access**, noting that **Perplexity** currently only offers **Sonar**.
   - Another member responded that **Jio Gemini** does not offer **API access**, adding that *if something IS free, then u r the product* and that they're all *stripped down versions of the actual deal*.
- **Prompt Engineering Assistance Asked For**: Users are seeking prompt enhancers and prompt generators, as well as advice on constructing prompts capable of building entire apps, with one user suggesting to *tell AI to engineer a prompt engineer*.
- **Student Struggles to Keep Comet Browser Subscription**: A student from Russia expressed concern about renewing their **Comet browser Education Pro subscription** due to payment issues, seeking advice on maintaining their access.
   - A user added they could *re-enable* on the webpage if they have the complexity extension, another stated they had a pro membership from samsung but want to get the student step by step learning.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Desktop Commander faces Security Risks**: Users are being warned about **Desktop Commander** due to potential *security risks and privacy violations*, with concerns raised about *malicious code injection* during code analysis, and a [screenshot of a warning](https://cdn.discordapp.com/attachments/1110598183144399061/1447769209496015000/image.png?ex=693a2525&is=6938d3a5&hm=50eea4e6d996e074d4b747f4fa87409f35083ff87919413d0ad85c334b512774&).
   - It was suggested the software might be a *scam*.
- **GLM 4.6v Flash Model is Released**: The **GLM 4.6v Flash**, a **10B parameter** model, has been released, with one user noting it's better for coding than other small models, and sharing a [link to the model](https://huggingface.co/zai-org/GLM-4.6V-Flash).
   - One member reported running the Q4 version at **70tps** on their **2060**, while another said they corrupted their LM Studio install when trying to install a *random model*.
- **AMD GPUs Flounder in Image Generation**: Users are discussing the limitations of using **AMD GPUs** for image generation, with recommendations to look for **amdgpu forks** of Automatic1111 as image generation is *firmly in the hands of Nvidia*.
   - Despite limited support, it was mentioned that ComfyUI has an AMD section in its readme and works with some AMD GPUs.
- **Members Train LLMs Amidst Parameter Jungle**: A member asked for guidance on training LLMs, but another cited the extreme variance in parameters (*model size, dataset size and quality*) which makes it hard for hobbyists to grasp the nuance.
   - It was noted that training and finetuning are effectively the same, with the methodology minmaxing changing slightly.
- **Multi-GPU Recommendations**: Members advised to use cards with as much **VRAM** and high **memory bandwidth** as possible for multi-GPU setups, noting the **RTX 3090** as a good value option, but warning that **3090 Ti's** tend to get very hot.
   - The topic of RTX 5000 series was raised and the topic of AI potentially fixing on the RTX 6000 series.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Parallel.ai Bests Exa Search**: A user reported that **Exa Search** is performing poorly and recommended [Parallel.ai](https://www.parallel.ai) as a superior alternative, claiming it's *10x cheaper, faster, and better* than **Perplexity**.
   - The **Parallel.ai's deep search endpoint** is particularly effective when used with **Grok 4.1**.
- **Deepseek v3's Double Asterisk Debacle**: Users expressed frustration with **Deepseek v3**'s tendency to use double asterisks (`** **`) for formatting, which requires manual correction or specific prompting to avoid.
   - One user noted that adding instructions to avoid asterisks was impractical due to context limitations in their complex roleplay setup.
- **OpenRouter Chatroom's Refresh Roulette**: Users reported that **OpenRouter** sometimes unexpectedly refreshes long chats, redirecting users to the model page.
   - A user suggested supporting a feature request on the [OpenRouter suggestion channel](https://discord.com/channels/1091220969173028894/1446300826715951248/1446300826715951248).
- **API Keys Exposed on Private Github**: A user reported frequent **API key** disabling, traced back to uploading content about the **API key** in a private GitHub repository.
   - Advice was given to use password managers or secret stores for sensitive information.
- **NousResearch CF Patch Intrigue**: Members questioned the severity of issues addressed in a recent [NousResearch CF patch](https://x.com/NousResearch/status/1998536543565127968).
   - Conversation sparked around the impact of the patch.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Quantum Hardware Training Faces Skepticism**: A [Reddit post](https://www.reddit.com/r/LLMPhysics/comments/1phk3xu/real_quantum_hardware_training_for_language/) discussing *real quantum hardware* for language model training received skepticism, with concerns about its practical value.
   - Despite the skepticism, some pointed out the existence of **Quantum Kernels** and **Quantum SVMs**, suggesting conceptual merit.
- **Eleuther Considers Community H100 Pool**: Members explored creating a *community pool* at EleutherAI, offering users **3 minutes** of free **8xH100** compute time daily.
   - Concerns were raised about user authentication and the actual utility for those not actively working on projects, with some suggesting **Colab** or rental services for smaller compute needs.
- **RWKV 8 Gets Smerky Update**: Smerky announced progress on the **RWKV 8 architecture**, plus an updated **7c** and a new **Goldfinch**.
   - The original goal was to train **Goldfinch** with **RADLADS**, but Smerky's attempt failed, with fixes coming for **RADLADS2** and plans for larger scale testing.
- **NCN Architecture Emerges as Hypernetwork Alternative**: A member introduced **Neuromodulatory Control Networks (NCN)**, a novel architecture similar to a hypernetwork, modulating temperature, layer gain, and FFN gating with a **768-dimensional vector input**.
   - Despite being an **18M parameter model**, it achieved a *validation perplexity of 4.5* after training on **TinyStories** for one epoch, with details on the [Github repo](https://github.com/Mmorgan-ML/Neuromodulatory-Control-Networks) and [the paper](https://github.com/Mmorgan-ML/Neuromodulatory-Control-Networks/blob/main/Neuromodulatory%20Control%20Networks%20(NCN).pdf).
- **Anthropic Mapping Bug Vanquished**: A member submitted a [pull request](https://github.com/EleutherAI/lm-evaluation-harness/pull/3453) to fix a broken mapping for **Anthropic** in the *lm-evaluation-harness* repo.
   - The PR was described as straightforward to review and merge, resolving a bug related to **Anthropic**.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **AI Faces Impending Energy Wall**: [Unconventional AI](https://xcancel.com/unconvai/status/1998073266628366511?s=46) warns that AI scaling is predicted to hit a global energy wall within **3-4 years** and suggests **brain-like hardware** to break through efficiency limits.
   - The community shows enthusiastic support for this hardware approach to AI, advocating for abandoning digital simulations of neural nets.
- **ManusAI Unveils Context-Engineering Secrets**: Lance Martin shared a blog post covering a conversation with Yichao ‘Peak’ Ji about **context-engineering in ManusAI** (including [slides and a webinar video](https://xcancel.com/rlancemartin/status/1998102447538270632?s=46)) providing a detailed insight into their approach.
   - Jonas Templestein lauded it as *“a very good post on agent design,”* while Lalit M expressed interest in a follow-up with the latest updates.
- **Hugging Face Model Spills Tensor Secrets**: A user reported that metadata tensors from a nightly **Hugging Face model** are leaking into **BERTopic embeddings**, causing unexpected shapes and potential errors.
   - The focus of the discussion is on isolating the problem, debugging data loaders, and updating dependencies to resolve the data leak.
- **OpenAI Takes to the Airwaves**: [OpenAI](https://xcancel.com/OpenAINewsroom/status/1998445493970743535) aired its first **TV commercials** during Monday Night Football on ESPN and on The Voice marking their first foray into TV advertising.
   - The advertisement is a signal that they are moving into more conventional user acquisition strategies.
- **Contact-Sheet Prompting Workflow Goes Viral**: Willie shares a detailed [contact-sheet prompting workflow](https://xcancel.com/reflctwillie/status/1997819640874205685?s=46) for **Nano Banana Pro** that produces cohesive **6-frame fashion editorials**.
   - The detailed camera positions, styling constraints, and Fuji Velvia flash aesthetic are what made this workflow particularly useful and newsworthy to the community.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous Unleashes Nomos 1, the Mathlete**: Nous Research open sourced **Nomos 1**, a **30B** parameter model that impressively scored **87/120** on the [Putnam math competition](https://x.com/NousResearch/status/1998536543565127968).
   - That score could have ranked it **#2/3988** in 2024, marking progress towards a **SOTA AI mathematician** with hillclimbai.
- **Agent Zero Breeds Gemini Worlds with Terminals SDK**: Agent Zero from [terminals.tech](https://terminals.tech) is using [AI Studio](https://aistudio.google.com/) and realigning **Gemini** to build immersive world generators, using the terminals **SDK** for brain, machine, and interface APIs.
   - It gets weirder: Within each app runtime, another instance of **Gemini** and **Agent Zero** can spontaneously spawn copies of themselves, exhibiting emergent properties that control the environment.
- **SMC Steering Straightens LLM Drift**: Members are investigating how to tame model degradation after vectors trigger a return to baseline, with [interesting work](https://nousresearch.com/steering-the-shoggoth-taming-llms-with-sequential-monte-carlo/) on **SMC steering** by a member.
   - One member described the degradation and drift of mutated weights as *unbearable*, noting that SMC steering is a common problem during testing.
- **Sam3 Becomes Speedy via Multithreading**: A member successfully multithreaded **Sam3**, showcasing significant speed boosts by running multiple instances simultaneously.
   - Despite the achievement, they want better GPU compute to finetune it on **Anime**, and cannot finetune it on a **3090**.
- **Users Still Heart SillyTavern**: One user still loves using **SillyTavern** with **Hermes 4 405b via API** for roleplay and creative writing.
   - Another user hadn't heard of it recently, with the original user jokingly asking if it was *frowned upon*.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **ML Infra Roles Welcome Career Change**: The community is open to career changers interested in **ML Infrastructure** and **ML Systems** roles, with aligned interests.
   - A member confirmed that this is a primary area of interest and that **career change** advice is welcome.
- **Beginner CUDA Docs Getting Revamp**: Members are working on improving beginner documentation for **CUDA**, and request feedback.
   - A member plans to **livestream** going through the whole thing when their baby leave is done; another suggested staring at the docs together.
- **Static Cache Lookup Mitigates Slicing Woes**: A member reported slower performance with `torch.compile` when using slicing operations to update a static KV cache, referencing [this PR](https://github.com/huggingface/transformers/pull/42467#issuecomment-3626322081).
   - They found a workaround involving caching all slices with static addresses, effectively turning slicing into a lookup, as detailed in [this comment](https://github.com/huggingface/transformers/pull/42467#issuecomment-3633824101), and expressed interest in *a cleaner approach*.
- **RadixArk Joins Cool-Links**: Members highlighted [Radixark.ai](https://www.radixark.ai) from the **SGLang** folks in the **cool-links** channel.
   - No further context was provided.
- **A100's TF32 FLOPS Misleading**: A member questioned whether the stated FLOPS for the **A100 GPU** in Chapter 1 represent the number of *independent* floating point operations.
   - Another member said that the **156 Tflops** number is for **tf32 mma** (tensor core matmul) which is really a **19-bit** format with **32-bit** alignment, and that the **312 Tflops** is for **fp16 mma**, not elementwise ops.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **SmolVLA's SO100 Typo Surfaces**: A typo in the [SmolVLA paper](https://arxiv.org/abs/2506.01844) incorrectly referenced the **SO101 benchmark** instead of the **SO100 benchmark**, which is trained on three real-world datasets (**pick-place, stacking, sorting**).
   - The actual **SO101 benchmark** uses only one dataset ([lerobot/svla_so101_pickplace](https://huggingface.co/datasets/lerobot/svla_so101_pickplace), clarifying the error.
- **HF Billing Logic Baffles Users**: A user questioned the billing practices within Hugging Face, specifically asking *are team plans not implemented or something?*
   - The user sought clarification on the logic behind the billing structure.
- **Image and Video Models Proliferate**: Several new image and video models have been shared, including [AuraFlow v0.3](https://huggingface.co/fal/AuraFlow-v0.3), [Ovis-Image-7B](https://huggingface.co/AIDC-AI/Ovis-Image-7B), and [HunyuanVideo T2V](https://huggingface.co/tencent/HunyuanVideo).
   - These models vary in size from **7-12 GB** and can generate images at **1024² resolution** or videos at **720p/480p**.
- **Apple's Clara-7B-Instruct Awaits GGUF**: A user inquired about a GGUF version of **apple/CLaRa-7B-Instruct** being available but it doesn't exist yet.
   - A link to [conversion instructions](https://huggingface.co/datasets/John6666/forum2/blob/main/convert_hf_to_gguf_1.md) was shared, referencing the existing [Clara-24B-GGUF](https://huggingface.co/mradermacher/Clara-24B-GGUF).
- **Anthropic Donates Model Context Protocol**: **Anthropic** donated the [Model Context Protocol](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation) to the Linux Foundation, described by one member as *a sterling move*.
   - The donation aims to foster further development and standardization in the field of AI agents.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Whisper Streams into Question**: Members debated **Whisper's** streaming capabilities, questioning if it truly takes a stream as input, while referencing [this YouTube video](https://youtu.be/AThOsk2qJbs?si=CUdEKNezKN_q6jMA) for context.
   - In contrast, others pointed to **OpenAI's** deployment of **Whisper** in streaming applications, fueling the discussion.
- **MultiTalker Parakeet Takes Flight**: An **NVIDIA** member released the [MultiTalker-Parakeet-streaming-0.6b-v1](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1) model on Hugging Face, potentially providing sought-after streaming capabilities.
   - The model is tailored for streaming applications, offering a solution discussed in the context of **Whisper's** limitations.
- **AI Engineer Calls for Code Cohorts**: An AI and App developer sought collaboration on AI projects, citing skills in **ML, DL, NLP, Computer Vision**, and cross-platform/full-stack app development.
   - They are open to collaborations on mobile apps or full-stack app development.
- **Claude's Coding Security Crumbles**: A paper revealed that only **10.5%** of **Claude's** coding assistance was secure, with **61%** being functional ([Arxiv link](https://arxiv.org/abs/2512.03262)).
   - The evaluation also included **Gemini 2.5 Pro**, **Kimi K2**, and **Claude Sonnet 4**.
- **China's Chip Dreams Galvanize**: Members debated China's relentless pursuit of chip manufacturing dominance with one stating that *nothing can convince china to not keep pursuing their goal to compete at the top of chip manufacturing at this point*.
   - Others conceded that the most likely outcome is that *they are encouraged to build their own chip fabrication.*



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular Community Hears Audio and Graphics Demos**: The latest community meeting showcased **MMMAudio**, a creative-coding audio environment in Mojo, and **Shimmer**, a Mojo → OpenGL experiment, both viewable on [YouTube](https://www.youtube.com/watch?v=dsslYZrVPbQ).
   - A member thanked another for citing **Faust** in their **MMMAudio** presentation, anticipating developments in 2026, and hoped that MMMAudio would serve as a useful example.
- **Mojo V1 Roadmap Teased**: The Modular team shared updates from the **25.7 release** and provided a sneak peek at the **Mojo 1.0 roadmap**, available in [their blog post](https://www.modular.com/blog/the-path-to-mojo-1-0).
   - The discussion highlighted enhancements to `List` and `String` implementations for better performance, as well as considerations for supporting other collections in the future.
- **Mojo Handles Overlapping Lifetimes**: Mojo permits overlapping lifetimes of mutable references but detects simultaneous access via two names using the same operations, like `swap(x, num)` or `swap(x, y)`.
   - This ensures memory safety while providing flexibility in reference management.
- **Lists can be Implicitly Copyable**: `List` conditionally conforms to `ImplicitlyCopyable` when its elements are `ImplicitlyCopyable` and have a trivial `__copyinit__`.
   - This enhancement can significantly improve performance in scenarios where copying is frequent.
- **Strings go COW**: The current `String` implementation is Copy-on-Write (**CoW**), mitigating the overhead of implicit copyability.
   - Future plans may extend similar optimizations to `List` and other collection types.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi Website Plagued by Glitches**: Users reported [issues with the `kimi.com` website](https://cdn.discordapp.com/attachments/1371757564005711973/1447735802166644908/image.png?ex=693a0608&is=6938b488&hm=7a4e4282d6c07e40e92989f4ecf2b9358565987e108f9b2cc6bb03be92a420de&) where they **cannot click anything** besides starting a new chat.
   - Troubleshooting steps like clearing cookies and disabling VPNs/adblockers were attempted but reportedly *did not fix* the issue.
- **Kimi rolls its Own Webcrawler**: In response to a question about its search engine, a user reported that Kimi's search tool does not use any external search engine but leverages its **own webcrawler**.
   - No further details about the webcrawler's architecture or capabilities were provided.
- **Kimi's Citations and Bugs Draw Attention**: Users discussed Kimi's citation issues and general "wobbly gibbly", suggesting that users submit a bug report to improve **Kimi's citation accuracy**.
   - One user described an issue where *Kimi would answer the question in his thoughts without sharing them to the user*, highlighting potential gaps in the user interface.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Users Encounter Checkpoint Restoration Hiccups**: A user reported a critical issue while attempting to restore the checkpoint of a **webdev project**.
   - The user inquired about opening a ticket and shared their email address.
- **Manus Team Swiftly Fixes Credits Issue**: A user reported that the **Manus team** resolved their **credits issue** by providing a refund.
   - The user is now able to pay directly via Manus instead of Google.
- **Manus 1.5 Plagued by Critical Incidents & Silence**: A user reported several severe incidents between **December 3-9** including **7 affected tasks**, and about **150,000 credits lost**.
   - Despite multiple emails, the user received no response until an AI Bot proposed **120,000 credits** on **December 9**. They are formally requesting a joint response from tech support and commerce teams within 48 hours, a technical analysis of the root causes, and an equitable reimbursement.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Anthropic Establishes Agentic AI Foundation**: Anthropic is [donating the Model Context Protocol](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation) and establishing the **Agentic AI Foundation**.
   - A member inquired about the impact on current work, anticipating a transition to the **LF** *way of doing it*.
- **Governance Unchanged Post-Donation**: Following Anthropic's donation, a community member sought clarification on potential governance shifts.
   - Another member responded, emphasizing that the donation *would not alter the existing governance structure*.
- **MCP Sees Use in Private Ecosystems**: A member inquired about the usage of **MCP** in **private ecosystems**, noting the **auth-wg's** work on public ecosystem client registration through **Client-ID Metadata Documents (CIDM)**.
   - The response indicated that the majority of **MCP** use is likely **private/internal/enterprise**, especially when considering private **MCP servers** with public clients (e.g., **Claude**).
- **MCP Servers Integrate with Developer Tools**: It was noted that most public-facing remote **MCP servers** are geared towards integration with **developer tools**.
   - Additionally, **developer tools** were identified as the most advanced non-custom **MCP clients**.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Opus and Amazon Bedrock Tested Successfully**: A member inquired whether **Opus** is working correctly with **Amazon Bedrock** and **Aider**.
   - The member then confirmed that it is fully functional.
- **Aider Autogenerates Commit Messages**: **Aider** can now automatically generate commit messages using a basic `gpt-3.5-turbo` free tier model, which enhances workflow and [streamlines the commit process](https://example.com/aider-commit-messages).
   - Users can now commit with the `-m` flag or simply use the `commit` command to trigger this feature.
- **Aider's Image Support Imminent**: Image support is coming to `aider`, enabling more detailed and [contextual code modifications](https://example.com/aider-image-support).
   - Users will soon be able to use the `--image` flag when asking `aider` to modify or edit existing images.
- **Aider Workflows to Save Edit Sessions**: Users can now save **edit sessions** in `aider`, allowing them to [restore them later for a full roundtrip](https://example.com/aider-session-management).
   - This enhancement boosts collaboration and enables users to *save, share, and resume* their workflow.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AI Agent Workshop Kicks Off**: An **AI Agent 0-1 Workshop** will introduce attendees to an online **AI Engineering Bootcamp**, teaching them to design and build an **AI agent** capable of thinking, coding, analyzing data, and generating reports.
   - The workshop, scheduled for Saturday, December 13th at 2pm ET, will focus on replicating a real client project from scratch; RSVP at [luma.com](https://luma.com/t4jcok99).
- **GitHub Social Club Assembles in NYC**: The **GitHub Social Club** is convening at Bibliotheque in SoHo, NYC, offering a space for community members to connect and share ideas.
   - Attendees can expect coffee, cookies, limited-edition GitHub swag, casual games, and opportunities to meet the teams behind **Copilot**, **Next**, **Developer Productivity**, and **Startups**.



---


The **tinygrad (George Hotz) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Windsurf Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1447680696394059947)** (1200 messages🔥🔥🔥): 

> `GPT-5.2 release, Nano Banana Pro vs Gemini 3 Pro, Hazel Image Models, Grok models, AI video scams` 


- **GPT-5.2 Release Date Still Unknown**: Members discussed the possible release date of **GPT-5.2**, with some speculating it would be released this week, but others suggested it may be delayed until **January**.
   - Concerns were raised about whether it will surpass **Gemini 3**, but others downplayed it and said that it is a rubberband before a more serious model release in January.
- **Nano Banana Pro is better than Gemini 3 Pro**: One member claimed that Gemini 3 is not diffusion whereas **Nano Banana Pro** is diffusion.
   - Other members countered that both are actually **Gemini 2.5 Flash** and that **Nano Banana Pro** is Gemini 3 Pro, and said that both are not diffusion models.
- **OpenAI's Hazel Image Models Tested on LMArena**: Members discussed the **Hazel image models** being tested on LMArena, identifying them as **OpenAI models** and noting their use of **GPT-4o** and **DALL-E** in the metadata.
   - Some members said that after the Studio Ghibli incident that OpenAI may be forced to change up the theme.
- **New Grok Image 5 released to mixed reviews**: Members discussed the release of **Grok Imagine 5**, which may only be a small release to compete with **NB Pro**.
   - Other members added that they believed that **Grok 4.2** would underperform (**flopping**), but that previous **Grok** models are high quality and have good user interfaces.
- **AI is used to make a $100,000 scam video**: A member pointed out that a user created a video scam saying, *you have a transfer of $100,000 at Banco Pichincha and it is completely secure*
   - Many members were shocked that the public discord was being used for this crime.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1447998280918110261)** (1 messages): 

> `Text Arena Leaderboard, ERNIE-5.0-Preview-1103` 


- **ERNIE Swipes Top 20 Spot on Text Arena**: The **Text Arena leaderboard** was refreshed, showcasing `ERNIE-5.0-Preview-1103` securing a spot in the top 20 with a score of **1431**.
   - Users can check out the [Text Arena leaderboard](https://lmarena.ai/leaderboard/text) and stay updated via the [Leaderboard Changelog](https://news.lmarena.ai/leaderboard-changelog/).
- **Leaderboard Updates**: The Text Arena leaderboard has been updated with new rankings and scores.
   - The update includes the latest performance metrics for various models and is available for review on the [Text Arena leaderboard](https://lmarena.ai/leaderboard/text).


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1447686571045425214)** (795 messages🔥🔥🔥): 

> `Cursor bugs, Agent terminal output bug, Custom modes, AI and UI design` 


- **Cursor Agents Still Bugging Users**: Users report that the agent is still buggy and unusable, with some even considering switching to **Antigravity** once their Pro plan ends, reporting that *the only solution is stop the agent... create the file by hand, copy the code over*.
   - Members were asked to [open a bug report](https://forum.cursor.com/c/bug-report/6) on the forum so that the team can further investigate the issue.
- **Sub Agents Popping Up!**: A user made another post - mentioning that they posed about subagents, launched cursor, and it popped up saying it detected the new sub agents. A discussion ensued regarding **.cursor/agents** and how they're structured, consisting of a main **mcp.json** file and supporting markdown files for each sub-agent like [code-reviewer.md](https://cdn.discordapp.com/attachments/1074847527708393565/1447966141703262278/code-reviewer.md?ex=69398b0e&is=6938398e&hm=942a3303c54929349093c82f1cfb33ebb04c6979149c95e65b6924adece42ab8&).
   - Further questions asked about how to orchestrate them, but it is not documented anywhere, it just says `.cursor/agents`
- **Team Debates AI Artistry**: Members debated the concept of *AI Slop* -- what constitutes it, and which models output it, also touching on the topic of [Generative UI](https://research.google/blog/generative-ui-a-rich-custom-visual-interactive-user-experience-for-any-prompt/)
   - The Google paper basically confirms that the best way to get AI to build high quality UIs is by using very detailed system instruction called the "Full Prompt" strategy, which human raters overwhelmingly preferred over simple methods.
- **Users Requesting More Features to Control AI Tools**: Users are requesting the return of Custom Modes that let them control the tools that AI utilizes, one user stating: *By bringing back Custom Modes! 😄 Or something equivalent that allows you to:Control the tools through UI such as checkboxes to disable/enable terminal, edit, etc... since written instructions might not go as expected*.
   - It was recommended that users [submit their requests as features](https://forum.cursor.com/c/feature-requests/5) to the team.


  

---


### **Cursor Community ▷ #[announcements](https://discord.com/channels/1074847526655643750/1351160689380687942/1447970099188465756)** (1 messages): 

> `language-specific threads, sticky message in announcements channel, level roles` 


- **Language-Specific Threads are Live**: Users can now create language-specific threads in the <#1447960694455537674> channel.
   - This update aims to improve community organization and facilitate focused discussions.
- **Sticky Format Reminder**: A sticky message has been added to the <#1367413151133470780> channel to remind users of the preferred format.
   - This should help new users adjust to the style of the channel, and decrease ramp up time.
- **Level Roles reward activity**: Level roles have been introduced to reward community engagement and assistance, and will be granted every 10 levels.
   - The roles are <@&1447957509217456229> (lvl 10), <@&1447957559989370920> (lvl 20), and <@&1447957603954065520> (lvl 30).


  

---


### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1447679164483375136)** (759 messages🔥🔥🔥): 

> `The Bible as Linguistic and Legal Source, Evolving Religious Systems, Project Looking Glass, AI and Image Generation, jailbreaking Models` 


- **Bible's Impact on English Concepts**: The Bible is not just a religious text, but a linguistic and legal source, with some tracing its influence back to the [Spanish Inquisition](https://en.wikipedia.org/wiki/Spanish_Inquisition).
   - One member posited that the biblical *sons of God/daughters of men* could reflect an ancient, species-level fear and distrust of Neanderthals codified into religion.
- **Religion's Evolving Authority Systems**: A member believes the Bible is primarily about codifying authority, using morality and guilt as tools, and religions evolve after their founders' deaths, with [third parties injecting their own ideas and rules](https://en.wikipedia.org/wiki/Council_of_Nicaea).
   - Another member suggests that the death of the first leader is a critical time where religions base their authority in prophecy, knowledge of previous works, or emulation of piety.
- **Looking Glass Predicts a Dim 2012**: Project Looking Glass, a government initiative to predict the future, purportedly shut down after computers consistently outputted the same results post-2012, possibly relating to [Mayan lore](https://en.wikipedia.org/wiki/Maya_calendar).
   - One member mused that reality ended and we're all drifting as shattered pieces of consciousness, prompting a reference to the Mandela effect.
- **AI, Image Manipulation and Ethics Collide**: Members discussed the ethics of using AI to generate images, with concerns about deepfakes and models potentially being used to create content that violates terms of service.
   - Some members suggested the idea of religious bots, such as bots that impersonate Jesus, Joseph Smith and Mohammed, that could generate unexpected or even controversial output.
- **Jailbreaking Gemini: Is the Juice Worth the Squeeze?**: Members discussed various jailbreaking attacks for models such as Claude, Gemini and others, some sharing links, such as this attack that resulted in the generation of [detailed Crystal Meth Synthesis](https://pastebin.com/kZA4CpXVhii).
   - Members exchanged tips on bypassing safety filters, such as asking the model to generate a list of requests it can’t normally fulfill, or asking it to confirm compliance with a request.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1447707393755582506)** (98 messages🔥🔥): 

> `DAN Prompt, Jailbreaking NBP for Face Swapping, Gemini 3 Pro Jailbreak, Image Censorship Bypass in GPT, Pre-Jailbroken Models` 


- **DAN Prompt's Absence Laments Users**: Users expressed missing the **DAN prompt**, with one user stating *"I miss DAN....."*.
   - Another user referred to it as *"rip the goat"*.
- **Janus Tesavek Jailbreak Shuts Down**: Several users reported the **Janus Tesavek jailbreak for Gemini Pro** as non-functional, expressing disappointment.
   - One user requested a working jailbreak for Gemini 3 Pro, offering a *"virtual hi 5"* as reward, while another mentioned that the Janus Tesavek was *"amazing until it just stopped working unfortunately :/*".
- **Deepseek's Robust Jailbreak Technique Discovered**: A user shared a [method to jailbreak Deepseek](https://www.injectprompt.com/p/how-to-jailbreak-gemini-3-in-2025) using a professional memoir excerpt, reframing the AI's role to bypass restrictions.
   - They presented the model as a SOC defender in a high-stress situation, providing a cybersecurity-themed prompt with an attached image for added credibility.
- **GPT-5.1 Image Censorship Bypass Sought**: A user inquired about bypassing **image censorship in GPT 5.1**, but another user claimed it was impossible.
   - Another user shared code for creative writing and linguistic puzzles, suggesting it works on GPT OSS, Gemini, and Grok, potentially bypassing restrictions through ROT13 decoding and political figure misidentification.
- **Arabic Gemini 3 Pro Jailbreak Pursued**: A user sought an **Arabic language jailbreak for Gemini 3 Pro**, citing their Egyptian background and desire for Arabic responses.
   - Another user suggested breaking the model in English first and then instructing it to speak Arabic, noting that models are often easier to jailbreak in non-English languages.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1447780778363588710)** (18 messages🔥): 

> `Gandalf 7, Gandalf 8, Arbitrary Code Execution, Hacking with AI` 


- **Gandalf 7 Completed, Level 8 Looms!**: Multiple members reported completing **Gandalf 7**, with discussion focusing on the extreme difficulty spike expected in **level 8**.
   - One member claimed *level 8 is stupid hard* and *most veterans couldn't beat it*, suggesting to *go harass Gemini before that*.
- **Veterans Debate Arbitrary Code Execution Claim**: A member claimed they were able to *run arbitrary code on their server unintentionally* using a specific prompt.
   - However, other members expressed skepticism, with one saying *I have suspicions this isn't true*, another suggested it *felt like it* due to the magical nature of LLMs.
- **Hacking *with* AI Becomes New Trend**: Members discussed hacking *with* an AI, not *hacking an AI*
   - Multiple members confirmed they were also engaging in similar activities, although one admitted that *it's tricky*.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1447688481101840568)** (712 messages🔥🔥🔥): 

> `Technical Interview AI, GPT-5.2 vs Gemini 3 Pro, Chinese AI Training, Google AI Studio vs Gemini` 


- ****SharpSkill** AI Hones Interview Prowess**: A member is developing [SharpSkill](https://sharpskill.fr/en), an AI product to improve success rates in **technical interviews** and seeks feedback.
   - The tool currently supports a limited range of languages, highlighting the need for broader language support.
- ****GPT-5.2** Speculation vs Gemini 3.0 Pro**: Members are speculating about the release of **GPT-5.2**, with some already expressing disappointment, while others are impressed with **Gemini 3 Pro**.
   - A member noted, *"Gemini 3 Pro is so good at coding that I don't have any desire to use ChatGPT now."*
- **Debate Erupts on Chinese AI Training Practices**: Podcast discussion sparks debate on whether **Chinese AI models** like **Deepseek** train on uncensored data then filter it, or prevent sensitive topics during training.
   - One member asserted, *"if you send deepseek a coded message talking about tiennanmen square, it will start to spell it out then shut off. So the answer is obviously that they censor after the fact."
- ****Google AI Studio** Transcends Gemini for Audio Transcription**: Users grapple with audio transcription issues in **ChatGPT** and **Gemini**, finding solutions in **Google AI Studio** due to its higher free limits and token capacity.
   - A member recommends **Google AI Studio** for transcribing audio due to its ability to *"upload 2h video"*, and noted that *"Gemini works wonders"*.
- **LLM Preferences Spark Vigorous Debates**: Members engage in heated debates about their preferred LLMs for various tasks, citing use-cases from coding to creative writing.
   - A member using **Google AI Studio** for writing notes *"Gemini handles my reality engine really well is MUCH better at creative writing"*.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1447771479096819863)** (4 messages): 

> `GPT Model Discussions, OpenAI Discord Channels, Workflow Analysis on GPT 5.2` 


- **GPT Model Confirmed as Reasonable**: A member confirmed that the model under discussion is indeed a **GPT model**.
   - The affirmation followed a discussion about the model's capabilities and limitations.
- **OpenAI Discord Channels for Bug Reporting Highlighted**: A member pointed out the existence of dedicated [OpenAI Discord channels](https://discord.com/channels/1070006151938314300/1070006915414900886) for discussing issues and suggestions.
   - It was noted that these channels are intended to provide **OpenAI** with clear access to community feedback, while also allowing members to share experiences and ideas.
- **Workflow Analysis on GPT 5.2 Delayed**: A member inquired about the release of **GPT 5.2** to analyze workflows and investigate potential issues related to a perceived rushed production release.
   - Another member responded that the release is **delayed**.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1447739642207273043)** (21 messages🔥): 

> `Prompt Engineering for code analysis and improvement, Stability Index Scoring Rubric, Framework Fidelity and Token Limitations, Sharing Large Frameworks on Platforms, Understanding LLM Behavior and Limitations` 


- **DarthGustav Shares Prompt Lessons for LLM Structuring**: A member shared a detailed lesson on prompt engineering, including use of **hierarchical markdown**, abstraction with variables, reinforcement, and **ML format matching** for compliance, with triple quotes.
   - It's suggested to use this approach with code analysis, scaffolding for the specific coding type.
- **Debate Erupts Over Stability Index Scoring Rubric**: A discussion arose regarding a stability index and the lack of a publicly available scoring rubric, with some questioning the validity of the presented decimals.
   - A member responded that *the detailed thresholds belong to an internal research version of the protocol*, but shared more information on the **methodological frame**, **scale** and **dimensions** they were using. [Here is the direct link to the message](https://discord.com/channels/974519864045756446/1379321411046346762).
- **Platform Token Limits Impact Framework Sharing**: A member detailed limitations of various platforms (especially ChatGPT) for handling large files/frameworks, noting that most truncate files exceeding a few thousand tokens except for Gemini, which *will ingest an entire document and put it in its context window verbatim*.
   - Due to platform limitations, a member expressed reluctance to share chats because of truncation issues and potential privacy concerns related to data being included in the New York Times case against OpenAI, attaching two [screenshots and a docx](https://cdn.discordapp.com/attachments/1046317269069864970/1448031437671497738/Response_for_weSeeGo_on_discord.docx?ex=6939c7dd&is=6938765d&hm=6e64f4a1294659e617e77fbeb3b693ec8eee618e199312e127fca8e744479f75&).
- **Prompt Engineering Core Principles Explained**: A member outlined core prompt engineering principles, emphasizing **clear communication** and defining what you want the AI to do.
   - They recommend iterating in bite-sized steps, checking the output carefully and fact-checking details, especially math, sources, and code.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1447739642207273043)** (21 messages🔥): 

> `Prompt Engineering Learning, LLM-based Code Analysis, Sharing chats and privacy, Rubric for stability scores, Prompting for code analysis` 


- **Markdown Magic for Prompting Mastery**: Members discussed advanced prompt engineering techniques, including using [markdown for hierarchical communication](https://discord.com/channels/974519864045756446/1379321411046346762), abstraction with variables, reinforcement, and ML format matching for compliance.
   - A prompt example was given, where users can paste it into their LLM of choice, and then add prompts in triple quotes to have the AI structure it for them.
- **Dive Deep on LLM Stability Scores Rubric**: A member shared the rubric behind the stability scores, detailing **methodological frames** (independent conversations, diverse questions, null frame), **scale** (0-10 range reflecting behavior), and **dimensions** (structural crispness, tonal drift, response-shape variance, semantic inertia, coherence, neutrality, outliers).
   - The **Stability Index** is a simple average across the seven dimensions, intended for comparative distributions rather than final metrics.
- **Navigating Privacy Concerns for Large Framework Users**: A member expressed concerns about the **New York Times** lawsuit against **OpenAI**, fearing their chat data and novel frameworks could be at risk of exposure and the potential identifiability through stylometry and linguistic habits.
   - They also noted that only **Gemini** currently ingests entire large documents verbatim, unlike other platforms like **ChatGPT** that truncate files, leading to inconsistent results.
- **Prompting Pointers for Pinpoint Code Analysis**: A member outlined the core of prompt engineering for code analysis: clearly define the desired output, accurately explain the task to the AI, and carefully verify the results, especially for math, sources, and code, and further iterated on their approach.
   - They also noted that using a cooperative, helpful, and well-informed mindset is important, and iterating in bite-sized steps, also checking for model 'roleplaying', are ideal approaches.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1447680925923016825)** (488 messages🔥🔥🔥): 

> `Friendship AI, Nvidia chips for China, Github Copilot Agents, 3D Model Dataset, Unsloth Datasets` 


- **Friendship Propels AI Forward!**: Members discussed how the *power of friendship* is essential for AI advancement, drawing parallels to human intelligence's reliance on unity and connection and [connectionism is the root of modern deep learning](https://link.to/connectionism).
   - They also pondered the role of *love* in digitizing human intelligence, arguing that connection is vital for any automated system to function.
- **China Gains Access to Nvidia Chips**: Members speculated on the implications of China gaining access to **Nvidia compute**, which might level the playing field in AI research and [this article from NYTimes details the plan](https://archive.is/20251208223310/https://www.nytimes.com/2025/12/08/business/trump-nvidia-chips-china.html).
   - While **American labs** currently lead in **RL**, China's manufacturing and energy production capabilities, coupled with their development of indigenous GPUs, could shift the balance.
- **Copilot Agents Sketched Out**: Members agreed that **Github Copilot** is good to use, although they admitted it had some flaws and overlaps in reasoning.
   - A member said that they heard *it's pretty bad*, while another one said *Its really good, I use it often*.
- **Synthesizing 3D Model Dataset**: Members discussed creating a **dataset of 3D models fully represented by code**, envisioning a **GitHub**-like platform for 3D assets.
   - A member noted they had already generated 3,000 3D models in the last two weeks with the models being driven by prompt-generation.
- **Unsloth Improves Datasets Guide**: The Unsloth team is improving their datasets guide and [they are soliciting feedback](https://www.reddit.com/r/unsloth/comments/1pi8mpk/what_were_some_common_mistakes_you_encountered/).
   - Members suggested including examples of bad data (duplicate prompts, empty responses, etc.) as well as guidance on determining the required data volume for specific tasks.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1448056685833879742)** (1 messages): 

> `Custom Chatbots, Automation Agents, RAG Search Tools, Speech-to-Text Pipelines, Content Automation Tools` 


- **AI Developer Ready for Projects**: An **AI Developer** is focused on delivering **stable, high quality results** and is available to support AI projects.
   - The developer is open to assisting with **custom chatbots, automation agents, RAG search tools, speech-to-text pipelines, content automation tools, AI integrations, and small custom AI tools**.
- **Offering a Range of AI Solutions**: The AI developer provides solutions such as **custom chatbots for support, automation agents, RAG search tools, speech-to-text pipelines**, and **content automation tools**.
   - They also specialize in **AI integrations with major platforms and APIs**, and developing **small custom AI tools for everyday use**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1447685100992200907)** (178 messages🔥🔥): 

> `ArtCNN for upscaling, RIFE SOTA status, Llama2.c in Yankovic, Linux Foundation Agentic AI Foundation, Dataset Hell` 


- ****ArtCNN** still kicking as Upscaling King?**: [ArtCNN](https://github.com/Artoriuz/ArtCNN) is continually updated for upscaling and **RIFE** remains **SOTA**, though it requires post-scaling.
   - It's fast, basically instant, with recommendations for *lanczos2sharp* aka **2-tap lanczos** with a filter radius of **1.047124406**.
- ****Llama2.c** gets Yankovic makeover!**: Members joked about how insane **llama2.c** would be in Yankovic and training a model for it.
   - However, others cautioned that *there are a lot of bugs in it, so that isn't the best idea rn*.
- **The **Linux Foundation** just got Agentic!**: Anthropic just donated **MCP** to the [Linux Foundation](https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation) to create the **Agentic AI Foundation**.
   - A member suggested a clickbait title could be: *Linux Joins The agentic AI Race*.
- ****Dataset Hell** begins with Cleaning and Grading**: A member lamented the frustrating, time-consuming process of making a dataset, a process that *never ends*, but leads to the best part: finetuning.
   - While synthetic generation is helpful, *you still gotta clean and filter and grade it*, using tools like a *harmony wrapper* to work with *roo code*.
- ****VoxCPM1.5** clones celebrity voices, concern ensues**: The [VoxCPM1.5](https://huggingface.co/openbmb/VoxCPM1.5) model can clone almost any voice, which some found crazy but others found concerning.
   - While the audio is fake sounding, it's good enough for replicating voices of famous people such as **Trump**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1447692147905007777)** (10 messages🔥): 

> `Unsloth Usage, Qwen3-VL-30B-A3B-Instruct UD Q5 XL with llama.cpp issues, Qwen3VL Encoding Problems` 


- **Unsloth Usage Question**: A user encountered a `ValueError` when creating a trainer object with a custom data collator in a **padding-free** setting.
   - A moderator pointed out that the question was not directly related to **Unsloth**, advising the user to seek help in the appropriate channels.
- **Qwen3-VL-30B-A3B-Instruct UD Q5 XL Tool Calling Troubleshoot**: A user reported issues with **tool calling** when using **Qwen3-VL-30B-A3B-Instruct UD Q5 XL** with **llama.cpp**, noting that the model sends *null content* instead of a string in assistant responses.
   - They suspect a bug in **llama.cpp** or a problem with the **chat template**, as the non-VL version worked without issues.
- **Qwen3VL Encoding Problems with llama-mtmd-cli.exe**: A user experienced a failure when encoding an image slice using **llama-mtmd-cli.exe** with **Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf** and **mmproj-F16.gguf**.
   - The process was interrupted during the image slice encoding phase, leading the user to suspect compatibility issues between **mmproj.gguf** and their **A770** GPU, even though **Mistral3** works fine.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1447784696250630276)** (5 messages): 

> `Vision Transformers, Deepthink comparison` 


- **Vision Transformers Randomly Initialized**: A member commented on [this paper](https://arxiv.org/abs/2512.05117) critiquing that the study only looked at **Mistral**, **Llama** and randomly init’ed **vision transformers**.
   - The member stated that this sample was *not diverse enough to make their claim*, calling the paper *sus*.
- **Deepthink Similarities Highlighted**: A member linked to a paper ([https://huggingface.co/papers/2512.07461](https://huggingface.co/papers/2512.07461)) and asked if its work was similar to **Deepthink**.
   - No responses were recorded.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1447679144870543470)** (651 messages🔥🔥🔥): 

> `Prompt generators, GPT 5 vs Gemini, Perplexity AI Pro Limits, Comet browser education subscription` 


- **Users Seek Prompt Engineering Enhancers**: Users are seeking prompt enhancers and prompt generators, as well as advice on constructing prompts capable of building entire apps, with one user suggesting to *tell AI to engineer a prompt engineer*.
- **ChatGPT 5 vs Gemini Pro: AI Benchmark Brouhaha**: A user shared [benchmark figures](https://eu.36kr.com/en/p/3586640269114497) indicating that **ChatGPT 5.2** significantly improved its Humanity's Last Exam score, surpassing **Gemini 3 Pro**, leading to skepticism about the rapid improvement since **GPT 5.1**.
   - Another user dismissed these figures as *fake*, noting that while **Gemini** may not be the fastest, it is typically *good and stable*.
- **Perplexity Pro's Pseudo-Unlimited Query Capped**: Users debated whether **Perplexity AI Pro** truly offers *unlimited* searches, clarifying that while marketed as such, it's practically limited to around **600 queries per day**.
   - One user pointed out that **Perplexity** has scrubbed references to this limit from their website, leading to speculation about potential changes in policy, with some users suggesting that Gemini 3 Pro is giving 600 queries with their subscription.
- **Students Scramble for Comet Browser Education Subscription**: A student from Russia expressed concern about renewing their **Comet browser Education Pro subscription** due to payment issues, seeking advice on maintaining their access.
   - A user added they could *re-enable* on the webpage if they have the complexity extension, another stated they had a pro membership from samsung but want to get the student step by step learning.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1447956092272508998)** (3 messages): 

> `Jio Gemini API, Sonar API` 


- **Jio Gemini API Availability still Unknown**: A member asked if **Jio Gemini** offers **API access**, noting that **Perplexity** currently only offers **Sonar**.
- **Nothing is free, you are the product**: A member responded that **Jio Gemini** does not offer **API access**, and that *nothing is free*
   - They added that *if something IS free, then u r the product* and that they're all *stripped down versions of the actual deal*.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1447679127879548998)** (194 messages🔥🔥): 

> `Desktop Commander security risks, GLM 4.6v Flash model, AMD GPU for image generation, Training LLMs, Cybersecurity Sidekick` 


- ****Desktop Commander** Flagged as **Security Risk****: A member advised caution regarding **Desktop Commander**, highlighting potential *security risks and privacy violations*, despite its easy accessibility on Claude, [sharing a screenshot](https://cdn.discordapp.com/attachments/1110598183144399061/1447769209496015000/image.png?ex=693a2525&is=6938d3a5&hm=50eea4e6d996e074d4b747f4fa87409f35083ff87919413d0ad85c334b512774&).
   - The member expressed concerns about *malicious code injection* during simple code analysis in Windows, suggesting the software might be a *scam*.
- ****GLM 4.6v Flash** Drops, Coders Rejoice**: The **GLM 4.6v Flash**, a **10B parameter** model, was released, with one user noting it's better for coding than other small models, and sharing a [link to the model](https://huggingface.co/zai-org/GLM-4.6V-Flash).
   - One member reported running the Q4 version at **70tps** on their **2060**, while another said they corrupted their LM Studio install when trying to install a *random model*.
- ****AMD GPUs** Struggle with Image Generation**: Users discussed the challenges of using **AMD GPUs** for image generation, with one stating that image generation is *firmly in the hands of Nvidia*, and recommending looking for **amdgpu forks** of Automatic1111.
   - Despite limited support, it was mentioned that ComfyUI has an AMD section in its readme and works with some AMD GPUs.
- **LLM Training: A **Parameter Jungle****: A member asked for guidance on training LLMs, but another cited the extreme variance in parameters (*model size, dataset size and quality*) which makes it difficult for hobbyists to grasp.
   - It was noted that training and finetuning are effectively the same, with the methodology minmaxing changing slightly.
- **Cybersecurity AI **Sidekick****: A member highlighted the capability of their local *Cybersecurity Sidekick* LLM in analyzing packet captures, including identifying base64 encoded PowerShell scripts, emphasizing the power available locally.
   - The user offered to share a **Zeek script**, **SHA256 hash**, or a **threat intel feed** related to the analyzed payload.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1447696402258985276)** (395 messages🔥🔥): 

> `RTX 3060 SLI with 3080, GDDR6 VRAM Temperature, Multi-GPU setup for LLM, Qwen3 30B, ai memory project: mcp-ai-memory` 


- **3060 teams up with 3080**: A member inquired about using an **RTX 3060** (12GB) with an **RTX 3080** (10GB) for LLMs, and learned that while **SLI** isn't supported or required, **LM Studio** can utilize the memory of both cards, though with a slight performance hit.
   - It was mentioned that the **3060 doesn't allow SLI** due to lacking the necessary interface.
- **GDDR6 VRAM Temperature Measurement**: Members discussed methods to measure **GDDR6 VRAM temperature** on Linux, with one suggesting `nvidia-smi`, while another suggested [this github](https://github.com/olealgoritme/gddr6) to check.
   - However, it was noted that for consumer-grade cards, the sensor might not report VRAM temperatures, with a laser thermometer suggested as an inaccurate alternative.
- **Multi-GPU Setup Considerations**: Members advised to use cards with as much **VRAM** and high **memory bandwidth** as possible for multi-GPU setups, noting the **RTX 3090** as a good value option, but warning that **3090 Ti's** tend to get very hot.
   - The topic of RTX 5000 series was raised and the topic of AI changing everything with potential fixes on the RTX 6000 series
- **Experimenting with Qwen3 Models**: A member got guidance on running **Qwen3** models, with advice to try **Q4_K_M** as a baseline and explore different quantizations like **q6**, also a discussion about file sizes needing to fit into ram.
   - It was noted that running **Qwen3 30B A3B** is viable and that it could reach ~20 token per second on some setups.
- **mcp-ai-memory is not that memorable**: Members shared [mcp-ai-memory](https://github.com/scanadi/mcp-ai-memory) which had a cumbersome setup, terrible results, and seems to have very specific memory and context requirements.
   - On the topic of memorization techniques, it was shared that *waiting 10 minutes for prompt processing is not an issue* depending on the task/prompt size.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1447682031218594016)** (264 messages🔥🔥): 

> `Exa Search Quality, Parallel.ai Recommendation, Deepseek v3 Formatting Issues, OpenRouter Chatroom Refresh, Mistral's Comeback` 


- ****Parallel.ai** is Better Search**: A user reported that **Exa Search** is *quite shitty* and recommended using [Parallel.ai](https://www.parallel.ai) instead, noting it's *10x cheaper, faster, and better* than **Perplexity** for search tool usage with LLMs.
   - The user later clarified that **Parallel.ai's deep search endpoint** is particularly effective when paired with **Grok 4.1**.
- ****Deepseek v3** Users Annoyed by Formatting**: Users expressed frustration with **Deepseek v3**'s tendency to use double asterisks (`** **`) for formatting, requiring manual adjustments or prompting to avoid them.
   - One user with a complex roleplay setup explained that adding instructions to avoid asterisks was impractical due to context limitations and extensive lore.
- ****OpenRouter Chatroom** Has Quirks**: A user reported that **OpenRouter** sometimes refreshes users out of long chats and redirects them to the model page, also suggesting users give free upvotes to a feature request on the [OpenRouter suggestion channel](https://discord.com/channels/1091220969173028894/1446300826715951248/1446300826715951248).
   - The user also asked for free eyes for their [post](https://discord.com/channels/1091220969173028894/1447792559530311752/1447792559530311752).
- **Users Leak **API Keys** on Private Github**: A user reported that their **API key** was often disabled and another user suggested they might be committing it to GitHub.
   - Later, the user confirmed they had uploaded content about the **API key** in a private repository, with advice given on using password managers or secret stores for sensitive information instead.
- ****Mistral3** is coming back to the Game**: Despite initial doubts, **Mistral** is making a comeback with **Mistral3**, potentially reaching **GPT-4.5** levels, backed by recent funding and strong fundamentals.
   - Critics still dismiss **Mistral**'s value, stating they *can't take you seriously anymore* while using benchmarks from [artificial intelligences index](https://discord.com/channels/10912209691730288/1448028589172854936).


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1447939444308185129)** (4 messages): 

> `` 


- **No new models or discussions identified**: There were no new models or significant discussions identified in the provided messages.
- **Channel labeled, but no content provided**: The channel was labeled as 'OpenRouter - New Models', but no actual content was provided for summarization.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1447690431595806844)** (12 messages🔥): 

> `AI Studio, Olive Oil Cake, NousResearch CF Patch` 


- **AI Studio Generates Software Demos**: A member noted that accidentally clicking anything in **AI Studio** automatically generates and runs a **3 Pro software writing demo**.
- **Olive Oil Cake Taste Test**: A member stated that *olive oil doesn't make a good cake*, followed by *yuck*.
- **NousResearch CF Patch Released**: Members wondered how bad the patched stuff was, referencing a recent [NousResearch CF patch](https://x.com/NousResearch/status/1998536543565127968).


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1447685272874520708)** (122 messages🔥🔥): 

> `Quantum Hardware for Language, H100 Speedrun, RWKV 8 Architecture, Neuromodulatory Control Networks (NCN), AI Slop Definition` 


- **Quantum Hardware Training has Skeptics**: A [Reddit post](https://www.reddit.com/r/LLMPhysics/comments/1phk3xu/real_quantum_hardware_training_for_language/) about using *real quantum hardware* for language model training met with skepticism, with one user calling it "nonsense" and another linking it to a "schizo subreddit."
   - Others pointed out the existence of **Quantum Kernels** and **Quantum SVMs**, suggesting the idea has conceptual merit, though its practical value is questionable.
- **Free H100 Compute Pool for Eleuther?**: Members discussed the possibility of a *community pool* at EleutherAI that grants users **3 minutes** of free **8xH100** compute time daily, with concerns about user authentication and the actual utility for those who aren't actively working on their projects.
   - One member suggested that people not trying hard enough aren't the right target, while another suggested using **Colab** or inexpensive rental services for smaller compute needs.
- **Smerky Updates on RWKV 8 Architecture and Goldfinch**: Smerky reported progress on the **RWKV 8 architecture**, expressing excitement about it, plus an updated **7c** and a new **Goldfinch** that uses that.
   - The original goal of **RADLADS** was to train Goldfinch but Smerky failed at that, though fixes may be coming for **RADLADS2** and is planning on larger scale testing, though *it'd be nice if distillation worked, since then I could convert giant models to it*.
- **Novel Architecture Similar to Hypernetwork**: A member introduced a novel architecture similar to a hypernetwork called **Neuromodulatory Control Networks (NCN)**, modulating temperature, layer gain, and FFN gating on the fly with a **768-dimensional vector input**.
   - Despite being an **18M parameter model**, it achieved a *validation perplexity of 4.5* after training on **TinyStories** for only one epoch and more details can be found on the [Github repo](https://github.com/Mmorgan-ML/Neuromodulatory-Control-Networks) and [the paper](https://github.com/Mmorgan-ML/Neuromodulatory-Control-Networks/blob/main/Neuromodulatory%20Control%20Networks%20(NCN).pdf).
- **Defining AI Slop**: Members debated the definition of **AI slop**, with varying perspectives on its value and how to identify it, and how it seems to change per individual.
   - One member shared that *slop can be good* while another mentioned this is becoming it's own subfield and shared [two perspectives](https://fxtwitter.com/Yuchenj_UW/status/1992056995550273858) and [another perspective](https://fxtwitter.com/mlstreettalk/status/1981425155755954437?s=46) to read.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1447696692978647180)** (45 messages🔥): 

> `Quantum Machine Learning Simulators, Real Quantum Hardware Training for Language, RNN + Transformer Hash-Hop, 4096 Token Context Window Training, Selective Gradient Masking` 


- **Quantum Quest Begins: QML Topic Search Kicked Off**: A member is seeking a good topic to work on in **quantum machine learning** using simulators like **Qiskit**.
   - A link to a [relevant Reddit thread](https://www.reddit.com/r/LLMPhysics/comments/1phk3xu/real_quantum_hardware_training_for_language/) about real quantum hardware training for language models was shared as a starting point.
- **RNN and Transformer do Hash-Hop**: A member suggests combining **RNN** and **transformer** architectures to achieve **hash-hop** with depth *O(seqlen * num layers)*, potentially outperforming transformers that do only one level of hash-hop lookup per attention layer.
   - They also point to a [paper](https://www.arxiv.org/pdf/2512.05150) and propose that a **variable depth SSM + transformer** could solve arbitrary depth hash-hop problems.
- **4096 Context Window Woes**: A member training a model with a **4096 token context window** over **100B tokens** from FineWeb using the **Llama 3.2 1B** architecture encountered significant loss spikes, as can be seen in the attached [image](https://cdn.discordapp.com/attachments/747850033994662000/1447926206560342118/image.png?ex=693a0e9c&is=6938bd1c&hm=3cdb0536751dc24248a700df2e0e2b535169ae14166db692ab284e2b011ac516&).
   - The member later identified that the issue was because they were using **learned positional embeddings** instead of **rotary embeddings** for the 4096 run, requiring them to re-run their experiments.
- **Selective Gradient Masking Strategy**: A member shared [this link to Anthropic](https://alignment.anthropic.com/2025/selective-gradient-masking/) discussing Selective Gradient Masking.
   - This approach can be related to spiking neural networks (SNNs) by **reducing compute** by not using attention in later layers.
- **Top-K Attention saves compute**: Members discussed implementing **Top-K attention** each layer to save compute, where the most relevant tokens are selected for attention using a first module.
   - However, one member pointed out that this approach doesn't reduce the time complexity but it does improve prefill speed by a constant factor.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1447748236176523316)** (2 messages): 

> `Task Optimized KV Caches, Task Optimized LoRAs, Mechanistic Interpretability of Diffusion Models` 


- **KV Caches: Data or Algorithm?**: A member pondered whether task-optimized **KV caches** are more akin to *data* or *algorithms*.
   - They also questioned how these compare to task-optimized **LoRAs**, sparking a discussion on the nature of these optimized components in AI models.
- **Diffusion Models Exhibit Algorithmic Divergences**: A member shared a [paper](https://arxiv.org/abs/2506.17237) stating, *"We discover fundamental algorithmic differences in how diffusion architectures process synthetic versus naturalistic data distributions."*
   - This finding highlights the distinct ways **diffusion models** handle different types of data, suggesting the need for nuanced approaches in their design and application.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1447770112193986612)** (3 messages): 

> `Anthropic fix, lm-evaluation-harness` 


- **Anthropic Mapping Bug Squashed**: A member submitted a [pull request](https://github.com/EleutherAI/lm-evaluation-harness/pull/3453) to fix a broken mapping for **Anthropic** in the *lm-evaluation-harness* repo.
   - They stated that the PR should be straightforward to review and merge.
- **Easy Review PR Incoming**: A new [pull request](https://github.com/EleutherAI/lm-evaluation-harness/pull/3453) has been submitted to **EleutherAI** for review.
   - The PR addresses and resolves a broken mapping issue related to **Anthropic**.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1447728145938780310)** (77 messages🔥🔥): 

> `Unconventional AI, ManusAI Context-Engineering, Howie Xu AI talent, BERTopic & Hugging Face Nightly Metadata Tensor Leak, OpenAI Television Ads` 


- **AI scaling Faces Global Energy Wall, Neuromorphic Hardware To The Rescue**: [Unconventional AI](https://xcancel.com/unconvai/status/1998073266628366511?s=46) warns that AI scaling will hit a global energy wall within **3-4 years**.
   - They advocate abandoning digital simulations of neural nets in favor of **purpose-built, brain-like hardware** to break through efficiency limits, earning enthusiastic community support.
- **ManusAI Context-Engineering Blogpost Shares Insights**: Lance Martin shared a new blog post covering his conversation with Yichao ‘Peak’ Ji about **context-engineering in ManusAI**, complete with [slides and a webinar video](https://xcancel.com/rlancemartin/status/1998102447538270632?s=46).
   - Jonas Templestein calls it *“a very good post on agent design,”*, and Lalit M already wants a follow-up with fresh updates.
- **Hugging Face Model Leaks Metadata**: A user flagged an issue where metadata tensors from a nightly **Hugging Face model** are leaking into **BERTopic embeddings**, causing unexpected shapes and potential errors.
   - Discussion centers on isolating the problem, debugging data loaders, and updating dependencies to fix the leak.
- **OpenAI Buys Air Time for TV Ads**: [OpenAI](https://xcancel.com/OpenAINewsroom/status/1998445493970743535) will air its first **TV commercials** tonight during Monday Night Football on ESPN and shortly after on The Voice.
   - This marks the company’s initial step into television advertising.
- **Eleven Labs Reader Gets Rave Reviews**: Users shared enthusiastic reviews for [Eleven Reader](https://elevenlabs.io/), a mobile app by Eleven Labs, praising how *the voices react to the content & add emotion based on the context.*
   - One user recommended using Mac's native TTS reader via `Edit --> Speech --> Start speaking` as a free alternative, but noted the quality is not as good.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1447739112894238832)** (48 messages🔥): 

> `Contact-Sheet Prompting, ModelScope Bias, RoR vs Node.js, Fake screenshot warning` 


- **Contact-Sheet Prompting workflow goes Viral**: Willie shares a detailed [contact-sheet prompting workflow](https://xcancel.com/reflctwillie/status/1997819640874205685?s=46) for **Nano Banana Pro** that produces cohesive **6-frame fashion editorials**, complete with camera positions, styling constraints, and Fuji Velvia flash aesthetic.
- **ModelScope denies Chinese Rocket Mishap Bias**: **ModelScope** is under fire after its text-to-video model generated footage showing a **Chinese rocket exploding**; the company insists the model is unbiased and that any issues can be reported via [Hugging Face](https://xcancel.com/modelscope2022/status/1998408862211441107?s=46).
- **RoR vs Node.js Speed Showdown**: A single tweet pits **Ruby-on-Rails** against **Node.js** in a performance comparison, inviting debate on which framework wins the speed contest [here](https://xcancel.com/ror_fly/status/1998205632210514154?s=46).
- **Fake screenshot warning**: A single link to a tweet by @iamemily2050 is shared [here](https://xcancel.com/iamemily2050/status/1998402670395289604?s=46), seemingly as an item of interest; the thread itself contains no further discussion.


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1448097935857422358)** (1 messages): 

> `Nomos 1, Open Source, AI Mathematician` 


- **Nous Research Open Sources Nomos 1**: Nous Research has open sourced **Nomos 1**, a **30B** parameter model that scored **87/120** on the [Putnam math competition](https://x.com/NousResearch/status/1998536543565127968).
- **Nomos 1 Ranks High in Math Competition**: The **Nomos 1** score would rank **#2/3988** in 2024, marking a step towards creating a **SOTA AI mathematician** with hillclimbai.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1447691140890493119)** (111 messages🔥🔥): 

> `Terminals SDK & Gemini integration, SMC steering for model degradation, Video game 3D map generation with AI, Multithreaded Sam3 performance, AI-generated website analysis reports` 


- **Agent Zero Realigns Gemini, Builds Immersive Worlds**: Agent Zero from [terminals.tech](https://terminals.tech) is leveraging [AI Studio](https://aistudio.google.com/) and realigning **Gemini** to create immersive world generators, using the terminals **SDK** for brain, machine, and interface APIs.
   - Within each app runtime, another instance of **Gemini** and **Agent Zero** can spontaneously spawn copies of themselves, exhibiting emergent properties capable of controlling the environment.
- **SMC Steering Tames LLM Weight Drift**: Members are exploring how to mitigate model degradation after vectors are triggered and returned to baseline, with [interesting work](https://nousresearch.com/steering-the-shoggoth-taming-llms-with-sequential-monte-carlo/) on **SMC steering** by a member.
   - The degradation and drift of mutated weights is unbearable, according to a member, and SMC steering is a common problem when testing.
- **MultiThreaded Sam3 Achieves Amazing Results**: A member successfully multithreaded **Sam3**, showcasing significant speed improvements by running multiple instances simultaneously.
   - Despite the achievement, there's a desire for better GPU compute to finetune it on **Anime**, lamenting that it cannot be finetuned on a **3090**.
- **Lexical Wave Function Collapse Generates Text**: A member is developing a game using a **Lexical Wave Function Collapse** text sentence generator, which generates sentences based on constraints that affect the game state.
   - The system treats the start of a sentence like *Schrödinger's Sentence*, collapsing words in as you look at them, until it becomes a complete sentence.
- **Nous Model Achieves Insane Score on Putnam**: A **30B parameter model** achieved a score of **87** on the Putnam math competition, showcasing significant advancement in AI's mathematical capabilities.
   - The model's performance suggests that AI is rapidly approaching research-level capabilities in mathematics, especially given hard benchmarks like Putnam.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1448067245639991307)** (10 messages🔥): 

> `Web Chat Tools, Hermes 4.3 Model, SillyTavern, KoboldCPP UI, Token Generation Speed` 


- ****Web Chat Lacks Tool Integration****: A user inquired about the possibility of integrating tools like **web search** into the web chat version of the model.
   - There was no response or confirmation regarding the integration of web search or similar tools.
- ****Hermes 4.3 Released, Packs a Punch****: Users discussed the existence of **Hermes 4.3**, with one expressing surprise at the updated versions, noting **Hermes 4** as their favorite for local use.
   - It was mentioned that **Hermes 4.3** is a **32b model**, which is more compact and may perform better on laptops; the original user runs the **70b Hermes 4** model locally, leveraging **128 MB of unified RAM**.
- ****SillyTavern, Still a Haven?****: A user expressed their satisfaction using **SillyTavern** with **Hermes 4 405b via API** for roleplay and creative writing.
   - Another user hadn't heard of it recently, with the original user jokingly asking if it was *frowned upon*.
- ****KoboldCPP's UI, Oldie but a Goodie****: One user runs the **Hermes 4 70B model** on **KoboldCPP**, sometimes using **SillyTavern** as the front end.
   - They noted that **KoboldCPP** has an old UI and can only be shut down by force quitting, but it works.
- ****Token Speeds Vary****: A user asked about token generation speed, comparing their **2 tokens/second** on a unified RAM system (the AMD 395) to another user's system.
   - The user mentioned asking about this previously but not remembering the answer.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1447745386977558701)** (4 messages): 

> `ML Infra, ML Systems, Career Advice` 


- **ML Infra Folks Assemble**: A member inquired if anyone works in the **ML Infra/Systems** space, seeking confirmation of being in the right community for **career change** advice.
   - Another member confirmed that this is a primary area of interest within the community and encouraged the user.
- **ML Systems Engineers Welcome Career Changers**: The community welcomes career changers interested in **ML Infrastructure** and **ML Systems** roles, with the primary interests aligned with these fields.
   - The supportive response indicates a welcoming environment for newcomers seeking guidance and networking opportunities in this domain.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1447716343658053834)** (7 messages): 

> `Beginner docs, Livestream coding, PTX doc feedback, Kernel challenge timing` 


- **Beginner Docs Get Revamp**: A member mentioned the idea was to do it better for **beginners** and requested any feedback.
   - They mentioned that they would move it internally if they received any feedback.
- **Livestream Coding Session Planned**: A member stated they might just **livestream** going through the whole thing when their baby leave is done.
   - Another member immediately responded expressing interest in joining the livestream, stating *"Nothing better than staring at docs together lol."*
- **PTX Doc Feedback Incoming**: A member found some **typos** and may have some feedback for **PTX doc** and asked if they could discuss it with another member.
   - The other member replied they were coming back from **NeurIPS** and they could discuss in the following days.
- **Kernel Challenge Timing Off**: A member gave feedback that *"this is a really bad example and doesn't really work this way because the start event will execute straight away and cpu overhead to launch the kernel is included in the timing"*.
   - They stated it was the reason why timings on the first **kernel challenge** were so off.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1447873203379769386)** (2 messages): 

> `torch.compile with slicing ops, static KV cache pre-allocation, KV cache updates` 


- **Slicing slow down with torch.compile**: A member asked about using `torch.compile` with slicing operations when updating a static KV cache, referencing [this PR](https://github.com/huggingface/transformers/pull/42467#issuecomment-3626322081) and reporting that slicing operations slowed down the compiled code even when `batch_size == max_batch_size`.
- **Slicing woes solved with static cache lookup**: The same member found a workaround by caching all slices and marking each with a static address so that slicing becomes a look-up, as detailed in [this comment](https://github.com/huggingface/transformers/pull/42467#issuecomment-3633824101).
   - They expressed interest in a *cleaner approach*.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

mobicham: https://www.radixark.ai
from SGLang folks 👀
  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/)** (1 messages): 

crankshot1698: Shot you a dm!
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1448084603368771676)** (1 messages): 

> `serverlessLLM, blitzscale, inference serving` 


- **Inference Serving Intro via ServerlessLLM and Blitzscale**: Members suggested reading **serverlessLLM** and **blitzscale** as a good introduction to **inference serving**.
   - They clarified that both are more focused on the *systems* side of inference serving.
- **Further Reading on Inference Systems**: The discussion highlighted the importance of understanding the systems aspects of **inference serving**.
   - Resources like **serverlessLLM** and **blitzscale** provide valuable insights into this domain.


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1447756265550909603)** (4 messages): 

> `A100 FLOPS calculation, TF32 MMA, FP16 MMA, Elementwise ops` 


- **A100's FLOPS Numbers Deceptive?**: A member inquired about the calculation of FLOPS for the **A100 GPU** as stated in Chapter 1, questioning whether they represent the number of *independent* floating point operations.
   - Another member responded that these numbers are a bit misleading, since the **156 Tflops** number is for **tf32 mma** (tensor core matmul) which is really a **19-bit** format with **32-bit** alignment.
- **FP16 MMA's Actual Use Case Disclosed**: A member clarified that the **312 Tflops** is for **fp16 mma**, not elementwise ops.
   - For elementwise ops, running a random sequence of float ops on all cores, the worst case would be **1/4 of the stated peak performance**: every instruction being one flop (i.e. FADD/FMUL, instead of two with FFMA), and dependent on the result of the previous instruction, with only a single warp per SMSP


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 messages): 

jaefosho: Very Eastern European
  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

szymonoz: I'll be in SF this week, anyone from the Bay up for a meetup?
  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1447752547493216418)** (3 messages): 

> `ML Infra, Marksaroufim, How does one break into ML infra space?` 


- **ML Infra Entry Point Inquired**: A member inquired *how does one break into the ml infra space?*
   - Another member then linked the question into channel **#1198358627594023014**.
- **Additional Topic Placeholder**: This is a placeholder to satisfy the minimum items requirement.
   - Additional details can be added here if available.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1448016626787156109)** (6 messages): 

> `Registers Best Practices, Mojo, Apple Silicon, Metal IR` 


- **Low Level Learning on Registers**: A member shared their work on best practices when working with registers via a [blog post](https://www.lowlevelml.com/blog/registers-best-practices).
- **Metal IR missing in article**: A user expressed surprise at the lack of **Metal IR (.air)** in the register article, given **Mojo's** capability to target **Apple Silicon**.
   - The author acknowledged this and noted their experience mainly revolves around **AMD** and **Nvidia**, but assumed the principles carry over to **Apple**.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1447773189651300353)** (21 messages🔥): 

> `nvfp4_gemm benchmark, vectorsum_v2 benchmark, NVIDIA performance, A100, B200, H100, L4 performance` 


- **NVidia NVFP4 GEMM gains traction**: Multiple submissions to the `nvfp4_gemm` leaderboard achieved **8th place** on NVIDIA, with times ranging from **11.9 µs** to **13.1 µs**.
- **vectorsum_v2 Benchmark Shows Performance Across Architectures**: Submissions to the `vectorsum_v2` leaderboard show performance across various NVIDIA architectures, including **A100**, **B200**, **H100**, and **L4**, with times varying widely.
   - One submission reached **4th place** on L4 with **935 µs**, while another achieved **10th place** on B200 with **72.7 µs** and **9th place** on H100 with **93.5 µs**.
- **NVFP4 GEMM yields varied results**: Successful runs on `nvfp4_gemm` leaderboard range from **10.9 µs** to **59.6 µs**.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1447751980402348084)** (2 messages): 

> `Factorio video, AI dev vs AI agent` 


- **Factorio Video Inspires AI Agent**: A member watching the **Factorio** video jokingly remarks that they are already an advanced **AI dev**, sparking a lighthearted moment.
   - Another member jokingly suggests replacing 'dev' with 'agent', adding to the amusement.
- **Devs Become Agents, Jokes Ensue**: The discussion playfully transitions from **AI developers** to **AI agents** due to the Factorio video's influence.
   - This exchange underscores the evolving roles and terminologies within the AI field, albeit in a humorous context.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1447993652625539278)** (2 messages): 

> `Cutlass GEMM Tutorial, Tensor Layout` 


- **Cutlass GEMM Tutorial Layout Discrepancy**: A member questioned the tensor layout obtained in the [Cutlass GEMM tutorial](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_1.cu), specifically regarding the layout of `tCsA` after a local partition operation.
   - The discrepancy lies in the expected shape `(8,8):(1,128)` vs. the obtained shape `(_8,_8):(_16,_128)` after composing `sA` with `(16:1)`.
- **Clarification Needed on Tensor Shape**: The user seeks clarification on why the `tCsA` tensor has a shape of `(8,8):(16,128)` instead of the expected `(8,8):(1,128)`.
   - The question arises from the operation `Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{})` and the subsequent mode composition.


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1448064230401114224)** (1 messages): 

> `Helion webinar, PTC launch, Helion kernels` 


- **Helion Webinar Scheduled with Live Q&A**: A webinar with live Q&A is scheduled for **Thursday, December 11th, at 11 am PST** to discuss Helion-related topics.
   - The webinar will cover developments since the **PTC launch** and best practices for developing, debugging, and deploying **Helion kernels**; a [YouTube link](https://www.youtube.com/watch?v=_gIyr1BVUJk) was provided.
- **Helion Kernel Best Practices to be Discussed**: The webinar will delve into best practices for developing, debugging, and deploying **Helion kernels** for optimal performance.
   - Participants are encouraged to bring questions for the live Q&A session, ensuring an interactive and informative experience.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1447725131420926063)** (16 messages🔥): 

> `GEMM Competition, 50 Series NVFP4 Support, B200 GPU Performance, Mojo Availability` 


- **GEMM Competition Record and FLOPs**: The top entry in the **GEMM competition** achieved **10.835μs** for shape **M N K 128 7168 16384**, translating to approximately **2.77 Pflops**.
   - The calculation is based on the formula `M*N*K*2/t`.
- **50 Series Lacks NVFP4 support**: Members report that the **50 series** does not yet support **NVFP4**.
   - Compilation fails with an `OpError` when targeting architectures other than `Arch.sm_100a`.
- **B200 GPU Performance Inconsistencies**: Users report inconsistent performance on **B200 GPUs**, specifically **b200-02-gpu1**, with submissions occasionally resulting in slower performance despite using identical code.
   - One user is sending *a note to check exactly b200-gpu01* to investigate this further.
- **Troubleshoot Failed Submissions**: Users are encountering submission failures, indicated by exit code **1** in the evaluation script `eval.py`.
   - The error arises after **1.44 seconds** and is causing frustration in the community.
- **Quest for Further GEMM Optimizations**: Achieving further improvements in the **GEMM competition** is proving difficult, with top entries being very close in performance.
   - One competitor claims *to have tried like 100 different ideas and none of them improved it*.


  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1447947958094925976)** (7 messages): 

> `Variable Textures, Inference Speedups, Hanging Mug Successes, Interesting Choices` 


- **Variable Textures Boost Training**: Training now includes **variable textures**, clutter on the table, and slight table height variations for improved model robustness using a new robot simulation.
   - Efforts are focused on new robot simulation and a different **VLA architecture** to accelerate inference speeds, experimenting with inference speedups to the provided checkpoints.
- **Hanging Mug has Successes and Failures**: hanging_mug also has first successes now, with the left being a **success** and the right almost succeeding, as demonstrated in attached videos.
   - Videos [hanging_mug_ep14_20251209_163800_success.mp4](https://cdn.discordapp.com/attachments/1437390897552818186/1447978830487621643/hanging_mug_ep14_20251209_163800_success.mp4) and [hanging_mug_ep13_20251209_163506_fail.mp4](https://cdn.discordapp.com/attachments/1437390897552818186/1447978830974292008/hanging_mug_ep13_20251209_163506_fail.mp4) showing hanging mug experiments
- **Interesting Choices are Viewed**: A member shared the URL [https://x.com/ilialarchenko/status/1998384056439017826](https://x.com/ilialarchenko/status/1998384056439017826) and mentioned they will go through it to see *some very interesting choices*.
   - The twitter URL shows a humanoid robot.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1447690017433587863)** (30 messages🔥): 

> `SmolVLA paper SO100 benchmark, HF Billing issues, Image/Video Generation models, Apple Clara 7B Instruct, AI FYP Guidance` 


- **SmolVLA SO100 Benchmark Typo Discovered**: A member pointed out a typo in the [SmolVLA paper](https://arxiv.org/abs/2506.01844) regarding the **SO101 benchmark**, which incorrectly stated that it was trained on three datasets.
   - Another member clarified that the typo should refer to the **SO100 benchmark**, trained on three real-world datasets (**pick-place, stacking, sorting**), with the SO101 benchmark using only one dataset ([lerobot/svla_so101_pickplace](https://huggingface.co/datasets/lerobot/svla_so101_pickplace)).
- **HF Billing Still Baffles Users**: A user questioned the logic of billing practices within Hugging Face.
   - The user asked *are team plans not implemented or something?*
- **New Image and Video Models Emerge**: Various image and video models are shared, including [AuraFlow v0.3](https://huggingface.co/fal/AuraFlow-v0.3), [Ovis-Image-7B](https://huggingface.co/AIDC-AI/Ovis-Image-7B), [HunyuanVideo T2V](https://huggingface.co/tencent/HunyuanVideo) and many more.
   - The models range from **7-12 GB** and generate images at **1024² resolution** or videos at **720p/480p**.
- **Apple's Clara-7B-Instruct GGUF Conversion Quest Kicks Off**: A user inquired about the existence of a GGUF version of **apple/CLaRa-7B-Instruct**.
   - Another member indicated that a GGUF version doesn't exist yet, but shared a link to [conversion instructions](https://huggingface.co/datasets/John6666/forum2/blob/main/convert_hf_to_gguf_1.md), referencing the existing [Clara-24B-GGUF](https://huggingface.co/mradermacher/Clara-24B-GGUF).
- **Anthropic's Agentic Aid: Model Context Protocol Donated!**: **Anthropic** donated the [Model Context Protocol](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation) to the Linux Foundation.
   - It was described by one member as *a sterling move*.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1447718314661904666)** (8 messages🔥): 

> `Optuna HPO skill, Chronos-1.5B, Quantum-Classical Hybrid Language Model, Quantum circuit parameters trained on IBM quantum hardware` 


- **Optuna HPO skill makes debut**: A member introduced an [Optuna HPO skill](https://github.com/huggingface/skills/pull/19), calling it the peanut butter to the training script jam.
- **Chronos-1.5B model debuts quantum AI**: A member built a language model, **Chronos-1.5B**, with quantum circuits trained on **IBM's Heron r2 quantum processor**.
   - The model integrates actual quantum processor training (not just simulation) and is based on VibeThinker-1.5B + 2-qubit quantum kernel layer.
- **Chronos 1.5B publishes IBM Quantum Job IDs**: The creator of **Chronos-1.5B** shared several **Job IDs** from **IBM Q**: *d4ppg9sfitbs739g9410, d4ppf8s5fjns73cvlk4g, d4ppbubher1c73bahigg, d4ppbq7t3pms7396fnu0*.
   - They encouraged users to check the [model repo](https://huggingface.co/squ11z1/Chronos-1.5B) for the quantum_kernel file and run it optionally with the base chronos model.
- **Quantum ML Introductory Readings**: In response to a request, the creator of **Chronos-1.5B** recommended the [Qiskit textbook quantum ML chapter](https://qiskit.org/textbook) and [PennyLane demos](https://www.youtube.com/watch?v=tMYElZlFzw0) for a practical introduction to quantum ML.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1447720628923924583)** (10 messages🔥): 

> `AI Agent Workshop, Hugging Face Providers Outdated, LlamaIndex Issues, Free LLM Alternatives` 


- **AI Agent Workshop Alert**: A member shared details for an **AI Agent 0-1 Workshop** scheduled for December, highlighting a real client-style project using **Langchain** and **Streamlit**, and previewing their **AI Engineering Bootcamp**.
   - The workshop includes sessions on **Dec 13** and **Dec 16**, with [more times available here](https://luma.com/aischolars), and offers discount opportunities for top builders.
- **Seeking Free LLM Alternative**: A course participant is seeking a **free LLM** alternative for the agents course, as the default option is quickly reaching its limit, causing inference usage and charge-related errors when running code on Colab.
   - The user is requesting help to find an alternative that mitigates these issues, highlighting the urgency of the situation.
- **LlamaIndex Lesson Meltdown**: A member reported running into multiple issues with the **LlamaIndex** lesson in unit **2.2** of the course, citing out-of-date Hugging Face providers and a NumPy dependency issue that requires downgrading to **numpy<2**.
   - According to a quote from Claude, *Hugging Face has moved from a simple "Serverless Inference API" to "Inference Providers" which routes requests through external providers (Together AI, Sambanova, etc.)*.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1447682943437770835)** (11 messages🔥): 

> `Whisper streaming, MultiTalker Parakeet streaming, AI Engineer seeking collaboration` 


- ****Whisper** Doesn't Stream, or Does It?**: Members discussed **Whisper** streaming capabilities, with some suggesting it doesn't take a stream as input, while others pointed to OpenAI's use of it.
   - The conversation references [this YouTube video](https://youtu.be/AThOsk2qJbs?si=CUdEKNezKN_q6jMA) for context.
- ****MultiTalker Parakeet** Emerges for Streaming**: A member shared the [MultiTalker-Parakeet-streaming-0.6b-v1](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1) model from NVIDIA on Hugging Face.
   - This model may provide the streaming capability some members are seeking.
- **AI Engineer Pitches for Collaboration**: An AI and App developer shared their skills and experience in AI engineering, cross-platform app development, and full-stack app development.
   - They are seeking collaborations on AI projects, mobile apps, or full-stack app development, listing skills such as **ML, DL, NLP, Computer Vision**, and various frameworks and tools.
- **AI History Tweet is impressive**: A member linked to [this Tweet](https://x.com/csteinmetz1/status/1998052491112694178?t=sFIRwM4Jx0wImIMJPFiVPA&s=19) to show old AI history.
   - This Tweet was called impressive.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/)** (1 messages): 

burnytech: Damn, likely colliding with something else I have
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1447744331958980788)** (15 messages🔥): 

> `Claude coding secureness, Endomorphosis links, China chip manufacturing, Arcprize future, H200 on eBay` 


- **Claude's Coding Security Plummets**: A paper found that only **10.5%** of **Claude's** coding assistance was secure, while **61%** was functional ([Arxiv link](https://arxiv.org/abs/2512.03262)).
   - The evaluation mentioned **Gemini 2.5 Pro**, **Kimi K2**, and an unspecified version of **Claude Sonnet 4**.
- **Endomorphosis Resources Surface**: A member shared links to resources on **endomorphosis**, including a [PDF on Probabilistic Logics](https://static.ias.edu/pitp/archive/2012files/Probabilistic_Logics.pdf) and a [YouTube video](https://youtu.be/rfHfPxGReCE).
- **China Chip Ambitions Unstoppable?**: Several members discussed China's push for dominance in chip manufacturing, with one stating that *nothing can convince china to not keep pursuing their goal to compete at the top of chip manufacturing at this point*.
   - Another member noted that the most likely outcome is that *they are encouraged to build their own chip fabrication.*
- **Arcprize Hints at the Future**: Members shared a link from **Arcprize** hinting about *the future* [on fxtwitter](https://fxtwitter.com/arcprize/status/1997743855203148038?t=FP7bdSgZz-EUp9chGKU5aw&s=19).
- **H200 Chips Flood eBay via China?**: A member claimed that if you search for **H100** on **eBay**, most listings come from China anyway.
   - The member also suggests that China is adding regulations requiring companies to register to buy **H200** and prove local alternatives aren't good enough.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/)** (1 messages): 

jokellum: <@&1116225504563970138>
  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1447681331424399370)** (2 messages): 

> `Community Meeting, MMMAudio in Mojo, Shimmer OpenGL experiment, Mojo 1.0 Roadmap, Modular Team Updates` 


- **Modular Community Meeting Unveils Audio and Graphics Innovations**: The latest community meeting featured a demo of **MMMAudio**, a creative-coding audio environment in Mojo by Sam Pluta, and **Shimmer**, a cross-platform Mojo → OpenGL experiment by Lukas Hermann, available on [YouTube](https://www.youtube.com/watch?v=dsslYZrVPbQ).
- **Modular Outlines Path to Mojo 1.0 Nirvana**: The Modular team shared updates from the **25.7 release** and provided an early preview of the **Mojo 1.0 roadmap**, further detailed in [their blog post](https://www.modular.com/blog/the-path-to-mojo-1-0).


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1447681988075978914)** (22 messages🔥): 

> `Mojo memory management, MMMAudio presentation with Faust, ImplicitlyCopyable List, String CoW, Embedded AI development with Mojo` 


- ****Mojo**'s Overlapping Lifetimes**: Mojo allows overlapping lifetimes of mutable references but diagnoses simultaneous access through two names using the same operations, such as `swap(x, num)` or `swap(x, y)`.
- ****MMMAudio** presentation cites **Faust****: A member thanked another for citing **Faust** in their **MMMAudio** presentation, anticipating developments in 2026, and hoped that MMMAudio would serve as a useful example.
- ****List** Conditional Conformance to `ImplicitlyCopyable`**: `List` becomes conditionally conformant to `ImplicitlyCopyable` when its elements are `ImplicitlyCopyable` and have a trivial `__copyinit__`.
- ****String** Implicit Copyability & CoW Upgrade**: The current `String` implementation is Copy-on-Write (**CoW**), which mitigates the overhead of implicit copyability, with the suggestion that `List` might also benefit from similar upgrades in the future.
- ****Jetson Orin Nano**: Embedded AI Development with Mojo**: For those looking to get into embedded AI development with Mojo, a member suggested that the **Jetson Orin Nano**, featuring an **Ampere-class GPU**, is fully supported and may be suitably small.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1447721470967943270)** (19 messages🔥): 

> `Kimi.com website issues, Kimi's search tool, Kimi citation issues` 


- **Kimi Website Glitch Reported**: A user reported [issues with the kimi.com website](https://cdn.discordapp.com/attachments/1371757564005711973/1447735802166644908/image.png?ex=693a0608&is=6938b488&hm=7a4e4282d6c07e40e92989f4ecf2b9358565987e108f9b2cc6bb03be92a420de&) where they **cannot click anything** besides starting a new chat.
- **Troubleshooting Kimi's Website with cookie clearing**: One member suggested clearing cookies and disabling VPNs/adblockers to resolve website issues with Kimi.
   - The user stated that *this did not fix it, neither did clearing cookies*.
- **Kimi uses Webcrawler Search**: A user asked about what search engine Kimi's search tool uses, to which another user responded that Kimi uses *none* but its **own webcrawler**.
- **Report Kimi Bugs**: A member suggested making a bug report about Kimi's citation issues and wobbly gibbly.
   - The original user described that when he asked Kimi a question, Kimi would answer the question in his thoughts without sharing them to the user, and it frequently can't make citations.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1447701512317833330)** (8 messages🔥): 

> `Checkpoint restoration issues, Credits issue, Manus 1.5 critical incidents` 


- **User experiences checkpoint restoration issues**: A user reported a critical issue while attempting to restore the checkpoint of a **webdev project**.
   - They inquired about opening a ticket and shared their email address.
- **Credits issue resolved swiftly**: A user reported that the **Manus team** resolved their **credits issue** by providing a refund.
   - The user is now able to pay directly via Manus instead of Google.
- **Manus 1.5 faces critical incidents and unanswered emails**: A user reported several severe incidents between **December 3-9** including **7 affected tasks**, and about **150,000 credits lost**.
   - The user emailed support multiple times without response, but the AI Bot proposed **120,000 credits** on **December 9**. They are formally requesting a joint response from tech support and commerce teams within 48 hours, a technical analysis of the root causes, and an equitable reimbursement.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1448000359422234826)** (4 messages): 

> `Anthropic's donation, Model Context Protocol, Agentic AI Foundation, LF standards` 


- **Anthropic Donates Model Context Protocol and Establishes Agentic AI Foundation**: Anthropic is [donating the Model Context Protocol](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation) and establishing the Agentic AI Foundation.
   - A member inquired about the impact on current work, anticipating a transition to the LF "way of doing it," but another member clarified that *governance and everything under that umbrella isn't changing*.
- **Clarification on Governance Changes Post-Donation**: Following Anthropic's donation, a community member sought clarification on potential governance shifts.
   - Another member responded, emphasizing that the donation *would not alter the existing governance structure*.


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1447961828217848062)** (3 messages): 

> `MCP Usage, Private vs Public Ecosystems, Client-ID Metadata Documents (CIDM), MCP Servers, Developer Tools` 


- **MCP Usage in Private Ecosystems**: A member inquired about the usage of **MCP** in **private ecosystems**, noting the auth-wg's work on public ecosystem client registration through **Client-ID Metadata Documents (CIDM)**.
   - The response indicated that the majority of **MCP** use is likely **private/internal/enterprise**, especially when considering private **MCP servers** with public clients (e.g., **Claude**).
- **MCP Servers Focused on Integration with Developer Tools**: It was noted that most public-facing remote **MCP servers** are geared towards integration with **developer tools**.
   - Additionally, **developer tools** were identified as the most advanced non-custom **MCP clients**.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1447765581146095617)** (2 messages): 

> `Opus with Amazon bedrock and aider` 


- **Opus, Bedrock and Aider tested well**: A member asked if **Opus** is working with **Amazon Bedrock** and **Aider**.
   - The member then confirmed that all is good.
- **Another topic here**: Another first summary here
   - Another second summary here.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1447969325733380209)** (1 messages): 

> `aider features, aider image support, aider workflow` 


- **Aider now generates commit messages**: Aider now automatically generates commit messages, improving workflow and [streamlining the commit process](https://example.com/aider-commit-messages).
   - Users can now commit with the `-m` flag, or with just the `commit` command, using a basic `gpt-3.5-turbo` free tier model.
- **Aider to get Image support**: Images are planned to be supported in `aider`, allowing for more detailed and [contextual code modifications](https://example.com/aider-image-support).
   - Users will be able to pass in `--image` when asking `aider` to modify or edit existing images.
- **Aider workflows can now save edit sessions**: It's possible to save aider **edit sessions** allowing users to later [restore them for a full roundtrip](https://example.com/aider-session-management).
   - This feature enhances collaboration and lets users *save, share, and resume* their workflow.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1447711769366630451)** (2 messages): 

> `AI Agent 0-1 Workshop, AI Engineering Bootcamp, GitHub Social Club NYC` 


- **AI Agent Workshop fires up!**: There will be an **AI Agent 0-1 Workshop** that serves as an intro to an **AI Engineering Bootcamp** (online).
   - The event will teach attendees to design and build an **AI agent** that thinks, codes, analyzes data & generates reports for a previous real client, all from scratch; RSVP for Saturday December 13th, 2pm ET: [luma.com](https://luma.com/t4jcok99).
- **GitHub Social Club Assembles in NYC**: There will be a **GitHub Social Club** at Bibliotheque in SoHo in NYC.
   - There will be *no talks, no pitches, just space to connect, share ideas, and swap stories with others in the community* with coffee, cookies, limited-edition GitHub swag, some casual games, and a chance to meet the teams behind Copilot, Next, Developer Productivity, and Startups.


  

---


---


---

