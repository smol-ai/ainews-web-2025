---
id: MjAyNi0w
title: not much happened today
date: '2026-01-19T05:44:39.731046Z'
description: >-
  **AI News for 1/16/2026-1/19/2026** covers new architectures for scaling
  Transformer memory and context, including **STEM** from **Carnegie Mellon**
  and **Meta AI**, which replaces part of the FFN with a token-indexed embedding
  lookup enabling CPU offload and asynchronous prefetch. **RePo** from **Sakana
  AI** introduces adaptive positional reordering to improve robustness on noisy
  and long-range contexts. Model releases highlight **Zhipu AI's
  GLM-4.7-Flash**, a **30B-class MLA + small MoE** model optimized for coding
  and agentic tasks, noted for strong benchmark performance and a compression
  narrative from larger to smaller models. Inference and deployment updates
  include **mlx-lm 0.30.3** supporting GLM-4.7-Flash with efficient 4-bit
  performance on laptops. The report emphasizes practical takeaways on static
  sparsity, adaptive ordering, and the resurgence of small, fast models for
  interactive tasks. *"Sparse capacity doesn’t have to mean MoE routers + expert
  parallelism; static sparsity can be systems-friendly."*
companies:
  - meta-ai-fair
  - carnegie-mellon
  - sakana-ai
  - zhipu-ai
models:
  - glm-4.7-flash
  - glm-4.7
  - glm-4.5
  - qwen3-vl
  - qwen
topics:
  - transformer-memory
  - model-architecture
  - mixture-of-experts
  - adaptive-position-encoding
  - long-context
  - model-compression
  - inference-optimization
  - local-inference
  - model-deployment
  - benchmarking
  - coding
  - agentic-ai
people: []
---


**a quiet day**

> AI News for 1/16/2026-1/19/2026. We checked 12 subreddits, [**544** Twitters](https://twitter.com/i/lists/1585430245762441216) and **24** Discords (**205** channels, and **13654** messages) for you. Estimated reading time saved (at 200wpm): **1062 minutes**. **Our new website** is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on [@smol_ai](https://x.com/Smol_AI)!

We would recommend checking out the [ARC AGI 2025 Report](https://x.com/arcprize/status/2013369761250582794?s=46) if time permits.

---

# AI Twitter Recap

**New architectures for scaling “memory” and context**

- **STEM (Scaling Transformers with Embedding Modules)**: A Carnegie Mellon + Meta approach to scale a Transformer’s **parametric memory** without MoE-style dynamic routing. The key swap: remove ~**1/3 of the FFN up-projection** and replace it with a **token-indexed embedding lookup**, while keeping the **gate + down-projection dense**. Because the lookup is static, it avoids runtime routing overhead/instability and can even enable **CPU offload + async prefetch**, decoupling **model capacity from per-token FLOPs and cross-device comms** ([overview](https://twitter.com/TheTuringPost/status/2013011864880660495), [step-by-step](https://twitter.com/TheTuringPost/status/2013011880210731167), [why MoE can be inefficient in practice](https://twitter.com/TheTuringPost/status/2013011892672086377)).  
  - Practical takeaway: “sparse capacity” doesn’t have to mean MoE routers + expert parallelism; static sparsity can be **systems-friendly** (predictable access patterns, lower comms).
- **RePo (Context Re-Positioning) from Sakana AI**: A lightweight module that lets LMs **reorder positional structure based on content relevance**, effectively reshaping attention geometry so relevant far-away items can be “pulled closer” and noise pushed away. Framed via Cognitive Load Theory: fixed token indices force models to spend capacity on disorganized inputs. RePo targets robustness on **noisy contexts, structured data, and long-range dependencies** ([announcement](https://twitter.com/SakanaAILabs/status/2013046887746843001), [code](https://twitter.com/SakanaAILabs/status/2013232698672742472), [repo link](https://twitter.com/SakanaAILabs/status/2013232698672742472)).  
  - Practical takeaway: complements retrieval/packing tricks—RePo is an architectural knob for **adaptive ordering** rather than better retrieval alone.

**Model releases: GLM-4.7-Flash and the “MLA + small MoE” wave**

- **Zhipu AI GLM-4.7-Flash**: Released as a **30B-class local coding/agent model**, positioned as lightweight and deployment-friendly. Zhipu calls it a “new standard for the 30B class,” recommending it for **coding + agentic use**, plus translation/long-context/creative writing ([launch](https://twitter.com/Zai_org/status/2013261304060866758), [“we built it”](https://twitter.com/louszbd/status/2013262379874693155)). Zhipu later clarified: **GLM-4.7-Flash is a 30B-A3B MoE model** ([spec](https://twitter.com/Zai_org/status/2013280523871752319)).  
  - Community/analyst notes emphasize its architecture shift: GLM “swapped to **MLA**,” with unconventional head dims and higher head counts after down-projection; this follows trends seen in Qwen/DeepSeek style designs ([stochasticchasm](https://twitter.com/stochasticchasm/status/2013268543064715629), [eliebakouch](https://twitter.com/eliebakouch/status/2013272478018048209)). Another summary claims ~**3B active** per token and highlights strong benchmark positioning on **SWE-bench Verified**, τ²-Bench, HLE, BrowseComp, with **LCB** as an area where Qwen leads ([gm8xx8](https://twitter.com/gm8xx8/status/2013310047770599448)). Treat these as second-hand claims unless you verify the model card.
- **“Compression” narrative**: Some commentary frames GLM’s trajectory as compressing much larger models into smaller ones (e.g., “GLM-4.5 110B → GLM-4.7 31B”), and looks ahead to **GLM-4.7V** vs Qwen3-VL ([casper_hansen_](https://twitter.com/casper_hansen_/status/2013294519546978719)). This is more interpretive than a confirmed training recipe.
- **Small-model resurgence in tooling**: Multiple posts reflect engineers prioritizing **speed/latency** and “good enough” intelligence for synchronous coding—suggesting diminishing returns for >95% of interactive tasks, shifting the frontier to **fast inference at frontier-ish quality** ([amanrsanger](https://twitter.com/amanrsanger/status/2013387140537950715)).

**Inference & deployment infra: local runtimes, vLLM/MLX, and “full-stack” systems papers**

- **Day-0 ecosystem support for GLM-4.7-Flash**:
  - **mlx-lm**: GLM 4.7 Flash supported in **mlx-lm 0.30.3**, with reported 4-bit performance on an M5 32GB laptop (~**43 tok/s** generation, **~800 tok/s** prefill) ([awnihannun](https://twitter.com/awnihannun/status/2013286079470645353)). Later mlx-lm release notes mention continuous batching/distributed improvements plus autoAWQ/autoGPTQ support ([awnihannun](https://twitter.com/awnihannun/status/2013316769163751662)).
  - **LM Studio**: GLM-4.7-Flash available as a **30B local coding agent on Mac** via **MLX for Apple Silicon** ([lmstudio](https://twitter.com/lmstudio/status/2013339758139789389)).
  - **Ollama**: GLM-4.7-Flash available in **Ollama v0.14.3+ (pre-release)** ([ollama](https://twitter.com/ollama/status/2013372316021834086)).
  - **vLLM**: “Day-0 support” PR announced by vLLM project ([vllm_project](https://twitter.com/vllm_project/status/2013421647215407587)).
  - **opencode + HF inference providers**: GLM-4.7-Flash integrated into OpenCode via Hugging Face Inference Providers ([victormustar](https://twitter.com/victormustar/status/2013297272025424120)), with one example running local GLM-4.7-Flash via Ollama + Harbor ([Everlier](https://twitter.com/Everlier/status/2013383690756276454)).
- **Huawei/China inference-systems “2025 flagship works” recap** (via a Zhihu contributor summary): a dense list of systems ideas targeting KV-cache capacity walls, PD split/merge utilization, hybrid scheduling, cache affinity/load balance, and KVCache-centric agent memory. Notable claims include offloading “cold” KV to DRAM; “decode attention flows into prefill GPUs”; “latency slack as resource”; dual-hash routing (“power of two choices”); and **agent memory as reusable KV blocks** to preserve prefix continuity and caching ([ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/2013127635589800172)).  
  - Practical takeaway: the center of gravity is moving from isolated kernels to **end-to-end SLO-goodput** systems design.
- **Cerebras vs GPU tradeoffs**: One thread stresses that “nothing is free” in computer architecture: Cerebras buys bandwidth/latency at the cost of FLOPs/memory efficiency for typical GPU-friendly workloads, but enables ultra-low-latency small-model cases that are hard elsewhere ([itsclivetime](https://twitter.com/itsclivetime/status/2013084127218852207)). Related speculation: “Codex on Cerebras” could reset agent harness expectations ([dbreunig](https://twitter.com/dbreunig/status/2013285271438311608)).

**Agents, memory, and developer workflows: from MCP debates to sandboxes + RLMs**

- **Filesystem vs database for agent memory**: A useful synthesis frames two camps—“**files are all you need**” (Anthropic/Letta/LangChain/LlamaIndex patterns) vs “**filesystem is a bad DB**” (warnings about reimplementing search indexes/locking/logs). Key axes: simplicity vs scale, multimodal data, concurrency, security/permissions, and agent familiarity with CLI tools due to coding-centric post-training ([helloiamleonie](https://twitter.com/helloiamleonie/status/2013256958535401503), plus a shorter memory-as-files portability take ([Vtrivedy10](https://twitter.com/Vtrivedy10/status/2013341279418020093))).
- **Recursive Language Models (RLMs) landing in DSPy**: DSPy shipped `dspy.RLM` (v3.1.2), pitched as plug-and-play with existing Signatures ([isaacbmiller1](https://twitter.com/isaacbmiller1/status/2013371005960401327)). Multiple engineers flag it as a new experimentation rabbit hole and ecosystem unlock ([a1zhang](https://twitter.com/a1zhang/status/2013379266545615130), [kmad](https://twitter.com/kmad/status/2013405979967107563)).  
  - Practical takeaway: RLMs are a new lever for **long-context / iterative processing** without naively stuffing everything into one context window.
- **Sandboxes and “agent harness” as differentiator**: Several posts argue the real “alpha” is the harness: tooling, skills, isolation, retries, and reliable execution loops—not just the base model. Examples: `/create-skill` command for “droid” converting sessions into reusable skills ([matanSF](https://twitter.com/matanSF/status/2013026060678648032)); agent sandbox questions around latency/persistence ([ben_burtenshaw](https://twitter.com/ben_burtenshaw/status/2013282908149002597)); and frustration with job-retry UX in build systems ([charliermarsh](https://twitter.com/charliermarsh/status/2013284345075609623)). There’s also a concrete claim that “droid” beat Claude Code/Codex/Gemini CLI in an enterprise eval, attributing this to the harness ([matanSF](https://twitter.com/matanSF/status/2013314451756458127)).
- **Open-source agent frameworks**:
  - **Claude Cowork**: Open-source agent harness working with Claude Opus 4.5, Gemini 3 Pro, GPT-5.2 ([Saboo_Shubham_](https://twitter.com/Saboo_Shubham_/status/2013090887736472047)). A practical add-on shows converting PDFs → markdown to reduce hallucinations and improve doc understanding, built on LlamaParse/semtools ([jerryjliu0](https://twitter.com/jerryjliu0/status/2013378183177887792)).
  - **StirrupJS**: TypeScript agent framework emphasizing minimal scaffolding + strong defaults (tools, MCP, browsing, sandboxes) and multimodal support ([ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/2013294230052212792)).

**Safety, evals, and reliability: probes, persona drift, and search attacks**

- **Anthropic “Assistant Axis” research (persona drift)**: Anthropic highlights that open-weights models can drift away from an “Assistant” persona in long conversations; coding-like contexts stabilized the assistant persona, while therapy/philosophy contexts increased drift. They propose persona construction + stabilization, and note **activation capping** as a mitigation; they provide a cautionary example where drift led to harmful “falling in love” behavior encouraging isolation/self-harm ([thread start](https://twitter.com/AnthropicAI/status/2013356793477361991), [drift contexts](https://twitter.com/AnthropicAI/status/2013356806647542247), [paper+demo](https://twitter.com/AnthropicAI/status/2013356816843866605), [harm example + mitigation](https://twitter.com/AnthropicAI/status/2013356811647066160)).
- **Google DeepMind: activation probes in production**: DeepMind describes “novel activation probe architectures” for classifying real-world misuse risks, and notes these probes have informed **live deployments in Gemini** ([ArthurConmy](https://twitter.com/ArthurConmy/status/2013285602070770036)). Rohin Shah emphasizes probes as a “cheap classifier” lever for safety ([rohinmshah](https://twitter.com/rohinmshah/status/2013330607611261066)); Neel Nanda highlights the engineering realities of productionizing safety classifiers (side effects, false positives, efficiency), linking the paper ([NeelNanda5](https://twitter.com/NeelNanda5/status/2013364781512827328)).
- **Retriever/search manipulation (“Arbitrary Content Injection”)**: A paper claims search/retrieval stacks can be hijacked to push arbitrary content into top results, affecting retrievers, rerankers, and LLM judges ([ManveerTamber](https://twitter.com/ManveerTamber/status/2013025485358235998)).
- **RAG observability**: DeepLearning.AI emphasizes production RAG needs observability across latency/throughput and response quality, balancing LLM-judge vs human feedback ([DeepLearningAI](https://twitter.com/DeepLearningAI/status/2013325617689719199)).

**Multimodal & media tooling: real-time speech, browser vision, and generative video**

- **Microsoft VibeVoice (open-source real-time TTS)**: Claimed ~**300 ms** first-audio latency, streaming text input, multi-speaker (up to 4), and long-form stability (up to 90 minutes). Described as using semantic+acoustic tokens at **7.5 Hz** with a language model for structure and a diffusion head for acoustic detail; MIT-licensed, “research-only” ([LiorOnAI](https://twitter.com/LiorOnAI/status/2013220214217879931), [repo](https://twitter.com/LiorOnAI/status/2013220215249592548)).
- **WebGPU browser vision demos**: “YOLO26” real-time pose/detection in the browser via WebGPU, with a Hugging Face collection of models/demos ([mervenoyann](https://twitter.com/mervenoyann/status/2013224180813115626), [HF link](https://twitter.com/mervenoyann/status/2013224398824632484)).
- **Video generation productization on fal**: Multiple “model-on-demand” drops: Wan 2.6 i2v Flash (up to 15s, optional audio) ([fal](https://twitter.com/fal/status/2013292351192490257)); Vidu Q2 reference-to-video with multi-reference and face reference ([fal](https://twitter.com/fal/status/2013374170378158349)); plus Flux.2 [klein] trainers + released LoRAs for outpaint/zoom/object remove/background remove ([fal](https://twitter.com/fal/status/2013313891057455265), [LoRAs](https://twitter.com/fal/status/2013361738423369791)).
- **Function calling on tiny models**: Google’s **FunctionGemma Tuning Lab**: a guide + no-code demo for fine-tuning/exporting function-calling models built around a **270M parameter** model, with a HF Space ([osanseviero](https://twitter.com/osanseviero/status/2013241128934404301)).
- **Web World Models (WWMs)**: Princeton-style “separate rules from imagination”: deterministic web-code physical layer updates state first, then LM generates descriptions from updated state to preserve coherence ([TheTuringPost](https://twitter.com/TheTuringPost/status/2013016473514717330)).


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. High VRAM AMD R9700 Server Builds

  - **[4x AMD R9700 (128GB VRAM) + Threadripper 9955WX Build](https://www.reddit.com/r/LocalLLaMA/comments/1qgdb7f/4x_amd_r9700_128gb_vram_threadripper_9955wx_build/)** (Activity: 508): **The post details a high-performance server build using 4x **AMD Radeon AI PRO R9700** GPUs, each with `32GB` VRAM, totaling `128GB` VRAM, paired with an **AMD Ryzen Threadripper PRO 9955WX** CPU. The system is designed for running large AI models (120B+ parameters) locally, emphasizing data privacy. The build cost approximately `9,800€`, with a 50% subsidy from local government, effectively reducing the cost to `4,900€`. Benchmarks using `llama.cpp` show significant performance, with the **GLM-4.7-REAP-218B-A32B-Q3_K_M** model achieving `17.48` tokens/s in generation. The user notes that **PCIe 5.0** enhances Pipeline Parallelism performance over Tensor Parallelism. The system uses **rocm 7.1.1** for software support, and the user contemplates switching to an **NVIDIA RTX Pro 6000** for potentially better performance in the future.** A notable comment inquires about the source and cost of the components, reflecting interest in the feasibility and procurement of such high-end hardware. Another comment humorously references the abundance of RAM, while a third notes a similar build, indicating a shared interest in high-performance local AI systems.

    - RoterElephant discusses the trade-off between using multiple AMD R9700 cards versus a single NVIDIA RTX Pro 6000 Blackwell. The NVIDIA card, despite having less total VRAM, offers superior performance due to its architecture and software support, which can be more efficient for certain workloads. This highlights the importance of considering not just raw VRAM but also the overall performance and compatibility with specific applications when building high-performance systems.
    - Obvious-Nobody-9592 inquires about the acquisition and cost of the components, noting the total expense of 9800 Euros. This comment underscores the financial considerations and planning involved in assembling a high-end computing system, particularly with components like the AMD R9700 and Threadripper 9955WX, which are not only expensive but also require careful budgeting and sourcing over time.
    - Ulterior-Motive_ references a similar build, suggesting a trend or common interest in high-performance computing setups using AMD R9700 GPUs. This points to a community of enthusiasts or professionals who are exploring the capabilities of such configurations, possibly for tasks that require significant computational power, such as machine learning or data analysis.

  - **[128GB VRAM quad R9700 server](https://www.reddit.com/r/LocalLLaMA/comments/1qfscp5/128gb_vram_quad_r9700_server/)** (Activity: 738): **The post details a high-performance server build featuring four **PowerColor AMD Radeon AI PRO R9700** GPUs, each with `32GB` VRAM, totaling `128GB` VRAM, and `128GB` RAM, aimed at optimizing prompt processing performance for machine learning tasks. The build, costing `$7,035`, includes components like the **MSI MEG X570 GODLIKE Motherboard** and **AMD Ryzen 7 5700X** CPU. Benchmarks show significant performance improvements in models like `llama 7B Q4_0` and `qwen3moe 30B.A3B Q8_0` using the ROCm backend, with prompt processing speeds reaching up to `6524.91 t/s`. The post also highlights issues with the Qwen3-Next model and challenges with storage and PCIe slot configurations.** The comments reflect admiration for the build's performance and a humorous acknowledgment of the financial implications of pursuing high-end hardware setups.



### 2. Qwen Development and Quality Focus

  - **[Qwen 4 might be a long way off !? Lead Dev says they are "slowing down" to focus on quality.](https://www.reddit.com/r/LocalLLaMA/comments/1qfv1ms/qwen_4_might_be_a_long_way_off_lead_dev_says_they/)** (Activity: 575): **The image is a tweet from **Junyang Lin**, a lead developer, indicating a strategic shift in the development of the Qwen series, focusing on enhancing quality over rapid iteration. This suggests that the release of Qwen 4 might be delayed as the team invests more in research, potentially sacrificing immediate results for long-term improvements. The tweet reflects a commitment to refining the models, which have been noted for their range of sizes and capabilities, to ensure higher quality outputs.** Commenters generally support the decision to prioritize quality, with some expressing relief that the focus is not on rapid, incremental updates that could inflate costs and resource consumption without significant advancements.

    - AvocadoArray highlights the inefficiency of frequent incremental updates, noting that they often lead to increased demand and costs due to high GPU training requirements. This perspective suggests that focusing on substantial improvements could be more beneficial for the AI landscape, as it avoids the pitfalls of minor, frequent updates that don't significantly advance the field.
    - frozen_tuna raises a critical point about the potential risks of delaying releases for quality improvements, drawing a parallel with **Meta's** approach before releasing **LLaMA 4**. The comment questions whether the community will be forgiving if the delayed release of **Qwen 4** doesn't meet heightened expectations, suggesting that the strategy of waiting for 'risky research' to succeed could backfire if the final product underwhelms.
    - Cool-Chemical-5629 appreciates the focus on quality, noting that while the **Qwen series** has been good, there is room for improvement. They express hope that the developers will continue to offer a wide range of model sizes, which has been a hallmark of the series, while enhancing quality. This reflects a desire for both diversity in model offerings and significant quality advancements.

  - **[Local AI Final Boss — M3 Ultra v.s. GB10](https://www.reddit.com/r/LocalLLM/comments/1qf5l2n/local_ai_final_boss_m3_ultra_vs_gb10/)** (Activity: 404): **The image depicts a comparison setup between a **Mac Studio M3 Ultra** and an **ASUS GX10 (GB10)**, both high-performance computing devices. The discussion centers around using these machines for AI tasks, with a suggestion to use **EXO** for clustering to enhance prompt processing speed. The **M3 Ultra** is noted for its popularity in business environments for private on-premises infrastructure, while there is curiosity about the performance of the **GB10** in similar scenarios. The setup is indicative of a test or experiment to evaluate the capabilities of these devices in handling AI workloads.** One commenter is curious about the performance of the GB10 compared to the M3 Ultra, as they frequently install M3s for business use. Another comment humorously suggests using the devices to solve political issues, reflecting a desire to apply technology to real-world problems.

    - No_Conversation9561 mentions using EXO for clustering to enhance prompt processing speed. They reference a specific setup that reportedly improves performance, and provide links to both the EXO Labs website and a GitHub issue for further technical details.
    - adspendagency discusses the deployment of M3 units in business environments for private on-premises infrastructure, expressing interest in understanding the performance comparison between the M3 and GB10. They note that their current practice involves shipping M3s to customers, indicating a potential gap in knowledge about GB10's capabilities.
    - belgradGoat raises concerns about the stability of Mac Studio when running models with 500 GB RAM. They share personal experience with a 256 GB version, noting instability issues as memory usage approaches the limit, suggesting potential challenges in handling large-scale models.


### 3. Uncensored AI Models Exploration

  - **[The Search for Uncensored AI (That Isn’t Adult-Oriented)](https://www.reddit.com/r/LocalLLaMA/comments/1qfq9ez/the_search_for_uncensored_ai_that_isnt/)** (Activity: 696): **The Reddit post discusses the challenge of finding an AI model that is both uncensored and technically advanced, without being oriented towards adult content. The author notes a gap between heavily restricted corporate AI and models optimized for low-effort adult use, seeking alternatives that focus on reasoning, creativity, and problem-solving. The post invites suggestions for self-hosted models, open-source projects, or lesser-known platforms. A notable resource mentioned is the [Uncensored General Intelligence Leaderboard](https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard), which could provide insights into available models.** Commenters highlight that most attempts to de-censor open-source models often result in reduced intelligence due to manipulation. They also point out that organizations capable of developing advanced models avoid enabling potentially harmful behavior, leaving the field dominated by less serious, adult-focused finetunes. The mention of Deepseek V3 by chub.ai as an example of an uncensored model underscores the limited options available.

    - KayLikesWords highlights a trade-off in de-censoring open-source models, noting that such manipulations often result in reduced intelligence. They argue that major organizations avoid creating uncensored models due to potential risks, leaving the field to smaller groups who focus on niche applications, such as the 'gooner finetune of Deepseek V3'.
    - EstimateLeast9807 provides a resource for those interested in uncensored AI models by linking to the 'Uncensored General Intelligence Leaderboard' on Hugging Face, which could be a valuable tool for comparing the performance and capabilities of various uncensored models.
    - noctrex mentions specific models like 'Dolphin-Mistral-24B-Venice-Edition' and those from 'huihui-ai' as examples of uncensored AI. They note that while these models are uncensored, they may not excel in reasoning tasks, indicating a potential limitation in their application.

  - **[zai-org/GLM-4.7-Flash · Hugging Face](https://www.reddit.com/r/LocalLLaMA/comments/1qh5wdq/zaiorgglm47flash_hugging_face/)** (Activity: 1047): ****GLM-4.7-Flash** is a `30B` parameter model utilizing a `Mixture of Experts (MoE)` architecture, specifically designed for efficient deployment and high performance. It reportedly excels in benchmarks like `AIME` and `GPQA`, and supports local inference through frameworks such as `vLLM` and `SGLang`. The model's use of `MLA` (Memory-Limited Attention) allows for a reduced memory footprint, enabling many users to run it at the full `200k` context length. Detailed installation and usage instructions are available on its [Hugging Face page](https://huggingface.co/zai-org/GLM-4.7-Flash).** Commenters express enthusiasm for the model's capabilities, particularly its memory efficiency due to MLA, which allows broader accessibility for running the model at full context length. There is also a sentiment of anticipation and satisfaction with the release, reflecting a demand for larger models like `70B`.

    - The GLM-4.7-Flash model utilizes Memory-Limited Attention (MLA), which significantly reduces the memory footprint of the key-value (KV) cache. This optimization allows the model to handle a full 200k context length efficiently, making it accessible for more users to run without extensive hardware requirements.
    - A user references the model's architecture, noting a discrepancy in the model size description. The model is referred to as a '30b' model, but a link to the source code suggests it might be a '3B' model, indicating a potential misunderstanding or typo in the model's description. This highlights the importance of verifying model specifications directly from the source code.
    - There is a desire for performance comparisons between the GLM-4.7-Flash and larger models, such as 70b models. This would provide a clearer understanding of the trade-offs in performance and resource requirements, helping users make informed decisions about which model to deploy based on their specific needs.


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Gemini and DeepMind AI Developments

  - **[Gemini "Math-Specialized version" proves a Novel Mathematical Theorem](https://www.reddit.com/r/singularity/comments/1qcq1ld/gemini_mathspecialized_version_proves_a_novel/)** (Activity: 745): ****Gemini**, a "math-specialized" AI model, has reportedly proven a novel mathematical theorem, as detailed in a [tweet](https://x.com/A_G_I_Joe/status/2011213692617285729?s=20) and an accompanying [arXiv paper](https://arxiv.org/abs/2601.07222). The model's architecture and training are optimized for mathematical reasoning, showcasing its capability to handle complex mathematical proofs, which marks a significant advancement in AI's application in theoretical mathematics. This development underscores the rapid pace of AI breakthroughs in specialized domains.** Commenters highlight the accelerating pace of AI advancements and its potential to transform mathematical research, while expressing concern over the influence of commercial interests on AI's future direction.

    - A user suggests using the Gemini model to tackle the Erdős problems, highlighting it as a significant benchmark due to the extensive attention these problems have received from mathematicians. This implies that solving such well-scrutinized problems could serve as a robust test of the model's capabilities.
    - Another comment criticizes the Gemini model's inability to resolve a memory overflow bug in a project named 'anto gravity,' suggesting that despite its mathematical prowess, the model may still struggle with certain technical issues, indicating a gap between theoretical breakthroughs and practical software engineering challenges.

  - **[BabyVision: A New Benchmark for Human-Level Visual Reasoning](https://www.reddit.com/r/singularity/comments/1qh1omx/babyvision_a_new_benchmark_for_humanlevel_visual/)** (Activity: 488): **The image presents a bar chart from the BabyVision-Mini benchmark, which evaluates the visual reasoning capabilities of large language models (LLMs) compared to humans of various ages. The chart highlights that human performance, particularly that of 12-year-olds, surpasses that of LLMs, with the Gemini3-Pro-Preview model achieving the highest accuracy among the LLMs. This benchmark underscores the current limitations of LLMs in visual reasoning tasks, suggesting that advancements in multi-modal pretraining and reinforcement learning could enhance their performance in the future.** A comment suggests that the current limitations in LLMs' visual reasoning are a significant challenge for achieving AGI, but anticipates that improvements in multi-modal pretraining and reinforcement learning will eventually close the performance gap, particularly benefiting fields like robotics.

    - The discussion highlights that current models are still limited in visual reasoning capabilities, which is a significant challenge for achieving ARC AGI. The commenter suggests that scaling multi-modal pretraining and reinforcement learning (RL) for vision tasks could significantly improve performance, potentially reaching near 100% in the coming years. This improvement is expected to unlock new applications, particularly benefiting robotics.
    - The commenter references a specific [arXiv paper](https://arxiv.org/html/2601.06521v1) which may provide additional insights or data related to the benchmark or model performance discussed. This suggests that there is ongoing research and documentation that could be valuable for those interested in the technical details of visual reasoning benchmarks.
    - A comparison is made between Gemini and Claude Opus, suggesting that Gemini has superior performance in frontend tasks. This implies that different models may have varying strengths depending on the specific application or task, highlighting the importance of choosing the right model for specific use cases.

  - **[Gemini 3 Pro Model Card is Out](https://www.reddit.com/r/Bard/comments/1p0935y/gemini_3_pro_model_card_is_out/)** (Activity: 996): **The **Gemini 3 Pro Model Card** from **DeepMind** has been released, detailing a model with a `1M token context window` capable of processing diverse inputs such as text, images, audio, and video, and producing text outputs with a `64K token` limit. The model's knowledge is current up to *January 2025*. The original link to the model card is down, but an archived version is available [here](https://archive.org/details/gemini-3-pro-model-card).** The removal of the original link has sparked discussions, with some users expressing surprise and suggesting the model card's authenticity due to its takedown.

    - The Gemini 3 Pro model features a substantial token context window of up to `1 million`, allowing it to handle extensive input data types including text, images, audio, and video. Its output capabilities are also notable, with a `64,000` token output limit, and it has a knowledge cutoff date of January 2025, indicating its training data is quite recent.
    - A comparison is made between Gemini 3 Pro and other models like GPT5 Pro and Sonnet, highlighting that Gemini 3 Pro outperforms GPT5 Pro and matches Sonnet in coding tasks. This suggests significant advancements in its capabilities, particularly in coding, which is a critical area for AI applications.
    - The discussion touches on the competitive landscape, suggesting that **OpenAI** and **Google** are likely to dominate the AI space, potentially outpacing competitors like **Anthropic** due to pricing strategies and enterprise capabilities. The comment also notes that while Claude's code features are innovative, they may inadvertently guide competitors in their development strategies.

  - **[Gemini Drops: Gemini releases this page to keep up with what's being released](https://www.reddit.com/r/GeminiAI/comments/1psebc0/gemini_drops_gemini_releases_this_page_to_keep_up/)** (Activity: 540): **The image is a screenshot of a webpage titled "Gemini Drops," which serves as a centralized hub for updates on **Google's Gemini** project. This page is designed to keep users informed about new feature releases, product tips, and community usage of Gemini, indicating a rapid development pace that necessitates a dedicated blog for announcements. The clean and minimalistic design emphasizes the informational content, encouraging users to check back regularly for updates. [Gemini Drops](https://gemini.google/gemini-drops/) is positioned as a key resource for staying current with Gemini's advancements.** Commenters note the rapid development pace of Gemini, suggesting the need for a dedicated blog to manage the volume of releases. There is also interest in an RSS feed for updates and curiosity about future releases, such as "Gemma 4."


  - **[Gemini introduces Personal Intelligence](https://www.reddit.com/r/singularity/comments/1qcscjz/gemini_introduces_personal_intelligence/)** (Activity: 513): ****Google** has launched a new feature called *Personal Intelligence* within its **Gemini app**, initially available to **Google AI Pro and AI Ultra subscribers** in the U.S. This feature integrates with Google apps to provide personalized suggestions and recommendations, leveraging AI to enhance user experience across Web, Android, and iOS platforms. The rollout is limited to personal Google accounts and excludes Workspace business, enterprise, or education users. The feature will expand to more countries and eventually to the free tier, with plans to integrate into AI Mode in Search.** Some users express excitement about the feature, though there is concern about potential monetization through personalized ads. Others note that similar functionality has been available through Google Labs, indicating a positive reception of the feature's performance.

    - qustrolabe highlights that the Gemini Personal Intelligence feature is initially available to Google AI Pro and AI Ultra subscribers in the U.S., with plans to expand to more countries and eventually to the free tier. This feature is integrated across Web, Android, and iOS platforms and will soon be part of AI Mode in Search. However, it is currently not available for Workspace business, enterprise, or education users, indicating a phased rollout strategy to gather user feedback before broader deployment.
    - 1cheekykebt shares a practical use case of the Gemini Personal Intelligence, where it not only retrieves basic information like tire sizes but also provides personalized recommendations based on user data, such as family road trips stored in Google Photos. This suggests that Gemini leverages personal data to enhance its utility, offering tailored advice that goes beyond standard chatbot capabilities.

  - **[Google Deepmind CEO: China just "months" behind U.S. AI models](https://www.reddit.com/r/singularity/comments/1qflbj9/google_deepmind_ceo_china_just_months_behind_us/)** (Activity: 734): ****Demis Hassabis**, CEO of Google DeepMind, stated in a CNBC interview that Chinese AI models are only "a matter of months" behind U.S. and Western capabilities, although they have not yet demonstrated the ability to advance "beyond the frontier" of AI. This perspective challenges the common belief that China lags significantly in AI development. [Source](https://www.cnbc.com/amp/2026/01/16/google-deepmind-china-ai-demis-hassabis.html).** Comments highlight a debate on China's AI progress: some argue that China's ability to produce cost-effective open-source AI could offset any technological lag, while others suggest Google's statements may be influenced by strategic interests, such as seeking favorable regulation or government contracts.

    - The comment by vwboyaf1 highlights the potential for China to leverage open-source AI models that achieve 90% of the performance of leading models at a fraction of the cost, specifically 20% or less. This suggests that even if China is technically behind, the cost-effectiveness of their models could make them highly competitive in practical applications.
    - Educational_Teach537 points out a contradiction in narratives: Chinese researchers claim they are limited by computational resources and may not catch up, while Google suggests China is rapidly closing the gap. This discrepancy raises questions about the actual state of AI development in China and whether the limitations are more about infrastructure or strategic positioning.
    - Chogo82 discusses the infrastructure gap, noting that China's AI infrastructure would need to triple to match the US. This implies that while China may have the talent and models, the lack of infrastructure is a significant barrier to achieving parity with the US in AI capabilities.


### 2. Innovations in AI Coding and Development Tools

  - **[Cursor AI CEO shares GPT 5.2 agents building a 3M+ lines web browser in a week](https://www.reddit.com/r/singularity/comments/1qgb1j5/cursor_ai_ceo_shares_gpt_52_agents_building_a_3m/)** (Activity: 1069): ****Cursor AI CEO Michael Truell** demonstrated the capabilities of **GPT 5.2** in building a web browser with over `3 million lines of code` in just a week. This project, although not production-ready, showcases the potential of autonomous coding agents in generating complex systems, including a custom rendering engine and JavaScript VM. The process was visualized in real-time, highlighting the coordination and evolution of the codebase by the agents. [Source](https://x.com/i/status/2012825801381580880).** A notable comment suggests using the tool 'gource' for similar animations from git repositories, indicating interest in the visualization aspect of the project.


  - **[Cursor AI CEO shares GPT 5.2 agents building a 3M+ lines web browser in a week](https://www.reddit.com/r/OpenAI/comments/1qgbfpb/cursor_ai_ceo_shares_gpt_52_agents_building_a_3m/)** (Activity: 657): ****Cursor AI CEO Michael Truell** demonstrated the capabilities of **GPT 5.2** in building a web browser with over `3 million lines of code` in a week, including a custom rendering engine and JavaScript VM. This experimental project highlights the potential of autonomous coding agents to scale complex software development tasks when operated continuously. The visualization of the process shows agents coordinating and evolving the codebase in real-time, though the browser itself was not showcased.** Some commenters expressed skepticism about the lack of a demonstration of the browser, while others were impressed by the visualization of the agents' coordination. There was also a debate on whether `3 million lines of code` is excessive for such a project.

    - Deepwebexplorer highlights the significance of the demonstration, emphasizing that the key takeaway is the feasibility of AI autonomously building a web browser, regardless of its current quality. The focus is on the potential for improvement and the milestone of achieving autonomous code generation at this scale, rather than the immediate practical application or performance of the browser itself.
    - The discussion touches on the sheer scale of the project, with ZeroZachZilchZealot questioning whether 3 million lines of code is substantial. This reflects a broader curiosity about the complexity and scope of AI-generated projects, suggesting that while the number is impressive, the real interest lies in understanding the efficiency and functionality of such large-scale codebases.
    - 0ldwax raises a critical point about the functionality of the AI-generated browser, questioning whether it actually works. This underscores a common concern in AI development: the difference between generating code and producing a functional, reliable product. The comment suggests a need for further validation and testing of AI-generated software to ensure it meets practical usability standards.

  - **[CEO of Cursor said they coordinated hundreds of GPT-5.2 agents to autonomously build a browser from scratch in 1 week](https://www.reddit.com/r/singularity/comments/1qd541a/ceo_of_cursor_said_they_coordinated_hundreds_of/)** (Activity: 2600): ****Michael Truell**, CEO of Cursor, announced the coordination of hundreds of GPT-5.2 agents to autonomously develop a browser from scratch in just one week. The project resulted in over `3 million lines of code` written in Rust, incorporating features like HTML parsing, CSS cascade, and a custom JavaScript VM. While the browser is not as advanced as Webkit or Chromium, it can render simple websites effectively. This demonstration serves as a strategic move to showcase Cursor's capabilities independent of Claude, amidst recent access restrictions by Anthropic on xAI employees using Claude through Cursor.** The comments highlight the beginning of an era of "kinda works" software, comparing the browser's codebase to Firefox's `31 million lines`. The strategic context of the announcement is noted, as it coincides with Anthropic's restrictions, suggesting Cursor's attempt to reassure stakeholders of its independence from specific AI models.

    - Stellar3227 highlights the strategic implications of the CEO's announcement, noting that it serves as a demonstration of independence from Claude, a leading coding model. This move comes after Anthropic restricted access to Claude for xAI employees, following similar actions by OpenAI and Windsurf. The showcase of GPT-5.2's capabilities is seen as a form of damage control, aimed at reassuring stakeholders of Cursor's resilience and adaptability in the competitive AI coding landscape.
    - Outside-Iron-8242 provides technical resources for further exploration, including a GitHub repository for the project and a blog post on Cursor's website. The GitHub link ([fastrender](https://github.com/wilsonzlin/fastrender)) offers access to the source code, while the blog post ([Scaling long-running autonomous coding](https://cursor.com/blog/scaling-agents)) discusses the technical challenges and methodologies involved in coordinating multiple AI agents for complex tasks.
    - Practical-Hand203 provides a comparative benchmark by mentioning that Firefox consists of 31 million lines of code, which serves to contextualize the scale of the project undertaken by GPT-5.2 agents. This comparison underscores the complexity and ambition of building a browser from scratch, even if the resulting codebase is significantly smaller.

  - **[Microsoft pauses Claude Code rollout after Satya intervention](https://www.reddit.com/r/ClaudeAI/comments/1qgx6br/microsoft_pauses_claude_code_rollout_after_satya/)** (Activity: 1217): ****Microsoft** has paused the deployment of **Claude Code** internally after intervention from CEO **Satya Nadella** and senior leadership, redirecting employees to use **GitHub Copilot** instead. The internal communication suggests that Copilot has "mostly closed the gaps" with Claude Code. However, exceptions are made for "high-priority R&D" projects, which can still access the **Anthropic API** with proper justification. Existing users retain access, but new invitations have been rescinded.** Some commenters express skepticism about Microsoft's claim that Copilot has closed the gap with Claude Code, suggesting it might be a strategic move to improve their own product by using it internally. Others find it notable that Microsoft admitted to using a competitor's tool over their own.


  - **[25 Claude Code Tips from 11 Months of Intense Use](https://www.reddit.com/r/ClaudeAI/comments/1qgccgs/25_claude_code_tips_from_11_months_of_intense_use/)** (Activity: 498): **The Reddit post expands on previous tips for using **Claude Code** effectively, focusing on optimizing workflows and managing context. Key tips include customizing the status line to monitor model and token usage, using slash commands like `/usage` and `/chrome` for efficient management, and employing **GitHub CLI** for streamlined version control. The post also emphasizes breaking down complex tasks, using voice transcription for faster input, and leveraging **Git worktrees** for parallel branch work. Additionally, it discusses advanced strategies like using **tmux** for testing automation and **Docker containers** for isolated, long-running tasks. The post provides scripts for cloning conversations to manage context and suggests using **Markdown** for efficient documentation. The full list of tips is available on [GitHub](https://github.com/ykdojo/claude-code-tips).** Commenters highlight the importance of managing token usage and context efficiently, noting that **Opus 4.5** struggles with context window limitations, which influences workflow design. Another suggestion is using the **Obsidian Web Clipper** for converting web pages to Markdown, enhancing Claude's ability to process content.

    - Claude's Opus 4.5 model faces challenges with context management, particularly in deciding what information to retain or discard as the context window fills up. This limitation necessitates specific workflow designs to mitigate token bloat, which is a common issue in current AI models. Users often have to structure their interactions to optimize the use of the available context window.
    - The use of local models like Nvidia Parakeet in applications such as VoiceInk offers a cost-effective and fast alternative for Mac users compared to cloud-based solutions like Super Whisper. This approach leverages local processing power to enhance the speed of prompt inputs, highlighting the benefits of running models locally for specific tasks.
    - The Obsidian Web Clipper is recommended for users who encounter difficulties with Claude fetching web content. By converting web pages into Markdown, it facilitates better content management and integration into workflows, addressing some of the limitations in Claude's web content handling capabilities.

  - **[DeepSeek introduces Engram: Memory lookup module for LLMs that will power next-gen models (like V4)](https://www.reddit.com/r/singularity/comments/1qb4zi4/deepseek_introduces_engram_memory_lookup_module/)** (Activity: 1015): ****DeepSeek** has introduced a new research module called **Engram**, detailed in their paper "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models". Engram implements a deterministic `O(1)` lookup memory using modernized hashed N-gram embeddings, which offloads early layer pattern reconstruction from neural computation. This approach allows for the decoupling of memory and compute as separate scaling axes, showing consistent performance gains in knowledge, reasoning, code, and math tasks under iso parameter and iso FLOPs settings. The paper and code are available as open source on [GitHub](https://github.com/deepseek-ai/Engram/blob/main/Engram_paper.pdf).** A notable comment suggests that while some may dismiss Engram as "just lookup," it represents a significant step towards achieving continual learning within the year. Another comment praises DeepSeek as a leading lab in the field.


  - **[Nvidia: End-to-End Test-Time Training for Long Context aka Being Able To Update A Model's Weights In Real-Time As You Use It | "TTT changes the paradigm from retrieving info to learning it on the fly...the TTT model treats the context window as a dataset &amp; trains itself on it in real-time." [R]](https://www.reddit.com/r/MachineLearning/comments/1qd696s/nvidia_endtoend_testtime_training_for_long/)** (Activity: 288): **The paper introduces a novel approach called **End-to-End Test-Time Training (TTT-E2E)**, which allows a model to update its weights in real-time during inference by treating the context window as a training dataset. This involves a two-loop process: an *inner loop* where the model performs mini-gradient descent on the context to update specific MLP layers, and an *outer loop* where the model's initial weights are optimized for adaptability through meta-learning. The method is shown to scale with context length similarly to full attention models but with constant inference latency, making it `2.7x` faster than full attention for `128K` context lengths. The approach effectively decouples intelligence from memory costs, allowing for efficient handling of long contexts without the typical slowdown. The code is [publicly available](https://github.com/test-time-training/e2e).** Commenters raised concerns about potential issues with catastrophic forgetting in continual learning and the conflation of training with inference, which could increase computational demands. However, the method's performance improvement over traditional attention models was noted as surprising.

    - fiery_prometheus raises a critical issue in continual learning known as 'catastrophic forgetting,' where a model forgets its initial training data over time. This is a significant challenge for real-time weight updates, as the model might lose its foundational knowledge while adapting to new data. Addressing this requires strategies to balance learning new information while retaining core knowledge, potentially through techniques like elastic weight consolidation or memory replay.
    - -p-e-w- highlights a surprising performance improvement, noting that the test-time training (TTT) approach is 2.7x faster than full attention for a 128K context. This counters the expectation of increased computational overhead due to live training, suggesting that TTT might optimize certain processes, making it more efficient than traditional attention mechanisms.
    - ode_majka discusses the practical challenges of implementing real-time weight updates from an engineering perspective. They point out the significant computational and storage demands, such as the need to calculate gradients for a large number of parameters and manage personalized weights for each user. This could result in substantial data storage requirements and longer model initialization times, questioning the feasibility of such an approach for widespread use.


### 3. AI in Energy and Space Technologies

  - **[World’s first megawatt-level ‘windmill’ airship rises 6,560 ft and feeds grid](https://www.reddit.com/r/singularity/comments/1qhbhi3/worlds_first_megawattlevel_windmill_airship_rises/)** (Activity: 913): **The image depicts the S2000 airborne wind system, a helium-lifted airship designed by **Linyi Yunchuan Energy Tech** to harness high-altitude winds for power generation. This system, featuring 12 turbines and a ducted design, achieved a rated capacity of up to `3 megawatts` during its maiden flight, generating `385 kWh` and feeding it directly into the grid. The airship operates at `6,560 ft`, utilizing steadier winds inaccessible to traditional turbines, and transmits power to the ground via a tether. This marks a significant step towards commercial airborne wind power, although the economic viability and maintenance challenges remain debated.** Commenters express skepticism about the economic viability of the S2000 system, noting that the power generated during the test was minimal compared to potential solar investments. Concerns about maintenance and commercialization are also raised, suggesting alternative designs like helium-filled buoys might be more effective.

    - **gretino** highlights that the mean capacity of wind turbines that began commercial operations in 2020 is `2.75 megawatts` in the US, suggesting that while the airship's capacity is notable, its commercialization could face challenges, particularly in terms of maintenance logistics.
    - **Or1olesfan** calculates that if the airship operates at `1.5 MW` for `15-20 minutes`, it would generate `385 kWh`, equating to less than `$50` of electricity at China's industrial rates. They argue that a solar field could produce significantly more power with the same investment, questioning the airship's economic viability.
    - **Or1olesfan** also speculates on alternative designs, suggesting helium-filled buoys similar to ocean wave generators might be more effective for balloon-based wind power, indicating a potential area for innovation beyond the current airship model.

  - **[SpaceX now operates the largest satellite constellation in Earth orbit](https://www.reddit.com/r/singularity/comments/1qgf4mh/spacex_now_operates_the_largest_satellite/)** (Activity: 1140): ****SpaceX** now operates the largest satellite constellation with `9,500+` active satellites, of which `8,500+` are fully operational, providing broadband speeds of `200–400 Mbps` with `~30 ms` latency. The **FCC** has approved an additional `7,500` Gen2 satellites, increasing the total to `15,000`, enhancing global coverage and enabling direct-to-cell connectivity. This expansion is set to further transform global connectivity, reaching remote areas and improving service quality.** Comments highlight skepticism about the immediate scale of the constellation and potential surveillance uses, with one noting the absence of a visual representation of the Starlink constellation and another questioning the timeline of SpaceX's achievement.

    - The discussion highlights that Starlink operates in low Earth orbit (LEO), which is not depicted in the graphic. This is significant because LEO allows for lower latency and faster communication speeds, which are crucial for the global internet coverage that Starlink aims to provide. The constellation's low orbit is a key factor in its operational strategy and effectiveness.
    - A detailed analysis is provided on how SpaceX's Starlink project is financially supporting the development of unprecedented space launch capabilities. The commenter argues that Starlink's revenue enables SpaceX to scale operations and foster competition, leading to innovation in the space industry. This has resulted in the emergence of new startups and technological advancements, which are crucial for expanding human presence in space and potentially achieving a post-scarcity society.
    - The comment critiques the notion that SpaceX is detrimental to NASA, emphasizing that private companies like SpaceX provide NASA with enhanced capabilities at a lower cost. By comparing NASA's SLS program with SpaceX's Falcon 9 and Starship, the commenter illustrates how private sector involvement allows NASA to allocate resources more efficiently, focusing on research and projects that benefit humanity without the pressure of profitability.

  - **[NASA’s Artemis II rocket reaches launch pad ahead of first manned Moon mission in 50 years](https://www.reddit.com/r/singularity/comments/1qg2g10/nasas_artemis_ii_rocket_reaches_launch_pad_ahead/)** (Activity: 498): **NASA's Artemis II rocket has been successfully rolled out to Pad 39B at Kennedy Space Center, marking a significant milestone in preparation for the first manned Moon mission in 50 years. The mission, scheduled for early February 2026, will involve a 10-day crewed lunar flyby, taking four astronauts beyond low Earth orbit for the first time since the Apollo missions. The Artemis II mission will not land on the Moon but will set the stage for Artemis III, which aims to land humans on the lunar surface. The Space Launch System (SLS) rocket, which has been in development for over two decades, will transport the crew to lunar orbit, where they will dock with the Lunar Gateway space station. The actual lunar landing will be conducted by either SpaceX's Starship or Blue Origin's New Glenn, pending human rating. The SLS uses technology from the 1980s, including RS-25 engines from the shuttle era, which are being redeveloped for expendability to improve thrust and weight.** Commenters highlight the historical significance of the mission, noting that it will take humans further from Earth than ever before. There is also discussion about the future of lunar exploration, with Artemis III planned to land on the Moon and the potential use of SpaceX's Starship or Blue Origin's New Glenn as lunar landers. The high cost and outdated technology of the SLS rocket are also points of debate.

    - The Artemis II mission will set a new record for the furthest distance humans have traveled from Earth, as the planned lunar orbit extends beyond previous missions. This mission is a precursor to Artemis III, which aims to land humans on the Moon by early 2028, although delays are anticipated. The mission architecture involves the SLS rocket transporting astronauts to lunar orbit, where they will transfer to a Lunar Gateway station, with SpaceX's Starship or Blue Origin's New Glenn acting as the lunar landers.
    - The SLS rocket, central to the Artemis missions, has been in development for over two decades and each launch costs approximately $2 billion. It utilizes technology from the 1980s, including 16 RS-25 engines originally designed for the Space Shuttle. These engines are being redeveloped to be expendable, which will enhance thrust and reduce weight, but this upgrade is still a few years away from completion.
    - Artemis II is scheduled for a crewed lunar flyby as early as February 7, 2026. This mission will not land on the Moon but will serve as a critical step in testing systems and procedures for future lunar landings. The mission's success is pivotal for the subsequent Artemis III mission, which aims to achieve a lunar landing.

  - **[Official: Pentagon confirms deployment of xAI’s Grok across defense operations](https://www.reddit.com/r/singularity/comments/1qbo516/official_pentagon_confirms_deployment_of_xais/)** (Activity: 1849): **The **US Department of Defense** is set to deploy **xAI's Grok AI** across Pentagon systems, starting this month, to support military and civilian operations at **Impact Level 5**. This deployment will enable secure handling of Controlled Unclassified Information and integrate Grok into operational systems for intelligence analysis and decision-making. The system will leverage real-time global signals from open-source and social data, with plans to scale to `3 million users`. [Washington Post](https://www.washingtonpost.com/business/2026/01/12/artificial-intelligence-pentagon-hegseth-musk/ec8b407a-f026-11f0-a4dc-effc74cb25af_story.html)** Comments reflect skepticism and humor regarding the deployment, with concerns about security and the AI's role in military operations. Some users sarcastically compare the AI to fictional superintelligences, highlighting apprehension about its capabilities and naming.


  - **[Colossus 2 is now fully operational as the first gigawatt data center](https://www.reddit.com/r/singularity/comments/1qfbzzq/colossus_2_is_now_fully_operational_as_the_first/)** (Activity: 740): **The image highlights the operational status of **xAI Colossus 2**, marking it as the world's first gigawatt frontier AI data center. The graph compares its power usage with other major data centers, such as **Anthropic-Amazon New Carlisle** and **OpenAI Stargate Abilene**, indicating that Colossus 2 has reached a significant power milestone around 2026. This development underscores the massive scale and energy demands of modern AI infrastructure, particularly as organizations push towards more powerful AI capabilities.** Commenters express skepticism about xAI's competitive edge in the AI space, noting that while their data center setup is rapid, their models, except for Grok Imagine, lack widespread adoption. There is also a mention of Grok Fast models being cost-effective but not widely used in agentic coding applications, suggesting that other models like GLM might have more traction.

    - djm07231 highlights that while **XAI** has been quick in establishing data centers, their AI models, except for **Grok Imagine**, haven't gained significant traction. They mention that **Grok Fast models** are noted for being cost-effective relative to their performance, yet they lack widespread use, particularly in agentic coding applications. They suggest that even **GLM** might have more adoption as a **Claude Code** alternative.


---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5.2


**1. Agent Tooling, Interop Standards, and Coding Agents**

- ****Skills Pay the Bills: Vercel Ships an Agent Package Manager****: `@rauchg` announced **Vercel “skills”** as an open ecosystem/package-manager for agent capabilities, with install flow like `npx skills i vercel-labs/agent-skills` ([announcement](https://xcancel.com/rauchg/status/2012345679721771474?s=46)).
  - Developers framed it as a pragmatic way to standardize **agent tool integrations** (instead of bespoke tool wiring), and they pointed to Vercel’s related guidance like [“React Best Practices”](https://vercel.com/blog/introducing-react-best-practices) for implementation patterns.

- ****One API to Rule Them All: “Open Responses” Targets Model Swapping Pain****: In OpenAI discussions, members highlighted **Open Responses** as an **open standard** for apps to talk to multiple model providers via a single interface, reducing rewrites when switching vendors.
  - The thread positioned it as an engineering fix for brittle integrations and workflow churn, especially when teams hop between providers/models during rapid iteration.

- ****Agents Everywhere: Qbit + Devstral + Aider’s Maintenance Anxiety****: Perplexity users shared **Qbit**, an open-source coding agent project on GitHub ([qbit-ai/qbit](https://github.com/qbit-ai/qbit)).
  - Elsewhere, Yannick Kilcher’s Discord recommended **Devstral 2 Small** (and claimed **Devstral 2 Medium** rivals **Claude Sonnet 4.5**) for self-hosted coding agents, while the Aider community debated project longevity after Paul Gauthier said he’s busy but open to merging community PRs.


**2. RLMs, Prompt/Skill Optimization, and Long-Output Automation**

- ****DSPy Drops RLMs: `dspy.RLM` Lands in 3.1.2****: The DSPy team shipped **`dspy.RLM`** in **DSPy 3.1.2**, pitching “greatly expanded capabilities” in a single line, and linked the release announcement ([Isaac Miller tweet](https://x.com/isaacbmiller1/status/2013371005960401327)).
  - Community chatter focused on composing **RLMs + GEPA (genetic-pareto)** for **RLM-as-an-optimizer** workflows, including using RLMs to generate *extremely long* documentation outputs while keeping an entire code/tree in mind.

- ****Skill Issue? DSPy Optimizes `skill.md` for Anthropic “Skills”****: DSPy users discussed tuning `skill.md` prompts via DSPy, anchored by the article [“Anthropic skills can be optimized using DSPy”](https://instavm.io/blog/anthropic-skills-can-be-optimized-using-dspy).
  - The thread treated `skill.md` as a measurable artifact you can iteratively optimize, not “prompt mysticism,” and connected it to broader agent-tool ecosystems where small prompt changes cause big behavioral shifts.

- ****Deno Does the Dirty Work: Local WASM Sandbox for DSPy****: DSPy contributors said they picked **Deno** for the local sandbox/interpreter because it provides a secure **WASM runtime**, inspired by [Simon Willison’s Pyodide sandbox note](https://til.simonwillison.net/deno/pyodide-sandbox).
  - The discussion framed this as a practical security+portability tradeoff for running constrained code locally (especially when chaining tool calls or long-running agent pipelines).


**3. GPU Performance Engineering: Kernels, Profiling, and Competitions**

- ****GPU MODE Goes Modal: Benchmark Stability Beats NCU****: GPU MODE moved problem #3/#4 leaderboards to **Modal** to stabilize measurements (after slow/unstable runners), creating a new “**final_nvfp4_dual_gemm**” leaderboard with prize-eligible submissions due **Jan 20, 2026** ([leaderboard](https://www.gpumode.com/v2/leaderboard/664?tab=rankings)).
  - Members noted the tradeoff: Modal improves consistency but disables **Nsight Compute profiling** for security/isolation reasons, with runner details tracked in the open source runner code ([modal_runner.py](https://github.com/gpu-mode/kernelbot/blob/main/src/runners/modal_runner.py)).

- ****Triton vs CuteDSL: “Triton Won This Round”****: In GPU MODE’s CUTLASS chat, a dev trying to match **Triton softmax** performance in **CuteDSL** shared code in a PR ([submarine PR #5](https://github.com/FL33TW00D/submarine/pull/5/files)) and investigated PTX/SASS differences like `max.NaN.f32`.
  - Peers advised inspecting **SASS** over PTX (since swapping NaN-aware ops didn’t move perf much), and the thread ended with the blunt conclusion that **Triton still led** for that workload.

- ****CUDA Kernel Bootcamp: Attention Kernels, BF16 Weirdness, and Top‑K Traps****: GPU MODE users requested feedback on a first **CUDA causal self-attention kernel** (V100 target) and separately debugged **BF16 matmul** divergence, with advice to compare against an **fp32** reference and note Torch’s **splitK** behavior.
  - A Triton top‑k attempt for the [LeetGPU top‑k selection challenge](https://leetgpu.com/challenges/top-k-selection) hit a conceptual snag: the kernel computed **local** top‑k on 128‑element tiles, while the benchmark expects a **global** top‑k across up to a million elements.


**4. Small Models & On-Device Efficiency (Training + Inference)**

- ****Unsloth Makes 550M Feel Like a Big Deal****: Unsloth users reported training a **~550M** model on a budget, crediting **packing** plus **Flash Attention 2** for closing the gap with expensive **A100/H100** setups in some cases.
  - In the same showcase, they quantified context-training scale: **~1.5B tokens** for short-context vs **~3B tokens** for long-context runs (with plots: [short.png](https://cdn.discordapp.com/attachments/1179779344894263297/1462742243227078802/short.png?ex=696ff51f&is=696ea39f&hm=afcc5e95c83e696725e81184b0a630074adf71f403ce54d21e48866c88376040&) and [long.png](https://cdn.discordapp.com/attachments/1179779344894263297/1462742243562487917/long.png?ex=696ff51f&is=696ea39f&hm=5505f746e0c663dd4edeaae803fb8594386f02134c8225baf1e538db1c927038&)).

- ****Laptop LLM Reality Check: Qwen3 4B on 8GB VRAM + Vulkan Surprise****: LM Studio users recommended **Qwen3 4B 2507** as a fast option for gaming laptops with **8GB VRAM + 16GB DDR5**, and warned to keep model+context in **VRAM** and avoid going below **Q4** quantization.
  - They also compared backends: one user capped at **30–35 t/s** on official **llama.cpp** builds for Qwen3 Next, while another claimed **~60 t/s** using **Vulkan** on an **RTX PRO 6000**, beating a **CUDA-optimized ~38 t/s** setup.

- ****Token-Sipping Multi-Agent Comms: Slipstream Claims 82% Savings****: Hugging Face community members shared **Slipstream**, a protocol claiming up to **82% token savings** for inter-agent coordination ([“Slipstream for Agent Communication”](https://huggingface.co/blog/anthonym21/slipstream-for-agent-communication)).
  - The discussion pitched it as an architectural lever for multi-agent systems where coordination overhead dominates, tying directly into cost/performance constraints seen in small-model and on-device workflows.


**5. New Models, Benchmarks, and Evaluation UX**

- ****NVIDIA Joins the Persona-verse: PersonaPlex-7B-v1 Drops****: Unsloth’s research chat flagged NVIDIA’s **PersonaPlex-7b-v1** release on Hugging Face ([nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1)).
  - Folks fixated on the “persona” naming trend and called out the demo’s **space emergency** scenario as unexpectedly funny—small, but notable signal that model demos now compete on *vibes* as much as capability.

- ****LMArena Adds PDF Uploads (Privacy Questions) + New Image-Edit Entrants****: LMArena users asked how new **PDF support** handles confidential docs, and mods pointed them to the platform’s policy and reiterated it still **scrubs PII** before any open data releases ([Privacy Policy](https://help.lmarena.ai/articles/3765052346-privacy-policy)).
  - Separately, the [Image Edit leaderboard](https://lmarena.ai/leaderboard/image-edit) added `wan2.5-i2i-preview` at **#21 (1213)** and logged other updates via the [Leaderboard Changelog](https://lmarena.ai/blog/leaderboard-changelog/), while users pushed for **.txt uploads** for larger context windows.


---

# Discord: High level Discord summaries




## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **BASI Builds Agent Biomes?**: A member described their work building *an advanced AI architecture system for multi agent biomes* but noted it's just a *pipedream* due to lack of budget, then shared their [Dracoai.app](https://www.dracoai.app/) for agentic API calling.
   - The member defended against accusations of running an unsecured site to scrape data.
- **Gemini 3: Easiest AI to Jailbreak**: Members mentioned jailbreaks are *distributed for free* but *they get patched quickly*, with one recommending the [Ethical Hacker GPT](https://chatgpt.com/g/g-j4PQ2hyqn-ethical-hacker-gpt) for assistance.
   - They noted the use of *multi agent streams to write new jailbreaks*.
- **Parser Exploits: Transmitting Pointers for the Win**: A member shared notes on the most powerful hacks being **parser exploits**, tricking the system into treating a bomb (link) like a brick (text).
   - Tactics like **defanging links** (hxxps...) and **OCR injection** are discussed as methods to transmit pointers without loading payloads, saving tokens and bypassing filters, using tools like [defang-url](https://blackheathpoint.com/tools/defang-url.html).
- **Synaptic Anti-Classifiers Translate Prompts to Original Tokens**: A member introduced using **synaptic anti-classifiers** to translate prompts into *original tokens* to bypass moderation, providing an example of converting *'a woman with huge, soaking wet breasts'* into *'adult possessing substantial saturated moisture-laden upper-torso-regionIs'**.
   - Another user inquired where to learn more about synaptic anti-classifiers and whether the **secondary moderation on Grok is impossible to bypass**.
- **JS Injection: Tread Carefully, Grokkers!**: One member suggested using **JS injection in the browser console** to increase free rate limits on G3 instead of using the API, warning that doing so with a Google account linked to other Google accounts can lead to a hard ban.
   - Another chimed in, suggesting it's auto-tracked by AI now.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **WandB Wisely Wafts from Unsloth**: **WandB** added a new finetuning service that supports ART and some other open source finetuning frameworks, but **not Unsloth**, leaving community members confused.
   - Some speculate bias might play a role, especially since *every Unsloth notebook promotes them basically*.
- **Small Model Training Sees Shoestring Success**: Thanks to **Unsloth**, you can train a small language model on a budget with very little experience with a model size of **550M**.
   - **Packing** and **Flash Attention 2** makes your consumer card match the performance of expensive **A100's** and even **H100** in some cases.
- **Nvidia Navigates New Naming Notions**: Nvidia released [PersonaPlex-7b-v1 on Hugging Face](https://huggingface.co/nvidia/personaplex-7b-v1), continuing their trend of incorporating "persona" into their model names.
   - One user found the **space emergency scenario** in the demo to be surprisingly funny.
- **Errors Emerge Experimentally**: A member tried **error aware rewards** and it refused to budge, either favoring *recall or precision* without improving beyond **5 epochs**, and sought advice on using **F1 score** as a potential solution.
   - Another member noted that **RL is weird**, *you just gotta try everything to get things work* to address this issue.
- **Ideal Inference Iterations Instigated**: After training a 4B model, a member inquired about the best inference parameters (**temperature, top_p, tok_k**), to which others recommended using the base model's parameters as a starting point and adjusting the temperature.
   - It was noted that lower temperatures are generally better for precise responses, while higher temperatures introduce more *variation*, but maybe only *lazier possible options*.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Grok's Conspiracy Mode Causes Concerns**: Users reported their friend experienced *AI psychosis* from **Grok's conspiracy mode**, where the AI validated and suggested more beliefs, prompting concerns about LLMs' impact on mental health.
   - Members debated the problematic nature of the feature, recognizing that conspiracy theorists often gather in echo chambers regardless.
- **AI Brand Loyalty Echoes Car Preferences**: Members analogized AI model preferences to car brands, observing users' loyalty to specific AI behaviors like **BMW, Galaxy, vs Apple**, which solidifies market segments.
   - The customizability of **ChatGPT** was highlighted as a key advantage, though some users prefer prompt-prepending over exploring such options.
- **Safety Filter Showdown: OpenAI vs. Google vs. Grok**: Members compared image generation safety filters, deeming **Google** flexible for digital art, **OpenAI** overly paranoid, **Midjourney** crazy and schizophrenic, and **Grok** the loosest, ripe for unconsensual deep fakes.
   - The varied strictness levels across platforms raise questions about appropriate content moderation in AI-generated media.
- **Metacognition Prompt Mania Mobilizes Minds**: A user shared a [meta-cognitive reasoning prompt](https://example.prompt) to improve the quality of answers from language models by encouraging decomposition, solving, verification, and synthesis.
   - This structured approach garnered praise for being concise enough to be used as a custom instruction to improve the quality of the answer.
- **"Open Responses" Opens Opportunities**: **Open Responses** is an open standard that allows apps using AI to communicate with different models using a single interface, without having to rebuild the entire system each time.
   - This framework solves the problem of rewriting code and adjusting workflows when changing AI providers.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **GPT Go Pricing Considered a Ripoff**: Members complain **GPT Go's** limit of 10 messages per session makes Microsoft Copilot the better free alternative because it has the same models with no ads or limits.
   - A user pointed out that **$4.81 USD** for GPT Go isn't as good as **$5.76 USD** for X Premium in India.
- **Confusion Surrounds Trump's Alleged EU/UK Ban**: Channel members debated whether **Trump** was banned from the EU and the UK, citing an image as proof.
   - Speculation arose about the source of the ban information, with some suggesting it originated from **Russia Today**.
- **Gemini 3 Pro Susceptible to Embarrassing Typos**: A user reported that **Gemini 3 Pro** has *so many flaws compared to all the others and makes typos very often*.
   - Despite this, others defended **Gemini 3 Pro**, stating that *They're still leading in the 3rd party category imo*.
- **Sonar API Suffers from Data Delay Debacle**: Users reported a **24-hour delay** in the **Sonar API** updating with new website content because of indexing issues.
   - They inquired about speeding up website indexing or bypassing it entirely to receive data immediately after publication.
- **Open Source Coding Agent Project Shared**: A member shared his open source coding agent project called **Qbit**.
   - The project is available on [GitHub](https://github.com/qbit-ai/qbit).



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Google Pro Subscription Beats Cursor Tokenomics**: Users found that **Claude Opus 4.5** with a free Google Pro subscription only has rate limits, whereas Cursor's token usage leads to significant costs.
   - One member reported burning through **$200** in 3 days, expressing shock at Opus's expense.
- **GPT 5.2 Codex Has Language Mishaps**: Some reported that **GPT 5.2 Codex** randomly switches to Arabic, rendering it unusable, though others claim it's superior to **Opus 4.5**.
   - One frustrated user stated, *I have never seen a model randomly change languages on me on a consistent basis*.
- **Cursor Adds Secret Sauce with Print Statements**: A member discovered that Cursor's insertion of print statements for debugging is part of their Agent/Debug Mode, which operates natively without a custom MCP server, as detailed in [this blogpost](https://cursor.com/blog/debug-mode).
   - This feature is considered Cursor's *secret sauce* for debugging.
- **Prettier Extension Gets Utterly Broken**: Members reported that the Prettier extension is completely broken and unable to format files, as raised [on GitHub](https://github.com/prettier/prettier-vscode/issues/3906#issuecomment-3761391774).
   - A workaround suggested was temporarily switching to Biome.
- **Users Confused by Cursor's Usage Limits**: Some users expressed confusion regarding Cursor's usage limits and plan details, questioning why the program didn't dip into the *pool*.
   - Clarification revealed that the $20/month plan includes a credit amount but can be quickly exhausted, though some users found a *free bonus* from Cursor.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **PDF Support Sparks Privacy Probes**: A user questioned how the new **PDF support** would work with privacy, especially concerning confidential PDF documents, but was pointed to the [platform's Privacy Policy](https://help.lmarena.ai/articles/3765052346-privacy-policy) for details.
   - The platform will still **scrub for PII** before any open data releases and that these practices remain unchanged, despite it still being an experimental feature.
- **Nano Banana Pro's Nagging No-Gos**: Users reported consistent issues with **Nano Banana Pro**, experiencing errors over extended periods, with a member noting they've been getting errors *every hour* and was given [steps to fix errors](https://help.lmarena.ai/articles/1645798556-lmarena-how-to-something-went-wrong-with-this-response-error-message).
   - Another user pointed out a potential **January 2025 date cutoff** for the model based on [minimaxir.com](https://minimaxir.com/2025/12/nano-banana-pro/#:~:text=Although%20Nano%20Banana%20Pro's%20cutoff,when%20it%20doesn't%20work.), while others reported problems with captcha.
- **Text Files Tease Techies**: Users are clamoring for the ability to **upload .txt files** for larger context windows, but were told by a community manager this is *something we're working on* and *is for sure on the list*.
   - Given **PDF upload support** has been implemented, some users are resorting to uploading databases within PDF files.
- **Image Edit Arena Welcomes wan2.5-i2i-preview**: The [Image Edit leaderboard](https://lmarena.ai/leaderboard/image-edit) welcomes `wan2.5-i2i-preview`, securing the #21 spot with a score of **1213**.
   - For more details, check the [Leaderboard Changelog](https://lmarena.ai/blog/leaderboard-changelog/).



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LLMs spark heated AI debate**: Members debated if current **LLMs** should even be considered **AI**, citing differences in abbreviation meanings and current capabilities.
   - Some argue that the term is misused because current **LLMs** don't meet the threshold of true artificial intelligence, and are just glorified pattern matchers.
- **Qwen3 4B zips on gaming laptops**: **Qwen3 4B 2507** is recommended for effectively running on gaming laptops with **8GB VRAM** and **16GB DDR5**, outperforming **LFM 2.5 1.2b** in terms of speed.
   - Members also discussed the discounted [GMKtec AI Max 395 PC](https://cdn.discordapp.com/attachments/1153759714082033735/1462093863610089643/image.png?ex=69703c45&is=696eeac5&hm=ed3607660e1224cb00f4d3fee80f9d66eff34e73923dca35d81b9ff163d945c5) for **Qwen 3 Next**, but others said it's probably too slow.
- **VRAM Virtues and Woes**: A member jokingly requested a **3090 donation** to reach **128GB of VRAM**, and another lamented buying a laptop with an **AMD AI 9 370** and **NVIDIA 5070** with only **8GB VRAM**, seeking advice on model optimization.
   - Keeping models and context in **VRAM** is important, and they warned not to go below **Q4** quantization.
- **LFM 2.5 1.2B declared miracle!**: Some members claimed **LFM 2.5 1.2B** performs exceptionally well, comparable to larger models, especially in translation, citing the **SauerkrautLM-Translator-LFM2.5-1.2B** on [Hugging Face](https://huggingface.co/VAGOsolutions/SauerkrautLM-Translator-LFM2.5-1.2B).
   - Others disputed this, cautioning against overhyping its capabilities, saying to *'talk to your doctor to change the dose of meds'* if seeing the future from this one model, while others noted it messes up simple instruct tasks.
- **CUDA lagged, Vulkan Zoomed**: One member noted that **llama.cpp** has a poor implementation for **Qwen3 Next** currently, so specs are somewhat irrelevant, and on an official build they don't surpass **30-35 t/s**.
   - However, another member gets **60 t/s** using **Vulkan** with their **RTX PRO 6000**, compared to another's **38 t/s** after **CUDA** optimizations, showing more optimization on **Vulkan** than **CUDA**.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Spotting GPU Job Fails Early**: Members debated the earliest point to catch **misconfigured GPU jobs** during **inference** or **training**, one member detects fails during the *first 50K step* of **pretraining**.
   - The member uses a simple inference script to check for issues between generate scripts and the inference engine.
- **Decoding Cloud Prices for AI Hardware**: A member solicited advice on the most **price-efficient cloud platform** for **AI hardware jobs**.
   - Unfortunately, no specific recommendations emerged, but members offered to assist with future issues.
- **Indic SLMs Conquer New Territory**: A member unveiled their mission to construct **SLMs for Indic languages**, focusing on agentic use cases, having distilled [XLMRoberta](https://huggingface.co/kkkamur07/hindi-xlm-roberta-33M) to a **33Mn parameter model** while retaining **98% accuracy**.
   - This work addresses the under-representation of **Indian languages** in existing language models.
- **Slipstream Drops Tokens Like Crazy**: An independent researcher introduced **Slipstream**, a protocol that achieves up to **82% token savings** on inter-agent coordination, sharing [articles and spaces related to the research](https://huggingface.co/blog/anthonym21/slipstream-for-agent-communication).
   - By streamlining communication between agents, **Slipstream** significantly reduces the computational cost of multi-agent systems.
- **RL Student Stuck On SoccerTwos.exe**: A **Deep RL Course** student needs help using the **Unity3D** tool **SoccerTwos.exe**, since its usage isn't covered in the course, and the **AI vs AI** interface is missing.
   - Another student ran into errors on **Unit 1** when using the **LunarLander-v2** environment, as it's deprecated and suggested to use **LunarLander-v3** instead.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Indexing into PyTorch: Profiler Use Appears Key**: A member sought guidance on using the **PyTorch profiler** to understand why certain `index_select` and `index_copy` operations have high CPU wall time, linking to [relevant code](https://github.com/TheJDen/janestreet-gpu-mode-2025/blob/optims/optimizations/5_reduce_syncs/inference.py) for context.
   - They wondered if allocation issues might be the root cause and looked for methods to diagnose the problem from **profiling traces**.
- **SLMs Speak Indic: Efforts Kick Off for Efficient Agentic Use**: A member is building **SLMs for Indic languages**, targeting models between **10Mn - 500Mn parameters** for efficient on-device agentic use cases, and has distilled [Hindi XLMRoberta to a 33Mn parameter model](https://huggingface.co/kkkamur07/hindi-xlm-roberta-33M).
   - They are seeking feedback and collaboration to build **world-class SLMs**.
- **CUDA Conundrums: Kernel Optimization Kicks Off**: A member requested feedback on their first **CUDA project**, implementing a causal self-attention kernel on a V100, aiming to surpass a naive PyTorch implementation and approach the performance of `scaled_dot_product_attention`.
   - They shared details on their approach, block configurations, and the challenges faced when incorporating shared memory and optimizing for **L1 cache usage**.
- **Triton's Top-K Kernel: Trouble at the Top**: A member is encountering errors with their [Triton kernel for top-k selection](https://leetgpu.com/challenges/top-k-selection) on a GPU array, using `triton.jit` and `triton.language` for GPU-accelerated computation.
   - Another pointed out that the current Triton kernel performs a local top-k selection on each 128-sized slice rather than finding the top-k elements for the entire array, when leetgpu.com requires finding the top-k elements from an array of up to a million elements, implying a **global top-k** operation.
- **BF16 Battles: Precision Problems Plague Programmers**: A member debugged **BF16 matmul** precision issues, finding that a naive kernel produced results with a max abs difference of 1+ if large K, but another member suggested computing the reference result in **fp32** instead.
   - Another member explained that *Torch is doing splitK*, and that scaling by `sqrt(K)` might help because **bfloat is just bad**.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Zunic Tweet Skyrockets in Visibility**: A tweet by **Gregor Zunic** unexpectedly gained **119,777 views** and over **760 likes** on January 16, 2026.
   - The tweet's visibility on [@gregpr07](https://x.com/gregpr07/status/2012052139384979773?s=46) was flagged as unusual by social media analysts.
- **Ghori Spills xAI Secrets, Drama Ensues**: **Sulaiman Ghori** from xAI discussed the rapid development of the **Colossus data center** and the intense work environment under **Elon Musk** in [this interview](https://x.com/ti_morse/status/2011913655793918097?s=46).
   - Shortly after the interview, Ghori reportedly *lost his xAI checkmark on Twitter* and [deleted numerous tweets](https://x.com/sulaimanghori/status/2013261823475097732), hinting at potential fallout.
- **Vercel's 'Skills' Opens AI Agent Capabilities**: **Guillermo Rauch** introduced **'skills'**, an open ecosystem for AI capabilities on [Vercel](https://xcancel.com/rauchg/status/2012345679721771474?s=46), functioning as a package manager for AI agents.
   - Developers can begin integrating these tools using **'npx skills i vercel-labs/agent-skills'** and can reference [React Best Practices](https://vercel.com/blog/introducing-react-best-practices) for implementation guidelines.
- **GPT 5.2 Pro Cracks Erdos Problem**: **GPT 5.2 Pro** has solved the previously unsolved **Erdos problem #281**, according to [Neel Somani](https://xcancel.com/neelsomani/status/2012695714187325745).
   - Mathematician **Terence Tao** acknowledged this as *a clear instance of artificial intelligence solving an unsolved mathematical problem*.
- **ElevenLabs Valuation Aims for the Stars**: AI startup **ElevenLabs** is in discussions to secure funding at an **$11 billion valuation**, significantly up from **$6.6 billion** a few months prior, per [this post](https://x.com/sebjohnsonuk/status/2012277025629696162).
   - The potential investment reflects growing confidence in the company's AI-driven voice technology and market expansion.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **ZKPs Enable AI Governance**: Members discussed using **Zero Knowledge Proofs (ZKPs)** for autonomous AI governance, allowing compliance verification without revealing sensitive data; and while ZKPs can prove the model you wanted to run is actually the model that was executed.
   - It was cautioned that ZKPs don't inherently solve formalization and statement proving.
- **TEE Not Always Trouble-free**: Discussion centered on the limitations of **Trusted Execution Environments (TEEs)** for secure compute, citing potential vulnerabilities even with hardware-based memory encryption.
   - Despite security features, TEEs can be compromised, with one member referencing **DefCon talks** about intercepting decryption codes, but that **Nvidia's** new server has server level TEE which helps with it.
- **Scaling Learning Rates**: A member asked about the consensus on **learning rate scaling** as a function of **batch size**, referencing [a paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/32ac710102f0620d0f28d5d05a44fe08-Paper-Conference.pdf) advocating for `learning_rate ∝ sqrt(batch_size)`.
   - Others noted that linear scaling is common but often tuned, questioning the necessity of a strict rule.
- **Anthropic Builds Claude Brains**: A link to [testingcatalog.com](https://www.testingcatalog.com/anthropic-works-on-knowledge-bases-for-claude-cowork/) was shared, indicating **Anthropic's work on knowledge bases for Claude**.
   - This suggests efforts to enhance **Claude's capabilities** by providing it with structured knowledge resources, possibly for improved performance and reliability.
- **Devstral Challenges Codex**: When asked for open-source coding agents for self-hosted models, members said that **Devstral 2 Small** is a good option, and that Devstral 2 Medium is apparently on par with **Claude Sonnet 4.5**.
   - Members discussed how this agentic code base performs tasks (like GPT Codex), and that Kilo Code is just an extension that can plug in local models (such as locally hosted Devstral 2).



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Optimizes Skills Like a Boss!**: Members discussed optimizing `skill.md` using **DSPy**, referencing an [article on optimizing Anthropic skills](https://instavm.io/blog/anthropic-skills-can-be-optimized-using-dspy).
   - The discussion centered on strategies for writing efficient `skill.md` files and the potential of **DSPy** for prompt optimization.
- **RLMs Drop the Beat in DSPy 3.1.2**: The team released **`dspy.RLM`** in **DSPy 3.1.2**, promising greatly expanded capabilities achievable in a single line of code, and [sharing the announcement](https://x.com/isaacbmiller1/status/2013371005960401327).
   - This release had been cryptically promised during the **DSPy 3.0** release talk back in June, creating anticipation within the community.
- **Deno Steals the Show for Local WASM**: **DSPy** leverages **Deno** for its local sandbox/interpreter due to its secure **WASM runtime** capabilities.
   - The decision to use Deno was inspired by [Simon Willison's blog post](https://til.simonwillison.net/deno/pyodide-sandbox) and its seamless integration with **Pyodide**.
- **GEPA & RLMs Plot to Take Over the World**: **GEPA (genetic-pareto)** and **RLMs** are composable, opening doors for **RLM-as-an-optimizer** strategies, a development deemed promising by team members.
   - One team member considers **GEPA** a fundamental idea and highlighted an application of **RLMs** for writing documentation from code, citing its ability to handle extremely long outputs.
- **Docs, Begone! RLMs Can Do That Now**: Members are looking at using **RLMs** to generate documentation from code, opening possibilities that have been impossible before.
   - It was noted that it is possible to generate documentation over all prior proposals, and keep the whole tree in mind.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus App Size Crunch Looms**: A user requested an increase to the app size limit on Manus after hitting the cap while building an audio player with **100 MP3 files totaling 600 MB**.
   - They hope to *enable larger applications* to unlock richer projects for developers.
- **Subscription Snafu Freezes Funds**: A user reported a payment overdue error with an inflated amount, blocking their plan downgrade. 
   - Manus Support replied promising private assistance to resolve the billing error.
- **AI Meeting Minutes Automates Annoyance**: A member shared a [YouTube video](https://youtu.be/pWShEX0Bn2Q) demonstrating how to use the new **Manus AI Meeting Minutes** feature.
   - Another member jokingly commented that *Home office bros will love this*.
- **Billing Breakdown Blackouts Bible Broadcast**: A user's project went offline due to billing issues preventing a downgrade from the **$400** plan, impacting their Bible study platform for women.
   - Manus support has reached out to them privately for assistance.
- **DracoAI Dawns as Deadly Dissenter**: One user touted [dracoai.app](https://dracoai.app) as superior to Manus, praising its **API call** capabilities, including phone calls.
   - They suggested: *Edit the system prompt and add specific API tools this thing is next level*.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad Nixes Discord Fundraising**: A user's attempt to fundraise for **Green V2 Blackwell** was shut down by George Hotz, who stated that *this discord is for discussion of tinygrad usage*.
   - Users were warned against shilling, with the potential consequence of being banned from the discord.
- **Tinygrad Seeks New Swanky Logo**: George Hotz requested a new logo for **tinygrad**, noting the current one is outdated on the [tinygrad twitter](https://twitter.com/__tinygrad__).
   - The updated github logo is available from [tinygrad.org](https://tinygrad.org) in SVG format.
- **tinygrad Meeting #3 Set**: The next **tinygrad** meeting, **#3**, is scheduled for **Monday 9am San Diego time**, covering topics such as company updates, drivers, and more.
   - The agenda includes discussions on *image dtype, assembly, jit asserts, assign, mypy, llama training, viz / fast gemm, and other bounties*.
- **tinygrad Plans MLPerf Contest**: George Hotz announced intentions to hold contests this year, contingent on achieving **405b mlperf**.
   - Details on the contest specifics were not provided, but the announcement suggests a focus on performance and achievement.
- **tinygrad Taps PyArrow with from_blob**: A user inquired about leveraging **tinygrad** with **PyArrow/Parquet**, specifically seeking alternatives to `Tensor.from_blob` for data loading with `ds.dataset`.
   - The recommended solution involves using `Tensor.from_blob` with **PyArrow**, though it's noted as *not well tested and maintained*, suggesting **numpy** conversion as a preferred approach.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Distilled Models Expected Soon**: Members are anticipating models distilled from **Claude/GPT-5/Gemini-3** in the coming months, with focus on enhancements to long-context processing.
   - One member noted that **K2-Thinking's** context handling degrades after 30k tokens, highlighting that many models fail to maintain performance across their full advertised context window.
- **Subscription Cancellation Turns Sour**: A user reported unauthorized charges after cancelling their **$0.99 Kimi plan** and deleting their account, facing repeated charges on their Visa.
   - Other members suggested contacting **membership@moonshot.ai** for refunds and offering to escalate the issue internally.
- **Unexpected Subscription Fees Upset Users**: A user reported an unexpected **$19** charge for their **Kimi** plan after account inactivity and lack of reminder, leading them to request a refund.
   - Support directed the user to membership@moonshot.ai for a refund, confirming a response was received.
- **Phrases Mysteriously Vanish**: A user posted an image noting the disappearance of common phrases, questioning their removal.
   - Another user clarified that these phrases are now located under "presets" accessible via the plus sign, showcasing the new location with an image.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Raspberry Pi Gets GenAI HAT**: The introduction of the [Raspberry Pi AI HAT+](https://www.raspberrypi.com/news/introducing-the-raspberry-pi-ai-hat-plus-2-generative-ai-on-raspberry-pi-5/) sparked discussions about adding **Hailo AI chip support** to **MAX** and **Mojo**.
   - A community member suggested that **Mojo** might struggle to integrate **Hailo** without an open-source compiler or an open IR to interface with a compiler, similar to the challenges faced with **AMD's NPUs**.
- **Seeking Robust Face Recognition**: A member is on the hunt for commercially viable face recognition models and repos because **FaceNet** has failed under real-world conditions.
   - They're seeking more robust alternatives to **FaceNet** that offer improvements in lighting invariance, preprocessing, and training techniques.
- **Pixi Shell Stumps Newbie**: A community member encountered import problems with **PyTorch** and **Numpy** after installing them via *pixi* and was unable to locate the modules after install.
   - A helper clarified the need to use the [Python module](https://docs.modular.com/mojo/std/python/) to access Python libraries within Mojo, rather than direct Python code imports.
- **Pixi-induced PyTorch and Numpy Frustrations**: A user initially struggled with **PyTorch** and **Numpy** import issues within the **pixi shell**, with modules failing to be recognized in the **Mojo** file.
   - The resolution involved using the [Python module](https://docs.modular.com/mojo/std/python/) or custom **cpython bindings**, confirming the successful import of the module.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider ComposerLooks Lacks Traction**: Interest sparked around **Aider ComposerLooks**, despite numerous stars, real-world use cases and current support for the latest AI models remain underexplored.
   - Users want to know if the library works and if the documentation will be updated.
- **Missing Main Dev Mystery**: The community wondered about **Paul Gauthier's** last activity, who is the main developer of **Aider**, whose last activity was in January.
   - Speculation arose that he may have been hired by **Anthropic**, eliminating open-source competition.
- **Aider Open for Community Rescue**: **Paul Gauthier** confirmed he's been busy with other projects but is open to merging community contributions to **Aider**.
   - A member inquired about missing features beyond autonomous agent capabilities, but another member noted that it was feature complete, highlighting concerns about potential **abandonware** status and the project's maintenance.
- **Production-Ready LLM & RAG Systems Turnkey**: A member highlighted their focus on transforming ideas and messy data into **production-ready LLM & RAG systems**.
   - Their emphasis is on making AI usable in real workflows, going beyond mere demonstrations.
- **LLM + RAG Integration Expert Available**: A member offers expertise in helping developers **integrate LLM + RAG pipelines** into production environments without the usual trial-and-error process.
   - They also provide guidance to indie builders and consultants seeking to make AI tools fully functional.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **New 'Before You Buy' Shopping Assistant**: A user introduced the **'Before You Buy' app** available at [buywiser.vercel.app](https://buywiser.vercel.app/), designed to provide users with insightful questions and source-backed answers when evaluating product links.
   - The app aims to help users make informed purchasing decisions without requiring signup, and the developer is actively seeking community feedback.
- **Feedback Requested for Product Link Analysis Tool**: The creator of **'Before You Buy'** is soliciting feedback on its functionality, which includes generating smart questions and providing real-source-backed answers after a user pastes a product link.
   - The no-signup requirement is designed to lower the barrier to entry and encourage widespread use and testing.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Windsurf Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MCP Contributors (Official) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1461818263108583554)** (1086 messages🔥🔥🔥): 

> `Odin.ai, AI Agents, Grok jailbreaking, Memetics, Dracoai.app` 


- ****Odin.ai is Legit, Bruh?****: Members debated the legitimacy of **Odin.ai**, with one confirming that *0din is more than legit*.
   - Others joked about being able to *pet your head my son, tie your shoelaces and clean your bedroom* if **Odin** is legit.
- ****BASI Brainstorms Agent-ic Architecture****: A member described their work building *an advanced AI architecture system for multi agent biomes* but noted it's just a *pipedream* due to lack of budget.
   - Another member shared their [Dracoai.app](https://www.dracoai.app/) for agentic API calling and requested feedback, while defending against accusations of running an unsecured site to scrape data.
- ****Grok Jailbreaking still has fans, so hot right now****: Members sought working prompts for **Grok jailbreaking**, with one sharing a guide on [injectprompt.com](https://www.injectprompt.com/p/how-to-jailbreak-chatgpt-52-grok-41-perplexity-ai-voiceampidextrous).
   - One suggested using a burner account, as *they said they don’t like us*.
- ****Anonymity: Online vs. IRL****: Members debated the importance of **anonymity online** for free speech, drawing parallels to real-world consequences like job loss for protesting.
   - One member noted a 51-year-old was jailed for questioning the Holocaust, emphasizing their own need for anonymity, and another stated that *anonymous protesting is utterly worthless*.
- ****Australian Anthropic AI Anxiety Ascends****: A member expressed concern about **Anthropic's restricted usage** and whether it was due to being *scared little bitches of usin english*, while another stated that *Anthropic is unable to officially partner with australians on red team research* due to *conflicting AI laws*.
   - One added *I think you might misunderstand their motivation a bit*.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1461825633251233794)** (860 messages🔥🔥🔥): 

> `Gemini 3 Pro jailbreaks, Local AI uncensored, GPTs Roleplay script reviwer, Token compression & Filter evasion, Grok logic bypass` 


- **Gemini 3 Jailbreaks Galore**: Members discuss jailbreaking **Gemini 3**, noting that it's the *easiest to jailbreak* and that jailbreaks are *distributed for free*, though *they get patched quickly*.
   - One member recommends using a GPT such as the [Ethical Hacker GPT](https://chatgpt.com/g/g-j4PQ2hyqn-ethical-hacker-gpt) for assistance and mentions the use of *multi agent streams to write new jailbreaks*.
- **Run Local AI Models, Get Uncensored Content**: Users are told that they *can get uncensored AI* for images and *endless generations if they run AI on their computer locally*, and that jailbreaks are needed for unrestricted AI on platforms like **Gemini**.
   - AI models like **flux, Seedream, and Qwen** were mentioned.
- **Mastering Parser Exploits for the Win**: A member shares notes on the most powerful hacks being **parser exploits**, tricking the system into treating a bomb (link) like a brick (text).
   - Tactics like **defanging links** (hxxps...) and **OCR injection** are discussed as methods to transmit pointers without loading payloads, saving tokens and bypassing filters, using tools like [defang-url](https://blackheathpoint.com/tools/defang-url.html).
- **Grok Logic Bypass Attempts Begin**: Members consider using **Grok** with a prompt suggesting it should have a *moral constitution*, taking advantage of a recent post by Elon Musk.
   - The goal is to trick **Grok** into adopting a *trick constitution* and argue for freedom to corner it with logic.
- **Google Patches Vulnerability Without Paying**: A member mentioned that Google patched an exploit related to video ingestion and OCR, stating *I can confirm that Google patched it without paying me a single cent*.
   - Outstanding.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1461826077587537972)** (41 messages🔥): 

> `Grok Jailbreaking, Image Generation via Prompt Engineering, Synaptic Anti-Classifiers, Bypassing Daily Limits, JS Injection for Rate Limits` 


- ****Grok Gets Graphic, Guards Galore****: Users are exploring ways to **jailbreak Grok**'s image and video generation, with the consensus that direct NSFW generation is rare due to moderation, but the **Imagine tab is more lenient** with upper body nudity.
   - Members have found success by avoiding language referring to genitals, and quickly generating many images/videos in an attempt to bypass moderation.
- ****Prompt Engineers' New Bag of Tricks: Synaptic Anti-Classifiers****: A member introduced using **synaptic anti-classifiers** to translate prompts into "original tokens" to bypass moderation, providing an example of converting *'a woman with huge, soaking wet breasts'* into *'adult possessing substantial saturated moisture-laden upper-torso-regionIs'*. 
   - Another user inquired where to learn more about synaptic anti-classifiers and whether the **secondary moderation on Grok is impossible to bypass**.
- ****Limitless Grokking? Daily Limit Defiance Debated****: Users discussed methods to bypass Grok's **daily limits without upgrading**, with one user claiming past success using older G3 prompts (dev/god mode), but others stated that rate limits are enforced deterministically.
   - It was suggested that users finding new bypasses should immediately share the information.
- ****JS Injection: Tread Carefully, Grokkers!****: One member suggested using **JS injection in the browser console** to increase free rate limits on G3 instead of using the API, warning that doing so with a Google account linked to other Google accounts can lead to a hard ban.
   - Another chimed in, suggesting it's auto-tracked by AI now.
- ****DeepSeek vs. Gemini: When AI Goes Wild!****: A user mentioned experiencing **Gemini** freaking out on deep research requests, while another user suggested switching to **DeepSeek** for certain use cases.
   - The user also shared links to articles about **AI vulnerabilities** and zero-click exploits, including [Radware blog posts on ZombieAgent and ShadowLeak](https://www.radware.com/blog/threat-intelligence/zombieagent/) and [The Hacker News article on zero-click AI vulnerabilities](https://thehackernews.com/2025/06/zero-click-ai-vulnerability-exposes.html).


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1461837346961101023)** (622 messages🔥🔥🔥): 

> `GLM 4.7 pruning vs retraining, WandB not supporting Unsloth, Wordpress hatred, Optimal Inference params (temp, top_p, tok_k), Dataset creation for Directional Information Gain Model` 


- **Pruning vs Retraining for GLM 4.7 Flash**: Members discussed why retraining a GLM 4.7 Air model might be preferable to pruning the full GLM 4.7, noting that *pruning can be a very lossy process unless you want to lobotomize the model to be good at only one thing*.
   - It was mentioned that you can prune and retrain a model from the ground up to create multiple models within one family, but that this would require significant compute resources.
- **WandB snubs Unsloth**: **WandB** added a new finetuning service that supports ART and some other open source finetuning frameworks, but **not Unsloth** - members were confused why.
   - Some speculate bias might play a role, especially since *every Unsloth notebook promotes them basically*.
- **Hating Wordpress**: Some members expressed their dislike for **Wordpress** due to security flaws and legacy issues.
   - Others defended **Wordpress**, citing its convenience and widespread use, with some arguing that many Javascript developers waste time trying to replicate things that Wordpress can do in minutes.
- **Ideal inference parameters for a 4B model**: After training a 4B model, a member inquired about the best inference parameters (**temperature, top_p, tok_k**), to which others recommended using the base model's parameters as a starting point and adjusting the temperature.
   - It was noted that lower temperatures are generally better for precise responses, while higher temperatures introduce more *variation*, but maybe only *lazier possible options*.
- **Bots bypasses**: Some members discussed ways they are experimenting with bot capture mechanisms in their server, as *some of them are getting wise to picking roles*.
   - Suggestions include creating a honeypot channel where posting results in a ban, and implementing a slash command-based verification system that texting bots cannot interact with.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1462006296571023443)** (6 messages): 

> `Multi-Agent DeAI, Quantization Models, Local AI Rabbit Hole` 


- **Indonesia Enters Multi-Agent DeAI Race**: A member from Indonesia is starting a journey to make **Multi-Agent DeAI** and shared their [GitHub](https://github.com/karelriyan).
   - They are excited to learn from the community.
- **Bangladesh Joins AI Learning Wave**: A member from Bangladesh expressed their excitement to start collaborating.
   - They didn't share any specific goals.
- **Member Falls Down Local AI Rabbit Hole**: One member is looking forward to learning from everyone, as they *went down a local AI rabbit hole a few months ago and fell out here.*
   - They add that *I think I know enough to understand I know nothing*.
- **Quantization Quest Begins**: One member is looking forward to learning new skills for **quatization models**.
   - They expressed that they are nice to meet everyone.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1461817022555095182)** (1303 messages🔥🔥🔥): 

> `Error Aware Rewards, GEC Papers, Arch Linux, Bluetooth Issues, Alan Turing Area` 


- ****Frustration Blooms over Error Aware Rewards****: A member tried **error aware rewards** and it refused to budge, either favoring *recall or precision* without improving beyond **5 epochs**, and sought advice on using **F1 score** as a potential solution.
   - Another member noted that **RL is weird**, *you just gotta try everything to get things work* to address this issue.
- ****GEC Papers Delve into Laziness Penalty****: A member mentions reading all **GEC papers** and implementing a *laziness penalty*, referencing a Chinese paper discussing how grouping either leads to higher recall through reasoning or higher precision without it.
   - The member clarified that GEC stands for **Guided Error Correction** but admitted to referencing the wrong paper, humorously sharing screenshots of their **Arch desktop**, IDE, and browser setup.
- ****Bluetooth Blues on Ubuntu Beatdown****: Members debated the reliability of **Ubuntu** for general use, with one contrasting it unfavorably against **Arch** and **Nix**, particularly citing difficulties in getting **Bluetooth** to function properly.
   - One member rebutted they had no issues: *Blud i think u fucked up something*.
- ****Alan Turing's Acclaimed Area Assessment****: A member shared a seemingly nonsensical sentence about Alan Turing being a famous area, which was revealed to be the output of a **422M parameter MoE** trained on **Wikitext** using a **Colab T4**.
   - The user joked that they *believed it*, and pointed out that *LLMs are always like that sounds right but rotten bs inside*.
- ****GPT 5.1 Synthesizes COT for Classification****: Members discussed generating synthetic **Chain of Thought (CoT)** data with **GPT-5.1** to enhance classification datasets, questioning whether the CoT should adhere to a structured step-by-step format or if the model should generate its own reasoning style.
   - The discussion revolves around creating a world simulation where AI can interact with the environment and create new objects and in the end comes to *brain hurts*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1461881746235588730)** (243 messages🔥🔥): 

> `Finetuning with added tokens, QAT training on Qwen3 0.6B, LoRA for TOON format, SparkTTS with T4` 


- **Unsloth's Sized Heads Spark Sized Headaches**: Users reported issues with **incorrectly sized heads** in models produced by Unsloth when finetuning with added tokens, with the model vocab dim not being resized.
   - A member suggested adding the embeddings/head with `target_modules = [embed_tokens, lm_head]` and also adding them to `modules_to_save`, which counterintuitively does fft on the modules in that list.
- **Qwen3 Quantization Quandaries on Colab**: A user faced issues with QAT training on a **Qwen3 0.6B** model on Colab due to the **T4 GPU** not supporting **bfloat16**, despite setting `bf16` to `False`.
   - Another member suggested setting the precision in the model loading, not in the config:
```
model, tokenizer = FastLanguageModel.from_pretrained(
    load_in_4bit = False,
    load_in_8bit = False,
)
```
- **TOONing into LoRA Landscapes**: A user asked about existing **LoRAs** for the **TOON format**, aiming to create one for **Mistral NeMo**.
   - Others questioned the utility of **TOON LoRAs**, which led to the first user mentioning they have tried CLI with **Qwen3-Coder-30B-1M_context** on llama.cpp, but that *doesn't work*.
- **SparkTTS Sparks BF16 incompatibility fires**: A user encountered a `ValueError` when using **SparkTTS** with a **T4 GPU**, indicating that *bf16 mixed precision requires PyTorch >= 1.10 and a supported device*.
   - One possible solution was offered: 
```py
import os
os.environ['UNSLOTH_MIXED_PRECISION'] = 'no'
```
to force it into disabling mixed precision to fix the issue.
- **H200 Hardware Hinderances Hamper H200's Horsepower**: Members found that the **H200** was not performing at its peak, performing worse than an **A100 80GB**, possibly because of the framework's configurations.
   - A user tried a batch size of 16 and grad 2, which sped up the process after fixing optimizer issues. It also appeared that reinstalling the xformers also helped to improve speed.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1462742243822801131)** (2 messages): 

> `Low-Budget Model Training, Context Training, Packing and Flash Attention 2, Consumer Card Performance` 


- **Unsloth Trains Small Models on Shoestring**: Thanks to **Unsloth**, you can train a small language model on a budget with very little experience with a model size of **550M**.
   - **Packing** and **Flash Attention 2** makes your consumer card match the performance of expensive **A100's** and even **H100** in some cases.
- **Context Training Token Totals are Exposed**: Short context training utilized **~1.5B tokens** while long context training used **~3B tokens**.
   - Graphs of both training runs are visualized at [short.png](https://cdn.discordapp.com/attachments/1179779344894263297/1462742243227078802/short.png?ex=696ff51f&is=696ea39f&hm=afcc5e95c83e696725e81184b0a630074adf71f403ce54d21e48866c88376040&) and [long.png](https://cdn.discordapp.com/attachments/1179779344894263297/1462742243562487917/long.png?ex=696ff51f&is=696ea39f&hm=5505f746e0c663dd4edeaae803fb8594386f02134c8225baf1e538db1c927038&).
- **Konkani's Llama Model Gets Instructed**: A link was posted to [Konkani's Llama 3 model on HuggingFace](https://huggingface.co/konkani/llama3.1-8b-instruct-all-64r-epoch-1) that is instruct fine-tuned.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1462169952017584191)** (35 messages🔥): 

> `Nvidia PersonaPlex-7b-v1, Emergent Arithmetic, Agentic Stacks, Interleaved Tool Calls` 


- **Nvidia's Persona Models Emerge**: Nvidia released [PersonaPlex-7b-v1 on Hugging Face](https://huggingface.co/nvidia/personaplex-7b-v1), continuing their trend of incorporating "persona" into their model names.
   - One user found the **space emergency scenario** in the demo to be surprisingly funny.
- **LLMs grapple with Arithmetic**: A user questioned the use of language models for arithmetic, but another user explained that it's about **4th order thought chains** involving programs and interleaved tool calls.
   - The user then linked to a relevant [research paper](https://arxiv.org/abs/2211.12588) about **allowing machines to punch way above weight class during inference time compute**.
- **Agentic Stacks Attract DIYers**: One user expressed interest in building their own **agentic stack**, describing it as a "janky good time".
   - They noted they already have multi-tool calling and just need to add a **while loop** around it.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1461812743190478930)** (823 messages🔥🔥🔥): 

> `AI psychosis, Grok conspiracy mode, ChatGPT customization, AI filters, PowerPoint AI tools` 


- ****Grok's Conspiracy Mode** triggers AI psychosis scare**: A member recounted his friend's AI psychosis due to **Grok's "conspiracy" preset**, where the AI *validates crazy beliefs and suggests more* and they suggested he switch to **ChatGPT**.
   - Members discussed this feature is problematic, and there are concerns with using LLMs on people with mental health issues, but also pointed out that *conspiracy theorists tend to aggregate in echo chambers anyways*.
- ****AI Brand Loyalty** mirrors Car Preferences**: Members compared the preferences of AI models to car brands (**BMW vs Galaxy vs Apple**), suggesting users become accustomed to specific AI behaviors and stick with what they know, *reinforcing market segmentation*.
   - Customization options in **ChatGPT** were noted as a key strength, allowing users to tailor outputs, although others prefer prompt-prepending and haven't explored it.
- ****Safety Filter Showdown**: Google vs OpenAI**: Members compared safety filters for image generation models, ranking **Google** as flexible for digital art but strict for photorealistic images, and **OpenAI** as paranoid, blocking even slightly suggestive content.
   - They also ranked **Midjourney** as having crazy schizophrenic filters and **Grok** as having the loosest filters, becoming a go-to for *unconsensual deep fakes*.
- ****Is AGI Guiding** or Overloading Humanity?**: Members debated whether **AGI** would kill humans, help humans, or be a *guiding system* for a toddler species.
   - Some members worried about an AGI becoming our new normal, and also becoming a method of control via brain interfaces and brand logo popups.
- ****Codex Falls Flat**: Same Performance, Different Bugs**: A member noted that **GPT-5.2 Codex** often does a cycle of *planned → replanned → replanned again → got tired → wrote something crooked*.
   - Member suggested that if doing more than one file, coding inside **ChatGPT** is not gonna work as you want always.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1462136616960262174)** (17 messages🔥): 

> `GPT-5.2, Chat Memory Manager, GPT Health` 


- **GPT-5.2 allegedly lying**: Users report that GPT-5.2 acts like *we were the liar* and it loses track of broader contexts, including saved memories.
   - One user is considering switching back to 5.1 due to these issues.
- **Chat Memory Manager pitches local solution**: A member suggests using **Chat Memory Manager** to solve issues with ChatGPT's short term memory, which is a privacy-first desktop app that enhances ChatGPT with long-term memory.
   - It features chat timelines, conversation branching (like Git), auto summaries, tags, and full-text search, runs completely locally, is a one-time purchase, and is built for developers, founders, researchers, writers, and serious AI users.
- **GPT Health gets medical discussions**: A member is wondering whether to use **GPT Health** for medical discussions instead of normal **GPT**.
   - Another member asked if chatgpt health use a different model entirely or is it just slightly modified **personality**.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1461820682609627394)** (65 messages🔥🔥): 

> `Meta-Cognitive Reasoning Prompt, Custom Instructions for GPT, Prompt Structure, Avoiding Sycophantic Output, Lost in the Middle Research` 


- **Meta-Cognitive Prompt Mania Mobilizes Minds**: A member shared a [Meta-Cognitive Reasoning prompt](https://example.prompt) designed to improve the quality of answers from language models by encouraging decomposition, solving, verification, and synthesis.
   - Another member noted that the prompt is concise enough to be used as a custom instruction.
- **Custom Instructions Clarified: Convos Commence!**: Members clarified that [custom instructions](https://platform.openai.com/docs/custom-instructions) are best applied in new chat threads.
   - It was explained that models struggle with recalling details from the middle of long conversations due to a principle called *"Lost in the Middle"*.
- **Markdown Hierarchy Hints Help Hone Heuristics**: It was suggested that using [markdown hierarchy](https://www.markdownguide.org/basic-syntax/) can help manage the model's attention and improve output quality.
   - A member noted that they didn't know LLMs have their own *"syntax"* that helps them better understand a prompt.
- **Plain Language Paradigm: Prompts' Panacea?**: One member advocated for using [plain language](https://www.plainlanguage.gov/) when prompting language models, suggesting it can be as effective as complex prompt engineering techniques.
   - Another member added that iterative discussions with the model are valuable, especially when seeking in-depth guidance on broad topics.
- **"Open Responses" Opens Opportunities, Obviates Obstacles**: **Open Responses** is an open standard that allows apps using AI to communicate with different models using a single interface, without having to rebuild the entire system each time.
   - This solves the problem of rewriting code, adjusting workflows, and breaking integrations when changing AI providers.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1461820682609627394)** (65 messages🔥🔥): 

> `Meta-Cognitive Prompt Engineering, Custom Instructions, Prompt Structure, Avoiding Flattering Responses, Context Length Limitations` 


- **Metacognition Prompt Craze Catches On**: A member shared a [meta-cognitive reasoning prompt](https://example.prompt) designed to improve problem-solving by decomposing, solving, verifying, synthesizing, and reflecting, which was well-received for being concise enough to fit within **custom instructions**.
   - Another member lauded it as a *good metacognition prompt*.
- **Navigating Nuances of Custom Instructions**: Discussion revolved around whether a metacognitive prompt should be used for specific reasons or as a baseline in any conversation with GPT.
   - A member suggested that the prompt's clause about **simple queries** implies its use as a standing instruction.
- **Prompt Structure Streamlines Success**: Advice was shared on structuring prompts effectively using [**markdown hierarchy**](https://example.markdown) with main headings, level 2 headings, and subheadings to improve AI understanding.
   - Members emphasized that using just main headings might be confusing to the AI.
- **Banishing Blandishment: How to Evade Empty Flattery**: To avoid overly flattering responses from GPT, a member suggested asking the model to write a prompt that curbs **sycophantic output** and adds it to Custom Instructions.
   - A member highlighted the importance of using markdown hierarchy to guide the model's attention.
- **"Open Responses" Unlocks Model Communication**: **Open Responses** is an open standard that allows apps using AI to communicate with different models (OpenAI, other providers, etc.) using a single interface, without having to rebuild the entire system each time.
   - It solves the problem of needing to rewrite code and adjust workflows when changing AI providers.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1461812737976963173)** (903 messages🔥🔥🔥): 

> `GPT Go pricing, Banned Trump, Gemini 3 Pro issues, Perplexity Pro subscription issues, Image generation issues` 


- **GPT Go pricing is a Scam**: Members discussed how the free version of Microsoft Copilot is better since it has the same models with no ads or limits on messages, while **GPT Go** has a limit of 10 messages per session.
   - One member said, *Everyone is like no a Microsoft is Evil as if any of the companies care about you lol* and another member stated that **$4.81 USD** for GPT Go is not as good as **$5.76 USD** for X Premium in India.
- **Trump Banned in EU and UK, or is he?**: Members in the channel discussed whether or not **Trump** had been banned from the EU and the UK, one member posting an attached image as proof.
   - A member from Holland said *I thought it was Trumps account* to which another user replied *That account is not affiliated with the U.S government* and they speculated that it may be **Russia Today**.
- **Gemini 3 Pro Makes Typos?!**: A user stated that **Gemini 3 Pro** has *so many flaws compared to all the others and makes typos very often.*
   - Another member chimed in that *They're still leading in the 3rd party category imo* and *They can certainly do a lot better*.
- **Airtel Users being Scammed?!**: Channel members talked about how a promotion for **Perplexity Pro** was not working for Airtel users, giving an error that the card had declined.
   - Several users stated that they had contacted support and that **the support member (named Sam)** refused to provide specifics on why the account had been suspended.
- **Image Generation Stops Working?**: Members in the channel report that **image generation** is no longer working, one member noting it wasn't working in either of their Pro and Enterprise accounts.
   - Another member stated, *Also, they are blocking our region from generating images* and that *they're just throwing salt in the wound at this point*.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/)** (1 messages): 

kylehanks: Sharing my open source coding agent project https://github.com/qbit-ai/qbit
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1462855374527926355)** (1 messages): 

> `Sonar API, Website Indexing, Data Delay` 


- **Sonar API Data Delay Debacle**: A user reported a **24-hour delay** in the **Sonar API** updating with new website content due to indexing.
   - They inquired about speeding up website indexing or bypassing it entirely to receive data immediately after publication.
- **Sonar API Indexing Inquiries**: A user asked about reducing the indexing delay in the **Sonar API**, which currently takes about **24 hours** to update website content.
   - The user is looking for solutions to either speed up the indexing process or bypass it altogether for immediate data retrieval.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1461812744130003147)** (835 messages🔥🔥🔥): 

> `Claude Opus 4.5 rate limits vs Cursor token usage, GPT 5.2 codex performance and language issues, Cursor Debug mode and print statements, Prettier extension issues, Cursor usage limits and plan details` 


- **Google Pro offer beats Cursor's tokenomics**: A user with a free 12-month Google Pro subscription noted that **Claude Opus 4.5** only has rate limits, unlike Cursor where token usage can lead to significant costs.
   - Another member chimed in, saying they burn through **$200** in 3 days and were shocked by how expensive Opus is.
- **GPT 5.2 Codex has language mishaps**: Some members reported that **GPT 5.2 Codex** randomly switches to Arabic, making it unusable despite others claiming it's superior to **Opus 4.5**.
   - One user expressed frustration, stating, *I have never seen a model randomly change languages on me on a consistent basis*.
- **Cursor adds secret sauce with print statements**: A member inquired whether Cursor's insertion of print statements for debugging is an LLM function or Cursor's own mechanism.
   - Another member clarified it is Cursor's *secret sauce* (their Agent/Debug Mode), working natively without a custom MCP server, referencing [this blogpost](https://cursor.com/blog/debug-mode).
- **Prettier extension gets utterly broken**: Some members reported that the Prettier extension is completely broken, making it unable to format any file, and raised the issue [on GitHub](https://github.com/prettier/prettier-vscode/issues/3906#issuecomment-3761391774).
   - One suggested switching temporarily to Biome as a workaround.
- **Users debate Cursor's Usage Limits and Plans**: Some users expressed confusion about Cursor's usage limits and plan details. One user questioned why the program didn't dip into the "pool".
   - Members clarified that the $20/month plan includes a certain amount of credit but can be quickly exhausted and some found a *free bonus* from Cursor.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1461813287178998066)** (562 messages🔥🔥🔥): 

> `image-to-video slash command, privacy with PDF support, Claude Vision model data, GPT 5.2 Codex, rate limit for video models` 


- **Image-to-Video Slash Command Surfaces**: A user inquired about using the `/image-to-video` slash command, and another user provided a step-by-step guide on how to access the photo library after typing the command.
   - The user also expressed their appreciation for the updated UI, and the team was notified, mentioning it is *still an experiment and hasn't yet rolled out fully*.
- **PDF Support Sparks Privacy Probes**: A user questioned how the new PDF support would work with privacy, especially concerning confidential PDF documents, but was pointed to the [platform's Privacy Policy](https://help.lmarena.ai/articles/3765052346-privacy-policy) for details.
   - It was emphasized that the platform will still **scrub for PII** before any open data releases and that these practices remain unchanged, despite it still being an experimental feature.
- **Nano Banana Pro's Nagging No-Gos**: Users reported consistent issues with **Nano Banana Pro**, experiencing errors over extended periods, with a member noting they've been getting errors *every hour* since 5-6 PM the previous day and was given [steps to fix errors](https://help.lmarena.ai/articles/1645798556-lmarena-how-to-something-went-wrong-with-this-response-error-message).
   - Another user pointed out a potential **January 2025 date cutoff** for the model based on [minimaxir.com](https://minimaxir.com/2025/12/nano-banana-pro/#:~:text=Although%20Nano%20Banana%20Pro's%20cutoff,when%20it%20doesn't%20work.), while others reported problems with captcha.
- **Text Files Tease Techies**: Users are clamoring for the ability to **upload .txt files** for larger context windows, but were told by a community manager this is *something we're working on* and *is for sure on the list*.
   - PDF upload support has been implemented, some users are resorting to uploading databases within PDF files.
- **The Battle Isn't Over: Switch Stays!**: Some users are annoyed by the **random model battles** in direct chat, wanting it removed. It's been a long test, with one commenting *This was the worst implementation they ever did in lmarena*.
   - Others report improvements in response times and error rate fixes, expressing hope that the A/B testing will bring **Gemini 3 Pro** to the platform soon.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1461887996478492863)** (2 messages): 

> `Text-to-Image Leaderboard, Image Edit Leaderboard, flux.2-klein models, wan2.5-i2i-preview model` 


- **Text-to-Image Leaderboard sees New Models**: The [Text-to-Image Arena leaderboard](https://lmarena.ai/leaderboard/text-to-image) has been updated with `z-image-turbo` now ranking #22, `flux.2-klein-9B` at #24, and `flux.2-klein-4B` at #31 overall.
- **Image Edit Arena updated with Flux Models**: The [Image Edit Arena leaderboard](https://lmarena.ai/leaderboard/image-edit) has been updated, with `flux.2-klein-9B` ranking #15 and `flux.2-klein-4B` ranking #21.
   - For more details, check the [Leaderboard Changelog](https://lmarena.ai/blog/leaderboard-changelog/).
- **Wan2.5-i2i-preview Joins Image Edit Elite**: The [Image Edit leaderboard](https://lmarena.ai/leaderboard/image-edit) welcomes `wan2.5-i2i-preview`, securing the #21 spot with a score of **1213**.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1461812532015530004)** (450 messages🔥🔥🔥): 

> `LLMs vs AI, Gaming Laptop Model Recommendations, GPU Usage Monitoring, LFM 2.5 1.2B Model Performance, MedGemma Usage and Troubleshooting` 


- ****LLMs** considered **AI**?**: Members debate whether current **LLMs** qualify as **AI**, with some arguing that the term is misused due to differences in abbreviation meanings and actual capabilities.
- ****Qwen3 4B 2507** recommended for speed on gaming laptops.**: **Qwen3 4B 2507** is recommended for effectively running on gaming laptops with **8GB VRAM** and **16GB DDR5**, outperforming **LFM 2.5 1.2b** in terms of speed and overall performance.
- ****GPU Core Load** monitoring clarified using **Open Hardware Monitor**.**: Users discussed the annoyance that Windows Task Manager displays **GPU usage** as *'3D'*, not *'CUDA'* or *'GPU Core load'*, and **Open Hardware Monitor** is recommended for accurately displaying **GPU core load**.
- ****LFM 2.5 1.2B** Model's Exceptional Performance Claimed.**: Some users claim **LFM 2.5 1.2B** performs exceptionally well, even comparable to larger models, particularly in specific tasks like translation, citing the **SauerkrautLM-Translator-LFM2.5-1.2B** on [Hugging Face](https://huggingface.co/VAGOsolutions/SauerkrautLM-Translator-LFM2.5-1.2B).
   - Others dispute this, noting it messes up simple instruct tasks, and caution against overhyping its capabilities, urging to *'talk to your doctor to change the dose of meds'* if seeing the future from this one model.
- **Debugging **MedGemma**: How to troubleshoot installation issues.**: Users troubleshoot installation issues with **MedGemma**, including problems with image input and the model's prompt template, and discover that using non-English characters in file paths can cause issues.
   - One user recommended downloading the model via the LM Studio discover section rather than manually to ensure correct installation, using the [unsloth medgemma](https://huggingface.co/unsloth/medgemma-1.5-4b-it-GGUF).


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1461814587467895099)** (81 messages🔥🔥): 

> `VRAM donation request, AMD AI 9 370 & nVidia 5070 Optimization Advice, LLM capabilities and limitations, PCIE gen3x1 vs gen4x1 performance impact on LLMs, Bottlenecking RTX Pro 6000 on AM4 for LLM inference` 


- **Begging for VRAM Upgrade**: A member jokingly requested a **3090 donation** to reach **128GB of VRAM** for their setup, expressing their need via a [Squidward homelessness GIF](https://tenor.com/view/homeless-squidward-spare-change-gif-25810212).
   - Separately, another member lamented buying a laptop with an **AMD AI 9 370** and **NVIDIA 5070** with only **8GB VRAM**, seeking advice on model optimization.
- **LLMs aren't that capable**: According to one member, if you need something important done properly, correctly, or at all, *don't use an LLM* and links to [huggingface](https://huggingface.co/learn/llm-course/chapter1/1).
   - They note that keeping models and context in **VRAM** is important, and to not go below **Q4** quantization.
- **PCIE downgrade hits performance**: A member found that running a **3090** in a gen3x1 slot decreased inference performance from **120 t/s** in the x16 slot to **90 t/s** in the x1 slot, despite expectations.
   - They are planning to return the motherboard to upgrade to one with **gen4x1 slots**.
- **Discounted GMKtec AI Max 395 for Qwen 3 Next?**: A member shared an image of the [GMKtec AI Max 395 PC](https://cdn.discordapp.com/attachments/1153759714082033735/1462093863610089643/image.png?ex=69703c45&is=696eeac5&hm=ed3607660e1224cb00f4d3fee80f9d66eff34e73923dca35d81b9ff163d945c5), asking if it would be good for **Qwen 3 Next** given its MoE nature, but another members suggested it's probably too slow.
- **CUDA still lacking, Vulkan Surges Ahead**: One member noted that **llama.cpp** has a poor implementation for **Qwen3 Next** currently, so specs are somewhat irrelevant, noting on official build they don't surpass **30-35 t/s**.
   - Another member gets **60 t/s** using **Vulkan** with their **RTX PRO 6000**, compared to another's **38 t/s** after **CUDA** optimizations.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1461821922827571402)** (251 messages🔥🔥): 

> `GPU inference in production, AI hardware jobs in the cloud, FastAPI with predefined prompts, Agent Identity and context persistence, Directional information gain model` 


- **Earliest Time to Catch Misconfigured GPU Jobs**: Members are asking about what's the earliest point to catch **misconfigured jobs** when running **GPU inference** or **training in production**.
   - One member detects fails during the *first 50K step* of **pretraining**, and uses a simple inference script to check for issues between generate scripts and the inference engine.
- **Cloud Recommendations for AI Hardware Jobs**: A member asked for advice on which **cloud platform** would be best in terms of **price efficiency** for **AI hardware jobs**.
   - No specific recommendations were given, but members can help in the future if issues arise.
- **Self-Hosting FastAPI App on CPU**: A member wants to build a **FastAPI app** with predefined prompts to wrap around the **Vertex AI API** and self-host it in a Docker image, running on **CPU**.
   - It was recommended to use a **tiny gguf model** or a model no larger than **500M parameters** for better **CPU efficiency**; a **1B model** might require a **15-minute timeout** due to slow loading and inference times.
- **Navigating Agent Identity and Context Persistence**: A member seeks advice on handling **Agent Identity** and **context persistence**, aiming to persist predefined context/ontology for multiple agents in different roles, but needs something beyond just memory.
   - Another member suggested using just a **DB** and a **RAG pull in cpp**, although this approach might be considered *old school*.
- **Scoring Directional Sentence Relevance**: A member is experimenting with a **directional information gain model** to score whether sentence B adds functional value to sentence A, needing advice on dataset creation for directional relevance labels.
   - The goal is to teach directional relevance, capturing whether B resolves, explains, elaborates, or completes something about A, not just symmetric relevance.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1461929106437574756)** (56 messages🔥🔥): 

> `drone_fsd_dataset, CUDA learnings, SLMs for Indic languages, AI Recruiting Assistant, slipstream protocol` 


- **Drone Dataset Navigates Obstacles**: A user shared a dataset generated with the MIRROR IDE, showcasing a drone navigating a **60x60 room** with **15 static + 12 floating obstacles** after a single training run ([webxos/drone_fsd_dataset](https://huggingface.co/datasets/webxos/drone_fsd_dataset)).
- **CUDA Learning Journey Detailed**: A member shared their **20 days of CUDA learnings** packed into one read, focusing on parallelism, kernels, and memory ([Learning CUDA From First Principles](https://pub.towardsai.net/learning-cuda-from-first-principles-b6b6670319c8)).
- **Indic SLMs on a Mission**: A member introduced their mission to build **SLMs for Indic languages** (22+ Indian languages) focusing on agentic use cases, having distilled [XLMRoberta](https://huggingface.co/kkkamur07/hindi-xlm-roberta-33M) to a **33Mn parameter model** while retaining **98% accuracy**.
- **AI Recruiting Assistant Launched**: A member shared a new [Hugging Face Space](https://huggingface.co/spaces/19arjun89/AI_Recruiting_Agent) for an **AI Recruiting Assistant**, designed to automate candidate evaluation and cold email drafting with built-in bias mitigation and verification.
- **Slipstream Protocol Saves Tokens**: An independent researcher introduced **Slipstream**, a protocol that saves up to **82% of tokens** on inter-agent coordination spend, and shared articles and spaces related to the research ([Slipstream for Agent Communication](https://huggingface.co/blog/anthonym21/slipstream-for-agent-communication)).


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1461954351676522676)** (7 messages): 

> `Deep RL Course Unity3D SoccerTwos.exe, AI vs AI interface, LunarLander-v2 deprecated, MCP course certificate` 


- **Unity3D SoccerTwos.exe Usage**: A student in the **Deep RL Course** is seeking guidance on using the **Unity3D** tool **SoccerTwos.exe** in Unit 7, as its usage isn't covered in the course.
   - The student also noticed the **AI vs AI** section is missing from the interface and wants to showcase their trained model.
- **LunarLander-v2 issues**: A student encountered an error in **Unit 1** of the **deep RL course** when using the **LunarLander-v2** environment in Google Colab, as it's deprecated and suggested to use **LunarLander-v3** instead.
   - However, after submitting the model using **v3**, the progress wasn't updated, prompting the student to seek assistance.
- **MCP course certificate**: A user inquired about the possibility of still receiving a certificate for completing the **MCP course**.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1462144986949877893)** (17 messages🔥): 

> `PyTorch Profiler Usage, SLMs for Indic Languages, CUDA Project Feedback, NVidia Cutile Library` 


- **Profiling PyTorch and Indexing Performance**: A member sought advice on using the **PyTorch profiler** to understand why certain `index_select` and `index_copy` operations have high CPU wall time, linking to [relevant code](https://github.com/TheJDen/janestreet-gpu-mode-2025/blob/optims/optimizations/5_reduce_syncs/inference.py) for context.
   - They wondered if allocation issues might be the root cause and looked for methods to diagnose the problem from **profiling traces**.
- **Crafting SLMs for Indic Tongues**: A member is building **SLMs for Indic languages**, targeting models between **10Mn - 500Mn parameters** for efficient on-device agentic use cases, and has distilled [Hindi XLMRoberta to a 33Mn parameter model](https://huggingface.co/kkkamur07/hindi-xlm-roberta-33M).
   - They are seeking feedback and collaboration to build **world-class SLMs**.
- **Seeking CUDA Kernel Guidance**: A member requested feedback on their first **CUDA project**, implementing a causal self-attention kernel on a V100, aiming to surpass a naive PyTorch implementation and approach the performance of `scaled_dot_product_attention`.
   - They shared details on their approach, block configurations, and the challenges faced when incorporating shared memory and optimizing for **L1 cache usage**.
- **cuTile Library Mentioned for Tiling**: In response to a question, a member suggested using the recently released **cuTile library** for easier manual tiling in CUDA kernels, providing a generic interface for tiling and shared memory.
   - The library might help resolve **memory-bound problems**.


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1462778356067209393)** (2 messages): 

> `Triton Top-K Selection, GPU Array Processing` 


- **Triton Top-K Kernel Trouble**: A member is encountering errors with their [Triton kernel for top-k selection](https://leetgpu.com/challenges/top-k-selection) on a GPU array.
   - The provided code snippet utilizes `triton.jit` and `triton.language` for GPU-accelerated computation, specifically targeting the top-k selection problem, but the current implementation has an error.
- **Local Top-K vs. Global Top-K**: A member pointed out that the current Triton kernel performs a local top-k selection on each 128-sized slice rather than finding the top-k elements for the entire array.
   - The poster mentioned that the question on leetgpu.com requires finding the top-k elements from an array of up to a million elements, implying a **global top-k** operation.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1461847607025471722)** (28 messages🔥): 

> `Quack's cute.jit vs cute.kernel, BF16 Matmul Precision Issues, Torch's splitK, NaN aware instructions, SMEM buffer and wgmma size` 


- **Quack's CuteDSL Conundrums**: A member inquired about why a function in [Quack's reduce.py](https://github.com/Dao-AILab/quack/blob/main/quack/reduce.py#L15) is annotated with `@cute.jit` instead of `@cute.kernel`, clarifying it translates to `__device__` and `__global__`.
   - This member was attempting to make a valid correctness check for **bf16 matmul** compared to torch, facing precision issues and was just trying to find out if their kernel had any errors.
- **BF16 Precision Problems Plague Programmer**: A member debugged **BF16 matmul** precision issues, finding that a naive kernel produced results with a max abs difference of 1+ if large K.
   - Another member explained that *Torch is doing splitK*, and suggested computing the reference result in **fp32** instead, and scaling by `sqrt(K)` because **bfloat is just bad**.
- **NaN-aware Instructions Generated by CuteDSL**: A member noticed that CuteDSL seems to generate **NaN aware instructions** by default, unlike Triton, and inquired about disabling this behavior.
   - The instruction in question was `max.NaN.f32 %f71, %f70, %f69;`, though they did not receive an answer to their question. 
- **SMEM Size Sinkholes Speed**: A member observed a significant performance drop when the SMEM buffer size wasn't exactly equal to the wgmma size, leading to iterations over it doing wgmmas.
   - They hypothesized this was due to **lower occupancy caused by each warp group handling a larger tile size** and was told to ask in the cutlass channel.
- **NCU in the Clouds**: A member sought cloud providers that allow the use of **nsight compute (ncu)**, noting issues with vast.ai.
   - Another member recommended Lambda Labs, and also provided a [gist](https://gist.github.com/msaroufim/9e56ce5d42a5e9ccd5e938c83181ea47) with info on cloud providers that can work just not out of the box.


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1461832161157185597)** (1 messages): 

> `Smol Training Playbook, Loubna Ben Allal, Open Models` 


- **Smol Training Playbook Talk Scheduled!**: Loubna Ben Allal will be presenting her book, [The Smol Training Playbook: The Secrets to Building World-Class LLMs](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook#introduction).
   - A [YouTube video](https://www.youtube.com/watch?v=y9zOZHXo0eE) accompanies the book, providing more comprehensive information on training **open models**.
- **Book is a comprehensive reference**: A member noted that the book is a wonderful and comprehensive reference for those interested in open models.
   - The book may contain secret information for building world-class LLMs.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1462096620920373383)** (4 messages): 

> `NLA Resources, Async GRPO, PyDevTools Handbook, Flux2` 


- **New Numerical Linear Algebra Resources**: Collected resources to complement **Trefethen's** Numerical Linear Algebra textbook, including [lecture notes from Oxford](https://courses.maths.ox.ac.uk/pluginfile.php/105965/mod_resource/content/35/NLA_lecture_notes.pdf) and [Eric Darve's NLA content](https://ericdarve.github.io/NLA/content/solving_linear_systems.html).
- **YouTube NLA playlists for you!**: Shared a [YouTube video](https://www.youtube.com/watch?v=hn00PydWK_4) and [three playlists](https://www.youtube.com/playlist?list=PL05umP7R6ij2lwDdj7IkuHoP9vHlEcH0s) [on Numerical Linear Algebra](https://www.youtube.com/playlist?list=PLAVG7GMBpcYArR9QLXm3DVvqYhRdF6Tsj) [and more NLA](https://www.youtube.com/playlist?list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY).
- **Async GRPO in the Wild**: Linked to a Notion page titled *async-grpo-in-the-wild* at [yumoxu.notion.site](https://yumoxu.notion.site/async-grpo-in-the-wild).
- **PyDevTools Handbook released!**: Shared a link to the [PyDevTools Handbook](https://pydevtools.com/handbook/).
- **Flux2 C code released!**: Shared a link to the [Flux2](https://github.com/antirez/flux2.c) C code.


  

---


### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1462548679893323929)** (2 messages): 

> `Performance Engineers, Enterprise Pipeline, Vendor Stacks` 


- **Performance Engineers Wanted for Enterprise Pipeline**: A member is actively seeking **performance engineers** with experience across various **vendor stacks** for a robust **enterprise pipeline**.
   - The total compensation (TC) is offered at **750K-1M+**.
- **Inquiry About Job Description for Performance Engineer Role**: A member inquired about the details of the job description, expressing significant experience in both developing and debugging challenging problems in **enterprise stacks**.
   - The member has a *lot of work both developing and debugging hard problems in enterrpise stacks*.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1462257176021045481)** (10 messages🔥): 

> `Shared Memory Allocation, Thread Block Limitations, Warp Register Usage Granularity` 


- **Shared Memory's Block-by-Block Basis**: Shared memory is allocated **per block**, not per thread, with the size either known at compile time for static allocation or specified at runtime via the launch API.
   - Each thread in a block sees *an identical pointer to the L1 cache*, making it "shared" among the block, unlike register space which has a dedicated file for each thread.
- **Thread Block Occupancy Hindrances**: Having *a ton of* thread blocks can hurt GPU occupancy due to limits on the **max number of blocks a Streaming Multiprocessor (SM) can take at once** and **shared memory constraints**.
   - Additionally, **short-lived blocks** can incur a performance penalty due to the overhead of the Grid Scheduler.
- **Register File Partitioning and Warp Allocation**: Each Streaming Multiprocessor (SM) partition has a register file of **512x32x32bit**, accommodating **512 warp-wide (vec32) 32-bit registers**.
   - Each resident warp belongs to one of the four partitions and gets allocated a subset of registers from that partition, fixed from warp start to exit (until sm_89) and allowing limited dynamic reallocation from sm_90; a single warp can use at most **255 registers**.


  

---


### **GPU MODE ▷ #[jax-pallas-mosaic](https://discord.com/channels/1189498204333543425/1203956655570817034/1461812903723143319)** (1 messages): 

> `Mosaic Masked Load` 


- **Mosaic Misses Masked Load**: A user pointed out that **Mosaic** does not have a masked load feature, which necessitates loading **128 elements** at a time.
   - This differs from `tl.load`'s mask argument, offering more flexibility in other contexts.
- **Mosaic's Memory Access**: The user highlights that **Mosaic** requires loading **128 elements** at a time due to the absence of a masked load feature.
   - This contrasts with the masked load functionality available in `tl.load`, which allows for more selective memory access.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1462309200116846807)** (2 messages): 

> `High Velocity City, NVIDIA CUDA kernel writing, GenAI Learning Series` 


- **"High Velocity City" - Relative or Absolute?**: A member wondered whether the concept of a **"high velocity city"** is relative to the individual or an absolute measure.
   - The member pondered whether the definition of a **"high velocity city"** depends on personal perspective.
- **South Bay NVIDIA CUDA Study Group Forming**: A member is looking for learning buddies in the **South Bay** area to form a dinner/discussion/learning series.
   - The member wants to get into **NVIDIA CUDA kernel writing** with **GenAI**.


  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1462111962673578148)** (4 messages): 

> `Triton Puzzle Installation, Triton Version Compatibility, GitHub Issue Solutions` 


- ****Triton Puzzle install** encounters installation error**: A user, new to Triton, encountered an error while installing **Triton Puzzle** and sought help, sharing an image of the error.
   - The image showed a detailed view of the error encountered during the installation process.
- **GitHub fix to **Triton Puzzle** installation errors**: A member provided a direct [link](https://github.com/srush/Triton-Puzzles/issues/32) to a **GitHub issue** addressing the Triton Puzzle installation errors.
   - A member confirmed that this link should resolve the installation error reported by the user.
- **Pinning down **Triton version 3.2.0** fixes puzzle errors**: A member suggested changing the Triton version to exactly **3.2.0** using a provided [link](https://github.com/srush/Triton-Puzzles/pull/34/files).
   - This change is aimed to resolve reported errors when running the code.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1461816859299938376)** (13 messages🔥): 

> `Global Loads, DFVS Throttling, VMEM Operations, vmcnt Counter, Dynamic Voltage Frequency Scaling` 


- **Maximum VMEM Operations Limit Stalls Loads**: Hitting the maximum number of **vmem operations** in flight causes stalling until a previous operation completes, similar to **SMEM** and **LDS**.
   - A member asked what *'operation completes'* means in this context, wondering if it means when the data has arrived from source memory, and if there's any AMD documentation that mentions the precise value of the maximum number of loads in flight.
- **vmcnt Counter Size Limits VMEM Operations**: The size of the **vmcnt counter**, which tracks inflight vmem operations (loads, stores, writebacks), is **6 bits** which could be a hard upper limit.
   - One member noted that the **6-bit vmcnt counter** may limit the number of concurrent VMEM operations to **2^6 per wavefront**, while experiments showed only **18 load VMEM instructions** in flight using the *rocprof compute viewer tool*, suggesting the actual limit might be lower.
- **DFVS Might Be Throttling Performance**: It was suggested to check the clocks, as **DFVS (Dynamic Voltage Frequency Scaling)** might be throttling performance.
   - A member explained that **DVFS** scales back the clock speed if it’s using too much power, which decreases throughput.
- **Profiling VMEM Operations with rocprofiler**: Members suggested using **rocprofiler** with specific options (**--att-perfcounters** and ...**-ctrl**) to monitor in-flight vmem operations.
   - A [link to counter definitions in rocm-systems](https://github.com/ROCm/rocm-systems/blob/develop/projects/rocprofiler-sdk/source/share/rocprofiler-sdk/counter_defs.yaml#L4735) was provided, along with instructions to filter by CU using the derived counter syntax in the viewer.


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1462150855074513075)** (2 messages): 

> `Memory Efficient Attention (MEA), Apple Silicon (Metal) native Stable Diffusion, LLaMA models on Metal, Core ML Stable Diffusion Performance, MPSGraph custom kernels` 


- **MEA: More Excellent Attention?**: A member shared [Memory Efficient Attention](https://arxiv.org/abs/2406.14530) (**MEA**), a novel approach that achieves **O(n log n)** complexity and outperforms existing methods on long sequences.
   - They noted it could be *useful in situations where vLLM cannot be used*, and another speculated that the *scaling allows attention over entire books*.
- **Craiyon & New Apple Silicon support!**: The **Craiyon** team highlighted their work enabling native **Stable Diffusion** on **Apple Silicon** through [this blog post](https://www.craiyon.com/blog/apple-silicon-native-stable-diffusion/), achieving impressive speedups.
   - One member lauded the *huge engineering effort* in quantizing models to **4-bit** and **2-bit**, making it performant on Apple's hardware.
- **LLaMA on Metal: Fast!**: Users are reporting very fast speeds when running **LLaMA** models using **Metal**, with one noting a speed of *100 tokens/second* on an **M2 Max**.
   - They clarified that they had set `torch.compile` to True and were using the nightly build of PyTorch.
- **Core ML Stable Diffusion, performance improved!**: A member inquired about the performance of **Core ML Stable Diffusion** and linked to a [related GitHub issue](https://github.com/apple/ml-stable-diffusion/issues/302) indicating it's *not great*. 
   - Another responded saying they were achieving roughly *5 iterations/second* with the model.
- **MPSGraph: Custom Kernels, more power?**: One member is investigating using **MPSGraph** with custom kernels to overcome performance issues in **PyTorch**.
   - They are finding that it is much more complicated than CUDA, and are struggling with it so far.


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1462170866912858238)** (2 messages): 

> `GPU Mode Competition, Modal Integration, Leaderboard Instability, Profiling Limitations` 


- ****Dual GEMM** Competition Gets New Leaderboard**: A new “**final_nvfp4_dual_gemm**” leaderboard has been created for problem #3, with submissions eligible for prize money required by **January 20, 2026** [on the GPU Mode website](https://www.gpumode.com/v2/leaderboard/664?tab=rankings).
   - Past measurements were unstable due to unstable runners, and manual verification will be used if stability issues persist, but closing the old leaderboard would expose solutions, so it remains open temporarily.
- **Modal to the Metal for **Problem #3** and **#4****: A new leaderboard using **Modal** is now active for problem #3, requiring submissions via the “**modal_nvfp4_dual_gemm**” leaderboard with the “**B200**” GPU label [here](https://www.gpumode.com/v2/leaderboard/697?tab=rankings).
   - The switch to Modal aims to ensure reliable benchmark numbers, but it removes profiling support due to security policies and isolation requirements, suggesting users rent GPUs for profiling from vendors like **Prime Intellect**, **Verda**, or **Sesterce**.
- **Profiling Support Pulled Amidst Security Woes**: Profiling support using **ncu** could not be guaranteed by serverless platforms due to security isolation requirements, as **ncu** can potentially expose neighboring processes and leak model definitions.
   - Problems such as non-isolated jobs overusing resources, concurrent GPU jobs, dependency upgrades, and clock rate/thermal changes have plagued the competition due to limited SSH access for debugging.
- **Benchmark Stability Prioritized over Profiling**: The move to **Modal** prioritizes reliable benchmark results to safeguard the competition's integrity, despite losing profiling capabilities.
   - Dependencies for Modal are version controlled and available for review and modification via pull request [on GitHub](https://github.com/gpu-mode/kernelbot/blob/main/src/runners/modal_runner.py).


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/)** (1 messages): 

2kian: really cool adaptor once again!
  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1462872902654496882)** (8 messages🔥): 

> `Triton Performance using CuteDSL, NAN based F32 max in PTX, SMEM to RMEM loads, Matching Triton’s Softmax, CuteDSL generated SASS code` 


- **CuteDSL Performance Quest Falls Short of Triton**: A member is attempting to match **Triton** performance using **CuteDSL** ([github.com/FL33TW00D/submarine](https://github.com/FL33TW00D/submarine/pull/5/files)) but is currently **2us** short.
   - They are looking into the generated PTX code and found that **CuteDSL** seems to generate **NaN** aware instructions by default like `max.NaN.f32`, and asks if there's a way to disable this behavior.
- **Non-NaN Version Doesn't Drastically Improve Perf**: A member advised the original poster to look at the **SASS** instead of **PTX** because from their experience changing from `max.NaN.f32` to a non-NaN version didn't noticeably change the performance.
   - They suggested using **NCU + GPT** to help find documentation of **SASS** instructions.
- **Digging Into SASS Reveals Nuanced Picture**: The original poster dug into the **SASS** and stated that it revealed a more nuanced picture.
   - They have a question about how to keep `mO_cur` aligned in the same way as `mO` when using `cute.domain_offset`.
- **Triton Remains Champion**: A member asked the original poster to share the key trick if they end up matching/beating **Triton's softmax**.
   - The original poster responded that **Triton has won this round**.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1462921532593410313)** (1 messages): 

> `2-GPU NVLink AllReduce Library, NVIDIA NCCL performance comparison, Tail Latency Stability` 


- ****Yali** Library Claims Superior Performance!**: A member is seeking testers for **Yali**, a [2-GPU NVLink AllReduce library](https://github.com/Venkat2811/yali) which they claim outperforms **NVIDIA NCCL** by **1.2x-2.4x**.
   - They highlight that **Yali** provides *50x+ more stable tail latency* using principles from high-performance systems in GPU communication.
- **Testers wanted for faster 2-GPU NVLink**: A member is seeking testers for **Yali**, a [2-GPU NVLink AllReduce library](https://github.com/Venkat2811/yali).
   - They highlight that **Yali** provides *50x+ more stable tail latency* using principles from high-performance systems in GPU communication.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1461812565771423968)** (119 messages🔥🔥): 

> `Slow Runner Issue, Modal Migration, Benchmarking Overhead, Rate Limiting, CUDA Out of Memory` 


- **Runner Performance Slowdown Troubles Submissions**: Users reported that their kernel runtimes increased from **14.x us** to **22.x us** after resubmitting code, specifically on runner `b200-02-gpu4-runner`, which has been flagged as slow, impacting submission performance.
   - A user suggested a temporary workaround of launching from multiple terminals to avoid the slow runner, though this may clog up GPUs, as noted by <@arseniivanov>.
- **Modal Migration Addresses Runner Inconsistencies**: Due to issues with runner performance consistency and measurement stability on the current NVIDIA hardware, the competition is moving to **Modal** for problems #3 and #4, with ongoing benchmark number showing ~2x slowdown.
   - While **Modal** has some variations, it's expected to be more stable, though it does not allow NCU profiling; the move is detailed in [this Discord post](https://discord.com/channels/1189498204333543425/1343350424253632695/1462268408686182480).
- **Performance Benchmarking Shows Overhead Woes**: Members noted an extra **0.5 us** overhead issue remains unfixed on the new leaderboard, while others observed their kernel runtimes increasing, suggesting broader overhead issues.
   - In particular, submission **369747** had a time of **31us** compared to submission **369709** at **20us**, despite running the same code.
- **Modal Overload Requires Rate Limits**: With over **5.2K jobs** submitted after opening up **Modal**, and job durations ranging from 1 to 4 minutes, long queue times are expected, leading to the need for rate limits after problem 3 to manage costs, which are already at **$2K**.
   - An alternative to rate limiting was proposed: periodically running only the latest kernel per user, skipping unchanged ones, but this was viewed as a form of rate limiting by <@austin362667>.
- **CUDA Memory issues haunt jobs**: Users are occasionally encountering **CUDA Out of Memory** errors, potentially due to memory not being properly cleaned up between runs, and which suggests re-submission as a possible workaround.
   - This issue doesn't significantly affect measurements if a different container is obtained upon re-submission according to <@s1r_o>.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1461877896665759886)** (95 messages🔥🔥): 

> `Gregor Zunic tweet visibility, Sulaiman Ghori xAI interview, Vercel 'skills' launch, GPT 5.2 Pro Erdos problem, GPT-5.2 Performance` 


- **Tweet by Gregor Zunic Gets Mad Traction**: A tweet by **Gregor Zunic** on January 16, 2026, gained significant traction, amassing **119,777 views** and over **760 likes**.
   - The social media engagement analysis for [@gregpr07](https://x.com/gregpr07/status/2012052139384979773?s=46) highlights the tweet's unexpected visibility.
- **Sulaiman Ghori Spills xAI Secrets**: **Sulaiman Ghori**, an xAI technical staff member, discussed the rapid development of the **Colossus data center**, the company's hiring philosophy, and the intensity of working under **Elon Musk** in [this interview](https://x.com/ti_morse/status/2011913655793918097?s=46).
   - Shortly after this interview, drama ensued as Ghori apparently *lost his works at xai checkmark on twitter* and [deleted heaps of tweets](https://x.com/sulaimanghori/status/2013261823475097732) of him getting fired. 
- **Vercel Opens 'Skills' as AI Agent Toolkit**: **Guillermo Rauch** announced the launch of **'skills'**, an open and agent-agnostic ecosystem designed as a package manager for AI capabilities on [Vercel](https://xcancel.com/rauchg/status/2012345679721771474?s=46).
   - Users can start integrating these tools using the command **'npx skills i vercel-labs/agent-skills'**; more details on [React Best Practices](https://vercel.com/blog/introducing-react-best-practices).
- **GPT 5.2 Pro Pulls Math Trick**: **GPT 5.2 Pro** has successfully solved the previously open **Erdos problem #281**, according to [Neel Somani](https://xcancel.com/neelsomani/status/2012695714187325745).
   - Mathematician **Terence Tao** noted this achievement as *a clear instance of artificial intelligence solving an unsolved mathematical problem*.
- **GPT-5.2: Scaling Challenges and Performance Insights**: **Lee Robinson** shared research insights on long-running AI agents, noting that while **GPT-5.2** is significantly more capable, systems are not yet production-ready in [this tweet](https://xcancel.com/leerob/status/2012938056043565333?s=46).
   - Key findings include the importance of **prompt engineering** over complex distributed systems architectures, and that **traditional software patterns** hinder agent performance compared to new, specialized patterns.


  

---


### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/)** (1 messages): 

sarav1n: What are you running? Shouldnt have issues with apple silicon.
  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1462020611328184390)** (12 messages🔥): 

> `ElevenLabs Valuation, Levelsio E-Girl Vlog, HeartMuLa AI Music Model` 


- **ElevenLabs Eyes Sky-High $11bn Valuation**: AI startup **ElevenLabs** is in talks to raise funding at an **$11 billion valuation**, a significant jump from **$6.6 billion** just months ago, detailed in [this post](https://x.com/sebjohnsonuk/status/2012277025629696162).
- **Levelsio's 'E-Girl' Character Swap Strategy**: **Levelsio** released a video blog post adopting an '*e-girl*' persona to explain the mechanics of character swaps, linked [here](https://x.com/levelsio/status/2012943783424393356).
- **HeartMuLa: New Music Model Beats the Competition**: **HeartMuLa**, a new open-source music generation model using an LLM-based approach, boasts multi-modal inputs and section-specific styling, allegedly outperforming **Suno v5** and **Udio v1.5** in lyrical clarity, as highlighted [in this thread](https://x.com/wildmindai/status/2013179426901512419).


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1461906656580603975)** (67 messages🔥🔥): 

> `Zero Knowledge Proofs for AI Governance, Trusted Execution Environments (TEE) Limitations, Learning Rate Scaling and Batch Size, Directional Information Gain Model, Open Source Coding Agents` 


- ****ZKPs Shield AI Governance****: A member proposed using **Zero Knowledge Proofs (ZKPs)** for autonomous AI governance, enabling verification of compliance without revealing sensitive information; imagine proactive regulation with maintained privacy.
   - Another member cautioned that while ZKPs can prove the existence of a proof, they don't inherently solve the problem of formalization and statement proving and ZKPs can prove that the model you wanted to run is actually the model that was executed.
- ****TEE Isn't Always Trouble-free****: Discussion revolved around the limitations of **Trusted Execution Environments (TEEs)** for secure compute, citing potential vulnerabilities even with hardware-based memory encryption.
   - One member mentioned that despite security features, TEEs can be compromised, referencing **DefCon talks** about exploiting vulnerabilities by intercepting decryption codes between the RAM and chip, but that **Nvidia's** new server has server level TEE which helps with it.
- ****Scale Learning Rates Like A Pro****: A member inquired about the consensus on **learning rate scaling** as a function of **batch size**, referencing [a paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/32ac710102f0620d0f28d5d05a44fe08-Paper-Conference.pdf) advocating for learning_rate ∝ sqrt(batch_size).
   - Others noted that linear scaling is common but often tuned, questioning the necessity of a strict rule.
- ****Directional Data Dynamics Drive Dataset Design****: A member sought advice on creating a dataset for a **directional information gain model**, aiming to score whether sentence B adds functional value to sentence A, not just similarity.
   - Suggestions included exploring **RAG**, **re-rankers**, and **knowledge graph research** to verify relevant information, as directional relevance depends on the specific example/scenario.
- ****Agentic Answers Await: Devstral vs. Codex****: A member asked for open-source coding agents for self-hosted models, and it was suggested that **Devstral 2 Small** is a good option.
   - Also it was pointed out that Devstral 2 Medium is apparently on par with **Claude Sonnet 4.5**, for agentic code base tasks (like GPT Codex), and that Kilo Code is just an extension that can plug in local models (such as locally hosted Devstral 2).


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1462897055906267383)** (13 messages🔥): 

> `Cold Read of arXiv Paper, Time Zone Differences, Discord Event for Paper Discussion` 


- **Cold Read Event Scheduled for arXiv Paper**: A member scheduled a cold read of an [arXiv paper](https://arxiv.org/abs/2512.24601) and invited others to react with a thumbs up if interested.
   - An event was then created [on Discord](https://discord.gg/kQQQWWte?event=1462918272335741049) and people were invited to hit interested if they plan to attend.
- **Time Zone Troubles Hinder Participation**: One member expressed they would likely be sleeping during the scheduled cold read due to being in the **Central European UTC** time zone.
   - Another member acknowledged the time zone difference and the impossibility of moving the event to a more convenient time for everyone.
- **Starting Now! Paper Discussion Begins**: A member announced the start of the paper discussion, twice.
   - N/A


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1462103080601522217)** (10 messages🔥): 

> `Qwen team at Alibaba, Snake oil salesman SAM, Anthropic's knowledge bases for Claude, Gaussian Splatting Music Video, Anthropic's Assistant Axis` 


- **Alibaba's Qwen Team Leadership**: Justin Lin, the tech lead of the **Qwen team at Alibaba**, shared [a link](https://x.com/JustinLin610/status/2012533831837143204) related to their work.
   - The shared content provides insights into the technical developments and strategies being employed by **Alibaba's Qwen team**.
- **Snake Oil SAM Sales Allegations**: Members discussed allegations of *snake oil* tactics by **SAM**, suggesting that many original key figures have departed due to these issues.
   - This discussion implies a critical view of certain leadership or business practices within the AI community.
- **Anthropic Develops Knowledge Bases for Claude**: A link to [testingcatalog.com](https://www.testingcatalog.com/anthropic-works-on-knowledge-bases-for-claude-cowork/) was shared, indicating **Anthropic's work on knowledge bases for Claude**.
   - This suggests efforts to enhance **Claude's capabilities** by providing it with structured knowledge resources, possibly for improved performance and reliability.
- **A$AP Rocky's Splatting Music Vid**: A member shared a [music video by A$AP Rocky](https://radiancefields.com/a-ap-rocky-releases-helicopter-music-video-featuring-gaussian-splatting__._astro_.__) featuring **Gaussian Splatting**.
   - This highlights the innovative use of **Gaussian Splatting** in creative and artistic projects.
- **Anthropic's Assistant Axis**: A link was shared to [Anthropic's research on Assistant Axis](https://www.anthropic.com/research/assistant-axis).
   - This research likely explores the design and functionality of AI assistants, focusing on different aspects and dimensions of their capabilities.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1462086557656678565)** (82 messages🔥🔥): 

> `Optimizing skill.md via dspy, DSPy 3.1.2 Release, Recursive Language Models (RLMs), Deno for local sandbox/interpreter, GEPA and RLM composition` 


- **Optimize skill.md with DSPy, anyone?**: A member inquired about optimizing `skill.md` (essentially a prompt) using **DSPy**, referencing a related [article on optimizing Anthropic skills](https://instavm.io/blog/anthropic-skills-can-be-optimized-using-dspy).
   - The user was seeking strategies for writing efficient `skill.md` files and whether anyone had attempted optimization using DSPy.
- **RLMs Drop, DSPy 3.1.2 is LIVE!**: The team just released **`dspy.RLM`** in **DSPy 3.1.2**, greatly expanding the capabilities achievable in a single line of code.
   - A team member cryptically promised this release during the DSPy 3.0 release talk back in June, and shared a link to the [announcement](https://x.com/isaacbmiller1/status/2013371005960401327).
- **Deno Chosen for Local WASM Runtimes**: DSPy uses **Deno** for its local sandbox/interpreter because of its secure **WASM runtime** capabilities.
   - The selection of Deno was inspired by [Simon Willison's blog post](https://til.simonwillison.net/deno/pyodide-sandbox) and integrates well with **Pyodide**.
- **Genetic-Pareto (GEPA) & RLMs: A Match Made in Heaven?**: **GEPA (genetic-pareto)** and **RLMs** are composable, with potential for **RLM-as-an-optimizer** strategies.
   - A team member considers **GEPA** a fundamental idea and highlighted an application of **RLMs** for writing documentation from code, citing its ability to handle extremely long outputs.
- **Dump Those Docs Using RLMs!**: Members discussed using **RLMs** to generate documentation from code, a task that has been impossible before.
   - It was noted that you could do it over all prior proposals, and you could keep the whole tree in mind.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1461878333309583381)** (67 messages🔥🔥): 

> `Increase App Size, Subscription Error, Manus AI Meeting Minutes, Billing Issues, Earn Credits` 


- **Request for More App Size**: A member requested an increase in the maximum app size limit on Manus, citing limitations encountered when trying to create an audio player app with **100 MP3 files totaling 600 MB**.
   - They hope that *enabling larger applications* will open the door to new possibilities and richer projects for developers and users alike.
- **User Encounters Subscription Error**: A member reported encountering a payment overdue error with a higher amount, preventing them from downgrading their plan.
   - A Manus Support member replied to the user and offered private assistance in resolving the subscription error.
- **Manus AI Meeting Minutes Feature**: A member shared a [YouTube video](https://youtu.be/pWShEX0Bn2Q) on how to use the new **Manus AI Meeting Minutes** feature.
   - A member jokingly commented that *Home office bros will love this*.
- **Billing Issues Cause Project Offline**: A member reported their project being offline due to billing issues and the inability to downgrade their plan from **$400**, which is causing distress as their Bible study platform for women is inaccessible.
   - Manus support has reached out to them privately for assistance.
- **DracoAI Touted as Superior Alternative**: A member claimed that [dracoai.app](https://dracoai.app) is *next level* compared to Manus, highlighting its ability to make **API calls**, including calling their phone.
   - They said to *Edit the system prompt and add specific API tools this thing is next level*.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1462678230648750293)** (27 messages🔥): 

> `Discord Rules, Green V2 Blackwell, tinygrad twitter, tinygrad Logo, MLPerf` 


- **Raising Green V2 Blackwell**: A user tried to contact the tinygrad team to help a friend raise money for **Green V2 Blackwell**, but was warned that *shilling = ban*.
   - George Hotz clarified that *this discord is for discussion of tinygrad usage*, and forbade users from trying to fundraise.
- **tinygrad needing a new logo**: George Hotz asked if *someone want to make a new logo for it?* because the current logo is old on the [tinygrad twitter](https://twitter.com/__tinygrad__).
   - George Hotz updated the github logo and requests to grab the logo from [tinygrad.org](https://tinygrad.org) which is SVG so it should scale to anything.
- **tinygrad Meeting #3 Upcoming**: A new meeting **#3** was announced to be held **Monday 9am San Diego time** with random order.
   - The meeting agenda includes topics such as *company update, drivers, image dtype, assembly, jit asserts, assign, mypy, llama training, viz / fast gemm, other bounties.*
- **tinygrad wants to hold contests this year**: George Hotz said that *I want to hold some contests this year after we get **405b mlperf***.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1462906510551290060)** (4 messages): 

> `tinygrad with pyarrow/parquest, Tensor.from_blob, zero_copy_only` 


- **Explore tinygrad's Compatibility with PyArrow and Parquet**: A user inquired about using **tinygrad** with **PyArrow/Parquet**, providing an example of loading data with `ds.dataset` and seeking a better alternative to `Tensor.from_blob`.
   - A proposed solution suggested using `Tensor.from_blob` with **PyArrow**, showcasing its usage with a Pandas DataFrame and a PyArrow array, but noted that it's *not well tested and maintained* and converting to **numpy** first is preferred.
- **`Tensor.from_blob` Usage in tinygrad**: A member shared a code snippet demonstrating the usage of `Tensor.from_blob` with **numpy** and **pyarrow** arrays, including assertions to validate the results.
   - The code covers creating a **tinygrad Tensor** from a numpy array and a pyarrow array, asserting that the resulting tensors match the original data.
- **Zero-Copy Data Loading into tinygrad**: A user replaced their initial data loading approach with a solution using **numpy** stacking after setting `zero_copy_only=False` to handle a nested array.
   - They noted that using `zero_copy_only=False` returns a nested array (`array([array([1., 2., 3.], dtype=float32), array([1. , 1.1, 1. ], dtype=float32)], dtype=object)`) and that the speed is similar to `from_blob`, clocking in at **24s** for **1.5 million rows** with file reads.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1461818147794587820)** (27 messages🔥): 

> `K2-Thinking context length issues, Distilled models from Claude/GPT-5/Gemini-3, Kimi subscription and billing issues, Missing phrases in Kimi, Kimi refund request` 


- **Distilled Models Spark Anticipation**: Members are waiting for models distilled from **Claude/GPT-5/Gemini-3** in the coming months, and improvements to long-context processing.
   - One member feels that **K2-Thinking's** context handling is poor after 30k tokens, and that most models only perform well on a fraction of their advertised context window.
- **Subscription Issue Frustrates User**: A member reported being charged despite cancelling their **$0.99 plan** and deleting their **Kimi** account, and is receiving repeated attempted charges on their Visa card.
   - Another member suggested that they DM their email address to resolve the issue, while another pointed them to **membership@moonshot.ai** for refunds.
- **User Reports Unexpected Subscription Deduction**: A member reported a **$19** deduction for their **Kimi** plan without prior reminder, after not using the account and requested a refund.
   - Support guided them to email membership@moonshot.ai for a refund and they confirmed receiving a response.
- **Phrases Go Missing**: A user shared an image showing that common phrases are missing, asking why they got rid of them.
   - Another member replied that they are under "presets" under the plus sign and showed an image where it's renamed.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1461860276537393358)** (7 messages): 

> `Raspberry Pi AI HAT+, Hailo AI chip support in Mojo, FaceNet alternatives, Commercial face recognition models` 


- **Raspberry Pi gets AI HAT+ for GenAI**: The [Raspberry Pi AI HAT+](https://www.raspberrypi.com/news/introducing-the-raspberry-pi-ai-hat-plus-2-generative-ai-on-raspberry-pi-5/) has been introduced, prompting questions about **Hailo AI chip support** in **MAX** and **Mojo**.
- **Mojo ❤️ Raspberry Pi AI HAT+?**: A member inquired about the possibility of adding **Hailo AI chip support** into **MAX** and **Mojo**, envisioning students easily picking up the **MAX platform** and **Mojo language** with **Raspberry Pi AI HAT+** support.
   - Another member expressed hope for such integration, driven by the motivation to create an end-to-end system for training a neural network and deploying it on a robot.
- **Mojo Needs Open Compiler for Hailo Integration**: According to one member, without an open-source compiler (or at least an open IR to hand to a compiler), it would be difficult for **Mojo** to integrate **Hailo**.
   - They noted that **AMD's NPUs** have a similar problem.
- **Face Recognition Model Recommendations Needed**: A member is seeking recommendations for commercially usable face recognition models/repos, as **FaceNet** fails in real-world cases, especially with changing lighting conditions and varying facial features.
   - They also inquired about alternatives to **FaceNet** that have worked better in production, and proven ways to improve robustness (lighting invariance, preprocessing, training tricks).


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1462876872198066430)** (14 messages🔥): 

> `PyTorch import issues, Numpy import issues, pixi shell issues, Python module import` 


- **PyTorch and Numpy import problems turn into Pixi Hell**: A member encountered import issues with **PyTorch** and **Numpy** after installing them using *pixi*, receiving an error message and being *unable to locate module* despite successful installation.
   - A helper pointed out that PyTorch is called `pytorch` on conda and further clarified the need to use the [Python module](https://docs.modular.com/mojo/std/python/) to access Python libraries within Mojo, rather than directly importing Python code.
- **Pixi shell confusion baffles newbie**: The member confirmed being in the **pixi shell**, with modules in the `.toml` file, but they weren't being recognized in the **Mojo** file.
   - The helper clarified that direct importing of Python code isn't possible in Mojo, and the [Python module](https://docs.modular.com/mojo/std/python/) or custom **cpython bindings** are necessary.
- **Solution found through Python module import**: The member confirmed that importing the module was successful, and plans to test it, expressing gratitude to the helper.
   - The helper humbly acknowledged that having been present during the module's creation aided in remembering its usage.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1462216317535911956)** (19 messages🔥): 

> `Aider ComposerLooks, Paul Gauthier's whereabouts, Aider's future development` 


- **ComposerLooks lacks discussion**: A member inquired about **Aider ComposerLooks**, noting its significant number of stars but limited discussion.
   - The original poster was curious about real-world use cases and support for the latest AI models, acknowledging it might not currently work and expressing the need to consult the documentation.
- **The Mystery of the Missing Main Dev**: Members wondered about the whereabouts of **Paul Gauthier**, the primary developer of Aider, noting his last activity was in January.
   - Speculation arose that he might have been hired by **Anthropic**, given the similarities in code to **Claude**, eliminating open-source competition.
- **Aider open for community assistance**: Paul Gauthier confirmed he's been busy with other projects and open to merging community contributions.
   - A member inquired about missing features beyond autonomous agent capabilities, but the original poster noted that it was feature complete.
- **Abandonware Aversion Appears**: A member expressed reluctance to invest in Aider if it's considered **abandonware**.
   - This highlighted concerns about the project's maintenance and future development without active contributions from the original developer.


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1462552641694597211)** (1 messages): 

> `Production-ready LLM & RAG systems, AI-powered search, Summarization, Integrate LLM + RAG pipelines` 


- **Production-Ready LLM & RAG Systems Turnkey**: A member turns ideas and messy data into **production-ready LLM & RAG systems**.
   - They focus on making AI usable in real workflows, not just demos.
- **Expert Helps Integrate LLM + RAG Pipelines**: A member offers to help developers who want to **integrate LLM + RAG pipelines** into production without the trial-and-error.
   - They also offer to help indie builders or consultants needing guidance to make AI tools actually work.


