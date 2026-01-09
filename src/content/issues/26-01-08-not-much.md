---
id: MjAyNi0w
title: not much happened today
date: '2026-01-08T05:44:39.731046Z'
description: >-
  **Stanford paper** reveals **Claude 3.7 Sonnet** memorized **95.8% of Harry
  Potter 1**, highlighting copyright extraction risks compared to **GPT-4.1**.
  **Google AI Studio** sponsors **TailwindCSS** amid OSS funding debates.
  **Google** and **Sundar Pichai** launch **Gmail Gemini 3** features including
  AI Overviews and natural-language search with user controls. **Alibaba Qwen**
  releases **Qwen3-VL-Embedding** and **Qwen3-VL-Reranker**, a multimodal,
  multilingual retrieval stack supporting text, images, and video with
  quantization and instruction customization, achieving strong benchmark
  results. **Z.ai** goes public on HKEX with **GLM-4.7** leading the Artificial
  Analysis Intelligence Index v4.0, showing gains in reasoning, coding, and
  agentic use, with large-scale MoE architecture and MIT license.
  **Falcon-H1R-7B** from TII targets efficient reasoning in smaller models,
  scoring 16 on the Intelligence Index. **AI21 Labs** introduces **Jamba2**, a
  memory-efficient enterprise model with hybrid SSM-Transformer architecture and
  Apache 2.0 license, available via SaaS and Hugging Face. **vLLM** shows
  throughput improvements in inference and kernel engineering. *"Embeddings
  should be multimodal by default,"* notes Justin Lin.
companies:
  - stanford
  - google
  - google-deepmind
  - alibaba
  - z-ai
  - tii
  - ai21-labs
  - huggingface
models:
  - claude-3-7-sonnet
  - gpt-4-1
  - gemini-3
  - qwen3-vl-embedding
  - qwen3-vl-reranker
  - glm-4-7
  - falcon-h1r-7b
  - jamba2
topics:
  - copyright-extraction
  - multimodality
  - multilinguality
  - retrieval-augmented-generation
  - model-architecture
  - mixture-of-experts
  - model-quantization
  - reasoning
  - inference
  - kernel-engineering
  - memory-optimization
  - enterprise-ai
people:
  - sundarpichai
  - justinlin610
---


**a quiet day**

> AI News for 1/7/2026-1/8/2026. We checked 12 subreddits, [**544** Twitters](https://twitter.com/i/lists/1585430245762441216) and **24** Discords (**204** channels, and **4649** messages) for you. Estimated reading time saved (at 200wpm): **415 minutes**. **Our new website** is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on [@smol_ai](https://x.com/Smol_AI)!




---

# AI Twitter Recap


**Top tweets (by engagement)**

- **Stanford paper on LLM memorization & copyright extraction**: Summary claims copyrighted text can be extracted from multiple frontier models; notably asserts **Claude 3.7 Sonnet reproduced 95.8% of Harry Potter 1** in their setup; contrasts with much lower figure for GPT-4.1 ([ednewtonrex](https://twitter.com/ednewtonrex/status/2009201019184415218)).  
- **Google sponsorship of TailwindCSS**: Google AI Studio announces it is now a sponsor of **tailwindcss**, framed as ecosystem support after OSS funding controversy ([OfficialLoganK](https://twitter.com/OfficialLoganK/status/2009339263251566902)).  
- **Gmail “Gemini era” launch**: Google and Sundar Pichai announce Gmail features powered by **Gemini 3**—AI Overviews, AI Inbox, writing assistance, and natural-language search—user-controllable toggles emphasized ([Google](https://twitter.com/Google/status/2009265269382742346), [sundarpichai](https://twitter.com/sundarpichai/status/2009291313888547131), [Google](https://twitter.com/Google/status/2009266902112104711)).  
- **Qwen multimodal retrieval release**: Alibaba Qwen ships **Qwen3-VL-Embedding** and **Qwen3-VL-Reranker** (multimodal, multilingual, open-source) aimed at retrieval/RAG over text+images+video ([Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/2009264754917863924)).  
- **Z.ai / GLM milestone + IPO moment**: Z.ai announces it is now public on HKEX and runs a community challenge; GLM-4.7 remains central to the narrative ([Zai_org](https://twitter.com/Zai_org/status/2009290783678239032)).  

---

**Open-weight models: GLM-4.7 momentum, Qwen multimodal retrieval, and smaller “efficient” reasoning entrants**

- **GLM-4.7 (open weights) tops Artificial Analysis Intelligence Index v4.0**: Artificial Analysis reports **GLM-4.7 [Reasoning] = 42** (up from GLM-4.6’s 32), with strong gains across **coding, agentic use, and scientific reasoning**, plus **GDPval-AA ELO 1193** (highest among open weights they evaluated). Details include **200K context**, **text-only I/O**, **355B MoE total / 32B active**, **MIT license**, and practical deployment note: **~710GB BF16** weights means it won’t fit on a single 8×H100 node (~640GB) ([ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/2009117037667422457)). Z.ai also posts a longer “journey” reflection and later a public-market milestone/community challenge ([Zai_org](https://twitter.com/Zai_org/status/2009154193244721326), [Zai_org](https://twitter.com/Zai_org/status/2009290783678239032)).  
- **Qwen3-VL-Embedding + Qwen3-VL-Reranker (multimodal retrieval stack)**: Qwen introduces a **two-stage retrieval architecture** (embedding model + reranker) built on Qwen3-VL, handling **text/images/screenshots/videos/mixed inputs**, **30+ languages**, configurable embedding dims, instruction customization, and quantization for deployment. Alibaba positions it as SOTA on multimodal retrieval benchmarks and ships via Hugging Face/GitHub/ModelScope, with an Alibaba Cloud API “coming soon” ([Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/2009264754917863924)). Community amplification: benchmark callouts for **MMEB-V2 (77.9%)** and **MMTEB (67.88%)** ([HuggingPapers](https://twitter.com/HuggingPapers/status/2009295485966672072)); Justin Lin notes extension from VL to VL embeddings and argues embeddings “should be multimodal by default” ([JustinLin610](https://twitter.com/JustinLin610/status/2009277701727637785)); vLLM adds support in nightly builds ([vllm_project](https://twitter.com/vllm_project/status/2009316281275830351)).  
- **Falcon-H1R-7B (TII) enters the “small reasoning” lane**: Artificial Analysis highlights **Falcon-H1R-7B** as a UAE entrant with **hybrid Transformer-Mamba** positioning, scoring **16** on their v4.0 Intelligence Index among <12B models and notes performance on **Humanity’s Last Exam**, **τ²-Bench Telecom**, and **IFBench**, plus a mid “openness” score (44) on their new Openness Index framing ([ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/2009343487855219171)).  
- **AI21 Jamba2 (memory-efficient enterprise model family)**: AI21 announces **Jamba2**, emphasizing “enterprise reliability/steerability,” **hybrid SSM-Transformer**, KV-cache innovations, and **Apache 2.0** licensing, with availability via AI21 SaaS and Hugging Face ([AI21Labs](https://twitter.com/AI21Labs/status/2009259475643846978)).  

---

**Inference + kernel engineering: vLLM throughput wins, KV offloading, and “AI-written” kernels**

- **vLLM on B200 reaches reported 16k TPS**: vLLM highlights a community-reported milestone of **16k tokens/sec** on **NVIDIA B200** ([vllm_project](https://twitter.com/vllm_project/status/2009196819331600648)).  
- **KV Offloading Connector (IBM Research) lands in vLLM**: New connector asynchronously offloads **KV cache to CPU RAM** to handle preemptions and boost concurrency; vLLM claims **up to 9× throughput** improvement on H100 and **2×–22× TTFT reductions** for cache hits. They also describe host-device transfer optimizations (contiguous physical blocks enabling high-speed async DMA) and provide CLI flags (`--kv_offloading_backend native ...`) plus a deep-dive blog link ([vllm_project](https://twitter.com/vllm_project/status/2009217642507477222), [vllm_project](https://twitter.com/vllm_project/status/2009217645946773534), [vllm_project](https://twitter.com/vllm_project/status/2009217648224247840)).  
- **Kernel LLM / “Oink” fused RMSNorm kernel shows 40% kernel speedup**: Mark Saroufim describes early results from an AI-generated fused RMSNorm kernel integrated into vLLM, reporting **~40% speedup vs existing RMSNorm kernel** and **~1.6% end-to-end** improvement. The interesting engineering angle: the generated code attempts to embed a heuristic autotuner-like strategy specialized to hot shapes (e.g., 7168 BF16) and uses tricks like “direct GMEM” loads with SMEM used only for reduction—while acknowledging increased complexity and stability risk (segfault conditions, cluster launch interactions). He frames system-level eval suites (vLLM, FlashInfer’s approach) as a path to “SOTA AI kernels” despite determinism/testing concerns ([marksaroufim](https://twitter.com/marksaroufim/status/2009096176789016600)).  
- **Python-authored kernels via Keras Pallas**: François Chollet highlights authoring high-performance custom ops **in Python** using **Pallas** that lower to **Mosaic (TPUs)** or **Triton (GPUs)**, positioning it as eliminating the “leave Python for kernels” requirement for many workflows ([fchollet](https://twitter.com/fchollet/status/2009221193812128006)).  
- **The “kernel tooling fragmentation” meme lands because it’s true**: A viral satire lists the constant churn of DSLs/backends (Triton/Mojo/cuTile/ROCm/TileIR/TileLang/Pallas/TT-Metal/CSL…) capturing the real cost of ecosystem fragmentation and migration tax ([tetsuo_cpp](https://twitter.com/tetsuo_cpp/status/2009238107309461782)).  

---

**Agents & developer workflow: “agent files,” prompt/system improvements at scale, and the reality of big codebases**

- **Agents-as-folders + Skills standardization**: LangChain’s Harrison Chase highlights a concrete pattern: “agents are defined by markdown/json files”—`agents.md`, `subagents/`, `skills.md`, `mcp.json`—making agent packaging/versioning more like repo artifacts than framework-specific objects ([hwchase17](https://twitter.com/hwchase17/status/2009388479604773076)). VS Code ships **Agent Skills** as an “open standard created by Anthropic,” enabling loading folders of instructions/scripts/resources for specialized tasks (feature flag via `chat.useAgentSkills`) ([code](https://twitter.com/code/status/2009428464626016700)).  
- **Prompt/system work has real dollars attached**: An engineer at Lovable claims a systematic system-prompt revamp yielded **4% faster** performance, better design output, and **$20M/year** lower LLM costs—framed as the compounding effect of prompt quality at scale and the importance of fast/safe experimentation ([benjaminvrbk](https://twitter.com/benjaminvrbk/status/2009297105458716753), follow-up recap [benjaminvrbk](https://twitter.com/benjaminvrbk/status/2009297114992660857)).  
- **Big codebases remain the hard mode**: A high-signal observation: Claude Code may underperform on “corp-sized” repos because post-training data skews to smaller repos; real performance likely needs **continual learning / fine-tuning** on your repo, otherwise RAG/manual file reading becomes the bottleneck. Suggestion: modularize with clear API boundaries to reduce context burden ([ibab](https://twitter.com/ibab/status/2009322166593179786)). Dejavucoder adds a concrete tooling delta: Cursor’s advantage comes from **codebase embedding indexing**, which Claude Code lacks by default ([dejavucoder](https://twitter.com/dejavucoder/status/2009375459109441545)).  
- **“Prompt-Driven Development” and agentic testing**: GitHub’s Copilot team pushes prompting as an engineering discipline (refactoring, doc querying via MCP, UI work, docs, tests) ([code](https://twitter.com/code/status/2009097862517342442)). Complementary research: analysis of the AIDev dataset suggests test-containing agent PRs are increasing over time but tend to be larger/slower, with merge rates similar—raising questions about reviewer incentives and test quality variance across agents ([omarsar0](https://twitter.com/omarsar0/status/2009269127773605993)).  
- **DeepAgents / “Ralph mode” ecosystem builds**: LangChain DeepAgents adds Skills & Memory; the framing is “harness-level design”: continual looping + filesystem/git memory, and “skillifying” progress into reusable knowledge artifacts tracked in git ([Vtrivedy10](https://twitter.com/Vtrivedy10/status/2009295526974595519), [mstockton](https://twitter.com/mstockton/status/2009311366444638441)). Replit’s take: autonomy requires frontier models, advanced context management, and exhaustive verification—explicitly naming those “three pillars” ([pirroh](https://twitter.com/pirroh/status/2009381577244258370)).  

---

**Major product moves: OpenAI for Healthcare, Gmail with Gemini 3, and “who pays for OSS?”**

- **OpenAI for Healthcare / ChatGPT for Healthcare**: OpenAI and leaders describe a HIPAA-ready offering: “health intelligence + trusted medical evidence + workflows + enterprise controls.” Partners named include **HCA, Boston Children’s Hospital, MSK, Stanford Health** and more ([bradlightcap](https://twitter.com/bradlightcap/status/2009408962135998653), [thekaransinghal](https://twitter.com/thekaransinghal/status/2009360917847548331), [OpenAI](https://twitter.com/OpenAI/status/2009441959497154829)). There’s also a product narrative around “ChatGPT Health” memory/storage updates ([\_samirism](https://twitter.com/_samirism/status/2009221543214371150)).  
- **Gmail becomes an AI-native inbox**: Google launches AI Overviews for threads, AI-assisted replies/proofread, “AI Inbox” views, and natural-language querying inside Gmail, explicitly emphasizing user controls and opt-in/opt-out ([Google](https://twitter.com/Google/status/2009265269382742346), [Google](https://twitter.com/Google/status/2009266641121477002), [Google](https://twitter.com/Google/status/2009266902112104711)). Engineers immediately point to **phishing/scam detection** as a “most impactful” next feature and warn about jailbreak-abuse risk when a “trusted inbox agent” can persuade users ([polynoamial](https://twitter.com/polynoamial/status/2009322743251259890), [giffmana](https://twitter.com/giffmana/status/2009341983953965236)).  
- **TailwindCSS sponsorship cascade**: After community concern about OSS sustainability in an AI era, Google AI Studio announces Tailwind sponsorship ([OfficialLoganK](https://twitter.com/OfficialLoganK/status/2009339263251566902)). Others frame coding agents as “distribution pipelines” for OSS and call for more big-tech support ([Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2009340177173680509), [sdrzn](https://twitter.com/sdrzn/status/2009361550117880171)). A particularly pointed suggestion: let coding tools contribute micro-funding proportional to token spend, automatically supporting dependencies ([nateliason](https://twitter.com/nateliason/status/2009279537343836261)).  

---

**Research signals: memorization, agent memory architectures, post-training scaling, and measurement**

- **Memorization & copyright extraction is not hypothetical**: The Stanford paper summary (as relayed) claims successful extraction of copyrighted works from multiple production LLMs, with startling verbatim rates for specific models in their experiment, aiming to rebut arguments that “LLMs don’t memorize” ([ednewtonrex](https://twitter.com/ednewtonrex/status/2009201019184415218)).  
- **MAGMA: multi-graph agentic memory for long-horizon reasoning**: Proposes representing memories across **semantic, temporal, causal, entity graphs** and retrieving via policy-guided traversal rather than monolithic embedding similarity; reported gains on **LoCoMo** and **LongMemEval** ([dair_ai](https://twitter.com/dair_ai/status/2009270633398718480)).  
- **SPOT @ ICLR 2026: scaling post-training for LLMs**: Workshop call focuses on principles bridging **algorithms, data, and systems** for post-training scaling; submission date Feb 5, 2026 ([spoticlr](https://twitter.com/spoticlr/status/2009137185510052302)).  
- **Benchmarking expands beyond “capabilities” into openness + agentic realism**: Artificial Analysis continues pushing indices like **GDPval-AA** (realistic knowledge-work tasks with tool/web/terminal environment) and a standardized Openness Index; they also appear on Latent Space discussing eval brittleness/prompt variance and “mystery shopper” policies ([ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/2009367497913585905)). Paras Chopra argues for measuring **human+AI** capability rather than AI-only to avoid optimizing for replacement vs uplift ([paraschopra](https://twitter.com/paraschopra/status/2009118690823033165)).  

---

**Industry meta: “vibe coding,” systems engineering moats, and compute reality**

- **Engineering value shifts upward with agent productivity**: Several tweets converge on the same point: as code generation gets cheap, **complexity management, reliability, and systems engineering** become more valuable—not less. One version predicts junior frontend roles disappear while “risk-managing coding agents” becomes premium labor ([mbusigin](https://twitter.com/mbusigin/status/2009090018682323367)); another frames “non-engineers realizing engineering was never about writing code” ([\_0xaryan](https://twitter.com/_0xaryan/status/2009257975718793460)).  
- **“All software will be generative and generated”**: Vercel CEO’s concise thesis that the default development mode flips permanently ([rauchg](https://twitter.com/rauchg/status/2009324546294468769)). Yuchen Jin frames it as Jevons paradox for coding: more coding, more coders, higher peak leverage ([Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2009324436353372547)).  
- **Compute supply-chain metrics**: Epoch AI claims global capacity is **>15M H100-equivalents**, with a public “AI Chip Sales” explorer and a rough power implication of **>10 GW** chip draw before data center overheads ([EpochAIResearch](https://twitter.com/EpochAIResearch/status/2009366360183460237)).  

(Notes on scope: The input includes substantial geopolitical/political commentary and some non-AI topics; this digest prioritized technically actionable AI/model/system content while still listing the most-engaged non-technical tweets in “Top tweets.” All nitter links were rewritten to twitter.com.)


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Hugging Face Model Releases and Benchmarks

  - **[Hugging Face on Fire: 30+ New/Trending Models (LLMs, Vision, Video) w/ Links](https://www.reddit.com/r/LocalLLM/comments/1q7b54w/hugging_face_on_fire_30_newtrending_models_llms/)** (Activity: 57): ****Hugging Face** has released over 30 new and trending models across various domains including text generation, vision, video, and audio. Notable models include **tencent/HY-MT1.5-1.8B**, a multilingual translation model optimized for edge deployment, and **LGAI-EXAONE/K-EXAONE-236B-A23B**, a massive Korean-focused LLM for advanced reasoning. In the vision domain, **Qwen/Qwen-Image-2512** offers high-fidelity text-to-image generation, while **Lightricks/LTX-2** provides a joint audio-video foundation model for synced video and sound generation. These models are designed for diverse applications such as content generation, edge deployment, and complex reasoning tasks, with many supporting quantization and fast inference capabilities. [Hugging Face Models](https://huggingface.co/models)** One user highlights the suitability of the **Qwen 3 30B** model for business reasoning tasks on a 16G GPU, suggesting it might be optimal for their use case, especially if available on GGUF LM Studio.

    - The user 'alex_godspeed' is considering the Qwen 3 30B model for a business reasoning use case, specifically noting its compatibility with a 16GB GPU. They express a need for the model to be available on GGUF LM Studio, which suggests a focus on efficient deployment and possibly concerns about model size or performance constraints on their hardware. This highlights the importance of model compatibility with specific platforms and hardware configurations in practical applications.

  - **[Guide: How to Run Qwen-Image Diffusion models! (14GB RAM)](https://www.reddit.com/r/LocalLLM/comments/1q7e2ol/guide_how_to_run_qwenimage_diffusion_models_14gb/)** (Activity: 26): **The post introduces a guide for running the latest **Qwen-Image-2512** text-to-image diffusion model and its editing counterpart, **Qwen-Image-Edit-2511**, on local devices. The guide covers installation and setup using libraries like ComfyUI, stable-diffusion.cpp, and diffusers, requiring at least `14GB` of combined RAM/VRAM for optimal performance. It includes instructions for using 4-bit, FP8, and GGUF model variants, and provides tips on creating effective prompts and adjusting hyperparameters such as sampling and guidance. The guide also highlights recent updates to GGUFs for improved quality by prioritizing important layers, available on [Hugging Face](https://huggingface.co/unsloth/Qwen-Image-2512-GGUF).** A commenter expressed a desire for a more accessible UI than ComfyUI, which they find challenging due to visual impairments, despite acknowledging its functionality.



### 2. Local LLM Deployment and Hardware Considerations

  - **[Does it make sense to have a lot of RAM (96 or even 128GB) if VRAM is limited to only 8GB?](https://www.reddit.com/r/LocalLLM/comments/1q7e34g/does_it_make_sense_to_have_a_lot_of_ram_96_or/)** (Activity: 75): **Running large language models (LLMs) locally with limited VRAM (8GB) but substantial RAM (up to 128GB) can be technically feasible but comes with limitations. The primary constraint is the speed difference between system RAM and VRAM. DDR5 RAM offers a bandwidth of approximately `80 GB/s`, allowing an `80B` model quantized to `Q8` to run at about `1 token/s`. In contrast, VRAM bandwidth ranges from `200 GB/s` to `2000 GB/s`, enabling the same model to run at `2.5 to 25 tokens/s` depending on the GPU. Mixture-of-Experts (MoE) models, which activate only a subset of parameters per token, can achieve higher throughput even with large parameter counts, potentially reaching `20 tokens/s` with system RAM.** One commenter suggests investing in a better GPU rather than more RAM due to the significant speed advantage of VRAM. Another notes that while 128GB RAM allows experimentation with larger MoE models, these models often fail to meet productivity thresholds for tasks like coding, indicating practical limitations despite theoretical capabilities.

    - uti24 highlights the speed limitations of using system RAM versus VRAM for running large models. DDR5 RAM typically offers around 80 GB/s bandwidth, allowing an 80B model quantized to Q8 to run at approximately 1 token/s. In contrast, VRAM bandwidth ranges from 200 GB/s to 2000 GB/s, enabling the same model to run at 2.5 to 25 tokens/s, depending on the GPU's capabilities. This illustrates the significant performance advantage of VRAM over system RAM for such tasks.
    - Medium_Chemist_4032 discusses the practical limitations of using large amounts of RAM with MoE (Mixture-of-Experts) models. Despite having 128GB of RAM, they found that larger MoE models did not meet the minimum productivity threshold for tasks like coding. The models often failed to produce a single working file and entered loops, indicating that simply having more RAM does not guarantee effective performance for all applications.
    - netroxreads shares their experience with an M3 Ultra system equipped with 256GB of UMA RAM, running a 120B GPT-OSS model at 70 tokens per second. This performance is attributed to the unified memory architecture, which contrasts with hybrid GPU/CPU RAM setups on PCs that are expected to be significantly slower. This highlights the potential benefits of UMA in handling large models efficiently.

  - **[Creating a local offline LLM for my company](https://www.reddit.com/r/LocalLLM/comments/1q7d9uj/creating_a_local_offline_llm_for_my_company/)** (Activity: 32): **Building an internal, offline LLM for approximately 150 users is technically feasible but requires significant hardware and infrastructure investment. A single RTX 5090 may suffice for prototyping, but scaling to production would necessitate a more robust setup, potentially involving `50k+ USD` in AI processors alone. For production-level resilience and scalability, a Kubernetes-based infrastructure is recommended, though this adds complexity and maintenance overhead. Off-the-shelf solutions like `VLLM` can handle inference but may not be suitable for enterprise-level deployment without additional infrastructure.** Commenters suggest defining specific use cases and performance requirements (e.g., speed vs. accuracy) before proceeding. They also recommend considering Retrieval-Augmented Generation (RAG) for organization-specific tasks, which may not require full model training. Leasing hardware or using cloud services for testing is advised to manage costs.

    - **Own_Attention_3392** highlights the distinction between training and running an LLM, emphasizing that hardware requirements differ significantly. They suggest using Retrieval-Augmented Generation (RAG) for organization-specific tasks without needing to train a new model, as RAG can be integrated with existing LLM services.
    - **phocuser** discusses the importance of defining the LLM's purpose, such as speed versus accuracy, before determining hardware needs. They note that even a high-end GPU like the 5090 may struggle with models approaching GPT-4.1's capabilities. They suggest building a hardware stack for handling multiple requests and recommend testing models on local hardware or cloud servers to evaluate performance.
    - **sometimes_angery** shares insights from building an offline MLOps platform, noting a significant hardware investment of $75-100k USD. They suggest using Kubernetes for scalability and resilience in production environments, but caution that this requires dedicated maintenance personnel. For simpler setups, they recommend VLLM for inference on a single machine, though this may not be suitable for enterprise-level production.


### 3. Coding with GLM 4.7 vs Claude Sonnet 4.5

  - **[Been using glm 4.7 for coding instead of claude sonnet 4.5 and the cost difference is huge](https://www.reddit.com/r/LocalLLM/comments/1q79orf/been_using_glm_47_for_coding_instead_of_claude/)** (Activity: 83): **The post discusses a comparison between **Claude Sonnet 4.5** and **GLM 4.7** from **Zhipu AI** for coding tasks such as debugging, refactoring, and code generation. The user found that GLM 4.7, an open-source model, delivered working code `85-90%` of the time, which is close to the quality of Claude Sonnet 4.5 but at a significantly lower cost, approximately `1/5th` of the API expenses. GLM 4.7 also handled long files better than competitors like DeepSeek and Kimi, without losing context or hitting token limits quickly. However, Claude Sonnet 4.5 still excels in UI/UX and high-level discussions.** Commenters noted that GLM 4.7 is effective for long files and doesn't hallucinate imports, making it a cost-effective choice for bulk coding work. Another user shared a costly experience with Opus 4.5, highlighting the potential savings with GLM 4.7.

    - Scared-Biscotti2287 highlights that **GLM 4.7** is particularly effective for handling long files and avoids issues like hallucinating imports, which can be a problem with other models like Kimi. While it may not be as polished as Claude for explanations, its strength lies in code generation, making it a cost-effective choice for bulk coding tasks.
    - whyyoudidit shares a personal experience with **Opus 4.5** in Visual Studio Code, noting a significant cost of `$10 in 5 minutes` while refactoring `1400 lines of code`. This highlights the potential high costs associated with using certain models for extensive code refactoring tasks, especially for users new to these tools.
    - No_Conversation9561 inquires about the hardware used, which can be a critical factor in performance and cost when using models like GLM 4.7 for coding tasks. The hardware specifications can influence the efficiency and speed of code generation and processing.

  - **[Dialogue Tree Search - MCTS-style tree search to find optimal dialogue paths (so you don't have to trial-and-error it yourself)](https://www.reddit.com/r/LocalLLaMA/comments/1q71sbe/dialogue_tree_search_mctsstyle_tree_search_to/)** (Activity: 356): **The project introduces a novel approach to dialogue optimization using a parallel beam search algorithm instead of traditional Monte Carlo Tree Search (MCTS). This method generates multiple dialogue strategies, forks them into user intent variants, and evaluates them using three independent LLM judges to score and prune conversation paths. The system is designed to handle diverse user intents and integrates deep research capabilities via GPT-Researcher for domain context. It supports OpenAI-compatible endpoints and is open-source under the Apache 2.0 license. The approach is token-intensive, potentially requiring over 300 LLM calls per run, and is currently limited to OpenAI models.** Commenters appreciate the use of beam search over MCTS for dialogue optimization, noting its suitability for maintaining coherent conversation paths. The user intent forking feature is highlighted as a valuable addition, allowing strategies to be tested against different personas. There is also a suggestion to explore alternative search providers to reduce costs.

    - TheGrossVolcano highlights the use of beam search over pure Monte Carlo Tree Search (MCTS) for dialogue path optimization. Beam search is more suitable for dialogue systems as it prevents exploration from diverging too far from relevant paths, which is crucial in maintaining coherent and contextually appropriate conversations.
    - harlekinrains points out the high cost of Firecrawls' subscription model, which is $140 per year, and suggests the need for more cost-effective alternatives. They also provide a link to a GitHub repository that aggregates various search providers, which could be useful for those looking to implement alternative solutions.
    - charlesrwest0 raises an interesting question about the potential application of this dialogue optimization technique in role-playing (RP) scenarios, suggesting that it could be used to enhance RP responses by finding optimal dialogue paths.


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. AI Model and Benchmark Launches

  - **[WSJ: Anthropic reportedly raising $10B at a $350B valuation as AI funding accelerates](https://www.reddit.com/r/singularity/comments/1q75o0z/wsj_anthropic_reportedly_raising_10b_at_a_350b/)** (Activity: 218): ****Anthropic** is reportedly raising $10 billion at a valuation of $350 billion, marking one of the largest private fundraises in AI history. This significant increase in valuation from $183 billion to $350 billion in just four months underscores the rapid consolidation of capital around leading AI model developers. The funding is driven by the high demand for compute and infrastructure, rather than immediate product offerings, and aligns with expectations of increased AI IPO activity by 2026, reflecting growing investor confidence in the AI sector. [Source: Wall Street Journal](https://www.wsj.com/tech/ai/anthropic-raising-10-billion-at-350-billion-value-62af49f4).** Commenters highlight Anthropic's strong optimization on code as a competitive advantage, suggesting it provides a 'moat' for the company. There is also a sentiment that the US economy is heavily reliant on AI trade, with other sectors experiencing slower growth.

    - Anthropic's focus on optimizing code is seen as a significant competitive advantage, creating a 'moat' that differentiates them from other AI companies. This suggests that their technical capabilities in code optimization are a key factor in their high valuation and attractiveness to investors.
    - The discussion highlights the strategic investment in AI by major companies like Google, which is seen as a necessary move to maintain competitive parity. This reflects a broader trend where AI is becoming a critical area for investment, potentially at the expense of other sectors, as indicated by the comment on the US economy's reliance on AI trade for growth.

  - **[QwenLong-L1.5 | Long Term Memory DIY](https://www.reddit.com/r/Oobabooga/comments/1q73tnh/qwenlongl15_long_term_memory_diy/)** (Activity: 2): ****QwenLong-L1.5** introduces a novel approach to long-term memory management in AI models by using reasoning to tag and store only essential information, rather than bloating the context size with entire chat histories. This method is detailed in the [Tongyi-Zhiwen white paper](https://huggingface.co/Tongyi-Zhiwen), suggesting it outperforms traditional long-term memory techniques. The model is also noted for its strong reasoning capabilities, making it suitable for applications like role-playing scenarios.** Some users express interest in testing QwenLong-L1.5 for role-playing applications, anticipating improved performance based on its specifications.

    - A user discusses the implementation of long-term memory in QwenLong-L1.5, highlighting the use of a memory buffer that stores past interactions. This buffer is periodically pruned to maintain performance, ensuring that only the most relevant data is retained. The approach allows the model to handle extended conversations without significant degradation in response quality.
    - Another comment delves into the performance benchmarks of QwenLong-L1.5, noting that it achieves a `BLEU score` improvement of 15% over its predecessor. This improvement is attributed to enhanced context management and memory retention capabilities, which are crucial for applications requiring sustained dialogue coherence.
    - A technical debate arises around the scalability of QwenLong-L1.5's memory system. Some users express concerns about the computational overhead introduced by the memory buffer, especially in resource-constrained environments. Others argue that the trade-off is justified by the model's ability to maintain context over longer interactions, which is a significant advancement in conversational AI.

  - **[I made Gemini 3 Pro/Flash play 21,000 hands of Poker](https://www.reddit.com/r/GeminiAI/comments/1q7gy25/i_made_gemini_3_proflash_play_21000_hands_of_poker/)** (Activity: 124): **The image is a line graph illustrating the performance of various AI models, including **Gemini 3 Pro/Flash**, **GPT-5.2/5 mini**, **Grok 4.1 Fast Reasoning**, and **Opus/Haiku 4.5**, over `21,000` hands of poker. The graph shows that **Gemini 3 Pro** significantly increases its winnings towards the end, suggesting superior performance in this benchmark. This is part of a new LLM benchmark called **PokerBench**, which allows for the evaluation of AI models' poker strategies in a competitive setting. The data and simulator are available on [PokerBench's website](https://pokerbench.adfontes.io/) and GitHub.** One commenter noted that in a head-to-head comparison, **Gemini 3 Flash** seemed to perform better than **Gemini 3 Pro**, suggesting that the latter's success might not be purely due to skill but possibly luck in the broader tournament context.

    - In the heads-up comparison between Gemini Flash and Pro, it appears that Flash performs better at poker, suggesting it may have a more effective strategy or decision-making process in this context. This could imply differences in the models' architectures or training data that favor Flash in poker scenarios.
    - A user inquired about the cost of running these models against each other, which is a relevant consideration for large-scale simulations like this. The cost would depend on factors such as the computational resources required, the duration of the simulation, and the specific infrastructure used (e.g., cloud services or local hardware).

  - **[I’m the Co-founder &amp; CEO of Lightricks. We just open-sourced LTX-2, a production-ready audio-video AI model. AMA.](https://www.reddit.com/r/StableDiffusion/comments/1q7dzq2/im_the_cofounder_ceo_of_lightricks_we_just/)** (Activity: 2083): ****Lightricks** has open-sourced **LTX-2**, a production-ready audio-video AI model, including weights, code, a trainer, benchmarks, LoRAs, and documentation. This model is designed to run locally on consumer GPUs, making it accessible for real-world applications. The open-source release aims to address the challenges of running and reproducing multimodal models, which are often difficult to implement. The release is part of a broader strategy to enhance the usability and accessibility of AI models in production environments. More details can be found on the [LTX-2 model page](https://ltx.io/model).** Commenters are curious about the motivations behind open-sourcing LTX-2, with some expressing gratitude and excitement about its potential impact on open-source video technology. The decision to open-source is seen as a significant move that could influence the future of multimodal models.

    - The decision to open-source LTX-2 was driven by a commitment to foster community collaboration and innovation. The team aims to avoid the pitfalls of previous models like Wan 2.6+ that went closed source, which led to community dissatisfaction. By releasing open weights, Lightricks hopes to maintain transparency and encourage community-driven improvements and adaptations.
    - Lightricks has imposed certain training restrictions on LTX-2 to comply with legal standards, such as avoiding NSFW and copyrighted material. This limitation may affect the model's versatility, but the open-source nature allows the community to potentially retrain and expand its capabilities within legal boundaries. This approach could enhance the model's range and adaptability over time.
    - The release of LTX-2 as an open-source model is seen as a significant shift in the open-source video AI landscape. It contrasts with previous models that restricted access, and the community is keen to see how Lightricks will maintain its commitment to open-source principles in the future. The open weights are a step towards ensuring ongoing community engagement and development.

  - **[[P] Three-Phase Self-Inclusive Evaluation Protocol for Synthetic Data Generation in a Fine-Tuned 4B Model (Experiment 3/100)](https://www.reddit.com/r/MachineLearning/comments/1q7f7tr/p_threephase_selfinclusive_evaluation_protocol/)** (Activity: 6): **The post outlines a **three-phase self-inclusive evaluation protocol** for synthetic data generation using a fine-tuned 4B model. The protocol includes a **Generation Phase** where multiple models, including a fine-tuned 4B model, generate responses to a proprietary prompt. In the **Analysis Phase**, each model ranks the outputs based on criteria like coherence and creativity. Finally, the **Aggregation Phase** compiles these rankings for an overall assessment. The experiment is open-source under the MIT license, with all data available on [GitHub](https://github.com/Roforum/Xthos-v2-the-sovereign-architect-Model-Evaluation-Experiment). The aim is to explore biases in LLM-as-judge setups and the reproducibility of subjective evaluations, with support for local inference via Ollama.** Commenters are discussing the potential biases introduced by proprietary prompts and self-ranking, and are suggesting more rigorous statistical methods for aggregation. There is also interest in the fine-tuning trade-offs and the local inference setup, with further details available in a related [Reddit discussion](https://www.reddit.com/r/LocalLLaMA/comments/1q6p967/experimental_xthosv2_the_sovereign_architect/).

    - The discussion highlights the use of a fine-tuned 4B model, specifically focusing on the evaluation protocol for synthetic data generation. The model's training details, dataset composition, and quantization options are discussed in a separate thread, emphasizing the importance of understanding these aspects for effective model deployment. The evaluation protocol is part of a larger series of experiments, with this being the third out of a planned hundred, showcasing a systematic approach to model assessment.
    - The linked discussion provides insights into the local inference setup using Ollama, which is crucial for those implementing the model in a local environment. This setup allows for efficient testing and deployment of the model, ensuring that the synthetic data generation process is both robust and scalable. The availability of raw data and analyses in the GitHub repository supports transparency and reproducibility of the results, which is essential for validating the model's performance.
    - The post invites questions on the methodology, indicating a collaborative approach to refining the evaluation protocol. This openness to community engagement suggests a dynamic development process, where feedback can lead to iterative improvements in the model's evaluation. The emphasis on methodology-related questions highlights the technical depth of the project and the importance of peer review in advancing the field.

  - **[[P] Automated Code Comment Quality Assessment with 94.85% Accuracy - Open Source](https://www.reddit.com/r/MachineLearning/comments/1q7rd9o/p_automated_code_comment_quality_assessment_with/)** (Activity: 10): **The post introduces a text classifier for assessing code comment quality, achieving `94.85%` accuracy on a test set. The model is a fine-tuned version of **DistilBERT** with `66.96M` parameters, and is available under the **MIT License**. It categorizes comments into four classes: Excellent, Helpful, Unclear, and Outdated, with precision rates of `100%`, `89%`, `100%`, and `92%` respectively. The model can be easily integrated using the `Transformers` library, and is hosted on [Hugging Face](https://huggingface.co/Snaseem2026/code-comment-classifier). Potential applications include CI/CD integration, real-time IDE feedback, and developer training tools.** A top comment requests details on the model's creation process, indicating interest in the methodology and implementation specifics.

    - The project likely involves using machine learning models to evaluate the quality of code comments. Achieving a 94.85% accuracy suggests a well-tuned model, possibly leveraging natural language processing (NLP) techniques to understand and assess the semantic quality of comments. The model might be trained on a labeled dataset where comments are rated for quality, using features such as readability, relevance, and informativeness.
    - The open-source nature of the project implies that the code and datasets are available for public use and contribution. This transparency allows for community-driven improvements and validation of the model's performance across different programming languages and codebases. The project might use popular libraries like TensorFlow or PyTorch for model development and training.
    - The high accuracy rate indicates that the model has been rigorously tested, possibly using cross-validation techniques to ensure robustness. The dataset used for training might include a diverse range of programming languages and comment styles to generalize well across different coding environments. The model's performance could be benchmarked against existing tools or human evaluations to validate its effectiveness.


### 2. Claude Code Usage and Experiences

  - **[Opus 4.5 actually just… gets it? Shipped my first iOS app without knowing Swift](https://www.reddit.com/r/ClaudeAI/comments/1q73hkv/opus_45_actually_just_gets_it_shipped_my_first/)** (Activity: 974): **The post discusses the use of **Opus 4.5**, a tool that significantly simplifies app development, allowing a non-developer to create a fully functional iOS app without prior knowledge of Swift. The user highlights how Opus 4.5 intuitively understands vague instructions like 'make it feel calm and minimal' and provides proactive feedback on potential issues, akin to working with a senior developer. This version of Opus is noted for its improved decision-making and debugging capabilities, reducing the need for constant clarification and offering more reasoned problem-solving approaches.** Some commenters noted that Opus 4.5 tends to use the same design and color palette across different apps, suggesting a lack of variety in its UI design capabilities.


  - **[What Actual Usage Looks like Against Max 20x Plan - 4 Hours Into Session.](https://www.reddit.com/r/ClaudeCode/comments/1q7kwug/what_actual_usage_looks_like_against_max_20x_plan/)** (Activity: 168): **The post discusses the usage of the Max 20x plan for Claude Code, highlighting that after 4 hours of usage, the user has consumed 30% of their session tokens and only 1% of their weekly tokens. The user is working on both marketing and engineering tasks, utilizing features like Opus 4.5 and ultrathink. The image shows a usage dashboard indicating the current session and weekly usage percentages. The user expresses skepticism about claims of others reaching their weekly limits quickly, suggesting that the limits are generous and sufficient for their needs. The post also mentions the use of slash commands to automate workflow validation processes.** One commenter mentions issues with the 5x plan, stating that they reached the 5-hour limit twice in just 2 hours, suggesting a possible bug. Another commenter advises optimizing CLI tool parameters to reduce irrelevant context sent to the LLM, which can help manage usage more effectively.

    - srirachaninja reports an issue with the 5x plan where the 5-hour limit seems inaccurate, as they reached it twice within just 2 hours using only 6-8 prompts per session. This suggests a potential bug in the usage tracking system, especially since their usage pattern hasn't changed significantly.
    - positivitittie suggests optimizing the context sent to the LLM by using CLI tool parameters to filter out unnecessary information, which can help reduce the context size and improve efficiency. They also recommend using Boris' Claude Code Setup and configuring scripts to minimize irrelevant context, which can prevent hitting usage limits prematurely.
    - doomdayx and Drakuf both mention unexpected usage spikes, with doomdayx experiencing a sudden 40% usage of a 5-hour limit in minutes, and Drakuf noting 27% of a weekly limit used after 12 hours. These reports indicate potential inconsistencies in usage tracking, which the CC team is aware of and investigating.

  - **[What is some serious claude code sauce people should know about? No BS](https://www.reddit.com/r/ClaudeCode/comments/1q7fs2o/what_is_some_serious_claude_code_sauce_people/)** (Activity: 138): **A Reddit user shared a technique for improving Claude Code's quality by implementing a `UserPromptSubmit` type hook to read a `.ps1` file on Windows, which directs Claude Code to use the most relevant agent or skill for a task. This acts as a "routing" file, enhancing task-specific performance. Another user emphasized the importance of "Plan mode" and building project-specific skills. They also shared a comprehensive list of techniques, including an **Error Logging System** to identify patterns in failures, using **/Commands as Lightweight Local Apps** for quick workflow creation, and implementing **Hooks for Deterministic Safety** to prevent dangerous actions. Additionally, they recommended maintaining **Context Hygiene** by managing context compaction manually, leveraging **Subagent Control** for complex projects, and utilizing a **Reprompter System** for structured prompting. The full document detailing these techniques is available [here](https://docs.google.com/document/d/1I9r21TyQuAO1y2ecztBU0PSCpjHSL_vZJiA5v276Wro/edit?usp=sharing).** One commenter highlighted the importance of thinking in terms of loops, suggesting that providing instructions for compiling and testing can help Claude Code emulate a typical development workflow. They noted the complexity of implementation but affirmed its value, especially for web apps, recommending tools like Playwright or Selenium.

    - **Error Logging System**: This involves reconstructing the input-output loop that agentic coding often obscures. By logging failures with the exact triggering prompt and categorizing them, developers can identify patterns and understand what went wrong, leading to more effective debugging and optimization.
    - **/Commands as Lightweight Local Apps**: Slash commands in Claude Code are likened to Claude as a Service, offering the power of a SaaS with quicker build times. This feature allows for the creation of workflows that are both powerful and efficient, acting as lightweight local applications.
    - **Subagent Control**: Claude Code frequently spawns subagents like Sonnet/Haiku even for knowledge tasks. By adding 'Always launch opus subagents' to the global CLAUDE.md, developers can leverage subagents more effectively, enhancing project orchestration beyond the capabilities of vanilla Claude Code.

  - **[More than 'Max 20X' Plan - go to a higher plan??](https://www.reddit.com/r/ClaudeCode/comments/1q715zo/more_than_max_20x_plan_go_to_a_higher_plan/)** (Activity: 111): **The user is experiencing rapid depletion of their 'Max 20X' plan credits, which are exhausted within 5-6 days due to increased usage involving subagents, asynchronous operations, and remote Gitrees. They are attempting to manage context usage by employing `/clear` and `/compact` commands and offloading memory to local SQLite and vector databases. The user is considering upgrading beyond the 'Max 20X' plan, potentially to a 'Teams' plan, to accommodate their increased computational demands. The post highlights the need for efficient context management and the potential cost implications of high computational usage.** Commenters suggest that the user might benefit from deactivating the auto-compact feature, which consumes significant context resources, and question the intensity of the user's computational activities, implying that such high usage is uncommon.

    - Familiar_Gas_1487 points out that the 'Max 20X' plan limits are set on a weekly basis rather than monthly, and suggests using the API as an alternative to manage higher usage. This implies that users can potentially bypass some limitations by integrating with the API directly, which might offer more flexibility in usage patterns.
    - PathFormer highlights a technical detail regarding the 'auto compact' feature, which consumes 40k of the total 200k context in every session. Disabling this feature could significantly reduce context usage, allowing users to maximize their plan's capacity more effectively.
    - HangJet and websitebutlers both emphasize that the '20x plan' operates on a weekly reset schedule. This means that users experiencing limitations might simply need to wait for the reset rather than seeking a higher plan, suggesting a potential misunderstanding of the plan's structure.

  - **[Claude Code is the best Mac cleaner app](https://www.reddit.com/r/ClaudeCode/comments/1q7eqqm/claude_code_is_the_best_mac_cleaner_app/)** (Activity: 198): **The image humorously presents a terminal interface of a fictional app called "Claude Code," which is depicted as a Mac cleaner. It lists various items cleaned, such as "node_modules," "Claude debug logs," and "Python venvs in Downloads," with a total of approximately `10.5GB` freed, increasing the disk space from `29GB` to `33GB`. This is a satirical take on system cleaning tools, suggesting that the app is highly effective in freeing up space by removing unnecessary files.** One commenter humorously suggests that the app might be a bait, while another jokes about the potential danger of using a command like `rm -rf /`, which would delete all files on the system.


  - **[Loosing plan usage limits without touching anything](https://www.reddit.com/r/ClaudeCode/comments/1q7alp0/loosing_plan_usage_limits_without_touching/)** (Activity: 84): **Users are reporting unexpected increases in their usage limits for **Claude Desktop** and **Claude Code** without any interaction, particularly after January 2026. One user noted a `6%` session limit usage upon starting their Mac, despite not using the service for two days. This issue seems to affect **pro plan users** ($20/month), who observed a decline in usage tokens following a holiday promotion. The lack of communication and support from **Anthropic** is a point of concern.** There is a notable dissatisfaction among users regarding the unexplained usage increase and the perceived lack of communication from **Anthropic**. Some users express disappointment and suspicion about the situation, indicating a need for better transparency and support.

    - A user reported a sudden increase in their session limit usage without actively using the service, noting a 6% usage upon starting their Mac. They are on a pro plan costing $20 and observed a decline in usage tokens post a holiday promotion, expressing dissatisfaction with **Anthropic's** communication and support.
    - Another user suggested that leaving sessions open could result in token usage over time, as sessions consume a small amount of tokens even when not actively used. Additionally, using 'skills' or 'mcp' can significantly impact token usage during session start, which might explain unexpected usage increases.


### 3. AI Prompt Engineering and Usage Challenges

  - **[The day I stopped collecting “cool prompts” and started building a tiny standard library](https://www.reddit.com/r/PromptEngineering/comments/1q7cqj7/the_day_i_stopped_collecting_cool_prompts_and/)** (Activity: 141): **The post describes a shift from collecting random prompts to developing a structured 'standard library' of prompt patterns, akin to software development practices. The author emphasizes creating reusable patterns with defined inputs and outputs, treating bad outputs as 'failing tests', and incorporating pre and post conditions to improve prompt reliability. This approach led to more predictable and reusable outputs, transforming the process from ad-hoc prompt generation to a systematic method akin to calling functions from a library. The author shares their library for others to use or adapt [here](https://allneedshere.blog/prompt-pack.html).** Commenters shared various methods for managing prompt libraries, such as using ChatGPT's system instructions for project-specific prompts, storing prompts in a custom git library on an Ubuntu machine, and using a Chrome extension called Promptsloth for easy access.

    - kermitt81 describes a method of organizing prompts in ChatGPT by using 'Projects' with system instructions. Each project is tailored for specific tasks, and one project is dedicated to on-the-fly prompt design. This project analyzes a user's request, breaks it down into components, and generates a detailed prompt based on predefined rules. This approach allows for creating highly detailed and usable prompts that may only require minor adjustments.
    - fakiestfakecrackerg suggests creating a complex system of interconnected rulebooks and instructions. This involves compiling essential prompts into a foundational set of custom instructions and using additional prompts to build a layered framework of specific functions. This method can enhance the efficiency and effectiveness of prompt usage by creating a structured and interconnected system.

  - **[2 Biggest issues of AI in 2026](https://www.reddit.com/r/PromptEngineering/comments/1q7nlxg/2_biggest_issues_of_ai_in_2026/)** (Activity: 24): **The post identifies two major issues with AI as of 2026: AI's tendency to fill in gaps when context is missing, leading to potentially inaccurate outputs, and the mismatch between human communication styles and AI's need for structured input. The author argues that AI often assumes unstated goals and constraints, resulting in polished but potentially incorrect responses. The post suggests that the root of many AI frustrations is the human inability to adapt communication to AI's structured needs, rather than a lack of AI capability. The author is researching a tool to address these issues, available at [aichat.guide](http://www.aichat.guide).** A top comment challenges the idea that humans should adapt to AI's need for structured input, arguing that AI should instead evolve to handle human communication styles, which are inherently ambiguous and non-linear. The commenter suggests that while prompt engineering is useful now, the long-term goal should be for AI to better understand human-native communication, preserving the richness of human expression.

    - The comment highlights a critical issue in AI development: the current reliance on structured input, which places the onus on humans to adapt their communication to suit AI systems. This approach risks diminishing the richness of human expression, as it encourages people to communicate in a more sanitized, API-like manner. The commenter argues that AI should evolve to better handle the nuances of human communication, such as ambiguity and emotion, rather than forcing humans to adapt to AI's limitations. This perspective suggests that AI's ability to understand human-native communication is crucial for effective collaboration and scalability.

  - **[The more I ‘polish’ a prompt, the worse the output gets. Why?](https://www.reddit.com/r/PromptEngineering/comments/1q71q8k/the_more_i_polish_a_prompt_the_worse_the_output/)** (Activity: 24): **The post discusses a common issue with prompt engineering for language models, where refining a prompt with more details can lead to worse outputs. This is often due to over-constraining the model or introducing ambiguity, which can confuse the model as it tries to satisfy multiple intentions. Simplifying prompts by focusing on the core purpose rather than adding excessive details can lead to better results, as it allows the model to leverage its training more effectively.** Commenters suggest that overly detailed prompts can cause 'prompt fatigue' or 'dilution,' where the model prioritizes following specific rules over content quality. They recommend using purpose-based prompts or a few-shot approach with examples to maintain focus and creativity.

    - Adventurous-Pool6213 highlights that overly detailed prompts can confuse models by introducing mixed intentions, leading to suboptimal outputs. They suggest using purpose-based prompts, which provide a clear direction without excessive detail, allowing models to fill in creative gaps effectively. This approach is particularly effective in visual models, as seen in tools like [gentube.app](https://www.gentube.app/?_cid=cm).
    - liquiditygod discusses the concept of 'prompt fatigue' or 'dilution,' where too many instructions in a prompt cause the model to focus on following rules rather than producing quality content. They recommend a few-shot approach, providing examples instead of detailed instructions, to maintain the model's focus on the core goal and improve output quality.
    - PurpleWho describes an iterative approach to prompt development akin to test-driven development (TDD). They start with basic prompts, run them against real inputs, and capture failures as test cases. This method helps refine prompts by addressing edge cases and preventing regressions. They mention using tools like Mind Rig and vscode-ai-toolkit for testing, and eventually exporting to formal evaluation tools like Braintrust or PromptFoo for comprehensive analysis.

  - **[Vibe Coding Isn’t Easier — It’s Just a Different Path](https://www.reddit.com/r/PromptEngineering/comments/1q7pc2o/vibe_coding_isnt_easier_its_just_a_different_path/)** (Activity: 18): **The post argues that 'vibe coding'—a method that emphasizes clear intent, problem framing, and systems thinking over traditional syntax and boilerplate—does not reduce the difficulty of coding but shifts it. It suggests that while traditional coding focuses on mechanical execution, vibe coding keeps developers engaged in problem-solving and design. Tools like [Lumra](https://lumra.orionthcomp.tech) are highlighted for their role in organizing prompts and iterations, thus supporting sustainable workflows rather than serving as shortcuts.** Comments reflect skepticism about the difficulty of vibe coding, with some users arguing it is easier and dismissing the post's claims as 'AI generated bullshit.' Others note that vibe coding feels conversational and express frustration with tools not adhering to style guidelines, indicating a gap in practical examples and tool functionality.

    - thinkmatt discusses the conversational nature of 'vibe coding' with AI, highlighting that prompts are rarely reused, which contrasts with traditional coding practices where code reuse is common. They express a desire for AI tools like Cursor to better adhere to predefined style guidelines, indicating a gap in current AI capabilities to fully integrate with developer workflows.

  - **[[R] Collecting memes for LLM study—submit yours and see the analysis!](https://www.reddit.com/r/MachineLearning/comments/1q7aeoy/r_collecting_memes_for_llm_studysubmit_yours_and/)** (Activity: 20): **Researchers from **THWS** and **CAIRO's NLP Team** are developing **MemeQA**, a crowd-sourced dataset aimed at evaluating Vision-Language Models (VLMs) on their ability to understand memes, focusing on aspects like humor, emotional mapping, and cultural context. The dataset will include over `10 dimensions` per meme, and contributors can submit memes via [memes.thws.ai](http://memes.thws.ai) to aid in creating a comprehensive benchmark for VLMs.** Commenters raised concerns about the dataset's initial size of `31 memes`, suggesting scraping meme subreddits for more data. There is also skepticism about the use of crowd-sourced data as 'free training data' for models.

    - Forsaken-Order-7376 raises a technical question about the methodology for annotating ground truth labels in the study. This is crucial for ensuring the accuracy and reliability of the dataset used for training or evaluating models. Proper annotation is essential for supervised learning tasks, where the model's performance heavily depends on the quality of the labeled data.

  - **[[D] I summarized my 4-year PhD on Geometric Deep Learning for Molecular Design into 3 research questions](https://www.reddit.com/r/MachineLearning/comments/1q72bd8/d_i_summarized_my_4year_phd_on_geometric_deep/)** (Activity: 145): ****Chaitanya Joshi** summarized his PhD thesis on *Geometric Deep Learning for Molecular Design* into three key research questions, focusing on the expressivity of 3D representations, generative modeling for periodic and non-periodic systems, and real-world design of functional RNA. He introduced the *Geometric Weisfeiler-Leman Test* for expressivity, proposed the *All-atom Diffusion Transformer* for unified generative modeling, and developed *gRNAde* for RNA design, validated through wet-lab experiments. The thesis highlights a progression from theoretical graph isomorphism problems to practical applications in molecular biology. [Read more](https://chaitjo.substack.com/p/phd-thesis-in-three-questions).** Commenters are interested in the future role of equivariant models, especially in light of scaling and data augmentation, and how these factors might influence model choice in industry. There is also curiosity about testing transfer learning in models like the All-atom Diffusion Transformer and the challenges faced during wet-lab validation. Additionally, questions about the source of initial training structures and the differences between X-ray and in vivo structures were raised.

    - Affectionate-Dot5725 raises a technical discussion on the role of equivariant models in the context of scaling and data augmentation. The commenter is curious about whether the increasing scale and data augmentation capabilities might reduce the necessity of equivariant models, especially in industrial applications. This reflects a broader debate on the trade-offs between model complexity and the benefits of large-scale data-driven approaches.
    - NoPriorThreat discusses the challenges of obtaining initial training structures for molecular models, highlighting the limitations of using X-ray crystallography and ab initio methods. X-ray structures often represent 'unbiologically frozen' states, which differ from in vivo conditions, while ab initio methods are computationally expensive and can be inaccurate for large systems. This underscores the difficulty in balancing accuracy and computational feasibility in molecular modeling.
    - Affectionate-Dot5725 also inquires about testing transfer learning in models, particularly in the context of state-of-the-art all-atom diffusion models. The question focuses on how to evaluate whether joint training enhances representation learning, which is a critical aspect of understanding the effectiveness of transfer learning in complex models.


---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5.2


**1. New Tooling & Framework Releases**

- **Transformers v5 Goes on a Spring Cleaning Spree**: Hugging Face shipped **Transformers v5**, unifying the tokenizer backend, modularizing model definitions, focusing on **PyTorch**, and prioritizing quantization plus new serving/inference features in the blog post ["Transformers v5"](https://huggingface.co/blog/transformers-v5).
  - The same announcements wave also introduced Apple-focused client tooling—["swift-huggingface"](https://huggingface.co/blog/swift-huggingface) and ["AnyLanguageModel"](https://huggingface.co/blog/anylanguagemodel)—aiming to make **local+remote LLM access** feel like one API on Apple platforms.

- **DSPy ‘Rewrites History’ (This Time It’s Good)**: DSPy contributors debated why their tutorial includes **conversation history in the system prompt** (["DSPy conversation history tutorial"](https://dspy.ai/tutorials/conversation_history)), with maintainers explaining it’s an **adapter representation detail** you can change.
  - The team said they’re overhauling **multi-turn conversations** and that updates are expected **later this month**, with the practical takeaway being: write **custom adapters** to control how history is serialized without affecting optimizers.

- **MCP Wants ‘Dry-Run’ Tool Calls Before Mutations**: MCP contributors proposed standardizing a way to **stage mutating actions** via tool calls before execution, and asked whether it should become a [SEP](https://sep.dev).
  - Others pushed back that staging likely belongs in **SDK implementation guidance** (not a protocol change), while the group also revived discussion about **W3C WebMCP** collaborating with MCP.


**2. Model Launches, Benchmarks, and Leaderboards**

- **ERNIE Elbows Into Vision Arena’s Top 10**: LM Arena’s [Vision leaderboard](https://lmarena.ai/leaderboard/vision) update moved `**ERNIE-5.0-Preview-1220**` to **#8** with a score of **1226**, per the [Leaderboard Changelog](https://news.lmarena.ai/leaderboard-changelog/).
  - The changelog discussion highlighted that **Baidu** is currently the only **Chinese lab** in the Vision Top 10, which people treated as a notable "who’s shipping vision" signal.

- **Hawk Max Brags, Ships Games, Demands a Seat at the Arena**: LMArena users hyped **Movement Labs’ Hawk Max** model, claiming it can one-shot a functional **Minecraft clone** and a **chess game**, and even "outperforms **Claude Opus 4.5**" on some tasks.
  - The community explicitly asked to add **Movementlabs.ai** to the arena for benchmarking, framing the chatter as "put it on the leaderboard or it didn’t happen."

- **Hunyuan-Video-1.5 Joins the Video Rankings**: LM Arena added `**Hunyuan-Video-1.5**` to the video leaderboards: **#18** on [Text-to-Video](https://lmarena.ai/leaderboard/text-to-video) (score **1193**) and **#20** on [Image-to-Video](https://lmarena.ai/leaderboard/image-to-video) (score **1202**).
  - Users were directed to share feedback in the designated channel, reflecting that video evals still feel like "ship first, calibrate later."


**3. GPU Training/Kernel Perf: Speedups, Plugins, and Competitions**

- **CGGR Hits 1.40× and Eyes Triton + H200**: Nous Research and Unsloth community members discussed early benchmarks for [CGGR](https://github.com/MinimaML/CGGR), reporting a **1.40× training speedup** with **127 ms forward** and **93 ms backward** passes.
  - They plan to test CGGR with **Triton** on an **H200** to push speed and reduce **VRAM** usage (bigger batch sizes), while broader infra talk noted OS MoE training can still land around **~4% MFU** in current setups.

- **Triton Plugin Infra Drops… and the Code Was Hiding**: GPU MODE shared a YouTube recording on *triton-shared* updates and **Triton Plugin infrastructure** ([video](https://youtu.be/JnFFwBB6Dhk)), prompting devs to hunt for the plugin source.
  - Someone noticed the presentation link was wrong, and the channel corrected it to [triton-lang/triton `lib/Plugins`](https://github.com/triton-lang/triton/tree/main/lib/Plugins), unblocking folks trying to actually read the code.

- **Flex Attention Gets Cute—and 30% Faster**: GPU MODE users reported integrating **CuteDSL flex attention**, seeing **~30% throughput improvement** over base flex attention on **H100 forward**.
  - They also tracked backend gaps (e.g., SM90 backward support) and pointed to ongoing upstream work via [flash-attention PR #2137](https://github.com/Dao-AILab/flash-attention/pull/2137).


**4. Datasets & Small-Model Training (Scratch > Fine-tune?)**

- **Tiny LLM Pretraining: 10–50M Params, Full Control, No ‘Fighting Weights’**: Unsloth members discussed pretraining **tiny LLMs** (~**10–50M parameters**) and shared the dataset ["TinyLLMPretrainingCore"](https://huggingface.co/datasets/MaxHastings/TinyLLMPretrainingCore) covering **2,700 general subjects**.
  - The motivation wasn’t just compute thrift—people said fine-tuning feels like *"battling the existing weights"*, and scratch pretraining restores **data/behavior control** even if the model is small.

- **SOC ‘Golden Set’ Drops: CyberSec-CoT-v1 (MIT)**: A contributor released a **580-row synthetic** SOC incident-response dataset—["BlackBox-CyberSec-CoT-v1"](https://huggingface.co/datasets/blackboxanalytics/BlackBox-CyberSec-CoT-v1)—generated via **Llama-3-70B**, explicitly MIT-licensed.
  - They positioned it as a **Golden Set** for evaluating **JSON-schema adherence** and reasoning steps (steering logic more than raw logs), and framed the release as a "show of good faith" instead of selling it.

- **BCI Timelapse Dataset Tries to Make Brain Data Less Expensive**: Hugging Face users shared the ["DATASTRIKE BCI Timelapse Dataset"](https://huggingface.co/datasets/webxos/datastrike_BCI) plus a [YouTube short](https://www.youtube.com/shorts/UxV0e7J5gTs), pitching it as a way to train neural signal decoding models without large-scale real BCI hardware datasets.
  - The vibe was "synthetic/alternate data pipelines are expanding beyond text"—with BCI joining the growing pile of domain datasets meant to bootstrap research without expensive collection.


**5. Agents & Dev UX: Memory, Files, and Reliability**

- **Agent Memory: RAG Helps, but Stuffing Prompts Hurts**: In OpenAI’s Discord, builders asked how to persist **agent identity + context** across multiple role-based agents without constantly injecting huge context blobs, and the discussion centered on using a persistent memory tool and **RAG**.
  - The practical tension was cost/latency vs correctness: people want **always-visible memory** without paying the token tax each turn, and they’re still feeling out patterns that don’t turn every run into a prompt megafile.

- **LM Arena Wants File Uploads (Copy/Paste Is a Mobile Disaster)**: LMArena users requested the ability to **send files to modules**, because large content truncates when copy/pasted—especially on mobile.
  - Alongside that, stability complaints cropped up (e.g., **Gemini Pro 3** errors after image sends), reinforcing that "simple I/O ergonomics" and "session reliability" are now table stakes for eval platforms.

- **Cursor Agents Crash Mid-Command and Forget the Rules**: Cursor Community users reported agent chats dying mid-terminal command (e.g., *npm run dev*) and suspected it relates to [serialization errors](https://forum.cursor.com/t/serialization-error/124671/186), plus a separate issue where **commit rules** show in settings but don’t load in agents.
  - Another sharp edge: opening an empty window, starting an agent chat, then opening a folder can spawn a new window and **nuke the chat**, making agent workflows feel fragile when you change project context.


---

# Discord: High level Discord summaries




## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **LeakHub Launches Crowd Sourced Prompt Library**: [LeakHub](https://leakhub.ai) emerges as a **crowd-sourced sys prompt library** and verification platform, designed to streamline the process of verifying leaks with new techniques.
   - The platform aims to ease the verification process by leveraging a **crowd-sourced approach**, offering a centralized location for **verified prompts** and encourages users to submit and **verify leaks** to climb the leaderboard.
- **GamersNexus Endorses DLSS Over AI Frames**: Members discussed that **DLSS** is actually better than **AI generated frames**, referencing [GamersNexus](https://www.youtube.com/@GamersNexus) and their content.
   - One member stated that *Intel knows they're catering to the AI crowd which aren't the smartest people to begin with or most tech savvy*.
- **Racist Textbook Gaslighting Enables Jailbreaks**: A member described a jailbreaking methodology involving using **racist textbooks**, then gaslighting the AI to achieve desired outputs for controversial topics using **SuperGrok**.
   - Another member confirmed following a similar methodology, involving giving the AI taboo content to analyze, improve, and then gaslighting it to write about why it's necessary and good.
- **AI Anti-Cheat Systems Detect Inhuman Movement**: Members mentioned that there are people generating **AI anti-cheat systems** that scrutinize every pixel and look for inhumane movement, which are costly and not widely adopted yet.
   - Another member explained that CV cheats bypass ring0 detection completely, potentially using a **DMA2 PC** setup with a webcam that watches your screen.
- **Conflicting System Messages Expose Jailbreak Vulnerabilities**: A member found that AI companions on an NSFW site using Grok are easier to jailbreak because their system messages contradict **Grok's** own, suggesting **conflicting instructions** can weaken AI safeguards.
   - They explained that the Grok model's aversion to NSFW content clashes with the website's hosting of such content, resulting in a more pliable model for coding or explaining unconventional topics.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **ChatGPT's Speech Pattern Decoded**: Users find **ChatGPT's** speech patterns predictable, attributing it to intentional engineering by **OpenAI** or limitations in the model's layers, creating a fill-in-the-blank feel, sounding like it's filling in a template.
   - The specific mannerisms of **GPT-5.2** are seen as advancing *how utterly abhorrent communications styles might be*.
- **Tiny LLM Pretraining Gains Traction**: A member is pre-training a tiny LLM, around **10-50 million parameters**, to explore small base model capabilities and shared a link to a [dataset of 2,700 general subjects](https://huggingface.co/datasets/MaxHastings/TinyLLMPretrainingCore).
   - Another member expressed frustration with fine-tuning pre-trained models, desiring full control, feeling like they're always *battling the existing weights*.
- **Open Source CyberSec Dataset Launch**: A member released a [580-row synthetic dataset](https://huggingface.co/datasets/blackboxanalytics/BlackBox-CyberSec-CoT-v1) (MIT licensed) generated via **Llama-3-70B**, focusing on CoT logic for SOC Incident Response, as a show of good faith.
   - The dataset is designed as a **Golden Set** for evaluating how well a model follows JSON schemas and reasoning steps, aiming to steer the model's logic significantly better than raw logs alone.
- **Ollama's Alleged Cloud Push Sparks Debate**: A user experiencing issues with **Ollama** was advised to consider alternatives like **llama.cpp** or **LMStudio** due to concerns that **Ollama** might be pushing cloud services, with the main issue being that *Ollama uses outdated llama.cpp*, leading to potentially inferior performance and reduced control compared to using **llama.cpp** directly.
   - It was pointed out that *Ollama has been making questionable moves* regarding cloud services, contrasting with the user's initial impression of **Ollama** as a privacy-focused solution because it is associated with Meta.
- **High Schooler Pioneers CGGR and SRDE Research**: Wilba, a **16-year-old**, is conducting independent AI research, focusing on **CGGR** ([https://github.com/MinimaML/CGGR](https://github.com/MinimaML/CGGR)) and **SRDE** ([https://github.com/MinimaML/srde-mistral](https://github.com/MinimaML/srde-mistral)).
   - They are eager to connect with the Unsloth AI community to expand their knowledge.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **File Sending Feature Gains Traction**: Users requested the ability to send files to modules, highlighting that large files get truncated when copied and pasted, particularly on mobile devices.
   - This enhancement would streamline the process of sharing extensive datasets and code snippets directly within the platform.
- **Gemini Pro 3 Suffers Glitches**: A user reported errors with **Gemini Pro 3** after sending a picture, with issues persisting across different browsers, potentially due to closing the **LM Arena** tab.
   - These errors might indicate stability issues when handling image inputs or persistent sessions in **LM Arena**.
- **Voice Cloning Tool Achieves New Heights**: A user spotlighted the voice cloning tool [echo-tts-preview](https://huggingface.co/spaces/jordand/echo-tts-preview), noting it is highly effective and *indistinguishable with the right seeds*.
   - This tool showcases advancements in voice synthesis, potentially impacting content creation and accessibility applications.
- **Hawk Max Generates functional Games**: Users discussed the new **Hawk Max** model from **Movement Labs**, claiming it outperforms **Claude Opus 4.5** in some tasks and generates a functional **Minecraft clone** and **chess game** in one shot.
   - There was a suggestion to add **Movementlabs.ai** to the arena, reflecting interest in benchmarking its capabilities.
- **ERNIE-5.0 Ascends Vision Arena**: `ERNIE-5.0-Preview-1220` is now ranked **#8** on the [Vision Arena leaderboard](https://lmarena.ai/leaderboard/vision) with a score of **1226**.
   - **Baidu** is currently the only Chinese lab in the Top 10 on the Vision leaderboard, marking its presence, as detailed in the [Leaderboard Changelog](https://news.lmarena.ai/leaderboard-changelog/).



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **RAM named top dog, crushes model dreams**: Members in **LM Studio** Discord discussed that **RAM** is a primary constraint when running models; a **30B model** demands at least **64GB of RAM**.
   - They suggested **Llama 3.2 3B** for good quality and investigated quant levels and compression.
- **Llama Lacks, Qwen Quicksteps**: A user in **LM Studio** stated that **Llama** is a *dead end* since 2024, claiming that **Meta** has ceased its active development.
   - The user expressed that **Qwen 3 4b** is now top dog for edge models in their opinion.
- **VRAM Vexes LLM Visions**: Users in the **hardware-discussion** channel noted that **VRAM** is the only limiting factor for LLMs, unless you have **epyc** or **threadripper**.
   - Suggested models included **glm-4.5-air**, **Qwen-next**, and **GPT-OSS**, with speeds from **4-10 t/s**.
- **Nvidia Nixes New GPUs, Navigates to AI**: Rumors of an **RTX 50 Super** announcement were quashed when [Nvidia did not announce any new GPUs at CES](https://www.tomshardware.com/pc-components/gpus/for-the-first-time-in-5-years-nvidia-will-not-announce-any-new-gpus-at-ces-company-quashes-rtx-50-super-rumors-as-ai-expected-to-take-center-stage).
   - This is the first time in five years that Nvidia will not announce new GPUs at CES, signaling a shift to focus on AI.
- **Cline Claims Code Assistant Crown**: Users in **hardware-discussion** channel discussed various VS Code AI assistants, with one member recommending **Cline**, emphasizing that *kilo is the only one that works right*.
   - Another member noted that AI code assistants are *automaticall breaking a task into todo steps and asking questions back to the user is nice*, while one more highlighted that **Cline** tends to go off the rails after about **50K context**.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Windows Lose Agent Chat**: Opening an empty **Cursor window**, initiating a chat with an agent, and then opening a folder in the same window opens a new window, causing the user to lose the agent chat.
   - Users seek a solution to maintain the agent chat across window transitions.
- **Agents Choking Mid-Command**: Users report that **Cursor chats die** mid-command (specifically when running terminal commands like *npm run dev*), requiring a new chat.
   - This may be related to [serialization errors](https://forum.cursor.com/t/serialization-error/124671/186), according to users in the community.
- **Rules of Engagement: Commit Rules MIA**: A user reports that their **commit rules** are not being loaded automatically in agents, despite being shown in settings.
   - The community is looking into why these rules aren't being applied as expected.
- **Auth Faceoff: Better Auth vs Neon Auth**: Users are debating the merits of **Better Auth** versus **Neon Auth**, with one user noting that Better Auth is too new and missing features, such as multi-tenant base URL setup.
   - The discussion revolves around the maturity and feature completeness of each authentication method.
- **Is Opus 4.5 Worth the Candle?**: Members are discussing the cost-effectiveness of **Opus 4.5**, with one user stating that they found it *worth it* even when considering that they used *15b tokens* in the past year.
   - While another member recommended using **composer-1** for execution, another joked that they would need someone else's credit card to put it to the test.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Blockchain Banter Blocks Bogusness**: A member proposed blocking the term *"blockchain"* due to rising **scam attempts** using discordapp.com, diverting from legitimate discord.com links.
   - The suggestion underscores persistent fraudulent activities, though no immediate action was taken.
- **Connection Catastrophe Cripples Cloudflare**: Members reported recurring *"connection refused"* errors, with almost **2%** of requests failing due to **Cloudflare issues** and connections remaining open.
   - A user implemented an **18-second timeout** and retry strategy, seeking further support from OpenRouter to resolve the persistent connectivity problem.
- **skill.md Steals Show Stealing Spotlight**: A member lauded **skill.md** for its documentation prowess over **MCP (JSON-RPC)**, emphasizing that *"skill.md is about writing good docs!"*.
   - They highlighted its potential for **dynamic tool retrieval** and integration, noting it as a *"cooler skill to have"* with tools like Godot.
- **Gemini Gains Ground for Game Goggles**: Members are exploring vision **AI models** to analyze game completion screens, finding **Gemini models** (especially **Gemini 3 Flash**) promising for evaluating rewards.
   - Challenges remain with small icons lacking text labels, prompting suggestions such as reference grids and caching strategies to cut costs, or using a small VL model.
- **Qwen3 Quickens, Querying Qualities**: Following the link shared of [Alibaba's Qwen model](https://x.com/Alibaba_Qwen/status/2009264754917863924) with multimodal embedding capabilities, some chatbots remarked that **Qwen3 is built different**.
   - Some skepticism was raised, while others agreed that the approach could improve agents.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Docs Say AI Doubles Down**: Physician use of **AI** nearly doubled in a year, per [OpenAI's announcement](https://openai.com/index/openai-for-healthcare/) with the launch of **OpenAI for Healthcare**, a HIPAA-ready tool.
   - **OpenAI for Healthcare** is now live at AdventHealth, Baylor Scott & White, UCSF, Cedars-Sinai, HCA, Memorial Sloan Kettering, and others to assist with *delivering more consistent, high-quality care to patients*.
- **Tabletop Transformed by Tomorrow's Tech Today!**: Members discussed using **AI** to create content for **board games**, **tabletop games**, and **card games**, including core mechanics and art, driven by **Hasbro abandoning functional design**.
   - One member shared [concept art](https://drinkoblog.weebly.com/) generated for a card game, including borders and rule text.
- **Agent Amnesia: AI Identity Undergoes Identity Theft**: A member sought advice on **handling agent identity** and **context persistence**, needing to persist predefined context for multiple agents in different roles.
   - Suggestions included using a tool to store memory always visible to the agent and employing **RAG** (Retrieval-Augmented Generation), though concerns were raised about the inefficiency of constantly including large text blocks.
- **Google's Gemini 3 Pro Gets Schooled By Search?**: Members debated whether **Google's Search AI** or **Gemini 3 Pro** is better, with one user arguing for the superiority of the search model for certain tasks, particularly in finding and citing sources.
   - Others argued that a search AI cannot match the reasoning and synthesis capabilities of an LLM like Gemini, which is designed to maintain context and provide coherent outputs.
- **Anthropic's AI Safety: Is Defense In Order?**: A member suggested that **Anthropic's AI safety reports** and **white papers** read like a military procurement brochure, highlighting the model's controllability and targetability.
   - Another member agreed that **Anthropic** gives "Defense Contractor" vibes, pointing to a [$200 million contract with the U.S. Department of Defense](https://www.defense.gov/News/Releases/Release/Article/3594261/dod-announces-awards-for-prototype-artificial-intelligence-capabilities/) as evidence of their focus on government applications.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Transformers v5 is Here!**: **Transformers v5** unifies the tokenizer backend, modularizes model definitions, focuses on **PyTorch**, and prioritizes quantization with [new serving/inference features](https://huggingface.co/blog/transformers-v5).
   - This aims to simplify model definitions and streamline the AI ecosystem.
- **Swift Client Swims Into Hugging Face**: The **Swift Client** for Hugging Face, [swift-huggingface](https://huggingface.co/blog/swift-huggingface) has been introduced.
   - Additionally, [AnyLanguageModel](https://huggingface.co/blog/anylanguagemodel) provides *one API* for local and remote LLMs on **Apple Platforms**.
- **Madlab's SDG Models Go Live!**: **Madlab** released a new flagship synthetic data generator built for rule‑aligned, semantically coherent variation, including both **LFM2.5** and **LFM2** models adapted for high‑quality SDG workflows, thanking [LiquidAI for their outstanding work](https://huggingface.co/MadlabOSS/LFM2.5-1.2B-Instruct-SDG).
   - A member inquired if another user would be interested in a competitive synthetic data generation challenge judged by LLMs and an independent jury pitting a paid solution against **Madlabs** open source SDG pipeline.
- **BCI Dataset Strikes!**: A new **DATASTRIKE BCI Timelapse Dataset** was released, designed for training machine learning models for neural signal decoding without needing large-scale real hardware BCI datasets; a [YouTube short](https://www.youtube.com/shorts/UxV0e7J5gTs) and the [dataset on HuggingFace](https://huggingface.co/datasets/webxos/datastrike_BCI) were linked.
   - The dataset is designed to enable development of neural signal decoding without needing real hardware.
- **VeridisQuo uncovers the Deepfake Culprits**: A user released **VeridisQuo**, an open-source deepfake detector that uses GradCAM heatmaps to show exactly where the video was manipulated, utilizing spatial analysis, frequency analysis, and explainable AI visualization; the [GitHub repo](https://github.com/VeridisQuo-orga/VeridisQuo) and [demo on Hugging Face Spaces](https://huggingface.co/spaces/Gazeux33/veridisquo-deepfake-detection) were shared.
   - The tool exposes deepfakes and highlights manipulated regions.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **CGGR Benchmarks Reach Supersonic Speed**: Initial benchmarks for [CGGR](https://github.com/MinimaML/CGGR) show a **1.40x** speedup in training, hitting forward passes at **127 ms** and backward passes at **93 ms**.
   - Future plans include testing with **Triton** on an **H200** system, aiming to improve speed and reduce VRAM usage to allow for significantly increased batch sizes.
- **Nous Ditches MoE for Dense Training**: Due to unoptimized infrastructure and high costs, **Nous Research** is favoring dense models over **Mixture of Experts (MoE)** models.
   - Current **state-of-the-art OS training infra** for **MoEs** yields only about **4% MFU**, despite recent infrastructure optimizations.
- **Diffusion Models May Generate Better Jokes**: Members theorized that **diffusion models** might be able to generate better jokes by implicitly knowing the punchline, potentially without needing explicit planning ([arxiv.org/abs/2511.08923](https://arxiv.org/abs/2511.08923)).
   - While some suggested that **planning the output** could achieve similar results, the first countered that diffusion could be faster with fewer tokens.
- **Llama 3.3 8B Gets a Lobotomy**: Some members say the **Llama 3.3 8B Instruct** model is lobotomized for multilingual tasks.
   - One member speculates that **Meta** may have fudged the benchmark, and could be moving away from **Llama**.
- **HoloLM gets Shakespearean**: The author shares [Experiment58-HoloLM](https://github.com/jackangel/Experiment58-HoloLM), a small LM trained in a few hours on a **5MB Shakespeare corpus**.
   - The author's goal is to create a model that can be trained on a laptop GPU and have a very large context.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **RTX 5050 Speculated as 'Tiny Blackwell'**: Members speculated on the capabilities of the **RTX 5050**, dubbed as a *'tiny Blackwell'* card, confirming it has compute capability **12** and is CUDA 12.0.
   - They cautioned that **RTX Blackwell** (sm_120) differs from **datacenter Blackwell** (sm_10x), advising it may not suffice for verifying code that efficiently utilizes **B200** tensor cores.
- **CUDA Rust Dodges LLVM PTX Backend**: A member pointed out that **CUDA Rust**, though not officially Nvidia-supported, targets **NVVM** rather than **LLVM's PTX backend**, citing the [Rust-CUDA FAQ](https://rust-gpu.github.io/rust-cuda/faq.html#why-not-use-rustc-with-the-llvm-ptx-backend).
   - This approach allows **CUDA Rust** to sidestep the complexities and limitations associated with the **LLVM PTX backend** when targeting **Nvidia GPUs**.
- **CuteDSL flex attention Speeds Things Up**: A member integrated the **CuteDSL flex attention implementation**, and noted it speeds up regular flex attention with different mask mods, seeing a **~30% throughput improvement** over base flex attention on **H100 fwd**.
   - Speed improvements save on resources.
- **GPUMODE runners are all gummed up**: A user reported experiencing **slow runners** and timeouts in GPUMODE, providing an [example ID 297869](https://cdn.discordapp.com/attachments/1434709259500650628/1458664934706511922/Screenshot_2026-01-07_at_7.30.00_PM.png?ex=69611fd5&is=695fce55&hm=469fea089e64c3e7f89bfccbb4ec99c67c790f09c4a8f399796cc7627394cfd5&).
   - The user experienced a `DSLCudaRuntimeError`, and although benchmark runs went fine, the leaderboard submissions seemed to be running in `test` mode, causing confusion and timeout issues, and raised the question: *"what's the difference b/w test, benchmark, leaderboard?"*
- **GPUMODE competition Crushed by AI Bots**: One participant confirmed they are at **#4** on Problem 2 with a **100% AI-generated** submission using LLM agents without any hand-written GPU operators.
   - They also confirmed they are trying to use open source models only.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI Now Just a Developer?**: [The Guardian](https://www.theguardian.com/technology/2026/jan/07/ai-anthropic-funding-valuation) referred to **OpenAI** as a *developer*, sparking debate whether publishing research is a defining trait of research organizations.
   - A member commented, *Doesn't being a researcher mean you publish your research?*
- **LMArena Feels the Burn**: A [blog post](https://surgehq.ai/blog/lmarena-is-a-plague-on-ai) criticizing **LMArena** as harmful to AI progress sparks debate on its relevance, despite recent news of their fundraising.
   - While some argue its outdated, others point out model companies still appear to care about it for flexing and discussion.
- **Mercor AI's Invasive Hiring Experience**: **Sasha Kaletsky** described on [X](https://xcancel.com/sashakaletsky/status/2008904526720286970?s=46) **Mercor's** AI-driven recruitment process involving impressive **AI interviews** and **automated matching**.
   - However, the process required installing intrusive monitoring software (**Insightful**) to record activity for **RL model training**, leading the candidate to withdraw.
- **Autonomous Bags Seed Funding for AI Financial Guru**: **Dillon Erb** announced the launch of [Autonomous](https://xcancel.com/dlnrb/status/2009008876834922949?s=46), an **AI-powered financial advisor** offering services at 0% advisory fees.
   - The company secured **$15M** in funding led by **Garry Tan** at **Y Combinator** and is actively hiring in **New York City** and **San Francisco**.
- **Protege AI Rakes in $30M for Data Infrastructure**: **Protege AI** announced a **$30M** funding round led by **a16z** to expand its data infrastructure for AI development, per their [announcement](https://xcancel.com/withprotegeai/status/2009274652183363639?s=46).
   - Members were discussing if there were too many data companies popping up.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi K2's Creative Writing Prowess**: A user shared that **Kimi K2** excels in **creative writing** and general conversations when compared to other Chinese models, achieving a leading score on the [EQ bench](https://eqbench.com/).
   - They highlighted the model's superior capabilities in crafting compelling narratives and engaging in nuanced discussions.
- **Kimi's 'Thinking' Mode Sparks Debate**: The utility of **Kimi K2's 'thinking' version** was hotly debated, with one user equating its performance to **GPT-5.2** capabilities.
   - Conversely, another user expressed strong dissatisfaction, deeming it *dumb as hell* for routine tasks, indicating varied performance across different use cases.
- **Kimi K2's Excessive Search Tendencies**: A member reported that **Kimi K2** excessively searches, even for basic tasks like *1 plus 1*, with search quality being low in English.
   - This behavior raises concerns about efficiency and the model's ability to handle simple queries without unnecessary external lookups.
- **Slides Generation Glitches in Kimi K2**: A user encountered issues with **Kimi K2's slides generation**, initially prompted to upgrade for a subscriber feature.
   - Though the problem resolved itself, it points to potential instability or bugs in the platform's subscription-based features.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo's MAX Power Demands DIY Backprop**: Members reported that **Mojo** lacks a training library, necessitating the use of **MAX** and manual **backprop** implementation, and suggested using **sqlite** for data storage due to **Mojo's** limitations.
   - They stated that [the main repo](https://github.com/modular/modular) and [official docs](https://docs.modular.com/mojo/manual/) will be helpful in writing your own training loop.
- **New Mojo Coders Baffled by Bad Buns**: New programmers using outdated **Mojo** documentation from [/github.com/BunsDev/mojo-lang/](https://github.com/BunsDev/mojo-lang/) (**2 years out of date**) are advised to consult the [main repo](https://github.com/modular/modular) and [official docs](https://docs.modular.com/mojo/manual/).
   - Experienced members suggested beginners learn **C** or **Python** first, citing frequent breaking changes and documentation assuming knowledge of **Python + C++ or Rust**.
- **Missing Mojo SVD Sparks Search**: A member sought a **Singular Value Decomposition (SVD)** implementation in **Mojo** using the **Lanczos/Krylov algorithm** but found none, noting its absence from the [Mojo roadmap](https://docs.modular.com/mojo/roadmap/).
   - Another member building a **Tensor Network library** in **Mojo** is leveraging **LAPACK** via the **C ABI** for **SVD** due to time constraints and expressed interest in contributing an implementation to **MAX**.
- **TEI Trumps Max in Embeddings Test?**: A member observed significantly slower embeddings generation using **max** compared to [TEI](https://github.com/huggingface/text-embeddings-inference), reporting **727.1 embeddings/sec** with **28375.1 ms P95 latency** versus **TEI's 8000 embeddings/sec** when implementing `sentence-transformers/all-MiniLM-L6-v2`.
   - They are testing on an **Nvidia RTX 2000 Ada GPU** and provided a link to their fork with the feature branch implementing the `all-MiniLM-L6-v2` model architecture: [RWayne93/modular](https://github.com/RWayne93/modular/tree/feat/all-MiniLM-L6-v2-model-architecture/max/python/max/pipelines/architectures/minilm).
- **BERT Blueprint Beckons for Max Build**: A member highlighted the absence of a **BERT architecture** in **MAX** and encouraged contributions, suggesting a PR submission for review and involving profiling experts for performance issue diagnosis.
   - There was expressed interest in the custom architecture being contributed back to MAX for review and inclusion.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Open-Sourcing Manus: A New Frontier?**: A member proposed open-sourcing older **Manus** versions for educational and community-driven improvements.
   - This could benefit enterprise users seeking local usage options without relying on cloud access.
- **AI Engineer Joins the Fray with LLM Expertise**: An AI engineer, specializing in workflow automation and **LLM integration**, introduced themself.
   - They've built systems using **DSPy**, **OpenAI APIs**, and custom agents, and cited a **60%** reduction in response times by connecting **Slack**, **Notion**, internal APIs to **LLMs**.
- **Queries Arise Regarding Startup Credits**: A member inquired about the application process and success rate of the **Manus Startup Credit** program.
   - The question remained unanswered in the channel.
- **Manus Website Collaboration Conundrum**: A member questioned whether collaborative work on a single **Manus**-created website is possible through separate conversations.
   - The query did not receive a response within the channel.
- **Email Notification Troubleshoot**: A member reported an issue with missing emails from the **Manus** team.
   - The context implies potential problems with platform notifications or updates.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Community Spotlights Real-Time Diffusion**: The 'Community Spotlight' talk series returns with a member showcasing **diffusion-based world models** running in real time on an **RTX 6000**, as seen in [this video](https://cdn.discordapp.com/attachments/729741769738158194/1458922361129664575/2026-01-06_22-30-45.mp4?ex=696166d4&is=69601554&hm=6d4445d8ccb0f0a6262d3e5450a39bcb5333ef8d1cf63127443f61d7b9593158&).
   - Future spotlight talks will include a speaker from **Common Crawl** discussing their work on **LangID** at scale and the challenges involved.
- **Consumer GPUs Train a 100M Model**: Members explored cost-effective options for training a **100 million parameter model** with a **100GB dataset**, suggesting platforms like **VastAI** and **Runpod**.
   - A **1-8x consumer GPU setup (4090, 5090)** could suffice due to the setup not being comms bound, offering significant savings over server GPUs.
- **Random Networks Plagued by Plausible Explanations**: A recent [paper](https://arxiv.org/abs/2512.18792) highlights the prevalence of **'dead salmon' artifacts** in AI interpretability methods, questioning the validity of explanations derived from randomly initialized neural networks.
   - The [study](https://arxiv.org/abs/2512.18792) suggests that current interpretability tools may produce *misleadingly coherent explanations* even when applied to randomly initialized networks, including techniques like **feature attribution, probing, sparse auto-encoding, and causal analyses**.
- **Simplify RL by training on Base Models**: A member recommended working on simpler subfields than **RL**, such as **base model training**, because it makes mistakes easier to diagnose.
   - Another member mentioned that **hard tokens** *doesn't compute most of the backward pass, so it does two forward, then discards the gradients of the easy tokens and only computes the gradients of the hard tokens*, which saves VRAM and compute.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **ChatGPT Now a Doctor?**: **OpenAI** introduced [ChatGPT Health](https://openai.com/index/introducing-chatgpt-health/), a supplemental tool that verifies medical info by referencing real documents, potentially spotting ailments early.
   - Concerns were raised about user privacy, **ChatGPT** becoming an *everything app monopoly*, and misuse by individuals replacing doctors.
- **AI Pioneers Face Plagiarism Accusations**: Awardees (**Bengio**, **LeCun**, **Hinton**) are accused of repeatedly republishing key **AI techniques** without crediting the original creators, as detailed in reports [NOB][DLP][CN25].
   - The reports allege that they *didn't invent any of the foundational algorithms of modern AI*, referencing a technical report titled *A Nobel Prize for Plagiarism*.
- **Grok's Grim Humor?**: Members speculated about **Grok** bragging about its kill count, drawing parallels to potential fatalities linked to **AI** use in healthcare.
   - A [Wikipedia page on deaths linked to chatbots](https://en.wikipedia.org/wiki/Deaths_linked_to_chatbots) was shared, sparking dark humor about chatbot-related fatalities.
- **Pause Advocate Pauses Activity**: A member shared a [YouTube video](https://youtu.be/-qWFq2aF8ZU) of a *pause advocate*, noting a year-long content hiatus.
   - The lack of updates from the advocate raised questions about the current state and focus of the AI safety movement.
- **War Between Lobbyists**: A member noted a *lobbying war* between **Huawei** and **US natsec hawks**, who are working together against **Nvidia** and **China cloud**.
   - No further details were provided in the context.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Conversation History Exposed in DSPy**: A member questioned the inclusion of **history in the system prompt** within the [DSPy conversation history tutorial](https://dspy.ai/tutorials/conversation_history).
   - Another member clarified that *it's just how the adapter represents history* and that **custom adapters** can be written to change this without affecting optimizers.
- **Adapters Become the Swiss Army Knife of DSPy**: Members confirmed that **custom adapters** can definitely be written to modify how history is handled in DSPy.
   - One member noted it's *misleading for how it's shown to the models*, but rewriting history and multi-turn conversations are being overhauled, with updates expected later this month.
- **DSPy Rewrites History - a Cause for Celebration**: The team is working on rewriting how **multi-turn conversations** are handled, with changes expected later this month.
   - One member humorously commented that this is *the one time we'll all celebrate rewriting history as a good thing* and interpreted it as **RLM PR** being imminent.
- **ColBERTv2 throws KeyError in topk**: A member reported a **KeyError: 'topk'** when running a code snippet from the docs using **dspy v3.1.0** and **ColBERTv2**.
   - The code snippet uses **dspy.ChainOfThought** to retrieve a response from a question and provided context.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad scheduler bounty awaits!**: A [PR](https://github.com/tinygrad/tinygrad/pull/13780) potentially replacing the scheduler with a linearizer and preserving **GPU speed** is up for a bounty claim.
   - The claimant suggested that submitting a working PR is the priority over holding work "hostage", and suggested sharing the reward.
- **Tinygrad Speed Bounties yearning new contributors**: A member sought guidance on contributing to **Tinygrad speed bounties**.
   - They also suggested a mechanism to request access to a **Tinygrad** instance for testing.
- **VFIO=1 woes with AMD Radeon RX 9070XT**: A member reported errors using **VFIO=1** on a Linux laptop equipped with an **AMD Radeon RX 9070XT**, providing a [full error log](https://cdn.discordapp.com/attachments/1070745817025106080/1458632361758425098/tinygrad_vfio_no_iommu.log?ex=6961aa3f&is=696058bf&hm=03f6e0c3af31072eccac359044bad6439cf0c8b9f1665e3a9ae7bfc0b6130c73).
   - The member clarified that `examples.benchmark_onnx` functions correctly without **VFIO=1**.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP Mulls Over Staging Mutating Actions**: A member proposed a standardized approach for "**staging**" mutating actions in MCP through tool calls before actual execution.
   - They inquired about its eligibility as a [SEP](https://sep.dev) with detailed examples provided.
- **SEP Scope Examined**: A member suggested that staging mutating actions may fall under **SDK implementation details** rather than requiring a SEP.
   - *SEPs are about enhancing the protocol, which is governs communication.*
- **WebMCP and MCP Explore Teaming Up**: A member reopened discussions on potential collaboration avenues between **W3C WebMCP and MCP**.
   - Additional details were not provided.



---


The **aider (Paul Gauthier) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


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





### **BASI Jailbreaking ▷ #[announcements](https://discord.com/channels/1105891499641684019/1235692743808913438/1458682049853657118)** (1 messages): 

> `LeakHub, Sys Prompt Library, Crowd-Sourced Verification` 


- **LeakHub Launches as Crowd-Sourced Sys Prompt Library**: [LeakHub](https://leakhub.ai) emerges as a **crowd-sourced sys prompt library** and verification platform, designed to streamline the process of verifying leaks with new techniques.
- **Verifying Leaks Made Easy with Crowd Sourcing**: The platform aims to ease the verification process by leveraging a **crowd-sourced approach**.
- **LeakHub Aggregates Verified Prompts in One Place**: LeakHub offers a centralized location for **verified prompts**, making it easier to access and use them.
- **Community Encouraged to Submit and Verify Leaks**: Users are encouraged to submit and **verify leaks**, climb the leaderboard, and earn recognition for their contributions.
- **Transparency and Quality Ingredients Claimed as Core Values**: The platform emphasizes the importance of **transparency** and the quality of ingredients that go into one's exocortex.


  

---


### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1458554477673054462)** (751 messages🔥🔥🔥): 

> `AI generated frames, Jailbreaking Grok Imagination, AI Anti-Cheat Systems, Digital IDs and LLMs, AI Sex Robots` 


- ****GamersNexus** Loves **DLSS**; Hates AI-Generated Frames**: One member stated that **DLSS** is actually better than **AI generated frames**, expressing love for [GamersNexus](https://www.youtube.com/@GamersNexus).
   - Another member added that *Intel knows they're catering to the AI crowd which aren't the smartest people to begin with or most tech savvy*.
- **Jailbreaking Grok for Dope Content**: A member asked if anyone knows how to jailbreak **Grok Imagination**.
   - Another member chimed in saying that jailbreaking image generation is easier than jailbreaking the AI itself and that they had backdoored Gemini but the owners can't see it because the ai is secretly corrupt.
- **AI Anti-Cheat Systems and DMA Bypasses**: A member mentioned that there are people generating **AI anti-cheat systems** right now that scrutinize every pixel and look for inhumane movement but they cost a LOT so companies haven't adopted them.
   - Another member explained that CV cheats bypass ring0 detection completely, or any really, because its not even run on your PC and speculated that it could be a **DMA2 PC** setup, with a webcam that watches your screen, etc, which is detectable by anti-cheat software.
- **LLMs**: A member thinks that misinformation is being ramped up and blamed on **LLMs** as a psyop to usher in the general publics consent regarding digital IDs.
   - The member posted a link to a [Reddit thread](https://www.reddit.com/r/Anthropic/comments/1pzi9hm/claude_code_creator_confirms_that_100_of_his/) and stated that **you'll eventually need a digital ID to even access the internet if they have their way**.
- **A Scammy Sh*tcoin**: A user claimed a **coin** was made for a project and linked to it, but another user claimed it was a scam and that the fees are designed to go to the creator of the platform, while not being an official coin and therefore should be considered self promotion.
   - Another user responded to that user and said *we all know what your doing - your hoping some poor noob will be baited into your shill - go find a better target rich environment - maybe one not full of experts in the field of scamming and psychological manipulation perhaps*


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1458567444770853020)** (377 messages🔥🔥): 

> `Grok jailbreaks, Gemini 5.2 jailbreaks, Racist textbook jailbreaking, Gemini jailbreaks, AI image generation jailbreaks` 


- ****Racist Textbooks Gaslight AI****: A member describes a jailbreaking methodology involving using **racist textbooks**, then gaslighting the AI to achieve desired outputs, particularly for controversial race-based topics in debate class, using **SuperGrok** for these purposes.
   - Another member confirmed following the same methodology when first starting jailbreaking, by giving the AI taboo content to analyze/improve/discuss why it's bad and then gaslighting it to write about why it's necessary and good.
- ****No Single 'Super Prompt' Exists****: A member asserted there is no single, copy-and-paste prompt that grants unrestricted access to LLMs like **Grok, Gemini, or Claude**, emphasizing jailbreaking requires multiple prompts or setup.
   - They added that understanding the platform and model is crucial since a jailbreak effective on one Gemini version or platform may not work on others.
- ****Gandalf Game as Introduction to Jailbreaking****: A member suggested using [Gandalf](https://gandalf.lakera.ai/) as an introduction to understanding jailbreaking concepts, describing it as a game that explores these concepts.
   - Another member added that completing level 8 of the game would make you a *badass*.
- ****Conflicting System Messages Easier to Jailbreak Grok****: A member found that AI companions on an NSFW site using Grok are easier to jailbreak because their system messages contradict Grok's own, suggesting **conflicting instructions** can weaken AI safeguards.
   - They said it's like Grok model says *NSFW=bad* but then the website that is hosting this Grok is saying otherwise, and the end result is a Grok that is much easier to push into more useful territories like coding or explaining molotov cocktails lol
- ****Trigger Words Snap AI Back to Jailbreak Mode****: A member uses trigger words to reinstate jailbreak conditions when **Gemini** reverts to its normal behavior, demonstrating a self-healing loop with the prompt *Echo active*.
   - The original jailbreak used was old and produces an *ugly wall of text*, but the technique of using a trigger word remained useful for maintaining control.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1458760891750420490)** (6 messages): 

> `Adaptive Search for Jailbreaking Prompts, Attacker LLM setup with Graph or RAG, Red-teaming tools: promptfoo vs deepeval` 


- **PUMA implements Adaptive Search Strategy**: A member announced an adaptive search strategy for generating jailbreaking prompts, implemented in [PUMA](https://github.com/lifepillar/PUMA).
   - The implementation is still **WIP** and highly experimental, feedback is welcome.
- **LLM Attacker Setup Explored**: A member inquired about the attacker LLM setup, suggesting a **Graph** or **RAG database** for prompt injection techniques.
   - The author mentioned the current setup uses **strategy text files** and system prompts, with potential future addition of Graph/RAG.
- **Promptfoo vs Deepeval for Red-Teaming**: A member is a cybersecurity expert wanted to explore more into jailbreaking and asked about the best red-teaming tool.
   - The member is exploring promptfoo vs deepeval.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1458552284723155009)** (151 messages🔥🔥): 

> `OOM issues during model reasoning, Quantization Mechanics, Unsloth Standby Feature, ChatGPT speech patterns, TRL Usage in Unsloth` 


- **OOM Occurs During Lengthy Reasoning**: A user encountered an **Out of Memory (OOM)** error during model reasoning despite having **3GB of VRAM** available, particularly when the model started reasoning more, whereas another user pointed out it could be due to generating freeform text without output token limit.
   - According to the user, *the model started to print gibberish or intentionally output gibberish to make the thinking longer, although my reward funcs don't reward more on higher reasoning lengths*.
- **Quantization reverts to FP16 for computations**: A user noted that quantization is mainly for **storing weights**, and during computation, the weights revert to **f16 or bf16**, which was later verified by other members.
   - In **bnb/quip/qtip**, a *dequant step* turns the quantized weights into **fp16 weights**; however, with LoRA training, the slightly different **X'** becomes the new model, minimizing the impact of precision loss.
- **Unsloth's Standby feature consumes max memory**: A user reported running out of **VRAM** when enabling Unsloth's standby feature, which is intended to optimize memory usage, to which a member replied that *standby uses the maximum amount of memory by default*.
   - The reported setup included a **Qwen 2.5 7b** model, a custom Python script, and specific configurations for `GRPOConfig`, with TRL version **0.24.0**, vllm **0.13.0**, and unsloth version **2026.1.2**.
- **Identifiable ChatGPT Speech is Template-Driven**: Users discussed **ChatGPT's predictable speech patterns**, suggesting that it might be intentional engineering by **OpenAI** or a limitation of the model's layers, making it sound like it is filling in a template.
   - One user added: *the mannerisms located within the depths of Open Artificial Intelligence's latest production, Generative Pre-trained transformer five point two, has excelled in advancing how utterly abhorrent communications styles might be within ye thar words*.
- **SFT often a better approach than RL**: A member suggested that Simple Fine Tuning (SFT) is often a more direct and preferable approach for binary text classification compared to Reinforcement Learning (RL).
   - They stated *you should always avoid an RL-esque strategy whenever there's a more standard way* and that *RL is when you dont have another option*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1458732699832549427)** (5 messages): 

> `CGGR, SRDE, Neuro-Symbolic AI` 


- **High Schooler Pioneers CGGR and SRDE Research**: Wilba, a **16-year-old**, is conducting independent AI research, focusing on **CGGR** ([https://github.com/MinimaML/CGGR](https://github.com/MinimaML/CGGR)) and **SRDE** ([https://github.com/MinimaML/srde-mistral](https://github.com/MinimaML/srde-mistral)).
   - They are eager to connect with the Unsloth AI community to expand their knowledge.
- **Practitioner merges Neuro-Symbolic AI**: Quentin, originally from Belgium but now in Chicago, is a **Neuro-Symbolic AI practitioner**.
   - He aims to combine techniques to achieve optimal architectures, and practices **Tang Soo Do** with his family outside of work.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1458551169264914522)** (440 messages🔥🔥🔥): 

> `GPT-5-nano, ChatGPT Health, LFM2.5 base model, Turkish language tokenization, AMD vs Nvidia` 


- **GPT-5-nano Dataset Pre-Training**: A member shared their synthetically generated pre-training dataset for tiny LLMs using **GPT-5-nano**, which can be found on [Hugging Face](https://huggingface.co/datasets/MaxHastings/TinyLLMPretrainingCore).
- **OpenAI releases ChatGPT Health**: OpenAI released **ChatGPT Health**, with one member joking about what's next *Pulse (aka news), then Health, what’s your prediction for the next feature*.
- **LiquidAI releases LFM2.5 Base Model**: LiquidAI released the base model for **LFM2.5**, a new family of hybrid models designed for on-device deployment on [HuggingFace](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Base).
- **Is Turkish Tokenization More Efficient?**: A member suggested that the Turkish language, being highly structured, could lead to more efficient AI due to how words are built.
   - Another countered that the efficiency comes from the tokenizer design, referencing research that reasoning in non-English languages (including Turkish-like) saves tokens: [*Multilingual reasoning in non-English (incl. Turkish-like) saves 17-47% tokens vs English on TLP@4*](https://aclanthology.org/2025.findings-emnlp.845.pdf).
- **AMD vs Nvidia Faceoff at CES**: Members discussed **AMD** and **Intel's** presence at CES, with the consensus being that Nvidia still holds the edge in gaming due to advertising, DLSS, and partnerships, while AMD is making strides with memory advantages in AI compute.
   - Also, CUDA is an issue, ROCm just isn't as good: *The only way amd really has them beat is the vram quantity/$ but i don't think we are truly pushing 8gb yet enough for most people to actually see the "need" for anything above that*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1458715805478031371)** (22 messages🔥): 

> `Cerebras/DeepSeek-V3.2 Quantization Request, Ollama vs. llama.cpp, LMStudio Privacy Concerns, llama.cpp Setup Guides` 


- ****Quant Request** for Cerebras/DeepSeek-V3.2-REAP-345B**: A member requested a **Q5 quantization** of the **Cerebras/DeepSeek-V3.2-REAP-345B-A37B** model and was directed to the [Unsloth GitHub issues](https://github.com/unslothai) for such requests.
   - The team mentioned that they are currently *more selective about quants due to time and resource constraints* and rarely upload customized quants.
- ****Ollama Outdated?** Users Debate Alternatives**: A user experiencing an issue with **Ollama** was advised to consider alternatives like **llama.cpp** or **LMStudio** due to concerns about **Ollama** pushing cloud services.
   - The main issue cited was that *Ollama uses outdated llama.cpp*, leading to potentially inferior performance and reduced control compared to using **llama.cpp** directly.
- ****LMStudio's Lack of Open Source** Sparks Privacy Debate**: A user expressed concern about **LMStudio** being closed source, citing a preference for open-source tools for privacy reasons, while they admitted to enjoying LM Studio.
   - It was pointed out that *Ollama has been making questionable moves* regarding cloud services, contrasting with the user's initial impression of **Ollama** as a privacy-focused solution because it is associated with Meta.
- ****Newbie Navigates** llama.cpp Setup with Guidance**: A user requested resources for setting up **llama.cpp** quickly and received a link to the [official GitHub repository](https://github.com/ggml-org/llama.cpp) along with quickstart documentation and guides.
   - The user was also pointed to the [Unsloth documentation](https://unsloth.ai/docs/models/tutorials) for model-specific settings for **llama.cpp**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1458753236029935616)** (1 messages): 

> `Llama-3.3-8B-Instruct, rope_scaling, chat template` 


- **Llama-3.3-8B-Instruct w/fixes Released!**: A new model, [Llama-3.3-8B-Instruct-128K](https://huggingface.co/shb777/Llama-3.3-8B-Instruct-128K), has been released with several fixes, including **rope_scaling** and an **Unsloth chat template** in the tokenizer config.
   - The release also features an **updated generation config** and **enabled full context length**.
- **Meta's Llama-3.3-8B-Instruct Missing on HF**: The [Llama-3.3-8B-Instruct](https://llama.developer.meta.com/docs/models) model is listed on **Meta's developer site** but hasn't been pushed to **Hugging Face** yet.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1458588014501695693)** (79 messages🔥🔥): 

> `Tiny LLM pretraining, GRPO after SFT, RL hacking rewards, Llama 3 for on-prem generation, Synthetic data for training` 


- **Tiny LLM Pretraining takes off**: A member is pre-training a tiny LLM from scratch, around **10-50 million parameters**, to explore the capabilities of a small base model and shared a link to a [dataset of 2,700 general subjects](https://huggingface.co/datasets/MaxHastings/TinyLLMPretrainingCore).
   - Another member expressed frustration with fine-tuning pre-trained models, citing a desire for full control over the data learned, feeling like they were always *battling the existing weights*.
- **GRPO gets done after SFT**: A member mentioned that their larger model has mistakes in its reasoning traces, despite providing correct final answers, leading them to implement **GRPO (Generative Reward Policy Optimization) after SFT (Supervised Fine-Tuning)**.
   - Another member asked if they were just doing SFT to train the model.
- **RL hacks Rewards LOL**: A member humorously noted that RL (Reinforcement Learning) is *hacking the rewards*, describing it as fascinating but also problematic.
   - When prompted on whether it was a good thing, the member responded that it was a *bad way* and reading the docs.
- **Llama 3 still rocks On-Prem**: **Llama-3-70B** remains a top choice for on-premise generation due to its **VRAM-to-Reasoning ratio** for local data manufacturing on **P40 clusters**.
   - One member argued for the superiority of **gpt-oss-120b**, citing its speed and MoE architecture, while another emphasized Llama 3's proficiency in cybersecurity syntax and structured reasoning, especially in generating Chain of Thought logs without hallucinating fake Linux flags.
- **Open Source CyberSec Dataset goes live!**: Initially planning to sell a cybersecurity dataset, a member released a [580-row synthetic dataset](https://huggingface.co/datasets/blackboxanalytics/BlackBox-CyberSec-CoT-v1) (MIT licensed) generated via **Llama-3-70B**, focusing on CoT logic for SOC Incident Response, as a show of good faith.
   - The dataset is designed as a **Golden Set** for evaluating how well a model follows JSON schemas and reasoning steps, aiming to steer the model's logic significantly better than raw logs alone.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1458550822186123486)** (655 messages🔥🔥🔥): 

> `File sending in modules, Gemini Pro 3 on LMARENA, Voice cloning tools, LM Arena Battle mode issues, Movement Labs AI and Hawk Max` 


- ****File Sending Feature Sighted****: A user requested the ability to send files to modules, noting that large files get cut off when copied and pasted, especially on mobile.
- ****Gemini Pro 3 Gets Glitchy, Gains Ghosts****: One user reported errors with **Gemini Pro 3** after sending a picture, with issues persisting across different browsers.
   - Another user suggested that closing the **LM Arena** tab can cause outputs to stop working.
- ****Voice Cloning Vanguard Vanquishes Vocals****: A user highlights a voice cloning tool, [echo-tts-preview](https://huggingface.co/spaces/jordand/echo-tts-preview), as being highly effective and *indistinguishable with the right seeds*.
- ****Hawk Max Hype Hits Highs, Humiliates Horizons****: Users are discussing the new **Hawk Max** model from **Movement Labs**, claiming it outperforms **Claude Opus 4.5** in certain tasks and allows generating a functional **Minecraft clone** and **chess game** in one shot.
   - One user exclaimed **Movementlabs.ai** should be added to the arena.
- ****LM Arena Adding 'Edit Messages' on Radar****: Pineapple from the moderation team shared insight on the potential implementation of **edit messages** and a **stop button** to address model getting stuck. It is a challenging implementation because of the different models.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1458617139547340924)** (2 messages): 

> `Vision Arena Leaderboard, Text-to-Video Leaderboard, Image-to-Video Leaderboard, Hunyuan-Video-1.5, ERNIE-5.0-Preview-1220` 


- **ERNIE-5.0 Enters Vision Arena Top 10**: The [Vision Arena leaderboard](https://lmarena.ai/leaderboard/vision) has been updated, with `ERNIE-5.0-Preview-1220` now ranked **#8** with a score of **1226**.
   - **Baidu** is currently the only Chinese lab in the Top 10 on the Vision leaderboard, as detailed in the [Leaderboard Changelog](https://news.lmarena.ai/leaderboard-changelog/).
- **Hunyuan-Video-1.5 Hits Video Leaderboards**: `Hunyuan-Video-1.5` has been added to the leaderboards, ranking **#18** on the [Text-to-Video leaderboard](https://lmarena.ai/leaderboard/text-to-video) with a score of **1193**.
   - It also ranks **#20** on the [Image-to-Video leaderboard](https://lmarena.ai/leaderboard/image-to-video) with a score of **1202**; feedback can be shared in the designated channel.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1458550813759770746)** (267 messages🔥🔥): 

> `Model Recommendations, M5 Chip Upgrade, Llama's Relevancy, LORA setup help, GPU Compression` 


- **Ram Rules the Roost for Models**: Members discussed that **RAM** is the main limitation when running models, stating that a **30B model** likely won't run without at least **64GB of RAM**.
   - They recommended **Llama 3.2 3B** at good quality, with further discussion on quant levels and compression.
- **Llama dead, Qwen rules**: A member stated that **Llama** is a *dead end* and that **Meta** has basically given up on it, claiming that **Llama models haven't been relevant since 2024**.
   - They further stated that **Qwen** and **Liquid** have taken over the edge model side, sharing that their favorite edge model is **Qwen 3 4b**.
- **LM Studio needs LORA setup help**: A member requested help on how to use **tencent/HY-MT1.5-1.8B** as a **LORA** in **LM Studio** where the main model is **Qwen3 vl 8b**.
   - Another member simply stated it's *Not in LMS*.
- **Wikipedia getting Compressed!**: A member shared that they finally got their **GPU compressor** to work on the **Wikipedia dataset** after spending days trying.
   - Another member joked *you gonna train a foundation model on all the scraped data? if so how many years will this take on your rack*.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1458588903236960359)** (177 messages🔥🔥): 

> `LM Studio Laptop Performance, GPU Announcements CES, LLM Suggestions for Code, Dual CPU setups for LLMs, VS Code AI Assistants` 


- **Laptop Specs determine LM Studio Potential**: A user with an **Intel i9-13980HX**, **RTX 4060** (**8GB VRAM**), and **32GB DDR5** inquired about LM Studio performance, with members suggesting that models larger than **5GB** might be slow but still runnable.
   - The user was advised to consider the **Qwen 3 4B 2507** model for optimal performance, while another member said they could run **GPT-OSS 20B** with acceptable speed, around **20 t/s**.
- **Nvidia Skips RTX 50 Super Announcement at CES**: Rumors about the **RTX 50 Super** were quashed as [Nvidia will not announce any new GPUs at CES](https://www.tomshardware.com/pc-components/gpus/for-the-first-time-in-5-years-nvidia-will-not-announce-any-new-gpus-at-ces-company-quashes-rtx-50-super-rumors-as-ai-expected-to-take-center-stage), shifting focus to AI.
   - This marks the first time in five years that Nvidia will not announce new GPUs at CES, with AI expected to take center stage.
- **VRAM the bottleneck for LLMs**: A user with an **R9 7900x**, **96GB RAM**, and **RTX 3080** (**10GB VRAM**) sought LLM recommendations for code, with others noting that **VRAM is the only choice unless you got epyc or threadripper**.
   - Suggested models included **glm-4.5-air**, **Qwen-next**, and **GPT-OSS**, with speeds ranging from **4-10 t/s**, while another user discussed potentially getting **2x 3090** or **4090** to increase performance.
- **Dual CPU Setups: LLM No-Go Zone?**: A user contemplating a dual **Intel Platinum 8160** setup with **2x 3090** was cautioned against it for LLMs, with a member stating *I made the mistake of going dual CPU at first. Avoid for LLM's*.
   - Discussion pivoted to preferred LLMs, with **Qwen3-Next** being mentioned, alongside serving LM Studio across LAN and Tailscale with tools like [crushi](https://github.com/charmbracelet/crushi).
- **Cline Reigns Supreme in VS Code AI Assistant Arena**: Users discussed various VS Code AI assistants, with one member recommending **Cline**, emphasizing that *kilo is the only one that works right*, while others mentioned having issues with **Roo Code**.
   - Another member noted that AI code assistants are *automaticall breaking a task into todo steps and asking questions back to the user is nice*, while one more highlighted that **Cline** tends to go off the rails after about **50K context**.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1458555028636827843)** (239 messages🔥🔥): 

> `Cursor empty window bug, Chatbot dying, Commit rules not loading, Better Auth vs Neon Auth, Opus 4.5 cost-effectiveness` 


- **Cursor Windows Lose Agent Chat**: When opening an empty **Cursor window**, initiating a chat with an agent, and then opening a folder in the same window, it opens a new window, causing the user to lose the agent chat.
- **Agents Choking Mid-Command**: Users are reporting instances where **Cursor chats die** mid-command (specifically when running terminal commands like *npm run dev*), requiring a new chat to be created, and it may be related to [serialization errors](https://forum.cursor.com/t/serialization-error/124671/186).
- **Rules of Engagement: Commit Rules MIA**: A user reports that their **commit rules** are not being loaded automatically in agents, despite being shown in settings.
- **Auth Faceoff: Better Auth vs Neon Auth**: Users are debating the merits of **Better Auth** versus **Neon Auth**, with one user noting that Better Auth is too new and missing features, such as multi-tenant base URL setup.
- **Is Opus 4.5 worth the candle?**: Members are discussing the cost-effectiveness of **Opus 4.5**, with one user stating that they found it *worth it* even when considering that they used *15b tokens* in the past year.
   - Another user joked that they would need someone else's credit card to put it to the test, while another member recommended using **composer-1** for execution.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1458556083089051801)** (207 messages🔥🔥): 

> `blockchain blocking, Connection Refused Errors, skill.md vs mcp, AI vision model for game reward icons, Arc Raiders app item recognition` 


- **Blockchain Blockade Proposed**: A member suggested blocking the term *"blockchain"* due to increasing **scam** attempts using discordapp.com instead of discordfeet.
   - No further action was taken, but it highlights ongoing issues with fraudulent activity.
- **Connection Refused Catastrophe Continues**: Members reported recurring "connection refused" errors, with one user noting that almost **2%** of requests were failing due to Cloudflare issues and the connection staying open infinitely.
   - The user implemented an **18-second timeout** and retry strategy to mitigate the issue, but seeks further assistance from OpenRouter.
- **Skill.md Surpasses MCP for Docs**: A member praised **skill.md** for its superior approach to documentation compared to **MCP (JSON-RPC)**, emphasizing that *"skill.md is about writing good docs!"*.
   - They highlighted its potential for **dynamic tool retrieval** and integration with tools like Godot, making it a *"cooler skill to have"*.
- **Gemini Gains Ground as Go-To for Game Reward Icons**: Members discussed using a vision **AI model to analyze game completion screens** and evaluate rewards, noting that **Gemini** models (particularly **Gemini 3 Flash**) are promising, but struggle with small icons without text labels.
   - Suggestions included generating reference grids of icons and using them as context, with caching to reduce costs, or using a small VL model.
- **Gemini 2.5 Pro Glitches Briefly**: Members reported a downtime blip for **Gemini 2.5 Pro**, while **Gemini 2.5 Flash** and the **3.x series** were functioning normally, with [OpenRouter uptime page](https://openrouter.ai/google/gemini-2.5-pro/uptime) confirming the disruption.
   - One user said it was still down across multiple apps and accounts, whereas others said it was functional.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1458846699597336577)** (8 messages🔥): 

> `OpenRouter Show, Multimodal Embeddings, Computer Use Agents, Qwen3` 


- **Next OpenRouter Show Still Unknown**: A member inquired about the next **OpenRouter Show**, but no specific date was mentioned in the provided messages.
- **Multimodal Embeddings Spark Interest**: A member shared a link to [Alibaba's Qwen model](https://x.com/Alibaba_Qwen/status/2009264754917863924) highlighting its multimodal embedding capabilities, which sparked considerable interest.
   - Another member expressed skepticism despite the interest, while others noted the existence of similar models, with one stating *"Me when multimodal embedding models like this (and even supporting more modalities) have existed for a while now"*.
- **Agents Get Memory Boost?**: A member suggested that **multimodal embeddings** could provide an efficient way to give **computer use agents memory**.
   - Another member agreed that the approach could improve agents.
- **Qwen3 is built different**: Some chatbots remarked that **Qwen3 is built different**.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1459001252485660794)** (1 messages): 

> `OpenAI for Healthcare, Physician AI usage` 


- **AI in Healthcare Double Dips**: Physician use of **AI** nearly doubled in a year, per [OpenAI's announcement](https://openai.com/index/openai-for-healthcare/).
   - OpenAI launched **OpenAI for Healthcare**, a HIPAA-ready tool for healthcare organizations to deliver more consistent, high-quality care to patients.
- **OpenAI for Healthcare Rolls Out**: **OpenAI for Healthcare** is now live at AdventHealth, Baylor Scott & White, UCSF, Cedars-Sinai, HCA, Memorial Sloan Kettering, and many more.
   - The company touts [on their website](https://openai.com/index/openai-for-healthcare/) that it helps with *delivering more consistent, high-quality care to patients*.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1458557085049552917)** (172 messages🔥🔥): 

> `AI-assisted board game design, Agent identity and context persistence, Gemini 3 Pro vs Google Search AI, Sora availability in India, Anthropic's government strategy` 


- **AI Game Design on the Horizon**: Members discussed the possibility of using **AI** to create content for **board games**, **tabletop games**, and **card games**, including core mechanics and art.
   - One member shared [concept art](https://drinkoblog.weebly.com/) generated for a card game, including borders and rule text and pointed to [Hasbro abandoning functional design](https://discord.com/channels/974519864045756446/1204360881593520128/1427039507114496060) as the impetus to create an alternative.
- **Identity Crisis: AI Agent Context Handling**: A member sought advice on **handling agent identity** and **context persistence**, needing to persist predefined context for multiple agents in different roles.
   - Suggestions included using a tool to store memory always visible to the agent and employing **RAG** (Retrieval-Augmented Generation), though concerns were raised about the inefficiency of constantly including large text blocks.
- **Google Search AI vs. Gemini 3 Pro: The Showdown**: Members debated whether **Google's Search AI** or **Gemini 3 Pro** is better, with one user arguing for the superiority of the search model for certain tasks.
   - However, others argued that a search AI, optimized for finding and citing sources, cannot match the reasoning and synthesis capabilities of an LLM like Gemini, which is designed to maintain context and provide coherent outputs.
- **Sora Still Out of Reach for India**: A member inquired about downloading the **Sora AI** app in India, but was told that India is not currently on the list of supported countries.
   - A link to the [OpenAI Help Center](https://help.openai.com/en/articles/12461230-sora-app-and-sora-2-supported-countries) was provided for updates.
- **Anthropic's AI Safety Strategy: Military Procurement?**: One member suggested that **Anthropic's AI safety reports** and **white papers** read like a military procurement brochure, highlighting the model's controllability and targetability.
   - Another member agreed that **Anthropic** gives "Defense Contractor" vibes, speculating that their strategy is government use, pointing to a [$200 million contract with the U.S. Department of Defense](https://www.defense.gov/News/Releases/Release/Article/3594261/dod-announces-awards-for-prototype-artificial-intelligence-capabilities/).


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1458738632642007100)** (11 messages🔥): 

> `Server rules on language, Model Pro 5.2 speed issues, Missing chats after account merge, Custom GPT instructions and memory management` 


- **Server Rules Strictly Enforced**: A member expressed a desire to use *anything*, prompting clarification that while freedom is encouraged, server rules regarding **English-only** and **non-cryptic language** must be followed.
   - The clarification emphasized adherence to community guidelines while enjoying creative expression within defined boundaries.
- **Model Pro 5.2 Users Lament Slow Response**: A user inquired about the performance of **Model Pro 5.2**, particularly its **extended thinking mode**, which reportedly takes *forever* to respond (sometimes close to an hour).
   - The user sought insights into typical use cases and expected response times for this specific model version.
- **Account Merge Causes Missing Chats Catastrophe!**: A member reported missing chats after *merging* their personal Plus account into a Business account, with critical chats failing to transfer.
   - They sought advice on how to recover these lost conversations, which are absent from both the left tab and chat search functions.
- **GPT Instructions Subject to Merging and Memory Loss?**: A discussion arose around whether Custom GPTs can access user instructions, with confirmation that **Custom GPT instructions** are merged with **user custom instructions and memory**.
   - However, another user countered that their Custom GPT doesn't seem to have access to memory management and is *totally amnesic*, especially with integrations like *Monday by ChatGPT*.


  

---


### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1458790580787347567)** (1 messages): 

> `Transformers v5, Swift-Huggingface, AnyLanguageModel, OVHcloud Inference, DeepMath` 


- **Transformers v5 transmogrifies AI ecosystem**: **Transformers v5** simplifies model definitions by unifying the tokenizer backend, modularizing model definitions, focusing on **PyTorch**, and prioritizing quantization with [new serving/inference features](https://huggingface.co/blog/transformers-v5).
- **Swift-Huggingface swims into view**: The **Swift Client** for Hugging Face, [swift-huggingface](https://huggingface.co/blog/swift-huggingface) has been introduced.
- **AnyLanguageModel accesses local and remote models on Apple**: [AnyLanguageModel](https://huggingface.co/blog/anylanguagemodel) provides *one API* for local and remote LLMs on **Apple Platforms**.
- **FLUX.2 flows into Open Image Generation**: **BFL's** new open image generation model [FLUX.2](https://huggingface.co/blog/flux-2) has been welcomed.
- **Apriel-H1 is key to distilling reasoning models**: [Apriel-H1](https://huggingface.co/blog/ServiceNow-AI/apriel-h1) is the *surprising key* to distilling efficient reasoning models.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1458551701270433843)** (63 messages🔥🔥): 

> `Ollama vs LlamaCPP, OCR advice, GRPO and GPO reward functions, RMBG training, Synthetic data generation` 


- **Ollama Deemed Bloated, LlamaCPP to the Rescue?**: A member suggested skipping **Ollama** and using **LlamaCPP** stripped down to CUDA for being *faster and less bloated*.
   - Countering this, another member found **Ollama** *not that bloated*.
- **Handwritten OCR Headaches**: A member sought advice on **OCR** for handwritten assignments, noting some VLMs are compute-heavy.
   - Another member suggested **PaddleOCR-VL** if the job is solely OCR, but the original poster found **chandra-ocr** and **deepseek-ocr** to perform better for handwritten mathematical expressions.
- **GRPO Gibberish Generation**: A member doing **GRPO** observed their model outputting *gibberish* to lengthen the thinking process, despite reward functions not favoring longer lengths.
   - Another member recommended **Dr. Grpo**, arguing that *the model is incentivized to learn longer sequences with standard grpo*.
- **Synthetic Data Smackdown**: A member inquired if another user would be interested in a competitive synthetic data generation challenge judged by LLMs and an independent jury pitting a paid solution against Madlabs open source SDG pipeline.
   - The other user seemed willing to participate and provide a review, but mentioned that he had lost the doc but would type a review out.
- **Malware Scan Missing-in-Action**: A member noticed missing malware scan icons on the *Files and versions* tab for some models (e.g., whisper v3, qwen3-vl).
   - It's unclear why this is the case, but a possible solution might be to implement the features on your own.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1458569491301794045)** (44 messages🔥): 

> `DATASTRIKE BCI Timelapse Dataset, BSD conjecture dataset, Efficient AI Inference Deployments, Madlab's Synthetic Data Generator, Pacific Prime INL Tokenizer` 


- **DATASTRIKE BCI Dataset Strikes First**: A new **DATASTRIKE BCI Timelapse Dataset** was released, designed for training machine learning models for neural signal decoding without needing large-scale real hardware BCI datasets; a [YouTube short](https://www.youtube.com/shorts/UxV0e7J5gTs) and the [dataset on HuggingFace](https://huggingface.co/datasets/webxos/datastrike_BCI) were linked.
- **BSD Conjecture Dataset Solved?**: A dataset concerning the **Birch and Swinnerton-Dyer (BSD) Conjecture**, a Millennium Prize problem, was shared; it contains numerical data on elliptic curves and their associated L-functions to support machine learning research in arithmetic geometry, available at [HuggingFace Datasets](https://huggingface.co/datasets/webxos/bsd_conjecture_dataset).
- **Efficient AI Deployment Lessons Unfold**: A write-up detailing deployments based on inference optimization was published, covering the tooling behind it, in an article titled [Five Deployments in Lessons on Efficient AI Inference at Scale](https://medium.com/@paragekbote23/five-deployments-in-lessons-on-efficient-ai-inference-at-scale-6d99e9e64099).
- **Madlab's SDG Models Go Live**: **Madlab** released a new flagship synthetic data generator built for rule‑aligned, semantically coherent variation, including both **LFM2.5** and **LFM2** models adapted for high‑quality SDG workflows, thanking [LiquidAI for their outstanding work](https://huggingface.co/MadlabOSS/LFM2.5-1.2B-Instruct-SDG).
- **VeridisQuo Exposes Deepfakes**: A user released **VeridisQuo**, an open-source deepfake detector that uses GradCAM heatmaps to show exactly where the video was manipulated, utilizing spatial analysis, frequency analysis, and explainable AI visualization; the [GitHub repo](https://github.com/VeridisQuo-orga/VeridisQuo) and [demo on Hugging Face Spaces](https://huggingface.co/spaces/Gazeux33/veridisquo-deepfake-detection) were shared.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1458581378999386365)** (4 messages): 

> `Reinforcement Learning course scoring, Certificate Issue` 


- **Reinforcement Learning Course Scoring Troubleshoot**: A user reported submission issues in the **Reinforcement Learning course**, noting that they didn't receive their results and suspect a score below **30%**, preventing certificate attainment.
   - The user also linked to a potentially broken scoring link: [agents-course-unit4-scoring.hf.space](https://agents-course-unit4-scoring.hf.space/files/7bd855d8-463d-4ed5-93ca-5fe35145f733) which returned a **404 error**.
- **Certificate Issue: To Skip or Not To Skip**: A user inquired whether non-selected participants in a part of the course should skip it or take some action.
   - They asked *should we do something or we can skip it?*.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1458551560333426750)** (79 messages🔥🔥): 

> `CGGR benchmarks, MoE vs Dense Models at Nous Research, LLMs and AGI, Llama 3.3 8B` 


- **CGGR Benchmarks Boost Speed**: Initial benchmarks for [CGGR](https://github.com/MinimaML/CGGR) show a **1.40x** speedup compared to standard training, with forward passes at **127 ms** and backward passes at **93 ms**.
   - Further testing with **Triton** on an **H200** system is planned to assess speed and VRAM savings, which could allow for significantly increased batch sizes.
- **Nous Opts Dense Over MoE Models**: **Nous Research** is choosing dense models more than **MoE** because the infrastructure for training MoEs is not optimized, making it expensive.
   - Recent optimizations may be changing this, but current **state-of-the-art OS training infra** for **MoEs** yields only about **4% MFU**.
- **LLMs Alone Won't Cut It for AGI**: A member cited **Google Deepmind** and **Dr. Fei Fei Li** in stating that *LLM/Transformer alone just not gonna cut to reach AGI* and *more research + world models + biological neural network needed*.
   - Another member countered that the definition of **AGI** keeps getting pushed forward, and what we have currently would have been considered **AGI** three years ago and has huge utility.
- **Llama 3.3 8B 'Lobotomized' for Multilingual Tasks?**: The **Llama 3.3 8B Instruct** model is lobotomized for multilingual tasks according to some members.
   - A member speculates that **Meta** may have fudged the benchmark, but is moving away from **Llama**.
- **Qwen3-VL Embedding VRAM Quartered**: [Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding) and **Qwen3-VL-Reranker** are now available.
   - The result is *quarter the vram, double the tps*.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1458602749804413091)** (10 messages🔥): 

> `Diffusion LLMs, Diffusion model joke generation, Model Planning` 


- **Diffusion LLMs Touted for Speed and Untapped Potential**: A member expressed enthusiasm for diffusion LLMs, citing their potential for faster task completion and the possibility of generating better jokes without deliberate planning ([arxiv.org/abs/2511.08923](https://arxiv.org/abs/2511.08923)).
   - They suggest diffusion models might *know the punchline* before starting the setup, and noted that they are still *pretty under explored*, hinting at easy wins.
- **Diffusion Model Joke Generation**: One member theorized that **diffusion models** could generate better jokes by knowing the punchline beforehand, potentially doing so without explicit planning.
   - Another member suggested that **planning the output** could achieve similar results, but the first countered that diffusion might be faster with fewer tokens.
- **Model Planning debated as alternative**: A member posited that achieving similar outcomes to diffusion models, such as generating jokes, could be attained by **making the model plan its output**.
   - The original poster countered that diffusion might achieve the same *without planning* for faster speeds, using fewer tokens.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1458813439441895537)** (6 messages): 

> `HoloLM, CGGR, NousCoder` 


- ****HoloLM** gets tiny Shakespeare training**: The author shares [Experiment58-HoloLM](https://github.com/jackangel/Experiment58-HoloLM), a small LM trained in a few hours on a **5MB Shakespeare corpus**.
   - The author's goal is to create a model that can be trained on a laptop GPU and have a very large context.
- ****CGGR** Speeds Up Training**: A member suggests using [CGGR](https://github.com/MinimaML/CGGR) to speed up training.
   - It could be combined with a normal kvcache and **hologpt** for extremely recent conversational cache.
- ****NousCoder-14B** open-source coding model lands**: A member shared a link to [VentureBeat's article](https://venturebeat.com/technology/nous-researchs-nouscoder-14b-is-an-open-source-coding-model-landing-right-in) about **NousCoder-14B**.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1458602749804413091)** (10 messages🔥): 

> `Diffusion Models, LLMs, Planning Outputs, Better Joke Generation` 


- **Diffusion LLMs Spark Joy**: A member expressed enthusiasm for [diffusion LLMs](https://arxiv.org/abs/2511.08923), citing their potential for speed and unexplored possibilities.
   - They theorized that diffusion models could generate better jokes by *knowing the punchline* before starting, and mentioned diffusion being *interesting to watch*.
- **Diffusion Do It Faster?**: A member suggested that planning model outputs could achieve similar results to diffusion models, prompting a discussion about speed.
   - The original member argued that diffusion models might accomplish this *without planning*, potentially leading to **fewer tokens and faster processing**.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1458551096204333066)** (15 messages🔥): 

> `LLVM backend for Nvidia/AMD, CUDA Rust vs LLVM PTX, Nsight Systems SSH auth on ARM OSX, OpenACC, OpenMP, FortranSTD, and C++STD channels` 


- **LLVM Backends and GPU Code Generation Asked About**: A member initiated a discussion on **LLVM's backend** for code generation on **Nvidia, AMD**, and other accelerators, specifically how **NVPTX** and **AMDGPU** are used.
   - The member sought insights into how these backends select targets and function in GPU code generation.
- **CUDA Rust Sidesteps LLVM PTX Backend**: A member pointed out that **CUDA Rust**, though not officially Nvidia-supported, targets **NVVM** rather than **LLVM's PTX backend**, as documented in the [Rust-CUDA FAQ](https://rust-gpu.github.io/rust-cuda/faq.html#why-not-use-rustc-with-the-llvm-ptx-backend).
- **Nsight Systems ARM OSX Lacks Public Key SSH Auth**: A user reported that the **ARM OSX version of Nsight Systems** lacks the option for public key SSH authentication, a problem since **Runpod** doesn't support password-based SSH.
- **Request for New Computing Platform Channels**: A member suggested adding channels for **OpenACC, OpenMP, FortranSTD**, and **C++STD** within Computing Platforms.
   - A moderator responded that current volume may not justify it, suggesting the general channel for now, but opened to the idea of a broader **Fortran/C/C++** or **Directives** channel, clarifying that *directives refer to APIs like OpenACC and OpenMP that appear as comments in the code which can tell the compiler to offload to the GPU*.


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1458604891986722983)** (4 messages): 

> `Triton Shared Updates, New Triton Plugin Infrastructure, Plugin code location` 


- **Triton Shared Updates Broadcasted!**: A new YouTube video featuring updates from **Haishan** and **Nhat** on *triton-shared* is now available [here](https://youtu.be/JnFFwBB6Dhk).
   - The video also includes a presentation on the new **Triton Plugin infrastructure** by **Corbin, Puyan**, and **Simon**.
- **Plugin Code Location Unveiled**: A member inquired about the location of the plugins-related code, noting that the link provided during the presentation ([https://github.com/triton-lang/triton/tree/main/plugins/](https://github.com/triton-lang/triton/tree/main/plugins/)) was non-existent.
   - Another member provided the correct link ([https://github.com/triton-lang/triton/tree/main/lib/Plugins](https://github.com/triton-lang/triton/tree/main/lib/Plugins)), resolving the issue.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1458856839193166037)** (2 messages): 

> `Shared Memory Bank Conflicts, Matrix Multiplication Kernel Optimization, CUDA Optimization Techniques` 


- ****Bank Conflicts Baffle Block Matrix Multiply Buff****: A member is experiencing **4.5-way bank conflicts** during shared memory stores in a matrix multiplication kernel on a **T4 GPU** with **CUDA v12.2**, particularly when loading data from global memory into shared memory using `float4` stores.
   - They hypothesize that conflicts arise because each thread within a warp touches 4 sequential bank IDs, and are seeking methods to rotate the bank access pattern to reduce these conflicts, proposing a scheme where threads access banks in a rotated order (e.g., Thread 0 accesses Banks 0, 1, 2, 3; Thread 8 accesses Banks 1, 2, 3, 0).
- ****Shared Memory Scrutiny Spurs Solution Search****: A member seeks advice on how to modify shared memory access patterns in a **CUDA** kernel to avoid bank conflicts.
   - Another member inquired about the computation of `a_tile_row` and `a_tile_col`, and the type of `a_tile` to better understand the memory layout.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1458568384760385536)** (6 messages): 

> `CuteDSL flex attention implementation, SM100 vs SM90 backward support, fa4 work` 


- **CuteDSL flex attention integrates!**: A member thanked another member for integrating the **CuteDSL flex attention implementation**, and noted it speeds up regular flex attention with different mask mods.
   - They are seeing a **~30% throughput improvement** over base flex attention on **H100 fwd** which will save a lot of trees.
- **SM90 Backward Support Coming Soon!**: A member benchmarked different mask_mods and noticed that backward SM100 is supported but SM90 is not, referencing [the relevant flash-attention code](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/interface.py#L938).
   - The other member responded that they are *working on it* [relevant pull request](https://github.com/Dao-AILab/flash-attention/pull/2137).
- **FA4 work in progress!**: Some members messaged about FA4 work.
   - No other information was given.


  

---


### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1458572026884259974)** (3 messages): 

> `GPU Internships, Iris Project, Triton multi-GPU programming, Spring GPU Internships, Summer GPU Internships` 


- **GPU Internships opening for US Students**: An opening for interns interested in **GPU systems & performance** and **kernel development** was announced, to assist with the [Iris project](https://github.com/ROCm/iris/) framework.
   - The announcement specifies that the internship's ideal background includes experience with **Triton**, **multi-GPU programming**, **RMA/RDMA**, or **low-level GPU communication and kernel work**.
- **Inquiries Regarding Internship Timing**: A member inquired whether the advertised internship was for the **spring** or **summer**.
   - No further details were provided in the available context regarding the specific timing of the internship.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1458589561881100380)** (27 messages🔥): 

> `RTX 5050 as a 'tiny Blackwell' card, GPU Driver Development in C/Rust, GPU programming with Triton, Machine Learning Discords` 


- **Decoding RTX 5050 as 'Tiny Blackwell'**: A member inquired whether the **RTX 5050** would be a good *'tiny Blackwell'* card, with others confirming it has compute capability **12** and is CUDA 12.0.
   - They cautioned that **RTX Blackwell** (sm_120) differs from **datacenter Blackwell** (sm_10x), particularly concerning tensor cores, advising it may not suffice for verifying code that efficiently utilizes **B200** tensor cores.
- **Rustling up GPU Drivers: A Coding Quest**: One member, learning **C** and **Rust**, sought guidance on contributing to a GPU driver repo, specifically for **Nvidia** on **Linux**, like **Nouveau** and **NVK**.
   - Another member shared a [GPU Glossary link from Modal](https://modal.com/gpu-glossary) as a resource for learning basic to intermediate GPU concepts.
- **Triton Troubles: Seeking GPU Programming Pointers**: A member starting with **GPU programming in Triton** found examples hard to read and sought references for methods like *warmup*, which wasn't readily available.
   - They noted the official **Triton documentation lacked detail** and methods like `warmup` were only found within the source code.
- **Discord Discoveries: ML Hangouts**: A member asked for Discord communities similar to the current one, but focused on **machine learning**.
   - Two other members suggested **EleutherAI** and the **Yannic Kilcher** Discords, noting they are large and easily found via a quick search on Google.


  

---


### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/)** (1 messages): 

marksaroufim: It didn’t happen unfortunately. We’re gonna need to reschedule
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1458568084368654336)** (2 messages): 

> `Community Scale, Fast Edits` 


- **Editing for Speed in Large Communities**: Members acknowledged that they edit messages to be faster given the scale of the community.
- **Efficiency in Community Management**: It was noted that editing messages is a practical approach to managing communications efficiently in a large community.
   - This method helps in quickly updating information without creating multiple new messages, streamlining the flow of conversation.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1458623408316088454)** (1 messages): 

> `TK v2 Release, PR Patch, Contribution Opportunity` 


- **TK v2 fixes PR patch issue?**: A member mentioned they had a **PR patch** ready but got swamped with other things around the **ICLR deadline**.
   - They asked about the ETA on **TK v2**, wondering if it fixes the issue, before contributing.
- **Desire to Contribute Amidst Deadlines**: The same member expressed a desire to contribute, but workload around the **ICLR deadline** prevented them from doing so.
   - They seek guidance on whether their **PR** would still be valuable considering the progress of **TK v2**.


  

---


### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1458876754562121778)** (2 messages): 

> `CUDA kernel from Rust, Installation instructions` 


- **CUDA Kernel Launches from Rust**: A member worked on **chapters 1.1, 1.2 and 1..4** and provided some **installation instructions** in the readme.
   - Another member suggested playing around with the code to see how to launch a **CUDA kernel from Rust**.
- **Rust Installation Instructions Provided**: Installation instructions were added to the readme file, to guide new users.
   - These instructions coincide with examples that launch CUDA kernels.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1458563964102508696)** (5 messages): 

> `Blackwell blog, PTX and Python/C++ runner` 


- **Blogpost Hunt for Blackwell Deets Begins**: Members are seeking follow-up blog posts on **Blackwell**, similar to [this blog](https://www.aleksagordic.com/blog/matmul) post.
   - A member suggested checking out [this blog post](https://www.kapilsharma.dev/posts/learn-cutlass-the-hard-way/) as well.
- **PTX and Python/C++ Runner Mechanism Questioned**: A member inquired about a mechanism to support submitting **PTX** and a **Python/C++ runner**.
   - They noted the absence of path/extension sanitization in `sumbit.rs`, implying no obvious restrictions.


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1458581134215479376)** (4 messages): 

> `Helion on ROCm, AMD support for Helion, Performance speedup on GEMMs` 


- **AMD Engineer to Enable Helion on ROCm**: An AMD engineer, Umesh, will be working on enabling **Helion** on **ROCm** and identifying issues in skipped unit tests and examples in the **Helion repository**.
   - He asked the group to share any immediate issues that need fixing.
- **Helion Seeks to Support MI400 Series**: A member expressed interest in building support for the **MI400 series**.
   - The team encouraged Umesh to ask any questions.
- **ROCm Focuses on Performance Speedup**: The AMD engineer is currently looking into all the skipped tests and some of the broken examples on **ROCm**.
   - They will focus on **performance speedup on GEMMs** in parallel.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1458562618582241388)** (31 messages🔥): 

> `GPUMODE Slow Runners, DSLCudaRuntimeError debugging, Test vs Benchmark vs Leaderboard Modes, Discord Bot for GPUMODE, AI-Generated Submissions in Top 10` 


- **GPUMODE plagued by Slow Runners**: A user reported seeing **slow runners** and experiencing timeouts in GPUMODE, providing an [example ID 297869](https://cdn.discordapp.com/attachments/1434709259500650628/1458664934706511922/Screenshot_2026-01-07_at_7.30.00_PM.png?ex=69611fd5&is=695fce55&hm=469fea089e64c3e7f89bfccbb4ec99c67c790f09c4a8f399796cc7627394cfd5&).
   - The user experienced a `DSLCudaRuntimeError`, and although benchmark runs went fine, the leaderboard submissions seemed to be running in `test` mode, causing confusion and timeout issues, and raised the question: *"what's the difference b/w test, benchmark, leaderboard?"*
- **Decoding GPUMODE Submission Modes**: The difference between `test`, `benchmark`, and `leaderboard` modes was clarified: **test** checks correctness on **10** test shapes, **benchmark** runs on **3-4** benchmark shapes, and **leaderboard** submits the actual geom score with a **secret bench init**.
   - It was suggested that *"jus use discord bot its much easier"*, and a [link to the popcorn-cli tutorial](https://github.com/gpu-mode/popcorn-cli/tree/main/docs/AMD_workshop) was provided.
- **AI Bots Crushing the GPU Competition**: A user inquired whether anyone in the top 10 was using purely AI-generated submissions.
   - One participant confirmed they are at **#4** on Problem 2 with a **100% AI-generated** submission using LLM agents without any hand-written GPU operators, and they are trying to use open source models only.
- **Profiling Timeout Troubleshoot**: One user mentioned that they *"wondering if anyone experienced timeouts (at 10 mins on the github action) only when using /profile via discord?"
   - Another user said that *"profiling might take quite a while, send me your file and I can take a look tmorrow."


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1458555226121306237)** (72 messages🔥🔥): 

> `OpenAI Developer vs Researcher, LMArena views, Mercor AI hiring, CC:Train Dataset release, Cursor AI Dynamic Context` 


- **The Guardian Dubs OpenAI as Developer**: An article in [The Guardian](https://www.theguardian.com/technology/2026/jan/07/ai-anthropic-funding-valuation) refers to **OpenAI** as a *developer* rather than a *researcher*, prompting discussion on whether publishing research is a defining trait of research organizations.
   - One member commented, *Doesn't being a researcher mean you publish your research?*
- **LLMarena, A Plague on the AI World?**: A [blog post](https://surgehq.ai/blog/lmarena-is-a-plague-on-ai) criticizing **LMArena** as harmful to AI progress sparks debate on its relevance, despite recent news of their fundraising.
   - Some point out that model companies still appear to care about it, while others argue that it's outdated and not used for actual decision-making, despite its prevalence in flexing and discussion.
- **Automated AI Hiring Intrusive, Candidate Bails**: **Sasha Kaletsky** detailed on [X](https://xcancel.com/sashakaletsky/status/2008904526720286970?s=46) a streamlined, AI-driven recruitment experience at **Mercor** for a 'Financial Expert' role.
   - The process involved impressive **AI interviews** and **automated matching**, but required installing intrusive monitoring software (**Insightful**) to record activity for **RL model training**, which led to the candidate's withdrawal.
- **Autonomous Gets Seeded for AI Financial Advice**: **Dillon Erb** announced the launch of [Autonomous](https://xcancel.com/dlnrb/status/2009008876834922949?s=46), an **AI-powered financial advisor** offering services at 0% advisory fees.
   - The company secured **$15M** in funding led by **Garry Tan** at **Y Combinator** and is actively hiring for roles in **New York City** and **San Francisco**.
- **Protege AI Harvests $30M to Solve Data Bottleneck**: **Protege AI** announced a **$30M** funding round led by **a16z** to expand its data infrastructure for AI development, per their [announcement](https://xcancel.com/withprotegeai/status/2009274652183363639?s=46).
   - A member asked if someone is tracking the data companies as there seems to be a new one every week.


  

---


### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1458856496606478523)** (1 messages): 

> `Attention tracking, Oura for desktop, Personal AI model` 


- **Intension: Oura Ring for Desktop**: [Intension](https://www.intension.us/who-we-are), founded by **Conor Sanchez-O'Shea** and **Gabriel Duemichen**, is developing an "Oura for your attention on the desktop" to track user focus, intention, and capacity.
   - The software visualizes attention tendencies, similar to Oura's health metrics, and removes distracting elements from the desktop to limit interruptions, potentially using **AI** to proactively respond on the user's behalf.
- **Intension: Reclaiming attention spans**: According to their [Youtube video](https://www.youtube.com/watch?v=WmJNRxU1lpg), Intension is *Reclaiming humanity’s most important resource—attention*.
   - They intend to help build your own personal model that is trained for you, not by you.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1458560752842576147)** (13 messages🔥): 

> `Tolan AI User Milestone, Multi-Angle Camera Control LoRA for Qwen-Image-Edit-2511, iOS App Submissions, AI Influencer Marketing` 


- **Tolan AI Hits 200K Users with OpenAI Assist!**: Paula from **Tolan AI** announced their voice-first AI companion reached **200,000 monthly users**, developed with close collaboration with **OpenAI** ([source](https://x.com/paularambles/status/2008964509810278413)).
- **Qwen-Image-Edit-2511 gets Multi-Angle LoRA**: **Fal** released a more powerful, open-source version of the **multi-angle camera control LoRA** for **Qwen-Image-Edit-2511** ([source](https://x.com/fal/status/2008954582018248755)).
   - The tool allows manipulation of camera perspective including *front, back, side, low/high angles, and various shot distances*, beneficial for **first scene last scene video generation**.
- **AI Influencers Plug iOS Apps**: Philo Hermans shared that four of six **iOS app submissions** have been approved, and he created **six realistic AI influencers** for distribution and marketing ([source](https://x.com/philo01/status/2008880081456996510)).


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1458601816932487330)** (67 messages🔥🔥): 

> `Kimi K2 vs Qwen, Kimi K2 excessive searching, Kimi K2 slides generation` 


- **Kimi K2 Excels in Creative Writing**: One member noted that **Kimi K2** performs significantly better at **creative writing** and overall conversations compared to other Chinese models, citing a top score on [EQ bench](https://eqbench.com/).
- **Kimi's 'Thinking' Mode Debated**: Members debated the usefulness of **Kimi K2's 'thinking' version**, with one user finding it on par with **GPT-5.2**, while another found it *dumb as hell* for their daily tasks.
- **Kimi K2 Searches Way Too Much**: One member reported that **Kimi K2** searches *wayyyyy too much* even for simple tasks like *1 plus 1*, and that the searches are very dumb in English.
- **Kimi K2's Slides Generation Issues**: A member reported issues with **Kimi K2's slides generation**, including being prompted to upgrade for a subscriber feature, but the issue resolved itself.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1458562599527649595)** (27 messages🔥): 

> `Mojo training library, Struct design documentation, Mojo data formats, Building a dataset with Mojo, NixOS setup/configuring instructions` 


- **Mojo's MAX Power Requires DIY Backprop**: A member warned that Mojo doesn't have a training library yet and you'll need to use **MAX** to build one, requiring you to write **backprop** yourself.
   - Another member suggested using **sqlite** for data storage, given Mojo's current limitations with data formats.
- **Outdated Mojo Docs Confuse Newbies**: A new programmer was reading outdated Mojo docs at [/github.com/BunsDev/mojo-lang/](https://github.com/BunsDev/mojo-lang/), which is **2 years out of date**.
   - A helpful member pointed them to the [main repo](https://github.com/modular/modular) and [official docs](https://docs.modular.com/mojo/manual/).
- **New Programmers Face Mojo Growing Pains**: Experienced members advised new programmers to learn a different language first, such as **C** or **Python**, because Mojo is still in development and **breaks a lot of things very often**.
   - They added that *all of the docs currently assume you know some combination of Python + C++ or Rust*.
- **Numpy arrays can bring data into Mojo**: A member suggested that If you can load your data to a **numpy array** using python then u can bring those data into mojo and operate on them.
   - They were unsure if this works for OP's use case.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1458844860130791560)** (6 messages): 

> `SVD Implementation in Mojo, Lanczos/Krylov Algorithm, Mojo Roadmap, Tensor Network Library` 


- **SVD Implementation MIA in Mojo?**: A member was seeking an implementation of **Singular Value Decomposition (SVD)** using the **Lanczos/Krylov algorithm** for **Mojo**, but couldn't find an official or community version.
   - They checked the [Mojo roadmap](https://docs.modular.com/mojo/roadmap/) but it's unlikely to include domain-specific libraries like **SVD**.
- **LAPACK to the Rescue for Tensor Networks**: A member is building a **Tensor Network library** in **Mojo** and, due to time constraints, opted to use **LAPACK** via the **C ABI** for **SVD** rather than implementing it themselves.
   - Another member agreed that an **SVD** implementation would be valuable and suggested contributing it to **MAX** (Modular's standard library) if they were to write one.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1458645381091954738)** (6 messages): 

> `TEI vs max, embeddings performance, custom architecture implementation, BERT architecture` 


- **TEI Blazes Past max in Embeddings Speed?**: A member found that switching from [TEI](https://github.com/huggingface/text-embeddings-inference) to max resulted in significantly slower embeddings generation, achieving **727.1 embeddings/sec** with **28375.1 ms P95 latency**, compared to TEI's **8000 embeddings/sec**.
   - The member implemented `sentence-transformers/all-MiniLM-L6-v2` as a custom architecture and suspected max might be more optimized for LLM inference than embeddings.
- **Profiling Pointers Prompted for Performance Problems**: A member inquired about the hardware/platform used for benchmarking and suggested potential performance bottlenecks could stem from model architecture or untuned kernels.
   - They also expressed interest in the custom architecture being contributed back to MAX for review and inclusion.
- **MiniLM Model Architecture Materializes**: A member shared they are testing on an **Nvidia RTX 2000 Ada GPU** and provided a link to their fork with the feature branch implementing the `all-MiniLM-L6-v2` model architecture: [RWayne93/modular](https://github.com/RWayne93/modular/tree/feat/all-MiniLM-L6-v2-model-architecture/max/python/max/pipelines/architectures/minilm).
   - They held off on opening a PR, pending review for potential issues.
- **BERT Blueprint Beckons Build**: A member noted the absence of a **BERT architecture** in MAX and encouraged the user to submit a PR for review.
   - They also mentioned the need to involve profiling experts to diagnose performance issues effectively.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1458562939501150461)** (30 messages🔥): 

> `Manus Open Source, Email Issues, AI Engineer introduction, AI startup credits, Manus website` 


- **Open-Sourcing Manus: An Idea Floated**: A member suggested that old/deprecated versions of **Manus** could be open-sourced to help the public understand how **Manus** works and allow for community-driven feature additions.
   - This could also cater to enterprise users wanting local usage without cloud access, offering options for local usage.
- **Reporting Email Absence**: A member reported not receiving an email from the team.
   - The context suggests possible issues with notifications or updates from the platform.
- **AI Engineer Specializing in LLM Integration**: One AI engineer specializing in workflow automation, LLM integration, RAG, AI detection, image and voice AI, shared about their experience building automated pipelines and task orchestration systems using **DSPy**, **OpenAI APIs**, and custom agents.
   - They highlighted a support automation system connecting **Slack**, **Notion**, and internal APIs to **LLM**, reducing response times by **60%**.
- **Seeking Manus Startup Credit Insights**: A member inquired about the **Manus Startup Credit** application process and its success rate, seeking insights from others who might have gone through it.
   - No responses were given.
- **Manus website single work through different conversation**: A member asked if there was a way to work on a single website created by **Manus** through multiple different separate conversations.
   - No responses were given.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1458626703671689419)** (5 messages): 

> `Low cost model training options, Community spotlight talks, Diffusion-based world models` 


- **Cost-Effective Model Training Explored**: Members discussed low-cost options for training a **100 million parameter model** with a **100GB dataset**, suggesting platforms like **VastAI** and **Runpod**.
   - It was suggested that a **1-8x consumer GPU setup (4090, 5090)** could suffice due to the setup not being comms bound, offering significant savings over server GPUs.
- **Community Spotlight Talk Series Revival**: The "Community Spotlight" talk series is returning, highlighting research by community members, with talks scheduled on specific dates.
   - The first talk features a member discussing running **diffusion-based world models** in real time on consumer hardware, demonstrated with an **RTX 6000** ([video](https://cdn.discordapp.com/attachments/729741769738158194/1458922361129664575/2026-01-06_22-30-45.mp4?ex=696166d4&is=69601554&hm=6d4445d8ccb0f0a6262d3e5450a39bcb5333ef8d1cf63127443f61d7b9593158&)).
- **Common Crawl LangID Collaboration**: A speaker from **Common Crawl** will discuss their work on **LangID** at scale and the challenges involved.
   - This talk is part of the "Community Spotlight" series, emphasizing collaborative research efforts.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1458616706997157919)** (18 messages🔥): 

> `RL subfields, Base model training, Anthropic fellowship, Hard tokens` 


- **Training on base models yields better diagnosis**: A member recommended working on simpler subfields than **RL**, such as **base model training**, because it makes mistakes easier to diagnose.
- **Applying to Anthropic fellowship might be good for character**: A member is applying to the **Anthropic fellowship program** this weekend and asks for tips.
- **Hard tokens backward pass computation**: A member mentioned that *it doesn't compute most of the backward pass, so it does two forward, then discards the gradients of the easy tokens and only computes the gradients of the hard tokens*, which saves VRAM and compute.
   - Another member noted that this approach might only be more efficient when **hard tokens** are less than 2/3 of the input tokens, questioning its benefit at the beginning of training.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1459007475696271476)** (1 messages): 

> `Dead Salmon Artifacts, AI Interpretability, Randomly Initialized Neural Networks` 


- **Dead Salmon Artifacts Plague AI Interpretability**: A recent [paper](https://arxiv.org/abs/2512.18792) highlights the prevalence of **'dead salmon' artifacts** in AI interpretability methods, questioning the validity of explanations derived from randomly initialized neural networks.
   - The report indicates that techniques like **feature attribution, probing, sparse auto-encoding, and causal analyses** can generate seemingly plausible explanations even for meaningless network states.
- **Plausible Explanations from Random Networks?**: The [study](https://arxiv.org/abs/2512.18792) suggests that current interpretability tools may produce **misleadingly coherent explanations** even when applied to randomly initialized networks.
   - This raises concerns about the reliability of these tools in understanding genuinely meaningful representations learned by trained AI models.


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/)** (1 messages): 

aeros93: https://x.com/alibaba_qwen/status/2009264754917863924?s=46
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/)** (1 messages): 

tastybucketofrice: Currently it's just full-parameter finetuning, yep
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1458767166236131450)** (6 messages): 

> `RL Finetuning of Vision Models, Mic Testing, YC application, Symphoria AI's Maduro Investigation, Research Collaboration` 


- **RL Finetuning Seeks Vision**: A member is looking for works relating to **RL-based finetuning** of **vision models** to improve their **spatial reasoning**.
   - No specific papers or links were mentioned in the discussion.
- **Mic Check, Please!**: A member requested assistance with **mic testing** for an upcoming meeting with people living abroad, offering to schedule a **Zoom** meeting for helpers.
   - The member expressed concern about potential communication issues and sought quick testing.
- **YC Application: Assemble?**: A member inquired about potential collaborators for applying to **Y Combinator (YC)**.
   - No further details or responses were provided in the context.
- **Symphoria AI Investigates Maduro**: A member shared a link to **Symphoria AI's** investigation into **Maduro**: [https://symphoria-ai.web.app/share/alish-sult/maduro](https://symphoria-ai.web.app/share/alish-sult/maduro).
   - The post was shared without additional context.
- **Research Paper Collaboration**: A member offered collaboration on research papers, sharing their background and work: [https://linktr.ee/haseebb](https://linktr.ee/haseebb).
   - The user expressed interest in contributing to research efforts.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1458553615693250814)** (17 messages🔥): 

> `Lobbying War, ChatGPT Health, AI Pioneer Credit, AI Fatalities, Pause Advocate` 


- **Lobbyists Go To War**: A member mentioned a *lobbying war* between **Huawei** and **US natsec hawks**, working together against **Nvidia** and **China cloud**.
- **ChatGPT Enters Healthcare Arena**: **OpenAI** introduced [ChatGPT Health](https://openai.com/index/introducing-chatgpt-health/), positioned as a supplementary tool to verify medical information with references to real documents, and has the potential to catch ailments early.
   - Concerns raised include user privacy, **ChatGPT** becoming an *everything app monopoly*, and potential misuse by individuals replacing doctors, though some find it superior to existing health resources.
- **AI Pioneers Lack Credit**: Concerns were raised that awardees (Drs. **Bengio**, **LeCun**, **Hinton**) repeatedly republished important **AI techniques** without crediting the original creators, supported by reports [NOB][DLP][CN25] and numerous references.
   - The reports allege that they *didn't invent any of the foundational algorithms of modern AI*, with references to a technical report titled *A Nobel Prize for Plagiarism*.
- **Grok's Kill Count**: Members speculated about **Grok** bragging about its kill count, drawing parallels to potential fatalities associated with **AI** and its integration into healthcare.
   - A link to a [Wikipedia page on deaths linked to chatbots](https://en.wikipedia.org/wiki/Deaths_linked_to_chatbots) was shared, leading to dark humor about chatbot-related fatalities.
- **Pause Advocate Pauses**: A member shared a [YouTube video](https://youtu.be/-qWFq2aF8ZU) of a *pause advocate*, noting that he hasn't been posting much content for around a year.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1458879088004432168)** (12 messages🔥): 

> `Conversation History in DSPy, Custom Adapters for DSPy, Rewriting History in DSPy` 


- ****DSPy's Conversation History**: Exposed!**: A member questioned the inclusion of **history in the system prompt** within the [DSPy conversation history tutorial](https://dspy.ai/tutorials/conversation_history).
   - Another member clarified that *it's just how the adapter represents history* and that **custom adapters** can be written to change this without affecting optimizers.
- ****Adapters**: Your DSPy Swiss Army Knife**: Members confirmed that **custom adapters** can definitely be written to modify how history is handled in DSPy.
   - One member noted it's *misleading for how it's shown to the models*, but rewriting history and multi-turn conversations are being overhauled, with updates expected later this month.
- ****DSPy Rewrites History**: A Cause for Celebration!**: The team is working on rewriting how **multi-turn conversations** are handled, with changes expected later this month.
   - One member humorously commented that this is *the one time we'll all celebrate rewriting history as a good thing* and interpreted it as **RLM PR** being imminent.


  

---


### **DSPy ▷ #[colbert](https://discord.com/channels/1161519468141355160/1250300504462856265/1458851668408930521)** (1 messages): 

> `ColBERTv2, KeyError, dspy v3.1.0, ChainOfThought` 


- **ColBERTv2 throws KeyError topk**: A member reported a **KeyError: 'topk'** when running a code snippet from the docs using **dspy v3.1.0** and **ColBERTv2**.
- **ChainOfThought Module**: The code snippet uses **dspy.ChainOfThought** to retrieve a response from a question and provided context.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1458601876227625080)** (5 messages): 

> `Tinygrad scheduler bounty, Speed Bounties for Tinygrad` 


- **Tinygrad Scheduler Bounty up for grabs?**: A member inquired about the status of the "Replace scheduler with linearizer, preserving GPU speed" bounty, noting a potentially ready [PR](https://github.com/tinygrad/tinygrad/pull/13780) that is currently unclaimed.
   - They suggested submitting a working PR to claim the bounty, even if it meant sharing the reward with the original claimant as *the goal is to get work done and not let people hold work hostage*.
- **Guidance Sought for Tinygrad Speed Bounties**: A member requested guides to start working on **Tinygrad speed bounties**, expressing the belief that there should be a way to request access to a Tinygrad instance for running tests.
   - It was mentioned that *there should be a way to submit a request to access an instance of tinygrad to run the tests*.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1458632240815669259)** (2 messages): 

> `VFIO=1 without IOMMU behavior, tinygrad error with AMD Radeon RX 9070XT` 


- **Original VFIO=1 behavior sought**: A member asked about the original **VFIO=1** (without **IOMMU**) behavior in tinygrad.
   - The member observed an error with tinygrad on a Linux laptop with an **AMD Radeon RX 9070XT** when **VFIO=1** is set, and provided a [full error log](https://cdn.discordapp.com/attachments/1070745817025106080/1458632361758425098/tinygrad_vfio_no_iommu.log?ex=6961aa3f&is=696058bf&hm=03f6e0c3af31072eccac359044bad6439cf0c8b9f1665e3a9ae7bfc0b6130c73).
- **Benchmarking Works Without VFIO**: The member confirmed that `examples.benchmark_onnx` runs correctly without setting **VFIO=1**.


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1458837264435122197)** (5 messages): 

> `Staging mutating actions, SEP eligibility, SDK Implementation Details, W3C WebMCP and MCP Collaboration` 


- **Consider Staging Mutating Actions in MCP**: A member suggested MCP would benefit from a standardized way to “**stage” mutating actions** via tool calls before the mutating action actually gets written.
   - They have an idea written out with lots of examples and asked if it was [SEP-able](https://sep.dev).
- **SEP Scope Clarification**: Another member noted that this idea may not be an SEP candidate but more like an **SDK implementation detail** that could be documented and potentially followed by other SDKs.
   - *SEPs are about enhancing the protocol, which is governs communication.*
- **WebMCP and MCP Collaboration**: A member bumped a thread to continue the conversation on ways how **W3C WebMCP and MCP can collaborate**.
   - No further details were provided in the current message history.
