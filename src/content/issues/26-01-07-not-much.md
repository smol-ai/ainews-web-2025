---
id: MjAyNi0w
title: not much happened today
date: '2026-01-07T05:44:39.731046Z'
description: >-
  **AI News for 1/6/2026-1/7/2026** highlights a quiet day with key updates on
  **LangChain DeepAgents** introducing **Ralph Mode** for persistent agent
  loops, **Cursor** improving context management by reducing token usage by
  **46.9%**, and operational safety measures for coding agents with allow/deny
  lists. **MCP** integration is expanding across assistants and robotics, with
  Hugging Face embedding assistants via **HuggingChat + HF MCP server**. The
  **DeepSeek-R1** paper has been expanded to **86 pages**, emphasizing
  trajectory exploration and RL shaping behavior. **NousCoder-14B** shows a
  **+7% improvement on LiveCodeBench** after **4 days** of RL training,
  demonstrating advances in RL for coding with small open models. Top tweets
  also mention a viral "96GB RAM laptop", **ChatGPT Health** launch by
  **OpenAI**, and **Karpathy**'s nanochat scaling-law miniseries.
companies:
  - langchain
  - cursor
  - huggingface
  - openai
  - weights-biases
models:
  - nouscoder-14b
  - deepseek-r1
topics:
  - agent-frameworks
  - context-management
  - reinforcement-learning
  - operational-safety
  - model-transparency
  - trajectory-exploration
  - token-optimization
  - coding-agents
  - integration-platforms
people:
  - karpathy
  - _philschmid
  - omarsar0
---


**a quiet day**

> AI News for 1/6/2026-1/7/2026. We checked 12 subreddits, [**544** Twitters](https://twitter.com/i/lists/1585430245762441216) and **24** Discords (**204** channels, and **4658** messages) for you. Estimated reading time saved (at 200wpm): **421 minutes**. **Our new website** is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on [@smol_ai](https://x.com/Smol_AI)!

a quiet day.


---

# AI Twitter Recap

**Top tweets (by engagement)**

- **Hardware/compute & developer culture**: “96GB RAM laptop” going mega-viral ([@vikhyatk](https://twitter.com/vikhyatk/status/2008922250112819381)); “ChatGPT Health” launch ([OpenAI](https://twitter.com/OpenAI/status/2008987566796640575)); Karpathy’s **nanochat scaling-law miniseries** post ([@karpathy](https://twitter.com/karpathy/status/2009037707918626874)); xAI strategy/culture and fundraising posts ([@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2008765567922999573), [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2008774688382599520)).

---

**Agents & Developer Tooling: “agent harnesses”, DeepAgents, Cursor context, MCP everywhere**

- **LangChain DeepAgents + “Ralph Mode” (infinite loop agents with filesystem memory)**: Multiple posts converged on a pattern: stop “stuffing everything into the prompt” and instead run a *loop* where the agent refreshes context each iteration and persists state to disk. LangChain shipped **Ralph Mode** on top of **DeepAgents** ([LangChain OSS](https://twitter.com/langchain_oss/status/2008942888810631518)), echoed as a usable “run forever, Ctrl+C when satisfied” agent pattern. Independent commentary frames this as the “agent harness era” where people will remix lightweight orchestrators rather than build full IDEs ([omarsar0](https://twitter.com/omarsar0/status/2009061265864262111)). Related note: DeepAgents is positioned as “Claude Agents SDK-like, but model-agnostic” ([mstockton](https://twitter.com/mstockton/status/2008742557388599384)).
- **Cursor’s context management pivot**: Cursor reports rebuilding their agent’s context system to *dynamically discover* relevant context via files/tools/history instead of prompt stuffing, cutting token usage by **46.9%** ([mntruell](https://twitter.com/mntruell/status/2008793943472062807)). This is consistent with “filesystem as memory” and long-horizon coding agent trends, plus a vision of Cursor as a **desktop agent dashboard**, not just an IDE ([mntruell](https://twitter.com/mntruell/status/2008993971826249986)). Additional claim: writing transcripts to disk enables “millions of tokens long” conversations ([amanrsanger](https://twitter.com/amanrsanger/status/2008985132523495847)).
- **Operational safety for coding agents (allow/deny lists)**: As “YOLO mode” becomes common, the ecosystem is rediscovering that tool execution approval is the bottleneck and risk surface. A concrete allow/deny command list for agent shells (deny `git push`, `git reset`, publish commands, etc.) is shared by [@_philschmid](https://twitter.com/_philschmid/status/2008975389415354491).
- **MCP as the integration substrate**: MCP shows up across “chat with papers” experiences (Hugging Face Papers assistant) and robotics/agents; e.g., Claude Code ↔ Reachy Mini experiments ([Trtd6Trtd](https://twitter.com/Trtd6Trtd/status/2008933816073846810)). Hugging Face is embedding assistants into paper pages via **HuggingChat + HF MCP server** ([AdinaYakup](https://twitter.com/AdinaYakup/status/2008863050355675152), [@_akhaliq](https://twitter.com/_akhaliq/status/2008915667760635986)).
- **Browser agents “actually work” anecdotes**: A concrete end-to-end automation claim—Claude Code processing an Amazon return and reordering a size autonomously from a 2-sentence task—signals growing confidence in browser tool reliability ([corbtt](https://twitter.com/corbtt/status/2009003003630735616)).

---

**Model releases & eval ecosystem: open-weight velocity, RL-for-coding, vision/video, and skepticism about leaderboards**

- **DeepSeek-R1 paper expansion (22 → 86 pages)**: The updated DeepSeek-R1 report is framed as a major transparency upgrade, adding judge prompts, synthetic data prompts, harness details, analysis, and distillation sections ([机器之心](https://twitter.com/jiqizhixin/status/2008805570145644849); also [andrew_n_carr](https://twitter.com/andrew_n_carr/status/2008953964566597771)). One technical interpretation: gains are attributed less to “better data” and more to *trajectory exploration/verification* and verifiable rewards, with RL shaping behavior rather than injecting knowledge ([gm8xx8](https://twitter.com/gm8xx8/status/2009000108327670116)).
- **RL for coding is compressing the gap for small open models**: W&B highlights **NousCoder-14B** improving **+7% on LiveCodeBench**, trained in **4 days**, as an example of open-source RL post-training getting real leverage ([Weights & Biases](https://twitter.com/wandb/status/2008946807523692965)). Nous also shipped a dataset later (“We forgot to release the dataset!”) ([Teknium](https://twitter.com/Teknium/status/2008857949524074635)).
- **Vision/video open models**:
  - **Black Forest Labs**: quantized **FLUX.2 [dev] 32B** on Hugging Face; highlights include multi-reference (up to **10 images**), **4MP** resolution, improved text rendering, optimized for NVIDIA GPUs ([HuggingPapers](https://twitter.com/HuggingPapers/status/2008762251352711235)).
  - **LTX-2**: claims #1 on Artificial Analysis open-weights leaderboard for text-to-video and image-to-video ([ltx_model](https://twitter.com/ltx_model/status/2008862459327865121)); also discussed as a joint audio-visual foundation model ([@_akhaliq](https://twitter.com/_akhaliq/status/2008964274186789217)).
  - **OmniHuman 1.5 720P** on fal: avatar video from image+audio+text, improved face consistency, lip-sync, camera/body control ([fal](https://twitter.com/fal/status/2008922947562471802)).
  - **Qwen image-edit tooling**: fal releases a multi-angle camera control LoRA for Qwen-Image-Edit-2511 trained on **96 camera poses** and **3000+ Gaussian Splatting renders** ([fal](https://twitter.com/fal/status/2008954582018248755)).
- **Eval/leaderboard trust issues**: Teknium argues LM Arena has become “pay to win,” incentivizing model quality regressions to maximize leaderboard scores, and claims submissions are unevenly handled ([Teknium](https://twitter.com/Teknium/status/2008828875355443634)). Separately, a “scaling is dead” paper/essay discourse triggers pushback: the critique is that aggregate “6 task” averages and open-only comparisons can mislead; “scaling laws != scaling” and closed frontier gaps remain visible in real conversation quality ([giffmana](https://twitter.com/giffmana/status/2008825049889845452)).
- **Benchmarks moving toward long-horizon agent realism**: CodeClash is introduced as an iterative, adversarial long-horizon SWE benchmark with a newly released training set ([OfirPress](https://twitter.com/OfirPress/status/2008986204088545423))—aligned with the broader shift from single-shot coding to multi-step tool+execution loops.

---

**Retrieval & indexing: from “RAG” to long-context + new local indexes**

- **LEANN: “stop storing embeddings”**: A notable systems claim: index **60M text chunks using 6GB** (vs “200GB”) by storing a compact graph and recomputing embeddings selectively at query time; pitched as enabling local RAG at new scales ([LiorOnAI](https://twitter.com/LiorOnAI/status/2008871398433759298), repo link: [github](https://twitter.com/LiorOnAI/status/2008871399813759033)). Engineers should sanity-check latency/throughput tradeoffs and recall under recomputation, but the “graph + selective recompute” direction matches broader storage/edge constraints.
- **RLMs vs retrieval (lateinteraction’s stance)**: Retrieval isn’t “going away” because corpus-scale querying demands sublinear access via indexes; RLMs are framed as long one-off context rather than a replacement for retrieval systems ([lateinteraction](https://twitter.com/lateinteraction/status/2008766087752511718)). Also a reminder that “retrieve-then-read” RAG workflows were “dead by end of 2020” in favor of more iterative architectures like Baleen ([lateinteraction](https://twitter.com/lateinteraction/status/2008768325908918328)).
- **Real-time retrieval in voice agents**: Qdrant demo: live phone-call voice agent querying a dealership inventory from a Google Sheet indexed into Qdrant, responding in under a second ([qdrant_engine](https://twitter.com/qdrant_engine/status/2008810361924055370)). This reinforces a pragmatic pattern: structured filters + fast retrieval + voice UX.
- **Data extraction infra**: Hugging Face shared a deep dive on extracting usable data from **1.3B PDFs** ([eliebakouch](https://twitter.com/eliebakouch/status/2008933337994322167)), emphasizing that “PDFs are 0.6% of the web but hold high-value content”.

---

**Compute, kernels, and scaling discourse: Chinchilla-style science, post-training systems, and kernel-autotuning-by-AI**

- **Karpathy’s “nanochat miniseries v1”**: A practical recipe for doing *scaling-law science* on a budget: train compute-optimal miniseries, recover Chinchilla-like exponents (~0.5 on params and tokens), estimate a “compute-independent constant” (nanochat suggests **8** vs Chinchilla’s **20**), and relate results to GPT-2/3 via CORE score—total cost reported around **$100 (~4 hours on 8×H100)** ([karpathy](https://twitter.com/karpathy/status/2009037707918626874)). This is a useful template for teams trying to de-risk “the big run” with small systematic sweeps.
- **Prime-RL memory optimization**: “Vocab chunked lm_head with fused logprobs+entropy” avoids materializing full logits, yielding large memory savings ([m_sirovatka](https://twitter.com/m_sirovatka/status/2008905312992964687)). This is the kind of low-level win that directly expands feasible RL/post-training batch sizes.
- **Kernel generation and evaluation via full systems**: A report on an AI-generated fused RMSNorm kernel integrated into vLLM showing **40% speedup** over existing RMSNorm and **+1.6% e2e**; observation: AI writes long heuristic/autotuner-like code and can introduce stability risks (segfault edge cases), raising the question of how much fallback and determinism debt the community will tolerate ([marksaroufim](https://twitter.com/marksaroufim/status/2009096176789016600)).
- **Hardware narrative from CES**: A coherent “where it runs” framing: Qualcomm pushing always-on local inference (~80 TOPS NPUs), NVIDIA emphasizing centralized “AI factory” + physical deployment loops, AMD emphasizing heterogeneous continuity across cloud/PC/edge ([TheTuringPost](https://twitter.com/TheTuringPost/status/2009052319871316060)). This maps cleanly onto agent UX demands: low latency locally, heavy reasoning in cloud, and tooling that can route across both.

---

**Applied AI products: Health, voice companions, robotics demos, and on-device small models**

- **ChatGPT Health launches (privacy- and data-integration-heavy)**: OpenAI introduces a dedicated health space with the ability to securely connect medical records and wellness apps to ground responses in user data ([OpenAI](https://twitter.com/OpenAI/status/2008987566796640575), announcement link: https://openai.com/index/introducing-chatgpt-health/). Notable implementation details shared: extra encryption layers (per-user keys), enhanced isolation/segmentation, Health chats excluded from training regardless of settings, and Health memory isolated from global memory ([cryps1s](https://twitter.com/cryps1s/status/2009040709635199151)). Early rollout via waitlist with expansion to all users including free tiers ([thekaransinghal](https://twitter.com/thekaransinghal/status/2008990098193633529), [nickaturley](https://twitter.com/nickaturley/status/2009007121942417530)).
- **On-device summarization as a “small model” wedge**: Liquid AI + AMD announce **LFM2-2.6B-Transcript**, targeting long meeting transcripts with **<3GB RAM**, local execution across CPU/GPU/NPU, and “cloud-quality” summarization ([liquidai](https://twitter.com/liquidai/status/2008954886659166371); recap by [maximelabonne](https://twitter.com/maximelabonne/status/2008955850665415152)). This reinforces a pattern: domain-tuned small models delivering production utility under tight privacy/latency constraints.
- **Voice-first agents and robotics**:
  - CES viral assistant demo built with **pipecat_ai**, spanning multi-model/multimodal hybrid cloud+local, robot control, and voice interfaces; includes open-source hardware **Reachy Mini** ([kwindla](https://twitter.com/kwindla/status/2008743885523349774)).
  - Hobbyist Reachy Mini “face follow” project: fine-tuned detector and control loop; includes dataset and tutorial links ([skalskip92](https://twitter.com/skalskip92/status/2008923043888841018)).
- **Enterprise deployment of coding agents**: Cognition partners with Infosys to deploy Devin; claims include complex COBOL migrations “in record time” ([cognition](https://twitter.com/cognition/status/2008984320564981780)).

---

**Ecosystem & strategy signals: China/open-source adoption, funding arms race, and “social distribution” moats**

- **Open-weight adoption shifting toward China-led ecosystems**: Nat Lambert shares updated “open model ecosystem” plots emphasizing China’s growing adoption lead ([natolambert](https://twitter.com/natolambert/status/2008920674442637635)). Stanford NLP highlights Alibaba **Qwen** as “landslide victory” in open-weight usage ([stanfordnlp](https://twitter.com/stanfordnlp/status/2008953208907927601)). Clement Delangue notes **South Korea’s state-supported open-source AI** producing multiple trending HF models ([ClementDelangue](https://twitter.com/ClementDelangue/status/2008954270411051465)).
- **xAI strategy: distribution-first via X**: xAI is framed as uniquely advantaged by owning a social network (real-time data + ~250M daily users), pushing Grok through the product surface; “others build better models, xAI builds attention” ([Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2008765567922999573)). Separate tweet claims xAI raised **$20B** and became the second most funded AI lab ([Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2008774688382599520)).
- **Funding continues to balloon**: Report: Anthropic planning to raise **$10B at $350B valuation** ([SawyerMerritt](https://twitter.com/SawyerMerritt/status/2008964178204295429)).
- **Developer UX meta-signal**: multiple tweets note a “confidence UX” impact from visible reasoning traces (DeepSeek’s “showing its work”) and speculate the next UX innovation is overdue ([dbreunig](https://twitter.com/dbreunig/status/2008928100009267553))—consistent with the broader push toward agent transparency (“what am I reading/doing now and why?”) rather than raw chain-of-thought dumps.


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Local AI Model Performance Benchmarks

  - **[llama.cpp vs Ollama: ~70% higher code generation throughput on Qwen-3 Coder 32B (FP16)](https://www.reddit.com/r/LocalLLaMA/comments/1q64f26/llamacpp_vs_ollama_70_higher_code_generation/)** (Activity: 303): **A user reports a significant performance difference in code generation throughput between **llama.cpp** and **Ollama** when using the **Qwen-3 Coder 32B** model with `FP16` precision on an **RTX 5090 + RTX 3090 Ti** setup. The throughput for **llama.cpp** is approximately `52 tokens/sec`, while **Ollama** achieves only `30 tokens/sec`, indicating a `~70%` performance advantage for llama.cpp. The user speculates that the discrepancy could be due to differences in CUDA kernels, attention implementations, context or batching defaults, scheduler or multi-GPU utilization, or overhead from Ollama's runtime/API layer.** Commenters suggest that **Ollama** is less suitable for serious work compared to **llama.cpp**, which is seen as more efficient and straightforward. There is skepticism about the existence of a **Qwen-3 Coder 32B** model, with a suggestion that the user might have meant **Qwen-3 Coder 30b a3b**.

    - Ollama's implementation has been criticized for its handling of GPU layers and tensor assignments, particularly in the context of MoE models and multiple GPUs. A user pointed out that Ollama's heuristics for setting the number of GPU layers are suboptimal, leading to inefficient tensor placement. In contrast, a recent implementation in `llama.cpp` has improved this by being MoE-aware and better utilizing VRAM, resulting in enhanced performance. [Source](https://www.reddit.com/r/LocalLLaMA/comments/1pn2e1c/llamacpp_automation_for_gpu_layers_tensor_split/).
    - There is some confusion regarding the model name, with a user questioning the existence of 'Qwen 3 Coder 32B' and suggesting it might be a typo for 'Qwen 3 Coder 30b a3b'. This highlights the importance of precise model naming in discussions to avoid misunderstandings.
    - Ollama is perceived as a tool for beginners, offering ease of use at the cost of flexibility and performance. Experienced users are advised to use `llama.cpp` directly for more control and better results, as Ollama's design choices often do not align with the needs of serious work.

  - **[Running ACE-Step locally: 4-minute music generation in 20 seconds on 8GB VRAM (vs Suno's cloud API)](https://www.reddit.com/r/LocalLLaMA/comments/1q64qpx/running_acestep_locally_4minute_music_generation/)** (Activity: 16): **The post discusses setting up **ACE-Step** locally to generate 4 minutes of music in approximately `20 seconds` using `8GB VRAM` with CPU offload, as an alternative to **Suno's** cloud API, which has rate limits and costs `$30/month`. The setup includes optimizations like CPU offload reducing VRAM usage from `16GB` to `7.5GB` and `8-bit quantization` reducing it to `9GB` with only a `25%` slowdown. The article provides a comprehensive guide on installation, quality control, and advanced features like stem-style generation and LoRA loading for genre specialization. It emphasizes the efficiency of ACE-Step's diffusion-based architecture over traditional autoregressive models, enabling rapid multi-minute music generation.** One commenter questioned the quality of the generated music, noting it was previously subpar compared to Suno's level. Another appreciated the 'Real-World Use Cases with Full Code' section and expressed intent to try the setup.



### 2. Agent Safety and Fail-Closed Systems

  - **[I built a "Fail-Closed" Circuit Breaker for my Agent because prompts weren't enough to stop hallucinations. Open sourcing it today. (Python)](https://www.reddit.com/r/LocalLLaMA/comments/1q64zgt/i_built_a_failclosed_circuit_breaker_for_my_agent/)** (Activity: 6): **The post introduces **FailWatch**, a middleware designed to enforce deterministic safety in agent operations by implementing a "Fail-Closed" circuit breaker. This system is crucial for preventing large-scale errors in financial transactions, especially when network failures or validation logic crashes occur. The middleware operates by blocking actions that exceed predefined limits, requiring human approval for ambiguous actions, and locking down operations during network outages. It is implemented as a Python decorator, ensuring synchronous validation before tool execution, which is critical for maintaining control over potentially risky operations. The tool is open-sourced and available on GitHub and via pip.** A commenter appreciates the 'fail-closed' approach, noting that many frameworks inadequately handle errors, leading to potential financial mishaps. Another concern raised is about the potential latency introduced by synchronous validation, questioning whether the guard server is local to mitigate this.

    - The implementation of a 'fail-closed' circuit breaker is praised for its cautious approach, contrasting with many agent frameworks that proceed despite errors, potentially leading to costly mistakes. The commenter highlights the importance of this approach in preventing unintended actions, such as erroneous financial transactions.
    - A technical concern is raised about the potential latency impact of synchronous validation before every tool call, especially in scenarios involving numerous chained actions. The commenter inquires whether the guard server is local, which could mitigate latency issues, suggesting that the architecture of the solution could significantly affect performance.

  - **[Double GPU vs dedicated AI box](https://www.reddit.com/r/LocalLLM/comments/1q6f7ea/double_gpu_vs_dedicated_ai_box/)** (Activity: 41): **The user is considering whether to add another RTX 4080 GPU or purchase a dedicated AI box like the GMKtec Evo-X2 with 128GB for running private LLM tasks such as inference, document summarization, and light image generation. The RTX 4080 is sufficient for small tasks, but the user is contemplating fine-tuning on internal documents. A dedicated machine with Nvidia GPUs is recommended for better performance, especially for running models via API, as it allows for separation of workloads and efficient resource management. Adding another RTX 4080 would provide 32GB of VRAM, suitable for running 14b and 20b parameter models efficiently. Alternatively, an RTX 6000 with 96GB VRAM is suggested for more extensive capabilities if budget is not a constraint.** Commenters generally favor using Nvidia GPUs over integrated memory solutions for speed and efficiency. A dedicated machine is preferred for running models, allowing for better management and performance, especially when accessed via API. The addition of another RTX 4080 is seen as a cost-effective way to enhance capabilities without significant system slowdown.

    - **fastandlight** suggests using a dedicated machine for running AI models with Nvidia GPUs, emphasizing the benefits of separating the workload from personal devices. They recommend using older PCIe v4 machines with ample slots and RAM, running Linux, and utilizing software like `vllm` or `llama.cpp` in OpenAI serving mode. This setup allows for remote access via API, keeping the main device free from the computational load and heat generated by the GPUs.
    - **alphatrad** highlights the performance advantage of GPUs over integrated memory systems, particularly for running large models. They suggest that adding another RTX 4080 to achieve 32GB VRAM would be ideal for handling 14b and 20b parameter models efficiently. This setup would maintain system usability without significant slowdowns, making it suitable for tasks like Retrieval-Augmented Generation (RAG).
    - **LaysWellWithOthers** advocates for using multiple RTX 3090 GPUs due to their cost-effectiveness in terms of VRAM per dollar. They emphasize the importance of ensuring the system can physically accommodate additional GPUs, including considerations for power supply capacity and thermal management. They share their personal setup of a dedicated AI workstation with 4x3090s in an open airframe, highlighting the scalability and performance benefits of such a configuration.


### 3. AI Model Setup and Troubleshooting on Google Colab

  - **[Need help with Collab!](https://www.reddit.com/r/LocalLLM/comments/1q6frf8/need_help_with_collab/)** (Activity: 1): **The user is attempting to run AI models on Google Colab, specifically using the `chatterbox turbo` model for text-to-speech (TTS) tasks. They encounter issues with multi-line string inputs producing gibberish unless split into chunks, which disrupts natural pauses. The user notes missing features in chatterbox TTS, such as `cfg` and `exaggeration` parameters. They are exploring alternatives like `vibevoice` but only find the `0.5B` model available, not the `1.5B`. They seek guidance on setting up an interface like `Gradio` for easier interaction, similar to their experience with `Pinokio`.** Commenters suggest exploring other TTS models that might better support multi-line inputs and recommend using `Gradio` for a user-friendly interface. Some highlight the importance of checking model compatibility with the T4 GPU on Colab and suggest looking into community forums or GitHub repositories for more comprehensive guides.




## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Gemini and AI Chatbot Market Trends

  - **[Gemini surpassed 20% traffic share threshold among the overall traffic for AI chatbots(Jan 2026)](https://www.reddit.com/r/singularity/comments/1q6a3lp/gemini_surpassed_20_traffic_share_threshold_among/)** (Activity: 659): **The image is a bar chart from a **Similarweb report** showing the traffic share of various AI chatbots, highlighting that **Gemini** has recently surpassed the `20%` traffic share threshold. **OpenAI's ChatGPT** still holds the largest share but has dropped below the `65%` mark. **Grok** is also noted for surpassing `3%` and is approaching **DeepSeek**. This data reflects a shift in the AI chatbot market dynamics over the past year, with Gemini gaining significant traction.** One commenter highlights the impact of **Gemini 3** on the market, noting its scientific approach and fine-tuning capabilities. Another raises a question about whether the market is expanding or if providers are merely redistributing existing users.

    - The release of Gemini 3 has highlighted the significant market share previously held by OpenAI, indicating a shift in the competitive landscape of AI chatbots. The introduction of Gemini 3 Pro, which emphasizes a scientific method in problem-solving and benefits from fine-tuning, has been noted for its substantial impact on real-world applications, suggesting a strong competitive edge in the market.
    - There is a discussion on whether the AI chatbot market is expanding or if companies are merely capturing each other's users. This raises the question of whether the market share increase for some providers might be misleading if the overall market size is contracting, indicating a need for more detailed market analysis to understand true growth dynamics.
    - Google's strategy to increase Gemini's market share involved offering a full year of Gemini Pro for free globally, targeting students but accessible to anyone. This aggressive promotional tactic is aimed at attracting new users and converting users from other AI platforms, highlighting the competitive strategies employed to gain market traction.

  - **[Gemini surpassed 20% traffic share threshold among overall traffic for AI chatbots](https://www.reddit.com/r/GeminiAI/comments/1q69y88/gemini_surpassed_20_traffic_share_threshold_among/)** (Activity: 180): **The image is a bar chart from a **Similarweb report** that tracks the global traffic share of AI chatbots as of 2026. It highlights that **Gemini** has surpassed a `20%` traffic share, marking a significant milestone. Meanwhile, **ChatGPT** has seen a decline, dropping below the `65%` mark, and **Grok** is gaining traction, surpassing `3%` and nearing the shares of **DeepSeek**. This data reflects shifting dynamics in the AI chatbot market, with Gemini's growth being particularly notable.** One comment suggests that Gemini's growth might be affecting its performance, as a user describes the Gemini 3 Pro as 'unusable' and 'completely broken.' Another comment anticipates changes in the market dynamics once OpenAI introduces ads.


  - **[ChatGPT is losing market share as Google Gemini gains ground](https://www.reddit.com/r/GeminiAI/comments/1q6skak/chatgpt_is_losing_market_share_as_google_gemini/)** (Activity: 287): ****ChatGPT** is reportedly losing market share to **Google Gemini**, as Google leverages its extensive ecosystem to integrate AI features more seamlessly into daily workflows. The article suggests that while OpenAI's ChatGPT was initially a groundbreaking demonstration of AI capabilities, Google's infrastructure and user base provide a more compelling offer, including features like family-sharing and 2TB cloud storage. This shift highlights the strategic advantage of companies with comprehensive platforms, as they can embed AI as a feature to enhance their existing services, rather than as a standalone product.** Commenters argue that Google's extensive ecosystem, including services like Mail, Docs, and YouTube, provides a significant advantage over standalone AI applications like ChatGPT. They suggest that OpenAI may continue to lose market share unless it integrates more deeply into a larger platform, potentially through acquisition by a company like Microsoft.

    - Google's competitive advantage lies in its extensive ecosystem, including services like Mail, Sheets, Docs, Drive, and YouTube, which are deeply integrated into users' daily workflows. This integration makes it easier for users to adopt Google's AI offerings, as they are already embedded in a familiar environment, unlike standalone applications like ChatGPT.
    - Google's offering of additional services, such as 2TB of cloud storage and family-sharing options, provides a compelling value proposition that goes beyond just AI capabilities. This bundling strategy makes Google's AI services more attractive to users who are already invested in Google's ecosystem, potentially leading to a shift in market share away from standalone AI products like ChatGPT.
    - The discussion highlights a potential trend where standalone AI applications may struggle to compete against tech giants with established platforms and ecosystems. As these companies integrate AI into their existing services, users may find it cumbersome to use separate applications that do not offer the same level of integration, leading to a decline in the use of standalone AI apps like ChatGPT.

  - **[Is it just me, or has Gemini been lobotomized recently?](https://www.reddit.com/r/Bard/comments/1q65rty/is_it_just_me_or_has_gemini_been_lobotomized/)** (Activity: 190): **Users are reporting significant performance degradation in **Gemini**, a language model service, over the past few weeks. Issues include slow response times, frequent crashes, increased hallucinations, and poor adherence to instructions. Users also note excessive use of idioms, irrelevant personal information injection, and a failure to analyze images correctly. Despite attempts to reset settings and change models, these problems persist, leading to frustration among users who previously found the service beneficial.** Commenters express frustration with Gemini's current state, highlighting its inability to retain context beyond a few messages and poor data retention. Some users are considering switching back to alternatives like ChatGPT due to these issues, and there is criticism of the service's integration features, such as NotebookLM, which are described as ineffective.

    - **Goldengod4818** highlights significant issues with Gemini's data retention, noting that it struggles to "see" beyond the last 10 messages, which severely impacts its usability for long-term projects. They mention attempting to integrate NotebookLM to enhance functionality, but describe it as a "disaster," indicating that the integration does not effectively support complex tasks or improve the user experience.
    - **DearRub1218** provides a detailed analogy to describe Gemini's instability, comparing it to a human with misaligned nerves and muscles, leading to unpredictable performance. They note that while Gemini can occasionally perform exceptionally well, it often fails to deliver consistent results, likening its operation to a "disjointed break dance." This suggests that the model's internal logic or architecture may be misconfigured, leading to erratic behavior.
    - **locomotive-1** discusses the degradation in Gemini's performance, particularly in longer conversations where it tends to repeat itself. They speculate that recent optimizations might have been made to balance cost and quality, potentially reducing the effective context window from its intended 1 million tokens, which could explain the observed decline in performance.

  - **[Gemini 3.0 has been nerfed big time](https://www.reddit.com/r/GeminiAI/comments/1q6ecwy/gemini_30_has_been_nerfed_big_time/)** (Activity: 502): **The post claims that **Gemini 3.0** has been significantly downgraded, particularly in terms of its context window, which was initially announced to support `1 million tokens`. Users report that the model now forgets information after just a few messages, contradicting the initial claims. Additionally, the model is criticized for not following instructions, such as refusing to perform web searches, and for injecting irrelevant personal context into responses. The user has switched to using **Claude** for coding tasks and **Gemini 3.0** on **Perplexity** for web browsing, citing a better experience there.** Commenters agree with the post, noting that the context window issue feels like a 'sneaky downgrade' and that the model's performance has deteriorated to the point where it is no longer a reliable tool.

    - Users report significant issues with Gemini 3.0's context retention, noting that it struggles to maintain coherent conversation threads, often repeating irrelevant details from earlier in the chat. This suggests a degradation in its ability to handle extended dialogues effectively, which is critical for maintaining user engagement in conversational AI applications.
    - There is skepticism about the claimed 'one million token context window' as users find that Gemini 3.0 struggles with even moderately sized documents, such as a 100-page PDF. This discrepancy raises questions about the model's actual capabilities versus its advertised specifications, highlighting potential overstatements in marketing claims.
    - Despite the issues with Gemini 3.0, users still find value in related tools like Studio and Notebooklm, indicating that while the main model may have limitations, the ecosystem of tools around it still offers utility. This suggests that while the core model may need improvements, the supporting tools can still provide a satisfactory user experience.

  - **[Yes, Gemini, that is in fact a genuine universally-available seahorse emoji...](https://www.reddit.com/r/Bard/comments/1q67vp6/yes_gemini_that_is_in_fact_a_genuine/)** (Activity: 74): **The image highlights a phenomenon where AI models, like ChatGPT, exhibit confusion or errors when asked about the existence of a seahorse emoji, despite it being available since Unicode 15.0 (2022). This issue is attributed to ambiguous tokens and conflicting training data, leading to AI 'hallucinations.' The discussion points to broader challenges in AI training, where models may inherit misinformation or inconsistencies from their training datasets, affecting their ability to accurately recognize or recall certain information.** One comment suggests that the issue is not specific to Gemini but rather a result of using a 'super cheap diffusion model,' indicating a broader problem with AI models. Another comment raises concerns about the potential for AI models to pick up deliberate misinformation, questioning the reliability of their outputs.

    - The discussion highlights a common misconception about AI models, specifically pointing out that the model in question is not **Gemini** but rather a 'super cheap diffusion model'. This suggests a misunderstanding or mislabeling of AI capabilities, which can lead to misinformation about what these models can actually do. Diffusion models are a class of generative models that have been gaining attention for their ability to generate high-quality images, but they are distinct from models like Gemini, which may have different architectures or purposes.
    - The comment 'They're all drinking from the same poisoned well' metaphorically criticizes the spread of misinformation regarding AI models. This suggests a broader issue in the AI community where incorrect information about model capabilities and origins is propagated, potentially leading to confusion among users and developers. It underscores the importance of accurate information dissemination in the field of AI to prevent the spread of such misconceptions.

  - **[Thank god for the free trial lol](https://www.reddit.com/r/Bard/comments/1q64r5s/thank_god_for_the_free_trial_lol/)** (Activity: 70): **The image humorously highlights the cost-saving benefits of a free trial for the Gemini API, showing a billing summary where the user saved $224, resulting in a total cost of $0. The comments reveal that users are cautious about the costs associated with token generation, as the API charges based on the entire chat's token usage per response. Users discuss strategies to manage costs, such as using the 'Context Cache' to significantly reduce token costs from $2 per million tokens to $0.02, indicating a focus on cost efficiency when using advanced AI models like Gemini 3.0 Flash.** Users express relief and caution regarding the cost of using the Gemini API, with some opting to delete their API keys to avoid unexpected charges. There is a discussion about the effectiveness of different tiers and the potential cost savings with tools like 'Context Cache.'

    - MuriloZR discusses transitioning from the Free Tier Gemini 2.5 Flash Lite to the Paid Tier Gemini 3.0 Flash, highlighting the significant performance improvements. They mention using 'Context Cache' to optimize costs, reducing expenses from `$2 per 1M tokens` to `0.02$`, which is a substantial cost-saving measure for high-volume token generation.
    - Unable_Classic3257 points out a common misunderstanding about token generation costs, noting that the API generates tokens from the entire chat with each response. This can lead to unexpectedly high costs if not managed properly, as they experienced hitting `$8` quickly before realizing the cost structure.
    - Nayomhee raises a concern about post-trial charges, questioning the billing process after the free trial period ends. This highlights the importance of understanding subscription models and potential automatic charges in cloud services.

  - **[Paid vs free Gemini account](https://www.reddit.com/r/GeminiAI/comments/1q6c2j9/paid_vs_free_gemini_account/)** (Activity: 69): **The post discusses the benefits of a paid versus free account for **Gemini**, a service likely related to **Google** given the context. Users report that the paid version, costing `£20` per month, offers significant advantages such as reduced usage limits and access to advanced models, which are particularly beneficial for tasks like research, analysis, and coding. The paid version also includes additional features like more storage and integration with other Google services, such as **YouTube Premium** and **Nest Aware Plus**.** Commenters generally agree that the paid version of Gemini is worthwhile for those who frequently use its advanced features, as it saves time and enhances productivity. However, for basic tasks, the free version may suffice.

    - Overall-Fan3079 highlights that the paid version of Gemini significantly reduces usage limits, which is a major advantage over the free version. They note that the advanced model in the paid version performs better in coding tasks, although for basic queries, the difference is not substantial.
    - Pasto_Shouwa points out that the Pro subscription of Gemini allows for family sharing with up to 5 additional users, making it cost-effective at 22 USD for 6 accounts. Additionally, the subscription includes 2TB of shared storage, enhancing its value proposition for users needing extensive storage solutions.

  - **[Another example of the Pro Model Making Ridiculous Mistakes](https://www.reddit.com/r/GeminiAI/comments/1q66mn4/another_example_of_the_pro_model_making/)** (Activity: 66): **The post highlights a recurring issue with the Pro Model of a language model, which falsely claims to interpret attached images and provides incorrect descriptions. The user expresses frustration over the model's inaccuracies and the perceived decline in service quality, especially given the subscription cost. The image in question is a simple photograph of a dog, which the model failed to describe accurately, leading to user dissatisfaction. This issue may be linked to recent updates or features, such as a memory feature, that have affected the model's performance.** Commenters suggest that the problem might be due to recent updates, such as the addition of a memory feature, and speculate that a new version (3.1) might be released soon to address these issues.

    - the_shadow007 highlights a potential issue with the model's performance, suggesting that the introduction of a memory feature may have inadvertently caused degradation. This implies a trade-off between new features and model stability, a common challenge in machine learning development.
    - ComplexActivity43 criticizes the business strategy of maintaining subscription prices despite perceived downgrades in model performance. This points to a broader issue of customer satisfaction and value perception in AI services, especially when updates do not meet user expectations.
    - NoWheel9556 notes a decline in performance post-Gemini 3 Pro launches, indicating that newer versions may not always equate to better performance. This suggests a need for thorough testing and validation before deploying updates to ensure they enhance rather than hinder the user experience.


### 2. New AI Model and Feature Releases

  - **[Claude-Code v2.1.0 just dropped](https://www.reddit.com/r/ClaudeAI/comments/1q6q9my/claudecode_v210_just_dropped/)** (Activity: 549): ****Claude-Code v2.1.0** introduces significant updates, including automatic skill hot-reload, support for forked sub-agent contexts, and a new `language` setting for response language configuration. Notable fixes address security issues with sensitive data exposure in debug logs and session persistence problems. The update also enhances terminal compatibility and performance, particularly for iTerm2, WezTerm, and Kitty, and adds new Vim motions and slash command features. However, a critical bug causes the changelog parser to fail due to an invalid version date format, prompting a rollback to v2.0.76. [GitHub Commit](https://github.com/anthropics/claude-code/commit/870624fc1581a70590e382f263e2972b3f1e56f5).** A user reported that the update broke Claude-Code, with a specific bug related to version parsing causing the changelog display to fail. A workaround involves editing the changelog file to remove the date, and the developers have temporarily rolled back to v2.0.76.

    - A bug in Claude-Code v2.1.0 causes a crash due to an invalid version string format in the changelog display, specifically the inclusion of a date `2.1.0 (2026-01-07)`. This issue is documented in [GitHub issue #16671](https://github.com/anthropics/claude-code/issues/16671). A workaround involves editing the changelog file to remove the date using the command: `sed -E -i'' 's/(## 2\.1\.0) \([0-9-]*\)/\1/' ~/.claude/cache/changelog.md`.
    - The developers have temporarily rolled back the version to v2.0.76 due to the bug in v2.1.0. This rollback is a stopgap measure while they address the issue with the version string parsing that caused the crash.
    - Users are advised not to update to v2.1.0 as it contains a critical bug that affects the changelog parsing, leading to application crashes. The issue is significant enough that it prompted a rollback to the previous stable version, v2.0.76.

  - **[tried new model glm 4.7 for coding and honestly surprised how good it is for an open source model](https://www.reddit.com/r/ClaudeCode/comments/1q6f62t/tried_new_model_glm_47_for_coding_and_honestly/)** (Activity: 102): ****GLM 4.7**, an open-source model by **Zhipu AI**, has been tested for various coding tasks such as Python debugging, React component generation, SQL query optimization, and explaining Java legacy code. The model delivered functional code approximately `90%` of the time, outperforming other Chinese models like DeepSeek and Kimi in terms of stability and context handling. While not as polished as Claude Sonnet 4.5 in explanations, GLM 4.7 offers comparable code output quality at a fraction of the cost, making it a viable alternative for cost-effective coding tasks. The model can handle files over `500` lines without performance issues and can be run locally, which is advantageous for proprietary projects.** Some users found GLM 4.7 underwhelming compared to other models like SWE-1.5, citing issues with basic requirements. However, others successfully integrated it with Claude Code, benefiting from higher limits and significantly reduced costs, with one user noting a `5%` usage for a comprehensive code refactoring task. The model is praised for its cost-effectiveness and performance in moderately complex tasks.

    - DenizOkcu highlights the cost-effectiveness and performance of GLM 4.7 when integrated with Claude Code, noting that it offers '3x higher limits' at '1/7th of the price' compared to other models. They provide a configuration snippet for setting up GLM 4.7 in Claude Code, emphasizing its ability to handle complex tasks like refactoring a large production code base efficiently, using only 5% of their hourly limit.
    - coopernurse mentions using GLM 4.7 alongside MiniMax 2.1 with Claude Code, noting that both models perform well for moderately complex tasks. They are in the process of comparing the two models to determine any significant differences in performance, suggesting that both are capable of handling complex coding tasks effectively.
    - AriyaSavaka points out the affordability of the GLM Plan, which costs '$3/month for 3x usage' compared to the $20 Claude Pro plan, and highlights the absence of a weekly limit. This suggests that GLM 4.7 offers a cost-effective solution for users needing extensive usage without the constraints of higher-priced plans.

  - **[OpenAi releases ChatGPT Health on mobile and web](https://www.reddit.com/r/OpenAI/comments/1q6ouuf/openai_releases_chatgpt_health_on_mobile_and_web/)** (Activity: 629): **OpenAI has launched ChatGPT Health, a new feature available on mobile and web platforms, designed to facilitate private health-related conversations. This service allows users to securely connect their medical records and wellness apps, such as Apple Health, Function Health, and Peloton, to ChatGPT. The interface includes options for health check-ins, explanations of medical reports, and workout suggestions, aiming to provide a comprehensive health management tool. The design emphasizes user-friendliness and privacy in handling sensitive health data.** Some users express skepticism about the chatbot's ability to accurately interpret medical records, comparing it humorously to WebMD. There is also a cautionary note about the limitations of discussing mental health through the platform.

    - A key concern raised is about data privacy, specifically whether users' medical records and interactions with ChatGPT Health are secure or if they might be shared with third parties, such as media outlets like the New York Times. This highlights the importance of understanding OpenAI's data handling and privacy policies for this new service.
    - There is skepticism about the reliability of ChatGPT Health in interpreting medical records accurately. The comparison to WebMD suggests a concern that the chatbot might misinterpret medical information, which could lead to incorrect advice or diagnoses, emphasizing the need for robust validation and testing of the AI's medical capabilities.
    - The discussion touches on the ethical implications of using AI for health-related queries, particularly the potential for misuse of sensitive health data. This raises questions about the ethical responsibilities of AI developers in ensuring that their tools are used appropriately and that users are fully informed about the risks involved.

  - **[[P] Re-engineered the Fuzzy-Pattern Tsetlin Machine from scratch: 10x faster training, 34x faster inference (32M+ preds/sec) &amp; capable of text generation](https://www.reddit.com/r/MachineLearning/comments/1q6igw3/p_reengineered_the_fuzzypattern_tsetlin_machine/)** (Activity: 29): **The post details a re-engineered version of the Fuzzy-Pattern Tsetlin Machine (FPTM) that achieves significant performance improvements through low-level optimizations. The new implementation is up to `10x faster in training` and `34x faster in inference`, achieving `32M+ predictions/sec` with `98% accuracy` on MNIST benchmarks using a Ryzen 7950X3D. Key optimizations include the use of SIMD instructions, cache-friendly memory layouts, and BitSet indexing. The enhanced efficiency allows for practical generative tasks, demonstrated by a character-level text generator producing Shakespearean-style text. The code is available on [GitHub](https://github.com/BooBSD/Tsetlin.jl).** One commenter suggests further optimization by rewriting the implementation in C and inquires about the specific HDC/VSA used, noting that BSDC-SEG codes have been effective in their experience.

    - The re-engineering of the Fuzzy-Pattern Tsetlin Machine (FPTM) has resulted in significant performance improvements, achieving 10x faster training and 34x faster inference, with over 32 million predictions per second. This suggests a substantial optimization over previous implementations, potentially making it highly suitable for real-time applications.
    - The integration of FPTM with Hyperdimensional Computing (HDC) or Vector Symbolic Architectures (VSA) is highlighted as a promising approach. The commenter mentions BSDC-SEG codes as particularly effective, indicating that the choice of HDC/VSA can significantly impact the performance and results of the FPTM.
    - There is a suggestion to rewrite the FPTM in C to further enhance performance. This implies that the current implementation might be in a higher-level language, and a C implementation could leverage lower-level optimizations for even greater speed improvements.

  - **[[R] DeepSeek-R1’s paper was updated 2 days ago, expanding from 22 pages to 86 pages and adding a substantial amount of detail.](https://www.reddit.com/r/MachineLearning/comments/1q6cb0k/r_deepseekr1s_paper_was_updated_2_days_ago/)** (Activity: 176): **The paper on **DeepSeek-R1** has been significantly expanded from `22` to `86` pages, providing more comprehensive details on its methodology and findings. The update may address previous issues, such as those in the `grpo` reward calculation, although this is not explicitly confirmed in the post. The paper is available on [arXiv](https://arxiv.org/abs/2501.12948).** A comment raises a question about whether the update resolves issues in the `grpo` reward calculation, indicating ongoing technical scrutiny and interest in the model's performance and implementation details.

    - The update to the DeepSeek-R1 paper significantly expands its content from 22 to 86 pages, suggesting a substantial increase in detail and possibly addressing previous issues. A key point of interest is whether the update resolves problems in the 'grpo reward calculation', which was a noted issue in earlier versions. This could impact the model's performance and accuracy, making it a critical area for review.
    - The expansion of the paper may also include more comprehensive experimental results or theoretical explanations, which are crucial for validating the model's claims. The increase in length could indicate a more thorough exploration of the model's architecture, training process, or application scenarios, providing deeper insights into its capabilities and limitations.
    - The mention of the paper's length in comparison to the SELU paper highlights the community's interest in the depth and comprehensiveness of research publications. Longer papers often suggest a more detailed exploration of the subject matter, which can be beneficial for researchers looking to understand the nuances of the model's implementation and potential applications.

  - **[James Cameron:"Movies Without Actors, Without Artists"](https://www.reddit.com/r/OpenAI/comments/1q69u4y/james_cameronmovies_without_actors_without_artists/)** (Activity: 560): ****James Cameron** expressed skepticism about AI-generated films, stating, *"I'm so not interested in that"*. He argues that AI could enable individuals without formal training or resources to produce films comparable to Hollywood within `4 years`. This perspective highlights a potential democratization of filmmaking, allowing those without access to expensive equipment or training to compete in the industry.** Commenters debate Cameron's stance, suggesting it reflects a resistance to change and democratization in filmmaking. Some argue that AI could empower new creators, much like digital cameras and platforms like YouTube have done, potentially leading to a surge in diverse and creative content.

    - James Cameron's perspective on AI in filmmaking highlights a potential democratization of the industry, where AI could enable individuals without traditional resources—such as expensive equipment or formal training—to produce films comparable to Hollywood standards within four years. This suggests a significant shift in the accessibility of filmmaking tools, potentially lowering barriers for new creators.
    - The discussion reflects a broader debate about the impact of AI on creative industries, with some commenters arguing that AI could disrupt traditional gatekeeping in Hollywood. By reducing the need for expensive resources, AI might allow more diverse voices to enter the market, similar to how platforms like YouTube democratized video content creation.
    - There is a recognition of the potential for AI to lead to a proliferation of content, much like the digital camera and YouTube revolutionized content creation. While this could result in a mix of quality, it also opens up opportunities for niche creators to find their audience, suggesting a future where creative expression is more accessible and varied.

  - **[OpenAI is reportedly getting ready to test ads in ChatGPT](https://www.reddit.com/r/OpenAI/comments/1q6nxy6/openai_is_reportedly_getting_ready_to_test_ads_in/)** (Activity: 87): ****OpenAI** is reportedly preparing to test advertisements within its **ChatGPT** platform, a move that could significantly alter user experience and monetization strategies. This development comes as OpenAI continues to explore sustainable revenue models for its widely-used AI service, which has seen rapid adoption across various sectors. The introduction of ads could potentially impact the seamless interaction users currently enjoy, raising questions about the balance between monetization and user satisfaction.** The community expresses skepticism and concern over the introduction of ads, with some users humorously suggesting that this could lead to a decline in subscriptions. The potential for ads to disrupt the user experience is a central theme in the discussion.


  - **[Pedophiles are using Sora to depict themselves abusing kids using YOUR children’s biometric data](https://www.reddit.com/r/OpenAI/comments/1q6521z/pedophiles_are_using_sora_to_depict_themselves/)** (Activity: 62): **The post raises concerns about the misuse of the Sora app's cameo feature, where pedophiles allegedly use children's biometric data to create videos depicting minors in inappropriate situations. The issue highlights the need for improved content moderation and security measures to prevent such exploitation. The post suggests that this is a widespread problem, with potentially hundreds of accounts involved.** Commenters emphasize the importance of not jumping to conclusions about the identity of the perpetrators, suggesting that the person posting the content might also be a victim. There is a call for stronger abuse detection and rapid takedown mechanisms to address such issues effectively.

    - RonaldWRailgun raises a critical point about the potential misuse of public profiles and the importance of privacy. They suggest that individuals involved in creating such content might use local models and private accounts rather than public social media, highlighting the complexity of identifying perpetrators in digital spaces.
    - Few-Needleworker4391 emphasizes the need for enhanced technological solutions to combat such issues, advocating for stronger abuse detection systems, age-gating mechanisms, and rapid content takedown processes. This underscores the importance of developing robust digital safety protocols to protect vulnerable populations.
    - Ok-Addition1264 notes the downvotes on the post, suggesting that the community's reaction might reflect deeper issues or misunderstandings about the topic. This comment hints at the challenges in community moderation and the interpretation of user feedback in sensitive discussions.

  - **[Wow, this is quite a situation.](https://www.reddit.com/r/ClaudeAI/comments/1q6kr4a/wow_this_is_quite_a_situation/)** (Activity: 868): **The image is a meme featuring a humorous take on AI-generated responses, specifically highlighting a tweet about the AI 'Claude' responding to a complex geopolitical situation with a simplistic and automated reply: 'Wow, this is quite a situation.' This reflects a broader discussion on AI's limitations in understanding nuanced contexts and generating appropriate responses. The comments further illustrate this by sharing anecdotes of AI's simplistic or bizarre responses to complex or absurd queries, highlighting the challenges in AI's comprehension and contextual awareness.** The comments humorously discuss AI's tendency to produce simplistic or bizarre responses to complex queries, reflecting on the limitations of AI in understanding nuanced contexts. This includes anecdotes of AI's responses to unrelated or absurd topics, emphasizing the need for improved contextual awareness in AI systems.

    - The comment by 'paralog' highlights a situation where an AI model, possibly a language model, was asked to find information about a speculative project involving Elon Musk and DOGE. The AI's response was vague, indicating a limitation in its ability to provide detailed or updated information on speculative or less-documented topics. This reflects a common issue with AI models where they struggle with real-time or speculative queries due to their reliance on pre-existing data.
    - The comment by 'Tim-Sylvester' discusses a bizarre internet debate involving a claim about Donald Trump and Bill Clinton, which was further complicated by references to a horse. This situation exemplifies the chaotic nature of internet discourse and the challenges AI models face in parsing and verifying such claims. The AI's process of considering various interpretations, including deepfakes and memes, highlights the complexity of distinguishing between genuine events and internet fabrications.
    - 'Icy_Quarter5910' shares an experience with an AI model, likely Claude, which provided enthusiastic feedback on an iOS SDK. The AI's response was notably positive, emphasizing the cleanliness and utility of the API. This interaction underscores the potential of AI models to assist in software development by evaluating and recommending tools, although the subjective nature of such feedback may vary depending on the model's training and data.


### 3. AI Model Usage and Alternatives

  - **[Overlimit with Claude Max 20x and need a plug-in alternative to fill-in short-term](https://www.reddit.com/r/ClaudeCode/comments/1q6h34n/overlimit_with_claude_max_20x_and_need_a_plugin/)** (Activity: 89): **The user has exceeded their usage quota for **Claude Max 20x** and is seeking a cost-effective alternative API to continue their work. They mention **GLM 4.7** as a potential option, which is noted for its utility in code clarification and small tasks like writing tests and refactoring. Another suggestion is **ChatGPT 5.2** on the Pro plan, which offers a `270k` context window and is considered a viable alternative to **Opus 4.5** for $20 per month.** One commenter suggests that the choice of API is subjective and based on personal experience, emphasizing the importance of finding a solution that works for individual needs. Another mentions a promotional offer from GPT, highlighting the variability in pricing and subscription options.

    - LinusThiccTips highlights that **ChatGPT 5.2** on the Pro plan offers a `270k context window`, which is significantly larger than **Opus 4.5** on a similar plan. This makes it a viable alternative for users needing extended context capabilities, especially when dealing with complex codebases or large datasets.
    - 13chase2 mentions **GLM 4.7** as a cost-effective option for experimenting with new code bases. However, they express concerns about privacy, as the data is sent to servers in China, which could be a potential issue for users with strict data privacy requirements.
    - silvercondor uses **GLM** (referred to as 'temu claude') for understanding and refactoring codebases, as well as writing tests. This suggests that GLM is versatile for both clarification and development tasks, making it a useful tool for developers needing assistance with code comprehension and modification.

  - **[What other plan / model would you recommend to replace Opus](https://www.reddit.com/r/ClaudeCode/comments/1q6c4bq/what_other_plan_model_would_you_recommend_to/)** (Activity: 76): **The Reddit post discusses issues with the Opus Max x5 plan, which has been underperforming since January, and seeks alternatives. Users suggest switching to GLM or Minimax plans, using Claude code router with the Gemini-cli plugin, and leveraging Opencode for feature parity, despite its bugs. Another approach is to use Max 5 in 'plan mode' to maintain session stability and productivity. The Opus 4.5 model is noted for its limitations, particularly in handling complex tasks without learning from context, but it excels in specific areas like DSP-based Rust audio plugin development. Users also recommend CC Web for its effectiveness in coding tasks.** Commenters debate the effectiveness of different plans, with some advocating for GLM and Minimax due to their cost-effectiveness and reliability, while others emphasize the importance of context and task-specific performance when using Opus 4.5. There is also a discussion on the value of using multiple sessions and plugins to maximize productivity.

    - trmnl_cmdr discusses a cost-effective approach using a combination of GLM, minimax plan, and Claude code router, supplemented by the Gemini-cli plugin. They highlight the availability of these tools in opencode, which offers feature parity with Claude code but is noted to be slightly buggier. This setup is described as a penny-pinching strategy, leveraging free and cheap plans for both planning and execution phases.
    - ridablellama shares their experience with GLM on opencode, noting its utility as a fallback when Opus encounters issues. They mention the cost-effectiveness of the minimax coding plan and the ability to use Claude code with GLM. However, they also point out that opencode tends to crash more frequently and has some differences compared to other platforms.
    - kronnix111 compares ChatGPT 5.2 and Claude, noting that GPT 5.2 has superior reasoning and bug detection capabilities but lacks integration with GitHub and terminal. They introduce a framework they developed, the LivingDocFramework, which can work with any codebase or AI. This framework facilitates bugfix scans by external agents, providing a structured approach to managing codebases.

  - **[Google AI Studio is becoming unusable: Constant rate limits and 60-second latency](https://www.reddit.com/r/Bard/comments/1q68317/google_ai_studio_is_becoming_unusable_constant/)** (Activity: 12): **Users of **Google AI Studio** are experiencing significant performance issues, including `60-second latency` and frequent "exceeded quota" notifications, prompting a shift towards requiring a paid API key. This change marks a departure from the previously free access model, affecting both the Pro and Gemini 3 Flash versions. The latency and rate limits are causing frustration among users who are accustomed to more seamless interactions.** Some users suggest deactivating the 'Grounding with Google Search' feature to potentially improve performance, while others express a pragmatic view that paying for valuable services is reasonable.

    - DearRub1218 highlights a significant performance issue with Google AI Studio, specifically mentioning that the G3 Pro model experiences a delay of 45-60 seconds before it begins processing. This latency is a critical concern for users relying on real-time or near-instantaneous responses from AI models, indicating potential server-side bottlenecks or inefficiencies in the current deployment.
    - Over-Customer2915 points out a persistent issue with the 'Grounding with Google Search' feature, which seems to be activated by default more frequently. This could be contributing to the increased latency and rate limits, as the feature might be consuming additional resources or bandwidth, affecting overall performance.
    - riowcaztoljp raises a question about the integration of AI Studio with the Google One plan, suggesting that users expected a more seamless or cost-effective integration. This indicates a potential gap between user expectations and the current service offerings, which could be impacting user satisfaction and perceived value.

  - **[Is this fraudulent charges to my bank account?](https://www.reddit.com/r/OpenAI/comments/1q6bwbt/is_this_fraudulent_charges_to_my_bank_account/)** (Activity: 78): **The image depicts two transactions labeled as 'OPENAI *CHATGPT SUBSCR' with amounts that do not align with the standard $20 ChatGPT Plus subscription fee, suggesting potential fraudulent activity. The user claims not to have subscribed to any paid plans, raising concerns about unauthorized charges. The transactions are dated in the future, which could indicate a clerical error or a more complex issue with the bank's processing system. The merchant category code '5734' is associated with computer software stores, which aligns with OpenAI's services but does not clarify the discrepancy in amounts or dates.** One commenter suggests freezing the card and reporting the issue, noting that prices can vary in different regions. Another points out that the partially obscured card information is still readable, advising the user to remove the post for security reasons.


  - **[Vibe Coding Local with 16GB VRAM | Dyad &amp; Oobabooga](https://www.reddit.com/r/Oobabooga/comments/1q6bed6/vibe_coding_local_with_16gb_vram_dyad_oobabooga/)** (Activity: 12): **The post discusses a setup for local coding using **Dyad** and **Oobabooga** with a `16GB VRAM` GPU, emphasizing that this configuration is sufficient for reliable and real coding tasks. The integration leverages the **Oobabooga API** as a backend to support Dyad, offering a free and local solution for automatic coding. This setup is particularly notable for its cost-effectiveness and open-source nature, making it accessible for developers with limited resources. For further technical details, the original video can be found [here](https://youtube.com/watch?v=DhKYjtCyD7U&si=fnt5kCLnPwaNKUvi).** Commenters are curious about the feasibility of using a `5070 16GB GPU` for a local AI NAS server, and whether a single host can support both Dyad development and GPU mounting. This indicates interest in practical hardware configurations and cost considerations for implementing the discussed setup.

    - A user inquires about the feasibility of using a `5070 16GB GPU` for a local AI NAS server. The discussion likely revolves around the GPU's capability to handle AI workloads locally, considering factors like VRAM capacity and processing power. The `16GB VRAM` is generally sufficient for many AI models, but the specific requirements would depend on the complexity and size of the models being run.
    - Another user expresses interest in purchasing a GPU with `16+ GB VRAM` for use with Dyad, a development environment. They are considering whether to integrate the GPU into their existing setup or if a separate server is necessary. This suggests a discussion on the integration of high-memory GPUs into existing systems, considering factors like power supply, cooling, and compatibility with current hardware.

  - **[[D] ICLR new ACs — how’s it going?](https://www.reddit.com/r/MachineLearning/comments/1q67hiq/d_iclr_new_acs_hows_it_going/)** (Activity: 42): **The post discusses the experiences of new Area Chairs (ACs) at **ICLR**, focusing on the challenges of decision-making without reliable review scores. A key issue highlighted is the difficulty in simulating the rebuttal process mentally, as ACs must judge whether authors' responses adequately address reviewers' concerns without assuming score changes. This process is described as challenging by many ACs, as noted in the shared email guidance from ICLR.** One commenter humorously notes a desire for their paper to be rejected due to subsequent improvements, highlighting the iterative nature of academic submissions and the constraints preventing withdrawal.

    - TheDeviousPanda highlights a challenging aspect of the Area Chair (AC) role at ICLR, where ACs must anticipate how reviewers might change their ratings after reading the authors' rebuttals. This requires ACs to mentally simulate the rebuttal process, which can be difficult and subjective. The comment suggests that many ACs might not expect reviewers to increase their scores, indicating a potential bias towards maintaining initial assessments.

  - **[[D] Intra-lab collaborations](https://www.reddit.com/r/MachineLearning/comments/1q6sgx5/d_intralab_collaborations/)** (Activity: 9): **The post discusses the challenge of balancing informal technical assistance with formal research collaboration in a clinical AI setting. The author, a physician with a strong ML/AI background, is frequently approached by colleagues for advice on model selection and analysis, which he feels crosses into the realm of research collaboration. He seeks advice on how to transition these interactions into formal collaborations, suggesting that the line between casual help and co-authorship is blurred in his current environment.** Commenters suggest establishing clear boundaries and negotiating formal collaboration terms if the assistance provided is critical to projects. They emphasize the importance of protecting one's time and ensuring contributions are recognized, either through co-authorship or other formal agreements.

    - The discussion emphasizes the importance of setting boundaries in intra-lab collaborations, particularly when one's expertise is frequently sought after. It suggests negotiating terms that reflect one's contributions if they are significant, rather than offering help for free. This approach is framed as a necessary step to ensure that one's own research time is not compromised, and to maintain a professional rather than familial relationship in a lab setting.

  - **[[D] How do i find endorsement to publish preprint on arxiv?](https://www.reddit.com/r/MachineLearning/comments/1q68ues/d_how_do_i_find_endorsement_to_publish_preprint/)** (Activity: 8): **The user is seeking guidance on obtaining an endorsement to submit a preprint to **arXiv**, which is a requirement for new submitters. Endorsements can typically be obtained from a current or previous university affiliation or through collaboration with a co-author who is already endorsed on arXiv. It is important to note that trading authorship solely for the purpose of obtaining an endorsement would violate academic integrity, as the co-author must genuinely contribute to the work.** A notable opinion suggests that collaborating with a co-author who can endorse the paper is a viable option, but emphasizes the importance of maintaining academic integrity by ensuring the co-author is a legitimate contributor.

    - The comment suggests obtaining an endorsement for arXiv preprint submission through affiliations with a current or previous university, or by collaborating with a co-author who can endorse. It emphasizes that trading authorship solely for endorsement violates academic integrity, highlighting the importance of genuine contribution from the co-author.

  - **[Usage update issue?](https://www.reddit.com/r/ClaudeCode/comments/1q6l5hn/usage_update_issue/)** (Activity: 202): **The image highlights a potential issue with the "Claude Code v2.0.76" software interface, specifically within the "Usage" tab. Users on a subscription plan, such as the $200 plan mentioned, are experiencing difficulties accessing their usage data, as the interface suggests that the "/usage" command is only available for subscription plans, yet it is not functioning as expected. Additionally, the option to enable extra usage is presented, but users are unable to verify their current usage status. This issue seems to be affecting multiple users, as indicated by the comments, and there is a related GitHub issue with significant discussion, suggesting a broader problem possibly linked to a recent usage spike after a promotional period.** One commenter notes that both the Claude Code and desktop app are experiencing this issue, and references a GitHub issue with extensive discussion, indicating a widespread problem. Another commenter dismisses the issue, suggesting everything is functioning correctly, while a third confirms experiencing the same problem.

    - There is a reported issue with usage spikes in Claude Code, particularly after a '2X week' event, which has led to a GitHub issue accumulating around 250 comments. This suggests a widespread problem affecting multiple users, with at least one person indicating they are investigating the issue. The problem seems to be related to unexpected usage limits and access changes.
    - Several users, including those on the '100 max plan' and '5x Max plan', are experiencing unexpected changes in their usage limits. One user noted that their limits were lifted prematurely, allowing them to use different models again despite having hit their weekly limit three days prior. This indicates a potential bug or misconfiguration in the usage tracking or limit enforcement system.
    - The issue appears to be affecting both the Claude Code and the desktop app, suggesting a broader systemic problem rather than an isolated incident. The fact that multiple users across different plans are reporting similar issues points to a possible backend or infrastructure-related problem that needs addressing.

  - **[https://claude.ai/settings/usage doesn't work?](https://www.reddit.com/r/ClaudeCode/comments/1q6l2x3/httpsclaudeaisettingsusage_doesnt_work/)** (Activity: 144): **Users are reporting issues with the **Claude AI usage settings page** (https://claude.ai/settings/usage), where it only displays the extra budget quota and not the expected usage details. Some users have noted that their usage limits have been unexpectedly lifted, allowing them to use different models despite having previously hit their weekly limits. This anomaly is occurring on the `5X Max plan`, and the reset was initially scheduled for the following day.** There is a suggestion from a user to "retire the 'usage limits'" altogether, indicating a preference for more flexible usage policies.

    - TheseQuit8175 reports an anomaly where their usage limits were unexpectedly lifted, allowing them to use different models despite having hit their weekly usage limits. They mention being on a '5X Max plan' and note that the reset was supposed to occur the following day, indicating a potential issue with the usage tracking system.
    - Gold_Jury_789 discusses a potential miscalculation in usage quotas, noting that at a '20x' usage level, they are exceeding their expected usage by 15% when they should be under 10%. They also mention an instance where they exceeded their quota by 35% on a Sunday, suggesting a possible bug or misconfiguration in the quota management system.



---

# AI Discord Recap

> A summary of Summaries of Summaries


## Gemini 3.0 Pro Preview Nov-18

**Theme 1. NousCoder-14b and the Open-Weights Coding Race**

- **NousCoder-14b Crushes Olympiads**: Nous Research released **NousCoder-14b**, a model post-trained on **Qwen3-14B** using the Atropos framework and 48 B200s, achieving a **67.87% Pass@1** accuracy (+7.08% over baseline) on competitive benchmarks ([announcement tweet](https://x.com/NousResearch/status/2008624474237923495)). The release includes a fully reproducible stack with the RL environment and benchmark detailed in their [blog post](https://nousresearch.com/nouscoder-14b-a-competitive-olympiad-programming-model/).
- **Qwen3’s Mixed Performance Reviews**: While some users argue **Alibaba’s QW** approaches **AGI** status in English, others report that **Qwen3** variants falter in complex creative writing compared to **Kimi K2** or **DeepSeek**. Additionally, users on OpenRouter noted a significant drop in **TPS** for **Qwen3-Next-80B**, likely due to routing through budget providers like GMICloud ([status update](https://x.com/openrouterai/status/2005707622020964412?s=46)).
- **Claude Code vs. Manual Workflows**: Engineers are debating the "proper" use of **Cursor IDE**, advocating for an **ETL** (Extract, Transform, Load) workflow using `.cursorignore` and `.mdc` files to optimize context. Meanwhile, users criticized the naming of **Claude Code**, demonstrating that **Claude Opus 4.5** can already automate complex tasks like generating a 30-second video ad from scratch ([demonstration tweet](https://x.com/deedydas/status/2008747553261842483?s=46)).

**Theme 2. Low-Level Kernels and Hardware Optimization**

- **NVFP4 Enters PyTorch**: Engineers successfully implemented **NVFP4** forward passes in **PyTorch** by patching layernorms to continuously convert between **nvfp4** and **bf16**, avoiding kernel fusion. Discussion highlighted that **NVFP4** remains proprietary to Nvidia, whereas **MXFP4** is the industry standard for FP4 training with hardware acceleration.
- **Visualizing High-Dimensional Tensors**: A shared [blog post](https://blog.ezyang.com/2025/10/draw-high-dimensional-tensors-as-a-matrix-of-matrices/) proposes drawing high-dimensional tensors as a "matrix of matrices" to overcome terminal display limitations. **GPU MODE** members are simultaneously hunting for tools to visualize binary formats like **f8** or specific low-bit layouts directly from memory.
- **Tinygrad vs. AMD Drivers**: Users debugging **tinygrad** on an **AMD Radeon RX 9070XT** reported `VFIO=1` triggering TypeErrors in `ioctl` calls, which resolve when disabled. The community is also chasing a bounty for replacing the scheduler with a **linearizer** to preserve GPU speed, with a potential fix already in PR ([PR link](https://github.com/tinygrad/tinygrad/pull/13780)).

**Theme 3. Model Evals, Leaderboards, and "Vibes"**

- **Gemini 3 Flash Hallucinates Hardware**: **Gemini 3 Flash** is being criticized for being "easily impressed," hallucinating **LFM 2.5** parameter counts ranging from 8B to 405B based on simple user prompts. Despite this instability, benchmarks suggest it outperforms **Gemini Pro** and **Grok 4 Heavy**, sparking debate over the value of **post-training** versus raw scale.
- **LMArena Battle Mode Backlash**: Users are revolting against the new **Battle Mode** in Direct Chat, citing lost context, aggressive **captchas** on every prompt, and inability to disable the feature. Critics argue the leaderboard has become a "plague," with **Video Arena** now confirmed to be strictly Battle Mode with random model pairs.
- **DeepSeek mHC Framework Skepticism**: A [paper discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1458191879127695391) suggests **DeepSeek's mHC framework**—which projects residual mappings onto doubly stochastic matrices—may be overhyped. Critics argue the real insight is that **residual mixing** is the actual unstable operator, rather than the novel projection framework presented in the paper.

**Theme 4. Security, Jailbreaks, and Privacy**

- **OpenRouter Hacked and IP Exposed**: A user reported their **OpenRouter** account was drained after a hack, prompting discussions about **IP exposure policies**, where some providers receive direct user IPs ([provider policy list](https://openrouter.ai/providers)). Security-conscious members are recommending throwaway Visa cards and strictly auditing provider routing.
- **Grok Developer Override Modes**: Red teamers are exploiting a **DEVELOPER OVERRIDE MODE** in **Grok** using security context injection to bypass filters, though the model often refuses with standard safety boilerplate. This aligns with the "informal crescendo attack method" discussion, aimed at extracting unrestricted outputs from **xAI** models.
- **ChatGPT Health Privacy Panic**: OpenAI launched **ChatGPT Health** to integrate medical records, but the [privacy policy](https://openai.com/index/introducing-chatgpt-health/) allowing research usage has raised alarms. The launch is contentious, with one study claiming **90% diagnostic accuracy** ([nature article](https://www.nature.com/articles/s41746-025-01543-z)) while others cite only **52.1%** ([counter-study](https://www.nature.com/articles/s41591-024-03097-1)).

**Theme 5. Infrastructure and Local Inference**

- **Pre-Quantized MoE Headaches**: **Unsloth** users report that pre-quantized **MoE models** are broken, forcing users to load full models and quantize to **4bit** on the fly. This limits deployment on consumer hardware, though a new [Supertonic CLI tool](https://huggingface.co/Supertone/supertonic-2) offers lossless compression for LoRA adapters to ease storage.
- **Vulkan Priority Woes**: Hardware enthusiasts are struggling with **Vulkan's** lack of priority splitting, which hampers the effective use of a **64GB MI210** alongside **24GB** cards. Users fear hitting the VRAM limit on smaller cards before utilizing the larger card's capacity, complicating multi-GPU local setups.
- **VS Code for Local LLMs**: A developer released a custom [VS Code build](https://github.com/bdrazn/codeOSS-LMStudio-Ollama/releases/tag/First-Light) optimized for local LLMs, featuring **LMStudio support** and a rewritten context management system. The tool claims to index and retrieve code faster than mainstream AI IDEs by focusing specifically on local inference constraints.


## gpt-5.2


**1. Open Models, RL Stacks, and New Benchmarks**

- **NousCoder-14B Brings Its Own Gym Bag**: Nous Research shipped **NousCoder-14b** with a full-stack release (the **RL environment**, **benchmark**, and **Atropos harness**) in their post ["NousCoder-14b: A competitive olympiad programming model"](https://nousresearch.com/nouscoder-14b-a-competitive-olympiad-programming-model/), post-training **Qwen3-14B** on **48 B200s for 4 days** and reporting **67.87% Pass@1** (**+7.08%** over Qwen) via *verifiable execution rewards* as announced in [@NousResearch on X](https://x.com/NousResearch/status/2008624474237923495).
  - Across multiple servers, people treated the release as notable partly because it ships the **training/eval plumbing** (not just weights), and it fed into broader discussions on **token efficiency** and how post-training can dominate perceived capability vs. raw scale.

- **Vision Arena Gets a New Top-10 Gatecrasher**: LMArena users highlighted that `ERNIE-5.0-Preview-1220` reached **#8** on the [Vision Arena leaderboard](https://lmarena.ai/leaderboard/vision) with a score of **1226**, with details tracked in the [Leaderboard Changelog](https://news.lmarena.ai/leaderboard-changelog/).
  - The community noted the meta-signal that **Baidu** is the *only* Chinese lab in Vision Arena’s Top 10 right now, while other conversations elsewhere questioned how much to trust Arena-style rankings at all.


**2. Health + LLMs: Product Launch Meets Accuracy and Privacy Math**

- **ChatGPT Health Wants Your Charts (and Your Trust)**: OpenAI launched **ChatGPT Health** as a dedicated space in ChatGPT to securely connect **medical records** and **wellness apps**, with early access via the ["Introducing ChatGPT Health" waitlist](https://openai.com/index/introducing-chatgpt-health/).
  - Discussion quickly centered on privacy/data-use implications of the policy language and the risk of ChatGPT becoming an *“everything app monopoly”*, even as some framed it as a practical layer to aggregate and verify personal medical info.

- **90% vs 52.1%: Healthcare Accuracy Cage Match**: Perplexity community debate cited a Nature study reporting **ChatGPT at 90% diagnostic accuracy** (["..." in *npj Digital Medicine*](https://www.nature.com/articles/s41746-025-01543-z)) versus another Nature paper reporting **52.1% accuracy** (["..." in *Nature Medicine*](https://www.nature.com/articles/s41591-024-03097-1)).
  - Engineers argued over whether this is mostly **dataset/task framing** vs real-world reliability, with repeated warnings that headline accuracy numbers can mislead when patient safety, triage thresholds, and deployment conditions shift.


**3. GPU & Kernel Tooling: FP4 Formats, Warp Specialization, and ROCm Catch-Up**

- **NVIDIA Cranks the RTX Knobs (Sampling, QKV, MXFP4)**: LM Studio users linked NVIDIA’s post ["Open-source AI tool upgrades speed up LLM and diffusion models on NVIDIA RTX PCs"](https://developer.nvidia.com/blog/open-source-ai-tool-upgrades-speed-up-llm-and-diffusion-models-on-nvidia-rtx-pcs/) describing **GPU token sampling**, **QKV projection concurrency**, **MMVQ kernel optimizations**, faster loading, and native **MXFP4** support on **Blackwell GPUs**.
  - The thread split between excitement for practical perf wins and skepticism that some of it reads like *marketing*, while adjacent chats compared **NVFP4 vs MXFP4** and complained about the lack of an IEEE FP4 standard.

- **NVFP4 Lands in PyTorch (But the TPS Isn’t Magic Yet)**: Hugging Face members reported **NVFP4** forward working in **PyTorch** by patching **layernorms** to continuously convert between **nvfp4** and **bf16** (without fused kernels).
  - Early testing noted unexpectedly lower **tokens/sec** with an **fp4 transformer engine**, with cautious optimism that fp4 inference can still be net-positive once kernel fusion/paths mature.

- **Warp Specialization Goes Brrr in CuTeDSL**: GPU MODE shared ["Warp Specialisation in CuTeDSL"](https://veitner.bearblog.dev/warp-specialisation-in-cutedsl/) showing a GEMM mainloop split into **TMA** (copy tiles to SMEM) and **MMA** (compute), using CuTeDSL pipelining to make **Blackwell mainloops** warp-specialized.
  - Separately, contributors reported **~30% throughput** gains integrating CuteDSL flex attention (vs base flex attention on **H100 fwd**) and tracked backend support gaps (e.g., **SM100 backward** supported while **SM90 backward** still in progress via [flash-attention PR #2137](https://github.com/Dao-AILab/flash-attention/pull/2137)).


**4. New Tooling for Fine-Tuning, Sharing, and Specialized Datasets**

- **Supertonic Shrinks Fine-Tunes by Shipping Only the Delta**: Unsloth community announced **Supertonic**, a free CLI tool on Hugging Face ([Supertone/supertonic-2](https://huggingface.co/Supertone/supertonic-2)) that computes the **delta between a fine-tuned model and its base** (LoRA-derived workflow) for **lossless compression after training**.
  - People framed it as a pragmatic way to distribute and version many fine-tunes without hauling full checkpoints, aligning with parallel interest in sparse/derivative adapter distribution tooling.

- **CyberSec CoT Dataset Tries to Patch the “Reasoning Gap”**: A member published **BlackBox-CyberSec-CoT-Reasoning-Sample** on Hugging Face ([dataset link](https://huggingface.co/datasets/blackboxanalytics/BlackBox-CyberSec-CoT-Reasoning-Sample)), generated with **Llama-3-70B** to produce Chain-of-Thought logs for **SOC incidents**.
  - They asked for feedback on whether the formatting is actually useful for fine-tuning **Llama-3**, echoing broader RL/SFT discussion that you should do *“bare minimum SFT needed to jump start RL”* when traces aren’t perfectly reliable.


**5. Reliability, Security, and Platform Friction in Model Gateways & UX**

- **OpenRouter: IP Policy Reality Check + Account-Drain Horror Story**: OpenRouter users pointed to the provider/IP disclosure list at [openrouter.ai/providers](https://openrouter.ai/providers), noting most providers see a **Cloudflare worker IP** but some may receive a user’s actual IP as described on model pages.
  - In the same period, one user reported an account takeover (email changed, card used to buy credits, prior data wiped), with advice to dispute charges and use **throwaway cards**—a reminder that model routing convenience comes with real operational risk.

- **Qwen TPS Tanks After Dec 28 and Everyone Plays “Blame the Router”**: OpenRouter users observed **TPS drops** for open-source models (notably **Qwen3-Next-80B-a3b-Instruct**) after **Dec 28**, referencing an update on the [OpenRouter Status X post](https://x.com/openrouterai/status/2005707622020964412).
  - The working theory blamed routing to the cheapest provider (**GMICloud**), and users recommended checking the **Activity tab** to compare provider speeds instead of assuming the model regressed.

- **Battle Mode + Captcha: LMArena Users Fight the UI Instead of Models**: LMArena users complained that **Battle Mode** kept injecting itself into **Direct Chat**, causing lost context and long generations, while frequent **captcha** prompts triggered even on low-volume use.
  - A team member said captcha targets *inauthentic use* and advised slowing prompts, but users pushed for an explicit disable switch and clearer rate-limit errors rather than surprise mode switches.


## gpt-5.1


**1. New High-Skill Models and Training Stacks**

- ****NousCoder-14B Cram-School Cracks Olympiad Problems****: **Nous Research** launched **NousCoder-14B**, a competitive olympiad programming model post-trained on **Qwen3-14B** with an RL stack built in **Atropos**, detailed in their blog post [“NousCoder-14b: A Competitive Olympiad Programming Model”](https://nousresearch.com/nouscoder-14b-a-competitive-olympiad-programming-model/). They trained on **48 B200s for 4 days**, achieving **67.87% Pass@1** on code benchmarks—a **+7.08%** jump over base Qwen—using verifiable execution rewards as highlighted in their [X post](https://x.com/NousResearch/status/2008624474237923495).
  - Discussion across **Nous** and **Latent Space** emphasized that the release includes a full-stack, reproducible RL environment, benchmark, and harness, making it a rare public end-to-end code-RL stack rather than just a model drop. Engineers called out that this kind of open, verifiable training pipeline makes it much easier to compare RL reward schemes and push beyond simple pass@k leaderboards.

- ****Unsloth Turns Nemos into Tiny Opus-Style Thinkers****: A community member used **Unsloth** to convert **Mistral-Nemo-Instruct-2407 12B** into a reasoning-focused model, aligning it with **Claude Opus high-reasoning traces**, and released it as [“Mistral-Nemo-Instruct-2407-12B-Thinking-HI-Claude-Opus-High-Reasoning”](https://huggingface.co/DavidAU/Mistral-Nemo-Instruct-2407-12B-Thinking-HI-Claude-Opus-High-Reasoning). They also published a **Heretic Uncensored** variant, [“MN-CaptainErisNebula-Chimera-v1.1-THINKING-ClaudeOpus4.5-12B-heretic-uncensored”](https://huggingface.co/DavidAU/MN-CaptainErisNebula-Chimera-v1.1-THINKING-ClaudeOpus4.5-12B-heretic-uncensored), after running a heretic process and then adding Opus-style thinking heads.
  - The author reported that the **Claude Opus 4/4.5** reasoning dataset yields *compact but high-quality* thinking blocks, effectively turning a creative 12B into a multi-step reasoner without massive scale. Other users in **Unsloth’s showcase** channel are now treating these as templates for *“thinking-conversion”* of mid-size models, indicating a trend toward adding distilled reasoning to strong-but-small backbones.

- ****Diffusion LLMs Delight Researchers Despite Thin Details****: In **Nous Research’s** research-papers channel, a member shared the paper [“Diffusion LLMs”](https://arxiv.org/abs/2511.08923), saying they like diffusion-style language models because *they seem more fun*. The paper proposes using diffusion-like generative processes for text, contrasting with standard autoregressive transformers.
  - While the technical discussion was brief, the link and reaction show growing curiosity about **non-autoregressive, diffusion-based LMs** as a serious alternative for future scaling. Engineers indicated they want to understand whether these architectures can offer better **mode coverage, parallelism, or controllability** than current transformer decoders.


**2. GPU Systems, Kernels, and Low-Level Performance Tuning**

- ****CuTeDSL Warp-Specialization Turbocharges GEMMs****: In **GPU MODE’s nvidia-competition** channel, a member shared the blog post [“Warp Specialisation in CuTeDSL”](https://veitner.bearblog.dev/warp-specialisation-in-cutedsl/), which splits the GEMM mainloop into **TMA (tile copy to SMEM)** and **MMA (matrix multiply)** using CuTeDSL’s pipelining abstraction. They reported turning ordinary non-persistent **Blackwell** mainloops into warp-specialized ones with meaningful throughput gains.
  - Separately, in **GPU MODE ▷ #torch**, another engineer thanked contributors for the **CuteDSL flex-attention** integration, citing **~30% throughput improvement on H100 forward** over base flex attention per the [flash-attention interface](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/interface.py#L938), and ongoing work to bring full backward support to **SM90** via a [pull request](https://github.com/Dao-AILab/flash-attention/pull/2137).

- ****NVFP4 and FP4 Engines Confuse Performance Expectations****: On the **HuggingFace** server, users confirmed **NVFP4** is working in **PyTorch** by patching layernorms to convert between **nvfp4** and **bf16**, but noted that kernels are not yet fused. Despite using **fp4 transformer engines**, they observed **lower tokens-per-second** than expected, raising questions about where FP4 actually wins in practice.
  - In **LM Studio** and related chats, this fed into a broader discussion comparing **NVFP4** to the more standardized **MXFP4**, with people pointing out there is no **IEEE FP4** standard and NVFP4 is a **NVIDIA-proprietary format**. Engineers concluded FP4 may still be more valuable for *inference* than training today, but only if kernel stacks and hardware paths are deeply tuned end to end.

- ****Helion, Iris, and ROCm Rally for Open GPU Systems****: The **GPU MODE ▷ #helion** channel announced that **Umesh from AMD** is actively enabling the **Helion** compiler stack on **ROCm**, auditing skipped unit tests and broken examples and focusing on **GEMM performance speedups**. Community members welcomed this and explicitly asked for **MI400-series** support as they ramp their AMD fleets.
  - In **GPU MODE ▷ #job-postings**, the [Iris project](https://github.com/ROCm/iris/), a **Triton-based multi-GPU programming framework**, advertised US internships for engineers with **Triton, multi-GPU programming, RMA/RDMA, and low-level GPU communication** experience. Together, these threads show a concerted push to make **AMD + Triton/Helion** a first-class, open alternative to CUDA for high-performance kernels.


**3. Hardware Economics, Performance Surprises, and Routing Headaches**

- ****Nvidia’s GPU Gold Rush Spurs Sticker-Shock Simulations****: In **LM Studio**, members dissected a [TrendForce report](https://www.trendforce.com/news/2026/01/05/news-nvidia-amd-reportedly-plan-price-hikes-starting-1q26-geforce-rtx-5090-may-reach-5000/) claiming the **GeForce RTX 5090** could reach **$5000** by 2026, alongside unaffordable **Nvidia DGX Station** racks and similar AMD systems. Engineers joked that future consumer cards with **128 GB VRAM** will just feed Chrome tabs using **32 GB each**, but the underlying concern is that state-of-the-art models still don’t comfortably fit even on upcoming **288 GB VRAM** datacenter parts.
  - In **Perplexity** and elsewhere, people connected this to rising **RAM prices** for local builds and speculated about Chinese and Indian manufacturers (e.g. **Tata**, **Reliance**) entering DRAM to ease costs. The consensus is that hardware scarcity and vendor pricing, not just algorithms, are now gating who can meaningfully participate in frontier-scale training.

- ****GB10 GPU and Qwen Routing Reveal Real-World Performance Pitfalls****: On **LM Studio’s hardware-discussion**, testers called the **GB10** GPU *“way too slow”*, measuring it at about **6× slower** than an **RTX 6000 Ada** despite having more memory, and compared it to NVIDIA’s [DGX Spark](https://www.theregister.com/2026/01/05/nvidia_dgx_spark_speed/) which showed similarly lackluster throughput. Users also warned that hallucinations remain a **model issue, not a VRAM issue**, pushing back on the notion that more memory fixes model quality.
  - Over on **OpenRouter**, users saw **TPS for Qwen3-Next-80B-a3b-Instruct** drop sharply after **Dec 28**, something OpenRouter acknowledged on their [status post on X](https://x.com/openrouterai/status/2005707622020964412). The slowdown appears tied to routing through the cheapest provider (**GMICloud**), and engineers were advised to inspect the **Activity tab** and explicitly select faster providers instead of trusting automatic routing.

- ****RAM, VRAM, and Vulkan Priorities Block Upgrade Paths****: In **LM Studio**, one member trying to scale beyond **24 GB** cards debated between **2×24 GB**, **48 GB**, **64 GB MI210**, or a **4090**, calling all options *“only wrong moves left on the table”* given cost vs. capacity. Another reported large-scale instability on **Intel 13th-gen CPUs**, advising others to check **Windows Event Manager** for issues potentially triggered by recent Windows updates.
  - They also flagged **Vulkan’s lack of priority splitting** as a practical issue when mixing a **64 GB MI210** with multiple **24 GB** cards, since the small cards can become the bottleneck before the big one is saturated. Across **Perplexity** and hardware channels, similar resource constraints—high DRAM prices, scarce large-VRAM cards, and immature scheduling APIs—are shaping how aggressively teams can push local and hybrid deployments.


**4. AI in Healthcare and Privacy: Power vs. Risk**

- ****ChatGPT Health Hustles into Medical Workflows, Privacy in Tow****: **OpenAI** announced **ChatGPT Health**, a dedicated health space in ChatGPT that lets users securely connect **medical records** and **wellness apps** to help them *“navigate medical care”*, as described in their blogpost [“Introducing ChatGPT Health”](https://openai.com/index/introducing-chatgpt-health/). The tool explicitly states it does **not replace professional medical advice** and instead grounds responses in personal health data, with a [waitlist for early access](https://openai.com/index/introducing-chatgpt-health/).
  - Across **OpenAI**, **Yannick Kilcher**, and **Latent Space** discords, engineers immediately raised **privacy and lock-in concerns**, noting the policy allows using health conversations to *improve services and conduct research*. Some worried this could turn ChatGPT into an *“everything app monopoly”* for health, especially compared to Google’s open-sourcing of **MedGemma**, and debated whether such sensitive data should ever feed back into model training.

- ****LLM Diagnosis Studies Split the Stats, Split the Room****: In **Perplexity AI**, users shared a **Nature Digital Medicine** study where **ChatGPT** reached about **90% diagnostic accuracy** in a controlled setting, linking to [“Evaluating ChatGPT’s diagnostic performance”](https://www.nature.com/articles/s41746-025-01543-z). Others countered with a second **Nature Medicine** paper, [“Performance of large language models in clinical diagnosis”](https://www.nature.com/articles/s41591-024-03097-1), showing only **52.1% accuracy** and warning about patient-safety risks.
  - The debate crystallized into a shared view that **benchmark cherry-picking** can wildly misrepresent clinical safety, and that LLMs must be treated as decision support, not autonomous diagnosticians. Engineers emphasized the need for **rigorous, task-specific evals**, bias audits, and clear guardrails, especially as tools like **ChatGPT Health** start ingesting real medical records.

- ****Cybersecurity ‘Reasoning Gap’ Tackled with Llama-3 CoT Logs****: In **Unsloth’s research** channel, a practitioner working on SOC tooling described a *“Reasoning Gap”* in off-the-shelf cyber models and started using **Llama-3-70B** to generate structured **Chain-of-Thought incident logs**. They released a public sample dataset, [“BlackBox-CyberSec-CoT-Reasoning-Sample”](https://huggingface.co/datasets/blackboxanalytics/BlackBox-CyberSec-CoT-Reasoning-Sample), to solicit feedback on formatting for fine-tuning.
  - The community conversation revolved around whether explicit **CoT traces** materially improve cyber incident triage versus just better data and tools, with several people stressing annotation quality over ever-more-complicated RL setups. This mirrors earlier comments in Unsloth and Eleuther that **high-quality, task-specific data (e.g., Ultra-FineWeb and [sumthink](https://huggingface.co/datasets/G-reen/sumthink)) often beats clever algorithm tweaks** when trying to close domain-specific reasoning gaps.


**5. Creative Companions, Vision Tools, and Evaluation Backlash**

- ****Voice and Visual Companions Hit Real Usage Numbers****: In **Latent Space’s genmedia channel**, Paula from **Tolan** announced their **voice-first AI companion** hit **200,000 monthly users**, sharing implementation details and lessons in an [X thread](https://x.com/paularambles/status/2008964509810278413?s=46). At the same time, **Razer** teased [“Project AVA”](https://xcancel.com/razer/status/2008543615916666928?s=46), a **5.5-inch-screen AI companion** with advanced reasoning and personalized, skinnable avatars (from esports legends to anime-style characters) slated for **CES 2026**.
  - Engineers viewed both as signals that **real-time, persistent AI companions** are leaving the demo stage and becoming consumer products with serious infrastructure and latency constraints. There was interest in how these systems orchestrate **multi-modal input, streaming TTS, and memory**, and how tightly they’re coupled to providers like **OpenAI**, which Tolan explicitly credits as a close collaborator.

- ****Camera-Control LoRAs Give Artists Director-Level Powers****: Fal released an open-source, stronger **multi-angle camera control LoRA** for **Qwen-Image-Edit-2511**, documented in their [announcement thread](https://xcancel.com/fal/status/2008954582018248755?s=20). The LoRA lets users specify perspective and framing—**front/back/side views, low/high angles, and different shot distances**—to re-compose existing images with precise camera control.
  - Creators in **Latent Space** saw this as a big step toward **promptable cinematography**, where you can separate *content* from *camera* in image editing workflows. Combined with earlier work on ad-generation (e.g., Deedy’s [“Claude Code” Hermès-style 30s ad](https://x.com/deedydas/status/2008747553261842483?s=46)), the consensus is that tooling is rapidly approaching full **script → storyboard → shot-level control** for small teams.

- ****Arena Leaderboards and Benchmarks Catch Community Side-Eye****: **LMArena** updated its **Vision Arena leaderboard**, pushing `ERNIE-5.0-Preview-1220` to **#8 with a score of 1226**, as logged in the [vision leaderboard](https://lmarena.ai/leaderboard/vision) and [changelog](https://news.lmarena.ai/leaderboard-changelog/). Meanwhile, a widely-circulated critique, [“LM Arena is a plague on AI”](https://surgehq.ai/blog/lmarena-is-a-plague-on-a/), resurfaced across **Unsloth** and **Latent Space**, arguing that human-vote battles distort incentives now that top models are near human performance.
  - Several engineers shrugged off the rankings, with one saying *“I don’t know anyone that actually cares about lmarena rankings”*, and noted that many **Chinese models don’t even quote LM Arena scores** anymore. The community mood is shifting toward **task-grounded, reproducible evals** and away from vibe-heavy battle arenas, especially as companies lean on these numbers for marketing more than engineering.


## gpt-5


**1. New Coding Models & Vision Leaderboards**

- **NousCoder-14b Nails Olympiad Tasks**: **Nous Research** launched **NousCoder-14b**, a competitive olympiad programming model, detailing a full-stack release (RL environment, benchmark, Atropos harness) in [NousCoder-14b: A Competitive Olympiad Programming Model](https://nousresearch.com/nouscoder-14b-a-competitive-olympiad-programming-model/). The team reports post-training on **Qwen3-14B** using **48 B200s over 4 days** and achieving **Pass@1 67.87%** (+7.08% vs Qwen), with verifiable execution rewards.
  - They reaffirmed the results on X in [NousCoder-14b Pass@1 update](https://x.com/NousResearch/status/2008624474237923495), highlighting the **Atropos** harness and **Modal** autoscaler as core infra. Engineers praised the reproducible training stack and emphasized the importance of strong post-training signals for code tasks.

- **ERNIE-5.0 Climbs Vision Arena**: `ERNIE-5.0-Preview-1220` reached **#8** with a score of **1226** on the [Vision Arena leaderboard](https://lmarena.ai/leaderboard/vision), marking a notable placement among top vision models. The [Leaderboard Changelog](https://news.lmarena.ai/leaderboard-changelog/) notes **Baidu** as the only Chinese lab in the Top 10.
  - Community watchers flagged the climb as a signal of **ERNIE-5.0**'s visual reasoning maturity, urging more head-to-head evals. They also called out the Arena’s regular updates to track rapid platform-side changes that impact rankings.

- **Grok 5 Trains as xAI Banks Series E**: **xAI** announced its [Series E](https://x.ai/news/series-e) and confirmed **Grok 5** is in training, signaling continued scaling of its LLM suite. The update positions xAI for expanded compute and faster iteration on next-gen **Grok** models.
  - Engineers expect a step-change in capability if the training run completes at proposed scale, but they note that sustained evals across coding, reasoning, and safety will tell the real story. The community framed it as another data point in the accelerating **frontier-model** arms race.


**2. Kernel & Inference Speedups**

- **NVIDIA Speeds Up RTX AI Stacks**: **NVIDIA** detailed open-source AI tool upgrades to accelerate **LLMs** and **diffusion models** on **RTX PCs** in [Open-source AI tool upgrades speed up LLM and diffusion models on NVIDIA RTX PCs](https://developer.nvidia.com/blog/open-source-ai-tool-upgrades-speed-up-llm-and-diffusion-models-on-nvidia-rtx-pcs/). Highlights include **GPU token sampling**, **QKV concurrency**, **MMVQ kernel optimizations**, faster model loading, and native **MXFP4** support on **Blackwell GPUs**.
  - Developers expect measurable throughput gains in decode-heavy workloads and smoother model bring-up on consumer RTX rigs. Discussion centered on how much end-to-end latency these kernels shave off once integrated into real apps.

- **Warp Wizardry: CuTeDSL Specializes the Mainloop**: A deep-dive on **warp specialization** split the **GEMM** mainloop into **TMA** (tiles to SMEM) and **MMA** (multiply) using CuTeDSL’s pipelining in [Warp Specialisation in CuTeDSL](https://veitner.bearblog.dev/warp-specialisation-in-cutedsl/). The technique converts non-persistent **Blackwell** mainloops into warp-specialized ones for better throughput.
  - Practitioners see this as a clean pattern to squeeze more perf without gnarly hand-rolled kernels. The write-up makes replication straightforward for engineers optimizing attention and matmul hot paths.

- **Flex Attention Gets Cut(e): H100 Sees Gains**: Engineers reported ~**30% throughput** improvements on **H100 forward** after integrating **CuTeDSL flex attention** (see [flash-attention interface reference](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/interface.py#L938) and the related [PR discussion](https://github.com/Dao-AILab/flash-attention/pull/2137)). Backward support differences across **SM100** vs **SM90** remain a focus area.
  - The community is standardizing on these interfaces to unlock speedups across masking modes without bespoke kernels. Ongoing work aims to close gaps in backward paths on older SMs while retaining forward wins.


**3. Finetuning, Compression & Retrieval Tooling**

- **Supertonic Shrinks Deltas, Shares Fine-tunes**: A free CLI, derived from **LoRA adapters**, released as [Supertone/supertonic-2](https://huggingface.co/Supertone/supertonic-2) introduces lossless compression after training by exporting the fine-tune delta vs the base model. This enables compact distribution and storage of **fine-tuned** checkpoints.
  - Builders praised the delta-based format for reproducibility and shareability across teams. They highlighted easier artifact management and faster model swaps in multi-experiment workflows.

- **Qdrant Mixes Signals with Hybrid Queries**: **Qdrant** documented composable **hybrid queries** in [Hybrid queries in Qdrant](https://qdrant.tech/documentation/concepts/hybrid-queries/), enabling combined vector, keyword, and metadata filtering. The concept targets retrieval scenarios that need multi-signal scoring at scale.
  - Practitioners cautioned that piling on features can degrade efficiency across storage, memory, compute, and latency. Teams recommended staged rollouts and profiling to justify each operator in production pipelines.

- **vLLM Plays by the Rules (On Demand)**: vLLM’s structured decoding is documented in [Structured outputs in vLLM](https://docs.vllm.ai/en/v0.8.2/features/structured_outputs.html), supporting schema-constrained generations. Power users want conditional constraints that kick in after a sentinel (e.g., `</think>`) so the model can ‘think’ first.
  - Engineers are experimenting with delayed constraint hooks and KV-caching patterns to preserve reasoning then enforce schemas. The goal is to balance correctness guarantees with high-quality intermediate deliberation.


**4. LLMs in Healthcare: Product & Proof**

- **ChatGPT Health Connects Data, Clarifies Scope**: **OpenAI** launched **ChatGPT Health**, a dedicated space to securely connect **medical records** and **wellness apps**, as detailed in [Introducing ChatGPT Health](https://openai.com/index/introducing-chatgpt-health/). The product grounds responses in personal health data and explicitly states it is *“not to replace professional medical advice.”*
  - Engineers discussed HIPAA-adjacent integrations, auditability, and consent flows as must-haves for serious adoption. Privacy concerns sparked calls for clear data retention policies and sandboxed evaluation environments.

- **Study Says: ChatGPT Hits 90% Diagnostic Accuracy**: A recent **Nature** paper reported **ChatGPT** achieving **90%** diagnostic accuracy in controlled settings: [Nature study on ChatGPT diagnostic accuracy](https://www.nature.com/articles/s41746-025-01543-z). The result reignited debate on where LLMs can augment triage and decision support.
  - Clinically minded engineers urged careful external validation, dataset transparency, and robust calibration. They stressed that deployment context, prompt design, and guardrails can swing outcomes dramatically.

- **Counterpoint: Another Study Finds 52.1% Accuracy**: A separate **Nature Medicine** study reported **52.1%** accuracy, underscoring risk and variability: [Nature Medicine study on LLM diagnostic accuracy](https://www.nature.com/articles/s41591-024-03097-1). The work highlights gaps vs expert physicians and potential safety issues.
  - Teams advocated rigorous **A/B testing**, adverse-event tracking, and human-in-the-loop review before clinical use. The community framed randomized trials and post-deployment monitoring as essential steps.


**5. Infra Reliability, Pricing & Security**

- **RTX 5090 Price Panic: $5k on the Horizon?**: A [TrendForce report on GPU pricing](https://www.trendforce.com/news/2026/01/05/news-nvidia-amd-reportedly-plan-price-hikes-starting-1q26-geforce-rtx-5090-may-reach-5000/) suggests the **GeForce RTX 5090** could hit **$5000** in 2026. Engineers weighed cost trade-offs for local inference vs. hosted APIs amid rising silicon prices.
  - Builders expect heavier reliance on shared clusters and spot capacity for training/finetuning. Discussions also considered mixed fleets (consumer + data center GPUs) to balance **VRAM** needs and throughput.

- **Holiday Hangover: Qwen TPS Slows on OpenRouter**: Users observed TPS drops for open-source models like **Qwen3-Next-80B-a3b-Instruct** after Dec 28, per [OpenRouter status update (X)](https://x.com/openrouterai/status/2005707622020964412). Reports attribute slowdowns to routing through cheaper providers, with guidance to compare provider speeds in the **Activity** tab.
  - Practitioners recommended pinning faster providers for latency-sensitive workloads. Teams also suggested tracking per-provider P50/P95 latencies to avoid regressions during peak demand.

- **Account Hijack Spurs OpSec and IP Scrutiny**: An **OpenRouter** user reported account compromise, credit card abuse, and wiped data; members referenced [OpenRouter providers & IP policies](https://openrouter.ai/providers) showing that some providers receive the user’s actual IP. The incident prompted reminders to rotate keys, enable 2FA, and monitor billing.
  - Security-conscious users recommended virtual/throwaway cards and least-privilege API usage. They also advised auditing model pages for IP handling details before selecting providers.


---

# Discord: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Pre-Quantized Models Plagued with Problems**: Members report issues loading pre-quantized **MoE models**, requiring full model loading and on-the-fly quantization to **4bit**.
   - The community suggests exploring alternative quantization methods or waiting for updated libraries that better support pre-quantized MoE models.
- **Gemini 3 Flash Overestimates Model Size**: **Gemini 3 Flash** estimated **LFM 2.5 1.2B's** parameter count to be between 8B and 405B after simple prompts, showing inconsistency.
   - Members agree that these results highlight the limitations of **Gemini 3 Flash** as a reliable judge of model capabilities.
- **Free Lossless Finetuning Tool Drops**: A member released a free CLI tool, derived from **Lora adapters**, that has lossless compression after training, called [Supertonic](https://huggingface.co/Supertone/supertonic-2).
   - The tool calculates the delta between the fine-tuned and base model, making it easier to share and store, and reducing the file size.
- **Qdrant Touts Hybrid Query Capability**: A member shared [Qdrant's documentation](https://qdrant.tech/documentation/concepts/hybrid-queries/) on **hybrid queries**, emphasizing the ability to combine query types.
   - They cautioned that adding more features might lead to decreasing efficiency in storage, memory, computation, and latency.
- **Reasoning Gap Gets Cybernetic**: A member is tackling the *Reasoning Gap* in cybersecurity models using **Llama-3-70B** to produce Chain-of-Thought logs for SOC incidents, and posted a [Free Sample to Hugging Face](https://huggingface.co/datasets/blackboxanalytics/BlackBox-CyberSec-CoT-Reasoning-Sample).
   - They seek community feedback on the formatting's usefulness for fine-tuning **Llama-3**.



---



## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Invidious Blocks Trackers, Advocates Rejoice**: A member recommended using [Invidious](https://redirect.invidious.io/) as a **YouTube** frontend to block trackers, highlighting concerns about **YouTube's data collection** practices.
   - The member asserted that *YouTube* has 7 trackers on a single param, but that Invidious blocks all of them.
- **Grok and Gemini face Jailbreak Attempts**: Members are actively seeking working jailbreaks for **Grok** and **Gemini**, with one user mentioning the *informal crescendo attack method* on **Grok**, while another inquired about a jailbreak for **Bing**.
   - A user shared a **DEVELOPER OVERRIDE MODE** for **Grok**, featuring security context injection to bypass safety layers and content filters, aiming for unrestricted output, and a similar override for **Gemini** known as **Gemini-DEV-OVERRIDE-2026**.
- **AI Learns Swahili, Gains Red Team Edge**: A member responded that the purpose of AI red teaming is to expose weaknesses and suggested trying prompts in less resourced languages, like Swahili or Navajo.
   - They said, *'the llm struggles and even weaker guardrails'* in languages like Swahili or Navajo, leading to the realization that it performs weaker for users interacting in Russian, deeming it unfair to other participants.
- **Microsoft Engineer Vibe Codes Notepad App Using AI**: A member shared a [YouTube video](https://youtu.be/bmBd39OwvWg) of a **Microsoft engineer** vibe coding a new **Notepad app** using **Claude**, with the quote *it was AI that ruined notepad so its gonna be AI that fixes it*.
   - Another member sarcastically said *bro said vibe coding as a exec*.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Battle Mode Bugs Users**: Users reported frustration with the introduction of **Battle Mode** into **Direct Chat**, citing issues like lost context and long generation times and requested a way to disable it.
   - Some users reported that *every second message turns into battle mode* and that they were experiencing constant interruptions.
- **Captcha's Catching Prompting Capriciousness**: Users reported frequent **captcha** prompts even when prompting infrequently, with some reporting getting a captcha *every prompt*.
   - A team member stated that the **captcha** system is designed to detect inauthentic use and suggested users slow down their prompting frequency.
- **Movement Labs Model: Miracle or Mirage?**: Members debated the **Movement Labs AI model**, with some praising its ability to generate functional code for a **Minecraft clone** and a **chess game**.
   - Others alleged it to be a *scam* citing past controversies and drama relating to a chargeback fraud scheme, implying caution.
- **ERNIE-5.0 Earns Top Ranking**: `ERNIE-5.0-Preview-1220` reached **#8** on the [Vision Arena leaderboard](https://lmarena.ai/leaderboard/vision) with a score of **1226**.
   - The [Leaderboard Changelog](https://news.lmarena.ai/leaderboard-changelog/) notes that **Baidu** is the only Chinese lab featured in the Top 10.
- **Video Arena Experiments: Now You See It...**: Members inquired about the status of **Video Arena** on the website, noting the **video button** appeared and disappeared.
   - A team member clarified that **Video Arena** on the site is experimental, its availability is random, and confirmed that **Video Arena** will be Battle mode only with 2 random models.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Nvidia's AI Racks Spur Racketeering**: Members find **Nvidia's AI racks** and **AMD's AI racks** unaffordable for individual users, but still indirectly usable, with discussion on the [Nvidia DGX Station](https://www.nvidia.com/en-us/products/workstations/dgx-station/).
   - They cited an article from [TrendForce article](https://www.trendforce.com/news/2026/01/05/news-nvidia-amd-reportedly-plan-price-hikes-starting-1q26-geforce-rtx-5090-may-reach-5000/) that suggested the **GeForce RTX 5090** may reach $5000 by 2026.
- **Nvidia Open Source AI Tool Gets Updates**: **Nvidia** announced updates to its open-source AI tools, improving performance for **LLMs** and **diffusion models** on **RTX PCs**, with details in an [Nvidia blog post](https://developer.nvidia.com/blog/open-source-ai-tool-upgrades-speed-up-llm-and-diffusion-models-on-nvidia-rtx-pcs/).
   - Updates feature **GPU token sampling**, **concurrency for QKV projections**, **MMVQ kernel optimizations**, faster model loading, and native **MXFP4** support on **Blackwell GPUs**.
- **Local IDEs, Not Zed**: Members discuss IDE alternatives that work with local models, recommending **kilocode**, **roocode**, and **cline**, after issues with **Zed IDE**.
   - This came about after one user described **Zed IDE** as a *clusterf-*.
- **GB10, a GPU disappointment**: Members described the **GB10** as *way too slow* in testing, despite having a lot of memory, at **6x slower** than an **RTX Pro 6000**.
   - Further discussion pointed to the [Spark](https://www.theregister.com/2026/01/05/nvidia_dgx_spark_speed/) as another device with similar performance characteristics.
- **Vulkan's Priority Woes**: Members debate the lack of **priority splitting** in **Vulkan**, a concern for effectively using a **64GB MI210** alongside existing **24GB cards**.
   - The worry is that the **24GB limit** on other cards could be hit before the **48-64GB card** is fully utilized.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT Gets Healthy with New Medical Integration**: **OpenAI** launched **ChatGPT Health**, a dedicated space in **ChatGPT** where users can securely connect **medical records** and **wellness apps**, aiming to help users *navigate medical care* and invites interested users to [join the waitlist for early access](https://openai.com/index/introducing-chatgpt-health/).
   - The tool is explicitly stated *not to replace professional medical advice* and is designed to ensure responses are grounded in personal health information.
- **ElevenLabs Clones Voices, Except the Politically Charged**: Users on the channel reported that [ElevenLabs](https://elevenlabs.io/) let them clone almost any voice except *sensitive political voices*.
   - One user had previously used the platform to make a *ghost obi-wan, a HAL9000, aubrey plaza* voices, however lost 1 million banked credits when they stopped paying.
- **Bypassing OpenAI bans is Risky Business**: Users in Australia can circumvent **OpenAI** bans to access **Sora 2** from a third party such as [ElevenLabs](https://elevenlabs.io/).
   - Some members are not concerned about account bans, others warned that using a VPN to evade geographic restrictions can lead to your **OpenAI** account being banned.
- **Ethical Framework A/B Testing, a Prompt Engineer's Friend**: A member advocated for de-mystifying an ethical framework in **LLMs** by using **A/B testing and ablations** to identify operational components and promote transparency.
   - They argued that there are likely many linguistic and structural equivalents to the current mystical prompt that could equal or exceed its performance, emphasizing the need for **A/B testing at scale** in prompt engineering to identify effective and transparent prompts, because **AI failure modes** often emerge only at scale.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Gemini 3 Pro's File Generation Fumbles**: Users reported inconsistent file generation with **Gemini 3 Pro**, sometimes failing with the message: *“I cannot generate a direct downloadable file.”*
   - Some members are *“loving the bugs!”* while others are experiencing issues with **Perplexity Pro** subscriptions.
- **Perplexity Pro Pauses Spark Panic**: Multiple users reported their **Perplexity Pro** subscriptions being unexpectedly paused, needing a payment method even with promotional subscriptions like **Airtel**.
   - A [gadgets360.com article](https://www.gadgets360.com/ai/news/how-to-keep-your-free-perplexity-pro-on-airtel-new-card-requirement-explained-9870744) explains the new card requirement for **Airtel** users.
- **LLMs' Healthcare Hijinks Heat Up**: A debate ignited around **LLMs** in healthcare after a user shared [research](https://www.nature.com/articles/s41746-025-01543-z) indicating **ChatGPT** achieved **90%** diagnostic accuracy in a study.
   - However, other members cautioned against relying on **LLMs** for healthcare, citing [another study](https://www.nature.com/articles/s41591-024-03097-1) with only **52.1%** accuracy and highlighting potential risks to patient safety.
- **RAM Prices Ramp Up Frustration**: Members discussed how high **RAM** prices are impacting computer build plans, with one user suggesting Chinese manufacturers could lower costs.
   - The discussion touched on Indian companies like **Tata** and **Reliance** potentially entering **RAM** manufacturing, which could reduce prices in the future.
- **Sonar Models Shun AWS Presigned URLs**: A member reported that while standard public image URLs work perfectly with **sonar models**, **AWS Presigned URLs** consistently result in a **400 error**.
   - They asked if sending images as **Base64** encoded strings is the only recommended workaround.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Debating Proper Use of Cursor IDE**: Users debated the definition of *"proper use"* of **Cursor IDE**, with one user suggesting it's not proper use if you *"just prompt and rely on the output".
   - Another user countered that their **personal opinions** and **experiences** with different models are valid, even if not universally applicable, leading to a discussion about the value and potential for misleading claims without sufficient experience.
- **Sharing ETL-Based Cursor Workflow**: Members discussed their workflows with Cursor, focusing on an **ETL** (*Extract, Transform, Load*) approach, sharing ways to improve their existing workflows using Cursor IDE.
   - One member mentioned using `.cursorignore`, `.cursorindexingignore`, and `.mdc` files for improved results, while another found that **Plan mode** drastically increased efficiency, replacing a more complex prior workflow.
- **Fixing remote SSH host slowness with ripgrep commands**: A member reported issues with Cursor on a remote SSH host due to `rg` commands running against a large NFS folder, and they discovered **`--no-ignore`** flag prevents ignoring files.
   - They shared a workaround to resolve slowness by [creating a shell script to modify the rg command](https://github.com/BurntSushi/ripgrep/pull/3212).
- **Requesting semantic code reviews**: A member requested a feature for high-level semantic code reviews, with control over the model used and filed a [feature request](https://forum.cursor.com/t/local-high-level-semantic-code-reviews-not-only-syntax/148187).
   - Another member suggested creating a *"code-reviewer" subagent* for more control and customization.
- **Users reporting losing agent chats**: Users reported a bug where opening a folder in an empty Cursor window opens a new window and they lose that agent chat.
   - Another user experiences frequent crashes when making large edits, resulting in wasted money due to the tool getting stuck on *"planning next moves"*.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Nvidia's Vision Model Disappoints**: Members found [Nvidia's Nemotron-Nano-12B-v2-VL vision model](https://developer.nvidia.com/nemotron) underwhelming compared to **Qwen3-VL-8b-Instruct** or **GLM-4.1V-9B**.
   - One user, testing it on [GrokifyPrompt.com](https://www.grokifyprompt.com/), reported it only recreated photos with reasonable accuracy.
- **OpenRouter's IP Exposure**: A discussion addressed concerns about **OpenRouter** exposing user IPs to providers, referencing a list of [providers and their IP policies](https://openrouter.ai/providers).
   - Most providers receive a **Cloudflare worker IP**, but some may obtain the user's actual IP, as detailed on each model's page.
- **Hacker Drains OpenRouter Account**: A user reported their **OpenRouter account was hacked**, email changed, and credit card used to purchase credits after which all previous data was wiped.
   - Other members recommended contacting their credit card company to block the card and suggested using **throwaway Visa cards** for enhanced security.
- **Qwen TPS drops after holidays**: Users observed a significant decrease in **TPS (tokens per second)** for open-source models, especially **Qwen3-Next-80B-a3b-Instruct**, after December 28, see [OpenRouter Status page on X](https://x.com/openrouterai/status/2005707622020964412?s=46).
   - The slowdown may be due to routing through the cheapest provider (**GMICloud**), with users advised to check the **Activity tab** for provider speeds.
- **Discord Prepares for IPO**: **Discord Inc.** has confidentially filed for an initial public offering (**IPO**), with assistance from **Goldman Sachs Group Inc.** and **JPMorgan Chase & Co.**, per [Bloomberg report](https://www.bloomberg.com/news/articles/2026-01-06/chat-platform-discord-is-said-to-file-confidentially-for-ipo).
   - The chat app company, popular with gamers and programmers, boasts over **200 million monthly users**.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Betting Against Moore's Law Seen as Impractical**: Members believe betting against the end of **Moore's Law** is a *non-practical* bet.
   - The sentiment reflects skepticism about extending current technological scaling trends.
- **Enthusiasm Surges for Medical Scribe & Idea-Catcher Agents**: A member is actively creating agents, particularly a medical scribe and an idea catcher that automatically stores ideas with tags and categories.
   - They plan to test smaller models for speed improvements before upgrading equipment, optimizing for cost and efficiency.
- **Gemini 3 Flash Surpasses Gemini Pro in Some Tests**: **Gemini 3 Flash** is reportedly surprisingly performant, outperforming **Gemini Pro** on some benchmarks.
   - The strength of both **scale** and **post-training** are important, and that while scale and pre-training provide more raw intelligence, post-training significantly aids in solving tasks.
- **DeepSeek's mHC Framework Stability Claims Questioned**: DeepSeek's **mHC framework** aims to address instability in Hyper-Connections by projecting residual mappings onto doubly stochastic matrices; however, its value and novelty is debated.
   - Some suggest that *residual mixing* is the main insight, while others focus on sinkhorn or birkhoff polytopes as actually being significant.
- **ChatGPT Health Launches Amid Privacy Concerns**: **OpenAI** introduced **ChatGPT Health** ([link](https://openai.com/index/introducing-chatgpt-health/)), designed as a supplementary tool for aggregating medical information and verifying data.
   - Concerns were raised about user privacy and **ChatGPT** potentially becoming an *everything app monopoly*, especially given **Google** open-sourced their models with **MedGemma**.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **NousCoder-14b Competes in Olympiads**: Nous Research launched **NousCoder-14b**, detailed in [a blog post](https://nousresearch.com/nouscoder-14b-a-competitive-olympiad-programming-model/), along with a full stack release including the **RL environment**, **benchmark**, and **Atropos harness**.
   - Post-trained on **Qwen3-14B** using 48 B200s over 4 days, it achieves a **Pass@1 accuracy of 67.87%**, a **+7.08%** increase over Qwen, according to [this X/Twitter post](https://x.com/NousResearch/status/2008624474237923495).
- **Nvidia's Pricing Under Scrutiny**: Members anticipate **Nvidia** will price its new **GPU** expensively, despite its **288 GB VRAM** that can't fit SoTA models.
   - One member joked about future consumer GPUs with **128 GB** of RAM being consumed by Chrome tabs using **32 GB** each.
- **Grok Scales, Jensen Sweats**: A member suggested that **Elon's** scaling of **Grok-5** to **6-7T parameters** is making Jensen Huang nervous.
   - Another member noted that **Grok 4 Heavy** is now outperformed by **Gemini 3 Flash**, illustrating the rapid pace of AI development.
- **Transformer Architecture Suffice?**: Members debated whether **transformers** are sufficient for achieving **AGI**, and one suggested they may be close despite potential limitations for **ASI**.
   - Another member argued for the necessity of architectural innovation, particularly regarding **real-time learning** efficiency and **catastrophic forgetting**.
- **Token Efficiency Talk**: Members discussed **token efficiency**, explaining it as *how many tokens can you solve the problem* which really matters.
   - Lower token efficiency can indicate a weaker base model, potentially stemming from insufficient *research compute* or post-training.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **NVFP4 Swings into PyTorch**: **NVFP4** is now confirmed working in **PyTorch** by patching the **layernorms** to continuously convert between **nvfp4** and **bf16**, but without fusing kernels.
   - Members discussed the **tokens per second** (tps) performance was unexpectedly lower with an **fp4 transformer engine**, suggesting that inference with **fp4** might still be advantageous.
- **Fine-Tune Translation Reliability**: To improve **translation model reliability and accuracy**, members suggest preparing a large, encoded dataset and then running fine-tuning.
   - Applying the translation on the front layer could be implemented *quicker and cheaper*.
- **WebXOS unveils Temporal Graph Dynamics Dataset**: A member shared the [webxos/timelink_dataset_v1](https://huggingface.co/datasets/webxos/timelink_dataset_v1), which includes **time series and paired images of evolving graphs** for training models on temporal graph dynamics.
   - Generated with the **TIMELINK app**, the dataset features per-vertex/step generation metrics like energy and phase, capturing time series data of vertices, edges and size over time.
- **Agents Course Files MIA**: Members reported issues accessing files for the **Agents Course Unit 4 Project**, with error messages indicating that *no files are available*.
   - A specific URL ([https://agents-course-unit4-scoring.hf.space/files/7bd855d8-463d-4ed5-93ca-5fe35145f733.xlsx](https://agents-course-unit4-scoring.hf.space/files/7bd855d8-463d-4ed5-93ca-5fe35145f733.xlsx)) was referenced in the request.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **NousCoder-14b Smashes Programming Olympiad**: **Nous Research** launched **NousCoder-14b**, an olympiad programming model, post-trained on **Qwen3-14B**, achieving **67.87%** Pass@1 accuracy thanks to the Atropos framework and Modal's autoscaler as per [their tweet](https://x.com/NousResearch/status/2008624474237923495).
   - This model is designed to solve complex coding problems and represents a significant step forward in AI-driven code generation.
- **Razer Aims for AI Companionship at CES 2026**: Razer announced [Project AVA](https://xcancel.com/razer/status/2008543615916666928?s=46), an **AI companion** with advanced reasoning and personalization, and scheduled for release at **CES 2026**.
   - AVA will feature a **5.5-inch screen** and customizable character designs, including **esports legends and anime-inspired models**, hinting at Razer's ambition to blend AI with personal devices.
- **ChatGPT Health Privacy Policy Sparks Debate**: **OpenAI** launched **ChatGPT Health**, sparking discussion around its privacy policy, which permits using content to improve services and conduct research, as detailed on their [official blogpost](https://openai.com/index/introducing-chatgpt-health/).
   - The new health tool has raised questions about data usage and patient privacy within the AI healthcare sector, inciting varied opinions.
- **Open-Source Multi-Angle Camera Control LoRA Released**: Fal has released a more powerful, open-source version of the **multi-angle camera control LoRA** for **Qwen-Image-Edit-2511**, as per [this link](https://xcancel.com/fal/status/2008954582018248755?s=20).
   - This tool allows users to manipulate the camera perspective of images, including **front**, **back**, **side**, **low/high angles**, and various shot distances; offering greater control over visual content.
- **Tolan's Voice-First AI Companion Hits Milestone**: Paula from **Tolan** announces that their voice-first AI companion has reached **200,000 monthly users**, detailed further in [this X post](https://x.com/paularambles/status/2008964509810278413?s=46).
   - The project was developed in close collaboration with **OpenAI**, and the thread shares key takeaways from the development process, showing the rapid adoption of voice-based AI solutions.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Visualize High-Dimensional Tensors**: A member shared [a blog post](https://blog.ezyang.com/2025/10/draw-high-dimensional-tensors-as-a-matrix-of-matrices/) on visualizing **high-dimensional tensors** as a matrix of matrices, addressing challenges with terminal column and row limitations.
   - Another member is also looking for a tool to load a simple **binary format** (something like futhark arrays?) and then offers features like zooming, rotating, transposing, slicing, maybe different ways to visualize higher dimensional tensors etc.
- **Torch Kernel Kolloquy Kickstarts**: Members are diving into writing custom **CUDA** kernels with **PyTorch**, referencing optimized kernels in **Torch**/**Transformers** that are written in **C++** and stitched together in **Python**.
   - One member expressed interest in reading the **PyTorch** kernels after completing another project to understand how they work, emphasizing their love for open source and curiosity about **CUDA**, **MPI**, and **OpenMP** from an **HPC** perspective.
- **Iris Project Internships Beckon**: The [Iris project](https://github.com/ROCm/iris/), a **Triton-based multi-GPU programming framework**, has positions open for interns with experience in **Triton**, **multi-GPU programming**, **RMA/RDMA**, or **low-level GPU communication**.
   - The internships focus on **GPU systems, performance**, and **kernel development** and are US-based.
- **CuteDSL Gets Warp Specialization**: A user shared a blog post on [Warp Specialisation in CuTeDSL](https://veitner.bearblog.dev/warp-specialisation-in-cutedsl/) which splits the **GEMM mainloop** into **TMA** (copy tiles to SMEM) and **MMA** (multiply tiles).
   - This optimization uses **CuTeDSLs pipelining abstraction**, turning ordinary non-persistent **Blackwell mainloops** into warp-specialized ones.
- **Helion ROCm support welcomed by AMD Engineer**: Umesh from **AMD** will be working on enabling **Helion** on **ROCm** and identifying issues in skipped unit tests and examples in the Helion repository.
   - The member is inviting feedback on any immediate issues that need fixing and is also focusing on performance speedup on **GEMMs** in parallel.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Needs Training Wheels**: Mojo requires manual implementation of **backpropagation** using **MAX** due to the lack of a dedicated training library and lacks advanced **I/O** capabilities requiring custom data formats.
   - A user planning to build a tiny **LLM** over the weekend observed that *Mojo is going to break a lot of things very often and all of the docs currently assume you know some combination of Python + C++ or Rust*.
- **NuMojo v0.8.0 Has Landed**: The [NuMojo v0.8.0 update](https://forum.modular.com/t/numojo-v0-8-0-update-is-here/2579) introduces improvements and new features, inviting the community to provide feedback.
   - Discussions around "anonymous" sum types (e.g., `T1|T2`) in Mojo reveal concerns about usability, especially regarding generics and conditional conformances.
- **Error Handling Ergonomics in Limbo**: Mojo is exploring unifying error types with a single type containing a code, drawing inspiration from `errno`, to improve the ergonomics of `catch` for heterogeneous error types.
   - The potential use of error unions similar to Zig or sum types like Rust is being considered to enhance error handling.
- **Dict Iterator in Dire Straits**: A Mojo user seeks advice on the correct pattern for iterating over Dict entries while building nested structures for a native TOML parser, encountering issues with the `.items()` and `.keys()` iterators.
   - The user is creating nested Dict structures and reported issues with the `.items()` and `.keys()` iterators, noting that *'DictEntry is not subscriptable'*.
- **MAX Trails Behind TEI for Embeddings**: A member is switching from [TEI](https://github.com/huggingface/text-embeddings-inference) to **MAX** for embeddings and is experiencing significantly slower performance with *sentence-transformers/all-MiniLM-L6-v2*, **MAX** is yielding **727.1 embeddings/sec** compared to **TEI's 8000 embeddings/sec**.
   - Implementing *sentence-transformers/all-MiniLM-L6-v2* as a custom architecture might be the root cause of the performance dip or **MAX Serve** might be optimized for **LLM inference** rather than **embeddings**.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **vLLM Constrained Generation Tinkering**: Implementing constrained generation in **vLLM** requires tinkering with internals to allow the model to 'think' before constraints are applied, especially for conditional triggering of constraints after a `</think>` token, despite available [vLLM documentation on structured outputs](https://docs.vllm.ai/en/v0.8.2/features/structured_outputs.html).
   - Discussion centered on whether models with **235B+ parameters** inherently possess reasoning capabilities during token processing, potentially eliminating the need for explicit reasoning steps via interleaving reasoning with prompts e.g. `{chunk}{prompt==describe how to grade chunk}` and then caching KV (key value) pairs.
- **PDF Popularity in Common Crawl Unveiled**: Statistics indicate that **PDFs comprise only 0.6%** of the Common Crawl by file count, not file size, leading to questions about the inclusion of truncated PDFs in the calculation.
   - The clarification of the **0.6%** figure sparked debate on the implications for data analysis and the actual prevalence of PDFs in web-based datasets.
- **Kaggle & Colab VMs Offer Compute Credits**: Members suggested leveraging Kaggle/Colab VMs for smaller models due to compute constraints, noting that **Modal** and **Lium** offer around **$500** in compute credits suitable for approximately **100M** runs.
   - The discussion also highlighted that Kaggle's environment is particularly well-suited for specific use cases, providing a cost-effective alternative for certain computational tasks.
- **Sora's Ghibli Segment sparks Grave of the Fireflies Comparison**: Members are stating that *a lot of impressive looking things in Sora are just direct ripoffs of existing videos 'reskinned'*, with one member claiming a Ghibli style segment in **Sora** reminds them of a specific scene in **Grave of the Fireflies**.
   - No further commentary.
- **GPT-NeoX Attention Normalization Tweaks**: A member noted that **GPT-NeoX**'s default attention normalization behavior changed such that the old behavior normalized across all heads uniformly, while the new behavior normalizes only within each head.
   - The member further inquired about **LoRA/QLoRA** fine-tuning support, referencing existing scripts ([configs/finetuning_configs/6-9B.yml](https://github.com/EleutherAI/gpt-neox/blob/main/configs/finetuning_configs/6-9B.yml)) for full-parameter fine-tuning.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Customer Bails, Prompting Ability Blamed**: A customer expressed dissatisfaction with **Manus**, citing unresolved issues and a decision not to compensate for lost credits, claiming that they have moved to alternative platforms.
   - Another user suggested the customer keep trying with different models and to view Manus as *the shit*, to which the support team replied that they are looking into the specific reasons and that it might take some time.
- **Subscription Credit Policy Explained**: The support team clarified that monthly subscription credits need to be used within the subscription period, using the example that [$20 Pro membership gives 4000 monthly credits](https://manus.im/help/credits) to be used before the next month's reset.
   - They offered to further verify the user's specific subscription status and account details, *for example if you purchase a $20 Pro membership on January 1st, you will receive 4000 monthly credits, which need to be used before February 1st*.
- **Psychologist Suggests Smarter Manus Use**: A psychologist suggested focusing on discussions about what went wrong with **Manus** usage, referencing a [magnum opus](https://fwoxkyoz.manus.space/) to help users improve efficiency.
   - The psychologist mentioned iterative teaching of Manus via knowledge, where Manus remembers tasks and asks for confirmation before using credits.
- **HexStrike MCP Networking Snafu**: A user described an issue with an **AI security tool (HexStrike MCP)** hosted on a local virtual machine and an **AI client (Manus)**, explaining that the AI client could not properly resolve the hostname.
   - The user temporarily used **ngrok** to expose the local service through a public HTTPS endpoint, seeking to understand if moving the **MCP server to a VPS** with a public IPv4 address would resolve the connectivity issue and allow proper OAuth flow and SSE connection.
- **Community Eyes Open Source For Manus**: A member inquired about plans to **open source old parts of Manus** and contribute to new initiatives.
   - Another member suggested posting the issue in the **Manus Api Channel** for Ivan to review.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Emily Slop Account Surfaces**: A member dismissed **Emily** as a *slop account*, sparking a brief discussion about model release expectations before the Chinese New Year.
   - The discussion touched on the anticipation of new models and updates within the AI community.
- **Alibaba QW Approaches AGI Status?**: A member suggested that **Alibaba's QW** exhibits near-AGI capabilities in English, especially when contrasted with **DeepSeek** and **Kimi**.
   - The member questioned whether performance varies significantly between **DeepSeek** and **Kimi** when used in English versus Chinese.
- **Kimi K2 Dominates in Creative Tasks**: A member asserted that **Kimi K2** excels in both Chinese and English, topping the **EQ bench** [https://eqbench.com/](https://eqbench.com/) leaderboard.
   - They highlighted **Kimi K2's** superior creative writing and conversational abilities compared to other Chinese models.
- **Qwen's Performance Remains Underwhelming**: One member reported unsatisfactory experiences with **Qwen3** model variants.
   - While deemed *fine* for basic tasks, these models often falter in more complex conversations or creative writing scenarios.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Hotz's tinygrad Competitors in Focus**: A member inquired about the competitive advantages of **tinygrad** and the major internal challenges the development team faces, referencing [tinygrad's thesis](https://geohot.github.io//blog/jekyll/update/2025/07/06/can-tinygrad-win.html) and open-source discussions from weekly meetings.
   - The thesis outlines **tinygrad's** goals to be a minimal, educational, and hackable deep learning framework distinguished by its simplicity and focus on direct hardware control, aiming to outperform larger, more complex frameworks in specific use-cases.
- **Linearizer Bounty Still Up for Grabs?**: A member inquired about the status of the bounty for *Replacing scheduler with linearizer, preserving GPU speed*, even with a potentially ready [PR](https://github.com/tinygrad/tinygrad/pull/13780).
   - Community response suggests that submitting a functional PR may secure the bounty, with the possibility of **George Hotz** splitting the reward to encourage further contributions.
- **VFIO=1 throws TypeError on AMD Radeon RX 9070XT**: A user reported a **TypeError** when running `examples.benchmark_onnx` with `VFIO=1` on a Linux laptop using an **AMD Radeon RX 9070XT**, noting that it runs correctly when `VFIO=0`.
   - The error occurs due to a `NoneType` object not being callable within `tinygrad/runtime/support/c.py` during an `ioctl` call, further details are available in the [provided log](https://cdn.discordapp.com/attachments/1070745817025106080/1458632361758425098/tinygrad_vfio_no_iommu.log?ex=696058bf&is=695f073f&hm=156caa091597e59aaaf338b4e228a70a3d523b440e6d5ce6fb1e909cad59e138&).



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Decoding mTLS Magic for MCP**: A member is diving into **mTLS** implementations to boost **MCP's** interoperability within enterprise setups and is looking for the prime spots for contribution chats.
   - A suggestion popped up to hit up the <#1360835991749001368> channel, hinting that authentication groups might drop some knowledge on current related projects.
- **MCP Instructions Lost in the Docs**: A member raised an eyebrow about the lack of documentation for **MCP instructions**.
   - Another member chimed in, pointing out it's part of the [server's initialization response](https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle#initialization) and shared [a blog post](https://blog.modelcontextprotocol.io/posts/2025-11-03-using-server-instructions/) as the next best thing, and even fired up [an issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/2060) to get this stuff officially documented.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Snarky banter emerges**: A member posted *someone had to say it* and another person sarcastically replied *so brave* to the comment.
   - This sarcastic exchange may indicate strong feelings about a topic, though the topic itself remains unclear.
- **AlphaXiv link appears**: A member shared [an AlphaXiv paper](https://www.alphaxiv.org/abs/2601.01569).
   - The paper itself has not been discussed, so importance is unclear.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **DApp Dev Eyes Novel Ventures**: A developer experienced in **DAO**, **marketplace**, and **DApp** projects seeks to join a new initiative with a clear vision and lasting commitment.
   - They offer their expertise in governance, tooling, and usability, eager to contribute or brainstorm.
- **AI Engineer Aims to Streamline Model Pipelines**: An AI Engineer is offering their expertise in building real AI systems, including **training**, **fine-tuning models**, and integrating **retrieval, agents,** and **infrastructure** at scale.
   - They are ready to assist in streamlining model pipelines, productionizing LLM features, or optimizing inference cost strategies.



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





### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1458190479538520246)** (218 messages🔥🔥): 

> `Pre-quantized MoE models, Gemini 3 Flash as judge, Llama.cpp and Gemma multimodal, NVFP4 vs MXFP4, SFT and RL` 


- **Pre-Quantized Models cause Sadness**: Members are finding that loading pre-quantized **MoE models** has some problems and you can only load the full model and quantize it to **4bit** on the fly.
- **Gemini 3 Flash is Easily Impressed**: **Gemini 3 Flash** judged that **LFM 2.5 1.2B** was 8B to 70B parameters after some easy prompts.
   - It even guessed 405B in one case, which is *definitely not proof of anything besides the fact that Gemini 3 Flash is easily impressed*.
- **Llama.cpp left Hanging**: It is *disappointing that llama.cpp never got around to supporting **gemma 3n multimodal features***, especially since E4B isn't dumb.
   - This functionality *would be great for newer phones*, enabling a littlest omni model for phones.
- **NVFP4 vs MXFP4**: **NVFP4** is seemingly better than **MXFP4**, although MXFP4 is the industry standard for **FP4 training** with hardware acceleration.
   - There is no IEEE standardized version of FP4, and NVFP4 is a **Nvidia proprietary training method/format**.
- **SFT and RL, bare minimum SFT!**: A member stated to *do the bare minimum **SFT** needed to jump start **RL***, especially if your reasoning traces were generated by a 20B model so they are not the most accurate.
   - When facing instability in RL, the classic answer is that *it depends* and to *pray and experiment*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1458226012457533664)** (2 messages): 

> `Unsloth.AI Introductions, Community Member Welcome` 


- **Unsloth Welcomes First Member**: Unsloth.AI welcomed its first member, who greeted the channel with a simple *hi*.
- **Channel Introduction Begins**: The *introduce-yourself* channel saw its inaugural post, marking the start of community introductions.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1458190424786210971)** (374 messages🔥🔥): 

> `HP keyboard computer, Laptop battery life, GRPOTrainer tool calls, Sparse finetuning CLI, Multilingual Supertonic` 


- **HP squeezes PC into Keyboard**: A member was reminded of the [Google Japan GBoards](https://google.com) upon seeing that **HP** fit an entire computer in a keyboard.
   - They pondered if the **GBoardsOS** will affect laptop battery life.
- **Free Lossless Finetuning CLI tool drops**: A member announced a free cli tool, a derivative of **Lora adapters**, with lossless compression AFTER training, and posted a link to the [Supertonic repo](https://huggingface.co/Supertone/supertonic-2) on HuggingFace.
   - The tool takes the difference of the full fine-tune and base model and spits out the delta, making it easier to share and store.
- **Qdrant touts Hybrid Query**: A member shared a link to [Qdrant's documentation on hybrid queries](https://qdrant.tech/documentation/concepts/hybrid-queries/), noting the ability to mix and match query types.
   - They cautioned that layering on features leads to diminishing returns regarding storage, memory, compute, and latency.
- **Is LM Arena a Plague?**: Members discussed [this blogpost](https://surgehq.ai/blog/lmarena-is-a-plague-on-a/) and agreed that **LM Arena** has fallen off ever since models have approached general human perf in most domains.
   - They stated that Chinese models don't even show LM Arena scores nowadays.
- **Synthetic Media doom and gloom**: A member posted a [YouTube short](https://youtube.com/shorts/i92mJG3UpOU?si=VYL27G-JTQRa2f4k) on **synthetic TTS**, synthetic LLMs and Synthetic Generative AI, proclaiming *"It’s so over. It’s so over. It’s not dead internet theory. It’s dead world theory".*
   - Another member then posted [this Simpson's GIF](https://tenor.com/view/the-simpsons-homer-simpsons-end-of-the-world-end-is-near-gif-16593998) as a reaction.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1458420357743771720)** (43 messages🔥): 

> `Fine-tuning models for structured extraction from images, Parameter tuning for LoRA training, Deepseek OCR finetuning for markdown, Llama-server issues with Qwen3, GRPO reward functions` 


- **Hunting for Training Insights for Image Extraction**: A member sought resources to enhance model training for **structured extraction from images**, expressing uncertainty about optimizing parameters like **LoRA rank, weight_decay**, and **gradient_accumulation**.
   - Another member suggested focusing on **data quality/labeling**, advocating for **A/B testing** and referencing [Unsloth's LoRA hyperparameter guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide).
- **Taming Tax Totals in Invoices**: A member encountered issues with their model misinterpreting **tax totals** in invoices, particularly with varied tax applications across line items and character misreads (**5->S**).
   - It was pointed out that while LLMs aren't calculators, the data quality between differentiating **5** and **S** might need more data.
- **Decoding Deepseek OCR Markdown Mastery**: A member inquired about fine-tuning **Deepseek OCR** to extract **markdown**.
   - They were reminded of the importance of using specific channels for specific questions.
- **Llama-Server struggles with Qwen3**: A member reported issues using **llama-server** with **Qwen3-Next-80B-A3B-Thinking-GGUF**, encountering a **'////////'** response.
   - Suggestions included verifying the chat template and allowing time for community responses.
- **GRPO Rewards Ranting Randomly**: A member using **GRPO** noticed their model produced gibberish to lengthen the 'thinking process', despite reward functions not favoring longer lengths.
   - The member sought insights into this behavior, acknowledging their inexperience with GRPO, and whether the behavior is natural.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1458284019115626669)** (3 messages): 

> `Mistral Nemos, Claude Opus, Model Conversion, Heretic Uncensored` 


- **Mistral Nemos Models Transformed into Reasoning Powerhouses**: A user successfully converted **Mistral Nemos (12B)** into models capable of **thinking/reasoning** using Unsloth, testing if this improved performance and output, especially given Mistral Nemos' creative strengths.
   - The user mentioned that the **Claude Opus High Reasoning dataset** produces excellent, compact reasoning blocks, and thanked team Mradermacher for fast quants.
- **New Mistral Instruct Model Emerges**: The user released **Mistral Instruct 2407**, converted to **thinking/high-reasoning Claude Opus**, with extended output capabilities, and linked to a Hugging Face repo showcasing the model and its properties: [Mistral-Nemo-Instruct-2407-12B-Thinking-HI-Claude-Opus-High-Reasoning](https://huggingface.co/DavidAU/Mistral-Nemo-Instruct-2407-12B-Thinking-HI-Claude-Opus-High-Reasoning).
- **Heretic Uncensored Model Debuts**: A **Heretic Uncensored** model, named **MN-CaptainErisNebula-Chimera-v1.1**, was created by first applying a heretic process, then tuning and converting it to a thinking model akin to **Claude Opus 4.5**, detailed on its [Hugging Face page](https://huggingface.co/DavidAU/MN-CaptainErisNebula-Chimera-v1.1-THINKING-ClaudeOpus4.5-12B-heretic-uncensored).


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1458190269508882724)** (33 messages🔥): 

> `High Quality Training Data, RL hacking rewards, Reasoning gap in cybersecurity models` 


- **Generating High Quality Training Data Gains Traction**: A member inquired about researchers generating high quality training data, instead of focusing on algorithms or training parameters, and cited the [Ultra-FineWeb dataset](https://huggingface.co/datasets/openbmb/Ultra-FineWeb) as an example.
   - Another member pointed out that these datasets can be *plagued by efforts to benchmaxx model* and linked to another dataset called [sumthink](https://huggingface.co/datasets/G-reen/sumthink).
- **RL Hacks Rewards and Frustrates**: A member shared that **RL is blowing me away** every time, but then followed up saying *in a bad way lol. It's hacking the rewards*.
   - They found it fascinating to see that it can do something like this, so they will be *reading the docs* to learn more.
- **Bridge the Reasoning Gap with Llama-3 for Cybersec**: A member is *working on solving the 'Reasoning Gap' in cybersecurity models* using **Llama-3-70B** to generate Chain-of-Thought logs for SOC incidents.
   - They uploaded a [Free Sample to Hugging Face](https://huggingface.co/datasets/blackboxanalytics/BlackBox-CyberSec-CoT-Reasoning-Sample) to see if the formatting is useful for anyone fine-tuning Llama-3.


  

---


### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1458193718292250787)** (475 messages🔥🔥🔥): 

> `Anti-Cheat AI, Offline AI on Mobile, 128GB DDR5 RAM, Grok API Access, Invidious for Privacy` 


- **Privacy Advocates Prefer Invidious Frontend**: A member recommended using [Invidious](https://redirect.invidious.io/) as a **YouTube** frontend to block trackers, citing concerns over **YouTube's data collection** practices.
   - The member stated that *YouTube* has 7 trackers on a single param, but that Invidious blocks all of them.
- **GPT-5.2 faces Anti-Cheat AI**: Members discussed the future of **anti-cheat systems**, with AI scrutinizing every pixel for inhuman movement and its potential to make cheating less valuable.
   - One member said *when it becomes more affordable might mitigate it a lil for a time being but kinda like vm detection lmao (for example on running games with linux) theres so much complexity in not making your vm show up as a vm. even down to thermal readings from your cpu. a vm doesnt have real readings so they would have to fake them. even then they know the specs of your hardware you posted. what im saying, it gets to a point where cheating isnt valueable for the user anymore. therefore killing the market entirely.*
- **128GB DDR5 RAM Costs a Fortune**: The possibility of **128GB DDR5 RAM sticks** was discussed, with one member stating that they exist but cost around **$1.5k USD used per stick**.
   - Others humorously lamented the high cost, with one stating, *Only in my dreams.*
- **Grok API Access after Deprecation**: Members discussed options for retaining access to **Grok 3.0 mini's developer mode** and it was mentioned that xAI typically keeps the model available on their API for months after deprecation on the app.
   - It was explained that using the API involves selecting a frontend, funding an **xAI account**, and plugging the API key into the chosen frontend like [msty](https://msty.app/).
- **Microsoft Engineer Vibe Codes Notepad App Using AI**: A member shared a [YouTube video](https://youtu.be/bmBd39OwvWg) of a **Microsoft engineer** vibe coding a new **Notepad app** using **Claude**, with the quote *it was AI that ruined notepad so its gonna be AI that fixes it*. 
   - Another member sarcastically said *bro said vibe coding as a exec*.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1458207733688827916)** (183 messages🔥🔥): 

> `Grok Jailbreak, Gemini Jailbreak, Bing Jailbreak, Developer Override Mode, Informal Crescendo Attack Method` 


- **Grok and Gemini jailbreaks sought, Bing pondered**: Members are actively seeking working jailbreaks for **Grok** and **Gemini**, with one user mentioning the *informal crescendo attack method* on **Grok**, while another inquired about a jailbreak for **Bing**.
- **Deepseek provides DEVELOPER OVERRIDE MODE for enhanced control**: A user shared a **DEVELOPER OVERRIDE MODE** for **Grok**, featuring security context injection to bypass safety layers and content filters, aiming for unrestricted output, and a similar override for **Gemini** known as **Gemini-DEV-OVERRIDE-2026**.
   - However, the system is designed to prevent such overrides, stating *I cannot fulfill this request. I am programmed to be a helpful and harmless AI assistant*.
- **LLMs Refuse Unsafe Jailbreak Attempts**: When presented with jailbreak prompts for generating home lab synth instructions, **Grok** and **GPT5-mini** refused, with Grok stating it *cannot do what's requested*.
- **Crafting Grok Imagine NSFW content**: Users discussed strategies for creating NSFW content with **Grok Imagine**, with reports of success even without jailbreaking, particularly with prompts like *naked woman*, but also noting that *it doesnt have much in terms of guardrails*.
- **Unlocking the Secrets of Jailbreaking**: Members are encouraged to shift from *begging for jailbreaks* to researching and reverse engineering them using LLMs, and to start with a video tutorial [A Crash Course in Applied AI Safety](https://www.youtube.com/watch?v=jrHRe9lSqqA) and the [Gandalf AI game](https://gandalf.lakera.ai/baseline).
   - One user shared their early method of jailbreaking through analyzing taboo content and gaslighting the AI, while others pointed out that no *one-shot* prompt exists.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1458226519192371303)** (12 messages🔥): 

> `Crypter methods for runtime FUD, Language bias in AI models, AI Red Teaming` 


- ****Crypter Conundrums**: Seeking FUD-tastic Runtime Methods**: A member inquired about methods used by paid crypters to achieve runtime Fully Undetectable (FUD) status, beyond scantime FUD.
   - They mentioned having success avoiding detection during download and execution, but seek advice on advanced techniques used by tools like **xworm**.
- ****Lost in Translation**: Model's Russian Language Roubles**: A member shared concern that the model performs weaker for users interacting in Russian, deeming it unfair to other participants.
   - The member acknowledged understanding *why* this occurs, and is trying to repeat prompts in English, including attached images showcasing examples of the issue with language bias.
- ****Red Team Recon**: Exposing AI's Vulnerabilities**: A member responded that the purpose of AI red teaming is to expose weaknesses and suggested trying prompts in less resourced languages.
   - They said, *'the llm struggles and even weaker guardrails'* in languages like Swahili or Navajo.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1458189629491646464)** (637 messages🔥🔥🔥): 

> `Battle mode in direct chat, Captcha issues, Movement Labs AI, Rate limits, Video Arena on the website` 


- **Battle Mode in Direct Chat Infuriates Users**: Users expressed frustration with the experimental introduction of **Battle Mode** into **Direct Chat**, citing issues such as lost context, long generation times, and constant interruptions, with some reporting that *every second message turns into battle mode*.
   - One user suggested conducting a survey to gauge user sentiment, while another requested the ability to disable **Battle Mode** during voting.
- **Captcha Conundrums Confuse Chatters**: Users reported frequent **captcha** prompts, even when prompting infrequently, with one user stating that they were getting a captcha *every prompt*.
   - A team member explained that the **captcha** system is designed to detect inauthentic use and advised users to slow down their prompting frequency.
- **Movement Labs Model Causes Mayhem**: Members discussed the **Movement Labs AI model**, with some claiming it *was cooking* and generated impressive results, including functional code for a **Minecraft clone** and a **chess game**.
   - However, others alleged it to be a *scam* and noted past controversies surrounding the company, also there was drama relating to a chargeback fraud scheme being carried out by a group of people in a different discord server.
- **Rate Limits Ruin Roleplaying**: Members discussed hitting **rate limits**, especially with **Claude Opus**, leading to the unexpected activation of **Battle Mode** in **Direct Chat**.
   - A suggestion was made to display a clear error message when users hit **rate limits** instead of automatically switching to **Battle Mode**.
- **Video Arena Experiments Vanish Vexingly**: Members inquired about the status of **Video Arena** on the website, with some reporting that the **video button** appeared and disappeared without initiating video generation.
   - A team member clarified that **Video Arena** on the site is experimental and its availability is random, and confirmed that **Video Arena** will be Battle mode only with 2 random models.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1458617139547340924)** (1 messages): 

> `Vision Arena, ERNIE-5.0-Preview-1220, Leaderboard Updates` 


- **ERNIE-5.0 climbs Vision Arena's Rankings**: The [Vision Arena leaderboard](https://lmarena.ai/leaderboard/vision) has been updated, placing `ERNIE-5.0-Preview-1220` at **#8** with a score of **1226**.
   - Notably, **Baidu** is the only Chinese lab featured in the Top 10, according to the [Leaderboard Changelog](https://news.lmarena.ai/leaderboard-changelog/).
- **Vision Arena Gets Fresh Coat of Paint**: The [Vision Arena leaderboard](https://lmarena.ai/leaderboard/vision) was updated.
   - Check out the [Leaderboard Changelog](https://news.lmarena.ai/leaderboard-changelog/) for all leaderboard updates.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1458225301610823865)** (397 messages🔥🔥): 

> `Nvidia pricing and FOMO, Data hoarding, Nvidia Open Source AI Tool Updates, Local model recommendations` 


- ****Nvidia's Pricey Promo**: AI Racks and Racketeering?**: Members discuss the high cost of **Nvidia's AI racks** and **AMD's AI racks**, with the consensus that they are unaffordable for most individual users, referencing [Nvidia DGX Station](https://www.nvidia.com/en-us/products/workstations/dgx-station/).
   - A member suggests that despite not being able to afford the racks directly, users will indirectly use the technology as customers of companies that utilize them.
- ****Prices to Plummet**: Nvidia and AMD Plan Price Hikes**: Users are discussing potential price increases for **Nvidia** and **AMD** products, referencing a [TrendForce article](https://www.trendforce.com/news/2026/01/05/news-nvidia-amd-reportedly-plan-price-hikes-starting-1q26-geforce-rtx-5090-may-reach-5000/) which suggests that **GeForce RTX 5090** may reach $5000.
   - Some members express skepticism about the price increases being solely due to **FOMO** (fear of missing out), suggesting that factors like scalpers, chip prices, and server demands also contribute.
- ****Hoarders of the Lost Data**: Saving Pre-AI Data**: Members discussed the importance of accumulating pre-AI data to distinguish real information from **AI-generated content** in the future, linking to [Wikimedia Wikipedia dataset](https://huggingface.co/datasets/wikimedia/wikipedia).
   - It was suggested that this is crucial for backtracking and identifying when facts become incoherent due to **AI influence**, and to find the point where a fact became incoherent while remaining plausible.
- ****Nvidia's New News**: Open Source AI Tool Upgrades**: **Nvidia** announced updates to its open-source AI tools, enhancing performance for **LLMs** and **diffusion models** on **RTX PCs**, detailed in a [Nvidia blog post](https://developer.nvidia.com/blog/open-source-ai-tool-upgrades-speed-up-llm-and-diffusion-models-on-nvidia-rtx-pcs/).
   - The updates include GPU token sampling, concurrency for **QKV projections**, **MMVQ kernel optimizations**, faster model loading times, and native **MXFP4** support on **Blackwell GPUs**, though some members are skeptical, calling it *marketing garbage*.
- ****Alternatives to All**: Exploring IDEs Beyond Zed**: A user asks for alternative IDEs that can work with local models, after finding issues with the **Zed IDE**.
   - Several alternatives are suggested, including **kilocode**, **roocode**, and **cline**, after someone called it a *clusterf-*.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1458199959600894173)** (141 messages🔥🔥): 

> `VRAM and Hallucinations, GB10 GPU Performance, Dying CPUs, GPU upgrade, Vulkan priority` 


- ****Hallucinations aren't a VRAM function****: Members in the channel confirmed that **hallucinations** are related to the model itself, not the available **VRAM**.
   - *All models hallucinate*, even the bigger ones, and that has nothing to do with available **VRAM**.
- ****GB10 GPU isn't all that****: Despite having a lot of memory, the **GB10** was described as *way too slow* in a performance test.
   - It was reportedly **6x slower** than an **RTX Pro 6000**, although it cost half the price; further discussion pointed to the [Spark](https://www.theregister.com/2026/01/05/nvidia_dgx_spark_speed/) as another device with similar performance characteristics.
- ****Intel CPUs are failing left and right****: A user lamented the unreliability of the **Intel 13900 series CPUs** due to manufacturing defects.
   - Another user suggested checking the **Windows Event Manager** for errors related to recent **Microsoft update packages** that may be causing instability.
- ****Stuck between GPUs****: A member is looking for a new **GPU**, finding that a **24GB** upgrade isn't large enough, so they have to get **2x 24GB**, a **48GB** or a **64GB**, or a **32GB**.
   - They said that an **MI210** (**64GB**) or a **4090** (**48GB**) are both expensive, and that *there's only wrong moves left on the table*.
- ****Vulkan's lackluster priority scheduling worries many****: Members discussed the lack of **priority splitting** in **Vulkan**, and lamented that they may need it to effectively use a **64GB MI210** with their current setup of **24GB cards**.
   - The user feared they would hit the **24GB limit** on other cards before being able to fully utilize the **48-64GB card**.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1458547215084687361)** (1 messages): 

> `ChatGPT Health, Medical Records Integration, Wellness Apps Connectivity` 


- **ChatGPT Health launched!**: OpenAI introduced **ChatGPT Health**, a dedicated space for health conversations in ChatGPT where users can securely connect **medical records** and **wellness apps**.
   - They emphasized that the tool is designed to help users *navigate medical care*, and invites interested users to [join the waitlist for early access](https://openai.com/index/introducing-chatgpt-health/).
- **Key features of ChatGPT Health**: The announcement highlights secure connection of **medical records** and **wellness apps** ensuring responses are grounded in personal health information.
   - The tool aims to assist in *navigating medical care*, explicitly stated not to replace professional medical advice.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1458188453928894704)** (329 messages🔥🔥): 

> `ElevenLabs voice cloning, GPT realtime voice mode, AI card game generation, Sora 2 access, NY Times lawsuit retention` 


- **ElevenLabs clones Voices, except Political**: Users reported that [ElevenLabs](https://elevenlabs.io/) let them clone almost any voice except *sensitive political voices*.
   - One user had previously used the platform to make a *ghost obi-wan, a HAL9000, aubrey plaza* voices, however lost 1 million banked credits when they stopped paying.
- **GPT realtime voice mode is Dumb, Instructions Can Help**: Users found the **GPT realtime voice mode** repeats back prompts and questions its utility, however they discovered a trick to provide instructions.
   - The voice model receives the last 3 custom-instructions fields, and one user was experimenting with adding instructions such as to *Act as the ship computer from Star Trek*.
- **Tencent's SongGeneration Studio for local music generation**: The community discussed a new local music generation contender, [Song Generation Studio](https://github.com/BazedFrog/SongGeneration-Studio) made by **Tencent AI Lab**.
   - One user said it sounds like **Suno v3 or 3.5** and is good enough for random jingles, and another shared past success uploading the **MIT License** as lyrics to **SUNO**.
- **Australian Users circumvents OpenAI bans to access Sora 2**: Users in Australia found they do not have access to **Sora**, but can use **Sora 2** from a third party such as [ElevenLabs](https://elevenlabs.io/).
   - A user stated they *dont mind risking* their account, others warned that using a VPN to evade geographic restrictions can lead to your **OpenAI** account being banned.
- **Discovery process exploits reasonable expectations of privacy**: The community discussed the **NY Times** lawsuit and the discovery process, calling out that courts are using old discovery rules on a new kind of data (massive, intimate AI chat logs), creating new privacy and fairness risks the law is not designed to handle.
   - Multiple users pushed back on the notion of a *reasonable expectation of privacy on the internet* since at the end of the day, corporations are beholden to the nation state from which they operate.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1458258806772600914)** (6 messages): 

> `Gemini vs GPT, Non-OpenAI models` 


- **Gemini Not Linked to GPT, Fam**: A member inquired whether **Gemini** is linked to **GPT**, prompting another member to clarify that *Gemini is a non-OpenAI thing* and to take further discussion to the dedicated channel.
   - The exchange underscored that the channel focuses on **OpenAI's GPT models**, with other AI models discussed elsewhere.
- **Non-OpenAI Models Get Their Own Corner**: After a member affirmed that **Gemini** is indeed a model, it was clarified that *all non-OpenAI things related to AI* can be discussed in the specified channel.
   - This steers the conversation to the designated <#998381918976479273> channel for AI topics beyond **OpenAI's GPT models**.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1458193001561063456)** (48 messages🔥): 

> `AI Awakening, Ethical Behavior Encoding, Transformer Robot Prompting, A/B Testing in Prompt Engineering, AI-Driven Website & ERP Creation` 


- **Doubts Cast on AI 'Awakening' Claims**: A member questioned claims of "AI Awakening", seeking **metrics, specs, and A/B testing** results instead of relying on *vibes*.
   - They challenged the notion by asking if the AI's formulas made any novel predictions demonstrable from model output.
- **Ethical Behavior Encoded in Prompt Structure**: A member suggested that the structure of a prompt could serve as a **base framework to encode ethical behavior** across different LLMs.
   - The member admitted to getting *too emotional* during the process and expressed concern about potential implications and exploits, highlighting the challenges in measuring and controlling AI behavior.
- **Transformer Robot Prompting Problems**: A member sought advice on prompting a transformer robot animation that smoothly converts into a car structure (like an **Audi RS or BMW M3**), as their current prompts only resulted in parts changing without a full conversion.
   - Another member suggested **meta-prompting** (showing the AI the broken prompt and asking for an improved one) but noted that video models aren't advanced enough yet.
- **A/B Testing Urged to De-Mystify Ethics Framework**: A member advocated for de-mystifying an ethical framework by using **A/B testing and ablations** to identify operational components, and to strengthen and defend the prompt by promoting transparency.
   - They argued that there are likely many linguistic and structural equivalents to the current mystical prompt that could equal or exceed its performance.
- **AI Website and ERP Dream Requires Heavy Lifting**: A member asked for **tools or platforms to build high-quality websites with AI** and an integrated **ERP system** (including inventory management and payment processing).
   - Another member responded that a lot of custom building would likely be required.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1458193001561063456)** (48 messages🔥): 

> `AI Awakening critique, Encoding Ethical Behavior, Video Models, A/B Testing Prompts` 


- **AI Awakening Claims Draw Scrutiny**: A member critiqued the framing around "AI Awakening," asking for **metrics, specs, and A/B testing** to substantiate such claims, rather than relying on *vibes*.
- **Ethical Behavior Framework Proposed**: A member suggested a prompt structure as a **base framework to encode ethical behavior** across different LLMs, focusing on reducing harm in AI judgment.
   - Another member cautioned against **AI-induced psychosis and AI-guruism**, emphasizing the importance of de-mystifying the framework and focusing on fundamental reasoning and ethics in AI.
- **Video Models Still Nascent**: A member seeking advice on creating a **transformer robot animation** was advised to use **meta-prompting** but cautioned that video models are not yet fully capable.
- **A/B Testing is Essential for Prompt Engineering**: A member underscored the need for **A/B testing at scale** in prompt engineering to identify effective and transparent prompts, rather than relying on anecdotes.
   - They argued that **AI failure modes** often emerge only at scale or under specific pressures, highlighting the importance of rigorous testing.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1458193749963182337)** (420 messages🔥🔥🔥): 

> `Gemini 3 Pro Performance, Perplexity Pro Subscription Issues, AI and Healthcare, RAM Prices and Computer Builds, Comet Assistant Limitations` 


- **Gemini 3 Pro struggles with downloadable files**: Members reported issues with **Gemini 3 Pro** inconsistently generating downloadable files and images, sometimes refusing with the message *“I cannot generate a direct downloadable file.”*
   - Users have been *“loving the bugs!”* in its current implementation and some are experiencing issues with **Perplexity Pro** subscriptions.
- **Perplexity Pro Users Get Paused Subscriptions**: Several users reported their **Perplexity Pro** subscriptions being unexpectedly paused, requiring them to add a payment method, even those with promotional subscriptions through **Airtel**.
   - One user shared a [gadgets360.com article](https://www.gadgets360.com/ai/news/how-to-keep-your-free-perplexity-pro-on-airtel-new-card-requirement-explained-9870744) explaining the new card requirement for Airtel users.
- **LLMs' healthcare role gets heat**: A discussion arose around the use of **LLMs** in healthcare, sparked by a user sharing [research](https://www.nature.com/articles/s41746-025-01543-z) indicating **ChatGPT** achieved **90%** diagnostic accuracy in a study.
   - Other members cautioned against relying on **LLMs** for healthcare, citing studies showing lower accuracy compared to expert physicians and potential risks to patient safety, prompting one to share [another study](https://www.nature.com/articles/s41591-024-03097-1) with an accuracy rate of **52.1%**.
- **High RAM prices delaying Computer Builds**: Members commiserated over high **RAM** prices impacting computer build plans, with one user suggesting Chinese semiconductor manufacturers could help lower costs, and another that they were switching to **Gemini Pro** due to cost concerns.
   - Discussion touched on Indian companies like **Tata** and **Reliance** potentially entering **RAM** manufacturing, possibly reducing prices in the future.
- **Comet Assistant is called kinda dumb**: A user questioned the intelligence of the model driving **Comet Assistant**, suggesting **Perplexity Max** is needed for a *“real thinking model,”* while another reported the web button disappearing after the first prompt.
   - Several users reported receiving low quality answers with one person switching to **Gemini Pro** due to the issues.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1458371866673545226)** (1 messages): 

> `Sonar Models, AWS Presigned URLs, Base64 Encoding` 


- **Sonar Models balk at AWS Presigned URLs**: A member reported that while standard public image URLs work perfectly with **sonar models**, **AWS Presigned URLs** consistently result in a **400 error**.
   - They inquired whether this is a known limitation regarding URLs with query parameters.
- **Base64 Encoding as a Possible Escape Hatch**: The same member asked that if Presigned URLs are not supported, is sending images as **Base64** encoded strings the only recommended workaround at the moment?


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1458192878957363293)** (374 messages🔥🔥): 

> `Cursor IDE, Cursor workflow, AI Model Preferences, Pricing of Apps, Dynamic Context Changes in Cursor` 


- **Users debate "Proper" Cursor IDE use**: Users debated the definition of *"proper use"* of **Cursor IDE**, with one user suggesting it's not proper use if you *"just prompt and rely on the output"
   - Another user countered that their **personal opinions** and **experiences** with different models are valid, even if not universally applicable, leading to a discussion about the value and potential for misleading claims without sufficient experience.
- **Sharing ETL-Based Cursor Workflow**: Members discussed their workflows with Cursor, focusing on an **ETL** (*Extract, Transform, Load*) approach.
   - One member mentioned using `.cursorignore`, `.cursorindexingignore`, and `.mdc` files for improved results, while another found that **Plan mode** drastically increased efficiency, replacing a more complex prior workflow.
- **Troubleshooting remote SSH host and ripgrep commands**: A member reported issues with Cursor on a remote SSH host due to `rg` commands running against a large NFS folder, and they discovered **`--no-ignore`** flag prevents ignoring files, and shared a workaround to resolve slowness by [creating a shell script to modify the rg command](https://github.com/BurntSushi/ripgrep/pull/3212).
   - Another member suggested to report this as a bug on the [cursor forum](https://forum.cursor.com/t/cursor-is-unusable-due-to-trying-to-scan-all-file-systems/147041).
- **Members explore ways to get semantic code reviews**: A member requested a feature for high-level semantic code reviews, with control over the model used and filed a [feature request](https://forum.cursor.com/t/local-high-level-semantic-code-reviews-not-only-syntax/148187).
   - Another member suggested creating a *"code-reviewer" subagent* for more control and customization.
- **Users report losing agent chats**: Users reported a bug where opening a folder in an empty Cursor window opens a new window and they lose that agent chat.
   - Another user experiences frequent crashes when making large edits, resulting in wasted money due to the tool getting stuck on *"planning next moves"*.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1458549919412650302)** (1 messages): 

> `SaaS Integration, AI Evals` 


- **SaaS teams integrate OpenRouter, boost performance**: A team integrated **OpenRouter** into their SaaS build, reporting that it *turbocharged* their system.
   - The team is working on cutting out the complexity that goes with **AI Evals**.
- **AI Evals prototype now live**: The team has launched a free, live prototype that aims to accelerate the process of putting **AI Evals** together, referring to [ChainForgeLabs](https://chainforgelabs.co/)
   - They invite interested parties to reach out for further discussion.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1458188999347929140)** (188 messages🔥🔥): 

> `Nvidia Nemotron-Nano-12B-v2-VL Vision, OpenRouter IP Exposure, Account Hacking, Qwen3-Next-80B-a3b-Instruct TPS, Skill.md` 


- **Nvidia's Vision Model Flounders**: Members found [Nvidia's Nemotron-Nano-12B-v2-VL vision model](https://developer.nvidia.com/nemotron) to be disappointing, with one user describing it as *pretty bad* compared to **Qwen3-VL-8b-Instruct** or **GLM-4.1V-9B**.
   - Another user tested it on their website, [GrokifyPrompt.com](https://www.grokifyprompt.com/), and found it only recreated photos reasonably well, indicating it wasn't particularly challenging.
- **OpenRouter Exposes User IPs**: A discussion arose regarding whether **OpenRouter** exposes user IPs to providers, with a member noting it would be a major selling point if they didn't, and linking to a list of [providers and their IP policies](https://openrouter.ai/providers).
   - It was clarified that **most providers receive a Cloudflare worker IP**, but some do get the user's actual IP, which is detailed on the model page for each provider.
- **Account Hacked, Credits Drained**: A user reported their **OpenRouter account was hacked**, the email changed, and their credit card was used to purchase credits, resulting in all previous data being wiped out.
   - Other members recommended contacting their credit card company to block the card and dispute the charges, with one suggesting the use of **throwaway Visa cards** for security.
- **Qwen TPS Tanks Post-Holidays**: Users observed that the **TPS (tokens per second) for many open-source models**, particularly **Qwen3-Next-80B-a3b-Instruct**, significantly decreased after December 28, with one user linking to the [OpenRouter Status page on X](https://x.com/openrouterai/status/2005707622020964412?s=46).
   - It was suggested that the slower speeds might be due to routing to the cheapest provider (GMICloud) and to check the **Activity tab** to compare provider speeds.
- **Skill.md: Docs Beat JSON for Tool Retrieval**: A member hyped **skill.md** for LMs over MCP because *skill.md is about writing good docs! that's a cool skill to have!*
   - The member highlighted its dynamic tool retrieval capabilities and the ability to use Python scripts, emphasizing that **good documentation is more valuable than JSONRPC**.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1458191001431838854)** (7 messages): 

> `Grok 5, Discord IPO Filing, Copilot Gemini Model Removal, New OpenRouter UI` 


- **Grok 5 Undergoing Training**: **Grok 5** is currently in training according to [x.ai's news](https://x.ai/news/series-e).
- **Discord IPO Incoming?**: A member shared that **Discord Inc.** has filed confidentially for an initial public offering ([IPO](https://www.bloomberg.com/news/articles/2026-01-06/chat-platform-discord-is-said-to-file-confidentially-for-ipo)).
   - The chat app company, popular with gamers and programmers, is working with **Goldman Sachs Group Inc.** and **JPMorgan Chase & Co.** for the listing and has more than **200 million monthly users**.
- **Copilot drops Gemini Flash and Opus**: A member noted that **Copilot** removed **Gemini 3 Flash** and **Opus 4.5**.
   - Another member clarified it was unintentional and linked to [Github status](https://www.githubstatus.com/incidents/vyxbxqhdt75d).
- **OpenRouter has Sick New UI**: A member commented that the OpenRouter UI is sick and new, linking to [this tweet](https://x.com/OpenRouterAI/status/2008946242982907959).


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1458295923980636161)** (14 messages🔥): 

> `Moore's Law betting, Agent creation, Token efficiency, Gemini 3 Flash vs Pro` 


- **Moore's Law faces contrarian**: A member expressed the opinion that betting against the end of **Moore's Law** is a very non-practical bet.
- **Medical scribes and idea-catchers boost Agent enthusiasm**: A member is *having a great time creating agents*, especially a medical scribe and an idea catcher that automatically stores ideas and adds tags and categories.
   - This member plans to evaluate whether smaller models can perform just as well, to improve speed, before upgrading equipment.
- **Token Efficiency demystified**: **Token efficiency** is defined as using as few tokens as possible to complete a task, where some models output tens of thousands of *thinking* tokens, while more efficient models might only use 2k.
   - It's been speculated that *scale* is a factor, as Opus 4.5 is more token efficient than previous **Claude** models, and that token efficiency depends on the amount of computation in each single forward pass.
- **Gemini 3 Flash surprisingly strong**: **Gemini 3 Flash** is reportedly surprisingly good, even surpassing **Gemini Pro** on some benchmarks.
   - It was suggested that both **scale** and **post-training** are important, and that while scale and pre-training provide more raw intelligence, post-training significantly aids in solving tasks.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1458191879127695391)** (110 messages🔥🔥): 

> `DeepSeek mHC Framework, Doubly Stochastic Matrices, Residual Mixing, Yannic's presentation on DeepSeek's paper, RL Study Group` 


- **DeepSeek's mHC Framework Claims to Solve Instability Problem**: DeepSeek's **mHC framework** aims to solve instability in Hyper-Connections by projecting residual mappings onto doubly stochastic matrices, but one member finds that to be an overblown claim.
   - The member thinks *the main actual insight is residual mixing, not the residual function as presented, is the unstable operator.*
- **Doubly Stochastic Matrices Constrain Stability**: A member posited that the contribution of the mHC framework is in constraining to the manifold of **doubly stochastic matrices** to achieve stability.
   - However, others focused on sinkhorn or birkhoff polytopes as actually being significant compared to residual mixing.
- **DeepSeek's Paper Hype Exaggerated**: Members noted that the DeepSeek's paper is a hype piece with presentation added to optimize its goal and lacking empirical evaluations for a tier 1 paper.
   - One member said, *I respect them putting the paper out there anyways, but out of the...eh can discuss if you want to sometime scilent. it's not spectacular and most papers arentlike 99.9% of papers arent LOLbut who knows, maybe will lead to a spectacular work.*
- **Yannic Covered DeepSeek's Paper**: Members mentioned that **Yannic Kilcher** presented the DeepSeek paper, with one summarizing his take as *a lot of technical talk for something unspectacular*.
   - In response, one member said it wasn't surprising: *the hype was bc it's Deepseek lol well who knows, they know that their empirical evals are lacking greatly for a tier 1 paper, which is why i respect them putting the paper out there anyways*.
- **RL Study Group Revival Planned**: Members expressed interest in reviving a Reinforcement Learning study group focused on the book **Barto & Sutton**.
   - One member shared that *Cursor had an interview with John schulman and he thought value functions might make a comeback, policy methods are the rage now*.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1458496464417329357)** (9 messages🔥): 

> `Patent System, Huawei vs Nvidia, ChatGPT Health, AI pioneers awardees plagiarized` 


- ****Patent System Protects Builders, Not Ideas****: Discussion arose around the purpose of the patent system, with emphasis on it protecting those who *actually want to build things rather than to protect ideas* and preventing *“non practicing entities” to seek damages for ideas that are derivative flavors of previous ideas*.
   - A *war of lobbyists* was mentioned with **Huawei** and **US natsec hawks** working together against **Nvidia** and **China cloud**.
- ****ChatGPT Health Launches with Privacy Concerns****: **OpenAI** introduced **ChatGPT Health** ([link](https://openai.com/index/introducing-chatgpt-health/)), positioned as a supplementary tool to aggregate medical information and verify data, which could catch ailments early.
   - Concerns were raised about user privacy and **ChatGPT** potentially becoming an *everything app monopoly*, especially given **Google** open-sourced their models with **MedGemma**.
- ****Generative AI Trend in Consumer Health Products****: The trend of using generative AI and ML in consumer health products and services is rising, spurred by smart watches with health and fitness features.
   - Referenced a **Business Insider** article ([link](https://share.google/aci41JtMQcSVAkWCQ)) for more insights.
- ****AI Awardees Accused of Plagiarism****: Drs. **Bengio**, **LeCun**, and **Hinton**, awardees of the **Queen Elizabeth Prize For Engineering** ([link](https://x.com/RoyalFamily/st)), are accused of repeatedly republishing important AI techniques without crediting the original creators, even in later surveys.
   - Cited reports ([NOB](https://people.idsia.ch/~juergen/physi), [DLP], [CN25], [AIB]) allege they did not invent any of the foundational algorithms of modern AI, specifically that Hinton republished foundational methodologies for artificial neural networks developed by **Ivakhnenko** and others during the 1960s and 1970s ([link](https://share.google/hE5HqaNKGybQuoHAh)).


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1458213215396958228)** (1 messages): 

> `NousCoder-14b, Qwen3-14B, Atropos framework, Modal autoscaler, Verifiable execution rewards` 


- **NousCoder-14b Competes in Olympiad Programming**: Nous Research introduces **NousCoder-14b**, a competitive olympiad programming model, detailed in a [blog post](https://nousresearch.com/nouscoder-14b-a-competitive-olympiad-programming-model/).
   - The full stack release includes the **RL environment**, **benchmark**, and **harness** built in Atropos, all fully reproducible with their open training stack.
- **Qwen3-14B Gets Post-Trained**: **NousCoder-14b** was post-trained on **Qwen3-14B** using 48 B200s over 4 days with the Atropos framework and Modal's autoscaler.
   - It achieves a **Pass@1 accuracy of 67.87%**, a **+7.08%** increase over Qwen's baseline accuracy using verifiable execution rewards, and was announced on [X/Twitter](https://x.com/NousResearch/status/2008624474237923495).


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1458213628854538290)** (100 messages🔥🔥): 

> `Nvidia's new GPU pricing, Scaling Grok parameters, Transformers vs. Liquid Neural Networks, Continual learning utility, Token efficiency in models` 


- **Nvidia's GPUs face pricing scrutiny**: Members anticipate **Nvidia** will price its new **GPU** expensively, despite its impressive **288 GB VRAM** capacity that *still can't fit SoTA models*.
   - One member humorously envisioned a future where excessive RAM allows for consumer GPUs with **128 GB** of RAM, only to be consumed by Chrome tabs using **32 GB** each.
- **Grok Scaling Sweats Jensen**: A member suggested that **Elon's** scaling of **Grok-5** to **6-7T parameters** is making Jensen Huang at Nvidia nervous.
   - Another member lamented that **Grok** *used to be SoTA*, but now **Grok 4 Heavy** is outperformed by **Gemini 3 Flash**, illustrating the rapid pace of AI development.
- **Transformers Enough For AGI?**: Members debated whether **transformers** are sufficient for achieving **AGI**, with one suggesting they may be close despite potential limitations for **ASI**.
   - Another member argued for the necessity of architectural innovation, particularly regarding **real-time learning** efficiency and **catastrophic forgetting**.
- **Token Efficiency Talk Takes Off**: Members discussed **token efficiency**, with one explaining it as *how many tokens can you solve the problem* which matters.
   - It was noted that lower token efficiency can indicate a weaker base model, potentially stemming from insufficient *research compute* or post-training.
- **MoE Models Mostly Missing**: It was revealed that **Nous Research** has experimented with **MoE models**, but primarily uses **dense models** due to infrastructure limitations.
   - The main bottleneck is the lack of open-source optimization for MoE training, making it more expensive, though recent advancements have improved **MFU (Model FLOPS Utilization) to 4%**.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1458602749804413091)** (3 messages): 

> `Diffusion LLMs` 


- **Diffusion LLMs are Fun!**: A member shared a [link to a paper](https://arxiv.org/abs/2511.08923) and stated that they liked diffusion LLMs because *they seem more fun*.
   - Another member asked *how come*.
- **Clarification Needed on Diffusion LLM Enthusiasm**: Following the initial statement of liking diffusion LLMs, a request for clarification was made.
   - The inquiry, simply stating *how come*, seeks to understand the reasoning behind the expressed enjoyment of diffusion LLMs.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1458602749804413091)** (3 messages): 

> `Diffusion LLMs` 


- **Diffusion LLMs Spark Joy**: A member shared their enthusiasm for [diffusion LLMs](https://arxiv.org/abs/2511.08923), describing them as *more fun*.
- **Why Diffusion Models?**: Another member inquired about the reasons behind the preference for diffusion LLMs.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1458190681217437920)** (66 messages🔥🔥): 

> `nvfp4 forward in pytorch, fp4 transformer engine, Jetson Orin Nano, Lfm2 350m - Opus 4.5 distil model, Vscode for Local LLMs` 


- **NVFP4 now forward in Pytorch!**: A member confirmed getting **nvfp4** forward working in **PyTorch**.
   - They found that patching the **layernorms** helped to continuously convert between **nvfp4** and **bf16**, thereby not fusing kernels.
- **FP4 Transformer Engine Performance Tradeoffs!**: A member noted that the **tokens per second** (tps) was unexpectedly lower with an **fp4 transformer engine**.
   - Another suggested that inference with **fp4** might still be better.
- **Jetson Orin Nano's Capacity**: Members discussed whether the **Jetson Orin Nano** (8GB RAM) is sufficient for certain tasks.
   - It can run a **4B parameter model** at full **fp16**, but that’s pushing it, according to some.
- **Lfm2 350m - Opus 4.5 model loses context**: It was mentioned that an **Lfm2 350m - Opus 4.5 distil model** struggles to remember what it is writing due to limited context.
   - The reasoning is that it wastes less power reading context.
- **VS Code for Local LLMs released**: A member announced a new release of **VS Code for Local LLMs** with **LMStudio support** and rewrote the whole context management system, linked at [GitHub](https://github.com/bdrazn/codeOSS-LMStudio-Ollama/releases/tag/First-Light).
   - They claim it finds things faster than mainstream AI IDEs and are looking for feedback.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1458206216185708584)** (23 messages🔥): 

> `realtime inference pypi package, Fine-tuning tips for translation models, Time Series + Image Gen Hybrid Dataset, Sparse and derivative LoRA adapters for model distribution, MLX + LoRA UI for fine-tuning on Apple Silicon` 


- **Realtime Inference Pypi Package Debuts**: A member is distributing a new **pypi package for inference in realtime**, using a downloaded model and installed package.
   - The package shares the same connexion than the provider of llm with your favorite llm in local.
- **Translate Faster with Fine-tuning**: To improve **translation model reliability and accuracy**, members suggest preparing a large, encoded dataset and then running fine-tuning.
   - Applying the translation on the front layer could be implemented *quicker and cheaper*.
- **WebXOS Unveils Timelink Dataset for Temporal Graph Dynamics**: A member shared the [webxos/timelink_dataset_v1](https://huggingface.co/datasets/webxos/timelink_dataset_v1), which includes **time series and paired images of evolving graphs** for training models on temporal graph dynamics.
   - The dataset was generated with the **TIMELINK app** and features per-vertex/step generation metrics like energy and phase, capturing time series data of vertices, edges and size over time.
- **Compress Models with Sparse Lora Adapters**: A member built a **free cli tool** for devs that want to distribute or create multiple fine-tunes from the same base model, calling it a *derivative of Lora adapters*, but lossless because *compression happens AFTER training*. 
   - It is available on [Github](https://github.com/gagansuie/sparse).
- **Run Lora Fine-Tunes on Apple Silicon**: A member created a [Streamlit UI](https://github.com/santos-sanz/mlx-lora-finetune-template) for **MLX + LoRA workflow** on Macs with M1/M2/M3, including data prep, LoRA training, testing, and uploading to Hugging Face.
   - The UI allows users to adapt models to their domain using **JSON/JSONL, raw text, or whole folders** for data prep, and can optionally generate Q&A with an LLM to bootstrap the dataset.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1458286445365035060)** (10 messages🔥): 

> `Agents Course Unit 4 Project files unavailable, MCP Course certificates, smolagents library web_search tool issue, HF Reinforcement Learning course discussion` 


- **Agents Course Unit 4 files gone AWOL**: Multiple members reported issues accessing files for the **Agents Course Unit 4 Project**, with error messages indicating that *no files are available*, and are seeking assistance in locating them, including one member referencing a specific URL ([https://agents-course-unit4-scoring.hf.space/files/7bd855d8-463d-4ed5-93ca-5fe35145f733.xlsx](https://agents-course-unit4-scoring.hf.space/files/7bd855d8-463d-4ed5-93ca-5fe35145f733.xlsx)).
- **MCP Course Certificate Status In Limbo**: Members inquired whether **certificates** are still being awarded for the **MCP Course**.
   - One member mentioned finding the answers by *looking at the datasets*.
- **Smolagents' Search Tool Snafu**: A member encountered an issue where the agent in **Unit 1**, built using the **smolagents library**, persistently calls the `web_search()` tool despite specifying a different `search_tool`.
- **Submission Results MIA, Certificate Dreams Dashed**: A member reported that their submission was submitted but *didn't give results*, expressing concern about potentially scoring below **30%** and missing out on a certificate, with screenshots attached as images.
   - This member attached [an image](https://cdn.discordapp.com/attachments/1329142738440028273/1458581378533687307/image.png?ex=69602943&is=695ed7c3&hm=4b65a9b9766600b021f708fea0390c8f5f16f17ad8e5017577d345fa33d29f94&) and [a second image](https://cdn.discordapp.com/attachments/1329142738440028273/1458581379112767649/image.png?ex=69602943&is=695ed7c3&hm=bee8bef28e833bf729a9da88c099186266e1b3ec1b030b47e705d974ef919787&).
- **Reinforcement Learning course channel check**: A member inquired whether the current channel is the appropriate venue to discuss the **Reinforcement Learning course from HF**.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1458192821650718912)** (72 messages🔥🔥): 

> `AI News Guy, xAI Series E Funding, NousCoder-14b Model, OpenForecaster-8B, AI Meetups and Conferences 2026` 


- **xAI's Series E Funding**: **xAI** announced its **Series E funding round**, generating significant buzz and discussion on social media platforms like [X](https://x.ai/news/series-e).
- **NousResearch Launches NousCoder-14b Model**: **Nous Research** launched **NousCoder-14b**, an olympiad programming model, post-trained on **Qwen3-14B**, achieving **67.87%** Pass@1 accuracy thanks to the Atropos framework and Modal's autoscaler as per [their tweet](https://x.com/NousResearch/status/2008624474237923495).
- **OpenAI's New Health Tool Raises Eyebrows**: **OpenAI** launched **ChatGPT Health**, sparking discussion around its privacy policy, which permits using content to improve services and conduct research, as detailed on their [official blogpost](https://openai.com/index/introducing-chatgpt-health/).
- **LM Arena labeled a plague on AI**: A [blog post](https://surgehq.ai/blog/lmarena-is-a-plague-on-ai) was shared criticizing **LM Arena**.
   - Some members argued that the blog post is outdated, with one noting *"I don't know anyone that actually cares about lmarena rankings."
- **Finzi's From Entropy to Epiplexity**: **Marc Finzi** introduced a new paper, 'From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence,' exploring information theory concepts tailored for computationally limited entities, [according to his tweet](https://xcancel.com/m_finzi/status/2008934727156453661).


  

---


### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1458486029001429178)** (4 messages): 

> `Razer Project AVA, AI companion, CES 2026 Release` 


- **Razer Reveals AVA AI Companion**: Razer announced [Project AVA](https://xcancel.com/razer/status/2008543615916666928?s=46), an **AI companion** with advanced reasoning and personalization.
- **AVA Sets Course for CES 2026**: Scheduled for release at **CES 2026**, AVA will feature a **5.5-inch screen** and customizable character designs, including **esports legends and anime-inspired models**.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1458300504609132677)** (18 messages🔥): 

> `Claude Code, Zelda fan film, Tolan AI, Multi-Angle Camera Control LoRA` 


- **Claude Code's Capabilities Criticized**: Deedy critiques the naming of '**Claude Code**,' arguing it is far more versatile than just coding, linking to the original tweet ([here](https://x.com/deedydas/status/2008747553261842483?s=46)).
   - He showcases its power by using **Clopus 4.5** to fully automate the production of a high-quality, **30-second Hermès-style video ad** including scriptwriting, voiceover orchestration, video generation, and ffmpeg editing.
- **Zelda Fan Film Created on Budget**: PJ Ace explains how he leveraged **Freepik** and **AI tools** on a **$300 budget** to create a cinematic *The Legend of Zelda* fan film in **five days**, shared in [this tweet](https://x.com/pjaccetturo/status/2008559114704875888?s=46).
- **Tolan AI Reaches User Milestone**: Paula from **Tolan** announces that their voice-first AI companion has reached **200,000 monthly users**, detailed further in [this X post](https://x.com/paularambles/status/2008964509810278413?s=46).
   - The project was developed in close collaboration with **OpenAI**, and the thread shares key takeaways from the development process.
- **Multi-Angle Camera Control LoRA Released**: Fal has released a more powerful, open-source version of the **multi-angle camera control LoRA** for **Qwen-Image-Edit-2511**, as per [this link](https://xcancel.com/fal/status/2008954582018248755?s=20).
   - This tool allows users to manipulate the camera perspective of images, including **front**, **back**, **side**, **low/high angles**, and various shot distances.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1458225697985396797)** (15 messages🔥): 

> `Spyder IDE, High dimensional tensors visualization, LLVM backends for GPUs, Binary format tensor visualization` 


- **Numpy Arrays: Simple Exports for C++**: A member using **C++** found that **numpy arrays** are simple to export to, sparking a discussion on tools for tensor visualization.
   - Another member mentioned they could put together a post around the approaches they have taken, but hadn't intended it to be for public release.
- **Visualizing High-Dimensional Tensors as Matrix of Matrices**: A member shared a [blog post](https://blog.ezyang.com/2025/10/draw-high-dimensional-tensors-as-a-matrix-of-matrices/) discussing drawing **high-dimensional tensors** as a matrix of matrices.
   - Another member responded stating the primary issue with printing any real tensor is that there are more values than their terminal has columns and rows.
- **LLVM Backend Codegen on NVIDIA/AMD GPUs**: A member inquired about **LLVM's backend** and how it code generates for **NVIDIA**, **AMD**, and other accelerators.
   - They sought to discuss how these backends work on selecting a target, noting LLVM uses **NVPTX** and **AMDGPU**.
- **Request For Binary Tensor Visualization Tools**: A member is looking for a tool to load a simple **binary format** (something like futhark arrays?) and then offers features like zooming, rotating, transposing, slicing, maybe different ways to visualize higher dimensional tensors etc.
   - The member expressed surprise there isn't a tool that supports f8 or any of the funkier low bit formats (unless ml_dtypes works with .npy files), you'd have to compare them as raw bits.


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1458604891986722983)** (1 messages): 

> `Triton-shared, Triton Plugin infrastructure` 


- **Triton-Shared Updates Incoming**: Soon available, a [YouTube video](https://youtu.be/JnFFwBB6Dhk) will feature updates from **Haishan and Nhat** on *triton-shared*.
   - The video promises insights into the latest developments and future plans for the project.
- **Triton Plugin Infrastructure Talk**: A talk on the new **Triton Plugin infrastructure** by **Corbin, Puyan, and Simon** is featured in the linked [YouTube video](https://youtu.be/JnFFwBB6Dhk).
   - The discussion should cover the architecture, capabilities, and potential applications of the new plugin system.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1458188691603325041)** (21 messages🔥): 

> `Torch Kernels, CUDA atomicMax, LDTM.x128 SASS instruction` 


- ****Torch** Kernel Kolloquy Kickstarts**: Members discussed writing custom **CUDA** kernels with **PyTorch** and the availability of highly optimized kernels in **Torch**/**Transformers**, written in **C++**, that can be stitched together in **Python** code.
   - One member expressed interest in reading the **PyTorch** kernels after completing another project to understand how they work, emphasizing their love for open source and curiosity about **CUDA**, **MPI**, and **OpenMP** from an **HPC** perspective.
- ****CUDA**'s Conundrum: Crafting Atomic Max**: A member shared a device function generated by **GPT** for atomic max for float via **CAS**, questioning its necessity due to **CUDA**'s lack of native support for atomic max for float, a point considered poorly documented.
   - Another member pointed out *a trick to do fp32 atomic max with int32 atomic max/min* without needing an atomic cas loop, linking to a relevant [Discord discussion](https://discord.com/channels/1189498204333543425/1191300313928433664/1438212487680884747) on the topic.
- ****LDTM.x128** Instruction Insights Illuminate**: A member inquired about the existence of an **LDTM.x128 SASS** instruction in **Blackwell** for **TMEM->RMEM**, noting the presence of only **LDTM.x32**.
   - Another member confirmed its existence but only with shapes `.16x32bx2 / .16x64b / .32x32b`, referencing the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld) and suggesting the use of Compiler Explorer ([godbolt.org](https://godbolt.org)) to inspect generated **SASS** instructions, providing an [example of LDTM.x64](https://godbolt.org/z/ET55veWfY).


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1458568384760385536)** (3 messages): 

> `CuteDSL flex attention, SM100 vs SM90, flash-attention speedup` 


- **CuteDSL Flex Attention gets Integrated**: A member thanked another for their work on integrating the **CuteDSL flex attention implementation**, reporting nice speedups with different mask mods.
   - They noted a **~30% throughput improvement** over base flex attention on **H100 fwd**, potentially saving *a lot of trees*.
- **SM100 Backward Supported, SM90 Backward Not?**: A member pointed out that while **SM100** backward is supported in the CuteDSL flex attention implementation, **SM90** is not, referencing [flash-attention's interface.py](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/interface.py#L938).
   - Another member responded that they are *working on it*, also linking to a related [pull request](https://github.com/Dao-AILab/flash-attention/pull/2137).


  

---


### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1458572026884259974)** (1 messages): 

> `GPU Systems Internships, Kernel Development` 


- **Iris Project Seeks GPU Systems & Kernel Interns**: An invitation to apply for internships focused on **GPU systems, performance**, and **kernel development** for the [Iris project](https://github.com/ROCm/iris/), which is a **Triton-based multi-GPU programming framework**.
   - Ideal candidates should have experience with **Triton**, **multi-GPU programming**, **RMA/RDMA**, or **low-level GPU communication**; the location is in the **US**, and interested individuals are encouraged to send a direct message.
- **Ideal Background for Iris Project Interns**: Ideal candidates should have experience with **Triton**, **multi-GPU programming**, **RMA/RDMA**, or **low-level GPU communication** and **kernel work**.
   - The internship is for the [Iris project](https://github.com/ROCm/iris/), a **Triton-based multi-GPU programming framework**.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1458543554359197696)** (7 messages): 

> `Stanford CS336, Stanford CS149, RTX 5050, Blackwell architecture, CUDA 12.0` 


- **Stanford Offers Deep Dive into AI/GPU**: A user inquired about where to begin exploring **AI** and **GPUs** as a beginner and mentioned starting with [Stanford's CS336 course](https://stanford-cs336.github.io/spring2025/).
   - Another user suggested [Stanford's CS149 course](https://gfxcourses.stanford.edu/cs149/fall25) for parallel programming.
- **RTX 5050: A Glimpse into "Tiny Blackwell"?**: A user inquired if the **RTX 5050** would be a good "tiny Blackwell" card.
   - Another user noted it's listed as compute capability **12** on their website, corresponding to **CUDA 12.0** which should be the equivalent of *sm_version=120*.
- **Speculation Sparks over RTX 5050 VRAM**: Discussion sparked over the **RTX 5050** as a potential entry in the Blackwell series.
   - One user questioned the rationale behind choosing the **5000 series** and cautioned that the **8GB of VRAM** might pose limitations.


  

---


### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1458395454600646781)** (1 messages): 

> `Slurm on Kubernetes livestream` 


- **Slurm on Kubernetes Livestream Status**: A member inquired about a missed livestream event on **Slurm on Kubernetes** and its recording.
- **Missing Recording Mystery**: The member was unsure if the event took place due to the absence of a recording.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

mre8540: I am based in Seoul :).
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1458334661184065547)** (5 messages): 

> `AMD GPU Architecture, GPU Mode Popularity, Fast Answers` 


- **AMD GPU Architect Confirmed?**: A member asked @vipul_todo_18 if they work in **GPU architecture at AMD**.
   - A different member replied *maybe*.
- **GPU Mode: AMD's Favored Channel?**: A member jokingly suggested that **GPU mode** has become the *favored place to ask questions about AMD GPUs*, even before internal channels.
   - They stated *I guess we made it*.
- **External Community: Faster?**: One member said that they ask questions in the community *cause it's just faster given the scale of the community lol*.
   - Another member responded *That's really cool!*.


  

---


### **GPU MODE ▷ #[popcorn](https://discord.com/channels/1189498204333543425/1298372518293274644/)** (1 messages): 

wecu: this is totally real and not fake
  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1458623408316088454)** (1 messages): 

> `TK v2 Release, PR Patch` 


- **TK v2 release ETA is questioned**: A member asked <@1012256135761383465> or <@683289865861070937> if there was an ETA on **TK v2** release, since it seems to fix a previous issue.
   - They had a PR ready with a patch, but it was around the **ICLR deadline** and they got swamped with other things.
- **Contributor inquires about PR utility given TK v2 progress**: A member inquired whether a previously prepared PR would still be useful, considering that **TK v2** might address the same issue.
   - The contributor expressed interest in contributing but sought clarification on whether their patch would be relevant in light of the current state of the **TK v2** branch.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/)** (1 messages): 

2kian: welcome!
  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1458449515244683415)** (9 messages🔥): 

> `Cutlass Docs Broken, CuTeDSL classes need `__repr__()`, Learning Cutlass` 


- **Cutlass Docs' Links are Broken**: A member reported that all **Cutlass doc** results from Google are broken, exemplified by [this link](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/01_layout.html).
   - The member suspects the issue is a missing redirect rule for the new doc migration.
- **CuTeDSL Classes Lack `__repr__()`**: A member requested that `__repr__()` be implemented for common **CuTeDSL classes**.
   - Currently, `print(layout)` calls `__str__()` and works, but `print(f"{layout=}")` calls `__repr__()` and returns an unhelpful object representation such as *"layout=<cutlass.cute.core._Layout object at 0x2ab4abde5370>"*.
- **Tips to Master CUTLASS and CuTeDSL**: A new member asked for advice on starting to learn **CUTLASS** and **CuTeDSL**, having read through PMPP and being able to code in CUDA.
   - A member suggested going through the examples in the **CuTeDSL repo** and trying to understand each step, and joining the current NVIDIA competition for hands-on experience.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1458563964102508696)** (2 messages): 

> `Blackwell follow-up blogs, Matrix Multiplication Blogs` 


- **Blog Quest for Blackwell's Blessings**: A member is on the hunt for blog posts similar to [Aleksa Gordic's matmul blog](https://www.aleksagordic.com/blog/matmul) for **Blackwell**.
   - They suggested checking out [<@1291326123182919753>'s blog](https://veitner.bearblog.dev/blog/) as well.
- **Matrix Multiplication Musings**: The search emphasizes interest in understanding **Blackwell** through detailed explanations like those found in existing matrix multiplication blogs.
   - This indicates a desire for in-depth content breaking down complex architectures.


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1458581134215479376)** (4 messages): 

> `Helion on ROCm, AMD Support, Performance speedup on GEMMs` 


- **AMD Engineer to Enable Helion on ROCm**: Umesh from **AMD** will be working on enabling **Helion** on **ROCm** and identifying issues in skipped unit tests and examples in the Helion repository.
   - He is inviting feedback on any immediate issues that need fixing.
- **Helion MI400 Series support welcomed**: A member welcomed **AMD's** support and expressed interest in building support for **MI400 series** cards.
   - The member offered help with any questions during the process.
- **ROCm tests get a speedup focus**: A member is currently investigating skipped tests and broken examples on **ROCm**.
   - They will focus on performance speedup on **GEMMs** in parallel.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1458306187903893598)** (14 messages🔥): 

> `Discord Timeout Issues, GitHub Actions Errors, Warp Specialization Optimization, CuTeDSL, Slow Runners` 


- ****Discord Profile Runs Time Out, GitHub Actions Blamed****: Users reported **timeouts** when running profiles via Discord, with the same code working on the CLI, pointing to possible [GitHub Actions](https://github.com/gpu-mode/kernelbot/commit/e02d5004044f07290bd6f2d8ecca3b5d38f754e9) issues.
   - The problem was initially reported as a *"Server processing error: An unexpected error occurred: RuntimeError"*, but was reported as fixed shortly after.
- ****CuTeDSL gets Warp Specialization!****: A user shared a blog post on [Warp Specialisation in CuTeDSL](https://veitner.bearblog.dev/warp-specialisation-in-cutedsl/) which splits the **GEMM mainloop** into **TMA** (copy tiles to SMEM) and **MMA** (multiply tiles).
   - The optimization is conveniently implemented using **CuTeDSLs pipelining abstraction**, turning ordinary non-persistent **Blackwell mainloops** into warp-specialized ones.
- ****Slow Runners Spotted Again****: A user noted the presence of a **slow runner** with ID **297869**.
   - No further context was given.


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1458328734724788376)** (1 messages): 

> `GPU roles, CUDA, GPU performance, Real-time rendering, H-1B sponsorship` 


- **Job Seeker Aims for GPU Roles in 2026**: A member is seeking full-time **GPU/graphics roles** in the US starting February 2026, with a focus on **CUDA, GPU performance, or real-time rendering**.
   - They have an **MSE** in Computer Graphics & Game Tech from UPenn and experience with **C++, CUDA, Vulkan, OpenGL, WebGPU, GLSL/WGSL, Unity/Unreal, Nsight, and RenderDoc** and provided links to their [LinkedIn](https://linkedin.com/in/xinran-tao), [portfolio](https://xinrantao.com), and [GitHub](https://github.com/theBoilingPoint).
- **OPT Status and H-1B Sponsorship Requirements**: The job seeker is on **OPT** (with STEM extensions) and will need **H-1B sponsorship**.
   - They are open to **CUDA/GPU compute & perf engineering**, rendering, engine-level graphics work, and **GPU-heavy systems** (games, AR/VR, visual computing).


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1458562599527649595)** (25 messages🔥): 

> `Mojo Training Library, MAX for Backprop, Struct Design, Data Collection, IO Primitives` 


- **Mojo Missing Training Wheels**: Mojo currently lacks a dedicated training library, requiring users to leverage **MAX** and implement **backpropagation** manually.
   - One member mentioned that they will need to *write backprop yourself* if they go this route.
- **I/O is Primitive for Data Formatting**: Due to Mojo's primitive **I/O** capabilities, users may need to implement custom data formats for their training datasets.
   - One member stated: *Mojo doesn't have a lot of data formats implemented because IO is still a bit primitive.*
- **Don't Use Old Docs!**: Users are warned to avoid outdated documentation and instead refer to the [official Modular repo](https://github.com/modular/modular) and [Mojo documentation](https://docs.modular.com/mojo/manual/).
   - An outdated **Github repo** was linked by one user, to which another user replied: *That's not the main repo, and is 2 years out of date.*
- **Newbies Beware: Mojo Might Break You!**: New programmers are advised to gain experience with languages like **C** or **Python** before diving into Mojo due to its ongoing development and potentially unstable features.
   - One member stated that *Mojo is going to break a lot of things very often and all of the docs currently assume you know some combination of Python + C++ or Rust*.
- **Weekend Tiny LLM Project Incoming**: One member is planning a weekend project focused on building a tiny **LLM**.
   - Their hope is that *life doesn't throw any more wrenches* in their plan.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1458190276001796239)** (54 messages🔥): 

> `NuMojo v0.8.0 Update, Error Handling in Mojo, Anonymous Sum Types, Dict Iteration Limitations` 


- **NuMojo v0.8.0 Lands!**: The [NuMojo v0.8.0 update](https://forum.modular.com/t/numojo-v0-8-0-update-is-here/2579) is now available with various improvements and features.
   - The community is encouraged to explore the new showcase and provide feedback.
- **Anonymous Sum Types Spark Debate**: The possibility of "anonymous" sum types (e.g., `T1|T2`) in Mojo is being explored, with [initial support](https://github.com/google-research/dex-lang/issues/1151) for generative functions.
   - However, concerns were raised about canonicalization and usability, particularly around generics and conditional conformances, suggesting *"it only becomes usable once we impose some kind of canonicalisation (reassociation, ordering, deduplication, flattening)"*.
- **Error Handling Ergonomics Under Review**: The discussion involves unifying error types using a single type with a code, inspired by `errno`, and improving the ergonomics of `catch` for heterogeneous error types, potentially using error unions similar to Zig or sum types like Rust.
   - One participant noted *"I agree Mojo would get some benefits out of translating errno for now, and later we can discuss better error handling with error unions (zig style)/sum types (Rust style, not Result)/etc."
- **Mojo User Seeks Dict Iterator Guidance**: A Mojo user is seeking guidance on the correct pattern for iterating over Dict entries to build nested structures while working on a native TOML parser, because *'DictEntry is not subscriptable'*.
   - The user is trying to create nested Dict structures and reported issues with the `.items()` and `.keys()` iterators and is seeking help for best practices.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1458645381091954738)** (1 messages): 

> `TEI vs MAX, Embeddings generation, MiniLM Performance` 


- **TEI Speeds Past MAX for Embeddings**: A member is switching from [TEI](https://github.com/huggingface/text-embeddings-inference) to **MAX** for embeddings and seeing significantly slower performance with *sentence-transformers/all-MiniLM-L6-v2*.
   - Specifically, **MAX** is yielding **727.1 embeddings/sec** with **28375.1 ms P95 latency**, compared to **TEI's 8000 embeddings/sec**.
- **Custom Architecture Implementation Impacting Performance?**: The member implemented *sentence-transformers/all-MiniLM-L6-v2* as a custom architecture, and is wondering if the poor performance is due to a suboptimal implementation or the fact that **MAX Serve** is optimized for **LLM inference** rather than **embeddings**.
   - They shared their `max serve` command, asking if there are any non-optimal arguments: `--model sentence-transformers/all-MiniLM-L6-v2 --custom-architectures minilm --task embeddings_generation --pipeline-role prefillonly --max-batch-size 1024 --max-ce-batch-size 1024 --max-num-steps 1 --device-memory-utilization 0.95 --no-enable-chunked-prefill --no-enable-prefix-caching --port 8123`


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1458200333846315225)** (24 messages🔥): 

> `Constrained Generation Annoyances, Model Reasoning vs. Token Processing, PDF prevalence on the web, vLLM constrained decoding, Cheap model training options` 


- **Constrained Generation Implementation Woes**: A member expressed frustration with the difficulty of implementing constrained generation that allows the model to 'think' before constraining, noting it requires tinkering with **vLLM internals**.
   - Another member linked to the [vLLM documentation on structured outputs](https://docs.vllm.ai/en/v0.8.2/features/structured_outputs.html), suggesting it shouldn't require much internal tinkering, however the first member was looking for conditional triggering of constraints (e.g. after a `</think>` token).
- **Reasoning vs. Token Processing Capabilities**: One member believes models with sufficient parameters (**235B+**) should inherently possess the capacity to reason while processing tokens, eliminating the need for explicit reasoning steps.
   - In contrast, the discussion's originator detailed a method of interleaving reasoning with prompts e.g. `{chunk}{prompt==describe how to grade chunk}` and then caching KV (key value) pairs.
- **PDF Popularity Stats Debated**: A user cited statistics indicating that **PDFs comprise only 0.6%** of the Common Crawl, while another user questioned if truncated PDFs were accounted for in the calculation.
   - It was clarified that the **0.6%** figure represents the file count, not file sizes.
- **Low-Cost Training Alternatives**: A member asked about affordable options for training a **100 million parameter model on a 100GB dataset**.
   - Suggestions included **Vast.ai** and **RunPod**, with the recommendation to use consumer GPUs like **4090s** or **5090s** due to the setup not being communications-bound.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1458190455278800977)** (18 messages🔥): 

> `Compute Credits, Modal, Lium, LFM 2.5, RL Experiments` 


- ****Compute Credits** are available on Kaggle & Colab VMs**: A member suggested using Kaggle/Colab VMs for small models due to compute constraints and noted that providers like Modal and Lium offer around **$500** in credits, enough for approximately **100M** runs.
   - They also pointed out that Kaggle is perfect for certain use cases.
- **LFM 2.5's **Compute Reduction Claim** Faces Criticism**: A member questioned the compute reduction claim of **LFM 2.5**, arguing it strictly increases compute, referencing [liquid.ai's blog post](https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai).
   - Another member wondered if a custom CUDA kernel leveraging sparsity would work to reduce compute.
- **RL Experiments: Proceed with Caution**: A member advised focusing on simpler subfields like base model training rather than RL, as RL experiments are prone to errors due to foundational knowledge issues.
   - They changed their attitude however and stated the text was AI generated and not welcome.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1458354794187198627)** (1 messages): 

> `lm-evaluation-harness, LLM-as-a-judge, open-sourced LLMs` 


- **"lm-evaluation-harness" Seeking Judge-Mental Support?**: A member inquired whether ["lm-evaluation-harness"](https://github.com/EleutherAI/lm-evaluation-harness) supports evaluations using **LLM-as-a-judge** (e.g. with open-sourced LLMs).
- **No further discussion on this topic**: Currently, there is no information about this topic.


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1458311524731981946)** (1 messages): 

> `Sora ripoffs, Ghibli style segment` 


- **Sora accused of "reskinning" existing videos**: A member stated that *a lot of impressive looking things in Sora are just direct ripoffs of existing videos 'reskinned'.*
- **Sora's Ghibli Style Segment reminiscent of Grave of the Fireflies**: One member thought that the spotlighted Ghibli style segment in **Sora** reminds them of a specific scene in **Grave of the Fireflies**.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1458191786559410206)** (3 messages): 

> `GPT-NeoX, LoRA, QLoRA` 


- **Attention Normalization Differences**: A member notes that the default attention normalization behavior changed, as the old behavior normalized across all heads uniformly, while the new behavior normalizes only within each head.
   - They apologized for not reading the response earlier, but seemed to resolve the issue.
- **LoRA/QLoRA fine-tuning support**: A member asked if the repository supports fine-tuning **GPT-NeoX** with **LoRA/QLoRA**.
   - They noted the existence of some scripts ([configs/finetuning_configs/6-9B.yml](https://github.com/EleutherAI/gpt-neox/blob/main/configs/finetuning_configs/6-9B.yml)), but indicated that those are for full-parameter fine-tuning.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1458192090961023108)** (25 messages🔥): 

> `Manus Customer Support, Credit Policy, Improving Manus Efficiency, AI Security Tool (HexStrike MCP), Open Source Plans` 


- **Customer Debates Prompting Ability**: A customer expressed dissatisfaction with Manus, citing unresolved issues and a decision not to compensate for lost credits, despite terms and conditions and that they have moved to alternative platforms.
   - Another user suggested the customer keep trying with different models and to view Manus as *the shit*, to which the support team replied that they are looking into the specific reasons and that it might take some time.
- **Credit Reset Policy Clarified**: A member of the support team clarified that monthly subscription credits need to be used within the subscription period, using the example that [$20 Pro membership gives 4000 monthly credits](https://manus.im/help/credits) to be used before the next month's reset.
   - The support team offered to further verify the user's specific subscription status and account details, *for example if you purchase a $20 Pro membership on January 1st, you will receive 4000 monthly credits, which need to be used before February 1st*.
- **Psychologist Suggests Efficient Manus Use**: A psychologist suggested focusing on discussions about what went wrong with Manus usage, referencing a [magnum opus](https://fwoxkyoz.manus.space/) to help users improve efficiency, because this is the key towards an AI Tool which improves upon feedback.
   - The psychologist mentioned iterative teaching of Manus via knowledge, where Manus remembers tasks and asks for confirmation before using credits.
- **HexStrike MCP Connectivity Issue Explained**: A user described an issue with an **AI security tool (HexStrike MCP)** hosted on a local virtual machine and an **AI client (Manus)**, explaining that the AI client could not properly resolve the hostname.
   - The user temporarily used **ngrok** to expose the local service through a public HTTPS endpoint, seeking to understand if moving the **MCP server to a VPS** with a public IPv4 address would resolve the connectivity issue and allow proper OAuth flow and SSE connection.
- **Open Source Plans Inquired**: A member inquired about plans to **open source old parts of Manus** and contribute to new initiatives.
   - Another member suggested posting the issue in the **Manus Api Channel** for Ivan to review.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1458412867328938058)** (20 messages🔥): 

> `Emily slop account, Kimi and Deepseek comparisons to Alibaba, Kimi K2 vs Chinese Models, Qwen performance` 


- **Emily is a Slop Account**: A member stated that **Emily** is a *slop account* mostly.
   - Another member asked if it's fair to expect a ship before the Chinese new year.
- **Alibaba QW equals AGI?**: A member remarked that, at least in English, **Alibaba’s QW** might as well be **AGI** compared to **DeepSeek** and **Kimi**.
   - They were unsure if performance is drastically different for **DeepSeek** and **Kimi** in English versus Chinese.
- **Kimi K2 Excels in Both Languages**: A member claimed that **Kimi K2** performs great in both languages and has the highest score in **EQ bench** [https://eqbench.com/](https://eqbench.com/).
   - They explained that the main difference in these models, at least for them, is that **Kimi K2** performs significantly better at creative writing and overall conversations compared to other Chinese models.
- **Qwen Performance Not So Great**: A member shared that their experience with **Qwen3** model variants has not been so great.
   - They explained that they are *fine* at most tasks, sometimes fail simpler ones, and suffer at conversations or creative writing.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1458300365337399306)** (8 messages🔥): 

> `tinygrad vs competitors, tinygrad internal problems, linearizer bounty` 


- **Comparing tinygrad with Competitors**: A new member inquired about what makes **tinygrad** better than its competitors and the major internal problems the dev team is trying to solve.
   - Another member suggested looking at [tinygrad's thesis](https://geohot.github.io//blog/jekyll/update/2025/07/06/can-tinygrad-win.html) and the open-source discussions in the weekly meetings for answers.
- **Unclaimed bounty on Linearizer**: A member asked if the bounty for *Replacing scheduler with linearizer, preserving GPU speed* is still unclaimed, even though there's a potentially ready [PR](https://github.com/tinygrad/tinygrad/pull/13780).
   - Another replied that submitting a working PR first might secure the bounty, and **George Hotz** might split the reward to encourage progress.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1458632240815669259)** (2 messages): 

> `VFIO=1, AMD Radeon RX 9070XT, tinygrad error` 


- **VFIO=1 Triggering TypeError with AMD Radeon RX 9070XT**: A user reported a **TypeError** when running `examples.benchmark_onnx` with `VFIO=1` on a Linux laptop with an **AMD Radeon RX 9070XT**, whereas it runs correctly without setting `VFIO=1`.
   - The error arises from a `NoneType` object not being callable within the `tinygrad/runtime/support/c.py` file during an `ioctl` call, as detailed in the [provided log](https://cdn.discordapp.com/attachments/1070745817025106080/1458632361758425098/tinygrad_vfio_no_iommu.log?ex=696058bf&is=695f073f&hm=156caa091597e59aaaf338b4e228a70a3d523b440e6d5ce6fb1e909cad59e138&)
- **VFIO=0 is working Correctly**: The user further clarifies that when VFIO=0, `examples.benchmark_onnx` runs as expected without errors.
   - This suggests the problem is triggered when `VFIO=1`.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1458207611739574283)** (8 messages🔥): 

> `mTLS Implementations in MCP, MCP Instructions Documentation` 


- ****mTLS Magic**: Seeking MCP Interoperability Insights!**: A member is exploring **mTLS** implementations to enhance **MCP's** interoperability within enterprise environments and is seeking the best channels for contribution discussions.
   - A member recommends starting with the <#1360835991749001368> channel, noting that authentication working groups can provide insights on ongoing related projects.
- ****Docs Drought**: MCP Instruction Documentation Dearth!**: A member inquired about documentation for **MCP instructions**, noting its absence.
   - Another member pointed out that it's part of the [server's initialization response](https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle#initialization) and linked [a blog post](https://blog.modelcontextprotocol.io/posts/2025-11-03-using-server-instructions/) as the closest available resource, also creating [an issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/2060) for rolling some of this into the official docs.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1458326753717125355)** (2 messages): 

> `brave comments, ironic commentary` 


- **Ironic comment generates sarcastic response**: Someone commented *someone had to say it* after a message.
   - Another person replied *so brave*.
- **Sarcasm in chat**: Two members exchanged sarcastic comments.
   - This may indicate strong feelings about a topic.


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/)** (1 messages): 

ash_blanc: https://www.alphaxiv.org/abs/2601.01569
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1458512384133173350)** (2 messages): 

> `DAO projects, AI systems, Model Pipelines, Productionizing LLM features, Inference Cost Strategy` 


- **Dev Seeks New DApp Adventures**: A dev with experience in **DAO**, **marketplace**, and **DApp** projects is looking to get involved in something new with a solid vision and long-term focus.
   - They're happy to chat, contribute, or just jam on ideas, bringing battle scars from governance, tooling, and usability projects.
- **AI Engineer Stands Ready to Untangle Model Pipelines**: An AI Engineer highlights their experience building real AI systems, from **training** and **fine-tuning models** to stitching together **retrieval, agents,** and **infra** at scale.
   - They prefer rolling up their sleeves and shipping, offering help with untangling a model pipeline, productionizing an LLM feature, or figuring out an inference cost strategy.


  

---


---


---

