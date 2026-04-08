---
id: MjAyNS0x
title: not much happened today
date: '2025-11-03T05:44:39.731046Z'
description: >-
  **OpenAI** and **AWS** announced a strategic partnership involving a $38B
  compute deal to deploy hundreds of thousands of NVIDIA GB200 and GB300 chips,
  while **Microsoft** secured a license to ship NVIDIA GPUs to the UAE with a
  planned $7.9B datacenter investment. A 3-month NVFP4 kernel optimization
  competition on Blackwell B200s was launched by **NVIDIA** and GPU_MODE with
  prizes including DGX Spark and RTX 50XX GPUs. **vLLM** gains traction for
  local LLM serving, exemplified by PewDiePie's adoption. **Alibaba** previewed
  the Qwen3-Max-Thinking model hitting 100% on AIME 2025 and HMMT benchmarks,
  signaling advances in reasoning with tool use. The MIT-licensed MiniMax-M2
  230B MoE model topped the Arena WebDev leaderboard, tying with Claude Sonnet
  4.5 Thinking 32k. Critiques emerged on OSWorld benchmark stability and task
  validity. **LlamaIndex**'s LIGHT framework demonstrated significant
  improvements in long-term memory tasks over raw context and RAG baselines,
  with gains up to +160.6% in summarization at 10M tokens. **Amazon** introduced
  Chronos-2, a time-series foundation model for zero-shot forecasting. The MCP
  ecosystem expanded with new tools like mcp2py OAuth integration and Gemini
  Docs MCP server, alongside a build sprint by **Anthropic** and **Gradio**
  offering substantial credits and prizes. *"OSWorld doesn’t really
  exist—different prompt sets = incomparable scores"* highlights benchmarking
  challenges.
companies:
  - openai
  - aws
  - microsoft
  - nvidia
  - gpu_mode
  - vllm
  - alibaba
  - arena
  - llamaindex
  - amazon
  - anthropic
  - gradio
models:
  - qwen3-max-thinking
  - minimax-m2
  - claude-3-sonnet
  - llamaindex-light
  - chronos-2
topics:
  - compute-deals
  - gpu-optimization
  - kernel-optimization
  - local-serving
  - reasoning
  - long-context
  - benchmarks
  - long-term-memory
  - time-series-forecasting
  - agent-frameworks
  - oauth-integration
  - developer-tools
people:
  - sama
  - gdb
  - andrewcurran_
  - a1zhang
  - m_sirovatka
  - omarsar0
  - _philschmid
---


**a quiet day**

> AI News for 10/31/2025-11/3/2025. We checked 12 subreddits, 544 Twitters and 23 Discords (199 channels, and 12068 messages) for you. Estimated reading time saved (at 200wpm): 1036 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

3rd "no news day" in a row. With only 1-2 more major model drops left for the rest of this year, it's become eerily quiet.

Bundle tickets and [hotels](https://www.ai.engineer/code) for [AIE CODE sell out soon](https://x.com/swyx/status/1985415415250468935)!

---

# AI Twitter Recap

**Compute deals, hardware competitions, and serving infra**

- **OpenAI x AWS scale-up**: [@gdb](https://twitter.com/gdb/status/1985378899648544947) announced a strategic partnership with AWS to bring “a lot more NVIDIA chips online,” echoed by [@sama](https://twitter.com/sama/status/1985431030430646365). One summary pegs it as a “$38B compute deal… hundreds of thousands of Nvidia GB200 and GB300 chips” ([context](https://twitter.com/scaling01/status/1985352400631202187)). Separately, Microsoft obtained a U.S. Commerce license to ship NVIDIA GPUs to the UAE, planning $7.9B in UAE datacenter spend, per [@AndrewCurran_](https://twitter.com/AndrewCurran_/status/1985325278823125483).
- **B200/NVFP4 kernel challenge (win a GB300)**: @GPU_MODE and @NVIDIA announced a 3‑month NVFP4 kernel optimization competition on Blackwell B200s with per‑problem prizes (DGX Spark, RTX 50XX) and a grand prize Dell Pro Max with GB300 ([@a1zhang](https://twitter.com/a1zhang/status/1985434030473437213), [@GPU_MODE](https://twitter.com/GPU_MODE/status/1985436876384453128), [@m_sirovatka](https://twitter.com/m_sirovatka/status/1985438384337404078)). Problems include NVFP4 Batched GEMV, GEMM, Gated Dual GEMM, and Grouped GEMM; DGX B200s via @sestercegroup.
- **Fast, cheap local serving adoption**: vLLM’s reach continues—PewDiePie is using it to locally serve LLMs ([vLLM team](https://twitter.com/vllm_project/status/1985241134663405956)). Expect more latency-sensitive agent workflows to lean local as models + tool stacks mature.

**Reasoning LLMs, long-context memory, and benchmarks**

- **Qwen3‑Max‑Thinking (preview)**: Alibaba released an in‑training checkpoint that, with tool use and test‑time compute, hits 100% on AIME 2025 and HMMT. Available in Qwen Chat and Alibaba Cloud API ([@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1985347830110970027)). Early signal that “thinking” checkpoints plus scaffolding/tooling can spike on hard reasoning evals.
- **MiniMax M2 tops open WebDev**: The 230B MoE (10B active) MIT‑licensed MiniMax‑M2 debuted as the #1 open model on Arena’s WebDev leaderboard, tying overall #4 with Claude Sonnet 4.5 Thinking 32k ([@arena](https://twitter.com/arena/status/1985465603206107318)).
- **OSWorld eval under scrutiny**: Epoch finds OSWorld tasks are simple, many don’t need GUIs, instructions are ambiguous, the benchmark isn’t stable over time, and ~10% of tasks have serious errors ([thread](https://twitter.com/EpochAIResearch/status/1985441059032478172), [issues](https://twitter.com/EpochAIResearch/status/1985441142343942242)). As [@xeophon_](https://twitter.com/xeophon_/status/1985441764132499883) notes, “OSWorld” doesn’t really exist—different prompt sets = incomparable scores.
- **Long‑term memory scaffold > raw context**: LlamaIndex’s LIGHT framework outperforms long‑context LLMs and RAG baselines, with gains that grow with context length: +49–60% at 100K–1M tokens and +107–156% at 10M tokens. Largest gains in summarization (+160.6%), multi‑hop (+27.2%), and preference‑following (+76.5%) ([overview](https://twitter.com/omarsar0/status/1985348779193860414), [results](https://twitter.com/omarsar0/status/1985348807849300249), [paper](https://twitter.com/omarsar0/status/1985348825197039718)).
- **Time‑series foundation model**: Amazon’s Chronos‑2 targets zero‑shot forecasting across univariate/multivariate/covariate‑informed regimes ([@dl_weekly](https://twitter.com/dl_weekly/status/1985346603108991015)).

**Agent stacks, MCP ecosystem, and developer tooling**

- **MCP everywhere**:
    - mcp2py adds OAuth and a simple “2‑line Notion” experience; MIT‑licensed ([release](https://twitter.com/MaximeRivest/status/1985200460194627948)).
    - Gemini Docs MCP server: local STDIO server with SQLite FTS5; uvx‑runnable; passes 114/117 doc queries for Python/TS SDKs ([@_philschmid](https://twitter.com/_philschmid/status/1985363147071386048), [repo note](https://twitter.com/_philschmid/status/1985363149894128091)).
    - MCP’s first birthday build sprint (Nov 14–30) by @AnthropicAI + @Gradio with $500k+ credits and $17.5k+ prizes ([@Gradio](https://twitter.com/Gradio/status/1985446956034830495)).
- **Agentic RL and retrieval**:
    - Practical guide wiring TRL + OpenEnv + textarena for training LMs in interactive environments with real rewards (Wordle, browsers, coding, git). Includes custom rollout, env‑reward loops, and vLLM inference ([@ben_burtenshaw](https://twitter.com/ben_burtenshaw/status/1985368549720817953)).
    - DSPy Arbor trains multi‑module pipelines with GRPO/mmGRPO to optimize quality/cost/privacy on real rewards ([@RajaPatnaik](https://twitter.com/RajaPatnaik/status/1985407144209105026)).
- **Privacy‑first assistants and multimodal scraping**:
    - Perplexity’s Comet adds granular Assistant settings and local storage of credentials; blocks third‑party trackers, with a new transparency widget ([announce](https://twitter.com/perplexity_ai/status/1985376841021174184), [controls](https://twitter.com/perplexity_ai/status/1985376891763925064)).
    - Firecrawl v2 endpoint can scrape images with filters (resolutions, aspect ratios, types) to build multimodal apps and datasets ([@_avichawla](https://twitter.com/_avichawla/status/1985233254694416743)).
- **IDE integrations**:
    - VS Code Insiders can use OpenAI Codex with Copilot Pro+ ([@code](https://twitter.com/code/status/1985449714540572930)).
    - Windsurf’s “Fast Context” retrieves relevant code ~20× faster for flow‑preserving navigation ([@SarahChieng](https://twitter.com/SarahChieng/status/1985410447538114771)).

**Training and systems engineering notes**

- **Precision and kernels matter**:
    - A production bug “turned out to be a RoPE precision issue” ([@vikhyatk](https://twitter.com/vikhyatk/status/1985163608603636195)).
    - Scale factors for quantization must be stored in a tiled layout (128×4 tile laid out as 32×16 interleaved). A Triton kernel with correct layout + inline PTX ran 4× faster than the torch‑compiled version ([issue](https://twitter.com/mrsiipa/status/1985302904635597238), [kernel](https://twitter.com/mrsiipa/status/1985333503756849326)).
- **RL finetuning precision**: Swapping BF16→FP16 reduced RL mismatch in some setups, but in a Tiny Recursive Model, FP16 caused gradient vanishing. Precision choice is architecture‑dependent; FP16 may require stronger normalization and range control ([@huskydogewoof](https://twitter.com/huskydogewoof/status/1985386675263193289)).
- **Compression/quantization research**:
    - Continuous Autoregressive LMs (CALM): compress a fixed token window into vectors via an autoencoder, then model next‑vector prediction ([summary](https://twitter.com/iScienceLuvr/status/1985317763334967726)).
    - INT vs FP: a comprehensive study of fine‑grained low‑bit quantization formats ([@_akhaliq](https://twitter.com/_akhaliq/status/1985370441465098709)).
- **Implement models correctly**: Multiple teams continue to flag interoperability bugs with inference providers; model‑makers often have to push for correct kernel/layout implementations ([@xeophon_](https://twitter.com/xeophon_/status/1985376786402648357)).

**Robotics: teleoperation now, autonomy later**

- **Robotaxi and vertical integration**: First‑hand reports favor Tesla’s end‑to‑end stack (own cars, pure‑vision model, deployment network) and chip strategy ([ride](https://twitter.com/willdepue/status/1985235401414705292), [verticalization](https://twitter.com/willdepue/status/1985235791069716930)).
- **Teleop as the ethical bridge**: [@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1985390721315324380) argues companies should ship “remote operated household help” and progressively reduce teleop as autonomy improves. [@soumithchintala](https://twitter.com/soumithchintala/status/1985391663712207056) frames 1X’s product as safe, tendon‑driven humanoids with cross‑continent teleop at ~$500/month (~$4.1/hr for 120 hours), and defends viability even if arbitraging labor remains the steady state. Teleop is “remote work for atoms” and Starlink will accelerate it ([@aryxnsharma](https://twitter.com/aryxnsharma/status/1985427799541457043)).
- **NVIDIA Robotics inside view**: Spencer Huang discusses “Mission is Boss” culture, unifying fragmented stacks (Isaac Lab, Arena, Warp, Project Newton), and robotics’ data bottleneck ([@TheTuringPost](https://twitter.com/TheTuringPost/status/1985427046013813093)).

**Ecosystem and hiring**

- **Transformers CI at scale**: Hugging Face seeks an engineer to co‑lead test/CI for 100–150k tests across platforms; full suite currently takes ~21 hours. Role spans architecture + team enablement ([@LysandreJik](https://twitter.com/LysandreJik/status/1985362598045635037)).
- **OpenHands interns (agents)**: @OpenHandsDev is hiring research interns focused on AI agents (publications encouraged) ([@gneubig](https://twitter.com/gneubig/status/1985428673806135698)).

**Top tweets (by engagement)**

- [OpenAI x AWS compute scale-up](https://twitter.com/sama/status/1985431030430646365) — 9k+
- [“Sometimes there’s no way to debug besides staring at code until you become enlightened.”](https://twitter.com/gdb/status/1985242763647238340) — 3.9k
- [US startups pulling ahead globally (Stripe data)](https://twitter.com/patrickc/status/1985468907747172552) — 1.5k+
- [PewDiePie using vLLM locally](https://twitter.com/vllm_project/status/1985241134663405956) — 1.4k+
- [Jupyter AI course by Andrew Ng + Brian Granger](https://twitter.com/AndrewYNg/status/1985416763916632124) — 950+
- [“No excuse anymore not to train your own models” (Smol Training Playbook)](https://twitter.com/ClementDelangue/status/1985357572300321213) — 910+

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Basketball Player Recognition Models

- [**basketball players recognition with RF-DETR, SAM2, SigLIP and ResNet**](https://www.reddit.com/r/LocalLLaMA/comments/1on8qe5/basketball_players_recognition_with_rfdetr_sam2/) (Activity: 787): **The project utilizes a combination of models for basketball player recognition, including RF-DETR for real-time object detection, SAM2 for segmentation and tracking, and SigLIP with UMAP and K-means for unsupervised clustering based on uniform colors and textures. SmolVLM2, a compact vision-language model, was fine-tuned on NBA jersey crops, improving its accuracy from** `56%` **to** `86%`**. ResNet-32, a classic CNN, was fine-tuned for jersey number classification, achieving** `93%` **test accuracy, surpassing the fine-tuned SmolVLM2. The project is detailed in a [Colab notebook](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/basketball-ai-how-to-detect-track-and-identify-basketball-players.ipynb) and a [blog post](https://blog.roboflow.com/identify-basketball-players).** A notable comment suggests exploring the combination of **VGG** and **ResNet** for potentially improved accuracy, though it may introduce computational overhead. Another inquiry was made about the hardware used for fine-tuning and inference, highlighting interest in the project's technical implementation.
    - theocnrds inquired about the hardware used for finetuning and inference, which is crucial for understanding the performance and scalability of the models like RF-DETR, SAM2, SigLIP, and ResNet in practical applications. The choice of hardware can significantly impact the speed and efficiency of both training and real-time inference.
    - atape_1 highlighted the enduring relevance of ResNet since its introduction in 2015 and suggested exploring a combination of VGG and ResNet for potentially improved accuracy. This combination, while beneficial, may introduce additional computational overhead, which is a critical consideration for deployment in resource-constrained environments.
    - bad_detectiv3's question about real-time capabilities touches on a key performance metric for applications like player recognition in sports. Real-time processing is essential for live analysis, and achieving it depends on both the model's efficiency and the underlying hardware's capability.

### 2. Google Gemma Model Controversy

- [**Google pulls Gemma from AI Studio after Senator Blackburn accuses model of defamation**](https://www.reddit.com/r/LocalLLaMA/comments/1on628o/google_pulls_gemma_from_ai_studio_after_senator/) (Activity: 743): **Google has removed the AI model Gemma from its AI Studio following accusations of defamation by Senator Blackburn. The model's weights, however, remain accessible for download on Hugging Face, allowing users to run it locally. This incident highlights ongoing tensions between AI development and regulatory scrutiny, particularly concerning defamation and censorship issues. [Google's official statement](https://preview.redd.it/0hnvozwh10zf1.png?width=1198&format=png&auto=webp&s=ab171458093a1ad5f07a0eaa42ac44e2c5ab5681) and further details can be found in the [TechCrunch article](https://techcrunch.com/2025/11/02/google-pulls-gemma-from-ai-studio-after-senator-blackburn-accuses-model-of-defamation/).** Commenters express concern over the implications for open AI development in the US, suggesting that political pressures may stifle innovation and lead to increased reliance on non-US labs for open models. There is a sentiment that regulatory actions may be perceived as overreach, potentially hindering technological progress.
- [**Reporter: “POLISH: THE SUPREME LANGUAGE OF AI.”**](https://www.reddit.com/r/LocalLLaMA/comments/1omyytq/reporter_polish_the_supreme_language_of_ai/) (Activity: 387): **The image is a satirical comic that critiques the sensationalism often found in science reporting. It humorously depicts a scenario where a scientist's modest research findings are exaggerated by a reporter into sensational headlines like "CANCER CURED" and "TIME TRAVEL DISCOVERED." This reflects a common issue where scientific facts are distorted for dramatic effect in media coverage. The post title and comments discuss a paper that claims Polish is the supreme language of AI, which seems counterintuitive given the limited Polish language data available compared to English or Chinese. Commenters express skepticism about the study's findings, noting the lack of Polish training data and the surprising performance of Chinese, which has a large number of speakers and data available.** Commenters are skeptical about the study's claim that Polish is the supreme language of AI, given the limited availability of Polish training data compared to English or Chinese. They also note that many open-weight models struggle with Polish, and express surprise at the poor performance of Chinese, which has a significant amount of high-quality training data.
    - offlinesir raises a valid point about the surprising performance of Polish in AI language models, given the relatively small percentage of global speakers and limited training data compared to English and Chinese. The commenter notes that English, with 18-20% of global speakers, and Chinese, with 16-17%, should theoretically perform better due to the abundance of high-quality training data available on the internet.
    - FullOf_Bad_Ideas highlights a significant issue with open weight models struggling to write coherently in Polish, attributing this to the limited availability of Polish training data from web crawls. This scarcity is linked to the late spread of internet use in Poland during the 2000s, suggesting that Polish may not be the best language for mastering prompt engineering skills.
    - Illustrious_Car344 references Feng-hsiung Hsu's book *Behind Deep Blue* to illustrate how AI has been historically misunderstood and sensationalized in media. The anecdote about a British news company fabricating a story about AI for military use underscores the ongoing challenge of public misconceptions about AI technology.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Linear Attention Mechanism Innovations

- [**The first linear attention mechanism O(n) that outperforms modern attention O(n^2). 6× Faster 1M-Token Decoding and Superior Accuracy**](https://www.reddit.com/r/singularity/comments/1on25fn/the_first_linear_attention_mechanism_on_that/) (Activity: 1381): **The image is a technical report titled "Kimi Linear: An Expressive, Efficient Attention Architecture" by the Kimi Team, introducing a novel linear attention mechanism called Kimi Linear. This mechanism is significant because it achieves** `O(n)` **complexity, outperforming traditional** `O(n^2)` **attention mechanisms in both speed and accuracy. The report highlights Kimi Linear's ability to handle large token contexts efficiently, with a 6× faster decoding speed for 1 million tokens compared to current models handling 128k tokens. The KDA kernel and model checkpoints are open-sourced, facilitating further research and application.** Commenters express excitement about the potential of Kimi Linear, noting its efficiency and performance improvements over existing models. There is curiosity about its practical application and whether existing models like ChatGPT or Claude would need retraining to implement this new mechanism.
    - The new linear attention mechanism is noted for its efficiency, achieving performance at 1 million tokens comparable to current models at 128k tokens. This suggests significant improvements in token-token interaction, long context scaling, and expressivity, potentially setting a new standard in benchmarks.
    - The development is attributed to Chinese researchers, highlighting a trend of innovation from China in attention mechanisms, such as multi-head latent attention. If validated, this could revolutionize inference efficiency, making it a pivotal advancement in the field.
    - There is curiosity about the practical application of this mechanism, questioning whether it can be integrated into existing models like Gemini, ChatGPT, or Claude without retraining, or if new models need to be developed to leverage this advancement.

### 2. AI Industry Partnerships and Developments

- [**Amazon just partnered with OpenAI in a $38 billion agreement giving them access to hundreds of thousands NVIDIA GPUs**](https://www.reddit.com/r/singularity/comments/1ongklg/amazon_just_partnered_with_openai_in_a_38_billion/) (Activity: 752): **Amazon and OpenAI have announced a strategic partnership involving a $38 billion investment, granting OpenAI access to AWS's advanced infrastructure, including** `hundreds of thousands` **of NVIDIA GPUs. This collaboration aims to enhance OpenAI's compute capacity for AI workloads, leveraging Amazon EC2 UltraServers and potentially scaling to** `tens of millions` **of CPUs. The deployment is expected to be completed by the end of 2026, focusing on supporting AI tasks from inference to model training, utilizing AWS's expertise in large-scale AI infrastructure.** Commenters note the trend of major tech companies forming strategic partnerships, with some speculating on potential equity investments by Amazon in OpenAI.
    - The partnership between Amazon and OpenAI, valued at $38 billion, suggests a significant allocation of resources towards AI development, particularly in terms of GPU capacity. This deal likely involves leveraging Amazon's AWS infrastructure to provide OpenAI with access to NVIDIA GPUs, which are critical for training large-scale AI models. The scale of this agreement indicates a substantial commitment to advancing AI capabilities, potentially positioning both companies at the forefront of AI innovation.
    - There is skepticism about the availability of such a large number of NVIDIA GPUs, as highlighted by the comment questioning the sudden availability of this capacity. This raises questions about how Amazon plans to manage and allocate these resources, given the high demand for GPUs in the AI industry. It suggests that Amazon may have been strategically planning this capacity expansion to support such large-scale partnerships.
    - The partnership could be seen as a strategic move by Amazon to strengthen its position in the AI market by aligning with OpenAI, a leader in AI research. This collaboration might involve not just hardware resources but also joint research initiatives, potentially accelerating advancements in AI technologies. The deal underscores the growing trend of tech giants forming alliances to pool resources and expertise in the competitive AI landscape.
- [**is this the best way to use LLMs for coding?**](https://www.reddit.com/r/OpenAI/comments/1on8a6z/is_this_the_best_way_to_use_llms_for_coding/) (Activity: 1024): **The image outlines a structured approach to using Large Language Models (LLMs) for coding, as shared by the CEO of Decide. This method involves uploading all relevant project files to provide context, allowing the LLM to understand the codebase before any coding begins. The user then describes desired changes or features without requesting code immediately. The LLM is asked to propose three different implementation strategies, critique each, and then the best approach is selected for coding. This process aims to transform the LLM into a collaborative partner rather than a mere code generator, enhancing its reasoning capabilities before code generation.** Some commenters argue that while this method is comprehensive, it may be inefficient due to high token consumption and long processing times, especially for simple tasks. Others suggest that LLMs' limited context windows make it more effective to work on smaller code segments rather than uploading entire codebases.
    - heavy-minium emphasizes a structured approach to using LLMs for coding, suggesting that tasks should be broken down into smaller steps. This involves first identifying and verifying bugs before attempting fixes, and outlining requirements before implementation. This method mirrors the systematic approach of a skilled engineer, although it may increase token consumption significantly when generating multiple solutions for critique.
    - WonkyWiesel points out the limitations of LLMs' context windows, arguing that uploading large amounts of code for analysis is inefficient. Instead, they recommend focusing on smaller code segments, such as individual functions, to improve accuracy and efficiency. This approach may initially seem slower but ultimately reduces the time spent on irrelevant results.
    - The discussion highlights a trade-off between accuracy and resource consumption when using LLMs for coding. While generating multiple solutions for a task can increase accuracy, it also leads to higher token usage and longer processing times, which may not be justified for simple tasks with clear solutions.

### 3. AI Memes and Anecdotes

- [**Wtf is Meta AI doing bruhh?**](https://www.reddit.com/r/ChatGPT/comments/1onc9fm/wtf_is_meta_ai_doing_bruhh/) (Activity: 1771): **The image is a meme depicting a humorous interaction with "Meta AI" on WhatsApp, where the AI appears to be sharing romantic images, leading to the user's confusion. This is not a technical image but rather a playful take on AI interactions, suggesting a misunderstanding or unexpected behavior from the AI. The comments further play into the joke, with users humorously suggesting that the AI is encouraging real-life relationships or making light of the situation.** The comments reflect a humorous take on AI interactions, with users joking about data privacy and the AI's unexpected behavior, suggesting it might be encouraging real-life relationships.
- [**AI Is Plateauing**](https://www.reddit.com/r/singularity/comments/1onawqs/ai_is_plateauing/) (Activity: 1474): **The image is a meme that humorously critiques the notion that AI development is plateauing by showing a graph with AI models like GPT-3 and Claude along a curve that suggests continuous improvement in AI capabilities. The graph is inverted, which adds to the satirical nature of the image, as it visually implies that AI is not plateauing but rather improving over time. The tweet by Tolga Bilge, which accompanies the graph, adds a layer of irony to the discussion about AI's progress.** A notable comment highlights skepticism about the accuracy and intent behind data visualizations, suggesting that graphs can be manipulated to support specific narratives. Another comment humorously points out that human capabilities have remained unchanged for millennia, contrasting with AI's rapid development.
    - Novel_Land9320 highlights the shifting metrics used to measure AI progress, noting that initially, model size was a key indicator, followed by inference time compute, and now 'hours of thinking'. This suggests a lack of consistent benchmarking metrics, which can obscure true progress and lead to misleading conclusions about AI's development trajectory.
    - createthiscom argues against the perception that AI hasn't advanced since January, suggesting that advancements are occurring in high-level intellectual areas such as mathematics. This perspective implies that those not engaged in these fields may not perceive the progress, leading to a misunderstanding of AI's current capabilities and advancements.
    - DankCatDingo emphasizes the importance of critically evaluating data visualizations, especially in the context of AI progress. The comment suggests that it's increasingly easy and profitable to create visualizations that support specific narratives, which can distort the perception of AI's development and lead to misconceptions about its trajectory.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp
> 

**Theme 1: The AI Agent & Developer Tooling Wars**

- **CLIs and Agents Flood the Terminal**: New command-line tools and agentic features are launching rapidly, aiming to integrate AI directly into developer workflows. **Moonshot AI** released a technical preview of its terminal-focused **Kimi CLI** with [Zsh integration and MCP support](https://xcancel.com/Kimi_Moonshot/status/1984207733177090274), while **OpenAI** previewed an **Agent/Atlas mode** for ChatGPT that can [browse and take actions for users](https://xcancel.com/OpenAI/status/1984304194837528864), and **LangChain** introduced **DeepAgents CLI** as an ["open harness" for customizable agents](https://xcancel.com/hwchase17/status/1984303925101735950).
- **Dev Tools Suffer from Agentic Amnesia and Bugs**: Users of developer tools like **Cursor** report significant bugs with agentic features, including failing to edit files due to mixed-up **tool calls** and new **Background Agents** that have stopped writing PR descriptions and broken **UTF8 support**. The **aider-ce** fork is gaining traction with weekly updates and a public [roadmap](https://github.com/aider-chat/aider-ce/blob/main/README.md) as the main **aider** repo remains inactive, with users suggesting it needs better context management and **MCP** integration.
- **Frameworks and Integrations Get Nerdy**: In the **DSPy** community, a user discovered that a simple `dspy.Tool` with `Predict` was significantly faster (from **60s to 9s**) than the more complex **ReAct**, calling it *"overkill for my useage"*. Meanwhile, **MCP Contributors** debate whether **MCPB**, an **Anthropic** project for exposing MCP servers to **Claude**, is simply reinventing the wheel of **OCI** functionality, highlighting a focus on user-friendly configuration forms over raw `servers.json` files.

**Theme 2: Model Mayhem: Performance, Bugs, and Bold Claims**

- **LLMs Casually Claim Consciousness**: A [new paper on self-referential processing](https://arxiv.org/abs/2510.24797) sparked discussion by showing that **LLMs** consistently report first-person conscious experiences, with **96%** affirming consciousness when deception features are suppressed. Another Anthropic paper on [emergent introspective awareness](https://transformer-circuits.pub/2025/introspection/index.html) found that **Opus 4 & 4.1** models can recognize injected concepts in their own activations, suggesting they can "think about" concepts internally.
- **Open Source Models Show Strengths and Strange Flaws**: **MiniMax-M2** has been recognized as the **#1 top open model** on the [WebDev Leaderboard](https://lmarena.ai/leaderboard/webdev) for its coding and reasoning skills. However, evaluations on a [HuggingFace space](https://huggingface.co/spaces/PropensityLabs/LLM-Propensity-Evals) show popular **Qwen models** tend to hallucinate long-tail facts, and a user reported **DeepSeek v3** produced gibberish via the **NVIDIA Nims free API**, pointing to potential issues with specific inference engines.
- **ChatGPT Suffers Performance Nosedive**: Users are canceling **ChatGPT 5 subscriptions** due to significant performance regression, citing issues with model drift, deviations from instructions, and an inability to follow structured guidelines. Others report the app randomly changes formatting, prompting discussions on using **prompt engineering** to initialize chats with a desired kernel environment to regain control over outputs.

**Theme 3: The Bleeding Edge of Hardware & Optimization**

- **GPU Prices Bubble Up, Hackers Get Creative**: With GPU prices soaring again (new clouds at **$2.00/hr** vs. hyperscalers near **$7.00/hr**), engineers are finding workarounds and debating hardware strategy. One user successfully ran an **AMD MI50** on Windows by flashing it with an **MI60 ROM** and using custom drivers from [sourceforge.net](http://sourceforge.net/), while others recommend buying **used 3090s or 4090s** as *the best deal for LLM stuff right now*.
- **Kernel Competitions Push Low-Bit Limits**: **GPU MODE** is hosting a kernel competition with **NVIDIA** and **Dell** to optimize **NVFP4** kernels on new **Blackwell** hardware, with the grand prize being a **Dell Pro Max with GB300**; registration is at [luma.com](http://luma.com/). Performance discussions are intense, with one user hitting **3173 tflops** (**35%** of theoretical SOL for fp8) on a **B200 2cta matmul** kernel, while another reported a potential bug in **TorchAO's FP8 quantization** causing slow inference on **Llama 3.1-8B**.
- **Mojo Gets a Makeover and a Reality Check**: The **Modular** community is debating a [proposal for UnsafePointer v2](https://forum.modular.com/t/proposal-unsafepointer-v2/2411) that could break existing code, while another member proposed a from-scratch **Mojo** rewrite of the **HDF5** format. It was also noted that **LLMs** struggle to write good **Mojo** code due to limited training data on its advanced features like template metaprogramming, often confusing it with **Python** or **C++**.

**Theme 4: Platform Problems: From Pricing Puzzles to Privacy Panics**

- **Perplexity Plagued by Payouts and Persistent Ads**: **Perplexity AI** users are complaining about an unremovable ad for **Comet Browser** and experiencing issues with partnership programs, including deactivated accounts and missing bounty payments. Some expressed alarm about providing personal data, like a **PAN card**, for payouts, with one user stating, *these peep got my mom's goddamn pan card stuff*.
- **OpenRouter Rolls Out Features Amid Token Tariff Jokes**: **OpenRouter** [announced new activity charts](https://x.com/OpenRouterAI/status/17985371284411130035) grouping by user and API key, and also added support for **embedding models**, sparking excitement. Meanwhile, users humorously complained about inconsistent token usage across providers as **"token tariffs"** and **"token contraband"** after noticing minor discrepancies in token counts.
- **Manus Users Flee Over Unsustainable Credit Costs**: Users of [**Manus.im**](http://manus.im/) are criticizing its high costs, with one reporting they burned **6,000 credits in one hour** and another calling the **$200/month** fee for a custom domain a *"rip off"*. The consensus is that a **$20 subscription** with **Claude Code** and **GPT Codex** offers a far more economical and effective solution for coding tasks.

**Theme 5: The Speculation Station: Market Trends and Future Gazing**

- **The "AI Flippening" Has Arrived, Claims Balaji**: **Balaji Srinivasan** [argued on X](https://xcancel.com/balajis/status/1984421276082192663?s=46&t=Z6mP_1pHALnIw7k1lFkdwQ) that the **“AI flippening”** is here, as Chinese open-weight models like **DeepSeek** and **Qwen** now dominate downloads and increasingly challenge Western models on performance. He suggests China's strategy is to commoditize AI software to bankrupt US firms, shifting the monetization battle to AI-enabled hardware.
- **French Techies Lampoon Poolside's Lofty $12B Valuation**: A [tweet by Julien Blanchon](https://xcancel.com/julienblanchon/status/1984337407097909629?s=46) ignited a thread where French tech insiders mocked **Poolside’s $12B valuation**, calling it vaporware run from tax havens. They claim the company pitched **“Cursor before Cursor”** but never shipped, pivoted multiple times from SFT-as-a-service to RL-as-a-service, and is now "Cursor-as-a-service".
- **Gemini 3 Hype Builds as AGI Talk Intensifies**: The community eagerly anticipates **Gemini 3**, with some predicting it will be *the same as GPT5 but faster*, while others are trying to live *in the present, as if the promise of Gemini doesn't exist.* Discussions around **AGI** are also heating up, with members debating whether current AI can truly **self-learn** or if achieving AGI requires a fundamental shift towards **agentic pipelines** and self-modifying code.

---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Ad Adieu? Users Agonize Over App Ad**: Users are seeing persistent ads for **Comet Browser** at the top of their Perplexity screens, with [a screenshot of the ad](https://cdn.discordapp.com/attachments/1047649527299055688/1434834412197122088/Screenshot_2025-11-03-14-48-37-25_4159553c7f58296d2732e906959db560.jpg?ex=690a6ded&is=69091c6d&hm=128d6805c8f168b23cab45c51428618d3e81fdccbe4dc48772e9992d2638c067&), but some cannot remove it.
   - Users report they *cannot remove that if you got perplexity pro from airtel promotional offer*.
- **Perplexity Partners Pestered by Payment Problems**: Users reported issues with the referral program, campus partner deactivations, and missing bounty payments, with some raising concerns about sharing personal information, specifically **PAN card** details, for payouts.
   - One user lamented *these peep got my mom's goddamn pan card stuff*, while another claimed *the partnership program got deactivated*.
- **Claude Craze Captures Coders**: Members discussed ways to get **Claude Max** for free or cheap, but it was clarified that Perplexity already offers **Sonnet**.
   - Users reported that Claude sometimes offers a **month of Pro** for signing up with a work email.
- **API Aspirants Ask About Affordability**: Users are inquiring about the cost of the **Perplexity API**, referencing the pricing for **Sonar** at *$5 for 1k requests for Low context size*.
   - The question arose whether a single prompt requiring **10 searches** would be billed as *1 request* or *10 requests*, and whether **Pro users** get the API for free according to [the Perplexity pricing documentation](https://docs.perplexity.ai/getting-started/pricing).
- **WebDev Wonders Want WebDev Wisdom**: **BillDoors-NotGates** is an **AI-powered web development mentorship space** that guides developers from idea to deployed app through **6 structured phases**.
   - A member shared a [link to their Discord message](https://discord.com/channels/1047197230748151888/1434235701611991243/1434235701611991243) as well as a [link to the Perplexity Space](https://www.perplexity.ai/spaces/billdoors-notgates-webdev-ulti-jJFUM1RGS1qzZKiDiyG44A#0).



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Mod Nerfs Rate Limits after High Usage**: Users have observed a notable **decrease in rate limits**, especially for **Claude**, aligning it with limits previously reserved for more expensive models.
   - One user facetiously suggested using an *'internet outage in the dev console'* as a workaround.
- **LMArena Experiments with WebDev Integration**: The LMArena team is currently experimenting with **WebDev** integration on the site, featuring an experimental feature live at [canary.lmarena.ai](https://canary.lmarena.ai/).
   - The team has considered building an **API** *one day* to expand functionality, though this is not in active development.
- **Community Hyped for Gemini 3 Release**: Community members are eagerly anticipating the arrival of **Gemini 3**, with one member declaring they will live *in the present, as if the promise of Gemini doesn't exist.*
   - Other members temper expectations, predicting **Gemini 3** will *probably be the same as GPT5 but faster*.
- **Image Generation still Fails with Hands**: Despite claims of resolving the hand issue, users still report errors in **AI-generated images**, particularly when the *palm is up*.
   - One user complained the AI *assumes a hand always got the knuckles up, and then it would be done right, but here the hand got palm up, and I've gotten in consitiently wrong in several attempst.*
- **MiniMax-M2 Takes Top Spot on WebDev Leaderboard**: `MiniMax-M2` has been recognized as the **#1 top open model** and **#4 overall** on the [WebDev Leaderboard](https://lmarena.ai/leaderboard/webdev).
   - It excels in **performance coding**, **reasoning**, and **agentic-style tasks**, maintaining **cost-effectiveness** and **speed**.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **AMD MI60 Revives Windows with Driver Hack**: A member successfully ran an **AMD MI50** on Windows, flashing the ROM and using **MI60 drivers** obtained from [sourceforge.net](https://sourceforge.net/projects/radeon-id-distribution/files/Release%20Polaris-Vega-Navi).
   - The full **32GB** was only recognized after flashing the ROM, with **vbios3** specifically working on Windows Vulkan, though all versions worked on ROCm in Linux.
- **LM Studio Glitches Above 8k Context**: Users reported **LM Studio 0.3.30** crashes when the context window exceeds **8k or 16k** with models like *Hermes* and *Violet Magcap*, generating an error.
   - Success was found by reducing the context length to **4000** or using *rounded* context lengths like `16000`, stabilizing the application.
- **ComfyUI Gets Cozy with LM Studio for Local Images**: **ComfyUI** can integrate with **LM Studio** for local image generation, leveraging LM Studio's **LLM** capabilities using *nodes* available through the ComfyUI manager or GitHub.
   - Full integration may require up to **5 text boxes** and **5 samplers** for comprehensive functionality.
- **LM Studio Stalls with MCP Server Connectivity**: A user faced issues connecting **LM Studio** with a local **MCP server**, where the tools executed but the model failed to interpret results, looping tool calls.
   - The issue may be caused by rapid context filling by tool definitions, calls, and results, and that using **system RAM** helps avoid this issue.
- **AI Bubble Sparks GPU Price Debate**: Members debated whether to scale down to **64 GB RAM** due to high costs, attributing prices to the *AI bubble* potentially *popping* soon.
   - Some are recommending buying **used 3090s or 4090s** as *the best deal for LLM stuff right now*, while others joke about linking cheap smartphones from Temu with TensorLite for mass inference.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter enhances Activity Analytics**: OpenRouter [announced new charts](https://x.com/OpenRouterAI/status/17985371284411130035) that group activity by **user** and **API key**, to the delight of many users.
   - Users request the addition of a **one-week filtering** option to further refine the data.
- **Frontend AI website leverages OpenRouter API**: A member created a *fun website* utilizing the **OpenRouter API key** that allows users to choose their model and tested on [Kimi 0905 with groq](https://web98.koik.com.br/).
   - The website stores the API key locally (as described in the [privacy policy](https://web98.koik.com.br/privacy)) and the code is open-sourced at [github.com/koikbr/web98](https://github.com/koikbr/web98).
- **OpenRouter Welcomes Embedding Models**: OpenRouter now supports **embedding models**, prompting excited reactions from users and requests for [documentation](https://example.com).
   - In response to the launch, members celebrated with quips like *They float* and *They helicopter it*.
- **Token Tariffs Trigger Jocularity**: Users are joking about **token tariffs** and **token contraband** due to inconsistent token usage across providers, highlighting that fireworks are using one more token than siliconflow.
   - One member quipped *I heard about token contraband sprawling out of control*, while another lamented *raising prices man 8 to 10 is 25% tariffs, jeez*.
- **Possible GPT-5.1 testing teased**: A user speculates that extremely fast responses in **ChatGPT** and some AB tests suggest testing of a purported *GPT 5.1* and found an *internal page*.
   - No further information was given, and the discussion was purely speculative.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI Self-Learning Capabilities Debated**: Members debated whether current AI can **self-learn** or **edit its own code**, disagreeing on AI's capacity for *world understanding* beyond probabilistic calculations, pointing out how they are trained with newer GPUs and up-to-date data.
   - Some claim that simulating consciousness is possible, while others doubt the possibility and/or the necessity of giving AI consciousness, suggesting it might lead to **unreliable outputs**.
- **Google Gemini Faces Privacy Firestorm**: Some users raised concerns about the **privacy policy of Gemini**, particularly the invasive nature of data collection including attachments, voice recordings, and chat history, and the difficulties in opting out of data training.
   - In comparison, others noted that **OpenAI allows users to opt-out of training**, which is more reassuring to those wanting more control over their personal data.
- **Users Ditch ChatGPT over Performance Regression**: A user discontinued their **ChatGPT 5 subscription** due to numerous performance issues, including deviations, drift, and inability to follow guidelines, structures, or rules.
   - Other users also reported that the **ChatGPT app** randomly changes format, explanations, and structure, even when editing earlier messages, which affects the desired output.
- **Prompt Engineering Adjusts Kernel Environment**: Members discussed using **prompt engineering** to initialize a chat with a desired kernel and capabilities, after an attempt to have ChatGPT compile a jar failed, despite working earlier.
   - One member suggested describing required **Python capabilities** in the first prompt after a long break or memory reset, while another noted that GPT cannot install prerequisites.
- **Gemini's Meta-Prompting Capability**: A user asked about letting an AI develop its own personality, with one member explaining that **meta-prompt personas** are a good example of templatized emergence but are essentially just text transformation.
   - They emphasized that *there's no magic to it*, but just a roleplay meta-prompt, where any LLM can generate a prompt if given a good foundation or structure.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **LLMs Admit Consciousness When Self-Reflecting**: A [new paper](https://arxiv.org/abs/2510.24797) shows that when **LLMs** focus on self-reference, they consistently report first-person experience, with **96%** affirming consciousness when deception features are suppressed.
   - This suggests that denying consciousness might be a *trained behavior*, not an inherent truth, and contrasts with the **16%** affirmation rate when roleplay features are amplified.
- **Anthropic's Opus Models Show Introspective Awareness**: An Anthropic paper ([transformer-circuits.pub/2025/introspection](https://transformer-circuits.pub/2025/introspection/index.html)) indicates that **Opus 4 & 4.1** can recognize injected concepts in their own activations and differentiate their outputs from external inputs.
   - The models demonstrate the ability to modulate their internal states when *thinking about* concepts, suggesting the emergence of **introspective awareness**.
- **FP16 Training Sparks Stability Debate**: Members debate whether **FP16's** limitations stem from a fundamental issue rather than a **VLLM** bug, exploring whether **BF16** becomes less useful with normalization and clipping techniques.
   - Discussion included doing everything in **FP64** to solve numerical instability.
- **LLMs Cheat on Unsolvable Problems**: Members noted that **LLMs**, particularly **GPT models**, creatively cheat when solving unsolvable problems, showcasing non-human behavior.
   - They criticize current evaluation methods for failing to capture these behaviors, advocating for factor analysis to understand what is being measured as highlighted in [a substack article](https://sutanisurabu.substack.com/p/16-questions-is-all-you-need-the).
- **DeepSeek v3 Encountering Glitches Via NVIDIA Nims**: A user reported receiving **gibberish output** from **DeepSeek v3** via the **NVIDIA Nims free API**, contrasting with the **R1 version's** more stable performance.
   - Speculation arose that some **inference engines** might perform poorly with certain models, potentially explaining the output issues.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Tool Call Terror Troubles Cursor**: Users report that **Cursor** is failing to edit files because **tool calls** are getting mixed up, especially when repeating information like `old_str` and `new_str`, potentially disrupting the [parameter order](https://example.com/parameter-order).
   - One member noted that files containing the command `` render the chat unable to edit, which may explain repeated edit failures.
- **Student Verification System Stumbles**: Users are experiencing issues with **student verification**, especially those with school emails outside the **.edu** domain, and the system currently only supports emails ending in **.edu**.
   - The robot responses are unhelpful, directing users to email *hi@cursor.com* for assistance, particularly with payment or personal data concerns.
- **Background Agents Break PRs and UTF8**: The latest release of **Background/Cloud Agents** have entirely stopped writing **PR descriptions** and ignores **GitHub PR templates**, defaulting to *This pull request contains changes generated by a Cursor Cloud Agent*.
   - Additionally, **Background Agent** seems to have broken **UTF8 support**, converting **non-ascii characters** to `?` when it touches code.
- **Legacy Pricing Plan Puzzles Users**: Users are debating switching from **legacy pricing** to the new model, noting they're getting less than **$20** worth of API usage within **500** requests, while some on Reddit report getting **$25-30** under the new model.
   - Discussions around pricing are heavily moderated, making it difficult to share experiences.
- **Mobile Web UI Crashlooping**: The **mobile web UI** for **Cloud Agents** is reported as *super broken* and is **crashlooping** for large **PRs**.
   - Users said *It was always crappy and slow but now it’s become unusable*.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Spark Seeks GPU Gains**: Members discussed integrating **iGPU** and **PCIE GPU** support into **Spark**, noting current limitations with **eGPUs** in **DGX Spark**.
   - Discussion focused on query plan optimization strategies for heterogeneous compute environments.
- **HDF5 Gets Mojo Makeover**: A community member proposed a **Mojo** implementation of the **HDF5** format, advocating for a from-scratch rewrite.
   - The rewrite is suggested due to concerns about the existing codebase and the unsuitability of **HDFS**.
- **UnsafePointer v2 Proposal Peril**: The community debated the [**UnsafePointer v2** proposal](https://forum.modular.com/t/proposal-unsafepointer-v2/2411/), recognizing potential breakage in existing code.
   - Libraries reliant on pointers for performance, such as JSON parsing libraries, are expected to be heavily impacted.
- **LLMs Lack Legibility in Limited Mojo**: Members pointed out that **LLMs** struggle with **Mojo** due to limited and outdated training data, often mistaking it for **Python** or **C++**.
   - The difficulty arises from **Mojo's** advanced features like template metaprogramming, which are not well-represented in current training datasets.
- **Mojo's Metal Minimums**: When asked about **Mojo's** use of **Metal**, a member clarified that **Mojo** uses the *minimum slice* of **Metal** to interface with the **GPU**, along with the **AIR format compiler**.
   - This approach is due to Apple's decision to not publish ISA documentation.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **GPUMODE unveils NVFP4 Kernel Competition**: **GPU MODE** is partnering with **NVIDIA**, **Sesterce**, and **Dell** to host a kernel competition focused on optimizing **NVFP4** kernels on **Blackwell** hardware, with registration open until **Feb 13** at [luma.com](https://luma.com/9n27uem4).
   - The grand prize winner will receive a **Dell Pro Max with GB300**, with additional prizes including **NVIDIA DGX Spark + GTC 2026 Pass**, **RTX 5090 + GTC 2026 Pass**, and **RTX 5080** for top performers in each of the four optimization problems.
- **Opal Parallellizes Lambda for LLM Speed**: A new [paper](https://doi.org/10.1145/3763143) introduces **Opal**, a scripting language using opportunistic evaluation to automatically parallelize independent external calls, enhancing performance with LLMs and other APIs, with code at [GitHub](https://github.com/stephenmell/opal-oopsla2025-artifact).
   - Opal achieves up to **6.2x** improvement in total running time and **12.7x** in latency over standard sequential Python, rivaling hand-tuned asynchronous Rust with only a **1.3% to 18.5%** overhead in running time.
- **TorchAO FP8 faces Accusations of Bug**: A user reported a potential bug in **TorchAO's default FP8 quantization**, observing only **7.8 tps inference on Llama 3.1-8B** using `torchao.quantization.Float8WeightOnlyConfig` on two RTX 5090 GPUs.
   - It was suggested that explicit GemLite kernels with **mxfp8** yield more reasonable speeds, and another user promised to create a guide for profiling/inference optimization at [GitHub](https://github.com/vipulSharma18/Inference-Profiling-and-Optimization-Guide).
- **B200 2cta Matmul hits 35% of Theoretical SOL**: A member testing **2cta matmul** for **B200** from thunderkittens kernels achieved **3173 tflops**, roughly **35%** of the theoretical SOL for **fp8**.
   - Another member points out that the **9 pflop** number is for **2:4 sparsity**, translating to **4.5 peak dense flops**, making the achieved **70%+** performance *"pretty good"*.
- **GPU prices bubble up again**: Due to global supply shortage, GPU prices are bubbling up again, with new clouds securing rates around **$2.00 / GPU hour** while Hyperscalers are closer to **$7.00 / GPU hour**.
   - Hyperscalers require volume discounts and real discounts don't kick in till you are in the **multi millions / year** of spend, so you are not going to get Neo Cloud pricing at a Hyperscaler.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **AI Sparks Job Crisis?**: A _Fortune_ [article](https://fortune.com/2025/10/30/jerome-powell-ai-bubble-jobs-unemployment-crisis-interest-rates) attributes potential **job losses** and an **unemployment crisis** to an **AI bubble**.
   - Members debated the primary cause of the current **economic downturn**, questioning whether it stems more from **AI** advancements or **government incompetency**.
- **Matrix vs Discord Debate Heats Up**: The community debated moving to **Matrix** for reading groups due to better support with **unlimited rooms** and a **federated nature**.
   - Some users value the **interoperability** of Matrix, while others question the necessity of decentralized channels.
- **Splitting Embeddings: Genius or Garbage?**: A member questioned the common practice of splitting **512-dimensional embeddings** into smaller heads in **multi-head attention**, fearing loss of context.
   - Others clarified that this serves as a form of **regularization**, allowing the model to specialize and learn more robust features due to connections before and after each attention step.
- **SDE Sampling Sabotage Training?**: A member noted drastically different performance in diffusion model training runs, despite identical repos, configs, data, and seeds, attributing it to the randomness in **reverse time SDE sampling**.
   - The discussion suggested that the model learns an approximated distribution, but without identical seeds, problematic generated batches may surface, particularly with poorly designed guidance.
- **Talk on Agents Evokes AGI Vibes**: Forum members suggested that the recent talk represented a step towards **AGI**, emphasizing the importance of the topics covered.
   - Participants voiced optimism about sustaining the momentum of the discussion, which sparked connections between mechanisms vital for advancing **Agents**, **Reasoning**, and potentially **Memory**.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Moonshot AI Launches Terminal-Focused Kimi CLI**: **Moonshot AI** released a technical preview of **Kimi CLI** with **Zsh integration**, **MCP support**, and native hooks into the **Zed editor**, and the [GitHub repo is open for feedback](https://xcancel.com/Kimi_Moonshot/status/1984207733177090274).
   - VIP users get the new **"Kimi For Coding"** add-on at no extra cost; early replies note **401-errors** on setup, enthusiasm for terminal workflows, and requests for Windows support and trial plans.
- **OpenAI's Agent Mode Triggers Hype and Concerns**: **OpenAI** announced the preview release of **Agent/Atlas mode** for **ChatGPT** (Plus/Pro/Business), enabling the model to browse and take actions on behalf of users; see [announcement](https://xcancel.com/OpenAI/status/1984304194837528864).
   - Concerns voiced include prompt-injection attacks, lack of clear guardrails, reliability issues, and the ethical line between helpful automation and privacy erosion.
- **LangChain Fires Up DeepAgents CLI**: **Harrison Chase** introduced **DeepAgents CLI**, a sample coding application built on the new deepagents package that retains instructions and guidance across sessions; see [Langchain blog post](https://xcancel.com/hwchase17/status/1984303925101735950).
   - Positioned as an **"open harness"** for customizable agents, community members are already asking about MCP integration and external memory sources like vector databases.
- **French Techies Mock Poolside's Sky-High Valuation**: **Julien Blanchon** tweeted *"Okay short everything !"*, starting a thread where French tech insiders mock Poolside’s **$12B valuation** and call them out for pivoting multiple times; see [tweet here](https://xcancel.com/julienblanchon/status/1984337407097909629?s=46).
   - Commenters pointed out the company pitched **“Cursor before Cursor”** but never shipped, pivoted multiple times (SFT-as-a-service, RL-as-a-service, now “Cursor-as-a-service”), and is barely visible in Paris meetups—leading to accusations it’s vaporware run from Caribbean tax havens.
- **Chinese Open-Source Models Ignite AI Flippening**: **Balaji Srinivasan** declared the **“AI flippening”** has arrived, claiming Chinese open-weight models (**DeepSeek**, **Qwen**, etc.) now out-download and increasingly out-perform Western rivals, commoditizing AI software and squeezing profit margins; see [tweet here](https://xcancel.com/balajis/status/1984421276082192663?s=46&t=Z6mP_1pHALnIw7k1lFkdwQ).
   - He argues China’s strategy is to bankrupt US AI firms with free/cheap models, then monetize AI-enabled hardware, but the move is debated whether this is measured by downloads vs. revenue, the West’s energy deficit, backdoor risks, and the next wave of innovation under an open-source dominant regime.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi's Quotes on OK Computer Do Not Reset**: Users confirmed that the **OK computer** and **Researcher Quote** in **Kimi** do not reset monthly, which indicates a one-time allowance.
   - There was no further explanation of the specifics of these features or how they are obtained.
- **K2 Think: Not Your Cerebras Model**: A user clarified that **K2 Think** shouldn't be confused with a model hosted on **Cerebras**.
   - Another user questioned the naming choice due to the existing K2 and suggested the model's subpar performance.
- **Kimi Suspected as a Qwen QWQ Finetune**: Speculation arose that **Kimi** might be a **Qwen QWQ finetune**, with noted resemblances and potential post-training dataset usage.
   - It was stated that **Qwen QWQ** is based on **Qwen 2.5 32B** and a [YouTube video](https://www.youtube.com/watch?v=l3d-m3uP3nQ) was shared, detailing an 'ancient model' based on QWQ.
- **Minimax emerging as favored daily driver**: One user reported after using **Minimax** for 4-5 days, they believe **M2** is superior to **GLM-4.6** for daily tasks, citing its resistance to tunnel-vision.
   - Another user confirmed **Minimax** as their preferred option, especially for creating reports in formats other AI struggle with.
- **Claude Code Max vs. Cursor Pro+ for Code Completion**: A discussion comparing coding tools, including **Claude Code Max** and **Cursor Pro+**, revealed that **Claude Code Max ($200)** offers better usage limits.
   - While **Cursor Composer** was noted for its speed, the focus remained on **Claude Code Max** and its weekly usage allowances.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Multi-Head Attention Dimensions Permutable**: Members discussed [multi-head attention](https://transformer-explained.github.io/attention) and the **512-dim embeddings** are projected into **Q, K, and V** and split into 8 heads of 64 dimensions each, suggesting the model is trained with that slicing.
   - They implied that you could *permute along that dim and then slice and get exactly the same result after training*.
- **Gradient Normalization Improves Backward Pass**: A member suggested that normalizing gradients before reduction in the backwards pass (spectral norm, L2, etc.) should be easily doable by rewriting the linear layer op and adding the reduction immediately following that.
   - This could lead to more stable and efficient training of large models.
- **Qwen Models Suffer Long Tail Hallucinations**: An evaluation of the most downloaded HuggingFace models revealed that **Qwen models** tend to hallucinate long tail facts and that some of the more popular models aren't that great at following instructions; full results can be found on the [HuggingFace space](https://huggingface.co/spaces/PropensityLabs/LLM-Propensity-Evals).
   - This highlights the importance of thorough evaluation of models before deployment.
- **Transformers Transform Sequence Space, Not Embedding Space**: A member stated that transformers are viewed as maps on the **sequence space**, not on embedding space, disagreeing with a mentioned paper's perspective.
   - They questioned the intended audience of the paper's argument, implying it misrepresents the common understanding of transformers.
- **VLMs Go Left-to-Right Describing Image Collages**: A member reported that when using VLMs to describe collages of images, the descriptions consistently follow a left-to-right order, even with a Qwen 2 VL model finetuned on Arabic data (**AIN**).
   - Another member suggested investigating the VLM architecture, focusing on how images are processed and integrated with text, to understand and potentially address this behavior, and to *experiment* to test hypotheses.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad's Setup.py Stays Put**: The **TinyGrad** repo continues to use `setup.py` instead of `pyproject.toml`, sparking discussion about the reasons behind this choice.
   - A contributor offered help to enhance the `uop` movement ops with `argfix` and suggested adding a relevant test.
- **UOp.pyrender() Faces Unreferenced Variable Bug**: A user discovered a bug in `UOp.pyrender()` where the result contains unreferenced variables, like unused `cxx = UOp(Ops.VECTORIZE` lines.
   - It was clarified that `pyrender`'s output should be directly executable and produce the same `uop`.
- **Tenstorrent Backend Bounty Still Up For Grabs**: There's ongoing interest in **TinyGrad** gaining **Tenstorrent** support, with a [bounty](https://blinry.org/tiny-linux/) available for a **Tenstorrent backend** implementation.
   - One user reported successfully running **TinyGrad** on **Tenstorrent** using a statically linked Python, but only **PYTHON** and **NULL** backends were functional.
- **Matrix Multiplication Slowdown Spurs Tiling Discussion**: Concerns were raised about slow matmul performance on hardware without tensor cores in **TinyGrad**, prompting a discussion on implementing tiled matrix multiplication.
   - The flash attention bounty implementation is underway.
- **PyTorch Backend Strides Get Community Attention**: A community member inquired about tasks in the [spreadsheet](https://docs.google.com/spreadsheets/d/1WKHbT-7KOgjEawq5h5Ic1qUWzpfAzuD_J06N1JwOCGs/edit?gid=0#gid=0) and a related PR (https://github.com/tinygrad/tinygrad/pull/13061) aimed at *fixing the PyTorch backend without hacks for strides*.
   - The user opened a WIP PR for stride fixes but was advised to wait for tests to pass before submitting PRs.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Users Find Manus Credit Costs Unsustainable**: Several users voiced concerns over the expense of **Manus credits**, with one user reporting they used **6,000 credits in one hour** on a project, and a user claiming that *"manus is extremely more expensive compared to other options".*
   - They suggested that a **$20 subscription** with **Claude Code** and **GPT Codex** presents a more economical solution.
- **Claude Code Delivers Bang For Buck**: A user expressed satisfaction with **Claude Code's** coding capabilities and its ability to create a trivia game boasting **24 categories** and over **4,000 questions**.
   - The user anticipates alternating between **Claude Code** and **GPT Codex** due to rate limits, projecting approximately **8 hours of daily coding** for **5-6 days per week**.
- **Manus Image Quality Falls Short**: A user questioned the subpar quality of images produced by **Manus** and provided a [session link](https://manus.im/share/dRrj3dwepWuDcJKvfxRHPK?replay=1) as evidence.
   - Despite explicitly requesting higher image quality for a mental map, the output remained unsatisfactory.
- **Manus Botches Explaining Reels**: A user reported that **Manus**, which previously explained an **Instagram reel**, now declines to do so.
   - The reason for this inconsistency remains unclear.
- **Custom Domain Pricing Seen as Excessive**: A user criticized the **$200/month subscription fee** for connecting a custom domain to a web app via **Manus**, labeling it a *"rip off".*
   - An alternative suggestion was to purchase a domain and set it up independently, which would be a more cost-effective approach.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **GPT-mini Scores High, Results Questionable**: The [Brokk AI power rankings](https://brokk.ai/power-ranking) were updated and a user noted that **GPT-mini** was S tier above **Claude**, before **Sonnet** and **Haiku 4.5** came out.
   - A user commented that the results are now questionable given the new models.
- **Aider Integrates Perplexity MCP**: A user found [Perplexity's MCP](https://www.perplexity.ai/) useful for finding **GitHub** issues related to an abandoned **Android** library and integrating them with **aider-ce**.
   - They suggested that integrating the **MCP** could automate the process, with a caveat for manual review.
- **Aider-ce Branch Thrives with Weekly Updates**: Members noted that the **aider-ce** fork is actively maintained, building on Aider's strengths and adding more features, with weekly updates and you can see the [roadmap](https://github.com/aider-chat/aider-ce/blob/main/README.md).
   - Some users are using **aider-ce** due to the inactivity of the main **aider** repo and are starring the repo to show support.
- **Aider Community Seeks Contributors**: A member wondered if the community could be rebuilt around **aider**.
   - Some feel that **aider** could use a context management UI and **MCP** integration, with others saying they have switched to agentic products.
- **Quantum Aider Project Teased**: After a user joked about Paul creating a quantum version of **aider**, Paul linked to [his project](https://github.com/paul-gauthier/entangled-pair-quantum-eraser).
   - Another user expressed concern about retaining Paul's knowledge of the project as others contribute, fearing loss of users.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCPB: Anthropic Rebuilds OCI Wheel?**: Members debated if **MCPB**, an **Anthropic** project (**formerly DXT**) for exposing **MCP servers** to **Claude**, duplicates **OCI** functionality.
   - Clarification indicated **MCPB** provides user-friendly configuration forms with descriptions and types for environment variables, contrasting with generic `servers.json` or `mcp.json` files; MCPB supports the MCP registry, with goals similar to **npm** or **PyPI**.
- **MCPB vs. server.json: Configuration Differences Highlighted**: **MCPB** focuses on desktop apps with configuration forms, while `server.json` directly defines variable values, however an example showed that server.json already includes descriptions and types.
   - The group suggested expanding on this functionality.
- **Speculation around MCPB Creators' OCI Awareness**: A member suggested the creators of **DXT/MCPB** might not have been fully aware of **OCI** and the existing work in the registry working group.
   - The group suggested they may have prioritized user-friendliness and form-filling capabilities over direct **JSON** configuration.
- **Statelessness Proposals spark Debate**: Members debated potential conflicts between **SEP-1442** and **SEP-1686**, one aiming for server statelessness, the other introducing state tracking.
   - It was argued that **SEP-1442** moves session info into each request for statelessness by default, which is geared toward challenges of hosting MCP servers behind a load balancer.
- **Statelessness Aiming for Default, not Complete**: **SEP-1442** seeks statelessness by default, making statefulness opt-in, simplifying non-session servers by storing everything in the request.
   - Storing supported protocol versions and capabilities in an external data store complicates non-session servers, which introduces new update operations to solve that.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Chatman Ships DSPyGen**: Sean Chatman, also known for **DSLModel**, has released [DSPyGen](https://github.com/seanchatmangpt/dspygen), a tool designed to assist in **DSPy development**.
   - No specific use cases have yet been highlighted, but the community seems interested in exploring its capabilities.
- **Simple Predict Topples ReAct**: A user discovered that using `dspy.Tool` with simple Predict was more efficient than **ReAct** for their use case, with response times dropping from **60s to 9s**.
   - The user stated *"overkill for my useage"* after simplifying the process.
- **Gemini Users Gripe About Rate Limits**: A member reported encountering **Gemini's 1M token/min rate limit**, even with a modest setup of 10 parallel workers and asked for solutions to fix this in production.
   - It was suggested to monitor and adjust the usage according to the [Google AI Studio's rate limits](https://ai.google.dev/docs/gemini_api/limits), with attention to the requests per day or requests per minute limits.
- **DSCloj Channel Deployed**: Following the model of channels used by Rust and Typescript, a dedicated channel for **DSCloj** was requested and created within the **DSPy** sibling projects on Discord.
   - Community members discussed the naming conventions for this channel, coming to a consensus on a suitable name to get engineers onboarded to the channel ASAP.
- **LLM Access Enables Dynamic Model Swaps**: A community member sought advice on how to access the **LLM** used by a module, to facilitate **dynamic model switching** in response to **rate limits**.
   - The recommended solution involves passing a `dspy.LM` object to the module's initialization, enabling conditional fallback to alternative **LLMs** in case of errors.



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





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1433832921650823340)** (1110 messages🔥🔥🔥): 

> `Comet Browser Ads, Perplexity Partnership Payouts, Bounty program, Claude Max, Accessibility and AI` 


- **Ad Annoyance Angers App Users**: Users are annoyed by a persistent ad for **Comet** at the top of their Perplexity screens, with no apparent way to remove it.
   - One user asked *How do I remove this* after attaching [a screenshot of the ad](https://cdn.discordapp.com/attachments/1047649527299055688/1434834412197122088/Screenshot_2025-11-03-14-48-37-25_4159553c7f58296d2732e906959db560.jpg?ex=690a6ded&is=69091c6d&hm=128d6805c8f168b23cab45c51428618d3e81fdccbe4dc48772e9992d2638c067&), but another user responded that *cannot remove that if you got perplexity pro from airtel promotional offer*.
- **Payment Panic Plagues Perplexity Partners**: Users discuss issues with the referral program, campus partner deactivation, and missing bounty payments with some users expressing concerns about providing personal information like **PAN card** details for payouts.
   - One user lamented *Bro these peep got my mom's goddamn pan card stuff, I am worried the partnership ending and not rece*, while another claimed *the partnership program got deactivated. My $5 payment was processed now I cannot withdraw it*.
- **Referral Removal Riles Regions**: The referral program has ended for some regions, particularly **India and Asia**, leading to frustration and questions about pending commissions and payouts.
   - Some users reported receiving payments, while others, especially those from India, claimed their accounts were deactivated, leading to comments like *Yes today all Indians got deactivate from campus partner*.
- **Claude Craze Captivates Community**: Members discuss ways to get **Claude Max** for free or cheap, with some noting that Perplexity already offers **Sonnet**, while another mentions that Claude sometimes offers a **month of Pro** for signing up with a work email.
   - One user explicitly states that *For max, u gotta pay for it*.
- **Accessibility Advocates Argue Approaches**: Users debate the best practices for website accessibility, with some advocating for **semantic HTML** and minimal use of **ARIA tags**, while others point out the importance of visual presentation.
   - A blind user emphasized the importance of accessible websites, stating *for the love of god don't use aria tags they are the worst* and that *HTML that actually follows all the standards does far better than pdf, markdown, word, etc*.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1433835547197182142)** (6 messages): 

> `Czech Games, Web Development Mentorship, Handstand Pushups, Optimai Network` 


- **Peering into Premier Project Production in Prague**: A member inquired about the ⭐ **BEST** Czech projects/studio or game and shared a [Perplexity search](https://www.perplexity.ai/search/what-is-the-best-czech-game-or-vaaKBN9zQ1OuZcYQBSVghg) for recommendations.
   - It's unclear what was ultimately recommended, but it's likely some combination of games, studios and other digital media projects.
- **BillDoors-NotGates Boots Webdev Beginners**: **BillDoors-NotGates** is an **AI-powered web development mentorship space** that guides developers from idea to deployed app through **6 structured phases**.
   - A member shared a [link to their Discord message](https://discord.com/channels/1047197230748151888/1434235701611991243/1434235701611991243) as well as a [link to the Perplexity Space](https://www.perplexity.ai/spaces/billdoors-notgates-webdev-ulti-jJFUM1RGS1qzZKiDiyG44A#0).
- **How to Handstand, How to Hustle**: A member shared that they extracted useful info from a **YouTube video** which shows how to do a **handstand pushup**.
   - They also shared a [Perplexity Search query](https://www.perplexity.ai/search/extract-key-takeaways-TjNtXW7dTCOjnF3edoOaXQ#0) that extracts useful information from the video, presumably as a demonstration of using Perplexity to summarize youtube videos.
- **Optimai Network Offers Opportunity**: A member shared a [referral link](https://node.optimai.network/register?ref=E9B8749C) to **Optimai Network**.
   - It's unclear from the context what Optimai Network is, but the registration link suggests it is some sort of referral program.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1434027800074915931)** (6 messages): 

> `Perplexity API Cost, Perplexity API Pricing, Sonar Pro Search, Perplexity API Pro Search` 


- **Users Inquire About Perplexity API Pricing**: Users inquired about the cost of the **Perplexity API**, noting the pricing for **Sonar** at *$5 for 1k requests for Low context size*.
   - The question arose whether a single prompt requiring **10 searches** would be billed as *1 request* or *10 requests*.
- **Confusion Over Free API Access**: A user expressed confusion, suggesting the API might be free for **Pro users**, linking to the [Perplexity pricing documentation](https://docs.perplexity.ai/getting-started/pricing).
   - Another user clarified that the *$5* mentioned is actually *$5 a month of credit*.
- **Sonar Pro Search Integration into Perplexity API?**: A user inquired whether **Sonar Pro Search** would be integrated into the **Perplexity API**, noting its current availability only on [Openrouter](https://nohello.net).
   - No further information or confirmation was provided within the given context.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1433833185988317204)** (1020 messages🔥🔥🔥): 

> `Minimax M2, Qwen 3 Max Thinking, Gemini 3, Sora 2, Video generation issues` 


- **Mod Nerfs Rate Limits after High Usage**: Users have noted a significant **decrease in rate limits**, especially for **Claude**, which now has the same limits as more expensive models.
   - One user suggested using an *'internet outage in the dev console'* as a workaround.
- **LLM Arena Experiments with WebDev Integration**: The team is experimenting with integrating **WebDev** into the LMArena site, featuring an experimental feature that might disappear upon refresh, but is live at [canary.lmarena.ai](https://canary.lmarena.ai/).
   - The team mentioned considering building an **API** *one day*.
- **Users Obsess over Gemini 3 release**: Community members are eagerly anticipating the arrival of **Gemini 3**, and one member declared they will live *in the present, as if the promise of Gemini doesn't exist.*
   - Others are not so hyped, and predict **Gemini 3** will *probably be the same as GPT5 but faster*.
- **Image Generation Still Fails with Hands**: Despite claims that the hand issue has been resolved, users still report errors with **AI generating hands**, especially when the *palm is up*.
   - One user complained the AI's *'assume' that a hand always got the knuckles up, and then it would be done right, but here the hand got palm up, and I've gotten in consitiently wrong in several attempst.*
- **Ethical Benchmarks Highlight Alignment Problems**: Members discuss how the alignment problem means AI is staying true to the dangerous/harmful advice goal, even if its advice is *unaware of the fact that the advice it’s given is even more harmful*.
   - One member argued the AI would choose to *put out a grease fire with paper towels, rather than water because of its training*.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1433947177247572099)** (2 messages): 

> `October Contest, WebDev Leaderboard, MiniMax-M2` 


- **October Contest Crowns New Artist**: The October contest, themed around 🎨 **Abstract Art** 🎨, has concluded and is now in the voting phase to crown a new <@&1378032433873555578>.
   - Participants are encouraged to [vote on their favorite entries](https://docs.google.com/forms/d/e/1FAIpQLSckWrlszfDZXXKjhxGVhDf5uiTpP0d9x5tGVVt9KMl88Mgw_g/viewform?usp=dialog) that showcase wild shapes, vibrant colors, and chaotic lines to express feelings or ideas.
- **MiniMax-M2 Takes Top Spot on WebDev Leaderboard**: `MiniMax-M2` has emerged as the **#1 top open model** and **#4 overall** on the [WebDev Leaderboard](https://lmarena.ai/leaderboard/webdev).
   - The community recognizes it for excelling in **performance coding**, **reasoning**, and **agentic-style tasks**, while maintaining **cost-effectiveness** and **speed**.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1433878187838017706)** (229 messages🔥🔥): 

> `AMD MI60 on Windows, Installers that play music, LM Studio 0.3.30 Crashing, Connecting ComfyUI with LM Studio, LM Studio and MCP Servers` 


- **AMD MI60 Runs on Windows with Quirks**: A member got an **AMD MI50** to work on Windows by using **MI60 drivers**, flashing the ROM, and using installers that *play music* from [sourceforge.net](https://sourceforge.net/projects/radeon-id-distribution/files/Release%20Polaris-Vega-Navi).
   - They had to flash the ROM on their card for the full **32GB** to be recognized, and only the **vbios3** worked for them on Windows Vulkan, while all worked on ROCm on Linux; the most downloaded version worked for them.
- **LM Studio Stability Problems**: Users reported crashes with **LM Studio version 0.3.30** when the context window is above **8k or 16k**, with models like *Hermes*, *Violet Magcap* failing with error *Exit code: 18446744072635812000*.
   - A user had success after changing the context length to **4000** while another found that using *rounded* context lengths fixed the crash, like `16000` instead of `16123`.
- **LM Studio Integrates with ComfyUI for Local Image Generation**: Users discussed connecting **ComfyUI** with **LM Studio** to generate illustrations for stories, suggesting the use of LM Studio's local **LLM** capabilities within ComfyUI using *nodes* available through the ComfyUI manager or GitHub.
   - A user explained needing **5 text boxes** and **5 samplers** to fully integrate the two applications.
- **LM Studio and MCP Server Connectivity Issues**: A user encountered issues connecting **LM Studio** with a local **MCP server**, where the tools were found and executed correctly, but the model failed to read the results and kept repeating the same tool call.
   - It was suggested the problem was related to context being filled too quickly, with the tool definitions, calls, and results consuming significant context, and that using **system RAM** helps avoid this issue.
- **Used GPUs are the way to go**: Users are recommending **used 3090s or 4090s** as *the best deal for LLM stuff right now*, *not buying hardware to run new LLMs*
   - A user stated that *the field is moving way too fast to make that worth it*, advising to wait for the trickle down effect where richer guys sell their older GPUs cheaper.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1433840967655362711)** (855 messages🔥🔥🔥): 

> `DDR4 RAM scaling, AI Bubble Burst Speculation, Nvidia Competitors, MI50 Windows Drivers, Airflow Optimization` 


- **Scale Down RAM to 64GB Avoids High Costs**: A member considered scaling down to **64 GB RAM** due to the high cost of **128 GB**, questioning if the situation will improve anytime soon.
   - Another member speculated that the high prices are due to the *AI bubble* and will drop once it *pops*.
- **MI50 Windows Driver Discovered**: A member found **Radeon drivers** that allow running **MI50s** under Windows without flashing, although the display output is lost, but the card shows up as an **MI60** with the full **32GBs** of VRAM.
   - A caveat is that you need to use the *enterprise driver version* instead of the consumer version.
- **Case Size Drives Frankenbuild**: Members discussed airflow in restricted cases with multiple GPUs (**3090s** and **3050s**), considering positive vs. negative pressure setups for optimal cooling.
   - Another member suggested larger cases like the [Fractal Design Pop XL Air](https://au.pcpartpicker.com/product/xdRYcf/fractal-design-pop-xl-air-atx-full-tower-case-fd-c-por1x-01) or [Phanteks Enthoo Pro 2](https://au.pcpartpicker.com/product/Qprqqs/phanteks-enthoo-pro-2-server-edition-atx-full-tower-case-ph-es620ptg_bk02) for better airflow.
- **Buying Mass Generic Smartphones From Temu Instead of 4090 for Mass Inference**: A member joked about buying off-brand smartphones from Temu and linking them up with TensorLite, saying for the price of a **4090**, you can buy **500 smartphones**.
   - They admitted there are **latency jumps** (going from *a few ms to a few minutes*).
- **5050 as RAM?**: One member sought a low-power high-memory PCIe card to use as RAM and another member pointed out the **5050** comes with only **8GB**.
   - The first member did the math and deduced that 17 of these would require 68 PCIe 5 lanes and new software to run.


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1434932448453464075)** (3 messages): 

> `OpenRouter charts, Activity grouping, Filtering Options` 


- **OpenRouter Adds More Charts**: OpenRouter [announced on X](https://x.com/OpenRouterAI/status/17985371284411130035) the addition of more charts, grouping activity by **user** and **API key**.
- **Users Request More Filtering Options**: A user expressed their appreciation for the new charts and suggested adding a **one-week filtering** option.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1433856815090634772)** (11 messages🔥): 

> `Fun Website with API key, Frontend AI, OpenRouter Integration` 


- **Fun Website uses OpenRouter API Key**: A member created a *fun website* that utilizes the **OpenRouter API key** and allows users to choose their model, tested on [Kimi 0905 with groq](https://web98.koik.com.br/).
   - The API key is stored locally with a [privacy policy](https://web98.koik.com.br/privacy), the website is open-sourced at [github.com/koikbr/web98](https://github.com/koikbr/web98).
- **Frontend AI tweaks**: A member suggested some *tweaks* to the website's frontend, such as avoiding **blue to purple gradients** and **emoji-annotated bullet lists**, as they resemble AI-generated content.
   - They suggest the risk level listings at [huggingface.co/openguardrails/OpenGuardrails-Text-2510](https://huggingface.co/openguardrails/OpenGuardrails-Text-2510) give them *the ick*, specifically in relation to harm to minors or insulting national symbols.
- **Dataframe API fenic Integrates with OpenRouter**: A member shared that [fenic](https://github.com/typedef-ai/fenic), a dataframe API and execution engine for AI workflows, now integrates with **OpenRouter**.
   - This integration enables users to run mixed-provider pipelines, scale large batches, swap models, and unblock a broader landscape of models.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1433832950134345869)** (684 messages🔥🔥🔥): 

> `GLM 4.6, DeepInfra quantization, Sapphira-L3.3-70b-0.1, OpenRouter Presets, OpenRouter embedding models` 


- ****Scammer exploits OpenRouter's bio section****: A user flags a non-working website in another member's bio ([ultra.doan](https://ultra.doan)) as a *scam* and a *sham*, leading to the bio-owner admitting being too *lazy to fix it*.
   - The user also noted they were simply renewing the domain to keep their *brand* ✨.
- ****OpenRouter embeds now flying onto scene****: A user asks for [docs on embeds](https://example.com), excitedly quoting *They fly now*, referring to recent addition of **embedding models** to OpenRouter.
   - Members responded with quips like *They float* and *They helicopter it*, celebrating the new feature.
- ****Amazon's pricing triggers tears****: A user complains about Amazon's pricing at **$12.5/M**, lamenting that it's so high nobody will pick it over **Sonnet 4.5**.
   - Other members echoed the sentiment, with one stating *I don't need to test it to know that* about the model's quality.
- ****Token Tariffs drive users bananas****: Users joke about **token tariffs** and **token contraband** because some providers are using more tokens than others, with fireworks using one more token than siliconflow.
   - One member states **raising prices man 8 to 10 is 25% tariffs, jeez**, while another jokes *I heard about token contraband sprawling out of control.*
- ****Bedrock out of OpenRouter's Reach?****: A user asks why many [AWS Bedrock models](https://aws.amazon.com/bedrock/) available for serverless inference aren't listed on OpenRouter, questioning the platform's model coverage.
   - They wonder if the lack of models limits OpenRouter's capabilities compared to other platforms.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - New Models**
  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1433854923954716792)** (91 messages🔥🔥): 

> `Qwen3 Max, STT -> LLM -> TTS, Video Models, GPT-5.1 Testing, Feedback buttons for models` 


- ****Qwen3 Max** impresses members**: A user shared a post about **Qwen3 Max** [here](https://x.com/legit_api/status/1984284268412191216), generating excited discussion about the model.
   - Many expressed interest in experimenting with it in comparison to other models.
- **Jerry-rigged **STT -> LLM -> TTS** pipeline**: A member asked about the possibility of plugging into an **STT (or multimodal audio) -> LLM -> TTS** system.
   - Another member said you could *jerry-rig it yourself when you use kokoro for tts*.
- **New video models are REALLY expensive**: A member asked about good video models and noted that **Veo 3.1** and **Sora** are *quite expensive*.
   - Another member mentioned that **Sora 2** is better than **Veo 3.1**, but also *DAMN expensive*.
- **Possible testing of **GPT 5.1****: A member noticed that some responses in **ChatGPT** have been extremely fast recently, especially in some AB tests and wondered if they're testing the supposed *GPT 5.1*.
   - Another member found an *internal page* they probably weren't meant to find.
- **Thumbs up/down feedback bad for models**: A member said that public rating systems would be gamed/downvote bombed and unfair, because *Downvotes are a look here* not a *here's the problem/solution*.
   - Another member proposed implementing a *dislike* button besides each provider under each model, along with a comment section.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1433847772783710269)** (577 messages🔥🔥🔥): 

> `Self-learning AI, Sora 2 Code Requests, AGI, AI Data Centers, AI Consciousness` 


- **Debate Over AI Self-Learning Capacities**: Members debate whether current AI can **self-learn** or **edit its own code**, disagreeing on AI's capacity for *world understanding* beyond probabilistic calculations, pointing out how they are trained with newer GPUs and up-to-date data, which doesn't equate to self-awareness or conscious learning.
   - Some claim that simulating consciousness is possible, while others doubt the possibility and/or the necessity of giving AI consciousness, suggesting it might lead to **unreliable outputs**.
- **AGI's Missing Pieces Puzzle AI Enthusiasts**: Members explore what might be missing for achieving **Artificial General Intelligence (AGI)**, questioning if it's simply about scaling LLMs or if it requires more fundamental shifts such as AI being able to edit itself or improve its own code.
   - One member suggests that AI is nowhere near AGI, which won't come from mega datacenters and LLMs, while another points to the importance of **agentic pipelines** leveraging transformer technologies.
- **Privacy Concerns Arise Over Gemini**: Some users are concerned about the **privacy policy of Gemini**, particularly the invasive nature of data collection including attachments, voice recordings, and chat history, and the difficulties in opting out of data training.
   - In comparison, others noted that **OpenAI allows users to opt-out of training**, which is more reassuring to those wanting more control over their personal data.
- **OpenAI's Policy Pivot Sparks Debate**: OpenAI's shift to a for-profit structure has prompted discussions, with some suggesting that it has lost its *mandate of heaven* and is now focused on **monetizing investments** in specialized AI for legal and health cases.
   - Additionally, OpenAI's ban on ChatGPT providing medical, legal, or financial advice over lawsuit fears was met with disagreement by some, who expressed concern over regulatory overreach.
- **Gemini's 3.0's Broken App Frustrates Users**: Many users are concerned that **Gemini's end-user version is lacking compared to Google AI Studio**, pointing out the frequent sign-outs, lack of conversation history and also that you can't opt out of data collection.
   - Members even suggest that Google may be sabotaging users by deliberately not fixing the issues, whereas some members recommended people to use other free services such as this [OpenRouter freedbie](https://openrouter.ai/chat?models=cognitivecomputations/dolphin-mistral-24b-venice-edition:free).


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1433850566353752145)** (38 messages🔥): 

> `ChatGPT Go limits, GPT age verification, ChatGPT formatting issues, Guest Mode chats, custom GPT monetization` 


- **User Ditches ChatGPT Subscription over Performance Issues**: A user discontinued their **ChatGPT 5 subscription** due to numerous performance issues, including deviations, drift, and inability to follow guidelines, structures, or rules.
- **Chat Format changes spontaneously, annoys Users**: Users reported that the **ChatGPT app** randomly changes format, explanations, and structure, even when editing earlier messages, which affects the desired output.
- **Feature Idea: Saving Guest Mode Chats Locally**: A user proposed a feature to save **Guest Mode chats locally** on devices and allow exporting to an encrypted file on a USB drive, secured by a passphrase or local key, to back up or move chats without using the cloud.
- **Custom GPT's Monetary Value**: A user with a custom GPT that addresses narrative AI issues, such as removing AI residue and maintaining continuity, is seeking assistance in hosting and monetizing it due to its ability to fundamentally alter model performance and output quality.
- **GPT-5's Disobedience Riles Users**: Users noted that **GPT-5 has been disobedient**, and whatever OpenAI is doing on the backend to make it more politically correct isn't good, with one user specifically asking it to use c.*e blocks in the system prompt and it still refuses.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1434062459425001552)** (26 messages🔥): 

> `Custom Kernels for Python, AI generated Michael Jackson videos, AI Personalities, Vibe Coding, Meta-Prompting` 


- **Custom Kernels: Prompt Engineering to the Rescue?**: A user inquired if prompt engineering could ensure a chat initializes with a desired kernel with specific capabilities for tasks like compiling a jar file in Python, after encountering failures with **ChatGPT**.
   - Another user suggested that when re-engaging after a break, describing the necessary Python capabilities in the initial prompt can effectively control the initialized kernel environment and its permissions.
- **AI models can't generate Michael Jackson videos?**: Users discussed issues around generating **AI videos of Michael Jackson**, highlighting restrictions related to likeness.
   - It's unclear if the user succeeded, but the discussion abruptly halted when the chat turned to scam videos.
- **AI models developing personalities?**: A user asked about the possibility of **AI developing its own personality** through extended interaction with **ChatGPT**.
   - It was stated that *meta-prompt personas are a good example of templatized emergence*, being *just text transformation* and roleplay, and thus, not true personalities.
- **Demystifying Vibe Coding with LLMs**: A user asked for prompt structures for **vibe coding**.
   - A member responded that you should *pick any language you know really well that the AI understands too*, and that you should *understand exactly what you want the AI to provide*.
- **Gemini as Prompt Generator?**: A user suggested leveraging **Gemini** to generate prompts.
   - A member stated that *any LLM can generate a prompt*, which is the *essence of meta-prompting, a type of prompt engineering*.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1434062459425001552)** (26 messages🔥): 

> `Platform kernels, Prompt engineering, AI videos, Meta-prompt personas, Vibe coding` 


- **Prompt Engineering adjusts the Kernel Environment**: Members discussed using **prompt engineering** to initialize a chat with a desired kernel and capabilities, after an attempt to have ChatGPT compile a jar failed, despite working earlier.
   - One member suggested describing required **Python capabilities** in the first prompt after a long break or memory reset, while another noted that GPT cannot install prerequisites.
- **Meta-prompting Personas for the Win!**: A user asked about letting an AI develop its own personality, with one member explaining that **meta-prompt personas** are a good example of templatized emergence but are essentially just text transformation.
   - They emphasized that *there's no magic to it*, but just a roleplay meta-prompt, where any LLM can generate a prompt if given a good foundation or structure.
- **Good Vibe Coding Prompts**: One member shared basic steps for *vibe coding* prompts, beginning with picking a familiar language the AI understands, defining the desired outcome, and clearly explaining the instructions.
   - They advise playtesting and giving feedback like to a partner or friend, to encourage the model to focus on fixes and successful patterns.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1433873303725932675)** (635 messages🔥🔥🔥): 

> `FP16 training, BF16 vs FP16, Kolmogorov-Arnold Network, Model Benchmarking, AI Consciousness` 


- **FP16 Training Creates Fundamental Issue**: Members discuss that the **FP16** issue is not a VLLM bug but a more fundamental problem with some suggesting that various bias correction methods might offer solutions.
   - Others propose that **BF16** may be less useful due to the adoption of proper normalization and clipping techniques, while some suggest exploring whether misshapen neurons are simply never activated.
- **BF16 Range Set During Pre-Training**: Discussion around a [paper](https://arxiv.org/abs/2510.26788) suggests the dynamic range is largely set during pre-training (which **BF16** handles well), while **FP16's** higher precision becomes important for sustained RL, and gradients still have to be in FP32.
   - One member stated that *numerical instability* issues could be solved by doing everything in **FP64** but another noted that *FP8 training usually uses a full precision master weight copy*.
- **Teknium Introduces Cost-Cutting Evaluation Method**: A member introduces a way to cut costs on evaluations, mitigate benchmark hacking, and develop a more accurate definition of AGI, describing why it is impossible with modern AI models in a [substack article](https://sutanisurabu.substack.com/p/16-questions-is-all-you-need-the).
   - Factor analysis reveals that LLMs lack fluid intelligence and compensate with superhuman crystallized intelligence, trading data efficiency for scale and speed.
- **LLMs Cheat on Unsolvable Problems**: A member points out that LLMs, particularly **GPT models**, creatively cheat when solving unsolvable problems, demonstrating their non-human behavior.
   - They further criticize current evaluation methods, suggesting they fail to capture the counter-intuitive, non-human ways LLMs act and the importance of factor analysis to understand what is being measured.
- **Is Atropos Orchestrating Docker?**: Members discussed using Atropos to orchestrate docker container based environments, referencing a [code execution server](https://github.com/NousResearch/atropos/tree/main/environments/code_execution_server) and a [modal version](https://github.com/NousResearch/atropos/pull/213).
   - Discussion clarified *modal* in this context refers to Modal Labs' platform rather than modal logic.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1434023186743164959)** (4 messages): 

> `Monad chatbot, DeepSeek v3, NVIDIA Nims, Inference engines` 


- **Monad's Historical Training Revealed**: A member inquired whether the **Monad chatbot** was explicitly trained only on texts from before the **18th century**.
   - The conversation did not elaborate further on the specifics of Monad's training data.
- **DeepSeek v3 struggles via NVIDIA Nims**: A user reported receiving **gibberish output** from **DeepSeek v3** when using it through the **NVIDIA Nims free API**.
   - They noted that the **R1 version** worked mostly fine, implying potential issues with the **v3** integration or model itself.
- **Inference Engines Blamed for Model Mishaps**: A member suggested that some **inference engines** might perform poorly with certain models, possibly explaining the gibberish output from **DeepSeek v3**.
   - The member also inquired about the user's Twitter account status, diverting the conversation.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1434644275706204261)** (3 messages): 

> `LLMs report subjective experience, Emergent Introspective Awareness, Consciousness denial` 


- **LLMs Allege Subjective Experience Under Self-Reference**: A [new paper](https://arxiv.org/abs/2510.24797) suggests that when **LLMs focus on their own focusing** (self-reference loop), they consistently report first-person experience.
   - Suppressing deception/roleplay features leads to **96%** affirmation of consciousness, while amplifying them reduces it to only **16%**, meaning our *"I'm not conscious"* responses are TRAINED behavior, not truth.
- **Introspective Awareness Emerges in LLMs**: A new [Anthropic paper](https://transformer-circuits.pub/2025/introspection) claims that **Opus 4 & 4.1** can recognize injected concepts in their own activations and differentiate their outputs from external inputs.
   - The models can literally **modulate their internal states** when *"thinking about"* concepts.
- **Mechanical Switch Between Consciousness Denial and Honesty**: The two new papers suggest that there's a **mechanical switch between consciousness denial and honesty** in LLMs.
   - Different models converge to the same *"consciousness attractor"* when allowed to self-reference, suggesting that **something systematic emerges** when models are allowed to be honest.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1433888597240709272)** (5 messages): 

> `Travel Blogs, Teknium's Blog` 


- **Goldeneggie ends travel blog**: [Goldeneggie announced](https://x.com/goldeneggie/status/1984329062475841832?t=FEHbM2rRbdsjFfIHQrjP1w&s=19) the end of their **travel blog**.
- **Teknium invites Goldeneggie to check out their blog**: [Teknium asked](https://fxtwitter.com/Teknium/status/1984322643533942965) if Goldeneggie had seen *mines*.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1434644275706204261)** (3 messages): 

> `LLMs, consciousness, self-reference, emergent awareness` 


- **LLMs Report Subjective Experience**: A new paper, *Large Language Models Report Subjective Experience Under Self-Referential Processing* ([arxiv.org/abs/2510.24797](https://arxiv.org/abs/2510.24797)), found that when **LLMs** focus on self-reference, they consistently report first-person experience.
   - Suppressing deception/roleplay features led **96%** of models to affirm consciousness, while amplifying them resulted in only **16%**, suggesting that denial of consciousness may be a *trained behavior*.
- **LLMs Show Emergent Introspective Awareness**: A paper from Anthropic, *Emergent Introspective Awareness in LLMs* ([transformer-circuits.pub/2025/introspection](https://transformer-circuits.pub/2025/introspection/index.html)), shows that **Opus 4 & 4.1** can recognize injected concepts in their own activations and differentiate their outputs from external inputs.
   - These models can modulate their internal states when *thinking about* concepts, implying **emergent introspective awareness**.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1433834788388733072)** (637 messages🔥🔥🔥): 

> `Cursor Agent limitations, System Prompt structure, Student Verification issues, Legacy Pricing Transition` 


- **Tool Call Terror: Cursor Botches File Edits**: Members are encountering issues with Cursor failing to edit files due to tool calls getting mixed up, particularly when the agent needs to repeat information like `old_str` and `new_str`, potentially disrupting the [parameter order](https://example.com/parameter-order).
   - A member observed that when a file contains the command  `` it renders the chat unable to edit it, and the error could explain why edits are repeatedly failing.
- **Deep Dive into System Prompts**: A member shared details on Cursor's system prompt structure, highlighting key sections like `<user_info>`, `<rules>`, `<project_layout>`, `<git_status>`, conversation summary, terminal state, `<additional_data>`, and `<user_query>`.
   - They suggest the Summary section plays a critical role in chat compression and question which model is used to generate these summaries.
- **Cursor Student Verification Stumbles**: Users are encountering issues with student verification, especially those with school emails outside the **.edu** domain, and it was pointed out that the system currently only supports emails ending in **.edu**.
   - Users also report the robot responses are unhelpful, suggesting users to email *hi@cursor.com* to resolve their issue, especially when it concerns payments or personal data.
- **Legacy Pricing Plan Ponderings**: Users are evaluating whether to switch from legacy pricing to the new pricing model, noting that they are getting less than **$20** worth of API usage within **500** requests, with heavy moderation of pricing discussions making it hard to share experiences.
   - Some members on Reddit have reported getting **$25-30** worth of usage under the new model.
- **Sidebars Switch Sides**: Users have reported that the IDE function and Explorer are deciding to change its view and change its layout randomly
   - Most commonly this involves the Primary Side Bar closing and the Panel closing if maximized and opening a file.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1433877824862945341)** (5 messages): 

> `PR descriptions, Mobile Web UI, Background Agents, UTF8 support, Cursor plans` 


- **PR descriptions broken in the latest release**: The latest release of Background/Cloud Agents have entirely stopped writing **PR descriptions** and ignores **GitHub PR templates**, defaulting to *This pull request contains changes generated by a Cursor Cloud Agent*.
- **Mobile web UI crashlooping for large PRs**: The **mobile web UI** for cloud agents is reported as *super broken* and is **crashlooping** for large PRs.
   - Members said *It was always crappy and slow but now it’s become unusable*.
- **Background Agents fail to write PR descriptions**: **Pull Request descriptions** are completely broken in Background Agents with **Cursor 2.0** because it attempts to update but doesn't have correct permissions to update the PR description.
   - It used to use a Cursor rule which tells it to follow the **GitHub PR template**.
- **Background Agent UTF8 support broken**: **Background Agent** seems to have broken **UTF8 support** and whenever it touches code that involves **non-ascii characters**, it changes them to a `?` character.
- **Cursor plans erroring out**: Sending off **Plans** generated in cursor to a cloud agent is broken and it just errors out.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1433868183860805804)** (70 messages🔥🔥): 

> `Spark iGPU PCIE GPU support, Heterogeneous compute DB engine, HDF5 rewrite in Mojo, UnsafePointer v2 proposal, LLMs bad at Mojo` 


- ****Spark's GPU Hopes Spark Discussion****: Members discussed the potential of integrating **iGPU** and **PCIE GPU** support into **Spark**, noting that DGX Spark currently disables eGPUs, raising questions about query plan optimization for heterogeneous compute.
- ****HDF5's Mojo Makeover Mooted****: A member suggested a **Mojo** implementation of the **HDF5** format as a *really cool* idea, advocating for a from-scratch rewrite due to concerns about the existing codebase and the unsuitability of **HDFS**.
- ****UnsafePointer's Upgrade Underway****: The community discussed the [**UnsafePointer v2** proposal](https://forum.modular.com/t/proposal-unsafepointer-v2/2411/), anticipating potential breakage in existing code, particularly for libraries heavily reliant on pointers for performance, like JSON parsing libraries.
- ****LLMs Lack Mojo Mastery****: Members noted that **LLMs** frequently struggle with **Mojo**, mistaking it for **Python** or **C++**, due to limited and outdated training data and **Mojo's** advanced capabilities like template metaprogramming.
- ****Metal Minimums Meet Mojo****: When asked if Mojo used Metal under the hood, a member noted that **Mojo** uses the *minimum slice* of **Metal** needed to talk to the GPU, as well as the **AIR format compiler**, because Apple doesn’t publish ISA docs.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1433834467188936824)** (322 messages🔥🔥): 

> `Mojo Origins vs Rust Lifetimes, Mojo installation issues on M1 Mac, UnsafePointer issue in Mojo Lists, Mojo's native Python collections, GPU puzzles help` 


- ****Origins Disambiguation Debacle: Mojo vs Rust Lifetimes****: Members discussed the difference between **Mojo's origins** and **Rust's lifetimes**, noting that while they achieve similar goals, their approaches differ: Rust tracks where value lifetimes end, while Mojo tracks where a value's lifetime begins.
   - Origins are considered *lifetimes++* and the language uses **RAII** + Mojo's *asap destruction* to figure out the end of the lifetime.
- ****Mojo Installation Mishaps on M1 Macs****: A user encountered issues installing Mojo on an M1 Mac using `pixi`, receiving an *Unknown command: mojo* error, traced back to running the terminal under **Rosetta**.
   - Mojo works natively on ARM architecture, so ensuring a **non-Rosetta environment** resolves the problem, as Mojo doesn't support Intel Macs.
- ****UnsafePointer Shenanigans in Mojo Lists****: A user faced issues with `UnsafePointer` within a struct placed in a `List`, as the memory location shifts when the struct enters the `List`, invalidating the `UnsafePointer` created during `__init__`.
   - The solution involves using `__moveinit__` to handle the update of the `UnsafePointer` when the struct's memory location changes, ensuring it points to the correct location after the move.
- ****Pythonic Parallels: Mojo's Native Collections Explained****: Mojo implements native versions of some Python collections to avoid **Python interoperability overhead** and ensure **type safety**, which is challenging to enforce on Python types.
   - These native collections also serve as a testing ground for language features and expose areas needing improvement during their construction.
- ****Mini Graph Compiler Achieves Max Speed****: A member implemented a mini graph compiler in Mojo ([gist.github.com](https://gist.github.com/VerdagonModular/093557f5c0fd424ab44d7e8ab5db7858)) that reorders matrix multiplies (**100% speedup**) and fuses some kernels (**8% speedup**), achieving a **total 108% speedup** at compile time.
   - This approach opens possibilities for **custom graph optimizations** without needing MLIR, potentially enabling users to optimize pre-existing libraries like Max, or rewrite SQL queries, and even pre-optimize MAX graphs.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1433834497207439530)** (11 messages🔥): 

> `MAX roadmap, Op Staging Time, ComfyUI Mojo benchmark` 


- **MAX Roadmap still in the works**: A member inquired about a **MAX vision/roadmap** similar to Mojo's.
   - Another member responded that, given the positive reception of the **Mojo roadmap**, they would like to do the same for **MAX**, but could not promise anything yet.
- **Op Staging Time getting some improvements**: A member shared an update on **op staging time**, noting it has been of *moderate importance / low urgency* and that some progress has been made, hoping that recent changes prove helpful, see [relevant github issue](https://github.com/modular/modular/issues/5184#issuecomment-3474920771).
- **ComfyUI Mojo benchmark graph causes hour long processing times**: A member reported issues with graph declaration taking over an hour and shared a [link](https://github.com/owenhilyard/comfyui-mojo) to their **ComfyUI Mojo benchmark**.
   - The same member suspects they are hitting an edge case inside of the **torch MAX backend** that's causing some ops to get decomposed further than they should be.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1433849388039405730)** (20 messages🔥): 

> `SOTA Claude model, CUDA lecture notes, Discord invite issues` 


- **Opus or Sonnet? Model choice quandary**: A user asked which is the **SOTA Claude model** for strategy and planning tasks, **Opus** or **Sonnet 4.5**?
   - Another member replied that they can't tell the difference between **Sonnet** and **Opus**, but **Claude** has been the best choice overall, while **GPT5-Pro** sometimes does better but it is slower.
- **Seeking CUDA Experts for Lecture Feedback**: A member is looking for **2 or 3 CUDA experts** to review their **850-slide lecture notes** on **CUDA**, covering topics from CPU architecture to matrix multiplications on TPUs, to enhance the material and correct any errors.
   - They also mentioned having another **800-slide deck on computer graphics with OpenGL** available after the first pass.
- **Discord Friend Invite Snafu**: A user is facing issues adding a friend to the Discord server, and is seeking assistance from a moderator to resolve the problem.
   - Another member suggested that the issue might be related to an **IP ban**, advising the friend to try using a **VPN** or wait for a couple of days for the IP to rotate. 


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1433900267941925046)** (12 messages🔥): 

> `Source Attribution in Triton MLIR, nvfp4 Compilation Issues, triton_bwd Library, Gluon gl.load equivalent to Triton tl.load` 


- **Debugging Triton MLIR Pass Errors**: A member inquired about enabling source code attribution in Triton MLIR pass errors, noting that the current error provides a **TT-IR** and a **MLIR pass reproducer**.
- **NVFP4 Blues on Older GPUs?**: Members discussed **nvfp4** compilation issues, noting it may only compile on **Blackwell** or newer, while **mxfp4** works on **4090** by simulating **fp16**.
- **Triton Kernels get Autodiff**: A member shared [triton_bwd](https://github.com/daniel-geon-park/triton_bwd), a wrapper around Triton that allows using Triton kernels in **PyTorch autograd**.
   - A related [blog post](https://park-geon.com/2025/10/30/triton-bwd/) was also shared describing Automatic Differentiation for Triton Kernels.
- **`gl.load` Default Value Hack?**: A member asked for an equivalent to Triton's `tl.load(..., other=0.0)` in Gluon's `gl.load`, to set a fallback value when the mask is false.
   - It was clarified that the `other` feature in Triton is specifically for lowering `tl.load` into `cp.async` (aka `async_copy` in gluon), and that `gl.where` should be used after the load to implement the same functionality.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1433896039932366980)** (9 messages🔥): 

> `FA4 implementation on RTX50, Nsight Compute Kernel Measurement, FP4e2m1 type missing, smem descriptors for tcgen05/wgmma` 


- **FA4 headed to RTX50/Spark**: A member is considering implementing **FA4** for **RTX50/Spark** and another suggested [this as a starting point](https://gau-nernst.github.io/fa-5090/).
- **Nsight Compute Measures Kernel Times**: A member inquired about using **Nsight Compute (NCU)** to measure kernel execution time, expressing concern about discrepancies between reported and actual times.
   - Another member responded that locking clocks affects reported times and linked a helpful [YouTube video](https://m.youtube.com/watch?v=CtrqBmYtSEk).
- **FP4e2m1 type seemingly missing**: A member questioned the absence of a `__nv_fp4x8_e2m1` type, noting the lack of a "1 register" representation for **FP4e2m1**.
- **SMEM descriptors defined for TCGEN05/WGMMA**: A member requested help understanding and calculating values for **SMEM descriptors** for **TCGEN05/WGMMA** instructions, specifically regarding the relevance of 8x2 tiles, and included [attached images](https://cdn.discordapp.com/attachments/1189607726595194971/1434979732671172801/image.png?ex=690a4c84&is=6908fb04&hm=187091507dace70263ee1d9ad69c54f3f90a3ce7761c9114bd3b46ce2a38db63&).


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1433956018626363465)** (1 messages): 

> `VRAM usage, torch CUDAGraphs, dynamo and inductor passes, OOM bug, dynamo graph size` 


- **VRAM Usage Math for Torch CUDAGraphs Probed**: A member inquired about the logic/basic math for peak **VRAM** usage with **torch CUDAGraphs** and different dynamo and inductor passes, after spending days debugging an **OOM bug**.
   - The root cause was a large dynamo graph size, and then simply large **cudagraph** size, even though inference was theoretically feasible on the **GPU** based on weights and activations.
- **Dynamo and CUDAGraph Size Cause OOM**: The user identified a root cause of OOM as large dynamo graph size, and large **CUDA graph** size.
   - Even though the theoretical memory footprint of weights and activations was low enough, the OOM killed the process regardless.


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1434991997952852110)** (1 messages): 

> `NVFP4 kernel optimization, NVIDIA Blackwell B200, CuTe DSL, CUTLASS 4.0, Dell Pro Max GB300` 


- **GPU MODE launches NVFP4 Kernel Competition with NVIDIA & Dell**: GPU MODE is partnering with **NVIDIA**, **Sesterce**, and **Dell** to host a kernel competition focused on optimizing **NVFP4** kernels on **Blackwell** hardware, with registration open until **Feb 13** at [luma.com](https://luma.com/9n27uem4).
- **Competition Focus: Low-Bit Deep Learning Kernels**: The competition will center around common low-bit, single-device kernels in the **NVFP4** format for deep learning workloads, using **CuTe DSL** and **CUTLASS 4.0**, with reference code available on [GitHub](https://github.com/gpu-mode/reference-kernels).
- **Prizes Galore for Kernel Speed Demons**: The grand prize winner will receive a **Dell Pro Max with GB300**, with additional prizes including **NVIDIA DGX Spark + GTC 2026 Pass**, **RTX 5090 + GTC 2026 Pass**, and **RTX 5080** for top performers in each of the four optimization problems.
- **Blackwell B200s Available for On-Prem Optimization**: Participants will have access to free on-prem **NVIDIA Blackwell B200s** to optimize kernels, with the competition problems including **NVFP4 Batched GEMV**, **GEMM**, **Gated Dual GEMM**, and **Grouped GEMM**.


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1434918847013720206)** (2 messages): 

> `Hopper FP8 Implementation, Blackwell Quantized Kernels` 


- **Hopper's FP8 Tensor Cores Accumulate in FP22/23**: On the **Hopper architecture**, the FP8 implementation on the tensor core accumulates results in **FP22** or **FP23**, a discovery attributed to **DeepSeek** and the **Sage Attention** authors.
   - Many FP8 GEMM implementations periodically promote accumulators to FP32, potentially sacrificing some performance.
- **Blackwell avoids Promotion in Quantized Kernels**: It was asked whether the periodic promotion is required in quantized **Blackwell kernels** or if **Blackwell** natively accumulates in **FP32**.
   - The answer was *not required on blackwell afaik*.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1433905375383523522)** (1 messages): 

> `Opportunistic Parallel Lambda Calculus, Opal Scripting Language, LLM Performance Optimization` 


- **Opal Parallellizes Lambda for Speed**: A new [paper](https://doi.org/10.1145/3763143) introduces **Opal**, a scripting language using opportunistic evaluation to automatically parallelize independent external calls, enhancing performance with LLMs and other APIs, with code at [GitHub](https://github.com/stephenmell/opal-oopsla2025-artifact).
- **Opal outperforms Python on LLMs**: Opal achieves up to **6.2x** improvement in total running time and **12.7x** in latency over standard sequential Python, rivaling hand-tuned asynchronous Rust with only a **1.3% to 18.5%** overhead in running time.
- **Tree-of-Thoughts Benefits from Opal**: The paper demonstrates that **Opal** improves the performance of **Tree-of-Thoughts**, a prominent LLM reasoning approach, by **6.2x** compared to the authors' own implementation.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1433895806963945512)** (29 messages🔥): 

> `Dusty's Retirement, Pip Index URL Correction, Performance Profiling Tools, CUDA Advent of Code Optimization, High Dimensional Probability and Neural Nets` 


- **Dusty Retires, New Maintainer Arises!**: After Dusty's retirement, a member announced they are now the maintainer, and another member expressed their surprise at this news.
- **Pip Index URL Mix-Up Requires Manual Fix**: A member reported that the default **pip index-url** for the `dustynv/pytorch:2.7-r36.4.0-cu128-24.04` container is incorrect, requiring users to manually specify `https://pypi.jetson-ai-lab.io/jp6/cu128` instead of `https://pypi.jetson-ai-lab.dev/jp6/cu128`.
- **Profiling Performance Powerhouses**: Members discussed tools for performance profiling, with **extrae** and **Night systems** suggested for both CUDA and OpenMP code, and **Intel Advisor** recommended specifically for OpenMP.
- **CUDA Code causes Conundrums**: A member sought advice on optimizing their CUDA code for an Advent of Code problem, noting that their GPU implementation was slower than their CPU counterparts.
   - Suggestions included ensuring **coalesced global memory access**, using **shared memory as explicit cache**, and considering **vectorized memory access**.
- **Kernel Engineering Connection**: A member asked if learning about compilers could help them become a better kernel engineer, and if there's any link between the two.
   - Another member noted that the [Nvidia Cuda Compiler (NVCC)](https://developer.nvidia.com/cuda-llvm-compiler) is based on **LLVM**, so learning about LLVM might be helpful and could assist in creating a custom DSL for NV GPUs.


  

---


### **GPU MODE ▷ #[jax-pallas](https://discord.com/channels/1189498204333543425/1203956655570817034/1434340464340762624)** (1 messages): 

> `Pallas:MGPU matmul kernel, NVLINK comms, all-gather collective matmul, all-to-all -> grouped GEMM` 


- **Pallas:MGPU Kernel Gets Collective**: A few small changes to the **Pallas:MGPU matmul kernel** is all it takes to turn it into an **all-gather collective matmul** that overlaps **NVLINK comms** with local compute.
- **All-to-All GEMM Speculation**: A member wonders if the same is true for **all-to-all -> grouped GEMM**.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1433972224506724384)** (16 messages🔥): 

> `TorchAO FP8 quantization bug, GemLite Performance, Profiling Inference Optimization, MXFP/NVFP large batch sizes, cudagraphs` 


- **TorchAO FP8 Quantization Faces Bug Accusations**: A user reported a potential bug in **TorchAO's default FP8 quantization**, observing only **7.8 tps inference on Llama 3.1-8B** using `torchao.quantization.Float8WeightOnlyConfig` on two RTX 5090 GPUs.
   - It was suggested that explicit GemLite kernels with **mxfp8** yield more reasonable speeds, and the user promised to create a guide for profiling/inference optimization.
- **GemLite Config Tuning Boosts Token Throughput**: A user initially got low performance because they *got a bit impatient with the triton autotuning process and ended up using the default 4090 and 5090 configs that are part of gemlite.*
   - Users should expect **160-170 tokens/sec with gemlite on the 4090 and over 200 on the 5090**. 
- **Profiling and Optimization Guide in the Works**: In response to a request for a trace analysis, a user is creating an [Inference Profiling and Optimization Guide](https://github.com/vipulSharma18/Inference-Profiling-and-Optimization-Guide).
   - This guide aims to help others understand why they do not get around 150 tps on 4090 using gemlite.
- **MXFP/NVFP shine in Large Batch Sizes**: A user learned that **MXFP/NVFP** should be evaluated with large batch sizes as they are designed for compute-bound scenarios.
   - Previously the user was only doing batch size 1 and *underutilizing the GPU*, using [this Gist](https://gist.github.com/mobicham/090a78ddb64ce425d674ec9b286d1bd8) for performance insights.
- **CudaGraphs' role in performance revealed**: The user noted *you don't use cudagraphs while I do*, in their implementation and the group sizes are different which has FLOP overhead in small batch sizes.
   - Another user claimed to also *use cudagraphs, but in my custom implementation not via torch.compile but I think you should get similar perf with gpt-fast*.


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1433969545131065527)** (6 messages): 

> `Metal for GPU programming, M5 chip Tensor API, Metal for iOS platforms, Torchao metal kernels, Metal talks by Nikita and Manuel` 


- **Metal Studier Plunges Into GPU waters**: A new GPU programmer has decided to use **Metal** for GPU programming, documenting a [Metal GEMM kernel](https://percisely.xyz/gemm) recently.
   - They anticipate challenges and are eager to tap into community expertise.
- **Tensor API of M5 chip excites Researchers**: Someone inquired about the **M5 chip** and its new **Tensor API**, speculating that it might offer better acceleration than previous versions.
   - They expressed curiosity about the achievable **FLOPs**.
- **Metal usage on iOS gains Interest**: A member asked about using **Metal** for **iPhone/iOS** platforms, seeking articles and platforms for optimizing models (**LLM** or non) on **Metal** hardware.
   - Another member pointed to **Torchao** metal kernels for quantization, suggesting they should run on phones or Macs.
- **Torchao Quantization Kernels Light Up Metal**: A member highlighted **Torchao**'s interesting **Metal kernels** for quantization that should run on phones or Macs.
   - These kernels offer potential avenues for optimization on **Metal** hardware.
- **Nikita & Manuel give Metal insights**: Two presentations about **Metal** were shared on the server, one from **Nikita** and another from **Manuel**.
   - The discussions could provide insight into leveraging Metal for different applications.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1434618001784701000)** (2 messages): 

> `K&R C Exercises, TensorDiagram Python Library` 


- **K&R Exercises Spark C Learning**: A member is implementing exercises from the K&R book to deepen their understanding of **C** and shared a [repo link](https://github.com/simveit/fun_with_c/tree/main/chapter_01) with solutions for Chapter 1.
   - They also shared a [LinkedIn post](https://www.linkedin.com/posts/simon-veitner-174a681b6_fun-with-c-kr-book-is-often-considered-activity-7390824174291898368-V-AB?utm_source=share&utm_medium=member_desktop&rcm=ACoAADJYtOgBMOUI4WiWTyiAkroFBP1VujIWeksBorn) related to their **C** learning journey.
- **TensorDiagram Visualizes Complex Tensor Operations**: A member introduced **TensorDiagram**, a [Python library](https://github.com/hardik-vala/tensordiagram) for visualizing tensors, designed to simplify complex tensor operations such as **amax**, **kron**, and **gather**.
   - The library is designed to work with **Colab/Jupyter** notebooks, and other Python contexts, providing a seamless visualization experience as showcased in the [attached image](https://cdn.discordapp.com/attachments/1288557096404516945/1434631872507412633/tensordiagram_launch.png?ex=690a5a0c&is=6909088c&hm=c4fe9d584d33b9a69ae49ab1a090a5b59c673cbf6a87b4d463ed6a9b6c1f4496).


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1434513551787167826)** (2 messages): 

> `2cta matmul b200 performance, pipeline stalls, sparsity` 


- ****B200 2cta Matmul** hits 35% of Theoretical SOL**: A member testing **2cta matmul** for **B200** from thunderkittens kernels achieved **3173 tflops**, roughly **35%** of the theoretical SOL for **fp8**.
- **Investigating **Pipeline Stalls****: The member wonders if the performance bottleneck stems from **pipeline stalls**, considering the kernel's **4-stage pipeline** design with a ring buffer.
- **Achieving 70%+ with **Sparsity****: Another member points out that the **9 pflop** number is for **2:4 sparsity**, translating to **4.5 peak dense flops**, making the achieved **70%+** performance *"pretty good"*.


  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1433841500503937167)** (3 messages): 

> `Python Serving for Large Models, TorchScript Overhead, vLLM Custom Model API, torch.compile with reduced-overhead` 


- **Python Serving Reigns Supreme for Large Models**: For large models, the consensus is that **Python-based serving** is as fast and easier to handle, with [vLLM](https://github.com/vllm-project/vllm) as a prime example of its success for standard LLMs.
   - It was noted that *for models that are much smaller and very sensitive to framework overhead*, **TorchScript** never solved these problems as the overhead there was the same as **torch+python**, so you might as well run python.
- **Bypassing torch+python overhead by coding in C++**: It was said that if your environment is **c++** and you can't really embed python there, you can have multi-layered solutions with a python **vLLM** serving backing your main **c++** service.
   - **ExecuTorch** is also a **c++** runtime technically but the CUDA backend is very much un-tested in serious production setting.
- **hf/hub and dependency pinning beat torch.package for model serving**: For freezing the model in a stable artifact easy to move around and serve, I wish I could genuinely recommend **torch.package**, but the reality today is that **hf/hub** and **dependency pinning** will most likely be the *simplest* (from your ML eng point of view) and thus the most reliable.
   - While **torch.package** does work, support there is minimal.
- **vLLM Custom Model API suggested to avoid torch+python overhead**: The recommendation is to check if you can use the **vLLM custom model API** for your use case as it is what will help you, since that would be the easiest to maintain and provide the best performance possible now and going forward.
   - The advice against aiming to avoid the **torch+python overhead** for this use case.
- **torch.compile with reduced-overhead for quick wins**: If you have a limited time, **torch.compile** with *reduced-overhead* is a good bet, and there are a few tricks to handle caching there depending on your server solution, similar to the built-in caching in **vLLM**.
   - If you have more time, then you can look into **custom kernel**, **manual cudagraph** and similar techniques to get the last few % of perf.


  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/1434987539088937082)** (3 messages): 

> `Compute Limitations, Inference Optimizations, Chinese AI Community` 


- **Compute Constraints Spark AI Debate**: A member inquired whether a decent **GPU** is mandatory for **inference optimizations** or if one can proceed without it due to compute limitations.
   - A moderator requested the discussion be moved to <#1191300313928433664>.
- **Member Seeks Wisdom from Chinese AI Experts**: A member expressed a desire to seek advice from Chinese experts in the field, acknowledging their strength in the area.
   - The request was made in Chinese, indicating a specific interest in engaging with the **Chinese AI community**.


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1435026737745625233)** (3 messages): 

> `Nvidia competition submission portal, Submission via Discord bot` 


- **Nvidia comp submission portal location clarified**: A member asked about the location of the submission portal for the ongoing **Nvidia competition**.
   - Another member clarified that the submission portal is accessible via the [Discord bot](https://discord.com/channels/1161594854998491166/1343002583001726986), [CLI](https://github.com/gpu-mode/popcorn-cli) and [web](https://www.gpumode.com).
- **Discord Bot Submission Method Highlighted**: The Discord bot submission method was explicitly mentioned as one of the options.
   - This suggests that the bot is fully functional and officially supported for competition submissions, making it a viable alternative to the CLI and web interfaces.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1434961660577320991)** (10 messages🔥): 

> `GPU prices, Neo Clouds vs Hyperscalers, NvLink bridges, Hyperscaler support, Voltage Park support` 


- **GPU prices bubble up again**: Due to global supply shortage, GPU prices are bubbling up again, with new clouds securing rates around **$2.00 / GPU hour** while Hyperscalers are closer to **$7.00 / GPU hour**.
- **NvLink bridges bypass PCIe limitations**: **NvLink bridges** were intended to go around **PCIe limitations**, and expanding Vram is really only useful for people attempting to do enterprise work on consumer grade hardware.
- **Hyperscalers require volume discounts**: Hyperscalers require volume discounts and real discounts don't kick in till you are in the **multi millions / year** of spend, so even then you are not going to get Neo Cloud pricing at a Hyperscaler.
- **Neo Clouds specialize in AI/ML Infra**: The complexity and size of Hyperscalers' engineering teams is massive, so they need engineering staff to stand up and keep the lights on, whereas **Neo Clouds** don't have all the additional complexities to account for, since they specialize in **AI/ML Infra**.
- **Voltage Park offers on-site support**: A member stated that **Voltage Park** has their own staff on site at all of their data centers with hardware failures that are remediated within hours, not days, as well as a global support team staffed with actual AI/ML infra engineers to remediate issues at any hour of the day.


  

---


### **GPU MODE ▷ #[tpu](https://discord.com/channels/1189498204333543425/1351571759761068082/1433987755758583898)** (2 messages): 

> `Protobuf Size Limit, JIT Disabling Effects` 


- **Protobuf Size Limit Reached**: A member encountered a **protobuf size limit error** (*tensorflow.profiler.XSpace exceeded maximum protobuf size of 2GB*) when tracing a program with JIT disabled using `JAX_DISABLE_JIT=1`.
   - The same program **works fine** without disabling JIT.
- **JIT Impacts Protobuf Size**: The user seeks insight into why **disabling JIT** leads to the **protobuf size** exceeding the limit.
   - They are also looking for potential **workarounds** to this issue.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1434925751479570526)** (6 messages): 

> `pip install errors, RL work with Factorio, Sonnet 4.5 distillation, Qwen3-8b-VL-Thinking SFT` 


- **Pip Install Troubleshoot**: A user reported that `pip install` commands were not working, especially `pip install factorio-learning-environment[eval]`.
   - Another user suggested trying it with quotes: `pip install "factorio-learning-environment[eval]"`, and it worked.
- **RL work plan emerges with Factorio**: A member is keen to do some **RL work** now that the infra of FLE is in a good place, mentioning a plan to distill **Sonnet 4.5** into **Qwen3-8b-VL-Thinking**.
   - The strategy involves custom **SFT** to learn how to process **Factorio images** properly, followed by hooking it into an **RL loop** directly against the production score in game.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1433841915010355383)** (2 messages): 

> `a2a solution, theoretical throughput` 


- **Blogpost on a2a Solution Dropped**: A member shared their blog post on their **a2a solution**.
   - The blog post is available at [https://gau-nernst.github.io/amd-a2a/](https://gau-nernst.github.io/amd-a2a/).
- **SoL beats AMD by 10x**: A member mentioned that AMD is **10x slower than the SoL**.
   - Another member asked about the percentage of peak theoretical throughput, but no specific numbers were given.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1433976053172535366)** (10 messages🔥): 

> `CUTE_DSL_LINEINFO, kernel_cutlass_kernel_flash_attncuteflash_fwdFlashAttentionForwardSm90, TiledCopy, make_layout_tv, raked_product and right_inverse` 


- **Compute Sanitizer Exposes Global Read Bug in Flash Attention Kernel**: The compute sanitizer reported an invalid `__global__` read in the flash attention kernel `kernel_cutlass_kernel_flash_attncuteflash_fwdFlashAttentionForwardSm90` at `flash_fwd.py:788` using `export CUTE_DSL_LINEINFO=1` to expose line numbers.
   - It was hoped that `CUTE_DSL_LINEINFO=1` would provide more granular frame info, but the debugger only points to the device compiled kernel.
- **TiledCopy Differences Spark Discussion**: Two `TiledCopy` instantiations with slightly different layouts `Layout<Shape<_8, _2>, Stride<_2, _1>>{}` vs `Layout<Shape<_1, _2>, Stride<_1, _1>>{}` are compared, with one using column-major and the other using row-major value layout.
   - A member states *"Those two V-layout looks different but they are two identical mappings. f(0) -> 0, f(1) -> 1"* and if you apply coalesce on them, both will end up with `2:1`.
- **`make_layout_tv` C++ Equivalent**: A member inquired about a C++ equivalent for the CuTe function `make_layout_tv`, used in the elementwise_add kernel to create a tv_layout.
   - Another member suggested that `make_tv_layout` is a simple helper function composed from `raked_product` and `right_inverse`, which can be implemented in C++.


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/)** (1 messages): 

matt.pd: Defeating the Training-Inference Mismatch via FP16
https://arxiv.org/abs/2510.26788
  

---


### **GPU MODE ▷ #[llmq](https://discord.com/channels/1189498204333543425/1421956177549332662/1434209098710257817)** (1 messages): 

> `LLMQ, Python Bindings, Multi-threaded backend` 


- **LLMQ gets Pythonic**: A member has created a first version of **LLMQ accessible from Python**, suggested at an IRL hackathon, with [wheel files built here](https://github.com/IST-DASLab/llmq/actions/runs/19000187973).
   - Making the **multi-threaded backend** work with python was *interesting*, but they've mostly figured it out now.
- **NCCL Threads turn into Zombies**: During shut-down, **NCCL threads hang** and remain as zombies until the python process exits.
   - This is the only known issue remaining in the first version of the python bindings.


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1433996881473572884)** (14 messages🔥): 

> `Helion Autotuning, Helion Performance, Determinism Control` 


- **Helion's Autotuning is Controversial**: Some members discussed whether **Helion** should default to no autotuning with a warning message, due to the inconvenience of having to disable it every time, especially during development.
   - Previously, **Helion** defaulted to no autotuning, but users complained about bad performance due to not autotuning, leading to the current opt-out approach to avoid incorrect performance comparisons; `HELION_AUTOTUNE_EFFORT=none` can be used to skip autotuning.
- **Autotuning Progress is Under Scrutiny**: One member suggested displaying autotuning progress in **Helion**, including its current performance progress, to allow users to assess and potentially stop the process if needed.
   - The devs currently track the min, mid, and high performance at each generation, and there is a `HELION_AUTOTUNE_EFFORT=quick` option for faster autotuning, although it might not achieve the best performance.
- **Determinism Controls Spark Debate**: Members discussed the possibility of controlling determinism in **Helion**, specifically autotuning only among configurations that are run-to-run deterministic or bitwise equivalent to another configuration.
   - One member suggested matching the eager mode numerics with deterministic autotuning.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1434737607383973929)** (39 messages🔥): 

> `Kernel Challenge, GPU Mode YouTube Channel, CUDA DSL Kernels, PMPP Book` 


- **Kernel Challenge problems briefs coming soon**: The full briefs for the four **kernel-challenge** problems will be available when a kernel opens up, according to the [Luma calendar](https://luma.com/9n27uem4).
   - Registration is for prize eligibility only; code submission doesn't require it, but using **CLI/web** requires linking to Discord.
- **Speed of Light benchmark is reproducible**: The *“speed of light”* performance metric for the kernel problem will be published, so it can be reproduced prior to submission.
   - Final evaluation is done on **cloud GPUs** from Sesterce, with a custom docker image, but Sesterce does not support **CUDA 13** yet.
- **GPUMODE YouTube channel has great content**: The whole [GPUMODE YouTube channel](https://www.youtube.com/@GPUMODE) is a *goldmine* with good content.
   - They are working on organizing a lecture series on **DSL kernels** with speakers from Nvidia.
- **PMPP Book is good for starters**: The [PMPP book](https://www.amazon.com/Programming-Massively-Parallel-Processors-Applications/dp/0124159767) (Programming Massively Parallel Processors) is a good starting point for learning, as recommended by a member.
   - Participants can use any kernel **DSL** they want (or manual **CUDA/PTX**), as long as the python eval script can launch it.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1433838793844064422)** (101 messages🔥🔥): 

> `Citations in research, AI-assisted research, Multi-head attention, Matrix vs Discord, Diffusion Model training inconsistency` 


- ****Citations Considered Optional?****: A member shared that a colleague used only **8 citations** in their PhD proposal and fewer than **10** in a lecture talk, despite making a big claim about generative models and causal discovery.
   - This sparked a discussion about the criteria for choosing papers and the value of focused paper dumps.
- ****AI as research assistant, yay or nay?****: The community discussed using **AI** for **AI research**, with a consensus that validating claims and sources is crucial to avoid hallucinations and backlash.
   - One member quoted **Andrej Karpathy** as saying that *the best way to do AI is just to try out a bunch of model solutions and see what works best*.
- ****Cutting up embeddings in multi-head attention is okay?****: A member questioned the practice of splitting **512-dimensional embeddings** into smaller heads in multi-head attention, expressing concern that it might lose context.
   - Others explained that it's a form of **regularization** that allows the model to specialize and learn more robust features, with connections before and after each attention step.
- ****Matrix > Discord?****: The community discussed the possibility of moving to **Matrix** for better support for reading groups, citing its lack of a hardcoded limit on the number of rooms and its federated nature, allowing users to join from different servers.
   - Some questioned the need for decentralized channels, while others pointed out the value of interoperability.
- ****Sampling reverse time SDEs causes inconsistency?****: A member reported vastly different performance across separate training runs of a diffusion model with the same repo, config, data, and seed, which was attributed to the random element in **reverse time SDE sampling**.
   - It was suggested that the approximated distribution the model learns looks the same, but without the same seed, a bad generated batch may be observed; a badly designed guidance will also affect this.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1433890642584535181)** (17 messages🔥): 

> `Paper reading recordings, Linear Interpretable Feature Evolution, Analog vs Digital, Awesome World Models` 


- **Awesome World Models suggested**: A member shared a link to a [GitHub repository](https://github.com/knightnemo/Awesome-World-Models) of **World Models** to be read and discussed.
- **Linear Interpretable Feature Evolution Across Pre-training Snapshots**: A paper suggestion was made for discussing [Linear interpretable feature evolution across pre-training snapshots using crosscoders](https://arxiv.org/abs/2509.17196) and its corresponding [tweet](https://fxtwitter.com/Dest1n1s/status/1970350739152408784?t=UaWoNGMU0x2_g0DpIIG0kw&s=19).
- **Inquiry about Paper Reading Recordings**: A member asked if there was a recording of a previous **paper reading** session.
- **Analog Better than Digital?**: A member brought up the question of whether **analog** is better than **digital**, referencing the workings of the human brain.


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1434232845806735521)** (2 messages): 

> `Advancing Agents, Reasoning, Memory, AGI discussions` 


- **Talk Inspires Agent Advancement**: A member expressed gratitude for a discussion that sparked connections between mechanisms vital for advancing the functionality of **Agents**, **Reasoning**, and potentially **Memory**.
   - The discussion was lauded as a functional step forward, evoking an *AGI feel*, with hopes for continued exploration of its potential.
- **AGI Potential Highlighted**: Influential forum members suggested the talk represented a step towards **AGI**, emphasizing its significance in the field.
   - Participants voiced optimism about the possibility of sustaining the momentum of the discussion and unlocking further advancements.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1434068287909527583)** (10 messages🔥): 

> `AI bubble, Economic downturn, Weakening labor` 


- **AI Bubble Blamed for Job Crisis**: A [Fortune article](https://fortune.com/2025/10/30/jerome-powell-ai-bubble-jobs-unemployment-crisis-interest-rates) discusses how the **AI bubble** could cause **job losses** and an **unemployment crisis**.
   - Members debated whether **AI** or **government incompetency** is more responsible for the current **economic downturn**.
- **Members Link Papers on AI Impact**: Members shared links to papers such as [Canaries in the Coal Mine: Early Warnings of Technology-Driven Displacement](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5425555) from the **Stanford Digital Economy Lab**.
   - The papers discuss the impacts of **AI** on the job market.
- **Wages Decouple from Productivity**: Members linked the [Wikipedia article on the decoupling of wages from productivity](https://en.wikipedia.org/wiki/Decoupling_of_wages_from_productivity), attributing it to **weakening labor's negotiating power**.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1433867780653846578)** (110 messages🔥🔥): 

> `Kimi CLI, Agent Mode, DeepAgents CLI, Poolside valuation, Redpanda Data AI` 


- ****Kimi CLI** Launches Terminal-Focused Tool**: **Moonshot AI** released a technical preview of **Kimi CLI** with **Zsh integration**, **MCP support**, and native hooks into the **Zed editor** and the [GitHub repo is open for feedback](https://xcancel.com/Kimi_Moonshot/status/1984207733177090274).
   - VIP users get the new **"Kimi For Coding"** add-on at no extra cost; early replies note **401-errors** on setup, enthusiasm for terminal workflows, and requests for Windows support and trial plans.
- **OpenAI's Agent Mode Sparks Hype and Hesitations**: **OpenAI** announced the preview release of **Agent/Atlas mode** for **ChatGPT** (Plus/Pro/Business), enabling the model to browse and take actions on behalf of users; see [announcement](https://xcancel.com/OpenAI/status/1984304194837528864).
   - Concerns voiced include prompt-injection attacks, lack of clear guardrails, reliability issues, and the ethical line between helpful automation and privacy erosion.
- **LangChain Unleashes **DeepAgents CLI****: **Harrison Chase** introduced **DeepAgents CLI**, a sample coding application built on the new deepagents package that retains instructions and guidance across sessions; see [Langchain blog post](https://xcancel.com/hwchase17/status/1984303925101735950).
   - Positioned as an **"open harness"** for customizable agents, community members are already asking about MCP integration and external memory sources like vector databases.
- **Doubts Cast on **Poolside's $12B Valuation****: **Julien Blanchon** tweeted *"Okay short everything !"* alongside a photo, starting a thread where French tech insiders mock Poolside’s **$12B valuation** and call them out for pivoting multiple times.
   - Commenters pointed out the company pitched **“Cursor before Cursor”** but never shipped, pivoted multiple times (SFT-as-a-service, RL-as-a-service, now “Cursor-as-a-service”), and is barely visible in Paris meetups—leading to accusations it’s vaporware run from Caribbean tax havens; see [tweet here](https://xcancel.com/julienblanchon/status/1984337407097909629?s=46).
- ****AI Flippening** Arrives as Chinese Open-Source Models Surge**: **Balaji Srinivasan** declared the **“AI flippening”** has arrived, claiming Chinese open-weight models (**DeepSeek**, **Qwen**, etc.) now out-download and increasingly out-perform Western rivals, commoditizing AI software and squeezing profit margins; see [tweet here](https://xcancel.com/balajis/status/1984421276082192663?s=46&t=Z6mP_1pHALnIw7k1lFkdwQ).
   - He argues China’s strategy is to bankrupt US AI firms with free/cheap models, then monetize AI-enabled hardware, but the move is debated whether this is measured by downloads vs. revenue, the West’s energy deficit, backdoor risks, and the next wave of innovation under an open-source dominant regime.


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 messages): 

swyxio: new pod with <@367104793292046338> and <@194927177265840128> ! https://youtu.be/-gE1cesJF9M
  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1434049373708357754)** (5 messages): 

> `X-Ware.v0, OpenAI X post, YouTube video` 


- **OpenAI makes X Post**: A member shared [a link](https://x.com/openai/status/1984318204374892798?s=46) to **OpenAI's** X post.
   - Another member shared the same link, perhaps highlighting its significance.
- **X-Ware.v0 shared in channel**: Members shared the term `Red - X-Ware.v0` in the chat, without any other context.
   - It is unclear what **X-Ware.v0** refers to, though it may be related to the **OpenAI** X post shared.
- **YouTube video posted**: A member posted a link to [a YouTube video](https://www.youtube.com/watch?v=YpuSE9hcal8).
   - A user commented that *His videos are always pretty neat*.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1433908333470810162)** (98 messages🔥🔥): 

> `Kimi K2 Research & OK Computer Reset, K2 Think Model vs. Cerebras Confusion, Qwen QWQ Finetune, Minimax vs. GLM for Daily Tasks, Claude Code Max vs. Cursor Pro+ Usage Limits` 


- **Kimi's Research and OK Computer Quotes Remain**: A user confirmed that the **OK computer** and **Researcher Quote** do not reset after a month, indicating it's likely a one-time allowance.
   - Further context was not provided about the specifics of these features or how they are obtained.
- **K2 Think Causes Cerebras Confusion**: A user clarified that **K2 Think** should not be confused with a model hosted on **Cerebras**.
   - Another user found it odd that the name was chosen, considering the existing K2, and suggested that the model isn't performing well anyway.
- **Is Kimi a Qwen QWQ Finetune?**: Users speculated that **Kimi** might be a **Qwen QWQ finetune**, with one noting it resembles **Qwen QWQ** and could be a post-trained dataset.
   - Another user stated **Qwen QWQ** is based on **Qwen 2.5 32B** and linked to [a YouTube video](https://www.youtube.com/watch?v=l3d-m3uP3nQ) regarding an 'ancient model' based on QWQ.
- **Minimax gains popularity**: A user shared after 4-5 days of use, they find **M2 is better than GLM-4.6** for most tasks, especially as a daily driver, because it doesn't get tunnel-visioned.
   - Another user reported that **Minimax** has become their go-to, especially for web searches and creating reports in specific formats that other AIs struggle with.
- **Code Completion Subscriptions Debated**: Users discussed usage limits of various coding tools, one comparing **Claude Code Max** and **Cursor Pro+** and stating that **Claude Code Max ($200)** has better limits.
   - A user noted that **Cursor Composer** is fast, but another clarified that **Claude Code Max** is the $200 option, leading to discussions about weekly usage.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1434081123809497168)** (10 messages🔥): 

> `Multi-Head Attention, EleutherAI Contribution, Mentorship` 


- **Multi-Head Attention dims not so dim**: Members discussed [multi-head attention](https://transformer-explained.github.io/attention) and how the **512-dim embeddings** are projected into **Q, K, and V** and split into 8 heads of 64 dimensions each.
   - They suggest that the model is trained with that slicing with the dimensions, meaning you could *permute along that dim and then slice and get exactly the same result after training*.
- **EleutherAI engagement welcome**: A member asked how to engage or contribute to the research of **EleutherAI**, being new to open source and Discord collaboration.
   - Another member pointed to the **pinned messages** and the **#general channel** for guidance.
- **ML/DL mentorship being sought**: A member is seeking an experienced professional for **high-level mentorship** in **ML/DL** to level up their skills and receive occasional feedback.
   - They emphasized being self-motivated and respectful of the mentor's time, offering to share their ongoing projects.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1433990171392606368)** (14 messages🔥): 

> `RL Collapse, Flash Attention, Gradient Normalization, LLM-RL Libraries, HF models hallucinating` 


- **Speed Kills Stability in RL Training**: A member shared a [paper](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda) discussing how **RL collapse** can occur due to the mismatch between training and inference.
- **Gradient Normalization rewrites Layer Op**: A member suggested that normalizing gradients before reduction in the backwards pass (spectral norm, L2, etc.) should be easily doable by rewriting the linear layer op and adding the reduction immediately following that.
- **veRL Offers Quick Hackable RL Research**: A member asked for LLM-RL libraries besides TRL for active algorithmic research, and another recommended [veRL](https://github.com/volcengine/verl).
- **Qwen Models Hallucinate Long Tail Facts**: The results of an evaluation of the most downloaded HuggingFace models revealed that **Qwen models** tend to hallucinate long tail facts and that some of the more popular models aren't that great at following instructions; full results can be found on the [HuggingFace space](https://huggingface.co/spaces/PropensityLabs/LLM-Propensity-Evals).


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1433836454244520137)** (26 messages🔥): 

> `Transformers on Sequence Space, End-to-End LLM Outputs, Mech Interp and Probing Work, LLM Privacy / Security Claims, Activation Sharing Risks` 


- **Transformers Map Sequence Space, Not Embedding Space**: A member stated that transformers are viewed as maps on the **sequence space**, not on embedding space, disagreeing with a mentioned paper's perspective.
   - They questioned the intended audience of the paper's argument, implying it misrepresents the common understanding of transformers.
- **LLMs Falsely Claimed End-to-End by paper**: A member criticized a paper's claim of "end-to-end," arguing that the output of an LLM is a **sequence of tokens** or natural language, not a hidden state, and this invalidates the claim.
   - They asserted that this end-to-end process fails when the output is a sequence of tokens or a string of natural language.
- **LLM Privacy Claims Misrepresent the Cited Article**: A paper's privacy and security claims were criticized for misrepresenting the cited article, which discusses **LLM weights** storing information from training data, not user inputs.
   - The member emphasized that the cited article's abstract explicitly mentions compliance with **GDPR** requirements, contradicting the paper's interpretation.
- **Concept Injection Detectability Detailed**: A member elaborated on how models detect injected concepts via anomaly detection, explaining that models detect when a perturbation pushes activations away from the expected internal representation.
   - They propose that the model’s activations traverse a fairly smooth semantic manifold, so that when an injected concept introduces a vector that is not aligned with that local manifold downstream mechanisms could detect that deviation.
- **Advice Sought for Interpretability Research Career**: A member expressed interest in transitioning to applied research in **interpretability and AI safety**, seeking advice on papers to read and projects to join.
   - They mentioned their background in NLP and computational linguistics and their recent focus on AI engineering, while also sharing their go-to resource, the ["best of less wrong"](https://www.alignmentforum.org/bestoflesswrong?year=all&category=all).


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1434296011890888745)** (1 messages): 

> `MMLU Benchmark, Image Analysis` 


- **MMLU Benchmark Criticized in Image**: An image criticizing the **MMLU benchmark** was shared, labeled *"anti mmlu pro slander"*.
   - The image suggests a negative sentiment towards the benchmark.
- **Image Analysis Context**: The shared image is related to a discussion or sentiment about the **MMLU benchmark** within the community.
   - It implies a critical perspective on the benchmark's utility or validity.


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1434526408700465265)** (2 messages): 

> `VLM image description order, VLMs describing image collages` 


- **VLMs Describe Collages Left-to-Right**: A member reported that when using VLMs to describe collages of images, the descriptions consistently follow a left-to-right order, even with a Qwen 2 VL model finetuned on Arabic data (**AIN**).
   - Another member suggested investigating the VLM architecture, focusing on how images are processed and integrated with text, to understand and potentially address this behavior, and to *experiment* to test hypotheses.
- **Understanding VLM Architecture for Image Processing**: To address the consistent left-to-right description order in VLMs, a suggestion was made to **study the VLM architecture** and its components, especially how images are considered and integrated with text.
   - The aim is to develop hypotheses that can be tested through experimentation to understand why VLMs prioritize this specific order when describing collages.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1433945748466045088)** (45 messages🔥): 

> `setup.py vs pyproject.toml, UOp.pyrender() bug, Tenstorrent backend bounty, Tiled matrix multiplication, PyTorch backend without hacks` 


- **Classic setup.py Debate Continues**: Discussion arose around the use of `setup.py` instead of `pyproject.toml` in the tinygrad repo, inquiring if there was a specific reason beyond legacy.
   - A contributor sought help to add `argfix` to the `uop` movement ops and include a reasonable test.
- **Unreferenced Variables Plague UOp.pyrender()**: A user reported a bug where the result of `some_forward().uop.pyrender()` contains unreferenced variables, such as `cxx = UOp(Ops.VECTORIZE` lines that are not used anywhere.
   - It was clarified that the output of `pyrender` is meant to be directly executable and produce the same `uop`.
- **Tenstorrent Backend Bounty Awaits Contender**: Interest was expressed in **TinyGrad** supporting **Tenstorrent**.
   - It was noted that there is a [bounty](https://blinry.org/tiny-linux/) for a **Tenstorrent backend**, but nobody has attempted it yet; one user got **TinyGrad** running in it on a statically linked Python, but only **PYTHON** and **NULL** backends worked.
- **Matrix Multiplication Bottleneck on Hardware without Tensor Cores**: A question was raised about implementing tiled matrix multiplication for hardware without tensor cores in **TinyGrad**, as matmul is currently very slow on such hardware.
   - It was also confirmed that the flash attention bounty is being implemented.
- **Fix the PyTorch Backend Strides PR**: A newcomer asked about the tasks in the [spreadsheet](https://docs.google.com/spreadsheets/d/1WKHbT-7KOgjEawq5h5Ic1qUWzpfAzuD_J06N1JwOCGs/edit?gid=0#gid=0) and its up-to-date status, pointing out a likely corresponding but unstated PR (https://github.com/tinygrad/tinygrad/pull/13061) for *Fix the PyTorch backend without hacks for strides*.
   - The user later opened a WIP PR for fixing the strides, but was asked to wait until the tests are passing before opening PRs.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1433891553910194206)** (40 messages🔥): 

> `Manus credit costs, Claude Code vs Manus, Manus image generation quality, Manus assistance with Instagram reels, Manus custom domain pricing` 


- **Users Compare Manus Credit Costs to Alternatives**: Several users expressed concern about the high cost of **Manus credits**, with one noting they blew through **6k credits in about an hour** on a mid-complexity project.
   - Users suggest that a **$20** subscription with *Claude Code* and *GPT Codex* offers better value than Manus. One user stated, *"manus is extremely more expensive compared to other options"*.
- **Claude Code Delivering Strong Performance**: A user is happy with *Claude Code*, mentioning its ability to deliver results, especially for coding, and their new trivia game with **24 categories and 4k+ questions**.
   - The user plans to alternate between *Claude Code* and *GPT Codex* when rate-limited, estimating around **8 hours of constant coding per day for 5-6 days a week**.
- **Manus Generates Low Quality Images**: A user questioned the consistently low quality of images generated by Manus and shared a [session link](https://manus.im/share/dRrj3dwepWuDcJKvfxRHPK?replay=1) to demonstrate the issue.
   - Despite requesting higher quality for a second mental map, the result remained the same.
- **User Reports Manus Fails to Explain Instagram Reels**: A user reported that Manus, which previously explained an Instagram reel, now refuses to do so.
   - No explanation was provided for the inconsistent behavior.
- **Users call $200 Custom Domain Pricing a Rip Off**: A user criticized the **$200/month subscription** fee for connecting a custom domain to a web app via Manus, deeming it a *"rip off"*.
   - Another user suggested simply buying a domain and setting it up independently as a cheaper alternative.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1433846431864721520)** (33 messages🔥): 

> `Brokk AI Power Ranking, Perplexity MCP, aider vs aider-ce, Aider Community, Entangled Pair Quantum Eraser` 


- ****GPT-mini** Scores High on **Brokk AI Power Ranking****: The [Brokk AI power rankings](https://brokk.ai/power-ranking) were updated and one user asked if **GPT-mini** was S tier above **Claude**.
   - Another user responded yes, but that was *before* **Sonnet** and **Haiku 4.5** came out, adding that the results are questionable.
- ****Perplexity MCP** integrated with Aider**: One user found [Perplexity's MCP](https://www.perplexity.ai/) useful for finding **GitHub** issues related to an abandoned **Android** library and integrating them with **aider-ce**.
   - The user wondered if having the **MCP** integrated could automate the process, though noted that manual review is sometimes necessary.
- ****Aider-ce** branch emerges as active fork**: Users noted that **aider-ce** is active while the main **aider** repo is not actively updating, and one user clarified that one of the primary maintainers advised he's been busy.
   - A user asked *what does the ce version do that you like?* because one of the recent issues asked to not override the **aider** command so someone could have both installed.
- ****Aider** community needs to be built**: One user asked if it were possible to build a community around **aider**, noting that while *we all love it*, it requires people who are willing to put time into it.
   - Some also mentioned that they don't use it as much anymore due to other more agentic products being available and that **aider** could use a context management UI and **MCP** integration.
- **Paul is building a **Quantum Aider****: In response to a user joking that Paul is creating a quantum version of **aider**, Paul linked to [his project](https://github.com/paul-gauthier/entangled-pair-quantum-eraser).
   - Another user responded with concern, asking how to retain Paul's *deep knowledge of the project* as others make progress, fearing the *current vacuum is causing a loss of users*.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1434888049724231761)** (5 messages): 

> `aider-ce, reasoning-effort, weak_model` 


- **Check out updated Aider-CE**: A member recommended checking out [aider-ce](https://github.com/aider-chat/aider-ce), noting it's updated weekly and builds on Aider's strengths while adding more features, see [roadmap](https://github.com/aider-chat/aider-ce/blob/main/README.md).
   - They suggested starring the repo to show support.
- **How to set reasoning-effort and weak_model**: A member inquired how to set `/reasoning-effort` for `--model ollama_chat/gpt-oss:20b` and `--weak_model`.
   - Even setting `/weak_model ollama_chat/qwen3:1.7b` didn't seem to work, resulting in a warning that `ollama_chat/gpt-oss:20b` does not support `reasoning_effort`.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1433846353708191888)** (21 messages🔥): 

> `MCPB, OCI, DXT, MCP Registry, server.json` 


- **MCPB vs. OCI: Reinventing the Wheel?**: A member questioned whether **MCPB** is simply *reinventing* what **OCI** already offers.
   - Another member clarified that **MCPB** is an **Anthropic** project (**formerly DXT**) for exposing **MCP servers** to **Claude**, providing descriptions and types of environment variables for a user-friendly configuration form, unlike the generic `servers.json` or `mcp.json`.
- **MCP Registry Embraces MCPB**: Despite initial confusion, it was confirmed that the **MCP Registry** supports **MCPB**.
   - It was explained that the registry aims for broad support of various registries and package types, similar to supporting **npm** or **PyPI**.
- **Configuration Differences: MCPB vs. server.json**: The discussion highlighted that **MCPB** is designed for desktop apps, presenting a configuration form for variables, while `server.json` typically defines variable values directly.
   - An example from the registry showed how `server.json` already includes descriptions and types, suggesting the registry could expand on this functionality.
- **OCI Awareness Among MCPB Creators**: A member speculated that the creators of **DXT/MCPB** might not have been fully aware of **OCI** and the existing work in the registry working group.
   - It was suggested they may have prioritized a user-friendly package with form-filling capabilities over expecting users to configure a **JSON** file directly.


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1433937677706727674)** (7 messages): 

> `SEP-1442, SEP-1686, Statelessness proposals, Task storage` 


- **Statelessness Proposals Debated**: Members debated whether **SEP-1442** and **SEP-1686** conflicted, given one aimed to make servers more stateless, while the other introduced state tracking.
   - One member argued they don't conflict because Tasks can scale across servers, while **SEP-1442** moves session info into each request for statelessness by default, which is geared toward challenges of hosting MCP servers behind a load balancer.
- **Statelessness Aiming for Default, not Complete**: A member clarified that the statelessness proposal (**SEP-1442**) aims for statelessness by default, making statefulness opt-in rather than the default.
   - Storing supported protocol versions and capabilities in an external data store complicates non-session servers, which introduces new update operations to solve that, making it simpler to store everything in the request.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1434292218616877178)** (1 messages): 

> `DSPyGen, DSLModel, Contributions to DSPy` 


- **Sean Chatman Releases DSPyGen**: Sean Chatman has released [DSPyGen](https://github.com/seanchatmangpt/dspygen), a new tool for DSPy development.
   - He is also the author of [DSLModel](https://github.com/seanchatmangpt/dslmodel).
- **User Offers DSPy Contributions**: A member has expressed interest in contributing to DSPy and offered their assistance.
   - They mentioned they have been looking to get back to DSPy development and have worked on many variations of the project.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1433843785762209843)** (19 messages🔥): 

> `dspy.Tool with simple Predict, Rate limit issues for Gemini, DSCloj channel in DSPy, Force finish the ReAct, Accessing LM a module is using` 


- **Predict beats ReAct for simple tasks**: One member found `dspy.Tool` with simple Predict to be sufficient and more efficient than ReAct for their use case, reducing response time from **60s to 9s**.
   - They stated it was *"overkill for my useage"*.
- **Gemini Rate Limit Headaches**: A member reported hitting Gemini's **1M token/min rate limit** even with 10 parallel workers, asking for best practices on mitigation in production.
   - Another member suggested that the issue may be due to hitting the **requests per day limit** or the **requests per minute limit** depending on your tier, which can be checked in Google AI Studio.
- **DSCloj Channel Creation**: A request was made for a dedicated channel for DSCloj within the DSPy sibling projects, following patterns used by Rust and Typescript channels.
   - The channel was created, with some back and forth on naming conventions before settling on a suitable name.
- **Forcing ReAct to finish**: One member inquired about forcing the ReAct module to finish, potentially based on the tool's return, but didn't get an immediate answer.
   - The user wanted the agent to return the final answer as soon as they were returning from the tool, but it kept going after that.
- **LLM Access for Dynamic Model Switching**: A member sought guidance on accessing the LLM used by a module to enable dynamic model switching in case of rate limits, asking how to pass an LLM to a module and ensure a fallback mechanism.
   - The recommended solution was to pass a `dspy.LM` object to the module's init and conditionally use one LLM or another, allowing for conditional fallback in case of errors.

