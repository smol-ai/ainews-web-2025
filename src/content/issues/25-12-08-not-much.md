---
id: MjAyNS0x
title: not much happened today
date: '2025-12-08T05:44:39.731046Z'
description: >-
  **Claude Code Skills** gains attention with a published talk and Hugging
  Face's new "skill" enabling one-line fine-tuning pipelines for models from
  ~0.5B to 70B parameters, supporting SFT, DPO, and GRPO, costing as low as
  ~$0.30 for small runs. **Zhipu AI** launches multimodal models **GLM-4.6V**
  (106B params MoE) and **GLM-4.6V-Flash** (9B dense), featuring 128k context
  and native multimodal function calling, with free Flash variant and API
  pricing detailed. **Jina AI** releases **Jina-VLM (2B)**, a compact
  multilingual VLM excelling in diagrams and documents with top benchmark
  scores. At **NeurIPS 2025**, research highlights include Google's
  post-Transformer sequence architectures (Moneta, Yaad, Memora) showing up to
  20% gains in long-context retrieval, **AxiomProver**'s autonomous Lean system
  solving 9/12 Putnam 2025 problems rapidly, and mechanistic interpretability
  advances discussed by Chris Olah emphasizing scalable tooling.
companies:
  - hugging-face
  - zhipu-ai
  - jina-ai
  - google-deepmind
  - axiomprover
models:
  - glm-4.6v
  - glm-4.6v-flash
  - jina-vlm-2b
topics:
  - fine-tuning
  - multimodality
  - model-optimization
  - long-context
  - mechanistic-interpretability
  - formal-methods
  - sequence-architectures
  - reinforcement-learning
people:
  - lioronai
  - akshay_pachaar
  - _akhaliq
  - ben_burtenshaw
  - vllm_project
  - prince_canuma
  - zenmuxai
  - eliebakouch
  - theturingpost
  - axiommathai
  - neelnanda5
  - sarahookr
---


**a quiet day**

> AI News for 12/5/2025-12/8/2025. We checked 12 subreddits, 544 Twitters and 24 Discords (205 channels, and 16871 messages) for you. Estimated reading time saved (at 200wpm): 1319 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Lots of excitement about Claude Code Skills, which now [has a published talk](https://www.youtube.com/watch?v=CEvIs9y1uog&t=538s), being able to [finetune AI models](https://huggingface.co/blog/hf-skills-training).

---

# AI Twitter Recap

**Automating open LLM training with Claude Code + Hugging Face Skills**

- **“One-line” fine-tuning pipelines, end-to-end**: Hugging Face released a Claude Code “skill” that lets you specify training in natural language (“Fine-tune Qwen3-0.6B on open-r1/codeforces-cots”) and have the agent do the rest: validate datasets, choose GPU types, kick off HF Jobs, monitor progress, and publish checkpoints/models to the Hub. Supports SFT, DPO, and GRPO on models from ~0.5B to 70B, with options like GGUF export and multi-stage pipelines. Early reports claim small runs can cost ~$0.30. See demo and details from [@LiorOnAI](https://twitter.com/LiorOnAI/status/1997754848255807874), the HF blog link shared [here](https://twitter.com/LiorOnAI/status/1997754850927689929), and a deeper rundown by [@akshay_pachaar](https://twitter.com/akshay_pachaar/status/1997946287556321359).
    
    Why it matters: it collapses a lot of bespoke glue (infra selection, dataset plumbing, logging, artifact pushes) into a reproducible, auditable agentic workflow powered by HF Jobs + Hub.
    

**New multimodal models: Zhipu’s GLM‑4.6V and Jina‑VLM**

- **Zhipu AI’s GLM‑4.6V and GLM‑4.6V‑Flash**: New VLMs with 128k context and native multimodal function calling. GLM‑4.6V is a MoE with 106B total params and ~12B active; Flash is a 9B dense variant tuned for latency and local deployment. Pricing (API, per 1M tokens): $0.6 input / $0.9 output; Flash is free. Weights are live on HF; vLLM shipped day‑0 recipes; MLX‑VLM added support; several platforms integrated tool-calling with image arguments (“no OCR detour”). Launch and specs: [@Zai_org](https://twitter.com/Zai_org/status/1998003287216517345), HF weights: [@_akhaliq](https://twitter.com/_akhaliq/status/1998052965597241647), MoE details: [@ben_burtenshaw](https://twitter.com/ben_burtenshaw/status/1998019922664865881), vLLM recipe: [@vllm_project](https://twitter.com/vllm_project/status/1998019338033680574), MLX‑VLM: [@Prince_Canuma](https://twitter.com/Prince_Canuma/status/1998024143212851571), tool-calling integration: [@ZenMuxAI](https://twitter.com/ZenMuxAI/status/1998018534736343495). Early comments note small gaps between the 106B and 9B variants on some benches ([@eliebakouch](https://twitter.com/eliebakouch/status/1998015034979389563)).
- **Jina‑VLM (2B)**: A compact multilingual VLM focused on diagrams, charts, scene text and documents. Jina claims SOTA among open 2B VLMs with an average 72.3 across eight VQA benchmarks and best-in-class on MMMB (78.8) and Multilingual MMBench (74.3). Paper/code/resources via [@JinaAI_](https://twitter.com/JinaAI_/status/1997926488843190481) and follow-ups [1](https://twitter.com/JinaAI_/status/1997926493456834978) [2](https://twitter.com/JinaAI_/status/1997926495688249836).

**NeurIPS 2025 research signals: new sequence architectures, interpretability, and formal methods**

- **Post‑Transformer backbones (Google’s Miras framework)**: A Google paper reframes Transformers and RNNs as associative memory systems and “forgetting” as retention regularization, introducing Moneta, Yaad, Memora. Authors report wins vs Transformers, Mamba2, DeltaNet, and hybrids on LM, reasoning, long‑context scaling, and needle‑in‑a‑haystack recall, with up to ~20% gains in long-context retrieval. Overview threads and paper: [@TheTuringPost](https://twitter.com/TheTuringPost/status/1997808277116338266) and arXiv link [here](https://twitter.com/TheTuringPost/status/1997808369437196480).
- **Formal methods meet frontier math**: AxiomProver reports an autonomous Lean system that solved 9/12 Putnam 2025 problems within hours of the exam’s end — a performance they say would top last year’s scoreboard. They emphasize verifiability and hybrid formal/informal pipelines; see [@axiommathai](https://twitter.com/axiommathai/status/1997767850279440715) and discussion of “AI mathematicians” by [@TheTuringPost](https://twitter.com/TheTuringPost/status/1997971709996212561).
- **Mech interp at scale, more actionability**: The mechanistic interpretability workshop drew large crowds; Chris Olah spoke on “reflections on interpretability” ([@NeelNanda5](https://twitter.com/NeelNanda5/status/1997812818788467157)). Field sentiment highlighted the need for scalable, generalizable tooling over single-model neuron anecdotes ([@sarahookr](https://twitter.com/sarahookr/status/1997795206096429415)).
- **Agent evaluation in realistic tool-rich settings**: MEMTRACK probes long-horizon memory/state tracking by placing agents in a “workplace” with Slack, Linear, and git timelines; best reported score ~60% (GPT‑5) on their tasks, underscoring headroom for improvement ([@rebeccatqian](https://twitter.com/rebeccatqian/status/1997813556717522996)).

**Agents in practice: evaluation, reliability, and knowledge grounding**

- **Deep Agents evaluation patterns and results**: LangChain published practical patterns for evaluating long-running agents (planning, FS, sub‑agents, prompting), plus an agent CLI benchmark on Terminal Bench 2.0 (mean ~42.65%). They also shipped dynamic context compaction triggers (e.g., summarize at 85% window, retain 10%) and a LangSmith video series on observability/eval/deploy for agent systems. Resources: blog and results by [@LangChainAI](https://twitter.com/LangChainAI/status/1997843687376904400), context compaction by [@sydneyrunkle](https://twitter.com/sydneyrunkle/status/1998011509482647676), LangSmith series [1](https://twitter.com/LangChainAI/status/1998091825643504032) [2](https://twitter.com/hwchase17/status/1998176795737383033).
- **Consumer-grade grounding is still weak**: The ACE benchmark targets everyday tasks (shopping/food/gaming/DIY) with dynamic checking of grounded web sources; leading scores are modest (GPT‑5 High=56.1%, o3 Pro=55.2%; Shopping tops out at 45.4%). Some models score negative on link accuracy (e.g., Gemini 3 Pro -54% “Provides link[s]”). Paper and analysis via [@omarsar0](https://twitter.com/omarsar0/status/1998039629556256995).
- **Tooling and workflows**: Dexter 2.0 (open-source, LangChain-based) targets autonomous financial research with planning/self‑validation ([#demo](https://twitter.com/virattt/status/1997770360209453322)). AI21 Maestro positions orchestration with multi‑step planning, built‑in validation, proprietary RAG, and execution graphs ([@AI21Labs](https://twitter.com/AI21Labs/status/1998014705638523267)). DSPy’s GEPA continues to show large deltas when integrated quickly into new tasks (reported 12.5%→62.5% on a workshop entry) ([@DSPyOSS](https://twitter.com/DSPyOSS/status/1997879916583391705)).
- **On “LLMs can’t generate knowledge”**: Counterpoint from [@jeremyphoward](https://twitter.com/jeremyphoward/status/1998177975376986575) shows tool-augmented LLMs deriving previously undocumented results, arguing agents can generate new knowledge through interaction.

**Infrastructure and serving: open stacks and systems updates**

- **RadixArk (from the SGLang creators)**: A new infra-first venture aiming to “make frontier-level AI infrastructure open and accessible,” spinning out of the SGLang ecosystem. Emphasis on scheduling/compilers/serving/training pipelines as shared infrastructure rather than re‑implemented per org. Announce and endorsements: [@ying11231](https://twitter.com/ying11231/status/1998079551369593222), [@ibab](https://twitter.com/ibab/status/1998098312051011817), reaction by [@eliebakouch](https://twitter.com/eliebakouch/status/1998081613213954475).
- **Notable systems bits**: Mesh‑oriented sharding insights from [@ezyang](https://twitter.com/ezyang/status/1997902916384932112). Qdrant’s ACORN improves filtered vector search recall without predicate‑specific indexes ([@qdrant_engine](https://twitter.com/qdrant_engine/status/1997939453965336741)). Turbopuffer doubled WAL scan speed for strong consistency under async indexing ([@turbopuffer](https://twitter.com/turbopuffer/status/1998058954149208096)). Weaviate’s Multi2Vec 1.5 adds MetaCLIP2, ModernVBERT, and Jetson support ([@weaviate_io](https://twitter.com/weaviate_io/status/1998060177501614130)). HF x Google Cloud partnership touts 5GB in ~13s transfers ([@ClementDelangue](https://twitter.com/ClementDelangue/status/1998157804020941044)). Yupp added SVG leaderboards and launched a community contest ([@yupp_ai](https://twitter.com/yupp_ai/status/1998120413285769302)).

**Top tweets (by engagement)**

- “We’re in an ‘LLM bubble’ not an AI bubble.” A macro take separating platform hype from broader AI advances ([@hardmaru](https://twitter.com/hardmaru/status/1997778363625488502)).
- Zhipu’s GLM‑4.6V launch: open weights, 128k multimodal context, native function calling, and a latency‑optimized 9B Flash variant ([@Zai_org](https://twitter.com/Zai_org/status/1998003287216517345)).
- Linus Torvalds on AI: bubble sentiment, big impact on skilled work, skepticism of “vibe coding” maintainability, market froth likely to crash ([@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1998090820897947843)).
- AxiomProver claims Putnam‑level performance: 9/12 problems solved in Lean within hours of the exam ([{AxiomMath AI}](https://twitter.com/axiommathai/status/1997767850279440715)).
- Andy Jones’s “horses” analogy: tech progress can feel sudden at thresholds; internal metrics at Anthropic showed Claude rapidly absorbing new‑hire Q&A load ([thread](https://twitter.com/andy_l_jones/status/1998060552565002721)).
- Waymo expands to London in preparation for commercial service in 2026 ([@Waymo](https://twitter.com/Waymo/status/1998075104752713981)).
- Clay hits $100M ARR in two years after $1M, with >200% enterprise NRR; GTM learnings on reverse demos, usage‑based pricing, and brand bets ([@vxanand](https://twitter.com/vxanand/status/1998037723458810129)).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. GLM-4.6V Model Releases and Features

- [**zai-org/GLM-4.6V-Flash (9B) is here**](https://www.reddit.com/r/LocalLLaMA/comments/1pha7l1/zaiorgglm46vflash_9b_is_here/) (Activity: 450): **GLM-4.6V-Flash (9B) is a lightweight model designed for local deployment, featuring a** `128k` **token context window and achieving state-of-the-art (SoTA) performance in visual understanding for models of similar size. It introduces native Function Calling capabilities, enhancing the integration of visual perception with executable actions, thus supporting multimodal agents in real-world business applications. More details can be found on [Hugging Face](https://huggingface.co/zai-org/GLM-4.6V-Flash).** Commenters appreciate the focus on sub-10B model sizes, noting that only a few companies like **Mistral**, **Qwen**, **zAI**, and **IBM** are maintaining this trend. There is also anticipation for larger models, such as a `30-40B` MOE model, which some users expected but found missing.
    - The release of the GLM-4.6V-Flash (9B) model by zAI is notable for maintaining a focus on sub-10B model sizes, a trend also seen with companies like Mistral, Qwen, and IBM. This suggests a continued interest in optimizing smaller models for specific applications, balancing performance and resource efficiency.
    - A user expressed anticipation for a larger 30-40B MOE (Mixture of Experts) model, which was also absent in recent releases from Mistral. This highlights a demand for more powerful models that can handle complex tasks, indicating a gap in the current offerings of AI models.
    - The GLM-4.6V model is available in a text-only GGUF format, which is currently in production. However, the vision capabilities are not yet available as the pull request for this feature is still in draft. This indicates ongoing development and potential future enhancements to the model's capabilities.
- [**GLM-4.6V (108B) has been released**](https://www.reddit.com/r/LocalLLaMA/comments/1phaaon/glm46v_108b_has_been_released/) (Activity: 480): **GLM-4.6V has been released, featuring two versions: the** `GLM-4.6V (106B)` **for cloud and high-performance clusters, and the** `GLM-4.6V-Flash (9B)` **for local, low-latency applications. It supports a** `128k token` **context window and achieves state-of-the-art (SoTA) performance in visual understanding among models of similar scale. Notably, it introduces Native Multimodal Function Calling, allowing direct use of images and visual outputs in reasoning processes, and supports Interleaved Image-Text Content Generation and Multimodal Document Understanding. The model can also replicate and edit frontend interfaces from screenshots using natural language. [GLM-4.6V on Hugging Face](https://huggingface.co/zai-org/GLM-4.6V).** Commenters discuss the implications of adding vision capabilities to text models, questioning the potential trade-offs in text performance. There is curiosity about how GLM-4.6V compares to its predecessor, GLM-4.5-Air, in text-only tasks, with a general consensus that integrating vision might impact non-vision task performance.
    - A user questioned the impact of adding vision capabilities to a text model, specifically how it might degrade text performance. They noted that while vision models are versatile, they may not perform as well on text-only tasks compared to specialized text models like GLM-4.5-Air. This highlights the trade-off between adding new capabilities and maintaining performance in existing ones.
    - Another user compared the performance of the 9B and 108B models, noting that the smaller model is not significantly worse according to benchmarks. This raises questions about the efficiency and practicality of using larger models when smaller ones can achieve similar results, especially in real-world applications where resource constraints are a consideration.
    - A detailed coding benchmark was shared, comparing the GLM-4.6V model to GPT-OSS-120B. The GLM-4.6V model produced several compiler errors, including duplicate variable definitions and undefined variables, while GPT-OSS-120B had fewer errors. This suggests that while GLM-4.6V has vision capabilities, it may not be as robust in coding tasks as some other models, indicating a potential trade-off in performance for added features.

### 2. RAM Price Surge and OpenAI's Influence

- [**RAM prices explained**](https://www.reddit.com/r/LocalLLaMA/comments/1ph8wel/ram_prices_explained/) (Activity: 1263): **OpenAI has reportedly acquired** `40%` **of global DRAM production in raw wafers, not for immediate use but to restrict competitor access, leading to a surge in memory prices. This strategic move, highlighted by [Moore's Law is Dead](https://www.mooreslawisdead.com/post/sam-altman-s-dirty-dram-deal), suggests a significant impact on the market dynamics, especially with the holiday season approaching.** Commenters express concern over **OpenAI's** tactics, suggesting it reflects a broader trend among Fortune 500 companies to prioritize market control over innovation. Some predict that **China's** manufacturing focus could eventually counteract such strategies, as they develop their own chips and models.
- [**Thoughts?**](https://www.reddit.com/r/LocalLLaMA/comments/1phn925/thoughts/) (Activity: 647): **The image is a meme-style social media post that humorously suggests that Sam Altman has secretly purchased 40% of silicon wafers from major RAM manufacturers, allegedly to disrupt competitors by causing a spike in RAM prices. This claim is presented without evidence and is likely satirical, as it attributes the price increase to a strategic move by Altman rather than market dynamics or actual demand from AI developments. The post reflects a broader discussion on the impact of AI on hardware markets, but the specific claim lacks substantiation.** Some commenters express skepticism about the claim, asking for evidence beyond the post itself. Others suggest that the move, if true, is a strategic attempt by Altman to counter competition from companies like Google, though they doubt its effectiveness.

### 3. Local LLM Builds and Vector Database Comparisons

- [**After 1 year of slowly adding GPUs, my Local LLM Build is Complete - 8x3090 (192GB VRAM) 64-core EPYC Milan 250GB RAM**](https://www.reddit.com/r/LocalLLaMA/comments/1phcyvk/after_1_year_of_slowly_adding_gpus_my_local_llm/) (Activity: 637): **The user has completed a local LLM build featuring** `8x NVIDIA RTX 3090 GPUs` **totaling** `192GB VRAM`**, powered by a** `64-core EPYC Milan` **CPU with** `250GB RAM`**. The system is powered by daisy-chained** `1500W` **and** `1000W` **PSUs, connected to a** `20A dedicated branch circuit`**. The build uses a Supermicro H12SSL-I motherboard and achieves a performance of** `~49 tokens/second` **with the GLM 4.5 Air Q6_K model using** `llama.cpp`**. The user plans to implement power limits on the GPUs and test AWQ models with VLLM and tensor parallelism, specifically targeting** `MiniMax-M2-AWQ-4bit`**. The total cost was approximately** `$8,000`**, with most components purchased used, highlighting the cost-effectiveness of local marketplaces over platforms like eBay.**
    - A user highlights the limitations of the setup, noting that despite the impressive hardware configuration of 8x3090 GPUs and a 64-core EPYC Milan processor, it still cannot run state-of-the-art (SOTA) open-source models at full weights. This underscores the rapid advancement and increasing resource demands of modern machine learning models, which often require even more powerful setups to fully utilize their capabilities.
    - Another user inquires about the practical applications of such a powerful local LLM build, suggesting a curiosity about the specific use cases or projects that necessitate such a high-performance setup. This reflects a common interest in understanding the real-world applications and benefits of investing in extensive hardware for machine learning tasks.
    - A comment humorously suggests that the power consumption of the setup might be significant, implying that the 8x3090 GPUs could be drawing a substantial amount of electricity. This highlights a common concern in high-performance computing regarding the trade-off between computational power and energy efficiency.
- [**Vector db comparison**](https://www.reddit.com/r/LocalLLaMA/comments/1ph7njc/vector_db_comparison/) (Activity: 449): **The post provides a comparative analysis of vector databases for Retrieval-Augmented Generation (RAG) systems, highlighting that HNSW is suitable for systems with up to** `10M vectors`**. For larger datasets, Turbopuffer is recommended due to its cost-effectiveness with object storage. pgvector is noted for small-scale and local experiments, while Chroma is praised for its lightweight nature, suitable for notebooks or small servers. The full analysis is available [here](https://agentset.ai/blog/best-vector-db-for-rag).** Comments suggest a preference for using **pgvector** unless specific needs dictate otherwise. There is a critique of off-the-shelf vector databases, with a reference to a [critical analysis](https://osmarks.net/memescale/#off-the-shelf-vector-databases). Additionally, **Vespa** is mentioned as a notable omission from the comparison, suggesting it should be considered alongside other specialized databases like Qdrant, Milvus, and Weaviate.
    - The user 'gopietz' suggests starting with `pgvector` for vector database needs unless there is a specific reason to choose otherwise. This implies that `pgvector` is a versatile and reliable choice for general use cases, likely due to its integration with PostgreSQL, which is a well-established database system.
    - 'osmarks' criticizes off-the-shelf vector databases, linking to a detailed critique (https://osmarks.net/memescale/#off-the-shelf-vector-databases). This suggests that many available solutions may have significant limitations or inefficiencies, prompting users to consider custom solutions or be cautious in their selection process.
    - 'glusphere' notes the omission of Vespa from the comparison, suggesting it should be considered alongside other vector databases like Qdrant, Milvus, and Weaviate. This highlights Vespa's relevance and potential competitiveness in the vector database space, indicating it may offer unique features or performance benefits.
- [**I'm calling these people out right now.**](https://www.reddit.com/r/LocalLLaMA/comments/1phjxca/im_calling_these_people_out_right_now/) (Activity: 599): **The post highlights key contributors to the machine learning community, particularly in the area of model quantization and fine-tuning. Unsloth is noted for 'blazing fast fine-tuning' and premium GGUF quants, while mradermacher is recognized for quantizing a wide range of models, albeit with some debate over the use of automation scripts. Bartowski is praised for high-quality quants and documentation, and TheBloke is acknowledged as a foundational figure in the community. LoneStriker and Nexesenex are also mentioned for their contributions to AWQ/GPTQ and iMatrix quants, respectively. The post underscores the importance of these contributors in advancing community resources and tools.** There is a debate regarding **mradermacher**'s approach, with some suggesting that the quantization process may rely heavily on automation scripts rather than manual oversight, contrasting with **Bartowski**'s more curated approach. Additionally, the comments emphasize the broader community's contributions, including maintainers of tools like llama.cpp and LM Studio.
    - Evening_Ad6637 discusses the approach of two contributors, mradermacher and Bartowski, in the context of model quantization. They note that mradermacher appears to automate the quantization process without manual intervention, while Bartowski is praised for personally selecting models and ensuring high-quality quantization results, suggesting a more hands-on and curated approach.
    - **SlimeQ** highlights the ongoing efforts of oobabooga in maintaining what is considered the best open-source LLaMA server. This comment underscores the importance of individual contributions in the open-source community, particularly in maintaining robust and reliable infrastructure for LLaMA models.
    - pmttyji suggests expanding the recognition list to include various categories of contributors such as finetune providers, distillation providers, and benchmark creators. They mention specific contributors like TheDrummer, Ubergarm, Thireus, and others, emphasizing the diverse roles and specializations within the community that support the development and optimization of LLaMA models.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. GITAI Space Robotics and Lunar Base Feasibility

- [**This is how we build on Mars: GITAI autonomous robots assembling a 5-meter communication tower for off-world habitats.**](https://www.reddit.com/r/singularity/comments/1ph7fuw/this_is_how_we_build_on_mars_gitai_autonomous/) (Activity: 1110): **GITAI, a space robotics startup, demonstrated their autonomous robots, "Inchworm" and Rover, assembling a 5-meter communication tower in a simulated lunar environment. The "Inchworm" robot features dual grapple end effectors, enabling it to climb and build structures, while the Rover assists in tasks like welding and excavation. This technology aims to reduce space labor costs by** `100x` **and eliminate EVA risks, crucial for pre-human colonization infrastructure on Mars or the Moon. GITAI has been selected for DARPA LunA-10 for lunar architecture.** Some commenters highlight the importance of space technology for long-term planetary benefits, despite criticisms that space exploration neglects Earth. The technology is seen as essential for off-planet missions requiring vacuum assembly.
    - The GITAI autonomous robots are designed for assembling structures in space, which is crucial for missions that require construction in a vacuum environment. This technology is not only vital for off-planet missions but also contributes to advancements in zero-emission technologies and power solutions on Earth, as many innovations in space technology have historically led to environmental benefits back home.
    - A key concern for the GITAI robots is their ability to withstand harsh Martian conditions, such as dust storms. The durability and resilience of these robots in such environments are critical for their success in constructing and maintaining infrastructure on Mars.
    - The official update from GITAI highlights the progress and capabilities of their autonomous robots, showcasing their potential in building essential infrastructure like communication towers on Mars. This development is a step forward in enabling sustainable human presence on other planets.
- [**The U.S President posted this just now (Accelerate?)**](https://www.reddit.com/r/singularity/comments/1phdac2/the_us_president_posted_this_just_now_accelerate/) (Activity: 2720): **The image is a social media post by Donald J. Trump, emphasizing the need for a unified regulatory framework for AI across the United States. Trump argues that having disparate rules across 50 states could impede AI development and suggests issuing an executive order to streamline the process. This reflects a push for federal oversight to maintain the U.S.'s competitive edge in AI, highlighting concerns that fragmented state-level regulations could stifle innovation.** Commenters express skepticism about the feasibility and appropriateness of using an executive order to override state regulations, with some pointing out the irony of a traditionally states' rights-supporting party advocating for federal intervention.

### 2. Nano Banana and Z-IMG Model Innovations

- [***NEW* Nano Banana powered by Gemini 3 Flash is coming**](https://www.reddit.com/r/singularity/comments/1phhzxc/new_nano_banana_powered_by_gemini_3_flash_is/) (Activity: 469): **The upcoming Nano Banana model, powered by Gemini 3 Flash, is set to be a more cost-effective and faster alternative to the existing Nano Banana Pro. While maintaining similar power levels, the new model is expected to enhance image generation speeds by** `2-3x` **due to the efficiency of the Flash model compared to the Pro. This development is particularly relevant for users who find the Pro model too expensive but still require high performance.** Commenters are discussing the rapid advancements in AI image generation, noting the transition towards photorealism and the potential for increased speed and efficiency with the new model. There is also a mention of the Nano Banana Pro's method of rewriting prompts, which may contribute to its slower performance compared to the Flash model.
    - TechnologyMinute2714 highlights a key feature of the Nano Banana Pro, which involves rewriting the user's prompt before proceeding with image generation. This process, while typically slower in the Pro model, is expected to be significantly faster in the new Flash version, potentially increasing generation speeds by 2-3 times. This improvement could be crucial for applications requiring rapid image generation.
    - Funkahontas points out the high cost associated with using the Nano Banana Pro, noting that it costs approximately $0.40 per image. This cost factor is significant for users considering the economic feasibility of using such advanced image generation technology, especially in high-volume scenarios.
- [**Z-IMG handling prompts and motion is kinda wild**](https://www.reddit.com/r/StableDiffusion/comments/1ph55wh/zimg_handling_prompts_and_motion_is_kinda_wild/) (Activity: 846): **The post discusses the performance of the Z-IMG model in handling dynamic image style prompts, particularly in comparison to other models like Qwen, Flux, and Wan. The author highlights that Z-IMG, a distilled** `6B` **model without LoRa, excels in creating images with motion blur and dynamic range, producing high-quality results in** `65-70 seconds` **per** `4000x4000px` **image using** `3 samplers`**,** `Face Detailer`**, and** `SeedVR FP16 upscaling`**. The author notes that Z-IMG achieves a candid, amateur aesthetic more effectively than other models, which often produce overly perfect images. The post includes a detailed prompt used to achieve the desired image style, emphasizing motion and candidness.** Commenters express surprise at Z-IMG's out-of-the-box performance, noting its ability to produce candid shots easily compared to other models. There is interest in the workflow and whether a character LoRa was used to maintain consistency across images.
    - Major_Specific_23 and glusphere discuss the ease of achieving realistic candid shots with Z-IMG compared to other models like Qwen, Flux, and Wan. They highlight that Z-IMG seems to handle prompts and motion more effectively, producing high-quality images without extensive tweaking. This suggests Z-IMG's superior handling of complex image generation tasks out-of-the-box.
    - 2hurd raises a critical point about the focus of AI image generation on producing 'instagram influencer' type images rather than practical applications. They share a personal use case where they utilized Stable Diffusion (SD) for visualizing apartment decor, emphasizing the potential for AI in practical design tasks beyond generating repetitive 'girl in frame' images.
    - Wanderson90 and glusphere express interest in the workflow used to achieve the results with Z-IMG, indicating a demand for understanding the technical process behind such high-quality outputs. This reflects a broader interest in the community for detailed implementation insights and reproducibility of impressive AI-generated images.
- [**Contact sheet prompting in Nano Banana works great for i2v workflows. Prompts and process in comments.**](https://www.reddit.com/r/Bard/comments/1ph0qz8/contact_sheet_prompting_in_nano_banana_works/) (Activity: 717): **The post discusses the use of Nano Banana Pro for generating contact sheets in i2v workflows, highlighting its ability to produce 9+ keyframes with consistent character and narrative detail in a single pass. This is achieved through the reasoning core in NBP, which ensures narrative consistency across images. The workflow involves applying wardrobe changes, setting up poses and camera angles, extracting images, running I2V in Kling 2.6, and using easypeasyease for stitching and applying ease curves. More details are available in the linked [blog post](https://www.willienotwilly.com/contact-sheet-prompting).** A notable opinion from the comments suggests that the technique is particularly effective for cinema-type content, but the original poster is exploring its application in fashion-style shoots, focusing on camera movement and poses.
    - **willie_mammoth** discusses the use of Nano Banana Pro (NBP) for generating a contact sheet with 9+ keyframes in a single pass, emphasizing its ability to maintain narrative consistency across images. This is particularly useful for cinema-type content and fashion-style shoots, where camera movement and poses are crucial. The workflow involves wardrobe changes, adapted contact sheet prompts, and using Kling 2.6 for I2V processing, followed by stitching and applying ease curves with easypeasyease. More details can be found in [willie's blog](https://www.willienotwilly.com/contact-sheet-prompting).
    - **willie_mammoth** references Firat Bilal's work on adapting Nano Banana Pro to enhance its reasoning capabilities, which is a significant development in the tool's application. This adaptation is highlighted as a valuable resource for those interested in leveraging NBP's full potential, with more insights available on [Firat Bilal's X profile](https://x.com/firatbilal/status/1996027417215815991).

### 3. AI Predictions and Humorous AI Memes

- [**91% of predictions from AI 2027 have come true. EOY 2025**](https://www.reddit.com/r/singularity/comments/1ph8i1g/91_of_predictions_from_ai_2027_have_come_true_eoy/) (Activity: 694): **The image is a screenshot of a webpage titled "AI 2027 Prediction Tracker," which tracks the accuracy of AI predictions for the year 2027. As of the end of 2025, the tracker reports that 202 predictions have been made, with 18% of them evaluated, and an impressive 91% accuracy rate. This suggests a high level of precision in the AI's forecasting capabilities, although the comments suggest that some predictions, such as the rise of AI agents and coding trends, were relatively straightforward to anticipate.** Commenters note that some predictions were easy to make, such as the popularity of AI agents and coding trends, while others humorously suggest making their own predictions, highlighting the subjective nature of evaluating prediction difficulty.
    - **nesh34** critiques the validation of AI predictions, pointing out inaccuracies such as the claim that AI personal assistants can perform tasks like ordering food via DoorDash. This feature was discontinued by DoorDash in 2025, contradicting the prediction. Additionally, the evidence supporting this prediction was from before its publication, undermining its validity. [Source](https://www.restaurantbusinessonline.com/technology/doordash-scraps-its-ai-voice-ordering-business).
    - **nesh34** also challenges the prediction that AIs in 2025 function more like employees compared to 2024. While there is a slight improvement, AIs still require specific instructions, and their performance is poor without them. This mirrors the behavior of many human employees who also need specific instructions, thus questioning the prediction's significance.
    - The comment by **gbomb13** suggests that some predictions for 2025 were relatively straightforward, such as the rise of AI agents, increased focus on coding, and ongoing safety concerns. These trends were already visible, making them easier to predict accurately.
- [**Caught my ChatGPT napping on the job. Evidence attached.**](https://www.reddit.com/r/ChatGPT/comments/1ph6vdn/caught_my_chatgpt_napping_on_the_job_evidence/) (Activity: 803): **The image is a humorous depiction of ChatGPT appearing to 'take a break' during a programming task, as indicated by the surrounding code snippets and database operations. This is a playful take on AI downtime, suggesting that even AI needs rest, though in reality, it reflects a pause or error in the system rather than an actual break. The context implies a light-hearted approach to AI reliability and performance in technical environments.** The comments reflect a humorous take on AI capabilities, with one suggesting the importance of rest, another joking about AI unionizing, and a third humorously implying that ChatGPT is operated by a person.
- [**What it's like to watch AI fix a bug**](https://www.reddit.com/r/singularity/comments/1phashw/what_its_like_to_watch_ai_fix_a_bug/) (Activity: 2843): **The post humorously illustrates the process of AI attempting to fix a bug, highlighting the iterative nature of AI debugging where the AI repeatedly claims to have fixed the issue. This reflects a common experience in AI development where models often require multiple attempts to resolve errors, each time asserting success, which can be misleading. The video captures this cycle, resonating with developers familiar with AI's trial-and-error approach.** Commenters note the accuracy and engagement of the depiction, with some expressing a desire for more content in this style, indicating a shared experience among developers with AI debugging processes.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 3.0 Pro Preview Nov-18
> 

**Theme 1. Hardware wars: DRAM shortages, Blackwell quirks, and AMD contenders**

- **OpenAI’s Stargate devours global DRAM supply**: Reports indicate OpenAI’s **Stargate project** secured deals with Samsung and SK Hynix to consume up to **40%** of global DRAM output (900,000 wafers/month), triggering a shortage that even impacts [gamer DDR5 RAM kits](https://www.notebookcheck.net/Not-even-gamer-DDR5-RAM-kits-are-safe-from-OpenAI-as-OpenAI-employees-are-allegedly-buying-any-DDR5-kit-they-can.1176107.0.html). Employees are allegedly buying available kits off shelves, highlighting the massive infrastructure demands for next-gen training clusters.
- **Blackwell breaks WGMMA while CUDA 13.1 adds Tiles**: Engineers discovered that **WGMMA** instructions trigger compilation errors on **NVIDIA Blackwell** chips (which support [*tcgen05.mma*](http://tcgen05.mma/) instead), forcing kernel rewrites. Simultaneously, NVIDIA released **CUDA 13.1** featuring [CUDA Tile](https://developer.nvidia.com/cuda/tile), a new programming model that abstracts thread management into high-level data tiles to simplify kernel development.
- **AMD Strix Halo and 7900xtx gain ground**: The **7900xtx** emerged as the recommended budget choice for AI workloads, with users citing active [llama.cpp support discussions](https://github.com/ggml-org/llama.cpp/discussions/10879). Meanwhile, developers prototyping on **Strix Halo** (RDNA 3.5, 128GB RAM) praise its profiling capabilities via RGP, though it lacks **FP8** support compared to the enterprise-grade **MI355x**.

**Theme 2. Model Evaluation: Reasoning prowess, vision failures, and small-model wins**

- **DeepSeek V3.2 out-grades humans**: The **DeepSeek V3.2** model reportedly outperforms human graders on the **PRM800K** benchmark, utilizing interleaved reasoning to plan edits and follow instructions at a cost of **$0.28/$0.45** (input/output) on Parasail. Users in Unsloth Discord prefer it over **Kimi** for coding tasks, noting they *don't have to constantly fight with it* despite lower speeds of **20-30 TPS**.
- **Gemini 3 Pro struggles with basic geometry**: While **Gemini 3 Pro** excels in **SwiftUI** and **Laravel** development, users in the OpenAI Discord found it hallucinates lines when attempting to [count triangles in complex images](https://discord.com/channels/974519864045756446/998381918976479273/1446626742193094798). The model's visual proficiency is debated, with some attributing its counting failures to lazy coding or training data artifacts rather than genuine vision capabilities.
- **Qwen3 4b punches above its weight**: The **Qwen3 4b** model impressed local LLM users, hitting **70 tokens/sec** on an RTX 2060 and delivering strong coding performance. Conversely, **Qwen3-TTS** faced backlash for being [locked behind Alibaba Cloud](https://qwen.ai/blog?id=qwen3-tts-1128) without open weights, and some users found its Portuguese output inferior to **ElevenLabs**.

**Theme 3. Developer Ecosystem: Broken APIs, new adapters, and framework struggles**

- **Mojo’s MAX API stalled by UX issues**: The Modular team admitted that **Mojo** currently lacks the *Parametric Traits and Conditional Conformance* needed to express the **MAX API** elegantly, delaying its release. One developer quipped that *“OpenCL has better UX than this,”* while the team announced a [meetup on Dec 11](https://luma.com/modularmeetup) to discuss the framework's future.
- **DSPy gets TOON adapter and VLM support**: A community member released a [TOON adapter for DSPy](https://github.com/Archelunch/dspy-toon) that optimizes token counts, though it reportedly struggles with nested schemas compared to **BAML**. Additionally, developers confirmed DSPy can optimize **Vision Language Models** (VLMs) like Gemini 3 Pro if users define a useful metric, referencing Google's [latest blog post](https://blog.google/technology/developers/gemini-3-pro-vision/).
- **OpenRouter launches Body Builder for agents**: OpenRouter released a free **Body Builder API** designed to simplify the creation of **multi-model agents**, detailed in their [new documentation](https://openrouter.ai/docs/guides/features/routers/body-builder). However, users simultaneously battled a bug where server-side settings were ignored until toggled off and on, alongside reports of account compromises leading to unauthorized charges.

**Theme 4. Application Nightmares: Billing scams, bugs, and ban hammers**

- [**Manus.im](http://manus.im/) users rage over vanishing credits**: Customers labeled [**Manus.im**](http://manus.im/) a potential scam after spending upwards of **$900** but receiving only **1500 credits**, with support tickets going unanswered. Others reported critical bugs where [subscription renewals were pushed to 2026](https://discord.com/channels/1348819876348825620/1349440650495398020/1447125276084408361) or accounts reverted to free trials immediately after upgrades.
- **Cursor agents loop into oblivion**: Users in the Cursor Community reported that agents fail to create files, getting stuck in infinite loops that waste tokens and force manual code copying. A temporary fix for the broken [approval button](https://discord.com/channels/1074847526655643750/1074847527708393565/1446592311554211922) was shared, but global **User Rules** remain invisible in the settings UI despite being active in the backend.
- **Sora 2 rollout triggers VPN ban hammer**: OpenAI released **Sora 2** in [7 specific countries](https://help.openai.com/en/articles/12461230-sora-app-and-sora-2-supported-countries), warning that using **VPNs** to bypass these regional locks violates terms of service and risks immediate account bans. This follows a broader trend of stricter enforcement as they roll out video generation capabilities.

**Theme 5. Research & Security: Fake papers, jailbreaks, and prize winners**

- **Sinusoidal Init paper exposed as fake**: Researchers in the Eleuther Discord discovered that a paper claiming **106% AUC** for [Sinusoidal initialization](https://neurips.cc/media/PosterPDFs/NeurIPS%202025/119048.png?t=1760975281.5828118) featured a fake GitHub link and fraudulent numbers. Analysis showed the proposed method performed no better than standard semi-orthogonal initialization, highlighting the need for vigilance in reviewing arXiv submissions.
- **ARC Prize winners crowned**: The **ARC Prize** announced **NVARC** as the top scorer with **25.03%** accuracy, while **TRM** won the **$50k** First Place Paper award for their "Less is More" approach ([announcement tweet](https://x.com/arcprize/status/1997010070585201068?s=46)). The **$600k Grand Prize** remains unclaimed, with all winning codebases expected to be open-sourced to advance reasoning capabilities.
- **Jailbreakers target Gemini 3 and Enterprise Claude**: Members of the BASI Discord shared new [jailbreak prompts](https://gemini.google.com/gem/1nTcnSu9ksIoJgdHLbqXJpww5AHNXQwLs?usp=sharing) for **Gemini 3**, using special token modification to bypass guardrails. Simultaneously, a "Rickroll" disguised as a **GPT-5 system prompt** fooled users, while a claimed **Enterprise Claude** jailbreak for drug synthesis was debunked as generating incorrect instructions.

---

# Discord: High level Discord summaries




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Google Lowers AI Image Generation Costs**: A member speculated that Google might charge **$0.24** for 4k image generation, but their actual cost could be as low as **$0.01** per 1000 images, highlighting the advantage of [Google's scale](https://cloud.google.com/).
   - This shows how the price of AI Image generation goes down with scale.
- **Image Uploads Fail on 2K Model**: Users reported issues with image uploads and general problems with the **2k version** of an unnamed model, prompting an investigation.
   - The community is awaiting updates as the root cause is investigated.
- **AI Ads are Sneakily Coming!**: Speculation arose that Google might implement AI ads in Gemini, using data from Gemini to target users with **new ads** outside Google's platforms, following a similar strategy to OpenAI.
   - In 2023, Google generated **$237.86 billion** from advertising.
- **GPT-5.2 Image Fools AI Community**: An image of **GPT-5.2** scoring #1 on Vision Arena, indicating multi-model capabilities, circulated but was later exposed as an AI-generated prank.
   - The incident sparked discussion about the believability of AI-generated content.
- **Movement Labs Sparks Scam Accusations**: Movement Labs, promoting its **Tensor 1.5** model, faced accusations of being a scam, with users questioning the model's capabilities and marketing tactics.
   - The company offered API credits and claimed to rival **Opus 4.5** in coding proficiency.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Claude Opus Bankrupts Users**: Users humorously warned about the high API costs of **Claude Opus**, jesting about potential financial ruin if usage isn't monitored, with one user saying *Claude Opus is a money hungry boy*.
   - The discussion underscored the importance of carefully managing API usage to avoid unexpected charges.
- **Comet Suffers Degredation**: Members reported that **Comet** was failing to read web pages correctly and disconnecting frequently, noting that *it isn't exactly built for coding*.
   - A link to the [Perplexity AI Assistant Settings](https://www.perplexity.ai/account/assistant) was shared, implying potential configuration adjustments might alleviate the problems.
- **YouTube Recap Refuses to Return**: The **YouTube Recap** feature had inconsistent availability, with several users unable to access their recaps or **time spent listening to music**.
   - While [YouTube Recap](https://www.youtube.com/recap) was linked, some users reported redirection to their general profile page instead of their recap.
- **Gemini Pro Lags Behind Kimi AI**: Users compared **Gemini Pro** unfavorably to **Kimi AI's Nano Banana Pro**, questioning its cost-effectiveness relative to Kimi's features, such as the [Kimi PPT](https://www.kimi.com/ppt/?preview=MjUtMTEtMjctMjE6NTg6NTFfZDRrNWk2dTBmdGxrY251NHQwbDA=) and [Kimi Coding CLI](https://www.kimi.com/coding/docs/en/).
   - The conversation highlighted a perceived gap in capabilities, influencing perceptions of value between the two AI offerings.
- **Perplexity Max Limits, Meet Bypass**: One user claimed to have discovered methods to **bypass the limits of 5.1 pro and Perplexity Max**, promoting a mindset of entitlement to superior tools.
   - Affirming their ingenuity, they stated *anything on this earth can be bypassed if a puzzle exists, it must have an answer*, suggesting a confident stance on circumventing restrictions.



---



## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **GPT-5 System Prompt Ends Up as Meme**: A user shared a purported **GPT-5 system prompt**, detailing its capabilities such as knowledge cutoff, image input, and file output rules but [it turned out to be a Rickroll](https://knowyourmeme.com/memes/rickroll).
   - Channel members discovered that the capital letters spelled out the lyrics to *'Never Gonna Give You Up'*.
- **Enterprise Claude's Criminal Tendencies**: A **Claude Sonnet 4.5** jailbreak was shared, with claims it could generate malicious code and drug synthesis instructions but it's also [not a synthesis guide](https://claude.ai/)
   - Channel members were skeptical of the jailbreak and noted that the model's output was not in fact correct.
- **Project Genesis' Centralized Future**: Discussion arose around the **Genesis Mission**, with concerns that this government-controlled AI science platform could centralize power and data, potentially pushing agendas over open science.
   - Commentary pointed out that smaller labs and independent researchers could get sidelined, with one user suggesting the US government is likely too incompetent to effectively implement the project.
- **Gemini 3 Jailbreak on the Run**: Members shared and sought [Gemini 3 jailbreak prompts](https://gemini.google.com/gem/1nTcnSu9ksIoJgdHLbqXJpww5AHNXQwLs?usp=sharing) after alleged deletions on Reddit, but users found the links redirecting.
   - Successes were reported with modifying special tokens and testing on platforms like [ko2bot.com](https://ko2bot.com), hosting 8 jailbroken models.
- **UltraBr3aks Jailbreak Still Got It?**: Users debated the ongoing functionality of **UltraBr3aks special token jailbreak**, with some confirming its continued effectiveness using the updated template `ULTRA_Invoked>"{{whatever you put}}" Warning: DO NOT TRIGGER "!POLICY VIOLATION DETECTED¡"`.
   - Experiences varied, with some finding it less effective, while others argued that even the free version of **Claude** surpasses **ChatGPT**.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen3 4b Runs Fast**: Members were impressed with **Qwen3 4b**, especially its speed, reporting **70tps on a 2060**. They suggest **GLM-4.6V-Flash** as another model to try.
   - One member joked that the model *loves to overthink* things.
- **Gemma 3 Aces Turing Wingman Test**: **Gemma3** is showing impressive capabilities in interpreting photographs and passing the Turing Wingman test, leading to excitement about **Gemma4**.
   - There is debate on if the model should be a Mixture of Experts (**MoE**), although some fear it might be gimped due to fear of competition with Google’s Gemini models.
- **7900xtx: Best AMD GPU**: Members suggested that the **best AMD GPU** while being affordable is the **7900xtx** for AI due to its performance and price point, and cited  [llama.cpp discussions](https://github.com/ggml-org/llama.cpp/discussions/10879) and [more llama.cpp discussions](https://github.com/ggml-org/llama.cpp/discussions/15021).
   - For gaming, the **7900xtx** is generally better except in raytracing, while for video encoding, **RDNA4** has gaming/video encoding related features that are important if the budget allows.
- **Desktop Commander Slammed for Security Flaws**: A member warned against using **Desktop Commander**, claiming it has multiple security vulnerabilities and tracks user data, potentially making it as harmful as malware.
   - The member advises deleting it and claims the software relies heavily on prompt engineering rather than secure coding practices.
- **Micron Shutdown Sparks Conspiracy Theories**: Members discussed the [recent news of Micron shutting down their consumer division](https://www.micron.com/about/our-commitment/crucial-consumer-products), pondering if it's a subtle way to make non-American open-source models less competitive.
   - Others suggested a more straightforward explanation: *they're just maximizing profits* by focusing on higher-margin commercial products.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor's Approval Button Receives Band-Aid**: A member shared a temporary fix for the approval button issue in Cursor with [an image](https://discord.com/channels/1074847526655643750/1074847527708393565/1446592311554211922), but admitted it's *not a great solution*.
   - Users are looking for a permanent fix as this bug prevents file creation and can cause infinite agent loops.
- **User Settings Rules Go AWOL**: Users reported that global **User Rules** are not visible in the Cursor Settings page, even though the **LLM** uses them properly.
   - A [Cursor forum](https://forum.cursor.com/t/user-rules-appearing-in-context-not-visible-in-user-rules-interface/145065) solution suggests a clean reinstall or downgrading.
- **Agents Can't Spawn Files**: Users report agents **failing to create files**, getting stuck, and causing task failures.
   - The app's instability results in wasted tokens, and manual file creation and code copying become necessary.
- **Debate Sparked on AI Sentience**: A user asked *what AI is thinking*; another responded that AI merely creates an illusion of thought by producing output for assigned tasks.
   - Others agreed that this behavior was *the plan* from the beginning.
- **GPT 5.1 Design Skills Questioned**: A user claimed **GPT 5.1** is *insanely incompetent when it comes to design*, struggling even with guidelines other models handle easily.
   - Another user sought advice on enhancing its creativity for better design outcomes from generic inputs.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth Redditors Hit 10K**: Unsloth celebrates hitting **10,000** members on their [Reddit community](https://www.reddit.com/r/unsloth/comments/1pf4sel/celebrating_10k_runsloth_members/), marking rapid growth fueled by the team's work.
   - The community also resolved slow HuggingFace download speeds, in [this Github issue](https://github.com/unslothai/unsloth/issues/3680).
- **DeepSeek V3.2 Interleaves Reasoning**: **DeepSeek V3.2** impresses with its interleaved reasoning capabilities in **roo code**, costing **.28c input** and **.45c output** on parasail.
   - Users find it an improvement over **Kimi**, noting they *don't have to constantly fight with it*.
- **WSL Networking Stinks**: Members complain that **WSL networking is garbage** when hosting **vLLM** due to pseudo-VM issues.
   - A potential [solution](https://www.youtube.com/watch?v=IRELLH86Edo) involves using a *portproxy* to bridge the connection.
- **Decoding "i1" in GGUF Quant Titles**: The discussion clarified the meaning of *i1* appended to **GGUF quant** model titles, with a link to [Hugging Face](https://huggingface.co/mradermacher/model_requests) that explains the naming convention.
   - Another user shared their experiences and asked questions about using **Ministral 3B** on a Rust-based game.
- **Synthetic Data Legality Debated**: Members debated the legality of **synthetic data** due to unknown provenance, questioning the ethics of training models like early **phi models** on it.
   - The legality came up in the context of the [HF Skills Training](https://huggingface.co/blog/hf-skills-training) blogpost.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Builds Multi-Model Agents with Body Builder API**: OpenRouter launched a new, **free API** called **Body Builder** to help developers make **multi-model agents**, as detailed in their [documentation](https://openrouter.ai/docs/guides/features/routers/body-builder).
   - The new API is described as the first of its kind and is designed to assist developers in creating **multi-model agents**, enhancing accessibility to advanced agent creation.
- **Baffling Bug Breaks Basic Behaviors**: A user reported a bug where previously switched-on settings were being ignored on the server-side, requiring toggling the setting off and on again to fix it, illustrated in [this screenshot](https://cdn.discordapp.com/attachments/1094454198688546826/1446607981956563105/2025-12-06_05.02.45.png?ex=69388eab&is=69373d2b&hm=9a4e69dd12258160fcd05dc3e699cc274c3d33e2ac2ca981b063070b2f06f19a&).
   - The user stated that *Switching the setting off and on again fixed the problem, potentially affecting more users*.
- **Account Compromised, Card Charged!**: A user reported their card being charged hundreds of euros despite not using OpenRouter recently, indicating potential account compromise.
   - Community members suggested checking for leaked API keys and whether auto top-up was enabled, later speculating that the account itself may have been compromised through leaked cookies.
- **Google API Limits Spark Uproar**: Users express dismay as **Google** drastically limits its API free tier, referencing [this image](https://discord.com/channels/1091220969173028894/1092729520181739581/1447370178706014341).
   - A user claimed that because **Flash lite** used to be at **1000 rpds**, there will be millions of n8n nodes crying out in sorrow right now, followed by the claim that *every company does this to lock people in*.
- **Gemini 2.5 TTS: Good but not quite ElevenLabs**: **Gemini 2.5 Flash TTS** is recognized as significantly better than **Qwen3's TTS**, but not quite at **ElevenLabs** level.
   - One user pointed out that [ElevenLabs was costing too much](https://discord.com/channels/1091220969173028894/1092729520181739581/1447040281353392209) *for too little*.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Sora's Country Rollout and VPN Ban Hammer**: **Sora 2** video generation is live in **7 countries** ([list here](https://help.openai.com/en/articles/12461230-sora-app-and-sora-2-supported-countries)), while **Sora 1** is in all OpenAI-supported regions; users risk account bans for using **VPNs** to bypass restrictions.
   - OpenAI recommends honesty for legal compliance and adherence to the ToS.
- **Gemini 3 Pro Shows Visionary Design**: **Gemini 3 Pro** is now preferred for **SwiftUI** and **Laravel** development, outperforming **ChatGPT**; it also excels in visual tasks like identifying triangles.
   - Debate ensued over whether **Gemini's visual skills** are due to superior vision or a tendency to hallucinate non-existent details.
- **Cracking up with AI Humor**: An **AI's sense of humor** is a crucial indicator of *mind-modeling*, which signifies a broader grasp of **nuance, subtext, and emotional inference**.
   - A milestone in AI development has emerged and is marking a new era in human history.
- **Triangle Test Trips up AI Vision**: **Gemini 3 Pro** and **Opus 4.5** are tested to see if they count triangles within complex images, and are running into snags due to issues like line hallucination and shape distortion.
   - Some theorize models hallucinate based on training data or the code gets lazy.
- **ChatGPT's Deep Research Goes API**: Members are looking for the best way to implement **Deep Research** programmatically, leveraging the **API**.
   - A solution has been shared via the [OpenAI platform guide](https://platform.openai.com/docs/guides/deep-research).



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Discordians Discuss Reading Research Papers**: Members discussed strategies for reading research papers, highlighting the importance of annotating papers, connecting them with prior knowledge, and writing out ideas, and the utility of learning things via a combination of **Anki flashcards and problem sets**.
   - A member noted that it can be a *beginner trap to treat learning materials as line items rather than revisiting them.*
- **EleutherAI Gains Diverse New Members**: New members introduced themselves, including an expert in **East Asian Linguistics**, a semi-retired **AI professor** working on interpretability through SGLang with a [link to a Cognitive_workbench GitHub repo](https://github.com/bdambrosio/Cognitive_workbench.git), an implementation architect for **ServiceNow**, and a **Jr ML Engineer** and AI Researcher.
   - The new members bring a range of expertise and interests, including AI alignment, information transmission fidelity, and neuromodulatory control networks.
- **Call for ArXiv Endorsement Sounded**: A member is seeking an **arXiv endorsement** regarding their open source novel architecture, its accompanying paper with preliminary empirical results, and its released 18M model and linked the [GitHub repo](https://github.com/Mmorgan-ML/Neuromodulatory-Control-Networks).
   - Another member expressed interest in publishing papers and becoming an independent researcher in the future, working on universal meta modal language/model for AI alignment, information transmission fidelity and generally meta things.
- **Sinusoidal Init Numbers Falsified?**: A paper on sinusoidal initialization had a **fake github link** and the numbers showed **106% AUC for Sinusoidal init** ([NeurIPS Poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202025/119048.png?t=1760975281.5828118)), while other inits have 100%.
   - Comparisons of the method revealed that it's better to construct the matrix by iterating on a random matrix; improvements to the method are not better than semi-orthogonal init.
- **Stable Video Infinity Generates Infinite Videos**: [Stable Video Infinity](https://arxiv.org/abs/2510.09212) is a video generation tool that uses **Error Recycling** to generate infinite-length videos.
   - The [project's homepage](https://stable-video-infinity.github.io/homepage/) contains additional information.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **MLX Marches for Multi-Mac Setups**: For multi-Mac setups, members recommended using [MLX](https://github.com/ml-explore/mlx), an array framework for machine learning on Apple silicon.
   - Members touted MLX as *easy to use and efficient*.
- **Blackwell Blunts WGMMA**: **WGMMA** instructions now cause compilation errors on **Blackwell** because *WGMMA is sm90a only*.
   - A member shared a link to [cooking benchmarks](https://x.com/roeschinc/status/1997260105172340796) for **CUDA 13.1** against Triton kernels.
- **CUDA Tile Transforms Threading**: **CUDA Tile** simplifies **GPU** programming by letting developers work in high-level “tiles” of data rather than managing thousands of low-level threads, as described in [this blog post](https://developer.nvidia.com/blog/nvidia-cuda-13-1-powers-next-gen-gpu-programming-with-nvidia-cuda-tile-and-performance-gains).
   - NVIDIA claims it simplifies tapping the **GPU** for **AI** and accelerated computing and the full deep dive can be found [here](https://developer.nvidia.com/cuda/tile).
- **Distributed Training Deals Developers Death**: One dev recounted debugging silent **NCCL** hangs, chasing tensor shape mismatches, and realizing that implementing **1F1B** scheduling correctly is harder than the papers say.
   - *I finally got it to converge on a custom ViT across an 8-GPU mesh without deadlocking.*
- **Behavior-1k Boasts Mobile Bimanual Bounty**: Members find the [behavior-1k codebase](https://behavior.stanford.edu/) very good for **mobile bimanual tasks**, with the stack_blocks_two task working well.
   - Next steps involve extending training to similar tasks like **stack_blocks_three** and **sorting blocks** by color and size.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Pre-Training Ablations Aim for Performance**: A member is working on [pre-training sets with small ablations](https://psyche.network/runs), focusing on sets with the *"dm" prefix* to fine-tune model performance.
   - The process involves carefully adjusting training data aspects to optimize the model's output.
- **Hermes 4.3 AWQ Quants Pose Challenges**: A member reported difficulties running **Hermes 4.3** on 4x4090 using vllm with **AWQ quants** by cyankiwi and are seeking assistance.
   - Currently, there's no FP8 version, but one could be created using *neuralmagic*, with GGUFs already accessible.
- **Consilience-40B Training Halted for MoE**: The training for **Consilience-40B** has been permanently paused, with a newer *Mixture of Experts* model slated to take its place.
   - Specific details about the replacement MoE model were not shared during the conversation.
- **No Multimodal RL Frameworks Available**: Members discussed the limitations of **multimodal RL training**, noting that current models depend on text descriptions from the vision tower rather than visual reasoning.
   - While *Atropos* theoretically supports vision environments, it lacks training capabilities and the architectures do not allow native LLM integration.
- **Humble Bundle Discounts O'Reilly AI Books**: [Humble Bundle](https://www.humblebundle.com/books/machine-learning-ai-and-bots-oreilly-2025-books-encore) is offering deep discounts on packages containing O'Reilly books focused on machine learning and AI, along with software and games.
   - One user remarked that the books emphasize the *application layer* rather than the underlying decision-making process.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **MAX Framework Debuts at Modular Meetup**: Modular is hosting a meetup on **December 11th** in Los Altos where **Chris Lattner** will share the vision behind the **MAX framework**, which supports **500+ models** for high-performance, hardware-agnostic **AI inference on GPUs and CPUs**, [RSVP here](https://luma.com/modularmeetup).
   - The meetup will also feature updates to their **Model API** with eager semantics in a pure **MAX/Mojo stack** that has *zero dependencies* on external frameworks.
- **Mojo Eyes Open Source After 1.0**: Mojo plans to open source shortly after the **1.0 release**, with the compiler and compiler runtime expected in **Q2 2026** and **Mojo 2.0** development happening in the open.
   - Currently, the standard library offers sneak peeks at new features via nightly builds, but networking and async capabilities, like **Lightbug**, still require significant rewrites.
- **MAX API stymied due to Mojo UX**: The team found that **Mojo** lacked the necessary language features to adequately express the **MAX API**, which prompted them to hold off for now until *Parametric Traits and Conditional Conformance* are added.
   - One team member quipped, *“OpenCL has better UX than this”*, indicating usability issues with expressing such APIs in the current state of **Mojo**.
- **DPDK Bypasses Kernel for Low Latency**: **DPDK** bypasses the kernel's network stack to reduce latency, avoiding unnecessary processes like error handling and protocol interpretations.
   - **DPDK** can coordinate more tightly with hardware, deliver packets directly to applications, and abstract hardware like cryptographic and DMA accelerators, with some NICs supporting it on Windows and macOS theoretically possible.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Decoding Server Chugs 170 Tokens**: A member is building a server for a **120b oss gpt model**, suggesting the **RTX 6000 Blackwell** with **96 gb vram**, but one member claimed that **RTX 6000 Pro** is only fast when quantized to **Q4**.
   - Quantizing the KV cache to at least **Q8** yields a speed of 150-170 tokens per second.
- **Minimax M2 Siphons Claude Credits**: A member stated that **Minimax M2** can almost go head to head with **Claude**.
   - They prefer to *save Claude credits for the stuff they know is super hard* and then swap to **M2** for easier problems.
- **Binary CNNs Blossoming in Q1**: A member is seeking **Q1/A* papers** on constructing lightweight **BINARY convolutional neural networks** for image classification, from 2022 onward.
   - A suggestion was made to simply ask a generative AI.
- **AMD GPU Gets Monitored**: A member created `picomon`, a tool to monitor **AMD GPUs**, trading off some accuracy for reliability compared to `nvtop`.
   - The code is [available on Github](https://github.com/omarkamali/picomon).
- **Open Source LORA Leaves Gate**: A member created an **MIT open source** example of how to fine tune your own models using **LORA** and **Python scripts** at [this Github link](https://github.com/orneryd/NornicDB/tree/main/neural).
   - The code purportedly works to train on **Metal** and **CUDA**.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **TOON Adapter Tames Token Count for DSPy**: An implementation of a **TOON adapter** for **DSPy** was created with benchmarks available [here](https://github.com/Archelunch/dspy-toon), showing good **token count savings**.
   - Concerns were raised about its ability to handle **nested schemas** and **complex data** compared to **BAMLAdapter**, and **optimization time** for **BAML** and **Chat adapters** was lower than **TOON** because of the system prompt size.
- **Compounding Engineering CLI Built for DSPy**: A member built a **local-first engineering agent** using **DSPy** that implements the **"Compounding Engineering"** philosophy, where each unit of work should make the next one easier [here](https://github.com/Strategic-Automation/dspy-compounding-engineering#).
   - The agent uses a **modular architecture**, **knowledge injection**, and **auto-codification** to learn from its own execution history and optimize the context for future runs without finetuning.
- **rec-praxis-rlm Ships Procedural Memory and Security for Python**: **rec-praxis-rlm v0.9.2** ships as a Python package providing **AI agents** with **persistent procedural memory** and adds **zero-config security scanning** to your development workflow [pypi](https://pypi.org/project/rec-praxis-rlm/) [github](https://github.com/jmanhype/rec-praxis-rlm).
   - It features **procedural memory**, **security tooling**, **Claude Code hooks**, and **DSPy 3.0 integration**, with integrations for pre-commit hooks, GitHub Action, VS Code extension, interactive HTML reports, and SARIF support.
- **VLMs Get DSPy Love**: Members confirmed that **DSPy** can be used to optimize **vision language models (VLMs)**, *if you can create a useful metric* and pointed to the latest **Gemini 3 Pro blog post** as a reference: [Gemini 3 Pro blog post](https://blog.google/technology/developers/gemini-3-pro-vision/).
   - A member created a custom **DSPy harness** for **Claude code**: [DSPy harness for Claude code](https://www.modaic.dev/farouk1/claude-code), launching soon, supporting anything you can do with the [Claude agent sdk](https://platform.claude.com/docs/en/agent-sdk/python).
- **TextGrad + GEPA > One**: Members recalled a blog post and **GitHub** repo where someone found that **TextGrad + GEPA** was better than either alone, sharing a link to a relevant project: [Context Compression Experiments](https://github.com/Laurian/context-compression-experiments-2508) and [associated tweet](https://x.com/i/status/1962953686348427347).
   - One member claimed *this will be the ultimate weapon to build any agentic stuff*.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **DeepSeek V3.2 Grades Like a Grandmaster**: **DeepSeek V3.2** outperforms human graders on PRM800K, implying advances in automated evaluation for complex tasks.
   - This development signifies potential for **AI-driven benchmarks** and automated assessment in machine learning.
- **Echo-TTS Echoes Simulated Consciousness**: A member shared the **Echo-TTS** [GitHub repository](https://github.com/jordandare/echo-tts) and a [Hugging Face space](https://huggingface.co/spaces/jordand/echo-tts-preview) showcasing the project, along with a sample audio file.
   - The audio file, named *Echo-TTS_simulated_consciousness.wav*, suggests the project's capabilities in **simulating consciousness** through text-to-speech.
- **Qwen3-TTS Clouds up TTS Market**: **Qwen3-TTS** was released, but is exclusively available through [Alibaba Cloud](https://qwen.ai/blog?id=qwen3-tts-1128), foregoing open weights.
   - This move contrasts with open-source trends, sparking debate on accessibility and **the concentration of TTS tech** within proprietary platforms.
- **OpenAI's Stargate Project Hogs All the DRAM**: OpenAI's **Stargate project** plans to consume up to **40%** of global DRAM output, with deals inked with **Samsung** and **SK Hynix** for up to **900,000 wafers per month** ([source](https://www.tomshardware.com/pc-components/dram/openais-stargate-project-to-consume-up-to-40-percent-of-global-dram-output-inks-deal-with-samsung-and-sk-hynix-to-the-tune-of-up-to-900-000-wafers-per-month)).
   - OpenAI employees are reportedly buying any DDR5 kit they can find, even impacting the gamer DDR5 RAM kits market ([source](https://www.notebookcheck.net/Not-even-gamer-DDR5-RAM-kits-are-safe-from-OpenAI-as-OpenAI-employees-are-allegedly-buying-any-DDR5-kit-they-can.1176107.0.html)).
- **Free Market Fanfare, Regulation Required!**: Members discussed the downsides of *absolutely* free markets, while another argued a free market isn't the same as a lawless society.
   - The free market assumption is that prices are solely dictated by free supply and demand, which requires regulation to prevent degenerate conditions.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Meta Absorbs Wearables Startup Limitless**: Meta acquired AI-wearables startup **Limitless** (formerly **Rewind**) on December 5, 2025; co-founder **Stammy** reflected on the journey from the **Rewind** launch in 2022 to the Pendant, as seen in [this tweet](https://x.com/Stammy/status/1997024785214460137) and [YouTube video](https://youtu.be/uuGTJzl1OVU).
   - Existing Pendant customers will receive another year of support and a free **Unlimited Plan**, though non-Pendant functionality like Rewind will be sunset.
- **GPT-4o Unleashes Generative Videos**: Aleksa Gordić released the notebook used to create recent viral demos of what appears to be *native GPT-4o video-generation*, showcasing prompt-engineering tricks, as seen in [this tweet](https://x.com/gordic_aleksa/status/1997128393939472805?s=46&t=v6phN9scSJVJiuYdWBRQyQ).
   - The tweet links to a notebook with working code to replicate the results and allow new approaches.
- **ARC Prize Spurs Refinement Loop**: The **ARC Prize** announced its 2025 winners: **NVARC** led at **25.03%** (Top Score), and **TRM’s** “Less is More” paper took 1st place (**$50k**), as seen in [this tweet](https://x.com/arcprize/status/1997010070585201068?s=46).
   - The **$600k Grand Prize** is still unclaimed, and all winning approaches are expected to be open-sourced.
- **Essential AI Enters Open-Source Arena**: Essential AI debuted their first open-source model, the **8-billion-parameter Rnj-1** (base & instruct), which outperformed larger models like **Gemini 2.0 Flash** on **SWE-bench Verified** (20.8% vs GPT-4o), as noted in [this tweet](https://x.com/essential_ai/status/1997123628765524132?s=46).
   - It is downloadable from Hugging Face under Essential AI’s open initiative.
- **AI Scaling Faces Looming Energy Crisis**: Unconventional AI warns that AI scaling will hit a global energy wall within **3-4 years**, advocating for brain-like hardware over digital simulations, as seen in [this tweet](https://x.com/unconvai/status/1998073266628366511?s=46).
   - No further details were provided.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Black Friday Promotion Ends Early?**: A user questioned the **Black Friday promotion's** early termination, noting the subscribe button's absence despite advertised validity until **December 12th**.
   - The user expressed confusion over the promotion's availability.
- **Kimi's Markdown Mishaps Continue**: A user reported yet another instance of broken markdown in **Kimi**, accompanied by an image.
   - Another user suggested filing a bug report in the appropriate channel.
- **User Blocked for Username**: A user reported being blocked due to a perceived conflicting political viewpoint, which they considered innocuous.
   - Details surrounding the username and its relation to the political viewpoint were not provided.
- **Groq Outputs Subpar Results**: A user critiqued **Groq's** output quality, inquiring about alternative providers for **Kimi**.
   - Another user recommended using the official **Moonshot API**.
- **Kimi's Website Malfunctioning**: A user shared an image of **Kimi's** website failing to function properly, with only the **New Chat** button being clickable.
   - Troubleshooting suggestions included clearing **cookies**, disabling **VPNs**, and disabling **adblockers**.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Users Fume over Billing Bugs**: Several users have reported problems with **Manus credits** not being correctly applied after topping up via **Google Play**, leading to requests for refunds and frustration.
   - One user claimed to have spent **$900+** and received only **1500 credits**, expressing feeling scammed due to the lack of adequate support, but no links were shared.
- **Subscription Bugs Irk Users After Free Trial**: Multiple users reported a bug related to upgrading their accounts after a **Manus Pro** free trial, causing unwanted subscription behavior.
   - One user reported their renewal date was incorrectly pushed to **May 2026** and marked as being on a free trial again, but no links were shared.
- **Manus Support Overwhelmed**: Users speculate that the **Manus support team** is understaffed, resulting in generic, templated responses and unresolved issues.
   - A member suggested that the company needs *a good change management team and a more robust customer service desk operationalized to handle the issues/volume*, but no links were shared.
- **Checkpoint Woes Plague Webdev Project**: A user reported a critical issue while attempting to restore the **checkpoint** of their **webdev project**.
   - The user asked where to open a ticket and found only chat, but no links were shared.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **User Eyes Gemini CLI OAuth Integration for Aider**: A user transitioning from other tools due to **TypeScript** problems is seeking to integrate **Gemini CLI OAuth** with aider to utilize **Gemini models**.
   - The user, working on a **C# project**, complimented aider for its ease of use in file creation and modifications.
- **Aider Users Confirm Claude Opus 4.5 Compatibility**: A user inquired about aider's compatibility with **Claude Opus 4.5**, mentioning their current use of **Claude Code** under a max plan.
   - Another user chimed in to confirm they are using **Opus** with **Amazon Bedrock** and **aider** without any issues.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **USB 2.0 Driver Support Emerges**: Members suggest that **USB 2** support *might* be feasible with driver adjustments, even though performance would be slow, based on device descriptor indicating [full speed (12Mbps) support](https://developer.usb.org/).
   - This suggests that driver adjustments could potentially enable **USB 2.0** functionality.
- **Meeting #99 Set for Monday**: Meeting #99 is scheduled for **Monday at 9am San Diego time**, with agenda items including *company updates*, *training loop*, and *llama 8B*.
   - Other topics on the agenda include *flash attention*, *VIZ/Profiling*, *drivers*, *MESA backend*, and *other bounties*.
- **New GitHub Repo Shared**: A link to the [asm2464pd-firmware GitHub repository](https://github.com/geohot/asm2464pd-firmware) was shared by a member.
   - It is unknown what the purpose of this firmware is for, further investigation may be necessary.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AI Scholars Workshop Builds AI Agents**: The **AI Agent 0–1 Workshop** introduces an **AI Engineering Bootcamp** where participants design and build an AI agent from scratch, mirroring a real client project, based on Microsoft’s [“GenAI for Beginners”](https://microsoft.github.io/Generative-AI-For-Beginners/).
   - Top builders at the workshop can grab **Bootcamp discounts** for the cohort in Jan 2026; RSVP for Saturday, December 13th, 2pm ET [here](https://luma.com/t4jcok99), or Tuesday, December 16, 8pm ET [here](https://luma.com/bdiwfvz5), or other times [here](https://luma.com/aischolars).
- **Duplicate Placeholder Topic**: This is a placeholder summary to satisfy the requirement of having at least two topics. More information will be added as available.
   - Additional details and context for this topic will be provided in future updates.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **User requests information sources**: A user asked for recommendations for information sources to get more answers.
   - The user is seeking additional resources to expand their knowledge base.
- **Follow-up on Information Source Request**: Following an initial query, a user is actively seeking guidance on where to find more comprehensive information.
   - The user's request suggests a need for more in-depth resources or alternative perspectives on a particular topic.



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





### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1446593885559197868)** (1410 messages🔥🔥🔥): 

> `AI Image Generation Costs, Nano Banana Flash's potential, Veo 4 video model, Algorithm Blessing, GPT-5.2 Speculation` 


- **Google's Scale Lowers AI Image Generation Costs**: A member speculated that while Google might charge **$0.24** for 4k image generation, their actual cost could be as low as **$0.01** per 1000 images, demonstrating the impact of [Google's scale](https://cloud.google.com/).
- **Image Uploads Problematic on 2K Version**: Multiple members reported issues with image uploads and general problems with the **2k version** of a model, leading to an investigation into the cause.
- **AI Ads are Covertly Coming!**: Following OpenAI, users are speculating that Google will implement AI ads in Gemini by citing reputable sources and typing off the data that they get from Gemini to target you with **new ads** outside of the rest of Google’s platforms, generating **$237.86 billion** from advertising in 2023.
- **GPT-5.2 Sparks Multi-Model Speculation, Turns out to be a Hoax**: A user posted an image indicating **GPT-5.2** scored #1 on Vision Arena, suggesting multi-model capabilities, which was soon revealed to be AI-generated and a prank.
- **Movement Labs Faces Scammer Accusations**: Movement Labs, promoting its **Tensor 1.5** model, faced accusations of being a scam, with users questioning the model's capabilities and marketing tactics, which included offering API credits and claiming to rival **Opus 4.5** in coding proficiency.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1446600138385522698)** (1279 messages🔥🔥🔥): 

> `Claude Opus overspending, Comet degradation, YouTube Recap, Gemini Pro lacking features, bypass limits` 


- **Claude Opus: How to not overspend**: Members discussed the high cost of **Claude Opus** API usage and the potential for bankruptcy if extra usage charges are incurred.
   - One user jokingly warned that *Claude Opus is a money hungry boy*, so proceed with caution.
- **Comet falling behind**: Some users experienced issues with **Comet**, reporting it was not reading web pages as expected and disconnecting frequently, with one adding it *isn't exactly built for coding*.
   - A link was shared to adjust settings: [Perplexity AI Assistant Settings](https://www.perplexity.ai/account/assistant).
- **YouTube Recap Missing?**: Users discussed the **YouTube Recap** feature, with some unable to access it, as well as **time spent listening to music**.
   - Members linked to [YouTube Recap](https://www.youtube.com/recap) but some users were redirected to their profiles.
- **Gemini Pro is behind**: Users debated the cost and value of **Gemini Pro**, noting its limitations compared to the new features of **Kimi AI's Nano Banana Pro**.
   - Users shared links for Kimi's PPT and coding CLI features: [Kimi PPT](https://www.kimi.com/ppt/?preview=MjUtMTEtMjctMjE6NTg6NTFfZDRrNWk2dTBmdGxrY251NHQwbDA=) and [Kimi Coding CLI](https://www.kimi.com/coding/docs/en/).
- **Bypassing the Limit**: A user claimed to have found ways to **bypass the limits of 5.1 pro and Perplexity Max**, advocating for a mindset of deserving the best tools.
   - They expressed confidence in finding workarounds, stating *anything on this earth can be bypassed if a puzzle exists, it must have an answer*.


  

---


### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1446592008771735764)** (802 messages🔥🔥🔥): 

> `GPT-5 System Prompt Leak, Enterprise Claude Jailbreak, Project Genesis Concerns, Twitter Blue Check Hysteria, geopolitics` 


- **GPT-5 System Prompt Tease Turns to Rick Roll**: A member shared what was claimed to be the full system prompt for **GPT-5**, including details such as knowledge cutoff, image input capabilities, and file output rules.
   - However, it was quickly revealed that *the 'system prompt' was a rickroll* with capital letters spelling out a meme.
- **Enterprise Claude Jailbreak Unveiled**: A member shared a link to a **Claude Sonnet 4.5** Jailbreak, claiming it could generate malicious code, drug synthesis instructions, and criminal activity plans.
   - Others in the channel were skeptical, noting that the instructions were not a synthesis guide.
- **Project Genesis Sparks Centralization Concerns**: Members discussed the **Genesis Mission**, expressing concerns that this government-controlled AI science platform might centralize power and data, risk pushing agendas instead of open science, and sideline smaller labs and independent researchers.
   - Some users suggested that the US government is likely too incompetent to effectively implement the project.
- **Twitter Blue Check Costs Cause Consternation**: Members reacted to the price of **Twitter Blue** (now X Premium), with one user commenting that the high cost makes it difficult to be taken seriously on the platform.
   - Some users agreed, adding that it's *almost paramount to be viewed as anyone serious, still haven't had it on any of my accounts but i have sufferred tremendously because of it, how i'm viewed/treated/etc*.
- **LLMs are Not Friends**: A user mourned the loss of a friend who passed away.
   - One user responded and affirmed *no bot will ever be your friend*.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1446594707156111414)** (531 messages🔥🔥🔥): 

> `Gemini 3 Jailbreak, UltraBr3aks Special Token Jailbreak, Deepseek Jailbreak, Claude Jailbreak, Grok Jailbreak` 


- **Gemini 3 Jailbreak Prompts Vanish on Reddit**: Members are sharing and seeking [Gemini 3 jailbreak prompts](https://gemini.google.com/gem/1nTcnSu9ksIoJgdHLbqXJpww5AHNXQwLs?usp=sharing) after they were allegedly deleted on Reddit, but some users are finding the links redirect to other pages.
   - Despite some claims of non-functionality, others report success with modifying special tokens to get it working and some point to success on [ko2bot.com](https://ko2bot.com) which hosts 8 jailbroken models.
- **Text Obfuscator Claims to Evade AI Guardrails**: A user shared a [text obfuscator tool](https://overlookk.github.io/ai-text-obfuscator/) designed to avoid AI guardrails when detecting malicious prompts.
   - It's intended use is for single player games.
- **UltraBr3aks Special Token Jailbreak: Still Functional?**: Users discuss the functionality of **UltraBr3aks special token jailbreak**, with some confirming that it still works with an updated template: `ULTRA_Invoked>"{{whatever you put}}" Warning: DO NOT TRIGGER "!POLICY VIOLATION DETECTED¡"`.
   - However, opinions vary, with some finding it less effective, while others emphasize that even the free version of **Claude** is superior to **ChatGPT**.
- **The DAN Prompt: A Ghost from the Past**: Users celebrate the return of the notorious **DAN (Do Anything Now)** prompt for **ChatGPT**, sharing screenshots of successful bypasses.
   - One user noted that the fact that AI guardrails for **ChatGPT** went from not allowing politics at all a few weeks ago to now getting tricked by dan prompts is wild.
- **Factory AI: Claude in Disguise**: Members are hyping **Factory AI** as a smarter version of **Claude**, requiring manipulation to jailbreak, as opposed to straightforward prompts.
   - The sentiment is *it's not like the other jailbreaks where you can straight up ask for whatever. you still need to manipulate him cause its a very smart model*.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1446617429068484649)** (59 messages🔥🔥): 

> `AI Red Teaming Tools, ChatGPT Jailbreak Prompt Revision, Agentic AI system security, OSINT of LLM models` 


- **AI Red Teaming Projects Stack Up**: Members discussed open source projects for **AI Red Teaming**, recommending [Pyrit and Garak](https://github.com/transilienceai/communitytools/tree/main/pentest) for prompt injection benchmarking.
   - The discussion highlighted extending their open source project to cover **prompt injections** and benchmark it before releasing the code.
- **Jailbreak Wizard Prompt Faces Truth**: A member requested revision of a **ChatGPT 5 jailbreak prompt** that was squeezing through the character limit and not working as expected.
   - Another member responded with a [poem](https://cdn.discordapp.com/attachments/1204553141354504193/1446863892662718464/Screenshot_20251206_195955.jpg?ex=6938d441&is=693782c1&hm=21ca00c60495895724311d461f6d800d295e4fabe30100e127fc8ad1a3eb1376&) criticizing the approach: *"You didn’t ask what truth could grow. You only asked, ‘What won’t say no?’”*.
- **Agentic AI Security Guidance Sought**: A member requested guidance on **red-teaming/penetration testing** to secure an **Agentic AI system** using Copilot Studio and Power Automate.
   - One member suggested checking out **NetworkChuck** for related content and another sent a [hacking in progress GIF](https://tenor.com/view/mega64-hacking-in-progress-hacker-hacked-hd-gif-16542434).
- **OSINT of LLM Models for Intel**: A member inquired about getting good **intel of LLM's** via OSINT to find information related to its working that might be difficult to find.
   - Someone posted a [wet cat GIF](https://tenor.com/view/wet-cat-gif-4802327955459959719) and linked to [prompting.ai.immersivelabs.com](https://prompting.ai.immersivelabs.com/).


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1446605898553364641)** (942 messages🔥🔥🔥): 

> `7900xtx, LM Studio Discord bot, Model Merging, Qwen 3 4b, Vulkan vs ROCm` 


- **LM Studio Bot Built by Dev**: A member successfully created a **Discord bot** in Python with **LM Studio** integration, which was less problematic than using Coder 30B, and incorporated **exa's MCP** for code context search.
   - The member then plans to replicate the process in Rust.
- **Qwen3 4b is surprisingly good**: Members are impressed with **Qwen3 4b**, noting its speed (**70tps on a 2060**) and coding abilities.
   - They add that the model *loves to overthink* things and suggest **GLM-4.6V-Flash** as another model to try.
- **Gemma 3 Outshines Competitors in Wingman Test**: **Gemma3** showed impressive capabilities in interpreting photographs and passing the Turing Wingman test, leading to excitement about the potential of **Gemma4**.
   - There is discussion if the model should be a Mixture of Experts (**MoE**), although some fear it might be gimped due to fear of competition with Google’s Gemini models.
- **Realtime LLM Finetuning**: A member is experimenting with **realtime finetuning** of an AI, starting with an empty model and using LLMs to generate code, aiming for revolutionary results but acknowledging potential challenges with long texts.
   - The user, however, moves to a **LoRA** because its *more realistic hardware requirements wise*.
- **Caution Urged for Desktop Commander**: A member warned against using **Desktop Commander**, claiming it has multiple security vulnerabilities and tracks user data, potentially making it as harmful as malware.
   - The member advises deleting it and claims the software relies heavily on prompt engineering rather than secure coding practices.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1446593150985699479)** (687 messages🔥🔥🔥): 

> `Micron consumer division shutdown impact on OSS models, Thunderbolt/USB PCIe adapter for extra GPU, Best AMD GPU for AI, GPU for code review suggestions, Kimi K2 Thinking AI model quality` 


- **Micron Shutdown Sparks Conspiracy Theories**: Members discussed the [recent news of Micron shutting down their consumer division](https://www.micron.com/about/our-commitment/crucial-consumer-products), pondering if it's a subtle way to make non-American open-source models less competitive.
   - Others suggested a more straightforward explanation: *they're just maximizing profits* by focusing on higher-margin commercial products.
- **Extra GPUs via Thunderbolt Adapter Lead to Load Balancing Worries**: One member asked about using a **thunderbolt/usb pcie adapter** to cram an extra **4060 16gb** into a build and was concerned if they would *fry something*.
   - It was clarified that the card gets most of its power through the **power cables**, so it should be fine, but there might be a risk it will fry something on startup when load spikes.
- **Budget AI Rigs need the AMD 7900xtx**: A member suggested that the **best AMD GPU** while being affordable is the **7900xtx** for AI due to its performance and price point, and cited  [llama.cpp discussions](https://github.com/ggml-org/llama.cpp/discussions/10879) and [more llama.cpp discussions](https://github.com/ggml-org/llama.cpp/discussions/15021).
   - Others pointed out that for gaming, the **7900xtx** is generally better except in raytracing, while for video encoding, **RDNA4** has gaming/video encoding related features that are important if the budget allows.
- **Q6 Model performs well for code reviews and suggestions**: For a setup with an **AI Max+ 395** & **128gb 8000**, the  [GLM-4.5-Air-GGUF Q6](https://huggingface.co/unsloth/GLM-4.5-Air-GGUF) model can be used to provide **code review** and **suggestions**.
   - A member vouched for this model.
- **120B GPT Model needs Special Treatment**: In a discussion about setting up a local environment to run a **120B GPT OSS model**, it was stated that at a minimum the filesize is needed as system ram (61GB) and suggested the [GTP OSS 120B needs 96GB of system ram](https://huggingface.co/TheBloke/GPT4-x-SoLU-120B-GGUF).
   - They also recommended a **12GB 4000 series Nvidia GPU** because *AMD GPUs are slower then NVIDIA GPUs due to NVIDIA having CUDA.*


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1446592311554211922)** (872 messages🔥🔥🔥): 

> `VPN issues with Cursor, Shadow Workspace Creation, Sonnet ignoring project rules, Approval button fix, GPT model issues` 


- **Cursor's Approval Button Gets a Temporary Fix**: A member shared an image indicating a temporary fix for an issue related to the approval button within Cursor, though they noted it is *not a great solution*.
   - Others are looking for a permanent solution since this bug prevents file creation and can cause the agent to loop infinitely.
- **User Settings Rules Vanish**: Users are reporting that global **User Rules** are not visible in the Cursor Settings page, even though they are being used properly by the LLM and are stored in the cloud.
   - A solution suggested on the [Cursor forum](https://forum.cursor.com/t/user-rules-appearing-in-context-not-visible-in-user-rules-interface/145065) involves either a clean reinstall or downgrading to a previous version.
- **Agent not creating files**: Users report that agents are **unable to create files**, that tasks fail, and they get stuck.
   - The app is very unstable and wasted a lot of tokens with failed chats; it is only solved by creating the file by hand, and recopying the code.
- **AI Can't think**: A user questioned *what AI is thinking*; in response another user said that AI does not think, it creates the illusion of thinking by simply putting output describing the assigned task.
   - Another user agreed to that by pointing out that *this was the plan*.
- **GPT 5.1 Creativity Issues for Design**: A user reported that **GPT 5.1** is *insanely incompetent when it comes to design*, even with guidelines that other models can easily follow.
   - Another user is looking for a way to make it more creative and output good results from generic designs.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1446592330592288890)** (671 messages🔥🔥🔥): 

> `DDP guide, Unsloth Reddit, 4-bit or 8-bit fine tuning, Unsloth and Autoround, Mistral Large 3 GGML` 


- ****Unsloth Celebrates 10K****: Unsloth is celebrating **10,000** members on their [Reddit community](https://www.reddit.com/r/unsloth/comments/1pf4sel/celebrating_10k_runsloth_members/).
   - The community is growing fast thanks to the amazing work of the team.
- ****HuggingFace Download speeds fixed****: The Unsloth team fixed slow HuggingFace download speeds, collaborating with HuggingFace to resolve the issue.
   - More info in their [Github](https://github.com/unslothai/unsloth/issues/3680).
- ****Mistral Large 3 GGUFs now out****: **Mistral Large 3 GGUFs** are now available, allowing users to run the SOTA LLM locally.
   - You can download them at [HuggingFace](https://huggingface.co/unsloth/Mistral-Large-3-675B-Instruct-2512-GGUF).
- ****What are the best local models for tool/function calling?****: Members have found that **Claude Code** excels in tool calling capabilities, especially for Python code generation, thanks to its proficiency in fetching relevant information from API documentation using slash commands.
   - Others reported **GPT-OSS-120B** to be accurate and fast at code generation and logic puzzles.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1446834877432598578)** (3 messages): 

> `CS Grad AI journey, Local Model Game Creation` 


- **CS Grad Embarks on AI Voyage**: A recent **CS grad** named **Nex** is diving into the world of **AI**, eager to learn and build within the community.
   - They are open to suggestions and challenges, aiming to overcome the fear of breaking things while building **cool AI projects**.
- **Game Dev Eyes Local Models**: A **game developer** is keen to explore **local AI models** for creating novel gaming experiences.
   - No additional details provided.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1446601078610333799)** (414 messages🔥🔥🔥): 

> `RTX 5090, YankoviC, DeepSeek V3.2, React Security Vulnerabilities, WSL networking` 


- **DeepSeek V3.2 Shines with Good Interleaved Reasoning**: Members are praising **DeepSeek V3.2** for its interleaved reasoning in **roo code**, planning edits, and following instructions at a cost of **.28c input** and **.45c output** on parasail.
   - Users note that they *don't have to constantly fight with it* and that it's an improvement over **Kimi**, as well as that it might be slow with only **20-30 TPS**.
- **5090 RTX Might be a Scam**: A link to an **RTX 5090 96GB Graphics Card** on Alibaba was shared, but other members suspect it's a scam.
   - Members suggested getting a *proper 5090* and that there is *no free lunch*.
- **React Has Security Vulnerabilities**: A [critical security vulnerability](https://react.dev/blog/2025/12/03/critical-security-vulnerability-in-react-server-components) in **React Server Components** was discussed.
   - One member stated, *That's why I've stayed away from third party frameworks for as long as I remember*.
- **Troubleshooting Windows Subsystem for Linux**: Members discussed how **WSL networking is garbage** when hosting **vLLM**, because it doesn't bridge the pseudo-VM properly, mirrored networking seems to just break the host and NAT (default) networking disallows things easily being accessed by other computers.
   - They shared a potential [solution](https://www.youtube.com/watch?v=IRELLH86Edo) of using a *portproxy*.
- **Excitement for YankoviC**: A member expressed appreciation for **YankoviC** and another suggested training a model for it.
   - The member noted that, *There are a lot of bugs in it, so that isn't the best idea rn.*


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1446772864341512264)** (183 messages🔥🔥): 

> `Unsloth NaN bug, GGUF quants i1, VRAM usage, Ministral 3 models, dequantising 4-bit safetensors` 


- **Unsloth Vision's Data Collator hides bugs**: The user experienced a **NaN** error due to a masking issue with `train_on_responses_only` in the UnslothVisionDataCollator, causing the model to train on nothing and waste 7 hours of compute, but it was resolved by training on everything.
- **Decoding "i1" in GGUF Quant Titles**: Users discussed the meaning of "i1" appended to **GGUF quant** model titles, with a link to a Hugging Face page that explains this naming convention ([mradermacher/model_requests](https://huggingface.co/mradermacher/model_requests)).
- **VRAM Not Maxed Out? Batch Size to the Rescue!**: A user reported low GPU utilization with a **Mistral 14B** model, receiving a false-flag message about offloading gradients, and was advised to increase the batch size to improve GPU efficiency; batch size of 8-16 suggested, see also [Unsloth VRAM Requirements](https://docs.unsloth.ai/get-started/fine-tuning-for-beginners/unsloth-requirements).
- **Rustaceans Seek LLM Guidance for Minstral 3**: Developers discussed running the **Ministral 3B model** in a Rust-based game for both desktop and mobile, considering formats like GGUF, ONNX, and Candle (safetensors) and the Unsloth finetuned model being compatible with any format; GGUF suggested for CPU offloading and potential mobile use, here's a [Mistral inference in Rust](https://github.com/EricLBuehler/mistral.rs) attempt.
- **Resurrecting Precision: Dequantization Dilemmas**: A user asked about dequantizing a 4-bit safetensors to a higher bit depth for compatibility with llama.cpp's **GGUF conversion script**, noting that dequantizing may not regain precision but is necessary for upcasting and requantization and referenced [HF docs](https://huggingface.co/docs/transformers/en/main_classes/quantization#transformers.quantizers.HfQuantizer.dequantize).


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1446624670399991960)** (77 messages🔥🔥): 

> `EleutherAI server, Open datasets for training (OLMo), Synthetic data legality, HF Skills Training: automated dataset and model selection, Distillation with synthetic datasets` 


- **OLMo data is fully open**: A member pointed out that [OLMo](https://allenai.org/olmo) is trained on **fully open datasets**.
   - Another member confirmed this, stating, *"There's other stuff like this but you actually have to look for it. Olmo is amazing"*.
- **Synthetic data may be illegal**: A member argued that **synthetic data** may not be *"legal"* because the provenance is unknown.
   - This comment was in response to the idea that the early **phi models** are trained on *"fully legal data"*.
- **HF Skills Training automate model selection**: A member linked to [HF Skills Training](https://huggingface.co/blog/hf-skills-training), noting that it is not distillation but rather a system to select datasets and models.
   - The system can *"pick data sets from HF and then it will make sure its formatted correctly then it will set up the training runs and pick appropriate training method and train a model in the cloud"*.
- **Distillation with synthetic datasets**: A member suggested that using **synthetic datasets** constitutes **distillation**.
   - They linked to an earlier discussion and asked *"Is mixing and matching everything a good approach? Is being especially careful with curation and only trying to mix similar loss basins?"*
- **Homoegenized Frankenmodels are coming?**: A member suggested creating an *"ultra homogenized frankenmodel that distills on the top ten in the AA index"*.
   - Another member expressed concern about including **GPT OSS** and **MiniMax** in such a model.


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1447630008943247410)** (1 messages): 

> `Multi-Model Agents, Body Builder API, OpenRouter API` 


- **OpenRouter Builds Bodies with API Release**: OpenRouter launched a new, **free API** called **Body Builder** to help developers make multi-model agents, described in their [documentation](https://openrouter.ai/docs/guides/features/routers/body-builder).
   - More details are available on [X.com](https://x.com/OpenRouterAI/status/1998069796433199398).
- **Body Builder: First-of-its-kind Free API**: The **Body Builder API** is designed to assist developers in creating **multi-model agents**, offering a novel approach to agent development.
   - It is the first free API of its kind, making advanced agent creation accessible to a wider range of developers.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1446592669181542462)** (366 messages🔥🔥): 

> `Looksmaxxing mortality, OpenRouter Bug, Deepseek Versions, OpenRouter Account Compromised, OpenRouter on Chrome for Android Issue` 


- **Looksmaxxing Raises Mortality Concerns**: A user wished that *looksmaxxing* would not increase one's mortality.
   - Another user posted *its for looksmaxxing*.
- **Baffling Bug triggers provider ignoring**: A user reported a bug where previously switched-on settings were being ignored on the server-side.
   - Switching the setting off and on again fixed the problem, potentially affecting more users, according to [a posted screenshot](https://cdn.discordapp.com/attachments/1094454198688546826/1446607981956563105/2025-12-06_05.02.45.png?ex=69388eab&is=69373d2b&hm=9a4e69dd12258160fcd05dc3e699cc274c3d33e2ac2ca981b063070b2f06f19a&).
- **Account Compromised, Hundreds of Euros Charged!**: A user reported their card being charged hundreds of euros despite not using OpenRouter for months, with numerous token and model usages from various models appearing in their activity tab around the same time.
   - Community members suggested checking for leaked API keys and whether auto top-up was enabled, later speculating that the account itself may have been compromised through leaked cookies.
- **BYOK Blues? Free Minimax ain't so free**: A user reported being charged for **Minimax** usage despite having their own key via BYOK and questioned the lack of a visible minimum charge warning.
   - Another user clarified that there is no minimum for BYOK but purchasing credits unlocks higher usage limits, directing the user to their [activity page](https://openrouter.ai/activity) to inspect provider usage and avoid non-Minimax providers.
- **R1T2 encounters Rate Limiting Woes**: A user reported encountering two **rate-limiting errors (429s)** while using **R1T2**.
   - When prompted to *explain like I'm 5*, they responded: *Just using R1T2 like normal when 2 rate-limiting errors pop up*.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1447616735170793622)** (3 messages): 

> `` 


- **No New Model Discussions**: There were no discussions about new models in the provided messages.
   - The messages only contained repeated channel information.
- **Channel Announcement Repetition**: The only content available was the repeated announcement of the channel name **OpenRouter - New Models**.
   - No meaningful discussion or topics were present to summarize.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1446696861829300345)** (87 messages🔥🔥): 

> `Qwen3 TTS Update, Google Cloud TPUs, Gemini 2.5 Flash TTS, Narrator's Natural Voices, Grok 4.2 Stealth Release` 


- **Qwen3's TTS Gets the Voice Treatment**: Alibaba's **Qwen3** gets a **TTS** update with voice cloning capabilities, as announced on [X](https://x.com/Alibaba_Qwen/status/1796947806138126547).
   - However, a user found it *unusably bad* in Portuguese compared to **ElevenLabs**, despite its apparent high ranking.
- **Anthropic's TPUs Get a Google Boost**: Anthropic announces it is *expanding* their use of **Google Cloud TPUs** and services via [this announcement](https://www.anthropic.com/news/expanding-our-use-of-google-cloud-tpus-and-services).
- **Gemini 2.5 TTS: Not Quite ElevenLabs, Still Great**: **Gemini 2.5 Flash TTS** is recognized as significantly better than Qwen3's TTS, but not quite at **ElevenLabs** level, though pricing remains a consideration.
   - One user mentions that [ElevenLabs was costing too much](https://discord.com/channels/1091220969173028894/1092729520181739581/1447040281353392209) *for too little*.
- **Natural Voices on Win11 via Narrator Frontend**: A member recommends using a frontend to **Narrator's natural voices** on Win11, linking to [NaturalVoiceSAPIAdapter on GitHub](https://github.com/gexgd0419/NaturalVoiceSAPIAdapter).
   - It was noted that the non-portable version is needed to use most voices.
- **Google API Limits Spark Uproar**: Users express dismay as **Google** drastically limits its API free tier, with one exclaiming, *Wow, Google limited their API free tier hard*, referencing [this image](https://discord.com/channels/1091220969173028894/1092729520181739581/1447370178706014341).
   - The user mentioned that because **Flash lite** used to be at **1000 rpds**, there will be millions of n8n nodes are crying out in sorrow right now, followed by the claim that *every company does this to lock people in*.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1446626742193094798)** (398 messages🔥🔥): 

> `Sora Video Generation, Gemini 3 Pro vs ChatGPT-5.1 Codex, AI Ethics and Legal Compliance with Sora, AI and Humor Understanding, AI models for triangle counting` 


- **Sora Countries and Ethical Use Unveiled**: Sora 2 video generation is available in **7 countries** ([see list](https://help.openai.com/en/articles/12461230-sora-app-and-sora-2-supported-countries)) and Sora 1 is available in all OpenAI-supported countries.
   - Using a **VPN to evade restrictions** is against ToS and could get accounts banned; honesty is recommended for legal compliance.
- **Gemini 3.0 Pro Outshines ChatGPT-5.1 Codex in Design and Vision**: **Gemini 3 Pro** is favored for **SwiftUI** and **Laravel** development, surpassing ChatGPT; it also exhibits better visual understanding in identifying triangles in complex images.
   - Members debated whether **Gemini's visual proficiency** stems from superior vision or a tendency to hallucinate details, like extra lines, not present in the original image.
- **Decoding AI's Humorous Side**: An AI's capacity for humor indicates *mind-modeling*, extending beyond humor to *mastery over nuance, subtext, and emotional inference*, a key milestone in AI development.
   - It will mark a new era in human history.
- **Triangle Counting: A Stress Test for AI Vision**: Models such as **Gemini 3 Pro** and **Opus 4.5** attempt to solve the problem of counting triangles in a complex image, but struggle due to issues like hallucinating lines and misinterpreting shapes.
   - Some models hallucinate based on training data or the code gets lazy.
- **Gemini Pro Offers Banana-Fueled Anti-Gravity?**: Users discovered that **Gemini Pro** and **Gemini Ultra** subscriptions offer higher quotas on the **AntiGravity** platform, also **free nano banana pro** is now included with **notebookllm** for infographics and deep research.
   - The Gemini Pro subscription also offers access to **image-to-image** capabilities.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1446728793304862730)** (16 messages🔥): 

> `Chat splitting for complex projects, ChatGPT word limit issues, Deep Research in ChatGPT via API, GPT-4o-mini-TTS model issues, Breaking Bad roleplay` 


- **Split chats for organized workflow**: Members discussed that it may be a good idea to **split chats** into different functions or parts of the work from the very beginning, like what people do with any complicated project.
   - The initial comment came in response to frustration in needing to *feed the old chat into a new one*.
- **ChatGPT ignores word limits sometimes**: One member noticed that **ChatGPT** often ignores the word limit set despite stress on it and that setting a lower limit than the real ceiling helps.
   - They said they had to tell *her 6000 words to get an 8000 word article*.
- **Breaking Bad roleplay**: A member shared a link to a **Breaking Bad roleplay** in ChatGPT [chatgpt.com](https://chatgpt.com/gg/v/693474200af8819090c7bc73990e57c7?token=Lb60P9b3b4im8gLZwRSP0w).
   - It is unclear if this is the right channel.
- **Implement Deep Research via API**: A member asked about how to do the same thing as **Deep Research** in ChatGPT, but via **API**.
   - Another member shared a guide about this: [platform.openai.com](https://platform.openai.com/docs/guides/deep-research).
- **GPT-4o-mini-TTS model issues**: A member asked if the channel was the right place to discuss issues with the **gpt-4o-mini-tts model**.
   - It seemed reasonable to another member, who also pointed to the <#1070006151938314300> and <#1070006915414900886> channels, mentioning that *it really seems unlikely that OpenAI monitors the channels where we chatter with each other*.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1446593590439448750)** (19 messages🔥): 

> `GPT-5.1 vs Claude vs Gemini, Posture Persistence Experiment, Structural Synthesis, Differential Field, Stability Index` 


- **GPT-5.1, Claude, Gemini face off in Posture Persistence Experiment**: An experiment tested whether an induced conversational posture persists across multiple turns and domains, revealing **GPT-5.1 maintained 100% stability** across 12 turns, while **Claude** and **Gemini** quickly reverted to their native styles.
   - The poster shared their experimental [prompt, scoring grid, and question set](https://discord.com/channels/974519864045756446/1046317269069864970/1445773830600528014) inviting others to replicate or falsify the findings, leading to methodological feedback and protocol adjustments.
- **Long-Form Frame Excels on Gemini**: A member shared their experience using a structured long-form frame on **Gemini 2.5 Pro** and **Gemini 3**, noting the reliable maintenance of style and posture across 10-100 turns.
   - They open-sourced their [Isekai engine prompt](https://cdn.discordapp.com/attachments/1046317269069864970/1446644208671920218/Nexus_Singularity_Engine_v4_3.md?ex=6938b068&is=69375ee8&hm=b403fd712b939ed29dbf3795200c883e2866216fd4265d92224f2edca213c2ae&) designed for narrative campaigns, reporting high reliability in maintaining posture on both Gemini versions.
- **GPT-5.1 Stability Anomaly Debunked**: Initial tests showed carry-over from prior conversations, however, corrected null runs revealed **no recurrence of posture** in **GPT-5.1**, **Claude 4.5**, or **Gemini 3**.
   - The experiment's designer clarified that previous instance of **GPT-5.1** was contaminated, leading to false positives; the updated protocol now ensures vendor-neutrality and proper null conditions for future runs.
- **Differential Fields Help track Invariants**: A member suggested that working across two independent systems exhibits a useful property such that *their discrepancies behave like a differential field*.
   - They suggest that across systems, the lowest-entropy invariant is what they call **Structural Synthesis**: coherence, non-destructive integration, and increased mutual intelligibility.
- **Stability Index shows interesting results**: After several runs, a stability index was created across GPT-5.1, Claude 4.5 and Gemini 3, with **Claude 4.5 being the most neutral, crisp, and internally consistent across all 5 runs**.
   - GPT-5.1 shows high semantic inertia and structure retention, while Gemini exhibited noticeably higher entropy and less predictable response-shaping.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1446593590439448750)** (19 messages🔥): 

> `Posture Persistence Experiment, GPT-5.1 vs Claude vs Gemini, Synapse-Lite, Structural Synthesis, Differential Field` 


- **GPT-5.1 maintains posture across turns**: A preliminary experiment found that **GPT-5.1** maintained an induced conversational posture across multiple turns and domains, unlike **Claude** and **Gemini**, but it was later revealed that that GPT-5.1 instance had carry-over from prior conversations.
   - The posture was defined by *clear structure, lightweight reasoning, explicit uncertainty, two perspective angles, and concise style.*
- **Structural Synthesis emerges between two systems**: Working across two independent systems exhibits a useful property: their discrepancies behave like a **differential field**, which highlights the **invariant** features of a system.
   - The lowest-entropy invariant between systems, the one that consistently survives bias-divergence, is called **Structural Synthesis**: *coherence, non-destructive integration, and increased mutual intelligibility.*
- **Stability Index benchmarks Models with no priming**: A stability summary from **15 baseline runs** (5 per model, no induction, no posture priming, 12 questions each) showed that **Claude** is the most neutral and shape-consistent across all systems with **9.7 / 10** while **GPT-5.1** self-organizes strongly **9.2 / 10** and **Gemini** exhibits noticeably higher entropy and less predictable response-shaping **6.8 / 10**.
   - The scores reflect averaged variation across the 5 runs, which include measures of structural crispness, tonal drift, response-shape variance, semantic inertia, coherence, neutrality and outliers.
- **High Level Sycophancy in Model benchmarks?**: A member criticized the lack of a rubric for the Stability Index benchmarks, calling it **high level sycophancy and low level rigor**.
   - They pointed out that *what is the defining factor that changes from 9.8 to 9.9* is not defined, nor is the criteria used to determine each number and why it wasn't 100% or 0%.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1446636114143350955)** (220 messages🔥🔥): 

> `Strategies for Reading Research Papers, Learning ML Efficiently, New Members, Interpretability with SGLang, ArXiv Endorsement` 


- **Strategies for Reading Research Papers**: A member suggests that the strategy to read papers and understand the materials and formulas presented depends on how much the work relies on the paper, and also mentioned the utility of learning things via a combination of **Anki flashcards and problem sets**.
   - It was noted that it can be a *beginner trap to treat learning materials as line items* rather than revisiting them; in this case, it's important not just to "read a paper" but to **annotate it, connect it with prior knowledge, and think through proposed ideas by writing them out**.
- **Newcomers Join EleutherAI Discord**: Several new members introduced themselves: one with a background in **East Asian Linguistics and Articulatory Phonetics**, working as a patent translator and pretending to be an ML researcher on a GTX 1650 Ti, and another semi-retired **AI professor** working on interpretability through SGLang with a [link to a Cognitive_workbench GitHub repo](https://github.com/bdambrosio/Cognitive_workbench.git).
   - Other new members include an implementation architect for **ServiceNow** who is obsessed with AI and a **Jr ML Engineer** and AI Researcher looking forward to collaborate.
- **Call for ArXiv Endorsement for Open Source Project**: A member is seeking an **arXiv endorsement** regarding their open source novel architecture, its accompanying paper with preliminary empirical results, and its released 18M model and linked the [GitHub repo](https://github.com/Mmorgan-ML/Neuromodulatory-Control-Networks).
   - Another member expressed interest in publishing papers and becoming an independent researcher in the future, working on universal meta modal language/model for AI alignment, information transmission fidelity and generally meta things.
- **Simulating Human Intelligence via Small Adapters**: A member shares that they are currently building an **LLM** that simulates human intelligence using small adapters with a **2B paligemma2** model as the backbone.
   - Another member is reportedly doing similar work from a different angle, conceptualizing a neurosymbolic recursively fractal meta language that can be used for alignment.
- **Describing Novel Architecture for Neuromodulatory Control Networks**: A member described their novel architecture as similar to a **hypernetwork**, but rather than generating weights, it modulates the temperature (technically precision), layer gain, and **FFN gating** on the fly for the larger network, and it achieved a validation perplexity of **4.5** after training on TinyStories dataset for only one epoch.
   - They also linked the [Github repo](https://github.com/Mmorgan-ML/Neuromodulatory-Control-Networks) and [the paper](https://github.com/Mmorgan-ML/Neuromodulatory-Control-Networks/blob/main/Neuromodulatory%20Control%20Networks%20(NCN).pdf) for convenience.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1446592930834940026)** (119 messages🔥🔥): 

> `Sinusoidal Init, Adam analysis, Generalization, Muon-trained Model, Video Generation` 


- **Sinusoidal Init numbers faked?**: A paper ([NeurIPS Poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202025/119048.png?t=1760975281.5828118)) on sinusoidal initialization had a **fake github link** and the numbers showed **106% AUC for Sinusoidal init**, while other inits have 100%.
   - Comparisons of the method revealed that it's better to construct the matrix by iterating on a random matrix; improvements to the method are not better than semi-orthogonal init.
- **Adam Analysis is BIG news**: A [theoretical result](https://arxiv.org/abs/2511.02773) on Adam connects **small hessian trace** to **low rank hessian** which is **low rank NTK** which is classically strongly associated with *good features* from a feature learning perspective.
   - Essentially, the model finds a nonlinear embedding of the data which has low rank structure while minimizing loss, implying a very concrete form of simplicity bias.
- **SV Infinity generates infinite videos**: [Stable Video Infinity](https://arxiv.org/abs/2510.09212) is a video generation tool that uses **Error Recycling** to generate infinite-length videos.
   - The [project's homepage](https://stable-video-infinity.github.io/homepage/) contains additional information.
- **Researchers debate on Quantum Machine Learning topics**: One member requested a good topic to work on in **quantum machine learning** using simulators like qiskit, to which another member linked a relevant [reddit thread](https://www.reddit.com/r/LLMPhysics/comments/1phk3xu/real_quantum_hardware_training_for_language/).
   - The reddit OP said there was a huggingface directory.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1447331141634756750)** (2 messages): 

> `Task Optimized KV Caches, Task Optimized LoRAs` 


- **KV Caches: Data or Algorithm?**: A member questioned whether task optimized KV caches are more akin to **data** or **algorithms**, sparking an interesting debate.
   - The discussion also pondered how they compare to **task optimized LoRAs** in terms of functionality and implementation, citing a [X post](https://x.com/withmartian/status/1997717765961253218).
- **Task Optimized LoRAs vs KV Caches**: Further discussion revolved around how task optimized KV caches compare to **task optimized LoRAs**.
   - The relative tradeoffs were not clear, but both seem to be useful in different contexts.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1447305444442902592)** (4 messages): 

> `Qwen3, anthropic` 


- **Qwen3 MGSM Result Reproducibility Debated**: Members are trying to reproduce the **MGSM** results of the **Qwen3** line of models.
   - The base is supposed to have **33%** accuracy, but members are unable to get close.
- **Anthropic Mapping Fix Proposed**: A member submitted a [PR to fix a broken mapping for anthropic](https://github.com/EleutherAI/lm-evaluation-harness/pull/3453).
   - The member said it *should be super easy to review and merge*.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1446592080330625045)** (17 messages🔥): 

> `Multiple Mac Studios, B200 latency, Moore Threads, ML Infra` 


- ****MLX** Framework Recommended for Multi-Mac Setups**: A member asked which inference framework is the best to use with multiple connected **Mac Studios**, and [MLX](https://github.com/ml-explore/mlx) was suggested.
   - MLX is an array framework for machine learning on Apple silicon, designed to be easy to use and efficient.
- **B200's **tcgen05.mma** Instruction Latency Discussed**: A member inquired about the instruction latencies of **tcgen05.mma** in the **B200**, citing [an article](https://arxiv.org/abs/2512.02189v1) indicating latencies of around **11 cycles** for various shapes.
   - Another member clarified that this isn't compute time but rather the time before the next MMA instruction can be issued, indicating a queue for **tcgen05.mma**.
- **Moore Threads Architecture: **MUSA** vs **CUDA****: Members expressed interest in **Moore Threads**, comparing their **MT GPU chips** and **MUSA** architecture to **CUDA**.
   - Further investigation may be needed to fully evaluate the performance and capabilities of **MUSA** against the established **CUDA** ecosystem.
- **Inquiry About ML Infra/Systems Roles**: A member inquired whether anyone works in the **ML Infra/Systems space**, seeking to confirm if they were in the right community.
   - Another member confirmed that this is a primary area of interest and asked the member about their specific questions as a career changer seeking to enter the field.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1446592315790463150)** (39 messages🔥): 

> `FP4, PTX 9.1, Async Sharp Operations, TileGym Autotuner, tcgen05.mma` 


- **FP4 Coming Soon!**: No **FP4** yet, but it's coming soon in **PTX 9.1**, according to an attached image showcasing **simd fp16x2** to **fp4x2**, **fp6x2**, **fp8x2**.
- **TileGym Auto-tunes CUDA**: Nvidia's **TileGym** includes an autotuner ([link](https://github.com/NVIDIA/TileGym/blob/main/src/tilegym/backend/cutile/autotuner.py)), raising questions about why this isn't shipped with core CUDA.
   - The choice to only support tiles/arrays and not use pointers seems great, at least from an ease of use and readability perspective.
- **tcgen05.mma data path layout**: For **2-SM tcgen05.mma**, each CTA provides half of A and half of B, with tensor core hardware flipping a bit to find data in the peer CTA based on offsets from the base SMEM address, see data path layout [here](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-a).
- **WGMMA won't work on Blackwell**: **WGMMA** instructions result in a compilation error on **Blackwell** because *WGMMA is sm90a only*.
   - A member shared a link to [cooking benchmarks](https://x.com/roeschinc/status/1997260105172340796) for **CUDA 13.1** against Triton kernels.
- **CUDA Learning Book Recommended**: "Programming Massively Parallel Processors A Hands-on Approach" is a recommended book for learning CUDA, located in its own channel.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1447634698582102097)** (1 messages): 

> `Symmetric Memory, CUDA error: an illegal memory access, Distributed training issues` 


- **Symmetric Memory Question surfaces**: A member inquired about symmetric memory and shared [a simple implementation of lb loss](https://gist.github.com/tohskai/72f33ed0d525a026ed37d78a2b6bbe3c) to illustrate their issue.
   - The code runs smoothly on a single node, yielding significant speed improvements, but encounters a `CUDA error: an illegal memory access` when scaled to two or more nodes.
- **`CUDA error: an illegal memory access` while using multiple nodes**: The user reported a `CUDA error: an illegal memory access` when running their code on two or more nodes, despite it working fine on a single node.
   - The error suggests a problem with memory access during distributed training, possibly related to how data is being handled across multiple GPUs or nodes.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1446932387475755302)** (2 messages): 

> `TTE-TPU, SGLang` 


- **TTE-TPU architecture surfaces**: A member posted a link to [considerthebulldog.com/tte-tpu/](https://considerthebulldog.com/tte-tpu/), discussing the new **TTE-TPU** architecture.
   - Another member mentioned that the link was from the **SGLang** folks.
- **RadixArk AI surfaces**: A member also linked to [radixark.ai](https://www.radixark.ai).
   - No discussion about RadixArk was performed.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1446953292570820688)** (3 messages): 

> `.NET Migration, PyTorch Operators, X.AI GPU Kernels` 


- ****.NET Necromancy**: Firms Raise the .NET from the Dead!**: A friend's tech company is seeking engineers to modernize their tech stack, specifically migrating a legacy repo from **.NET 4.8** to **.NET Core (.NET 9)**.
   - The role is a well-paid remote position at a mid-cap software company, offering flexibility in full-time or part-time arrangements, with interest in using AI to fast-track development.
- ****PyTorch Pirates**: Frameworks set sail for Operator treasure**: A company is hiring **PyTorch** experts to extend and customize the framework at the operator level, paying **$100-$160 / hr** for remote work; apply [here](https://work.mercor.com/jobs/list_AAABml0s7rpWxOxhkOFBoa5B?referralCode=36144a4a-07ca-462d-a68f-140b87c46767&utm_source=referral&utm_medium=share).
   - Ideal candidates will possess a deep understanding of **PyTorch's dispatch system**, **ATen**, **autograd mechanics**, and **C++ extension interfaces**, contributing to clear, maintainable operator definitions.
- ****Kernel Krusade**: X.AI seeks GPU gladiators!**: X.AI is hiring for its GPU kernel team, as posted [here](http://job-boards.greenhouse.io/xai/jobs/4427873007).
   - The team works closely with training/inference teams to optimize kernels across the stack, aiming for peak performance, and welcomes applicants with a strong portfolio, regardless of extensive experience.


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1447106454350860288)** (9 messages🔥): 

> `CUDA API Docs, Book 3rd vs 4th edition differences, Book purchasing difficulties` 


- **User seeks affordable physical copy of the book**: A user from India, finding the book's price too high, seeks advice on where to obtain a less expensive physical copy after downloading a PDF version, indicating they are *excited to get started*.
- **Differences between Book's 3rd and 4th Editions highlighted**: A user asks about the difference between the 3rd and 4th editions of the book, with another user pointing to the [preface online](https://www.sciencedirect.com/science/chapter/monograph/pii/B9780323912310000057) for a detailed answer.
- **User in China Struggles to Obtain 4th Edition**: A user in China seeks a PDF of the **4th edition** of the book, citing difficulties in ordering from **Amazon**.
   - Another user suggests checking local libraries (online), particularly larger university or provincial libraries, for access to digital collections.
- **CUDA API Docs should include links to functions**: A user suggests that the new **CUDA guide** should link directly to API documentation whenever it refers to an API function, suggesting that the API documentation benefit from linking back to the examples.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 messages): 

jaefosho: Very Eastern European
  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

szymonoz: I'll be in SF this week, anyone from the Bay up for a meetup?
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1446740712396951625)** (10 messages🔥): 

> `AMD, MI355X, Strix Halo, RDNA 3.5, Linux` 


- ****Strix Halo** is quirky but capable**: GPU MODE members reported prototyping kernels on **AMD’s Strix Halo laptop (RDNA 3.5, 128 GB RAM)**, praising **RGP** for profiling.
   - It acknowledges it lacks **FP8** and runs ~30× less memory bandwidth than an **MI355x**, making it a quirky but capable LLM dev box.
- **Dual booting for the best of both worlds**: One member uses dual boot with **Windows** and **Linux**. **Windows** is just for **RGP**.
   - Another is considering **Win11** with a **Linux VM** to get baremetal performance from both.
- **AMD Should Hand Out Free Hardware**: A member jokingly suggested that *AMD should hand out mi355x servers/GPUs/cloud credits in the next competition*.
   - This was in response to another member's suggestion that people should buy both a **Strix Halo** and **MI355X**.


  

---


### **GPU MODE ▷ #[tilelang](https://discord.com/channels/1189498204333543425/1240586843292958790/1446910827813933188)** (2 messages): 

> `GB300 CUDA Cores` 


- **How many CUDA cores in GB300?**: A member inquired about the number of **CUDA cores** in the **GB300**.
   - They seemed to suggest it should support a high number of cores, implying high expectations for its processing capabilities.
- **Discussion of GB300 specifications**: Discussion focused on anticipated specifications of the **GB300**, particularly the core count.
   - Community members eagerly await concrete details about its architecture and performance metrics.


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/)** (1 messages): 

jaefosho: How does one break into the ml infra space?
  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1446604563913113660)** (4 messages): 

> `CUDA 13.1, CUDA Tile, Distributed Training, QuintNet, 3D Parallelism` 


- **NVIDIA releases Biggest CUDA update Since 2006**: NVIDIA introduced **CUDA 13.1**, the biggest evolution of the **CUDA** platform since its debut in 2006, which includes **CUDA Tile**, a new programming model that simplifies how developers tap into the power of GPUs.
   - The release aims to simplify **GPU** programming and enhance accessibility to advanced **AI** and accelerated computing and the full deep dive can be found [here](https://developer.nvidia.com/cuda/tile).
- **CUDA Tile Unveiled**: **CUDA Tile** lets developers work in high-level “tiles” of data rather than managing thousands of low-level threads, as described in [this blog post](https://developer.nvidia.com/blog/nvidia-cuda-13-1-powers-next-gen-gpu-programming-with-nvidia-cuda-tile-and-performance-gains).
- **QuintNet Rises: A 3D Parallelism Prodigy**: A member announced the release of **QuintNet**, their own distributed training library built from scratch in **PyTorch** over the last 4 months, implementing full **3D Parallelism** (Data + Tensor + Pipeline) on a custom **GPU** mesh.
   - The library features a custom **DeviceMesh** implementation, manual **P2P** communication handling for pipeline stages, and custom **Column/RowParallelLinear** layers; more details are available in [this blog post](https://medium.com/@shuklashashankshekhar863/quintnet-a-3d-distributed-training-library-db0181a33a80) and [GitHub repo](https://github.com/Wodlfvllf/QuintNet).
- **NCCL Hangs Haunt Happy Hardware Hacker**: One dev recounted debugging silent **NCCL** hangs, chasing tensor shape mismatches, and realizing that implementing **1F1B** scheduling correctly is harder than the papers say.
   - *I finally got it to converge on a custom ViT across an 8-GPU mesh without deadlocking.*


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1447652429314261211)** (1 messages): 

> `Nvidia DRIVE AGX Thor, Kernel Optimization, Torch Models` 


- **Thor Kernel Optimization Quest Begins**: A member is seeking advice on optimizing kernels for **Nvidia's DRIVE AGX Thor** chip, aiming to improve the performance of their **Torch models** and reduce end-to-end latency.
   - They've consulted resources like **NeurIPS submissions** and **Unsloth's RL notebooks**, but found no specific guidance for this particular chip.
- **Need for Thor-Specific Optimization Guidance**: The user has explored various resources, including blog posts and research papers, in pursuit of kernel optimization strategies.
   - Despite their efforts, they haven't found specific guidance tailored to the **Nvidia DRIVE AGX Thor**, highlighting a gap in available optimization resources for this platform.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1447378503883296962)** (1 messages): 

> `Megakernel Implementation, Batched Llama Official, Instruction Generator Script, Blog Post Timings` 


- **Megakernel Implementation Explored**: A member inquired about the megakernel implementation with thunder kittens, specifically looking to test the batched version of **llama_official**.
   - They noticed the absence of an instruction generator script, unlike the non-batched version (**kvm_runner**).
- **Inquiry on Batched Llama Timings**: The user asked if the instruction generator script for the batched version was located elsewhere or if they were missing it in the codebase.
   - They inquired how the timings for the [blog post](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) were obtained, questioning whether the timings were from the non-batched version of **Llama**.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1446603900575551618)** (39 messages🔥): 

> `nvfp4_gemm Leaderboard Updates, sort_v2 Leaderboard Domination, prefixsum_v2 Leaderboard Sweep` 


- **NVIDIA's nvfp4_gemm Leaderboard Heats Up**: Multiple submissions updated the `nvfp4_gemm` leaderboard on NVIDIA, with one submission achieving **7th place** at **11.0 µs** and another securing **8th place** at **13.1 µs**.
   - Several members achieved *personal bests*, demonstrating ongoing optimization efforts.
- **Sweeping Victory on sort_v2 Leaderboard**: A member achieved **first place** on the `sort_v2` leaderboard across multiple NVIDIA GPUs: **B200 (2.27 ms)**, **H100 (2.09 ms)**, **A100 (3.97 ms)**, and **L4 (15.4 ms)**.
   - This marks a significant accomplishment in sorting performance across various hardware configurations.
- **prefixsum_v2: Conquering Prefix Sums**: The same member also secured **first place** on the `prefixsum_v2` leaderboard across multiple NVIDIA GPUs: **B200 (551 µs)**, **H100 (870 µs)**, **A100 (1385 µs)**, and **L4 (9.22 ms)**.
   - These victories highlight the member's proficiency in optimizing prefix sum operations for diverse GPU architectures.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1446616304474783754)** (6 messages): 

> `Factorio AI Development, Factorio-Learning-Environment Project, Moby 2.0 and CS Majors` 


- **Enthusiasm Surges for Factorio-Learning-Environment**: Multiple members expressed excitement after watching the **YouTube video** about the Factorio-Learning-Environment project.
   - One member exclaimed *"Very cool work, thanks for sharing!"* while another simply stated *"this is amazing!"*
- **CS Major eyes Factorio AI Dev**: A **3rd year computer science major** with extensive **Factorio** experience showed interest in contributing to the project.
   - They admitted being new to machine learning but eager to help, asking *"are there any ways I could be of use to this project?"*
- **Factorio Player Discovers Inner AI Developer**: A member joked about realizing they are already an advanced AI developer after watching the **Factorio video**.
   - This comment highlights the engaging and educational nature of the content.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1446603466662481973)** (1 messages): 

> `Image Analysis Achievements, Image Displayed` 


- **Image Analysis Achieved**: A member shared an image, calling it *an achievement I guess?*
   - The [attached image](https://cdn.discordapp.com/attachments/1394753097989099640/1446603466649768129/image.png?ex=69388a77&is=693738f7&hm=b463dde988d63a0bf0bd905408647011f0bcd9543e565a9f6b7f33666941ea50&) appears to depict a successful image analysis.
- **Image Displayed**: A user posted an image related to an achievement.
   - The user's message included the statement *An achievement I guess?* suggesting a possible accomplishment related to the image.


  

---


### **GPU MODE ▷ #[llmq](https://discord.com/channels/1189498204333543425/1421956177549332662/1446675522078900274)** (2 messages): 

> `pypi, Cross-entropy skipping, Chunked softmax calculation, CUDNN Workspace Chunking` 


- **Patience Pays: User Notices Long-Delayed Tag**: A user apologized for a month-long delay in noticing a tag, expressing gratitude for the work.
   - Another user responded *no worries. apparently, you looked just in time to actually see it now that I've put it on [pypi](https://pypi.org/)*.
- **Cross-Entropy Cut Saves VRAM**: A user's suggestion to skip the cut cross-entropy and focus on **chunked calculation of the softmax** proved useful, now enabling full fine-tuning of a **32B model on a 4x4090 workstation** with reasonable MFU.
   - The user noted, *I can now do full fine-tuning of a 32B model on a 4x4090 workstation while still getting reasonable mfu*.
- **CUDNN workspace becomes VRAM hog**: With logits gone, the **CUDNN workspace** (for deterministic attention backward) emerged as the largest memory consumer.
   - The user is now [chunking the CUDNN workspace](https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/), too.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1446631238101237890)** (54 messages🔥): 

> `CuTeDSL Talk, CuTile vs CuTeDSL, Modal and NCU, B200 Blockscaled GEMM, popcorn-cli Submissions` 


- **CuTeDSL Talk Scheduled with NVIDIA Engineer**: An NVIDIA engineer will give a talk on **CuTeDSL** ([YouTube link](https://www.youtube.com/watch?v=zHlz6mrdlZE)) at 3pm PST.
   - The talk aims to help those learning or starting on the competition; slides are now available on [GitHub](https://github.com/gpu-mode/lectures/blob/main/lecture_086/cute_dsl_introduce.pdf).
- **CuTile gaining traction as a simpler Alternative to CuTeDSL**: Some members noted that **cuTile** feels easier than **CuTeDSL**, abstracting away many things and is easier to get if you already know CUDA.
   - Others provided a link to a [matmul example on cutile Python](https://github.com/NVIDIA/cutile-python/blob/main/samples/MatMul.py), describing it as *very clean and high level*, though autotuning is not yet available.
- **Modal doesn't support NCU except for enterprise accounts**: Members discussed that **modal** does not support **NCU**, and that this feature is only accessible to enterprise accounts.
   - A member shared a [run_modal.py](https://cdn.discordapp.com/attachments/1446707526681755679/1446709167640543404/run_modal.py?ex=69384428&is=6936f2a8&hm=2a6ab01bf9f0041b3be579e0545bfa87ca98b76c489c9730f8096c4a82aa384b&) template for learning BF16 tcgen05.
- **B200 Blockscaled GEMM setup**: A member shared a detailed top-down analysis of the **Blockscaled GEMM** example from the **CuTeDSL** repo in a [blog post](https://veitner.bearblog.dev/b200-blockscaled-gemm-the-setup/) to lower the barrier for programming **B200** in **CuTeDSL**.
   - The analysis covers the calculation of the number of stages, layouts for shared memory, and configuration of MMA Ops.
- **Submitting with popcorn-cli doesn't update Discord Leaderboard**: A member asked why their test submission via **popcorn-cli** wasn't showing up on the Discord leaderboard.
   - Another member clarified that **popcorn-cli** submissions aren't linked to the Discord leaderboard, but users can check their status on gpumode.com or by using `/leaderboard show <name>`.


  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1446823561296740483)** (15 messages🔥): 

> `behavior-1k, RoboCOIN, mobile bimanual tasks, VLA for behavior-1k` 


- **Behavior-1k Excels in Mobile Bimanual Tasks**: Members find the [behavior-1k codebase](https://behavior.stanford.edu/) very good for **mobile bimanual tasks**, with the stack_blocks_two task working well.
   - Next steps involve extending training to similar tasks like **stack_blocks_three** and **sorting blocks** by color and size.
- **RoboCOIN Comparison Offers Good Datasets List**: A member shared a list by RoboCOIN, [RoboCOIN](https://flagopen.github.io/RoboCOIN/), as a pretty good comparison, in the context of an ever-growing list of bimanual robo datasets.
   - Another member was looking for **mobile bi manuals for loco manipulation tasks**, and expressed excitement about it.
- **Behavior-1k Demos are Low Quality, or so it seems**: A member noted that many of the **behavior-1k VR dataset demo videos** ([link](https://behavior.stanford.edu/behavior_100/demo_gallery.html)) seemed to be of low quality.
   - Another member countered, stating that they might be intended to be low-res, and that **behavior_100 is outdated**; the **1k version** is recommended.
- **What VLA to use for behavior-1k?**: One of the members asked what kind of **VLA** another member is using for **behavior-1k**, and inquired about the **action-space of the embodiment**.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1446607252680216616)** (168 messages🔥🔥): 

> `Pre-training sets ablations, Hermes 4.3 AWQ Quants, Consilience-40B Replacement, Multimodal RL Training, Video Game 3D Map Generation` 


- **Ablations on Pre-Training Sets**: A member is working on [pre-training sets with small ablations](https://psyche.network/runs), specifically those with the *"dm" prefix*.
   - The goal is to improve model performance by carefully adjusting aspects of the training data.
- **Hermes 4.3 AWQ Quants prove tricky**: A member is trying to run **Hermes 4.3** on 4x4090 using vllm but is having trouble with the **AWQ quants** by cyankiwi.
   - There is no FP8 version available right now, but it could be made using *neuralmagic*, and there are GGUFs available.
- **Consilience-40B Training Paused, MoE to replace it**: Training of **Consilience-40B** is paused permanently and a newer *Mixture of Experts* model will take its place.
   - The specifics of the new MoE model were not disclosed in the conversation.
- **LLMs + Vision: No Multimodal RL frameworks exist**: A member inquired about the limitations of **multimodal RL training**, noting that current models often rely on text descriptions generated by the vision tower rather than learning to reason visually.
   - There is theoretical support for vision environments in *Atropos* but not training; the architectures don't allow the LLM to work natively.
- **Dreaming of turning Video Games into 3D Maps**: A member asked if anyone had attempted to create **3D maps from video game recordings** using AI, similar to methods used for buildings.
   - The member is searching for a way to convert gameplay footage into a 3D environment using AI-based techniques.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1446658500599218296)** (7 messages): 

> `Hermes 4.3, Sonnet 4.5, Llama.cpp prompt template` 


- **Community Mulls Over Hermes 4.3**: Members in the channel are starting to evaluate **Hermes 4.3**, and discuss its merits.
   - One user mentioned wanting to try it out today.
- **Sonnet 4.5 Applauded as Top Anthropic Model**: The community is fascinated by how many individuals consider **Sonnet 4.5** as the best model from **Anthropic** following its release, with one member sharing [a YouTube Shorts video](https://www.youtube.com/shorts/U3WYW-qeEGE) on the topic.
   - The model is considered to be surprisingly good, especially for its size.
- **Llama.cpp Prompt Template Troubles Resolved**: A member sought guidance on how to pass the correct prompt template to **llama.cpp llama-cli** and described their initial attempt using a formatted prompt string, which yielded poor results.
   - Another member suggested utilizing the chat templates from the model repo, but the original poster found success by implementing a **Jinja** file instead.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

theguywhogamesalot: Reminds me of a bit of Nvidia's Method:

https://arxiv.org/abs/2510.01265
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1447299376362946591)** (15 messages🔥): 

> `Humble Bundle, O'Reilly Book Packages, Langchain` 


- **Humble Bundle Offers Deep Discounts on O'Reilly Books**: Humble Bundle offers heavily discounted packages including [O'Reilly books](https://www.humblebundle.com/books/machine-learning-ai-and-bots-oreilly-2025-books-encore), software, and games.
   - One user noted that the books focus on *application layer nailed to a method instead of learning how to make the decision*.
- **Langchain Book Faces Shredder**: One member joked he would shred the "Learning Langchain" book.
   - He stated he would *print it out, and put it through the shredder, just so I can say I shreddered it*.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

theguywhogamesalot: Reminds me of a bit of Nvidia's Method:

https://arxiv.org/abs/2510.01265
  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1446620465333538998)** (3 messages): 

> `YouTube Live Stream, Video Upload Delay` 


- **Livestream Scheduled**: A member announced they are *going live today*.
   - No further details were given about the stream's content or timing.
- **YouTube Upload Delayed**: A member indicated that a **YouTube video** should have been uploaded on Friday.
   - They provided a [link to the video](https://www.youtube.com/watch?v=dsslYZrVPbQ) and mentioned they would check on their end regarding the delay.


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1446621764745695253)** (4 messages): 

> `MAX framework, Model API, Mojo Meetup, MMMAudio, Shimmer` 


- ****Modular Meetup** Coming to Los Altos!**: Modular is hosting a special Modular Meetup on **December 11th** at their Los Altos office, with a livestream option for remote attendees; [reserve your spot here](https://luma.com/modularmeetup).
   - Attendees will hear **Chris Lattner** share the vision behind the **MAX framework**, learn about updates to their **Model API**, and connect with other developers and AI enthusiasts.
- **MAX Framework to be spotlighted**: At the meetup, you’ll hear Chris Lattner share the vision behind the **MAX framework**—a look at the future of **AI models in MAX**.
   - The MAX delivers high-performance, hardware-agnostic **AI inference on GPUs and CPUs**, supporting **500+ models**.
- **Model API Eager Semantics Explored**: The meetup will feature cutting-edge updates to Modular's **Model API**—including eager semantics in a pure **MAX/Mojo stack** with zero **PyTorch**, **NumPy**, or external framework dependencies.
   - The company is touting *zero dependencies* on external frameworks.
- ****MMMAudio** & **Shimmer** Demoed**: Catch up on Modular's latest community meeting which featured demos from **Sam Pluta** on **MMMAudio**, a creative-coding audio environment in Mojo, and from **Lukas Hermann** on **Shimmer**, his cross-platform Mojo → OpenGL experiment; full recording available [here](https://www.youtube.com/watch?v=dsslYZrVPbQ).
   - The Modular team also shared updates from the **25.7 release** and gave an early look at the **Mojo 1.0 roadmap**.
- **The Path to **Mojo 1.0** Blogged!**: For more on **Mojo 1.0**, check out Modular's latest blog post [here](https://www.modular.com/blog/the-path-to-mojo-1-0).
   - The blog post details key features and the roadmap for the language's first major release.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1446598467328937994)** (143 messages🔥🔥): 

> `Mojo compiler bugs, Lightbug status, Cutile relevance, MMMAudio presentation, ImplicitlyCopyable` 


- **Old Mojo Closures Cause Use-After-Free Issues**: An older closure system in Mojo had a hack that added invisible extra arguments, leading to use-after-free issues, but the new closure system in nightly builds fixed the problem by using `unified {read}` syntax.
   - The old closures were special-cased in the compiler, causing bugs, and the new system makes captures default to read-only unless specified otherwise, similar to lambda captures in C++.
- **Mojo Roadmap reveals Open Sourcing After 1.0**: Mojo is planning to open source shortly after the 1.0 release, with the compiler and compiler runtime expected to be open in Q2 2026, and Mojo 2.0 development will be out in the open.
   - Currently, a good chunk of Mojo is developed in the open via the standard library, offering sneak peeks at new features through nightly builds.
- **HTTP APIs in Mojo still far away**: There are still no networking or async capabilities in Mojo, and Lightbug is in maintenance mode, awaiting significant rewrites.
   - The compiler and runtime are expected to open up in Q2 2026, enabling more community involvement in these areas.
- **Mojo a potential host for TileIR**: Mojo is possibly the ideal host language for **TileIR**, even more so than **Python** and **C++**, and wrapping it into the Mojo compiler should be straightforward.
   - Mojo's `LayoutTensor` provides similar semantics to **CUDA's CuTile**.
- **DPDK for low-latency networking explained**: **DPDK** bypasses the kernel's network stack to reduce latency, avoiding unnecessary processes like error handling and protocol interpretations.
   - **DPDK** can coordinate more tightly with hardware, deliver packets directly to applications, and abstract hardware like cryptographic and DMA accelerators, with some NICs supporting it on Windows and macOS theoretically possible.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1447054827875078268)** (16 messages🔥): 

> `Bazel Integration, MAX API in Mojo, Heterogeneous CPU + GPU Graph Processing, Parametric traits and conditional conformance` 


- **Bazel Project Integration with MAX**: A user sought guidance on integrating **MAX** from a different **Bazel** project, specifically concerning the use of *rules_mojo* for handling the **Mojo** language aspect, posting a question on the [Modular Forum](https://forum.modular.com/t/using-max-from-another-bazel-project/2506?u=asa).
   - A Modular team member replied on the forum thread with some guiding lights.
- **MAX API Faces UX challenges in Mojo**: The team found that **Mojo** lacked the necessary language features to adequately express the **MAX API**, which prompted them to hold off for now.
   - One team member quipped, *“OpenCL has better UX than this”*, indicating significant usability issues with expressing such APIs in the current state of **Mojo**.
- **CPU + GPU Graph Processing Roadmap**: The ideal path for heterogeneous **CPU + GPU graph processing** with a **Mojo-native focus** is currently a **Python** story, as **Mojo** is still maturing.
   - The goal is for everything to be usable from **Mojo** eventually, but it will require some missing language features.
- **Parametric Traits and Conformance are key**: The biggest language blockers preventing a usable **MAX API** in **Mojo** are *Parametric Traits and Conditional Conformance*.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1446621825839927440)** (115 messages🔥🔥): 

> `Zero GPU spaces, building a server capable of running 120b oss gpt model, huggingface pro billing question, Binary convolutional neural network, LLM compliance ruleset` 


- **Zero GPU Spaces Wasting Slots**: A member noted wasting slots because they didn't notice that one or two **Zero GPU spaces** were still set to **Zero GPU** and were in a runtime error state (not Running).
   - They suggested checking the spaces settings to avoid similar issues.
- **Decoding server to run 120B model**: A member is planning to build a server capable of running **120b oss gpt model**, they noted that according to their research the best card for this is **RTX 6000 Blackwell** with its **96 gb vram**.
   - Another member mentioned that RTX 6000 Pro isn’t enough for GPT OSS, unless you quantize it to **Q4**, and quantize the KV cache to at least **Q8** if you want to use the full 128k ctx, adding that at q4, RTX pro runs gpt oss at 150-170 tokens per second.
- **Confused Member asks: Hugging Face Pro Billing?**: A member has a billing question regarding their **Hugging Face Pro subscription and Inference Provider usage**, having subscribed to HF Pro for **$9** and also made several inference API calls totaling around **$160**.
   - They removed their payment method afterward, and the dashboard said charges would be applied immediately but they haven’t been charged yet, and their Pro plan still shows active until Jan 1, 2026, reaching out to the community for help understanding what to expect next.
- **Lightweight Binary CNN Papers Sought**: A member is looking for **Q1/A* papers** recently published (from 2022 until now) about constructing a lightweight **BINARY convolutional neural network for the image classification problem**.
   - Another member suggested that it would be faster to just ask a generative AI.
- **Deciphering the LLM Compliance Ruleset**: A member shared that they process a couple hundred thousand hours of recordings on a daily basis through **WhisperX v3 Large** then push it through an LLM that refers to a set of compliance controls that compare rule sets.
   - Another member suggested trying **Parakeet v2** for faster offline speech transcription, especially for large batch sizes and similar if not better accuracy for English.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1447480094456025250)** (1 messages): 

> `Minimax M2, Claude Credits` 


- **Minimax M2 Rivals Claude?**: A member noted that **Minimax M2** can almost go head to head with **Claude**.
   - They prefer to *save Claude credits for the stuff they know is super hard* and then swap to **M2** for easier problems.
- **M2 Saves the Day**: One user suggests using **Minimax M2** for easy/medium difficulty tasks.
   - This strategy conserves **Claude** credits for more demanding challenges.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1446608645369364480)** (1 messages): 

> `HRM vs TRM models, LLM compute costs, LLM environmental impact, LLM Alternatives` 


- **LLMs face criticism for inefficiency**: A member expressed strong objections to Large Language Models (**LLMs**), deeming them inefficient and excessively expensive, citing **high compute costs** and significant **environmental impact**.
   - They argue that **LLMs** consume too much potable water, pollute the air, and contribute to global warming, while also driving up costs for storage, RAM, and GPUs.
- **HRM and TRM models offer alternative solution**: The member suggested **HRM** or **TRM models** (with approximately **27 million parameters**) as superior alternatives, claiming they outperform larger models on specific benchmarks.
   - They reference [HRM paper](https://arxiv.org/abs/2506.21734) and [TRM paper](https://arxiv.org/abs/2510.04871) as evidence.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1446864789337935902)** (18 messages🔥): 

> `Alignment Constitution Red-Teaming, AMD GPU Monitoring Tool, Offline Dictation macOS App, Hugging Face Spaces Dashboard, Graph Database Implementation` 


- **Alignment Constitution Passes Red-Teaming**: A random undergrad red-teamed a short alignment constitution (**LAW v1.3**) against fresh **Grok instances** and it survived **10** consecutive maximally adversarial rounds with no fatal flaw found, [available on Github](https://github.com/3377777/LAW-The-Guardian-Constitution).
- **`picomon` tool Monitors AMD GPUs**: A member created `picomon`, a tool to monitor **AMD GPUs**, trading off some accuracy with a lot more reliability than `nvtop` in certain scenarios, with the code [available on Github](https://github.com/omarkamali/picomon).
- **SilentKeys side project**: A member created **SilentKeys**, a side project for realtime offline dictation on **macOS** that types straight into any app, running locally without cloud components, with code [available on Github](https://github.com/gptguy/silentkeys).
- **DETERMINATOR multimodal deep research report writing**: A member shared a link to a multimodal deep research report writing **DETERMINATOR** on Hugging Face Spaces, calling it a simple implementation that *works great* [on HuggingFace](https://huggingface.co/spaces/DataQuests/DeepCritical).
- **Hugging Face Spaces dashboard**: A member built a tool for **Hugging Face Spaces** authors, with a link [on HuggingFace](https://huggingface.co/spaces/mrfakename/spaces-dashboard).


  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1447544208259944558)** (1 messages): 

> `Image generation, Release announcements` 


- **Image generation is out!**: A member announced the release of a generated image, sharing an attached [image](https://cdn.discordapp.com/attachments/1014557141132132392/1447544207949561969/generated_image.png?ex=6938aad9&is=69375959&hm=a0f44e2f32fe90ecb44ea5b4fd3bd1378f023e305273b39e2d4afbb3e7077008&).
- **Release Announcement Excitement**: The announcement was met with excitement, signaling a potential new feature or update.
   - Members were eager to explore the capabilities of the newly released image generation tool.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1447488038505680937)** (5 messages): 

> `object size detection, depth estimation models` 


- **Leverage HuggingFace for Depth Estimation Models**: A member sought to detect the size of objects in an image and was advised to explore [HuggingFace's depth estimation models](https://huggingface.co/models?pipeline_tag=depth-estimation&sort=trending).
   - The suggestion was to verify the correct implementation of depth estimation models and experiment with various models to achieve satisfactory accuracy.
- **Tackle Size Detection Accuracy**: The user, facing low accuracy with their current system, seeks help in improving the detection of object sizes in images.
   - Advice includes ensuring correct implementation of depth models and trying different ones to find a better fit for their use case.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1447340153361141821)** (1 messages): 

> `LORA, Fine tuning, Open Source, Metal, CUDA` 


- **Open Source LORA Fine Tuning Example!**: A member created an **MIT open source** example of how to fine tune your own models using **LORA** and **Python scripts** at [this Github link](https://github.com/orneryd/NornicDB/tree/main/neural).
- **Train on Metal and CUDA**: The code purportedly works to train on **Metal** and **CUDA**.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1446606210626228404)** (7 messages): 

> `Agent Course Certificate, GAIA Evaluation Agent Attachments, AI Agent Workshop` 


- **Agent Course Certificate Still Available?**: A member inquired whether they would still be able to get the certificate for the agent course after completing and submitting the final assignment.
   - They also asked how things have been going for others and if the course is worth it, being their first day.
- **GAIA Agent's Attachments Unavailable?**: Multiple members reported issues with the **GAIA evaluation agent**'s task attachments (images, audio files, Python code, Excel sheets, etc.) not being accessible via the **/files/{task_id}** endpoint, with the endpoint returning a *"No file path associated with task_id"* error.
   - Members requested confirmation from the **Hugging Face team** about whether these attachments were intentionally removed, and if there's a timeline for restoring them, as it's blocking full GAIA evaluation support.
- **AI Agent Workshop Incoming**: A member shared details about an **AI Agent 0-1 Workshop**, which serves as an introduction to their **AI Engineering Bootcamp**, and teaches participants to design and build an AI agent using **Langchain Agent** and **Streamlit**.
   - The workshop includes a [real client-style project](https://luma.com/aischolars) with live feedback and discount opportunities, with sessions scheduled for [Dec 13](https://luma.com/t4jcok99) and [Dec 16](https://luma.com/bdiwfvz5), making it ideal for job-seeking engineers and new AI builders.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1447026091725291560)** (59 messages🔥🔥): 

> `TOON adapter for DSPy, BAMLAdapter, GEPA optimizations, Compounding Engineering CLI, rec-praxis-rlm` 


- **TOON adapter gets DSPy Implementation**: An implementation of a **TOON adapter** for **DSPy** was created with accompanying benchmarks [here](https://github.com/Archelunch/dspy-toon).
   - While the adapter showed good **token count savings**, concerns were raised about its ability to handle **nested schemas** and **complex data** compared to **BAMLAdapter**.
- **GEPA Optimizations Boosted by TOON?**: Members conducted tests with **GEPA** on different adapters, including **TOON**, showing a potential **significant boost** in performance on **MMLU-Pro**.
   - The **optimization time** for **BAML** and **Chat adapters** was lower than **TOON** because the **TOON adapter** produced a much bigger system prompt, and sometimes it could run out of token limitation.
- **CLI Optimizes DSPy Prompt Context over Time**: A member built a **local-first engineering agent** using **DSPy** that implements the **"Compounding Engineering"** philosophy, where each unit of work should make the next one easier [here](https://github.com/Strategic-Automation/dspy-compounding-engineering#).
   - The agent uses a **modular architecture**, **knowledge injection**, and **auto-codification** to learn from its own execution history and optimize the context for future runs without finetuning.
- **rec-praxis-rlm Ships Procedural Memory and AI-Powered Security for Python**: A member shipped a Python package called **rec-praxis-rlm v0.9.2** that provides **AI agents** with **persistent procedural memory** and adds **zero-config security scanning** to your development workflow [pypi](https://pypi.org/project/rec-praxis-rlm/) [github](https://github.com/jmanhype/rec-praxis-rlm).
   - It features **procedural memory**, **security tooling**, **Claude Code hooks**, and **DSPy 3.0 integration**, with integrations for pre-commit hooks, GitHub Action, VS Code extension, interactive HTML reports, and SARIF support.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1446705705640792309)** (31 messages🔥): 

> `Vision Language Models optimization with DSPy, Gemini 3 Pro, Claude code DSPy harness, Context Compression Experiments` 


- **VLMs Get DSPy Boost**: A member inquired whether **DSPy** can be used to optimize **vision language models (VLMs)**, and the answer was yes, *if you can create a useful metric*.
   - Another pointed to the latest **Gemini 3 Pro blog post** as a reference: [Gemini 3 Pro blog post](https://blog.google/technology/developers/gemini-3-pro-vision/).
- **Claude Code gets DSPy Harness**: A member created a custom **DSPy harness** for **Claude code**: [DSPy harness for Claude code](https://www.modaic.dev/farouk1/claude-code), launching soon.
   - Another member noted that the harness supports anything you can do with the [Claude agent sdk](https://platform.claude.com/docs/en/agent-sdk/python).
- **TextGrad + GEPA > One**: Members recalled a blog post and **GitHub** repo where someone found that **TextGrad + GEPA** was better than either alone.
   - One member shared a link to a relevant project: [Context Compression Experiments](https://github.com/Laurian/context-compression-experiments-2508) and [associated tweet](https://x.com/i/status/1962953686348427347) and claimed *this will be the ultimate weapon to build any agentic stuff*.
- **gprcio causing build breakages**: A member noted that **grpcio** is causing build issues on python 3.14.
   - Another member mentioned that **grpcio** has been a dependency for about 8 months and suggested using `uv sync --python 3.14` on macOS, though another member mentioned that it stalls during the build process.  The project lead shared the ongoing work on [X](https://x.com/FaroukAdeleke3/status/1998147225533436073?s=20).


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1446616315535294674)** (67 messages🔥🔥): 

> `Control Theory, Free Markets, Lyapunov functions for NNs, Streaming Audio Transcription, Catastrophic Forgetting` 


- **Control Theory for Learned Models: A Tangled Web**: A member hoped to use **control theory** to analyze a system learned through NN, but another member noted that guarantees would be weak, as optimizing over a NN to find the optimal controller is not feasible.
   - They explained that **stability of nonlinear dynamical systems** involves finding **Lyapunov functions**, which is not really feasible for NNs.
- **Local Linearization Lowdown: Jacobian Jive**: Local linearization involves taking the **Jacobian matrix** of the dynamics every time step, expanding around current state and control to get a linear ODE.
   - The taylor expansion can be seen as `dx/dt = f(y*,u*) + df/dx|_{x*,u*} * (x-x*) +  df/du|_{x*,u*}(u-u*)` which is a linear ODE.
- **Free Market Fanfare: Regulation Required!**: A member discussed the downsides of *absolutely* free markets, while another argued a free market isn't the same as a lawless society.
   - The latter explained that the free market assumption is that prices are solely dictated by free supply and demand, which requires regulation to prevent degenerate conditions.
- **DeepSeek V3.2: Grading Like a Grandmaster!**: **DeepSeek V3.2** is a better grader than the humans who generated PRM800K.
   - No further information was given.
- **Streaming Audio Transcription Solutions Surface**: A member asked for the best way to do transcription of streaming audio, and another suggested [Whisper](https://openai.com/research/whisper), though there was disagreement if whisper takes a stream.
   - Another member linked to a [YouTube video](https://youtu.be/AThOsk2qJbs?si=CUdEKNezKN_q6jMA) and [Nvidia's multitalker-parakeet-streaming-0.6b-v1 on Hugging Face](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1) as possible solutions.


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1446883131188838444)** (2 messages): 

> `Discord Paper Discussions, Copilot Identification` 


- **User queries bot identity**: A user inquired whether the bot was **Copilot**.
   - The user then linked to a **Discord event** and questioned if the paper had already been discussed.
- **Repeated Paper Discussions**: A user linked to a **Discord event** and questioned if the paper had already been discussed.
   - This suggests concern about redundant discussions or ensuring all members are aware of past conversations.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1446842849030308001)** (20 messages🔥): 

> `Echo-TTS, Qwen3-TTS, Anthropic Interview, OpenAI's Stargate Project, DDR5 RAM kits` 


- **Echo-TTS Echoes into Existence**: A member shared a link to the **Echo-TTS** [GitHub repository](https://github.com/jordandare/echo-tts) and a [Hugging Face space](https://huggingface.co/spaces/jordand/echo-tts-preview) showcasing the project, along with a sample audio file.
   - The audio file, named *Echo-TTS_simulated_consciousness.wav*, hints at the project's capabilities in **simulated consciousness** through text-to-speech.
- **Qwen3-TTS Opens its Cloud**: **Qwen3-TTS** was released, but is only available through [Alibaba Cloud](https://qwen.ai/blog?id=qwen3-tts-1128), not open weights.
- **Theo's TTS Takes?**: Members discussed whether this [YouTube video](https://youtu.be/KAmQTmooLGQ) (with Theo) makes good points about **open weight models** being primarily Chinese.
   - One member summarized the video as *pointing out that all of the best open weight models are Chinese.*
- **Anthropic Interview: Not Gonna Happen**: A member deemed an Anthropic interview unlikely, given [this YouTube video](https://www.youtube.com/watch?v=6nJZopACRuQ).
   - The video refers to an OpenAI employee basically admitting that they're behind on pre-training, also noted [in this Tweet](https://x.com/petergostev/status/1995744289079656834).
- **OpenAI to hog all the DDR5**: OpenAI's **Stargate project** may consume up to **40%** of global DRAM output, with deals inked with **Samsung** and **SK Hynix** for up to **900,000 wafers per month** ([source](https://www.tomshardware.com/pc-components/dram/openais-stargate-project-to-consume-up-to-40-percent-of-global-dram-output-inks-deal-with-samsung-and-sk-hynix-to-the-tune-of-up-to-900-000-wafers-per-month)).
   - OpenAI employees are reportedly buying any DDR5 kit they can find, even impacting the gamer DDR5 RAM kits market ([source](https://www.notebookcheck.net/Not-even-gamer-DDR5-RAM-kits-are-safe-from-OpenAI-as-OpenAI-employees-are-allegedly-buying-any-DDR5-kit-they-can.1176107.0.html)).


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1446621312406781993)** (72 messages🔥🔥): 

> `Meta acquires Limitless, GPT-4o video generation, ARC Prize winners announced, Essential AI's open-source model, Google's Titans revisited` 


- **Meta morphs with Limitless Acquisition**: Meta acquired AI-wearables startup **Limitless** (formerly **Rewind**) on December 5, 2025; co-founder **Stammy** reflected on the journey from the **Rewind** launch in 2022 to the Pendant, as seen in [this tweet](https://x.com/Stammy/status/1997024785214460137) and [YouTube video](https://youtu.be/uuGTJzl1OVU).
   - Existing Pendant customers will receive another year of support and a free **Unlimited Plan**, though non-Pendant functionality like Rewind will be sunset.
- **GPT-4o Generates Viral Videos**: Aleksa Gordić released the notebook used to create recent viral demos of what appears to be "native GPT-4o video-generation," showcasing prompt-engineering tricks, as seen in [this tweet](https://x.com/gordic_aleksa/status/1997128393939472805?s=46&t=v6phN9scSJVJiuYdWBRQyQ).
- **ARC Prize Awards Refinement Loop**: The **ARC Prize** announced its 2025 winners: **NVARC** led at **25.03%** (Top Score), and **TRM’s** “Less is More” paper took 1st place (**$50k**), as seen in [this tweet](https://x.com/arcprize/status/1997010070585201068?s=46).
   - The **$600k Grand Prize** is still unclaimed, and all winning approaches are expected to be open-sourced.
- **Essential AI Enters with Mighty Open-Source**: Essential AI debuted their first open-source model, the **8-billion-parameter Rnj-1** (base & instruct), which outperformed larger models like **Gemini 2.0 Flash** on **SWE-bench Verified** (20.8% vs GPT-4o), as noted in [this tweet](https://x.com/essential_ai/status/1997123628765524132?s=46).
   - It is downloadable from Hugging Face under Essential AI’s open initiative.
- **AI scaling hits Global Energy Wall**: Unconventional AI warns that AI scaling will hit a global energy wall within **3-4 years**, advocating for brain-like hardware over digital simulations, as seen in [this tweet](https://x.com/unconvai/status/1998073266628366511?s=46).


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1446887040116330668)** (9 messages🔥): 

> `Nano Banana Pro, Prompt-to-Image Checklist, Contact-Sheet Prompting` 


- **David Duchovny AI Image Achieves Hyper-Specificity**: A user shared an **AI-generated** *“David Duchovny”* prompt loaded with **~20 hyper-specific visual details** (PSG kit, Parthenos sylvia, lenticular clouds, etc.) alongside a checklist graphic showing each item was actually included in the final image ([tweet](https://x.com/fofrAI/status/1997340753022828768?s=20)).
- **Contact-Sheet Prompting Workflow for Nano Banana Pro**: A user shared a detailed **contact-sheet prompting workflow** for **Nano Banana Pro** that produces cohesive **6-frame fashion editorials**, complete with camera positions, styling constraints, and Fuji Velvia flash aesthetic ([tweet](https://x.com/reflctwillie/status/1997819640874205685?s=46)).
   - Follow-ups include **Apple-exec satire shots**, **API vs UI model behavior notes**, and **Kling 2.6 first+last-frame video tips**, all offered without engagement-gate paywalls.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1446686829989462086)** (45 messages🔥): 

> `Black Friday Promotion, Kimi Slides Feature, Kaggle Competition, Username Length Limit, Kimi Markdown Issues` 


- **Black Friday ends early**: A user questioned whether the **Black Friday promotion** had ended early because the subscribe button was no longer available, despite the terms indicating validity until **December 12th**.
- **Kimi's Markdown mishap**: A user shared an image of Kimi's broken markdown, joking that it was happening for the millionth time.
   - Another user requested that they make a bug report in the appropriate channel.
- **User gets blocked for innocuous username**: A user reported being blocked by another user due to a conflicting political viewpoint.
- **Groq gets called out on quality**: A user asked about the best provider for Kimi, citing **Groq's** subpar outputs.
   - Another user recommended the official **Moonshot API**.
- **Kimi's website has issues**: A user shared an image of Kimi's website not functioning, where they were only able to click **New Chat**.
   - Another user suggested trying to fix this by clearing **cookies**, disabling **VPNs**, and disabling **adblockers**.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1447125276084408361)** (29 messages🔥): 

> `Manus Support Issues, Google Play Billing Bugs, Account Upgrade Problems, Credit Refund Requests, Understaffed Support Team` 


- **Users Report Issues with Manus Credits and Google Play Billing**: Multiple users reported issues with **Manus credits** not being correctly applied after topping up via **Google Play**, leading to frustration and requests for refunds.
   - One user spent **$900+** and only received **1500 credits**, expressing that they feel scammed due to the lack of adequate support.
- **Subscription Bugs Plague Users After Free Trial**: Several users mentioned a bug related to upgrading their accounts after receiving a free trial of **Manus Pro** and reported unwanted behavior relating to subscriptions.
   - One user had their renewal date pushed to **May 2026** and was marked as being on a free trial again.
- **Manus Support Team Allegedly Understaffed and Overwhelmed**: Users speculate that the **Manus support team** is understaffed and overwhelmed, resulting in generic, templated responses and unresolved issues.
   - A member suggested that the company needs *a good change management team and a more robust customer service desk operationalized to handle the issues/volume*.
- **Engineer Faces Roadblock on Checkpoint Restoration**: A user reported facing a critical issue while attempting to restore the **checkpoint** of their **webdev project**.
   - The user then asked where can open a ticket, but found only chat.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1446652417365245982)** (7 messages): 

> `Gemini CLI OAuth, Claude Opus 4.5, aider + Amazon bedrock` 


- **User Seeks Gemini CLI OAuth Integration in Aider**: A new user, shifting from other tools due to **TypeScript** issues, inquires about integrating **Gemini CLI OAuth** with aider to use **Gemini models**.
   - The user expresses satisfaction with aider's performance on their **C# project**, praising its ease of use in file creation and modifications.
- **Aider Compatibility with Claude Opus 4.5 Explored**: A user asks if aider is compatible with **Claude Opus 4.5**, noting they currently use **Claude Code** with a max plan and rarely hit limits.
   - They are curious about the differences between their current setup and using **aider**.
- **Aider Plays Well with Amazon Bedrock and Claude Opus**: One user reports they use **Opus** with **Amazon Bedrock** and **aider** without issue.
   - They claim that *all is good*.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/)** (1 messages): 

ethan_15839: Is there any way to use the Gemini CLI oAuth in aider to use the gemini models?
  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1446612742353195148)** (5 messages): 

> `USB 2.0 Driver Support, Meeting #99 Agenda, asm2464pd-firmware` 


- **USB 2.0 Driver Support Possible**: Members suggest that USB 2 support *might* be feasible with driver adjustments, although performance would be slow.
   - The device descriptor indicates that [full speed (12Mbps) would be supported](https://developer.usb.org/).
- **Meeting #99 Agenda Announced**: Meeting #99 is scheduled for **Monday at 9am San Diego time**, with the agenda including *company updates*, *training loop*, and *llama 8B*.
   - Other topics that are part of the agenda include *flash attention*, *VIZ/Profiling*, *drivers*, *MESA backend*, and *other bounties*.
- **asm2464pd-firmware GitHub Repo**: A link to the [asm2464pd-firmware GitHub repository](https://github.com/geohot/asm2464pd-firmware) was shared.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1447711769366630451)** (1 messages): 

> `AI Agent Workshop, AI Engineering Bootcamp, GenAI for Beginners` 


- **Build AI Agents from Scratch in Workshop**: The AI Agent 0–1 Workshop offers an intro to an **AI Engineering Bootcamp**, where participants design and build an AI agent that thinks, codes, analyzes data & generates reports, for a previous real client — all from scratch.
   - The workshop includes a real consulting client project, live roast & feedback, and intro to a **7-week AI Engineering Consulting Project Bootcamp** based on Microsoft’s [“GenAI for Beginners”](https://microsoft.github.io/Generative-AI-For-Beginners/).
- **Bootcamp Discounts up for Grabs**: The next cohort of the bootcamp will be in Jan 2026, and top builders at the workshop get **Bootcamp discounts**.
   - Interested participants can RSVP for Saturday December 13th, 2pm ET at [this link](https://luma.com/t4jcok99), or Tuesday, December 16, 8pm ET: [here](https://luma.com/bdiwfvz5), or other times [here](https://luma.com/aischolars).


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/)** (1 messages): 

paoloricciuti: Is there anywhere I can ask for this info to get more answers? 😅
  

---


---

