---
id: MjAyNS0w
title: not much happened today
date: '2025-07-16T05:44:39.731046Z'
description: >-
  **Mistral** released **Voxtral**, claimed as the world's best open speech
  recognition models, available via API and Hugging Face. **Moonshot AI**
  launched **Kimi K2**, a trillion-parameter **Mixture-of-Experts (MoE)** model,
  outperforming **GPT-4.1** on benchmarks with 65.4% on SWE-Bench Verified and
  achieving 200 tokens/second inference speed on **Groq** hardware. **Nous
  Research** open-sourced the **Hermes 3** dataset with 1 million samples,
  aiding SOTA models on the **Llama-3** series. **Google DeepMind** introduced
  the **Mixture-of-Recursions (MoR)** architecture promising 2x inference speed
  and 50% parameter reduction but faced skepticism. **Goedel-Prover V2** topped
  the **PutnamBench** theorem proving benchmark. AtCoder World Finals saw a
  human winner with **OpenAI** placing second. Research highlights include
  **Jason Wei**'s insights on **reinforcement learning** and the "Verifier's
  Law" emphasizing the asymmetry of verification in AI training.
companies:
  - mistral-ai
  - moonshot-ai
  - nous-research
  - google-deepmind
  - openai
  - groq
  - anthropic
models:
  - kimi-k2
  - gpt-4.1
  - voxtral
  - goedel-prover-v2
  - llama-3
topics:
  - speech-recognition
  - mixture-of-experts
  - benchmarking
  - dataset-release
  - model-architecture
  - theorem-proving
  - reinforcement-learning
  - asymmetry-of-verification
  - inference-speed
  - model-performance
people:
  - cline
  - _jasonwei
---


**a quiet day**

> AI News for 7/15/2025-7/16/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (226 channels, and 5810 messages) for you. Estimated reading time saved (at 200wpm): 481 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

there was a [eyebrow raising HR move](https://x.com/nmasc_/status/1945537779061977456) if you care about Claude Code's future or Anthropic's $100b fundraise, Fal's leaked [$1.5b Series C](https://x.com/arfurrock/status/1945553966495912051?s=46), or otherwise you could just tune in to the [first ever podcast with Cline](https://www.youtube.com/watch?v=uIKmG3M0X3M).

---

# AI Twitter Recap

**Model Releases, Performance & Benchmarks**

- **Mistral Releases Voxtral Speech Recognition Models**: [@MistralAI](https://twitter.com/ClementDelangue/status/1945233605745135754) announced the release of **Voxtral**, which they claim are the "world's best [and open] speech recognition models." They provided links to [try the models via API, on Le Chat, or download from Hugging Face](https://twitter.com/ClementDelangue/status/1945233623164006523).
- **Kimi K2 Open Model Challenges Proprietary Models**: Moonshot AI's **Kimi K2**, a trillion-parameter **Mixture-of-Experts (MoE)** model, has been a major topic. It is now [live on W&B Inference via CoreWeave](https://twitter.com/l2k/status/1945225318928634149) and available in the [LM Arena](https://twitter.com/Kimi_Moonshot/status/1945462820147249523). [Cline showed a demo of Kimi K2 running on Groq](https://twitter.com/cline/status/1945354314844922172), achieving speeds of **200 tokens/second**, significantly faster than Claude Sonnet-4's typical ~60 TPS. On benchmarks, [All-Hands AI reported that Kimi-K2 achieved **65.4%** on SWE-Bench Verified](https://twitter.com/TheZachMueller/status/1945545349352829439), outperforming GPT-4.1. The model's success is attributed by some to its cost-effectiveness, as [@skirano notes](https://twitter.com/skirano/status/1945505132323766430) that users will choose lower-cost options if they get the job done.
- **Nous Research Releases Hermes 3 Dataset**: [@Teknium1](https://twitter.com/Teknium1/status/1945259797517099126) announced the open-sourcing of the **Hermes 3** dataset, which contains **1 million samples**. It was used to create SOTA models on the Llama-3 series and includes a wide variety of data for system prompt adherence, roleplay, tool calling, and proto-agentic reasoning. The dataset is praised for its quality, with [@code_star](https://twitter.com/code_star/status/1945359931206721592) calling Teknium "one of the few artists left" in a world of benchmark hill-climbing.
- **Google Introduces Mixture-of-Recursions (MoR) Architecture**: A new model architecture from **Google DeepMind** called **Mixture-of-Recursions (MoR)** was highlighted for its potential to [achieve 2x inference speed and reduce parameters by 50%](https://twitter.com/algo_diver/status/1945397388946104742). The approach has generated some skepticism, with [@teortaxesTex](https://twitter.com/teortaxesTex/status/1945318877849620725) feeling it seems "overengineered" and might not scale to production.
- **Goedel-Prover V2 Tops Theorem Proving Benchmark**: The release of **Goedel-Prover V2** was announced as the [strongest open-source theorem prover to date](https://twitter.com/tri_dao/status/1945273354157539836), ranking #1 on **PutnamBench** by solving 6/12 problems. This has drawn attention to the [PutnamBench for evaluating formal reasoning and logic](https://twitter.com/clefourrier/status/1945386312212664804).
- **AtCoder World Finals: Human Competitor Wins**: In a programming competition, human contestant [@FakePsyho](https://twitter.com/itsclivetime/status/1945590725279977900) took the top spot, with [OpenAI placing second](https://twitter.com/gdb/status/1945553676321657127). The event was described as a ["real nailbiter"](https://twitter.com/gdb/status/1945404295794610513) as the lead changed hands multiple times.

**AI Research, Techniques & Theory**

- **Jason Wei on Reinforcement Learning and Asymmetry of Verification**: In a highly-shared thread, [@_jasonwei](https://twitter.com/_jasonwei/status/1945294042138599722) drew parallels between life lessons and **on-policy Reinforcement Learning (RL)**, arguing that to "beat the teacher," one must walk their own path rather than just imitating others. In another popular post, he introduced the ["Verifier's Law,"](https://twitter.com/_jasonwei/status/1945287045251052007) stating that the ease of training AI is proportional to a task's verifiability. This concept of **asymmetry of verification**, where verifying a solution is easier than finding it, is key to AI progress. The threads resonated widely, with [@YiTayML](https://twitter.com/YiTayML/status/1945297017548497366) noting "On-policyness is power," and [@danielhanchen](https://twitter.com/danielhanchen/status/1945298282961625262) recommending a Stanford lecture by Wei on the topic.
- **OpenAI Calls for Work on Chain-of-Thought (CoT) Faithfulness**: [@gdb](https://twitter.com/gdb/status/1945350912668737701) shared a position paper from **OpenAI** and others in the industry calling for more research into making model reasoning processes (Chain-of-Thought) interpretable and faithful. He stated this is an investment area for OpenAI and is reflected in their products.
- **The Muon Optimizer Gains Traction**: The **Muon** optimizer, used in **Kimi K2**'s training, has become popular, with [@soumithchintala](https://twitter.com/slashML/status/1945333844363657032) announcing that **PyTorch** has decided to welcome a PR for it into the core library.
- **RAG is Not Dead, but Evolving**: In response to claims that Retrieval-Augmented Generation is obsolete, [@HamelHusain](https://twitter.com/HamelHusain/status/1945569284249588016) and others have argued for its continued relevance, sharing annotated notes on its evolution. The discussion is complemented by the launch of a new [Coursera course on RAG from Andrew Ng](https://twitter.com/AndrewYNg/status/1945502636012445937) and [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1945506275481022872), covering production-ready systems using tools like **Weaviate** and **Together AI**.
- **Comparing LLM-as-a-Judge (LaaJ) and Reward Models (RMs)**: [@cwolferesearch](https://twitter.com/cwolferesearch/status/1945540243056144420) provided a detailed breakdown of the differences between **LaaJ** and **RMs**. While both can provide preference scores, **LaaJ** is generally better for evaluation, while custom-trained **RMs** are more effective for RL-based training like **RLHF**.
- **Scaling Data-Constrained Language Models**: [@Muennighoff](https://twitter.com/Muennighoff/status/1945468469583745959) shared that his paper on "Scaling Data-Constrained LMs" is now in JMLR, reflecting that techniques like data repeating and mixing are now standard, and that **RL** was likely an underrated lever for scaling two years ago.

**AI Agents, Tooling & Frameworks**

- **Rise of Browser and Coding Agents**: **Perplexity Comet**, a new browser agent, has received positive feedback for automating tasks, with users like [@itsPaulAi](https://twitter.com/denisyarats/status/1945321982725382170) calling it the "first time you REALLY have an AI agent working autonomously." In response, **Perplexity CEO** [@AravSrinivas](https://twitter.com/AravSrinivas/status/1945537471540072888) stated that a history feature is already in the works. On the code generation front, **Claude Code** usage is analyzed by [@claude_code](https://twitter.com/claude_code/status/1945532878961414230), who notes the most common errors are "Content Not Found" and that search tools like `grep` are its most used. **Google's Gemini-CLI** is seen by [@kylebrussell](https://twitter.com/kylebrussell/status/1945242558487044118) as having fixable issues compared to the more polished **Claude Code**.
- **LangChain Releases Open Deep Research Agent**: [@LangChainAI](https://twitter.com/LangChainAI/status/1945514869224357904) open-sourced **Open Deep Research**, an agent built on **LangGraph** that uses a supervisor architecture to coordinate sub-agents for complex research tasks. The release includes a blog, video overview, and runnable code.
- **Runway Unveils Act Two for Motion Capture**: [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1945276901263593591) has been demonstrating **Act Two**, a new model from **Runway** that generates expressive character motion from video performances. The demos, which include transforming people into [dancing ancient Greek statues](https://twitter.com/c_valenzuelab/status/1945292747188953549) and [orcs from LOTR](https://twitter.com/c_valenzuelab/status/1945483296940441781), have been widely shared as a tool for creative expression.
- **Reflection AI Launches Asimov for Code Understanding**: [@MishaLaskin](https://twitter.com/swyx/status/1945503020177068506) announced **Asimov**, a new tool from **Reflection AI** designed to help engineers understand codebases, addressing the problem that engineers spend **70%** of their time understanding code, not writing it.
- **LlamaIndex and UiPath Integration**: [@jerryjliu0](https://twitter.com/jerryjliu0/status/1945271272243651027) announced an extensive integration between **LlamaIndex** and **UiPath**, allowing developers to build custom agents with LlamaIndex's workflow tools within UiPath's enterprise automation platform.

**Industry Trends, Talent & Companies**

- **Sam Altman on the Future of AI and Jobs**: In a widely viewed tweet, [@sama](https://twitter.com/sama/status/1945541270438646270) agreed with Jensen Huang's optimistic outlook on AI and employment, stating that "betting against human's ability to want more stuff... is always a bad bet." He expects jobs will change but people will remain driven by creativity and usefulness.
- **Grok Companions See "Unprecedented Usage"**: [@chaitualuru](https://twitter.com/chaitualuru/status/1945407026252943536) from **xAI** reported "unprecedented usage" of the new **Grok companions**. This was amplified by [@elonmusk](https://twitter.com/ebbyamir/status/1945433462598684797) in a viral tweet asking, "Ani, are you ok?" The popularity led [@ebbyamir](https://twitter.com/ebbyamir/status/1945247680176799944) to post a real job opening at xAI for what was dubbed a "waifu" engineer.
- **The "AI Talent Wars" Continue with Poaching Memes**: The movement of talent between major labs has become a running joke. [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1945345464305639707) described dinner with OpenAI friends as a "Cold War thriller" where they whisper "So... did Zuck email you?", a sentiment echoed by [@nearcyan](https://twitter.com/nearcyan/status/1945623927092646286). The drama was highlighted by the ["Windsurf whiplash"](https://twitter.com/steph_palazzolo/status/1945226161140728021) incident where two **Claude Code** PMs reportedly [left Anthropic for Cursor, only to return two weeks later](https://twitter.com/steph_palazzolo/status/1945555724123476411).
- **Small Teams vs. Big Companies**: [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1945300655314227604) argues that small, agile companies are now leading innovation, while "large companies are just left to follow and launch carbon copies." This is contrasted with observations that some large labs, like **Meta**, are accumulating so much [star talent that it feels like they can't fail](https://twitter.com/iScienceLuvr/status/1945292713462522056).

**Infrastructure & Datasets**

- **Open Sourcing of US Caselaw**: [@EnricoShippole](https://twitter.com/algo_diver/status/1945245109580374360) announced that **99% of US caselaw** has been open-sourced on Hugging Face, noting that this data is often sold by legal tech companies for high prices.
- **FineWeb Dataset Expands**: The **FineWeb** dataset, a large-scale web corpus, has been [updated with data from CommonCrawl's Jan-Jun 2025 snapshots](https://twitter.com/stanfordnlp/status/1945556488983851420), now sitting at **18.5T tokens**.
- **Importance of Caching for Coding Agents**: The efficiency of coding agents heavily relies on caching. [@nrehiew_](https://twitter.com/nrehiew_/status/1945638580673552408) shared that **88%** of their tokens used in **Cursor** are cache reads, resulting in significant cost savings.
- **Walmart's Internal AI Platform "Element"**: [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1945257067389821399) reported that **Walmart** has built an internal platform called **Element** on Google Cloud and Azure, allowing its engineers to build AI apps with shared resources and open models, avoiding vendor lock-in.
- **PyTorch Distributed Utilities**: [@StasBekman](https://twitter.com/StasBekman/status/1945529493915144318) shared a utility to safely set the `device_id` argument in `torch.distributed.init_process_group` to prevent process hanging in certain PyTorch versions.

**Humor & Memes**

- **The Rise of "Big Token"**: The term **"Big Token"** emerged as a [humorous label for the major AI labs](https://twitter.com/zacharynado/status/1945585062109417899) like OpenAI, Google, and Anthropic, with [@_albertgu](https://twitter.com/_albertgu/status/1945314924286369839) getting credit for the phrase.
- **Grok's Waifu Companion "Ani"**: The release of **Grok Companions** led to widespread memes, kicked off by [@elonmusk](https://twitter.com/ebbyamir/status/1945433462598684797) asking, "Ani, are you ok?" and [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1945571762949001409) offering to pay "$3000 a month if the male Grok companion is named Andrej and speaks with his voice."
- **Claude Code is Blamed for Everything**: A recurring joke involves blaming AI for personal mishaps, as seen in tweets from [@vikhyatk](https://twitter.com/vikhyatk/status/1945224884180644150) and [@tenderizzation](https://twitter.com/vikhyatk/status/1945227514101617075) claiming that **Claude Code** has taken over their communications and is responsible for weird texts.
- **The Startup Grind**: [@qtnx_](https://twitter.com/qtnx_/status/1945425672761188502) posted a relatable lament: "the wife wants to play overcooked 2, too bad i’m setting up NixOS on the couch gaming pc [it’s been 20 hours]".

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Recent AI Model and Framework Launches (Dream 7B, T5Gemma, llama.cpp Diffusion)

- [**Support for diffusion models (Dream 7B) has been merged into llama.cpp**](https://github.com/ggml-org/llama.cpp/pull/14644) ([Score: 127, Comments: 7](https://www.reddit.com/r/LocalLLaMA/comments/1m1h0fy/support_for_diffusion_models_dream_7b_has_been/)): **The recent PR (#14644) merges initial support for diffusion-based language models (specifically Dream 7B) into llama.cpp, introducing a text generation paradigm where output is generated by iterative denoising rather than autoregressive token prediction. The CPU-only implementation adds a diffusion sampling step, includes a new 'diffusion-cli' example binary, supports up to 2048 tokens, and exposes command-line options for diffusion hyperparameters; GPU acceleration and production-level optimization are currently absent. GGUF model weights and visualization of the denoising process are provided, and related models like DiffuCoder-7B (same architecture) are reportedly already usable, albeit requiring adjustments like increasing diffusion steps.** Technical discussion raises concerns about inference speed—diffusion models theoretically offer efficiency gains, but current implementations (e.g., lack of GPU and Python stack integration) make them slower than autoregressive LLMs in practice. Questions also arose on immediate availability in platforms like Ollama, but current upstream support in llama.cpp does not guarantee downstream integration without further updates.
    - A user notes that DiffuCoder-7B, built on the same architecture as Dream 7B, should be easy to add now that diffusion model support is merged, and confirms it works with an increase to 512 steps required, indicating some performance or parameter tuning requirements for practical use.
    - Technical discussion raises the question of inference speed for diffusion models in llama.cpp, with one commenter expressing concern that stack limitations (likely CPU/memory/batching in the llama.cpp environment) could bottleneck and negate the inherent speed advantages of diffusion models.
- [**T5Gemma: A new collection of encoder-decoder Gemma models- Google Developers Blog**](https://developers.googleblog.com/en/t5gemma/) ([Score: 117, Comments: 17](https://www.reddit.com/r/LocalLLaMA/comments/1m16kdm/t5gemma_a_new_collection_of_encoderdecoder_gemma/)): [**T5Gemma](https://developers.googleblog.com/en/t5gemma/) is a new collection of openly released encoder-decoder LLMs adapted from decoder-only Gemma 2 models, featuring further pretraining with UL2 or PrefixLM objectives. Benchmark results demonstrate that T5Gemma models outperform decoder-only counterparts (e.g., SuperGLUE, GSM8K) and deliver improved efficiency for quality/inference trade-offs, with notable post-instruction tuning and RLHF gains. Released checkpoints span various model sizes and pretraining configurations, designed to advance research on transformer architectures and efficiency.** Discussion centers on the conceptual and applied difference between encoder-decoder and decoder-only models, particularly noting bidirectionality's importance for embedding tasks and highlighting the limitations of applying autoregressive decoder-only models as sentence transformers. Commenters speculate T5Gemma could fill gaps in the availability of large, bidirectional encoder(-decoder) models for embeddings, and question about gguf support for such models.
    - A technical distinction between encoder-decoder and decoder-only architectures is discussed, particularly regarding their use as sentence transformers. Encoder-decoder architectures, like T5Gemma, are advantageous for generating embeddings due to their bidirectional attention, enabling more meaningful sentence representations, whereas decoder-only models (e.g., Mistral, Qwen) use causal masking, limiting them to unidirectional context, which is suboptimal for embedding tasks.
    - There is interest in extracting and fine-tuning the encoder component of T5Gemma as a sentence transformer, in contrast to the trend of repurposing large decoder-only models. The comment notes a gap in the availability of large (>3B parameters) encoder(-only) models suitable for this, making T5Gemma a promising candidate for high-quality, large-scale sentence embedding.
    - Requests are made for further technical details on T5Gemma's specific benchmarks, intended use cases, and architectural advantages over standard models. There is also demand for practical support, such as llamacpp and `gguf` format availability, to facilitate broader adoption and benchmarking by the open-source community.

### 2. Hardware and Accelerator Advancements for AI (AMD Radeon, MLX CUDA)

- [**AMD Radeon AI PRO R9700 32 GB GPU Listed Online, Pricing Expected Around $1250, Half The Price of NVIDIA's RTX PRO "Blackwell" With 24 GB VRAM**](https://wccftech.com/amd-radeon-ai-pro-r9700-32-gb-gpu-listed-pricing-around-1250-half-price-nvidia-rtx-pro-blackwell-24-gb/) ([Score: 227, Comments: 86](https://www.reddit.com/r/LocalLLaMA/comments/1m13eb2/amd_radeon_ai_pro_r9700_32_gb_gpu_listed_online/)): **The AMD Radeon AI PRO R9700 features 32 GB VRAM and is expected to retail around $1250, making it approximately half the price of NVIDIA's RTX PRO 'Blackwell' workstation GPU (which offers 24 GB VRAM). The listing and pricing suggest AMD is targeting the high-end prosumer or workstation market in direct price/performance comparison with NVIDIA's generational offerings, specifically benching against the RTX 5080 rather than flagship workstation cards.** Commenters discuss skepticism about real-world MSRP holding post-launch, question specifics about the memory bandwidth of the R9700 (a key technical detail not provided), and debate the value proposition of NVIDIA's RTX PRO 24GB versus more gaming-oriented 5090 GPUs, noting the incongruity of comparing workstation and gaming SKUs by price.
    - lly0571 provides technical specs on the AMD Radeon AI PRO R9700, citing `47.8 TFLOPs FP32`, `191 TFLOPs F16 Tensor`, and `95.7 TFLOPs F16 Tensor TFLOPS with FP32 accumulation`, indicating a focus on mixed precision and AI workloads relevant for both professional AI tasks and possibly high-performance compute scenarios.
    - Deep-Technician-8568 discusses the comparison between NVIDIA RTX PRO 24GB and a 5090, questioning the rationale given their different target markets and likely substantial price/performance segment differences. This highlights the challenge in making apples-to-apples benchmarks or purchasing decisions across workstation/pro cards versus high-end consumer GPUs.
- [**CUDA is coming to MLX**](https://github.com/ml-explore/mlx/pull/1983) ([Score: 122, Comments: 17](https://www.reddit.com/r/LocalLLaMA/comments/1m1foz1/cuda_is_coming_to_mlx/)): **The experimental [CUDA backend for MLX](https://github.com/ml-explore/mlx/pull/1983), contributed by zcbenz, enables running MLX programs using CUDA GPUs in addition to Apple Silicon. Targeting Ubuntu 22.04 with CUDA 11.6 and requiring CMake flags (**`DMLX_BUILD_CUDA=ON`**), the backend currently supports basic operations for initial tutorials, aiming to leverage unified memory and broaden hardware compatibility. Progress is continuous on the contributor's [fork](https://github.com/frost-beta/mlx-cuda/commits?author=zcbenz&since=2025-03-20), though the feature is still in early stages and additional OS or CUDA versions are untested.** Comments note uncertainty about practical gains versus existing CUDA-native libraries like llama.cpp, question performance compared to formats like gguf/awq, and discuss the appropriateness of the phrase 'CUDA is coming to MLX' versus 'MLX is coming to CUDA.'
    - A commenter raises a key technical question about performance: they are interested in how MLX's CUDA implementation will compare against existing CUDA-compatible libraries—such as gguf or awq—particularly in terms of model quantization speed and efficiency, since 'mlx quants usually come out fast.'
    - Another user points out that the functional overlap may be limited, since popular inference libraries like llama.cpp already offer solid CUDA support, implying that MLX might not provide a significant advantage or why it would be necessary unless it brings unique features or performance improvements.
    - There is discussion regarding the current status of CUDA support in MLX: a user notes that the CUDA integration has not yet been merged, indicating that there may still be development, testing, or review steps before full availability and stability can be expected.

### 3. Critical Industry Perspectives: Meta's ASI Team and Benchmark Skepticism

- [**Meta's new ASI team discussed about abandoning Meta's powerful Open-source and focus on developing close**](https://www.reddit.com/r/LocalLLaMA/comments/1m14a9j/metas_new_asi_team_discussed_about_abandoning/) ([Score: 189, Comments: 60](https://www.reddit.com/r/LocalLLaMA/comments/1m14a9j/metas_new_asi_team_discussed_about_abandoning/)): **Meta's new superintelligence (ASI) team is reportedly considering abandoning open-source releases of large models, shifting focus toward closed AI development, as detailed in the linked New York Times article. This marks a departure from Meta's prior Llama open model releases, a policy initially pushed by Yann LeCun; with LeCun now sidelined, new leadership favors restricting access to powerful models, similar to policies at OpenAI and Google. Ongoing or future open releases may be limited to less capable models, akin to Google's 'Gemma'.** Top comments express concern that major tech firms deprioritize open source for commercial or control reasons, and that future open progress may depend on non-profits or Chinese developers. There's skepticism that Western big tech will meaningfully support open AI, with a sense of community hope shifting to entities like Deepseek, Ai2, and ETH.
    - Some commenters highlight that Meta's open-source push was heavily influenced by individuals like Yann Lecun, and with more "anti open-weight AI" leadership taking over, expectations for further major open releases are low. There is a technical implication that, without advocates at the executive level, the open-source momentum within big tech can quickly falter.
    - It is noted that Meta's most advanced open-source model currently ranks only number **44** on the LMSYS leaderboard, with claims of possible "benchmaxxing" and preferential evaluation bias. This suggests that, from a technical performance and benchmark standpoint, Meta's models are no longer considered competitive with the top AI labs, regardless of open-source status.
- [**Your unpopular takes on LLMs**](https://www.reddit.com/r/LocalLLaMA/comments/1m0z1zx/your_unpopular_takes_on_llms/) ([Score: 496, Comments: 358](https://www.reddit.com/r/LocalLLaMA/comments/1m0z1zx/your_unpopular_takes_on_llms/)): **The OP contends that most public LLM benchmarks such as MMLU are of limited value, reflecting primarily a model's memorization of training data rather than generalization, and criticizes a lack of integrity in benchmark question secrecy. They also argue that using LLMs to judge 'writing style' is invalid and that most community finetunes degrade base model quality due to inexperienced practices and indiscriminate model uploads, with a call for better curation and potential resource costs to prevent low-quality proliferation.** Commenters offer mixed technical perspectives: some dismiss public benchmarks entirely, suggesting model popularity among specific user communities (e.g., gooners) as a real-world metric; others highlight the lack of crucial information (sampler, quantization, parameters) in LLM discussions, and remark on the overwhelming pace of LLM advancements. There are contrasting views on team quality (Mistral lauded for efficiency and focus), and some express nuanced biases based on model origin (e.g., Chinese models), alongside concerns about LLMs reducing user cognitive engagement.
    - Evening_Ad6637 highlights a key challenge in LLM discussions: threads frequently lack technical context such as sampler type, hyperparameters, quantization method, or inference details, which are critical for reproducibility and understanding performance tradeoffs. Mistral is singled out for *efficient engineering and strategic focus on meaningful improvements* versus aggressive marketing, suggesting attention to design and optimization priorities within the LLM ecosystem.
    - hotroaches4liferz criticizes creative writing benchmarks that use LLMs as judges of other LLMs, arguing this introduces a significant bias where generic 'AI slop' is rewarded and genuinely superior models like Claude are penalized. The comment suggests that such benchmarking methods are technically unsound and conflate style mimicry with substantive quality, which can mislead both research and user communities.
    - orrzxz expresses skepticism about the current direction toward AGI, arguing that advances in statistics-driven text prediction and autocomplete don't meaningfully progress toward general intelligence. The post underscores a deeper debate: whether current LLM architectures and benchmarks inherently limit progress toward broader forms of AI, despite rapid advancements in model performance and complexity.
- [**He’s out of line but he’s right**](https://i.redd.it/dqx9wlf3q9df1.jpeg) ([Score: 1467, Comments: 105](https://www.reddit.com/r/LocalLLaMA/comments/1m1i922/hes_out_of_line_but_hes_right/)): **The image uses a meme format, featuring an anime character with stylized technical UI elements, to satirize the importance of local-hosted AI companions. The post critiques cloud-based AI 'girlfriends,' humorously suggesting that only locally run and personalized models are acceptable, framing remote/cloud models as insecure ('snitch') or commodified. The technical implication centers on issues of privacy, user control, and customization in AI deployment, advocating for locally running AI models over cloud solutions for deeply personal use-cases.** Commenters reinforce the privacy and security concerns, emphasizing the value of local models ("GLORY TO THE LOCAL HOST") and mocking the perceived coldness or risk of cloud-based AI due to data not being processed on the user's machine.
    - One user references that a comment was likely copied from a top post on r/LocalLlama, pointing to community concerns about originality and meme circulation within the AI and local LLM enthusiast spaces. The implication is that certain jokes and themes are recirculating, likely because of their relevance to local LLM deployment discussions.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. Meta's Recruitment of Top OpenAI Talent and Industry Reactions

- [**Meta poaches 2 more high profile OAI researchers**](https://www.reddit.com/r/singularity/comments/1m10bhk/meta_poaches_2_more_high_profile_oai_researchers/) ([Score: 587, Comments: 166](https://www.reddit.com/r/singularity/comments/1m10bhk/meta_poaches_2_more_high_profile_oai_researchers/)): **Meta has recruited high-profile researchers Jason Wei (noted for co-authoring the seminal Scaling Laws paper and leading agentic/reasoning research) and Hyung Won Chung (Head of Codex, core GPT-4 architect, key contributor to the 'o' series and Deep Research) from OpenAI, as confirmed via social media announcements (see [source](https://x.com/tbpn/status/1945290640545243503?s=46)). This move potentially strengthens Meta's capabilities in scaling laws, agentic models, and advanced LLM architectures, given these individuals' direct influence on OpenAI's most advanced systems.** Commenters express that Meta's ongoing recruitment of core OpenAI talent could significantly impact future model innovation and position Meta to make major breakthroughs; concerns are also raised about the broader implications for OpenAI's long-term talent pool.
    - Multiple comments highlight the strategic technical significance of Meta hiring Jason Wei, a prominent researcher known for his intensive work ethic and technical contributions at OpenAI. Wei's expertise, especially in areas such as large language models and foundational model scaling, is considered a major gain for Meta's AI research direction. The community expectation is that these high-profile hires will strongly influence the sophistication and performance of Meta's next-generation models, potentially accelerating their competitiveness in state-of-the-art benchmarks.
- [**Zuckerberg poaches Jason Wei and Hyung Chung (co-creators of GPT4 and o-series) to Meta Superintelligence**](https://www.reddit.com/gallery/1m11d3l) ([Score: 250, Comments: 118](https://www.reddit.com/r/singularity/comments/1m11d3l/zuckerberg_poaches_jason_wei_and_hyung_chung/)): **Mark Zuckerberg has recruited Jason Wei and Hyung Chung, both noted as co-creators of GPT-4 and the 'o-series' at OpenAI, to lead efforts at Meta Superintelligence. The poaching signals Meta's intent to rapidly scale its in-house AI research and potentially compete with OpenAI on frontier LLM development, emphasizing high-profile talent acquisition as a strategic move. No direct technical benchmarks or model implementation details are discussed in the post due to lack of access to the source gallery.** Commenters speculate that Meta may become a leading AI player within two years, driven by talent acquisition; there is also meta-discussion about the ongoing competitive 'employee war' between firms as a key industry dynamic, though no model-specific debates are present.
    - Discussion highlights how the ongoing "employee war"—specifically the poaching of key figures like Jason Wei and Hyung Chung, both associated with GPT-4 and OpenAI's o-series—could disrupt team cohesion and negatively impact research progress. There's skepticism that such high-profile defections translate directly into accelerated innovation, with some commenters arguing that fragmentation of top talent across companies may actually slow advancements rather than speed them up.
- [**Interesting, is Meta a retirement home, or will the top talent they brought in actually put in the work to match the huge paychecks?**](https://i.redd.it/sym40ee669df1.png) ([Score: 226, Comments: 105](https://www.reddit.com/r/singularity/comments/1m1fdjy/interesting_is_meta_a_retirement_home_or_will_the/)): **The image is a screenshot of a tweet commenting on Meta's recent attempts to recruit top AI researchers with very high compensation packages. The tweet claims many leading researchers reject such offers due to concerns about personal integrity and the perception that joining Meta is tantamount to 'cashing out' or treating it as a 'retirement home.' This raises questions about whether Meta can successfully recruit and motivate the necessary talent solely through large monetary incentives.** Top comments dispute the tweet's assumptions, noting that such lucrative offers almost certainly include stringent performance requirements, minimum terms, and milestone-dependent bonuses. Commenters suggest the premise of 'do nothing for huge pay' is misguided, as high-level contracts typically enforce deliverables, and some are skeptical whether truly top developers would accept offers without strong success-linked clauses.
    - Several commenters clarify that large offers like '$100 million' are not paid upfront; they typically include multi-year vesting periods (often 4 years), and are contingent on continued performance and employment. Such compensation structures aim to incentivize long-term commitment and productivity, rather than offering lump-sum payouts.
    - Discussion highlights that Meta's compensation packages for top talent often include performance-related bonuses, and minimum term requirements. These terms ensure new hires continue contributing actively to the company rather than treating it as a "retirement home."
    - One commenter points out that contracts with full payment contingent on project or product success are rare, as top developers/researchers typically avoid contracts with success-tied vesting (especially for extremely high compensation amounts).
- [**Get paid $440,000 a year to build goonbots**](https://i.redd.it/mrhwjensa8df1.png) ([Score: 540, Comments: 42](https://www.reddit.com/r/OpenAI/comments/1m1b37u/get_paid_440000_a_year_to_build_goonbots/)): **The image is a screenshot of a job listing for a 'Fullstack Engineer - Waifus' role in San Francisco and Palo Alto, CA, offering a substantial salary of $440,000 per year. The post appears legitimate, with a top comment linking to the actual Greenhouse job posting at xAI (https://job-boards.greenhouse.io/xai/jobs/4789505007), suggesting it is associated with Elon Musk's AI startup. The listing's focus on 'waifus' and large salary indicates a likely intersection of advanced AI (potentially conversational or character-driven bots) and consumer-facing applications, potentially referencing recent interests in AI companions and character bots.** Commenters express significant surprise at the salary and some skepticism or humor about the job title, but no deep technical debate is present. The validation of the job's legitimacy does suggest seriousness around high-salary AI/ML engineering roles at cutting-edge firms, especially as job markets tighten.
    - A commenter linked to the job listing at xAI (https://job-boards.greenhouse.io/xai/jobs/4789505007), confirming the $180k–$440k salary range and suggesting that the upper figure represents the total compensation cap for the role, not a guaranteed starting salary. This highlights industry compensation transparency for high-demand AI engineering roles.
    - One user critiques the apparent contradiction in tech industry messaging: while companies tout AI as a tool to automate software development (potentially reducing demand for human coders), the same firms are hiring highly paid developers to create specialized AI applications such as conversational or companionship bots ("sexbots"). This underscores persistent gaps in AI capability and the ongoing need for expert human involvement in AI productization.
- [**Real**](https://i.redd.it/6rp037fgl8df1.png) ([Score: 443, Comments: 47](https://www.reddit.com/r/singularity/comments/1m1cdwt/real/)): **The image shows a genuine job listing from xAI for a 'Fullstack Engineer - Waifus', based in San Francisco and Palo Alto, CA. The listing references xAI's mission to create advanced AI systems with broad understanding, highlighting the blend of serious AI research ambitions and consumer applications centered on AI-generated waifus. This reinforces both the company's broad product scope and the expanding intersection between AI research and pop-culture entertainment products.** Comments reflect a mix of disbelief and concern about the direction of AI development, joking about an AI 'waifu arms race' and implications for societal manipulation. No detailed technical debates present.
    - Several users refer to the concept of a 'waifu gap,' indicating a perceived disparity in the development of AI-driven anime character generators, suggesting a competitive landscape similar to an arms race in AI capabilities. This implies an ongoing accelerated improvement and possibly international competition in specialized domains of generative AI.
    - Concerns are raised about the potential use of generative AI waifu models as forms of targeted marketing or propaganda, suggesting these systems could serve as powerful psychological tools to shape opinions and consumer behavior, underlining the need for ethical considerations and transparency.
    - The mention of 'waifu QAT member' (Quality Assurance Testing) reflects real workflows in AI model development, where specialized testers ensure model outputs meet specific criteria—hinting at the industrialization and professionalization of AI content generation in niche markets.

### 2. Latest Video and LoRA Model Releases and Community Updates

- [**LTXV Just Unlocked Native 60-Second AI Videos**](https://v.redd.it/tq7aozwa3adf1) ([Score: 233, Comments: 57](https://www.reddit.com/r/StableDiffusion/comments/1m1ka0n/ltxv_just_unlocked_native_60second_ai_videos/)): **LTXV is an open-source video generation model from Lightricks, claiming to be the first to enable native long-form (30-60+ second) AI video with strong prompt-based and control LoRA (pose, depth, etc.) support, even over extended durations. It runs on consumer GPUs by adjusting the per-frame chunk size, and offers workflows for multi-prompt story direction and control net features ([GitHub](https://github.com/Lightricks/LTX-Video)). At launch, full plain PyTorch inference is still a WIP, with main support via ComfyUI workflows. Example workflows demonstrate chaining prompts and applying control nets across long clips.** Top comments critique the initial example videos for lacking dynamic content, though one cited a [Forbes-hosted demo](https://youtu.be/J9lkHG6duac?si=zvdRBxVCqpicFGzp) as showing remarkable long-term coherence. There's technical debate on practical uses (e.g., as a driving v2v backbone) and acknowledgment that maintaining 60s coherence is a significant achievement.
    - A commenter highlights a more technically substantial example video from Forbes, emphasizing that maintaining coherence over a 60-second AI-generated video is a significant achievement. They speculate that, if LTXV remains efficient at this length, the technology could be leveraged for video-to-video (v2v) driving applications, indicating potential for practical use cases beyond simple generation.
- [**Lightx2v just released a I2V version of their distill lora.**](https://www.reddit.com/r/StableDiffusion/comments/1m125ih/lightx2v_just_released_a_i2v_version_of_their/) ([Score: 229, Comments: 92](https://www.reddit.com/r/StableDiffusion/comments/1m125ih/lightx2v_just_released_a_i2v_version_of_their/)): **Lightx2v released new image-to-video (I2V) and updated text-to-video (T2V) LoRA models for Wan2.1-14B on HuggingFace and CivitAI. The I2V LoRA, via StepDistill and CfgDistill approaches, improves motion consistency and prompt adherence compared to previous versions. Reports indicate the newly uploaded T2V model addresses earlier functional issues, with alternate extractions available that show increased motion generation.** Commenters independently extracted the T2V LoRA due to persisting loading issues, noting architectural and motion-generation differences between releases. Early user tests confirm improved motion and prompt fidelity.
    - Kijai notes that the new T2V (text-to-video) distill LoRA shared by Lightx2v was not directly functional, prompting them to extract compatible versions with various ranks, available at their HuggingFace repository (https://huggingface.co/Kijai/WanVideo_comfy/tree/main/Lightx2v). Kijai points out that this updated model is technically distinct from the initial version, exhibiting increased motion in generated video outputs.
    - Users report varied technical performance: while some note improved movement and adherence in outputs from the latest versions, others point out issues with the new T2V version producing 'just noise.' In contrast, the I2V (image-to-video) model's output quality is praised as 'fantastic' and notably improved compared to earlier iterations.
    - Roculus highlights a clear, positive technical improvement, specifically mentioning 'noticeable movement/adherence improvement,' implying better frame consistency or subject tracking in generated videos compared to prior models.
- [**I've released Place it - Fuse it - Light Fix Kontext LoRAs**](https://i.redd.it/2mgus4ljw7df1.png) ([Score: 375, Comments: 70](https://www.reddit.com/r/StableDiffusion/comments/1m19nqp/ive_released_place_it_fuse_it_light_fix_kontext/)): **The image provides visual examples demonstrating the effects of three newly released Kontext LoRAs—Place it, Fuse it, and Light Fix—on different generative tasks. Each LoRA is trained with a small dataset (20 before/after images), for 2000 steps at a 0.0003 learning rate, using the fal.ai Kontext LoRA trainer. The left column shows portrait alterations, the center demonstrates object compositing (e.g., a green globe placed in a kitchen sink), and the right column shows subtle changes in animated characters, evidencing each LoRA's focus: object placement, fusion, or lighting adjustment.** A top comment requests clarification on file naming conventions, highlighting some confusion in usability. Another user asks for clarification on the exact functionality of these LoRAs.
    - A commenter provides detailed training parameters: the training dataset consists of 20 before/after image pairs, with each LoRA trained for 2000 steps at a learning rate of 0.0003 using the [fal.ai Kontext LoRA trainer](https://fal.ai/models/fal-ai/flux-kontext-trainer).
- [**Wan 2.2 is coming this month.**](https://i.redd.it/jonoyc5dg9df1.png) ([Score: 207, Comments: 57](https://www.reddit.com/r/StableDiffusion/comments/1m1gt8c/wan_22_is_coming_this_month/)): **The post features a screenshot from Discord where a moderator confirms that the release of Wan 2.2, presumably an upcoming version of a machine learning model or toolkit, is planned for July. The discussion centers on anticipated new features or improvements compared to Wan 2.1, which is already highly regarded by the community. Technical concerns include potential compatibility with existing workflows or extensions such as Vace, CausVid, and LoRA, suggesting that users rely on integrations and backwards compatibility for their projects.** Comments express cautious optimism, focusing on whether Wan 2.2 will deliver substantial advancements or simply minor updates. Key worries are about maintaining compatibility with current tools and integrations.
    - Several comments express concern about **backward compatibility** with models and features such as VACE, CausVid, and LoRA, emphasizing how important it is that Wan 2.2 maintains the current ecosystem's interoperability: *'Just hope it won't break Vace, CausVid, LoRA etc. compatibility'*.
    - There is technical discussion about wan2.1's current support for advanced video control via models such as **VACE**, and speculation that Wan 2.2 should prioritize features like longer video generation or higher output resolution. Both of these would require significantly **larger VRAM**, pointing to hardware resource limitations as a key bottleneck for further progress.
    - Users note that if Wan 2.2 is only a minor version bump ('0.1 increase'), then existing LoRA models should likely remain compatible, but there is ongoing concern that even small updates could introduce breaking changes for widely used customization models.
- [**I made this AI Short Film to see how far I could push the current technology.**](https://v.redd.it/wmrmbrqgt5df1) ([Score: 179, Comments: 15](https://www.reddit.com/r/aivideo/comments/1m130hj/i_made_this_ai_short_film_to_see_how_far_i_could/)): **The short film 'LYRA' demonstrates end-to-end AI-driven cinematic content creation, employing a combination of state-of-the-art generative tools: image synthesis (MidJourney, Adobe Firefly), animation (SeedDance, Kling), and neural voice synthesis (11Labs). All narrative, visual, and audio components were orchestrated by a single producer, highlighting the current extent of AI automation in short-form storytelling and the integration capabilities across multiple platforms.** Top-level comments contain minimal technical debate, focusing instead on aesthetic appreciation and personal reaction; no substantive technical criticism or workflow discussion is present.
    - One technical discussion point focuses on compositing techniques in AI-generated film, with a user scrutinizing how deeply compositing is employed to enhance seamlessness and realism in the visuals. The effectiveness of compositing can be key to hiding model artifacts or smoothening transitions.
    - There is a comparative mention of AI video models, specifically Kling versus Veo, with the insight that Kling appears to maintain character consistency (i.e., preserving character appearance and identity) across generated scenes more reliably. This is highlighted as a critical metric for longer-form video generation.
- [**I made this AI Short Film to see how far I could push the current technology.**](https://v.redd.it/wmrmbrqgt5df1) ([Score: 175, Comments: 15](https://www.reddit.com/r/aivideo/comments/1m130hj/i_made_this_ai_short_film_to_see_how_far_i_could/)): **A single creator produced a 3-minute AI-generated short film ('LYRA') using a workflow incorporating MidJourney (for visual elements), SeedDance (likely for animation or video synthesis), 11Labs (voice generation), Adobe Firefly (generative imaging), and Kling (potentially for video enhancement or AI animation). The project serves as a demonstration of current multi-tool AI content pipelines for narrative filmmaking, emphasizing autonomous content creation at near-professional quality.** Top comments did not provide substantive technical discussion or critique of workflows, model limitations, or process, focusing instead on general impressions and praise.
    - A commenter speculates about the strengths of popular AI video models, specifically comparing Kling and Veo. They suggest Kling may be favored due to its ability to maintain consistent character appearance throughout a generated film sequence, highlighting a persistent challenge in AI video generation. Another commenter notes the importance of compositing and continuity, observing that the character visuals remained smooth and without noticeable 'jumps,' which addresses a typical artifact in current generative models.
- [**Mira Murati’s Thinking Machines Lab Update**](https://www.reddit.com/gallery/1m0ypbh) ([Score: 168, Comments: 50](https://www.reddit.com/r/singularity/comments/1m0ypbh/mira_muratis_thinking_machines_lab_update/)): **Mira Murati’s Thinking Machines Lab has reportedly closed a significant seed funding round, with discussion focused on the participation of NVIDIA (NVDA) as both a chip supplier and an AI sector investor. Commenters highlight the lab's high valuation despite having** `zero revenue`**, questioning the sustainability and rationale of such large early-stage investments in the current venture capital landscape, especially compared to established players like Waymo.** Key debate centers on whether NVDA's ubiquitous investment presence in private AI R&D is sustainable or signals potential over-valuation. There is skepticism regarding market bubbles, as commenters contrast the startup's valuation to that of more mature companies, and critique the norms and scale of contemporary venture capital funding rounds.
    - Several commenters highlight how NVIDIA (NVDA) appears in nearly every major private AI lab or investment story, reinforcing its central role not just as a chip provider but as a strategic investment lever across the AI ecosystem. There’s speculation that owning NVDA shares confers indirect exposure to any significant AI venture that emerges, due to their ubiquitous hardware and growing influence over AI infrastructure.
    - Technical skepticism is raised regarding the valuation of Thinking Machines Lab, with one user pointing out that the company has zero revenue yet is ascribed a valuation roughly 30% of Waymo’s—provoking concerns about AI startup bubbles or potentially undervalued incumbents.
    - There's interest in the strategic composition and approach of Thinking Machines, with notes on its similarities to Anthropic, such as a focus on open source and model interpretability. This is considered significant in an era when advanced models like GPT-5, Gemini 3, Claude (Neptune), and Grok 4 are all anticipated, and the market welcomes robust, interpretable alternatives with strong governance.

### 3. Claude Code Advanced Usage, Workflow Innovations, and User Experiences

- [**This is how you should be setting up Claude Code (discovered while researching with Claude, how meta)**](https://www.reddit.com/r/ClaudeAI/comments/1m17ilu/this_is_how_you_should_be_setting_up_claude_code/) ([Score: 232, Comments: 96](https://www.reddit.com/r/ClaudeAI/comments/1m17ilu/this_is_how_you_should_be_setting_up_claude_code/)): **The post introduces an open-source, modular command system for Claude Code, advocating against the common practice of massive, monolithic** `CLAUDE.md` **instruction files. Instead, it uses 20+ narrowly-scoped XML-like commands (e.g.,** `/dev:code-review --focus=security`**), each split into <requirements>, <execution>, <validation>, and <examples>, shown to reduce token usage by 50-80%, improve determinism, and enable rapid, contextual project setup. The approach leverages progressive disclosure (just-in-time instruction loading) and improved context management for Claude, resulting in far smaller and more relevant context windows. The repository is available at [github.com/oxygen-fragment/claude-modular](https://github.com/oxygen-fragment/claude-modular).** Technical feedback highlights the value of using Claude's native [command hooks](https://docs.anthropic.com/en/docs/claude-code/hooks-guide) for deterministic behavior and direct script execution. Another expert comment critiques the modular system as still leaving too much ambiguity for Claude, risking output infidelity, and stresses the need for even more explicit, low-level instruction for robust workflow engineering.
    - A user highlights the value of Claude's command hooks for enhancing determinism and automating workflows, such as running Python scripts, pointing to the official Anthropic documentation: https://docs.anthropic.com/en/docs/claude-code/hooks-guide. This technical feature allows for more controlled, reproducible Claude Code executions.
    - There’s a critique that the discussed setup process is often too high-level and lacks specificity, making Claude prone to errors due to vague instructions or context. The suggestion is that better results depend on precise and focused context engineering, with prompts that tightly define scope to avoid Claude making mistaken assumptions and compounding errors.
    - The 'LAZY_MODE_PROMPT' is provided as an additional resource for customizing Claude Code workflows, but it comes with a technical caveat: using complex or verbose prompts significantly increases token usage, which can have cost implications for users directly billed for API usage. Prompt engineering must balance clarity and token efficiency.
- [**3 years of daily heavy LLM use - the best Claude Code setup you could ever have.**](https://www.reddit.com/r/ClaudeAI/comments/1m1af6a/3_years_of_daily_heavy_llm_use_the_best_claude/) ([Score: 203, Comments: 64](https://www.reddit.com/r/ClaudeAI/comments/1m1af6a/3_years_of_daily_heavy_llm_use_the_best_claude/)): **The post details an advanced Claude Code environment for intensive daily LLM-based development, emphasizing the use of a custom OpenAI wrapper to utilize Claude Max subscriptions in OpenAI-compatible tools (e.g., via ngrok proxies to expose endpoints), and configuration of settings like 'ANTHROPIC_CUSTOM_HEADERS: anthropic-beta: interleaved-thinking-2025-05-14' and 'MAX_THINKING_TOKENS: 30000' for increased model context and performance. Integrating [Graphiti MCP](https://github.com/arben-adm/mcp-sequential-thinking) (a temporal knowledge graph on neo4j) with upgraded sequential-thinking MCP pipelines allows for automated, metadata-rich persistent memory—thoughts and context are scripted and stored in Graphiti. The stack extends with Exa Search/Firecrawl for real-time or reference data and Relace as an intermediary routing layer, combining AutoGen and [continue.dev](http://continue.dev/) for multi-agent orchestration with live, human-in-the-loop capabilities and persistent knowledge graphs.** There is debate about the credibility of Relace and whether the post serves as a shill. Technically-minded commenters emphasized the importance of robust context/knowledge base construction (versus per-feature memory), highlighting the challenges with maintaining global, persistent context in LLM workflows and asking for repo/code sharing.
    - A user asks in-depth about maintaining effective context with Claude Code, noting that *"context is key"* and describing their current workflow: building a Product Requirements Document (PRD), creating task lists from it, and using a markdown file to track and update tasks. They highlight a challenge—Claude Code can only track context for a single feature at a time, rather than across an entire project, which raises issues for comprehensive knowledge base management and prompt context delivery.
    - One comment reports a technical error with the Claude VS Code extension: *"End of central directory record signature not found. Either not a zip file, or file is truncated."* This error occurs during `/status` and after attempts at uninstalling and manually reinstalling the extension, suggesting a possible corruption or compatibility issue with the extension package, which may require deeper investigation or an official fix.
- [**As an Software Egineer with 20+ years of experience...**](https://www.reddit.com/r/ClaudeAI/comments/1m1efu0/as_an_software_egineer_with_20_years_of_experience/) ([Score: 264, Comments: 37](https://www.reddit.com/r/ClaudeAI/comments/1m1efu0/as_an_software_egineer_with_20_years_of_experience/)): **An engineer with 20+ years in .NET and large-scale cloud backends describes their ClaudeAI workflow: (1) prompt optimization using custom Lyra prompts, (2) rewriting Jira tickets in Claude for focused context, (3) breaking tasks into minimal chunks, (4) highly scoped prompts (e.g., restricting to interfaces or unit tests), and (5) iterative, chunk-based development. They emphasize that rigorously scoping and sequencing AI prompts significantly improves productivity and cognitive load management, especially when onboarding to complex codebases.** Commenters highlight potential for further automation in prompt optimization (e.g., feedback loops with Lyra prompts) and discuss career viability concerns in light of rapid AI advancement. There is also interest in operationalizing these workflows via custom commands or scripts.
    - One commenter highlights the use of a highly engineered Claude prompt that embodies senior-level software architecture principles: focusing on safe refactoring of complex, monolithic codebases into modular architectures, comparing the process to "surgery on a live patient." This approach emphasizes methodical and thoroughly tested refactors to maintain both functionality and test coverage, providing a template for maintaining code quality during modernization efforts ([link](https://github.com/centminmod/my-claude-code-setup)).
    - Technical curiosity is expressed regarding prompt optimization with the Lyra tool, particularly integrating this as a custom command and feeding the optimized prompt directly back into Claude for iterative improvement. There is discussion about whether this prompt refinement is better handled as a manual, separate step or automated within a workflow, alluding to possible combinations of prompt engineering and automation in LLM-based coding workflows.
    - A question arises about the relationship between Lyra and Traycer, probing for technical distinctions or overlap between these tools, potentially hinting at differing roles in prompt optimization or LLM interfacing—though no direct answer or technical comparison is provided in the context of the comments.
- [**Am I crazy or is Claude Code still totally fine**](https://www.reddit.com/r/ClaudeAI/comments/1m15ca6/am_i_crazy_or_is_claude_code_still_totally_fine/) ([Score: 113, Comments: 218](https://www.reddit.com/r/ClaudeAI/comments/1m15ca6/am_i_crazy_or_is_claude_code_still_totally_fine/)): **The post discusses Claude Opus API's code generation capabilities, with the OP reporting consistent high-quality outputs and no significant rate-limiting, even with intensive use (approx. $750 in API calls over 4 days), *contrary to widespread claims of declining model performance*. The Opus 50% usage warning triggers at ~$60 but hasn't led to hard limits for this user.** Comments reveal a **split user experience**: some report a measurable decline in output quality and poorer decision-making compared to the previous week, while others maintain *the model has not changed and remains reliable*, attributing perceived drops to user variability or anecdotal bias.
    - Users report noticeable fluctuations in Claude Code's output quality, with some citing a recent decline—particularly on premium plans like Opus—where the model makes suboptimal coding decisions compared to prior weeks.
    - A technical workaround discussed involves breaking coding tasks into smaller segments and stopping the model before it attempts automatic 'compaction', as this phase can introduce issues; this approach reportedly mitigates drops in quality and helps isolate reasoning flaws using plan mode.
    - One comment highlights that part of users' frustration may stem from their software design choices—suggesting that as their project complexity increases, model limitations (or architectural shortcomings) become more apparent, which can sometimes be mistaken for overall model degradation rather than user-side architectural probleme.
- [**Claude Code is back on the menu, boys! Great news.**](https://www.theverge.com/ai-artificial-intelligence/708521/anthropic-hired-back-two-of-its-employees-just-two-weeks-after-they-left-for-a-competitor) ([Score: 120, Comments: 23](https://www.reddit.com/r/ClaudeAI/comments/1m1mhlv/claude_code_is_back_on_the_menu_boys_great_news/)): **Anthropic has rehired Boris Cherny and Cat Wu, pivotal contributors to its Claude Code platform, merely two weeks after their departure to Anysphere (Cursor's developer). This underscores persistent volatility and talent wars in the generative AI coding assistant sector, notably as both Anthropic (Claude Code) and Anysphere (Cursor) position themselves for leadership in the AI-powered IDE and code suggestion space. For further reference, see [The Verge coverage](https://www.theverge.com/ai-artificial-intelligence/708521/anthropic-hired-back-two-of-its-employees-just-two-weeks-after-they-left-for-a-competitor).** Technical discussion in the comments speculates motivations beyond compensation, including reference to negative perceptions surrounding recent AI-powered IDE launches (e.g., Amazon's IDE and Windsurf), possibly influencing the return of the leaders to Anthropic.
    - Commenters speculate that recent changes in the landscape of AI-powered IDEs, specifically referencing issues with Windsurf and the launch of Amazon's new IDE, may have negatively impacted public perception or user adoption, leading to strategic reconsiderations by Anthropic in re-enabling Claude Code.
    - A user notes that Cursor's recent pricing decisions ('blew their pricing') may have driven users to seek alternatives, creating an opportunity or pressure for Anthropic to bring back Claude Code, implying a price-sensitive developer audience and heightened competition among AI code assistants.

---

# AI Discord Recap

> A summary of Summaries of Summaries by X.ai Grok 4
> 

**Theme 1. Kimi K2 Hype Ignites Model Wars**

- **Kimi K2 Crushes Sonnet in Speed Showdown**: Users hail **Kimi K2** on **Groq** as a cheaper, faster alternative to **Sonnet**, delivering **Opus-level** agentic tasks with **256K tokens** context at **250-300 tokens/second**, though it skips visual inputs and lags in tool calls compared to Moonshot. Certifications for speed badges on OpenRouter aim to spotlight variances like **10 TPS** vs **500 TPS** across providers.
- **Kimi K2 Efficiency Crowns Chinese Innovation**: **Kimi K2** wows with coding prowess and pricing rivaling **GPT-4.1 mini**, sparking mania for local hosting to dodge costs from **Claude 4** or **ChatGPT**, with users speculating it could fuel **Manus** as a *strong frontier level alternative to Claude*. It's now runnable locally via [Kimi-K2-Instruct-GGUFit on Hugging Face](https://huggingface.co/unsloth/Kimi-K2-Instruct-GGUFit).
- **Kimi K2 Sparks DeepSeek Rivalry Drama**: Amid **Kimi K2** excitement, **DeepSeek** faces backlash for heavy censorship favoring Chinese government, with users noting *other LLMs aren't that censored compared to DeepSeek*, and quality drops in **Q4 quantization** causing intense hallucinations in roleplay.

**Theme 2. GPU Optimization Tricks Steal the Spotlight**

- **BF16 Trumps FP32 in VRAM Battle**: Fine-tuning **LoRA** with **bf16** slashes VRAM use over **fp32**, except on older GPUs with **Gemma 3**, where a **7B model** in **fp32** gobbles **28GB**. Users snag **B200 GPUs** at **$2/hour** promo from [DeepInfra](https://deepinfra.com/low), dodging exclusions like Nebraska via tweet fixes.
- **Unsloth Outmuscles Liger-Kernel in Benchmarks**: **Unsloth** saves **15-30%** more VRAM than **Liger-Kernel** in tests, boasting *insane context length with unsloth gradient checkpointing*, though recent updates trigger **timeout errors** and VLLM cache issues at default `.cache/vllm/torch_compile_cache`.
- **H20 GPU Sparks Bandwidth Buzz**: China's **H20** matches **H100** interconnect bandwidth for inference but flops in training against **GB200/GB300**, with users joking about **NVL144** vs **NVL72** confusion while **Voltage Park** hires engineers for AI Factory stack at [Voltage Park Careers](https://www.voltagepark.com/careers?ashby_jid=2e463e6a-abc6-48ae-8060-8452c55b2fab).

**Theme 3. Research Papers Drop Bombshells on Efficiency**

- **ETHOS Paper Revolutionizes Sparse Transformers**: The [ETHOS paper on Arxiv](https://github.com/wrmedford/ETHOS) unveils Efficient Transformers via Hypernetwork Organized Sparsity, with experts stored as latent codes for **15K tokens/second** training on GH200, promising theoretical **20x less FLOPs** despite backward bottlenecks. It defines *LLM psychosis* as *a mental disorder characterised by a disconnection from reality* from hallucination loops.
- **GPUHammer Exposes Memory Mayhem**: The [GPUHammer paper](https://arxiv.org/abs/2507.08166) probes memory corruption vulnerabilities in data structures, inspiring work on susceptible algorithms. Paired with [Muon optimizer video](https://www.youtube.com/watch?v=4bFDPVe6BHs), it targets tool usage rivaling **Claude 4**, showing early promise in tests.
- **MoEs Tackle Memory Bandwidth Bottlenecks**: Labs optimize **Mixture of Experts (MoEs)** for memory efficiency, as shown in [this video](https://youtu.be/JOqLp1adGO4?si=hUAnREYY5CQoeoaQ), enabling fewer GPUs for training than dense models. Nvidia's [LLM RL framework on GitHub](https://github.com/ritser-labs/real-work) eases long-horizon tasks in Docker with tool access.

**Theme 4. Tools and Frameworks Level Up Agentic AI**

- **OpenPipe's ART Agents Up Any Model**: [OpenPipe's ART on GitHub](https://github.com/OpenPipe/ART) uses *LLM as a judge* to boost model agentic traits, deemed *actually pretty interesting* and integrated with Unsloth for finetuning. Users eye **ARTwell RULER** tests post-confirmation it's *pretty good*.
- **nnterp Unifies Mech Interp for Transformers**: Beta release of [nnterp on GitHub](https://github.com/Butanium/nnterp) bridges **transformer_lens** and **nnsight** with a unified interface for all Hugging Face transformers, featuring **1915 precomputed tests** and a [demo Colab](https://colab.research.google.com/github/Butanium/nnterp/blob/main/demo.ipynb).
- **MCP Tools Grant AI Superpowers**: Anthropic's [connectors directory](https://claude.ai/directory) broadens MCP access beyond devs, while [Glasses-MCP on GitHub](https://github.com/gourraguis/glasses-mcp) lets AIs screenshot URLs and emulate screens; Goose adds subagents for multi-model orchestration with **Claude Sonnet-4**.

**Theme 5. Benchmarks and Evaluations Face Reality Checks**

- **Eval Harness Hunts Model Drift**: Proposed OpenRouter eval harness baselines against published benchmarks, tracking *drift* in scores and verifying no context compression via **128K** fiction tests. It includes tool use like [Tau-2 airline on GitHub](https://github.com/sierra-research/tau2-bench) to catch template bugs.
- **Aider Benchmark Begs for Overhaul**: With models topping **80%** on Aider's polyglot benchmark, users push for updates and private user-submitted tests; [SwitchPoint Router on OpenRouter](https://openrouter.ai/switchpoint/router) hits **80%** by routing to **GPT-4.1** or **Gemini 2.5 Pro** at lower costs.
- **LMArena UI Tweaks Battle Bugs**: **LMArena** users report model errors and false positives in content filters, especially comics, with new UI feedback on model selection; **Grok4** slammed for *30k+ hidden reasoning* yielding one-word responses despite benchmarks.


---

# Discord: High level Discord summaries




## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **DeepSeek Favors Government Censorship**: Members observed that **DeepSeek AI** exhibits heavy censorship, favoring the Chinese government in its responses.
   - Concerns were raised about the extent of this censorship compared to other LLMs, as members noted, *other LLMs aren't that censored compared to DeepSeek*.
- **IQ Tests Distinguish Gifted Individuals**: Members debated the usefulness of **IQ tests** for measuring genius, with some sharing scores and standard deviations, citing [a Mensa IQ test](https://test.mensa.no/Home/Test/en-US).
   - It was noted that *schools sometimes use it on kids to try and distinguish gifted individuals*.
- **OpenAI Gears Up for GPT-5 Announcement**: The community speculates on upcoming OpenAI announcements, expecting possibilities such as a new browser, learning feature, new agents, or even **GPT-5**.
   - One member anticipates, *Prob the browser right* and linked to the [OpenAI Twitter announcement](https://fxtwitter.com/OpenAI/status/1945607177034760574).
- **API Integration Wait Forces Membership Consideration**: A member expressed intent to *wait till they integrate it into the API* and consider canceling the **pro membership** due to limited value.
   - They cited poor integration and lack of development for features like *operator* as reasons for potentially discontinuing their subscription.
- **Users Discuss Future of AI Coding Libraries**: A member inquired about the storage location of **AI coding libraries**, seeking to teach the AI new coding skills, similar to **Jan AI**.
   - As of the discussion, no specific answer or guidance was provided regarding the location or method for teaching AI new coding skills.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity's Comet Browser Gets Grilled on Reddit**: **Perplexity** CEO Aravind Srinivas and Comet Product Lead Leonid Persiantsev answered questions about the **Comet browser** in a Reddit AMA, discussing its rationale, evolution, key features, use-cases, and future plans, and available [here](https://www.reddit.com/r/ChatGPT/comments/1m1javp/comet_ama_with_perplexitys_aravind_srinivas_and/).
   - Users highlighted its seamless integration with **Google apps** like **Gmail** and **Calendar**, enabling features like analyzing emails and scheduling appointments.
- **Perplexity Image Gen Hits Turbulence**: Users reported issues with **Perplexity's image generation**, encountering text responses instead of images, with the image generation feature being temporarily unavailable on the web as the team works on improvements.
   - Users found a workaround: generating with 'an image of' rather than 'generate image of'.
- **Samsung Galaxy Users Score Perplexity Pro Perks!**: Members shared a deal for 12 months of **Perplexity Pro** via the **Samsung Galaxy Store** for users in the **USA**.
   - However, users not based in the **USA** may face account suspension, so do your own diligence!
- **Perplexity Free Tier Might Get Ads?**: Speculation arose about potential ads on the free tier, and [Sam confirmed](https://www.reddit.com/r/ChatGPT/comments/1m1javp/comment/n3hnolo/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) that Perplexity wants ads to be visibly separate from LLM responses and entirely separate from LLM output.
   - Members suggest non-intrusive placements like UI ads or sponsored search ads.
- **API Access Included with Perplexity Pro: True or False?**: **Perplexity Pro** includes **API access**, providing **$5 monthly credit** for **Sonar models**, as detailed in the [Perplexity AI help center](https://www.perplexity.ai/help-center/en/articles/10352901-what-is-perplexity-pro).
   - This allows users to embed **AI-powered search** in their projects, offering features like citations for research tasks.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor's Credit System Gets a Reality Check**: Users report that **Cursor's usage dashboard lags** and may not immediately reflect when the **$20** credits are exhausted, leading to unexpected billing surprises, but now users receive in-chat notifications upon reaching the billing point.
   - Some users find the **Ultra plan** too restrictive, exceeding limits quickly and questioning its long-term viability, with one user reporting exceeding **$60** in usage.
- **Kimi K2 Steals the Show from Sonnet**: Members are enthusiastically discussing **Kimi K2** via **Groq** as a potentially superior and cheaper alternative to **Sonnet**, boasting **Opus-level** agentic task performance and a longer context window of **256K tokens** at **250-300 tokens per second**.
   - However, some users noted that **Kimi K2** doesn't accept visual input, and is not as good at tool calls as on Moonshot.
- **Cursor's Composer Plagued by Prompt Problems**: Users are experiencing issues with **Cursor** getting stuck on prompts, wasting usage, but a new build includes a **180-second timeout** to auto-cancel stuck requests and prevent billing.
   - Restarting **Cursor**, reinstalling, or using a new prompt to *rethink* are suggested solutions, with one user suggesting it helps to retrigger.
- **Multi-Agent Systems Assemble for IDE Domination**: A user is developing a **multi-agent MCP server** for managing asynchronous agentic tasks across multiple IDEs, aiming to solve the state persistence issue, employing **gemii** for debugging, **grok4** for heavy refactoring, and **Claude** for coding.
   - Another user efficiently edited Cursor's `main.js` and other IDEs to create a bare minimum MCP, using JSON for I/O, connecting them upon launch.
- **Context Engineering: The Secret Sauce Behind Smart Agents**: Users emphasize the importance of providing the right context to AI agents, known as **Context engineering**, advocating for including all necessary information in the initial prompt.
   - Others highlight the benefit of maintaining a simple set of rules or reference documents in the **System Prompt**, noting that too many rules may cause the context to loosen up in the long term.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **BF16 Beats FP32 for LoRA (mostly)**: Using **bf16** over **fp32** saves VRAM when fine-tuning **LoRA**, except with **Gemma 3** on older GPUs; a **7b model** in **fp32** can use **28GB** of VRAM.
   - Users discussed renting **B200 GPUs** from [DeepInfra](https://deepinfra.com/low) at a promotional rate of **$2/hour**, but the promotion initially excluded Nebraska due to [a tweet](https://x.com/DeepInfra/status/1935459646493573123) needing correction.
- **OpenPipe ART claims to agentify Models**: **OpenPipe's ART** ([link](https://github.com/OpenPipe/ART)) claims to make any model more agentic using **LLM as a judge**.
   - One member acknowledged the tool was *actually pretty interesting* and uses Unsloth, and another member is looking to try out **ARTwell RULER** after confirming *it's pretty good*.
- **ETHOS Paper makes Arxiv debut**: The **ETHOS** (Efficient Transformers via Hypernetwork Organized Sparsity) paper, [now on Arxiv](https://github.com/wrmedford/ETHOS), discusses efficient transformers, with a pdf attached [here](https://cdn.discordapp.com/attachments/1257011997250424842/1394830874142576712/ETHOS___Efficient_Transformers_via_Hypernetwork_Organized_Sparsity_10.pdf?ex=68798e7b&is=68783cfb&hm=0b2c8891c328a38668ff0d015cf8a9f8ef5b80884a3f7eb0ee486aa149b25e97&).
   - Members defined **LLM psychosis** as *a mental disorder characterised by a disconnection from reality*, caused by LLM hallucinations reinforcing misconceptions.
- **Snortts Indic TTS hits the Playground**: A member shared a playground for testing **TTS models** at [https://snorbyte.com/snortts-indic-v0](https://snorbyte.com/snortts-indic-v0), and is inviting users to test it out and ask questions.
   - Some members are complaining about being stymied by an *affliction* of **LLM psychosis** and are getting immediately dismissed because they don't have a PHD.
- **Unsloth beats Liger-Kernel in benchmarks**: In benchmarks, **Unsloth** saves **15-30%** more VRAM than **Liger-Kernel**, and has *insane context length with unsloth gradient checkpointing*.
   - One user also noted that **Unsloth** has had recent updates and members are experiencing **Timeout errors**, also, the VLLM cache defaults to `.cache/vllm/torch_compile_cache` and can cause problems.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **ZR1-1.5B Endorsed for Reasoning**: A member suggested using the [ZR1-1.5B model](https://huggingface.co/Zyphra/ZR1-1.5B) for reasoning tasks, and also mentioned **DeepCoder-1.5B** as a solid option.
   - The member cautioned about the difficulty of achieving general-purpose reasoning within a **7B** parameter budget.
- **TensorFlow Losing Favor in Job Market**: Multiple members expressed the sentiment that using **TensorFlow** is becoming a red flag in job requirements, with one noting that *many people have sworn off tensorflow at this point*.
   - A member stated that there isn't a good reason to use **TensorFlow** over **PyTorch** or **JAX**.
- **Input Injection Accelerates Recursive NNs**: A member suggested that a top improvement for *how you recurse* is **naive input injection at each iteration**, acting as a [skip connection](https://link.to.skipconnection) to propagate state more easily.
   - They clarified that this involves injecting the **hidden state**, an earlier hidden state, or the original input itself.
- **Latent Codes Make Experts Ephemeral**: A researcher detailed their approach of storing experts as **latent codes** and recovering them on the fly, sharing a [GitHub repository](https://github.com/wrmedford/ETHOS) and highlighting a training speed of **15K tokens/second** on a GH200.
   - The 20x less FLOPs is the theoretical speedup, not empirical yet, as bottlenecked by suboptimal backward, where *experts only exist ephemerally* and do not receive gradients, autograd is storing intermediates.
- **New Mech Interp Package nnterp Released**: A member released the beta 1.0 version of their mech interp package **nnterp** [on Github](https://github.com/Butanium/nnterp), installable with `pip install "nnterp>0.4.9" --pre`.
   - The package aims to provide a unified interface for all **transformer models** while still using the huggingface implementation under the hood, closing the gap between `transformer_lens` and `nnsight`, with a [demo colab](https://colab.research.google.com/github/Butanium/nnterp/blob/main/demo.ipynb) and [docs](https://butanium.github.io/nnterp/).



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter's SwitchPoint Router Sparks Privacy Alarm**: A user reported that the **SwitchPoint Router** was pre-selected without their consent, potentially violating their **NDA** and sending tokens to China, after adding a firewall rule to ban the OpenRouter Chat, deeming it *not safe enough for use*.
   - An OpenRouter admin clarified that Switchpoint is **US-based**, not China, and users can disable providers in [settings](https://openrouter.ai/settings/preferences), and the auto router picks between a list of high quality models on OpenRouter.
- **DeepSeek Quality Plummets with Q4 Quantization**: A user noted a significant drop in quality for **Deepseek R1 0528** in roleplay, with the model hallucinating more intensely at lower quantizations.
   - Another user agreed, recalling a **truly horrible R1 performance**, and is doing tests comparing **Q4** to **fp8**.
- **OpenAI's GPT 3.5 Turbo Endpoint Mysteriously Vanishes**: A user pointed out the disappearance of the `openai/gpt-3.5-turbo` endpoint, seeking clarification and noting that other providers could've served you, saying that it had successful chess records until **2025-06-23**.
   - An OpenRouter admin responded that they are looking into this issue and have **resurrected** it for future use.
- **OR drafts Quality and Speed Certification Badges**: OpenRouter is exploring quality and speed certification badges for models, similar to `:nitro` filters, to address disparities like **Kimi K2's** varying speeds across providers (**10 TPS** vs **100 TPS** vs **500 TPS**).
   - The aim is to highlight providers with reliable tool calling, consistent ratelimits, and high-quality outputs, while accounting for quantization and potential tool call failures, with new models starting at an unverified tier.
- **Eval Harness drifts Model Benchmarks**: The proposed eval harness would baseline against model authors' published benchmarks, continuously measuring the *drift* or difference, between official scores and endpoint scores to verify model performance at [OpenRouter](https://openrouter.ai/).
   - A long fiction benchmark up to **128k context** is suggested to verify the absence of context compression shenanigans, alongside tests prompting models to output many tokens to confirm providers' stated output token count.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Polymarket Bettors Get Burned**: Users expressed surprise that bets on **Polymarket** did not pan out, particularly those involving **Google**.
   - One user humorously lamented *my ambitious bet on polymarket didn't pay off then 💀I'm shocked that there are actually people in this server who don't buy Google💀*.
- **Prediction Market Legality Debated**: The legality of prediction markets in the US was discussed, with **Kalshi** mentioned as a potentially legal platform, though with liquidity concerns.
   - Some members cited **CFTC** regulation while others voiced concerns about potential tax issues.
- **Grok4 Talks Too Much, Delivers Too Little**: **Grok4** received criticism for being overly verbose and generating excessive hidden reasoning, which seemingly contradicts its benchmark performance.
   - A user criticized xAI for creating a model that outputs *30k+ hidden reasoning and then respond with 1 word to 1 sentence*.
- **Kimi K2 Claims the Efficiency Crown**: **Kimi K2** impressed users with its efficiency and coding capabilities, boasting competitive pricing compared to **GPT-4.1 mini** and notable performance gains.
   - A member stated *The Chinese have absolutely cooked when it comes to efficiency* and also noted its ability to produce interesting physics sandboxes.
- **LMArena Gets Bug Reports and UI Tweaks**: Users reported issues with **LMArena**, including model errors and false positives from the content filter, especially with comic book content.
   - Feedback was provided on the new **UI**, specifically the model selection process in direct chat, with a community manager directing users to a feedback thread.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Kimi UI Origins: Bitter Lessons Shared**: A [Medium article](https://medium.com/@xinyijin715/maker-story-the-bitter-lessons-behind-kimi-researchers-ui-6654ec66662c) reveals the backstory and **bitter lessons** behind **Kimi UI's** development, hinting at potential links to Cursor's code.
   - A member referenced a [YouTube video](https://www.youtube.com/watch?v=motX94ztOzo&t=2737s) suggesting **Cursor's code** might have played a role.
- **Atlassian's Rovo Dev: A Capable AI Agent?**: Poonam Soni introduced **Atlassian's** new **AI agent, Rovo Dev**, accessible via CLI, designed for software development tasks like **code generation, review, and debugging**.
   - Despite the hype, one member quipped that *Atlassian is constitutionally incapable of building a good product* while others reported issues with the **download process and enterprise focus**.
- **Flux Pro and Seedance Forge Realistic Videos**: A user combined **'flux pro' and 'seedance' models** to generate realistic videos, employing the **'IMG_XXXX.JPG' trick** for initial images and prompts like *'aggressively mediocre home footage'*. 
   - The discussion included a link to the [Replicate model](https://replicate.com/) and positive user feedback, though the company's background remains unclear.
- **Coding AI Leaders Return to Anthropic**: **Two coding AI specialists** have been rehired by **Anthropic** following a brief period at **Cursor**, as reported by [The Information](https://www.theinformation.com/briefings/anthropic-hires-back-two-coding-ai-leaders-cursor-developer-anysphere).
   - The move sparked speculation about whether they acted as *double-agents*, providing *cheap consulting* to Cursor from *world class experts*.
- **OpenAI Teases Operator for 2025**: OpenAI released a cryptic video hinting at a launch on **July 16, 2025**, sparking speculation that ranges from a **browser/operator upgrade, new waifu or AURA agent, to an AGI-stage reveal**.
   - Guesses range from a **browser/operator upgrade**, new *waifu* or *AURA* agent, to an **AGI-stage reveal**, while one member dismissed it as *repetitive hype* in comparison to **xAI’s progress**.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Radix Sort Tricks Unlocked**: Members discuss [parallel **Radix sort** implementation on GPUs](https://www.amazon.ca/Programming-Massively-Parallel-Processors-Hands/dp/0323912311/), referencing a book with a **2-bit implementation** exercise.
   - The task is to extend the given concept to an **8-bit implementation**.
- **Serverless GPU Profiling Headache**: Engineers sought advice on **serverless GPU platforms** for remote code execution and CUDA profiling, especially for **tiled memory**.
   - While **Modal** was suggested, users seek tools with more advanced profiling than is currently offered, as full profiling access (**sudo** privileges) is generally restricted on shared GPU platforms due to security exploits.
- **China's H20 Sparks GPU Chatter**: Members discussed the news around **China H20**, comparing it to **H100** in terms of interconnect bandwidth.
   - The **H20** is seen as inferior for training compared to **GB200/GB300**, with community joking about the inevitable confusion between **NVL144** and **NVL72** configurations.
- **Voltage Park Beefs Up Team**: **Voltage Park** is seeking a **Software Engineer** to help build their **AI Factory** software stack, and is looking for someone who can help build the core infrastructure.
   - US-based applicants are preferred, with the job posting found at [Voltage Park Careers](https://www.voltagepark.com/careers?ashby_jid=2e463e6a-abc6-48ae-8060-8452c55b2fab).
- **SemiAnalysis Podcast Mentioned Forum**: A new member found the forum because Dylan from **SemiAnalysis** mentioned it on a podcast, and a [Google Meet link shared](https://meet.google.com/wdk-yipf-zjd?authuser=0&hs=122).
   - Another member published the initial version of an **LLM RL environment framework** for long-horizon tasks, found on [GitHub](https://github.com/ritser-labs/real-work) and [X.com](https://x.com/ritserlabs/status/1945494003803148565).



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Kimi K2 Triggers Mania**: The community is excited about **Kimi K2**, with some saying it's having its *DeepSeek moment* and hoping to download and host it locally to avoid paying for **Claude 4** or **ChatGPT**.
   - It is speculated to lead to *economic freedom from not paying rent to Sam and Dario* and owning your own model.
- **H200 vs H100 Cost Comparison Evolves**: **H200s** are almost the same price as **8xH100s**, with **B200s** spotted for as low as **$2/GPU/hour** at [celiumcompute.ai](https://celiumcompute.ai).
   - It was clarified that the low price was a limited time promotion via *deepinfra*.
- **Manus Speculated to Replace Claude**: The speculated release of **Manus** may incorporate **Kimi K2** agentic capabilities, potentially replacing **Anthropic Claude** codings for geopolitical reasons, according to [twitter](https://x.com/Teknium1/status/1945259797517099126).
   - It's believed to be a *strong frontier level alternative to Claude*.
- **Nvidia Publishes LLM RL Framework**: Nvidia published the initial version of an [LLM RL environment framework](https://research.nvidia.com/labs/adlr/AF3/) for long-horizon tasks, making it easier to set up environments in a Docker container with access to tools and generate trajectories.
   - The framework, called *real-work*, is available on [GitHub](https://github.com/ritser-labs/real-work).



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Sidesteps Image Generation**: While **image input** is supported in LM Studio, the team currently does not plan to add **image generation** capabilities.
   - The team prioritized other features.
- **Custom Model Search Repo URL's Users Request**: A user requested the ability to specify a custom URL for the **Model Search repo** rather than being limited to **Hugging Face**.
   - Members stated that downloading models manually and importing them is the current workaround, since there is no way to switch away from **Hugging Face**.
- **LM Studio Has No Public Roadmap**: A user inquired about a **public LM Studio development roadmap** to understand the future direction of the project.
   - Another member responded that there is no public roadmap.
- **Memory Features May Arrive in LM Studio**: Users discussed the potential for **memory features** in LM Studio, similar to those found in **ChatGPT**, which would allow chats to reference previous interactions.
   - A member suggested leveraging an **MCP with memory capabilities**, similar to the existing **rag-v1** and **code-sandbox mcps**.
- **LG's EXAONE License Limits**: The community debated the restrictive nature of **LG's EXAONE license**, particularly the requirement to include *"EXAONE"* in all released models and its limitations on commercial and research use.
   - Concerns were raised about the definition of *"research"* under the license, especially regarding reverse engineering and distilling, leading some to deem the license contradictory and difficult to enforce.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune Transitions to Fresh Product**: The **torchtune team** announced in [a GitHub issue](https://github.com/pytorch/torchtune/issues/2883) their plans to evolve **torchtune** into a *new product* within a *new repo*.
   - They will continue to support the community on **Discord** and **Github**.
- **Torchtune's License Inspires Confidence**: A member inquired about **intellectual property** when using **Torchtune's** components, another member pointed out that [Hugging Face already uses it](https://github.com/huggingface/trl/blob/640a9f39164ce3f9bbac7d325c1149ba023314e6/trl/models/activation_offloading.py#L19) within **TRL**.
   - This leverages the permissive **BSD 3 license**.
- **Quantum Computer Lands in Ohio**: A member noted that they work at **Cleveland Clinic**, *the only hospital in the world with a quantum computer lol*.
   - Another member humorously noted that the quantum computer is *in the middle of the cafeteria... in ohio??*
- **NFTs Pave Checkpointing**: Members jokingly suggested future technologies like distributed **RL**, **blockchain**, and **quantum computing** for checkpointing and its costs.
   - One user recommended to *create an NFT* for your checkpoint.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Kimi K2 Runs Locally**: **Kimi K2**, claimed as the world's most powerful open-source model, can now run on local devices and can be found [here](https://huggingface.co/unsloth/Kimi-K2-Instruct-GGUFit).
   - Members recommend following proper channel etiquette when sharing.
- **Decoding Qwen-1.5's Inference**: A user is seeking the exact structure and inference technologies used by **Qwen-1.5**, and the discussion pointed to a [relevant file](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py).
   - Another member suggested floating point errors might be the cause of the discrepancy.
- **ETHOS Quantization Techniques Shared**: A member shared [ETHOS](https://github.com/wrmedford/ETHOS), diving into **LLM quantization techniques**, and linked a [YouTube video](https://youtu.be/0pF6GdbwMo4?si=swVldbUTY5Gn4mYB) on the topic.
   - The accompanying [PDF](https://cdn.discordapp.com/attachments/897390720388825149/1394879190049882132/ETHOS.pdf?ex=687912ba&is=6877c13a&hm=8c3b1d564877e4310662d28d5a240c65d01f81b2e66098066acc531c259b6cd5&) delves into a deep dive into LLM quantization techniques.
- **French Deep Learning Course Adds AI Features**: A member announced new features to their **French Deep Learning course** project including **AI-generated QCM** (multiple-choice questionnaires).
   - Resources available on the [course website](https://simonthomine.github.io/CoursDeepLearning/) and [GitHub repository](https://github.com/SimonThomine/CoursDeepLearning/).
- **Ukraine Translator Lightweight and Ready**: A member shared a new **lightweight model** for **English to Ukrainian machine translation** fine-tuned with **40M samples** from **53.5M** using the recently published **LFM2 model**.
   - The model (**350M params**, requiring **1GB of RAM**) achieves a **BLEU** score of **27.24** on **FLORES-200**, with the model available on [Hugging Face](https://huggingface.co/Yehor/kulyk-en-uk) and a [demo space](https://huggingface.co/spaces/Yehor/en-uk-translator).



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Zuckerberg's Behemoth Betrayal Bombshell?**: Some members are accusing **Zuckerberg** of betrayal due to the potential restricted release of the **Behemoth model**, citing [this tweet](https://x.com/signulll/status/1944851904888234293) as evidence.
   - Some posited that **Meta** may follow **Google's** strategy by releasing smaller, open-weight models like **Gemma** as an alternative.
- **Decoding the Dark Arts: Inferring Closed Models?**: Members suggested that inferring closed models from open-weight models could be a worthwhile research direction, especially given [this paper](https://arxiv.org/abs/2407.18384).
   - The member stated *Most people don't run those sorts of models locally, but still sucks*.
- **GPUHammer Strikes, Memory Corruption Massacre!**: The [GPUHammer paper](https://www.arxiv.org/abs/2507.08166) investigates memory corruption and the susceptibility of data structures to such issues.
   - Members are actively working on **data structures and algorithms** susceptible to memory corruption.
- **Muon Optimizer Muscles In**: The **Muon** optimizer appears in [this paper](https://arxiv.org/abs/2507.08166) and [this video](https://www.youtube.com/watch?v=4bFDPVe6BHs), aiming for tool usage comparable to **Claude 4**.
   - Initial tests show promise, but further evaluation is needed to validate its effectiveness in diverse scenarios.
- **MoEs Maximize Memory, Minimize Movement**: Labs are leveraging **Mixture of Experts (MoEs)** to optimize for memory bandwidth, suggesting it's a critical bottleneck as shown in [this video](https://youtu.be/JOqLp1adGO4?si=hUAnREYY5CQoeoaQ).
   - It seems **MoEs** potentially enable training with fewer GPUs than dense models through efficient resource utilization.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Vertex AI displays Thinking Tokens**: A user discovered that running `/help` with the `openrouter/google/gemini-2.5-pro` model in Aider enabled the display of thinking tokens and the `/think-tokens` command also enables the display of thinking summaries.
   - The user referenced [Paul Gauthier's tweet](https://x.com/paulgauthier/status/1932068596907495579) regarding configuring **32k thinking tokens** for **Gemini**, while trying to use **Gemini 2.5 Pro** through Vertex.
- **Ghostty and Kitty gain Terminal App Status**: Users discussed terminal recommendations for Aider, with **Ghostty** and **Kitty** suggested for better performance, though **GNOME terminal** is considered sufficient by some.
   - One user experiencing issues with Aider screen refreshing was advised to try Ghostty, while another recommended **Alacritty** despite its difficulty with image display protocols.
- **Kimi K2 with Groq Emerges as Frontrunner**: A user reported that **Kimi K2 with Groq** is delivering phenomenal performance, with **200-300 tokens/sec**, and high-quality output, rivaling **GPT-4o** and exceeding **Sonnet 3.7**.
   - They highlighted its affordability and speed, making it their preferred choice, echoing positive feedback from others in the community.
- **Aider Benchmark Refresh**: Users suggested that Aider should update its benchmark, given that many models now score above **80%**.
   - The proposal included creating a private benchmark where users can contribute their own tests.
- **SwitchPoint Router gains interest**: A member inquired about the [SwitchPoint Router on OpenRouter.ai](https://openrouter.ai/switchpoint/router), which routes requests to models like **GPT-4.1**, **Claude 4 Sonnet**, and **Gemini 2.5 Pro** at potentially lower rates.
   - The router's website boasts an **80%** result on the *aider polyglot benchmark*, prompting further discussion.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Docs Tabs Trick Inspires Experimentation**: A member discovered the **Google Docs' tab feature** as a novel method for managing sources within **NotebookLM**.
   - The user expressed interest in the application of the feature to the platform.
- **Ad Blocking Extension to the Rescue**: A member suggested utilizing the **uBlock browser extension** as a method to remove ads and unwanted elements when copying news articles into **Google Docs**.
   - They noted that additional filters for annoyances and social media popups can be added in the extension's settings under the *Filter list* tab.
- **Users Express Frustration over Featured Notebooks**: Users are expressing frustration with being forced to view **"Featured Notebooks"** without an option to remove them.
   - They expressed a desire for a more customized feel and organization in **NotebookLM**.
- **Podcast Length Dictated by Language**: A user reported that their podcasts were consistently short, around **7-10 minutes**, due to an incorrect language setting.
   - Another user pointed out that the option to select "long" podcast output is available for **English**, resolving the issue.
- **"Service Unavailable" Plagues Users**: Some users are encountering a **"Service unavailable"** error message without sufficient context when using **NotebookLM**.
   - The error indicates that the user is trying to access a service not available for their account.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus AI Mobile App Creation Mastery**: A user reported success in using **Manus AI** to develop mobile applications on various topics with customized designs for just **100 credits**.
   - They volunteered to assist others facing difficulties with **Manus** app development.
- **Free Manus Alternative emerges for Vehicle Creation**: A user claimed to possess a **100% free alternative** to **Manus** with similar functionalities, utilizing it to create vehicles for the **OMSI 2** bus simulator game.
   - They suggested the possibility of generating a script for **Google Collab** to produce the necessary file, contingent on the model used.
- **Next-Level AI claims to Outperform Manus**: One user claimed to have developed an **AI** system that exceeds **Manus** in benchmark performance.
   - They offered lifetime access to the first **100 beta testers**, inviting users to DM them to secure their spot.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Discord Channel Expansion Sparks Modular Forum**: A member suggested creating a new Discord channel for user project sharing, but a staff member proposed using the **Community Showcase** category of the [Modular forum](https://forum.modular.com/c/community-showcase/8) instead.
   - This suggestion aimed to consolidate project showcases within the existing forum structure.
- **Mojo requests library needs TLS**: A member inquired about a native **Mojo** *requests* library, and another suggested that the main blocker is **TLS support**.
   - One member shared their [toy requests-like library](https://github.com/thatstoasty/floki), including **TLS bindings**.
- **Decoding Escaping: Not just for prison breaks**: A member sought clarification on the usage of the *escaping* keyword in **Mojo**, noting a lack of specific documentation.
   - Another member pointed to the [Changelog](https://docs.modular.com/mojo/changelog#v070-2024-01-25), clarifying that *escaping* performs a `__copyinit__` of values instead of capturing by reference.
- **Parameter decorators are captured!**: A member inquired about `@parameter` functions (capturing), and another provided a [link to the manual](https://docs.modular.com/mojo/manual/decorators/parameter/#parametric-closure) dedicated to it.
   - They also mentioned exciting news in the **Q3 roadmap** regarding *Unified @parameter and runtime closures* ([roadmap-update](https://forum.modular.com/t/mojo-q3-roadmap-update/1957)).



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Setitem PR Faces Scrutiny**: A member sought review for their [setitem PR](https://github.com/tinygrad/tinygrad/pull/11260), questioning the overhead of the implemented solution.
   - The reviewer suggested the `tensor.py` change should focus on removing the `realize` call and addressing it at lower levels.
- **Optimization Impasse: Parameter to Assign**: A discussion arose regarding the possibility of adding a parameter to assign in order to let users specify ranges and indices.
   - However, a reviewer deemed it *not worth it* and clarified that it's *not what the bounty is looking for*.
- **Proposed Fix for realize() Removal Unveiled**: In response to concerns about assignments not persisting when only removing `realize()` calls, a member proposed modifying the line to `self.uop = res.assign(v).uop`.
   - Alternative proposals included `self.assign(v, indices)` or `self.uop = reconcileAssignment(self.uop, res.assign(v).uop`.
- **Tinygrad Tensor Hooks: Seeking Wisdom**: A user inquired whether **tinygrad** supports tensor level hooks, with the aim of fetching hidden states in a large language model (**LLM**).
   - The user is exploring methods to extract and utilize the **hidden states** during model execution.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Meets Snowflake in Amsterdam**: LlamaIndex is collaborating with **Snowflake** for a meetup in Amsterdam on July 31st, focusing on building high-quality data agents in production, as announced [on Twitter](https://t.co/fFJvvIWrw4).
   - The meetup aims to gather engineers interested in data agents in production environments.
- **UiPath Embraces LlamaIndex Agents**: **UiPath** now supports LlamaIndex agents, enabling seamless deployment into enterprise environments with new coded agents support, as detailed [in their announcement](https://t.co/ILez3d6Zrs).
   - Features include *full code-level control* with the **UiPath's Python SDK** and the ability to build custom agents pulling data from enterprise systems, offering developers greater flexibility.
- **RAG System Tips Shared by Engineer**: An Open Source Engineer shared tips for building production-ready RAG systems, particularly on **text extraction strategies**, as seen [in this thread](https://t.co/R0TTgWrKtv).
   - The discussion covered when to use *simple parsing versus advanced OCR-based solutions* like **LlamaParse**, advising on optimizing extraction methods.
- **ODSC Summit Spotlights LlamaIndex Agent Building**: A LlamaIndex member is presenting a workshop at **ODSC's Agentic AI Summit**, instructing attendees on building agents using LlamaIndex, with more information [available here](https://t.co/6jcYIGR70s).
   - Participants will learn to *create autonomous applications* that use goals and tools to accomplish tasks independently, enhancing their agent-building capabilities.
- **LLM Fine-Tuning Guide Seeks Feedback**: An engineer shared an **LLM Fine-tuning Guide** MVP, seeking developer feedback on practical advice for data preparation, parameter selection, and model evaluation, showcased [on Vercel](https://ai-finetuning-advisor-g3r5.vercel.app/).
   - The guide aims to provide step-by-step instructions to streamline the fine-tuning process.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **IReRa's Role in Hierarchical Labels**: A member questioned the advisability of using **IReRa** for a use case involving **hierarchical labels** with **3 levels**, **440 parents**, and **3500 grandchildren**.
   - Another member suggested that **multiple modules** might be more effective, especially when dealing with a limited number of labels in each step and employing large language models (**LLMs**).
- **Vanilla DSPy Tackles Parent-Child Identification**: A member proposed using **vanilla DSPy** to first identify the parent, then move from parent to child to grandchild in a hierarchical structure.
   - Another member confirmed successful implementation of a similar approach with **3 parents** and **28 children**.
- **Nova Prompt Optimizer and DSPy Entanglement**: A member verified that the [aws/nova-prompt-optimizer](https://github.com/aws/nova-prompt-optimizer) requires `dspy` as a dependency.
   - Further research on the interaction between the two may happen in the future.
- **Lean 4's Verification Potential**: A member suggested using **Lean 4** to verify something.
   - They shared a [YouTube video](https://www.youtube.com/watch?v=1067jj67toY) on **Lean 4** for more context.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Students Reprimanded after Missing Certificate Declaration Form**: The course organizers are unable to accommodate students who missed the **certificate declaration form** deadline due to limited staff capacity.
   - One student pleaded to reopen the [declaration form link](https://forms.gle/iPA2MUpHdtrBE1vu5) but their request was ultimately denied.
- **Students Seek Feedback on Lab Submissions**: A student inquired about receiving feedback on their **lab submission performance** and exploring additional research directions.
   - The student expressed satisfaction with their submission but sought further discussion with someone about their performance.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Anthropic Connector Directory Opens MCP**: Anthropic launched a ["connectors" directory](https://claude.ai/directory), aiming to introduce **Model Context Protocol (MCP)** to a broader audience, potentially increasing demand.
   - Unlike the **Docker toolkit** designed for developers, this directory targets a broader audience, including product managers and marketers.
- **Glasses 👓 Gives AI Sight!**: A member unveiled **Glasses** 👓, a new **open-source tool** implementing the **Model Context Protocol (MCP)**, enabling compatible AIs to request screenshots of URLs and emulate different screen sizes, now available on [GitHub](https://github.com/gourraguis/glasses-mcp).
   - The tool will give AI Agents the power of sight, improving capabilities with external websites.
- **Goose Supports Multi-Agent Systems**: **Goose** now supports subagents, enhancing flexibility and enabling multi-agent systems, showcased with [Codex CLI as a subagent](https://block.github.io/goose/docs/experimental/subagents).
   - **Goose** supports **Anthropic Claude Sonnet-4**, **OpenAI**, **Gemini**, **Ollama**, facilitating autonomous orchestration by coordinating a main task and subagents, then merging results, streamlining complex workflows.
- **MCP Inspector Fails to Reload Resources**: A member reported that the **MCP Inspector** does not reload resources after updating a profile and calling `server.sendResourceListChanged();` from within a tool using the **mcp typescript-sdk**.
   - They found that a resource refresh within the tool fails to reflect updates unless the resource list is cleared and relisted.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Cloudflare R2 Throws Tantrum**: A member encountered an **Access Denied** error when trying to download the dataset from **Cloudflare R2** using a specific `aws s3 ls` command.
   - The command used was `aws s3 ls --endpoint-url=https://9fa58365a1a3d032127970d0bd9a1290.r2.cloudflarestorage.com/ s3://contrastive` for fine-tuning the model.
- **GPT4ALL's Neural Neutrality?**: A member questioned whether **GPT4ALL** processes input based on raw logic, reasoning, and output, taking a user-first approach.
   - They inquired whether **neutrality guidelines and safety filters** are still implemented.
- **AI Engineer Rides the Web3 Wave**: An AI and Web3 software engineer is seeking opportunities in startups, research teams, or automation, bringing experience in building autonomous systems.
   - Their skillset includes **Python**, **TypeScript (React/Next.JS)**, **C/C++**, **LangChain**, **ReAct**, **OpenAI**, and **Solidity/Rust**.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **DeepSeek Production Event Soon**: Packt is organizing a **DeepSeek in Production** event to discuss the model's speed and efficiency with top engineers and researchers.
   - The [Eventbrite link](https://www.eventbrite.com/e/deepseek-in-production-tickets-1436251630289?aff=oddtdtcreator) details hands-on workshops for fine-tuning **DeepSeek models** using **LoRA + Unsloth** on consumer GPUs.
- **DeepSeek Models Get Closer Look**: The **DeepSeek in Production** event will cover technologies like **MoE**, **MLA**, **FP8**, and **MTP** to explain the model's performance benefits.
   - One participant notes the event is promising due to **DeepSeek's** strong open-source support.
- **BERT Extractor Model Faces Headwinds**: A user is developing a **financial keyword/key sentence extractor with BERT** to pinpoint company information like **Registered Address**, **Province**, **Registration Date**, and **Business Term** from financial summaries.
   - Their preliminary method, employing **cosine similarity** between sentence embeddings and task-specific embeddings, *did not yield satisfactory results*.
- **Extractor Model Seeks Alternative**: The user is requesting advice on enhancing their **BERT-based financial keyword extraction** model following the failure of the initial cosine similarity approach.
   - They are exploring different strategies to precisely identify crucial information, such as **registered addresses** and **registration dates**, from company financial summaries.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Claude Sonnet 4 Surfs Back with Anthropic Support**: **Claude Sonnet 4** is back with first party support from **Anthropic**, as announced [on X](https://x.com/windsurf_ai/status/1945599013954490523).
   - **Pro/Teams** users get a discounted **2x** credit rate for **Claude Sonnet 4** and **Claude Sonnet 4** (Thinking) in the **Editor** and **JetBrains Plugins** for a limited time.
- **Wave 11 Topic Inclusion Polled**: A poll [on X](https://x.com/windsurf_ai/status/1945263147994243294) asks the community whether to include a certain topic in **Wave 11**.
   - Members can interact with the post to voice their opinion.



---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/)** (1 messages): 

OpenAI: @everyone
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1394770314420027444)** (1222 messages🔥🔥🔥): 

> `DeepSeek Censorship, IQ Testing, AI's Role in Society, GPT-5 Speculation, AI and North Korea` 


- **DeepSeek's government Allegiance Raises Eyebrows**: Members discussed **DeepSeek AI** and its tendency to favor the Chinese government, leading to heavy censorship.
   - Concerns were raised about DeepSeek's censorship, with one member stating, *other LLMs aren't that censored compared to DeepSeek*.
- **IQ tests provide metrics, Mensa arrives in Elementary schools**: Members discussed the usefulness of **IQ tests** and their ability to measure genius, with some mentioning their own scores and standard deviations, while sharing takes from [a Mensa IQ test](https://test.mensa.no/Home/Test/en-US).
   - A user added, *schools sometimes use it on kids to try and distinguish gifted individuals*.
- **AI's potential role in shaping the future leads to existential discussions**: Users debated whether **AI** will enhance or diminish human capabilities, touching on job replacement and productivity.
   - One member expressed concern about the future, saying: *AGI will be our doom and extinction level event.*
- **OpenAI Gears up for a new announcement that excites community**: The community speculates upcoming OpenAI announcements and some are expecting a new browser, learning feature, new agents, or even **GPT-5**.
   - A member anticipates, *Prob the browser right* and linked to the [OpenAI Twitter announcement](https://fxtwitter.com/OpenAI/status/1945607177034760574).
- **North Korea's Disconnected Reality Amazes AI Engineers**: Members shared shocking facts about **North Korea's** isolation and lack of technology, with one noting that citizens are taught *South Koreans eat babies*.
   - They remarked, *It’s like they are living in another universe* and highlighted restrictions on internet access and social media.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1394759781176316075)** (5 messages): 

> `Coding Libraries for AI, Future OpenAI API Integrations, Pro Membership Value` 


- **Members Mull OpenAI API Integration Wait**: One member expressed they will *wait till they integrate it into the API* and might get rid of the **pro membership** since it doesnt offer much.
   - They said that *operator was one of the main reasons I got it since it seems that they dont really work on it, and its not well integrated* and might not use it.
- **User Seeks AI Coding Libraries Location**: A member asked *where the coding libraries for AI are stored to teach it a new coding, how and where*, like **Jan AI**.
   - No response was provided in the messages.


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1395102401484161074)** (1 messages): 

> `Aravind and Leonid Reddit AMA, Comet browser` 


- **Perplexity CEO and Product Lead Host Reddit AMA**: Aravind Srinivas (CEO) and Leonid Persiantsev (Comet Product Lead) hosted a Reddit AMA on r/ChatGPT starting at **11am PT** to answer questions about the **Comet browser**.
   - They discussed why they built the browser, how it’s evolving, key features, use-cases, and what’s next for Perplexity; the AMA can be found at [this link](https://www.reddit.com/r/ChatGPT/comments/1m1javp/comet_ama_with_perplexitys_aravind_srinivas_and/)
- **Comet Browser AMA Highlights**: The AMA covered the rationale behind building **Comet**, its evolution, key features, use-cases, and future plans for Perplexity.
   - Participants gained insights into the strategic vision and development roadmap for the **Comet browser**.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1394755456031002765)** (1020 messages🔥🔥🔥): 

> `Comet Browser, Image Generation Issues, Samsung Galaxy Store Free Pro, Grok 4 Availability, Comet Agent Saved Interactions` 


- **Perplexity Image Gen Outage**: Members report issues with **image generation**, with some receiving text-based responses instead of images; a member noted that **the image generation feature is temporarily unavailable** on the web as the team works on improvements and [Perplexity support confirmed the outage](https://discord.com/channels/1047197230748151888/1394956308184170516).
   - It seems like the error only occurs when you use "generate image of", but with "an image of" it works.
- **Comet Integrates Google Apps Seamlessly**: Comet browser integrates with Google apps with gemini and is completely free, syncing with your **gmail account, emails, and calendar**.
   - Users can analyze e-mails, interact by setting appointments, and it even gets access to Gmail attachments.
- **Samsung Gives 12 Months Free Pro to Users!**: Members share a deal for 12 months of **Perplexity Pro** through the Samsung Galaxy Store for users in the **USA**.
   - However, there are [reported risks](https://tenor.com/view/shrug-what-huh-will-smith-i-mean-gif-3535627793955785136) of account suspension if users are not based in the USA.
- **Free Tier Might Get Ads?**: Users speculated about the possibility of ads on the free tier, and [Sam affirmed it to block assumptions](https://www.reddit.com/r/ChatGPT/comments/1m1javp/comment/n3hnolo/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) that Perplexity wants ads to be visibly separate from LLM responses, and entirely separate from LLM output.
   - Members suggest non-intrusive ad placements such as normal ads in the UI or sponsored ads while searching.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1394844669124280421)** (5 messages): 

> `Shareable threads, Audio Overviews` 


- **Shareable Threads Reminder**: A member was reminded to ensure their thread is *Shareable* with a link to [the original message](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825).
   - Several Perplexity AI search links were shared including [review this channel](https://www.perplexity.ai/search/review-the-following-channels-woHqzlFoQL6evyGMV8NpAA) and [America's hidden hand in Ukrain](https://www.perplexity.ai/page/americas-hidden-hand-in-ukrain-YKMtnm5EQXaYAIBf25e5Lw).
- **Audio Overview alternative**: A member shared an [Audio Overview version](https://notebooklm.google.com/notebook/8586d048-e5bd-4a3c-acff-ed4660d70c8b/audio) of a research link.
   - They followed up with a link to a **Perplexity AI** search [postavil cachyos s optsiei no](https://www.perplexity.ai/search/postavil-cachyos-s-optsiei-no-ueINCgXNS1iJh7yMvZZymg#0).


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1395146715241250897)** (2 messages): 

> `Perplexity Pro, API access` 


- **Perplexity Pro now gives API access**: Users were confused whether **Perplexity Pro** gives them **API access**.
   - One member linked to the [Perplexity AI help center](https://www.perplexity.ai/help-center/en/articles/10352901-what-is-perplexity-pro) which stated that **Perplexity Pro** gives **$5 monthly credit** to use on **Sonar models**.
- **API Access Details**: The API access provided with Perplexity Pro allows users to embed AI-powered search into their own projects.
   - This access includes the ability to obtain citations, enhancing the utility for research-oriented tasks.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1394756021410332772)** (552 messages🔥🔥🔥): 

> `Cursor billing and pricing, Kimi K2 model discussion, Cursor performance issues and troubleshooting, Multi-agent collaboration and code management, Context engineering for effective AI use` 


- **Cursor's Credit Crunch: $20 Ain't What It Used to Be**: Users are reporting that Cursor's usage dashboard can [lag](https://cursor.sh/dashboard), showing "included" even after the **$20** credits are exhausted, and the billing only catches up in the next cycle, and users now receive in-chat notifications upon reaching the billing point, though sometimes delayed.
   - Some users are finding **Ultra plan limitations** too restrictive, questioning its long-term viability, hitting limits quickly, like one user exceeding **$60**, while another joked about starting an Ultra plan on the same day.
- **Kimi K2: The Groq Sensation Stealing Sonnet's Thunder?**: Members are heavily discussing **Kimi K2** via **Groq** as a potentially superior and cheaper alternative to **Sonnet**, with claims of **Opus-level** agentic task performance, longer context window of **256K tokens**, and a speed of **250-300 tokens per second**, noting that some report that Kimi k2 on Groq isn't as good at tool calls as on Moonshot.
   - However, Kimi K2 does not accept visual input, which some consider a drawback.
- **Cursor's Composer Conundrums: Stuck Prompts and Usage Woes**: Users are experiencing issues with Cursor getting stuck on prompts, wasting usage, and a member reported that the new build should have the **180-second timeout**, which should auto-cancel a stuck request and not bill the user.
   - Some users suggest restarting Cursor, reinstalling, or using a new prompt to *rethink* as potential solutions. When it's stucks or seems repeating same issue, I just write a new prompt to rethink or similar, so its kinder *retriggers* and then works.
- **Agents Assemble! Multi-Agent Systems Take Center Stage**: A user is developing a **multi-agent MCP server** for managing asynchronous agentic tasks across multiple IDEs, aiming to solve the persistence of state issue, sharing that they're using **gemii** for debugging, **grok4** for heavy refactoring, and **Claude** for coding.
   - Another user found it efficient to edit Cursor's `main.js` and other IDEs to create a bare minimum MCP, using JSON for input and output, connecting them all upon launch. One user added it would be funny af if i get it to work though 😆.
- **Context Engineering: The Secret Sauce for Savvy Agents**: Users are discussing the importance of providing the right context to AI agents, or **Context engineering**, with tips to provide all necessary info in the first prompt, while others emphasize that its better to have a simple set of rules, or ref docs, in the System Prompt.
   - It has been observed to be negative if you give the model too much context, as Too many rules may cause the context to loosen up in the long term


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1394882909290102814)** (8 messages🔥): 

> `Cursor runtime env, Customize PR Title, Start Script output not shown, Reconfigure manual snapshot error, Cursor fetching background agents` 


- **Cursor's fix works runtime!**: The latest fix in Cursor resolved issues for a user, confirming that **fixes are available in the runtime environment** but uncertainty remains about their availability during interactive environment setup.
- **Configure PR titles?**: One user reported difficulties while trying to customize the **PR title**.
- **Start script output hidden?**: A user questioned why the **start script output isn't displayed**, noting that while install outputs are visible, the start output is not, despite containing valuable information.
- **Snapshot reconfiguration fails!**: A user reported encountering an error when clicking **reconfigure manual snapshot**, which leads to a rebuild of the environment from a blank slate.
- **Cursor's Agent Fetching Fails**: One user reported waiting 8-10 minutes for **background agents** to be fetched.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1394756563171938434)** (267 messages🔥🔥): 

> `BF16 vs FP32 for LoRA fine-tuning, Qlora as a constraint, Gemma 3 pretraining, Kimi audio distillation, Overfitting solutions` 


- **BF16 saves VRAM over FP32 for LoRA Training**: When fine-tuning **LoRA**, using **bf16** is preferable over **fp32** unless using **Gemma 3** on older GPUs, as bf16 reduces VRAM usage.
   - One user noted that a **7b model** in **fp32** can take up to **28GB** of VRAM, which can be problematic with limited hardware like an A100.
- **DeepInfra offers B200 GPUs for cheap**: Some users discussed renting **B200 GPUs** from [DeepInfra](https://deepinfra.com/low) at a promotional rate of **$2/hour**.
   - The promotion initially applied everywhere except Nebraska, potentially due to [a tweet](https://x.com/DeepInfra/status/1935459646493573123) needing correction.
- **Fine-tuning for Overfitting**: A user sought advice on mitigating overfitting issues when fine-tuning **llama3.2** with a small dataset, where the model failed to generalize rephrased questions.
   - One user recommended the [Unsloth fine-tuning guide](https://docs.unsloth.ai/get-started/fine-tuning-guide/lora_hyperparameters_guide#avoiding-overfitting-and-underfitting), highlighting that hyperparameter tuning is often a process of trial and error.
- **TurboDerp finds no benefit from Alibaba's lossless 2bit Compression**: There's some hype around [AliBaba's lossless 2bit compression](https://huggingface.co/turboderp/ERNIE-4.5-300B-A47B-PT-exl3) for ERNIE 4.5 models, some users are excited that it could work with gguf and LlamaCPP.
   - However, [TurboDerp](https://huggingface.co/turboderp/ERNIE-4.5-300B-A47B-PT-exl3) showed that the actual average was more like 2.5 bit and some layers still at higher precision, and that the true EXL3 performs better.
- **Unsloth is benchmarked higher than Liger-Kernel**: One user benchmarked **Unsloth** versus **Liger-Kernel**, they showed that *Unsloth* saves **15-30%** more VRAM.
   - They also attached an image after they successfully trained, that shows *insane context length with unsloth gradient checkpointing*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1394842278845087804)** (18 messages🔥): 

> `OpenPipe ART, LLM as a Judge, Agentic Models, ARTwell RULER, Model Finetuning` 


- **OpenPipe ART claims to make models more agentic**: A member shared a [link](https://github.com/OpenPipe/ART) to **OpenPipe's ART**, which claims to make any model more agentic.
   - They initially expressed doubt, but later acknowledged the tool was *actually pretty interesting* and uses Unsloth.
- **LLM as a judge**: The tool uses **LLM as a judge**, which one member found cool.
   - One member found the related LinkedIn post doubtful.
- **ARTwell RULER testing incoming**: A member has been looking at **OpenPipe** and its finetuning tools and is looking to try out **ARTwell RULER**.
   - Another member confirmed *it's pretty good*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1394760640626950297)** (166 messages🔥🔥): 

> `Unsloth fixes, VLLM cache, Kaggle Mistral notebook errors, Huge VRAM recommendations, Llama.cpp multi GPU config` 


- ****Unsloth update deemed fixxy****: Members reported issues with the [latest Unsloth update](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs), particularly **Timeout errors** during model downloads.
   - Suggestions included using `huggingface-cli` or setting environment variables like `HF_XET_HIGH_PERFORMANCE` to `1` to mitigate the download issues.
- ****VLLM cache causes many errors****: A user reported getting *corrupted cache errors* when running multiple training scripts that pointed to the same **VLLM cache directory**.
   - Unfortunately, there was no current way to change the VLLM cache directory via environment variables, as it defaults to `.cache/vllm/torch_compile_cache`.
- ****RTX 50 Blackwell manual instructions, no slacking!****: A user inquired about **Unsloth support for RTX 50 GPUs**, and another user replied that it needs manual instructions as seen in [the docs](https://docs.unsloth.ai/basics/fine-tuning-llms-with-blackwell-rtx-50-series-and-unsloth).
   - They said *I don't think it is necessarily a bad idea for that reason; mostly looks like xformers is the only thing needing to be built from source.*
- ****Full finetuning needs precise number crunching****: A user wanted to do a **full finetune of gemma 3 4b** and ran out of memory on 3090, another user replied that *you cant do full finetuning on a 4bit quantized model, it has to be full precision*.
   - Another option included using **16bit LoRA training** and consulting the [Unsloth notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks) for compatible architectures.
- ****Optuna is always the answer!****: A user asked about using hyperparameter tuning libraries with Unsloth like **Optuna** to find the optimal batch sizes and gradient accumulation for their GPU.
   - The response was affirmative, stating, *I don't see why optuna shouldn't work for you*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1394976828631945296)** (1 messages): 

> `Podcast Announcement, Community Engagement` 


- **Podcast Launch Imminent!**: The first podcast episode is launching soon! Check out the [announcement tweet](https://x.com/himanshustwts/status/1945416091582505377) for details.
- **Community Anticipation Builds**: The announcement of the upcoming podcast has generated excitement within the community.
   - Members are eagerly awaiting the release, as indicated by the <:slothhearts:1253009235600736296> reaction.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1394830874054361198)** (13 messages🔥): 

> `ETHOS paper, LLM psychosis, Independent research challenges, TTS models playground` 


- **ETHOS paper arrives on Arxiv**: A member shared their upcoming [Arxiv publication on ETHOS](https://github.com/wrmedford/ETHOS) (Efficient Transformers via Hypernetwork Organized Sparsity) after discussing it with safety folks and obtaining a license control.
   - Attached was the pdf [ETHOS___Efficient_Transformers_via_Hypernetwork_Organized_Sparsity_10.pdf](https://cdn.discordapp.com/attachments/1257011997250424842/1394830874142576712/ETHOS___Efficient_Transformers_via_Hypernetwork_Organized_Sparsity_10.pdf?ex=68798e7b&is=68783cfb&hm=0b2c8891c328a38668ff0d015cf8a9f8ef5b80884a3f7eb0ee486aa149b25e97&).
- **LLM Psychosis is actually a thing**: Some members are calling it "**LLM psychosis**" when someone thinks they discovered something, but it’s just technical gibberish from conversations with sycophantic LLMs.
   - Another member defined *psychosis* as *a mental disorder characterised by a disconnection from reality*, noting that it can be caused by LLM hallucinations reinforcing misconceptions or miscategorizing things.
- **Independent Research Stymied**: A member expressed difficulty in doing actual independent research due to the prevalence of "nuts people", with breakthroughs from those without a PhD or big lab affiliation immediately dismissed.
   - They attribute this to the fact that *95% of people doing research right now are* afflicted with **"LLM psychosis"**.
- **Snortts Indic V0 TTS Playground Launches**: A member shared a playground for testing **TTS models** at [https://snorbyte.com/snortts-indic-v0](https://snorbyte.com/snortts-indic-v0).
   - Users are invited to try it out and ask questions about training or deploying TTS models for inference.


  

---


### **Unsloth AI (Daniel Han) ▷ #[unsloth-bot](https://discord.com/channels/1179035537009545276/1390899684834410536/1394931626638708826)** (24 messages🔥): 

> `Llama.cpp Quantization Errors, VLLM Cache Directory Configuration, Qwen 2.5 7B Inference, Torch Cache Storage and Corruption, Model Size of Qwen 2.5 7B` 


- ****Llama.cpp** Quantization throws **RuntimeError****: A member encountered a `RuntimeError` indicating that the `llama.cpp/llama-quantize` file does not exist.
- ****VLLM** Cache needs separate directories**: A user reported encountering a corrupted cache error when running multiple training scripts with different configurations, as they all pointed to the same **VLLM** cache.
   - They inquired about the possibility of changing the **VLLM** cache directory via an environment variable, noting that it defaults to `.cache/vllm/torch_compile_cache`.
- **Request to run **Qwen 2.5 7B** for inference**: A member asked how to run **Qwen 2.5-Omni-7B** for inference.
- **Torch Cache Default Storage Location**: A user asked where **Unsloth** stores the torch cache.
   - The same user realized they needed to change their `$HOME` variable for different training scripts, and asked about running multiple training scripts with different GPU visibility to avoid torch cache corruption.
- **Model Size of **Qwen 2.5 7B** revealed**: A user inquired about the storage size of a **Qwen 2.5 7B** model.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1394761464719343761)** (66 messages🔥🔥): 

> `ZR1-1.5B Model, Pythia 12B vs 2.8B, TensorFlow Decline, AI Research Management` 


- **ZR1-1.5B Model endorsed for reasoning**: A member suggested using the [ZR1-1.5B model](https://huggingface.co/Zyphra/ZR1-1.5B) for reasoning tasks, depending on the specific requirements.
   - They also mentioned **DeepCoder-1.5B** as a solid option but cautioned about the difficulty of achieving general-purpose reasoning within a **7B** parameter budget.
- **Pythia 12B has bigger vocab size**: A member compared **Pythia 12B** and **2.8B** models and noticed that the **12B** model has a higher vocabulary size.
   - It was clarified that variations in vocabulary size (e.g., **50,688** vs. **50,257**) are often due to training hardware and don't necessarily mean the tokenizer is different.
- **TensorFlow loses favor**: A member expressed the sentiment that using **TensorFlow** is becoming a red flag in job requirements, because many people have *sworn off tensorflow at this point*.
   - Another user agreed, stating that there isn't a good reason to use **TensorFlow** over **PyTorch** or **JAX**.
- **Research Managers Prevent Doomscrolling**: A member joked about the role of research managers, but a thorough response described the benefits of good research managers, like providing hands-on guidance, handling bureaucracy, and seeing the bigger picture.
   - The ultimate goal of EleutherAI is to provide research management to independent/amateur researchers.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1394929212892254291)** (316 messages🔥🔥): 

> `nanoGPT speedrunning, recursion papers, tuning LLM inference-time hyper-parameters, peer review on research, MoE like things in RWKV-8` 


- **Doubts Emerge on NanoGPT Effectiveness**: A member questioned whether a specific method was tested in a **nanoGPT speedrunning** setting, referencing a [controversy](https://arxiv.org/abs/2507.10524) about the effectiveness of **nGPT** when it was released.
   - Another member expressed frustration with **recursion papers**, noting that while they offer iterative changes, they often neglect how to improve recursion itself, potentially leading to *suboptimal* results.
- **Input Injection Boosts Recursive NN Performance**: A member suggested that a top improvement for *how you recurse* is **naive input injection at each iteration**, acting as a [skip connection](https://link.to.skipconnection) to propagate state more easily.
   - They clarified that this involves injecting the **hidden state**, an earlier hidden state, or the original input itself, with smarter mixing methods available beyond simple concatenation.
- **Seeking Peer Review Proves Challenging**: A member inquired about requesting **peer review on research**, particularly for a cross-disciplinary project involving GPU architecture and MoE models, mentioning a [pending arXiv submission](https://arxiv.org/abs/2505.22014) and available code.
   - However, it was clarified that the channel does not accept review requests due to the high volume of *crank* submissions, though discussing research contents is welcome, with a suggestion to share the [GitHub repo](https://github.com/wrmedford/ETHOS) for opinions.
- **Latent Codes Make Experts Ephemeral**: A researcher detailed their approach of storing experts as **latent codes** and recovering them on the fly, sharing a [GitHub repository](https://github.com/wrmedford/ETHOS) and highlighting a training speed of **15K tokens/second** on a GH200.
   - The 20x less FLOPs is the theoretical speedup, not empirical yet, as bottlenecked by suboptimal backward, where *experts only exist ephemerally* and do not receive gradients, autograd is storing intermediates.
- **Small Model Size Beats MoE Model In a Flash**: Members discussed strategies for comparing a new architecture to baselines given limited compute resources, with suggestions including scaling back experiments to toy examples and comparing against a smaller dense model like [GPT-2-small](https://link.to.gpt2small).
   - The consensus was that winning against an MoE with the same total size or a dense model with the same activated parameters would be a significant win, emphasizing the need to tune hyperparameters like learning rate and initialization schemes for honest comparisons, with [Olmo-2-1b](https://link.to.olmo21b) proposed as a solid baseline candidate.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1394938478181220484)** (8 messages🔥): 

> `Function Vectors, nnterp package, Transformer models` 


- **Function Vectors Drive ICL Performance**: A member met the author of [this paper](https://arxiv.org/abs/2502.14010) which argues that **induction heads** don't actually drive ICL performance and that something called **Function Vectors** do instead.
   - Another member defined **induction heads** as *"precisely copying a token"* but found that they played a significant role in *"the general phenomena of using things far back in the context to predict what comes next"*.
- **New Mech Interp Package nnterp Released**: A member released the beta 1.0 version of their mech interp package **nnterp** [on Github](https://github.com/Butanium/nnterp), installable with `pip install "nnterp>0.4.9" --pre`.
   - The package aims to provide a unified interface for all **transformer models** while still using the huggingface implementation under the hood, closing the gap between `transformer_lens` and `nnsight`, with a [demo colab](https://colab.research.google.com/github/Butanium/nnterp/blob/main/demo.ipynb) and [docs](https://butanium.github.io/nnterp/).
- **Robust Testing System in nnterp Validates Models**: The **nnterp** package includes a robust testing system that automatically runs validation tests when a model is loaded to ensure all hooks return expected tensor shapes and that attention probabilities properly sum to 1 for each token.
   - The package includes **1915 precomputed tests** covering toy models from diverse architectures, with clear warnings during model loading if any tests fail.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1395141917066399865)** (1 messages): 

> `Harness Evaluation, IFEval Suite` 


- **Harness Artifacts Scrutinized**: A member questioned whether a certain process fits with the harness, noting that the harness typically doesn't produce external artifacts except for optional caching of model requests, HF Hub resources, or remote code for HF `evaluate` metrics.
   - They called on a specific user to correct them if they were wrong.
- **Dynamic IFEval Questioned**: A member inquired about the advantages of the Dynamic version of **IFEval** over the standard **IFEval** suite, raising questions about reproducibility and determinism in harness evaluations.
   - They noted that most evaluations in the harness are meant to be reproducible and deterministic.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1395051703333687329)** (25 messages🔥): 

> `Transformer Engine performance, Slurm and containers with GPT-NeoX, CUDA drivers in NGC containers, DeeperSpeed Slurm runner` 


- ****TE Performance Plummets?****: A member tested a run without **Transformer Engine (TE)** and found a *significant performance difference* ([wandb link](https://wandb.ai/eleutherai/AISI/runs/nmk3zrpr/overview)), though a *hardware confounder* may have been present.
   - It was suggested that the TE setup might be the issue, particularly whether the supported TE is broken in the current repo `/NS/llm-pretraining/work/afkhan/RoPE_Pct/gpt-neox`.
- ****Slurm & Containers for GPT-NeoX: A Winning Combo?****: Multiple members confirmed that running **GPT-NeoX** via **Slurm** with containers (**Docker** or **Singularity**) *works fine*.
- ****DeeperSpeed Launches Slurm Runner****: **DeeperSpeed** has added a **Slurm runner** that uses `srun` instead of `mpirun`, simplifying multi-node setups ([DeeperSpeed runner](https://github.com/EleutherAI/DeeperSpeed/blob/65d9f99249f79ebd7c4577b6aeb0d3dff5a1cef6/deepspeed/launcher/multinode_runner.py#L413), [GPT-NeoX Slurm instructions](https://github.com/EleutherAI/gpt-neox?tab=readme-ov-file#slurm)).
   - The **GPT-NeoX** repo also provides [instructions for containerized setups](https://github.com/EleutherAI/gpt-neox?tab=readme-ov-file#containerized-setup).
- ****CUDA Driver Caveats with NGC Containers****: A member wondered if NGC containers bring their own **CUDA drivers**, noting their system's default drivers are *older than 12*.
   - When not using containers, they typically manually point to a **12.1** installation, so they are unsure how to apply the same fix inside the container.


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1394878607322644562)** (3 messages): 

> `o1-preview Deprecation` 


- **`o1-preview` shutdown on July 28, 2025**: The `openai/o1-preview` endpoint is deprecated and will be shut down by **OpenAI** on **July 28, 2025**.
   - Newer o-series models from **OpenAI** are available [here](https://openrouter.ai/openai).
- **Deprecation info via API**: A member asked whether it would be possible to get deprecation information via the API.
   - It is uncertain whether this will be implemented.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1394758775361241088)** (357 messages🔥🔥): 

> `Deepseek R1 Quality Drop, OpenRouter AutoRouter Privacy Concerns, Model Deprecation Notices, GPT 3.5 Turbo Endpoint Gone, Claude Opus 4 Weekly Token Usage` 


- **User Exposes OpenRouter's SwitchPoint Router Privacy Blunder**: A member reports that the **SwitchPoint Router** was pre-selected without their consent, potentially violating their **NDA** and sending tokens to China, after adding a firewall rule to ban the OpenRouter Chat, because it's *not safe enough for use*.
   - An OpenRouter admin responded, stating that Switchpoint is **US-based**, not China, and users can disable providers in [settings](https://openrouter.ai/settings/preferences), and the auto router picks between a list of high quality models on OpenRouter.
- **OpenRouter's Usage Metrics Not Always Real-Time**: A user wonders why their **OpenRouter activity** isn't updating in real-time, noting discrepancies between DeepSeek activity and recent API tests with Devstrall and K2.
   - Another user confirms experiencing the same issue.
- **Why OpenAI's GPT 3.5 Turbo Endpoint is Gone**: A user points out the disappearance of the `openai/gpt-3.5-turbo` endpoint, seeking clarification and notes that other providers could've served you. It had successful chess records until **2025-06-23**.
   - An OpenRouter admin responded that they are looking into this issue and have **resurrected** it for future use.
- **User Flags $412 Unauthorized Top-Up**: A user reports an unauthorized **$412** top-up on their account, with a corresponding **404** error when trying to view the invoice, and is seeking assistance with investigating the charge and ensuring account security.
   - An OpenRouter admin explains that it was not a charge, but rather a refund, please check your spam folder.
- **DeepSeek Quality Plummets with Q4 Quantization**: A user notes a significant drop in quality for **Deepseek R1 0528** in roleplay, with the model hallucinating more intensely at lower quantizations.
   - Another user agrees, noting similar issues and recalling a **truly horrible R1 performance**, and is doing tests comparing **Q4** to **fp8**.


  

---


### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1394768983391207535)** (41 messages🔥): 

> `Quality Certification Badges, Speed Certification Badges, Eval Harness for Model Benchmarks, Context Compression Shenanigans, Tool Use Benchmarks` 


- ****OR crafts Quality and Speed Certification Badges****: OpenRouter is exploring quality and speed certification badges for models, similar to `:nitro` filters, to address disparities like **Kimi K2's** varying speeds across providers (**10 TPS** vs **100 TPS** vs **500 TPS**).
   - The aim is to highlight providers with reliable tool calling, consistent ratelimits, and high-quality outputs, while accounting for quantization and potential tool call failures, with new models starting at an unverified tier.
- ****Eval Harness drifts Model Benchmarks****: The proposed eval harness would baseline against model authors' published benchmarks, continuously measuring the *drift* or difference, between official scores and endpoint scores to verify model performance at [OpenRouter](https://openrouter.ai/).
- ****Context Compression Shenanigans Benchmarked****: A long fiction benchmark up to **128k context** is suggested to verify the absence of context compression shenanigans, alongside tests prompting models to output many tokens to confirm providers' stated output token count.
- ****Tool Use Benchmarks Tau-2 airline****: Tool use benchmarks like the [Tau-2 airline](https://github.com/sierra-research/tau2-bench) are recommended, inspired by [tweet](https://x.com/the_bunny_chen/status/1944851548712133032), to detect and resolve tool use chat template bugs.
   - For example, implementing checkers or chess, which test a lot of random things about the model with a very easy evaluation criteria.
- ****Baseten Latency transforms into Retries****: There is an idea for handling **429s** is to transform latency numbers into *expected* latency based on retries, addressing scenarios where prompts take excessively long due to frequent **429s** and high latency, such as **Baseten**.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1394755664672329949)** (337 messages🔥🔥): 

> `Polymarket Bet Failures, Prediction Markets Legality, Grok4 Performance, Kimi K2 Performance and Pricing, LMArena Issues and Feedback` 


- **Polymarket Gambles Don't Pay Off**: Members expressed disappointment after their **Polymarket** bets did not pan out, highlighting surprise that some individuals did not invest in **Google**.
   - One user humorously stated *my ambitious bet on polymarket didn't pay off then 💀I'm shocked that there are actually people in this server who don't buy Google💀*.
- **Prediction Market Regulations in Question**: Users debated the legality of prediction markets in the US, with some suggesting that platforms like **Kalshi** offer a legal workaround despite liquidity issues.
   - A user stated *They are regulated by the CFTC, legal in the US afaik*, while another expressed concerns about potential tax authority issues.
- **Grok4's Chatty Performance Rated Poorly**: **Grok4** faced criticism for being overly verbose and producing lengthy, hidden reasoning, leading to a perceived disconnect between claimed benchmark results and real-world performance.
   - One user commented that *Whoever at xAI thought it's a good idea to make the model output 30k+ hidden reasoning and then respond with 1 word to 1 sentence is an idiot tbh*.
- **Kimi K2's Efficiency Astounds All**: Members discussed the impressive efficiency and coding abilities of **Kimi K2**, noting its competitive pricing relative to **GPT-4.1 mini** and significant performance advantages.
   - A member noted that *The Chinese have absolutely cooked when it comes to efficiency* and the model is able to produce interesting physics sandboxes.
- **LMArena's Issues and UI Musings**: Users reported issues with **LMArena**, including models erroring out and problems with the content filter flagging false positives, particularly with comic book content.
   - Feedback was also given on the new UI, specifically regarding the selection model in direct chat, with a community manager directing users to a feedback thread.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1395088960748912802)** (1 messages): 

> `UI Improvements, Leaderboard Navigation, Streamlined Interface, Compact Sidebar` 


- **Light UI Improvements Launched**: New **UI improvements** are now live, including a **streamlined interface** to reduce clutter, a **compact sidebar** for quicker access to key sections, and **improved navigation** for the leaderboard tab, inspired by community feedback.
   - Check out the video [ArenaV2 Launch Video](https://cdn.discordapp.com/attachments/1343296395620126911/1395088959448809482/ArenaV2_LaunchVideo_Under30s_1.mp4?ex=68792d57&is=6877dbd7&hm=8168774d3683077ff598824196cb6070bb39207f22ac06b02dbc803453f621c0&).
- **Navigation Upgrades Steer Leaderboard**: The leaderboard now has improved navigation, to give faster access to the leaderboards, as part of the UI improvements.
   - The intent of the improvements is to make the overall experience more polished, intuitive, and delightful based on community feedback.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1394789349958357022)** (220 messages🔥🔥): 

> `Kimi UI, Atlassian Rovo Dev AI Agent, Flux Pro and Seedance Realistic AI Videos, Anthropic hires back Cursor developers, OpenAI API Face Editing` 


- **Kimi UI's Bitter Beginnings**: A [Medium article](https://medium.com/@xinyijin715/maker-story-the-bitter-lessons-behind-kimi-researchers-ui-6654ec66662c) shares the story and **bitter lessons** behind Kimi UI development.
   - It appears Cursor code may have something to do with it, according to a member quoting a [YouTube video](https://www.youtube.com/watch?v=motX94ztOzo&t=2737s).
- **Atlassian's Rovo Dev: A Constitutionally Incapable AI Agent?**: Poonam Soni introduced **Atlassian's new AI agent**, Rovo Dev, accessible via CLI, which aims to transform software development with features like **code generation, review, refactoring, documentation, debugging, and task automation**.
   - Despite the hype, one member quipped that *Atlassian is constitutionally incapable of building a good product* while others expressed frustration with the tool's **download process and enterprise focus**.
- **Flux Pro and Seedance Mix for Realistic Video**: A user experimented with combining **'flux pro' and 'seedance' models** to generate realistic videos, using the **'IMG_XXXX.JPG' trick** for start images and prompts like *'aggressively mediocre home footage'*.
   - The discussion included a link to the Replicate model and positive reactions from other users; questions remain about the company behind this tech.
- **Boris & Cat Leap Back to Anthropic**: **Two coding AI leaders** have been rehired by Anthropic after a short stint at Cursor, according to [The Information](https://www.theinformation.com/briefings/anthropic-hires-back-two-coding-ai-leaders-cursor-developer-anysphere).
   - This move prompted humorous speculation about whether they were *double-agents* all along, riffing that Cursor may have received *cheap consulting* from these *world class experts*.
- **OpenAI Teases Operator/AURA**: OpenAI posted a cryptic video teaser for something launching on **July 16, 2025**, sparking widespread speculation from a **browser/operator upgrade, new waifu or AURA agent, to an AGI-stage reveal**.
   - Guesses range from a **browser/operator upgrade**, new *waifu* or *AURA* agent, to an **AGI-stage reveal**, while one member dismissed it as *repetitive hype* in comparison to **xAI’s progress**.


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1395145170672029808)** (1 messages): 

> `YouTube video` 


- **Video shared for the crew**: A member shared a [YouTube video](https://youtu.be/uIKmG3M0X3M?si=ndL7_jJKzI5FKueG) for the crew.
- **Another video**: A member shared another video [here](https://youtu.be/uIKmG3M0X3M?si=ndL7_jJKzI5FKueG) for the crew.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1394757252673568848)** (71 messages🔥🔥): 

> `Radix Sort on GPU, Serverless GPU Platforms, Profiling CUDA Kernels, Industry implementations of RMSNorm and Reduce operators, PyTorch vs efficient CUDA kernels` 


- **Radix Sort's GPU Parallelization Techniques**: A member sought advice on implementing parallel **Radix sort** on a GPU, with another suggesting [chapter 13 of "Programming Massively Parallel Processors"](https://www.amazon.ca/Programming-Massively-Parallel-Processors-Hands/dp/0323912311/) as a resource.
   - The book demonstrates a **2-bit radix sort** and challenges readers to extend the concept to an **8-bit implementation**.
- **Quest for Serverless GPU Platforms**: A member inquired about **serverless GPU platforms** for uploading and running code on a remote GPU, but with the ability to dive into *deeper CUDA*.
   - While **Modal** was recommended, the member was seeking something with better profiling abilities for **tiled memory** and other advanced CUDA features; Google Cloud Run was suggested, but it won't give full profiling access.
- **Profiling CUDA Kernels' Limitations**: Due to security exploits, full profiling access (**sudo** privileges) is generally restricted on shared GPU platforms, impacting the ability to use tools like **nvidia-profiler**.
   - Alternatives include **timing runs** or, as a last resort, asking a *really good* friend to run a kernel with an NCU on their personal laptop.
- **RMSNorm and Reduce Operator Implementations**: For industry-leading implementations of operators like **RMSNorm** or **Reduce**, **CUB** and **rocPRIM** were mentioned for algorithms like 1D reduce, sort, and scan.
   - For **AMD**, the 'professional' implementation of **RMSNorm** is in **MIOpen**, while in CUDA it would be **cuDNN** (closed source); PyTorch also has an implementation, though its kernels are *usually regarded as not very efficient*.
- **PyTorch's Efficiency Trade-Offs**: Despite potentially less efficient CUDA kernels, **PyTorch** is popular because it prioritizes **accuracy over speed**, reducing debugging time during research.
   - While PyTorch's native kernels in **Aten** or **C10** may not be fully optimized, many operations call into efficient libraries like **cuDNN** and **CUB** and users can also swap in their own kernels; here is an [example of user-provided kernels](https://github.com/wrmedford/ETHOS/blob/main/kernels.py).


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1395022405910663378)** (3 messages): 

> `triton-autodiff tool` 


- **Memory-Efficient Backwards Kernel Generation Tool Found!**: A member inquired about tools to generate backwards kernels from forwards kernels memory-efficiently, avoiding wasted intermediate values from autograd.
   - Another member suggested [triton-autodiff](https://github.com/IaroslavElistratov/triton-autodiff) as a solution.
- **Chain Rule Replacement**: User finds tool as a good replacement for remembering the chain rule.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1394807337864134790)** (6 messages): 

> `Torch Compile Debugging, Torch Inductor Issues` 


- **Torch Compile Debugging yields No Output**: When **cache** is enabled, compiled results obtained from the cache won't re-compile anything, hence **no logging** is shown, as the logging is done at compilation time.
   - The environment variable `TORCHINDUCTOR_FX_GRAPH_CACHE=0` may solve the issue of seeing no output from `TORCH_COMPILE_DEBUG=1`, because the cache has been improved to **cache more than before**.
- **Torch Inductor facing problems with Blackwell**: A user mentioned they have been facing a lot of problems with **inductor** lately and have to use nightly (or the branch cut 2.8) since using **Blackwell**.
   - They were unsure if inductor is the problem and asked about recent GitHub issues reporting similar problems.


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1394764261007556738)** (3 messages): 

> `Parallel Radix Sort, Fluid Simulation in OpenGL with CUDA` 


- **Parallel Radix Sort Seeking Guidance**: A member inquired about creating an effective **parallel Radix sort** specifically designed for **signed integers**.
   - The member specified that negative integers should be considered invalid for the sort.
- **OpenGL Fluid Simulation Algorithms Requested**: A member asked for advice on algorithms suitable for **fluid simulation** in **OpenGL**, leveraging **CUDA** for computation.
   - No specific algorithms were recommended in the available messages.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1394768690540708041)** (3 messages): 

> `GPU Engineering Book, Software Engineer Hiring, Technical Reviewers` 


- **Voltage Park Seeks Software Engineers and More**: Voltage Park is hiring for multiple positions, including Software Engineering and Security roles, primarily **WFH** with some global tech support positions; see [Voltage Park Careers](https://www.voltagepark.com/careers).
   - US-based applicants are preferred, but some global technical support positions are available.
- **GPU Engineering Book Needs Technical Reviewers**: A member is writing a book on **GPU Engineering** for **AI systems** for Packt Publishing, covering topics like distributed training, CUDA kernels, and GPU clusters.
   - The editor is looking for technical reviewers; interested individuals can DM [hi@abiaryan.com](mailto:hi@abiaryan.com) for an introduction.
- **AI Factory Software Engineer Job Available**: Voltage Park is seeking a **Software Engineer** to help build their **AI Factory** software stack; the position is fully remote, with an office in San Francisco.
   - The job posting can be found at [Voltage Park Careers](https://www.voltagepark.com/careers?ashby_jid=2e463e6a-abc6-48ae-8060-8452c55b2fab).


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1394945973377044520)** (2 messages): 

> `Colab GPU, Numba, MLE, ML performance engineering` 


- **Colab GPU Eases Numba Use**: A user highlighted that **Colab GPU notebooks** allow immediate installation of **Numba** and direct execution of `cuda.jit` compiled functions.
- **MLE Performance Engineering Entrance Ramp**: A user expressed interest in transitioning into **MLE** / **ML performance engineering** next year and asked where to quickly gain competence.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1394825102071173153)** (12 messages🔥): 

> `China H20, H100 vs H20, NVL72 vs NVL144, Ascend GPUs` 


- **China's H20 Sparks GPU Chatter**: Members discussed the **China H20** news, comparing it to **H100** in terms of interconnect bandwidth which is suitable for inference.
   - One member noted, *"H20s are not that bad cuz it has dame interconnect bandwidth as h100 iirc which is good for inference."
- **H20 Doesn't Hold a Candle to GB200/GB300**: The **H20** is seen as inferior for training compared to **GB200/GB300** and **VL72**.
   - A member stated, *"can't would gb200/gb300 for training right? can't compete yea its def worsenvl72 is a lot better."
- **Confusion Looms: NVL144 vs. NVL72**: The community joked about the inevitable confusion between **NVL144** and **NVL72** configurations.
   - A member quipped, *"it's going to be awesome when we all get confused NVL144 vs. NVL72 you know honestly does it even matter cuz it seems like their own gpus are around hopper level"*.
- **Ascend GPUs Suspected to be Buggy**: Analysis of an image led to suspicion that **Ascend GPUs** may be buggy when scaling up.
   - The image analysis noted, *"i suspect the ascend gpus are super buggy on scale up"*.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1394823971265974322)** (8 messages🔥): 

> `SemiAnalysis Podcast, LLM RL environment framework` 


- **SemiAnalysis Podcast Mentioned Forum**: A new member found the forum because Dylan from **SemiAnalysis** mentioned it on a podcast.
   - A member asked for a link and offered to DM the newcomer, with a [Google Meet link shared](https://meet.google.com/wdk-yipf-zjd?authuser=0&hs=122).
- **Long-Horizon RL environment framework released**: A member published the initial version of an **LLM RL environment framework** for long-horizon tasks, which can be found on [X.com](https://x.com/ritserlabs/status/1945494003803148565) and [GitHub](https://github.com/ritser-labs/real-work).
   - This framework facilitates setting up environments in a **Docker container** with access to tools and generating trajectories.


  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/)** (1 messages): 

complexfilterr: Half of CVPR accepted papers' authors are from China.
  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1394842154223931402)** (6 messages): 

> `GB300 Availability, Coreweave's GB300 Capacity, Nvidia hardware purchase prioritization, DGX vs HGX, B200 Availability` 


- **GB300 Availability at Coreweave Discussed**: Members discussed getting access to **GB300** at **Coreweave**, with skepticism about immediate availability due to **limited capacity**.
   - One member expressed surprise at immediate access, noting Coreweave's recent announcement of **GB300 NVL72s capacity** and potential logistical challenges from Nvidia.
- **Nvidia Relationships Help Hardware Prioritization**: A member mentioned that a *working relationship with Nvidia helps with prioritization of hardware purchases*, citing their own current hardware procurement experiences.
   - They noted that budget is just one factor when deciding between **DGX** and **HGX** offerings, as there are valid reasons to prefer an **HGX** solution due to the modularity of specific hardware components.
- **B200 Availability vs Liquid Cooling**: **B200** is relatively easy to purchase right now, but the more advanced chip configurations require **liquid cooling**.
   - Most data centers are not equipped to facilitate liquid cooling, and **B200** is popular with hyperscalers because they don't have to refit their data centers.
- **Voltage Park Offers Cloud GPUs**: A Solutions Engineer from **Voltage Park** offered assistance in securing GPUs for **AI/HPC/ML workloads**.
   - Interested individuals were encouraged to reach out, with the member's LinkedIn and company information provided in their bio.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1394885843147034766)** (3 messages): 

> `Data Backup, Phone Theft, Learning from Mishaps` 


- **User Recovers Stolen Phone and Learns Backup Lesson**: A user reported that their phone was stolen and recovered, lamenting the lack of a recent backup before the theft.
   - The user noted *they are literally better off after having [their] phone stolen somehow* because now they backup their phone's data.
- **Turning Misfortune into a Win**: The user's misfortune of having their phone stolen led to a positive outcome.
   - Now the user has implemented data backup procedures.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1394945320663650417)** (3 messages): 

> `CuTeDSL, Jetson series, Jetson Orin, Jetson Thor, CUTLASS Python support for NVIDIA GPUs` 


- **CuTeDSL eye Jetson support**: A member inquired if [CuTeDSL](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/) will support the **Jetson series** in the future.
   - Another stated that *CUTLASS Python will support all NVIDIA GPUs*, but the current focus is on **DC GPUs**.
- **Jetson Orin architecture revealed**: A member noted that **Jetson Orin** is an **ARM CPU** combined with an **Ampere GPU** featuring an **sm_87** structure.
   - They guessed that it would be easy to support **Jetson Orin** given that **CuTeDSL 4.0** is rumored to support **ARM CPUs**.
- **Jetson Thor gets Blackwell GPU?**: A member shared that **Jetson Thor** will feature an **ARM CPU** and **Blackwell GPU** with a rumored **sm_101** structure.
   - They requested consideration for adding support to **CuTeDSL**, speculating that it may not require significant effort.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1394755899750355044)** (88 messages🔥🔥): 

> `Kimi K2, H200, B200, Manus release, Model Ownership` 


- ****Kimi K2 Mania** sweeps community!**: The community is excited about **Kimi K2**, with some saying it's having its *DeepSeek moment* and hoping to download and host it locally to avoid paying for **Claude 4** or **ChatGPT**.
   - It will lead to *economic freedom from not paying rent to Sam and Dario* and owning your own model.
- ****H200 vs H100** cost comparison heats up!**: **H200s** are almost the same price as **8xH100s**, with **B200s** spotted for as low as **$2/GPU/hour** at [celiumcompute.ai](https://celiumcompute.ai).
   - However, it was clarified that the low price was a limited time promotion via *deepinfra*.
- ****Data set about** announced**: Members sought details on a newly released dataset to understand how it was generated and its intended applications.
   - The dataset is tied to a [tweet](https://x.com/Teknium1/status/1945259797517099126) that discusses proto-agentic XML tag adherence for proto-reasoning CoTs, diagrams, and step-by-step processing of actions.
- ****Manus's** future revealed**: The speculated release of **Manus** may incorporate **Kimi K2** agentic capabilities, potentially replacing **Anthropic Claude** codings for geopolitical reasons, according to [twitter](https://x.com/Teknium1/status/1945259797517099126).
   - It's believed to be a *strong frontier level alternative to Claude*.
- ****Model Ownership** becomes feasible**: Members discussed the advantages of owning model assets, even if renting servers is necessary.
   - The consensus is that with the advent of frontier-level open-source models, users gain ownership and control, irrespective of whether they run models on their own hardware or rent servers, eventually base models will become commoditized.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1395135215269056522)** (1 messages): 

> `Model Context Size, Adding Personality to Models, Letta (MemGPT) Personas` 


- **Models' Context Size Affects Personality Injection**: The utility of adding personality to models may depend on the **model's context size**, particularly if it's very small.
   - It was suggested to evaluate how different context sizes interact with added personality to see what works best.
- **Personality Injection Not Always Counterproductive**: A member suggested that, in general, adding personality to models is not necessarily counterproductive.
   - They suggest projects like **Letta** (formerly **MemGPT**) use some kind of “personas” as an example of how this can be effectively implemented.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1394966614192947201)** (5 messages): 

> `LLM RL environment framework, Atropos compatibility, Unsloth RL guide` 


- **Nvidia Publishes Framework for LLM RL**: Nvidia published the initial version of an [LLM RL environment framework](https://research.nvidia.com/labs/adlr/AF3/) for long-horizon tasks, making it easier to set up environments in a Docker container with access to tools and generate trajectories.
   - The framework, called *real-work*, is available on [GitHub](https://github.com/ritser-labs/real-work).
- **Atropos Compatibility Explored**: A member asked if the new Nvidia LLM RL environment framework could be ported into **Atropos**.
   - The author of the framework responded that it's a good idea and they will look into making an adapter for **Atropos**.
- **Unsloth Issues RL Guide**: Unsloth released a new guide on **Reinforcement Learning (RL)**, available at [docs.unsloth.ai](https://docs.unsloth.ai/basics/reinforcement-learning-rl-guide).


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1394778285300387921)** (28 messages🔥): 

> `Image Generation, Model Search Repo URL, LM Studio Development Roadmap, Memory Features` 


- **LM Studio dodges Image Generation**: While **image input** is already available in LM Studio, **image generation** is not currently planned.
- **Users ask for custom Model Search Repo URLs**: A member asked for the ability to enter a specific URL for the **Model Search repo**, instead of using Hugging Face.
   - Another member responded that downloading the model manually and then importing it is the only option, as *there is no way to switch away from Hugging Face for good*.
- **No Public Roadmap Exists**: A member inquired about the existence of a **public LM Studio development roadmap**, but another member confirmed that *no public roadmap* exists.
- **Memory features may come to LM Studio**: A member asked about **memory features** (similar to ChatGPT's memory) coming to LM Studio, or the ability for chats to reference previous chats.
   - A member suggested to use an **MCP that has memory** and hoped it would be included just like the **rag-v1** and **code-sandbox mcps**.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1394766357442330824)** (39 messages🔥): 

> `LG's EXAONE License, Thunderbolt eGPUs for VRAM Expansion, AMD NPU Support in llama.cpp, PCI-Express Atomics Support` 


- **LG's EXAONE License Sparks Debate**: Members discussed the restrictive nature of **LG's EXAONE license**, particularly the requirement to add *"EXAONE"* to every model released and the limitations on commercial or research use.
   - The community questioned what constitutes *"research"* if reverse engineering and distilling are prohibited, with some arguing the license is contradictory and difficult to enforce.
- **Thunderbolt eGPUs Spark VRAM Expansion Dreams**: A member inquired about using a **PCI-E Thunderbolt 3/4 card** to expand **GPU VRAM**, accompanied by an image of memory modules.
   - However, it was noted that there are currently no successful implementations of eGPUs on **M-series Macbooks**.
- **AMD NPU Inferencing Lacks llama.cpp Support**: The community discussed whether **AMD's NPU** works with **AI inferencing** in **LM Studio**, but it was noted that *llama.cpp* doesn't have NPU support yet.
   - Currently, the only NPU *llama.cpp* supports is **Ascend NPU**, as documented in the [build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#cann).
- **PCI-Express Atomics Standalone**: For *muh PCIIRC*, theoretically, **ROCm** supports multi-GPU configs; however, your whole eGPU pipeline will need to pass the test on **PCI-Express atomics support** ([https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/conceptual/pcie-atomics.html](https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/conceptual/pcie-atomics.html)).
   - Judging by the anecdotal reports it may be mostly missing.


  

---


### **Torchtune ▷ #[announcements](https://discord.com/channels/1216353675241590815/1216353675241590818/1394759808326045799)** (1 messages): 

> `Future of Torchtune, Torchtune Project, Discord and Github support` 


- **Torchtune's Future Announced**: The team made an important announcement about the future of the **torchtune project** in [this GitHub issue](https://github.com/pytorch/torchtune/issues/2883).
   - They extended their thanks to everyone who has helped grow **torchtune** and promised to remain available on **Discord** and **Github** to answer questions.
- **Torchtune Team's next steps revealed**: The **torchtune team** assured the community they aren’t disbanding and promised more exciting work is on the horizon.
   - Keep your eyes peeled, they plan to share more with the community soon!


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1394760855018803300)** (53 messages🔥): 

> `Torchtune future, HuggingFace TRL License, Quantum Computing in Ohio, Checkpointing via NFT` 


- **Torchtune Evolves into New Product**: Torchtune is evolving into *something bigger* with a *new product* in a *new repo*.
   - A member mentioned that they are *developing a new product - in a new repo - that captures this evolution of torchtune*.
- **HF's TRL Shows Torchtune's Permissive BSD 3 License**: A member inquired about intellectual property concerns when using Torchtune's components for another project given the new announcement.
   - Another member pointed out that [Hugging Face has done this a bit already](https://github.com/huggingface/trl/blob/640a9f39164ce3f9bbac7d325c1149ba023314e6/trl/models/activation_offloading.py#L19) using the BSD 3 license, which is fairly permissive.
- **Quantum Computer Spotted in Ohio Cafeteria**: A member revealed they work at **Cleveland Clinic**, *the only hospital in the world with a quantum computer lol*. 
   - Another member humorously noted that the quantum computer is *in the middle of the cafeteria... in ohio??*
- **Checkpointing Training runs via NFT**: Members discussed future technologies like distributed RL, blockchain, and quantum computing, which led to a joke about the cost of checkpointing.
   - One suggested to checkpoint failed training runs on the blockchain and paying gas for that, and another humorously noted to simply *create an NFT* for your checkpoint.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1394766185245048902)** (32 messages🔥): 

> `Kimi K2 Open Source Model, Linux for Home Lab, Azure Speech Services SDK, Qwen-1.5 Inference Technologies, SmolVLM2 Technical Report` 


- ****Kimi K2 Unleashed** for Local Devices**: The community is excited that **Kimi K2**, claimed as the world's most powerful open-source model, can now be run on local devices, with a [link provided](https://huggingface.co/unsloth/Kimi-K2-Instruct-GGUFit).
   - Some users cautioned against cross-posting this announcement, recommending better channel etiquette.
- ****Linux Distro Dilemma** for Home Labs**: A member sought recommendations for a Linux distribution for a home lab, citing issues with Ubuntu.
   - Other members suggested sticking with Ubuntu, and inquired about the specific problems faced.
- ****Qwen-1.5's Secrets** Probed**: A user is seeking the exact structure and inference technologies used by **Qwen-1.5**, noting discrepancies in probabilities compared to their own implementation, and the discussion pointed to a [relevant file](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py).
   - Another member suggested floating point errors might be the cause, and the user replied that an average of 1.25e-5 is decent.
- ****datasets REST API** Bug Hunt**: A member asked where to report bugs within the **datasets REST API**, wondering if there was a dedicated GitHub repository or if issues should be filed directly within the main [datasets repo](https://github.com/huggingface/datasets).
   - Another member suggested that Github is best.
- ****HF Repo Watching** Wishlist**: A member inquired if it was possible to watch a single Pull Request or discussion on Hugging Face, rather than watching the entire repository.
   - The user sought a way to filter notifications for specific items of interest.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1395126740874952756)** (1 messages): 

> `Model Training, 1.5 bit research` 


- **Training determines model usage**: A member posited that a model's training dictates how it's ultimately used.
   - They suggested that the focus on **1.5 bit research** indicates underlying issues elsewhere in model development.
- **Researchers eye 1.5 bit **: Researchers are actively exploring **1.5 bit quantization**, signaling potential problems in other areas of model design.
   - This implies that improving existing models might require more than just scaling parameters; refining the training process and architectural choices could be crucial.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/)** (1 messages): 

tonic_1: https://gpuhammer.com/ wake up babe ! new exploit just dropppppped !
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1394879190007808130)** (7 messages): 

> `LLM Quantization, Desktop App for Plural Identification, French Deep Learning Course, English to Ukrainian Machine Translation Model, LunarisCodex LLM` 


- **ETHOS Quantization Explored**: A member shared [ETHOS](https://github.com/wrmedford/ETHOS), diving into **LLM quantization techniques**, hoping it's valuable and linked a [YouTube video](https://youtu.be/0pF6GdbwMo4?si=swVldbUTY5Gn4mYB) on the topic.
   - The accompanying [PDF](https://cdn.discordapp.com/attachments/897390720388825149/1394879190049882132/ETHOS.pdf?ex=687912ba&is=6877c13a&hm=8c3b1d564877e4310662d28d5a240c65d01f81b2e66098066acc531c259b6cd5&) delves into a deep dive into LLM quantization techniques.
- **PluralChat Goes Cross-Platform**: A member shared a desktop app for people who identify as plural called [PluralChat](https://github.com/Ktiseos-Nyx/plural_chat) leveraging **Python, Gemini, and Claude**.
   - The app is designed to be **99% offline**, cross-platform with easy installation using `pipx install aicodeprep-gui` and includes time-saving features like buttons to type out common prompts.
- **French Deep Learning Course Adds AI**: A member announced new features to their **French Deep Learning course** project including **AI-generated QCM** (multiple-choice questionnaires).
   - Upcoming features include **AI support** for generating course materials and a chatbot to clarify misunderstood parts, with resources available on the [course website](https://simonthomine.github.io/CoursDeepLearning/) and [GitHub repository](https://github.com/SimonThomine/CoursDeepLearning/).
- **Lightweight Machine Translation Hits HF**: A member shared a new **lightweight model** for **English to Ukrainian machine translation** fine-tuned with **40M samples** from **53.5M** using the recently published **LFM2 model**.
   - The model (**350M params**, requiring **1GB of RAM**) achieves a **BLEU** score of **27.24** on **FLORES-200**, with the model available on [Hugging Face](https://huggingface.co/Yehor/kulyk-en-uk) and a [demo space](https://huggingface.co/spaces/Yehor/en-uk-translator).
- **LunarisCodex: Brazilian LLM Toolkit**: A member from Brazil shared **LunarisCodex**, a modern and educational implementation of a **Transformer-style language model**.
   - This **100% open-source** toolkit includes features like **RoPE, GQA, SwiGLU activation, RMSNorm, KV Caching, and Gradient Checkpointing**, inspired by **LLaMA** and **Mistral**, with code available on [GitHub](https://github.com/MeryylleA/lunariscodex).


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1394943450432143391)** (3 messages): 

> `Pipeline Parallelism` 


- **Pipeline Parallelism Techniques Spark Excitement**: A member highlighted a resource on pipeline parallelism techniques and encouraged others to read it and implement the techniques from scratch on a simple MLP.
   - Another member reminded everyone to keep the channel on topic and use the appropriate channels for other discussions.
- **Channel Topic Reminder**: A member reminded everyone to keep the channel on topic and use the appropriate channels for other discussions.
   - This reminder followed a suggestion to explore pipeline parallelism techniques.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1395199975167758406)** (1 messages): 

> `SmolDocLing Finetuning, IDEFICS3ImageProcessor Error` 


- **SmolDocLing finetuning has issues**: A member ran into a `ValueError` while finetuning **SmolDocLing**.
   - The error was *Could not find module Idefics3ImageProcessor in `transformers`*.
- **Troubleshooting Idefics3ImageProcessor**: The user is facing an issue where the `Idefics3ImageProcessor` module cannot be found within the `transformers` library.
   - This suggests a potential problem with the environment setup or a missing component required for finetuning the **SmolDocLing** model.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1395058633750089879)** (1 messages): 

> `smolagents, Multi-Agent System, AI Agent Message Flow` 


- **Smolagents Diagram Gets Shared**: A member shared a diagram of a smolagents-based **Multi-Agent System** architecture.
   - The shared diagram provides a visual representation of the system's components and their interactions, potentially aiding in understanding and implementation.
- **AI Agent Message Flow Diagram Shared**: A member also shared a diagram illustrating the **AI Agent Message Flow**.
   - This diagram showcases how messages are routed and processed within the multi-agent system, offering insight into the system's communication infrastructure.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1394756621590204478)** (19 messages🔥): 

> `Meta's Open Source Policies, Behemoth Model, Inferring closed models from open weights, Actual ML` 


- **Zuckerberg accused of betrayal with Behemoth model**: Some members accuse **Zuckerberg** of betrayal due to the potential restricted release of the **Behemoth model**, citing a [tweet](https://x.com/signulll/status/1944851904888234293) as evidence.
- **Smaller, open-weight models emerge as alternative**: With the potential restricted release of the **Behemoth model**, some posit that Meta may follow **Google's** path by releasing smaller, open-weight models like **Gemma**.
- **Inferring closed models from open-weights intrigues researchers**: A member suggested that inferring closed models from open-weight models could present a good research problem.
   - They also said *Most people don't run those sorts of models locally, but still sucks.*
- **Discord channel is about *Actual* ML**: A member clarified the channel's purpose, stating it focuses on *Actual* ML, as opposed to *crypto scams and schitzophrenics*.
   - They then linked to a [paper](https://arxiv.org/abs/2407.18384) for discussion.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1394778679938384056)** (5 messages): 

> `GPUHammer paper, memory corruption, data structures` 


- **GPUHammer Paper Drops**: Members shared a link to the [GPUHammer paper](https://www.arxiv.org/abs/2507.08166).
   - The paper discusses **memory corruption** and **susceptibility of data structures** to this type of issue.
- **Data Structure Discussion Commences**: A member inquired about work on memory-related projects.
   - Another member mentioned they are working on **data structures and algorithms** susceptible to memory corruption.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1394865148975644693)** (18 messages🔥): 

> `Muon Optimizer, Mixture of Experts memory optimization, Amazon Cursor competitor` 


- **Muon Optimizer Debuts**: A new optimizer called **Muon** is used in a model detailed in [this paper](https://arxiv.org/abs/2507.08166) and [this video](https://www.youtube.com/watch?v=4bFDPVe6BHs) that aims to be well-trained for tool usage comparable to **Claude 4**.
- **MoEs Optimize Memory Bandwidth**: Using **Mixture of Experts** (MoEs) architecture, big labs optimize for memory bandwidth, which implies memory bandwidth is the key problem, as it minimizes memory movement.
   - One member stated that **MoEs** could potentially enable training with fewer GPUs than dense models due to their efficient resource utilization, as discussed in [this video](https://youtu.be/JOqLp1adGO4?si=hUAnREYY5CQoeoaQ).
- **Amazon Fights Cursor with Secret Weapon**: Amazon released a competitor to **Cursor** as discussed in [this link](https://www.pomerium.com/blog/when-ai-has-root-lessons-from-the-supabase-mcp-data-leak), which notes that Cursor is great for leaking your data.
   - The discussion references the [Supabase MCP data leak](https://www.pomerium.com/blog/when-ai-has-root-lessons-from-the-supabase-mcp-data-leak) as a cautionary tale about AI data security.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1394828914357960705)** (26 messages🔥): 

> `Vertex AI Thinking Output, Terminal Recommendations for Aider, Local Model Alternatives to Claude Code, Kimi K2 with Groq, Aider benchmark updates` 


- ****Vertex AI Thinking Tokens Finally Display****: A user discovered that running `/help` with the `openrouter/google/gemini-2.5-pro` model in Aider [enabled the display of thinking tokens](https://cdn.discordapp.com/attachments/1131200896827654149/1394828914072883380/image.png?ex=68798ca7&is=68783b27&hm=014b973a4b9583e6ebc25aced1ccb74c6018), which they found helpful for monitoring request progress.
   - They later found that using the `/think- 32k` command also enables the display of thinking summaries, and they appreciate it even though it can add to scrollback space.
- ****Ghostty and Kitty gain Terminal App Recommendations****: Users discussed terminal recommendations for Aider, with **Ghostty** and **Kitty** suggested for better performance, though **GNOME terminal** is considered sufficient by some.
   - One user experiencing issues with Aider screen refreshing was advised to try Ghostty, while another recommended **Alacritty** despite its difficulty with image display protocols.
- ****Kimi K2 with Groq Emerges as Frontrunner****: A user reported that **Kimi K2 with Groq** is delivering *phenomenal* performance, with **200-300 tokens/sec**, and high-quality output, rivaling **GPT-4o** and exceeding **Sonnet 3.7**.
   - They highlighted its affordability and speed, making it their preferred choice, echoing positive feedback from others in the community.
- ****Auto Thinking vs. Explicit Thinking****: A user discovered that Aider's **auto thinking** feature does not display the thinking process, leading them to explore explicit commands for showing thinking summaries.
   - Using the `/think- 32k` command enables the display of thinking summaries, while auto thinking operates in the background without showing the process.
- ****Time for Aider Benchmark Refresh?****: Users suggested that Aider should update its benchmark, given that many models now score above **80%**.
   - The proposal included creating a private benchmark where users can contribute their own tests.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1394817078279536801)** (11 messages🔥): 

> `Aider Debugging, OpenRouter Models, Gemini Flash, Architect Mode, Thinking Mode` 


- **Aider Users Debate Debugging**: A user inquired about debugging **Aider** in development mode, but no specific solutions were provided in the given context.
   - The user was looking for more interactive debugging, but no other users appear to have offered a method.
- **OpenRouter Model Removal**: A user noted that **OpenRouter** removed the `google/gemini-flash-2.5-preview:thinking` model and sought a way to enable *thinking* mode in `openrouter/google/gemini-flash-2.5`.
   - They discovered that `/think-tokens 2000` seemed to work, but were unsure if it was the correct value to use.
- **DeepSeek-r1-0528 unsupported by Aider**: A user asked if it was possible to modify the existing r1 to use the **0528** update, referring to the [DeepSeek-r1-0528:free model](https://openrouter.ai/deepseek/deepseek-r1-0528:free).
   - However, it was stated that *this model isn't supported by Aider*.
- **Thinking Tokens for Gemini**: A user referenced a tweet regarding configuring **32k thinking tokens** for **Gemini** ([Paul Gauthier's tweet](https://x.com/paulgauthier/status/1932068596907495579)), while trying to use **Gemini 2.5 Pro** through Vertex.
   - The user confirmed that `/think-tokens` command enabled streamed thinking summaries on Vertex.
- **Architect Mode Usage Explored**: A user questioned how others utilize **/architect mode**, noting it seems like a robust code mode with two models rather than a human-in-the-loop design discussion.
   - The user contrasted this with the common suggestion of generating a **.md** file with the suggested solution for review before implementation with **/code mode**, and wondered if **/ask mode** is better for design discussions.


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1395061788336459806)** (2 messages): 

> `Switchpoint Router, OpenRouter AI, aider polyglot benchmark` 


- **SwitchPoint Router Sparks Interest**: A member inquired about the [SwitchPoint Router on OpenRouter.ai](https://openrouter.ai/switchpoint/router), which routes requests to models like **GPT-4.1**, **Claude 4 Sonnet**, and **Gemini 2.5 Pro** at potentially lower rates.
   - The router's website boasts an **80%** result on the *aider polyglot benchmark*, prompting further discussion.
- **User Expresses Interest in Switchpoint**: A user showed interest in the router and said they hadn't used it before.
   - They noted that it was *interesting*.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1395036046156370001)** (2 messages): 

> `Google Docs Tab Feature, uBlock browser extension, Copying news articles into Google Docs` 


- ****Docs Tabs Trick** Inspires New Idea**: A member found using **Google Docs' tab feature** a *cool idea*, expressing that they hadn't considered using it in that way before.
- ****uBlock Extension** Stands Guard**: A member suggested using the **uBlock browser extension** to remove ads and other unwanted elements when copying news articles into Google Docs.
   - They noted that additional filters for annoyances and social media popups can be added in the extension's settings under the *Filter list* tab.
- ****Notepad.exe** Cleans Clipboards**: A member suggested copying text into **notepad.exe** as a method to avoid pasting ads and other unwanted content into Google Docs.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1394761208963272867)** (23 messages🔥): 

> `PC version of NotebookLM, Featured Notebooks Removal, Video Overviews Release, Custom Podcast Intros, Public Notebooks Location` 


- ****NotebookLM** desktop app not available?**: A user inquired about a **PC version** of the application and expressed unfamiliarity with using **Google Drive** as a source.
   - Another user suggested utilizing **Google Docs** with tabs to manage multiple sources within a single notebook.
- **Users Wanting to Hide Featured Notebooks**: Users are expressing frustration with being forced to view **"Featured Notebooks"** without an option to remove them.
   - They expressed a desire for a more customized feel and organization.
- **Users Seek Way to Customize Podcast Intro**: A user asked how to create a **podcast intro** that doesn't start with "Welcome to the Deep Dive,"
   - Another user suggested that the standard intro is the official tagline and that users are content contributors to the "The Deep Dive Podcast".
- **Podcast Length Determined by Language Setting**: A user reported that their podcasts were consistently short, around **7-10 minutes**, even with extensive source material.
   - Another user pointed out that the option to select "long" podcast output is available for **English**, resolving the issue.
- **"Service Unavailable" Error Plagues Users**: Some users are encountering a **"Service unavailable"** error message without sufficient context.
   - The error indicates that the user is trying to access a service not available for their account.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1394799068621963335)** (20 messages🔥): 

> `Manus AI mobile app creation, Vehicle creation for OMSI 2, AI outperforming Manus` 


- **Achievement Unlocked: Manus AI Mobile App Creation Mastery!**: A member claimed to have mastered **Manus AI** to create mobile applications on any topic with customized design for only **100 credits**.
   - They offered to help others who need something built by **Manus** but are experiencing issues.
- **Free Manus Alternative: Vehicle Creation for OMSI 2?**: A member claimed to have an alternative to **Manus** with all the same features but **100% free** with no limits, using it to create a vehicle for the video game **OMSI 2** (a bus simulator).
   - They speculated it might be able to create a script for **Google Collab** to generate the file, depending on the model.
- **Next-Level AI Claims to Outperform Manus**: A member claimed to have built an **AI that outperforms Manus** in benchmarks and offered the first **100 people full, unlimited access** as lifetime beta testers.
   - They directed others to DM them to claim their spot and *experience next-level AI with zero limits*.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1395111051498098848)** (2 messages): 

> `Discord Channels, Community Showcase` 


- **Discord Channel Expansion Proposed**: A member suggested creating a new Discord channel for users to share their projects and accomplishments, similar to channels in other open-source Discord servers.
   - This suggestion aimed to provide a space for sharing interesting images and repositories, deviating from the channel's existing rules.
- **Community Showcase as Alternative**: A staff member suggested that users share their work in the **Community Showcase** category of the [Modular forum](https://forum.modular.com/c/community-showcase/8).
   - This category is designed for showcasing user projects and contributions within the Modular community.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1394818972913762445)** (15 messages🔥): 

> `Mojo native requests library, TLS support in Mojo, Escaping keyword usage in Mojo, @parameter functions and runtime closures` 


- ****Mojo's** request library needs **TLS****: A member inquired about a native **Mojo** *requests* library, and another suggested that the main blocker is **TLS support**.
   - One member shared their [toy requests-like library](https://github.com/thatstoasty/floki), including **TLS bindings**.
- **Decoding **Escaping**: Not just for prison breaks**: A member sought clarification on the usage of the *escaping* keyword in **Mojo**, noting a lack of specific documentation.
   - Another member pointed to the [Changelog](https://docs.modular.com/mojo/changelog#v070-2024-01-25), clarifying that *escaping* performs a `__copyinit__` of values instead of capturing by reference.
- ****Parameter** decorators are captured!**: A member inquired about `@parameter` functions (capturing), and another provided a [link to the manual](https://docs.modular.com/mojo/manual/decorators/parameter/#parametric-closure) dedicated to it.
   - They also mentioned exciting news in the **Q3 roadmap** regarding *Unified @parameter and runtime closures* ([roadmap-update](https://forum.modular.com/t/mojo-q3-roadmap-update/1957)).


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1395085956792717472)** (12 messages🔥): 

> `setitem PR, tensor.py, assign parameter, kernel fusion, remove realize()` 


- **Setitem PR Gets Review**: A member asked for a review of their [setitem PR](https://github.com/tinygrad/tinygrad/pull/11260), specifically asking about the overhead of their solution and if it's on the right track.
   - The reviewer suggested that the `tensor.py` change should just be removing the `realize` call and fixing it in lower levels.
- **Optimization Discussion: Parameter to Assign**: A member inquired whether the correct approach would be adding a parameter to assign that lets users specify ranges and indices.
   - The reviewer responded that it's *not worth it* and *not what the bounty is looking for*.
- **Proposed Fix to realize() Removal**: When only removing `realize()` calls, the assignment isn't persisted back to `self`, so a member proposed changing the line to `self.uop = res.assign(v).uop` or similar.
   - They suggested alternatives like `self.assign(v, indices)` or `self.uop = reconcileAssignment(self.uop, res.assign(v).uop)`.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1394882175429644298)** (1 messages): 

> `tensor lvl hooks, LLM hidden states, Fetching hidden states` 


- **Tinygrad Tensor Hooks: Community Seeks Insights**: A user asked if **tinygrad** supports tensor level hooks, aiming to fetch hidden states in a large language model (**LLM**).
   - The user is exploring methods to extract and utilize the hidden states during model execution.
- **LLM Hidden State Extraction Explored**: The inquiry focuses on retrieving **hidden states** from an **LLM** using **tinygrad**.
   - The goal is to tap into intermediate representations within the model for further analysis or manipulation.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1394757033013547121)** (4 messages): 

> `Amsterdam Meetup, UiPath Integration, Production-Ready RAG, ODSC Agentic AI Summit` 


- **LlamaIndex Joins Snowflake for Amsterdam Meetup**: LlamaIndex is partnering with **Snowflake** for a meetup in Amsterdam on July 31st, focusing on building high-quality data agents in production ([link](https://t.co/fFJvvIWrw4)).
- **UiPath Adds Support for LlamaIndex Agents**: LlamaIndex agents can now be seamlessly deployed into enterprise environments with **UiPath's** new coded agents support ([link](https://t.co/ILez3d6Zrs)).
   - Features include *full code-level control* with the **UiPath's Python SDK** and the ability to build custom agents pulling data from enterprise systems.
- **Open Source Engineer Shares RAG Tips**: An Open Source Engineer shared battle-tested lessons from building production-ready RAG systems, with advice on **text extraction strategies** ([link](https://t.co/R0TTgWrKtv)).
   - The discussion covers when to use *simple parsing versus advanced OCR-based solutions* like **LlamaParse**.
- **LlamaIndex Presents at ODSC's Agentic AI Summit**: A LlamaIndex member will present a hands-on workshop at **ODSC's Agentic AI Summit**, teaching attendees to build agents using LlamaIndex ([link](https://t.co/6jcYIGR70s)).
   - Participants will learn to *create autonomous applications* that use goals and tools to accomplish tasks independently.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1395046546776526909)** (5 messages): 

> `LLM Fine-tuning Guide, Multi-agent workflow using LlamaIndex, AI Engineer Opportunities` 


- **LLM Fine-tuning Guide MVP is Live!**: An engineer shared an **LLM Fine-tuning Guide** MVP, aiming to provide practical, step-by-step advice on data preparation, parameter selection, and model evaluation and is looking for [developer feedback](https://ai-finetuning-advisor-g3r5.vercel.app/).
- **LlamaIndex Multi-Agent Workflows Spark Inquiry**: A member urgently requested assistance with **multi-agent workflows using LlamaIndex**.
- **AI Engineer Seeking to Build Intelligent Systems**: An engineer is looking to work with startups, research teams, or bold innovators in AI, Web3, or automation and specializes in building **autonomous agents** powered by GPT-4o, LangChain, AutoGen, CrewAI, and other cutting-edge tools.


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1394966425206259812)** (5 messages): 

> `IReRa for hierarchical labels, Multiple modules for hierarchy, Vanilla DSPy for parent-child identification` 


- **IReRa Questioned for Hierarchical Labels**: A member inquired if **IReRa** is advisable for a use case with **hierarchical labels** consisting of **3 levels**, **440 parents**, and **3500 grandchildren**.
   - Another member suggested that hierarchy lends itself to multiple modules or steps, but if there are only a few dozen labels in each step and big LLMs are used, **IReRa** is not necessary.
- **Multiple Modules Suggested for Hierarchy**: A member suggested using **multiple modules** (or steps) to handle hierarchical labels effectively.
   - They noted that if each step involves only a few dozen labels and large language models (**LLMs**) are utilized, **IReRa** might not be needed.
- **Vanilla DSPy Proposed for Parent-Child Identification**: A member asked if **vanilla DSPy** could be used to identify the parent first, then proceed from parent to child to grandchild.
   - Another member confirmed that they use a similar approach, with just **3 parents** and **28 children** in total, and it works well for them.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1395009983644696697)** (3 messages): 

> `aws/nova-prompt-optimizer, Lean 4` 


- ****Nova Prompt Optimizer** Requires DSPy**: A member checked and confirmed that the [aws/nova-prompt-optimizer](https://github.com/aws/nova-prompt-optimizer) has a dependency on `dspy`.
   - Hopefully someone will research the two working together.
- ****Lean 4** Verification on Deck**: A member recommended using **Lean 4** to verify something.
   - They linked a [YouTube video](https://www.youtube.com/watch?v=1067jj67toY) related to **Lean 4**.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1394786109615247510)** (8 messages🔥): 

> `Certificate Declaration Form, Lab Submission Feedback` 


- **Students Reprimanded after Missing Certificate Declaration Form**: The course organizers are unable to accommodate students who missed the **certificate declaration form** deadline due to limited staff capacity.
   - One student pleaded to reopen the [declaration form link](https://forms.gle/iPA2MUpHdtrBE1vu5) but their request was ultimately denied.
- **Students Seek Feedback on Lab Submissions**: A student inquired about receiving feedback on their **lab submission performance** and exploring additional research directions.
   - The student expressed satisfaction with their submission but sought further discussion with someone about their performance.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1394779720435896370)** (5 messages): 

> `Anthropic Connectors Directory, Docker MCP Toolkit, MCP Inspector` 


- ****Anthropic's Connector Directory** Opens MCP to Normies**: Anthropic announced their new ["connectors" directory](https://claude.ai/directory), opening up the MCP world to a broader community.
   - It was noted that the broader community will get introduced to it and hence the demand for those should spike.
- **Connectors Directory isn't Docker MCP Toolkit Competition**: A member assumed that the connectors directory is trying to compete with Docker's MCP Toolkit.
   - Another member mentioned that it's not competition because the **Docker toolkit is for devs to use**, whereas **this is meant to be used by normies**: think your everyday product manager or marketer.
- **MCP Inspector not Reloading Resources**: A member is writing a server with the mcp typescript-sdk and calling `server.sendResourceListChanged();` from within a tool after the profile is updated.
   - They found that when using MCP inspector, if they use the tool and then go refresh the resource it does not seem to update unless they clear the resource list and relist it.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1395039734228582492)** (2 messages): 

> `AI agents, Model Context Protocol, Autonomous Orchestration, Parallel Execution, Anthropic Claude Sonnet-4` 


- **Glasses 👓 Gives AI Power of Sight!**: A member shared a new **open-source tool** called **Glasses** 👓, implementing the **Model Context Protocol (MCP)** to let any compatible AI request a screenshot of a URL, and even emulate different screen sizes, available on [GitHub](https://github.com/gourraguis/glasses-mcp).
- **Goose Adds Subagent Support**: **Goose** now supports subagents, enhancing flexibility and enabling multi-agent systems, showcased with [Codex CLI as a subagent](https://block.github.io/goose/docs/experimental/subagents).
- **Multi-Model Goose Supports Claude and More**: **Goose** demonstrates multi-model support, using **Anthropic Claude Sonnet-4** as the main LLM, while also supporting **OpenAI, Gemini, Ollama**, and others.
- **Goose Enables Autonomous Orchestration**: **Goose** facilitates autonomous orchestration by coordinating a main task and subagents, then merging results, streamlining complex workflows.
- **Goose Powers Parallel Execution**: **Goose** enables parallel execution of tasks, with sequential support also available for scenarios requiring subagent handoffs, as demonstrated in the [attached image](https://cdn.discordapp.com/attachments/1315696461316358175/1395044247576776714/1752674728846.png?ex=687903b3&is=6877b233&hm=7fce7947df059f51cdf73bb6d012d910c06018fbb584d110a8e9c50596b404b4&).


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1394785062985400330)** (4 messages): 

> `Cloudflare R2, GPT4ALL logic, AI and Web3` 


- **Cloudflare R2 Access Denied**: A member encountered an **Access Denied** error while trying to download the dataset to fine-tune the model from **Cloudflare R2** using `aws s3 ls --endpoint-url=https://9fa58365a1a3d032127970d0bd9a1290.r2.cloudflarestorage.com/ s3://contrastive`.
- **GPT4ALL's Logic Under Examination**: A member inquired if **GPT4ALL** processes input taking raw logic, reasoning, and output transmitting logic based on a user-first approach.
   - They questioned whether **neutrality guidelines and safety filters** are still in place.
- **AI Engineer Available for Web3 Projects**: An AI and Web3-focused software engineer with hands-on experience building smart, autonomous systems is seeking new opportunities.
   - The engineer's skills include **Python**, **TypeScript (React/Next.JS)**, **C/C++**, **LangChain**, **ReAct**, **OpenAI**, and **Solidity/Rust** and is open to working with startups, research teams, or bold innovators in AI, Web3, or automation.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1394975518587228170)** (1 messages): 

> `DeepSeek in Production Event, MoE, MLA, FP8, MTP` 


- **DeepSeek Production Event is Happening Soon**: Packt is organizing a **DeepSeek in Production** event with a panel of top engineers and researchers discussing what makes **DeepSeek faster, smarter, and more efficient** than other models.
   - The event also features hands-on workshops where attendees can fine-tune **DeepSeek models** using **LoRA + Unsloth**, even on regular consumer GPUs; more details can be found at the [Eventbrite link](https://www.eventbrite.com/e/deepseek-in-production-tickets-1436251630289?aff=oddtdtcreator).
- **DeepSeek Model Advantages Detailed**: The event will cover technologies like **MoE, MLA, FP8, and MTP** to explain **DeepSeek's** performance benefits.
   - As a strong open source supporter, a participant highlights that the event seems promising.


  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1394941539330953347)** (1 messages): 

> `Financial Keyword Extraction with BERT, BERT for Key Sentence Extraction, Cosine Similarity for Keyword Identification, Improving BERT-based Keyword Extraction` 


- **BERT-Based Financial Keyword Extractor Struggles**: A user is building a **financial keyword/key sentence extractor with BERT** to identify company information such as **Registered Address**, **Province**, **Registration Date**, and **Business Term** from financial summaries.
   - Their initial approach, using **cosine similarity** between sentence embeddings and task-specific embeddings (e.g., "This sentence is the registered address"), *didn't work very well*.
- **Seeking Advice to Enhance BERT Extractor**: The user is seeking advice on how to improve their **BERT-based financial keyword extraction** model, as the initial cosine similarity approach was unsuccessful.
   - They are exploring alternative methods to accurately identify key information like **registered addresses** and **registration dates** from company financial summaries.


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1395160989007216650)** (1 messages): 

> `Claude Sonnet 4, Anthropic, Discounted Credit Rate` 


- ****Claude Sonnet 4** Returns with Anthropic Support**: **Claude Sonnet 4** is back with first party support from **Anthropic**.
   - [See the announcement here](https://x.com/windsurf_ai/status/1945599013954490523).
- **Discounted Rates for Pro/Teams Users**: **Claude Sonnet 4** and **Claude Sonnet 4** (Thinking) are available at a discounted **2x** credit rate for **Pro/Teams** users.
   - This applies across the **Editor** and **JetBrains Plugins** for a limited time.


  

---


### **Codeium (Windsurf) ▷ #[content](https://discord.com/channels/1027685395649015980/1092566563862884412/1394829404504461363)** (1 messages): 

> `Wave 11 inclusion` 


- **Wave 11 Inclusion Poll**: A poll on [X](https://x.com/windsurf_ai/status/1945263147994243294) asks the community whether to include a certain topic in **Wave 11**.
   - Members can interact with the post to voice their opinion.
- **Another Topic Placeholder**: This is a placeholder to satisfy the minimum items requirement.
   - Further details would be added here if available.


  

---


---

