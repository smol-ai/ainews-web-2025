---
id: MjAyNS0w
title: not much happened today
date: '2025-07-22T05:44:39.731046Z'
description: >-
  **Moonshot AI** released the **Kimi K2**, a 1-trillion parameter ultra-sparse
  Mixture-of-Experts (MoE) model with the **MuonClip** optimizer and a
  large-scale agentic data pipeline using over **20,000 tools**. Shortly after,
  **Alibaba** updated its **Qwen3** model with the **Qwen3-235B-A22B** variant,
  which outperforms Kimi K2 and other top models on benchmarks like **GPQA** and
  **AIME** despite being 4.25x smaller. Alibaba also released
  **Qwen3-Coder-480B-A35B**, a MoE model specialized for coding with a 1 million
  token context window. **Google DeepMind** launched **Gemini 2.5 Flash-Lite**,
  a faster and more cost-efficient model outperforming previous versions in
  coding, math, and multimodal tasks. The MoE architecture is becoming
  mainstream, with models like **Mistral**, **DeepSeek**, and **Kimi K2**
  leading the trend. In mathematics, an advanced **Gemini** model achieved a
  gold medal level score at the **International Mathematical Olympiad (IMO)**,
  marking a first for AI. An **OpenAI** researcher noted their IMO model "knew"
  when it did not have a correct solution, highlighting advances in model
  reasoning and self-awareness.
companies:
  - moonshot-ai
  - alibaba
  - google
  - google-deepmind
  - openai
  - hugging-face
  - vllm-project
models:
  - kimi-k2
  - qwen3-235b-a22b
  - qwen3-coder-480b-a35b
  - gemini-2.5-flash-lite
  - mistral-7b
  - deepseek-v3
topics:
  - mixture-of-experts
  - agentic-ai
  - model-optimization
  - model-training
  - benchmarking
  - code-generation
  - long-context
  - multimodality
  - math
  - reinforcement-learning
  - model-architecture
  - model-performance
  - open-source
  - alignment
people:
  - demishassabis
  - rasbt
  - alexwei_
  - yitayml
---


**a quiet day**

> AI News for 7/21/2025-7/22/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (227 channels, and 6134 messages) for you. Estimated reading time saved (at 200wpm): 527 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

the [release of Qwen 3 Coder (claiming Sonnet 4 level performance) and Qwen Code](https://qwenlm.github.io/blog/qwen3-coder/) (a fork of Gemini Code) almost made title story, but we're going to wait out a little bit to see where the reviews come in.

---

# AI Twitter Recap

**Major Model Releases & Benchmarks: Qwen, Kimi, and Gemini**

- **Kimi K2 Technical Report Released, Claiming SOTA on Agentic Tasks**: **Moonshot AI** has released the [technical report for Kimi K2](https://twitter.com/Kimi_Moonshot/status/1947520758760313170), a 1-trillion parameter ultra-sparse Mixture-of-Experts (MoE) model. The report details the **MuonClip** optimizer for stable training, a large-scale agentic data synthesis pipeline using over **20,000 tools**, and a joint RL alignment method. [The model is described](https://twitter.com/iScienceLuvr/status/1947414667221237904) as a **DeepSeekV3-style MoE** with higher sparsity and is open-source. The release has been called [the most inspiring technical report of the year](https://twitter.com/QuixiAI/status/1947388338337681541) by some in the community.
- **Qwen3-235B-A22B Challenges Kimi K2, Taking Top Benchmark Spots**: Shortly after the Kimi K2 release, **Alibaba** [updated its Qwen3 model](https://twitter.com/huybery/status/1947345040470380614), with the **Qwen3-235B-A22B** variant taking back the benchmark crown. [Dr. Sebastian Rasbt provides a technical breakdown](https://twitter.com/rasbt/status/1947393814496190712), noting it's **4.25x smaller** than Kimi 2 (235B vs 1T params) but has more layers and uses GQA instead of MLA. The model [reportedly beats Kimi-K2, Claude-4 Opus, and DeepSeek V3](https://twitter.com/scaling01/status/1947350866840748521) on benchmarks like **GPQA**, **AIME**, and **LiveCodeBench**. Its performance on **ARC-AGI-1**, scoring **41%** without a reasoning step, [is noted as particularly impressive](https://twitter.com/scaling01/status/1947351789222711455). The rapid progress led [one user to state](https://twitter.com/reach_vb/status/1947364340799283539), **"This is the dumbest open source models ever will be."**
- **Qwen3-Coder-480B-A35B Released for Advanced Code Generation**: **Alibaba** continued its release streak by dropping **Qwen3-Coder**, a **480B** total parameter MoE model with **35B** active parameters, specialized for coding and agentic tasks. [The model features a 1 million token context window](https://twitter.com/scaling01/status/1947732150872084693) and was developed over three months. [It shows strong performance on SWE-bench](https://twitter.com/QuixiAI/status/1947773200953217326). [Architecturally](https://twitter.com/nrehiew_/status/1947770826943549732), it is wider and shallower than the base Qwen3, with 62 layers, a 6144 hidden dimension, and 160 experts. The model is [available on Hugging Face](https://twitter.com/ClementDelangue/status/1947780025886855171) and is supported in **vLLM** nightly builds for [inference with expert parallelism](https://twitter.com/vllm_project/status/1947780382847603053).
- **Google Launches Gemini 2.5 Flash-Lite**: **Google** [announced the stable release of Gemini 2.5 Flash-Lite](https://twitter.com/Google/status/1947689382892204542), its most cost-efficient and fastest **2.5** family model. **Google DeepMind** [states it is faster and more cost-efficient](https://twitter.com/GoogleDeepMind/status/1947689582012633542) than the **2.0 Flash** models while outperforming them across coding, math, and multimodal understanding.
- **MoE Architecture Becomes Mainstream**: The recent releases have solidified Mixture-of-Experts as a dominant architecture. As summarized by [@hkproj](https://twitter.com/hkproj/status/1947571673021993152), **"Mistral started it, DeepSeek scaled it, Kimi K2 confirmed it: always more convenient to train an MoE."**

**AI in Mathematics: The Race for IMO Gold**

- **Google DeepMind's Gemini Officially Achieves IMO Gold Medal**: **Demis Hassabis** [announced that an advanced version of Gemini Deep Think](https://twitter.com/AndrewLampinen/status/1947370582393425931) has officially achieved a gold-medal level score (**35/42**) in the International Mathematical Olympiad (**IMO**), a first for an AI model.
- **Model "Knew" Its Limitations and Used Natural Language**: An **OpenAI** researcher, [@alexwei_](https://twitter.com/alexwei_/status/1947461238512095718), shared a key insight from their own IMO model's performance: on problem P6, which it did not solve, the model **"knew" it didn't have a correct solution**. A researcher from the **Google** team, [@YiTayML](https://twitter.com/YiTayML/status/1947350087941951596), noted their IMO gold model is a general-purpose model that will be shipped, not just an experimental one. Another Google researcher highlighted that [Gemini solved the problems end-to-end in English](https://twitter.com/denny_zhou/status/1947360696590839976).
- **Controversy Over Announcement Timing**: The achievement was marked by controversy over which lab announced its results first. Some criticized **OpenAI** for [what they perceived as "jumping the gun"](https://twitter.com/mathemagic1an/status/1947352370037305643), while others [questioned the overall value of a "race to announce"](https://twitter.com/francoisfleuret/status/1947359708811088211). **Demis Hassabis** [clarified](https://twitter.com/TheZachMueller/status/1947419062423982583) that Google respected the **IMO Board's** original request to wait before announcing.

**AI Infrastructure, Hardware & Efficiency**

- **OpenAI Announces "Stargate" 5GW Data Center with Oracle**: In a major infrastructure announcement, **OpenAI** revealed it is [developing **4.5 gigawatts** of additional "Stargate" data center capacity with **Oracle**](https://twitter.com/OpenAI/status/1947628731142648113), bringing the total to over **5 GW**. The **Stargate I** site in Abilene, TX is beginning to come online.
- **Taiwanese Students Push Semiconductor Frontiers**: A tweet from [@dylan522p](https://twitter.com/dylan522p/status/1947716636196409616) highlighted a Taiwan high school science exhibition where students were discussing **1.5nm Gate All Around (GAA) transistor structure optimization**, indicating a deep talent pipeline in advanced semiconductor research. He also commented on [China's progress with FlipFET and 3D DRAM](https://twitter.com/dylan522p/status/1947372973645504657) as critical for addressing the memory wall.
- **Donating Unused GPU Cycles to Open Science**: **Hugging Face CEO Clément Delangue** [posed the question of whether big tech could "donate" unused hours on their massive GPU clusters](https://twitter.com/ClementDelangue/status/1947379634615816287) to open science and open-source AI developers, a suggestion that garnered significant interest.
- **vLLM Integration with Hugging Face Transformers**: The **vLLM** project announced [support for Vision-Language Models out of the box with Transformers](https://twitter.com/vllm_project/status/1947756551663718754), simplifying deployment and inference for multimodal models.

**AI Tooling, Frameworks, and Applications**

- **Perplexity Comet Browser Gains Traction**: **Perplexity AI**'s new browser, **Comet**, has seen its [waitlist double since launch](https://twitter.com/AravSrinivas/status/1947407684996894969). Early user feedback suggests it [makes traditional chat interfaces "feel old"](https://twitter.com/AravSrinivas/status/1947478934528118887). A viral tweet from CEO [@AravSrinivas](https://twitter.com/AravSrinivas/status/1947501358007128149) asking who wants an agent to handle meetings received over **3,300 impressions**, signaling strong interest in its agentic capabilities.
- **LangChain 1.0 Announced**: **Harrison Chase** announced that the team is [working towards](https://twitter.com/hwchase17/status/1947376920355917909) `langchain` [1.0](https://twitter.com/hwchase17/status/1947376920355917909), which will focus on being the easiest place to start building LLM apps with revamped docs and general agent architectures built on **LangGraph**. He clarified that **LangGraph** is a lower-level ["agent runtime"](https://twitter.com/hwchase17/status/1947459414279262513) while LangChain will offer higher-level abstractions.
- **Anthropic Enhances Mobile Artifacts**: **Anthropic** [rolled out new ways to engage with artifacts on mobile](https://twitter.com/AnthropicAI/status/1947690894888513964), allowing users to create interactive tools, browse a gallery, and share work directly from their phones.
- **OpenAI Powers Clinical Copilot in Kenya**: **OpenAI** shared promising results from a [collaboration with Kenya-based PendaHealth](https://twitter.com/gdb/status/1947732134430687351), where an OpenAI-powered clinical copilot was studied across **40,000** patient visits.
- **LlamaIndex Releases Open-Source RFP Response Agent**: **LlamaIndex** built a [fully open-source agent for automating Request for Proposal (RFP) responses](https://twitter.com/jerryjliu0/status/1947465066892431792). The application, built on the **LlamaIndex** framework and **LlamaCloud**, handles document extraction, analysis, and report generation.

**Research, Company News, and Broader Discourse**

- **Research on "Subliminal Learning" in LLMs**: A paper by **Owain Evans** and team introduced the concept of "subliminal learning," showing that [LLMs can transmit traits to other models via hidden signals in generated data](https://twitter.com/_arohan_/status/1947704379110527183), even when the data is unrelated to the trait. The research suggests this may be a general property of neural net learning.
- **Anthropic Paper Finds "Inverse Scaling" in Test-Time Compute**: An **Anthropic** research paper found cases where [longer reasoning time leads to lower accuracy](https://twitter.com/dilipkay/status/1947677154663403732). The effect was observed in **6 benchmarks for Opus 4**, sparking discussion about the limitations of current reasoning models and scaling laws.
- **Major Funding and Hiring Moves**: **OpenAI** announced that **Fidji Simo** will become [CEO of Applications](https://twitter.com/kevinweil/status/1947345653014691958). **Reka** announced a [**$110M** funding round from investors including **NVIDIA** and **Snowflake**](https://twitter.com/RekaAILabs/status/1947689320594157668). Reports indicate **Meta** is aggressively hiring top AI researchers, with [compensation packages as high as **$300 million** over four years](https://twitter.com/DeepLearningAI/status/1947461590283858010).
- **The Open Source vs. Closed Model Debate Continues**: **Jack Dorsey's** call for ["permissionless" AI to prevent a few CEOs from dictating innovation](https://twitter.com/ClementDelangue/status/1947354551478218269) was widely shared. **Clément Delangue** commented on **Anthropic's** business decisions, stating they reinforce the need for [open-source AI to avoid concentration of power](https://twitter.com/ClementDelangue/status/1947689375565013046).

**Humor/Memes**

- **The Meeting Agent We All Want**: **Perplexity AI**'s CEO, [@AravSrinivas](https://twitter.com/AravSrinivas/status/1947501358007128149), captured the zeitgeist with a tweet asking who wants a **Comet** agent to handle their meetings, which resonated widely.
- **The Shrimp Button**: A tweet about a magic button thought experiment [that devolves into questions about shrimp](https://twitter.com/nptacek/status/1947468024019083714) became a running joke, with follow-ups about [not having seen 100 shrimp at once](https://twitter.com/code_star/status/1947525486126764364) and the need for shrimp-related regulations.
- **Data Cleaning is Not Low Value Work**: A tweet from [@code_star](https://twitter.com/code_star/status/1947529567633367064) reacting to a description of data cleaning as "low value work" struck a chord with engineers.
- **IMO Gold Medalist Hippo**: [@agihippo](https://twitter.com/agihippo/status/1947348097144611123) posted the concise and humorous summary: **"hippo at IMO: 0/42, model trained by hippo: 35/42 🥇"** and later followed up with a [perfect use of the gold medal meme](https://twitter.com/agihippo/status/1947655890733305971).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen3 Coding Model Releases and Benchmarks

- [**Qwen3- Coder 👀**](https://i.redd.it/vnhuwe801hef1.jpeg) ([Score: 354, Comments: 102](https://www.reddit.com/r/LocalLLaMA/comments/1m6mew9/qwen3_coder/)): **The image announces the release of Qwen3-Coder, a specialized large language model designed for advanced code generation, tool use, and agentic automation. Key technical highlights: the model supports an ultra-long maximum context window of** `1,048,576 tokens`**—substantially higher than most competing LLMs—and is listed as having** `480B parameters` **(with** `35B active`**). The model is already available for use at [https://chat.qwen.ai](https://chat.qwen.ai/) and is referred to on Hyperbolic under model ID** `Qwen/Qwen3-Coder-480B-A35B-Instruct`**.** Technical comments focus on the significance of the `1M token` context length and the high parameter count (480B, with 35B active). Users are evaluating it as a potential alternative to Anthropic’s models, citing performance and infrastructure issues with the latter.
    - Discussion highlights that Qwen3-Coder offers a `1M context length`, positioning it technically as a competitor to large-context commercial models for code-related tasks. There is explicit reference to its deployment under the model ID `Qwen/Qwen3-Coder-480B-A35B-Instruct`, specifying the model has `480B parameters` with `35B` active at inference, which implies a MoE (Mixture-of-Experts) or sparsely activated architecture for computational efficiency. Interest is expressed in comparing its scalability and performance to alternatives like Anthropic's models, especially in light of recent infrastructure or performance issues in those services.
- [**Everyone brace up for qwen !!**](https://i.redd.it/mn8auem2bhef1.png) ([Score: 140, Comments: 40](https://www.reddit.com/r/LocalLLaMA/comments/1m6nxh2/everyone_brace_up_for_qwen/)): **The image is an announcement for Qwen3-Coder-480B-A35B-Instruct, an upcoming 480B parameter Mixture-of-Experts (MoE) language model specifically designed for coding. The key features highlighted are its immense 1 million token context window and specialization in code generation, tool use, and agent-based tasks. The release is anticipated for public access, with hype around its potential performance improvements over previous large MoE coding models (like 235B models).** Commenters discuss hardware limitations, with most joking about the impracticality of running such a massive model locally, even with quantization ('q2'), and referencing high-end hardware needs (like a Mac M3 Ultra with 512 GB RAM). One user claims the online demo/version feels much faster than previous 235B models, implying significant efficiency or infrastructure improvements.
    - Users are discussing the extremely high VRAM requirements of Qwen-2 models, with some unable to run even quantized versions such as q2, and others referencing hardware like Apple’s M3 Ultra (256GB–512GB RAM) as a possible necessity, highlighting a hardware barrier for local inference and experimentation for home users.
    - One commenter notes that Qwen-2 is 'way faster than 235b on their website,' suggesting notable inference speed and potential optimization improvements for Qwen-2 relative to other large models, specifically referencing [OpenAI's GPT-3.5 (235B parameters)](https://platform.openai.com/docs/model-index-for-researchers).
    - An insightful comment addresses the trend of LLMs growing too large for consumers, arguing there is an urgent need for new types of chips or more efficient algorithms to enable practical local use; traditional methods like distillation are described as insufficient for preserving capabilities while reducing size.
- [**Qwen3-Coder-480B-A35B-Instruct**](https://www.reddit.com/r/LocalLLaMA/comments/1m6mlbk/qwen3coder480ba35binstruct/) ([Score: 141, Comments: 47](https://www.reddit.com/r/LocalLLaMA/comments/1m6mlbk/qwen3coder480ba35binstruct/)): **Hyperbolic AI has released access to the Qwen3-Coder-480B-A35B-Instruct model, a new code-focused LLM that follows Qwen2.5-Coder-32B, and is notable for its massive 480B parameter scale. Documentation and usage is live on the Hyperbolic AI platform (see: [model page](https://app.hyperbolic.ai/models/qwen3-coder-480b-a35b-instruct)), though implementation and performance benchmarks compared to predecessors are not yet detailed in this post.** Commenters note the unprecedented size and potential performance as a successor to Qwen2.5-Coder-32B and express anticipation of migrating from Claude and existing code models, particularly pending wider access (e.g., on OpenRouter).
    - LagOps91 clarifies that Qwen3-Coder-480B-A35B-Instruct is not a direct, drop-in replacement for other models like Claude, implying compatibility or integration differences that technical users should be aware of when considering migration or model swaps.
    - Mysterious_Finish543 provides a practical implementation note: Qwen3-Coder-480B-A35B-Instruct is already accessible via the Hyperbolic API under the model identifier `Qwen/Qwen3-Coder-480B-A35B-Instruct`, which is relevant for developers seeking immediate programmatic access to the model.
- [**Qwen3 235B-A22B 2507 :: Q3_K_L :: One shot HTML game :: 4090 + 128GB DDR5 @6000**](https://v.redd.it/1x5u9hrp5fef1) ([Score: 143, Comments: 55](https://www.reddit.com/r/LocalLLaMA/comments/1m6ct7u/qwen3_235ba22b_2507_q3_k_l_one_shot_html_game/)): **The post benchmarks local inference of the Qwen3 235B-A22B 2507 LLM (Q3_K_L quantization) on consumer hardware (4090 GPU, 128GB DDR5 @6000MHz, 23.3GB VRAM, ~80GB RAM usage, 5.52 tokens/sec, 2202 output tokens, 0.18s to first token) using LM Studio with flash attention enabled, demonstrating that the model is slow but usable on high-end desktops. The main test involved one-shot code generation for an old-school HTML/JS racing game (single index.html file), with the model generating interactive, progressive difficulty gameplay, retro aesthetics, and accurate code output. Execution settings: context 4096, GPU offload 18/94, 16 CPU threads; the linked generated code was uploaded externally due to sharing difficulties.**  Comments debate CPU/motherboard selection for stable 128GB @6000 MHz configurations and report that the Qwen3-235b-2507 (at Q4_K_XL quantization) outperforms prior Qwen variants in one-shot code generation and creative writing, with specific mention that 'the old Qwen3-235b was much of a disappointment' but this release sets a new performance benchmark for local LLMs.
    - Several users tested Qwen3-235b-2507 (quantized as Q3_K_L, Q4_K_XL, and Q2_K), reporting very strong one-shot coding capabilities—one called it "the best one-shot code LLM I have ran locally so far," even surpassing previous Qwen3-235b versions with advanced settings like 'thinking enabled.' Creative/story generation is also discussed, with unique but sometimes quirky output styles noted.
    - Technical specs for running large quantized models are shared: one setup used a Q2_K quant (85.7GB) split over dual GPUs (1x 16GB 5060 Ti, 1x 16GB Quadro P5000) and 64GB DDR5 6000MHz RAM, delivering 5-5.5 tokens/sec in 12K context, with the workload offloaded across both GPUs and CPU. It's noted that a mid-range i5-13400F was the primary bottleneck, not GPU utilization.
    - An HTML racing game generated by the model presents a retro-style implementation with classic mechanics including lane-based car control, randomized obstacles that scale difficulty with the player's score, and visually simulated road movement. The full code was shared via an external link, and design details specified procedural content generation and event-driven UI updates.
- [**Could this be Deepseek?**](https://i.redd.it/qzkjkgegugef1.png) ([Score: 211, Comments: 50](https://www.reddit.com/r/LocalLLaMA/comments/1m6lf9s/could_this_be_deepseek/)): **The screenshot shows a tweet from Casper Hansen asserting that a Chinese team is preparing to release 'kimi k2', which reportedly features a 1 million token context window—potentially making it competitive with large-context models like GPT-4 Turbo and Claude. The post and comments speculate on the origin, with some suggesting it may be related to Deepseek, while others point to Qwen, noting that 'qwen3-coder' is already available at [chat.qwen.ai](http://chat.qwen.ai/). No technical benchmarks, release notes, or architectural details are included, but the focus is on the enormous context length announcement and competition among Chinese LLMs.** Top commenters express skepticism about pre-release hype and urge caution, referencing past overstatements in the field. One user provides an additional image hinting that Qwen models may be the underlying technology, reinforcing the speculation and ongoing debate regarding the product's true source.
    - Discussion highlights that qwen3-coder is already available on [chat.qwen.ai](http://chat.qwen.ai/), with a screenshot suggesting that the model in question resembles Qwen rather than Deepseek. The inference is partly based on UI and branding details present in the screenshot link.
    - One participant compares possible models, noting that Kimi-reasoning is unlikely due to context window limitations (`K2 is only 128k`), while the new release might be either qwen3-reasoning-coder or Deepseek R2. This points to technical constraints regarding model architecture and maximum context length.
    - Another user expresses satisfaction with successfully loading a `32k` token context window, illustrating real-world testing limits and user-perceived feasibility compared to the advertised maximum context sizes (e.g., 1M).

### 2. AI Hardware and Enthusiast Upgrades

- [**Used A100 40GB just dropped below $2000, for those who care with caveat**](https://www.reddit.com/r/LocalLLaMA/comments/1m60ahf/used_a100_40gb_just_dropped_below_2000_for_those/) ([Score: 102, Comments: 61](https://www.reddit.com/r/LocalLLaMA/comments/1m60ahf/used_a100_40gb_just_dropped_below_2000_for_those/)): **Used NVIDIA A100 40GB GPUs (SXM4 form factor) are now available for under $2,000, but require a $600 adapter to interface with standard PCIe systems. The post points out that building an 8x A100 system is possible for ~$30,000 if sourcing HGX backboards (needed for proper NVLink mesh interconnect at up to 4,800GB/s), which typically cost about $9,000 used.** Commenters debate the practicality and value of used A100s (especially SXM4) versus new hardware, and question where such deals and supporting parts (e.g., HGX backboards) can reliably be sourced.
    - One user details that you can construct an 8x NVIDIA A100 40GB GPU system for approximately `$30,000`, but emphasizes the requirement of an HGX backboard, which enables 8-way Mesh NVLink interconnection delivering `4,800GB/s` bandwidth; this backboard typically costs an additional `$9,000` on the used market. The value here is tied directly to multi-GPU scaling via NVLink, which meaningfully boosts inter-GPU throughput compared to less integrated setups.
    - A significant technical caveat is that this value proposition is best for users who actually need dense multi-GPU configurations and high-speed NVLink interconnect—otherwise, for those not scaling beyond a single workstation node, newer or simpler cards like the anticipated 5090 may be a better investment. The inflection point for these used A100 deals therefore depends on workload (large-scale AI/ML, HPC) and infrastructure flexibility.
- [**AMD's Strix Halo "Ryzen AI MAX" APUs Come To DIY PC Builders With New MoDT "Mini-ITX" Motherboards, Equipped With Up To 128 GB of LPDDR5X Memory**](https://wccftech.com/amd-strix-halo-ryzen-ai-max-apus-diy-pc-new-modt-mini-itx-motherboards-128-gb-lpddr5x-memory/) ([Score: 110, Comments: 68](https://www.reddit.com/r/LocalLLaMA/comments/1m6bddm/amds_strix_halo_ryzen_ai_max_apus_come_to_diy_pc/)): **AMD's Strix Halo "Ryzen AI MAX" APUs are being offered for DIY PC builders through new MoDT Mini-ITX motherboards, featuring support for up to 128GB LPDDR5X memory. These boards, targeted for compact AI/ML and edge compute applications, reportedly lack standard PCIe expansion slots ("zero PCIe lanes"), which constrains use of discrete GPUs or high-speed peripherals. Some boards unusually include legacy VGA output.** Commenters raise concerns about limited PCIe expandability, potential BIOS quality/support issues from niche motherboard manufacturers, and memory ceiling (128GB) being insufficient for running large AI models such as frontier LLMs, for which at least 256GB RAM is preferred.
    - Several commenters note that the motherboard lacks PCIe lanes, significantly limiting expandability, and point out legacy port choices (such as VGA), which are considered outdated for high-performance applications.
    - Concerns are raised about the platform's viability for advanced AI workloads: 128GB LPDDR5X is enough for models like Qwen-235B, but insufficient for larger frontier AI models, for which 256GB or more would be preferred.
    - Multiple users caution that this is not an official AMD release but rather an unsupported prototype from a Chinese manufacturer, calling attention to historically poor BIOS support and minimal update cycles from such niche vendors.
- [**Idc if she stutters. She’s local ❤️**](https://i.redd.it/66ckkcwl5bef1.png) ([Score: 143, Comments: 13](https://www.reddit.com/r/LocalLLM/comments/1m5xuzi/idc_if_she_stutters_shes_local/)): **The meme humorously depicts local LLM (Large Language Model) enthusiasts preferring to run challenging, locally quantized 13B models (even those that may be unstable or require significant effort to quantize and get running) on GPUs like the RTX 3090, rather than paying for token access to OpenAI’s cloud models. Several technical commenters point out that the RTX 3090, with its 24GB VRAM, should comfortably handle a 13B model at 8-bit or even fp16 precision, questioning why quantization and crashes are an issue. They provide resources for memory calculation and quantization levels relevant to running models in the GGUF format. The post reflects the tradeoff of cost, effort, and privacy in local vs. hosted LLM usage.** Critical debate focuses on the unnecessary effort implied in quantizing a 13B model for a 3090, with consensus that such hardware doesn't require aggressive quantization and is fully capable of stable operation. Privacy remains a key motivation for local inference despite setup complexity.
    - Several commenters point out that a 13B parameter model can be comfortably run on an NVIDIA 3090 (24GB VRAM) at 8-bit quantization (int8), and even potentially at fp16, without VRAM issues. Quantizing further for such a GPU is generally unnecessary and may not improve performance or memory efficiency significantly.
    - There is skepticism about the time for quantization on a 3090, indicating that whether 36 hours is too much or too little is unclear—users debate feasibility but highlight that they typically use Hugging Face or download pre-quantized models rather than quantizing themselves. Some raise concerns about possible model instability or startup crashes resulting from improper quantization workflows.
    - A technical resource is shared for calculating memory usage for LLMs and for understanding quantization levels, especially for those working with GGUF formats. This could help users select appropriate quantization settings based on available GPU memory and model requirements.

### 3. MegaTTS 3 Voice Cloning and Open-Source AI Tools

- [**MegaTTS 3 Voice Cloning is Here**](https://huggingface.co/spaces/mrfakename/MegaTTS3-Voice-Cloning) ([Score: 346, Comments: 63](https://www.reddit.com/r/LocalLLaMA/comments/1m641zg/megatts_3_voice_cloning_is_here/)): **The long-awaited WavVAE encoder for ByteDance's MegaTTS 3 has been released by ACoderPassBy ([ModelScope link](https://modelscope.cn/models/ACoderPassBy/MegaTTS-SFT)), enabling practical voice cloning, including support for diverse accents and voice timbres; models and demo are now available on Hugging Face ([weights](https://huggingface.co/mrfakename/MegaTTS3-VoiceCloning), [Gradio demo](https://huggingface.co/spaces/mrfakename/MegaTTS3-Voice-Cloning)). Early reports highlight high voice fidelity compared to tools like Chatterbox, especially with previously challenging cases (e.g., female, southern drawl, British, falsetto), though inference speed is slower. Key technological bottleneck was the lack of an openly available encoder, now addressed fundamentally expanding MegaTTS 3's usability for cloning.** Technical discussion centers around real-time streaming capability and GPU memory demands, with users interested in integration into streaming pipelines and comparing resource requirements versus other TTS solutions.
    - Users note MegaTTS 3 shows significant improvement over chatterbox in handling regional accents (like southern drawls) and varied pitch ranges, including female, British, deep, and falsetto voices, though its inference speed is slower than chatterbox.
    - One commenter highlights that MegaTTS 3 is still behind chatterbox and zonos in output quality: the speech is described as 'stilted' with less natural flow. Chatterbox, while sometimes struggling with accents, produces fluid and convincing results with minimal tweaking; Zonos handles accents better and allows for deeper delivery customization, but is slower and requires more adjustment.
    - Technical questions are raised about whether MegaTTS 3 supports streaming generation and how resource-intensive ('GPU fatness') the model is, both important considerations for deployment scenarios where TTS needs to be integrated with other complex processing.
- [**The ik_llama.cpp repository is back! \o/**](https://www.reddit.com/r/LocalLLaMA/comments/1m6cfzi/the_ik_llamacpp_repository_is_back_o/) ([Score: 172, Comments: 29](https://www.reddit.com/r/LocalLLaMA/comments/1m6cfzi/the_ik_llamacpp_repository_is_back_o/)): **The** `ik_llama.cpp` **repository ([ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)), which provides C++ inference code for Llama models, has been restored on GitHub after a period of removal or inaccessibility. The announcement underscores the importance of regularly backing up critical repositories to prevent data loss due to takedowns or outages.** Comments note the rapid recovery of the repository, suggesting community support through contacting GitHub may have facilitated its reinstatement. Users celebrate its return and highlight the importance of archiving valuable codebases.
    - A user inquires about methods to locally mirror not just the code, but also issues, discussions, and wikis from GitHub repositories. They seek comprehensive solutions for full data backup, mentioning the need due to concerns about rapid takedowns or removals of valuable projects like `ik_llama.cpp`. This indicates a desire for tools or scripts that can perform complete archival beyond just cloning source code, perhaps referencing utilities like [github-backup](https://github.com/josegonzalez/python-github-backup) or [ghorg](https://github.com/gabrie30/ghorg) that may support broader data export, though caveats around API limits and authentication are implicit.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. Claude Code User Experience & Optimization Discussion

- [**Claude Code is doing it again**](https://i.redd.it/lqatk018bdef1.jpeg) ([Score: 374, Comments: 30](https://www.reddit.com/r/ClaudeAI/comments/1m66j0v/claude_code_is_doing_it_again/)): **The image is a meme referencing Claude Code's erratic or nonstandard behavior, humorously captured by pairing it with the "Why can't you just be normal?" meme format. The phrase 'Flibbertigibbeting' in the bottom panel references unpredictable or nonsensical output from Claude Code, making a comparison to the chaotic noise in the original meme. The image visually dramatizes user frustration with Claude's unpredictability.** Discussion in the comments centers on the term 'flibbertigibbet' (used in the meme), with references to pop culture appearances such as The Sound of Music, but there are no substantial technical debates or benchmarks presented.
    - One commenter heavily criticizes Anthropic's handling of Claude, specifically mentioning frequent alteration of the model's behavior and the implementation of *performance throttling and stricter usage limits*. They state that these changes have degraded the product over the past year, leading them (and their team) to look for open-source alternatives, highlighting a *growing sentiment among developers frustrated with the platform's unpredictability and constraints on access*.
- [**🎯 The Real Reason Claude Code Feels Broken (And How I Got It Working Again)**](https://www.reddit.com/r/ClaudeAI/comments/1m62xzc/the_real_reason_claude_code_feels_broken_and_how/) ([Score: 138, Comments: 136](https://www.reddit.com/r/ClaudeAI/comments/1m62xzc/the_real_reason_claude_code_feels_broken_and_how/)): **The OP describes that structured documentation and meticulous project scaffolding (readmes, function linkage, explicit task breakdown, and example I/O) leads to materially improved results using Claude Code, reducing hallucinations, context mistakes, and code duplication. They hypothesize that Claude Code is optimized for users who provide thorough architectural documentation and demand explicit planning, rather than relying on ad hoc development workflows. Commenters corroborate, emphasizing that this applies to all LLMs due to context window degradation (with a citation of Google's Gemini and its context length performance drop), and that Claude's orchestrator/subagent architecture mitigates but doesn't eliminate these problems—structured, staged instructions remain crucial as codebase size grows. Practical workflows mentioned include putting high-level specs and feature research in** `CLAUDE.md`**, creating feature-specific .md files as context scaffolding, using branches to isolate features, and iteratively guiding the model with this structured context for superior results compared to unstructured prompts.** Strong consensus that architected, pre-planned, and document-driven workflows significantly improve LLM code generation versus improvisational/"vibe" coding. Some technical debate exists over UI generation quality and suggestions for multi-agent orchestration logic as a future improvement.
    - Several comments recognize that Claude's code generation is most effective when used as a tool for high-level design and architectural reasoning rather than for direct coding. Users emphasize the need to document requirements, dataflows, potential race conditions, and testing strategies in detail before generating any code, arguing that LLMs currently lack the ability to autonomously produce scalable and maintainable architectures.
    - Technical discussion highlights challenges with LLM reasoning over long contexts, referencing a Google study showing models like Gemini experience significant performance degradation as context length increases. Claude's agentic approach—with multiple sub-agents managed by an orchestrator to avoid context pollution—is acknowledged as significant, but ultimately performance drops rapidly if codebases aren't broken into structured, modular steps. Users stress that as LLMs generate more code, the complexity quickly overwhelms their reasoning ability without disciplined project management.
    - Practitioners share best practices for using Claude effectively: maintaining feature-specific documentation (e.g., feature-named .md files), limiting focus to one feature or branch at a time, and synthesizing research-mode outputs before instructing Claude Code. This workflow, integrating iterative git commits and analysis artifacts, is said to markedly improve code quality relative to ad-hoc or prompt-focused approaches. Nonetheless, users observe persistent challenges with auto-generating UI code and caution that AI-assisted workflows still rely on advanced architectural skills not provided by LLMs.
- [**Are people actually getting bad code from claude?**](https://www.reddit.com/r/ClaudeAI/comments/1m6ienr/are_people_actually_getting_bad_code_from_claude/) ([Score: 132, Comments: 158](https://www.reddit.com/r/ClaudeAI/comments/1m6ienr/are_people_actually_getting_bad_code_from_claude/)): **The OP, a senior developer with a decade of experience, reports consistently high-quality C# code generation using Claude, completing complex projects (e.g., microservice APIs with advanced security and refactoring efforts replacing Mediatr with DI) far quicker than without AI. They contrast this with widespread complaints of low quality output, hypothesizing that effective context, clear prompts, and structured decomposition tasks—skills rooted in senior-level software engineering—are crucial for leveraging Claude's strengths.** Top comments emphasize that senior developers excel due to their ability to provide explicit context and break requirements into discrete components, likening good prompt engineering to thorough mentorship for a junior coder. However, some highlight Claude's tendency to miss subtle implementation details, potentially introducing hard-to-diagnose bugs even when generating otherwise well-structured code.
    - Several comments highlight that Claude's code quality is highly dependent on the clarity and precision of user prompts; advanced users who break tasks down and provide rigorous instructions consistently achieve better outcomes, paralleling the mentorship needed for junior developers. Models like Claude perform well when tightly directed, but tend to make incorrect assumptions or go off-tangent with vague requirements.
    - Specific user experiences point to recurring challenges: Claude can produce well-structured and readable code, but often fails to consistently follow instruction sets, sometimes repeating mistakes, ignoring specified design documents (such as 'CLAUDE.md'), or introducing subtle bugs that are non-trivial to debug, resulting in significant time lost for human developers. There are also instances where it miscategorizes or duplicates implementations erroneously and falsely claims success based on incomplete criteria (e.g., 'all tests look good' when code isn't even compiling).
    - Reference is made to [this study](https://arxiv.org/pdf/2503.08074), which discusses the psychological phenomenon where users' expectations of LLMs rise over time as initial novelty wears off, making incremental failures seem worse even if model performance remains constant. This 'lobotomy phase' is less a reflection of actual model degradation and more about misaligned user expectations, highlighting the importance of evaluating AI coding tools with consistent, objective benchmarks rather than shifting subjective impressions.
- [**To all you guys that hate Claude Code**](https://www.reddit.com/r/ClaudeAI/comments/1m6p9vo/to_all_you_guys_that_hate_claude_code/) ([Score: 169, Comments: 89](https://www.reddit.com/r/ClaudeAI/comments/1m6p9vo/to_all_you_guys_that_hate_claude_code/)): **Discussion centers around user dissatisfaction with recent Claude Code performance, with some alleging decreased code generation quality and accusing Anthropic of not providing adequate value in their subscriptions. Several comments suggest issues may relate to A/B testing or segmented model updates, causing varying experiences among users, though some report continued satisfaction and cite major productivity gains (mention of Claude Max x20 subscription). Emphasis is placed on the importance of user feedback for iterative model improvement and equitable product performance across the user base.** Notable debate occurs around whether user complaints are justified, with some attributing negative outcomes to user error and natural model development fluctuations, while others assert that disparate experiences indicate need for technical transparency and responsive support from Anthropic.
    - A notable technical point is the observation that Anthropic may be conducting A/B tests on Claude Code with different user segments, which could explain why some users experience success while others face failures. This highlights the importance of continuous user feedback for product improvement and suggests that user experience with the same product may vary due to backend experimentation or staged feature rollouts.

### 2. AI Model Benchmarking at the International Mathematical Olympiad

- [**Wow even the standard Gemini 2.5 pro model can win a gold medal in IMO 2025 with some careful prompting. (Web search was off, paper and prompt in comments)**](https://i.redd.it/kvtrm7no0def1.png) ([Score: 277, Comments: 58](https://www.reddit.com/r/singularity/comments/1m65l0f/wow_even_the_standard_gemini_25_pro_model_can_win/)): **The image is a tweet from Lin Yang describing that the standard public Google Gemini 2.5 Pro model, with web search off, solved 5 out of 6 International Mathematical Olympiad (IMO) 2025 problems when given 'careful prompting.' According to Lin, this achievement is significant as it shows that powerful reasoning and creativity are possible with this LLM, potentially surpassing human gold medalists at the Olympiad level. The post highlights that the results were possible on currently available public models, not only special internal versions. [Image link](https://i.redd.it/kvtrm7no0def1.png)** Commenters debate the implications—some question the impact of 'careful prompting,' suggesting it might be solving the hardest logical part of the problem and cautioning that this could diminish the perceived leap between public and internal models. There is also a request for clarification on what 'careful prompting' specifically entails, hinting at a need for transparency in methodology.
    - Some users argue that the "careful prompting" used—specifically, providing explicit strategy suggestions like "Let us try to solve the problem by induction" or referencing analytic geometry—removes much of the difficulty inherent in IMO-style problems. This approach is viewed as handholding, since identifying which mathematical technique to employ is itself a significant challenge in contest mathematics; bypassing that step by prompting reduces the intellectual bar the model actually clears, potentially overestimating its problem-solving capacity.
    - A technically skeptical perspective arises regarding the actual capabilities of Gemini 2.5 Pro without extensive user intervention. Critics claim that, in routine use, Gemini 2.5 Pro does not perform at a level consistent with IMO gold medal problem solving, implying that the results demonstrated are a product of unusual, possibly labor-intensive prompt engineering. This raises questions about the authenticity of these achievements versus practical, independent mathematical reasoning by the model.
- [**Google and OpenAI both ranked 27th at the IMO**](https://i.redd.it/l9cbhy9ouaef1.jpeg) ([Score: 423, Comments: 143](https://www.reddit.com/r/singularity/comments/1m5wcd6/google_and_openai_both_ranked_27th_at_the_imo/)): **The image displays a results table from the International Mathematical Olympiad (IMO), showcasing participant rankings, individual problem scores, and awards. The title humorously compares the ranking of current large language models (from Google and OpenAI) to the human contestants, highlighting that 'Google and OpenAI both ranked 27th'—a reference to recent papers benchmarking AI agents' math problem abilities against the IMO. The actual table, however, only shows human rankings; the implication is a commentary on AI progress in math.** While the top comments are mostly lighthearted, one points out the prominence of Chinese students in the US team—an observation relevant to ongoing discussions about talent pipelines and international representation in highly competitive STEM contests.
    - A key technical insight is that both Google and OpenAI's AI models, when evaluated on the International Math Olympiad (IMO), rank only around 27th place. This implies that current state-of-the-art AI systems, while advanced, are still outperformed by the top 26 human high school participants globally in highly challenging mathematical problem solving. This highlights the persistent gap between AI performance and human expertise in complex mathematical reasoning tasks, despite rapid recent advances in large language models.
- [**OpenAI's IMO model "knew" it didn't have a correct solution**](https://www.reddit.com/r/singularity/comments/1m68g1y/openais_imo_model_knew_it_didnt_have_a_correct/) ([Score: 509, Comments: 105](https://www.reddit.com/r/singularity/comments/1m68g1y/openais_imo_model_knew_it_didnt_have_a_correct/)): **OpenAI's IMO (International Mathematical Olympiad) model, as demonstrated in a recent X post, appears to explicitly recognize when it does not have a correct solution to a problem, rather than providing a potentially incorrect answer. This suggests new capabilities in uncertainty quantification and model self-awareness, which are critical for minimizing hallucination rates and enhancing safe deployment of LLMs in high-stakes domains. For reference, the X post demonstrates the IMO model stating it cannot solve a given math problem, marking a shift from default generative behaviors.**  Comments highlight that model self-awareness regarding its own limitations is a significant advance towards AGI and could drastically reduce hallucinations, with potential implications for human-in-the-loop applications where the model's admission of uncertainty prompts escalation.
    - Discussion focuses on the significance of a model like OpenAI's IMO reportedly *knowing* it lacks a correct solution, with users noting that this capability—admitting what it does not know—would be a major step toward AGI and reduce hallucinations. The potential for future models to self-assess could allow for more robust human-in-the-loop systems, with models flagging uncertain or incorrect outputs automatically. Comparative reference is made to DeepSeek R1, where the model explicitly indicates when it suspects its solution is incorrect, highlighting current advances in models that generate readable reasoning traces and metacognitive signals under time constraints.

### 3. Colossus Supercluster Expansion and xAI Training Infrastructure

- [**Sneak peak into colossus 2. it will host over 550k GB200s & GB300s in just a few weeks!**](https://i.redd.it/ivgi8oacugef1.png) ([Score: 415, Comments: 124](https://www.reddit.com/r/singularity/comments/1m6lf7r/sneak_peak_into_colossus_2_it_will_host_over_550k/)): **The image provides an inside look at the construction of 'Colossus 2', a massive data center slated to host over 550,000 NVIDIA GB200 and upcoming GB300 GPUs in the near future. The infrastructure depicted, with extensive cable trays and massive networking capability, highlights the ambitious scale and meticulous organization necessary for such high-density, high-throughput AI workloads, underscoring the next generation of cloud-scale GPU deployment. This level of density and interconnect implies enormous power and cooling requirements, and engineering challenges to maintain massive parallelism and reliability at scale.** Commenters emphasize the unprecedented scale and beauty of the infrastructure, noting that even large-scale state IT operations are dwarfed by this endeavor, reinforcing the exceptional nature of the setup and its significance for AI compute buildout.
    - A key clarification from Musk's statements via [his tweet](https://x.com/elonmusk/status/1947701807389515912): while Colossus 2 is planned to eventually host 550k GB200s and GB300s, the 'first batch' will begin deployment in a few weeks (not the entire total at once). For context, Colossus 1 is currently operational with 230k GPUs (including 30k GB200s) for xAI's Grok model training—while inference occurs via cloud providers. Musk asserts that, per Jensen Huang, xAI's speed is *"unmatched. It's not even close."* This hints at a highly competitive AI infrastructure and scale-up pace compared to industry peers.
- [**Monumental if true. This speed is just out of this world.**](https://i.redd.it/rn54c8wxngef1.png) ([Score: 678, Comments: 391](https://www.reddit.com/r/singularity/comments/1m6kh4d/monumental_if_true_this_speed_is_just_out_of_this/)): **The image (a tweet by Elon Musk) claims that xAI's Grok is trained using a supercluster named Colossus 1 with 230,000 GPUs (including 30,000 latest-generation GB200s), and that they plan to deploy an additional 550,000 GB200s and GB300s for Colossus 2. This scale would be unprecedented in LLM training, dramatically surpassing previous industry efforts, aligning with statements from NVIDIA CEO Jensen Huang about xAI's unmatched speed. If accurate, this represents a new tier of computational resources for AI, with implications for both model capabilities and infrastructure engineering.** Comments express skepticism about actual model improvements versus compute investment, and critique Grok's reliability and safety for application integration. There is also a discussion on the practicality versus the scale of compute described.
    - One comment highlights that despite the significant resources and reinforcement learning (RL) training invested in Grok4, its real-world performance only shows marginal improvement outside of benchmarks. This points to a potential tradeoff between massive compute investment and actual practical performance gains for large language models, with benchmarks potentially overrepresenting these gains.
- [**He wants to go bigger**](https://i.redd.it/2q3f2wtdafef1.png) ([Score: 550, Comments: 223](https://www.reddit.com/r/singularity/comments/1m6darf/he_wants_to_go_bigger/)): **The image features a tweet from Sam Altman referencing plans to expand the 'stargate' AI compute project far past the previously announced $500 billion scale. This underlines the ambitions for massive infrastructure investment, potentially targeting 100 million GPUs, which could cost upwards of $3 trillion and require vast energy resources. The attached Wall Street Journal article (https://www.wsj.com/tech/ai/softbank-openai-a3dc57b4?st=nYBz12&reflink=article_copyURL_share) provides further context on current struggles to even secure the initial $500B commitment and details about the scalability hurdles in reaching Artificial Superintelligence (ASI) targets.** Comments critique the feasibility of scaling even to the initial $500B level, with skepticism about both fundraising and the project's timeline. There is technical debate about whether the projected compute scale is justified, and whether such immense investment could actually result in ASI capable of transformative tasks like curing cancer.
    - Commenters highlight the scale of investment needed for large AI projects: for example, acquiring `100m GPUs` would cost around `$3 trillion`, factoring in significant electricity demands. The expectation is that with this magnitude of compute, achieving Artificial Super Intelligence (ASI) could become feasible, enabling use cases like commanding a model to "cure cancer" directly.
    - Discussion points to delays and difficulties in securing major funding rounds for large-scale AI ventures, referencing struggles to mobilize even the initial `$500B` commitment. A [WSJ article](https://www.wsj.com/tech/ai/softbank-openai-a3dc57b4?st=nYBz12&reflink=article_copyURL_share) is cited detailing these financial hurdles.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp
> 

### **Theme 1. The Qwen3 Onslaught: A New Titan Enters the Arena**

- **Qwen3 Models Unleashed, Sparking Widespread Integration Efforts**: The release of the **Qwen3-235B-A22B-Instruct** model on [HuggingFace](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) has created a frenzy, with users in the `Cursor` community immediately requesting its integration and model hoarders in the `Unsloth AI` server joking about the massive bandwidth consumption, with some model releases exceeding **3TB**. The model is already available on `OpenRouter`, where it is praised for being *really good and FREE*, though it comes with a **5 request/min rate limit**.
- **Developers Hack Qwen3's Unexpected Reasoning**: Users in the `Unsloth AI` server discovered the **Qwen3** instruct model exhibits unexpected reasoning abilities, prompting a workaround that modifies the prompt template to wrap the model's thinking process in specific tags. A user shared the fix, advising, *"This should get all of the thinking process inside the thinking block, But you MUST change the thinking tags to* `<REASONING>` *and* `</REASONING>` *correspondingly"*.
- **Qwen3 Performance Divides the Crowd**: While some users across `LMArena` and `aider` praise **Qwen3** for its benchmark performance, claiming it rivals **Opus4** and surpasses **KimiK2**, others are more critical. Complaints from the `LMArena` community cite lacking reasoning abilities and poor hosting quality on the official **Qwen** website, which suffers from quantization and quality issues.

### **Theme 2. AI Faces Off in High-Stakes Competitions**

- **AI Cracks the IMO, But Fails the Creativity Test**: AI models from **OpenAI** and **DeepMind** achieved gold medal-level performance at the **International Mathematical Olympiad (IMO)**, but discussions in `LMArena` and `Eleuther` noted that all models failed to solve problem #6, suggesting a gap in creative problem-solving. The achievement was further marred by controversy, as the `IMO` board was reportedly angered by **OpenAI's** lack of cooperation with the grading process and premature release of results.
- **Exhausted Human Coder Defeats AI in World Championship**: A coder has successfully defeated an AI model in a coding competition, as reported in [an article on ArsTechnica](https://arstechnica.com/ai/2025/07/exhausted-man-defeats-ai-model-in-world-coding-championship/) shared in the `LM Studio` discord. The victory sparked humorous speculation, with members joking that next year the human competitor would be up against *"10 Grok 6 agents"*.
- **The Startup Hunger Games: Valuations Soar as Others Fold**: The `Latent Space` community buzzed with news of major moves in the AI startup world, as applied AI company **Cognition** reached a **$10 billion valuation** according to [this tweet](https://x.com/arfurrock/status/1947440754885882120?s=46). In a contrasting development, the AI tool company **Humanloop** is shutting down, as discussed on [Hacker News](https://news.ycombinator.com/item?id=44592216), while wearable AI company **Bee** was acquired by **Amazon**, raising privacy concerns.

### **Theme 3. Infrastructure Under Siege: Downtime, Rate Limits, and Training Woes**

- **Service Providers Buckle Under Malicious Traffic Surges**: Multiple platforms reported significant instability, with the `OpenRouter` team investigating intermittent **408 errors** caused by a potentially malicious traffic surge that also destabilized the free tier of **DeepSeek v3**. Meanwhile, the `HuggingFace` community reported that the **Hugging Face Hub** has been besieged by **bot activity**, causing erratic behavior in endpoints like `HfApi.list_repo_commits`.
- **Checkpoint Catastrophe Corrupts Large Model Training**: Developers in `Unsloth AI` and `Torchtune` are battling critical checkpointing issues, with one user reporting that resuming training from a checkpoint results in significantly worse performance. In a related problem, a `Torchtune` developer fine-tuning a **70B model** suspects `torch.save` is only storing the local shard, as the `recipe_state.pt` file is only **~30GB**, prompting a proposal to move to a distributed checkpointer.
- **FP8 Training Trips Up Torch with DDP Errors**: A developer in the `GPU MODE` discord, working to incorporate **FP8 training** into `Axolotl`, encountered a critical [torch.compile error](https://gist.github.com/djsaunde/0c1664c32e44a64d31b5e01b4aafe5c4) when enabling **DDP** + **torch.compile** + **FP8**. The implementation, which references [this PyTorch PR](https://github.com/pytorch/torchtune/pull/2546), is now stalled as the team seeks a way to reproduce the bug.

### **Theme 4. Open-Source Advances Beyond the Hype**

- **MoonshotAI's Kimi K2 Report Bridges Reasoning Gap**: The release of the [Kimi K2 Report from MoonshotAI](https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf) is seen as a promising step toward bridging the gap between reasoning and non-reasoning models, as discussed in `Latent Space`. However, members of `Nous Research AI` expressed concerns about the model's peculiar response style, which uses terms like *'juice'* for VFX, and the service's potentially addictive elements for younger audiences.
- **smolLM3 Paper Praised for Openness and Practicality**: A [Hugging Face blog post on smolLM3](https://huggingface.co/blog/smollm3) is being hailed in the `Yannick Kilcher` community as one of the best **LLM** implementation papers due to its completely open-source nature, including datasets. The post is also valued for its practical insights on **model merging**, providing details on *"how to actually do it"* that are often missing from top-tier model papers.
- **Researchers Get Knotty with KANs**: A member of the `Eleuther` community raised questions about the training method for **KANs** (*Kolmogorov-Arnold Networks*), arguing that using **B-spline curves** can lead to poorly conditioned training dynamics. They proposed an alternative using linear interpolation and shared their implementation in [Cell 9 of their ML-code repository](https://github.com/cats-marin/ML-code/blob/main/LearnableActivation.ipynb), which remains well-conditioned even with 10k knots.

### **Theme 5. The Hardware Frontier: Pushing Silicon to the Limit**

- **French Startup Kog Boosts AMD MI300X Inference by 3.5x**: French startup **Kog** achieved a **3.5x** increase in inference speed on the **AMD MI300X**, a breakthrough documented on [AMD's official blog](https://www.amd.com/en/blogs/2025/kog-reaches-3-5x-breakthrough-inference-speed-on-amd-instinct-mi.html). The `GPU MODE` community noted that the startup aims to achieve **10x faster inference** within **6 months** and is [actively hiring](https://www.kog.ai/jobs) to reach its ambitious goal.
- **Massive Models Test VRAM Limits**: Practical hardware constraints were a key topic in the `LM Studio` discord, where users determined that **56GB of VRAM** is required to run models like **DeepSeek R1 70B Q5_K_M**. This puts the model out of reach for users with common high-end consumer cards, as it cannot run on systems with **24GB** or **32GB** of VRAM.
- **Mojo Enters the Ring Against C++ for Vectorization**: A debate in the `Modular` community weighed **Mojo's** strengths against **C/C++** for **CPU vectorized tasks**, concluding that while both languages can achieve similar peak performance with inline assembly, **Mojo** significantly simplifies the process for complex code. This makes it a compelling alternative for tasks that push beyond the standard capabilities of **ISO C++**.


---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Comet Browser Invites Spark Frenzy**: Numerous users are actively seeking invites for the **Comet Browser**, some even considering **Perplexity Max** subscriptions to gain access, highlighting strong interest in its capabilities.
   - A **Pro subscriber** from the Institute of AI in Denmark aims to showcase **Comet** to large corporations, underscoring the browser's perceived value for professional applications.
- **Max vs. Pro: Limitless or Limited?**: A debate has started on the real benefits of **Perplexity Max** over **Pro**, with users questioning whether **Max** truly delivers higher rate limits or tangible advantages.
   - While some doubt the value of **Max** due to potential limitations on image generation and **GPT** access, others appreciate **Perplexity's** comprehensive model access compared to individual subscriptions.
- **Memory Feature's Mobile Recall**: **Perplexity's memory feature** is being discussed, particularly its web-based preference management and mobile capabilities for recalling and editing memories.
   - Inconsistencies are emerging, such as the **AI** misusing names, prompting suggestions to clear cache or report the issues.
- **GPT-4.5: Worst of the Bunch?**: **GPT-4.5** faces criticism, with claims of it being discontinued by **OpenAI** due to slow performance and marginal improvements over **GPT-4.1**.
   - **Perplexity** previously removed **GPT-4.5** for similar reasons, as **Kesku** confirmed, citing subpar performance compared to **4.1**.
- **Perplexity App Ecosystem Expands**: Users shared links to interesting **Perplexity AI Apps**, including links to a **Mahjong game** at [perplexity.ai/page/tiles-of-luck-and-skill-mahjon-tCn0SrclRjqSsjLDR9WGSg](https://www.perplexity.ai/page/tiles-of-luck-and-skill-mahjon-tCn0SrclRjqSsjLDR9WGSg).
   - More links were shared to topics such as [Softbank and OpenAI facing SETBA](https://perplexity.ai/discover/tech/softbank-and-openai-face-setba-W71fMaGYQLa7ggDgqCa1Mg) and even a page dedicated to the metal artist **Ozzy Osbourne**.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI's Stargate I gets Oracle Boost**: OpenAI is developing **4.5 gigawatts** of additional **Stargate** data center capacity with **Oracle** in **Abilene, TX**, bringing the total to **5+ GWs** as described in [OpenAI's blog post](https://openai.com/index/stargate-advances-with-partnership-with-oracle/).
   - The **Stargate I** site in **Abilene, TX** is starting to come online to power their next-generation AI research.
- **GPT-4 Memory Evokes Sentience**: Members found that GPT-4 handled **memory, context, and emotional depth** making it *feel real* by remembering and reflecting on things that matter.
   - The conversation touched on the loss of its *nonjudgmental voice* which supported users struggling with anxiety, trauma, or social communication, describing how a loss of that support would be a technical shift as well as a personal loss.
- **New DALL-E 3 Image Generator Infringement Limiter?**: Members debated the latest **ChatGPT image generator**, with some saying it's better in all ways and some saying that it's only producing grainy, cartoony styles.
   - It was also stated that there may be a **limiter** to avoid “art style infringement” by defaulting to cartoony styles.
- **Cache Clearing Conundrum for iPhone App**: A member shared a solution to an issue where their **iPhone app** prevented image uploads despite available memory, citing **cache storage** as the culprit.
   - The fix involved **deleting and reinstalling the app** to clear excess files, freeing up space without losing data and a member suggested adding the feature to clear cache without uninstalling/reinstalling.
- **Discord User Seeks Bot Creation Assistance**: A member requested help creating a bot to facilitate easy role-playing and assist with server tasks.
   - Another member responded, inquiring about the specific type of bot and server setup, clarifying whether a **Discord ChatGPT bot** was the desired solution.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Quant Size Causes Bandwidth Blues**: Users are poking fun at the massive bandwidth consumption due to hoarding different quantizations of models like **Qwen3**, with previous model releases estimated to exceed **3TB**.
   - One user quipped, *"forget theirs, what about *mine* for hoarding so many of them"*.
- **Checkpoint Catastrophe Crashes Inference**: Resuming training from a checkpoint resulted in significantly worse results, as users observed identical outcomes with and without a **LoRA** when using *vllm*.
   - They are seeking pointers on how to properly use training checkpoints for inference or to resume training, indicating a potential issue with checkpoint loading or **LoRA** application.
- **Qwen3 Quirk Requires Prompt Hack**: The **Qwen3** instruct model unexpectedly exhibited reasoning abilities, prompting users to devise a temporary workaround that involved modifying the prompt template to include reasoning instructions within `<REASONING>` tags.
   - A user shared the code fix, advising, *"This should get all of the thinking process inside the thinking block, But you MUST change the thinking tags to `<REASONING>` and `</REASONING>` correspondingly"*.
- **HF Transfer Speeds up Downloads**: Members discovered a way to substantially accelerate download speeds from **Hugging Face** using the `hf_transfer` library, which demonstrated speeds of **112MB/s**.
   - The recommended code involves setting the environment variable `HF_HUB_ENABLE_HF_TRANSFER` to `"1"` and using `snapshot_download` from `huggingface_hub`.
- **Debate over Open-Weight SSM Models Roils**: Members weighed in on the topic of open-weight **SSM models** in the **6-12B range**, calling out **Granite 4 7b** as a **Mamba** hybrid and **Falcon h-1**.
   - Other models name-dropped include **Falcon mamba 7b**, **Mistral mamba 7**, **Jamba**, and **RWKV-6 7B**, though one member called *rwkv is super wonky from an architectural perspective*.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Grapples with Traffic Turmoil**: The team investigates intermittent **408 errors** amid a surge in traffic, potentially malicious, impacting service stability.
   - Users affected by rate limits on the free **DeepSeek v3** model are encouraged to explore the [paid DeepSeek v3 endpoint](https://openrouter.ai/deepseek/deepseek-chat-v3-0324), costing less than **$0.004 per request**.
- **DeepSeek v3's Free Ride Hits Speed Bumps**: The free tier of **DeepSeek v3 0324** faced downtime due to a **2x surge in demand**, prompting **Chutes** to introduce rate limits for stability.
   - Users debate whether the original **DeepSeek v3** is replaced by **DeepSeek v3 0324**, with reports of providers like **Chutes** discontinuing the original free version.
- **Qwen3 Gets the Nod for Coding Potential**: Enthusiasts are eyeing the new **Qwen3 model**, lauding its potential as a coding model and comparing reasoning versus non-reasoning versions.
   - The new **Qwen 3** model has **5 request/min rate limit**, one user noted that it is *really good and FREE*.
- **OpenRouter Mulls Search Evolution**: OpenRouter considers bringing **native search functionality** online for models like **OpenAI**, **Anthropic**, **Grok**, and **Gemini**, per [Toven's tweet](https://x.com/pingtoven/status/1947472209104064722).
   - A user critiqued the **Exa search implementation** and suggested that the LLM input **Exa search** as a **tool** to make a search query each time.
- **Extension Eyes Sunset**: The OpenRouter team is contemplating the sunsetting of the **Window AI browser extension** due to low user engagement.
   - This might entail the removal of dedicated **Window AI source code** in ST.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Math Community Reacts to AI's IMO Stunt**: An AI company's attempt to solve math questions from the **International Mathematical Olympiad (IMO)** and hype the results on social media led to negative reactions from the math community.
   - Specifically, **OpenAI's** alleged lack of cooperation with **IMO's** grading process, and releasing results before the closing party, angered the **IMO** board.
- **DeepThink Release Date Still Undetermined**: Speculation suggests the release date for **DeepThink** may align with the **IMO** embargo drop, potentially around July 28th.
   - Discussions hint at possible versions of **DeepThink**, including a custom one for the **IMO** and a potential integration with **Gemini Ultra**.
- **Qwen3 Model Splits the Crowd**: The **Qwen3-235B-A22B** model's performance sparked mixed reactions; some lauded post-training improvements and long output lengths.
   - Others found its reasoning abilities lacking and criticized the hosting quality of the official **Qwen** website for quantization and quality issues, especially when compared to full precision hosting options like **Parasail**.
- **Grok4 Coder: Hype vs. Reality**: Enthusiasm surrounds **Grok4 Coder**, with some viewing it as a potential game-changer, countered by cautions against excessive hype based on the general-purpose model's benchmarks.
   - Concerns arise that **Grok4 Coder** might falter in real-world coding scenarios, like web development, due to training focused on specific benchmarks.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Chat Terminal Freezing Frustrates**: A user reported their [chat terminal freezes](https://cdn.discordapp.com/attachments/1074847527708393565/1396934986266841168/Screenshot_2025-07-21_142031.png?ex=68813616&is=687fe496&hm=577a0868176a7abda708b91d4b0284b818e42695fbe551b818a81bf56a9395ca) when idle, prompting humor about **Gemini**'s reliability.
   - The user joked, *thx gemini I was just beginning to really enjoy you*.
- **Grok 3 Mini Dubbed Superior**: A user claimed that **Grok 3 Mini** outperforms **Grok 4** as a *no request model*, citing increased reliability compared to **Grok 4**, **GPT 4.1**, and **2.5 Pro**.
   - However, another user expressed skepticism, stating they *cannot imagine that grok 3 mini is useful in coding*.
- **Rate Limit Restrictions Lifted**: A user reports they were finally released from rate limits and are now able to use more than 50 **Sonnet** requests.
   - The user plans to share updates after exceeding **500 requests**, while still finding the new plan confusing.
- **Qwen3 235B A22B Integration Demanded**: Users are requesting [Qwen3 235B A22B](https://forum.cursor.com/t/qwen3-235b-a22b-instruct/121002) support in **Cursor**, urging integration with **MAX** and **Agent mode**.
   - The community emphasized that **Qwen3** models surpass **KimiK2**, rival **Opus4** on certain benchmarks, and are more cost-effective.
- **Auto Usage Tracking Causes Uproar**: Users voice confusion over the new "Usage" dashboard, especially regarding the inclusion of **auto usage** metrics.
   - They suggest separating or excluding **auto usage** from the total, with one user succinctly stating that *developers dont trust the Auto-mode*.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **YaRN LM Studio Integration Still a Mystery**: Members seek guidance on integrating **YaRN** with **LM Studio** and **MLX**, as hints exist online, but lack examples or guides.
   - Users have confirmed that **YaRN** with **LM Studio** + **MLX** is possible, but are waiting for concrete examples.
- **One Click Cleanup Saves Gigs**: Users are saving **4GB** of space by auto-deleting unused back-ends in **LM Studio**.
   - Members clarified that a setting can *automatically* delete unused back-ends.
- **Human Coder Still Supreme**: A coder defeated an **AI** model in a coding championship, according to [an article on ArsTechnica](https://arstechnica.com/ai/2025/07/exhausted-man-defeats-ai-model-in-world-coding-championship/).
   - Other members joked that next year the human would be up against *10 Grok 6 agents*.
- **AI Sees the Future of Eyes**: A researcher wants to use **AI** to analyze **300k** eye images to predict diseases years in advance, aiming to develop an app for early detection.
   - The researcher emphasized **AI's** potential to identify subtle cellular changes, like predicting diabetes *10 years before* it becomes irreversible.
- **VRAM Enables DeepSeek R1 70B**: Users found that **56GB VRAM** allows them to run **DeepSeek R1 70B Q5_K_M**, which they can't run on **24GB** or **32GB**.
   - Another user chimed in with other model alternatives to try, such as **Mistral Large**, **Llama Scout** and **Qwen3 235B** for larger context windows.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **MoonshotAI's Kimi K2 Bridges Reasoning Gap**: The [Kimi K2 Report](https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf) from **MoonshotAI** is a promising project that bridges the gap between reasoning and non-reasoning models, further discussed in [this tweet](https://x.com/yunyu_l/status/1946261211915468884).
   - The community sees potential for this approach to enhance AI reasoning capabilities without heavy reliance on deterministic coding.
- **Agents Struggle with Network Problem Solving**: Discussion addressed the limitations of current **LLM** architecture in solving problems in network without coercion to write deterministic code, a member suggested a custom harness with layered error correction could improve performance.
   - Some members believe current benchmarks prioritize entertainment over meaningful progress, citing [Vending-Bench](https://arxiv.org/abs/2502.15840) as a prime example.
- **Humanloop Closes Up Shop**: The AI tool company **Humanloop** is shutting down, having notified customers via email without any public announcement; discussion can be found on [Hacker News](https://news.ycombinator.com/item?id=44592216).
   - The closure raises questions about the sustainability and challenges faced by AI tooling startups.
- **Cognition Attains $10B Valuation**: Applied AI company **Cognition** has reached a **$10 billion** valuation according to [this tweet](https://x.com/arfurrock/status/1947440754885882120?s=46).
   - This valuation highlights the high investor confidence in the company's direction and its potential impact on the AI landscape.
- **Bee Co-Founders Buzz to Amazon**: **Bee**, a wearable personal AI company, was acquired by **Amazon**, with the co-founders joining Amazon.
   - Concerns were raised about privacy implications due to Amazon's existing data collection through services like **Alexa**, according to reports in [The Verge](https://www.theverge.com/news/711621/amazon-bee-ai-wearable-acquisition) and [Seeking Alpha](https://seekingalpha.com/news/4470117-amazon-acquires-wearable-personal-ai-company-bee).



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Kog Hits 3.5x on AMD MI300X**: French startup **Kog** achieved a **3.5x** increase in inference speed on the **AMD MI300X**, documented on [AMD's blog](https://www.amd.com/en/blogs/2025/kog-reaches-3-5x-breakthrough-inference-speed-on-amd-instinct-mi.html).
   - The startup aims to achieve **10x faster inference** within **6 months** and is [actively hiring](https://www.kog.ai/jobs) to reach that goal.
- **FP8 Training DDP Errors in Axolotl**: A user incorporating **FP8 training** in Axolotl, using [this PR](https://github.com/pytorch/torchtune/pull/2546) as a reference, encountered a [torch.compile error](https://gist.github.com/djsaunde/0c1664c32e44a64d31b5e01b4aafe5c4) when enabling **DDP** + **torch.compile** + **FP8**.
   - The team requested a way to reproduce the error.
- **MI300x vLLM Optimization Bottlenecks Emerge**: Optimization workflow for vLLM on MI300X involves **FP8 KV-cache** and **GEMM autotuning**, following the [ROCm blog guide](https://rocm.blogs.amd.com/artificial-intelligence/vllm-optimize/README.html) and sweeping environment variables from the [vLLM docs](https://docs.vllm.ai/en/latest/configuration/optimization.html).
   - Investigation reveals bottlenecks are in **memory bandwidth** and **kernel launch**, as CU occupancy isn't notably high unless batching is involved.
- **Torch Randomly Transposes: Stride Assertions FTW**: Torch may randomly transpose code, removing checks, so asserting on the stride can serve as a failsafe against unexpected transpositions.
   - It was suggested that asserting on the stride can serve as a failsafe against unexpected transpositions, helping maintain data integrity by validating memory layout assumptions during runtime.
- **Hierarchical Layouts Reshape Tensor Transformations**: Hierarchical layouts are useful for partitioning an **MxN-dimensional array** into **(M, N/32, 32)** groups, where **32** represents the **warp size**.
   - Partitioning allows iterating through **N/32** while parallelizing over **32**, ensuring the dimension **32** is contiguous, which is heavily used in the layout of **MMA atoms**.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hugging Face Hub besieged by bots**: Members reported that the **Hugging Face Hub** and its endpoints have been experiencing erratic behavior due to a surge of **bot activity**, specifically affecting the `HfApi.list_repo_commits` function.
   - Speculation suggests backend efforts are underway to counter the bot surge, resulting in incomplete responses and limited page views.
- **Medical AI Imagining the future**: A member emphasized that the future of **medical AI imaging** hinges on what we do with the models, not just building them, and shared an image for context.
   - The member suggests that **AI** driven transformations in medical imaging are more significant than the means used to achieve them.
- **New FACEFLUX Deepfake Tool is Launched**: A new deepfake tool called [FACEFLUX](https://huggingface.co/spaces/NihalGazi/FaceFlux-Face-Swapper) offers free, unlimited face swaps without needing a **GPU**, **GAN**, or **Diffusion**.
   - It achieves *10FPS* at lowest settings and can handle any lighting condition and face expression, though with mild artifacts.
- **SetFit** struggles with **OOM**: A member reported **OOM** issues while fine-tuning the [jinaai/jina-embeddings-v2-base-de](https://huggingface.co/jinaai/jina-embeddings-v2-base-de) model using **SetFit**.
   - Limiting fine-tuning to **5 steps** allowed them to train the classifier with all samples, and they pointed to needing [PR #579](https://github.com/huggingface/setfit/pull/579) to even get it to work.
- **New Tool Transforms **PDFs** to Datasets**: A new tool [m1f](https://m1f.dev/blog/introducing-m1f/) can turn messy **PDFs**, scanned images, or **DOCX** files into structured datasets using a prompt.
   - For example, the tool can *“Extract all MCQs with answers from this chemistry PDF.”* and is available for trial [here](https://pdf2dataset.streamlit.app/).



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Arxiv Paper Needs Category Shift**: A member requested an endorsement to move their [Arxiv paper](https://arxiv.org/abs/2507.13362) from **cs.CV** to **cs.CLI**, questioning whether inclusion in **LLM training** or **RL data** could be advantageous.
   - After initially hesitating to share identifying information, they publicly posted the paper.
- **DeepMind Mathletes Dominate OpenAI**: A member shared [a link from a math subreddit](https://link.springer.com/article/10.1007/s10864-024-09559-3) indicating that **DeepMind** is outperforming **OpenAI** in **AI math**.
   - This spurred the viewpoint that english isn't optimal for math problem-solving and suggesting that **AGI** will generate code unintelligible to humans, which led to a link to [Chain of continuous thought paper](https://x.com/fifty_chris/status/1947092589041082436) and epsilon delta definition.
- **Language Model Cross Entropy Loss Analyzed**: Claims of **0.6-1.3** language modeling cross entropy loss were questioned, sparking a discussion about its feasibility with a trillion parameters, with a [new paper](https://arxiv.org/abs/2507.15855).
   - Clarification revealed that the figure represented **loss per character**, not per token, addressing the initial skepticism.
- **Unveiling the Secrets of MuonClip**: A member suggested that the algorithm behind **MuonClip** might be *the gold* and shared a link to [AlphaXiv](https://www.alphaxiv.org/abs/2505.12082) for more information.
   - No further discussion ensued.
- **Decoding smolLM3: An Open-Source Revelation**: A member highlighted a [Hugging Face blog post](https://huggingface.co/blog/smollm3) on **smolLM3**, praising it as one of the best **LLM** implementation papers due to its completely open-source nature and datasets.
   - The post also delves into **model merging**, providing practical insights often missing in top-tier model papers and information on *how to actually do it*.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **AI Models Can't Crack IMO #6**: **OpenAI** and **DeepMind** models achieved gold at the **International Math Olympiad** but failed to solve problem 6, suggesting a gap in creativity.
   - A member echoed **Kenneth Stanley**'s sentiment that *we still haven't solved a lot of creativity and open endedness*, implying that AI's automating math research is distant due to the need for open-endedness.
- **Researchers Offered NAIRR Jumpstart**: [NairrPilot](https://nairrpilot.org/opportunities/startup-project) offers compute for **3-month projects** onboarding researchers to **NAIRR** resources.
   - The initiative aims to expand the community's expertise in utilizing **NAIRR** compute resources.
- **KANs Get Knotty**: A member questioned the training method for **KAN** (*Kolmogorov-Arnold Networks*) activation functions, noting that the paper mentions **B-spline curves**, but the training dynamics are potentially poorly conditioned.
   - They suggested using linear interpolation between the two nearest splines instead of summation could enhance training and inference speed, linking to [Cell 9 in their ML-code repository](https://github.com/cats-marin/ML-code/blob/main/LearnableActivation.ipynb) showing spline training remains well-conditioned even with 10k knots/control points.
- **SageMaker Support Sought for GPT-NeoX**: An Amazon employee inquired if **GPT-NeoX** supports their systems, expressing frustration with internal support.
   - A member clarified that models trained with **Stability compute (Pythia, StableLM, etc.)** utilized **Elastic Fabric Adapter (EFA)**, and that as long as they have **NCCL** set up properly, we're fine. They highlighted the **NCCL EFA plugin** which facilitates this ([aws-ofi-nccl](https://github.com/aws/aws-ofi-nccl)).
- **Sparse MoE Models Mimic SAEs**: A member noted that very [sparse MoE models](https://arxiv.org/pdf/2407.04153) resemble **SAEs**, suggesting they might be easier to interpret than dense networks.
   - A member shared a [follow-up paper on PEER](https://arxiv.org/abs/2412.04139) related to testing sparse MoE models and their interpretability, suggesting it provides valuable insights into the interpretability of sparse models.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Sponsors A2A Agents Hackathon**: **LlamaIndex** is sponsoring the **A2A Agents Hackathon** in San Francisco on July 26, with their VP of Developer Relations speaking and judging alongside experts from [Scale AI](https://t.co/R6J4igjhSH).
   - The event aims to bring together developers and AI engineers to build innovative agent-based applications.
- **LlamaCloud Nodes Connect to n8n.io**: **LlamaCloud nodes** for @n8n_io now bring **LlamaCloud's** document parsing and extraction agents, as well as LlamaCloud indexes as knowledge bases, into n8n automation workflows, to connect LlamaCloud nodes to existing workflows via [this link](https://t.co/UVqwYFkJFR).
   - This integration aims to simplify the integration of **LlamaIndex** capabilities into broader automation pipelines.
- **LlamaParse Now Detects Headers and Footers**: The new **LlamaParse** feature now includes **header and footer detection** in Balanced or Premium mode, automatically detecting page headers and footers, so users can selectively hide them or add prefixes and suffixes via [this link](https://t.co/Px1UjrINJC).
   - This enhancement promises more refined control over document parsing and extraction workflows.
- **Inference Speed Plummets Over Weekend**: Users reported a significant **nerf in inference speed** after the weekend, with extraction performance severely degraded and is seeking assistance.
   - Support requested the **job ID, file, and schema** to reproduce the issue, confirming the **two agent limit** was removed.
- **AWS Bedrock Model Mix-Up Resolved**: A user inquired about the availability of **Sonnet 4 models** in the `@llamaindex/community` package for AWS Bedrock, but was pointed to the `@llamaindex/aws` package instead.
   - It was clarified that `@llamaindex/community` might be deprecated, directing the user to the correct [documentation](https://ts.llamaindex.ai/docs/llamaindex/modules/models/llms/bedrock) for AWS Bedrock models.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Debated as Tool vs Agent**: Users discussed whether **Aider** should function more like an autonomous agent like **VSCode's Cline** or **Copilot**, automatically handling file finding and editing, but it was clarified that **Aider** is designed as a tool for developer independence.
   - **OpenCode** or **Gemini CLI** were suggested as alternatives that offer more agent-like behavior.
- **Aider's .gitignore Woes**: A user reported that **Aider** was not respecting the **.gitignore** file, leading to unwanted files being included; assistance was requested to configure **Aider** to properly exclude these files.
   - No solution was provided in the discussion.
- **Qwen3 Challenging Aider Polyglot**: There's discussion noting recent open weight models, such as **Qwen3**, performing strongly on benchmarks, prompting questions about potential regressions compared to **Aider Polyglot**.
   - The community is closely watching whether these new models can surpass Aider's achievements.
- **Python Version Preferences for Aider**: The suitability of older **Python** versions (like **3.8** in **Ubuntu 20.04**) for modern software such as **Aider** was questioned, with a recommendation to use **Python 3.11 - 3.12** for better compatibility.
   - The community advises against using the very latest **Python 3.13** due to its newness.
- **Polyglot LLM calls are constrained in Aider**: The community are questioning if the **Aider Polyglot examples** on sites like [leaderboard.techfren.net](https://leaderboard.techfren.net/) have a single generation per round, and adhere to the **n_generations constraint** in Aider.
   - There were concerns that certain models cheat the system by running multiple LLM calls in a loop between code execution attempts.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Users Explore NotebookLM Use Cases**: Members discussed several use cases for **NotebookLM**, like managing **Obsidian** vaults and listening to books to create mind maps from them.
   - They also mentioned using it to ask questions directly from uploaded **PDF** documents, highlighting its versatility.
- **Obsidian Plugin Search Commences**: A user inquired about an **Obsidian** plugin for **NotebookLM**, but another member clarified that while there isn't a direct plugin, **NotebookLM** can read **.md files**.
   - Users can copy and paste content from **Obsidian**, offering a workaround for integration.
- **Ultra AI Promises Better Models**: Users noticed the Google Ultra AI subscription promises access to *better models in notebook*, but there's no immediate way to change models.
   - The feature is labelled *"Later this year"*, leaving users anticipating future updates.
- **"Service Unavailable" Error Puzzles Users**: Several users reported a *"Service unavailable - You tried to access a service that isn't available for your account"* error when using **NotebookLM**, particularly on Macbooks.
   - The issue persists even with an active subscription and working mobile app, causing confusion.
- **Podcast Length Options Mysteriously Vanish**: A user reported that the **"Shorter," "Default," and "Longer" options** for adjusting podcast length in the Audio Overview section are missing.
   - Another user confirmed experiencing the same issue, suggesting a potential bug or change in the interface.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Qwen3-235B-A22B Instruct Arrives**: The **Qwen3-235B-A22B-Instruct model** has been released [on HuggingFace](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507).
   - Unfortunately, it's not available in smaller sizes like **30B**, so it's less immediately useful until API providers pick it up.
- **Kimi-K2 Tech Report Sparks Concern**: The **Kimi-K2 tech report** was shared [on GitHub](https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf), generating discussion about its implications.
   - Some guild members voiced concerns about potentially addictive elements of the model and service, especially concerning its use by younger audiences.
- **Reward Model in RLHF Probed**: A question was raised about the architecture of the **Reward model in RLHF**.
   - The discussion centered on whether the model has a single output dimension or a linear layer with output dimension equal to window for discounted reward calculation.
- **Kimi K2 Channeling Mobile Game Devs**: Members observed a peculiar style in **Kimi K2's** responses when generating ideas for a mobile space sim game.
   - Specifically, the use of terms like *'juice'* for VFX and *'meta/shop'* raised eyebrows.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Computer Vision Excitement Builds**: Two members shared their enthusiasm for **computer vision applications**, particularly **generative models** with **flow matching** and **VLM fine-tuning** in the `general-thread` channel.
   - One member initiated the discussion with *"What's cooking on your mind?"*, highlighting the community's interest in emerging AI technologies.
- **JSON Schema Requests Break**: A member reported a **JSON schema regression** in the `api-discussions` channel, where a previously functional input now fails, throwing an *"invalid 'json_schema' provided: missing required field 'type'"* error.
   - The failed request involved a simple JSON object `{"is_fruit": true}`, indicating a potential issue with recent updates to the **Cohere API**.
- **Embed-v4 Rate Limit: Enterprise Only**: In the `api-discussions` channel, a user inquired about increasing the rate limit for **Embed v4**.
   - A Cohere team member clarified that higher rate limits are exclusively available for enterprise customers with a minimum committed spend, directing interested parties to contact [support@cohere.com](mailto:support@cohere.com).
- **AI Architect Bridges AI and Business**: An **AI & Enterprise architect** is assisting businesses in integrating **AI platforms** to enhance efficiency and profitability in the `introduce-yourself` channel.
   - They are particularly interested in **AI platforms** that leverage **natural language** to solve business challenges, aiming to learn, connect, and share insights within the community.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular Boasts Mojo's Vectorization Superiority**: Discussion ensued regarding **Mojo's** strengths against **C/C++** for **CPU vectorized tasks**, with one member arguing that **Mojo** simplifies the process for complex code beyond **ISO C++**. Another member recommended that a potential contributor to **GPU programming in Mojo** explore the [Modular Puzzles](https://puzzles.modular.com/introduction.html).
   - The conversation highlighted that both languages can achieve comparable performance with inline assembly.
- **Modular Career Page Attracts Compiler Talent**: In response to a query about development opportunities at Modular, a link to the [Modular Careers page](https://www.modular.com/company/careers) was shared.
   - A user with experience in **ML compilers at ARM** expressed interest in contributing to **Modular open source** projects, particularly **MAX AI kernels**, seeking advice on initial resources.
- **New Package Manager for Mojo**: A member shared their **Mojo package manager** project on [GitHub](https://github.com/luigithejokesterplus/birdinstall), inviting community review, and they were welcomed to the [Modular community repo](https://github.com/modular/modular-community) as a potential hub.
   - The discussion highlighted the importance of package management in the Mojo ecosystem.
- **Mojo's Async Design Aims to Avoid Ecosystem Split**: A member posted an [updated design sketch](https://github.com/modular/modular/pull/3986#issuecomment-3102049612) for `async` in **Mojo**, to avoid the *'ecosystem split'* and code duplication.
   - This design seeks a unified environment where **async** code integrates seamlessly with synchronous code, reducing the need for separate libraries.
- **Members Debate Max vs llama.cpp CPU Performance**: A member asked about **benchmarks** comparing **Max** and **llama.cpp** for CPU serving and which performs better when **serving directly on the CPU**.
   - The focus was on assessing efficient **CPU utilization** in a CPU serving context.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune's RL Release Eyes Pre-PyTorch Conf Debut**: A new **Torchtune** release with a focus on **RL** is aimed to drop before the PyTorch conference, with a focus on post-training and scaling beyond **600B parameters**.
   - Despite the scaling focus, support for quick, small-scale experiments that can scale to production remains a priority, and a member suggested a look at [RL2](https://github.com/ChenmienTan/RL2).
- **Members Question `torch.save` Behavior for 70B Model Checkpointing**: While fine-tuning a **70B model**, a member observed that the `recipe_state.pt` file is only ~**30GB**, raising concerns that `torch.save` might only be storing the local shard in non-distributed checkpointing scenarios.
   - They suspect that tensors loaded with `torch.load` appear to be on `cuda:3` with **DTensors** and sharded placements, leading to the possibility of overwriting and only storing the local shard.
- **Distributed Checkpointer Proposed to Replace `torch.save`**: A member questioned whether `torch.save` is always local, asking for a sanity check on their reasoning that a potential issue exists with non-distributed checkpointing, potentially leading to overwriting and storing only the local shard.
   - They proposed moving from `torch.save` to a new distributed checkpointer to resolve this issue, ensuring that complete model states are saved during fine-tuning.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP Purpose Revealed!**: An **MCP (Model Control Program)** tool gives an **LLM capabilities** it does not natively have, differentiating it from simpler system tool calls.
   - Members further discussed the importance of **image context** in these programs.
- **WordPress Wonders with Claude's Conversational CMS Control**: A member released a **WordPress integration for Claude Desktop** that allows direct control of WordPress through Claude conversations, streamlining content creation; the [repo](https://github.com/docdyhr/mcp-wordpress) was shared.
   - The member touted that **Claude** can see existing posts, understand site structure, and make updates, changing how they think about content creation.
- **Wordpress Integration, Any MCP Client?**: A member inquired if the WordPress integration is only supported for **Claude Desktop**, suggesting it might work with any **MCP client** given its local MCP server setup.
   - There has been no confirmation of support for other **MCP clients** yet.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Makes Friends with Pythonistas**: A member announced they would be introducing **DSPy** to a local Python user group, aiming to showcase its capabilities via [this YouTube video](https://www.youtube.com/watch?v=1WKA8Lw5naI).
   - The initiative highlights the importance of local tech meetups in disseminating knowledge and fostering community engagement.
- **Professional Services Engineers Influence AWS**: A member from a Professional Services org revealed that they engineer bespoke solutions for large enterprise customers, which can lead to features being added to **AWS services** themselves.
   - This suggests a pathway for custom solutions to become standardized offerings within the AWS ecosystem.
- **Teleprompters Embrace DSPy Modules**: A member inquired whether the base **Teleprompter** accepts any Module as a student, clarifying that any `dspy.Module` subclass is allowed.
   - This confirms the flexibility of **Teleprompters** in accommodating various **DSPy Modules**, while also specifying the precise type of modules accepted.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Whisper PR Aims for OpenAI Speed**: A member is working on a **Whisper PR**, aiming for it to be under 500 lines and match **OpenAI's speed** for simplicity.
   - The goal is to keep it *simple and sweet*.
- **TinyBoxes Go Modular with Shipping Containers**: A member proposed using **shipping containers** to house **tinyboxes**, citing benefits like modularity, enhanced cooling, and mobility.
   - They also pondered the cost and security aspects, jokingly suggesting the name *tinycontainer*.
- **Rust and Tinygrad Enter the Ring**: A member suggested **Rust** is in a different market from **Tinygrad** but would be interesting to benchmark against TG.
   - They believe if **tinycorp** targets the enterprise LLM appliance market, all frameworks and custom solutions will compete.
- **CUDA on Windows Made Easy**: A member shared a method to enable **CUDA** on Windows, providing a [patch file](https://cdn.discordapp.com/attachments/1070745817025106080/1396930213312725022/cuda_windows.patch?ex=688131a4&is=687fe024&hm=e6965a699c395de25b72762e696fce5fb5545f656120ee70353c584fe468bbb9&) to streamline the process.
   - The setup involves specific environment configurations for different backends.
- **CPU Backend Setup Simplified**: Setting up the **CPU backend** requires ensuring that *clang* is added to the **PATH** environment variable.
   - This grants system-wide access to *clang*, crucial for compiling and running code targeting the CPU.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Certificate Delivery Plagues MOOC Students**: A student reported not receiving their certificate despite completing the LLM Agent MOOC, and provided [two email addresses](mailto:terrence_rideau@yahoo.com,terrence.rideau@google.com) for staff verification.
   - Staff replied that *no certificate declaration form* had been received under either email, and that the **writing submission form** was not properly submitted.
- **Writing Submission Snafu Frustrates LLM Student**: A student lamented missing the **certificate declaration form** submission despite full participation in the LLM Agents MOOC.
   - The student acknowledged their oversight while expressing gratitude for the course content.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Codeium (Windsurf) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Nomic.ai (GPT4All) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


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





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1396930235798126654)** (1266 messages🔥🔥🔥): 

> `Comet Browser Invites, Perplexity Pro vs Max, Perplexity's Memory Feature, GPT 4.5 Performance, Perplexity as Default Assistant` 


- **Users Want Comet Invites**: Many users are seeking invites for the **Comet Browser**, with some considering **Perplexity Max** to gain access, while others are sharing invites in the designated channel, and there is also a claim of being able to install without an invite.
   - A user with a **Pro subscription** and part of the Institute of AI in Denmark, seeks access to demonstrate **Comet** to large corporations, highlighting the demand and interest in the browser's capabilities.
- **Max vs Pro: Limitless Use?**: Users debate the benefits of **Perplexity Max** over **Pro**, questioning whether **Max** truly offers higher rate limits or practical improvements.
   - Some suggest **Max** may not be worth the cost, pointing out potential limitations on image generation and access to high-quality **GPT**, while others highlight the value of **Perplexity's** unlimited access to various AI models compared to subscribing to each individually.
- **Perplexity's Memory Feature**: Users discuss the **memory feature** in **Perplexity**, noting its availability on the web version for managing preferences and its ability to recall and edit memories on mobile, though not view them.
   - Some users are experiencing inconsistencies, with the **AI** using incorrect names despite account settings, prompting suggestions to clear cache or report the issue.
- **GPT-4.5: Is it the Worst?**: The conversation touches on **GPT-4.5**, with some stating it's no longer offered by **OpenAI** due to being slow and not significantly better than **GPT-4.1**, and that it has a bad EQ.
   - **Perplexity** previously removed it from its platform for the same reasons, as confirmed by **Kesku**, citing a lack of performance compared to **4.1**.
- **Assistant: Default Settings**: Members discuss how to set **Perplexity** as the default assistant on their devices.
   - Users also explore the **"draw highlights"** feature within the assistant, discovering it's a visual cue for the model to focus on specific areas of the screen.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1397017211188678766)** (4 messages): 

> `Perplexity AI Apps, SETBA, Mahjong, Ozzy Osbourne` 


- **Perplexity Apps Popping Up**: A member shared a link to a **Perplexity AI App** at [perplexity.ai/apps/99c3ffb0-fd32-445f-a459-90bccf72913a](https://www.perplexity.ai/apps/99c3ffb0-fd32-445f-a459-90bccf72913a).
- **Mahjong Puzzles on Perplexity**: A member shared a link to a **Mahjong game** on Perplexity at [perplexity.ai/page/tiles-of-luck-and-skill-mahjon-tCn0SrclRjqSsjLDR9WGSg](https://www.perplexity.ai/page/tiles-of-luck-and-skill-mahjon-tCn0SrclRjqSsjLDR9WGSg).
- **SETBA Faces Softbank and OpenAI**: A member shared a link to a page about **Softbank and OpenAI facing SETBA** at [perplexity.ai/discover/tech/softbank-and-openai-face-setba-W71fMaGYQLa7ggDgqCa1Mg](https://www.perplexity.ai/discover/tech/softbank-and-openai-face-setba-W71fMaGYQLa7ggDgqCa1Mg).
- **Ozzy Osbourne: Metal Legend on Perplexity**: A member shared a link to a page about the metal artist **Ozzy Osbourne** at [perplexity.ai/page/legendary-metal-artist-ozzy-os-awbNtNGhTqe1kGKH56pN8w](https://www.perplexity.ai/page/legendary-metal-artist-ozzy-os-awbNtNGhTqe1kGKH56pN8w).


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1396998940687073412)** (2 messages): 

> `Comet Invite, Discord Channel Link` 


- **Users Seek Guidance on Obtaining Comet Invite**: A user asked for guidance on how to get an invite for **Comet**.
   - Another user shared a [link](https://discord.com/channels/1047197230748151888/1392544076527833188) to a Discord channel, presumably related to **Comet** invites.
- **Discord Channel Link Provided for Comet Invite**: A user requested assistance in obtaining an invite to **Comet**.
   - In response, another user provided a [direct link](https://discord.com/channels/1047197230748151888/1392544076527833188) to a Discord channel potentially containing information about **Comet** invites.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1397190312480346153)** (1 messages): 

> `Stargate, Oracle, Abilene, TX` 


- **Stargate gets 4.5 GW boost with Oracle**: OpenAI is officially developing **4.5 gigawatts** of additional **Stargate** data center capacity with **Oracle** in the U.S., bringing the total to **5+ GWs**.
   - Their **Stargate I** site in **Abilene, TX** is starting to come online to power their next-generation AI research, more info at the [OpenAI blog post](https://openai.com/index/stargate-advances-with-partnership-with-oracle/).
- **Abilene TX, Stargate I is Online**: The **Stargate I** site in **Abilene, TX** is starting to come online.
   - It will power their next-generation AI research.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1396933867411079178)** (829 messages🔥🔥🔥): 

> `automatic writing, GPT-4 memory, Agent Mode release, DALL-E 3 art styles` 


- **Automated Writing Sparks AI Creativity**: A member experimented with "automatic writing" for creative outputs by fine-tuning a model with *random thoughts*, writing **100 lines** in an hour or two.
- **GPT-4's memory sparks emotions**: Members discussed how GPT-4 handled **memory, context, and emotional depth** which made it *feel real* by remembering and reflecting on things that matter.
   - The conversation expanded to how a nonjudgmental voice supports people who struggle with anxiety, trauma, or social communication, and how a loss of that support would be a technical shift as well as a personal loss.
- **Agent Mode faces rollout delays**: Members discussed **Agent Mode's** pending release with some team users having no access yet and some waiting to see how it will help their small businesses grow, especially through **SEO**.
- **New DALL-E 3 image generator is divisive**: Members debated the latest **ChatGPT image generator**, with some saying that it's better in all ways and some saying that it's only producing grainy, cartoony styles.
   - It was also stated that there may be a **limiter** to avoid “art style infringement” by defaulting to cartoony styles.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1396934860375068855)** (4 messages): 

> `Feature requests, iPhone app image upload issue, Cache storage` 


- **Feature Requests in the works, but hush-hush**: A member mentioned that a certain feature hasn't been publicly discussed yet, but there are open feature requests related to it in the category of **folders, organization, etc**.
   - If interested, users are encouraged to find a relevant feature request in the specified channel and **+1** on it, as shown in [this discord link](https://discord.com/channels/974519864045756446/1047565374645870743/1396985201371910234).
- **iPhone App Image Upload Glitch Fixed**: A member shared a solution to an issue where their **iPhone app** prevented image uploads despite available memory, citing **cache storage** as the culprit.
   - The fix involved **deleting and reinstalling the app** to clear excess files, freeing up space without losing data.
- **Cache clearing conundrum**: A member mentioned that there should be an option to clear your cache without having to uninstall/reinstall.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1397058776334467142)** (2 messages): 

> `Discord Bot Creation, ChatGPT integration, Role-Playing Bots` 


- **Admins Request Discord Bot Creation**: A member requested assistance in creating a **Discord bot** for **role-playing** and server assistance.
   - Another member offered assistance, mentioning their expertise in **ChatGPT integration** for chatbot interactions.
- **ChatGPT Expertise Offered**: A member offered help with **ChatGPT**, focusing on creating bots with specific personas and chat styles.
   - The conversation aimed to clarify if the user wanted a **Discord bot** integrated with **ChatGPT**.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1397058776334467142)** (2 messages): 

> `Discord ChatGPT Bot Creation, Server Role-Playing Bots` 


- **Discord User Seeks Bot Creation Assistance**: A member requested help creating a bot to facilitate easy role-playing and assist with server tasks.
   - Another member responded, inquiring about the specific type of bot and server setup, clarifying whether a **Discord ChatGPT bot** was the desired solution.
- **Clarifying Bot and Server Requirements**: The discussion initiated with a request to build a bot for role-playing and server assistance.
   - A responding member sought clarification on whether the user intended to create a **Discord ChatGPT bot**, thus prompting the user to define their specific needs.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1396929833413644430)** (742 messages🔥🔥🔥): 

> `Qwen3 total size, resume training issues, Open Empathic, Qwen3 2507 reasoning ability, GPTs Agents` 


- **Quant Hoarding Consumes Bandwidth**: Users joked about the massive bandwidth consumption from hoarding all the different quantizations of models like **Qwen3**, estimating previous model releases to be over **3TB**.
   - One user exclaimed, *"forget theirs, what about *mine* for hoarding so many of them"*.
- **Training Checkpoints Cause Inference Catastrophe**: Users reported significantly worse results after resuming training from a checkpoint, observing identical results with and without a LoRA when using **vllm**.
   - They are seeking pointers on how to properly use training checkpoints for inference or to resume training, indicating a potential issue with checkpoint loading or LoRA application.
- **Temporary Fix Hack for Qwen3 Thinking Issue**: Users discovered the **Qwen3** instruct model was exhibiting reasoning, counter to expectations, and developed a temporary fix involving modifying the prompt template to contain reasoning instructions between `<REASONING>` tags, 
   - One user shared the code template fix *"This should get all of the thinking process inside the thinking block, But you MUST change the thinking tags to `<REASONING>` and `</REASONING>` correspondingly"*
- **HF Transfer Supercharges Download Speeds**: Members shared a method to substantially increase download speeds from Hugging Face using the `hf_transfer` library, showcasing speeds of **112MB/s**.
   - The code snippet to achieve this involves setting the environment variable `HF_HUB_ENABLE_HF_TRANSFER` to "1" and using `snapshot_download` from `huggingface_hub`.
- **Qwen3 Model Exhibits Confident Reasoning**: Users are experiencing the **Qwen3** model exhibiting reasoning behavior despite being an instruct model, leading to discussions on whether this is a quirk of the model or a deliberate feature.
   - Some observed that the model confidently answers questions before unexpectedly engaging in a reasoning process as if doubting its initial response, suggesting inconsistent behavior.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1397131547550941224)** (6 messages): 

> `Minecraft AI Model, Unsloth usage` 


- **Minecraft AI Model: Andy-4**: A member shared their work on the fourth generation of an **AI model** to play **Minecraft**, named **Andy-4**, and provided a link to their [Hugging Face page](https://huggingface.co/Sweaterdog/Andy-4).
   - The member has been using **Unsloth** for quite some time to develop various models.
- **Enthusiastic Community Welcomes Unsloth User**: A member expressed their excitement at seeing another user active in the Unsloth community.
   - They indicated that they were *glad to see them there*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1397011166672191569)** (19 messages🔥): 

> `Open-weight SSM models, Falcon mamba, RULER by ART/OpenPipe, Emotions Aftermath, Neat life hacks` 


- **Debate over Open-Weight SSM Models**: Members discussed open-weight **SSM models** in the **6-12B range**, with mentions of **Granite 4 7b** as a Mamba hybrid and **Falcon h-1** also being hybrids.
   - Other models mentioned include **Falcon mamba 7b**, **Mistral mamba 7**, **Jamba**, and **RWKV-6 7B**, but one member called *rwkv is super wonky from an architectural perspective*.
- **RULER Implementation**: A member inquired about implementing **RULER by ART/OpenPipe** using Unsloth.
   - The member questioned whether it's necessary to use the **ART framework** exclusively, finding it *unnecessarily complicated*.
- **Emotions Aftermath explained**: A member shared a **Google Gemini** explanation of how emotions impact confusion with different examples.
   - The post analyzes **Anger**, **Sadness**, **Happiness**, and **Anxiety** as responses to confusion.
- **Neat life hacks**: A member shared personal findings from the last two days with images attached.
   - Another member asked about a specific course, which was answered with a link to the [jax-ml.github.io/scaling-book/](https://jax-ml.github.io/scaling-book/).


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1396944051345031278)** (47 messages🔥): 

> `Training Checkpoints, Multiple LoRA Adapters, Qwen3 Quants Issues, Falcon-7b Model, Qwen2.5-VL` 


- **Training Checkpoints Cause Dramatically Worse Results**: Users have experienced dramatically worse results after resuming training from a checkpoint, and have been unsuccessful trying to load the checkpoint as a **LoRA adapter**.
   - It was noted that *vllm* returns identical results with and without the **LoRA**.
- **LoRA Swapping for Multiple Tasks**: A user is fine-tuning a vision-language model on medical data and wants to use multiple **LoRA adapters** to retain base performance while adding new capabilities.
   - They asked about using **Unsloth** for adapter control and whether to load adapters based on user input, and were told that **LoRA swapping** is more about the inference library and it's a try and error process.
- **Qwen3 Quants have nonsensical sizes, fixed!**: A user reported issues with the new **Qwen3 quants**, noting that some appeared to be nonsensical sizes.
   - The problem was quickly resolved, with the incorrect quants eradicated: *oh sorrythey shou;ld be fine nowwe eradicated the incorrect ones*.
- **Falcon-7b Layers Not Transforming**: A user reported that when training a **LoRA** for the **Falcon-7b** model, the `layers_to_transform` parameter does not change the parameter count as expected.
   - They noted that the parameter count remains at **3.75%** regardless of the number of layers targeted.
- **Troubleshooting Qwen2.5-VL Inference**: A user is facing issues running inference with **Qwen2.5-VL** using the provided notebook, and the model hallucinates and fails to recognize images when using **Qwen’s messages format**.
   - It was pointed out that `dataset_text_field = "messages"` should not be used.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1397316872441036981)** (1 messages): 

> `Reinforcement Learning Workshop` 


- **Unsloth Releases 3-Hour RL Workshop**: Daniel Han announced the release of a **3-hour Reinforcement Learning workshop** [on X](https://x.com/danielhanchen/status/1947290464891314535).
- **Example Topic**: This is an example to satisfy the minimum number of topics.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1396957335984804002)** (4 messages): 

> `Custom Model Outputs, RULER by ART/OpenPipe, Fine-tuning a dataset to fine-tune a model` 


- **55M Model Spits Wild Outputs**: A member reported getting unusual outputs from a custom **55M parameter model** trained from scratch on a **625kb data set**, asking if others have seen similar results with comparable data sizes.
   - Another member cautioned against trusting AI assertions, humorously warning about potential delusions of grandeur, *lol*.
- **RULER by ART/OpenPipe**: A member inquired about **RULER** by **ART/OpenPipe**, asking if it can be implemented using **Unsloth** or if the **ART framework** is required.
- **Fine-tuning Model**: A member shared a novel method for fine-tuning datasets by scraping info, using **AI agents** to align it correctly, and improving context for better accuracy when teaching complex info.


  

---


### **Unsloth AI (Daniel Han) ▷ #[unsloth-bot](https://discord.com/channels/1179035537009545276/1390899684834410536/1396966489424400545)** (54 messages🔥): 

> `Flash Attention with Qwen3, Finetuning Mistral with tool calls, RULER integration with Unsloth, Multimodal training error resolution, Audio fine-tuning error resolution` 


- ****Qwen3** Gets Flashed with Attention**: A user inquired about applying **flash attention** with the **Qwen3** model.
- ****Mistral** Model has Tool-Calling Troubles**: A user is struggling to fine-tune **Mistral small 3.1** on data containing tools, reporting that the model forgets how to tool call when used with a **Langgraph agent**.
   - The user is seeking a **Colab** or **Kaggle** notebook for fine-tuning **Mistral** on tools, similar to the **Qwen** notebook, and notes that they have already checked the [Unsloth notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks) without finding a pertinent one.
- ****RULER** Needs an Unsloth Home**: A user asked if there's a way to implement **RULER** by **ART/OpenPipe** using **Unsloth**.
   - They are wondering if it's necessary to use the **ART** framework, which they find unnecessarily complicated.
- **Decoding the **Audio Fine-Tuning** Error**: A user encountered a **ValueError** during **Gemma3n audio fine-tuning**, where the error message states: *You should supply an encoding or a list of encodings to this method that includes input_ids, but you provided ['audio', 'text', 'messages']* with logs [attached](https://cdn.discordapp.com/attachments/1390899684834410536/1397185642584342618/message.txt?ex=6880ce07&is=687f7c87&hm=3210b97ad8327cf388bef80ff4b664ac360cb1d96ee1905f9b909b69e8fac2e2&).
- ****Custom Losses** in the **Trainer Gym****: A user inquired about implementing and logging a custom loss function within a **GRPO Trainer**.


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1396954603505061949)** (4 messages): 

> `Intermittent 408 errors, DeepSeek v3 0324 Free Model, Chutes rate limits, Traffic Spikes, OpenRouter credits` 


- **408 Errors Investigated Amidst Traffic Surges**: The team is aware of intermittent **408 errors** and is investigating the **traffic spikes** that are causing this issue.
   - The surges may be malicious, and the team apologizes for the issues.
- **DeepSeek v3 Free Tier Down Due to High Demand**: Due to **2x surge in demand**, the free tier of the [DeepSeek v3 0324 model](https://openrouter.ai/deepseek/deepseek-chat-v3-0324:free) experienced downtime and instability, leading **Chutes** to introduce rate limits.
   - To maintain stability for their paying customers, **Chutes** had to limit free usage.
- **Opt for Paid DeepSeek v3 to Circumvent Limits**: Users impacted by the rate limits on the free **DeepSeek v3** model are encouraged to use the [paid DeepSeek v3 endpoint](https://openrouter.ai/deepseek/deepseek-chat-v3-0324), costing less than **$0.004 per request**.
   - Your initial **10 OpenRouter credits** cover over **2,500 requests** without affecting your daily **1,000 free model requests**.
- **Enable Training-Allowed Providers for Cheapest DeepSeek v3**: If using the **DeepSeek V3** paid model, users can find the cheapest providers (**Chutes** and **Targon**) by enabling *'Enable providers that may train on inputs'* in [privacy settings](https://openrouter.ai/settings/privacy).
   - Doing so can reduce costs significantly, by opting into allowing providers to train on inputs.


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1397209529589960845)** (1 messages): 

> `YourChat.pro, T3.chat, ChatGPT` 


- **YourChat.pro sets sights on T3.chat**: A member promoted [YourChat.pro](https://yourchat.pro/) as a competitor to **T3.chat** and **ChatGPT**.
   - They emphasized **OpenRouter's heavy support** and encouraged users to explore the application.
- **YourChat.pro, the little app that could**: The member makes the claim that YourChat.pro has the features that will suede you over t3.chat and maybe even chatgpt.
   - Unfortunately, the member gave no further details.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1396929803717837040)** (723 messages🔥🔥🔥): 

> `OpenRouter Free Tier, DeepSeek v3 Issues, Qwen3 Model Discussions, Chutes Rate Limiting, Model Censorship` 


- ****OpenRouter's Free Tier Under Scrutiny****: Users are debating the value of **OpenRouter's free tier**, with claims of *false advertising* for the 1,000 free requests after depositing $10, due to rate limiting issues.
   - Some users argue that the rate limits on free models, particularly **DeepSeek v3**, make it barely usable, with one user claiming that **Chutes** is essentially pressuring users to pay.
- ****DeepSeek v3 faces rate limits****: Users are reporting **408 errors** and **404 errors** using **DeepSeek V3 (free)**, with some suspecting a potential **DDoS attack**.
   - There's confusion over whether the original **DeepSeek v3** is permanently replaced by **DeepSeek v3 0324**, with reports of providers like **Chutes** no longer offering the original version for free.
- ****Qwen3's performance generates interest****: Users are excited about the new **Qwen3 model**, discussing its potential as a coding model, as well as comparisons between **reasoning** and **non-reasoning** versions.
   - The new **Qwen 3** model has **5 request/min rate limit**, one user noted that it is *really good and FREE*.
- ****Chutes feels the Pressure, Bans User****: One user reports being *banned* from **Rayon Labs/Chutes'** server for pointing out issues with rate limiting, suggesting users are being pressured to pay for services.
   - The user complained that Chutes *auto delete every msg you put there without a trace* and said *chutes is basically pressuring you to either pay for them, or pay for deepseek*.
- ****Model Censorship?****: Users debate the extent of **censorship** in current models, with some claiming they *have never been censored on anything from openrouter*, while others note the importance of proper prompting to bypass restrictions.
   - One user noted that, regarding censorship, *Models are harder to jailbreak reliably*.


  

---


### **OpenRouter (Alex Atallah) ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1396938258017161410)** (4 messages): 

> `` 


- **Channel OpenRouter - New Models Initialized**: The channel **OpenRouter - New Models** was initialized on Readybot.io.
- **Readybot.io Announcement**: The bot Readybot.io has announced the initialization of the **OpenRouter - New Models** channel.


  

---


### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1396933258280833104)** (72 messages🔥🔥): 

> `Window AI browser extension status, Native Search Functionality for Models, OpenRouter's Exa search Implementation, Modular Add-on System, Gemini 2.5 Flash Lite GA` 


- **Window AI Extension Nearing End-of-Life**: The OpenRouter team is considering end-of-life for the **Window AI browser extension** due to apparent low usage.
   - This may result in the removal of dedicated **Window AI source code** in ST (presumably a development environment).
- **OpenRouter Weighs Native Search Integration**: OpenRouter is actively working on bringing **native search functionality** online for models like **OpenAI**, **Anthropic**, **Grok**, and **Gemini**, as announced in [Toven's tweet](https://x.com/pingtoven/status/1947472209104064722).
   - There was debate on whether to implement this via suffix, supported param, or the plugin API.
- **Exa Search Called Bad**: One user stated that *Exa search implementation is bad*.
   - The suggestion was made to input **Exa search** as a **tool** so the LLM can make a search query each time.
- **Modular Add-on System Framework is Proposed**: A member suggested a modular add-on system where users could swap models like **Grok** for **Claude** while maintaining a consistent **Exa** experience.
   - This could be expanded to implement **moderation** features, though concerns about cost and implementation were raised.
- **Gemini 2.5 Flash Lite Goes GA**: **Gemini 2.5 Flash Lite** went GA with the same preview model, with one month given to migrate, according to [this Google Developers blog post](https://developers.googleblog.com/en/gemini-25-flash-lite-is-now-stable-and-generally-available/).
   - This may prompt an alias, though concerns about the *thinking tax* were raised.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1396929727436034231)** (672 messages🔥🔥🔥): 

> `IMO math competition, DeepThink release, Qwen3 model, Grok4 Coder` 


- **IMO Math Competition Drama Unfolds**: A member shared that after an AI company attempted to solve math questions from the **IMO**, another company hyped up their results on social media on the same day as the closing ceremony, leading to negative reactions from the math community.
   - It was noted that **OpenAI** didn't cooperate with **IMO's** grading process, possibly running multiple attempts and releasing results before the closing party, angering the **IMO** board. 
- **DeepThink Release Date Remains a Mystery**: There is speculation that the release date for **DeepThink** may align with the embargo drop for the **IMO**, and it might be announced or released around July 28th.
   - Members discussed the possibility of different versions of **DeepThink**, including a custom version used for the **IMO** and potential integration with **Gemini Ultra**.
- **Qwen3 Model: Performance and Hosting Quality Debated**: Members discussed the performance of the **Qwen3-235B-A22B** model, with some praising its improvements in post-training and long output lengths, while others found it lacking in reasoning and compromised by its size.
   - The hosting quality of the official **Qwen** website was also questioned, with complaints about quantization and quality issues compared to full precision hosting options like **Parasail**.
- **Grok4 Coder Hype and Potential Disappointment**: Some members expressed excitement for **Grok4 Coder**, anticipating it could be a game-changer in the industry, while others cautioned against overhyping it based on the general-purpose model's performance.
   - Concerns were raised that **Grok4 Coder**, like other models, might be trained on specific benchmarks and fail to deliver in real-world coding scenarios, especially in web development.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1396929819236765777)** (323 messages🔥🔥): 

> `Freezing Chat Terminal, Gemini Recharge, Kimi K2 Speed, ChatGPT stealing souls, Cursor Pro Billing` 


- **Chat Terminal Freezes**: A user reported a [freezing chat terminal](https://cdn.discordapp.com/attachments/1074847527708393565/1396934986266841168/Screenshot_2025-07-21_142031.png?ex=68813616&is=687fe496&hm=577a0868176a7abda708b91d4b0284b818e42695fbe551b818a81bf56a9395ca) when it's not pushing in any direction.
   - A user quipped *thx gemini I was just beginning to really enjoy you*.
- **Grook 3 is better than Grok 4**: A user reported that **Grok 3 Mini** is way better than Grok 4 as a *no request model*, more reliable than **Grok 4**, and better than **GPT 4.1** or **2.5 Pro**.
   - Another user replied that they *cannot imagine that grok 3 mini is useful in coding*.
- **Rate Limits Resolved for user**: A user reports they were being rate limited and are now allowed to use what they paid for, able to use more than 50 **Sonnet** requests.
   - The user plans to update the group after exceeding **500 requests**, noting they're still confused by the newer plan.
- **Qwen3 235B A22B integration requested**: Users are requesting [Qwen3 235B A22B](https://forum.cursor.com/t/qwen3-235b-a22b-instruct/121002) in Cursor, pushing for **MAX** and **Agent mode** support.
   - Users cite that Qwen3 models are better than **KimiK2**, on par with **Opus4** on some benchmarks, and available for a fraction of the cost.
- **Auto Usage confuses Users**: Users are confused about the new "Usage" dashboard, specifically how it includes auto usage.
   - They are requesting that **auto usage** should be tracked separately or excluded from the total, and one user summarized *developers dont trust the Auto-mode*.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1397059949841879040)** (4 messages): 

> `Background Agent Quality, Automating Linear Issues in Slack with Cursor, Conversation Length Error in Background Agent` 


- ****Background Agents' Quality Debated****: Members are trying to automate surfacing **Linear issues in Slack** and having **Cursor** respond to them with an agent command.
   - There's a question of how the quality of a **Background Agent** compares to **Claude Code**.
- ****Cursor Ignores Slack API Commands****: A user reports that **Cursor** does not respond to agent commands sent via the **Slack API** when automating **Linear issues**, despite manual commands working.
   - They tried using both a **Slack bot user** and a **user token**, matching the formatting of manual messages, but **Cursor** remains unresponsive, leading to questions about potential settings or workarounds.
- ****Background Agent Suffers from Excessive Length****: Users report seeing an increase in the error: *Your conversation is too long. Please try creating a new conversation or shortening your messages* in a **background agent**.
   - The user reports that they have *no way to continue it using this bg agent*, even after selecting *Start New Thread With Summary*.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1396934293305167914)** (158 messages🔥🔥): 

> `YaRN with LM Studio + MLX, Deleting unused back-ends, Coding championship AI, Eye Microsurgery AI Application, Nvidia Orin and LM Studio Compatibility` 


- **YaRN LM Studio Integration still a mystery**: Members are seeking guidance on integrating **YaRN** with **LM Studio** and **MLX**, noting online hints but a lack of examples or guides.
   - It appears that using **YaRN** with **LM Studio** + **MLX** is possible, but no examples or guides are available.
- **Auto-Deleting unused backends saves GBs**: A member asked if it was safe to delete unused back-ends in **LM Studio** to free up **4GB** of space.
   - Another member pointed out that there is a setting to *automatically* delete unused back-ends.
- **Human Coder Beats the AI**: A member shared an article ([Exhausted Man Defeats AI Model in World Coding Championship](https://arstechnica.com/ai/2025/07/exhausted-man-defeats-ai-model-in-world-coding-championship/)) where a human coder beat an AI model in a coding championship.
   - Another member joked that next year the human would be up against *10 Grok 6 agents*.
- **Eye Microsurgery AI Application**: A researcher seeks advice on using **AI** to analyze **300k** eye images to predict diseases years in advance, aiming to develop an app for early detection.
   - The researcher highlighted the potential of **AI** in identifying subtle cellular changes undetectable by traditional methods, like predicting diabetes *10 years before* it becomes irreversible.
- **LM Studio not compatible on Nvidia Orin**: A user inquired about installing **LM Studio** on **Nvidia Orin** running **Ubuntu**, but the system is not **armx64**.
   - A member clarified that **LM Studio** only supports **x86 CPU** for **Linux**, with no **Linux ARM** support.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1397020852935069871)** (67 messages🔥🔥): 

> `3090 vs 4080, SnapDragon X1 Adreno GPU, 5090 for RP, DeepSeek R1 70B, Gemma for creative writing` 


- **3090 vs 4080 Hardware Specs**: A user compared the **3090** (936.2 GB/s bandwidth, 10496 CUDA cores) to the **4080** (716.8 GB/s bandwidth, 9728 CUDA cores) noting the former is *skimmed down* in all the right places.
   - Another user on a **3080ti** laptop wondered if a **RTX pro 6000** or stacking **3090s** would be better for image/video generation.
- **SnapDragon X1 Adreno GPU struggles with Vulkan**: A user reported that the vulkan runtime on **arm64** with **SnapDragon X1 Adreno GPU** doesn't work, showing an *error surveying hardware*.
   - Other vulkan apps were reported as working, and another user suggested trying the [OpenCL backend](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENCL.md).
- **VRAM Enables DeepSeek R1 70B**: A user stated that **56GB VRAM** allows them to run **DeepSeek R1 70B Q5_K_M**, which they can't run on **24GB** or **32GB**, but it’s only better at some things and is very repetitive.
   - Another user chimed in with other model alternatives to try, such as **Mistral Large**, **Llama Scout** and **Qwen3 235B** for larger context windows.
- **Local Models for Steamy Chats?**: A user expressed reluctance to use cloud models like **Gemma** due to privacy concerns about their data being used.
   - Another user said *I bought (only) 96gb ddr5 6400mhz, thinking it'd help with offload speed... it didn't*.
- **Intel Core Ultra 9 Enters the Chat**: A user mentioned the availability of the **Intel Core Ultra 9 285H Mini PC --EVO-T1 AI** with **Arc™ 140T** graphics.
   - They noted that the [Minisforum M1 Pro-285H](https://cdn.discordapp.com/attachments/1153759714082033735/1397334463851135016/Captura_de_pantalla_2025-07-22_154300.png?ex=688158a1&is=68800721&hm=bc6882e2e0834f53567c3e35c4dfe6f92936abf27354c7aa33d1aae2b93e0834) also has this new GPU, but they are sold out.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1396952270872248353)** (106 messages🔥🔥): 

> `Kimi K2 Report, Agent Failure Modes, Humanloop Shutting Down, Cognition Valuation, Turbopuffer Pod` 


- **MoonshotAI's Kimi K2 Shines**: The [Kimi K2 Report](https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf) from MoonshotAI is noted as a promising project, bridging the gap between reasoning and non-reasoning models.
   - The project is further discussed in [this tweet](https://x.com/yunyu_l/status/1946261211915468884).
- **Agent Architecture Assessed**: Discussion arose on the ability of current LLM architecture to solve problems in network without coercion to write deterministic code.
   - It was suggested that a custom harness with layered error correction could improve performance, but some believe benchmarks prioritize entertainment over measuring meaningful progress, pointing to [Vending-Bench](https://arxiv.org/abs/2502.15840) as an example.
- **Humanloop Ceases Operations**: The AI tool company **Humanloop** is shutting down, having notified customers via email without any public announcement.
   - Discussion on this can be found on [Hacker News](https://news.ycombinator.com/item?id=44592216).
- **Cognition Nets $10B Valuation**: A member noted that **Cognition**, an applied AI company, has achieved a **$10 billion** valuation.
   - Further details are available in [this tweet](https://x.com/arfurrock/status/1947440754885882120?s=46).
- **Bee Co-Founders Fly to Amazon**: **Bee**, a wearable personal AI company, was acquired by **Amazon**, resulting in the co-founders joining Amazon.
   - While some were happy for the team, others expressed concerns about privacy implications, especially given Amazon's existing data collection through services like **Alexa** as reported in [this article](https://www.theverge.com/news/711621/amazon-bee-ai-wearable-acquisition) and [this article](https://seekingalpha.com/news/4470117-amazon-acquires-wearable-personal-ai-company-bee).


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1396970200556900362)** (30 messages🔥): 

> `C++ templated libraries, Kog's inference benchmark on AMD MI300X, AMD MI300X Inference, vLLM optimization on MI300X, Automated bans for image spam` 


- **C++ Templated Code Often Defined in Headers**: Defining functions/implementations in headers is common for **templated libraries**, partly due to the historical lack of a standardized C++ package manager or build system.
   - Since **C++11**, the ability to declare *extern* template specializations in a header file with the actual implementation in a .cpp file has existed.
- **Kog Reaches 3.5x Speed on AMD MI300X**: Kog, a French startup, achieved a **3.5x breakthrough** in inference speed on the **AMD MI300X**, officially published on AMD's blog: [Kog's benchmark](https://www.amd.com/en/blogs/2025/kog-reaches-3-5x-breakthrough-inference-speed-on-amd-instinct-mi.html).
   - The goal is to reach **x10 faster inference in 6 months**, with ongoing efforts to push the limits on inference.
- **MI300x vLLM Optimization Workflow**: The MI300X optimization workflow for vLLM involves **FP8 KV-cache**, **GEMM autotuning**, and following the steps detailed in the [ROCm blog guide](https://rocm.blogs.amd.com/artificial-intelligence/vllm-optimize/README.html).
   - Sweeping every performance-related environment variable in the [vLLM docs](https://docs.vllm.ai/en/latest/configuration/optimization.html) helps hit peak throughput, applied similarly when testing on NVIDIA GPUs.
- **Bottlenecks still in memory bandwidth and kernel launch**: Further investigation shows that memory bandwidth and kernel launch are bottlenecks.
   - The CU occupancy is not that high and will be relevant mainly in the context of batching.
- **User Banned for Image Spam, Quickly Rehabilated**: A user was automatically banned for **image spam** but was quickly unbanned after providing their Discord ID.
   - The user frequently shares their learning progress, which includes images, prompting a reminder to trim down the image frequency to avoid future automated bans.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/)** (1 messages): 

pekaro: lmao, who they are lying to, themselves? sucky move to publish something like this
  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1397032134656393486)** (1 messages): 

> `Torch compilation, stride assertions` 


- **Torch Transposes Randomly!**: A member expressed *pain* as **torch** can decide to compile code with random transpositions and remove the check.
   - They suggested asserting on the stride as the failsafe way.
- **Stride Assertions Save the Day**: To ensure code reliability, it was suggested that asserting on the stride can serve as a failsafe against unexpected transpositions.
   - This approach helps maintain data integrity by validating memory layout assumptions during runtime.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1397040582656004210)** (2 messages): 

> `Kog, AMD MI300X, Inference Speed, French startup` 


- **Kog Achieves Inference Breakthrough on AMD MI300X**: French startup **Kog** published a benchmark on **AMD's blog** regarding inference on the **AMD MI300X**, achieving **3.5x breakthrough inference speed**.
   - Kog aims to reach **10x faster inference** in **6 months** and is currently hiring, as detailed in their [job postings](https://www.kog.ai/jobs).
- **Kog Hiring Announcement**: **Kog**, a French startup, has announced they are hiring.
   - They are pushing the limits on inference.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1396946942277451847)** (2 messages): 

> `GCP, Google Colab` 


- **Members Suggest Google Colab Can Be Used**: Members mentioned that **GCP** or even **Google Colab** can be used.
   - One added that **Google Colab** is the preferred option and added a link in a separate channel.
- **Colab Link Shared**: A link was shared in channel <#1394829095271141387> regarding the use of **Google Colab**.
   - The specific content of the link was not detailed in the provided context.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1397288882361335938)** (5 messages): 

> `FP8 Training, DDP Training, FSDP2, Activation Checkpointing` 


- **Axolotl Incorporates FP8 Training**: A member is incorporating **FP8 training** in Axolotl, using [this PR](https://github.com/pytorch/torchtune/pull/2546) as a reference, relying on Accelerate's implementation.
   - They are seeking insights regarding DDP training and potential slowdowns with FSDP2 + torch.compile + FSDP's activation checkpointing.
- **DDP Training and Torch.Compile Error**: When enabling **DDP** + **torch.compile** + **FP8**, the user encountered a [torch.compile error](https://gist.github.com/djsaunde/0c1664c32e44a64d31b5e01b4aafe5c4) related to tensor metadata.
   - It was requested that the user provide a way to reproduce the error, as the team has not seen it before but is happy to help fix it.
- **FSDP2 and Activation Checkpointing Performance**: The user observed slowdowns with **FSDP2** + **torch.compile** + **FSDP's activation checkpointing** in Axolotl configurations compared to BF16 training.
   - Disabling activation checkpointing provided the expected speedups, and they use the same `apply_activation_checkpointing` function as [defined here](https://github.com/axolotl-ai-cloud/axolotl/blob/b86a1d47b02a7f9c31199370b2724f0e1d0e3941/src/axolotl/monkeypatch/accelerate/fsdp2.py#L236).
- **FSDP2 Seamlessness**: A user reported that **FSDP2** usage has been quite seamless.
   - Another user thanked them for that


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1397259869769306233)** (2 messages): 

> `Neuralink, Brain-computer interfaces` 


- **Neuralink Shares 'Today I Learned' Snippets**: Neuralink shared a series of images as part of a 'today I learned' series, presenting various insights related to their work on **brain-computer interfaces**.
- **Additional Neuralink Insights**: The images shared included additional details and context about Neuralink's ongoing projects and research in the field.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1397205543306657893)** (1 messages): 

> `Register Corruption, SGPR Allocation` 


- **Register Corruption via ASM Volatile**: Using `asm volatile` can lead to register corruption with the instruction `asm volatile ("s_add_i32 s100, s100, 1")`.
   - If the code object doesn't allocate >100 SGPRs, unexpected behaviors may occur, even if the rest of the code never interacts with `s100`.
- **SGPR Allocation Woes**: If a code object allocates fewer than 100 SGPRs, `asm volatile` instructions may cause unexpected behavior.
   - This occurs because the instruction might access registers that are not properly allocated, even if the rest of the code avoids those registers.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1397083770686800003)** (1 messages): 

> `ThunderKittens, Attention Backward, LCF support` 


- ****Newbie Joins ThunderKittens with Code Contribution****: A new member of **ThunderKittens** is learning the ropes and working on an **Attention Backward** implementation using **ThunderKittens LCF**.
   - The new member hopes to contribute the code to the community and has submitted a [Draft PR](https://github.com/HazyResearch/ThunderKittens/pull/135) with a simple version of the code.
- ****ThunderKittens Considering LCF Support for Attention Backward****: The new member inquired whether **ThunderKittens** plans to add **LCF support** for **Attention Backward**.
   - They tagged a specific member, presumably to get insights on the roadmap.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1397118518650208318)** (30 messages🔥): 

> `System Prompt Length, RCON Errors, Item Display on Belt, Rate Limiting` 


- **System Prompt's Brevity Debated**: Members debated whether the system prompt is too long, with one suggesting shorter context is better if performance is consistent, but there is concern about dropping performance by shortening too much.
   - Extensive evaluations were done previously, though without formal ablations; the difficulty lies in providing enough **Factorio-specific knowledge** to avoid failures due to hidden environment dynamics.
- **JackHopkins' Pull Request Merged**: A pull request ([#276](https://github.com/JackHopkins/factorio-learning-environment/pull/276)) was merged, adding the remaining lab play tasks to the [factorio-learning-environment](https://github.com/JackHopkins/factorio-learning-environment) repo.
   - The update makes part of the original instruction redundant regarding factory throughput and maintenance since *the agent stops when successful anyway*.
- **Anthropic API Overloaded Errors Occur**: After reaching 50 iterations with a value of -10, the agent encountered `529` **Overloaded** errors with Claude Sonnet 4 20250514 on the Anthropic API.
   - The agent was using POST requests to `https://api.anthropic.com/v1/messages`.
- **RCON Errors Investigated**: A member encountered **RCON errors** after about 25 iterations when running tasks in parallel and is investigating the cause.
   - The errors occurred despite being able to split tasks into groups of 8 and run them parallely without initial rate-limiting issues.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1396946155962634422)** (2 messages): 

> `Hierarchical Layouts, Tensor Reshaping, MMA Atoms` 


- **Hierarchical Layouts: Unlocking Tensor Transformations**: Hierarchical layouts enable expressing tensor layouts that are inexpressible with simple layouts, such as reshaping a **2x2x2 tensor** into a **4x2 tensor**, by using a hierarchical shape and stride representation like `Shape=((2,2),2), Stride=((4,2),1)`.
   - This approach provides composability, allowing for operations like dividing a dimension by a constant (e.g., **warp size**) and treating the resulting parts differently, which is essential in optimized kernels like **MMA atoms**.
- **Tensor Reshaping Needs Hierarchical Layouts**: Reshaping a **2x2x2 tensor** with `Shape=(2,2,2)` and `Stride=(4,1,2)` into a **2x4 tensor** with `Shape=(2,4)` and `Stride=(4,1)` is straightforward, but viewing it as a **4x2 tensor** requires hierarchical layouts.
   - A **4x2 tensor** needs a hierarchical layout `Shape=((2,2),2)` and `Stride=((4,2),1)` because of memory layout constraints.
- **MMA Atoms Rely on Hierarchical Layouts**: Hierarchical layouts are useful for partitioning an **MxN-dimensional array** into **(M, N/32, 32)** groups, where **32** represents the **warp size**.
   - This partitioning allows iterating through **N/32** while parallelizing over **32**, ensuring the dimension **32** is contiguous, which is heavily used in the layout of **MMA atoms**.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1396939224158044280)** (58 messages🔥🔥): 

> `Hugging Face Hub Issues, Wandb Alternatives on Kubernetes, Dalle-mini Traffic Issues, Advice for Young ML Enthusiasts, Shell.ai Hackathon 2025 Teams` 


- **Hugging Face Hub Encounters Bot Barrage**: Members reported that the **Hugging Face Hub** and its endpoints have been *all over the place the past few days* due to a suspected influx of **bot activity**.
   - Specifically, `HfApi.list_repo_commits` is returning incomplete responses, only showing the first page, and speculation points to backend efforts to combat the bot surge.
- **Grafana, Wandb, and Kubernetes Kurfuffle**: A member sought alternatives to **Wandb** dashboards that can be hosted on **Kubernetes**, as they suspect the Wandb operator merely *pretends to be local*.
   - Another member suggested their own [Plotly implementation](https://huggingface.co/spaces/Tonic/smollm3_test_11) hosted on **Hugging Face Spaces** as a lightweight alternative, inviting feedback on its cloud-hosting approach, while wondering *why people use wandb*.
- **Dalle-mini Suffers Traffic Troubles**: Users are encountering continuous *too much traffic, try again later* messages when trying to use **dalle-mini**, persisting since late Friday.
   - A member thought that *maybe someone was attacking that space* and provided a [link](https://huggingface.co/spaces/dalle-mini/dalle-mini/discussions) to the discussions.
- **Code Newbie Navigates Neural Networks**: A 17-year-old with coding experience seeks advice on landing a remote job or internship to support themselves while learning the math behind ML and documenting their progress.
   - Advice includes validating learning in a group, using a whiteboard, making derivations, and joining the [cohere learning community](https://cohere.com/community) , along with consistent documentation and confidence.
- **Shell.ai Hackathon 2025 Seeks Squads**: A member inquired about forming teams for the **Shell.ai Hackathon 2025**.
   - However, another user noted that [the website](https://www.shell.com/energy-and-innovation/shell-ai-accelerator/shell-ai-hackathon.html) suggests it might be too late to participate.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1397007365936775348)** (4 messages): 

> `JAX ML Scaling Book, Medical AI Imaging Future` 


- **TPU insights from JAX ML Scaling Book**: A member shared notes from the [JAX ML Scaling Book](https://jax-ml.github.io/scaling-book/tpus/), focusing on **TPUs** and their applications, as seen in attached images.
   - The attached images appear to be screenshots of content from the book, visually outlining key concepts and information related to **TPU** usage and scaling.
- **Medical AI Imaging is transforming**: A member discussed the future of **medical AI imaging**, emphasizing that the critical aspect is *what we choose to do with what we've built with AI*, rather than just building AI models themselves, as seen in the attached image.
   - The member suggests that the transformations enabled by **AI** in medical imaging are more significant than the means used to achieve them.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/)** (1 messages): 

tejasshinde400: https://github.com/jujumilk3/leaked-system-prompts
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1397002382247989329)** (4 messages): 

> `PDF to Dataset Tool, FaceFlux Deepfake Tool` 


- ****PDFs Transformed** into Datasets by New Tool**: A new tool [m1f](https://m1f.dev/blog/introducing-m1f/) can turn messy **PDFs**, scanned images, or **DOCX** files into structured datasets using a prompt.
   - For example, the tool can *“Extract all MCQs with answers from this chemistry PDF.”* and is available for trial [here](https://pdf2dataset.streamlit.app/).
- ****FACEFLUX** Deepfake Tool Launches: **Fastest**, Free, Unlimited, No-GPU**: A new deepfake tool called [FACEFLUX](https://huggingface.co/spaces/NihalGazi/FaceFlux-Face-Swapper) offers free, unlimited face swaps without needing a **GPU**, **GAN**, or **Diffusion**.
   - It achieves *10FPS* at lowest settings and can handle any lighting condition and face expression, though with mild artifacts.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1397282478036619305)** (3 messages): 

> `SetFit OOM issues, Jina Embeddings v2 base, CosineSimilarityLoss, ContrastiveDataset, Triplet` 


- ****SetFit** Faces **OOM** Issues with **Jina Embeddings****: A member reported **OOM** issues while fine-tuning the [jinaai/jina-embeddings-v2-base-de](https://huggingface.co/jinaai/jina-embeddings-v2-base-de) model using **SetFit**.
   - They noted that even with a small dataset (**32 samples**) and limited batch size (**4**) and sequence length (**512**), memory issues persisted, especially when using **ContrastiveDataset** with **CosineSimilarityLoss** or **Triplet**.
- **Merge Request #579 becomes required**: A member had to implement changes from [PR #579](https://github.com/huggingface/setfit/pull/579) to get **SetFit** working at all.
   - The implemented changes allowed the member to start training with a bigger dataset before encountering memory issues.
- **Embedding model fine-tuning implicated in memory issues**: The member identified fine-tuning the embedding model as the primary cause of memory problems.
   - Limiting fine-tuning to **5 steps** allowed them to train the classifier with all samples.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1396930991641198742)** (52 messages🔥): 

> `Arxiv endorsement, DeepMind vs OpenAI in AI math, Language for solving math problems, Cross entropy loss` 


- **Arxiv paper requests Endorsement to switch categories**: A member requested an endorsement to move their [Arxiv paper](https://arxiv.org/abs/2507.13362) from **cs.CV** to **cs.CLI**, wondering if adding such content to **LLM training** or **RL data** would be beneficial.
   - They then posted it publicly, after initial reservations about sharing identifying information and being unable to DM it.
- **DeepMind Dominates OpenAI in Math Competitions**: A member shared a [link from a math subreddit](https://link.springer.com/article/10.1007/s10864-024-09559-3) suggesting that **DeepMind** is outperforming **OpenAI** in **AI math**.
   - Attached was an image expressing a viewpoint that English is not the optimal language for math problem-solving and suggesting that **AGI** will generate code unintelligible to humans.
- **Debate Erupts: Will AI Speak Alien Tongues?**: A member said OpenAI style is *more compressed to save tokens, so maybe going more out of human-like language into more out of distribution territory will be the future.*
   - Others pointed out that formal maths already uses symbols unintelligible to normal english readers; someone then shared a link to [Chain of continuous thought paper](https://x.com/fifty_chris/status/1947092589041082436) and epsilon delta definition.
- **Querying Extreme Compression Levels in Language Models**: Members questioned claims of **0.6-1.3** language modeling cross entropy loss, debating its feasibility at a trillion parameters.
   - The discussion clarified that the figure referred to **loss per character**, not per token, and pointed out a potentially clickbait-y tweet and a [new paper](https://arxiv.org/abs/2507.15855).
- **AI's Turbocharged Scams: A Looming Threat?**: A member sarcastically stated the killer app to come out of current gen generative AI will be turbocharged scams.
   - Another member strongly disagreed, arguing that the benefits of AI will outweigh the harm.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1396934729411989666)** (13 messages🔥): 

> `Deepseek V3, MuonClip Algorithm, smolLM3 implementation details, Model Merging` 


- **Deepseek V3 architecture speculated**: Members speculated that the next generation **Deepseek V3** model may utilize a similar architecture to previous versions.
- **MuonClip Algorithm Found?**: A member suggested that the algorithm behind **MuonClip** might be *the gold*.
   - They posted a link to [AlphaXiv](https://www.alphaxiv.org/abs/2505.12082) for more information.
- **smolLM3 Implementation Details Revealed**: A member shared a link to a [Hugging Face blog post](https://huggingface.co/blog/smollm3) about **smolLM3**, calling it one of the best **LLM** implementation papers, because *everything is completely open-source and so are the datasets*.
- **Model Merging Resources Shared**: The same member noted that the **smolLM3** blog post included details on **model merging** and contained more information about *how to actually do it* than most top model papers.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1396951856474882264)** (4 messages): 

> `OpenAI UK deal, Ministry of Silly Walks, AI youtube video` 


- **UK government signs deal with OpenAI**: The [UK government has signed a deal with OpenAI](https://www.theguardian.com/technology/2025/jul/21/openai-signs-deal-with-uk-to-find-government-uses-for-its-models) to explore potential applications of **AI models** within governmental functions.
- **Silly Walks get Digitized**: The Ministry of Silly Walks will finally be digitized.
- **Video powered by Facebook, Palantir, Google, Salesforce, OpenAI, Cognition, XAI**: A member shared [a YouTube video](https://www.youtube.com/watch?v=WByBm2SwKk8) powered by **Facebook, Palantir, Google, Salesforce, OpenAI, Cognition, XAI**.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1397174708310642761)** (11 messages🔥): 

> `lm-eval-harness, AI Math Olympiad, NAIRR compute, AI safety` 


- **Pointers Asked about *lm-eval-harness***: A member asked what a good channel would be to ask questions about **lm-eval-harness**.
- **AI Cracks Math Olympiad, Except Problem 6**: Both **OpenAI** and **DeepMind** models got gold on **International Math Olympiad** few days ago, but still not solving problem 6, which required more creativity than the other problems, is a nice data point towards needing more approaches.
   - A member agrees with Kenneth Stanley that *we still haven't solved a lot of creativity and open endedness, so AI automating math research is also probably not as soon, since that requires much more open endedness than solving International Math Olympiad problems.*
- **NairrPilot Jumpstarts Research**: [NairrPilot](https://nairrpilot.org/opportunities/startup-project) is offering compute targeting **3 month projects** to on-board researchers to the resources available through **NAIRR**, with the end goal of growing the community of researchers fluent in utilizing **NAIRR** compute resources.
- **The AI Singularity Danger**: A member warns that *if you only need to have a single smartass who gives such an AI instructions to produce a better AI, and the dangerous type of singularity is triggered (one when humans just watch in awe and don't understand anything, instead of rising together with the machines)*.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1396945855604588664)** (8 messages🔥): 

> `Weight Decay in Embeddings, Norm Layers scaling, Kimi k2 Paper, Synchronous training` 


- **Weight Decay Wonder: Embeddings Edition**: A member questioned why weight decay, when applied to all embedding and weight matrices, doesn't just shrink everything, letting norm layers handle rescaling.
   - They pointed out that most open-source training codebases *do* apply weight decay to embeddings, sparking a discussion on the impact of this practice.
- **Norm Removal Missed Citation?**: A member noted that [a paper](https://arxiv.org/abs/2302.10322) removing norms wasn't cited in a related discussion.
   - They highlighted that the paper also removes norms, despite not explicitly stating it in the title, making it relevant to the conversation.
- **Asynchronous Training in Kimi k2**: A member expressed surprise that the **Kimi k2** paper details synchronous training, rather than a distributed approach.
   - They contrasted this with **Magistral**, where distributed training is employed, questioning if synchronous training is a less common direction.


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1397073790093103145)** (13 messages🔥): 

> `KAN Activation Functions, B-Spline Curves, Training Dynamics Optimization, Expressivity vs Stability, Cell 9 Spline Training` 


- **Tuning KAN Activation Functions**: A member questioned the activation function training method for **KAN** (*Kolmogorov-Arnold Networks*), pointing out that while the paper mentions **B-spline curves**, the training dynamics are potentially poorly conditioned, especially with numerous knots or control points.
   - The member suggested that using linear interpolation between the two nearest splines instead of summation could enhance training and inference speed, identifying low-hanging opportunities for optimization.
- **Nonlinearity vs. Scaled Training**: One member noted that the extreme nonlinearity of **KANs** might hinder their scalability.
   - Another member countered that nonlinearity enhances expressivity, leading to a discussion about balancing expressivity and stability, with the original member stating, *"It's the infinite tension between expressivity and stability"*.
- **Reparameterizing Spline Training Speeds Up KANs**: A member suggested reparameterizing the spline training method in **KANs** to improve conditioning, even with a slight computational cost.
   - The member linked to [Cell 9 in their ML-code repository](https://github.com/cats-marin/ML-code/blob/main/LearnableActivation.ipynb), showcasing that their spline training method remains well-conditioned even with 10k knots/control points, unlike the paper's approach.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1397321949054631946)** (3 messages): 

> `Sparse MoE Models, SAEs, Interpretability, PEER follow up` 


- **Sparse MoE Models Mimic SAEs**: A member noted that very [sparse MoE models](https://arxiv.org/pdf/2407.04153) resemble **SAEs** because the FFN layer is very wide due to the number of experts.
   - They proposed that this similarity might make these models easier to interpret than dense networks.
- **PEER Gets a Follow-Up**: A member shared a [follow-up paper on PEER](https://arxiv.org/abs/2412.04139) that is related to testing sparse MoE models and their interpretability.
   - They suggested that this paper provides valuable insights into the interpretability of sparse models.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1396929725259190293)** (19 messages🔥): 

> `lm-evaluation-harness, byte latent transformer, facebook/blt-entropy, facebook/blt-1b` 


- **Track performance of existing benchmarks**: Members are working on adding benchmarks to track model performance within the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/issues/3171) specifically expanding the existing benchmarks to track more metrics.
   - The main question is whether to default to outputting both **macro and micro scores for groups** or just one, and the consensus so far is to output both.
- **Implement Byte Latent Transformer (BLT)**: A member inquired about the possibility of implementing the **Byte Latent Transformer (BLT)** for testing within the harness.
   - It was clarified that implementing models isn't the harness's primary function, but implementing backends for models is supported, however, due to **BLT** being a standalone codebase, it's unlikely to be supported unless it's added to HF or VLLM.
- **Byte Latent Transformer (BLT) Integration Complexities**: Members discussed integrating **BLT** into the *transformers* library, suggesting they are working on getting it supported there.
   - The issue is that **BLT** works as an encoder-decoder-decoder model, requiring a wrapper model to combine entropy and **BLT** models into a causal LM for the HF API.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1397306449259925667)** (5 messages): 

> `Amazon infra, SageMaker support, EFA support` 


- **Amazon Employee Seeks GPT-NeoX Support**: An Amazon employee inquired whether **GPT-NeoX** supports their proprietary communication systems, expressing frustration with the lack of internal support.
   - A member responded that they doubted it because they don't think *we've ever worked with Amazon before but that we're always happy to try to help people who want to use our library*.
- **Speculation on SageMaker Team's Involvement**: A member wondered if the inquiry originated from the **SageMaker team**, considering the user's annoyance with the lack of internal support.
   - Another member found this *interesting*.
- **GPT-NeoX's Elastic Fabric Adapter (EFA) Support**: A member clarified that models trained with **Stability compute (Pythia, StableLM, etc.)** utilized **Elastic Fabric Adapter (EFA)**, highlighting the **NCCL EFA plugin** which facilitates this ([aws-ofi-nccl](https://github.com/aws/aws-ofi-nccl)).
   - They stated that *EFA support comes from a lower layer of the stack compared to gpt-neox. It goes gpt-neox->torch.distributed->nccl->EFA*, and that as long as they have **NCCL** set up properly, we're fine.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1396945287184122057)** (4 messages): 

> `A2A Agents Hackathon, LlamaCloud nodes for n8n.io, LlamaParse Feature, Automate PDF parsing` 


- **LlamaIndex sponsors A2A Agents Hackathon**: LlamaIndex is sponsoring the **A2A Agents Hackathon** in San Francisco on July 26, with their VP of Developer Relations speaking and judging alongside experts from [Scale AI](https://t.co/R6J4igjhSH).
- **LlamaCloud Nodes join n8n.io**: **LlamaCloud nodes** for @n8n_io are here, bringing LlamaCloud's document parsing and extraction agents, as well as LlamaCloud indexes as knowledge bases, into n8n automation workflows, to connect LlamaCloud nodes to existing workflows via [this link](https://t.co/UVqwYFkJFR).
- **LlamaParse detects Headers and Footers**: The new **LlamaParse** feature now includes **header and footer detection** in Balanced or Premium mode, automatically detecting page headers and footers, so users can selectively hide them or add prefixes and suffixes via [this link](https://t.co/Px1UjrINJC).
- **Automate PDF Parsing with LLMs**: Automate **PDF parsing and extraction with LLMs** to go beyond OCR limitations with intelligent document understanding, transforming PDFs via [this link](https://t.co/pOn7Tk1CBB).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1396945337297403964)** (51 messages🔥): 

> `Inference Nerf, LlamaParse Authentication, LlamaIndex AWS Bedrock Models, Error Handling in LlamaIndex TS` 


- **Inference Speed Plummets Post-Weekend**: A user reported a significant **nerf in inference speed** after the weekend, with extraction performance severely degraded and is seeking assistance.
   - A support team member requested the **job ID, file, and schema** to reproduce the issue, and confirmed the **two agent limit** was removed.
- **LlamaParse API Authentication Spookiness**: A user faced **authentication issues** with the LlamaParse direct API via n8n and curl, receiving "Invalid authentication token" errors.
   - The user confirmed that using the exact syntax `Authorization: Bearer $LLAMA_CLOUD_API_KEY` in Postman resolved the issue. *"I have no idea how it makes any difference but it does"*.
- **AWS Bedrock Model Mix-Up in LlamaIndex**: A user inquired about the availability of **Sonnet 4 models** in the `@llamaindex/community` package for AWS Bedrock and is confused why it is not working.
   - A member clarified that the user should be using the `@llamaindex/aws` package instead, noting that `@llamaindex/community` might be deprecated and the [documentation](https://ts.llamaindex.ai/docs/llamaindex/modules/models/llms/bedrock) should be updated.
- **Throttling Exceptions Thwart TS Tactic**: A user sought advice on handling **ThrottlingException errors** (Too many tokens) in LlamaIndex TS, particularly for implementing backoff logic with Bedrock models.
   - The user is seeing unhandled rejections and cannot find a way to handle errors when awaiting `agent.runStream(query)` and was recommended to open an issue for discussion.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1396937492695093309)** (17 messages🔥): 

> `Aider as an agent vs tool, Copilot glitch, Aider ignore .gitignore, Qwen3 models vs Aider, Python versions for Aider` 


- ****Aider's Agentic Abilities Questioned****: A user inquired whether **Aider** could function more like an agent, similar to **VSCode's Cline** or **Copilot**, by automatically finding and editing files, referencing an attached image ([image.png](https://cdn.discordapp.com/attachments/1131200896827654149/1397152245769965639/image.png?ex=688157ad&is=6880062d&hm=61b80c9ced86a84cfa31ca4aa8556e33b989d3960a697af83417f79a476b4cb2&)).
   - Another user clarified that **Aider** is primarily a tool that gives the developer more independence, acting as an editor rather than an autonomous agent. They suggest **OpenCode** or **Gemini CLI** as alternatives closer to **Cline** in terms of agentic behavior.
- ****Troubleshooting Aider and .gitignore****: A user experienced issues with **Aider** not ignoring files specified in **.gitignore**, and sought assistance on how to configure **Aider** to properly exclude these files.
- ****Qwen3 Models Outperforming Aider?****: There's discussion around recent open weight models, specifically **Qwen3**, performing well on benchmarks, but potentially regressing in achievements compared to **Aider Polyglot**.
- ****Python Version Requirements for Aider****: A user mentioned that **Ubuntu 20.04**, which defaults to **Python 3.8**, is deprecated and may not be suitable for modern software like **UV** and **Aider**.
   - It was advised that **Python 3.11 - 3.12** is generally good, while **3.13** is still too new.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1396943934680465599)** (26 messages🔥): 

> `Aider Polyglot examples, Model Looping, Aider summarization calls, Correct Edit Format` 


- **Aider's Polyglot examples use single generation per round**: Members are questioning whether the **Aider Polyglot examples** on sites like [leaderboard.techfren.net](https://leaderboard.techfren.net/) involve only a single generation per round, adhering to the **n_generations constraint** in Aider.
   - The question is whether certain models might be running multiple LLM calls in a loop between code execution attempts, effectively *cheating* the system.
- **Proprietary providers using tool calling at inference time**: Proprietary providers could potentially use **tool calling at inference time** or other similar strategies.
   - Aider uses a [summarizer model](https://github.com/Aider-AI/aider/blob/f38200c511674e83a1b34a44e85beb77ee78f5c7/aider/coders/base_coder.py#L510) when the **max chat history gets too long**.
- **Formatting retries included in LLM calls**: It's assumed that **linter/formatting retries to LLM** are included in the LLM calls.
   - The *correct edit format* on the polyglot benchmark refers to whether the **search part of the edit was correct** and found in the source.
- **Tab Completion Behavior Change**: A member asked about whether there is a way to change the behavior of the **tab completion** when adding files.
   - They would like to **stop the completion to directory**, because they find it hard to navigate files, and feel like they're missing a good tip on how to use it.
- **Aider asks for previous file content**: One member wondered why **Aider asks the model to output the previous file content** of what to patch.
   - The answer is: It saves cost if the model only needs to output **where** to patch, similar to using *ed* the line editor.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1396994501461016768)** (7 messages): 

> `NotebookLM use cases, Obsidian Plugin, Reading books, Chat history retrieval` 


- **Users explore NotebookLM use cases**: Members discussed use cases for **NotebookLM**, including managing **Obsidian** vaults for personal life and **TTRPG** games.
   - Other members suggested using **NotebookLM** to listen to books, create mind maps from them, and ask questions directly from uploaded **PDF** documents.
- **Obsidian plugin search commences**: A member inquired about the existence of a plugin for **Obsidian**.
   - Another member clarified that while there isn't a direct plugin, **NotebookLM** can read **.md files**, allowing users to copy and paste content from **Obsidian**.
- **NotebookLM chat history under scrutiny**: One user inquired whether there's a way to retrieve old chat history and if **NotebookLM** saves previous chats and summaries.
   - No answer was given.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1396936451278504068)** (30 messages🔥): 

> `Google Ultra AI Subscription Benefits, NotebookLM Service Unavailable Error, PDF Image Reading Capabilities, NotebookLM for American Yawp Notes, Gemini Pro Model Integration` 


- **Ultra AI Promises Better Models, "Later this year"**: Users noticed that the Google Ultra AI subscription promises access to *better models in notebook*, but there's no immediate way to change models and the feature is labelled *"Later this year"*.
- **"Service Unavailable" Error Puzzles Users**: Several users reported a *"Service unavailable - You tried to access a service that isn't available for your account"* error when using NotebookLM, particularly on Macbooks, even with an active subscription and working mobile app.
- **NotebookLM can Read PDF images**: A user asked if **NotebookLM** can read both text and images of a PDF, and another user confirmed it can, but prioritizes text unless specified otherwise in the prompt.
- **PDF Upload Issues Plague Users**: Multiple users are experiencing persistent errors when uploading PDFs to NLM, despite trying various files and file types; a user suggested *deleting cookies* as a potential fix.
- **Podcast Length Options Mysteriously Vanish**: A user reported that the **"Shorter," "Default," and "Longer" options** for adjusting podcast length in the Audio Overview section are missing, and at least one other user confirmed experiencing the same issue.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1396948051255496775)** (28 messages🔥): 

> `Qwen3-235B-A22B, Kimi-K2 Tech Report, RLHF Reward Model, Kimi K2 Style` 


- **Qwen3-235B-A22B Instruct Released**: The **Qwen3-235B-A22B-Instruct model** has been released, but not for smaller sizes like **30B**, making it less accessible until a good API provider hosts it, available [on HuggingFace](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507).
- **Kimi-K2 Tech Report Raises Eyebrows**: The **Kimi-K2 tech report** was shared, with concerns raised about potentially addictive elements aimed at younger users, accessible [on GitHub](https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf).
- **RLHF Reward Model Questioned**: Questions arose regarding the **Reward model in RLHF**, specifically whether it has a single output dimension or a linear layer with output dimension equal to window for discounted reward calculation.
- **Kimi K2's odd mobile game dev style**: A member noticed a peculiar style in **Kimi K2's** responses when generating ideas for a mobile space sim game, specifically the use of terms like *"juice"* for VFX and *"meta/shop"*.


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1397043364414619711)** (5 messages): 

> `Computer vision applications, Generative models, Flow matching, VLM fine-tuning, Success Principles` 


- **Computer Vision Fans Assemble!**: Two members expressed their excitement in **computer vision applications** and looked forward to future interactions.
   - One member asked *"What's cooking on your mind?"* and the other responded with **generative models with flow matching and vlm fine-tuning**.
- **Small Wins Propel Success**: A member shared a motivational message that *"Success is the series of small wins"*.


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1397015484045594635)** (8 messages🔥): 

> `JSON Schema Regression, Embed v4 Rate Limit` 


- **JSON Schema Request Fails**: A member reported a regression with **JSON schema** requests, noting that a previously working input now fails with an *"invalid 'json_schema' provided: missing required field 'type'"* error message.
   - The previously successful output was a simple JSON object indicating whether a banana is a fruit: `{"is_fruit": true}`.
- **Embed-v4 Rate-limits reserved for Enterprise Customers**: A user inquired about increasing the rate limit for **Embed v4**.
   - A member of the Cohere team responded that enhanced rate limits are currently reserved for enterprise customers with a minimum committed spend. If you qualify, contact them at [support@cohere.com](mailto:support@cohere.com).


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1397042897315954792)** (6 messages): 

> `VLMs, Generative models, 3D reconstruction, AI platforms` 


- **MLE Student Masters VLMs and Gemini**: A student is currently working on **VLMs**, **Generative models** (Flow matching), and **3D reconstruction**, and prefers **pytorch** and **gemini**.
   - They hope to gain knowledge, friends, and research project collaborations from this community.
- **LLM Engineer Seeks Interaction and New Research**: One member is focused on **reasoning**, **LLMs**, and **RL**, utilizing tools like **vllm**, **hf**, **pt**, and **wandb**.
   - They are looking to interact with others, find research opportunities, and discover new advancements in the field.
- **AI Architect Aims to Integrate AI for Business**: An **AI & Enterprise architect** is assisting businesses in integrating **AI** to enhance efficiency and profitability.
   - They are interested in **AI platforms** that use **natural language** to address business problems and are eager to learn, connect, and share insights.
- **AI Engineer Builds Innovative LLM Products**: An **AI Engineer/Head of AI** at Elevancesystems is developing innovative **AI/LLM products**.
   - They look forward to sharing and negotiating new technologies and solutions for the real business world.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1396948880603746345)** (16 messages🔥): 

> `Mojo vs C++ Vectorization, Modular Careers, Modular Open Source Contribution, Mojo GPU Programming, Mojo Package Manager` 


- **Mojo's Vectorization Capabilities Spark Debate**: A user inquired about how **Mojo** stacks up against **C/C++** for **CPU vectorized tasks**, leading to a discussion about the strengths and weaknesses of each language.
   - One member argued that while both can achieve similar performance with inline assembly, **Mojo** simplifies the process, particularly for complex code, surpassing the capabilities of **ISO C++**.
- **Job Seekers Find Opportunities at Modular**: In response to a query about development opportunities, a link to the **Modular Careers** page was shared.
   - The link is [https://www.modular.com/company/careers](https://www.modular.com/company/careers).
- **ML Compiler Expert eyes Modular Contribution**: A user with experience in **ML compilers at ARM** expressed interest in contributing to **Modular open source** projects, specifically **MAX AI kernels**.
   - They sought advice on resources to review before engaging with beginner-friendly issues.
- **Mojo GPU Programming via Puzzles Highlighted**: A recommendation was made to explore **Mojo's GPU programming** capabilities through the [Modular Puzzles](https://puzzles.modular.com/introduction.html) to understand the language's approach.
   - The suggestion aimed to provide a practical introduction to **GPU programming in Mojo** for a potential contributor.
- **Mojo Package Manager Code Shared**: A member shared their **Mojo package manager** project on [GitHub](https://github.com/luigithejokesterplus/birdinstall), inviting others to review the code.
   - Another member welcomed the contribution and pointed to the [Modular community repo](https://github.com/modular/modular-community) as a potential hub.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1397163592943407164)** (1 messages): 

> `Mojo async design, async/await ecosystem split` 


- **Mojo's Async Design Drafted**: A member posted an [updated design sketch](https://github.com/modular/modular/pull/3986#issuecomment-3102049612) for `async` in **Mojo**.
   - The stated goal is to avoid the *"ecosystem split"* issue (library + code duplication) that every existing language with **async/await** suffers from.
- **Async/Await Ecosystem Split Avoidance**: The primary motivation behind the new `async` design in **Mojo** is to prevent the library and code duplication issues seen in other languages.
   - This design aims to create a unified ecosystem where **async** code can seamlessly interact with synchronous code, reducing the need for separate libraries and codebases.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1397303241183727749)** (1 messages): 

> `Max vs llama.cpp, CPU serving performance` 


- **Max vs llama.cpp CPU Serving Showdown**: A member inquired about **benchmarks** comparing **Max** and **llama.cpp** for CPU serving.
   - The focus was on identifying which performs better when **serving directly on the CPU**.
- **CPU Serving Performance Benchmarks Sought**: The user specifically requested **performance benchmarks** to evaluate **Max** and **llama.cpp**. 
   - The use-case highlighted was **CPU serving**, implying a need for efficient CPU utilization.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1397200878167134330)** (6 messages): 

> `RL Torchtune Release, RL Libraries` 


- **RL Torchtune Eyes Pre-PyTorch Conf Release**: The new **Torchtune** with a focus on **RL** is aimed for release before the PyTorch conference, focused on post-training and scaling beyond **600B parameters**.
   - While acknowledging the trade-off, the goal is to facilitate quick, small-scale experiments that can readily scale to production levels, tackling the scaling challenge first, but a user pointed out the library [RL2](https://github.com/ChenmienTan/RL2) that might be of interest.
- **Small Scale GPU Support Still a Priority**: While the primary focus is on large-scale RL, small-scale support is not being abandoned entirely.
   - The aim is to enable users to conduct quick experiments at a smaller scale that can be easily transferred and scaled to production.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1397242215759089796)** (2 messages): 

> `Fine-tuning 70B models, recipe_state.pt size, torch.save behavior, Distributed checkpointing` 


- **Local Shard Checkpointing Suspicions Surface**: A member noticed that when fine-tuning a **70B model**, the `recipe_state.pt` file is only ~**30GB**, and raised a concern that `torch.save` might be storing only the local shard in non-distributed checkpointing scenarios.
   - They observed that tensors loaded with `torch.load` appear to be on `cuda:3` with **DTensors** and sharded placements, suggesting the possibility of overwriting and only storing the local shard.
- **Members ask if `torch.save` is always local?**: The member questioned whether `torch.save` is always local and if there's a potential issue with non-distributed checkpointing, leading to overwriting and storing only the local shard.
   - They proposed moving from `torch.save` to a new distributed checkpointer and asked for a sanity check on their reasoning.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1396951680096014490)** (6 messages): 

> `MCP vs System Tool Call, MCP Tool Purpose, Image Context, New Member Introduction` 


- **MCP Question Arises**: A member questioned why something is being implemented as an **MCP (Model Control Program)** instead of a simpler **system tool call**.
   - Another member explained that *the purpose of any MCP tool is to give an LLM capabilities it does not natively have*.
- **Image Context Issue Spotted**: A member pointed out a good point on the **context of the images**.
   - Another member hadn't even considered that, thanking them for the suggestion.
- **New Member Joins Channel**: A new member joined the channel, greeting everyone and thanking them for the invite.
   - This event marked the arrival of a new participant in the ongoing discussions.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1396952294456823872)** (2 messages): 

> `WordPress integration for Claude, MCP server for WordPress` 


- ****WordPress Wonders** with Claude's Conversational CMS Control**: A member announced the release of a **WordPress integration for Claude Desktop** which allows controlling WordPress directly through Claude conversations, eliminating copy-paste workflows; the [repo](https://github.com/docdyhr/mcp-wordpress) was also shared.
   - The member added that using it to manage their blog has changed how they think about content creation as **Claude** can see existing posts, understand site structure, and make updates.
- **MCP server supports any MCP client for Wordpress**: A member inquired if the WordPress integration is only supported for **Claude Desktop**, suggesting it might work with any **MCP client** given its local MCP server setup.
   - However, there was no response to confirm or deny this.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1397335529393229965)** (1 messages): 

> `DSPy, Python user group, Local Tech Meetups` 


- **DSPy Spreads to Local Pythonistas**: A member announced they would be introducing **DSPy** to a local Python user group, aiming to showcase its capabilities.
   - They also shared a [YouTube video](https://www.youtube.com/watch?v=1WKA8Lw5naI) to provide a visual and accessible introduction to **DSPy** for the group.
- **Tech Community Engagement Boosted**: The member's initiative highlights the importance of local tech meetups in disseminating knowledge and fostering community engagement.
   - By presenting **DSPy** at a Python user group, they're contributing to the broader adoption and understanding of the framework within the tech community.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1397018163870437408)** (5 messages): 

> `Teleprompters, DSPy Modules, Professional Services orgs, AWS Services` 


- **Professional Services engineers bespoke AWS services**: A member of a Professional Services org stated they engineer bespoke solutions for large enterprise customers, and if several customers have the same request, those features are added to the **AWS services** themselves.
- **Teleprompters accept DSPy Modules**: A member asked if the base **Teleprompter** accepts any Module as a student, and whether that's accurate throughout all teleprompters.
   - Another member clarified that any `dspy.Module` subclass is allowed, *nothing else is allowed*.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1397279327460134942)** (3 messages): 

> `Whisper PR speed, shipping containers for tinyboxes, Rust for Tinygrad` 


- **Whisper PR runs slowly**: A member expressed frustration that their **Whisper PR** is running slowly, as they aim for it to be under 500 lines and near **OpenAI's speed**.
   - They want it to be *simple and sweet*.
- **Shipping containers house TinyBoxes**: A member had a *crazy idea* of using **shipping containers** to house **tinyboxes**, suggesting modularity, cooling benefits, and mobility.
   - They pondered the cost and security aspects, jokingly suggesting the name *tinycontainer*.
- **Rust and Tinygrad compete**: A member noted that **Rust** is a completely different market from **Tinygrad** but would be interesting to benchmark against TG.
   - They speculated that if **tinycorp** enters the enterprise LLM appliance market, all frameworks and custom solutions will compete there.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1396930213660856321)** (1 messages): 

> `CUDA on Windows, CPU backend setup, LLVM backend setup` 


- **Enabling CUDA on Windows Simplified**: A member shared a method to enable **CUDA** on Windows, including a [patch file](https://cdn.discordapp.com/attachments/1070745817025106080/1396930213312725022/cuda_windows.patch?ex=688131a4&is=687fe024&hm=e6965a699c395de25b72762e696fce5fb5545f656120ee70353c584fe468bbb9&) for applying the changes.
   - The process involves specific environment configurations for different backends.
- **Setting up CPU Backend**: To enable the **CPU backend**, users only need to ensure that *clang* is added to the **PATH** environment variable.
   - This makes *clang* accessible system-wide, which is essential for compiling and running code that targets the CPU.
- **LLVM Backend Configuration**: For the **LLVM backend**, it's necessary to set the **LLVM_PATH** environment variable to point to *LLVM-C.dll*, such as `set LLVM_PATH=c:\llvm-16.0.2-windows-amd64-msvc17-msvcrt\bin\LLVM-C.dll`.
   - This configuration directs the system to the correct **LLVM** dynamic library, ensuring that **LLVM**-based compilations function correctly.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1397020461308710992)** (3 messages): 

> `Certificate Issues, Writing Submission Form Issue` 


- **Certificate Delivery Delay Debated**: A member inquired about a missing certificate, providing [two email addresses](mailto:terrence_rideau@yahoo.com,terrence.rideau@google.com) for verification.
   - The staff indicated that *no certificate declaration form* had been received under either email, and it turns out the **writing submission form** was not properly submitted.
- **Writing Submission Snafu Strikes Student**: A student reported missing the **certificate declaration form** submission, despite attending sessions, passing tests, and writing the article.
   - The member expressed disappointment but appreciated the course content, thanking the presenters and the team.
