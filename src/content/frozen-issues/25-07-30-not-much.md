---
id: MjAyNS0w
title: not much happened today
date: '2025-07-30T05:44:39.731046Z'
description: >-
  **Chinese AI labs** have released powerful open-source models like **GLM-4.5**
  and **GLM-4.5-Air** from **Zhipu AI**, **Qwen3 Coder** and **Qwen3-235B** from
  **Alibaba**, and **Kimi K2** from **Moonshot AI**, highlighting a surge in
  permissively licensed models. **Zhipu AI's GLM-4.5** is a 355B parameter MoE
  model competitive with **Claude 4 Opus** and **Gemini 2.5 Pro**. **Alibaba's
  Qwen3 Coder** shows strong code generation performance with a low edit failure
  rate, while **Moonshot AI's Kimi K2** is a 1 trillion-parameter MoE model
  surpassing benchmarks like **LiveCodeBench**. In video and image generation,
  **xAI** launched **Grok Imagine**, and **Wan2.2** impressed with innovative
  image-to-video generation. Robotics advances include **Figure's Figure-01 and
  Figure-02** humanoid robots and **ViTPose++** for pose estimation in
  basketball analysis. **SmolLM3** training and evaluation code was fully
  released under Apache 2.0. **OpenAI** introduced **Study Mode** in **ChatGPT**
  to enhance interactive learning, and **Runway** rolled out **Runway Aleph**, a
  new in-context video model for multi-task visual generation. The community
  notes a competitive disadvantage for organizations avoiding these Chinese
  open-source models. *"Orgs avoiding these models are at a significant
  competitive disadvantage,"* noted by @corbtt.
companies:
  - zhipu-ai
  - alibaba
  - moonshot-ai
  - x-ai
  - figure
  - openai
  - runway
  - mlx
  - ollama
  - deeplearningai
models:
  - glm-4.5
  - glm-4.5-air
  - qwen3-coder
  - qwen3-235b
  - kimi-k2
  - grok-imagine
  - wan-2.2
  - smollm3
  - figure-01
  - figure-02
  - vitpose++
  - chatgpt
topics:
  - model-releases
  - model-performance
  - moe
  - image-generation
  - video-generation
  - pose-estimation
  - robotics
  - training-code-release
  - interactive-learning
  - in-context-learning
people:
  - yuchenj_uw
  - corbtt
  - reach_vb
  - ollama
  - deeplearningai
  - gdb
  - sama
  - c_valenzuelab
  - adcock_brett
  - skalskip92
  - loubnabenallal1
  - hojonathanho
  - ostrisai
---


**a quiet day.**

> AI News for 7/29/2025-7/30/2025. We checked 12 subreddits, 544 Twitters and 29 Discords (227 channels, and 5378 messages) for you. Estimated reading time saved (at 200wpm): 467 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

A lot of hype about GPT5 releasing tomorrow due to random Twitter anon speculation.

---

# AI Twitter Recap

**Model Releases and Performance**

- **China's Open-Source Offensive**: In July, Chinese labs released a wave of powerful, permissively licensed models, a trend highlighted by [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1950034092457939072). Key releases include **GLM-4.5** & **GLM-4.5-Air** from **Zhipu AI**, **Wan-2.2** (video), the **Qwen3 Coder** and **Qwen3-235B** family from **Alibaba**, and **Kimi K2** from **Moonshot AI**. This contrasts with a perceived slowdown in Western open-source releases, prompting [@corbtt](https://twitter.com/corbtt/status/1950334347971874943) to note that orgs avoiding these models are at a "significant competitive disadvantage."
- **Zhipu AI's GLM-4.5 Models**: **Zhipu AI** released **GLM-4.5**, a 355B parameter MoE (32B active) model, and **GLM-4.5-Air**, both with **MIT licenses**. The company announced [they are working to scale resources](https://twitter.com/Zai_org/status/1950164491125043515) due to high demand. The models are noted as being competitive with **Claude 4 Opus** and beating **Gemini 2.5 Pro** [in some benchmarks](https://twitter.com/Zai_org/status/1949970927006949430). The community quickly made them available on platforms like **MLX** and **DeepInfra**.
- **Qwen3 and Kimi K2 Models**: **Alibaba's Qwen3 Coder** shows strong performance, with a low **5.32% diff edit failure rate** in **Cline**, placing it alongside **Claude Sonnet 4** and **Kimi K2** [according to @cline](https://twitter.com/cline/status/1949973297455599998). A **30B MoE (3B active)** version with a 256K context is now runnable locally via **MLX** and **Ollama**, as noted by [@reach_vb](https://twitter.com/reach_vb/status/1950263476271947822) and [@ollama](https://twitter.com/ollama/status/1950291777216262259). **Moonshot AI's Kimi K2**, a **1 trillion-parameter MoE (32B active)** model, was released with a modified MIT license and surpasses other open-weights models on benchmarks like **LiveCodeBench** and **AceBench** [as reported by @DeepLearningAI](https://twitter.com/DeepLearningAI/status/1950183277161005418).
- **Video and Image Generation**: **xAI** launched **Grok Imagine**, an image and video generation tool, [behind a waitlist](https://twitter.com/chaitualuru/status/1949946519869685952). The **Wan2.2 5B** video model impressed developers with its approach to **Image-to-Video (I2V)**, where each latent frame has its own denoising timestep, potentially allowing for infinitely long video generation, [as analyzed by @ostrisai](https://twitter.com/ostrisai/status/1950129158618591646). **Ideogram** released **Ideogram Character**, a character consistency model that works with a single reference image, [noted by @hojonathanho](https://twitter.com/hojonathanho/status/1950261122365333806).
- **Vision and Robotics**: **Figure** showcased a comparison between its **Figure-01** and the newer **Figure-02** humanoid robots, highlighting advancements in hardware and capability [in a video shared by @adcock_brett](https://twitter.com/adcock_brett/status/1950291267730207125). **ViTPose++** demonstrated impressive pose estimation, accurately tracking complex interactions between basketball players, which is now being integrated into a basketball analysis AI that can determine if a player is in the paint [according to @skalskip92](https://twitter.com/skalskip92/status/1950231824933982428).
- **SmolLM3 Code Release**: The full training and evaluation code for **SmolLM3** has been released, including pretraining scripts (**nanotron**), post-training code (**TRL/alignment-handbook** for SFT+APO), and evaluation scripts, along with over 100 intermediate checkpoints, all under an **Apache 2.0 license** [as announced by @LoubnaBenAllal1](https://twitter.com/LoubnaBenAllal1/status/1950139809034305568).

**AI Agents, Tooling & Applications**

- **ChatGPT Study Mode**: **OpenAI** is rolling out **Study Mode** in **ChatGPT**, an interactive feature designed to guide users through learning concepts step-by-step, acting as a tutor rather than just providing answers, [as announced by @gdb](https://twitter.com/gdb/status/1950309323936321943) and [@sama](https://twitter.com/sama/status/1950299705751327149).
- **Runway Aleph In-Context Video Model**: **Runway** is rolling out access to **Runway Aleph**, a new in-context video model for multi-task visual generation. [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1950138170806312974) demonstrated its power by comparing a complex, manual video editing workflow for a "day-to-night" effect with simply prompting Aleph to "make it night". A similar comparison was made for [removing cars from a scene](https://twitter.com/c_valenzuelab/status/1949921138689396976) and [adding an explosion](https://twitter.com/c_valenzuelab/status/1950257984715571606).
- **Google's AI Mode in Search**: **Google** expanded its **AI Mode** in Search to the U.K. and introduced new features, including the ability to upload photos and PDFs for queries, a "Canvas" for organizing projects, and "Search Live" for real-time help, [as detailed by @Google](https://twitter.com/Google/status/1950241246779232260).
- **LangChain & LangGraph for Agentic Workflows**: **LangChain** released a guide on applying six common context engineering approaches using **LangGraph**, providing both video and code examples [in a popular tweet](https://twitter.com/LangChainAI/status/1950226846538485918). They also highlighted how to build a self-correcting RAG agent for code generation. The ecosystem continues to grow, with [**LangSmith Traces** now integrating server logs](https://twitter.com/LangChainAI/status/1949948616182768010) for better observability.
- **Perplexity's Comet Browser**: **Perplexity** has seen strong initial adoption for its **Comet** browser, with CEO [@AravSrinivas](https://twitter.com/AravSrinivas/status/1950042752655241234) noting that its default search is **Perplexity**, potentially driving significant query volume. He also demonstrated Comet performing a complex task of [booking a flight on United, including seat selection](https://twitter.com/AravSrinivas/status/1949937085164482846).
- **Development & Tooling**: **BlockDL**, a free, open-source GUI for visually designing **Keras** neural networks, was released by [@fchollet](https://twitter.com/fchollet/status/1950244806967603207). On the tooling front, the new **Hugging Face jobs CLI** is now powered by **uv** for faster environment setup, [as shared by @_lewtun](https://twitter.com/_lewtun/status/1949915717836431744). For developers building agentic apps, [@_avichawla](https://twitter.com/_avichawla/status/1950282234893656101) highlighted a method to deploy any model, RAG, or agent as an **MCP server** in just 10 lines of code.

**Infrastructure, Efficiency & Optimization**

- **Long Context Training on H200**: [@StasBekman](https://twitter.com/StasBekman/status/1950232169227624751) demonstrated that **1.2M sequence length** training for a Llama-8B model is now possible on a single **H200 GPU**. This was achieved using a combination of **ALST**, **FA3 (FlashAttention-3)**, and **Liger-Kernel**, with the latter two having recently received fixes for int64 indexing.
- **GSPO in TRL**: **Alibaba's Group Sequence Policy Optimization (GSPO)** algorithm, which has gained significant attention, is now available in the Hugging Face **TRL** library, as [announced by @_lewtun](https://twitter.com/_lewtun/status/1949951668914659636).
- **AMD Contributions to llama.cpp**: [@ggerganov](https://twitter.com/ggerganov/status/1950047168280060125) noted that **AMD** teams are now actively contributing to the **llama.cpp** codebase, signaling broader hardware support for the popular inference framework.
- **StepFun Open Sources StepMesh**: Chinese AI company **StepFun** has open-sourced **StepMesh**, a communication library designed for inference systems using **Attention-FFN disaggregation**, [as noted by @teortaxesTex](https://twitter.com/teortaxesTex/status/1950127131754651655).
- **Qdrant Edge for On-Device Vector Search**: **Qdrant** has launched a private beta for **Qdrant Edge**, a lightweight, embedded vector search engine designed to run on-device for robotics, mobile, and IoT applications, [as announced by @qdrant_engine](https://twitter.com/qdrant_engine/status/1950165409639833603).

**Research, Techniques & Evaluation**

- **History of Backpropagation**: [@SchmidhuberAI](https://twitter.com/SchmidhuberAI/status/1950194864940835159) provided a detailed history of **backpropagation**, clarifying that its modern form was first published in **1970** by **Seppo Linnainmaa**, with precursors from **Henry J. Kelley** in **1960**. He emphasizes that it is not simply the chain rule but an efficient application of it for neural networks.
- **The Evaluation Crisis**: A growing sentiment is that standard benchmarks are becoming less reliable. [@ShunyuYao12](https://twitter.com/ShunyuYao12/status/1950090043344707832) asked, "How to evaluate llms when we can’t trust benchmark numbers anymore?". [@teortaxesTex](https://twitter.com/teortaxesTex/status/1949912968394940518) echoed this, stating excitement will come when a model releases alongside a "radically new eval suite." **DailyBench** was released by [@jacob_dphillips](https://twitter.com/andersonbcdefg/status/1949936665637593102) as an automated daily benchmark to track frontier models on fresh problems.
- **New Optimization Techniques**: A paper on **Reflective Prompt Evolution** shows it can outperform **GRPO**, highlighting the power of learning via natural-language reflection, as [shared by @lateinteraction](https://twitter.com/lateinteraction/status/1949984215191208078). **Alibaba's Group Sequence Policy Optimization (GSPO)** paper was the third most popular on Hugging Face for July, with [@ClementDelangue](https://twitter.com/ClementDelangue/status/1949934196148895799) predicting it will have a massive impact.
- **Physics of LLMs**: Researchers released code for their "Physics of Language Models" work, claiming their **8B@1T** model beats **Llama-3.1-8B** using only **7% of the compute**, [as shared by @giffmana](https://twitter.com/giffmana/status/1950276478861517236).
- **Reasoning and Consciousness**: A discussion on what constitutes reasoning emerged, with [@teortaxesTex](https://twitter.com/teortaxesTex/status/1950158521493811458) provocatively suggesting it's a "super-Turing computation" that should be able to solve the halting problem. Meanwhile, [@jxmnop](https://twitter.com/jxmnop/status/1950229423849869672) reminisced about how the field has moved from arguing if GPT-2 understood negation to debating "mostly-sentient IMO-winning" models.

**Industry & Broader Discourse**

- **The $400M Meta Offer Story**: A major point of discussion was the revelation that top AI talent is turning down **$400 million** offers from **Meta**, [a tweet from @willdepue that went viral](https://twitter.com/willdepue/status/1950253835064086979). This has led to speculation about what other companies are building that could inspire researchers to reject such large offers.
- **Energy as a Bottleneck**: A comment from a former **Meta** employee surfaced, stating that **energy** is the biggest bottleneck to scaling compute, even more so than capital for GPUs. [The tweet was amplified by @code_star](https://twitter.com/code_star/status/1950263396420767845).
- **API vs. Open Weights Safety**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1950226004984942829) argued against the idea that API-based models are inherently safer than open-weights models. He posits that by making models easier to use, APIs can increase the volume of misuse from bad actors by "orders of magnitude" without adding significant control.
- **Hiring and Community**: **Anthropic** announced [it is expanding its Fellows program](https://twitter.com/EthanJPerez/status/1950278824102678586), which pairs external researchers with internal teams to work on safety problems. **Sakana AI** is [hosting an open house](https://twitter.com/SakanaAILabs/status/1950016555799953523) to recruit for its Applied Engineer team.
- **Geopolitics**: Multiple high-impression tweets touched on the political climate, including a tweet from **Speaker Pelosi** criticizing a decision by **Donald Trump** regarding a visit from Taiwan's President Lai, [shared by @zacharynado](https://twitter.com/zacharynado/status/1950056521330532640).

**Humor & Memes**

- **Glowing Gardens and Architectural Diffusion**: A tweet joking "half my garden glows in the dark now" in response to a news story about glowing plants [gained massive traction via @nptacek](https://twitter.com/nptacek/status/1950265375658020991). The "they did diffusion on *checks notes* a house" meme also circulated widely, [retweeted by @sedielem](https://twitter.com/sedielem/status/1950190227475046877).
- **Bizarre History and Passwords**: A popular tweet from [@DavidSHolz](https://twitter.com/DavidSHolz/status/1950104321783218193) shared a 1930s proposal to build a **190MPH roller coaster** on top of the Golden Gate Bridge. In a separate viral post, [@jxmnop](https://twitter.com/jxmnop/status/1950272775052284351) shared a screenshot of a user's password being "Woman", with the comment "you can't make this stuff up".
- **AI Parody**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1950323192641503571) posted a "Hot Dog or Not Hot Dog" example using the **Comet** browser. [@typedfemale](https://twitter.com/typedfemale/status/1950337102828143000) posted a meme about "bisexual luke farritor".
- **Relatable Engineering Life**: A post about being physically locked in a room resonated with the feeling of being "locked in" on a project, [posted by @stevenheidel](https://twitter.com/stevenheidel/status/1950316382450823320). A pitch at **a16z** featuring a "Magic Talking Dog and a human pyramid" was [shared by @KevinAFischer](https://twitter.com/KevinAFischer/status/1949958038905127340).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen3-30B-A3B and Related Model Launches and Performance Discussion

- [**Qwen3 Coder 30B-A3B tomorrow!!!**](https://i.redd.it/zv92612t11gf1.png) ([Score: 429, Comments: 49](https://www.reddit.com/r/LocalLLaMA/comments/1md93bj/qwen3_coder_30ba3b_tomorrow/)): **The post references the imminent release of Qwen3 Coder 30B-A3B, an AutoAWQ-quantized variant (A3B implied) of the Qwen3 Coder model, suggesting high anticipation within the open-source LLM community. The image likely teases or confirms the release timing, contributing to ongoing competitive momentum highlighted in recent Qwen releases. The technical context from user comments indicates enthusiasm for Qwen's rapid progress and hints at its impact on rival projects and figures (e.g., Llama/Lizard and Altman).** Commenters are notably excited about Qwen's pace of innovation, regarding it as overtaking alternatives ("Friendship ended with Lizard boy"), and humorously referencing competitive tensions with OpenAI leadership ("Scam Altman").
    - One commenter highlights a distinction between the Qwen3 Coder 30B-A3B variant and a hypothetical Qwen3 Coder 32B with all parameters fully active, suggesting that the 30B-A3B release likely involves some degree of parameter deactivation or pruning compared to a model leveraging all 32 billion parameters. This distinction may have implications for performance or memory requirements when running these models.
- [**🚀 Qwen3-30B-A3B-Thinking-2507**](https://i.redd.it/eaag1cpuz0gf1.jpeg) ([Score: 414, Comments: 104](https://www.reddit.com/r/LocalLLaMA/comments/1md8t1g/qwen330ba3bthinking2507/)): **The image appears to be a non-technical promotional post for the new Qwen3-30B-A3B-Thinking-2507 model, highlighting its competitive performance on reasoning tasks (math, science, code), tool use capabilities, and a native 256K-token context window (extendable to 1M). Links to Hugging Face and ModelScope provide model details. The post is part of a series of recent Qwen3 releases, with commenters noting upcoming coder variants and availability of GGUF quantized versions for efficient inference.** Commenters express excitement about upcoming releases (such as Qwen3-30B-A3B-Coder), while others discuss the availability of quantized GGUF versions for practical use. Some users show fatigue with the frequency of new Qwen releases.
    - Benchmarks for Qwen3-30B-A3B-Thinking-2507 were shared, highlighting its competitive performance. An image link shows rankings comparing the model to other 30B-70B class LLMs, suggesting it performs at or near the top for its parameter class based on current evaluation benchmarks.
    - Downloads for the model in GGUF format are now available on Hugging Face at https://huggingface.co/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF, ensuring accessibility for users seeking quantized versions suitable for optimized inference with libraries like llama.cpp or similar frameworks.
- [**Qwen/Qwen3-30B-A3B-Thinking-2507 · Hugging Face**](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507) ([Score: 127, Comments: 32](https://www.reddit.com/r/LocalLLaMA/comments/1md8rxu/qwenqwen330ba3bthinking2507_hugging_face/)): **The post discusses running the Qwen3-30B-A3B-Thinking-2507 model (available on Hugging Face) on a P40 24GB GPU using Q4 UD XL GGUF quantization with a context window up to 90,000 tokens. Performance benchmarks show decoding speeds of ~40 tokens/sec at context 0, dropping to ~10 tokens/sec at 40k context tokens, with reading at 110t/s at maximum context. The model was evaluated on a complex, multi-step code generation task involving GUI/game dashboard creation, handled via Roo Code with iterative prompts, and demonstrated fewer initial mistakes compared to the Flash 2.5 model.** Commenters provide technical resources (GGUFs), share screenshots of the benchmark environment, and note stable performance at high context, making Qwen3-30B suitable for extended code generation tasks.
    - A user reported running Qwen3-30B-A3B-Thinking-2507 on a P40 24GB GPU with a Q4 UD XL quantization and a `90k` context length, achieving `40~25~10t/s` (tokens/sec) generation speed from 0~10~40k context respectively. They described an extensive coding task involving multi-step prompting (9 core tasks and several fix-up prompts) and observed performance of `10t/s` writing and `110t/s` reading at the 40k context window, noting *fewer initial mistakes than Flash 2.5*, their usual baseline model.
    - A community member shared that they created GGUF-format weights of Qwen3-30B-A3B-Thinking-2507 and made them available for download at [Hugging Face](https://huggingface.co/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF), facilitating local quantized inference and broader compatibility with tools like llama.cpp.
- [**Qwen3-30b-a3b-thinking-2507 This is insane performance**](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507) ([Score: 341, Comments: 85](https://www.reddit.com/r/LocalLLaMA/comments/1md8slx/qwen330ba3bthinking2507_this_is_insane_performance/)): **The post queries whether the recently-released Qwen3-30B-A3B matches the performance of Qwen3-235B. Commenters note that Qwen3-30B-A3B is achieving comparable results to proprietary LLM APIs (e.g., Gemini 2.5 Flash, Claude Haiku) at significantly lower prices (**`$0.30-.45/million tokens`**), suggesting a major cost-benefit shift. The A3B variant reportedly achieves** `5-10 tokens/second` **(quantized) on cheap laptops without GPUs, and documentation recommends extremely long output lengths (**`32,768` **to** `81,920` **tokens) for complex tasks, challenging recent literature that suggests shorter 'thinking' may be optimal.** Discussion highlights skepticism about profit margins of major providers and interest in Qwen3-30B-A3B's feasibility as a local inference model with high performance and low cost, especially at the edge (laptop/CPU-level). There is technical debate contrasting recent papers advocating shorter outputs versus Qwen3's suggestion of maximizing output length for complex benchmarks.
    - Several commenters highlight that the Qwen3-30B-A3B-Thinking-2507 model rivals proprietary LLMs (like Gemini 2.5 Flash/o3 mini/Claude Haiku) at a dramatically lower cost—$0.30-0.45/million tokens, compared to 5-10x higher prices for closed offerings, even though the performance is similar.
    - The model's efficiency is notable: with quantization, users report achieving 5-10 tokens/second generation rates on commodity laptops without GPUs, making large-scale deployment and local inference practical for non-enterprise users.
    - Technical advice references recent studies on output length, recommending `32,768` tokens for most queries and up to `81,920` for complex math or programming benchmarks, with longer outputs specifically improving performance on challenging tasks.
- [**Kudos to Qwen 3 team!**](https://www.reddit.com/r/LocalLLaMA/comments/1md00oc/kudos_to_qwen_3_team/) ([Score: 125, Comments: 17](https://www.reddit.com/r/LocalLLaMA/comments/1md00oc/kudos_to_qwen_3_team/)): **The post discusses the release of Qwen3-30B-A3B-Instruct-2507 by Alibaba, praising its quality but noting that the older Qwen3-32B currently outperforms it on standardized benchmarks. The OP requests the release of additional variants such as Qwen3-32B Instruct/Thinking and Qwen3-30B-A3B-Thinking-2507, suggesting interest in broader model availability and further benchmark comparisons. Commenters also express demand for high-performing coder models (14B and 30B) and seek clarification on the technical performance gap between the new and the older (Qwen3-32B) versions.** Technical discussion centers around demand for more coder-oriented and competitive models, specifically those able to outperform established baselines like Sonnet 3.5, and queries regarding direct benchmarking between Qwen3-30B-A3B-Instruct-2507 and Qwen3-32B.
    - Discussion highlights that Qwen-32B generally outperforms Qwen-30B in benchmarks, but this comes at the cost of significantly slower inference speeds and higher hardware requirements—specifically, Qwen-32B typically needs a high-memory GPU, whereas Qwen-30B is more feasible on lower-tier hardware including some CPUs. This delineation underscores model selection trade-offs between performance and resource accessibility.
    - Several users express interest in a Qwen3 coder model in the 14B parameter range that could potentially outperform Sonnet 3.5, reflecting current interest in both high-capacity and specialized coder models, along with comparative ambitions in the open-source model landscape.
    - A point of technical frustration is raised regarding the absence of live inference (API) endpoints for new Qwen versions on HuggingFace, impacting immediate accessibility for testing and deployment workflows.
- [**After 6 months of fiddling with local AI. Here’s my curated models list that work for 90% of my needs. What’s yours?**](https://i.redd.it/jzljyi4tw2gf1.jpeg) ([Score: 104, Comments: 65](https://www.reddit.com/r/LocalLLaMA/comments/1mdjb67/after_6_months_of_fiddling_with_local_ai_heres_my/)): **The user shares a detailed summary of their local AI model setup, specifically using multiple quantized LLMs (mostly Unsloth UD Q4_K_XL) such as Mistral-24B for QA, Gemma3-27B (IQ3 quantization) for general tasks, Qwen3-30B for coding, and Medgemma for medical inquiries, all running on Llama.cpp with 48GB shared RAM and a Vulkan backend. They report practical inference performance (e.g., Mistral-24B at 4t/s, Gemma3-27B at 5t/s, and Qwen3-30B at 20-22t/s) at 10-12k context lengths, and describe their division of labor for these models (e.g., using Gemma3n-2B as an AI-enhanced Siri, Granite3.3-2B for summaries).** Technical comments discuss the benefits of using Gemma3-27B with IT QAT quantization for notably better performance and critique Medgemma's reliability for reading chest X-rays—while good for general advice, it may miss subtle/critical details. Another comment irrelevantly lists historical countries as favorites.
    - Gemma3-27B IT QAT provides a notable performance improvement over standard quantized models, with users reporting it feels similar in quality to a Q5KM quantization, particularly if the user's hardware can support it; this highlights efficiency and inference speed advancements in quantization-aware training.
    - MedGEMMA's domain-specific performance, particularly in reading chest X-rays, is questioned as it tends to overlook subtle but clinically significant features, underlining the challenges of LLMs in medical imaging tasks and the need for thorough evaluation before clinical use. However, it is also lauded for being a privacy-preserving, locally runnable LLM for general health conversation, matching a general practitioner in some attributes for private, offline healthcare advice.
    - GLM 4.5 Air's release has led to significant model consolidation among users, with one citing a model library reduction from 1.2TB to 411GB, suggesting GLM 4.5 Air's capability and comprehensiveness may replace the need for multiple smaller models on local systems.

### 2. GLM4.5 Model Launches, Benchmarks, and User Impressions

- [**GLM4.5 EQ-Bench and Creative Write**](https://i.redd.it/ubwsl0gdb0gf1.jpeg) ([Score: 133, Comments: 24](https://www.reddit.com/r/LocalLLaMA/comments/1md5k8f/glm45_eqbench_and_creative_write/)): **The post shares a leaderboard graphic for GLM4.5's performance on EQ-Bench and Creative Writing evaluations, with a focus on comparisons among open and closed models (e.g., QWEN, DeepSeek, OAI's Sonnet, Kimi 2) for creative writing. The image ranks various LLMs, highlighting open-weight models topping the benchmark. Critiques in the comments note that LM-as-judge benchmarks are increasingly seen as outdated, and emphasize that current evaluations often fail to account for long-context creative writing, where OAI and Google models are cited as more consistent due to their effective handling of long context windows.** Commenters debate the real applicability of these rankings, pointing out that benchmark results (especially for creative writing) often misrepresent actual user experience—e.g., issues like narrative consistency and context retention over longer texts. There is skepticism about certain models' high rankings based on first-hand usage, with observations that some highly-ranked models struggle with tense consistency and prompt adherence, while some less-recognized models (e.g., Mistral Nemo) perform unexpectedly well in practical use cases.
    - Several users note that many current creative writing benchmarks, such as EQ-Bench, are potentially outdated due to their reliance on LMs as automated judges—a methodology similar to LMSYS's AutoArena—which may not robustly reflect creative or contextual capabilities of LLMs.
    - Critique is raised regarding benchmark evaluations of story-writing: while QWEN, GLM, and DeepSeek perform well for short, character-based tasks, users report all become repetitive or lose track of narrative structure well before reaching their full context window (often less than 1/10th of advertised lengths). In contrast, models from OAI and Google (e.g., with 1M context claims) are credited with improved performance over extended contexts (staying coherent at up to 100K tokens).
    - Model-specific feedback highlights that Kimi 2 does not deliver significant improvements over smaller models like Mistral Small 3.2, with criticisms focused on prose quality despite better prompt adherence. Additionally, QWQ is noted to struggle with tense consistency, particularly in first person narratives, and experienced users still rate Mistral Nemo highly for overall narrative quality, but flag its occasional inconsistencies with character details.
- [**glm-4.5-Air appreciation poist - if you have not done so already, give this model a try**](https://www.reddit.com/r/LocalLLaMA/comments/1mdhfhs/glm45air_appreciation_poist_if_you_have_not_done/) ([Score: 127, Comments: 54](https://www.reddit.com/r/LocalLLaMA/comments/1mdhfhs/glm45air_appreciation_poist_if_you_have_not_done/)): **The post highlights glm-4.5-Air (4-bit quantized, mlx implementation) as a high-performing LLM that excels in multistep tool chaining and project management integration, proving both fast and contextually robust in complex workflows. The user notes glm-4.5-Air's ability to provide in-depth analysis without losing context, outperforming Qwen 32B in daily assistant tasks, particularly on supported hardware (notably Mac with mlx backend).** Top comments focus on lack of support in llama.cpp (pending [PR #14939](https://github.com/ggml-org/llama.cpp/pull/14939)), highlighting current platform limitations (notably for dual 3090 users), optimism about future releases with GGUF compatibility, and mention the performance trade-offs between quantization levels (3-bit vs 4-bit) relative to parameter count and model quality.
    - The lack of llama.cpp and GGUF support for the GLM architecture currently limits usage for non-Mac hardware, particularly those running dual 3090 GPUs, but active pull requests (see https://github.com/ggml-org/llama.cpp/pull/14939) suggest imminent compatibility upgrades.
    - Multiple users report successful runs of GLM-4.5-Air at aggressive quantization levels (such as 3-bit and 4-bit), indicating that the model's large parameter count compensates for quantization-induced quality loss, with some noting comparable or better performance to recent Qwen3-30B quant models.
    - There's active discussion and concern regarding performance on non-Mac hardware, especially systems using DDR4 RAM (e.g., i7-13700k with 64GB DDR4), due to potential memory speed bottlenecks that may negatively impact inference speed compared to Mac's higher bandwidth memory.

### 3. Meta Superintelligence Strategy and Community Reactions

- [**Bye bye, Meta AI, it was good while it lasted.**](https://www.reddit.com/r/LocalLLaMA/comments/1md6t2h/bye_bye_meta_ai_it_was_good_while_it_lasted/) ([Score: 1098, Comments: 366](https://www.reddit.com/r/LocalLLaMA/comments/1md6t2h/bye_bye_meta_ai_it_was_good_while_it_lasted/)): **Meta CEO Mark Zuckerberg released a statement outlining Meta's future direction on AI superintelligence, noting that concerns about safety will inform stricter control over open-sourcing their most advanced models (see the official statement at [meta.com/superintelligence](https://www.meta.com/superintelligence/)). This signals a shift from Meta's prior approach (e.g., releasing Llama derivations under permissive licenses) to a more proprietary model as capabilities advance. The decision contrasts with ongoing releases of open-weight, frontier-level models by other organizations, suggesting Meta may not contribute their top advances to the open community going forward.** Commenters broadly critique Meta's reversal, pointing out prior public advocacy for open source AI and speculating the shift is motivated by monetization rather than safety. Some note there are already open-weight models rivaling or surpassing Llama 4, so Meta's withdrawal may have limited impact on open frontier progress.
    - One comment highlights that there are already several open-weights, frontier-level models considered stronger than Llama 4, implying Meta's relative lag in the open-source AI race and questioning their stated reasoning for changes in open model deployment.
    - Another comment references earlier advocacy by Meta leadership (e.g., letters to Congress) promoting open-source AI as being safer and better for society, pointing out the apparent shift toward monetization and reduced transparency, which may undermine previously stated commitments to openness.
- [**Chinese models pulling away**](https://i.redd.it/727keqreo3gf1.png) ([Score: 291, Comments: 35](https://www.reddit.com/r/LocalLLaMA/comments/1mdmsu9/chinese_models_pulling_away/)): **The image (https://i.redd.it/727keqreo3gf1.png) is likely a chart or comparison metric indicating the rapid progress and performance improvements of Chinese open-source language models, such as those from Qwen, over Western counterparts like Llama and Mistral. The discussion highlights users migrating from Meta's Llama and Mistral models to more recent, less-censored and higher-performing Chinese models (notably Qwen3-30B-A3B). Commenters note Mistral's continued development but recognize the growing dominance of Chinese models in certain benchmarks and applications.** Several commenters debate the future adoption of Western vs. Chinese LLMs, with some expressing nostalgia for the earlier dominance of Llama and ongoing support for Mistral, while acknowledging the technical appeal and rapid progress of Chinese alternatives.
    - Users are sharing a progression in open source LLM adoption: starting with LLaMA 3.1-3.2, moving to Mistral 3 Small, then to a distilled R1-Mistral variant (with reduced censorship such as Dolphin), and now to Qwen3-30B-A3B, indicating how Chinese models like Qwen are attracting advanced users due to ongoing technical improvements and model openness.
    - Mistral models are highlighted for their consistently strong performance, with emphasis on recent multiple releases of small models this month. There's technical anticipation around the release of a new Mistral large model expected later in the year, signaling active and sustained development from Mistral.
    - For practical coding use cases, particularly frontend coding, it is noted that Mistral models perform effectively, as evidenced by external resources demonstrating their capabilities (e.g., [designarena.ai](http://designarena.ai/)). This suggests Mistral maintains competitive performance in applied coding tasks despite competition from newer entrants like Qwen.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. OpenAI GPT-5 Anticipation and Evidence

- [**More snippets of GPT 5, seems like release really is imminent.**](https://i.redd.it/fbo023zrzzff1.jpeg) ([Score: 390, Comments: 69](https://www.reddit.com/r/singularity/comments/1md46vv/more_snippets_of_gpt_5_seems_like_release_really/)): **The post discusses new snippets allegedly from GPT-5 and implies its public release may be imminent, referencing the lack of a 'gpt-5-mini' model in current leaks, which contrasts with previous OpenAI release patterns. The image is likely a screenshot or informational snippet further supporting speculation around the release timing and model lineup segmentation.** Technical discussion centers around rollout strategy: some users speculate OpenAI will release the largest, most headline-grabbing models first, followed by smaller 'mini' models and potentially open-source versions later, diverging from previous behavior.
    - One user noted the absence of a 'gpt-5-mini' model, despite prior leaks suggesting its existence, hypothesizing that OpenAI may prioritize releasing high-profile, powerful models first to maximize headlines before potentially unveiling smaller or open-source variants later.
- [**Cryptic post from the ChatGPT Twitter account… GPT-5 tomorrow?**](https://i.redd.it/d8z6h7u743gf1.jpeg) ([Score: 192, Comments: 77](https://www.reddit.com/r/singularity/comments/1mdjzgd/cryptic_post_from_the_chatgpt_twitter_account/)): **The image in question, posted by the ChatGPT Twitter account, displays "ChatGPT" written in Japanese katakana, with a red kanji symbol meaning 'fortune,' 'good luck,' or 'blessing.' This has led to speculation in the community about a potential announcement, such as GPT-5, particularly since the post notes that tomorrow is Thursday—often a day for tech releases. The technical content of the image is limited, serving mainly as a teaser without explicit technical details or announcements.** A comment clarifies the Japanese text, correcting initial speculations of a hidden hint, and points out the usage of katakana and kanji, grounding the discussion in linguistic rather than technical novelty. Some speculate on the release of 'GPT-5 or even GPT-6,' but this is speculative and not based on any evidence from the image itself.
    - One commenter points out that the Japanese sumi-e brushstroke in the visual features associated with the announcement could symbolically imply *"maturity, depth, and deliberateness"* in the upcoming model, possibly alluding to improvements in reasoning or interpretive nuance compared to prior releases.
- [**"gpt-5-auto" and "gpt-5-reasoning" have been added as models to the ChatGPT MacOS app in a recent update**](https://i.redd.it/xk0egrlaxzff1.png) ([Score: 289, Comments: 39](https://www.reddit.com/r/OpenAI/comments/1md3xoo/gpt5auto_and_gpt5reasoning_have_been_added_as/)): **The screenshot shows the ChatGPT MacOS app's model selection menu featuring two new models: 'gpt-5-auto' and 'gpt-5-reasoning'. This suggests that OpenAI is preparing to roll out or test these GPT-5 variants with different operational focuses—'auto' likely for general-purpose use and 'reasoning' potentially optimized for more complex logical tasks. No benchmarks or official release notes are specified; visual evidence only confirms their presence in the UI.** A key user sentiment stresses the importance of manual model selection, expressing appreciation for the ability to opt between modes (like router/auto and explicit one-model choices). No deep technical debate is present, as most comments are excited anticipation rather than critical analysis.
    - One commenter draws a parallel between the new "gpt-5-auto"/"gpt-5-reasoning" model distinction and xAI's approach, where users can select between "fast/auto" and "advanced" model options, and notes that the continued presence of a separate "reasoning" model suggests "auto" isn't a fully unified, all-capabilities model. This implies possible internal specialization within the GPT-5 lineup (e.g., "auto" for speed/routing, "reasoning" for complex tasks) rather than a single monolithic model for all purposes, similar to strategies seen in other AI providers.
- [**GPT 5 spotted !! It's near !**](https://i.redd.it/m7o4ma8zk0gf1.jpeg) ([Score: 150, Comments: 17](https://www.reddit.com/r/OpenAI/comments/1md6qsd/gpt_5_spotted_its_near/)): **The post, titled 'GPT 5 spotted !! It's near !', likely references a rumor or speculative sighting of GPT-5's potential release, but the image content cannot be analyzed for technical detail. The discussion centers on anticipation regarding the release date of GPT-5 (either imminent or later in August) and hopes it will launch alongside a new reasoning model, reflecting skepticism about whether GPT-5 alone would outperform OpenAI's 'o3.'** Commenters debate the likely release timeline and express both eagerness and anxiety about GPT-5's job market implications, echoing ongoing industry concerns about advanced language models disrupting employment.
    - A user expresses skepticism that GPT-5, at launch, would clearly outperform OpenAI's own "o3" model, highlighting current uncertainty around performance differences between state-of-the-art and next-gen releases. There is also anticipation for the rumored "reasoning model" and speculation that releasing it together with GPT-5 could significantly boost capability beyond conventional LLMs.
- [**Somethings going on**](https://i.redd.it/qilyyd52d0gf1.jpeg) ([Score: 147, Comments: 63](https://www.reddit.com/r/singularity/comments/1md5q6z/somethings_going_on/)): **The image appears to depict a screenshot of ChatGPT with the 'gpt-8-auto' model selected, suggesting the presence of a future, unreleased model variant (potentially GPT-5 or beyond). Commenters note that entering arbitrary model names in the URL can produce similar UI results, implying this is not a real model but rather a UI artifact or placeholder accessible via direct URL manipulation (e.g., https://chatgpt.com/?model=gpt-8-auto). The context is heightened by speculation regarding an imminent GPT-5 announcement or release, supported by recent leaks and high community anticipation.** Several users debate the legitimacy of the screenshot, with some attributing it to URL manipulation rather than an actual model leak, and others speculating about potential release timelines based on industry rumor and recent activity.
    - Some users are noting that manipulating the model query string in ChatGPT’s web interface (e.g., https://chatgpt.com/?model=gpt-8-auto or ?model=gpt-5-auto) merely changes the displayed model name but does not load any unreleased models; actual requests are still routed to established models like GPT-3.5. This is evident from the unchanged functionality regardless of the model name in the query and observation of no backend model differences.
    - There is technical speculation based on recent leaks and industry chatter that a new GPT model (possibly GPT-5) could be released imminently, supported by references to model names like 'LMarena Anon' in leaks and the timing of previous model rollouts. However, there are no official release confirmations or infrastructure changes detected such as livestream or rollout announcements, adding skepticism to the claims.
    - Direct examination and screenshot share show that, despite any query string manipulation, backend enforcement is intact and prevents unauthorized access to unreleased GPT models via the UI, confirming expected security around OpenAI’s model endpoints.

### 2. WAN 2.2 Animation Model Release and Community Tools

- [**WAN 2.2 is going to change everything for indie animation**](https://v.redd.it/trtpftp1hzff1) ([Score: 415, Comments: 94](https://www.reddit.com/r/StableDiffusion/comments/1md2d20/wan_22_is_going_to_change_everything_for_indie/)): **WAN 2.2, an update to the indie animation model, is noted for visual improvements over WAN 2.1, but users are finding limitations in prompt adherence, particularly with action and camera motion control (e.g., difficulty generating stationary subjects while animating camera movement). The usability of WAN 2.2 in production contexts is debated, as its output may present risks to developers' reputations due to unresolved issues.** Comments highlight a consensus that while WAN 2.2 presents aesthetic progress, its persistent issues with prompt interpretation and lack of reliability make it unsuitable for professional or developer use, sparking concerns about negative community reception.
    - Technical users are discussing the actual improvements in WAN 2.2 over 2.1, characterizing them as modest visual upgrades rather than groundbreaking changes, challenging the claim of it “changing everything.”
    - A practical prompt engineering issue is highlighted: even with explicit negative prompts ("walking" and "talking"), characters in WAN 2.2 persistently perform unwanted actions, indicating limitations in prompt fidelity and model control for animation tasks.
    - Some note that despite visual improvements, the model remains largely unusable for production by developers due to persistent issues (output consistency, control mechanisms), and deployment in projects could risk reputational harm rather than provide substantive benefit.
- [**I created a detailed Prompt Builder for WAN 2.2, completely free to use.**](https://i.redd.it/b9mfy32dxyff1.png) ([Score: 364, Comments: 30](https://www.reddit.com/r/StableDiffusion/comments/1md0qed/i_created_a_detailed_prompt_builder_for_wan_22/)): **The image appears to showcase the UI of a new, free, browser-based prompt builder tailored for WAN 2.2 (likely a Stable Diffusion model variation), accessible at dengeai.com/prompt-generator. The interface visually represents prompt components, supporting detailed video prompt creation, and aims to enhance usability with intuitive visual cues, as noted by commenters. Technical users inquire about potential for local deployment, suggesting demand for an open-source or downloadable version that could extend beyond current model specificity.** A productive debate centers on the benefits and feasibility of offering the tool as a local or universal application, with community interest in collaborative development to broaden its utility beyond just the WAN model.
    - A commenter inquires about the possibility of releasing the Prompt Builder as a local, standalone software solution, highlighting community interest in having a universal tool not limited to WAN 2.2. They suggest collaborative development to generalize the tool's applicability to other models or workflows.
- [**Wan 2.2 I2V game characters with SeerV2**](https://v.redd.it/kape9d80xzff1) ([Score: 288, Comments: 55](https://www.reddit.com/r/StableDiffusion/comments/1md3zfe/wan_22_i2v_game_characters_with_seerv2/)): **The post details an Image-to-Video (I2V) workflow using Wan 2.2 and SeerV2, emphasizing the effectiveness of Wan 2.2 for game character renders. The workflow is implemented in ComfyUI, with LightXv2 at 0.5 strength, 20 steps, CTF 3.5, ModelSamplingSD3 at 5, sampler set to dpmpp_2m, and scheduler as Beta. Final output is rendered at 832x512 and upscaled to 4K using SeedVR (with reference guide: [one-step-4k-video-upscaling-and-beyond-for-free-in-comfyui-with-seedvr2](https://www.ainvfx.com/blog/one-step-4k-video-upscaling-and-beyond-for-free-in-comfyui-with-seedvr2/)).** Commenters highlight the impressive quality of Wan 2.2, with one noting it as 'the best one yet,' but no significant technical debate is present in the discussion.
    - A user detailed their workflow for SeerV2 I2V game character generation using ComfyUI, highlighting use of Wan 2.2 and LightXv2 at 0.5 strength, 20 steps with CTF 3.5, ModelSamplingSD3 set to 5, and dpmpp_2m sampler with Beta scheduler. They rendered at 832x512 and used SeedVR for upscaling, referencing a guide for 4K upscaling in ComfyUI (https://www.ainvfx.com/blog/one-step-4k-video-upscaling-and-beyond-for-free-in-comfyui-with-seedvr2/). The workflow involves specific negative prompts ("Slow, Slowed in negative") to adjust output character motion.
- [**Pleasantly surprised with Wan2.2 Text-To-Image quality (WF in comments)**](https://www.reddit.com/gallery/1md4u30) ([Score: 224, Comments: 94](https://www.reddit.com/r/StableDiffusion/comments/1md4u30/pleasantly_surprised_with_wan22_texttoimage/)): **The post discusses results from the Wan2.2 text-to-image model (14B parameters, FP16), with linked workflows and sample outputs. Users highlight strong texture and skin realism, citing prompt adherence as moderate but improved when combined with Flux (e.g., by using Nunchaku to generate a latent, then reprocessing in WAN2.2, which cuts generation time by approximately 40%). The community seeks improved control tools (e.g., tile/blur controlnet) for more flexible output manipulation.** There is consensus that WAN2.2 surpasses Flux in photorealism and texture quality, though Flux is considered superior in fine prompt control; practical blending of the two yields compelling results and improved efficiency.
    - Users report that Wan2.2 demonstrates strong realism, especially in generating detailed textures such as skin, outperforming models like Flux Dev in this regard. One user notes using the 14B FP16 version for best results, with a recommendation to view uncompressed samples to appreciate texture fidelity, and suggests that adding tile/blur controlnet support would further enhance the model's versatility.
    - A workflow optimization is described: users replace the High Noise pass with Flux Dev (via Nunchaku for the half-point latent), then re-encode into ksampler for a "WAN finish." This hybrid pipeline reportedly maintains WAN 2.2's output quality while reducing image generation time by approximately 40%.
    - Backward compatibility for LoRAs from WAN 2.1 is confirmed, enabling users to reuse their existing assets. Additionally, mention is made of potential for future video rendering capabilities, indicating ongoing extensions to functionality.
- [**All in one WAN 2.2 model merges: 4-steps, 1 CFG, 1 model speeeeed (both T2V and I2V)**](https://huggingface.co/Phr00t/WAN2.2-14B-Rapid-AllInOne) ([Score: 199, Comments: 108](https://www.reddit.com/r/StableDiffusion/comments/1mddzji/all_in_one_wan_22_model_merges_4steps_1_cfg_1/)): **The post details a customized WAN 2.2 model merge for text-to-video (T2V) and image-to-video (I2V) that consolidates the 'high' and 'low' WAN 2.2 models (as first/middle blocks) and WAN 2.1 output blocks into a single, streamlined architecture. The model integrates VAE and CLIP for simplified deployment, incorporates Lightx2v and PUSA LoRAs for distillation and supports 4-step, 1 classifier-free guidance (CFG) sampling, while maintaining WAN 2.1 LoRA compatibility. The approach recommends using sa_solver with a beta scheduler and supports native checkpoint loading, emphasizing speed/simplicity with potential for tradeoff against running larger, separate models.** A top commenter requested quantitative comparisons against baseline models and asked if anyone would quantize ("quant") the merge, indicating interest in empirical benchmarks and possible further speed/efficiency gains. Suggestions also touch on missed titling opportunities but no technical content there.
    - Users express interest in quantized versions of the merged WAN 2.2 model for greater efficiency. There is a call for benchmarks comparing speed and output quality between the original two-model/accelerated pipelines and the merged model, using the same seeds for direct comparison.
    - Technical concerns are raised about potential quality degradation when merging steps and models: a user specifically asks if there is a drop in quality compared to running the full 20 steps and separate models, indicating an interest in performance trade-offs.
    - A user notes that on a 12GB GPU, the all-in-one model delivers strong performance, especially for T2V tasks, but comments that output diversity appears reduced—outputs are less varied and more reminiscent of WAN 2.1, rather than showcasing the wider range of WAN 2.2. Preference is expressed for running the original models with lightx2v at 1.0 for better diversity, despite longer runtimes.
- [**Wan 2.2 I2V is really amazing! so far**](https://v.redd.it/z0r6axrpszff1) ([Score: 191, Comments: 34](https://www.reddit.com/r/StableDiffusion/comments/1md3ggp/wan_22_i2v_is_really_amazing_so_far/)): **The post discusses the performance of the Wan 2.2 I2V (image-to-video) model, highlighting its remarkable capability for animating images into video with notable motion and detail, particularly when using the full Wan 2.2 f16 32GB model via the kijai wrapper workflow. Comparisons suggest that the full-size model outputs significantly more detailed and dynamic results than scaled-down variants, underscoring the importance of using the full model for high-fidelity video synthesis.** Commenters note the surprising lack of appreciation for such advanced generative video technology in the user community and speculate on compelling use cases, such as integrating these models into indie video game cutscenes to reduce production time and resource requirements.
    - A user compared the output of Wan 2.2 using the Kijai wrapper workflow with the full WAN 2.2 F16 32GB model, stating that using the full-size model results in *more motion and details in the generated videos* compared to scaled-down versions. This suggests a technical trade-off between model size and detail fidelity in output.
    - Performance and hardware are points of interest, with questions around *generation times and hardware requirements* for running WAN 2.2 at full resolution. The discussion indicates that hardware specs and model size (32GB for F16) heavily influence workflow feasibility and results quality.
- [**Wan 2.2 i2v Continous motion try**](https://v.redd.it/o978gmrhizff1) ([Score: 130, Comments: 44](https://www.reddit.com/r/StableDiffusion/comments/1md2jzi/wan_22_i2v_continous_motion_try/)): **OP experimented with WAN 2.2 for image-to-video (i2v) generation by iteratively extrapolating video from a static image (WAN t2i) and then chaining segment continuations using the last frame of each as the seed. Technical discussion centers on handling degradation (e.g., motion blur, lighting, inconsistent features) and potential mitigations. Workflow was 720p@16fps, upscaled/interpolated to 1080p@60fps; key issue identified is frame selection due to motion blur hindering continuity.** Commenters note the WAN 2.2 model appears to maintain quality across chained iterations, with little visual degradation, which may suggest robustness of the model. There is technical debate on possibly applying image-to-image (i2i) detail enhancement to the last frame before feeding it back, but WAN 2.2's resilience might render this unnecessary.
    - A commenter observes that, contrary to expected video quality degradation common in iterative frame extension workflows, Wan 2.2 maintains high quality through continuous frame generation—both the starting and final frames exhibit little to no visible degradation, suggesting robust video frame consistency with this model version.
    - There's technical speculation about using an intermediate image-to-image (I2I) pass to enhance detail in the last frame before recycling it for further generations. However, given Wan 2.2's output stability, this extra step may now be unnecessary, potentially streamlining continuous video synthesis with fewer manual quality interventions.
    - One user notes that despite some stitch or transition glitches, these are minor enough to be overlooked when focusing on story content, indicating that visual artifacts from frame chaining with Wan 2.2 are at an acceptable level for immersive applications.
- [**It’s definitely at the point where you can mistake it for real at first glance.**](https://v.redd.it/ujbvunncx2gf1) ([Score: 607, Comments: 64](https://www.reddit.com/r/ChatGPT/comments/1mdj19u/its_definitely_at_the_point_where_you_can_mistake/)): **The post discusses recent advancements in AI-generated video, highlighting that clips created with minimal prompts ("barely a couple of sentences each") are now visually realistic enough to be mistaken for genuine footage at first glance. The content underscores significant progress in generative model realism, likely referencing improvements in models like Sora or similar text-to-video syntheses.** Top comments reflect positive reception to the realism and entertainment value of the AI output, but do not contribute technical discussion or critique.
    - AndrewH73333 highlights a limitation in the current AI model's video generation capabilities: the AI struggles with generating plausible outcomes for less-documented events (e.g., 'what happens after a piano is under water') due to insufficient training data in those domains. This points to a broader challenge in generative models—synthesizing realistic outputs for scenarios with sparse or no examples in the training dataset, suggesting a dependency on comprehensive, diverse data to improve fidelity in edge-case or uncommon situations.
    - eOMG notes a characteristic of current AI-generated speech: while it is highly advanced and often realistic at first glance, attentive listeners can still identify subtle artifacts or unnaturalness that distinguishes it from genuine human speech. This observation is indicative of persisting minor deficiencies in text-to-speech or multimodal generative models, which may be attributed to limitations in prosody, intonation, or contextual awareness.

### 3. Anthropic Claude Feature Expansion and Community Usage

- [**Claude can now text, email, and schedule for you on mobile**](https://v.redd.it/ash0cvh572gf1) ([Score: 214, Comments: 45](https://www.reddit.com/r/ClaudeAI/comments/1mdf9ns/claude_can_now_text_email_and_schedule_for_you_on/)): **Anthropic's Claude AI mobile app now supports drafting emails, texts, and meeting invites, which can be sent via users' native apps with a single tap ([announcement](http://claude.ai/download)). Technical implementation, especially for Android, likely leverages the standard Android OS intent system to compose drafts by pre-filling subject and body fields, leaving recipients unspecified for privacy. This workflow minimizes app permissions and user data exposure, aligning with privacy-conscious design.** Top comments raise concerns about rate limiting on high-tier plans ('Max x5 plan'), and express preference for the indirect approach that avoids direct address book access. Users note this obviates the need for DIY automation projects using communication platforms like Signal.
    - A commenter speculates that Claude's Android implementation likely uses the native Android OS intent system to initiate email drafts, passing only the subject and body but leaving the recipient field for the user, which prioritizes user privacy by not granting address book access.
    - A technical limitation discussed is Claude's conversation length cap, which can disrupt workflows; users request features like early warnings, context summarization or rolling conversations forward, similar to what GPT offers, to manage longer threads and context retention more efficiently.
    - Comparative feedback from technical users points out that the GPT Desktop experience at the $20 tier is preferred over Claude Desktop at the higher-priced plan (20x), citing better usability for long-form dialog and integration, and requesting that Claude Code gain native Gmail and Calendar integration without requiring additional permission portals or middleware.
- [**Claudeholism: The New Addiction of the Century 😞**](https://www.reddit.com/r/ClaudeAI/comments/1mdc09s/claudeholism_the_new_addiction_of_the_century/) ([Score: 132, Comments: 58](https://www.reddit.com/r/ClaudeAI/comments/1mdc09s/claudeholism_the_new_addiction_of_the_century/)): **The post reports on strong user dependence on Claude Code (by Anthropic), citing its superior productivity and code generation compared to alternatives such as GitHub Copilot, Google Gemini, JetBrains AI Assistant, and local StarCoder 2 models. Users cite Claude Code's effectiveness in generating working code (e.g., Pulumi scripts, Clean Architecture) and fluent context-aware code commentary, whereas alternatives suffer from hallucination, poor code understanding, lack of domain alignment, or high hardware requirements.** Commenters agree that Claude Code provides dramatically increased productivity (rapid prototyping, migration, and tooling) and an addictive workflow; they note that its contextual integration (including access to local files) leaves competing LLMs less useful, even accounting for occasional hallucinations. Some note the behavioral shift where frictionless LLM-assisted development changes usage patterns and expectations.
    - One user details rapid prototyping enabled by Claude, migrating a forum from a poor platform to a custom Nuxt + Supabase setup in hours, and building specialized tools for consultancy quickly, emphasizing the efficiency gains and reduced barriers compared to traditional workflows.
    - Another highlights how integrating LLMs with local computer access drastically streamlines development by reducing friction, despite occasional hallucinations. The fluidity and productivity benefits are described as outweighing those of other solutions currently available.
    - Discussion about Opus subscription tiers reveals that downgrading from a $200 to a $100 plan (Pro) imposed significant usage limits, making the higher tier necessary for heavy, creative Opus use. Users note that demand surges from "vibe coders" can further restrict access, resulting in approximately 70% fulfillment of their workflow needs on the lower tier.
- [**What y'll are building that is maxing out Claude Code**](https://www.reddit.com/r/ClaudeAI/comments/1md0etv/what_yll_are_building_that_is_maxing_out_claude/) ([Score: 105, Comments: 110](https://www.reddit.com/r/ClaudeAI/comments/1md0etv/what_yll_are_building_that_is_maxing_out_claude/)): **The OP, a senior engineer with deep backend, full stack, and ML experience, questions what types of projects are consuming the entire context window of Claude Code (Sonnet and Opus models), given their background in building complex, maintainable systems. They note that historically, code iteration is a team process and that platform complexity (Supabase, PlanetScale, etc.) is just an extra layer compared to "rolling your own" systems. They seek concrete examples of workflows or codebases that truly stretch Claude Code's limits, specifically in a single bounded coding session.** Commenters note that heavy usage is often about maximizing context window rather than project size—often involving elaborate onboarding of the model (full file context, related classes, docs, and linting standards per prompt). This leads to rapid token depletion, with one user reporting maxing out Opus in 5-6 prompts due to these practices. Discussion also suggests that much of this usage may not represent productive work, implying some users are generating "junk" code or abusing the available quota.
    - The main technical bottleneck with Claude Code appears to be its context window: Unlike human developers with project familiarity, each new Claude session is like onboarding a new dev, requiring the full relevant files, classes, abstraction layers, implementation examples, and documentation on every run. This onboarding overhead, combined with strict coding standards and post-generation linting/fixing, consumes tokens rapidly—users report maxing out the $100 Opus plan in as little as 5-6 prompts over a few hours and sometimes receiving token usage warnings after a single prompt.
    - Some users approach the session as a bulk onboarding for meaningful development, loading large codebases and reference materials in order to get precise, standards-compliant code and automated linting. This workflow is token-expensive but necessary for deep bugfixing or large-scale refactoring, and highlights a current limitation of the context/token architecture when dealing with complex projects.
- [**Claude is now on X (@claudeai)**](https://i.redd.it/5q4o2j4x23gf1.png) ([Score: 210, Comments: 51](https://www.reddit.com/r/ClaudeAI/comments/1mdjt9r/claude_is_now_on_x_claudeai/)): **The post announces that Claude, Anthropic's AI model, now has an official presence on the social media platform X (formerly Twitter), under the handle @claudeai. The attached image is not described in technical detail, but presumably showcases either the account or a related announcement. Comments focus on confirming the account's authenticity and make light observations, but add no technical debate.** There is curiosity about the authenticity of the account, with a user asking 'is this true?'. No technical discussion about model implementation, capabilities, or integration into X is present.
    - One user detailed a persistent behavior in Claude's responses that can't be suppressed, even when using cust`omization features like [Claude.md](http://claude.md/) at both project and global scope. The user highlights a recurring phrase—"You're absolutely right!"—that remains despite attempts to configure the model's output, suggesting there are deeply embedded, hardcoded response structures or insufficient user-level configurability. This highlights potential limitations in current prompt customization or extensibility for Anthropic's models, as the model's core behavior overrides ongoing user preferences.
- [**Anthropic vs xAI**](https://i.redd.it/vj8g05pmhyff1.png) ([Score: 792, Comments: 47](https://www.reddit.com/r/ClaudeAI/comments/1mczbbb/anthropic_vs_xai/)): **The post is a comparison between Anthropic and xAI, two prominent AI companies, implicitly referencing their business strategies, reputational differences, and alignment with major contracts (such as military). Commenters highlight Anthropic's focus on securing military contracts, while xAI's origins and intentions are debated, particularly in the context of founder motivations and profit incentives. The image itself serves as a visual contrast between the two entities for community discussion, not providing technical benchmarks or model details directly.** Commenters express skepticism about xAI's motives, referencing Elon Musk's prior statements and suggesting profit or catch-up motives; others contrast the companies' ethical branding and financial focus.
    - The discussion briefly references the pursuit of military contracts as a key strategic focus that shapes the AI model landscape, suggesting this is a significant, if often overlooked, factor in large-scale AI development (e.g., Anthropic possibly securing such contracts).
    - There's implicit comparison of business models and profit strategies between major AI players—Anthropic, xAI, and OpenAI—hinting that revenue from contracts (including military and enterprise) is central to their long-term sustainability and competitive positioning.
    - Participants debate the reputational and ethical implications of brand naming, model features, and leaders’ public statements, highlighting ongoing concerns about the alignment between stated safety ethics and actual product directions in leading AI labs (e.g., skepticism about Elon Musk's motives with xAI and OpenAI).

---

# AI Discord Recap

> A summary of Summaries of Summaries by X.ai Grok-3-mini
> 

**Theme 1. New Model Releases and Comparisons**

- [**Qwen3 Battles GPT-4o for AI Supremacy**](https://www.notion.so/swyx/various_discords): Qwen released the **Qwen3-30B** model, claiming it rivals **GPT-4o** in English and Chinese benchmarks, running in full precision with just **33GB RAM** via the [Unsloth GGUF version](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF). Users reported that a quantized version needs only **17GB RAM**, sparking debates on its real-world performance against **OpenAI's GPT-4o**. This model excites engineers for its potential in multilingual tasks, with benchmarks showing it matches or exceeds **GPT-4o** in specific tests.
- [**GPT-5 Rumors Ignite AI Hype Wars**](https://www.notion.so/swyx/openai_discord): Engineers speculate **GPT-5** might launch in early **August**, with hints like "gpt-5-auto" found in ChatGPT Mac app cache files on [X](https://xcancel.com/apples_jimmy/status/1950514936444305534?s=46&t=fRVjULzONZQAlwHruKTgQg). Discussions compare it to rivals like **Gemini 2.5 Pro**, noting potential integrations in **Microsoft Copilot** and impacts on model rankings. Users debate whether **GPT-5** will dominate, especially in reasoning tasks, based on leaked architecture details.
- [**GLM 4.5 Air Clones Gemini's Tricks**](https://www.notion.so/swyx/unsloth_ai_discord): Engineers tested **GLM 4.5 Air**, finding it mimics **Gemini's** behavior in chatting and poetry analysis, as detailed in a [blog post](https://z.ai/blog/glm-4.5). Tool use broke in **vllm** but worked for document search, drawing comparisons to **GPT-4o**. This model highlights **Alibaba's** push for versatile alternatives, with users noting its stability despite quirks.

**Theme 2. API and Integration Hurdles**

- [**APIs Block Shoddy Quantization Shenanigans**](https://www.notion.so/swyx/openrouter_discord): OpenRouter's API lets engineers specify quantization levels to dodge low-precision models like **FP4**, per [provider routing docs](https://openrouter.ai/docs/features/provider-routing#quantization-levels). This blocks unwanted formats, ensuring efficient model routing across platforms. Users praised the control for reducing errors in production setups.
- [**MCP Servers Fumble Context Crises**](https://www.notion.so/swyx/mcp_discord): Engineers debated adding user-context separation layers in cloud-deployed **MCP servers** to prevent data leaks, referencing [issue #1087](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1087). One setup connected via **Cursor** but failed with **Claude Desktop**, highlighting integration flaws. This underscores the need for robust protocols in multi-client environments.
- [**Litellm API Dodges Connection Chaos**](https://www.notion.so/swyx/aider_discord): Users reported **litellm.APIConnectionError** parsing issues but noted functionality remained intact, often tied to improper chunk formatting. Engineers shared fixes involving double-quoted property names in API calls. This error reveals common pitfalls in AI API chains, pushing for better error-handling standards.

**Theme 3. Performance Optimization Tactics**

- [**Quantization Turbocharges Compute Speeds**](https://www.notion.so/swyx/unsloth_ai_discord): Unsloth's quantization not only shrinks memory needs but also cuts bandwidth and boosts compute speed, as seen in tests with **Qwen3-30B**. Engineers noted keeping vision ops like conv layers in **FP32** creates bottlenecks, recommending mixed precision for balance. This approach slashes inference times without sacrificing accuracy.
- [**Gemma 3 Ditches Watermarks Post-Fine-Tune**](https://www.notion.so/swyx/unsloth_ai_discord): After fine-tuning **Gemma 3 4B** with **16k context**, experiments removed watermarks entirely, improving model stability as shown in a [screenshot](https://cdn.discordapp.com/attachments/1179039861576056922/1399849804045221948/Screenshot_2025-07-29_at_2.13.07_PM.jpeg?ex=688bd0b9&is=688a7f39&hm=82cad388163625496b8cb3e6dd62035b51ae1d16ce0015df29ea910e04c4471f&). Engineers debated trade-offs, with one competition announced on [X](https://x.com/UnslothAI/status/1950553466591723640) ending in **7 days**. Fine-tuning enhanced utility for technical tasks, making it a go-to for uncensored applications.
- [**Expert Parallelism Fails Qwen's Hype Test**](https://www.notion.so/swyx/gpu_mode_discord): Engineers found **Expert Parallelism (EP)** underperformed **Tensor Parallelism (TP)** in **Qwen32B** and **Qwen 235B** due to all-reduce overhead, as detailed in [a blog post](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/?utm_source=chatgpt.com#ttir-triton-ir). EP only shone for models with **MLA and DP attention**. This insight guides hardware choices for efficient scaling.

**Theme 4. Data Privacy and Security Debates**

- [**Dot.lol Sells Out User Data Drama**](https://www.notion.so/swyx/lmarena_discord): Engineers warned **dot.lol** might sell user data for profiling and influence, urging identity protection as discussed in community threads. Data collection was deemed *inevitable* by some, but others pushed for stricter measures like EU **GDPR**. This highlights risks in AI platforms, with calls for anonymous usage.
- [**OpenAI's Censorship Crackdown Frustrates Users**](https://www.notion.so/swyx/unsloth_ai_discord): Members slammed **OpenAI's heavy censorship** in **ChatGPT**, sharing experiences of *canned answers* and lectures, per [Unsloth's Llama 4 page](https://docs.unsloth.ai/basics/llama-4-how-to-run-and-fine-tune). Engineers debated uncensoring techniques, noting drawbacks like reduced utility. This censorship debate exposes trade-offs in safe AI deployment.

**Theme 5. Community Education and Events**

- [**Diffusion Models Study Group Kicks Off MIT Deep Dive**](https://www.notion.so/swyx/mlops_chipro_discord): AI Scholars launched a **12-person, 5-month** study group on diffusion models using [MIT’s curriculum](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf), featuring free sessions on **Aug 2** and **Aug 9** via [Luma](https://lu.ma/kv8zf6va). Members include AI pros like a film tool CTO, emphasizing hands-on projects. This initiative equips engineers with generative AI skills through peer-led learning.
- [**DSPy Parameters Proposal Sparks Optimization Buzz**](https://www.notion.so/swyx/dspy_discord): Engineers proposed adding learnable parameters to **DSPy**, creating [a GitHub issue](https://github.com/stanfordnlp/dspy/issues/8593) to brainstorm prompt optimizations. The idea lets templates act as variables for better results, drawing YouTube comparisons to **GEPA**. This enhances DSPy's utility for fine-tuning, fostering collaborative tweaks.

---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **R1-der R.I.P: Model Removed From LLM Selector**: The **R1 1776 model** has been removed from the LLM selector but is still accessible via the [OpenRouter API](https://openrouter.ai/perplexity/r1-1776).
   - Users are contemplating a switch to **O3** or **Sonnet 4** for reasoning tasks after the removal.
- **Android Comet App to Launch Soon**: The **Comet for Android** app is under development and is scheduled for release towards the end of the year.
   - While one user questioned the browser's potential capabilities, others lauded its performance on Windows.
- **Gemini 2.5 Pro Gets a Speed Boost**: Users reported a significant speed increase in **Gemini 2.5 Pro**, speculating it might be using **GPT-4.1**.
   - This improved performance may come with limitations like daily message caps for reasoning models.
- **Spaces Craze: Custom Instructions Heat Up**: Members discussed optimizing the use of the **Spaces** feature by adding custom instructions.
   - One user clarified that the **instruction field offers more options**, such as adding specific websites for data retrieval.
- **Deep Research API has Structured Outputs**: A member building a product stated that they're familiar with the **deep research and structured outputs API**.
   - They also asked to chat with somebody about **Enterprise API pricing**, early access, rate limits, and support, and requested some credits to test and integrate the API appropriately.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen3-30B Shuts Down Chatbot**: **Qwen released Qwen3-30B** and one user reported that when they asked **Qwen3** why China censors the internet on their chatbot, the system immediately shut down their requests.
   - The release of **Qwen3-30B** excited the community, with a link to [Hugging Face](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF) provided for further exploration.
- **GLM 4.5 Air Mimics Gemini**: Users discussed **GLM 4.5 Air**, with one mentioning it feels like **Gemini** and highlighted a [blog post comparing it to other models](https://z.ai/blog/glm-4.5).
   - Members noted that the **tool use is broken** in vllm but it worked well for chatting, poetry analysis, and document search.
- **Quantization Speeds Up Compute**: Quantization isn’t just about fitting models into memory, it also **reduces memory bandwidth** and can **significantly improve compute speed**.
   - A member noted that keeping the vision head ops like conv layers in **FP32** seems suboptimal, as they tend to be quite slow and is a bottleneck.
- **Gemma 3's watermarks Removed After 16k Fine-Tuning**: After fine-tuning **Gemma 3 4B** with **16k** context, experiments found that **watermarks** were completely removed and models were more stable, according to an attached [screenshot](https://cdn.discordapp.com/attachments/1179039861576056922/1399849804045221948/Screenshot_2025-07-29_at_2.13.07_PM.jpeg?ex=688bd0b9&is=688a7f39&hm=82cad388163625496b8cb3e6dd62035b51ae1d16ce0015df29ea910e04c4471f&).
   - A new **Gemma 3 competition** was announced with the notebooks being available and the competition ending in **7 days** with more info available on [Xitter](https://x.com/UnslothAI/status/1950553466591723640).
- **Unsloth Cracks Down on OpenAI's Censorship**: Members expressed disappointment with **OpenAI's heavy censorship**, sharing experiences of canned answers and lectures from ChatGPT.
   - One user pointed out [Unsloth's Llama 4 page](https://docs.unsloth.ai/basics/llama-4-how-to-run-and-fine-tune) that directs users to never use phrases that imply moral superiority or a sense of authority, and to generally avoid lecturing.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor's MCP Browser Automations Coming Soon**: Members are actively developing browser-reliant automations via **Cursor's MCP**, with early access slated for release in the coming weeks.
   - A member highlighted the ease of setup, noting it features a *one-click MCP setup* to facilitate building browser automations directly.
- **Parallel Agent Coordination Conundrums**: Members are grappling with managing parallel tasks with dependencies, given agents' lack of shared workspaces complicates simultaneous triggering.
   - Proposed solutions involve **external orchestration via API**, **file-based coordination**, and **Docker-based parallel execution**, including a sample [task queue script](https://github.com/example).
- **Cursor Background Agents Hijack Ports**: Engineers reported unexpected port hijacking by **Cursor Background Agents**, leading to debugging efforts to restore their development environments.
   - Mitigation suggestions include setting the `ports` property to an empty array or disabling *auto forward ports* in VSCode settings.
- **Background Agents Considered for Research**: A member explored utilizing background agents for research-oriented tasks, such as **refactor research** or **feature implementation research**.
   - They inquired about optimal workflows, considering options like having the agent draft markdown in the PR or directly implementing changes.
- **Fishy Terminal Defaults**: A member encountered an issue with **Cursor's integrated terminal** defaulting to the **fish** shell and sought solutions to change it.
   - Attempts to modify the shell via settings and wrappers eventually succeeded after temporarily renaming the fish binary, but the root cause remains unknown.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Dot-lol Data Collection Under Fire**: Concerns arose about [dot.lol](https://dot.lol)'s potential to **sell user data** and profiling information, cautioning users against assuming their data won't be used for targeted influence or profit.
   - While some worry about the impact of extensive data collection, others argue **data collection is inevitable** and users should focus on not linking data to their personal identity.
- **GPT-5 Buzz: August Release?**: Rumors hint at a possible **GPT-5** release in early **August**, with potential evidence of router architecture preparations found in the ChatGPT Mac app.
   - Community members are speculating on its impact and whether it will surpass other models, with some hoping for a free tier.
- **GDPR: Teeth or Toothless?**: Members debated the effectiveness of the EU's **GDPR** in preventing data collection by AI companies, with differing opinions on its impact.
   - Some believe that **GDPR** primarily affects the *use* of data rather than the *collection*, while others countered that *data collection is turned off for EU consumers*.
- **Zenith's Arena Encore**: Enthusiasm brewed for the return of the **Zenith** model to the **LMArena**, with anticipation for its potential **ELO** rating and overall performance.
   - While some lamented missing the chance to try it out, others held strong opinions on its value to the platform.
- **Lights, Camera, AI: Video Arena Launches**: The **LMArena** team launched an experimental **Video Arena** on Discord where users can generate and compare videos from top AI models for free, using the LMArena bot to **generate videos, images, and image-to-videos**.
   - A **Staff AMA** with the bot's developer, [Thijs Simonian](https://www.linkedin.com/in/thijsdev/), was announced, inviting users to submit questions via [Google Forms](https://docs.google.com/forms/d/e/1FAIpQLSfS_zh67vgEeftUXzX6ujlNiiFoTQNQd7P7-Nx6aD5zpFVOBg/viewform?usp=dialog) for the AMA.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF Spaces go Boom**: Members discussed that **HF Spaces** might unexpectedly restart, suggesting pinning dependency versions to avoid issues, as described in [the docs](https://huggingface.co/docs/hub/spaces-sdks-gradio-hardware#restarts).
   - One user reported that both *Ilaria RVC* and *UVR5 UI* stopped working and recommended factory rebuilds, while others continued to work fine.
- **P104-100 GPU on Sale for $15!**: Users debated the merits of the **P104-100** GPU for AI tasks, with one claiming it's *practically a 1080* for **15 GBP** (though others called it a scam), available from [Taobao](https://item.taobao.com/item.htm?id=897611642586&sku=Video%20memory%20capacity:8gb;Color%20classification:Galaxy%20p104-100%20(card%20length%2028cm);&sku_id=5919375243616).
   - Some cited limitations like **PCIE 1.0 x4**, while others emphasized its price-to-performance for LLM inference, even when sharding models across multiple cards.
- **Qwen 30B Challenges GPT-4o**: Users highlighted the release of the **Qwen 30B** model, claiming it rivals **GPT-4o** and can run locally in full precision with just 33GB RAM using the [Unsloth GGUF version](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF).
   - Users noted that a quantized version can run with 17GB RAM.
- **Diffusion Model MIT Curriculum Study Group**: A new **study group** will focus on learning **diffusion models** from scratch using **MIT’s curriculum** with early sign-up for **$50/month**, and two free intro sessions are available for non-members: **Aug 2** and **Aug 9**, details on [Luma](https://lu.ma/kv8zf6va).
   - The study group will be based on [MIT’s lecture notes](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf) and [previous recordings](https://aischolars.notion.site/).
- **MoviePy Makes Video Editing Server**: A member built a **MCP server** using **MoviePy** to handle basic video/audio editing tasks, integrating with clients like **Claude Desktop** and **Cursor AI**, available on [GitHub](https://github.com/Aditya2755/video-edit-mcp).
   - The author is seeking collaboration on features like object detection-based editing and TTS/SST-driven cuts.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-5 Hype Restsless Fans**: Users are actively debating the release of **GPT-5**, with some claiming access via **Microsoft Copilot/Azure**, but skeptics await an announcement from **OpenAI**.
   - One user humorously criticized **Sam Altman** for fueling hype and making fans *restless*.
- **Study and Learn: Distraction or Innovation?**: **OpenAI** launched a new **Study and Learn** feature, viewed by some as a simple system prompt and perhaps a distraction from **GPT-5** expectations.
   - One user even dumped the [system prompt](https://discord.com/channels/974519864045756446/998381918976479273/1399983224880496711) into the O3 model for analysis.
- **Copilot and ChatGPT Face Off**: Discussions clarify that **Microsoft Copilot** uses either **GPT-4o** or **O4-mini-high**, with potential future integration of **GPT-5** based on source code hints.
   - Copilot's unlimited daily message cap prompts questions about why users still prefer **ChatGPT**, though some users still believe Google's [Imagen4-Ultra](https://discord.com/channels/974519864045756446/998381918976479273/1400170254902235246) is the best image generator.
- **Chat History Vanishes into Thin Air**: Multiple users reported disappearing **ChatGPT chat histories**, despite attempts to troubleshoot by logging in and out and clearing cache.
   - OpenAI support suggests this *could be an isolated bug*, emphasizing that *chat history recovery isn’t possible once it’s lost*.
- **Engineering new AI Memory Format**: A member introduced a new memory format proposal with [AI_ONLY_FORMAT_SPEC](https://discord.com/channels/1046317268651794482/1046317269069864970?event=1200057848907847709), aimed at optimized AI VM, systems interfacing with vector databases, or for protected symbolic transfers, emphasizing speed and efficiency over human readability
   - Another member provided a detailed line-by-line reading of the format, highlighting its core principles like **token embedding**, **semantic grouping**, and **binary encoding**.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **DeepSecure Unveils Open-Source AI Agent Auth**: **DeepTrail** introduced **DeepSecure** ([https://github.com/DeepTrail/deepsecure](https://github.com/DeepTrail/deepsecure)), an open-source auth and delegation layer for AI agents, enabling authorization, agent-to-agent delegation, policy enforcement, and secure proxying across models, platforms, and frameworks.
   - The technology features a split-key architecture, gateway/proxy, separate control/data plane, policy engine, and macaroons, exemplified in integrations for Langchain/LangGraph like [secure multi-agent workflows](https://github.com/DeepTrail/deepsecure/blob/dev/examples/05_langchain_secure_tools.py) with fine-grained access controls and [delegation workflows](https://github.com/DeepTrail/deepsecure/blob/dev/examples/09_langchain_delegation_workflow.py).
- **OR Gives Free Messages for $10 Top-Up**: A one-time **$10** top-up on OpenRouter unlocks **1000 daily free messages**, even after the initial credits are depleted.
   - Users confirmed that after spending the initial $10, the **1000 requests/day** limit remains active.
- **API Blocks Unwanted Quantization**: OpenRouter's API now lets users specify acceptable quantization levels to avoid lower-precision models like **FP4** using the [provider routing documentation](https://openrouter.ai/docs/features/provider-routing#quantization-levels).
   - The API allows excluding specific quantization levels, such as allowing everything *except* FP4 models.
- **Deepinfra's Secret Gemini Pro Deal**: **DeepInfra** negotiated a lower rate with **Google** for **Gemini 2.5 Pro** and passed the savings to customers, indicated by a 'partner' tag on DeepInfra's listing.
   - Unlike the **Kimi K2** model, DeepInfra's **Gemini 2.5 Pro** has a partner tag, signaling a direct partnership with Google.
- **Ori Bot Underperforms**: Users report that the **OpenRouter Ori bot** may be a *net negative* due to inaccurate responses, particularly in payment processing issues.
   - One user suggested that **Ori** often puts the fault on the user and asks questions that *lead nowhere*, and a developer is now working to update **Ori's** knowledge.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Metislist Rank Provokes François Frenzy**: A user shared [Metislist](https://www.metislist.com/), sparking debate over **François Chollet's** rank at #80, with many feeling the creator of **Keras** deserved a higher placement.
   - Some felt that Chollet should be in the top 50, with one user quipping *Damn you got beef with my boy François?*.
- **Arcee Aces AFM-4.5B Model Drop**: Lucas Atkins announced the release of **AFM-4.5B** and **AFM-4.5B-Base** on Hugging Face from [Arcee.ai](https://xcancel.com/LucasAtkins7/status/1950278100874645621), touting the flexibility, high performance, and quality due to a data partnership with DatologyAI.
   - The models incorporate architectural improvements such as **grouped query attention** and **ReLU² activations**, with future releases planned for reasoning and tool use.
- **NotebookLM Now Summarizes Videos**: **NotebookLM** introduced a new feature for video overviews of articles and blog posts ([xcancel.com](https://xcancel.com/NotebookLM/status/1950298236914139234)), enabling users to grasp content without reading full texts.
   - Users lauded the innovation and suggested further development for interactive modes.
- **GPT-5 Spotted on MacOS**: References to **gpt-5-auto** and **gpt-5-reasoning** were discovered in MacOS app cache files ([xcancel.com](https://xcancel.com/apples_jimmy/status/1950514936444305534?s=46&t=fRVjULzONZQAlwHruKTgQg)), hinting at the imminent arrival of **GPT-5**.
   - Further corroboration mentioned **gpt-5-reasoning-alpha** in a biology benchmarks repository, leading to speculation about a potential announcement or release.
- **Anthropic Aiming High for $170B Valuation**: Anthropic is reportedly seeking to raise **$5 billion**, potentially valuing the AI startup at **$170 billion**, with a projected revenue of **$9 billion** by the end of the year ([xcancel.com](https://xcancel.com/EdLudlow/status/1950561790695448810)).
   - The news was met with comparisons to OpenAI and xAI.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Tableau Server Flexes LLM Orchestration Muscle**: A member reported successfully integrating the latest **Tableau Server edition** to incorporate an **LLM (server/on prem)** for **Vizql NLP**.
   - This setup aims to enable more sophisticated natural language processing capabilities within Tableau visualizations.
- **Gemini Agentic Framework Prototype Emerges**: A member shared a [**Gemini agentic framework**](https://cdn.discordapp.com/attachments/1124403655819415592/1399853283404939454/gemini-agentic-framework.zip?ex=688bd3f6&is=688a8276&hm=101f03e62cae13a72e1f4fdc681064aef0e5a3713de20aebac608c958f845b8b) prototype, describing it as a **one-shot prototype**.
   - The prototype leverages **AI Studio** for building agentic apps, emphasizing clear intention-setting for the builder agent to facilitate phased testing and focused model development.
- **NotebookLM Bypasses Bot Restrictions for Podcast Dreams**: In response to inquiries about podcast creation limitations due to bot restrictions on **NotebookLM**, a member clarified that the tools are accessible via the **API**.
   - They suggested rebuilding the workflow and manually loading reports into **NotebookLM**, as an alternative solution.
- **Obsidian and NotebookLM cozy up together**: An article detailing the integration of **NotebookLM**, **Obsidian**, and **Google Drive** was shared [here](https://www.xda-developers.com/using-notebooklm-obsidian-google-drive-together/).
   - A member volunteered to offer more detailed guidance on **Obsidian** usage, tailored to individual user needs.
- **NotebookLM Audio output varies between 8-15 minutes**: Users reported audio file generation with **NotebookLM** averaging *8-10 minutes*, though some have achieved up to *15 minutes*.
   - This discussion underscores the variability in output length, potentially influenced by content complexity and processing efficiency.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Tsunami Alert Issued After Russian Earthquake**: An **8.7 magnitude earthquake** off the coast of Russia triggered tsunami warnings for Hawaii and watches for the west coast of the United States.
   - Residents in affected areas were advised to monitor updates closely due to potential tsunami arrival hours later.
- **LM Studio Users Request Enhanced Conversation Handling**: Users are requesting a feature in LM Studio to copy and paste entire conversations, which are stored in **JSON format**.
   - One user pointed others to the [feature request channel](https://discord.com/channels/1110598183144399058/1128339362015346749) noting that many would find this feature useful.
- **LM Studio Model Relives February 18th**: A user reported that their LM Studio model was repeatedly referencing **February 18, 2024**, even when asked about current events.
   - Another user suggested checking the **system prompt** or **Jinja template** for the date.
- **Strix Halo APUs Pricey, Spark EPYC Debate**: The price of **Strix Halo APUs** is around **$1.6k** for 64GB and **$2k** for 128GB, but some members suggest that **EPYC** systems offer better value.
   - One member lamented *soldered memory* on such devices, drawing a comparison to a recent DIMM failure on a server and pointing to a [Corsair AI Workstation 300 with Strix Halo APU](https://www.guru3d.com/story/compact-ai-pc-corsair-ai-workstation-300-with-strix-halo-apu/).
- **9070 XT Performance Disappoints**: The **9070 XT** is significantly slower than a **4070 Ti Super**, with one user reporting a model running at **7 t/s** on their **4070 Ti Super** only achieved **3 t/s** on the **9070 XT**.
   - Another member suggested that RAM bandwidth limitations might be the cause.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaCloud Struggles to Read PDFs**: A member reported that **LlamaCloud** could not detect a **PDF file** and process it via API, using **n8n** for workflow simplification and linked to a [screenshot](https://cdn.discordapp.com/attachments/1059201661417037995/1399848961832911031/Screenshot_2025-07-29_at_22.13.16.png?ex=688bcff0&is=688a7e70&hm=b8f51e99fbeae087df203303f7665c4eab8447bb0890b55823fd36074c5ad539&).
   - This issue occurred when trying to process the **PDF file** via API.
- **Character AI Sparking Building Discussions**: Members discussed building a **character AI** with deep understanding of a large story, using a classic **RAG** pipeline with chunked text, embeddings, and a vector database.
   - This includes leveraging a **RAG** pipeline to create an AI with deep understanding.
- **Neo4j's Knowledge Graph Runs into Snags**: A member reported their simple graph storage implementing **Neo4j** is taking *ridiculously long to load* and their server is not compatible with **Neo4j 5.x**, and **LlamaIndex** doesn't seem to like **4.x**.
   - **Aura** is also blocked by the server proxy, presenting further roadblocks to implementation.
- **Flowmaker Gemini 2.5 Pro Bug meets Rapid Fix**: A member reported an error when using **Flowmaker** with **Gemini API** due to an invalid model name, requiring a number like *gemini-2.5-pro*.
   - A fix was [committed](https://github.com/run-llama/flow-maker/blob/aad0f47a81cacba662a07c4f2d70bd3425606e29/src/lib/llm-utils.ts#L19) and deployed swiftly, resolving the issue.
- **Community Offers RAG Debugging Assistance**: A member offered help with a **MIT-licensed repo** designed to debug tricky **RAG issues**, including sparse retrieval, semantic drift, chunking collapse, and memory breakdowns.
   - Following the initial offer, a community member inquired about specific complex issues addressed by the repo, focusing on concrete examples of *sparse retrieval* and *semantic drift*.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Expert Parallelism Exhibits Excellence?**: A member sought examples of **Expert Parallelism (EP)** outperforming **Tensor Parallelism (TP)**, but found that with **Qwen32B** and **Qwen 235B**, the all-reduce communication overhead made **EP** less performant.
   - They observed **EP** being beneficial only for models using **MLA** and needing **DP attention**.
- **Triton Treasures in Torch Compile**: To extract **PTX code**, members suggested using `TORCH_LOGS="output_code" python your_code.py` or accessing the `compiled_kernel.asm.keys()` dictionary, as detailed in [this blog post](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/?utm_source=chatgpt.com#ttir-triton-ir).
   - The `keys()` dictionary contains keys for intermediate representations including **llir, ttgir, ttir, ptx, and cubin**.
- **Inductor's Intriguing Influence on Triton**: To force **Triton code generation** for matmuls, members suggested configuring settings in [torch._inductor.config.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py#L459-L461), by modifying settings such as **use_aten_gemm_kernels**, **autotune_fallback_to_aten**, **max_autotune_conv_backends**, and **max_autotune_gemm_backends**.
   - However, it was noted that built-in kernels are often faster and not every op is converted to Triton by default.
- **CuTeDSL Compiler Cuts Prefetching Code**: A member shared a [blogpost](https://veitner.bearblog.dev/let-the-compiler-do-the-work-in-cutedsl/) and [code](https://github.com/simveit/software_pipelining_cute_dsl) on using **CuTeDSL** for **GEMM on H100**, explaining how to let the compiler handle prefetching.
   - The blogpost details an experimental argument to the `cutlass.range` operator to hint for prefetching, matching the performance of manual prefetching with simpler code.
- **Gmem Guardian: Synchthreads Saves the Day**: After copying from global memory (**gmem**) to shared memory, manually inserting a `synchthreads` (or equivalent) is necessary to sync all threads before moving on.
   - This guarantees that all shared memory elements are available for collective calculations such as **gemm**, reduction, or scan.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **M3 Ultra Crowned Local Inference King**: The **M3 Ultra** is gaining traction as a top choice for local inference, due to its **80-core GPU** and **512GB memory**, according to [this Reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/1j43ziq/the_new_king_m3_ultra_80_core_gpu_512gb_memory/).
   - One member instead bought a used **M1 16g** because nobody responded to their post.
- **Solving Model Latency Offshore**: A member is seeking solutions for running **LLMs offshore** with low latency, despite poor internet conditions.
   - Other members simply stated that they're fine with spending money on whatever they want.
- **Microsoft's Audio Instruction Research**: Members expressed interest in researching open-source solutions for improving **audio instruction-following** in **Speech-LLM models** to create better speech UIs, pointing to **Microsoft's Alignformer**.
   - Alignformer is not open source, so collaboration may be necessary.
- **ICL Demolishes Diagnostics?**: Members speculate that **in-context learning (ICL)** could break **interpretability tools** like **Sparse Autoencoders (SAEs)**, as **ICL** pushes activations out of their original training distribution, referencing the **Lucas Critique** and [this paper](https://arxiv.org/abs/2501.00070v1).
   - This issue isn't exclusive to **ICL**, but arises whenever **SAEs** encounter activation distributions different from those they were trained on, according to members.
- **Grouped GEMM Gaining Ground**: A member highlighted a PR supporting **torch._grouped_mm** in **GPT-NeoX**, now in PyTorch core, suggesting performance gains for **MoE** and linking to [this MoE implementation](https://github.com/EleutherAI/gpt-neox/blob/d12c771198388980ee054617e537665f044e0584/megatron/model/moe_mlp.py#L221).
   - They noted that interested users can use a one liner from TorchAO for **low precision MoE training**.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Paper on CUDA Generalization Surfaces**: A member shared a [paper](https://huggingface.co/papers/2507.14111) about generalizing beyond **CUDA**, prompting a reminder to post such content in the appropriate channel.
   - No further discussion or alternative perspectives were provided.
- **Mojo Newbie Braves TLS Handshake Gremlins**: A new Mojo user reported a **TLS handshake EOF error** when attempting to run a Mojo project with **pixi** and **magic shell**, using a **Dockerfile** from Microsoft Copilot.
   - A suggested fix involved using the latest nightly `mojo` package with **pixi** and specific commands (`pixi init my-project -c https://conda.modular.com/max-nightly/ -c conda-forge` and `pixi add mojo`), but this fix failed even with a **VPN**.
- **Mojo external_call Anatomy Examined**: Users questioned why Mojo's `external_call` uses specific functions like `KGEN_CompilerRT_IO_FileOpen` instead of standard `fopen` from **libc**, with a concern if this choice is about safety.
   - A member clarified that these are artifacts from older **Mojo** versions and are not a high priority to fix, and that the **KGEN** namespace belongs to Modular and will be opened up eventually.
- **Python call from Mojo suffers Overhead**: A user discovered significant overhead when calling a Python no-op function from Mojo (4.5 seconds for 10 million calls) compared to direct Python execution (0.5 seconds).
   - Members explain that Mojo needs to start a **CPython** process, and CPython is embedded via `dlopen libpython`, so you shouldn't call it in a **hot loop**.
- **Spraying Glue In a Race Car Engine**: Discussion covered the performance impact of calling Python from Mojo, especially in hot loops for tasks like **OpenCV** or **Mujoco** robot simulation.
   - Members note that a lot of fast python libs are actually C libs with a wrapper, and that *interacting with the context dicts alone can easily eat several hundred cycles*.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Deepseek-chat Suffers on OpenRouter**: Members observed that **Deepseek-chat** performed worse on **OpenRouter** compared to the official **DeepSeek API**, specifically when used as the editor model in architect mode.
   - A recommended fix involves using `aider --model openrouter/deepseek/deepseek-r1` to ensure the default config with `edit-format: diff` from [aider/resources/model-settings.yml](https://github.com/Aider-AI/aider/blob/main/aider/resources/model-settings.yml#L548) is applied.
- **Aider as Coding Model Training Catalyst?**: There was a suggestion that **Aider** could aid in coding model training by logging linting and undo actions during development workflows.
   - This approach would use **Aider** in *careful* development to generate valuable training data, although the commenter did not explicitly request developers to implement this feature.
- **Qwen3 Coder 30B-A3B Bursts Onto Scene**: An announcement was made of the new **Qwen3 Coder 30B-A3B**, shared via an image to validate its legitimacy.
   - Details are still emerging for this new model.
- **Litellm API Flounders With Connection Errors**: A user reported encountering numerous `litellm.APIConnectionError: Error parsing chunk: Expecting property name enclosed in double quotes: line 1 column 2` errors.
   - Despite the errors, functionality remained unaffected for the user.
- **Open Model R1 v Qwen Coder Duel Requested**: A member requested advice on the best open model for **aider**, considering unlimited hardware, and expressed interest in testing **R1** and **Qwen Coder** models.
   - The member mentioned having **Runpod credits** to use, indicating plans for practical testing with these models.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **LLM Safety Research Resources Sought**: A PhD student requested resources on current **LLM safety/alignment research**, with suggestions including blogs from the **AI alignment forum**.
   - Specifically, the blog posts [Thought Anchors: Which LLM Reasoning Steps Matter](https://www.alignmentforum.org/posts/iLHe3vLur3NgrFPFy/thought-anchors-which-llm-reasoning-steps-matter) and [Measuring Beliefs of Language Models During Chain-of-Thought](https://www.alignmentforum.org/posts/a86uAnPykqNtmEbDH/measuring-beliefs-of-language-models-during-chain-of-thought-1) were mentioned as helpful starting points.
- **Coding CUDA with Claude Confounds Coders**: A member discovered that writing in **CUDA** with **Claude** is difficult, requiring extensive preparation, comprehension, and arrangement.
   - They posited that the ultimate evaluation would be whether a Python programmer with some GPU and CUDA familiarity could utilize **Claude** to manage writing **kernels** and enhance performance, including [an image](https://cdn.discordapp.com/attachments/986699377257119794/1400233738369110026/image.png?ex=688be4ca&is=688a934a&hm=1bcb11346477e61edf05cde9751d5e62ee8992a2f64216c07e4a1a8f8fb14cc4).
- **Z.AI's 54 Repos Spark Interest**: A member inquired about the new **Z.AI 54 open source repos** and whether anyone has explored them, prompting curiosity within the community.
   - However, specific details regarding the contents or functionalities of these repositories were not elaborated upon.
- **Qwen3 allegedly equals GPT-4o**: A member shared a post noting that [Qwen3 30B A3B 2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) rivals **OpenAI's GPT-4o** in both English and Chinese.
   - The community is enthusiastic about **Qwen3** as a potential strong contender in the language model space with benchmarks indicating performance may be on par with **GPT-4o**, particularly in multilingual applications.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi Chatbot speaks Mandarin**: A user reported that the **Kimi chatbot** replied in **Chinese** despite receiving **English** prompts, possibly due to being logged out.
   - Screenshots revealed that while the reply was in English, the suggested sources and questions were in Chinese.
- **Kimi Learns from Social Media**: One member joked that **Kimi's training dataset** includes **Instagram** and **TikTok** comments, and linked to [Kimi on OpenHands v0.50.0k2](https://github.com/All-Hands-AI/OpenHands/releases/tag/0.50.0k2) to support this claim.
   - They suggest that this focus on social media data is what makes **Kimi** so good.
- **Moonshot AI's Got the Best Vibe**: A member stated that *moonshot got the best vibe no cap*, and linked to [X post](https://x.com/crystalsssup/status/1944287779896328668) about the AI community vibe check.
   - Another agreed that the community needs some competition.
- **Scale AI Provides Data**: A member stated that **Alexandr Wang** is the **founder and CEO of Scale AI**, which provides training data, annotation, and evaluation services.
   - They pointed out that **Scale AI** is essential for developing machine learning models.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Lume Edges Out Suna in Head-to-Head Showdown**: Members debated the merits of the **Lume** and **Suna** agentic systems; one member found that **Lume** coded out specific things with fewer mistakes.
   - The member noted they couldn't compare to **Manus** due to cost, and conceded they may not have prompted **Suna** correctly.
- **Manus' Comic Creation: A Diamond in the Rough?**: One member suggested that **Manus**' comic creation feature is nice but still can be improved, especially for free users.
   - Another member stated the service was declining in quality, with restrictive limits for free users, and questioned *whether Manus is dead*.
- **AI's Rosy View of Manus vs. Skeptical Human**: One member asked an AI what it thinks about the future of **Manus**, and it replied *I think the future of Manus is bright*.
   - Another member expressed skepticism, citing the release of agent modes from **OAI** and **Google** as competitive pressures.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP Server Faces Context Crisis**: A user questioned the need for additional **user-context separation** layers in a single cloud-deployed **MCP server instance** to prevent data sharing between unique sessions, referencing [issue #1087](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1087) and [MCP Docs](https://modelcontextprotocol.io/docs/tutorials/use-remote-mcp-server#understanding-remote-mcp-servers).
   - Another user reported challenges connecting to their **MCP server** via **Claude Desktop**, despite successful deployment to **EC2** with proper **SSL** configuration, but could still connect via **Cursor**.
- **Cucumber and LLMs Cooking up BDD**: A user shared a production-ready **Behavior-Driven Development (BDD)** side project, which includes a [diagram of the solution](https://cdn.discordapp.com/attachments/1312302100125843476/1399854833565044756/bael.jpeg?ex=688bd568&is=688a83e8&hm=2e86139e9f117265cd7cbef2afcc1a23a34a091e79402df9a0e051261231c695&).
   - Another user reported the **state tool** not working with **Windows-MCP** by **CursorTouch** in **Claude desktop**.
- **DeepTrail Unveils Deepsecure for AI Agent Auth**: **DeepTrail**, supported by Berkeley SkyDeck, is developing **Deepsecure**, an open-source auth and delegation layer for AI agents, documented on [GitHub](https://github.com/DeepTrail/deepsecure).
   - **Deepsecure's** architecture features a split-key design, gateway/proxy, separate control/data plane, policy engine, and macaroons for agent-agent delegation, detailed in its [technical overview](https://github.com/DeepTrail/deepsecure/blob/dev/docs/design/deepsecure-technical-overview.md).
- **FastMCP Asked About Tooling Dynamics**: A user asked about **FastMCP**'s dynamic tool selection capabilities when multiple tools are defined on the server.
   - Specifically, they want to know if **FastMCP** has logic to automatically select tools (e.g., math, web search, RAG, data interpreter) on the client side.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Gets Parameterized: Learnable Parameters Proposal Sparks Excitement**: A proposal to add learnable parameters (`dspy.variable` or `dspy.parameter`) to DSPy sparked enthusiasm, leading to [an issue on GitHub](https://github.com/stanfordnlp/dspy/issues/8593) to collect ideas and use cases.
   - The goal is to enable optimizers to generate optimal prompts by allowing *templates to be parameters/variables* and optimizing template variable placement.
- **F-Strings Flounder: Signature Implementation Stymied**: A member ran into issues implementing a signature using an f-string to verify code against a description and asked for help.
   - Another member advised against this approach, recommending that the parameter description be placed within `dspy.InputField()`.
- **DSPy Enters Genetic Prompting Arena**: A member highlighted a YouTube video comparing **DSPy** to **GEPA** and linked the [YouTube video](https://www.youtube.com/watch?v=o6RbVPFOslg) where it's said *DSPy optimizes the prompt you gave it; GEPA evolves a prompt you never imagined*.
   - The same member suggested evolving **MIPRO** into a reflective, genetic-style frontier engine for DSPy to generate prompts, challenging the YouTuber's claim.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **AMD Blazes Ahead in Gaming and AI**: A member suggested pairing the **7900XT** and **7800X3D** for gaming, noting AMD's consumer AI usability and long-term community benefits.
   - They linked to [a tweet](https://x.com/Teknium1/status/1950596567968477382) arguing in favor of AMD over Nvidia's **9070** and **9900X**.
- **Qwen Launches Coding Model with Thinking**: A member announced the release of the **Qwen3-30B-A3B-Thinking-2507** coding model on [Hugging Face](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507).
   - The linked Hugging Face model offers a new tool for code generation.
- **RLVR: Algorithm or Marketing?**: A member questioned the classification of **RLVR** (Reinforcement Learning, Virtual Reality) as a reinforcement learning algorithm, inciting discussion.
   - Another member, teknium, stated *"RLVR is just not a RL Algo its just a target of rl"* in response to [an NVIDIA tweet](https://fxtwitter.com/NVIDIAAIDev/status/1950279130450444670).



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Researcher Requests AI Safety Summer School Channel**: A new member requested a dedicated channel for an AI Safety **summer school** that happened weeks ago.
   - This reflects ongoing interest in academic and learning-focused discussions within the community.
- **Bias Buffs Band Together to Battle Bias**: A PhD student at JKU Linz is focusing on **mitigating social bias in LLMs**, with interests in **attribution for generative models**, **AI generated text detection**, and **domain adaptation**.
   - The student is keen to connect with others working on practical ethical concerns in domain-specific LLMs, seeking collaboration.
- **Kernel Koder Kodes Kernels in CUDA**: A member named Ali is deeply involved in **optimizing GPU kernels in Triton/CUDA** specifically for **autoregressive models**.
   - Ali is open to discussing low-level GPU programming, offering expertise in accelerating model performance.
- **Citation Configuration Conundrums Confront Cohere**: A member reported difficulty changing the **citation mode** using `citation_options` on `langchain_cohere.ChatCohere` and asked about implicit methods to pass citation options.
   - The member also inquired about the status of the [langchain-cohere repo](https://github.com/langchain-ai/langchain-cohere), noting its lack of recent updates and asking *if pull requests are welcomed*.
- **Senior Software Sage Seeks Sanctuary in South**: A remote **Senior Software Engineer** role is advertised, paying **$2K/month**, for a long-term contract, located in **Africa** or the **Americas**.
   - The role requires experience in **Ruby on Rails**, **Node.js**, **C#/.NET**, **Python**, **Java**, and strong English communication.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune User Requests LoRA-Style Adapters**: A Torchtune user expressed interest in **LoRA-style adapter support** that freezes original model weights and applies updates through additional trainable layers.
   - The user specified a desire for adapters that maintain the same forward compute path without increasing computational cost.
- **Torchtune Merges Weights After Training**: A user pointed out that Torchtune merges weights back in after training with an adapter, referencing the [Torchtune end-to-end workflow documentation](https://docs.pytorch.org/torchtune/0.6/tutorials/e2e_flow.html).
   - The user's comment sparked questions about the implications of merging weights in Torchtune.
- **ACL Paper Receives Accolades**: A member shared their **ACL paper** that won an award, with a link to the paper [here](https://aclanthology.org/2025.acl-long.266/).
   - No further discussion followed this announcement.
- **Glianorex finetunes trigger discussion**: A user asked if the **Glianorex finetunes** are public.
   - This comment could be interpreted as a complaint: *Glianorex is killing me and my doctor has been no help*.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Certificate Declaration Still Needs Completion**: A staff member has reminded another member to complete the **certificate declaration form** for the **LLM Agents (Berkeley MOOC)**.
   - The staff reiterated that the **form was never received**, despite a previous notification to the member.
- **Second Reminder for Certificate Submission**: The staff emphasized the importance of submitting the **Certificate Declaration Form** to finalize the **MOOC** requirements.
   - Failure to submit the form prevents the issuance of the certificate, impacting course completion verification.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AI Scholars Launches Diffusion Models Study Group**: A new study group is kicking off a **12-person**, **5-month** program (**2-4 hrs/week**) by **AI Scholars** using [MIT’s curriculum](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf) to study diffusion models, a key architecture in generative AI.
   - Confirmed members include the CTO of an AI film tool, AI art instructor, 2 LLM instructors, and 2 full-time AI researchers.
- **First Two Diffusion Models Sessions are Free**: The first two introductory sessions are free and open to non-members: August 2nd on *Flow Matching & Diffusion Models* and August 9th on *PDEs, ODEs, SDEs + A Brief History of Diffusion Models* ([links here](https://lu.ma/kv8zf6va)).
   - The program features peer-led sessions, mentor Q&A, hands-on projects, real research papers, and a tight, trusted cohort with a weekly format of 2 hours live class + 2 hours self-study, with students rotating teaching.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **User Seeks Cloud Deployment Strategy**: A user is seeking advice on deploying a language model, trained with a custom folder of PDFs, to the cloud for public use, and specifically wants a simple **GUI** for user queries.
   - Nomic suggested that the **enterprise plan** wasn't a good fit, and the user wondered about **Hugging Face deployment** as an alternative.
- **Enterprise Plan Inadequate for User Needs**: Nomic indicated that the **enterprise plan** isn't a good fit for the user's needs regarding deploying a custom language model.
   - The user is exploring alternative deployment strategies, such as **Hugging Face**, to make their language model accessible.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Plan to Eval Kimi 2?**: A member inquired about plans to evaluate **Kimi 2**.
   - They expressed curiosity about its **tool-use capabilities** after post-training.
- **Kimi 2 Tool-Use Post Training Interest**: Interest was expressed in evaluating **Kimi 2's tool-use performance** after its post-training phase.
   - The inquiry highlights the importance of assessing how well models adapt and utilize tools following their initial training.



---


The **tinygrad (George Hotz) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Codeium (Windsurf) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1399830622255710320)** (1152 messages🔥🔥🔥): 

> `R1 1776 Removal, Comet for Android, Gemini 2.5 Pro Speed, OpenRouter API for R1 1776, Daily Message Cap Limits` 


- ****R1-der R.I.P:** Model Removed From LLM Selector**: Members noticed that the **R1 1776 model** was removed from the LLM selector, but it is still accessible via the [OpenRouter API](https://openrouter.ai/perplexity/r1-1776).
   - Users speculate about switching to **O3** or **Sonnet 4** for reasoning tasks following the removal.
- ****Android Comet:** Mobile App to Launch Soon**: The **Comet for Android** app is currently in development and is expected to be released towards the end of the year.
   - A user expressed concerns about the browser's potential capabilities, while others praised its performance on Windows.
- ****Pro Gemini 2.5** Gets a Speed Boost**: Users have observed a significant increase in the speed of **Gemini 2.5 Pro**, suggesting it might be using GPT-4.1 instead of Gemini.
   - Members noted that this performance boost might come with limitations, such as daily message cap limits for reasoning models.
- ****Spaces Craze:** Custom Instructions Heat Up**: Members discussed how to best utilize the **Spaces** feature by adding custom instructions.
   - A user asked whether the space description field or the instruction field is better for setting the context for the space - with a member responding that the **instruction field offers more options** like adding certain websites from which to pull specific data.
- ****API API!:** Perplexity API Guidance**: Users shared tips on using the **Perplexity API**, with one noting that pro subscribers get a monthly allocation of **$5 USD** in credits.
   - A user who was facing a 401 error was advised to ensure their code and model selection (**sonar-pro**) are correct.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1399842279937806439)** (4 messages): 

> `Enterprise API pricing, Deep Research API` 


- **Deep Research API has structured outputs**: A member building a product that needs a lot of deep research capability and raising money soon for launch stated that they're relaxed on dev questions because they are *aquainted with the **deep research and structured outputs api**.*
   - They also asked to chat with somebody about **Enterprise API pricing**, early access, rate limits, and support, and requested some credits just to test and integrate the API appropriately.
- **Team ready to answer questions**: A member confirmed that the team is taking a look and asked what questions another member might have.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1399837823863095307)** (670 messages🔥🔥🔥): 

> `Hater Spams, GLM 4.5 Air, Qwen3-30B, OpenAI Censorship, Unsloth and Llama 3.1` 


- ****Discord Deals with Hater Spam****: A user reported a **hater spamming hate** and direct messaging everyone, leading to calls for banning certain words.
   - Members confirmed that it was *nothing important* and mentioned GPU-shaming as a common issue.
- ****Users Rave About GLM 4.5 Air****: Users discussed **GLM 4.5 Air**, with one mentioning it feels like **Gemini** and highlighted a [blog post comparing it to other models](https://z.ai/blog/glm-4.5).
   - Members noted that the **tool use is broken** in vllm but it worked well for chatting, poetry analysis, and document search.
- ****Qwen3-30B Release Excites Community****: **Qwen released Qwen3-30B** and [provided a link](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF), leading to discussions about its performance.
   - One user reported that when they asked Qwen3 why China censors the internet on their chatbot, the system immediately shut down their requests.
- ****OpenAI's heavy Censorship Disappoints Users****: Members expressed disappointment with **OpenAI's heavy censorship**, sharing experiences of canned answers and lectures from ChatGPT.
   - One user pointed out [Unsloth's Llama 4 page](https://docs.unsloth.ai/basics/llama-4-how-to-run-and-fine-tune) that directs users to never use phrases that imply moral superiority or a sense of authority, and to generally avoid lecturing.
- ****Unsloth Enhances Llama 3.1****: Users inquired about the difference between **unsloth/Llama 3.1** and just **Llama 3.1**, with community members clarifying that Unsloth provides fixes, template adjustments, and tokenizer improvements.
   - Unsloth's team also makes it easier to fine-tune models on consumer hardware, offering faster fine-tuning speeds with less memory usage.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1400100378560958464)** (6 messages): 

> `Unsloth introduction, Low end-to-end latency in TTS voice cloning` 


- **Indians and Students flock to Unsloth**: A member from India joined Unsloth after hearing about it on the **Hugging Face** official Discord, hoping to learn about finetuning and model deployment.
   - Another member mentioned he's planning to *get cracked at llms and join a dope company like Unsloth*.
- **Guidance Sought for Low-Latency TTS Voice Cloning**: A new member is seeking *concrete guidance* on achieving low end-to-end latency in **TTS voice cloning**.
   - The member requested advice on frameworks, model optimizations, or hardware strategies, and another member suggested our TTS fine-tuning notebooks.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1399849804346949803)** (9 messages🔥): 

> `Gemma 3 4B Fine-Tuning, Custom Tokens, Watermark Removal, RoPE FTW, Language Translation` 


- **Gemma 3 4B Finishes 16k Training**: After fine-tuning **Gemma 3 4B** with **16k** context, experiments found that **watermarks** were completely removed and models were more stable.
   - The research results were posted with an attached [screenshot](https://cdn.discordapp.com/attachments/1179039861576056922/1399849804045221948/Screenshot_2025-07-29_at_2.13.07_PM.jpeg?ex=688bd0b9&is=688a7f39&hm=82cad388163625496b8cb3e6dd62035b51ae1d16ce0015df29ea910e04c4471f&).
- **Custom Tokens create Chaos**: It was noted that unless training from scratch or with very large datasets, it's better to avoid **custom tokens** because models understand token splits.
   - For example, if a model sees *Yuki* split into *<*, *yuki*, *>*, it can better understand that Yuki = yuki, and that it is my token.
- **RoPE Gets a Novel Prize**: The poster praised the genius who invented **RoPE (Rotary Positional Embedding)**, as it works very well especially on smaller models.
   - However, to support huge contexts on inference, the poster thinks the field needs to invert some better optimizations, and quantization isn't enough.
- **Gemma speaks all Languages**: After testing translation abilities, the poster joked *OMG, it knows every (popular, at least) single languageOpenAI is doomed*
   - They also mentioned they keep getting this [song](https://www.youtube.com/watch?v=NgQoLPnuEDM) stuck in their head.
- **New Gemma 3 Notebooks**: A new **Gemma 3 competition** was announced with the notebooks being available and the competition ending in **7 days**.
   - More info available on [Xitter](https://x.com/UnslothAI/status/1950553466591723640).


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1399831912234881115)** (64 messages🔥🔥): 

> `Phi-4 Generate Flags Error, GGUF Conversion and Quantization of Fine-Tuned Models, Llama-CLI Performance Issues, RuntimeError in Google Colab, Unsloth BNB 4-bit Conversion` 


- **Phi-4 needs `do_sample=True`**: When generating with **Unsloth's Phi-4** model, users encountered an error related to invalid generation flags (temperature, min_p) and discovered adding `do_sample=True` resolves the issue, although it is not documented in the [official notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb#scrollTo=kR3gIAX-SM2qHi).
- **Qwen2.5 GGUF conversion woes**: Users are facing issues when trying to merge, convert to **GGUF**, and quantize a fine-tuned model based on **Qwen2.5-14B-Instruct-unsloth-bnb-4bit**, encountering a `ValueError: Can not map tensor 'model.layers.0.mlp.down_proj.weight.absmax'` error during export to **FP16 GGUF**.
- **Slow Llama-CLI with UD-Q2_K_XL model**: A user reported extremely slow performance (0.5 tokens/s) using **Llama-CLI** with the **Q2_K_XL** model, despite using a high-end system with a **5090**, **178GB RAM**, and **EPYC 9334** processor, and settings that should provide much better performance.
- **LLama3 fine-tuning faces RuntimeError**: A user reported a `RuntimeError: PassManager::run failed` error when fine-tuning **llama-3-8b-Instruct-bnb-4bit** on Google Colab, using a custom dataset formatted with **ShareGPT** templates and the Unsloth library.
- **Whisper input_ids error**: A user found that setting `task_type = None` in the `FastModel.get_peft_model` function resolves the `input_ids` error encountered when training the **Whisper** notebook, referring to [this issue](https://github.com/huggingface/peft/issues/1988#issuecomment-2751367819) for context.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1400049478131777556)** (8 messages🔥): 

> `Quantization optimization, Dynamic 4bit quantization, Hi-Fi Gan replacement, Autoregressive models, Mels dislike` 


- **Quantization Speeds Up Compute**: Quantization isn’t just about fitting models into memory, it also **reduces memory bandwidth** and can **significantly improve compute speed**.
   - A member noted that keeping the vision head ops like conv layers in **FP32** seems suboptimal, as they tend to be quite slow and is a bottleneck.
- **Dynamic 4bit quantization blogpost**: A member shared the [Dynamic 4bit quantization blogpost](https://unsloth.ai/blog/dynamic-4bit), relating to quantization optimization.
   - This blogpost is directly related to **quantization isn’t just about fitting models into memory**.
- **Hi-Fi Gan faces Autoregressive competition**: A member asked about the possibility to replace **Hi-Fi Gan** with [this](https://arxiv.org/abs/1609.03499) in **VITS**.
   - Another member asked if it was for autoregressive reasons, since the first member dislikes Mels; however the first member later decided against it due to long training times.


  

---


### **Unsloth AI (Daniel Han) ▷ #[unsloth-bot](https://discord.com/channels/1179035537009545276/1390899684834410536/1399847900598636554)** (102 messages🔥🔥): 

> `GRPO trainer batch size, SFTrainer validation error, Model fine-tuning parameters, Llama 3.2 data preparation, Gemma 3 fine-tuning` 


- **GRPO Trainer's Batch Size Deconstructed**: In the **GRPO trainer**, *per_device_train_size* represents the number of batches, which is then multiplied by the number of generations to determine the effective batch size.
   - For instance, with *per_device* set to **1** and *num_generation* to **6**, the configuration yields **3** unique prompts each with **6** generations under a single GPU, potentially leading to CUDA out-of-memory issues when expanding to **15k** tokens, especially considering the impact of GPU memory utilization on activation weights.
- **Seeking SFTrainer Validation Error Savior**: A user encountered an *evaluation_strategy* unexpected keyword error while trying to save the validation error with **SFTrainer**.
- **Llama 3.2 data format**: A user requested an example of the data preparation format to fine-tune **Llama 3.2 8B**.
- **Gemma 3 Text-Only Tuning Tactics**: Unsloth offers a [text-only notebook](https://docs.unsloth.ai/basics/tutorials-how-to-fine-tune-and-run-llms/gemma-3-how-to-fine-tune-and-run-llms/gemma-3-how-to-run-and-fine-tune#unsloth-fine-tuning-fixes-for-gemma-3) for fine-tuning **Gemma 3**, providing fixes for Unsloth fine-tuning.
- **Unlocking Positional Parameters for Adapter Loading**: When using *model.load_adapter*, the *adapter_name* is a required positional argument.
   - A user encountered a *ValueError* related to unsupported target modules (**ModuleDict**) and sought guidance on fixing this issue, aiming to merge fine-tuned LoRA adapters to the base model (**unsloth/gemma-3-4b-it-unsloth-bnb-4bit**) for GGUF conversion using Llama.cpp.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1399830240024465498)** (475 messages🔥🔥🔥): 

> `MCP Browser, Parallel Agent Tasks, VSCode Marketplace with Cursor, Automatic Scrolling, Sonnet Model` 


- **Cursor's MCP Browser setup**: Members are building browser-reliant automations directly via **Cursor's MCP** and early access will be coming out in the next few weeks ideally.
   - One member stated, *It has a one-click MCP setup so you can build browser-reliant automations directly via MCP*.
- **Parallel Agent Coordination Conundrums**: Members are discussing how to handle parallel tasks with dependencies, since agents don’t share workspaces, making it difficult to trigger them all at once.
   - Suggested solutions include an **external orchestration with API**, **file-based coordination**, or a **Docker-based parallel execution**, complete with a detailed example [task queue script](https://github.com/example).
- **VSCode Marketplace Integration Inquiry**: A member inquired about the possibility of using the **VSCode marketplace with Cursor**.
   - There was no clear answer in the discussion.
- **Automatic Scrolling Woes**: A new Cursor user asked if it is possible to disable **auto-scrolling** when using the **Agent chat window** to better read the thought process of Claude and the generated code.
   - There was no clear answer in the discussion, but the [changelog 1.3](https://cursor.com/changelog) was posted.
- **The Curious Case of the Terminating Terminal**: A member was having trouble with Cursor deciding which shell to start in its integrated terminal in the agent, and it defaults to **fish**.
   - Attempts to change the shell via settings and wrappers led to the temporary renaming of the fish binary and subsequent success, though the underlying cause remains a *fishy* mystery.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1399860099563917344)** (8 messages🔥): 

> `Background Agent Commands, Docker Build Cache, Port Hijacking, Background Agents for Research` 


- **Running Commands at Background Agent's Run's End**: A member inquired about executing a command, specifically a formatter, at the **end of a background agent's run**.
   - The member noted that `terminals` can be run during setup, but that's only at the beginning.
- **Busting Docker Build Cache**: A member sought advice on **busting the build cache** when using a custom Dockerfile with edited layers.
   - Another member suggested using `docker builder prune -f` or `docker system prune -a` to remove unused containers, networks, and images.
- **Cursor Background Agents Hijack Ports**: Engineers wasted time debugging why their dev environment was suddenly broken only to figure out that **Cursor Background Agents hijacked the ports**.
   - A member asked if setting the `ports` property to an empty array stops Cursor Background Agent from forwarding any ports, and another user suggested disabling “auto forward ports” in the VSCode settings instead.
- **Background Agents for Research**: A member inquired about using background agents for research, such as **refactor research or feature implementation research**.
   - The member asked about workflows, suggesting having the agent write markdown in the PR or letting it make changes directly.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1399828973818478592)** (413 messages🔥🔥🔥): 

> `dot.lol data, GPT-5 Release, EU GDPR impact, Zenith Model Relaunch, Video Arena Channels` 


- **Dot-lol faces data collection scrutiny**: Members discussed the potential for [dot.lol](https://dot.lol) to **sell user data** and profiling information, emphasizing the naivety of assuming data will never be used for targeted influence or profit.
   - Concerns were raised about data collection outweighing the usefulness of online services, while others argued that **data collection is inevitable** and users should prioritize not linking data to their identity.
- **GPT-5 release slated for August?**: Rumors suggest that **GPT-5** may be released in early **August**, with possible mentions of preparations in the ChatGPT Mac app confirming a router architecture.
   - Some community members speculate about its potential impact, debating whether it will surpass other models and even expressing hope for a free tier.
- **EU's GDPR Effectiveness Debated**: The effectiveness of the EU's **GDPR** in preventing data collection by AI companies was discussed, with opinions split on whether it sufficiently affects data collection practices.
   - Some argue GDPR primarily affects the *use* of data rather than the *collection* of it, but that *data collection is turned off for EU consumers.*
- **Zenith: Still on User's Minds**: Members expressed interest in the return of the **Zenith** model to the LMArena, speculating on its potential **ELO** rating and performance.
   - Some members lamented not having the chance to try the model, while others voiced strong opinions on its value to the platform.
- **New Arena Channels for Videos Pop Up**: Community members discussed the existence of **multiple video arena channels** and whether this was intentional.
   - A moderator explained that the video arena channels are meant *to spread the generations out a bit* and that too much activity in one channel *would be a bit much*.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1400151134299095092)** (1 messages): 

> `Video Arena, LMArena bot, Staff AMA` 


- **Video Arena Launches on Discord!**: The LMArena team launched an experimental **Video Arena** on Discord, allowing users to generate and compare videos from top AI models for free.
   - Users can learn how to use the bot in the specified channels and start generating videos, images, and image-to-videos, then vote on their preferred generation.
- **Vote on community generated videos!**: The LMArena bot allows users to **generate videos, images, and image-to-videos**, then lets anyone vote on which generations they prefer.
   - After a certain number of votes, the bot reveals the models used to generate each video.
- **Staff AMA with Bot Developer!**: To celebrate the launch of the **Video Arena**, a **Staff AMA** with the bot's developer, [Thijs Simonian](https://www.linkedin.com/in/thijsdev/), was announced.
   - Users were invited to submit questions via [Google Forms](https://docs.google.com/forms/d/e/1FAIpQLSfS_zh67vgEeftUXzX6ujlNiiFoTQNQd7P7-Nx6aD5zpFVOBg/viewform?usp=dialog) for the AMA.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1399829791682723901)** (290 messages🔥🔥): 

> `HF Space restarts, P104-100 GPU, LLM Deployment, Qwen 30B, SmolLM3` 


- **HF Spaces unexpectedly restart**: Members discussed why HF Spaces might unexpectedly restart and suggested pinning dependency versions to avoid issues caused by automatic reloads, as described in [the docs](https://huggingface.co/docs/hub/spaces-sdks-gradio-hardware#restarts).
   - One user reported that both *Ilaria RVC* and *UVR5 UI* stopped working, while others worked fine, and a factory rebuild was suggested.
- **P104-100: The $15 GPU**: Users debated the merits of the **P104-100** GPU for AI tasks, with one claiming it's *practically a 1080* for **15 GBP** (though others called it a scam), available from [Taobao](https://item.taobao.com/item.htm?id=897611642586&sku=Video%20memory%20capacity:8gb;Color%20classification:Galaxy%20p104-100%20(card%20length%2028cm);&sku_id=5919375243616).
   - Some noted its limitations (PCIE 1.0 x4, potentially only 4GB VRAM accessible), while others emphasized its price-to-performance ratio for LLM inference, even when sharding models across multiple cards.
- **Tiny LLMs for Edge Deployment**: Members sought advice on **low-latency LLM deployment** in remote, bandwidth-limited marine environments, with suggestions including edge/cloud hybrid approaches and **aggressive quantization**.
   - One user recommended checking out the [latest pytorch quant of smollm3 on HF](https://huggingface.co/pytorch/SmolLM3-3B-8da4w) for mobile deployment, and another suggested deploying apps when docked.
- **Qwen 30B Model: GPT-4o Challenger?**: Users highlighted the release of the **Qwen 30B** model, claiming it rivals **GPT-4o** and can run locally in full precision with just 33GB RAM ([Unsloth GGUF version](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF)).
   - It was noted that a quantized version can run with 17GB RAM.
- **SmolLM3 Quantization Problems**: A member mentioned they were having quantization issues with **SmolLM3** using *torchao*, and others suggested trying *hqq* or the [official SmolLM3-3B-8da4w](https://huggingface.co/pytorch/SmolLM3-3B-8da4w).
   - One member noted that *--jinjai* should be used if using *llama.cpp*.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1400252994091487382)** (4 messages): 

> `Muon Optimizer, Smithery` 


- **Muon Optimizer Hailed!**: Members shared a link to the **Muon Optimizer** ([https://kellerjordan.github.io/posts/muon/](https://kellerjordan.github.io/posts/muon/)).
   - They exclaimed *smithery! smithery so killer!*
- **Smithery is killer**: Another member replied that **Smithery** is so killer
   - It seems that **Smithery** is well received by this channel.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1399853091410415798)** (5 messages): 

> `Petite Elle Model, Gradio MBTI App, Video Editing MCP Server, Github Python Dataset` 


- **Petite Elle Gets the Mrad Treatment**: A member's model, [Petite Elle-L'aime-3-sft](https://huggingface.co/Tonic/petite-elle-L-aime-3-sft), received the **mradermacher treatment** and is expected to be among the best French-speaking models at that size.
   - The quantized versions are available at [mradermacher's Hugging Face page](https://huggingface.co/mradermacher/petite-elle-L-aime-3-sft-GGUF).
- **Gradio App Flows with Gemini for MBTI**: A new Gradio app for **MBTI (Myers-Briggs)** uses **PocketFlow** and **Gemini**.
   - Check out the [app](https://huggingface.co/spaces/Fancellu/mbti-pocketflow) and the underlying [PocketFlow library](https://github.com/The-Pocket/PocketFlow).
- **MoviePy Makes Video Editing Server**: A member built a **MCP server** using **MoviePy** to handle basic video/audio editing tasks, integrating with clients like **Claude Desktop** and **Cursor AI**.
   - The code is available on [GitHub](https://github.com/Aditya2755/video-edit-mcp), with the author seeking collaboration on features like object detection-based editing and TTS/SST-driven cuts.
- **Github Python Dataset Released**: A new dataset, [Github Python](https://huggingface.co/datasets/jblitzar/github-python), contains all Python files of a reasonable size after 2015 in repositories with over 10 stars on Github.
   - The dataset is filtered for deduplication and permissive licenses.


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1399862051274232003)** (2 messages): 

> `Diffusion Models, Flow Matching, MIT curriculum` 


- **Study Group Kicks off for Diffusion Models**: A new **study group** will focus on learning **diffusion models** from scratch, a core architecture in generative AI.
   - The group is based on **MIT’s curriculum** and consists of **12 people** over **5 months** (**2–4 hrs/week**).
- **Free Intro Sessions open to non-members**: The first two free intro sessions are available for non-members: **Aug 2** - What is **Flow Matching & Diffusion Models**?; **Aug 9** - **PDEs, ODEs, SDEs** + A Brief History of Diffusion Models.
   - The sessions are at **12 PM EST**, further details on [Luma](https://lu.ma/kv8zf6va).
- **MIT lecture notes will be used**: The study group will be based on [MIT’s lecture notes](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf) and [previous recordings](https://aischolars.notion.site/).
   - Early sign-up is **$50/month** (goes up to **$100/month**); funds go towards paying the **teaching assistant**.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/)** (1 messages): 

hedi1421: Thanks 😅
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1400040059947712543)** (1 messages): 

> `Fixing transformers issue, DeepSpeed Integration` 


- **Transformers issue fix sought**: A member is requesting assistance with fixing [this transformers issue](https://github.com/huggingface/transformers/issues/39753).
   - No additional details about the issue were provided.
- **DeepSpeed Integration**: Discussion around DeepSpeed integration within the Hugging Face ecosystem.
   - Members are exploring the best practices and potential performance gains.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1399919753157673102)** (1 messages): 

> `DuckDuckGo deprecation, Smolagents merge` 


- **DuckDuckGo Search Package Faces Deprecation**: The `duckduckgo-search` package is still deprecated, as discussed in [this pull request](https://github.com/huggingface/smolagents/pull/1548).
   - A member inquired about the timeline for its merge into `smolagents`.
- **Smolagents Merge on the Horizon**: The proposed merge aims to integrate updates and fixes into the `smolagents` library.
   - Community members are eagerly awaiting the completion of the merge to leverage the latest improvements.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1399995784853848126)** (3 messages): 

> `RAG System Construction, Tool Definition Problems in Unit 1` 


- **RAG System scans conversation history**: A member plans to build a **RAG system** and use an **LLM** to scan conversation history, extract user-specific cases, and embed them in vector space using a for loop.
   - They intend to test the feasibility of this approach.
- **Unit 1 Tool Definition Troubleshoot**: A member reports that their tool definition in **app.py** isn't reflected at runtime in **Unit 1**.
   - They have already tried restarting the space without success and is asking for suggestions.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1399829176923717713)** (261 messages🔥🔥): 

> `Study and Learn feature, GPT-5, Copilot, Gemini vs ChatGPT, AI Ecosystems` 


- **OpenAI Releases **Study and Learn** Feature**: OpenAI launched a new **Study and Learn** feature, which some users see as a simple system prompt rather than a significant update and some users believe this feature is **intended to distract users from GPT-5 hype**.
   - One user dumped the [system prompt](https://discord.com/channels/974519864045756446/998381918976479273/1399983224880496711) into O3 model.
- **GPT-5 Speculation Fuels Discussion**: Members are actively debating the release of **GPT-5**, with some claiming to have access via Microsoft Copilot/Azure; however, many users are skeptical and are expecting a formal announcement from OpenAI.
   - One user commented on the CEO by saying *frock you **Sam Altman** for making your chatgpt fans restless waiting for the hypes you create*.
- **Comparing Copilot to ChatGPT**: Users discussed whether **Microsoft Copilot** uses **GPT-5**, however it was clarified that it uses either **GPT-4o** or **O4-mini-high**, and some leakers have found hints in the source code that **GPT-5** may be integrated in the future.
   - The daily message cap for **Copilot** is unlimited, leading some to question why people prefer **ChatGPT** for its tools.
- **ChatGPT and Gemini Comparison Underway**: Users debated preferences between **ChatGPT** and **Google Gemini**, with one user listing six key reasons for preferring **ChatGPT**, including connectors, RAG capabilities, style matching, memory, and deep research, but others were quick to counterpoint them.
   - One user noted, that [Google's Imagen4-Ultra](https://discord.com/channels/974519864045756446/998381918976479273/1400170254902235246) generates the best images after one user posted different AI-generated pictures of an *ultra rich man house like iron man house*.
- **Navigating the AI Ecosystem**: Members discussed choosing the right **AI ecosystem**, weighing options like **Apple One + ChatGPT Plus** against **Microsoft 365 + Microsoft Copilot** or **Google One AI Premium + Google Gemini**.
   - It was suggested to try out both to determine which best suits individual needs, with some mentioning Gemini's integration with Google Docs and Slides.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1399902317595328642)** (24 messages🔥): 

> `GPT-5 versions, O4 mini vs 4o, Missing Chat History, ChatGPT memory issues` 


- **GPT-5 Versions Speculations**: Members speculate that mid and high tier versions of **GPT-5** will be superior, with one member noting that **Zenith** might be the top coding model until a new one is released.
   - No links provided.
- **O4 Mini Debated over 4o Model**: A member inquired whether **O4 mini** should be used over **4o (the free model)** for more intelligent responses, referencing **O4 mini** and **O3's advanced reasoning** capabilities.
   - No links provided.
- **ChatGPT Chat Histories Vanish, Stressing Users**: Several users reported their **ChatGPT chat histories disappearing**, with one user logging in and out, clearing cache, and checking multiple devices, to no avail.
   - An OpenAI support member suggested it *could be an isolated bug* and that *chat history recovery isn’t possible once it’s lost*, advising to periodically save copies of important information outside of ChatGPT.
- **Memory Issues Plague ChatGPT Users**: A user mentions they are working on a local workaround for **memory issues** experienced while using **ChatGPT**.
   - This was after another user mentioned that their recent chats after October 2024 were not loading, but they could still access new chats and custom instructions.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1399879683956277360)** (10 messages🔥): 

> `Personalized GPT setup, AI Memory Format, Optimized AI VM` 


- **Personalized GPT Setup Resource Quest Begins**: A new user seeks resources for setting up **GPT projects** to track **food/diet**, **exercise**, and create a **planner** with time expectations, requesting instruction resources and prompts to enhance their account's capabilities.
   - Another member suggests a **personalized approach**, advising direct interaction with the model to discuss desired features and consider additional options, and mentioned that not everyone agrees what 'more powerful' means.
- **New AI Memory Format: Speed vs Readability Debated**: A member introduced a new memory format proposal, aimed at optimized AI VM, systems interfacing with vector databases, or for protected symbolic transfers, emphasizing speed and efficiency over human readability, using a [AI_ONLY_FORMAT_SPEC](https://discord.com/channels/1046317268651794482/1046317269069864970?event=1200057848907847709) to keep memories from being massively inefficient in terms of storage and readability.
   - Another member provided a detailed line-by-line reading of the format, highlighting its core principles like **token embedding**, **semantic grouping**, and **binary encoding**, cautioning against running it due to potential compression or encoding responses.
- **Prompt Engineering Effectiveness Examined**: The discussion involves the effectiveness of prompt engineering, with one member finding it *exhausting to explicitly spell out every aspect of the ask*, preferring to provide context and rely on the AI's reasoning skills.
   - Another member suggests that such conversational users are precisely who prompt engineers design system prompts for.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1399879683956277360)** (10 messages🔥): 

> `GPT project guidance, Personalized AI models, AI memory format` 


- **Guidance Sought for GPT Project Setup**: A new user is seeking resources for setting up projects on their **GPT account**, specifically for tracking food/diet and exercise, and creating a planner with time expectations.
   - They are looking for guidance and tools to enhance the instructions for these common projects, potentially making them *more powerful*.
- **Personalized AI is Key**: A user suggests personalizing AI models by discussing desired features and considerations with the model itself.
   - They emphasize that what constitutes *more powerful* varies, with personalization being crucial for tailoring the AI to specific interests and goals.
- **Memory Format for AI**: A member introduced a new memory format suggestion for AI, designed for efficient storage and retrieval of conversational memories.
   - This format aims to improve persistent memory by using a compact, binary-encoded structure with semantic compression and implicit context, optimized for AI VM and vector database interfacing.


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1399839193567465624)** (2 messages): 

> `DeepTrail, DeepSecure, AI agent authorization, Agent delegation, Policy enforcement` 


- ****DeepTrail** builds open-source **DeepSecure****: A member is building **DeepTrail**, an open source auth and delegation layer for AI agents backed by Berkeley SkyDeck.
   - With **Deepsecure** ([https://github.com/DeepTrail/deepsecure](https://github.com/DeepTrail/deepsecure)), developers can integrate authorization, agent-to-agent delegation, policy enforcement, and secure proxying across any model, platform, or framework with just a few lines of code.
- ****DeepSecure's** technicals under the hood**: The technology involves a split-key architecture, gateway/proxy, separate control/data plane, policy engine, and macaroons for agent-agent delegation, detailed in the [technical overview](https://github.com/DeepTrail/deepsecure/blob/dev/docs/design/deepsecure-technical-overview.md).
   - There are also several simple examples and integrations for Langchain/LangGraph.
- ****DeepSecure** examples using Langchain/LangGraph**: The member has built some examples of **DeepSecure** integrations for Langchain/LangGraph, including [secure multi-agent workflows](https://github.com/DeepTrail/deepsecure/blob/dev/examples/05_langchain_secure_tools.py) with fine-grained access controls.
   - The repo also features [delegation workflows](https://github.com/DeepTrail/deepsecure/blob/dev/examples/09_langchain_delegation_workflow.py), [advanced delegation patterns](https://github.com/DeepTrail/deepsecure/blob/dev/examples/11_advanced_delegation_patterns.py), and [platform agent bootstrapping](https://github.com/DeepTrail/deepsecure/blob/dev/examples/12_platform_expansion_bootstrap.py).


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1399849676123013373)** (152 messages🔥🔥): 

> `NotebookLLM, OpenRouter Pricing, Blocking quants via API, Becoming a provider, API Key Issues` 


- ****Unlock Daily Free Messages with a One-Time OR Top-Up****: Adding **$10** to OpenRouter credits unlocks **1000 daily free messages** as a one-time purchase, even if the credits are used up.
   - Users confirmed that even after the initial $10 of credits are exhausted, the **1000 requests/day** limit remains unlocked.
- ****API Enables Quantization Control****: Users can now specify acceptable quantization levels via the API to avoid lower-precision models like **FP4**, using the [provider routing documentation](https://openrouter.ai/docs/features/provider-routing#quantization-levels).
   - The API allows specifying exclusions, like allowing everything *except* FP4 models.
- ****Pydantic-AI and Kimi-K2 Join Forces to Tackle Bugs****: A user highlighted the benefits of **pydantic-ai**, including its fully pydantic approach, MCP server support, model/provider adapters, and automated graph crafting and noted they had fixed a bug using **Kimi-K2**.
   - The user emphasized pydantic-ai's ability to focus on business logic rather than *fiddling with gluing together an agentic framework from disparate bloatparty repos*.
- ****OR Faces Kimi-K2 Tool Calling Issue****: A user believes they've identified an issue with **Kimi K2's** tool calling support on OpenRouter, potentially fixable via model template adjustments.
   - The user provided [research](https://discord.com/channels/1091220969173028894/1400028050007265340) with examples from frameworks like vllm and expressed that fixing this issue could lead to **80% savings** for their business, and suggested that they will move to moonshot.
- ****Gemini Flash 1.5 Faces Overload Issues****: **Google Gemini Flash 1.5** is reportedly showing *error 503: The model is overloaded*, with a user sharing the [pricing structure](https://discord.com/channels/1091220969173028894/1195014798837043240/1400220368765194350).
   - The model has a fluctuating pricing, with input ranging from **$0.075 to $0.15** and output from **$0.30 to $0.60**.


  

---


### **OpenRouter (Alex Atallah) ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1400206735389491232)** (2 messages): 

> `` 


- **No new models updates in OpenRouter**: There were no significant discussions or updates regarding new models in the OpenRouter channel.
   - The channel remained inactive, lacking any substantive information for summarization.
- **Readybot.io logs no new activity**: The Readybot.io logs indicate a period of silence in the OpenRouter - New Models channel.
   - Consequently, there are no specific topics or discussions to report from this time.


  

---


### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1399913686432219186)** (63 messages🔥🔥): 

> `Quantized Providers, Groq's Quantization, Deepinfra Pricing, Vertex for Claude, OpenRouter's Ori bot` 


- **Quantized Providers' Default Status Debated**: A user suggested that quantized providers should be disabled by default, which could affect **Groq** due to its unique quantization approach.
   - Another user cautioned about the risks of publicly shaming providers before reaching a *critical mass* of users, potentially causing providers to exit **OpenRouter**.
- **Deepinfra Offers Gemini 2.5 Pro via Google**: **DeepInfra** reportedly negotiated a lower rate with **Google** for **Gemini 2.5 Pro** and passed the savings onto customers, confirmed by a user who cited a message from DeathMax and the presence of a 'partner' tag on DeepInfra's listing.
   - DeepInfra's **Gemini 2.5 Pro** has a "partner" tag unlike the **Kimi K2** model, indicating a direct partnership with Google.
- **Vertex proves Victorious for Claude 4**: One user reported better quality, throughput, and uptime by using **Vertex** for **Claude 4 Sonnet**.
   - The user also noted that AWS/GCP/Azure mirrors for closed models could provide qualitative differences.
- **OpenRouter Ori Bot's Accuracy Under Scrutiny**: A user suggested that **OpenRouter's Ori bot** might be a *net negative* due to inaccurate responses and should be limited or disabled.
   - The user pointed out that **Ori** often puts the fault on the user and asks questions that *lead nowhere*, especially in payment processing issues.
- **Adding Knowledge Update Feature for Ori bot**: One of the developers is working to add ways to update **Ori's** knowledge when it gets things wrong.
   - Others pointed out that **Ori** is missing a lot of knowledge and is hallucinating incorrect knowledge, and suggested limiting the bot's answers to a maximum of 2-3.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1399843438395916339)** (188 messages🔥🔥): 

> `Metislist ranking, Arcee.ai Releases AFM-4.5B, NotebookLM video overviews, Claude abuse, ryOS release` 


- **Metislist Sparks Chollet Placement Debate**: A user shared a link to [Metislist](https://www.metislist.com/), a ranking of individuals in the AI field, which spurred discussion about François Chollet's placement at number 80.
   - Many agreed that Chollet, known for Keras, should be in the top 50, or even removed from the list entirely, one user joking *Damn you got beef with my boy François?*.
- **Arcee AI ships AFM-4.5B models**: Lucas Atkins announced the release of **AFM-4.5B** and **AFM-4.5B-Base** models on Hugging Face from [Arcee.ai](https://xcancel.com/LucasAtkins7/status/1950278100874645621), highlighting their design for flexibility, high performance, and quality due to a data partnership with DatologyAI.
   - The models feature architectural tweaks like **grouped query attention** and **ReLU² activations**, and the team plans to release future models for reasoning and tool use.
- **NotebookLM now does Video Overviews**: **NotebookLM** announced a new feature for video overviews of articles and blog posts ([xcancel.com](https://xcancel.com/NotebookLM/status/1950298236914139234)), enabling users to quickly grasp content without reading everything.
   - Users praised the innovation and suggested further development for learning tools and interactive modes.
- **GPT-5 spotted prowling MacOS**: References to **gpt-5-auto** and **gpt-5-reasoning** have been found in the MacOS app cache files ([xcancel.com](https://xcancel.com/apples_jimmy/status/1950514936444305534?s=46&t=fRVjULzONZQAlwHruKTgQg)), suggesting the imminent release of **GPT-5**.
   - Other users have corroborated this, mentioning **gpt-5-reasoning-alpha** in a biology benchmarks repository, while some speculated on an imminent announcement or release.
- **Anthropic Seeks Sky-High Valuation**: Anthropic is reportedly in talks to raise **$5 billion**, potentially valuing the AI startup at **$170 billion**, with projected revenue of **$9 billion** by year-end ([xcancel.com](https://xcancel.com/EdLudlow/status/1950561790695448810)).
   - This news sparked comparisons to other AI companies like OpenAI and xAI, although one user commented that he has *some second hand info that this is not the case*.


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1400171262210474005)** (17 messages🔥): 

> `Anthropic Fellows Papers, LLM Paper Club, Social Media Engagement` 


- **Anthropic Fellows Papers Featured**: The Latent Space Discord announced it would cover recent **Anthropic Fellows papers** in the <#1107320650961518663> channel.
- **LLM Paper Club Seeks Volunteers**: The **LLM Paper Club** is seeking volunteers to cover papers in future clubs; interested individuals are encouraged to sign up via a [Luma link](https://lu.ma/6uti3zzy).
- **Call for Retweets Misses the Mark**: A member posted a [link on X](https://x.com/latentspacepod/status/1950613048303231121) advertising the club, but lamented his low engagement.
   - He jokingly claimed to be *“not professional yapper”* and bad at tweeting.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1399847059045093637)** (32 messages🔥): 

> `Tableau Vizql NLP Orchestration, Gemini Agentic Framework Prototype, NotebookLM Podcast Creation, Obsidian and NotebookLM Integration, NotebookLM Usage Analytics` 


- **Tableau Server used for LLM Orchestration**: A member mentioned using the latest **Tableau Server edition** to fit in the **LLM (server/on prem)** for **Vizql NLP**.
- **Gemini Agentic Framework Prototype Shared**: A member shared a [**Gemini agentic framework**](https://cdn.discordapp.com/attachments/1124403655819415592/1399853283404939454/gemini-agentic-framework.zip?ex=688bd3f6&is=688a8276&hm=101f03e62cae13a72e1f4fdc681064aef0e5a3713de20aebac608c958f845b8b) prototype, noting it was a **one-shot prototype**.
   - They suggested using **AI Studio** to build agentic apps by detailing intentions clearly for the builder agent to create a working prototype, allowing for phased testing and model focus.
- **Bypassing Bot Restrictions for Podcast Creation**: A member inquired about creating podcasts with **NotebookLM**, given bot restrictions on logins.
   - Another member clarified that the tools driving **NotebookLM** are available via the **API**, suggesting a rebuild in another workflow and manual report loading into **NotebookLM**.
- **Obsidian and NotebookLM Integration Discussed**: A member shared an article on integrating **NotebookLM**, **Obsidian**, and **Google Drive** [here](https://www.xda-developers.com/using-notebooklm-obsidian-google-drive-together/).
   - Another member offered to provide more specifics on using **Obsidian**, depending on the other member's usage.
- **NotebookLM Audio output averages 8-15 minutes**: Some members asked about long audio file generation with **NotebookLM**.
   - Another member stated that their average output is *8-10 minutes*, although other members had generated up to *15 minutes*.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1399828961021788412)** (156 messages🔥🔥): 

> `NotebookLM Video Overview Limits, Studio UI Changes, Video Generation Length, NotebookLM Rollout, Missing the new Notebook LM Feature` 


- **Pro Users Get 20 Video Overviews Daily**: Pro users of NotebookLM receive **20 video overviews per day**, as confirmed by a member, and found in [Google Support documentation](https://support.google.com/notebooklm/answer/16213268?hl=en&ref_topic=16175214&sjid=12603864792385823108-NA#:~:text=With%20NotebookLM%2C%20you,daily%20video%20generations.).
   - However, some users are experiencing delays in accessing the Video Overview feature and the updated Studio UI despite being Pro users.
- **Studio Interface Needs Sort/Filter Outputs**: A user suggested that the **Studio interface** needs the ability to *sort/filter outputs* and a *delete all option*, as well as the ability to **stop a video generation in progress** that won’t complete.
   - Another user highlighted that the *save all notes to source is gone*, potentially creating an issue with the source limit of **50** in the free version.
- **Video Generation Times Vary Wildly**: Users reported varying video generation times, with one user experiencing a **30-minute** generation from an Economist article, leading to discussion on whether **Veo 3** is being used.
   - A user described the output as *more of a presentation than an animated video*, noting its tendency towards typography, suitable for text-heavy content.
- **Rollout Reaches Germany, Slow for Others**: The **Video Overview feature** became available in **Germany** on Pro accounts, while many users, including those with Google Ultra, are still waiting for the rollout.
   - Google confirmed that the update is rolling out gradually to all users over the next week.
- **Video Overview Glitches Exposed**: Users are reporting that video overviews are limited to **6-7 minutes** and are experiencing abrupt transitions between sections.
   - There's a glitch where the video loads indefinitely until the page is refreshed.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1399838809134334052)** (117 messages🔥🔥): 

> `LM Studio model renaming, Earthquake off the coast of Russia, LM Studio copy/paste conversation feature request, LM Studio Model stuck in time, Qwen 30B garbage output` 


- **Tsunami warning off the coast of Russia**: An **8.7 magnitude earthquake** occurred off the coast of Russia triggering tsunami warnings for Hawaii and watches for the west coast of the United States.
   - Residents in affected areas were advised to monitor updates closely due to potential tsunami arrival hours later.
- **Feature Request for LM Studio: Copy-Pasting Whole Conversations**: A user inquired about a feature in LM Studio to copy and paste entire conversations, which are stored in **JSON format** and relatively easy to manipulate.
   - Another user mentioned they started a Python app for extracting conversations but got distracted, suggesting others add a feature request in the [feature request channel](https://discord.com/channels/1110598183144399058/1128339362015346749) as many would find it useful.
- **LM Studio Model Stuck in Time Loop**: A user reported that their LM Studio model was repeatedly referencing **February 18, 2024**, even when asked about current events, providing a screenshot as evidence.
   - Another user suggested checking the **system prompt** or **Jinja template** for the date, as it might be causing the model to think it's in that specific date.
- **Qwen 30B's Sampler Settings**: A user noted that **Qwen 30B** frequently produces garbage output unless the prompt is reprocessed.
   - Another user suggested trying the official samplers or the ones provided to see if the output improves, with one noting a similar issue on Linux that was resolved by updating to experimental drivers.
- **lmstudio MCP client needs resources**: Users discussed the potential use of resources in the **LM Studio MCP client**, highlighting use cases such as lower-cost read-only references for syntax guides or dynamic codes.
   - One user mentioned they use resources for discovery, documentation, and navigation helpers, preferring resources over tool calls for quick reference information and hoping for client updates to support features from the **2025-06-18** update of the MCP specification.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1399876515449147545)** (64 messages🔥🔥): 

> `GPU Usage, Strix Halo, Threadripper vs Epyc, Soldered RAM, 9070 XT Performance` 


- **Splitting Models Across GPUs Affects Usage**: When splitting a model between two GPUs, each GPU operates at **50% usage**, potentially reducing heat and noise, though this is dependent on how the model layers are split and processed sequentially.
   - A 4090 paired with a slower 3070 might result in the 4090 being idle while waiting for the 3070 to complete its tasks, but improvements in performance are still seen with an increase from **8 to 32 tok/sec**.
- **Strix Halo APUs Get Mixed Reviews**: The price of **Strix Halo APUs** seems to be set at around **$1.6k** for 64GB and **$2k** for 128GB, but some members suggest that EPYC systems offer better value for the price due to greater memory bandwidth and upgradeability.
   - One member lamented *soldered memory* on such devices, drawing a comparison to a recent DIMM failure on a server and pointing to a [Corsair AI Workstation 300 with Strix Halo APU](https://www.guru3d.com/story/compact-ai-pc-corsair-ai-workstation-300-with-strix-halo-apu/).
- **Threadripper and Epyc Face Off!**: While **Threadripper** is often considered best for consumers, **EPYC** can be a cheaper option due to the availability of refurbished parts, unlike **Threadripper**, which tends to be more expensive and harder to find.
   - A member noted that Epyc is cheaper because *there's quite a market of refurb/sh parts*, and pointed to [this reddit thread](https://old.reddit.com/r/LocalLLaMA/comments/1mcrx23/psa_the_new_threadripper_pros_9000_wx_are_still/) to further discussion.
- **Is Soldered RAM a SCAM?**: Members expressed confusion over why people buy PCs with *soldered storage*, especially given the high prices and limited memory bandwidth, such as **2500€** for a **128GB** soldered RAM device with **256GB/s** memory bandwidth.
   - One user stated *it's like asking to be scammed*, while another likened the concept to a console where everything comes in one package, albeit with the ability to build a better PC for the same price.
- **9070 XT Gets Humble Pie in Performance Tests**: The **9070 XT** is significantly slower than a **4070 Ti Super**, with one user reporting that a model running at **7 t/s** on their **4070 Ti Super** only achieved **3 t/s** on the **9070 XT**; however, another member suggested that RAM bandwidth limitations might be the cause.
   - It was stated that CUDA is good, but maybe Vulkan is not bad either.  One member found a **5070 Ti** for **749 eur**, but the price jumped to **1100 eur** the next day.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1399844353538523288)** (4 messages): 

> `LlamaCloud document agents, LlamaCloud Managed Embeddings, Automated Asset Manager Fund Analysis, LexiconTrail agentic AI systems` 


- ****AI Agents Parse Financial Docs****: Transform complex financial documents into actionable data with **AI-powered document agents** that handle real-world formats like **10-Ks**, earnings reports, and regulatory filings; more information through the [LlamaIndex Webinar](https://twitter.com/llama_index/status/1950285220663742516).
- ****LlamaCloud Manages Embeddings****: **LlamaCloud Indexes** now have managed embeddings, meaning you no longer need to bring your own API key to embed your content; vectors will be embedded for you in addition to hosting them, according to [this Tweet](https://twitter.com/llama_index/status/1950345618779754644).
- ****Automated Asset Manager Fund Analysis Available****: Build an automated asset manager fund analysis with this comprehensive notebook, that shows how to process complex financial documents and extract actionable insights for investment analysis using **LlamaParse** to convert PDF to structured markdown, outlined in [this Tweet](https://twitter.com/llama_index/status/1950590734685671931).
- ****LexiconTrail 10x Agentic AI Systems****: **LexiconTrail** demonstrates how to build **10x faster agentic AI systems** using **NVIDIA Small Language Models** with advanced indexing capabilities, according to [this blog post](https://twitter.com/llama_index/status/1950662723785850911).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1399848962025984000)** (126 messages🔥🔥): 

> `LlamaCloud PDF detection issues, Character AI architecture, Neo4j Knowledge Graph issues, Flowmaker Gemini 2.5 Pro bug` 


- **LlamaCloud can't detect PDFs, member needs direction**: A member reported that **LlamaCloud** could not detect a **PDF file** and process it via API, using **n8n** for workflow simplification and linked to a [screenshot](https://cdn.discordapp.com/attachments/1059201661417037995/1399848961832911031/Screenshot_2025-07-29_at_22.13.16.png?ex=688bcff0&is=688a7e70&hm=b8f51e99fbeae087df203303f7665c4eab8447bb0890b55823fd36074c5ad539&).
- **Character AI Building Discussion sparks**: Members discussed building a **character AI** with deep understanding of a large story, using a classic **RAG** pipeline with chunked text, embeddings, and a vector database.
- **Neo4j Woes and Graph Storage Overload**: A member is trying to implement **Neo4j** as their simple graph storage takes *ridiculously long to load*, but their server is not compatible with **Neo4j 5.x**, and **LlamaIndex** doesn't seem to like **4.x**, and **Aura** is blocked by the server proxy.
- **Flowmaker Fix Flashes Fast for Gemini 2.5 Pro Bug**: A member reported an error when using **Flowmaker** with **Gemini API**, due to an invalid model name and another quickly pointed out that [the model name](https://ai.google.dev/gemini-api/docs/models) required a number, e.g. *gemini-2.5-pro*.
   - A fix was [committed](https://github.com/run-llama/flow-maker/blob/aad0f47a81cacba662a07c4f2d70bd3425606e29/src/lib/llm-utils.ts#L19) and deployed swiftly, resolving the issue, where the member thanked them for their swift assistance.


  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1400095143251677184)** (3 messages): 

> `RAG debugging, Sparse retrieval, Semantic drift, Chunking collapse, Memory breakdowns` 


- **User offers RAG debugging assistance with MIT-licensed Repo**: A member offered help with a **MIT-licensed repo** designed to debug tricky **RAG issues**, including sparse retrieval, semantic drift, chunking collapse, and memory breakdowns.
   - Another member asked to share complex problems solved with the repo, specifically asking for more details on *sparse retrieval* and *semantic drift*.
- **Inquiries Regarding Specific RAG Debugging Issues**: Following the initial offer, a community member inquired about specific complex issues addressed by the MIT-licensed repo, focusing on concrete examples.
   - The inquiry specifically requested detailed instances of how the repo tackles **sparse retrieval** and **semantic drift**, seeking a more granular understanding beyond the general descriptions provided.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1400002443323904084)** (5 messages): 

> `Expert Parallelism (EP) vs Tensor Parallelism (TP), Merge Sort troubles on GitHub` 


- **Expert Parallelism Experiences Explored**: A member is seeking examples where **Expert Parallelism (EP)** outperforms **Tensor Parallelism (TP)**, noting that in their experience with **Qwen32B** and **Qwen 235B**, the added communication from all-reduce operations after attention makes **EP** less performant.
   - They are finding **EP** only useful for models employing **MLA** and requiring **DP attention**.
- **Merge Sort Remainder Rescue Requested**: A member needs help with merge sort remainder issues in their [RinomXE GitHub project](https://github.com/maybeJosiah/RinomXE).
   - They are struggling with the draw order remainder logic, where doubling steps until over shape number fails to sort correctly and posted a javascript snippet for simulation to help.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1399868515422441522)** (17 messages🔥): 

> `Torch Compile, Triton Code Generation, PTX Code Extraction, Inductor Configuration, GEMM Autotuning` 


- **Unlock PTX and Triton Codes from Torch Compile**: To get the **PTX code**, use `TORCH_LOGS="output_code" python your_code.py` or access the `compiled_kernel.asm.keys()` dictionary as detailed in [this blog post](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/?utm_source=chatgpt.com#ttir-triton-ir).
   - The dictionary contains keys for different intermediate representations including **llir, ttgir, ttir, ptx, and cubin**.
- **Bypass Non-Triton Code Generation in Torch Inductor**: To force **Triton code generation** for matmuls, configure settings in [torch._inductor.config.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py#L459-L461), but be aware that not every op is converted to Triton by default.
   - Options like **max_autotune_gemm_backends="TRITON"** and **max_autotune_conv_backends** can influence the autotuning process, although built-in kernels are often faster.
- **Achieve Pure Triton Code by Tinkering Inductor Configs**: To get inductor to *only* use triton code, members recommend modifying `config.py` and `utils.py`, specifically settings such as **use_aten_gemm_kernels**, **use_triton_template**, **autotune_fallback_to_aten**, **max_autotune_conv_backends**, and **max_autotune_gemm_backends**.
   - This involves preventing autotuning and fallback to prewritten kernels, potentially requiring exploration of the **'/tmp/torchinductor_{username}'** directory.
- **TMA Support Arrives with Triton 3.4.0**: **TMA (Tensor Memory Accelerator) support** is not yet available in the official Triton release; users must await version **3.4.0**.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1399872492797038774)** (9 messages🔥): 

> `livestream review, request accepted` 


- **Streamer plans Review Livestream**: A streamer was asked to do a [livestream review](https://www.twitch.tv/) for the community.
   - The streamer responded *I'm not sure if this is the confirmation mail. I'll confirm once I'm free! i doubt it's a welcome to our email list*.
- **Requests Accepted!**: A member stated the team has accepted all requests.
   - A different member confirmed their request was accepted: *Although I'm new to this but something cool to explore*.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1400083758618775636)** (2 messages): 

> `CUPTI metrics in kineto, torch.profiler metrics` 


- **Struggle enabling CUPTI metrics in kineto**: A member inquired about enabling **CUPTI metrics** in **kineto**, possibly through a custom build, within the **torch.profiler**.
   - They referenced a [relevant pull request](https://github.com/pytorch/pytorch/pull/125685), but indicated that it did not resolve their issue.
- **torch.profiler Configuration**: The member attempted to use **torch.profiler** with specific configurations to measure kernel performance.
   - They tried to configure **experimental_config** with **profiler_metrics** such as *kineto__tensor_core_insts*, *dram__bytes_read.sum*, and *dram__bytes_write.sum*, setting **profiler_measure_per_kernel** to True.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1399912770396487731)** (6 messages): 

> `CUDA streams, Megatron-LM, Group GEMM, NYC Hackathon, Beginner Hackathon Tips` 


- **CUDA Streams and GEMM Performance**: A member inquired about the advantages of using **multiple CUDA streams** when running **GEMM kernels**, particularly in the context of **Megatron-LM** and **cuBLAS multi-stream Group GEMM**.
   - The user questioned the benefits compared to a single stream, noting concerns about overhead and the limited number of thread blocks.
- **Hackathon in NYC**: A member asked about a hackathon, and another member pointed them to a specific channel for more info.
   - The hackathon appears to be located in NYC.
- **General Hackathon tips for beginners**: A member shared a [link to general hackathon tips](https://x.com/ayushgun/status/1950444463899512960) on X, noting it's useful for beginners.
   - The tips are very general and not specific to GPUs.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

ali_8366: Anyone here from Montreal? Would love to have a coffee chat
  

---


### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/)** (1 messages): 

vishomaru: Hello, anybody here was successful in profiling compute shaders with AMD GPU Profiler?
  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1399941036360601765)** (3 messages): 

> `AI Hackathon, CuTeDSL Blogpost, Software Pipelining` 


- **AI Hackathon Post Plugged**: A member shared a [LinkedIn post](https://www.linkedin.com/posts/nadiveedishravanreddy_ai-hackathon-qwen-ugcPost-7355265897877434369-7RI5?utm_source=share&utm_medium=member_android) about an **AI Hackathon**.
   - The event now features **15 speakers**, including representatives from **Prime Intellect, Snowflake, and Jane Street**.
- **Course Plug with Star-Studded Speakers Lineup**: A member re-plugged a course, mentioning that it now includes **15 speakers** such as **Prime Intellect**, **Snowflake**, **Jane Street (Sylvain Gugger)**, and **Daniel Han** ([course link](https://maven.com/walk-with-code/scratch-to-scale?promoCode=gpumode40)).
   - They encouraged those with cost concerns to reach out for a discussion about potential assistance.
- **Compiler Automates CuTeDSL Optimization**: A member shared a [blogpost](https://veitner.bearblog.dev/let-the-compiler-do-the-work-in-cutedsl/) and [code](https://github.com/simveit/software_pipelining_cute_dsl) on using **CuTeDSL** for **GEMM on H100**, detailing how to let the compiler handle prefetching.
   - The blogpost explains an experimental argument to the `cutlass.range` operator to hint for prefetching, achieving performance on par with manual prefetching with simpler code.


  

---


### **GPU MODE ▷ #[general-leaderboard](https://discord.com/channels/1189498204333543425/1343002580531417211/1400119615429804166)** (3 messages): 

> `Popcorn-cli DeserializationError, BadCredentialsException on MI300, B200 Timeout Issues, Discord Run Errors` 


- **Popcorn-cli hits DeserializationError on H100 and A100**: A user reported a *"DeserializationError | Raw Error: Deserialization failed because the 'libkernelbot' module is not available in the local environment"* when using the latest **popcorn-cli** version built from source on **H100** and **A100** GPUs.
   - The error is impacting **H100** Discord runs as well.
- **MI300 faces BadCredentialsException**: The user also encountered a *"BadCredentialsException | Raw Error: 401 {"message": "Bad credentials", "documentation_url": "https://docs.github.com/rest", "status": "401"}"* error on the **MI300**.
- **B200 Timed Out**: The user experienced a **300s timeout** on the **B200** for a run that previously completed successfully two weeks prior.
- **Popcorn devs are on the case!**: A member stated that the team is aware of the issues and actively working on a fix.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1400147055963541671)** (5 messages): 

> `Benchmarking Explanation` 


- **Benchmarking Explanation Meeting Delayed**: Members coordinated a meeting to explain the benchmarking process, but the original meeting ended early due to low attendance.
   - A member apologized for oversleeping but confirmed availability for a follow-up meeting to explain benchmarking.
- **Overslept Member Still Available**: Despite oversleeping, a member confirmed they are still available to explain the benchmarking process.
   - The member aims to reschedule and provide the benchmarking explanation as planned.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1399949797603020840)** (20 messages🔥): 

> `gmem synchthreads, cp.async.cg vs cp.async.ca, cutedsl ptx wrapper, nvvm wrapper, cutedsl older drivers` 


- **SyncThreads Savior after Gmem Transfer**: After copying from global memory (**gmem**) to shared memory, manually inserting a `synchthreads` (or equivalent) is necessary.
   - This ensures that all elements in shared memory have arrived before participating in collective calculations like **gemm**, reduction, or scan.
- **Controlling Cp.Async Choice in Cutedsl**: A member asked about controlling whether it’s `cp.async.cg` vs `cp.async.ca` in **cutedsl**.
   - The suggestion was to write custom assembler code and provide it as a copy operation, although this was not tested.
- **PTX Wrapper Revelation in Cutedsl**: There is no API for **ptx** wrapper in cutedsl, according to one member.
   - However, another member shared a link to example code on how to do it and said *In the official CuTeDSL code there is also something.* ([quack/utils.py](https://github.com/Dao-AILab/quack/blob/main/quack/utils.py#L67)).
- **Nvvm Wrapper Navigation Notes**: A member shared a link on how to write **nvvm** wrappers.
   - They shared a link to the [cutlass repo](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/arch/nvvm_wrappers.py) as an example and a link to the [cutedsl docs](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_cpasync.html#module-cutlass.cute.nvgpu.cpasync).
- **Cutedsl Compatibility Concerns Clarified**: A member asked if it's okay to use **cutedsl** with older drivers.
   - They hadn't encountered any issues but wondered if internal testing had revealed any problems.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1400002892735189022)** (1 messages): 

> `Distributed Training, LLMs, Distributed memory tricks` 


- **Ultrascale Playbook a Fantastic Resource**: The [Ultrascale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) on Hugging Face Spaces is a fantastic resource for distributed training of **LLMs**.
- **Memory optimization for Distributed Training**: The playbook offers a lot of distributed memory tricks for training LLMs.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1399836795881132233)** (38 messages🔥): 

> `GPU Inference vs M3 Ultra, LLMs Offshore with low latency and bad internet, Topological data analysis experts, Speech-LLM models and audio instruction-following capabilities, Manipulating vector embeddings for machine translation` 


- **M3 Ultra as the local inference King**: A member suggested buying an **M3 Ultra** for local inference, linking to a [Reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/1j43ziq/the_new_king_m3_ultra_80_core_gpu_512gb_memory/) discussing its **80-core GPU** and **512GB memory**.
   - Another member shared that they bought a used **M1 16g** citing the lack of response from other members.
- **Running LLMs Offshore with low latency**: A member is trying to get **LLMs running offshore** with low latency and bad internet, seeking suggestions on where to look for solutions.
   - Another member responded that spending hundreds/thousands is an acceptable use case if it's what they want to do.
- **Speech-LLM Audio Instruction Research**: A member expressed interest in conducting open-source research on improving **audio instruction-following capabilities** in **Speech-LLM** models, crucial for creating reliable user interfaces with speech integration.
   - They noted that the latest research is **Alignformer** from Microsoft, but its code is not open source and is gauging interest in collaboration.
- **Vector Embedding Manipulation for Machine Translation**: A member is planning to manipulate a vector embedding in a vector space off based off of a multilingual model.
   - The member wants to take the embedding and add the differences of the mean vectors of both languages, then rotate them till the loss is minimal around the new language mean, which someone else pointed out is a solved problem in [this paper](https://arxiv.org/abs/1309.4168).


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1400101077415886860)** (2 messages): 

> `REST models, Compute cost` 


- **Community Model Payment Protocol Idea Floated**: A member wondered if a community model served via REST could use a **402** response to quote compute cost and enable client auto-payment.
   - They pondered how *single-rail vs. h402 multi-rail affects openness* in such a payment system.
- **Openness Implications**: Discussion revolves around the implications on openness when implementing a **402**-based payment system.
   - Concerns raised about single-rail versus multi-rail approaches.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1399832395791728763)** (17 messages🔥): 

> `In-Context Learning (ICL), Interpretability Tools, Sparse Autoencoders (SAEs), Lucas Critique, Activation Distributions` 


- ****ICL** Might Break **Interpretability Tools**, Claims Spark Concerns**: A member speculated that **in-context learning (ICL)** could potentially break **interpretability tools** like **Sparse Autoencoders (SAEs)** by pushing activations out of the distribution they were trained on.
   - The member referenced the **Lucas Critique**, arguing that interventions (like prompting an LLM) require predictions based on microfoundations invariant to those interventions, and [shared a paper](https://arxiv.org/abs/2501.00070v1) to support their view.
- ****SAEs** Face Generalization Challenges with **ICL****: A member agreed that applying **SAEs** to contexts with significant **ICL** would likely fail because sparse representations don't generalize well to activation distributions they weren't trained on.
   - They clarified that this issue isn't specific to **ICL** but arises whenever **SAEs** are applied to activation distributions different from those they were trained on.
- ****ICL**'s Impact on Activation Distributions: **OOD**?**: A member posited that conditioning a model's behavior to a tiny slice of a large distribution via **ICL** might break diagnostics built for unconstrained circumstances, potentially leading to novel internal behaviors.
   - Countering, another member suggested **ICL** might push activations *in-distribution*, citing examples where **SAE** features activate on specific instances within a context, pointing to papers on *function vectors/task vectors*.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1399902669526663381)** (1 messages): 

> `Model Evaluation Metrics` 


- **Debugging Model Evaluation Metrics**: A member offered to assist in debugging a function that takes a single input document and model predictions to return evaluation metrics.
- **Understanding Function Processes**: The suggestion involved understanding how the function processes data to identify potential issues.


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1399859285504168006)** (1 messages): 

> `Diffusion Models Study Group, Flow Matching, MIT Curriculum` 


- **New Diffusion Models Study Group Launches**: A new **12-person, 5-month study group** is starting to learn diffusion models from scratch, based on **MIT’s curriculum** ([lecture notes](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf)).
   - The group is designed for those working in AI and includes CTOs, AI instructors, and AI researchers.
- **Attend Free Intro Sessions on Flow Matching and PDEs**: The study group is hosting **two free intro sessions** on **August 2nd** ([Flow Matching & Diffusion Models](https://lu.ma/kv8zf6va)) and **August 9th** ([PDEs, ODEs, SDEs + A Brief History of Diffusion Models](https://lu.ma/uk6ecrqo)), both at 12 PM EST.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1399899213365645353)** (4 messages): 

> `MoE Implementation, grouped_mm, Low Precision Training, Float8 Training` 


- **Grouped GEMM Implementation Talk**: A member inquired about a PR supporting **torch._grouped_mm** in **GPT-NeoX**, now available in PyTorch core, for potential performance benefits, specifically mentioning [this MoE implementation](https://github.com/EleutherAI/gpt-neox/blob/d12c771198388980ee054617e537665f044e0584/megatron/model/moe_mlp.py#L221).
   - They said that users interested in **low precision MoE training** could use a one liner from TorchAO.
- **Digging Into PyTorch's Grouped GEMM Implementation**: A member asked about the underlying implementation of PyTorch's **_grouped_mm** and requested performance comparisons with megablocks grouped GEMMs.
   - Another member pointed out that it uses a **CUTLASS kernel** under the hood, linking to the [relevant source code](https://github.com/pytorch/pytorch/blob/62f98dbb44fb338ba849f93c491ea170af4c187c/aten/src/ATen/native/cuda/GroupMM.cu#L418).
- **Float8 Blockwise Pretraining Renaissance**: One member questioned the perceived lack of interest in **low-precision training** due to convergence issues, claiming it a *"hard sell unless perf is attractive.*"
   - Another member countered, citing **DeepseekV3's float8 blockwise pretraining** and their own stable convergence results with **FP8 rowwise**, achieving ~30-40% throughput improvement as detailed in [this PyTorch blog post](https://pytorch.org/blog/accelerating-large-scale-training-and-convergence-with-pytorch-float8-rowwise-on-crusoe-2k-h200s/).


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1400078020387147887)** (7 messages): 

> `CUDA generalization paper, TLS handshake EOF error, Mojo package installation, Region-specific access issues` 


- **CUDA Generalization Paper Shared**: A member shared a [paper on generalizing beyond CUDA](https://huggingface.co/papers/2507.14111).
   - Another member thanked them for sharing and suggested posting such content to the appropriate channel in the future.
- **User Faces TLS Handshake EOF Error**: A new Mojo user reported encountering a **TLS handshake EOF error** while trying to run a Mojo project through both **pixi** and **magic shell**.
   - They noted issues with installing a **dockerfile** obtained from Microsoft Copilot.
- **Suggested Solution for TLS Handshake Issues**: A member suggested that the **TLS handshake issues** could be related to region-specific access to package repositories and provided a solution to try the new `mojo` package in the latest nightly using **pixi**.
   - The suggested commands were: `pixi init my-project -c https://conda.modular.com/max-nightly/ -c conda-forge`, followed by `cd my-project` and `pixi add mojo`.
- **VPN Fails to Fix Installation Issue**: The user who encountered the **TLS handshake EOF error** reported that the suggested solution did not work, even with a VPN.
   - Another member mentioned that region issues should be resolved since a migration to a new host a few months prior.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1399916351182868503)** (41 messages🔥): 

> `Mojo external calls vs libc, Mojo to Python overhead, Embedding CPython in Mojo binaries, Python performance, Mojo and hot loops` 


- **Mojo's external calls raise questions**: Users questioned why Mojo's `external_call` uses specific functions like `KGEN_CompilerRT_IO_FileOpen` instead of standard `fopen` from libc, and whether this is for safety.
   - A member clarified that many of these are artifacts from when Mojo was less capable and are not a high priority to fix right now, and that the KGEN namespace belongs to Modular and will be opened up eventually.
- **Mojo to Python overhead seems steep**: A user found that calling a Python no-op function from Mojo is significantly slower (4.5 seconds for 10 million calls) compared to calling it directly from Python (0.5 seconds), noting that this difference is more significant than Python to Rust interop with Pyo3.
   - Others chimed in pointing out that Mojo needs to start a CPython process to execute Python functions, incurring overhead, and comparing it to Rust to Python interop is not an equal comparison.
- **Embedded CPython is in the binary**: Discussion arose whether CPython is embedded in the Mojo binary or if the Python code is compiled, affecting the performance overhead of Python calls from Mojo.
   - It was clarified that CPython is embedded via `dlopen libpython` and a pointer to the interpreter is maintained, and every call reuses the same interpreter, so you shouldn't call it in a hot loop for performance reasons.
- **Low latency trumps hot loops in mojo**: Discussion on performance impact of calling Python from Mojo, especially in hot loops for tasks like OpenCV or Mujoco robot simulation, emphasizing that doing so is like *spraying glue in a race car engine*.
   - Members note that a lot of fast python libs are actually C libs with a wrapper, and that *interacting with the context dicts alone can easily eat several hundred cycles*.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1399878169456152708)** (38 messages🔥): 

> `Aider site framework, Deepseek-chat OpenRouter Issues, SWE-bench Leaderboard, Aider's Role in Model Training, Qwen3 Coder 30B-A3B announcement` 


- **Deepseek-chat Dumbs Down on OpenRouter**: Members noticed **Deepseek-chat** performing worse on **OpenRouter** compared to the official **DeepSeek API**, returning entire functions instead of selective diffs when used as the editor model in architect mode.
   - Using `aider --model openrouter/deepseek/deepseek-r1` was suggested as a fix, because that ensures usage of the default config at [aider/resources/model-settings.yml](https://github.com/Aider-AI/aider/blob/main/aider/resources/model-settings.yml#L548) which has the `edit-format: diff` setting.
- **Aider Could Train Coding Models**: It was suggested that **Aider** could assist with coding model training by logging where linting or undo actions were needed during development workflows.
   - This would leverage Aider's use in *careful* development to provide valuable training data, though the commenter did not advocate for the developers to implement this feature.
- **Qwen3 Coder 30B-A3B Announced**: An image was shared of the new **Qwen3 Coder 30B-A3B**.
   - The image in question was a screenshot of the announcement, validating its legitimacy.
- **User Encounters Litellm API Connection Errors**: A user reported encountering numerous `litellm.APIConnectionError: Error parsing chunk: Expecting property name enclosed in double quotes: line 1 column 2` errors.
   - The errors didn't seem to impact functionality.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1400180575096012831)** (3 messages): 

> `Open Model Selection, Hardware Considerations for Aider, Runpod Credits, R1 Model, Qwen Coder Model` 


- **Open Model Showdown: R1 vs Qwen Coder**: A member sought advice on the best open model to use with **aider**, given unlimited hardware and pondered testing **R1** and **Qwen Coder** models.
   - The member mentioned having **Runpod credits** to burn, suggesting an intention to conduct practical tests with these models.
- **Llama3 faces Integration Discussions With Aider**: Members had integration discussions regarding **Llama3** and compatibility issues with Aider.
   - Some members suggested some helpful integration improvements to existing model options.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1399838202122207312)** (29 messages🔥): 

> `LLM Safety Alignment Research, AI alignment blogs, CUDA with Claude, Z.AI 54 open source repos, Math in paper discussion` 


- **LLM Safety Alignment Research Resources Sought**: A PhD student sought suggestions on how to catch up on current **LLM safety/alignment research**, particularly good survey papers.
   - Suggestions include four blogs from **AI alignment forum**, including [Thought Anchors: Which LLM Reasoning Steps Matter](https://www.alignmentforum.org/posts/iLHe3vLur3NgrFPFy/thought-anchors-which-llm-reasoning-steps-matter), [Measuring Beliefs of Language Models During Chain-of-Thought](https://www.alignmentforum.org/posts/a86uAnPykqNtmEbDH/measuring-beliefs-of-language-models-during-chain-of-thought-1), and more.
- **CUDA Coding with Claude causes Complexities**: A member found that writing in **CUDA** with the help of **Claude** is complex, requiring planning, deep understanding, and organization.
   - They suggested that the real test of intelligence would be if a Python developer with some GPU and CUDA knowledge could use **Claude** to navigate writing **kernels** and optimize performance, and included an [image](https://cdn.discordapp.com/attachments/986699377257119794/1400233738369110026/image.png?ex=688be4ca&is=688a934a&hm=1bcb11346477e61edf05cde9751d5e62ee8992a2f64216c07e4a1a8f8fb14cc4).
- **Z.AI 54 Open Source Repos Spark Interest**: A member asked if others have seen the new **Z.AI 54 open source repos** and tried looking into them.
   - No further details were provided about the repos or their specific content.
- **Love expressed for Voyager**: A member shared a link to [Voyager](https://www.youtube.com/watch?v=H0XYANRosVo) exclaiming their love for it, another immediately said "me too!".
   - Another member shared a link to a [Simons Foundation](https://www.youtube.com/playlist?list=PLWAzLum_3a18wO6C7TP8_4XGw4pDxy6G5) playlist that had dropped today.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1399983269071818853)** (1 messages): 

> `Qwen3, GPT-4o` 


- **Qwen3 30B Matches GPT-4o Performance**: A member shared a post that [Qwen3 30B A3B 2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) is on par with **OpenAI GPT-4o** in English and Chinese.
- **Qwen3 Gains Traction**: Community members are excited about the potential of **Qwen3** as a strong competitor in the language model arena.
   - Early benchmarks suggest it may offer comparable performance to **GPT-4o** in certain tasks, particularly in multilingual contexts.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1399833974972350474)** (30 messages🔥): 

> `Kimi chatbot, Moonshot AI vibe, OpenHands, Training dataset of Kimi, Scale AI` 


- **Kimi Chatbot Replies in Chinese**: A user reported that the **Kimi chatbot** replied in **Chinese** despite asking questions in **English**, suggesting that it may be related to being logged out.
   - Screenshots revealed that while the reply was in English, the suggested sources and questions were in Chinese.
- **Kimi's Training Data Leans on Social Media**: One member joked that **Kimi's training dataset** seems to include **Instagram** and **TikTok** comments, suggesting it's why it's good.
   - They linked to [Kimi on OpenHands v0.50.0k2](https://github.com/All-Hands-AI/OpenHands/releases/tag/0.50.0k2) to support this claim.
- **Moonshot AI Vibe**: One member said that *moonshot got the best vibe no cap*, while another agreed that the community needs some competition.
   - They linked to [X post](https://x.com/crystalsssup/status/1944287779896328668) about the AI community vibe check.
- **Alexandr Wang Founder of Scale AI**: A member mentioned **Alexandr Wang** is the **founder and CEO of Scale AI**, which is a data infrastructure company.
   - They pointed out that **Scale AI** provides training data, annotation, and evaluation services that are essential for developing machine learning models.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1399839990069854348)** (25 messages🔥): 

> `Lume vs Suna, Manus' Comic Creation, The Future of Manus` 


- **Members Debate: Lume edges out Suna**: Members debated the merits of the **Lume** and **Suna** agentic systems; one member stated that *Lume did a much better job* to code out specific things and with less mistakes, but conceded they may not have prompted Suna correctly.
   - The member noted they couldn't compare to **Manus** due to prohibitive costs for certain tasks.
- **Manus Comic Creation: A Diamond in the Rough?**: One member suggested that **Manus**' comic creation feature is nice but still can be improved.
   - Another member stated the service was declining in quality, with restrictive limits for free users, and questions whether **Manus is dead**.
- **Optimistic AI vs. Skeptical Human: The Future of Manus**: One member asked an AI what it thinks about the future of **Manus**, and it replied *I think the future of Manus is bright*.
   - Another member expressed skepticism, citing the release of agent modes from **OAI** and **Google**.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1399834491513475083)** (22 messages🔥): 

> `MCP Server Security, BDD Testing with LLMs and MCP, Windows-MCP issues with CursorTouch and Claude, FastMCP tool selection, Hosted MCP server` 


- **MCP Server Needs User-Context Separation**: A user is seeking clarification on whether a single cloud-deployed **MCP server instance** requires additional layers for **user-context separation** to prevent data sharing between unique sessions when accessed by multiple clients simultaneously, referencing [issue #1087](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1087) and [MCP Docs](https://modelcontextprotocol.io/docs/tutorials/use-remote-mcp-server#understanding-remote-mcp-servers).
- **Cursor Connects, Claude Fails**: A user reported successfully deploying an **MCP server to EC2** with properly configured **SSL certification and domain settings**, but they can only connect via **Cursor** and not **Claude Desktop**.
- **Cucumber, BDD, and LLMs Team Up!**: One member shared a side project based on **Behavior-Driven Development (BDD)** that is production-ready; they also included a [diagram of the solution](https://cdn.discordapp.com/attachments/1312302100125843479/1399854833565044756/bael.jpeg?ex=688bd568&is=688a83e8&hm=2e86139e9f117265cd7cbef2afcc1a23a34a091e79402df9a0e051261231c695&).
- **Windows-MCP State Tool Tantrums**: One user is struggling with Windows-MCP by CursorTouch in Claude desktop because the **state tool doesn't work at all** and states: *Error calling tool 'State-Tool': 'Taskbar'*.
- **FastMCP Dynamic Tool Selection**: A user inquired whether **FastMCP** includes logic for **dynamically and automatically selecting tools** (e.g., math, web search, RAG, data interpreter) on the client side when multiple tools are defined on the server.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1399914906672959628)** (2 messages): 

> `DeepTrail, Deepsecure, Open Source Auth, Delegation Layer for AI agents, Secure Multi-agent workflows` 


- ****DeepTrail** Introduces **Deepsecure** for AI Agent Authorization**: **DeepTrail**, backed by Berkeley SkyDeck, is developing **Deepsecure**, an open-source auth and delegation layer for AI agents, enabling integration of authorization, agent-to-agent delegation, policy enforcement, and secure proxying with minimal code via [GitHub](https://github.com/DeepTrail/deepsecure).
- **Exploring **Deepsecure's** Architecture and Secure Multi-Agent Workflows**: **Deepsecure's** architecture features a split-key design, gateway/proxy, separate control/data plane, policy engine, and macaroons for agent-agent delegation, detailed in its [technical overview](https://github.com/DeepTrail/deepsecure/blob/dev/docs/design/deepsecure-technical-overview.md).
- **Examples of **Deepsecure** Integration with Langchain/LangGraph**: Examples of **Deepsecure** integration with Langchain/LangGraph: *Secure Multi-agent workflows* ([code link](https://github.com/DeepTrail/deepsecure/blob/dev/examples/05_langchain_secure_tools.py)), *Delegation Workflow* ([code link](https://github.com/DeepTrail/deepsecure/blob/dev/examples/09_langchain_delegation_workflow.py)), *Advanced Delegation Patterns* ([code link](https://github.com/DeepTrail/deepsecure/blob/dev/examples/11_advanced_delegation_patterns.py)), and *Platform Agent Bootstrapping* ([code link](https://github.com/DeepTrail/deepsecure/blob/dev/examples/12_platform_expansion_bootstrap.py)).
- **Premium Directory with Community Features and Marketplace**: A member started working on what is intended to be *a premium directory, with community features and evolve into a 1-click-install and even marketplace*, at [protocoldepot.dev](https://protocoldepot.dev/).


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1399896190509912094)** (18 messages🔥): 

> `DSPy learnable parameters proposal, Signature implementation using f-strings, DSPy vs GEPA` 


- **DSPy Learnable Parameters Proposal Sparking Interest**: Members discussed a proposal for adding learnable parameters (`dspy.variable` or `dspy.parameter`) to DSPy, and [an issue was created](https://github.com/stanfordnlp/dspy/issues/8593) to gather ideas and use cases.
   - One member described it as a *really shiny proposal*, hoping to allow *templates to be parameters/variables* so the optimizer can spit out optimal prompts, as well as template variable placement.
- **F-Strings Cause Signature implementation Problems**: A member asked for help implementing a signature using an f-string, wanting to verify a code against a description.
   - Another user recommended against this approach and suggested *putting the parameter description within `dspy.InputField()`*.
- **DSPy Faces Off Against GEPA in Prompt Optimization Throwdown**: A member noted a YouTube video where **DSPy** was compared to **GEPA**, with the hot take *DSPy optimizes the prompt you gave it; GEPA evolves a prompt you never imagined* and linked the [YouTube video](https://www.youtube.com/watch?v=o6RbVPFOslg).
   - The same member proposed *turning MIPRO into a reflective, genetic-style frontier engine* for DSPy to spawn and maintain a Pareto-frontier of prompts, aiming to disprove the YouTuber's claim.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1400072601677856810)** (15 messages🔥): 

> `AMD vs Nvidia for gaming, Qwen coding model release, RLVR discussion` 


- **AMD: The Way to Game and Blaze New AI Trails**: One member suggested getting the **7900XT** instead of the **9070** for gaming, pairing it with a **7800X3D** rather than a **9900X**, also noting AMD's consumer AI usability and potential long-term community benefits.
   - They linked to a [tweet](https://x.com/Teknium1/status/1950596567968477382) to bolster their argument.
- **Qwen Thinks Aloud with Coding Model Launch**: A member announced the impending release of the **Qwen3-30B-A3B-Thinking-2507** coding model on [Hugging Face](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507).
   - The link to this Hugging Face model indicates a new tool for code generation.
- **Nvidia's RLVR: Is it Really an RL Algorithm?**: A member questioned the classification of **RLVR** (Reinforcement Learning, Virtual Reality) as a reinforcement learning algorithm, linking to an [NVIDIA tweet](https://fxtwitter.com/NVIDIAAIDev/status/1950279130450444670) prompting the discussion.
   - Another member, teknium, stated *"RLVR is just not a RL Algo its just a target of rl"*.


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1400127686264881253)** (3 messages): 

> `MRC model comparison, Summer school channel request, Senior Software Engineer remote job` 


- **Member Enquires about Summer School Channel**: A new member is asking about a dedicated channel for a **summer school** that took place some weeks ago.
- **MRC Model Comparison Tactics**: A member is asking whether to compare a custom **MRC model** against a large pretrained model's **zero-shot performance**, or to **fine-tune** the large model on the same dataset for a fairer comparison.
- **Long-Term Remote Senior Software Engineer Role Advertised**: A senior software engineer role is advertised, paying **$2K/month**, for a long-term remote contract, located in **Africa** or the **Americas**.
   - The role requires experience in **Ruby on Rails**, **Node.js**, **C#/.NET**, **Python**, **Java**, or similar, as well as strong communication skills in native or near-native English.


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1400077240896590014)** (1 messages): 

> `Langchain-Cohere citation mode, langchain_cohere.ChatCohere` 


- **Citation options do not work on Langchain-Cohere**: A member was having problems to change the citation mode using `citation_options` on `langchain_cohere.ChatCohere`.
   - The member asked if there is any implicit way to pass the citation options since `langchain_cohere.ChatCohere` does not accept it explicitly.
- **Langchain-Cohere repo status: unmaintained?**: A member asked if the [langchain-cohere repo](https://github.com/langchain-ai/langchain-cohere) is an official repo.
   - They noted that the repository have not been updated in the past few months and also inquired *if pull requests are welcomed there*.


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1400056898665058364)** (6 messages): 

> `AI Safety, LLM Bias Mitigation, GPU Kernel Optimization` 


- **Stats Student Seeks Safe Spaces in AI**: A statistics master's student expressed interest in **ML research**, particularly in **technical AI safety**, and is open to research collaborations.
- **PhD Focuses on Ethical LLMs**: A PhD student at JKU Linz Austria is working on **mitigating social bias in LLMs**.
   - Their other interests include **attribution for generative models, AI generated text detection, and domain adaptation**, and they're interested in connecting with people working on practical ethical concerns with domain-specific LLMs.
- **RAGs and Graphs Grab Grad's Growth**: A recent Master's graduate from the Technical University of Munich is working on personal projects to gain more experience with **RAGs**, **knowledge graphs**, and new programming languages.
   - They hope to gain research experience, collaborate on projects, and meet like-minded people to stay on top of new tech.
- **Ali Aces Autoregressive Acceleration**: A member named Ali is working on **optimizing GPU kernels in Triton/CUDA for autoregressive models**.
   - They are always happy to chat about low-level GPU programming.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1399896806522884186)** (2 messages): 

> `LoRA-style adapter in Torchtune, Merged weights in Torchtune` 


- **LoRA-style adapter requested for Torchtune**: A user inquired about **LoRA-style adapter support** in Torchtune, specifically one that retains the exact forward compute path without altering computational cost but freezes original model weights and applies updates through additional trainable layers.
   - They are looking for **additional trainable layers**.
- **Torchtune merges weights after training with an adapter**: A user shared a [link](https://docs.pytorch.org/torchtune/0.6/tutorials/e2e_flow.html) to the Torchtune documentation on **end-to-end workflow**, highlighting that Torchtune supports training with an adapter but merges the weights back in.
   - They are asking questions about merged weights.


  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1400146555914293331)** (2 messages): 

> `ACL Paper Award, Glianorex finetunes` 


- **ACL Paper wins Award**: A member shared their **ACL paper** that just won an award, linked [here](https://aclanthology.org/2025.acl-long.266/).
- **Glianorex Finetunes released**: A member asked if the **finetunes** are public, complaining that their *Glianorex is killing me and my doctor has been no help*.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1400165808369438871)** (2 messages): 

> `Certificate Declaration Form` 


- **Certificate Declaration Form needs completing**: A member was reminded that the certificate declaration form had not been completed.
   - The staff confirmed that *we never got a certificate declaration form from you unfortunately*.
- **Certificate Still Required**: The staff reiterated that the certificate declaration form hadn't been received.
   - The member was previously informed that their form was missing.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1399831422461804604)** (2 messages): 

> `Diffusion Models Study Group, MIT Diffusion Models Curriculum, Flow Matching, Generative AI, AI Education` 


- **Learn Diffusion Models from Scratch in New Study Group**: A new study group is starting a **12-person**, **5-month** program (**2-4 hrs/week**) based on [MIT’s curriculum](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf) to learn diffusion models, which are now a core architecture in generative AI.
   - The first two introductory sessions are free and open to non-members: August 2nd on *Flow Matching & Diffusion Models* and August 9th on *PDEs, ODEs, SDEs + A Brief History of Diffusion Models* ([links here](https://lu.ma/kv8zf6va)).
- **AI Scholars Announces New Diffusion Models Study Group**: AI Scholars is launching a study group on Diffusion Models, with confirmed members including the CTO of an AI film tool, AI art instructor, 2 LLM instructors, and 2 full-time AI researchers.
   - The program features peer-led sessions, mentor Q&A, hands-on projects, real research papers, and a tight, trusted cohort with a weekly format of 2 hours live class + 2 hours self-study, with students rotating teaching.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1399993952987512853)** (1 messages): 

> `Deploying custom language models, Hugging Face deployment, GUI for user queries` 


- **Cloud Deployment Strategy Sought**: A user inquired about how to deploy a language model, trained with a custom folder of PDFs, to the cloud for public use, specifically seeking a simple GUI for user queries.
   - Nomic suggested that the enterprise plan wasn't a good fit, and the user wondered about **Hugging Face deployment** as an alternative.
- **Enterprise Plan Not Suited**: Nomic indicated that the enterprise plan isn't a good fit for the user's needs.
   - The user is exploring alternative deployment strategies, such as Hugging Face, to make their language model accessible.

