---
id: e46d711c-9bcd-4063-a752-d4e2a7ee14a8
title: 'Llama 3.1 Leaks: big bumps to 8B, minor bumps to 70b, and SOTA OSS 405b model'
date: '2024-07-23T01:12:50.598107Z'
original_slug: ainews-llama-31-leaks
description: >-
  Meta’s **Llama 3.1** model weights have leaked ahead of its official release,
  revealing major upgrades including a *128k context length*, *405B dense
  parameter* variant, and significant performance improvements on multilingual
  and coding tasks. Meanwhile, **OpenAI** launched **GPT-4o mini**, a smaller,
  cost-efficient version of GPT-4o priced *over 60% cheaper than GPT-3.5 Turbo*,
  showing strong benchmark results but with some reasoning limitations.
tags:
  - meta
  - openai
  - alibaba
  - ollama
  - llama-3.1
  - llama-3-8b
  - llama-3-70b
  - llama-3-405b
  - gpt-4o
  - gpt-4o-mini
  - qwen-2
  - large-language-models
  - model-leaks
  - multilingual-models
  - long-context
  - synthetic-data
  - math-datasets
  - reasoning-benchmarks
  - cost-efficiency
  - cancer-detection
  - vram-management
  - local-llms
companies:
  - meta
  - openai
  - alibaba
  - ollama
models:
  - llama-3.1
  - llama-3-8b
  - llama-3-70b
  - llama-3-405b
  - gpt-4o
  - gpt-4o-mini
  - qwen-2
topics:
  - large-language-models
  - model-leaks
  - multilingual-models
  - long-context
  - synthetic-data
  - math-datasets
  - reasoning-benchmarks
  - cost-efficiency
  - cancer-detection
  - vram-management
  - local-llms
---


<!-- buttondown-editor-mode: plaintext -->**TODO: ONELINE SUBTITLE**

> AI News for 7/19/2024-7/22/2024. We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**474** channels, and **7039** messages) for you. Estimated reading time saved (at 200wpm): **765 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

We know it's coming tomorrow (with [Soumith's ICML keynote](https://x.com/soumithchintala/status/1814704963332767833)), so we really tried to avoid discussing the leaks since we're going to be covering it tomorrow, but Llama 3.1 is leaking like a sieve ([weights](https://x.com/rohanpaul_ai/status/1815371623815356751), [evals](https://x.com/rohanpaul_ai/status/1815459760227168764), [model card](https://pastebin.com/clone/9jGkYbXY)) that is leaky, so unfortunately it is all the community is talking about today, despite a lot of it being repeats of the [first Llama 3 release in April](https://buttondown.email/ainews/archive/ainews-llama-3/).

Apart from the well telegraphed 405B dense model release, here are the diffs to Llama 3.1 as far as we can tell, mostly from the [model card](https://pastebin.com/clone/9jGkYbXY) spells out the various priorities they had:

- "The Llama 3.1 instruction tuned text only models (8B, 70B, 405B) are optimized for **multilingual** dialogue use cases and outperform many of the available open source and closed chat models on common industry benchmarks."
- Explicitly advertising "Multilingual text **and code** as an output modality
- every model bumped up to **128k context length** (up from 8k)
- Training utilized a cumulative of **39.3M GPU hours** of computation on H100-80GB (TDP of 700W): 1.5m for 8B, 7m for 70B, 31M for 405B.
-  Llama 3.1 was pretrained on ~15 trillion tokens of data from publicly available sources. The fine-tuning data includes publicly available instruction datasets, as well as **over 25M synthetically generated examples**.
- sizable bumps to the 8B and 70B benchmarks (MMLU from 65 to 73 for the 8B (+8 points), and from 81 to 86 for the 70B (+5 points), and MATH from 29 to 52 (+23 points) for the 8B 

We [made a diff spreadsheet](https://x.com/swyx/status/1815553411808653513) to visualize - TLDR, HUGE bump for the 8B, across the board and instruct 70b is mildly better. 405B is still behind flagship models.:

 ![image.png](https://assets.buttondown.email/images/1cd032fd-5219-402d-80a5-bcd95eac43dd.png?w=960&fit=max) 

However some [independently run evals have Llama 3.1 70b doing better than GPT 4o](https://x.com/mattshumer_/status/1815444612414087294) - jury still out.


---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

> all recaps done by Claude 3.5 Sonnet, best of 4 runs.

**GPT-4o Mini Release and Performance**

- **GPT-4o Mini Launch**: [@adcock_brett](https://twitter.com/adcock_brett/status/1815054909378580728) announced the release of GPT-4o mini, a compact and cost-efficient version of the GPT-4o model, with **pricing at 15 cents per million input tokens and 60 cents per million output tokens, over 60% cheaper than GPT-3.5 Turbo**.
- **Strong Performance**: [@swyx](https://twitter.com/swyx/status/1815037679014388172) highlighted GPT-4o mini's impressive performance, achieving **82 MMLU at $0.15/mtok**, outperforming models that were state-of-the-art just 3 months ago.
- **Reasoning Deficits**: [@JJitsev](https://twitter.com/JJitsev/status/1815011239912755392) tested GPT-4o mini on AIW problems and found **basic reasoning deficits and lack of robustness** on simple problem variations, performing worse than Llama-3-8B despite similar compute scale.

**Synthetic Data and Model Performance**

- **Surpassing Teachers**: [@_philschmid](https://twitter.com/_philschmid/status/1814982420602421414) shared that the AI-MO team's winning dataset with a fine-tuned @Alibaba_Qwen 2 model **approaches or surpasses GPT-4o and Claude 3.5** in math competitions, demonstrating the potential of synthetic datasets to enable models to outperform their teachers.
- **NuminaMath Datasets**: [@_lewtun](https://twitter.com/_lewtun/status/1814958635732140336) introduced the **NuminaMath datasets, the largest collection of ~1M math competition problem-solution pairs**, which were used to win the 1st Progress Prize of the AI Math Olympiad. Models trained on NuminaMath achieve **best-in-class performance among open weight models**.

**Reasoning and Robustness Benchmarks**

- **Comprehensive Reasoning Task List**: [@Teknium1](https://twitter.com/Teknium1/status/1815105755613376792) suggested creating a master list of reasoning tasks for people to contribute to, aiding dataset builders in targeting tasks that improve reasoning capabilities.
- **Illusion of Strong Performance**: [@JJitsev](https://twitter.com/JJitsev/status/1815011276684173312) argued that current benchmarks overlook clear deficits in SOTA LLMs, creating an illusion of strong performance for models that manage to score high, despite their inability to perform basic reasoning robustly.

**Memes and Humor in the AI Community**

- **Meme Potential**: [@kylebrussell](https://twitter.com/kylebrussell/status/1815096890595369165) shared a meme, suggesting its potential impact in the AI community.
- **AI-Generated Humor**: [@bindureddy](https://twitter.com/bindureddy/status/1815162164115808691) shared an AI-generated image, highlighting the role of AI in creating humorous content and providing a break from serious topics.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. AI-Powered Mathematics Training**

- **NuminaMath datasets: the largest collection of ~1M math competition problem-solution pairs** ([Score: 53, Comments: 1](https://reddit.com//r/LocalLLaMA/comments/1e8kme3/numinamath_datasets_the_largest_collection_of_1m/)): **NuminaMath unveils massive math dataset**: The **NuminaMath** collection, featuring approximately **1 million math competition problem-solution pairs**, has been released on the **Hugging Face Hub**. This dataset, accompanied by models and a technical report, represents the largest collection of its kind, potentially advancing AI capabilities in mathematical problem-solving.

**Theme 2. Local LLM Resource Optimization**

- **[large-model-proxy allows to run multiple LLMs on different ports of the same machine while automatically managing VRAM usage by stopping/starting them when needed.](https://github.com/perk11/large-model-proxy)** ([Score: 68, Comments: 10](https://reddit.com//r/LocalLLaMA/comments/1e8hges/largemodelproxy_allows_to_run_multiple_llms_on/)): **large-model-proxy** is a tool that enables running **multiple Large Language Models (LLMs)** on different ports of the same machine while **automatically managing VRAM usage**. The proxy dynamically stops and starts models as needed, allowing users to efficiently utilize their GPU resources without manual intervention. This solution addresses the challenge of running multiple memory-intensive LLMs on a single machine with limited VRAM.
  - The author developed **large-model-proxy** to efficiently manage multiple LLMs in their workflow. It automates **VRAM management** and model starting/stopping, making it easier to script and utilize various models without manual intervention.
  - A user points out that **Ollama** offers similar functionality, allowing multiple models to run concurrently with automatic unloading/loading based on VRAM availability, without the need for multiple ports or config file editing.
  - Another developer mentions using **Python** and **OpenResty Lua scripts** to proxy OpenAI API requests and manage LLaMa.cpp instances on demand, expressing interest in the VRAM management aspect of large-model-proxy.


## All AI Reddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**Theme 1. LLaMA 3 405B Model Release and Implications**


- [/r/singularity] **[Looks like we’re getting LLama3 405B this week](https://i.redd.it/4kvx4omgtxdd1.jpeg)** ([Score: 443, Comments: 113](https://reddit.com//r/singularity/comments/1e8wptx/looks_like_were_getting_llama3_405b_this_week/)): **LLaMA 3 405B model imminent**: Meta's AI research team is expected to release the **LLaMA 3 405B** model this week, according to insider information. This new model is anticipated to be a significant upgrade from its predecessors, potentially rivaling or surpassing the capabilities of **GPT-4**.


**Theme 2. AI in Healthcare: Improving Cancer Detection**


- [/r/singularity] **[GPs use AI to boost cancer detection rates in England by 8%](https://www.theguardian.com/society/article/2024/jul/21/gps-use-ai-to-boost-cancer-detection-rates-in-england-by-8)** ([Score: 203, Comments: 25](https://reddit.com//r/singularity/comments/1e8htmy/gps_use_ai_to_boost_cancer_detection_rates_in/)): **AI-assisted cancer detection** in England has led to an **8% increase** in referrals for suspected cancer. **General practitioners** using the **C the Signs** tool have referred **92,000 more patients** for urgent cancer checks over a **two-year period**. This AI system helps doctors identify potential cancer symptoms and determine appropriate next steps, demonstrating the potential of AI to enhance early cancer detection in primary care settings.

**Theme 3. LocalLLaMA Advancements and Applications**



- [/r/StableDiffusion] **[On the same day, two people stole my design and app, and posted them as their own.](https://i.redd.it/xrwxvzxjqtdd1.png)** ([Score: 259, Comments: 195](https://reddit.com//r/StableDiffusion/comments/1e8gter/on_the_same_day_two_people_stole_my_design_and/)): **Plagiarism strikes AI developer**: The creator of a **virtual dress try-on Chrome extension** using **LocalLLaMA** reports that two individuals copied their design and application, presenting them as original work on the same day. This incident highlights the ongoing challenges of **intellectual property protection** in the rapidly evolving field of AI development and application.
  - **Two individuals** allegedly copied OP's **virtual dress try-on Chrome extension**, sparking debate about **intellectual property** in AI development. Many users pointed out that similar products already exist, questioning OP's claim to originality.
  - Users highlighted that the project, using **fal.ai API**, could be recreated in **15 minutes** with basic inputs and a button. This ease of replication raised questions about the value of simple AI implementations and the need for more robust barriers to entry.
  - Discussion centered on the importance of **open-sourcing projects** and properly **crediting ideas**. Some argued that ideas cannot be copyrighted, while others emphasized the need to acknowledge inspirations, even for simple implementations.


---

# AI Discord Recap

> A summary of Summaries of Summaries


**1. LLM Model Releases and Benchmarks**

- **DeepSeek-V2 Tops Benchmarks**: **DeepSeek-V2**, a 236B parameter MoE model (21B activated per token), is praised for its excellent performance and cost-efficiency at $0.14 per input token, outperforming GPT-4 in some areas like **AlignBench** and **MT-Bench**.
   - The model's impressive 1-bit quantization results for **DeepSeek-V2-Chat-0628** showed optimized CPU performance, ranking #7 globally on LMSYS Arena Hard. Users noted its strong performance in multilingual tasks.
- **Llama 3.1 Leak Sparks Excitement**: Leaked evaluations for **Llama 3.1** suggest its 8B, 70B, and 405B models might outperform current state-of-the-art models, even before instruct tuning, with the 70B model noted as being very close to leading models.
   - The leak revealed a 405B model distilled into 8B and 70B versions with 128k context. Community members expressed excitement about potential capabilities, especially after instruct tuning is applied.
  
**2. AI Infrastructure and Optimization**

- **Elon Musk's Memphis Supercluster Announcement**: Elon Musk announced the launch of the **Memphis Supercluster**, claiming it to be the world's most powerful AI training cluster with 100k liquid-cooled H100s on a single RDMA fabric.
   - However, [fact-checks](https://x.com/dylan522p/status/1815494840152662170?s=46) revealed discrepancies in power usage and GPU availability, suggesting that the facility is not yet fully operational as claimed.
- **Advances in Model Quantization**: Discussions highlighted advancements in model quantization techniques, with **AQLM** and **QuaRot** aiming to run large language models (**LLMs**) on individual GPUs while maintaining performance.
   - An example shared was the [AQLM project](https://github.com/Vahe1994/AQLM) successfully running **Llama-3-70b** on an RTX3090, showcasing significant progress in making large models more accessible on consumer hardware.
  
**3. AI Model Performance and Efficiency**

- **Implicit CoT boosts GPT-2 performance**: **[Implicit Chain-of-Thought (CoT)](https://arxiv.org/pdf/2405.14838)** internalizes steps by removing intermediate stages and finetuning, enabling **GPT-2 Small** to solve 9-by-9 multiplication with 99% accuracy.
   - This method also enhances **Mistral 7B**, achieving over 50% accuracy on GSM8K without intermediate steps.
- **ReFT shocks with parameter efficiency**: **ReFT** achieves 15x-60x more parameter efficiency than LoRA and fine-tunes models like **Llama 2 7B** in under a minute on an A10 with ~100 examples.
   - Greg Schoeninger [discussed](https://www.reddit.com/r/MachineLearning/comments/1e8qwnl/r_discussion_of_reft_paper_with_lead_author/) its practical applications and challenges, diving deeper in a [YouTube video](https://www.youtube.com/watch?v=to2oKwnknUk).
- **DeepSeek impresses with 1-bit quant results**: 1-bit quantization for **DeepSeek-V2-Chat-0628** showed impressive CPU optimization, ranking #7 globally on LMSYS Arena Hard ([link](https://huggingface.co/nisten/deepseek-0628-gguf)).
   - *kotykd* queried the model’s coherence and performance changes from previous versions.
    


**4. Knowledge Graphs and Retrieval-Augmented Generation (RAG)**

- **Triplex cuts KG costs by 98%**: **Triplex** from [SciPhi.AI](https://www.sciphi.ai) cuts knowledge graph extraction costs by 98%, outperforming GPT-4 at 1/60th the price by using local graph building with SciPhi's R2R.
   - [R2R](https://r2r-docs.sciphi.ai/cookbooks/knowledge-graph) supports multimodal data and hybrid search, optimizing knowledge graphs and [Microsoft’s method](https://neo4j.com/developer-blog/global-graphrag-neo4j-langchain/) using deeper adjacency matrices for more efficient RAG.
- **Deploying RAG app to production**: A member shared a [tutorial](https://www.mongodb.com/docs/atlas/atlas-vector-search/ai-integrations/langchain/) on using MongoDB Atlas with LangChain to build a RAG implementation.
   - The tutorial covers setting up the environment, storing data, creating search indices, and running vector search queries.
- **Improving RAG via Deasie Workshop**: A [YouTube session](https://t.co/cJPsNaWgoc) with Deasie cofounders covers advanced parsing and metadata for improved RAG.
   - Parsing and metadata enhancements are highlighted as key techniques for boosting RAG performance.
    

**5. Community Contributions and Open-Source Projects**

- **Nemotron-340B bounty sparks interest**: [Nathan](https://x.com/natolambert/status/1814735390877884823) offered a bounty starting at $75 for converting **Nemotron-340B** to **HuggingFace** with **FP8 quantization** and multi-node implementation.
   - The bounty has skyrocketed to **over $2,000**, with considerable interest from the synthetic data community.
- **OpenRouter Provider for GPTScript Now Available**: A new [OpenRouter provider for GPTScript](https://github.com/RobinVivant/gptscript-openrouter-provider) has been announced, with an image and detailed description on GitHub.
   - This tool contributes significantly to the development of GPTScript applications.
- **Bud-E presents new demo with open-source goals**: A demo of the **Bud-E voice assistant** was shared, showcasing the vision of a future where everyone has access to highly capable, open-source systems for the cost of electricity.
   - The code base currently optimized for Ubuntu will be restructured for clean separation between client, server, and interchangeable ASR, TTS, LLM components.


---

# PART 1: High level Discord summaries




## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Implicit CoT boosts GPT-2 performance**: [Implicit Chain-of-Thought (CoT)](https://arxiv.org/pdf/2405.14838) internalizes steps by removing intermediate stages and finetuning, enabling **GPT-2 Small** to solve 9-by-9 multiplication with 99% accuracy.
   - This method also enhances **Mistral 7B**, achieving over 50% accuracy on GSM8K without intermediate steps.
- **ReFT shocks with parameter efficiency**: **ReFT** achieves 15x-60x more parameter efficiency than LoRA and fine-tunes models like **Llama 2 7B** in under a minute on an A10 with ~100 examples.
   - Greg Schoeninger [discussed](https://www.reddit.com/r/MachineLearning/comments/1e8qwnl/r_discussion_of_reft_paper_with_lead_author/) its practical applications and challenges, diving deeper in a [YouTube video](https://www.youtube.com/watch?v=to2oKwnknUk).
- **DeepSeek impresses with 1-bit quant results**: 1-bit quantization for **DeepSeek-V2-Chat-0628** showed impressive CPU optimization, ranking #7 globally on LMSYS Arena Hard ([link](https://huggingface.co/nisten/deepseek-0628-gguf)).
   - *kotykd* queried the model’s coherence and performance changes from previous versions.
- **Graphs boost RAG performance**: **Triplex** from [SciPhi.AI](https://www.sciphi.ai) cuts knowledge graph extraction costs by 98%, outperforming GPT-4 at 1/60th the price by using local graph building with SciPhi's R2R.
   - [R2R](https://r2r-docs.sciphi.ai/cookbooks/knowledge-graph) supports multimodal data and hybrid search, optimizing knowledge graphs and [Microsoft’s method](https://neo4j.com/developer-blog/global-graphrag-neo4j-langchain/) using deeper adjacency matrices for more efficient RAG.
- **QuietStar sparks auto-generated prompts**: *QuietStar* inspired a discussion about LLMs generating subsequent prompts in parallel, aiming to enhance their reasoning capabilities dynamically.
   - Participants debated adapting LLM architectures for better token-level reasoning through intermediate representations and type systems.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hermes 2.5 outperforms Hermes 2**: After adding [code instruction examples](https://link.to.examples), **Hermes 2.5** appears to perform better than **Hermes 2** in various benchmarks.
   - Hermes 2 scored a **34.5** on the MMLU benchmark whereas Hermes 2.5 scored **52.3**.
- **Mistral has struggles expanding beyond 8k**: Members stated that **Mistral** cannot be extended beyond 8k without continued pretraining and [this is a known issue](https://link.to.issue).
   - They pointed to further work on *mergekit* and *frankenMoE finetuning* for the next frontiers in performance.
- **Discussion on Model Merging Tactics**: A member suggested applying the difference between **UltraChat** and base **Mistral** to **Mistral-Yarn** as a potential merging tactic.
   - Others expressed skepticism, but this member remained optimistic, citing successful past attempts at what they termed "cursed model merging".
- **Open Empathic Project Plea for Assistance**: A member appealed for help in expanding the categories of the **Open Empathic** project, particularly at the lower end.
   - They shared a [YouTube video on the Open Empathic Launch & Tutorial](https://youtu.be/GZqYr8_Q7DE) that guides users to contribute their preferred movie scenes from YouTube videos, as well as a link to the [OpenEmpathic project itself](https://dct.openempathic.ai/).
- **SmolLM Arena Launches**: A [new project](https://huggingface.co/spaces/as-cle-bert/smolLM-arena) called SmolLM Arena has been launched, allowing users to compare various Small Language Models (<1.7B params).
   - The arena features a chatbot interface, runs faster, and includes usage instructions for a smoother user experience.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 2.0: Socket Implementation**: Discussions centered around [various socket implementations](https://tigerbeetle.com/blog/a-friendly-abstraction-over-iouring-and-kqueue) for Mojo, focusing on platform-specific challenges like Windows sockets.
   - Members also highlighted the potential of using Rust’s socket implementation and discussed the importance of accommodating SCTP for future protocols.
- **Debate on Dual Stack Sockets**: **Dual stack sockets** were preferred for new server sockets, allowing both IPv4 and IPv6 connections, mirroring Python’s implementation.
   - There was a consensus on using `io_uring` for Linux for handling high-performance workloads.
- **Flat Buffers Shine in Community Meeting**: The [Mojo 🔥 Community Meeting #4](https://www.youtube.com/watch?v=_QVs626Vn2k) covered topics like **Flat Buffers** for memory-efficient serialization and updates on **Forge Tools**.
   - Discussions highlighted optimizing data handling and extending the **Mojo standard library**.
- **Newton's Method for Float Literals**: A member shared an implementation of Newton’s Method for Float Literals in Mojo, prompting detailed discussions on capturing keywords for numerical equations.
   - This led to conversations on closures and solving complex numerical problems within Mojo.
- **Mojo GPU: Eyeing the Summer**: Former XLA team from Google has joined Mojo, bringing new insights into AI infrastructure development.
   - Mojo GPU support is expected to be available this summer, enhancing computational capabilities.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **ComfyUI recommended over Forge**: Multiple users suggested switching from **Forge** to **ComfyUI** for a better experience, citing issues with Forge's functionality and compatibility.
   - Users praised **ComfyUI** for having significantly more tools and features, although noting it has a more complex node-based interface.
- **Comparing Forge to Easy Diffusion**: A user noted that while **Forge** is faster than **Easy Diffusion**, it lacks some features and outputs errors when upscaling.
   - Others commented on upscaling issues and resolutions not being handled properly in **Forge**, suggesting alternatives.
- **Using Latent mode for Regional Prompter**: Guidance was provided on using **Latent mode** instead of **Attention mode** for **Regional Prompter** to prevent character blending.
   - Detailed prompts and clarifications were shared for improving the use of **Latent mode** with multiple character LoRAs.
- **VRAM and GPU compatibility issues**: Discussions covered VRAM requirements for stable diffusion and issues with VRAM on GPUs, particularly with **AMD** cards.
   - Solutions included using **local installations** and cloud GPUs for those with limited home GPU capability.
- **Errors with upscaling in Forge**: Users encountered 'NoneType' errors when upscaling images with **Forge**.
   - Suggestions included switching to **hi-res fix** and alternative upscaling tools like **real-ESRGAN**.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **SVD CUDA Implementation Woes**: A user inquired about how **cusolver/hipSolver** performs SVD as most implementations found were just wrappers for closed-source solutions.
   - A [GitHub repository](https://github.com/Michalos88/Randomized_SVD_in_CUDA) was referenced for insights, mentioning methods like **Gram-Schmidt**, **Householder reflections**, and **Givens rotations**.
- **Building with LLMs: Year in Review**: A [video and blog post](https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/) summarized lessons from a year of practitioners working with LLMs, highlighting tactical, operational, and strategic insights.
   - The author created a visual TLDR to make the series more accessible, emphasizing the value of watching the video for a better understanding.
- **Triton Profiling Tools**: Members discussed tools suitable for profiling Triton kernels, specifically for tasks like peak memory usage.
   - *nsight-compute* and *nsight-systems* were recommended for detailed profiling, with a note that *nvprof* should be avoided as it has been succeeded by these tools.
- **FP16 vs FP32 Performance on A100**: Discussions focused on why **FP16 performance** in A100 is **2x that of FP32**, despite the seemingly expected ratio from computational complexity being around **2.8x** ([Ampere Architecture Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)).
   - Members investigated if the performance issue might be **I/O-bound** rather than computation bound, discussing hardware architecture and overheads.
- **CUDA MODE IRL Invites on September 21st**: **CUDA MODE** team members were invited to an IRL event on **September 21st** in San Francisco, coinciding with PyTorch devcon, including potential talk on **llm.c** for 20 minutes.
   - Logistics and details for the event were shared via [Google Document](https://docs.google.com/document/d/10LkM5_xLh9r_ycul2ywOfgrOGmNP9YDTa9c4V755QgY/edit).



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Parsing GGUF file metadata in C#**: A user announced the parsing of GGUF file metadata in C#, introducing a console tool designed to report metadata and offer statistics.
   - Initially intended as a ShellExtension for file properties, the developer faced registration issues and opted to focus on the console tool.
- **Quantization demystified**: A detailed explanation on the quantization process (q5, q6, f16) aimed to help fit large models on older hardware.
   - Discussions included how q5 is more quantized than q8, with insights on running large models on limited VRAM using devices like a **Dell laptop with RTX 3050**.
- **Hugging Face API disrupts LM Studio search**: LM Studio's search functionality broke due to issues with the Hugging Face API, causing numerous troubleshooting attempts by users.
   - The issue was eventually resolved, restoring search capabilities within the app.
- **Nexusflow launches Athese model**: **Nexusflow** introduced the **Athese model**, showing impressive results and potentially being the current SOTA for its size.
   - The model demonstrates exceptional **multilingual performance**, making it suitable for users beyond English-speaking communities.
- **Creating Discord bot with LM Studio**: A developer shared a [blog post](https://dev.to/mrdjohnson/i-made-a-discord-bot-with-lmstudiojs-4fd6) regarding making a Discord bot with LM Studio.js.
   - The post includes a tutorial and source code available on [GitHub](https://github.com/mrdjohnson/lmstudio-discord-bot), detailing necessary modifications for private responses.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Chrome Restart Fixes Pro Image Generation Issue**: A user resolved an issue of generating only one image with a Pro subscription by [restarting their Chrome browser](https://www.perplexity.ai/page/image-generation-issue). **Pro users can expect smoother image generation after this fix**.
   - Community members noted the need for better error handling within the image generation feature to avoid relying on browser restarts.
- **GPTs Agents Struggle Post Training**: Discussion emerged about GPTs agents being unable to learn from new information after their initial training phase.
   - Suggestions to overcome this included incremental updates and community-driven patches for enhanced adaptability.
- **Perplexity API Clarifications on Token Billing**: **Perplexity API** now charges for both inbound and outbound tokens, as discussed in a recent thread.
   - Users expressed concerns over billing transparency and asked for detailed documentation to better understand these charges.
- **YouTube Tests its Conversational AI Features**: Perplexity AI covered [YouTube's testing](https://www.perplexity.ai/page/youtube-tests-ai-conversationa-WMQ_b8XNQZuIhMpPPMyfGg) of new AI conversational capabilities to assess their efficacy in enhancing user engagement.
   - Initial community responses were mixed, with some excited about the potential for better interaction and others skeptical about the AI's conversational depth.
- **OpenAI Introduces GPT-4.0 Mini**: [OpenAI's GPT-4.0 Mini](https://www.perplexity.ai/page/openai-drops-gpt-4o-mini-viKDYptISzufyJDPoL3Etg) debuted, offering a scaled-down version focused on accessibility without compromising on sophisticated functionalities.
   - Early feedback highlighted its impressive balance between compute efficiency and performance, making it suitable for a wider range of applications.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **4O Mini outshines Sonnet 3.5**: **4O Mini** can solve complex questions that even the entire Claude family struggles with, showing its superior capabilities.
   - This generates excitement among users for the potential of GPT-4o mini to phase out GPT-3.5 and dominate advanced use cases.
- **Finetuning multimodal models for coral reef research**: A user inquired about finetuning multimodal models to study coral reefs, but another suggested using Azure AI's Custom Vision Service for better accuracy.
   - No specific models or datasets were cited, highlighting a need for more tailored recommendations in this area.
- **API lacks real-time voice integration**: A discussion highlighted that OpenAI’s new voice features are available in ChatGPT but not in the API, generating concerns about functional limitations.
   - Members noted the significant latency and quality differences, with opinions leaning towards ChatGPT being more suited for end-user real-time interactions.
- **Improving ChatGPT response modifications**: A user faced challenges instructing ChatGPT to modify only specific text parts without rewriting entire responses, a common issue among users.
   - Suggestions included using the 'Customize ChatGPT' section in settings, and sharing detailed custom instructions for better accuracy.
- **ChatGPT Voice vs API Text-to-Speech**: Concerns were raised about the latency and quality differences between ChatGPT's new voice feature and the API’s Text-to-Speech endpoint.
   - Members suggested potential improvements and alternative solutions but acknowledged the current limitations of the API for real-time application.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Rankings Page Slows Due to Migration**: The rankings page will be slow to update over the weekend and often present **stale data** while the migration to new infrastructure occurs.
   - Users should expect delays and inaccuracies in rankings during this timeframe.
- **OpenRouter Provider for GPTScript Now Available**: A new [OpenRouter provider for GPTScript](https://github.com/RobinVivant/gptscript-openrouter-provider) has been announced, with an image and detailed description on GitHub.
   - This tool contributes significantly to the development of GPTScript applications.
- **Dolphin Llama 70B's Performance Issues**: Using **Dolphin Llama 70B** at 0.8-1 temperature led to erratic behavior in a 7k token context chat, producing incoherent content split between code and unrelated output.
   - Another member noted similar issues with **Euryale 70B's** fp8 quantized models, suggesting potential problems stemming from the quantization process.
- **DeepSeek's Low Cost and Efficiency**: **DeepSeek v2**, a 236B parameter MoE model (21B activated per token), is praised for its excellent performance and cost-efficiency at $0.14 per input token.
   - *“DeepSeek’s pricing is very competitive, and it seems to be hugely profitable,”* explaining their strategy of using high batch sizes and compression techniques.
- **Leaked Information on Llama 3.1 405B**: **Llama 3.1 405B Base** apparently leaked early due to a misstep on HuggingFace, sparking discussions about its extended context capabilities via RoPE scaling.
   - Members are excited, anticipating software updates for efficient utilization and eager for the official instruct model release.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Harnessing LLMs for RPG Games**: The community showed interest in using **LLMs** for classifications, JSON, and dialogue generation in RPG projects, with **CMD-R** emerging as a top choice due to its strict instruction adherence.
   - After successful tests, members discussed further enhancements and integration possibilities, expanding AI's role in RPG gameplay.
- **Structured Outputs in Cohere API**: Cohere announced that **Command R** and **Command R+** can now produce structured outputs in **JSON** format, enhancing integration and data analysis for downstream applications.
   - This new feature is thoroughly documented [here](https://docs.cohere.com/docs/structured-outputs-json) and aims to streamline data workflows for developers.
- **New Enterprise AI Services from Cohere and Fujitsu**: Cohere and **Fujitsu** formed a strategic partnership to deliver new enterprise AI services in Japan, as detailed in their [blog](https://cohere.com/blog/toolkit-features-july-2024).
   - This collaboration targets improved AI service accessibility and performance for various applications, highlighting advancements in the Cohere toolkit.
- **Interactive Multiplayer Text Games with Command R+**: A member introduced **Command R+**, a Discord app for creating and playing multiplayer text games, enhancing the social and interactive aspects of gaming communities.
   - This app was showcased on [Product Hunt](https://www.producthunt.com/posts/create-n-play), offering unlimited possibilities for engaging community experiences.
- **Developer Office Hours 2.0 Launched**: Cohere hosted another Developer Office Hours session, discussing new API features, toolkit updates, and recent **Cohere For AI** research papers.
   - The community was invited to join these sessions to discuss updates, share insights, and connect over various initiatives.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Nemotron-340B bounty sparks interest**: [Nathan](https://x.com/natolambert/status/1814735390877884823) offered a bounty starting at $75 for converting **Nemotron-340B** to **HuggingFace** with **FP8 quantization** and multi-node implementation.
   - The bounty has skyrocketed to **over $2,000**, with considerable interest from the synthetic data community.
- **Hypernetworks and the scaling law debate**: Hypernetworks face constraints due to scaling laws, needing to be **less than O(scaling_law(output_model_compute)(target_error))** to achieve target error.
   - Discussion focused on the necessity for the task predicting the neural network being simpler or having a 'nice' scaling law to be effective.
- **Feature Contamination and OOD Generalization**: A paper on [out-of-distribution generalization](https://arxiv.org/abs/2406.03345) details that neural networks suffer from feature contamination, affecting generalization.
   - Relevant discussions highlighted the significant roles of inductive biases and SGD dynamics in creating a potential unified theory to explain these model failures.
- **Scaling Exponents Across Parameterizations and Optimizers**: A [tweet](https://x.com/main_horse/status/1810647037718999342) about scaling exponents discussed findings across optimizers and models, involving over **10,000 models**.
   - Key insights: **O(1/n) LR schedule** outperformed mUP, successful **hparam transfer** across configurations, and a new **Adam-atan2 optimizer** was proposed to avoid gradient underflow issues.
- **MATS 7.0 Applications Open**: **Neel Nanda** and **Arthur Conmy** have opened applications for their Winter MATS 7.0 streams, with a deadline on *August 30th*. [Announcement](https://x.com/NeelNanda5/status/1813921161052635209) and [admissions doc](https://tinyurl.com/neel-mats-app) provided.
   - The MATS program emphasizes its unique contribution to fostering mechanistic interpretability research.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Nemotron-4-340B conversion to HuggingFace**: [Nathan Lambert](https://x.com/natolambert/status/1814735390877884823) offers a paid bounty of $75 to convert **nvidia/Nemotron-4-340B-Instruct** to HuggingFace.
   - This effort aims to unlock synthetic permissive data for distillation projects, requiring both FP8 quantization and multi-node implementation.
- **Llama-3 and 3.1 leaks spark excitement**: Rumors and leaks about **Llama-3 405b** and **Llama 3.1** models, including benchmarks, were widely discussed, referencing [Azure's GitHub](https://github.com/Azure/azureml-assets/pull/3180/files) and [Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1e9hg7g/azure_llama_31_benchmarks/).
   - Leaked benchmarks show **Llama 3.1** outperforming GPT-4 in several areas, excluding HumanEval, sparking conversation on its potential superiority.
- **ICML 2024 celebrates Faithfulness Measurable Models**: **Andreas Madsen** announced their ICML 2024 spotlight on a [new approach for interpretability](https://x.com/peterbhase/status/1814692347407429706?s=46): Faithfulness Measurable Models, claiming 2x-5x better explanations and accurate faithfulness metrics.
   - A user pointed out its resemblance to a 2021 NeurIPS paper, emphasizing the need for improved **literature reviews** in submissions.
- **Meta AI's potential premium offerings**: Speculation arose that **Llama 405B** might be part of a premium offering from Meta AI, hinted by snippets of code and a [tweet by Testing Catalog](https://x.com/testingcatalog/status/1815439546722451493?s=46).
   - The buzz includes the possible Meta AI API platform, AI Studio, with announcements expected on July 23.
- **Surprising Effectiveness of UltraChat**: Discussion noted that the **Zephyr paper** significantly filtered UltraChat data from **1.5M** to **200k**, questioning the data quality.
   - Despite the rigorous filtering, **UltraChat** was surprisingly effective, leading to further inquiries about its generation process.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Langfuse Outshines Langsmith**: Anecdotal feedback from users suggests that [Langfuse](https://github.com/langfuse/langfuse) is performing better than Langsmith, with positive experiences shared about its ease of self-hosting and integration.
   - *Clemo_._*, the founder, encouraged more community interaction, emphasizing their commitment to maintaining a great OSS solution.
- **GPT-4o Mini Enables AI-generated Content**: OpenAI's new [GPT-4o mini model](https://batchmon.com/blog/ai-cheaper-than-ads/) costs $0.15 per 1M input tokens, making it possible to create dynamic AI-generated content supported entirely by ads.
   - Discussion includes the potential impact on web content, hypothesizing a shift towards more AI-generated outputs.
- **Harvey AI Rumors and Predictions**: Rumors and skepticism about [Harvey AI's](https://x.com/emilyinvc/status/1814741780010844289?s=46) viability emerged, with some calling it a 'smoke and mirrors company'.
   - Debates ensued about the challenges facing vertical AI startups, including dependency on big AI labs and the industry's current cycle.
- **Elon Musk's Memphis Supercluster**: Elon Musk announced the launch of the Memphis Supercluster, claiming it to be the world's most powerful AI training cluster with 100k liquid-cooled H100s on a single RDMA fabric.
   - However, [fact-checks](https://x.com/dylan522p/status/1815494840152662170?s=46) reveal discrepancies in power usage and GPU availability, suggesting that the facility is not yet fully operational.
- **LLaMA 3.1 Leaks Spark Excitement**: Leaked evaluations for [LLaMA 3.1](https://x.com/mattshumer_/status/1815444612414087294?s=46) suggest that its 8B, 70B, and 405B models might outperform current state-of-the-art models, even before instruct tuning.
   - These leaks have led to widespread anticipation and speculation about the future capabilities of open-source AI models.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Triplex Cuts KG Costs by 98%**: SciPhi's new [Triplex model](https://huggingface.co/SciPhi/Triplex) slashes knowledge graph creation costs by 98%, outperforming **GPT-4** at a fraction of the cost.
   - The model extracts triplets from unstructured data and operates locally, making affordable, accessible knowledge graphs a reality.
- **Mistral 12b Tokenizer Troubles**: Multiple members raised issues with **Mistral 12b**'s tokenizer, which outputs text without spaces despite its promising metrics.
   - The outputs were criticized as 'garbage', likely tied to special token handling problems.
- **LLaMA 3.1 Benchmarks Impress**: Members lauded the benchmarks for **LLaMA 3.1**, highlighting stellar performances by the 8B and 70B models.
   - The 70B model was noted to be particularly **very close behind the leading models**, even outperforming some expectations.
- **DeepSpeed Zero-3 Compatibility Fix**: A user solved **DeepSpeed Zero-3** compatibility issues involving a **ValueError** tied to `low_cpu_mem_usage=True` and custom `device_map` settings.
   - The problem was fixed by deleting the accelerate config, resuming error-free setup.
- **Axolotl Training Hits GPU Roadblocks**: Training errors in Axolotl traced back to GPU memory roadblocks, as noted by **Phorm**.
   - Troubleshooting steps included reducing batch size, adjusting gradient accumulation, and switching to mixed precision training.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Hermes 2.5 outperforms Hermes 2**: After adding [code instruction examples](https://link.to.examples), **Hermes 2.5** appears to perform better than **Hermes 2** in various benchmarks.
   - Hermes 2 scored a **34.5** on the MMLU benchmark whereas Hermes 2.5 scored **52.3**.
- **Triplex slashes knowledge graph creation costs**: The newly open-sourced [Triplex](https://huggingface.co/SciPhi/Triplex) by SciPhi.AI reduces knowledge graph creation costs by 98%, outperforming GPT-4 at 1/60th the cost.
   - Triplex, a finetuned version of Phi3-3.8B, extracts triplets from unstructured data, enhancing RAG methods like Microsoft's Graph RAG.
- **Deploying RAG app to production**: A member shared a [tutorial](https://www.mongodb.com/docs/atlas/atlas-vector-search/ai-integrations/langchain/) on using MongoDB Atlas with LangChain to build a RAG implementation.
   - The tutorial covers setting up the environment, storing data, creating search indices, and running vector search queries.
- **LangChain beginner-friendly article published**: A user shared a [Medium article](https://medium.com/@ambaliaharshit25/ea17820a5c01) about LangChain and its components, aimed at beginners interested in understanding its applications.
   - *Imagine having a virtual assistant that can handle complex tasks through simple natural language commands*, the article delves into why these components are important.
- **AI-powered function builder for TypeScript**: A new project called [AI Fun](https://github.com/mishushakov/ai-fun) was developed at a hackathon to build LLM-powered functions for TypeScript.
   - The project leverages AI to automate and simplify TypeScript function building processes.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Bud-E presents new demo with open-source goals**: A demo of the **Bud-E voice assistant** was shared, showcasing the vision of a future where everyone has access to highly capable, open-source systems for the cost of electricity.
   - The code base currently optimized for Ubuntu will be restructured for clean separation between client, server, and interchangeable ASR, TTS, LLM components.
- **Join the BUD-E discord server for collaboration**: Volunteers are invited to join the new **BUD-E discord server** to help develop the voice assistant further and contribute new skills akin to Minecraft Mods.
   - Daily Online-Hackathon meetings will occur every day at 9 PM CEST to onboard new volunteers and coordinate project work.
- **Switch Back to Epochs for Plotting Loss Curves**: A member initially plotted their loss curves with wall-clock time but found it more meaningful to measure model learning efficiency by switching back to epochs.
   - The member found **WandB** useful for this purpose but admitted the initial change was incorrect and a 'foolish' decision.
- **Mem0 Introduces Smart Memory Layer for LLMs**: **[Mem0](https://docs.mem0.ai/overview)** has released a memory layer for Large Language Models, enabling personalized AI experiences with features like user, session, and AI agent memory, and adaptive personalization.
   - For more information on integration and features, view the **[GitHub page](https://github.com/mem0ai/mem0)** for Mem0.
- **Datadog Publishes SOTA Results in Time Series Modeling**: Datadog has published **[state-of-the-art results](https://www.datadoghq.com/blog/datadog-time-series-foundation-model/)** on time series modeling and is actively recruiting for research roles.
   - Datadog's foundation models aim to handle time series data effectively by identifying trends, parsing high-frequency data, and managing high-cardinality data.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **PostgresML Boosts Reranking**: [PostgresML](https://t.co/HWfitT0CJt) for reranking enhances search result relevancy with additional parameters for precise control.
   - A [guest blog](https://t.co/HWfitT0CJt) explains how a managed index approach optimizes reranking in practical applications.
- **LLMs as Production Judges**: A [session](https://t.co/i84Cg5pqsy) with Yixin Hu and Thomas Hulard discussed deploying LLMs as judges in production systems.
   - *This session covered key concepts and practices behind RAG evaluation for development.*
- **Merlinn: Open-source On-call Copilot**: [Merlinn](https://t.co/rAM5OOxQ34) introduces an AI-powered Slack assistant for incident management.
   - *It integrates with observability and incident management tools like Datadog.*
- **Multimodal RAG Simplified with Ollama and Qdrant**: [Pavan Mantha](https://t.co/0gcz4GfCh5) presents an article on setting up multimodal RAG with Ollama and Qdrant.
   - The guide includes steps for ingesting audio/video sources and indexing data through text transcription.
- **Improving RAG via Deasie Workshop**: A [YouTube session](https://t.co/cJPsNaWgoc) with Deasie cofounders covers advanced parsing and metadata for improved RAG.
   - Parsing and metadata enhancements are highlighted as key techniques for boosting RAG performance.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **GPT4o-mini Model Struggles with Verbosity**: **GPT4o-mini** has been reported to be verbose and repetitive, impacting data extraction compared to **GPT3.5-turbo**.
   - This issue causes significant inefficiencies in data pipelines, necessitating better model tuning or alternative approaches.
- **DSPy Tracing Release Enhances Workflow**: The new **DSPy tracing** feature is now available, offering efficient tracking of modules, predictions, LMs, and retrievers ([documentation here](https://docs.langwatch.ai/integration/python/guide#capturing-llm-spans)).
   - This update is expected to streamline debugging and performance tracking significantly.
- **TypedPredictors Compatibility Limited**: **GPT-4o** and **Sonnet-3.5** uniquely handle complex pydantic class generation, whereas other models fall short.
   - This limitation calls for careful selection of models based on project requirements, especially in handling intricate data structures.
- **Joint Optimization in DSPy Yields Big Gains**: A new [DSPy paper](https://x.com/lateinteraction/status/1815423177272824022) reveals that alternating between prompt optimization and finetuning results in up to **26% performance gains**.
   - The study validates the efficiency of a dual optimization strategy over single-method approaches ([paper link](https://arxiv.org/abs/2407.10930)).
- **Reliability of DSPy Optimizers Discussed**: The **BootstrapFewShotWithRandomSearch** optimizer is highlighted as a reliable and straightforward starting point.
   - Members debated the reliability of various optimizers, pointing out **BootstrapFewShotWithRandomSearch** for its simplicity and robustness.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **George Hotz Spurs OpenPilot Insights**: George Hotz shared [OpenPilot model run analysis](https://gist.github.com/geohot/8d7edc7ac2fd9a31ea563c134b66cddb) focusing on documenting kernel changes and potential slowdowns.
   - He mentioned that the task should be accessible for anyone technically inclined but noted that some beginners might overlook initial problem-solving.
- **Debate on Bitcast Shapes in Tinygrad**: Tyoc213 questioned if the `bitcast` function in Tinygrad should align with TensorFlow's `bitcast`, especially regarding shape differences.
   - George Hotz and members agreed that syncing Tinygrad with TensorFlow/Torch/Numpy was sensible and Tyoc213 committed to the necessary updates.
- **Promising PRs in Tinygrad**: George Hotz recognized a [pull request by Tyoc213](https://github.com/tinygrad/tinygrad/compare/master...tyoc213-contrib:tinygrad:tyoc213/bitcast-all) as highly promising and noteworthy for its thorough testing.
   - Tyoc213 appreciated the acknowledgment and revealed plans to align it further with other framework standards.
- **Tinygrad Weekly Meeting Highlights**: Chenyuy shared the agenda for the Monday meeting, detailing updates on tinybox, hcopt speed recovery, and [MCTS search enhancements](https://github.com/tinygrad/tinygrad/blob/master/extra/mcts_search.py).
   - Discussions also included better search features, conv backward fusing, fast Llama improvements, and various bounties aimed at kernel and driver improvements.
- **Debate Over Viability of Tinygrad**: Members debated **Tinygrad's viability** versus PyTorch, with questions on whether to switch now or wait for **version 1.0**.
   - The discussion reflected productivity concerns and was notably fueled by a detailed [YouTube implementation tutorial](https://www.youtube.com/watch?v=g1rCrv1fx1A) on **Shapetrackers**.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **First Day Update at Crowdstrike**: [Vinceflibustier](https://fixupx.com/vinceflibustier/status/1814233715641389456) shared a lighthearted update about their first day at **Crowdstrike**, mentioning they pushed a small update and took the afternoon off.
   - The message ended with a peace sign emoji, creating a casual and friendly tone.
- **Python Subinterpreters in Python 3.12**: A member shared a [tutorial on Python subinterpreters](https://realpython.com/python312-subinterpreters/?utm_source=perplexity) detailing enhancements in **GIL control** and parallelism for Python 3.12 and a preview of changes in 3.13.
   - The tutorial discusses changes to **CPython's global state**, aimed at improving parallel execution and suggests familiarity with Python basics.
- **Meta Llama 3.1 Repository Leak**: [AlpinDale confirmed](https://x.com/alpindale/status/1814814551449244058?s=12) that Meta Llama 3.1 includes a 405B model distilled into 8B and 70B, with 128k context and noted that the 405B model can't draw unicorns.
   - The repository was [accidentally made public](https://x.com/alpindale/status/1814717595754377562?s=46) ahead of time, retaining the same architecture as **Llama 3**, with the instruct tuning possibly being safety aligned.
- **Deepseek Chat v2 6.28 Outperforms Deepseek Coder**: A member mentioned that the *Deepseek chat v2 6.28 update* performs incredibly well, even surpassing the *Deepseek coder* and being more cost-effective than the **4o mini**.
   - The update underscores Deepseek chat v2 6.28's improved performance metrics and cost advantages.
- **Launch of Pinokio's Augmentoolkit on GitHub**: **Pinokio**'s new [Augmentoolkit](https://github.com/pinokiofactory/augmentoolkit) has been released on GitHub for public access, featuring tools to enhance AI applications.
   - The project has gathered momentum across [Discord](https://discord.gg/TQdNwadtE4), [GitHub](https://github.com/pinokiocomputer/pinokio), and [Twitter](https://twitter.com/cocktailpeanut), generating substantial interest.



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Finetuning with GPT models proves costly**: Finetuning GPT models is rare due to **high costs** and **vendor lock-in**. This involves expensive API calls and dependency on specific company infrastructure.
   - A discussion in the #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1264159625255587840) channel highlighted how these factors deter many from adopting finetuning practices.
- **OpenAI credits remain elusive**: Issues in receiving [OpenAI credits](https://link.to/openai-credits) have been reported, with members providing details like organizational ID **org-EX3LDPMB5MSmidg3TrlPfirU** and multiple form submissions.
   - Despite following the process, credits are not being allocated, as detailed in the #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/1264575901673193625) channel.
- **Exploring Openpipe with alternate providers**: Inquiries were made on using **Openpipe** with providers like **Replicate** or **Modal**, aside from **OpenAI** or **Anthropic**.
   - Discussions focused on integrating models from **Replicate** while ensuring compatibility with existing systems, as seen in the #[openpipe](https://discord.com/channels/1238365980128706560/1245927847437008896/1264115136164270090) channel.
- **East Coast Meetup scheduled for late August**: A proposition for a late August meetup in New York was made in the #[east-coast-usa](https://discord.com/channels/1238365980128706560/1245411101718610001/) channel.
   - Members are considering the logistics for this informal gathering.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **OpenAI Scale Tier Confusion**: Discussion on the new [OpenAI Scale Tier](https://openai.com/api-scale-tier) left many puzzled, particularly regarding the throughput per second (TPS) calculations for different models.
   - Queries centered around the calculation of 19 TPS on the pay-as-you-go tier in comparison to GPT-4-o's throughput of about 80 TPS.
- **Websim seeks Founding AI Engineer**: [Websim](https://websim.ai/) is on a mission to create the world's most adaptable platform for software creation, empowering individuals to solve their own challenges.
   - The company is hiring a Founding AI Engineer to establish systems for rapidly iterating on non-deterministic programs targeting automated product development.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **Insightful Year of Building with LLMs**: A user shared a [video and blog post](https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/) summarizing a three-part series on lessons learned by practitioners building with LLMs for a year.
   - The summary highlights tactical, operational, and strategic insights, with a recommendation to consume the content via video for better understanding.
- **BUD-E Voice Assistant Invites Collaboration**: A user shared a [YouTube video](https://youtu.be/O4IXfa8CROs) showcasing a demo of the open-source BUD-E voice assistant, inviting others to join their new Discord server for collaboration.
   - Daily online hackathons will begin at **9 PM CEST** to onboard new volunteers and coordinate project work.



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Artist Aria Calls for Creative Collaboration**: Aria introduced themselves as a **2D/3D artist** looking for collaboration opportunities in the AI community.
   - They invited interested members to **direct message** them for potential partnership projects.
- **No Additional Topics Available**: There were no other significant topics discussed or shared in the provided message history.
   - This summary reflects the lack of further technical discussions, announcements, or noteworthy events.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Clarify Target Audience Needs**: A member raised a query regarding the target audience and the main goals behind the communication strategy.
   - The discussion brought up the need to craft different approaches for engineers, aspiring engineers, devrels, and solution architects when discussing products.
- **Strategic Communication for Roles**: Various communication strategies were explored for effectively engaging engineers, devrels, solution architects, and aspiring engineers.
   - Participants agreed that each role necessitates uniquely tailored messages to clearly convey product features and benefits.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Lessons from 1 Year of Building with LLMs**: [Lessons from 1 Year of Building with LLMs](https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/) detail tactical, operational, and strategic insights from six practitioners.
   - A visual [TLDR video](https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/) accompanies the series, making the lessons more digestible.
- **TLDR Series Heroic Entry**: The TLDR series provides in-depth, actionable advice for those deeply involved with LLMs, as shared by [six authors](https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/).
   - The series is recommended by its authors as a vital resource for LLM practitioners.



---


The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1264137046340009984)** (3 messages): 

> - `Implicit Chain-of-Thought (CoT)`
> - `UltraChat and Multi-turn Interactions`
> - `Multiagent Debate Models` 


- **Achieving High Accuracy with Implicit CoT in GPT-2**: A new method is proposed to internalize **Chain-of-Thought (CoT)** steps by starting with a model trained for explicit CoT, gradually removing intermediate steps, and finetuning it, enabling a **GPT-2 Small** model to solve 9-by-9 multiplication with 99% accuracy ([arxiv](https://arxiv.org/pdf/2405.14838)).
   - The same method improves performance in larger models like **Mistral 7B**, achieving over 50% accuracy on GSM8K without producing any intermediate steps.
- **Seeking Papers on Multi-turn User-Agent Interactions**: A member is seeking recommendations for papers similar to **UltraChat** that discuss building multi-turn user-agent interactions and mentions the **SODA** paper as a potential read ([arxiv](https://arxiv.org/abs/2212.10465)).
   - "The SODA paper is cited in UltraChat for having a similar average number of turns in a given figure."
- **Factuality and Reasoning Improved by Multiagent Debate**: A new approach enhances **factuality and reasoning** in language models through multiagent debate, where multiple language model instances propose and debate responses over multiple rounds ([composable-models](https://composable-models.github.io/llm_debate/)).
   - This method significantly improves mathematical and strategic reasoning across various tasks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://composable-models.github.io/llm_debate/"> Improving Factuality and Reasoning in Language Models with Multiagent Debate</a>: Improving Factuality and Reasoning in Language Models with Multiagent Debate</li><li><a href="https://arxiv.org/abs/2212.10465">SODA: Million-scale Dialogue Distillation with Social Commonsense Contextualization</a>: Data scarcity has been a long standing issue in the field of open-domain social dialogue. To quench this thirst, we present SODA: the first publicly available, million-scale high-quality social dialog...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[datasets](https://discord.com/channels/1053877538025386074/1105324249721356298/)** (1 messages): 

fedorovist: https://huggingface.co/datasets/jdpressman/retroinstruct-mix-v0.2
  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1264698222387925073)** (5 messages): 

> - `PPLX Pro Search AI`
> - `ReFT paper discussion`
> - `Greg Schoeninger's Reddit post on ReFT`
> - `YouTube video on ReFT`
> - `Oxen.ai community and Paper Club` 


- **PPLX Pro Search discovers it's an AI**: A user humorously remarked that **PPLX Pro Search** on discovering it's an AI rather than a human could entertain the idea vice-versa.
- **ReFT paper impresses with parameter efficiency**: ReFT achieves 15x-60x more parameter efficiency than LoRA, fine-tunes models rapidly, such as Llama 2 7B in under a minute on an A10 with ~100 examples.
   - The technique operates on the **residual stream** instead of the K-V matrices, making it both **efficient** and **composable**.
- **Greg Schoeninger's insights on ReFT**: Greg Schoeninger shared his [Reddit post](https://www.reddit.com/r/MachineLearning/comments/1e8qwnl/r_discussion_of_reft_paper_with_lead_author/) discussing the ReFT paper with a link to his Notion notes.
- **YouTube video on ReFT with Zhengxuan Wu**: A YouTube video titled *"How ReFT Works w/ Author Zhengxuan Wu"* dives into the ReFT paper from Stanford, providing both deep knowledge and relatable explanations by Greg.
   - Watch the video on [YouTube](https://www.youtube.com/watch?v=to2oKwnknUk&t=2770s) to understand how the ReFT technique works in detail.
- **Oxen.ai fosters a community of AI enthusiasts**: Oxen.ai promotes a community for academic researchers and developers by hosting **Paper Clubs** every Friday to discuss and apply research papers.
   - Join the community and subscribe to future Paper Club invites on [Oxen.ai](https://oxen.ai/community).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=to2oKwnknUk&t=2770s">How ReFT Works w/ Author Zhengxuan Wu</a>: We dive into the ReFT paper from Stanford with one of the authors Zhengxuan Wu. --Use Oxen AI 🐂           https://oxen.ai/Oxen AI makes versioning your data...</li><li><a href="https://oxen.ai/community">Community Resources | Oxen.ai</a>: Manage your machine learning datasets with Oxen AI.</li><li><a href="https://www.reddit.com/r/MachineLearning/comments/1e8qwnl/r_discussion_of_reft_paper_with_lead_author/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

alexanderlong_84476: https://pluralisresearch.substack.com/p/decentralized-ai-looms
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1263934853448077343)** (434 messages🔥🔥🔥): 

> - `1 bit quant results for DeepSeek-V2-Chat-0628`
> - `New AI projects and model updates`
> - `Llama 3.1 benchmarking`
> - `AI model legality concerns`
> - `Tech tools and deployment experiences` 


- **Insane 1-bit quant results for DeepSeek**: A member shared [crazy results](https://huggingface.co/nisten/deepseek-0628-gguf) for 1-bit quantization of DeepSeek-V2-Chat-0628, optimized for CPU and currently ranked #7 globally on LMSYS Arena Hard.
   - *kotykd* questioned the model's coherence and the differences from previous versions, highlighting specific performance and memory usage data.
- **Hermes achieves recursive function calling**: A member shared that the [Hermes-2-Pro-Llama-3-8b model](https://ollama.com/interstellarninja/hermes-2-pro-llama-3-8b-tools) on Ollama has achieved recursive function calling, with an example provided from a Jupyter notebook.
   - Discussion included potential improvements and configurations for tool calling in similar models.
- **Llama 3.1 set to launch soon**: The Llama 3.1 model is reportedly being launched soon with significant improvements over 3.0, including a 405B distillation for better performance.
   - Members discussed expected benchmarks and features, such as native tool calling in the upcoming 3.1 instruct versions.
- **AI tool deployment and legal gray areas**: Authors shared experiences deploying AI tools, including a member's difficulties with fine-tuning Mistral instruct using chat data and applying the correct templates.
   - Another user raised concerns about hosting a leaked model and potential legal ramifications.
- **New AI research and educational content**: A new blog post titled ['Lessons from 1 year of building with LLMs'](https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/) was shared, providing a TLDR of a year’s worth of practitioner experiences.
   - Members discussed the usefulness of accessible educational content, especially in video format, to make complex concepts understandable.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/_akhaliq/status/1815225121637834802">Tweet from AK (@_akhaliq)</a>: Nvidia presents ChatQA 2  Bridging the Gap to Proprietary LLMs in Long Context and RAG Capabilities  In this work, we introduce ChatQA 2, a Llama3-based model designed to bridge the gap between open-a...</li><li><a href="https://x.com/nightgrey_/status/1815443846265774571">Tweet from Nico (@nightgrey_)</a>: @Teknium1 Also these are apparently scores from the base models, not the instruct tuned ones!  LFG!!</li><li><a href="https://discord.gift/ud3CQyFM2f6M6CdQ">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord is great for playing games and chilling with friends, or even building a worldwide community. Customize your own space to talk, play, and hang out.</li><li><a href="https://x.com/elonmusk/status/1815187468691316946">Tweet from Elon Musk (@elonmusk)</a>: High time for an AI fashion show</li><li><a href="https://huggingface.co/nisten/deepseek-0628-gguf">nisten/deepseek-0628-gguf · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/UCLA-AGI/Gemma-2-9B-It-SPPO-Iter3">UCLA-AGI/Gemma-2-9B-It-SPPO-Iter3 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/mlx-community/Meta-Llama-3-70B-Instruct-4bit">mlx-community/Meta-Llama-3-70B-Instruct-4bit · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/twuK.gif">Cat Keyboard GIF - Cat Keyboard Cats - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/NousResearch/Hermes-Function-Calling/blob/main/examples/ollama_openai_tools_recursive.ipynb">Hermes-Function-Calling/examples/ollama_openai_tools_recursive.ipynb at main · NousResearch/Hermes-Function-Calling</a>: Contribute to NousResearch/Hermes-Function-Calling development by creating an account on GitHub.</li><li><a href="https://www.deepspeed.ai/docs/config-json/">DeepSpeed Configuration JSON</a>: DeepSpeed is a deep learning optimization library that makes distributed training easy, efficient, and effective.</li><li><a href="https://ollama.com/interstellarninja/hermes-2-pro-llama-3-8b-tools">interstellarninja/hermes-2-pro-llama-3-8b-tools</a>: [HERMES TOOL TEMPLATE] Hermes 2 Pro is an upgraded, retrained version of Nous Hermes 2, consisting of an updated and cleaned version of the OpenHermes 2.5 Dataset, as well as a newly introduced Functi...</li><li><a href="https://github.com/exo-explore/exo">GitHub - exo-explore/exo: Run your own AI cluster at home with everyday devices 📱💻 🖥️⌚</a>: Run your own AI cluster at home with everyday devices 📱💻 🖥️⌚ - exo-explore/exo</li><li><a href="https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/">TLDR: 1 year of building with LLMs &#8211; D-Squared</a>: no description found</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1e9hg7g/azure_llama_31_benchmarks/">Azure Llama 3.1 benchmarks</a>: Posted in r/LocalLLaMA by u/one1note • 263 points and 245 comments
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1264297511066144768)** (20 messages🔥): 

> - `Captcha Solving with LLMs`
> - `VRAM Estimates for LLama-3 405B`
> - `Consumer Level Hardware for AI Models` 


- **Challenges in Solving Text Captchas with LLMs**: A member shared their experiences using models like **Florence**, **Moondream**, and **InternVL2** for solving text captchas, noting varying levels of accuracy.
   - While InternVL2 showed the most **success**, the member could not run it locally and had to rely on online demos.
- **VRAM Requirements for LLama-3 405B Quantization**: Inquiries were made about the **VRAM estimates** for running LLama-3 405B at different quantization levels, revealing it requires approximately **410 GB at 8 bit, 205 GB at 4 bit, and 102 GB at 2 bit**.
   - A participant noted this would mean needing at least **9 24GB GPUs** for 4-bit, making it impractical for most consumer setups despite having a server with multiple GPU slots.
- **Frustration over High VRAM Requirements**: There was a desire for more feasible consumer-level hardware capable of running large models like LLama-3 405B locally.
   - Users expressed frustration over the hardware constraints and noted the potential need to explore cloud hosting solutions.


  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1264006054543626293)** (40 messages🔥): 

> - `Triplex for knowledge graph construction`
> - `R2R platform for local LLM experimentation`
> - `Application of knowledge graphs in RAG/LLM`
> - `Microsoft's Graph RAG`
> - `Deeper adjacency matrix and symbolic reasoning` 


- **Triplex revolutionizes knowledge graph creation**: Triplex, a finetuned version of Phi3-3.8B, offers a 98% cost reduction in knowledge graph creation and outperforms GPT-4 by being 1/60th the cost, enabling local graph building with SciPhi's R2R.
   - Developed by [SciPhi.AI](https://www.sciphi.ai), Triplex extracts triplets from unstructured data to create cost-efficient knowledge graphs.
- **R2R bridges gap between local LLMs and scalable RAG**: [R2R](https://r2r-docs.sciphi.ai/cookbooks/knowledge-graph) by SciPhi.AI supports multimodal data ingestion and hybrid search, allowing efficient creation and management of knowledge graphs and documents with a RESTful API.
   - Features include **graph RAG**, observability, app management, and extensibility, facilitating scalable and production-ready applications.
- **Graphs enhance RAG for better conversational AI**: Knowledge graphs enable new forms of analysis in Retrieval-Augmented Generation (RAG) by facilitating symbolic reasoning and improving usability in recommendation systems.
   - **Mixed mode RAG** can leverage graphs for intermediate symbolic reasoning steps, optimizing retrieval of supporting facts from a knowledge base.
- **Microsoft's Graph RAG method revolutionizes subjective datasets**: **Microsoft's Graph RAG** method extends knowledge graphs to create enhanced RAG datasets for more general Q/A tasks, showing potential in handling subjective data.
   - Their technique integrates knowledge graphs into LLMs for deeper context and robust responses.
- **Optimizing knowledge graphs with deep adjacency matrix**: Triplet extraction is a starting point, but deeper **adjacency matrices** are needed to fully leverage LLMs' context lengths.
   - Entity deduplication and resolution will further enhance the accuracy and utility of knowledge graphs in symbolic reasoning tasks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://r2r-docs.sciphi.ai/cookbooks/knowledge-graph.">Introduction - The best open source AI powered answer engine.</a>: no description found</li><li><a href="https://kg.sciphi.ai/">SOTA Triples Extraction</a>: no description found</li><li><a href="https://huggingface.co/SciPhi/Triplex">SciPhi/Triplex · Hugging Face</a>: no description found</li><li><a href="https://neo4j.com/developer-blog/global-graphrag-neo4j-langchain/?utm_source=Twitter&utm_medium=OrganicSocial&utm_campaign=GenAI-RAG--&utm_ID=&utm_term=&utm_content=-DevBlog--&utm_creative_format=&utm_marketing_tactic=&utm_parent_camp=&utm_partner=&utm_persona=">Implementing ‘From Local to Global’ GraphRAG with Neo4j and LangChain: Constructing the Graph</a>: Learn how to combine text extraction, network analysis, and LLM prompting and summarization for improved RAG accuracy.</li><li><a href="https://huggingface.co/datasets/xlangai/BRIGHT?row=1">xlangai/BRIGHT · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1264807709685973024)** (3 messages): 

> - `World Sim`
> - `World Client`
> - `Nous Research` 


- **Introduction to World Sim**: A member asked for an explanation about **World Sim**, which led to a direct link to the [World Sim by Nous Research](https://worldsim.nousresearch.com) being provided.
- **Contributions to World Sim/World Client**: Another member inquired if someone is a **contributor** to the **World Sim/World Client** project.
   - *No detailed community discussion or opinions followed this inquiry.*



**Link mentioned**: <a href="https://worldsim.nousresearch.com">worldsim</a>: no description found

  

---


### **Nous Research AI ▷ #[reasoning-tasks-master-list](https://discord.com/channels/1053877538025386074/1264666760972472481/1264666774624931922)** (691 messages🔥🔥🔥): 

> - `QuietStar`
> - `Auto-generate prompts`
> - `Type systems for LLMs`
> - `Intermediate representations`
> - `Task structure in Open-Reasoning-Tasks` 


- **QuietStar concept inspires auto-generating prompts**: During a discussion about *QuietStar*, members debated how LLMs could benefit from architectures that handle context in parallel, suggesting the idea of models constructing their own subsequent prompts automatically.
   - Participants explored how adapting LLM architectures could enable dynamic prompt generation, enhancing reasoning capacities at the token level.
- **LLM's type system proposal stirs debate**: A member proposed implementing a type system allowing LLMs to construct and verify code, sparking a complex discussion on its feasibility and necessity.
   - Despite objections and confusion, the debate highlighted different perspectives on the importance of formalizing LLM outputs into machine-checkable and expressive languages.
- **Intermediate representations bridge code and natural language**: The community delved into using intermediate representations to manage LLM outputs, balancing between code and natural language, especially for complex tasks like reasoning and refactoring.
   - Discussions underscored the potential of frameworks that translate natural language tasks into structured intermediates to facilitate better programmatic control and verification.
- **Task structure refinement in Open-Reasoning-Tasks repository**: Participants contributed to refining the structure of reasoning tasks in the *Open-Reasoning-Tasks* repository, emphasizing the need for clearer examples and potentially separate files per task.
   - There were considerations about making the task examples more rigorous and the task descriptions more readable and machine-parsable.
- **Various frameworks and tools proposed for LLM reasoning enhancement**: In debates over boosting LLM reasoning capabilities, Prolog, ProbLog, and other logic programming languages emerged as contenders alongside Python for incorporating formal logic into LLM tasks.
   - The conversation highlighted the necessity of probabilistic reasoning and multi-agent systems, inspired by tools like *Logic-LLM* for empirical, multi-framework reasoning.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://operation-athena.repleteai.com/">no title found</a>: no description found</li><li><a href="https://arxiv.org/abs/2401.06751">The Unreasonable Effectiveness of Easy Training Data for Hard Tasks</a>: How can we train models to perform well on hard test data when hard training data is by definition difficult to label correctly? This question has been termed the scalable oversight problem and has dr...</li><li><a href="https://arxiv.org/abs/2402.01825">Fractal Patterns May Illuminate the Success of Next-Token Prediction</a>: We study the fractal structure of language, aiming to provide a precise formalism for quantifying properties that may have been previously suspected but not formally shown. We establish that language ...</li><li><a href="https://x.com/HusseinHElafifi/status/1815107404046233979">Tweet from H (@HusseinHElafifi)</a>: Here&#39;s an adapted version of the things we usually ask of kids:   1. Logical reasoning - Syllogism solving - Truth table completion - Identifying logical fallacies  2. Mathematical reasoning - Wor...</li><li><a href="https://baai-agents.github.io/Cradle/">Cradle: Empowering Foundation Agents Towards General Computer Control</a>: no description found</li><li><a href="https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow">Thinking, Fast and Slow - Wikipedia</a>: no description found</li><li><a href="https://en.wikipedia.org/">Wikipedia, the free encyclopedia</a>: no description found</li><li><a href="https://docs.google.com/document/d/1_AaViP_OrfQ8K256cHAYjS9pbKidCSgEXGaXMAyyWw4/edit">Complex Reasoning Taxonomy</a>: Taxonomy Adapted from: Dimensions of Learning (Marzano &amp; Pickering); The New Taxonomy of Educational Objectives [Marzano &amp; Kendall} ...Address Situations 81 Issues ...Clarify Phenomena &amp; E...</li><li><a href="https://tenor.com/view/eliezer-yudkowsky-george-hotz-ai-alignment-ai-safety-gif-2309383719188990402">Eliezer Yudkowsky George Hotz GIF - Eliezer yudkowsky George hotz Ai alignment - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/_saurabh/status/1763626711407816930">Tweet from Saurabh Srivastava (@_saurabh)</a>: More than 50% of the reported reasoning abilities of LLMs might not be true reasoning.  How do we evaluate models trained on the entire internet? I.e., what novel questions can we ask of something tha...</li><li><a href="https://pastebin.com/fU2fHbBr">master task list - Pastebin.com</a>: Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.</li><li><a href="https://x.com/jd_pressman/status/1815109651220107309">Tweet from John David Pressman (@jd_pressman)</a>: @Teknium1 Some process for making sure the conclusion of an argument changes if one of the necessary steps changes. Chain of thought prompting often has the problem that if you change one of the inter...</li><li><a href="https://x.com/max_paperclips/status/1783412470025187398?t=ymrIzKbiZ-6OfqcpJ-IEsQ&s=19">Tweet from Shannon Sands (@max_paperclips)</a>: I&#39;m currently working on a scripting language that&#39;s intended for LLMs rather than humans to use. By doing something similar to this I was able to get Claude out of &#34;rewriting Python&#34; ...</li><li><a href="https://gist.github.com/pipinstallyp/ba91773cba35a0e30c9dc26101e74dde">latex_draft.md</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://www.overleaf.com/latex/templates">Templates - Journals, CVs, Presentations, Reports and More - Overleaf, Online LaTeX Editor</a>: An online LaTeX editor that’s easy to use. No installation, real-time collaboration, version control, hundreds of LaTeX templates, and more.</li><li><a href="https://huggingface.co/datasets/jdpressman/retro-easy-prose-repair-diffs-v0.1">jdpressman/retro-easy-prose-repair-diffs-v0.1 · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/westoncb/latent-langs/blob/main/ConceptScript/samples/AlzheimersDisease.txt">latent-langs/ConceptScript/samples/AlzheimersDisease.txt at main · westoncb/latent-langs</a>: Contribute to westoncb/latent-langs development by creating an account on GitHub.</li><li><a href="https://github.com/NousResearch/Open-Reasoning-Tasks">GitHub - NousResearch/Open-Reasoning-Tasks: A comprehensive repository of reasoning tasks for LLMs (and beyond)</a>: A comprehensive repository of reasoning tasks for LLMs (and beyond) - NousResearch/Open-Reasoning-Tasks</li><li><a href="https://github.com/probcomp/hfppl">GitHub - probcomp/hfppl: Probabilistic programming with HuggingFace language models</a>: Probabilistic programming with HuggingFace language models - probcomp/hfppl</li><li><a href="https://x.com/VivekGRamaswamy/status/1815093862748303795">Tweet from Vivek Ramaswamy (@VivekGRamaswamy)</a>: Best way to predict the future: just follow the incentives. It’s shocking how precisely right you can be, right down to the exact timing.</li><li><a href="https://huggingface.co/datasets/jdpressman/retroinstruct-mix-v0.2">jdpressman/retroinstruct-mix-v0.2 · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks">lm-evaluation-harness/lm_eval/tasks at main · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/teacherpeterpan/Logic-LLM">GitHub - teacherpeterpan/Logic-LLM: The project page for &quot;LOGIC-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning&quot;</a>: The project page for &quot;LOGIC-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning&quot; - teacherpeterpan/Logic-LLM</li><li><a href="https://en.wikipedia.org/wiki/Automated_theorem_proving">Automated theorem proving - Wikipedia</a>: no description found</li><li><a href="https://github.com/MineDojo/Voyager">GitHub - MineDojo/Voyager: An Open-Ended Embodied Agent with Large Language Models</a>: An Open-Ended Embodied Agent with Large Language Models - MineDojo/Voyager</li><li><a href="https://github.com/METR/public-tasks">GitHub - METR/public-tasks</a>: Contribute to METR/public-tasks development by creating an account on GitHub.</li><li><a href="https://pastebin.com/Ki4kq4GX">• Friedrich August Kekulé: The renowned German chemist, Kekulé was responsible f - Pastebin.com</a>: Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.</li><li><a href="https://huggingface.co/datasets/jdpressman/retro-weave-eval-rubrics-v0.1">jdpressman/retro-weave-eval-rubrics-v0.1 · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/openai/evals/tree/main/evals/registry/evals">evals/evals/registry/evals at main · openai/evals</a>: Evals is a framework for evaluating LLMs and LLM systems, and an open-source registry of benchmarks. - openai/evals</li><li><a href="https://thangs.com/explore">Explore 3D models on Thangs</a>: Explore 3D models from top designers</li><li><a href="https://en.wikipedia.org/wiki/First-order_logic">First-order logic - Wikipedia</a>: no description found</li><li><a href="https://plato.stanford.edu/entries/ryle/#Car">
Gilbert Ryle (Stanford Encyclopedia of Philosophy)
</a>: no description found</li><li><a href="https://en.wikipedia.org/wiki/Martin_Heidegger">Martin Heidegger - Wikipedia</a>: no description found</li><li><a href="https://doi.org/10.3758/s13423-013-0467-3">A taxonomy of inductive problems - Psychonomic Bulletin &amp; Review</a>: Inductive inferences about objects, features, categories, and relations have been studied for many years, but there are few attempts to chart the range of inductive problems that humans are able to so...
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1263935133032124579)** (959 messages🔥🔥🔥): 

> - `LLM fine-tuning`
> - `GPU and hardware capabilities`
> - `Model deployment issues`
> - `Whisper Model for transcriptions`
> - `LLM code and architecture troubleshooting` 


- **Challenges fine-tuning LLMs for different tasks**: Users discussed fine-tuning Large Language Models (LLMs) on different datasets and architectures, including frustrations over limited training time and GPU capabilities.
- **In-depth discussions on GPU capabilities and configurations**: Community members shared their experiences with various GPUs, including RTX 3060, RTX 4070, and H100, for training and deploying AI models, highlighting differences in performance.
- **Exploring available resources for model deployment**: A user inquired about automating speaker diarization and Whisper transcriptions with timestamps for audio files, mentioning previous use of Whisper Large v3 on an RTX 4070.
- **Troubleshooting LLM code and architecture**: Detailed troubleshooting of language models was discussed, focusing on issues like model size, training duration, and specific layers targeted during optimization.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/HochreiterSepp/status/1813189814373462295">Tweet from Sepp Hochreiter (@HochreiterSepp)</a>: xLSTM excels in times series prediction. &#34;Our xLSTMTime model demonstrates excellent performance against state-of-the-art transformer-based models as well as other recently proposed time series mo...</li><li><a href="https://x.com/alpindale/status/1814814551449244058?s=12">Tweet from Alpin (@AlpinDale)</a>: Have confirmed that there&#39;s 8B, 70B, and 405B. First two are distilled from 405B. 128k (131k in base-10) context. 405b can&#39;t draw a unicorn. Instruct tune might be safety aligned. The architec...</li><li><a href="https://x.com/alpindale/status/1814717595754377562?s=46">Tweet from Alpin (@AlpinDale)</a>: It seems like someone at HF forgot to private this repo on time, and Google indexed it:  sllhf/Meta-Llama-3.1-405B-Instruct-FP8  The 405B is llama 3.1? Very interesting. I wonder if they&#39;ll only r...</li><li><a href="https://doc.rust-lang.org/book/#the-rust-programming-language">The Rust Programming Language - The Rust Programming Language</a>: no description found</li><li><a href="https://x.com/teknium1/status/1815103461404606561?s=46">Tweet from Teknium (e/λ) (@Teknium1)</a>: Does anyone want to work on a master list of task types for &#34;reasoning&#34; so there can be some target tasks to focus on for improving reasoning that everyone can reference?  Let me know!</li><li><a href="https://x.com/teknium1/status/1815114033399558237?s=46">Tweet from Teknium (e/λ) (@Teknium1)</a>: Update: We started a channel in @NousResearch discord for this project, please join @ https://discord.gg/NousResearch  Will be making a github repo later today for compiling ideas</li><li><a href="https://www.meetup.com/data-scientist-meetup-in-seoul/events/302347555/">Pretraining LLMs with Upstage - [Online Study Group]📚🍰🤖🌏🤼‍♂️, Sun, Jul 28, 2024, 9:00 PM   | Meetup</a>: This is a study group session for AI/ML practitioners/learners.🤖📚  We will go through and discuss what we have learnt from the Pretraining LLMs with Upstage short course[</li><li><a href="https://arxiv.org/abs/2407.10240">xLSTMTime : Long-term Time Series Forecasting With xLSTM</a>: In recent years, transformer-based models have gained prominence in multivariate long-term time series forecasting (LTSF), demonstrating significant advancements despite facing challenges such as high...</li><li><a href="https://huggingface.co/starsnatched/MemeGPT">starsnatched/MemeGPT · Hugging Face</a>: no description found</li><li><a href="https://preview.devin.ai/">Devin (the Developer)</a>: Your reliable AI software engineer</li><li><a href="https://huggingface.co/spaces/Xenova/whisper-speaker-diarization">Whisper Speaker Diarization - a Hugging Face Space by Xenova</a>: no description found</li><li><a href="https://huggingface.co/ylacombe/parler-tts-mini-v1">ylacombe/parler-tts-mini-v1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/meta-releases/Meta-Llama-3.1-405B-Instruct">meta-releases/Meta-Llama-3.1-405B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://youtu.be/PmhsIHl27ZM">Top 5 Fastest Programming Languages in 45 secs</a>: Based on research from Google, Wikipedia, and other platforms, I’ve selected the top five fastest programming languages compared to assembly languages. I hop...</li><li><a href="https://huggingface.co/v2ray/Llama-3.1-405B">v2ray/Llama-3.1-405B · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/bugs-bunny-no-no-bunny-bugs-gif-7909500831201365932">Bugs Bunny No GIF - Bugs bunny no No Bunny - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/sad-upset-violin-sponge-bob-mr-crab-gif-3466351">Sad Violin GIF - Sad Upset Violin - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/blog/nroggendorff/train-with-llama-architecture">Train a Llama model from scratch</a>: no description found</li><li><a href="https://huggingface.co/datasets/nroggendorff/colors/commits/main">Commits · nroggendorff/colors</a>: no description found</li><li><a href="https://tenor.com/view/batman-mad-angry-tell-me-interogating-gif-17869813">Batman Mad GIF - Batman Mad Angry - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/biggest-boy-family-guy-chris-griffin-dancing-gif-17316116">Biggest Boy Family Guy GIF - Biggest Boy Family Guy Chris Griffin - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/stfu-kanye-kanye-west-shut-up-dance-gif-23839788">Stfu Kanye GIF - Stfu Kanye Kanye West - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/what-gif-21384529">What GIF - What - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/good-morning-gif-11437316614611695342">Good Morning GIF - Good morning - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/subida-gif-18379274">Subida GIF - Subida - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/spaces?search=Whi">Spaces - Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/main/en/chat_templating">Templates for Chat Models</a>: no description found</li><li><a href="https://tenor.com/view/patrick-stupid-drooling-patrick-star-spongebob-gif-12221001666588210206">Patrick Stupid GIF - Patrick Stupid Drooling - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/scared-dog-shivering-dog-dog-shaking-meme-gif-26566244">Scared Dog Shivering Dog GIF - Scared Dog Shivering Dog Dog Shaking Meme - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/muslehal/xLSTMTime">GitHub - muslehal/xLSTMTime: xLSTMTime for time series forecasting</a>: xLSTMTime for time series forecasting . Contribute to muslehal/xLSTMTime development by creating an account on GitHub.</li><li><a href="https://huggingface.co/spaces/nroggendorff/zelda-lora">Zelda Diffusion XL - a Hugging Face Space by nroggendorff</a>: no description found</li><li><a href="https://huggingface.co/spaces/nroggendorff/animexl">Anime Diffusion XL - a Hugging Face Space by nroggendorff</a>: no description found</li><li><a href="https://github.com/lucidrains/e2-tts-pytorch/blob/04b9c1cdacff4b459187e101344e718ed5b7142c/e2_tts_pytorch/e2_tts.py#L425">e2-tts-pytorch/e2_tts_pytorch/e2_tts.py at 04b9c1cdacff4b459187e101344e718ed5b7142c · lucidrains/e2-tts-pytorch</a>: Implementation of E2-TTS, &quot;Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS&quot;, in Pytorch - lucidrains/e2-tts-pytorch</li><li><a href="https://tenor.com/view/mark-zuckerberg-gif-14169217">Mark Zuckerberg GIF - Mark Zuckerberg - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.techpowerup.com/gpu-specs/voodoo3-3000-agp.c3555#:~:text=it%20might%20not%20be%20able%20to%20run%20all%20the%20latest%20games)">3dfx Voodoo3 3000 AGP Specs</a>: 3dfx Avenger, 166 MHz, 1 Pixel Shaders, 0 Vertex Shaders, 2 TMUs, 1 ROPs, 16 MB SDR, 166 MHz, 128 bit</li><li><a href="https://huggingface.co/docs/api-inference/quicktour">Overview</a>: no description found</li><li><a href="https://tenor.com/view/wizard-dance-ena-gif-27696814">Wizard Dance GIF - Wizard Dance Ena - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/lindsey-stirling-lindsey-stirling-cute-adorable-gif-19359953">Lindsey Stirling Cute GIF - Lindsey Stirling Lindsey Stirling - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://discuss.huggingface.co/t/metrics-for-training-set-in-trainer/2461/5">Metrics for Training Set in Trainer</a>: I did this by adding a custom callback which calls the evaluate() method with train_dataset at the end of every callback.  class CustomCallback(TrainerCallback):          def __init__(self, trainer) -...</li><li><a href="https://github.com/OpenDevin/OpenDevin">GitHub - OpenDevin/OpenDevin: 🐚 OpenDevin: Code Less, Make More</a>: 🐚 OpenDevin: Code Less, Make More. Contribute to OpenDevin/OpenDevin development by creating an account on GitHub.</li><li><a href="https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/">TLDR: 1 year of building with LLMs &#8211; D-Squared</a>: no description found</li><li><a href="https://github.com/ToonCrafter/ToonCrafter">GitHub - ToonCrafter/ToonCrafter: a research paper for generative cartoon interpolation</a>: a research paper for generative cartoon interpolation - ToonCrafter/ToonCrafter</li><li><a href="https://huggingface.co/spac">Spac (Stéphan Pacchiano)</a>: no description found</li><li><a href="https://www.reddit.com/r/MachineLearning/s/muGnbfl6yf">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.techpowerup.com/gpu-specs/geforce-rtx-5090.c4216">NVIDIA GeForce RTX 5090 Specs</a>: NVIDIA GB202, 2520 MHz, 20480 Cores, 640 TMUs, 192 ROPs, 28672 MB GDDR7, 2500 MHz, 448 bit</li><li><a href="https://www.techpowerup.com/gpu-specs/geforce-rtx-5060.c4219">NVIDIA GeForce RTX 5060 Specs</a>: NVIDIA GB206, 2520 MHz, 4608 Cores, 144 TMUs, 48 ROPs, 8192 MB GDDR7, 2500 MHz, 128 bit
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1263987039850729492)** (4 messages): 

> - `Knowledge Graphs`
> - `News Reading Experience`
> - `Hugging Face Model Kwargs`
> - `Speaker Diarization & Whisper Transcription` 


- **Exploring Knowledge Graphs**: A user offered assistance on **Knowledge Graphs**, mentioning their past experience in building and using them.
   - *They are very fun and useful!* was the sentiment shared about working with Knowledge Graphs.
- **Improving News Reading Experience**: Another user expressed interest in **improving news reading experiences and sentiment analysis**, praising Hugging Face's tools: *I am no coder, really loved what Huggingface built so far!*
- **Finding Model Kwargs on Hugging Face**: A user inquired about where to look for **model_kwargs** when using models on Hugging Face.
   - They shared an example snippet where they used `{'temperature': 1.0, 'top_k': 20, 'max_length': 128}` as model_kwargs.
- **Automating Speaker Diarization and Transcription**: A member asked for advice on automating a pipeline for **speaker diarization, Whisper transcriptions, and timestamps**.
   - They are willing to learn about database management for the output and looking for recommended models or open-source repositories.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1264459140256763965)** (8 messages🔥): 

> - `Apple Intelligence`
> - `LoRA fine-tuning`
> - `AI paper on arXiv`
> - `AI's world transformation`
> - `Free online courses` 


- **Apple's single model capabilities impress**: An enthusiastic member shared insights on how Apple Intelligence uses a single model for multiple tasks, emphasizing its genius and versatility.
   - They recommended readers to check out a [YouTube video tutorial on LoRA fine-tuning](https://www.youtube.com/watch?v=8N9L-XK1eEU) and a detailed paper on [LoRA on arXiv](https://arxiv.org/abs/2106.09685).
- **Interesting AI paper on arXiv**: A member found an insightful [AI paper on arXiv](https://arxiv.org/pdf/2407.08683v1), sharing it with the community for further reading.
   - *No additional comments provided.*
- **How AI is transforming the world**: A member shared a comprehensive article from [Brookings](https://www.brookings.edu/articles/how-artificial-intelligence-is-transforming-the-world/) discussing AI's impact on various sectors and offering policy recommendations.
   - The article emphasizes AI’s role in improving decision making and transforming everyday life.
- **Free online courses with certificates**: Several free online courses were recommended, including [Introduction to Python Programming](https://www.udacity.com/course/introduction-to-python--ud1110), [Product Design](https://www.udacity.com/course/product-design--ud509), and [Data Analysis](https://www.udacity.com/course/intro-to-data-analysis--ud170) on Udacity.
   - The courses cover essential skills such as Python, product validation, and data analysis using popular Python libraries.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.brookings.edu/articles/how-artificial-intelligence-is-transforming-the-world/">How artificial intelligence is transforming the world | Brookings</a>: Darrell West and John Allen examine the societal and political aspects of developing artificial intelligence technologies.</li><li><a href="https://www.udacity.com/course/introduction-to-python--ud1110">Free Intro to Python Course | Udacity</a>: Learn online and advance your career with courses in programming, data science, artificial intelligence, digital marketing, and more. Gain in-demand technical skills. Join today!</li><li><a href="https://www.udacity.com/course/product-design--ud509">Product Design | Udacity</a>: Learn online and advance your career with courses in programming, data science, artificial intelligence, digital marketing, and more. Gain in-demand technical skills. Join today!</li><li><a href="https://www.udacity.com/course/intro-to-data-analysis--ud170">Introduction to Data Analytics | Data Analysis | Udacity</a>: Learn online and advance your career with courses in programming, data science, artificial intelligence, digital marketing, and more. Gain in-demand technical skills. Join today!</li><li><a href="https://www.ft.com/content/0b210299-4659-4055-8d81-5a493e85432f?utm_source=superhuman&utm_medium=newsletter&utm_campaign=the-godmother-of-ai-is-building-a-new-startup">‘Godmother of AI’ Fei-Fei Li builds $1bn start-up in 4 months</a>: no description found</li><li><a href="https://medium.com/@teendifferent/the-secret-behind-apple-intelligence-one-model-endless-possibilities-833ad1b989af)">no title found</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=8N9L-XK1eEU)">How to fine-tune a model using LoRA (step by step)</a>: LoRA is a genius idea.By the end of this video, you&#39;ll know everything important about how LoRA works. I&#39;ll show you an example where we&#39;ll fine-tune a model...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1263947778686718002)** (49 messages🔥): 

> - `Hermes 2.5`
> - `Model Merging`
> - `Open Empathic`
> - `Gary4live`
> - `SmolLM` 


- **Gary4live: Ableton Plugin for MusicGen**: A new Ableton plugin called **Gary4live** is demonstrated in a [speedrun video](https://x.com/thepatch_kev/status/1814386138972598446?s=46) showcasing its capabilities in music generation.
   - The plugin has been proven to work collaboratively, creating songs with others in real-time musical jam sessions.
- **SmolLM Arena Launches**: A [new project](https://huggingface.co/spaces/as-cle-bert/smolLM-arena) called SmolLM Arena has been launched, allowing users to compare various Small Language Models (<1.7B params).
   - The arena features a chatbot interface, runs faster, and includes usage instructions for a smoother user experience.
- **Manifold Research Call**: Manifold Research Group is hosting a [Community Research Call](https://www.manifoldrg.com/community-research-call/) covering topics like Generalist Multimodality, LLM Agents, and Robotics.
   - The call aims to provide project updates and facilitate Q&A sessions to involve more community members in their efforts.
- **On-Device LLMs Workshop**: A [YouTube workshop](https://www.youtube.com/watch?v=zuNMvL4dtvM) by Enrico Rampazzo discusses the future of on-device Large Language Models (LLMs) and their capabilities.
   - The session explores how on-device LLMs can be revolutionary for mobile applications and AI deployment.
- **Rust Client Library for Gradio**: A new Rust client library for [Gradio](https://github.com/JacobLinCool/gradio-rs) has been released, designed to facilitate easier integration with various Gradio spaces.
   - The library supports models like `hf-audio/whisper-large-v3` and `stabilityai/stable-diffusion-3-medium`, and the developer is seeking feedback and contributions from the community.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/thepatch_kev/status/1814386138972598446?s=46">Tweet from thecollabagepatch (@thepatch_kev)</a>: another speedrun with  the open source ableton plugin that jams with you  gary4live  this one&#39;s a collab with the homie tom&#39;s beat  skip to 2:56 to hear it  @_buildspace @_nightsweekends   @ma...</li><li><a href="https://huggingface.co/spaces/ptx0/PixArt-900M-EDiffi">PixArt 900M 1024px E-Diffi (Mixture-of-Experts) - a Hugging Face Space by ptx0</a>: no description found</li><li><a href="https://huggingface.co/spaces/KingNish/OpenCHAT-mini">OpenCHAT Mini - a Hugging Face Space by KingNish</a>: no description found</li><li><a href="https://www.manifoldrg.com/community-research-call/">Community Research Calls</a>: Community Research Calls are our dedicated sessions to providing the larger Manifold community with organization-level and project-level updates. If you are interested in attending these update sessio...</li><li><a href="https://huggingface.co/spaces/as-cle-bert/smolLM-arena">SmolLM Arena - a Hugging Face Space by as-cle-bert</a>: no description found</li><li><a href="https://huggingface.co/spaces/gokaygokay/UltraPixel">UltraPixel - a Hugging Face Space by gokaygokay</a>: no description found</li><li><a href="https://x.com/AnindyadeepS/status/1815284584332099840">Tweet from Anindya (@AnindyadeepS)</a>: Happy Monday. I know I am late to this game, but today, I published the very first blog of my written series on MakeMore.   https://link.medium.com/BKXOLshVqLb  For a while, I studied Andrej Karpathy&...</li><li><a href="https://huggingface.co/spaces/KingNish/Image-Gen-Pro">Image Gen Pro - a Hugging Face Space by KingNish</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=zuNMvL4dtvM.">On-Device LLMs: The Future of AI in Your Pocket | Beyond ChatGPT</a>: 🚀 Dive into the Future of AI: On-Device Large Language Models (LLMs) 🧠📱In this cutting-edge session, we explore the revolutionary world of on-device Large...</li><li><a href="https://huggingface.co/blog/nroggendorff/create-diffusers-dataset">Create a Diffusers-compatible Dataset for Stable Diffusion Fine-tuning</a>: no description found</li><li><a href="https://youtu.be/Gscelu22FWI">Design TikTok&#39;s Recommendation System | ML System Design | #systemdesign</a>: Do you know why TikTok&#39;s recommendation algorithm is so good? In this video, we design TikTok&#39;s recommendation system. The video covers machine learning aspe...</li><li><a href="https://www.udacity.com/course/introduction-to-python--ud1110">Free Intro to Python Course | Udacity</a>: Learn online and advance your career with courses in programming, data science, artificial intelligence, digital marketing, and more. Gain in-demand technical skills. Join today!</li><li><a href="https://www.udacity.com/course/product-design--ud509">Product Design | Udacity</a>: Learn online and advance your career with courses in programming, data science, artificial intelligence, digital marketing, and more. Gain in-demand technical skills. Join today!</li><li><a href="https://www.udacity.com/course/intro-to-data-analysis--ud170">Introduction to Data Analytics | Data Analysis | Udacity</a>: Learn online and advance your career with courses in programming, data science, artificial intelligence, digital marketing, and more. Gain in-demand technical skills. Join today!</li><li><a href="https://github.com/cappuch/mfetch">GitHub - cappuch/mfetch: pure python fetching utility (like neofetch)</a>: pure python fetching utility (like neofetch). Contribute to cappuch/mfetch development by creating an account on GitHub.</li><li><a href="https://medium.com/@teendifferent/the-secret-behind-apple-intelligence-one-model-endless-possibilities-833ad1b989af)">no title found</a>: no description found</li><li><a href="https://github.com/aaryadevchandra/seq2seq-machine-translation">GitHub - aaryadevchandra/seq2seq-machine-translation: English to German language translator using a vanilla sequence-to-sequence model</a>: English to German language translator using a vanilla sequence-to-sequence model - aaryadevchandra/seq2seq-machine-translation</li><li><a href="https://github.com/ParagEkbote/Hugging_Face_Docs">GitHub - ParagEkbote/Hugging_Face_Docs: Documentation support for Hugging Face libraries.</a>: Documentation support for Hugging Face libraries. Contribute to ParagEkbote/Hugging_Face_Docs development by creating an account on GitHub.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e9bhs8/rd2md_the_missing_link_between_reddit_and_your/?utm_source=share&utm_medium=mweb3x&utm_name=mweb3xcss&utm_term=1&utm_content=share_button">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/JacobLinCool/gradio-rs">GitHub - JacobLinCool/gradio-rs: Gradio Client in Rust.</a>: Gradio Client in Rust. Contribute to JacobLinCool/gradio-rs development by creating an account on GitHub.</li><li><a href="https://isari.ai">Isari - AI-Enhanced Workforce</a>: no description found</li><li><a href="https://github.com/Cycls/examples/blob/main/openai.py">examples/openai.py at main · Cycls/examples</a>: Cycls SDK example apps. Contribute to Cycls/examples development by creating an account on GitHub.</li><li><a href="https://huggingface.co/spaces/gokaygokay/Inspyrenet-Rembg">Inspyrenet Remove Background - a Hugging Face Space by gokaygokay</a>: no description found</li><li><a href="https://huggingface.co/spaces/gokaygokay/360PanoImage">360PanoImage - a Hugging Face Space by gokaygokay</a>: no description found</li><li><a href="https://huggingface.co/spaces/gokaygokay/TileUpscalerV2">Tile Upscaler V2 - a Hugging Face Space by gokaygokay</a>: no description found</li><li><a href="https://huggingface.co/spaces/gokaygokay/AuraFlow-with-Captioner">AuraFlow with Captioner - a Hugging Face Space by gokaygokay</a>: no description found</li><li><a href="https://huggingface.co/spaces/gokaygokay/PonyRealism">Pony Realism ++ - a Hugging Face Space by gokaygokay</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1263940291623387237)** (2 messages): 

> - `Event Creation`
> - `Diagram Feedback` 


- **Event Creation Notification**: <@607608455976583206> notified about creating an [event](https://discord.com/events/879548962464493619/1263939506281779342) and sought feedback or edits.
- **Positive Feedback on Diagram**: Member *lunarflu* praised the event, specifically noting the diagram: *'awesome, love the diagram as well!'*.


  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1264778383703674912)** (1 messages): 

> - `SD3 training bugs`
> - `Diffusers repository` 


- **Fixed SD3 Training Bugs**: A community effort led to the discovery and fixing of bugs in the **SD3 training scripts** in the [Diffusers repository](https://github.com/huggingface/diffusers/pull/8917/).
   - The fix addresses issues #8887 and #8708 and adds an option to control the pre-conditioning behavior on the model outputs.
- **Diffusers Repository Update**: The [Diffusers repository](https://github.com/huggingface/diffusers/pull/8917/) has been updated to fix training script bugs.
   - Community contributions greatly aided in identifying and resolving the issues in the SD3 training process.



**Link mentioned**: <a href="https://github.com/huggingface/diffusers/pull/8917/.">[Training] SD3 training fixes by sayakpaul · Pull Request #8917 · huggingface/diffusers</a>: What does this PR do? Fixes #8887 and #8708. Additionally, it adds an option to control the pre-conditioning behavior on the model outputs. Multiple folks have reported that for rectified-flows we ...

  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1264494540166004786)** (19 messages🔥): 

> - `Hybrid Model with Inception and ViT`
> - `Scrabble Board Tile Detection`
> - `Binary Segmentation Projects` 


- **Creating Hybrid Model with Inception and ViT**: A member inquired about integrating an **Inception network** with **ViT** for image classification, and another suggested using the [hybrid ViT implementation in timm](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer_hybrid.py), detailing the process of flattening the feature map to feed into ViT.
- **Challenges in Scrabble Board Tile Detection**: A member described their attempt to detect Scrabble tiles using CV and ML but faced accuracy issues, while others suggested methods including using a [scrabble-opencv GitHub project](https://github.com/jheidel/scrabble-opencv) for ideas.
   - Recommendations included marking the board using tools like CVAT and applying a hardcoded method considering fixed board dimensions and camera angles, although ML might be unavoidable for better accuracy.
- **Binary Segmentation Project Discussion**: A brief interaction occurred where a member asked if anyone had experience with binary segmentation projects and received a response from another member who has worked on it using **UNet**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://app.cvat.ai/.">Computer Vision Annotation Tool</a>: no description found</li><li><a href="https://github.com/jheidel/scrabble-opencv/tree/master">GitHub - jheidel/scrabble-opencv: A fun scrabble score keeper using OpenCV</a>: A fun scrabble score keeper using OpenCV. Contribute to jheidel/scrabble-opencv development by creating an account on GitHub.</li><li><a href="https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer_hybrid.py">pytorch-image-models/timm/models/vision_transformer_hybrid.py at main · huggingface/pytorch-image-models</a>: The largest collection of PyTorch image encoders / backbones. Including train, eval, inference, export scripts, and pretrained weights -- ResNet, ResNeXT, EfficientNet, NFNet, Vision Transformer (V...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1264088496507846756)** (10 messages🔥): 

> - `SQL RAG`
> - `Date Extraction`
> - `Open-source Text-to-HTML/CSS Model`
> - `Fine-tuning Language Models`
> - `Metrics in Transformer Fine-tuning` 


- **SQL RAG sees positive use case**: A member noted that **Perplexity AI** successfully implements SQL Retrieval-Augmented Generation (RAG) for correct results.
- **Challenges in Date Extraction in NLP**: A user has difficulty extracting correct **start and end dates** from statements like 'in the last 6 months' using dateparser and an LLM called **Qwen2**.
- **Seeking Open-source Text-to-HTML/CSS Models**: A user is looking for an **open-source text-to-HTML/CSS generation model** and requests recommendations from the community.
- **Identical Metrics during Transformer Fine-tuning**: A user experienced **identical values** for recall, F1, accuracy, and precision metrics while fine-tuning a **LILT model** and queries if others have encountered the same.
- **High-Accuracy Character Impersonation with LLMs**: A beginner user wants to fine-tune **Llama3** model for impersonating a philosopher using unlabelled text data.
   - Another user suggests trying **retrieval-augmented generation** but the beginner remains skeptical about its accuracy and coherence for character impersonation.


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1264286329118789682)** (4 messages): 

> - `Open-source text-to-html/css generation`
> - `Artistic styles in SDv1.4`
> - `Evaluating diffusion models`
> - `Stable diffusion paper encoder methods` 


- **Hunt for open-source text-to-html/css model**: A member is seeking an **open-source text-to-html/css generation model** and is asking the community for guidance.
- **Curiosity about SDv1.4 artistic styles**: A member inquired about the list of **artistic styles** used in the training of **SDv1.4**.
- **Tips on evaluating diffusion models**: A newcomer to **diffusion models** and **image generation** is seeking advice on how to **evaluate their models**.
- **Stable diffusion paper encoder techniques**: A member is questioning the **construction of the latent space** in the original **stable diffusion paper**, specifically whether the model was trained on quantized weights or another method.


  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1263962787798061117)** (213 messages🔥🔥): 

> - `Mojo Socket Implementation`
> - `Dual Stack Sockets Issue`
> - `Mojo Integration with Existing Libraries`
> - `Interoperability between Mojo and Python`
> - `Production Readiness of Mojo` 


- **Mojo Socket Implementation Discussion**: Users explored various socket implementations for Mojo, with a specific focus on the challenges posed by platform-specific differences, such as those in Windows sockets.
   - *darkmatter__* noted that Rust’s socket implementation might provide cleaner references, and the group also highlighted the importance of accommodating SCTP for future protocols.
- **Debate on Dual Stack Sockets in Mojo**: There was consensus on preferring the dual stack approach for new server sockets, where a single socket accepts both IPv4 and IPv6 connections, mirroring Python’s implementation.
   - *darkmatter__* suggested using `io_uring` for Linux to unify the handling of incoming connections under high performance workloads.
- **Integration of Mojo with External Libraries**: The community discussed the use of various external libraries with Mojo, including *darkmatter__* working on a Zig-like `translate-c` solution to bridge the gap until proper interop is available.
   - There were also mentions of calling into DPDK for network operations and the potential complexities of dependencies in large codebases.
- **Handling Python Classes in Mojo**: Users attempted to integrate Python classes into Mojo for custom data types, facing challenges with error handling and aliasing for Python’s dynamic types.
   - The discussion highlighted the limitations of Mojo in supporting Python classes fully and explored possible workarounds, including adopting built-in Mojo structures.
- **Community Input on Mojo’s Production Readiness**: There were mixed opinions on Mojo's readiness for production use, with *darkmatter__* advising caution until the language reaches a stable 1.0 release.
   - *darinsimmons* provided a longer-term perspective, suggesting that stable production use might be realistic within a couple of years, depending on the specific use cases and features needed.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tigerbeetle.com/blog/a-friendly-abstraction-over-iouring-and-kqueue">A Programmer-Friendly I/O Abstraction Over io_uring and kqueue</a>: The financial transactions database to power the next 30 years of Online Transaction Processing.</li><li><a href="https://docs.kernel.org/networking/tls.html">Kernel TLS &#8212; The Linux Kernel  documentation</a>: no description found</li><li><a href="https://www.youtube.com/watch?v">YouTube</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=5znybwzUZog)">Asynchronous and Direct IO for PostgreSQL on FreeBSD Thomas Munro</a>: Full description at https://www.bsdcan.org/events/bsdcan_2022/schedule/session/90-asynchronous-and-direct-io-for-postgresql-on-freebsd/</li><li><a href="https://github.com/Legrandin/pycryptodome">GitHub - Legrandin/pycryptodome: A self-contained cryptographic library for Python</a>: A self-contained cryptographic library for Python. Contribute to Legrandin/pycryptodome development by creating an account on GitHub.</li><li><a href="https://man7.org/linux/man-pages/man7/sctp.7.html">sctp(7) - Linux manual page</a>: no description found</li><li><a href="https://github.com/modularml/mojo/issues/3262">Ability to Link to C Libraries · Issue #3262 · modularml/mojo</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? Ideally there would be something like a @link(…) decor...</li><li><a href="https://github.com/dmitry-salin/io_uring">GitHub - dmitry-salin/io_uring: The io_uring library for Mojo</a>: The io_uring library for Mojo. Contribute to dmitry-salin/io_uring development by creating an account on GitHub.</li><li><a href="https://github.com/mzaks/compact-dict">GitHub - mzaks/compact-dict: A fast and compact Dict implementation in Mojo 🔥</a>: A fast and compact Dict implementation in Mojo 🔥. Contribute to mzaks/compact-dict development by creating an account on GitHub.</li><li><a href="https://modul.ar/community-meeting-doc.">[Public] Mojo Community Meeting</a>: Mojo Community Meeting This doc link: https://modul.ar/community-meeting-doc  This is a public document; everybody is welcome to view and comment / suggest.  All meeting participants must adhere to th...</li><li><a href="https://github.com/bytecodealliance/rustix/tree/main/src/net">rustix/src/net at main · bytecodealliance/rustix</a>: Safe Rust bindings to POSIX-ish APIs. Contribute to bytecodealliance/rustix development by creating an account on GitHub.</li><li><a href="https://github.com/rust-lang/rfcs/blob/master/text/3128-io-safety.md">rfcs/text/3128-io-safety.md at master · rust-lang/rfcs</a>: RFCs for changes to Rust. Contribute to rust-lang/rfcs development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot: From *Modular*:
<https://twitter.com/Modular/status/1815463417391837596>
  

---


### **Modular (Mojo 🔥) ▷ #[📺︱youtube](https://discord.com/channels/1087530497313357884/1098713700719919234/1264995931292700685)** (1 messages): 

> - `Mojo 🔥 Community Meeting #4`
> - `Flat Buffers`
> - `Forge Tools`
> - `Mojo 🔥 Standard Library`
> - `Mojo 🔥 Gen` 


- **Mojo 🔥 Community Meeting #4 highlights**: Modular just posted a new [YouTube video](https://www.youtube.com/watch?v=_QVs626Vn2k) titled '**Mojo 🔥 Community Meeting #4**,' featuring discussions on **Flat Buffers**, **Forge Tools**, and extending the **Mojo 🔥 standard library**.
   - The meeting covered topics such as memory-efficient serialization, new tools for the Mojo 🔥 ecosystem, and updates on Mojo 🔥 Gen.
- **Flat Buffers: efficient serialization**: A key highlight from the Mojo 🔥 Community Meeting #4 was the discussion on */Flat Buffers/* for memory-efficient serialization.
   - Attendees explored their use in optimizing data handling within the Mojo 🔥 framework.



**Link mentioned**: <a href="https://www.youtube.com/watch?v=_QVs626Vn2k">Mojo 🔥 Community Meeting #4</a>: Recording of the Mojo Community Meeting #4🫓 Flat Buffers: memory efficient serialization⚒️ Forge Tools: extending the Mojo 🔥 standard library🔄 Mojo 🔥 Gen...

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1263990875340673158)** (185 messages🔥🔥): 

> - `Anti-Pattern Discussions`
> - `OpenSSL and Mojo Projects`
> - `Newton's Method for Float Literals`
> - `Future of Mojo's Async Scheduler`
> - `CPU Performance Comparisons` 


- **Anti-pattern sparks lively debate**: Members humorously discussed an 'anti-pattern' workaround for conditional conformance not working, leading to a mix of reactions from agreement to playful banter.
- **OpenSSL integration poses challenges**: Discussion revealed the massive size of OpenSSL and the hurdles in integrating it with Mojo projects, emphasizing the complexity of cryptography implementations.
- **Newton's Method helper proposal**: A member shared an implementation of Newton's Method for Float Literals in Mojo, sparking detailed discussions on closures and capturing keywords for numerical equation solving.
- **Future of Mojo's async scheduler debated**: Members debated on whether Mojo should include a default async scheduler in its stdlib, with arguments for better ecosystem interoperability and concerns over limiting alternative developments.
- **AMD vs Intel for CPU performance**: The community compared AMD and Intel CPUs for various tasks, with a focus on recent stability issues with Intel and the relative performance benefits of AMD's L3 cache for specific use cases.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://learn.microsoft.com/en-us/windows/win32/api/ioringapi/">ioringapi - Win32 apps</a>: Provides APIs for creating and managing I/O rings.</li><li><a href="https://www.youtube.com/watch?v=QzHcrbT5D_Y">Intel has a Pretty Big Problem</a>: Intel&#39;s 13900k&#39;s and 14900k&#39;s are crashing at an alarming rate? Why isn&#39;t anyone talking about it and what is Intel&#39;s solution?Forum Thread here: https://for...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1264897624964599839)** (1 messages): 

> - `Matrix Multiplication in Mojo`
> - `Comparing Mojo to Numpy Performance` 


- **Matrix Multiplication Examples in Mojo**: A user reported trying out [matrix multiplication examples](https://docs.modular.com/mojo/notebooks/Matmul) from the documentation and found them amazing, transitioning from pure Python to optimized implementations.
   - The notebook starts with a Python-like implementation and optimizes by adding types, vectorizing, tiling, and parallelizing.
- **Mojo Implementation 4x Slower than Numpy**: A user shared a real-life assessment comparing Mojo's final version to `numpy`, finding it still **4x slower** on their machine.
   - Looking for insights, they questioned if their expectations were unrealistic.



**Link mentioned**: <a href="https://docs.modular.com/mojo/notebooks/Matmul">Matrix multiplication in Mojo | Modular Docs</a>: Learn how to leverage Mojo&#x27;s various functions to write a high-performance matmul.

  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1264154200078549094)** (5 messages): 

> - `nightly/max feed reliability`
> - `getting started with MAX`
> - `open source contributions to MAX`
> - `guidance for new contributors` 


- **Issues with nightly/max feed reliability**: A member reported issues with the `nightly/max` feed not keeping `nightly/mojo` in sync, leading to them switching back to `nightly/mojo`.
   - Another member acknowledged the issues and mentioned that work is ongoing to resolve them soon.
- **Getting started with MAX**: New member eager to learn was directed to the [MAX documentation](https://docs.modular.com/max/get-started) and [quickstart guide](https://docs.modular.com/max/install) for getting started.
   - Resources include a [blog post on contributing to Mojo standard library](https://www.modular.com/blog/how-to-contribute-to-mojo-standard-library-a-step-by-step-guide) and a [contribution guide](https://github.com/modularml/mojo/blob/nightly/CONTRIBUTING.md).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/max/get-started">Get started with MAX | Modular Docs</a>: Welcome to the MAX quickstart guide!</li><li><a href="https://www.modular.com/blog/how-to-contribute-to-mojo-standard-library-a-step-by-step-guide">Modular: How to Contribute to Mojo Standard Library: A Step-by-Step Guide</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: How to Contribute to Mojo Standard Library: A Step-by-Step Guide
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[max-gpu](https://discord.com/channels/1087530497313357884/1212827673257316453/1264293038574801047)** (3 messages): 

> - `XLA Involvement`
> - `Mojo on GPU Availability` 


- **Former XLA team joins Mojo**: A member shared that many of their team were directly involved in XLA at Google, and they are bringing their learnings and a different philosophy to AI infrastructure development at Mojo.
- **Mojo GPU support coming this summer**: When asked about GPU availability for Mojo, a member confirmed that it should be available this summer.


  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1264102837827141703)** (83 messages🔥🔥): 

> - `Nightly Mojo Compiler Updates`
> - `LegacyPointer and DTypePointer Discussion`
> - `Issues with memcpy Function`
> - `Changes in Mojo API`
> - `Community Interaction and Documentation` 


- **Nightly Mojo Compiler 2024.7 Releases**: A new nightly Mojo compiler has been released, now updated to `2024.7.2205` with the command: `modular update nightly/mojo`. Check the [current changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) and [raw diff](https://github.com/modularml/mojo/compare/b007d77a5832026553a9fccd2ea99ef21031336d...0d805c178526fc755edc6cd226af22b8da7341b6) for detailed changes.
- **LegacyPointer and DTypePointer Deprecated**: A tight discussion revealed that `LegacyPointer` and `DTypePointer` are deprecated, with current and future support focusing on `UnsafePointer`.
   - *Users express concerns* that frequent changes make code maintenance difficult, citing issues with the constant deprecation of old functions.
- **Migration Issues with memcpy Override**: Changes to the `memcpy` function with three new overrides have sparked confusion among users. **Carl Caulkett** shared a *workaround* involving the transition from `DTypePointer` to `UnsafePointer`.
- **Mojo API Changes for Improved Pointer Safety**: The community discussed API changes aimed at enhancing pointer safety, specifically the shift from `DTypePointer` to `UnsafePointer`. Daniel confirmed that using `UnsafePointer[Float64]` will replace `DTypePointer[DType.float64]`.
- **Demand for Better Documentation**: Users highlighted the necessity for better documentation to manage the fast-evolving APIs. The community acknowledges the technical debt involved with extensive documentation that needs frequent updates.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://sourcegraph.com/search">Sourcegraph</a>: no description found</li><li><a href="https://youtu.be/_QVs626Vn2k?t=3617">Mojo 🔥 Community Meeting #4</a>: Recording of the Mojo Community Meeting #4🫓 Flat Buffers: memory efficient serialization⚒️ Forge Tools: extending the Mojo 🔥 standard library🔄 Mojo 🔥 Gen...</li><li><a href="https://github.com/modularml/mojo/blob/nightly/stdlib/src/memory/__init__.mojo">mojo/stdlib/src/memory/__init__.mojo at nightly · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/3126)">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/blob/16cc60dc3fbed1eff01a8f5fee94f97cf97cca33/stdlib/src/memory/__init__.mojo">mojo/stdlib/src/memory/__init__.mojo at 16cc60dc3fbed1eff01a8f5fee94f97cf97cca33 · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo-marathons](https://discord.com/channels/1087530497313357884/1255303604894437388/1263990373001330699)** (43 messages🔥): 

> - `NumPy Performance Testing`
> - `Understanding CPU FLops`
> - `Matrix Multiplication Benchmarks`
> - `Architecture-Specific Optimizations`
> - `Mojo's Generics Limitations` 


- **NumPy Performance Numbers Raise Eyebrows**: A member shared results from benchmarking `numpy` indicating a significant performance difference, with values like **271.864 GFLOPS** for NumPy and **249.371 GFLOPS** for a pure Python implementation.
   - They expressed surprise over almost no performance difference with unrolling, questioning if older architecture-specific optimizations in BLAS might be the cause.
- **Calculating & Comparing Theoretical Peak FLops**: Members discussed how to calculate peak GFLOPS using core count, frequency, and SIMD width, providing an example calculation for an i7 7th Gen CPU.
   - *One nailed down that achieving around 80% of the theoretical peak value of **358 GFLOPS** is reasonable for NumPy performance.*
- **Pipeline Efficiency in Matrix Multiplications**: Discussions dove into pipeline efficiencies where the focus was to keep the ALU fed by avoiding memory-related stalls.
   - *A specific note was made about FMA instructions being 2-cycle operations on Kaby Lake architecture with reference to [uops.info](https://uops.info/table.html).*
- **Calling CPUID in Mojo for Target Information**: The conversation shifted towards using cpuid instructions for gathering target-specific architecture information within Mojo, with some members expressing the need for better exposure of such details.
   - Existing tools like [CpuId.jl](https://github.com/m-j-w/CpuId.jl) and Intel's C libraries were discussed but noted to be architecture-specific.
- **Challenges with Mojo for Generics in High-Performance Computing**: Members agreed that Mojo needs to expose more information about the target architecture to support truly generic high-performance algorithms.
   - There was also a mention of supercomputers having architecture-specific BLAS libraries, underlining the complexity and specificity required.



**Link mentioned**: <a href="https://github.com/m-j-w/CpuId.jl">GitHub - m-j-w/CpuId.jl: Ask the CPU for cache sizes, SIMD feature support, a running hypervisor, and more.</a>: Ask the CPU for cache sizes, SIMD feature support, a running hypervisor, and more. - m-j-w/CpuId.jl

  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1263965843717623880)** (424 messages🔥🔥🔥): 

> - `ComfyUI recommendations`
> - `Forge vs. Easy Diffusion experience`
> - `Using Latent mode for Regional Prompter`
> - `Issues with VRAM and GPU compatibility`
> - `Upscaling errors in Forge` 


- **ComfyUI recommended over Forge**: Multiple users suggested switching from Forge to **ComfyUI** for a better experience, citing issues with **Forge's** functionality and compatibility.
   - Users praised **ComfyUI** for having significantly more tools and features, although noting it has a more complex node-based interface.
- **Comparing Forge to Easy Diffusion**: A user noted that while **Forge** is faster than **Easy Diffusion**, it lacks some features and outputs errors when upscaling.
   - Others commented on upscaling issues and resolutions not being handled properly in **Forge**, suggesting alternatives.
- **Using Latent mode for Regional Prompter**: Guidance was provided on using **Latent mode** instead of **Attention mode** for **Regional Prompter** to prevent character blending.
   - Detailed prompts and clarifications were shared for improving the use of **Latent mode** with multiple character LoRAs.
- **VRAM and GPU compatibility issues**: Discussions covered VRAM requirements for stable diffusion and issues with VRAM on GPUs, particularly with **AMD** cards.
   - Solutions included using **local installations** and cloud GPUs for those with limited home GPU capability.
- **Errors with upscaling in Forge**: Users encountered 'NoneType' errors when upscaling images with **Forge**.
   - Suggestions included switching to **hi-res fix** and alternative upscaling tools like **real-ESRGAN**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://comfyanonymous.github.io/ComfyUI_examples/controlnet/">ControlNet and T2I-Adapter Examples</a>: Examples of ComfyUI workflows</li><li><a href="https://www.youtube.com/watch?v=O8-0ZidswTw">Invoke AI (Stable Diffusion) Outpainting Time Lapse #01</a>: Just played around with the current InvokeAI build for Windows.Total time was about 70 minutes but of course I have sped it up ...I was messing around a lot,...</li><li><a href="https://www.shakker.ai/activitys/shake-the-world">Shakker AI - Premium Stable Diffusion Model Hub</a>: no description found</li><li><a href="https://www.nasa.gov/missions/mars-2020-perseverance/perseverance-rover/heres-how-ai-is-changing-nasas-mars-rover-science/">Here’s How AI Is Changing NASA’s Mars Rover Science - NASA</a>: Artificial intelligence is helping scientists to identify minerals within rocks studied by the Perseverance rover.</li><li><a href="https://github.com/2kpr/ComfyUI-UltraPixel">GitHub - 2kpr/ComfyUI-UltraPixel</a>: Contribute to 2kpr/ComfyUI-UltraPixel development by creating an account on GitHub.</li><li><a href="https://huggingface.co/stab">Stab (Fua)</a>: no description found</li><li><a href="https://youtu.be/kg9qpyupXbI">Follow-Your-Emoji: Fine-Controllable and Expressive Freestyle Portrait Animation</a>: Follow-Your-Emoji: Fine-Controllable and Expressive Freestyle Portrait Animation</li><li><a href="https://liveportrait.github.io/">Efficient Portrait Animation with Stitching and Retargeting Control</a>: no description found</li><li><a href="https://github.com/comfyanonymous/ComfyUI/blob/master/extra_model_paths.yaml.example">ComfyUI/extra_model_paths.yaml.example at master · comfyanonymous/ComfyUI</a>: The most powerful and modular stable diffusion GUI, api and backend with a graph/nodes interface. - comfyanonymous/ComfyUI</li><li><a href="https://github.com/wootwootwootwoot/ComfyUI-RK-Sampler">GitHub - wootwootwootwoot/ComfyUI-RK-Sampler: Batched Runge-Kutta Samplers for ComfyUI</a>: Batched Runge-Kutta Samplers for ComfyUI. Contribute to wootwootwootwoot/ComfyUI-RK-Sampler development by creating an account on GitHub.</li><li><a href="https://www.gmktec.com/products/amd-ryzen-7-7840hs-mini-pc-nucbox-k6?variant=7530f28e-cce6-4e22-ac92-e7999375a6be)">AMD Ryzen 7 7840HS Mini PC--NucBox K6</a>: A compact desktop computer featuring an AMD Ryzen 7 7840HS processor, GPU equal to GTX1060Ti. Dual 2.5G LAN. Dual M.2 slots. Dual fans.</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui.git">GitHub - AUTOMATIC1111/stable-diffusion-webui: Stable Diffusion web UI</a>: Stable Diffusion web UI. Contribute to AUTOMATIC1111/stable-diffusion-webui development by creating an account on GitHub.</li><li><a href="https://github.com/ehristoforu/DeFooocus">GitHub - ehristoforu/DeFooocus: Always focus on prompting and generating</a>: Always focus on prompting and generating. Contribute to ehristoforu/DeFooocus development by creating an account on GitHub.</li><li><a href="https://github.com/comfyanonymous/ComfyUI">GitHub - comfyanonymous/ComfyUI: The most powerful and modular stable diffusion GUI, api and backend with a graph/nodes interface.</a>: The most powerful and modular stable diffusion GUI, api and backend with a graph/nodes interface. - comfyanonymous/ComfyUI</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers">stabilityai/stable-diffusion-3-medium-diffusers · Hugging Face</a>: no description found</li><li><a href="https://www.stablediffusiontutorials.com/2024/03/stable-diffusion-error-amd.html">Fixing Stable Diffusion Errors while running on AMD</a>: no description found</li><li><a href="https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/">TLDR: 1 year of building with LLMs &#8211; D-Squared</a>: no description found
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1264220301546361033)** (28 messages🔥): 

> - `Google Meet Troubleshooting`
> - `CUDA and SVD`
> - `Building with LLMs`
> - `ECC Memory in Workstations`
> - `Register Allocation in Flash Attention` 


- **Google Meet link for event confirmed**: After confirming, it was noted that the correct Google Meet link for the event is [this one](https://meet.google.com/pha-emvg-pem).
   - One user mentioned experiencing issues with Firefox, but switching to Chrome resolved the problem.
- **Understanding SVD performance on GPUs**: A user inquired about how **cusolver/hipSolver** performs SVD as most implementations found were just wrappers for closed-source solutions.
   - A [GitHub repository](https://github.com/Michalos88/Randomized_SVD_in_CUDA) was referenced for insights, mentioning methods like **Gram-Schmidt**, **Householder reflections**, and **Givens rotations**.
- **LLMs: Key lessons from one year of building**: A [new video and blog post](https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/) summarized lessons from a year of practitioners working with LLMs, highlighting tactical, operational, and strategic insights.
   - The author created a visual TLDR to make the series more accessible, emphasizing the value of watching the video for a better understanding.
- **Debating the necessity of ECC memory in workstations**: Users debated the necessity of **ECC memory** in workstations, with consensus suggesting it's not crucial for most desktop applications.
   - Discussion included considerations of **cosmic radiation** in specialized computing environments and the impact on **CPU options and costs**.
- **Clarification on register allocation in Flash Attention**: A user questioned how to explicitly allocate registers in Flash Attention and whether initial projections from input matrices are fused into one kernel.
   - The doubt concerned if matrix sizes are too large to support such kernel fusion effectively.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/Michalos88/Randomized_SVD_in_CUDA">GitHub - Michalos88/Randomized_SVD_in_CUDA: FAST Randomized SVD on a GPU with CUDA 🏎️</a>: FAST Randomized SVD on a GPU with CUDA 🏎️. Contribute to Michalos88/Randomized_SVD_in_CUDA development by creating an account on GitHub.</li><li><a href="https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/">TLDR: 1 year of building with LLMs &#8211; D-Squared</a>: no description found
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1264729033674068031)** (4 messages): 

> - `Profiling Triton Kernels`
> - `Memory Usage`
> - `CUDA tools for Profiling`
> - `Nsight Compute`
> - `Nsight Systems` 


- **Discussing tools for profiling Triton kernels**: A member inquired about which tools are suitable for profiling Triton kernels, specifically for tasks like peak memory usage.
   - *nsight-compute* and *nsight-systems* were recommended for detailed profiling, with a note that *nvprof* should be avoided as it has been succeeded by these tools.
- **Compatibility of standard CUDA tools with Triton kernels**: There was a conversation about whether standard CUDA tools like *nvprof* and built-in torch profilers work with Triton kernels.
   - A member mentioned that while traditional tools may not work, newer tools like *nsight-compute* and *nsight-systems* offer the needed functionality.


  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1263941656420421763)** (7 messages): 

> - `at::Tensor.mutable_data_ptr`
> - `torch.cond`
> - `torch control flow operators` 


- **Confusion over const method in at::Tensor**: A member expressed confusion about why `at::Tensor.mutable_data_ptr` is a `const` method, bringing into question the design choice behind it.
- **Emulating jax.lax.while_loop() in Torch**: A member inquired if Torch supports anything similar to `jax.lax.while_loop()` and whether this behavior could be emulated with `torch.cond()`.
   - Another user pointed out that `torch.cond()` currently only supports inference, not training, citing the [official documentation](https://pytorch.org/docs/stable/generated/torch.cond.html#torch.cond).



**Link mentioned**: <a href="https://pytorch.org/docs/stable/generated/torch.cond.html#torch.cond">torch.cond &mdash; PyTorch 2.3 documentation</a>: no description found

  

---


### **CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1264219545598562324)** (1 messages): 

> - `AMD ROCm`
> - `Composable Kernel library`
> - `AMD tech stack` 


- **AMD ROCm's Composable Kernel library presentation**: A member announced an opportunity to learn about AMD ROCm's **Composable Kernel library** presented by <@813802565426479145> in about 4 minutes.
   - The talk is highlighted as a chance to dive deeper into the **AMD tech stack**.
- **AMD Tech Stack Overview**: <@813802565426479145> is set to provide an overview of AMD's tech stack, focusing on the ROCm ecosystem.
   - This includes insights into how the **Composable Kernel** library integrates within ROCm.


  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1264249726530879559)** (3 messages): 

> - `Similarity Search Algorithm`
> - `Schotastic Rounding in Quantization` 


- **Choosing Efficient Similarity Search Algorithm for SQLite**: A member asked for recommendations on the best similarity search algorithm for a **lightweight and efficient vector database** like SQLite.
   - *Any helpful suggestions or personal experiences regarding specific algorithms would aid in decision-making.*
- **Schotastic Rounding Validity in Quantization**: A member queried about whether introducing a random element maintains the definition of **schotastic rounding** in the context of quantization.
   - *No responses provided additional clarity or alternatives.*


  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1263989830673502249)** (27 messages🔥): 

> - `CubeCL`
> - `FlashAttention2 Custom Mask`
> - `FLUTE Kernel` 


- **CubeCL enables CUDA kernels in Rust**: The [CubeCL project](https://github.com/tracel-ai/cubecl) introduces multi-platform high-performance compute language extension for Rust, using nvrtc for CUDA and a comptime system for kernel specialization.
   - Discussion points include the limitations of borrow checking and pointers, with CubeCL's system still far from matching Zig's capabilities but useful for avoiding bounds checks and loop unrolling.
- **Custom Masking in FlashAttention2**: A new [GitHub repo](https://github.com/alexzhang13/flashattention2-custom-mask) by az_zz_za introduces custom masks to the Triton implementation of FlashAttention2, addressing a common limitation.
   - Discussion involves the correct masking dimensions and potential easy modifications to implement arbitrary attention biases.
- **FLUTE accelerates LLM inference**: [FLUTE](https://github.com/HanGuo97/flute) provides a CUDA kernel for non-uniformly quantized LLM inference via a lookup table, with significant speed improvements over traditional methods.
   - The kernel is integrated with vLLM and uses CUTLASS 3, TensorCore, and Async Copy to achieve up to 2-3x faster performance, with discussions on potential implementations in Triton and Torch.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://ziglang.org>">no title found</a>: no description found</li><li><a href="https://x.com/hanguo97/status/1815104671289413664?s=46">Tweet from Han Guo (@HanGuo97)</a>: Introducing FLUTE, a CUDA kernel for non-uniformly quantized (via a lookup table) LLM Inference. It accelerates QLoRA&#39;s NormalFloat (NF) out of the box and more.  As an application, we extended NF...</li><li><a href="https://github.com/alexzhang13/flashattention2-custom-mask">GitHub - alexzhang13/flashattention2-custom-mask: Triton implementation of FlashAttention2 that adds Custom Masks.</a>: Triton implementation of FlashAttention2 that adds Custom Masks. - alexzhang13/flashattention2-custom-mask</li><li><a href="https://github.com/tracel-ai/cubecl">GitHub - tracel-ai/cubecl: Multi-platform high-performance compute language extension for Rust.</a>: Multi-platform high-performance compute language extension for Rust. - tracel-ai/cubecl
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1264131761311580221)** (17 messages🔥): 

> - `Triton Block Size`
> - `Non-GPU part of NVCC Compiler and VLAs`
> - `FP16 vs FP32 Performance`
> - `Triton Multi-Stage Pipelining` 


- **Triton Block Size Explained**: A member clarified that the **BLOCK_SIZE in Triton** is more like **TILE_SIZE in CUDA**, with **num_warps** controlling the size of a thread block ([reference](https://triton-lang.org/main/python-api/generated/triton.Config.html)).
   - *'I get confused at first too*,' another member admitted while discussing the similarity and differences.
- **NVCC Compiler and Variable-Length Arrays**: A member asked whether using **runtime-defined length arrays** (VLAs) in the non-GPU part of the NVCC compiler is feasible, even without concerns for memory or speed ([reference](https://en.wikipedia.org/wiki/Variable-length_array)).
   - Another member explained that VLAs are technically a **C feature** and may not be fully supported in **C++** standards, even though most compilers implement them.
- **FP16 Performance Considerations**: Discussion focused on why **FP16 performance** in A100 is **2x that of FP32**, despite the seemingly expected ratio from computational complexity being around 2.8x ([Ampere Architecture Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)).
   - Members investigated if the performance issue might be **I/O-bound** rather than computation bound, discussing hardware architecture and overheads.
- **Understanding Triton Multi-Stage Pipelining**: A member queried the purpose and implementation specifics of **pipelining stages in Triton**, especially the use of oddly specific fixed values in kernel stages ([getting started tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)).
   - They struggled to visualize what multi-stage pipelining accomplishes beyond basic CPU architecture pipelining, indicating a need for further clarification.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Variable-length_array">Variable-length array - Wikipedia</a>: no description found</li><li><a href="https://triton-lang.org/main/python-api/generated/triton.Config.html">triton.Config &mdash; Triton  documentation</a>: no description found</li><li><a href="https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html#sphx-glr-getting-started-tutorials-06-fused-attention-py">Fused Attention &mdash; Triton  documentation</a>: no description found
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1264673584341581844)** (3 messages): 

> - `Model Weights in Shared Memory`
> - `CUDA Register Capacity` 


- **Discussing Model Weights in Shared Memory**: A user questioned whether **model weights** could be a good example of using shared memory if register files were not an issue.
   - Another member responded affirmatively but asked why not put all model weights into **registers** instead.
- **Clarification on Register Capacity**: A discussion was prompted on whether the issue of register capacity affects the decision to place model weights in shared memory.
   - The clarification pointed out that if register capacity were not a limiting factor, it would indeed be a viable option.


  

---


### **CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/)** (1 messages): 

andreaskoepf: https://youtu.be/-732zELVbpU?si=HBXEE8t2fxCKhC5v
  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1264620506754060371)** (8 messages🔥): 

> - `Request for Cuda/C++ Channel`
> - `Discussion on existing channel usage`
> - `LLM Deployment Cost Inquiry` 


- **Request for a dedicated Cuda/C++ channel**: A member asked about the possibility of creating a **Cuda/C++** channel, citing interest from people who write **Triton** and **Cuda C++**.
   - Another user redirected them to an existing channel, but the requester was unsure if it was the right one due to limited relevant discussion.
- **Discussion on existing channel usage**: Members debated whether an existing channel covers **Cuda** and **Triton discussions** adequately.
   - One user noted the last mention of **Triton** was on June 5th, sparking doubts about the channel's relevance.
- **Inquiry into LLM deployment costs**: A member inquired about accurate numbers for **LLM deployment costs at scale**, comparing various providers like **Mistral, OpenAI, and Fireworks AI**.
   - They speculated that companies might be losing money unless they have nearly 100% hardware utilization or access to very cheap hardware.


  

---


### **CUDA MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1264560997079842816)** (6 messages): 

> - `ICML Meetup at Conference`
> - `CUDA Learners in Berkeley` 


- **ICML Meetup Coordinated**: A member mentioned being at **ICML**, prompting another to suggest meeting up at the conference.
   - The proposal was positively received with, *'sure'*, marking a mutual interest in meeting up.
- **Seeking CUDA Learners in Berkeley**: A member inquired about **CUDA learners** in the Berkeley area, expressing interest in forming a community for meetups and solving book chapter problems together.
   - Another user reminisced about [past interest in similar meetups](https://link.to/nolink), indicating prior enthusiasm for such activities.


  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1263988699260129311)** (297 messages🔥🔥): 

> - `CUDA MODE IRL Event`
> - `Train GPT-2 to GPT-3 Progress`
> - `Multi-GPU and ZeRO Implementation`
> - `MuP Branch Progress`
> - `FP8 Training Optimizations` 


- **CUDA MODE IRL Invites on September 21st**: **CUDA MODE** team members were invited to an IRL event on **September 21st** in San Francisco, coinciding with PyTorch devcon, including potential talk on **llm.c** for 20 minutes.
   - Logistics and details for the event were shared via [Google Document](https://docs.google.com/document/d/10LkM5_xLh9r_ycul2ywOfgrOGmNP9YDTa9c4V755QgY/edit).
- **Shift from train_gpt2.c to train_llama3.cu**: Among the tasks covered for CUDA MODE developments include transitioning the codebase from **train_gpt2.c** to **train_llama3.cu**.
   - This involves significant rework due to differences with Meta’s LLaMA release, including issues with torchrun and complex undocumented code.
- **Multi-GPU and ZeRO Challenges**: Discussion around **multi-GPU training** includes dealing with challenges like integrating ZeRO-2 and ZeRO-offload to manage large model weights effectively.
   - The core issues are balancing memory between GPU and CPU, maintaining determinism, and integrating master weights efficiently.
- **MuP Branch Stability Issues**: Attempts to integrate **MuP branch** with master to stabilize training runs encountered conflicts and required extensive rebasing.
   - Current focus is on resolving these conflicts and testing MuP's potential to enhance stability during long runs.
- **Advancements in FP8 Training Optimizations**: Efforts to include **FP8 training optimizations** are ongoing, with plans to implement techniques such as cuDNN FP8 forwards and backwards attention.
   - Further plans include adding visualization features for tensor values to track and debug model performance during training optimizations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lambdalabs.com/nvidia-gh200">Lambda Reserved Cloud Powered by NVIDIA GH200</a>: Lambda Reserved Cloud with NVIDIA GH200. The design of the GH200 Superchip forms a high-bandwidth connection between the NVIDIA Grace™ CPU and Hopper™ GPU.</li><li><a href="https://arxiv.org/abs/2407.05872">Scaling Exponents Across Parameterizations and Optimizers</a>: Robust and effective scaling of models from small to large width typically requires the precise adjustment of many algorithmic and architectural details, such as parameterization and optimizer choices...</li><li><a href="https://arxiv.org/abs/2309.14322">Small-scale proxies for large-scale Transformer training instabilities</a>: Teams that have trained large Transformer-based models have reported training instabilities at large scale that did not appear when training with the same hyperparameters at smaller scales. Although t...</li><li><a href="https://x.com/Yuchenj_UW/status/1814703583453192272">Tweet from Yuchen Jin (@Yuchenj_UW)</a>: Comparison between different sizes of GPT-2 (from 124M to 2.7B) trained with @karpathy&#39;s llm.c + FineWeb-Edu🤖  GPT-2 initially came in 4 sizes, with the largest at 1.5B. Using the llm.c codebase,...</li><li><a href="https://github.com/karpathy/llm.c/pull/700/files">Fix integer overflow by using `size_t` for parameter sizes. by YuchenJin · Pull Request #700 · karpathy/llm.c</a>: Due to some parameters in the GPT-2 7.3B model being quite large, the current llm.c code has integer overflow issues. This is because it uses a 32-bit int to store the number of bytes for weights a...</li><li><a href="https://arxiv.org/abs/2406.16793">Adam-mini: Use Fewer Learning Rates To Gain More</a>: We propose Adam-mini, an optimizer that achieves on-par or better performance than AdamW with 45% to 50% less memory footprint. Adam-mini reduces memory by cutting down the learning rate resources in ...</li><li><a href="https://developer.download.nvidia.com/cg/atan2.html">atan2</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/blob/master/llmc/matmul.cuh#L134">llm.c/llmc/matmul.cuh at master · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://docs.google.com/document/d/10LkM5_xLh9r_ycul2ywOfgrOGmNP9YDTa9c4V755QgY/edit?usp=sharing">CUDA MODE IRL invitation</a>: This is a formal invitation to the first ever CUDA MODE IRL Hackathon and we&#39;d like to invite you to give one of our keynote talks.  The event is being sponsored by Accel and will be hosted in the...</li><li><a href="https://github.com/karpathy/llm.c/pull/705">Refactor C code by gordicaleksa · Pull Request #705 · karpathy/llm.c</a>: Realized we do activation size computation inline, so changed it to be consistent with how we compute parameter sizes.</li><li><a href="https://github.com/karpathy/llm.c/pull/702">Restore from master weights (&amp; allow restoring from a checkpoint of different precision) by ademeure · Pull Request #702 · karpathy/llm.c</a>: This is fully deterministic for new checkpoints where the new rng_state_last_update is saved, so that stochastic rounding from master weights is done with the exact same seeds (while restoring the ...</li><li><a href="https://github.com/Dao-AILab/flash-attention/blob/74b0761ff7efc7b90d4e5aeb529c1b2a09a7458c/README.md?plain=1#L201">flash-attention/README.md at 74b0761ff7efc7b90d4e5aeb529c1b2a09a7458c · Dao-AILab/flash-attention</a>: Fast and memory-efficient exact attention. Contribute to Dao-AILab/flash-attention development by creating an account on GitHub.</li><li><a href="https://github.com/jorahn/llm.c">GitHub - jorahn/llm.c: LLM training in simple, raw C/CUDA</a>: LLM training in simple, raw C/CUDA. Contribute to jorahn/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/pull/694">Model init cleanup by ngc92 · Pull Request #694 · karpathy/llm.c</a>: consolidate model parameter allocation to a single source location made gradient buffer accumulation eager moved encoder determinism helper buffers so that they are eagerly allocated by forward -&gt; ...</li><li><a href="https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g3f0bdfc73288f9dda45e5c9be7811c9d">CUDA Math API :: CUDA Toolkit Documentation</a>: no description found</li><li><a href="https://x.com/rosieyzh/status/1811790177246888075">Tweet from Rosie Zhao @ ICML (@rosieyzh)</a>: In our new work on evaluating optimizers for LLM training, we perform a series of experiments to investigate the role of adaptivity in optimizers like Adam in achieving good performance and stability....</li><li><a href="https://arxiv.org/abs/2407.07972">Deconstructing What Makes a Good Optimizer for Language Models</a>: Training language models becomes increasingly expensive with scale, prompting numerous attempts to improve optimization efficiency. Despite these efforts, the Adam optimizer remains the most widely us...</li><li><a href="https://github.com/EurekaLabsAI/mlp/pull/11">Add C implementation by gordicaleksa · Pull Request #11 · EurekaLabsAI/mlp</a>: Added a C implementation following the style of llm.c&#39;s train_gpt2.c. I did deviate a bit where i thought I can do a potentially better job. Tested for equivalency -&gt; logits, loss, grads are al...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1264244442253234196)** (10 messages🔥): 

> - `ROCm hardware entry options`
> - `Differences between RDNA and CDNA`
> - `MI300 capabilities`
> - `Challenges with AMD GPUs in ROCm`
> - `FP8 MMA acceleration on MI300` 


- **ROCm Hardware Entry Options Discussed**: Members questioned whether **Radeon RX 7900 XTX** and **Radeon PRO W7900** are the best options for ROCm development, comparing them to the **MI300X**.
   - *Another asked about the value of having a weaker AMD GPU locally versus using cloud solutions for ROCm kernel development.*
- **Diverging Paths of RDNA and CDNA**: RDNA and CDNA GPUs have diverged since GCN, with RDNA GPUs having more registers per CU, more shared memory, and supporting both w32 and w64.
   - It was noted that RDNA dGPUs lack XNACK, a feature allowing detailed page fault handling, and this deficiency restricts features like ASAN-enabled kernel compilation.
- **MI300 Series Capabilities Highlighted**: **MI300A** has memory shared with the CPU while **MI300X** does not and lacks texture units, with no Vulkan driver available for MI300.
   - *The MI300X is primarily available through cloud providers and these differences impact certain texture cache operations but are generally irrelevant to HIP.*
- **Challenges in Using AMD GPUs for ROCm Development**: Recent AMD GPUs can be used for ROCm kernel development, though some like RX6500 lack certain operations and support issues arise with APUs.
   - The need to recompile the ROCm stack for unsupported GPUs and lack of performance tuning were also cited as challenges.
- **Exclusive Feature: FP8 MMA Acceleration on MI300**: One unique feature of the MI300 is its support for **FP8 MMA acceleration**, setting it apart from other GPUs.


  

---


### **CUDA MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1263933386515419208)** (1 messages): 

> - `Arithmetic Intensity`
> - `GPU Performance Metrics` 


- **Question on Arithmetic Intensity's use in GPU performance**: A member asked why **Arithmetic Intensity 1** is used to check if a workload is memory or compute bound, questioning whether it should depend on the specific GPU's FLOPS/GB/s bandwidth ratio.
   - The member pointed out that this ratio can be as high as **20**, depending on the GPU model.
- **Impact of GPU FLOPS/GB/s ratio on performance assessment**: The discussion highlighted how the GPU's FLOPS/GB/s ratio, which can vary, affects the determination of memory or compute-bound workloads.
   - *This variation necessitates considering the specific GPU model when assessing performance*.


  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1263994137670320259)** (168 messages🔥🔥): 

> - `GGUF file metadata in C#`
> - `Quantization process (q5, q6, f16)`
> - `Issues with LM Studio & Hugging Face`
> - `Running large models locally`
> - `Using local LLMs for profit` 


- ****GGUF file metadata parsed in C#****: A user announced the parsing of GGUF file metadata in C#, introducing a console tool designed to report metadata and perhaps offer some statistics.
   - Initially intended as a ShellExtension for file properties, the user faced registration issues and decided to focus on the console tool.
- **Quantization process confusion**: A detailed explanation was given about the quantization process (e.g., **q5**, **q6**, **f16**) used to make large models fit on older hardware, with **q5** being more quantized than **q8**.
   - One user clarifies that **f16 is not quantized** and much larger, while another shares experiences running large models on limited VRAM with a **Dell laptop and RTX 3050**.
- **LM Studio and Hugging Face search issues**: Multiple users reported that LM Studio's search was broken, likely due to issues with the Hugging Face API.
   - Despite troubleshooting steps like using VPNs and reinstalling, the search functionality intermittently failed until it was confirmed to be resolved.
- **Effectively running large models locally**: Users shared their setups for running large models locally, balancing between VRAM and RAM usage efficiently using tools like LM Studio and llama.cpp.
   - One user recommended updating NVIDIA and CUDA drivers to solve loading issues with large models, making local inference smoother.
- **Local LLMs for profit and utility**: The discussion explored reasons for using local LLMs: maintaining data privacy for corporate use and offline capabilities for personal projects.
   - One user highlighted that unlike cloud models like **ChatGPT**, local LLMs keep data on the user's machine, emphasizing profit is generally through utility, not direct monetary gain like crypto mining.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.ibm.com/topics/ai-hallucinations">What Are AI Hallucinations? | IBM</a>: AI hallucinations are when a large language model (LLM) perceives patterns or objects that are nonexistent, creating nonsensical or inaccurate outputs.</li><li><a href="https://github.com/ggerganov/llama.cpp/commit/50e05353e88d50b644688caa91f5955e8bdb9eb9">llama : add Mistral Nemo inference support (#8604) · ggerganov/llama.cpp@50e0535</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e73zxd/is_there_any_working_gemma_27b_in_gguf_format/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/lmstudio-community/gemma-2-27b-it-GGUF">lmstudio-community/gemma-2-27b-it-GGUF · Hugging Face</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/8608">Support for SmolLM · Issue #8608 · ggerganov/llama.cpp</a>: Prerequisites I am running the latest code. Mention the version if possible as well. I carefully followed the README.md. I searched using keywords relevant to my issue to make sure that I am creati...</li><li><a href="https://x.com/alpindale/status/1814814551449244058?s=12">Tweet from Alpin (@AlpinDale)</a>: Have confirmed that there&#39;s 8B, 70B, and 405B. First two are distilled from 405B. 128k (131k in base-10) context. 405b can&#39;t draw a unicorn. Instruct tune might be safety aligned. The architec...</li><li><a href="https://x.com/alpindale/status/1814717595754377562?s=46">Tweet from Alpin (@AlpinDale)</a>: It seems like someone at HF forgot to private this repo on time, and Google indexed it:  sllhf/Meta-Llama-3.1-405B-Instruct-FP8  The 405B is llama 3.1? Very interesting. I wonder if they&#39;ll only r...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/8579">llama : Added support for Tekken pre-tokenizer (#8577) by m18coppola · Pull Request #8579 · ggerganov/llama.cpp</a>: Adding Tekken pre-tokenizer support for Mistral Nemo models.  Added tokenizer type for Mistral-Nemo-Base-2407 in convert-hf-to-gguf-update.py Added the chkhsh for Mistral-Nemo-Base-2407 in convert-...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1263936712871252018)** (126 messages🔥🔥): 

> - `Open Model Test Results`
> - `Converting Models to GGUF`
> - `New Jail-Breaking Technique`
> - `DeepSeek-Coder Issues`
> - `Memory Usage with Qwen2 72B` 


- **Open Model Test Results Shared**: A member shared new results of various **open models tested** and inquired if others had any major discrepancies outside of **Gemma 2 27B**.
   - They mentioned that they couldn't fix the **borked Gemma 2 27B** issue.
- **Converting Models to GGUF for LM Studio**: A user inquired about converting `microsoft/llava-med-v1.5-mistral-7b` to **GGUF for LM Studio**, mentioning they needed it for a hobby project.
   - Another member confirmed if they succeeded in obtaining the **GGUF conversion**.
- **New Jail-Breaking Technique Unveiled**: A member announced a new **jail-breaking** technique for frontier models and shared a [paper link](https://arxiv.org/pdf/2407.11969).
   - They suggested using it before it gets patched.
- **DeepSeek-Coder V2 Lite Instruct Model Malfunctions**: Users encountered issues with **DeepSeek-Coder V2 Lite Instruct** generating incoherent text after initial normal output.
   - Troubleshooting efforts included changing context windows, disabling flash, and downgrading **LM Studio versions**, with no lasting success.
- **High Memory Usage for Qwen2 72B in llama.cpp**: A member reported excessively high memory usage when loading **Qwen2 72B** using `llama.cpp`.
   - Another member advised lowering the **context length** to manage memory usage more effectively.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/win10/DeepSeek-Coder-V2-Lite-Instruct-Q8_0-GGUF">win10/DeepSeek-Coder-V2-Lite-Instruct-Q8_0-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF#model-settings">lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF/discussions/2">lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF · Problem with LLM Studio</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1264981042142515202)** (2 messages): 

> - `Hugging Face API Networking Errors`
> - `Issue Resolution` 


- **Hugging Face API hit with Networking Errors**: An announcement was made about **intermittent networking errors** affecting the **Hugging Face API**, causing search issues within the app.
   - *We will update here as we know more.*
- **Issue with Hugging Face API Resolved**: The networking errors with the **Hugging Face API** have been resolved, and users should be able to search through the app again.
   - *Sorry for the inconvenience.*


  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1264054340847669390)** (11 messages🔥): 

> - `Failed model load`
> - `Flash Attention troubleshooting`
> - `HuggingFace API issues`
> - `Alternative model repositories` 


- **Failed to load model due to llama.cpp error**: A member shared an error about failing to load a model in LM Studio due to an unknown pre-tokenizer type: 'mistral-bpe' in llama.cpp.
   - Another member mentioned that newly released models would not work in current LM Studio versions and needed llama.cpp architecture support updates.
- **Flash Attention may cause model load issues**: A user suggested that having Flash Attention enabled might cause models not to load and recommended turning off Flash Attention to resolve this.
   - *Try turning it off and reloading your model* to see if it resolves the issue.
- **HuggingFace API stability issues affecting LM Studio**: HuggingFace API stability issues are causing the Model Explorer in LM Studio to be non-functional at this time.
- **Request for alternative model repository mirrors**: A user suggested adding the ability to switch to third-party repository mirrors within LM Studio due to ongoing HuggingFace API issues.
   - An example given was [hf-mirror.com](https://github.com/padeoe/hf-mirror-site), with further details on using mirror scripts like [hfd.sh](https://gist.github.com/padeoe/697678ab8e528b85a2a7bddafea1fa4f) for downloads.



**Link mentioned**: <a href="https://huggingface.co">Hugging Face – The AI community building the future.</a>: no description found

  

---


### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1264345538682294374)** (2 messages): 

> - `Renaming Presets`
> - `Finding Models` 


- **Renaming Presets**: A member asked if they could rename presets they had created for different use cases.
   - They noted that they figured it out soon after asking.
- **Finding Models**: The same member mentioned that they had finally found a nice model suited for their needs.
   - They planned to create 4 prompts focusing on their specific use cases.


  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1263938288167686263)** (40 messages🔥): 

> - `NVidia Tesla P40 usage`
> - `AMD GPU compatibility`
> - `Home NAS recommendations`
> - `Choosing GPUs for AI/ML workloads`
> - `Finetuning Large Language Models` 


- **Tesla P40 runs with mixed GPUs**: Some users successfully run NVidia Tesla P40 alongside other GPUs like the RTX 3070 on Windows 10 x64 by installing specific drivers, such as the default P40 datacenter driver and then the studio RTX driver.
- **AMD GPUs of different generations can be used**: A user asked about running LM Studio on an RX 6800 and potentially an RX 7900XTX, to which another user replied that if both support ROCM, it should work.
- **Setting up a home NAS for iPhones/iPads**: A user discussed setting up a NAS for home use to store content from iPhones and iPads, rather than buying devices with larger memory.
- **Upgrading GPU setups for AI workloads**: Debates ensued about the suitability of various GPU combinations for AI/ML workloads, with some users considering dual RTX 3090s for enhanced VRAM and performance.
   - One user discussed the challenge of high power consumption with certain GPUs like the RTX 3090 compared to 4060 but was advised that 3090s would significantly improve their ability to handle larger models.
- **Challenges in finetuning quantized models**: Members discussed the difficulties of finetuning GGUF quantized models, with guidance suggesting it's more effective to use base safetensors models for finetuning.
   - One user shared a helpful [Microsoft article](https://learn.microsoft.com/en-us/ai/playbook/technology-guidance/generative-ai/working-with-llms/fine-tuning) on getting started with large language model finetuning.



**Link mentioned**: <a href="https://learn.microsoft.com/en-us/ai/playbook/technology-guidance/generative-ai/working-with-llms/fine-tuning">Getting started with LLM fine-tuning</a>: Large Language Model (LLM) Fine-tuning is the process of adapting the pre-trained model to specific tasks. This process is done by updating its parameters on a new dataset. Specifically, the LLM is pa...

  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1264315511152443515)** (18 messages🔥): 

> - `Nemo issues`
> - `Search functionality problems`
> - `Huggingface API` 


- **Users report issues with Nemo from Mistral**: *madan.pandit* inquired about problems they were facing with the Nemo module from **Mistral**.
- **Search function fails in LM Studio 0.2.27**: Multiple users reported that **LM Studio 0.2.27** search function was broken, yielding 0 results on Linux and Mac M2.
   - Reports indicate *search works on Huggingface directly* but not within the app.
- **Huggingface API down affects LM Studio**: Issues were confirmed by *heyitsyorkie* who noted that the **Huggingface API** was down, impacting the Model Explorer in LM Studio.
   - *a_dev_called_dj_65326* acknowledged the API issue and indicated that the team had been notified.


  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/)** (1 messages): 

captainpumpkinhead: "natively"
  

---


### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1264584877794267156)** (1 messages): 

> - `Athese by Nexusflow`
> - `Model benchmarks`
> - `Multilingual performance` 


- **Nexusflow introduces Athese model**: Athese by Nexusflow shows impressive improvements across a wide array of categories and may be the current SOTA for its size for general use.
   - Nexusflow used an internally developed benchmark to create a high quality dataset of RLHF, resulting in benchmarks that speak for themselves. [Check out the model here](https://huggingface.co/lmstudio-community/Athene-70B-GGUF).
- **Athese excels in multilingual performance**: Athese by Nexusflow boasts massively improved **multilingual performance**, making it an excellent choice for use beyond English.


  

---


### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1264760981859336223)** (2 messages): 

> - `LM Studio Discord bot`
> - `Private responses with LM Studio`
> - `GitHub tutorial link` 


- **Create Discord bot with LM Studio**: A developer shared a blog post titled '[I made a Discord bot with LM Studio.js](https://dev.to/mrdjohnson/i-made-a-discord-bot-with-lmstudiojs-4fd6)', detailing how to create a Discord bot that responds using LM Studio.
   - The blog post includes a tutorial and source code available on [GitHub](https://github.com/mrdjohnson/lmstudio-discord-bot), providing insights into modifications necessary for the bot to respond privately.
- **Thanks for sharing**: Another member thanked the developer for sharing the blog post about creating a Discord bot using LM Studio.
   - They expressed appreciation for the tutorial and the provided GitHub link.



**Link mentioned**: <a href="https://github.com/mrdjohnson/lmstudio-discord-bot">GitHub - mrdjohnson/lmstudio-discord-bot: A tutorial for creating a Discord bot that responds using LM Studio! This code is based on a blogpost found here: https://dev.to/mrdjohnson/i-made-a-discord-bot-with-lmstudiojs-4fd6</a>: A tutorial for creating a Discord bot that responds using LM Studio! This code is based on a blogpost found here: https://dev.to/mrdjohnson/i-made-a-discord-bot-with-lmstudiojs-4fd6 - mrdjohnson/lm...

  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1263935249000435762)** (338 messages🔥🔥): 

> - `Image generation with Pro subscription`
> - `GPTs Agents`
> - `Issues with Pro search`
> - `Profile and collection prompts`
> - `Perplexity's context window` 


- ****Image generation issue resolved by browser restart****: A user reported being unable to generate more than one image with a Pro subscription, but restarting the Chrome browser resolved the issue.
- ****GPTs Agents cannot learn after initial training****: A member shared a concern about GPTs agents not learning from additional information provided after their initial training.
- ****Pro search limitations and preferences****: Several users discussed that Pro search can sometimes disrupt focus and not adhere to user prompts, making it less preferable for certain tasks.
- ****Profile and collection prompts not always effective****: Users discussed that profile and collection prompts often do not influence the AI's search process during Pro search, limiting their utility.
- ****Perplexity's context window and token usage****: Perplexity's context window is limited to 32k tokens per turn, but cumulative multi-turn conversations can handle up to 128k tokens.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://paul.kinlan.me/use-bookmarklets-on-chrome-on-android/">Use Bookmarklets on Chrome on Android</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=ts3DqM_t3fA&t=38s">This Could Make AI Actually Useful</a>: Use our link https://ground.news/TechLinked to get 40% off the Vantage plan. Access local perspectives to better understand world politics and current events...</li><li><a href="https://prateekkeshari.gumroad.com/l/peek">Peek - Free App to Access AI (ChatGPT, Gemini, Perplexity, Poe, Claude) and Threads anywhere on MacOS </a>: Updated with a new version as of April 30, 2024 🚀Peek is a MacOS Menu Bar application that allows you to interact with multiple AI chatbots in one place. List of AI you can access:ChatGPTGeminiPerple...</li><li><a href="https://x.com/perplexity_ai/status/1815431484767142272?s=61">Tweet from Perplexity (@perplexity_ai)</a>: When you know, you know.</li><li><a href="https://llmtokencounter.com/">LLM Token Counter</a>: no description found</li><li><a href="https://zapier.com/blog/add-search-engine-to-chrome/">How to add a custom search engine to Chrome | Zapier</a>: When you add a custom search engine to Chrome, you can easily search your go-to sites or apps directly from Chrome. Here&#x27;s how to do it. </li><li><a href="https://arxiv.org/html/2407.10887">Hey, That’s My Model! Introducing Chain &amp; Hash, An LLM Fingerprinting Technique</a>: no description found
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1263941449926312067)** (23 messages🔥): 

> - `YouTube tests AI conversational capabilities`
> - `OpenAI drops GPT-4.0 mini`
> - `Unraveling Chaos project`
> - `CrowdStrike global IT outage`
> - `Possibilities of developing software` 


- **YouTube AI Conversations Take a Test Drive**: Perplexity AI explored [YouTube testing new AI conversational features](https://www.perplexity.ai/page/youtube-tests-ai-conversationa-WMQ_b8XNQZuIhMpPPMyfGg) to understand their functionality and user interactions.
- **OpenAI Launches GPT-4.0 Mini**: OpenAI recently released the [GPT-4.0 Mini](https://www.perplexity.ai/page/openai-drops-gpt-4o-mini-viKDYptISzufyJDPoL3Etg), a compact version aimed at improving accessibility while maintaining advanced AI capabilities.
- **CrowdStrike Faces Major Global IT Outage**: CrowdStrike experienced a significant [global IT outage](https://www.perplexity.ai/page/crowdstrike-global-it-outage-qKRKi2QWRuaWxf44d1G5nQ), impacting numerous clients and highlighting vulnerabilities in its service infrastructure.
   - Community discussions emphasize the need for better resilience and contingency planning for cloud-based security providers.
- **Mistral NeMo's AI Leap Dissected**: Perplexity AI's [latest YouTube video](https://www.youtube.com/embed/TA4E69jtF_U) covers Mistral NeMo's significant advancements in AI technology and CrowdStrike's global outage alongside a surprising sulfur discovery on Mars.
   - *'This shows exciting times ahead for AI developments and space exploration,'* says an enthusiast.
- **Stegosaurus Fossil Auctioned for $44.6M**: A Stegosaurus fossil was sold for an astonishing [ $44.6 million](https://www.perplexity.ai/search/stegosaurus-sells-for-44-6m-_E76GAUBQ4mAHurns1O.ZA), capturing the interest of both paleontology enthusiasts and art collectors.
   - The record-breaking sale underscores the high value placed on rare historical artifacts.



**Link mentioned**: <a href="https://www.youtube.com/embed/TA4E69jtF_U">YouTube</a>: no description found

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1264253137041227807)** (9 messages🔥): 

> - `Feature Roadmap`
> - `Perplexity API Token Charges`
> - `Online Models Usage` 


- **Feature Roadmap Page Disappears**: **Feature Roadmap** page disappeared from the [documentation site](https://docs.perplexity.ai/docs/feature-roadmap).
- **Perplexity API Charges In and Out Tokens**: **Perplexity API** charges for both in and out tokens as clarified in a discussion.
- **Using Online Models to Explain PDFs**: A member used Perplexity's online models to explain PDF documents.


  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1263933578358554716)** (170 messages🔥🔥): 

> - `Sonnet 3.5 vs 4O Mini`
> - `Finetuning multimodal models`
> - `Cost-effective TTS solutions`
> - `Voice assistant apps`
> - `GPT-4o mini vs GPT-5` 


- **4O Mini outshines Sonnet 3.5**: **4O Mini** can solve complex questions that even the entire Claude family struggles with, highlighting its superior capabilities.
- **Finetuning multimodal models for coral reef research**: A user inquired about finetuning multimodal models to study coral reefs, but another suggested using Azure AI's Custom Vision Service for better accuracy.
- **Choosing cost-effective TTS solutions**: A member recommended using OpenAI's [text-to-speech](https://platform.openai.com/docs/guides/text-to-speech) service for voice handling in chats.
   - Other options like **Coqui.ai** and running models locally were also discussed for cost efficiency and better performance.
- **Voice assistant apps with GPT integration**: Members discussed voice assistant apps, mentioning **VoiceGPT** as an open-source option that overlaid ChatGPT listening animations on Android screens.
   - The assistant couldn't open apps directly, so alternatives like **macrodroid/tasker** or integrating with **Home Assistant** were suggested.
- **Excitement builds around GPT-4o mini and GPT-5**: The community showed excitement for **GPT-4o mini**, discussing its potential impact and speculating about future advancements like **GPT-5**.



**Link mentioned**: <a href="https://search.arc.net/0p86iGsUHFk34vkGOclc">System Prompt in Markdown Code Block | Arc Search</a>: Arc Search read websites across the internet to make you this perfect tab.

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1264073037695811585)** (32 messages🔥): 

> - `GPT-4o mini replacing GPT-3.5`
> - `Differences between GPT-4o and GPT-4o mini`
> - `New features for GPT-4o`
> - `API vs. ChatGPT features`
> - `GPT-4o mini for longform content` 


- **GPT-4o Mini to Replace GPT-3.5**: A member asked if **GPT-4o mini** will become the new free model and phase out **GPT-3.5**, receiving a simple confirmation.
   - *lumirix*: 'Yes.'
- **API Lacks Real Time Voice Integration**: A discussion highlighted that OpenAI’s new voice features are available in ChatGPT but not in the API.
   - One member mentioned that real-time chat voice features are unlikely to come to the API soon due to functional differences between the platforms.
- **ChatGPT Voice vs API Text-to-Speech**: Concerns were raised about the latency and quality differences between ChatGPT's new voice feature and the API’s Text-to-Speech endpoint.
   - *the_big_block_pc*: 'I know of the TTS endpoint but that can add a lot of latency and doesn't sound as good as the new ChatGPT voice feature.'
- **Features Rollout: API vs ChatGPT**: One member questioned why new features like voice are added to ChatGPT before the API.
   - It was suggested that user demand for ChatGPT features might be higher and that real-time chat is more of an end-user experience rather than a developer tool.
- **GPT-4o Mini for Longform Content**: Discussion emerged on whether **GPT-4o mini** can generate longform content like YouTube scripts or stories.
   - A member expressed interest in using the API for such tasks, to incorporate Text-to-Speech and maybe a video model.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1264196613518131321)** (9 messages🔥): 

> - `Solving Mathematical or Logical Problems Accurately`
> - `Custom Instructions in ChatGPT`
> - `Using Custom Instructions Effectively`
> - `Guidance AI for Prompt Engineering` 


- **Mathematical problem accuracy issue**: A user discussed a method for solving mathematical or logical problems step-by-step but was concerned about losing accuracy if only the final answer was given.
- **Custom Instructions issue when modifying previous responses**: A user shared a problem with ChatGPT rewriting entire texts instead of modifying specific parts and asked for a solution.
   - Another member suggested using the 'Customize ChatGPT' section in the settings, but the issue persisted for the user.
- **Detailed help offered for Custom Instructions issue**: A suggestion was made to share the custom instructions, a shared link of the chat, and specifics on desired changes for further help.
- **Guidance AI for prompt engineering**: A user inquired about the use of Guidance AI or other tools for prompt engineering.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1264196613518131321)** (9 messages🔥): 

> - `Improving ChatGPT Accuracy`
> - `Modification of ChatGPT Responses`
> - `Custom Instructions for ChatGPT`
> - `Prompt Engineering Tools` 


- **Tackling Math Problems with Exact Answers**: A user raised a concern about generating isolated final answers for math problems, but noted the method caused a drop in accuracy.
- **Improving ChatGPT Response Modifications**: A user, Sparrow.hwk, asked for help with instructing ChatGPT to modify only specific parts of text without repeating the entire response.
- **Sharing Custom Instructions for Better Help**: Sparrow.hwk tried custom instructions but faced repeated issues.
- **Prompt Engineering Tools Used**: A user asked if others use tools like Guidance AI for prompt engineering or if they use alternative methods.


  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1264026781984493653)** (1 messages): 

> - `Rankings page update`
> - `Infrastructure migration` 


- **Rankings Page Slows Due to Migration**: The rankings page will be slow to update over the weekend and often present **stale data** while the migration to new infrastructure occurs.
   - Users should expect delays and inaccuracies in rankings during this timeframe.
- **Infrastructure Migration Causes Delays**: The notice pointed out that the infrastructure migration will specifically impact the **rankings page** over the weekend.


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1264226677194620938)** (2 messages): 

> - `OpenRouter provider for GPTScript`
> - `gptscript on command line`
> - `gptscript demo video` 


- **OpenRouter provider for GPTScript now available**: A new [OpenRouter provider for GPTScript](https://github.com/RobinVivant/gptscript-openrouter-provider) has been announced, with an image and detailed description on GitHub.
   - This tool contributes significantly to the development of GPTScript applications.
- **GPTScript impresses on command line**: Discussion highlights the [gptscript GitHub repo](https://github.com/gptscript-ai/gptscript), noted for its capability to build AI assistants that interact directly with systems.
   - One member mentioned an *impressive demo video* available on the repository page.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/RobinVivant/gptscript-openrouter-provider">GitHub - RobinVivant/gptscript-openrouter-provider</a>: Contribute to RobinVivant/gptscript-openrouter-provider development by creating an account on GitHub.</li><li><a href="https://github.com/gptscript-ai/gptscript">GitHub - gptscript-ai/gptscript: Build AI assistants that interact with your systems</a>: Build AI assistants that interact with your systems - gptscript-ai/gptscript
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1263970764978257933)** (202 messages🔥🔥): 

> - `Hermes 2.5`
> - `GPTs Agents`
> - `OpenRouter Feature Requests`
> - `Model Merging`
> - `Dolphin Llama 70B` 


- **Dolphin Llama 70B's Performance Issues**: Using Dolphin Llama 70B at 0.8-1 temperature led to erratic behavior in a 7k token context chat, producing incoherent content split between code and unrelated output.
   - Another member noted similar issues with Euryale 70B's fp8 quantized models, suggesting potential problems stemming from the quantization process.
- **DeepSeek's Low Cost and Efficiency**: DeepSeek v2, a 236B parameter MoE model (21B activated per token), is praised for its excellent performance and cost-efficiency at $0.14 per input token.
   - *“DeepSeek’s pricing is very competitive, and it seems to be hugely profitable,”* explaining their strategy of using high batch sizes and compression techniques.
- **GPT-4o mini's Multilingual Capabilities and Code Performance**: Members discussed that **GPT-4o mini** performs worse in coding tasks compared to **DeepSeek**, but better in multilingual capabilities than **gemma2-27b**.
   - One member noted, *“4o mini seems better at multilingual capabilities compared to gemma2-27b, but worse at reasoning.”*
- **Leaked Information on Llama 3.1 405B**: Llama 3.1 405B Base apparently leaked early due to a misstep on HuggingFace, sparking discussions about its extended context capabilities via RoPE scaling.
   - Members are excited, anticipating software updates for efficient utilization and eager for the official instruct model release.
- **Issues with Free vs Paid Model Limits**: A user discovered that free model variants like **google/gemma-2-9b-it:free** have stricter token limits (4096 tokens) compared to their paid counterparts (8192 tokens).
   - The disparity led to confusion and error messages, prompting discussions on potential misconceptions or misconfigurations in how token limits are enforced.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/google/gemma-2-9b-it:free">Google: Gemma 2 9B (free) by google</a>: Gemma 2 9B by Google is an advanced, open-source language model that sets a new standard for efficiency and performance in its size class.  Designed for a wide variety of tasks, it empowers developers...</li><li><a href="https://openrouter.ai/docs/requests#images-_-multimodal-requests">Requests | OpenRouter</a>: Handle incoming and outgoing requests</li><li><a href="https://openrouter.ai/docs/limits">Limits | OpenRouter</a>: Set limits on model usage</li><li><a href="https://openrouter.ai/models?order=newest&supported_parameters=tools),">Models | OpenRouter</a>: Browse models on OpenRouter</li><li><a href="https://openrouter-3d-generator.vercel.app/browse">no title found</a>: no description found</li><li><a href="https://www.ben-evans.com/benedictevans/2024/7/9/the-ai-summer">The AI summer &mdash; Benedict Evans</a>: Hundreds of millions of people have tried ChatGPT, but most of them haven’t been back. Every big company has done a pilot, but far fewer are in deployment. Some of this is just a matter of time. But L...</li><li><a href="https://github.com/NolanGC/window-3d-demo">GitHub - NolanGC/window-3d-demo: Generative 3D in the web via window.ai, Next, Three, Neon, Drizzle, and GCS.</a>: Generative 3D in the web via window.ai, Next, Three, Neon, Drizzle, and GCS. - NolanGC/window-3d-demo</li><li><a href="https://openrouter.ai/docs/parameters">Parameters | OpenRouter</a>: Configure parameters for requests</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e9hg7g/azure_llama_31_benchmarks/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1263940542023209132)** (156 messages🔥🔥): 

> - `Open source contributions`
> - `LLM use in RPG games`
> - `Nextjs app router update`
> - `Developer office hours format`
> - `Integrating Cohere models` 


- **Open Source Contributions Welcomed for Toolkit**: Community members highlighted the openness of the toolkit, encouraging others to build, change, and contribute to the project.
   - "I'd happily contribute if I could," one member expressed, showing enthusiasm for collaborating on the toolkit.
- **Using LLMs in RPG Games**: Lyrcaxis discussed using LLMs for classifications, JSON, and dialogue generation for their AI-powered RPG project.
   - After offloading classification tasks to local models, CMD-R is their top choice for dialogue generation due to its strict instruction following.
- **Nextjs App Router Update**: Hamedmp appreciated the timely update of the toolkit to Nextjs app router, expressing excitement about incorporating components into projects.
   - "I'd still love to see more support for Cohere in the AI SDK," they mentioned, noting the need for extended tool call support.
- **Tweaking Developer Office Hours Format**: The switch from Discord to Twitter Spaces for developer office hours was debated, with concerns about accessibility and platform dependencies.
   - "Maybe a mirror could be possible," suggested one member, advocating for combined Discord and Twitter formats for broader reach.
- **Integrate Cohere Models into Writing Work**: Petersmall discussed using Gemini and ChatGPT for narrative creation and considered adding a Cohere-based product for enhanced outputs.
   - After testing, they found that responses from Cohere products like CMD-R provided impressive results and complemented other AI tools.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.bloomberg.com/news/articles/2024-07-22/ai-startup-cohere-valued-at-5-5-billion-in-new-funding-round">Bloomberg - Are you a robot?</a>: no description found</li><li><a href="https://youtu.be/KFXeTHzIgf0">Westworld S02E04 last person</a>: You live only as long as the last person who remembers you.-- Westworld, Season 2 Episode 4</li><li><a href="https://jsfiddle.net/razodactyl/2s60uyw5/">88 Key Piano - JSFiddle - Code Playground</a>: no description found</li><li><a href="https://x.com/i/spaces/1eaJbaEkgdVGX]">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter</li><li><a href="https://x.com/i/spaces/1eaJbaEkgdVGX">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1264014615055171596)** (4 messages): 

> - `chat GUI with local LLMs`
> - `multiplayer text games Discord app` 


- **Exciting Chat GUI with Local LLMs Project**: A member shared an ongoing project of a chat GUI powered by local **LLMs** with features like Web Search, Python Interpreter, and Image Recognition. They've provided the [GitHub repository](https://github.com/yamikumo-DSD/chat_cmr) for more details.
- **Create and Play Multiplayer Text Games on Discord**: A member introduced a Discord app, **Command R+**, allowing users to create and play multiplayer text games.



**Link mentioned**: <a href="https://www.producthunt.com/posts/create-n-play"> Create &#x27;n&#x27; Play - Create AI multiplayer text games for your Discord server! | Product Hunt</a>: AI-powered Discord bot turns your server into a text game paradise! Craft ANY game with our /search-games command. Endless possibilities—it&#x27;s the ultimate community engagement tool. Let AI fuel y...

  

---


### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1264927720672858125)** (1 messages): 

> - `Developer Office Hours`
> - `Structured Generations in the API`
> - `Cohere Toolkit features`
> - `Cohere For AI research papers`
> - `Community events` 


- **Cohere Developer Office Hours 2.0 Announced**: Cohere hosted another Developer Office Hours session discussing recent product updates with <@1199000650982379524>.
   - They covered new features in the API, toolkit updates, and recent **Cohere For AI** research papers, inviting the community to join.
- **Structured Generations in Cohere API**: Cohere announced that their models, **Command R** and **Command R+**, can now generate structured outputs in **JSON** format.
   - This feature (documented [here](https://docs.cohere.com/docs/structured-outputs-json)) allows better integration and data analysis for downstream applications.
- **New Cohere Toolkit Features in July 2024**: Cohere and **Fujitsu** unveiled their strategic partnership offering new enterprise AI services for the Japanese market, announced via the [blog](https://cohere.com/blog/toolkit-features-july-2024).
   - This collaboration aims to enhance AI service accessibility and performance in various applications.
- **Upcoming Community Events**: Cohere mentioned upcoming community events, encouraging participation and engagement from members.
   - They urged everyone to join the session with questions and to connect over a cup of coffee.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.cohere.com/docs/structured-outputs-json">Structured Generations (JSON)</a>: no description found</li><li><a href="https://cohere.com/blog/toolkit-features-july-2024">New Cohere Toolkit Features: Authentication, HTML and More</a>: At Cohere, we’re continuing to enable developers to accelerate generative AI application development. We’re expanding our open source Cohere Toolkit features, introducing HTML rendering, configurable ...</li><li><a href="https://x.com/cohere">Tweet from undefined</a>: no description found
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1263981952600506389)** (13 messages🔥): 

> - `Softmax invariance and z loss`
> - `Understanding large language models (LLM)`
> - `GPT model example code`
> - `Scaling diarization pipelines`
> - `Vector quantization models` 


- **Softmax is shift invariant, needs z loss**: A member pointed out that **softmax** is shift invariant, necessitating the use of **z loss**.
   - *Official reasons include keeping logits from drifting too far from zero and encouraging normalized log-probabilities*.
- **Simple GPT model explained**: A member shared a minimal **GPT model code** to help understand its workings, incorporating components like **LayerNorm**, **Linear layers**, and **scaled dot product attention**.
   - [Example code](https://github.com/alibaba/easydist/blob/3dbb146812fddf6259590a0b4611a251f3e7cbe5/benchmark/torch/model/gpt.py) resembles Alibaba’s **easydist/benchmark/GPT model**.
- **Scaling diarization and vector quantization models**: A member sought insights on scaling **diarization pipelines** and **vector quantization models**, considering running a slow pipeline with **Whisper** and **Pyannote** to pretrain an LSTM.
   - They also explored training a model to produce unified codebooks from composite audio, keyframes, and text, leveraging **Perceiver IO** as an encoder.



**Link mentioned**: <a href="https://github.com/alibaba/easydist/blob/3dbb146812fddf6259590a0b4611a251f3e7cbe5/benchmark/torch/model/gpt.py#L112">easydist/benchmark/torch/model/gpt.py at 3dbb146812fddf6259590a0b4611a251f3e7cbe5 · alibaba/easydist</a>: Automated Parallelization System and Infrastructure for Multiple Ecosystems - alibaba/easydist

  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1263949707126771752)** (39 messages🔥): 

> - `Self-Modeling in AI`
> - `Hybrid Post-Quantum Encryption`
> - `Feature Contamination in Neural Networks`
> - `Human-like Response Time in CNNs`
> - `Efficient Dictionary Learning with Switch SAE` 


- **Self-models Simplify Neural Networks**: The paper on [self-models](https://arxiv.org/abs/2407.10188) shows that networks predicting their internal states as an auxiliary task become simpler and more regularized, making them more parameter-efficient.
   - However, the results using datasets like MNIST, CIFAR, and IMDB show expectations rather than surprising insights, with some members questioning the novelty of these findings.
- **Implementing Hybrid Post-Quantum Encryption**: The [liboqs-python GitHub repository](https://github.com/qompassai/liboqs-python) describes using Kyber/Dilithium python bindings and Podman for rootless container management to implement hybrid post-quantum encryption.
   - This approach aims to minimize attack surfaces during this vulnerable time, showcasing advancements in secure encryption techniques.
- **Feature Contamination Limits OOD Generalization**: A paper on [out-of-distribution generalization](https://arxiv.org/abs/2406.03345) finds that neural networks suffer from feature contamination, where irrelevant features can hinder their performance.
   - The discussion suggests that inductive biases and SGD dynamics play a crucial role, hinting at a potential unified theory to explain model failures.
- **RTNet Mimics Human Response Times**: The [RTNet model](https://www.biorxiv.org/content/10.1101/2022.08.23.505015v2.full) reproduces human-like response times and confidence in decision-making by utilizing stochastic decisions.
   - Although the model's practical applications are debatable, it provides an accurate prediction of human behavior on image classification tasks by using sampled Gaussian weights.
- **Switch SAE for Efficient Dictionary Learning**: A new architecture, [Switch SAE](https://www.lesswrong.com/posts/47CYFbrSyiJE2X5ot/efficient-dictionary-learning-with-switch-sparse), is proposed for scaling sparse autoencoders efficiently, aiming to recover features from superintelligent language models.
   - Leveraging conditional computation, Switch SAE offers a practical solution to scale SAEs to billions of features, overcoming current computational limitations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.13623">Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies</a>: Research on scaling large language models (LLMs) has primarily focused on model parameters and training data size, overlooking the role of vocabulary size. % Intuitively, larger vocabularies enable mo...</li><li><a href="https://arxiv.org/abs/2407.10188">Unexpected Benefits of Self-Modeling in Neural Systems</a>: Self-models have been a topic of great interest for decades in studies of human cognition and more recently in machine learning. Yet what benefits do self-models confer? Here we show that when artific...</li><li><a href="https://arxiv.org/abs/2406.03345">Feature Contamination: Neural Networks Learn Uncorrelated Features and Fail to Generalize</a>: Learning representations that generalize under distribution shifts is critical for building robust machine learning models. However, despite significant efforts in recent years, algorithmic advances i...</li><li><a href="https://arxiv.org/abs/2208.10291">Efficient Planning in a Compact Latent Action Space</a>: Planning-based reinforcement learning has shown strong performance in tasks in discrete and low-dimensional continuous action spaces. However, planning usually brings significant computational overhea...</li><li><a href="https://github.com/qompassai/liboqs-python">GitHub - qompassai/liboqs-python: A Qompass fork of open-quantum-safe/liboqs-python</a>: A Qompass fork of open-quantum-safe/liboqs-python. Contribute to qompassai/liboqs-python development by creating an account on GitHub.</li><li><a href="https://www.biorxiv.org/content/10.1101/2022.08.23.505015v2.full">RTNet: A neural network that exhibits the signatures of human perceptual decision making</a>: Convolutional neural networks currently provide the best models of biological vision. However, their decision behavior, including the facts that they are deterministic and use equal number of computat...</li><li><a href="https://www.lesswrong.com/posts/47CYFbrSyiJE2X5ot/efficient-dictionary-learning-with-switch-sparse">Efficient Dictionary Learning with Switch Sparse Autoencoders — LessWrong</a>: Produced as part of the ML Alignment &amp; Theory Scholars Program - Summer 2024 Cohort …
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1263950155401662464)** (3 messages): 

> - `Scaling laws and hypernetworks`
> - `Brains and backprop`
> - `Scaling Exponents Across Parameterizations and Optimizers` 


- **Hypernetworks constrained by scaling laws**: A user discussed the constraints hypernetworks face due to scaling laws, arguing that for a hypernetwork to achieve a target error, its size must be **less than O(scaling_law(output_model_compute)(target_error))**.
   - The user mentioned that for hypernetworks to be effective, the task of predicting a neural network has to be simpler or the output model's scaling law must be **'really nice'**.
- **Brains only approximate backprop**: A statement was made that **brains only approximate backprop**, suggesting a possibility of alternative learning mechanisms beyond traditional backpropagation.
- **Scaling Exponents Across Parameterizations and Optimizers**: A [tweet](https://x.com/main_horse/status/1810647037718999342) discussed a paper on scaling exponents across different parameterizations and optimizers, involving **10,000+ models** with varying optimizers, model sizes, and parameterizations.
   - Key findings include **O(1/n) LR schedule outperforming mUP**, successful hparam transfer across all tested configurations, and a proposed Adam-atan2 optimizer avoiding gradient underflow issues seen in Adam.



**Link mentioned**: <a href="https://x.com/main_horse/status/1810647037718999342">Tweet from main (@main_horse)</a>: Scaling Exponents Across Parameterizations and Optimizers  [GDM] [nocode/weights] https://arxiv.org/abs/2407.05872  trains 10,000+ (!) models, varying * optim (SGD/Adam/Adafactor) * model size (1.1B ~...

  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1263938275144499312)** (9 messages🔥): 

> - `MATS 7.0 Applications`
> - `**nnsight** Paper Release`
> - `Apollo's Mech Interp Projects List`
> - `Tokengrams Project`
> - `Suffix Arrays for Pile` 


- **MATS 7.0 Applications Open Now**: **Neel Nanda** and **Arthur Conmy** have opened applications for their Winter MATS 7.0 streams, with the deadline on August 30th.
   - Nanda highlighted the program's unique value in fostering mechanistic interpretability research and provided an [announcement link](https://x.com/NeelNanda5/status/1813921161052635209) and an [admissions doc](https://tinyurl.com/neel-mats-app).
- **New Mech Interp Paper: **nnsight****: A new paper titled **nnsight** will be available on arXiv soon.
   - It's expected to cover significant advancements in mechanistic interpretability.
- **Apollo Shares 45 New Mech Interp Project Ideas**: Apollo Research shared a list of 45 mechanistic interpretability project ideas in a recent [Alignment Forum post](https://www.alignmentforum.org/posts/KfkpgXdgRheSRWDy8/a-list-of-45-mech-interp-project-ideas-from-apollo-research).
   - The post prompts a [discussion](https://docs.google.com/document/d/1XRb-EDDw-h6c-L6pKltI9A8zJmufE71I2_4wx8d5rMg/edit?usp=sharing) on making small language models more useful for interpretability research.
- **Tokengrams Project Inquiry**: Users inquired about obtaining the `document.bin` file for datasets like the Pile and Dolma for the Tokengrams project.
   - A team member confirmed the upcoming release of suffix arrays for shards of the Pile dataset, with other datasets being considered in the future.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/NeelNanda5/status/1813921161052635209">Tweet from Neel Nanda @ ICML (@NeelNanda5)</a>: Are you excited about @ch402-style mechanistic interpretability research? I&#39;m looking to mentor scholars via MATS - apply by Aug 30!  I&#39;m impressed by the work from past scholars, and love men...</li><li><a href="https://tinyurl.com/neel-mats-app">Neel Nanda / Arthur Conmy MATS 7.0 Stream -  Admissions Procedure + FAQ</a>: Neel Nanda / Arthur Conmy MATS Stream - Admission Procedure + FAQ How to Apply Fill out the general MATS application form (&lt;10 mins). Deadline Friday Aug 30 11:59pm PT. Note that this is a special ...</li><li><a href="https://www.alignmentforum.org/posts/KfkpgXdgRheSRWDy8/a-list-of-45-mech-interp-project-ideas-from-apollo-research),">A List of 45+ Mech Interp Project Ideas from Apollo Research’s Interpretability Team — AI Alignment Forum</a>: Why we made this list:  •  * The interpretability team at Apollo Research wrapped up a few projects recently[1]. In order to decide what we’d work on…</li><li><a href="https://docs.google.com/document/d/1XRb-EDDw-h6c-L6pKltI9A8zJmufE71I2_4wx8d5rMg/edit?usp=sharing)**!">[Community Draft] Extending Tinystories</a>: Improved Datasets of Short Simple Stories for Training Interpretable Language Models What problem are we solving? TinyStories is a well received dataset of ca. 2M model-generated short stories in simp...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1263933785179820093)** (56 messages🔥🔥): 

> - `Zeno Upload Feature`
> - `Commit Branch Queries`
> - `Logging Issues`
> - `Multinode Inference Support`
> - `PSA: ICML Conference` 


- **Zeno Upload Feature Confusion**: A member faced issues with the Zeno upload feature in `visualize_zeno.py`, particularly with the `get_latest_filename` function.
   - It was recommended by another member to use the `main` branch instead of `big-refactor`, and to use `pip install -e .` for correct installations.
- **Commit Branch Version Clarification**: There was confusion regarding the version number in `pyproject.toml` showing 0.4.3 while the README announces 0.4.4.
   - Members agreed that this discrepancy needed a check, with one suggesting the need to update the README and add a new FAQ document.
- **Logger Information Not Printing**: A member reported that `eval_logger.info` statements in `lm-eval-harness` were not printing while `print` statements worked fine.
   - It was confirmed that the install was editable, and suggestions were made to check logger configurations.
- **Multinode Inference for Large Models**: Questions were raised about the ability of the eval harness to support inference from sharding across 2 nodes for large models.
   - It was discussed that a PR from the Open LLM Leaderboard team could enable this, and using `vllm` with its inter-node PP might be an effective solution.
- **PSA: ICML Conference Attendance**: A public service announcement was made that some team members would be attending ICML, potentially delaying PR reviews.
   - Members were encouraged to reach out in the channel or meet at ICML for faster responses and discussions.



**Link mentioned**: <a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/2008">Refactor API models by baberabb · Pull Request #2008 · EleutherAI/lm-evaluation-harness</a>: This PR introduces a new superclass for API request models, providing:  Modularity for downstream classes Overloadable methods for request transformation, API requests and response parsing Tokeniza...

  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1264480868668280884)** (4 messages): 

> - `Latent downscaling`
> - `Image classification performance`
> - `Generated latents` 


- **Latent Downscaling Challenges**: A user discussed the challenge of applying operations directly to latents and suggested downscaling images before encoding them.
- **Generated Latents with ImageNet**: A user mentioned creating a generated version of **ImageNet** with a native resolution of **128x128x4** for the generator and **64x64x4** for the classifier, achieving a **20%** performance boost through image resizing versus naive latent resizing.
   - They are exploring cost-effective methods for latents to achieve similar classification performance benefits.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1264299956324405330)** (19 messages🔥): 

> - `Nemotron-340B specifics`
> - `Nathan's bounty for Nemotron-340B conversion`
> - `vLLM multinode inference`
> - `Multi-node performance for Nemotron-340B`
> - `Evaluation harness discussion` 


- **Nathan offers bounty for Nemotron-340B HF conversion**: [Nathan](https://x.com/natolambert/status/1814735390877884823) offers a bounty starting at $75 for converting **Nemotron-340B-Instruct to HuggingFace** with **FP8 quantization** and multi-node HF implementation.
   - The bounty has now grown to **over $2,000** with added donors from the synthetic data community.
- **Debating Nemotron-340B's architecture uniqueness**: Members discuss that **Nemotron-340B** is essentially standard with few unique components like **sqrelu** and **custom rope pct**.
   - Hailey notes, *if we had a setup we could easily load the model on, I could add it to vllm without too much difficulty*.
- **vLLM multinode inference feasibility**: Hailey discusses the feasibility of **vLLM multinode inference** but isn't sure about the performance stating, *I actually don’t know how good if at all vllm multinode is. I think bad?*.
   - Stella notes that a good and easy to run multinode inference setup doesn't exist currently and isn't particularly reasonable for most available hardware.
- **Multi-node performance and testing challenges**: The group acknowledges that supporting Nemotron's architecture and running it efficiently multi-node are conflated issues, indicating a lack of multiple nodes for testing.
   - Tastybucketofrice believes the setup is doable, and comments, *I'll do it tn* after seeing the bounty increase.
- **Nerdsniping and evaluation harness**: Catboy_slim_ shows interest in *scutwork* and discusses integrating a *uncheatable eval* into the evaluation harness.
   - Baber_ humorously notes, *It ceases to be uncheatable once you add it to the harness*.



**Link mentioned**: <a href="https://x.com/natolambert/status/1814735390877884823?s=46">Tweet from Nathan Lambert (@natolambert)</a>: I&#39;m offering a paid bounty to successfully convert nvidia/Nemotron-4-340B-Instruct to HuggingFace / related libraries.  Starting reward $75  We really need this to unlock synthetic permissive data...

  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1264294243862581278)** (61 messages🔥🔥): 

> - `Nemotron-4-340B conversion to HuggingFace`
> - `Llama-3 and 3.1 leaks`
> - `Meta AI's potential premium offerings`
> - `Distillation techniques for large models`
> - `SOC2 compliance for HuggingFace` 


- **Nemotron-4-340B conversion to HuggingFace**: [Nathan Lambert](https://x.com/natolambert/status/1814735390877884823) is offering a paid bounty to convert **nvidia/Nemotron-4-340B-Instruct** to HuggingFace, with initial donations totalling $75.
   - The goal is to unlock synthetic permissive data and enable distillation projects, requiring both FP8 quantization and multi-node implementation.
- **Llama-3 and 3.1 leaks spark excitement**: Rumors and leaks about **Llama-3 405b** and **Llama 3.1** models, including benchmarks and potential features, were widely discussed, with links pointing to specific [Azure's GitHub benchmarks](https://github.com/Azure/azureml-assets/pull/3180/files) and community [Reddit threads](https://www.reddit.com/r/LocalLLaMA/comments/1e9hg7g/azure_llama_31_benchmarks/).
   - Leaked benchmarks show **Llama 3.1** outperforming GPT-4 in several areas, excluding HumanEval, sparking conversation on its potential superiority.
- **Meta AI's potential premium offerings**: There is speculation that **Llama 405B** might be part of a Premium offering from Meta AI, as suggested by snippets of code and a [tweet by Testing Catalog](https://x.com/testingcatalog/status/1815439546722451493?s=46).
   - A possible Meta AI API platform, AI Studio, was also hinted at, creating buzz around the upcoming July 23 announcements.
- **SOC2 compliance concerns for HuggingFace**: Discussion highlighted that HuggingFace's **SOC2 compliance** might be causing some issues, though no specific details were provided.
   - Nathan Lambert expressed surprise that HuggingFace has SOC2 compliance, suggesting it could be contributing to delays or complexities.
- **Discussion on distillation for Llama 3.1**: Nathan Lambert considered writing an article on distillation techniques, inspired by the potential impact on **Llama 3.1**.
   - There was speculation that a significant portion of Llama 3.1's performance gains could be attributed to distillation methods, similar to **Gemma 2**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.bloomberg.com/news/articles/2024-07-22/ai-startup-cohere-valued-at-5-5-billion-in-new-funding-round">Bloomberg - Are you a robot?</a>: no description found</li><li><a href="https://x.com/TheXeophon/status/1815317803164971469">Tweet from Xeophon (@TheXeophon)</a>: o7</li><li><a href="https://x.com/testingcatalog/status/1815439546722451493?s=46">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: Besides that, it seems that LLama 405B may become a part of the Premium offering and in this case, Meta AI Premium could be announced on Jul 23 as well (Spotted in the code).  Also, a mention of AI St...</li><li><a href="https://x.com/danielhanchen/status/1814752981725946227?s=46">Tweet from Daniel Han (@danielhanchen)</a>: @AlpinDale Unconfirmed rumors but fp8 was converted from fp16, so hopefully fp16 will come  & Llama-3 405b is just the LLM. Text + image delayed until fall  & Llama 3.1 refresh for 8B and 70B (base + ...</li><li><a href="https://x.com/gblazex/status/1815441807385252024?s=46">Tweet from Blaze (Balázs Galambosi) (@gblazex)</a>: here is 405B vs GPT-4 Omni =&gt; Llama better almost across the board  Except HumanEval (not the best coding benchmark).  Most notable is MMLU STEM, but note that 4-Turbo would be closer in this one.</li><li><a href="https://x.com/natolambert/status/1814735390877884823">Tweet from Nathan Lambert (@natolambert)</a>: I&#39;m offering a paid bounty to successfully convert nvidia/Nemotron-4-340B-Instruct to HuggingFace / related libraries.  Starting reward $75  We really need this to unlock synthetic permissive data...</li><li><a href="https://x.com/gblazex/status/1815426702425928118?s=46">Tweet from Blaze (Balázs Galambosi) (@gblazex)</a>: LLama 3.1 benchmark results leaked on Azure&#39;s Github account, including 405B, 70B, 8B  source: https://github.com/Azure/azureml-assets/pull/3180/files  spotted by: https://www.reddit.com/r/LocalLL...</li><li><a href="https://x.com/kalomaze/status/1815305220118769952?s=46">Tweet from kalomaze (@kalomaze)</a>: oh that&#39;s llama3.1-405b leaked at 3am the day before on 4chan</li><li><a href="https://www.joelonsoftware.com/2002/06/12/strategy-letter-v/">Strategy Letter V</a>: When I was in college I took two intro economics courses: macroeconomics and microeconomics. Macro was full of theories like &#8220;low unemployment causes inflation&#8221; that never quite stood u…</li><li><a href="https://x.com/morqon/status/1815118198985101444?s=46">Tweet from morgan — (@morqon)</a>: big week for zuck</li><li><a href="https://x.com/JoeBiden/status/1815080881981190320">Tweet from Joe Biden (@JoeBiden)</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e9hg7g/comment/leedpl3">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1264381727124488324)** (4 messages): 

> - `WizardMath paper`
> - `Instruction Reward Model (IRM)`
> - `PRM by Uesato et al 2022 paper`
> - `Step-by-step reward labeling`
> - `UltraChat vs Zephyr paper` 


- **Instruction Reward Model in WizardMath sparks curiosity**: [WizardMath paper](https://arxiv.org/abs/2308.09583) introduces the **Instruction Reward Model (IRM)** which rates the quality of instructions and uses this rating to influence PPO rewards.
   - A user questioned whether conditioning on instruction quality is valuable and wondered if similar ideas are used elsewhere.
- **Binary vs Categorical vs Scalar Rewards for Reasoning Steps**: A user compared the PRM system from Uesato et al 2022, which uses binary rewards, with the Let's Verify Step by Step paper that uses categorical labeling (positive, negative, neutral).
   - They questioned why researchers might choose different reward systems for reasoning steps in model training.
- **Debate over UltraChat's Data Quality**: A user noted that the Zephyr paper significantly filtered UltraChat data from **1.5M** to **200k** and asked for opinions on UltraChat's generation process.
   - They compared the top-down approach of **UltraChat** with the diversely-sourced seed documents approach (e.g., WRAP/Rephrasing the Web), questioning which method is more effective.
- **Surprising Effectiveness of UltraChat**: A user expressed surprise at the effectiveness of **UltraChat** despite significant filtering and generation process scrutiny.
   - Another user acknowledged the comment but didn't elaborate further.


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1264299609992335500)** (3 messages): 

> - `ICML 2024 Spotlight`
> - `Lit Review Concerns`
> - `Harvey Legal AI Criticism` 


- **ICML 2024 celebrates Faithfulness Measurable Models**: **Andreas Madsen** announced their ICML 2024 spotlight on a [new approach for interpretability](https://x.com/peterbhase/status/1814692347407429706?s=46): Faithfulness Measurable Models, boasting 2x-5x better explanations and accurate faithfulness metrics at no cost.
   - A user pointed out that this closely resembles their 2021 NeurIPS paper, emphasizing the need for improved **literature reviews** in submissions and reviews.
- **Legal AI company Harvey slammed as 'smoke and mirrors'**: A comment on [Harvey, the legal AI company](https://x.com/emilyinvc/status/1814724166593225050?s=46) predicted its failure, dismissing it as 'smoke and mirrors.'
   - *Emilyinvc* bluntly predicted that Harvey would end up as 'roadkill on the side of the highway.'


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/peterbhase/status/1814692347407429706?s=46">Tweet from Peter Hase (@peterbhase)</a>: The GIF for this ICML 2024 spotlight almost exactly describes our 2021 paper at neurips (as well as Vafa et al EMNLP 2021).  Congrats to Andreas and others for the work (I think it&#39;s a good idea) ...</li><li><a href="https://x.com/emilyinvc/status/1814724166593225050?s=46">Tweet from emily is in sf (@emilyinvc)</a>: calling it now:   harvey (the legal AI co) will end up being roadkill on the side of the highway. complete smoke and mirrors company.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1263943111210700934)** (52 messages🔥): 

> - `Interview Invitations`
> - `MosaicML Sword Tradition`
> - `Claude AI Text Restrictions`
> - `Reward Model Innovations`
> - `Stripe Settings Issue` 


- **Lambert reflects on interview strategies**: Lambert considered reaching out to various high-profile individuals like **Karpathy**, **Sebastian Raschka**, and **Gary Marcus** for interviews, noting a reluctance to contact **Andreessen** due to political concerns.
   - He also mentioned upcoming plans involving **Ross Tayler** and **Andrew Trask**, and expressed excitement about potential collaborations with **HuggingFace**.
- **MosaicML's sword gifting phased out**: Discussion revealed that the tradition of gifting swords at **MosaicML** was phased out with the advent of an HR department, shifting to more professional norms.
   - A humorous note mentioned members of **Databricks**' legal team allegedly receiving swords, and Lambert toyed with the idea of introducing **Interconnects swords**.
- **Claude AI faces text restrictions**: Members discussed **Claude AI** refusing certain texts, particularly sacred texts like the **'I Have a Dream'** speech.
   - A workaround involved pre-filling responses to overcome using restricted texts in the API.
- **Innovations in reward models discussed**: Members shared links and discussed the [Absolute-Rating Multi-Objective Reward Model](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1) which boasts of Mixture-of-Experts aggregation.
   - The discourse included a mention of **RewardBench** challenges and the relative youth of Reward Models (RMs).
- **Stripe settings phone number issue**: Lambert was dealing with an issue where his phone number appeared on receipts due to Stripe settings.
   - He joked about switching to a **Google Voice number** and discussed alternative options like virtual mailboxes.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1">RLHFlow/ArmoRM-Llama3-8B-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://x.com/soldni/status/1695087021520457939">Tweet from Luca Soldaini 🎀 (@soldni)</a>: bless @DippedRusk who got me in my natural habitat (my desk at the office) with my @MosaicML sword</li><li><a href="https://github.com/project-numina/aimo-progress-prize">GitHub - project-numina/aimo-progress-prize</a>: Contribute to project-numina/aimo-progress-prize development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[nlp](https://discord.com/channels/1179127597926469703/1208183200099344445/1265052470636187790)** (2 messages): 

> - `Blog post on distillation`
> - `Lilian Wang`
> - `Surprise at lack of resources` 


- **Seeking blog post on distillation**: A member inquired if anyone has a favored blog post on **distillation**.
- **Surprise at lack of Lilian Wang's work**: Another member expressed surprise at the absence of a comprehensive **Lilian Wang** blog post on the topic, suggesting there isn't a 20k word post available.


  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1264116021552218192)** (2 messages): 

> - `Yitay blog post on model architectures`
> - `Encoder vs. Encoder-Decoder models`
> - `Evolution of LLMs`
> - `@srush_nlp tweet` 


- **Yitay introduces series on model architectures**: [Yitay's blog post](https://www.yitay.net/blog/model-architecture-blogpost-encoders-prefixlm-denoising) discusses the shift from encoder models like **BERT** to new trends in LLMs.
   - He aims to update those interested in language and NLP on the evolution of model architectures, referencing a deleted [tweet by @srush_nlp](https://x.com/srush_nlp/status/1779938508578165198).
- **Where have all the encoder models gone?**: **Yitay** tackles the question of why scaling BERT and similar encoder models fell out of favor despite their success.
   - He plans to explore this topic in-depth in a series of blog posts, starting with the linked primer.



**Link mentioned**: <a href="https://www.yitay.net/blog/model-architecture-blogpost-encoders-prefixlm-denoising">What happened to BERT &amp; T5? On Transformer Encoders, PrefixLM and Denoising Objectives &mdash; Yi Tay</a>: A Blogpost series about Model Architectures Part 1: What happened to BERT and T5? Thoughts on Transformer Encoders, PrefixLM and Denoising objectives

  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1263971600932540416)** (74 messages🔥🔥): 

> - `Langfuse vs Langsmith`
> - `GPT-4o mini and AI-generated content`
> - `Rumors about Harvey AI`
> - `Elon Musk's Memphis Supercluster`
> - `LLaMA 3.1 leaks and evaluations` 


- **Langfuse Outshines Langsmith**: Anecdotal feedback from users suggests that [Langfuse](https://github.com/langfuse/langfuse) is performing better than Langsmith, with positive experiences shared about its ease of self-hosting and integration.
   - *Clemo_._*, the founder, encouraged more community interaction, emphasizing their commitment to maintaining a great OSS solution.
- **GPT-4o Mini Enables AI-generated Content**: OpenAI's new [GPT-4o mini model](https://batchmon.com/blog/ai-cheaper-than-ads/) costs $0.15 per 1M input tokens, making it possible to create dynamic AI-generated content supported entirely by ads.
   - Discussion includes the potential impact on web content, hypothesizing a shift towards more AI-generated outputs.
- **Harvey AI Rumors and Predictions**: Rumors and skepticism about [Harvey AI's](https://x.com/emilyinvc/status/1814741780010844289?s=46) viability emerged, with some calling it a 'smoke and mirrors company'.
   - Debates ensued about the challenges facing vertical AI startups, including dependency on big AI labs and the industry's current cycle.
- **Elon Musk's Memphis Supercluster**: Elon Musk announced the launch of the Memphis Supercluster, claiming it to be the world's most powerful AI training cluster with 100k liquid-cooled H100s on a single RDMA fabric.
   - However, [fact-checks](https://x.com/dylan522p/status/1815494840152662170?s=46) reveal discrepancies in power usage and GPU availability, suggesting that the facility is not yet fully operational.
- **LLaMA 3.1 Leaks Spark Excitement**: Leaked evaluations for [LLaMA 3.1](https://x.com/mattshumer_/status/1815444612414087294?s=46) suggest that its 8B, 70B, and 405B models might outperform current state-of-the-art models, even before instruct tuning.
   - These leaks have led to widespread anticipation and speculation about the future capabilities of open-source AI models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/emilyinvc/status/1814724166593225050?s=46">Tweet from emily is in sf (@emilyinvc)</a>: calling it now:   harvey (the legal AI co) will end up being roadkill on the side of the highway. complete smoke and mirrors company.</li><li><a href="https://share.snipd.com/snip/686d53b1-5696-427f-bb8d">Snipd — Highlight &amp; share the best moments in podcasts</a>: no description found</li><li><a href="https://x.com/brunokoba_/status/1814893302698926326?s=46">Tweet from Bruno Koba (@brunokoba_)</a>: I&#39;ve been investigating vertical AI startups profoundly for the past few weeks. I think we&#39;re in a very strange part of the cycle in AI startup funding/development. Some thoughts (hot takes?) ...</li><li><a href="https://www.bloomberg.com/news/articles/2024-07-22/ai-startup-cohere-valued-at-5-5-billion-in-new-fu">Bloomberg - Are you a robot?</a>: no description found</li><li><a href="https://x.com/dylan522p/status/1815494840152662170?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Dylan Patel @ ICML (@dylan522p)</a>: Elon is lying There is 7MW currently being drawn from the grid ~4k GPU August 1st, 50MW will be available if http://X.ai finally signs a deal with the Tennessee Valley Authority The 150MW substation i...</li><li><a href="https://www.bloomberg.com/news/articles/2024-07-22/ai-startup-cohere-valued-at-5-5-billion-in-new-funding-round">Bloomberg - Are you a robot?</a>: no description found</li><li><a href="https://x.com/nisten/status/1814558806770172181?s=46">Tweet from nisten (@nisten)</a>: Since ya&#39;ll reaaally want to know the actual full prompt... sigh..here you go knock yourselves out.</li><li><a href="https://x.com/maccaw/status/1815435539669283204?s=46">Tweet from Alex MacCaw (@maccaw)</a>: If this is real, the world is about to change.</li><li><a href="https://x.com/sarahookr/status/1815360812787380701?s=46">Tweet from Sara Hooker (@sarahookr)</a>: Is bigger always better? 🐘 The idea that scaling more than any other ingredient has driven progress has become formalized as the “bitter lesson”  Is Sutton right?  📜https://arxiv.org/abs/2407.05694v...</li><li><a href="https://x.com/emilyinvc/status/1814741780010844289?s=46">Tweet from emily is in sf (@emilyinvc)</a>: in the DMs from a biglaw partner that &#34;uses&#34; harvey  Quoting emily is in sf (@emilyinvc)   calling it now:   harvey (the legal AI co) will end up being roadkill on the side of the highway. com...</li><li><a href="https://x.com/latentspacepod/status/1815411709085143197">Tweet from Latent.Space (@latentspacepod)</a>: 🆕 The Winds of AI Winter  The vibes have shifted.  On @leopoldasch vs Sequoia, Goldman Sachs, @benedictevans, @cpaik&#39;s End of Software and why the AI Engineer rises above it all.  The future is h...</li><li><a href="https://share.snipd.com/snip/e0e81b28-78d1-430b-831b-1ff1b76b58a8">Efficiency gains from AI will be competed away | 1min snip from No Priors: Artificial Intelligence | Technology | Startups</a>: 1min snip from How AI is opening up new markets and impacting the startup status quo with Sarah Guo and Elad Gil | No Priors: Artificial Intelligence | Technolo…</li><li><a href="https://x.com/mattshumer_/status/1815444612414087294?s=46">Tweet from Matt Shumer (@mattshumer_)</a>: Leaked (possibly real?) evals for Llama 3.1.  Base models, not instruct.  Open-source is about to be SOTA — even the 70B is &gt; gpt-4o, and this is before instruct tuning, which should make it even b...</li><li><a href="https://share.snipd.com/snip/686d53b1-5696-427f-bb8d-7e22ef157558">Proprietary Models for Frontier AI: The Cost Perspective | 2min snip from Grit</a>: 2min snip from #200 CEO and Co-Founder Together AI, Vipul Ved Prakash w/ Bucky Moore: Super Cycle | Grit</li><li><a href="https://x.com/tszzl/status/1814787334166224962?s=46">Tweet from roon (@tszzl)</a>: @Teknium1 @dionysianyawp @sama i can definitely say and stake my reputation on this not being true. ai progress is currently blindingly fast</li><li><a href="https://share.snipd.com/snip/53b18e90-8408-42ea-8824-e2ea17faf693">The Expensive Nature of Generative AI Core Operations | 1min snip from Grit</a>: 1min snip from #200 CEO and Co-Founder Together AI, Vipul Ved Prakash w/ Bucky Moore: Super Cycle | Grit</li><li><a href="https://x.com/cohere/status/1815377543182303410?s=46">Tweet from cohere (@cohere)</a>: We’re excited to announce our Series D financing to accelerate growth, expand our team, & develop our next class of frontier, enterprise-grade, data privacy-focused AI technology.  We’re bringing high...</li><li><a href="https://x.com/alpindale/status/1814814551449244058?s=46">Tweet from Alpin (@AlpinDale)</a>: Have confirmed that there&#39;s 8B, 70B, and 405B. First two are distilled from 405B. 128k (131k in base-10) context. 405b can&#39;t draw a unicorn. Instruct tune might be safety aligned. The architec...</li><li><a href="https://www.economist.com/business/2024/06/13/a-price-war-breaks-out-among-chinas-ai-model-builders">A price war breaks out among China’s AI-model builders</a>: It may stymie innovation</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e98zrb/llama_31_405b_base_model_available_for_download/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://x.com/elonmusk/status/1815325410667749760?s=46">Tweet from Elon Musk (@elonmusk)</a>: Nice work by @xAI team, @X team, @Nvidia & supporting companies getting Memphis Supercluster training started at ~4:20am local time.  With 100k liquid-cooled H100s on a single RDMA fabric, it’s the mo...</li><li><a href="https://batchmon.com/blog/ai-cheaper-than-ads/">AI paid for by Ads – the gpt-4o mini inflection point</a>: AI is cheaper than ever with OpenAI&#x27;s latest announcement of gpt4-o mini. AI is so cheap now, that it&#x27;s cheaper than the average ad impression.</li><li><a href="https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/">TLDR: 1 year of building with LLMs &#8211; D-Squared</a>: no description found</li><li><a href="https://news.ycombinator.com/item?id=41010188">AI paid for by Ads – the GPT-4o mini inflection point | Hacker News</a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 messages): 

swyxio: big monthly recap is up: https://x.com/latentspacepod/status/1815411709085143197
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1263948175635513405)** (29 messages🔥): 

> - `Audio Issues`
> - `Layout Detection`
> - `Texify vs Mathpix`
> - `Presentation Feedback`
> - `Model Training Dataset` 


- **Audio Issues Plague Meeting**: Several members reported audio issues, with **nuvic_** being heard by none while **vikas.p** was clear to other members.
   - The problem was not isolated as **slono** observed similar issues occurring for different people on Discord.
- **Layout Detection Demystified**: Discussion focused on the mechanics of layout detection, speculating whether it's based on classical object detection with extensive training data.
   - Members appreciated the explanation of task decomposition in the context of layout detection and reading order models.
- **Texify Versus Mathpix**: A question was raised about how **Texify** compares to **Mathpix** in terms of performance and usage.
   - The query did not receive a direct comparison but generated interest in the distinctive methodologies used by both tools.
- **Presentation Receives High Praise**: *This whole presentation was 🤯 👏*, said a member, indicating strong approval of the session.
   - The session concluded with overwhelming positive feedback and thanks from various attendees.
- **Query on Training Datasets**: Members were curious about the creation of training datasets, asking whether the reading order model's labels were manual or heuristic.
   - The explanation provided was well-received, with a positive acknowledgment from the members.



**Link mentioned**: <a href="https://github.com/VikParuchuri">VikParuchuri - Overview</a>: VikParuchuri has 90 repositories available. Follow their code on GitHub.

  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1263973960295645214)** (64 messages🔥🔥): 

> - `Training Qwen2-7b`
> - `Triplex for Knowledge Graphs`
> - `Mistral 12b Issues`
> - `LLaMA 3 Inconsistencies`
> - `LLaMA 3.1 Benchmarks` 


- **Qwen2-7b Training Setup Issues**: A member inquired about the configuration for training **Qwen2-7b**, seeking guidance on the appropriate settings to use.
   - *Specifically, the user asked if anyone had the config for running `dolphin-2.9.2-qwen2-7b`*.
- **Triplex Cuts Knowledge Graph Costs by 98%**: SciPhi's new [Triplex model](https://huggingface.co/SciPhi/Triplex) reduces knowledge graph creation costs by 98%, outperforming **GPT-4** at a fraction of the cost.
   - This model, which extracts triplets from unstructured data, can operate locally, making knowledge graphs accessible and less expensive.
- **Mistral 12b Faces Tokenizer Problems**: Multiple members reported significant issues with **Mistral 12b**, particularly with its tokenizer outputting text without spaces.
   - Despite promising loss metrics, the **outputs were deemed 'garbage'**, indicating unresolved issues potentially tied to special tokens.
- **LLaMA 3.1 EOS Token Issue**: A discrepancy in the **EOS token** configuration for LLaMA 3 repositories on Hugging Face was identified, causing significant problems.
   - The EOS token was set incorrectly, and a member provided updated token configurations to resolve this issue.
- **LLaMA 3.1 Benchmarks Impress Community**: Members were highly impressed by the benchmarks for **LLaMA 3.1**, noting particularly strong performances by the 8B and 70B models.
   - The 405B model also performed well, but the 70B model was remarked as being **very close behind the leading models**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct/discussions/8">NousResearch/Meta-Llama-3-8B-Instruct · Match official tokenizer_config.json. Changes were made after this repo was created.</a>: no description found</li><li><a href="https://huggingface.co/SciPhi/Triplex">SciPhi/Triplex · Hugging Face</a>: no description found</li><li><a href="https://r2r-docs.sciphi.ai/cookbooks/knowledge-graph">Knowledge Graphs - The best open source AI powered answer engine.</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1264054648663183470)** (22 messages🔥): 

> - `Knowledge Distillation Support`
> - `DPO Improvements`
> - `DeepSpeed Zero-3 Compatibility` 


- **Members discuss adding Knowledge Distillation support**: Members brainstormed adding support for **knowledge distillation** with pretokenized datasets, specifically focusing on providing logits for each input id instead of labels.
   - *They considered the feasibility within the hf trainer constraints and noted TRL might not provide a compatible trainer out of the box.*
- **Conversations around merging DPO improvements**: Members initiated merging **DPO improvements** and discussed the necessity of achieving parity with updated chat_template functionalities.
   - [The DPO enhancements](https://github.com/axolotl-ai-cloud/axolotl/pull/1725) may require a deep dive into TRL's DPO Trainer for finer controls like masking out parts of the query during loss computation.
- **DeepSpeed Zero-3 compatibility issues resolved by members**: A user encountered a **ValueError** with DeepSpeed Zero-3 stating incompatibility with `low_cpu_mem_usage=True` and custom `device_map`.
   - The issue was resolved by deleting the **accelerate config**, which allowed the setup to proceed without errors.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/axolo">axolo - Overview</a>: Adversity leads to prosperity. axolo has 32 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/axolotl-ai-cloud/axolotl/pull/1756">Add flexible configuration options for `chat_template` dataset training by Tostino · Pull Request #1756 · axolotl-ai-cloud/axolotl</a>: Introduce roles_to_train parameter to set training labels by role Add train_on_eos option to configure training on end-of-sequence tokens Implement per-message training configuration in dataset All...
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1264005864730267728)** (5 messages): 

> - `Axolotl training error`
> - `GPU memory issues`
> - `Batch size adjustment`
> - `Mixed precision training`
> - `NCCL timeouts` 


- **Axolotl training hits GPU memory roadblocks**: Member asked about the cause of an error during Axolotl training, and **Phorm** responded that it is likely due to running out of GPU memory, which is common with large models or batch sizes.
   - *Phorm* suggested troubleshooting steps such as reducing batch size, adjusting gradient accumulation steps, and using mixed precision training to mitigate the issue.
- **Tackle NCCL timeouts for smoother training**: For NCCL timeouts during distributed training, **Phorm** directed users to refer to the [NCCL guide](https://github.com/openaccess-ai-collective/axolotl/tree/main/docs/nccl.qmd) for detailed troubleshooting tips.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/openaccess-ai-collective/axolotl/tree/main/README.md#L562L614)">axolotl/README.md at main · axolotl-ai-cloud/axolotl</a>: Go ahead and axolotl questions. Contribute to axolotl-ai-cloud/axolotl development by creating an account on GitHub.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=15dfc26f-b460-49e5-ae58-0ffd7039cc47)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.
</li>
</ul>

</div>
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1264269426392432732)** (57 messages🔥🔥): 

> - `LangChain.js Token issues`
> - `Video and blog post on LLM lessons`
> - `Beginners guide on LangChain`
> - `Vector store filtering`
> - `Discussion on deploying RAG app to production` 


- **LangChain.js Token issues persist**: A user inquired if the LangChain.js Token for 4omini is still broken.
   - No specific response or resolution was mentioned in the channel.
- **New video and blog post on LLM lessons**: A member shared a [video and blog post](https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/) highlighting lessons from a year of building with LLMs, recommending viewers to watch the video for better understanding.
   - The post summarized a three-part series by six practitioners focusing on tactical, operational, and strategic insights.
- **Beginner's guide on LangChain**: A user shared a [Medium article](https://medium.com/@ambaliaharshit25/ea17820a5c01) offering a beginner-friendly introduction to LangChain and its components.
   - The article aims to explain why these components are used and their importance, using relatable examples like JARVIS from Iron Man.
- **Filter application in vector stores**: Multiple users discussed applying filters when using MongoDB Atlas VectorStore within LangChain, with detailed code snippets provided.
   - Methods for custom retrievers and integrating these with the EnsembleRetriever were also explained.
- **Deploying RAG app to production**: A member shared a [tutorial](https://www.mongodb.com/docs/atlas/atlas-vector-search/ai-integrations/langchain/) on using MongoDB Atlas with LangChain to build a RAG implementation.
   - The tutorial covers setting up the environment, storing data, creating search indices, and running vector search queries.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://medium.com/@ambaliaharshit25/ea17820a5c01">LangChain components. Why and How?</a>: Study langChain with me as complete beginner, This blog is not just about what, but WHY and HOW do we utilize components of langChain.</li><li><a href="https://js.langchain.com/v0.2/docs/integrations/vectorstores/mongodb_atlas/#metadata-filtering>).">MongoDB Atlas | 🦜️🔗 Langchain</a>: Only available on Node.js.</li><li><a href="https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/">TLDR: 1 year of building with LLMs &#8211; D-Squared</a>: no description found</li><li><a href="https://python.langchain.com/v0.2/docs/integrations/vectorstores/google_vertex_ai_vector_search/#use-vector-store-as-retriever>)">Google Vertex AI Vector Search | 🦜️🔗 LangChain</a>: This notebook shows how to use functionality related to the Google Cloud Vertex AI Vector Search vector database.</li><li><a href="https://github.com/langchain-ai/langchain/issues/14227>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://www.mongodb.com/docs/atlas/atlas-vector-search/ai-integrations/langchain/">arrow-right</a>: no description found</li><li><a href="https://github.com/langchain-ai/langchain/issues/17464>),">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/19885>),">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/2095>),">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/14227>).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1264009384497778749)** (6 messages): 

> - `Triplex model`
> - `Vector visualization for embeddings`
> - `Semantic search with LangChain`
> - `AI function builder for TypeScript`
> - `LangChain tutorial` 


- **Triplex slashes knowledge graph creation costs**: The newly open-sourced [Triplex](https://huggingface.co/SciPhi/Triplex) by SciPhi.AI reduces knowledge graph creation costs by 98%, outperforming GPT-4 at 1/60th the cost.
   - Triplex, a finetuned version of Phi3-3.8B, extracts triplets from unstructured data, enhancing RAG methods like Microsoft's Graph RAG.
- **Visualize vectors in NLP using graphs**: A new [GitHub project](https://github.com/rajatasusual/realtime-vector-embeddings) was created to help visualize vectors on a graph, making it easier to understand embeddings in NLP tasks.
   - *Graphs are easier to understand than equations* and can aid in text classification, clustering, and recommendation systems.
- **Semantic Search tutorial with LangChain**: [A new blog post on Substack](https://sonamcoffeenlp.substack.com/p/semantic-search-to-glean-valuable-deb) dives into implementing semantic search using LangChain, Cohere LLM, and ApertureDB.
   - The author described implementing a chat module using Cohere's Command R+ and encouraged feedback from readers.
- **AI-powered function builder for TypeScript**: A new project called [AI Fun](https://github.com/mishushakov/ai-fun) was developed at a hackathon to build LLM-powered functions for TypeScript.
   - The project leverages AI to automate and simplify TypeScript function building processes.
- **Create a Scheduler Agent with Composio and LangChain**: A [detailed guide](https://git.new/scheduler) was shared for creating a Scheduler Agent that uses Composio, LangChain, and ChatGPT to schedule events based on emails.
   - The guide aims to empower users to leverage these tools for more efficient task management.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/SciPhi/Triplex">SciPhi/Triplex · Hugging Face</a>: no description found</li><li><a href="https://git.new/scheduler">composio/python/examples/scheduler_agent at master · ComposioHQ/composio</a>: Composio equips agents with well-crafted tools empowering them to tackle complex tasks - composio/python/examples/scheduler_agent at master · ComposioHQ/composio</li><li><a href="https://github.com/mishushakov/ai-fun">GitHub - mishushakov/ai-fun: The LLM-powered function builder for TypeScript</a>: The LLM-powered function builder for TypeScript. Contribute to mishushakov/ai-fun development by creating an account on GitHub.</li><li><a href="https://github.com/rajatasusual/realtime-vector-embeddings">GitHub - rajatasusual/realtime-vector-embeddings: This project is created to understand how similar different pieces of text are in a multi-dimensional space. This is a crucial concept in Natural Language Processing (NLP) tasks such as text classification, clustering, and recommendation systems.</a>: This project is created to understand how similar different pieces of text are in a multi-dimensional space. This is a crucial concept in Natural Language Processing (NLP) tasks such as text classi...</li><li><a href="https://medium.com/@ambaliaharshit25/ea17820a5c01">LangChain components. Why and How?</a>: Study langChain with me as complete beginner, This blog is not just about what, but WHY and HOW do we utilize components of langChain.
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1264844424622637178)** (3 messages): 

> - `LangChain article on Medium by Harshit Ambalia`
> - `Guide for deploying a RAG app`
> - `Scheduler Agent guide using Composio, LangChain, and ChatGPT` 


- **LangChain beginner-friendly article published**: A user shared a [Medium article](https://medium.com/@ambaliaharshit25/ea17820a5c01) about LangChain and its components, aimed at beginners interested in understanding its applications.
   - *Imagine having a virtual assistant that can handle complex tasks through simple natural language commands*, the article delves into why these components are important.
- **Guide for Scheduler Agent using Composio**: A user posted a [GitHub guide](https://git.new/scheduler) for creating a Scheduler Agent leveraging Composio, LangChain, and ChatGPT to schedule events based on received emails.
   - The guide provides detailed steps and the user encouraged others to try it out and star the repository.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://git.new/scheduler">composio/python/examples/scheduler_agent at master · ComposioHQ/composio</a>: Composio equips agents with well-crafted tools empowering them to tackle complex tasks - composio/python/examples/scheduler_agent at master · ComposioHQ/composio</li><li><a href="https://medium.com/@ambaliaharshit25/ea17820a5c01">LangChain components. Why and How?</a>: Study langChain with me as complete beginner, This blog is not just about what, but WHY and HOW do we utilize components of langChain.
</li>
</ul>

</div>
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1263980830934700042)** (34 messages🔥): 

> - `sdxl vae latents`
> - `HF hosting advantages`
> - `new BUD-E demo`
> - `local LLM on Linux terminal`
> - `Kolors diffusion model` 


- **Saving storage with sdxl vae latents**: **Ramimmo** discussed the potential benefits of using **sdxl vae latents** to reduce storage costs for large image datasets.
   - *Puffy310* raised concerns about copyright implications, but *Nodja* clarified that latent data compression does not exempt from copyright laws.
- **HF pays storage bills, use it**: **Thejonasbrothers** recommended uploading datasets to **Hugging Face** since they cover **S3 storage costs**, making it a cost-effective solution.
   - *Unknown* pointed out that despite this, HF remains profitable, hinting at efficient financial management.
- **Watch new BUD-E demo on YouTube**: **Spirit_from_germany** shared a [new demo](https://youtu.be/O4IXfa8CROs) of the **BUD-E** voice assistant on YouTube.
   - The demo invites the community to join their Discord and help build the assistant together.
- **Kolors model runs on 3090**: **Spirit_from_germany** inquired about running the **Kolors diffusion model** on an NVIDIA 3090.
   - *Segmentationfault8268* confirmed its compatibility, recommending the **ComfyUI** flow and using **Int8 precision** for optimal performance.
- **Seeking local LLM for Linux terminal**: **Alexiosthesixth** looked for a **local LLM** that runs in a Linux terminal via CPU, due to an AMD card limitation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Kwai-Kolors/Kolors-diffusers">Kwai-Kolors/Kolors-diffusers · Hugging Face</a>: no description found</li><li><a href="https://youtu.be/O4IXfa8CROs">BUD-E - Demo</a>: Join our Discord Community, try BUD-E yourself &amp; help us to build the voice assistant me and BUD-E talk about in the video:https://discord.gg/sTKSB2AwBvhttps...
</li>
</ul>

</div>
  

---


### **LAION ▷ #[announcements](https://discord.com/channels/823813159592001537/826154622644649985/1264865683662045226)** (1 messages): 

> - `Bud-E voice assistant`
> - `Daily Online-Hackathons`
> - `BUD-E Discord Server` 


- **Bud-E presents new demo with open-source goals**: A demo of the **Bud-E voice assistant** was shared, showcasing the vision of a future where everyone has access to highly capable, open-source systems for the cost of electricity.
   - The code base currently optimized for Ubuntu will be restructured for clean separation between client, server, and interchangeable ASR, TTS, LLM components.
- **Join the BUD-E discord server for collaboration**: Volunteers are invited to join the new **BUD-E discord server** to help develop the voice assistant further and contribute new skills akin to Minecraft Mods.
   - Daily Online-Hackathon meetings will occur every day at 9 PM CEST to onboard new volunteers and coordinate project work.
- **Daily Online-Hackathons kick off for BUD-E development**: **Beginning today (Monday 22. July),** daily Online-Hackathon-Meetings will be held at 9 PM CEST to provide an overview, onboard volunteers, and coordinate project efforts.
   - These meetings will take place in a dedicated voice room on Discord: [https://discord.gg/nMexRzbJ3W](https://discord.gg/nMexRzbJ3W).


  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1264387914712875089)** (14 messages🔥): 

> - `Plotting Loss Curves`
> - `Mem0 AI Memory`
> - `Datadog Time Series Modeling`
> - `Research Recruitment` 


- **Switch Back to Epochs for Plotting Loss Curves**: A member initially plotted their loss curves with wall-clock time but found it more meaningful to measure model learning efficiency before regretting and switching back to epochs.
   - The member mentioned the ease of using **WandB** for this purpose but admitted the change was incorrect and a 'foolish' decision.
- **Mem0 Introduces Smart Memory Layer for LLMs**: [Mem0](https://docs.mem0.ai/overview) has released a memory layer for Large Language Models, enabling personalized AI experiences with features like user, session, and AI agent memory, and adaptive personalization.
   - For more information on integration and features, view the [GitHub page](https://github.com/mem0ai/mem0) for Mem0.
- **Datadog Publishes SOTA Results in Time Series Modeling**: A member shared that Datadog has published [state-of-the-art results](https://www.datadoghq.com/blog/datadog-time-series-foundation-model/) on time series modeling and is actively recruiting for research roles.
   - Datadog's foundation models aim to handle time series data effectively by identifying trends, parsing high-frequency data, and managing high-cardinality data.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.mem0.ai/overview">📚 Overview - Mem0.ai</a>: no description found</li><li><a href="https://www.datadoghq.com/blog/datadog-time-series-foundation-model/">Introducing Toto: A state of the art time series foundation model by Datadog</a>: Introducing Toto, or Time Series Optimized Transformer for Observability, Datadog's state-of-the-art foundation model for time series forecasting that we have trained on a trillion data points.</li><li><a href="https://github.com/mem0ai/mem0">GitHub - mem0ai/mem0: The memory layer for Personalized AI</a>: The memory layer for Personalized AI. Contribute to mem0ai/mem0 development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1263946279734480986)** (8 messages🔥): 

> - `PostgresML for Reranking`
> - `LLMs as Judges`
> - `Merlinn: Open-source On-call Copilot`
> - `Multimodal RAG with Ollama and Qdrant`
> - `Deasie RAG Workshop` 


- **Enhance Results with PostgresML Reranking**: Using [PostgresML](https://t.co/HWfitT0CJt) for reranking can significantly boost the relevance of your search results with an extra parameter or two.
   - A guest post on [the blog](https://t.co/HWfitT0CJt) details how this managed index approach works.
- **LLMs as Judges in Production**: A recording featuring [Yixin Hu and Thomas Hulard](https://t.co/i84Cg5pqsy) showcases how to use LLMs as judges to bring an application to production.
   - *This session covered key concepts and practices behind RAG evaluation for development.*
- **Merlinn: AI-powered On-call Copilot**: [Merlinn](https://t.co/rAM5OOxQ34) is an open-source, LLM-powered Slack assistant that listens to and resolves production incidents.
   - *It integrates with observability and incident management tools like Datadog.*
- **Multimodal RAG with Ollama and Qdrant**: An introductory article by [Pavan Mantha](https://t.co/0gcz4GfCh5) explains setting up a multimodal RAG application with Ollama and Qdrant.
   - It details how to ingest audio/video sources through text transcription and index the multimodal data.
- **Improving RAG with Advanced Parsing and Metadata**: A workshop with Deasie cofounders discusses improving RAG with advanced parsing and metadata, [recording available on YouTube](https://t.co/cJPsNaWgoc).
   - Key takeaways include that both parsing and metadata significantly enhance RAG performance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://t.co/HWfitT0CJt">Improving Vector Search - Reranking with PostgresML and LlamaIndex — LlamaIndex, Data Framework for LLM Applications</a>: LlamaIndex is a simple, flexible data framework for connecting custom data sources to large language models (LLMs).</li><li><a href="https://t.co/vTV3t8daqT">TiDB Future App Hackathon 2024</a>: Innovate and Create Amazing AI Applications
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1263948982380531723)** (33 messages🔥): 

> - `llama-parse API issues`
> - `ReActAgent max iterations`
> - `VectorStoreIndex embedding model`
> - `LlamaIndex webinar`
> - `Extracting pictures from PDFs` 


- **llama-parse API displays garbled output**: Users reported issues with the llama-parse API producing garbled outputs, consisting of symbols like :->|>11_NaN<|<-:.
   - One member suggested re-running the job and sharing the JobID for log examination.
- **ReActAgent hits max iterations**: A member faced the `ValueError: Reached max iterations` while using ReActAgent with the retriever.
   - It was suggested to increase the `max_iterations` value, but concerns about the agent being stuck were raised.
- **Specifying custom embedding model for VectorStoreIndex**: A user wanted to use a custom embedding model with VectorStoreIndex instead of defaulting to OpenAI calls.
   - The solution involved setting `Settings.embed_model` to the custom model globally or directly passing the model during initialization.
- **Recent LlamaIndex webinar available on YouTube**: A member asked where to find the latest LlamaIndex webinar recording.
   - The webinar is available on the [LlamaIndex YouTube channel](https://youtu.be/V_-WNJgTvgg?si=_b5qNd3gM6NXRfWy) titled 'Improving RAG with Advanced Parsing + Metadata Extraction'.
- **Extract pictures from PDFs with llama-parse**: A user inquired about extracting pictures and corresponding captions from PDFs using LlamaIndex.
   - It was advised to use the JSON mode in llama-parse to return images and leverage multi-modal LLMs for further processing.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/V_-WNJgTvgg?si=_b5qNd3gM6NXRfWy">LlamaIndex Webinar: Improving RAG with Advanced Parsing + Metadata Extraction</a>: In this video we cohost a workshop with the cofounders of Deasie (Reece, Leonard, Mikko) on improving RAG with advanced parsing and metadata.​The data proces...</li><li><a href="https://github.com/carolinedlu/llamaindex-chat-with-streamlit-docs/blob/main/streamlit_app.py?ref=blog.streamlit.io">llamaindex-chat-with-streamlit-docs/streamlit_app.py at main · carolinedlu/llamaindex-chat-with-streamlit-docs</a>: Build a chatbot powered by LlamaIndex that augments GPT 3.5 with the contents of the Streamlit docs (or your own data). - carolinedlu/llamaindex-chat-with-streamlit-docs</li><li><a href="https://github.com/run-llama/llama_parse">GitHub - run-llama/llama_parse: Parse files for optimal RAG</a>: Parse files for optimal RAG. Contribute to run-llama/llama_parse development by creating an account on GitHub.</li><li><a href="https://docs.unstructured.io/open-source/concepts/document-elements#elements-coordinates)">Document elements and metadata - Unstructured</a>: no description found
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1263982167613116498)** (4 messages): 

> - `ETL for Video and Music Data`
> - `SycoPhancy in LLMs`
> - `Korvus for RAG Pipeline` 


- **Exploring New ETL Methods for Unstructured Data**: A user mentioned Jerry Liu's discussion about a [new type of ETL](https://www.youtube.com/watch?v=imlQ1icxpBU) aimed at making video and music data digestible by LLMs.
   - They believe text and PDFs can be handled well but are curious if the community has achieved similar results for other types of unstructured data.
- **Analyzing SycoPhancy in LLMs**: A user shared a [LinkedIn article](https://www.linkedin.com/posts/subham-kundu-2746b515b_llm-knowledgesharing-evaluation-activity-7220695691021455361-72PE) detaling their analysis on the concept of **SycoPhancy** in LLMs, hoping it will provide insight to the community.
- **Korvus for Simplified RAG Pipeline**: A member is curious whether **Korvus** truly simplifies the RAG pipeline without sacrificing quality.
   - They provided a [GitHub link](https://github.com/postgresml/korvus) to **Korvus**, a search SDK built on PostgreSQL that unifies the entire RAG pipeline in a single database query.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=imlQ1icxpBU">Jerry Liu - What is LlamaIndex, Agents &amp; Advice for AI Engineers</a>: In this episode, we sit down with Jerry Liu, the visionary founder of LlamaIndex, a cutting-edge python framework designed for the development of LLM (Large ...</li><li><a href="https://github.com/postgresml/korvus">GitHub - postgresml/korvus: Korvus is a search SDK that unifies the entire RAG pipeline in a single database query. Built on top of Postgres with bindings for Python, JavaScript, Rust and C.</a>: Korvus is a search SDK that unifies the entire RAG pipeline in a single database query. Built on top of Postgres with bindings for Python, JavaScript, Rust and C. - postgresml/korvus
</li>
</ul>

</div>
  

---



### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1263946422554722479)** (44 messages🔥): 

> - `Issues with GPT4o-mini Model`
> - `DSPy Tracing Release`
> - `TypedPredictors Compatibility`
> - `DSPy Paper Release`
> - `Optimizers Reliability in DSPy` 


- **Issues with GPT4o-mini Model**: A member reported that **GPT4o-mini** was verbose and repeated much of the structure of examples compared to **GPT3.5-turbo**, which impacted data extraction pipelines.
- **DSPy Tracing Release Enhances Workflow**: New **DSPy tracing** feature now available, tracks all modules, predictions, LMs, and retrievers efficiently ([documentation link](https://docs.langwatch.ai/integration/python/guide#capturing-llm-spans)).
- **Challenges with TypedPredictors and Complex Pydantic Classes**: A member noted that only **GPT-4o** and **Sonnet-3.5** handle complex pydantic class generation successfully, while other models fail.
- **DSPy Paper Validates Joint Optimization Approach**: New [paper release](https://x.com/lateinteraction/status/1815423177272824022) shows alternating between prompt optimization and finetuning delivers up to **26% gains** over single methods.
- **Reliability of DSPy Optimizers Discussed**: Members discussed the reliability of DSPy optimizers, noting that **BootstrapFewShotWithRandomSearch** is a simple, reliable start.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/stanford-futuredata/lotus">GitHub - stanford-futuredata/lotus</a>: Contribute to stanford-futuredata/lotus development by creating an account on GitHub.</li><li><a href="https://x.com/lateinteraction/status/1815423177272824022">Tweet from Omar Khattab (@lateinteraction)</a>: 🚨When building LM systems for a task, should you explore finetuning or prompt optimization?  Paper w/ @dilarafsoylu @ChrisGPotts finds that you should do both!  New DSPy optimizers that alternate opt...</li><li><a href="https://arxiv.org/abs/2407.10930">Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together</a>: Natural Language Processing (NLP) systems are increasingly taking the form of multi-stage pipelines involving multiple distinct language models (LMs) and prompting strategies. Here we address the ques...</li><li><a href="https://docs.langwatch.ai/integration/python/guide#capturing-llm-spans">Python Integration Guide - LangWatch</a>: no description found
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1263973288133132348)** (8 messages🔥): 

> - `OpenPilot Model Run Analysis`
> - `Bitcast Functionality in Tinygrad`
> - `Promising Pull Requests`
> - `Weekly Tinygrad Meeting` 


- **Analyze OpenPilot model run performance**: George Hotz shared a [trace of a 14.64 ms OpenPilot model run](https://gist.github.com/geohot/8d7edc7ac2fd9a31ea563c134b66cddb) and outlined steps to document kernel changes and their potential slowdowns.
   - He emphasized that this task is approachable for anyone with a technical background, but noted that beginners often ask questions without thorough initial thought.
- **Debating bitcast shape consistency**: Tyoc213 raised a question on whether the `bitcast` function in Tinygrad should align with TensorFlow's `bitcast`, especially considering shape differences.
   - George Hotz and another member agreed that matching TensorFlow/Torch/Numpy makes sense, and Tyoc213 promised to follow up with the needed changes.
- **Most promising PR gets recognition**: George Hotz praised a [PR by tyoc213](https://github.com/tinygrad/tinygrad/compare/master...tyoc213-contrib:tinygrad:tyoc213/bitcast-all), calling it the most promising he's seen and noting that it included the expected tests.
   - Tyoc213 thanked him and mentioned plans to check other frameworks for further alignment.
- **Tinygrad weekly meeting key points**: Chenyuy shared an agenda for the Monday meeting, including updates on tinybox, hcopt speed recovery, and [MCTS search improvements](https://github.com/tinygrad/tinygrad/blob/master/extra/mcts_search.py).
   - Other highlights included better search features, conv backward fusing, fast Llama improvements, and various bounties for kernel and driver enhancements.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://stats.tinygrad.org>,">no title found</a>: no description found</li><li><a href="https://github.com/tinygrad/tinygrad/compare/master...tyoc213-contrib:tinygrad:tyoc213/bitcast-all">Comparing tinygrad:master...tyoc213-contrib:tyoc213/bitcast-all · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - Comparing tinygrad:master...tyoc213-contrib:tyoc213/bitcast-all · tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/pull/3740/files">Support shape changing bitcast by obadakhalili · Pull Request #3740 · tinygrad/tinygrad</a>: Addresses the first part of #3422:  Currently you can&#39;t bitcast between dtypes with different itemsize
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1263940950254686269)** (11 messages🔥): 

> - `Composing LazyBuffers`
> - `Shapetrackers Tutorial`
> - `Viability of Tinygrad versus PyTorch` 


- **Composing LazyBuffers debated**: Discussion about **composing lazybuffers** and how **srcs** and a **base** form a tree, sparking the idea of sequences.
   - Members compared it to **PyTorch's layout/view system**, but noted that **Tinygrad's** system seems more powerful and **shapetracker**-dependent.
- **Shapetrackers Tutorial Implementation Livestream**: A member shared a [YouTube video](https://www.youtube.com/watch?v=g1rCrv1fx1A) on implementing the **Shapetrackers** tutorial with a focus on view merging.
   - The video provided a detailed code walkthrough on a **tinygrad fork**, guiding viewers through the optimization techniques.
- **Viability of Tinygrad for PyTorch users questioned**: Members debated **tinygrad's viability** as a PyTorch alternative, with one member considering switching.
   - Questions were raised about whether to wait for version **1.0** or proceed with **0.9**, reflecting productivity concerns.



**Link mentioned**: <a href="https://www.youtube.com/watch?v=g1rCrv1fx1A">Tinygrad tutorial - shapetrackers and view merging - machine learning optimization</a>: tinygrad fork with code tutorial: https://github.com/Zaffer/tinygrad/tree/tuturial-notebooktinygrad docs: https://docs.tinygrad.org/tinygrad notes: https://m...

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1264054041751851071)** (4 messages): 

> - `Crowdstrike update`
> - `Python subinterpreters`
> - `Meta Llama 3.1` 


- **First day update at Crowdstrike**: [Vinceflibustier](https://fixupx.com/vinceflibustier/status/1814233715641389456) shared a lighthearted update about their first day at Crowdstrike, mentioning they pushed a little update and took the afternoon off. They ended the message with a peace sign emoji ✌️.
- **Discover Python subinterpreters**: A member shared a [tutorial on Python subinterpreters](https://realpython.com/python312-subinterpreters/?utm_source=perplexity) with the upcoming release of Python 3.12 and a preview of changes in Python 3.13, emphasizing enhancements in GIL control and parallelism.
   - The tutorial provides insights on Python subinterpreters, changes to CPython's global state, and potential enrichments in the succeeding version, [suggesting familiarity with Python basics and GIL](https://realpython.com/learning-paths/python-basics/).
- **Meta Llama 3.1 repo leak**: AlpinDale confirmed that Meta Llama 3.1 includes a 405B model distilled into 8B and 70B, with 128k context and a curious inability to draw unicorns.
   - [AlpinDale's posts](https://x.com/alpindale/status/1814814551449244058?s=12) note that the 405B's instruct tuning might be safety aligned and that the repo was accidentally made public before time, retaining the same architecture as Llama 3.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fixupx.com/vinceflibustier/status/1814233715641389456">Tweet from Vincent Flibustier 👽 (@vinceflibustier)</a>: First day at Crowdstrike, pushed a little update and taking the afternoon off ✌️</li><li><a href="https://x.com/alpindale/status/1814814551449244058?s=12">Tweet from Alpin (@AlpinDale)</a>: Have confirmed that there&#39;s 8B, 70B, and 405B. First two are distilled from 405B. 128k (131k in base-10) context. 405b can&#39;t draw a unicorn. Instruct tune might be safety aligned. The architec...</li><li><a href="https://x.com/alpindale/status/1814717595754377562?s=46">Tweet from Alpin (@AlpinDale)</a>: It seems like someone at HF forgot to private this repo on time, and Google indexed it:  sllhf/Meta-Llama-3.1-405B-Instruct-FP8  The 405B is llama 3.1? Very interesting. I wonder if they&#39;ll only r...</li><li><a href="https://realpython.com/python312-subinterpreters/?utm_source=perplexity">Python 3.12 Preview: Subinterpreters – Real Python</a>: In this tutorial, you&#x27;ll preview one of the upcoming features of Python 3.12 and a proposed change to Python 3.13, addressing how subinterpreters work in the CPython program. The changes are desc...
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1264352779724656710)** (8 messages🔥): 

> - `Deepseek chat v2 6.28`
> - `4o mini performance`
> - `Apple Watch support`
> - `Device shipping updates`
> - `Coqui model on MacOS` 


- **Deepseek chat v2 6.28 outperforms Deepseek coder**: A member mentioned that *Deepseek chat v2 6.28 update* is incredible, even outperforming *Deepseek coder* and being cheaper than *4o mini*.
- **4o mini excels at logic, struggles with code**: Although **4o mini** is great at logic and reasoning, it performs poorly at coding.
- **Inquiries about Apple Watch support for iOS app**: A member inquired whether the **iOS app** supports the **Apple Watch**, stating they have a research use case where the **01** could shine if supported.
- **Updates on device shipping timelines requested**: Users are asking for updates on when the **devices** will ship.
- **Contributing to the project as a developer**: A member asked if there is an opportunity for a capable developer to contribute to the project. The response indicated that help is welcome with many [open issues on GitHub](https://github.com).


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1264859221611118634)** (1 messages): 

> - `Augmentoolkit on GitHub`
> - `Pinokio Project Launch` 


- **Launch of Pinokio's Augmentoolkit on GitHub**: Pinokio's new [Augmentoolkit](https://github.com/pinokiofactory/augmentoolkit) has been released on GitHub for public access, featuring tools for enhancing AI applications.
   - The project launch was announced on multiple platforms including [Discord](https://discord.gg/TQdNwadtE4), [GitHub](https://github.com/pinokiocomputer/pinokio), and [Twitter](https://twitter.com/cocktailpeanut).
- **Pinokio Project Gathers Momentum**: The **Pinokio** project is gaining attention on social media and developer forums.
   - [Click here for more details on Twitter](https://twitter.com/cocktailpeanut) and join the discussion on [Discord](https://discord.gg/TQdNwadtE4).



**Link mentioned**: <a href="https://pinokio.computer/item?uri=https://github.com/pinokiofactory/augmentoolkit">Pinokio</a>: AI Browser

  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1264159625255587840)** (4 messages): 

> - `Finetuning with GPT models`
> - `Issues with OpenAI credits` 


- **Why finetuning with GPT models isn't common**: **Costs and vendor lock-in** are main reasons why GPT models are rarely fine-tuned. It involves using expensive API calls and dependency on specific company's infrastructure.
- **Problems receiving OpenAI credits**: Members have reported issues with not receiving their promised [OpenAI credits](https://link.to/openai-credits). One member shared organization ID **org-EX3LDPMB5MSmidg3TrlPfirU** and stated they had filled out forms multiple times.


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jarvis-labs](https://discord.com/channels/1238365980128706560/1241117895740625099/)** (1 messages): 

vishnu9158: Nope
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[east-coast-usa](https://discord.com/channels/1238365980128706560/1245411101718610001/)** (1 messages): 

karmapa: Yes how about late august meetup in NY?
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openpipe](https://discord.com/channels/1238365980128706560/1245927847437008896/1264115136164270090)** (1 messages): 

> - `Openpipe with other providers`
> - `Integration of Replicate models`
> - `Modal model compatibility` 


- **Openpipe with Replicate or Modal models**: A member inquired about using Openpipe with providers other than **OpenAI** or **Anthropic**, such as models hosted on **Replicate** or **Modal** with OpenAI compatible APIs.
   - *Anyone have insights?*
- **Integration of Replicate models into Openpipe**: Discussion centered around integrating models hosted on **Replicate** into **Openpipe**, ensuring API compatibility.
   - The main concern was the ease of adding these models while maintaining compatibility with the existing system.


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/1264575901673193625)** (1 messages): 

> - `Issues with credit allocation`
> - `Course forms` 


- **Credit allocation issue for course participants**: A member reported not receiving their credits despite filling out the necessary forms for their organization, **org-EX3LDPMB5MSmidg3TrlPfirU**.
   - They mentioned having filled the forms at the start of the course and again on the reporting day.
- **Repeated form submissions for course credits**: The same member reiterated filling in the forms multiple times to ensure the receipt of credits.
   - There seems to be an issue with credit allocation despite compliance with the form submission process.


  

---



### **LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1263995128284971128)** (3 messages): 

> - `OpenAI Scale Tier`
> - `TPS Calculation`
> - `GPT-4-o throughput` 


- **Confusion about new OpenAI Scale Tier**: A user expressed confusion about the new [OpenAI Scale Tier](https://openai.com/api-scale-tier), asking if anyone understood it.
   - They seemed particularly puzzled by the specific calculations involved in the throughput per second (TPS) for different models.
- **Unclear calculation for TPS in Pay-As-You-Go Tier**: A user questioned OpenAI's calculation of 19 tokens per second (TPS) on the pay-as-you-go tier, comparing it to GPT-4-o's throughput of around 80 tokens per second.


  

---


### **LLM Perf Enthusiasts AI ▷ #[jobs](https://discord.com/channels/1168579740391710851/1169107992587812864/1264720227552329778)** (1 messages): 

> - `Websim platform`
> - `Founding AI Engineer role`
> - `AI-assisted software creation`
> - `Non-deterministic programs`
> - `Human-AI system` 


- **Websim makes software creation malleable**: Websim aims to build the world's most malleable software creation platform where everyone can solve their own problems and realize their dreams.
- **Join Websim as a Founding AI Engineer**: Websim is looking for a Founding AI Engineer to establish foundations for quickly iterating on non-deterministic programs aimed at automated product development.



**Link mentioned**: <a href="https://websim.ai/">websim.ai</a>: no description found

  

---



### **Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1264662138966904944)** (2 messages): 

> - `Building with LLMs`
> - `BUD-E Voice Assistant` 


- **Lessons from a Year of Building with LLMs**: A user shared a [video and blog post](https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/) summarizing a three-part series on lessons learned by practitioners building with LLMs for a year.
   - The summary highlights tactical, operational, and strategic insights, with a recommendation to consume the content via video for better understanding.
- **BUD-E Voice Assistant Demo and Collaboration Invite**: A user shared a [YouTube video](https://youtu.be/O4IXfa8CROs) showcasing a demo of the open-source BUD-E voice assistant, inviting others to join their new Discord server for collaboration.
   - Daily online hackathons will begin at **9 PM CEST** to onboard new volunteers and coordinate project work.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/O4IXfa8CROs">BUD-E - Demo</a>: Join our Discord Community, try BUD-E yourself &amp; help us to build the voice assistant me and BUD-E talk about in the video:https://discord.gg/sTKSB2AwBvhttps...</li><li><a href="https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/">TLDR: 1 year of building with LLMs &#8211; D-Squared</a>: no description found
</li>
</ul>

</div>
  

---



### **AI Stack Devs (Yoko Li) ▷ #[team-up](https://discord.com/channels/1122748573000409160/1128471951963328512/)** (1 messages): 

ari991963: Hi all, I am Aria a 2D/3D artist, if you are interested to collaborate dm
  

---



### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1263992625602494536)** (1 messages): 

> - `Target Audience Clarification`
> - `Communication Strategy` 


- **Clarifying the Target Audience**: A member asked about the target audience and the primary intent behind the communication strategy.
   - The discussion highlighted different approaches for engineers, aspiring engineers, devrels, and solution architects when discussing products.
- **Communication Strategies for Different Roles**: Different strategies were discussed for communicating with engineers, devrels, solution architects, and aspiring engineers.
   - Each role may require tailored messages to effectively convey product features and benefits.


  

---



### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1264661996553502790)** (1 messages): 

> - `1 year of building with LLMs`
> - `TLDR series on LLMs`
> - `Lessons from LLM practitioners` 


- **TLDR Series on Lessons from Building with LLMs**: [Lessons from 1 Year of Building with LLMs](https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/) summarize a three-part series detailing tactical, operational, and strategic lessons learned.
   - *Six practitioners* published the series, which is recommended for those serious about LLMs.
- **Visual TLDR Video on LLM Learnings**: A visual [TLDR video](https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/) accompanies the blog post to make the lessons more accessible.
   - *The author suggests watching the video* for a better understanding of the visuals discussed.



**Link mentioned**: <a href="https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/">TLDR: 1 year of building with LLMs &#8211; D-Squared</a>: no description found

  

---



---



{% else %}


> The full channel by channel breakdowns have been truncated for email. 
> 
> If you want the full breakdown, please visit the web version of this email: [{{ email.subject }}]({{ email_url }})!
>
> If you enjoyed AInews, please [share with a friend](https://buttondown.email/ainews)! Thanks in advance!

{% endif %}
