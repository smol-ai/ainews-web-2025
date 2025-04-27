---
id: 673a1f3a-267d-4d1b-a03d-73462dbe536b
title: Not much happened today
date: '2024-04-02T21:04:12.327421Z'
original_slug: ainews-not-much-happened-today-8015
description: >-
  **RAGFlow** open sourced, a deep document understanding RAG engine with
  **16.3k context length** and natural language instruction support. **Jamba
  v0.1**, a **52B parameter** MoE model by Lightblue, released but with mixed
  user feedback. **Command-R** from **Cohere** available on Ollama library.
  Analysis of **GPT-3.5-Turbo** architecture reveals about **7 billion
  parameters** and embedding size of **4096**, comparable to OpenChat-3.5-0106
  and Mixtral-8x7B. AI chatbots, including **GPT-4**, outperform humans in
  debates on persuasion. **Mistral-7B** made amusing mistakes on a math riddle.
  Hardware highlights include a discounted **HGX H100 640GB** machine with 8
  H100 GPUs bought for $58k, and CPU comparisons between **Epyc 9374F** and
  **Threadripper 1950X** for LLM inference. GPU recommendations for local LLMs
  focus on VRAM and inference speed, with users testing **4090 GPU** and
  **Midnight-miqu-70b-v1.0.q5_k_s** model. Stable Diffusion influences gaming
  habits and AI art evaluation shows bias favoring human-labeled art.
companies:
  - cohere
  - lightblue
  - openai
  - mistral-ai
  - nvidia
  - amd
  - hugging-face
  - ollama
models:
  - jamba-v0.1
  - command-r
  - gpt-3.5-turbo
  - openchat-3.5-0106
  - mixtral-8x7b
  - mistral-7b
  - midnight-miqu-70b-v1.0.q5_k_s
topics:
  - rag
  - mixture-of-experts
  - model-architecture
  - model-analysis
  - debate-persuasion
  - hardware-performance
  - gpu-inference
  - cpu-comparison
  - local-llm
  - stable-diffusion
  - ai-art-bias
people: []
---


<!-- buttondown-editor-mode: plaintext -->> AI News for 4/1/2024-4/2/2024. We checked 5 subreddits and [**364** Twitters](https://twitter.com/i/lists/1585430245762441216) and **26** Discords (**382** channels, and **4481** messages) for you. Estimated reading time saved (at 200wpm): **463 minutes**.


So you have time to either:

- [watch a 30min intro to GPT from 3B1B](https://www.youtube.com/watch?v=wjZofJX0v4M&t=2s)
- [watch a 4hr complete inro to LLMs from Soheil Feizi](https://twitter.com/FeiziSoheil/status/1774833586736189911)
- [try out an open source Devin competitor scoring 12.29% on SWE-bench](https://twitter.com/jyangballin/status/1775114444370051582?t=90xQ8sGy63D2OtiaoGJuww)

And congrats to [Logan on joining Google](https://twitter.com/officiallogank/status/1775222819439149424?t=6FDPaNxZcbSsELal6Sv7Ug).

---

**Table of Contents**

[TOC] 


---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence. Comment crawling still not implemented but coming soon.


**Open Source Models and Libraries**

- **RAGFlow open sourced**: RAGFlow, a deep document understanding based RAG engine leveraging pip-library-etl-1.3b, is now open source. Key features include 16.3k context length, automated library parsing, example tuning, static and dynamic function analysis, and natural language instruction support. ([link](https://www.reddit.com/r/MachineLearning/comments/1bt0vg9/n_open_source_13b_multicapabilities_model_and/), [link](https://www.reddit.com/r/MachineLearning/comments/1bt1ky8/p_ragflow_the_deep_document_understanding_based/))
- **Jamba v0.1 released**: Lightblue released Jamba v0.1, a 52B parameter Apache-licensed model with Mixture-of-Experts (MoE) architecture. However, some users found its outputs repetitive and underwhelming compared to expectations. ([link](https://huggingface.co/lightblue/Jamba-v0.1-chat-multilingual), [link](https://www.reddit.com/r/LocalLLaMA/comments/1btg38m/thoughts_on_jamba/)) 
- **Command-R from Cohere**: Command-R, a model from Cohere available on the ollama library, is reportedly pretty good but not widely discussed. ([link](https://www.reddit.com/r/LocalLLaMA/comments/1bt9i4o/why_is_no_one_talking_about_commandr_from_cohere/))

**Model Performance and Capabilities**

- **GPT-3.5-Turbo architecture details**: An analysis of GPT-3.5-Turbo's logits estimated it has an embedding size of around 4096 and around 7 billion parameters, aligning with sizes of recent open models like OpenChat-3.5-0106 and Mixtral-8x7B. ([link](https://www.reddit.com/r/LocalLLaMA/comments/1btpk4h/logits_of_apiprotected_llms_leak_proprietary/))
- **AI chatbots beat humans in debates**: In a study, AI chatbots were more persuasive than humans in debates on contentious topics. People were more likely to change their minds when challenged by GPT-4 compared to human debaters. ([link](https://www.newscientist.com/article/2424856-ai-chatbots-beat-humans-at-persuading-their-opponents-in-debates/))
- **Mistral-7B makes amusing mistakes**: Mistral-7B made amusing mistakes when answering a simple math riddle about counting fruits, failing to understand that shoes are not fruits and giving inconsistent answers. ([link](https://www.reddit.com/r/LocalLLaMA/comments/1btc98u/i_have_4_oranges_1_apple_and_2_pairs_of_shoes_i/))

**Hardware and Performance**

- **Discounted HGX H100 machine**: Someone snagged a new HGX H100 640GB machine with 8 H100s on eBay for only $58k, a steep discount from the $270k retail price. ([link](https://i.redd.it/i8z79zdn7wrc1.png))
- **Epyc vs Threadripper for LLM inference**: A performance comparison of the Epyc 9374F and Threadripper 1950X CPUs on LLM inference tasks was posted. ([link](https://www.reddit.com/gallery/1bt8kc9))
- **GPU recommendations for local LLMs**: Advice was sought on the best GPU options (GTX 1070, RTX 3050 8GB, RTX 2060S) for running 7-13B parameter LLMs locally on Windows, with key considerations being VRAM and inference speed. ([link](https://www.reddit.com/r/LocalLLaMA/comments/1btbmmv/which_gpu_can_run_llm_locally_faster_gtx_1070_or/))
- **Optimizing local LLM with 4090 GPU**: A user with a 4090 GPU and 64GB RAM asked for recommendations on the best local LLM for uncensored roleplay/chatbot use, finding Midnight-miqu-70b-v1.0.q5_k_s too slow at 0.37 it/s. ([link](https://www.reddit.com/r/LocalLLaMA/comments/1bta2un/4090_64gb_ram_best_local_llm_for_uncensored_rpchat/))

**Stable Diffusion and Image Generation**

- **SD reduces gaming time**: Stable Diffusion reduced gaming time for some users who now prefer exploring their creativity with generative AI. ([link](https://www.reddit.com/r/StableDiffusion/comments/1bt03v1/has_sd_cut_down_your_gaming_time/))
- **Bias in evaluating AI art**: In a study, people preferred AI-generated art when they thought it was made by humans, but struggled to tell the difference, suggesting bias in evaluating AI art. ([link](https://www.sciencenorway.no/art-artificial-intelligence/people-liked-ai-art-when-they-thought-it-was-made-by-humans/2337417))
- **Regional prompting experiments**: Regional prompting experiments with A8R8, Forge, and a forked Forge Couple extension allow more granular control over image generation, with the new interface supporting dynamic attention regions, mask painting, and prompt weighting to minimize leakage. ([link](https://www.reddit.com/r/StableDiffusion/comments/1btrf4p/part_2_experimenting_with_regional_prompting_with/))
- **Choosing best epoch for LoRA**: Questions were raised about choosing the best epoch when training LoRA models, as results at 30% training sometimes looked better than 100%. Finding the right settings to recreate the LoRA's captured aesthetic also requires trial and error. ([link](https://www.reddit.com/r/StableDiffusion/comments/1btld1v/no_one_ever_discusses_the_postlora_settings_why/))
- **Recreating MidJourney styles in SD**: A user is learning to recreate MidJourney styles in Stable Diffusion for more control and consistency, seeking advice on reverse engineering a retro Siamese cat image. ([link](https://i.redd.it/lxkfcqjuywrc1.jpeg))

**Miscellaneous**

- **Google AI chief on AI hype**: Google's AI chief said the billions going into AI means "a bunch of hype and maybe some grifting", but we may be at the beginning of a new scientific renaissance. He believes AGI has a 50% chance of arriving in the next decade. ([link](https://www.reddit.com/r/LocalLLaMA/comments/1bt7bjf/googles_ai_chief_says_the_billions_going_into_ai/))
- **Frustration with OpenAI's dominance**: OpenAI's pervasiveness in workplaces is frustrating some who object to its closed model, profiting from web-scraped data, and abandonment of open source principles. However, there is pressure to use it to pay the bills. ([link](https://www.reddit.com/r/MachineLearning/comments/1bt9y8p/d_cant_escape_openai_in_my_workplace_anyone_else/))
- **One-click image tagging app**: A free Windows app was created to rename and add relevant metadata to images/GIFs using GPT vision in one click. ([link](https://v.redd.it/0x2h6v29tzrc1.gif))
- **DagsHub Storage Buckets launched**: DagsHub launched DagsHub Storage Buckets, an S3-compatible Google Drive alternative integrated with Google Colab, aiming to provide scalable storage for ML workflows. ([link](https://www.reddit.com/r/MachineLearning/comments/1bt5hxw/p_scalable_and_mloriented_google_drive/))

**Memes and Humor**

- **Stale Diffusion paper**: A humorous "Stale Diffusion" paper proposed hyper-realistic 5D movie generation using old-school methods. The authors lamented its rejection by even unserious venues. ([link](https://www.reddit.com/r/MachineLearning/comments/1bt9u0o/p_stale_diffusion_hyperrealistic_5d_movie/))
- **OpenAI removing Sam Altman meme**: An image macro joked about OpenAI removing Sam Altman's ownership of its startup fund. ([link](https://i.redd.it/5yv7ah8pxzrc1.png))
- **"I can take it" meme**: A meme depicted someone confidently claiming they can "take it" in response to an unknown challenge. ([link](https://i.redd.it/lrakavx6cvrc1.png))


# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.

**AI Models and Architectures**

- **DBRX**: [@DbrxMosaicAI](https://twitter.com/DbrxMosaicAI/status/1774916729149513956) noted DBRX is the **top open-source model on the latest WildBench Leaderboard on HuggingFace**, trained on 12T tokens of high-quality data with efficiency improvements like fine-grained MoE, GPT-4 tokenizer, and architecture modifications.
- **Jamba**: [@AI21Labs](https://twitter.com/AI21Labs/status/1774824070053331093) released the Jamba whitepaper detailing the **novel hybrid SSM-Transformer architecture interleaving Mamba, Transformer and MoE**. [@amanrsanger](https://twitter.com/amanrsanger/status/1774928438039810475) noted Jamba's KV cache is 8x smaller than a pure Transformer, but requires more memory for long context generation.
- **Gecko**: [@kelvin_guu](https://twitter.com/kelvin_guu/status/1774855490687918561) shared Gecko is the **strongest model under 768-dim on the Massive Text Embedding Benchmark (MTEB)**, available on Google Cloud for RAG, retrieval, vector databases, etc.
- **DenseFormer**: [@fchollet](https://twitter.com/fchollet/status/1774843303420420382) highlighted DenseFormer which takes a **weighted average of all prior blocks' output at each transformer block**, improving performance with a sparsification strategy to avoid IO bottlenecks.

**Retrieval Augmented Generation (RAG)**

- **RAG with LlamaParse and Local Models**: [@llama_index](https://twitter.com/llama_index/status/1774832426000515100) shared a tutorial on **building advanced PDF RAG with LlamaParse and local models** like @GroqInc, FastEmbed by @qdrant_engine, and flag-embedding-reranker for efficient RAG setups.
- **RAG with Chroma**: [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1774919695373603040) noted RAG is effective if retrieved information is relevant, and the course "Advanced Retrieval for AI with @trychroma" teaches **techniques to improve retrieval relevancy**.
- **RAFT**: [@llama_index](https://twitter.com/llama_index/status/1774814982322172077) is hosting a webinar on **retrieval-augmented fine-tuning (RAFT)** with @tianjun_zhang and @shishirpatil_, lead co-authors of RAFT, to discuss fine-tuning and RAG.
- **RAGFlow**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1774890566179774733) shared RAGFlow, a **RAG engine based on deep document understanding** that is now open-sourced and works with on-premise LLMs.

**Tooling and Infrastructure**

- **Keras 3 + JAX**: [@fchollet](https://twitter.com/fchollet/status/1774843979869343788) benchmarked Keras 3 + JAX against popular HuggingFace models, showing **1.5-3x speedups**, noting Keras leverages the compiler for performance so users can write simple, readable code.
- **Instructor**: [@jxnlco](https://twitter.com/jxnlco/status/1774813661900558442) released instructor 1.0.0 with **proper autocomplete, helper methods for partials, iterables, and original responses**, while maintaining the simple instructor.from_openai and instructor.patch APIs.
- **LangChain Financial Assistant**: [@virattt](https://twitter.com/virattt/status/1774909569723932850) added **Buffett-inspired tools to the LangChain Financial Assistant** like owner earnings, ROE, ROIC calculations, with plans to add price-based tools next.
- **FastEmbed**: [@qdrant_engine](https://twitter.com/qdrant_engine/status/1774723490567860634) announced FastEmbed now allows **generating efficient, interpretable sparse vector embeddings** using the SPLADE++ model.

**Research and Techniques**

- **Spatial Consistency in Text-to-Image Models**: [@RisingSayak](https://twitter.com/RisingSayak/status/1775018332191617436) shared a paper investigating **spatial consistency in T2I diffusion models**, improving it with systematic re-captioning methods and the SPRIGHT dataset.
- **Sequoia**: [@AlphaSignalAI](https://twitter.com/AlphaSignalAI/status/1774858806817906971) shared a paper on Sequoia, a **hardware-aware speculative decoding algorithm that can improve LLM inference speed by 10x** by optimizing the storage of speculated tokens based on available hardware.
- **Stealing from LLMs**: [@AlphaSignalAI](https://twitter.com/AlphaSignalAI/status/1774858806817906971) noted a paper showing you can **exploit logprobs from an LLM API to extract information about the model** like hidden dimensions or token embeddings.
- **Temporal Alignment**: [@AlphaSignalAI](https://twitter.com/AlphaSignalAI/status/1774858806817906971) shared a paper exploring **aligning an LLM's knowledge to a certain point in time** to create temporal grounding.
- **Pretraining Data**: [@cwolferesearch](https://twitter.com/cwolferesearch/status/1774817920704512013) highlighted a survey paper aggregating **best practices for creating high-quality LLM pretraining datasets** as more researchers share details on pretraining data construction.

**Memes and Humor**

- [@francoisfleuret](https://twitter.com/francoisfleuret/status/1774709480116060240) joked that **research is an "eternal stream of glorious sudoku or crossword grids designed by the universe itself"** and solutions accumulate on the "heap of human knowledge".
- [@teortaxesTex](https://twitter.com/teortaxesTex/status/1774703886558781763) poked fun at **Americans' ability to come up with "bizarre non-standard units of measurement** by comparing $stuff to something in their immediate visual field".
- [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1774821522932117613) shared a **meme about AI solving problems**, with the image showing a poorly trained GAN.
- [@suno_ai_](https://twitter.com/suno_ai_/status/1774857974260855071) joked about **SuNoise™, a new AI-generated frequency** "previously thought to be outside the range of human hearing" that only 2.5% of people can hear.
- [@AravSrinivas](https://twitter.com/AravSrinivas/status/1774885274809721140) quoted a **fake Steve Jobs quote for April Fool's Day**: "Don't believe all you read on the Internet".

---

# AI Discord Recap

> A summary of Summaries of Summaries

- **Claude 3 Haiku Impresses as Budget-Friendly Opus Alternative**: The smaller and cheaper **Claude 3 Haiku** model is generating buzz for its effective reasoning and trick question handling, posing as a cost-efficient alternative to Opus in Perplexity AI. Discussions also focused on Perplexity's potential plans to introduce ads and the preference for the Writing focus mode over All focus for cleaner LLM interactions. ([Perplexity AI Discord](https://discord.com/channels/1047197230748151888))

- **Gecko and Aurora-M Push Boundaries in Text Embedding and Multilingual LLMs**: The new **Gecko** model demonstrates robust performance on the Massive Text Embedding Benchmark (MTEB) and may accelerate diffusion model training, as detailed in its [Hugging Face paper](https://huggingface.co/papers/2403.20327) and [arXiv abstract](https://arxiv.org/abs/2403.20327). Meanwhile, the **Aurora-M** model, with 15.5B parameters, is geared towards multilingual tasks and has processed over 2 trillion training tokens, as highlighted on [Twitter](https://twitter.com/__z__9/status/1774965364301971849?s=20) and [arXiv](https://arxiv.org/abs/2404.00399). ([LAION Discord](https://discord.com/channels/823813159592001537))

- **Efficient Fine-Tuning Techniques Spark Debate**: Conversations in the Unsloth AI community revolved around strategies for dataset splitting, the efficacy of sparse fine-tuning (SFT) versus quantization methods like QLora, and the steep costs associated with model pre-training. Members also highlighted the need for robust detection systems to combat AI misuse and protect Discord servers from malicious bots and scams. ([Unsloth AI Discord](https://discord.com/channels/1179035537009545276))

- **Stable Diffusion Community Anticipates SD3 and Tackles Model Challenges**: The Stable Diffusion community is buzzing with anticipation for the 4-6 week release timeline of **Stable Diffusion 3 (SD3)**, while also addressing challenges with rendering facial and hand details using tools like Adetailer and various embeddings. Discussions touched on the rapid pace of AI development, ethical considerations around using professional artwork for training, and the potential memory demands of future SD versions. ([Stability.ai Discord](https://discord.com/channels/1002292111942635562))

- **Mojo 24.2 Introduces Python-Friendly Features as Tensor Talk Heats Up**: The **Mojo Programming Language** community is abuzz with the release of **Mojo 24.2**, which brings a host of Python-friendly features and enhancements. Discussions delved into Mojo's handling of parallelism, value types, and tensor performance optimizations. The announcement of the **MAX Engine and C/C++ interop** in Mojo also generated excitement for its potential to streamline **Reinforcement Learning (RL) Python training**. ([Modular Discord](https://discord.com/channels/1087530497313357884))

- **Tinygrad Grapples with AMD GPU Instability and Cultural Resistance**: The tinygrad community expressed frustration with severe system instability when using **AMD GPUs**, highlighting issues like memory leaks and non-recoverable errors. Skepticism was directed towards AMD's commitment to resolving underlying problems, with calls for open-source documentation and modern software practices. Discussions also touched on workaround strategies and the need for a fundamental cultural shift in AMD's approach to software and firmware. ([tinygrad Discord](https://discord.com/channels/1068976834382925865))

- **LLM Serving Platforms Compete as Triton Alternatives Emerge**: Discussions in the LM Studio and MAX Serving communities focused on the capabilities of different **LLM serving platforms**, with MAX Serving being explored as a potential alternative to Triton. Users sought guidance on migrating existing setups and inquired about support for features like GPU-hosted models. The LM Studio community also grappled with error messages and compatibility issues across various models and hardware configurations. ([LM Studio Discord](https://discord.com/channels/1110598183144399058), [Modular Discord](https://discord.com/channels/1087530497313357884))

- **Retrieval-Augmented Fine-Tuning (RAFT) Takes Center Stage**: LlamaIndex hosted a webinar featuring Retrieval-Augmented Fine-Tuning (RAFT) with lead co-authors Tianjun Zhang and Shishir Patil, delving into how RAFT combines the benefits of retrieval-augmented generation (RAG) and fine-tuning to improve language models' performance in domain-specific settings. The webinar aimed to provide insights and resources for those interested in implementing RAFT in their own projects. ([LlamaIndex Discord](https://discord.com/channels/1059199217496772688))

- **Axolotl Advances with Lisa Merge and DeepSpeed Challenges**: The Axolotl AI Collective celebrated the approval of the latest PR for `lisa` and the addition of a YAML example for testing. However, developers encountered out-of-memory errors when attempting to train models with DeepSpeed or FairScale Single-Process Single-GPU (FSDP). The collective also made strides in dataset unification efforts and expressed interest in exploring **runpod serverless** for very large language models (VLLM). ([OpenAccess AI Collective Discord](https://discord.com/channels/1104757954588196865))

- **FastLLM and RankLLM Push Boundaries in Retrieval and Reranking**: Qdrant introduced **FastLLM**, a language model boasting a 1 billion token context window, aimed at enhancing AI-driven content generation, as detailed in their [announcement post](https://qdrant.tech/blog/fastllm-announcement/). Meanwhile, **RankLLM** by @rpradeep42 et al., an open-source collection of LLMs fine-tuned for reranking, was recommended for those building advanced RAG systems, with emphasis on the importance of choosing the right reranker. ([HuggingFace Discord](https://discord.com/channels/879548962464493619), [LlamaIndex Discord](https://discord.com/channels/1059199217496772688))

---



# PART 1: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Claude 3 Haiku Enters the Fray**: The smaller and cheaper **Claude 3 Haiku** is generating buzz for its effective reasoning and trick question handling, posing as a cost-efficient alternative to Opus.

- **Perplexity Users Ponder Advertising Prospects**: Discussion thrives around a potential shift in **Perplexity AI's strategy** with the introduction of ads, stoking debates about the authenticity of recent announcements, possibly tied to an **April Fools' gag**.

- **Selecting the Superior Search**: Participants advocate for the *Writing* focus over the *All* focus in Perplexity for a streamlined and less problematic **Large Language Model (LLM) interaction**.

- **Prompt Defence Protocols in Question**: Security concerns heighten over **prompt attacks** on Perplexity AI's models. Discourse turns to the necessity for robust safeguards against malicious injections and data poisoning.

- **Price Tag Shock for Gemini 1.5 Pro API:** An active dialogue contests the steep pricing of **Gemini 1.5 Pro API**, leading to conversations about more budget-conscious tiered pricing structures based on **token consumption**.

- **Embracing the Cascade**: Keen members exchange insights into **Stable Cascades**, referencing Perplexity AI for in-depth understanding.

- **A Peek into Perplexity's Neo**: Queries launched to unravel what sets **Neo** apart, with an eye on distinct attributes.

- **Managing Perplexity Subscriptions Skillfully**: A hiccup arises as API credits get ensnared in "Pending" limbo, and the lack of a team signup option for Perplexity's API draws attention.

- **Comparing Token Economies**: Resources are shared to contrast the token expense between Perplexity and ChatGPT, fostering informed decisions for users.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

**Gecko Climbs to New Heights in Text Embedding**: The new *Gecko* model demonstrates robust performance on the Massive Text Embedding Benchmark (MTEB) and may accelerate diffusion model training, as detailed in its [Hugging Face paper](https://huggingface.co/papers/2403.20327) and [arXiv abstract](https://arxiv.org/abs/2403.20327). Interest in Gecko's practical application is reflected in queries about the availability of its weights.

**Aurora-M Lights Up Multilingual LLM Space**: The Aurora-M model, with 15.5B parameters, is geared towards multilingual tasks while adhering to guidelines set by the White House EO and is celebrated for processing over 2 trillion training tokens, as highlighted on [Twitter](https://twitter.com/__z__9/status/1774965364301971849?s=20) and [arXiv](https://arxiv.org/abs/2404.00399).

**Hugging Face's Diffusers Under the Spotlight**: Contributions to Hugging Face's *Diffusers* stirred debates around efficiency, with a focus on a PR regarding autocast for CUDA in Diffusers and incomplete unification in pipelines, as seen in [discussion #551](https://github.com/huggingface/diffusers/issues/551) and [PR #7530](https://github.com/huggingface/diffusers/pull/7530).

**PyTorch Gears Up with 2.6 Stirring Curiosity**: Discussions around updates in PyTorch versioning sparked interest, especially regarding the silent addition of bfloat16 support in PyTorch 2.3, and anticipation for new features in the upcoming PyTorch 2.6. Noteworthy contributions include a critique of autocast performance with details in a [GitHub thread](https://github.com/pytorch/pytorch/issues/120930).

**LangChain Event Hooks in AI Engineers With Harrison Chase**: Harrison Chase, CEO of LangChain, prepares to talk at an online event about leveraging LangSmith in moving from prototype to production on **April 17 at 6:30 PM**, with registration available [here](https://www.meetup.com/fr-FR/langchain-and-llm-france-meetup/events/300045589/). His company focuses on using LLMs for context-aware reasoning applications.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**Model Might on a Budget**: Guild members actively debated **cost vs quality in AI modeling**, with discussions ranging from **$50K to several million dollars** needed for pre-training varying in dataset size. A strong emphasis was placed on finding a balance between **resource efficiency** and maintaining **high-quality outputs**.

**Scam Shield Tightens**: Concerned with an increase in malicious bots and scams, the engineer community underscored the need for robust **detection systems** to thwart AI misuse and protect Discord servers.

**Precision in Saving Space**: Tips were shared on conserving space when saving finetuned models on platforms like **Google Colab**, with one user suggesting a method that saves 8GB of space but warned of a slight loss in accuracy.

**Training Tactics Tussle**: The optimal approach for **division of datasets** and the application of **sparse fine-tuning (SFT) versus quantization methods** was a hot topic, with insights into the trade-offs between performance and cost-effectiveness being highly sought after.

**Integration Enthusiasm for DeepSeek**: A user-proposed integration of the **DeepSeek model** into **Unslotsh 4bit**, showcasing the community's push for model diversity and efficiency improvements, with an accompanying [Hugging Face repository](https://huggingface.co/deepseek-ai) and a [Google Colab notebook](https://colab.research.google.com/drive/1NLqxHHCv3kFyw45t8k_CUfNlcepMdeDW?usp=sharing) set for implementation.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

**Cyberrealistic vs. EpicRealism XL**: Debate is ongoing about the performance of two Stable Diffusion models: while **Cyberrealistic** demands precise prompts, **EpicRealism XL** outshines with broader prompt tolerance for realistic imagery.

**SD3 Is Coming**: The community is buzzing with the 4-6 weeks anticipated release schedule for **Stable Diffusion 3 (SD3)**, with some doubt about the timing but evident excitement for improved features, notably a fixed text function.

**Fixing Faces and Hands**: The Stable Diffusion aficionados are tackling challenges with rendering facial and hand details, recommending tools such as **Adetailer** and various embeddings to enhance image quality without sacrificing processing speed.

**CHKPT Model Confusion**: In the sea of **CHKPT models**, users seek guidance for best use cases, pointing towards models like **ponyxl**, **dreamshaperxl**, **juggernautxl**, and **zavychroma** as part of a suggested checkpoint "starter pack" for Stable Diffusion.

**Ethics and Performance in Model Development**: Discussions touch on the rapid pace of AI development, ethical questions around using professional artwork for AI training, and speculated memory demands for future Stable Diffusion versions, all peppered with light-hearted community banter.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**DBRX Revealed**: A new open-source language model titled **DBRX** is making waves, claiming top performance on established benchmarks. [Watch the introduction of DBRX](https://www.youtube.com/watch?v=dqFvOqC43rQ).

**Whisper Models Under the Microscope**: **WhisperX** might replace **BetterTransformer** given concerns over the latter's high error rates. Community mulling over [Transforming the Web](https://www.f5.com/company/blog/transforming-the-web-the-end-of-silos) and [Apple's latest paper on reference resolution](https://arxiv.org/pdf/2403.20329.pdf).

**Speed Meets Precision in LLM Operations**: **LlamaFile** boasts 1.3x - 5x improved speed over llama.cpp on CPU for specific tasks, potentially altering future local operations. A configuration file for Hercules fine-tuning resulted in decreased accuracy, stirring debates over settings like `lora_r` and `lora_alpha`.

**Hugging Face Misstep Halts Upload**: **ModelInfo** loading issues caused by `safetensors.sharded` metadata from Hugging Face are preventing uploads to the chain, driving discussions for fixes.

**Brainstorming for WorldSim**: **WorldSim** enthusiasts propose a **"LLM Coliseum"** with competitive benchmarks, file uploads facilitating pre-written scripts, and speculation on future developments like competitive leaderboards and AI battles. 

**Traffic Signal Dataset Signals Opportunity**: A [traffic signal image dataset](https://huggingface.co/datasets/Sayali9141/traffic_signal_images) surfaced, promising to aid vision models despite Hugging Face's viewer compatibility issues.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**Trouble in GPU Paradise**: AMD GPUs are causing major headaches for tinygrad users, with system crashes and memory leak errors like `"amdgpu: failed to allocate BO for amdkfd"`. Users share workarounds involving PCIe power cycling but remain unimpressed by AMD's perceived lack of commitment to addressing these bugs.

**A Virtual Side-Eye to AMD's Program**: An invitation to AMD's Vanguard program drew skepticism from George Hotz and others, sparking a debate over the effectiveness of such initiatives and the need for open-source solutions and better software practices at AMD.

**Learning Curve for Linear uOps**: A detailed [write-up](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/uops.md) explaining linear uops was shared in the #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1224265332848328765) channel, aiming to demystify the intermediate representation in tinygrad, complemented by a [tutorial on the new command queue](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/commandqueue.md) following a significant merge.

**Tinygrad Pull Requests Under Scrutiny**: Pull Request [#4034](https://github.com/tinygrad/tinygrad/pull/4034) addressed confusion around unit test code and backend checks. A focus on maintaining proper test environments for various backends like CLANG and OpenCL is emphasized.

**Jigsaw Puzzle of Jitted Functions**: A knowledge gap regarding why jitted functions don't show up in command queue logs led to discussions about the execution of jitted versus scheduled operations within tinygrad’s infrastructure.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**LM Studio Tangles with Model Troubles**: Engage with caution, LM Studio is throwing unknown exceptions particularly with *estopian maid 13B q4* models on RTX 3060 GPUs, and users report crashes during prolonged inferencing. There's a growing need for **Text-to-Speech and Speech-to-Text functionality**, but currently, one must tether tools like *whisper.cpp* for voice capabilities.

**In Quest for Localized Privacy**: While the quest for privacy in local LLMs continues, one suggestion is to pair LM Studio with AnythingLLM for a confidential setup, though LM Studio itself does not have built-in document support. Meanwhile, **Autogen** is producing a mere 2 tokens at a time, leaving users to wonder about optimal configurations.

**GPU Discussions Heat Up**: SLI isn't necessary for multi-GPU setups; however, VRAM rather than combined VRAM is what's at play - an important spec for running models. A dual Tesla P40 setup is touting 3-4 tokens/sec for 70B models, while those on a budget admire P40s' VRAM, weighing it against the prowess of the 4090 GPU.

**Top Models for Anonymous Needs**: For the discrete engineer, the **Nous-Hermes 2 Mistral DPO** and **Nous-Hermes-2-SOLAR-10.7B** models come recommended, particularly for those needing to handle NSFW content. Tech hiccups with model downloads and execution have left some discontented, suspecting missing proxy support as the culprit.

**Desiring Previous Generation Functionality**: The convenience of splitting text on each new generation is missed, as current **LM Studio updates overwrite existing output**, prompting requests for a revert to the previous modality.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**Google Packs the Web in RAM**: Engineers noted Google's robust search performance may be due to embedding the web in RAM using a distributed version of FAISS and refined indexing strategies like **inverted indexes**. The discussion delved into Google's infrastructure choices, hinting at methods for handling complex and precise search queries.

**Sleuthing Google's Programming Paradigms**: Participants dissected Google's use of programming strategies that include otherwise shunned constructs like global variables and `goto`, illustrating a pragmatic approach to problem-solving and efficiency in their systems.

**Sparse Autoencoders Reveal Their Secrets**: A new visualization library for Sparse Autoencoder (SAE) has been released, shedding light on their feature structures. Mixed reactions to categorizing SAE features in AI models reflect both the detailed complexities and the abstract challenges in AI interpretability.

**New Horizons in Music AI**: A paper examining **GANs** and **transformers** in music composition was discussed, hinting at potential future directions in music AI, including text-to-music conversion metrics. Meanwhile, gaps in **lm-eval-harness** benchmarks for **Anthropic Claude** models suggest a growing interest in comprehensive model evaluation frameworks.

**Batch Size Trade-offs in GPT-NeoX**: Tuning **GPT-NeoX** for uneven batch sizes may introduce computational bottlenecks due to load imbalances, as larger batches hold up processing speed.

**Bonus Bullet for AI Sportsmanship**: Suggestions were made for EleutherAI community engagement in the [Kaggle AI Mathematical Olympiad competition](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/overview). Compute grants could support these inclinations towards "AI in science" initiatives.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT Goes Unrestricted**: OpenAI has introduced a new way to use [ChatGPT instantly](https://openai.com/blog/start-using-chatgpt-instantly), enabling access without sign-up requirements, aiming to broaden AI accessibility and confirming that user interactions help enhance model performance, with optional data contribution.
  
- **Prompt Pondering and Managerial Models**: Engineers discussed the efficacy of schema versus open-ended prompts in converting PDFs to JSON, raising concerns about potential Terms of Service breaches, and seeking advice on prompts to automate managerial tasks, including the division of directives and performance planning.

- **AI Creative Limits and Originality Inquisition**: A comparison of different AI responses to song recognition challenges revealed the boundaries of AI's creativity, and a [study](https://arxiv.org/abs/2310.17567) pointed to AI demonstrating emergent behaviors, potentially offering original outputs not evident in their training sets.

- **Anticipating GPT-5 and Navigating GPT-4**: Dialogues within the community reflected on the reflective capabilities of Large Language Models (LLMs), joked about April Fools' tech links, discussed GPT-4's advancements over Opus and server stability issues, and shared the use of **DALL-E 3's** image editing feature, with a nod towards the potential of the anticipated GPT-5.

- **AI Serving Diversity in Functions**: Engineers are exploring various AI tools, like **Claude 3 Sonnet** and **Midjourney**, for image description, and discussing compatibility challenges with AI apps on devices such as the Samsung Galaxy Note 9, with solutions involving checking system versions or utilizing mobile web browsers as alternatives.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Get Schooled on RAFT**: A **Retrieval-Augmented Fine-Tuning (RAFT)** webinar with Tianjun Zhang and Shishir Patil is scheduled for **Thursday at 9am PT**, promising insights into RAFT's advantages over traditional fine-tuning in language models. Prep materials include RAFT [blog posts](https://gorilla.cs.berkeley.edu/blogs/) and the full RAFT [paper](https://arxiv.org/pdf/2403.10131.pdf), with registration available [here](https://lu.ma/v1bdat63).

- **LlamaIndex's Call for Webinar Participation**: LlamaIndex is hosting a webinar on **RAFT**, comparing it to taking an "open-book exam," and has shared a schematic diagram for constructing RAG frameworks using different tools which can be found [here](https://twitter.com/llama_index/status/1774950945488834684) in a step-by-step guide.

- **Troubleshooting with LlamaIndex**: There were reports of outdated **LlamaIndex** documentation, difficulties with the `OpenAI.organization` setting, and deprecated models like `text-davinci-003`. Also discussed was the use of **WeatherReader** for weather-related queries within RAG, and manual methods for handling images in PDFs using **LlamaParse**.

- **Question Over-Simplification in Agent-Based Systems**: In the realm of creating a multi-document RAG system, one user highlighted an issue where the *top_agent* over-simplified the input question resulting in inadequate search outcomes. They shared details about incorrect narrowing of queries like "expiration date of chocolate," reducing it merely to "expiration date."

- **Tutorial Worth Watching**: A user recommended a YouTube tutorial on building a RAG application using **LlamaIndex**, highlighting integration with Pinecone and Gemini Pro for content scraping, embedding conversion, and querying, which can be accessed [here](https://youtu.be/B9mRMw0Jhfo).



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **JSON Juggling Woes**: Engineers are discussing challenges with parsing JSON in **LangChain**, where each line currently creates a separate Document instead of one Document with comprehensive metadata. The issue is detailed in the [JSON loader documentation](https://js.langchain.com/docs/integrations/document_loaders/file_loaders/json), but a solution has not been posted.

- **Token Tally Rising with Tool Use**: There's a noted 50% increase in token usage when LangChain agents employ tools, attributed to the tools' process of data retrieval and tokenization. While the system prompt is executed once for inference assumptions, not all tools necessitate this.

- **LangGraph Labyrinth**: Insights into utilizing a base model as the state in **LangGraph** were shared alongside a [Github notebook example](https://github.com/langchain-ai/langgraph/blob/961ddd49ed498df7ffaa6f6d688f7214b883b34f/examples/state-model.ipynb). Moreover, StructuredTool fields in LangChain can be validated using Pydantic's `BaseModel` and `Field` classes, as referenced in [Github issues](https://github.com/langchain-ai/langchain/issues/8066).

- **Fine-tuning Foil or Friend?**: Dialogues around achieving structured output from a chain suggest employing two agents to balance specialized knowledge and general intelligence post fine-tuning. However, no clear consensus or strategy has been provided to address this challenge.

- **PDFs and PersonaFinders Proliferate**: The discourse includes attempts to map content across PDFs using vector embeddings for matching paragraphs semantically, while a new release called **PersonaFinder GPT** promises conversational AI abilities based on identified personal attributes and invites testing on [PersonaFinder Pro](https://chat.openai.com/g/g-xm4VgOF5E-personafinder-pro).



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **LinkedIn Badges: Stat or Fad?**: A LinkedIn user flaunted having over 30 Top Voice badges, raising questions on the value of such accolades; [LinkedIn's badges in question](https://www.linkedin.com/posts/jillanisofttech_datascience-artificialintelligence-analyticalskills-activity-7180469402079789056-oPrY?utm_source=share&utm_medium=member_desktop).

- **AI Hallucinates, Developers Take Notes**: Software packages imagined by AI are being created and mistakenly utilized by major companies like Alibaba, showcasing a potential malware vector; more in [The Register's coverage](https://www.theregister.com/2024/03/28/ai_bots_hallucinate_software_packages/).

- **Billion-Token Models on the Horizon**: Qdrant introduces **FastLLM**, capable of a 1 billion token context window, aimed at enhancing AI-driven content generation; dive into the details in their [announcement post](https://qdrant.tech/blog/fastllm-announcement/).

- **Depth in Diffuser Channels**: Discussions focused on the intricacies of **LoRA** with **diffusers**, touching upon model queries without clear resolutions, and tackling the challenge of fine-tuning language models on PDF files without conclusive advice being provided.

- **Gradio 4.25 Debuts, Brings Enhanced UX**: **Gradio 4.25.0** rolls out features like auto-deletion of `gr.State` variables, `cache_examples="lazy"`, a fix for streaming audio outputs, and a more intuitive `gr.ChatInterface` to streamline user interactions.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Mojo Gets Mighty with MAX Engine**: The imminent introduction of the **MAX Engine and C/C++ interop** in Mojo aims to streamline **RL Python training**, potentially allowing Python environments to be speedily re-implemented in Mojo, as detailed in the [Mojo Roadmap](https://docs.modular.com/mojo/roadmap#cc-interop). Meanwhile, **Mojo 24.2** has excited developers with its focus on Python-friendly features, whose depth is explored in the [MAX 24.2 announcement](https://www.modular.com/blog/max-24-2-is-here-whats-new) and the blog post on [Mojo open-source](https://www.modular.com/blog/the-next-big-step-in-mojo-open-source).

**Tune in to Modular's Frequencies**: Modular's busy Twitter activity seems part of an outreach or announcement series, and details on their ideas can be tracked on [Modular's Twitter](https://twitter.com/Modular) for those interested in their updates or campaigns.

**Tensors, Tests, and Top-level Code Talk**: Open dialogue about the quirks and features of **Mojo** continued with insights like the need for improved **Tensor performance**, which was tackled by reducing copy initialization inefficiencies. Engineers also raised issues around **top-level code** and SIMD implementations, highlighting challenges like **Swift-style concurrency** and intrinsic function translations, with some guidance available in the [Swift Concurrency Manifesto](https://gist.github.com/lattner/31ed37682ef1576b16bca1432ea9f782).

**Unwrapping the CLI with Prism**: The `Prism` CLI library's overhaul brings new capabilities like shorthand flags and nested command structures, harmonizing with Mojo's 24.2 update. Enhancements include command-specific argument validators, with the development journey and usability of References being a point of focus, as seen on [thatstoasty's Prism on GitHub](https://github.com/thatstoasty/prism).

**Deploy with MAX While Anticipating GPU Support**: Questions about using **MAX as a Triton backend alternative** point to MAX Serving's utility, though currently lacking GPU support; documentation can guide trials via local Docker, found in the [MAX Serving docs](https://docs.modular.com/serving/get-started). Ongoing support and clarifications for prospective MAX adopters are discussed, emphasizing that ONNX models could fit smoothly into the MAX framework.

**Nightly Mojo Moves and Documentation**: Dedicated Mojo users were alerted about the **nightly build updates** and directed to use `modular update` commands, with changes listed in the [nightly build changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md). Additionally, valuable guidelines for local **Mojo stdlib development** and best testing practices are documented, suggesting `testing` module use over FileCheck and pointing to [stdlib development guide](https://github.com/modularml/mojo/blob/nightly/stdlib/docs/development.md#a-change-with-dependencies).



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Miniconda Shrinks the Stack**: **Miniconda** is validated as an effective, smaller substitute to Anaconda for those needing lighter installs without sacrificing functionality.

- **Call to Collaborate on OhanaPal**: The *OhanaPal* app is an innovative tool leveraging **OpenAI GPT APIs** to aid neurodivergent individuals, with the developers seeking contributors for further brainstorming and prototyping. Interested parties can engage through their [website](https://www.ohanapal.app/).

- **3D Printing Size Adjustments for Gadgets**: When 3D printing the **O1 Light**, scale the model up by 119.67% to properly accommodate an M5 Atom Echo, and a GitHub [pull request #214](https://github.com/OpenInterpreter/01/pull/214) enhances the M5Atom with auto-reconnection features.

- **Windows Package Management Enhanced**: A tip for Windows users: consider **winget** and **scoop** as viable tools for software package management, alongside the traditional Microsoft offerings.

- **Open Source AI Fosters Independence**: The **fabric** repository on GitHub provides an open-source AI augmentation framework to solve specific problems using crowdsourced AI prompts, and **Microsoft’s UFO** ([GitHub - microsoft/UFO](https://github.com/microsoft/UFO)) explores UI-Focused Agents for Windows interaction.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**Chatbot Prefix Quirk in OpenRouter**: The **undi95/remm-slerp-l2-13b:extended** model is unexpectedly prefixing responses with `{bot_name}:` in OpenRouter during roleplay chats; however, recent prompt templating changes were ruled out as the cause. The usage of the `name` field in this scenario is under investigation.

**SSL Connection Mystery**: A connection attempt to OpenRouter was thwarted by an **SSL error** described as *EOF occurred in violation of protocol*, yet the community did not reach a consensus on a solution.

**New Book Alert: Architecting with AI in Mind**: **Obie Fernandez** has launched an early release of his book, *Patterns of AI-Driven Application Architecture*, spotlighting OpenRouter applications. The book is accessible [here](https://leanpub.com/patterns-of-application-development-using-ai).

**Nitro Model Discussion Heats Up**: Despite concerns over the availability of nitro models, it's been affirmed that **nitro models are still accessible and forthcoming**. Confusion around the performance of different AI models suggests a prominent interest in optimizing speed and efficiency.

**Model Troubleshooting & Logit Bias**: Users encountered issues with models like **NOUS-HERMES-2-MIXTRAL-8X7B-DPO** and debated alternatives such as **Nous Capybara 34B** for specific tasks, noting its 30k context window for improved performance. Clarifications were made regarding OpenRouter logit bias application, which is currently limited to OpenAI's models only.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **NumPy's Unexpected Thread Behavior**: A member was surprised to find that **NumPy** wasn't fully utilizing threads, which was confirmed by benchmarking code showing better performance with a custom `matmul` function. This highlighted **NumPy's** suboptimal multi-threading capabilities.

- **Prompting for llamafile Documentation**: The impending release of **llamafile 0.7** sparked conversations on prompt templating within **openchat 3.5**, revealing a need for better documentation to clear confusion among users. The community eagerly awaits clearer guidance on integration specifics.

- **TinyBLAS Offers a CUDA-Free Alternative**: The discussion addressed **TinyBLAS** as an alternative for GPU acceleration, though it was noted that its performance is contingent upon the specific graphics card used. This option enables GPU support without needing the installation of CUDA or ROCm SDKs, which could significantly ease setup for some users.

- **Windows ARM64 Compatibility Hurdles with llamafile**: Users inquiring about **Windows ARM64** support for **llamafile** discovered that while the ARM64X binary format is supported, there are emulation issues with **AVX/AVX2**, a detail crucial to developers working within the Windows ARM64 ecosystem.

- **Local Deployment Troubles**: Participants encountered an **"exec format error"** during local deployment of **llamafile**, sparking a troubleshooting discussion that included suggestions to switch from zsh to bash and details on the correct execution of **Mixtral** models dependent on hardware configurations.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**15 Billion Reasons to Consider AMD**: MDEL's successful training of a **15B model** on **AMD GPUs** suggests that AMD may be a viable option in the hardware landscape for large-scale AI models.

**The Mystery of the Training Freeze**: Post-epoch training hangs were reported without the apparent use of `val_set_size` or `eval_table`, with hints suggesting the cause could be due to insufficient storage or yet-unidentified bugs in certain models or configurations.

**Axolotl Development Continues Amid Pranks**: The Axolotl Dev team approved a **PR merge for `lisa`**, added a **YAML example for testing**, and jovially proposed an April Fool's partnership with OpenAI. However, there are issues with missing documentation and out of memory errors potentially related to **DeepSpeed or FSDP** training attempts.

**Unified Data Dilemma**: There's a significant effort to combine 15 datasets into a unified format, with members tackling hurdles from data volume to misaligned translations.

**Rigorous Runpod Reviews Requested**: Interest has been shown in the use of **runpod serverless** offerings for very large language models, seeking insights from community experiences.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

**FastLLM Blasts into the AI Scene**: [Qdrant announced FastLLM (FLLM)](https://qdrant.tech/blog/fastllm-announcement/), a language model boasting a 1 billion token context window for Retrieval Augmented Generation, though skeptics suggest the timing of its announcement on April 1 may signal a jest.

**Visualization for Understanding GPTs**: A [visual introduction to Transformers and GPTs](https://www.youtube.com/watch?v=wjZofJX0v4M&t=2s) by popular YouTube channel 3Blue1Brown has garnered attention among AI professionals looking for a clearer conceptual understanding of these architectures.

**Engineers Build Open Source LLM Answer Engine**: An open source "llm-answer-engine" project unveiled on [GitHub](https://github.com/developersdigest/llm-answer-engine) has intrigued the community with its use of Next.js, Groq, Mixtral, Langchain, and OpenAI to create a Perplexity-Inspired Answer Engine.

**Structured Outputs from LLMs Become Simpler**: The engineering crowd noted the release of instructor 1.0.0, a tool aimed at ensuring Large Language Models (LLMs) produce structured outputs that conform to user-defined Pydantic models, assisting in seamless integration into broader systems.

**Google Powers Up AI Division**: In a pivot to bolster its AI offerings, Google has tapped Logan Kilpatrick to lead AI Studio and advance the [Gemini API](https://x.com/officiallogank/status/1775222819439149424?s), signaling the tech giant's intensified commitment to becoming the hub for AI developers.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Engaging with GPTs Made Easy**: Members highlighted an educational [video](https://www.youtube.com/watch?v=wjZofJX0v4M) designed to explain the fundamentals of transformers and GPTs in an easy-to-understand format, intended for those new to the machine learning field.
- **Geek Dreams Realized**: An ambitious project to create a homebrew GPU capable of running Quake was shared, demonstrating a successful FPGA design by an individual with a background in the gaming industry; further details can be found on their [blog](https://www.furygpu.com/blog/hello).
- **CPU Equals GPU? Not Quite**: A post from [Justine Tunney's blog](https://justine.lol/matmul/) was circulated discussing CPU matrix multiplication optimization tactics, noting the differences from GPU methods such as warptiling.
- **Triton Takes the Stage with Profiling**: The use of Nsight Compute in profiling Triton code was a major topic, with insights on optimizing performance and specific commands like `ncu --target-processes all --set detailed --import-source yes -o output_file python your_script.py` shared for better developmental workflow. Key performance improvements were emphasized, referencing resources from [Accelerating Triton](https://pytorch.org/blog/accelerating-triton/).
- **Benchmarking Battles**: Concerns were raised by the PyTorch team regarding recent benchmark comparisons with JAX and TensorFlow, prompting an official response, while a tweet by Jeff Dean presenting JAX as the fastest GPU performer for several tests stirred community discussion; the related benchmarking table is available [here](https://x.com/JeffDean/status/1774274156944859455).



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **RLAIF Could Boost Opus**: It's speculated that applying **Reinforcement Learning with Augmented Intermediate Features (RLAIF)** could further enhance **Opus** by refining its decision-making accuracy.

- **Google's Bold AI Aspiration**: A new AI product leader at **Google** announced their commitment to making Google the paramount destination for AI developers, supported by the **AI Studio** and **Gemini API**.

- **Advancements and Discussion in DPO**: A recent [preprint](https://x.com/rm_rafailov/status/1774653027712139657?s=46) explores verbosity issues in Direct Preference Optimization (DPO) at large scales. The discourse also mentioned the rebuttal of a study on verbosity exploitation in Reinforcement Learning from Human Feedback (RLHF), available on [arXiv](https://arxiv.org/abs/2403.19159).

- **A Veil Over AI Post-GPT-4**: Post-GPT-4, AI communities notice a trend toward increased secrecy from companies sharing less about model intricacies, deviating from prior norms of transparency.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

**Jamba's Speed Insight**: Engineers scrutinized how **Jamba's** end-to-end throughput efficiency improves with more tokens during the decoding process. Some members questioned the increase, given decoding is sequential, but the consensus highlighted that throughput gains exist even as the context size increases, impacting decoding speed.

**Decoding Efficiency Puzzler**: A pivotal discussion unfolded around a graph showing **Jamba's** decoding step becoming more efficient with a larger number of tokens. Confusion was addressed, and it was elucidated that the higher throughput per token affects decoding phase efficiency, countering initial misconceptions.

---

# PART 2: Detailed by-Channel summaries and links



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1224278402006122586)** (888 messages🔥🔥🔥): 

- **Claude 3 Haiku Holds Its Own Against Opus**: Users discussed the effectiveness of **Claude 3 Haiku** within Perplexity AI, analyzing how well it handles reasoning and trick questions compared to Opus, and its cost-effectiveness as a smaller, cheaper model.

- **Concerns Over Introducing Ads to Perplexity**: There's speculation and concern among users regarding **Perplexity AI's potential plans to introduce ads**, especially in relation to the **Pro subscription**. The credibility of the news, possibly being an **April Fools' joke**, is being debated, with references to articles from AdWeek and Gadgets360 discussing **advertising strategies**.

- **Prevalence of Writing Mode Focus**: Discussions centered around whether *Writing* focus mode within Perplexity is superior, with users suggesting it provides a **better user experience and less problematic results** than the *All* focus mode which encompasses web search. There is a clear preference for writing mode for its **cleaner LLM interactions**.

- **Prompt Attacks Security Concerns**: A user inquired about how Perplexity AI secures its models, like Sonar, against **prompt attacks** and other security vulnerabilities. The conversation shifted towards the broader issue of protecting LLMs against policy violations due to poisoned data or **prompt injections**.

- **Gemini 1.5 Pro API Pricing Commentary**: Users discussed the preview pricing of **Gemini 1.5 Pro**, which is noted to be expensive at **$7 per million tokens** for its 1 million token context ability. Conversations point to hopes for future price adjustments and the potential for tiered pricing based on context window usage.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://community.spiceworks.com/">no title found</a>: no description found</li><li><a href="https://x.com/LinusEkenstam/status/1774847013752070457?t=7tzw85sz9QgE_TN7zRA82Q&s=09">Tweet from Linus ●ᴗ● Ekenstam (@LinusEkenstam)</a>: 🚨 Breaking 🚨  Apple is in talks to acquire perplexity  This could be the start of something very exciting</li><li><a href="https://fxtwitter.com/AravSrinivas/status/1775229252973334902?t=p2-h_dWeQhz6swoCVL66SA&s=19">Tweet from Aravind Srinivas (@AravSrinivas)</a>: good vibes are essential</li><li><a href="https://tenor.com/view/tiny-text-cant-see-ken-jeong-gif-5957945">Tiny Text GIF - Tiny Text Cant - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/working-on-it-under-construction-gif-23162421">Working On GIF - Working On It - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/whine-give-up-pout-frustrated-gif-5313242348381105216">Whine Give GIF - Whine Give Up - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/ouch-gif-12136515515962044163">Ouch GIF - Ouch - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/yes-no-gif-16236377">Yes No GIF - Yes No - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/degout%C3%A9-chanceux-chance-chance-d%C3%A9butant-mr-miyaki-gif-22039391">Degouté Chanceux GIF - Degouté Chanceux Chance - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.instagram.com/perplexity.ai">Login • Instagram</a>: no description found</li><li><a href="https://www.adweek.com/media/gen-ai-search-engine-perplexity-has-a-plan-to-sell-ads/">Gen-AI Search Engine Perplexity Has a Plan to Sell Ads</a>: no description found</li><li><a href="https://www.reddit.com/r/singularity/comments/1bp885i/claude_3_haiku_is_the_new_budget_king/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.instagram.com/perplexityai?igsh=ZDg4MmZseWoweDJh">Login • Instagram</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=t-Nz6us7DUA">Morpheus explains what is real</a>: Morpheus says, “Your appearance now is what we call residual self-image. It is the mental projection of your digital self.”  Running his hands along a winged...</li><li><a href="https://www.youtube.com/watch?v=rie-9AEhYdY">WE MUST ADD STRUCTURE TO DEEP LEARNING BECAUSE...</a>: Dr. Paul Lessard and his collaborators have written a paper on &quot;Categorical Deep Learning and Algebraic Theory of Architectures&quot;.  They aim to make neural ne...</li><li><a href="https://www.youtube.com/watch?v=poncZ1K9Tio">Scrubs - Two Coins, 30 cents</a>: Funny clip from Scrubs Episode 304, season 3.</li><li><a href="https://www.cbsnews.com/news/att-data-breach-2024-cbs-news-explains/">What customers should know about AT&amp;T's massive data breach</a>: AT&amp;T said Saturday that a dataset found on the dark web&#8203; contains information such as Social Security numbers and passcodes for roughly 73 million current and former customers.</li><li><a href="https://www.gadgets360.com/ai/news/perplexity-ai-powered-search-engine-could-soon-show-ads-report-5357479">AI Search Engine Perplexity Could Soon Show Ads to Users: Report</a>: As per the report, Perplexity will show ads in its related questions section.</li><li><a href="https://slashdot.org/story/24/04/01/1653221/perplexity-an-ai-startup-attempting-to-challenge-google-plans-to-sell-ads">Perplexity, an AI Startup Attempting To Challenge Google, Plans To Sell Ads - Slashdot</a>: An anonymous reader shares a report: Generative AI search engine Perplexity, which claims to be a Google competitor and recently snagged a $73.6 million Series B funding from investors like Jeff Bezos...</li><li><a href="https://youtu.be/wjZofJX0v4M?feature=shared">But what is a GPT?  Visual intro to Transformers | Deep learning, chapter 5</a>: An introduction to transformers and their prerequisitesEarly view of the next chapter for patrons: https://3b1b.co/early-attentionSpecial thanks to these sup...</li><li><a href="https://tenor.com/view/spongebob-gif-7921357">Spongebob GIF - Spongebob - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/the-bachelorette-you-see-what-im-saying-understand-get-it-you-feel-me-gif-16430678">The Bachelorette You See What Im Saying GIF - The Bachelorette You See What Im Saying Understand - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/i-feel-your-pain-all-the-feels-keezia-leigh-comfort-i-feel-you-gif-16152618">I Feel Your Pain All The Feels GIF - I Feel Your Pain All The Feels Keezia Leigh - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/protect-attack-punch-me-fight-me-prepare-gif-17784080">Protect Attack GIF - Protect Attack Punch Me - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://downforeveryoneorjustme.com/perplexity">Perplexity down? Current problems and status. - DownFor</a>: Perplexity won't load? Or, having problems with Perplexity? Check the status here and report any issues!</li><li><a href="https://docs.perplexity.ai/page/application-status">Application Status</a>: no description found</li><li><a href="https://www.quora.com">Quora - A place to share knowledge and better understand the world</a>: no description found</li><li><a href="https://www.reddit.com/r/AskReddit">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/NoStupidQuestions">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/explainlikeimfive">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/subreddits/search">search results</a>: no description found</li><li><a href="https://community.spiceworks.com">no title found</a>: no description found</li><li><a href="https://discuss.codecademy.com">Codecademy Forums</a>: Community discussion forums for Codecademy.</li><li><a href="https://hashnode.com">Start a Developer Blog: Hashnode - Custom Domain, Sub-path, Hosted/Headless CMS.</a>: Developer blogging with custom domains, hosted/headless CMS options. Our new headless CMS streamlines content management for devtool companies.
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1224304009352056885)** (19 messages🔥): 

- **Detecting the Details of Stable Cascades**: A member shared a link to [Perplexity AI](https://www.perplexity.ai/search/Stable-cascade-7jHkofc5RZCwQbYUqWWy4g) concerning the details and insights into **Stable Cascades**.
- **Exploring Neo's Uniqueness**: Curiosity about what makes **Neo** stand out led to a search query on [Perplexity AI](https://www.perplexity.ai/search/why-is-Neo-26.mxjH_TEmzTnxhxetBDQ).
- **Clarity on How to Proceed with Perplexity**: A member's query redirected to [Perplexity AI](https://www.perplexity.ai/search/How-can-I-VBSI6VXuQOiKraNH.DMbrw), while a reminder was given about ensuring threads are shareable with a **Discord** [link](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825) for reference.
- **April Fools’ or Tech Innovation?**: The line between humor and advancement was explored with a [Perplexity AI search about April fool technology](https://www.perplexity.ai/search/April-fool-tech-Au6YyiG1TCCPZdIBAawqHw).
- **GPTDevil Collection Intrigues**: One member pointed to the [Perplexity AI collections](https://www.perplexity.ai/collections/GPTDevil-A.GvNcQZS0yjMGDDJtDPUQ) about **GPTDevil**, signaling interest in the topic.
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1224277874119544852)** (16 messages🔥): 

- **Issue with Adding API Credits**: A member reported a problem with the addition of API credits, stating that the process gets stuck on "Pending" and the invoice displays as "void."
- **Request for Retrieving Sources in API**: A member inquired about the possibility of getting sources for prompts through the Perplexity API, similar to the feature available in the browser prompt.
- **Token Costs Comparison Resources**: A member provided links for others who are looking to compare token costs for Perplexity models and ChatGPT, with [Perplexity's pricing](https://docs.perplexity.ai/docs/pricing) and OpenAI’s pricing available for review.
- **No Team Signup for Perplexity API**: A member asked about the process for signing up a team for the Perplexity API and was informed that team signups are currently not an option.
- **Confusion Over Rate Limits**: A member raised concerns regarding inconsistent rate limiting behavior when using the `sonar-medium-online` model, experiencing 429 errors despite adhering to the stated 20 requests per minute limit.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.perplexity.ai/docs/pricing">Pricing</a>: no description found</li><li><a href="https://openai.com/pricing">Pricing</a>: Simple and flexible. Only pay for what you use.
</li>
</ul>

</div>
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1224372448158289980)** (525 messages🔥🔥🔥): 

- **Hugging Face Diffusers PR Discussion**: Community members discussed a PR about disabling autocast for CUDA devices on the [Hugging Face Diffusers GitHub](https://github.com/huggingface/diffusers/pull/7530). The conversation pivots to a critique of Hugging Face for not having unified code across different pipelines and trainers, pointing to efficiency and absurdity.
  
- **Persistent Issue with Merging Different SD Pipelines**: Community members highlighted an ongoing issue captured in [GitHub discussion #551](https://github.com/huggingface/diffusers/issues/551) about merging different Stable Diffusion pipelines, noting the complication persists due to being decided to keep separate pipelines.

- **Criticisms of Hugging Face's Engineering Priorities**: A discussion emerged criticizing Hugging Face's engineering work on Diffusers, relating to both a lack of enough engineers and too many 'AI code thought leaders,' as well as conflicting approaches like the adoption of microservices frameworks by the engineers.

- **PyTorch Version Specific Discussions**: Community members had extensive technical discussions on pytorch versions, mentioning the silent addition of bfloat16 support in PyTorch 2.3 and the complexities of nightly builds. There were comments on autocast performance issues and possible fixes, details added on a [GitHub thread](https://github.com/pytorch/pytorch/issues/120930), and the anticipation for the PyTorch 2.6 release.

- **AI Generated Images and Sampling Settings**: The quality of images generated by various diffusion model versions and configurations were critiqued, with a particular focus on images of hammers. Differences in samplers and their configurations led to an exchange on the efficacy and correct use of these parameters.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/StableDiffusion/comments/1axbjrp/psa_recent_pytorch_nightlies_support_enough/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://tenor.com/view/old-gregg-easy-calm-down-relax-man-peach-gif-26522310">Old Gregg Easy GIF - Old Gregg Easy Calm Down - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/JrNL.gif">Tom And Jerry Mouse GIF - Tom And Jerry Mouse Bumped - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/blKOX.gif">Pelicula Western GIF - Pelicula Western Meme - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/pytorch/pytorch/issues/120930>">Issues · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - Issues · pytorch/pytorch</li><li><a href="https://github.com/pytorch/pytorch/issues/71631>">Issues · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - Issues · pytorch/pytorch</li><li><a href="https://github.com/huggingface/diffusers/issues/7563">[mps] training / inference dtype issues · Issue #7563 · huggingface/diffusers</a>: when training on Diffusers without attention slicing, we see: /AppleInternal/Library/BuildRoots/ce725a5f-c761-11ee-a4ec-b6ef2fd8d87b/Library/Caches/com.apple.xbs/Sources/MetalPerformanceShaders/MPS...</li><li><a href="https://github.com/huggingface/diffusers/pull/7530/files">7529 do not disable autocast for cuda devices by bghira · Pull Request #7530 · huggingface/diffusers</a>: What does this PR do?   Fixes #7529 Before submitting   This PR fixes a typo or improves the docs (you can dismiss the other checks if that&amp;#39;s the case).  Did you read the contributor guideline...</li><li><a href="https://github.com/huggingface/diffusers/pull/7530#discussion_r1547822696">7529 do not disable autocast for cuda devices by bghira · Pull Request #7530 · huggingface/diffusers</a>: What does this PR do?   Fixes #7529 Before submitting   This PR fixes a typo or improves the docs (you can dismiss the other checks if that&#39;s the case).  Did you read the contributor guideline?  D...</li><li><a href="https://github.com/huggingface/diffusers/issues/551">Merging Stable diffusion pipelines just makes sense · Issue #551 · huggingface/diffusers</a>: Following the Philosophy, it has been decided to keep different pipelines for Stable Diffusion for txt-to-img, img-to-img and inpainting. Here is the result: PR #549 : code duplicated 4 times (onnx...
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1224354219436281906)** (9 messages🔥): 

- **Introducing Gecko for Efficient Text Embedding**: Gecko, a compact text embedding model, showcases strong retrieval performance by distilling knowledge from large language models into a retriever. It outperforms existing models on the Massive Text Embedding Benchmark (MTEB), with details available in the [Hugging Face paper](https://huggingface.co/papers/2403.20327) and the [arXiv abstract](https://arxiv.org/abs/2403.20327).

- **Potential Application of Gecko in Diffusion Models**: The conversation suggests exploring the use of Gecko to potentially accelerate diffusion model training, replacing the usage of T5. The discussion is speculative about the impact on model performance, especially in terms of embeddings.

- **Gecko Weights Inquiry**: A member inquired if the weights for the aforementioned Gecko are available, indicating interest in its practical application.

- **Assessing Large Vision-Language Models**:
The [MMStar Benchmark](https://mmstar-benchmark.github.io/) examines the efficacy of evaluating Large Vision-Language Models, pinpointing issues such as unnecessary visual content for problem-solving where text suffices.

- **Announcement of Aurora-M, a Multilingual LLM**: The new preprint for Aurora-M, a 15.5B parameter, red-teamed, open-source, and continually pre-trained multilingual large language model, is introduced. It has processed over 2T training tokens and meets the guidelines of the White House EO, with more details found on [Twitter](https://twitter.com/__z__9/status/1774965364301971849?s=20) and [arXiv](https://arxiv.org/abs/2404.00399).

- **Improving Spatial Consistency in t2i Translations**: Incorporating better spatial descriptions in captions during fine-tuning enhances the spatial consistency of images generated by text-to-image models. The study's results are detailed in an [arXiv preprint](https://arxiv.org/pdf/2404.01197.pdf).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mmstar-benchmark.github.io/">MMStar</a>: no description found</li><li><a href="https://huggingface.co/papers/2403.20327">Paper page - Gecko: Versatile Text Embeddings Distilled from Large Language Models</a>: no description found</li><li><a href="https://x.com/__z__9/status/1774965364301971849?s=20">Tweet from ً ‎ (@__z__9)</a>: New preprint! The first multi-lingual red-teamed open-source continually pre-trained LLM - **Aurora-M** in accordance with the #WhiteHouse Executive Order on the Safe, Secure, and Trustworthy developm...</li><li><a href="https://arxiv.org/abs/2404.00399">Aurora-M: The First Open Source Multilingual Language Model Red-teamed according to the U.S. Executive Order</a>: Pretrained language models underpin several AI applications, but their high computational cost for training limits accessibility. Initiatives such as BLOOM and StarCoder aim to democratize access to p...
</li>
</ul>

</div>
  

---


**LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/1224793707606184117)** (1 messages): 

- **Join the LangChain Event with Harrison Chase**: Harrison Chase, the CEO and co-founder of **LangChain**, will be speaking at an upcoming event on the challenges companies face when moving from prototype to production using LangSmith. The online meetup is scheduled for **April 17 at 6:30 PM** and interested individuals can register [here](https://www.meetup.com/fr-FR/langchain-and-llm-france-meetup/events/300045589/).
- **From Prototypes to Production, Harrison Chase Explains**: The talk will cover the transition from easy-start **GenAI apps** to full production, highlighting the new challenges this brings. Chase's company, LangChain, focuses on simplifying the use of LLMs for developing context-aware reasoning applications.

**Link mentioned**: <a href="https://www.meetup.com/fr-FR/langchain-and-llm-france-meetup/events/300045589/">Meetup #3 LangChain and LLM: Using LangSmith to go from prototype to production, mer. 17 avr. 2024, 18:30   | Meetup</a>: Nous avons le plaisir d&#x27;accueillir Harrison Chase, le Co-Founder et CEO de LangChain, pour notre troisième Meetup LangChain and LLM France !  Ne loupez pas cette occasion u

  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1224281655204642826)** (212 messages🔥🔥): 

- **Discussions on Efficient Model Training**: Members discussed strategies for forming conversational AI models, pondering whether to keep datasets split based on response times or use LLMs for higher quality samples. The consensus tilted towards starting simple before proceeding to more complex splitting.
- **Quantization and Fine-Tuning Practices Debated**: Conversations revolved around the efficacy and performance loss involved in quantization methods like QLora versus Sparse Fine Tuning (SFT), where fine-tuning on blocks (LoRA/QLoRA) may not match the individual weight focus of SFT. Users shared insights into the balance between resource efficiency and model quality.
- **Cost Conversations**: The community exchanged thoughts on the steep costs associated with model pre-training, with speculation on prices ranging from $50K for small-scale efforts to millions for larger datasets and more resources, highlighting a focus on cost-efficiency and the potential need for more affordable training mechanisms.
- **Prevention Against Scams and Malicious Bots Urged**: Members noted an uptick in bots and scams targeting Discord servers, pointing out the risks of AI technologies being leveraged by scammers and the need for increased vigilance and better detection systems for fraudulent content.
- **Fine-Tuning Model Prompt Clarifications and Tutorials**: Specific advice was given on fine-tuning prompts and model formats, with conversations about the need for clearer tutorials, video content on fine-tuning processes, and the utility of collaborative notebooks and GitHub resources for the community.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/muzeke-gif-27066384">Muzeke GIF - Muzeke - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/jondurbin/airoboros-gpt-3.5-turbo-100k-7b">jondurbin/airoboros-gpt-3.5-turbo-100k-7b · Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1br8ry8/finetuning_a_llm_for_longform_creative_writing/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://tenor.com/view/am-ia-joke-to-you-am-ia-joke-is-this-a-joke-do-you-think-this-is-funny-do-you-think-this-is-a-joke-gif-14191111">Am Ia Joke To You Is This A Joke GIF - Am IA Joke To You Am IA Joke Is This A Joke - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/i-aint-no-fool-wiz-khalifa-still-wiz-song-im-not-a-fool-im-not-an-idiot-gif-21822363">I Aint No Fool Wiz Khalifa GIF - I Aint No Fool Wiz Khalifa Still Wiz Song - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=g68qlo9Izf0&t=850s">Efficient Fine-Tuning for Llama-v2-7b on a Single GPU</a>: The first problem you’re likely to encounter when fine-tuning an LLM is the “host out of memory” error. It’s more difficult for fine-tuning the 7B parameter ...</li><li><a href="https://github.com/unslothai/unsloth/wiki#chat-templates">Home</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/intel-analytics/ipex-llm">GitHub - intel-analytics/ipex-llm: Accelerate local LLM inference and finetuning (LLaMA, Mistral, ChatGLM, Qwen, Baichuan, Mixtral, Gemma, etc.) on Intel CPU and GPU (e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max). A PyTorch LLM library that seamlessly integrates with llama.cpp, HuggingFace, LangChain, LlamaIndex, DeepSpeed, vLLM, FastChat, ModelScope, etc.</a>: Accelerate local LLM inference and finetuning (LLaMA, Mistral, ChatGLM, Qwen, Baichuan, Mixtral, Gemma, etc.) on Intel CPU and GPU (e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max)...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1224319749425725440)** (311 messages🔥🔥): 

- **Unsloth Update Breaks Inference**: Users reported encountering size mismatch errors during inference with Unsloth AI after an update. A fix was applied, reverting changes, which resolved the issue for users.

- **Model Saving Challenges on Colab**: A user struggling with saving a finetuned 13B model on limited storage available on Google Colab was advised to try Kaggle for its free access to 2x Tesla T4s. Another user suggested using the `model.save_pretrained_gguf("model", tokenizer, quantization_method = "q5_k_m", first_conversion = "q8_0")` method to save 8GB of space but noted a potential 0.1% loss in accuracy.

- **Jamba Model Support Speculation**: Conversation about the complexity of adding Jamba model support to Unsloth, acknowledging the difficulty due to it being a Mamba and MoE model.

- **Finetuning Evaluation Clarification**: A detailed response was given regarding the Unsloth SFT Trainer's evaluation process, explaining how to get evaluation metrics by explicitly passing the evaluation dataset and strategy.

- **Load Dataset Slowness and Potential IPv6 Issue**: Users discussed significant delays when using `load_dataset` in local Jupyter notebooks, suspecting it might be related to IPv6 settings on Ubuntu systems. The same command was reported to work normally on Windows with WSL and IPv4.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ollama.com/pacozaa/tinyllama-alpaca-lora">pacozaa/tinyllama-alpaca-lora</a>: Tinyllama Train with Unsloth Notebook, Dataset https://huggingface.co/datasets/yahma/alpaca-cleaned</li><li><a href="https://docs.wandb.ai/guides/track/jupyter">Track Jupyter Notebooks | Weights &amp; Biases Documentation</a>: se W&amp;B with Jupyter to get interactive visualizations without leaving your notebook.</li><li><a href="https://colab.research.google.com/drive/1lBzz5KeZJKXjvivbYvmGarix9Ao6Wxe5?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://huggingface.co/unsloth/mistral-7b-instruct-v0.2-bnb-4bit/tree/main">unsloth/mistral-7b-instruct-v0.2-bnb-4bit at main</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">Home</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#manually-saving-to-gguf">Home</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/issues/295">Inference is not working · Issue #295 · unslothai/unsloth</a>: Hey, I am running the colab you have on the docs on Mistral, the training works fine, but when doing inference I get size mismatch error: Setting `pad_token_id` to `eos_token_id`:2 for open-end gen...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1224402368553291997)** (4 messages): 

- **DeepSeek Model Suggestion for Unslotsh 4bit**: A member proposed adding the smallest **DeepSeek model** to **Unslotsh 4bit**, referring to it as a good base model, and provided the official [Hugging Face repository](https://huggingface.co/deepseek-ai) for DeepSeek. Another member agreed with the suggestion and confirmed they would implement it.
- **Colab Notebook for Consideration**: A shared [Google Colab notebook](https://colab.research.google.com/drive/1NLqxHHCv3kFyw45t8k_CUfNlcepMdeDW?usp=sharing) was flagged for review, indicating that the material within should be **implemented**.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1NLqxHHCv3kFyw45t8k_CUfNlcepMdeDW?usp=sharing#scrollTo=Rdsd82ngpHCG">Google Colaboratory</a>: no description found</li><li><a href="https://huggingface.co/deepseek-ai/">deepseek-ai (DeepSeek)</a>: no description found</li><li><a href="https://chat.deepseek.com/">DeepSeek</a>: Chat with DeepSeek AI.
</li>
</ul>

</div>
  

---



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1224258874895503440)** (377 messages🔥🔥): 

- **Cyberrealistic vs. EpicRealism XL**: Participants discussed the Cyberrealistic and EpicRealism XL models in relation to realistic image generation. They found that while Cyberrealistic requires detailed prompts, EpicRealism XL produces better outcomes with more relaxed prompts.

- **SD3 Anticipation**: There's community anticipation for SD3 release with a timeframe mentioned of 4-6 weeks from a previous announcement. Doubts are expressed regarding the release timing, with some users sharing their eagerness for the new version's capabilities and improvements, especially the fixed text function.

- **Face and appendage model challenges**: Users described issues with face and hand rendering using Stable Diffusion, with various fixes like Adetailer and embeddings suggested. Efforts are made to find quick and reliable solutions that can minimize additional processing time during batch image generations.

- **CHKPT Model Guidance**: Queries about guides to CHKPT models were shared due to the vast quantity available, seeking information on which models are best for specific purposes. Suggestions for specific models like ponyxl, dreamshaperxl, juggernautxl, and zavychroma were given as part of a stable diffusion checkpoint "starter pack."

- **Model Performance Discussions**: Conversations spanned various topics including the speed of AI development, ethical considerations surrounding AI training with professional artwork, and the potential memory requirements for upcoming Stable Diffusion versions. There was also banter and jokes, highlighting the lighter side of community engagement.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/smirk-teehee-pokemon-laugh-psyduck-gif-10282192216865852036">Smirk Teehee GIF - Smirk Teehee Pokemon - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/anime-help-tears-cry-sad-gif-17104681">Anime Help GIF - Anime Help Tears - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://civitai.com/models/367412">Geeky Ghost Vid2Vid Organized v1 - v4.0 | Stable Diffusion Workflows | Civitai</a>: This workflow is designed for advanced video processing, incorporating various techniques such as style transfer, motion analysis, depth estimation...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1brcntc/so_openai_never_had_any_magic_sauce/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://feedback.civitai.com/p/pass-or-fail-a-simple-and-controlled-model-ranking-feature>)">Feedback - Civitai</a>: Give Civitai feedback on how they could improve their product.</li><li><a href="https://huggingface.co/ostris/ip-composition-adapter/tree/main">ostris/ip-composition-adapter at main</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=0D6opXdC7ew">epicRealism - THIS is the Model you WANT!!!!</a>: epicRealism  is the 1.5 Model with perfect realism. Highly detailed Skin. Athentic images. Super realistic Faces. Natural Light. Also perfect for less clothi...</li><li><a href="https://civitai.com/models/229002">ICBINP XL - v4 | Stable Diffusion Checkpoint | Civitai</a>: If you do like this work, consider buying me a coffee :) Use this model for free on Stable Horde The long awaited followup to ICBINP, this model is...</li><li><a href="https://youtu.be/B-Wd-Q3F8KM">The Count Censored</a>: My music facebook page https://www.facebook.com/pencilfacemusic</li><li><a href="https://civitai.com/models/277058/epicrealism-xl">epiCRealism XL - V5-Ultimate | Stable Diffusion Checkpoint | Civitai</a>: Update2: back on Track, i refined from V1 - probably last Version for SDXL until SD3 Keep this Secret for yourself SDXL magic 🧙‍♂️ happens with: 30 St...</li><li><a href="https://youtube.com/shorts/C5cIib7hiK8?si=z8FW2_UFwgZEn0LK">1万年かけて成長するフリーレン(上半身)/Frieren growing over 10,000 years(upper body) #葬送のフリーレン #frieren #アニメ</a>: no description found</li><li><a href="https://github.com/jhc13/taggui/releases">Releases · jhc13/taggui</a>: Tag manager and captioner for image datasets. Contribute to jhc13/taggui development by creating an account on GitHub.</li><li><a href="https://civitai.com/models/4201/realistic-vision-v60-b1">Realistic Vision V6.0 B1 - V5.1 (VAE) | Stable Diffusion Checkpoint | Civitai</a>: I recommend checking out the information about Realistic Vision V6.0 B1 on Hugging Face. This model is available on Mage.Space (main sponsor) and S...</li><li><a href="https://asia.nikkei.com/Business/Technology/Japan-panel-pushes-to-shield-copyrighted-work-from-AI-training">Japan panel pushes to shield copyrighted work from AI training</a>: Unauthorized use of protected material could be violation, draft document says</li><li><a href="https://civitai.com/models/15003/cyberrealistic">CyberRealistic - v4.2 | Stable Diffusion Checkpoint | Civitai</a>: Want to buy me coffee? ( Buy a cup ) The optional CyberRealistic negatives used in the samples check huggingface SDXL version of CyberRealistic Int...</li><li><a href="https://civitai.com/models/312530/cyberrealistic-xl">CyberRealistic XL - v1.1 (VAE) | Stable Diffusion Checkpoint | Civitai</a>: Want to buy me coffee? ( Buy a cup ) CyberRealistic XL It took a while, but here is the SDXL version of CyberRealistic. The criteria for this model...
</li>
</ul>

</div>
  

---



**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1224287410691768350)** (4 messages): 

- **April Fools' Enthusiasm**: A member expressed a desire to create a gguf that repeats "April fools" and to share it as a mock GPT4 named q0_k_xl.
- **DBRX New AI Unveiled**: A video was shared titled "DBRX: A New State-of-the-Art Open LLM," which introduces DBRX as an open, general-purpose language model claiming to set new records on benchmarks. Watch the video [here](https://www.youtube.com/watch?v=dqFvOqC43rQ).

**Link mentioned**: <a href="https://www.youtube.com/watch?v=dqFvOqC43rQ">DBRX: A New State-of-the-Art Open LLM</a>: Introducing DBRX, an open, general-purpose LLM created by Databricks. Across a range of standard benchmarks, DBRX sets a new state-of-the-art for established...

  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1224265625526861915)** (18 messages🔥): 

- **Surprising Error Rates for HF Transformers**: Members shared surprise over the high error rates with HF transformers and mentioned using **BetterTransformer** with Distil whisper v3 large. The intention to consider **WhisperX** for future projects was expressed.
  
- **Anticipating the End of Web Silos**: A link was shared discussing impending transformations of the web, including a shift in search engines from information retrieval to a more predictive and personalized approach, the need for revenue strategy overhauls due to interconnected digital ecosystems, and potential obsolescence of dedicated Web User Interfaces (UIs). Read about the impending digital transformations in [Transforming the Web](https://www.f5.com/company/blog/transforming-the-web-the-end-of-silos).

- **New Apple Paper on Reference Resolution**: Discussion about a new Apple paper suggesting an 80M parameter model outperforms GPT-3.5 and a 250M parameter model beats GPT-4 in most benchmarks for reference resolution. The conversation noted the critical importance of reference resolution in the effectiveness of AI agents. Catch up on the Apple paper [here](https://arxiv.org/pdf/2403.20329.pdf).

- **Reference Resolution's Role in AI Accuracy**: Continuing on the subject of reference resolution, it was highlighted that this could be a significant factor in AI agents committing errors during task execution. Further dialogue involved an interpretation that 80M parameter models performed surprisingly well on unseen tasks, potentially due to a high margin of error across models or similarities in accuracy.

- **Latest Reference from Twitter**: A Twitter post highlighting newly released information was suggested for comparison in the ongoing discussions. It could be added as a further point of reference for model comparisons. Check out the new information [on this Twitter post](https://vxtwitter.com/maxaljadery/status/1775196809893478797?s=46&t=stOPrwZiN_fxSK0RuC8Flg).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2403.20329">ReALM: Reference Resolution As Language Modeling</a>: Reference resolution is an important problem, one that is essential to understand and successfully handle context of different kinds. This context includes both previous turns and context that pertain...</li><li><a href="https://www.f5.com/company/blog/transforming-the-web-the-end-of-silos">Transforming the Web: The End of Silos</a>: The way we use the internet is about to change in a big way. The shift towards a more unified and efficient Web navigation method is a big leap from the traditional siloed Web browsing we’ve gotten us...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1224275749834326017)** (104 messages🔥🔥): 

- **LlamaFile's Speed Leap**: [Justine Tunney announced](https://x.com/justinetunney/status/1774621341473489024) that LlamaFile is now 1.3x - 5x faster than llama.cpp on CPU for many prompt/image evaluation use cases, which was detailed in their matrix multiplication blog post [here](https://justine.lol/matmul/).
- **Exploring PII Masking**: A member shared a trending dataset on Hugging Face for PII (personally identifiable information) masking, which can be accessed [here](https://huggingface.co/datasets/ai4privacy/pii-masking-300k), while discussing the challenges of handling personal-sensitive data.
- **Debate on the Direction of Hermes Project**: There's an ongoing discussion about the Hermes project's future direction, with some members highlighting the importance of the model handling PII well, and a reference to similar capabilities in smaller models like open-llama-3b.
- **Concern Over Instaban on Anthropic**: A member expressed confusion over being instantly banned from the Claude AI site after logging in and out too quickly, speculating it may have appeared suspicious to the system.
- **Discussion on Language Model Programs**: An extensive debate ensued on the potential of using solvers with DSPy to create highly compressed foundational models optimized for "Logical Coherence," including a mention of Stanford researchers interested in "The next step in LLM efficiency," all sparked by a video from Omar Khattab on DSPy which can be watched [here](https://youtu.be/Y94tw4eDHW0?t=549).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2403.19928">DiJiang: Efficient Large Language Models through Compact Kernelization</a>: In an effort to reduce the computational load of Transformers, research on linear attention has gained significant momentum. However, the improvement strategies for attention mechanisms typically nece...</li><li><a href="https://arxiv.org/abs/2009.03393">Generative Language Modeling for Automated Theorem Proving</a>: We explore the application of transformer-based language models to automated theorem proving. This work is motivated by the possibility that a major limitation of automated theorem provers compared to...</li><li><a href="https://x.com/p00ssh/status/1775185708887539864?s=20">Tweet from poosh (e/λcc) (@p00ssh)</a>: attention is what you need, anon</li><li><a href="https://arxiv.org/abs/1806.00608">GamePad: A Learning Environment for Theorem Proving</a>: In this paper, we introduce a system called GamePad that can be used to explore the application of machine learning methods to theorem proving in the Coq proof assistant. Interactive theorem provers s...</li><li><a href="https://huggingface.co/datasets?sort=trending&search=pii">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://x.com/justinetunney/status/1774621341473489024">Tweet from Justine Tunney (@JustineTunney)</a>: I just made llamafile 1.3x - 5x faster than llama.cpp on CPU for many prompt / image evaluation use cases and hardware. https://justine.lol/matmul/</li><li><a href="https://arxiv.org/abs/2012.14474">Paraconsistent Foundations for Probabilistic Reasoning, Programming and Concept Formation</a>: It is argued that 4-valued paraconsistent truth values (called here &#34;p-bits&#34;) can serve as a conceptual, mathematical and practical foundation for highly AI-relevant forms of probabilistic log...</li><li><a href="https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf">unilm/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf at master · microsoft/unilm</a>: Large-scale Self-supervised Pre-training Across Tasks, Languages, and Modalities - microsoft/unilm</li><li><a href="https://www.youtube.com/watch?v=ZYf9V2fSFwU">AI Pioneer Shows The Power of AI AGENTS - &quot;The Future Is Agentic&quot;</a>: Andrew Ng, Google Brain, and Coursera founder discusses agents&#39; power and how to use them. Join My Newsletter for Regular AI Updates 👇🏼https://www.matthewb...</li><li><a href="https://github.com/YuchuanTian/DiJiang">GitHub - YuchuanTian/DiJiang: The official implementation of &quot;DiJiang: Efficient Large Language Models through Compact Kernelization&quot;, a novel DCT-based linear attention mechanism.</a>: The official implementation of &quot;DiJiang: Efficient Large Language Models through Compact Kernelization&quot;, a novel DCT-based linear attention mechanism. - YuchuanTian/DiJiang</li><li><a href="https://huggingface.co/shisa-ai/shisa-jamba-v1-checkpoint-4228">shisa-ai/shisa-jamba-v1-checkpoint-4228 · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/unchained-foxx-silent-django-gif-4956511">Unchained Foxx GIF - Unchained Foxx Silent - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://youtu.be/Y94tw4eDHW0?si=cbH5-LV2dkXkkb0_&t=549">Programming Foundation Models with DSPy / Multivector Semantic Search with ColBERT - Omar Khattab</a>: Omar Khattab is a PhD Candidate at Stanford University and an Apple Scholar in AI/ML. In this conversation, Omar explains how to program foundation model pip...</li><li><a href="https://app.wordware.ai/r/5cad80d6-e0bf-4f37-b147-5b44b2273038">Wordware - prompt extraction analysis</a>: text -&gt; prompt
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1224368184103272539)** (37 messages🔥): 

- **LLM Implementation Inquiry**: A newcomer inquired about where to start learning to implement open-source LLMs like `llama 2` into a website as a fine-tuned chatbot. No specific resources or solutions were provided in response.
- **Quantized LLM Training Quandary**: In discussing the training of a 110m parameter model using Hugging Face's Llama architecture, a member expressed concerns over GPU memory access inefficiencies, noting the GPU spends ~97% of the time accessing memory. A sophisticated *BitLinear implementation* was shared, and the possibility that this issue could be caused by model size or next-token prediction demands was suggested.
- **Fine-Tuning Frustration**: A user shared a *configuration file* for fine-tuning `NousResearch/Hermes-2-Pro-Mistral-7B` on a custom domain benchmark but found accuracy decreased post-fine-tuning. Items in the config, like `lora_r`, `lora_alpha`, and `sequence_len` were outlined, but no diagnosis was given for the observed accuracy drop.
- **Agent Inclusivity in Sample Time Reward Models**: A conversation was initiated about the use of reward models during sample time, noting that better outcomes could sometimes be achieved through *best-of-n sampling* where a reward-tuned model doesn't always generate the optimal answer according to its own standards.
- **Supervised Fine-Tuning Scripts and Tokenizer Configurations Shared**: Links to resources for fine-tuning the LLMs `OLMo-Bitnet-1B` and `NousResearch/Hermes-2-Pro-Mistral-7B` were provided, including discussions on handling special tokens and tokenizer configurations. A specific technique ensured tokenizer configs contained necessary tokens for primary functionality, and PRs for these configurations were merged.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B/">NousResearch/Hermes-2-Pro-Mistral-7B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/OLMo-Bitnet-1B/tree/main">NousResearch/OLMo-Bitnet-1B at main</a>: no description found</li><li><a href="https://wandb.ai/emozilla/olmo/runs/wwx8x2o3)">emozilla</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://hastebin.com/share/avuredixuq.yaml">Hastebin</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B/discussions/12">NousResearch/Hermes-2-Pro-Mistral-7B · Add Padding Tokens</a>: no description found
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1224260834235256874)** (4 messages): 

- **Traffic Signal Images for Vision Models**: A dataset containing [traffic signal images](https://huggingface.co/datasets/Sayali9141/traffic_signal_images) was shared, considered instrumental for structured output and tool-use with vision models. Note that the Hugging Face dataset viewer does not support this dataset due to the execution of arbitrary python code.
- **Dataset Development Interest**: Members expressed interest in building a dataset based on the provided traffic signal images link. However, no specific progress or details about the dataset construction were shared.
- **Acknowledging Dataset Utility**: A brief acknowledgment was made regarding the potential usefulness of the proposed traffic signal images dataset.

**Link mentioned**: <a href="https://huggingface.co/datasets/Sayali9141/traffic_signal_images">Sayali9141/traffic_signal_images · Datasets at Hugging Face</a>: no description found

  

---


**Nous Research AI ▷ #[bittensor-finetune-subnet](https://discord.com/channels/1053877538025386074/1213221029359657021/1224713092542365716)** (2 messages): 

- **Upload to Chain Fails Due to Hugging Face Metadata**: A member has reported an issue with uploading to the chain, where **Hugging Face** unexpectedly adds a `safetensors.sharded = true/false` key to the model metadata. This key is not accepted by the Hugging Face Python library's `hf_api.py` method, causing a failure to load **ModelInfo** necessary for pushing and validating models.
- **Seeking Solutions in Discord**: The member experiencing the upload issue inquired if others are facing the same problem and asked for possible workarounds. Another member provided a link to a Discord message, but no additional context is available in this summary.
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1224583539249385523)** (5 messages): 

- **Exploring the Scratchpad's Utility**: An example was shared showing the use of `<scratchpad>` to gather evidence for RAG (retrieval-augmented generation) from the Claude prompt-engineering guide.
- **Debating Scratchpad Versus Full Context**: A member questioned if using a scratchpad is better than including the full context in the prompt, to which another member responded negatively, suggesting that having the right context is still the preferred method.
- **The Role of Scratchpads in Workflows**: Another member illustrated how scratchpads can be helpful by mentioning their workflow, which includes `notes` similar to a scratchpad intended for user interaction.
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1224296713305591879)** (110 messages🔥🔥): 

- **Envisioning File Uploads & Local Cache**: Members discussed the value of **file uploading** as a feature for **WorldSim**, suggesting it could enhance efficiency by running pre-written scripts. Another suggestion was maintaining a **local cache** to simulate file system navigation and the concept of a "Generative Virtual Machine (GVM)" for dumping files to maintain consistency.

- **WorldSim Easter Egg Uncovered**: Users discovered an **April Fools' Day easter egg** within WorldSim. The easter egg is triggered when discussing morality and adds a playful twist to interactions.

- **Competitive WorldSim Challenges**: A member proposed a concept akin to "LLM Coliseum," where a similar competitive benchmark involving **WorldSim** tasks could test LLMs against each other in a competitive setting, potentially even with an LLM acting as a judge for the competitions.

- **WorldSim Future Features and Roadmap Speculation**: There was discussion on the potential for future **WorldSim** features, such as a **competitive leaderboard**, **text to video integration**, and input/output capabilities. Users express a desire for a publicly available **roadmap or updates** for transparency on upcoming developments.

- **WorldSim as a Platform for AI Battles**: Ideas were shared about **AI battles** within WorldSim, where a "constellation of BBS's" could run games from various eras, with an emergent theme of unifying opposites and philosophical dimensions, including treasure hunts and alchemical lore.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://worldsim.nousresearch.com/">world_sim</a>: no description found</li><li><a href="https://worldsim-web.vercel.app/">world_sim</a>: no description found</li><li><a href="https://arxiv.org/abs/2402.19459">Anomalous contribution to galactic rotation curves due to stochastic spacetime</a>: We consider a proposed alternative to quantum gravity, in which the spacetime metric is treated as classical, even while matter fields remain quantum. Consistency of the theory necessarily requires th...</li><li><a href="https://en.wikipedia.org/wiki/Core_War">Core War - Wikipedia</a>: no description found</li><li><a href="https://lostpedia.fandom.com/wiki/Hanso_Foundation">Hanso Foundation</a>: The Hanso Foundation is an organization founded by Alvar Hanso, whose aim was to &quot;reach out to a better tomorrow&quot; by researching ways to preserve human life and promote well-being. It was es...</li><li><a href="https://pastebin.com/DucD5J7N">div {  animation: none;  animation-play-state: paused;  line-height: 1.25; - Pastebin.com</a>: Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.</li><li><a href="https://tenor.com/view/her-theodore-joaquin-phoenix-scarlett-johannson-samantha-gif-5203383">Her Theodore GIF - Her Theodore Joaquin Phoenix - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://youtu.be/rSKMYc1CQHE?si=I6TPYRIMX9k6DUVE">Coding Adventure: Simulating Fluids</a>: Let&#39;s try to convince a bunch of particles to behave (at least somewhat) like water.Written in C# and HLSL, and running inside the Unity engine.Source code:h...
</li>
</ul>

</div>
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1224342691660304476)** (244 messages🔥🔥): 

- **GPU Stability Nightmares**: Users shared experiences of severe system instability when using AMD GPUs, highlighting issues such as memory leaks and non-recoverable errors after running benchmarks and stressing the GPUs. Errors like `"amdgpu: failed to allocate BO for amdkfd"` and `"amdgpu: Failed to create process device data"` were reported, indicating hardware/firmware level issues.

- **AMD's Invitation Viewed Skeptically**: One user received an invitation to AMD's Vanguard program after bringing attention to GPU reset issues on AMD's subreddit. However, George Hotz and others express doubt about AMD's commitment to resolving underlying problems, emphasizing actions over words and stressing the importance of open-source documentation and sources for meaningful progress.

- **Perception of AMD within Tinygrad Community**: There is palpable frustration with AMD's approach to software and drivers within the tinygrad community. Hotz predicts future regrets for large investments in AMD's MI300X due to poor testing and cultural resistance to modern software practices.

- **Approaches to Dealing with GPU Resets**: Discussions surround workaround strategies like PCIe power cycling and redundancy to tackle the inability to reset AMD cards after crashes. Various anecdotes and potential solutions like "PCIe hotswap" or "GPUs in RAID 1" are humorously proposed, but ultimately signify the gravity of the issue.

- **Reflections on Software and Firmware Practices**: There's an ongoing dialogue about the need for a fundamental cultural shift in AMD regarding their software and firmware practices. George Hotz speculates that with the right management and testing protocols, such as CI and fuzzing, it may be possible to replace the current complex firmware with something simpler and more robust.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://looking-glass.io)">no title found</a>: no description found</li><li><a href="https://www.phoronix.com/news/AMD-Bridgman-Retires">Tweet from AMD&#039;s Longtime Open-Source Linux Graphics Driver Advocate Retires - Phoronix</a>: no description found</li><li><a href="https://forum.level1techs.com/t/radeon-7900xt-reset-bug/196270/4">Radeon 7900XT, reset bug</a>: With AMD the essence is that you either get a card that resets or you get one that does not. There is not way to be sure since the design is proprietary and AMD still has not managed to fix the reset ...</li><li><a href="https://www.reddit.com/r/Amd/comments/1bsjm5a/letter_to_amd_ongoing_amd/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/Amd/comments/1bsjm5">Reddit - Dive into anything</a>: no description found</li><li><a href="https://mastodon.gamedev.place/@NOTimothyLottes/112190982123087000">NOTimothyLottes (@NOTimothyLottes@mastodon.gamedev.place)</a>: Anyway it looks like I&#39;m back to a compiler perf bug that is impossible to workaround and too horrible to ignore.  Wave-coherent (should be fast) dynamic descriptor choice acts like a batch break ...</li><li><a href="https://github.com/tinygrad/tinygrad/blob/bec2aaf404aa3bdc569330a8d66c6678f8bfc459/examples/beautiful_mnist_multigpu.py">tinygrad/examples/beautiful_mnist_multigpu.py at bec2aaf404aa3bdc569330a8d66c6678f8bfc459 · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad</li><li><a href="https://github.com/geohot/7900xtx/blob/master/docs/MEC.md">7900xtx/docs/MEC.md at master · geohot/7900xtx</a>: Contribute to geohot/7900xtx development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1224265332848328765)** (31 messages🔥): 

- **Linear uOps Exposed**: A member shared a [write-up on linear uops](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/uops.md) to assist others in understanding the intermediate representation used in tinygrad. They note that while the content is based on personal study notes, feedback and suggestions are welcomed.

- **Command Queue Clarification**: There was a discussion around the new command queue implementation in tinygrad. A [tutorial has been shared](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/commandqueue.md) that explains changes following a recent merge, indicating that the command queue is a replacement for the "run_schedule" function.

- **Test Code Puzzles**: A Pull Request to tinygrad raised questions about commented-out unittest code and backend checks. It was [clarified and fixed in PR #4034](https://github.com/tinygrad/tinygrad/pull/4034), ensuring tests could be run with different backends like CLANG and OpenCL on Intel GPUs.

- **ShapeTracker Specs Scrutinized**: A specification for a high-level shapetracker was briefly discussed, touching upon the topic of mergeability and expressing strides for shapes in mathematical notation (Z^n), which can be negative.

- **Findings on Jitted Functions**: Members were trying to understand why jitted functions didn't appear in the command queue logs, discussing operational aspects of tinygrad's command queue and how it affects the execution of scheduled items.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/backends.md">tinygrad-notes/backends.md at main · mesozoic-egg/tinygrad-notes</a>: Contribute to mesozoic-egg/tinygrad-notes development by creating an account on GitHub.</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/commandqueue.md">tinygrad-notes/commandqueue.md at main · mesozoic-egg/tinygrad-notes</a>: Contribute to mesozoic-egg/tinygrad-notes development by creating an account on GitHub.</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/uops.md">tinygrad-notes/uops.md at main · mesozoic-egg/tinygrad-notes</a>: Contribute to mesozoic-egg/tinygrad-notes development by creating an account on GitHub.</li><li><a href="https://github.com/tinygrad/tinygrad/pull/4034">re-enable has_local check for linearizer test by thedanhoffman · Pull Request #4034 · tinygrad/tinygrad</a>: Discussed in the learning tinygrad channel, conclusion was this is probably an oversight from a previous PR. Hit by running CLANG=1 python3 -m pytest test/test_linearizer_overflows.py</li><li><a href="https://github.com/tinygrad/tinygrad/pull/3623/files">bring ptx back by geohot · Pull Request #3623 · tinygrad/tinygrad</a>: no description found
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1224264931126415390)** (89 messages🔥🔥): 

- **LM Studio Error Troubles**: Members are experiencing errors with LM Studio, such as unknown exceptions during inferencing and crashes after a few messages with Quantized models, specifically citing issues with *estopian maid 13B q4* on an RTX 3060 GPU.

- **Seeking Voice Understanding Models in LM Studio**: A member inquired about models that understand voice directly, to which a response clarified that **LM Studio requires a separate tool** for speech-to-text since there's no TTS (Text-to-Speech) or STT (Speech-to-Text) functionality built in, mentioning *whisper.cpp* as an example.

- **Optimizing Context Length and Model Selection for Development**: Discussions about how to manage context length in LM Studio and which models are best for software development surfaced, echoing that **best practices and model choice can vary based on the user's hardware**.

- **LM Studio Settings and Performance Queries**: Users are seeking tips on how to increase performance, with advice given on how to set the model to use more GPU via the "GPU Offload" option in settings, and confirming that **LM Studio can load locally stored GGUF files**.

- **Updates, Downgrades, and Usage Help**: Individuals are navigating issues with model loading, seeking ways to downgrade to a previous stable version of LM Studio, and looking for specific and potentially nonexistent features such as support for PKL models or running embedding models—both of which aren't supported at this time.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/docs/welcome">Welcome | LM Studio</a>: LM Studio is a desktop application for running local LLMs on your computer.</li><li><a href="https://rentry.org/LMSTudioFAQ#how-do-i-use-already-downloaded-gguf-models-in-lmstudio">The unofficial LMStudio FAQ!</a>: Welcome to the unofficial LMStudio FAQ. Here you will find answers to the most commonly asked questions that we get on the LMStudio Discord. (This FAQ is community managed).  LMStudio is a free closed...</li><li><a href="https://github.com/moritztng/fltr">GitHub - moritztng/fltr: Like grep but for natural language questions. Based on Mistral 7B or Mixtral 8x7B.</a>: Like grep but for natural language questions. Based on Mistral 7B or Mixtral 8x7B. - moritztng/fltr</li><li><a href="https://github.com/Mozilla-Ocho/llamafile/blob/main/llamafile/sgemm.cpp">llamafile/llamafile/sgemm.cpp at main · Mozilla-Ocho/llamafile</a>: Distribute and run LLMs with a single file. Contribute to Mozilla-Ocho/llamafile development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1224270193015132161)** (44 messages🔥): 

- **LM Studio Lacks File Input Features**: Members are looking for a way to provide files to models in LM Studio, similar to OpenAI's API, but **LM Studio currently lacks this feature**. It has been noted that this capability is on their ToDo list.
- **No Local Qwen MoE support**: [Support for Qwen MoE in llama.cpp](https://github.com/ggerganov/llama.cpp/pull/6074) is a work in progress according to a shared GitHub pull request.
- **Privacy-Conscious Local LLM Solution Proposed**: It's suggested that combining LM Studio with AnythingLLM can provide a private Local LLM + RAG chatbot experience, even though LM Studio itself lacks built-in document support.
- **7B Models Discussed for NSFW Content**: Members are discussing the best performing uncensored 7b models, like **Nous-Hermes 2 Mistral DPO** and **Nous-Hermes-2-SOLAR-10.7B**, with a focus on how to handle NSFW content.
- **Tech Issues with Model Downloads and Execution**: Some users reported difficulty downloading and running models locally, encountering silent failures potentially due to the **lack of proxy support** or trying to load large models without a GPU.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.marktechpost.com/2024/03/31/mistral-ai-releases-mistral-7b-v0-2-a-groundbreaking-open-source-language-model/">no title found</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=-Rs8-M-xBFI&ab_channel=TimCarambat">Stop paying for ChatGPT with these two tools | LMStudio x AnythingLLM</a>: In this video, we are installing two user-friendly tools that make downloading, running, and managing a powerful local LLM to replace ChatGPT. Seriously.Toda...</li><li><a href="https://myanimelist.net/">MyAnimeList.net - Anime and Manga Database and Community </a>: Welcome to MyAnimeList, the world&#039;s most active online anime and manga community and database. Join the online community, create your anime and manga list, read reviews, explore the forums, follo...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6074">Add qwen2moe by simonJJJ · Pull Request #6074 · ggerganov/llama.cpp</a>: This PR adds the support of codes for the coming Qwen2 MoE models hf. I changed several macro values to support the 60 experts setting. @ggerganov
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1224632728834539615)** (1 messages): 

- **Regeneration Overwrites Previous Output**: A member expressed concerns about **continuing generation** overwriting the existing assistant output instead of creating a new entry as it did before. They hope for a rollback to former functionality that split the text on every new generation.
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1224357313356365925)** (80 messages🔥🔥): 

- **GPU Support Confusion Cleared**: Members clarified that **SLI is not required** for running multiple GPUs. [One user's setup indicates](https://www.techpowerup.com/gpu-specs/quadro-rtx-8000.c3306) that the system detects the combined VRAM, but actual VRAM utilization is limited to that of the primary GPU. 
- **NVIDIA Tesla P40 Performance Insights**: A shared [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/17zpr2o/nvidia_tesla_p40_performs_amazingly_well_for/) details that a dual **Tesla P40** configuration can run a 70B parameter model with *approximate* performance of 3-4 tokens/second. 
- **Resourceful Dual GPU Setup**: One contributor described a budget-wise build using **Triple P40 GPUs**, revealing that 70B models run about 4 tokens/second at 8192 context length. The DIY build recommendations came from a [Mikubox Triple-P40 guide](https://rentry.org/Mikubox-Triple-P40).
- **VRAM Over Compute Power Debate**: Discussions surfaced comparing the value of VRAM over compute power, with one user considering replacing a **4090 GPU** with **P40s** for their higher VRAM, despite the 4090's superior performance.
- **Balancing Budget and Performance**: Members joke about the trade-offs between affordable hardware and desired performance, comparing it to choosing low-end car upgrades that get the job done but compromise on comfort and quality.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://rentry.org/Mikubox-Triple-P40">Mikubox Triple-P40 build</a>: Dell T7910 &quot;barebones&quot; off ebay which includes the heatsinks. I recommend the &quot;digitalmind2000&quot; seller as they foam-in-place so the workstation arrives undamaged. Your choice of Xe...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/17zpr2o/nvidia_tesla_p40_performs_amazingly_well_for/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.techpowerup.com/gpu-specs/quadro-rtx-8000.c3306#:~:text=The%20card%20also%20has%2072,MHz%20(14%20Gbps%20effective).">NVIDIA Quadro RTX 8000 Specs</a>: NVIDIA TU102, 1770 MHz, 4608 Cores, 288 TMUs, 96 ROPs, 49152 MB GDDR6, 1750 MHz, 384 bit
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1224718735068364962)** (1 messages): 

- **Autogen Troubles with Token Limits**: A member encountered an issue where *Autogen* is only generating about 2 tokens at a time and incorrectly assuming the agent is done. They're seeking advice on whether special configuration is necessary to make **LM Studio** work effectively with *Autogen*.
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1224273178365394965)** (54 messages🔥): 

- **Model Distillation and Claude3 Haiku Performance**: Users discussed the distillation of larger models like Claude3 Opus into smaller, more efficient ones like Claude3 Haiku. Some are impressed by Haiku's performance, considering it might suffice for many use cases originally thought to require GPT-4.
  
- **Residual Block Discussion Sparks Technical Debate**: A technical conversation emerged around why residual blocks in neural architectures often use two linear layers. Users explained that two layers with non-linearity increase expressiveness and allow for flexible parameterization.

- **AI Mathematical Olympiad Engagement**: Mention of the [Kaggle AI Mathematical Olympiad competition](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/overview) sparked a suggestion that the EleutherAI community could form groups to compete. Compute grants for "AI in science" could potentially support such initiatives.

- **Resource Sharing and Project Joining**: New members introduced themselves, sharing their research interests in fields like alignment, privacy, fine-tuning of large language models, and autoformalisation. They are looking to contribute to projects and learn from the community.

- **Building a Credible Benchmarking Dataset**: One user inquired about the necessity of a peer-reviewed paper when creating a benchmarking dataset for a new language, seeking advice on establishing credibility for their dataset.

**Link mentioned**: <a href="https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/overview">AI Mathematical Olympiad - Progress Prize 1 | Kaggle</a>: no description found

  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1224272095111024660)** (111 messages🔥🔥): 

- **Deciphering Google's Search Infrastructure**: Discussion revolved around Google's ability to embed the entire web in RAM for rapid indexing and retrieval. Participants talked about the potential infrastructure, with comments stating that **Google may use a distributed version of FAISS and operates primarily with data stored in RAM** to ensure fast response times, essential to their operations.

- **Musing Over Google's Approach to Programming**: In a further conversation about Google's technical strategies, it was mentioned that Google isn't afraid to utilize "bad" programming constructs like global variables or **`goto`** if they serve a purpose. There's also reference to utilizing thread local storage to streamline context handling in remote procedure calls.

- **Discussing the Limits of Text Indexing**: Questions arose around how Google handles obscure text search queries that necessitate exact matches, leading to an explanation of Google's use of **inverted indexes**. Different indexing strategies, such as full-text search and inverted indexes, were considered for handling wide-ranging and exact match queries efficiently.

- **Suspense Over New Research Papers**: There was anticipation for new papers being shared, with specific interest in the **robustness of safety filters for LLMs**. A link to recent research was provided, with a nod to the essential nature of continued exploration in the field, involving safeguarding against reverse-engineering or misusing language models.

- **Nonchalant Revelation of Sleek OSS Agent**: A link was shared highlighting an open-source software agent from Princeton NLP named **SWE-Agent**, which claims to perform on par with proprietary agents like Devin in software engineering tasks. This piqued interest as an example of cutting-edge open-source contributions to NLP.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://swe-agent.com/">SWE-Agent</a>: no description found</li><li><a href="http://arxiv.org/abs/2403.20327">Gecko: Versatile Text Embeddings Distilled from Large Language Models</a>: We present Gecko, a compact and versatile text embedding model. Gecko achieves strong retrieval performance by leveraging a key idea: distilling knowledge from large language models (LLMs) into a retr...</li><li><a href="https://en.wikipedia.org/wiki/Inverted_index?wprov=sfla1">Inverted index - Wikipedia</a>: no description found</li><li><a href="https://x.com/anthropicai/status/1775211248239464837?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">Tweet from Anthropic (@AnthropicAI)</a>: New Anthropic research paper: Many-shot jailbreaking.  We study a long-context jailbreaking technique that is effective on most large language models, including those developed by Anthropic and many o...</li><li><a href="https://x.com/blancheminerva/status/1774901289773584531?s=46">Tweet from Stella Biderman (@BlancheMinerva)</a>: It&#39;s known that finetuning can incidentally remove RLHF guards https://arxiv.org/abs/2310.03693. Can you solve this by including examples with refusals mixed into the data? Does it matter if those...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1224340286151458858)** (4 messages): 

- **Visualizing Sparse Autoencoder Features**: A new [visualization library](https://www.lesswrong.com/posts/nAhy6ZquNY7AD3RkD/sae-vis-announcement-post-1) for Sparse Autoencoder (SAE) features has been released and is praised for its usefulness in illuminating SAE feature structures in the context of AI alignment.
- **SAE Features Under the Spotlight**: A [forum post](https://www.lesswrong.com/posts/BK8AMsNHqFcdG8dvt/a-selection-of-randomly-selected-sae-features-1) is shared which examines whether Sparse Autoencoder features are revealing properties of the model or just reflecting the data distribution, acknowledging the complexity of this question and its relevance to AI alignment.
- **Abstract Animal Architecture?**: One member expressed bemusement at conceptualizing a house as something between 'concrete giraffe' and 'abstract giraffe', showing the complexity and sometimes whimsical nature of categorizing features in AI models.
- **Emojis Speak Louder Than Words**: A member responded to the perplexity over categorizing features with a shrug emoji, 🤷‍♂️, indicating either confusion or acceptance of the inherent ambiguities in AI interpretability.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/neelnanda5/status/1774463606656282806">Tweet from Neel Nanda (@NeelNanda5)</a>: Great visualisation library for Sparse Autoencoder features from @calsmcdougall! My team has already been finding it super useful, go check it out: https://www.lesswrong.com/posts/nAhy6ZquNY7AD3RkD/sa...</li><li><a href="https://www.lesswrong.com/posts/BK8AMsNHqFcdG8dvt/a-selection-of-randomly-selected-sae-features-1">A Selection of Randomly Selected SAE Features — LessWrong</a>: In this post, we interpret a small sample of Sparse Autoencoder features which reveal meaningful computational structure in the model that is clearly…
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1224362016030986413)** (18 messages🔥): 

- **Music Composition AI Review**: A [link to an arXiv paper](https://arxiv.org/abs/2402.15294) discusses current research on using **GANs** and **transformers** for music composition, focusing on style replication and transfer, with suggestions that future work could include metrics for text-to-music on a leaderboard, hinting at future directions for music AI evaluation.
- **Claude Benchmarked with lm-eval-harness?**: A user inquired if there's a repository of **lm-eval-harness** results for **Anthropic Claude** models, indicating a gap and potential area for sharing findings in API support and task adaptation.
- **ValueError Troubleshooting in lm-eval-harness**: A user reported persistent `ValueError` issues despite using **DEBUG verbosity** and was advised to share the YAML configuration file contents for further troubleshooting assistance.
- **Multilingual Capabilities Highlight Language Family Nuances**: A member machine-translated the **arc_challenge** to various languages and found that even models not trained in a specific language can perform well if it's in a related language family, suggesting the use of more generative evaluations to test language capabilities.
- **LM Evaluation Task List Compiled**: After discussion and debugging efforts, a user successfully compiled a list of **lm-evaluation-harness** tasks that require *generate until* functionality by searching GitHub, facilitating the discovery of suitable evaluation benchmarks.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2402.15294">A Survey of Music Generation in the Context of Interaction</a>: In recent years, machine learning, and in particular generative adversarial neural networks (GANs) and attention-based neural networks (transformers), have been successfully used to compose and genera...</li><li><a href="https://github.com/search?q=repo%3AEleutherAI%2Flm-evaluation-harness+output_type%3A+generate_until&type=code">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1224452465437442178)** (3 messages): 

- **Seeking Multimodal Vision Research Groups**: A member inquired about research groups or Discords focused on computer vision tasks like bias detection and emotion recognition for their Ph.D. proposal. They requested information on key papers and a broad view on bias in computer vision or multimodal applications.
- **LAION Suggested for Computer Vision**: A participant responded by suggesting **LAION** as a community, but noted it might not have the specific subfocus of bias detection and emotion recognition in computer vision.
- **Discover Prisma's Multimodal Group**: Another member mentioned the **Prisma multimodal group** as fairly active, though it does not specialize in the same subfocus sought after. They included a [link to a tweet](https://twitter.com/soniajoseph_/status/1767963443699790256) for reference.
  

---


**Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1224409336051273768)** (2 messages): 

- **Uneven Batch Sizes May Lead to Bottlenecks**: A user pointed out that while it is possible to hack **GPT-NeoX** to use uneven batch sizes, doing so could cause load imbalance. The largest batches would bottleneck the process, as the system waits for the GPUs handling them.
  

---



**OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1224408081409118298)** (1 messages): 

- **ChatGPT without the Wait**: OpenAI introduces the option to use [ChatGPT instantly, without needing to sign-up](https://openai.com/blog/start-using-chatgpt-instantly), aiming to make AI accessible to a broader audience. The tool is already used weekly by over 100 million people in 185 countries.

- **No Account, No Problem**: The company is rolling out this feature gradually, with a commitment to its mission of making AI tools like ChatGPT widely available.

- **Your Input Helps Improve AI**: Users' interactions with ChatGPT may be used to improve model performance, but there is an option to opt out of this in the settings—even without creating an account. More details on data usage can be found in the [Help Center](https://help.openai.com/en/articles/5722486-how-your-data-is-used-to-improve-model-performance).

**Link mentioned**: <a href="https://openai.com/blog/start-using-chatgpt-instantly">Start using ChatGPT instantly</a>: We’re making it easier for people to experience the benefits of AI without needing to sign up

  

---


**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1224279806548508672)** (95 messages🔥🔥): 

- **Nuances in Fine-Tuning**: Members discussed how using "\n" or line breaks could *affect fine-tuning language models*, noting **LLMs learn what they're given**, and that human-like formatting could be beneficial.
- **AI Song Recognition Falters**: An AI comparison shows **ChatGPT, Gemini, and Claude** failed to correctly list songs referred to in Laurent Voulzy's song "Rockollection", prompting a user to highlight the differing responses and **creative limits** of the AIs when offered the correct list.
- **Exploring AI-Generated Content's Originality**: There was a conversation around whether language models are just parroting content or if they can create original content. A **[study](https://arxiv.org/abs/2310.17567)** suggests AI can combine skills in ways not found in training data, highlighting that **AI may exhibit emergent behaviors** beyond simple reproduction.
- **AI Image Description Seekers**: Users discussed their search for AI tools for describing images, mentioning **Claude 3 Sonnet** and **Midjourney's** `/describe` command, while also observing limitations in the effectiveness of such tools and their availability.
- **AI App Compatibility Issues**: A user highlighted trouble accessing an AI app on a Samsung Galaxy Note 9, discussing potential compatibility or support issues, with suggestions to verify **Android and Play Store versions** as well as to use the mobile browser as a workaround.

**Link mentioned**: <a href="https://arxiv.org/abs/2310.17567">Skill-Mix: a Flexible and Expandable Family of Evaluations for AI models</a>: With LLMs shifting their role from statistical modeling of language to serving as general-purpose AI agents, how should LLM evaluations change? Arguably, a key ability of an AI agent is to flexibly co...

  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1224262417563910235)** (38 messages🔥): 

- **Exploring LLM's Reflective Capabilities**: Members discussed the possibility of prompting LLM to reflect internally, with one sharing insights that while LLM operates as a text predictor, structuring prompts effectively can avoid leaps in logic, with a reference to OpenAI's official guide on prompt engineering found [here](https://platform.openai.com/docs/guides/prompt-engineering/give-the-model-time-to-think).

- **April Fools' Technical Jokes?**: One member flagged a link presuming it to be an April Fools' joke, while another confirmed the functionality of the discussed feature as operational, despite not being able to provide a screenshot due to permission restrictions.

- **GPT Model Comparisons and Anticipation for GPT-5**: The conversation included a commentary on **GPT-4's** perceived superiority over **Opus**, despite its larger context window, and expressed anticipation for GPT-5, suggesting that once it includes better reasoning, code interpretation, and internet access, it could become a strong contender.

- **Server Stability Issues Garner User Attention**: Several members experienced issues with server stability, affecting their ability to log in and use services with one reporting a persistent "Error in input stream" for an extended period and soliciting known solutions.

- **Diverse AI Utilization and Development**: Users shared their developments and experiences with different AI services, including a custom-built GPT for finding persona details and a discussion about the availability of the new image editing feature in **DALL-E 3**, linking to the official instructions [here](https://help.openai.com/en/articles/9055440-editing-your-images-with-dall-e).
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1224447127976153118)** (7 messages): 

- **JSON Conundrums with PDFs**: A member inquired about the best approach to convert a pdf into a JSON object using GPT and pondered whether a schema or an open-ended prompt would be more effective. Another user suggested always sending a schema, although they noted the process can be quite random in effectiveness.

- **TOS Warning Over PDF Conversion**: It was pointed out that converting a PDF to JSON potentially violates the terms of service.

- **Call for Research Participants**: Anna, a Computer Science graduate conducting research at the American University of Armenia, invited ML engineers, content creators, prompt engineers, and other language model users for a 20-minute interview to discuss challenges associated with large language models.

- **Seeking Manager Replacement Prompts**: A member requested suggestions for effective prompts to replace managerial tasks, focusing on division of directives and performance planning for middle to C-suite management positions.
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1224447127976153118)** (7 messages): 

- **Choosing the Best JSON Approach**: Members are discussing the optimal way to extract JSON from a PDF using GPT. While one member has tried specifying the JSON schema, another is experimenting with a more open-ended approach to let GPT capture as much data as possible.

- **Random Results in Schema Enforcement**: In the process of converting documents to JSON, schema provision to GPT was addressed, with experiences indicating varying levels of success and an element of unpredictability in the results.

- **Understanding LLM Use-Cases**: Anna, a recent graduate in her research phase, is seeking to discuss with ML engineers and other professionals on their experiences and challenges in using large language models, asking interested parties to direct message or respond for a potential meetup.

- **Exploring Manager Replacement Prompts**: A member is seeking advice on good manager replacement prompts related to middle and C suite management tasks, such as dividing up directives and performance plans, hinting at potential advancements in automating managerial functions.
  

---



**LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1224391599367258227)** (1 messages): 

- **Dive into RAFT with LlamaIndex Webinar**: Join LlamaIndex's special webinar on **Retrieval-Augmented Fine-Tuning (RAFT)** featuring lead co-authors Tianjun Zhang and Shishir Patil for an in-depth session. Register for the event scheduled for this **Thursday at 9am PT** [here](https://lu.ma/v1bdat63).

- **Understanding RAFT via Upcoming Webinar**: The webinar will explore how RAFT combines the benefits of retrieval-augmented generation (RAG) and fine-tuning to improve language models' performance in domain-specific settings. Take part to learn from the experts behind this technique on **Thursday at 9am PT**.

- **Complementary Resources for RAFT Enthusiasts**: For additional context on the RAFT methodology, check out the dedicated RAFT [blog posts](https://gorilla.cs.berkeley.edu/blogs/) and access the full RAFT [paper](https://arxiv.org/pdf/2403.10131.pdf) to prepare for the webinar.

- **Generate Your Own RAFT Dataset**: Thanks to @ravithejads, you can now create a dataset for RAFT using the **RAFTDatasetPack** provided by LlamaIndex. Access the pack [here](https://llamahub.ai/l/llama-packs/llama-index-packs-raft-dataset?from=) and find the corresponding notebook on [GitHub](https://github.com/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-raft-dataset/examples/raft_dataset.ipynb).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lu.ma/v1bdat63">LlamaIndex Webinar: Retrieval-Augmented Fine-Tuning (RAFT) · Zoom · Luma</a>: RAFT - Retrieval Augmented Fine Tuning 🔥 Retrieval-Augmented Fine-Tuning (RAFT) by Zhang et al. is a new technique to fine-tune pre-trained LLMs for specific domain RAG...</li><li><a href="https://x.com/llama_index/status/1774814982322172077?s=20">Tweet from LlamaIndex 🦙 (@llama_index)</a>: New LlamaIndex Webinar 🚨 - come learn how to do retrieval-augmented fine-tuning (RAFT)!  Doing RAG is like taking an open-book exam without studying. It’s marginally better than a closed-book exam wh...
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1224374342494847108)** (4 messages): 

- **RAFT Webinar Alert**: *LlamaIndex* announced a [new webinar](https://twitter.com/llama_index/status/1774814982322172077) focused on **retrieval-augmented fine-tuning (RAFT)**, comparing it to taking an "open-book exam" and touting its benefits over traditional fine-tuning for language models.
- **Advanced PDF RAG Tutorial by Mesudarshan**: A tutorial by @mesudarshan demonstrates building advanced **PDF Retrieval-Augmented Generation (RAG)** using LlamaParse and local models was featured, highlighting extraction as a crucial step. Models from *GroqInc* and *FastEmbed* by @qdrant_engine are utilized in the process, as detailed in this [tweet](https://twitter.com/llama_index/status/1774832426000515100).
- **Step-by-Step RAG Building Guide**: @lambdaEranga's detailed **schematic diagram** for constructing RAG with local models including tools from @llama_index, @ollama, and @huggingface was shared, showing the utility of wrapping it all in a *Flask server* for development purposes. Here's the [guide](https://twitter.com/llama_index/status/1774950945488834684) on Twitter.
- **RankLLM for Advanced RAG Reranking**: **RankLLM** by @rpradeep42 et al., an open-source collection of **LLMs fine-tuned for reranking**, was recommended for those building advanced RAG systems. The significance of choosing the right reranker is emphasized, with the RankZephyr model getting a specific mention in the [tweet](https://twitter.com/llama_index/status/1775166279911186930).
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1224262634547707964)** (118 messages🔥🔥): 

- **Setting up LlamaIndex with OpenAI LLM**: A user experienced issues when specifying the organization id for OpenAI LLM. The correct way is to either set `openai.organization` in the code or pass the organization parameter when initializing `OpenAI` with `Settings.llm = OpenAI(organization="orgID",...)`.
- **Tackling Outdated LlamaIndex Documentation**: Users reported frustrations with outdated tutorials and broken links to GitHub files. They are seeking guidance for structured data and natural language SQL queries, and a transition from deprecated classes like `NLSQLTableQueryEngine` to newer ones like `SQLTableRetriever`.
- **Customizing Agents and Error Troubleshooting**: Discussions revolved around creating custom agents using `OpenAIAgent.from_tools`, handling errors such as `404 Not Found` with server endpoints like ollama, and implementing features in RAG such as weather queries with **WeatherReader**.
- **Dealing with Deprecated OpenAI Models**: A user encountered an error due to the `text-davinci-003` model being deprecated. It was suggested to replace this model with `gpt-3.5-turbo-instruct` in their GPTVectorStoreIndex setup.
- **Image Handling in PDF with LlamaParse**: Inquiries about handling images in PDF files were directed towards **LlamaParse**, a proprietary parsing tool capable of reading images, and alternative manual methods involving `unstructured` extraction and vector store summarization.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/phase1-collect-underpants-gnome-south-park-phase2-gif-22089237">Phase1 Collect Underpants GIF - Phase1 Collect Underpants Gnome - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.llamaindex.ai/blog/introducing-llamacloud-and-llamaparse-af8cedf9006b">Introducing LlamaCloud and LlamaParse — LlamaIndex, Data Framework for LLM Applications</a>: LlamaIndex is a simple, flexible data framework for connecting custom data sources to large language models (LLMs).</li><li><a href="https://github.com/run-llama/llama_index/tree/main/llama-index-core/llama_index/core/prompts">llama_index/llama-index-core/llama_index/core/prompts at main · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/readers/weather/?h=weather">Weather - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/customization/llms/SimpleIndexDemo-Huggingface_camel/?h=huggingface">HuggingFace LLM - Camel-5b - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom/?h=custom#example-using-a-custom-llm-model-advanced">Customizing LLMs - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/low_level/oss_ingestion_retrieval/">Building RAG from Scratch (Open-source only!) - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/prompts/prompt_mixin/#accessing-prompts">Accessing/Customizing Prompts within Higher-Level Modules - LlamaIndex</a>: no description found</li><li><a href="https://github.com/agronholm/sqlacodegen?tab=readme-ov-file">GitHub - agronholm/sqlacodegen: Automatic model code generator for SQLAlchemy</a>: Automatic model code generator for SQLAlchemy. Contribute to agronholm/sqlacodegen development by creating an account on GitHub.</li><li><a href="https://github.com/ollama/ollama?tab=readme-ov-file#rest-api">GitHub - ollama/ollama: Get up and running with Llama 2, Mistral, Gemma, and other large language models.</a>: Get up and running with Llama 2, Mistral, Gemma, and other large language models. - ollama/ollama</li><li><a href="https://github.com/run-llama/create-llama?tab=readme-ov-file#customizing-the-llm">GitHub - run-llama/create-llama: The easiest way to get started with LlamaIndex</a>: The easiest way to get started with LlamaIndex. Contribute to run-llama/create-llama development by creating an account on GitHub.</li><li><a href="https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/structured_data/">Structured Data - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/index_structs/struct_indices/SQLIndexDemo/#part-2-query-time-retrieval-of-tables-for-text-to-sql">Text-to-SQL Guide (Query Engine + Retriever) - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/chat_engine/chat_engine_personality/">Chat Engine with a Personality ✨ - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/8a8324008764a7fefb6f25b0e3aac81089590322/llama-index-legacy/llama_index/legacy/prompts/system.py#L4">llama_index/llama-index-legacy/llama_index/legacy/prompts/system.py at 8a8324008764a7fefb6f25b0e3aac81089590322 · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1224493601124388954)** (4 messages): 

- **Top Agent Simplification Issue**: A member reported a problem while building a *multi-document rag system* using Agents where the *top_agent* oversimplifies the question. For instance, a query about the expiration date of chocolate gets reduced to just "expiration date," leading to unsatisfactory search results.

- **Specific Query Simplification Examples**: The same member further illustrated the issue with another example, where the user asked for the expiration date of a fire extinguisher, but the agent only queried the *retrieval engine* with the term "expiration date."

- **IPEX-LLM and LlamaIndex Could Revolutionize Chat and Text Generation**: A link to a Medium article titled "Unlocking the Future of Text Generation and Chat with IPEX-LLM and LlamaIndex" was shared, discussing the potential impacts these tools could have on the future of text generation and chat applications. [Read the article here](https://medium.com/ai-advances/unlocking-the-future-of-text-generation-and-chat-with-ipex-llm-and-llamaindex-c98b84cdb3a2).

- **Tutorial Alert: Creating a RAG App with LlamaIndex**: A member shared a YouTube video tutorial that provides a step-by-step guide for building a simple RAG application using LlamaIndex, Pinecone, and Gemini Pro. Key processes such as scraping content, converting to vector embeddings, storing on Pinecone index, and using LlamaIndex to query Gemini Pro are covered. [Watch the tutorial here](https://youtu.be/B9mRMw0Jhfo).

**Link mentioned**: <a href="https://youtu.be/B9mRMw0Jhfo">How to build a RAG app using Gemini Pro, LlamaIndex (v0.10+), and Pinecone</a>: Let&#39;s talk about building a simple RAG app using LlamaIndex (v0.10+) Pinecone, and Google&#39;s Gemini Pro model. A step-by-step tutorial if you&#39;re just getting ...

  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1224332768759255173)** (109 messages🔥🔥): 

- **Handling Complex JSON with LangChain**: A user encountered difficulty when each JSON line created a Document instead of one Document with metadata for the full JSON. They inquired about a solution, but no follow-up was given, the original issue is described in the [JSON loader documentation](https://js.langchain.com/docs/integrations/document_loaders/file_loaders/json).
  
- **Increased Token Usage with Agents Using Tools**: A user noticed a 50% increase in token usage with agents using tools. It was clarified that tools retrieve and tokenize data, hence more tokens are used, and the system prompt is run once for inference assumptions but not every tool requires it.

- **Discussions on LangGraph and Structured Tool Validation**: Users discussed the potential to use a base model as the state in LangGraph and provided a Github [notebook example](https://github.com/langchain-ai/langgraph/blob/961ddd49ed498df7ffaa6f6d688f7214b883b34f/examples/state-model.ipynb). Also, instructions on using Pydantic's `BaseModel` and `Field` classes to validate a field in a StructuredTool in LangChain were shared from [Github issues](https://github.com/langchain-ai/langchain/issues/8066) and LangChain documentation.

- **Issues with Structured Output and Fine-tuning**: Users discussed problems related to obtaining structured output from a chain and the preservation of base knowledge after fine-tuning a model. A user suggested having two agents, a fine-tuned one and a regular GPT model, to maintain both specialized and general knowledge. No definitive solution to the structured output issue was posted within the conversation.

- **Mapping Content Between PDFs Using LangChain**: A user was attempting to map related content between PDFs using a RAG with RetrievalQA chain and was advised to try using vector embeddings to match paragraphs based on semantic content. They also asked about handling images in LangHub, encountering a deserialization error, but again, no solution was provided within the conversation.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://localhost:8000.>">no title found</a>: no description found</li><li><a href="https://python.langchain.com/docs/templates/openai-functions-agent#usage>).">openai-functions-agent | 🦜️🔗 Langchain</a>: This template creates an agent that uses OpenAI function calling to communicate its decisions on what actions to take.</li><li><a href="https://python.langchain.com/docs/guides/structured_output">[beta] Structured Output | 🦜️🔗 Langchain</a>: It is often crucial to have LLMs return structured output. This is</li><li><a href="https://js.langchain.com/docs/integrations/document_loaders/file_loaders/json">JSON files | 🦜️🔗 Langchain</a>: The JSON loader use JSON pointer to target keys in your JSON files you want to target.</li><li><a href="https://api.python.langchain.com/en/latest/chains/langchain.chains.structured_output.base.create_structured_output_runnable.html#langchain.chains.structured_output.base.create_structured_output_runnable.">langchain.chains.structured_output.base.create_structured_output_runnable &mdash; 🦜🔗 LangChain 0.1.14</a>: no description found</li><li><a href="https://github.com/langchain-ai/langgraph/blob/961ddd49ed498df7ffaa6f6d688f7214b883b34f/examples/state-model.ipynb">langgraph/examples/state-model.ipynb at 961ddd49ed498df7ffaa6f6d688f7214b883b34f · langchain-ai/langgraph</a>: Contribute to langchain-ai/langgraph development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/1358>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/8066>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://python.langchain.com/docs/use_cases/tool_use/tool_error_handling#tryexcept-tool-call>).">Tool error handling | 🦜️🔗 Langchain</a>: Using a model to invoke a tool has some obvious potential failure modes.</li><li><a href="https://github.com/langchain-ai/langchain/issues/13662>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langgraph/blob/961ddd49ed498df7ffaa6f6d688f7214b883b34">GitHub - langchain-ai/langgraph at 961ddd49ed498df7ffaa6f6d688f7214b883b34f</a>: Contribute to langchain-ai/langgraph development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1224453985591759021)** (5 messages): 

- **Langgraph Advocated for Conversational Bots**: A member praised **langgraph** for making it easy to implement cycles, highlighting its importance in creating advanced conversational taskbots. This feature sets it apart from other LLM app frameworks and will be further documented through community contributions like blog posts.

- **Custom Food Ordering with OpenGPTs**: A user showcased the extensibility of **OpenGPTs** by integrating a custom food ordering API, demonstrating the platform's adaptability for custom AI applications. Feedback is sought on their [YouTube demo](https://youtu.be/V1SKJfE35D8) titled "Hack OpenGPT to Automate Anything".

- **PersonaFinder GPT Released**: A member has developed **PersonaFinder GPT**, a conversational AI that can provide information on individuals based on their name, country, and profession. The tool is available for testing at [PersonaFinder Pro](https://chat.openai.com/g/g-xm4VgOF5E-personafinder-pro).

- **Call for Proficient Prompters to Test New Tool**: There's a request for proficient prompters to test and provide feedback on a new tool designed for automated code transformations to maintain code standards and quality for production deployment. The tool is accessible [here](https://tinyurl.com/gitgud-langchain).

- **Kleinanzeigen Ad Shared**: An ad on **Kleinanzeigen** for a picture was shared, though it seems unrelated to AI or projects on the LangChain AI Discord. The **Mona Bild** can be viewed [here](https://www.kleinanzeigen.de/s-anzeige/mona-bild-repost/2724274253-246-1564?utm_source=other&utm_campaign=socialbuttons&utm_medium=social&utm_content=app_ios).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tinyurl.com/gitgud-langchain">GitGud</a>: no description found</li><li><a href="https://www.kleinanzeigen.de/s-anzeige/mona-bild-repost/2724274253-246-1564?utm_source=other&utm_campaign=socialbuttons&utm_medium=social&utm_content=app_ios">Mona Bild repost</a>: Bild bekannt aus tiktok -,Mona Bild repost in Wuppertal - Elberfeld-West</li><li><a href="https://youtu.be/V1SKJfE35D8">Hack OpenGPT to Automate Anything</a>: Welcome to the future of custom AI applications! This demo showcases the incredible flexibility and power of OpenGPTs, an open source project by LangChain. W...
</li>
</ul>

</div>
  

---



**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1224266681153749002)** (79 messages🔥🔥): 

- **LinkedIn's Virtual Badges Cause a Stir**: LinkedIn user touts achieving over 30 Top Voice badges and shares a [post link](https://www.linkedin.com/posts/jillanisofttech_datascience-artificialintelligence-analyticalskills-activity-7180469402079789056-oPrY?utm_source=share&utm_medium=member_desktop), with others questioning the actual benefits of collecting such badges.
- **AI's Fictitious Package Hazard**: In a shocking find, [The Register reports](https://www.theregister.com/2024/03/28/ai_bots_hallucinate_software_packages/) that a software package previously hallucinated by generative AI has been made real, leading several big businesses, including Alibaba, to erroneously incorporate it, potentially opening the door to malware threats if not for the benign nature of the test package.
- **Challenges in PPO Algorithm Performance**: A member shares a curve indicating an issue with a Proximal Policy Optimization (PPO) algorithm, with others suggesting rollback to a previous checkpoint when the agent's performance declines.
- **Stable Diffusion's Evolution in AI Image Generation**: Members discuss stable diffusion advancements with new variants like *stable cascade* and OOT diffusion, and the introduction of new tools such as ControlNet for more input control, while others highlight persistent performance issues related to VRAM requirements.
- **Integrating Language Models and Tackling Misconceptions**: A user seeks help for integrating language models with their code base, discussing SDK utilization and expressing concerns about switching providers and prompt modification without recoding, while others approach chatbot development, seeking advice on avoiding AI disclosure from the responses.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://wccftech.com/chinese-ai-firm-unveils-deepeye-ai-box-featuring-up-to-48-tops-affordable-designs/">Chinese AI Firm Unveils &quot;DeepEye&quot; AI Box, Featuring Up To 48 TOPS &amp; Affordable Designs</a>: Chinese firm Intellifusion has unveiled a new AI-dedicated box that promises affordable AI solutions with an extensive upgrade roadmap.</li><li><a href="https://www.theregister.com/2024/03/28/ai_bots_hallucinate_software_packages/">AI bots hallucinate software packages and devs download them</a>: Simply look out for libraries imagined by ML and make them real, with actual malicious code. No wait, don&#39;t do that</li><li><a href="https://www.reddit.com/r/photoshop/comments/r7c2bh/evenout_lighting_for_a_tileable_texture/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/moritztng/fltr">GitHub - moritztng/fltr: Like grep but for natural language questions. Based on Mistral 7B or Mixtral 8x7B.</a>: Like grep but for natural language questions. Based on Mistral 7B or Mixtral 8x7B. - moritztng/fltr
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 messages): 

docphaedrus: https://youtu.be/7na-VCB8gxw?si=azqUL6dGSMCYbgdg
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1224350511785050323)** (5 messages): 

- **FastLLM Breaks the Billion-Token Barrier**: **FastLLM (FLLM)**, Qdrant's lightweight Language Model for **Retrieval Augmented Generation** (RAG), enters Early Access with an impressive context window of **1 billion tokens**. It is specifically designed to integrate with Qdrant, heralding a revolution in AI-driven content generation and retrieval capabilities. Read more about it on their [announcement post](https://qdrant.tech/blog/fastllm-announcement/).

- **Reinforcement Learning with Entropy in Mind**: An academic paper on *Soft Actor-Critic*, a method for **Off-Policy Maximum Entropy Deep Reinforcement Learning**, is shared, providing insights into **stochastic actor** approaches for reinforcement learning. The full text can be found on [arXiv](https://arxiv.org/pdf/1801.01290v2.pdf).

- **Finding the Right Open Source Status Page**: A blog post on Medium introduces the **6 best open-source status page alternatives for 2024**, offering insights for developers and teams looking to efficiently monitor and communicate their systems' status. The full article is available on [Medium](https://medium.com/statuspal/6-best-open-source-status-page-alternatives-for-2024-b68e5a967cc1).

- **IPEX-LLM and LlamaIndex Lead the Way**: A new Medium article discusses **IPEX-LLM and LlamaIndex** as potential game-changers in the realm of text generation and chat capabilities. The detailed piece on these advanced tools is accessible [here](https://medium.com/ai-advances/unlocking-the-future-of-text-generation-and-chat-with-ipex-llm-and-llamaindex-c98b84cdb3a2).

**Link mentioned**: <a href="https://qdrant.tech/blog/fastllm-announcement/">Introducing FastLLM: Qdrant’s Revolutionary LLM - Qdrant</a>: Lightweight and open-source. Custom made for RAG and completely integrated with Qdrant.

  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1224297516200230972)** (12 messages🔥): 

```html
<ul>
  <li><strong>Stream of Bot Conscience:</strong> Introducing <strong>LLMinator</strong>, a context-aware streaming Chatbot that enables running LLMs locally with Langchain and Gradio, compatible with both CPU and CUDA from HuggingFace. Check it out on <a href="https://github.com/Aesthisia/LLMinator">GitHub</a>.</li>
  <li><strong>Data Management Made Easier:</strong> DagsHub launches a new integration for Colab with DagsHub Storage Buckets, promising a better data management experience akin to a scalable Google Drive for ML. Example notebook is available on <a href="https://colab.research.google.com/#fileId=https%3a%2f%2fdagshub.com%2fDagsHub%2fDagsHubxColab%2fraw%2fmain%2fDagsHub_x_Colab-DagsHub_Storage.ipynb">Google Colab</a>.</li>
  <li><strong>Python's New Rival, Mojo:</strong> Speculations arise about the Mojo Programming Language surpassing Python in performance, as discussed in a YouTube video titled "Mojo Programming Language killed Python." Watch the full explanation <a href="https://youtu.be/vDyonow9iLo">here</a>.</li>
  <li><strong>Robotics Showcase:</strong> A member has built an advanced line follower and wall follower robot with a colour sensor, demonstrated in a YouTube video by SUST_BlackAnt. Find the full presentation <a href="https://www.youtube.com/watch?v=9YmcekQUJPs">here</a>.</li>
  <li><strong>Launch SaaS with OneMix:</strong> The new SaaS boilerplate OneMix claims to accelerate project launches by providing essentials like landing page, payment, and authentication setup. More details are available at <a href="https://saask.ing">saask.ing</a> and a demo on <a href="https://www.youtube.com/watch?v=NUfAtIY85GU&t=8s&ab_channel=AdityaKumarSaroj">YouTube</a>.</li>
</ul>
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/#fileId=https%3a%2f%2fdagshub.com%2fDagsHub%2fDagsHubxColab%2fraw%2fmain%2fDagsHub_x_Colab-DagsHub_Storage.ipynb">Google Colaboratory</a>: no description found</li><li><a href="https://x.com/__z__9/status/1774965364301971849?s=20">Tweet from ً ‎ (@__z__9)</a>: New preprint! The first multi-lingual red-teamed open-source continually pre-trained LLM - **Aurora-M** in accordance with the #WhiteHouse Executive Order on the Safe, Secure, and Trustworthy developm...</li><li><a href="https://arxiv.org/abs/2404.00399">Aurora-M: The First Open Source Multilingual Language Model Red-teamed according to the U.S. Executive Order</a>: Pretrained language models underpin several AI applications, but their high computational cost for training limits accessibility. Initiatives such as BLOOM and StarCoder aim to democratize access to p...</li><li><a href="https://youtu.be/7na-VCB8gxw?si=4u6MDWEfCT3e0b0S">On Virtual Containers: Safekeeping Your Python App</a>: A brief demo on how to use a platform like Podman (https://podman.io/) or Docker (http://docker.io/) to &quot;containerize&quot; your software projects for production....</li><li><a href="https://youtu.be/vDyonow9iLo">Mojo Programming Language killed Python</a>: I&#39;ll share with you why Mojo will be very popular very soon. It&#39;s killing Python performance wise making it very competitive and the key is: while keeping th...</li><li><a href="https://www.youtube.com/watch?v=9YmcekQUJPs">An advanced line follower and wall follower robot with colour sensor. Presented by SUST_BlackAnt</a>: This is an advanced line follower track. Feel free to like, comment and share. Let me know how you like it. If you want to contact me feel free to send an em...</li><li><a href="https://github.com/Aesthisia/LLMinator">GitHub - Aesthisia/LLMinator: Gradio based tool to run opensource LLM models directly from Huggingface</a>: Gradio based tool to run opensource LLM models directly from Huggingface - Aesthisia/LLMinator</li><li><a href="https://huggingface.co/collections/SauravMaheshkar/hypergraph-datasets-65fe10c95c6c7162e41e3f05">HyperGraph Datasets - a SauravMaheshkar Collection</a>: no description found</li><li><a href="https://x.com/MaheshkarSaurav/status/1775176529414086787?s=20">Tweet from Saurav Maheshkar ☕️ (@MaheshkarSaurav)</a>: I&#39;m working on HyperGraph Representation Learning at the moment and have spent the last few days creating a @huggingface collections consisting of: 👉 processed datasets 👉 papers 👉 @Gradio space...</li><li><a href="https://saask.ing">SaaS King | Best SaaS Boilerplates</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=NUfAtIY85GU&t=8s&ab_channel=AdityaKumarSaroj">One Mix by SaaS King | Boilerplate Demo</a>: A quick introduction to OneMix by SaaS King. OneMix is made with Remix (Vite), Tailwind, Supabase, Prisma, Stripe and Resend.How can OneMix by SaaS King help...</li><li><a href="https://www.youtube.com/watch?v=p77U2eyJFPU">made a musical slot machine then built a song with it - captains chair 21</a>: 00:00 - start01:35 - building the track08:28 - the trackour first @HuggingFace space. it&#39;s pretty ridiculous.https://huggingface.co/spaces/thepatch/the-slot-...</li><li><a href="https://huggingface.co/spaces/thepatch/the-slot-machine">The Slot Machine - a Hugging Face Space by thepatch</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/)** (1 messages): 

grimsqueaker: yay! thanks!
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1224278505546711100)** (7 messages): 

- **Batch Size Equivalence Query**: A member questioned if using **batch size 32 with accumulation of 2** is comparable to **batch size 64** for training different sizes of architectures, such as ConvNeXt.

- **Research Outreach in Quantum Neural Networks**: A member shared they are conducting research on the **performance of quantum neural network models** on traditional image datasets and faced a hiccup.

- **Feature Extraction & Quantum SVM Inquiry**: The member elaborated on their research, mentioning they extracted features using a **transformer model** and seeking advice for using these in a **Quantum SVM (QSVM)** for multi-class classification.

- **Seeking Quantum Kernel & Hyperparameter Guidance**: Recommendations for choosing an appropriate **quantum kernel** and **hyperparameters** for QSVM were sought, specifically within **Qiskit 1.0.2**.

- **Open Collaboration Invitation**: An interest in the QSVM research was expressed by another member, leading to an open invitation for direct messaging and potential collaboration.
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1224302630470287440)** (8 messages🔥): 

- **Triggering LoRA with Diffusers**: A member inquired about **how to trigger LoRA with diffusers** for which the response provided guidance on using [PEFT for inference](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference) including loading and managing adapters, especially LoRA, with the *DiffusionPipeline*.

- **Model Usage Confirmation**: The same member followed up with another query about knowing if a **model is being used**, which did not receive a direct response within the provided messages.

- **Seeking Assistance with PDFs**: A community member requested help for **fine-tuning an open-source language model on PDF files**, expressing challenges in this endeavor, yet no specific advice was offered in the chat record provided.

- **Checking In on Mistral**: A check-in was made regarding updates to **Mistral**, but no new information or responses followed the inquiry.

- **Realtime Video Jitters in Technology Discussion**: A community member shared **observations of jitter and drift** in a realtime video, questioning whether this could be due to rounding errors or a bug in the process, and hoping for insights into controlling this issue for better output.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://spright-t2i.github.io/">SPRIGHT</a>: SOCIAL MEDIA DESCRIPTION TAG TAG</li><li><a href="https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference">Load LoRAs for inference</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1224778620971515994)** (1 messages): 

- **Gradio 4.25.0 Drops with Smoother Performance**: Gradio's latest update introduces *automatic deletion of gr.State variables* to enhance performance for high-traffic demos and includes an *unload event* for actions when a browser tab is closed.
- **Lazy Example Caching Now Available**: Gradio 4.25.0 adds a **lazy example caching** feature with `cache_examples="lazy"`, especially benefiting **ZeroGPU** users by caching examples upon their first request rather than at server startup.
- **Streaming Audio Bug Squashed**: An update has fixed a bug related to **streaming audio outputs** in the new Gradio version.
- **More Intuitive gr.ChatInterface**: The **gr.ChatInterface** received upgrades, notably allowing the pasting of images directly from the clipboard.
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1224585697344753674)** (8 messages🔥): 

- **RL Integration Challenges in Mojo**: A member enquired about the challenges in running **Reinforcement Learning (RL) Python training** in Mojo, specifically the use of a PyTorch environment within Mojo. They were informed about the upcoming **MAX Engine** and **C/C++ interop** in Mojo, as detailed in the [Mojo Roadmap](https://docs.modular.com/mojo/roadmap#cc-interop), which will allow for re-implementing PyTorch interfaces and speed up RL environment development and execution.

- **Mojo Doc Cheerleader**: In response to a discussion about documentation, a member praised the **new documentation** for Mojo, saying it is quite comprehensive.

- **Mathematical Symbols in Mojo Variable Names**: There was a question about Mojo's support for mathematical names similar to Julia. It was clarified that Mojo currently supports only **ASCII characters** for variable names and follows Python's conventions for variable names.

- **String Handling Curiosity in Mojo**: A member was curious about why the division operator "/" isn't "Stringable" in Mojo, questioning if all string entities should inherently possess the Stringable trait.

- **Emoji Variable Naming Workaround**: A different member pointed out that in Mojo, symbols (including emojis) can be used as variable names by enclosing them in **backticks**, providing an example where an emoji is used as a variable.

**Link mentioned**: <a href="https://docs.modular.com/mojo/roadmap#cc-interop">Mojo🔥 roadmap &amp; sharp edges | Modular Docs</a>: A summary of our Mojo plans, including upcoming features and things we need to fix.

  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1224461746820091905)** (10 messages🔥): 

- **Tweeting Spree from Modular**: Modular shared a series of tweets, possibly as part of a campaign or event. Specific content of tweets was not provided.
- **Check Out Modular's Twitter**: For updates and information, follow the series of posted tweets by Modular at their official Twitter using the provided [links](https://twitter.com/Modular).
  

---


**Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1224783263936024597)** (1 messages): 

- **MAXimum Mojo Momentum Unveiled**: The latest release, MAX 24.2, has been made available for download, bringing a trove of new features specifically aimed at Python developers adopting Mojo. For more depth on these features, there are dedicated posts in the [MAX 24.2 announcement](https://www.modular.com/blog/max-24-2-is-here-whats-new) and the [Mojo open-source](https://www.modular.com/blog/the-next-big-step-in-mojo-open-source) blog.

**Link mentioned**: <a href="https://www.modular.com/blog/whats-new-in-mojo-24-2-mojo-nightly-enhanced-python-interop-oss-stdlib-and-more">Modular: What’s new in Mojo 24.2: Mojo Nightly, Enhanced Python Interop, OSS stdlib and more</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: What’s new in Mojo 24.2: Mojo Nightly, Enhanced Python Interop, OSS stdlib and more

  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1224259296297095188)** (47 messages🔥): 

- **Parallelism Challenge in Mojo**: Mojo's handling of non-embarrassingly parallel problems is still under development with some considering it similar to Swift's concurrency model as outlined in the [Swift Concurrency Manifesto](https://gist.github.com/lattner/31ed37682ef1576b16bca1432ea9f782).
- **Value Types vs. Identity**: A discussion clarified that Mojo's value types, unlike Python, do not inherently have identity, meaning that the `is` operator might have different semantics, focusing on value equality rather than object identity.
- **Tensor Performance Inquisition**: A comparison of tensor operations using Mojo's Tensor struct versus direct use of `DTypePointer` showed significant performance differences, which was attributed to inefficient copy initialization and was rectified by improving the implementation [see gist for details](https://gist.github.com/modularbot/88f71a13c2d3f546b9f4ee8a144ddd8e).
- **Top-Level Code and Escaping Operator Mystery**: Questions were raised about the implementation of top-level code in Mojo and the seeming absence of documentation on the "escaping" operator, with a member unable to find substantial information in the official docs.
- **SIMD Naïve Search Exploration**: A member expressed the desire to implement SIMD Naïve Search as outlined in an [academic paper](https://arxiv.org/pdf/1612.01506.pdf), but faced uncertainty on how to translate SSE2 intrinsic functions into Mojo constructs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/search?q=escaping+">Modular Docs</a>: no description found</li><li><a href="https://docs.modular.com/mojo/stdlib/tensor/tensor#__mul__).">tensor | Modular Docs</a>: Implements the Tensor type.</li><li><a href="https://gist.github.com/modularbot/88f71a13c2d3f546b9f4ee8a144ddd8e">playground.mojo</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://gist.github.com/lattner/31ed37682ef1576b16bca1432ea9f782">Swift Concurrency Manifesto</a>: Swift Concurrency Manifesto. GitHub Gist: instantly share code, notes, and snippets.
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1224463118445314068)** (2 messages): 

- **Refactoring Triumph for Prism CLI library**: The `Prism` CLI library modeled after Cobra underwent significant refactoring with the 24.2 update, resulting in a slew of new features such as shorthand flag support and enhanced command structure, which now manages parent and child relationships within struct fields. The update also ensures commands can use customized positional argument validation functions; the library however comes with several built-in validators. Check out the details and examples on [GitHub](https://github.com/thatstoasty/prism).

- **Easing the Reference Wrangle**: The creator of `Prism` has signaled a strong interest in the evolution of References, citing them as a main challenge during development. Better usability around References is eagerly anticipated in future updates.

**Link mentioned**: <a href="https://github.com/thatstoasty/prism">GitHub - thatstoasty/prism: Mojo CLI Library modeled after Cobra.</a>: Mojo CLI Library modeled after Cobra. Contribute to thatstoasty/prism development by creating an account on GitHub.

  

---


**Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1224291651669725284)** (4 messages): 

- **Matrix Multiplication Mystery**: A member encountered an error running `matmul.mojo` related to `test_matrix_equal[matmul_vectorized](C, A, B)`; adjusting the tolerance fixed the issue, suggesting a problem with result consistency between implementations.
- **Rounding Suspected**: By changing `DType.float32` to `DType.float64` at the top of the `matmul.mojo` file, the member was able to eliminate the error for some matrix elements but not all, indicating the error might be related to rounding.
  

---


**Modular (Mojo 🔥) ▷ #[⚡serving](https://discord.com/channels/1087530497313357884/1212827597323509870/1224265805278085180)** (7 messages): 

- **Exploring MAX Beyond Triton**: A member inquired about the potential benefits of MAX beyond just serving as a Triton backend. MAX Serving is described as a wrapper around MAX Engine which can be tried out using a local Docker container, with details available in the [MAX Serving documentation](https://docs.modular.com/serving/get-started).

- **Migration Clarification Sought**: The same member asked about migrating from a current setup using Triton inference server with two models (a tokenizer and an ONNX/TensorRT model) to MAX, questioning whether the migration would be as simple as updating the backend in the config.

- **Assistance Offered for Migration to MAX**: A rep offered help to the member contemplating the migration to MAX, expressing eagerness to understand the use case better and to support their pipeline's performance upgrade.

- **Details Matter in Migration**: The rep asked for specifics regarding the member's setup, inquiring about how the tokenizer model was implemented and how the two models are connected, particularly if Ensemble Models or Business Logic Scripting features were being used.

- **Seamless ONNX, GPU Support Pending**: While confirming that ONNX models would seamlessly work with a simple backend change in the config, the rep noted that MAX doesn't currently support GPU-hosted models, stating that it's being actively developed.

**Link mentioned**: <a href="https://docs.modular.com/serving/get-started">Get started with MAX Serving | Modular Docs</a>: A walkthrough showing how to try MAX Serving on your local system.

  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1224435153498013726)** (11 messages🔥): 

- **New Nightly Mojo Build Released**: The Mojo nightly build has been updated, and you can upgrade with `modular update nightly/mojo`. A changelog detailing the differences between the stable and new nightly builds can be found on [GitHub](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md).
- **Inspecting Differences Between Mojo Releases**: To compare the differences between the Mojo releases, a diff link has been provided: [Comparing releases on GitHub](https://github.com/modularml/mojo/compare/4feb92e..1a8f912).
- **Local Development of Mojo Stdlib**: If you are looking to test local modifications to the stdlib in Mojo, there's documentation available on how to develop and test it [here](https://github.com/modularml/mojo/blob/nightly/stdlib/docs/development.md#a-change-with-dependencies), along with the use of the `MODULAR_MOJO_NIGHTLY_IMPORT_PATH` environment variable for configuration.
- **Testing Best Practices for Mojo**: New tests in Mojo should prefer using the methods from the `testing` module over FileCheck for better practices.
- **Collaboration Channels for Contributors**: There's a suggestion to use the current channel for general discussions among Mojo contributors, while specific issues should be moved to GitHub repositories.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modularml/mojo/blob/nightly/docs/changelog.md">mojo/docs/changelog.md at nightly · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/blob/nightly/stdlib/docs/development.md#a-change-with-dependencies">mojo/stdlib/docs/development.md at nightly · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/compare/4feb92e..1a8f912">Comparing 4feb92e..1a8f912 · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1224253236182122526)** (17 messages🔥): 

- **Miniconda as a Lightweight Alternative**: A member inquired if **Miniconda** could be an effective substitute for Anaconda due to its smaller size, which was confirmed to be a viable option.
- **OhanaPal Seeks Collaborators**: *OhanaPal*, an app designed to assist neurodivergent individuals with everyday tasks and learning, is looking for community help in brainstorming and prototyping, mentioning they currently use **OpenAI GPT APIs**. Access to the prototype and more information can be found on their [website](https://www.ohanapal.app/).
- **Community Call for Involvement**: Participants were reminded of the upcoming **April House Party** and invited to contribute ideas on how **Open Interpreter** can improve the human condition for everyone in a new designated channel.
- **Anticipation for Open Interpreter Mobile App**: Discussion around the potential for an **Open Interpreter iPhone app** has been spurred by a Twitter post by Jordan Singer. Members expressed enthusiasm, and the project was encouraged for open-source development within the community.
- **Beginnings of a React Native App**: A community member disclosed they have started developing a react native app for **Open Interpreter**, though it is not fully completed.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/jsngr/status/1774110742070882478?s=46&t=kwbSfLYCOimQnegJhHK_iA">Tweet from jordan singer (@jsngr)</a>: ✨ talk to your computer remotely from your phone  i call it Teleport</li><li><a href="https://discord.gg/fjPmtRk8?event=1221828294811586572">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.</li><li><a href="https://www.ohanapal.app/">OhanaPal | Super App for the Super Abled</a>: Welcome to OhanaPal—where empowerment and inclusion meet, making every day extraordinary for the super abled.</li><li><a href="https://github.com/FiveTechSoft/tinyMedical">GitHub - FiveTechSoft/tinyMedical: TinyLLama trained with medical dataset and saved as GGUF file</a>: TinyLLama trained with medical dataset and saved as GGUF file - FiveTechSoft/tinyMedical
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1224272736092950658)** (45 messages🔥): 

- **Scaling Up for a Perfect Fit**: When 3D printing an **O1 Light**, it is recommended to **scale the model up to 119.67%** to fit the M5 Atom Echo properly into its slot. A user clarified that the sizing issue arises because the Atom must be removed from its case before installing.
- **Seamless Reconnection on M5Atom**: A new **pull request** on GitHub ([#214](https://github.com/OpenInterpreter/01/pull/214)) addresses an update for the M5Atom, allowing it to automatically reconnect to the last successful WiFi and Server URL upon reboot.
- **Calling on Creativity for Mobile Control**: The team at OpenInterpreter hinted at the potential for mobile phone control in the future and shared a GitHub repo ([open-interpreter-termux](https://github.com/MikeBirdTech/open-interpreter-termux)) to get OI running on Android through Termux.
- **Revamped Windows Installation Docs**: Updated instructions for installing OpenInterpreter on Windows were shared, detailing necessary tools like **Git**, **virtualenv/MiniConda**, **Chocolatey**, and **Microsoft C++ Build Tools**. For Linux users, discussions about various errors during the installation were noted, with a call to share those errors for collaborative troubleshooting.
- **Alternative Windows Package Managers**: In addition to the traditional Windows package management tools, users were advised to consider **winget** ([Microsoft's official package manager](https://learn.microsoft.com/en-us/windows/package-manager/winget/)) and **scoop** as alternative options for Windows package management needs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://01.openinterpreter.com/services/language-model">Language Model - 01</a>: no description found</li><li><a href="https://learn.microsoft.com/en-us/windows/package-manager/winget/">Use the winget tool to install and manage applications</a>: The winget command line tool enables developers to discover, install, upgrade, remove and configure applications on Windows computers.</li><li><a href="https://scoop.sh/">no title found</a>: no description found</li><li><a href="https://github.com/OpenInterpreter/01/pull/214/">Automatically reconnect to last successful WiFi and Server URL, if available by aramsdale · Pull Request #214 · OpenInterpreter/01</a>: Automatically reconnects to last successful WiFi and Server URL, if available Utilizing Preferences, detect successful WiFi connection, store to ssid preferences, and recall on reboot. Same for ser...</li><li><a href="https://github.com/MikeBirdTech/open-interpreter-termux">GitHub - MikeBirdTech/open-interpreter-termux: Instructions for installing Open Interpreter on your Android device.</a>: Instructions for installing Open Interpreter on your Android device. - MikeBirdTech/open-interpreter-termux</li><li><a href="https://git-scm.com/download/win">Git - Downloading Package</a>: no description found</li><li><a href="https://visualstudio.microsoft.com/visual-cpp-build-tools.">Microsoft C++ Build Tools - Visual Studio</a>: no description found
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1224257328644100116)** (7 messages): 

- **Exploring Open Interpreter**: A member shared their YouTube video titled "Open Interpreter Advanced Experimentation - Part 2," which may contain new experiments with the **Open Interpreter**. The video is available at [YouTube](https://www.youtube.com/watch?v=v9uXdRwAQ0c).

- **Fabric, the AI Augmentation Framework**: A GitHub repository named **fabric** was introduced; it's an open-source framework for augmenting humans with AI. It utilizes a crowdsourced set of AI prompts for solving specific problems, accessible at [GitHub - danielmiessler/fabric](https://github.com/danielmiessler/fabric).

- **Microsoft's UFO for Windows OS Interaction**: A member found **Microsoft's UFO**, a GitHub project described as a UI-Focused Agent for Windows OS Interaction. Questions arose if this is Microsoft's testing ground for implementing **Open Interpreter (OI)** on Windows, repository available at [GitHub - microsoft/UFO](https://github.com/microsoft/UFO).

- **Visual Intro to Transformers on YouTube**: A video titled "But what is a GPT? Visual intro to Transformers | Deep learning, chapter 5" was shared, providing an introduction to transformers, the technology behind **LLMs (Large Language Models)**. The video can be watched on [YouTube](https://www.youtube.com/watch?v=wjZofJX0v4M).

- **Community Excitement for GPT Educational Content**: Members expressed excitement about the educational content regarding transformers and **GPTs**. They shared their anticipation and approval with comments like "bookmarked!" and "Awesome 🚀".
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=v9uXdRwAQ0c">Open Interpreter Advanced Experimentation - Part 2</a>: ➤ Twitter - https://twitter.com/techfrenaj➤ Twitch  - https://www.twitch.tv/techfren➤ Discord  - https://discord.com/invite/z5VVSGssCw➤ TikTok - https://www....</li><li><a href="https://www.youtube.com/watch?v=wjZofJX0v4M">But what is a GPT?  Visual intro to Transformers | Deep learning, chapter 5</a>: An introduction to transformers and their prerequisitesEarly view of the next chapter for patrons: https://3b1b.co/early-attentionSpecial thanks to these sup...</li><li><a href="https://github.com/microsoft/UFO">GitHub - microsoft/UFO: A UI-Focused Agent for Windows OS Interaction.</a>: A UI-Focused Agent for Windows OS Interaction. Contribute to microsoft/UFO development by creating an account on GitHub.</li><li><a href="https://github.com/danielmiessler/fabric">GitHub - danielmiessler/fabric: fabric is an open-source framework for augmenting humans using AI. It provides a modular framework for solving specific problems using a crowdsourced set of AI prompts that can be used anywhere.</a>: fabric is an open-source framework for augmenting humans using AI. It provides a modular framework for solving specific problems using a crowdsourced set of AI prompts that can be used anywhere. - ...
</li>
</ul>

</div>
  

---



**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1224327044431614022)** (66 messages🔥🔥): 

- **Bot Name Prefix in Chatbot Responses**: A user encountered responses starting with `{bot_name}:` from the **undi95/remm-slerp-l2-13b:extended model** when using OpenRouter for roleplay chat using the `messages` key and queried whether it was due to a prompt error or required text replacement. It was clarified that recent updates to prompt templating shouldn't have caused this and the issue was discussed further, exploring whether the `name` field was being used.

- **Error Connecting to OpenRouter**: A user reported an **SSL error** (*EOF occurred in violation of protocol*) when trying to connect to OpenRouter, but no solution was directly offered in the chat.
  
- **Announcement of "Patterns of Application Development Using AI"**: **Obie Fernandez** announced the early release of his book, *Patterns of AI-Driven Application Architecture*, highlighting the use of OpenRouter.

- **Enquiry About Model Performance and Availability**: Users discussed the performance of various models and the availability of **nitro and non-nitro models**, with one seeking the fastest options available after the unavailability of nitro models. It was confirmed that **nitro models are still available, and more are on the way**.

- **General Troubleshooting and Model Suggestions**: Users shared experiences with model failures such as **NOUS-HERMES-2-MIXTRAL-8X7B-DPO**, and gave advice on alternative models for specific tasks like roleplay, with suggestions including **Nous Capybara 34B** equipped with 30k context window. Concerns about **OpenRouter logit bias** not working on certain models were addressed with an explanation that it's supported only on OpenAI's models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/blog/abhishek/autotrain-mixtral-dgx-cloud-local">Finetune Mixtral 8x7B with AutoTrain</a>: no description found</li><li><a href="https://leanpub.com/patterns-of-application-development-using-ai">Patterns of Application Development Using AI</a>: Discover practical patterns and principles for building intelligent, adaptive, and user-centric software systems that harness the power of AI.
</li>
</ul>

</div>
  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1224268340306903050)** (41 messages🔥): 

- **Benchmarks and Revelation of Thread Utilization**: A member expressed surprise at the significant improvements over **NumPy**, assuming it was already heavily optimized, and requested to see benchmarking code. The code shared utilizes both NumPy and a custom `matmul` function to demonstrate performance differences, revealing that **NumPy** does not use threads.

- **Eager Anticipation for New AI Updates**: Discussion revolves around the release of **llamafile 0.7** and attempts to use it with **openchat 3.5.** Members sought clarification on prompt templating and the use of variables within the UI, highlighting confusion due to a lack of documentation.

- **TinyBLAS vs Proprietary Libraries**: In discussing **llamafile's** performance on CPU vs. GPU, it was stated that the **`--tinyblas`** flag can be used for GPU support without installing CUDA or ROCm SDKs, but performance may vary based on the graphics card.

- **Compatibility Queries for Windows ARM64**: Discussion on **Windows ARM64** compatibility with **llamafile** raised questions about support and binary formats, revealing that Windows on ARM supports PE format with ARM64X binaries, but has issues with **AVX/AVX2** emulation.

- **Exercise and Troubleshooting in Local Deployment**: Users encountered an **"exec format error"** when trying to run **llamafile locally**, with suggestions to use bash instead of zsh, and clarifications provided for running **Mixtral** models on specific hardware configurations.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://json-schema.org/learn/getting-started-step-by-step">JSON Schema - Creating your first schema</a>: no description found</li><li><a href="https://huggingface.co/jartine/Mixtral-8x7B-Instruct-v0.1-llamafile/tree/main">jartine/Mixtral-8x7B-Instruct-v0.1-llamafile at main</a>: no description found</li><li><a href="https://huggingface.co/TheBloke/bagel-8x7b-v0.2-GGUF">TheBloke/bagel-8x7b-v0.2-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/TheBloke/dolphin-2.7-mixtral-8x7b-GGUF">TheBloke/dolphin-2.7-mixtral-8x7b-GGUF · Hugging Face</a>: no description found</li><li><a href="https://github.com/Septillihedron/SuperheroesPlusSchema/releases">Releases · Septillihedron/SuperheroesPlusSchema</a>: A documentation of the SkillsLibrary plugin. Contribute to Septillihedron/SuperheroesPlusSchema development by creating an account on GitHub.</li><li><a href="https://learn.microsoft.com/en-us/windows/arm/arm64x-pe">Arm64X PE Files</a>: Arm64X are a type of PE file in the Windows 11 SDK used for x64 compatibility on Arm64. Arm64X may be a good solution for developers of middleware or plugins, where code could get loaded into x64 or A...</li><li><a href="http://www.emulators.com/docs/abc_arm64ec_explained.htm">ARM64 Boot Camp: ARM64EC and ARM64X Explained</a>: no description found
</li>
</ul>

</div>
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1224562518127808553)** (6 messages): 

- **MDEL Marches Ahead with AMD**: MDEL has successfully trained a **15B model** using **AMD GPUs**, marking a potentially interesting development in hardware utilization for large-scale models.

- **Mistral Opens Its Doors**: The Mistral team invited community members to an **office hour session** for questions, signaling an open channel for dialogue and support. 

- **Skepticism Over New Release**: A member jokingly inquired if the **v0.2 release** is an **April Fools' prank**, reflecting community surprise or skepticism towards the update.

- **Dataset Unification Challenges**: A contributor is working on unifying approximately **15 different datasets** into TSV and pickle-formatted index files, facing challenges such as misaligned translations and the sheer data volume. They're considering the creation of a single, gigantic JSON of language pairs without weighting.

- **Seeking Runpod Experience**: A user inquired about experiences with **runpod serverless** for very large language models (**VLLM**), suggesting interest in community knowledge on this service.
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1224259266928574474)** (18 messages🔥): 

- **Pull Request Good to Merge**: The latest PR for `lisa` has been approved for merging, indicating that the changes made are considered solid and beneficial.
- **YAML Example Added and Run Initiated**: An example YAML file has been added to the codebase, and a subsequent testing run has been initiated to monitor the new configuration's performance.
- **Axolotl's Prank Proposal**: There's a tongue-in-cheek proposal for an April Fool's announcement claiming Axolotl has partnered with OpenAI, humorously suggesting that Axolotl can now fine-tune GPT-4 and future models.
- **DeepSpeed Out Of Memory (OOM) Issues**: Developers are encountering out of memory errors when trying to train models with DeepSpeed or FairScale Single-Process Single-GPU (FSDP), particularly with Lisa changes.
- **Documentation Update Inquiries**: A member observed updates to the Axolotl documentation but pointed out that the Table of Contents appears to be missing. Another member acknowledged the issue and plans to address it soon.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openaccess-ai-collective.github.io/axolotl/">Axolotl</a>: no description found</li><li><a href="https://github.com/OptimalScale/LMFlow/issues/726#issuecomment-2029701152">[BUG] LISA: same loss regardless of lisa_activated_layers · Issue #726 · OptimalScale/LMFlow</a>: Describe the bug I think there might be something wrong with the current LISA implementation. There is no difference in training loss, no matter how many layers are active. Not using LMFlow but HF ...</li><li><a href="https://github.com/kyegomez/BitNet/tree/main">GitHub - kyegomez/BitNet: Implementation of &quot;BitNet: Scaling 1-bit Transformers for Large Language Models&quot; in pytorch</a>: Implementation of &quot;BitNet: Scaling 1-bit Transformers for Large Language Models&quot; in pytorch - kyegomez/BitNet
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1224331766807138334)** (16 messages🔥): 

- **Training Hangs Mysteriously**: A member encountered a training issue where the process was stuck after the first epoch. They verified that the issue was not related to evaluation as `val_set_size: 0` had been set, suggesting evaluation was not being performed.
- **Hunt for the Culprit in Training Freeze**: Through discussion, it was suggested that the problem might be linked to lack of storage or potentially buggy features like `eval_table`, which is known for generating predictions and uploading to `wandb` during evaluation.
- **Eval Table: Buggy or Not?**: One member clarified that they do not have `eval_table` enabled since they do not perform evaluation during training. A hint was given that this feature could be buggy.
- **Config Consistency Not Guaranteing Performance**: The same member mentioned facing the training issue with some models but not with others, despite having used the same configuration, just changing the model name, adding to the mystery of the root cause.
- **Inference Prompt Practices Discussed**: In a separate thread, it was advised to use the same prompt format during model inference as used in training, specifically in the context of instructions within user input and impact on few-shot prompting after fine-tuning.
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1224327676286865451)** (29 messages🔥): 

- **FastLLM Launches with Big Claims**: Qdrant announced their new language model **FastLLM (FLLM)** designed for Retrieval Augmented Generation with a staggering context window of 1 billion tokens. The AI community highlighted this as potentially effective trolling, referring to its announcement on April Fools' Day.
  
- **New Instructional Gem on Transformers**: A video by 3Blue1Brown titled "But what is a GPT? Visual intro to Transformers | Deep learning, chapter 5" received attention for offering a visual introduction to transformers and GPTs.

- **LLM Answer Engine Github Project Unveiled**: An open source project titled "llm-answer-engine" on GitHub garnered interest for building a Perplexity-Inspired Answer Engine using a robust stack including Next.js, Groq, Mixtral, Langchain, and OpenAI.

- **Instructor Abstraction for Structured LLM Outputs**: The release of instructor 1.0.0 was noted, which is a tool that ensures structured outputs from LLMs align with user-defined Pydantic models, simplifying the interaction and integration with other system modules.
  
- **Google Revs Up on AI with New Leadership**: Logan Kilpatrick announced his move to Google to lead product for **AI Studio** and support the **Gemini API**, indicating a significant focus on making Google a prime location for developers in AI.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/officiallogank/status/1775222819439149424?s">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: Excited to share I’ve joined @Google to lead product for AI Studio and support the Gemini API.  Lots of hard work ahead, but we are going to make Google the best home for developers building with AI. ...</li><li><a href="https://qdrant.tech/blog/fastllm-announcement/">Introducing FastLLM: Qdrant’s Revolutionary LLM - Qdrant</a>: Lightweight and open-source. Custom made for RAG and completely integrated with Qdrant.</li><li><a href="https://9to5mac.com/2024/04/01/apple-ai-gpt-4/">Apple AI researchers boast useful on-device model that ‘substantially outperforms’ GPT-4 - 9to5Mac</a>: Siri has recently been attempting to describe images received in Messages when using CarPlay or the announce notifications feature. In...</li><li><a href="https://jack-clark.net/2024/03/28/what-does-1025-versus-1026-mean/">What does 10^25 versus 10^26 mean?</a>: A brief look at what FLOPs-based regulation nets out to  Recent AI regulations have defined the trigger points for oversight in terms of the amount of floating point operations dumped into training…</li><li><a href="https://x.com/jyangballin/status/1775114444370051582?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from John Yang (@jyangballin)</a>: SWE-agent is our new system for autonomously solving issues in GitHub repos. It gets similar accuracy to Devin on SWE-bench, takes 93 seconds on avg + it&#39;s open source!  We designed a new agent-co...</li><li><a href="https://x.com/_cgustavo/status/1775139142948552748?s=46&t=Tc6nPt_FP2Ybqya6_6Xu-w">Tweet from Gustavo Cid (@_cgustavo)</a>: I used to beg LLMs for structured outputs.   Most of the time, they understood the job and returned valid JSONs. However, around ~5% of the time, they didn&#39;t, and I had to write glue code to avoid...</li><li><a href="https://x.com/_cgustavo/status/1775139142948552748?s=46&t=Tc6nPt_FP2Ybqya">Tweet from Gustavo Cid (@_cgustavo)</a>: I used to beg LLMs for structured outputs.   Most of the time, they understood the job and returned valid JSONs. However, around ~5% of the time, they didn&#39;t, and I had to write glue code to avoid...</li><li><a href="https://x.com/sucralose__/status/1774782583731020200?s=46&t=JE84TqLviekDnEt8MAT-Eg">Tweet from Will Anthonio Zeppeli (@sucralose__)</a>: I did more investigation of ChatGPT&#39;s backend and found solid evidence of a model named &#34;GPT Alpha&#34; that I believe is the successor to GPT-4. It&#39;s possible to enable it early, but it r...</li><li><a href="https://x.com/officiallogank/status/1775222819439149424?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: Excited to share I’ve joined @Google to lead product for AI Studio and support the Gemini API.  Lots of hard work ahead, but we are going to make Google the best home for developers building with AI. ...</li><li><a href="https://www.youtube.com/watch?v=wjZofJX0v4M&t=2s">But what is a GPT?  Visual intro to Transformers | Deep learning, chapter 5</a>: An introduction to transformers and their prerequisitesEarly view of the next chapter for patrons: https://3b1b.co/early-attentionSpecial thanks to these sup...</li><li><a href="https://github.com/developersdigest/llm-answer-engine">GitHub - developersdigest/llm-answer-engine: Build a Perplexity-Inspired Answer Engine Using Next.js, Groq, Mixtral, Langchain, OpenAI, Brave &amp; Serper</a>: Build a Perplexity-Inspired Answer Engine Using Next.js, Groq, Mixtral, Langchain, OpenAI, Brave &amp; Serper - developersdigest/llm-answer-engine
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1224477813810270208)** (4 messages): 

- **"But what is a GPT?" Video Shared**: A [YouTube video](https://www.youtube.com/watch?v=wjZofJX0v4M) providing a visual introduction to transformers and GPTs was highlighted in the channel. It's aimed at those new to machine learning, offering an accessible stepping stone into the topic.
- **Homemade GPU Achieves New Milestone**: A member discussed their [personal project](https://www.furygpu.com/blog/hello) on developing a custom full-stack GPU capable of playing Quake after years of experience in the games industry. This FPGA design represents the culmination of a long-term engagement with the hardware aspect of rendering.
- **Justine's CPU Matmul Optimization Insights**: A link was shared to [Justine Tunney's blog](https://justine.lol/matmul/) which describes CPU matrix multiplication optimization and its similarities with GPU optimization, albeit without certain GPU-specific techniques like warptiling or explicit caching.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.furygpu.com/blog/hello">Hello! &mdash; FuryGpu</a>: After almost four years of developing a custom full-stack GPU in my spare time, I figured it was about time to start putting together some materials to highlight the project, its technical details, an...</li><li><a href="https://www.youtube.com/watch?v=wjZofJX0v4M">But what is a GPT?  Visual intro to Transformers | Deep learning, chapter 5</a>: An introduction to transformers and their prerequisitesEarly view of the next chapter for patrons: https://3b1b.co/early-attentionSpecial thanks to these sup...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1224381448191086662)** (6 messages): 

- **Triton Code Profiling with Nsight Compute**: A member shared they have seen profiling of Triton code using Nsight Compute, which allows viewing PTX and Python code simultaneously, as seen in the "Vectorized Load" section of [Accelerating Triton](https://pytorch.org/blog/accelerating-triton/). They inquired how to set up this insightful feature for their work.
- **Benchmarking Triton Kernels**: The blog post discussed provides a thorough template for accelerating Triton kernels, mentioning a significant improvement in performance from 275us to 47us for a typical Llama style inference input.
- **Nsight Compute Trace Download Versus Remote Launch**: One individual asked about the best practice for using Nsight Compute, questioning whether to generate a trace file and download it or to use a remote launch, hinting that setting up the remote launch could be cumbersome.
- **Command for Profiling Triton with Nsight**: A helpful command provided for generating a detailed profile using Nsight Compute was shared: `ncu --target-processes all --set detailed --import-source yes -o output_file python your_script.py`. This allows for profiling and subsequent analysis of the Triton code.
- **Successful Profiling Setup Confirmation**: A member confirmed the successful setup of Triton code profiling and expressed gratitude, indicating the utility and value of the profiling information obtained.

**Link mentioned**: <a href="https://pytorch.org/blog/accelerating-triton/">Accelerating Triton Dequantization Kernels for GPTQ</a>: TL;DR  

  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1224411486214951132)** (3 messages): 

- **Installation Troubles with Nsight DL Design on Ubuntu**: A member detailed their attempt to install **Nsight DL Design** by running the `.run` file on Ubuntu, confirming the execution rights with `chmod +x` and using `sudo`, but was unable to find the DL Design application post-installation. They sought advice on how to open the **DL Design app** after installation.
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1224392530720850010)** (2 messages): 

- **PyTorch Team Preparing a Response**: PyTorch representative indicated concerns about issues with the benchmarks comparing JAX, TensorFlow, and PyTorch, stating that the Keras team did not consult them to address these issues. A response is currently being formulated.
- **Benchmark Showdown**: A tweet from Jeff Dean was shared, highlighting key benchmarks from a given link showing that **JAX** is the fastest on GPU for the majority of tests, with TensorFlow also performing strongly, while **PyTorch** lagged behind on speed. The tweet references a recent benchmarking table, viewable [here](https://x.com/JeffDean/status/1774274156944859455).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/JeffDean/status/17742">Tweet from derek dukes (@ddukes)</a>: Bonfire at the beach about to watch the sunset.  Can you beat california?</li><li><a href="https://x.com/JeffDean/status/1774274156944859455">Tweet from Jeff Dean (@🏡) (@JeffDean)</a>: Here&#39;s the key benchmark table from the link. The JAX backend on GPUs is fastest for 7 of 12 benchmarks, and the TensorFlow backend is fastest for the other 5 of the 12. The Pytorch backend is not...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 messages): 

c_cholesky: Thank u 😊
  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1224755324100149390)** (1 messages): 

- **Potential for Opus Enhancement**: A discussion hints at the possibility that **Opus** might gain additional benefits from further **RLAIF** (Reinforcement Learning with Augmented Intermediate Features) application, following an assumption of accurate *Opus judgement*.
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1224783900036042772)** (2 messages): 

- **New Captain for Google's AI Ship**: An individual announced joining **Google** to lead product for AI Studio and support the Gemini API. They expressed determination to make Google the top destination for developers in AI with the sentiment: "I’m not going to settle for anything less."
- **Unexpected Power Move**: The move was surprising to members of the channel, prompting reactions like "did not expect this move AT ALL."

**Link mentioned**: <a href="https://fxtwitter.com/OfficialLoganK/status/1775222819439149424">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: Excited to share I’ve joined @Google to lead product for AI Studio and support the Gemini API.  Lots of hard work ahead, but we are going to make Google the best home for developers building with AI. ...

  

---


**Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1224375780545396889)** (5 messages): 

- **New Preprint on DPO and Verbosity**: A link to a [new preprint](https://x.com/rm_rafailov/status/1774653027712139657?s=46) was shared, highlighting the interplay between Direct Preference Optimization (DPO) and verbosity, with initial feedback pointing out issues faced when training on a LARGE scale.
- **Reinforcement Learning from Human Feedback Study**: An [arXiv preprint](https://arxiv.org/abs/2403.19159) discusses the exploitation of biases in human preferences, specifically verbosity, in Reinforcement Learning from Human Feedback (RLHF) and explores this under-researched area in the context of Direct Preference Optimization (DPO).
- **Endorsement of Rafael's Work**: A member mentioned that they "should prolly read" the preprint and acknowledged Rafael’s expertise in the field, noting that they have engaging conversations.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2403.19159">Disentangling Length from Quality in Direct Preference Optimization</a>: Reinforcement Learning from Human Feedback (RLHF) has been a crucial component in the recent success of Large Language Models. However, RLHF is know to exploit biases in human preferences, such as ver...</li><li><a href="https://x.com/rm_rafailov/status/1774653027712139657?s=46">Tweet from Rafael Rafailov (@rm_rafailov)</a>: New preprint is out on interplay between DPO and verbosity. Some of the first feedback we got on DPO was that training on LARGE scale the model becomes increasingly verbose until it diverges. Verbosit...
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[sp2024-history-of-open-alignment](https://discord.com/channels/1179127597926469703/1223784028428177510/1224755945343811624)** (1 messages): 

- **Shift to Secrecy Post-GPT-4**: A discussion highlighted the movement towards secrecy among companies after the release of the GPT-4 technical report, which notably omitted model details, marking a change from open data sharing to guarded science.
  

---



**AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1224364644559360194)** (8 messages🔥): 

- **Jamba's Speed Puzzle**: A member questioned how **Jamba** becomes faster with an increased number of tokens, specifically during the decoding step which is sequential. The graph referenced shows end-to-end throughput efficiency *per token*, leading to the confusion.
- **Decoding Speed Misconceptions**: The discussion then moved to clarify that the depicted speed-up in the plot is also present during the decoding phase, not just encoding. Despite decoding being a sequential process, the throughput increases even as context size grows.
