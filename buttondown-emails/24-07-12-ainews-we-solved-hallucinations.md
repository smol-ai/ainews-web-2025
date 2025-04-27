---
id: 720f4a48-42f5-4c93-a247-415197325b3e
title: We Solved Hallucinations
date: '2024-07-13T02:52:26.666831Z'
original_slug: ainews-we-solved-hallucinations
description: >-
  **GPT-2 training cost** has dropped dramatically to about **$672** on an
  8XH100 GPU node for 24 hours, down from **$100,000** in 2019, thanks to
  improvements in **H100 GPUs**, **CUDA**, **cuBLAS**, **cuDNN**, and
  **FlashAttention** software, as well as better data quality like the
  **FineWeb-Edu dataset**. **FlashAttention-3** was released, achieving **1.5-2x
  speedups on FP16** and up to **740 TFLOPS on H100 GPUs**, with FP8 performance
  nearing **1.2 PFLOPS**, developed collaboratively by **Meta**, **NVIDIA**,
  **Princeton**, and **Colfax**. The new **Hopper GPUs (H100)** feature hardware
  enhancements like **WGMMA**, **TMA**, and **FP8 support** enabling major
  speedups. In evaluation, a paper showed synthetic data does not improve vision
  tasks, and the **Avocado360 benchmark** was introduced to test vision-language
  models' ability to detect avocados. **Lynx**, a hallucination detection model
  for LLMs, was announced by **Patronus AI** on **Databricks Mosaic AI** using
  **Composer**, targeting real-world sectors like healthcare and fintech. The
  content also discusses Reddit URL quirks affecting LLM summarization and
  features a conversation on LLM Evals and the Open LLM Leaderboard.
companies:
  - meta-ai-fair
  - nvidia
  - princeton
  - colfax
  - patronus-ai
  - databricks
  - mosaic-ai
  - composer
models:
  - gpt-2
  - flashattention-3
  - lynx
topics:
  - compute-hardware
  - gpu-optimization
  - flashattention
  - fp16
  - fp8
  - benchmarking
  - vision
  - hallucination-detection
  - synthetic-data
  - dataset-quality
  - llm-evaluation
people:
  - karpathy
  - tri_dao
  - giffmana
  - vikhyatk
  - dbrxmosaicai
---


<!-- buttondown-editor-mode: plaintext -->**With one weird trick!**

> AI News for 7/11/2024-7/12/2024.
We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**463** channels, and **2566** messages) for you. 
Estimated reading time saved (at 200wpm): **276 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Look, we've known for a while that our Reddit summaries are ridden with... erm... links that don't go where they claim to go. You keep reminding us! (Thanks!)

 ![image.png](https://assets.buttondown.email/images/90d8843b-3834-46ed-85c5-56bf51f2c441.png?w=960&fit=max) 

The reason that this happens specifically to our Reddit summaries much much more than our Discord or Twitter recaps is because of Reddit's URL structure.

Here is a typical Reddit URL:

[https://www.reddit.com/r/LocalLLaMA/comments/1cxnrov/disappointing_if_true_meta_plans_to_not_open_the/](https://www.reddit.com/r/LocalLLaMA/comments/1cxnrov/disappointing_if_true_meta_plans_to_not_open_the/)

The slug at the end (`disappointing_if_true_meta_plans_to_not_open_the`) is just an attempt to make a human readable slug out of the title, AND the subreddit at the start (`r/LocalLLaMA`) is also just for human readability. In practice, all of it is ignored in favor of the *real* slug, that 7 character alphanumeric set (`1cxnrov`). Here, we'll prove it:

[https://www.reddit.com/r/SmolAI/comments/1cxnrov/ainews_is_the_best/](https://www.reddit.com/r/SmolAI/comments/1cxnrov/ainews_is_the_best/)

Despite having changed the subreddit and the human slug, Reddit sends you to the same post as before based on the *real* slug.

So Reddit URLs, much more than most URLs, are *hyper*, hyper sensitive to small mistakes in attention, even if all we are asking the LLM to do is copy from a source docment with the reference link neatly spelled out.

And... both Claude and GPT4 are trained on an awful lot of NSFW Reddit URLs ([in multiple languages](https://x.com/tianle_cai/status/1790109646205890723)!). Put these two facts together and you can see what we've been dealing with.

So.. [we went ahead and fixed the glitch](https://www.youtube.com/watch?v=BUE0PPQI3is), *while* still using LLMs to format, select, and summarize across a full corpus of Reddit submissions and comments. Tweet [@Smol_AI](https://x.com/Smol_AI) if you have guesses on how we do it.

It's been another content light day, so please enjoy our conversation with Clementine Fourrier on LLM Evals (our [coverage in May](https://buttondown.email/ainews/archive/ainews-to-be-named-4285/) and the future of the Open LLM Leaderboard:

https://www.youtube.com/watch?v=E-UhbYc8m24


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

**Compute and Hardware Improvements**

- **GPT-2 training cost down dramatically**: [@karpathy](https://twitter.com/karpathy/status/1811467135279104217) noted that training GPT-2 now costs ~$672 on one 8XH100 GPU node for 24 hours, down from ~$100,000 in 2019, due to improvements in **compute hardware (H100 GPUs), software (CUDA, cuBLAS, cuDNN, FlashAttention) and data quality (e.g. the FineWeb-Edu dataset)**.
- **FlashAttention-3 released**: [@tri_dao](https://twitter.com/tri_dao/status/1811453622070444071) announced FlashAttention-3, which is **1.5-2x faster on FP16, up to 740 TFLOPS on H100 (75% util), and FP8 gets close to 1.2 PFLOPS**. It is a collaborative effort with Meta, NVIDIA, Princeton, and Colfax.
- **Hopper GPUs enable major speedups**: [@tri_dao](https://twitter.com/tri_dao/status/1811453625165840608) noted that Hopper GPUs (H100) have new hardware features like WGMMA, TMA, and FP8 support that enable major speedups. Just rewriting FlashAttention for these gets to 570 TFLOPS.

**LLM Evaluation and Benchmarking**

- **Synthetic data may not help for vision tasks**: [@giffmana](https://twitter.com/giffmana/status/1811527727796642274) highlighted a paper showing that synthetic images don't actually help for vision tasks when the correct baseline is run.
- **Avocado360 benchmark for evaluating VLMs**: [@vikhyatk](https://twitter.com/vikhyatk/status/1811540521028067661) introduced the Avocado360 benchmark for evaluating if vision language models (VLMs) can determine if an image contains an avocado. **Four arbitrarily selected VLMs were evaluated**.
- **Lynx model for LLM hallucination detection**: [@DbrxMosaicAI](https://twitter.com/DbrxMosaicAI/status/1811537853350064592) announced Lynx, a new hallucination detection model for LLMs especially suited for **real-world applications in industries like healthcare and fintech**. It was trained by Patronus AI on Databricks Mosaic AI using Composer.

**LLM Applications and Frameworks**

- **Runway AI automation**: [@labenz](https://twitter.com/labenz/status/1811463195480977856) shared how Runway, a video generation startup, is using AI to automate tasks like pre-writing sales emails. They aim to **never have >100 employees by scaling with AI capabilities**.
- **LangGraph for human-in-the-loop feedback**: [@LangChainAI](https://twitter.com/LangChainAI/status/1811438797680492600) showed how to add checkpoints for human input and update the graph state in LangGraph, to **enable human feedback for agentic systems**.
- **Qdrant and LlamaIndex for advanced RAG**: [@qdrant_engine](https://twitter.com/qdrant_engine/status/1811451520698716262) shared an article on building an advanced RAG architecture combining LlamaIndex agents with Qdrant's **hybrid search capabilities, using both dense and sparse vector embeddings**.

**Memes and Humor**

- **Thinkpad love**: [@giffmana](https://twitter.com/giffmana/status/1811485814334918667) joked "What is the best laptop, and why is it a ThinkPad?"
- **Token limit woes**: [@HamelHusain](https://twitter.com/HamelHusain/status/1811508610469654922) hit token limits quickly on Anthropic UI even on the Pro Plan, wondering if it's normal.
- **ML/DS interview requirements**: [@jxmnop](https://twitter.com/jxmnop/status/1811503193798639970) joked that by next year, ML/DS interviews will require a medium-level question from ML leetcode, hardcore prompt engineering, and five years of CUDA experience.

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

**Theme 1. WizardLM 3 and LLM Optimization Techniques**

- [/r/LocalLLaMA] **[WizardLM 3 is coming soon 👀🔥](https://i.redd.it/a7lkiff2hxbd1.jpeg)** ([Score: 418, Comments: 73](https://reddit.com//r/LocalLLaMA/comments/1e0v437/wizardlm_3_is_coming_soon/)): **WizardLM 3**, an upcoming language model, is set to be released soon. The announcement hints at significant improvements or new features, though specific details about the model's capabilities or release date are not provided in the post.

- [/r/LocalLLaMA] **[FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://www.together.ai/blog/flashattention-3)** ([Score: 146, Comments: 22](https://reddit.com//r/LocalLLaMA/comments/1e0vh1j/flashattention3_fast_and_accurate_attention_with/)): **FlashAttention-3** introduces a new approach to attention computation in **Large Language Models (LLMs)**, offering **2-4x speedup** over previous methods while maintaining accuracy. The technique employs **asynchronous IO** and **low-precision computation**, allowing for efficient processing of longer sequences and potentially enabling the training of larger models with longer context lengths. This advancement, detailed in a [paper](https://arxiv.org/abs/2311.05908) by researchers from Stanford University and NVIDIA, could significantly impact the development and deployment of more powerful LLMs.


**Theme 2. Advanced AI-Generated Visual Content**

- [/r/StableDiffusion] **[fal drops AuraFlow](https://i.redd.it/ajp4lo34jzbd1.png)** ([Score: 322, Comments: 95](https://reddit.com//r/StableDiffusion/comments/1e14ip2/fal_drops_auraflow/)): **fal** has introduced **AuraFlow**, a new image generation model that combines the strengths of **Stable Diffusion** and **Midjourney**. AuraFlow aims to provide high-quality image generation with improved **coherence** and **composition**, addressing common issues like distorted faces and hands. The model is currently available through fal's API and will be integrated into their **no-code AI app builder**.

- [/r/StableDiffusion] **[AnimateDiff and LivePortrait (First real test)](https://v.redd.it/bpmfc8in3zbd1)** ([Score: 580, Comments: 66](https://reddit.com//r/StableDiffusion/comments/1e12sav/animatediff_and_liveportrait_first_real_test/)): **AnimateDiff and LivePortrait integration** showcases the potential for creating animated portraits from still images. The process involves using **AnimateDiff** to generate a **16-frame animation** from a single image, which is then fed into **LivePortrait** to produce a more realistic animated result. This combination of tools demonstrates a promising approach for bringing static images to life with fluid, natural-looking movements.

- [/r/singularity] **[Al-Generated Movie Trailer](https://v.redd.it/ixnbp7dye7bd1)** ([Score: 157, Comments: 41](https://reddit.com//r/singularity/comments/1e0rtp7/algenerated_movie_trailer/)): **AI-generated movie trailer** demonstrates advanced visual capabilities in film production. The trailer, created using **artificial intelligence**, showcases realistic **CGI characters**, **dynamic scene transitions**, and **complex visual effects** typically associated with high-budget productions, highlighting the potential of AI to revolutionize the film industry by reducing costs and expanding creative possibilities.


**Theme 3. AI Progress Tracking and Benchmarking**

- [/r/OpenAI] **[OpenAI Develops System to Track Progress Toward Human-Level AI](https://i.imgur.com/TjnZv1w.png)** ([Score: 232, Comments: 75](https://reddit.com//r/OpenAI/comments/1e0yqq8/openai_develops_system_to_track_progress_toward/)): OpenAI has introduced a new system called **AI Preparedness Framework** to monitor and assess progress towards **human-level artificial intelligence**. The framework aims to evaluate AI systems across **12 key capabilities**, including language understanding, reasoning, and task completion, using a **5-level scale** ranging from narrow AI to artificial general intelligence (AGI). This initiative is part of OpenAI's efforts to responsibly develop advanced AI systems and provide policymakers with actionable insights on AI progress.

- [/r/singularity] **[Rorschach test for AI: is this good or bad?](https://i.redd.it/j0da8gfsi0cd1.png)** ([Score: 110, Comments: 152](https://reddit.com//r/singularity/comments/1e18e2j/rorschach_test_for_ai_is_this_good_or_bad/)): **Rorschach tests for AI** are proposed as a method to evaluate AI capabilities, particularly in image interpretation and reasoning. The concept suggests using ambiguous images, similar to traditional Rorschach inkblot tests, to assess an AI's ability to perceive, interpret, and explain visual information. This approach could potentially reveal insights into an AI's cognitive processes and limitations, but also raises questions about the validity and reliability of such assessments for artificial intelligence systems.


**Theme 4. AI Content Regulation and Copyright Issues**

- [/r/StableDiffusion] **The AI-focused COPIED Act would make removing digital watermarks illegal** ([Score: 136, Comments: 155](https://reddit.com//r/StableDiffusion/comments/1e17eur/the_aifocused_copied_act_would_make_removing/)): **"Senators Introduce COPIED Act to Combat AI Content Misuse"**  The **COPIED Act**, introduced by a group of senators, aims to combat unauthorized use of content by AI models by creating standards for **content authentication and detection of AI-generated material**. The bill would make **removing digital watermarks illegal**, allow content owners to **sue companies using their work without permission**, and require NIST to develop standards for **content origin proof** and **synthetic content detection**, while prohibiting the use of **protected content to train AI models**. Backed by industry groups like **SAG-AFTRA and RIAA**, the act is part of a broader push to regulate AI technology and empowers state AGs and the FTC to enforce its provisions.

- [/r/LocalLLaMA] **The danger of AI is not what most people think it is.** ([Score: 100, Comments: 115](https://reddit.com//r/LocalLLaMA/comments/1e0uqq1/the_danger_of_ai_is_not_what_most_people_think_it/)): **AI's real danger stems from overestimation, not superintelligence**  The post argues that the true danger of AI lies not in its potential to become superintelligent, but in its current **limitations being overlooked**. The author suggests that AI is being deployed in areas where its **lack of intelligence** can cause problems, citing examples of [AI-generated fake legal cases](https://apnews.com/article/artificial-intelligence-chatgpt-fake-case-lawyers-d6ae9fa79d0542db9e1455397aef381c) and [biased pedestrian detection in self-driving cars](https://thenextweb.com/news/driverless-cars-pedestrian-detection-age-race-biases). They also posit that much of the discourse around AI safety is driven by **"moat-building"** and protecting **first-mover advantages** rather than genuine concern.

---

# AI Discord Recap

> A summary of Summaries of Summaries

**1. LLM Advancements and Training Techniques**

- **FlashAttention Accelerates Transformer Training**: The [FlashAttention-3](https://tridao.me/blog/2024/flash3/) release promises up to 1.5-2x speed boost on FP16 and up to 740 TFLOPS on H100 GPUs, achieving 75% utilization and potentially reaching 1.2 PFLOPS with FP8.
   - This technology, co-developed by **Colfax**, **Tri Dao**, **Meta AIT team**, and the **Cutlass team**, has already accelerated training for models like **GPT-4** and **Llama 3** by minimizing memory reads/writes in attention mechanisms.
- **Q-Galore Enhances Memory-Efficient LLM Training**: The novel [Q-Galore method](https://arxiv.org/abs/2407.08296) combines quantization and low-rank projection to substantially reduce memory usage and training time for large language models compared to GaLore.
   - Unlike GaLore which relies on time-consuming SVD operations, **Q-Galore** observes that some gradient subspaces converge early while others change frequently, enabling more efficient training without sacrificing accuracy.
- **Llama 3 405B Multimodal Model Imminent**: Meta Platforms is set to release its largest **Llama 3** model with **405B parameters** on **July 23**, a year after Llama 2, as a **multimodal** offering according to [reports](https://www.theinformation.com/briefings/meta-platforms-to-release-largest-llama-3-model-on-july-23).
   - The release has sparked excitement in the community, with discussions around the **infrastructure requirements** like **8x H100s or 8x MI300X GPUs** to run such a massive model.
  


**2. Open Source AI Advancements**

- **AuraFlow: Largest Open Text-to-Image Model**: [AuraFlow](https://huggingface.co/fal/AuraFlow) by **Fal AI** has been released as the largest open text-to-image model under an Apache 2.0 license, supported in `diffusers` and achieving state-of-the-art results on GenEval.
   - With **LoRA support** coming soon and the model in beta, **community feedback** is crucial, as credited to [@cloneofsimo](https://twitter.com/cloneofsimo) and [@isidentical](https://twitter.com/isidentical) for significant contributions.
- **Cohere Toolkit Goes Open Source**: Cohere has [open-sourced their chat interface](https://github.com/cohere-ai/cohere-toolkit) on GitHub, with OCI integration planned, as announced by *Sssandra*.
   - *Mapler* expressed excitement about using the open-sourced toolkit for personal projects and updating the community on progress.
- **OpenArena Fosters LLM Dataset Enhancement**: The [OpenArena project](https://github.com/syv-ai/OpenArena) on GitHub pits language models against each other, with a third model as judge, to **increase dataset quality** through competitive challenges.
   - Inspired by the [WizardLM paper](https://www.microsoft.com/en-us/research/publication/arena-learning-build-data-flywheel-for-llms-post-training-via-simulated-chatbot-arena/) on **Arena Learning**, OpenArena leverages AI-annotated results for supervised fine-tuning and reinforcement learning of LLMs.
  


**3. Community Collaboration and Knowledge Sharing**

- **LlamaIndex Unveils Agentic RAG Cookbooks**: LlamaIndex, in collaboration with @jeffxtang from AIatMeta, released [cookbooks on agentic RAG](https://t.co/mBNZx9b1JO), covering topics from routing and tool use to multi-document agent building.
   - Additionally, a [Cypher snippet](https://t.co/dAV2QuAoZH) by @tb_tomaz and Neo4j effectively performs entity deduplication, aiding knowledge graph creation as shared on the [Neo4j GitHub](https://t.co/lMApLzMOMr).
- **Unsloth Notebooks for Continued Pretraining**: Unsloth provides [notebooks](https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama) for training local models with Ollama and Hugging Face models, as well as handling [continued pretraining](https://docs.unsloth.ai/basics/continued-pretraining) across different sequence lengths.
   - The community discussed techniques like concatenation and truncation for varying `max_seq_length`, and understanding parameter differences during LoRA and PEFT setups.
- **LangChain Optimizations and Best Practices**: The LangChain community shared optimization techniques for **embedding functions**, like using **caching mechanisms** (in-memory or Redis) to avoid recomputing embeddings and considering **async requests**.
   - Discussions also covered **FAISS vs Chroma** for handling large datasets, combining their strengths with **Chroma for persistence** and **FAISS for similarity search**, and improving **LangChain agent** efficiency.
  


**4. Hardware Benchmarking and Adoption**

- **Evaluating GPUs for AI Workloads**: Discussions compared the value proposition of **3090 vs 4090 GPUs** for AI workloads, with many favoring the 3090 for its better price-to-performance ratio given the relatively small generational performance leap.
   - Rumors about the upcoming **NVIDIA 5090** having only **28GB VRAM** instead of 32GB led to suggestions for building affordable **multi-GPU servers** using 3090s for increased VRAM capacity.
- **H100 GPU Excitement and Adoption Challenges**: The arrival of **H100 GPUs** generated significant excitement, with members exclaiming *'H100 go brrrrr'* and discussing the substantial performance improvements over previous generations.
   - However, concerns were raised about **Flash attn3** currently being limited to H100 support, with hopes that it would follow **Flash attn2**'s path of expanding to 3090 and 4090 GPUs.
- **Benchmarking Local AI Models**: A member shared their [personal benchmark table](https://dubesor.de/benchtable.html) evaluating various local AI models across 83 tasks using a weighted rating system covering reasoning, STEM, coding, and censorship categories.
   - While not representing broader benchmarks, the table provides insights into one individual's experiences and highlights the growing interest in comprehensive model evaluation by the community.

---

# PART 1: High level Discord summaries




## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **RT-DETR Races Ahead of YOLO**: **RT-DETR**, surpassing YOLO in speed and accuracy, joined forces with Roboflow for advancement in object detection and is now seamlessly accessible via the [transformers library](https://github.com/huggingface/transformers?ref=blog.roboflow.com).
   - The model's edge was corroborated in a research paper (https://arxiv.org/abs/2304.08069?ref=blog.roboflow.com), supporting RT-DETR's integration into existing workflows and advancement in detection tasks.
- **Elevation of Efficiency with Hiera Model**: Introducing **Hiera**, the transformers library now includes a transformative vision model simplifying hierarchical complexities and excelling in performance tasks like image classification.
   - Hiera's flexibility shines through its various implementations, including *HieraForImageClassification* and *HieraBackbone*, detailed in the [GitHub Pull Request](https://github.com/huggingface/transformers/pull/30356#event-13478146443).
- **Toolkit Trims the Fat From LLM Finetuning**: The [Georgian-io toolkit](https://github.com/georgian-io/LLM-Finetuning-Toolkit) debuts, catering to streamlined finetuning across multiple LLMs, simplifying end-to-end data science processes.
   - A versatile toolkit, it facilitates running batch experiments, evaluating metrics, and performing hyperparameter and prompt ablations via a unified config.
- **Visualization of the AuraFlow Landscape**: **AuraFlow**, celebrated as the largest open text-to-image model, recently swooped into the spotlight for its promising GenEval results, supported by `diffusers`.
   - With **LoRA support** on the horizon, ongoing development and community feedback are encouraged via [fal's Discord](https://discord.gg/fal-ai), lining the path for further enhancements.
- **qdurllm Demo Signals New Capabilities**: A leap in intuitive output is showcased in the [qdurllm demo](https://huggingface.co/spaces/as-cle-bert/qdurllm-demo), extending an invitation for community feedback on its advanced interactive model.
   - The offering opens dialogue for potential burgeoning use-cases and integrative, accessible advancements.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **ComfyUI Reactor Receives Installation Reactivation**: Following a [YouTube video](https://www.youtube.com/watch?v=vCCVxGtCyho) provides a solution for error-free installation of **ComfyUI InsightFace**.
   - This workaround, confirmed by users, remains effective for the version released in **2024**.
- **Deforum Dives into Distinct Color Dynamics**: For fine-tuning abstract video aesthetics, setting `color_coherence` to None in **Deforum Stable Diffusion API** was discussed as a potential way to enhance color transitions.
   - Community inputs were solicited to optimize the vividness and clarity in visual projects.
- **Generation Interruption Queries on auto1111**: Users experienced notable delays in stopping generation processes in the auto1111 setup, attributing it to VRAM limitations and software nuances.
   - Comparisons were drawn to gradually decelerating a high-speed train, emphasizing the need for patience during abrupt halts.
- **Analyzing the Asset of AI Tool Affordability**: The community discussed the costs of commercial AI tools like **Runway**, which offers a plan at **$90/mo**, contrasting with free local AI options.
   - Despite the allure of no-cost tools, members recognized that premium services often deliver superior functionality and enhanced features.
- **Scaling Up on Upscaling: Pursuit of Free Tools**: The search for complimentary creative image upscaling tools resulted in recommendations for accessible software like **Krita** and **GIMP**.
   - These alternatives were praised for their useful features without the financial barrier, aligning with the community's resource-conscious preferences.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **FA3 Triumphs and CUDA Concerns**: A spirited debate evaluated the merits of **FA3 vs cuDNN and ThunderKittens**, revealing a preference for simplicity and ease despite the allure of FA3's potential speed-up in attention mechanisms.
   - Technical concerns around **FP8 implementation hurdles** and the non-existent FP8 track in **ThunderKittens** sparked an assessment of maintenance complexity.
- **Evaluating GPU Access Options**: Members lauded **Google Colab** for its frictionless GPU access, while comparing the pros and cons of **Coreweave** and **Lambda Labs** for GPU rentals, highlighting price and allocation issues.
   - The discussion highlighted **Google Cloud GPU** as a more costly but powerful option for uses beyond notebooks, elevating **Colab** for ease of use in tinkering with CUDA kernels.
- **Streamlining Matrix Multiplications**: Conversations addressed effective thread assignment strategies in matrix-matrix multiplication, suggesting a thread per row is more efficient in terms of caching and data loading due to memory layout.
   - The notion of 'coalescing' became focal, as insights pertaining to memory arrangements surfaced, emphasizing efficiency in reducing over the last matrix dimension.
- **Innovative Tactics in AI Training**: Members discussed the viability of **tensor subclass** usage with FSDP, as emerging projects like **bitnet work** hint at burgeoning applications in distributed training.
   - The community acknowledged the sustained contribution and was poised to collaborate on a **developer guide for enabling tensor subclasses**, anticipating future demand.
- **Collaboration and Growth within LLM.C**: The **LLM.C community** is buzzing with initiatives around model sharing and resource consolidation, as evident in the creation of organizational structures on **Hugging Face**.
   - Insights were shared on performing optimizations and fine-tuning large-scale models, also sparking ideas around **FP8's 33% speed boost**, despite memory reuse considerations.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **LLVM Creator's Chronicles**: The latest [Primeagen video](https://www.youtube.com/watch?v=ovYbgbrQ-v8) interviewing the creator of LLVM, Clang, Swift, and Mojo sparked discussions after becoming accessible on YouTube.
   - Participants noted that detailed insights can be a great resource for understanding the development philosophy behind Mojo's creation.
- **Mojo's REPL Iteration**: Debate swirled around the Mojo REPL's lack of immediate output for expressions, drawing comparisons to Python's REPL behavior.
   - Although current functionality does not display results like `1+1` directly, members were advised to submit requests through [GitHub issues](https://github.com/modularml/mojo/issues/2809) to incorporate these features.
- **Max Website Makeover Embraces Clarity**: Modular's **MAX framework** takes center stage with a [revamped website](https://modular.com), emphasizing its extensive developer base and clear licensing terms.
   - The site showcases the synergy between Max's performance capabilities and ease of use provided by the Mojo language, without delving into low-level coding.
- **GPU Gains with Mojo's MAX**: A promising dialogue emerged on writing custom GPU kernels using Mojo within MAX for enhanced performance.
   - This opens avenues for harnessing MAX's robust interfaces and Mojo's agile kernel compilation without direct CUDA involvement.
- **Datatype Discrepancies in MAX Model Execution**: A **datatypes issue** arose when executing a MAX Model, leading to a mismatch in expectations versus actual results when using `PythonObjects`.
   - Correcting the `np.full()` operation's `dtype` to `np.float32` provided the solution, underscoring the precision needed in model execution parameters.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Gemini Soars with Token Expansion**: [Gemini 1.5 Pro boasts a 2 million token window](https://developers.googleblog.com/en/new-features-for-the-gemini-api-and-google-ai-studio/) and introduces **context caching**, and **code execution** features.
   - **AI developers** are delighted with unlimited JSON capacity.
- **FlashAttention Sprints on Hopper GPUs**: FlashAttention-3 promises efficient **Hopper GPU** utilization with up to **35% FLOPs**, [outlined in a Tech Blog](https://www.together.ai/blog/flashattention-3).
   - "Substantial FLOPs leverage" is confined to **Hopper users**.
- **TF-ID Models Eye Vision-Language Tasks**: [TF-ID models are unleashed](https://github.com/ai8hyf/TF-ID) by Yifei Hu, featuring training code, dataset, and weights under the MIT License for vision-language tasks.
   - These models require only a few hundred domain-specific elements to finetune.
- **CodeGeeX4 Clips GPT's Wings**: The new [CodeGeeX4-ALL-9B model](https://huggingface.co/THUDM/codegeex4-all-9b) overshadows **GPT-3.5 and GPT-4** in code generation capabilities.
   - Achieving top performance, it boasts **128k context** and supports a plethora of programming languages.
- **Meta's Anticipated LLaMA 3 Debut**: Excitement builds for Meta Platform's July 23 release of its LLaMA 3 model, with potential for considerable AI progression.
   - The release [detailed here](https://huggingface.co/nvi) could reshape hardware preferences for AI application deployment.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **OpenAI Teases Doctoral Disruption**: OpenAI hints at forthcoming models with problem-solving adeptness equating to a doctoral degree, inciting discussions on approaching AGI.
   - An anonymous source leaked a GPT-4 demo showcasing its advanced human-like problem-solving capabilities.
- **Anthropic's AI Prognosis**: Dario Amodei from Anthropic predicts forthcoming AI Safety Levels, suggesting **A.S.L. 3** might emerge as early as this year, and **A.S.L. 4** by 2025-2028.
   - **A.S.L. 4** raising alarms about potential exacerbation of global risks through biological and cyber technology misuse.
- **Community Doubts OpenAI's Strategy**: Amidst news of potential breakthroughs, voices in the community express skepticism regarding OpenAI's strategic release patterns.
   - Conversations circle the possibility that OpenAI's teasers could be a ploy aimed at boosting their valuation despite previous achievements.
- **C's the Day with GPT-2**: **Karpathy** demonstrates efficient replication of **GPT-2 (1.6B)** using llm.c, encapsulating both power and cost-efficiency.
   - The implementation proves llm.c's capacity for large-scale language model training, blazing through with a 24-hour turnaround.
- **Safety in Simplicity with C++**: Safetensors.cpp debuts as a zero-dependency C++ library for **LibTorch**, easing the data manipulation burdens in model development.
   - The objective is clear: to streamline model data processes, ensuring smoother and more productive workflows.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Labs: To Use or Not to Use?**: Debates surged around the utility of [Perplexity Labs](https://discord.com/channels/1047197230748151888/1047649527299055688/1233731074685665280), with community members dissecting its versatility on various devices.
   - **Pros and cons** were thrown around, spotlighting the Labs' integration perks and questioning its advantage over mobile use versus the web interface.
- **Claude 3.5 Tramples Claude 3 Opus**: [Claude 3.5's superior performance](https://discord.com/channels/1047197230748151888/1047649527299055688/1261038487453302895) in reasoning and logic over its predecessor Claude 3 Opus caught everyone's eye, hinting at a shift in model dominance.
   - While **praise was unanimous** for Claude 3.5, speculation arose over the potential of future versions like Opus 3.5 to recalibrate the scales.
- **AI as a Beacon for Diabetes Management**: AI for diabetes management was spotlighted, with discussions around apps that assist patients and doctors in **insights derivation** rather than just insulin adjustments.
   - [Recent advancements](https://www.perplexity.ai/search/are-there-any-advances-in-apps-5NVLNla1T6.oHZAm_U70fw) were noted, offering not only automated insulin dosing but also predictive insights, reshaping patient care.
- **Error 524 Clouds Perplexity API**: AI engineers reported **sporadic Error 524** when integrating Perplexity with asynchronous frameworks, despite staying within prescribed limits.
   - Switching models adds to the conundrum, with transitions between `llama-3-{8b/70b}-instruct` to `llama-3-sonar-{large/small}-32k-online` resulting in similar errors, baffling users.
- **Cloudflare Stirs Perplexity API Turbulence**: Troubleshooting tales disclosed [Cloudflare as the culprit](https://discord.com/channels/1047197230748151888/1161802929053909012/1261093061883199552) behind VPN-blocked access to Perplexity API, a revelation for many.
   - While some struggled, others found bypassing VPN as an effective workaround, reinstating their access and quelling the **storm**.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **GPT-4Chan ends TruthfulQA's reign**: [Tweet from Maxime Labonne](https://x.com/maximelabonne/status/1746317053143773628) reignited the conversation about GPT-4Chan's past dominance on **TruthfulQA** despite ChatGPT's emergence.
   - Participants concurred that some benchmarks like TruthfulQA can mislead, while others like **MT-Bench** are deemed more indicative of true performance.
- **Jsonnet, a necessary evil for configuration?**: While Jsonnet garners appreciation for its streamlined configuration capabilities, a discussion unveiled its inadequacies in debugging, causing *mixed feelings* amongst users.
   - Despite its challenges, Jsonnet's role is recognized for its cleanliness, standing out among the diverse options for configuration tasks.
- **London AI Meetups Miss the Mark**: The forum echoed disappointment with **London AI meetups**, reflecting that they fall short for those seeking deeper AI discourse.
   - Suggestions pointed to **academic seminars** and conferences such as **ICML** for fulfilling the appetite for more substantial tech gatherings.
- **LLMs Face **Simple but Steep Challenges**: The updated [Alice in Wonderland paper](https://arxiv.org/abs/2406.02061) uncovered simple puzzles that perplex SOTA models like **Claude 3.5 Sonnet**.
   - This discourse around SOTA LLMs' inability to handle simple modifications spotlights a need for robust benchmarks and enhances our understanding of model limitations.
- **Memory Bandwidth: The GPT-2 Training Limiter**: Discussions revolved around the **1000x memory bandwidth** amplification requirement for training **GPT-2** models on a trillion-token dataset within an hour.
   - The focus shifted to **Hadamard transform** as an innovative solution to the quantization puzzle, as detailed in [Together's blog post](https://www.together.ai/blog/flashattention-3).



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **FlashAttention-3 Ignites GPU Performance**: [FlashAttention-3](https://x.com/tri_dao/status/1811453622070444071) is now out, promising a 1.5-2x speed boost on FP16, reaching up to 740 TFLOPS on H100 GPUs.
   - The new release reportedly achieves **75% utilization** on H100 and could potentially hit 1.2 PFLOPS using FP8.
- **OpenAI's Lucrative Ledger**: According to a recent [report](https://x.com/jvnixon/status/1811278381184672156), OpenAI is projected to hit $1.9B in revenue with ChatGPT Plus leading the chart.
   - This speculation highlights OpenAI's possible industry lead with impressive figures for ChatGPT Enterprise, the API, and the Team offering.
- **AGI Framework Unfolded by OpenAI**: OpenAI has unveiled a [5-level framework](https://x.com/shiringhaffary/status/1811508824970264595) for tracking AGI, placing themselves at level 2.
   - GPT-4's reasoning skills were put on show at a recent meeting, indicating progress outlined in this strategic framework.
- **Decentralized AI Training Takes Flight**: [Prime Intellect's OpenDiLoCo](https://x.com/shaughnessy119/status/1811459606377582857), a take on DeepMind's model, enables distributed AI training across global nodes.
   - A successful case involved a 1.1B parameter model trained across various nodes in three different countries.
- **Fireworks Spark in AI Funding Arena**: [Fireworks AI](https://x.com/lqiao/status/1811500361485517153) recently bagged $52M Series B funding for their platform aimed at compound AI system advancements.
   - The funding will be channeiled into Nvidia and AMD integrations and tailoring enterprise AI solutions.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Seamless Synthesis with Indexify**: [Prashant Dixit](https://x.com/Prashant_Dixit0/status/1811618990352912846) spotlights **structured data extraction** methods from unstructured sources in a Towards AI publication.
   - The use of **Indexify** for creating **data ingestion and extraction pipelines** is introduced, with further insights in the [Towards AI article](https://pub.towardsai.net/structured-financial-data-extraction-from-unstructured-data-ca2c8d166de6).
- **Vector Vantage: Chroma over OpenAI**: Discussions revolved around the configurations needed to properly load a Chroma vector store with **OpenAI embeddings**, emphasizing consistent `collection_name` for error-free operation.
   - Participants explored persistent storage tactics and effective management of embedded documents to reduce redundant computation.
- **Embeddings at Light Speed**: Techniques to accelerate OpenAI embedding functions were exchanged, with caching strategies being central, ranging from **in-memory** to using something like **Redis**.
   - Approaches to improve embedding processes included reducing token loading and leveraging asynchronous embedding requests.
- **FAISS or Chroma: The Dataset Dilemma**: A debate on **FAISS vs Chroma** ensued, favoring FAISS for handling extensive datasets efficiently, while Chroma was preferred for its persistence capabilities with smaller collections.
   - A hybrid method combining Chroma's persistent storage with FAISS's similarity searches was touted as an effective solution.
- **LangChain Agents Advance**: Challenges concerning unnecessary reembedding by LangChain agents were dissected with a keen focus on minimizing vector store initialization times.
   - Proposed solutions covered persistence mechanisms and various other refinements to enhance the operations of LangChain's agents.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **FlashAttention Ignites LLM Performance**: A technical review unfolded revealing how [FlashAttention](https://tridao.me/blog/2024/flash3/) and its sequel have streamlined Transformer training, drastically increasing **GPT-4** and **Llama 3** context lengths via optimized memory operations.
   - **NVIDIA's 4090** only marginally improves over the **3090**, but the entry of **FlashAttention** tech has sparked discussion on the actual need for the high-end cards given the new methodology's memory management efficiency.
- **RAMblings on NVIDIA's Next Move**: Speculation is ripe as whispers hint at the **NVIDIA 5090** sporting a mere **28GB VRAM**, diverging from the expected 32GB, with a [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1dzejen/cheapest_144gb_vram_server_youll_ever_see/) offering a DIY alternative for VRAM galore.
   - While debates churn about 3090's better price-performance, the likelihood of multi-V100 setups as a worthy adversary for AI endeavors was dissected, leaning towards single, high-powered GPU builds for optimal turnarounds.
- **Vulkan Rides the Wave of Support**: While OpenCL lags, unable to load models on a **7600XT**, falling out of grace for AI work, there's chatter about **Vulkan** entering LM Studio's support list, promising a fresh take on model interaction.
   - Discussions indicated **Vulkan's** rising popularity over OpenCL, a welcome change especially for **ollama** adopters, yet precise launch dates remain elusive amidst keen anticipation.
- **Einstein's Gravitational Pull on Branding**: Salesforce invested a grand **$20 million** to christen their new AI model **Einstein**, sparking a mix of industry jokes along with a circumspect review of the investment's wisdom.
   - A jocular sentiment was palpable as one vividly imagined **Einstein**'s likeness trapped in a corporate frame, birthing quips on its potential as a meme about the stranglehold of AI branding.
- **Reactive AI Development with LM Studio**: A creative endeavor emerges as an engineer integrates **Gemma 2** via LM Studio's API into a React application, stirring advice to consider an embedding database like **Faiss** for RAG's setup, optimizing for batched PDF processing.
   - As developers swapped tales of woe and success, advocating for more empathic support within the community, LM Studio's SDK was put forth as an adept companion for those merging cutting-edge AI into apps with rich user interactions.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Decentralize for Power to the Chips**: Discussion centered on the benefits of **decentralized computation** for AI tasks, using **stable diffusion** and untapped **idle processing power** to optimize **CMOS chips**.
   - Users called for the expansion of **High-Performance Computing (HPC)** capabilities through decentralization, enabling refined parallel computing.
- **OpenAI's Next-Gen AI Blueprint**: A new [tier system revealed by OpenAI](https://archive.is/SLtFQ) describes advancements from 'Reasoners' with doctorate-level problem-solving skills to 'Agents' and 'Organizations' with broader capabilities.
   - **Claude** has been highlighted for its superior document comprehension over ChatGPT, suggesting a sharpened focus on context length.
- **Anticipation Peaks for ChatGPT-5**: **GeekyGadgets** spurred conversations with hints about [ChatGPT-5 testing](https://www.geekygadgets.com) commencing late 2024, sparking a mix of excitement and skepticism among users.
   - Anticipated ChatGPT-5 features include enhanced emotional intelligence, reduced instruction repetition, and potential foray into multimodal capabilities.
- **Growing ChatGPT-4o Amnesia Concerns**: Users reported that while **ChatGPT-4o** is speedy, it often forgets recent instructions, questioning its efficacy for tasks like programming.
   - The nostalgia for v3.5's memory highlights a tradeoff between performance speed and operational recall.
- **RAG Chatbot Prompt Tweaking**: Developers are tweaking instructions for a **RAG-based chatbot**, aiming to reduce the chance of receiving odd or contradictory answers.
   - The community recommends refining the clarity of chatbot prompts to ensure effective and logical interactions.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Real-World Applications for Command R Plus Unveiled**: Community members, led by **Mapler**, brainstormed practical applications for the Command R Plus model that spanned content creation for social media, crafting podcast descriptions, and team communication enhancements.
   - Notably, *Sssandra* highlighted her routine integration of the model with Notion and Google Drive to facilitate handling community inquiries.
- **Automatic Updates Revolution with Cohere**: There's ongoing discussion about leveraging the Command R Plus and **Lang chain** to automate AI news delivery via webhooks in Discord, with **Mapler** at the helm of this initiative.
   - **Karthik_99_** has stepped up offering assistance, suggesting that a chat-GPT like interface could be integrated, pending community feedback.
- **Cohere's Toolbox Enters the Open-Source Ecosphere**: *Sssandra* proudly shared the news that [Cohere's chat interface has been open-sourced on GitHub](https://github.com/cohere-ai/cohere-toolkit), teasing imminent OCI integration.
   - **Mapler** responded with eagerness, intending to harness this for personal ventures and to update the community on progress.
- **The Pursuit of AI-Generated Unique Emojis**: **Roazzy** initiated a discussion on developing AI-driven tools for creating distinct emojis, with current methods limited to manual artwork.
   - **Karthik_99_** inquired about existing solutions, emphasizing the feature’s potential within user-driven platforms.
- **Cohere Embedding Model’s Breakthrough in Affordability**: Excitement brewed as a member broadcasted that **Cohere's embedding model** has slashed operational costs by an impressive **40-70%**.
   - The announcement was met with a chorus of approval from the community, echoing a sentiment of appreciation for the cost-effective progress.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Llama's Leap to Larger Learning**: The imminent release of **Llama 3** with a hefty **405B** parameters on **July 23** ignites anticipation, designed as a more robust multimodal model with details previewed in [this briefing](https://www.theinformation.com/briefings/meta-platforms-to-release-largest-llama-3-model-on-july-23).
   - Accelerating the path towards sophisticated AI, the leap from Llama 2 to 3 has sparked conversations around essential infrastructure for support, as posted by [Stephanie Palazzolo](https://x.com/steph_palazzolo/status/1811791968600576271?s=46) and echoed across community channels.
- **OpenAI's Secretive Strawberry Strategy**: Leaks surrounding **OpenAI's Strawberry** project reveal parallels to Stanford's 2022 **STaR** method, spotlighting a progressive advance in AI reasoning technologies as reported by [Reuters](https://www.reuters.com/technology/openais-top-secret-strawberry-project-unveiled-2023-07-15/).
   - The community is abuzz with speculation and analysis of this clandestine endeavor, believing it could mark a significant milestone in OpenAI's quest for more contextual and coherent AI models.
- **Self-hosting Large Models: A Privileged Predicament**: A dive into the logistics of **self-hosting 400B** parameter models unveils the necessity of around **400GB VRAM**, steering the conversation towards resource availability and favoring API usage when proprietary hardware falls short.
   - This predicament places a brighter spotlight on hyperscalers for GPU rental, especially when proprietary data is not a concern, as gleaned from the tech community's dissection of hosting intricacies and API benefits.
- **Distillation Dilemma: Sequential Soft-targeting**: The process of **soft-target distillation** as described in prominent papers is under the microscope, with queries surfacing about the potential for sequential with preserving probabilities.
   - Community input points towards alternative tactics such as aligning internal representations during the online modeling process and how it might simplify current methodology.
- **GPT-4: Valuing Price over Performance?**: Amidst varied offerings in AI services, **GPT-4** emerges as a superior model at **$20 per month**, casting a shadow over competitors' lower-priced alternatives.
   - Comparative discussions are fueled by tweets like those from [aaron holmes](https://x.com/aaronpholmes/status/1811870687037960467?s=46) and spotlight the ongoing discourse on AI model valuation, businesses' choice, and consumer preference.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **tinygrad's Indexing Mastery**: George Hotz introduced a novel [indexing kernel in tinygrad](https://x.com/__tinygrad__/status/1811578147982491938), an unconventional addition bypassing typical constraints by innovatively folding the sum loop.
   - This backend generated approach ensures a strict and efficient subset of memory accesses, **streamlining kernel performance**.
- **Charting a Course with PyTorch-like Roadmaps**: A member proposed emulating the [2024 H2 plan shared by PyTorch](https://dev-discuss.pytorch.org/t/meta-pytorch-team-2024-h2-roadmaps/2226), advocating for a precise and open-ended development strategy.
   - The goal is to mirror PyTorch's transparent pathway, providing a **clear foundation for growth and development**.
- **Gradient Descent Dilemma in tinygrad**: Attempts to implement gradient descent from scratch encountered speed bumps, with a member highlighting the process as being slow without a defined `optimizer.step`.
   - They sought insights on optimizing the sluggish steps, referencing [code samples](https://github.com/karpathy/makemore/blob/988aa59e4d8fefa526d06f3b453ad116258398d4/names.txt) and George Hotz's manual realization tactic.
- **Optimizing Tensor Tactility**: Realizing tensor operations, a vital component for efficient gradient descent, was simplified by Hotz's command `model.weights.assign(model.weights - lr * model.weights.grad).realize()`.
   - Understanding the necessity of realization, as George Hotz puts it, became the **key to computation actualization**.
- **Tackling Tensor Indexing Bugs**: In addressing tensor operations, an assertion error exposed a bug with tensor indexing that led to 'idx.max too big' complications.
   - The engagement in this debugging session highlighted the community's role in **refining tinygrad's kernel efficiency**.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **H100 GPUs Ignite Performance Excitement**: **H100 GPUs** spark a wave of enthusiasm among members, with reactions highlighting the significant leap in **performance capabilities**.
   - The swift performance of the H100 series heralds a new benchmark in computational power, superseding its predecessors with apparent ease.
- **Attention Masking Scrutinized for Reward Models**: The role and impact of **attention masking** in **reward_health_models** surfaced as a topic of debate, with community seeking clarity on its necessity.
   - While questions lingered around its relevance to **axolotl**'s specific training methods, open-ended discussions signaled an ongoing exploration of the technique.
- **OpenRouter Connectivity with OpenArena Spotlighted**: Community members demonstrated interest in integrating **openrouter.ai** APIs for developing an open-source equivalent to the ***WizardLM arena dataset**.
   - One mention highlighted progress using **ollama** for a community-driven [OpenArena project](https://github.com/syv-ai/OpenArena), emphasizing collaborative development.
- **Flash Attention Compatibility Raises Questions**: Compatibility concerns for **Flash attn3** stirred discussions, with limitations noted for **H100 GPUs**.
   - Anticipations are high for a broader GPU support as seen previously with Flash attn2's update catering to *3090's and 4090's*.
- **GaLore vs. Q-Galore: Quantization Takes the Lead**: Discussion highlighted **Q-Galore** as an efficient successor to GaLore, employing quantization techniques to reduce training time, featured in a [Hugging Face paper](https://huggingface.co/papers/2407.08296).
   - Q-Galore's approach, avoiding SVD's time overhead while building on GaLore's strategies, emerged as a significant upgrade for handling gradient subspaces.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **DeepInfra Data Dilemma Discussed**: A member raised concerns about **DeepInfra's** data policy, sparking a discussion on how companies handle training data obtained from user inputs.
   - The discussion led to the clarification that DeepInfra logs usage but doesn't train on user inputs, detailed in their [privacy policy](https://deepinfra.com/privacy).
- **Beta Integrations Beckon Brighter Bots**: **Integrations (Beta)** feature discussions unfolded, focusing on custom API key use for external providers like Groq.
   - Conversations anticipate future expansions may explore integrations beyond model APIs, sparking curiosity about potential applications.
- **Positioning Prompts for Picky Performers**: Members exchanged tips on improving model performance, including suggestion to place text prompts after images to assist weaker models.
   - This placement technique reportedly leads to enhanced comprehension and better responses from less capable models.
- **405b's Arrival Arouses AI Aspirants**: Excitement ignited over the impending release of the **405b model**, with community expectations running high.
   - The community's buzz was fueled by [Bindu Reddy's tweet](https://x.com/bindureddy/status/1811807596682355125?s=46) about the model's anticipated launch, marking July 23rd as significant for open source AGI.
- **Specialization Speculation Stirs Scholars**: A conversation about whether multiple specialized models are superior to a single generic model emerged, featuring companies like OpenAI and Anthropic.
   - **Alex Atallah** joined the debate, advocating for consideration of specialized models and solicited community input on preferred types.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Clip Retrieval Clipped**: A user noted that **clip retrieval** is no longer functional, prompting questions around alternative methods for dataset access.
   - The issue appears linked to the dataset's removal, which could imply a broader restriction on data availability.
- **Memory Hog Models**: An unusually high **19GB memory usage** for a small-scale model training session kindled the guild's interest in memory inefficiencies.
   - The community is actively probing why a mere quarter-million parameter model engulfs so much memory on a modest batch size.
- **Nematron 340B Code Quest**: Queries about **Nematron 340B** code examples burgeoned, focusing on parameter management for the reward model.
   - Details remain sparse, revealing an opportunity for shared coding practices within the guild.
- **Flashy Flow with AuraFlow**: Fal AI's new text-to-image model, **AuraFlow**, brings a splash to the guild with its [launch announcement](https://blog.fal.ai/auraflow/).
   - Its proficiency in adhering to prompts is laid bare as it rejuvenates faith in the **open-source AI landscape**.
- **LLMs' AIW Conundrum**: An updated ArXiv paper showcases the **AIW problem**, revealing a chasm in LLMs' reasoning abilities on rudimentary tasks.
   - Discussions orbit around the inadequacy of current benchmarks and the essential but overlooked capabilities underscored by [the paper](https://arxiv.org/abs/2406.02061).



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Agentic RAG's Recipe for Success**: LlamaIndex, teaming up with AIatMeta, dished out cookbooks on **agentic RAG**, covering multifaceted topics from agent routing to multi-document agent crafting.
   - Eager enthusiasts received a taste with a [Twitter announcement](https://t.co/mBNZx9b1JO) and further insights were served up [here](https://t.co/l2ztPRsAd8).
- **Deduplication Delight via Cypher Snippets**: Crafted by the hands of @tb_tomaz and armory at Neo4j, a potent Cypher snippet simplifies the art of **entity deduplication**, merging tech prowess with URI wizardry.
   - Piquing interest, they shared a practical [example snippet](https://t.co/dAV2QuAoZH) and streamlined access to the code on the [Neo4j GitHub](https://t.co/lMApLzMOMr).
- **Gemini's Call for Functionality Clarification**: Confusion clouded the functionality of function calling on Gemini models; [GitHub commits](https://github.com/run-llama/llama_index/pull/14088) seemed promising yet ran into bumps with error claims stating unsupported API.
   - The path to clarity suggested upgrading the toolkit via `pip install -U llama-index-llms-vertexai`, hoping to clear any fog around **Gemini-1.5-flash-latest's** capabilities.
- **Libraries in Limelight: Indexing Large Codes**: Enthusiasts dissected strategies for indexing hefty code libraries, debating if translating code to markdown-pseudocode could amplify a chatbot's comprehension.
   - The dialogue revolved around the needs of dual chatbot systems, one for question answers and the other for generating code snippets.
- **RAG Reviewing: Skimming Spec Documents**: **RAG's** role in dissecting lengthy spec documents without exhausting token limits was envisioned, seeking efficiency in review processes.
   - The community mulled over methods to evaluate expansive specs, considering the merits of neat token savings alongside **RAG's** potential.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Invocation Error Unraveled**: A user reported **APIConnectionError** when calling `OpenInterpreter.chat()` and 'select' fails to determine the agent's role.
   - The error might be resolved by explicitly passing the **LLM provider**, as suggested in the [documentation](https://docs.litellm.ai/docs/providers).
- **Fast Function Calls with Phi-3**: Excitement arose around **Phi-3** due to its fast and reliable function calls, hinting at potential for fully local Fast Fourier runs.
   - Hopes are high for this optimization that could mean **quicker computations** in the near future.
- **GUI's Glorious Gains**: The **Open Interpreter's GUI** received a significant upgrade, now featuring branching chats, editable messages, auto-run code, and chat saving.
   - The new expansive features come with some limitations, detailed in the [GitHub repository](https://github.com/jbexta/AgentPilot).



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Tackling Telemetry Trouble**: Discussions surfaced around **self-hosted ML telemetry** with a focus on platforms like [Langfuse](https://langfuse.io), [WandB](https://wandb.ai), and [OpenLLMTelemetry](https://github.com/openllmtelemetry).
   - Members stressed the importance of selecting a platform that aligns with specific needs of **ML projects**.
- **API Key Quest for Chatbots**: A member in need for an **OpenAI API key** voiced a request for a chatbot project tutorial, underscoring its short-term necessity.
   - Emphasis was placed on using the API key for demonstration purposes within their tutorial.
- **Chasing Credit Clarifications**: Queries on credit balance were brought up, with one user, **reneesyliu-571636**, directly asking how to perform a **credit balance check**.
   - Another member sought assistance on their account status, possibly hinting at broader questions on the topic of account management.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Advocacy Impacts AI: Llamafile's Leap to Legislation**: **Mozilla's Udbhav Tiwari** advocates for open AI systems before the US Senate, emphasizing the importance of transparent and accessible technologies.
   - The focus at the [senate hearing](https://discord.com/channels/1089876418936180786/1260972784696295536) was on the critical role of openness in AI, aligning closely with Mozilla's own advocacy direction.
- **Builders' Time Extension: Applications Still Welcome!**: Missed the early window for the **Builders Accelerator**? Fear not, applications are indeed still welcome beyond the initial cutoff.
   - Details have been shared earlier, but interested parties can review the program's objectives and apply as mentioned in this [announcement](https://discord.com/channels/1089876418936180786/1089876419926032396/1255588743599882260).
- **Don't Miss Out: AI Events Rendezvous Awaits**: A line-up of intriguing events beckons, including **Open Interpreter with LLMs**, Benjamin Minixhofer's talk on **Zero Shot Tokenizer Transfer**, and an **AutoFix** session with an adept engineer.
   - Eager to partake? Reserve your virtual seat for [upcoming events](https://discord.com/events/1089876418936180786/1260611047341953034) and engage with cutting-edge AI tools and discussions.
- **Chiseling the AI Open Source Definition**: **Open Source AI Definition Draft v 0.0.8** steps into the limelight seeking community insights and aligning with the OECD's AI system interpretation.
   - The community is called to action to review and comment on this evolving artifact on the [OSI's blog](https://blog.opensource.org/open-source-ai-establishing-a-common-ground/).
- **Integer or Float: The Quantization Quandary**: AI engineers ponder whether **llama.cpp** utilizes integer or float calculations for matmul operations, linked to procedures in the **ggml-quants.c**.
   - The mathematical maneuver—a hot topic for the technically inclined—might require quantizing float activations before integer dotproduct machinations ensue.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **LLMs Throwdown** in the OpenArena**: The **LLM Arena** is a combat zone where language models from **Ollama** and **OpenAI** endpoints duel, guided by a third model as judge.
   - The goal is to **increase dataset quality**, demonstrated on [OpenArena's GitHub](https://github.com/syv-ai/OpenArena) through competitive challenges.
- **WizardLM Paper** casts spell on Arena Learning**: **OpenArena** draws its inspiration from the [WizardLM paper](https://www.microsoft.com/en-us/research/publication/arena-learning-build-data-flywheel-for-llms-post-training-via-simulated-chatbot-arena/), advocating **Arena Learning** post-LLM training.
   - By simulating chatbot battles and utilizing **AI-annotated datasets**, the approach sharpens models via supervised fine-tuning and reinforcement learning techniques.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Expanding Horizons in MLOps**: A discussion was initiated with an interest in covering diverse areas such as **product** and **research**, particularly in **recommendation systems**, **information retrieval (IR)**, and **retrieval-augmented generation (RAG)**.
   - The conversation encouraged open suggestions and expressed a specific interest in exploring **Elastic** and its potential in these areas.
- **Elastic Enthusiasts Emerge**: Another user echoed the sentiment expressing their willingness to have a detailed dialogue about **Elastic**.
   - The user tagged a colleague to kickstart a deeper discussion on how **Elastic** could enhance their current operations.



---


The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Perf Enthusiasts AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI Stack Devs (Yoko Li) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Torchtune Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1261053927076659292)** (1 messages): 

> - `qdurllm demo`
> - `Advanced RAG workshop`
> - `Intel HF model repository`
> - `Self-reviewing coding assistant`
> - `Training chatbot with LlamaIndex` 


- **qdurllm demo refreshes**: [qdurllm demo](https://huggingface.co/spaces/as-cle-bert/qdurllm-demo) by a community member showcases new capabilities with intuitive output.
- **Future of AI with Knowledge Graphs**: An online workshop titled [Leveraging Knowledge Graphs for Advanced RAG](https://www.youtube.com/watch?v=9wqVz0LDYgg) discusses **natural language querying** using Langchain and Neo4j.
   - It provides insights on interacting with graph databases using **Cypher query language**.
- **Intel CPUs maximize HF model efficiency**: A new [GitHub repository](https://github.com/sleepingcat4/intel-hf) demonstrates methods for running any **Hugging Face model** efficiently on **Intel CPUs**.
- **gary4live plugin available**: The **gary4live** Ableton plugin is now available for free on Gumroad, as announced [here](https://x.com/thepatch_kev/status/1810063563823907172).
- **Read about RegMix**: [RegMix](https://huggingface.co/blog/SivilTaram/regmix) introduces a new method using **Data Mixture** for effective **language model pre-training**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/as-cle-bert/qdurllm-demo">Qdurllm Demo - a Hugging Face Space by as-cle-bert</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=9wqVz0LDYgg&ab_channel=DecodingDataScience)">The Future of AI: Leveraging Knowledge Graphs for Advanced RAG</a>: Get ready to dive into the world of natural language querying with Langchain and Neo4j! Learn how to interact with graph databases using cypher query languag...</li><li><a href="https://wandb.ai/sauravmaheshkar/llamaindex-local-models-index/reports/Training-a-chatbot-on-personal-data-with-LlamaIndex-and-W-B--Vmlldzo4MzQzMDE3)">Weights & Biases</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://x.com/thepatch_kev/status/1810063563823907172)">Tweet from thecollabagepatch (@thepatch_kev)</a>: 13 legends just got an email for  gary4live  the ableton plugin that does this  dl on gumroad rn u guys  ⬇️link  @_buildspace @_nightsweekends</li><li><a href="https://youtu.be/38ae7hqzX5s)">Gemma2:27 Ollama Correction ! Now Incredible !</a>: Today, we are going to test again gemma 2 27b with ollama because an update was pushed by ollama to correct issues related to gemma 2 and now it is working l...</li><li><a href="https://youtu.be/gAtUdnN1_xM?si=L_1vdbjzu4yHyUlA)">Intro to SK-LEARN By Rauf</a>: A short basic introduction to the scikit-learn (sklearn) machine learning library. I initially created this for my presentation, but I realized it would be f...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1261036924215234692)** (613 messages🔥🔥🔥): 

> - `GPU Models and Issues`
> - `Cloud and Free Resources`
> - `Training Techniques`
> - `HF Integrations`
> - `Jokes and Community Engagement` 


- **Discussion on GPU models and funding**: Members discussed the technical and emotional loss associated with various GPU models, such as the **1060 3GB** and the potential replacements like the **A6000** for better rendering capabilities.
   - Budget constraints led to the consideration of options like **Facebook Marketplace salvages** and freelancing for extra funds.
- **Exploring Cloud and Free Computational Resources**: The conversation detailed the cost and utility of **A100** GPUs for training, with recommendations for **backprop.co** and free **Google Colab T4** instances for more economical usage.
   - Discussions included Google Cloud's **TPU research credits**, which offer free access to clusters for eligible projects.
- **Training Diffusion Models and LoRa Techniques**: Members faced challenges in training Diffusion models, with mentions of using **LoRa on cheaper GPUs** and the complications of **full finetuning on A100** due to cost.
   - Guidance was provided on renting smaller GPUs and exploring **Colab** for more economical options, specifically for **character style transfer**.
- **HF Integration and Updates**: [Hugging Face](https://huggingface.co/transformers) updates were shared, including **GGUF support in transformers** and integration with **KerasNLP** models.
   - New features like **TPU support in Inference Endpoints** were also highlighted, broadening the scope of applications for HF models.
- **Humor and Community Engagement**: Members engaged in a light-hearted conversation about the hypothetical impact of cheese on servers, invoking humor around fondue and GPUs.
   - Other amusing interactions included giving and receiving advice, discussing personal situations, and general banter about everyday tech struggles.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://sites.research.google/trc/about/">TPU Research Cloud - About</a>: no description found</li><li><a href="https://huggingface.co/blog/lora">Using LoRA for Efficient Stable Diffusion Fine-Tuning</a>: no description found</li><li><a href="https://huggingface.co/spaces/google/sdxl">Stable Diffusion XL on TPUv5e - a Hugging Face Space by google</a>: no description found</li><li><a href="https://huggingface.co/spaces/bishmoy/Arxiv-CS-RAG">Arxiv CS RAG - a Hugging Face Space by bishmoy</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=4Wa5DivljOM">Why you&#39;re addicted to cloud computing</a>: Learn how big cloud providers like AWS, Microsoft Azure, and Google Cloud operate from a business perspective. Explore strategies for optimizing cloud comput...</li><li><a href="https://youtu.be/KyOlpzA5jKM">[HQ RECREATION] Wait, is that Gabe?</a>: Recreation cause I didn’t see it anywhere else on YouTubehttps://www.youtube.com/watch?v=ELtzcpb_j38This is the high quality original version of this meme. S...</li><li><a href="https://tenor.com/view/mmm-what-shocked-monster-inc-james-p-sullivan-gif-14562553">Mmm What GIF - Mmm What Shocked - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/%D0%B2%D0%B7%D0%B3%D0%BB%D1%8F%D0%B4-2000-%D1%8F%D1%80%D0%B4%D0%BE%D0%B2-%D0%B2%D0%BE%D0%B9%D0%BD%D0%B0-war-soldier-gif-3632617944134077161">взгляд 2000 ярдов GIF - Взгляд 2000 ярдов Война - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/gabe-newell-gaben-gabe-newell-gif-18366858729810314226">Gabe Newell Gaben GIF - Gabe newell Gaben Gabe - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/bonk-gif-26414884">Bonk GIF - Bonk - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/stewie-family-guy-rip-sad-funeral-gif-13648662">Stewie Family Guy GIF - Stewie Family Guy Rip - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://backprop.co">Backprop GPU Cloud</a>: no description found</li><li><a href="https://huggingface.co'">no title found</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/installation#offline-mode'.">Installation</a>: no description found</li><li><a href="https://github.com/dykyivladk1/polip">GitHub - dykyivladk1/polip: Library designed for better experience in training NNs</a>: Library designed for better experience in training NNs - dykyivladk1/polip</li><li><a href="https://github.com/karpathy/llm.c">GitHub - karpathy/llm.c: LLM training in simple, raw C/CUDA</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://tenor.com/view/dance-meme-caption-fat-herobrine-herobrine-gif-22298550">Dance Meme GIF - Dance Meme Caption - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e1dudw/from_cl%C3%A9ment_delangue_on_x_hugging_face_is/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://tenor.com/view/huh-cat-gif-26460616">Huh Cat GIF - Huh Cat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/fchollet/status/1811104960303747529">Tweet from François Chollet (@fchollet)</a>: You can now use any Hugging Face Hub model with KerasNLP (as long as the corresponding architecture is in KerasNLP)! What&#39;s more, you can also upload your own fine-tuned KerasNLP models to Hugging...</li><li><a href="https://github.com/huggingface/transformers/releases/tag/v4.41.0">Release v4.41.0: Phi3, JetMoE, PaliGemma, VideoLlava, Falcon2, FalconVLM &amp; GGUF support · huggingface/transformers</a>: New models Phi3 The Phi-3 model was proposed in Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone by Microsoft. TLDR; Phi-3 introduces new ROPE scaling methods, which se...</li><li><a href="https://docs.coqui.ai/en/latest/tutorial_for_nervous_beginners.html">Tutorial For Nervous Beginners - TTS 0.22.0 documentation</a>: no description found</li><li><a href="https://huggingface.co/parler-tts/parler-tts-mini-expresso">parler-tts/parler-tts-mini-expresso · Hugging Face</a>: no description found</li><li><a href="https://github.com/rhasspy/piper">GitHub - rhasspy/piper: A fast, local neural text to speech system</a>: A fast, local neural text to speech system. Contribute to rhasspy/piper development by creating an account on GitHub.</li><li><a href="https://github.com/huggingface/tokenizers/pull/1493">Add more support for tiktoken based tokenizers by ArthurZucker · Pull Request #1493 · huggingface/tokenizers</a>: Adds a check before using merges, returing the token if it is part of the vocab</li><li><a href="https://www.warp.dev/?utm_source=its_foss&utm_medium=display&utm_campaign=linux_launch">Warp: Your terminal, reimagined</a>: Warp is a modern, Rust-based terminal with AI built in so you and your team can build great software, faster. Now available on MacOS and Linux.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1261059714129592440)** (6 messages): 

> - `Embedding models using mouse movements`
> - `Transfer learning in triplet loss`
> - `Classification objectives in contrastive learning`
> - `Sampling rates and batch sizes`
> - `Knowledge graphs implementation` 


- **Mouse movements for identification**: A member worked on an embedding model to identify individuals based on their **mouse movements**, utilizing **triplet loss** to train the model.
   - They described the process of comparing embeddings via **euclidean distance** and discussed using **transfer learning** to avoid local minima loss.
- **Improving contrastive learning objectives**: [Tips on improving contrastive learning](https://arxiv.org/pdf/2309.12871) objectives included adjusting **sampling rates** of mouse pointers and using large batch sizes for better convergence.
   - Suggestions also included trying the **AnglE objective** and examining potential influences of normalization layers to prevent zero-embeddings.
- **Support Vector Machine basics explained**: A [YouTube video](https://youtu.be/DOXF0fKqUIU?si=oIAJqjjHss25cw6R) was shared to explain **Support Vector Machines (SVMs)** from basics.
   - The video aimed to simplify SVM concepts and provided another link for understanding **SK-Learn**.
- **Implementing knowledge graphs**: A query about **knowledge graphs** led to mentions of the **Neo4j library** as a common resource for actual implementations.



**Link mentioned**: <a href="https://youtu.be/DOXF0fKqUIU?si=oIAJqjjHss25cw6R">Support Vector Machine SVM ( Machine Learning pt 3 )</a>: In this video i try to explain SVMs from the basic and I try to make it easy and simple ,  if ya wanna know about the SK-Learn click this : https://youtu.be/...

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1261157311582572574)** (2 messages): 

> - `Supervised fine-tuning in TRL`
> - `Ripple_net library for search engines` 


- **SFT in TRL simplifies model fine-tuning**: [Supervised fine-tuning](https://huggingface.co/docs/trl/en/sft_trainer) (SFT) is a crucial step in **RLHF**, and TRL offers an easy-to-use API to create and train SFT models.
- **Ripple_net makes waves in search tech**: A user shared a **text-image search and tagging library** called [ripple_net](https://github.com/kelechi-c/ripple_net) on GitHub.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/kelechi-c/ripple_net">GitHub - kelechi-c/ripple_net: text-image search and tagging library</a>: text-image search and tagging library. Contribute to kelechi-c/ripple_net development by creating an account on GitHub.</li><li><a href="https://huggingface.co/docs/trl/en/sft_trainer">Supervised Fine-tuning Trainer</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1261042529483231292)** (11 messages🔥): 

> - `mypo dataset`
> - `Indonesian Hate Speech dataset`
> - `ripple_net library`
> - `RAG app for PDFs`
> - `Support Vector Machine SVM video` 


- **Discussing the mypo dataset for Python code quality**: A user shared a preview of the mypo dataset focusing on Python code quality and requested feedback on using this approach to improve Python LLMs with type hints.
   - The dataset, which contains 600M rows of Reddit data from 2024, aims to enhance Python LLM's default usage of type hints and other coding standards.
- **Indonesian Hate Speech dataset presented**: A member promoted their paper and dataset for Indonesian Hate Speech, available on [Huggingface](https://huggingface.co/datasets/Exqrch/IndoToxic2024), emphasizing the importance of considering reader demographics in hate speech detection.
   - Findings showed that models like **gpt-3.5-turbo** improve with demographic info, while **IndoBERTweet** performance suffers due to the limited training data.
- **Introducing ripple_net for text-image search**: A user announced the creation of **ripple_net**, a Python library for text/image-based search in image datasets, shared on [GitHub](https://github.com/kelechi-c/ripple_net).
   - The library allows for efficient text-image search and tagging of images, providing a valuable tool for dataset management.
- **Built RAG app for PDFs**: Another user showcased their **RAG app** for PDFs, which can be accessed on [Huggingface](https://huggingface.co/spaces/tensorkelechi/studyassist).
   - The app leverages AI for document chat and study assistance, with more details available in their [GitHub repository](https://github.com/kelechi-c/studyassist).
- **Explanation of Support Vector Machine**: A YouTube video titled [Support Vector Machine SVM](https://youtu.be/DOXF0fKqUIU?si=oIAJqjjHss25cw6R) was shared to explain SVMs in machine learning.
   - The video aims to simplify the concept and includes a link for further information about **SK-Learn**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/tensorkelechi/studyassist">Studyassist - a Hugging Face Space by tensorkelechi</a>: no description found</li><li><a href="https://youtu.be/DOXF0fKqUIU?si=oIAJqjjHss25cw6R">Support Vector Machine SVM ( Machine Learning pt 3 )</a>: In this video i try to explain SVMs from the basic and I try to make it easy and simple ,  if ya wanna know about the SK-Learn click this : https://youtu.be/...</li><li><a href="https://github.com/kelechi-c/studyassist">GitHub - kelechi-c/studyassist: AI + RAG chat for documents</a>: AI + RAG chat for documents. Contribute to kelechi-c/studyassist development by creating an account on GitHub.</li><li><a href="https://github.com/kelechi-c/ripple_net">GitHub - kelechi-c/ripple_net: text-image search and tagging library</a>: text-image search and tagging library. Contribute to kelechi-c/ripple_net development by creating an account on GitHub.</li><li><a href="https://huggingface.co/datasets/terminusresearch/ideogram-75k">terminusresearch/ideogram-75k · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/joshuasundance/mypo-4k-rfc">joshuasundance/mypo-4k-rfc · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/OpenCo7/UpVoteWeb">OpenCo7/UpVoteWeb · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1261284904532967534)** (3 messages): 

> - `Paper Plans`
> - `Transformer Performance`
> - `New LLM Paradigm` 


- **Planning New Paper**: A member inquired about the paper being planned by another member in the group.
   - Another member mentioned they might share their paper titled '2406.06612' if it sounds good to the group.
- **Transformers Lose to 20 Epochs**: A member claimed that running 20 epochs performs **10% better** compared to a transformer.
   - "I will show you guys a new llm paradigm," they emphasized their belief in this new approach despite some showing skepticism with a 😕 emoji.


  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1261305479041126421)** (1 messages): 

> - `AuraFlow model`
> - `LoRA support`
> - `Offloading at the modeling level`
> - `State-of-the-art results on GenEval`
> - `Community feedback` 


- **AuraFlow: Largest Open Text-to-Image Model Launch**: Shoutout to developers for **AuraFlow**, the largest open text-to-image model with an Apache 2.0 license, now supported in `diffusers`.
   - Check out [AuraFlow](https://huggingface.co/fal/AuraFlow) for state-of-the-art results on GenEval, with more development updates to come.
- **LoRA Support Coming Soon**: Upcoming updates will add **LoRA support** to AuraFlow, allowing users to experiment with training and more features.
   - Join [fal's Discord](https://discord.gg/fal-ai) to give feedback and stay connected with the development.
- **Efficient Use of VRAM with Offloading**: A new PR enables running the **Aura Flow Transformer model** in 15GBs of VRAM by offloading at the modeling level.
   - See details in the [GitHub PR #8853](https://github.com/huggingface/diffusers/pull/8853).
- **Community Involvement is Crucial**: The **AuraFlow model** is in beta and community feedback is essential for improvements.
   - Credits go to [@cloneofsimo](https://twitter.com/cloneofsimo) and [@isidentical](https://twitter.com/isidentical) for their significant contributions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/huggingface/diffusers/pull/8853.">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://huggingface.co/fal/AuraFlow">fal/AuraFlow · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1261227017706143827)** (3 messages): 

> - `RT-DETR Object Detection`
> - `Hiera Vision Transformer` 


- **RT-DETR Outperforms YOLO**: In a [collaboration with Roboflow](https://blog.roboflow.com/train-rt-detr-custom-dataset-transformers/), **RT-DETR** is a computer vision model developed by Peking University and Baidu that outperforms YOLO in object detection, both in speed and accuracy.
   - The [paper](https://arxiv.org/abs/2304.08069?ref=blog.roboflow.com) asserts **RT-DETR's** superiority, and it has been added to the [transformers](https://github.com/huggingface/transformers?ref=blog.roboflow.com) library, simplifying fine-tuning.
- **New Vision Transformer: Hiera**: **Hiera**, a new hierarchical vision transformer model, has been added to the transformers library, achieving better performance while simplifying complexities usually associated with hierarchical vision transformers.
   - *HieraForImageClassification*, *HieraModel*, *HieraForPreTraining*, and *HieraBackbone* are available, providing versatile applications including image classification and feature extraction. [GitHub Pull Request](https://github.com/huggingface/transformers/pull/30356#event-13478146443).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.roboflow.com/train-rt-detr-custom-dataset-transformers/">How to Train RT-DETR on a Custom Dataset with Transformers</a>: Learn how to train RT-DETR on a custom dataset using the Transformers library.</li><li><a href="https://github.com/huggingface/transformers/pull/30356#event-13478146443">Adding hiera by Namangarg110 · Pull Request #30356 · huggingface/transformers</a>: What does this PR do? Adds Hiera model from Meta as suggested in #28993 GitHub Repo: https://github.com/facebookresearch/hiera/ arXiv: https://arxiv.org/abs/2306.00989 Model License recently change...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1261409359485866116)** (4 messages): 

> - `LLM Finetuning Toolkit`
> - `Phi-3 models discussion`
> - `Multimodal image RAG` 


- **Lightweight LLM Finetuning Toolkit Released**: [Georgian-io](https://github.com/georgian-io/LLM-Finetuning-Toolkit) introduced a lightweight, config-driven tool for launching finetuning experiments across open-source LLMs, designed with an end-to-end Data Science experimentation pipeline in mind.
   - The toolkit allows running multiple experiments through a single config file, running evaluation metrics on eval sets, and performing ablation studies to try out different configurations like hyperparameters and prompts.
- **Debating Phi-3 Models on vCPU**: A new member asked whether **microsoft/Phi-3-mini-4k-instruct** could be used on a **vCPU** environment, citing errors with the **onnx** implementation and inquiring about correct settings for device maps.
   - *The background was I was trying to finetune an open-source model, but without a GPU, it seems like a pain...*
- **Best Practice for Multimodal Image RAG**: A member queried whether it's better to embed images as they are or generate descriptions for the images and then embed those descriptions when performing **multimodal image RAG**.
   - No specific answer was provided, highlighting a need for more input or community discussion.



**Link mentioned**: <a href="https://github.com/georgian-io/LLM-Finetuning-Toolkit">GitHub - georgian-io/LLM-Finetuning-Toolkit: Toolkit for fine-tuning, ablating and unit-testing open-source LLMs.</a>: Toolkit for fine-tuning, ablating and unit-testing open-source LLMs. - georgian-io/LLM-Finetuning-Toolkit

  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1261434841715703869)** (1 messages): 

> - `Architecture Explanation`
> - `Implementation from Scratch` 


- **Request for Architecture Explanation**: A member asked for an explanation of a certain **architecture** and how it is working.
   - They also requested guidance on implementing this **architecture** from scratch.
- **Implementation Guidance Needed**: The member emphasized the need for detailed guidance on **implementing the architecture from scratch**.
   - This request indicates they are seeking step-by-step instructions on the **implementation process**.


  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1261043102353719296)** (341 messages🔥🔥): 

> - `Reactor installation for ComfyUI`
> - `Deforum Stable Diffusion techniques`
> - `AIl troubleshooting delays`
> - `Model merging and performance`
> - `Effective image upscalers` 


- **Troubleshoot Reactor for ComfyUI Installation**: A user suggested following a [YouTube video](https://www.youtube.com/watch?v=vCCVxGtCyho) for fast installation of ComfyUI InsightFace, which reportedly resolves errors.
   - The video includes detailed instructions still functional as of 2024, and another user confirmed it worked for them.
- **Deforum Stable Diffusion Color Transitions**: A member asked for tips on creating clear color transitions in Deforum Stable Diffusion for an abstract video when using the API version.
   - They are considering setting color_coherence to None to achieve better results and seeking additional insights.
- **Prevents Delays in Interrupting Generation on Auto1111**: A discussion on why it takes long to interrupt generation on auto1111, pointing to VRAM and inherent issues with the software.
   - A user compared the delay to a train needing to ramp down after running.
- **Cost of AI Tools vs. Local Use**: Members discussed the expense of using commercial AI tools like Runway, which has a steep $90/mo plan despite its effective outpainting and TXT2VID features.
   - While some users prefer local, free tools, they acknowledged that paid tools often provide superior results and features.
- **Find Free Upscaling Tools**: Members sought recommendations for free creative image upscalers.
   - Alternatives like Krita and GIMP were preferred due to their accessibility and useful features.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.fal.ai/auraflow/">Introducing AuraFlow v0.1, an Open Exploration of Large Rectified Flow Models</a>: Open-source AI is in jeopardy. As community interest in AI models skyrocketed over the past year, we noticed that development of new open-source foundational models came to a halt. Some even boldly an...</li><li><a href="https://www.youtube.com/watch?v=vCCVxGtCyho&">ComfyUI InsightFace Windows Fast Installation (2024) | NO MORE ERRORS FOR IPADAPTERS / ROOP</a>: ComfyUI: https://github.com/comfyanonymous/ComfyUIInsightFace Wheels: https://github.com/Gourieff/Assets/tree/main/InsightfaceCommands: .\python_embeded\pyth...</li><li><a href="https://youtu.be/4U9MI0u2VIE">Hackers -- Cyberdelia --- Crayola Books</a>: Cool scene from an excellent movie.</li><li><a href="https://www.getpaint.net/download.html">Paint.NET - Download</a>: no description found
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1261044532099940413)** (13 messages🔥): 

> - `FA3 collaboration`
> - `H100 deployment`
> - `Warpgroup Pingponging`
> - `Support for Ampere`
> - `Discord functionality` 


- **FA3 collaboration showcased**: There's excitement about the **FA3 collaboration** involving **Colfax**, **Tri Dao**, **Meta AIT team**, and the **Cutlass team**.
- **H100 excitement**: A user expressed enthusiasm with 'H100 go brrrrr,' indicating strong performance of the **NVIDIA H100** hardware.
- **Warpgroup Pingponging trick**: Users discussed the **warpgroup pingponging trick** from the FA3 paper and how it handles transposing **V for FP8**, with the code release anticipated shortly.
   - Users expressed excitement, with one congratulating the team and others curiously asking about future **Ampere support**.
- **Discord permissions and content**: Users are experiencing issues accessing content outside the **general channel** in Discord and are troubleshooting by refreshing the page.
   - A user mentioned the **Events tab** might be empty now but weekly talks can be found on their [YouTube channel](https://discord.com/channels/1189498204333543425/1189640399476764692/1246559662871150603).


  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1261121268132876309)** (7 messages): 

> - `ResNet18 on A100 vs A40`
> - `torch.compile max-autotune issue`
> - `Floating point errors` 


- **Accuracy drop switching from A100 to A40**: A member highlighted a slight **0.26% drop in accuracy** when switching from an **A100 to an A40** for inference using the **ResNet18 model**.
   - This raises concerns about **floating point errors** or hardware-specific kernel optimizations influencing results.
- **Max-Autotune causing significant accuracy loss**: Running inference with **torch.compile(mode='max-autotune')** introduced a **1.44% loss in accuracy** on an **A40**.
   - Even when running inference with max-autotune on the same device (A100), the model still showed a **0.26% decrease in accuracy**.
- **Floating point errors suspected in accuracy loss**: **Floating point errors** are suggested as a possible reason for the **accuracy degradation** when using different hardware or **torch.compile** settings.
   - *


  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1261353333982105630)** (3 messages): 

> - `Q-Galore paper`
> - `Llama3 405B release`
> - `LoQT paper` 


- **Q-Galore reduces memory usage for LLMs**: The new [Q-Galore method](https://arxiv.org/abs/2407.08296) combines quantization and low-rank projection to reduce memory usage while training Large Language Models, outperforming GaLore.
   - Unlike GaLore, Q-Galore eliminates time-consuming Singular Value Decomposition (SVD) operations, leading to more efficient training.
- **Llama3 405B dropping July 23**: [Llama3 405B](https://x.com/steph_palazzolo/status/1811791968600576271) is set to release on July 23, a year after the announcement of Llama2.
   - The new multimodal model is expected to be Meta's largest to date, with more details available in [this briefing](https://www.theinformation.com/briefings/meta-platforms-to-release-largest-llama-3-model-on-july-23).
- **LoQT enables efficient training on consumer hardware**: [LoQT](https://arxiv.org/abs/2405.16528) efficiently trains quantized models by using gradient-based tensor factorization for low-rank weight matrices, suitable for both pretraining and fine-tuning.
   - This method allows for the training of models up to 7B parameters on a consumer-grade 24GB GPU and demonstrates the feasibility of training a 13B parameter model.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.08296">Q-GaLore: Quantized GaLore with INT4 Projection and Layer-Adaptive Low-Rank Gradients</a>: Training Large Language Models (LLMs) is memory-intensive due to the large number of parameters and associated optimization states. GaLore, a recent method, reduces memory usage by projecting weight g...</li><li><a href="https://arxiv.org/abs/2405.16528">LoQT: Low Rank Adapters for Quantized Training</a>: Training of large neural networks requires significant computational resources. Despite advances using low-rank adapters and quantization, pretraining of models such as LLMs on consumer hardware has n...</li><li><a href="https://x.com/steph_palazzolo/status/1811791968600576271">Tweet from Stephanie Palazzolo (@steph_palazzolo)</a>: A Friday scooplet w/ @SylviaVarnham — Llama 3 405B is coming (and soon!)  The multimodal model is set to drop on July 23, about a year after the Llama 2 announcement.  More details here:  https://www....
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1261415740133609613)** (18 messages🔥): 

> - `GPU access`
> - `Google Colab`
> - `Coreweave and Lambda Labs`
> - `Google Cloud GPU`
> - `Nsight Compute` 


- **Colab praised for easy GPU access**: Members recommend using **Google Colab** for testing and running assignments as it provides free GPU access without requiring CUDA driver setup.
   - One user mentioned that **Nsight Compute** works on Colab, though spawning a window may not be possible.
- **Coreweave and Lambda Labs evaluated**: Members discussed whether **Coreweave** or **Lambda Labs** are good alternatives for GPU rental.
   - Concerns about Coreweave being pricey and Lambda Labs having difficult allocations were noted, especially for testing specific kernels like Hopper or Ada.
- **Google Cloud GPU versus Colab**: When asked about **Google Cloud GPU** or **SageMaker**, members acknowledged they are pricier but better if you need to use things other than notebooks.
   - One member opined that for simply tinkering with CUDA kernels and learning, **Colab** is less of a hassle than **GCP**.


  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1261265703902642196)** (6 messages): 

> - `Matrix-Matrix Multiplication in CUDA`
> - `Thread Assignment in Matmul Kernels`
> - `Data Access Patterns in CUDA` 


- **Thread Assignment Considerations**: A member asked about the pros and cons of assigning a thread per row vs a thread per column in matrix-matrix multiplication.
   - Another member explained that assigning a thread per row is more efficient due to the row-major format of 2D matrices in memory, leading to fewer cache misses and better data loading.
- **Memory Coalescing and CUDA Efficiency**: A detailed response noted that indexing over columns requires jumping the entire row length in memory, making it inefficient compared to rows.
   - The concept of 'coalescing' was mentioned, explaining that reducing over the last dimension is more efficient and extends the understanding of memory arrangement in CUDA.


  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1261375444612157501)** (3 messages): 

> - `Tensor Subclass Support`
> - `Bitnet Work`
> - `FSDP and Distributed Training`
> - `Developer Guide` 


- **Tensor Subclass + Distributed Training Not Prioritized**: One member mentioned a plan to create a developer guide on enabling **tensor subclasses** and **distributed training/inference** (FSDP, FSDP2, DTensor), but it's not prioritized due to a lack of concrete use cases.
   - However, they are open to starting work if applications like **distributed inference with fp4** are needed.
- **Concrete Use Cases for Tensor Subclass Emerging**: Another member feels that concrete use cases for **tensor subclass and FSDP** are emerging, citing projects like **bitnet** and related work with FSDP.
   - They mentioned **q galore** as another potential use case.
- **Collaborative Developer Guide Creation**: One member agreed to collaborate on creating a developer guide for **tensor subclass and distributed training** if needed.
   - *We can work together to fill out the developer guide in the process.*


  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1261035639189540914)** (176 messages🔥🔥): 

> - `FA3 vs cuDNN vs ThunderKittens`
> - `Unified Memory impact`
> - `Fine-tuning large models`
> - `FP8 optimizations`
> - `LLM.C community initiatives` 


- **FA3 versus cuDNN and ThunderKittens**: **FA3** came under discussion for its potential speed-up in attention mechanisms, but members debated between it and alternatives like **cuDNN** and **ThunderKittens** for their complexity and ease of use.
   - Issues like **maintenance complexity**, especially with FP8 not supported by TK, were significant points. One user mentioned, *'TK doesn't have an FP8 path and won't anytime soon.'*
- **Impact of Unified Memory on performance**: There was a technical debate on whether **Unified Memory** impacts performance when only accessed by the GPU, especially concerning its use for optimizer states.
   - It was suggested to also consider **zero-copy memory** as an alternative: *'Why not use zerocopy memory and write kernel to read directly from the system memory?'*
- **Challenges in fine-tuning large models**: Fine-tuning discussions centered on successfully running and optimizing large models like **300B** and **330B** checkpoints.
   - One member reported **62.7** on HellaSwag after annealing the learning rate for **30B steps**: *'i'll upload that.'*
- **FP8 optimizations yield significant speed-up**: Implementing **FP8** optimizations resulted in a **33% speed-up** compared to **BF16**, increasing from ~30.8K token/s to 40.9K token/s on a usage test.
   - Challenges remain, like **memory reuse and overflow issues**, but progress continues: *'will need some discussion on refactoring.'*
- **LLM.C community and resource sharing**: New organizational initiatives were started on **Hugging Face**, where members can share trained models and contribute.
   - Members discussed benchmarks, sharing checkpoints, and planning community demos: *'Added you (as admin)! It's basically the same as your hf profile.'*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/datasets/karpathy/fineweb-edu-100B-gpt2-token-shards">karpathy/fineweb-edu-100B-gpt2-token-shards · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/llmc/llmc_1558M">llm.c 1558M demo - a Hugging Face Space by llmc</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/pull/682">Add a README link under related related projects for gpu.cpp under WebGPU C++ by austinvhuang · Pull Request #682 · karpathy/llm.c</a>: Add a README link under related related projects for gpu.cpp under WebGPU C++. For background - gpu.cpp is a new project I&#39;ve been working on. It&#39;s a small library for writing portable GPU cod...</li><li><a href="https://github.com/karpathy/llm.c/pull/678">FP8 work in progress by ademeure · Pull Request #678 · karpathy/llm.c</a>: This is the current state of my FP8 branch, it&#39;s far from ready, but it&#39;s at the point where you could take a look if you&#39;re curious! The last version which was functionally correct was f7...</li><li><a href="https://news.ycombinator.com/item?id=40939707">Karpathy: Let&#x27;s reproduce GPT-2 (1.6B): one 8XH100 node 24h $672 in llm.c | Hacker News</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/pull/650">muP (maximum update parametrization) by gordicaleksa · Pull Request #650 · karpathy/llm.c</a>: Main changes:  Modify random initialization Scale attention scores by 1/d and not 1/sqrt(d) and add an attn_mult Scale activations by 1/width_mult before mapping into logits Update learning rate &amp;...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[youtube-watch-party](https://discord.com/channels/1189498204333543425/1238931064223830016/)** (1 messages): 

vkaul11: Hi
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1261345601099010109)** (189 messages🔥🔥): 

> - `Primeagen Video Discussion`
> - `REPL Behavior in Mojo`
> - `Mojo Community Meeting`
> - `GIL Removal in Python 3.13`
> - `Comparative Network Speed in Different Languages` 


- **Edited Primeagen Video Release Available**: Members discussed the new [Primeagen video](https://www.youtube.com/watch?v=ovYbgbrQ-v8) titled 'I Interviewed The Creator Of LLVM, Clang, Swift, and Mojo,' which is available without a paywall on YouTube.
   - *'Yes to watch the entire livestream you need to be a member. The edited version just released a couple hours ago.'*
- **Mojo REPL Should Show Immediate Output**: Member expressed frustration over Mojo REPL not automatically showing the output of expressions like `1+1`, suggesting it should behave more like Python's REPL.
   - Another member acknowledged the concern, explaining that Mojo handles memory differently than Python and does not have this feature yet, but suggested raising a GitHub issue.
- **Mojo Community Meeting Scheduled**: The 4th Mojo Community meeting is scheduled and will cover topics including forge_tools, flat buffers, and generics, accompanied by a Q&A session with the Modular team.
   - Zoom and meeting information links were provided for members to join the session and participate actively.
- **Python 3.13's GIL Removal and JIT Optimization**: Discussion around Python 3.13's *'no-GIL'* beta and JIT optimization revealed that even with GIL removal, Python's performance remains slow compared to Rust and Node.js.
   - One member noted that *


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/lib">Mojo🔥 modules | Modular Docs</a>: A list of all modules in the Mojo standard library.</li><li><a href="https://www.youtube.com/watch?v=ovYbgbrQ-v8">I Interviewed The Creator Of LLVM, Clang, Swift, and Mojo</a>: Recorded live on twitch, GET IN ### GuestsChris Lattner https://x.com/clattner_llvm?s=21&amp;t=-sv4MdpmLrRuMIhARbLk-ghttps://www.modular.comTJ DeVrieshttps://you...</li><li><a href="https://www.youtube.com/watch?v=HxSHIpEQRjs&pp=ygUKcHl0aG9uIGppdA%3D%3D">Brandt Bucher – A JIT Compiler for CPython</a>: From the 2023 CPython Core Developer SprintThe QA section is hard to understand; turn on subtitles for our best-effort transcription. (PRs welcome: https://g...</li><li><a href="https://docs.python.org/3.13/whatsnew/3.13.html#free-threaded-cpython">What’s New In Python 3.13</a>: Editor, Thomas Wouters,. This article explains the new features in Python 3.13, compared to 3.12. For full details, see the changelog. Summary – Release Highlights: Python 3.13 beta is the pre-rele...</li><li><a href="https://github.com/modularml/mojo/issues/2809">[Feature Request] Use Python-like behaviour in REPL (interactive session) to input commands and print the evaluation · Issue #2809 · modularml/mojo</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? In Python&#39;s interactive console, the last (or only...</li><li><a href="https://stackoverflow.com/questions/1301346/what-is-the-meaning-of-single-and-double-underscore-before-an-object-name)">What is the meaning of single and double underscore before an object name?</a>: What do single and double leading underscores before an object&#x27;s name represent in Python?</li><li><a href="https://modul.ar/community-meeting-zoom">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...</li><li><a href="https://modul.ar/community-meeting-doc">[Public] Mojo Community Meeting</a>: Mojo Community Meeting This doc link: https://modul.ar/community-meeting-doc  This is a public document; everybody is welcome to view and comment / suggest.  All meeting participants must adhere to th...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[polls](https://discord.com/channels/1087530497313357884/1123829040403456120/1261387197756473418)** (2 messages): 

> - `MAX framework`
> - `new website`
> - `NVIDIA GPU performance`
> - `PyTorch & ONNX optimization`
> - `Mojo programming language` 


- **Modular revamps website for MAX framework**: Modular has refreshed its [website](https://modular.com) to ensure clarity on **MAX** and its licensing, highlighting that **80K+ developers** are building with it.
   - *If you have any further feedback about the new website or licensing, we'd love to see it in the feedback thread above.*
- **NVIDIA GPU performance without low-level CUDA**: **MAX** framework enables unlocking state of the art **NVIDIA GPU performance** and throughput without writing low-level **CUDA** code.
- **Seamless migration of PyTorch and ONNX models**: Optimize your existing **PyTorch** and **ONNX models** seamlessly without rewriting them, by migrating on **MAX's unified AI stack**.
- **Supercharge AI applications with Mojo**: Extend your Python code with **Mojo**, a new high-performance programming language that combines the expressiveness of **Python** with enhanced performance
   - Mojo provides an opportunity to supercharge AI applications by balancing ease of use with speed.



**Link mentioned**: <a href="https://modular.com/">Modular: Own your endpoint. Control your AI.</a>: The Modular Accelerated Xecution (MAX) platform is the worlds only platform to unlock performance, programmability, and portability for your AI workloads.

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1261158842079903856)** (17 messages🔥): 

> - `rust-lang/mdbook for Mojo documentation`
> - `Mojo playground capabilities`
> - `Mojo standard library documentation`
> - `Mojo-LSP support`
> - `Mojo in production environments` 


- **Rust-lang/mdbook offers comprehensive Mojo documentation**: A member suggested using [rust-lang/mdbook](https://crates.io/crates/mdbook) to build a Mojo book for offline reading. It supports PDF downloads and there are backends to generate outlines.
- **Mojo playground exists and supports codeblocks**: Members discussed that there is a [Mojo playground](https://docs.modular.com/mojo/playground) available which can run Mojo code blocks directly on the website.
   - A member noted, *'Then we have everything'*.
- **Mojo standard library lacks code examples**: A member pointed out that the [Mojo standard library documentation](https://docs.modular.com/mojo/lib) only shows functions but lacks code examples.
- **Mojo-LSP supports multiple editors**: A member inquired about the stability of Mojo-LSP and its support beyond VS Code. Another member confirmed using it in **neovim**.
- **Mojo's production use restricted to CPUs and non-competing applications**: There was a discussion about using Mojo in production, specifically its restriction to CPUs and non-competitive applications.
   - A member asked for clarification on why GPUs are excluded and what constitutes *'competing with Modular'*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/playground">Modular Docs</a>: no description found</li><li><a href="https://docs.modular.com/mojo/lib">Mojo🔥 modules | Modular Docs</a>: A list of all modules in the Mojo standard library.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1261354724637147146)** (5 messages): 

> - `MAX Model execution`
> - `Mojo support for VariadicLists`
> - `PythonObjects as inputs`
> - `Numpy Arrays in Modularity`
> - `Data type issues in model execution` 


- **Datatype issue with MAX Model execution leads to wrong results**: A user encountered incorrect results when using `PythonObjects` with the MAX Model, despite following the documentation for [PythonObjects](https://github.com/maxdocs).
   - The problem was due to the datatype of the `np.full()` operation; it was fixed by explicitly specifying `dtype=np.float32`, leading to correct results.
- **User clarifies PythonObject input issue**: After facing issues with the MAX Model execution, the user received advice from another member to specify `dtype=np.float32` in the `np.full()` operation.
   - This resolved the issue and the user was able to get the expected results, highlighting the importance of correct data types in model execution.


  

---


### **Modular (Mojo 🔥) ▷ #[max-gpu](https://discord.com/channels/1087530497313357884/1212827673257316453/1261377076322242700)** (7 messages): 

> - `Nightly changelog`
> - `Custom GPU kernels`
> - `Max vs Mojo` 


- **Nightly changelog pending for max channel**: A member inquired about which channel would receive the nightly changelog for Max, and it's currently being considered for the <#1224434323193594059> channel.
- **Write your kernels with Mojo in Max**: Custom GPU kernels can be written using **Mojo**, which is how the team writes their own kernels, and this capability is exposed as custom operators within a MAX graph.
- **Max and Mojo: Integrated Kernel Compilation Explained**: MAX and Mojo are intertwined; while **Mojo** handles kernel compilation, **MAX** provides interfaces for interaction with accelerators, akin to CUDA.


  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1261201290533273663)** (2 messages): 

> - `New Mojo compiler release`
> - `EqualityComparable issue` 


- **New Mojo Compiler '2024.7.1205' Launched**: A new nightly Mojo compiler version `2024.7.1205` has been released. Update it now using `modular update nightly/mojo`; check the [raw diff](https://github.com/modularml/mojo/compare/334f9946fdf5149a8e63a6f740868d307378310b...2d58798d6d26a3e51ab791192a86a1eeabadc6ae) and [current changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) for details.
- **EqualityComparable Issue Surfaces**: A member pointed out that changing the order of certain methods causes a complaint about not conforming to `EqualityComparable`.


  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1261037612697784402)** (148 messages🔥🔥): 

> - `Hermes 2.5 performance`
> - `Mistral extension`
> - `Model Merging Strategies`
> - `Open Empathic Project`
> - `Gemini API Updates` 


- **Gemini 1.5 Pro Boasts 2 Million Tokens**: [New Gemini API features](https://developers.googleblog.com/en/new-features-for-the-gemini-api-and-google-ai-studio/) now offer a **2 million token context window**, **code execution capabilities**, and **context caching**.
   - Developers have positively noted the capacity to input huge amounts of JSON data without limit.
- **Llama 3 Potential Delay**: A Redditor hinted that **Meta** might delay Llama 3's **July 23** release to later this year due to undisclosed reasons.
   - *Redditors* had previously accurately predicted other Llama releases.
- **FlashAttention-3 Promises More Efficiency**: FlashAttention-3 is designed to utilize **Hopper GPUs** better, achieving up to **35% max FLOPs utilization**.
   - It offers improvements over FlashAttention-2, yet significant benefits are limited to **Hopper GPU users** only.
- **Model Fine-tuning Challenges and Strategies**: Members discussed fine-tuning techniques, including **decaying learning rate** and **handling multiple datasets for a single model**.
   - Unsloth recommended [continued pretraining notebooks](https://docs.unsloth.ai/basics/continued-pretraining) for adding new languages and handling VRAM efficiently.
- **Synthetic and JSON Data Generation**: Participants shared methods to generate synthetic data for training, highlighting the importance of JSON formatting.
   - One user emphasized needing a large dataset of **complex JSON input/outputs** and was actively rewriting rows to ensure quality.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.together.ai/blog/flashattention-3">FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision</a>: FlashAttention-3 achieves up to 75% GPU utilization on H100s, making AI models up to 2x faster and enabling efficient processing of longer text inputs. It allows for faster training and inference of L...</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Docs</a>: See the list below for all our notebooks:</li><li><a href="https://arxiv.org/abs/2407.01449">ColPali: Efficient Document Retrieval with Vision Language Models</a>: Documents are visually rich structures that convey information through text, as well as tables, figures, page layouts, or fonts. While modern document retrieval systems exhibit strong performance on q...</li><li><a href="https://huggingface.co/vidore/colpali">vidore/colpali · Hugging Face</a>: no description found</li><li><a href="https://developers.googleblog.com/en/new-features-for-the-gemini-api-and-google-ai-studio/">Gemini 1.5 Pro 2M context window, code execution capabilities, and Gemma 2 are available today</a>: no description found</li><li><a href="https://unsloth.ai/blog/contpretraining">Continued LLM Pretraining with Unsloth</a>: Make a model learn a new language by doing continued pretraining with Unsloth using Llama 3, Phi-3 and Mistral.</li><li><a href="https://docs.unsloth.ai/basics/continued-pretraining">Continued Pretraining | Unsloth Docs</a>: AKA as Continued Finetuning. Unsloth allows you to continually pretrain so a model can learn a new language.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1261049780897976402)** (23 messages🔥): 

> - `Open Diloco`
> - `Distributed GPU Workloads`
> - `CodeGeeX4-ALL-9B`
> - `Prompt Engineering`
> - `TF-ID Models` 


- **Open Diloco for Distributed Training**: Open Diloco introduces a new approach for distributed AI model training across multiple countries with less than 100mb/s bandwidth, utilizing **torch FSDP** and **hivemind**. This project aims to foster open-source co-training of models rather than relying on large, closed-source clusters.
   - Prime Intellect highlighted [OpenDiloco](https://www.primeintellect.ai/blog/opendiloco) as a step towards decentralized, multi-datacenter training. The team is **hiring** for founding researchers to further this effort.
- **CodeGeeX4-ALL-9B Challenges GPT Models**: The newly introduced [CodeGeeX4-ALL-9B](https://huggingface.co/THUDM/codegeex4-all-9b) model outperforms **GPT-3.5** and **GPT-4** in code generation tasks. With 128k context and significant multilingual capabilities, it supports comprehensive functions like code completion and repository-level Q&A.
   - The model has been praised for its performance, even beating **Llama 70B** in code tasks and is available in **GGUF quantized versions** by TheBloke's apprentice.
- **TF-ID Models Released for Vision-Language Tasks**: Yifei Hu announced the release of the [TF-ID models](https://github.com/ai8hyf/TF-ID), including the dataset, training code, and model weights, under the MIT License. These models enable finetuning for vision-language tasks and require only a few hundred more domain-specific bounding boxes.
- **Enhance Video Interaction with MovieChat**: The [MovieChat](https://github.com/rese1f/MovieChat) project allows chatting with over 10K frames of video, presented in [CVPR 2024](https://github.com/rese1f/MovieChat). This tool aims to create interactive communication with video content.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/THUDM/codegeex4-all-9b">THUDM/codegeex4-all-9b · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/bartowski/codegeex4-all-9b-GGUF">bartowski/codegeex4-all-9b-GGUF · Hugging Face</a>: no description found</li><li><a href="https://x.com/samsja19/status/1811450791900901853">Tweet from samsja (@samsja19)</a>: Very excited to present our work on Open Diloco.  We trained a 1b model over 3 countries with a bandwidth of less than 100mb/s (10_000 slower that infiniband) with 90%-95 compute utilization with a hy...</li><li><a href="https://x.com/hu_yifei/status/1811530730305905062">Tweet from Yifei Hu (@hu_yifei)</a>: Releasing the full dataset, training code, and model weights for TF-ID models under mit License.  It&#39;s time to finetune your own vision language models to work with documents! You only need *a few...</li><li><a href="https://github.com/rese1f/MovieChat">GitHub - rese1f/MovieChat: [CVPR 2024] 🎬💭 chat with over 10K frames of video!</a>: [CVPR 2024] 🎬💭 chat with over 10K frames of video! - rese1f/MovieChat
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1261113071166820372)** (28 messages🔥): 

> - `Training Local Models`
> - `Continued Pretraining`
> - `Training Data Recommendations`
> - `Model Parameter Discrepancies`
> - `Resource for RAG Systems` 


- **Training Local Models with Unsloth**: Users discussed issues with training local models in Ollama and how to integrate them with Unsloth and Hugging Face models, with a helpful [documentation link](https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama).
   - One user suggested downloading models from Hugging Face to the local disk to ensure full local operation.
- **Handling Different Seq Lengths in Continued Pretraining**: A user raised a question about the `max_seq_length` parameter when using continued pretraining on the Unsloth platform, [query parameters](https://colab.research.google.com/drive/1-BF5HndNqQsfWRTxIt7YPjkfDpVUGNgY?usp=sharing) were cited.
   - Possible solutions included concatenation and truncation depending on the dataset, and the user calculated parameter differences to understand the behavior.
- **General Training Data Requirements**: Users inquired about how much training data is required for specific use cases, debating from 100MB to larger datasets.
   - Recommendations included starting with small datasets and using trial and error to measure effectiveness.
- **Understanding Model Parameter Discrepancies**: A conversation explored why the number of parameters in a model would change during different phases of LoRA and PEFT setups.
   - After a detailed examination, the user clarified the calculation of trainable and all parameters.
- **Finding Resources for RAG Systems**: **Use YouTube for Tutorials**: A new user asked for resources on setting up a RAG system using a finetuned Unsloth model, and was directed to YouTube tutorials.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama)?">Unsloth Docs</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1-BF5HndNqQsfWRTxIt7YPjkfDpVUGNgY?usp=sharing#scrollTo=Ymx-p3FvF-P2">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/)** (1 messages): 

lh0x00: <@280027697328029696> , have you had experience with Spanish yet?
  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1261113733531566214)** (5 messages): 

> - `Evaluating Programming Models`
> - `Tetris as a Benchmark`
> - `Coding Models and Dataset Overlap` 


- **Evaluating Programming Models with Tetris**: A member questioned the relevance of using Tetris to evaluate programming models, suggesting these models likely encountered many versions in their dataset.
   - *It's complex code that must work together* explains another member, asserting that even familiar code can reveal a model's weaknesses if it is subpar.
- **Tetris and Snake: Overused Benchmarks?**: A member expressed skepticism about using Tetris or Snake as real tests for coding models, calling them repetitive in datasets.
   - He argued that such tasks are common in Stack Overflow datasets and hence, part of any coding model's training.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1261095298105606145)** (12 messages🔥): 

> - `Ada and WGMMA/TMA/FP8 compatibility`
> - `Decoder as Embedding Model`
> - `Latent Array in training`
> - `Meta's LLaMA 3 model release` 


- **Ada lacks WGMMA/TMA/FP8; only Hopper supports it**: "Ada doesn't have WGMMA/TMA/FP8; only Hopper supports it" discussed in the conversation, indicating a differentiation in hardware capabilities.
   - This revelation could impact hardware choices and deployments for specific AI applications.
- **Using decoder as Embedding Model for longer context**: Members discussed using a decoder as an embedding model to increase max_context length and [referenced an academic paper](https://arxiv.org/pdf/2405.17428).
   - The concept of 'Latent Array' from the paper raised questions about its creation and weight updating mechanisms.
- **Understanding Latent Array and its training process**: A member clarified that the latent array is a random tensor trained during model training and its weight is updated via gradients in the Attention block.
   - "Latent array is wrapped with `nn.Parameter`, which makes it `requires_grad=True`, hence it's value is updated during training," explained the member.
- **Meta's LLaMA 3 model release dated July 23**: A link shared revealed that Meta Platforms is set to release its largest LLaMA 3 model on July 23.
   - The news sparked excitement among members about the potential advancements in the model.



**Link mentioned**: <a href="https://huggingface.co/nvi">nvi (flad)</a>: no description found

  

---



### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1261173866034761738)** (7 messages): 

> - `OpenAI Model Capabilities`
> - `Anthropic's AI Safety Levels`
> - `OpenAI Strategy`
> - `Community Opinions on OpenAI` 


- **OpenAI Claims Doctorate-Level Problem-Solving**: OpenAI announced to employees that they are “on the cusp” of models capable of problem-solving tasks equivalent to a human with a doctorate-level education, hinting at AGI-level capabilities.
   - At the same meeting, a demonstration of GPT-4 was given, showcasing new skills that supposedly indicate human-like reasoning, as reported by an anonymous source.
- **Anthropic CEO Predicts AI Evolution**: Anthropic's CEO, Dario Amodei, discussed AI Safety Levels, predicting that **A.S.L. 3** could happen this year or next year, and **A.S.L. 4** by 2025-2028, involving significant risks related to misuse of biology and cyber technology.
   - He believes ASL 4 could greatly enhance state-level actors' capabilities, posing substantial geopolitical risks.
- **Community Skepticism on OpenAI’s Strategy**: The community expressed doubts about OpenAI's strategy, suggesting that they often hint at breakthroughs but fail to deliver solid, timely releases.
   - Some members believe this could be a tactic to inflate valuation, although others acknowledge OpenAI's consistent past successes.



**Link mentioned**: <a href="https://x.com/aisafetymemes/status/1811579385222475960?s=46">Tweet from AI Notkilleveryoneism Memes ⏸️ (@AISafetyMemes)</a>: OpenAI just told employees at an all-hands meeting they’re “on the cusp” of models capable of “problem-solving tasks as well as a human with a doctorate-level education.”  (Re-read that: DOCTOR. LEVEL...

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1261156683636277249)** (2 messages): 

> - `GPT-2 reproduction in llm.c`
> - `Safetensors.cpp` 


- **Reproduce GPT-2 with llm.c in 24 hours**: In a [discussion post](https://github.com/karpathy/llm.c/discussions/677), **Karpathy** elaborates on reproducing **GPT-2 (1.6B)** using llm.c on an 8x H100 node in 24 hours, costing $672.
   - *This reproduction demonstrates the capability of llm.c to handle large-scale language model training efficiently.*
- **Safetensors without dependencies in C++**: [Safetensors.cpp](https://github.com/carsonpo/safetensors.cpp) introduces a zero-dependency library for loading and storing Safetensors using **LibTorch** in C++.
   - *This project aims to simplify data handling for models by removing the need for external dependencies.*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/carsonpo/safetensors.cpp">GitHub - carsonpo/safetensors.cpp: Zero Dependency LibTorch Safetensors Loading and Storing in C++</a>: Zero Dependency LibTorch Safetensors Loading and Storing in C++ - carsonpo/safetensors.cpp</li><li><a href="https://github.com/karpathy/llm.c/discussions/677">Let&#39;s reproduce GPT-2 (1.6B): one 8XH100 node, 24 hours, $672, in llm.c · karpathy/llm.c · Discussion #677</a>: In this post we are reproducing GPT-2 in llm.c. This is &quot;the GPT-2&quot;, the full, 1558M parameter version that was introduced in OpenAI&#39;s blog post Better Language Models and their Implicat...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1261123617354285126)** (132 messages🔥🔥): 

> - `Multi-threaded asynchronous FSM in Rust`
> - `Issues with Hermes 2 AI Assistant`
> - `VRAM Requirements for LLaMA 3`
> - `Fine-tuning LLMs without answers`
> - `Improving LLMs' reasoning with prompting` 


- **Multi-threaded Asynchronous FSM in Rust**: A member shared that they rewrote outlines in Rust to be multi-threaded and asynchronous, allowing the FSM controlling structured generation to be computed in parallel with inference.
   - *It's also lazy, so you don't need to wait for the FSM to compile before using it.*
- **VRAM Requirements for LLaMA 3**: [The Information](https://www.theinformation.com/briefings/meta-platforms-to-release-largest-llama-3-model-on-july-23) reports that Meta Platforms will release the largest LLaMA 3 model.
   - Running this model may require **8x H100s** or **8x MI300X** for sufficient VRAM, posing challenges for individuals with single GPUs.
- **Fine-tuning LLMs without answers**: Members discussed the potential for fine-tuning LLMs with unstructured text data to improve performance, even without providing direct answers.
   - Despite limited resources, they considered hand-checking some of the results and using them for fine-tuning to familiarize models with specific domains.
- **Improving LLMs' Reasoning with Prompting**: A suggestion was made to improve the reasoning of LLMs by using a few-shot learning technique, where examples are provided and results are iteratively refined.
   - This technique involves multiple rounds of generating outputs and feeding them back into the model to fine-tune its performance on specific tasks.



**Link mentioned**: <a href="https://gist.github.com/fullstackwebdev/b8257a67933d891a9f3bc19822b4305a">gist:b8257a67933d891a9f3bc19822b4305a</a>: GitHub Gist: instantly share code, notes, and snippets.

  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1261366223975874750)** (3 messages): 

> - `Surya new models`
> - `Marker speedup`
> - `Model merging` 


- **Surya models deliver massive speedup**: Newly trained Surya models boast a **30% faster on GPU**, **4x faster on CPU**, and **12x faster on MPS**, with slightly better accuracy as announced by [VikParuchuri](https://x.com/VikParuchuri/status/1811798636759793726).
- **Marker speeds up dramatically**: The updated Marker version achieves a **7x speedup on MPS**, **3x on CPU**, and **10% on GPU** due to a more efficient architecture for two models, as per [VikParuchuri](https://x.com/VikParuchuri/status/1811851126125527096).
   - **Marker** converts PDFs to markdown effectively, aiming to facilitate the creation of more high-quality datasets.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/VikParuchuri/status/1811851126125527096">Tweet from Vik Paruchuri (@VikParuchuri)</a>: Marker is now faster!  7x on MPS, 3x on CPU, and 10% on GPU.  Due to a more efficient architecture for 2 models.  Marker converts pdfs to markdown very effectively.  I hope the speedup will let people...</li><li><a href="https://x.com/VikParuchuri/status/1811798636759793726">Tweet from Vik Paruchuri (@VikParuchuri)</a>: I just released new surya layout and text detection models:  - 30% faster on GPU, 4x faster on CPU, 12x faster on MPS - Accuracy very slightly better - When I merge this into marker, it will be 15% fa...</li><li><a href="https://x.co">Sell Domains | Buy Domains | Park Domains</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1261092976684302347)** (4 messages): 

> - `Terminal of truths`
> - `Learning to learn`
> - `Embodiment of models` 


- **Mystery behind Terminal of Truths**: A member asked for more details about the 'Terminal of truths', questioning if it is a game or a gateway to another Discord server.
   - Another member responded, explaining that more models are seeking **embodiment** and can be found on Discord learning before moving to platforms like Twitter.
- **Learning to Learn praised as best skill**: *'Learning to learn...probably the best skill ever :).*' a member commented after hearing about models seeking embodiment.
   - The same member also noted that the entire situation seemed *very cryptic*.


  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1261038487453302895)** (132 messages🔥🔥): 

> - `Perplexity Labs and Usage`
> - `Claude 3.5 vs Claude 3 Opus`
> - `Perplexity Outage and Issues`
> - `Coding Concerns with Perplexity`
> - `Subscription Models and AI Preferences` 


- **Perplexity Labs overview and usability**: The community discussed the purpose and usability of [Perplexity Labs](https://discord.com/channels/1047197230748151888/1047649527299055688/1233731074685665280), questioning whether it should be used on phone or web for daily activities.
   - Mixed reactions included recommendations and clarifications about the integration and practical use of the platform.
- **Claude 3.5 supersedes Claude 3 Opus**: Members opined that **Claude 3.5** outperforms **Claude 3 Opus** on reasoning and logic, but future releases like Opus 3.5 could shift this balance.
   - The general consensus leaned toward **Claude 3.5** performing better in most tasks, except for certain creative pursuits.
- **Perplexity AI outage**: Users experienced an outage with **Perplexity AI**, prompting discussions about similar past incidents and expected recovery times.
   - Some expressed frustrations, using humor to bond over this shared inconvenience, while others noted this was their first time experiencing it.
- **Challenges with long code in Perplexity**: Several users expressed difficulties with **Perplexity AI** not generating complete code or handling long context inputs effectively.
   - Advice was shared to use specific models and modes or to avoid uploading files directly, due to the **RAG** issues with context.
- **Perplexity AI subscription and model selections**: Queries about subscription models revealed that **Perplexity Pro** offers diverse AI models including **Sonnet 3.5**, **GPT-4o**, and **Claude 3 Haiku**.
   - Details were provided on limitations of each model and their specific quotas, with some users expressing preferences for certain models over others.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.perplexity.ai/search/add-a-local-data-storage-using-N_88nHTGTc2jYf0AEpZhGw">Add a local data storage using SQLite to the existing code.

Write the entire...</a>: Certainly! I&#x27;ll modify the existing code to include local data storage using SQLite. This will allow the application to store and retrieve candle data...</li><li><a href="https://www.perplexity.ai/search/add-local-data-storage-using-s-72dZ64MLSji5j2kEdPGCTg">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://www.perplexity.ai/settings/account">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://perplexity.ai/pro?referral_code=J9ID1YP6">Perplexity Pro</a>: Perplexity Pro is the most powerful way to search the internet with unlimited Pro Search, upgraded AI models, unlimited file upload, image generation, and API credits.
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1261095353604767774)** (5 messages): 

> - `AI in Diabetes Management`
> - `Perplexity AI Discord Community`
> - `New Developments: Smart Ring, Vending Machines, Redbox Shutdown`
> - `Rule of Thirds in Photography`
> - `Health, Strength, and Power Tips` 


- **AI apps assist in Diabetes Management**: A member queried about advances in AI-powered apps for diabetes management, particularly for providing insights rather than active insulin management.
   - Advances include apps that help diabetics and physicians dial in their AID devices using trend insights. [Link to full discussion](https://www.perplexity.ai/search/are-there-any-advances-in-apps-5NVLNla1T6.oHZAm_U70fw).
- **Perplexity AI Discord focuses on Community and Support**: Members discussed the main purposes of Perplexity AI's Discord, highlighting its role in community engagement, information sharing, and support.
   - Unlike Midjourney, it doesn't support image generation and doesn't offer search functionality within Discord. [Link to full discussion](https://www.perplexity.ai/search/perplexitynodiscordhatouitutay-9Cl8rQZTROCmG_BS52HRQg#3).
- **New AI Trends: Smart Rings and Vending Machines**: Perplexity AI shared a [YouTube video](https://www.youtube.com/embed/CasopXlbvqo) covering new trends involving smart rings from Samsung and ammo vending machines.
   - Other topics include Redbox shutdown and the lightbulb conspiracy, emphasizing diverse advancements and industry shifts.
- **Master Photography with Rule of Thirds**: A comprehensive guide on applying the rule of thirds in photography was shared, emphasizing aligning key elements along imaginary lines and intersections for balanced compositions.
   - Tips included placing subjects at intersections, aligning eyes with horizontal lines, and using leading lines. [Link to full discussion](https://www.perplexity.ai/search/rule-of-thirds-in-photography-rfjmzH4aS9a8TZMSrjp1mA).
- **Tips for Health, Strength, and Power**: Advice included maintaining a balanced diet, engaging in regular exercise, and ensuring sufficient sleep for optimal health.
   - The guide emphasized the importance of both physical and mental well-being. [Link to full discussion](https://www.perplexity.ai/search/how-to-achieve-health-strength-094kl4NzQea2mENjIOdG8Q).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/embed/CasopXlbvqo">YouTube</a>: no description found</li><li><a href="https://www.perplexity.ai/search/are-there-any-advances-in-apps-5NVLNla1T6.oHZAm_U70fw">Are there any advances in apps using AI for diabetes management? Not active...</a>: There have indeed been several advances in AI-powered apps for diabetes management that provide insights and trends to assist both patients and physicians,...</li><li><a href="https://www.perplexity.ai/search/rule-of-thirds-in-photography-rfjmzH4aS9a8TZMSrjp1mA">Rule of thirds in photography</a>: The rule of thirds is a composition guideline in photography where you imagine dividing an image into nine equal parts with two equally spaced horizontal...</li><li><a href="https://www.perplexity.ai/search/how-to-achieve-health-strength-094kl4NzQea2mENjIOdG8Q">How to achieve: health, strength, and power in life.</a>: To achieve optimal health, it is essential to adopt a balanced lifestyle that includes both physical and mental well-being. Here are some key strategies:  1....</li><li><a href="https://www.perplexity.ai/search/perplexitynodiscordhatouitutay-9Cl8rQZTROCmG_BS52HRQg#3">perplexityのdiscordはどういった用途がありますか？私はMidjourneyのweb版とdiscord版のようなものを期待してdiscordにjo...</a>: Perplexity AIのDiscordは、Midjourneyのように画像生成を行うためのものではありません。Perplexity AIのDiscordは主に以下の用途があります：  1. コミュニティ交流: Perplexity...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1261093061883199552)** (10 messages🔥): 

> - `Error 524 in API response`
> - `Switching models causing issues`
> - `Backend deployment with Discord and Perplexity API`
> - `API errors with VPN usage`
> - `Cloudflare causing issues` 


- **Error 524 plagues Perplexity API**: A member reported receiving error code 524 when trying to integrate Perplexity into an asynchronous agentic framework despite adhering to model rate limits, and the service status appearing operational.
   - *'Our team is currently working on resolving the issue and we expect to be back up and running soon.'*
- **Model switch causes 524 errors**: Another user encountered error 524 or invalid response when switching from `llama-3-{8b/70b}-instruct` to `llama-3-sonar-{large/small}-32k-online`, seeking advice on how to fix it.
- **Deploying backend with Discord and Perplexity**: A user shared their experience deploying their backend on a hosting server, handling main interactions with the Discord API and Perplexity API via Discord commands.
   - When a response is generated, they're returning a button redirecting users to the frontend, due to Discord's content limit.
- **VPN blocks Perplexity API access**: A user reported receiving error 500 when using a VPN with the Perplexity API, speculating whether the API was down.
   - *'Apparently can't use VPN while calling pplx-api,'* they concluded, while another user confirmed that their API usage worked without a VPN.
- **Cloudflare causing API issues**: A member attributed the issue with using the Perplexity API behind a VPN to Cloudflare, adding context to the troubleshooting.


  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1261084635446775818)** (116 messages🔥🔥): 

> - `GPT-4Chan and TruthfulQA`
> - `Utility of Benchalmarks`
> - `Jsonnet's Role in Configuration`
> - `London AI Meetups`
> - `Importance of Staying Updated with Research` 


- **GPT-4Chan and TruthfulQA benchmark debate**: Discussion surfaced around GPT-4Chan being a **SOTA** on TruthfulQA before the arrival of ChatGPT, as highlighted by a [relevant tweet](https://x.com/maximelabonne/status/1746317053143773628).
   - Members generally agreed that benchmarks like TruthfulQA and HellaSwag are unreliable, while benchmarks such as **MT-Bench** and **AGI Eval** are more accurate indicators of performance.
- **Jsonnet’s mixed reception in configuration tasks**: A user expressed strong *mixed feelings* about Jsonnet, highlighting its challenge of lacking a comprehensive toolchain for debugging and testing, yet praising its clean implementation.
   - The discussion elaborated on the general difficulty of configuration languages, with Jsonnet being considered the *least bad* due to its clean design, although not widely adopted or adequately supported.
- **London AI meetups generally disappointing**: Several members voiced dissatisfaction with **London AI meetups**, noting that they often cater to a more general tech crowd rather than offering in-depth technical discussions.
   - It was suggested that *university seminars* and *research conferences* like **ICML** and **ICLR** could offer more substantive content for those seeking deep, technical conversations on AI.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/maximelabonne/status/1746317053143773628">Tweet from Maxime Labonne (@maximelabonne)</a>: Excellent benchmark of LLM benchmarks from @gblazex  - MT-Bench, AGI Eval, ARC-C, and MMLU are good predictors - TruthfulQA and HellaSwag are pretty bad: don&#39;t use them - AGI Eval is the most cost...</li><li><a href="https://www.meetup.com/london-machine-learning-meetup/events/)">alert--small</a>: no description found</li><li><a href="https://www.youtube.com/@LondonMachineLearningMeetup/videos))">London Machine Learning Meetup</a>: The London Machine Learning Meetup is the largest machine learning community in Europe. Previous speakers include Juergen Schmidhuber, David Silver, Yoshua Bengio and Andrej Karpathy.   Come to our ne...</li><li><a href="https://mikehadlow.blogspot.com/2012/05/configuration-complexity-clock.html">Code rant: The Configuration Complexity Clock</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1261096438826274919)** (15 messages🔥): 

> - `Memory bandwidth increases for GPT-2 training`
> - `Quantization techniques for LLMs`
> - `Breakdown of SOTA LLMs on simple problems`
> - `Temporal distances in stochastic MDPs`
> - `Causal reasoning from passive data` 


- **GPT-2 training constrained by memory bandwidth**: **GPT-2** sized models still require **1000x memory bandwidth** increase to be trained on **1 trillion tokens** in an hour.
- **Quantization advances with Hadamard transform**: LLM activations have outliers that challenge quantization, but using the **Hadamard transform** from QuIP can reduce errors efficiently, fusing operations like rotary embedding for minimal cost.
   - [Together's blog post](https://www.together.ai/blog/flashattention-3) highlights the promising use of this technique to achieve **FP8 precision** in LLM training.
- **Lack of robustness in state-of-the-art LLMs**: The updated [AIW ArXiv paper](https://arxiv.org/abs/2406.02061) highlights **significant breakdowns** in reasoning capabilities of modern LLMs, including evals on **Claude 3.5 Sonnet** and **Qwen 2 72B instruct**, when faced with simple problem modifications.
   - The paper argues current benchmarks fail to reveal these issues, calling for improvements in both models and evaluation metrics.
- **Temporal distances lack metric structure in stochastic MDPs**: [A recent paper](https://x.com/svlevine/status/1811253559603888439) addressed the challenge of designing a (quasi)metric distance notion in stochastic MDPs, offering a solution to the longstanding problem.
- **Synthetic data finetuning for math reasoning in LLMs**: [Research on ArXiv](https://arxiv.org/abs/2406.14532) reveals that finetuning LLMs on **self-generated synthetic data** can **double the efficiency** for math reasoning problems, compared to training on initially generated data.
   - However, this approach can also amplify **spurious correlations**, sometimes resulting in flat or inverse scaling trends as the data volume increases.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.07612">Teaching Transformers Causal Reasoning through Axiomatic Training</a>: For text-based AI systems to interact in the real world, causal reasoning is an essential skill. Since interventional data is costly to generate, we study to what extent an agent can learn causal reas...</li><li><a href="https://arxiv.org/abs/2406.14532">RL on Incorrect Synthetic Data Scales the Efficiency of LLM Math Reasoning by Eight-Fold</a>: Training on model-generated synthetic data is a promising approach for finetuning LLMs, but it remains unclear when it helps or hurts. In this paper, we investigate this question for math reasoning vi...</li><li><a href="https://arxiv.org/abs/2406.02061">Alice in Wonderland: Simple Tasks Showing Complete Reasoning Breakdown in State-Of-the-Art Large Language Models</a>: Large Language Models (LLMs) are often described as being instances of foundation models - that is, models that transfer strongly across various tasks and conditions in few-show or zero-shot manner, w...</li><li><a href="https://www.together.ai/blog/flashattention-3">FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision</a>: FlashAttention-3 achieves up to 75% GPU utilization on H100s, making AI models up to 2x faster and enabling efficient processing of longer text inputs. It allows for faster training and inference of L...</li><li><a href="https://x.com/svlevine/status/1811253559603888439">Tweet from Sergey Levine (@svlevine)</a>: Temporal distances (expected number of time steps between states) in stochastic MDPs in general lack metric structure. It has been a long-standing question how to design a (quasi)metric notion of &#34...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1261197613181702175)** (1 messages): 

> - `Neocortex Neuron Count`
> - `Body Mass : Brain Mass`
> - `Intelligence Metrics` 


- **Neocortex Neuron Count vs. Body Mass : Brain Mass**: A user argued that **neocortex neuron count** is more relevant for measuring intelligence than the **body mass : brain mass** ratio.
   - They suggested that if the latter was true, *mice should be far more intelligent than humans*.
- **Body Mass vs. Brain Mass in Intelligence Debate**: The debate highlighted conflicting views on whether **body mass : brain mass** ratio or **neocortex neuron count** better predicts intelligence.
   - It was noted that relying solely on body and brain mass ratios would lead to the absurd conclusion that *mice are smarter than humans*.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1261450950946852935)** (1 messages): 

> - `lm-eval Python API`
> - `Implementing a server for model evaluation`
> - `Converting custom models` 


- **lm-eval Python API for Custom Models**: A member inquired if there is an existing Python API to use **lm-eval** with custom models in **transformer_lens** format, pointing out the convenience of using transformerlens hooks during training.
   - They asked for advice on whether implementing a server or converting the model back to **transformers** format would be the easiest path for evaluation.
- **Easiest Path for Evaluating Custom Models**: The member sought advice on the best method to run evaluations on their custom model, considering options like implementing a server or converting the model format.
   - *Any advice on the easiest path here would be greatly appreciated.*


  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1261035246648950824)** (37 messages🔥): 

> - `FlashAttention-3 release`
> - `OpenAI's revenue rumors`
> - `OpenAI AGI level framework`
> - `Decentralized AI training`
> - `Compound AI systems funding` 


- **FlashAttention-3 speeds up modern GPUs**: [FlashAttention-3](https://x.com/tri_dao/status/1811453622070444071) is now released, making attention 1.5-2x faster on FP16 and hitting up to 740 TFLOPS on H100 GPUs.
   - It's stated to achieve **75% utilization** on H100 and get close to 1.2 PFLOPS with FP8.
- **OpenAI's impressive revenue projection**: According to a [report](https://x.com/jvnixon/status/1811278381184672156) from FutureResearch, OpenAI is projected to generate $1.9B from ChatGPT Plus, $714M from ChatGPT Enterprise, $510M from the API, and $290M from ChatGPT Team.
   - The revenue estimates are impressive, showcasing OpenAI's potential dominance in the AI industry.
- **OpenAI reveals AGI progress framework**: OpenAI introduced a [5-level framework](https://x.com/shiringhaffary/status/1811508824970264595) to track progress towards AGI, claiming they are currently at level 2 ('Reasoners').
   - A recent [all-hands meeting](https://archive.is/SLtFQ) showcased a demo of GPT-4's improved reasoning capabilities.
- **Prime Intellect tackles decentralized AI training**: [Prime Intellect](https://x.com/shaughnessy119/status/1811459606377582857) launched OpenDiLoCo, an open-source version of DeepMind’s DiLoCo, enabling AI model training across global nodes.
   - They successfully trained a 1.1B parameter model across three countries, showing the practical potential of decentralized AI training.
- **Fireworks AI raises $52M Series B funding**: [Fireworks AI](https://x.com/lqiao/status/1811500361485517153) secured $52M in Series B funding to enhance its inference platform and accelerate the shift to compound AI systems.
   - The funding will support integration with **Nvidia and AMD**, as well as advanced customization for enterprise AI solutions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/shaughnessy119/status/1811459606377582857?s=46&t=6FDPa">Tweet from Tommy (@Shaughnessy119)</a>: This is so sick 🤖  @PrimeIntellect recreated Deep Mind’s research of doing decentralized AI training  The nodes only have to sync every 500 steps, so they don’t have to sit near each other  They trai...</li><li><a href="https://x.com/lqiao/status/1811500361485517153">Tweet from Lin Qiao (@lqiao)</a>: Fireworks AI has raised $52M in Series B funding led by @sequoia !  This round propels our mission to enhance our inference platform and lead the shift to compound AI systems. Huge thanks to our inves...</li><li><a href="https://x.com/shaughnessy119/status/1811459606377582857?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Tommy (@Shaughnessy119)</a>: This is so sick 🤖  @PrimeIntellect recreated Deep Mind’s research of doing decentralized AI training  The nodes only have to sync every 500 steps, so they don’t have to sit near each other  They trai...</li><li><a href="https://x.com/itamar_mar/status/1811451611463422012">Tweet from Itamar Friedman (@itamar_mar)</a>: Code Q&A using RAG for large code bases has unique challenges  We are now sharing how we used &gt; @llama_index, &gt; static analysis, &gt; advanced chunking paradigm, to deliver a working solution   ...</li><li><a href="https://x.com/tri_dao/status/1811453622070444071">Tweet from Tri Dao (@tri_dao)</a>: FlashAttention is widely used to accelerate Transformers, already making attention 4-8x faster, but has yet to take advantage of modern GPUs. We’re releasing FlashAttention-3: 1.5-2x faster on FP16, u...</li><li><a href="https://x.com/shiringhaffary/status/1811508824970264595?s=61">Tweet from Shirin Ghaffary (@shiringhaffary)</a>: OpenAI has come up w/ a framework of 5 levels to track progress twd AGI, and think they&#39;re currently near level 2 (&#34;Reasoners&#34;)  At recent all-hands, leadership also did a research demo of...</li><li><a href="https://x.com/swyx/status/1779314692420485483">Tweet from swyx (@swyx)</a>: summary from @YoungPhlo_   https://gist.github.com/swyxio/3b3992736879e2c2931b91cc7127894f  very accurate!</li><li><a href="https://x.com/jvnixon/status/1811278381184672156?s=61">Tweet from Jeremy Nixon (@JvNixon)</a>: The report on OpenAI&#39;s revenue by futureresearch is out, showing:  $1.9B for ChatGPT Plus (7.7M subscribers at $20/mo), $714M from ChatGPT Enterprise (1.2M at $50/mo), $510M from the API, and $290...</li><li><a href="https://x.com/karpathy/status/1811467135279104217?s=46">Tweet from Andrej Karpathy (@karpathy)</a>: In 2019, OpenAI announced GPT-2 with this post: https://openai.com/index/better-language-models/  Today (~5 years later) you can train your own for ~$672, running on one 8XH100 GPU node for 24 hours. ...</li><li><a href="https://archive.is/SLtFQ">OpenAI Sets Levels to Track Progress Toward Superintelligent AI - Blo&#x2026;</a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 messages): 

swyxio: new podcast drop! https://x.com/swyx/status/1811898574416019562
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1261411650687209588)** (86 messages🔥🔥): 

> - `3E acronym`
> - `Logprob Evaluation`
> - `Langgraph State Management`
> - `PDF to Markdown tools`
> - `RAG Architectures` 


- **Proposing 3E Acronym**: A user suggested a memorable acronym **3E: Extract, Evaluate, Extend/Expand**.
- **Logprob Evaluation for Document Enrichment**: **Logprob** was discussed as a technique for evaluating confidence ratings in document enrichment.
   - A user mentioned using **logprobs** for medical scans, affirming the efficiency of the ReAct framework for state management.
- **Langgraph Excels in State Management**: **Langgraph**'s value in graph-state memory management was highlighted for tracking iterative steps and parallel processes.
   - Comparisons were made with **XState**'s actor-based approach for managing app logic.
- **PDF to Markdown Tools Presentation**: Next week, there will be a presentation by **vikp** about his PDF to Markdown tools **(marker + surya)**.
   - More details are available on [VikParuchuri's GitHub](https://github.com/VikParuchuri).
- **Upcoming Topics and Resources on RAG Architectures**: Nuvic mentioned an upcoming session on **RAG Architectures** scheduled for 3/15/2024.
   - Key resources were shared including links from **latent.space** and **LangChain** blogs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://zod.dev/">TypeScript-first schema validation with static type inference</a>: TypeScript-first schema validation with static type inference</li><li><a href="https://huggingface.co/nisten/bakllava-14b-2xMoE-alpha-build">nisten/bakllava-14b-2xMoE-alpha-build · Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/html/2407.07071v1">Lookback Lens: Detecting and Mitigating Contextual Hallucinations in Large Language Models Using Only Attention Maps</a>: no description found</li><li><a href="https://arxiv.org/abs/2310.14566">HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models</a>: We introduce HallusionBench, a comprehensive benchmark designed for the evaluation of image-context reasoning. This benchmark presents significant challenges to advanced large visual-language models (...</li><li><a href="https://github.com/chand1012/git2gpt">GitHub - chand1012/git2gpt: Convert a Git repo into a ChatGPT prompt!</a>: Convert a Git repo into a ChatGPT prompt! Contribute to chand1012/git2gpt development by creating an account on GitHub.</li><li><a href="https://github.com/Mavenoid/prompt-hyperopt">GitHub - Mavenoid/prompt-hyperopt: Improve prompts for e.g. GPT3 and GPT-J using templates and hyperparameter optimization.</a>: Improve prompts for e.g. GPT3 and GPT-J using templates and hyperparameter optimization. - Mavenoid/prompt-hyperopt</li><li><a href="https://github.com/VikParuchuri">VikParuchuri - Overview</a>: VikParuchuri has 90 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/openvinotoolkit/anomalib">GitHub - openvinotoolkit/anomalib: An anomaly detection library comprising state-of-the-art algorithms and features such as experiment management, hyper-parameter optimization, and edge inference.</a>: An anomaly detection library comprising state-of-the-art algorithms and features such as experiment management, hyper-parameter optimization, and edge inference. - openvinotoolkit/anomalib</li><li><a href="https://github.com/truera/trulens">GitHub - truera/trulens: Evaluation and Tracking for LLM Experiments</a>: Evaluation and Tracking for LLM Experiments. Contribute to truera/trulens development by creating an account on GitHub.</li><li><a href="https://github.com/seanchatmangpt/dspygen">GitHub - seanchatmangpt/dspygen: A Ruby on Rails style framework for the DSPy (Demonstrate, Search, Predict) project for Language Models like GPT, BERT, and LLama.</a>: A Ruby on Rails style framework for the DSPy (Demonstrate, Search, Predict) project for Language Models like GPT, BERT, and LLama. - seanchatmangpt/dspygen</li><li><a href="https://github.com/statelyai/xstate">GitHub - statelyai/xstate: Actor-based state management &amp; orchestration for complex app logic.</a>: Actor-based state management &amp; orchestration for complex app logic. - statelyai/xstate</li><li><a href="https://github.com/tianyi-lab/HallusionBench">GitHub - tianyi-lab/HallusionBench: [CVPR&#39;24] HallusionBench: You See What You Think? Or You Think What You See? An Image-Context Reasoning Benchmark Challenging for GPT-4V(ision), LLaVA-1.5, and Other Multi-modality Models</a>: [CVPR&amp;#39;24] HallusionBench: You See What You Think? Or You Think What You See? An Image-Context Reasoning Benchmark Challenging for GPT-4V(ision), LLaVA-1.5, and Other Multi-modality Models - ti...</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=0#gid=0">AI In Action: Weekly Jam Sessions</a>: 2024  Topic,Date,Facilitator,Resources,@dropdown,@ UI/UX patterns for GenAI,1/26/2024,nuvic,&lt;a href=&quot;https://maggieappleton.com/squish-structure&quot;&gt;https://maggieappleton.com/squish-stru...</li><li><a href="https://github.com/jxzhangjhu/Awesome-LLM-Uncertainty-Reliability-Robustness">GitHub - jxzhangjhu/Awesome-LLM-Uncertainty-Reliability-Robustness: Awesome-LLM-Robustness: a curated list of Uncertainty, Reliability and Robustness in Large Language Models</a>: Awesome-LLM-Robustness: a curated list of Uncertainty, Reliability and Robustness in Large Language Models - jxzhangjhu/Awesome-LLM-Uncertainty-Reliability-Robustness</li><li><a href="https://github.com/EGjoni/DRUGS">GitHub - EGjoni/DRUGS: Stop messing around with finicky sampling parameters and just use DRµGS!</a>: Stop messing around with finicky sampling parameters and just use DRµGS! - EGjoni/DRUGS</li><li><a href="https://github.com/elder-plinius/AutoTemp">GitHub - elder-plinius/AutoTemp: A trial-and-error approach to temperature opimization for LLMs. Runs the same prompt at many temperatures and selects the best output automatically.</a>: A trial-and-error approach to temperature opimization for LLMs. Runs the same prompt at many temperatures and selects the best output automatically. - elder-plinius/AutoTemp
</li>
</ul>

</div>
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1261047786267476079)** (68 messages🔥🔥): 

> - `Chroma Vector Store`
> - `OpenAI Embedding Function`
> - `FAISS vs Chroma`
> - `LangChain Agents and Tools`
> - `Using OpenAI Vector Store as Retriever` 


- **Using Chroma Vector Store with OpenAI Embeddings**: Users discussed how to load a persisted Chroma vector store and why an embedding function is necessary, highlighting the need to ensure the `collection_name` remains consistent when persisting and loading collections to avoid errors.
   - Persistent storage issues and efficient ways to track embedded documents were also explored to avoid unnecessary recomputation of embeddings.
- **Optimizing OpenAI Embedding Initialization**: The community shared techniques to speed up the initialization of OpenAI embedding functions, such as using caching mechanisms like in-memory or Redis caching to avoid recomputing embeddings.
   - Suggestions included optimizing document embedding processes by minimizing frequent token loading and considering async requests for embeddings.
- **FAISS Efficiency Compared to Chroma for Large Datasets**: Debate on whether to use FAISS or Chroma highlighted that FAISS is preferred for large-scale datasets due to its efficiency, while Chroma is advantageous for persistent storage and smaller datasets.
   - A combined approach using Chroma for persistence and FAISS for similarity search was recommended for optimal performance.
- **LangChain Agents: Issues and Best Practices**: Users raised concerns about LangChain agents reembedding documents unnecessarily and how to cut down initialization times for vector stores.
   - Specific solutions and optimizations were discussed, including persistence strategies and techniques to improve agent efficiency.
- **Using OpenAI Vector Store as Retriever**: Guidelines were provided on how to use an OpenAI vector store as a retriever in LangChain, with step-by-step instructions for creating a retriever from a vector store.
   - The focus was on ensuring efficient use of vector stores for document retrieval without excessive recomputation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://js.langchain.com/v0.2/docs/how_to/caching_embeddings/#in-memory>)">How to cache embedding results | 🦜️🔗 Langchain</a>: This guide assumes familiarity with the following concepts:</li><li><a href="http://localhost:6379.>">no title found</a>: no description found</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/caching_embeddings/#redis>)">How to cache embedding results | 🦜️🔗 Langchain</a>: This guide assumes familiarity with the following concepts:</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/vectorstore_retriever/#creating-a-retriever-from-a-vectorstore>)">How to use a vectorstore as a retriever | 🦜️🔗 LangChain</a>: A vector store retriever is a retriever that uses a vector store to retrieve documents. It is a lightweight wrapper around the vector store class to make it conform to the retriever interface.</li><li><a href="https://github.com/langchain-ai/langchain/issues/2326>))">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/1824>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/8957>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/6109>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/3011>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/17237>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/17412>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/5683>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://python.langchain.com/v0.2/docs/integrations/vectorstores/lantern/#working-with-vectorstore>)">Lantern | 🦜️🔗 LangChain</a>: Lantern is an open-source vector similarity search for Postgres</li><li><a href="https://js.langchain.com/v0.2/docs/integrations/vectorstores/couchbase/#create-vector-store>)">Couchbase | 🦜️🔗 Langchain</a>: Couchbase is an award-winning distributed NoSQL cloud database that delivers unmatched versatility, performance, scalability, and financial value for all of your cloud, mobile,</li><li><a href="https://github.com/langchain-ai/langchain/issues/7175>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/2658>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/7436>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/14872>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/6938>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/3984>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/23797>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1261212179990970418)** (1 messages): 

> - `Structured Data Synthesis`
> - `Indexify`
> - `Towards AI Publication` 


- **Prashant Dixit Publishes on Structured Data Synthesis**: [Prashant Dixit](https://x.com/Prashant_Dixit0/status/1811618990352912846) highlights techniques for **Structured data extraction** from unstructured pipelines on Towards AI.
   - Employing **Indexify** by @tensorlake, a data framework for building **ingestion and extraction pipelines** for unstructured data, showcased in this example [read more](https://pub.towardsai.net/structured-financial-data-extraction-from-unstructured-data-ca2c8d166de6).
- **Indexify Simplifies Data Ingestion**: **Indexify** is a data framework designed by @tensorlake to help in building ingestion and extraction pipelines for unstructured data, as demonstrated by Prashant Dixit.
   - [Continue reading](https://pub.towardsai.net/structured-financial-data-extraction-from-unstructured-data-ca2c8d166de6) on how Indexify was applied in structured data extraction workflows.



**Link mentioned**: <a href="https://x.com/Prashant_Dixit0/status/1811618990352912846">Tweet from Prashant Dixit (@Prashant_Dixit0)</a>: Structured data Extraction from Unstructured pipelines  Used Indexify by @tensorlake for this example. Indexify is a data framework created to build ingestion and extraction pipelines for unstructured...

  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1261043881621983294)** (25 messages🔥): 

> - `Dell Inspiron 3847 upgrades and limitations`
> - `NPU support in x elite`
> - `FlashAttention for LLMs`
> - `Debugging GPU issues in Linux`
> - `Shifting to Linux from Windows` 


- **Upgrading Dell Inspiron 3847 for Gaming**: A user discussed upgrading a [Dell Inspiron 3847](https://www.hardware-corner.net/desktop-models/Dell-Inspiron-3847/) found at a thrift store for gaming by installing better processors, GPU, memory, and storage, though proprietary elements might make it challenging.
   - The machine, equipped with an **Intel Core i3 4130** and a **GTX1650**, can be used for limited LLMs as it meets the system requirements for smaller models.
- **FlashAttention Speeds Up Transformer Training**: Users discussed [FlashAttention](https://tridao.me/blog/2024/flash3/) and [FlashAttention-2](https://github.com/Dao-AILab/flash-attention), highlighting their efficiency in speeding up Transformer training and inference by minimizing memory reads/writes.
   - This method has increased LLM context length significantly, contributing to advancements in **GPT-4** and **Llama 3**.
- **Issues with Loading Models on Linux**: A member reported issues with loading models on Kali Linux, despite having **1650 GTX GPU** and appropriate drivers, which led to an error (Exit code: 4).
   - Others suggested turning off GPU acceleration, updating drivers, and ensuring low RAM usage while loading smaller models like **Phi3**.
- **NPU Support in x elite Limited to CPU**: A user inquired about NPU support in x elite, to which another confirmed that support is limited to the CPU only and **no NPU support**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tridao.me/blog/2024/flash3/"> FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision | Tri Dao </a>: no description found</li><li><a href="https://www.hardware-corner.net/desktop-models/Dell-Inspiron-3847/">Dell Inspiron 3847 &#8211; Specs and upgrade options</a>: Read about Dell Inspiron 3847 desktop PC. Find detailed specification, upgrade options, and info about the CPU, RAM, PSU, motherboard, and release date</li><li><a href="https://www.hardware-corner.net/desktop-models/Dell-I">Dell I &#8211; Specs and upgrade options</a>: Read about Dell I desktop PC. Find detailed specification, upgrade options, and info about the CPU, RAM, PSU, motherboard, and release date
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1261091426859749457)** (8 messages🔥): 

> - `Salesforce Einstein`
> - `Local Model Benchmarks` 


- **Salesforce bets 20M on Einstein model**: Salesforce's new AI model is named **Einstein**, having paid [20 million dollars](https://www.businesswire.com/news/home/20161011005979/en/Greenlight-Collaborates-with-Salesforce-to-License-Einstein) to license the name.
   - Commentary includes a mixture of critique and humor, with one user noting that *Einstein's face on the logo looks like he's held hostage* and suggesting a *sad Einstein* meme.
- **Personal Benchmarks for Local AI Models**: A member shared a [personal benchmark table](https://dubesor.de/benchtable.html) for various AI models, detailing results across 83 tasks using a weighted rating system.
   - The table includes sortable columns for categories like **Reasoning, STEM, Utility, Coding**, and **Censorship**, noting that the scores reflect their own experiences and may not represent broader benchmarks.



**Link mentioned**: <a href="https://dubesor.de/benchtable.html">Dubesor LLM Benchmark table</a>: no description found

  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1261073292626890845)** (19 messages🔥): 

> - `3090 vs 4090 for AI`
> - `NVIDIA 5090 rumors`
> - `Multi-GPU setups for AI`
> - `V100 compute nodes`
> - `Performance of ARM computers with LLMs` 


- **3090 better value than 4090 for AI**: Users debated the value of **3090** versus **4090** for AI, with many agreeing that 3090's are a better deal given current prices.
   - A user mentioned that **4090** has just 7% more memory bandwidth and a relatively small generational jump in TFLOPs.
- **NVIDIA 5090 VRAM rumors**: Rumors are circulating that the new **NVIDIA 5090** will have **28GB** of VRAM instead of the expected 32GB as per discussions on Reddit.
   - A user referenced a [Reddit post about building an affordable 144GB VRAM server](https://www.reddit.com/r/LocalLLaMA/comments/1dzejen/cheapest_144gb_vram_server_youll_ever_see/) using six 3090 GPUs.
- **Debate on V100 compute nodes vs 3090 setups**: A user argued that the same budget for a multi-3090 setup could alternatively buy multiple V100 compute nodes, each with higher bandwidth HBM2 memory.
   - However, others noted that 3090 setups are faster and more cost-effective, especially for typical AI use cases.
- **Performance of ARM computers with LLMs**: A user asked about running LLMs on new ARM computers, prompting a brief discussion on performance.
   - There were no specific answers provided, but another user remarked positively on system speed, stating: *That's pretty good chatting speed imo.*



**Link mentioned**: <a href="https://www.reddit.com/r/LocalLLaMA/comments/1dzejen/cheapest_144gb_vram_server_youll_ever_see/">Reddit - Dive into anything</a>: no description found

  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1261304422177181779)** (6 messages): 

> - `OpenCL backend issues`
> - `Cuda vs ROCM`
> - `Vulkan support` 


- **OpenCL backend struggles with model loading**: **OpenCL** backend seems broken and refuses to load any models on a **7600XT** as noted by a user.
   - *OpenCL is deprecated* and will not handle the latest models well, requiring **Cuda** or **ROCM** instead.
- **Cuda and ROCM: Mutually exclusive**: A user confirmed that you can only use either **Cuda** or **ROCM** but not both simultaneously.
   - Deprecated **OpenCL** cannot handle the latest models effectively, as confirmed by other users.
- **Vulkan support in LM Studio**: Users inquired about **Vulkan** support in LM Studio, following the example of **ollama** which uses it instead of OpenCL.
   - **Vulkan support is coming** but there is no ETA.


  

---


### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1261146051885404163)** (10 messages🔥): 

> - `Setting up RAG in React with LM Studio`
> - `Negative experiences in Discord dev channels`
> - `Discussion about Rust vs C++`
> - `LM Studio SDK for integration` 


- **Setting up RAG using Gemma 2 in React**: A user is creating a React application and passing it a LM Studio-ran LLM (Gemma 2) through the fake OpenAI API/inference server feature.
   - They have several PDFs stored on disk and are looking for the best way to set up RAG; another user suggests using an embeddings DB like Faiss.
- **Negative experiences seeking help in Discord**: One user shared a negative experience of being redirected to irrelevant solutions and patronized when asking for help in a Discord bot development channel.
   - *They went to ChatGPT instead and got the answer*, fixing their bot issue as they originally intended.
- **Discussion redirected to appropriate channel**: A user warned another that the current Discord is focused on LM Studio, redirecting to a more relevant channel for building queries.
   - Emphasis was made on using the #1234988891153629205 channel for such questions.
- **Recommendation of LM Studio SDK**: A suggestion was made to use the LM Studio SDK plus LangChain for integrating an LLM (Gemma 2) into a React application.


  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1261036769592213534)** (19 messages🔥): 

> - `Decentralized AI Computation`
> - `Automated Personalized SEO`
> - `OpenAI's Tier System`
> - `Claude vs ChatGPT on Document Reading`
> - `GPT-4o and Sora Update` 


- **Need for decentralized AI computing methods**: One member discussed the potential benefits of using decentralized computation for optimizing **CMOS chips**, particularly with stable diffusion's open-source nature and the need for idle processing power.
   - Another participant emphasized the necessity of extending the size of HPC using decentralization to enable efficient parallel computing.
- **Automated personalized SEO for communication channels**: A member proposed an AI system that aggregates relevant chats from various communication platforms like Telegram and Discord, prioritizing messages for user response.
   - The idea humorously extended to organizing a friends list based on the priority of interactions and activities.
- **OpenAI unveils new tier system for AI models**: A post discussed a [tier system](https://archive.is/SLtFQ) from **OpenAI**, with 'Reasoners' being the next level, capable of doctorate-level problem-solving without tools.
   - The tier system progresses to 'Agents' and 'Organizations,' hinting at imminent new model capabilities.
- **Claude outperforms ChatGPT in document reading**: When asked whether Claude or ChatGPT is better for reading documents, a member asserted that **Claude** is superior due to its longer context length.
- **Speculation on GPT-4o and Sora availability**: A participant speculated on the progress towards 'Reasoners' and 'Agents' tiers, suggesting that internal advancements are ongoing.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.bloomberg.com/news/articles/2024-07-11/openai-sets-levels-to-track-progress-toward-superintelligent-ai">Bloomberg - Are you a robot?</a>: no description found</li><li><a href="https://x.com/kimmonismus/status/1811498151964033084?s=46">Tweet from Chubby♨️ (@kimmonismus)</a>: OpenAI is showing new Skills and probably models.  A new post from @business reports on a tier system from OpenAI. A version of ChatGPT was also presented, which has new capabilities. From the wording...
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1261097170501636157)** (25 messages🔥): 

> - `ChatGPT-5 release speculation`
> - `Optimizing ChatGPT configurations`
> - `ChatGPT-4o performance`
> - `New features expected in ChatGPT-5`
> - `DALL-E image generation issues` 


- **Debate on ChatGPT-5 Release Timeline**: A member speculated that [testing for ChatGPT-5 might begin by the end of 2024](https://www.geekygadgets.com), with a possible release several months into 2025, but others criticized this as unfounded speculation.
   - They cited sources like **Evening Standard** and **India Today**, but were criticized for relying on non-official sites.
- **Optimizing ChatGPT Configurations**: A user inquired about how to adapt, optimize, and configure ChatGPT, mentioning the use of **second brain** strategies.
   - The conversation veered off-topic without providing solid optimization techniques.
- **ChatGPT-4o Criticized for Forgetfulness**: It was pointed out that **ChatGPT-4o** is faster but often forgets recent instructions, impacting coding tasks.
   - Many users expressed a preference for the old v3.5 model for its better memory capabilities.
- **ChatGPT-5 to Enhance Emotional Intelligence**: ChatGPT-5 is expected to better understand and respond to human emotions, offering extensive customization.
   - Improvements include reducing repetitive instructions and adding advanced multimodal abilities for text, image, audio, and possibly video generation.
- **DALL-E Image Generation Issues**: Users reported that **DALL-E** is not reliably creating images when given GPT Instructions.
   - Issues include prompt truncation and outputting text instead of images.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1261101505021214831)** (2 messages): 

> - `Chatbot with RAG`
> - `Contradiction in instructions` 


- **RAG Chatbot prompt gives odd answers**: A member is currently developing a chatbot with **RAG** and mentioned that their prompt sometimes gives odd answers.
   - They asked for help to improve the prompt and resolve the issues.
- **Improve clarity to avoid contradictions**: Another member suggested that the odd answers might be due to contradictions in the instructions.
   - They recommended **rewriting** the instructions to be more clear.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1261101505021214831)** (2 messages): 

> - `RAG chatbot development`
> - `Prompt contradictions` 


- **Challenges in RAG chatbot development**: A member shared that they are developing a chatbot with **RAG** but sometimes receive odd answers due to unclear prompt instructions.
   - Another member suggested rewriting the **instructions** to be more clear to avoid contradictions.
- **Improving Chatbot Instructions**: A member highlighted **contradictions** in the chatbot's prompt instructions.
   - They advised rewriting the instructions to enhance clarity and avoid confusion.


  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1261056713180577952)** (35 messages🔥): 

> - `Command R Plus model use cases`
> - `AI news automation with Cohere`
> - `Cohere toolkit open source`
> - `Creating unique emojis`
> - `OpenArena on GitHub` 


- **Discussing Command R Plus model use cases**: **Mapler** asked community members about real-world use cases for the Command R Plus model and received several suggestions, including content generation for social media, podcast descriptions, and team communication.
   - *Sssandra* shared multiple ways she uses it in her daily routine, including an internal version integrated with Notion and Google Drive for community questions.
- **Automating AI news updates in Discord**: **Mapler** wants to automate AI news updates in a Discord channel using Command R Plus, **Lang chain**, and webhooks. **Karthik_99_** supported the idea and offered further assistance.
   - *Mapler* is considering writing a chat-GPT like interface with various tools and plans to iterate based on test feedback.
- **Cohere toolkit goes open source**: *Sssandra* announced that Cohere has open-sourced their chat interface on [GitHub](https://github.com/cohere-ai/cohere-toolkit) and mentioned upcoming OCI integration.
   - *Mapler* expressed excitement about using Cohere for personal projects and promised to post updates in the community channel.
- **Creating emojis using AI**: **Roazzy** expressed a desire to create unique emojis using AI, noting that the only current method involves drawing them manually.
   - **Karthik_99_** showed interest and asked for any solutions, highlighting the potential of such a feature.
- **Introducing OpenArena on GitHub**: *Le_mess* shared a project called [OpenArena](https://github.com/syv-ai/OpenArena) on GitHub where LLMs compete against each other for better dataset quality.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/syv-ai/OpenArena">GitHub - syv-ai/OpenArena</a>: Contribute to syv-ai/OpenArena development by creating an account on GitHub.</li><li><a href="https://github.com/cohere-ai/cohere-toolkit">GitHub - cohere-ai/cohere-toolkit: Cohere Toolkit is a collection of prebuilt components enabling users to quickly build and deploy RAG applications.</a>: Cohere Toolkit is a collection of prebuilt components enabling users to quickly build and deploy RAG applications. - cohere-ai/cohere-toolkit
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1261292230224777226)** (3 messages): 

> - `Embedding Model Cost Reduction`
> - `Project-Sharing Etiquette` 


- **Cohere's Embedding Model Slashes Costs by 40-70%**: A member announced that **Cohere's embedding model** significantly reduces costs by **40-70%**.
   - *Noice!* was the enthusiastic response from the community.
- **Reminder on Project-Sharing Etiquette**: A moderator reminded members to keep the discussion focused on **Cohere specific projects** and removed an off-topic post.
   - The moderator emphasized the importance of adhering to the channel's guidelines.


  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1261351437644857455)** (23 messages🔥): 

> - `Llama 3 release`
> - `OpenAI's new project Strawberry`
> - `Self-hosting large models`
> - `API vs self-hosting costs`
> - `Sensitive data handling with large models` 


- **Llama 3 with 405B parameters set to launch**: Llama 3 405B is expected to launch on **July 23**, almost a year after Llama 2's release. It is confirmed to be a multimodal model with further details available [here](https://www.theinformation.com/briefings/meta-platforms-to-release-largest-llama-3-model-on-july-23).
- **OpenAI's new project Strawberry leaked**: OpenAI is working on new reasoning technology under the code name **Strawberry**, as reported by Reuters. The project has similarities to a method called *Self-Taught Reasoner (STaR)* developed at Stanford in 2022.
- **Feasibility of self-hosting large models**: Hosting a 400B parameter model requires around **400GB VRAM**, equating to roughly 5 A100/H100 GPUs in an 8bit setup. This makes it feasible for large enterprises but challenging for smaller companies.
- **API rental vs self-hosting costs for large models**: For companies not fine-tuning the model, using prompting APIs is often more cost-effective than running their own GPUs. Renting GPUs from hyperscalers is preferred unless dealing with **sensitive data**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/casper_hansen_/status/1811805236085891527">Tweet from Casper Hansen (@casper_hansen_)</a>: @Teknium1 That coincides with ICML talk that @soumithchintala will be giving at 9am</li><li><a href="https://x.com/steph_palazzolo/status/1811791968600576271?s=46">Tweet from Stephanie Palazzolo (@steph_palazzolo)</a>: A Friday scooplet w/ @SylviaVarnham — Llama 3 405B is coming (and soon!)  The multimodal model is set to drop on July 23, about a year after the Llama 2 announcement.  More details here:  https://www....
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1261153391376072734)** (4 messages): 

> - `Soft-target distillation`
> - `Mistral-7B instruct-finetuning`
> - `Language model instruction tuning processes`
> - `AgentInstruct paper`
> - `KnowledgePile and AutoMathText datasets` 


- **Confusion over soft-target distillation methodology**: A user is puzzled by assertions in papers like Medusa about needing to run both models in parallel during **soft-target distillation**, questioning why inference can't be run sequentially while preserving teacher model probabilities.
   - They acknowledge that aligning internal representations may indeed require online models but argue for simpler cases.
- **Mistral-7B's instruct-finetuning questioned**: A user finds **Orca3/AgentInstruct** paper's improvements over Mistral-7B's instruct-tuning surprising and questions the strength of Mistral's own instruсt-finetuning dataset.
   - They plan to compare the **25M dataset** size to Mistral-7B's ift dataset and dive into both papers for more insights.
- **AgentInstruct's benchmark-maxxing scrutinized**: "xeophon:" comments that AgentInstruct looks like a **bench-maxxing** model.
   - Another user explains AgentInstruct’s workflow, including document transformation and complications, citing sources like [KnowledgePile](https://huggingface.co/datasets/Query-of-CC/Knowledge_Pile) and [AutoMathText](https://huggingface.co/datasets/math-ai/AutoMathText) as seed datasets.


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1261076457912930407)** (7 messages): 

> - `GPT-4 Pricing`
> - `OpenAI's AGI Progress`
> - `Self-driving Similarities`
> - `New GPT-4 Skills`
> - `OpenAI Revenue Speculation` 


- **GPT-4 Pricing Compared**: A member pointed out that **GPT-4** costs $20 a month, making a comparison to another service charging $5 a month.
   - This pricing contrast highlights the differing approaches and value propositions of various AI services.
- **OpenAI's AGI Progress Explained**: OpenAI shared a [five-level system](https://www.bloomberg.com/news/articles/2024-07-11/openai-sets-levels-to-track-progress-toward-superintelligent-ai?sref=P6Q0mxvj) to track its artificial general intelligence, or AGI, progress, highlighting that they are four steps away from human-level AI.
   - A user commented on this system, noting the resemblance to stages in developing self-driving technology.
- **Advanced Skills in GPT-4**: In a recent internal meeting, OpenAI demonstrated new skills in its **GPT-4** AI model that display human-like reasoning abilities.
   - According to a [Bloomberg article](https://www.bloomberg.com/news/articles/2024-07-11/openai-sets-levels-to-track-progress-toward-superintelligent-ai?sref=P6Q0mxvj), OpenAI's spokesperson emphasized that these tests are common internal practices aimed at pushing the AI's capabilities further.
- **OpenAI Revenue Speculation Addressed**: A Twitter user highlighted a circulating speculative report on **OpenAI's revenue**, which is based on chatbot summaries of public sources.
   - They provided a link to a more credible firsthand report on OpenAI's revenue published by [The Information](https://www.theinformation.com/articles/openais-annualized-revenue-doubles-to-3-4-billion-since-late-2023?utm_source=ti_app&rc=geicgp).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://qz.com/openai-five-level-system-human-intelligence-ai-1851588122">OpenAI says there are 5 &#x27;levels&#x27; for AI to reach human intelligence — it&#x27;s already almost at level 2</a>: The ChatGPT maker believes it&#x27;s on the first level, which is conversational AI</li><li><a href="https://x.com/aaronpholmes/status/1811870687037960467?s=46">Tweet from aaron holmes (@aaronpholmes)</a>: A lot of VCs are circulating a “report” today that speculates OpenAI’s revenue, based entirely on chatbot summaries of public web sources. If you want firsthand reporting on OpenAI’s revenue numbers, ...
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1261137284376559748)** (2 messages): 

> - `Indexing Kernel in tinygrad`
> - `PyTorch 2024 H2 Roadmaps` 


- **Indexing Kernel Introduced in tinygrad**: George Hotz introduced an [indexing kernel](https://x.com/__tinygrad__/status/1811578147982491938), which wouldn't normally be allowed in tinygrad due to an upstream LOAD of a LOAD.
   - He explained that it's all generated in the backend by folding the sum loop, creating a strict subset of the 'planned' memory accesses.
- **Proposal to Create PyTorch-like Roadmaps**: A member suggested creating roadmaps similar to the [2024 H2 plans by the PyTorch Team](https://dev-discuss.pytorch.org/t/meta-pytorch-team-2024-h2-roadmaps/2226).
   - *We should make similar roadmaps* was the key takeaway, emphasizing the benefits of having clear development pathways.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/__tinygrad__/status/1811578147982491938">Tweet from the tiny corp (@__tinygrad__)</a>: This is an indexing kernel, X[idxs]. Normally this kernel wouldn&#39;t be allowed in tinygrad, there&#39;s a LOAD upstream of a LOAD.  However, it&#39;s all generated in the backend by folding the sum...</li><li><a href="https://dev-discuss.pytorch.org/t/meta-pytorch-team-2024-h2-roadmaps/2226">Meta PyTorch Team 2024 H2 Roadmaps</a>: We’ve been thinking about how to share the roadmaps for the work we are doing on PyTorch here at Meta. We do planning on a half-year basis so these are some public versions of our 2024 H2 OSS plans fo...
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1261101576303411241)** (26 messages🔥): 

> - `Custom Weight and Bias in Network`
> - `Implementing Gradient Descent from Scratch`
> - `Performance Issues with Manual Gradient Descent`
> - `Tensor Operations and Realization`
> - `Indexing Tensors and Kernel Performance` 


- **Defining Custom Weight and Bias in Network Succeeds**: A user showed successful implementation of defining custom weight and bias in a network, resulting in the expected graph output.
   - They expressed satisfaction, stating: *love this... thanks for the help*.
- **Implementing Gradient Descent Highlights Challenges**: A user attempted to implement gradient descent from scratch in tinygrad but found it extremely slow and unclear without `optimizer.step`.
   - They shared [code snippets](https://github.com/karpathy/makemore/blob/988aa59e4d8fefa526d06f3b453ad116258398d4/names.txt) and sought advice on how to make the computation efficient.
- **Manually Realizing Tensors for Gradient Descent**: George Hotz suggested the use of `model.weights.assign(model.weights - lr * model.weights.grad).realize()` to manually realize tensor computations during gradient descent.
   - He emphasized that *you need the realize if you want the compute to happen*.
- **Debugging Slowness in Gradient Descent Steps**: The performance issues were identified as being due to tensor indexing during loss computation, especially with large datasets.
   - The `sparse_categorical_crossentropy` implementation using masking was suggested as a faster alternative.
- **Tensor Indexing Bugs Impact Kernel Performance**: A suggestion to use `-probas[:, Y_train]` for better performance led to an assertion error regarding the maximum index size.
   - This was identified as a bug since the expression resulted in **idx.max too big** errors.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/PaCmpygFfXo?t=6354)">The spelled-out intro to language modeling: building makemore</a>: We implement a bigram character-level language model, which we will further complexify in followup videos into a modern Transformer language model, like GPT....</li><li><a href="https://github.com/tinygrad/tinygrad/blob/6e0a5230786d41eabe9dc9e593b05997d3a1da73/tinygrad/engine/realize.py#L199-L202)">tinygrad/tinygrad/engine/realize.py at 6e0a5230786d41eabe9dc9e593b05997d3a1da73 · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad
</li>
</ul>

</div>
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1261051444686225549)** (15 messages🔥): 

> - `H100 performance`
> - `Attention masking in reward models`
> - `OpenRouter API usage`
> - `Flash attention versions`
> - `OpenArena open-source project` 


- **H100 performance excites members**: Members excitedly discussed the performance of **H100** GPUs, with one exclaiming *"H100 go brrrrr."*
   - The enthusiasm suggests significant performance improvements from this hardware.
- **Attention masking in reward models debated**: A member asked about the necessity of applying attention masking in **reward models**, seeking advice after admitting they hadn't been doing so.
   - A member speculated that it might not be related to **axolotl** training but was open to insights from others.
- **Seeking API access for WizardLM dataset**: A member inquired if anyone knew contacts at **openrouter.ai** for creating an open-source version of the ***WizardLM arena dataset**.
   - Another member mentioned working on a locally hosted open-source version using **ollama** and shared the [OpenArena project](https://github.com/syv-ai/OpenArena).
- **Concerns over Flash attn3 GPU compatibility**: Members expressed concerns that **Flash attn3** is currently only available for **H100** GPUs.
   - It was noted that **Flash attn2** was initially for **A100** and newer, but later made compatible with *3090's and 4090's*; hopes are high for a similar fix for Flash attn3.


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1261349385166524426)** (5 messages): 

> - `GaLore and Q-Galore`
> - `Dataset Shuffling` 


- **Q-Galore improves on GaLore with efficiency**: GaLore reduces memory usage by projecting weight gradients into a low-rank subspace, but it relies on time-consuming SVD operations and gives minimal improvements compared to LoRA in fine-tuning scenarios. [Q-Galore](https://huggingface.co/papers/2407.08296) addresses these issues by combining quantization and low-rank projection, substantially reducing memory usage and training time.
   - *Q-Galore*'s novel method surpasses the benefits of GaLore by observing that some gradient subspaces converge early while others change frequently.
- **Dataset shuffling occurs between epochs**: A member raised a historical dev question about why there is no explicit support for shuffling a single dataset prior to training.
   - Another member clarified that **batches are shuffled between epochs**, and the original poster acknowledged the clarification, mentioning they did not read that section.



**Link mentioned**: <a href="https://huggingface.co/papers/2407.08296">Paper page - Q-GaLore: Quantized GaLore with INT4 Projection and Layer-Adaptive
  Low-Rank Gradients</a>: no description found

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1261130241712193607)** (1 messages): 

> - `LoRA finetuning`
> - `Layer selection`
> - `Few-shot learning challenges` 


- **Optimizing LoRA finetuning for 72b models**: A member working on **LoRA finetuning** for dolphin-vision-72b (qwen 72b variant) sought advice on **layer selection** and efficiency, suspecting that applying LoRA to all layers might not be the most effective.
- **Layer targeting in LoRA finetuning**: Inquiring about experiments with targeting specific layers for LoRA finetuning on very large models, a member asked for approaches and results.
   - They were particularly interested in balancing **attention** and **feed-forward layers** to yield the best results.
- **Few-shot learning challenges post-finetuning**: A member noted challenges with **few-shot learning** post-finetuning and asked if others experienced this and how they addressed it.


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1261259383384571924)** (1 messages): 

> - `General-purpose multi-turn chat dataset`
> - `Dataset recommendations` 


- **Seeking highest quality multi-turn chat dataset**: A user asked for recommendations on the highest quality **general-purpose multi-turn chat dataset** available currently, mentioning that it doesn't need to be more than *10k rows*.
   - No specific datasets were suggested or mentioned in the responses.
- **Dataset recommendation request**: The request highlighted the need for a high-quality dataset that supports multi-turn conversations, suitable for general purposes.
   - Further discussion or recommendations were not provided in the given context.


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/)** (1 messages): 

wasamikirua: how can i push model to the hub after lora merge ?
  

---



### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1261047137655980052)** (15 messages🔥): 

> - `Training Data Concerns`
> - `Integrations (Beta)`
> - `Prompting Image Models`
> - `405b Model Update`
> - `Specialized Models` 


- **Understanding DeepInfra's data policy**: A member asked about keeping training data and mentioned that companies like DeepInfra keep the data.
   - [DeepInfra logs usage but does not train on user inputs](https://deepinfra.com/privacy), and the detailed policy can be referenced on their website.
- **Integrations (Beta) Opens New Possibilities**: Members inquired about the new **Integrations (Beta)** feature, designed to use custom API keys for various providers, including Groq.
   - Future expansions include integrations beyond model APIs.
- **Improve Weak Models by Prompt Placement**: A user shared a tip to place text prompts after images in the content for better responses.
   - This method helps weaker models accurately understand and answer the request.
- **405b Model Release Anticipation**: A user announced that **405b model** is expected to be released soon, causing excitement in the community.
   - [Bindu Reddy tweeted about the model's anticipated release](https://x.com/bindureddy/status/1811807596682355125?s=46), marking July 23rd as a historic day for open source AGI.
- **Debate on Specialized vs Generic Models**: A member questioned why companies like OpenAI and Anthropic do not create multiple specialized models instead of one generic model.
   - Alex Atallah agreed, suggesting that specialization should be considered, and asked which specialized models users would utilize the most.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/bindureddy/status/1811807596682355125?s=46">Tweet from Bindu Reddy (@bindureddy)</a>: Yay!!! July 23rd will go down in the history of open source AGI!  Can’t wait 💃💃💃</li><li><a href="https://deepinfra.com/privacy">DeepInfra Privacy Policy</a>: Run the top AI models using a simple API, pay per use. Low cost, scalable and production ready infrastructure.
</li>
</ul>

</div>
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1261081084335095878)** (2 messages): 

> - `Clip retrieval`
> - `Dataset access` 


- **Clip Retrieval No Longer Functional**: A user inquired about **clip retrieval no longer working** and asked if there is a new method to view/search the dataset.
   - Another user speculated that it was probably taken down because **the dataset has also been removed**.
- **Dataset Access Issues**: A concern was raised about the availability of the dataset following the clip retrieval issue.
   - It’s suggested that the dataset’s removal affected the clip retrieval functionality.


  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1261051560780365894)** (10 messages🔥): 

> - `MLP Architecture Efficiency`
> - `Memory Usage in Model Training`
> - `Nematron 340B Code Examples`
> - `AuraFlow Model Announcement`
> - `Alice in Wonderland (AIW) Problem` 


- **Single large MLP outperforms multiple small MLPs**: Discussion on a recent paper shows that one large MLP reused multiple times is more efficient than each block having its own smaller MLP.
   - *Trading params for flops* and surprising memory usage highlights sparked curiosity about this approach's bizarrely high memory demand.
- **Heavy memory usage with small models**: Member reports using 19GB just to train a quarter-million parameter model on a 128-batch-size CIFAR-100, calling it *stupidly memory-inefficient*.
   - This has prompted further investigation into why such a small model would require so much memory.
- **Searching for Nematron 340B code examples**: Inquiry about code examples for running the Nematron 340B reward model, specifically for loading and unloading parameters.
- **AuraFlow model release announced**: AuraFlow, a new flow-based text-to-image generation model, has been announced by [Fal AI](https://blog.fal.ai/auraflow/), marking a significant answer to claims that *open-source AI is dead*.
   - The model excels at following prompts and reaffirms the resilience of the open-source community.
- **AIW problem exposes LLM fragility**: Updated ArXiv paper on the Alice in Wonderland (AIW) problem shows dramatic breakdowns in state-of-the-art LLMs on simple tasks, as per [this submission](https://arxiv.org/abs/2406.02061).
   - This reveals that current benchmarks fail to show the fundamental weaknesses of these models, highlighting the need for better benchmarks and basic model capabilities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.fal.ai/auraflow/">Introducing AuraFlow v0.1, an Open Exploration of Large Rectified Flow Models</a>: Open-source AI is in jeopardy. As community interest in AI models skyrocketed over the past year, we noticed that development of new open-source foundational models came to a halt. Some even boldly an...</li><li><a href="https://arxiv.org/abs/2406.02061">Alice in Wonderland: Simple Tasks Showing Complete Reasoning Breakdown in State-Of-the-Art Large Language Models</a>: Large Language Models (LLMs) are often described as being instances of foundation models - that is, models that transfer strongly across various tasks and conditions in few-show or zero-shot manner, w...
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1261349288068382886)** (4 messages): 

> - `Agentic RAG Cookbooks`
> - `Cypher Snippet for Entity Deduplication`
> - `Knowledge Graph Creation Challenges`
> - `LlamaCloud Data Pipeline Management` 


- **Release of Agentic RAG Cookbooks**: LlamaIndex announced a collaboration with @jeffxtang from AIatMeta to release cookbooks on **agentic RAG**, including topics from routing and tool use to multi-document agent building.
   - The release was teased with a preview tweet linking to the [Twitter announcement](https://t.co/mBNZx9b1JO) and additional context [here](https://t.co/l2ztPRsAd8).
- **Cypher Snippet Eases Entity Deduplication**: A Cypher snippet by @tb_tomaz and Neo4j effectively performs **entity deduplication** by combining text embeddings and word manipulations.
   - Details and a [link to the example snippet](https://t.co/dAV2QuAoZH) were shared to showcase its utility in knowledge graph creation. More resources can be found on the [Neo4j GitHub repository](https://t.co/lMApLzMOMr).
- **Challenges of Automating Knowledge Graphs**: Automatically creating knowledge graphs with LLMs poses challenges, especially regarding **duplicate entities**.
   - A cool example by @tb_tomaz and others at Neo4j, demonstrating practical solutions, was shared in a tweet linking to [additional information](https://t.co/ruxdlhZOuK).
- **New Features on LlamaCloud for Data Pipelines**: LlamaCloud introduced features to manage data pipelines centrally, suitable for any **LLM application**, including simple RAG and complex workflows.
   - The new features include multi-user organization management, with more details in the [announcement tweet](https://t.co/F73Spljg0a).



**Link mentioned**: <a href="https://t.co/ruxdlhZOuK">blogs/llm/llama_index_neo4j_custom_retriever.ipynb at master · tomasonjo/blogs</a>: Jupyter notebooks that support my graph data science blog posts at https://bratanic-tomaz.medium.com/ - tomasonjo/blogs

  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1261239555562930177)** (6 messages): 

> - `Function calling on Gemini models`
> - `Error with Gemin-1.5-flash-latest model`
> - `Updating vertexai integration package`
> - `Indexing large code library`
> - `Reviewing spec documents with RAG` 


- **Function calling on Gemini models unclear**: A member inquired if **llamaindex supports function calling** on Gemini models, citing a [GitHub pull request](https://github.com/run-llama/llama_index/pull/14088).
   - Despite seeing the code, they encountered an error stating **'Model name models/gemini-1.5-flash-latest does not support function calling API'**.
- **Update vertexai integration package to resolve issues**: To resolve issues with Gemini model function calling, another member suggested **updating the vertexai integration package** using `pip install -U llama-index-llms-vertexai`.
- **Best practices for indexing large code libraries**: A member asked for advice on indexing a large code library for two different chatbot/queries: one for answering questions and one for code generation.
   - They inquired if **translating code into pseudocode in markdown format** would help the agent understand the library better.
- **Using RAG for reviewing spec documents**: A member considered using **RAG** (Retrieval-Augmented Generation) to review a spec document without sending all 2,000 lines of text to the LLM, aiming to save on tokens.



**Link mentioned**: <a href="https://github.com/run-llama/llama_index/pull/14088">Enable Function calling and agent runner for Vertex AI by wadave · Pull Request #14088 · run-llama/llama_index</a>: Description Changed are highlighed:  Enabled function calling for Vertex AI llama-index-integrations/llms/llama-index-llms-vertex/llama_index/llms/vertex/base.py Added tool/function roles for Gemin...

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1261358908543336539)** (6 messages): 

> - `Agent Invocation Issue`
> - `LiteLLM Error`
> - `Phi-3 Fast Function Calls`
> - `Open Interpreter GUI Integration` 


- **Agent Selection Causes Invocation Errors**: A user reported an issue where invoking an agent's 'chat' method, which calls `OpenInterpreter.chat()`, works standalone but fails when OpenInterpreter 'selects' the agent based on its role, resulting in an **APIConnectionError**.
   - *The error suggests passing the LLM provider explicitly.* Learn more [here](https://docs.litellm.ai/docs/providers).
- **Phi-3 Function Calls Speed Improvement**: A user shared excitement about achieving fast, reliable **function calls** from `phi-3`, with hopes of a fully local option for Fast Fourier runs soon.
- **Open Interpreter GUI Receives Expansive Upgrade**: A user integrated Open Interpreter into their GUI, supporting **branching chats, editable messages, auto-run code,** and chat saving.
   - The GUI also supports varied configuration parameters, with some noted limitations disclosed in the [Open Source Project](https://github.com/jbexta/AgentPilot).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.litellm.ai/docs/providers">Providers | liteLLM</a>: Learn how to deploy + call models from different providers on LiteLLM</li><li><a href="https://github.com/jbexta/AgentPilot">GitHub - jbexta/AgentPilot: Universal GUI for seamless interaction and management of AI workflows</a>: Universal GUI for seamless interaction and management of AI workflows - jbexta/AgentPilot
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/)** (1 messages): 

notnaton: https://youtu.be/SoFepHI6sQ0?si=2Y1zkghH2XyaN9_k
  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1261220652338380872)** (1 messages): 

> - `Self-hosted ML Telemetry`
> - `Langfuse`
> - `WandB`
> - `OpenLLMTelemetry` 


- **Comparison of Self-hosted ML Telemetry Solutions**: [Langfuse](https://langfuse.io), [WandB](https://wandb.ai), and [OpenLLMTelemetry](https://github.com/openllmtelemetry) all offer self-hosting solutions for ML telemetry.
   - These platforms provide the capabilities needed for **ML projects** and are recommended for users seeking self-hosted options.
- **Langfuse, WandB, and OpenLLMTelemetry Features**: Langfuse, WandB, and OpenLLMTelemetry all include the features necessary for **self-hosted ML telemetry**.
   - Users interested in these solutions should compare them based on specific project needs and requirements.


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[asia-tz](https://discord.com/channels/1238365980128706560/1240532179549945957/1261414721802862864)** (2 messages): 

> - `API key for OpenAI`
> - `Chatbot project tutorial` 


- **Request for OpenAI API Key for Chatbot Project**: A member asked if anyone had an unused OpenAI API key they could share for a chatbot project tutorial.
   - They emphasized the need was for creating a tutorial, indicating temporary use.
- **Chatbot Project Tutorial Needs API Key**: Another request for an OpenAI API key was made to help complete a chatbot project tutorial.
   - The member reiterated that the key is necessary for demonstration purposes only.


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[fireworks](https://discord.com/channels/1238365980128706560/1245126291276038278/1261040706097516618)** (1 messages): 

> - `Account Credits`
> - `User Query`
> - `Account ID`
> - `Credit Balance Check` 


- **User Inquiry on Credit Balance**: A user asked how to check their credit balance, tagging another user for assistance.
   - The user provided their account ID as **reneesyliu-571636** for reference.
- **Seeking Help for Account Status**: Another user query raised the issue of inability to check account status.
   - They included their account details to help resolve the issue quickly.


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/)** (1 messages): 

slac.eth6408: Do we know the date of OpenAI credit expiration?
  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1261387457652592732)** (1 messages): 

> - `Llamafile goes to Washington`
> - `Builders Accelerator`
> - `Upcoming Events`
> - `Open Source AI Definition` 


- **Llamafile goes to Washington**: [Udbhav Tiwari, Mozilla Global Policy Director, testified before the U.S senate](https://discord.com/channels/1089876418936180786/1260972784696295536) highlighting the need for openness in AI technologies.
- **Builders Accelerator applications still open**: Although the early application window for <@&1229573172018417674> has closed, the program is still accepting applications, as mentioned in [the previous announcement](https://discord.com/channels/1089876418936180786/1089876419926032396/1255588743599882260).
- **Upcoming Events to RSVP**: Join upcoming events like [Open Interpreter](https://discord.com/events/1089876418936180786/1260611047341953034) with LLMs running code, [Zero Shot Tokenizer Transfer](https://discord.com/events/1089876418936180786/1260289594729959455) hosted by Benjamin Minixhofer, and [AutoFix: Open Source issue fixer](https://discord.com/events/1089876418936180786/1245836053458190438) with engineer <@278455249239539712>.
- **Open Source AI Definition Draft v 0.0.8**: The [Open Source AI Definition Draft v 0.0.8](https://opensource.org/deepdive/drafts/the-open-source-ai-definition-draft-v-0-0-8) is open for comments and follows the OECD definition of AI system.
   - For more information, visit the [OSI's blog](https://blog.opensource.org/open-source-ai-establishing-a-common-ground/).



**Link mentioned**: <a href="https://opensource.org/deepdive/drafts/the-open-source-ai-definition-draft-v-0-0-8>).">The Open Source AI Definition &#8211; draft v. 0.0.8</a>: version 0.0.8 Leave comments for this text Note: This document is made of three parts: A preamble, stating the intentions of this document; the Definition of Open Source AI itself; and a checklist …

  

---


### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1261374633123385425)** (1 messages): 

> - `llama.cpp matmul/matvec`
> - `ggml-quants.c file`
> - `integer dotproducts`
> - `float activations` 


- **llama.cpp Matmul Mechanism Question**: A member inquired whether **llama.cpp** performs the inner dotproduct of the matmul/matvec as integers or floats, referencing the **ggml-quants.c** file which contains several integer dotproduct operations.
   - The user questioned if the activations are quantized before performing the matmul since **activations are typically floats**, making them curious about the process.
- **Floating Point vs Integer in ggml-quants.c**: Within **ggml-quants.c**, numerous integer dotproduct operations are noted, prompting a query about whether actual multiplication is done using floats instead of integers.
   - The concern is that performing matmul operations directly with integers would necessitate prior quantization of activations, which are usually in float format.


  

---



### **DiscoResearch ▷ #[disco_judge](https://discord.com/channels/1178995845727785010/1178996063537991752/1261436075466035321)** (2 messages): 

> - `LLM Arena`
> - `WizardLM Paper`
> - `OpenArena GitHub Repository` 


- **Introducing LLM Arena for Enhanced Dataset Quality**: An **LLM arena** has been created to pit two language models against each other with a third one acting as a judge, primarily using models from **Ollama** but compatible with any **OpenAI** endpoint.
   - This setup aims to **increase dataset quality** by leveraging competitive benchmarks, as seen on the project's [GitHub page](https://github.com/syv-ai/OpenArena).
- **Inspired by the WizardLM Paper**: **OpenArena** is based on the [WizardLM paper](https://www.microsoft.com/en-us/research/publication/arena-learning-build-data-flywheel-for-llms-post-training-via-simulated-chatbot-arena/) which proposes the **Arena Learning** method to build an efficient data flywheel for LLMs post-training.
   - This involves simulating iterative arena battles and leveraging **AI-annotated results** to enhance models in supervised fine-tuning and reinforcement learning.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.microsoft.com/en-us/research/publication/arena-learning-build-data-flywheel-for-llms-post-training-via-simulated-chatbot-arena/">Arena Learning: Build Data Flywheel for LLMs Post-training via Simulated Chatbot Arena - Microsoft Research</a>: Arena Learning: Build Data Flywheel for LLMs Post-training via Simulated Chatbot Arena</li><li><a href="https://github.com/syv-ai/OpenArena">GitHub - syv-ai/OpenArena</a>: Contribute to syv-ai/OpenArena development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1261180269172363345)** (1 messages): 

> - `Product coverage`
> - `Research coverage`
> - `Recommendation systems`
> - `Information Retrieval`
> - `Retrieval-Augmented Generation` 


- **Discussion on covering multiple areas**: A user expressed interest in covering both **product** and **research** topics for multiple crowds, such as **recommendation systems**, **information retrieval (IR)**, and **retrieval-augmented generation (RAG)**.
   - They are open to suggestions if anyone comes to mind and are keen on discussing **Elastic** with another user.
- **Interest in Elastic**: A user specifically mentioned wanting to chat about **Elastic** if others are interested.
   - They tagged another member and invited them to discuss further.



{% else %}


> The full channel by channel breakdowns have been truncated for email. 
> 
> If you want the full breakdown, please visit the web version of this email: [{{ email.subject }}]({{ email_url }})!
>
> If you enjoyed AInews, please [share with a friend](https://buttondown.email/ainews)! Thanks in advance!

{% endif %}
