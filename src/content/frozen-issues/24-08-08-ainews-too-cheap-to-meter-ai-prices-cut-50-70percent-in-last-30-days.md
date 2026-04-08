---
id: 7d8d0c98-cad0-44de-a76d-4882abd5a690
title: 'Too Cheap To Meter: AI prices cut 50-70% in last 30 days'
date: '2024-08-09T04:27:56.926040Z'
original_slug: ainews-too-cheap-to-meter-ai-prices-cut-50-70-in
description: >-
  **Gemini 1.5 Flash** has cut prices by approximately **70%**, offering a
  highly competitive free tier of **1 million tokens per minute** at
  **$0.075/mtok**, intensifying the AI model price war. Other significant price
  reductions include **GPT-4o** (~50% cut to **$2.50/mtok**), **GPT-4o mini**
  (70-98.5% cut to **$0.15/mtok**), **Llama 3.1 405b** (46% cut to
  **$2.7/mtok**), and **Mistral Large 2** (62% cut to **$3/mtok**). **Deepseek
  v2** introduced context caching, reducing input token costs by up to **90%**
  to **$0.014/mtok**. New model releases include **Llama 3.1 405b**, **Sonnet
  3.5**, **EXAONE-3.0** (7.8B instruction-tuned by LG AI Research), and
  **MiniCPM V 2.6** (vision-language model combining SigLIP 400M and Qwen2-7B).
  Benchmarks show **Mistral Large** performing well on ZebraLogic and
  **Claude-3.5** leading LiveBench. **FlexAttention**, a new PyTorch API,
  simplifies and optimizes attention mechanisms. **Andrej Karpathy** analyzed
  RLHF, highlighting its limitations compared to traditional reinforcement
  learning. Google DeepMind research on compute-optimal scaling was also
  summarized.
companies:
  - llamaindex
  - together-ai
  - deepinfra
  - deepseek-ai
  - mistral-ai
  - google-deepmind
  - lg-ai-research
  - llamaindex
  - llamaindex
  - llamaindex
models:
  - gpt-4o
  - gpt-4o-mini
  - llama-3-1-405b
  - mistral-large-2
  - gemini-1.5-flash
  - deepseek-v2
  - sonnet-3.5
  - exaone-3.0
  - minicpm-v-2.6
  - claude-3.5
  - gpt-4o-2024-08-06
topics:
  - price-cuts
  - context-caching
  - instruction-tuning
  - vision
  - benchmarks
  - pytorch
  - attention-mechanisms
  - reinforcement-learning-from-human-feedback
  - compute-optimal-scaling
people:
  - rohanpaul_ai
  - akhaliq
  - mervenoyann
  - sophiamyang
  - chhillee
  - karpathy
---


<!-- buttondown-editor-mode: plaintext -->**Gemini Flash is all you need?**

> AI News for 8/7/2024-8/8/2024. We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **28** Discords (**249** channels, and **2423** messages) for you. Estimated reading time saved (at 200wpm): **247 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

A simple list of **all the price cuts in the last 30 days in AI** (measured in "mtok" aka "per million tokens" - the bulk of the cost is usually input), by LMsys Elo/Rank:

- **Elo 1286 Rank 2**: [GPT-4o cut ~50%](https://buttondown.com/ainews/archive/ainews-gpt4o-august-100-structured-outputs-for/) from May to Aug (**$2.50/mtok**)
- **Elo 1277 Rank 3**: [GPT-4o mini](https://buttondown.com/ainews/archive/ainews-lskjd/) effectively **cut prices between 70-98.5%** depending if you compare with GPT3.5T or GPT4T (**$0.15/mtok**)
- **Elo 1264 Rank 4**: [Llama 3.1 405b](https://buttondown.email/ainews/archive/ainews-llama-31-the-synthetic-data-model/) was initially offered at $5/15 by Together AI - within 48 hours this was [cut 46% to $2.7/mtok](https://x.com/openrouterai/status/1816234833896694270?s=46) by DeepInfra with [Lepton not far behind](https://x.com/jiayq/status/1816246934925107393?s=46) (**$2.7/mtok**) 
- **Elo 1249 Rank 8**: [Mistral Large 2](https://buttondown.email/ainews/archive/ainews-mistral-large-2/) cut prices vs [Feb's Large v1](https://x.com/gblazex/status/1762127672673468566) by 62% (**$3/mtok**)
- **Elo 1228 Rank 17**: [Gemini 1.5 Flash cut ~70%](https://x.com/OfficialLoganK/status/1821601298195878323)   - on top of their existing [1 million tokens per minute free tier](https://reddit.com//r/LocalLLaMA/comments/1em9545/best_summarizing_llms_for_average_pcs/?utm_source=ainews&utm_medium=email) (**$0.075/mtok**)
- **Elo 1213 Rank 17**: [Deepseek v2 beats Gemini to a GA release of context caching](https://x.com/rohanpaul_ai/status/1820833952149487898), reducing cache hit input token price by a maximum 90% (**$0.014/mtok** (not a typo)). This is after their [original $0.14/mtok pricing which may have set off the price war in the last month](https://x.com/EMostaque/status/1813991810823340521)

Given Gemini 1.5's extremely generous free tier, every model below Lmsys Rank 17 - currently featuring things like Gemma 2, Nemotron 4, GLM 4, Reka Flash, Llama 3 7b, Qwen 72B and others - are effectively dead on arrival for most individual and team usecases.

The [Price-Intelligence frontier](https://x.com/swyx/status/1815892458519289946) advances by another order of magnitude in another quarter.



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

**AI Model Developments and Releases**

- **New Models and Capabilities**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1821214544284282947) reported on the release of Llama3.1 405b and Sonnet 3.5, available for free with Google Cloud's $300 credit. [@_akhaliq](https://twitter.com/_akhaliq/status/1821327180497842205) announced EXAONE-3.0, a 7.8B instruction-tuned model from LG AI Research, demonstrating competitive performance against other state-of-the-art open models of similar size. [@mervenoyann](https://twitter.com/mervenoyann/status/1821103721213722683) highlighted MiniCPM V 2.6, a vision-language model combining SigLIP 400M and Qwen2-7B, outperforming proprietary models on various benchmarks.

- **Model Performance and Benchmarks**: [@sophiamyang](https://twitter.com/sophiamyang/status/1821119082432712938) noted that Mistral Large is performing well on the ZebraLogic benchmark despite being smaller than other models. [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1821326039714246902) shared that Claude-3.5 remains at the top of LiveBench Benchmarks for the new GPT-4o-2024-08-06.

- **AI Tools and Frameworks**: [@cHHillee](https://twitter.com/cHHillee/status/1821253769147118004) introduced FlexAttention, a new PyTorch API allowing for many attention variants to enjoy fused kernels in a few lines of PyTorch code. This development aims to simplify and optimize various attention mechanisms in neural networks.

**AI Research and Insights**

- **RLHF and Model Training**: [@karpathy](https://twitter.com/karpathy/status/1821277264996352246) provided an in-depth analysis of Reinforcement Learning from Human Feedback (RLHF), discussing its limitations and comparing it to traditional Reinforcement Learning. He argued that RLHF is "just barely RL" and highlighted the challenges in applying it to large language models.

- **Compute-Optimal Scaling**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1821314621417922949) summarized a paper from Google DeepMind on compute-optimal scaling for test-time computation in large language models. The research introduces methods to adaptively allocate test-time compute based on prompt difficulty, potentially allowing smaller base models to outperform much larger ones.

- **Model Merging Techniques**: [@cwolferesearch](https://twitter.com/cwolferesearch/status/1821250560508465387) explained various model merging techniques, including linear merging, task vectors, TIES merging, and DARE merging. These methods allow for combining capabilities of multiple LLMs without additional training data or compute resources.

**AI Applications and Tools**

- **SAM 2 for Object Segmentation**: [@AIatMeta](https://twitter.com/AIatMeta/status/1821229074754498667) announced SAM 2, a unified model for real-time, promptable object segmentation in images and videos. [@swyx](https://twitter.com/swyx/status/1821298796841541956) highlighted that SAM 1 saved an estimated 35 years of time for users in just one year on images alone.

- **AI Avatars**: [@synthesiaIO](https://twitter.com/synthesiaIO/status/1821152878418944260) launched personal AI avatars, demonstrating their realism in a live event with 4,000+ attendees.

- **LlamaIndex Developments**: [@llama_index](https://twitter.com/llama_index/status/1821227063812223338) shared a tutorial on building a documentation chatbot using Firecrawl for web scraping and Qdrant for vector storage and retrieval.

**AI Ethics and Policy**

- **Structured Outputs and Safety**: [@AlphaSignalAI](https://twitter.com/AlphaSignalAI/status/1821226608314777708) reported on OpenAI's release of their most performant GPT-4o assistant model, featuring structured outputs with 100% reliability and improved token limits and pricing.

- **AI Safety Concerns**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1821151485293437237) summarized a paper on jailbreaking safety-tuned LLMs with human-fluent prompts, achieving high attack success rates while maintaining low perplexity.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Free Access to Advanced LLMs: Llama 3.1 405B and Sonnet 3.5**

- **Llama3.1 405b + Sonnet 3.5 for free** ([Score: 304, Comments: 108](https://reddit.com//r/LocalLLaMA/comments/1emddb4/llama31_405b_sonnet_35_for_free/)): Google Cloud is offering **free access** to **Llama 3.1 405B** and **Sonnet 3.5** models through their Vertex AI Model Garden, providing up to **$300** worth of API usage, which translates to approximately **20 million output tokens** for Sonnet 3.5 per Google account. A related project, the **Open Answer Engine**, demonstrates how to create a **405B model** with Google search functionality using this API service, as detailed in a Weights & Biases report.

- **[Experimenting llama3-s: An early-fusion, audio & text, multimodal model](https://homebrew.ltd/blog/can-llama-3-listen)** ([Score: 92, Comments: 16](https://reddit.com//r/LocalLLaMA/comments/1emjyq0/experimenting_llama3s_an_earlyfusion_audio_text/)): **Llama3-s**, an early-fusion multimodal model integrating **audio and text**, has been released for experimentation. The model, trained on **1.4 trillion tokens** of text and **700 billion tokens** of audio, demonstrates capabilities in **transcription**, **translation**, and **audio understanding** tasks, while also maintaining strong performance on text-only benchmarks.

**Theme 2. Optimized Inference and Quantization for ARM-based Processors**

- **Snapdragon X CPU inference is fast! (Q_4_0_4_8 quantization)** ([Score: 83, Comments: 39](https://reddit.com//r/LocalLLaMA/comments/1emd3bg/snapdragon_x_cpu_inference_is_fast_q_4_0_4_8/)): The **Snapdragon X CPU** demonstrates impressive inference speeds with **Q_4_0_4_8 quantization** for **Llama 3.1 8B**, achieving **15.39 tokens per second** on a **Surface Pro 11** with a **10-core Snapdragon X Plus chip**. The post provides instructions for optimizing performance, including using **-win-llvm-arm64.zip** releases, setting Windows power mode to **Best Performance**, and requantizing existing GGUF models to **Q4_0_4_8** using the **llama-quantize.exe** command, noting that these results are comparable to **MacBook M2 and M3** performance levels.

- **[LG AI releases Exaone-3.0, a 7.8b SOTA model](https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct)** ([Score: 144, Comments: 77](https://reddit.com//r/LocalLLaMA/comments/1emfm03/lg_ai_releases_exaone30_a_78b_sota_model/)): LG AI has released **Exaone-3.0**, a **7.8 billion parameter** language model achieving state-of-the-art performance across multiple benchmarks. The model demonstrates superior capabilities in **Korean and English** languages, outperforming larger models like **GPT-3.5** on certain tasks while being significantly smaller in size.

**Theme 3. Summarization Techniques and Model Comparison for Large Texts**

- **Best summarizing LLMs for average PCs?** ([Score: 68, Comments: 72](https://reddit.com//r/LocalLLaMA/comments/1em9545/best_summarizing_llms_for_average_pcs/)): The post discusses **summarizing LLMs** compatible with **consumer-grade hardware**, specifically an **Nvidia RTX 3060 12GB** GPU and **32GB DDR5 RAM**. The author recommends **Qwen2**, **InternLM**, and sometimes **Phi3 mini and medium 128k** for summarizing **20-25 thousand word chunks**, noting that larger LLMs are incompatible with their setup and that **Llama 3.1** underperforms for this task.
  - **Llama3.1** and **GLM-4-9b** are used for summarizing YouTube video transcripts. The process involves creating an outline of chapters, then generating detailed descriptions for each item, which works well for long content using a rolling window approach.
  - The free tier of **Gemini 1.5 Flash** offers impressive summarization capabilities with a **1 million token context window** and **1 million free tokens per minute**, as clarified by a user linking to [Google AI's pricing page](https://ai.google.dev/pricing).
  - **Obsidian's Copilot plugin** allows for easy summarization of selected text using local LLMs, offering a streamlined process for saving summaries directly within the application.


**Theme 4. Repurposing Mining Hardware for AI Workloads**

- **[Picked up a mining rig for testing . . .](https://i.redd.it/ikzm89f14dhd1.jpeg)** ([Score: 143, Comments: 62](https://reddit.com//r/LocalLLaMA/comments/1emw6eq/picked_up_a_mining_rig_for_testing/)): A user acquired a **mining rig** with **7x 3060 GPUs**, discovering it's a complete PC with a weak processor and RAM rather than just PSUs and risers. They're seeking advice on **loading an AI model** onto this rig and **distributing the output to a host LLM application**, aiming to repurpose the mining hardware for AI inference tasks.
  - **llama.cpp** can run **LLaMA 3.1 70B Q8** on the rig's **84GB VRAM**, with Q6 for more context. Users suggest trying smaller models first, starting with **2B** and scaling up to test performance.
  - Upgrading the **motherboard** and **CPU** is recommended, with suggestions for **dual E5 v3/v4 server CPUs** and boards supporting multiple PCIe slots. **PCIe Bifurcation splitters** can allow one 16x slot to handle multiple GPUs.
  - **vLLM** is recommended for distributed setup, while **ExLlamaV2** offers built-in generator/queue functionality. The rig's **single PCIe lane per GPU** may be a bottleneck, but once models are loaded into VRAM, CPU and system RAM usage is minimal.


## All AI Reddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI Model Improvements and Techniques**

- **Flux with LoRA dramatically improves photorealism**: In r/StableDiffusion, a post demonstrates how using Flux with LoRA significantly [enhances the realism of generated images](https://www.reddit.com/r/StableDiffusion/comments/1emrprx/feel_the_difference_between_using_flux_with/), particularly for skin textures and facial details. Users noted the first image looked indistinguishable from a real photo.

- **Midjourney to Runway video generation impresses**: A [post in r/singularity](https://www.reddit.com/r/singularity/comments/1emnyxq/midjourney_to_runway_is_scary_good/) showcases the impressive capabilities of using Midjourney images as input for Runway's video generation, highlighting the rapid progress in AI-generated video.

**OpenAI Developments and Speculation**

- **Project Strawberry teased**: OpenAI's social media posts hinting at ["Project Strawberry"](https://www.reddit.com/r/OpenAI/comments/1emwh93/whats_going_on/) sparked discussion and speculation. Some users suggested it could be related to improving ChatGPT's ability to count letters in words like "strawberry", which has been a known issue.

- **Potential new reasoning technology**: A [Reuters article](https://www.reuters.com/technology/artificial-intelligence/openai-working-new-reasoning-technology-under-code-name-strawberry-2024-07-12/) was shared, indicating OpenAI is working on new reasoning technology under the codename "Strawberry".

**AI Model Behavior and Limitations**

- **ChatGPT struggles with letter counting**: Multiple users tested ChatGPT's ability to count the number of 'r's in "strawberry", with the model consistently answering incorrectly. This highlighted ongoing limitations in certain types of reasoning tasks.

- **Tokenization impact on model performance**: Some knowledgeable users pointed out that the letter counting issue is related to how language models tokenize words, explaining why ChatGPT struggles with this seemingly simple task.

**Community Reactions and Discussions**

- **Skepticism towards OpenAI's marketing**: Several users expressed frustration with OpenAI's marketing tactics, viewing the "Strawberry" teasers as overhyped or distracting from other issues.

- **Debate on AI progress**: The posts sparked discussions about the current state of AI capabilities, with some users impressed by the rapid progress in image and video generation, while others pointed out persistent limitations in reasoning tasks.

---

# AI Discord Recap

> A summary of Summaries of Summaries by GPT4O-Aug (gpt-4o-2024-08-06)

**1. Model Performance and Optimization**

- **BiRefNet Surpasses RMBG1.4**: **BiRefNet** demonstrates superior performance for background removal compared to **RMBG1.4**, with enhanced high-resolution image segmentation capabilities as detailed in the [arXiv paper](https://arxiv.org/pdf/2401.03407).
  - Developed by [Nankai University](https://huggingface.co/ZhengPeng7/BiRefNet), this model employs bilateral reference techniques that significantly optimize image processing tasks.
- **Torchao v0.4.0 Boosts Optimization**: The release of **torchao v0.4.0** introduces **KV cache quantization** and **quantization aware training (QAT)**, enhancing low bit optimizer support.
  - The community discussed a GitHub issue regarding Intx Tensor Subclasses, inviting further input on the tracker to experiment with low bit quantization.
- **RoPE Optimization Simplifies Code**: Members analyzed the **RoPE** implementation, advocating for simplification by shifting to direct trigonometric operations instead of complex numbers.
  - This adjustment was seen as a move towards enhancing code clarity while retaining functional integrity in the training logic.


**2. Open Source AI Developments**

- **Harambe Revolutionizes Bug Hunting**: The introduction of **Harambe**, an open-source bug hunting tool, aims to streamline API analysis using LLMs to generate API endpoint suggestions.
  - This shift from traditional fuzzing techniques provides a more efficient method for identifying potential issues in code.
- **EurekAI Platform Launches for Researchers**: [EurekAI](https://eurekai-web-app.vercel.app/signup) is introduced as a cross-collaboration platform for researchers, aiming to streamline the research process with AI features to enhance productivity.
  - Currently in alpha, it promises functionalities such as project creation and integrated journaling designed to foster research engagement.
- **Midjourney CEO Critiques Open Source**: Midjourney CEO expressed **skepticism towards open source**, arguing that local models can't compete with their service using **64 GPUs**, and dismissed **ControlNet** as a lone success.
  - Critics countered that Midjourney's product is akin to **inferior versions** of what open source can achieve, highlighting **overfitting** issues in **Flux**: *'it just has a sort of plastic look to it.'*


**3. AI Infrastructure and Market Dynamics**

- **Hugging Face Expands with XetHub Acquisition**: Hugging Face announced the acquisition of [XetHub](https://www.forbes.com/sites/richardnieva/2024/08/08/hugging-face-xethub-acquisition/) to enhance its collaboration infrastructure for large models, aiming for better dataset management.
  - CEO Clem Delangue highlighted that this move is critical for scaling AI model development and unifying their operational strategies.
- **OpenAI's Price Cuts Ignite Competition**: OpenAI is reportedly implementing a **70% price reduction** on its GPT-4o model, stirring substantial interest across the industry.
  - This drastic price shift could lead to revised pricing strategies among competitors in the AI model space.
- **Vercel Outage Impacts OpenRouter**: Vercel currently faces intermittent outages impacting the OpenRouter service, as detailed in their [status update](https://x.com/OpenRouterAI/status/1821267624228966781).
  - After several updates, services were stable again by **3:45 PM ET**, with ongoing monitoring.


**4. Prompt Engineering and Fine-tuning**

- **Self-Discover Prompting Gains Attention**: A member highlighted the potential of **Self-Discover** prompting, asserting its power and effectiveness beyond traditional Chain-of-Thought (CoT) approaches.
  - They emphasized its applicability in crafting customized prompts that yield better outputs.
- **RAG Pipeline Needs Enhanced Observability**: Concerns surfaced about the **RAG pipelines** needing better observability to capture query-time traces and the significance of proper document chunking.
  - Improper context chunking could lead to retrieval issues, as emphasized by a [tweet](https://twitter.com/llama_index/status/1821332562310205918).
- **Optimizing Chat History for LLMs**: Discussion centered around implementing a **custom function** to limit chat history for LLM applications, aimed at improving performance.
  - Maintaining user-specific context was identified as a key factor in streamlining chat retention across different user interactions.


**5. AI Applications and Tools**

- **SAM 2 Pod Launch is Live**: The latest episode of the [Latent Space podcast](https://x.com/latentspacepod/status/1821296511260504408) features **SAM 2**, with insights from **Nikhila Ravi** and **Joseph Nelson**.
  - Listeners learned that **49 million images** were labeled using **SAM** on RoboFlow, which saved an estimated **35 years** of user time.
- **Stable Diffusion Optimizes in Python**: Members discussed utilizing the **Diffusers** library to implement **Stable Diffusion** in Python, focusing on optimizing performance and VRAM usage.
  - They stressed the importance of setting parameters correctly to attain the **desired output quality**.
- **MiniCPM-V 2.6 Shines in Performance Tests**: **MiniCPM-V 2.6** has been reported to outperform its competitors, including **Gemini 1.5 Pro**, **GPT-4V**, and **Claude 3.5 Sonnet**, particularly in multi-image applications.
  - For more details, members shared links to its [Hugging Face page](https://huggingface.co/openbmb/MiniCPM-V-2_6) and the [GitHub repository](https://github.com/OpenBMB/MiniCPM-V).

---

# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **4bit GGUF Models Present Loading Challenges**: Discussions arose around the **4bit GGUF models**, noting potential **precision loss** when using `load_in_4bit` during model loading, as highlighted by the occurrence of OOM errors without this option.
   - While **4bit** decreases VRAM consumption, the trade-off in performance needs careful consideration before implementation.
- **Issues Arise with PPO Trainer Implementation**: A member reported **negative KL divergence errors** while attempting to use a customized binary reward function with the **PPO Trainer**.
   - Exploring **DPO** as a simpler alternative raised concerns regarding its performance compared to **PPO** among members.
- **Unsloth Rolls Out Multi-GPU Support**: Confirmation of **multi-GPU support** rollout for trusted Unsloth users could lead to reduced VRAM consumption and increased processing speeds.
   - Debates ensued about whether this feature would be made available in open-source repositories or remain exclusive to paid subscriptions.
- **Successful Quantization of Mistral Models**: Insights were shared on quantizing the **123B Mistral-Large-Instruct-2407** model, achieving a size reduction with minimal accuracy drop using the **EfficientQAT** algorithm.
   - This optimization reinforces the feasibility of improving model efficiency without substantial output degradation.
- **Harambe: The New Bug Hunting Assistant**: The introduction of **Harambe**, an open-source bug hunting tool, aims to streamline API analysis using LLMs to generate API endpoint suggestions.
   - This shift from traditional fuzzing techniques provides a more efficient method for identifying potential issues in code.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **BiRefNet surpasses RMBG1.4**: **BiRefNet** demonstrates superior performance for background removal compared to **RMBG1.4**, with enhanced high-resolution image segmentation capabilities as detailed in the [arXiv paper](https://arxiv.org/pdf/2401.03407).
   - Developed by [Nankai University](https://huggingface.co/ZhengPeng7/BiRefNet), this model employs bilateral reference techniques that significantly optimize image processing tasks.
- **Launch of EurekAI Platform**: [EurekAI](https://eurekai-web-app.vercel.app/signup) is introduced as a cross-collaboration platform for researchers, aiming to streamline the research process with AI features to enhance productivity.
   - Currently in alpha, it promises functionalities such as project creation and integrated journaling designed to foster research engagement.
- **Performance Evaluation of AI Models**: Members compared pre-trained translation models like [Facebook's M2M100](https://huggingface.co/facebook/m2m100_418M) and [SeamlessM4T](https://huggingface.co/docs/transformers/model_doc/seamless_m4t_v2), which showed promising prospects in multi-language translations.
   - Discussions highlighted differences in transcription capabilities between SeamlessM4T-v2 and Whisper models, with a focus on real-world usability.
- **Exciting Updates in Gradio v4.41**: The release of **Gradio v4.41** introduces notable features such as **full screen images** for `gr.Image`, enhancing output viewing with improved user interaction mechanisms.
   - The update also strengthens security against unauthorized access and XSS attacks, providing a more robust framework for deploying applications.
- **Papers with Code Resource Insights**: A member highlighted [Papers with Code](https://paperswithcode.com/sota) as an essential resource for summarizing state-of-the-art performance in computer vision, featuring **11,272 benchmarks** and **137,097 papers with code**.
   - This invaluable platform aids users in exploring various machine learning applications, enhancing literature comprehensibility.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **BPF Insights for CUDA Profiling**: A member inquired if anyone was using **BPF** to profile **CUDA**, with some stating that **eBPF** lacks visibility on GPU activities, being limited to the OS kernel.
   - Concerns were raised about its efficacy, with members suggesting alternatives like **Nsight Compute** and **Nsight Systems** for comprehensive GPU application monitoring.
- **Attention Gym Links & FlexAttention**: Members reported a malfunctioning link for **Attention Gym**, expressing appreciation for its detailed content on softcapping.
   - Additionally, discussions emerged about integrating **FlexAttention** into HF models, indicating plans to wait for PyTorch version **2.5** for smoother integration.
- **torchao v0.4.0 is Here**: The announcement of **torchao v0.4.0** brought enhancements such as **KV cache quantization** and **quantization aware training (QAT)**, with excitement about its low bit optimizer support.
   - Community engagement involved a GitHub issue regarding Intx Tensor Subclasses for low bit quantization experimentation, inviting further input on the tracker.
- **Memory Usage and KV Cache Optimization**: A member's implementation of **KV Cache** optimized memory usage, enabling full bfloat16 fine-tuning on a single **80GB GPU**, albeit at the edge of memory limits.
   - Discussions suggested exploring managed memory to alleviate constraints while preparing pull requests focused on code cleanup and maintainability.
- **RoPE Optimization Discussions**: Members analyzed the **RoPE** implementation, advocating for simplification by shifting to direct trigonometric operations instead of complex numbers.
   - This adjustment was seen as a move towards enhancing code clarity while retaining functional integrity in the training logic.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro reduces daily limits**: **Perplexity Pro** users reported a reduction in the daily limit from **600 to 450**, raising frustration regarding communication of changes.
   - One member expressed distrust, stating they received no prior notifications about this shift.
- **API outages causing access issues**: Users are facing **major outages** with the **Perplexity API**, leading to concerns about the scope of the problem.
   - Reports indicate that some users are resolving issues via VPNs to different regions, suggesting potential geo-based discrepancies.
- **Google's antitrust ruling shakes market**: On **August 5, 2024**, a U.S. court ruled against Google for maintaining an illegal monopoly, a significant win for the **Department of Justice**.
   - The ruling confirmed that *'Google is a monopolist'* and outlined unlawful practices that maintain its market dominance.
- **Discussions on quantum theories in neuroscience**: Research into **quantum entanglement** in the brain is sparking debate, particularly around theories like **Orch-OR**, which suggest cognitive influence.
   - Skeptics argue that the brain's **warm, wet** conditions may not support sustained quantum states.
- **Non-English responses lacking coherence**: Users noted that prompts in non-English languages often produce **incoherent** responses, highlighting limitations in multilingual processing.
   - One instance in French led to repetitive outputs, raising concerns about the model's robustness across diverse languages.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Optimize Stable Diffusion in Python Projects**: Members discussed utilizing the **Diffusers** library to implement **Stable Diffusion** in Python, focusing on optimizing performance and VRAM usage.
   - They stressed the importance of setting parameters correctly to attain the **desired output quality**.
- **Upgrade Your Old PCs for AI Work**: A user sought advice on upgrading their outdated PC setup for AI tasks, looking for affordable components that wouldn't require a complete overhaul.
   - Suggestions included using **Fiverr** for assembly assistance and considering barebones prebuilt PCs as alternatives.
- **Face Swapping on Intel CPUs**: A user requested recommendations for face swapping techniques compatible with **Intel CPUs**, expressing a willingness to pay for expert help.
   - This highlighted the demand for practical solutions targeting users with less powerful hardware configurations.
- **Enhancing Images with SAM Workflow**: The community shared insights on utilizing the **SAM** detector to improve **image detailing**, enabling enhanced workflows.
   - One member emphasized detailing beyond just people, including backgrounds and structures, broadening the potential use cases.
- **NSFW Generation on Mac - Web Tools Needed**: A user asked for the best web-based tools for **NSFW content generation** that would work efficiently on a **MacBook Air M2 with 16GB RAM**.
   - The discussion included performance implications tied to model complexity and the benefits of local installation based on hardware capabilities.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **NVIDIA Cards untouched by current issues**: Current performance issues impact **CPUs** only, garnering relief from members regarding their **NVIDIA Cards**.
   - Discussions highlighted preferences for CPU vs. GPU setups, showcasing advantages of CPU-driven workloads.
- **CPU usage reports create confusion**: A conversation emerged about CPU usage numbers exceeding 100%, explained by applications reporting total usage based on core counts.
   - Members pointed out varied reporting standards among operating systems, leading to prevalent misunderstandings.
- **Dual GPUs not speeding up inference**: Members confirmed that **LM Studio** supports dual GPUs, but inference speed remains akin to a single GPU configuration.
   - Recommendations surfaced for hardware improvements to enhance token throughput for better performance.
- **Performance debate: 4090 vs. 3080**: User dissatisfaction was voiced over the **4090** performing similarly to the **3080**, with only a **20 ms per epoch** training speed advantage.
   - While the **4090** excels in gaming, others highlighted the **3080**'s efficiency in handling models under 8B.
- **Limited VRAM hampers model choices**: **2GB VRAM** proves insufficient for most models, resulting in poor performance with low VRAM options.
   - Users noted the necessity of splitting larger models across **VRAM** and **system RAM**, which significantly constrains efficiency.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI Releases GPT-4o System Card**: OpenAI shared the [GPT-4o System Card](https://openai.com/index/gpt-4o-system-card/), detailing assessments aimed at tracking **frontier model risks** and outlining audio capabilities with preset voices.
   - This card ensures proper guardrails against harmful content, enhancing user trust and understanding.
- **Free Users Access DALL·E 3**: ChatGPT Free users can now create up to **two images per day** using **DALL·E 3**, making content generation more accessible.
   - This feature enables personalized creative outputs for projects such as presentations and custom cards through seamless requests.
- **Ongoing Website Access Problems**: Multiple users reported connectivity issues accessing OpenAI's main site, resulting in persistent errors and intermittent accessibility.
   - This situation confirms growing frustration among members and unexpected difficulties across the community.
- **Confusion over Message Quotas**: Members expressed frustration regarding early message quota limits when using the platform, particularly in relation to the **GPT-4o**.
   - This experience led to discussions on the inconsistency of hitting limits unexpectedly, affecting user interaction.
- **Struggles with OpenAI Python SDK**: Users faced challenges replicating results using the OpenAI Python SDK, especially when encountering discrepancies in Python versions.
   - This indicated potential compatibility issues that hinder accurate output across varying coding environments.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **MindSearch AI Enhances Information Retrieval**: The paper *MindSearch: Mimicking Human Minds Elicits Deep AI Search* presents an Agentic AI that improves information retrieval through a dual-system approach with *WebPlanner* and *WebSearcher*, outpacing current search models.
   - This innovative structure effectively handles complex queries, demonstrating significant enhancements in intelligent information seeking.
- **Tavus Phoenix Model Takes Video Generation by Storm**: Tavus launched the **Phoenix** model that creates hyper-realistic talking head videos with the ability to synchronize **natural face movements** using advanced techniques.
   - Developers can access the *Phoenix* model through Tavus' **Replica API**, enabling diverse and high-level customizations for video content.
- **Models Crash on Upside Down Text**: Various models like Mistral and ChatGPT fail to generate coherent upside down text, while Claude Opus and Sonnet 3.5 handle it effortlessly with accurate outputs.
   - These observations highlight Claude models' superior capabilities, particularly in generating and rewriting upside down texts without errors.
- **Community Discusses AI Discord Resources**: A member shared a [Reddit post](https://www.reddit.com/r/nousresearch/comments/1elmrjr/most_helpful_ai_discordscommunities/?share_id=L2tAJZE66RY4dOPfIbiMw) listing several useful AI Discord channels, including **Replete-AI** and **Unsloth**.
   - These resources provide varied insights and support for those navigating the AI landscape within Discord.
- **Claude API Faces Server Overload Issues**: Users pointed out that the Claude API frequently gives overload messages during peak usage times, which disrupts their workflow.
   - Uncertainty remains on whether these issues stem from server limitations or bans affecting access.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **LM Harness Dataset Requirements Clarified**: A member inquired about the required format for datasets intended for **LM Harness**, questioning the necessary dictionary keys. They were directed to [YAML files](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/gsm8k) for structured guidance on key design.
   - This emphasizes the flexibility in formatting, which is crucial for developers working on dataset integration.
- **Debating CBRN Risks of AI Models**: Members discussed whether models can advise on chemistry without CBRN risks, highlighting concerns that filtering might jeopardize scientific capabilities.
   - The discussion pointed out that knowledgeable users might still extract harmful info, challenging the effectiveness of current filtration strategies.
- **Consequences of Filtering Pretraining Data**: Participants argued that erasing 'bad' data could diminish the model's overall comprehension and alignment effectiveness.
   - It was mentioned that lacking negative examples might impede the model's capacity to avoid harmful activities, raising concerns over competency regression.
- **Frustrations with AI Journalism**: Members shared their dissatisfaction with how journalists represent AI, often emphasizing sensational risks without adequate context.
   - This creates broader concerns about the safety narratives around AI outputs and their potential misrepresentation.
- **Searching for Open Source Reward Models**: A query came up regarding effective **open source Process Based Reward Models** for verifying mathematical tasks.
   - This underlines a pressing need for reliable verification tools within the domain of math problem-solving.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Hugging Face Expands with XetHub Acquisition**: Hugging Face announced the acquisition of [XetHub](https://www.forbes.com/sites/richardnieva/2024/08/08/hugging-face-xethub-acquisition/) to enhance its collaboration infrastructure for large models, aiming for better dataset management.
   - CEO Clem Delangue highlighted that this move is critical for scaling AI model development and unifying their operational strategies.
- **Qwen2-Math Dominates Math Tasks**: The newly launched [Qwen2-Math model series](https://qwenlm.github.io/blog/qwen2-math) by Alibaba outperforms both GPT-4o and Claude 3.5 in specialized math tasks.
   - This marks a significant leap for math-specific language models, indicating potential shifts in domain-specific applications.
- **AI Infrastructure Unicorns on the Rise**: A discussion series reveals how AI infrastructure builders like Hugging Face and Databricks are shaping generative AI markets.
   - Hugging Face's recent financing efforts position it to rival GitHub in the open-source domain, reflecting a robust growth strategy.
- **OpenAI's Price Cuts Ignite Competition**: OpenAI is reportedly implementing a **70% price reduction** on its GPT-4o model, stirring substantial interest across the industry.
   - This drastic price shift could lead to revised pricing strategies among competitors in the AI model space.
- **Clarifications on Token Count for GPT-4**: Reports establish that **GPT-4** utilizes **10 trillion tokens**, a figure corroborated by multiple sources in the chat.
   - Despite this consensus, members labeled GPT-4 as **ancient technology**, suggesting the fast-paced evolution of model capabilities.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Fixing LangChain in AWS Lambda**: A user faced **pydantic** errors when trying to import **LangChain** modules in AWS Lambda with Python **3.12** runtime, highlighting potential version conflicts.
   - Suggestions included double-checking the **lambda layer** setup to resolve the import issues.
- **Optimizing Chat History for LLMs**: Discussion centered around implementing a **custom function** to limit chat history for LLM applications, aimed at improving performance.
   - Maintaining user-specific context was identified as a key factor in streamlining chat retention across different user interactions.
- **LangChain vs. Other Frameworks Debate**: Users expressed frustration that switching from **OpenAI** to **Anthropic** with LangChain required substantial code rewrites due to functional differences.
   - Participants agreed that despite LangChain's abstraction, specific adjustments remain necessary based on the behavior of individual LLMs.
- **LLM Reliability Concerns**: Concerns were raised about **Claude 3.5** experiencing internal server errors, stressing the reliability of AI systems in production.
   - This led to broader discussions on whether LangChain is the right choice for stable AI system implementations.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **GPT-4o enhances input and output capabilities**: The **GPT-4o** model can process text, audio, image, and video, significantly boosting versatility and response speed, akin to human interaction.
   - It's also **50% cheaper** in API usage and shows improved performance across multiple languages.
- **Gemini 1.5 Flash slashes pricing**: GoogleAI cut the pricing for **Gemini 1.5 Flash** by about **70%**, making it far more accessible for developers.
   - The **AI Studio** is now available for all workspace customers, facilitating better experimentation with new languages.
- **DALL·E 3 opens up for Free users**: ChatGPT Free users can now generate **two images per day** with **DALL·E 3**, improving content creation accessibility.
   - While this feature is welcomed, some skepticism still exists regarding its broader applications.
- **Mistral Agents broaden functional integration**: **Mistral Agents** can now utilize Python in various workflows, highlighting their greater adaptability.
   - Users are keen on features that facilitate API consumption, enhancing real-world applications.
- **SAM 2 Pod Launch is live**: The latest episode of the [Latent Space podcast](https://x.com/latentspacepod/status/1821296511260504408) features **SAM 2**, with insights from **Nikhila Ravi** and **Joseph Nelson**.
   - Listeners learned that **49 million images** were labeled using **SAM** on RoboFlow, which saved an estimated **35 years** of user time.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Midjourney CEO critiques open source**: Midjourney CEO expressed **skepticism towards open source**, arguing that local models can't compete with their service using **64 GPUs**, and dismissed **ControlNet** as a lone success.
   - Critics countered that Midjourney's product is akin to **inferior versions** of what open source can achieve, highlighting **overfitting** issues in **Flux**: *'it just has a sort of plastic look to it.'*
- **ASL language model concept emerges**: A user proposed developing an app to translate **speech to ASL**, considering the challenge of training a model with images of signs.
   - Suggestions included fine-tuning existing models, and another user discussed refining voice recognition models to use **emojis** for representing hand gestures.
- **Synthetic voice dataset idea proposed**: A member proposed using **so-vits-svc** to create synthetic datasets by transforming voices in audio files, aiming to enhance variety while retaining content.
   - This approach could facilitate capturing a wider range of **emotions** in voice representation and improve model differentiation in **demographic classifications**.
- **Flux model discussions continue**: Users reflected on **Flux**, with some labeling it 'a fun toy' that hasn’t made significant advances, raising concerns about its **overfitting**.
   - The ongoing dialogue emphasized the need for more intentional **fine-tuning** comparing Flux to Midjourney.
- **Multiple AI applications for accessibility**: Various suggestions for AI aimed at enhancing accessibility were shared, including a **privacy-respecting** IP Relay app for speech recognition.
   - Members focused on local inference techniques to help those with hearing impairments, showcasing a robust interest in impactful AI applications.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Multi-backend Refactor Installed Smoothly**: One member confirmed they successfully installed the **multi-backend-refactor** without any issues and is ready to monitor future developments.
   - This smooth installation process boosts confidence in its stability and utility in ongoing projects.
- **Google Gemini Slashes Prices**: A member shared a [YouTube video](https://www.youtube.com/watch?v=3ICC4ftZP8Y) titled 'Google Gemini Insane Price Cuts!!!', featuring reductions on **Gemini 1.5 Flash**.
   - The video outlines substantial markdowns, and viewers can find additional details in the [Google Developers blog](https://developers.googleblog.com/en/gemini-15-flash-updates-google-ai-studio-gemini-...).
- **Call for H100s in the Metaverse**: A humorous remark was made suggesting that **Zuck** needs to deliver more **H100** GPUs in the metaverse, highlighting demand for advanced resources.
   - This statement underscores the ongoing need for high-performance computing in virtual environments.
- **Training with 38k Dataset**: One member reported training their model with a **38k** item dataset, taking **32 hours** on an **RTX 4090**.
   - They raised concerns that the **learning rate** in their current setup might be too high.
- **Correct Prompt Formatting Discussion**: Members stressed the necessity of the **Alpaca format** for task-specific prompts during inference to ensure consistency.
   - They emphasized that output during chatting must mirror the format utilized in fine-tuning for optimal results.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Almosnow seeks API file upload guidance**: A member wanted to replicate the PDF querying functionality from the UI on coral.cohere.com using the API but struggled to find the relevant documentation.
   - *An error occurred: could not convert string to float: 'float_'* indicates an underlying issue with input formatting.
- **Mapler provides RAG resources**: Mapler responded with resources on using **Retrieval-Augmented Generation** via the Cohere API, linking to a [blog post](https://cohere.com/llmu/rag-start) and additional documentation.
   - They shared a code snippet for producing grounded answers, enhancing understanding of RAG use.
- **Azure AI Search integration woes**: Users reported inconsistent results with **Cohere embeddings** in Azure AI Search, despite vectorized data successfully being indexed.
   - [Integrated vectorization with models from Azure AI Studio](https://learn.microsoft.com/en-sg/azure/search/vector-search-integrated-vectorization-ai-studio?tabs=cohere) was highlighted as a potential resource for addressing issues.
- **Cohere-toolkit enhancements for tool activation**: A discussion emerged about enabling a tool by default in **Cohere-toolkit** by adding `always use the <tool> tool` to the preamble.
   - It was noted that the tool must be listed for it to function correctly during calls.
- **User experiences custom deployment hurdles**: A member shared attempts to modify `invoke_chat_stream` for default tool loading in their custom deployment with limited model selection.
   - Confusion arose due to UI discrepancies showing tools not activated, emphasizing a need for clarification in model feedback.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Announcements on the Horizon**: An announcement for LlamaIndex is set to happen in **5 minutes**, which generated buzz among members in the announcements channel.
   - Members are eagerly awaiting highlights or updates that might come from this event.
- **RAG Pipeline Needs Enhanced Observability**: Concerns surfaced about the **RAG pipelines** needing better observability to capture query-time traces and the significance of proper document chunking.
   - Improper context chunking could lead to retrieval issues, as emphasized by a [tweet](https://twitter.com/llama_index/status/1821332562310205918).
- **LongRAG Paper Comparison Ignites Discussion**: The shared **LongRAG paper** indicates that long-context models outperform **RAG** when adequately resourced, prompting discussions on its methodologies.
   - Members expressed a desire for comparisons involving **Claude 3.5** and insights from Lance of LangChain, enhancing community discourse.
- **Self-Routing Technique Revolutionizes Efficiency**: The **Self-Route method** introduced in the LongRAG paper routes queries based on self-reflection, cutting costs while preserving performance.
   - Proposals for **parent-document retrieval** leveraging metadata surfaced to boost retrieval systems, highlighting reliability challenges in metadata labeling.
- **Workflows Abstraction Stirs Excitement**: The team demonstrated the ease of building complex AI applications with **Workflows**, particularly in rebuilding LlamaIndex's Sub-Question Query Engine showcased in a [new video](https://twitter.com/llama_index/status/1821575082516660440).
   - This positions Workflows effectively for deploying intricate query engines in generative AI applications.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Concerns on LLAMA 3 Generation Quality**: Using the **LLAMA 3 8B instruct model**, a member found that prompting with 'anything' led to unexpected outputs, raising concerns about generation quality.
   - They directed others to share experiences or refer to [GitHub issue #1285](https://github.com/pytorch/torchtune/issues/1285) for further discussion.
- **Evaluating RTX A4000 and A2000 for Fine Tuning**: The discussion highlighted performance characteristics of **RTX A4000** and **RTX A2000**, each equipped with **16GB** of memory, revealing underwhelming fine-tuning results with **1.5B** models.
   - One member suggested increasing the default batch size to better manage memory costs, possibly fitting workloads into **12GB**.
- **Memory Optimization Parameters Under Review**: There’s ongoing **guesswork** on memory optimization parameters, with mentions of **LoRA** not currently being prioritized despite its effectiveness.
   - The potential for optimization is evident, especially for members using GPUs with **8GB VRAM**, who could experience improvements of over **2x**.
- **Discussion on RLHF Cleanup**: A member raised questions about necessary cleanups for **RLHF** prior to public sharing, recalling earlier notes on required adjustments.
   - They expressed willingness to collaborate on creating a **tutorial** or **blog post**, acknowledging the effort involved.
- **Plans to Publicize and Document Work**: Eager to initiate discussions on **publicizing** their work and developing **documentation** or **tutorials**, a member outlined a loose roadmap.
   - They welcomed community input and assistance to enhance these efforts, indicating a collective approach.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Freedom to Build Up on AI Infrastructure**: Members discussed that it's acceptable to deploy anything on **AI Infrastructure** as long as there's no intent to commercialize, referencing the [pricing page](https://link.to.pricing).
   - Internal tools usage appears fine as long as the aim isn't commercialization, but guidelines remain somewhat unclear.
- **VS Code + WSL: A Dynamic Duo for Mojo**: One user explored running **Mojo** in a Windows dev environment using **Mojo Max** on WSL, recommending **VS Code** to bridge Windows and Linux seamlessly.
   - *You pretty much forget you're developing in Linux* when leveraging this setup, though some limitations exist in reproducibility.
- **FancyZones Boosts Workflow Management**: A member introduced the [FancyZones utility](https://learn.microsoft.com/en-us/windows/powertoys/fancyzones), enhancing window management on Windows by snapping applications into defined zones for better productivity.
   - This tool allows for efficient screen use, helping developers streamline their workflow in a multi-window setup.
- **Active Directory: Not Quite a Distributed Database**: A humorous debate unfolded over calling **Active Directory** a distributed database, with members noting it lacks characteristics like true consistency despite being labeled as such.
   - Further discussion emerged about existing distributed databases on Windows, showcasing an interest in clarifying terminology within the community.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Inspect Tool Teased for LLM Evaluation**: A member queried about [Inspect](https://github.com/UKGovernmentBEIS/inspect_ai) for LLM observability, looking for integration insights with DSPy.
   - While no experiences were shared, the tool seems positioned to enhance **large language model evaluations**.
- **DSPy Gains Advantage Over Langgraph**: A distinction emerged with DSPy optimizing prompt space instructions, while LangGraph acts at a lower level within **LangChain** architecture.
   - Essentially, DSPy is about performance boosts, whereas LangGraph handles system-level interfacing.
- **Optimize_signature Triumphs Over COPRO**: Users reported that **optimize_signature** outperformed **COPRO** in Chain of Thought tasks on GSM8K, achieving a score of **20/20**.
   - In contrast, COPRO struggled to secure a zero-shot instruction solution, maxing out at **18/20**.
- **User Seeks Help with DSPy-Multi-Document-Agent**: A member faced challenges locating the **requirements.txt** for **DSPy-Multi-Document-Agent**, questioning if they missed crucial files.
   - This inquiry pointed to potential documentation gaps or unclear resource links.
- **Interest in Advanced Retrieval with qdrant_dspy**: A link to the [qdrant_dspy GitHub repository](https://github.com/vardhanam/qdrant_dspy) highlights building RAG pipelines using **Gemma-2b**, **DSPy**, and **Qdrant**.
   - Another resource, [dspy/retrieve/qdrant_rm.py](https://github.com/stanfordnlp/dspy/blob/main/dspy/retrieve/qdrant_rm.py), emphasizes DSPy's utility in local VectorDB programming.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **ValueError Strikes getenv Function**: A user faced a `ValueError`, specifically **'invalid literal for int() with base 10: WARN'**, while importing, which pointed to an **environment variable** issue.
   - A member suggested that checking **environment variables** would help, confirming that the **DEBUG** variable set to **'WARN'** was the source of the problem.
- **DEBUG Variable Causes Trouble**: The **DEBUG** environment variable being set to **'WARN'** led to issues with the getenv function in a notebook environment, despite the user's Python script functioning well.
   - This highlights potential compatibility differences between notebook and standalone script environments in tinygrad.
- **Tinygrad Tensor Puzzles** Challenge Launched**: Members introduced **Tinygrad Tensor Puzzles**, a collection of **21 engaging puzzles** aimed at mastering tensor libraries like tinygrad from first principles, avoiding magic functions.
   - This initiative, building on **Sasha's PyTorch Tensor-Puzzles**, encourages contribution from both newcomers and experienced developers, fostering a community of problem solvers.
- **Tutorials to Explore Tinygrad Internals**: A set of [tutorials](https://mesozoic-egg.github.io/tinygrad-notes/) was shared, designed to enhance understanding of tinygrad's internals and promote contributions, along with a [quickstart guide](https://github.com/tinygrad/tinygrad/blob/master/docs/quickstart.md) for foundational insights.
   - While not entirely beginner-friendly, these resources provide essential knowledge for developers looking to engage with tinygrad effectively.
- **Optimizing Tinygrad with Computer Algebra Techniques**: Recent discussions included [computer algebra study notes](https://github.com/mesozoic-egg/computer-algebra-study-notes/tree/main) relevant to optimization processes in tinygrad, enhancing potential performance insights.
   - This integration showcases valuable methodologies that could support developers in refining tinygrad's capabilities.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Seeking Open Source Vision Models**: Members are actively looking for recommendations on **open source models** suited for **vision tasks**, inquiring about both local and API options for implementation.
   - One member showed curiosity by asking for insights on the availability and performance of such models within the community.
- **MiniCPM-V 2.6 Shines in Performance Tests**: **MiniCPM-V 2.6** has been reported to outperform its competitors, including **Gemini 1.5 Pro**, **GPT-4V**, and **Claude 3.5 Sonnet**, particularly in multi-image applications.
   - For more details, members shared links to its [Hugging Face page](https://huggingface.co/openbmb/MiniCPM-V-2_6) and the [GitHub repository](https://github.com/OpenBMB/MiniCPM-V).
- **Inquiry on Shipping Updates**: A member raised the question of **shipping updates**, indicating interest in the timeline and status.
   - Although no specific answers were provided, a link to a relevant [Discord channel](https://discord.com/channels/1146610656779440188/1194880263122075688/1266055462063964191) was shared for potential discussions.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Llama Team engages with queries on arXiv**: The **Llama team** is responding to questions on the [arXiv discussion forum](https://alphaxiv.org/abs/2407.21783v1), providing an opportunity for direct technical engagement.
   - This initiative allows for deeper insights into the **Llama 3** models and their applications.
- **Quora launches Poe Hackathon**: Quora is hosting an in-person and virtual [hackathon](https://x.com/poe_platform/status/1820843642782966103) focused on building bots with the new **Previews feature** for **Poe**.
   - Participants will develop innovative in-chat generative UI experiences utilizing advanced LLMs like **GPT-4o** and **Llama 3.1 405B**.
- **Exploring Non-Generative AI Applications**: A member sparked a conversation about the significance of **non-generative AI**, encouraging others to share their thoughts.
   - *What kinds of AI applications do you have in mind?* stirred interest in exploring various applications.
- **Diverse AI Applications Identified**: Suggestions flowed in for **computer vision**, **forecasting**, **recommendation systems**, and **NLP** as key non-generative AI areas.
   - These examples illustrate the **broad spectrum** of AI technologies that serve various niches beyond generative models.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Vercel's Outage Affects OpenRouter**: Vercel currently faces intermittent outages impacting the OpenRouter service, as detailed in their [status update](https://x.com/OpenRouterAI/status/1821267624228966781). After several updates, services were stable again by **3:45 PM ET**.
   - Vercel continues to monitor the issue and ensures updates will be posted on the [Vercel status page](https://www.vercel-status.com/).
- **Anthropic's High Error Rates Mitigated**: Anthropic has been addressing elevated error rates affecting the **3.5 Sonnet** and **3 Opus** models, implementing mitigation strategies that restored normal success rates as of **Aug 8, 17:29 PDT**.
   - They have provided updates ensuring access for **Claude.ai** free users is now restored while closely monitoring the situation.



---


The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Finetuning (Hamel + Dan) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1270821680217985176)** (80 messages🔥🔥): 

> - `4bit and GGUF Models`
> - `PPO Trainer Challenges`
> - `Multi-GPU Support in Unsloth`
> - `Continuous Batching with lmdeploy and vllm`
> - `Quantization of Mistral Models` 


- **Challenges with 4bit and GGUF Models**: Discussions around loading merged **4bit GGUF models** raised questions about potential precision loss when using `load_in_4bit` option in the model loading function.
   - Members noted that using **4bit** helps with VRAM consumption, but loading GGUF without this option often results in OOM errors.
- **Issues Encountered with PPO Trainer**: One member reported difficulties with the **PPO Trainer**, particularly receiving negative KL divergence errors while trying to implement a customized binary reward function.
   - Suggestions included exploring DPO as a potentially simpler alternative, although concerns were raised about its performance compared to PPO.
- **Unsloth Expands to Multi-GPU Support**: It was confirmed that **multi-GPU support** is being rolled out to trusted Unsloth supporters, offering benefits like VRAM reduction and increased speeds.
   - There was interest in whether this feature would be integrated into the open-source repository or limited to paid subscriptions.
- **Continuous Batching in lmdeploy and vllm**: Members inquired about **continuous batching** features in **lmdeploy** and how async requests would be handled during processing.
   - It was clarified that using the async engine in **vllm** and presumably in **lmdeploy** means there's no additional implementation required for batching requests.
- **Quantization of Mistral Models**: A user shared insights regarding their successful quantization of the **123B Mistral-Large-Instruct-2407** model, achieving significant size reduction with a small accuracy drop.
   - The use of the **EfficientQAT** algorithm allowed them to optimize model performance without significant degradation in outputs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/danielhanchen/status/1821284720606638158">Tweet from Daniel Han (@danielhanchen)</a>: Llama 3.1 chat template quirks:  1. Is &#34;Cutting Knowledge Date&#34; optional? Official repo tests don&#39;t add it. Docs add it & is it &#34;cutoff&#34;?  2. BOS? I worked with HF to add a default...</li><li><a href="https://huggingface.co/docs/trl/main/en/how_to_train#how-to-generate-text-for-training">Training FAQ</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1elbn3q/quantize_123b_mistrallargeinstruct2407_to_35_gb/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://docs.vllm.ai/en/latest/dev/engine/async_llm_engine.html">AsyncLLMEngine &#8212; vLLM</a>: no description found</li><li><a href="https://youtu.be/TKmfBnW0mQA?si=lz2sHuGY_IXBYbN_">Fixing bugs in Gemma, Llama, &amp; Phi 3: Daniel Han</a>: The story behind our 8 bug fixes for Gemma, multiple tokenization fixes for Llama 3, a sliding window bug fix and Mistral-fying Phi-3, and learn about how we...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1271027245506629695)** (1 messages): 

> - `Loss Function in AI Models`
> - `Understanding Token Labels` 


- **Clarifying the meaning of 'y' in token labels**: A member inquired about the meaning of `y`, questioning if it indicates a loss of **0** when the correct token is absent in a chunk, paralleling information from the graph and paragraph provided.
   - They expressed concerns about potential misinterpretations, seeking clarity on the label's representation in the context of chunks.
- **Final logsumexp reduction justifications**: The discussion raised a question regarding the necessity of performing a final **logsumexp reduction** across all chunks, especially if the correct chunk could suffice as the definitive output.
   - The member suggested that since the cross-entropy loss is **0** for other chunks, utilizing only the correct chunk might lead to a more streamlined process.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1270830145527222428)** (135 messages🔥🔥): 

> - `Model Loading Issues`
> - `Dataset Processing`
> - `Hugging Face Integration`
> - `Inference Optimization`
> - `Colab Limitations` 


- **Error when typing to the model**: A user successfully loaded a model but encountered an 'error' message when attempting to chat with it.
   - Another member noted that this issue only occurs with models other than **Llama 3.1 8B Instruct**, as support is currently limited.
- **Processing Smaller Datasets**: A user faced issues with downloading entire datasets instead of a smaller subset from Hugging Face.
   - A suggestion was made to fork the code and modify it to limit processing to **200 files** after reviewing repository lines.
- **Hugging Face Model Integration**: A user inquired about uploading their model to Hugging Face and correctly referencing it in their chat script.
   - Support was provided on how to push just the model weights to Hugging Face for use in a new Colab environment.
- **Optimizing Inference on A100 GPUs**: A user sought advice on state-of-the-art technologies for fast inference of LLMs on an A100 80 GB server.
   - They shared **vLLM** parameters being used and requested recommendations for improvement or alternative approaches.
- **Colab Usage and Limitations**: Concerns were raised about Colab's disk space limitations affecting model loading and training efforts.
   - Options for upgrading to Colab Pro were discussed, along with alternative methods for less resource-intensive testing.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1vbH6h760iesRfcVQlm4-KVv1zYM5sB9k?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/datasets/ncbi/pubmed/blob/main/pubmed.py#L40">pubmed.py · ncbi/pubmed at main</a>: no description found</li><li><a href="https://github.com/unslothai/studio">GitHub - unslothai/studio: Unsloth Studio</a>: Unsloth Studio. Contribute to unslothai/studio development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1270858117814685808)** (293 messages🔥🔥): 

> - `Harambe tool for bug hunting`
> - `LLMs for URL analysis`
> - `Open source collaboration`
> - `Development challenges`
> - `Productivity and sleep patterns` 


- **Introducing Harambe: The Bug Hunting Assistant**: A member introduced **Harambe**, an open-source tool designed to assist bug hunters by providing a viewer for HTTP requests and responses, enriched with analytics and tools for better analysis.
   - The goal is to streamline the bug hunting process by using LLMs to generate plausible API endpoint suggestions, reducing the reliance on conventional fuzzing techniques.
- **Need for LLM Fine-tuning Support**: The project requires fine-tuning multiple LLM models (1B, 2B, 3B, and 7B parameters) to ensure effective URL analysis and wordlist creation tailored to various hardware capabilities.
   - The developer seeks community support, emphasizing that those helping will not receive monetary rewards but will be acknowledged in project documentation.
- **Development and Productivity Insights**: Members discussed their personal productivity, noting different approaches to sleep and work schedules, highlighting preferences for late-night work sessions.
   - One member reflected on programming habits and how engaging in projects early in life influences ongoing interests and career paths.
- **Learning New Programming Languages**: A discussion emerged regarding the willingness to learn new programming languages as one grows older, with members sharing their experiences and preferences for high-level versus low-level languages.
   - While low-level programming offers unique challenges and satisfaction, many find high-level languages more suitable for rapid prototyping and project development.
- **Speculation on AGI Development**: Members speculated about the future of AGI, emphasizing the complexity and unpredictability of developing such technology, with references to the mixed results of various development methodologies.
   - They recognized that while pursuing advanced artificial intelligence is ambitious, current tools and societal structures primarily focus on specialized skills and roles.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://noclip.website/">noclip</a>: no description found</li><li><a href="https://www.reddit.com/r/howdidtheycodeit/comments/183a5en/how_do_i_recreate_this/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1270831460345057389)** (9 messages🔥): 

> - `Prompt Classification for Evaluation`
> - `FlexAttention in PyTorch`
> - `Attention Implementation Challenges`
> - `Non-Contaminated Packing`
> - `Hugging Face Integration` 


- **Prompt classification improves evaluation methods**: A study revealed that **prompts** can classify conversations effectively, helping create evaluation functions for context-specific tones such as friendly or romantic.
   - This approach allows the model to provide nuanced dataset evaluations based on the relationships identified between speakers.
- **Introducing FlexAttention for diverse attention variants**: The new [FlexAttention](https://pytorch.org/blog/flexattention/) API allows for easy implementation of various attention mechanisms in *PyTorch* without extensive kernel rewrites.
   - This functionality resolves the ‘software lottery’ issue faced by ML researchers, enabling greater experimentation with fused kernels.
- **Challenges with optimized attention implementations**: Despite efficiency gains from **FlashAttention**, ML researchers now face difficulties implementing new attention variants due to the need for custom kernels.
   - The result is slower runtimes and CUDA OOM issues when exploring variations beyond existing optimized kernels.
- **Potential for integrating non-contaminated packing**: The feature discussed on [document masking jagged sequences](https://pytorch.org/blog/flexattention/#document-maskingjagged-sequences) may facilitate adding non-contaminated packing to the Unsloth project.
   - This could improve the workflow and efficiency of implementations utilizing attention mechanisms.
- **Proposals for direct Hugging Face integration**: One member proposed that **Hugging Face** should directly integrate the FlexAttention features to enhance usability.
   - This suggestion followed discussions on the improvements FlexAttention could bring to various projects in machine learning.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/blog/flexattention/#document-maskingjagged-sequences">FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention</a>:   </li><li><a href="https://pytorch.org/blog/flexattention/">FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention</a>:   </li><li><a href="https://x.com/cHHillee/status/1821253769147118004">Tweet from Horace He (@cHHillee)</a>: For too long, users have lived under the software lottery tyranny of fused attention implementations.   No longer.   Introducing FlexAttention, a new PyTorch API allowing for many attention variants t...
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1271232340126863410)** (1 messages): 

> - `BiRefNet Background Removal`
> - `ActionGemma Model for Function Calling`
> - `Unity ML-Agents`
> - `Segment Anything Model Insights`
> - `ArabicWeb24 Dataset` 


- **BiRefNet Outperforms RMBG1.4**: Thanks to the team at [Nankai University](https://huggingface.co/ZhengPeng7/BiRefNet), **BiRefNet** showcases better performance for background removal compared to **RMBG1.4**.
   - The model leverages bilateral reference for high-resolution dichotomous image segmentation, with further details available in the [arXiv paper](https://arxiv.org/pdf/2401.03407).
- **Introducing ActionGemma Model**: [ActionGemma](https://huggingface.co/KishoreK/ActionGemma-9B) represents a fine-tuned 9B model tailored for function calling, merging Gemma's multilingual capabilities with function calls from the xLAM dataset.
   - Developed by **KishoreK**, this model combines robust performance across multiple languages.
- **Exploring Unity ML-Agents**: A new [YouTube video](https://youtube.com/live/J-de9K_3xDw?feature=share) details how to pretrain a large language model from scratch using Unity ML-Agents.
   - This engaging tutorial guides viewers through building intelligent chatbots with cutting-edge technologies.
- **Insights from Segment Anything Model**: Recent blogposts discuss the capabilities of vision models like **CLIP** and **ALIGN** in the context of segmentation tasks, which have not advanced at the same speed as text models.
   - Key discussions include the challenges faced in progressing core computer vision tasks and exploring engineered prompts for improved results.
- **High Quality Arabic Dataset: ArabicWeb24**: A new blog post presents [ArabicWeb24](https://huggingface.co/blog/MayFarhat/arabicweb24), a dataset tailored for high-quality pre-training in Arabic.
   - This resource is essential for developing applications that require extensive Arabic language understanding.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/ZhengPeng7/BiRefNet">ZhengPeng7/BiRefNet · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/KishoreK/ActionGemma-9B">KishoreK/ActionGemma-9B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/DamarJati/FLUX.1-DEV-Canny">FLUX.1-DEV Canny - a Hugging Face Space by DamarJati</a>: no description found</li><li><a href="https://youtube.com/live/J-de9K_3xDw?feature=share">Unity ML-Agents | Pretrain an LLM from Scratch with Sentence Transformers| Part 1</a>: Welcome to our exciting journey of creating an intelligent chatbot using Unity ML-Agents and Sentence Transformers! 🚀In this video, we walk you through the ...</li><li><a href="https://www.lightly.ai/post/segment-anything-model-and-friends">Segment Anything Model and Friends</a>: The Segment Anything Model (SAM) and its successors made a significant leap forward in computer vision, particularly in image and video segmentation. Along with SAM’s innovative approach to promptable...</li><li><a href="https://huggingface.co/spaces/Delik/Anitalker">Anitalker - a Hugging Face Space by Delik</a>: no description found</li><li><a href="https://huggingface.co/spaces/LEGENT/LEGENT">LEGENT - a Hugging Face Space by LEGENT</a>: no description found</li><li><a href="https://www.lightly.ai/post/using-self-supervised-learning-for-dense-prediction-tasks">Using Self-Supervised Learning for Dense Prediction Tasks</a>: Overview of Self-Supervised Learning methods for dense prediction tasks such as object detection, instance segmentation, and semantic segmentation</li><li><a href="https://dev.to/tonic/dockers-testcontainers-are-great-42cl">no title found</a>: no description found</li><li><a href="https://huggingface.co/blog/prithivMLmods/lora-adp-01">Unlocking Creativity with Text-to-Image Generation: Exploring LoRA Models and Styles</a>: no description found</li><li><a href="https://youtu.be/fnKrReaqQgc">How does Uber predict Arrival Times (ETA) for trips? | Uber ML System Design | #systemdesign</a>: Do you know how cab companies like Uber, Ola, and Lyft predict the Expected Time of Arrival (ETA) for the trips? In this video, we design an end-to-end machi...</li><li><a href="https://huggingface.co/blog/MayFarhat/arabicweb24">ArabicWeb24: Creating a High Quality Arabic Web-only Pre-training Dataset </a>: no description found</li><li><a href="https://github.com/Rivridis/LLM-Assistant">GitHub - Rivridis/LLM-Assistant: Locally running LLM with internet access</a>: Locally running LLM with internet access. Contribute to Rivridis/LLM-Assistant development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1270819545782685718)** (224 messages🔥🔥): 

> - `Web Search Functions in LLM Apps`
> - `Animated Clone Avatars`
> - `Performance of AI Models`
> - `Discord vs. Forums for Communication`
> - `Minecraft Server Experiences` 


- **Web Search Alternatives to Google and Bing**: Members discussed alternatives to Google and Bing for web search functions in LLM applications, highlighting [DuckDuckGo's API](https://duckduckgo.com) as free and widely used for building custom search models.
   - Another member mentioned the [Brave Search API](https://brave.com/search/api/) as another viable alternative for powering search functionalities.
- **Creating Animated Clone Avatars with AI**: A user inquired about AI models capable of generating animated clone avatars using video data and lip-syncing them to text input, specifically mentioning [Rask.ai](https://rask.ai).
   - Wav2Lip was suggested as an option, but others recommended exploring [SeamlessM4T-v2](https://huggingface.co/docs/transformers/model_doc/seamless_m4t_v2) for its extensive functionality in translation.
- **Performance Evaluation of AI Models**: Members shared their experiences with pre-trained translation models like [Facebook's M2M100](https://huggingface.co/facebook/m2m100_418M) and [SeamlessM4T](https://huggingface.co/docs/transformers/model_doc/seamless_m4t_v2) as potentially suitable options for multi-language translations.
   - Discussion included comparisons regarding transcription capabilities between SeamlessM4T-v2 and Whisper models.
- **Discord Chat vs. Forum Communication**: Users expressed mixed feelings about using Discord threads for serious discussions, noting the difficulty in following conversations compared to traditional forums.
   - There was a consensus that having a structured method, like auto-threading or linking to messages, could make the real-time chat more manageable and informative.
- **Nostalgic Experiences on Minecraft Servers**: A member shared their nostalgic experiences running a Minecraft server with friends, recalling the enjoyment of revisiting old builds left untouched after map wipes.
   - They noted the sentiment attached to the server as a personal space despite changes in friendships and dynamics over time.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ladybird.org/">Ladybird</a>: Ladybird is a truly independent web browser, backed by a non-profit.</li><li><a href="https://brave.com/search/api/">Brave Search API | Brave</a>: Power your search and AI apps with the fastest growing independent search engine since Bing. Access an index of billions of pages with a single call.</li><li><a href="https://huggingface.co/facebook/m2m100_418M">facebook/m2m100_418M · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=qbfwadEo_t4">Why We Named Our Company &quot;Hugging Face&quot; 🤗</a>: Watch the full interview with Clem Delangue here:https://youtu.be/BPExesApMXU#Clem Delangue #HarryStebbings #20VC #shorts #HuggingFace #ai #🤗 #generativeai ...</li><li><a href="https://youtu.be/t-NIB6L_3zk">Dev Readers Notebook 2 : 20 Concepts of Django in 2 mins</a>: In this Dev Notebook Series video, I&#39;ll cover 20 basic concepts of Django to give you a comprehensive overview of what it is and how it works. If you haven&#39;t...</li><li><a href="https://python.langchain.com/v0.2/docs/integrations/tools/">Tools | 🦜️🔗 LangChain</a>: If you&#x27;d like to write your own tool, see this how-to.</li><li><a href="https://huggingface.co/bakrianoo/sinai-voice-ar-stt">bakrianoo/sinai-voice-ar-stt · Hugging Face</a>: no description found</li><li><a href="https://youtu.be/CoqfMasnk0A?t=1241">Community call - August 2024</a>: This is a monthly community call for the Firefox product community to discuss the upcoming product releases, community updates, and various contribution oppo...</li><li><a href="https://huggingface.co/docs/transformers/model_doc/seamless_m4t_v2">SeamlessM4T-v2</a>: no description found</li><li><a href="https://github.com/huggingface/datasets/issues/7092)">Issues · huggingface/datasets</a>: 🤗 The largest hub of ready-to-use datasets for ML models with fast, easy-to-use and efficient data manipulation tools - Issues · huggingface/datasets</li><li><a href="https://github.com/hiyouga/LLaMA-Factory/wiki/Performance-Comparison)">Home</a>: A WebUI for Efficient Fine-Tuning of 100+ LLMs (ACL 2024) - hiyouga/LLaMA-Factory</li><li><a href="https://www.shadertoy.com/view/XfByW3,">Error - Shadertoy BETA</a>: no description found</li><li><a href="https://www.shadertoy.com/view/XfBcWc">Shadertoy</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1270839917597036659)** (4 messages): 

> - `Neural Network Optimization`
> - `AI in Healthcare`
> - `Embedding Serialization and Deserialization` 


- **Neural Network Optimization with Layer-Wise Scaling**: After implementing **layer-wise scaling** to minimize outlier features, the model's **1b loss** now gets stuck at **3.8**, improved from **5.0**.
   - The **50m loss** doesn't diverge anymore but converges slowly, and there's an observation that **attention entropy** collapses simultaneously, indicating slow convergence.
- **Exploring Future of AI in Healthcare**: Check out the [YouTube video](https://www.youtube.com/watch?v=Z--q7RO2TrU) featuring Professor Andrew Janowczyk from Emory University discussing **AI in Healthcare**.
   - He is associated with the **Emory Precision AI for Health Institute** and provides insights from his extensive experience spanning nearly **15 years**.
- **Learning to Serialize and Deserialize Embeddings**: A member is focused on learning the techniques to **serialize and deserialize embeddings data** between Python and C#.
   - This process is crucial for effective data handling in AI applications.



**Link mentioned**: <a href="https://www.youtube.com/watch?v=Z--q7RO2TrU">The Future of AI in Healthcare, ft. Professor Andrew Janowczyk (Emory University)</a>: Dr. Andrew Janowczyk is an Asst. Prof. at Emory Precision AI for Health Institute and a data analyst at Geneva University Hospitals. With nearly 15 years of ...

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1271023723188584450)** (3 messages): 

> - `Transformers Architecture`
> - `EU AI Regulations`
> - `AI Risk-based Regulation` 


- **Dominance of Transformers Explained**: An informative article details why **Transformers** are the dominant architecture in the current era, highlighting their performance benefits.
   - It provides intuition on attention mechanisms and architectural advantages, supporting their widespread adoption.
- **EU's New AI Regulations Roll Out**: The **European Union's** **AI Act** has officially come into force as of August 1, 2024, starting a timeline for compliance for various AI applications.
   - This includes a series of staggered compliance deadlines, the first of which prohibits certain uses of AI, such as **remote biometrics** in law enforcement, starting in six months.
- **Implications of Staggered Compliance Deadlines**: The **risk-based regulation** framework will fully apply by mid-2026, affecting different types of AI developers and applications.
   - Most provisions concerning AI applications will require compliance, indicating a significant shift in how AI can be utilized across sectors.



**Link mentioned**: <a href="https://techcrunch.com/2024/08/01/the-eus-ai-act-is-now-in-force/">The EU&#039;s AI Act is now in force | TechCrunch</a>: The European Union&#039;s risk-based regulation for applications of artificial intelligence has come into force starting from today.

  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1270860414204252161)** (16 messages🔥): 

> - `EurekAI platform`
> - `Gemma2 9B fine-tuning`
> - `Flux.1 Dev Controlnet Canny`
> - `Text-to-Image Diffusion Models`
> - `TTS optimizations` 


- **EurekAI: The GitHub for Researchers**: A member announced the launch of [EurekAI](https://eurekai-web-app.vercel.app/signup), a platform facilitating cross-collaboration among researchers with AI features to streamline the research process.
   - Currently in alpha, the platform promises features like project creation and integrated journaling to enhance productivity and engagement in research.
- **Fine-Tuning Gemma2 for Function Calling**: A user is fine-tuning the [Gemma2 9B model](https://huggingface.co/KishoreK/ActionGemma-9B) using the xLAM dataset, aiming to create a versatile function calling model for multiple languages.
   - Questions arose about the model's public accessibility, with a note that an emoji in the link may have caused issues.
- **New Hugging Face Space for Flux.1**: A member created a Hugging Face Space for [Flux.1 Dev Controlnet Canny](https://huggingface.co/spaces/DamarJati/FLUX) leveraging code from a GitHub repository.
   - The project aims to showcase capabilities in an easily accessible manner as part of the growing collection of Hugging Face Spaces.
- **Efficient Training of Diffusion Models by Apple**: A researcher from Apple introduced a new Python package for efficiently training text-to-image diffusion models, linked to their [ICLR 2024 paper](https://github.com/apple/ml-mdm).
   - This package is designed for optimal performance with limited data, offering a novel approach within the field of machine learning.
- **TTS Optimizations Blog Post**: A member shared their [Medium blog](https://medium.com/@mllopart.bsc/optimizing-a-multi-speaker-tts-model-for-faster-cpu-inference-part-1-165908627829) detailing TTS optimizations, aimed at improving CPU inference performance.
   - Discussion ensued regarding possible accessibility issues and formatting concerning the provided links.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/apple/ml-mdm">GitHub - apple/ml-mdm: Train high-quality text-to-image diffusion models in a data &amp; compute efficient manner</a>: Train high-quality text-to-image diffusion models in a data &amp; compute efficient manner - apple/ml-mdm</li><li><a href="https://huggingface.co/KishoreK/ActionGemma-9B">KishoreK/ActionGemma-9B · Hugging Face</a>: no description found</li><li><a href="https://youtube.com/live/J-de9K_3xDw?feature=share">Unity ML-Agents | Pretrain an LLM from Scratch with Sentence Transformers| Part 1</a>: Welcome to our exciting journey of creating an intelligent chatbot using Unity ML-Agents and Sentence Transformers! 🚀In this video, we walk you through the ...</li><li><a href="https://huggingface.co/spaces/DamarJati/FLUX.1-DEV-Canny">FLUX.1-DEV Canny - a Hugging Face Space by DamarJati</a>: no description found</li><li><a href="https://github.com/XLabs-AI/x-flux">GitHub - XLabs-AI/x-flux</a>: Contribute to XLabs-AI/x-flux development by creating an account on GitHub.</li><li><a href="https://eurekai-web-app.vercel.app/signup).">EurekAI App</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1271056966084464640)** (4 messages): 

> - `Transformers Architecture`
> - `New Ideas in AI Experimentation` 


- **Transformers Simplified for Everyone**: A member shared a [Medium post](https://medium.com/@jatinprrrrt/transformers-made-simpler-a-overview-d71585f45fe6) explaining the **transformers architecture** in simpler terms, seeking feedback to validate their understanding.
   - They emphasize the importance of learning through sharing, hoping to engage others in providing feedback.
- **Exciting New Experiment With Promising Results**: Another member expressed excitement about a **very promising idea** that had not been tried before, noting that their first experiment worked very well.
   - While specific details were not shared, the enthusiasm suggests potential significant advancements in the field.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1271017097106948208)** (9 messages🔥): 

> - `Papers with Code for Computer Vision`
> - `Converting Handwriting to Stroke Format`
> - `IAM On-Line Handwriting Database` 


- **Papers with Code provides rich resources**: A member recommended [Papers with Code](https://paperswithcode.com/sota) as a valuable resource for summarizing the state of the art in computer vision, featuring **11,272 benchmarks**, **5,031 tasks**, and **137,097 papers with code**.
   - This platform compiles data assisting users in exploring various aspects of machine learning applications.
- **Converting handwriting to a stroke format**: A member inquired about methods for converting images of handwriting into a stroke format composed of individual strokes with color and timing details.
   - They emphasized the need to differentiate this format from SVG, aiming for a single stroke representation.
- **Details on IAM On-Line Handwriting Database**: A member shared a link to the IAM On-Line Handwriting Database, providing information on its data format, highlighting stored directories and XML files containing essential details.
   - The XML files include elements such as unique form IDs, writer IDs, and comprehensive time capture data, as outlined in the provided [data format document](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database/data-format).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database/data-format">Research Group on Computer Vision and Artificial Intelligence &mdash; Computer Vision and Artificial Intelligence</a>: no description found</li><li><a href="https://paperswithcode.com/sota">Papers with Code - Browse the State-of-the-Art in Machine Learning</a>: 11272 leaderboards • 5031 tasks • 10360 datasets • 137097 papers with code.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1271034157610303590)** (2 messages): 

> - `AutoProcessor Availability`
> - `InternLM 2.5 Features` 


- **Query on AutoProcessor for Models**: A member inquired about how to check which models have an **AutoProcessor** associated with them, mentioning issues with **InternLM**.
   - They attempted to use [InternLM's repository](https://huggingface.co/internlm/internlm2_5-7b-chat), but ran into an error indicating the absence of a recognized processing class when calling the `AutoProcessor`.
- **InternLM 2.5 Model Highlights**: **InternLM 2.5** boasts outstanding reasoning capabilities, outperforming other models like **Llama3** and **Gemma2-9B** in mathematical reasoning tasks.
   - It features a **1M context window**, enabling exceptional performance on long-context tasks, and supports information gathering from over **100 web pages**.



**Link mentioned**: <a href="https://huggingface.co/internlm/internlm2_5-7b-chat">internlm/internlm2_5-7b-chat · Hugging Face</a>: no description found

  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1270942893452038154)** (29 messages🔥): 

> - `Flux Transformer Training`
> - `Using Multiple GPUs`
> - `LoRA Training`
> - `CUDA Resource Management` 


- **Challenges with Flux Transformer on RTX GPUs**: Members discussed the difficulties in loading the large **FluxTransformer2DModel** on their RTX 4090, especially when attempting to utilize multiple GPUs.
   - _CUDA should handle VRAM allocation automatically_, but users still face issues with model size exceeding available memory.
- **Advice on Training with Limited VRAM**: Suggestions were made to train **LoRAs** instead of full models since training the latter requires significant GPU resources and multi-GPU setups aren't efficiently implemented.
   - One user noted it isn't simple to train LoRAs and expressed concern about how much additional space would be required.
- **Confusion over Device Mapping for Model Loading**: There was confusion regarding the use of `device_map` for splitting models across GPUs, with consensus that the current implementations only distribute different models, not the segments of a single model.
   - Despite efforts to use `device_map='auto'`, users encountered errors indicating this feature is not yet supported.
- **Discussion on Resource Efficiency**: Users discussed the inefficiency of splitting a model across GPUs as opposed to pooling resources for optimal integration, with suggestions to ensure **CUDA and Flash attention** are properly set up.
   - A member pointed out that with **12 billion parameters**, even fp16 models might cause slow performance due to potential VRAM sharing.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/RisingSayak/status/1821510483729318197">Tweet from Sayak Paul (@RisingSayak)</a>: If you have multiple GPUs and would like them to share during running inference with FLUX, it&#39;s possible in diffusers.   Here, I am mimicking three GPUs, each having 16G, 16G, and 24G, respectivel...</li><li><a href="https://huggingface.co/spaces/tori29umai/sketch2lineart">Sketch2lineart - a Hugging Face Space by tori29umai</a>: no description found</li><li><a href="https://huggingface.co/docs/diffusers/main/en/tutorials/inference_with_big_models#device-placement.">Working with big models</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1271174813825765417)** (1 messages): 

> - `Gradio v4.41 Release`
> - `New Features in Gradio`
> - `Security Improvements`
> - `Bug Fixes`
> - `Documentation Enhancements` 


- **Gradio v4.41 Released with Exciting Features**: The new version of Gradio, **v4.41**, introduces several enhancements including **full screen images** for components like `gr.Image` and improved output viewing options.
   - This update aims to streamline user experience with new buttons for easier viewing and copying outputs.
- **Plotting Improvements in Gradio**: The updated version features **documentation improvements** for plots and the ability to set a `.double_click()` event listener for better interaction.
   - This enhancement provides developers with more control over how users interact with visual data.
- **Enhanced Security Measures Implemented**: **Security fixes** in v4.41 significantly tighten the CI and prevent unauthorized file access as well as XSS attacks, enhancing overall platform safety.
   - The changes also include improved security mechanisms around setting `share=True`, ensuring safer user interactions.
- **Numerous Bug Fixes Delivered**: The release addresses various bugs across components including **gr.Model3D**, **gr.Chatbot**, and **gr.Interface**, ensuring smoother operation.
   - For a complete list of fixes, users can refer to the [changelog](https://www.gradio.app/changelog), which details all changes made.


  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1270833978412240956)** (14 messages🔥): 

> - `Profiling CUDA`
> - `BPF wizardry`
> - `Nsight tools`
> - `eBPF for GPU monitoring` 


- **Inquiry about BPF for CUDA Profiling**: A member inquired if anyone was using **BPF** to profile **CUDA**.
   - Some clarified that **eBPF** is part of the OS kernel and lacks visibility on GPU activities.
- **Nsight Tools Efficient for CUDA Profiling**: Another member shared their positive experience, recommending **Nsight Compute** for understanding single kernel performance and **Nsight Systems** for monitoring entire CPU/GPU applications.
   - *It's good, gets the job done,* according to one user referring to **Nsight**.
- **Potential of eBPF for GPU Monitoring**: **eBPF profiling** was suggested as useful for monitoring GPU fleets in production environments.
   - However, doubts were raised about the effectiveness of **Nsight Systems** for this specific use.
- **Curiosity About Nsight Preferences**: A member asked for opinions on **Nsight**, prompting another to affirm its utility in CUDA performance evaluation.
   - Specific links to resources were requested but not provided.
- **The Wizardry of eBPF**: The discussion included mentions of the 'wizardry' required to effectively implement eBPF for CUDA profiling.
   - The dialogue reflects a growing curiosity about advanced profiling techniques in GPU contexts.


  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1270874788608348170)** (31 messages🔥): 

> - `Attention Gym issues`
> - `Integration of FlexAttention`
> - `Torch serialization challenges`
> - `Flash Attention and Paged Attention connection` 


- **Attention Gym link malfunction**: Members noted that the link to **Attention Gym** under the softcapping section is currently not functioning.
   - *Just finished reading it!* reiterated how comprehensive and detailed the article's explanation is.
- **Plans for FlexAttention integration**: A member inquired about the plans to help integrate **FlexAttention** into models at HF, citing prior difficulties with new PyTorch features.
   - They also discussed how **HF** is likely to wait for the release of version **2.5** before they can proceed.
- **Torch serialization complexities**: Members expressed concerns over **Torch serialization**, with one noting this issue haunts them and it's mentioned to be fixed in the nightlies.
   - Additionally, a workaround was provided, suggesting using `model.compile()` instead of `torch.compile()` to avoid state dict complications.
- **Query on device type for autocast**: A user asked if the default device type for **torch.autocast** is CUDA when none is specified.
   - This inquiry was met with a response of understanding and interest, though further clarification wasn't provided.
- **Flash Attention compatibility**: A member queried why **Flash Attention** cannot be utilized with **Paged Attention**.
   - The response pointed to the fact that initial implementations of **FlashAttention** did not support paged attention, requiring later kernel modifications.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://dev-discuss.pytorch.org/t/how-to-bring-compile-time-down-to-zero-our-plans-and-direction-may-14th-edition/2089">How To Bring Compile Time Down to Zero: Our Plans and Direction (May 14th Edition)</a>: We are excited to announce that over the course of the first half of 2024 we have been prioritizing improving compile times for torch.compile workflows. Swift iterations and efficient development cycl...</li><li><a href="https://discuss.pytorch.org/t/how-to-serialize-models-with-torch-compile-properly/175626/2">How to serialize models with torch.compile properly</a>: There is no serialization solution yet for torch.compile but it’s high priority  Regarding the 2 forums this is more of a community forum for general pytorch users wheras the dev forum is more for pyt...</li><li><a href="https://github.com/pytorch-labs/attention-gym/blob/bf80fecf39edee616be620ed6204aec786403b9a/attn_gym/masks/causal.py#L5">attention-gym/attn_gym/masks/causal.py at bf80fecf39edee616be620ed6204aec786403b9a · pytorch-labs/attention-gym</a>: Helpful tools and examples for working with flex-attention - pytorch-labs/attention-gym</li><li><a href="https://github.com/pytorch/pytorch/pull/120143">Add sliding window attention bias  by drisspg · Pull Request #120143 · pytorch/pytorch</a>: Summary This PR adds a new attnetion-bias torch_function designed to interact with SDPA. This implements sliding window and updates &amp;quot;aten.sdpa_flash&amp;quot;  to expose the window_size_left ...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

iron_bound: the code released finally https://github.com/Aleph-Alpha/trigrams
  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1271149203116261426)** (1 messages): 

> - `2D Conv Kernels`
> - `Constant Memory Usage`
> - `Dynamic Kernel Sizes` 


- **Navigating Kernel Size in 2D Conv**: A member questioned how frameworks provide flexibility in selecting **kernel sizes** while using `__constant__` memory, which requires prior knowledge of the kernel size.
   - They wondered if common sizes are pre-compiled and if dynamic kernel sizes are employed for less common variations.
- **Understanding Memory Management in Convolution**: The discussion highlighted the importance of memory management when dealing with **2D convolutional kernels** and constant memory usage.
   - Participants shared insights on optimizing **filter allocations** and reducing overhead through effective management strategies.


  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1270870741104988171)** (6 messages): 

> - `Release of torchao v0.4.0`
> - `Intx Tensor Subclasses Quantization`
> - `Update on issue #577`
> - `ModelRunner.run() complexity` 


- **torchao v0.4.0 is now live!**: The **0.4 release** of torchao has been announced, adding features like **KV cache quantization** and **quantization aware training (QAT)** which can enhance performance.
   - It was highlighted that this version includes support for **low bit optimizer** functionality, making it an exciting update for users.
- **Discussion on Intx Tensor Subclasses Quantization**: An ongoing **GitHub issue (#439)** aims to implement sub byte unsigned integer quantization baselines in PyTorch to allow low bit quantization experimentation.
   - A request for additional items to be added to the tracker was made, opening the floor for suggestions from the community.
- **Clarification sought on issue #577 progress**: An update was provided on **GitHub issue #577**, with insight into the complexities of calling **call_function** within **ModelRunner.run()** with new **MultiTensor**.
   - The author expressed a desire for a review of their work on **annotated_gptq**, seeking guidance to ensure they are heading in the right direction.
- **Challenges with model execution for activations**: A detailed explanation noted that running the model sequentially for each linear layer can result in slow execution times, raising concerns about performance.
   - The proposed approach suggests pausing after each linear to update weights before continuing, potentially increasing efficiency.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/ao/issues/577:">Issues · pytorch/ao</a>: The missing pytorch dtype and layout library for training and inference - Issues · pytorch/ao</li><li><a href="https://github.com/pytorch/ao/releases/tag/v0.4.0">Release v0.4.0 · pytorch/ao</a>: v0.4.0 Highlights We are excited to announce the 0.4 release of torchao! This release adds support for KV cache quantization, quantization aware training (QAT), low bit optimizer support, composing...</li><li><a href="https://github.com/pytorch/ao/issues/439">[RFC] Intx Tensor Subclasses Quantization · Issue #439 · pytorch/ao</a>: Objective: Implement sub byte unsigned integer quantization baselines from 1-7 to enable users to experiment with low bit quantization in pytorch. Tracker: Create a UIntx Tensor Subclass per #391 I...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1270819014159110207)** (179 messages🔥🔥): 

> - `KV Cache Implementation`
> - `RoPE Optimization`
> - `Training Efficiency`
> - `Fine-tuning Strategies`
> - `Code Cleanup and Refactoring` 


- **KV Cache Implementation to Improve Memory Usage**: A member implemented KV Cache to optimize memory usage, allowing full bfloat16 fine-tuning with batch size 1 and sequence length 4096 on a single 80GB GPU.
   - Despite this enhancement, it pushes the memory limits, prompting discussions on memory optimization strategies like using managed memory.
- **RoPE Implementation Analysis**: There was a discussion about simplifying the RoPE implementation to avoid complex numbers, favoring direct trigonometric operations for readability and maintainability.
   - Members agreed that while complex abstractions could be commented on, the straightforward use of sin/cos might be more approachable.
- **Training Logic and Refactoring Opportunities**: Concerns were raised about certain training logic, particularly a condition in the generate function that could be simplified or removed for clarity.
   - Refactoring opportunities were identified to enhance the manageability of the code, especially concerning the training loop.
- **Fine-tuning Strategies and Memory Management**: Discussion about fine-tuning revealed that using strategies like qlora has gained popularity due to memory constraints during model training.
   - A pull request was shared that aims to allocate managed memory to alleviate issues when device memory runs out, facilitating slower training on smaller systems.
- **Collaboration on Code Improvements**: Members planned to make smaller PRs focusing on code cleanup and refactoring once initial drafts have been merged.
   - This approach aims to enhance collaboration and maintain code quality as the project evolves.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/jrahn/gpt2_350M_edu_hermes">jrahn/gpt2_350M_edu_hermes · Hugging Face</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/pull/709">Allocate managed memory if device memory runs out by ngc92 · Pull Request #709 · karpathy/llm.c</a>: Use cudaMallocManaged to allocate optimizer states if we run out of device memory, so we can still train (slowly) even if we cannot fit the optimizer state This is based on #694 , which should be m...</li><li><a href="https://github.com/pytorch/torchchat/blob/fe73ef737e84794694bd7c48a4b6bd0fd9028cb2/build/model.py">torchchat/build/model.py at fe73ef737e84794694bd7c48a4b6bd0fd9028cb2 · pytorch/torchchat</a>: Run PyTorch LLMs locally on servers, desktop and mobile - pytorch/torchchat</li><li><a href="https://github.com/karpathy/llm.c/pull/725/files">Add LLaMA 3 Python support by gordicaleksa · Pull Request #725 · karpathy/llm.c</a>: Add LLaMA 3 support in our Python code acting as a reference. The code supports only inference right now and is equivalent with nano llama 3.</li><li><a href="https://github.com/karpathy/llm.c/pull/730">Demo equivalence - tmp by gordicaleksa · Pull Request #730 · karpathy/llm.c</a>: cc: @karpathy here is the minimal change on top of my PR that gives equivalent code to nano llama 3 reference.py from commit karpathy/nano-llama31@d0dfb06 Steps:  Check out this PR Check out the co...</li><li><a href="https://github.com/karpathy/llm.c/pull/730/commits/298a49ac61a219f0be4a681ad4c3175ec0a95f2f">Demo equivalence - tmp by gordicaleksa · Pull Request #730 · karpathy/llm.c</a>: cc: @karpathy here is the minimal change on top of my PR that gives equivalent code to nano llama 3 reference.py from commit karpathy/nano-llama31@d0dfb06 Steps:  Check out this PR Check out the co...
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1270834387050565724)** (179 messages🔥🔥): 

> - `Perplexity Pro Limits`
> - `API Usability`
> - `Changes in Model Availability`
> - `User Experience with Alternatives`
> - `Service Stability Issues` 


- **Perplexity Pro Limits Reduced**: Users reported that the daily limit for the Perplexity Pro plan has been changed from **600 to 450** in a short time, causing frustration among subscribers.
   - *One user expressed disappointment* over not receiving notifications for these changes, highlighting a lack of trust in the platform's communication.
- **Concerns Over API Usability**: Several members discussed the potential costs of using Perplexity's API, indicating it could be more cost-effective for non-heavy users on a pay-as-you-go basis.
   - Some mention that the API offers grounded web search replies, which is beneficial compared to other models.
- **Discussion on Alternatives Like Poe**: Users compared their experiences with Perplexity and other AI services like **Poe**, noting that Perplexity provides a more satisfying user experience despite recent doubts.
   - One user reported switching from Poe to Perplexity after dissatisfaction with the former's interface and limitations.
- **Model Availability Concerns**: There was interest expressed in adding the new **Gemini 1.5 Pro** to the Perplexity platform, as competitors are already updating their offerings.
   - Users are keen on staying up-to-date with the latest models to maintain competitive utility in their tasks.
- **Service Stability and Reliability Issues**: Discussions revealed concerns about the stability of services like Sonnet and Opus, particularly during recent outages affecting user access.
   - One member noted that the meltdown in service is causing interruptions during important tasks, emphasizing the need for reliable performance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://status.perplexity.com/">Perplexity - Status</a>: Perplexity Status</li><li><a href="https://status.anthropic.com/">Anthropic Status</a>: no description found</li><li><a href="https://www.perplexity.ai/search/google-anti-trust-lawsuit-806TIg85QL65_n0vbFqLlg#0">Google anti trust lawsuit</a>: On August 5, 2024, a landmark ruling was issued in the antitrust case against Google, marking a significant victory for the U.S. Department of Justice and...</li><li><a href="https://x.com/testingcatalog/status/1821298236910374975?s=46&t=JsxhFTRLBknd8RUv1f73bA">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: Perplexity is getting closer and closer to UGC 👀  A small but important change - now you can notice that loads of Perplexity pages on the Discover feed are no longer curated solely by the Perplexity ...</li><li><a href="https://x.com/aravsrinivas/status/1821637031002566671?s=61">Tweet from Aravind Srinivas (@AravSrinivas)</a>: Light mode  Quoting Aravind Srinivas (@AravSrinivas)   Don&#39;t be evil
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1270882647958360154)** (11 messages🔥): 

> - `Quantum Entanglement in the Brain`
> - `Google Antitrust Lawsuit`
> - `Perplexity Pro Features`
> - `Microsoft's Advertising Strategies`
> - `Node.js Module Exports` 


- **Quantum Entanglement in the Brain Sparks Debate**: The concept of **quantum entanglement** in the brain has prompted research suggested in theories like **Orch-OR**, claiming it may influence cognition.
   - However, many argue that the brain's **warm, wet** conditions are not conducive to maintaining quantum states.
- **Google Faces Major Antitrust Ruling**: On August 5, 2024, a U.S. District Court ruled that **Google** has maintained an illegal monopoly in the online search market, marking a win for the **Department of Justice**.
   - The ruling stated, *'Google is a monopolist, and it has acted as one to maintain its monopoly,'* establishing key practices as unlawful.
- **Perplexity Pro Offers Unique Features**: **Perplexity Pro** enhances search with **real-time information retrieval** and transparent source citation, beneficial for academic research.
   - It allows users to upload and analyze documents and provides access to varying **AI models** for tailored searches.
- **Microsoft Challenges Apple's Critiques**: Microsoft's **'I'm a PC'** campaign countered Apple's **'Get a Mac'** ads, promoting the **versatility** of its products against perceived inferiority.
   - This campaign utilized **cultural commentary** and significant financial backing to reshape public perception of Windows.
- **Understanding Node.js Module Exports**: `module.exports` is essential in **Node.js** for exporting functions and values, promoting modular programming across files.
   - Each module's `exports` object allows developers to encapsulate code, facilitating better maintainability and separation of concerns.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/embed/Thm7aUsLNaU">YouTube</a>: no description found</li><li><a href="https://www.perplexity.ai/search/google-anti-trust-lawsuit-806TIg85QL65_n0vbFqLlg#0">Google anti trust lawsuit</a>: On August 5, 2024, a landmark ruling was issued in the antitrust case against Google, marking a significant victory for the U.S. Department of Justice and...</li><li><a href="https://www.perplexity.ai/search/perplexity-produ-you-shi-yao-q-OTW4JySeRQy.Pbuz1DtF9w">perplexity pro都有什么其他炫酷的功能</a>: Perplexity Pro 是一项高级订阅服务，提供了一系列增强功能，使用户的搜索体验更加丰富和高效。以下是一些主要功能： Pro Searches：每天至少可进行 300 次高级搜索，每次搜索会消耗一个积分，积分在24小时后恢复。 强大的 AI 模型：用户可以选择使用不同的高级 AI 模型，如 GPT-4...</li><li><a href="https://www.perplexity.ai/page/quantum-entanglement-in-the-br-7rokEdmsR4uZQmYOlx5J.A">Quantum Entanglement in the Brain</a>: The concept of quantum entanglement in the brain, where particles become interconnected in ways that could influence consciousness and cognition, has sparked...</li><li><a href="https://www.perplexity.ai/search/is-naturland-s-tobotronc-the-l-GVann50ESpqyNuB4wT4qvw">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://www.perplexity.ai/search/what-are-module-exports-gbtvCVuJToC16kRkJ649KQ">what are module.exports</a>: In Node.js, module.exports is a key feature of the CommonJS module system, which allows developers to export functions, objects, or values from one module so...</li><li><a href="https://www.perplexity.ai/search/generate-an-image-of-a-cat-rid-VSs5RFLnRqytc_sBugsosA">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://www.perplexity.ai/search/how-can-i-use-inline-padding-vSnbnuMzStiI1sZKGs4HhA?__cf_chl_rt_tk=Ssx1APDVq7zXMbPOHijPl8mFfEfk1JbgdUF6zBGNwT8-1723119550-0.0.1.1-8468">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://www.perplexity.ai/search/debian-ChlYBL_WRAW.x6c1nybKHg">Debian</a>: Debian is a widely recognized Linux distribution known for its commitment to free and open-source software. It was initiated by Ian Murdock in 1993 and has...</li><li><a href="https://www.perplexity.ai/search/what-is-the-link-to-your-disco-GBkoe8paT5.QBOhdWhL3nQ">What is the link to your discord?</a>: The link to join the Perplexity AI Discord community is:  [https://discord.com/invite/perplexity-ai](https://discord.com/invite/perplexity-ai).</li><li><a href="https://www.perplexity.ai/search/is-microsoft-rubbish-at-market-SEzx0MhMSxWXzJZDaJNOhw">is Microsoft rubbish at marketing?</a>: The question of whether Microsoft is &quot;rubbish&quot; at marketing is quite subjective and depends on various perspectives. However, analyzing the available data and...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1270889502541353084)** (17 messages🔥): 

> - `Perplexity API outage`
> - `Geo-based access issues`
> - `Claude outage impact`
> - `Non-English language incoherence`
> - `Google Maps URL issues` 


- **Perplexity API experiencing major outages**: Users reported being unable to access the **Perplexity API**, with some noting a **major outage** according to the status page.
   - One user expressed concern about whether the outage is limited or widespread, and several others shared their own experiences of failure to access the API.
- **Geo-based access discrepancies**: Some members suggested that the API outages might be **geo-based**, with one user from Central US noting their lack of access while another mentioned it working for them.
   - Using a **VPN to Europe** was proposed as a workaround by one user experiencing issues.
- **Claude outage may affect API functionality**: A user speculated that the issues they faced with the API might actually stem from a **Claude outage**, since they rely on it for processing results.
   - This suggests that interdependencies between services may be influencing access to the Perplexity API.
- **Incoherence in non-English language processing**: A member highlighted issues with **incoherence** in responses generated in non-English languages, citing a prompt in French that led to repetitive results.
   - This raises concerns about the effectiveness of the model in handling diverse languages and prompts accurately.
- **Challenges with Google Maps URLs**: One user asked about difficulties in generating accurate **Google Maps URLs** for a trip itinerary, noting that many provided URLs were incorrect.
   - This reflects ongoing challenges with integrating real-time data into applications and ensuring the accuracy of results obtained.



**Link mentioned**: <a href="https://docs.perplexity.ai/discuss">Discussions</a>: no description found

  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1270823988381089883)** (144 messages🔥🔥): 

> - `Stable Diffusion usage`
> - `Hardware upgrades for AI`
> - `Face swapping technology`
> - `Workflow with SAM`
> - `Web version recommendations for Mac` 


- **Using Stable Diffusion in Python Projects**: Discussion on utilizing the **Diffusers** library for implementing **Stable Diffusion** in Python projects to optimize performance and VRAM usage.
   - Members emphasized the importance of setting parameters correctly to achieve the desired output quality.
- **Upgrading Old PCs for AI Tasks**: A user described their struggles with an outdated PC setup and sought recommendations for upgrading components without needing a complete overhaul.
   - Suggestions included using services like Fiverr for assembly help and considering the purchase of barebones prebuilt PCs.
- **Face Swapping Techniques on Intel CPUs**: A user inquired about face swapping methods specifically compatible with **Intel CPUs** and expressed willingness to pay for assistance.
   - Their query emphasized a need for practical solutions targeted at users with less powerful hardware.
- **SAM Workflow for Detailing Images**: Members discussed utilizing the **SAM** detector to enhance image detailing capabilities, allowing for a versatile workflow.
   - One member highlighted the possibility of detailing various elements in images beyond just people, including backgrounds and structures.
- **Web Version Recommendations for NSFW Generation on Mac**: A user sought advice on the best web-based tools for NSFW content generation, specifying the need for compatibility with a **MacBook Air** M2 and 16GB RAM.
   - Discussion diverged into the performance implications of model complexity, emphasizing local installation speeds based on hardware capabilities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/tori29umai/sketch2lineart">Sketch2lineart - a Hugging Face Space by tori29umai</a>: no description found</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-3-medium/tree/main/text_encoders">stabilityai/stable-diffusion-3-medium at main</a>: no description found</li><li><a href="https://github.com/vosen/ZLUDA?tab=readme-ov-file#warning-important-warning">GitHub - vosen/ZLUDA: CUDA on ??? GPUs</a>: CUDA on ??? GPUs. Contribute to vosen/ZLUDA development by creating an account on GitHub.</li><li><a href="https://github.com/TraceMachina/nativelink">GitHub - TraceMachina/nativelink: NativeLink is an open source high-performance build cache and remote execution server, compatible with Bazel, Buck2, Reclient, and other RBE-compatible build systems. It offers drastically faster builds, reduced test flakiness, and specialized hardware.</a>: NativeLink is an open source high-performance build cache and remote execution server, compatible with Bazel, Buck2, Reclient, and other RBE-compatible build systems. It offers drastically faster b...
</li>
</ul>

</div>
  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1270819394943062177)** (107 messages🔥🔥): 

> - `NVIDIA Cards Performance`
> - `CPU Usage Confusion`
> - `Model Inference with GPUs`
> - `LM Studio Functionality`
> - `Tauri vs Electron` 


- **NVIDIA Cards not affected by current issues**: It was confirmed that current performance issues are affecting CPUs only, with members expressing relief regarding their NVIDIA Cards, as noted by one user.
   - Members shared personal preferences regarding CPU and GPU setups, highlighting advantages of CPU-driven workloads.
- **Confusion surrounding CPU usage reports**: A discussion arose regarding CPU usage showing over 100%, with an explanation that certain applications report total usage based on core counts.
   - Members noted differences in reporting across operating systems, emphasizing the lack of a universal standard which leads to misunderstandings.
- **Using Dual GPUs for improved model inference**: It was confirmed that LM Studio supports dual GPUs, but model inference speed remains the same as a single GPU, with denser models being loadable.
   - Discussions involved recommendations around hardware improvements to increase tokens per second for better performance.
- **LM Studio's capabilities and limitations**: Members clarified that LM Studio primarily operates locally without a REST API for direct online access, though it does provide a REST API for server tab usage.
   - The significance of using third-party applications or UI for GUI interactions was emphasized.
- **Tauri Framework vs. Electron**: A debate emerged regarding the advantages of Tauri over Electron, with personal experiences highlighting a more streamlined development process with Tauri.
   - Members expressed frustrations with Electron's community responsiveness compared to their positive experiences with Tauri.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/vilarin/Llama-3.1-8B-Instruct">Meta-Llama3.1-8B - a Hugging Face Space by vilarin</a>: no description found</li><li><a href="https://tauri.app">Build smaller, faster, and more secure desktop applications with a web frontend | Tauri Apps</a>: Tauri is a framework for building tiny, blazing fast binaries for all major desktop platforms. Developers can integrate any front-end framework that compiles to HTML, JS and CSS for building their use...</li><li><a href="https://tenor.com/view/aw-cry-sad-grandpa-gif-14766695">Aw Cry GIF - Aw Cry Sad - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/ggerganov/llama.cpp/commit/5442939fcc5e6ae41abf40612a95fd71377e487e">llama : support small Granite models (#7481) · ggerganov/llama.cpp@5442939</a>: * Add optional MLP bias for Granite models
 
 Add optional MLP bias for ARCH_LLAMA to support Granite models.
 Partially addresses ggerganov/llama.cpp/issues/7116
 Still needs some more changes to ...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1270894503204749413)** (35 messages🔥): 

> - `4090 vs. 3080 Performance`
> - `VRAM and Model Training Requirements`
> - `Mac vs. Nvidia GPUs for AI Tasks`
> - `AI Clustering with Mac Mini`
> - `Gemma Model Performance` 


- **Users question 4090 vs. 3080 performance**: A user expressed dissatisfaction with the **4090's** performance compared to the **3080**, citing only a **20 ms per epoch** difference in training speed.
   - Others observed that while the **4090** has advantages in gaming and **LLM** tasks, the **3080** still efficiently handles models under 8B.
- **Insufficient VRAM impacts model choices**: Discussions revealed that **2GB of VRAM** is not enough to run most models, with options running in low VRAM yielding poor results.
   - It was noted that larger models should be split across **VRAM** and **system RAM**, severely affecting performance.
- **Mac vs. Nvidia GPUs for efficiency in AI**: While the **Mac** offers benefits like **MLX framework**, it's generally slower for pure AI tasks compared to a high-end **4090** rig.
   - A user indicated they might return the **4090** for a **Mac**, but forum members recommended keeping the 4090 for better AI training capabilities.
- **AI clustering possibilities with Mac Mini**: Members discussed the feasibility of using **Mac Mini** systems in an **AI cluster**, highlighting their efficiency.
   - This perspective was met with enthusiasm, suggesting that the **Mac Mini** could be a viable option for AI tasks.
- **Gemma model impresses over larger alternatives**: A recommendation was made to try the **Gemma 2 27B** model, praised for its performance compared to **Yi 1.5 34B**.
   - Strategies for maximizing utility from existing models were emphasized, particularly in the context of limited resources.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.macrumors.com/2023/10/31/apple-m3-pro-less-memory-bandwidth/">Apple M3 Pro Chip Has 25% Less Memory Bandwidth Than M1/M2 Pro</a>: Apple&#39;s latest M3 Pro chip in the new 14-inch and 16-inch MacBook Pro has 25% less memory bandwidth than the M1 Pro and M2 Pro chips used in...</li><li><a href="https://nanoreview.net/en/cpu-compare/apple-m3-pro-vs-apple-m2-ultra">Apple M3 Pro vs M2 Ultra: performance comparison</a>: We compared Apple M3 Pro (4.05 GHz) against M2 Ultra (3.5 GHz) in games and benchmarks. Find out which CPU has better performance.
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1271154665509097515)** (2 messages): 

> - `GPT-4o System Card`
> - `DALL·E 3 image creation for Free users` 


- **GPT-4o System Card released**: OpenAI shared the [GPT-4o System Card](https://openai.com/index/gpt-4o-system-card/), detailing safety assessments and measures taken to track **frontier model risks**.
   - The card highlights evaluations of GPT-4o's audio capabilities and guardrails against harmful content, ensuring it only generates audio in preset voices.
- **DALL·E 3 now available for Free users**: ChatGPT Free users can now create up to **two images per day** using **DALL·E 3** for various purposes like slide decks or personalized cards.
   - Users can simply ask ChatGPT to generate the images they need, enhancing content creation and personalization options.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1270823230156050534)** (55 messages🔥🔥): 

> - `Website Access Issues`
> - `Quota and Limits`
> - `Python SDK Issues`
> - `Patent and Open Source Concerns`
> - `Model Performance Queries` 


- **Users facing website access issues**: Multiple members reported trouble accessing the main site, with worsened connectivity and errors being flagged for investigation.
   - Despite attempts to reload or access in private mode, issues persisted, leading to confirmation from others facing similar problems.
- **Confusion on message limits**: A discussion emerged over the limits imposed when sending messages on the platform, with some expressing frustration at hitting them too early.
   - Participants exchanged experiences, frequently commenting on their message encounters with the **4o** limits and differences in reaching the errors.
- **Struggles with OpenAI Python SDK**: Several users expressed difficulties replicating results after copying sample code with the latest OpenAI Python SDK, particularly with different versions leading to varied outputs.
   - One member noted a discrepancy in Python versions being used and acknowledged this as the root cause of the problem.
- **Patenting technology versus open source**: A user sought advice on patenting a technology developed with chatGPT, expressing a desire to keep it in the public domain despite understanding the high costs involved in patenting.
   - Their intention to protect ideas from exploitation while ensuring open access led to discussions about the feasibility and practicality of patents in the AI realm.
- **Query on model performance**: Questions arose regarding the performance of the **Mistral NeMo** model, particularly concerning its capacity with an M1 chip and 16GB RAM.
   - Additionally, a member commented on issues with accessing **Claude**, confirming it was experiencing a downtime.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1270819300143272116)** (25 messages🔥): 

> - `API Key Authentication Issues`
> - `GPT-4o Features for Non-Plus Users`
> - `GPT-3.5 Turbo Quota Limits`
> - `Custom GPT Updates Pending`
> - `Using CSV Files with Langchain` 


- **API Key Authentication Issues**: A user expressed struggles with passing the API key in the request, leading to a 'Missing API Key' error despite setting it in the UI.
   - Another member clarified that OpenAI handles key authentication automatically, and users may need to set 'X-Api-Key' in some cases.
- **GPT-4o Features for Non-Plus Users**: A user inquired whether GPT-4o features like vision and advanced voice mode would become available to non-Plus users.
   - Another member pointed out that costs associated with providing such features could prevent widespread rollout.
- **GPT-3.5 Turbo Quota Limits**: A user queried about experiencing 'quota' errors with a free account while using GPT-3.5-turbo.
   - A member reported that GPT-3.5-turbo is no longer available in the free tier, resulting in quota errors for free users.
- **Custom GPT Updates Pending**: A user sought clarification on the 'Updates pending' message for custom GPTs, wondering if changes to instructions were properly saved.
   - Another member suggested that it's a temporary bug, and refreshing the page usually resolves the message, ensuring that updates are applied.
- **Using CSV Files with Langchain**: A user inquired about resources for utilizing CSV files as RAG (Retrieval-Augmented Generation) documents in Langchain with OpenAI.
   - This topic remains open for discussion as no specific resources were provided in the chat.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1270853582912098477)** (11 messages🔥): 

> - `Self-Discover prompting strategy`
> - `Reverse prompting techniques`
> - `Custom GPTs development`
> - `Groq API for summary notes` 


- **Self-Discover: A New Prompt Engineering Strategy**: A member highlighted that **Self-Discover** is a powerful new prompting strategy that surpasses Chain-of-Thought (CoT) methods.
   - This technique has been explored since January, showcasing potential for tools within prompting frameworks.
- **Promising Reverse Prompting Techniques**: Members discussed the potential of using **reverse prompting** to develop more effective prompts tailored for specific use cases.
   - Such techniques allow for a shift from abstract templates to more precise outputs during the custom GPT development process.
- **Creating Custom GPTs Using the Builder**: A member shared their approach to building custom GPTs using the builder instead of the configuration pane, utilizing a dynamic commenting system.
   - By moving the comment section throughout the process, they effectively manage custom instructions derived from their template.
- **Seeking Help for Groq API Wrapper**: A member is trying to engineer a GPT wrapper with the **Groq API** to generate consistent lecture summaries and flashcards.
   - Another member pointed out that the issue lies in programming for the API rather than prompt engineering and directed them to a relevant resource.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1270853582912098477)** (11 messages🔥): 

> - `Prompt Engineering Strategies`
> - `Self-Discover Prompting`
> - `Custom GPT Development`
> - `Groq API Integration` 


- **Exploring Prompt Engineering Innovations**: A member highlighted the potential of prompt engineering techniques, particularly focusing on *prompting* and *reverse prompting* models as tools within frameworks.
   - They noted that these ideas are still in their infancy but could evolve into robust strategies.
- **Self-Discover Prompting Gains Attention**: Another member shared insights on the *Self-Discover* strategy, asserting its power and effectiveness beyond traditional Chain-of-Thought (CoT) approaches.
   - They emphasized its applicability in crafting customized prompts that yield better outputs.
- **Developing Custom GPTs with Dynamic Instructions**: A user detailed their method of creating custom GPTs using a builder template that includes a rolling comments section to guide the process.
   - This approach allows for seamless instruction updates as they work towards completing the custom instructions from the template.
- **Challenges with Groq API for Lecture Notes**: A member requested assistance in prompt engineering a GPT wrapper using the Groq API to produce reliable summaries and flashcards.
   - Another member clarified that the issue might relate more to programming for the API rather than prompt engineering, offering a link for further help.


  

---



### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1271108152330551367)** (2 messages): 

> - `MindSearch AI`
> - `Information Seeking Systems`
> - `Audio Research Communities` 


- **MindSearch AI bridges LLMs and search engines**: The paper *MindSearch: Mimicking Human Minds Elicits Deep AI Search* introduces an Agentic AI designed to improve information retrieval by mimicking human problem-solving processes through a dual-system approach: the WebPlanner and WebSearcher.
   - This structure allows MindSearch to efficiently handle **complex questions** and outperform existing search systems, demonstrating its potential in the realm of intelligent information seeking.
- **WebPlanner and WebSearcher work in tandem**: MindSearch features the **WebPlanner**, which decomposes complex queries into manageable sub-questions, and the **WebSearcher**, which finds answers by employing hierarchical retrieval tactics for effective information synthesis.
   - By leveraging these roles, MindSearch can tackle **multi-step inquiries** more adeptly than traditional LLMs.
- **Community seeking audio-focused discussions**: A member inquired about communities similar to the Nous Research Discord for audio-related topics, noting that the old Harmonai Discords have become inactive.
   - They expressed a desire for a space to engage with informed discussions on challenging research problems.



**Link mentioned**: <a href="https://x.com/intuitmachine/status/1821498263532429571?s=46">Tweet from Carlos E. Perez (@IntuitMachine)</a>: 1/n Unlocking the Web&#39;s Knowledge:  An Agentic AI That Reads Between the Links  In our age of information overload, finding the right answers often feels like searching for a needle in a haystack ...

  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1270834599953567826)** (14 messages🔥): 

> - `Machine Learning Discord Channels`
> - `Nous Artist Compliments`
> - `Commission Work`
> - `Reddit Recommendations` 


- **Recommendations for Machine Learning Discord Channels**: A member shared a [Reddit post](https://www.reddit.com/r/nousresearch/comments/1elmrjr/most_helpful_ai_discordscommunities/?share_id=L2tAJZE66RY4dOPfIbiMw) containing various AI Discord links, helpful for exploring communities.
   - Some highlighted channels include **Replete-AI**, **Unsloth**, and **Nous-Research**, providing diverse resources in AI and ML.
- **Nous Artist Gets Props**: **Hy3na_xyz** complimented the **Nous artist**, stating that their aesthetic is 'on point', demonstrating community appreciation.
   - **Kainan_e** humorously pointed out that this was a compliment, adding a light-hearted touch to the conversation.
- **Query about Commission Work**: **Hy3na_xyz** inquired if the **Nous artist**, **john0galt**, accepts commissions for work, to which they replied that it's rare and must be worthwhile.
   - This indicates an interest in potential collaborations while highlighting the exclusivity of the artist's commission work.



**Link mentioned**: <a href="https://www.reddit.com/r/nousresearch/comments/1elmrjr/most_helpful_ai_discordscommunities/?share_id=L2tAJZE66RY4dOPfIbiMw">Reddit - Dive into anything</a>: no description found

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1270896144452686040)** (1 messages): 

> - `Tavus Phoenix`
> - `Real-Time Video Cloning`
> - `Neural Radiance Fields`
> - `Video Generation API` 


- **Tavus introduces the Phoenix model**: Tavus' new **Phoenix** model offers remarkably realistic talking head videos, featuring precise **natural face movements** and expressions synchronizing with input.
   - The development teams utilized advanced techniques to bypass traditional methods, leading to enhanced realism in their video outputs.
- **Real-time capabilities on 8xH100 server**: It's reported that Tavus can run their video cloning in **real time** on a **8xH100 server**, showcasing impressive computational efficiency.
   - This capability allows for instant video generation, enhancing interactivity and engagement for users accessing the technology.
- **Accessing the Phoenix model through API**: Developers are encouraged to access the **Phoenix** model via Tavus' **Replica API**, which allows for high-level customization and realism.
   - *This API enables a variety of applications*, making it a versatile tool for content creators and developers.
- **Neural Radiance Fields redefine video creation**: The novel approach of using **neural radiance fields (NeRFs)** allows Tavus to construct dynamic, three-dimensional **facial scenes**.
   - This method significantly improves the quality and realism of generated videos, setting a new standard in the field.



**Link mentioned**: <a href="https://www.tavus.io/developer">Tavus | Developers</a>: Tavus builds advanced AI models in digital replicas, lip syncing, dubbing, text-to-video, accessible to developers via APIs.

  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1270869426954178570)** (84 messages🔥🔥): 

> - `Upside Down Poem Generation`
> - `Model Comparison`
> - `API vs Chat Interface`
> - `Training Data for Tokenization`
> - `Server Overloads` 


- **Models Struggle with Upside Down Text**: Multiple models, including Mistral, ChatGPT, and LLama 405b, fail to generate coherent upside down text, producing random sequences instead.
   - In contrast, Claude Opus and Sonnet 3.5 produce accurate and coherent upside down messages consistently.
- **Expected Performance of Claude Models**: Claude models, particularly Sonnet 3.5, appear to have superior capabilities, including the planning of text in reverse for upside down poems.
   - They also consistently recognize their own upside down texts and can rewrite them accurately, unlike other models that often generate incorrect interpretations.
- **Training Data and Tokenization Discussions**: Users discuss the need for diverse and interesting training data to improve performance on tasks like generating upside down text.
   - There are speculations on whether existing tokenization issues affect model outputs, particularly with longer lines.
- **Comments on API Usage and Server Reliability**: Users highlight that the Claude API occasionally results in overload messages, impacting usage and performance during high demand periods.
   - There is uncertainty about whether they were banned or if server issues were responsible for access problems.
- **Contemplating System Prompts and Antthinking**: Discussion arises about the role of system prompts and the antthinking feature in Claude's writing processes.
   - It appears this feature does not influence the generation of upside down poems significantly, as users report mixed responses from different interfaces.


  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1270988480272404491)** (91 messages🔥🔥): 

> - `Dataset compatibility with LM Harness`
> - `CBRN risks and model responses`
> - `Filtering pretraining data`
> - `Impact of knowledge removal on model performance`
> - `Journalistic challenges with AI-generated content` 


- **Making Datasets for LM Harness**: A member inquired if datasets for LM Harness need to be in .jsonl format, wondering about the required dictionary keys.
   - Another suggested referring to [YAML files](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/gsm8k) for more structured guidance, emphasizing the flexibility in key design.
- **Exploring CBRN Risks with AI Models**: Members debated whether a model could be designed to advise on chemistry without posing CBRN risks, with concerns that filtering knowledge may hinder scientific capability.
   - The discussion highlighted that even with knowledge removal, smart users could potentially derive harmful information, questioning the effectiveness of such filtration methods.
- **Effects of Filtering Pretraining Data**: Participants argued that removing 'bad' information from models could lead to a decrease in overall understanding and alignment effectiveness.
   - It was noted that the absence of negative examples in training might impair the model's ability to recognize and avoid harmful activities, potentially leading to a regression in competencies.
- **Perception of Journalism Regarding AI**: Members expressed frustration with how journalists portray AI models, often focusing on sensational stories about potential risks without sufficient context.
   - This perception contributed to broader concerns about safety discussions revolving around AI outputs and their misrepresentation.
- **Balancing Knowledge and Risk in AI**: A member described the complexities of ensuring AI models give accurate scientific guidance while mitigating risks related to misuse, highlighting a tension between knowledge availability and safety.
   - The conversation emphasized the fine line between enabling scientific discussion and preventing the dissemination of harmful methods, suggesting that simple filtration may not suffice for effective risk management.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/gsm8k">lm-evaluation-harness/lm_eval/tasks/gsm8k at main · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/wikitext/wikitext.yaml#L9)">lm-evaluation-harness/lm_eval/tasks/wikitext/wikitext.yaml at main · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/wikitext/preprocess_wikitext.py#L5)">lm-evaluation-harness/lm_eval/tasks/wikitext/preprocess_wikitext.py at main · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1270821750535356477)** (6 messages): 

> - `Perception Queries with RNN`
> - `Open Source Process Based Reward Models`
> - `Pythia Checkpoints and WandB Logs`
> - `Synchronizing Model Curricula` 


- **Updating Perception Queries with RNN**: A user suggested implementing **RNNs** to update perception queries over time, possibly improving efficiency.
   - This approach could enhance the adaptability of models in processing information.
- **Searching for Open Source Reward Models**: A query was raised about suitable **open source Process Based Reward Models** to verify math tasks.
   - This highlights a need for effective verification tools in mathematical problem-solving.
- **Matching Pythia Checkpoints to WandB Logs**: A user inquired about an easy method to correlate **Pythia checkpoints** with their corresponding **WandB logs**.
   - Currently, there appears to be no straightforward solution available for matching these resources.
- **Synchronizing Curricula of Models**: An inquiry was made on whether it’s possible to synchronize the curricula of two different models to maintain the same minibatches.
   - Another user suggested recording the order and grouping of training data to achieve this for the second model.


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/)** (1 messages): 

brain4brain: Ohhhhhh I see, thanks for the info
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1270824681443954772)** (3 messages): 

> - `Model Parallelism`
> - `GPU Data Splitting` 


- **Understanding Model Parallelism in GPU Systems**: It was confirmed that model parallelism involves splitting the model across several GPUs before creating copies of those segmented layers on the remaining GPUs.
   - This approach allows for efficient use of available hardware resources to enhance model performance.
- **Data Distribution Across GPUs**: With 8 GPUs, running 4 separate processes means that GPUs 0 and 1 will handle **1/4** of the data with a model copy split between them, while GPUs 2 and 3 manage another **quarter** similarly.
   - This systematic splitting ensures that each group of GPUs effectively processes its portion of the data, optimizing resource use.


  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1270841961976168571)** (49 messages🔥): 

> - `Hugging Face Acquires XetHub`
> - `Qwen2-Math Model Release`
> - `AI Infrastructure Unicorns`
> - `OpenAI's Price Reductions`
> - `Text-to-Image Leaderboard` 


- **Hugging Face Acquires XetHub**: Hugging Face announced the acquisition of [XetHub](https://www.forbes.com/sites/richardnieva/2024/08/08/hugging-face-xethub-acquisition/) to scale the collaboration for large models, enhancing their infrastructure.
   - CEO Clem Delangue highlighted that this acquisition aims to drastically improve the development process in AI, focusing on larger datasets.
- **Qwen2-Math Model Outperforms Competitors**: The new [Qwen2-Math model series](https://qwenlm.github.io/blog/qwen2-math) by Alibaba was launched, with its flagship outclassing GPT-4o and Claude 3.5 in math tasks.
   - This model series represents significant advancements in performance in specialized language processing.
- **Insight into AI Infrastructure Unicorns**: A new series discusses various infrastructure builders like Hugging Face and Databricks, which support growing generative AI markets.
   - Hugging Face's substantial financing aims to strengthen its position similar to GitHub's in the open-source opportunity.
- **OpenAI Announces Price Reductions**: Recent discussions revealed OpenAI offering a **70% price reduction** for its GPT-4o model, prompting significant interest.
   - This move could reshape pricing strategies within the industry regarding AI models.
- **New Leaders in Text-to-Image**: The Text to Image Leaderboard has been topped by FLUX.1 from bfl_ml, knocking Midjourney off its long-held position.
   - This shift indicates a competitive landscape in AI image generation, highlighting the emergence of open-weight models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.forbes.com/sites/richardnieva/2024/08/08/hugging-face-xethub-acquisition/">AI Unicorn Hugging Face Acquires A Startup To Eventually Host Hundreds Of Millions Of Models</a>: Acquired for an undisclosed sum, Hugging Face thinks the buyout will help developers build large-scale models, on par with OpenAI and Google. </li><li><a href="https://x.com/swishfever/status/1821284583171887236?s=46">Tweet from fishy business (@swishfever)</a>: cutoff date of new anthropic model (most likely) august 31st 2024</li><li><a href="https://x.com/artificialanlys/status/1821569675370930395?s=46">Tweet from Artificial Analysis (@ArtificialAnlys)</a>: Congratulations to @bfl_ml and @midjourney on taking the Artificial Analysis Text to Image Leaderboard by storm!   Welcome to the new frontier: 🥇FLUX.1 [pro] from @bfl_ml  🥈Midjourney v6.1 from @mid...</li><li><a href="https://huggingface.co/blog/introducing-private-hub">Introducing the Private Hub: A New Way to Build With Machine Learning</a>: no description found</li><li><a href="https://www.turingpost.com/p/databricks">Databricks: the Future of Generative AI in the Enterprise Arena</a>: Explore Databricks&#x27; unusual history, its contributions to the generative AI field for Enterprise, and the company&#x27;s strategy and vision of the AI industry.</li><li><a href="https://x.com/minimaxir/status/1821597473103905025?s=46">Tweet from Max Woolf (@minimaxir)</a>: OpenAI just leaked the plot of Black Mirror&#39;s next season. https://openai.com/index/gpt-4o-system-card/</li><li><a href="https://x.com/Alibaba_Qwen/status/1821553401744015816">Tweet from Qwen (@Alibaba_Qwen)</a>: Today we release a new model series for math-specific language models, Qwen2-Math, which is based on Qwen2. The flagship model, Qwen2-Math-72B-Instruct, outperforms proprietary models, including GPT-4...</li><li><a href="https://huggingface.co/blog/the-partnership-amazon-sagemaker-and-hugging-face">The Partnership: Amazon SageMaker and Hugging Face</a>: no description found</li><li><a href="https://x.com/julien_c/status/1821540661973160339?s=46">Tweet from Julien Chaumond (@julien_c)</a>: I am super excited to announce that we&#39;ve acquired XetHub! 🎊  The @xetdata team has developed technologies to enable Git to scale to TeraByte-size repositories.  Under the hood they&#39;ve been a...</li><li><a href="https://www.turingpost.com/p/huggingfacechronicle">Democratizing AI: The Hugging Face Ethos of Accessible ML</a>: The inside story of Hugging Face&#x27;s journey from chatbot builders to ML evangelists
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1270822224139517983)** (23 messages🔥): 

> - `Tokens vs Epochs`
> - `GPT-4 Token Count`
> - `Anthropic CI Debates`
> - `RLHF Critique`
> - `Recruitment Challenges` 


- **Rumors Confusing Tokens and Epochs**: There's speculation that **rumors may be messing up** the understanding of **tokens vs epochs** in discussions.
   - Several members noted the frequent confusion in online forums but provided no specific details.
- **GPT-4 Token Count Confirmed**: It's reported that **GPT-4** utilized **10 trillion tokens**, a number echoed by multiple members in the chat.
   - There's a consensus that this figure seems accurate, even as members remarked that GPT-4 is now considered **ancient technology**.
- **Anthropic's Use of Debate Training**: Discussions included curiosity about how **Anthropic's** debate training contributes to their **Claude 3 model**, which may already use agent debates.
   - It was mentioned that an update on this topic was posted on the alignment forum, drawing significant interest from members.
- **Critical Overview of RLHF**: A member quoted **Karpathy**, stating that **RLHF is just barely RL**, as it relies heavily on a reward model that mimics human preferences.
   - The critique highlighted potential pitfalls of RLHF, asserting it as fundamentally different from real reinforcement learning techniques seen in systems like **AlphaGo**.
- **Challenges in Recruitment**: There's a sense of urgency in recruitment, as one member expressed the need for people who have actual experience in training **LLMs**.
   - Amid this, humor surfaced with playful remarks about **Karpathy's** video and its role in hiring, emphasizing a lighter community vibe.



**Link mentioned**: <a href="https://x.com/karpathy/status/1821277264996352246?s=46">Tweet from Andrej Karpathy (@karpathy)</a>: # RLHF is just barely RL  Reinforcement Learning from Human Feedback (RLHF) is the third (and last) major stage of training an LLM, after pretraining and supervised finetuning (SFT). My rant on RLHF i...

  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1270891264606343332)** (6 messages): 

> - `Gary Marcus Predictions`
> - `Audience Capture`
> - `Contrarian Perspectives on AI` 


- **Gary Marcus predicts AI bubble collapse imminently**: [Gary Marcus](https://x.com/GaryMarcus/status/1819525054537126075) expressed regret over his prediction that the AI bubble would collapse in 2025, stating, *“It’s going to be days or weeks from now, not months.”*
   - This prediction incited discussions regarding his credibility, with opinions noting that such takes seem to lack seriousness.
- **Nathan's take on Gary Marcus**: Nathan described Gary Marcus as a *bozo* driven by irrelevant takes and a history of peculiar viewpoints, suggesting a level of audience capture in his commentary.
   - He voiced concerns, stating, *“I have a hard time with people who make their career on tech but chronically hate tech.”*
- **Mixed views on Marcus's insights**: Another member acknowledged that while Gary Marcus provides some sensible critiques regarding **LLMs**, he also has a penchant for contrarianism.
   - They noted that his genuine points get overshadowed by a desire to be right, seeking opportunities to proclaim, *“I told you so.”*



**Link mentioned**: <a href="https://x.com/GaryMarcus/status/1819525054537126075">Tweet from Gary Marcus (@GaryMarcus)</a>: I just wrote a great piece for WIRED predicting that the AI bubble will in collapse in 2025, and now I wish I hadn’t.  Clearly, I got the year wrong. It’s going to be days or weeks from now, not month...

  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/)** (1 messages): 

chygao: https://youtu.be/6QWuJRvMtxg?si=SYXsRvYbfcdtYLC2
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/)** (1 messages): 

SnailBot News: <@&1216534966205284433>
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1270987563343024200)** (75 messages🔥🔥): 

> - `LangChain issues with AWS Lambda`
> - `Limiting chat history in LLM applications`
> - `Managing user-specific history in Slack RAG app`
> - `Comparing LangChain with other frameworks`
> - `Challenges with different LLMs` 


- **Troubleshooting LangChain in AWS Lambda**: A user reported difficulties with **LangChain imports** in their AWS Lambda function, specifically related to **pydantic** errors when trying to import LangChain modules.
   - They confirmed using Python **3.12** runtime and properly set up a lambda layer, leading to suggestions about potential version conflicts.
- **Chat history management for LLMs**: Discussion arose about a user’s implementation of limiting chat history in an LLM application to improve performance and maintain user context.
   - It was noted that a **custom function** would be required to standardize the number of messages retained in the chat history across users.
- **LangChain vs. Other Frameworks**: A user expressed frustration that switching from **OpenAI** to **Anthropic** using LangChain required rewriting code due to differences in LLM functionalities.
   - It was agreed that while LangChain offers some abstraction, LLMs still require specific adjustments based on their characteristics.
- **Challenges with LLM Downtime**: Another user highlighted that **Anthropic's Claude 3.5** faced an internal server error, expressing concerns about the reliability of AI systems for production use.
   - This raised questions about the overall readiness of AI systems and whether LangChain was the right choice for their needs.
- **Seeking B2B sales mentorship**: A user inquired about acquiring source code and having a mentor to learn more about **B2B sales**.
   - This indicates a desire for practical guidance and resources to navigate sales in the business landscape.



**Link mentioned**: <a href="https://python.langchain.com/v0.2/docs/how_to/trim_messages/">How to trim messages | 🦜️🔗 LangChain</a>: This guide assumes familiarity with the following concepts:

  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/)** (1 messages): 

_johnny1984: Stupid spambot doesn't stand a change against my rapper AI:
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1270845417889861702)** (61 messages🔥🔥): 

> - `GPT-4o capabilities`
> - `Gemini 1.5 Flash updates`
> - `DALL·E 3 free access`
> - `Mistral Agents`
> - `Academic papers on AI` 


- **GPT-4o offers extensive input and output capabilities**: The GPT-4o model can process any combination of text, audio, image, and video, significantly enhancing its versatility and speed with a response time similar to humans.
   - It also boasts improved performance across languages and is 50% cheaper in API usage compared to previous models.
- **Gemini 1.5 Flash sees major updates for developers**: GoogleAI has reduced the Gemini 1.5 Flash pricing by approximately 70%, making it more accessible for developers.
   - Additionally, AI Studio is now available to all workspace customers, allowing for greater experimentation and integration with new languages.
- **OpenAI rolls out DALL·E 3 for free users**: ChatGPT Free users can now create up to two images per day using DALL·E 3, enhancing accessibility for casual users.
   - This feature enables users to create personalized content easily, though some skepticism remains about its wider usage.
- **Mistral Agents expand capabilities**: The Mistral Agents are now able to utilize Python and integrate into various workflows, showcasing their versatility.
   - Users expressed interest in features that allow for API consumption and the application of these capabilities in practical scenarios.
- **Insights on academic AI research papers**: Discussion around recent academic papers, such as those related to the llama3 model, indicates a growing interest in multimodal models.
   - Authors are sharing their research openly, providing access to code, data, and weights to foster community collaboration.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://buildship.com)">no title found</a>: no description found</li><li><a href="https://news.ycombinator.com/item?id=41184559">no title found</a>: no description found</li><li><a href="https://news.ycombinator.com/item?id=41188647">no title found</a>: no description found</li><li><a href="https://docs.mistral.ai/capabilities/agents/">Agents | Mistral AI Large Language Models</a>: What are AI agents?</li><li><a href="https://homebrew.ltd/blog/can-llama-3-listen">Homebrew</a>: Building human-augmenting AIs that run on energy-efficient hardware.</li><li><a href="https://x.com/OfficialLoganK/status/1821601298195878323">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: Good news for @GoogleAI developers:  - Gemini 1.5 Flash price is now ~70% lower ($0.075 / 1M) - Gemini 1.5 Flash tuning available to all - Added support for 100+ new languages in the API - AI Studio i...</li><li><a href="https://x.com/karpathy/status/1821286855310242020">Tweet from Andrej Karpathy (@karpathy)</a>: Fair, I couldn&#39;t find a picture like that in a quick google search. I&#39;d spend some time to make one but I was worried that this would have a risk of being misleading in a different way. In Go ...</li><li><a href="https://x.com/karpathy/status/1821277264996352246?s=46&t=6FDPaNxZcbSsELal6Sv7U">Tweet from Andrej Karpathy (@karpathy)</a>: # RLHF is just barely RL  Reinforcement Learning from Human Feedback (RLHF) is the third (and last) major stage of training an LLM, after pretraining and supervised finetuning (SFT). My rant on RLHF i...</li><li><a href="https://x.com/vboykis/status/1821527144922566745">Tweet from vicki (@vboykis)</a>: The moat right now is the model, the moat in a year will be memory (how well the system remembers YOU and your queries and picks the right prompt and model, aka we are headed back to recsys real soon....</li><li><a href="https://x.com/mckaywrigley/status/1821307469114769903?s=46">Tweet from Mckay Wrigley (@mckaywrigley)</a>: Here’s a 17min deep dive on advanced prompting techniques for LLMs.  Fully demonstrated on a real-world, multi-step AI workflow.  Watch for a complete breakdown.</li><li><a href="https://x.com/OpenAI/status/1821644904843636871">Tweet from OpenAI (@OpenAI)</a>: We’re rolling out the ability for ChatGPT Free users to create up to two images per day with DALL·E 3.   Just ask ChatGPT to create an image for a slide deck, personalize a card for a friend, or show ...</li><li><a href="https://x.com/clementdelangue/status/1821559961555554469?s=46">Tweet from clem 🤗 (@ClementDelangue)</a>: This is the real 🍓 - welcome to @xetdata - we&#39;re just getting started!</li><li><a href="https://x.com/openai/status/1821595015279472736?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from OpenAI (@OpenAI)</a>: We’re sharing the GPT-4o System Card, an end-to-end safety assessment that outlines what we’ve done to track and address safety challenges, including frontier model risks in accordance with our Prepar...</li><li><a href="https://qwenlm.github.io/blog/qwen2-math/">Introducing Qwen2-Math</a>: GITHUB HUGGING FACE MODELSCOPE DISCORD 🚨 This model mainly supports English. We will release bilingual (English and Chinese) math models soon. Introduction Over the past year, we have dedicated signi...</li><li><a href="https://x.com/karpathy/status/1821277264996352246?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Andrej Karpathy (@karpathy)</a>: # RLHF is just barely RL  Reinforcement Learning from Human Feedback (RLHF) is the third (and last) major stage of training an LLM, after pretraining and supervised finetuning (SFT). My rant on RLHF i...</li><li><a href="https://x.com/_clashluke/status/1820810798693818761">Tweet from Lucas Nestler (@_clashluke)</a>: http://x.com/i/article/1820791134500642816</li><li><a href="https://x.com/karpathy/status/1821257161726685645?s=46">Tweet from Andrej Karpathy (@karpathy)</a>: At one point a while back autoregressive language model papers were like that too. Formulating the joint likelihood, factorizing it, deriving the maximum likelihood estimate, discussing connections to...</li><li><a href="https://x.com/sama/status/1821207141635780938">Tweet from Sam Altman (@sama)</a>: i love summer in the garden
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1270857754445221990)** (5 messages): 

> - `SAM 2 Pod Launch`
> - `User Statistics for SAM`
> - `Future Predictions for SAM 2`
> - `Video Content in SAM 2`
> - `Connections to Past Episodes` 


- **Launch of the SAM 2 Pod**: The latest episode of [Latent Space podcast](https://x.com/latentspacepod/status/1821296511260504408) is now live, focusing on **SAM 2**.
   - Listeners are encouraged to check out the episode featuring insights from **Nikhila Ravi** and **Joseph Nelson**.
- **SAM User Statistics Reveal Massive Impact**: A fun quote from guest co-host mentions that **49 million images** were labeled using **SAM** on RoboFlow, saving an estimated **35 years** of user time.
   - In just the last **30 days**, **5 million** of those images were labeled, highlighting the ongoing user engagement.
- **Predictions for SAM 2's Efficiency**: With **SAM 2** being **6x faster** and capable of processing video, there's excitement about the amount of time it could potentially save users.
   - *Given the past success, the community wonders how much more time savings will be realized with the new capabilities.*
- **Increased Emphasis on Video Content**: The SAM 2 episode features more **video content** than usual, marking a shift in the podcast’s presentation style.
   - Listeners are directed to the [YouTube demo](https://www.youtube.com/watch?v=lOO_gH4kAn8) accompanying the episode for hands-on insights.
- **Referencing Past Successful Episodes**: The podcast hosts draw connections to their previous coverage of various AI tools, positioning **SAM 2** within their ongoing narrative.
   - Episodes like [Segment Anything 1](https://www.latent.space/p/segment-anything-roboflow) and others are mentioned as foundational to their evolving discussion.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/latentspacepod/status/1821296511260504408">Tweet from Latent.Space (@latentspacepod)</a>: http://x.com/i/article/1821295838208909312</li><li><a href="https://x.com/swyx/status/1821298796841541956">Tweet from swyx 🍓 (@swyx)</a>: Our SAM 2 pod with @nikhilaravi is out! Fun SAM1 quote from guest cohost @josephofiowa:  &#34;I recently pulled statistics from the usage of SAM in @RoboFlow over the course of the last year. And user...</li><li><a href="https://www.latent.space/p/sam2">Segment Anything 2: Demo-first Model Development</a>: Don&#x27;t bother keeping absolutely still: This vision model has memory now! Covering SAM 2 with Nikhila Ravi of Facebook AI Research, and special returning guest host Joseph Nelson of Roboflow</li><li><a href="https://youtu.be/lOO_gH4kAn8">Segment Anything 2: Memory + Vision = Object Permanence — with Nikhila Ravi and Joseph Nelson</a>: Don&#39;t bother keeping absolutely still: This vision model has memory now! Covering SAM 2 with Nikhila Ravi of Facebook AI Research, and special returning gues...
</li>
</ul>

</div>
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1270826421563953152)** (47 messages🔥): 

> - `Midjourney CEO's stance on open source`
> - `ASL model discussion`
> - `Synthetic voice dataset creation`
> - `Flux image generation`
> - `AI applications for accessibility` 


- **Midjourney CEO critiques open source**: Midjourney CEO expressed **skepticism towards open source**, arguing that local models can't compete with their service using **64 GPUs**. He claimed that open source models lack coherence and make no business sense, dismissing **controlnet** as a lone success.
   - Critics countered that Midjourney's product is akin to **inferior versions** of what open source can achieve, and pointed to the **issues of overfitting** in Flux: 'it just has a sort of plastic look to it.'
- **ASL language model concept emerges**: A user inquired about developing an app that translates **speech to ASL**, considering the challenges of training a model with images of signs. Suggestions were made to fine-tune existing models to create an effective **ASL translation tool**.
   - Another user offered input on refining voice recognition models to examine whether they could effectively **harness** emojis for **representing hand gestures**.
- **Synthetic voice dataset idea proposed**: A member proposed using **so-vits-svc** to create synthetic datasets by transforming voices in audio files, enhancing variety while retaining content. This could potentially allow models to capture a wider range of **emotions** in voice representation.
   - The proposal emphasized generating a more diverse dataset, facilitating better outcomes in scenarios where models struggle to differentiate between basic **demographic classifications**.
- **Flux model discussions continue**: There was an ongoing dialogue among users regarding **Flux**, with some describing it as 'a fun toy' but stating that it hasn't significantly progressed. Concerns about its apparent **overfitting** were expressed, posing questions about its overall effectiveness in image generation.
   - Discussion highlighted the perception of **Flux** in comparison to Midjourney, with remarks about the importance of more innovative and intentional **fine-tuning**.
- **Multiple AI applications for accessibility**: Various suggestions emerged for AI applications aimed at enhancing accessibility, such as a **privacy-respecting** IP Relay app for speech recognition. The discourse emphasized leveraging **local inference techniques** to assist those with hearing impairments by converting speech to text locally.
   - Crowdsourced ideas illustrated a strong interest in harnessing AI for meaningful impact, particularly focusing on real-world applications that transcend simple tech solutions.


  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1270903254041235507)** (2 messages): 

> - `Frequency Space Analysis`
> - `Visual Data Comparison` 


- **Frequency Space Analysis Awaits**: A member expressed that they haven't explored the **raw data in frequency space** yet and indicated a need to do so.
   - This suggests a potential area for deeper investigation within the dataset.
- **Visual Data Comparison Seems Uniform**: Another member suspects that the data in frequency space looks nearly **identical to the human eye**.
   - This perception raises questions about the visual discernibility of the data's variations.


  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1270999957884895243)** (3 messages): 

> - `Multi-backend Refactor`
> - `Google Gemini Price Cuts`
> - `H100 in the Metaverse` 


- **Multi-backend Refactor Installed Smoothly**: A member confirmed they successfully installed the **multi-backend-refactor** without any issues.
   - They expressed readiness to keep an eye on ongoing developments.
- **Google Gemini Slashes Prices**: A member shared a [YouTube video](https://www.youtube.com/watch?v=3ICC4ftZP8Y) titled 'Google Gemini Insane Price Cuts!!!' highlighting new reductions on **Gemini 1.5 Flash**.
   - The video pointed out significant markdowns in pricing, directing viewers to the [Google Developers blog](https://developers.googleblog.com/en/gemini-15-flash-updates-google-ai-studio-gemini-...) for detailed information.
- **Call for H100s in the Metaverse**: A member humorously stated that **Zuck** needs to provide **H100** GPUs in the metaverse.
   - This highlights the ongoing demand for advanced computing resources in virtual environments.



**Link mentioned**: <a href="https://www.youtube.com/watch?v=3ICC4ftZP8Y">Google Gemini Insane Price Cuts!!!</a>: Google Gemini 1.5 Flash has some insane price cuts!🔗 Links 🔗Details - https://developers.googleblog.com/en/gemini-15-flash-updates-google-ai-studio-gemini-...

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1271078532713287692)** (29 messages🔥): 

> - `Training Dataset Size`
> - `Prompt Formatting for Inference`
> - `LoRA Import Errors`
> - `Using Alpaca Format for Fine-tuning`
> - `Llama 3 Model Information` 


- **Training with 38k Dataset**: A member mentioned that they trained their model with a **38k** item dataset, which took **32 hours** on an **RTX 4090**.
   - They speculated that the **learning rate** might be too high in their configuration.
- **Correct Prompt Formatting**: Multiple members discussed the importance of using the **Alpaca format** for task-specific prompts during inference.
   - They highlighted that for chatting, the output needs to follow the same format that was used during fine-tuning.
- **LoRA Import Errors**: A user encountered errors when importing their **LoRA** and had to remove two values from their adapter configuration.
   - Another member suggested merging the LoRA into the base model to potentially resolve this issue.
- **Configuration Clarifications**: Clarifications were made regarding the distinction between fine-tuning configurations and conversation prompt requirements.
   - It was emphasized that using the correct output structure is crucial for effective model performance.
- **Details on Llama 3 Training**: A member inquired about the training specifics of the **Llama 3.1 70B** model and the data/masks used during its fine-tuning.
   - They noted the renaming of existing tokens to serve as special tokens instead of creating new ones.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/axolotl-ai-co/llama-3-8b-chatml">axolotl-ai-co/llama-3-8b-chatml · Hugging Face</a>: no description found</li><li><a href="https://github.com/oobabooga/text-generation-webui">GitHub - oobabooga/text-generation-webui: A Gradio web UI for Large Language Models.</a>: A Gradio web UI for Large Language Models. Contribute to oobabooga/text-generation-webui development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1271064477319692339)** (8 messages🔥): 

> - `Querying PDFs via API`
> - `Retrieval-Augmented Generation`
> - `Cohere Documentation`
> - `Cohere and Fujitsu Partnership`
> - `Langchain Integration` 


- **Almosnow seeks API file upload guidance**: A member wanted to replicate the PDF querying functionality from the UI on coral.cohere.com using the API, but couldn't find the relevant documentation.
   - They noted various POST endpoints like `/v1/conversations/upload_file` but were unclear if these are officially documented.
- **Mapler provides helpful resources**: Another member, mapler, responded with resources on using Retrieval-Augmented Generation via the Cohere API, linking to a [blog post](https://cohere.com/llmu/rag-start).
   - They also shared documentation for RAG and a code snippet to demonstrate how to produce grounded answers.
- **Almosnow appreciates the help**: Almosnow expressed gratitude for the resources shared by mapler and indicated they would review the provided materials.
   - This shows a collaborative atmosphere where members are eager to assist each other in their inquiries.
- **Introduction by Rashmi**: A member named rashmi introduced themselves with a simple greeting, indicating their presence in the discussion.
   - This highlights the community aspect, where new members are welcomed.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://cohere.com/llmu/rag-start">Getting Started with Retrieval-Augmented Generation</a>: Part 1 of the LLM University module on Retrieval-Augmented Generation.</li><li><a href="https://docs.cohere.com/docs/retrieval-augmented-generation-rag">Retrieval Augmented Generation (RAG) - Cohere Docs</a>: Retrieval Augmented Generation (RAG) is a method for generating text using external data sources to improve accuracy. The Chat API in combination with the Command model can help generate grounded text...
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1270978870241132588)** (5 messages): 

> - `Cohere embeddings error`
> - `RAG model with Llamaparse`
> - `Azure AI Search integration issues`
> - `500 errors from embed endpoint`
> - `Using preamble ID for prompts` 


- **Cohere embeddings encounter issue**: A user faced a ValueError message stating it could not convert string to float when using `CohereEmbeddings` with a provided API key.
   - *An error occurred: could not convert string to float: 'float_'* indicates an underlying issue with input formatting.
- **RAG model struggles with Llamaparse**: One user inquired about successfully using the RAG model combined with Embedding and Reranking processes alongside Llamaparse, where lines may contain mixed incomplete sentences.
   - They sought insights from the community on making sense of non-related text chunks during data processing.
- **Inconsistent results from Azure AI Search integration**: Another user reported inconsistent query results when integrating Cohere embeddings into the Azure AI Search index, despite having vectorized data successfully.
   - '@search.answers' often returned empty even when 'value' had scored documents, hindering effective RAG usage.
- **Encountering 500 errors on embed endpoint**: A user mentioned experiencing intermittent 500 errors from the embed endpoint, referencing a specific error ID for tracking.
   - They requested assistance in determining if their data submission was the cause of these errors.
- **Seeking help for using preamble ID**: A user reached out for help with utilizing a preamble ID to generalize prompts across entire input texts.
   - Their inquiry indicated a need for guidance on effective prompt engineering techniques.



**Link mentioned**: <a href="https://learn.microsoft.com/en-sg/azure/search/vector-search-integrated-vectorization-ai-studio?tabs=cohere).">Integrated vectorization with models from Azure AI Studio - Azure AI Search</a>: Learn  how to vectorize content during indexing on Azure AI Search with an AI Studio model.

  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1270831369727115375)** (16 messages🔥): 

> - `Cohere-toolkit`
> - `Default tool activation`
> - `Preamble adjustment`
> - `Custom deployment` 


- **Enabling a Tool by Default in Cohere-toolkit**: A member discussed how to enable a tool by default within the **Cohere-toolkit**, suggesting to add `always use the <tool> tool` to the preamble.
   - They noted that it’s essential to include the tool in the tools list for it to function correctly.
- **First Attempt at Default Tool Loading**: Another member shared their experience modifying `invoke_chat_stream` in `cohere_platform.py` for default tool loading and adding a preamble.
   - They also expressed the intention to create a custom deployment that offers limited model selection with default tools.
- **UI Implementation Oddities**: It was mentioned that while the model uses the tool, the UI does not show it as activated, leading to some confusion.
   - This discrepancy raises questions about UI feedback versus the actual functionality of the model.
- **Tool List Requirement for Proper Functionality**: Members confirmed that for the model to utilize the tool, it must also be added to the tool list in the call to `co.chat`.
   - This is necessary regardless of whether it's a custom model or not.


  

---



### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/)** (1 messages): 

jerryjliu0: happening in 5 minutes! ^^
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1270892087906402334)** (3 messages): 

> - `RAG pipeline observability`
> - `Workflows abstraction`
> - `LlamaIndex Sub-Question Query Engine`
> - `Agent debugging with Workflows` 


- **RAG Pipeline Needs Enhanced Observability**: An underrated necessity for **RAG pipelines** is to not only capture query-time traces but also examine how source documents are chunked. Concerns were raised about retrieval issues correlated to improper context chunking, asking if it is cut in the middle.
   - This highlights the importance of ensuring optimal chunking for improving retrieval performance, as noted in a recent [tweet](https://twitter.com/llama_index/status/1821332562310205918).
- **Excitement around Workflows for Gen AI**: The team continues to be enthusiastic about **Workflows**, a new abstraction for building complex agentic generative AI applications. They demonstrated its ease in handling real workflows by showcasing the rebuild of LlamaIndex's built-in Sub-Question Query Engine in a [new video](https://twitter.com/llama_index/status/1821575082516660440).
   - This sets a strong foundation for deploying complex query engines and workflows effectively.
- **Power of Building Agents with Workflows**: A blog post from **@ArizePhoenix** emphasized the advantages of using **Workflows** to observe and debug agents. It detailed how an event-based architecture allows for more flexible and cyclical designs compared to traditional approaches, shared in the [post](https://twitter.com/llama_index/status/1821617012080308543).
   - This serves as a valuable resource for those looking to enhance their agent-building capabilities using the **Phoenix** platform.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1271011022538276928)** (20 messages🔥): 

> - `LongRAG paper`
> - `Self-routing techniques`
> - `Evaluation benchmarks for LLMs`
> - `Token size measurement`
> - `Using APIs with LlamaIndex` 


- **LongRAG paper comparison sparks interest**: The LongRAG paper was shared, which compares Retrieval Augmented Generation (RAG) and long-context LLMs, noting that **long-context models** outperform RAG when sufficiently resourced.
   - Members expressed a wish for comparisons involving **Claude 3.5**, as well as discussing *methodologies presented by Lance from LangChain* in relation to long contexts.
- **Self-routing technique improves efficiency**: The proposed **Self-Route method** in the LongRAG paper routes queries to RAG or long-context models based on self-reflection, significantly cutting computation costs while maintaining performance.
   - Members suggested innovations like **parent-document retrieval** using metadata to enhance retrieval systems, highlighting the challenges in creating reliable metadata labeling.
- **Evaluation benchmarks raise concerns**: Discussion emerged around the susceptibility of evaluation datasets to **data leakage**, particularly those based on known datasets, impacting model assessment accuracy.
   - It was noted that many datasets are too cleanly formatted, which does not reflect real-world applications, suggesting a need for more **realistic evaluation metrics**.
- **Token sizing tools recommended**: A member inquired about tools for measuring document token sizes, prompting recommendations for **tiktoken** from OpenAI as a popular option, especially for those using OpenAI models.
   - There was interest in finding a tool accessible to the community for this purpose, with GitHub resources being shared.
- **Integrating APIs with LlamaIndex**: A member sought guidance on using **serpapi** or **serperapi** with LlamaIndex to enhance query results, aiming for integration that includes web crawling capabilities.
   - In response, it was suggested that users could create custom engines that leverage serpapi or utilize agents with built-in serpapi tools.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/abacaj/status/1821626025584828747?s=46&t=lqHyE7PE7Ct6jUJNcJPPxg">Tweet from anton (@abacaj)</a>: 42 page PDF (it says 30k tokens) that contains both images and text thrown into gemini-flash 1.5, gets every answer correct... it&#39;s so over</li><li><a href="https://arxiv.org/abs/2407.16833">Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach</a>: Retrieval Augmented Generation (RAG) has been a powerful tool for Large Language Models (LLMs) to efficiently process overly lengthy contexts. However, recent LLMs like Gemini-1.5 and GPT-4 show excep...</li><li><a href="https://www.youtube.com/watch?v=UlmyyYQGhzc)">Is RAG Really Dead? Testing Multi Fact Retrieval &amp; Reasoning in GPT4-128k</a>: One of the most popular benchmarks for long context LLM retrieval is @GregKamradt&#39;s Needle in A Haystack: a fact (needle) is injected into a (haystack) of co...</li><li><a href="https://github.com/openai/tiktoken">GitHub - openai/tiktoken: tiktoken is a fast BPE tokeniser for use with OpenAI&#39;s models.</a>: tiktoken is a fast BPE tokeniser for use with OpenAI&#39;s models. - openai/tiktoken
</li>
</ul>

</div>
  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1270862078797811853)** (1 messages): 

> - `LLAMA 3 model performance`
> - `GitHub Issue #1285` 


- **Concerns on LLAMA 3 Generation Quality**: A member reported using the **LLAMA 3 8B instruct model** and prompted it with 'anything,' resulting in unexpected outputs.
   - *Let me know if there is any other input needed from my side,* prompting others to share their experiences or discuss the issue on the [GitHub issue page](https://github.com/pytorch/torchtune/issues/1285).
- **Discussion on GitHub Issue #1285**: The member referenced a specific [GitHub issue](https://github.com/pytorch/torchtune/issues/1285) that addresses generation quality concerns with the model.
   - They invited others to contribute comments or insights regarding the topic, emphasizing the need for collective feedback.



**Link mentioned**: <a href="https://github.com/pytorch/torchtune/issues/1285">Generation quality · Issue #1285 · pytorch/torchtune</a>: I use the LLAMA 3 8B instruct model and prompt it with &quot;anything&quot; and I get the below result: chat_format: null checkpointer: _component_: torchtune.utils.FullModelMetaCheckpointer checkpoin...

  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1270849984954237081)** (17 messages🔥): 

> - `RTX A4000 and A2000 Performance`
> - `Memory Optimization Techniques`
> - `RLHF Cleanup Discussions`
> - `Torchchat Generation Optimizations`
> - `Documentation and Tutorial Plans` 


- **Evaluating RTX A4000 and A2000 for Fine Tuning**: Members discussed the performance characteristics of the **RTX A4000** and **RTX A2000**, both with **16GB** of memory, noting that despite good memory metrics, the performance for full fine-tune on **1.5B** appears somewhat low.
   - One member suggested managing memory costs by increasing the default batch size, potentially fitting the workload into **12GB**.
- **Memory Optimization Parameters Under Review**: A member mentioned **guesswork** on memory optimization parameters, indicating that while configurations like **LoRA** are effective, they are not focused on them for the moment.
   - They acknowledged that others might have GPUs with **8GB VRAM** which could perform over **2x** faster, hinting at broader optimization potential.
- **Discussion on RLHF Cleanup**: A member inquired about necessary cleanups on **RLHF** before broader public sharing, recalling earlier mentions of needing adjustments.
   - There was an indication of a willingness to collaborate on creating a **tutorial** or **blog post**, recognizing the effort it would require.
- **Plans to Publicize and Document**: The same member expressed eagerness to initiate discussions about **publicizing** the work and developing **documentation** or **tutorials**, with a loose roadmap in mind.
   - They welcomed any input and assistance from the community to enhance these efforts.
- **Torchchat Generation Speed Ups**: A member mentioned an interest in investigating how **torchchat** achieves faster model generation speeds, seeking insights into their optimizations.
   - This inquiry aligns with ongoing efforts to streamline and expedite the performance of different models discussed.



**Link mentioned**: <a href="https://wandb.ai/salman-mohammadi/torchtune/?nw=nwusersalmanmohammadi">salman-mohammadi</a>: Weights & Biases, developer tools for machine learning

  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1270845213241376928)** (4 messages): 

> - `AI Infrastructure Deployment`
> - `Commercialization Concerns`
> - `Internal Tools Usage` 


- **Freedom to Build on AI Infrastructure**: Members discussed that you are free to build anything and deploy it as long as there is no intent to commercialize an AI infrastructure platform, according to the [pricing page](https://link.to.pricing).
   - *As long as it is not commercial* and it truly qualifies as an internal tool, it seems to be acceptable.
- **Clarification on Internal Tools**: One member expressed that internal tools are likely fine as long as they don't aim for commercialization, though they're not an authority on the matter.
   - They reiterated that clear guidelines for using these tools are still somewhat ambiguous.
- **Commercialization Assistance Offer**: A member humorously noted that if anyone does want to commercialize an AI infrastructure platform, they should reach out to Modular for assistance.
   - This reflects the community's openness to innovation while navigating the commercialization landscape.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1270854292747714682)** (13 messages🔥): 

> - `Running Mojo in Windows DEV Environment`
> - `VS Code with WSL Support`
> - `FancyZones Utility`
> - `Active Directory and Distributed Databases` 


- **Using VS Code with WSL for Mojo Development**: A user inquired about running **Mojo** in a Windows development environment after installing **Mojo Max** on WSL. Another user suggested using **VS Code**, which supports WSL to edit in Windows while building in Linux.
   - *You pretty much forget you're developing in Linux* when using this setup.
- **Benefits and Limitations of WSL**: Discussion highlighted that **WSL** offers a quarantined development environment away from antivirus interference, although it still runs on **C** drive. Members noted the limitations related to *reproducibility* and other advantages WSL provides.
   - One noted the peculiar situation of balancing between Windows and Linux environments: *You just have to live a dual life*.
- **FancyZones Utility for Windows**: A member shared a link to [FancyZones utility](https://learn.microsoft.com/en-us/windows/powertoys/fancyzones), a tool that helps arrange and snap windows into efficient layouts to improve workflow. This utility allows customizable zone locations for better window management on Windows.
   - Dragging windows into a defined zone resizes and repositions them, enhancing efficiency while developing.
- **Debate on Active Directory as Distributed Database**: A member made a humorous remark that calling **Active Directory** a distributed database is an insult to real distributed databases. They detailed its sync nature, mentioning it only provides availability without true consistency or partition tolerance.
   - Another member confirmed that Microsoft does indeed run distributed databases on Windows, sparking further discussion on the topic.



**Link mentioned**: <a href="https://learn.microsoft.com/en-us/windows/powertoys/fancyzones">PowerToys FancyZones utility for Windows</a>: A window manager utility for arranging and snapping windows into efficient layouts

  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/)** (1 messages): 

seanchatmangpt: https://www.loom.com/share/0ffc1312c47c45fdb61a2ad00102b3da
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1270907182262652931)** (10 messages🔥): 

> - `Inspect for LLM observability`
> - `DSPy vs. Langgraph`
> - `Performance of optimize_signature vs. COPRO` 


- **Inspect for LLM brings eval capabilities**: A user inquired about experiences with [Inspect](https://github.com/UKGovernmentBEIS/inspect_ai) for LLM observability and how well it integrates with DSPy.
   - No specific user experiences were shared, but the tool appears to focus on large language model evaluations.
- **DSPy and Langgraph operate at different levels**: A member explained that DSPy optimizes instructions and examples in the prompt space, while LangGraph serves as a lower-level interface for working with LangChain.
   - Essentially, DSPy enhances performance, whereas LangGraph focuses on system architecture.
- **Optimize_signature outperforms COPRO on GSM8K**: One user reported better results using **optimize_signature** over **COPRO** for Chain of Thought (CoT) tasks on GSM8K, achieving a score of **20/20** swiftly.
   - In contrast, COPRO failed to reach a zero-shot instruction solution, with the highest score being **18/20**.
- **Users share experiences with DSPy in production**: A user asked if anyone uses DSPy in production and what the differences are between Langgraph and DSPy.
   - The discussion highlighted that while both tools are complementary, DSPy is more focused on optimization.



**Link mentioned**: <a href="https://github.com/UKGovernmentBEIS/inspect_ai">GitHub - UKGovernmentBEIS/inspect_ai: Inspect: A framework for large language model evaluations</a>: Inspect: A framework for large language model evaluations - UKGovernmentBEIS/inspect_ai

  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1270941482240442471)** (1 messages): 

> - `DSPy-Multi-Document-Agent`
> - `requirements.txt file` 


- **User struggles to find requirements.txt**: A member expressed difficulty in locating the **requirements.txt** file for the **DSPy-Multi-Document-Agent**.
   - *Am I missing anything?* was the query left, indicating a possible gap in documentation or guidance.
- **Clarification sought about missing files**: The same member specifically asked if they were missing anything related to the **DSPy-Multi-Document-Agent** setup.
   - This inquiry suggests that there might be confusion or lack of clarity in the provided resources.


  

---


### **DSPy ▷ #[colbert](https://discord.com/channels/1161519468141355160/1250300504462856265/1270828835977433088)** (3 messages): 

> - `qdrant_dspy`
> - `ColBERT and FastEmbed` 


- **Exploring qdrant_dspy GitHub Repository**: A member shared a link to the [qdrant_dspy GitHub repository](https://github.com/vardhanam/qdrant_dspy), focusing on designing a RAG pipeline using **Gemma-2b**, **DSPy**, and **Qdrant**.
   - The repository highlights the integration of these technologies, with an active focus on building efficient retrieval-augmented generation systems.
- **DSPy Framework Resource Shared**: Another member provided a link to [dspy/retrieve/qdrant_rm.py](https://github.com/stanfordnlp/dspy/blob/main/dspy/retrieve/qdrant_rm.py), which is part of the **DSPy** framework for programming foundation models.
   - This resource emphasizes the versatility and capabilities of DSPy, enhancing the understanding of local VectorDB interactions.
- **Interest in STORM with ColBERT and FastEmbed**: A member expressed a desire to run **STORM** while utilizing **ColBERT** and **FastEmbed** for searching over a local VectorDB.
   - This approach reflects the growing interest in combining multiple advanced technologies to optimize local vector searching tasks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/stanfordnlp/dspy/blob/main/dspy/retrieve/qdrant_rm.py">dspy/dspy/retrieve/qdrant_rm.py at main · stanfordnlp/dspy</a>: DSPy: The framework for programming—not prompting—foundation models - stanfordnlp/dspy</li><li><a href="https://github.com/vardhanam/qdrant_dspy">GitHub - vardhanam/qdrant_dspy: Designing a RAG pipeline using Gemma-2b, DSPy, and Qdrant</a>: Designing a RAG pipeline using Gemma-2b, DSPy, and Qdrant - vardhanam/qdrant_dspy
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1270823029110210712)** (6 messages): 

> - `DEBUG environment variable issue`
> - `Tinygrad Tensor Puzzles`
> - `getenv function ValueError` 


- **ValueError in getenv function**: A user encountered a `ValueError` stating **'invalid literal for int() with base 10: WARN'** while importing, linked to an environment variable.
   - Another member suggested checking **environment variables** since something is set to **WARN**, which was confirmed by the user.
- **DEBUG environment variable set to WARN**: The user found that the **DEBUG** environment variable was set to **'WARN'**, which might have caused the issue with the getenv function.
   - They noted that their Python script works fine, implying it could be specific to the notebook environment.
- **Introduction of Tinygrad Tensor Puzzles**: A member introduced **Tinygrad Tensor Puzzles** as a collection of **21 fun and challenging puzzles** to master tensor libraries from first principles.
   - The project adapts **Sasha's PyTorch Tensor-Puzzles** to tinygrad and encourages both beginners and experienced users to contribute and author puzzles.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/obadakhalili/status/1821587868562940179">Tweet from Obada Khalili (@obadakhalili)</a>: Introducing @__tinygrad__ Tensor Puzzles 🎉 A collection of fun and challenging 21 tensor puzzles to master tensor libraries like tinygrad from first principles, without relying on magic functions or ...</li><li><a href="https://github.com/obadakhalili/tinygrad-tensor-puzzles">GitHub - obadakhalili/tinygrad-tensor-puzzles: Solve puzzles to improve your tinygrad skills</a>: Solve puzzles to improve your tinygrad skills. Contribute to obadakhalili/tinygrad-tensor-puzzles development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1270944437761085440)** (2 messages): 

> - `tinygrad notes`
> - `fine-tuning tutorials`
> - `computer algebra optimization` 


- **Exploring tinygrad with helpful notes**: A series of [tutorials/study-notes](https://mesozoic-egg.github.io/tinygrad-notes/) are shared that aim to help understand the internals of **tinygrad** and start contributing.
   - Additionally showcased are resources like the [quickstart guide](https://github.com/tinygrad/tinygrad/blob/master/docs/quickstart.md) which may not be beginner-friendly but provide great foundational insights.
- **Computer Algebra Notes for Optimization**: Recent updates include [computer algebra study notes](https://github.com/mesozoic-egg/computer-algebra-study-notes/tree/main) which, while not directly tied to tinygrad, relate significantly to its optimization processes.
   - The integration of computer algebra techniques could provide valuable perspectives for developers looking to enhance tinygrad performance.
- **Inquiry on Fine-Tuning Tutorials**: A member inquired about available **fine-tuning** tutorials, highlighting a need for resources in this specific area.
   - No specific tutorials were mentioned in the messages exchanged, indicating a potential gap in resources for users seeking this information.



**Link mentioned**: <a href="https://mesozoic-egg.github.io/tinygrad-notes/">Tutorials on Tinygrad</a>: Tutorials on tinygrad

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1270916275131912214)** (4 messages): 

> - `Open Source Vision Models`
> - `MiniCPM-V 2.6 Performance` 


- **Queries for Open Source Vision Models**: Members are seeking recommendations for open source models suited for **vision tasks**, asking for both local and API options.
   - *One member expressed curiosity alongside the initial question.*
- **MiniCPM-V 2.6 Surpasses Competitors**: A member noted that the **MiniCPM-V 2.6** model outperforms **Gemini 1.5 Pro**, **GPT-4V**, and **Claude 3.5 Sonnet** in multi-image applications, sharing links to its resources.
   - They provided a link to the [Hugging Face page](https://huggingface.co/openbmb/MiniCPM-V-2_6) and the [GitHub repo](https://github.com/OpenBMB/MiniCPM-V) for further exploration.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/monkey-monkey-eating-monkey-eating-strawberries-kardie-gif-gif-22488578">Monkey Monkey Eating GIF - Monkey Monkey Eating Monkey Eating Strawberries - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/openbmb/MiniCPM-V-2_6">openbmb/MiniCPM-V-2_6 · Hugging Face</a>: no description found</li><li><a href="https://github.com/OpenBMB/MiniCPM-V">GitHub - OpenBMB/MiniCPM-V: MiniCPM-V 2.6: A GPT-4V Level MLLM for Single Image, Multi Image and Video on Your Phone</a>: MiniCPM-V 2.6: A GPT-4V Level MLLM for Single Image, Multi Image and Video on Your Phone - OpenBMB/MiniCPM-V
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1271131103822086248)** (2 messages): 

> - `Shipping updates` 


- **Inquiry on Shipping Updates**: A member expressed interest in any **updates on shipping**.
   - No specific replies were provided, but a link to a relevant Discord thread was shared for further reference.
- **Mikebirdtech responds**: A member shared a link to a Discord channel where shipping updates might be discussed.
   - The link directs to [this Discord message](https://discord.com/channels/1146610656779440188/1194880263122075688/1266055462063964191) for additional context.


  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1271173919029727252)** (3 messages): 

> - `Llama Team Q&A`
> - `Poe Hackathon` 


- **Llama Team answers questions on arXiv**: If anyone is interested, the **Llama team** is answering questions on the [arXiv discussion forum](https://alphaxiv.org/abs/2407.21783v1).
   - This is an opportunity for technical queries and insights directly from the team.
- **Quora hosts a hackathon for bot development**: Quora, which builds **Poe**, is running an in-person and virtual [hackathon](https://x.com/poe_platform/status/1820843642782966103) focused on building bots using the new **Previews feature**.
   - Participants will create innovative in-chat generative UI experiences with the latest LLMs like **GPT-4o** and **Llama 3.1 405B**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/poe_platform/status/1820843642782966103">Tweet from Poe (@poe_platform)</a>: We’re excited to announce a one-day hackathon with @agihouse_org around our new Previews feature! Compete to create the most innovative and useful in-chat generative UI experiences using the latest LL...</li><li><a href="https://alphaxiv.org/abs/2407.21783v1">The Llama 3 Herd of Models | alphaXiv</a>: Modern artificial intelligence (AI) systems are powered by foundation models. This paper presents a new set of foundation models, called Llama 3. It is a herd of language models that natively support ...</li><li><a href="https://x.co">Sell Domains | Buy Domains | Park Domains</a>: no description found
</li>
</ul>

</div>
  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1271024432315367445)** (2 messages): 

> - `General AI vs Non-General AI`
> - `Types of AI Applications` 


- **Exploring AI Beyond Generative**: A member initiated a discussion expressing their appreciation for **AI** that isn't generative, inviting others to share their thoughts.
   - *What kinds of AI applications do you have in mind?*
- **Diverse AI Applications Suggested**: Another member responded with suggestions including **computer vision**, **forecasting**, **recommendation systems**, and **NLP** as types of AI that are not generative.
   - These applications highlight the **broad spectrum** of AI technologies that extend beyond generative capabilities.


  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1270823164359868490)** (3 messages): 

> - `Vercel outage`
> - `Anthropic error rates` 


- **Vercel experiences intermittent outages**: Vercel is currently facing an outage, impacting the OpenRouter service, as noted in their [status update](https://x.com/OpenRouterAI/status/1821267624228966781). After several updates, services were reported stable again at 3:45pm ET.
   - Monitoring continues as Vercel implements fixes and will keep the [Vercel status page](https://www.vercel-status.com/) updated.
- **Anthropic tackles high upstream error rates**: Anthropic reported elevated error rates affecting their services, particularly on 3.5 Sonnet and 3 Opus, and has implemented a mitigation and a workaround. As of Aug 8, 17:29 PDT, success rates have returned to normal levels, and access for Claude.ai free users has been restored.
   - They are closely monitoring the situation and continuing to provide updates as issues are resolved.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://status.anthropic.com/">Anthropic Status</a>: no description found</li><li><a href="https://x.com/OpenRouterAI/status/1821267624228966781">Tweet from OpenRouter (@OpenRouterAI)</a>: Notice: we&#39;re seeing downtime due to a @Vercel platform outage, which doesn&#39;t appear in their status page.  Our status will be visible on https://status.openrouter.ai/
</li>
</ul>

</div>
  

---



---



---



---



---



{% else %}


> The full channel by channel breakdowns have been truncated for email. 
> 
> If you want the full breakdown, please visit the web version of this email: [{{ email.subject }}]({{ email_url }})!
>
> If you enjoyed AInews, please [share with a friend](https://buttondown.email/ainews)! Thanks in advance!

{% endif %}
