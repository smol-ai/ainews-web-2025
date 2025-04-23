---
id: fd9ea842-feac-4042-b9a3-ffbdf7a49551
title: not much happened today
date: '2024-09-27T21:53:11.435082Z'
original_slug: ainews-not-much-happened-today-1696
description: >-
  Meta released Llama 3.2, introducing lightweight 1B and 3B models optimized
  for on-device AI applications, enabling personalized and privacy-preserving
  functionalities. Additionally, Google DeepMind announced AlphaChip, an
  AI-driven method that revolutionizes microchip design by creating a feedback
  loop between chip development and AI model training.
tags:
  - meta
  - llama-3.2
  - on-device
  - google
  - deepmind
  - alphachip
  - chip-design
---


<!-- buttondown-editor-mode: plaintext -->**a quiet day is all you need**

> AI News for 9/26/2024-9/27/2024. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **31** Discords (**224** channels, and **2635** messages) for you. Estimated reading time saved (at 200wpm): **288 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Just a lot of non-headline news today:

- [GDM announced AlphaChip](https://deepmind.google/discover/blog/how-alphachip-transformed-computer-chip-design/)
- [FTC crackdown on deceptive AI claims](https://www.ftc.gov/news-events/news/press-releases/2024/09/ftc-announces-crackdown-deceptive-ai-claims-schemes)
- [Copilot now on GitHub.com in browser](https://x.com/ashtom/status/1839393494366138530)
- [Looots of reporting on OpenAI drama](https://x.com/garrisonlovely/status/1839655744850772272?s=46)
- [GGML starting to monetize thru HuggingFace](https://x.com/ggerganov/status/1839703977073487993)

You could tune in to the latest [Latent Space with Shunyu Yao and Harrison Chase](https://www.latent.space/p/shunyu) while you browse the news below!

If you are in SF for DevDay, consider bringing your demos and hot takes to [our DevDay pregame](https://lu.ma/devday-pregame) on Monday.

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

**AI Model Releases and Developments**

- **Llama 3.2 Release**: Meta released Llama 3.2, including lightweight 1B and 3B models for on-device AI applications. [@AIatMeta](https://twitter.com/AIatMeta/status/1839365639687086308) noted these models enable developers to build personalized, on-device agentic applications with capabilities like summarization, tool use, and RAG where data never leaves the device. [@awnihannun](https://twitter.com/awnihannun/status/1839330067039887622) demonstrated Llama 3.2 1B in 4-bit running at ~60 tokens/sec on an iPhone 15 pro.

- **Molmo Multimodal Model**: A new multimodal model called Molmo was released, with [@osanseviero](https://twitter.com/osanseviero/status/1839398112701386912) highlighting its data pipeline and training process. The model uses a dense captioning dataset of 712k images/1.3M captions and various datasets for supervised fine-tuning.

- **AlphaChip**: Google DeepMind announced AlphaChip, an AI method for chip design. [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1839306984480231852) stated it has transformed the way they design microchips, from TPUs for AI models to CPUs in data centers. [@demishassabis](https://twitter.com/demishassabis/status/1839354651206160563) noted the feedback loop where AlphaChip is used to design better AI chips, which are then used to train better models.

**AI Infrastructure and Platforms**

- **Hugging Face Milestone**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1839375655688884305) announced that Hugging Face crossed 1,000,000 free public models, emphasizing the importance of smaller, specialized models for specific use cases.

- **RAG Applications**: [@svpino](https://twitter.com/svpino/status/1839364380947054596) discussed challenges with scaling RAG applications, noting that more data can make them worse due to limitations of vector similarity search. He highlighted research showing accuracy degradation as knowledge bases grow.

- **On-Device AI**: Several tweets discussed the potential of on-device AI, with [@cognitivecompai](https://twitter.com/cognitivecompai/status/1839448460619128962) predicting that in another year, ChatGPT-level AI will be running on mobile/embedded devices.

**AI Research and Benchmarks**

- **Reliability in LLMs**: [@omarsar0](https://twitter.com/omarsar0/status/1839332359554163127) shared insights from a Nature paper suggesting that larger and more instructable LLMs may become less reliable, with issues in difficulty concordance, task avoidance, and prompting stability.

- **Elo Benchmarking**: [@sarahookr](https://twitter.com/sarahookr/status/1839399320048763247) announced the acceptance of work on Elo benchmarking in NLP at NeurIPS 2024, addressing reliability issues in this widely used evaluation method.

**AI Ethics and Regulation**

- **Free Speech and AI**: [@ylecun](https://twitter.com/ylecun/status/1839402554809373144) emphasized responsible use of free speech, warning about potential legal consequences for spreading harmful conspiracy theories.

- **AI Regulation**: Several tweets discussed SB-1047, a bill in California that could impact open-source AI development. [@ylecun](https://twitter.com/ylecun/status/1839398310899339699) expressed hope that Governor Gavin Newsom would veto it.

**AI Development Tools and Techniques**

- **PyTorch Optimization**: [@cHHillee](https://twitter.com/cHHillee/status/1839421129682997723) discussed performance improvements in PyTorch for reinforcement learning workloads using cudagraphs and torch.compile, reporting >5x speedups.

- **Web Scraping**: [@AlphaSignalAI](https://twitter.com/AlphaSignalAI/status/1839348216317505735) shared a GitHub repo for easily scraping web pages and outputting in LLM-friendly formats like JSON, cleaned HTML, and markdown.


---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Llama 3.2: Performance Gains and EU Regulatory Challenges**

- **Llama 3.2 Vision Models image pixel limits** ([Score: 40, Comments: 3](https://reddit.com//r/LocalLLaMA/comments/1fqawht/llama_32_vision_models_image_pixel_limits/)): The new **Llama 3.2 Vision Models** have a maximum image size of **1120x1120 pixels** for both the **11B** and **90B** versions, with a **2048** token output limit and **128k** context length. These models support **gif, jpeg, png, and webp** image file types, information which was not readily available in official documentation and required extensive testing to determine.
  - The **maximum image size** for Llama 3.2 Vision Models is actually **4 560x560 images**, as revealed in the [preprocessor config on Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct/blob/main/preprocessor_config.json). This configuration specifies **"max_image_tiles": 4** and image dimensions of **560x560**.
- Users appreciated the information provided about the model's capabilities, noting its usefulness for practical applications.
- **[Running Llama 3.2 on Android via ChatterUI](https://v.redd.it/gqbakkmtc6rd1)** ([Score: 39, Comments: 9](https://reddit.com//r/LocalLLaMA/comments/1fpze6d/running_llama_32_on_android_via_chatterui/)): The post announces the release of **ChatterUI v0.8.0-beta3**, which now supports running **Llama 3.2** models on **Android** devices. Using a **Snapdragon 7 Gen 2** processor, the app achieves **50 tokens per second** for prompt processing and **10 tokens per second** for text generation, demonstrating good performance on modern Android hardware. The author provides a [link to the beta release](https://github.com/Vali-98/ChatterUI/releases/tag/v0.8.0-beta3) and invites feedback, particularly on character list and chat history changes.
  - Users expressed interest in **larger models** for mobile devices, with one user finding the **Llama release disappointing** compared to bigger models they're already running.
  - There's interest in an **iOS version** of ChatterUI, but the developer cited the cost of a **Mac** as a barrier to publishing on the **App Store**.
  - The app's performance on **Android** devices with **Llama 3.2** models was noted, achieving **50 tokens per second** for prompt processing and **10 tokens per second** for generation.
- **Is Llama 3.2 Banned to Use in EU?** ([Score: 71, Comments: 132](https://reddit.com//r/LocalLLaMA/comments/1fqhjs9/is_llama_32_banned_to_use_in_eu/)): The **Llama 3.2** license on Huggingface reportedly **restricts license rights for EU-based individuals and companies** to use multimodal models, though this restriction is **not present** in the GitHub license. This discrepancy raises questions about potential **data collection and user fingerprinting** in new Llama multimodal versions, possibly in response to **EU data protection laws**.
  - The **EU AI Act** and **GDPR** are cited as reasons for **Meta's** restrictions on **Llama 3.2** in the EU, with concerns about training on personal data without consent. The **AI Act's implementation** starts in **February 2025**, raising questions about Meta's preemptive actions.
  - Discussions centered on the implications of **EU regulations** for AI models, particularly regarding **biometric categorization** and **copyright issues**. Some users expressed frustration with EU regulations, while others defended their importance for data protection.
  - There's debate about whether running AI models **locally** exempts them from EU regulations. The **"household exemption"** in GDPR was mentioned, but uncertainty remains about how regulators and courts will interpret these laws for open-source AI models.


**Theme 2. Next-Gen Hardware for AI: NVIDIA RTX 5090 Specs Leaked**

- **[RTX 5090 will feature 32GB of GDDR7 (1568 GB/s) memory](https://videocardz.com/newz/nvidia-geforce-rtx-5090-and-rtx-5080-specs-leaked)** ([Score: 87, Comments: 40](https://reddit.com//r/LocalLLaMA/comments/1fq2aad/rtx_5090_will_feature_32gb_of_gddr7_1568_gbs/)): The **RTX 5090** is rumored to feature **32GB of GDDR7 memory** with a bandwidth of **1568 GB/s**. This represents a significant upgrade from the current generation, potentially offering substantial performance improvements for AI and graphics-intensive applications.
  - Pricing discussions dominate, with users speculating the **RTX 5090** could cost **$3500** or even **$5090**. Some hope for cheaper previous-gen cards, but **3090s** prices have remained steady or increased in some regions.
  - The card's **600W power consumption** raises concerns about power limits. Users debate the significance of the **32GB memory upgrade**, with some calling it "huge" while others argue it's insufficient after three generations of 24GB.
  - Memory bandwidth calculations were scrutinized, with users suggesting the correct figure should be **1792 GB/s** instead of 1568 GB/s. The potential for running **70B models** and possibly the **90B Llama 3.2** on a single card was noted.


**Theme 3. Quantization and Performance Analysis of Large Language Models**

- **Estimating Performance Loss: Qwen2.5 32B Q4_K_M vs BF16 MMLU PRO evaluation results** ([Score: 79, Comments: 15](https://reddit.com//r/LocalLLaMA/comments/1fps3vh/estimating_performance_loss_qwen25_32b_q4_k_m_vs/)): The post compares the performance of **Qwen2.5 32B** model in **Q4_K_M** quantization against its **BF16** version using an incomplete **MMLU PRO** evaluation. Despite the limitations of an incomplete dataset, the comparison provides a rough estimate of performance degradation due to quantization, showing results across various subject categories and an overall performance drop from **66.58%** (BF16) to **64.23%** (Q4_K_M). The evaluation was conducted using **Ollama** as the backend and a GitHub-hosted evaluation tool, with specific configuration details provided.
  - **Qwen2.5 32B** model's performance on the **MMLU-Pro leaderboard** was discussed, with users noting its close performance to the 72B version. The leaderboard allows **self-reported results** via JSON uploads, raising questions about submission sources.
  - Users expressed interest in comparing **Q4_K_M** quantization to other formats like **IQ4_XS / NL** with suitable calibration data. Some suggested creating a sorted bar plot to better visualize performance differences between quantizations.
  - The **Q4_K_M** quantization showed unexpected improvements in certain categories like history, which was attributed to potential "lucky dice rolls" in quantization. Users also discussed the minimal performance loss compared to **BF16**, considering it a valuable trade-off for reduced resource requirements.
- **Running inference on the new Llama 3.2 1B model at 21 tok/s on an 8-core laptop with Rust** ([Score: 58, Comments: 8](https://reddit.com//r/LocalLLaMA/comments/1fqb0zd/running_inference_on_the_new_llama_32_1b_model_at/)): The author extended their **Rust-based project** to support inference on the new **Llama 3.2 1B and 3B models**, achieving **21 tokens per second** on an **8-core laptop** without using ML libraries. The project, available on [GitHub](https://github.com/samuel-vitorino/lm.rs), now includes a **light WebUI** as an alternative to the terminal chat interface for local CPU inference.
  - Users praised the project's **performance**, comparing it to **iPhone** capabilities. The author emphasized the **learning experience** of building from scratch, describing it as a *"mix of pain and reward when you finally get it right"*.
  - Requests for a **Windows GUI chat executable** were discussed. The author acknowledged this as a requested feature and suggested adapting the backend to be compatible with existing **frontends** that support multiple operating systems.
  - Debate arose over using **web browsers** versus **native applications** for the GUI. Web browsers were criticized for high RAM consumption and lower CPU/GPU performance compared to native apps.


**Theme 4. Advancements in Creative Writing and Roleplay AI Models**

- **[This is the model some of you have been waiting for - Mistral-Small-22B-ArliAI-RPMax-v1.1](https://huggingface.co/ArliAI/Mistral-Small-22B-ArliAI-RPMax-v1.1)** ([Score: 36, Comments: 22](https://reddit.com//r/LocalLLaMA/comments/1fpvj0o/this_is_the_model_some_of_you_have_been_waiting/)): **Mistral-Small-22B-ArliAI-RPMax-v1.1**, a new AI model for creative writing and roleplay, has been released. This model is based on **Mistral's 22B parameter** foundation and is designed to excel at **character-based interactions**, offering improved coherence and creativity compared to previous versions.
  - The **Mistral Small 22B ArliAI RPMax v1.1** model achieved a training and eval loss below 1.0, surpassing the **Llama 3.1 70B** version. This performance suggests the model may excel at creative writing and roleplay tasks despite its smaller size.
  - The **RPMax dataset** is curated to eliminate repetitions and synthetic generations, focusing on quality over quantity. The training approach uses a single epoch, low gradient accumulation, and higher learning rate to prevent overfitting to specific character tropes or stories.
  - Users expressed interest in the model's performance for short story writing and requested **public release of the dataset**. Some inquired about **VRAM requirements** and **EXL2 quantization** options for running the model on systems with limited resources.

- **Abliteration doesn't only effect how the model behaves and responds, but also effects how its fictional written characters think and respond as well** ([Score: 58, Comments: 15](https://reddit.com//r/LocalLLaMA/comments/1fqfmu6/abliteration_doesnt_only_effect_how_the_model/)): The post discusses an unexpected consequence of **"abliteration"** on **AI language models**, noting that it not only affects the model's direct responses but also influences the behavior of **fictional characters** created by the model. The author observes that **abliterated models** tend to produce characters who react more **positively and agreeably** in situations where they would typically display anger, defiance, or upset, effectively removing refusal behaviors from both the model and its fictional creations.
  - Users tested **abliterated models** with **system prompts**, finding they can still be steered to refuse requests. Some argue these models are better suited as **work tools**, particularly in fields like **healthcare** where compliance is crucial.
  - The impact of **abliteration** varies depending on the extent applied. Some models, like **Gemma 2 9b**, showed unexpected behaviors (e.g., "homicidal bias") even when vanilla. The [EQ Bench creative writing table](https://eqbench.com/creative_writing.html) suggests **Gemma2 finetunes** perform well in this area.
  - Some users noted that **abliterated models** may still have censorship, but express it through misunderstanding or reinterpretation of requests. This behavior might extend to roleplay contexts, affecting how fictional characters respond.


**Theme 5. Hugging Face Milestone: 1 Million Models**

- **Hugging Face just passed 1,000,000 models** ([Score: 167, Comments: 18](https://reddit.com//r/LocalLLaMA/comments/1fpx9ve/hugging_face_just_passed_1000000_models/)): **Hugging Face** has reached a significant milestone, surpassing **1,000,000 models** available on their platform. This achievement was announced by **Julian Bilcke** on X (formerly Twitter) and can be verified on the [Hugging Face models page](https://huggingface.co/models), showcasing the platform's extensive collection of machine learning models.
  - **Duplicate models** are prevalent on Hugging Face, with users noting multiple uploads of the same model (e.g., Llama-3.2-1B-Instruct.Q4_K_M.gguf) and questionable fine-tuning claims. **SomeOddCodeGuy** mentioned seeing "**5-15 q4 or q5 gguf repos**" for older models.
  - Users discussed the potential for **evolutionary AI development**, with **balcell** suggesting treating weights as DNA and introducing genetic algorithm properties. An example of a successful small-scale evolutionary simulation was shared by **involviert**.
  - Concerns were raised about model quality and functionality, with **remyxai** noting that "**half the time there is no model card**" when querying the hub APIs. Others questioned how many models actually perform their intended functions.

## Other AI Subreddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI Research and Model Developments**

- **Google DeepMind's AlphaChip transforms microchip design**: [Google DeepMind announced](https://www.reddit.com/r/singularity/comments/1fpx6sh/google_deepmind_our_ai_for_chip_design_method/) that their AI-powered chip design method AlphaChip has significantly improved the process of designing microchips. This advancement could accelerate AI hardware development.

- **New "blueberry" image generation model emerges**: A [mysterious new image generation model called "blueberry"](https://www.reddit.com/r/singularity/comments/1fpwuu7/a_new_mysterious_image_gen_model_called_blueberry/) has appeared on leaderboards, outperforming existing models like FLUX.1. Its origin is unknown but some speculate it could be from OpenAI.

- **Google's NotebookLM adds audio and video input**: [Google's NotebookLM tool now allows users to submit YouTube videos and audio files](https://www.reddit.com/r/singularity/comments/1fq02im/notebooklm_now_allows_submitting_youtube_videos/) as knowledge sources, expanding its multimodal capabilities.

**AI Industry and Company News**

- **OpenAI leadership changes**: Several [high-profile departures from OpenAI](https://www.reddit.com/r/OpenAI/comments/1fpt5gy/one_left/) have occurred recently, including Mira Murati, Bob McGrew, and Barret Zoph. This has sparked discussion about potential internal issues at the company.

- **OpenAI plans massive data centers**: OpenAI has [asked the US government to approve 5GW data centers](https://www.reddit.com/r/singularity/comments/1fpx4ml/openai_asked_us_to_approve_energyguzzling_5gw/), highlighting the enormous computing power needed for advanced AI development.

- **Sam Altman pushes for rapid breakthroughs**: Reports indicate Sam Altman is [pressuring OpenAI employees to quickly turn research breakthroughs into public releases](https://www.reddit.com/r/singularity/comments/1fq93b6/sam_altman_says_in_the_next_couple_of_years_we/), potentially accelerating AI progress.

**AI Policy and Societal Impact**

- **UN prioritizes AI governance**: The [United Nations is calling for AI to be treated with the same urgency as climate change](https://www.reddit.com/r/singularity/comments/1fq3811/the_united_nations_wants_to_treat_ai_with_the/), signaling growing global concern about AI's societal impact.

- **US government forms AI infrastructure task force**: The Biden administration has [created a task force to coordinate policy on AI data center infrastructure](https://www.reddit.com/r/singularity/comments/1fpx4ml/openai_asked_us_to_approve_energyguzzling_5gw/), demonstrating increased government involvement in AI development.

**AI Model Releases and Improvements**

- **Flux.1 Dev adds ControlNet Outpainting**: The [Flux.1 Dev model now supports ControlNet Outpainting in ComfyUI](https://www.reddit.com/r/StableDiffusion/comments/1fq1wfa/flux1_dev_controlnet_outpainting_comfyui/), expanding its image generation capabilities.

- **Elektroschutz LoRA released**: A new [Stable Diffusion LoRA called Elektroschutz](https://www.reddit.com/r/StableDiffusion/comments/1fqgo3l/elektroschutz_styled_warnings_nobody_asked_for/) has been released, demonstrating continued innovation in open-source AI models.


---

# AI Discord Recap

> A summary of Summaries of Summaries by O1-mini

**Theme 1. Language Model Performance and New Releases**

- [**ColQwen2 Dominates Vidore Leaderboard**](https://x.com/manuelfaysse/status/1839657285053788483): The **ColQwen2** model, powered by a **Qwen2-VL backbone**, achieves a remarkable **+5.1 nDCG@5** score, surpassing **colpali-v1.1** on the Vidore Leaderboard.
  
- [**Phi-3.5's Censorship Sparks Community Debate**](https://huggingface.co/SicariusSicariiStuff/Phi-3.5-mini-instruct_Uncensored): **Microsoft's Phi-3.5** model is criticized for its extensive censorship, leading users to explore an [**uncensored version**](https://huggingface.co/SicariusSicariiStuff/Phi-3.5-mini-instruct_Uncensored) on **Hugging Face**.

- [**Llama 3.2 Enhances Vision and Token Handling**](https://github.com/ggerganov/llama.cpp/issues/8010): **Llama 3.2 11B Vision** model now supports up to **128k tokens** and introduces improved vision features, though performance benchmarks show mixed results.

**Theme 2. Tooling, Integrations and New Features**

- [**Aider Launches Architect/Editor Mode for Efficient Coding**](https://github.com/paul-gauthier/aider/blob/main/aider/website/_posts/2024-09-26-architect.md): The new **architect/editor mode** in **Aider** streamlines coding workflows, enabling faster bug fixes with models like **o1-preview** and **Claude 3.5**.

- [**OpenInterpreter Debuts Electron Frontend**](https://x.com/parrotexplore/status/1839721139515302137): **OpenInterpreter** unveils an **Electron frontend**, enhancing user experience and fostering greater community engagement.

- [**LangChain Integrates Langfuse and PostHog for MistralAI Tracking**](https://t.co/KGxjjoO0vM): A [**tutorial**](https://t.co/KGxjjoO0vM) demonstrates setting up **Langfuse** within **LangChain** for comprehensive **LLM application monitoring** and **user analytics** via **PostHog**.

**Theme 3. Hardware and GPU Performance in AI Workloads**

- [**NVIDIA RTX 5090 Rumored to Feature 32GB VRAM**](https://wccftech.com/nvidia-geforce-rtx-5090-32-gb-rtx-5080-16-gb-specs): Speculations suggest the upcoming **NVIDIA RTX 5090** will include a **32GB VRAM** variant, while the **RTX 5080** might receive a **24GB** upgrade post its initial **16GB** release.

- [**TensorWave Offers MI300X GPUs to Community**](https://github.com/NVIDIA/TransformerEngine/pull/1019): **Darrick from TensorWave** announces the availability of **MI300X** units for community members, aiming to boost **GPU adoption** and **educational initiatives**.

- [**AMD GPUs Underperform in AI Benchmarks**](https://www.tomshardware.com/pc-components/gpus/stable-diffusion-benchmarks): **AMD GPUs**, such as the **5700 XT** and **7900 XTX**, are reported to lag behind **NVIDIA 3070** in productivity tasks like **Stable Diffusion** and **Blender**, highlighting performance discrepancies.

**Theme 4. Deployment Updates and API Enhancements**

- [**Cohere Releases API v2 with Enhanced Chat Capabilities**](https://docs.cohere.com/reference/chat-v2): **Cohere's API v2** introduces new endpoints like **v2/chat** with features including a `messages` parameter and **system message support**, enhancing **chat interactions**.

- [**OpenRouter Shifts to Token-Based Pricing for Gemini Models**](https://x.com/OpenRouterAI/status/1839738812877918617): **OpenRouter** transitions to counting **tokens** instead of characters for **Gemini** models, adjusting pricing to offer an estimated **50% cost reduction** for **Flash** and **1.5 Pro** models.

- [**Meta's Orion AR Glasses Integrated into Perplexity AI**](https://www.perplexity.ai/search/city-with-the-most-bike-lanes-hhNCIS6oRRCli0fdq8Z32g): **Meta's Orion AR Glasses** are incorporated into **Perplexity AI**, aiming to revolutionize user interactions within **augmented reality** environments.

**Theme 5. Model Training and Optimization Techniques**

- [**DSPy Integrates with Langtrace for Advanced Experiment Management**](https://docs.langtrace.ai/supported-integrations/llm-frameworks/dspy): **DSPy** now supports **Langtrace**, enabling automatic capture of **traces**, **checkpoints**, and **evaluation score visualizations**, significantly enhancing **AI experiment workflows**.

- [**Fine-Tuning Llama Models Raises Overfitting Concerns**](https://github.com/unslothai/unsloth/issues/1040): Users report challenges with **fine-tuning Llama 3.2-3B**, highlighting risks of **overfitting** with low training losses and emphasizing the need for proper **data handling** and **tokenizer adjustments**.

- [**LoRA+ Optimizations Improve Model Training Efficiency**](https://github.com/axolotl-ai-cloud/axolotl/pull/1932): **LoRA+** optimization parameters are updated to fix default learning rate issues, enhancing the efficiency and stability of **model training** processes.

---

# PART 1: High level Discord summaries

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Architect/Editor Mode Streamlines Coding**: The new **architect/editor mode** in Aider enhances coding workflows, promoting faster bug fixes with models like **o1-preview** and **Claude 3.5**.
   - Users suggest leveraging **Sonnet 3.5** for design tasks to maximize efficiency.
- **Model Performance Benchmarking Encouraged**: Users are prompted to benchmark various model combinations like **o1-preview** with **o1-mini** and **Sonnet 3.5** to optimize performance.
   - Performance may vary depending on project size and editing context, suggesting that tailored setups deliver the best results.
- **New /copy Command Proposed**: A proposal for a new **/copy command** aims to let users easily copy the last LLM output into the clipboard for further use.
   - This feature enhances workflow, particularly for those utilizing the **/ask command** frequently.
- **Streamlit's Interactivity Limitations Discussed**: Members noted that **Streamlit** has limitations for Aider use cases, suggesting redesign necessities for improved interactivity.
   - While potential redesign was acknowledged, it's not currently deemed a priority by the group.
- **Observations on Token Usage**: Discussion centered around **token usage** in Aider, with advice to keep files minimized to 1-3 to avoid performance hits.
   - Members were recommended to use `/tokens` to monitor usage, as exceeding **30k tokens** could lead to unpredictable behavior.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Molmo's Compatibility with LM Studio**: New Vision models won't be supported in LM Studio for a while because they are incompatible with **llama.cpp**.
   - Users noted the **Llama 3.2 11b** is similar to the **3.1 8b** but adds parameters to enhance vision features.
- **Queries on Llama 3.2's Text Generation**: The community has raised questions about **Llama 3.2's** token support, with claims it can handle up to **128k tokens**.
   - Mixed reports have emerged about the model's performance and issues related to buggy integrations.
- **Upgrade Concerns for LM Studio**: Users expressed unease around upgrading from version **0.2.31** to **0.3.x** regarding model compatibility and retention of settings.
   - It was confirmed that transitioning to **0.3.x** would not lead to data loss, though it would replace previous versions.
- **NVIDIA GPU Rumors Heat Up**: Rumors indicate the upcoming **NVIDIA RTX 5090** might feature **32GB VRAM**, while the **RTX 5080** could have a **24GB** variant after its **16GB** launch.
   - Skepticism abounds regarding the **5080's** capabilities, with users claiming it's not equipped for current gaming and AI demands.
- **Load Testing LLM Performance Recommendations**: For effective load testing, users recommend employing local server API calls in **LM Studio** to manage multiple requests efficiently.
   - One member is creating a tutorial focused on these load testing methods, emphasizing the use of custom datasets.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Call to Form Group for Llama 3.2 Models**: A proposal has been made to create a working group to integrate **Llama 3.2 vision models** into **llama.cpp**, as discussed in [this GitHub issue](https://github.com/ggerganov/llama.cpp/issues/8010).
   - The issue notes that multimodal support can be reinstated once related components are refactored.
- **Interest in Optimizing Code for Cerebras Chips**: Discussions around optimizing code for **Cerebras chips** have highlighted the community's desire for insights on effective usage.
   - Members are intrigued about reaching out to connections at Cerebras for additional guidance on this hardware.
- **Seeking Latest Triton Wheel for Windows**: A member is searching for the latest compiled [Triton wheel for Windows](https://link.to.triton.windows) that works with Python 3.10, reflecting broader compatibility needs.
   - Community engagement around installation issues continues to be a focal point for Triton users on multiple platforms.
- **M2 Pro Benchmarks shared**: A member expressed enthusiasm over their **M2 Pro benchmarks** and cited [DiffusionKit](https://github.com/argmaxinc/DiffusionKit) for performing on-device inference of diffusion models.
   - They included visuals that reinforce the benchmark capabilities of the M2 Pro in a practical context.
- **TensorWave Offers MI300X to Boost Adoption**: Darrick from TensorWave announced potential availability of **MI300X** units for community members, aimed at enhancing education on its use.
   - This opportunity has sparked positive engagement, with members expressing excitement over the offer.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Fine-tuning Llama Models Spark Confusion**: Users discussed the nuances of fine-tuning **Llama models**, noting confusion over data formats like `chatml` and the necessity of adjusting tokenizer settings for special tokens.
   - Concerns arose regarding overfitting, with members warning against low training losses that signal potential model memorization traps.
- **Model Checkpoint Loading Errors Emerge**: A user encountered a data mismatch error while trying to load the `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` model, spotlighting specific exceptions.
   - This initiated troubleshooting discussions, with suggestions focusing on sizing and configuration settings as possible culprits.
- **Speculation Surrounds New Graphics Cards**: Community members debated the specs and rumored release of upcoming GPUs like the **5090**, projecting a likely **32GB VRAM** option despite skepticism.
   - Opinions varied widely, demonstrating that while rumors circulate, actual benchmarks are needed to settle disputes.
- **Data Packing Enhances Training Efficiency**: Members highlighted that packing data allows training frameworks to manage unrelated parts, streamlining the process and enabling predictions of subsequent tokens efficiently.
   - This technique was noted to significantly improve training dynamics through effective management of multiple examples.
- **Updates on Transformers and Model Compatibility**: Users confirmed having the latest **transformers** version (4.45.1) installed, suggesting ongoing efforts to refine their model implementations.
   - Discussion around quantization challenges, particularly with **Phi3.5**, showcased the need for alternative strategies due to fatal **vocab size mismatch** errors.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Challenges with Uncensored Models**: Users noted that certain **Hugging Face models** are censored, making it difficult to create a game bot using a 12B chat model, suggesting alternatives like **Venice.ai**.
   - This discussion emphasized the need for uncensored models for broader creative applications.
- **Neuralink's CUDA Implementation Explored**: One participant shared insights on the use of **CUDA** with **Neuralink** to enhance model performance in advanced GPU programming.
   - This has implications for improving execution efficiency across various AI applications.
- **Alibaba Introduces MIMO Technology**: **Alibaba** launched **MIMO**, a new AI capable of creating realistic character videos from simple inputs, showcased through **10 demos** including **Interactive Scene Control**.
   - This technology illustrates the potential for new immersive experiences in AI-generated content.
- **Seeking Repos for Text-to-Video Model Training**: A request was made for repositories focused on **distributed GPU training** for **text-to-video (T2V)** models, indicating a need for enhanced training resources.
   - Suggestions like checking out the [CogVideo SAT finetuning](https://github.com/THUDM/CogVideo/blob/main/sat/README.md) have been made to aid this pursuit.
- **Cybersecurity Services Offered by Expert Hacking**: An individual emerged as an expert hacker offering various **cybersecurity courses and services**, inviting collaboration in these areas.
   - This highlights an interesting intersection of AI and cybersecurity, which is increasingly relevant in today's tech landscape.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemini Tokens Count Change**: OpenRouter will transition to counting **tokens** instead of characters for **Gemini** models, decreasing token counts by about fourfold on the `/activity` page.
   - This adjustment leads to a pricing change where costs double, but offers an estimated **50%** cost reduction for Flash and 1.5 Pro models.
- **Llama 3.2 Vision Parameters Discussion**: Users questioned the parameters for **Llama 3.2 vision** to avoid rejection, particularly in attractiveness evaluations.
   - The consensus suggested safety-focused training could prevent the model from responding adequately to such queries.
- **Database Upgrade Downtime Annulled**: The planned downtime for a database upgrade was cancelled, allowing services to remain operational.
   - Further scheduling updates for the upgrade will be communicated once determined.
- **Chatroom UI Gets a Major Upgrade**: OpenRouter announced a revamped UI for the Chatroom that displays model responses with reasoning collapsed by default, enhancing clarity.
   - Further UI enhancements are promised, aiming for a better user interface experience.
- **OpenRouter Hits Rate Limits**: Users reported encountering a **429 Resource Exhausted** error, signaling the model can't process requests due to rate limit breaches.
   - Efforts to negotiate higher rate limits with Google are ongoing to alleviate these issues.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Channel Etiquette Clarified**: Miscommunication arose about posting in the wrong channel, prompting brief *clarification on channel appropriateness*. Some members still enjoyed non-Cohere content being shared.
   - A member expressed optimism for their project’s launch, thanking the community for helpful posting direction.
- **Embed-English-v3 Model Finetuning In Limbo**: Inquiries about finetuning the **embed-english-v3** model led to the realization that currently, **no embedder can be finetuned**.
   - Suggestions were made to utilize **custom embedding models** from Hugging Face for those needing specific adjustments.
- **API v2 Endpoints Make Their Debut**: New API **v2** endpoints have launched, notably enhancing **Chat V2** with new features like a `messages` parameter. More info can be found in the [API Reference](https://docs.cohere.com/reference/chat-v2).
   - Users discussed the implications of trial key rate limits, clarifying they're account-based, thus cutting down the benefits of rotating keys.
- **Cultural Multilingual LMM Benchmark Gathers Momentum**: The **MBZUAI** team is building a **Cultural Multilingual LMM Benchmark** covering **100 languages**, aiming to improve their multimodal dataset.
   - Volunteers aiding in translations will be invited as co-authors for a submission to **CVPR'2025**, creating a community-driven effort.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Tiled Upscale Offers Slower Alternative to ADetailer**: **Tiled Upscale** can replace **ADetailer** with similar effects, but it works about **50 times slower** as it processes the entire image.
   - This slower alternative raises questions about efficiency when detailed area-specific upscaling is needed.
- **AMD GPUs Struggle in Productivity**: Discussion explored how AMD GPUs, like the **5700 XT**, falter in **Stable Diffusion** and **Blender** tasks, proving more effective for gaming.
   - Users reported a **3070** outperforming a **7900 XTX** in productivity benchmarks, highlighting GPU performance discrepancies.
- **Refurbished GPUs Gaining Favor**: The advantages of choosing refurbished GPUs over used ones sparked lively debate, focusing on improved reliability through repairs and checks.
   - One user celebrated their experience with a refurbished **3090 TI**, emphasizing it performed nearly as well as a new card.
- **SSD Proven Essential for Load Times**: Confirmed findings suggest using an **SSD** for **Stable Diffusion** can cut model load times by **10x or more** compared to traditional HDDs.
   - Members noted that models running on **M.2 SSDs** greatly enhance image generation speed over older technologies.
- **Creative Prompting for Object Sizes**: Participants shared insights on effective prompting techniques for sizing objects in image generation, suggesting various descriptive terms.
   - Humorous phrases like *'yuge'* and *'bigly'* were jokingly proposed, though simpler terms were ultimately preferred.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity UI Issues Frustrate Users**: Multiple users faced errors on the Perplexity website, reporting that interactions resulted in `net::ERR_BLOCKED_BY_CLIENT` errors while the Android app remained functional.
   - This led to significant frustration among users, particularly as the issue persisted across both desktop and mobile browsers.
- **API Functionality Sparks Queries**: Users expressed a desire to access the latest news on generative AI through the Perplexity API, questioning current limitations on specific API functionalities.
   - Concerns were raised about the robustness of existing solutions and the need to explore improvements.
- **Subscription Promotions Cause Confusion**: Frustration mounted as a user struggled to redeem a promotional code for a pro subscription without gaining access, sparking further queries about account transfers.
   - Others chimed in to clarify the steps involved in transferring subscriptions.
- **Meta's Orion AR Glasses Enhance Experiences**: Meta's recent announcement about [Orion AR Glasses](https://www.perplexity.ai/search/city-with-the-most-bike-lanes-hhNCIS6oRRCli0fdq8Z32g) aims to revolutionize user interactions in augmented reality.
   - Initial feedback suggests the potential for significant shifts in how users engage in virtual environments.
- **OpenAI Shifts Toward For-Profit Future**: OpenAI's [for-profit pivot](https://www.perplexity.ai/search/what-happened-with-wordpress-w-V8a7N3D4QMqBc3vZdzXzVg) potentially reshapes its funding strategies amidst competitive pressures in AI.
   - This shift raises questions about the implications for its operational strategies going forward.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **GPU Memory Size Discrepancy Sparks Debate**: Discussions highlighted the differences in memory sizes between **5080** and **5070** GPUs, with the **5080** model suggested to have nearly **20GB**.
   - Members noted a trend of doubling memory sizes across generations, referencing the **3080** and **3090** models.
- **Buzz Builds for DisTrO Paper Release**: Curiosity surrounds the release date of the **DisTrO** paper, with members eager for insights, especially from a recent talk.
   - Helpful links to the full talk were shared after requests for easier access were made.
- **Knowledge Graphs and Bitcoin Ordinal Theory Converge**: A member discussed their work with **knowledge graphs** and unique embeddings derived from **Bitcoin Ordinal Theory**.
   - They proposed that LLMs form **graph-based representations** from semantic richness, hinting at possible avenues for emergent intelligence.
- **Claude Sonnet 3.5 Delivers Improved Reasoning**: Progress was noted in the reasoning capabilities of **Claude Sonnet 3.5**, attributed to utilizing example reasoning traces.
   - A standout example demonstrated improvements, indicating future directions for further exploration of reasoning enhancements.
- **Hermes Available for Local Run on 4090**: A member confirmed that **Hermes** can be run locally on a **4090 GPU** using **LMStudio**, which supports any **GGUF version**.
   - This allows users an easy way to find and utilize **Hermes** without needing API access.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Agentic Search Project Faces Budget Cuts**: A developer shared their **Agentic Search** project's failure due to costly compute and token usage, prompting them to consider fine-tuning a smaller model like **Llama 3b**.
   - This shift highlights the resource constraints larger models impose on development teams in the AI landscape.
- **AI Adoption in Academia Skyrockets**: Discussion revealed that over **50%** of master's students are using AI-generated content for assignments, stirring debates on productivity vs. academic integrity.
   - Participants expressed concern over the potential long-term impacts on learning as AI becomes more ingrained in educational settings.
- **AI's Energy Use Sparks Debate**: Questions surfaced regarding AI systems' **energy consumption**, highlighting increasing awareness of their environmental impact.
   - Members discussed the need for sustainable practices as AI technologies become more prevalent in various industries.
- **Game-Changing Tools for Developers**: A member recommended the **ChatGPT Toolbox** Chrome extension, featuring chat history search and prompt management to boost productivity with ChatGPT.
   - Attention also turned to the anticipated **Orion model**, expected to introduce powerful new tools that could revolutionize the development process.
- **Future Generations at Risk of Skill Loss**: Concerns emerged that future generations may lose traditional skills like writing by hand due to increasing technology reliance.
   - Participants humorously speculated on societal views of basic skills in a tech-dominated future, raising questions about the evolution of learning tools.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Exploring Sponsorship for Open Source Models**: A member inquired if Eleuther offers any sponsorship programs for open source models, expressing a lack of resources to fully train their projects.
   - This raises a discussion about community support for such initiatives within the open source realm.
- **Innovations in LLM Search Space Simulation**: A concept was proposed involving an abstract search space for LLMs, utilizing Monte Carlo tree search to simulate continuous thinking with text diffusion.
   - *This method aims to rank the most coherent thoughts during the computational process,* suggesting potential advancements in LLM architecture.
- **Comparing Weight Distributions Pre and Post FP6**: Discussion revolved around comparing weight distributions of models before and after **FP6**, with hints at using libraries like [seaborn](https://seaborn.pydata.org/) for visualization.
   - The goal is to see if any anomalies arise, as members suggested experimenting with multiple plotting libraries.
- **ColQwen2 Makes Waves**: A new model, **ColQwen2**, was announced as a top visual retriever, surpassing **colpali-v1.1** with a **+5.1 nDCG@5** score on the Vidore Leaderboard.
   - This model utilizes a **Qwen2-VL backbone**, promising superior performance in visual retrieval tasks, as noted in [this post](https://x.com/manuelfaysse/status/1839657285053788483).
- **Testing on H100s for Small Models**: A member expressed willingness to assist with testing on **H100s** for small models, indicating confidence in their ability to contribute.
   - This sparked enthusiasm and appreciation from others in the discussion.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Langtrace Enhances DSPy Experiment Management**: Langtrace now supports running DSPy experiments with automatic capture of **traces**, **checkpoints**, and **eval score visualizations**, significantly improving management workflows.
   - Users can create individual projects for each pipeline block, allowing for targeted optimizations and effortless deployment of checkpointed prompts.
- **MIPROv2 Compilation Runs Encounter Issues**: Users reported challenges in tracking evaluation data during MIPROv2 compilation runs despite visible traces in the logs, suggesting a configuration mishap.
   - Troubleshooting revealed the need for proper attributes during the `compile()` call to ensure accurate data tracking.
- **DSPy Optimization Tools Spark Discussion**: Members expressed curiosity about DSPy's optimization tools, similar to **Tensorboard**, for tracking metrics efficiently in AI workflows.
   - They shared insights about tools such as the [DSPy Visualizer](https://link.to.visualizer) and additional support available via Langtrace.
- **Exploring DSPy ReAct Agents for RAG**: Members inquired about examples of using **DSPy ReAct agents**, especially in conjunction with a **LlamaIndex retriever** for ReAct RAG implementations.
   - Other users pointed to existing examples in the **repo (examples/agents/)** and pledged to add more comprehensive examples soon.
- **Feature Requests for RAG Agents Optimization**: There were requests for integrating more **vector databases** like **Qdrant** and **LanceDB** with DSPy RAG agents, capturing a trend towards hybrid search capabilities.
   - The discussion about multimodal RAG pipeline optimization received confirmation of forthcoming developments in this area.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Poll on Mojo MAX Desktop Backgrounds**: A member initiated a poll for **Mojo / MAX branded desktop backgrounds** inviting votes featuring adorable Mojo flames and MAX astronauts.
   - Reaction was mixed, with one member simply stating, *'Bruh'*, indicating surprise or disinterest.
- **Verification Now Required for Posting**: Verification is now a necessity for posting in all channels except a few specific ones listed, enhancing control.
   - Members are directed to visit the verification channel, where a demo GIF explains the process.
- **Error Handling Needs in Mojo**: Members discussed how the current error messages in Mojo do not reference user code, which hampers debugging.
   - There's concern about improvements in this area due to the limitations of the existing implementation.
- **Proposing Safe Tagged Union for Variant Type**: A member proposed evolving the **Variant** type into a *safe* tagged union to enhance pattern matching capabilities.
   - The discourse centered around ensuring compatibility with existing models and expectations in pattern matching.
- **Call for Enhanced Mojo Documentation**: Members agreed on the urgent need for improved documentation on Mojo and MLIR dialects to clarify user ambiguities.
   - Confusion over existing constructs has hindered development, necessitating clearer guidelines.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **FTC Crackdown on Misleading AI Marketing**: The FTC initiated a crackdown on misleading claims related to AI tools, particularly affecting companies like **Do Not Pay**, cited in their [complaint PDF](https://www.ftc.gov/system/files/ftc_gov/pdf/DoNotPayInc-Complaint.pdf).
   - Concerns emerged regarding the FTC's definition of AI, with community members worried it may lead to scrutiny of many startups.
- **Sustainability of Generative AI Under Fire**: An article discussed the potentially unsustainable nature of the current generative AI boom, predicting a major collapse that might impact big tech, linked in their [newsletter](https://www.wheresyoured.at/subprimeai/?ref=ed-zitrons-wheres-your-ed-at-newsletter).
   - Critics argued that tools like **GitHub Copilot** showcase clear business value, which counters claims of unsustainability.
- **Geohot's AMD Discontent**: Geohot expressed dissatisfaction with AMD, questioning the company's innovative trajectory after noting no significant products post-RDNA3.
   - This frustration is symptomatic of a wider community concern regarding stagnation and motivation within AMD's technical advancements.
- **Launch of ColQwen2 Model**: The community cheered the introduction of the **ColQwen2** model, which integrates a **Qwen2-VL** backbone for enhanced performance and efficiency.
   - This launch marked a major improvement in visual recognition capabilities, celebrated for its significant impact on the Vidore Leaderboard.
- **AI Engineering Interviews Generate Excitement**: A member shared enthusiasm about securing an interview opportunity leading to a potential AI Engineering role.
   - *“Had an interview that could transition into an AI Engineering role so me be happy.”*



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Paragon Builds a Feature-Packed Chatbot**: A blog post and video from [useparagon](https://t.co/KEE2LOnGoR) illustrate their use of create-llama from LlamaIndex to create a chatbot interfacing with customer data from **Slack**, **Google Drive**, and **Notion**.
   - *It ingests data continuously and in real time,* making the integration highly effective.
- **Langfuse and PostHog Enhance MistralAI**: A tutorial shared in a [Jupyter notebook](https://t.co/KGxjjoO0vM) explains how to set up **Langfuse** for tracking LLM applications and integrates **PostHog** for user analytics.
   - This setup enables comprehensive **monitoring** and **analytics** for AI applications, streamlining the development process.
- **NLTK's punkt resource missing**: A user reported encountering a *Resource punkt not found* error while using **NLTK**. Another member suggested checking the version of **llama-index** as the latest versions utilize *punkt_tab*.
   - *Resource-related issues* with NLTK's punkt hinted at potential compatibility concerns.
- **Challenges Loading Fine-tuned Models**: A user struggled to load their locally fine-tuned **Llama3.1-8B** for the Text2SQL task onto their GPU. Members recommended manually loading the model and tokenizer, ensuring it was on the GPU.
   - A detailed code snippet was shared, illustrating how to set up the model using quantization for optimized performance.
- **Optimizing Vector Search for Customer Support**: A proposed strategy for optimizing vector search involved storing questions in the vector chunk while keeping answers in metadata. This method aimed to enhance accuracy by focusing on question semantics during searches.
   - The user sought validation and welcomed suggestions for further improvements to their approach.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI rushed GPT-4o release amid concerns**: Executives aimed to debut **GPT-4o** ahead of Google’s developer conference, resulting in a rushed release with **incomplete safety data** that later marked the model as too risky to deploy. Staff reportedly endured **20-hour days** to meet this tight deadline while managing safety evaluations.
   - An article by [Garrison Lovely](https://x.com/garrisonlovely/status/1839655744850772272?s=46) sheds light on the intense pressure faced by safety teams during this high-stakes launch.
- **OpenAI grapples with compensation demands**: As outlined in [The Information](https://www.theinformation.com/articles/behind-openais-staff-churn-turf-wars-burnout-compensation-demands), OpenAI faces ongoing employee grievances over compensation as its valuation skyrockets. Staff have cashed out **over $1.2 billion** from profit units, spurring researchers to threaten resignation amidst fierce competition for talent.
   - New CFO **Sarah Friar** now navigates this turbulent environment, where many researchers demand substantial pay increases to remain amid leadership turnover.
- **OpenAI's leadership instability**: The recent departures of key figures **Mira, Bob, and Barret** add to ongoing leadership instability at OpenAI, raising concerns over its long-term direction. The emotional response from team members reflects the broader challenges of retaining talent in a competitive landscape.
   - In promoting transparency, one intern humorously likened their resignation to experiencing the bittersweet nature of **cherishing a newborn**.
- **Substack taps into iPhone IAP subscriptions**: As a **Substack best seller**, there is newfound access to **iPhone In-App Purchase subscriptions**, indicating a shift toward digital publishing on mobile devices. This opens channels for content creators to monetize their works more effectively on popular platforms.
   - The implications for content creators in the mobile market are significant, paving the way for increased engagement and revenue opportunities.
- **Apple App Store management challenges revealed**: Members share captivating insights into the **Apple App Store**, often viewed as a **horror show** by app developers, discussing the complexities of its management. The conversation highlights the necessity for developers to navigate the challenging landscape created by App Store policies.
   - While the realities can be daunting, the discussion brings to light potential strategies that developers can employ to manage the intricate workings of their app distribution.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Open Source Community Lags in Multimodal Support**: A member highlighted that the **open-source community** is falling behind in adopting **multimodal support**, while the broader industry shifts in that direction.
   - This sentiment reflects a growing concern about the speed of **innovation** in the community.
- **Understanding Area Chair Roles**: A member explained that **AC** refers to a meta reviewer known as an **area chair**, who plays a critical role in the review process.
   - This insight underscores the importance of organization in academic and collaborative environments.
- **Python Snippet for Training Conversation Splitting**: A user presented a Python snippet aimed at **splitting conversations** for training purposes, ensuring conversations do not exceed **maximum sequence length**.
   - They emphasized its utility, particularly for handling long conversations while retaining context in training datasets.
- **Flex Attention for Optimization Discussion**: A member highlighted **Flex Attention** as a new optimized implementation that provides flexibility compared to previous attention methods.
   - Several resources were shared, including a [link to the PyTorch blog](https://pytorch.org/blog/flexattention/) detailing its design.
- **Update on LoRA+ Optimization Parameters**: A member requested the setting of `loraplus_lr_embedding` to a specific value, referencing a [fix in a recent GitHub PR](https://github.com/axolotl-ai-cloud/axolotl/pull/1932).
   - They explained that the fix was essential due to the failure to use a default value for this parameter.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **IOMMU's Role in Nvidia P2P**: A user inquired why **IOMMU** must be disabled for **Nvidia P2P** support when using the [tinygrad GPU modules](https://github.com/tinygrad/open-gpu-kernel-modules), signaling a need for further technical insight.
   - This uncertainty highlights an area ripe for discussion as users seek to clarify critical hardware interactions.
- **GPU Cloud Pricing Competition Sparks Discussion**: George Hotz suggested a competitive rate of **$0.50/hr for GPUs**, leading to comparisons with options from providers like salad.com and vast.ai.
   - Participants raised concerns about whether this price incorporates VAT and reflects true market competitiveness.
- **CLOUD=1 Features Debated**: Debates flared over whether **CLOUD=1** includes **CPU** resources; discomfort among users was expressed about mandatory device connectivity.
   - They emphasized that saving costs needs to be complemented by robust solutions to justify the service model.
- **Challenges with Data Upload for ML Tasks**: A member highlighted severe issues with connecting and uploading large datasets for training, hoping **tinygrad** could alleviate these frustrations.
   - The discussion noted that the **data-compute ratio** is crucial for efficiency, particularly in smaller models like **mini LLMs and CNNs**.
- **Considerations on Persistent Storage Costs**: Concerns emerged over **persistent storage billing**, with questions about whether tinygrad would address such charges, as many cloud providers have separate fees.
   - This points to a broader conversation on cost management in cloud service architecture.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Llama 3.2 11B Vision available for free**: TogetherCompute partnered with AIatMeta to offer **Llama 3.2 11B Vision** for free, allowing developers to experiment with open-source multimodal AI. Access this innovative tool [here](https://api.together.ai/playground/chat/meta-llama/Llama-Vision-Free).
   - For enhanced performance, paid Turbo endpoints for **Llama 3.2 11B & 90B** are also provided.
- **Unlimited Access Sparks Ideas**: Members discussed the implications of unlimited access to **Llama 3.2**, suggesting it might humorously caption the entire **LAION dataset**. This led to light-hearted community engagement around creative applications.
   - The playful conversation emphasized a collective enthusiasm for pushing the creative boundaries of AI tools.
- **Concerns Arise Over Family Photo Generation**: A member inquired about the effectiveness of a specific app for generating **family photos**, highlighting the keen interest in AI-driven personalized content. This discussion underscored the growing push for practical applications in daily life.
   - The inquiry reflects an ongoing curiosity about the capabilities of AI in generating relatable imagery.
- **Victory in Copyright Enforcement Celebrated**: A member shared a LinkedIn post celebrating a successful win in copyright enforcement, emphasizing that **the good guys won this round**. This was hailed as a significant victory for integrity within the community.
   - The sentiment contributed to a positive atmosphere, reaffirming the community's commitment to ethical practices.
- **Discussion on Positional Information in Neural Networks**: Members expressed confusion over how positional information integrates into the feature vector of latent pixels, noting the absence of positional encoding in CLIP text embeddings. They emphasized that self-attention steps in models also contribute to this process.
   - This led to constructive insight on the importance of convolution edges in yielding **positional data** for attention comparisons.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Lectures Set for LLM Safety Focus**: Concerns arose about lectures addressing **social alignment** in relation to LLM agents, given the previous focus on **AI safety**. Prof. Dawn Song is expected to touch on this during her talk scheduled for **December 2**.
   - This indicates an ongoing dialogue about balancing safety and alignment in educational content.
- **Course Sign-Up Process Confirmed**: Clarifications on course enrollment confirmed that filling out the Google form ensures access to all course materials with assignment deadlines noted as **December 12, 2024**. Participants expressed gratitude for this clear communication.
   - This highlights the importance of clarity in administrative processes for a smooth learning experience.
- **Confusion with Assignment Deadlines**: A participant questioned discrepancies in assignment due dates between Berkeley and MOOC students, with confirmations that all assignments are due on **December 12, 2024**. Provisions for uniform deadlines improve course accessibility.
   - It’s crucial for students to have clear timelines, as confusion can affect focus and performance.
- **Qquiz 3 Availability Muddled**: Participants struggled to locate **Qquiz 3**, prompting discussions about its accessibility, confirming that it remains live on the **MOOC students' website**. This has led to more inquiries regarding quiz structuring.
   - Ensuring all students can access quizzes is essential for fostering an equitable learning environment.
- **Lab Assignment Release Timeline Questioned**: A user inquired about the timeline for lab assignment releases, noticing a gap in information on the MOOC website. Continued discussions around course clarity remain key for students tracking assignments.
   - Effective communication about assignment schedules will enhance student engagement and preparedness.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **OpenInterpreter showcases on-chain analytics prowess**: A member demonstrated how to use [OpenInterpreter](https://x.com/parrotexplore/status/1839721139515302137) to transition from code that's **probably working** to **fully functional code** for on-chain analytics with a shared [Google Colab](https://t.ly/vBSPe) link.
   - This shift in approach was well-received, inviting further **reposts** from the community.
- **Multimodal Support Hiccups in LLaMA**: Discussion emerged around the removal of **multimodal support** in the LLaMA project since **#5882**, with updates contingent on the refactoring of **llava**.
   - A tracking thread was created, consolidating insights and links to relevant issues for any follow-up.
- **Electrifying Frontend Development Buzz**: Excitement grew over the development of an **Electron frontend** for OpenInterpreter, as a member highlighted its potential.
   - The enthusiasm reflects a positive sentiment about ongoing development within the **OpenInterpreter** community.
- **HF's Latest with 90b Vision Model**: **HF** announced an update introducing a **90b vision** model, now available for various vision tasks.
   - This update is expected to considerably enhance real-world applications in related tasks.
- **OpenInterpreter's Heartwarming Impact**: A member shared how **OpenInterpreter** transformed their life, allowing them to forge incredible friendships and explore the A.I. landscape, reflecting gratitude towards the community.
   - They quoted a [viral demo](https://x.com/MikeBirdTech/status/1839750338179674590) from a year back, underscoring the project's transformational potential in their journey.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Optimizing vector search for customer support**: A new strategy optimizing **vector search** aims to store questions in the vector chunk and answers in metadata, enhancing **precision** in question matches.
   - This method focuses on the **semantics** of questions, streamlining search results by filtering out irrelevant information.
- **Challenges extracting context from Excel**: A member reported struggles with **contextual extraction** from complex Excel files to generate meaningful outputs for LLMs.
   - Despite thorough searching, they haven't found effective methods to tackle this issue.
- **CF Booking Chatbot simplifies conference room management**: The newly built **CF Booking Chatbot** helps manage conference rooms by checking availability and booking, with a [demo video showcasing its features](https://www.linkedin.com/posts/ismile-bharmal-3b82241ab_langchain-chainlit-ai-activity-7245291326919872512-O06M).
   - Plans are underway to integrate **Google Calendar** for automatic syncing, further streamlining the process.
- **Unize Storage generates high-quality knowledge graphs**: Introducing **Unize Storage**, an AI system that creates accurate knowledge graphs from any input text, outperforming existing systems like **LangChain's LLMGraphTransformer** with an **85% accuracy** on larger inputs.
   - This showcases a significant leap over LangChain’s **55% accuracy**, pushing the boundaries of graph generation.
- **Free API access with Unize Storage**: The **Unize API** offers free credits and a chance for users to experiment with the new **Unize Storage** system, allowing for the visualization of generated knowledge graphs.
   - Interested users can start interacting with the system using [this Playground](https://api.unize.org/signup).



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Enforcing PackedDataset Size Limits**: A member proposed to enforce that the packed size cannot exceed **2x the dataset max length** to prevent errors when processing sequences.
   - This suggestion emerged as a potential safeguard against **runtime inconsistencies**.
- **Max Sequence Length Failure Case Uncovered**: It was demonstrated that the current implementation can fail even with a single input exceeding **max_seq_len**, especially with mismatched configurations.
   - A fix using explicit gating for token length was suggested to prevent these **runtime errors**.
- **GitHub Error Discussion Highlights**: The conversation pointed to a [GitHub error](https://github.com/pytorch/torchtune/blob/main/torchtune/datasets/_packed.py#L130) indicating a possible decision to allow sequences greater than **max_seq_len**.
   - This link potentially clarifies the reasoning behind the current handling of **packed dataset sizes**.
- **Collaboration Mandate for Review**: A member suggested that another user should review the content of this discussion upon their return, emphasizing its **importance**.
   - This highlights the **collaborative nature** of the troubleshooting process.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **User Confusion on Function Calling Evaluation**: A user expressed confusion about the **function calling evaluation** process and inquired if they could use their own **evaluation dataset** for analysis with a structure of `<prompt>, <llm_response>, <ideal response>`.
   - They are specifically interested in a package for effective **error breakdown analysis**.
- **Local LLM Deployment Interest**: Another point raised was the desire for functionalities supporting a **locally deployed LLM** to extract error metrics with personal datasets.
   - The user requested recommendations for codebases suited for **function calling capabilities** in this context.
- **Integration of LLMs in Applications**: The conversation highlighted the integration of **Large Language Models (LLMs)** in applications such as Langchain and AutoGPT, referencing models like **GPT, Gemini, Llama,** and **Mistral**.
   - Their advanced **function calling abilities** in powering software solutions were recognized as a growing trend.
- **Valuable Resource: Berkeley Function-Calling Leaderboard**: The user highlighted the **[Berkeley Function-Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)** as a resource for evaluating LLM function calling capabilities.
   - They noted that the leaderboard is based on user-centric function calling use cases.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Exploring OpenAI SDK Use with Jamba**: A user inquired about how to utilize the **OpenAI SDK** with **Jamba**, questioning its feasibility.
   - This inquiry highlights curiosity about integrating different AI tools for enhanced functionalities within the **Jamba** framework.
- **Jamba's Integration Queries Pile Up**: The conversation around **Jamba** is buzzing, particularly on how to streamline processes with the **OpenAI SDK**.
   - Such discussions indicate a growing interest among developers to connect frameworks and enhance their project capabilities.



---


The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Finetuning (Hamel + Dan) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1288943654467145749)** (328 messages🔥🔥): 

> - `Architect/Editor Mode`
> - `Model Comparisons`
> - `Copy Command Feature`
> - `File Handling in Aider`
> - `Token Usage and Efficiency` 


- **Architect/Editor Mode Enhancements**: The new architect/editor mode in Aider allows for improved coding workflows, enabling faster bug fixes and better handling of complex tasks using models like o1-preview and Claude 3.5.
   - Users have reported that using architect mode can streamline the coding process, though it is advised to use Sonnet 3.5 for design-related tasks instead.
- **Benchmarking Models for Performance**: Users are encouraged to benchmark various model combinations, such as o1-preview with different editor models like o1-mini and Sonnet 3.5, to determine the most efficient setups.
   - Feedback suggests that the best performance may depend on context, project size, and editing requirements.
- **Introduction of the /copy Command**: The addition of a new /copy command has been proposed to allow users to easily copy the last output from the LLM to the clipboard for use in other documents.
   - This feature aims to enhance user experience, particularly for those who frequently use the /ask command to obtain information.
- **Handling Multi-Line Input in Aider**: Users discussing the pasting of multi-line text into Aider noted that the recent update allows for the entry of such text without issues.
   - The ability to streamline input methods has been emphasized for improving the coding experience.
- **Token Usage and Editing Formats**: Users have noted varying token usage when switching between edit formats, particularly advocating for the 'whole' edit format to reduce errors during larger edits.
   - It was highlighted that while some may experience increased token usage, overall results tend to improve when opting for the appropriate format based on the project.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://artifacts.e2b.dev/">AI Artifacts by E2B</a>: no description found</li><li><a href="https://aider.chat/docs/faq.html#how-can-i-add-all-the-files-to-the-chat">FAQ</a>: Frequently asked questions about aider.</li><li><a href="https://aider.chat/docs/faq.html#how-do-i-include-the-git-history-in-the-context">FAQ</a>: Frequently asked questions about aider.</li><li><a href="https://aider.chat/docs/usage/commands.html#entering-multi-line-chat-messages">In-chat commands</a>: Control aider with in-chat commands like /add, /model, etc.</li><li><a href="https://aider.chat/docs/usage/modes.html">Chat modes</a>: Using the chat, ask and help chat modes.</li><li><a href="https://aider.chat/2024/09/26/architect.html">Separating code reasoning and editing</a>: An Architect model describes how to solve the coding problem, and an Editor model translates that into file edits. This Architect/Editor approach produces SOTA benchmark results.</li><li><a href="https://code.fittentech.com">免费好用的AI编程助手 Fitten Code - 支持VS Code、PyCharm、Intellj、Visual Studio</a>: no description found</li><li><a href="https://aider.chat/docs/llms/warnings.html">Model warnings</a>: aider is AI pair programming in your terminal</li><li><a href="https://github.com/paul-gauthier/aider/blob/main/aider/website/_posts/2024-09-26-architect.md">aider/aider/website/_posts/2024-09-26-architect.md at main · paul-gauthier/aider</a>: aider is AI pair programming in your terminal. Contribute to paul-gauthier/aider development by creating an account on GitHub.</li><li><a href="https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json">litellm/model_prices_and_context_window.json at main · BerriAI/litellm</a>: Python SDK, Proxy Server (LLM Gateway) to call 100+ LLM APIs in OpenAI format - [Bedrock, Azure, OpenAI, VertexAI, Cohere, Anthropic, Sagemaker, HuggingFace, Replicate, Groq] - BerriAI/litellm</li><li><a href="https://github.com/paul-gauthier/aider/commit/c2c4dbd2a8319f3eab72939f60e2b199a452ff1d">Merge pull request #1595 from jbellis/paste · paul-gauthier/aider@c2c4dbd</a>: feat: rename /clipboard to /paste</li><li><a href="https://github.com/fry69/aider/tree/copy-command">GitHub - fry69/aider at copy-command</a>: aider is AI pair programming in your terminal. Contribute to fry69/aider development by creating an account on GitHub.</li><li><a href="https://aider.chat/docs/usage/conventions.html">Specifying coding conventions</a>: Tell aider to follow your coding conventions when it works on your code.</li><li><a href="https://github.com/paul-gauthier/aider.git">GitHub - paul-gauthier/aider: aider is AI pair programming in your terminal</a>: aider is AI pair programming in your terminal. Contribute to paul-gauthier/aider development by creating an account on GitHub.</li><li><a href="https://github.com/paul-gauthier/aider/pull/1768/files?short_path=cc1e175#diff-cc1e1755d30fcde78f0ba0eb881bb3418d6e5f6b5e29c54de244eeda17059bbb">Proposed changes to the senior / junior editing modes by cschubiner · Pull Request #1768 · paul-gauthier/aider</a>: •	Default to Fast Mode for Efficiency: For quick changes and straightforward coding, Aider would use the standard model. •	Switch to Architect Mode When Needed: For more complex and deliberate codi...</li><li><a href="https://fireworks.ai/blog/cursor">How Cursor built Fast Apply using the Speculative Decoding API </a>: Cursor, an AI-native IDE, leveraged Fireworks inference stack to enhance its features like Instant Apply, Smart Rewrites, and Cursor Prediction. The blog post introduces the Speculative Decoding API, ...
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1288943560896417832)** (22 messages🔥): 

> - `Feedback loops with Aider`
> - `Streamlit limitations`
> - `File creation issues`
> - `Claude 3.5 benefits`
> - `Token usage in Aider` 


- **Improving feedback loops with Aider**: Members discussed challenges in enhancing feedback loops when working with GUIs like *Streamlit*, specifically beyond just describing front-end requirements.
   - One suggested that improvements might require redesigning Aider, as current interactions seem somewhat limited.
- **Streamlit constraints acknowledged**: A user observed that *Streamlit* appears limited for use cases like Aider, suggesting potential redesign might be necessary for more interactivity.
   - Responses implied that while redesigning could improve functionality, it isn't currently a high priority.
- **Aider struggles with file creation**: A user expressed frustration that Aider sometimes fails to create or edit files as expected, experiencing inconsistent behavior.
   - Another noted similar issues are likely due to backend LLM slowdowns, reinforcing the idea of indeterminate behavior across sessions.
- **Assessing Claude 3.5 advantages**: One member inquired whether using *Claude 3.5* offers quality benefits for both weak and strong models, hinting at a potential minor cost trade-off.
   - Another confirmed slight improvements in chat summaries and commit messages, suggesting users should try it and switch back if unsatisfied.
- **Token usage impacts Aider's performance**: A participant noted that Aider works significantly better with fewer files, ideally 1-3 at a time, especially to avoid performance degradation.
   - It was advised to monitor token usage with the command `/tokens`, as exceeding 20k to 30k tokens may lead to unexpected behavior.



**Link mentioned**: <a href="https://aider.chat/docs/config/aider_conf.html">YAML config file</a>: How to configure aider with a yaml config file.

  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/)** (1 messages): 

fry69_61685: https://erikbern.com/2024/09/27/its-hard-to-write-code-for-humans.html
  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1288938892900307037)** (77 messages🔥🔥): 

> - `Molmo and LM Studio`
> - `Llama 3.2 Capabilities`
> - `LM Studio Update Issues`
> - `Using CLI for Local Models`
> - `Conversation Export in LM Studio` 


- **Molmo's Compatibility with LM Studio**: New Vision models won't be supported in LM Studio for a while, as they are not compatible with llama.cpp.
   - Llama 3.2 11b is essentially the same as 3.1 8b but has additional parameters for vision features.
- **Queries on Llama 3.2's Text Generation**: The Llama 3.2 model has been queried regarding token limitations, with mixed reports suggesting it should support up to 128k tokens.
   - Community members discussed various issues with the model's performance and how it handles updates, referring to buggy integrations.
- **Upgrade Concerns for LM Studio**: Users are concerned about transitioning from version 0.2.31 to 0.3.x, particularly regarding model compatibility and settings retention.
   - It was confirmed that upgrading to 0.3.x would replace prior versions but won’t cause data loss.
- **Using LMS CLI for Model Management**: Some users reported issues with the LMS CLI not recognizing that the LM Studio server was running, prompting discussions on troubleshooting.
   - The community shared their findings on accessing local models via WebSocket and discussed the need for official documentation.
- **Conversation Exporting in LM Studio**: In version 0.3.*, the ability to export conversations was removed, leading to concerns about sharing discussions easily.
   - Users were informed that the feature may return in future updates since the current version is a complete overhaul.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2309.10668">Language Modeling Is Compression</a>: It has long been established that predictive models can be transformed into lossless compressors and vice versa. Incidentally, in recent years, the machine learning community has focused on training i...</li><li><a href="https://github.com/lmstudio-ai/lms">GitHub - lmstudio-ai/lms: LM Studio CLI</a>: LM Studio CLI. Contribute to lmstudio-ai/lms development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1288963697695522848)** (246 messages🔥🔥): 

> - `Performance of 70B models at low Q`
> - `Rumors about NVIDIA's RTX 5090 and 5080`
> - `Comparison of different GPU options for AI`
> - `Load testing methods for LLMs`
> - `CPU cooling issues and upgrades` 


- **High Performance with Low Q on 70B Models**: A user reported achieving **18 tokens/sec** on a **70B model** using a **24GB VRAM GPU**, highlighting its usability at **IQ2**.
   - Another user noted that even at this speed, the output is exceedingly faster than manual typing, emphasizing its effectiveness for certain tasks.
- **Rumors Swirl Around NVIDIA GPUs**: Rumors suggest that the upcoming **NVIDIA RTX 5090** could sport **32GB of VRAM**, while discussions reveal a possibility of a **24GB** version of the **RTX 5080** after its **16GB** launch.
   - Users expressed skepticism about the **5080's** specifications, deeming it insufficient for current gaming and AI requirements.
- **Evaluating GPU Options for AI Applications**: Several GPUs were discussed including **RTX 3090s** priced around **$650**, **3090 TIs** at **$850**, and **P40s** at **$300**, each with their respective performance implications.
   - Opinions varied on the best option for AI workloads, with many users emphasizing the importance of newer GPUs and the potential limitations of older models.
- **Load Testing LLM Performance**: For load testing, users recommended using local server API calls in **LM Studio** to send multiple requests efficiently, leveraging custom datasets.
   - One user indicated that they were in the process of creating a tutorial for load testing models within LM Studio.
- **Concerns Over CPU Cooling Solutions**: Discussion arose regarding CPU cooling, with a user noting issues with their **Corsair AIO** cooler making buzzing noises, possibly affecting performance.
   - Alternatives and replacement options were considered, emphasizing the importance of maintaining adequate cooling for optimal hardware functionality.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.amazon.ca/PowerColor-Radeon-7900-XT-Graphics-Card/dp/B0BMWHCGBZ/">no title found</a>: no description found</li><li><a href="https://x.com/kopite7kimi/status/1839343725727941060">Tweet from kopite7kimi (@kopite7kimi)</a>: GeForce RTX 5090 PG144/145-SKU30 GB202-300-A1 21760FP32 512-bit GDDR7 32G 600W</li><li><a href="https://tenor.com/view/you-dont-turn-your-back-on-family-you-cant-walk-away-from-family-you-cant-leave-family-behind-you-cant-ignore-family-you-cant-disregard-family-gif-16058425">You Dont Turn Your Back On Family You Cant Walk Away From Family GIF - You Dont Turn Your Back On Family You Cant Walk Away From Family You Cant Leave Family Behind - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/paulwnos-gif-26909845">Paulwnos GIF - Paulwnos - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://wccftech.com/nvidia-24-gb-geforce-rtx-5080-gpu-after-16-gb-first-gaming-blackwell-shipments-spotted/">NVIDIA Rumored To Launch A 24 GB GeForce RTX 5080 GPU After 16 GB Variant, First Gaming Blackwell Shipments Spotted</a>: NVIDIA&#039;s GeForce RTX 5080 GPU is rumored to get a 24 GB upgrade after the launch of the 16 GB model as reported at Chiphell Forums.</li><li><a href="https://wccftech.com/nvidia-geforce-rtx-5090-32-gb-rtx-5080-16-gb-specs-5090-20k-cores-600w-5080-10k-cores-400w/">NVIDIA GeForce RTX 5090 32 GB &amp; RTX 5080 16 GB Specs Uncovered: 5090 Over 20K Cores &amp; 600W, 5080 Over 10K Cores &amp; 400W</a>: NVIDIA&#039;s GeForce RTX 5090 &amp; RTX 5080, the next generation of GPUs from the green team for gamer, have their specs revealed by Kopite7kimi.</li><li><a href="https://youtu.be/bJKj1yIc4sA">$60 AI GPU???</a>: Benchmarking the NVIDIA P102-100. An old crypto mining card that can be reused for AI inference. It is extremely cheap and a great value for those people wit...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fqsafn/nvidia_jetson_agx_thor_will_have_128gb_of_vram_in/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://lmstudio.ai/docs">Getting Started | LM Studio Docs</a>: Run open-source LLMs locally on Mac, Windows, or Linux</li><li><a href="https://www.overclockers.co.uk/8pack-supernova-mk3-amd-ryzen-threadripper-pro-extreme-pc-sys-8pk-00076.html">8Pack Supernova MK3 - AMD Ryzen Threadripper Pro Extreme PC</a>: Order 8Pack Supernova MK3 - AMD Ryzen Threadripper Pro Extreme PC now online and benefit from fast delivery.</li><li><a href="https://wccftech.com/nvidia-geforce-rtx-5090-32-gb-rtx-5080-16-gb-specs-5090-20k-cores-600w-5080-10k">NVIDIA GeForce RTX 5090 32 GB &amp; RTX 5080 16 GB Specs Uncovered: 5090 Over 20K Cores &amp; 600W, 5080 Over 10K Cores &amp; 400W</a>: NVIDIA&#039;s GeForce RTX 5090 &amp; RTX 5080, the next generation of GPUs from the green team for gamer, have their specs revealed by Kopite7kimi.</li><li><a href="https://videocardz.com/newz/nvidia-geforce-rtx-5090-and-rtx-5080-specs-leaked">NVIDIA GeForce RTX 5090 and RTX 5080 specs leaked - VideoCardz.com</a>: GeForce RTX 5090 to feature 21760 CUDA cores, 32GB GDDR7 memory and 600W, RTX 5080 gets 16GB VRAM Coming from Kopite7kimi himself.  One of the most reliable NVIDIA leakers has now confirmed the specs ...</li><li><a href="https://www.canadacomputers.com/index.php?cPath=43_557_559&sf=:3_22&co=&mfr=&pr=">Shop for Powered By Nvidia &amp; more - Canada Computers</a>: no description found
</li>
</ul>

</div>
  

---



### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1288956818726457355)** (4 messages): 

> - `Llama 3.2 vision models`
> - `Cerebras chip optimization` 


- **Call to Form Group for Llama 3.2 Models**: There is a proposal to form a working group to integrate **Llama 3.2 vision models** into **llama.cpp** as discussed in [this GitHub issue](https://github.com/ggerganov/llama.cpp/issues/8010).
   - The issue highlights that multimodal support was removed and can be reinstated following the refactoring of related components.
- **Interest in Optimizing Code for Cerebras Chips**: A member inquired about efforts to optimize code for **Cerebras chips** and whether purchasing them is a wise choice.
   - Another member expressed interest in connecting with anyone from Cerebras for insights, indicating a desire for more information on this topic.



**Link mentioned**: <a href="https://github.com/ggerganov/llama.cpp/issues/8010">server: Bring back multimodal support · Issue #8010 · ggerganov/llama.cpp</a>: Multimodal has been removed since #5882 Depends on the refactoring of llava, we will be able to bring back the support: #6027 This issue is created mostly for tracking purpose. If someone want to t...

  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1288942895545581629)** (26 messages🔥): 

> - `Triton Windows Wheel`
> - `Understanding BLOCK_SIZE`
> - `Inline Assembly in Triton`
> - `TMA in Triton Kernel`
> - `MPS Fault Handling` 


- **Seeking Latest Triton Wheel for Windows**: A user is looking for the latest compiled [Triton wheel for Windows](https://link.to.triton.windows) that is compatible with Python 3.10.
   - This reflects the community's ongoing interest in ensuring proper installation for different platforms.
- **Clarifying BLOCK_SIZE in Triton**: Discussion arose about `BLOCK_SIZE`, with one member noting it differs from thread counts, describing it as the dimension size for parallelization, such as in arrays.
   - Others compared `BLOCK_SIZE` to CUDA's `blockDim`, emphasizing it defines the number of elements operated on within a block.
- **Using Inline Assembly for Triton Kernels**: A query about performing specific assembly operations in Triton led to insights about how to use [inline assembly](https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html#triton.language.inline_asm_elementwise) without requiring large shared memory.
   - Inline assembly was suggested, but concerns were raised about its limitations versus more optimal broadcast methods.
- **Exploring TMA in Triton Kernel Development**: A member inquired about the necessity of using full TMA instructions in Triton versus simpler methods like `tl.make_block_ptr` to achieve similar performance boosts.
   - This highlights the community's exploration of efficient data management within the Hopper architecture and the nuances of kernel optimization.
- **MPS Faults Cause Execution Halts**: Concerns were raised about MPS faults not being managed by Python, resulting in total application halts when issues occur.
   - This sentiment reflects broader frustrations with Apple's MPS, leading to discussions about alternative frameworks and their functionalities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html#triton.language.inline_asm_elementwise)">triton.language.inline_asm_elementwise &mdash; Triton  documentation</a>: no description found</li><li><a href="https://pytorch.org/blog/hopper-tma-unit/">Deep Dive on the Hopper TMA Unit for FP8 GEMMs</a>: Abstract  </li><li><a href="https://github.com/triton-lang/triton/blob/1e093fbfff2fb3bd4406d9379f7aa62deaf74965/python/test/unit/hopper/test_gemm.py#L56-L57">triton/python/test/unit/hopper/test_gemm.py at 1e093fbfff2fb3bd4406d9379f7aa62deaf74965 · triton-lang/triton</a>: Development repository for the Triton language and compiler - triton-lang/triton
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1289006842688700416)** (17 messages🔥): 

> - `PyTorch Profiler performance counters`
> - `torch.flip HIP error`
> - `Swin2SR GitHub repository`
> - `PyTorch benchmarking repositories`
> - `Updating dictionaries in TorchScript` 


- **PyTorch Profiler performance counter query**: A user inquired if the **PyTorch Profiler** required performance counters enabled when using an NVIDIA GPU, suspecting it simply measures time and VRAM.
   - *It shouldn't,* according to another user's belief, as it does not track metrics like L2 hit rate.
- **HIP error encountered with torch.flip**: A user reported experiencing a **HIP error: invalid device function** when using **torch.flip** with **swin2sr**.
   - They noted that the code executes normally on CPU but fail with GPU, prompting a request for troubleshooting help.
- **Swin2SR GitHub project shared**: The conversation highlighted the GitHub repository for **swin2sr**, an Efficient Transformer for image super-resolution, with a link shared for reference.
   - The project description indicates its ties to the **ECCV 2022** conference and its practical applications, with a suggestion to try out its functionalities.
- **Finding unoptimized PyTorch code for benchmarking**: A user asked about the existence of online repositories containing simple, unoptimized **PyTorch code** along with benchmarking metrics for practice.
   - In response, another member provided a link to a **nsys tutorial** that illustrates diagnosing performance bottlenecks regarding data loading and GPU.
- **Efficient dictionary updates in TorchScript**: A member raised a question regarding efficient methods to update a **dictionary** containing long to int mappings in **TorchScript** with large input batches.
   - They noted the limitation of using for loops when compiling to PyTorch and sought alternative solutions.



**Link mentioned**: <a href="https://github.com/mv-lab/swin2sr">GitHub - mv-lab/swin2sr: [ECCV] Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration.  Advances in Image Manipulation (AIM) workshop ECCV 2022. Try it out! over 3.3M runs https://replicate.com/mv-lab/swin2sr</a>: [ECCV] Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration.  Advances in Image Manipulation (AIM) workshop ECCV 2022. Try it out! over 3.3M runs https://replicate.com/...

  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1288993317950062613)** (6 messages): 

> - `.clangd Configurations`
> - `CUDA Path Issues`
> - `Profiling Kernel Function Performance` 


- **Updating .clangd to fix errors**: A member successfully updated their `.clangd` config file to resolve unknown command errors by removing problematic flags.
   - Despite these changes, they still faced issues related to **libdevice** and CUDA installation paths.
- **Confusion around CUDA installation version**: A member expressed confusion about the error stating that the installation at `/usr/local/cuda-12.6 is 11.0`, despite it being correct.
   - They attempted to add `--cuda-path=/usr/local/cuda` but didn’t find any improvements, leading to further troubleshooting.
- **Inquiry on profiling kernel functions**: A member sought advice on profiling the performance of a HIP kernel, which reportedly performed 5x worse than a compression kernel.
   - They explored using `clock()` and a separate timing thread and asked for additional debugging strategies from others.
- **Command line argument mix-up**: A member acknowledged confusion between command line arguments for `clang` and `nvcc`, leading to incorrect configurations.
   - They successfully landed on a new `.clangd` configuration with appropriate flags, which helped resolve most issues.


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1288975422062723092)** (5 messages): 

> - `Image Processing in C`
> - `Popular Book Sections`
> - `stb_image Library`
> - `Yann LeCun Mention` 


- **Recommendations for Image Reading in C**: A user inquired about recommendations for reading images in C, specifically in chapter 3 of a book they were reading.
   - Another member suggested the [stb_image library](https://github.com/nothings/stb/blob/master/stb_image.h), noting it is a 'drop-in and forget about it' solution for most purposes.
- **Yann LeCun's Endorsement of Ocean C++**: A member highlighted that **Yann LeCun** mentioned using the **Ocean C++** image processing library at Meta, lending credibility to the suggestion.
   - This endorsement from a well-known figure in AI and image processing was well-received by the community.
- **User Satisfaction with stb_image Library**: After checking out the recommended **stb_image library**, the original user expressed their satisfaction, stating it does the job perfectly.
   - This positive feedback reflects the library's effectiveness for reading images in C.
- **Popularity of Book's Sections**: Another user shared their enjoyment of section 4.7 of the book, contributing to the discussion about its popularity.
   - Their enthusiasm indicates that certain parts of the book resonate well with readers.



**Link mentioned**: <a href="https://github.com/nothings/stb/blob/master/stb_image.h">stb/stb_image.h at master · nothings/stb</a>: stb single-file public domain libraries for C/C++. Contribute to nothings/stb development by creating an account on GitHub.

  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1288942346355871807)** (138 messages🔥🔥): 

> - `Windows support for GPU models`
> - `FP8 and Int8 training issues`
> - `Torchao performance profiling`
> - `Issues with quantized training`
> - `Links and resources on GPU programming` 


- **Windows support proves valid**: A member shared that contrary to the theory that only Linux users are interested in GPU programming, around **10-15%** of downloads for bitsandbytes are from Windows users, indicating a rising interest.
   - Another member humorously speculated about possible contributions from 'zombie' Windows machines in various setups.
- **Challenges with FP8 and Int8 training**: Members discussed experiencing out-of-memory (OOM) errors on NVIDIA systems when attempting to load FP8 models, while Int8 worked fine, suggesting weight quantization wasn't effectively applied.
   - The conversation highlighted a need to ensure FP8 works efficiently with memory and raised questions about whether FP8 primarily benefits compute speed rather than memory reduction.
- **Profiling Torchao's CPUOffloadOptimizer**: One member sought to review profiling results of the Torchao CPUOffloadOptimizer by reaching out for insights from the original author and the community.
   - Another member recommended creating a discussion thread for a broader audience to participate in the conversation.
- **Issues with model quantization**: The ongoing discussion about model behaviors pointed out that while weights do not appear quantized in FP8, they work in Int8, raising concerns around memory usage in both cases.
   - Members noted significant memory consumption, around **24,000 MB**, prompting questions about optimization and the effectiveness of FP8.
- **Feedback on public resources and links**: There were mentions of minor typos found in a blog related to Torchao, with one member noting a broken link that redirected improperly, raising clarity issues.
   - The community encouraged communication about such errors to facilitate better resources for GPU programming.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/PyTorch/status/1839696520527929384">Tweet from PyTorch (@PyTorch)</a>: We’re happy to officially launch torchao, a PyTorch native library that makes models faster and smaller by leveraging low bit dtypes, quantization & sparsity. Our techniques are written in easy-to-rea...</li><li><a href="https://scholar.google.com/citations?user=_2_KAUsAAAAJ">Furkan G�z�kara</a>: Assistant Professor Computer Engineer, Toros University - Cited by 20 - Data Mining - Sentiment Analysis - Text Classification - Product Clustering - Clustering</li><li><a href="https://github.com/pytorch/ao/issues/957">is this only for linux? · Issue #957 · pytorch/ao</a>: I installed on windows and failing from torchao.quantization import quantize_ pip freeze Microsoft Windows [Version 10.0.19045.4894] (c) Microsoft Corporation. All rights reserved. R:\CogVideoX_v1\...</li><li><a href="https://huggingface.co/THUDM/CogVideoX-5b">THUDM/CogVideoX-5b · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/bu2mX.gif">Huang Jensen Nvidia Ceo GIF - Huang Jensen Nvidia Ceo - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/pytorch/ao/blob/63cb7a9857654784f726fec75c0dc36167094d8a/torchao/prototype/quantized_training/int8.py#L124">ao/torchao/prototype/quantized_training/int8.py at 63cb7a9857654784f726fec75c0dc36167094d8a · pytorch/ao</a>: PyTorch native quantization and sparsity for training and inference - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/tree/main/torchao/prototype/quantized_training#int8-mixed-precision">ao/torchao/prototype/quantized_training at main · pytorch/ao</a>: PyTorch native quantization and sparsity for training and inference - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/blob/main/benchmarks/quantized_training/pretrain_llama2.py">ao/benchmarks/quantized_training/pretrain_llama2.py at main · pytorch/ao</a>: PyTorch native quantization and sparsity for training and inference - pytorch/ao</li><li><a href="https://github.com/bghira/SimpleTuner/pull/986/files#diff-327015d4d445c4efaaa945a93701df4c68e3bc401dc4ddb7e55f2b5dc7854d6fR103-R116>">(wip, int8 only) torchao: fp8/int8 by bghira · Pull Request #986 · bghira/SimpleTuner</a>: no description found</li><li><a href="https://github.com/bghira/SimpleTuner/pull/986/files#diff-327015d4d445c4efaaa945a93701df4c68e3bc401">(wip, int8 only) torchao: fp8/int8 by bghira · Pull Request #986 · bghira/SimpleTuner</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1288949705530802259)** (17 messages🔥): 

> - `Edge LLM Challenge`
> - `Integration of Sonnet and Voice`
> - `Prompting vs Speaking`
> - `Meta AR Glasses`
> - `Code Execution on New Platforms` 


- **Teams forming for Edge LLM Challenge**: Participants expressed interest in teaming up for the **Edge LLM Challenge**, which requires teams to develop compression methods for pre-trained LLMs running on smartphones.
   - The challenge involves creating models like **Phi-2**, **Llama-3-8B**, and **Qwen-7B** evaluated on the **OpenCompass benchmark**.
- **Exploring Sonnet's Voice Features**: A member inquired about the integration of **Sonnet** with voice input and output, curious if it supports TTS functionalities.
   - Despite expressing skepticism about current tools like ChatGPT, they acknowledged the potential of conversational AI.
- **Preference for Text Over Audio Responses**: One member expressed a personal preference for text responses over audio, citing speed and clarity, especially with complex content like code.
   - They noted that while speaking could enhance input speed, the lack of editability in voice responses poses challenges.
- **Interest in Meta's AR Glasses Direction**: A participant shared enthusiasm for the advancements by **Meta** in developing AR glasses, particularly their application in coding.
   - They emphasized the desire for the glasses to serve as a viable platform, akin to the **iPhone**, to unlock coding capabilities.



**Link mentioned**: <a href="https://edge-llms-challenge.github.io/edge-llm-challenge.github.io/challenge">no title found</a>: no description found

  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1288969786352472065)** (2 messages): 

> - `Meetups in Guatemala`
> - `GPU reading/work groups in London` 


- **Seeking Meetups in Guatemala**: A member mentioned they are in **Guatemala** and inquired if anyone else is around for meetups, suggesting interest in neighboring countries like **Belize** and **Mexico**.
   - There's an openness to connecting with others in the region for discussions or collaborations.
- **Recommendations for GPU Groups in London**: Another member asked for recommendations on any **GPU reading or work groups** in **London** that could provide collaborative opportunities.
   - This highlights a desire for local engagement and learning within the GPU community.


  

---


### **GPU MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1288990620446232586)** (33 messages🔥): 

> - `RMSNorm integration`
> - `MLP block backpropagation`
> - `Kernel efficiency concerns`
> - `Attention backward pass issues`
> - `RepKV backward debugging` 


- **RMSNorm Backward Pass Fully Integrated**: The **backward pass** of **RMSNorm** has been successfully integrated, and gradients after the final RMSNorm match as expected.
   - *Next up is the SwiGLU backward* to further advance integration towards the transformer.
- **MLP Block Backprop Success**: It was confirmed that backpropagation through the **MLP block** of the transformer is functioning correctly, with plans to tackle the attention block next.
   - Several small changes were noted, but the integration is on track and working as intended.
- **Concerns Over Kernel Efficiency**: There are concerns that the current implementation of the kernel wastes a significant number of threads, particularly noted when the replicate factor is set to 4, leading to potential **75% of threads being noop**.
   - *An extra scratch buffer* has been created for backward to address dbias_buffer size issues.
- **Struggles with Attention Backward Pass**: The backward pass for **Attention** is raising concerns, appearing suspiciously high, especially due to the different replication methods compared to **PyTorch**.
   - The user suspects that one of **RoPE** or **repKV** backward implementations is broken after extensive debugging.
- **Confusion Over Debugging Results**: After considerable debugging, the user has concluded that the issue is likely **not with repKV** backward, leading to confusion about the actual source of the problem.
   - Despite rewriting the repKV backward CPU reference to be thoroughly safe, the user expressed frustration and decided to take a break from the task.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1289312419910389771)** (6 messages): 

> - `TensorWave MI300X Offer`
> - `Community Engagement` 


- **TensorWave Offers MI300X to Boost Adoption**: Darrick from TensorWave announced their willingness to provide some **MI300X** units to the community, aiming to enhance **adoption and education** of the platform.
   - Darrick encouraged interested members to *send him a DM*, highlighting this opportunity as something **exciting**.
- **Community Responds to Offer**: A community member expressed enthusiasm by stating, *DM sent! This is super exciting* after Darrick's announcement.
   - This indicates a **positive reception** and eagerness to engage further with TensorWave's offer.


  

---


### **GPU MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1289001896027226132)** (6 messages): 

> - `Quantized Training Repo`
> - `Using Multi-GPU for Training`
> - `Distillation from Quantized Model`
> - `Config File for Larger Models` 


- **Quantized Training Repo for Bigger Datasets**: A member shared a [GitHub repo](https://github.com/gau-nernst/quantized-training) that explores training for quantized models and uses streaming HF datasets, which accommodates large datasets.
   - The repo supports **FSDP2** and **DDP**, with a notable difference from the torchao PR regarding the usage of quantized activations in gradient calculation.
- **Multi-GPU Training Solutions**: Issues arose when trying to utilize a second GPU during training; a member advised using **torchrun** to enable multi-GPU support.
   - To activate DDP mode, simply add `--ddp` when launching, as the default setting is **fsdp2**.
- **Exploring Distillation Options**: One member suggested considering **distillation from a large quantized Llama model** as a potentially interesting approach.
   - This could open pathways to effective model size reduction and performance enhancements.
- **Configurations Needed for Model Scaling**: Another user reported needing to add a config file to support larger models in the training repo.
   - They indicated that the current script settings were insufficient for scaling up effectively.



**Link mentioned**: <a href="https://github.com/gau-nernst/quantized-training">GitHub - gau-nernst/quantized-training: Explore training for quantized models</a>: Explore training for quantized models. Contribute to gau-nernst/quantized-training development by creating an account on GitHub.

  

---


### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1288976710234013727)** (1 messages): 

> - `LiteRT functionalities`
> - `gpu.cpp cross-platform capabilities` 


- **Inquiry on LiteRT vs gpu.cpp**: A new member asked whether **LiteRT** covers functionalities that **gpu.cpp** aims to provide, specifically for running models across platforms like Android, iOS, and PCs.
   - The focus is on using **on-device GPU compute** seamlessly without friction.
- **Discussion on Cross-Platform Model Running**: Members discussed the importance of efficiently running models across various platforms, emphasizing the role of **gpu.cpp** in simplifying this integration.
   - The conversation highlighted the need for tools like LiteRT to streamline cross-platform functionality and GPU utilization.


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1289109579820699732)** (5 messages): 

> - `Liger Kernel Weight Handling`
> - `Family Trip Update`
> - `Lambda Vendor Recommendation` 


- **Liger Kernel Struggles with Weight Copying**: Questions arose about the Liger Kernel's treatment of existing model layers, particularly regarding skipping weight copying during application of the kernel. A member shared that this is a critical issue, especially for scenarios like **LoRA training** and **SFT** on pre-trained models.
   - Two GitHub issues were highlighted: one relates to an error involving the **AutoLigerKernelForCausalLM** leading to a **ValueError** while loading, and another where **loss does not drop** when using Liger Kernel at Qwen2.5.
- **Byron's Family Trip Revelations**: Byron shared that he just returned from a delightful **two-week family trip** and mentioned plans to resume reviewing **pull requests** and **issues** today.
- **Lambdas for Vendor Solutions**: A member recommended **Lambdas** as a top vendor choice, praising its pricing as relatively good. This suggestion indicates a search for cost-effective solutions among members.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/linkedin/Liger-Kernel/issues/268">inference qwen2 model ,The reasoning is garbled  and  ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?) · Issue #268 · linkedin/Liger-Kernel</a>: 🐛 Describe the bug when I load model with AutoLigerKernelForCausalLM ,I get ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?) when load mdoel Apply Model-Specific Pat.....</li><li><a href="https://github.com/linkedin/Liger-Kernel/issues/257">Loss does not drop when using Liger Kernel at Qwen2.5 · Issue #257 · linkedin/Liger-Kernel</a>: 🐛 Describe the bug I am trying to instruction tuning Qwen2.5-14B-Instruct with Liger Kernel. I know that the liger kernel is supported in the dev version of huggingface transformers. However, when .....
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1289255869116190861)** (2 messages): 

> - `Apple hardware support`
> - `Metal Shading Language Specification` 


- **Locating Apple Hardware Supported Dtypes**: A user expressed difficulty in finding a list of **dtypes** supported by Apple hardware.
   - It was suggested that the answer could be found in the [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).
- **Discussion Regarding the Metal Shading Language**: Members discussed the necessity of accessing the **Metal Shading Language Specification** for understanding support details of **dtypes**.
   - The provided link was highlighted as a valuable resource to clarify such queries.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1289272252927316049)** (1 messages): 

> - `Live Meeting Announcement` 


- **Live Broadcast on Microsoft Teams**: A notification was shared regarding a live meeting that can be joined via [this link](https://teams.microsoft.com/l/meetup-join/19%3ameeting_YzgwY2EzMWMtYTA0Zi00NDhjLTk0MmMtN2Y4MDRlMjQ2MTI2%40thread.v2/0?context=%7b%22Tid%22%3a%2243083d15-7273-40c1-b7db-39efd9ccc17a%22%2c%22Oid%22%3a%22bc6a8639-bf95-4464-af3e-20c110ea129f%22%7d).
- **Team Engagement Through Live Meetings**: The announcement highlights the importance of live meetings in fostering team engagement and collaboration.



**Link mentioned**: <a href="https://teams.microsoft.com/l/meetup-join/19%3ameeting_YzgwY2EzMWMtYTA0Zi00NDhjLTk0MmMtN2Y4MDRlMjQ2MTI2%40thread.v2/0?context=%7b%22Tid%22%3a%2243083d15-7273-40c1-b7db-39efd9ccc17a%22%2c%22Oid%22%3a%22bc6a8639-bf95-4464-af3e-20c110ea129f%22%7d">Join conversation</a>: no description found

  

---


### **GPU MODE ▷ #[diffusion](https://discord.com/channels/1189498204333543425/1288899271193526342/1288938713962909708)** (9 messages🔥): 

> - `M2 Pro Benchmarks`
> - `DiffusionKit`
> - `Flux Diagram`
> - `Mini Diffusion Model`
> - `Visuals in Chat` 


- **M2 Pro Benchmarks shared**: A member expressed excitement about having **M2 Pro benchmarks** and referenced [DiffusionKit](https://github.com/argmaxinc/DiffusionKit) for on-device inference of diffusion models.
   - They also included an image link related to the DiffusionKit repository.
- **Challenges with Non-Quantized Testing**: A member mentioned it would be cool to test **non-quantized models**, but noted limitations due to only having **16GB of RAM**.
   - This highlights the hardware constraints when experimenting with high-performance models.
- **Seeking Flux Diagram**: A user asked if anyone had a **drawn-out diagram of flux**, and others discussed what type of diagram would be helpful, possibly a block diagram.
   - *Promptsiren* pointed to a diagram that was previously shared on Reddit, indicating that resources are available.
- **Discussion on Mini Diffusion Model**: One member inquired if a topic related to a **mini diffusion model** from scratch was being discussed, suggesting a focus on foundational understanding.
   - This indicates active exploration among members regarding the implementation of diffusion models.
- **Visuals and Photos Shared**: A member mentioned that some **nice photos** were coming up, prompting a light-hearted engagement in the chat.
   - Such interactions contribute to a friendly and creative atmosphere within the channel.



**Link mentioned**: <a href="https://github.com/argmaxinc/DiffusionKit">GitHub - argmaxinc/DiffusionKit: On-device Inference of Diffusion Models for Apple Silicon</a>: On-device Inference of Diffusion Models for Apple Silicon - argmaxinc/DiffusionKit

  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1288939314360881153)** (176 messages🔥🔥): 

> - `Llama Model Fine-tuning`
> - `Model Checkpoints and Loading Issues`
> - `Graphics Card Rumors`
> - `Training Neural Networks`
> - `AI Application in Gaming` 


- **Discussion on Llama Model Fine-tuning**: Users discussed various aspects of fine-tuning Llama models, including confusion over data formats like `chatml` and the need to adjust tokenizer settings for special tokens.
   - The conversation touched on the importance of avoiding overfitting, with some members expressing concerns over low training losses indicating model memorization.
- **Model Checkpoints and Loading Errors**: A user reported an error while trying to load the `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` model, indicating specific exceptions related to data mismatch during processing.
   - This sparked conversation about troubleshooting checkpoint loading, with insights into potential issues with sizing and configuration settings.
- **Graphics Card Specifications and Rumors**: The community engaged in a debate over upcoming GPU models like the 5090 and speculated on VRAM sizes, with claims that a 32GB option is likely despite skepticism.
   - Users shared their personal preferences and experiences, indicating that while rumors are prevalent, proof from benchmarks is what’s needed to settle ongoing speculations.
- **Training Neural Networks and Overfitting**: Members discussed optimal configurations for training neural networks, exploring concepts like sequence length and gradient accumulation to improve convergence rates.
   - There were questions about using intelligent evaluation techniques to terminate training at optimal points, emphasizing the balance between training efficiency and resource management.
- **AI Applications in Gaming**: Discussion included the impact of VRAM on AI applications within gaming, with emphasis on how increased memory can enhance performance for advanced LLM tasks.
   - Users noted the ongoing challenge of ensuring developers optimize games properly to utilize the capabilities of high-end graphics cards.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1aqlNQi7MMJbynFDyOQteD2t0yVfjb9Zh?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://jan.ai/">Turn your computer into an AI computer - Jan</a>: Run LLMs like Mistral or Llama2 locally and offline on your computer, or connect to remote AI APIs like OpenAI’s GPT-4 or Groq.</li><li><a href="https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF">unsloth/Llama-3.2-3B-Instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/issues/421">config.json file not found, fine tuning llama3 with unsloth, after saving the file to hugging face  · Issue #421 · unslothai/unsloth</a>: i use unsloth to fine tune llama 3-8B..., after traning complete i save this model to hugging face by using &#39;push_to_hub&#39;, but it shows these files : .gitattributes README.md adapter_config.js...</li><li><a href="https://github.com/unslothai/unsloth/issues/1040#issuecomment-2377762522">What is the right way to load Qwen2&#39;s chat interface? · Issue #1040 · unslothai/unsloth</a>: I get this error: chat_template, stop_word, yes_map_eos_token, ollama_modelfile = CHAT_TEMPLATES[chat_template] ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^ KeyError: &#39;Qwen2-1.5B&#39; From this code: def test_un...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1289055985322954823)** (6 messages): 

> - `Job search frustrations`
> - `Active AI subscriptions`
> - `AI relationships` 


- **Job search frustrations from school job board**: A member expressed disappointment regarding the lack of interviews from the **school job board**.
   - They noted that, currently, they have not secured any opportunities despite ongoing applications.
- **Active OpenAI & Claude subscriptions**: The same member confirmed that they have active subscriptions to both **OpenAI** and **Claude**, indicating a commitment to exploring AI tools.
   - This might suggest they're leveraging these resources in their current circumstances.
- **Discussion on AI relationships**: A humorous query arose regarding the status of not having a girlfriend, leading to the question: does an AI count as a partner?.
   - Another member prompted for elaboration on this concept, indicating an interest in the interplay of human and AI relationships.
- **Member's health status**: The member also mentioned feeling sick, adding a personal aspect to their status update.
   - This health concern seems to intertwine with their other updates on job search and relationships.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1288938513072652301)** (80 messages🔥🔥): 

> - `Transformers Updates`
> - `Quantization Issues`
> - `Fine-Tuning Techniques`
> - `Model Loading Challenges`
> - `Optimizer Errors in Lighting AI` 


- **Transformers and Model Updates**: Users reported that they have the latest version of **transformers** (4.45.1) installed, suggesting they are keeping their libraries updated.
   - Several members discussed potential improvements to their models and issues around specific model loading and quantization strategies.
- **Quantization Challenges with Phi3.5**: A user identified that the quantization for **Phi3.5** leads to a **vocab size mismatch** error, prompting discussions on alternatives.
   - Another user mentioned successfully using the **Phi3 mini** model without issues, suggesting differences in compatibility based on model size.
- **Fine-Tuning Strategies for LLMs**: A conversation centered around fine-tuning strategies for **LLAMA 3.1**, with advice to incorporate continued pre-training on a specific Q&A dataset.
   - Participants emphasized that pre-training is essential to infuse domain-specific knowledge into such models before fine-tuning.
- **Issues with Loading Models**: There were discussions about challenges faced when loading models in **GGUF** format, particularly in terms of fine-tuning capabilities.
   - Users highlighted the need to convert or access original models and suggested utilizing tools compatible with both quantization and model format.
- **Optimizer Errors in Lighting AI**: A user encountered an **AttributeError** related to the **AdamW** optimizer in Lighting AI, indicating potential issues with the selected version.
   - Suggestions were made to try alternative optimizers or revert to earlier versions of **torch**, but the issue persisted with specific updates.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.vllm.ai/en/latest/getting_started/examples/offline_inference.html">Offline Inference &#8212; vLLM</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing,">Google Colab</a>: no description found</li><li><a href="https://docs.vllm.ai/en/latest/getting_started/quickstart.html">Quickstart &#8212; vLLM</a>: no description found</li><li><a href="https://docs.vllm.ai/en/v0.6.1/getting_started/examples/offline_inference.html">Offline Inference &#8212; vLLM</a>: no description found</li><li><a href="https://docs.vllm.ai/en/v0.6.1/getting_started/examples/offline_inference_chat.html">Offline Inference Chat &#8212; vLLM</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1288943588939661405)** (13 messages🔥): 

> - `Data Packing in Training`
> - `Frameworks for Pretraining GPT-2`
> - `Discussion Etiquette in Technical Queries`
> - `Deepspeed for Pretraining`
> - `Handling Data Masking` 


- **Data packing boosts training efficiency**: A member explained that by packing data, training frameworks manage unrelated parts effectively, enabling streamlined training with multiple examples at once.
   - They elaborated that the framework predicts the second token after the first, facilitating enhanced training dynamics.
- **Exploring suitable frameworks for GPT-2 pretraining**: A member inquired about frameworks for pretraining a small **GPT-2** model, pointing out that **trl** and **LlamaFactory** are primarily used for fine-tuning.
   - Another member recommended using **Deepspeed** for pretraining due to its capabilities.
- **Technical query etiquette gets highlighted**: A discussion unfolded on how to approach questions about frameworks, emphasizing the balance between doing research and asking for help.
   - Members expressed that searching beforehand could lead to more informed questions, reducing redundancy in technical queries.
- **Call for patience in discussions about pretraining**: One member responded to critiques about their inquiries, stating they had conducted ample research but sought further clarification from experienced users.
   - They encouraged openness to acknowledging limits in knowledge, suggesting this would foster better discussions.


  

---



### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1288940269877854218)** (184 messages🔥🔥): 

> - `Discussion on Hugging Face models`
> - `Challenges with uncensored models`
> - `Techniques for bypassing AI restrictions`
> - `Creating datasets for chat models`
> - `Using multiple LLMs for aggregation` 


- **Challenges with Uncensored Models**: A user inquired about an uncensored 12B chat model, specifically for creating a game bot, indicating that Llama doesn't allow certain topics.
   - Participants noted that Hugging Face models are censored and suggested looking into Venice.ai as an uncensored alternative.
- **Techniques for Jailbreaking AI Models**: Users discussed various methods to bypass AI restrictions, including classic jailbreaks like asking the model to do the opposite of what is said.
   - Another suggested technique involved backloading objectionable content after building rapport with the AI through innocuous topics.
- **Creating Datasets for Chat Models**: One user expressed interest in training a smaller chat model and was looking for datasets that focus strictly on conversations.
   - Suggestions included using synthetic data generators and exploring existing datasets on Hugging Face, as well as scraping personal texts and emails.
- **Aggregation of Responses from Multiple LLMs**: A user inquired if there is a service that queries multiple LLMs and aggregates responses for improved output quality.
   - It was mentioned that while such a service does not currently exist, it would be relatively easy to create.
- **Discussion on Model Limitations**: Participants discussed the limitations of Llama models concerning token processing and their effectiveness on platforms like Raspberry Pi.
   - There was an emphasis on understanding input and output limits in relation to performance and model interaction.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://digitalcommons.mtu.edu/mobiletext/">
Mobile Text Dataset and Language Models | Department of Computer Science | Michigan Technological University
</a>: no description found</li><li><a href="https://stackoverflow.com/help/how-to-ask">How do I ask a good question? - Help Center</a>: Stack Overflow | The World&#x2019;s Largest Online Community for Developers</li><li><a href="https://huggingface.co/spaces/argilla/synthetic-data-generator">Synthetic Data Generator - a Hugging Face Space by argilla</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1BJ4_U1V-ohJAUqedVSs-6h1qZm7anfeV#scrollTo=hp78IDn1NQzo">Google Colab</a>: no description found</li><li><a href="https://tenor.com/view/squirrel-huh-what-up-dog-gif-18781858">Squirrel Huh GIF - Squirrel Huh What - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/datasets/argilla/FinePersonas-Synthetic-Email-Conversations">argilla/FinePersonas-Synthetic-Email-Conversations · Datasets at Hugging Face</a>: no description found</li><li><a href="https://youtu.be/TsIzbYkMXa4">The Legend of Zelda: Twilight Princess, Temple of Time Theme Visualizer in 4K HDR</a>: ...See? I told you we&#39;d meet again.#3dart #cinematic #temple #music #ambient #zelda #blender</li><li><a href="https://huggingface.co/docs/accelerate/main/en/package_reference/launchers#accelerate.notebook_launcher">Launchers</a>: no description found</li><li><a href="https://www.scientificamerican.com/article/there-is-no-such-thing-as-conscious-thought/#:~:text=We%20are%20not%20simply%20puppets%20manipulated%20by%20our%20unconscious%20thoughts,">There Is No Such Thing as Conscious Thought | Scientific American</a>: no description found</li><li><a href="https://huggingface.co/datasets/shaunck96/wiki-cot-with-reflection/viewer/default/train?p=1">shaunck96/wiki-cot-with-reflection · Datasets at Hugging Face</a>: no description found</li><li><a href="https://www.biorxiv.org/content/10.1101/2020.07.01.183384v1.full">High-performance brain-to-text communication via imagined handwriting</a>: Brain-computer interfaces (BCIs) can restore communication to people who have lost the ability to move or speak. To date, a major focus of BCI research has been on restoring gross motor skills, such a...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1289314495260524627)** (1 messages): 

> - `Neuralink CUDA usage`
> - `7b FP8 model`
> - `BF16 and FP32 confusion` 


- **Exploring Neuralink's CUDA Implementation**: A member discussed their experience with **CUDA** in working with Neuralink's technologies for better model performance.
   - They noted that this experience contributes to their understanding of advanced GPU programming techniques.
- **Working with 7b FP8 Model**: The participant mentioned their involvement in the **7b FP8** model, highlighting its significance in the context of processing capabilities.
   - They are focusing on optimizing this model for enhanced efficiency in practical applications.
- **Clarifying BF16 with FP32 Master Weights**: A member clarified a previous mention of a **yellow line**, stating it accurately refers to **bfloat16 with FP32 master weights**, correcting a typo.
   - This correction underscores the importance of precision in discussing neural network configurations.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1288945486090211399)** (9 messages🔥): 

> - `Two Minute Papers`
> - `Alibaba's MIMO`
> - `Tokenizer Training Research`
> - `Interactive Scene Control` 


- **Two Minute Papers Under Fire**: Some members expressed disappointment in **Two Minute Papers**, stating it has shifted from informative content towards **marketing** focused videos.
   - One user noted that there is a **gap for good video coverage** ever since another channel ceased operations.
- **Alibaba Launches MIMO Technology**: A member shared that **Alibaba** introduced a new AI called **MIMO** that creates realistic character videos from simple inputs such as character, motion, and scene.
   - They highlighted that **10 demos** were showcased, particularly focusing on **Interactive Scene Control**.
- **Request for Tokenizer Training Papers**: A user sought recommendations for **research papers on tokenizer training** to enhance the multilingual capabilities of LLMs.
   - This inquiry indicates ongoing interest in improving language model performance through effective tokenization strategies.
- **Two Minute Papers YouTube Video Discussion**: A user shared a link to a YouTube video titled '**OpenAI’s New ChatGPT: 7 Incredible Capabilities**' from **Two Minute Papers**.
   - This sparked a conversation about the overall quality and direction of the channel, emphasizing contrasting opinions on its current value.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/minchoi/status/1838949848516547040">Tweet from Min Choi (@minchoi)</a>: Alibaba introduces MIMO  New AI that creates realistic character videos from simple inputs like character, motion, and scene.  10 demos  1. Interactive Scene Control</li><li><a href="https://www.youtube.com/watch?v=QDfE0HwDBo8">OpenAI’s New ChatGPT: 7 Incredible Capabilities!</a>: ❤️ Check out Lambda here and sign up for their GPU Cloud: https://lambdalabs.com/paperPlay the Tron game: https://agpallav.com/tron.htmlSources:https://www.y...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1288998802086301726)** (12 messages🔥): 

> - `VividNode update`
> - `AI tweet going viral`
> - `Game promotion`
> - `Leaderboard feedback`
> - `Flux-schnell demo` 


- **VividNode gets major upgrade!**: A member shared that their AI personal assistant program, **VividNode (pyqt-openai)**, now supports major LLMs like **Gemini, Claude, and Llama** and includes **Speech-to-Text** (STT) and **Text-to-Speech** (TTS) features.
   - *They are seeking contributors to enhance the project and facilitate integration of additional LLMs.* [View the release here](https://github.com/yjg30737/pyqt-openai/releases/tag/v1.2.0).
- **AI tweet gaining traction**: A member expressed excitement that their tweet is going viral, contributing to their project's visibility.
   - *Another member mentioned promoting them at an AI meetup, even though the audience was small.*
- **Leaderboard discussion heats up**: Concern was raised about a leaderboard that only shows **price** and **max tokens**, lacking useful metrics like **latency** and **throughput** for effective model comparisons.
   - *Members pointed to an alternative website demonstrating more comprehensive comparisons and mentioned adding further performance metrics to their own leaderboard to aid beginners.*
- **Game development hype**: A member confidently stated that the game they've developed is actually pretty cool, generating excitement in the chat.
   - *The general atmosphere reflects positivity and newfound interest in AI projects.*
- **Flux-schnell demo in the works**: A member announced they are working on a demo for **regional prompt attention** in **flux-schnell**, planning to later share the source code and comfyui node.
   - *This showcases ongoing development efforts within the community.*



**Link mentioned**: <a href="https://github.com/yjg30737/pyqt-openai/releases/tag/v1.2.0">Release v1.2.0 · yjg30737/pyqt-openai</a>: VividNode(pyqt-openai) v1.2.0 Release Notes New Features  Random Prompt Generation: Added a feature for random prompt generation to enable continuous image creation TTS and STT Support:  Implemente...

  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1288941859732717569)** (4 messages): 

> - `4D Scene Understanding`
> - `3D Data Rendering`
> - `Temporal Data in Models`
> - `2D Video Consistency`
> - `Achievements in Computer Vision` 


- **4D Scene Understanding still needs work**: The discussion highlights that achieving **4D scene understanding**—including **temporal data**—is still a challenging frontier, with significant progress made but more needed.
   - *A member noted*, 
- **3D Data Rendering not evident yet**: Concerns were raised about the challenges of **rendering 3D data**, noting that it's not yet straightforward.
   - *A member acknowledged*, 
- **Temporal Data in 2D Models**: The inclusion of **temporal data** is often applied in **2D models**, providing consistent video outputs and creating a foundation for future developments.
   - *Community members indicated* that combining **3D with time** facilitates 4D scene understanding, pointing to a crucial evolution in the field.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1289156386902638657)** (2 messages): 

> - `Tokenizer Training for Multilingual LLMs`
> - `Pytorch Techniques for Weight Management` 


- **Seeking Research Papers on Tokenizer Training**: A member requested suggestions for good research papers focusing on **tokenizer training** to enhance the **multilingual capabilities** of large language models.
   - They are particularly interested in **effective methodologies** that could inform their approach to this area of research.
- **Freezing Weights During Training**: A member discussed the concept of copying back the weights of the old tokens after each batch update during training.
   - They also mentioned considering a more *pytorchic* method to freeze specific rows in a matrix, reflecting on their own similar thoughts regarding the issue.


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1288977779848773683)** (9 messages🔥): 

> - `Cybersecurity services`
> - `Text-to-video model training`
> - `Image sharpening technique`
> - `Flux.1-dev optimization` 


- **Cybersecurity Services Offered by Expert Hacking**: An individual presented themselves as an expert hacker, claiming to provide various **cybersecurity courses and services**.
   - *Hit me up for all your hacking services/courses* indicates a clear invitation for collaboration.
- **Seeking Repos for Distributed GPU Training of T2V Models**: A user inquired about repositories for **distributed GPU training** pipelines specifically for **text-to-video (T2V) models**.
   - Another user suggested checking out the [CogVideo SAT finetuning](https://github.com/THUDM/CogVideo/blob/main/sat/README.md) for relevant resources.
- **Creating an Image Sharpener Tool**: A member discussed their project aimed at creating an image sharpener focused on **removing frames and outfilling Pokémon cards**.
   - They are preparing for **training** but want to ensure they've covered all bases first.
- **Questions on Using Flux.1-dev with Limited VRAM**: A user sought advice on which **Flux.1-dev quant** would fit within their **6GB VRAM** while using ComfyUI.
   - Another participant recommended asking directly in the **ComfyUI Discord server**, mentioning that most optimizations are handled automatically.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://hastebin.com/share/sosukokohu.ruby">Hastebin</a>: no description found</li><li><a href="https://github.com/THUDM/CogVideo/blob/main/sat/README.md">CogVideo/sat/README.md at main · THUDM/CogVideo</a>: Text-to-video generation: CogVideoX (2024) and CogVideo (ICLR 2023) - THUDM/CogVideo</li><li><a href="https://github.com/Stability-AI/stablediffusion/blob/main/ldm/models/diffusion/ddpm.py#L380">stablediffusion/ldm/models/diffusion/ddpm.py at main · Stability-AI/stablediffusion</a>: High-Resolution Image Synthesis with Latent Diffusion Models - Stability-AI/stablediffusion
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1288950180002926643)** (3 messages): 

> - `Gemini Tokenization`
> - `Database Upgrade Delay`
> - `Chatroom UI Enhancements` 


- **Gemini Token Changes Simplified**: OpenRouter will transition to counting **tokens** instead of characters for Gemini models, effectively reducing token numbers by a factor of ~4 on the `/activity` page.
   - In addition, prices will *double* to align with the lower tier per-token prices on AI Studio, leading to an estimated **50% cost cut** for Flash and 1.5 Pro models.
- **Database Upgrade Downtime Cancelled**: Scheduled downtime for a database upgrade was initially set to begin in 10 minutes, but the upgrade has been delayed, resulting in no downtime.
   - An update will be provided once a new schedule is determined.
- **Chatroom UI Gets a Facelift**: [OpenRouterAI](https://x.com/OpenRouterAI/status/1839738812877918617) announced enhanced UI for the Chatroom, showing responses from models with their reasoning collapsed by default.
   - More improvements are on the way, promising an even better user experience in the future.



**Link mentioned**: <a href="https://x.com/OpenRouterAI/status/1839738812877918617">Tweet from OpenRouter (@OpenRouterAI)</a>: The Chatroom now shows responses from models with their reasoning collapsed by default.  o1 vs Gemini vs Sonnet on 🍓:

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1288952397493964822)** (186 messages🔥🔥): 

> - `Llama 3.2 vision parameters`
> - `OpenRouter error messages`
> - `Claude 3.5 Sonnet tool calling issues`
> - `Translation model recommendations`
> - `Model hosting criteria on OpenRouter` 


- **Llama 3.2 vision parameters**: A user inquired about parameters to use with **Llama 3.2 vision** to avoid rejections, especially when evaluating attractiveness.
   - Members discussed that the model might be trained not to respond to such queries due to safety concerns.
- **OpenRouter error messages**: Several users reported encountering a **429 Resource Exhausted** error, indicating that the model could not process requests due to hitting rate limits.
   - Responses indicated that OpenRouter has been pushing for higher rate limits with Google to mitigate these issues.
- **Claude 3.5 Sonnet tool calling issues**: A user encountered an error while trying to use the **Claude 3.5 Sonnet** model, noting discrepancies in the required message formatting.
   - Discussions revealed that omitting parameters in function calls works for OpenAI models but causes issues with Anthropics' models.
- **Translation model recommendations**: A user sought advice for translation models without strict content restrictions, particularly for translating fictional dialogues.
   - They shared a prompt they were using but faced challenges with dialogues being flagged for inappropriate content.
- **Model hosting criteria on OpenRouter**: A user asked about the criteria for adding new models to the **OpenRouter** infrastructure, looking into hosting options.
   - It was clarified that a provider must be able to host the model at scale for it to be considered.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/chat">Chatroom | OpenRouter</a>: LLM Chatroom is a multimodel chat interface. Add models and start chatting! Chatroom stores data locally in your browser.</li><li><a href="https://molmo.allenai.org/">Molmo by Ai2</a>: Multimodal Open Language Model built by Ai2</li><li><a href="https://x.com/openrouterai/status/1839738812877918617?s=46&t=nM71JKV50FJ0CR4r6r2_Rg">Tweet from OpenRouter (@OpenRouterAI)</a>: The Chatroom now shows responses from models with their reasoning collapsed by default.  o1 vs Gemini vs Sonnet on 🍓:</li><li><a href="https://openrouter.ai/docs/transforms">Transforms | OpenRouter</a>: Transform data for model consumption</li><li><a href="https://openrouter.ai/credits">Credits | OpenRouter</a>: Manage your credits and payment history</li><li><a href="https://github.com/billmei/every-chatgpt-gui/blob/main/README.md">every-chatgpt-gui/README.md at main · billmei/every-chatgpt-gui</a>: Every front-end GUI client for ChatGPT. Contribute to billmei/every-chatgpt-gui development by creating an account on GitHub.</li><li><a href="https://openrouter.ai/docs/requests#tool-calls">Requests | OpenRouter</a>: Handle incoming and outgoing requests</li><li><a href="https://github.com/e2b-dev/ai-artifacts/pull/61">  Add OpenRouter support with Claude 3.5 Sonnet model by PierrunoYT · Pull Request #61 · e2b-dev/ai-artifacts</a>: This pull request adds support for OpenRouter as a new AI provider, specifically integrating the Claude 3.5 Sonnet mod from Anthropic through OpenRouter&amp;#39;s API. Key changes:   Updated lib/model...</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/quotas#error-code-429">no title found</a>: no description found
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1289186720197382236)** (6 messages): 

> - `Channel Etiquette`
> - `Project Progress` 


- **Incorrect Channel Usage Addressed**: A user pointed out that a message was posted in the wrong channel, leading to *clarification on channel appropriateness.*
   - Another member mentioned that messages unrelated to **Cohere** could still be allowed in the channel due to their enjoyment of the content.
- **Awaited Project Completion**: A member expressed optimism about their project, stating that a lot of months have been invested in making it work, hinting at its upcoming launch.
   - They thanked the community for the direction on posting, indicating enthusiasm for the project's reception in the proper channel.


  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1288983727292289033)** (19 messages🔥): 

> - `Finetuning embed-english-v3`
> - `Custom embedding models`
> - `RAG inclusions format`
> - `Embeddings in construction domain` 


- **Finetuning the embed-english-v3 model**: A user inquired about the possibility of finetuning the **embed-english-v3** model for specific use cases, particularly for use in the embed API.
   - However, members clarified that **no embedder is available to be finetuned** at the moment and suggested using a custom embedding model from Hugging Face if finetuning is necessary.
- **Feedback on embedding improvement requests**: Another user mentioned that while current embeddings work fine, the results could be improved for the **construction space** by utilizing specific construction terms in finetuning.
   - Members recognized the feedback and expressed willingness to share it with the relevant teams for consideration.
- **Inquiry on RAG inclusions**: A user sought advice on formatting **instructional headers** and how **RAG inclusions** would appear when appended to strings sent to the LLM.
   - No responses were provided regarding formatting or examples related to RAG inclusions in the discussion.


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1289270623192875008)** (146 messages🔥🔥): 

> - `New API v2 endpoints`
> - `Flashcard generation`
> - `Fine-tuning models`
> - `Rate limits for trial keys` 


- **Launch of API v2 Endpoints**: New versions of API endpoints have been announced: v2/chat, v2/embed, v2/classify, and v2/rerank, with **Chat V2** receiving the most significant updates, including a `messages` parameter and support for system messages.
   - For more details, users are encouraged to check the [API Reference](https://docs.cohere.com/reference/chat-v2) and provide feedback on their experiences.
- **Challenges in Flashcard Generation**: A user expressed the challenge of generating flashcards from a large corpus of text, noting that the model struggles to identify the 'right' terms for extraction and association with definitions.
   - To improve generation quality, they suggested that **fine-tuning** might be beneficial, particularly for crafting responses that align with previous cards without direct quoting.
- **Concerns Over Fine-Tuning**: Another user mentioned that fine-tuning is often not worth the investment for individual projects, particularly if the output adjustments can be achieved through other means.
   - They highlighted that while fine-tuning helps, it may be unnecessary for straightforward tasks like flashcard generation, which can be managed with effective prompt engineering.
- **Trial Key Limitations Explained**: A discussion about the use of multiple trial keys revealed that rate limits are account-based rather than key-based, which means using many trial keys provides no additional benefit.
   - Users noted that anyone utilizing trial keys might consider the implications on usage limits, especially if they rotate accounts frequently.
- **Community Support for Projects**: A community member offered to assist those working on projects by providing credits for effective usage of Cohere's services, emphasizing a collaborative spirit in the community.
   - They encourage developers to share their projects, suggesting that having multiple trial keys is not the best approach and reiterating the potential for community support.



**Link mentioned**: <a href="https://docs.cohere.com/docs/rate-limits">API Keys and Rate Limits — Cohere</a>: This page describes the limitations around Cohere&#x27;s API.

  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1289328195514142858)** (1 messages): 

> - `Cultural Multilingual LMM Benchmark`
> - `Volunteer Native Translators`
> - `Co-authorship Invitation`
> - `CVPR 2025 Submission` 


- **Developing a Cultural Multilingual LMM Benchmark**: The team at **MBZUAI** is working to develop a **Cultural Multilingual LMM Benchmark** for **100 languages** with a newly created multimodal dataset and their translations into local languages.
   - They are seeking **native translators as volunteers** to help rectify mistakes in the current version of the dataset.
- **Invitation for Volunteer Native Translators**: Volunteers who assist in the translation effort will receive an invitation to co-author the paper intended for submission to **CVPR'2025**.
   - A wide range of languages are included in the project, covering **Indian, South Asian, African, and European** languages.
- **List of Languages Needed**: The project specifically needs help with languages such as **Hindi, Swahili, and Hungarian**, among others.
   - The complete list of languages is shared to attract potential volunteers who can read, write, or speak these languages.
- **Connect with Ashmal Vayani**: For anyone interested in the project, Ashmal Vayani invites connections via [LinkedIn](https://www.linkedin.com/in/ashmal-vayani/) or personal messages.
   - He encourages direct messages here for further discussion regarding the project.


  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1288941221590601728)** (160 messages🔥🔥): 

> - `Tiled Upscale vs ADetailer`
> - `Using AMD GPUs for SD`
> - `Performance Differences of GPUs`
> - `Refurbished vs Used GPUs`
> - `SSD Impact on Model Load Times` 


- **Tiled Upscale is an Alternative to ADetailer**: It was noted that **Tiled Upscale** serves as an alternative to **ADetailer**, offering similar effects with an easier setup.
   - However, the downside is that it operates approximately **50 times slower**, as it upscales the entire image rather than a specific area.
- **Concerns About AMD GPUs for SD Workflows**: AMD GPUs, like the 5700 XT, were discussed as underperforming for **Stable Diffusion (SD)** and **Blender**, with users noting they are better suited for gaming than productivity tasks.
   - One user mentioned that a **3070** could outperform a **7900 XTX**, indicating limited performance in productivity applications.
- **Refurbished vs Used GPUs Discussion**: Participants discussed the advantages of refurbished GPUs over used ones, highlighting that refurbished cards are often repaired and double-verified, minimizing risks.
   - One member shared their experience of buying a refurbished **3090 TI**, stating it's almost as good as new, while there are concerns regarding mining-related wear on used cards.
- **SSD Impact on Model Load Times**: It was confirmed that running **Stable Diffusion** on an **SSD** significantly decreases model load times, potentially by **10x or more** compared to HDD configurations.
   - Users indicated loading models from an **M.2 SSD** vastly improves performance for image generation tasks, while older spinning disk technologies lag behind.
- **Prompting for Object Sizes**: Inquiries were made about effective prompting for sizing objects in image generation, with ideas on phrasing for comparative sizes shared.
   - Humorous suggestions like 'yuge' and 'bigly' were made, though it was advised to avoid such terms.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/bingbangboom/flux_dreamscape">bingbangboom/flux_dreamscape · Hugging Face</a>: no description found</li><li><a href="https://hub.docker.com/r/rocm/pytorch">no title found</a>: no description found</li><li><a href="https://www.tomshardware.com/pc-components/gpus/stable-diffusion-benchmarks">Stable Diffusion Benchmarks: 45 Nvidia, AMD, and Intel GPUs Compared</a>: Which graphics card offers the fastest AI performance?
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1288940422030299168)** (74 messages🔥🔥): 

> - `User Interface Issues`
> - `API Functionality Queries`
> - `Subscription Promotions`
> - `Freelancing Platforms`
> - `Model Availability` 


- **Users facing UI issues and errors**: Multiple users reported that the Perplexity website stopped responding to clicks and interactions, with several seeing `net::ERR_BLOCKED_BY_CLIENT` errors in the console.
   - One user noted that the issue persists on both desktop and mobile browsers, while others mentioned that the Android app continues to function normally.
- **Questions regarding API functionality**: Users inquired about accessing the latest news through the Perplexity API, specifically seeking up-to-date information on generative AI.
   - Concerns were also raised about limitations faced while using specific APIs, including suggestions to check for any robust solutions available.
- **Subscription promotion issues**: A user expressed confusion about redeeming a promotional code for a pro subscription, stating they have not gained access after redeeming it.
   - Another user requested information on how to transfer their pro subscription to a friend's account, asking about the procedure involved.
- **Freelancing platform inquiries**: A user sought recommendations for freelancing platforms similar to Upwork, indicating a desire to find suitable alternatives for freelance work.
   - This inquiry sparked discussions on various platforms as users shared their experiences and suggestions.
- **Llama 3.2 model updates**: An individual asked if the Llama 3.2 model has been added yet, showing interest in the latest advancements in AI models.
   - The query reflects ongoing curiosity among users regarding the addition of new models and their functionalities.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1288950513697554463)** (8 messages🔥): 

> - `Meta's Orion AR Glasses`
> - `OpenAI's For-Profit Pivot`
> - `New Blood Type Discovery`
> - `Skin Cancer Information`
> - `Neural Fields in Visual Computation` 


- **Meta's Orion AR Glasses Revealed**: A recent update discusses [Meta's Orion AR Glasses](https://www.perplexity.ai/search/city-with-the-most-bike-lanes-hhNCIS6oRRCli0fdq8Z32g) aimed at enhancing augmented reality experiences.
   - Early feedback indicates potential impacts on user interaction in virtual spaces.
- **OpenAI Shifts to For-Profit Model**: OpenAI has made a pivotal [for-profit pivot](https://www.perplexity.ai/search/what-happened-with-wordpress-w-V8a7N3D4QMqBc3vZdzXzVg), likely affecting its future funding and operational strategies.
   - This change is seen as a response to competitive pressures in the AI landscape.
- **Significant New Blood Type Discovered**: A groundbreaking discovery reports a new [blood type](https://www.youtube.com/embed/J7cra2xt_DQ), which could reshape transfusion protocols.
   - Researchers emphasize its relevance for specific demographic groups.
- **Skin Cancer Awareness**: Insightful information on [skin cancer](https://www.perplexity.ai/search/about-skin-cancer-W7CNdzsDTkie3137nI2X0g#0) attracts attention regarding prevention and early detection.
   - Community discussions highlight ongoing concerns about skin health.
- **Exploration of Neural Fields in Visual Computation**: [Neural fields](https://www.perplexity.ai/search/neural-fields-in-visual-comput-HIz9FQKQTCqDXhF3eRDIXw) are being examined for their applications in visual computing and AI.
   - This emerging field promises innovative techniques to enhance visual data processing.


  

---



### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1288963523325464609)** (60 messages🔥🔥): 

> - `Memory Size in GPUs`
> - `DisTrO Paper Release`
> - `Knowledge Graphs and AI`
> - `Claude Sonnet 3.5 Performance`
> - `HW/SW Integration in AI` 


- **Discrepancy in GPU Memory Sizes**: Discussions arose about the memory size differences between **5080** and **5070** GPUs, with some suggesting the 5080 should have near **20GB**.
   - Members highlighted a pattern of doubling memory sizes across GPU generations, citing the **3080** and **3090** as examples.
- **Anticipation for DisTrO Paper**: Inquiries were made about the release date of the **DisTrO** paper, with mentions of an insightful talk possibly containing abstract concepts.
   - Links to the full talk were shared after some members expressed difficulty in finding it on YouTube.
- **Exploration of Knowledge Graphs**: A new member shared insights on their work with **knowledge graphs** and how applying concepts of **Bitcoin Ordinal Theory** led to unique embeddings.
   - They described their hypothesis on how LLMs develop **graph-based representations** from semantic richness, possibly indicating paths towards emergent intelligence.
- **Improved Reasoning with Claude Sonnet 3.5**: A member shared their success in enhancing the reasoning capabilities of **Claude Sonnet 3.5** by utilizing example reasoning traces.
   - The isolated example showed promising improvements, indicating potential directions for further exploration.
- **Challenges with Computational Tools**: Members discussed issues with certain computational tools crashing browsers, indicating **Firefox** struggles with intensive tasks.
   - Workarounds and updates on ongoing processes were shared, with hopes for future improvements in tool performance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/aieintern/status/1836828882307026997">Tweet from aie intern (@aieintern)</a>: Nous Forge notes and transcription below  Quoting Alex Volkov (Thursd/AI) (@altryne)   And finally we get a glimpse of @NousResearch Forge from @karan4d (cc @max_paperclips )  Test time compute infere...</li><li><a href="https://x.com/bradthilton/status/1839718742051184842">Tweet from Brad Hilton (@bradthilton)</a>: Claude Sonnet 3.5 was able to solve an AIME problem with the help of o1 reasoning traces that it was otherwise unable to solve 🤯  1/🧵</li><li><a href="https://x.com/swyx/status/1836624609850069138">Tweet from swyx.ai (@swyx)</a>: @elder_plinius @leonardtang_ @haizelabs DisTrO full talk  from Nous Chief Scientist @bloc97_  (who we talked to for @latentspacepod here https://www.latent.space/p/iclr-2024-recap )  unexpected applau...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1288952448240586876)** (17 messages🔥): 

> - `Hermes deployment options`
> - `Llama 3.2 requirements`
> - `Hyperparameter adjustment for models` 


- **Users inquire about running Hermes locally**: A member asked if it's possible to run **Hermes** locally on a 4090 GPU or if it requires API access, to which another member confirmed it can be run locally using **LMStudio**.
   - LMStudio supports any **GGUF version** and offers a model search capability for users to find **Hermes** easily.
- **Llama 3.2 and GPU requirements**: There were questions about whether the **Llama 3.2 1B** model can run on software without a GPU, with members confirming that a GPU is still needed for execution.
   - One member shared that a MacBook performs well with **H3 8B**, indicating GPU dependency persists for optimal performance.
- **Need for hyperparameter adjustments**: Discussion revealed that **hyperparameter adjustments** are necessary when training models of different sizes, with emphasis on the need established through experience as mentioned by one member.
   - Members mentioned applying adjustments to learning rates, batch sizes, and epochs, as indicated in the **H3 paper**, to manage performance effectively.
- **Training dynamics for larger models**: It was noted that different training configurations were necessitated for the **70B and 405 models**, leading to lesser epochs and reduced learning rates for larger parameter counts.
   - Further inquiries were raised regarding whether these adjustments stemmed from **scaling laws** or prior experiences.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1289047656655487016)** (1 messages): 

> - `Arduino-Based Current Sensor`
> - `Power Outage Detection`
> - `Related Research Access` 


- **Seeking Free Resources on Arduino Current Sensors**: A BSEE student proposed research on developing an **Arduino-Based Current Sensor** for **Power Outage Detection** and is seeking related literature.
   - *I'm currently broke right now,* so any free access to academic resources or research papers would be greatly appreciated.
- **Request for Research Literature Recommendations**: The student is asking the community if they know where to find relevant **related research** without incurring membership or download fees.
   - They emphasize their situation, indicating that cost is a significant concern at this moment.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1289047656655487016)** (1 messages): 

> - `Arduino-Based Current Sensor`
> - `Power Outage Detection`
> - `Research Literature Access` 


- **Seeking literature on Arduino sensor development**: A BSEE student proposed a research project titled 'Development of an Arduino-Based Current Sensor for Power Outage Detection' and is seeking related literature.
   - *They expressed a need for sources that do not require membership fees* due to current financial constraints.
- **Challenges in accessing research papers**: The student highlighted the difficulty in obtaining research literature without incurring costs.
   - *This raises broader concerns on access to essential academic resources* for students facing financial challenges.


  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1288938888106213510)** (61 messages🔥🔥): 

> - `Agentic Search Challenges`
> - `AI in Education`
> - `Energy Use of AI`
> - `AI Tools for Productivity`
> - `Future Generations and Technology` 


- **Agentic Search is Costly**: A developer shared that their **Agentic Search** project was too expensive in terms of compute and tokens, leading them to kill the project.
   - They considered fine-tuning a smaller model like **Llama 3b** due to resource constraints with larger models.
- **AI Usage in Academia Ramps Up**: A discussion highlighted that many students at the master's level are using AI to complete assignments, with over **50%** reportedly pasting AI-generated content.
   - This sparked debate on the implications of AI usage as a tool for productivity versus academic integrity.
- **Debate on AI's Energy Consumption**: Questions arose regarding the energy used by AI systems after a member noted that some say, 
- **Tech Tools for Developers**: A recommendation was made for the Chrome extension **ChatGPT Toolbox**, which includes features like chat history search and prompt management to enhance ChatGPT experiences.
   - It was also suggested to wait for the upcoming **Orion model** for novel developing tools that could enhance productivity significantly.
- **Concerns Over Future Skills**: The conversation touched on the idea that future generations might not learn traditional skills like writing with a pen due to increasing reliance on technology.
   - Participants joked about how future society might view basic skills, questioning the evolution of learning tools and their implications.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1289037553223209060)** (16 messages🔥): 

> - `Voice feature issues`
> - `Advanced voice mode functionality`
> - `Attachment capabilities`
> - `Deployment timelines` 


- **Voice feature's sound output issue persists**: Multiple users reported issues with the **standard voice feature** where sound output is absent in the **GPT store** while it works fine in the **advanced voice mode**.
   - *One user expressed frustration,* stating they could not utilize the voice feature in any custom GPTs, leading to further discussion about potential glitches.
- **Workarounds for voice feature**: A user suggested that switching voices within a **custom GPT** might temporarily resolve sound issues, even though it takes them out of the specific GPT.
   - Another member confirmed the workaround's effectiveness but pointed out the limitation concerning the ability to use custom uploaded PDFs.
- **Anticipation for attachment features in advanced voice model**: Inquired about adding **attachments** such as PDF or Docx files to the **advanced voice model**, prompting speculation on a timeline for release.
   - A member advised to be patient with new feature rollouts, elaborating on various dependencies affecting availability like platform, region, and user tier.


  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1289055392780783660)** (10 messages🔥): 

> - `Open Source Model Sponsorship`
> - `LLM Search Space Simulation`
> - `OpenAI Function Calling API`
> - `Model Validity and Tuning` 


- **Exploring Sponsorship for Open Source Models**: A member inquired if Eleuther offers any sponsorship programs for open source models, expressing a lack of resources to fully train their projects.
   - This raises a discussion about community support for such initiatives within the open source realm.
- **Innovations in LLM Search Space Simulation**: A concept was proposed involving an abstract search space for LLMs, utilizing Monte Carlo tree search to simulate continuous thinking with text diffusion.
   - *This method aims to rank the most coherent thoughts during the computational process,* suggesting potential advancements in LLM architecture.
- **Curiosity on OpenAI's Function Calling API Mechanics**: A community member cross-posted a question about how OpenAI’s function calling API operates, speculating on whether it uses a fine-tuned model to ensure output validity.
   - Another member theorized it might not be fine-tuned but could involve additional prompting or logit bias to guarantee responses conform to **valid JSON**.
- **Skepticism on OpenAI's Model Approach**: A participant expressed doubts about the effectiveness of OpenAI's approach, suggesting it produces results similar or inferior to alternative methods.
   - *They noted that implementing extra prompts and logit bias could achieve similar outcomes without the complexity of multi-model setups.*


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1288954521690705970)** (52 messages🔥): 

> - `FP6 and FP16 Weight Distributions`
> - `Verbatim Memorization in LLMs`
> - `Looped Transformers vs Universal Transformers`
> - `Layerwise Positional Encoding`
> - `Confidence Metrics in Inference` 


- **Comparing Weight Distributions Pre and Post FP6**: Discussion revolved around comparing weight distributions of models before and after **FP6**, with hints at using libraries like [seaborn](https://seaborn.pydata.org/) for visualization.
   - The goal is to see if any anomalies arise, as members suggested experimenting with multiple plotting libraries.
- **Study on Verbatim Memorization in LLMs**: A recent study introduced a framework to assess **verbatim memorization** in LLMs, highlighting control over long sequences with implications for privacy.
   - Key findings revealed that non-trivial repetition leads to memorization, as demonstrated in an independent [paper](https://arxiv.org/abs/2407.17817).
- **Debate on Looping Transformers vs Universal Transformers**: A debate centered on the novelty of looped Transformers as presented in a recent paper, with opinions suggesting it's not particularly innovative compared to **Universal Transformers** (UTs).
   - Concerns were raised regarding the modeling assumptions, particularly the need for ground-truth iterations during training.
- **Effectiveness of Layerwise Positional Encoding**: Discussion on whether a layerwise positional encoding could aid extrapolation in inference was inconclusive, with members doubting its overall impact.
   - It was suggested that, while beneficial for very specific problems, it may not offer a significant advantage in broader tasks.
- **Evaluating Confidence Metrics for Inference**: Members discussed the use of confidence as a metric for evaluating when to stop inference, with differing opinions on its effectiveness.
   - It was acknowledged that while confidence could be valuable, no significant stability issues currently exist for most implementations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.17817">Demystifying Verbatim Memorization in Large Language Models</a>: Large Language Models (LLMs) frequently memorize long sequences verbatim, often with serious legal and privacy implications. Much prior work has studied such verbatim memorization using observational ...</li><li><a href="https://arxiv.org/abs/2409.15647">Looped Transformers for Length Generalization</a>: Recent work has shown that Transformers trained from scratch can successfully solve various arithmetic and algorithmic tasks, such as adding numbers and computing parity. While these Transformers gene...</li><li><a href="https://arxiv.org/abs/2210.02671">A Logic for Expressing Log-Precision Transformers</a>: One way to interpret the reasoning power of transformer-based language models is to describe the types of logical rules they can resolve over some input text. Recently, Chiang et al. (2023) showed tha...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1288994110749020246)** (1 messages): 

> - `Embedding states in KV`
> - `Text representation factors` 


- **Recurrent Information Dominates Embedding States**: A member suggested that there is **substantially more recurrent information** stored in embedding states within **KV** than previously considered.
   - They emphasized that the **current text representation** may play a lesser role, primarily acting as an input in the overall process.
- **Current Text Representation's Role**: The discussion highlighted that the **text representation** might not significantly impact embedding outcomes, mainly serving as an input.
   - This raises questions about the importance placed on text representation when considering overall model performance.


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1288972244248301632)** (2 messages): 

> - `Vision LLMs`
> - `ColQwen2 Model`
> - `Visual Retrievers` 


- **Running Vision LLMs Locally**: A user inquired about the process of running **vision LLMs locally**, showcasing an interest in practical implementation.
   - *No specific methods were provided, highlighting a potential gap in shared knowledge on this topic.*
- **ColQwen2 Makes Waves**: A new model, **ColQwen2**, was announced as a top visual retriever, surpassing **colpali-v1.1** with a **+5.1 nDCG@5** score on the Vidore Leaderboard.
   - This model utilizes a **Qwen2-VL backbone**, promising superior performance in visual retrieval tasks, as noted in [this post](https://x.com/manuelfaysse/status/1839657285053788483).
- **Impressive Performance Metrics**: ColQwen2 is trained on the same data as its predecessor, **colpali-v1.1**, marking a significant advancement in the field.
   - *The emphasis on metrics like nDCG@5 reflects the strong focus on performance within visual model evaluations.*



**Link mentioned**: <a href="https://x.com/manuelfaysse/status/1839657285053788483">Tweet from Manuel Faysse (@ManuelFaysse)</a>: 🚨 New model alert: ColQwen2 ! It&#39;s ColPali, but with a Qwen2-VL backbone, making it the best visual retriever to date, topping the Vidore Leaderboard with a significant +5.1 nDCG@5 w.r.t. colpali...

  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1288980929196326953)** (4 messages): 

> - `Testing on H100s`
> - `FA3 integration`
> - `Maintaining FA2 alongside FA3` 


- **Testing on H100s for Small Models**: A member expressed willingness to assist with testing on **H100s** for small models, indicating confidence in their ability to contribute.
   - This sparked enthusiasm and appreciation from others in the discussion.
- **Pointers for FA3 Work**: A request for guidance on adding **FA3** was made, with a suggestion to utilize support from ongoing work on GitHub, specifically referencing [pull request #1282](https://github.com/EleutherAI/gpt-neox/pull/1282).
   - The conversation guided the member on how to route attention for FA3 within the transformer engine.
- **Differentiating FA3 from FA2**: Clarification was made that while integrating **FA3**, the team would not replace **FA2**; both would be necessary moving forward.
   - This indicates a strategic approach to enhance model capabilities without losing previous improvements.
- **Encouragement to Explore Resources**: Another member confirmed they would explore the mentioned references and thanked the initial proposer for the links.
   - This reflects collaborative support and resource-sharing within the community.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/gpt-neox/pull/1282.">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://github.com/NVIDIA/TransformerEngine/pull/1019/files#diff-0af6d715a51b3efcebd6067805b5d17b64d25ef84399e256bade01a602ce4192).">Add support for flash-attn 3 by cyanguwa · Pull Request #1019 · NVIDIA/TransformerEngine</a>: Description This PR integrates flash-attn 3 into TE&amp;#39;s FlashAttention module. This includes FP16/BF16 fwd+bwd, and FP8 fwd. As of Aug 22, 2024, FA3 can be installed via the following commands: ...</li><li><a href="https://github.com/EleutherAI/gpt-neox/pull/1035">integrated flash attention 2 by a663E-36z1120 · Pull Request #1035 · EleutherAI/gpt-neox</a>: Integrated flash attention 2 (ver. 0.2.2 -&gt; 2.2.1). Wall-clock performance improvement observed for 125M param model with seq length 4096 and batch size 128 running on 1 a10 GPU.</li><li><a href="https://github.com/NVIDIA/TransformerEngine/pull/1019">Add support for flash-attn 3 by cyanguwa · Pull Request #1019 · NVIDIA/TransformerEngine</a>: Description This PR integrates flash-attn 3 into TE&amp;#39;s FlashAttention module. This includes FP16/BF16 fwd+bwd, and FP8 fwd. As of Aug 22, 2024, FA3 can be installed via the following commands: ...
</li>
</ul>

</div>
  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1288947311778599004)** (15 messages🔥): 

> - `Langtrace integration with DSPy`
> - `MIPROv2 compilation runs`
> - `Experiment tracking issues` 


- **Langtrace Enhances DSPy Experiment Management**: Langtrace now supports running DSPy experiments with automatic capture of **traces**, **checkpoints**, and **eval score visualizations**, greatly enhancing pipeline management.
   - One user has found it useful to create individual projects for each pipeline block, allowing for targeted optimizations and effortless deployment of checkpointed prompts.
- **MIPROv2 Compilation Runs Encounter Issues**: A user reported issues with tracking evaluation data in their MIPROv2 compilation runs despite seeing evaluation traces in the logs, indicating a possible misconfiguration in their setup.
   - After troubleshooting with suggestions from other users, it was discovered that proper attributes needed to be passed with the `compile()` call.
- **Long Experiment Names Cause Logging Failures**: Another user found that using excessively long experiment names resulted in no traces being logged under their experiment, while shorter names worked without issues.
   - The discussion led to a realization about potential limits on experiment name lengths affecting logging functionality.



**Link mentioned**: <a href="https://docs.langtrace.ai/supported-integrations/llm-frameworks/dspy#dspy)">DSPy - Langtrace AI Docs</a>: no description found

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1288939490706194462)** (38 messages🔥): 

> - `BootstrapFewshot Page Availability`
> - `New LM with Azure's OpenAI APIs`
> - `DSPy Optimization Tools`
> - `Nesting Signatures in DSPy`
> - `Building DSPy Analytics Pipeline` 


- **BootstrapFewshot Explanation Page Delay**: A user inquired about the availability of the [BootstrapFewshot explanation page](https://link.to.page), which is currently unavailable.
   - Another member couldn't specify which page was being referenced, indicating some confusion around the request.
- **Transition to New LM with Azure's APIs**: A user transitioning to the new LM noted issues with API path construction in litellm, resulting in errors during use.
   - After a brief struggle, they reported that the upgrade resolved previous parsing errors associated with Predict.
- **DSPy Optimization Tooling Insights**: A new user expressed curiosity about DSPy's optimization tools similar to Tensorboard for tracking metrics in AI workflows.
   - Members discussed existing tools including the [DSPy Visualizer](https://link.to.visualizer) and additional support from Langtrace.
- **Nesting Signatures in DSPy**: A user asked about nesting Signatures to pass input/output field values, but was informed that it's not possible in that structure.
   - Alternative approaches using TypedPredictors with Pydantic were suggested, leading to further requests for example code to clarify the concept.
- **Designing a DSPy Analytics Pipeline**: One user detailed their approach to designing an analytics pipeline in DSPy and asked for validation of their method.
   - The response encouraged starting simple and iteratively refining the flow, sharing documentation that outlines effective practices for building with DSPy.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://dspy-docs.vercel.app/docs/building-blocks/solving_your_task">Using DSPy in 8 Steps | DSPy</a>: Using DSPy well for solving a new task is just doing good machine learning with LMs.</li><li><a href="https://x.com/karthikkalyan90/status/1839395049936953362">Tweet from Karthik Kalyanaraman (@karthikkalyan90)</a>: An example of how I think about building and optimizing compound AI pipelines with DSPy and CrewAI . Strongly believe this is how compound AI systems optimized for high performance and reliability wil...</li><li><a href="https://github.com/stanfordnlp/dspy/pull/1546">Make output processing in typed Pred compatible to LM Module by krypticmouse · Pull Request #1546 · stanfordnlp/dspy</a>: Main issue what the output was a BaseModel already so no need to create one, added:         if isinstance(dsp.settings.lm, dspy.LM):             parsing_result = output         else:             pa...
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1289045789858988052)** (8 messages🔥): 

> - `DSPy ReAct agents`
> - `RAG agents integration`
> - `Multiple RAG tools`
> - `Vector databases integration`
> - `Multimodal RAG optimization` 


- **Learning about DSPy ReAct agents**: A member asked for examples of using **DSPy ReAct agents** and expressed interest in integrating them with a **LlamaIndex retriever** for ReAct RAG.
   - Another member noted that there are examples in the **repo (examples/agents/)** and promised a better example soon.
- **Interest in RAG agent use cases**: The original poster shared a preference for **RAG agents** and suggested that examples should include **DSPy integrations** with various retrievers like vector DBs and knowledge graphs.
   - They also expressed curiosity about wrapping retrievers as **DSPy.tool instances** and the potential for **several RAG tools** to be accessible to the LM.
- **Feature requests for DSPy agents**: A member put forth feature requests for integrations with more **vector databases** like **Qdrant** and **LanceDB**, highlighting their trend towards hybrid search capabilities.
   - They also proposed the idea of **multimodal RAG pipeline optimization**, which another member confirmed is forthcoming.
- **Discussion on multiple RAG tools**: The idea of utilizing **multiple RAG tools** received positive feedback from members, as they recognized the complexity of tool selection for LLMs.
   - One member noted the potential of **DSPy optimization** to address the challenges associated with selecting the correct tools.


  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1288965754682736701)** (2 messages): 

> - `Mojo MAX desktop backgrounds`
> - `Emoji Voting` 


- **Poll for Mojo MAX Desktop Backgrounds**: A member initiated a poll asking if others would be interested in **Mojo / MAX branded desktop backgrounds** featuring adorable Mojo flames and MAX astronauts.
   - Participants were encouraged to **emoji vote** with options for yes or no.
- **User Reaction to Poll**: Another member expressed their reaction to the poll with a simple remark, stating, *'Bruh'*, indicating surprise or disinterest.
   - This suggests mixed feelings about the proposal for themed backgrounds.


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1289203510453473452)** (1 messages): 

> - `Verification Requirements`
> - `Posting Restrictions` 


- **New Verification Requirement to Post**: Verification is now required to post in any channels except <#1149739720146952292>, <#1238540905129054350>, and <#1212827673257316453>.
   - Members are encouraged to visit <#1098713770961944628> for verification, with a quick demo GIF available in the earlier post.
- **Restrictions on Posting Channels**: Members will face restrictions on posting in certain channels unless they complete verification.
   - This change is aimed at ensuring better control over channel participation and enhancing community engagement.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1288938665523023902)** (58 messages🔥🔥): 

> - `Error handling in Mojo`
> - `Improvements to Variant type`
> - `Sum types in programming languages`
> - `Mojo documentation needs`
> - `Pattern matching and exhaustiveness checking` 


- **Error messages in Mojo not referencing user code**: A member pointed out that their code is not mentioned in the error messages, which instead rely on standard library implementations.
   - Another member noted the limitations in improving these messages in the short term given their current implementation.
- **Proposal to evolve Variant into safe tagged union**: A member plans to propose evolving the `Variant` type into a *safe* tagged union, allowing for better pattern matching.
   - Discussion revolved around ensuring that this change integrates well with existing traits and pattern matching expectations.
- **Desire for proper sum types**: Members expressed their desire for proper sum types akin to Rust or Swift enums, highlighting their efficiency in message-passing systems.
   - One member noted the ergonomic advantages of using simple variants, contrasting them with polymorphic variants that introduce complexity.
- **Call for improved Mojo documentation**: Members agree on the need for better public documentation on Mojo and MLIR dialects due to existing user confusion.
   - The absence of documentation has led to improper usage of constructs, hindering development efforts.
- **Exhaustiveness checking and type inference**: Discussion highlighted the importance of exhaustiveness checking in systems design, enabling safer refactoring practices.
   - Concerns were raised about the pitfalls of relying solely on type inference, which can lead to unexpected type clashes.



**Link mentioned**: <a href="https://github.com/VitWW/rfcs/blob/partial_types3/text/0000-partial_types.md">rfcs/text/0000-partial_types.md at partial_types3 · VitWW/rfcs</a>: RFCs for changes to Rust. Contribute to VitWW/rfcs development by creating an account on GitHub.

  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1288990792093929577)** (20 messages🔥): 

> - `FTC Crackdown on AI Tool Claims`
> - `Concerns About Generative AI Sustainability`
> - `Geohot's Frustration with AMD`
> - `ColPali Model with Qwen2-VL`
> - `Effectiveness of AI in Software Development` 


- **FTC Crackdown Targets Misleading AI Claims**: The FTC announced a crackdown on deceptive claims regarding AI tools, highlighting potential fraud cases around companies like **Do Not Pay** due to misleading marketing practices, as detailed in their [complaint PDF](https://www.ftc.gov/system/files/ftc_gov/pdf/DoNotPayInc-Complaint.pdf).
   - Discussion included skepticism about the FTC's definition of AI and concerns that many startups might fall under scrutiny as a result of these actions.
- **Skepticism About Sustainability of Generative AI**: An article argues that the current boom in generative AI is unsustainable, predicting a potentially catastrophic collapse that could harm big tech and the startup ecosystem, as elaborated in the [newsletter](https://www.wheresyoured.at/subprimeai/?ref=ed-zitrons-wheres-your-ed-at-newsletter).
   - Critics pointed out a perceived lack of understanding of technology and use cases in the argument, noting that tools like GitHub Copilot have demonstrated clear business value.
- **Geohot Expresses Frustration with AMD**: Geohot shared feelings of demotivation toward continued work with AMD, expressing skepticism about the company's future after recognizing no significant chips are planned post-RDNA3.
   - This sentiment reflects a broader concern about lack of motivation in the community due to perceived stagnation in advancements at AMD.
- **Excitement Over New ColPali Model Release**: The community celebrated the launch of a new model, **ColQwen2**, which improves accuracy and efficiency by leveraging a **Qwen2-VL** backbone, leading to substantial performance gains over prior iterations.
   - This model is seen as a significant step in visual recognition capability, marking a notable improvement in the Vidore Leaderboard performance.
- **Debate on AI's Business Value**: There is a growing debate about the true business value of generative AI, with some voices claiming it lacks effectiveness, while others cite evidence of substantial value generation in software engineering use cases like GitHub Copilot.
   - Studies indicate that tools have saved companies like Amazon hundreds of millions annually, prompting calls for a more nuanced discussion around the actual impact and application of AI technologies.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/simonw/status/1839030384949854642">Tweet from Simon Willison (@simonw)</a>: Looks like the FTC have a definition of AI which they are extremely confident about  Quoting Charlie Dolan (@cdolan92)   FTC announced AI related crackdowns  Hot Take: Good! There&#39;s a lot of scams...</li><li><a href="https://x.com/__tinygrad__/status/1839221471182512632?s=46">Tweet from the tiny corp (@__tinygrad__)</a>: @AMD After realizing there&#39;s no big chips after RDNA3, we sort of lost motivation to keep working on it. From our perspective there&#39;s no real future in AMD.  I know it shoots us in the foot fo...</li><li><a href="https://www.wheresyoured.at/subprimeai/?ref=ed-zitrons-wheres-your-ed-at-newsletter">The Subprime AI Crisis</a>: None of what I write in this newsletter is about sowing doubt or &quot;hating,&quot; but a sober evaluation of where we are today and where we may end up on the current path. I believe that the artifi...</li><li><a href="https://www.latent.space/p/mar-jun-2024">The Winds of AI Winter</a>: Mar-Jun 2024 Recap: People are raising doubts about AI Summer. Here&#x27;s why AI Engineers are the solution.</li><li><a href="https://x.com/ggerganov/status/1839703977073487993">Tweet from Georgi Gerganov (@ggerganov)</a>: Yes, http://ggml.ai will be receiving a revenue share from all llama.cpp-powered endpoints used on HF. So for anyone who wants to support us, make sure to give those endpoints a try ♥️  Quoting swyx.a...</li><li><a href="https://www.wheresyoured.at/subprimeai/?ref=ed-zitrons-wheres-your-ed-at-newsl">The Subprime AI Crisis</a>: None of what I write in this newsletter is about sowing doubt or &quot;hating,&quot; but a sober evaluation of where we are today and where we may end up on the current path. I believe that the artifi...</li><li><a href="https://greaterdanorequalto.com/ai-code-generation-as-an-agent-of-tech-debt-creation/">AI code generation as an agent of tech debt creation</a>: I&#x27;ve disabled all LLM-based AI Assistants/Copilots/whatever-you-call-&#x27;ems in my IDE.</li><li><a href="https://x.com/garrisonlovely/status/1839655744850772272?s=46">Tweet from Garrison Lovely (@GarrisonLovely)</a>: This article is full of bombshells. Excellent reporting by @dseetharaman.   The biggest one: OpenAI rushed testing of GPT-4o (already reported), released the model and then subsequently determined the...</li><li><a href="https://x.com/jobergum/status/1839667559404093658?s=46">Tweet from Jo Kristian Bergum (@jobergum)</a>: Our prayers have been heard.  A ColPali model, but with a Qwen2-VL backbone! This is huge because   - It lifts accuracy (huge gain)  - Fewer patch vectors, more efficient  - Permissive licensed  Quoti...</li><li><a href="https://www.ftc.gov/news-events/news/press-releases/2024/09/ftc-announces-crackdown-deceptive-ai-claims-schemes">FTC Announces Crackdown on Deceptive AI Claims and Schemes</a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1289315495488786433)** (38 messages🔥): 

> - `AI Engineering Interviews`
> - `Screen Sharing Issues`
> - `Using Local Models`
> - `Braintrust JSON Mode`
> - `Cot Experimentation` 


- **Excitement over AI Engineering Interviews**: A member expressed happiness about an interview that could lead to a role in **AI Engineering**.
   - *“Had an interview that could transition into an AI Engineering role so me be happy.”*
- **Struggles with Screen Sharing**: Multiple members experienced issues with screen sharing, with suggestions to reload or switch platforms provided.
   - *“Leaving coming & back in worked for me...”* highlighted different methods tried to resolve ongoing loading struggles.
- **Local Model Discussions**: A question was raised about whether creating a **local model** would enhance functionality, expressing intrigue about its potential benefits.
   - *“Do you think Making a local model to do such things would improve?”*
- **Questions on Braintrust Capabilities**: A member asked about **Braintrust** and if it supports a **JSON mode**, seeking clarity on its integrations.
   - *“I'm not familiar with braintrust, do they allow for a json mode?”* sparked a discussion about constraints and flexibility.
- **Experimenting with Cot Techniques**: Members shared playful comments about using **chain-of-thought (COT)** techniques and discussed overall experimentation experiences.
   - *“rare cot L 😂”* reflected a light-hearted view on the complexities of model experimentation.


  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1288995177469444096)** (4 messages): 

> - `Paragon integration`
> - `Langfuse and PostHog tutorial`
> - `LlamaIndex with Box`
> - `FinanceAgentToolSpec`
> - `RAG and LlamaIndex` 


- **Paragon Builds a Feature-Packed Chatbot**: A blog post and video from [useparagon](https://t.co/KEE2LOnGoR) illustrate their use of create-llama from LlamaIndex to create a chatbot that interfaces with customer data from **Slack**, **Google Drive**, and **Notion**.
   - *It ingests data continuously and in real time,* making the integration highly effective.
- **Langfuse and PostHog Enhance MistralAI**: A tutorial shared in a [Jupyter notebook](https://t.co/KGxjjoO0vM) explains how to set up **Langfuse** for tracking LLM applications and integrates **PostHog** for user analytics.
   - This setup enables comprehensive **monitoring** and **analytics** for AI applications, streamlining the development process.
- **LlamaIndex and Box Collaboration Revealed**: Alex Novotny and **@seldo** discuss the integration of **LlamaIndex** with **Box**, highlighting features like **LlamaParse**, **LlamaCloud**, and **LlamaHub** in a recent chat.
   - They also delve into how **RAG** works and what users should consider with this integration. Check the video [here](https://t.co/KL0kkDTY65).
- **FinanceAgentToolSpec Unlocks Financial Data**: The **FinanceAgentToolSpec** package on **LlamaHub** allows agents to query public financial data from sources including **Polygon**, **Finnhub**, and **Seeking Alpha**.
   - Hanane's post elaborates on how this package proves useful for doing financial analysis with LlamaIndex. Read more [here](https://t.co/7bsEm4Er1m).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1288980334410469429)** (28 messages🔥): 

> - `NLTK Resource Issue`
> - `Loading Fine-tuned Models on GPU`
> - `Best Open Source Vector Database`
> - `Self-hosted Observability Tools`
> - `Vector Search Optimization Strategy` 


- **NLTK's punkt resource missing**: A user reported encountering a *Resource punkt not found* error while using **NLTK**. Another member suggested checking the version of **llama-index** as the latest versions utilize *punkt_tab*.
   - *Resource-related issues* with NLTK's punkt were mentioned, hinting at potential compatibility concerns.
- **Challenges with loading fine-tuned Llama3.1-8B**: A user struggled to load their locally fine-tuned **Llama3.1-8B** for the Text2SQL task onto their GPU. Members recommended manually loading the model and tokenizer, ensuring it was on the GPU during initialization.
   - A detailed code snippet was shared, illustrating how to set up the model using quantization for optimized performance.
- **Discussion on open-source vector databases**: A user inquired about the best open-source vector database for applications without advanced retrieval mechanisms. They emphasized the need for options that still function well in simpler setups, particularly for their advanced RAG application.
   - They shared a link to a video related to vector databases, likely seeking further insights from the community.
- **Self-hosted observability tools recommendations**: Recommendations for self-hosted observability tools included **Arize Phoenix**, which provides frameworks for tracing LlamaIndex applications. Another option mentioned was **LlamaIndex's in-house instrumentation** capabilities for observability.
   - A user expressed a desire for solutions that work easily with Docker, highlighting the need for simplified setup processes.
- **Optimizing vector search for customer support**: A proposed strategy for optimizing vector search involved storing questions in the vector chunk while keeping answers in metadata. This method aimed to enhance accuracy by focusing on question semantics during searches.
   - The user sought validation of this strategy and welcomed suggestions for further improvements to their approach.



**Link mentioned**: <a href="https://docs.llamaindex.ai/en/stable/examples/cookbooks/oreilly_course_cookbooks/Module-5/Observability/">Observability - LlamaIndex</a>: no description found

  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1289248624173256787)** (11 messages🔥): 

> - `OpenAI's rushed GPT-4o release`
> - `Safety staff challenges`
> - `Employee compensation demands`
> - `Leadership turnover`
> - `Talent recruitment efforts` 


- **OpenAI rushed GPT-4o release amid competition**: Executives aimed to debut **GPT-4o** ahead of Google's developer conference, leading to a hasty release despite concerns. The decision was reportedly based on **incomplete safety data**, which later indicated the model was too risky to deploy.
   - An article by [@dseetharaman](https://x.com/garrisonlovely/status/1839655744850772272?s=46) highlights how safety staff worked **20-hour days**, leaving little time for thorough checks.
- **OpenAI faces internal grievances over compensation**: [The Information's article](https://www.theinformation.com/articles/behind-openais-staff-churn-turf-wars-burnout-compensation-demands) reveals persistent compensation demands as OpenAI's valuation rises. Employees have reportedly cashed out **over $1.2 billion** from selling profit units in recent years.
   - Leaders, including new CFO **Sarah Friar**, are grappling with researchers threatening to quit over financial concerns amid growing competition for talent.
- **Leadership turnover impacts OpenAI's stability**: An ongoing leadership turnover has been linked to the **demand for increased compensation** among key researchers. This has created tension as leaders negotiate to retain talent amidst competing offers from startups like **Safe Superintelligence**.
   - These frustrations and recruitment efforts have exacerbated the talent crisis, prompting OpenAI leaders to extend generous counteroffers to keep their researchers.



**Link mentioned**: <a href="https://x.com/garrisonlovely/status/1839655744850772272?s=46">Tweet from Garrison Lovely (@GarrisonLovely)</a>: This article is full of bombshells. Excellent reporting by @dseetharaman.   The biggest one: OpenAI rushed testing of GPT-4o (already reported), released the model and then subsequently determined the...

  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1288980447895883786)** (15 messages🔥): 

> - `OpenAI leadership changes`
> - `Public statements of employees`
> - `AI culture differences`
> - `Emotional responses in tech`
> - `Gamer culture terms` 


- **OpenAI faces leadership transitions**: A heartfelt farewell from a team member highlights the departures of key leaders **Mira, Bob, and Barret**, equating it to the resilience of parents in the Middle Ages facing loss.
   - Despite its imperfections, OpenAI is praised for its talented workforce, and the departing leaders are wished well on their future endeavors.
- **Employees share drama publicly**: Members jokingly noted how OpenAI employees publicly express internal drama, likening it to royal family PR statements.
   - One intern humorously lamented their resignation, comparing their time at OpenAI to cherishing a newborn, raising questions about the culture of transparency.
- **Contrast of AI culture versus other sectors**: Discussion emerged regarding the distinct culture in AI, with comments about high **entitlement** in the AI community.
   - This led to conclusions about how uniquely expressive AI workers are compared to those in traditional tech companies.
- **Gamer culture referenced in conversation**: A member humorously pointed out the cultural differences, relating AI employee public expressions to gamer culture with the term 'tilted'.
   - This sparked a conversation on whether such terminology is widely recognized outside gamer circles.
- **Anthropic remains stable amid OpenAI's changes**: In contrast to OpenAI, it was noted that all co-founders of **Anthropic** are still with the company post-departures at OpenAI.
   - This adds a layer of stability which is juxtaposed against the emotional turmoil seen in the leadership changes at OpenAI.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/woj_zaremba/status/1839696945008582672?s=46">Tweet from Wojciech Zaremba (@woj_zaremba)</a>: It’s sad to see Mira, Bob, and Barret go—not only because they are excellent leaders but also because I will miss seeing them day to day. They are my friends.  Their departures made me think about the...</li><li><a href="https://www.urbandictionary.com/define.php?term=tilted>)">Urban Dictionary: tilted&gt;)</a>: When you&apos;re getting slightly mad at someone/something.</li><li><a href="https://x.com/sashadem/status/1839728129935540589?s=46">Tweet from Sasha de Marigny (@sashadem)</a>: Happy to report that Anthropic’s co-founders all still merrily work at the company. None have been lost to Middle Age plagues or jacuzzis.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1289294779179470879)** (4 messages): 

> - `Substack Best Seller`
> - `Apple App Store Management` 


- **Substack achieves iPhone IAP subscriptions**: As a **Substack best seller**, there is new access to **iPhone In-App Purchase subscriptions**.
   - This opportunity highlights the potential for growth in digital publishing on mobile platforms.
- **Behind the scenes of the Apple App Store**: Discussions reveal interesting insights about managing the **Apple App Store**, often seen as a **horror show** for app developers.
   - Members expressed intrigue over the **behind the scenes implementations** that tackle these challenges.


  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/)** (1 messages): 

420gunna: https://x.com/venturetwins/status/1839685317462458650
Instalocking this
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1289000758473265194)** (7 messages): 

> - `Multimodal support`
> - `Area Chair roles`
> - `Conversation splitting in training` 


- **Open Source Community Lags in Multimodal Support**: A member highlighted that the **open-source community** is falling behind in adopting **multimodal support**, while the broader industry shifts in that direction.
   - This sentiment reflects a growing concern about the speed of **innovation** in the community.
- **Understanding Area Chair Roles**: A member explained that **AC** refers to a meta reviewer known as an **area chair**, who plays a critical role in the review process.
   - This insight underscores the importance of organization in academic and collaborative environments.
- **Python Snippet for Conversation Handling**: A user presented a Python snippet aimed at **splitting conversations** for training purposes, ensuring conversations do not exceed **maximum sequence length**.
   - They emphasized its utility, particularly for handling long conversations while retaining context in training datasets.
- **Enhancements for Conversation Preprocessing**: Another member suggested implementing checks for **message length** before appending to conversation segments, ensuring data integrity.
   - They emphasized the potential usefulness of this feature within the **preprocessing** pipeline for certain datasets.


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1288959433853898814)** (19 messages🔥): 

> - `Multi-modal VLM Assistance`
> - `YAML Configuration Issues`
> - `Flex Attention Discussion`
> - `LoRA+ Optimization Update`
> - `Default Learning Rates in LoRA+` 


- **Seeking Help on Multi-modal VLM Challenges**: A member expressed frustration with their current **multi-modal** setup, saying **pre-processing** data isn't optimal.
   - They invited others to contribute, suggesting collaborative problem-solving could help move forward.
- **Confusion Around YAML Configuration for Learning Rates**: Multiple members discussed specific parameters in their YAML configurations, including `loraplus_lr_ratio` and `loraplus_lr_embedding`.
   - One member noted they needed a complete YAML file to troubleshoot their reproduction issues.
- **Introducing Flex Attention for Optimization**: A member highlighted **Flex Attention** as a new optimized implementation that provides flexibility compared to previous attention methods.
   - Several resources were shared, including a [link to the PyTorch blog](https://pytorch.org/blog/flexattention/) detailing its design.
- **Update on LoRA+ Optimization Fixes**: A member requested the setting of `loraplus_lr_embedding` to a specific value, referencing a [fix in a recent GitHub PR](https://github.com/axolotl-ai-cloud/axolotl/pull/1932).
   - They explained that the fix was essential due to the failure to use a default value for this parameter.
- **Discussion on Default Learning Rates in LoRA+**: Members questioned whether to stick with the default learning rate for `loraplus_lr_embedding` or match it to their main learning rate.
   - They noted that the **LoRA+** paper used `1e-6` for their primary learning rate, which could explain the default setting.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/blog/flexattention/">FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention</a>:   </li><li><a href="https://github.com/meta-llama/llama-recipes/blob/8fc300b747aa09e09ab80be0b11ab70726985e26/src/llama_recipes/finetuning.py#L226-L245">llama-recipes/src/llama_recipes/finetuning.py at 8fc300b747aa09e09ab80be0b11ab70726985e26 · meta-llama/llama-recipes</a>: Scripts for fine-tuning Meta Llama with composable FSDP &amp;amp; PEFT methods to cover single/multi-node GPUs. Supports default &amp;amp; custom datasets for applications such as summarization and Q&...</li><li><a href="https://github.com/axolotl-ai-cloud/axolotl/pull/1932">fix for empty lora+ lr embedding by winglian · Pull Request #1932 · axolotl-ai-cloud/axolotl</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/)** (1 messages): 

invisietch: Fp8 runs on 2x 80GB A100 for me, should be fine on 2x H100 also
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1289009015283519590)** (21 messages🔥): 

> - `Nvidia P2P Support and IOMMU`
> - `Pricing and Competition in GPU Cloud Services`
> - `CLOUD=1 Service Details`
> - `Data Upload Challenges for Training`
> - `Persistent Storage Billing` 


- **IOMMU's Role in Nvidia P2P**: A user inquired why IOMMU needs to be turned off for Nvidia P2P support while using the [tinygrad GPU modules](https://github.com/tinygrad/open-gpu-kernel-modules), seeking clarity on its interaction.
   - There was no immediate response, indicating a need for further insights on the technical interaction.
- **GPU Cloud Pricing Competition Sparks Discussion**: George Hotz proposed a rate of **$0.50/hr for GPUs**, billed by the second, prompting comparisons with cheaper options from salad.com, vast.ai, and runpod.
   - Concerns were raised about VAT implications and whether this pricing accounts for tax-covered costs by competitors.
- **CLOUD=1 Features Debated**: Discussion ensued regarding whether **CLOUD=1** includes CPU resources or solely GPUs, with users expressing discomfort about needing to keep their devices online.
   - Participants feel that beyond price savings, better solutions are necessary to justify the service's structure.
- **Challenges with Data Upload for ML Tasks**: A member highlighted that connecting and uploading large datasets is a significant pain point, expressing hope that tinygrad could streamline this.
   - The challenge lies in the data-compute ratio which could affect efficiency for smaller models like **mini LLMs and CNNs**.
- **Considerations on Persistent Storage Costs**: A user questioned whether there are plans to address **persistent storage billing**, noting that many providers charge separately for this.
   - This concern reflects broader apprehensions about the overall cost structure in cloud computing services.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1289103130667126835)** (4 messages): 

> - `Pull Request #6779`
> - `Device Loading Issues`
> - `PR Comparison` 


- **Pull Request #6779 Submitted**: A user submitted their first attempt at a PR titled [get_available_backends](https://github.com/tinygrad/tinygrad/pull/6779) stating it may not be as concise as desired since George preferred a single line implementation.
   - They asked for feedback on areas to investigate further for improvement.
- **Unnoticed Competing PR**: The user realized another PR had been submitted that they were unaware of, which seemed better than their own attempt.
   - *Bummer* for them, as they identified improvements they could have made to their own PR.
- **George Critiques Existing PRs**: George commented that the competing PR isn't good either, indicating ongoing concerns regarding quality.
   - The user speculated whether the issue stemmed from `Device.DEFAULT` loading every device, signaling potential issues in the current implementation.



**Link mentioned**: <a href="https://github.com/tinygrad/tinygrad/pull/6779">get_available_backends for device by i-jared · Pull Request #6779 · tinygrad/tinygrad</a>: Try loading each backend. Return any that load successfully. It&#39;s not 1 line like George wanted #6689, but it works and follows conventions existing in the codebase.

  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1288940590301446184)** (15 messages🔥): 

> - `Llama 3.2 11B Vision`
> - `Voice Cloning`
> - `Family Photo Generation`
> - `Copyright Enforcement`
> - `Maintaining Independence` 


- **Llama 3.2 11B Vision available for free**: TogetherCompute partnered with @AIatMeta to offer **Llama 3.2 11B Vision** for free so developers can experiment with open-source multimodal AI, and unlimited access can be found [here](https://api.together.ai/playground/chat/meta-llama/Llama-Vision-Free).
   - For better performance, they also provide paid Turbo endpoints for Llama 3.2 11B & 90B.
- **Discussion on Unlimited Access**: Members discussed the implications of unlimited access to Llama 3.2, suggesting it could be used to caption the entire **LAION dataset** humorously.
   - This sparked a light-hearted suggestion for community involvement.
- **Voice Cloning Conversation**: A member humorously mentioned talking to a voice clone of themselves, adding a light-hearted feel to the discussion.
   - This sparked engagement and laughter among the members.
- **Concerns about Photo Generation Apps**: A member inquired about the effectiveness of a specific app for generating **family photos**, showing interest in AI-powered solutions.
   - This indicates a growing interest in AI's capabilities in creating personal and user-specific content.
- **Victory in Copyright Enforcement**: A member shared a LinkedIn post celebrating the recent win in copyright enforcement, suggesting that **the good guys won this round**.
   - This was emphasized as a victory for integrity and independence within the community.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/togethercompute/status/1839071026728333778">Tweet from Together AI (@togethercompute)</a>: 🚀 We&#39;ve partnered with @AIatMeta to offer Llama 3.2 11B Vision for FREE so developers can experiment with open-source multimodal AI at no cost.  In addition to our free credits, for a limited tim...</li><li><a href="https://smallpdf.com/result#r=a30cd403fcb0a6c119f2933a843cfe07&t=share-document?trk=feed-detail_comments-list_comment-text">Smallpdf.com</a>: no description found
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1289070350336000073)** (8 messages🔥): 

> - `Positional Information in CNNs`
> - `Positional Encoding in Transformers`
> - `Scaling Laws in Machine Learning`
> - `Fourier Feature Extraction`
> - `Trends in Neural Network Architectures` 


- **Confusion Over Latent Pixel Positioning**: A member expressed confusion about how position information is incorporated into the feature vector of a latent pixel, noting the absence of positional encoding despite their role in cross-attention with CLIP text embeddings.
   - Another member pointed out that self-attention steps in models also contribute to this process, emphasizing that convolution edges yield positional data for attention comparisons.
- **CNNs Implicitly Learn Positional Information**: Discussion highlighted a paper that discusses how CNNs achieve efficiency by using local filters, while also noting the importance of absolute position information that may be implicitly learned.
   - The paper suggests that padding techniques, including zero-padding, help in delivering position information that contributes to learning in convolutional layers.
- **Scaling Laws Impact on Machine Learning Principles**: A member recommended reading a paper that revisits machine learning principles in light of scaling laws, indicating a shift from minimizing generalization error to reducing approximation error.
   - This paper challenges the validity of certain regularization principles in the context of large language models, highlighting a phenomenon called 'scaling law crossover.'
- **The Future: A Singular Transformer Model?**: One member humorously concluded that the future of neural networks might just lead to 'one big transformer'.
   - This comment reflects a broader trend in simplification of architectures as advancements continue in transformer models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2409.15156">Rethinking Conventional Wisdom in Machine Learning: From Generalization to Scaling</a>: The remarkable success of large language pretraining and the discovery of scaling laws signify a paradigm shift in machine learning. Notably, the primary objective has evolved from minimizing generali...</li><li><a href="https://arxiv.org/abs/2001.08248">How Much Position Information Do Convolutional Neural Networks Encode?</a>: In contrast to fully connected networks, Convolutional Neural Networks (CNNs) achieve efficiency by learning weights associated with local filters with a finite spatial extent. An implication of this ...</li><li><a href="https://emu.baai.ac.cn/about">Emu3</a>: no description found
</li>
</ul>

</div>
  

---



### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1288959843440529544)** (16 messages🔥): 

> - `Lecture Coverage on Social Alignment`
> - `Course Enrollment Confirmation`
> - `Assignment Deadlines and Clarifications`
> - `Qquiz Availability`
> - `Lab Assignment Release Timing` 


- **Lectures Expected to Cover Social Alignment**: A concern was raised about whether the lectures would address **social alignment** of LLM agents, as the last two seem focused on **AI safety**.
   - *Prof Dawn Song's research will likely touch on these themes* during her talk on **LLM Safety** scheduled for **December 2**.
- **Course Sign-Up Process Clarified**: A user inquired whether they would receive confirmation after signing up for the course using the provided Google form link.
   - Another posted that filling out the sign-up form grants access to all course materials, with deadlines for assignments noted as **December 12**, 2024.
- **Assignment Deadlines Cause Confusion**: A participant sought clarification on assignment due dates, noting potential discrepancies between posted deadlines for Berkeley and MOOC students.
   - The confirmation was given that **all assignments** on the MOOC website are due on **December 12**, 2024, and quiz availability is confirmed on the site.
- **Confusion Over Qquiz 3 Availability**: A participant expressed difficulty finding **Qquiz 3**, wondering if it had been removed.
   - Others clarified that it remains accessible on the **MOOC students' website**, not the Berkeley students' site.
- **Inquiry on Lab Assignment Release**: A participant following up on the lab assignment sought information on its release schedule after reading through the MOOC website.
   - This indicates ongoing discussions about the course structure and timeline, emphasizing clarity among students.



**Link mentioned**: <a href="https://llmagents-learning.org/f24">Large Language Model Agents</a>: no description found

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1289034521693651035)** (6 messages): 

> - `OpenInterpreter application`
> - `Multimodal support in LLaMA`
> - `Frontend development for OI`
> - `On-chain analytics demonstration` 


- **Transitioning to Final Solutions with OpenInterpreter**: A member demonstrated how to use [OpenInterpreter](https://x.com/parrotexplore/status/1839721139515302137) for on-chain analytics, showcasing a shift from code that's **probably working** to **fully functional code**.
   - They shared a [Google Colab](https://t.ly/vBSPe) link for the Python code, and appreciated any **reposts** from the community.
- **Multimodal Support Issue in LLaMA**: A discussion around the issue of bringing back **multimodal support** in the LLaMA project was noted, indicating that it was removed since **#5882**.
   - Updates depend on the refactoring of **llava**; this thread is mainly created for tracking progress with links to relevant issues.
- **Electrifying Frontend Development for OI**: A member expressed their excitement about creating an **Electron frontend** for OpenInterpreter, highlighting its cool factor.
   - The excitement indicates a positive reception of ongoing development efforts within the community.
- **Sharing Good Vibes in OpenInterpreter Community**: A member shared the OpenInterpreter application post with the broader **Open Interpreter X community**, emphasizing its value.
   - This sharing spirit highlights an encouraging environment among those involved in OpenInterpreter's development.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://t.ly/vBSPe">Google Colab</a>: no description found</li><li><a href="https://x.com/parrotexplore/status/1839721139515302137">Tweet from Parrot Explorator (@parrotexplore)</a>: Fantastic @OpenInterpreter can be used to transition from &#34;ChatGPT giving us code that&#39;s probably working&#34; to &#34;getting a final solution with code that achieves it.&#34;   Here’s a demo...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/8010#issuecomment-2345831496">server: Bring back multimodal support · Issue #8010 · ggerganov/llama.cpp</a>: Multimodal has been removed since #5882 Depends on the refactoring of llava, we will be able to bring back the support: #6027 This issue is created mostly for tracking purpose. If someone want to t...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/8010#issuecomment-2376339571">server: Bring back multimodal support · Issue #8010 · ggerganov/llama.cpp</a>: Multimodal has been removed since #5882 Depends on the refactoring of llava, we will be able to bring back the support: #6027 This issue is created mostly for tracking purpose. If someone want to t...
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1289301733473914972)** (7 messages): 

> - `Decoding Packet Error`
> - `Server Connection Issues`
> - `Request for Setup Information` 


- **Decoding Packet Error Notification**: A user reported a warning message indicating an **Error decoding packet** due to *invalid data found* when processing input with specific process and job IDs.
   - This issue appears consistently every time the server restarts or a client connection is attempted without any other terminal errors.
- **Phone Stuck on 'Starting...' Screen**: Another user described their phone being stuck on the **'Starting...'** page while trying to connect.
   - This persistent issue raises further questions about the system's connection stability.
- **Request for Detailed Setup Information**: A community member advised the user to create a post detailing their **setup** (OS, steps to reproduce) and to include a longer print of the terminal output.
   - They suggested sharing this information in a specified channel for better assistance and troubleshooting.


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1289152581037326452)** (3 messages): 

> - `HF 90b vision update`
> - `Impact of OpenInterpreter` 


- **HF introduces 90b vision model**: A member announced that **HF** has been updated with a **90b vision** model, which is now available for use.
   - This update is anticipated to enhance various vision-related tasks significantly.
- **OpenInterpreter transforms lives**: A member shared their experience, stating that **OpenInterpreter** has completely changed their life, enabling them to make incredible friends and dive deep into the A.I. world.
   - They expressed gratitude for the community and emphasized their mission to accelerate **open source tech** while quoting their viral demo from a year ago.



**Link mentioned**: <a href="https://x.com/MikeBirdTech/status/1839750338179674590">Tweet from Mike Bird (@MikeBirdTech)</a>: One year ago today, I made a little demo of this cool new tool I found online. Just wanted to show off what it could do and then it went a little viral  Since then @OpenInterpreter has completely chan...

  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1289179389874933853)** (2 messages): 

> - `Vector search optimization`
> - `Contextual extraction from Excel` 


- **Optimizing vector search for customer support**: A proposed strategy for optimizing vector search involves storing questions in the vector chunk and answers in metadata with a prompt, aimed at improving **precision** for question matches.
   - In this approach, the focus remains on the **semantics** of questions, potentially leading to more accurate search results by reducing irrelevant information.
- **Challenges with contextual extraction from complex Excel files**: A member expressed difficulty in finding effective methods for **contextual extraction** from complex Excel files that would enable meaningful responses from LLMs.
   - Despite extensive searching, they have yet to discover a viable solution to facilitate this process.


  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1289085277553557557)** (2 messages): 

> - `CF Booking Chatbot`
> - `Unize Storage AI System`
> - `Knowledge Graph Generation`
> - `Google Calendar Integration`
> - `LangChain Performance Comparison` 


- **CF Booking Chatbot simplifies conference room management**: A member posted about their newly built **CF Booking Chatbot** using LangChain that streamlines checking availability, booking, and managing conference rooms. The post includes a [demo video showcasing its features](https://www.linkedin.com/posts/ismile-bharmal-3b82241ab_langchain-chainlit-ai-activity-7245291326919872512-O06M) and mentions plans to integrate **Google Calendar** for automatic syncing.
- **Unize Storage promises enhanced graph generation**: Another member introduced an AI system called **Unize Storage**, which generates high-quality knowledge graphs from any input text. They highlighted its ability to outperform existing systems, including **LangChain's LLMGraphTransformer**, achieving **85% accuracy** compared to LangChain's **55%** on larger inputs.
- **Unize provides free API access for experimentation**: The **Unize API** offers users the opportunity to experiment with the new **Unize Storage** system and receive free API credits. They can visualize knowledge graphs and get started with [this Playground](https://api.unize.org/signup) designed for easy access.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.unize.org/p/introducing-unize-storage">Introducing Unize Storage</a>: An AI system to generate high-quality knowledge graphs at scale.</li><li><a href="https://developers.unize.org/kgstorage.">Introduction - Unize API</a>: no description found</li><li><a href="https://api.unize.org/signup">Unize</a>: no description found
</li>
</ul>

</div>
  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1288966136146296832)** (3 messages): 

> - `PackedDataset constraint`
> - `max_seq_len handling`
> - `RuntimeError in dataset processing`
> - `Error discussion on GitHub` 


- **Need for PackedDataset size enforcement**: A member proposed to enforce that the packed size cannot exceed **2x the dataset max length** to prevent errors when processing sequences.
   - This discussion emerged as a potential safeguard against runtime inconsistencies.
- **Simplified failure case in dataset**: It was demonstrated that the current implementation can fail even with a single input exceeding **max_seq_len**, particularly with configurations that create a mismatch in bounds.
   - A fix using explicit gating for token length was suggested to prevent these runtime errors.
- **Discussion on GitHub error link**: The conversation pointed to a [GitHub error](https://github.com/pytorch/torchtune/blob/main/torchtune/datasets/_packed.py#L130) that indicates a decision might have been made to allow sequences greater than **max_seq_len**.
   - This link potentially clarifies the reasoning behind the current handling of packed dataset sizes.
- **Mentions for further review**: A member suggested that another user should review the content of this discussion when they return, indicating its importance.
   - This highlights the collaborative nature of the troubleshooting process.



**Link mentioned**: <a href="https://github.com/pytorch/torchtune/blob/main/torchtune/datasets/_packed.py#L130">torchtune/torchtune/datasets/_packed.py at main · pytorch/torchtune</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.

  

---



### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1288953002731765780)** (1 messages): 

> - `Function Calling Evaluation`
> - `Customization of Evaluation Dataset`
> - `Integration with LLMs`
> - `Berkeley Function-Calling Leaderboard`
> - `Error Breakdown Analysis` 


- **Confusion Around Function Calling Evaluation in Codebase**: A user expressed confusion regarding the evaluation process of the codebase, questioning whether they could provide their own **evaluation dataset** for analysis.
   - They specifically sought a package capable of breaking down errors using a dataset with **<prompt>, <llm_response>, <ideal response>**.
- **Interest in Local LLM Deployment**: The user is interested in functionalities that allow for using a **locally deployed LLM** with their dataset to extract error metrics effectively.
   - They requested recommendations for other codebases that might handle such requirements, particularly in the context of **function calling capabilities**.
- **Exploration of LLM Integration in Applications**: The conversation highlighted the growing trend of integrating **Large Language Models (LLMs)** in various applications like Langchain and AutoGPT.
   - Models including **GPT, Gemini, Llama,** and **Mistral** were mentioned for their impressive function calling abilities in powering software solutions.
- **Berkeley Function-Calling Leaderboard as a Resource**: The user referenced the **[Berkeley Function-Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)** as a valuable resource for evaluating LLM function calling capabilities.
   - They noted that the leaderboard is built from user-centric function calling use cases, addressing various function call forms.



**Link mentioned**: <a href="https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html#metrics">Berkeley Function Calling Leaderboard</a>: no description found

  

---



### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/)** (1 messages): 

azaw: How do we use the openAI sdk for jamba ? is it possible ?
  


{% else %}


> The full channel by channel breakdowns have been truncated for email. 
> 
> If you want the full breakdown, please visit the web version of this email: [{{ email.subject }}]({{ email_url }})!
>
> If you enjoyed AInews, please [share with a friend](https://buttondown.email/ainews)! Thanks in advance!

{% endif %}
