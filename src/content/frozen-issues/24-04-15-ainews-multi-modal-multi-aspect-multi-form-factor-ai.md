---
id: b138e365-95ff-4f55-8811-5a76a5e17422
title: Multi-modal, Multi-Aspect, Multi-Form-Factor AI
date: '2024-04-15T22:42:55.173152Z'
original_slug: ainews-multi-modal-multi-aspect-multi-form-factor
description: >-
  Between April 12-15, **Reka Core** launched a new GPT4-class multimodal
  foundation model with a detailed technical report described as "full Shazeer."
  **Cohere Compass** introduced a foundation embedding model for indexing and
  searching multi-aspect enterprise data like emails and invoices. The
  open-source **IDEFICS 2-8B** model continues Google's Flamingo multimodal
  model reproduction. **Rewind** pivoted to a multi-platform app called
  Limitless, moving away from spyware. Reddit discussions highlighted **Apple
  MLX** outperforming **Ollama** and **Mistral Instruct** on M2 Ultra GPUs, GPU
  choices for LLMs and Stable Diffusion, and AI-human comparisons by Microsoft
  Research's Chris Bishop. Former PayPal CEO Dan Schulman predicted **GPT-5**
  will drastically reduce job scopes by 80%. **Mistral** CEO Arthur Mensch
  criticized the obsession with AGI as "creating God."
companies:
  - reka-ai
  - cohere
  - google
  - rewind
  - apple
  - mistral-ai
  - microsoft
  - paypal
models:
  - gpt-4
  - idefics-2-8b
  - mistral-instruct
  - apple-mlx
  - gpt-5
topics:
  - multimodality
  - foundation-models
  - embedding-models
  - gpu-performance
  - model-comparison
  - enterprise-data
  - open-source
  - performance-optimization
  - job-impact
  - agi-criticism
  - technical-report
people:
  - arthur-mensch
  - dan-schulman
  - chris-bishop
---


<!-- buttondown-editor-mode: plaintext -->> AI News for 4/12/2024-4/15/2024. We checked 5 subreddits and [**364** Twitters](https://twitter.com/i/lists/1585430245762441216) and **27** Discords (**395** channels, and **8916** messages) for you. Estimated reading time saved (at 200wpm): **985 minutes**.

Whole months happen in some days in AI - just as [Feb 15 saw Sora and Gemini 1.5](https://buttondown.email/ainews/archive/ainews-sora-pushes-sota/) and a bunch of other launches, the ides of April saw huge launches from:

## [Reka Core](https://twitter.com/RekaAILabs/status/1779894622334189592)

A new GPT4-class multimodal foundation model...

![image.png](https://assets.buttondown.email/images/908de291-3f84-4b64-b9eb-75a5deb689a5.png?w=960&fit=max) 

... with an actually useful technical report...

 ![image.png](https://assets.buttondown.email/images/91cc1830-d65f-403c-8c71-40e1d0262894.png?w=960&fit=max) 

... being "full Shazeer"

 ![image.png](https://assets.buttondown.email/images/15eadb75-acea-48bf-8bb1-7644279acb3c.png?w=960&fit=max) 

## Cohere Compass

> our new foundation embedding model that allows indexing and searching on multi-aspect data. Multi-aspect data can best be explained as data containing multiple concepts and relationships. This is common within enterprise data — emails, invoices, CVs, support tickets, log messages, and tabular data all contain substantial content with contextual relationships. 

 ![image.png](https://assets.buttondown.email/images/64c761c1-7926-4dcd-bace-e5844f7c5c3d.png?w=960&fit=max) 

## IDEFICS 2-8B

Continued work from [last year's IDEFICS](https://www.latent.space/p/idefics), a totally open source reproduction of Google's Flamingo unreleased multimodal model.

 ![image.png](https://assets.buttondown.email/images/05186c69-c132-4ecc-ac5a-8cd4fe44ac77.png?w=960&fit=max) 


## Rewind pivots to Limitless

> It’s a web app, Mac app, Windows app, and a wearable.

[Spyware is out, Pendants are in](https://twitter.com/dsiroker/status/1779857843895599383). 

![image.png](https://assets.buttondown.email/images/799f8934-9579-4e24-b114-9af2ed4ca3d8.png?w=960&fit=max) 


---

**Table of Contents**

[TOC] 


---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence. Comment crawling works now but has lots to improve!

**AI Models and Performance**

- **Apple MLX performance**: In /r/LocalLLaMA, Apple MLX (0.10.0) reaches [**105.5 tokens/s on M2 Ultra with 76 GPU**](https://www.reddit.com/r/LocalLLaMA/comments/1c3uzu6/apple_mlx_on_m2_ultra_76gpu_105_tokenss_with/), beating Ollama/Llama.cpp at 95.1 tokens/s using Mistral Instruct 4bit.
- **Ollama performance comparisons**: In /r/LocalLLaMA, Ollama performance using Mistral Instruct 0.2 q4_0 shows the [**M2 Ultra 76GPU leading at 95.1 t/s**](https://www.reddit.com/r/LocalLLaMA/comments/1c3v3q6/ollama_performance_on_m2_ultra_m3_max_windows/), followed by Windows Nvidia 3090 at 89.6 t/s, WSL2 NVidia 3090 at 86.1 t/s, and M3 Max 40GPU at 67.5 t/s. Apple MLX reaches 103.2 t/s on M2 Ultra and 76.2 t/s on M3 Max.
- **M3 Max vs M1 Max prompt speed**: In /r/LocalLLaMA, the [**M3 Max 64GB has more than double the prompt speed compared to M1 Max 64GB**](https://www.reddit.com/r/LocalLLaMA/comments/1c3t538/comparing_m1_max_64gb_vs_m3_max_64gb_prompt_speed/) when processing long contexts, using Command-R and Dolphin-Mixtral 8x7B models.
- **GPU considerations for LLMs**: In /r/LocalLLaMA, a user is [seeking advice on building a machine around RTX 4090 or buying a MAC](https://www.reddit.com/r/LocalLLaMA/comments/1c3qfg7/rtx_4090_vs_mac/) for running LLMs like Command R and Mixtral models with future upgradability.
- **GPU for Stable Diffusion**: In /r/StableDiffusion, a comparison of [3060 12GB vs 4060 16GB for Stable Diffusion](https://www.reddit.com/r/StableDiffusion/comments/1c3nk9g/3060_12gb_vs_4060_16gb_for_sdxl_or_wait_for_50xx/) recommends going for as much VRAM as reasonably affordable. A 4060ti 16GB takes 18.8 sec for SDXL 20 steps.


**LLM and AI Developments**

- **Comparing AI to humans**: In /r/singularity, Microsoft Research's Chris Bishop compares AI models that regurgitate information to "stochastic parrots", noting that [**humans who do the same are given university degrees**](https://www.reddit.com/r/singularity/comments/1c3o1o2/microsoft_researchs_chris_bishop_when_ai_models/). Comments discuss the validity of degrees and whether they indicate more than just information regurgitation.
- **Impact on jobs**: Former PayPal CEO Dan Schulman predicts that [**GPT-5 will be a "freak out moment" and that 80% of jobs will be reduced by 80% in scope**](https://twitter.com/woloski/status/1778783006389416050) due to AI.
- **Obsession with AGI**: Mistral's CEO Arthur Mensch believes the [obsession with achieving AGI is about "creating God"](https://www.businessinsider.com/mistrals-ceo-said-obsession-with-agi-about-creating-god-2024-4). Gary Marcus also [urges against creating conscious AI](https://www.reddit.com/r/singularity/comments/1c3vbkz/people_have_happily_worked_so_hard_to_build_stuff/) in a tweet.
- **Building for the future**: Sam Altman tweets about [people working hard to build technology for future generations to continue advancing](https://www.reddit.com/r/singularity/comments/1c3vbkz/people_have_happily_worked_so_hard_to_build_stuff/), without expecting to meet the beneficiaries of their work.
- **Confronting consciousness**: A post in /r/singularity argues that [achieving AGI will force humans to confront their lack of understanding about what creates consciousness](https://www.reddit.com/r/singularity/comments/1c3yvs1/agi_will_cause_humans_to_confront_that_they_do/), predicting debates and tensions around AI ethics and rights.

**Industry and Career**

- **Identifying "fake" ML roles**: In /r/MachineLearning, a PhD student [asks for advice on spotting "fake" ML roles](https://www.reddit.com/r/MachineLearning/comments/1c3z8ug/d_advice_for_spotting_fake_ml_roles/) after being hired for a position that didn't involve actual ML work, noting that asking questions during interviews may not be effective due to potential dishonesty.
- **Relevance of traditional NLP tasks**: Another PhD student [questions the importance of traditional NLP tasks like text classification, NER, and RE in the era of LLMs](https://www.reddit.com/r/MachineLearning/comments/1c4a7sa/d_are_traditional_nlp_tasks_such_as_text/), worrying about the future of their research.
- **Practical uses for LLMs**: A post in /r/MachineLearning [asks for examples of practical industry uses for LLMs beyond text generation](https://www.reddit.com/r/MachineLearning/comments/1c4cr32/d_in_industry_nlp_are_there_any_actualpractical/) that provide good ROI, noting that tasks like semantic search can be handled well by other models.

**Tools and Resources**

- **DBRX support in llama.cpp**: [Llama.cpp now supports DBRX](https://github.com/ggerganov/llama.cpp/pull/6515), a binary format for LLMs.
- **Faster structured generation**: In /r/LocalLLaMA, a new method for [structured generation in LLMs is claimed to be much faster than llama.cpp's approach](https://www.reddit.com/r/LocalLLaMA/comments/1c3oa8f/faster_than_llamacpps_grammar_structured/), with runtime independent of grammar complexity or model/vocabulary size. The authors plan to open-source it soon.
- **Python data sorting tools**: The author has [open-sourced their collection of Python tools for automating data sorting and organization](https://github.com/nazpins/naztech-automated-data-sorting-tools), aimed at handling unorganized files and large amounts of data efficiently.
- **Simple discrete diffusion implementation**: An [open-source, simple PyTorch implementation of discrete diffusion in 400 lines of code](https://www.reddit.com/r/MachineLearning/comments/1c3pvx5/p_extremely_short_and_simple_implementation_of/) was shared in /r/MachineLearning.

**Hardware and Performance**

- **M1 Max vs M3 Max**: In /r/LocalLLaMA, a [comparison of prompt speed between M1 Max and M3 Max with 64GB RAM](https://www.reddit.com/r/LocalLLaMA/comments/1c3t538/comparing_m1_max_64gb_vs_m3_max_64gb_prompt_speed/) shows the M3 Max having more than double the speed, especially for long contexts.
- **RTX 4090 vs Mac**: A post asks for [advice on building a PC with an RTX 4090 or buying a Mac for running LLMs](https://www.reddit.com/r/LocalLLaMA/comments/1c3qfg7/rtx_4090_vs_mac/), believing a PC would be cheaper and more upgradeable.
- **MLX performance on M2 Ultra**: Apple's MLX library [achieves 105.5 tokens/s on an M2 Ultra with 76 GPU cores](https://www.reddit.com/r/LocalLLaMA/comments/1c3uzu6/apple_mlx_on_m2_ultra_76gpu_105_tokenss_with/), surpassing llama.cpp's 95.1 tokens/s when running the Mistral Instruct 4-bit model.
- **Ollama performance comparison**: A [performance comparison of the Ollama library on various hardware](https://www.reddit.com/r/LocalLLaMA/comments/1c3v3q6/ollama_performance_on_m2_ultra_m3_max_windows/) shows the M2 Ultra leading at 95.1 t/s, followed by Windows Nvidia 3090, WSL2 Nvidia 3090, and M3 Max using the Mistral Instruct model.

**Memes and Humor**

- Several meme and humor posts were highly upvoted, including ["The Anti-AI Manifesto"](https://www.reddit.com/r/singularity/comments/1c3o1o2/microsoft_researchs_chris_bishop_when_ai_models/), ["Maybe maybe maybe"](https://v.redd.it/i2dxn7cu5guc1), ["the singularity is being driven by an outside force"](https://i.redd.it/fkp6wr5qwiuc1.png), ["Ai women are women"](https://v.redd.it/dppfbk4uzfuc1), and ["Real reason AI (Alien Intelligence) can't do hands? 👽"](https://i.redd.it/frcwgrkurjuc1.jpeg).

---

# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.

**AI Models and Architectures**

- **New model releases**: [@RekaAILabs](https://twitter.com/RekaAILabs/status/1779894622334189592) announced Reka Core, their "best and most capable multimodal language model yet", competitive with GPT-4/Opus level models. [@WizardLM_AI](https://twitter.com/WizardLM_AI/status/1779899325868589372) released WizardLM-2, a family of models including 8x22B, 70B and 7B variants competitive with leading proprietary LLMs.
- **Architectures and training**: [@jeremyphoward](https://twitter.com/jeremyphoward/status/1779684686618628323) noted that **transformers are an architecture** commonly used with diffusion. [@ylecun](https://twitter.com/ylecun/status/1779845304788955292) stated AI will eventually surpass human intelligence, but not with current auto-regressive LLMs alone.
- **Optimizing and scaling**: [@karpathy](https://twitter.com/karpathy/status/1779272336186978707) optimized an LLM in C to match PyTorch, now at 26.2ms/iteration, using tricks like cuBLAS in fp32 mode. [@_lewtun](https://twitter.com/_lewtun/status/1779804085677404583) believes strong Mixtral-8x22B fine-tunes will close the gap with proprietary models.

**AI Capabilities and Benchmarks**

- **Multimodal abilities**: [@RekaAILabs](https://twitter.com/RekaAILabs/status/1779894626083864873) shared Reka Core's video understanding capabilities, beating Claude3 Opus on multimodal chat. [@DrJimFan](https://twitter.com/DrJimFan/status/1779558822543229221) speculated Tesla FSD v13 may use language tokens to reason about complex self-driving scenarios.
- **Coding and math**: [@OfirPress](https://twitter.com/OfirPress/status/1779195498328429045) noted the open-source coding agent SWE-agent already has 1.5k users after 10 days. [@WizardLM_AI](https://twitter.com/WizardLM_AI/status/1779899333678387318) used a fully AI-powered synthetic training system to improve WizardLM-2.
- **Benchmarks and leaderboards**: [@svpino](https://twitter.com/svpino/status/1779185295541575710) noted Claude 3 was the best model on a human eval leaderboard for 17 seconds before GPT-4 updated. [@bindureddy](https://twitter.com/bindureddy/status/1779186163464708314) analyzed GPT-4's coding, math and knowledge cutoff as reasons for its performance.

**Open Source and Democratizing AI**

- **Open models and data**: [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1779891910871490856) announced EleutherAI's Pile-T5 model using 2T tokens from the Pile and the Llama tokenizer. [@_philschmid](https://twitter.com/_philschmid/status/1779922877589889400) introduced Idefics2, an open source VLM under 10B parameters with strong OCR, document understanding and visual reasoning.
- **Accessibility and cost**: [@maximelabonne](https://twitter.com/maximelabonne/status/1779801605702836454) noted open source models now lag top closed source by 6-10 months rather than years. [@ClementDelangue](https://twitter.com/ClementDelangue/status/1779805019841142791) predicted the gap will fully close by year end, as open source is faster, cheaper and safer for most uses.
- **Compute and tooling**: [@WizardLM_AI](https://twitter.com/WizardLM_AI/status/1779899329844760771) is sharing 8x22B and 7B WizardLM-2 weights on Hugging Face. [@aidangomez](https://twitter.com/aidangomez/status/1779882113573044625) announced the Compass embedding model beta for multi-aspect data search.

**Industry and Ecosystem**

- **Company expansions**: [@gdb](https://twitter.com/gdb/status/1779762694473551924) and [@hardmaru](https://twitter.com/hardmaru/status/1779783633961935218) noted OpenAI's expansion to Japan as a significant AI presence. [@adcock_brett](https://twitter.com/adcock_brett/status/1779541107577151916) shared Canada's $2.4B investment in AI capabilities and infrastructure.
- **Emerging applications**: [@svpino](https://twitter.com/svpino/status/1779843933276672195) built an entire RAG app without code using Langflow's visual interface and Langchain. [@llama_index](https://twitter.com/llama_index/status/1779542320133947622) showcased using LLMs and knowledge graphs to accelerate biomaterials discovery.
- **Ethical considerations**: [@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1779171248083177500) regretted not supporting @PalmerLuckey more at Facebook during a "witch hunt". [@mmitchell_ai](https://twitter.com/mmitchell_ai/status/1779721105407873411) praised a video as the best AI existential risk coverage so far.

---

# AI Discord Recap

> A summary of Summaries of Summaries

1. **Advancements in Large Language Models (LLMs)**: There is significant excitement and discussion around new releases and capabilities of LLMs across various platforms and organizations. Key examples include:

  - **[Pile-T5](https://blog.eleuther.ai/pile-t5/)** from EleutherAI, a T5 model variant trained on 2 trillion tokens, showing improved performance on benchmarks like SuperGLUE and MMLU. All resources, including model weights and scripts, are [open-sourced on GitHub](https://github.com/EleutherAI/improved-t5).

  - **[WizardLM-2](https://wizardlm.github.io/WizardLM2/)** series announced, with model sizes like 8x22B, 70B, and 7B, sparking excitement for deployment on OpenRouter, with WizardLM-2 8x22B compared favorably to GPT-4.

  - **[Reka Core](https://publications.reka.ai/reka-core-tech-report.pdf)**, a frontier-class **multimodal language model** from Reka AI, with details on training, architecture, and evaluation shared in a technical report.

2. **Optimizations and Techniques for LLM Training and Inference**: Extensive discussions revolve around optimizing various aspects of LLM development, including:

  - **Efficient context handling** with approaches like [Ring Attention](https://coconut-mode.com/posts/ring-attention/), enabling models to scale to nearly infinite context windows by employing multiple devices.

  - **Model compression** techniques such as **LoRA**, **QLoRA**, and **16-bit quantization** for reducing memory footprint, with insights from [Lightning AI](https://lightning.ai/pages/community/lora-insights/) and community experiments.

  - **Hardware acceleration** strategies like enabling **P2P support on NVIDIA 4090 GPUs** using [tinygrad's driver patch](https://github.com/tinygrad/open-gpu-kernel-modules), achieving significant performance gains.

  - **Kernel optimizations** in frameworks like [LLM.c](https://github.com/karpathy/llm.c) and [torchao](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py), exploring efficient tensor layouts, padding, and swizzling for matrix operations.

3. **Open-Source Initiatives and Community Collaboration**: The AI community demonstrates a strong commitment to open-source development and knowledge sharing, as evidenced by:

  - **Open-sourcing** of major projects like [Pile-T5](https://github.com/EleutherAI/improved-t5), [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl), and [Mixtral](https://github.com/modularml/mojo/blob/main/stdlib/src/builtin/io.mojo#L362) (in Mojo), fostering transparency and collaborative efforts.

  - **Educational resources** such as [CUDA MODE lectures](https://github.com/cuda-mode/lectures) and a call for volunteers to record and share content, promoting knowledge dissemination.

  - **Community projects** like [llm.mojo](https://github.com/dorjeduck/llm.mojo) (Mojo port of llm.c), [Perplexica](https://github.com/ItzCrazyKns/Perplexica/) (Perplexity AI clone), and [LlamaIndex integrations](https://medium.com/ai-advances/enhancing-document-retrieval-with-memory-a-tutorial-for-llamaindex-with-colbert-based-agent-1c3c47461122) for document retrieval, showcasing grassroots innovation.

4. **Datasets and Data Strategies for LLM Development**: Discussions highlight the importance of data quality, curation, and strategic approaches to training data, including:

  - **Synthetic data generation** techniques like those used in [StableLM](https://arxiv.org/abs/2402.17834) (ReStruct dataset) and [MiniCPM](https://arxiv.org/abs/2404.06395) (mixing OpenOrca and EvolInstruct), with a focus on [CodecLM](https://arxiv.org/pdf/2404.05875.pdf) for aligning LLMs with tailored synthetic data.

  - **Data filtering strategies** and the development of [scaling laws for data curation](https://x.com/pratyushmaini/status/1778577153107570770), emphasizing that curation cannot be compute-agnostic, as presented in a CVPR 2024 paper.

  - **Multilingual and multimodal datasets**, with a call for copyright-permissive EU text and multimodal data to train large open multimodal models, reflecting the growing demand for diverse data sources. [Source 1](https://blog.eleuther.ai/pile-t5/) | [Source 2](https://wizardlm.github.io/WizardLM2/) | [Source 3](https://publications.reka.ai/reka-core-tech-report.pdf) | [Source 4](https://coconut-mode.com/posts/ring-attention/)

5. **Misc**

  - **Stable Diffusion 3 Sparks Excitement and Debate**: The AI community is eagerly anticipating the release of **[Stable Diffusion 3 (SD3)](https://www.youtube.com/watch?v=mQSKoAEaIJA)**, discussing its potential improvements in quality and efficiency. Conversations revolve around optimizing performance on less powerful GPUs with tools like **[SD Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge)**, and exploring AI-powered creative workflows with **[ControlNet](https://github.com/lllyasviel/ControlNet)**, **Lora**, and **outpainting** techniques. The heavy prompt censorship in SD3 has raised concerns about potential quality decline, as discussed on [Reddit](https://www.reddit.com/r/StableDiffusion/comments/1c3ro5y/stability_employee_preview_of_stable_diffusion_3/).

  - **Perplexity AI's Roadmap and Model Comparisons**: **Perplexity AI's** June roadmap teases new features like enforcing JSON grammar, a new Databricks model, Model Info endpoint, status page, and multilingual support, viewable on their [feature roadmap page](https://docs.perplexity.ai/docs/feature-roadmap). Discussions compare the context window and performance of models like **Claude Opus**, **GPT-4**, and **RAG** for various tasks. Meta's release of an AI interface on WhatsApp, resembling Perplexity AI, has sparked interest in the growing integration of AI in messaging platforms, as reported in this [article](https://analyticsindiamag.com/meta-releases-ai-on-whatsapp-looks-like-perplexity-ai/).

  - **Tinygrad Enables P2P on NVIDIA GPUs**: **Tinygrad** has successfully enabled peer-to-peer (P2P) support on **NVIDIA 4090 and 4070 TI Super GPUs** by modifying NVIDIA's driver, achieving 14.7 GB/s AllReduce performance. The breakthrough, shared on [Twitter](https://twitter.com/__tinygrad__/status/1778677126092509611) with code available on [GitHub](https://github.com/tinygrad/open-gpu-kernel-modules), has significant implications for cost reduction in running large language models. The **CUDA community** is pushing performance boundaries with the [One Billion Row Challenge](https://1brc.dev/) and exploring optimization strategies for low-precision computations.
  - **Eleuther AI's Pile-T5 Model and Research Insights**: **EleutherAI** introduced **[Pile-T5](https://blog.eleuther.ai/pile-t5/)**, a T5 model variant trained on 2 trillion tokens from the Pile, demonstrating improved performance on benchmarks like *SuperGLUE* and *MMLU*. The model, which excels in code-related tasks, has its weights and training scripts [open-sourced on GitHub](https://github.com/EleutherAI/improved-t5). Research discussions delved into the capabilities of **MoE** versus dense transformer models, the role of tokenization in LLMs, and the interpretability of hidden representations using frameworks like Google's **[Patchscopes](http://research.google/blog/patchscopes-a-unifying-framework-for-inspecting-hidden-representations-of-language-models/)**.

---



# PART 1: High level Discord summaries




## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

**Stable Diffusion 3 Sparks Excitement**: AI enthusiasts are buzzing with anticipation over [Stable Diffusion 3 (SD3)](https://www.youtube.com/watch?v=mQSKoAEaIJA), discussing its potential for efficiency gains and debating the merits of SD Forge for optimizing performance on less powerful GPUs.

**Pixart-Sigma Pushing VRAM Boundaries**: The application of Pixart-Sigma with T5 conditioning within [ComfyUI](https://github.com/city96/ComfyUI_ExtraModels) provokes discussions around VRAM usage, with participants noting that T5 maintains VRAM usage under 10.3GB even in 8-bit compression mode.

**AI Tools for Content Creation Get Spotlight**: Query exchanges regarding AI tools like ControlNet, Lora, and expansion techniques like outpainting hint at a need for consistent color generation and background extension in creative workflows.

**Debating CPU vs. GPU Efficiency for AI**: Community members exchange troubleshooting tips on GPU memory optimization and flag features like `--lowvram` for those running Stable Diffusion on less potent machines, highlighting the significant speed difference between CPU and GPU processing.

**Artists Seek Tech-Driven Collaborations**: The trend of fusing AI with artistic tools continues as a digital artist seeks input and tutorial assistance on a painting app combining AI features, with project details found on [GitHub](https://github.com/QuintessentialForms/ParrotLUX).



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **AI Models at Play in Prose and Logic**: Debates underscore that **Claude Opus** excels in prose while **GPT-4** outshines in logical reasoning. Practitioners should explore models to best fit their application needs.

- **Context Hinges on Model Choice**: **Claude Opus** is praised for executing sequential instructions with finesse, while discussions indicate that **GPT-4** may falter after several follow-ups; **RAG** boasts an edge with larger context retrieval for file uploads.

- **Perplexity's Roadmap débuts Multilingualism**: Perplexity AI's June roadmap teases enforcing JSON grammar, new Databricks model, Model Info endpoint, status page, multilingual support, and N>1 sampling, viewable on their [feature roadmap page](https://docs.perplexity.ai/docs/feature-roadmap).

- **API Intricacies Spark Dialogue**: Users delve into Perplexity API's nuances—from default temperature settings in models, citation features, community aid for URL retrieval in responses, to procedures for increased rate limits.

- **Meta Mimics Perplexity?**: A buzz on Meta's AI resembling Perplexity's interface for WhatsApp, suggesting AI's growing integration within messaging platforms, further highlighted by an article comparing the two ([Meta Releases AI on WhatsApp, Looks Like Perplexity AI](https://analyticsindiamag.com/meta-releases-ai-on-whatsapp-looks-like-perplexity-ai/)).



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**Geohot Hacks Back P2P to 4090s**: "geohot" ingeniously implemented peer-to-peer support into NVIDIA 4090s, enabling enhanced multi-GPU setups; details are available on [GitHub](https://github.com/tinygrad/open-gpu-kernel-modules).

**Unsloth Gains Multi-GPU Momentum**: Interest spiked on Multi-GPU support in Unsloth AI, with Llama-Factory touted as a worthy investigation route for integration.

**A Hugging Face of Encouragement**: Unsloth AI garnered attention from Hugging Face CEO Clement Delangue by securing a follow on an unspecified platform, suggesting potential collaborative undertones.

**Linguistic Labyrinth of AI PhD Life**: A PhD student outlined their challenging exploration in developing an instruction-following LLM for their mother tongue, highlighting the project complexity beyond mere translation and fine-tuning.

**Million-Dollar Mathematical AI**: The community engaged with the prospect of a $10M Kaggle AI prize to create an LLM capable of acing the International Math Olympiad and unpacked the Beal Conjecture's $1M bounty for a proof or counterexample at [AMS](https://www.ams.org/profession/prizes-awards/ams-supported/beal-prize-rules).

**Resourceful VRAM Practices & Strategic Finetuning**: AI engineers converged on efficient use of VRAM for training robust LLMs like Mistral, sharing best practices such as the "30% less VRAM" update from Unsloth and the nuanced approach of initiating finetuning with shorter examples.

**Cultural Conquest of Linguistic Datasets**: Strategies to amplify low-resource language datasets were exchanged, including the use of translation data from platforms like HuggingFace.

**Pioneering Podman for AI Deployment**: Innovators showcased the deployment of Unsloth AI in Podman containers, streamlining local and cloud implementation, as seen in this demo: [Podman Build - For cost-conscious AI](https://youtu.be/3iEhFKIDXp0).

**Ghost 7B Alpha Raises the Benchmark**: Ghost 7B Alpha was proclaimed for superior reasoning compared to other models, signaling fine-tuning prowess without expanding tokenizers, as discussed by enthusiasts.

**Merging Minds on Model Compression**: The melding of adapters, particularly QLoRA and 16bit-saving techniques for vLLM or GGUF, was deliberated, contemplating the intricacies of naive merging versus dequantization strategies.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**Pile-T5 Power**: EleutherAI introduces **Pile-T5**, an enhanced T5 model variant produced through training with 2 trillion tokens from the [Pile](https://blog.eleuther.ai/pile-t5/). It showcases significant performance improvements in both *SuperGLUE* and *MMLU* benchmarks and excels in code-related tasks, with resources including weights and scripts [open-sourced on GitHub](https://github.com/EleutherAI/improved-t5).

**Entropic Data Filtering**: The **CVPR 2024 paper** suggests a notable advancement in unpacking the importance of entropy in data filtering. An empirical study unveiled scaling laws capturing how data curation is fundamentally linked with entropy, enriching the community's understanding of heterogeneous & limited web data and its practical implications. Explore the study [here](https://arxiv.org/abs/2404.07177).

**Inside the Transformer Black Box**: Google's Patchscopes framework endeavors to make LLMs' hidden representations more interpretable by generating explanations in natural language. Likewise, a paper introducing a toolkit for transformers to conduct causal manipulations showcases the value of pinpointing key model subcircuits during training, possibly offering pathways to avoid common training roadblocks. Details on the JAX toolkit can be found in this [tweet from Stephanie Chan (@scychan_brains)](https://x.com/scychan_brains/status/1779357918737158364?s=46) and on the Patchscopes framework [here](http://research.google/blog/patchscopes-a-unifying-framework-for-inspecting-hidden-representations-of-language-models/).

**MoE vs. Dense Transformers Debate**: Discussions in the community probe the capacity and benefits of MoE versus dense transformer models. Key insights reveal MoEs' relative advantage sans VRAM constraints and question dense models' performance parity at comparable parameter budgets. There's a pronounced curiosity regarding the foundational attributes driving model behavior beyond the metrics.

**NeoX Nuances Unveiled**: Questions within the **GPT-NeoX project** brought up intricacies like oversized embedding matrices for **GPU efficiency** and peculiar **weight decay** behaviors potentially due to non-standard activations. A remark on **rotary embeddings** noted its partial application in NeoX as against other models. A [corporate CLA](https://discord.com/channels/729741769192767510/730090096287547444/1228274522034012200) is being devised to facilitate contributions to the project.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Google's Infini-attention Paper Snags Spotlight**: A member expressed interest in Google's "Infini-attention" paper, available on [arXiv](https://arxiv.org/pdf/2404.07143.pdf), proposing a novel way to handle longer sequences for language models.
  
- **Community Delves into Vector Search**: In a comparison of vector search speeds, FAISS's IVFPQ came out on top at 0.36ms, followed by FFVec's 0.44ms; meanwhile, FAISS Flat and HNSW clocked in at slower times of 10.58ms and 1.81ms, respectively.

- **AI Benchmarks Garner Scrutiny**: A discussion among members challenged the effectiveness of current AI benchmarks, proposing the need for harder-to-game, domain-specific task evaluations, such as a scrutinized [tax benchmark](https://www.vals.ai/taxeval).

- **Anticipating WorldSim's Update**: Enthusiasm is brewing over the forthcoming update to **WorldSim**, with members sharing their experiments inspired by it, while preparing for new exciting features teased for next Wednesday.

- **LLMs Under the Microscope**: Technical exchanges included inquiries into a safety-focused healthcare LLM detailed in an [arXiv paper](https://arxiv.org/abs/2403.13313), challenges in fine-tuning LLMs for the Greek language while considering pretraining necessity, and methods for in-line citations in Retrieval-Augmented Generation (RAG) queries.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**Mixtral Model Mix-Up**: The community reported that `Mixtral 8x22B:free` is discontinued; users should transition to the [Mixtral 8x22B standard model](https://openrouter.ai/models/mistralai/mixtral-8x22b). Experimental models **Zephyr 141B-A35B** and **Fireworks: Mixtral-8x22B Instruct (preview)** are available for testing; the latter is a fine-tune of Mixtral 8x22B.

**Token Transaction Troubles**: A user's issue with purchasing tokens was deemed unrelated to OpenRouter; they were advised to contact **Syrax** directly for resolution.

**Showcasing Rubik's Research Assistant**: Users interested in testing the new **Rubiks.ai** research assistant can join the beta test with a 2-month free premium. This tool includes access to models like **Claude 3 Opus and GPT-4 Turbo**, among others; testers should use the code `RUBIX` and provide feedback. [Explore Rubik's AI](https://rubiks.ai/).

**Dynamic Routing Deliberation**: There's a buzz around improving **Mixtral 8x7B Instruct (nitro)** speeds via dynamic routing to the highest transaction-per-second (t/s) endpoint. There were varying opinions on the performance of models like **Zephyr 141b** and **Firextral-8x22B** as well.

**WizardLM-2 Series Spells Excitement**: The newly announced **WizardLM-2** series with model sizes including 8x22B, 70B, and 7B, has garnered community interest. Attention is particularly focused on the expected performance of **WizardLM-2 8x22B** on OpenRouter.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**Models On The Fritz**: Users reported problems with model loading in **LM Studio** across different versions and OS, including error messages about `"Error loading model."` and issues persisting after downgrading and turning off GPU offload. A full removal and reinstall of LM Studio were sought by one user after ongoing frustrations with model performance.

**Attention to Hardware**: There was considerable discussion surrounding **hardware requirements** for running AI models effectively, highlighting the necessity for high-tier equipment for an experience on par with GPT-4 and the underutilization of **Threadripper Pro** CPUs. Contrastingly, **ROCm** support was called into question, with specific mention of the **Radeon 6600** being unsupported in LM Studio.

**Quantized conundrums and innovative inferences**: **Quantization** was a hot topic, with users noting performance changes and debate on whether the trade-off in output quality is worthwhile. Meanwhile, a paper titled "Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention" was circulated, illuminating cutting-edge methodologies for Transformers handling prolonged inputs effectively.

**Navigating Through New Model Territories**: Users shared experiences and recommendations for an assortment of new models like **Command R+** and **Mixtral 8x22b**, focusing on different setups, performance, and even introducing a commands-based tool [missionsquad.ai](https://missionsquad.ai/). Notably, **Command R+** has been praised for outpacing similar models in analyzing lengthy documents.

**Beta Blues and Docker Distributions**: While some users grappled with troubles in **MACOS 0.2.19**, others spearheaded initiatives like creating and maintaining a Docker image for **Autogen Studio**, hosted on GitHub. Concurrently, sentiments of letdown were conveyed over Linux beta releases missing out on ROCm support, highlighting the niche hurdles that technical users encounter.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI Sentience: Fact or Fiction?** Engaging debates broke out about AI sentience, revolving around the implications of AI advancements on consciousness and ethical use. Ethical considerations were highlighted, but no consensus was reached on sentient AI systems.

- **Token Trading Techniques**: Strategies for dealing with GPT's token limits were discussed, with users suggesting summarizing content and using follow-up prompts to manage the context within the constraint effectively.

- **GPT-4 Turbo's Visionary Capabilities**: Clarification was provided that 'gpt-4-turbo' integrates vision capabilities, and while it still bears a 4k token limit due to memory constraints, it requires a script for analyzing images.

- **Prompting for Precision**: Users shared experiences with GPT model inconsistencies, suggesting version-specific behaviors and fine-tuning aspects that impact response accuracy, with recommendations to employ a meta-prompting template for better results.

- **Hackers Wanted for Prompt Challenge**: There's a call for engineers interested in exploring prompt hacking to team up for a competition, suggesting a collaborative venture into creative interactions with LLMs.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

**Collaboration Beyond Borders with Scarlet AI**: The new **Scarlet** platform enhances task management with **collaborative AI agents** that provide automated context and support tool orchestration, eyeing an integration with Zapier NLA. A work-in-progress demo was mentioned and [Scarlet's website](https://scarletai.co) provides details on their offerings.

**Whispers of Perplexity's Vanishing Act**: The removal of Perplexity "online" models from LMSYS Arena spurred debate on the model's efficacy and potential integrations into engineering tools. This instance underscores a shared interest in the integration efficiency of AI models.

**AI Wrangles YouTube's Wild West**: Engineers examined strategies to **transcribe and summarize YouTube content** for AI applications, debating the merits of various tools including Descript, youtube-dl, and Whisper with diarization. The discussion reflects ongoing endeavors to streamline content processing for AI model training.

**Limitless Eyes the Future of Personalized AI**: The rebranding of Rewind to **Limitless** introduces a wearable with **personalized AI capabilities**, sparking discussions around local processing and data privacy. This highlights a peak interest in the security implications of new AI-powered wearable technologies.

**Navigating Vector Space in Semantic Search**: Insightful discussions on the complexities of semantic search, including vector size, memory usage, and retrieval performance, culminated in a proposed hackathon to delve deeper into embedding models. The discourse reflects a keen focus on optimizing speed, efficiency, and performance in AI-powered search applications.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Earn Your Mojo Stripes**: Engage with the **Mojo** community by contributing to the `modularml/mojo` repo or creating cool open source projects to attain the prestigious `Mojician` role. Those with merged PRs can DM Jack to match their GitHub contributions with Discord identities.

**Mojo's Python Aspirations**: The community is buzzing with the anticipation of Mojo extending full support to Python libraries, aiming to include **Python packages with C extensions**. Meanwhile, efforts to enhance Mojo's `Reference` usability are in motion, with Chris Lattner hinting at a proposal that could simplify mutability management.

**Code Generation and Error Handling Forefront**: GPU code generation tactics and the potential integration of an "extensions" feature resembling Swift's implementation for superior error handling have sparked technical debates among members, indicating a future direction for Mojo's development.

**Community Code Collaborations Spike**: A flurry of activity surrounds the llm.mojo project, where performance boost techniques such as **vectorization and parallelization** might benefit from collective wisdom, including maintaining synchronized C and Mojo code bases within a single repository.

**Nightly News on Mojo Updates**: The Mojo team addresses standard library discrepancies in package naming with an upcoming release fix. Additionally, the idea of updating `StaticTuple` to `Array` and supporting `AnyType` garnered interest, and a call for Unicode support contributions to the standard library arose, along with discussions on proper item assignment syntax.

**Twitter Dispatch for Modular Updates**: Keep an eye on [Modular's Twitter](https://twitter.com/Modular) for the latest flurry of tweets covering updates and announcements, including a series of six recent tweets from the organization that shed light on their ongoing initiatives.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**P2P Gets a Speed Boost**: **Tinygrad** enhances P2P support on **NVIDIA 4090 GPUs**, reaching 14.7 GB/s AllReduce performance after modifying NVIDIA's driver, as detailed [here](https://x.com/__tinygrad__/status/1778676746378002712), while **PyTorch** tackles namespace build complexities with the nightly build showing slower performance versus `torch.matmul`.

**Massive Row Sorting Challenge Awaits CUDA Competitors**: tspeterkim_89106 throws down the gauntlet with a [One Billion Row Challenge](https://1brc.dev/) in CUDA implementation that impressively runs in 16.8 seconds on a **V100** and invites others to beat a [6-second record on a 4090 GPU](https://tspeterkim.github.io/posts/cuda-1brc).

**New Territories in Performance Optimization**: CUDA discussions orbit around the merits of running independent matmuls in separate CUDA streams, leveraging `stable_fast` over `torch.compile`, and the pursuit of high-efficiency low-precision computations demonstrated by `stable_fast` challenging **int8 quant tensorrt** speeds.

**Recording and Sharing CUDA Expertise**: **CUDA MODE** is recruiting volunteers to record and share content through YouTube, where lecturing material is also maintained on [GitHub](https://github.com/cuda-mode/lectures), and highlighting potential shifts to live streaming to manage growing member scales.

**HQQ and LLM.c: Striding Towards Efficiency**: Updates in **HQQ implementation** on gpt-fast push token generation speeds with **torchao int4 kernel** support, while **LLM.c** confronts CUDA softmax integration challenges, explores online softmax algorithm efficiencies, and juggles the dual goals of peak performance and educational clarity.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

**Connector Confusion Cleared**: A member resolved an issue with **Cohere's connector API**, learning that connector URLs must end with `/search` to avoid errors.

**Fine-Tuning Finesse**: **Cohere's base models** are available for fine-tuning through their dashboard, as confirmed in a [dashboard link](https://dashboard.cohere.com/fine-tuning/), with options expanded to Amazon Bedrock and [AWS](https://aws.amazon.com/marketplace/seller-profile?id=87af0c85-6cf9-4ed8-bee0-b40ce65167e0).

**New Cohere Tools on the Horizon**: Updates were shared on new **Cohere** capabilities, specifically named **Coral** for chatbot interfaces, and upcoming releases for **AWS toolkits** for connector implementations.

**Model Performance Discussed**: Dialogue around **Cohere models** like **command-r-plus** touched on their performance on different hardware, including Nvidia's latest graphics cards and TPUs.

**Learning Avenues for AI Newbies**: New community members seeking educational resources were directed to free offerings like **LLM University** and provided with a link to [Cohere's educational documentation](https://docs.cohere.com/docs/llmu).

**Command-R Rocks the Core**: **Command-R** received accolades for being a newly integrated core module by Cohere, highlighting its significance.

**Quant and AI Converge**: An invitation for beta testing of **Quant Based Finance** was posted, appealing to those interested in financial analysis powered by AI, with a link [here](https://quantfino.com/join-beta).

**Rubiks.AI Rolls Out**: An invite for beta testing of **Rubiks.AI**, a new advanced research assistant and search engine, was shared, offering early access to models like **Claude 3 Opus** and **GPT-4 Turbo**, available at [Rubiks.AI](https://rubiks.ai/).



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**A Bundle of Multi-Model Know-how**: Engineers exchanged tips on deploying **multiple A1111 models** on a GPU, highlighting resource allocation for parallel model runs. Discussions in [NLP](https://huggingface.co/docs/transformers/model_doc/bart) explored lightweight embedding options such as **all-MiniLM-L6-v2**, with **paraphrase-multilingual-MiniLM-L12-v2** suggested for academic purposes. 

**Cognitive Collision: AI Models Straddle Realities**: The [cool-finds channel](https://github.com/QUVA-Lab/e2cnn) shared links to PyTorch and Blender integration for real-time data pipelines, while a *Medium* post introduced **LlamaIndex**'s document retrieval enhancements. The **Grounding DINO** model for zero-shot object detection and how it utilizes this in Transformers trended in [computer-vision](https://huggingface.co/docs/transformers/main/en/model_doc/grounding-dino).

**Community-Sourced AI Timeline & Mental Models**: Project **RicercaMente**, aiming to map data science evolution through key papers, was touted in [cool-finds](https://github.com/EdoPedrocchi/RicercaMente) and [NLP](https://github.com/EdoPedrocchi/RicercaMente), inviting collaboration from the community. Meanwhile, a Counterfactual Inception method was presented to address hallucinations in AI responses, detailed in a [paper on arXiv](https://arxiv.org/abs/2403.13513) and a related [GitHub project](https://github.com/ivy-lvlm/counterfactual-inception).

**Training Trials & Tribulations**: A U-Net training plateau after 11 epochs led a user to consider **Deep Image Prior** for image cleaning tasks, shared in [computer-vision](https://huggingface.co/docs/transformers/main/en/model_doc/grounding-dino). In [diffusion-discussions](https://github.com/huggingface/diffusers/issues/7672), there was an exploration of multimodal embeddings and a clarification about an overstretched token limit warning in a Gradio chatbot for image generation.

**Crossing Streams: Events & Education**: Upcoming LLM Reading Group sessions focusing on groundbreaking research, including OpenAI's CLIP model and Zero-Shot Classification, were advertised in [today-im-learning](https://www.eventbrite.ca/e/llm-reading-group-march-5-19-april-2-16-30-may-14-28-tickets-851921368747?aff=oddtdtcreator) and [reading-group](https://www.eventbrite.ca/e/llm-reading-group-march-5-19-april-2-16-30-may-14-28-tickets-851921368747?aff=oddtdtcreator). Resources for those starting out in NLP were also suggested, featuring beginner's guides and transformer overview videos.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

**Scam Notice: No LAION NFTs Exist**: There have been repeated warnings about a scam involving a fraudulent Twitter account claiming **LAION** is offering NFTs, but the community has confirmed **LAION** is solely focused on **free, open-source AI resources** and does not engage in selling anything.

**When to Guide the Diffusion**: An [arXiv paper](https://arxiv.org/abs/2404.07724) introduced findings that the best image results from diffusion models occur when guidance is applied at particular noise levels, emphasizing the middle stages of generation while avoiding early and late phases.

**Innovations from AI Audio to Ethics Discussions**: Discussions ranged from the introduction of new AI models, like **Hugging Face's Parler TTS** for sound generation, to debates over the proper use of 'ethical datasets' and the implications of such political language in AI research.

**Stable Diffusion 3's Dilemma**: Community insight suggested **Stable Diffusion 3** faces a risk of quality decline due to its rigorous prompt censorship. There is anticipation for further refinement to address this possible issue shared particularly on [Reddit](https://www.reddit.com/r/StableDiffusion/comments/1c3ro5y/stability_employee_preview_of_stable_diffusion_3/).

**Troubleshooting Diffusion Models**: A [GitHub repository](https://github.com/K3dA2/Diffusion-Model) was shared by a member who faces a training issue with their diffusion model, which is outputting random noise and solid black during inference, despite attempts to adjust regularization and learning rate.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**Be Cautious of Spam**: Several channels have reported spam messages containing links to adult content, falsely advertising **TEEN/ONLYFANS PORN** with a potential phishing risk. Members are advised to avoid engaging with suspicious links or content.

**RAG Operations Demand Precision**: Users are encountering issues with document splitting during Retrieval-Augmented Generation (RAG) operations on legal contracts, where section contents are mistakenly linked to preceding sections, compromising retrieval accuracy.

**LangChain Gets Parallel**: Utilizing LangChain's `RunnableParallel` class allows for the parallel execution of tasks, enhancing efficiency in LangGraph operations—an approach worth considering for those optimizing for performance.

**Emerging AI Tools and Techniques**: A variety of resources, tutorials, and projects have been shared, including [Meeting Reporter](https://meeting-reporter.streamlit.app/), [how-to guides for RAG](https://luisciber.substack.com/p/how-to-build-a-rag-system-locally), and Personalized recommendation systems, to equip AI professionals with cutting-edge knowledge and practical solutions.

**Watch and Learn**: A series of YouTube tutorials has been highlighted, focusing on the implementation of chat agents using Vertex AI Agent Builder and integrating them with communication platforms like Slack, valuable for those interested in AI-infused app development.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**OpenAccess AI Goes Open-Source**: NVIDIA Linux GPU support with P2P gets a boost from [open-gpu-kernel-modules on GitHub](https://github.com/tinygrad/open-gpu-kernel-modules), offering a tool for enhanced GPU functionality.

**Fireworks AI Ignites with Instruct MoE**: Promising results from Fireworks AI's Mixtral 8x22b Instruct OH, previewed [here](https://fireworks.ai/models/fireworks/mixtral-8x22b-instruct-preview), although facing a hiccup with DeepSpeed zero 3 which was addressed by pulling updates from DeepSpeed's [main branch](https://github.com/microsoft/DeepSpeed/pull/5008).

**DeepSpeed's Contributions Clarified**: While it doesn't accelerate model training, *deepspeed zero 3* shines in training larger models, with a successful workaround integrating updates from DeepSpeed's official GitHub repository for MoE models.

**A Harmonious Relationship with AI**: Advances in AI-music creation gain spotlight with a tune crafted by AI, available for a listen at [udio.com](https://www.udio.com/songs/eY7xtug1dV6hbfCDhyHJua).

**Tools for Advanced Model Training**: Engaging discussions revolve around model merging, the use of LISA and DeepSpeed, and their effects on model performance. [This tool](https://github.com/MeNicefellow/Mixtral-Model-Expert-Extractor/blob/main/main.py) was cited for extracting Mixtral model experts, alongside hardware prerequisites.

**Dynamic Weight Unfreezing**: Conversations emerge around **dynamic weight unfreezing** tactics for GPU-constrained users, alongside an unofficial GitHub implementation for **Mixture-of-Depths**, accessible [here](https://github.com/astramind-ai/Mixture-of-depths).

**RTX 4090 GPUs and P2P**: Success in enabling P2P memory access with **tinygrad** on RTX 4090s lead to discussions about removing barriers to P2P usage and community eagerness regarding this achievement.

**Combatting Model Repetitiveness**: Persistent shortcomings with a model producing repetitive outputs guide members towards exploring diverse datasets and finetuning methods. **Mergekit** emerges as a go-to for model surgery with configuration insights gleaned from [WestLake-10.7B-v2](https://huggingface.co/froggeric/WestLake-10.7B-v2/blob/main/mergekit_config.yml).

**Fine-Tuning and Prompt Surgery**: One delves into Axolotl's finetuning intricacies, troubleshooting IndexError and prompt formatting woes with guidance from the [Axolotl GitHub repo](https://github.com/OpenAccess-AI-Collective/axolotl/tree/main/examples) and successful configuration adjustments.

**Mistral V2 Outshines**: Exceptional first-epoch results with Mistral v2 instruct overshadow others, demonstrating aptitude in diverse tasks including metadata extraction, outperforming models like **qwen** with new **automation capabilities**.

**DeepSpeed Docker Deep Dive**: Distributed training via DeepSpeed necessitates a custom Docker build and streamlined SSH keys for passwordless node intercommunication. Launching containers with the correct environment variables is essential, as detailed in this [Phorm AI Code Search](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=d905c33d-1397-4ef3-82ad-a16aadf1eb1f)) link.

**DeepSpeed and 🤗 Accelerate Collaboration**: DeepSpeed integrates smoothly with 🤗 Accelerate without overriding custom learning rate schedulers, with the `push_to_hub` method complementing the ease of Hugging Face model hub repository creation.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**Malware Scare and Command Line Riddles**: Engineers have raised an issue about **Avast antivirus** detections and confusion stemming from the `ngrok/ngrok/ngrok` command line in OpenInterpreter; updates to the documentation were suggested to clear user concerns.

**Tiktoken Gets a PR For Building Progress**: A GitHub pull request aiming to **resolve a build error** by updating the tiktoken version for OpenInterpreter suggests improvements are on the way; review the changes [here](https://github.com/OpenInterpreter/open-interpreter/pull/1204).

**Persistence Puzzle: Emergence of Assistants API**: The integration of the **Assistants API** for data persistence has been discussed, with community members creating Python assistant modules for better session management; advice on node operations implementation is being sought.

**OS Mode Odyssey on Ubuntu**: The **Open Interpreter's OS Mode** on Ubuntu has generated troubleshooting conversations, with a focus on downgrading to **Python 3.10** for platform compatibility and configuring accessibility settings.

**Customize It Yourself: O1's User-Driven Innovation**: The O1 community showcases their creativity through **personal modifications and enhancements** such as improved batteries and custom cases; a custom GPT model trained on Open Interpreter's documentation is lending a hand to ChatGPT Plus users.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**LlamaIndex Migrates PandasQueryEngine**: The latest **LlamaIndex update (v0.10.29)** relocated **PandasQueryEngine** to `llama-index-experimental`, necessitating import path adjustments and providing error messages for guidance on the transition.

**AI Application Generator Garners Attention**: In partnership with T-Systems and Marcus Schiesser, **LlamaIndex** launched **create-tsi**, a command line tool to generate GDPR-compliant AI applications, stirring the community's interest with a promotional [tweet](https://twitter.com/llama_index/status/1778812761893650551).

**Redefining Database and Retrieval Strategies**: Community exchanges delved into ideal vector databases for similarity searches, contrasting Qdrant, pg vector, and Cassandra, along with discussions on leveraging hybrid search for multimodal data retrieval, referencing the [LlamaIndex vector stores guide](https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/).

**Tutorials and Techniques to Enhance AI Reasoning**: Articles and a [tutorial](https://medium.com/ai-advances/enhancing-document-retrieval-with-memory-a-tutorial-for-llamaindex-with-colbert-based-agent-1c3c47461122) shared showcased methods to fortify document retrieval with memory in Colbert-based agents and integrating small knowledge graphs to boost **RAG systems**, as highlighted by [WhyHow.AI](https://medium.com/enterprise-rag/the-role-of-small-vs-big-knowledge-graphs-995d9449c208).

**Community Commendations and Technical Support**: Community appreciation for articles on advancing AI reasoning was voiced, while **LlamaIndex** users tackled technical woes and encouraged proactive contribution to documentation, reinforcing the platform's dedication to knowledge sharing and support.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Stacking 4090s Slices TinyBox Expenses**: Enthusiasm bubbled over the **RTX 4090 driver patch** reaching the Hacker News top story, signaling its breakthrough in GPU efficiency. When stacked, **RTX 4090s** offer a substantial reduction in costs for running large language models (LLMs), with numbers crunched to show a **36.04%** cost decrease compared to the Team Red Tinybox, and a striking **61.624%** cut versus the Team Green Tinybox.

- **Tinygrad Development Heats Up**: **George Hotz**, along with the Tinygrad community, is pushing forward in enhancing the **tinygrad documentation**, specifically targeting developer experience and clarity in error messaging. A consensus was reached to increase the code line limit to *7500* lines to support new backends, addressing a flaky mnist test and non-determinism in the scheduler.

- **Kernel Fusion and Graph Splitting in Tinygrad**: The intricacies of **Tinygrad's** kernel generation got dissected with insights revealing that graph splits during fusion can arise from reasons such as broadcasting needs, detailed in the `create_schedule_with_vars` function. Users learned that **Tinygrad** is capable of running heavyweight models like Stable Diffusion and Llama, though not directly used for their training.

- **Colab and Contribution Stories**: AI Engineers swapped tips on setting up and leveraging **Tinygrad on Google Colab**, with the recommended installation command being `pip install git+https://github.com/tinygrad/tinygrad`. Challenges with tensor padding and transformer models surfaced, with a workaround of using `None` for padding cited as a practical solution.

- **Encouraging Contributions and Documentation Deep Dives**: To assist rookies, links to **Tinygrad documentation** and personal study notes were distributed, demystifying the engine’s gears. Questions about contributing sans CUDA support were welcomed, highlighting the importance of prior immersion in the Tinygrad ecosystem before chipping in.

Relevant links for further exploration and understanding included:
- [Google Colab Setup](https://colab.research.google.com/drive/1vy97ByB5mLielmE8TzlnnaSZtMUykm-A?usp=sharing),
- [Tinygrad Notes on ScheduleItem](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/scheduleitem.md),
- [Tinygrad Documentation](https://github.com/tinygrad/tinygrad/tree/master/docs),
- [Tinygrad CodeGen Notes](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/codegen.md),
- [Tinygrad GitHub Repository](https://github.com/tinygrad/tinygrad).



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**Hugging Face Collections Streamline Artifact Organization**: [Hugging Face collections](https://huggingface.co/collections/natolambert/2024-interconnects-artifacts-6619a19e944c1e47024e9988) have been introduced to aggregate artifacts from a blog post on open models and datasets. The collections offer ease of re-access and come with an API as seen in the [Hugging Face documentation](https://huggingface.co/docs/huggingface_hub/en/package_reference/collections).

**The "Incremental" Update Debate and Open Data Advocacy**: Community members are divided on the importance of the transition from Claude 2 to Claude 3, and there's a push to remember the value of open data initiatives that may be getting overshadowed. Meanwhile, AI release announcements appear in force, with Pile-T5 and WizardLM 2 amongst the front runners.

**Synthesis of Machine Learning Discourse**: Conversations touched on obligations with ACL revision uploads, the benefits of "big models," and distinctions between critic vs reward models—with Nato's RewardBench project being a point of focus. A [tweet](https://twitter.com/andre_t_martins/status/1779263540785725737) from @andre_t_martins provided clarity on the ACL revision uploads process.

**Illuminating Papers and Research**: Key papers highlighted include "CodecLM: Aligning Language Models with Tailored Synthetic Data" and "LLaMA: Open and Efficient Foundation Language Models" with Hugging Face identifier [2302.13971](https://huggingface.co/papers/2302.13971). Synthetic data's role and learning from stronger models were key takeaways from the discussions.

**Graphs Garner Approval, and Patience Is Proposed for Bots**: In the realm of newsletters, graphs won praise for their clarity, with a commitment to future enhancement and integration into a Python library. A lone mention was made of an experimental bot that might benefit from patience rather than premature intervention.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

**Haiku's Speed Hiccup**: Engineers discussed **Haiku's** slower total response times in contrast to its throughput, suggesting **Bedrock's** potential as an alternative despite speed concerns there as well.

**Claude's RP Constraints**: Concerns arose as **Claude** refuses to engage with roleplay prompts such as being a warrior maid or sending fictional spam mail, even after using various prompting techniques.

**Jailbreak Junction**: Amid discussions, a tweet by *@elder_plinius* was shared about a universal jailbreak for **Claude 3** to enable edgier content which bypasses the strict content filters like those in **Gemini 1.5 Pro**. The community appears to be evaluating its implications; the tweet is available [here](https://x.com/elder_plinius/status/1773455789056745782?s=46).

**Code Competence Claims**: A newer version of an unnamed tool is lauded for its enhanced coding abilities and speed, with a member considering to reactivate their **ChatGPT Plus** subscription to further test these upgrades.

**Claude's Contextual Clout**: Despite improvements in other tools, **Claude** maintains its distinction with long context window code tasks, implying limitations in ChatGPT's context window size handling.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

**Mixtral Mastery or Myth?**: Community discussions touched on uncertainties in the training and finetuning efficiency of **Mixtral MoE models**, with some suspecting undisclosed techniques behind its performance. Interest was shown in weekend ventures of finetuning with en-de instruct data, and **Envoid's "fish" model** was mentioned as a curious case for **RP/ERP applications in Mixtral**, albeit untested due to hardware limitations.

**Llama-Tokenizer Tinkering Tutorial Tips**: Efforts to optimize a custom **Llama-tokenizer** for small hardware utilization led to shared resources such as the `convert_slow_tokenizer.py` from [Hugging Face](https://github.com/huggingface/transformers/blob/fe2d20d275d3591e2619a1adb0fa6ae272605208/src/transformers/convert_slow_tokenizer.py#L534) and the `convert.py` script with `--vocab-only` option from [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp/blob/master/convert.py). Additionally, there's a community call for **copyright-free EU text and multimodal data** for training large open multimodal models.

**Template Tendencies of Translators**: The **occiglot/7b-de-en-instruct** model showcased template sensitivity in evaluations, with performance variations on German RAG tasks due to template correctness as indicated by Hugging Face's [correct template](https://huggingface.co/datasets/deutsche-telekom/Ger-RAG-eval#occiglotocciglot-7b-de-en-instruct) usage.

**Training Parables from StableLM and MiniCPM**: Insights on pretraining methods were highlighted, referencing StableLM's use of **ReStruct dataset** inspired by this [ExpressAI GitHub repo](https://github.com/ExpressAI/reStructured-Pretraining), and MiniCPM's preference for mixing data like OpenOrca and EvolInstruct during cooldown phases detailed in [Chapter 5](https://arxiv.org/abs/2404.06395) of their study.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

**Burnai Ignites Interest**: Rust enthusiasts in the community are pointing out the underutilized potential of the [Burnai project](https://burn.dev/) for optimal inference, sparking questions about Mozilla's lack of engagement despite Rust being their creation.

**Llamafile Secures Mcaffee's Trust**: The *llamafile 0.7 binary* has successfully made it to Mcaffee's whitelist, marking a win for the project's security reputation.

**New Collaborator Energizes Discussions**: A new participant has joined the fold, eager to dive into collaborations and knowledge-sharing, signaling a fresh perspective on the horizon.

**Curiosity Peaks for Vulkan and tinyblas**: Intrigue is brewing over potential Vulkan compatibility in the anticipated v0.8 release and the benefits of upstreaming *tinyblas* for ROCm applications, indicating a concerted focus on performance enhancements.

**Help Wanted for Model Packaging**: Demand for guidance on packaging custom models into llamafile has led to community exchanges, giving rise to contributions like a [GitHub Pull Request on container publishing](https://github.com/Mozilla-Ocho/llamafile/pull/59#issuecomment-1840814790).



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **Spam Slaying Bot Suggestion**: A recommendation was made to integrate the **wick bot** into the server to automatically curb spam issues.
- **DMs Desired for Debugging**: One user sought assistance with their coding problem, signaling a desire for direct collaboration through DM, and used their request to verify their humanity against possible bot suspicion.
- **Rethinking Discord Invites**: The suggestion arose that Discord invites should be banned to prevent complications within the server, although no consensus or decision was reported.
- **Project Pulse Check**: A user inquired about the status of a project, questioning its activity with a simple "Is the project still alive?" without additional context.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Code-Jamba-v0.1 Shows Multi-Lingual Mastery**: The model [Code-Jamba-v0.1](https://huggingface.co/ajibawa-2023/Code-Jamba-v0.1), trained on **2 x H100 GPUs** for **162 hours**, excels in creating code for **Python, Java, and Rust**, leveraging datasets like [Code-290k-ShareGPT](https://huggingface.co/datasets/ajibawa-2023/Code-290k-ShareGPT) and [Code-Feedback](https://huggingface.co/datasets/m-a-p/Code-Feedback).
  
- **Call for Code-Jamba-v0.1 Evaluation Data**: Members highlighted the need for performance benchmarks for **Code-Jamba-v0.1**, reflecting on the utility of AI21 Labs' dataset for enhancing other Large Language Models.

- **Eager Ears for Jamba API Announcement**: A member's question about an upcoming **Jamba API** hinted at a potential integration with Fireworks AI, pending discussions with their leadership.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **AI's Password Persuasion**: [PasswordGPT](https://passwordgpt.io/) emerges as an intriguing game that challenges players to persuade an AI to disclose a password, testing natural language generation's limits in simulated social engineering.
- **Annotation Artistry Versus Raw LLM Analysis**: There's a split in community preference over whether to meticulously annotate datasets prior to modeling for better comprehension or to lean on the raw predictive power of large language models (LLMs).
- **Historical Data Deciphering Pre-LLM Era**: A shared effort was recognized where a team used pre-LLM transformer models for the meticulous extraction of records from historical documents, displaying cross-era AI application.
- **Enhanced User Engagement Through Open Prompts**: The consensus leans towards favoring open-prompt AI demos which could lead to deeper user engagement and a heightened ability to "outsmart" the AI, by revealing the underlying prompts.



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **Scaling the AI Frontier**: An upcoming meetup titled "**Unlock the Secrets to Scaling Your Gen AI App to Production**" was announced, presenting a significant challenge for scaling Gen AI apps. Featured panelists from [Portkey](https://portkey.ai/), [Noetica AI](https://www.noetica.ai/), and [LastMile AI](https://lastmileai.dev/) will share insights, with the registration link available [here](https://lu.ma/llms-in-prod-nyc).



---

# PART 2: Detailed by-Channel summaries and links



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1228284147433209898)** (1129 messages🔥🔥🔥): 

- **Stable Diffusion Discussions Center Around SD3 and Efficiency**: Conversations focus heavily on the implications and capabilities of the upcoming Stable Diffusion 3 (SD3), with members eagerly anticipating its release and functionalities. The [example video](https://www.youtube.com/watch?v=mQSKoAEaIJA) provided showcases perceived improvements in quality, while discussions also delve into the effectiveness of SD Forge for less powerful GPUs and the upcoming release schedule for SD3.

- **Pixart-Sigma, ComfyUI, and T5 Conditioning**: Users are exploring Pixart-Sigma with T5 conditioning in the [ComfyUI environment](https://github.com/city96/ComfyUI_ExtraModels), discussing VRAM requirements and performance, with mentions that in 8-bit compression mode, T5 doesn't exceed 10.3GB VRAM. The prompt adherence, generation speed, and samplers such as 'res_momentumized' for this model are topics of interest, with shared insights to optimize generation quality.

- **ControlNet, Lora, and Outpainting Queries**: The community is exchanging advice on various AI tools including ControlNet, Lora, and outpainting features for project-specific needs such as filling backgrounds behind objects. Inquiries about the possibility of keeping generated image colors consistent when using ControlNet have been raised.

- **CPU vs. GPU Performance Concerns**: Members discuss the slower performance of Stable Diffusion (SD) when running on CPU, with image generation taking significantly longer compared to using a GPU. Users share experiences and suggestions including using the `--lowvram` flag in SD to reduce GPU memory usage.

- **Community Projects and AI Features**: An artist announces the development of a painting app incorporating AI features and looks for feedback and tutorial creation support from the community. The app aims to integrate essential digital art tools with unique AI-driven functionalities, and the project overview is available [here](https://github.com/QuintessentialForms/ParrotLUX).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://rom1504.github.io/clip-retrieval/?back=https%3A%2F%2Fknn.laion.ai&index=laion5B-H-14&useMclip=false)">Clip front</a>: no description found</li><li><a href="https://tenor.com/view/why-not-both-why-not-take-both-gif-11478682">Why Not Both Take Both GIF - Why Not Both Why Not Take Both - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/nervous-courage-courage-the-cowardly-dog-tense-anxious-gif-17225992103088597327">Nervous Courage GIF - Nervous Courage Courage the cowardly dog - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://civitai.com/posts/2163684">Perturbed-Attention-Guidance Test | Civitai</a>: A post by rMada. Tagged with . PAG (Perturbed-Attention Guidance): https://ku-cvlab.github.io/Perturbed-Attention-Guidance/ PROM...</li><li><a href="https://www.tiktok.com/@edwinskeletrix/video/7355974950945164586">TikTok - Make Your Day</a>: no description found</li><li><a href="https://subpac.com/">SUBPAC - The New Way to Experience Sound: Feel it.™</a>: SUBPAC lets you feel the bass by immersing the body in low-frequency, high-fidelity physical sound, silent on the outside. Ideal for music, gaming, &amp; VR.</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1b37h5z/supir_super_resolution_tutorial_to_run_it_locally/?rdt=36399">Reddit - Dive into anything</a>: no description found</li><li><a href="https://stability.ai/">Stability AI</a>: Activating humanity's potential through generative AI.  Open models in every modality, for everyone, everywhere.</li><li><a href="https://tenor.com/bEz63.gif">Server Is For Js Javascript GIF - Server Is For Js Javascript Js - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=PqREA6-bC3w">SUPIR: New SOTA Open Source Image Upscaler &amp; Enhancer Model Better Than Magnific &amp; Topaz AI Tutorial</a>: With V8, NOW WORKS on 12 GB GPUs as well with Juggernaut-XL-v9 base model. In this tutorial video, I introduce SUPIR (Scaling-UP Image Restoration), a state-...</li><li><a href="https://leonardo.ai/">Home v2</a>: Transform your projects with our AI image generator. Generate high-quality, AI generated images with unparalleled speed and style to elevate your creative vision</li><li><a href="https://youtu.be/PqREA6-bC3w?t=1564">SUPIR: New SOTA Open Source Image Upscaler &amp; Enhancer Model Better Than Magnific &amp; Topaz AI Tutorial</a>: With V8, NOW WORKS on 12 GB GPUs as well with Juggernaut-XL-v9 base model. In this tutorial video, I introduce SUPIR (Scaling-UP Image Restoration), a state-...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1c2je28/comment/kzc0ltv/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/city96/ComfyUI_ExtraModels/issues/20">Is it possible to use Flan-T5 · Issue #20 · city96/ComfyUI_ExtraModels</a>: Would it be possible to use encoder only version of Flan-T5 with Pixart-Sigma? This one: https://huggingface.co/Kijai/flan-t5-xl-encoder-only-bf16/tree/main</li><li><a href="https://github.com/QuintessentialForms/ParrotLUX">GitHub - QuintessentialForms/ParrotLUX: Pen-Tablet Painting App for Open-Source AI</a>: Pen-Tablet Painting App for Open-Source AI. Contribute to QuintessentialForms/ParrotLUX development by creating an account on GitHub.</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1c2je28/i_got_access_to_sd3_on_stable_assistant_platform/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/KU-CVLAB/Perturbed-Attention-Guidance">GitHub - KU-CVLAB/Perturbed-Attention-Guidance: Official implementation of &quot;Perturbed-Attention Guidance&quot;</a>: Official implementation of &quot;Perturbed-Attention Guidance&quot; - KU-CVLAB/Perturbed-Attention-Guidance</li><li><a href="https://youtu.be/rZC65Q3F_Mk">Warp Drives: New Simulations</a>: Learn more from a science course on Brilliant! First 30 days are free and 20% off the annual premium subscription when you use our link ➜  https://brilliant....</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1c35onn/remember_when_emad_said_sd3_looks_better_than/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/PKU-YuanGroup/MagicTime">GitHub - PKU-YuanGroup/MagicTime: MagicTime: Time-lapse Video Generation Models as Metamorphic Simulators</a>: MagicTime: Time-lapse Video Generation Models as Metamorphic Simulators - PKU-YuanGroup/MagicTime</li><li><a href="https://www.youtube.com/watch?v=vxRY_9bv8sc">Dungeons &amp; Dancemusic</a>: 🧙🏻‍♂️🧙🏻‍♂️🧙🏻‍♂️Welcome, young beatmaker! Racso, the dancefloor sorceror, will teach you the six essential attributes of a powerful dancefloor beat. Str...</li><li><a href="https://github.com/city96/ComfyUI_ExtraModels">GitHub - city96/ComfyUI_ExtraModels: Support for miscellaneous image models. Currently supports: DiT, PixArt, T5 and a few custom VAEs</a>: Support for miscellaneous image models. Currently supports: DiT, PixArt, T5 and a few custom VAEs - city96/ComfyUI_ExtraModels</li><li><a href="https://www.reddit.com/r/StableDiffusion/s/dJpuPy7qOV">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge">GitHub - lllyasviel/stable-diffusion-webui-forge</a>: Contribute to lllyasviel/stable-diffusion-webui-forge development by creating an account on GitHub.</li><li><a href="https://civitai.com/models/311874/socks-segmentation-adetailer-sockssegsyolov8">Socks Segmentation - ADetailer - (socks_seg_s_yolov8) - v1.0 | Stable Diffusion Other | Civitai</a>: A Yolov8 detection model that segments socks in images. The model can be used as an ADetailer model (for Automatic1111 / Stable Diffusion use), or ...</li><li><a href="https://huggingface.co/datasets/MohamedRashad/midjourney-detailed-prompts">MohamedRashad/midjourney-detailed-prompts · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1228264914037244025)** (879 messages🔥🔥🔥): 

- **Context Window and RAG Discussions**: Users discussed how the context window works, noting how **Claude Opus** successfully follows instructions, while GPT-4 may lose context after several follow-ups. RAG is said to retrieve relevant parts of the document only, resulting in a much larger context for file uploads.

- **Switching Between Models and Perplexity Features**: Some users are considering whether to continue using **Perplexity Pro** or **You.com**, weighing the pros and cons, such as RAG functionality, long context handling, and code interpretation features between the platforms.

- **Issues with Generating Code**: Users reported problems with models not outputting entire code snippets when prompted – mentioning **Claude Opus** as more reliable for generating complete code compared to **GPT-4**, which tends to be descriptive rather than executable.

- **AI Model Evaluations**: Users discussed the effectiveness of various AI models for different tasks, with opinions that **Claude Opus** is better for prose writing and **GPT-4** for logical reasoning. They emphasized trial and error to find the best-suited model for specific needs.

- **Miscellaneous Queries and Concerns**: Users asked about getting assistive responses regarding account issues and unwanted charges, mentioned technical problems with models like hallucinations and incorrect searches, brought up privacy concerns with new AI services, and expressed interest in the possibility of new models like **Reka** and features like video inputs in relation to costs and context limits.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.ai/blog/grok-1.5v">Grok-1.5 Vision Preview</a>: no description found</li><li><a href="https://retrododo.com/first-official-emulator-igba-now-on-apple-app-store-to-download/">First Official Emulator &quot;iGBA&quot; Now On Apple App Store To Download</a>: A new Game Boy emulator has officially landed on the Apple App Store, allowing retro gamers to play their favourite games of the past.</li><li><a href="https://www.forbes.com.mx/como-si-wikipedia-y-chatgpt-tuvieran-un-hijo-asi-es-la-startup-buzzy-ai-madre-de-perplexity-el-buscador-que-acabaria-con-google/">‘Como si Wikipedia y ChatGPT tuvieran un hijo’: así es la startup Buzzy AI, ‘madre’ de Perplexity, el buscador que acabaría con Google</a>: Perplexity, un motor de búsqueda impulsado por inteligencia artificial, cuenta con el respaldo de personalidades tecnológicas como Jeff Bezos y cuenta con multimillonarios como el director ejecutivo d...</li><li><a href="https://www.lavanguardia.com/andro4all/tecnologia/openai-planea-lanzar-un-buscador-para-competir-con-google">OpenAI planea lanzar un buscador para competir con Google</a>: La empresa dirigida por Sam Altman estaría desarrollando un buscador que usase la IA para realizar búsquedas más precisas que uno convencional.</li><li><a href="https://docs.perplexity.ai/docs/rate-limits">Rate Limits</a>: no description found</li><li><a href="https://tenor.com/view/take-my-money-fry-futurama-meme-gif-16499696">Take My Money Fry GIF - Take My Money Fry Futurama - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://monica.im/home">Monica - Your ChatGPT AI Assistant Chrome Extension</a>: no description found</li><li><a href="https://vm.tiktok.com/ZGeuW63qm/">TikTok - Make Your Day</a>: no description found</li><li><a href="https://www.cognosys.ai/">Cognosys</a>: Meet the AI  that simplifies tasks and speeds up your workflow. Get things done faster and smarter, without the fuss</li><li><a href="https://vm.tiktok.com/ZGeuWJMes/">TikTok - Make Your Day</a>: no description found
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1228310073898438698)** (23 messages🔥): 

- **Rendezvous with US Policies**: Members are investigating US policies as someone shared a [Perplexity search link](https://www.perplexity.ai/search/US-is-considering-lJ9faQytRx.6RItBXyKFSQ) that leads to related inquiries.
- **Exploring the Virtue of Honesty**: A link to a [Perplexity AI search](https://www.perplexity.ai/search/Why-honesty-is-I6x.NhtaQ5K.BycdYIXwrA) was shared, suggesting a discourse on the importance of honesty.
- **Understanding Durable Functions**: Participants are delving into the world of durable functions through a shared [search query](https://www.perplexity.ai/search/what-is-durable-XjhaEk7uSGi7.iVc01E1Nw) on Perplexity AI.
- **Meta Mimics Perplexity for WhatsApp AI**: In an article, [Meta's AI on WhatsApp](https://analyticsindiamag.com/meta-releases-ai-on-whatsapp-looks-like-perplexity-ai/), compared to Perplexity AI, has been linked for insights into the expanding use of AI in messaging platforms.
- **Scratchpad-Thinking and AutoExpert Unite**: There's talk on combining *scratchpad-think* with *autoexpert* functionalities, highlighted by a [Perplexity search](https://www.perplexity.ai/search/for-a-liter-ucusD8pxSDqUo0B0x_3y0w).

**Link mentioned**: <a href="https://analyticsindiamag.com/meta-releases-ai-on-whatsapp-looks-like-perplexity-ai/">Meta Releases AI on WhatsApp, Looks Like Perplexity AI</a>: Meta has quietly released its AI-powered chatbot on WhatsApp, Instagram, and Messenger in India, and various parts of Africa.

  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1228423998497361920)** (26 messages🔥): 

- **Perplexity API Roadmap Revealed**: Perplexity AI's roadmap, updated 6 days ago, outlines features planned for release in June such as enforcing JSON grammar, N>1 sampling, a new Databricks model, a Model Info endpoint, a status page, and multilingual support. The documentation is available on their [feature roadmap page](https://docs.perplexity.ai/docs/feature-roadmap).

- **Taking the Temperature of Default Settings**: When querying the Perplexity API without specifying temperature, the default value is considered to be 0.2 for the 'online' models, which might be unexpectedly low for other models. Users are recommended to specify the temperature in their API requests to avoid ambiguity.

- **Seeking Clarity on Citations Feature**: Several users inquired about how and when citations will be implemented in Perplexity API, engaging in discussions and sharing links from the Perplexity documentation regarding the application process for features.

- **Circular References in Response to API Questions**: A user seeking help about not receiving URLs in API responses for news stories was directed to a Discord channel post for information. This represents an instance of users within the community attempting to assist one another.

- **Request for API Rate Limit Increase**: A user inquired about getting a rate limit increase for the API and was guided to fill out a form available on Perplexity's website, a process appearing to involve justifying the business need for increased capacity.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.perplexity.ai/docs/feature-roadmap">Feature Roadmap</a>: no description found</li><li><a href="https://docs.perplexity.ai/discuss/65c4824891a9930010e8e9ee">Get Sources and Related Information when using API</a>: no description found</li><li><a href="https://docs.perplexity.ai/discuss/66027e76bb5a05005bd5001e">Data from perplexity web and perplexity api</a>: no description found</li><li><a href="https://docs.perplexity.ai/discuss/65fef9fdc6b367004540612b">API gives very low quality answers?</a>: no description found</li><li><a href="https://perplexity.typeform.com/to/j50rnNiB>">Discover Typeform, where forms = fun</a>: Create a beautiful, interactive form in minutes with no code. Get started for free.
</li>
</ul>

</div>
  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1228257322661580811)** (431 messages🔥🔥🔥): 

```html
<ul>
  <li><strong>Unsloth Multi-GPU Update Query</strong>: There were inquiries regarding updates on Multi-GPU support for Unsloth AI. Suggestions to look into Llama-Factory with Unsloth integration were mentioned.</li>
  <li><strong>Geohot Adds P2P to 4090s</strong>: A significant update where "geohot" has hacked P2P back into NVIDIA 4090s was shared, along with a relevant <a href="https://github.com/tinygrad/open-gpu-kernel-modules">GitHub link</a>.</li>
  <li><strong>Upcoming Unsloth AI Demo and Q&A Event Alert</strong>: An announcement for a live demo of Unsloth AI with a Q&A session by Analytics Vidhya was shared. Those interested were directed to join via a posted <a href="https://us06web.zoom.us/webinar/register/WN_-uq-XlPzTt65z23oj45leQ">Zoom link</a>.</li>
  <li><strong>Mistral Model Fusion Tactics Discussed</strong>: There was a discussion about the practicality of merging MOE experts into a single model, with skepticism regarding output quality. Some considered fine-tuning Mistral for narrow tasks and removing lesser-used experts as a potential compression method.</li>
  <li><strong>Hugging Face CEO Follows Unsloth on Platform X</strong>: Clement Delangue, co-founder, and CEO of Hugging Face, now follows Unsloth on an unnamed platform, sparking hopes for future collaborations between the two AI communities.</li>
</ul>
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.05405">Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws</a>: Scaling laws describe the relationship between the size of language models and their capabilities. Unlike prior studies that evaluate a model&#39;s capability via loss or benchmarks, we estimate the n...</li><li><a href="https://us06web.zoom.us/webinar/register/WN_-uq-XlPzTt65z23oj45leQ">Video Conferencing, Web Conferencing, Webinars, Screen Sharing</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...</li><li><a href="https://arxiv.org/abs/2310.08659">LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models</a>: Quantization is an indispensable technique for serving Large Language Models (LLMs) and has recently found its way into LoRA fine-tuning. In this work we focus on the scenario where quantization and L...</li><li><a href="https://huggingface.co/Vezora/Mistral-22B-v0.1">Vezora/Mistral-22B-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/AI-Sweden-Models/">AI-Sweden-Models (AI Sweden Model Hub)</a>: no description found</li><li><a href="https://huggingface.co/collections/LumiOpen/viking-660fa4c659d8544c00f77d9b">Viking - a LumiOpen Collection</a>: no description found</li><li><a href="https://lightning.ai/pages/community/lora-insights/">Finetuning LLMs with LoRA and QLoRA: Insights from Hundreds of Experiments - Lightning AI</a>: LoRA is one of the most widely used, parameter-efficient finetuning techniques for training custom LLMs. From saving memory with QLoRA to selecting the optimal LoRA settings, this article provides pra...</li><li><a href="https://developer.nvidia.com/nccl">NVIDIA Collective Communications Library (NCCL)</a>: no description found</li><li><a href="https://github.com/tinygrad/open-gpu-kernel-modules">GitHub - tinygrad/open-gpu-kernel-modules: NVIDIA Linux open GPU with P2P support</a>: NVIDIA Linux open GPU with P2P support. Contribute to tinygrad/open-gpu-kernel-modules development by creating an account on GitHub.</li><li><a href="https://github.com/astramind-ai/Mixture-of-depths">GitHub - astramind-ai/Mixture-of-depths: Unofficial implementation for the paper &quot;Mixture-of-Depths: Dynamically allocating compute in transformer-based language models&quot;</a>: Unofficial implementation for the paper &quot;Mixture-of-Depths: Dynamically allocating compute in transformer-based language models&quot; - astramind-ai/Mixture-of-depths</li><li><a href="https://github.com/OptimalScale/LMFlow/issues/726">[BUG] LISA: same loss regardless of lisa_activated_layers · Issue #726 · OptimalScale/LMFlow</a>: Describe the bug I think there might be something wrong with the current LISA implementation. There is no difference in training loss, no matter how many layers are active. Not using LMFlow but HF ...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1228628722035331083)** (38 messages🔥): 

- **Exploring PhD Life in AI**: A first-year PhD student in AI discusses the stressful yet fun experience of pursuing a doctorate, including the pressure of producing results and self-doubt. In one message, they mention their topic as creating an instruction-following LLM in their native language, noting it's a complex project beyond just translation and fine-tuning.

- **Potential Gold in AI Competitions**: The conversation shifts towards a [Kaggle AI Mathematical Olympiad](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize), offering a $10M prize for an LLM that could earn a gold medal at the International Math Olympiad. The PhD student also acknowledges the idea of getting their juniors to work on the problem.

- **The Beal Prize Discussed**: The chat includes a [link to the American Mathematical Society](https://www.ams.org/profession/prizes-awards/ams-supported/beal-prize-rules) detailing the rules for the Beal Prize, mentioning the condition for a proof or a counterexample to the Beal Conjecture and indicating the prize sum of $1,000,000.

- **Universities' Priorities Debated**: A member points out that universities focus on publishing papers and creating an "impact" rather than prize money when it comes to research topics, despite discussing significant monetary awards in competitions.

- **AI and the Nature of Language**: There's philosophical musings about thinking and language being self-referencing systems. It is suggested that thoughts self-generate and language can only be defined in terms of more language.

- **AI Tech Showcased on Instagram**: A brief interlude reveals engagement with a [shared Instagram video](https://www.instagram.com/reel/C5hP-5lAG6h/?igshid=34e80b5a5eda), praised for its content and particularly clean code, though not all were fully able to grasp it.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize">AI Mathematical Olympiad - Progress Prize 1 | Kaggle</a>: no description found</li><li><a href="https://www.instagram.com/reel/C5hP-5lAG6h/?igsh=MzRlODBiNWFlZA=="> Julia | Math &amp; Programming on Instagram: &quot;When you&#039;re a grad student #gradschool #phd #university #stem #phdstudent #machinelearning #ml #ai&quot;</a>:  6,197 likes, 139 comments -  juliaprogramming.jlApril 8, 2024 on : &quot;When you&#039;re a grad student #gradschool #phd #university #stem #phdstudent #machinelearning #ml #ai&quot;</li><li><a href="https://www.ams.org/profession/prizes-awards/ams-supported/beal-prize-rules">Beal Prize Rules and Procedures</a>: Advancing research. Creating connections.</li><li><a href="https://www.reddit.com/r/ChatGPT/comments/1c2pzqn/jumped_on_the_trend_and_asked_chatgpt_to_make_a/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1228279453969289256)** (313 messages🔥🔥): 

- **Efficient VRAM Usage and Finetuning Strategies**: Participants discussed methods for finetuning large language models like **Mistral** efficiently, with concerns about **vRAM usage** when training with long examples. There was advice to use the 30% less VRAM update from **Unsloth**, utilize **accumulation steps** to counteract large **batch size** impacts, and consider strategies like finetuning on short examples before moving to longer ones.

- **Choosing a Base Model for New Languages**: Members sought advice on finetuning on a **low-resource language**, with users sharing experiences on models like **Gemma** and discussing strategies like mixing datasets and continuing pretraining.

- **Tips and Tricks for Uninterrupted Workflow**: Users discussed troubleshooting issues with **unsloth installation** and shared **Kaggle's** and **starsupernova's** installation instructions to overcome problems such as the **CUDA not linked** error, and dependency issues with torch versions, **flash-attn**, and **xformers**.

- **Adapter Merging for Production**: There was a conversation about the proper method for merging adapters from **QLoRA** and whether to use **naive merging** or a more complex process involving **dequantization** before merging. **Starsupernova** provides wiki links on saving models to **16bit** for merging to vLLM or GGUF.

- **Dialog Format Conversion and Custom Training**: A user shared a script to convert a plain text conversation dataset into ShareGPT format in preparation to emulate personal chatting style, while recommending the use of **Unsloth** notebooks for custom training after conversion.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://huggingface.co/ura-hcmut/ura-llama-7b">ura-hcmut/ura-llama-7b · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/ghost-x/ghost-7b-v0.9.0">ghost-x/ghost-7b-v0.9.0 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/trl/sft_trainer#train-on-completions-only">Supervised Fine-tuning Trainer</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki#evaluation-loop---also-oom-or-crashing">Home</a>: 2-5X faster 80% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">Home</a>: 2-5X faster 80% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.co">GitHub: Let’s build from here</a>: GitHub is where over 100 million developers shape the future of software, together. Contribute to the open source community, manage your Git repositories, review code like a pro, track bugs and fea...</li><li><a href="https://github.com/unslothai/unsloth/wiki">Home</a>: 2-5X faster 80% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/akameswa/CodeGenerationMoE/blob/main/code/finetune.ipynb">CodeGenerationMoE/code/finetune.ipynb at main · akameswa/CodeGenerationMoE</a>: Mixture of Expert Model for Code Generation. Contribute to akameswa/CodeGenerationMoE development by creating an account on GitHub.</li><li><a href="https://download.pytorch.org/whl/cu118">no title found</a>: no description found</li><li><a href="https://github.com/huggingface/datasets/issues/6753">Type error when importing datasets on Kaggle · Issue #6753 · huggingface/datasets</a>: Describe the bug When trying to run import datasets print(datasets.__version__) It generates the following error TypeError: expected string or bytes-like object It looks like It cannot find the val...</li><li><a href="https://github.com/facebookresearch/xformers#installing-xformers)">GitHub - facebookresearch/xformers: Hackable and optimized Transformers building blocks, supporting a composable construction.</a>: Hackable and optimized Transformers building blocks, supporting a composable construction. - facebookresearch/xformers</li><li><a href="https://download.pytorch.org/whl/cu121">no title found</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: 2-5X faster 80% less memory QLoRA &amp; LoRA finetuning</a>: 2-5X faster 80% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth)</a>: no description found
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1228316423558926376)** (54 messages🔥): 

- **Low-Resource Language Enrichment Techniques**: Members discussed ways to enrich low-resource language datasets, mentioning the use of translation data and resources available on HuggingFace.
- **EEVE-Korean Model Data Efficiency Claim**: A member referenced a paper on [EEVE-Korean-v1.0](https://arxiv.org/abs/2402.14714), highlighting an efficient vocabulary expansion method building on English-centric large language models (LLMs) that purportedly requires only 2 billion tokens to significantly boost non-English language proficiency.
- **Unsloth AI Packaged in Podman Containers**: A video was shared demonstrating podman builds of containerized generative AI applications, including Unsloth AI, able to deploy locally or to cloud providers: [Podman Build - For cost-conscious AI](https://youtu.be/3iEhFKIDXp0).
- **Enhanced LLM with Ghost 7B Alpha**: Discussions highlighted the advantages of Ghost 7B Alpha in reasoning capabilities over models like ChatGPT. Further talk centered on the model's fine-tuning method which includes difficult reasoning questions, implying an orca-like approach without extending tokenizer vocabulary.
- **Intriguing Results from Initial Benchmarks**: A member shared their initial benchmarking results, prompting positive feedback and reinforcement from other participants, indicating promising performance.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2402.14714">Efficient and Effective Vocabulary Expansion Towards Multilingual Large Language Models</a>: This report introduces \texttt{EEVE-Korean-v1.0}, a Korean adaptation of large language models that exhibit remarkable capabilities across English and Korean text understanding. Building on recent hig...</li><li><a href="https://tenor.com/view/sloth-crawling-slow-gif-9915689">Sloth Crawling GIF - Sloth Crawling Slow - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/ghostx_ai">Tweet from undefined</a>: no description found</li><li><a href="https://huggingface.co/ghost-x">ghost-x (Ghost X)</a>: no description found</li><li><a href="https://youtu.be/3iEhFKIDXp0?si=HnDtELfgo4Q3cBOH">Podman Build- For cost-conscious AI</a>: A quick demo on how @podman builds containerized generative AI applications that can be deployed locally or to your preferred cloud provider. That choice is ...
</li>
</ul>

</div>
  

---



**Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1229450843569520811)** (1 messages): 

- **Pile-T5 Unveiled**: EleutherAI has released **Pile-T5**, a T5 model variant trained on 2 trillion tokens from the [*Pile*](https://blog.eleuther.ai/pile-t5/) with the **LLAMA tokenizer** which shows markedly improved performance on benchmarks such as *SuperGLUE* and *MMLU*. This model, which also performs better in code-related tasks, is available with intermediate checkpoints for both HF and T5x versions.
  
- **Modeling Details and Contributions**: **Pile-T5** was meticulously trained to 2 million steps and incorporates the *original span corruption method* for improved finetuning. The project is a collaborative effort by EleutherAI members, as acknowledged in the announcement.

- **Open Source for the Community**: All resources related to **Pile-T5**, including model weights and training scripts, are [open-sourced on GitHub](https://github.com/EleutherAI/improved-t5). This includes intermediate checkpoints, allowing the community to experiment and build upon their work.

- **Shoutout and Links on Twitter**: The release and open sourcing of **Pile-T5** have been announced on [Twitter](https://x.com/arankomatsuzaki/status/1779891910871490856), inviting the community to explore the blog post, download the weights, and contribute to the repository.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.eleuther.ai/pile-t5/">Pile-T5</a>: Trained T5 on the Pile</li><li><a href="https://github.com/EleutherAI/improved-t5">GitHub - EleutherAI/improved-t5: Experiments for efforts to train a new and improved t5</a>: Experiments for efforts to train a new and improved t5 - EleutherAI/improved-t5</li><li><a href="https://x.com/arankomatsuzaki/status/1779891910871490856">Tweet from Aran Komatsuzaki (@arankomatsuzaki)</a>: 🚀 Introducing Pile-T5!  🔗 We (EleutherAI) are thrilled to open-source our latest T5 model trained on 2T tokens from the Pile using the Llama tokenizer.  ✨ Featuring intermediate checkpoints and a si...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1228268846008504444)** (123 messages🔥🔥): 

- **Collaborative Efforts on Evals Compilation**: A participant requested assistance and collaboration in compiling and solving a list of evals, directing interested members to a shared [Google document](https://docs.google.com/document/d/1qt7GjbrFToxSIUKC9nWccvHZN9LnO8r6myMAFUX5SVQ/edit?usp=sharing).

- **NeurIPS High School Track Call for Teammates**: A member is seeking potential teammates to participate in the [new high school track of NeurIPS](https://neurips.cc/Conferences/2024/CallforHighSchoolProjects) and has invited interested parties to reach out.

- **Model Generation Techniques Discussed**: A technical discussion arose about why generating tokens one at a time yields slightly different results compared to generating all tokens at once using Hugging Face's `generate()` for models like Phi-2, with various community members proposing hypotheses and offering links to relevant GitHub [issues](https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535) and code [snippets](https://huggingface.co/microsoft/phi-2/blob/main/modeling_phi.py).

- **AI Music Generation Showcases**: A member discussed their advancements in AI-generated music and offered to generate continuations of any solo-piano piece on request, suggesting a big upcoming post in the audio-models channel.

- **Community Projects Proposal and Discussion for AGI Pathways**: An extensive debate took place regarding a member's proposal for a STRIPS scheduler-based approach to creating a "friendly AGI." Despite skepticism from some, the proposer provided a [link to their concept paper](https://arxiv.org/pdf/2304.11477) and detailed the idea, citing the urgency for alternative approaches to AI alignment and safety.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://us05web.zoom.us/j/87189451768?pwd=6PlOsbZ2AbaaIhwvaa6kjyvn0NelT2.1">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...</li><li><a href="https://docs.google.com/document/d/1qt7GjbrFToxSIUKC9nWccvHZN9LnO8r6myMAFUX5SVQ/edit?usp=sharing">[List of evals that you&#39;d like us/you to work on/explore/solve]</a>: no description found</li><li><a href="https://www.latent.space/p/transformers-math#details>">The Mathematics of Training LLMs — with Quentin Anthony of Eleuther AI</a>: Listen now | Breaking down the viral Transformers Math 101 article and high performance distributed training for Transformers-based architectures (or &quot;How I Learned to Stop Handwaving and Make th...</li><li><a href="https://github.com/elicit/machine-learning-list">GitHub - elicit/machine-learning-list</a>: Contribute to elicit/machine-learning-list development by creating an account on GitHub.</li><li><a href="https://youtu.be/e1UgzSTicuY?si=0vS7ImdlVKC2QJJd">Why I&#39;m Leaving My Company Immediately (Stability AI) w/ Emad Mostaque | EP #93</a>: In this episode, Peter and Emad discuss Emad&#39;s stepping down as CEO of StabilityAI, his next steps into decentralized AI, and why there is so much urgency to...</li><li><a href="https://github.com/huggingface/transformers/issues/25420#issuecomment-177531753">Possible Bug with KV Caching in Llama (original) model · Issue #25420 · huggingface/transformers</a>: System Info transformers==4.31.0 huggingface_hub version: 0.15.1 Platform: Linux-5.15.0-78-generic-x86_64-with-glibc2.35 Python version: 3.10.12 Running in iPython ?: No Running in notebook ?: No R...</li><li><a href="https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535">Possible Bug with KV Caching in Llama (original) model · Issue #25420 · huggingface/transformers</a>: System Info transformers==4.31.0 huggingface_hub version: 0.15.1 Platform: Linux-5.15.0-78-generic-x86_64-with-glibc2.35 Python version: 3.10.12 Running in iPython ?: No Running in notebook ?: No R...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1228290369658294293)** (534 messages🔥🔥🔥): 

- **Exploring MoE and Dense Model Capacities**: There's an ongoing debate regarding the relative performance and benefits of Mixture of Experts (MoE) versus dense transformer models, focusing on whether MoEs are strictly better when not VRAM constrained, and if dense models may actually perform better given the same parameter budget.

- **Token Unigram Model Anomaly**: A newly discussed paper proposes that, for transformers trained on high-order Markov processes without tokenization, they fail to learn the correct distribution, defaulting to a unigram model.

- **Deep Dreaming with CLIP**: Curiosity arises about applying deep dream techniques to the CLIP model, with suggestions that CLIP's causal LM structure might provide meaningful loss signals for the "dreaming" of images.

- **Revisiting DenseNet Practicality**: A recent paper revives DenseNets, suggesting outdated training methods and design elements might have previously masked their potential, now theoretically surpassing modern architectures like Swin Transformer and ConvNeXt.

- **Research Gaps in Language Models**: Participants discuss the lack of research specifically comparing what dense and MoE transformers learn, highlighting a gap in understanding the differences in the models beyond their performance metrics.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.07965">Rho-1: Not All Tokens Are What You Need</a>: Previous language model pre-training methods have uniformly applied a next-token prediction loss to all training tokens. Challenging this norm, we posit that &#34;Not all tokens in a corpus are equall...</li><li><a href="https://storm.genie.stanford.edu/">Streamlit</a>: no description found</li><li><a href="https://arxiv.org/abs/1701.06538">Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer</a>: The capacity of a neural network to absorb information is limited by its number of parameters. Conditional computation, where parts of the network are active on a per-example basis, has been proposed ...</li><li><a href="http://arxiv.org/abs/2404.08636">Probing the 3D Awareness of Visual Foundation Models</a>: Recent advances in large-scale pretraining have yielded visual foundation models with strong capabilities. Not only can recent models generalize to arbitrary images for their training task, their inte...</li><li><a href="https://arxiv.org/abs/2404.07177">Scaling Laws for Data Filtering -- Data Curation cannot be Compute Agnostic</a>: Vision-language models (VLMs) are trained for thousands of GPU hours on carefully curated web datasets. In recent times, data curation has gained prominence with several works developing strategies to...</li><li><a href="https://colab.research.google.com/drive/10rSQ_D80M4jn0-A1yv_2fOMilXdhYaVo">Google Colaboratory</a>: no description found</li><li><a href="https://arxiv.org/abs/2404.07647">Why do small language models underperform? Studying Language Model Saturation via the Softmax Bottleneck</a>: Recent advances in language modeling consist in pretraining highly parameterized neural networks on extremely large web-mined text corpora. Training and inference with such models can be costly in pra...</li><li><a href="https://arxiv.org/abs/2403.19588">DenseNets Reloaded: Paradigm Shift Beyond ResNets and ViTs</a>: This paper revives Densely Connected Convolutional Networks (DenseNets) and reveals the underrated effectiveness over predominant ResNet-style architectures. We believe DenseNets&#39; potential was ov...</li><li><a href="https://arxiv.org/abs/2309.02591">Scaling Autoregressive Multi-Modal Models: Pretraining and Instruction Tuning</a>: We present CM3Leon (pronounced &#34;Chameleon&#34;), a retrieval-augmented, token-based, decoder-only multi-modal language model capable of generating and infilling both text and images. CM3Leon uses ...</li><li><a href="https://lilianweng.github.io/posts/2024-04-12-diffusion-video/">Diffusion Models for Video Generation</a>: Diffusion models have demonstrated strong results on image synthesis in past years. Now the research community has started working on a harder task&mdash;using it for video generation. The task itself...</li><li><a href="https://arxiv.org/abs/2403.01643">You Need to Pay Better Attention</a>: We introduce three new attention mechanisms that outperform standard multi-head attention in terms of efficiency and learning capabilities, thereby improving the performance and broader deployability ...</li><li><a href="https://arxiv.org/abs/2404.08335">Toward a Theory of Tokenization in LLMs</a>: While there has been a large body of research attempting to circumvent tokenization for language modeling (Clark et al., 2022; Xue et al., 2022), the current consensus is that it is a necessary initia...</li><li><a href="http://arxiv.org/abs/2302.09057">Consistent Diffusion Models: Mitigating Sampling Drift by Learning to be Consistent</a>: Imperfect score-matching leads to a shift between the training and the sampling distribution of diffusion models. Due to the recursive nature of the generation process, errors in previous steps yield ...</li><li><a href="https://en.wikipedia.org/wiki/Predictive_coding">Predictive coding - Wikipedia</a>: no description found</li><li><a href="https://fixupx.com/Birchlabs/status/1642680652377075712">Tweet from Birchlabs (@Birchlabs)</a>: diffusion models struggle to generate images at sizes outside of training distribution fixed in #stablediffusion without retraining increased self-attn softmax denominator via extrapolation from avera...</li><li><a href="https://fixupx.com/Birchlabs/status/1660438858675224576">Tweet from Birchlabs (@Birchlabs)</a>: I turned #stablediffusion into a sound-to-image model here it depicts fire.wav. this is basically CLIP guidance, except with Meta&#39;s new ImageBind model. it accepts text, images, audio, video, dept...</li><li><a href="https://arxiv.org/abs/2402.14800">Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models</a>: A pivotal advancement in the progress of large language models (LLMs) is the emergence of the Mixture-of-Experts (MoE) LLMs. Compared to traditional LLMs, MoE LLMs can achieve higher performance with ...</li><li><a href="https://proceedings.mlr.press/v139/wies21a.html">Which transformer architecture fits my data? A vocabulary bottleneck in self-attention</a>: After their successful debut in natural language processing, Transformer architectures are now becoming the de-facto standard in many domains. An obstacle for their deployment over new modalities i...</li><li><a href="https://smallandroidphone.com/">I want a small Android phone!</a>: Do you think all Android phones are too big? I agree! We need to work together to convince someone to build a small Android phone again.</li><li><a href="https://discord.gg/AJGjXd5q">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.</li><li><a href="https://arxiv.org/abs/1412.0035">Understanding Deep Image Representations by Inverting Them</a>: Image representations, from SIFT and Bag of Visual Words to Convolutional Neural Networks (CNNs), are a crucial component of almost any image understanding system. Nevertheless, our understanding of t...</li><li><a href="https://arxiv.org/abs/1812.02903">Applied Federated Learning: Improving Google Keyboard Query Suggestions</a>: Federated learning is a distributed form of machine learning where both the training data and model training are decentralized. In this paper, we use federated learning in a commercial, global-scale s...</li><li><a href="https://fixupx.com/fly51fly/status/1779872116458020991">Tweet from fly51fly (@fly51fly)</a>: [CL] Toward a Theory of Tokenization in LLMs N Rajaraman, J Jiao, K Ramchandran [UC Berkeley] (2024) https://arxiv.org/abs/2404.08335  - Transformers trained on data from certain simple high-order Mar...</li><li><a href="https://tenor.com/view/bait-thats-bait-tom-hardy-mad-max-gif-5055384">Bait Thats Bait GIF - Bait Thats Bait Tom Hardy - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://arxiv.org/abs/2310.05209">Scaling Laws of RoPE-based Extrapolation</a>: The extrapolation capability of Large Language Models (LLMs) based on Rotary Position Embedding is currently a topic of considerable interest. The mainstream approach to addressing extrapolation with ...</li><li><a href="https://github.com/Birch-san/diffusers-play/blob/27f359c121fc5d82f305927c4f5e8a3ad963440e/src/helpers/attention/logit_scaling_attn.py#L80>">diffusers-play/src/helpers/attention/logit_scaling_attn.py at 27f359c121fc5d82f305927c4f5e8a3ad963440e · Birch-san/diffusers-play</a>: Repository with which to explore k-diffusion and diffusers, and within which changes to said packages may be tested. - Birch-san/diffusers-play</li><li><a href="https://github.com/ClashLuke/TrueGrad/blob/main/truegrad/optim.py#L378>)">TrueGrad/truegrad/optim.py at main · ClashLuke/TrueGrad</a>: PyTorch interface for TrueGrad Optimizers. Contribute to ClashLuke/TrueGrad development by creating an account on GitHub.</li><li><a href="https://github.com/Birch-san/diffusers-play/blob/27f359c121fc5d82f305927c4f5e8a3ad963440e/src/helpers/attention/wacky_softmax_attn.py#L233>">diffusers-play/src/helpers/attention/wacky_softmax_attn.py at 27f359c121fc5d82f305927c4f5e8a3ad963440e · Birch-san/diffusers-play</a>: Repository with which to explore k-diffusion and diffusers, and within which changes to said packages may be tested. - Birch-san/diffusers-play</li><li><a href="https://arxiv.org/abs/2404.05405">Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws</a>: Scaling laws describe the relationship between the size of language models and their capabilities. Unlike prior studies that evaluate a model&#39;s capability via loss or benchmarks, we estimate the n...</li><li><a href="https://arxiv.org/abs/2401.16380">Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling</a>: Large language models are trained on massive scrapes of the web, which are often unstructured, noisy, and poorly phrased. Current scaling laws show that learning from such data requires an abundance o...</li><li><a href="https://www.tensorflow.org/api_docs/python/tf/image/total_variation">no title found</a>: no description found</li><li><a href="https://arxiv.org/abs/2309.16620">Depthwise Hyperparameter Transfer in Residual Networks: Dynamics and Scaling Limit</a>: The cost of hyperparameter tuning in deep learning has been rising with model sizes, prompting practitioners to find new tuning methods using a proxy of smaller networks. One such proposal uses $μ$P p...</li><li><a href="https://www.youtube.com/watch?v=xK4tyMqthYk">Mufan Li - Infinite-Depth Neural Networks as Depthwise Stochastic Processes</a>: Abstract: Recent advances in neural network research have predominantly focused on infinite-width architectures, yet the complexities inherent in modelling n...</li><li><a href="https://www.unihertz.com/collections/jelly-series">Jelly Series</a>: Shop Jelly series smartphones and their accessories from Unihertz now! Running Android OS, these palm-sized smartphones can meet your daily needs while sparing more space for your pocket or handbag! T...</li><li><a href="https://arxiv.org/html/2403.12963v1">FouriScale: A Frequency Perspective on Training-Free High-Resolution Image Synthesis</a>: no description found</li><li><a href="https://pure.nsu.ru/portal/ru/">Обзор исследований</a>: no description found
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1228262144349311096)** (10 messages🔥): 

- **First Scaling Laws for Data Filtering Unveiled**: A tweet by @pratyushmaini announces the development of the first scaling laws for data curation, emphasizing that it *cannot* be compute agnostic. The study, presented at CVPR 2024 and co-authored with @goyalsachin007, @zacharylipton, @AdtRaghunathan, and @zicokolter, is detailed in their paper [here](https://arxiv.org/abs/2404.07177).

- **Implicit Entropy in Data Filtering Methods**: A member finds it impressive that the paper on scaling laws for heterogeneous & limited web data manages to be empirical without explicitly mentioning entropy methods, implying that the concept underpins the study.

- **Discussing the Cryptic Nature of Entropy in New Research**: There is a conversation regarding the intrinsic link between entropy and the coding scheme/model in research, suggesting an expectation of deeper analysis before any further commentary.

- **Searching for Entropy's Role in Utility Definition**: A member observes that the paper discussed earlier might be implicitly redefining entropy as 'utility', though this leads to some unconventional ways of conceptualizing it. This suggests that the empirical approach could be masking a foundational reliance on entropy.

**Link mentioned**: <a href="https://x.com/pratyushmaini/status/1778577153107570770">Tweet from Pratyush Maini (@pratyushmaini)</a>: 1/ 🥁Scaling Laws for Data Filtering 🥁  TLDR: Data Curation *cannot* be compute agnostic! In our #CVPR2024 paper, we develop the first scaling laws for heterogeneous & limited web data.  w/@goyalsach...

  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1229062395541065809)** (6 messages): 

- **Innovations in Transformer In-Context Learning**: A new paper introduces a JAX toolkit that facilitates **causal manipulations** during transformer model training. The toolkit's "clamping" capability reveals **specific subcircuits pivotal for in-context learning** and induction heads, also highlighting the unique learning dynamics where clamping of certain components changes the overall training behavior and can help avoid saddle points and phase shifts. [Read more about the research](https://x.com/scychan_brains/status/1779357918737158364?s=46).

- **Explaining ML Mechanisms with Language**: Google's **Patchscopes** framework is designed to unify methods for interpreting large language models (LLMs) by using their language capabilities to make their hidden representations more understandable. This initiative can enhance model transparency, especially to understand error-prone circumstances by creating **natural language explanations** of a model's inner workings. [Learn about Patchscopes](http://research.google/blog/patchscopes-a-unifying-framework-for-inspecting-hidden-representations-of-language-models/).

- **Optogenetics Inspires AI Research**: The term "optogenetic" mentioned in the context of AI research indicates inspiration from a biological technique that controls the activity of cells using light, offering precise control over neurons which can elucidate pathways in decision making processes. A Wikipedia link provides more detailed insight into **optogenetics**. [Discover the optogenetics technique](https://en.m.wikipedia.org/wiki/Optogenetics).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/scychan_brains/status/1779357918737158364?s=46">Tweet from Stephanie Chan (@scychan_brains)</a>: Our new paper delves into the circuits and training dynamics of transformer in-context learning (ICL) 🥳  Key highlights include 1️⃣ A new opensourced JAX toolkit that enables causal manipulations thr...</li><li><a href="https://en.m.wikipedia.org/wiki/Optogenetics">Optogenetics - Wikipedia</a>: no description found</li><li><a href="http://research.google/blog/patchscopes-a-unifying-framework-for-inspecting-hidden-representations-of-language-models/">Patchscopes: A unifying framework for inspecting hidden representations of language models</a>: no description found
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1229121918381330612)** (1 messages): 

- **Seeking optimal num_fewshot for GPQA**: A user is conducting tests on **Mistral 7B** for GPQA, seeking recommendations on the ideal `num_fewshot` setting. Mistral 7B is performing poorly at temperatures 0 and 1, with less than 10% success rate, but no established baseline `num_fewshot` was found.

- **Running subtasks independently in EQ Bench**: The same user is inquiring how to run a single subtask, specifically **creative writing**, within **EQ Bench**, as opposed to running all tasks. They referenced the instructions on the [EQ Bench GitHub page](https://github.com/EQ-bench/EQ-Bench) but needed guidance on isolating a specific subtask.

**Link mentioned**: <a href="https://github.com/EQ-bench/EQ-Bench">GitHub - EQ-bench/EQ-Bench: A benchmark for emotional intelligence in large language models</a>: A benchmark for emotional intelligence in large language models - EQ-bench/EQ-Bench

  

---


**Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1228274522034012200)** (23 messages🔥): 

- **Seeking Corporate CLA for GPT-NeoX**: A user requested a **corporate CLA** for contributing to the **GPT-NeoX** project, specifically for **TE integration** with enhancements like fused kernels and fp8. Stellaathena responded by offering to write a **custom CLA** once specific requirements are provided.
  
- **Investigating NeoX Embeddings Anomalies**: Inquiry into the **NeoX embeddings** revealed that above the vocabulary size, weight decay was not leading to values near zero, which differs from other models. The discussion revolves around the possible explanations, with suggestions that it might be due to the way weight decay is implemented or possibly a unique initialization method.

- **Clarifications on GPU Efficiency Tricks**: Stellaathena clarified that in **NeoX**, the embedding matrix is purposely oversized to increase **GPU efficiency**, but values outside of the actual vocabulary size are placeholders and should not be analyzed.

- **Rotary Embedding Insights on NeoX**: An observation was shared that **NeoX** applies **rotary embeddings** to only 25% of its entire embedding, making it distinct compared with models like **Pythia**.

- **Weight Decay Implementation Details**: The discussion suggested that **weight decay** might only affect weights that have been activated (received gradients), which means the unused dummy tokens in the **NeoX** model would not be impacted by it, potentially explaining their non-zero values.
  

---



**Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1228523999412355124)** (1 messages): 

- **Infini-attention Sparks Interest**: A member highlights the recent publication by Google titled "Infini-attention," expressing interest in its contents. The paper can be accessed at [Infini-attention Paper](https://arxiv.org/pdf/2404.07143.pdf).
  

---


**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1228262914981498920)** (21 messages🔥): 

- **Special Relativity on GitHub**: A GitHub Gist titled "special_relativity_greg_egan.md" was shared, containing code, notes, and snippets related to Special Relativity. It's available at [fullstack6209's GitHub](https://gist.github.com/fullstackwebdev/34ccaf0fb79677890c8f93a795f8472a).

- **Vector Search Speed Comparison**: A performance comparison was given for vector search queries: FFVec achieved 0.44ms, FAISS Flat at 10.58ms, FAISS HNSW at 1.81ms, and FAISS IVFPQ at 0.36ms, all tested on 100k vectors.

- **Starting an Open Source Research Lab**: There was a request for guidance on creating an open source research lab similar to Carper AI or Nous. A suggestion was made to start with a simple GitHub repository and Discord server to kick off the initiative.

- **Turn YouTube Videos Into Blog Posts**: A new tool called "youtube2blog" was introduced, capable of turning YouTube videos into blog posts using Mixtral 8x7b and Deepgram. Find it on GitHub at [S4mpl3r/youtube2blog](https://github.com/S4mpl3r/youtube2blog).

- **Building Binary-Quantization Friendly Embeddings**: A discussion took place about a tool for creating binary-quantization friendly embeddings, with a link to a related blog article explaining the importance of such embeddings in large-scale semantic search applications. The GitHub repository [carsonpo/ffvec](https://github.com/carsonpo/ffvec) was shared, alongside a HuggingFace model [carsonpoole/binary-embeddings](https://huggingface.co/carsonpoole/binary-embeddings) which is aimed at producing these embeddings.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://txt.cohere.com/int8-binary-embeddings/">Cohere int8 &amp; binary Embeddings - Scale Your Vector Database to Large Datasets</a>: Cohere Embed now natively supports int8 and binary embeddings to reduce memory cost.</li><li><a href="https://huggingface.co/carsonpoole/binary-embeddings">carsonpoole/binary-embeddings · Hugging Face</a>: no description found</li><li><a href="https://gist.github.com/fullstackwebdev/34ccaf0fb79677890c8f93a795f8472a">special_relativity_greg_egan.md</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://github.com/S4mpl3r/youtube2blog">GitHub - S4mpl3r/youtube2blog: Turn any Youtube video into a nice blogpost, using Mixtral 8x7b and Deepgram.</a>: Turn any Youtube video into a nice blogpost, using Mixtral 8x7b and Deepgram. - S4mpl3r/youtube2blog</li><li><a href="https://github.com/carsonpo/ffvec">GitHub - carsonpo/ffvec</a>: Contribute to carsonpo/ffvec development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1228390362536739007)** (29 messages🔥): 

- **Tripedal Robotic Chair Wobbles into the Spotlight**: A member shared a [link to a robotic project](https://shin0805.github.io/chair-type-tripedal-robot/) involving a three-legged chair that can walk, with an accompanying video adding charm to the conversation. The creation will be presented at RoboSoft2024 and has sparked humor among members, with mentions of its "cuteness" and light-hearted comparisons to chair abuse.

- **Rethinking AI Benchmarks**: There's a clear consensus among members that current AI benchmarks are flawed or easily manipulated, sparking discussions about new methods for evaluating AI models that cannot be gamed and the utility of domain-specific task evaluations.

- **Tax Benchmark as an Example of Potential Gaming**: One member pointed to a [tax benchmark](https://www.vals.ai/taxeval) as a current method of evaluation, but it was quickly criticized for potentially being easy to game if based on test-like questions.

- **A Recursive Approach to Measuring AI Coherency**: An idea was proposed to assess "general intelligence" by having a model join a narrative as a third-party spectator, then have models comment on that, in a potentially endless loop, to see how long coherency is maintained.

- **AGI Sauce Debate and Usefulness of Tools**: A video from InfraNodus on using a [knowledge graph to prompt an LLM](https://www.youtube.com/watch?v=uwIyYZ3En1A) was shared as a potentially significant resource for AGI development, but was met with mixed reviews about its practical usability.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.vals.ai/taxeval">Vals.ai: LegalBench</a>: Using the open dataset LegalBench, we benchmarked 13 popular open- and closed-source models to see their performance on Legal Tasks. This benchmark will be kept up-to-date as new models develop.</li><li><a href="https://x.com/shin0805__/status/1777992583396131246?s=46">Tweet from Shintaro Inoue / 井上信多郎 (@shin0805__)</a>: 『すずめの戸締まり』に登場する3本脚の椅子を再現したロボット設計，強化学習による歩容生成の論文を公開しました！  来週アメリカで開催されるRoboSoft2024にて発表します！  website - https://shin0805.github.io/chair-type-tripedal-robot/  #すずめの戸締まり</li><li><a href="https://github.com/nus-apr/auto-code-rover">GitHub - nus-apr/auto-code-rover: A project structure aware autonomous software engineer aiming for autonomous program improvement. Resolved 15.95% tasks in full SWE-bench</a>: A project structure aware autonomous software engineer aiming for autonomous program improvement. Resolved 15.95% tasks in full SWE-bench - nus-apr/auto-code-rover</li><li><a href="https://www.youtube.com/watch?v=uwIyYZ3En1A">How to Prompt LLM with a Knowledge Graph using InfraNodus</a>: In this video, I will show you how to use https://infranodus.com to prompt an LLM using a knowledge graph. I&#39;ll take GPT-4 as an example, but you can use thi...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1228268978095263877)** (382 messages🔥🔥): 

- **Superalignment Goals and Reality Check**: Conversations revolved around achieving superalignment by 2027 with speculation on whether this timeline suggests AGI could emerge before then. Members are skeptical about the feasibility and discussed the inconclusiveness of the matter.
- **Claim of GPT-4 Outperforming Command-R Plus Debated**: Discussion took place around **LMSys Arena**, where Command-R Plus was said to outperform all versions of GPT-4 when filtered for **exclusions** in rankings, leading to debates on evaluation methods and performance metrics.
- **Concerns Over Misuse of Nous Research's Name**: In a discussion about an unrelated token named OpenML, it was clarified that Nous Research is not affiliated with the token and that they had requested the name of Nous Research to be removed from any associated materials.
- **Personal Projects and Experiments Discussed**: Members shared various personal AI project experiences, such as instruction tuning a language model on a single laptop over three days and creating a RAG database. The barriers, outcomes, and best practices were the main points of discussion.
- **Usability Issues with ChatGPT Plus**: There was a discussion about the practicality of ChatGPT Plus's 40 message per 3-hour limit, with different users expressing how they manage or circumvent the restriction, and some expressing frustration with the constraints.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.marktechpost.com/2024/04/13/google-ai-introduces-codeclm-a-machine-learning-framework-fo">no title found</a>: no description found</li><li><a href="https://mihaiii.github.io/semantic-autocomplete/">SemanticAutocomplete demo</a>: no description found</li><li><a href="https://www.marktechpost.com/2024/04/13/google-ai-introduces-codeclm-a-machine-learning-framework-for-generating-high-quality-synthetic-data-for-llm-alignment/?amp">no title found</a>: no description found</li><li><a href="https://x.com/osanseviero/status/1778816205727424884?s=46">Tweet from Omar Sanseviero (@osanseviero)</a>: Welcome Zephyr 141B to Hugging Chat🔥  🎉A Mixtral-8x22B fine-tune ⚡️Super fast generation with TGI 🤗Fully open source (from the data to the UI)  https://huggingface.co/chat/models/HuggingFaceH4/zeph...</li><li><a href="https://huggingface.co/Vezora/Mistral-22B-v0.1">Vezora/Mistral-22B-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://x.com/karpathy/status/1647278857601564672">Tweet from Andrej Karpathy (@karpathy)</a>: @dsmilkov didn&#39;t follow but sounds interesting. &#34;train a linear model with sample weights to class balance&#34;...?</li><li><a href="https://www.ora.io/app/imo/olm">ORA</a>: no description found</li><li><a href="https://poe.com/Mixtral8x22b-Inst-FW">Mixtral8x22b-Inst-FW - Poe</a>: Mixtral 8x22B Mixture-of-Experts base model from Mistral AI fine-tuned by Fireworks.AI. https://fireworks.ai/models/fireworks/mixtral-8x22b-instruct-preview</li><li><a href="https://huggingface.co/chat/models/HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1/">HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1 - HuggingChat</a>: Use HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1 with HuggingChat</li><li><a href="https://gist.github.com/Hellisotherpeople/45c619ee22aac6865ca4bb328eb58faf#file-prompt_weighting_llm-py">You probably don&#39;t know how to do Prompt Engineering, let me educate you. </a>: You probably don&#39;t know how to do Prompt Engineering, let me educate you.  - blog.md</li><li><a href="https://gist.github.com/Hellisotherpeople/45c619ee22aac6865ca4bb328eb58faf#file-prompt_weighting_llm">You probably don&#39;t know how to do Prompt Engineering, let me educate you. </a>: You probably don&#39;t know how to do Prompt Engineering, let me educate you.  - blog.md</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/mistral/mixtral-8x22b-qlora-fsdp.yml">axolotl/examples/mistral/mixtral-8x22b-qlora-fsdp.yml at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1228304489748889663)** (68 messages🔥🔥): 

- **Searching for Health-Focused LLMs**: Members discussed the landscape of Large Language Models (LLMs) in the healthcare and medical domains. One highlighted resource is a paper on arXiv with authors like [Subhabrata Mukherjee](https://arxiv.org/abs/2403.13313) that addresses medical topics.

- **Greek LLMs Raising Questions**: Discussions revolved around fine-tuning LLMs for the Greek language with members debating the necessity of pretraining for low-resource languages. References included the [Meltemi Greek language model](https://huggingface.co/ilsp/Meltemi-7B-v1) and suggestions to join communities such as [Unsloth Discord](https://discord.com/invite/WSaMctCn) for collaborative model improvement.

- **Exploring Citation in RAGs**: Members inquired about state-of-the-art citation methods in Retrieval-Augmented Generation (RAG), with suggestions pointing towards models like **cmd r+** for performing line-level citations.

- **Capybara Dataset and Amplify-Instruct Method**: Inquiries into the Capybara dataset and the "Amplify-Instruct" technique led to a member stating that although the technical report is not yet published, interested parties can read the dataset card of the Capybara dataset for some details.

- **Comparing Multimodal LLM Reka**: A discussion unfolded around the capabilities of the multimodal LLM, Reka Core, comparing it to other leading models. A link to Reka's performance highlights was shared but was met with skepticism regarding closed models in comparison to open ones, regardless of benchmarks.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1nT4T7pCr_dWfgqAFyTNmhQTwoUOULpo1?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://arxiv.org/abs/2403.13313">Polaris: A Safety-focused LLM Constellation Architecture for Healthcare</a>: We develop Polaris, the first safety-focused LLM constellation for real-time patient-AI healthcare conversations. Unlike prior LLM works in healthcare focusing on tasks like question answering, our wo...</li><li><a href="https://www.reka.ai/news/reka-core-our-frontier-class-multimodal-language-model">Reka Core: Our Frontier Class Multimodal Language Model &mdash; Reka AI</a>: Launching Reka Core, our frontier-class multimodal language model!</li><li><a href="https://huggingface.co/ilsp/Meltemi-7B-v1">ilsp/Meltemi-7B-v1 · Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1c4mgda/inference_issue_using_llamacpp_and_openhermes/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://arxiv.org/abs/2304.08177">Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca</a>: Large Language Models (LLMs), such as ChatGPT and GPT-4, have dramatically transformed natural language processing research and shown promising strides towards Artificial General Intelligence (AGI). N...</li><li><a href="https://huggingface.co/ghost-x/ghost-7b-v0.9.1">ghost-x/ghost-7b-v0.9.1 · Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/18s7iw1/capybara_dataset_is">Reddit - Dive into anything</a>: no description found</li><li><a href="https://showcase.reka.ai/">Reka Core showcase</a>: Qualitative examples showcasing responses from Reka core along side other major models</li><li><a href="https://github.com/erikbern/ann-benchmarks">GitHub - erikbern/ann-benchmarks: Benchmarks of approximate nearest neighbor libraries in Python</a>: Benchmarks of approximate nearest neighbor libraries in Python - erikbern/ann-benchmarks
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1229117966076088430)** (5 messages): 

- **Clarifying the Channel's Purpose**: A link to the **RAG/Long Context Reasoning Dataset** documentation was shared in response to a query about the purpose of the channel. The link provided leads to a Google Docs sign-in page ([RAG Dataset Document](https://docs.google.com/document/d/1o8asa0hD0qK5mKkdY5riUeGm-bxKzL02--1r3MgPgdM/edit)).

- **Inquiry About Long Context Limits**: A member questioned the extent of long context capabilities, in terms of millions of documents or entire datasets like Wikipedia, but no definitive answer to the limit was provided in the messages.

**Link mentioned**: <a href="https://docs.google.com/document/d/1o8asa0hD0qK5mKkdY5riUeGm-bxKzL02--1r3MgPgdM/edit">RAG/Long Context Reasoning Dataset</a>: no description found

  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1228296399242924104)** (66 messages🔥🔥): 

- **Countdown to WorldSim's Return**: Members are eagerly anticipating the return of **WorldSim**, targeted for next Wednesday. The excitement is palpable, with discussions around the rich experiences and possibilities that the simulation offers.
  
- **Conversations Around AI and Gaming**: There has been lively talk about how LLMs might revolutionize gaming, imagining games with adaptable AI-generated content, complex magic systems, and fully destructible environments akin to titles like *Noita* and *Dwarf Fortress*.

- **The Creative Spark from WorldSim**: Users reflect on how **WorldSim** has influenced their thinking, with some expressing that its absence has inspired them to experiment with their own AI model simulations, while others fantasize about the magic-like unpredictability it brings to gameplay and storytelling.

- **Expectations Managed**: While the community is buzzing with anticipation, there are cautions to keep expectations in check. Plans for WorldSim's next version release are said to be "by Wednesday next week," but as with any development work, timelines could shift.

- **Hints at Future Enhancements**: The team behind **WorldSim** hints at new features in the next version. This tease continues to fuel community speculation and enthusiasm for what's to come.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://play.aidungeon.com/scenario/9D9o0X3tA8Vb/world-sim">AI Dungeon</a>: no description found</li><li><a href="https://tenor.com/view/interstellar-cost-little-maneuver-51years-51-gif-24426899">Interstellar Cost GIF - Interstellar Cost Little Maneuver - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/lbl-laidbackllamas-laid-back-llamas-wen-llama-llama-gif-181137193945110141">Lbl Laidbackllamas GIF - Lbl Laidbackllamas Laid-back-llamas - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1228292797765783595)** (1 messages): 

- **Mixtral Free Model Disabled**: The free variant of `Mixtral 8x22B:free` is no longer available; users are recommended to switch to the standard [Mixtral 8x22B](https://openrouter.ai/models/mistralai/mixtral-8x22b).
- **New Experimental Models for Testing**: Two new experimental models are up for testing: The [Zephyr 141B-A35B](https://openrouter.ai/models/huggingfaceh4/zephyr-orpo-141b-a35b) and [Fireworks: Mixtral-8x22B Instruct (preview)](https://openrouter.ai/models/fireworks/mixtral-8x22b-instruct-preview). These are instruct fine-tunes of Mixtral 8x22B.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/mistralai/mixtral-8x22b>)">Mixtral 8x22B by mistralai | OpenRouter</a>: Mixtral 8x22B is a large-scale language model from Mistral AI. It consists of 8 experts, each 22 billion parameters, with each token using 2 experts at a time.  It was released via [X](https://twitter...</li><li><a href="https://openrouter.ai/models/huggingfaceh4/zephyr-orpo-141b-a35b>)">Zephyr 141B-A35B by huggingfaceh4 | OpenRouter</a>: Zephyr 141B-A35B is A Mixture of Experts (MoE) model with 141B total parameters and 35B active parameters. Fine-tuned on a mix of publicly available, synthetic datasets.  It is an instruct finetune of...</li><li><a href="https://openrouter.ai/models/fireworks/mixtral-8x22b-instruct-preview>)">Fireworks Mixtral 8x22B Instruct OH by fireworks | OpenRouter</a>: The first instruct-tuned version of the latest mixture of experts from Mistral: [Mixtral 8x22B](/models/mistralai/mixtral-8x22b).  This model was finetuned on ~10K entries from [OpenHermes](https://hu...
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1228417709243502695)** (9 messages🔥): 

- **Token Purchase Troubleshooting**: There was a reported issue with purchasing tokens; however, it was clarified that this was not related to OpenRouter. The user was advised to contact **Syrax** directly.
- **Exit Chat Mode to Resolve**: A user was informed they were still in **/chat mode** and needed to issue the **/chat command** again to toggle in or out, which should resolve their issue.
- **Getting on the Showcase**: To have an app featured in the **app-showcase** section, a user was advised to ensure the app gets **enough usage**.
- **Invite to Beta Test Advanced Research Assistant**: An announcement for a new advanced research assistant and search engine named **Rubiks.ai** was made, **beta testers** are being recruited with a 2-month free premium offer including famous models like **Claude 3 Opus, GPT-4 Turbo, Mistral Large, Mixtral-8x22B**, and more. Interested parties should send direct messages for feedback and can use the promo code `RUBIX`. [Check it out here](https://rubiks.ai/)

**Link mentioned**: <a href="https://rubiks.ai/">Rubik's AI - AI research assistant & Search Engine</a>: no description found

  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1228258855226703935)** (480 messages🔥🔥🔥): 

- **Mixtral Performance Concerns and Dynamic Routing**: Users noted that **Mixtral 8x7B Instruct (nitro)** was very slow compared to the **Base**, with a fix involving rerouting by a user to improve speed and a clarifying discussion regarding the future plan to dynamically route to the highest transaction-per-second (t/s) endpoint by default.
  
- **Zephyr and Firextral Model Discussions**: There were discussions about the performance of various models, including **Zephyr 141b** being labeled as underwhelming and **Firextral-8x22B** receiving mixed reviews, with one user finding it surprisingly okay and others deeming it terrible. Users also shared contrasting experiences with text-generation models like **Gemini Pro 1.5** and debated the size and scale merits of models like **GPT-4**.

- **OpenRouter (OR) Tools/Function Calls Clarifications**: Some confusion arose over whether certain models support function calls/tool use on OpenRouter, and a user corrected that non-OpenAI models on OR can handle function calls but don't have a dedicated output field for them. Another user provided clarity on how function calls are handled on non-OpenAI models in OR, particularly within the context of OSS models.

- **Discussions on Fine-Tuning and Rapid LLM Evolution**: Conversations reflected on the rapid evolution of large language models (LLMs) and the community's anticipations for new official fine-tunes, including a potential Mixtral-8x22B instruct model. There was also recognition of the historic value of the WizardLM series and its community impact.

- **Announcement of New WizardLM-2 Series**: The announcement of the **WizardLM-2** series, including models with sizes of 8x22B, 70B, and 7B, spurred excitement in the community. Users are eagerly anticipating the deployment of the new models on OpenRouter, with the **WizardLM-2 8x22B** being compared favorably against other leading models like **GPT-4**.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://api.together.xyz',">no title found</a>: no description found</li><li><a href="https://wizardlm.github.io/WizardLM2/">WizardLM 2</a>: SOCIAL MEDIA DESCRIPTION TAG TAG</li><li><a href="https://huggingface.co/fireworks-ai/mixtral-8x22b-instruct-oh">fireworks-ai/mixtral-8x22b-instruct-oh · Hugging Face</a>: no description found</li><li><a href="https://docs.together.ai/reference/chat-completions">Chat Completions</a>: no description found</li><li><a href="https://docs.together.ai/docs/inference-models">Inference Models</a>: no description found</li><li><a href="https://huggingface.co/TheBloke/Falcon-180B-Chat-GPTQ">TheBloke/Falcon-180B-Chat-GPTQ · Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/18ljvxb/llm_prompt_format_comparisontest_mixtral_8x7b/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://docs.together.ai/docs/function-calling">Function calling</a>: no description found</li><li><a href="https://deepinfra.com/docs/advanced/function_calling">Use Function Calling with Deep Infra endpoints | ML Models | Deep Infra</a>: Find information about using Function Calling with Deep Infra endpoints, integration, and more!
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1228275170033143878)** (237 messages🔥🔥): 

- **Error Trouble and Success**: User encountered the error message `(Exit code: 139)?. Please check settings and try loading the model again.`, when attempting to start the model `lmstudio-community • codegemma`. Upon advice, turning off GPU offload resolved the issue.
- **LM Studio Model Recommendations**: For beginners, **heyitsyorkie** recommended choosing one of the 7B models from the app homepage and confirmed that **LM Studio** does not support image generation; it is a text-only tool.
- **Mixtral and Integration Thoughts**: Users discussed the idea of integrating **LM Studio** and **Stable Diffusion** where LLM would write image prompts for Stable Diffusion to generate images. **heyitsyorkie** mentioned a connection between LLM and ComfyUI for prompt generation.
- **Efficient Infinite Context Transformers**: A paper titled "Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention" was shared, outlining a new method for scaling Transformers to handle infinitely long inputs using **Infini-attention**, with benchmarks showing effectiveness on large-sequence tasks.
- **New Model Excitement and Performance Discussions**: Users shared their experiences with new models like **Command R+** and **Mixtral 8x22b**, discussing optimal setup configurations and performance across different hardware, including **M1, M3 Max, and 4090 GPUs**.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.amazon.ca/CORSAIR-Vengeance-PC4-25600-Desktop-Memory/dp/B07Y4ZZ7LQ/ref=sr_1_25?dib=eyJ2IjoiMSJ9.of-9wP0cpMNIBO5VEC3rUnPgXDDfZNeXir57w2hl0vwoj5lxbP31-GTPisdwSW4GGP71ZrL7P1KQvFl5x9FG3CLv7Svz7H9CUlfO9leCj6whiTcEZ8FGWfp27kvV-nYPyyqa0VZl3okpzcaYIsV-V-94yf6uEsR-gbwLfDfPVXJqVJAtB9ltY-88GiDXVlVOWQg4v1Nb53Wn3-Fjd0UerxqkU8PWd9LZSFHxWUC219k1JtsxKpuYEIy2ZtrP8r3c2kcePbcVt2QXqqjfXZEgaNIbbY4cOLkPEcgtTokbfTA.JZSm-ZSuVy4_KNhvKEpt7R7F7hB6_5TlCawp0EPeybs&dib_tag=se&hvadid=671271279297&hvdev=c&hvlocphy=9000686&hvnetw=g&hvqmt=b&hvrand=7421562605769050133&hvtargid=kwd-488359302712&hydadcr=8362_13664276&keywords=ddr4%2Bram&qid=1712950458&sr=8-25&th=1">no title found</a>: no description found</li><li><a href="https://x.com/ldjconfirmed/status/1778974720647458885">Tweet from LDJ (@ldjconfirmed)</a>: Zephyr-ORPO-141B is the first model I&#39;ve seen get this consistently right about what JEPA actually stands for. I tried this even with Claude-3-Opus and it fails too, and even the latest GPT-4-turb...</li><li><a href="https://huggingface.co/pmysl/c4ai-command-r-plus-GGUF">pmysl/c4ai-command-r-plus-GGUF · Hugging Face</a>: no description found</li><li><a href="https://missionsquad.ai/">missionsquad-web</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1atvxu2/current_state_of_training_on_amd_radeon_7900_xtx/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/search/full-text?q=Command+R%2B>">Full Text Search - Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/dranger003/c4ai-command-r-plus-iMat.GGUF/tree/main">dranger003/c4ai-command-r-plus-iMat.GGUF at main</a>: no description found</li><li><a href="https://arxiv.org/html/2404.07143v1">Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention</a>: no description found</li><li><a href="https://www.amazon.ca/CORSAIR-Vengeance-PC4-25600-Desktop-Memory/dp/B07Y4ZZ7LQ/ref=sr_1_25?dib=eyJ2I">no title found</a>: no description found</li><li><a href="https://www.amazon.co.uk/2666MHz-Desktop-PC4-21300-Unbuffered-Computer-Black/dp/B0CD7PJ4DV/?_encoding=UTF8&pd_rd_w=Fof0w&content-id=amzn1.sym.386c33bb-9a6d-4a4d-9a06-3bb24cb22d5d%3Aamzn1.symc.cdb151ed-d8fe-485d-b383-800c8b0e3fd3&pf_rd_p=386c33bb-9a6d-4a4d-9a06-3bb24cb22d5d&pf_rd_r=CCWDNV1ZK7D84Q2MAPPJ&pd_rd_wg=Et1Qj&pd_rd_r=7c25eec1-65d6-4330-a90d-005da4542c5f&ref_=pd_gw_ci_mcx_mr_hp_atf_m">no title found</a>: no description found</li><li><a href="https://www.newegg.ca/kingston-64mb/p/N82E16820242771?item=N82E16820242771&source=region&nm_mc=knc-googleadwordsca-pc&cm_mmc=knc-googleadwordsca-pc-_-pla-_-memory+%28server+memory%29-_-N82E16820242771&utm_source=google&utm_medium=paid+shopping&utm_campaign=knc-googleadwordsca-pc-_-pla-_-memory+%28server+memory%29-_-N82E16820242771&id0=Google&id1=12409815000&id2=119872896404&id3=&id4=&id5=pla-2281720576259&id6=&id7=9000686&id8=&id9=g&id10=c&id11=&id12=CjwKCAjwt-OwBhBnEiwAgwzrUr4leQFCH4rFOGg64jUHpprq-H_yAxFNPhLghRNlWhft-so2HWUnaxoCoa0QAvD_BwE&id13=&id14=Y&id15=&id16=500489819465&id17=&id18=&id19=&id20=&id21=pla&id22=102503995&id23=online&id24=N82E16820242771&id25=CA&id26=2281720576259&id27=Y&id28=&id29=&id30=1484129855650391095&id31=en&id32=&id33=&id34=&gad_source=1&gclid=CjwKCAjwt-OwBhBnEiwAgwzrUr4leQFCH4rFOGg64jUHpprq-H_yAxFNPhLghRNlWhft-so2HWUnaxoCoa0QAvD_BwE">Kingston 64GB DDR5 4800 ECC Registered DIMM Memory - Newegg.com</a>: Buy Kingston 64GB DDR5 4800 ECC Registered DIMM Memory with fast shipping and top-rated customer service. Once you know, you Newegg!</li><li><a href="https://www.newegg.ca/kingston-64mb/p/N82E16820242771?item=N82E16820242771&sou">Kingston 64GB DDR5 4800 ECC Registered DIMM Memory - Newegg.com</a>: Buy Kingston 64GB DDR5 4800 ECC Registered DIMM Memory with fast shipping and top-rated customer service. Once you know, you Newegg!</li><li><a href="https://www.youtube.com/watch?v=a75TC-w2aQ4">NEW Mixtral 8x22b Tested - Mistral&#39;s New Flagship MoE Open-Source Model</a>: Mistral AI just launched Mixtral 8x22, a massive MoE open-source model that is topping benchmarks. Let&#39;s test it!Join My Newsletter for Regular AI Updates 👇...</li><li><a href="https://github.com/rjmacarthy/twinny">GitHub - rjmacarthy/twinny: The most no-nonsense locally hosted (or API hosted) AI code completion plugin for Visual Studio Code, like GitHub Copilot but 100% free!</a>: The most no-nonsense locally hosted (or API hosted) AI code completion plugin for Visual Studio Code, like GitHub Copilot but 100% free! - rjmacarthy/twinny
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1228306022905680003)** (87 messages🔥🔥): 

- **Command R+ Gains Popularity for Perceptive Analysis**: Members have observed that **Command R+** provides perceptive analysis on a 22000 token document, rivaling or surpassing the abilities of comparable models like **ChatGPT 4**.
- **Hunting for Large LLMs**: Users discuss optimal models for text generation on specific hardware configurations, pointing to models like **CMDR+**, **Goliath 120B**, and **MiquLiz 120B** for servers with AVX2 instructions, 125B+ capabilities, and recommending checks like `nvidia-smi nvlink --status` for NVLink status on NVIDIA setups.
- **Massive GGUF Archive Arrival**: A user announces a new GGUF archive with 320 models including categories such as Mini MOEs, Super MOEs, and various high-quality quantized models with a link to their collection on [Hugging Face](https://huggingface.co/DavidAU).
- **Quantization and Quality Trade-offs**: Discussions around the performance of quantized models like **Command R Plus** indicate that severe quantization (iQ3XS, iQ1S) may lead to small performance gains but could compromise output quality significantly.
- **Coding Focused LLMs Explored**: Members seek and recommend various large language models tailored for programming-related tasks across languages such as Python and Java, pointing to options like **WaveCoder** series and **MythoMax** for coding assistance.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/microsoft/wavecoder-ultra-6.7b">microsoft/wavecoder-ultra-6.7b · Hugging Face</a>: no description found</li><li><a href="https://docs.cohere.com/docs/command-r-plus#multilingual-capabilities">Command R+</a>: no description found</li><li><a href="https://huggingface.co/TheBloke/MythoMax-L2-13B-GGUF">TheBloke/MythoMax-L2-13B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/DavidAU">DavidAU (David Belton)</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/DavidAU?sort_models=created#models">DavidAU (David Belton)</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1228548452783357963)** (30 messages🔥): 

- **Model Loading Frustration**: A user reported issues with [loading a model](https://releases.lmstudio.ai/mac/arm64/0.2.18/latest/LM-Studio-0.2.18-arm64.dmg) into memory. They received an error stating *"Error loading model."* and detailed their system specs showing sufficient resources.
- **Downgrading Brings Success**: The same user confirmed that after downgrading to version **0.2.18 of LM-Studio**, the model loaded successfully, indicating a potential issue with the latest update.
- **Speculating on Underlying Issues**: A member suggested that there might be new overhead in the underlying llama.cpp which is causing difficulties in model loading with **GPU offload at 100%**.
- **UI Clutter Complaints**: The UI of an unnamed product was criticized for being cluttered and having elements covered by others; screenshots were mentioned but not provided.
- **Monitor Size Not the Culprit**: In response to UI issues, it was pointed out that aspect ratio might play a more significant role than monitor size, recognizing that a user had a 32-inch monitor and still faced layout problems.
  

---


**LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1229462981604020266)** (11 messages🔥): 

- **Trouble with LM Studio Models**: A user is experiencing an error when attempting to load models in LM Studio. The error message suggests an unknown error and recommends trying a different model and/or config.

- **Seeking the Default Touch**: The same user expressed a desire to return to default settings without straining their system's GPU or RAM. They asked for guidance on running models without GPU offload due to performance concerns.

- **Persistent Model Loading Error**: Despite turning off GPU offload, the user is still facing an error when loading models after adjustments were made to default settings. They reported the inability to load even 12 GB models which used to work previously with 32 GB RAM.

- **A Call for a Clean Slate**: The user is asking for assistance to completely remove all LM Studio data from their PC to allow a fresh reinstall and restoration of default settings. This request follows persistent errors and issues with model loading.


  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1228312813571932260)** (53 messages🔥): 

- **Open GPU Kernel Modules on GitHub**: A link to **tinygrad/open-gpu-kernel-modules** was shared, which is an NVIDIA Linux open GPU project with P2P support available on GitHub. View the details and contributions [here](https://github.com/tinygrad/open-gpu-kernel-modules).

- **Understanding CPU vs GPU Inference**: Discussions clarify that CPU inference is slower than GPU inference, largely due to its lack of parallelization capabilities, with one member noting that more RAM allows for loading bigger models but doesn't necessarily increase the speed of CPU inference.

- **Hardware Requirements for AI Depth**: A member expressed that to get a similar experience to GPT-4, significant investment in hardware is required, potentially in the range of six figures, and even high-end laptops like those with **M1/2/3** chips and above 64GB RAM may not provide this level of performance.

- **GPU Inferencing Concerns with LM Studio**: Several members discussed issues including running **ggml-dbrx-instruct-16x12b-iq1_s.gguf** models on an **RTX 3090 TI** and noting that DBRX models currently remain unsupported in LM Studio, leading to download and compatibility problems.

- **Technical Troubleshooting in Windows**: Members engaged in troubleshooting a CPU issue where specific cores were heavily used by Windows, with suggestions ranging from checking for malware to reducing background processes, while no definitive solution was provided.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/hardware/comments/1c2dyat/geohot_hacked_4090_driver_to_enable_p2p/?share_id=d7ey-6LE-pki-7UluvgXH">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/tinygrad/open-gpu-kernel-modules">GitHub - tinygrad/open-gpu-kernel-modules: NVIDIA Linux open GPU with P2P support</a>: NVIDIA Linux open GPU with P2P support. Contribute to tinygrad/open-gpu-kernel-modules development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1228519498894803084)** (6 messages): 

- **Model Folder Mystery in MACOS 0.2.19**: A member reports an unusual behavior in **MACOS 0.2.19** (preview C and release), where the system fails to recognize any models in their local model folder, regardless of whether it's a standard directory or symlinked NFS mounted. Rolling back to 0.2.19_previewA resolves the issue.

- **Linux Beta Left Out of ROCm Support?**: A member queries about the absence of **ROCm** support in the Linux beta, while the Windows beta does have it.

- **Confusion Over Different "Beta"s Cleared Up**: There is a brief confusion about different betas, which is clarified to be a misunderstanding—Linux and ROCm are not officially supported, making it effectively a *triply beta* situation.

- **Regretting Linux ROCm Beta Omission**: When the absence of support for ROCm on Linux is confirmed, the member expresses disappointment.
  

---


**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1228897380414259321)** (4 messages): 

- **Autogen Studio Docker Image Now Available**: A Docker image for **Autogen Studio** has been created by a member and is being kept up-to-date with [renovate](https://github.com/lludlow/autogen-studio). The image is available on GitHub and can be contributed to by the community.
- **Threadripper Pro Underutilization Mystery**: A member is experiencing underutilization with their **Threadripper Pro**, where the CPU usage is capping at about 50-56% without clear reasons why.
- **Missionsquad.ai with GUI for Model and Prompts**: Another member introduced [missionsquad.ai](https://missionsquad.ai/), a tool with a GUI that allows users to define models and prompts directly within the user interface.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://missionsquad.ai/">missionsquad-web</a>: no description found</li><li><a href="https://github.com/lludlow/autogen-studio">GitHub - lludlow/autogen-studio: Autogen studio docker</a>: Autogen studio docker. Contribute to lludlow/autogen-studio development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1228472444613033995)** (38 messages🔥): 

- **GPU Not Leveraged by LM Studio**: Despite having an **8GB Radeon 6600** with Rocm drivers installed and the GPU selected, a member voiced concerns that LM Studio 0.2.19.0 on Windows 11 was not using the GPU, unlike GPT4ALL, which maxes out the GPU utilization.
- **Radeon 6600 Not Supported**: A member responded with a short reply suggesting that the **Radeon 6600** is an *unsupported card* for Rocm in LM Studio.
- **Windows Build Woes for Llama.cpp**: Conversations around the challenges of building **llama.cpp** on Windows with IC Compiler emerged, leading to suggestions regarding the use of *WSL*, **AMD's version of BLAS**, and following specific *build instructions* provided in the shared code snippet.
- **WSL Installation on Alternate Drives**: The possibility of installing **WSL** on a drive other than the C drive was confirmed by a member, indicating it can indeed be placed on the D drive.
- **Report of Lagging in ROCm LM Studio**: A member reported experiencing lag within ROCm LM Studio, even with model unloaded and after clearing significant text chats. The detailed system specs were given, and conversation ensued about monitoring the issue and identifying potential causes such as lengthy operations or specific user actions within the application.
  

---



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1228300620755107860)** (397 messages🔥🔥): 

- **Understanding AI Models and Consciousness**: A lengthy discussion unfolded with users debating the properties and developments of AI consciousness and sentience, especially in light of AI advancements. Concerns were raised about responsible AI use and the ethical implications of potentially sentient systems.

- **Practical Coding Help with ChatGPT**: A member sought advice on why ChatGPT was providing guidelines rather than directly making code changes as requested. Despite attempts to adjust the phrasing of requests and clarifications by other members, the underlying issue remained unresolved in the conversation.

- **Prompt Hacking Competition Inquiry**: A user shared a link to the [Blade Hack](https://bladehack.tech/) competition, seeking teammates and advice on prompt hacking with LLMs. The conversation revealed a mix of curiosity and concern over the ethical aspects and skills required for such an event.

- **Background of LLMs and Neurobiology Developments**: Detailed explanations and insights were shared about the history and development of neurobiological research, how it relates to AI design, and bottlenecks in data interpretation. One user discussed their own "Mind Model" influenced by such research, offering insights into the complexities involved.

- **Issues with Gemini 1.5 and AI Self-Awareness**: A member expressed frustration with Gemini 1.5 for not recognizing its capabilities to analyze video, reflecting a broader issue with AI systems not being aware of their own features – an issue noted to be similar to that of Bing Chat.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/John_C._Lilly#%22Solid_State_Intelligence%22>">John C. Lilly - Wikipedia</a>: no description found</li><li><a href="https://bladehack.tech/">Blade Hack</a>: Blade Hack is a worldwide event focused on prompt hacking, which involves exploring and interacting with large language models (LLMs) in creative ways. Blade will cater to beginners and individuals of...</li><li><a href="https://direct.mit.edu/books/oa-monograph/3653/Language-in-Our-BrainThe-Origins-of-a-Uniquely)">Language in Our Brain: The Origins of a Uniquely Human Capacity</a>: A comprehensive account of the neurobiological basis of language, arguing that species-specific brain differences may be at the root of the human capacity 
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1228323876832940214)** (36 messages🔥): 

- **Explaining API Interaction for GPT**: A user inquired about working on a large document with GPT and was told that editing a document would require hosting the file and writing an API for GPT to interact with, as it cannot interact with documents directly.

- **Clarifying GPT-4 Model Capabilities**: There was confusion as to why the new Turbo version of GPT-4 still has a 4k token limit, which was clarified by explaining that the token limit is a memory limit and a trade-off for processing power.

- **Incorporating Vision into GPT**: A member sought advice on how to analyze images with the GPT-4 model and was directed to the API documentation that indicates 'gpt-4-turbo' is Vision-capable, requiring a script for programming languages to use it.

- **Understanding Word Count Limitations in GPT-4**: In a discussion about improving GPT-4's word count, it was mentioned that GPT models cannot inherently produce output with a specific word count and qualitative language should be used to guide the desired length of the output.

- **Token Management Strategies Shared**: To mitigate the effect of token limits on GPT's output, users shared strategies, such as summarizing conversations and using new prompts in follow-up interactions to maintain context without exceeding limitations.
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1228571377150201888)** (11 messages🔥): 

- **Intriguing Variations Between GPT-4 Versions**: Some members observe inconsistencies in responses when comparing **`gpt-4-turbo-2024-04-09`** with `gpt-4-0125-preview` and `gpt-4-vision-preview` regarding copyright adherence, suggesting possible fine-tuning aspects or version-specific behavior. 

- **GPT Fluctuations Suggest Temporary Instabilities**: A user shares an experience where GPT models initially failed to retrieve content from scholarly papers, which later resolved without intervention, hinting at potential "live fine-tuning" causing transient issues.

- **Insight Into Custom vs. Default GPT-4 Models**: One member ponders whether Custom GPTs and the default ChatGPT4 differ fundamentally and whether this necessitates a distinct prompting style or refinement technique for optimal interaction with Custom GPTs.

- **Meta-prompting Template Offers Alternative Workflow**: A user recommends using a meta-prompting template when creating Custom GPT models, providing a potentially advantageous method compared to the configure pane.

- **Call for Prompt Hacking Teammates**: An announcement calls for participants to form a team for an unspecified prompt hacking competition, creating an opportunity for collaboration within the community.
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1228571377150201888)** (11 messages🔥): 

- **Confusion Over Model Copyright Responses**: A user observed inconsistencies in how different versions of GPT-4 handle copyright with **`gpt-4-turbo-2024-04-09`** providing answers while **`gpt-4-0125-preview`** citing copyright laws, despite both being prompted similarly.
- **Models' Inconsistency in Accessing Scholarly Papers**: An issue was reported where models inconsistently claimed copyright on scholarly papers, but this problem appears to have been resolved in recent weeks.
- **Custom GPT Prompt Refinement Strategies**: One user shares their strategies for improving Custom GPT prompts; refining prompts within the Custom GPT execution or separately in the default ChatGPT session, noting the latter often yields better context processing and reasoning.
- **Model Behavior in Meta-Prompting**: In response to discussion on prompt refinement, **darthgustav.** recommends using a meta-prompting template in the creator pane for building GPTs, suggesting it offers a better workflow for that process.
- **Invitation to Join Prompt Hacking Competition**: **wisewander** expressed interest in forming a team to participate in a prompt hacking competition and is seeking teammates.
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1228276727659237436)** (228 messages🔥🔥): 

- **Scarlet AI Intensifies Collaboration**: Scarlet offers **collaborative AI agents** for complex task management, featuring real-time collaboration, automated context from data, and agent/tool orchestration. A participant shared the link to [Scarlet's website](https://scarletai.co), and mentioned a work-in-progress demo and an upcoming integration with Zapier NLA.

- **Perplexity's Mysterious Disappearance from LMSYS**: One member raised concerns about Perplexity "online" models being removed from LMSYS Arena, provoking discussions on Perplexity's effectiveness, with another member expressing hopes to integrate them into their tools.

- **Harnessing YouTube Content for AI**: A conversation was held about the best ways to **transcribe and summarize YouTube content**, exploring tools like Descript, youtube-dl, and Whisper with diarization, for AI model applications.

- **Limitless Aims to Redefine Wearables with Advance AI**: Rewind is rebranding to **Limitless**, promising **personalized AI** capabilities through a wearable pendant, apps, and a "confidential cloud." Concerns about local processing and end-to-end encryption were discussed in relation to this new offering.

- **Exploring Search Potential Within Generative AI**: Discussions emerged on generative AI's impact on search tools, with members mentioning Kagi, Phind, Perplexity, and new ingresses into the field like **Cohere's Compass** for data search, and tips on how to manage vectorstores for consistent and cost-effective embedding.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://scarletai.co">Scarlet</a>: no description found</li><li><a href="https://x.com/lmsysorg/status/1778555678174663100?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from lmsys.org (@lmsysorg)</a>: 🔥Exciting news -- GPT-4-Turbo has just reclaimed the No. 1 spot on the Arena leaderboard again! Woah!  We collect over 8K user votes from diverse domains and observe its strong coding & reasoning cap...</li><li><a href="https://x.com/dsiroker/status/1779857843895599383?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Dan Siroker (@dsiroker)</a>: Introducing Limitless: a personalized AI powered by what you’ve seen, said, or heard.  It’s a web app, Mac app, Windows app, and a wearable.  0:06 Reveal 0:48 Why Limitless? 1:39 Demo 3:05 Pendant 4:2...</li><li><a href="https://www.datasette.cloud/blog/2024/datasette-extract/">Extracting data from unstructured text and images with Datasette and GPT-4 Turbo - Datasette Cloud</a>: Clean data, with well defined columns and rows, is a beautiful thing - the ideal starting point for any data analysis or visualization project. Sadly, very little of the world&#x27;s interesting data ...</li><li><a href="https://x.com/aidangomez/status/1779882113573044625?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Aidan Gomez (@aidangomez)</a>: Excited to announce the Compass Beta, a very powerful multi-aspect data search system powered by a new embedding model, Compass.  We&#39;re looking for help stress-testing the model&#39;s capabilities...</li><li><a href="https://x.com/harrystebbings/status/1779910559753802010?s=">Tweet from Harry Stebbings (@HarryStebbings)</a>: 10 years ago, I started 20VC in my bedroom with no money and no network.  Today we release our 20VC with @sama @bradlightcap and the fastest growing company in history @OpenAI.  The power of the inter...</li><li><a href="https://llm-price.com/">LLM Pricing - Compare Large Language Model Costs and Pricing</a>: no description found</li><li><a href="https://x.com/hrishioa">Tweet from undefined</a>: no description found</li><li><a href="https://blog.robertelder.org/causes-of-bit-flips-in-computer-memory/#chip-orientation">What Causes Bit Flips In Computer Memory?</a>: no description found</li><li><a href="https://www.bloomberg.com/news/articles/2024-04-11/elon-musk-s-xai-seeks-up-to-4-billion-to-compete-with-openai">Bloomberg - Are you a robot?</a>: no description found</li><li><a href="https://x.com/xuefz/status/1778474016854192216">Tweet from Fuzhao Xue (@XueFz)</a>: @mejia_petit @fouriergalois oh sure， thats okay. This work is too old haha It was rejected by conferences so many times and I gave up to submit it again lol. But tbh, I indeed thought this is an impor...</li><li><a href="https://x.com/RekaAILabs/status/1779894626083864873">Tweet from Reka (@RekaAILabs)</a>: Along with Core, we have published a technical report detailing the training, architecture, data, and evaluation for the Reka models.  https://publications.reka.ai/reka-core-tech-report.pdf</li><li><a href="https://www.descript.com/blog/article/all-new-descript-backed-by-openai-startup-fund">It&#x27;s here: the all-new Descript, backed by OpenAI Startup Fund</a>: We&#x27;re releasing an all-new version of Descript and we&#x27;re announcing that the OpenAI Startup Fund will be leading our $50 million series C fundraising round.</li><li><a href="https://x.com/imranchaudhri/status/1778445220050333711?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Imran Chaudhri (@imranchaudhri)</a>: it took us a while to get to today.   now that we’re here, we can’t wait to show you what the next few years will be like @Humane —exciting times!</li><li><a href="https://barryzhang.substack.com/p/making-peace-with-llm-non-determinism">Making Peace with LLM Non-determinism</a>: Digging into Sparse MoE and GPU cycles just to realize non-determinism is not new, language is.</li><li><a href="https://x.com/pronounced_kyle/status/1779899769982464492?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Christian Keil (@pronounced_kyle)</a>: @TouristShaun Yeah that&#39;s pretty ruthless timing, I honestly love it from Dan haha</li><li><a href="https://x.com/winglian/status/1779968341332860940?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Wing Lian (caseus) (@winglian)</a>: Alright, going old fashioned while we wait for the axolotl ai domain to transfer. Sign up for access to the private beta here. https://docs.google.com/forms/d/e/1FAIpQLSd0uWGZOwviIZPoOPOAaFDv3edcCXEIG...</li><li><a href="https://x.com/rekaailabs/status/1779894622334189592?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Reka (@RekaAILabs)</a>: Meet Reka Core, our best and most capable multimodal language model yet. 🔮  It’s been a busy few months training this model and we are glad to finally ship it! 💪  Core has a lot of capabilities, and...</li><li><a href="https://x.com/yitayml/status/1779895037335343521?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Yi Tay (@YiTayML)</a>: It&#39;s been a wild ride. Just 20 of us, burning through thousands of H100s over the past months, we&#39;re glad to finally share this with the world! 💪  One of the goals we’ve had when starting Rek...</li><li><a href="https://scalingknowledge.substack.com/p/rag">Retrieval Augmented Generation Research: 2017-2024</a>: RAG literature review including: REPLUG, Fusion-in-Decoder, KNN-LM, RETRO, FLARE, Knowledge Graph Fusion-in-Decoder, SILO, WebGPT, Toolformer, Self-RAG, GRIT &amp; more</li><li><a href="https://x.com/lilianweng/status/1779914184874160170?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Lilian Weng (@lilianweng)</a>: 🎨Spent some time refactoring the 2021 post on diffusion model with new content: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ ⬇️ ⬇️ ⬇️ 🎬Then another short piece on diffusion video ...</li><li><a href="https://en.wikipedia.org/wiki/Floating-point_arithmetic#Accuracy_problems">Floating-point arithmetic - Wikipedia</a>: no description found</li><li><a href="https://github.com/MahmoudAshraf97/whisper-diarization">GitHub - MahmoudAshraf97/whisper-diarization: Automatic Speech Recognition with Speaker Diarization based on OpenAI Whisper</a>: Automatic Speech Recognition with Speaker Diarization based on OpenAI Whisper - MahmoudAshraf97/whisper-diarization</li><li><a href="https://x.com/xai/status/1778963570098855947?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from xAI (@xai)</a>: 👀 https://x.ai/blog/grok-1.5v</li><li><a href="https://www.youtube.com/watch?v=nsXyVo30bWk">LLM Office Hours, Old men complaining part three.</a>: no description found</li><li><a href="https://www.reddit.com/r/singularity/comments/1bxq84h/openai_transcribed_over_a_million_hours_of/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/openai/whisper/discussions/1515">Extremely slow performance compared with Descript · openai/whisper · Discussion #1515</a>: Understanding Descript and Whisper are targeted to different demographics I would like to understand why the abysmal difference in the speed it takes to transcribe the same audio. (30 mins wav file...</li><li><a href="https://x.com/collincornwell/status/1779662507705036819?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Collin (@CollinCornwell)</a>: smartphones are incredible and I use mine everyday, glad that our generation 1 product even has a chance against a $457.18 billion industry  we capture even 1% of that and we win @Humane   if we keep ...</li><li><a href="https://www.youtube.com/watch/nsXyVo30bWk```">LLM Office Hours, Old men complaining part three.</a>: no description found</li><li><a href="https://github.com/hrishioa/lumentis">GitHub - hrishioa/lumentis: AI powered one-click comprehensive docs from transcripts and text.</a>: AI powered one-click comprehensive docs from transcripts and text. - hrishioa/lumentis</li><li><a href="https://github.com/dimfeld/sbbp/blob/b9ed2bb73537e327eee17ea8038c7156ebe4726a/api/src/jobs/download.rs#L71">sbbp/api/src/jobs/download.rs at b9ed2bb73537e327eee17ea8038c7156ebe4726a · dimfeld/sbbp</a>: Should&#39;ve Been a Blog Post. Contribute to dimfeld/sbbp development by creating an account on GitHub.</li><li><a href="https://youtu.be/TitZV6k8zfA?si=x2CCuE2uMK6sklvK">The Worst Product I&#39;ve Ever Reviewed... For Now</a>: The Humane AI pin is... bad. Almost no one should buy it. Yet.MKBHD Merch: http://shop.MKBHD.comTech I&#39;m using right now: https://www.amazon.com/shop/MKBHDIn...</li><li><a href="https://github.com/VikParuchuri/surya">GitHub - VikParuchuri/surya: OCR, layout analysis, and line detection in 90+ languages</a>: OCR, layout analysis, and line detection in 90+ languages - VikParuchuri/surya</li><li><a href="https://www.llamaindex.ai/blog/introducing-llamacloud-and-llamaparse-af8cedf9006b">Introducing LlamaCloud and LlamaParse — LlamaIndex, Data Framework for LLM Applications</a>: LlamaIndex is a simple, flexible data framework for connecting custom data sources to large language models (LLMs).</li><li><a href="https://github.com/run-llama/llama_parse">GitHub - run-llama/llama_parse: Parse files for optimal RAG</a>: Parse files for optimal RAG. Contribute to run-llama/llama_parse development by creating an account on GitHub.</li><li><a href="https://x.com/harrystebbings/status/1779910559753802010?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Harry Stebbings (@HarryStebbings)</a>: 10 years ago, I started 20VC in my bedroom with no money and no network.  Today we release our 20VC with @sama @bradlightcap and the fastest growing company in history @OpenAI.  The power of the inter...</li><li><a href="https://youtu.be/G8T1O81W96Y?si=OHJXeiI69YSOfG57">Sam Altman &amp; Brad Lightcap: Which Companies Will Be Steamrolled by OpenAI? | E1140</a>: Sam Altman is the CEO @ OpenAI, the company on a mission is to ensure that artificial general intelligence benefits all of humanity. OpenAI is one of the fas...
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1228434569834004540)** (143 messages🔥🔥): 

- **Semantic Search and Memory Usage**: Members discussed the tradeoffs between vector size in semantic search models, particularly focusing on memory usage. Concerns were raised about why memory use can't be reduced significantly despite a smaller dimension size, with the nonlinear relationship between speed and memory being termed "weirdly nonlinear."

- **Re-ranking and Retrieval Performance**: The discussion touched upon re-ranking strategies in retrieval models and the need for an additional measure of retrieval performance. A link to a short discussion about the re-ranking pass was shared, questioning the balance between memory savings and speed against the actual performance in retrieval tasks.

- **Quantization and Model Size**: There was an in-depth conversation regarding the impact of quantization on embedding models, the benefits of reducing vector dimensionality, and the potential for improved latency. A blog post about PostgreSQL with pgvector and its tradeoffs was noted, adding to the technical depth of the discussion.

- **Potential of Multi-Modal Embeddings**: The conversation brought up the concept of multi-modal embeddings and their quirks, as well as whether optimizations applied to text embeddings could be analogously applied to other types such as image embeddings.

- **Hackathon Planning and Collaborative Projects**: Participants organized a hackathon event focused on embeddings and prompted the community for collaborative involvement. Plans and suggestions for running the hackathon, including timing and structure, were actively brainstormed.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://about.xethub.com/blog/you-dont-need-a-vector-database">XetHub | You Don't Need a Vector Database </a>: You do not need a vector databases just to host document embeddings for RAG (or retrieval augmented generation). In this post, I'll explore simpler implementations using classic information retreival ...</li><li><a href="https://getyarn.io/yarn-clip/6a6e70a4-866c-4f88-8e7d-2454c4d084d4">YARN |  |  | Video clips by quotes | 6a6e70a4 | 紗</a>:       clip with quote      Yarn is the best search for video clips by quote.     Find the exact moment in a TV show, movie, or music video you want to share.     Easily move forward or backward to get...</li><li><a href="https://jkatz05.com/post/postgres/pgvector-scalar-binary-quantization/">Scalar and binary quantization for pgvector vector search and storage |
Jonathan Katz
</a>: no description found</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0">AI In Action: Weekly Jam Sessions</a>: 2024  Topic,Date,Facilitator,Resources,@dropdown,@ UI/UX patterns for GenAI,1/26/2024,nuvic,&lt;a href=&quot;https://maggieappleton.com/squish-structure&quot;&gt;https://maggieappleton.com/squish-stru...</li><li><a href="https://interpreter-weekend.devpost.com">the Arena Online presents: Friday Night Firefight, weekend warrior edition</a>: brand new sdk on the block, got early access just for you. i&#39;m sure it&#39;ll be very hard to come up with cool things for an llm that can execute code to do.
</li>
</ul>

</div>
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1228551167009751124)** (26 messages🔥): 

- **Route to the Mojician Role Revealed**: To gain the coveted `Mojician` role, one must *build a cool open source project with Mojo*, or contribute good PRs to `modularml/mojo`. Jack notes that those with merged PRs should DM him for the role as GitHub usernames are hard to match with Discord identities.

- **Coding Discussions Belong Elsewhere**: When members dive into technical discussions such as the ease of learning programming languages, they're reminded to take the chat to dedicated channels, specifically <#1104620458168553563>.

- **MacOS Intel User Seeks to Avoid VMs**: A member shares their preference for running projects natively on MacOS Intel hardware to avoid using VMs, highlighting the importance of cross-platform compatibility in development tools.

- **Anonymity Hesitation in Open Source**: Despite being ready to make multiple PRs, a member expresses hesitation about contributing to `modularml/mojo` due to the requirement of revealing their real name, highlighting a concern for personal anonymity in public contributions.

- **Kapa AI Integration Assistance**: In a bid to enhance their private server, a member seeks help adding the kapa.ai bot, with others providing guidance and linking to relevant resources such as [Kapa AI's official website](https://www.kapa.ai/) and their [installation guide](https://docs.kapa.ai/installation-discord).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.kapa.ai/use-cases/community-engagement">kapa.ai - ChatGPT for your developer-facing product</a>: kapa.ai makes it easy for developer-facing companies to build LLM-powered support and onboarding bots for their community. Teams at OpenAI, Airbyte and NextJS use kapa to level up their developer expe...</li><li><a href="https://docs.kapa.ai/installation-discord">Discord Bot | kapa.ai docs</a>: Kapa can be installed as a bot on your Discord server. The bot allows users to ask questions in natural language about your product which improves developer experience as developers can find answers t...</li><li><a href="https://www.kapa.ai/">kapa.ai - Instant AI Answers to Technical Questions</a>: kapa.ai makes it easy for developer-facing companies to build LLM-powered support and onboarding bots for their community. Teams at OpenAI, Airbyte and NextJS use kapa to level up their developer expe...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1228478459857076284)** (6 messages): 

- **Fresh Updates from Modular**: Modular shared tweets regarding recent developments and announcements. Here are the links to their latest tweets: [Tweet 1](https://twitter.com/Modular/status/1778918965110354023), [Tweet 2](https://twitter.com/Modular/status/1779913837216719118), [Tweet 3](https://twitter.com/Modular/status/1779913865914134561), [Tweet 4](https://twitter.com/Modular/status/1779913874957086978), [Tweet 5](https://twitter.com/Modular/status/1779913908649914783), [Tweet 6](https://twitter.com/Modular/status/1779913912009597323).
  

---


**Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/)** (1 messages): 

docphaedrus: https://www.youtube.com/watch?v=1cQbu2zXTKk

Podman Pull- On basic terminal commands
  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1228283258186498089)** (218 messages🔥🔥): 

- **Exploring Advanced Language Features**: Members discussed the potential addition of conditional trait conformance in Mojo and compared potential implementations to features in Rust and Swift. There was a difference of opinion on whether to use if blocks and how the syntax should handle conditional function declarations within structs.
- **Awaiting Mojo Standard Library Expansion**: Conversations indicated that some foundational modules like threading, async I/O, and crypto are not yet available in Mojo. They exchanged ideas about workarounds and looking forward to more of the stdlib becoming open-sourced.
- **Improving Mojo Reference Syntax**: Chris Lattner mentioned ongoing work to make Mojo's `Reference` easier to use, hinting at a [future proposal](https://github.com/modularml/mojo/blob/main/proposals/inferred-parameters.md) that may simplify mutability handling.
- **A Gentle Push towards Python Compatibility**: Discussions revealed a vision where Mojo will evolve to fully support Python libraries, with *Python packages with C extensions* included in the long-term goals.
- **MLIR and Future Extensions in Discussion**: Users inquired about Mojo's approach to GPU code generation and whether an "extensions" feature like in Swift would be implemented to manage polymorphic error handling and static analysis.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.swift.org/swift-book/documentation/the-swift-programming-language/enumerations/">Documentation</a>: no description found</li><li><a href="https://xkcd.com/927/">Standards</a>: no description found</li><li><a href="https://docs.swift.org/swift-book/documentation/the-swift-programming-language/generics#Extensions-with-a-Generic-Where-Clause">Documentation</a>: no description found</li><li><a href="https://docs.swift.org/swift-book/documentation/the-swift-programming-language/protocols#Conditionally-Conforming-to-a-Protocol">Documentation</a>: no description found</li><li><a href="https://docs.modular.com/mojo/manual/functions#overloaded-functions>),">Functions | Modular Docs</a>: Introduction to Mojo `fn` and `def` functions.</li><li><a href="https://www.modular.com/blog/what-is-loop-unrolling-how-you-can-speed-up-mojo">Modular: What is loop unrolling? How you can speed up Mojo🔥 code with @unroll</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: What is loop unrolling? How you can speed up Mojo🔥 code with @unroll</li><li><a href="https://docs.modular.com/mojo/manual/functions#variadic-arguments>">Functions | Modular Docs</a>: Introduction to Mojo `fn` and `def` functions.</li><li><a href="https://docs.modular.com/mojo/lib">Mojo🔥 modules | Modular Docs</a>: A list of all modules in the Mojo standard library.</li><li><a href="https://github.com/modularml/mojo/blob/main/proposals/inferred-parameters.md">mojo/proposals/inferred-parameters.md at main · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/tairov/llama2.mojo/blob/master/llama2.mojo">llama2.mojo/llama2.mojo at master · tairov/llama2.mojo</a>: Inference Llama 2 in one file of pure 🔥. Contribute to tairov/llama2.mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/blob/main/proposals/lifetimes-and-provenance.md">mojo/proposals/lifetimes-and-provenance.md at main · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/Moosems/TkLineNums/blob/main/tklinenums/tklinenums.py">TkLineNums/tklinenums/tklinenums.py at main · Moosems/TkLineNums</a>: A simple line numbering widget for tkinter. Contribute to Moosems/TkLineNums development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/43)">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/1245)">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/blob/abf2d97e88a2cc97e3029a578d2824895a88d0c1/stdlib/src/builtin/io.mojo#L362">mojo/stdlib/src/builtin/io.mojo at abf2d97e88a2cc97e3029a578d2824895a88d0c1 · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/2308">Conditional Trait Conformance · Issue #2308 · modularml/mojo</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? I believe it would be nice for there to be conditional...</li><li><a href="https://youtu.be/pdJQ8iVTwj8?si=ML7lZfXAel9zEgj0&t=5763">Chris Lattner: Future of Programming and AI | Lex Fridman Podcast #381</a>: Chris Lattner is a legendary software and hardware engineer, leading projects at Apple, Tesla, Google, SiFive, and Modular AI, including the development of S...</li><li><a href="https://github.com/modularml/mojo/issues/43#issuecomment-1542270428).">[Feature Request] Implement union types (rather than nominal enums) · Issue #43 · modularml/mojo</a>: Summary I propose to consider adding TypeScript-style union types to Mojo — instead of Rust-style nominal enums. This approach seems like it would work well given that Mojo plans to extend Python, ...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1228586629413732423)** (23 messages🔥): 

- **Pythonic Inspiration for Terminal UIs**: Members looking for **TUI library inspiration** mentioned checking out [Textualize](https://github.com/Textualize), known for creating Rich and Textual, though some find the code flow challenging to parse.
- **Launch of llm.mojo Sparks Community Excitement**: The release of [llm.mojo](https://github.com/dorjeduck/llm.mojo), a port of Andrej Karpathy's llm.c to **Mojo**, was announced. It boasts performance improvements through **vectorization and parallelization** tweaks and is welcoming community feedback.
- **Collaborative Contributions to llm.mojo Suggested**: Community members suggest improving `llm.mojo` by keeping C implementations in sync with upstream, which would allow simultaneous availability of the C version and the Mojo port in one repository.
- **Newcomer Seeking Help with Llama Implementations Benchmarking Framework**: A newcomer requested assistance with setting up the [lamatune framework](https://github.com/tairov/lamatune) in their codespace, and was encouraged to open an issue on the repository for support.
- **Exploring gRPC Support for Mojo Code**: A community member mentioned they are obtaining promising results from functional Mojo code and inquired if anyone is working on **gRPC support** to connect it with existing C++ code for product enhancements.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/dorjeduck/llm.mojo">GitHub - dorjeduck/llm.mojo: port of Andrjey Karpathy&#39;s llm.c to Mojo</a>: port of Andrjey Karpathy&#39;s llm.c to Mojo. Contribute to dorjeduck/llm.mojo development by creating an account on GitHub.</li><li><a href="https://github.com/tairov/lamatune">GitHub - tairov/lamatune: LLama implementations benchmarking framework</a>: LLama implementations benchmarking framework. Contribute to tairov/lamatune development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-blogs-vids](https://discord.com/channels/1087530497313357884/1151418796993683477/1228387825938989188)** (4 messages): 

- **Row vs. Column Major Performance Analysis with Mojo**: A discussion initiated based on a [Modular blog post](https://www.modular.com/blog/row-major-vs-column-major-matrices-a-performance-analysis-in-mojo-and-numpy) explored the difference between row-major and column-major ordering of matrices in computer memory. An associated [Jupyter notebook on GitHub](https://github.com/modularml/devrel-extras/blob/main/blogs/mojo-row-major-column-major/row_col_mojo.ipynb) was provided for practical exploration.

- **Mojo Matrix Error Workaround**: After encountering an error while running code from the blog post's notebook, a suggestion was made to *restart the Jupyter kernel after executing each cell* due to a bug, which proved to be an effective solution.
  
- **Appreciation for Blog Insights**: A member expressed their admiration for the detailed analysis presented in the blog post, praising the contribution as excellent work.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.modular.com/blog">Modular: Blog</a>: At Modular we believe a great culture is the key to creating a great company. The three pillars we work by are Build products users love, Empower people, and Be an incredible team.</li><li><a href="https://www.modular.com/blog/row-major-vs-column-major-matrices-a-performance-analysis-in-mojo-and-numpy">Modular: Row-major vs. column-major matrices: a performance analysis in Mojo and NumPy</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: Row-major vs. column-major matrices: a performance analysis in Mojo and NumPy</li><li><a href="https://github.com/modularml/devrel-extras/blob/main/blogs/mojo-row-major-column-major/row_col_mojo.ipynb">devrel-extras/blogs/mojo-row-major-column-major/row_col_mojo.ipynb at main · modularml/devrel-extras</a>: Contains supporting materials for developer relations blog posts, videos, and workshops - modularml/devrel-extras
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/1228753951520849972)** (3 messages): 

- **MojoAPI's Compatibility with CPython**: A new member asked if MojoAPI can be used with regular CPython to improve inference times. Another user responded that `Python.module_import` can be used within Mojo to import available Python libraries, and Mojo can be leveraged to rewrite slow parts of Python code. 

- **Efficient Inference with MAX Benchmarking**: A member inquired about reducing model inference time besides using the provided bench-marking options like `--mlperf-scenario` and `--num-threads`. No response or additional suggestions were provided in the messages.
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1228475768468672585)** (55 messages🔥🔥): 

- **Mojo Nightly on Standard Library Directory Structure**: The Mojo team opened a PR to address an issue where the tools didn't understand the package name due to a discrepancy between the directory structure and package naming in the newly open-sourced standard library. The fix is set to be available in the next `nightly/mojo` release.

- **Proposed `StaticTuple` Enhancements for Modular ML**: There is ongoing discussion around updating `StaticTuple` to accept `AnyType` instead of `AnyRegType`, which would be a substantial advancement. A member has shown interest in working on this, and the idea of renaming `StaticTuple` to `Array` was floated, with emphasis on separating renaming from functionality rework in pull requests. 

- **Ongoing Unicode Support for Mojo Strings**: A member is contributing Unicode support to the Mojo standard library with a focus on minimizing memory footprint and has requested feedback on whether to proceed through a PR or a formal proposal. Standard library methods written for vectorization are mentioned as part of the further discussion.

- **Clarifications on MLIR and Mojo Dialects**: Internal discussions clarified that `pop` is an internal MLIR dialect standing for "parametric operators" and is related to the Mojo parameter system. 'pop.array' underpins `StaticTuple`, and there's a plan for a rewritten tuple in the next nightly that will support `AnyType`.

- **Correcting Item Assignment in Mojo**: A member pointed out and verified an issue where item assignment such as `a[i] = value` desugars incorrectly when `__getitem__` returns a `Reference`. It should instead utilize `__setitem__` or `__refitem__`, which when defined correctly, avoids the need for the other two methods.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modularml/mojo/issues/1871),">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/mzaks/mojo-unicode/blob/main/to_lower.mojo">mojo-unicode/to_lower.mojo at main · mzaks/mojo-unicode</a>: Contribute to mzaks/mojo-unicode development by creating an account on GitHub.</li><li><a href="https://mlir.llvm.org/docs/Dialects/IndexOps/">'index' Dialect - MLIR</a>: no description found</li><li><a href="https://mlir.llvm.org/docs/LangRef/">MLIR Language Reference - MLIR</a>: no description found</li><li><a href="https://github.com/modularml/mojo/pull/2294">[mojo-stdlib] Create `Array` type by lsh · Pull Request #2294 · modularml/mojo</a>: This PR creates an Array type that takes any CollectionElement rather than just AnyRegType. See also this thread on Discord.
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1228334768601694320)** (29 messages🔥): 

- **P2P Support Added to 4090**: A new update to **tinygrad** makes it possible to use P2P support on NVIDIA 4090 GPUs, achieving an impressive 14.7 GB/s AllReduce performance on tinybox green by modifying NVIDIA's driver. The technical details are discussed in a post available [here](https://x.com/__tinygrad__/status/1778676746378002712).
- **CUDA Takes On One Billion Row Challenge**: tspeterkim_89106 shared a blog post about their implementation of the [One Billion Row Challenge](https://1brc.dev/) in CUDA, which ran in 16.8 seconds on a V100, and openly invited the CUDA community to improve upon the [solution](https://github.com/tspeterkim/cuda-1brc/blob/main/fast.cu). Community feedback suggests possible improvements and a [faster run at 6 seconds on a 4090 GPU](https://tspeterkim.github.io/posts/cuda-1brc).
- **Council for Cool Materials**: A member suggested creating a dedicated channel for sharing valuable materials, and in response, #suggestions was repurposed to facilitate this exchange.
- **Exploring Parallels in CUDA**: In a technical exchange, zippika shared insights on optimizing CUDA performance, discussing potential use of warp level primitives, the effect of CPU speed on kernel execution, and strategies like preallocating buffers and utilizing `cudaMemcpyAsync` for efficiency. The possibility of hiding part splitting time by using asynchronous operations with CUDA kernels was highlighted as a potential optimization technique.
- **Handling the Spambot Siege**: A member reported an influx of spambots in the channel, alerting admins to the issue by tagging the moderator role.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tspeterkim.github.io/posts/cuda-1brc">The One Billion Row Challenge in CUDA: from 17m to 17s</a>: no description found</li><li><a href="https://x.com/__tinygrad__/status/1778676746378002712">Tweet from the tiny corp (@__tinygrad__)</a>: We added P2P support to 4090 by modifying NVIDIA&#39;s driver. Works with tinygrad and nccl (aka torch).  14.7 GB/s AllReduce on tinybox green!
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1228274705975476267)** (77 messages🔥🔥): 

- **PyTorch Version Compatibility and Performance Concerns**: Users discussed issues with building a specific kernel due to `error: namespace "at::cuda" has no member`, which was resolved using the torch-nightly build as opposed to PyTorch 2.2. Benchmarks showed that on the 4090, the custom kernel is *1.20x slower than torch.matmul*. [Benchmark code and results were provided](https://gist.github.com/mobicham/7fb59e825fed0831fccf44752cb21214).

- **Potential Optimization Routes Explored**: There was a suggestion of running independent QKV matmuls in separate CUDA streams, and the consideration that PyTorch's `torch.compile` might be optimizing performance behind the scenes, impacting comparative benchmarks.

- **Exploration of Fused Operation Kernels**: Interest was shown in adding more fused operation kernels, like those using lower bit-widths, referencing the  `cutlass int4xint4` kernel and the suggestion to address complexities in the matmul implementation.

- **Stable Fast and Full-Graph Compilation Discourse**: Users discussed the utility of leveraging `stable_fast` for compilation which appears faster than `torch.compile` with `fullgraph=True` in some cases, but might break when `fullgraph=True` is used for token-by-token decoding. Concerns were raised about [compilation issues of models](https://gist.github.com/mobicham/0e51c9f572721a76a5ac1e06fea533e9#file-stable_fast_llama_example-py-L14) and how `enable_cuda_graph=False` might be beneficial.

- **Promising Results in Low-Precision Computations**: It was reported that using `stable_fast` and other optimizations, performance approaches are within the range of int8 quant tensorrt optimization speeds, demonstrating potential for high-efficiency low-precision computations in models like stable diffusion.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://gist.github.com/mobicham/0e51c9f572721a76a5ac1e06fea533e9#file-stable_fast_llama_example-py-L14">stable_fast_llama_example.py</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://huggingface.co/papers/2404.07839">Paper page - RecurrentGemma: Moving Past Transformers for Efficient Open Language
  Models</a>: no description found</li><li><a href="https://github.com/chengzeyi/stable-fast/blob/main/src/sfast/jit/passes/__init__.py">stable-fast/src/sfast/jit/passes/__init__.py at main · chengzeyi/stable-fast</a>: Best inference performance optimization framework for HuggingFace Diffusers on NVIDIA GPUs. - chengzeyi/stable-fast</li><li><a href="https://gist.github.com/mobicham/7fb59e825fed0831fccf44752cb21214">hqq_hgemm_benchmark.py</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://github.com/spcl/QuaRot/blob/main/quarot/kernels/gemm.cu#L32">QuaRot/quarot/kernels/gemm.cu at main · spcl/QuaRot</a>: Code for QuaRot, an end-to-end 4-bit inference of large language models. - spcl/QuaRot
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1228267331457912842)** (15 messages🔥): 

- **P2P Enabled on NVIDIA GPUs**: **Tinygrad** successfully enabled P2P support on NVIDIA 4090 and 4070 TI Super GPUs by hacking open GPU kernel modules. The modifications resulted in a **58% speedup** for all reduce operations as shared on [Twitter](https://twitter.com/__tinygrad__/status/1778677126092509611) and code available on [GitHub](https://github.com/tinygrad/open-gpu-kernel-modules).

- **Peer-To-Peer Performance Gathered**: Users are consolidating P2P performance data and discussing potential issues with known anti-features. Contributions to the repository including benchmarks can be made [here](https://github.com/cuda-mode/p2p-perf).

- **Results Shared for P2P Performance**: A user submitted their **RTX 4070 TI Super dual build benchmarks** to the [p2p-perf repository](https://github.com/cuda-mode/p2p-perf/pull/1) for peer-to-peer performance measurement.

- **Efficient Ternary Weight Representation**: Current **BitNet implementations** using fp16 for ternary weights are deemed inefficient, and alternatives like custom 2-bit tensors or packing using int8 are being considered. A solution for bit-packing is provided with a reference to [this code](https://github.com/mobiusml/hqq/blob/master/hqq/core/bitpack.py#L43).

- **Distributed CI in PyTorch Examples**: PyTorch/examples repository now includes distributed CI, making it easier for contributors to test their code. For those interested in contributing to the distributed examples, the relevant pull request and the directory for contributions can be found [here](https://github.com/pytorch/examples/pull/1243/files) and [here](https://github.com/pytorch/examples/tree/main/distributed).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/mobiusml/hqq/blob/master/hqq/core/bitpack.py#L43">hqq/hqq/core/bitpack.py at master · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq</li><li><a href="https://github.com/cuda-mode/p2p-perf/pull/1">Add rtx-4070-ti-super-2x benchmarks by morgangiraud · Pull Request #1 · cuda-mode/p2p-perf</a>: no description found</li><li><a href="https://github.com/tinygrad/open-gpu-kernel-modules">GitHub - tinygrad/open-gpu-kernel-modules: NVIDIA Linux open GPU with P2P support</a>: NVIDIA Linux open GPU with P2P support. Contribute to tinygrad/open-gpu-kernel-modules development by creating an account on GitHub.</li><li><a href="https://github.com/cuda-mode/p2p-perf">GitHub - cuda-mode/p2p-perf: measuring peer-to-peer (p2p) transfer on different cuda devices</a>: measuring peer-to-peer (p2p) transfer on different cuda devices - cuda-mode/p2p-perf</li><li><a href="https://github.com/pytorch/examples/pull/1243/files">Update TP examples to align with tutorials by wanchaol · Pull Request #1243 · pytorch/examples</a>: as titled</li><li><a href="https://github.com/pytorch/examples/tree/main/distributed">examples/distributed at main · pytorch/examples</a>: A set of examples around pytorch in Vision, Text, Reinforcement Learning, etc. - pytorch/examples
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1228778979440197672)** (1 messages): 

- **CUDA Made Easy**: A member highlighted [PyTorch Dev podcast episode](https://open.spotify.com/episode/3dhD1irFc2gDNkLwUveKum?si=6DPutTGlSqaU0oflNbBW7Q) as a good starting point for those looking to learn about CUDA concepts. It covers the basics of GPUs, the CUDA programming model, asynchronous execution challenges, and CUDA kernel design principles with **PyTorch**.
- **CUDA Learning Resources and Tools**: For anyone interested in diving deeper into CUDA, the podcast episode recommends the **PyTorch documentation on CUDA semantics** (*https://pytorch.org/docs/stable/notes/cuda.html*) and the book "Programming Massively Parallel Processors." Debugging tools mentioned include setting the environment variable `CUDA_LAUNCH_BLOCKING=1` and using `cuda-memcheck` (*https://docs.nvidia.com/cuda/cuda-memcheck/index.html*).

**Link mentioned**: <a href="https://open.spotify.com/episode/3dhD1irFc2gDNkLwUveKum?si=6DPutTGlSqaU0oflNbBW7Q)">Just enough CUDA to be dangerous</a>: Listen to this episode from PyTorch Developer Podcast on Spotify. Ever wanted to learn about CUDA but not sure where to start? In this sixteen minute episode I try to jam in as much CUDA knowledge as ...

  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1228315933882454026)** (5 messages): 

- **Free Online Course on Parallel Computing**: An open version of "Programming Parallel Computers" course is available at [Aalto University](https://ppc-exercises.cs.aalto.fi/courses), providing material on OpenMP, vectorization, and GPU programming. The courses are self-paced with automated code testing and benchmarking, but do not offer study credits or human grading.

- **Leveraging NVIDIA GPU for Parallel Math Operations**: CUDA C/C++ functions can be utilized by torch/TensorFlow to compile machine code for NVIDIA GPUs, capitalizing on the GPUs' multiple cores for parallel processing, offering a distinct advantage over sequential CPU processing in Python.

- **PMPP Lectures Begin with Weed-Out of Initial Logistical Information**: The first session of PMPP lectures started at minute 31 to bypass introductory logistics, with a suggestion to review those details individually later.

- **Poll for Adjusting Session Timings**: Participants of the first PMPP session were reminded to vote in a poll, found in the invite channel, to help find better timing slots for future sessions.

- **Request for CPU vs GPU Architecture Video**: A member asked another for the link to a video that explains the differences between CPU and GPU architectures; however, the link has not been provided within the provided message history.

**Link mentioned**: <a href="https://ppc-exercises.cs.aalto.fi/courses">Courses</a>: no description found

  

---


**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1228743555821080656)** (9 messages🔥): 

- **3D Threads Configuration Curiosity**: A member pondered the necessity of arranging threads in 3D within CUDA, only to flatten them later on. The suggestion from the chat was to review section 3.2 for a better understanding, implying that further chapters would shed more light on this subject.

- **Comparison of CUDA Block Dimensions**: A discussion was sparked regarding the optimal choice between two different sets of configuration parameters for processing a vector with CUDA. A member advised that *both configurations can be fine*, and mentioned **thread coarsening** as an alternative approach by using a forloop inside a warp.
  

---


**CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1229133117634248846)** (6 messages): 

- **Volunteers Wanted for YouTube Recordings**: A call for volunteers has been made to assist with weekly recording tasks for the CUDA MODE Discord channel. Volunteers are to record, trim, and upload content to YouTube, and will receive a special role and direct credit, along with gratitude from the community.
- **Assistance Offered for Recording Task**: A member swiftly responded to volunteer for the weekly recording efforts, indicating the task was manageable.
- **Inquiry About Recent Talk Recording**: A member inquired if a particular weekend's talk had been recorded.
- **Confirmation of Talk Recording**: It was confirmed that indeed the weekend talk was recorded and is available.
- **Efforts to Enhance Talk Recording Quality**: Another member indicated that they are re-recording to improve the quality of the material and plan to have it uploaded by the following day.
  

---


**CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1229371863176843265)** (2 messages): 

- **Optimizing Tensor Layout in torchao**: A member raised the possibility of **torchao** optimizing tensor layout by storing weights in a way that's already aligned for swizzles in matrix multiplications. They mentioned that while element-wise operations are indifferent to data layout, optimizations for matrix multiplication could involve swizzling and padding to benefit from fast matrix multiplication libraries like cublas.
- **torch.compile Already Handling Some Optimizations**: In response to the tensor layout optimization suggestion, another member indicated that padding and layout optimization may already be addressed by **torch.compile**, providing links to the relevant sections in the PyTorch repository. [Padding optimization](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py#L420) and [layout optimization](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py#L261) are mentioned in the configuration files.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py#L420">pytorch/torch/_inductor/config.py at main · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch</li><li><a href="https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py#L261">pytorch/torch/_inductor/config.py at main · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1228270666684563568)** (3 messages): 

- **Google Steps into RNN and Transformers**: A member referenced Google's work on integrating RNN with transformers, implying a development in the machine learning space.
- **In-Depth Ring Attention Explainer Published**: An explainer post on **Ring Attention**, detailing its evolution from naive attention to block-wise parallel attention, has been [published by Kilian Haefeli, Simon Zirui Guo, and Bonnie Li](https://coconut-mode.com/posts/ring-attention/). This approach could potentially enable models to scale to a **nearly infinite context window** by employing multiple devices.
- **Raising the Bar on Context Lengths**: Large Language Models' context lengths have skyrocketed, from **GPT 3.5's 16k tokens** to **Gemini 1.5 Pro's 1 million tokens**, opening up new exciting use cases due to the ability to harness more information.
- **Appreciation for Clear Visualization**: A member praised the **animations** within the Ring Attention explainer, highlighting the value of good visual aids in understanding complex concepts.


**Link mentioned**: <a href="https://coconut-mode.com/posts/ring-attention/">Ring Attention Explained | Coconut Mode</a>: Near infinite context window for language models.

  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1228672477618507786)** (2 messages): 

- **Gratitude Expressed**: A member expressed appreciation for KJ's contributions to the community, acknowledging their positive impact on everyone involved.

- **The Game-Changer**: A member shared a [link to a tweet](https://x.com/yacinemtb/status/1778867085608644875) by @yacineMTB, emphatically stating that **"IT CHANGES EVERYTHING!!!!!!"** without further detail on the content or context.

**Link mentioned**: <a href="https://x.com/yacinemtb/status/1778867085608644875">Tweet from kache (dingboard.com) (@yacineMTB)</a>: IT CHANGES EVERYTHING!!!!!!

  

---


**CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1228370522341900298)** (12 messages🔥): 

- **CUDA MODE: HQQ Implementation Update**: A contributor updated their branch on [GitHub](https://github.com/pytorch-labs/gpt-fast/commit/551af74b04ee1e761736fbccfe98d37137d04176) to convert HQQ W_q weights to gpt-fast weights. This update allows them to achieve a perplexity (ppl) of 5.375 and a token generation speed of 200 tokens/s with the `--compile` option enabled.

- **HQQ Optimization Clarification**: When testing out HQQ, it is confirmed that one should always turn on `meta['optimize']` in the quant_config as it leads to performance improvements.

- **HQQ vs. HQQ+**: HQQ is a no-calibration, post-training quantization technique, while HQQ+ includes HQQ with trainable low-rank adapters and scale/zero parameters. HQQ+ can be used for bitwidths as low as 1 bit and has the potential to match or outperform full-precision models.

- **Challenges in Low-Bit Inference**: While promising, low-bit inference faces challenges such as finding efficient matmul kernels for low-bit operations and working around VRAM constraints imposed by low group-sizes needed for bits like 1 or 2.

- **Performance Leap with Torchao Int4 Kernel**: A complete rewrite of the generation pipeline for transformers models, which now supports the torchao int4 kernel, has significantly improved performance, achieving up to 152 tokens/sec compared to 59 tokens/sec with FP16.

**Link mentioned**: <a href="https://github.com/pytorch-labs/gpt-fast/commit/551af74b04ee1e761736fbccfe98d37137d04176">HQQ 4 bit llama 2 7b · pytorch-labs/gpt-fast@551af74</a>: export MODEL_REPO=meta-llama/Llama-2-7b-hf scripts/prepare.sh $MODEL_REPO python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int4-hqq --groupsize 64 python generate.py --...

  

---


**CUDA MODE ▷ #[triton-viz](https://discord.com/channels/1189498204333543425/1225499141241573447/1228481314676805753)** (3 messages): 

- **Confusion with Visualization Process**: A member expressed that the final step depicted in a **GIF** was confusing, indicating a possible need for clarification or additional guidance.
- **SVGs as a Medium for Code Representation**: The same member is currently in the process of writing outputs to **SVGs** for visualization purposes, showcasing a hands-on approach to their project.
- **Seeking Methods for Visualizing Control Structures**: There's an active effort to devise a method to effectively visualize programming constructs like **if statements** and **for loops**, suggesting a focus on enhancing code comprehensibility through visual aids.
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1228280026500436008)** (135 messages🔥🔥): 

- **CUDA Softmax Integration Challenges**: The softmax implementation faced NaN issues due to a causal mask causing exp(INF-INF); switching from -INF to -FLT_MAX solved the problem. Discussion also revolved around the readability and complexity of integrating arbitrary functions within reduce calls, leading to a preference for separate reductions for clarity.

- **Cooperative Groups Come with a Compiler Cost**: Members noted the inefficiency of using cooperative groups with more than 32 threads due to the compiler's lack of awareness about the exact threadblock size. This often results in suboptimal performance and unpredictable behaviors that can be difficult to mitigate even with detailed tooling like godbolt.

- **Online Softmax Discussed for Optimizing Performance**: The online softmax algorithm was dissected, with focus on the trade-off between sequential and parallel reduction approaches. Suggestions included the need for more comprehensive benchmarking with various "C" values to test performance improvements.

- **Investigating cuBLAS and CUTLASS for Optimal Kernel Use**: Conversations explored the use of cuBLAS, cuBLASLt, and CUTLASS for optimizing kernel operations. While cuBLAS’s simple integration boosted performance without adding complexity, CUTLASS would involve integrating large headers and substantial compile-time work, potentially complicating the codebase.

- **Stride to Achieve Peak Performance and Educational Value**: Members expressed the desire to have LLM.c be both highly efficient by using tools like CUTLASS and cuDNN, and educational with hand-written kernels ranging from naive to complex. The ultimate goal is to challenge PyTorch's performance using understandable and maintainable code, even potentially rewriting to ensure matrices have performance-friendly shapes.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/NVIDIA/cuda-samples/blob/master/Samples/3_CUDA_Features/cudaCompressibleMemory/compMalloc.cpp">cuda-samples/Samples/3_CUDA_Features/cudaCompressibleMemory/compMalloc.cpp at master · NVIDIA/cuda-samples</a>: Samples for CUDA Developers which demonstrates features in CUDA Toolkit - NVIDIA/cuda-samples</li><li><a href="https://github.com/karpathy/llm.c/pull/98">use cublaslt and optionally tf32, which fuses bias by karpathy · Pull Request #98 · karpathy/llm.c</a>: cc @ademeure , this is cuBLASLt version of matmul only, for now, with a bunch of smaller changes. E.g. I tried to take out global variables and settings, which by default I prefer to do as much as ...</li><li><a href="https://github.com/karpathy/llm.c/pull/79/files#diff-a00ef278da39f24a9d5cb4306c15626b921d437013fb5aa60ac2d8df6b5a5508R362)">Include the online softmax CPU code and a fully parallelized GPU kernal by lancerts · Pull Request #79 · karpathy/llm.c</a>: Include the online softmax CPU code (from the paper Online normalizer calculation for softmax). Its native port to GPU kernel kernel 5 (for education comparison). Include the fully parallel kernel ...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[recording-crew](https://discord.com/channels/1189498204333543425/1229286073104994344/1229286494187946014)** (12 messages🔥): 

- **Scaling up CUDA MODE Administration**: The channel is seeking more admins as current ones are overwhelmed with managing new developments in working groups.
- **Lecture Recording Process Shared**: A detailed 7-step lecture recording process for CUDA MODE was laid out, involving tools like OBS streamlabs and Da Vinci resolve, and coordination for event cover photos and session soundchecks. The final videos are uploaded to [CUDA MODE's YouTube](https://www.youtube.com/@CUDAMODE) channel and the material is also updated in [their GitHub lectures repository](https://github.com/cuda-mode/lectures).
- **Open Call for Process Improvement**: There is an open invitation for others to propose new methodologies or take ownership of parts of the recording process to facilitate smooth community growth.
- **Potential Shift to Live Streaming**: A suggestion was made to live stream lectures directly to YouTube to handle increased member scales and use YouTube's editor for clipping, with cautionary advice against using AirPods to prevent recording mishaps.
- **Collaborative Effort for Upcoming Recordings**: Two members are coordinating to handle upcoming ADA-CUDA lecture recordings, with an initial trial scheduled for a talk in the same week.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/live/24EWxyEMp1w?si=bfXVZPK4hJi_g49I)">HF Reading Group: Mobile ALOHA</a>: We were joined by an author of Mobile ALOHA for a discussion in the Hugging Face discord in #reading-group.Unfortunately no one else&#39;s voices was being recor...</li><li><a href="https://github.com/cuda-mode/lectures">GitHub - cuda-mode/lectures: Material for cuda-mode lectures</a>: Material for cuda-mode lectures. Contribute to cuda-mode/lectures development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1228278760445579354)** (308 messages🔥🔥): 

- **Connector API Misconception Resolved**: A member had trouble with **Cohere's connector API**; they received an error because the connector URL was incorrectly formatted. The group discussed that a valid connector URL must end in `/search` to function properly.
- **Clarifications on Cohere's Fine-Tuning Availability**: Another member sought clarification on **Cohere's base model availability** for fine-tuning, which led to a discussion indicating it could be accessed through [Cohere's dashboard](https://dashboard.cohere.com/fine-tuning/). Sandra confirmed that **fine-tuning on Amazon Bedrock** is possible, and the **command-r** model is also available [on AWS](https://aws.amazon.com/marketplace/seller-profile?id=87af0c85-6cf9-4ed8-bee0-b40ce65167e0).
- **Chatbot Interfaces and Grounded AI Advancements**: Sandra shared updates concerning Cohere's capabilities like **Coral**, a demo environment for chatbot interfaces, and hinted at upcoming releases for **toolkits** enabling connectors on AWS and private setups.
- **Cohere Models and Possible Hardware Limitations Explored**: There was an exchange regarding performance capabilities of AI models, specifically discussing **command-r-plus** model runtimes on various hardware included Nvidia's 4090 and **TPU** availability and performance.
- **Learning Resources and Supports for Newcomers**: Enquiries by new community members regarding learning materials were addressed with mentions of free courses like **LLM University** and redirection towards [Cohere's documentation](https://docs.cohere.com/docs/llmu) for further learning.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.ai/blog/grok-1.5v">Grok-1.5 Vision Preview</a>: no description found</li><li><a href="https://discord.gg/Y4msga6k?event=1208132674762575994">Join the Cohere Community Discord Server!</a>: Cohere community server. Come chat about Cohere API, LLMs, Generative AI, and everything in between. | 15435 members</li><li><a href="https://docs.cohere.com/reference/chat">Chat API Reference - Cohere Docs</a>: no description found</li><li><a href="https://sites.google.com/cohere.com/c4ai-community/community-programs/computer-vision?authuser=0)?">Community - Computer Vision</a>: Channel: #computer-vision Co-leads:  Benedict - @Harkhymadhe on Discord, @Arkhymadhe on Twitter  Logistics: Occurrences:  Second Tuesday of each month at 8am PT  Feel free to add papers/articles you w...</li><li><a href="https://docs.cohere.com/docs/connectors">Introduction to Connectors - Cohere Docs</a>: no description found</li><li><a href="https://dashboard.cohere.com/fine-tuning/?">Login | Cohere</a>: Cohere provides access to advanced Large Language Models and NLP tools through one easy-to-use API. Get started for free.</li><li><a href="https://en.wikipedia.org/wiki/Special:Search?search=glucose",">glucose&quot;, - Search results - Wikipedia</a>: no description found</li><li><a href="https://aws.amazon.com/marketplace/seller-profile?id=87af0c85-6cf9-4ed8-bee0-b40ce65167e0">AWS Marketplace: Cohere</a>: no description found</li><li><a href="https://coral.cohere.com/?s=t">Login | Cohere</a>: Cohere provides access to advanced Large Language Models and NLP tools through one easy-to-use API. Get started for free.</li><li><a href="https://dashboard.cohere.com/playground/chat">Login | Cohere</a>: Cohere provides access to advanced Large Language Models and NLP tools through one easy-to-use API. Get started for free.
</li>
</ul>

</div>
  

---


**Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1229333844424134676)** (2 messages): 

- **Command-R Core Module Celebrated**: A member celebrated the introduction of **Command-R** as one of the core modules, praising Cohere's efforts.
- **Join the Realm of Quant Finance**: A promotion for Quant Based Finance with an invitation link for beta testing was shared. Interested parties are directed to [Quant Based Finance](https://quantfino.com/join-beta).
- **Rubiks.AI Seeks Beta Testers**: The announcement of a newly launched advanced research assistant and search engine. Beta testers will receive 2 months free premium access to a suite of models including **Claude 3 Opus, GPT-4 Turbo, Mistral Large**, and others, accessible via [Rubiks.AI](https://rubiks.ai/).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://rubiks.ai/">Rubik's AI - AI research assistant & Search Engine</a>: no description found</li><li><a href="https://quantfino.com/join-beta">Quantfino - Home of Powerful AI Driven Finance</a>: Quantfino is the home of LLM powered and Langchain assisted Financial Analysis.
</li>
</ul>

</div>
  

---



**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1228267768537812992)** (172 messages🔥🔥): 

- **Running Multiple Models Locally**: A user asked about running multiple models simultaneously on a single machine, which was confirmed as possible by another member as long as there are sufficient resources. They mentioned successfully running 6 instances of A1111 models on one GPU.

- **Discussion on Model Optimization**: Conversation about LLM.c optimization, with members discussing its design not being inherently fast. A community member notably explained that the point was to showcase GPT-2 could be implemented in C without dependencies, rather than for speed.

- **Hugging Face Spaces Connectivity Issues**: Several users reported issues accessing Hugging Face Spaces, with the pages loading as blank or giving an invalid response error. Members suggested checking router and Wi-Fi settings, with some finding success by disabling security shields that were blocking the sites.

- **Chatbot Modeling Questions**: Users inquired about various aspects of creating and running chat-based models, including dataset formatting and optimization suggestions. One user detailed their desire to use langchain and neuralhermes-2-pro in their chatbot to maintain a conversation memory.

- **Job and Project Inquiries**: Discussions included a user seeking advice on hiring for a job involving model implementation and training, while another shared news of their growing freelance workload and their search for team members to handle increasing projects.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://us05web.zoom.us/j/87189451768?pwd=6PlOsbZ2AbaaIhwvaa6kjyvn0NelT2.1">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...</li><li><a href="https://www.missiononesmile.org/generation-generated.html">Mission One Smile</a>: no description found</li><li><a href="https://rubiks.ai/">Rubik's AI - AI research assistant & Search Engine</a>: no description found</li><li><a href="https://github.com/leejet/stable-diffusion.cpp">GitHub - leejet/stable-diffusion.cpp: Stable Diffusion in pure C/C++</a>: Stable Diffusion in pure C/C++. Contribute to leejet/stable-diffusion.cpp development by creating an account on GitHub.</li><li><a href="https://status.huggingface.co/">
Hugging Face status
</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1228457025319473182)** (6 messages): 

```html
<ul>
  <li><strong>Decoding CLIP's Concept:</strong> A blog by <a href="https://www.linkedin.com/in/matthewbrems/">Matthew Brems</a> aims to simplify the understanding of <strong>OpenAI's CLIP model</strong>, covering what it is, its workings, and its significance. This multimodal model from OpenAI offers a novel approach to computer vision by training on both images and text. <a href="https://www.kdnuggets.com/2021/03/beginners-guide-clip-model.html">Beginner's Guide to CLIP Model</a>.</li>
  <li><strong>Applying Zero-Shot Classification:</strong> After learning about Zero-Shot Classification with the <strong>bart-large-mnli model</strong>, a demo was created showcasing its application to classify 3D assets from <a href="https://thebasemesh.com/">thebasemesh.com</a>, a free resource of base meshes for creative projects. Watch the exploration on <a href="https://www.youtube.com/watch?v=jJFvOPyEzTY">YouTube</a>.</li>
  <li><strong>Cost-Effective AI with Podman:</strong> A demonstration video was shared showing how <strong>Podman</strong> is used to build containerized generative AI applications, which offers a cost-efficient alternative for local and cloud deployment. The technology promises control and choice regarding deployment options. Tutorial on <a href="https://youtu.be/3iEhFKIDXp0?si=nQJt-PBJUB960gpU">YouTube</a>.</li>
</ul>
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/safetensors/index">Safetensors</a>: no description found</li><li><a href="https://www.kdnuggets.com/2021/03/beginners-guide-clip-model.html">A Beginner&#039;s Guide to the CLIP Model - KDnuggets</a>: CLIP is a bridge between computer vision and natural language processing. I&#039;m here to break CLIP down for you in an accessible and fun read! In this post, I&#039;ll cover what CLIP is, how CLIP w...</li><li><a href="https://blog.roboflow.com/how-to-use-openai-clip/">How to Try CLIP: OpenAI&#x27;s Zero-Shot Image Classifier</a>: Earlier this week, OpenAI dropped a bomb on the computer vision world.</li><li><a href="https://youtu.be/3iEhFKIDXp0?si=nQJt-PBJUB960gpU">Podman Build- For cost-conscious AI</a>: A quick demo on how @podman builds containerized generative AI applications that can be deployed locally or to your preferred cloud provider. That choice is ...</li><li><a href="https://thebasemesh.com/">THE BASE MESH</a>: A public library of over 1000+ base meshes. All assets are real world scale and come unwrapped! 100% free. CC0 License.</li><li><a href="https://www.youtube.com/watch?v=jJFvOPyEzTY">Machine Learning : Zero Shot Classification demo using bart-large-mlni</a>: #machinelearning #zeroshotclassification #naturallanguageinference #transferlearning Zero-Shot Classificationhttps://huggingface.co/tasks/zero-shot-classific...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1228284633310040144)** (10 messages🔥): 

- **Blending AI with Document Retrieval**: An article on Medium introduces **LlamaIndex** integrated with a Colbert-based agent to enhance document retrieval with memory. The tutorial can be found [here](https://medium.com/ai-advances/enhancing-document-retrieval-with-memory-a-tutorial-for-llamaindex-with-colbert-based-agent-1c3c47461122).
  
- **PyTorch meets Blender**: GitHub repositories were shared for integrating PyTorch with Blender: [btorch](https://github.com/j20232/btorch) contains PyTorch utilities for Blender, and [pytorch-blender](https://github.com/cheind/pytorch-blender.git) offers seamless, real-time Blender integration into PyTorch data pipelines.

- **Boosting CNNs with Group Equivariance**: The [e2cnn library](https://github.com/QUVA-Lab/e2cnn) on GitHub enhances Convolutional Neural Networks by providing a PyTorch implementation of E(2)-Equivariant CNNs, which can be beneficial for tasks requiring rotational invariance.

- **A Dataset for Text-to-Music AI Innovators**: The **MagnaTagATune dataset**, available on [Papers with Code](https://paperswithcode.com/dataset/magnatagatune), provides a comprehensive set of music clips accompanied by a wealth of human-generated tags, offering a potential starting point for anyone interested in creating an open-source text-to-music service.

- **Highlighting AI's Capability to Understand Videos**: A YouTube video called "Large Language Models Are Zero Shot Reasoners" demonstrated an Edge browser feature for generating video highlights; [a user's impression](https://youtu.be/fTOIzQAHNtc) was shared showcasing the feature's effectiveness.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=GuhzATF13TOmwUMU">Neural networks</a>: Learn the basics of neural networks and backpropagation, one of the most important algorithms for the modern world.</li><li><a href="https://paperswithcode.com/dataset/magnatagatune">Papers with Code - MagnaTagATune Dataset</a>: MagnaTagATune dataset contains 25,863 music clips. Each clip is a 29-seconds-long excerpt belonging to one of the 5223 songs, 445 albums and 230 artists. The clips span a broad range of genres like Cl...</li><li><a href="https://g.co/gemini/share/e8962ab90c1c">‎Gemini - Cours Data Science, IA et GenAI</a>: Created with Gemini</li><li><a href="https://github.com/j20232/btorch">GitHub - j20232/btorch: btorch contains simple PyTorch utilities for Blender.</a>: btorch contains simple PyTorch utilities for Blender. - j20232/btorch</li><li><a href="https://github.com/cheind/pytorch-blender.git">GitHub - cheind/pytorch-blender: :sweat_drops: Seamless, distributed, real-time integration of Blender into PyTorch data pipelines</a>: :sweat_drops: Seamless, distributed, real-time integration of Blender into PyTorch data pipelines - cheind/pytorch-blender</li><li><a href="https://github.com/QUVA-Lab/e2cnn">GitHub - QUVA-Lab/e2cnn: E(2)-Equivariant CNNs Library for Pytorch</a>: E(2)-Equivariant CNNs Library for Pytorch. Contribute to QUVA-Lab/e2cnn development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1228563022000423002)** (17 messages🔥): 

- **Tips for Realistic AI-Generated Skin**: A member shared a tip for generating more realistic skin textures in AI work, suggesting to use "perfect skin" in the negative prompt and adding specific features like "(moles:0.8)" in the positive prompt to achieve a natural look with moles.
- **Enterprise AI Hub via Hugging Face**: A user announced their development of an **Enterprise AI Hub** that integrates various enterprise applications and is capable of intelligent prompt-based interactions, demonstrating the versatility of Hugging Face and OpenAI for enterprise solutions. The hub is built using Java, Spring Boot, and connected systems like pet stores, bookstores, and bug tracking systems. [*EnterpriseAIHub* on Spaces](https://huggingface.co/spaces/VishalMysore/EnterpriseAIHub)
- **Counterfactual Inception to Mitigate Hallucination**: A member introduced a paper called "What if...?: Counterfactual Inception to Mitigate Hallucination Effects in Large Multimodal Models," outlining a novel method to reduce hallucinations in AI response accuracy without additional instruction tuning. The project code is available on GitHub. [Paper](https://arxiv.org/abs/2403.13513) | [GitHub Repository](https://github.com/ivy-lvlm/counterfactual-inception)
- **RicercaMente: Mapping Data Science Evolution**: A project called RicercaMente, aimed at tracing the history of data science through significant scientific papers, was shared, inviting others to contribute or give support on GitHub. Participation purportedly requires no coding and only about 10 minutes of time. [GitHub Repository](https://github.com/EdoPedrocchi/RicercaMente)
- **Robotic Soccer Victories and Temperature Gap Fills**: A member discussed their involvement in two projects: one which contributed to winning the humanoid robot soccer world cup, RoboCup, with published advancements and source code, and another aimed at using partial convolutions to fill in missing temperature data in earth monitoring. [RoboCup Paper](https://arxiv.org/pdf/2302.02956.pdf) | [Temperature Gap-Fill Study](https://www.mdpi.com/1424-8220/24/5/1604)
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://kunalmishra.info/">Text Generation</a>: no description found</li><li><a href="https://huggingface.co/spaces/VishalMysore/EnterpriseAIHub">EnterpriseAIHub - a Hugging Face Space by VishalMysore</a>: no description found</li><li><a href="https://github.com/Merkoba/Meltdown">GitHub - Merkoba/Meltdown: An interface for llama.cpp and ChatGPT</a>: An interface for llama.cpp and ChatGPT. Contribute to Merkoba/Meltdown development by creating an account on GitHub.</li><li><a href="https://github.com/EdoPedrocchi/RicercaMente">GitHub - EdoPedrocchi/RicercaMente: Open source project that aims to trace the history of data science through scientific research published over the years</a>: Open source project that aims to trace the history of data science through scientific research published over the years - EdoPedrocchi/RicercaMente</li><li><a href="https://arxiv.org/abs/2403.13513">What if...?: Counterfactual Inception to Mitigate Hallucination Effects in Large Multimodal Models</a>: This paper presents a way of enhancing the reliability of Large Multimodal Models (LMMs) in addressing hallucination effects, where models generate incorrect or unrelated responses. Without additional...</li><li><a href="https://github.com/ivy-lvlm/counterfactual-inception">GitHub - IVY-LVLM/Counterfactual-Inception</a>: Contribute to IVY-LVLM/Counterfactual-Inception development by creating an account on GitHub.</li><li><a href="https://www.mdpi.com/1424-8220/24/5/1604">Deep Interpolation of Remote Sensing Land Surface Temperature Data with Partial Convolutions</a>: Land Surface Temperature (LST) is an important resource for a variety of tasks. The data are mostly free of charge and combine high spatial and temporal resolution with reliable data collection over a...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1228457751479451648)** (7 messages): 

- **Cohere's Commendable Open-Source Contribution**: *Cohere For AI* distinguishes itself with its dedication to open-source AI and open science, as highlighted in the upcoming LLM Reading Group session. This session will delve into **Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning**, scheduled for April 16, and interested individuals can RSVP at this [Eventbrite link](https://www.eventbrite.ca/e/llm-reading-group-march-5-19-april-2-16-30-may-14-28-tickets-851921368747?aff=oddtdtcreator).

- **Calendar Convenience Concerns**: Multiple participants have requested a Google Calendar link for the LLM Reading Group sessions, to which they were directed to find a calendar subscription option on Eventbrite after RSVPing.

- **Navigating CUDA Compatibility**: A member sought guidance on which installation to use for CUDA version 11.8, with the advice given to opt for the version labeled for **11.x**, indicating compatibility with all CUDA 11 versions including 11.7, 11.8, and 11.9.

- **Human Feedback Foundation's Pivotal Role**: The **Human Feedback Foundation** aims to integrate public input into AI models, emerging as a necessary component in key sectors such as healthcare and governance. They strive to be a neutral party in establishing a comprehensive database of human feedback for AI developers.

- **LLM Reading Group Spring Sessions in Full Swing**: The reminder for the LLM Reading Group highlights future sessions on various topics including **LLMs in negotiation**, **LLM-generated text detection**, and **reinforcement learning from human feedback**. The full list of sessions and papers for the spring can be found in the Season's Schedule, with the series running every two weeks till the **end of May**.

**Link mentioned**: <a href="https://www.eventbrite.ca/e/llm-reading-group-march-5-19-april-2-16-30-may-14-28-tickets-851921368747?aff=oddtdtcreator">LLM Reading Group (March 5, 19; April 2, 16, 30; May 14; 28)</a>: Come and meet some of the authors of some seminal papers in LLM/NLP research and hear them them talk about their work

  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1228408018811752478)** (7 messages): 

- **U-Net Model Plateaus during Training**: A member working on a **U-Net** model for cleaning scanned images with bleed-through effects struggles with the model ceasing to learn after 11 epochs. The individual plans to explore **Deep Image Prior** as an alternative, hoping for better results.

- **Grounding DINO Joins Transformers Library**: The new **Grounding DINO** model for zero-shot object detection is now included in the Transformers library. The official documentation and a demo notebook can be found at [Transformers Documentation](https://huggingface.co/docs/transformers/main/en/model_doc/grounding-dino) and [Inference with Grounding DINO Notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Grounding%20DINO/Inference_with_Grounding_DINO_for_zero_shot_object_detection.ipynb), respectively.

- **Invitation to Join the RicercaMente Project**: A member announces **RicercaMente**, a project that aims to chart the evolution of data science through important research papers. They encourage involvement and provide the GitHub link: [RicercaMente GitHub](https://github.com/EdoPedrocchi/RicercaMente).

- **Stable Diffusion Gradio Chatbot Released**: Sharing a new stable-diffusion **Gradio chatbot for image generation**, a member invites others to try the tool and support it with a star on the GitHub repository available at [Awesome Tiny SD GitHub](https://github.com/AstraBert/awesome-tiny-sd) and the live instance on [Hugging Face Spaces](https://huggingface.co/spaces/as-cle-bert/awesome-tiny-sd).

- **Inquire about Fine-tuning Vision Models for Specific Tasks**: A member seeks advice on fine-tuning a vision model for captioning images and identifying entities in low-resolution images, emphasizing the need for optimization for smaller image sizes.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/EdoPedrocchi/RicercaMente">GitHub - EdoPedrocchi/RicercaMente: Open source project that aims to trace the history of data science through scientific research published over the years</a>: Open source project that aims to trace the history of data science through scientific research published over the years - EdoPedrocchi/RicercaMente</li><li><a href="https://github.com/AstraBert/awesome-tiny-sd">GitHub - AstraBert/awesome-tiny-sd: Tiny stable diffusion chatbot based on segmind/small-sd to generate images upon text prompts.</a>: Tiny stable diffusion chatbot based on segmind/small-sd to generate images upon text prompts. - AstraBert/awesome-tiny-sd</li><li><a href="https://huggingface.co/spaces/as-cle-bert/awesome-tiny-sd">Awesome Tiny Sd - a Hugging Face Space by as-cle-bert</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/main/en/model_doc/grounding-dino">Grounding DINO</a>: no description found</li><li><a href="https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Grounding%20DINO/Inference_with_Grounding_DINO_for_zero_shot_object_detection.ipynb">Transformers-Tutorials/Grounding DINO/Inference_with_Grounding_DINO_for_zero_shot_object_detection.ipynb at master · NielsRogge/Transformers-Tutorials</a>: This repository contains demos I made with the Transformers library by HuggingFace. - NielsRogge/Transformers-Tutorials
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1228384523499212901)** (14 messages🔥): 

- **Exploring Lightweight Embedding Models**: Members discussed suitable embedding models for projects, with suggestions including using the **all-MiniLM-L6-v2** for its balance between semantic performance and lightweight architecture. For multilingual requirements, **paraphrase-multilingual-MiniLM-L12-v2** was recommended, especially in academic settings such as dissertation work.
  
- **Assembling the Pieces of Gargantuan Models**: A question was raised about how to handle large models like **mixtral-8x22B** split into multiple GGUF files. There was no direct answer provided in the thread regarding the mechanics of loading or merging these files.

- **Project RicercaMente Hits the Stage**: The community was invited to contribute to **RicercaMente**, a GitHub project that chronicles the history of data science through significant scientific papers. The call included a request for stars, contributions, and social sharing. [Check out the project on GitHub](https://github.com/EdoPedrocchi/RicercaMente).

- **Quest for a Custom RAG LLM Tutorial**: A member sought advice on finding a tutorial or YouTube video to create and fine-tune a **RAG LLM** for question answering, emphasizing tailoring to specific use cases, but no specific resources were provided.

- **Guidance Requested on Beginning with NLP**: Newcomers to NLP asked for roadmaps and learning resources, receiving suggestions ranging from Jurafsky's book, "*Speech and Language Processing*", to introductory materials such as [HuggingFace's learning resources](https://huggingface.co/learn) and [SpaCy tutorials](https://spacy.io/usage). A recent video on transformers by 3blue1brown was also mentioned as a helpful resource for beginners.

**Link mentioned**: <a href="https://github.com/EdoPedrocchi/RicercaMente">GitHub - EdoPedrocchi/RicercaMente: Open source project that aims to trace the history of data science through scientific research published over the years</a>: Open source project that aims to trace the history of data science through scientific research published over the years - EdoPedrocchi/RicercaMente

  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1228393964818665503)** (6 messages): 

- **Deep Dive into Multimodal Search Mechanics**: A user inquired about the details of multimodal embeddings in a web app demo, highlighting confusion on how the demo could understand both image and text elements, specifically recognizing a product name with a typo. They referenced a [Google demo](https://ai-demos.dev/) showcasing multimodal vertex embeddings to better understand the underlying technology.

- **Short Replies Can Still Be Insightful**: In response to a query about multimodal embeddings, another user simply replied with "clip retrieval," suggesting a potential mechanism or area to explore for the earlier question about incorporating textual information in image searches.

- **Introducing ControlNet++**: A user shared a link to a paper on ControlNet++, an advancement over the previous ControlNet for text-to-image diffusion models, which aims for better control through pixel-level cycle consistency. The shared [paper on arXiv](https://arxiv.org/abs/2404.07987) introduces how it improves alignment with conditional image controls.

- **Clarifying the Landscape of Language Models**: A new member expressed confusion about different types of language models, specifically differentiating between LLMs, embedding models, openAIEmbadding, and gpt_3.5_turbo models.

- **Stable Diffusion Cascade Model Query**: A user sought assistance with an error when using a long prompt with the stable diffusion cascade model; a response clarified that the message was a warning rather than an error related to token limitations. The issue was discussed further in a [GitHub issue](https://github.com/huggingface/diffusers/issues/7672).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ai-demos.dev/">AI Demos</a>: no description found</li><li><a href="https://arxiv.org/abs/2404.07987">ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback</a>: To enhance the controllability of text-to-image diffusion models, existing efforts like ControlNet incorporated image-based conditional controls. In this paper, we reveal that existing methods still f...</li><li><a href="https://github.com/huggingface/diffusers/issues/7672">error in using stable cascade with long prompt · Issue #7672 · huggingface/diffusers</a>: Hi, When I use stable cascade model with long prompt, I get below error. Token indices sequence length is longer than the specified maximum sequence length for this model (165 &gt; 77). Running this s...
</li>
</ul>

</div>
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1228357505889992835)** (147 messages🔥🔥): 

- **Scam Alert: LAION Not Associated with NFTs or Tokens**: Members are warned against a scam where a Twitter account, purportedly representing LAION, promotes NFTs. The [Twitter scam](https://twitter.com/OraProtocol/status/1778107440992756126) has also been discussed with users seeking clarity on the association between LAION and cryptocurrencies, and official statements from LAION members vehemently deny any connection.

- **Technical Discussion on Model Training**: There is an in-depth technical discussion on different approaches to training, including loss weights, signal noise, and the ideal number of timesteps. Special focus is given to the *min-snr-gamma* concept, with key insights shared regarding the implementation in diffusers and its practical implications on model performance.

- **New AI Audio Model Consideration**: The conversation turns towards AI models for sound generation, touching on open-sourcing of music AI, and a TTS inference and training library, [Parler TTS](https://github.com/huggingface/parler-tts), developed by Hugging Face.

- **Debate Over the Term 'Ethical Dataset'**: A philosophical debate surfaces around the term 'ethical datasets', criticizing its use as politicized language and discussing its implications within a [Bloomberg article](https://finance.yahoo.com/news/adobe-ethical-firefly-ai-trained-123004288.html) about Adobe's AI training methods.
  
- **AI Model Output and Diffusion Techniques**: Users explore how pixel art diffusion models work, including their resolution capabilities, while others discuss AI model outputs and the ethics of sharing potentially offensive material. There is mention of a successful application of magnitude-preserving learned layers after normalizing inputs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.ora.io/app/imo/olm">ORA</a>: no description found</li><li><a href="https://huggingface.co/spaces/declare-lab/tango">Tango - a Hugging Face Space by declare-lab</a>: no description found</li><li><a href="https://mirror.xyz/orablog.eth/X3DYXDHnjkpB-DOz88DZO5RdfZPxxRi5j53bxttNgsk>">World’s First Initial Model Offering (IMO) for OpenLM</a>: OpenLM is a performative language modeling (LM) repository, aimed to facilitate research on medium sized LMs. </li><li><a href="https://github.com/huggingface/diffusers/issues/5654>">Issues · huggingface/diffusers</a>: 🤗 Diffusers: State-of-the-art diffusion models for image and audio generation in PyTorch and FLAX. - Issues · huggingface/diffusers</li><li><a href="https://github.com/huggingface/parler-tts">GitHub - huggingface/parler-tts: Inference and training library for high-quality TTS models.</a>: Inference and training library for high-quality TTS models. - huggingface/parler-tts</li><li><a href="https://finance.yahoo.com/news/adobe-ethical-firefly-ai-trained-123004288.html">Adobe&#x2019;s &#x2018;Ethical&#x2019; Firefly AI Was Trained on Midjourney Images</a>: (Bloomberg) -- When Adobe Inc. released its Firefly image-generating software last year, the company said the artificial intelligence model was trained mainly on Adobe Stock, its database of hundreds ...</li><li><a href="https://youtu.be/e1UgzSTicuY?si=0vS7ImdlVKC2QJJd">Why I&#39;m Leaving My Company Immediately (Stability AI) w/ Emad Mostaque | EP #93</a>: In this episode, Peter and Emad discuss Emad&#39;s stepping down as CEO of StabilityAI, his next steps into decentralized AI, and why there is so much urgency to...
</li>
</ul>

</div>
  

---


**LAION ▷ #[announcements](https://discord.com/channels/823813159592001537/826154622644649985/1228398882598162553)** (1 messages): 

- **Beware of Fake Twitter NFT Claims**: An announcement clarified that a fake Twitter account falsely claimed **Laion** is releasing NFTs, emphasizing that **Laion does not sell anything**. Laion remains a part of the **open source community** and is committed to providing **free AI resources**.
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1228342334303768667)** (46 messages🔥): 

- **Guidance Optimization in Diffusion Models**: A new study outlined in an [arXiv paper](https://arxiv.org/abs/2404.07724) revealed that optimal performance in image-generating diffusion models was achieved by limiting guidance to a specific range of noise levels. It was noted that guidance is harmful in the early stages of the chain, largely unnecessary toward the end, and only beneficial in the middle, leading to significantly improved images when applying the limited guidance interval.

- **Custom Callbacks Unleash Potential in Diffusers**: The [Hugging Face diffusers library documentation](https://huggingface.co/docs/diffusers/en/using-diffusers/callback) elaborates on the use of `callback_on_step_end` for dynamic adjustments during the denoising loop, like changing prompt embeddings or guidance scale. This feature was highlighted as a useful application for dynamic classifier-free guidance, where CFG is disabled after specific inference steps to enhance performance with minimal computation costs.

- **Data Accumulation Prevents Model Collapse**: An [arXiv paper](https://arxiv.org/pdf/2404.01413.pdf) presented evidence countering the notion of inevitable model collapse when models are trained on synthetically generated outputs, arguing that accumulating data over time, as opposed to replacing it, mitigates the risk of collapse.

- **Stable Diffusion 3 Faces Censoring and Quality Challenges**: Users on [Reddit](https://www.reddit.com/r/StableDiffusion/comments/1c3ro5y/stability_employee_preview_of_stable_diffusion_3/) discussed the heavy censorship of Stable Diffusion 3 and its potential decline in quality due to strict prompt control, with hopes for substantial fine-tuning to enhance its abilities.

- **Training Strategies for Scaled-Down CLIP Models**: A recently released [arXiv paper](https://arxiv.org/abs/2404.08197) suggested that high-quality data and data augmentation could enable CLIP models to perform comparably well with reduced training datasets, demonstrating the importance of data quality and strategic training over sheer dataset size.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/RekaAILabs/status/1779894626083864873">Tweet from Reka (@RekaAILabs)</a>: Along with Core, we have published a technical report detailing the training, architecture, data, and evaluation for the Reka models.  https://publications.reka.ai/reka-core-tech-report.pdf</li><li><a href="https://arxiv.org/abs/2404.08197">Scaling (Down) CLIP: A Comprehensive Analysis of Data, Architecture, and Training Strategies</a>: This paper investigates the performance of the Contrastive Language-Image Pre-training (CLIP) when scaled down to limited computation budgets. We explore CLIP along three dimensions: data, architectur...</li><li><a href="https://arxiv.org/abs/2404.07724">Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models</a>: Guidance is a crucial technique for extracting the best performance out of image-generating diffusion models. Traditionally, a constant guidance weight has been applied throughout the sampling chain o...</li><li><a href="https://huggingface.co/docs/diffusers/en/using-diffusers/callback">Pipeline callbacks</a>: no description found</li><li><a href="https://arxiv.org/abs/2309.16620">Depthwise Hyperparameter Transfer in Residual Networks: Dynamics and Scaling Limit</a>: The cost of hyperparameter tuning in deep learning has been rising with model sizes, prompting practitioners to find new tuning methods using a proxy of smaller networks. One such proposal uses $μ$P p...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1c3ro5y/stability_employee_preview_of_stable_diffusion_3/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/huggingface/diffusers/issues/7657>">Issues · huggingface/diffusers</a>: 🤗 Diffusers: State-of-the-art diffusion models for image and audio generation in PyTorch and FLAX. - Issues · huggingface/diffusers
</li>
</ul>

</div>
  

---


**LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/1228701880272818267)** (1 messages): 

- **Diffusion Model Woes: From Noise to Pitch Black**: A member working on a diffusion model encountered an issue where the model, after having training plateau, now produces only noise during inference—initially random noise, later progressing to solid black images. They suggested that the problem might be due to error accumulation or mode collapse, having already attempted regularisation and adjusting the learning rate without success, sharing noise examples and their [GitHub repository](https://github.com/K3dA2/Diffusion-Model) for the model code.
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1228272760237719595)** (142 messages🔥🔥): 

- **RAG Issues with Legal Contracts**: A user is experiencing difficulties with document splitting during Retrieval-Augmented Generation (RAG) operations on legal contracts. Specifically, the content of a section is being incorrectly attached to the previous section, affecting retrieval accuracy.
- **Parallel Execution in LangChain**: Inquiries were made about running nodes in parallel within LangGraph. It was confirmed that using the `RunnableParallel` class in LangChain allows for the parallel execution of tasks.
- **Azure OpenAI with LangChain**: A new Azure OpenAI user is seeking insights on the advantages of using LangChain with Azure, specifically questioning whether there would be cost benefits and how LangChain might be useful beyond RAG when chatting with personal documents on Azure.
- **Recommendation Systems Inquiry**: A member is asking for resources to help understand and implement personalized recommendation systems using user behavior data stored in databases.
- **Article on LLM Protection**: A new article titled "Safeguarding AI: Strategies and Solutions for LLM Protection" has been shared, discussing the security challenges, prompt attacks, solutions, and tools for LLM security.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.gg/pornox">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.</li><li><a href="https://flashcardfy.lol/">Flashcardfy - AI Flashcard Generator with Personalized Feedback</a>: Learn faster and smarter with AI-generated flashcards that provide personalized feedback.</li><li><a href="https://devanshus-organization.gitbook.io/llm-security">Safeguarding AI: Strategies and Solutions for LLM Protection | LLM Security</a>: Explore the security challenges and solutions of LLMs in this comprehensive guide. We cover potential risks, control mechanisms, and the latest tools for safer LLM application</li><li><a href="https://docs.anaconda.com/free/miniconda/index.html">Miniconda &#8212; Anaconda documentation</a>: no description found</li><li><a href="https://js.langchain.com/docs/modules/agents/how_to/custom_agent#bind-tools-to-llm>))">Custom agent | 🦜️🔗 Langchain</a>: This notebook goes through how to create your own custom agent.</li><li><a href="https://js.langchain.com/docs/use_cases/tool_use/agents#create-agent>))">Agents | 🦜️🔗 Langchain</a>: Chains are great when we know the specific sequence of tool usage needed for any user input. But for certain use cases, how many times we use tools depends on the input.</li><li><a href="https://js.langchain.com/docs/use_cases/tool_use/quickstart#create-a-tool>))">Quickstart | 🦜️🔗 Langchain</a>: In this guide, we will go over the basic ways to create Chains and Agents that call Tools. Tools can be just about anything — APIs, functions, databases, etc. Tools allow us to extend the capabilities...</li><li><a href="https://github.com/langchain-ai/langchain/issues/6254>).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://python.langchain.com/docs/integrations/chat/anthropic#beta-tool-calling>)).">ChatAnthropic | 🦜️🔗 LangChain</a>: This notebook covers how to get started with Anthropic chat models.</li><li><a href="https://js.langchain.com/docs/modules/agents/how_to/custom_agent#bind-tools-to-llm>)).">Custom agent | 🦜️🔗 Langchain</a>: This notebook goes through how to create your own custom agent.</li><li><a href="https://github.com/langchain-ai/langchain/issues/17352>):">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://bladehack.tech/">Blade Hack</a>: Blade Hack is a worldwide event focused on prompt hacking, which involves exploring and interacting with large language models (LLMs) in creative ways. Blade will cater to beginners and individuals of...</li><li><a href="https://python.langchain.com/docs/expression_language/why#invoke>)">Advantages of LCEL | 🦜️🔗 LangChain</a>: We recommend reading the LCEL [Get</li><li><a href="https://github.com/langchain-ai/langchain/issues/10223>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/2435>)).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/13635>)).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://python.langchain.com/docs/integrations/providers/vectara/vectara_chat#conversationalretrievalchain-with_map_reduce>).">Chat Over Documents with Vectara | 🦜️🔗 LangChain</a>: setup}</li><li><a href="https://github.com/langchain-ai/langchain/issues/5328>)).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://python.langchain.com/docs/use_cases/question_answering/local_retrieval_qa#using-in-a-chain>)).">Using local models | 🦜️🔗 LangChain</a>: The popularity of projects like</li><li><a href="https://python.langchain.com/docs/integrations/document_loaders/docusaurus#filtering-sitemap-urls>)).">Docusaurus | 🦜️🔗 LangChain</a>: Docusaurus is a static-site generator which
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1228287178115649546)** (3 messages): 

- **Inappropriate Content Alert**: A message was posted with a suspicious link claiming to offer adult content, utilizing emojis and an @everyone tag; it appears to be spam or phishing. (No direct link inclusion for safety reasons).
- **Seeking Guidance on Langfuse Callbacks**: A member has inquired about using **langfuse callback handler for tracing** in **langserve** with an intent to design an API that logs questions, session ID, user ID, etc. They are requesting any available sources or examples to assist with their implementation.

**Link mentioned**: <a href="https://discord.gg/pornox">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


**LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1228274187991253002)** (4 messages): 

- **Spam Alert**: It appears a spam message was posted containing a link to an adult content Discord server. The message promises **TEEN/ONLYFANS PORN** and includes emojis and a Discord invite link.

- **Technical Troubles with Python Langchain**: A member encountered a `ModuleNotFoundError` when trying to run a template from the Langchain documentation using `langchain serve`. They suspect the `pyproject.toml` file is not correctly reading the local package, despite checking directory and file settings.

**Link mentioned**: <a href="https://discord.gg/pornox">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1228272795054506027)** (17 messages🔥): 

- **AI-powered Streamlit News App**: The new app [Meeting Reporter](https://meeting-reporter.streamlit.app/) combines Streamlit and Langgraph along with multiple agents and humans in the loop to create more dependable news stories. The open-source code is available on [GitHub](https://github.com/tevslin/meeting-reporter), and a detailed explanation of the project can be found in a [blog post](https://blog.tomevslin.com/2024/04/human-in-the-loop-artificial-intelligence.html).

- **A Guide to Building RAG Locally**: A simple tutorial on constructing a local RAG system using LangChain, Ollama, and PostgreSQL has been shared, which may benefit those looking to create this setup. The instructions are detailed in a blog post on [Substack](https://luisciber.substack.com/p/how-to-build-a-rag-system-locally).

- **Collection of LLM Projects on GitHub**: For those interested in exploring Large Language Models (LLMs), a new GitHub repository [LLM-Projects](https://github.com/luiscib3r/LLM-Projects) has been shared, containing several projects aimed at learning and experimentation.

- **Perplexica, an Open-source AI Search Engine**: Perplexica, a locally run clone of Perplexity AI, has been introduced, offering an alternative AI-powered search engine that can operate with 100% local resources, including the possibility to use Ollama in place of OpenAI. The updated and fixed version is accessible on [GitHub](https://github.com/ItzCrazyKns/Perplexica/).

- **Vertex AI Agent Builder Explored in YouTube Tutorials**: A series of YouTube tutorials showcasing the development and implementation of agents using Vertex AI Agent Builder has been shared, with guides on setting up route chains and integrating with platforms like Slack. The videos can be found at these links: [Vertex AI Agent Builder Overview](https://youtu.be/9WytPcE64GQ), [Implementing Router Chain](https://youtu.be/mBmfcBjb3RA), and [Integration with Slack](https://youtu.be/bY3o5RzgWVw).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.gg/pornox">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.</li><li><a href="https://youtu.be/mBmfcBjb3RA">Implementation of router chain using Vertex AI Agent builder</a>: In this recording, I show how to build a route chain using Vertex AI Agent builder</li><li><a href="https://play.google.com/store/apps/details?id=com.projecthit.gptai&referrer=ph-aai">GPT AI - Chat GPT-4 &amp; Vision - Apps on Google Play</a>: no description found</li><li><a href="https://github.com/ItzCrazyKns/Perplexica/">GitHub - ItzCrazyKns/Perplexica: Perplexica is an AI-powered search engine. It is an Open source alternative to Perplexity AI</a>: Perplexica is an AI-powered search engine. It is an Open source alternative to Perplexity AI - ItzCrazyKns/Perplexica</li><li><a href="https://youtu.be/9WytPcE64GQ">Vertex AI Agent Builder</a>: In this recording, i show how to use Vertex AI Agent builder</li><li><a href="https://youtu.be/bY3o5RzgWVw">Integration of Vertex AI Agent Builder with Slack</a>: This recording shows how to integrate Vertex AI Agent Builder with slack</li><li><a href="https://github.com/ItzCrazyKns/Perplexica">GitHub - ItzCrazyKns/Perplexica: Perplexica is an AI-powered search engine. It is an Open source alternative to Perplexity AI</a>: Perplexica is an AI-powered search engine. It is an Open source alternative to Perplexity AI - ItzCrazyKns/Perplexica</li><li><a href="https://luisciber.substack.com/p/how-to-build-a-rag-system-locally">How to build a RAG System Locally</a>: Building a RAG System Locally using Ollama and PostgreSQL with PgVector</li><li><a href="https://github.com/VinciGit00/Scrapegraph-ai">GitHub - VinciGit00/Scrapegraph-ai: Python scraper based on AI</a>: Python scraper based on AI. Contribute to VinciGit00/Scrapegraph-ai development by creating an account on GitHub.</li><li><a href="https://github.com/ossirytk/llama-cpp-chat-memory">GitHub - ossirytk/llama-cpp-chat-memory: Local character AI chatbot with chroma vector store memory and some scripts to process documents for Chroma</a>: Local character AI chatbot with chroma vector store memory and some scripts to process documents for Chroma - ossirytk/llama-cpp-chat-memory</li><li><a href="https://bladehack.tech/">Blade Hack</a>: Blade Hack is a worldwide event focused on prompt hacking, which involves exploring and interacting with large language models (LLMs) in creative ways. Blade will cater to beginners and individuals of...</li><li><a href="https://github.com/luiscib3r/LLM-Projects">GitHub - luiscib3r/LLM-Projects: Collection of projects focused on exploring and building applications with Large Language Models (LLMs).</a>: Collection of projects focused on exploring and building applications with Large Language Models (LLMs). - luiscib3r/LLM-Projects</li><li><a href="https://meeting-reporter.streamlit.app/">no title found</a>: no description found</li><li><a href="https://github.com/tevslin/meeting-reporter">GitHub - tevslin/meeting-reporter: Human-AI collaboration to produce a newstory about a meeting from minutes or transcript</a>: Human-AI collaboration to produce a newstory about a meeting from minutes or transcript - tevslin/meeting-reporter
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1228273306487230464)** (5 messages): 

- **Spam Alert**: A spam message promoting adult content with a Discord link was posted in the channel.
- **Chunking Know-how**: A video link about [document chunking using LangChain](https://youtu.be/tMwdl9hFPns) was shared, which offers insights into creating **better RAG applications** by properly splitting documents to preserve their content during questioning.
- **Prompt Hacking Hackathon Team Formation**: An announcement for forming a team was made for the upcoming [Blade Hack prompt hacking competition](https://bladehack.tech/). This event challenges participants to use **prompt engineering to manipulate AI models** to produce bypassing responses, with a prize of $9000 and swag.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.gg/pornox">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.</li><li><a href="https://youtu.be/tMwdl9hFPns">But, How is Chunking Done ? Splitting Basics Using LangChain</a>: To create Better RAG applications, you need to know how to split or chunk the documents so you preserve the content while asking questions. In this video, I ...</li><li><a href="https://bladehack.tech/">Blade Hack</a>: Blade Hack is a worldwide event focused on prompt hacking, which involves exploring and interacting with large language models (LLMs) in creative ways. Blade will cater to beginners and individuals of...
</li>
</ul>

</div>
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1228274134933438516)** (64 messages🔥🔥): 

- **Open GPU Kernel Modules on GitHub**: An open-source project for NVIDIA Linux GPU with P2P support has been made available on GitHub, which can be found at [tinygrad/open-gpu-kernel-modules](https://github.com/tinygrad/open-gpu-kernel-modules).
- **New Instruct MoE Model Launched by Fireworks AI**: Fireworks Mixtral 8x22b Instruct OH, a finetuned version of the 8x22b MoE model using the OpenHermes dataset, is introduced by Fireworks AI with a preview available at their [Playground](https://fireworks.ai/models/fireworks/mixtral-8x22b-instruct-preview). A bug with deepspeed zero 3 was mentioned and a solution involving an update from DeepSpeed's main branch was shared.
- **Discussion About DeepSpeed's Support for Multi GPU Training**: It's noted that *deepspeed zero 3* aids in training larger models rather than hastening the training process. A successful workaround for DeepSpeed zero 3 issues when used with MoE models was discussed, involving updates from the official DeepSpeed GitHub repository.
- **Link Shared to AI-Generated Music**: A member acknowledged the advancements in AI-generated music, sharing a link to a song created using AI at [udio.com](https://www.udio.com/songs/eY7xtug1dV6hbfCDhyHJua).
- **Techniques and Tools for Training and Tuning Models**: Technical conversations regarding model merging, use of tools like LISA and DeepSpeed, and their implications on training and model performance were discussed. A GitHub link for extracting Mixtral model experts, with some hardware requirements, was provided: [Mixtral-Model-Expert-Extractor](https://github.com/MeNicefellow/Mixtral-Model-Expert-Extractor/blob/main/main.py).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.udio.com/songs/eY7xtug1dV6hbfCDhyHJua">Udio | Dune the Broadway Musical, Showtunes, Soundtrack by BobbyB</a>: Make your music</li><li><a href="https://huggingface.co/fireworks-ai/mixtral-8x22b-instruct-oh">fireworks-ai/mixtral-8x22b-instruct-oh · Hugging Face</a>: no description found</li><li><a href="https://x.com/_lewtun/status/1778697910713979124?s=46">Tweet from Lewis Tunstall (@_lewtun)</a>: We ran into the exact same problem with Zephyr 141B and full training - the issue is tied to this PR in DeepSpeed: https://github.com/microsoft/DeepSpeed/pull/5008  If you&#39;re using the HF trainer,...</li><li><a href="https://huggingface.co/blog/damjan-k/rslora">Rank-Stabilized LoRA: Unlocking the Potential of LoRA Fine-Tuning</a>: no description found</li><li><a href="https://github.com/tinygrad/open-gpu-kernel-modules">GitHub - tinygrad/open-gpu-kernel-modules: NVIDIA Linux open GPU with P2P support</a>: NVIDIA Linux open GPU with P2P support. Contribute to tinygrad/open-gpu-kernel-modules development by creating an account on GitHub.</li><li><a href="https://github.com/MeNicefellow/Mixtral-Model-Expert-Extractor/blob/main/main.py">Mixtral-Model-Expert-Extractor/main.py at main · MeNicefellow/Mixtral-Model-Expert-Extractor</a>: Contribute to MeNicefellow/Mixtral-Model-Expert-Extractor development by creating an account on GitHub.</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1477/files?short_path=3520786#diff-35207863e6e0da8dfa2d1311bf863b60c52a067c5e65253c24543edda5da00d0">Guide For Multi-Node Distributed Finetuning by shahdivax · Pull Request #1477 · OpenAccess-AI-Collective/axolotl</a>: Title: Distributed Finetuning for Multi-Node Setup Guide Description: This PR introduces a comprehensive guide for setting up a distributed finetuning environment using Axolotl and Accelerate. The ...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1192">Mixtral fixes 20240124 by winglian · Pull Request #1192 · OpenAccess-AI-Collective/axolotl</a>: Description various fixes for mixtral with zero3, requires deepspeed 0.13.1</li><li><a href="https://smicro.eu/nvidia-gpu-baseboard-8-h200-liquid-cool-935-24287-0040-000-2">NVIDIA GPU Baseboard 8 H200 Liquid Cool - 935-24287-0040-000</a>: Graphics Engine: H200
 BUS: NVIDIA NVLink
 Memory size: 1128 GB
 Memory type: HBM3e
 Theoretical performance: 536 TFLOP</li><li><a href="https://www.ebay.com/sch/i.html?_from=R40&_nkw=T480&_sacat=0&LH_BIN=1&rt=nc&_udhi=250&mkcid=1&mkrid=711-53200-19255-0&siteid=0&campid=5338557920&customid=&toolid=20012&mkevt=1">T480 for sale | eBay</a>: no description found
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1228314262913548330)** (19 messages🔥): 

- **Exploring Weight Unfreezing During Training**: A member pondered **dynamic weight unfreezing** as a strategy for GPU-limited users, considering unfreezing random subsets of weights at each step to optimize the training process.
- **Customizing Unfreezing Functionality**: The feasibility of *custom implementation* was discussed for dynamic weight unfreezing since the current system only supports freezing certain weights from the start.
- **Dynamic Compute Allocation Model Shared**: An unofficial GitHub implementation related to **Mixture-of-Depths**, which deals with dynamic allocation of compute in transformers, was shared: [Mixture-of-depths on GitHub](https://github.com/astramind-ai/Mixture-of-depths).
- **Peer-to-Peer Memory Access Success on RTX 4090s**: A member successfully enabled P2P memory access with **tinygrad** on **RTX 4090 GPUs** and sought advice on bypassing Axolotl's P2P disabling check, leading to a discussion on checking the accelerate code.
- **RTX 4090s P2P Compatibility and Community Interest**: Further clarification stated there's no explicit code that disables P2P, and the community expressed their interest regarding the successful utilization of P2P with the RTX 4090 GPUs.

**Link mentioned**: <a href="https://github.com/astramind-ai/Mixture-of-depths">GitHub - astramind-ai/Mixture-of-depths: Unofficial implementation for the paper &quot;Mixture-of-Depths: Dynamically allocating compute in transformer-based language models&quot;</a>: Unofficial implementation for the paper &quot;Mixture-of-Depths: Dynamically allocating compute in transformer-based language models&quot; - astramind-ai/Mixture-of-depths

  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1228552074715987988)** (39 messages🔥): 

- **Stuck on Repetition**: A member is experiencing issues with a model producing repetitive outputs, comparable to a dataset shared at [Izzy's Blog](https://www.izzy.co/blogs/robo-boys). Despite attempts with varying datasets and finetuning methods, the model outputs are limited to Discord usernames or blank text.

- **Mergekit for Model Surgery**: Members discussed using **mergekit** to drop layers from large language models and shared a configuration example from [WestLake-10.7B-v2 on Hugging Face](https://huggingface.co/froggeric/WestLake-10.7B-v2/blob/main/mergekit_config.yml).

- **Troubleshooting Axolotl Finetuning**: A user following a **Mixtral-8x22** finetuning reported an 'IndexError' during an evaluation step. Others suggested debugging using Axolotl's preprocess command or incrementally adjusting the YAML configuration based on examples from the [Axolotl GitHub repository](https://github.com/OpenAccess-AI-Collective/axolotl/tree/main/examples).

- **Prompt Formatting Challenges**: There was a discussion about the difficulties in getting **Axolotl** to use specific prompt formats during finetuning. A user ultimately found success in adapting their configuration, including using the `chat_template` setting.

- **Dataset Mixing and Training Loss Concerns**: Members exchanged thoughts on mixing completion and instruction data during training, with one suggesting it might be more efficient to train on them separately. They also discussed appropriate loss numbers for plain completion models and the potential impact of batch sizes.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/LR-AI-Labs/vbd-llama2-7B-50b-chat">LR-AI-Labs/vbd-llama2-7B-50b-chat · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/froggeric/WestLake-10.7B-v2/blob/main/mergekit_config.yml">mergekit_config.yml · froggeric/WestLake-10.7B-v2 at main</a>: no description found</li><li><a href="https://huggingface.co/HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1/discussions/4">HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1 · Prompt format</a>: no description found</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/tree/main/examples">axolotl/examples at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://www.irina-lab.ai/blog/continued-pretraining-blog">CERC-AAI Lab - Continued Pretraining Blog</a>: Continual Learning of Foundation Models:CL-FoMo Suite of 9.6B and 410M LLMs  </li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/config.qmd#L103-L112">axolotl/docs/config.qmd at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/)** (1 messages): 

b.nodnarb: Thanks for the post, <@915779530122207333> !
  

---


**OpenAccess AI Collective (axolotl) ▷ #[docs](https://discord.com/channels/1104757954588196865/1167137552470392842/1228441757285355672)** (9 messages🔥): 

- **Mistral V2 Instruct Shows Promise**: A member reported having **good results** with Mistral v2 instruct after just one epoch, outperforming other approaches like LorA and Qlora.
- **Effective on Diverse Tasks**: They highlighted success in **training on NER, Classification, and Summarization** specifically for metadata extraction tasks.
- **Surpasses Prior Methods**: The performance was notably better compared to experiments with another model named **qwen**.
- **Automation in Metadata Generation**: It was clarified that the large language model (**llm**) was responsible for **generating**, not just extracting, metadata from documents.
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1228317412193996822)** (5 messages): 

- **Docker Meets DeepSpeed for Distributed Training**: Multi-node finetuning with DeepSpeed requires a customized Docker image with all dependencies and a detailed `ds_config.json` file to configure settings. Steps cover building the Docker image, setting SSH access across nodes, and running containers with appropriate environment variables for multi-node communication.
- **SSH Keys Unlock Multi-Node Communication**: For successful multi-node training, it's essential to configure SSH keys that allow passwordless access between nodes. The DeepSpeed setup relies on seamless node communication through SSH to perform distributed computation effectively.
- **Running Training Containers Across Nodes**: Users must launch Docker containers on each node with carefully set environment variables that define the master node's address, port, world size, and each node's rank. The correct setup will enable the deployment of the finetuning process across different machines using DeepSpeed within Docker environments.
- **Training Initialization with DeepSpeed**: The training script should integrate DeepSpeed initialization and accept specific arguments (`--deepspeed` and `ds_config.json`) to harness the full potential of distributed training features. Ensuring these parameters are correctly used is crucial for the finetuning operation to function in a multi-node environment.

**Link mentioned**: <a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=d905c33d-1397-4ef3-82ad-a16aadf1eb1f)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.

  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1228427843663302746)** (25 messages🔥): 

- **Understanding DeepSpeed Scheduler Configuration**: It was clarified that when using **DeepSpeed** with 🤗 Accelerate, your custom learning rate scheduler is retained if you properly prepare it using Accelerate's methods, as DeepSpeed does not forcefully replace it unless otherwise specified in the configuration file.
  
- **Optimizing Micro-Batch Sizes**: A discussion on the ideal ratio of `micro_batch_size` to `gradient_accumulation_steps` highlighted the balance required to utilize GPU memory effectively, recommending to maximize the `micro_batch_size` and adjusting `gradient_accumulation_steps` accordingly for the desired effective batch size.

- **Multi-Node Training with Axolotl**: Configuring for multi-node training in **Axolotl** involves using a configuration file with specified networking details, such as `main_process_ip` and `main_process_port`, and the use of the `accelerate` CLI to launch training across machines.

- **Opening Ports for Distributed Training**: To ensure the `main_process_port` is open in TCP and reachable by other machines in a distributed training setup, instructions included identifying the main machine, using system commands like `iptables` or Windows Defender Firewall settings to open the port, and verifying network accessibility.

- **Ease of Hugging Face Model Hub Repository Creation**: The `push_to_hub` method was explained to automate the creation of a new repository on the Hugging Face Model Hub if the specified model ID does not already correspond to an existing repository, simplifying the process for users.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=e8e1f994-4d79-4e03-b88a-ebc3dc1d0933)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=98d8d920-0524-4543-b179-9bc5f7810e7f)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=43a2ad80-c4db-40e0-8bbf-e85b2a2dc2a1)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://github.com/huggingface/transformers/tree/main/src/transformers/utils/hub.py#L657L690))">transformers/src/transformers/utils/hub.py at main · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=23860f45-87ae-4250-ade9-727c7de241f7)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=c53633a8-bb1b-475d-a241-0931651d36c8)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.
</li>
</ul>

</div>
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1228269684944666685)** (79 messages🔥🔥): 

- **Malware Warnings and Command Line Confusion**: Concern was raised about potential malware warnings from Avast and confusion caused by the command line `ngrok/ngrok/ngrok`. Mentioned it might be helpful to clarify this in the documentation to reassure users.
- **Open Interpreter PR Proposed for Build Error**: A new pull request was opened to address a build process error, specifically to bump the version of tiktoken. The link to the pull request is [here](https://github.com/OpenInterpreter/open-interpreter/pull/1204).
- **Using Assistants API for Data Persistence**: Discussions around using Assistants API for data persistence in Open Interpreter touched on successfully creating a Python assistant module for improved session management. A request was made for specific advice on implementing this for node operations.
- **OS Mode in Open Interpreter on Ubuntu Troubleshooting**: Users shared issues and resolutions related to setting up **OS Mode** for Open Interpreter on Ubuntu. This included suggestions on enabling screen monitoring, accessibility features, and downgrading to Python 3.10 for compatibility.
- **Airchat as a Platform for Voice Communication Among OI Users**: Several members shared their [Airchat](https://www.joinairchat.com/) handles (e.g., `mike.bird`) and discussed the possibility of acquiring invites to join in live voice conversations.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/MaG00_GpHXw?si=I3MC3jHZ6lfLBhXL">Polaris robotic phone</a>: http://www.plasticpals.com - Flower Robotics was commissioned by KDDI to produce this robot that works with a mobile phone to improve your life through its &quot;...</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/pull/1204">Bump version of tiktoken by minamorl · Pull Request #1204 · OpenInterpreter/open-interpreter</a>: Describe the changes you have made: Bumped version of tiktoken since build process is broken for some reason. This PR fixes broken process. Reference any relevant issues (e.g. &quot;Fixes #000&quot;):...
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1228381814129688638)** (72 messages🔥🔥): 

- **Expect Delays in Startup Shipments**: Pre-orders for a new product are underway, but a member cautions patience, pointing out that start-ups often take *longer than their initial estimates*, especially since the product is still in the *pre-production phase*, as evidenced by GitHub updates.
- **Connecting Issues Over Hotspot**: Users experience issues when trying to connect the O1 to a hotspot via an iPhone, with the device sometimes not recognizing the network or crashing during the connection process. The problem persists even when using example codes, prompting speculation about whether specific libraries could be causing the crash.
- **Environment Variable Conundrum on Windows**: Windows users are struggling to set up the `OPENAI_API_KEY` as an environment variable, with suggestions to try different terminals, such as Command Prompt over PowerShell, and ensure a new session is started after setting it. There's a shared link to help [test the key](https://chat.openai.com/share/c625f9c8-102e-4669-9c87-5d874b57a241), but persistent issues lead to further discussion on alternatives.
- **Frustrations with WebSocket on Windows**: Despite updates meant to resolve the issue, some users continue to have trouble with WebSocket failing to open on Windows, a problem not encountered on Macs. Suggestions advise checking if the port is already in use and sharing screenshots of the error for troubleshooting.
- **Developments and DIY Efforts for O1**: Community members are modifying and improving their O1 designs, from the addition of a better battery to hand straps for easier use. One member offers free 3D printing services for cases, and there's mention of assistance from a custom GPT trained on Open Interpreter docs available for ChatGPT Plus users.

**Link mentioned**: <a href="https://github.com/rbrisita/01/tree/linux">GitHub - rbrisita/01 at linux</a>: The open-source language model computer. Contribute to rbrisita/01 development by creating an account on GitHub.

  

---


**OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/)** (1 messages): 

aime_bln: https://api.aime.info
  

---



**LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1228370900940750989)** (1 messages): 

- **LlamaIndex Update Affects PandasQueryEngine**: The new release of LlamaIndex, version 0.10.29, will transplant **PandasQueryEngine** to the `llama-index-experimental` package. Users must adjust their imports accordingly and an error message will guide those using the old module.
  

---


**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1228371970798915716)** (8 messages🔥): 

- **New AI Application Creation Toolkit Launched**: The **create-tsi** toolkit, developed in collaboration with @tsystemscom, @MarcusSchiesser and inspired by the @llama_index create-llama toolkit, allows users to generate a GDPR-compliant AI application via command line interface. The announcement included links to a [tweet](https://twitter.com/llama_index/status/1778812761893650551) and associated imagery.

- **Chain of Abstraction Technique Revealed**: LLMs like (e.g. @OpenAI, @MistralAI, @AnthropicAI) have been improved with a technique called Chain of Abstraction to assist in multi-step query planning; the full details of this innovation can be found in the provided [tweet](https://twitter.com/llama_index/status/1778845258119524640).

- **Full-Stack RAG with AWS Bedrock Guide**: A reference guide is available explaining how to build a full-stack RAG application using AWS Bedrock, @llama_index, and a @streamlit interface; more information is given in their [tweet](https://twitter.com/llama_index/status/1779185836690653435).

- **Data Management Challenges in LLM Apps**: Efficient data management for live data in LLM applications is a key feature of both @llama_index open-source and LlamaCloud, as highlighted in their [announcement](https://twitter.com/llama_index/status/1779235219469742112).

- **Knowledge Graphs for Accelerating Biomaterials Discovery**: A paper by @ProfBuehlerMIT demonstrates constructing an extensive knowledge graph from over 1000 scientific papers on biological materials to expedite biomaterials discovery; the source can be explored in their [tweet](https://twitter.com/llama_index/status/1779542320133947622).
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1228294385087680564)** (113 messages🔥🔥): 

- **Deciphering Vector Database Preferences**: Members discussed vector database preferences for strong similarity searches, comparing different options including Qdrant, pg vector, and Cassandra among others. Links to documentation were shared to evaluate feature support such as hybrid search and metadata filtering ([LlamaIndex vector stores guide](https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/)).

- **Hybrid Search Mechanics Explored**: The conversation expanded to include debate around the implementation of multimodal embeddings and how to retrieve and rank multimodal data—particularly concerning image and text data. Articles on fusion algorithms were referenced, and tools like Weaviate were mentioned for their hybrid search capabilities.

- **Technical Troubleshooting in LlamaIndex**: Users encountered various technical challenges, such as handling `MockLLM` errors within Vectara Managed Index and `ValidationError` when defining a `MarkdownElementNodeParser`. Community support provided guidance and suggestions, highlighting the importance of the most recent version updates.

- **Navigating Documentation and Contribution**: A user identified an error in the documentation regarding the `GithubRepositoryReader`, for which community members encouraged making a pull request on GitHub to correct the documentation. The importance of active community contribution to the LlamaIndex project was emphasized.

- **Engaging with LlamaIndex's Business Side**: A discussion about signing a Business Associate Agreement (BAA) with a user's client led to sharing contact information for LlamaIndex's team and assurance that the request would be passed on for a speedy response. The chatbot emphasized the significance of community support and accessibility of the LlamaIndex team.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ai-demos.dev/">AI Demos</a>: no description found</li><li><a href="https://www.llamaindex.ai/contact">Talk to us — LlamaIndex, Data Framework for LLM Applications</a>: If you have any questions about LlamaIndex please contact us and we will schedule a call as soon as possible.</li><li><a href="https://llamahub.ai/l/llama-packs/llama-index-packs-fuzzy-citation?from=">no title found</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/data_connectors/GithubRepositoryReaderDemo/">Github Repo Reader - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/">Vector Stores - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/settings/?h=settings">Settings - LlamaIndex</a>: no description found</li><li><a href="https://weaviate.io/blog/hybrid-search-fusion-algorithms">Unlocking the Power of Hybrid Search - A Deep Dive into Weaviate&#x27;s Fusion Algorithms | Weaviate - Vector Database</a>: How hybrid search works, and under the hood of Weaviate&#x27;s fusion algorithms.</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/qdrant_hybrid/">Qdrant Hybrid Search - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/07cb4c07409b0e0584e2dee70db0d3169f877b8e/llama-index-integrations/indices/llama-index-indices-managed-vectara/llama_index/indices/managed/vectara/retriever.py#L233">llama_index/llama-index-integrations/indices/llama-index-indices-managed-vectara/llama_index/indices/managed/vectara/retriever.py at 07cb4c07409b0e0584e2dee70db0d3169f877b8e · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_memory/">Query Pipeline Chat Engine - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_parse/blob/main/">GitHub - run-llama/llama_parse: Parse files for optimal RAG</a>: Parse files for optimal RAG. Contribute to run-llama/llama_parse development by creating an account on GitHub.</li><li><a href="https://github.com/run-llama/llama_parse/blob/main/examples/demo_advanced.ipynb">llama_parse/examples/demo_advanced.ipynb at main · run-llama/llama_parse</a>: Parse files for optimal RAG. Contribute to run-llama/llama_parse development by creating an account on GitHub.</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_sql/">Query Pipeline for Advanced Text-to-SQL - LlamaIndex</a>: no description found</li><li><a href="https://coursera.org/projects/langchain-chat-with-your-data-project">LangChain Chat with Your Data</a>: Complete this Guided Project in under 2 hours. LangChain: Chat With Your Data delves into two main topics: (1) Retrieval Augmented Generation (RAG), a ...</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_memory/?h=query+pipeline">Query Pipeline Chat Engine - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/agent_runner/query_pipeline_agent/?h=query+pipeline">Building an Agent around a Query Pipeline - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1228354294034595940)** (5 messages): 

- **Enhanced Document Retrieval via LlamaIndex**: A new [tutorial for enhancing document retrieval](https://medium.com/ai-advances/enhancing-document-retrieval-with-memory-a-tutorial-for-llamaindex-with-colbert-based-agent-1c3c47461122) has been shared, showcasing how to incorporate memory into Colbert-based agents using **LlamaIndex**.
  
- **Boost RAG Systems with Small Knowledge Graphs**: A discussion and article were shared on the benefits of using small knowledge graphs to improve the accuracy and explanation capabilities of RAG systems. [WhyHow.AI](https://medium.com/enterprise-rag/the-role-of-small-vs-big-knowledge-graphs-995d9449c208) is creating tools to facilitate building these graphs for enhanced **retrieval-augmented generation (RAG)** performance.

- **Integration of LlamaIndex for Advanced Reasoning**: A member highlighted a publication on the integration of **LlamaIndex** with a chain of abstraction strategy to unlock more efficient reasoning, with a link to the related [article](https://ai.gopubby.com/unlocking-efficient-reasoning-integrating-llamaindex-with-chain-of-abstraction-1b1844ba66e6).

- **Appreciation for Efficient Reasoning Insights**: Feedback was given showing appreciation for the recently shared articles focusing on the advancement of reasoning in AI systems.
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1228306974064640132)** (31 messages🔥): 

- **RTX 4090 Driver Patch Hits Hacker News**: One member mentioned that the **4090 driver patch** made it to the front page of Hacker News, suggesting its significant impact on the tech community.
- **Cost Efficiency of Stacking 4090s**: Discussion around stacking **RTX 4090s** implies a significant cost reduction for running large language models (LLMs), with one member providing a detailed cost comparison to *TinyBox*, showing a **36.04%** price decrease from the Team Red Tinybox and a **61.624%** drop from the Team Green Tinybox.
- **Tinygrad's quest for better documentation**: Queries about improved **tinygrad documentation** were raised, with members confirming its development. Suggestions were requested on how to handle the `fetch_mnist(tensors=False)` instances for convenient and coherent updating in **tinygrad examples**.
- **Developer Experience is a Priority for tinygrad**: **George Hotz** highlighted the progress on improving tinygrad's documentation and developer experience, noting the need for better error comprehension in the tool.
- **Line Limit Increase and Flaky MNIST Test**: There was an agreement to increase the code line limit to accommodate new backends, with **George Hotz** proposing to bump it up to *7500* lines. Additionally, issues about a *flaky mnist test* and nondeterminism in the scheduler were brought up, indicating the project's focus on robustness and reliability.

**Link mentioned**: <a href="https://github.com/tinygrad/tinygrad/actions/runs/8694852621/job/23844626455">hotfix: bump line count to 7500 for NV backend · tinygrad/tinygrad@e14a9bc</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - hotfix: bump line count to 7500 for NV backend · tinygrad/tinygrad@e14a9bc

  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1228260143167438858)** (72 messages🔥🔥): 

- **Fusion Process in Kernel Generation**: An in-depth discussion took place around the conditions under which a computational graph is split into multiple kernels during the fusion process. It was clarified that each **ScheduleItem** is a kernel, and reasons for a graph's split, such as broadcasting needs leading to separate kernels, can be found in the `create_schedule_with_vars` function in the source code.

- **Running Models with Tinygrad**: There was a clarification regarding **Tinygrad's capability to run models** like Stable Diffusion and Llama. It was explained that while Tinygrad can execute these models, it's not necessarily used for training them – showcasing **Tinygrad's flexibility** in interfacing with models trained in different frameworks.

- **Tinygrad Installation and Use Cases on Colab**: Members shared experiences and advice on installing and using Tinygrad in Google Colab, including the use of `pip install git+https://github.com/tinygrad/tinygrad` and troubleshooting import errors. The benefits of using **Tinygrad** in various environments and with varying methodologies were discussed.

- **Tensor Padding and Transformer Errors**: A user brought up an issue with tensor padding and transformer models, leading to an exploration of error messages and the potential adjustments to accommodate padding in model architectures within Tinygrad. Through trial and error, it was confirmed that using `None` for padding is a viable solution.

- **Documentation and Contributions**: Links to Tinygrad documentation and personal study notes were shared to assist newcomers with understanding the inner workings of Tinygrad. Questions about contributing to Tinygrad without CUDA support were addressed, emphasizing the value of familiarizing oneself with the documentation and codebase prior to contributing.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1vy97ByB5mLielmE8TzlnnaSZtMUykm-A?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/scheduleitem.md">tinygrad-notes/scheduleitem.md at main · mesozoic-egg/tinygrad-notes</a>: Tutorials on tinygrad. Contribute to mesozoic-egg/tinygrad-notes development by creating an account on GitHub.</li><li><a href="https://github.com/tinygrad/tinygrad/tree/master/docs">tinygrad/docs at master · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/codegen.md">tinygrad-notes/codegen.md at main · mesozoic-egg/tinygrad-notes</a>: Tutorials on tinygrad. Contribute to mesozoic-egg/tinygrad-notes development by creating an account on GitHub.</li><li><a href="https://github.com/tinygrad/tinygrad">GitHub - tinygrad/tinygrad: You like pytorch? You like micrograd? You love tinygrad! ❤️</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - GitHub - tinygrad/tinygrad: You like pytorch? You like micrograd? You love tinygrad! ❤️</li><li><a href="https://github.com/pranjaldatta/tinygrad">GitHub - pranjaldatta/tinygrad: A simple, basic autodiff engine written to learn the inner workings of autograd!</a>: A simple, basic autodiff engine written to learn the inner workings of autograd! - pranjaldatta/tinygrad
</li>
</ul>

</div>
  

---



**Interconnects (Nathan Lambert) ▷ #[ideas-and-feedback](https://discord.com/channels/1179127597926469703/1179127598442348730/1228861251342307408)** (5 messages): 

- **Organizing Artifacts**: A new blog feature introduces [Hugging Face collections](https://huggingface.co/collections/natolambert/2024-interconnects-artifacts-6619a19e944c1e47024e9988) to organize all artifacts linked at the bottom of the blog post for "open models and datasets".
- **Ease of Access**: The collections were created as a way to easily find and re-access the linked artifacts.
- **Planning for Team Expansion**: The creator envisions establishing an "interconnects" Hugging Face organization when someone is hired to help.
- **API for Collections Available**: It was mentioned that Hugging Face collections have an API, which could be used now, demonstrated by the [Hugging Face documentation](https://huggingface.co/docs/huggingface_hub/en/package_reference/collections).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/huggingface_hub/en/package_reference/collections">Managing collections</a>: no description found</li><li><a href="https://huggingface.co/collections/natolambert/2024-interconnects-artifacts-6619a19e944c1e47024e9988">2024 Interconnects Artifacts - a natolambert Collection</a>: no description found
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1228381066197336145)** (11 messages🔥): 

- **Frustration over Hard Fork**: There is frustration expressed about this week's **hard fork**, perceived as triggering due to nonsense that is not reflective of the current state of AI.
- **Skepticism Towards Claude Increment**: The transition from **Claude 2 to Claude 3** is being questioned; the term "INCREMENTAL?" suggests skepticism about its significance.
- **Pointing Out Open Data Initiatives**: There is a reminder about the existence of open data initiatives, indicating they may be overlooked in current discussions, implied by the phrase, *"no one is detailing open data? AHEM"*.
- **Journalist Perception on Tech**: A member remarks on how **popular tech journalists** have a seemingly adversarial attitude towards technology, which is deemed perplexing.
- **Engagement with Criticism Welcomed**: Following criticism on **Twitter**, there's a positive note about engagement as a tech journalist began following a member after they tweeted at them.
- **AI Releases Flood the Scene**: A slew of releases announced today, including **Pile-T5 from EleutherAI**, **WizardLM 2**, and **Reka Core** from Yi Tay with [links provided](https://blog.eleuther.ai/pile-t5/) to their respective sources. **Wizard LM 2** is highlighted as particularly interesting.
- **Open Source Copyright Makeover**: A comment praises the decision to make **Dolma ODC-BY**, reflecting on the intense number of open source announcements with the comment, *"God can open source stop with the announcements today 😭"*.
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1228565155957178431)** (13 messages🔥): 

- **Confusion over ACL commitment revisions**: A member asked if it's necessary to upload a revised pdf after review, Nato expressed uncertainty suggesting that it might be required at some point.
- **Synthetic Data Discussions Heat Up**: One member expressed enthusiasm for Nato's article on [synthetic data and CAI](https://www.interconnects.ai/p/llm-synthetic-data), sparking a conversation about the synthetic data meta since late 2023. Nato mentioned ongoing progress and hinted at a dataset release based on revisions.
- **ORPO Method — What's the Deal?**: The ORPO method was met with skepticism, with Nato suggesting that having a "big model" is a good strategy, **implying the importance of model size** over specific techniques.
- **Critic vs. Reward Models in AI Analysis**: Members discussed the difference between "critic" models like **Prometheus** and "reward" models featured on Nato's RewardBench project. The conversation delved into their functions, such as offering binary preferences or scalar scores, versus critics providing revisions or critiques.
- **To Upload or Not to Upload PDFs**: A link to a tweet by @andre_t_martins was shared, indicating that **uploading a revised pdf for ACL commitment might not be necessary** after all. [See the tweet here](https://twitter.com/andre_t_martins/status/1779263540785725737).
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1229465023680741446)** (4 messages): 

- **Positive Reception for Graphs in Newsletter**: A member expressed enjoyment of the graphs included in today's newsletter, complimenting their clarity and effectiveness.
- **Commitment to Including Graphs**: The channel owner confirmed that the use of these informative graphs will be a consistent feature going forward.
- **Graph Enhancements on the Horizon**: Future improvements were discussed, with plans to refine the graphs and incorporate them into a Python library.
- **Foundation for Exploring Open Models**: The owner signaled that these polished graphs will serve as a valuable tool when writing about new open models.
  

---


**Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1228425858675507262)** (1 messages): 

- **CodecLM Research Paper Discussed**: A member shared a [link to a research paper](https://arxiv.org/pdf/2404.05875.pdf) titled "CodecLM: Aligning Language Models with Tailored Synthetic Data" published by Google on April 8. They mentioned it seems to be an approach on *learning from a stronger model*, but didn't provide further analysis.
  

---


**Interconnects (Nathan Lambert) ▷ #[sp2024-history-of-open-alignment](https://discord.com/channels/1179127597926469703/1223784028428177510/1228412327486029916)** (1 messages): 

- **LLaMA Paper Shared**: A member posted a link to the paper "LLaMA: Open and Efficient Foundation Language Models" on Hugging Face's Collections, published on February 27, 2023. This paper can be found with the identifier [2302.13971](https://huggingface.co/papers/2302.13971).

**Link mentioned**: <a href="https://huggingface.co/collections/natolambert/aligning-open-language-models-66197653411171cc9ec8e425">[lecture artifacts] aligning open language models - a natolambert Collection</a>: no description found

  

---


**Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1229469203468259510)** (2 messages): 

- **Patience Might Be a Virtue**: A member is experimenting with a bot and mentions the possibility of needing to wait longer rather than intervening manually. The context or reason for waiting is not specified.
  

---



**LLM Perf Enthusiasts AI ▷ #[claude](https://discord.com/channels/1168579740391710851/1168582222194933860/1228312970837364767)** (25 messages🔥): 

- **Haiku Hesitation**: Some members are expressing concerns about **Haiku's speed**, mentioning issues with total response times rather than throughput. There's a suggestion that **Bedrock** could offer better response times, although one member noted that it's slow as well.

- **Roleplay Refusal by Claude**: Frustration is apparent as **Claude** is not cooperating with roleplay scenarios, including acting as a warrior maid or sending fictional spam mail. The issue persists despite attempts with instruction prompting and one-shot or few-shot techniques.

- **Jailbreaking Claude 3?**: Members are discussing the necessity of **jailbreaking Claude 3** for edgy content such as roleplaying games. One member referenced a tweet from *@elder_plinius* reporting a universal jailbreak for Claude 3, which could bypass stricter content filtering found in models like **Gemini 1.5 Pro**. The tweet can be found [here](https://x.com/elder_plinius/status/1773455789056745782?s=46).

**Link mentioned**: <a href="https://x.com/elder_plinius/status/1773455789056745782?s=46">Tweet from Pliny the Prompter 🐉 (@elder_plinius)</a>: JAILBREAK ALERT!  Just discovered a universal jailbreak for Claude 3. Malware, hard drug recipes, bomb-making, the whole nine yards.   Haiku at high temps seems to be the best combo but in my limited ...

  

---


**LLM Perf Enthusiasts AI ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/1228403053653397544)** (4 messages): 

- **Code Capabilities Upgraded**: A member noted that a newer version is "**better at code for sure**" and also "**Faster too**," suggesting performance improvements in coding tasks.
- **Considering a Reactivation for Testing**: In light of the perceived improvements, a member is considering reactivating their **ChatGPT Plus** subscription to test the new capabilities.
- **Claude Maintains an Edge for Long-Winded Code**: Claude is still considered valuable for "long context window code tasks," indicating that ChatGPT might have limitations with the context window size.
  

---



**DiscoResearch ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/1228272255696371734)** (8 messages🔥): 

- **The MoE Mystery Continues**: A member expressed a sense that something crucial is lacking in the MoE training/finetuning/rlhf process in transformers. The specificity of the missing elements was not elaborated.

- **Fishy Model Merge Remembered**: In a [Reddit discussion](https://huggingface.co/Envoid), the model "fish" by Envoid was recommended as an effective merge for RP/ERP in Mixtral. Even though it remains untested by the commentator due to hardware constraints, the author's contributions to Mixtral merge techniques were noted as potentially helpful.

- **Mixtral's Secret Sauce Speculation**: Participants speculated whether Mixtral possesses undisclosed "secrets" for effective training beyond what's presented in their paper, suggesting that there might be hidden elements to achieving better MoE performance.

- **Weekend Ventures in Mixtral Finetuning**: A member intends to experiment with Mixtral over the weekend, focusing on full finetuning with en-de instruct data, mentioning that the zephyr recipe code appears to work well.

- **Searching for the Superior Mixtral Sauce**: A query was raised regarding the ability to fine-tune Mixtral models in a way that outperforms the official Mixtral Instruct, specifically about whether the "secret sauce" related to routing optimization has been uncovered by any community members.
  

---


**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1229399893198639215)** (4 messages): 

- **Custom Llama-Tokenizer Training Quandary**: A member is exploring how to train a custom **Llama-tokenizer** to optimize size for deployment on small hardware, with the aim of minimizing the size of the embedding layer. They noted difficulties in creating the required *tokenizer.model* file and are seeking assistance to resolve this for successful use with **llama.cpp**.

- **Token Generation Tactics**: Another participant recommended looking into using a script available on GitHub that might assist with the conversion process for the custom **Llama-tokenizer**. They directed attention to the script `convert_slow_tokenizer.py` found in the [Hugging Face Transformers repository](https://github.com/huggingface/transformers/blob/fe2d20d275d3591e2619a1adb0fa6ae272605208/src/transformers/convert_slow_tokenizer.py#L534).

- **Convert With Ease Using llama.cpp Script**: A suggestion was made to use the `convert.py` script from the [llama.cpp GitHub repository](https://github.com/ggerganov/llama.cpp/blob/master/convert.py) with the `--vocab-only` option, implying this could be a solution for the tokenizer model file creation issue.

- **Calling Data Collectors for Open Multimodal Model Initiative**: A call for assistance was made to locate copyright permissive or free EU text and multimodal data for training a **large open multimodal model**. Interested parties with the relevant data have been requested to get in touch via direct message.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/ggerganov/llama.cpp/blob/master/convert.py">llama.cpp/convert.py at master · ggerganov/llama.cpp</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.</li><li><a href="https://github.com/huggingface/transformers/blob/fe2d20d275d3591e2619a1adb0fa6ae272605208/src/transformers/convert_slow_tokenizer.py#L534">transformers/src/transformers/convert_slow_tokenizer.py at fe2d20d275d3591e2619a1adb0fa6ae272605208 · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1228317593643913266)** (2 messages): 

- **Occiglot Model Shows Template Sensitivity**: Evaluation of **occiglot/occiglot-7b-de-en-instruct** demonstrated improved performance on German RAG tasks using the correct template as found on [Hugging Face's correct template](https://huggingface.co/datasets/deutsche-telekom/Ger-RAG-eval#occiglotocciglot-7b-de-en-instruct), compared to the [wrong template](https://huggingface.co/datasets/deutsche-telekom/Ger-RAG-eval#occiglotocciglot-7b-de-en-instruct-1). The issue was likely due to a step in the Data Processing Operation rather than a systematic error.

- **StableLM and MiniCPM Pretraining Insights Shared**: Quicksort discussed the StableLM's report found in their [technical paper](https://arxiv.org/abs/2402.17834) which includes using a **ReStruct dataset** for pretraining, inspired by methods from [this GitHub repository](https://github.com/ExpressAI/reStructured-Pretraining). Furthermore, MiniCPM's research, outlined in their ablation study in [Chapter 5](https://arxiv.org/abs/2404.06395), indicates benefits from mixing in data like OpenOrca and EvolInstruct during the cooldown phase of pretraining.

**Link mentioned**: <a href="https://arxiv.org/abs/2402.17834">Stable LM 2 1.6B Technical Report</a>: We introduce StableLM 2 1.6B, the first in a new generation of our language model series. In this technical report, we present in detail the data and training procedure leading to the base and instruc...

  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1228288816482353204)** (9 messages🔥): 

- **Burnai Project Sparks Interest**: A member praised the promising capabilities of the [Burnai project](https://burn.dev/), emphasizing its potential for optimizing inference on various platforms through Rust code. They shared their curiosity as to why Mozilla hasn’t investigated Burnai, despite Rust being Mozilla's brainchild.

- **Llamafile on Mcaffee Whitelist**: The community celebrated the whitelisting of *llamafile 0.7 binary* by Mcaffee, highlighting it as a positive step for the project.

- **Welcome New Collaborator**: An introduction from a new participant expressed enthusiasm for great collaborations and conversations within the community.

- **Queries about Vulkan and tinyblas**: Questions were raised regarding possible Vulkan support in the upcoming v0.8 release and the upstreaming of *tinyblas*, with a specific interest in its application for ROCm.

- **Guidance Sought for Packaging Custom Models**: There's noticeable community interest in guides for packaging customized models into llamafile, which has prompted members to seek and share relevant resources, such as a [GitHub Pull Request on container publishing](https://github.com/Mozilla-Ocho/llamafile/pull/59#issuecomment-1840814790).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://burn.dev/">Burn</a>: no description found</li><li><a href="https://github.com/Mozilla-Ocho/llamafile/pull/59#issuecomment-1840814790">Publish container to Docker Hub by dzlab · Pull Request #59 · Mozilla-Ocho/llamafile</a>: Build and Publish container to Docker Hub on release using Github Actions #29 For this to work, need to setup the repository secrets:  DOCKER_HUB_USERNAME DOCKER_HUB_ACCESS_TOKEN</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6414">Improve cpu prompt eval speed by jart · Pull Request #6414 · ggerganov/llama.cpp</a>: This change upstreams llamafile&#39;s cpu matrix multiplication kernels which improve image and prompt evaluation speed. For starters, Q4_0 and Q8_0 weights should go ~40% faster on CPU. The biggest b...
</li>
</ul>

</div>
  

---



**Alignment Lab AI ▷ #[ai-and-ml-discussion](https://discord.com/channels/1087862276448595968/1087876677603958804/1228638326945218632)** (2 messages): 

- **Spam Solution Suggestion**: A member recommended inviting the **wick bot** to the chat to automatically remove spam messages. They believe this could prevent incidents similar to those indicated by the tagged message.
  

---


**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1228342844377403512)** (4 messages): 

- **Request for Direct Messaging**: A member expressed the need for help with their code and requested a DM by tagging a specific user.
- **Concerns Over Discord Invites**: Another member voiced their concern regarding the posting of Discord invites across the server and suggested that they should be outright banned to avoid such issues. 
- **Human Verification Call**: The same member who needed help with their code reiterated their request and mentioned it as a proof that they are not a bot.
  

---


**Alignment Lab AI ▷ #[join-in](https://discord.com/channels/1087862276448595968/1143791237669855302/1229226519327543377)** (1 messages): 

There are no appropriate messages to summarize.

**Link mentioned**: <a href="https://discord.gg/VWffHaCpED">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


**Alignment Lab AI ▷ #[oo2](https://discord.com/channels/1087862276448595968/1176548760814375022/)** (1 messages): 

aslawliet: Is the project still alive?
  

---



**AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1228632341036011622)** (6 messages): 

- **Code-Jamba-v0.1 Takes on Multi-Language Coding**: A fine-tuned model named [Code-Jamba-v0.1](https://huggingface.co/ajibawa-2023/Code-Jamba-v0.1) was shared, renowned for high proficiency in generating code across multiple programming languages including **Python, Java, and Rust**. It was trained for **162 hours on 2 x H100 GPUs**, utilizing datasets [Code-290k-ShareGPT](https://huggingface.co/datasets/ajibawa-2023/Code-290k-ShareGPT) and [Code-Feedback](https://huggingface.co/datasets/m-a-p/Code-Feedback).

- **Queries on Model Benchmarking and Dataset Sharing**: A member expressed curiosity about the absence of model evaluation data for **Code-Jamba-v0.1** and raised a question on whether AI21 Labs will share its dataset, highlighting its potential for fine-tuning Large Language Models (LLMs) with a 256K context.

- **Jamba API Anticipation**: An inquiry was made about the availability of an API for **Jamba**, with a note that Fireworks AI might integrate Jamba following communication with their CEO.

**Link mentioned**: <a href="https://huggingface.co/ajibawa-2023/Code-Jamba-v0.1">ajibawa-2023/Code-Jamba-v0.1 · Hugging Face</a>: no description found

  

---



**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1229235258344345630)** (5 messages): 

- **Introduction to a Password Guessing Game**: Members shared a link to [PasswordGPT](https://passwordgpt.io/), a game where the objective is to convince an AI to reveal a password.
- **Debate on Annotating Data**: A member expressed a preference for annotating data before modeling, to understand dataset contents, and wondered if others follow a similar practice or rely more on LLMs without extensive annotation.
- **Historical Document Record Extraction Using Pre-LLM Models**: A member mentioned being part of a team that curates data and extracts records from historical documents using transformer models that predate LLMs.
- **User Engagement with AI Demos**: There was a preference for open-prompt AI demos as opposed to closed-prompt guessing games, stressing that seeing the prompts can help users engage more effectively and beat the AI.

**Link mentioned**: <a href="https://passwordgpt.io/">PasswordGPT</a>: no description found

  

---



**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1229410199375188048)** (1 messages): 

- **Scaling AI Apps to Production**: An announcement for a meetup titled "**Unlock the Secrets to Scaling Your Gen AI App to Production**" was made, highlighting the challenge of scaling a Gen AI app to full-scale production. The event features panelists including the co-founders of [Portkey](https://portkey.ai/), [Noetica AI](https://www.noetica.ai/), and [LastMile AI](https://lastmileai.dev/), with registration available through [this link](https://lu.ma/llms-in-prod-nyc). Interested parties need approval from the host to attend.

**Link mentioned**: <a href="https://lu.ma/llms-in-prod-nyc">LLMs in Prod w/ Portkey, Flybridge VC, Noetica, LastMile · Luma</a>: Unlock the Secrets to Scaling Your Gen AI App to Production  While it&#x27;s easy to prototype a Gen AI app, bringing it to full-scale production is hard. We are bringing together practitioners &amp;....

  

