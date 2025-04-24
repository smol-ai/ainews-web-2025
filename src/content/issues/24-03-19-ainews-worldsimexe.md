---
id: d367a428-c79b-48d6-ac14-2d8b0a5bfb67
title: World_sim.exe
date: '2024-03-20T00:46:48.498362Z'
original_slug: ainews-to-be-named-9615
description: >-
  NVIDIA announced Project GR00T, a foundation model for humanoid robot learning
  leveraging their tech stack, and revealed the DGX Grace-Blackwell GB200 system
  capable of over 1 exaflop compute, able to train GPT-4’s *1.8 trillion*
  parameter model in *90 days*. Additionally, Stability AI released Stable Video
  3D, an open-source text-to-video generation tool that creates 3D meshes from
  single images, marking a significant advancement in video generation
  technology.
tags:
  - nvidia
  - stability
  - openai
  - nous-research
  - langchain
  - xai
  - anthropic
  - perplexity
  - unsloth
  - cohere
  - gpt-4
  - gpt-5
  - grok-1
  - stable-video-3d
  - stable-cascade
  - sdxl
  - claude-3
  - claude-3-opus
  - llama-cpp
  - command-r
  - foundation-models
  - humanoid-robotics
  - multimodal-instructions
  - high-performance-computing
  - large-language-models
  - retrieval-augmented-generation
  - text-to-video-generation
  - model-benchmarking
  - fine-tuning
  - hardware-optimization
  - photonic-computing
  - open-source-models
  - model-parameter-scaling
  - AI-reasoning
  - cultural-implications-of-LLMs
companies:
  - nvidia
  - stability
  - openai
  - nous-research
  - langchain
  - xai
  - anthropic
  - perplexity
  - unsloth
  - cohere
models:
  - gpt-4
  - gpt-5
  - grok-1
  - stable-video-3d
  - stable-cascade
  - sdxl
  - claude-3
  - claude-3-opus
  - llama-cpp
  - command-r
topics:
  - foundation-models
  - humanoid-robotics
  - multimodal-instructions
  - high-performance-computing
  - large-language-models
  - retrieval-augmented-generation
  - text-to-video-generation
  - model-benchmarking
  - fine-tuning
  - hardware-optimization
  - photonic-computing
  - open-source-models
  - model-parameter-scaling
  - AI-reasoning
  - cultural-implications-of-LLMs
---


<!-- buttondown-editor-mode: plaintext -->> AI News for 3/18/2024-3/19/2024. We checked [**358** Twitters](https://twitter.com/i/lists/1585430245762441216) and **21** Discords (**337** channels, and **9841** messages) for you. Estimated reading time saved (at 200wpm): **1033 minutes**.


Lots of Nvidia GTC recaps out there - [youtube](https://www.youtube.com/watch?v=bMIRhOXAjYk) does a better job than we can.

We were accidentally part of the news cycle yesterday, with Karan (CEO of Nous Research) [demoing his world_sim.exe explorations](https://twitter.com/swyx/status/1769920689832972574). It's purely for fun, but a very interesting exploration of where roleplay prompt engineering can take you.

---

**Table of Contents**

[TOC] 


---

# PART X: AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs


**NVIDIA GTC Announcements**

- [NVIDIA announced](https://twitter.com/DrJimFan/status/1769860044324319658) Project GR00T, an initiative to create a foundation model for humanoid robot learning that can understand multimodal instructions and perform useful tasks. It is born on NVIDIA's tech stack including Isaac Lab, OSMO, and Jetson Thor. (495k views)
- [NVIDIA revealed](https://twitter.com/DrJimFan/status/1769829758479876130) the DGX Grace-Blackwell GB200, exceeding 1 exaflop compute in a single rack. It can train GPT-4 1.8T parameters in 90 days on 2000 Blackwells. (291k views)
- [Jensen Huang announced](https://twitter.com/ethanCaballero/status/1769821908285964642) that GPT-4 is 1.8 trillion parameters. (36k views)
- [NVIDIA's new GPU GB200](https://twitter.com/danielhanchen/status/1769927958976963042) has float4/6 precision. It uses ~3 bits per parameter, similar to the 1.58bit paper. 40,000 TFLOPs of fp4 is with 2x sparsity. On fp8 it gets 20 PFLOPs vs H100's 8 PFLOPs. It has 384 GB VRAM. (19k views)

**Open Source LLMs and Implementations** 

- [Grok-1, a 340B parameter open source model, was released.](https://twitter.com/ibab_ml/status/1769770983924142475) The repo is gaining popularity. (208k views)
- [Nous worked with @voooooogel](https://twitter.com/Teknium1/status/1769752208466383205) to implement Steering Vectors/Control Vectors into Llama.CPP based on the original paper by @jam3scampbell. (13k views)
- [Stability AI released SV3D](https://twitter.com/slashML/status/1769938991577489555), an open-source solution for text-to-video generation including the entire training process. (1k views)

**Retrieval Augmented Generation (RAG)**

- [A new long-form (5.5 hour) RAG tutorial is available](https://twitter.com/mrdbourke/status/1769897780796117227), building a pipeline from scratch to create "NutriChat" using a 1200 page nutrition PDF. It uses free open-source HF models, no APIs needed. (6k views)
- [LangChain released a 10th video](https://twitter.com/LangChainAI/status/1769759838094106818) in their RAG From Scratch series, focusing on query routing using logical reasoning with an LLM or semantic similarity. (35k views)
- [LangChain announced an integration](https://twitter.com/LangChainAI/status/1769851779003695143) with NVIDIA NIM for GPU-optimized LLM inference in RAG applications. (15k views)

**Emerging Trends and Opinions**

- [Yann LeCun shared thoughts](https://twitter.com/ylecun/status/1769768065712177615) that language is distinct from other cognitive abilities like planning, reasoning, empathy, etc. Vast portions of cognition appear unrelated to language. (151k views) 
- [Sam Altman implied AGI may arrive in 6 years](https://twitter.com/AISafetyMemes/status/1769766600486584553) in his Lex Fridman interview. He said the leap from GPT-4 to GPT-5 will be similar to GPT-3 to GPT-4. OpenAI is less caught up in the culture war compared to others. (66k views)
- [There are discussions on whether LLMs like Claude are "alive"](https://twitter.com/KevinAFischer/status/1769771964443316563) in some philosophical sense, marking a new cultural divide. (6k views)
- [Opinions suggest](https://twitter.com/finbarrtimbers/status/1769733698499535069) most companies shouldn't train models from scratch as it often doesn't make sense. (13k views)


---

# PART 0: Summary of Summaries of Summaries


> Since [Claude 3 Haiku was released recently](https://x.com/anthropicai/status/1768018310615151002?s=46&t=90xQ8sGy63D2OtiaoGJuww), we're adding them to this summary run for you to compare. We'll keep running these side by side for a little longer while we build the AINews platform for a better UX.

## Claude 3 Haiku (3B?)


- **Hardware Optimization for AI**: Members actively shared strategies and experiences around building efficient hardware setups for running LLMs, including discussions on leveraging EPYC CPUs, cooling Tesla K80 GPUs, and evaluating the cost-benefit tradeoffs of different GPU options.

- **Advancements in Photonic Computing**: The community expressed excitement about the potential of photonic computer chips, with members sharing educational resources like [Asianometry's videos on silicon photonics and light meshes](https://www.youtube.com/watch?v=29aTqLvRia8) and [a video on a new chip breakthrough](https://youtu.be/8ohh0cdgm_Y).

- **Exploring Retrieval-Augmented Generation (RAG)**: Discussions delved into optimizing RAG models, designing specialized "RAG modes," and leveraging Cohere's model robustness, with members sharing resources like a [step-by-step guide on implementing RAG](https://youtu.be/w7Ap6gZFXl0) and a [Medium article on empowering AI assistants with RAG](https://medium.com/ai-advances/empowering-voices-ai-assistant-with-rag-pipeline-memory-and-llamaindex-11c4e319d915).

- **Benchmarking and Evaluating Large Language Models**: The community expressed skepticism about current LLM benchmarking systems, questioning their reliability and reproducibility, especially for larger models like GPT-4 and Grok-1. Members discussed the need for more robust and standardized evaluation approaches.

- **Advancements in AI Reasoning and Inference**: Discussions highlighted novel approaches to enhancing LLM reasoning capabilities, such as the [Maisa Knowledge Processing Unit (KPU)](https://maisa.ai/blog/kpu) and the [Quiet-STaR project](https://arxiv.org/abs/2403.09629) on teaching LLMs to "think before speaking."

## Claude 3 Sonnet (14B?)

- **Large Language Model (LLM) Innovations and Releases**:

  - Stability AI released [Stable Video 3D](https://stability.ai/news/introducing-stable-video-3d), creating 3D meshes from single images.
  - xAI released [Grok-1](https://github.com/xai-org/grok-1), a 314B parameter Mixture-of-Experts model, sparking discussions on performance and resource requirements.
  - GPT-4's rumored 1.8T parameters from an [NVIDIA GTC keynote](https://www.youtube.com/watch?v=Y2F8yisiS6E) intrigued multiple communities.

- **Model Performance Comparisons and Benchmarking**:

  - Users compared Stable Cascade and SDXL, with Cascade handling complex prompts better but running slower (Stability.ai).
  - Grok-1's performance was debated, with some skepticism about its real-world capabilities despite its size (Nous Research AI, Latent Space).
  - Concerns were raised about the reliability of LLM benchmarking systems and the reproducibility of openLLM leaderboard results (Eleuther).

- **Fine-tuning and Model Optimization Techniques**:

  - [Unsloth AI repository](https://github.com/unslothai/unsloth) trended for its faster and memory-efficient QLoRA & LoRA fine-tuning (Unsloth AI).
  - Discussions covered Gemma 7b vs Mistral-7b performance and [Mixtral branch support](https://github.com/unslothai/unsloth/pull/145) (Unsloth AI).
  - A promising method to improve model accuracy and sample efficiency was teased, with plans for larger-scale testing (Skunkworks AI).

- **Hardware and Infrastructure Discussions**:

  - NVIDIA's upcoming [GeForce RTX 50-series "Blackwell"](https://www.techpowerup.com/320185/nvidia-geforce-rtx-50-series-blackwell-to-use-28-gbps-gddr7-memory-speed) with GDDR7 memory and 55% higher bandwidth was noted (Nous Research AI, OpenAccess AI Collective).
  - ROCm compatibility issues and multi-GPU support in LM Studio were discussed (LM Studio).
  - Strategies for optimizing hardware for LLMs, including DIY EPYC systems and GPU cooling, were explored (LM Studio).

- **API and Model Integrations**:

  - Users anticipated the [Command-R model integration](https://github.com/ggerganov/llama.cpp/pull/6033) in LM Studio.
  - Perplexity AI's API inconsistencies and token limitations were discussed (Perplexity AI).
  - OpenAI's API was used for experiments in personalized blog post translations (Datasette - LLM).

- **Open Source Definitions and Licensing**:

  - The release of Grok-1 under the Apache 2.0 license sparked discussions about the implications of open-sourcing large models (Interconnects).
  - The community debated the need for a clear, practical definition of open source to avoid disputes (Interconnects).

## Claude 3 Opus (>220B?)

- **Stable Video 3D Revolutionizes Content Creation**: Stability AI released [Stable Video 3D](https://stability.ai/news/introducing-stable-video-3d), which creates 3D meshes and novel multi-views from single object images, surpassing the capabilities of previous technologies like **Stable Zero123** and its open-source alternatives.

- **Cascade Outperforms SDXL in Complexity**: In user benchmarks, Stable Cascade topped SDXL in handling complex prompts, despite having a slower generation time—approximately 90 seconds versus SDXL's 30 seconds.

- **Claude 3 Opus Shines but "Unlimited" Queries Mislead**: Perplexity Pro users get "unlimited" daily queries on **Claude 3 Opus**, but a 600-query cap leaves some feeling misled. Meanwhile, Claude 3 Opus impresses in creative writing tasks and comparisons, like [experimenting with phone comparisons](https://www.perplexity.ai/search/how-does-the-tzGt2.woRkCHJhNXY8V_Gg).

- **Impersonator Alert for Unsloth AI**: Users warn about a scam account impersonating Unsloth developer **Daniel Han** ([starsupernova0](https://discord.com/users/starsupernova0)) sending friend requests. The [Unsloth AI repository](https://github.com/unslothai/unsloth) is trending on GitHub, offering a toolkit for faster and memory-efficient QLoRA & LoRA fine-tuning.

- **Grok-1 Release Stirs Debate**: The release of **Grok-1**, a 314B parameter MoE model, has the AI community reacting with a mix of excitement and skepticism about its real-world performance. Discussions touch on the model's [open-source release on GitHub](https://github.com/xai-org/grok-1) and a potential [GPT-4 leak during NVIDIA's GTC keynote](https://www.youtube.com/watch?v=Y2F8yisiS6E).

- **Photonics Piques Interest in CUDA Community**: Advancements in photonic computer chips, like NVIDIA's upcoming **GeForce RTX 50-series "Blackwell"** with [28 Gbps GDDR7 memory](https://www.techpowerup.com/320185/nvidia-geforce-rtx-50-series-blackwell-to-use-28-gbps-gddr7-memory-speed), are a hot topic. Educational resources like [Asianometry's videos on silicon photonics](https://www.youtube.com/watch?v=29aTqLvRia8) are shared.

- **Triton Puzzles Challenge GPU Enthusiasts**: A new set of [Triton Puzzles](https://colab.research.google.com/drive/1AJc8RFsDeJ3Vx3gRq5dUqmcb-Cy1G8qh?usp=sharing) is introduced for educational GPU problem-solving, despite some initial bugs. The CUDA community actively discusses memory management strategies like "Producer Provides" and "Consumer Takes" for optimizing pipeline parallel implementations for LLM inference.

- **Axolotl Boosts Performance with ScatterMoE**: The Axolotl dev team introduces **ScatterMoE**, an optimization promising significant throughput improvements over Huggingface MoE implementation, with code available on their [GitHub branch](https://github.com/OpenAccess-AI-Collective/axolotl/tree/scatter_moe). PyTorch 2.2 or higher is recommended for compatibility.

- **APIs Leak LLM Secrets**: Researchers discover APIs can inadvertently reveal information about proprietary LLMs, including architecture details, with less than $1,000 spent experimenting on [OpenAI's GPT-3.5-turbo](https://arxiv.org/abs/2403.09539). Concerns grow over underestimated LLM sizes and potential use of Mixture of Experts (MoE) architectures.

- **Maisa's KPU Promises Reasoning Leaps**: Maisa unveils its new **Knowledge Processing Unit (KPU)**, which integrates with LLMs for enhanced complex task solving. The [KPU white paper and blog](https://maisa.ai/blog/kpu) detail its architecture and potential, but some express skepticism without more substantial evidence and comparisons to GPT-4-turbo.


## ChatGPT (GPT4T)

- **AI Content Creation Evolution**: Stability AI's [Stable Video 3D](https://stability.ai/news/introducing-stable-video-3d) marks a significant leap in AI-driven 3D content generation, illustrating the rapid evolution of AI capabilities in creating complex, multi-view content from single images. This advancement not only showcases technological progression but also raises discussions on the future of content creation, pushing the boundaries of what's possible with AI.

- **Blockchain's Influence on AI**: Stability AI's foray into blockchain partnerships reflects a broader tension within the AI community regarding blockchain technology's role in AI development. While some see potential for innovation, others express concerns about accessibility, openness, and the future direction of AI platforms, highlighting a critical debate on balancing technological advancement with community values and accessibility.

- **Model Performance and Ethical Considerations**: The anticipation and skepticism surrounding Perplexity AI's Claude 3 Opus and Stability AI's partnerships underscore ongoing concerns about model performance, ethical considerations, and transparency within the AI community. These discussions reflect broader debates on the ethical implications of AI technologies, the need for transparent communication from AI companies, and the importance of aligning AI development with ethical standards and user expectations.

- **Technological Enhancements and Community Engagement in AI Optimization**: The community's support for Unsloth AI's GitHub project highlights a keen interest in technological enhancements aimed at improving AI efficiency and reducing resource consumption. This engagement signifies the community's drive towards optimizing AI technologies for better performance and lower barriers to entry, reflecting a collective effort to push the boundaries of AI optimization and application.

- **AI Ethics, Openness, and Accessibility Debate**: The discussions around AI's role in scientific peer review, as seen in Nous Research AI's examination, and the debate on LLMs sparked by Latent Space, highlight an ongoing discourse on the ethical considerations, openness, and accessibility of AI technologies. These debates encompass concerns about the impact of AI on scientific integrity, the ethical use of AI in content creation, and the balance between proprietary advancements and open innovation, underscoring the complexity of navigating AI development within ethical and accessible frameworks.

- **AI Training Methodologies and Data Management**: The conversation in Eleuther about optimizing language model scaling laws and LAION's update on the DALL-E 3 dataset relocation to a new [Hugging Face repository](https://huggingface.co/datasets/OpenDatasets/dalle-3-dataset) exemplifies the continuous innovation in AI training and data handling practices. These discussions emphasize the AI community's focus on enhancing model efficiency, accuracy, and general capabilities while ensuring data accessibility and reproducibility, showcasing the ongoing effort to refine AI technologies for broader applicability and impact.

These themes collectively capture the dynamic and multifaceted nature of the AI landscape, characterized by rapid technological advancements, ethical and policy debates, community engagement in optimization efforts, and the ongoing quest for enhancing AI training and data management practices.

---

# PART 1: High level Discord summaries




## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Video 3D Revolutionizes Content Creation**: Stability AI released [Stable Video 3D](https://stability.ai/news/introducing-stable-video-3d), which creates 3D meshes and novel multi-views from single object images, surpassing the capabilities of previous technologies like **Stable Zero123** and its open-source alternatives.

- **Cascade Outperforms SDXL in Complexity**: In user benchmarks, Stable Cascade topped SDXL in handling complex prompts, despite having a slower generation time—approximately 90 seconds versus SDXL's 30 seconds.

- **Blockchain Buzz Kills the Mood**: Concerns were voiced by users over Stability AI's partnerships with blockchain entities, fearing this may negatively impact the future openness and accessibility of models like the much-anticipated Stable Diffusion 3 (SD3).

- **The Wait for SD3 Beta Heats Up**: The user community is on the edge of their seats, eagerly anticipating the beta release of Stable Diffusion 3, which is rumored to combine high-quality outputs with efficient runtime.

- **Safetensor Conversion Confusion**: An inquiry about converting PyTorch .pt files to safetensors sparked a discussion, with the consolation being that most user interfaces prevent script execution from .pt files, thereby lowering security risks, though no direct solution was mentioned.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Unlimited But Not Really**: Perplexity Pro users are promised **unlimited daily queries** on Claude 3 Opus but find the term "unlimited" misleading due to an apparent 600-query cap. Concerns point towards possible misrepresentation and legal repercussions.

- **Claude 3 Opus Under the Creative Microscope**: Users delve into Claude 3 Opus' abilities in creative writing with a prompt about escalating intelligence, reminding others to share threads for increased visibility, and experimenting with phone comparisons.

- **Midjourney Draws Lines on Stability AI**: In a critical discussion for AI developers, Midjourney's stance on Stability AI sparks conversations about policy and partnerships within the AI community.

- **API Model's Fate Hangs in the Balance**: Engineers discuss the unexpected continued service of a model slated for deactivation and note a discrepancy in the Sonar API's news responses regarding Donald Trump, highlighting the unpredictability in content consistency.

- **Job Searches and Token Limitations via API**: Whilst Perplexity's API intrigue users with its potential to retrieve job postings, there's frustration with inconsistent results and curiosity about how max token settings impact the quality of responses.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Beware of the Impersonator**: Alert issued on a fake **Daniel Han** ([starsupernova0](https://discord.com/users/starsupernova0)) attempting to scam Unsloth AI users via friend requests. Users are urged to report such accounts to maintain community security.

- **Unsloth AI Shines on GitHub**: The [Unsloth AI repository](https://github.com/unslothai/unsloth) is trending, offering a toolkit for **2-5X faster and 70% less memory usage during QLoRA & LoRA fine-tuning**. The community's support in starring the repo is appreciated.

- **Fine-tuning and Model Discussions Rage On**: Debate centered around **Gemma 7b versus Mistral-7b** for domain-specific tasks with Unsloth AI fixing bugs in Gemma. Additionally, [Mixtral's branch support](https://github.com/unslothai/unsloth/pull/145), and resources like [Uiverse.io](https://uiverse.io/elements) for open-source UI elements in CSS or Tailwind, were shared.

- **Fine-tuning Trials and Tribulations**: Users grappled with issues in fine-tuning models such as **Mistral 7b on Colab** and differencing between **LoRA and QLoRA**. Difficulties included errors saving models to platforms like Hugging Face and deployment inquiries about models not supported by Unsloth, such as **OpenAI's GPT-4**.

- **Epochs vs Knowledge & Model Integration Insights**: Engagement on the number of epochs necessary for effective model training, with consensus unclear on the benefits of longer training for LLMs like **Tiny Mistral**. Discussions included findings on configuration settings ([Tiny Mistral model on Hugging Face](https://huggingface.co/Dans-DiscountModels/TinyMistral-v2.5-MiniPile-Guidelines-E1)), where suggestions were made for ideal rank and alpha values to handle large datasets.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Eager Engineers Eye New Model Integrations**: Users are eagerly discussing upcoming integrations like **Command-R model** with **LM Studio** post-Pull Request #6033 merger. Meanwhile, **Grok**, a colossal 314B model, tickles the fancy of many but seems unfeasible for local runs due to immense resource requirements.

- **Hardware Hustle**: There's a hot buzz around optimizing hardware. Folks with **RTX 2070 Super** and **1660 Super GPUs** seek the best models for their rigs, while others consider buying a **Tesla K80** for **€150** and cooling it with 3D printed shrouds. The DIY crowd debates over building **EPYC systems**, weighing the benefits of ample PCIe lanes against the costs.

- **Model Management Mysteries**: Model compatibility sparked confusion in the community. It was clarified that **llama.cpp** isn't fully aboard with **Cohere's Command-R model** yet, requiring updates past Mar 16, 2024, and release [b2440](https://github.com/ggerganov/llama.cpp/releases/tag/b2440) to function with **GGUF files**. It's good news, though, for **AVX beta users**, as established models like **Mistral** should cruise fine, albeit not the latest shinies like **starcoder2 or gemma**.

- **ROCm Riddle**: The query bat-signal is up for fellow **ROCm users** as someone seeks kinship in this niche, while another flagged that **AMD's Radeon 6700 XT** doesn't groove with **ROCm**, and LM Studio currently limits itself to the primary GPU.

- **Plugin Pursuits and Configuration Quests**: LM Studio aficionados are digging through [lmstudio-ai/configs](https://github.com/lmstudio-ai/configs) for the holy grail of model presets, while another intrepid soul seeks guidance on engaging models with JSON function calling on **Local Inference Servers**.

- **AI Agent Aspirations**: A single communiqué reveals a longing for the right agent system to consummate a creative concept, suggesting a mix of curiosity and algorithmic artistry bubbling beneath the surface.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **NVIDIA Accelerates Into the Future**: The upcoming [NVIDIA GeForce RTX 50-series "Blackwell"](https://www.techpowerup.com/320185/nvidia-geforce-rtx-50-series-blackwell-to-use-28-gbps-gddr7-memory-speed) is set to feature GDDR7 memory at 28 Gbps, promising a whopping 55% higher bandwidth compared to GDDR6.

- **AI's "Horny Claudes" Stir Debate**: A chatbot experiment claiming that "horny claudes" enhance AI outputs led to heated discussions, with comparisons drawn to "reverse Sydney," while others focused on substantive topics like Apple's AI work and the novel ORPO algorithm on [Hugging Face](https://huggingface.co/papers/2403.07691).

- **Grok-1's Release Stirs Debate**: The AI community reacts to Grok-1, a 314 billion parameter model, with some skepticism about its real-world performance. Simultaneously, there's intrigue about GPT-4's rumored 1.8 trillion parameters after a potential slip by NVIDIA's CEO during a GTC keynote, which can be viewed on [YouTube](https://www.youtube.com/watch?v=Y2F8yisiS6E).

- **Perplexity Plagues Llama-2 Users**: Experimentation to calculate perplexity for Llama-2 using a [Kaggle notebook](https://www.kaggle.com/code/philculliton/calculating-the-perplexity-of-4-bit-llama-2/notebook) caused confusion, while discussions around scaling up and down LLMs like Mistral and Llama took center stage, with a focus on financial and technical viability.

- **RAG-Tag Team Goes Deep**: Discourse within the RAG (retrieval-augmented generation) community delves into optimizing RAG model properties, designing a specialized "RAG mode," and leverages **Cohere's** model robustness, all aimed at improving functionality like context handling and varied output structuring. Additionally, the scripting approach in Python was shared on [GitHub](https://github.com/EveryOneIsGross/scratchTHOUGHTS/blob/main/commanDUH.py).



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Ivy League Secrets for Free!**: Engineers discussed the value of accessing freely available Ivy League courses, such as lectures from MIT and Stanford, and shared a professor's webpage from Carnegie Mellon University showcasing contributions to algorithms and machine learning. The page with nearly 7 years of content can be found [here](https://www.cs.cmu.edu/~dwoodruf/).

- **Pushing the Limits of AI Reasoning with KPUs**: An [AI acceleration project](https://github.com/trevorpogue/algebraic-nnhw) focusing on matrix multiplication and Maisa's **Knowledge Processing Unit** ([Maisa KPU](https://maisa.ai/blog/kpu)) aiming to advance Large Language Model (LLM) task complexity was a prominent share. The discussion also broached the concept of 'ThoughtStream' tokens and pause tokens in LLMs to enhance reasoning and inference.

- **Grok's Groans and GPU Grandstanding**: Amid debates over model performance and benchmarking, xAI's **Grok**, a **314-billion parameter Mixture-of-Experts model**, faced scrutiny regarding its real-world utility, with skepticism about speculative sampling in different architectures like **Mamba**. There was notable criticism of current LLM benchmarking systems, especially regarding the questionable reproducibility of openLLM leaderboard results for models like **Llama2-70b**.

- **Scale Matters: Implications of Data Complexity on Scaling Laws**: Members discussed how language model scaling laws vary with data complexity and the impact of syntactic properties and compressibility on scaling properties. Using compression metrics, like gzip, could potentially inform the creation of optimal training data mixtures by identifying datasets with favorable lexical densities.

- **Bigram Beginnings and N-Gram Nuances**: In the realm of n-gram statistics, the community exchanged insights on how to sample strings with specified n-gram statistics autoregressively. A script [generate_bigrams.py](https://github.com/EleutherAI/features-across-time/blob/main/scripts/generate_bigrams.py), aiding in the process, suggests that higher-order n-gram specifications inherently decide lower-order statistics.

- **Llama on Gaudi2 and the Quest for Harness Updates**: Users shared experiences with implementing functions for **Llama** on **Gaudi2** using **lm-eval-harness**, faced model selection issues in the tool and discussed the newly released version 0.4.2, accessible [here](https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.2). The community scrutinized method differences like `loglikelihood_rolling` for perplexity evaluation in tasks such as `wikitext`.

- **To Shuffle or Not to Shuffle The Pile**: Discussions arose around the preparation of [the Pile](https://pile.eleuther.ai/) for training. It was clarified that while the original files weren't shuffled, the pretokenized data on Hugging Face is ready-to-use, having been employed successfully by Pythia, and the train/test/val split in the original Pile might be shuffled.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI Models Struggling with Playwright Code**: Some users are experiencing issues with **GPT-3.5 Turbo** failing to generate correct **Playwright** test code, suggesting the model may not be familiar with the latest library updates. There's advice to potentially use **GPT-4** and to consider breaking down tasks into chunks to improve performance.

- **The Great AI Refusal Uproar**: The community has noted an uptick in instances where models are refusing task completion, leading to discussions on strategies like meta-prompting. The concerns also touch on the current content policies, with a hope expressed that OpenAI will address and relax these.

- **Classification Headaches and Context Windows**: In prompt engineering discussions, advice is circulating about optimizing classification prompts for better recall and fewer false positives. Key suggestions include testing the impact of prompt position within the total context window and potentially moving to a more powerful model.

- **Anticipation for GPT-5 and GPT-4 Web Search Skills**: Users are eagerly inquiring about the release of **GPT-5** but without definitive answers. Meanwhile, there's curiosity on how to integrate **GPT-4**'s web search capabilities into the API, a feature admired for its enhanced conversational abilities.

- **Navigational Tips for OpenAI's API and Privacy Policies**: Concerns around the management of API keys and data privacy led users to the [OpenAI enterprise privacy policy](https://openai.com/enterprise-privacy) for details. Additionally, reports of **GPT** unresponsiveness prompted references to OpenAI's support at [help.openai.com](https://help.openai.com) for assistance.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Eager Techie in Search of NL2SQL Nirvana**: A member working on a **NL2SQL pipeline** expressed concerns about accuracy using BAAI/llm-embedder and TheBloke/nsql-llama-2-7B-GGUF with FAISS and sought advice for more precise models and embeddings.

- **Hugging Face Huddle**: Enthusiasm was shown for new **Hugging Face** initiatives, including a model and data leaderboard. There were also discussions about the platform's capacity, pragmatic usage guidelines for newcomers, and links to learning resources like the [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/chapter1/1).

- **Marvel at NVIDIA's Computing Might**: The **Nvidia GH200 Grace Hopper Superchip** was a topic of excitement, symbolizing advancements in computational efficacy; however, further technical specifics were not discussed.

- **Pioneering Medusa's Parallel Token Predictions**: Interest peaked with [Medusa](https://arxiv.org/abs/2401.10774), an innovative parallel token prediction method aimed at enhancing Large Language Model (LLM) inference, positing a break from limited sequential token generation.

- **AI's Quiet Steps into Scientific Peer Review**: A study shed light on potential LLM modifications in peer review texts, finding that certain review characteristics might correlate with AI-altered content. This presents an intriguing cross-discipline concern about LLMs altering scientific discussions ([study link](https://arxiv.org/abs/2403.07183)).



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**New Tricks for Old Dogs**: Interactive techniques to treat documents as dynamic entities in the **Retrieval-Augmented Generation (RAG)** pipeline are proposed, potentially improving RAG performance through more sophisticated interactions. The discussion included a [step-by-step guide](https://youtu.be/w7Ap6gZFXl0) covering effective RAG implementation with tools like **LlamaParse** and **Qdrant**.

**LlamaIndex 0.10.20 Instrumental for Engineers**: The release of **LlamaIndex v0.10.20**, with its new **Instrumentation module**, offers enhanced observability features and API call monitoring, illustrated in shared notebooks. The release announcement and resources can be found through their [Twitter update](https://twitter.com/llama_index/status/1768730443921396220).

**Search Safari**: A novel method termed **Search-in-the-Chain**, integrating retrieval and planning for ultimate question answering prowess, is showcased - possibly revolutionizing real-time adjustment abilities in QA pipelines. A paper on the matter was highlighted, and community interest seemed piqued by the [tweet](https://twitter.com/llama_index/status/1769035278063399208).

**Resume Routing Revolution**: A blog post demonstrates a new model that marries **LlamaParse** and **LlamaIndex** to facilitate efficient job matching, parsing complex CV formats with relative ease. Kyosuke Morita's post on the subject is findable in this [Twitter thread](https://twitter.com/llama_index/status/1769147791002264008).

**Agentic Memory Architecture Arrives**: The advent of **MemGPT**, an architecture designed to enhance memory functions of AI agents, seems to promise significant improvements to assistant APIs, focusing on reliable memory operations. Engineers are directed to a [webinar tweet](https://twitter.com/llama_index/status/1769408792633229455) for more enlightenment.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Yann LeCun's Inner Monologue Dilemma**: [Yann LeCun's controversial take on LLMs](https://x.com/kk_slider_k_/status/1768464173657158132?s=46&t=6FDPaNxZcbSsELal6Sv7Ug) ignited debates on whether language primarily aids reasoning or if visuospatial processing is more fundamental. The concept of 'wordcels' paralleled 'shape rotators,' stemming from the revelation that LeCun allegedly does not have an inner monologue.

- **Speculations Stir for GPT-5**: There's anticipation for GPT-5's potential leaps in capabilities, fueled by Sam Altman's hints at major improvements and OpenAI's presumed developmental vanguard. Discussions included expectations of "trillion parameter chatbots" tied to Nvidia's GTC event and the potential of quantum jumps in LLM progress.

- **Grok-1's Pi Day Surprise**: The tech crowd reacted to xAI's release of *Grok-1*, a 314B parameter MoE model dropped on Pi Day, which sparked evaluations of its capabilities against other top-tier LLMs. Conversations ranged from performance and possible motivations for the open-source release to jokes about its size and parallel computational strategies.

- **Lex's Lackluster OpenAI Interview**: Sam Altman's appearance on the [Lex Fridman Podcast](https://youtu.be/jvqFAi7vkBc?si=WTfgLyNfGhkP2Azx) left the community craving more substantial takeaways. Dialogues noted the absence of deeper insights into OpenAI's strategies and Ilya Sutskever's involvement, with playful banter thrown in about Lex's podcast style.

- **Penetrating the Transformer Paradigm**: The **Paper Club** session provided valuable insights into the allure of transformers; their attention mechanisms solve encoding constraints of past models and allow in-train parallel processing, clarifying doubts about computational efficiency within LLMs. A forthcoming blog post was hinted at, promising a detailed recap.
  
- **1990s Hip-Hop AI Throws a Reflective Beat**: An AI by [Suno](https://app.suno.ai/song/83680b6f-db37-44de-adf9-3f7fff6b79d9) put together a song with vibes of '90s hip-hop, pondering AI's challenging role in creativity and spurring discussions on the bounds of machine-generated artistry.

- **AI-in-Action: A Guild Unites**: An informative and varied conversation included a member's sneak peek at a detailed blog article, a sharing of resources on [advanced RAG techniques](https://towardsdatascience.com/advanced-rag-01-small-to-big-retrieval-172181b396d4), and a citation of a [collaborative learning document](https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0) for the club.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **DALL-E 3 Dataset Makes a Move**: The **DALL-E 3** dataset was relocated, it was not removed as previously thought, and engineers can now access it at its new [Hugging Face repository](https://huggingface.co/datasets/OpenDatasets/dalle-3-dataset).

- **Commit to Your Datasets**: Huggingface datasets can be loaded using specific commit ids, enhancing the reproducibility of AI experiments; this functionality is documented on [Hugging Face's dataset loading guide](https://huggingface.co/docs/datasets/en/loading#hugging).

- **Grokking the Grok Model**: **Grok**, a 314B-parameter model by xai-org, is center stage in performance discussions, with engineers contrasting it with the smaller **Mixtral**; the GitHub repository for **Grok-1** can be found [here](https://github.com/xai-org/grok-1).

- **Enhanced Captioning with Cog**: Metadata is being used to improve caption accuracy in the **Cog model**, with some users sharing their strategies and scripts, one of which is available on [GitHub](https://github.com/victorchall/EveryDream2trainer/blob/main/caption_cog.py).

- **GPT-4 Architecture Speculation**: There's buzz around **GPT-4's potential architecture**, with leaks suggesting a 1.8 trillion parameter MoE model, but no confirmation yet; the speculation can be further explored through this [tweet image](https://pbs.twimg.com/media/GI-reRIW0AAZpMC?format=jpg&name=large).



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**Axolotl Stepping Up the Model Optimization Game**: Axolotl devs have introduced **ScatterMoE**, an optimization aimed at boosting Huggingface throughput, and users are directed to its [GitHub branch](https://github.com/OpenAccess-AI-Collective/axolotl/tree/scatter_moe) for more details. Upgrading to **PyTorch 2.2** or above is necessary for compatibility, with some already on PyTorch **2.2.1**.

**Groking Grok's Gargantuan Size**: The release of **Grok-1** model weights with **314 Billion parameters** was a topic of discussion, with a member commenting on suboptimal performance and the resource-intensiveness of running it. While only the **int8 version** is released, there's speculation about managing it using Axolotl's **qLoRA FSDP**, per the [Grok GitHub page](https://github.com/xai-org/grok-1).

**NVIDIA's Hardware Hype Hits New Heights**: Expected around 2025, NVIDIA's RTX 5000 series could bring a 50% VRAM increase and 78% bandwidth boost; specifics can be found in articles from [Heise](https://www.heise.de/news/GeForce-RTX-5000-Geruechte-zu-Nvidias-naechster-Grafikkartengeneration-9655220.html) and [TechPowerUp](https://www.techpowerup.com/320185/nvidia-geforce-rtx-50-series-blackwell-to-use-28-gbps-gddr7-memory-speed).

**Model Training and Conversion Conundrums**: Tokenizer issues were noticed when using `<summary>` tags, leading one to discover tokenization inconsistencies. Another user struggled with local model and data setups, leading to `HFValidationError` challenges. Conversational data fine-tuning errors were resolved with reference to Axolotl's readme, addressing empty dataset "role" arrays by mapping additional roles and excluding short conversations.

**Datasets Dialogue Drives Discovery**: A user showed interest in a **Mistral** model fine-tuned on math and coding datasets, with a suggestion floated about utilizing merging tactics such as **mergekit** to handle extensive data without individual training. The compatibility of different model chat formats during merging was also questioned, but not conclusively addressed.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Photonic Future Shines Bright**: Members discussed advancements in **photonic computer chips**, sharing a [video on a breakthrough](https://youtu.be/8ohh0cdgm_Y) and suggesting further learning through two [Asianometry educational videos](https://www.youtube.com/watch?v=29aTqLvRia8) on silicon photonics and light meshes. NVIDIA's CEO also hinted at AI's future at **GTC 2024**, discussing a new **Sota model with 1.8 trillion parameters** and **B100 hardware with 192GB HBM**.
  
- **Triton Gains New Puzzle and Visualizer**: A new **[visualizer for Triton](https://colab.research.google.com/drive/1AJc8RFsDeJ3Vx3gRq5dUqmcb-Cy1G8qh?usp=sharing)** was announced to help with debugging complex functions, alongside a set of **Triton Puzzles** for educational GPU problem-solving, despite having some bugs like occasional double visualizations and segmentation faults.

- **CUDA Enthusiasts Tackle Memory and Efficiency**: CUDA-focused discussions were lacking in consensus on warp scheduler engagement and defining active warps but provided deeper insights into memory management strategies such as "Producer Provides" and "Consumer Takes," and applying these tactics for pipeline parallel implementations for LLM inference was of keen interest.

- **ML Systems Converge with Hardware**: Resources like [Prof. Mohamed Abdelfattah's research group channel](https://www.youtube.com/@mabdelfattah88) and the [ECE 5545 (CS 5775) course page](https://abdelfattah-class.github.io/ece5545/) were suggested for exploring the nexus between ML and hardware optimization. The community actively engaged in discussions about **ring-attention** and **flash-attention implementations**, addressing memory scaling issues with links to research and GitHub repositories.

- **CUDA and ML Knowledge Exchange**: A member's background in CUDA with expertise in areas like **memory coalescing** and **warp divergence** was deemed a good foundation for ML, highlighting resources such as the **"Programming Massively Parallel Processors"** book and the **Zero to Hero ML series by Andrej Karpathy**. A debate about the sharing of exercise answers from the **[Programming Massively Parallel Processors](https://www.amazon.de/-/en/Wen-mei-W-Hwu/dp/0323912311)** book called for clarity on educational sharing ethics.

- **Offbeat Exchanges and GTC Anticipation**: The community shared a poetic note on the MLSys 2024's tagline, humor around smartphone issues, and clarified the order of operations in math. GTC meetups were coordinated among members with one expressing regret over not being able to attend.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Llama Formatting Gets the Green Light**: The "system," "user," and "assistant" format has been approved for use with llama models, supporting structured conversations.
- **Monetizing Models with Topups**: For payment queries, it was clarified that users need to *top up their balance* rather than directly connecting a credit card, affecting ways to monetize interactions with models.
- **Sonnet Wins the Roleplay Game**: **Sonnet** emerged as the model of choice among users for consistent roleplay experiences, outperforming others in maintaining narrative without repetition or irrelevant output.
- **Navigating Prompt Prowess**: Discussion on guiding Large Language Models (LLMs) revealed that typically only the first system message is used as a prompt, and subsequent instructions may need to be embedded within user messages.
- **API Development and Model Marketplaces**: Conversations touched on various technical points such as the integration of public APIs, being listed on platforms, and affiliate programs, while considering the cost, efficiency, and the OpenRouter API's flexibility for models like **Sonnet**.

Relevant links of interest from these discussions include [OpenRouter](https://openrouter.ai) and xai-org's [Grok open release on GitHub](https://github.com/xai-org/grok-1).



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**API Confusion in LangChain Land**: Members debated the merits of **LangChain's astream_log** versus **astream_events**, with concerns about the latter being in beta and potential deprecation. However, there was no clear consensus on whether one API is favoured over the other or if they are meant to serve distinct purposes.

**Community to the Documentation Rescue**: The call for clarifications and contributions to **LangChain documentation** resonated, as users faced challenges with navigation and found the materials somewhat lacking, particularly for newcomers to the platform.

**Rubik’s AI Assembles Its Beta Testing Squad**: An invitation for beta testing a robust research assistant called **Rubik's AI** was issued, promising access to high-powered models like **Claude 3 Opus**, **GPT-4 Turbo**, and **Mistral Large**. Keen participants are directed to their [waitlist](https://rubiks.ai/).

**LangChain AI Showcase**: From **AI chatbots for data analysis** to **living bookmarks** and **personalized nutrition apps**, members shared their LangChain-powered innovations with the community. Projects demonstrated integration of advanced features with repositories available on GitHub and demonstrations via [YouTube](https://youtu.be/vHjc5CEoIJE).

**Streaming Stuck in Static**: Technical issues arose with LangChain's `RemoteRunnable` when attempting to stream outputs in JavaScript, which diverts to an `/invoke` call rather than the expected `/stream`. The matter appears complex, with no recent documentation or changes addressing the JavaScript-specific streaming conundrum.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **APIs May Spill the Beans on LLMs**: Researchers discovered that APIs can inadvertently reveal information about proprietary large language models (LLMs), including architecture details, with less than $1,000 spent experimenting on OpenAI's GPT-3.5-turbo. Concerns grew over underestimated LLM sizes, with skepticism over a 7 billion parameter estimate and suggestions that Mixture of Experts (MoE) architecture might inflate the true size.
  
- **Vexation Over Vague Open Source Definitions**: There's brewing contention over the definition of open source, with @rasbt's tweet foreshadowing possible discord within the OSS community. Members acknowledge the need for a clear consensus on what qualifies as open source, considering the range of licenses from *Apache 2.0* to *GPLv3*, and there's a move to create a *practical definition* to mitigate potential disputes.

- **Behemoth Among Us: Grok-1 Debuts**: xAI's announcement of unveiling Grok-1, a colossal 314 billion parameter Mixture-of-Experts model released under Apache 2.0, has sent ripples across the community. Grok-1's performance metrics suggest it could overshadow competing models like Falcon, and its unorthodox torrent distribution method has spurred debate on policy and reputation issues in open-sourced AI.

- **Data Size Speculations and Chinchilla Relations**: With Grok-1's impressive performance, the community speculates on the size of its training dataset and ponders how findings from Chinchilla research may relate to MoE models, reflecting on the trade-offs between data scale and model optimality.

- **Model Delivery Humor Hits Bandwidth Woes**: Amidst discussions about Grok-1, a jest about shipping AI models physically to sidestep cloud egress fees highlights the real challenges and costs associated with transferring massive amounts of data.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **Airbus Ambiguity in Alignment Lab**: A shared [tweet about Airbus from Alignment Lab](https://twitter.com/alignment_lab/status/1758949148143841379) sparked confusion among members, with one seeking clarification on what exactly is being built.
- **Search Party for HTTP Savvy Embeddings**: A query has been raised about the existence of an **embeddings model trained** specifically on HTTP responses, with the suggestion that a properly trained transformer might fulfill this role.
- **Double-Dose Mistral Model Missing**: Questions emerged on whether a **Mistral model** fine-tuned on both **orca-math-word-problems-200k dataset** and **nvidia/OpenMathInstruct-1** exists, indicating a gap in accessible combined training.
- **Grok 1's Fine-Tuning Rally**: There's a call for collaboration on fine-tuning **Grok 1**, with emphasis on the hefty computational resources and expertise needed, such as **64-128 H100 GPUs**, while highlighting an existing **MoE training infrastructure** with impressive efficiency.
- **Grok 1: A Gem or Just Glass?**: Skepticism surrounding **Grok 1's performance** has surfaced, but some members noted its impressive capabilities, referencing performance on a [Hungarian national high school finals exam dataset](https://huggingface.co/datasets/keirp/hungarian_national_hs_finals_exam) comparable to GPT-4 and Claude.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **AI's Controlled Opposition or Genuine Fear?**: A [tweet](https://x.com/tszzl/status/1768530219378631137?s=20) shared in the guild sparked a debate about **Anthropic's** true motives, with implications it might be acting as controlled opposition to put the fear of God into technical staff.
- **Struggling AI Content Moderations**: Guild members note that content moderation systems are failing to effectively moderate images with people, raising concerns over the reliability of these algorithms in practical applications.
- **Scaling Woes with Claude Sonnet**: **Claude Sonnet's** scalability was questioned for use in projects with several dozen million tokens per month, with guild members seeking input on the model's performance at high volumes.
- **KPU: Breakthrough or Hype**?: Maisa's new **Knowledge Processing Unit (KPU)** has been described in a [blog post and white paper](https://maisa.ai/blog/kpu), raising discussion about its true potential and comparisons to current models like GPT-4. Members highlighted the importance of including **GPT-4-turbo** in any direct comparisons and expressed skepticism without more substantial evidence.
- **Skeptical Engineers Poke Fun at AI Startup Trends**: With Maisa's KPU introduction, guild members jest about the typical AI startup pattern of hyping technology with impressive graphs and waitlists, while also critically pondering the practical drawbacks, such as potential latency issues. Further clarity came from a [tweet by @davipar](https://x.com/davipar/status/1768683151780683919?s=20) explaining KPU's capability to work with LLMs.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **DiscoLM's German Fluster**: Users report that **DiscoLM-mixtral-8x7b-v2** struggles with German language generation post fine-tuning, and **LeoLM** models display inconsistency. API issues also arose when **DiscoLM** returned German responses to English prompts, and a **ValueError** occurred during fine-tuning for a classification task.

- **SOS for Server Shift**: The demo's server move from a home kitchen to a professional setting led to unexpected networking problems; efforts to resolve the issue will commence next week. Meanwhile, members appreciated guidance on the model's training and prompt adherence, with a humorous nod to the quirky reliability of hobbyist setups.

- **Benchmark Blues & Collaboration Calls**: Discord chats reveal concerns over benchmarks for German language models, with varying performance linked to templates and end token conventions. A call to collaborate on better benchmarks and quality datasets echoes through the discussions, alongside a playful suggestion to involve academia or secure benchmarks through private channels.

- **Github and More**: Members shared GitHub links like [grok model code](https://github.com/xai-org/grok/blob/main/model.py) and various benchmarks such as [SuperGLEBer](https://www.informatik.uni-wuerzburg.de/datascience/news/single/news/our-paper-supergleber-german-language-understanding-evaluation-benchmark-was-accepted-at-the-naacl-2024/) and [XTREME](https://github.com/google-research/xtreme). Reddit threads also surfaced as part of the discourse on the search for optimal German language models.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Explosion Makes Prompt Engineering a Breeze**: Engineers highlighted [Prodigy's prompt engineering tools](https://prodi.gy/features/prompt-engineering) from Explosion, noting the advancement of turning prompt engineering into a data annotation task for improved precision.

- **Cross-Model Prompt Testing with PromptTools**: The open-source resource, [PromptTools](https://github.com/hegelai/prompttools), was brought up for its utility in prompt testing and experimentation across different LLMs and vector databases, despite missing version management features.

- **Helicone Jumps into Prompt Management**: [Helicone](https://www.helicone.ai/), lauded for its generative AI application-building capabilities, is gaining traction for incorporating prompt management tools, versioning, and analysis capabilities for a more integrated AI development experience.

- **PromptFoo Joins the CI/CD League**: [PromptFoo](https://github.com/promptfoo/promptfoo) received attention for its features that allow users to test and compare LLM outputs, manage prompt quality, and integrate with CI/CD pipelines, supporting models from various platforms including OpenAI and Azure GPT.

- **Personalized Translation Tailors Reader Experience**: One engineer shared their experiment with using gpt-3.5-turbo to personalize blog post translations, attempting to tailor content to different personas for better understanding and engagement, showcased at [How to Build a Buzzword](https://www.dbreunig.com/2020/02/28/how-to-build-a-buzzword.html).

*Unfortunately, the query about recovering the seed used by the OpenAI models for a previous API request did not contain sufficient detail to warrant inclusion in this summary.*



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **Model Accuracy Breakthrough Incoming**: A member is preparing an article on a new method that improves the **global accuracy** of AI models and enhances sample efficiency, with plans to share it after refining the results and visuals.
- **Seeking Supercomputing Superheroes**: The same member seeks resources to **test their method on larger AI models**, having demonstrated a significant test accuracy improvement from 0.04 to 0.1 in one epoch with VGG16 on CIFAR100.
- **Resource Rally to the Rescue**: An offer of **compute power and resources** was made to assist in the validation and testing phase of the novel method.
- **Recruiting for Quiet-STaR**: There's an open call for individuals with knowledge of **PyTorch and transformers** to contribute to the "Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking" project.
- **Off-topic Oddities**: A single message in an unrelated channel linked to a [YouTube video](https://www.youtube.com/watch?v=ZlJbaYQ2hm4), context or relevance to technical discussions not provided.



---

# PART 2: Detailed by-Channel summaries and links



**Stability.ai (Stable Diffusion) ▷ #[announcements](https://discord.com/channels/1002292111942635562/1002292398703001601/1219396298176991303)** (1 messages): 

- **Stable Video Goes 3D**: [Stable Video 3D](https://stability.ai/news/introducing-stable-video-3d) has been released, based on the existing **Stable Video Diffusion** technology. It takes a single object image as input and outputs novel multi-views, allowing for the generation of 3D meshes.
- **Taking 3D Tech a Notch Higher**: The **Stable Video 3D** model boasts greatly improved quality and multi-view capabilities over **Stable Zero123**. It also outperforms open source alternatives like [Zero123-XL](https://objaverse.allenai.org/docs/zero123-xl/).
- **Announcing Two Variants**: The release includes two variants: `SV3D_u`, generating orbital videos from single images without camera conditioning, and `SV3D_p`, which extends the capabilities allowing for more complex video generation.

**Link mentioned**: <a href="https://stability.ai/news/introducing-stable-video-3d">Introducing Stable Video 3D: Quality Novel View Synthesis and 3D Generation from Single Images &mdash; Stability AI</a>: When we released Stable Video Diffusion, we highlighted the versatility of our video model across various applications. Building upon this foundation, we are excited to release Stable Video 3D. This n...

  

---


**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1218109086101540905)** (988 messages🔥🔥🔥): 

- **Stable Diffusion Cascade vs SDXL**: A user shared their experience that Stable Cascade is better at executing complex prompts compared to SDXL, though it runs slower on their hardware. Cascade takes roughly 90 seconds per generation compared to SDXL's 30 seconds.

- **Concerns Over Open Source and Crypto**: There was a discussion expressing disappointment that Stability AI is partnering with various blockchain companies. Users speculated about the implications for SD3’s future and the potential shift towards proprietary models post-SD3 release.

- **Anticipation for SD3**: Users are eagerly awaiting the public release of Stable Diffusion 3, with invites to the beta expected to go out soon. There's speculation that SD3 will provide quality comparable to other tools while being more efficient to run.

- **Converting .pt to Safetensors**: A user inquired about an alternative method for converting PyTorch files (.pt) to safetensors without coding complexities. Another user mentioned most UIs don't execute scripts from .pt files, minimizing security concerns, but no alternative tool was provided.

- **Stable Video 3D Announcement**: Stability AI announced the release of Stable Video 3D (SV3D), a model that can create 3D meshes from a single image input. The announcement highlighted the improvement over previous models like Stable Zero123 and its versatility for creating orbital videos and accommodating pose conditioning.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://academictorrents.com/details/5f96d43576e3d386c9ba65b883210a393b68210e">grok-1</a>: Grok-1 is a 314B parameter Mixture of Experts model - Base model (not finetuned) - 8 experts (2 active) - 86B active parameters - Apache 2.0 license - Code:  - Happy coding! p.s. we re hiring: </li><li><a href="https://tenor.com/view/iron-man-mr-clean-mop-ai-floors-gif-27596354">Iron Man Mr Clean GIF - Iron Man Mr Clean Mop - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/coqui/XTTS-v2">coqui/XTTS-v2 · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/yess-yes-gif-25420589">Yess GIF - Yess Yes - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://stability.ai/news/introducing-stable-video-3d">Introducing Stable Video 3D: Quality Novel View Synthesis and 3D Generation from Single Images &mdash; Stability AI</a>: When we released Stable Video Diffusion, we highlighted the versatility of our video model across various applications. Building upon this foundation, we are excited to release Stable Video 3D. This n...</li><li><a href="https://tenor.com/view/avatar-cuddle-hungry-yummy-food-gif-5610436">Avatar Cuddle GIF - Avatar Cuddle Hungry - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://thedailywtf.com/articles/The_Complicator_0x27_s_Gloves">The Complicator&#39;s Gloves</a>: Good software is constantly under attack on several fronts. First, there are The Amateurs who somehow manage to land that hefty contract despite having only finished &quot;Programming for Dummies&quot...</li><li><a href="https://docs.python.org/3/library/pickle.html">pickle — Python object serialization</a>: Source code: Lib/pickle.py The pickle module implements binary protocols for serializing and de-serializing a Python object structure. “Pickling” is the process whereby a Python object hierarchy is...</li><li><a href="https://civitai.com/models/351450/proteus-rundiffusion?dialog=commentThread&commentId=372974">Proteus-RunDiffusion - withoutclip | Stable Diffusion Checkpoint | Civitai</a>: Introducing Proteus-RunDiffusion In the development of Proteus-RunDiffusion, our team embarked on an exploratory project aimed at advancing the cap...</li><li><a href="https://www.youtube.com/watch?v=fibDNwF8bjs">WKUK - Anarchy [HD]</a>: Economic ignorance at its most comical.— &quot;Freedom, Inequality, Primitivism, and the Division of Labor&quot; by Murray Rothbard (http://mises.org/daily/3009).— &quot;Th...</li><li><a href="https://www.pny.com/professional/software-solutions/about-nvidia-gpus/nvlink">NVLink | pny.com</a>: no description found</li><li><a href="https://www.pny.com/professional/software-so">Page Not Found | pny.com</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=YTE0OTVOnZU">Vancouver, Canada 1907 (New Version) in Color [VFX,60fps, Remastered] w/sound design added</a>: I colorized , restored and I added a sky visual effect and created a sound design for this video of Vancouver, Canada 1907, Filmed from the streetcar, these ...</li><li><a href="https://youtu.be/m9jg1fdOiVY?t=412">Install ComfyUI on Mac OS (M1, M2 or M3)</a>: This video is a quick wakthrough to show how to get Comfy UI installed locally on your m1 or m2 mac. Find out more about AI Animation, and register as an AI ...</li><li><a href="https://www.youtube.com/watch?v=5mIWo6dgTmI&ab_channel=Megaprojects">The Mushroom Motherboard: The Crazy Fungal Computers that Might Change Everything</a>: Unlock the secrets of fungal computing! Discover the mind-boggling potential of fungi as living computers. From the wood-wide web to the Unconventional Compu...</li><li><a href="https://new.reddit.com/r/StableDiffusion/comments/1b6skvx/wheres_waldo_beach_scenes_as_an_animated_loop/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://civitai.com/models/207992/stable-video-diffusion-svd)">Stable Video Diffusion - SVD - img2vid-xt-1.1 | Stable Diffusion Checkpoint | Civitai</a>: Check out our quickstart Guide! https://education.civitai.com/quickstart-guide-to-stable-video-diffusion/ The base img2vid model was trained to gen...</li><li><a href="https://huggingface.co/PollyannaIn4D">PollyannaIn4D (Pollyanna)</a>: no description found</li><li><a href="https://youtu.be/ruANV24h0Dw?si=rVFKZqowCdpKTzgp">Короткометражный мультфильм &quot;Парк&quot; (сделан нейросетями)</a>: Короткометражный мультфильм &quot;Парк&quot; - невероятно увлекательный короткометражный мультфильм, созданный с использованием нейросетей.</li><li><a href="https://github.com/GraftingRayman/ComfyUI-Trajectory">GitHub - GraftingRayman/ComfyUI-Trajectory</a>: Contribute to GraftingRayman/ComfyUI-Trajectory development by creating an account on GitHub.</li><li><a href="https://github.com/DiffusionDalmation/pt_to_safetensors_converter_notebook#">GitHub - DiffusionDalmation/pt_to_safetensors_converter_notebook: This is a notebook for converting Stable Diffusion embeddings from .pt to safetensors format.</a>: This is a notebook for converting Stable Diffusion embeddings from .pt to safetensors format. - DiffusionDalmation/pt_to_safetensors_converter_notebook</li><li><a href="https://github.com/mix1009/sdwebuiapi">GitHub - mix1009/sdwebuiapi: Python API client for AUTOMATIC1111/stable-diffusion-webui</a>: Python API client for AUTOMATIC1111/stable-diffusion-webui - mix1009/sdwebuiapi</li><li><a href="https://github.com/chaojie/ComfyUI-DragAnything/tree/main">GitHub - chaojie/ComfyUI-DragAnything</a>: Contribute to chaojie/ComfyUI-DragAnything development by creating an account on GitHub.</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API)">Home</a>: Stable Diffusion web UI. Contribute to AUTOMATIC1111/stable-diffusion-webui development by creating an account on GitHub.</li><li><a href="https://github.com/Stability-AI/generative-models">GitHub - Stability-AI/generative-models: Generative Models by Stability AI</a>: Generative Models by Stability AI. Contribute to Stability-AI/generative-models development by creating an account on GitHub.</li><li><a href="https://stable-diffusion-art.com/regional-prompter/)">Regional Prompter: Control image composition in Stable Diffusion - Stable Diffusion Art</a>: Do you know you can specify the prompts for different regions of an image? You can do that on AUTOMATIC1111 with the Regional Prompter extension.
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1219057096780419163)** (1 messages): 

- **Unlimited Queries for Claude 3 Opus**: The announcement revealed that **Perplexity Pro users** now have **unlimited daily queries** on Claude 3 Opus, touted as the best LLM in the market today. Pro users can now enjoy extended access without query limits.
  

---


**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1218100055626743851)** (795 messages🔥🔥🔥): 

- **Confusion Over "Unlimited" Usage Terms**: Users are perplexed by Perplexity.ai's use of the term "unlimited," finding it misleading given the actual cap of 600 uses per day. The concern is that such wording could misrepresent the service and potentially lead to legal challenges.
- **Discussions on Educating Young Children**: A debate arose regarding the capability of children, particularly five-year-olds, to understand complex concepts. Some argue that children can grasp advanced topics when explained appropriately, often using analogies or AI assistance, while others express skepticism about their cognitive capacity.
- **User Experience of Pro Subscription**: There's interest and satisfaction among users about Perplexity's latest integration with Claude 3 Opus, eagerly discussing its efficiency and aptness for different tasks, including roleplay, with one user successfully testing its response to sensitive topics.
- **Family Use of AI**: Parents share how they use AI tools like ChatGPT Pro and Perplexity.ai to answer their children’s curiosities, indicating that kids come up with various insightful questions that AI can help address, fostering their inquisitive nature.
- **AI for Career and Comedy**: A user humorously claims to have utilized Perplexity's Claude 3 Opus to secure a job at McDonald's, which they later declined, leading to amusing banter about the various application of AI in everyday life and potential impacts on industries like fast food.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/shikimori-shikimoris-not-just-cute-shikimoris-not-just-a-cutie-anime-anime-an">no title found</a>: no description found</li><li><a href="https://x.com/AravSrinivas/status/1769475725965566167?s=20">Tweet from Aravind Srinivas (@AravSrinivas)</a>: We have made the number of daily queries on Claude 3 Opus (the best LLM in the market today) for Perplexity Pro users, unlimited! Enjoy!</li><li><a href="https://www.theverge.com/2024/3/18/24104626/apple-license-google-gemini-generative-ai-openai-chatgpt">Apple’s AI ambitions could include Google or OpenAI</a>: Another big Apple / Google deal could be on the horizon.</li><li><a href="https://tenor.com/view/shikimori-shikimoris-not-just-cute-shikimoris-not-just-a-cutie-anime-anime-anime-girl-gif-26002811">Shikimori Shikimoris Not Just Cute GIF - Shikimori Shikimoris Not Just Cute Shikimoris Not Just A Cutie Anime - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://us.nothing.tech/pages/perplexity">Nothing Perplexity Offer</a>: Here at Nothing, we’re building a world where tech is fun again. Remember a time where every new product made you excited? We’re bringing that back.</li><li><a href="https://x.com/AravSrinivas/status/1769485603622867394?s=20">Tweet from Aravind Srinivas (@AravSrinivas)</a>: Yep, thanks to @elonmusk and xAI team for open-sourcing the base model for Grok. We will fine-tune it for conversational search and optimize the inference, and bring it up for all Pro users!  ↘️ Quoti...</li><li><a href="https://x.com/technology/status/1769597406243360937?s=20">Tweet from Bloomberg Technology (@technology)</a>: EXCLUSIVE: Apple is in talks to build Google’s Gemini AI engine into the iPhone in a potential blockbuster deal https://trib.al/YMYJw2K</li><li><a href="https://fxtwitter.com/BrivaelLp/status/1769482175005577571?s=20">Tweet from Brivael (@BrivaelLp)</a>: Zuck just reacted to the release of Grok, and he is not really impressed.  &#34;314 billion parameter is too much. You need to have a bunch of H100, and I already buy them all&#34; 🤣</li><li><a href="https://youtu.be/OPoWMXqq62Q?si=jk-ZbhjfkZtRkjz7">What Are These Companies Hiding?</a>: Thoughts on the Rabbit R1 and Humane Ai PinIf you&#39;d like to support the channel, consider a Dave2D membership by clicking the “Join” button above!http://twit...</li><li><a href="https://youtube.com/clip/Ugkx9gPr2y53Be9C99y-EVVWfZPjRxNQo6FL?si=0r1zDbn2FfjmrsuB">✂️ Sam Altman on AI LLM Search</a>: 47 seconds · Clipped by Syntree · Original video &quot;Sam Altman: OpenAI, GPT-5, Sora, Board Saga, Elon Musk, Ilya, Power &amp; AGI | Lex Fridman Podcast #419&quot; by Le...</li><li><a href="https://fccid.io/2BFB4R1">FCC ID 2BFB4R1 AI Companion by Rabbit Inc.</a>: FCC ID application submitted by Rabbit Inc. for AI Companion for FCC ID 2BFB4R1. Approved Frequencies, User Manuals, Photos, and Wireless Reports.
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1218101595586429048)** (35 messages🔥): 

- **Creative Writing Exploration**: Claude 3 Opus takes on creative writing with a prompt about *“ever increasing intelligence until it's unintelligible to humans”*. The exploration can be found [here](https://www.perplexity.ai/search/increasing-intelligence-of-HLUn3nOzSx6Nc5ecNpe5pA).
- **Visibility Matters**: Users are reminded to **share their threads** for visibility so that others can view them. Instructions can be found [here](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825).
- **Comparative Research**: A user mentions utilizing Perplexity to compare phones in an innovative way, signaling practical applications of the platform. For specifics, the comparison can be observed [here](https://www.perplexity.ai/search/how-does-the-tzGt2.woRkCHJhNXY8V_Gg).
- **Public Announcement on Midjourney**: Midjourney's decisions regarding Stability AI are central to a recent discussion, indicating significant moves in the AI space. Details of the ban can be read [here](https://www.perplexity.ai/search/Midjourney-bans-Stability-nGDGZxh5SIucTa3mCbZz8w).
- **Search Reference for Clinical Studies**: A user shared a link to the MINDSET Study, possibly pointing out the importance of mental health research. Interested parties can find details about the study [here](https://www.perplexity.ai/search/MINDSET-Study-clinical-asL8eAZuQPmkJ_2hYCgIIw).
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1218160850670583828)** (64 messages🔥🔥): 

- **API Model Uncertainty**: Users discuss the continued functionality of a model, which was due to be deprecated on March 15 but is still operational, raising questions about whether plans have changed or the deprecation will occur later in the day.
- **Inconsistency in News Responses with Sonar**: There is a comparison between responses from the `sonar-medium-online` model and web browser versions, showing significant differences in the content provided about Donald Trump, with some users getting detailed responses and others not.
- **Getting Job Posting Links via API Troubles**: Users experimenting with the API to generate job search results notice that the API sometimes returns valid and working job posting links, but results can be inconsistent, varying between actual job listings and generic links to job sites.
- **Discussion on Max Tokens and Response Quality**: A user inquired about the effect of setting a low max tokens value on response quality, sparking a discussion on the API's behavior when given a limit, as well as the feasibility of the model generating comprehensive answers within the specified token constraints.
- **Interest in Grok Support Confirmation**: Users show interest in the possibility of Perplexity supporting Grok now that it's open source, with mentions that the company is planning to do so following a tweet by the CEO.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.perplexity.ai">pplx-api</a>: no description found</li><li><a href="https://perplexity.typeform.com/to/j50rnNiB">pplx-api form</a>: Turn data collection into an experience with Typeform. Create beautiful online forms, surveys, quizzes, and so much more. Try it for FREE.
</li>
</ul>

</div>
  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1218108656428650526)** (853 messages🔥🔥🔥): 

- **Impersonator Alert**: Users are warning about a scam account impersonating Unsloth developer Daniel Han (**starsupernova**) by sending friend requests. The impersonator's Discord username is **starsupernova0**, and the community is encouraged to report the account to Discord.
- **Grok-1 Brewing Open Source Stir**: Elon Musk's xAI team released Grok-1, a colossal 314B parameter model, which stirred up discussions on practicality and purpose due to its size and the impractical GPU resources needed for utilization.
- **Fine-tuning Assistance with New Tools**: Users shared resources for fine-tuning models, such as [Unsloth's repository](https://github.com/unslothai/unsloth) for reduced VRAM usage during training and an [integration of Unsloth with aikit](https://sozercan.github.io/aikit/) for finetuning via configuration and creating model images with Docker.
- **Optimizing Fine-tuning Practices**: Concerns about setting optimal hyperparameters for QLoRA and other fine-tuning strategies were discussed. Users are advised to refer to example notebooks for guidance on hyperparameters and the structure of datasets for training LLMs.
- **Technical Glitches with Online Platforms**: Members exchanged tips for overcoming issues with Kaggle environments and deploying AI models, stressing the need for a streamlined process and voicing frustration over intermittent problems.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lightning.ai/live-session/a35263e0-0428-40b6-8828-8e72773a284d">Lightning AI | Turn ideas into AI, Lightning fast</a>: The all-in-one platform for AI development. Code together. Prototype. Train. Scale. Serve. From your browser - with zero setup. From the creators of PyTorch Lightning.</li><li><a href="https://docs.anthropic.com/claude/page/cosmic-keystrokes">Cosmic keystrokes</a>: no description found</li><li><a href="https://huggingface.co/Crystalcareai/GemMoE-Beta-1">Crystalcareai/GemMoE-Beta-1 · Hugging Face</a>: no description found</li><li><a href="https://x.ai/blog/grok-os">Open Release of Grok-1</a>: no description found</li><li><a href="https://substack.recursal.ai/p/eaglex-17t-soaring-past-llama-7b">🦅 EagleX 1.7T : Soaring past LLaMA 7B 2T in both English and Multi-lang evals (RWKV-v5)</a>: A linear transformer has just cross the gold standard in transformer models, LLaMA 7B, with less tokens trained in both English and multi-lingual evals. A historical first.</li><li><a href="https://x.ai/about">About xAI</a>: no description found</li><li><a href="https://huggingface.co/xai-org/grok-1">xai-org/grok-1 · Hugging Face</a>: no description found</li><li><a href="https://x.ai/">Blog</a>: no description found</li><li><a href="https://arxiv.org/abs/2401.04088">Mixtral of Experts</a>: We introduce Mixtral 8x7B, a Sparse Mixture of Experts (SMoE) language model. Mixtral has the same architecture as Mistral 7B, with the difference that each layer is composed of 8 feedforward blocks (...</li><li><a href="https://x.ai/blog/grok">Announcing Grok</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://huggingface.co/spaces/HirCoir/Piper-TTS-Spanish">Piper TTS Spanish - a Hugging Face Space by HirCoir</a>: no description found</li><li><a href="https://unsloth.ai/blog/gemma-bugs">Unsloth Fixing Gemma bugs</a>: Unsloth fixing Google&#x27;s open-source language model Gemma.</li><li><a href="https://huggingface.co/damerajee/Llamoe-test">damerajee/Llamoe-test · Hugging Face</a>: no description found</li><li><a href="https://openhands.ai4bharat.org/en/latest/instructions/datasets.html#supported-datasets">ISLR Datasets &mdash; 👐OpenHands  documentation</a>: no description found</li><li><a href="https://x.com/UnslothAI/status/1768991010938404879">Tweet from Unsloth AI (@UnslothAI)</a>: Unsloth is trending on GitHub this week! 🙌🦥  Thanks to everyone & all the ⭐️Stargazers for the support!  Check out our repo: http://github.com/unslothai/unsloth</li><li><a href="https://sozercan.github.io/aikit/">Introduction | AIKit</a>: AIKit is a one-stop shop to quickly get started to host, deploy, build and fine-tune large language models (LLMs).</li><li><a href="https://huggingface.co/papers/2402.18668#65f0f5f8de069cd5c55f1dd2">Paper page - Simple linear attention language models balance the recall-throughput
  tradeoff</a>: no description found</li><li><a href="https://huggingface.co/Qwen/Qwen1.5-72B">Qwen/Qwen1.5-72B · Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/abs/2310.17680">CodeFusion: A Pre-trained Diffusion Model for Code Generation</a>: Imagine a developer who can only change their last line of code, how often would they have to start writing a function from scratch before it is correct? Auto-regressive models for code generation fro...</li><li><a href="https://www.youtube.com/watch?v=jvqFAi7vkBc">Sam Altman: OpenAI, GPT-5, Sora, Board Saga, Elon Musk, Ilya, Power &amp; AGI | Lex Fridman Podcast #419</a>: Sam Altman is the CEO of OpenAI, the company behind GPT-4, ChatGPT, Sora, and many other state-of-the-art AI technologies. Please support this podcast by che...</li><li><a href="https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-Tune-an-LLM-Part-1-Preparing-a-Dataset-for-Instruction-Tuning--Vmlldzo1NTcxNzE2">How to Fine-Tune an LLM Part 1: Preparing a Dataset for Instruction Tuning</a>: Learn how to fine-tune an LLM on an instruction dataset! We&#39;ll cover how to format the data and train a model like Llama2, Mistral, etc. is this minimal example in (almost) pure PyTorch.</li><li><a href="https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py">transformers/src/transformers/models/mixtral/modeling_mixtral.py at main · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers</li><li><a href="https://youtu.be/rANv5BVcR5k">Mistral Fine Tuning for Dummies (with 16k, 32k, 128k+ Context)</a>: Discover the secrets to effortlessly fine-tuning Language Models (LLMs) with your own data in our latest tutorial video. We dive into a cost-effective and su...</li><li><a href="https://github.com/xai-org/grok-1">GitHub - xai-org/grok-1: Grok open release</a>: Grok open release. Contribute to xai-org/grok-1 development by creating an account on GitHub.</li><li><a href="https://github.com/jiaweizzhao/GaLore?tab=readme-ov-file#install-galore-optimizer">GitHub - jiaweizzhao/GaLore</a>: Contribute to jiaweizzhao/GaLore development by creating an account on GitHub.</li><li><a href="https://github.com/AI4Bharat/OpenHands">GitHub - AI4Bharat/OpenHands: 👐OpenHands : Making Sign Language Recognition Accessible. | **NOTE:** No longer actively maintained. If you are interested to own this and take it forward, please raise an issue</a>: 👐OpenHands : Making Sign Language Recognition Accessible. | **NOTE:** No longer actively maintained. If you are interested to own this and take it forward, please raise an issue - AI4Bharat/OpenHands</li><li><a href="https://github.com/mistralai/mistral-src">GitHub - mistralai/mistral-src: Reference implementation of Mistral AI 7B v0.1 model.</a>: Reference implementation of Mistral AI 7B v0.1 model. - mistralai/mistral-src</li><li><a href="https://huggingface.co/datasets/teknium/GPT4-LLM-Cleaned">teknium/GPT4-LLM-Cleaned · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/argilla">argilla (Argilla)</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/xai-org/grok-1/issues/6#issuecomment-2002664859">Error when installing requirements · Issue #6 · xai-org/grok-1</a>: i have installed python 3.10 and venv. Trying to &quot;pip install -r requirements.txt&quot; ERROR: Ignored the following versions that require a different python version: 1.6.2 Requires-Python &gt;=3...</li><li><a href="https://the-decoder.com/falcon-180b-open-source-language-model-outperforms-gpt-3-5-and-llama-2/">Falcon 180B open-source language model outperforms GPT-3.5 and Llama 2</a>: The open-source language model FalconLM offers better performance than Meta&#039;s LLaMA and can also be used commercially. Commercial use is subject to royalties if revenues exceed $1 million.</li><li><a href="https://github.com/unslothai/unsloth/pull/97">Staging PR for implimenting Phi-2 support. by cm2435 · Pull Request #97 · unslothai/unsloth</a>: ….org/main/getting-started/tutorials/05-layer-norm.html]</li><li><a href="https://github.com/huggingface/transformers/pull/29588">FEAT / Optim: Add GaLore optimizer by younesbelkada · Pull Request #29588 · huggingface/transformers</a>: What does this PR do? As per title, adds the GaLore optimizer from https://github.com/jiaweizzhao/GaLore Fixes: #29512 This is how I am currently testing the API: import torch import datasets from ...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1218580567453470860)** (1 messages): 

- **Unsloth AI Stars on the Rise**: The **Unsloth AI** repository is trending on GitHub this week, with a nod of thanks to the community and stargazers. The team encourages more people to star the repo on [GitHub](https://github.com/unslothai/unsloth), which promotes faster and more memory-efficient QLoRA & LoRA finetuning.

**Link mentioned**: <a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth

  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1218112720994308122)** (25 messages🔥): 

- **The Coincidence Tsar**: A user discussed the phenomenon of thinking about something, only to encounter it shortly after—a curious instance of synchronicity with usernames and browsing experiences.
- **Poetry in Motion**: Abhiabhi shared part of a poetic composition titled "An Appeal to A Monkey," musing on the primitive yet vivid essence of life in contrast with the human-created systems.
- **Fine-tuning Performance Comparisons**: Goutham_city sought advice on whether to use **Gemma 7b** for a domain-specific classification task after trying **Mistral-7b**. Starsupernova mentioned that Unsloth has fixed all bug fixes and suggested that Gemma compares variably with Mistral.
- **In Search of the Mixtral Branch**: Dogsofwarren was redirected to a pull request on GitHub for the elusive "Mixtral branch" they were searching for. The link provided directed to [tohrnii's branch of the Unsloth Ai repository](https://github.com/unslothai/unsloth/pull/145).
- **Sharing Pokemon RL Agents on a Map & Open-Source UI Elements**: Iron_bound highlighted a visualization of people's environment training in *Pokemon RL* shared on a single map, while Yahir9023 shared [Uiverse.io](https://uiverse.io/elements), a site featuring open-source UI elements made with CSS or Tailwind.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pwhiddy.github.io/pokerl-map-viz/">Pokemon Red Map RL Visualizer</a>: no description found</li><li><a href="https://uiverse.io/elements">4203 UI elements: CSS &amp; Tailwind</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/pull/145">[WIP] add support for mixtral by tohrnii · Pull Request #145 · unslothai/unsloth</a>: Mixtral WIP
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1218104575022727230)** (568 messages🔥🔥🔥): 

- **Woes of Model Conversion and Training**: Users are attempting to fine-tune models such as **Unsloth's Mistral 7b** using Colab and encountering various issues like exceeding token limits and errors during the saving process to platforms like **HuggingFace**. One suggestion made for such errors was to manually run GGUF conversion commands.

- **Seeking Clarification on Quantization**: There's confusion about the difference between **LoRA** and **QLoRA** pertaining to whether `load_in_4bit = True` signifies QLoRA and what exactly the model name should be for each type. **MRdragonfox** clarified that `4bit` indeed indicates a QLoRA.

- **Troubleshooting Dataset and Template Issues**: Users are facing problems with chatbot models generating unsolicited content beyond given prompts. One hypothesis is that this could be due to incorrect application or absence of end-of-sequence tokens in the chat templates.

- **Future Plans for Full Fine Tuning (FFT)**: There's an interest in whether **Unsloth** might support full fine tuning in addition to **LoRA** and **QLoRA** in the future. Currently, Unsloth specializes in LoRA and QLoRA and doesn't fully support FFT, though the team is open to future possibilities.

- **Deployment Dilemmas**: Some users are asking about deploying models like **OpenAI's GPT-4** with Unsloth, which is not supported. Unsloth primarily facilitates finetuning and quantization of LLMs like **Mistral, Llama, Gemma**, with detailed instructions and notebooks provided for these specific models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1X_PHYBawrsCgKfMEPxvIDX__rYa1-v97?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://huggingface.co/ybelkada/Mixtral-8x7B-Instruct-v0.1-bnb-4bit">ybelkada/Mixtral-8x7B-Instruct-v0.1-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://www.kaggle.com/code/danielhanchen/kaggle-mistral-7b-unsloth-notebook">Kaggle Mistral 7b Unsloth notebook</a>: Explore and run machine learning code with Kaggle Notebooks | Using data from No attached data sources</li><li><a href="https://colab.research.google.com/drive/1Dyauq4kTZoLewQ1cApceUQVNcnnNTzg_?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0">TinyLlama/TinyLlama-1.1B-Chat-v1.0 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/settings/tokens">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://huggingface.co/unsloth/mistral-7b-instruct-v0.2-bnb-4bit">unsloth/mistral-7b-instruct-v0.2-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16">Home</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">Home</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/artidoro/qlora/blob/main/qlora.py#L746">qlora/qlora.py at main · artidoro/qlora</a>: QLoRA: Efficient Finetuning of Quantized LLMs. Contribute to artidoro/qlora development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-to-gguf">Home</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#chat-templates">Home</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing#scrollTo=FqfebeAdT073,">Google Colaboratory</a>: no description found</li><li><a href="https://pastebin.com/ybSeKHhU">Unsloth: Merging 4bit and LoRA weights to 16bit...Unsloth: Will use up to 5.34 - Pastebin.com</a>: Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.</li><li><a href="https://github.com/unslothai/unsloth/wiki#evaluation-loop---also-oom-or-crashing">Home</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://huggingface.co/docs/trl/main/en/dpo_trainer#accelerate-dpo-fine-tuning-using-unsloth">DPO Trainer</a>: no description found</li><li><a href="https://github.com/vllm-project/vllm">GitHub - vllm-project/vllm: A high-throughput and memory-efficient inference and serving engine for LLMs</a>: A high-throughput and memory-efficient inference and serving engine for LLMs - vllm-project/vllm</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://docs.gpt4all.io/gpt4all_python.html">Generation - GPT4All Documentation</a>: no description found</li><li><a href="https://github.com/huggingface/trl/issues/1041">Does DPOTrainer loss mask the prompts? · Issue #1041 · huggingface/trl</a>: Hi quick question, so DataCollatorForCompletionOnlyLM will train only on the responses by loss masking the prompts. Does it work this way with DPOTrainer (DPODataCollatorWithPadding) as well? Looki...</li><li><a href="https://huggingface.co/docs/trl/v0.7.11/en/sft_trainer#train-on-completions-only).">Supervised Fine-tuning Trainer</a>: no description found</li><li><a href="https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha/discussions/2/files">HuggingFaceH4/zephyr-7b-alpha · Add chat template</a>: no description found</li><li><a href="https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha#intended-uses--limitations">HuggingFaceH4/zephyr-7b-alpha · Hugging Face</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py#L56">unsloth/unsloth/chat_templates.py at main · unslothai/unsloth</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments">Trainer</a>: no description found</li><li><a href="https://github.com/huggingface/alignment-handbook/issues/45#issuecomment-1845598205">Reproducing of Lora Model  Result on MT-Bench · Issue #45 · huggingface/alignment-handbook</a>: Recently, I attempted to fit the DPO on my own dataset. Initially, I tried to reproduce the results of your LORA model( 7.43 on MT-Bench). However, I encountered some issues. Despite using all your...</li><li><a href="https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md">llama.cpp/examples/server/README.md at master · ggerganov/llama.cpp</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.</li><li><a href="https://github.com/abetlen/llama-cpp-python">GitHub - abetlen/llama-cpp-python: Python bindings for llama.cpp</a>: Python bindings for llama.cpp. Contribute to abetlen/llama-cpp-python development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1218239216975351928)** (21 messages🔥): 

- **Training Dilemmas: More Epochs, More Knowledge?**: A discussion on training models with 800,000 lines of data explored the idea that increasing epochs could lead to better training. However, it was pointed out that while more epochs *might* be better for voice neural networks, for LLMs like **Tiny Mistral**, this could result in the model forgetting its pre-existing knowledge.

- **Finding the Knowledge Sweet Spot**: In the quest for maximum knowledge retention, the suggestion was made to remove excess data, with a recognition that simply having a large dataset (like 3 million lines) might not be beneficial for finetuning models to improve performance.

- **Model Integration and Configuration Discussed**: A Tiny Mistral model configured with the Axolotl badge was showcased, with detailed settings and dataset processes provided. The relevant configurations and datasets used can be found at [Tiny Mistral model on Hugging Face](https://huggingface.co/Dans-DiscountModels/TinyMistral-v2.5-MiniPile-Guidelines-E1).

- **Exploring the Impact of Model Parameters**: A conversation about how to achieve optimal results from large datasets included suggestions on the ideal rank and alpha values, indicating that a rank of 32 or 64 and an alpha value (rank * 2) might be suited for an 800k line dataset.

- **Response to Model Integration Attempt**: The sharing of Tiny Mistral models for potential integration into the Unsloth Repo drew mixed reactions, with one response highlighting intrigue and another expressing that the results were underwhelming.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Dans-DiscountModels/TinyMistral-v2.5-MiniPile-Guidelines-E1">Dans-DiscountModels/TinyMistral-v2.5-MiniPile-Guidelines-E1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/M4-ai/TinyMistral-6x248M-Instruct/tree/main">M4-ai/TinyMistral-6x248M-Instruct at main</a>: no description found
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1218098224586293319)** (301 messages🔥🔥): 

- **Seeking LLM Suggestions for Local Runs**: Users suggested **CodeLlama** or **DeepSeek** for coding and mentioned models are limited to 13b on an MBP with 18GB memory. For running on M3 Pro, recommendations included **qwen** and models within the constraints of available memory.
- **Exploring Local Face Swapping Models**: A user inquired about local models for face swapping in videos similar to **Reface**, and it was mentioned that face swapping is more of a task for stable diffusion rather than language models, with **facefusion** suggested as an alternative.
- **Feature Feedback for Version 0.2.17**: Users discussed early access previews for the upcoming **LM Studio version 0.2.17** on different operating systems, with positive initial feedback.
- **GPU Utilization and Configuration Rather Tricky**: There were technical discussions around configuring **LM Studio** to utilize specific GPUs when multiple are present, with success stories using tensor split configurations. Compatibility issues with older **Tesla cards**, such as K40 and possibilities of running modern LLMs on them were also discussed.
- **Open Source Model Grok Released**: A discussion centered around **Grok**, a 314B parameter model, being open-sourced and its incompatibility with LM Studio due to the requirement for extremely high resources to run. Users are interested but recognize the limitations for local use.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://academictorrents.com/details/5f96d43576e3d386c9ba65b883210a393b68210e">grok-1</a>: Grok-1 is a 314B parameter Mixture of Experts model - Base model (not finetuned) - 8 experts (2 active) - 86B active parameters - Apache 2.0 license - Code:  - Happy coding! p.s. we re hiring: </li><li><a href="https://tenor.com/view/ratha-gif-26742750">Ratha GIF - Ratha - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=zjkBMFhNj_g&">[1hr Talk] Intro to Large Language Models</a>: This is a 1 hour general-audience introduction to Large Language Models: the core technical component behind systems like ChatGPT, Claude, and Bard. What the...</li><li><a href="https://huggingface.co/xai-org/grok-1/discussions/30">xai-org/grok-1 · 314B  params  has  297G  file size ?</a>: no description found</li><li><a href="https://github.com/continuedev/continue/issues/713"">Issues · continuedev/continue</a>: ⏩ The easiest way to code with any LLM—Continue is an open-source autopilot for VS Code and JetBrains - Issues · continuedev/continue</li><li><a href="https://www.youtube.com/watch?v=lCZRwrRvrWg&">Mistral: Easiest Way to Fine-Tune on Custom Data</a>: This video is sponsored by Gradient.ai, check them out here: https://gradient.1stcollab.com/engineerpromptIn this video, we will learn how to fine-tune Mistr...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1218119135423234058)** (138 messages🔥🔥): 

- **Command-R Model Awaits LM Studio Update**: Anticipation builds around LM Studio's upcoming release with support for Command-R model from CohereForAI after [Pull Request #6033](https://github.com/ggerganov/llama.cpp/pull/6033) integration on GitHub.

- **Hardware Constraints Dictate Model Performance**: Members are navigating the constraints of their hardware setups, with some GPU owners like those with a "RTX 2070 Super" or "1660 Super" seeking the best-suited models for their systems.

- **Grok-1 Model Sparks Interest but Exceeds Typical Resources**: Discussion of xAI's new Grok-1 model release touches on its immense size, questioning the practicality of running it on typical personal computers and considering alternative hosting strategies.

- **Updates and Inferences**: Members are curious about the availability of new models like Grok in LM Studio, while others express caution, noting models like Grok's high resource requirements or base models' potential need for additional training to realize their potential.

- **Yi-200k and Template Troubles**: There's an ongoing discussion about the best template for Yi-200k, with the community sharing insights and resources, including links to [Hugging Face](https://huggingface.co/01-ai/Yi-9B-200K), to better utilize this particular model.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://news.ycombinator.com/item?id=39737281">no title found</a>: no description found</li><li><a href="https://x.ai/blog/grok-os">Open Release of Grok-1</a>: no description found</li><li><a href="https://arxiv.org/abs/2403.09611">MM1: Methods, Analysis &amp; Insights from Multimodal LLM Pre-training</a>: In this work, we discuss building performant Multimodal Large Language Models (MLLMs). In particular, we study the importance of various architecture components and data choices. Through careful and c...</li><li><a href="https://huggingface.co/01-ai/Yi-34B/discussions/23">01-ai/Yi-34B · Prompt template?</a>: no description found</li><li><a href="https://huggingface.co/01-ai/Yi-9B-200K">01-ai/Yi-9B-200K · Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://youtu.be/PAbZRGGYNyM?si=xVNZCYUddDvoFUly">What are  Parameters in Large Language Model?</a>: What are the Parameters in the Large Language Model? 00:26 💡 Parameters in large language models like GPT-3 are variables learned during training to minimiz...</li><li><a href="https://youtu.be/zjkBMFhNj_g?si=Rn96V9CMqEHLy6-7">[1hr Talk] Intro to Large Language Models</a>: This is a 1 hour general-audience introduction to Large Language Models: the core technical component behind systems like ChatGPT, Claude, and Bard. What the...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6033">Add Command-R Model by acanis · Pull Request #6033 · ggerganov/llama.cpp</a>: Information about the Command-R 35B model (128k context) can be found at: https://huggingface.co/CohereForAI/c4ai-command-r-v01 Based on the llama2 model with a few changes:  New hyper parameter to...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1218213037060657273)** (12 messages🔥): 

- **Confusion over llama.cpp Support**: A member thought **GGUF files** listed as compatible with the llama.cpp indicated support for **Cohere's Command-R model**, but was corrected noting that llama.cpp **does not currently support c4ai**.
- **Clarity on GGUF Compatibility**: Despite the appearance of compatible files, clarification was made that **GGUF files for Command-R 35B v1.0** require llama.cpp from Mar 16, 2024, onwards, starting from release [b2440](https://github.com/ggerganov/llama.cpp/releases/tag/b2440). The files are split and need to be joined due to Hugging Face's 50GB limit.
- **Mistake Acknowledged in Model Support Discussion**: A user acknowledged their mistake in the discussion regarding file compatibility, as there was previous mention in the models-discussion channel that they had overlooked.
- **GPU Usage Note for AMD Linux Users**: A member requested that a note be added on the Linux version download page stating that AMD users need **OpenCL drivers** to use GPUs with the program.
- **Plugin and Document Interaction Query in LM Studio**: Query arose whether **LM Studio** supports chatting with one's own documents or adding plugins like autogen, with a response pointing out server mode can be turned on to connect plugins already supported by LM Studio.

**Link mentioned**: <a href="https://huggingface.co/andrewcanis/c4ai-command-r-v01-GGUF">andrewcanis/c4ai-command-r-v01-GGUF · Hugging Face</a>: no description found

  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1218129474348912711)** (480 messages🔥🔥🔥): 

- **K80 Purchase and Cooling Plans**: A member considered buying a Tesla **K80 GPU** from eBay for **€150** with plans to improve its cooling using 3D printed shrouds from Thingiverse. 
- **Server and GPU Configurations Talk**: There was discussion about different **server** setups and whether **Threadripper** or **EPYC CPUs** would better suit a **4-slot motherboard** for a **multi-GPU system** for LLM inference.
- **Amazon's Resell Market**: Members touched on how **Amazon** has become a marketplace for resellers, often marking up products originally listed on **eBay** or **AliExpress**.
- **Tesla K80's Memory Bandwidth for LLMs**: A Tesla **K80 GPU** was discussed for LLM tasks with consideration of its slow memory bandwidth, with a suggestion that **RX480 GPUs** might perform better due to having higher memory bandwidth.
- **EPYC System Build Discussion**: There were discussions around building a **DIY EPYC system** for LLM use, with a focus on ensuring sufficient PCIe lanes and considering cost-effective components such as the **EPYC 7232P CPU**. Members debated whether the price of certain CPUs and GPUs like the **NVIDIA H100** was justified compared to **second-hand K80 GPUs**.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.amazon.com/AMD-3200MHZ-SYSTEM-COMPONENTS-PROCESSORS/dp/B07XP9S55C/ref=sr_1_2">no title found</a>: no description found</li><li><a href="https://lmstudio.ai/#can-i-use-lm-studio-at-work?">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://lmstudio.ai/beta-releases.html">LM Studio Beta Releases</a>: no description found</li><li><a href="https://www.amazon.de/-/en/HHCJ6-NVIDIA-Server-Accelerator-Renewed/dp/B07GJ45V3D/ref=sr_1_2?crid=1O8IZM1RV0TIH&dib=eyJ2IjoiMSJ9.B2ZUEDxvj_Z73GUX0GJebEDmX0cqUrowZhMOgYhwtCaPdx9UH8NiM39aqowgVAc5YENjqRh8_cc1qHbgwPJMprvhMhnuusRAJuQqLmWDyskupHMP8ACQI354KZZjKYrdtnPPNGnuoJdVlHxoPQ8ll9ilsDZZ334_L6TwueHlrTelgoIjaTt650I3FQyWgOFmpTvAb3YigqPDURnBJMq1D6wanBHjVSaSdFOEnWlP2cUV8J9Hq4Lh_0bJbRh-kAaca58OndCeXm-tGVmNFLi7TuMKGZORpZ0Q6IcMd6Vz11w.MFnlYLfXX9YWUon0J_Dg0ds2eKFM6AwZgazWMdxeEjE&dib_tag=se&keywords=Tesla+K80&qid=1710787582&s=computers&sprefix=tesla+k80%2Ccomputers%2C421&sr=1-2">no title found</a>: no description found</li><li><a href="https://coral.ai/products/m2-accelerator-dual-edgetpu#description">M.2 Accelerator with Dual Edge TPU | Coral</a>: Integrate two Edge TPUs into legacy and new systems using an M.2 (E key) interface.</li><li><a href="https://www.aliexpress.com/item/100500634581">404 page</a>: no description found</li><li><a href="https://www.ebay.co.uk/itm/273788651049?">Dell T710 Tower Server Dual 6-CORE X5650 **144Gb RAM**240gb SSD +6X 600G SFF SAS  | eBay</a>: no description found</li><li><a href="https://www.newegg.com/asrock-rack-romed8-2t/p/N82E16813140044">Asrock Rack ROMED8-2T ATX Server Motherboard AMD EPYC 7003 (with AMD 3D V-Cache Technology)/7002 series processors SP3 (LGA 4094) Dual 10GbE - Newegg.com</a>: Buy Asrock Rack ROMED8-2T Server Motherboard AMD EPYC 7003 (with AMD 3D V-Cache Technology)/7002 series processors SP3 (LGA 4094) Dual 10GbE with fast shipping and top-rated customer service. Once you...</li><li><a href="https://www.aliexpress.com/item/1005006525215524.html">no title found</a>: no description found</li><li><a href="https://www.ebay.de/itm/125947603377?itmmeta=01HS9HRSJMXBV00M1XW59H5NAE&hash=item1d530fe9b1:g:fHQAAOSwWVxkbefZ&itmprp=enc%3AAQAJAAAA4A6tXSRz7NxXocQqxCeo%2F2TdOTiIP1AMtfRCBxeBISSicEa3bP%2FtSfa9CmVAH74vTwUFyfwFd1VhNC71wMalgSqfYNDwr7svQreF5j3Gqk4Brm8Zn7hMHU6mRQVuxRyyv5VyA1PeZKdylhbJH0O%2BC2IM8GdP7yLRbRw6sOGTb2KMO0V0m%2B7aGkzXe6h33qOgF16cjz2vh2TITEEOr1eYGfz7ViQZ846gljR8VFArZiDwxgIU8naY8yQRPUJe4Znn3GYEn3GT3DNHxdg5zoB7qyMOytwL9TKozBLIkBQVtyyq%7Ctkp%3ABk9SR8KZ47HKYw">New /Wave ®AI Server NF5688M6 NVIDIA HGX TESLA A800 80G octet GPU server/Futures  | eBay</a>: no description found</li><li><a href="https://www.ebay.ca/itm/126375063761">AMD EPYC 7232P 8-Core 3.1GHz 32MB L3 Processor - Socket SP3 - 100-000000081  | eBay</a>: no description found</li><li><a href="https://www.ebay.co.uk/itm/115960685949?">AMD EPYC 7F72 CPU PROCESSOR 24 CORE 3.20GHz 192MB CACHE 240W - 100-000000141  | eBay</a>: no description found</li><li><a href="https://www.ebay.de/itm/126352871326?epid=11041255665&itmmeta=01HS9333CQ68S4STA8BZJ3V0BH&hash=item1d6b37cf9e:g:DOEAAOSweRlkuVOG&itmprp=enc%3AAQAJAAAA0GtLL6BuVwKKMH1iyVWS1kdp6p0LvQb%2Fcu8c94aisQZDISgf4yKcfrjNbigVkO4IGdfBt3tcIr6du3Nb1xXGbEe2CNScd%2B4RoCdoEx%2BQMPtNGs0TtY3wzAbszVam1AHN8tC%2Bzq%2BVoVhSwCmdZ77779duZUVHF%2Fq1ckL28OWoVp%2FRStC3u0NyyTZtUke6tEsgNdQYOKI4%2BqNOIN11tc8XuhOtaovFo6WzH87nIC6BUNiaWYnvWcqUPH3NUs6Gxi%2FWnel1Vj9wokxL8oELjbCFBOA%3D%7Ctkp%3ABFBMyLaMo8pj">AMD EPYC 7232P CPU PROCESSOR 8 CORE 3.10GHz 32MB CACHE 120W - 100-000000081  | eBay</a>: no description found</li><li><a href="https://www.ebay.co.uk/itm/296113403496?">Dell T710 Tower Server Dual 6-CORE X5670 **24 cores**64GB RAM  | eBay</a>: no description found</li><li><a href="https://www.ebay.de/itm/145329120119?epid=507128083&itmmeta=01HS9DKVRXS2WQPFX74KY649GW&hash=item21d64a6377:g:kacAAOSw~q1lFEwb&itmprp=enc%3AAQAJAAAA4GTzwRZBHO82ltgqug5ARkRZ5JKlaikKECFytG5%2FNjvBMzyE2UGOBW0yRbeW%2B%2F3prx2LD9sPaLsinW103607IHMVVMe2tg6FIa2KVc%2FUVWqCGgQPrRRS97i9Q%2FZW0nnLz5XSLuFob%2FicmlhLi7Ve68FV47SLRenj5tDoUD8mwpvdoxA5uQtR0DNACYnvlVQe4BeXKFAWKA8iKA6WdrVikWOsQcODTpcW916%2FL8jFOUSFjg9D5%2FP1xg4foswYBWrIeaD4Pm9rguigAFQvYGqHFLKNXgB4CjCD0BczHhSZYunI%7Ctkp%3ABk9SR8i8z63KYw">Nvidia Tesla K80 24GB GPU GDDR5 PCI-E GPU Accelerator 12 Month warranty  | eBay</a>: no description found</li><li><a href="https://www.ebay.de/itm/145329120119?epid=507128083&itmmeta=01HS9DKVRXS2WQPFX74KY649GW&hash=item21d6">Nvidia Tesla K80 24GB GPU GDDR5 PCI-E GPU Accelerator 12 Month warranty  | eBay</a>: no description found</li><li><a href="https://www.thingiverse.com/search?q=K80+cooling+&page=1&type=things&sort=relevant">Search Thingiverse - Thingiverse</a>: Download files and build them with your 3D printer, laser cutter, or CNC.</li><li><a href="https://www.techpowerup.com/cpu-specs/core-i5-3470.c1039#:~:text=Programs%20using%20Advanced%20Vector%20Extensions,performance%20for%20calculation%2Dheavy%20applications.">Intel Core i5-3470 Specs</a>: Ivy Bridge, 4 Cores, 4 Threads, 3.2 GHz, 77 W</li><li><a href="https://www.microcenter.com/product/677156/nvidia-geforce-rtx-3090-founders-edition-dual-fan-24gb-gddr6x-pcie-40-graphics-card-(refurbished)">Micro Center - Computers and Electronics</a>: Micro Center - Computers and Electronics - Thousands of products to buy: desktops, laptops, monitors, build your own PC parts, upgrades, digital imaging, printing supplies, portable devices, audio equ...</li><li><a href="https://zifa666.aliexpress.com/store/5885523/pages/all-items.html?productGroupId=40000003590095&shop_sortType=bestmatch_sort">Luckim Official Store - Amazing products with exclusive discounts on AliExpress</a>: no description found</li><li><a href="https://www.aliexpress.com/item/1005006345813657.html">no title found</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1219065221327355974)** (4 messages): 

- **Seeking Model Presets Guide**: A user inquired about the availability of **presets for different models**. They were provided with a link to a GitHub repository containing JSON configuration files and a collection of example configs at [lmstudio-ai/configs](https://github.com/lmstudio-ai/configs).
- **ROCm User Roll Call**: A member seeking other users of **ROCm** was directed to a specific channel (**#1195858490338594866**) to find and engage with the relevant community.

**Link mentioned**: <a href="https://github.com/lmstudio-ai/configs">GitHub - lmstudio-ai/configs: LM Studio JSON configuration file format and a collection of example config files.</a>: LM Studio JSON configuration file format and a collection of example config files. - lmstudio-ai/configs

  

---


**LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1219051718172606537)** (1 messages): 

- **Inquiry on Local Inference Server Capabilities**: A member inquired about the possibility of getting a model with **JSON function calling** to operate on the **Local Inference Server**. No other members have yet provided insights or shared experiences regarding this functionality.
  

---


**LM Studio ▷ #[avx-beta](https://discord.com/channels/1110598183144399058/1177047883237822536/1219383598193311744)** (5 messages): 

- **Clarification on AVX Beta**: A member questioned whether the beta version of the app in question is simply using AVX. Another member confirmed this and added that it's also an older version, saying **"avx support really isn't a high priority item."**
- **Compatibility with Older Models Assured**: It was confirmed that while models will work in the beta app, **newer ones like starcoder2, gemma** and such will not be supported.
- **Mistral Model Compatibility Query**: One member asked for confirmation if **Mistral** would run on the app, implying compatibility with at least some established models.
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1218206050495234070)** (5 messages): 

- **GitHub Resource Shared**: A link to [Prebuilt Windows ROCm Libs for gfx1031 and gfx1032](https://github.com/brknsoul/ROCmLibs) was shared, providing resources for those looking to use **Windows ROCm** libraries on specific AMD GPUs.

- **Anticipating Multi-GPU Support in LM Studio**: A member expressed interest in using **multiple AMD GPUs** with LM Studio but noted the current version seems to rely only on the primary GPU. They questioned the timeline for multi-GPU support within the platform.

- **Compatibility Issues with Radeon 6700 XT**: It was clarified that the **AMD Radeon 6700 XT** is not officially supported by **AMD for ROCm**, and LM Studio uses the ROCm libraries unmodified, indicating why the 6700 XT may not function with ROCm.

- **Hope for Multi-GPU Usage with Different AMD Models**: Responding to compatibility concerns, it was indicated that if one had another GPU from the **7000 series**, LM Studio would likely utilize them in parallel.

- **Current Multi-GPU Support with KoboldCPP-ROCm**: Despite compatibility issues, it was mentioned that **KoboldCPP-ROCm** could indeed work fine with multiple GPUs in the current state.

**Link mentioned**: <a href="https://github.com/brknsoul/ROCmLibs">GitHub - brknsoul/ROCmLibs: Prebuild Windows ROCM Libs for gfx1031 and gfx1032</a>: Prebuild Windows ROCM Libs for gfx1031 and gfx1032 - brknsoul/ROCmLibs

  

---


**LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1219265025487667200)** (1 messages): 

- **In Search of the Right Agent**: A member inquired about another's progress in selecting an agent system, expressing a shared interest in the process. They mentioned aiming to **deepen and validate a creative concept** through different agents.
  

---



**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1218144997094723615)** (56 messages🔥🔥): 

- **NVIDIA's Next-Gen GDDR7 Memory**: The upcoming [NVIDIA GeForce RTX 50-series "Blackwell"](https://www.techpowerup.com/320185/nvidia-geforce-rtx-50-series-blackwell-to-use-28-gbps-gddr7-memory-speed) is rumored to implement GDDR7 memory with speeds of 28 Gbps, offering 55% higher bandwidth over GDDR6, despite GDDR7 chips supporting up to 32 Gbps.
- **Customer Service Woes**: A member shared optimism about a support ticket getting escalated, joking about "Greg" potentially fixing their OAuth login error.
- **Model Advancements vs. Agent Related Improvements**: Members discuss the likelihood of model advancements versus agent improvements, speculating that OpenAI’s upcoming releases may focus more on agent-related capabilities, with mentions of a robust "agent control interface" expected.
- **Anticipating Game-Changing Agents**: There's anticipation about novel AI agents with advanced reliability, as pointed out in a discussion on how AI model improvements are necessary for high-quality agent interfaces, referencing Sam Altman's predictions regarding AI's growing capabilities.
- **Conversations about Responsive AI Assistants**: A dialogue is ongoing regarding techniques to make AI assistants more interactive by stopping and resuming output intelligently, with suggestions ranging from editing conversation history to simple audio pause-resume logic, with some community members offering help and examples.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/unkjdgames?s=21">Tweet from undefined</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=ZlJbaYQ2hm4">Plan-and-Execute using Langgraph</a>: how to create a &quot;plan-and-execute&quot; style agent. This is heavily inspired by the Plan-and-Solve paper as well as the Baby-AGI project.The core idea is to firs...</li><li><a href="https://www.techpowerup.com/320185/nvidia-geforce-rtx-50-series-blackwell-to-use-28-gbps-gddr7-memory-speed">NVIDIA GeForce RTX 50-series &quot;Blackwell&quot; to use 28 Gbps GDDR7 Memory Speed</a>: The first round of NVIDIA GeForce RTX 50-series &quot;Blackwell&quot; graphics cards that implement GDDR7 memory are rumored to come with a memory speed of 28 Gbps, according to kopite7kimi, a reliabl...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1218108265854926899)** (16 messages🔥): 

- **"Horny Claudes" Enhancing AI Outputs?**: A chatbot experiment reportedly found that "horny claudes" create better mermaid diagrams. The phenomena led to discussions with some finding the tactic sickening, drawing comparisons to "reverse Sydney."

- **Diving into Model Experimentation**: A member shared a personal research project focused on getting started with PyTorch, which they described as perhaps not groundbreaking but a valuable learning experience. The project findings are available at [Derbydefi's Twitter](https://vxtwitter.com/derbydefi/status/1768767386419970071).

- **Apple Drops AI Model Info**: Apple is finally discussing their AI model work, as noted by a member who shared a [link to a tweet by Aran Komatsuzaki](https://twitter.com/arankomatsuzaki/status/1768446729710371115), sparking responses in the community regarding the lack of released weights.

- **ORPO: A New Algorithm for Preference Alignment**: A paper on Hugging Face introduces ORPO, a novel algorithm that claims to streamline supervised fine-tuning for language models by using an odds ratio. The [abstract of the paper](https://huggingface.co/papers/2403.07691) outlines the concept and its promise for preference alignment without additional phases.

- **Recreating Self-Rewarding Language Model Research**: The Oxen.ai Community is working to reproduce the Self-Rewarding Language Model paper from MetaAI. The progress and code are documented on their [GitHub repository](https://github.com/Oxen-AI/Self-Rewarding-Language-Models).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/repligate/status/1768521441329434937?s=20">Tweet from j⧉nus (@repligate)</a>: @xlr8harder I didn&#39;t let it go very far but there&#39;s someone in the room with me right now talking about how theyve created a network of &#34;horny claudes&#34; and how the claudes create bette...</li><li><a href="https://arxiv.org/abs/2402.16823">Language Agents as Optimizable Graphs</a>: Various human-designed prompt engineering techniques have been proposed to improve problem solvers based on Large Language Models (LLMs), yielding many disparate code bases. We unify these approaches ...</li><li><a href="https://fxtwitter.com/burny_tech/status/1769530798242255129">Tweet from Burny — Effective Omni (@burny_tech)</a>: My thoughts on Musk destabilizing other gigantic players in the intelligence wars by possibly leading open source using Grok   Grok 1 is a 314B parameter model and it&#39;s a mixture of experts archit...</li><li><a href="https://huggingface.co/papers/2403.07691">Paper page - ORPO: Monolithic Preference Optimization without Reference Model</a>: no description found</li><li><a href="https://github.com/Oxen-AI/Self-Rewarding-Language-Models">GitHub - Oxen-AI/Self-Rewarding-Language-Models: This is work done by the Oxen.ai Community, trying to reproduce the Self-Rewarding Language Model paper from MetaAI.</a>: This is work done by the Oxen.ai Community, trying to reproduce the Self-Rewarding Language Model paper from MetaAI. - Oxen-AI/Self-Rewarding-Language-Models
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1218105907615895562)** (656 messages🔥🔥🔥): 

- **Yi License Confusion**: Discussions about the commercial use of Yi-9B model triggered speculation, with users unsure if automatic email approval is marketing or a real open door for commercial application.

- **Grok-1's Release Stirs Debate**: Grok-1, a large mixture-of-experts model with 314 billion parameters, garners reactions from the AI community. Some express skepticism about its performance despite its size, comparing it unfavorably to smaller, more efficient models like Mistral-7B.

- **Continual Pretraining and MoEs**: Conversations around continuous pretraining of MoEs (Mixture-of-Experts) unfold. Users share experiences and mention that while it helped with less-resourced languages, the process is uncharted for MoEs, and larger models might not benefit without high-quality data.

- **Alignment and "Wokeness" Discussions**: The community debates "wokeness" in AI, starting with concerns about models being overtly politically correct or altering prompts based on company ethos. The discussion evolves into talks about AI alignment, the use of steering vectors, and the idea of AI models reflecting constitutional principles.

- **GPT-4 Architecture Leak during GTC Keynote**: NVIDIA's CEO, Jensen Huang, may have inadvertently confirmed rumors about GPT-4’s 1.8 trillion parameter count and its MoE architecture during a GTC keynote address, leaving the AI community surprised and speculating about the model's capabilities and cost.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/aravsrinivas/status/1769485603622867394?s=46&t=TOasxww3M5DjlB4iBWa_ig">Tweet from Aravind Srinivas (@AravSrinivas)</a>: Yep, thanks to @elonmusk and xAI team for open-sourcing the base model for Grok. We will fine-tune it for conversational search and optimize the inference, and bring it up for all Pro users!  ↘️ Quoti...</li><li><a href="https://fxtwitter.com/intrstllrninja/status/1768942321129697790?s=20">Tweet from interstellarninja (@intrstllrninja)</a>: Hermes 2 Pro function-calling model integrated with search engine by @ExaAILabs👀  ↘️ Quoting Barton Rhodes 🦺 (@bmorphism)   added @ExaAILabs support for use with @NousResearch new function-calling m...</li><li><a href="https://arxiv.org/abs/2403.09611">MM1: Methods, Analysis &amp; Insights from Multimodal LLM Pre-training</a>: In this work, we discuss building performant Multimodal Large Language Models (MLLMs). In particular, we study the importance of various architecture components and data choices. Through careful and c...</li><li><a href="https://arxiv.org/abs/2402.17764">The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits</a>: Recent research, such as BitNet, is paving the way for a new era of 1-bit Large Language Models (LLMs). In this work, we introduce a 1-bit LLM variant, namely BitNet b1.58, in which every single param...</li><li><a href="https://arxiv.org/abs/2403.08763">Simple and Scalable Strategies to Continually Pre-train Large Language Models</a>: Large language models (LLMs) are routinely pre-trained on billions of tokens, only to start the process over again once new data becomes available. A much more efficient solution is to continually pre...</li><li><a href="https://x.ai/blog/grok-os">Open Release of Grok-1</a>: no description found</li><li><a href="https://fxtwitter.com/intrstllrninja/status/1768948484479049897?s=20">Tweet from interstellarninja (@intrstllrninja)</a>: &lt;cmd&gt; run world_sim.exe --epoch &#34;Earth in 2500&#34; --civilization_type &#34;Type-II on Kardashev scale&#34; &lt;/cmd&gt;  ↘️ Quoting mephisto (@karan4d)   im opensourcing worldsim of course...</li><li><a href="https://arxiv.org/abs/2402.10588">Do Llamas Work in English? On the Latent Language of Multilingual Transformers</a>: We ask whether multilingual language models trained on unbalanced, English-dominated corpora use English as an internal pivot language -- a question of key importance for understanding how language mo...</li><li><a href="https://huggingface.co/Replete-AI/Mistral-11b-v0.1">Replete-AI/Mistral-Evolved-11b-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered">anon8231489123/ShareGPT_Vicuna_unfiltered · Datasets at Hugging Face</a>: no description found</li><li><a href="https://fxtwitter.com/intrstllrninja/status/1769773746896662873?s=20">Tweet from interstellarninja (@intrstllrninja)</a>: @Cyndesama claude 3 opus runs ai town simulation with python42</li><li><a href="https://huggingface.co/datas">datas (shu nakamura)</a>: no description found</li><li><a href="https://x.com/whyarethis/status/1769269824587542692?s=46">Tweet from Parzival - 🌞/⏫ (@whyarethis)</a>: Now we are going somewhere.</li><li><a href="https://x.com/itsandrewgao/status/1769460684956602527?s=46">Tweet from Andrew Kean Gao (@itsandrewgao)</a>: i think grok-4bit is just barely too big for an H100 GPU :(  ↘️ Quoting Andrew Kean Gao (@itsandrewgao)   HOLY SH*T @grok IS 314 BILLION PARAMETERS  Mixture of 8 Experts, not RLHFd/moralized  THIS IS ...</li><li><a href="https://x.com/burkov/status/1769496949252673550?s=46&t=TOasxww3M5DjlB4iBWa_ig">Tweet from Andriy Burkov (@burkov)</a>: We are yet to see how good Grok is compared to GPT-4, but what we can tell for sure is that if you are to train a competitor to OpenAI/Anthropic today, you would not need to start from scratch anymore...</li><li><a href="https://huggingface.co/migtissera/Tess-70B-v1.6">migtissera/Tess-70B-v1.6 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/openchat/openchat_sharegpt4_dataset/tree/main">openchat/openchat_sharegpt4_dataset at main</a>: no description found</li><li><a href="https://fxtwitter.com/lqiao/status/1768045066776707226?s=20">Tweet from Lin Qiao (@lqiao)</a>: We are thrilled to collaborate on Hermes 2 Pro multi-turn chat and function calling model with @NousResearch. Finetuned on over 15k function calls, and a 500 example function calling DPO datasets, Her...</li><li><a href="https://arxiv.org/abs/2303.11934">Sparse Distributed Memory is a Continual Learner</a>: Continual learning is a problem for artificial neural networks that their biological counterparts are adept at solving. Building on work using Sparse Distributed Memory (SDM) to connect a core neural ...</li><li><a href="https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO/discussions/10/files">NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO · Adding Evaluation Results</a>: no description found</li><li><a href="https://fxtwitter.com/intrstllrninja/status/1769424961192529962?s=20">Tweet from interstellarninja (@intrstllrninja)</a>: &lt;cmd&gt; sudo python3 akashic_records.py --entity [&#34;sam altman&#34;, &#34;elon musk&#34;] --mode &#34;email thread&#34; --topic &#34;superintelligence scenarios&#34; &lt;/cmd&gt;</li><li><a href="https://huggingface.co/01-ai/Yi-9B">01-ai/Yi-9B · Hugging Face</a>: no description found</li><li><a href="https://github.com/furlat/Abstractions/blob/main/abstractions/goap/causality.ipynb">Abstractions/abstractions/goap/causality.ipynb at main · furlat/Abstractions</a>: A Collection of Pydantic Models to Abstract IRL. Contribute to furlat/Abstractions development by creating an account on GitHub.</li><li><a href="https://www.hd-computing.com/">HD/VSA</a>:   </li><li><a href="https://arxiv.org/abs/2403.08540">Language models scale reliably with over-training and on downstream tasks</a>: Scaling laws are useful guides for developing language models, but there are still gaps between current scaling studies and how language models are ultimately trained and evaluated. For instance, scal...</li><li><a href="https://www.youtube.com/watch?v=Y2F8yisiS6E">GTC March 2024 Keynote with NVIDIA CEO Jensen Huang</a>: Watch NVIDIA CEO Jensen Huang’s GTC keynote to catch all the announcements on AI advances that are shaping our future.Dive into the announcements and discove...</li><li><a href="https://www.youtube.com/watch?v=t6SQj8YidGA">Accelerationism Accelerationism (Acc/Acc)</a>: Accelerationism accelerationism is when you accelerate accelerationism to apply accelerationism to accelerationismparts that were too edgy: https://www.patre...</li><li><a href="https://docs.pydantic.dev/latest/concepts/json_schema/">JSON Schema - Pydantic</a>: no description found</li><li><a href="https://www.youtube.com/wa">Liam Johnson DESTROYS Heckler | New York Stand-up</a>: Last weekend Liam Johnson decided to finally make his first appearance here at Giggle Nerd. He performed on Sunday from 23:00 to 23:25 and our audience loved...</li><li><a href="https://github.com/furlat/Abstractions/blob/main/abstractions/goap/gridmap.ipynb">Abstractions/abstractions/goap/gridmap.ipynb at main · furlat/Abstractions</a>: A Collection of Pydantic Models to Abstract IRL. Contribute to furlat/Abstractions development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=oYFjDt4-hFw&ab_channel=NewEconomicThinking">Cosma Shalizi - Why Economics Needs Data Mining</a>: Cosma Shalizi urges economists to stop doing what they are doing: Fitting large complex models to a small set of highly correlated time series data. Once you...</li><li><a href="https://github.com/furlat/Abstractions/blob/main/abstractions/goap/system_prompt.md">Abstractions/abstractions/goap/system_prompt.md at main · furlat/Abstractions</a>: A Collection of Pydantic Models to Abstract IRL. Contribute to furlat/Abstractions development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=zduSFxRajkE">Let&#39;s build the GPT Tokenizer</a>: The Tokenizer is a necessary and pervasive component of Large Language Models (LLMs), where it translates between strings and tokens (text chunks). Tokenizer...</li><li><a href="https://huggingface.co/01-ai/Yi-9B-200K">01-ai/Yi-9B-200K · Hugging Face</a>: no description found</li><li><a href="https://github.com/PrismarineJS/mineflayer">GitHub - PrismarineJS/mineflayer: Create Minecraft bots with a powerful, stable, and high level JavaScript API.</a>: Create Minecraft bots with a powerful, stable, and high level JavaScript API. - PrismarineJS/mineflayer</li><li><a href="https://x.com/grok/status/1769441648910479423?s=46">Tweet from Grok (@grok)</a>: @elonmusk @xai ░W░E░I░G░H░T░S░I░N░B░I░O░</li><li><a href="https://www.biorxiv.org/content/10.1101/2024.03.11.584515v1">Whole-body simulation of realistic fruit fly locomotion with deep reinforcement learning</a>: The body of an animal determines how the nervous system produces behavior. Therefore, detailed modeling of the neural control of sensorimotor behavior requires a detailed model of the body. Here we co...</li><li><a href="https://hack.meetmeinshibuya.com/">HacksTokyo</a>: AI x Digital Entertainment Hackathon in Tokyo!</li><li><a href="https://github.com/Prismarin">Prismarin - Overview</a>: Prismarin has 3 repositories available. Follow their code on GitHub.
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1218205298729156648)** (25 messages🔥): 

- **Perplexing Perplexity for Llama-2**: A member followed a [Kaggle notebook guide](https://www.kaggle.com/code/philculliton/calculating-the-perplexity-of-4-bit-llama-2/notebook) to compute perplexity for **Llama-2** but ended up with a perplexity of 90.3, seeking potential solutions from experiences of others.
- **Ambitious Upscaling Dreams Crushed by Reality**: Individuals discussed the viability of upscaling **Llama-2 13b** to 20b to surpass **Mistral** performance. It was speculated that acquiring funds of $300,000 for a 20b model seems daunting; another suggests a more economical $30,000 for continued pretraining.
- **Scaling Down As the New Black**: One member shared their current work on downscaling models with continuous pretraining and provided a link to their project *Smallstral*, a layer-pruned Mistral version, highlighting comparative performance metrics with [Mistral-7B](https://huggingface.co/AlexWortega/smallstral).
- **Quest for a Larger Model Premise**: Participants in the discussion theorized about the future production of large models like **genstruct 70b** or **openhermes 2.5** at 70b, considering financial and technological limitations.
- **Open-hermes Grok Speculation and Continued LoRA Pretraining Inquiry**: There was mention of the release of **Grok** by xai-org and speculation about **Open-hermes** adopting **Grok**. Additionally, a query was made regarding attempts to continued pretrain on domain-specific data using **LoRA** techniques.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.kaggle.com/code/philculliton/calculating-the-perplexity-of-4-bit-llama-2/notebook">Calculating the Perplexity of 4-bit Llama 2</a>: Explore and run machine learning code with Kaggle Notebooks | Using data from multiple data sources</li><li><a href="https://github.com/xai-org/grok-1">GitHub - xai-org/grok-1: Grok open release</a>: Grok open release. Contribute to xai-org/grok-1 development by creating an account on GitHub.</li><li><a href="https://huggingface.co/AlexWortega/smallstral">AlexWortega/smallstral · Hugging Face</a>: no description found</li><li><a href="https://wandb.ai/alexwortega/cpm_rus/runs/w5t4dsat?nw=nwuseralexwortega">alexwortega</a>: Weights & Biases, developer tools for machine learning
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[bittensor-finetune-subnet](https://discord.com/channels/1053877538025386074/1213221029359657021/1218181932853104720)** (18 messages🔥): 

- **Inquiries on Link Status**: A user asked if a specific link was broken, to which another user confirmed that the link was not broken.
- **Reflections on an Unspecified Idea**: User fullstack6209 expressed being in awe for several days over an idea, leading to a brief inquiry about their vague remark.
- **Bittensor Chain Issues Reported**: Users discussed the Bittensor chain's outage over the past 11 hours, joking on the situation and noting delays in fixing the issue.
- **Call for Information on Subtensor Update**: There was a mention that Bittensor is operational again, but it requires an update to subtensor which not everyone has completed.
- **Queries about Bittensor Participation**: One user sought advice on purchasing TAO tokens for registration, with a recommendation to use MEXC exchange. Additionally, there was a conversation about the hardware requirements with suggestions on setting up a qlora trainer and appropriate memory for different models.
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1218682432610373703)** (100 messages🔥🔥): 

- **Rethinking RAG Properties**: The team has discussed desirable properties for models within a **RAG (retrieval-augmented generation) pipeline**: low latency, context handling, knowledge breadth, function calling, and varied output structuring potentially using markdown and HTML-style citations. They emphasized the balance between the recall of provided context and reasoning, through modes that dictate response style and intent understanding.
- **Cohere's Model Piques Interest**: A member highlighted **Cohere's** model, which is already equipped for RAG tasks, as it can work with 128K contexts and offers features like span outputs and inline citation capabilities, potentially advancing the state of RAG systems.
- **Seeking a Delicate Balance**: The conversation shifted toward the idea of training models to either rely solely on external context or to mix external knowledge with the model's internal knowledge. They debated whether models should default to include their knowledge and only restrict it upon explicit instruction.
- **The RAG Mode Conundrum**: Members contemplated introducing a **"RAG mode"** to Hermes, envisioning a model capable of handling large document contexts with diverse functionalities, from citing sources to structuring outputs or writing code, aligning with user queries.
- **Exploring Specialized RAG Models**: They explored the potential of smaller, specialized models for RAG tasks to manage complexity in RAG pipelines, likening it to combining attributes of models like Opus and Haiku to enhance speed and efficiency during multiple calls to the model in a single document processing workflow.

**Link mentioned**: <a href="https://github.com/EveryOneIsGross/scratchTHOUGHTS/blob/main/commanDUH.py">scratchTHOUGHTS/commanDUH.py at main · EveryOneIsGross/scratchTHOUGHTS</a>: 2nd brain scratchmemory to avoid overrun errors with self. - EveryOneIsGross/scratchTHOUGHTS

  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1218167767379742813)** (273 messages🔥🔥): 

- **Exploring Open Education Resources**: A member expressed excitement about freely available Ivy League courses and noted that watching lectures from prestigious institutions like MIT and Stanford helped during their university studies. This was contextualized by other members noting the commonality of this practice.
- **Innovations and Discussions in Computer Science**: The discussion included a [professor's webpage](https://www.cs.cmu.edu/~dwoodruf/) from Carnegie Mellon University, which impressively spans back almost 7 years, showcasing contributions to fields like algorithms and machine learning.
- **Novel AI Research and Projects Shared**: Links to various AI-related projects and papers were shared, such as a [GitHub repository](https://github.com/trevorpogue/algebraic-nnhw) offering AI acceleration through matrix multiplication, and a white paper on Maisa's **Knowledge Processing Unit** ([Maisa KPU](https://maisa.ai/blog/kpu)), a framework aiming to leverage the power of Large Language Models (LLMs) for complex tasks.
- **Debate on 'ThoughtStream' Tokens in LLMs**: A member proposed the concept of adding 'ThoughtStream' tokens in Regular Language Models (LLMs) that allow for a stream of thoughts without affecting the loss function, sparking discussions about similar ideas in papers like Quiet-STaR and Feedback Transformers, both aiming to enhance reasoning in sequential tasks.
- **Potential Benefits and Challenges of Pause Tokens**: A conversation around incorporating pause tokens in models, intended to provide more compute to the transformers for better inference, gathered several insights about implementation challenges and comparisons to Universal Transformers and RNNs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/maisaAI_/status/1768657114669429103?s=20">Tweet from Maisa (@maisaAI_)</a>: Introducing Maisa KPU: The next leap in AI reasoning capabilities.  The Knowledge Processing Unit is a Reasoning System for LLMs that leverages all their reasoning power and overcomes their intrinsic ...</li><li><a href="https://x.ai/blog/grok">Announcing Grok</a>: no description found</li><li><a href="https://arxiv.org/abs/2403.09629">Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking</a>: When writing and talking, people sometimes pause to think. Although reasoning-focused works have often framed reasoning as a method of answering questions or completing agentic tasks, reasoning is imp...</li><li><a href="https://tenor.com/view/excited-fuego-gif-26833875">Excited Fuego GIF - Excited Fuego - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://maisa.ai/blog/kpu">KPU - Maisa</a>: AI-Powered Knowledge Processing Platform. A simple API for executing business tasks. Abstracting the complexities of using the latest AI architectures for software and app developers</li><li><a href="https://arxiv.org/abs/2002.09402">Addressing Some Limitations of Transformers with Feedback Memory</a>: Transformers have been successfully applied to sequential, auto-regressive tasks despite being feedforward networks. Unlike recurrent neural networks, Transformers use attention to capture temporal re...</li><li><a href="https://arxiv.org/abs/2203.07852">Block-Recurrent Transformers</a>: We introduce the Block-Recurrent Transformer, which applies a transformer layer in a recurrent fashion along a sequence, and has linear complexity with respect to sequence length. Our recurrent cell o...</li><li><a href="https://en.wikipedia.org/wiki/Wikipedia:Database_reports/Most_edited_articles_last_month">Wikipedia:Database reports/Most edited articles last month - Wikipedia</a>: no description found</li><li><a href="https://arxiv.org/abs/2312.12705">Optimizing Distributed Training on Frontier for Large Language Models</a>: Large language models (LLMs) have demonstrated remarkable success as foundational models, benefiting various downstream applications through fine-tuning. Recent studies on loss scaling have demonstrat...</li><li><a href="https://www.npr.org/sections/publiceditor/2009/08/19/112034424/free-transcripts-now-available-on-npr-org>">Free Transcripts now Available on NPR.org</a>: Transcripts of favorite, missed or maddening stories on NPR used to cost $3.95 each, but now they are free on NPR.org.</li><li><a href="https://www.youtube.com/watch?v=Sq1QZB5baNw),">Figure Status Update - OpenAI Speech-to-Speech Reasoning</a>: no description found</li><li><a href="https://github.com/pytorch/pytorch/issues/122123)">Issues · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - Issues · pytorch/pytorch</li><li><a href="https://aideadlin.es/?sub=ML,CG,NLP,RO,SP,DM,CV">AI Conference Deadlines</a>: no description found</li><li><a href="https://github.com/EleutherAI/cookbook/blob/main/calc/calc_transformer_flops.py">cookbook/calc/calc_transformer_flops.py at main · EleutherAI/cookbook</a>: Deep learning for dummies. All the practical details and useful utilities that go into working with real models. - EleutherAI/cookbook</li><li><a href="https://github.com/trevorpogue/algebraic-nnhw">GitHub - trevorpogue/algebraic-nnhw: AI acceleration using matrix multiplication with half the multiplications</a>: AI acceleration using matrix multiplication with half the multiplications - trevorpogue/algebraic-nnhw</li><li><a href="https://github.com/xai-org/grok-1">GitHub - xai-org/grok-1: Grok open release</a>: Grok open release. Contribute to xai-org/grok-1 development by creating an account on GitHub.</li><li><a href="https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/">RT-2: New model translates vision and language into action</a>: Introducing Robotic Transformer 2 (RT-2), a novel vision-language-action (VLA) model that learns from both web and robotics data, and translates this knowledge into generalised instructions for...</li><li><a href="https://www.cs.cmu.edu/~dwoodruf/">David P. Woodruff</a>: no description found
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1218100666493304852)** (245 messages🔥🔥): 

- **Debating Model Performance**: There's a heated discussion over the capabilities of **Grok**, a **314-billion parameter Mixture-of-Experts model** released by xAI. Skepticism arises around its effectiveness compared to its claimed parameter count and despite Git-release, real-world utility remains questionable with discussions highlighting limitations in independent evaluations and benchmarks.

- **Speculative Sampling in Transformers**: The conversation pivots towards speculative sampling efficiency in different model architectures. It's suggested that while speculative sampling, as discussed in a [PyTorch blog post](https://pytorch.org/blog/accelerating-generative-ai-2/), might offer gains for transformers, its applicability to models like **Mamba** is less clear due to different operation mechanics.

- **LLMs and Benchmarking**: There's widespread criticism of current benchmarking systems for LLMs with questions on overfitting given the best-of-N approach, where the results could simply be due to statistical variance. This skepticism extends to the efficacy of benchmarks as discriminative tools, especially for larger models like GPT-4 and Claude-3.

- **Behind Technical Discussions**: The chat hints at various technical aspects of LLMs such as the inefficiency of layerwise gradient operations in PyTorch and the compatibility issues with speculative decoding across models with differing layers or state dimensions. The discussion also touches upon the potential strategies like label smoothing and comparative FLOPs breakdown for MLPs vs attention layers.

- **A Company's Personnel and Product Relations**: There is a philosophical disagreement on whether a company’s personnel can predict the quality of its products, particularly within the context of AI models and startups. While some argue that capable teams can logically lead to superior outcomes, others remain skeptical, citing examples where this has not been the case and pushing for empirical evidence rather than reliance on authority.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Aaditya6284/status/1762558439354409345">Tweet from Aaditya Singh (@Aaditya6284)</a>: We study the effect of this choice in GPT-3.5 and GPT-4 – specifically, we look at the effect of tokenizing left-to-right (L2R) vs right-to-left (R2L), enforced by using delimiters such as commas. We ...</li><li><a href="https://x.ai/blog/grok">Announcing Grok</a>: no description found</li><li><a href="https://arxiv.org/abs/2403.06963">The pitfalls of next-token prediction</a>: Can a mere next-token predictor faithfully model human intelligence? We crystallize this intuitive concern, which is fragmented in the literature. As a starting point, we argue that the two often-conf...</li><li><a href="https://arxiv.org/abs/2403.06504">Adding NVMe SSDs to Enable and Accelerate 100B Model Fine-tuning on a Single GPU</a>: Recent advances in large language models have brought immense value to the world, with their superior capabilities stemming from the massive number of parameters they utilize. However, even the GPUs w...</li><li><a href="https://arxiv.org/abs/2403.09539">Logits of API-Protected LLMs Leak Proprietary Information</a>: The commercialization of large language models (LLMs) has led to the common practice of high-level API-only access to proprietary models. In this work, we show that even with a conservative assumption...</li><li><a href="https://arxiv.org/abs/2403.09611">MM1: Methods, Analysis &amp; Insights from Multimodal LLM Pre-training</a>: In this work, we discuss building performant Multimodal Large Language Models (MLLMs). In particular, we study the importance of various architecture components and data choices. Through careful and c...</li><li><a href="https://arxiv.org/abs/2402.18510">RNNs are not Transformers (Yet): The Key Bottleneck on In-context Retrieval</a>: This paper investigates the gap in representation powers of Recurrent Neural Networks (RNNs) and Transformers in the context of solving algorithmic problems. We focus on understanding whether RNNs, kn...</li><li><a href="https://arxiv.org/abs/2401.16380">Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling</a>: Large language models are trained on massive scrapes of the web, which are often unstructured, noisy, and poorly phrased. Current scaling laws show that learning from such data requires an abundance o...</li><li><a href="https://arxiv.org/abs/2403.10430">Construction of Arithmetic Teichmuller Spaces IV: Proof of the abc-conjecture</a>: This is a continuation of my work on Arithmetic Teichmuller Spaces developed in the present series of papers. In this paper, I show that the Theory of Arithmetic Teichmuller Spaces leads, using Shinic...</li><li><a href="https://arxiv.org/abs/2403.09394">GiT: Towards Generalist Vision Transformer through Universal Language Interface</a>: This paper proposes a simple, yet effective framework, called GiT, simultaneously applicable for various vision tasks only with a vanilla ViT. Motivated by the universality of the Multi-layer Transfor...</li><li><a href="https://arxiv.org/abs/2403.09635">Transformers Get Stable: An End-to-End Signal Propagation Theory for Language Models</a>: In spite of their huge success, transformer models remain difficult to scale in depth. In this work, we develop a unified signal propagation theory and provide formulae that govern the moments of the ...</li><li><a href="https://arxiv.org/abs/2403.04706">Common 7B Language Models Already Possess Strong Math Capabilities</a>: Mathematical capabilities were previously believed to emerge in common language models only at a very large scale or require extensive math-related pre-training. This paper shows that the LLaMA-2 7B m...</li><li><a href="https://pytorch.org/blog/accelerating-generative-ai-2/">Accelerating Generative AI with PyTorch II: GPT, Fast</a>: This post is the second part of a multi-series blog focused on how to accelerate generative AI models with pure, native PyTorch. We are excited to share a breadth of newly released PyTorch performance...</li><li><a href="https://arxiv.org/abs/2402.00691">Comparative Study of Large Language Model Architectures on Frontier</a>: Large language models (LLMs) have garnered significant attention in both the AI community and beyond. Among these, the Generative Pre-trained Transformer (GPT) has emerged as the dominant architecture...</li><li><a href="https://arxiv.org/abs/2403.07183">Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews</a>: We present an approach for estimating the fraction of text in a large corpus which is likely to be substantially modified or produced by a large language model (LLM). Our maximum likelihood model leve...</li><li><a href="https://bytez.com/read/arxiv/2403.07183">Bytez: Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews</a>: This study examines the use of large language models (LLMs), like ChatGPT, in scientific peer review. The authors developed a method to estimate the percentage of text in peer reviews that is generate...</li><li><a href="https://x.ai/blog/grok-os">Open Release of Grok-1</a>: no description found</li><li><a href="https://github.com/xai-org/grok">GitHub - xai-org/grok-1: Grok open release</a>: Grok open release. Contribute to xai-org/grok-1 development by creating an account on GitHub.</li><li><a href="https://github.com/enfiskutensykkel/ssd-gpu-dma">GitHub - enfiskutensykkel/ssd-gpu-dma: Build userspace NVMe drivers and storage applications with CUDA support</a>: Build userspace NVMe drivers and storage applications with CUDA support - enfiskutensykkel/ssd-gpu-dma</li><li><a href="https://github.com/bigscience-workshop/bloom-dechonk">GitHub - bigscience-workshop/bloom-dechonk: A repo for running model shrinking experiments</a>: A repo for running model shrinking experiments. Contribute to bigscience-workshop/bloom-dechonk development by creating an account on GitHub.</li><li><a href="https://artificialanalysis.ai/">Model &amp; API Providers Analysis | Artificial Analysis</a>: Comparison and analysis of AI models and API hosting providers. Independent benchmarks across key metrics including quality, price, performance and speed (throughput &amp; latency).
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1218832533517766666)** (11 messages🔥): 

- **Sensitivity of Scaling Laws to Data Complexity**: A member shared results indicating that **language model scaling laws** are affected by **data complexity**, with syntactic properties of a PCFG and gzip compression predicting dataset-specific scaling properties. This insight is being further explored with comprehensive experiments and the use of a particular package for fitted laws is anticipated.

- **Visual Representation Requires Clarity**: Members provided feedback on a graphical representation, highlighting issues with legibility and scale differences that made interpreting the data difficult.

- **Considerations in Measuring Perplexity and Loss**: The discussion evolved around perplexity, its relationship with intrinsic entropy, and its comparability across datasets with varying 'densities.' A suggestion was made that higher compressibility might indicate less information and this could influence the formulation of scaling laws.

- **Potential Applications for Data Preparation**: There was a notion that understanding how entropy differences between datasets affect scaling laws could inform approaches to crafting training data mixtures. This could be a valuable strategy in pretraining efficiency.

- **Lexical Density as a Key Factor in Data Compression**:
  One member pointed out that using compression metrics like gzip could serve as a method for filtering data with optimal lexical densities, which may be beneficial for efficient pretraining strategies.
  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1218288738728284241)** (13 messages🔥): 

- **Delving into N-Gram Statistics**: A member inquired about sampling strings from a *specified set of n-gram statistics*, wondering if there is a standard method for this process.
- **N-Gram Dependencies Explained**: It was clarified that specifying the statistics for a higher-order n-gram also determines the statistics for all lower-order grams, with **end-of-sequence and beginning-of-sequence token considerations** being mostly inconsequential.
- **Sampling Methodology Specified**: The conversation moved towards a solution, stating that sampling from the n-gram distribution can be done in an **autoregressive** manner to maintain the maximum entropy distribution.
- **Sampling Mechanics Broken Down**: The process involves starting with a sample from the unigram distribution, followed by sampling from the bigram conditional distribution, and so on, building the string one token at a time.
- **Script for Generating Bigrams Shared**: A GitHub link to a script for generating bigrams was shared, providing a practical implementation resource: [generate_bigrams.py](https://github.com/EleutherAI/features-across-time/blob/main/scripts/generate_bigrams.py).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Word_n-gram_language_model">Word n-gram language model - Wikipedia</a>: no description found</li><li><a href="https://github.com/EleutherAI/features-across-time/blob/main/scripts/generate_bigrams.py">features-across-time/scripts/generate_bigrams.py at main · EleutherAI/features-across-time</a>: Understanding how features learned by neural networks evolve throughout training - EleutherAI/features-across-time
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1218143473916575765)** (31 messages🔥): 

- **Newcomer Struggles with Llama Integration**: A user new to **lm-eval-harness** expressed difficulty in implementing **generate_until** and **log_likelihood** functions for their **Llama** model on **Gaudi2**. They were seeking a reference implementation or demo code, questioning inheritance of functions in sub-classes, and inquiring about the command-line tool's handling of hyperparameters for these functions.

- **Model Selection Conundrum in Main Branch**: Running the latest main branch version, a user faced a discrepancy where specifying a different model still resulted in using **gpt-2-small**. This was later resolved by the user discovering that duplicate **model_args** in the command caused the issue, with the first instance being ignored.

- **Questioning LLM Leaderboard Metrics**: A user queried about not being able to replicate the reported 69% score for **Llama2-70b** on the MMLU task, as showcased on the openLLM leaderboard, always achieving only between 62-64%. A response clarified that the open LLM leaderboard takes an unweighted average over MMLU subtasks, unlike the official implementation.

- **WMT14 Deadlock Issue on the Radar**: Several users reported an issue with a **deadlock** in the `wmt14-en-fr` task of the **lm-evaluation-harness**, leading to indefinite stalls during evaluation. Some explored solutions, noting concurrency might be implicated, while temporary workarounds included avoiding multiprocessing within that task.

- **Managing Cached LLMs**: Inquiring about the download location for **lm-harness models**, a user learned that models are likely in the Huggingface cache directory (`~/.cache`). Environment variables such as **HF_HOME**, **TRANSFORMERS_CACHE**, and **HF_DATASETS_CACHE** can control the directory's path.

- **New Release of lm-eval Harness**: Version 0.4.2 of **lm-eval** was launched and can be found on [PyPI](https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.2). This release includes various contributions from the community, and the team is open to providing reviews on pending PRs.

- **Perplexity and Rolling Windows Clarified**: The difference between `likelihood` and `likelihood_rolling` was discussed in relation to their use in **lm-evaluation-harness**. The `loglikelihood_rolling` method is suitable for evaluating perplexity in tasks like `wikitext`, while `loglikelihood` is meant for conditional likelihood evaluations and multiple-choice tasks.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/transformers/perplexity">Perplexity of fixed-length models</a>: no description found</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md">lm-evaluation-harness/docs/model_guide.md at main · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/">GitHub: Let’s build from here</a>: GitHub is where over 100 million developers shape the future of software, together. Contribute to the open source community, manage your Git repositories, review code like a pro, track bugs and fea...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/issues/1485">`wmt14-en-fr` deadlock issue · Issue #1485 · EleutherAI/lm-evaluation-harness</a>: While running evaluation on this task, during ter metric computation, the program gets stuck forever. The command: lm_eval --model hf --model_args pretrained=microsoft/phi-2,trust_remote_code=True ...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.2">Release v0.4.2 · EleutherAI/lm-evaluation-harness</a>: lm-eval v0.4.2 Release Notes We are releasing a new minor version of lm-eval for PyPI users! We&#39;ve been very happy to see continued usage of the lm-evaluation-harness, including as a standard test...</li><li><a href="https://github.com/huggingface/evaluate/blob/8dfe05784099fb9af55b8e77793205a3b7c86465/metrics/perplexity/perplexity.py">evaluate/metrics/perplexity/perplexity.py at 8dfe05784099fb9af55b8e77793205a3b7c86465 · huggingface/evaluate</a>: 🤗 Evaluate: A library for easily evaluating machine learning models and datasets. - huggingface/evaluate
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1219336845310038047)** (3 messages): 

- **Shuffling the Pile for Training**: One member inquired about whether [the Pile](https://pile.eleuther.ai/) was pre-shuffled and if it requires additional shuffling before pretraining. The response clarified that while the originally distributed files were not shuffled, the preprocessed and pretokenized data on HF is ready-to-go and was used by Pythia.
- **Data Organization and Train/Test Split**: Another member provided additional clarification stating that components of the Pile are definitely not shuffled, some due to being organized by date. However, the train/test/val split in the original Pile *might* be shuffled since the training data is in evenly-sized chunks, which suggests random sampling to achieve a diversified mix.
  

---



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1218173412522852483)** (193 messages🔥🔥): 

- **Managing OpenAI Services**: Users discussed how to manage API keys in the OpenAI dashboard and whether ChatGPT Team admins can view user chats. The [enterprise privacy policy](https://openai.com/enterprise-privacy) was linked to clarify how data ownership and control work.

- **Debating AI's Potential Risks and Morality**: A conversation unfolded around the morality and risks involved in prioritizing AI development. While some argued for embracing a future led by advanced intellects, others worried that dismissing human-centric values might not be prudent.

- **GPT vs. Copilot for Image Generation**: Users compared different aspects of OpenAI's ChatGPT+ and Microsoft's Copilot regarding image generation capabilities. Conversation points included quality comparison, content policies, image saving and editing features, and the usefulness of out-painting/in-painting tools.

- **Entering the AI and PyTorch Terrain**: Users engaged in a discussion on the minimum mathematical background needed to learn about AI and PyTorch. The consensus leaned towards having as strong a foundation as possible, with specific pointers towards calculus, vector math, and matrix operations.

- **ChatGPT vs. Claude Performance Debate**: Members of the community compared the conversation quality of GPT-4 with other models like Claude, discussing the nuances of their performance in different tasks and preferences for different use cases.

**Link mentioned**: <a href="https://openai.com/enterprise-privacy">Enterprise privacy</a>: no description found

  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1218428016573812888)** (34 messages🔥): 

- **Intrigue Around GPT-5's Arrival**: Members inquired about the release timeline of the next iteration, **GPT-5**, with questions emphasizing anticipation but no clear answers provided.
- **Integrating Web Search into GPT**: A user asked how to integrate web search functionality into the GPT API, similar to what **ChatGPT 4** can do, but did not receive a direct solution within the provided messages.
- **Customizing Chatbots on Mobile**: Members discussed the desire to create and customize an OpenAI chatbot via a mobile device, specifically mentioning tools like **botghost** for integrating with Discord, but no complete guidance was shared.
- **Technical Help for Code Generation with Playwright**: A member expressed difficulties getting GPT Turbo 3.5 to correctly generate **Playwright** code according to their method specifications, suspecting a lack of access to the latest Playwright libraries might be the issue.
- **GPT Unresponsiveness and Access to Support**: Users reported issues with GPT not responding to prompts and queried where to report such problems; a reference to OpenAI's support ([help.openai.com](https://help.openai.com)) was provided after some initial confusion on where bug reports could be filed.
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1218207072064114718)** (79 messages🔥🔥): 

- **Prompt Architecture Strategy Session**: A user sought advice for optimizing their classification use case with OpenAI, aiming for better recall and fewer false positives. Strategies suggested include checking the need for a custom GPT model and considering the total context window used.

- **GPT-3.5 Plays the Wrong Notes with Playwright**: A user was frustrated that GPT-3.5 Turbo was not producing executable code for Playwright tests, possibly due to lack of familiarity with the latest Playwright library. The suggestion to try GPT-4 for improved results was met with further discussion on managing context and chunking tasks across API calls for better performance.

- **Recalcitrant Responses from ChatGPT**: Users reported increasing instances where the model refuses to perform tasks, with suggestions to use meta-prompting as a workaround.

- **Concern Over Content Policies**: There was a debate on the model's stringent refusal patterns, with some interpreting it as a content policy issue which they hope OpenAI would relax.

- **Improving Multi-Query Web Searches**: A user raised a complex question about getting GPT to use multiple queries for more comprehensive web searches in a single response, though conversation threads around this topic showed some confusion around the difference between queries and sources.
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1218207072064114718)** (79 messages🔥🔥): 

- **Exploring Classification with AI**: A user is seeking advice on how to test and improve recall for a classification use case using OpenAI, considering how much context to provide in the prompt. Suggestions include monitoring the retrieval rate with context position and possibly using a stronger model.

- **Playwright Conundrum with Model Versions**: Users discuss difficulties in generating usable Playwright test code with GPT-3.5 turbo, speculating that the model may not be up to date with the latest Playwright libraries. Despite attempts at correcting formats, the issue persists, and the suggestion is to try GPT-4 or break tasks into chunks.

- **Handling Model Refusals**: There is frustration and curiosity regarding the frequency and rationale behind the AI refusing tasks, with some users noting an uptick in refusals. Suggestions range from meta-prompting to avoid content-related refusals, to providing clear examples of when refusals happen for better diagnosis.

- **Navigating Soft Warnings and Aggressive Algorithmic Bias Minimization**: Some members discuss their experiences with the AI's refusal to complete prompts and receiving "sorry I can't do that" responses. Various strategies, including prompt engineering, are proposed to circumvent this perception of increasing AI obstinacy.

- **Web Search Queries and Comprehensive Results**: Users debate the best way to prompt the AI to provide multiple sources and varied perspectives when conducting web searches. The discussion clarifies the difference between controlling search queries and the sources returned by those queries.
  

---



**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1218106794698739782)** (96 messages🔥🔥): 

- **Cross-encoder Query on Multi-GPU Utilization**: A member inquired about fine-tuning a cross-encoder model using multiple GPUs and asked which parameters need to be modified or included.
- **Gradio Interface Enhancement Call-to-action**: A community member announced a contribution to the Aya demo and opened a request for help in adding a Gradio interface slider via a PR.
- **Tech Talk on High-Power GPUs and CPUs**: Conversations unfolded around Nvidia's GH100 GPU, server CPUs on the same board, the power consumption going up to 850W, lead time for purchasing, and intricacies of chip cooling.
- **HuggingFace Data Leaderboard**: A member highlighted a new data leaderboard initiative on HuggingFace, sparking conversations on the extent and implications of data hosted on the platform.
- **Concerns and Limitations of Large Language Models**: Users discussed the challenges of slow token generation in models like airllm and the trade-offs between memory and token generation speed, with references to Github projects and the efficiency of larger models on consumer-grade GPUs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.ai/blog/grok-os">Open Release of Grok-1</a>: no description found</li><li><a href="https://academictorrents.com/details/5f96d43576e3d386c9ba65b883210a393b68210e">grok-1</a>: Grok-1 is a 314B parameter Mixture of Experts model - Base model (not finetuned) - 8 experts (2 active) - 86B active parameters - Apache 2.0 license - Code:  - Happy coding! p.s. we re hiring: </li><li><a href="https://www.phoronix.com/review/nvidia-gh200-gptshop-ben">Tweet from Linux Performance, Benchmarks &amp; Open-Source News - Phoronix</a>: no description found</li><li><a href="https://huggingface.co/spaces/ivrit-ai/whisper-large-v3-space">Whisper Large V3 - a Hugging Face Space by ivrit-ai</a>: no description found</li><li><a href="https://fxtwitter.com/Weyaxi/status/1768779404442739147">Tweet from Weyaxi (@Weyaxi)</a>: 🤔Have you ever wondered how much data we host on @huggingface?  Well, I did after seeing  @TheBlokeAI&#39;s model count and 120B models just chilling on the platform 😅  📊 So I scraped all repositor...</li><li><a href="https://huggingface.co/spaces/Tonic/Aya/discussions/3">Tonic/Aya · Set a repetition_penalty constant as 1.8</a>: no description found</li><li><a href="https://github.com/gradio-app/gradio/issues/7722">Video-LLaVA demo api not working with Gradio-Client · Issue #7722 · gradio-app/gradio</a>: Describe the bug Im trying to use the python api for the Video-LLaVA model demo on hugging face spaces but I get an error: Traceback (most recent call last): File &quot;/Users/kamakshiramamurthy/Deskt...</li><li><a href="https://github.com/moritztng/fltr">GitHub - moritztng/fltr: Like grep but for natural language questions. Based on Mistral 7B or Mixtral 8x7B.</a>: Like grep but for natural language questions. Based on Mistral 7B or Mixtral 8x7B. - moritztng/fltr
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1218115205553324112)** (12 messages🔥): 

- **Bayesian Optimization Confusion**: A member expressed confusion about **Bayesian Optimization** after listing it among different optimization techniques like GridSearch and RandomSearch. There was no further discussion or resolution provided on the topic.

- **What is Hugging Face? Help Needed**: A new member asked for help understanding what **Hugging Face** is and how to use it. Another member responded by explaining that Hugging Face offers tools for NLP, including the Transformers library, and referred them to the [main website](https://huggingface.co/) for more information.

- **Creating AI Vocal Duets**: A member inquired about making AI covers that involve duets and bands, expressing difficulty in achieving good quality. Another member recommended separately creating two high-quality individual voices using AI and then manually overlaying them.

- **Workshop Notebook Located**: A member requested a notebook pertaining to the "**MLOps: End-to-End Hugging Face Transformers with the Hub & SageMaker Pipelines**" workshop and then found it, sharing a [detailed blog post](https://www.philschmid.de/mlops-sagemaker-huggingface-transformers) outlining how to use Amazon SageMaker Pipelines for deploying Hugging Face transformers.

- **Error Accessing Hugging Face Model**: A member shared a snippet of Python code that resulted in a `404` error when trying to access a model on the Hugging Face hub, asking for help on how to access models locally. 
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co).">no title found</a>: no description found</li><li><a href="https://www.philschmid.de/mlops-sagemaker-huggingface-transformers">MLOps: End-to-End Hugging Face Transformers with the Hub &amp; SageMaker Pipelines</a>: Learn how to build an End-to-End MLOps Pipeline for Hugging Face Transformers from training to production using Amazon SageMaker.
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1218346001421570138)** (12 messages🔥): 

- **Fascination with Multilingual Model Performance**: The discussion touched on impressive multilingual model capabilities, particularly between highly divergent languages like **Chinese and English**. There was an expression of surprise at the model's ability to bridge such linguistically distinct languages.

- **Medusa's Parallel Predictions Peek Interest**: [Medusa, an efficient LLM inference method](https://arxiv.org/abs/2401.10774), sparked interest for its potential to improve performance by predicting multiple subsequent tokens in parallel. It aims to resolve the limitation of sequential token generation by utilizing additional decoding heads to verify multiple candidate continuations simultaneously.

- **Language Dominance Could Skew Multilingual Models**: Concerns were raised about the possibility of an **English bias** in multilingual models, potentially skirting authentic language patterns and cognitive associations unique to other languages.

- **Multimodal Models on the Horizon**: Enthusiasm for recent work on Multimodal Large Language Models (MLLMs) was shared, specifically the recipe for building such models by balancing pre-training data sources. The mentioned paper emphasized important architecture components like image encoders and data choices that significantly impact state-of-the-art few-shot results ([link to paper](https://huggingface.co/papers/2403.09611)).

- **Peer Reviews Potentially Modified by LLMs**: A study indicated that between 6.5% and 16.9% of peer review text for AI conferences could have been modified by LLMs. The paper suggests that these modifications are more likely to occur in reviews submitted late, with lower confidence, or by reviewers who might not respond to author rebuttals, pointing toward a need for interdisciplinary exploration of LLM impact on information practices ([link to paper](https://arxiv.org/abs/2403.07183)).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2401.10774">Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads</a>: The inference process in Large Language Models (LLMs) is often limited due to the absence of parallelism in the auto-regressive decoding process, resulting in most operations being restricted by the m...</li><li><a href="https://huggingface.co/papers/2403.09611">Paper page - MM1: Methods, Analysis &amp; Insights from Multimodal LLM Pre-training</a>: no description found</li><li><a href="https://arxiv.org/abs/2403.07183">Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews</a>: We present an approach for estimating the fraction of text in a large corpus which is likely to be substantially modified or produced by a large language model (LLM). Our maximum likelihood model leve...</li><li><a href="https://bytez.com/read/arxiv/2403.07183">Bytez: Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews</a>: This study examines the use of large language models (LLMs), like ChatGPT, in scientific peer review. The authors developed a method to estimate the percentage of text in peer reviews that is generate...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1218158991570636900)** (18 messages🔥): 

- **Seeking Advice on NL2SQL Pipeline**: A channel member is working on a **NL2SQL pipeline**, facing issues with accuracy. They're using BAAI/llm-embedder, TheBloke/nsql-llama-2-7B-GGUF, and FAISS, seeking recommendations for better embedding and NL2SQL models.

- **Hype Over NVIDIA's Marvel**: Excitement bubbles up as a chat member introduces the **Nvidia GH200 Grace Hopper Superchip**, indicative of strides in computing power and efficiency, yet further details like links or in-depth discussion did not ensue.

- **Guidance Request for NLP Beginners**: Members have recommended various resources for starting NLP, including [HuggingFace's NLP Course](https://huggingface.co/learn/nlp-course/chapter1/1), the latest textbook manuscript by [Jurafsky from Stanford](https://web.stanford.edu/~jurafsky/slp3/), and [Stanford's cs224n course notes](https://web.stanford.edu-Class-cs224n).

- **Looking for LLM API for Production Use**: A member queried about **free LLM APIs** suitable for deployment in production, to which another suggested ollama for local deployment but left queries about production-focused solutions open for further advice.

- **A Conformer Model for ASR and LoRA Training Query Goes Unanswered**: There were inquiries about finding a tutorial for training a conformer model for ASR, as well as the best practices to train a LoRA quickly, but no responses followed the questions.

**Link mentioned**: <a href="https://huggingface.co/learn/nlp-course/chapter1/1">Introduction - Hugging Face NLP Course</a>: no description found

  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1218217429868478474)** (7 messages): 

- **Interactive Documents Elevate RAG Performance**: A new approach for the **Retrieval-Augmented Generation (RAG)** pipeline is suggested: treating each retrieved document not just as text, but as an interactive tool. This method allows for more complex and dynamic interactions with large documents. [Enhanced RAG interactions on Twitter](https://twitter.com/llama_index/status/1768658182308794421)

- **LlamaIndex Unveils Instrumentation Module**: LlamaIndex releases version 0.10.20, introducing an **Instrumentation module**. The update is showcased through notebooks demonstrating observability features and API call monitoring. [Announcement of LlamaIndex v0.10.20](https://twitter.com/llama_index/status/1768730443921396220)

- **Advancing QA with Search-in-the-Chain**: A paper presents a new method called **Search-in-the-Chain** that integrates retrieval and planning for enhanced question-answering. It allows for real-time verification and adjustment during the answering process. [Search-in-the-Chain paper highlight](https://twitter.com/llama_index/status/1769035278063399208)

- **Matching Resumes to Jobs with RAG**: A blog post by Kyosuke Morita demonstrates how to utilize **LlamaParse** alongside **LlamaIndex** to create a job matching assistant that can efficiently extract relevant information from complex CV formats. [LlamaParse CV matching blog post](https://twitter.com/llama_index/status/1769147791002264008)

- **Memory Tool Integration in Assistant APIs**: MemGPT, an agentic architecture, is introduced in a webinar, designed to enhance an agent's memory functions for read/write operations to "core" memory. This innovation aims to empower assistant APIs with function calling memory. [MemGPT webinar tweet](https://twitter.com/llama_index/status/1769408792633229455)
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://t.co/GY4unUYOwl">llama_index/docs/examples/instrumentation/basic_usage.ipynb at main · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://t.co/E1d9dtkqAI">llama_index/docs/examples/instrumentation/observe_api_calls.ipynb at main · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1218113300764819488)** (303 messages🔥🔥): 

- **Chaining OpenAI Agents Query**: A member inquired about chaining multiple OpenAI agents and faced an ["invalid_request_error"](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/tools/root.html) when attempting. Discussion suggested possible code examples were required to assist further.
- **Xinference Support in LlamaIndex**: Assistance was sought for using **Xinference CPU Cluster** with LlamaIndex. Links to [local deployment guides](https://docs.llamaindex.ai/en/latest/examples/llm/xinference_local_deployment.html) and the [GitHub page](https://github.com/jerryjliu/llama_index/blob/main/docs/examples/llm/xinference_local_deployment.ipynb) were provided, though specific instructions for cluster deployment were not found in the shared resources.
- **Adding a Node Postprocessor to RetrieverQueryEngine**: A member asked how to incorporate a *node_postprocessor* with a `RetrieverQueryEngine`, citing the process involved defining a `node_postprocessor`, such as `KeywordNodePostprocessor`, and adding it via the `from_args` method.
- **Troubleshooting a ToolRetrieverRouterQueryEngine Use Case**: Discussions included technical issues when trying to use a `ToolRetrieverRouterQueryEngine` with a `FunctionTool`, where the solution involved creating an `agent` with the `TransationsToolIndex` before employing it as a `QueryEngineTool` for the `RouterQueryEngine`.
- **Multi-Modal LLMs Differentiation and Legacy Package Maintenance**: A member discussed the challenge of integrating multimodal content into LLMs. Concerns were raised about potential maintenance burdens and the possibility of API changes leading to necessary reimplementation.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://127.0.0.1:9997>">no title found</a>: no description found</li><li><a href="http://localhost:{port}">)">no title found</a>: no description found</li><li><a href="https://www.promptingguide.ai/techniques/fewshot">Prompt Engineering Guide</a>: A Comprehensive Overview of Prompt Engineering</li><li><a href="https://www.promptingguide.ai/techniques/rag">Prompt Engineering Guide</a>: A Comprehensive Overview of Prompt Engineering</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_documents.html">Defining and Customizing Documents - LlamaIndex 🦙 v0.10.20.post1</a>: no description found</li><li><a href="https://qdrant.tech/documentation/tutorials/llama-index-multitenancy/">Multitenancy with LlamaIndex - Qdrant</a>: Qdrant is an Open-Source Vector Database and Vector Search Engine written in Rust. It provides fast and scalable vector similarity search service with convenient API.</li><li><a href="https://docs.llamaindex.ai/en/stable/use_cases/extraction.html">Structured Data Extraction - LlamaIndex 🦙 v0.10.20.post1</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/api/llama_index.core.node_parser.CodeSplitter.html">CodeSplitter - LlamaIndex 🦙 v0.10.20.post1</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/multi_modal/image_to_image_retrieval.html">Image to Image Retrieval using CLIP embedding and image correlation reasoning using GPT4V - LlamaIndex 🦙 v0.10.20.post1</a>: no description found</li><li><a href="https://cloud.llamaindex.ai">LlamaCloud</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/5c53f41712785e5558156372bdc4f33a6326fa5f/docs/examples/vector_stores/Qdrant_using_qdrant_filters.ipynb">llama_index/docs/examples/vector_stores/Qdrant_using_qdrant_filters.ipynb at 5c53f41712785e5558156372bdc4f33a6326fa5f · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="http://localhost:{port}",>">no title found</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/tools/root.html">Tools - LlamaIndex 🦙 v0.10.20.post1</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/llms/llama-index-llms-ollama/llama_index/llms/ollama/base.py">llama_index/llama-index-integrations/llms/llama-index-llms-ollama/llama_index/llms/ollama/base.py at main · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://github.com/hofstadter-io/hof/blob/_dev/flow/chat/prompts/dm.cue">hof/flow/chat/prompts/dm.cue at _dev · hofstadter-io/hof</a>: Framework that joins data models, schemas, code generation, and a task engine. Language and technology agnostic. - hofstadter-io/hof</li><li><a href="https://github.com/run-llama/llama_index/issues/12034">[Question]: custom llm but is blocked · Issue #12034 · run-llama/llama_index</a>: Question Validation I have searched both the documentation and discord for an answer. Question the code is from typing import Optional, List, Mapping, Any from llama_index.core import SimpleDirecto...
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1218542835754860564)** (4 messages): 

- **Tutorial for RAG Implementation**: A link to a [YouTube video](https://youtu.be/w7Ap6gZFXl0) was shared providing a step-by-step guide on creating an effective Retriever-And-Generator (RAG) using **LlamaParse**, **Qdrant**, and **Groq**.

- **Inquiry for RAG Preparation Tips**: A member asked for the top five tips on preparing a document for RAG and how to automatically add metadata to **Pinecone** for optimal retrieval.

- **Exploring AI Assistants with RAG**: A Medium article titled "[Empowering Voices: AI Assistant with RAG Pipeline, Memory, and LlamaIndex](https://medium.com/ai-advances/empowering-voices-ai-assistant-with-rag-pipeline-memory-and-llamaindex-11c4e319d915)" was shared, exploring the use of RAG pipelines in conjunction with **LlamaIndex**.

- **Troubleshooting RAPTOR RAG with Hugging Face**: A member shared their code for using **Hugging Face** models within a **RAPTOR pack** for RAG implementation and sought advice on resolving errors encountered when adapting from OpenAI models to **Hugging Face** models.

**Link mentioned**: <a href="https://youtu.be/w7Ap6gZFXl0">RAG with LlamaParse, Qdrant and Groq | Step By Step</a>: In this video, I will show you how to create a effective RAG with LlamaParse, Qdrant and Groq. I will explain what LlamaParse is and briefly walk you through...

  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1218154073912639508)** (202 messages🔥🔥): 

- **Yann LeCun's Views on LLMs Spark Debate**: Discussions center around [Yann LeCun's perspective](https://x.com/kk_slider_k_/status/1768464173657158132?s=46&t=6FDPaNxZcbSsELal6Sv7Ug) on Language Models (LLMs). A parallel is drawn between 'shape rotators' and 'wordcels', discussing the use of language for reasoning versus visuospatial thinking, sparked by the notion that *Yann has no inner monologue*.

- **Anticipating Quantum Jumps in LLMs**: The community discusses the potential advancements in GPT-5 and beyond, with some quoting Sam Altman on the significance of such progress and the dangers of underestimating it. There’s speculation on the sort of advancements GPT-5 might bring, and OpenAI's lead in development is noted as a possible indicator.

- **NVIDIA's GTC Keynote Generates Buzz**: There's excitement and discussion around Nvidia's GTC, with [Jensen Huang's keynote expected to reveal](https://www.youtube.com/watch?v=USlE2huSI_w) significant AI advancements. There’s a particular reference to "trillion param chatbots", hinting at large-scale LLMs.

- **OpenAI's Sama Appears on Lex Fridman Podcast**: The channel mentions a [Lex Fridman](https://youtu.be/jvqFAi7vkBc?si=WTfgLyNfGhkP2Azx) podcast featuring Sam Altman from OpenAI. Members show disappointment in the lack of "alpha" or insights, especially relating to OpenAI's direction and Ilya Sutskever, and there are jokes about challenging Lex to a climbing podcast.

- **Grok-1 Model Release Leads to Mixed Reactions**: The open-source release of *Grok-1* by xAI, a large **314B parameter Mixture-of-Experts model**, leads to discussions on its capabilities compared to other LLMs. The community reviews the model's potential and the probable motivation behind releasing it on Pi Day, given the numerical pun relating to its parameter count.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.ai/blog/grok-os">Open Release of Grok-1</a>: no description found</li><li><a href="https://x.com/teknium1/status/1768452864995942469?s=46&t=6FDP">Tweet from Teknium (e/λ) (@Teknium1)</a>: This explains why Yann is so bearish on LLMs... 😲</li><li><a href="https://arxiv.org/abs/2402.10171">Data Engineering for Scaling Language Models to 128K Context</a>: We study the continual pretraining recipe for scaling language models&#39; context lengths to 128K, with a focus on data engineering. We hypothesize that long context modeling, in particular \textit{t...</li><li><a href="https://x.com/teortaxestex/status/1769460562763604375?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: @aidan_mclau 0) Rocket man bad 1) it&#39;s not much worse 2) As you can see it&#39;s a sparse-upcycled Grok-0. It&#39;s undercooked. In 2023, continual pretraining has been ≈solved, and having validat...</li><li><a href="https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space">Explaining the SDXL latent space</a>: no description found</li><li><a href="https://x.com/altryne/status/1768683178888208816?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Alex Volkov (Thursd/AI) (@altryne)</a>:   Sora team showing up at Berkley to talk about SORA</li><li><a href="https://huggingface.co/collections/suno/bark-6502bdd89a612aa33a111bae">Bark - a suno Collection</a>: no description found</li><li><a href="https://substack.recursal.ai/p/eaglex-17t-soaring-past-llama-7b">🦅 EagleX 1.7T : Soaring past LLaMA 7B 2T in both English and Multi-lang evals (RWKV-v5)</a>: A linear transformer has just cross the gold standard in transformer models, LLaMA 7B, with less tokens trained in both English and multi-lingual evals. A historical first.</li><li><a href="https://x.com/swyx/status/1769776691562324215?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from swyx (@swyx)</a>: how is it possible to have a 2hr conversation with sama and get zero alpha  but hey we talked about aliens again thats fun</li><li><a href="https://x.com/openinterpreter/status/1769448726660337875?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Open Interpreter (@OpenInterpreter)</a>: 100 years in the making. 100 hours to go.</li><li><a href="https://x.com/repligate/status/1769241542420738126?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from j⧉nus (@repligate)</a>: this was the result of navigating to the ../../microsoft/bing/bing_chat directory in claude&#39;s backrooms, then letting claude use commands to look around on its own, then running:  &lt;cmd_soul&gt;...</li><li><a href="https://x.com/Francis_YAO_/status/1759986097365627054?s=20">Tweet from Yao Fu (@Francis_YAO_)</a>: Frontier models all have at least 100k context length, Gemini 1.5 has even 1m context. What about research and open source?   Introducing Long Context Data Engineering, a data driven method achieving ...</li><li><a href="https://x.com/burny_tech/status/1769549895835226613?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Burny — Effective Omni (@burny_tech)</a>: New details about GPT-5 from Sam Altman He’s basically admitting that GPT-5 will be a massive upgrade from GPT-4, so we can expect a similar jump from 3 to 4. &#34;&#34;If you overlook the pace of imp...</li><li><a href="https://x.com/xlr8harder/status/1769454853506638008?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from xlr8harder (@xlr8harder)</a>: I think I speak for everyone here when I say: 314 billion parameters what the hell</li><li><a href="https://x.com/granawkins/status/1768530196557365599?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Grant♟️ (@granawkins)</a>: &#34;Between Q1-24 and Q4-25, there will be a 14x increase in compute.  Then, if you factor in algorithmic efficiency doubling every 9 months, the effective compute at the end of next year will be alm...</li><li><a href="https://www.nfx.com/post/ai-like-water">Tweet from AI Is Like Water</a>: Generative AI is like water. The phrase was borne out of frustration, but it opens up a new world of AI playbooks.</li><li><a href="https://x.com/joshwalkos/status/1767745681375015076?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Champagne Joshi (@JoshWalkos)</a>: This is a fascinating conversation with a girl who lacks an internal monologue. She articulates the experience quite well.</li><li><a href="https://x.com/teknium1/status/1768452864995942469?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Teknium (e/λ) (@Teknium1)</a>: This explains why Yann is so bearish on LLMs... 😲</li><li><a href="https://x.com/kk_slider_k_/status/1768464173657158132?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from KZ (@kzSlider)</a>: This makes so much sense. Yann’s always been looking for models that reason visually or using planning rather than purely in language  ↘️ Quoting Teknium (e/λ) (@Teknium1)   This explains why Yann is ...</li><li><a href="https://x.com/francis_yao_/status/1769575936994013611?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Yao Fu (@Francis_YAO_)</a>: Grok&#39;s MMLU is only on par with Mixtral, despite one order of magnitude larger. I believe it has great potential but not fully released, and good continue pretrain data may substantially lift the ...</li><li><a href="https://www.youtube.com/watch?v=USlE2huSI_w">WATCH: Jensen Huang&#39;s Nvidia GTC Keynote - LIVE</a>: Tune in at 1:00pm PT / 4:00pm ET when Nvidia CEO Jensen Huang kicks off its biannual GTC conference.Never miss a deal again! See CNET’s browser extension 👉 ...</li><li><a href="https://x.com/emmanuel_2m/status/1768360522028876045?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Emm (@emmanuel_2m)</a>: 🚨 Today, we&#39;re excited to launch the Scenario #UPSCALER! Elevate your AI creations up to 10k resolution.  🚀 Built for unmatched #CreativeControl & guided workflows.  💰 It starts at just $15/mo ...</li><li><a href="https://youtu.be/jvqFAi7vkBc?si=WTfgLyNfGhkP2Azx">Sam Altman: OpenAI, GPT-5, Sora, Board Saga, Elon Musk, Ilya, Power &amp; AGI | Lex Fridman Podcast #419</a>: Sam Altman is the CEO of OpenAI, the company behind GPT-4, ChatGPT, Sora, and many other state-of-the-art AI technologies. Please support this podcast by che...</li><li><a href="https://youtu.be/I-HMKky7Qsw?si=yCvekF3a0zr_1IgA&t=718">Beyond Transformers - Intro to RWKV Architecture &amp; The World To... Eugene Cheah &amp; Harrison Vanderbyl</a>: Beyond Transformers - Intro to RWKV Architecture &amp; The World Tokenizer - Eugene Cheah &amp; Harrison Vanderbyl, Recursal AIWhats comes next after transformers?In...</li><li><a href="https://youtu.be/J0p_thJJnoo?si=IaGuEgUcs1BRgjhF">#51 FRANCOIS CHOLLET - Intelligence and Generalisation</a>: In today&#39;s show we are joined by Francois Chollet, I have been inspired by Francois ever since I read his Deep Learning with Python book and started using th...</li><li><a href="https://github.com/FranxYao/Long-Context-Data-Engineering">GitHub - FranxYao/Long-Context-Data-Engineering: Implementation of paper Data Engineering for Scaling Language Models to 128K Context</a>: Implementation of paper Data Engineering for Scaling Language Models to 128K Context - FranxYao/Long-Context-Data-Engineering</li><li><a href="https://x.com">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter</li><li><a href="https://www.nvidia.com/gtc/?ncid=ref-inor-332714">GTC 2024: #1 AI Conference</a>: Register now. Streamed online. March 18-21, 2024.</li><li><a href="https://docs.google.com/document/d/1HZ326V6KNK4QIlG7uEldQEizFgTaO7Hg9uJxURYy9f8/edit">NVIDIA &amp; Harpreet Sahota GTC 2024</a>: no description found</li><li><a href="https://buttondown.email/ainews/archive/ainews-mm1-apples-first-large-multimodal-model/">[AINews] MM1: Apple&#x27;s first Large Multimodal Model</a>: AI News for 3/14/2024-3/15/2024. We checked 358 Twitters and 20 Discords (332 channels, and 2839 messages) for you. Estimated reading time saved (at 200wpm):...</li><li><a href="https://arxiv.org/abs/2402.10588">Do Llamas Work in English? On the Latent Language of Multilingual Transformers</a>: We ask whether multilingual language models trained on unbalanced, English-dominated corpora use English as an internal pivot language -- a question of key importance for understanding how language mo...</li><li><a href="https://bytez.com/read/arxiv/2402.10588">Bytez: Do Llamas Work in English? On the Latent Language of Multilingual Transformers</a>: In this research study, scientists wanted to know if language models (that can generate text) use English as a &quot;pivot&quot; language internally, even when prompted in other languages. They found ...</li><li><a href="https://huggingface.co/collections/stereoplegic/multilingual-65389b21be39573b3b2db98d">Multilingual - a stereoplegic Collection</a>: no description found</li><li><a href="https://x.com/danielhanchen/status/1769550950270910630?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Daniel Han (@danielhanchen)</a>: Had a look through @Grok&#39;s code: 1. Attention is scaled by 30/tanh(x/30) ?! 2. Approx GELU is used like Gemma 3. 4x Layernoms unlike 2x for Llama 4. RMS Layernorm downcasts at the end unlike Llama...
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1218137415068422164)** (2 messages): 

- **Paper Club Session Alert**: The **Paper Club** session on "A Comprehensive Summary Of Large Language Models" is happening in two minutes. All are invited to join and participate.

- **Artificial Melody Maker**: An AI powered by [Suno](https://app.suno.ai/song/83680b6f-db37-44de-adf9-3f7fff6b79d9) has created a song mimicking '90s hip-hop complete with lyrics that reflects on how AI models are challenging human artists by being trained on extensive datasets. The lyrics provide a meta-commentary on AI's growing influence in the creative domain.

**Link mentioned**: <a href="https://news.ycombinator.com/item?id=39746163">Suno, an AI music generator | Hacker News</a>: no description found

  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1218135292574306328)** (20 messages🔥): 

- **Decoding the Genesis of Attention**: A member clarified that **attention mechanisms** were created as a solution for models to access all information in input sequences, unlike with previous models which had fixed-length encoding vectors.

- **Parallelization Perks Explained**: It was explained that the **transformer attention mechanism** promotes parallelization by allowing independent processing of different tokens, leading to more efficient computation and faster training.

- **Clarity on Computational Efficiency**: A member acknowledged that their confusion about **parallelization** was resolved, understanding now that scaled dot product operations in transformers eliminate the sequential 'wait' in processing.

- **Closing Gratitude for Insights**: Participants expressed their thanks for the session which offered deeper understanding of the development and intuition behind **large language models (LLMs)**.
  

---


**Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1218287754715201638)** (36 messages🔥): 

- **Passive Attendance at the AI Club**: A club member mentioned being in an IRL meeting and tuning in passively to the ongoing discussion.
- **Members Exchange Greetings**: Several users greeted each other with brief messages in the chat, indicating active engagement.
- **In-Depth Blog to Come**: One user hinted at a detailed version of their discussion to be posted later on their personal blog.
- **Useful RAG Technique Resources Shared**: A user shared a link to a [Towards Data Science blog post](https://towardsdatascience.com/advanced-rag-01-small-to-big-retrieval-172181b396d4) discussing advanced RAG techniques.
- **Collaborative Learning Document Cited**: There was mention of a [Google spreadsheet](https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0) detailing upcoming topics and resources for AI-in-action club sessions.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://towardsdatascience.com/advanced-rag-01-small-to-big-retrieval-172181b396d4">Advanced RAG 01: Small-to-Big Retrieval</a>: Child-Parent RecursiveRetriever and Sentence Window Retrieval with LlamaIndex</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0">AI In Action: Weekly Jam Sessions</a>: 2024  Topic,Date,Facilitator,Resources,@dropdown UI/UX patterns for GenAI,1/26/2024,nuvic,&lt;a href=&quot;https://maggieappleton.com/squish-structure&quot;&gt;https://maggieappleton.com/squish-struct...
</li>
</ul>

</div>
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1218220293865345024)** (168 messages🔥🔥): 

- **DALL-E 3 Dataset Location Update**: A dataset once queried about its removal from Huggingface was actually moved to a new location; the DALL-E 3 dataset can be accessed [here](https://huggingface.co/datasets/OpenDatasets/dalle-3-dataset).
  
- **Regarding Loading Datasets via Huggingface**: It's been highlighted how you can load datasets by specifying a commit id using the `load_dataset()` function, [detailed in the Hugging Face documentation](https://huggingface.co/docs/datasets/en/loading#hugging).

- **A Discussion on Grok Model and Comparisons with Mixtral**: There's been chatter around Grok, a 314B-parameter model, with opinions on its performance relative to Grok's heavyweight nature and comparisons with the smaller but capable Mixtral model. A link was shared to the Grok-1 Github repository, [found here](https://github.com/xai-org/grok-1).

- **Innovations in Captioning with Cog Model**: Users are sharing strategies to improve caption accuracy by incorporating metadata into prompts when using the Cog model. One user has shared a script from a repository available on GitHub, [check out the script here](https://github.com/victorchall/EveryDream2trainer/blob/main/caption_cog.py).

- **Discussions Around LLaMA 1.6 34b Model and COG VLM**: Users are discussing various AI models, particularly LLaMA and COG, with a focus on captioning abilities, inference speeds, and practical usability on consumer grade GPUs like the RTX 3090.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/imgn_ai/status/1769791182270333067">Tweet from imgnAI (@imgn_ai)</a>: catgirls are at NVIDIA GTC ✨  meowing for your creative freedom 👊  this is a message that needs to be heard 🐱💕</li><li><a href="https://tenor.com/view/silicon-valley-yes-cheer-think-gif-9010547">Silicon Valley Yes GIF - Silicon Valley Yes Cheer - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.economist.com/business/2023/11/23/why-chinese-companies-are-flocking-to-mexico">Why Chinese companies are flocking to Mexico</a>: The country offers a back door to the United States</li><li><a href="https://huggingface.co/docs/datasets/en/loading#hugg">Load</a>: no description found</li><li><a href="https://huggingface.co/docs/datasets/en/loading#hugging-face-hub">Load</a>: no description found</li><li><a href="https://github.com/victorchall/EveryDream2trainer/blob/main/caption_cog.py">EveryDream2trainer/caption_cog.py at main · victorchall/EveryDream2trainer</a>: Contribute to victorchall/EveryDream2trainer development by creating an account on GitHub.</li><li><a href="https://github.com/xai-org/grok-1">GitHub - xai-org/grok-1: Grok open release</a>: Grok open release. Contribute to xai-org/grok-1 development by creating an account on GitHub.</li><li><a href="https://www.reddit.com/r/aiwars/comments/1bbxtp6/the_people_behind_the_nightshade_glaze_account/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/datasets/OpenDatasets/dalle-3-dataset">OpenDatasets/dalle-3-dataset · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1218181910669295716)** (13 messages🔥): 

- **Web UIs Discussed in The Wrong Channel**: Members briefly mentioned the risks of using web UIs, noting that they can't be employed with the free version of Colab.
- **Generative Audio Video Text World Model Document Shared**: A link to a Google Doc titled "Generative Audio Video Text world model" was posted but no details or further discussion about the content of the document were provided. [Access the document here.](https://docs.google.com/document/d/1f6CpVjdApmQl3nXsUACGtSd9nML3CypI1iE889i4JbM/edit?usp=drivesdk)
- **Paper on Efficient Continual Learning of LLMs Shared**: A link to an arXiv paper on continual pre-training of large language models (LLMs), discussing methods to adapt to new data while retaining performance on previous data. [Read the abstract or download the paper](https://arxiv.org/abs/2403.08763).
- **Grok-1 Open Release on GitHub**: A GitHub link to a project called Grok-1 was shared; appears to be an open release of the named software by xai-org. [Check out the GitHub repository here.](https://github.com/xai-org/grok-1)
- **Speculation on GPT-4's Architecture**: Discussions abound regarding the architecture of GPT-4, with claims supported by Nvidia sources labeling it as a MoE model with 1.8 trillion parameters. No confirmation was supplied whether it's GPT-4 or another model. [See the speculative tweet image.](https://pbs.twimg.com/media/GI-reRIW0AAZpMC?format=jpg&name=large)
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2403.08763">Simple and Scalable Strategies to Continually Pre-train Large Language Models</a>: Large language models (LLMs) are routinely pre-trained on billions of tokens, only to start the process over again once new data becomes available. A much more efficient solution is to continually pre...</li><li><a href="https://arxiv.org/abs/2403.09611">MM1: Methods, Analysis &amp; Insights from Multimodal LLM Pre-training</a>: In this work, we discuss building performant Multimodal Large Language Models (MLLMs). In particular, we study the importance of various architecture components and data choices. Through careful and c...</li><li><a href="https://github.com/xai-org/grok-1">GitHub - xai-org/grok-1: Grok open release</a>: Grok open release. Contribute to xai-org/grok-1 development by creating an account on GitHub.</li><li><a href="https://docs.google.com/document/d/1f6CpVjdApmQl3nXsUACGtSd9nML3CypI1iE889i4JbM/edit?usp=drivesdk">Generative Audio Video Text world model</a>: no description found
</li>
</ul>

</div>
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1218118852454383667)** (99 messages🔥🔥): 

- **Navigating Chat Formats and Conversions**: One exchange discussed the efficacy of training on chat data, with a focus on whether to retain the chat format or convert to a Q/A structure. The member was informed about using the sharegpt format and subsequent conversion to llama chat model through Axolotl using specific attributes during training.

- **Axolotl Eases Model Finetuning Processes**: Axolotl is highlighted for simplifying the finetuning process by allowing for configurations via a yaml file instead of scripting, and it supports the use of LoRA. Despite concerns about loss of control compared to traditional methods, Axolotl provides a user-friendly alternative with less setup overhead.

- **The Promise of NVIDIA's RTX 5000 Series**:
    - Rumors suggest considerable performance gains for the upcoming NVIDIA RTX 5000 series, plotted for release around 2025, with notable improvements like 50% more VRAM and 78% increased bandwidth, potentially benefitting AI model training.
    - A link to a [news article](https://www.heise.de/news/GeForce-RTX-5000-Geruechte-zu-Nvidias-naechster-Grafikkartengeneration-9655220.html) discussing the RTX 5000 series was shared, speculating on its significance for consumer-grade training.
    - Additional details on potential memory configurations were debated, with insights from a [TechPowerUp article](https://www.techpowerup.com/320185/nvidia-geforce-rtx-50-series-blackwell-to-use-28-gbps-gddr7-memory-speed) discussing GDDR7 memory speeds for the new series.

- **Comparing Quantization Techniques and Model Integrations**: The discussion touched upon available quantization methods, like AQML, and their integration with various models. A link to the AQML GitHub repository was shared ([GitHub - AQLM](https://github.com/Vahe1994/AQLM)), along with comments on its efficiency compared to other methods.

- **Exploring Grok-1, MoE, and Inference Strategies**:
    - The community reacted to the release of Grok-1 model weights, pondering the capabilities and potential hardware requirements to run this large-scale MoE model. The discussion involved the implications of its MoE architecture on VRAM usage and inference speeds, with Sequoia being mentioned as a potential solution for optimizing inference on consumer GPUs.
    - Points were raised about the possibility of offloading or caching systems as strategies for handling the substantial resource demands of models like Grok-1.
    - Insights into NVIDIA's GTC announcements, including a tease about the GPT-4 model's parameter count of 1.8T, were shared through a YouTube link to the event ([GTC March 2024 Keynote with NVIDIA CEO Jensen Huang](https://www.youtube.com/watch?v=Y2F8yisiS6E)).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/BrivaelLp/status/1769482175005577571?s=20">Tweet from Brivael (@BrivaelLp)</a>: Zuck just reacted to the release of Grok, and he is not really impressed.  &#34;314 billion parameter is too much. You need to have a bunch of H100, and I already buy them all&#34; 🤣</li><li><a href="https://www.together.ai/blog/sequoia">Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding</a>: no description found</li><li><a href="https://tenor.com/view/wizard-cat-magus-cat-witch-cat-wicca-wiccan-gif-26941843">Wizard Cat Magus Cat GIF - Wizard Cat Magus Cat Witch Cat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.techpowerup.com/320185/nvidia-geforce-rtx-50-series-blackwell-to-use-28-gbps-gddr7-memory-speed">NVIDIA GeForce RTX 50-series &quot;Blackwell&quot; to use 28 Gbps GDDR7 Memory Speed</a>: The first round of NVIDIA GeForce RTX 50-series &quot;Blackwell&quot; graphics cards that implement GDDR7 memory are rumored to come with a memory speed of 28 Gbps, according to kopite7kimi, a reliabl...</li><li><a href="https://www.youtube.com/watch?v=Y2F8yisiS6E">GTC March 2024 Keynote with NVIDIA CEO Jensen Huang</a>: Watch NVIDIA CEO Jensen Huang’s GTC keynote to catch all the announcements on AI advances that are shaping our future.Dive into the announcements and discove...</li><li><a href="https://www.heise.de/news/GeForce-RTX-5000-Geruechte-zu-Nvidias-naechster-Grafikkartengeneration-9655220.html">GeForce RTX 5000: Gerüchte zu Nvidias nächster Grafikkartengeneration</a>: Nvidias nächste große Gaming-GPU könnte mehr und schnelleren Speicher bekommen – zusammen mit mehr Shader-Kernen.</li><li><a href="https://github.com/xai-org/grok">GitHub - xai-org/grok-1: Grok open release</a>: Grok open release. Contribute to xai-org/grok-1 development by creating an account on GitHub.</li><li><a href="https://github.com/Vahe1994/AQLM">GitHub - Vahe1994/AQLM: Official Pytorch repository for Extreme Compression of Large Language Models via Additive Quantization https://arxiv.org/pdf/2401.06118.pdf</a>: Official Pytorch repository for Extreme Compression of Large Language Models via Additive Quantization https://arxiv.org/pdf/2401.06118.pdf - Vahe1994/AQLM</li><li><a href="https://www.techpowerup.com/320185/nvidia-geforce-rtx-50-s">NVIDIA GeForce RTX 50-series &quot;Blackwell&quot; to use 28 Gbps GDDR7 Memory Speed</a>: The first round of NVIDIA GeForce RTX 50-series &quot;Blackwell&quot; graphics cards that implement GDDR7 memory are rumored to come with a memory speed of 28 Gbps, according to kopite7kimi, a reliabl...
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1218207901873606667)** (24 messages🔥): 

- **ScatterMoE Optimizes MoE Models**: The Axolotl development team is excited about the potential of **ScatterMoE**, an optimization over the Huggingface implementation, promising significant improvements in throughput. Further details and code can be seen on their [GitHub branch](https://github.com/OpenAccess-AI-Collective/axolotl/tree/scatter_moe).

- **Clarifying Training with ScatterMoE**: A member clarified that to train with the new ScatterMoE, one would need to use a **mixtral** model type, but emphasized that the system hasn't been thoroughly tested for training correctness yet.

- **Required Upgrade to PyTorch 2.2**: There's a discussion about needing to upgrade **axolotl** to PyTorch version **2.2** or higher to benefit from the new kernels and resolve compatibility issues; some members confirmed using or testing PyTorch **2.2.1**.

- **Grok Weight Performance Under Scrutiny**: The **Grok** model's weights (**314 Billion parameters**) were tried with Axolotl, but one user remarked on the less than impressive performance relative to the model's size. There's curiosity about who would run such a model considering its resource demands.

- **Introducing Grok Into Axolotl**: In light of the **Grok** model release discussions, a user jokes about whether Axolotl's **qLoRA FSDP** might eventually handle this "monster", while another points out that only the **int8 version** of the model has been released as per the [Grok GitHub page](https://github.com/xai-org/grok-1).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1407">implement post training by ehartford · Pull Request #1407 · OpenAccess-AI-Collective/axolotl</a>: Does this look right?</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1407/commits/9c221a6761195c9739c02e11f9fe864bc947e53b">implement post training by ehartford · Pull Request #1407 · OpenAccess-AI-Collective/axolotl</a>: Does this look right?</li><li><a href="https://github.com/xai-org/grok-1">GitHub - xai-org/grok-1: Grok open release</a>: Grok open release. Contribute to xai-org/grok-1 development by creating an account on GitHub.</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/tree/scatter_moe">GitHub - OpenAccess-AI-Collective/axolotl at scatter_moe</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1218257987445981234)** (35 messages🔥): 

- **Tokenizer Confusion Leads to Tagging Trouble**: A member is experiencing an issue where the fine-tuned model often omits the first `<summary>` tag or inserts it with a leading space. They've checked tokenization behavior expecting `<summary>` but observing `▁<summary>` and are concerned about a potential tokenizer problem.

- **Local Model and Data Mismatch**: One user is new to LLM and wants to adjust their config file to use local model and training data instead of pulling from Huggingface, leading them through a series of trial and error with path specifications and facing `HFValidationError` issues.

- **Training Data Conversations Cause Chaos**: Another member is struggling with 'index out of range' errors when fine-tuning conversation data, with configurations like `one_shot` and `alpaca` not working as expected due to empty "role" arrays in their dataset.

- **Readme to the Rescue for Configuration Confusion**: In addressing the above issue, they were advised to verify the prompt strategies mentioned in the readme and found that empty "from" and "value" fields in their dataset were causing the problem, resolving it by mapping additional roles and ignoring conversations with length less than 2.

- **Evaluation Set Size Inconsistency**: A bug is flagged where Axolotl is citing an evaluation set as too small for sample packing during a 2-epoch run but deems it fine for a 10-epoch run, even though the eval set is independent and should not vary with the number of epochs.
  

---


**OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1218770755920072767)** (8 messages🔥): 

- **NVIDIA NeMo Curator Introduced**: A member shared a [GitHub link](https://github.com/NVIDIA/NeMo-Curator) to **NVIDIA NeMo-Curator**, a scalable toolkit for data curation. However, no further discussion or personal experiences with the toolkit were provided.
- **Seeking a Specialized Mistral FT**: One queried if anyone possesses or has knowledge of a **Mistral** model fine-tuned on both the *orca-math-word-problems-200k* dataset and *nvidia/OpenMathInstruct-1*, highlighting interest in combining reasoning with coding capabilities.
- **Considering Model Merging with mergekit**: In response to whether a merge kit could avoid individual training of **Mistral** on extensive datasets, another member affirmed that a merge kit would be a good choice, provided the chat formats are aligned.
- **Curiosity About Model Format Compatibility for Merging**: The conversation evolved with a question about the possibility of fine-tuning a subsection of one model to align the chat format, revealing an adaptability interest in model merging tactics.

**Link mentioned**: <a href="https://github.com/NVIDIA/NeMo-Curator">GitHub - NVIDIA/NeMo-Curator: Scalable toolkit for data curation</a>: Scalable toolkit for data curation. Contribute to NVIDIA/NeMo-Curator development by creating an account on GitHub.

  

---


**OpenAccess AI Collective (axolotl) ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/)** (1 messages): 

duh_kola: Is it possible to use different lora adapter to do dpo on another model
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1218310691103178803)** (43 messages🔥): 

- **Shining Light on Photonics**: A member shared a [YouTube video](https://youtu.be/8ohh0cdgm_Y) titled "New Chip Breakthrough: x1000 Boost with Light and Radio waves", also pointing to a company called Lightmatter that focuses on photonic computer chips aiming to power AI more efficiently.
- **Photonic Insights with Asianometry**: In a discussion about photonics technology, a member recommended two educational videos from Asianometry, available on [YouTube](https://www.youtube.com/watch?v=29aTqLvRia8) and [YouTube](https://www.youtube.com/watch?v=t0yj4hBDUsc), covering silicon photonics and light meshes for neural networks.
- **PyTorch Tensor Management by Design**: The explicit tensor memory management in PyTorch was debated, with members discussing the complications of hiding memory copies in TensorFlow. A [GitHub gist](https://gist.github.com/robieta/4c6e94f25a2ab87330bb6bd8026074a6) demonstrated TensorFlow's behavior with cross-device tensors.
- **Seeking the Latest GPU Facilities**: Cloud GPU services like [RunPod](https://www.runpod.io/) and [Lambda Labs](https://lambdalabs.com/) were suggested for profiling kernel operations on newer GPUs, though members mentioned issues with profiling permissions on these platforms.
- **GTC 2024 Hints at New Horizons**: NVIDIA's CEO Jensen Huang's keynote at GTC 2024 sparked conversations about the future of AI models and hardware, discussing a Sota model with 1.8 trillion parameters and new B100 hardware with 192GB HBM.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.cerebras.net/product-chip/">Product - Chip - Cerebras</a>: no description found</li><li><a href="https://www.runpod.io/">Rent Cloud GPUs from $0.2/hour</a>: no description found</li><li><a href="https://www.youtube.com/live/Y2F8yisiS6E?si=g5MChTXs3a9gGykE">GTC March 2024 Keynote with NVIDIA CEO Jensen Huang</a>: Watch NVIDIA CEO Jensen Huang’s GTC keynote to catch all the announcements on AI advances that are shaping our future.Dive into the announcements and discove...</li><li><a href="https://lambdalabs.com/">GPU Cloud, Clusters, Servers, Workstations | Lambda</a>: GPU Cloud, GPU Workstations, GPU Servers, and GPU Laptops for Deep Learning &amp; AI. RTX 4090, RTX 3090, RTX 3080, RTX A6000, H100, and A100 Options. Ubuntu, TensorFlow, and PyTorch Pre-Installed.</li><li><a href="https://youtu.be/8ohh0cdgm_Y?si=q3wOMlzp_Nmn8_AJ">New Chip Breakthrough: x1000 Boost with Light and Radio waves</a>: Get TypeAI PREMIUM now! Start your FREE trial by clicking the link here:  https://bit.ly/Mar24AnastasiInTechThe paper: https://www.nature.com/articles/s41586...</li><li><a href="https://lightmatter.co/">Lightmatter®</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=29aTqLvRia8">Silicon Photonics: The Next Silicon Revolution?</a>: My deepest thanks to friend of the channel Alex Sludds of MIT for suggesting this topic and helping me with critical resources. Check him out here: https://a...</li><li><a href="https://www.youtube.com/watch?v=t0yj4hBDUsc">Running Neural Networks on Meshes of Light</a>: I want to thank Alex Sludds for his efforts in helping me research and produce his video. Check out his work here: https://alexsludds.github.ioLinks:- The As...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1218241351582482493)** (7 messages): 

- **Introducing Triton Debugging Visualizer**: A member announced the creation of a visualizer aimed at simplifying the process of debugging in Triton by showing the spatial structure of load/stores. The tool was designed to assist when implementing complex functions, though a preview of the visualizer interface was not provided in the message.
- **New: Triton Puzzles for Learning and Testing**: A set of [Triton Puzzles](https://colab.research.google.com/drive/1AJc8RFsDeJ3Vx3gRq5dUqmcb-Cy1G8qh?usp=sharing) was shared, aimed at providing a challenging but educational experience for those familiar with GPU puzzles. There are currently two known bugs: occasional double visualizations and segmentation faults.
- **Guidance for Triton Newbies**: In response to a query for Triton learning resources, members suggested that, beyond the official tutorials, the new Triton Puzzles could help, and there was a suggestion to examine and annotate popular Triton kernels from the community to aid understanding.
- **Encouragement for Triton CPU Debugging**: A member expressed enthusiasm regarding the interpreter that runs Triton on the CPU, highlighting it as a useful feature for those without immediate access to a GPU.
- **Community Engagement with Triton Puzzles**: The community members showed interest in engaging with the new Triton Puzzles, recognizing their potential usefulness, and one made a minor correction to the text, suggesting an edit for clarity.

**Link mentioned**: <a href="https://colab.research.google.com/drive/1AJc8RFsDeJ3Vx3gRq5dUqmcb-Cy1G8qh?usp=sharing">Google Colaboratory</a>: no description found

  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1218467001450627072)** (68 messages🔥🔥): 

- **Diving into Warp Schedulers and Thread Efficiency**: A member asked about configuring the number of warp schedulers and understanding how many threads each can control in CUDA to optimize execution efficiency and occupancy, but no specific answers or resources were provided in the messages.

- **Clarity Sought on Active Warps**: A member inquired about the definition of "active warp" within CUDA and whether a warp with no active threads can still be considered active, suggesting that for exercise purposes, "active warp" should mean a warp with at least one active thread.

- **Decoding Memory Managers in CUDA**: The member [@morousg#cudapassion](https://link.to.morousgprofile) clarified the intentions behind providing multiple memory management options in CUDA, highlighting strategies such as "Producer Provides" and "Consumer Takes," to facilitate efficient data management among different memory spaces.

- **Understanding Provide-Take Semantics in Memory Management**: A detailed discussion ensued between members regarding the semantics of memory management when using "Produces" and "Takes," exploring how these options can influence memory allocation and the possible need for streamSynchronization in CUDA applications.

- **Keen Interest in Memory Management for Pipeline Parallel Inference**: At the conclusion of a talk, a member expressed a deep interest in applying memory management strategies to improve a pipeline parallel implementation for large language model (LLM) inference, discussing potential solutions with [@morousg#cudapassion](https://link.to.morousgprofile), including asynchronous copying and optimizing GPU usage.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=Y2F8yisiS6E">GTC March 2024 Keynote with NVIDIA CEO Jensen Huang</a>: Watch NVIDIA CEO Jensen Huang’s GTC keynote to catch all the announcements on AI advances that are shaping our future.Dive into the announcements and discove...</li><li><a href="https://github.com/tspeterkim/flash-attention-minimal">GitHub - tspeterkim/flash-attention-minimal: Flash Attention in ~100 lines of CUDA (forward pass only)</a>: Flash Attention in ~100 lines of CUDA (forward pass only) - tspeterkim/flash-attention-minimal
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[suggestions](https://discord.com/channels/1189498204333543425/1189868872887705671/1219091487455711414)** (5 messages): 

- **Exploring the Nexus of Hardware and ML**: A user shared a YouTube link to [Prof. Mohamed Abdelfattah's research group](https://www.youtube.com/@mabdelfattah88) at Cornell University, which focuses on reconfigurable computing and efficient machine learning.
- **Deep Dive into Optimizing ML for Hardware**: The [ECE 5545 (CS 5775) course](https://abdelfattah-class.github.io/ece5545/) page was highlighted for providing a hardware-centric viewpoint on machine learning systems and their optimizations, from microcontrollers to multi-GPU systems.
- **Course Textbook Mystery**:
  - A user pointed out the oddity of the course website mentioning "the textbook" but not specifying which textbook it refers to.
  - Another user clarified that the textbook details are provided in the first lecture video.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://abdelfattah-class.github.io/ece5545/">ML Hardware and Systems</a>: no description found</li><li><a href="https://www.youtube.com/@mabdelfattah88">Prof. Mohamed Abdelfattah</a>: This is the channel for Prof. Mohamed Abdelfattah&#39;s research group at Cornell University. We are researching reconfigurable computing and efficient machine learning. For more information check out...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/)** (1 messages): 

vim410: Depends. But yes.
  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1219389682241110147)** (5 messages): 

- **Solid CUDA skills as a foundation for ML**: The member's strong background in CUDA for GPU computing, including experience with memory coalescing, warp divergence, and kernel profiling, appears to be a **solid foundation** for branching into machine learning with CUDA.
- **Recommendation for diving into ML/DL**: It's suggested to start experimenting with a deep learning framework like **PyTorch**, as ML essentially involves optimization techniques, such as matrix multiplications and normalizations.
- **Programming Massively Parallel Processors—The Must-Have Book**: A specific book titled **"Programming Massively Parallel Processors"** is recommended for deepening CUDA knowledge, and is praised as an excellent resource, although it is noted to contain limited content on deep learning.
- **Learning from the Luminaries**: Following **Andrej Karpathy's Zero to Hero series** is mentioned as part of a good learning path for ML concepts alongside exploring CUDA-focused lectures.

**Link mentioned**: <a href="https://www.amazon.de/-/en/Wen-mei-W-Hwu/dp/0323912311">no title found</a>: no description found

  

---


**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1218146385942286407)** (6 messages): 

- **Understanding Stride Multiplication in CUDA Indexing**: A member initially had a doubt regarding the use of `i = blockIdx.x * blockDim.x + threadIdx.x * 2` for CUDA indexing from chapter 2 question 2. Another member explained that this approach erroneously leads to double counting the index `i`, with an example showing two different threads yielding the same index value.

- **Caution Advised on Sharing Instructor Content**: A member raised a concern about whether certain content might be restricted to instructors only. This was in response to a discussion about the appropriateness of blogging exercise answers.

- **Blogging Exercise Answers: A Dilemma**: One member expressed their intention to blog their answers to the exercises due to a lack of response from the authors, highlighting the struggle of not having an educational address for correspondence.

- **Awaiting Authorial Guidance on Sharing Answers**: It was suggested that the appropriateness of blogging exercise answers is uncertain, and further guidance would be sought from Wen-mei, presumably an author or authority related to the content in question.
  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1218239914542366790)** (14 messages🔥): 

- **Busy Week for a Member**: A member briefly expressed that they are **very busy** this week and will update when their schedule clears up.
- **Searching for Code**: One user was looking for code and managed to find a **Triton kernel** on GitHub, providing the link to **[Ring-Flash-Attention commit](https://github.com/zhuzilin/ring-flash-attention/commit/10d992c3c84a2ee1a2e47dd596615d9aad46f7d5)**.
- **Blog Post Conundrum**: A member writing a blog post on **ring-attention** sought clarification on why memory requirements are said to scale linearly with block size in related papers, despite the squared chunk size memory need in SRAM.
- **Looking for Answers**: In response to the confusion about memory scaling, another member suggested examining **[flash-attention source code](https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_kernel.h)**, especially how FlashAttention is possibly implemented without forming a matrix of size c^2.
- **Clarification on Memory Requirement Language**: Additional members joined the discussion, one suggesting that memory requirements might be meant to scale linearly with the number of blocks, rather than with block size itself.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2311.09431">Striped Attention: Faster Ring Attention for Causal Transformers</a>: To help address the growing demand for ever-longer sequence lengths in transformer models, Liu et al. recently proposed Ring Attention, an exact attention algorithm capable of overcoming per-device me...</li><li><a href="https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_kernel.h">flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h at main · Dao-AILab/flash-attention</a>: Fast and memory-efficient exact attention. Contribute to Dao-AILab/flash-attention development by creating an account on GitHub.</li><li><a href="https://github.com/zhuzilin/ring-flash-attention/commit/10d992c3c84a2ee1a2e47dd596615d9aad46f7d5">add naive triton kernel for varlen · zhuzilin/ring-flash-attention@10d992c</a>: no description found
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1218332053032927322)** (5 messages): 

- **MLSys 2024 Conference Alert**: A member shared information about the MLSys 2024 conference in May, highlighting its focus on interdisciplinary collaboration at the intersection of Machine Learning and Systems. The conference is deemed significant for the era of AI, particularly in developing efficient AI systems. [Check out the conference](https://mlsys.org/).
- **A Poetic Perspective on Conference Taglines**: An observation was made that the phrase "The Conference for the Era of AI" fits the rhythm of an iambic pentameter.
- **Smartphone Woes**: A user humorously referred to a smartphone as "Not so smart phone," possibly indicating some frustration or issue with the device.
- **Math Operation Order Clarification**: There was a discussion correcting the sequence of operations in mathematical expressions, emphasizing that multiplication and division should be carried out from left to right.
- **Scientific Calculator Debate**: The conversation about math operations extended to how scientific calculators interpret certain expressions differently, indicating there may be variations in computation outcomes based on calculator design.

**Link mentioned**: <a href="https://mlsys.org/">MLSys 2024</a>: no description found

  

---


**CUDA MODE ▷ #[gtc-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1218444664315711498)** (9 messages🔥): 

- **Early Bird Gets the Worm**: *marksaroufim* mentions planning to be at the event from Monday morning and is open to meetups, offering to DM their phone number for coordination.
- **Long-Haul Attendee**: *neurondeep* states they will be attending GTC from 14th to 25th March and plans to go on all days of the event.
- **Meetup Enthusiast**: *_t_vi_* expresses their presence at the event and shows interest in meeting up with others.
- **Fully Booked Schedule**: *marksaroufim* originally planned to attend GTC for 1-2 days but has decided to stay for the entire week, influenced by the compelling schedule and the availability of decent wifi.
- **GTC FOMO**: *mr.osophy* shares a humorous sentiment about not being able to attend GTC, along with a past failed attempt to volunteer for a free pass.

**Link mentioned**: <a href="https://www.youtube.com/watch?v=Sfrjpy5cJCs">I Snuck Into A Secret Arms-Dealer Conference</a>: Get an exclusive video every month at https://www.patreon.com/Boy_BoyWe made this in collaboration with the legendary Australian political satire group The C...

  

---



**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1218183723200155748)** (159 messages🔥🔥): 

- **Approval for Llama Message Format**: A user inquired if the format incorporating "system", "user", and "assistant" is suitable for llama models, which got confirmed as acceptable.
- **Exploring Payment Methods**: When asked about the necessity of connecting a credit card and how to make payments, it was clarified that users need to *topup their balance*.
- **Discussing Model Choice for Roleplay Consistency**: Users debated on which model performs best for roleplay without repeating or spitting nonsense, with **Sonnet** ultimately highlighted as the top choice for its consistency.
- **Prompt Format Guidance for Models**: After querying how to use system messages for guiding Large Language Models (LLMs) beyond the first message, members discussed limitations, providing insights that typically only the first system message is employed, with subsequent instruction potentially embedded in user messages.
- **Development Intentions and Affiliations**: Users tackled diverse topics ranging from setting up public APIs and being listed on platforms, to affiliate programs and model choices, while discussing the cost and efficiency of various models and the flexibility of the OpenRouter API.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai">OpenRouter</a>: A router for LLMs and other AI models</li><li><a href="https://github.com/xai-org/grok-1">GitHub - xai-org/grok-1: Grok open release</a>: Grok open release. Contribute to xai-org/grok-1 development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1218212402127175711)** (95 messages🔥🔥): 

- **API Choices for LangChain Agents**: A member questioned whether **astream_log** is preferred over **astream_events** and whether the latter being in beta signifies a move towards deprecation or if they're simply distinct APIs.
- **Beta Testers Wanted for Research Assistant**: Users are being sought for beta testing an advanced research assistant and search engine built by a member, featuring premium access to models like **Claude 3 Opus**, **GPT-4 Turbo**, and **Mistral Large**. Interested individuals are directed to a [waitlist page](https://rubiks.ai/) for the service named **Rubik's AI**.
- **Collaboration and Feedback for LangChain Docs**: Several members expressed difficulty in navigating through **LangChain documentation**, specifically for beginners, with offers from others to help clarify or add missing pages.
- **Structured Output Parsing with LangChain**: Members discussed ways to retrieve structured outputs using **LangChain and pydantic** with examples of code provided for parsing complex data structures. Users shared snippets and offered help to those trying to implement similar features in their projects.
- **Beta Testing Appeal for New Service**: A member is calling for beta testers for a new service that provides rapid access generators (RAG) for applications or personal documents, committing to a week-long coding spree to finalize the platform.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://rubiks.ai/">Rubik's AI - Waitlist</a>: no description found</li><li><a href="https://codelabs.developers.google.com/codelabs/gemini-function-calling#4.">no title found</a>: no description found</li><li><a href="https://bloon.ai">Bloon AI</a>: Redefining Intelligent Learning</li><li><a href="https://github.com/langchain-ai/langchain/discussions/19239">Feature Request: Support for Negative Embeddings in Similarity Searches · langchain-ai/langchain · Discussion #19239</a>: Checked I searched existing ideas and did not find a similar one I added a very descriptive title I&#39;ve clearly described the feature request and motivation for it Feature request I propose adding ...</li><li><a href="https://www.teradata.com/insights/ai-and-machine-learning/using-natural-language-to-query-teradata-vantagecloud-with-llms">Using Natural Language to Query Teradata VantageCloud With LLMs| Teradata</a>: Learn to translate your English queries into SQL and receive responses from your analytic database in plain English.
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1219304272244510741)** (45 messages🔥): 

- **Streaming Woes with RemoteRunnable**: A member is **experiencing issues with streaming output** from a `RemoteRunnable` in LangChain. The member notes that when calling from Python, streaming works properly, but the **equivalent JavaScript code always triggers an `/invoke` call** instead of `/stream`.

- **Potential Inheritance Issue in Streaming Sequence**: The member questions whether the problem arises from `RunnableSequence` inheriting a default `_streamIterator` from `Runnable`, which invokes an `invoke` call. The member suggests that this could be causing the streaming function to fail in JavaScript.

- **Seeking Help from the LangChain Team**: When asked how to report the issue to the LangChain team, the AI instructs to open an issue on **GitHub** or **email the team** for support.

- **No Recent Changes Noted**: There is **no mention of recent changes** made in the past month that could have resolved the JavaScript streaming issue. For updates, members are advised to check LangChain's GitHub commits and release notes.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://api.js.langchain.com/classes/langchain_core_runnables_remote.RemoteRunnable.html#pipe>):">RemoteRunnable | LangChain.js - v0.1.28</a>: no description found</li><li><a href="https://js.langchain.com/docs/security#reporting-a-vulnerability>).">Security | 🦜️🔗 Langchain</a>: LangChain has a large ecosystem of integrations with various external resources like local and remote file systems, APIs and databases. These integrations allow developers to create versatile applicat...</li><li><a href="https://github.com/langchain-ai/langchain/issues/13126>)),">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/11998>)),">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/13723>)).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/17315>)).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1218223379690029179)** (11 messages🔥): 

- **AI Chatbot for Conversational Data Analysis Unveiled**: Haste171 released a [GitHub project](https://github.com/Haste171/langchain-chatbot) featuring an AI chatbot designed to analyze and extract information from data in a conversational format.

- **Bookmarks Come to Life with AI**: Codegriot has created a Discord AI chatbot for managing Raindrop.io bookmarks with the goal of finding them relevant later. The bot is presented as open source and available on [GitHub](https://github.com/uogbuji/living-bookmarks).

- **AI Scraping Made Simpler**: VinciGit00 developed an AI-based scraper using langchain, which operates with OpenAI keys and is planned to be compatible with other models. With over 2300 installations in under a month, they encourage support by starring the [GitHub repo](https://github.com/VinciGit00/Scrapegraph-ai).

- **Personalized Nutrition AI App Showcased**: Esxr_ shared a [YouTube video](https://youtu.be/vHjc5CEoIJE) demonstrating Nutriheal, an AI app for personalized patient care, leveraging tools for local hosting and data privacy. Further insights are available on their website [navvy.co](https://navvy.co/).

- **AI-Driven Sales Development Representative**: Sivasurend took on a Twitter challenge to automate the role of an SDR/AE using the Lyzr Automata framework. Detailed methods were demonstrated on Twitter and the source code is accessible on their [GitHub page](https://github.com/LyzrCore/lyzr-automata).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://calendly.com/neurofusion/30min">User Interview 🔎 - NEUROFUSION Research, Inc.</a>: Hey, I&#39;m building a digital advisor to help improve how you show up to work and other areas of your life. I&#39;d love to speak with you to learn about your needs around productivity, physical and...</li><li><a href="https://medium.com/@bsouleymane78/staying-up-to-date-with-latest-advancements-on-ai-applied-to-financial-industry-using-ai-b995da14800f">Staying up to date with latest advancements on AI applied to financial industry using AI</a>: Automatize analysis of latests scientific papers with AI to keep an eye on latests advancements on the sector.</li><li><a href="https://github.com/Haste171/langchain-chatbot">GitHub - Haste171/langchain-chatbot: AI Chatbot for analyzing/extracting information from data in conversational format.</a>: AI Chatbot for analyzing/extracting information from data in conversational format. - Haste171/langchain-chatbot</li><li><a href="https://github.com/VinciGit00/Scrapegraph-ai">GitHub - VinciGit00/Scrapegraph-ai: Python scraper based on AI</a>: Python scraper based on AI. Contribute to VinciGit00/Scrapegraph-ai development by creating an account on GitHub.</li><li><a href="https://youtu.be/vHjc5CEoIJE">Making an AI application in 15 minutes</a>: Stack- Custom UI and RAG: A tweaked version of Open-webui- Local LLM Hosting: Ollama for locally hosted LLMs.- Data Privacy: Integrates Pebblo by DaxaAI to e...</li><li><a href="https://navvy.co/.">Home</a>: I’m deeply passionate about AI. Let’s connect to unlock AI’s potential and collaborate on innovative projects!</li><li><a href="https://x.com/siva_1gc/status/1768997890544800070?s=20">Tweet from Siva Surendira (@siva_1gc)</a>: It took a bit more time than we thought.. But here it is.. 😎  Automation of SDR & AE function with @lyzrai Automata and @OpenAI... Runs on @awscloud - secure and private..  How it works? 👇  Agent 1:...</li><li><a href="https://github.com/LyzrCore/lyzr-automata">GitHub - LyzrCore/lyzr-automata: low-code multi-agent automation framework</a>: low-code multi-agent automation framework. Contribute to LyzrCore/lyzr-automata development by creating an account on GitHub.</li><li><a href="https://amzn.eu/d/3Dcdsbk">Die Reise vom Ego zur Seele in einem holistischen Universum: Die Rolle der Meditation, der Naturerfahrung und der Astronomie bei der Transformation (10.000 Follower TikTok Content dank ChatGPT 2) eBook : Schulze, Carsten, Bing, chatgpt, google, Bard: Amazon.de: Kindle-Shop</a>: no description found</li><li><a href="https://amzn.eu/d/2uVnCp8">no title found</a>: no description found</li><li><a href="https://www.facebook.com/casi.schulze.10">no title found</a>: no description found
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1218824643436085321)** (2 messages): 

- **AI App Development Made Effortless**: A member showcased the creation of a personalized nutrition AI application called *Nutriheal* utilizing **Ollama**, **Open-webui**, and **Langchain's Pebblo integration** by Daxa AI. The member highlighted the ease of creating an AI application with a [tutorial video](https://youtu.be/vHjc5CEoIJE) and shared their portfolio at [navvy.co](https://navvy.co/).

- **Local AI Deployment Demystified**: Tutorials on how to set up and run sophisticated AI models locally bust the myth that AI is exclusively for tech giants, as demonstrated in blog posts like [Build and Deploy GenAI Solutions Locally](//build-and-deploy-genai-solutions-locally) and [Local LLMs - Making a Generic UI for Custom LLM Assistants](/generic-ui-for-custom-llm-assistants).

- **Plan-and-Execute with Langgraph**: A video tutorial was shared that demonstrates how to create a "plan-and-execute" style agent using **Langgraph**, inspired by the Plan-and-Solve paper and the Baby-AGI project. Viewers can watch and learn from the [YouTube video](https://www.youtube.com/watch?v=ZlJbaYQ2hm4) provided.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=ZlJbaYQ2hm4">Plan-and-Execute using Langgraph</a>: how to create a &quot;plan-and-execute&quot; style agent. This is heavily inspired by the Plan-and-Solve paper as well as the Baby-AGI project.The core idea is to firs...</li><li><a href="https://youtu.be/vHjc5CEoIJE">Making an AI application in 15 minutes</a>: Stack- Custom UI and RAG: A tweaked version of Open-webui- Local LLM Hosting: Ollama for locally hosted LLMs.- Data Privacy: Integrates Pebblo by DaxaAI to e...</li><li><a href="https://navvy.co/.">Home</a>: I’m deeply passionate about AI. Let’s connect to unlock AI’s potential and collaborate on innovative projects!
</li>
</ul>

</div>
  

---



**Interconnects (Nathan Lambert) ▷ #[other-papers](https://discord.com/channels/1179127597926469703/1179142630517518397/1218217772765544448)** (8 messages🔥): 

- **APIs Expose LLM Secrets**: A paper on [arXiv](https://arxiv.org/abs/2403.09539) reveals that proprietary large language models (LLMs) can leak significant information through their API outputs. The leakage is attributed to a softmax bottleneck, enabling the discovery of model architecture details for “under $1,000” cost in OpenAI's gpt-3.5-turbo case.
  
- **LLM Size Underestimation Debate**: A member expresses surprise at the estimated 7 billion parameter size for a model discussed in the recent paper, suggesting that the actual parameter count could be higher.

- **Skepticism Over Model Size Estimates**: Skepticism arises as some members suggest the LLM size estimate might be incorrect, especially if the model in question, assuming GPT 3.5, utilizes a Mixture of Experts (MoE) architecture.

- **Mixture Model Mechanism Speculated**: A conversation speculates about a possible mechanism in turbo models, drawing parallels to a past paper where starting tokens from a larger model boosted subsequent performance from a smaller model.
  
- **The Intricacies of Modeling Performance**: It's suggested that *"Mixtral"*, another LLM, has a notably high embedding dimension (4096), indicating the complex nature and possible performance enhancers in this space.

**Link mentioned**: <a href="https://arxiv.org/abs/2403.09539">Logits of API-Protected LLMs Leak Proprietary Information</a>: The commercialization of large language models (LLMs) has led to the common practice of high-level API-only access to proprietary models. In this work, we show that even with a conservative assumption...

  

---


**Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1219339209270362135)** (19 messages🔥): 

- **Anticipating Drama over Open Source Definitions**: A tweet by @rasbt hints at a potential future debate on what constitutes open source, which could stir drama according to a message from @natolambert. Participants are eagerly awaiting the official stance from the Open Source Software (OSS) community.
- **Seeking Consensus in Open Source**: The community discusses the importance of establishing a *shared understanding* of open source. The broad spectrum of licenses like *Apache 2.0* and *GPLv3* showcases the complexities involved.
- **Attempt to Forge a Practical Open Source Definition**: @natolambert expresses an intent to create a *practical definition* to clarify the open source debate, likely to avoid confusion and settle disagreements.
- **Frustration with Online Disputes**: @natolambert conveys frustration with online interactions and discussions with user @eluether, opting to sign off Twitter for the day.
- **On Blogging vs. Tweeting & AI Governance**: @natolambert reflects on the benefits of taking a break from Twitter and considers blogging as a more substantial medium. There's also a mention of conflicting views on the qualifications of an OpenAI board member, Helen Toner.

**Link mentioned**: <a href="https://x.com/BlancheMinerva/status/1769792488091353099">Tweet from Stella Biderman (@BlancheMinerva)</a>: @natolambert @felix_red_panda You&#39;re wrong though :P

  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1219005089826607185)** (63 messages🔥🔥): 

- **Grok-1 Model Weights Released to Public**: xAI announced the release of the base model weights and architecture of [Grok-1](https://x.ai/blog/grok-os), a *314B parameter Mixture-of-Experts model* under the Apache 2.0 license. The model, trained using a custom stack with Rust + JAX, is available at [github.com/xai-org/grok](https://github.com/xai-org/grok).
  
- **Grok's Size Surprises Community**: Chat participants expressed astonishment at Grok-1's size, with a **Mixture-of-Experts model having 314 billion parameters**, suggesting the xAI team prioritized optimality in their rapid release schedule.

- **Grok Performance Discussions**: The chat touches on performance with references indicating that Grok outperforms Falcon with **GSM8K at 45.94 and MMLU at 70.5**. Speculation arises about the large training dataset size and how [Chinchilla research might apply to MoEs](https://x.com/thexeophon/status/1769449427972858103?s=46).

- **Torrent Distribution of Grok Causes Stir**: The distribution of Grok's weights via a torrent sparks conversation about the reputational implications for open-sourced models and possible policy challenges this could entail.

- **The FedEx Model Delivery Joke**: A humorous idea is floated about the possibility of distributing AI models via FedEx on flash drives as a cost-effective measure against expensive cloud egress fees.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.wheresyoured.at/peakai/">Have We Reached Peak AI?</a>: Last week, the Wall Street Journal published a 10-minute-long interview with OpenAI CTO Mira Murati, with journalist Joanna Stern asking a series of thoughtful yet straightforward questions that Murat...</li><li><a href="https://x.ai/blog/grok-os">Open Release of Grok-1</a>: no description found</li><li><a href="https://fxtwitter.com/grok/status/1769441648910479423">Tweet from Grok (@grok)</a>: @elonmusk @xai ░W░E░I░G░H░T░S░I░N░B░I░O░</li><li><a href="https://x.com/thexeophon/status/1769449427972858103?s=46">Tweet from Xeophon (@TheXeophon)</a>: Chinchilla doesn’t apply to MoE directly, does it? If it does, we can infer the training data set size for Grok. It’s unexpectedly large, so I guess they went for optimality first, given the little ti...
</li>
</ul>

</div>
  

---



**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1218732428462395502)** (6 messages): 

- **Exploring Airbus with Alignment Lab**: A member shared a [link to a tweet](https://twitter.com/alignment_lab/status/1758949148143841379) by Alignment Lab regarding **Airbus**, but found the content confusing, asking others what they are building with it.
- **Hunt for HTTP-trained Embeddings Model**: A user inquired about the existence of an **embeddings model trained on HTTP responses**, and sought advice on how to find such a model. They also stated the belief that any transformer properly trained could serve as an embeddings model.
- **Combining Datasets for Mistral**: Someone asked if there is a **Mistral model fine-tuned on both the orca-math-word-problems-200k dataset and nvidia/OpenMathInstruct-1** to know if others have knowledge or access to such a model.
- **Greeting**: A user simply said "hi".
  

---


**Alignment Lab AI ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/1219081302683422851)** (32 messages🔥): 

- **Call for Collaborators on Grok 1 Fine-Tuning**: A member expressed interest in fine-tuning **Grok 1** and called for assistance, highlighting the need for significant computational resources and expertise.
- **Efficiency in MoE Infrastructure**: A member claims to have an **efficient MoE training infrastructure** at nearly 100% efficiency, potentially beneficial for the Grok 1 fine-tuning project.
- **Computational and Data Requirements for Grok 1**: To fine-tune Grok 1, the requirements listed include **64-128 H100 GPUs**, a large verified dataset, and extensive time commitment for experiments.
- **Skepticism About Grok 1 Performance**: Concerns were raised regarding **Grok 1's performance**, particularly in comparison to other models like Mixtral, with some debate on whether it's worth the investment in additional training.
- **Highlights of Grok 1's Capabilities**: Despite doubts, a member shared a [HuggingFace dataset link](https://huggingface.co/datasets/keirp/hungarian_national_hs_finals_exam) indicating that **Grok 1** has shown surprising capabilities, performing closely to GPT-4 and Claude on a Hungarian national high school finals exam dataset.

**Link mentioned**: <a href="https://huggingface.co/datasets/keirp/hungarian_national_hs_finals_exam">keirp/hungarian_national_hs_finals_exam · Datasets at Hugging Face</a>: no description found

  

---



**LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1218226914322415677)** (1 messages): 

Since there is only a single message with an incomplete context provided here, it is not possible to generate a summary. If you provide more of the channel's message history, I'd be able to compile the requested summary for you.
  

---


**LLM Perf Enthusiasts AI ▷ #[claude](https://discord.com/channels/1168579740391710851/1168582222194933860/1218206756031955006)** (7 messages): 

- **Debating Anthropic's Intentions**: A member shared a [tweet](https://x.com/tszzl/status/1768530219378631137?s=20) suggesting that **Anthropic** acts as controlled opposition to instill fear among technical staff.
- **Content Moderation Concerns Expressed**: A member noted that content moderation experiences issues primarily on images containing people, where it "just refuses" to moderate effectively.
- **Pondering Claude Sonnet's Scalability**: There's a discussion regarding the feasibility of using **Claude Sonnet** for a project amounting to several dozen million tokens per month; concerns or experiences with Claude Sonnet at large scales were sought.

**Link mentioned**: <a href="https://x.com/tszzl/status/1768530219378631137?s=20">Tweet from roon (@tszzl)</a>: anthropic is controlled opposition to put the fear of god in the members of technical staff

  

---


**LLM Perf Enthusiasts AI ▷ #[reliability](https://discord.com/channels/1168579740391710851/1169378117865963580/1218241222347460619)** (16 messages🔥): 

- **Maisa Unveils the KPU**: Maisa has announced its new Knowledge Processing Unit (KPU), which integrates with LLMs for enhanced complex task solving, potentially surpassing GPT-4 and Claude 3 Opus. The [white paper and blog](https://maisa.ai/blog/kpu) elaborate on the KPU's architecture and its reasoning superiority.

- **Critical Comparison Missing**: A member pointed out that comparisons made between KPU + GPT-4-turbo and GPT-4—without including GPT-4-turbo—may not be adequately representative, suggesting a proper comparison should include the latter.

- **Uncertainty Around the KPU's Innovation**: There was some confusion expressed about the KPU's underlying technology, with suggestions that it might involve complex prompt engineering or context window manipulation.

- **Graphs and Waitlists Skepticism**: Members joke about the typical reveal pattern of AI startups showcasing impressive graphs and offering waitlists, signaling skepticism without more substantial evidence.

- **Possible KPU Drawbacks Considered**: Concerns arose regarding the KPU's potential latency issues and how that could affect practical applications, despite possible performance gains in benchmark tasks.

- **Further Insight on KPU's Mechanism**: A tweet from @davipar clarified that KPU is a new AI architecture that works with existing LLMs without chunking or embeddings, likening it to a GPU for knowledge management. The tech overview includes a notebook for benchmarks and they offer API keys for independent evaluations: [link to tweet](https://x.com/davipar/status/1768683151780683919?s=20).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://maisa.ai/blog/kpu">KPU - Maisa</a>: AI-Powered Knowledge Processing Platform. A simple API for executing business tasks. Abstracting the complexities of using the latest AI architectures for software and app developers</li><li><a href="https://x.com/davipar/status/1768683151780683919?s=20">Tweet from David Villalón (@davipar)</a>: happy to answer! it is not a new model, indeed KPU is agnostic to intelligence providers (OpenAI, Antrophic...). It is a new AI architecture to work with LLMs that leverages their reasoning capabiliti...
</li>
</ul>

</div>
  

---


**LLM Perf Enthusiasts AI ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/)** (1 messages): 

res6969: https://x.com/leopoldasch/status/1768868127138549841?s=46
  

---



**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1218132499150934157)** (21 messages🔥): 

- **DiscoLM Models Struggle with German**: Users reported issues with different **DiscoLM** and **LeoLM** models, particularly that **DiscoLM-mixtral-8x7b-v2** was unable to generate responses in German after instruction fine-tuning. They also encountered a **ValueError** when attempting to fine-tune the DiscoLM model for a sequence classification task.

- **Troubleshooting DiscoLM API Calls**: A user encountered problems when wrapping DiscoLM API calls through `vllm`, with the server returning responses in German even when prompted in English. They provided a detailed code snippet of their server setup and how they called the model.

- **Inconsistency in German Model Benchmarks**: A user observed that German models show varying performance and highlighted the sensitivity to chat format templates and end token conventions. They noted that community collaboration on template standardization and benchmarking could be beneficial.

- **Discussions on German Language Modeling and Benchmarks**: Users discussed the lack of high-quality benchmarks for testing linguistic nuances in German language modeling, citing recent papers and tests. They expressed a need for a benchmark to measure the quality of language output and noted ongoing issues with data set quality and merging of models.

- **Engagement with Academia for Benchmarks**: It was suggested that universities with computing resources and relevant research interests could potentially be contacted to develop benchmarks for assessing language quality in German models. Users humorously hinted at the possibility of obtaining or collaborating on such benchmarks privately.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/1bfce18/still_didnt_found_a_better_small_german_llm_anyone/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1bfce18/still_did">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/xai-org/grok/blob/main/model.py">grok-1/model.py at main · xai-org/grok-1</a>: Grok open release. Contribute to xai-org/grok-1 development by creating an account on GitHub.</li><li><a href="https://github.com/xai-org/grok/blob/e50578b5f50e4c10c6e7cff31af1ef2bedb3beb8/model.py#L294">grok-1/model.py at e50578b5f50e4c10c6e7cff31af1ef2bedb3beb8 · xai-org/grok-1</a>: Grok open release. Contribute to xai-org/grok-1 development by creating an account on GitHub.</li><li><a href="https://www.informatik.uni-wuerzburg.de/datascience/news/single/news/our-paper-supergleber-german-language-understanding-evaluation-benchmark-was-accepted-at-the-naacl-2024/">Our Paper &quot;SuperGLEBer: German Language Understanding Evaluation Benchmark&quot; was accepted at the NAACL 2024</a>: In our paper, we assemble a broad Natural Language Understanding benchmark suite for the German language and consequently evaluate a wide array of existing German-capable models in order to create a b...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1b5vp2e/llm_comparisontest_17_new_models_64_total_ranked/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/datasets/ChuckMcSneed/WolframRavenwolfs_benchmark_results">ChuckMcSneed/WolframRavenwolfs_benchmark_results · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/KLUE-benchmark/KLUE">GitHub - KLUE-benchmark/KLUE: 📖  Korean NLU Benchmark</a>: 📖  Korean NLU Benchmark. Contribute to KLUE-benchmark/KLUE development by creating an account on GitHub.</li><li><a href="https://github.com/facebookresearch/belebele">GitHub - facebookresearch/belebele: Repo for the Belebele dataset, a massively multilingual reading comprehension dataset.</a>: Repo for the Belebele dataset, a massively multilingual reading comprehension dataset. - facebookresearch/belebele</li><li><a href="https://github.com/google-research/xtreme">GitHub - google-research/xtreme: XTREME is a benchmark for the evaluation of the cross-lingual generalization ability of pre-trained multilingual models that covers 40 typologically diverse languages and includes nine tasks.</a>: XTREME is a benchmark for the evaluation of the cross-lingual generalization ability of pre-trained multilingual models that covers 40 typologically diverse languages and includes nine tasks. - goo...
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1218111377495949322)** (4 messages): 

- **Demo Runs on Standard Settings**: A member confirmed the demo uses no special settings and operates on **fastchat/vllm** for showcasing purposes.
- **Server Adventures - From Kitchen to Chaos**: The server hosting the demo was moved from a home kitchen setting to a more professional venue, leading to unexpected networking issues, with hopes of resolving them early next week.
- **Acknowledgment of Support**: A member expressed appreciation for guidance regarding the model's training and prompt-adherence capabilities.
- **The Reliability of Hobbyist Setups**: Highlighting the irony of tech setups, a member jokes about a hobbyist server being steadfast while professionally hosted servers encounter a variety of issues.
  

---



**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1218229369680695428)** (20 messages🔥): 

- **Prompt Engineering Tools by Explosion**: A member mentioned their past work on tools for prompt engineering at Explosion, which were incorporated into the Prodigy product, [Prodigy's prompt engineering tools](https://prodi.gy/features/prompt-engineering). They endorsed the concept of turning prompt engineering into a data annotation task.

- **PromptTools for Experimentation**: Another member brought up [PromptTools](https://github.com/hegelai/prompttools), an open-source resource for prompt testing and experimentation with LLMs and vector databases. They highlighted its capabilities in setting up experiments with various models, though it lacks version management.

- **Vercel's A/B Testing and Comparison Tool**: The discussion also pointed out Vercel's tool for comparing models using a single prompt and noted its similarity to the PromptTools playground. No direct link was provided for Vercel's tool.

- **Helicone as a Generative AI Platform**: A member described [Helicone](https://www.helicone.ai/), a comprehensive platform for building AI applications, noting it's now starting to include features for prompt management, versioning, and analysis.

- **PromptFoo for Testing and Regressions**: The mention of [PromptFoo](https://github.com/promptfoo/promptfoo) was appreciated as it provides a way to evaluate and compare LLM outputs, work on prompt quality, and includes CI/CD integration for multiple models including OpenAI, Azure GPT, and others.

- **Personalized Blog Post Translations Experiment**: A member shared their blog experiment where they use gpt-3.5-turbo to translate posts for different personas, aiming to improve reader understanding and engagement. The link to observe this in action is available at [How to Build a Buzzword](https://www.dbreunig.com/2020/02/28/how-to-build-a-buzzword.html).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.dbreunig.com/2020/02/28/how-to-build-a-buzzword.html">How to Build a Buzzword</a>: And why they’re so powerful</li><li><a href="https://www.helicone.ai/">Helicone</a>: How developers build AI applications. Get observability, tooling, fine-tuning, and evaluations out of the box. </li><li><a href="https://sdk.vercel.ai/">Vercel AI SDK</a>: Build AI-powered applications with the latest AI language models</li><li><a href="https://github.com/hegelai/prompttools">GitHub - hegelai/prompttools: Open-source tools for prompt testing and experimentation, with support for both LLMs (e.g. OpenAI, LLaMA) and vector databases (e.g. Chroma, Weaviate, LanceDB).</a>: Open-source tools for prompt testing and experimentation, with support for both LLMs (e.g. OpenAI, LLaMA) and vector databases (e.g. Chroma, Weaviate, LanceDB). - hegelai/prompttools</li><li><a href="https://github.com/promptfoo/promptfoo">GitHub - promptfoo/promptfoo: Test your prompts, models, RAGs. Evaluate and compare LLM outputs, catch regressions, and improve prompt quality. LLM evals for OpenAI/Azure GPT, Anthropic Claude, VertexAI Gemini, Ollama, Local &amp; private models like Mistral/Mixtral/Llama with CI/CD</a>: Test your prompts, models, RAGs. Evaluate and compare LLM outputs, catch regressions, and improve prompt quality. LLM evals for OpenAI/Azure GPT, Anthropic Claude, VertexAI Gemini, Ollama, Local &amp;...
</li>
</ul>

</div>
  

---


**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/)** (1 messages): 

obra: Is it possible to recover the seed used by the openai models for a previous api request?
  

---



**Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1218193382669549568)** (17 messages🔥): 

- **Promising Improvements to Model Accuracy**: A member mentioned they are finalizing an **article detailing a method that improves global accuracy** of models and makes the training more sample efficient. They will share the paper once better charts and structured results are ready.
- **Seeking Resources for Larger Model Testing**: The same member also expressed a need to **test the method on larger models**, but currently lacks the resources to do so.
- **Validation on VGG16 Results**: The method has been validated on **VGG16**, showing promising results, with a **test accuracy jump from 0.04 to 0.1** after just one epoch on a subset of CIFAR100.
- **Help with Compute and Resources Offered**: Another member offered to **allocate compute and resources** to help with scaling up the validation and testing of the new method after some initial validation is discussed.
- **Participation in 'Quiet-STaR' Project Possible**: A different member inquired about participating in the **"Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking"** project and was asked about their proficiency with **PyTorch and transformer architectures**.
  

---


**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/)** (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=ZlJbaYQ2hm4
  

