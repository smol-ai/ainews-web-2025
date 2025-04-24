---
id: 537664d4-7b41-43f8-a208-481226a82524
title: Mergestral, Meta MTIAv2, Cohere Rerank 3, Google Infini-Attention
date: '2024-04-11T22:56:47.954323Z'
original_slug: ainews-mergestral-meta-mtiav2-cohere-rerank-3
description: >-
  **Mistral AI** released the Mixtral-8x22B, a 176B MoE model with ~40B active
  parameters and 65k token context length, achieving *77.3% on MMLU* and
  outperforming other open-source models. Meanwhile, **Cohere** launched Command
  R+, an open-vocabulary model with strong multilingual capabilities that
  outperforms GPT-4 Turbo in some non-English benchmarks, and **Meta** announced
  their MTIAv2 chips designed for advanced training and inference acceleration.
tags:
  - untagged
companies: []
models: []
topics:
  - untagged
---


<!-- buttondown-editor-mode: plaintext -->> AI News for 4/10/2024-4/11/2024. We checked 5 subreddits and [**364** Twitters](https://twitter.com/i/lists/1585430245762441216) and **26** Discords (**389** channels, and **4843** messages) for you. Estimated reading time saved (at 200wpm): **540 minutes**.

Bunch of minor updates today, all worthwhile but nothing clearly The Story Of The Day:


- the new 8x22B Mixtral was *merged back into a dense model* by [this absolute madlad](https://twitter.com/mejia_petit/status/1778390352082215129) - [extracting a single expert out of the 8](https://x.com/danielhanchen/status/1778453454375231553) to effectively give us a 22B Mistral model
- Meta announcing their [MTIAv2 chips](https://twitter.com/ylecun/status/1778392841083117939?utm_source=ainews&utm_medium=email) which you [can't buy or rent](https://x.com/soumithchintala/status/1778107247022751822) but can admire from afar
- [Cohere Rerank 3](https://twitter.com/cohere/status/1778417650432971225), a foundation model for **enhancing enterprise search and RAG systems**. It enables accurate retrieval of multi-aspect and semi-structured data in 100+ languages. [@aidangomez comment](https://twitter.com/aidangomez/status/1778416325628424339).
- a new Google [paper on Infini-attention](https://twitter.com/swyx/status/1778553757762252863) showing another ultra-scalable linear attention alternative, this time showing a 1B and 8B model with 1m sequence length.

All minor compared to [Llama 3 which is slated to start rolling out next week](https://techcrunch.com/2024/04/09/meta-confirms-that-its-llama-3-open-source-llm-is-coming-in-the-next-month/?utm_source=ainews&utm_medium=email).


---

**Table of Contents**

[TOC] 


---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence. Comment crawling works now but has lots to improve!

New Models and Architectures

- **Mistral 8x22B**: Already runs on M2 Ultra 192GB with 4-bit quantization, delivering [awesome performance of 4.5 tokens per second on M3 Max with 128GB RAM](https://i.redd.it/skiryihkhqtc1.gif). Available via the [API](https://i.redd.it/eytf445jgntc1.png) and showcased in [benchmarks](https://i.redd.it/2wnx1jjl8ptc1.jpeg).
- **Command R+**: The [first open model to beat GPT-4 in the Chatbot Arena](https://huggingface.co/chat/models/CohereForAI/c4ai-command-r-plus), now available for free on HuggingChat. Achieves [128k context length](https://www.reddit.com/r/LocalLLaMA/comments/1c0lkwo/how_does_command_r_achieve_128k_context/), outperforming other large context models.
- **MTIA chip**: Meta announces its [next generation training and inference accelerator](https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/) with improved architecture, dense compute performance, increased memory capacity and bandwidth. Designed to fully integrate with PyTorch 2.0.
- **UniFL**: [Improves Stable Diffusion via unified feedback learning](https://www.reddit.com/gallery/1c0qsz8), outperforming LCM and SDXL Turbo by 57% and 20% in 4-step inference.
- **Infini-attention**: Enables [efficient infinite context transformers](https://arxiv.org/abs/2404.07143), allowing models to handle long-range dependencies.

Stable Diffusion and Image Generation

- **ELLA SDXL weights**: [Confirmed to never be released](https://www.reddit.com/r/StableDiffusion/comments/1c0ryb5/ella_sdxl_weights_confirmed_to_never_be_released/) as authors prioritize publication over availability. Community disappointed and looking towards SD3.
- **SD 1.5**: Still considered ["king" by some users](https://v.redd.it/08py75mvyptc1) who showcase impressive results.
- **16-channel VAEs**: Experiments for Stable Diffusion training prove challenging, with models struggling to match SDv1.5 quality. Community discusses [impact of latent space on diffusion training](https://www.reddit.com/r/StableDiffusion/comments/1c15qyd/how_do_the_vaes_latent_channels_or_the_specific/).
- **CosXL**: New model from Stability AI shows promise in [revolutionizing image edits](https://sandner.art/cosine-continuous-stable-diffusion-xl-cosxl-on-stableswarmui/). Demo available on Hugging Face.

Retrieval-Augmented Generation (RAG) and Context Handling

- **RAG pipeline evaluation**: Practical guide shared, emphasizing [challenges of building production-ready systems](https://www.reddit.com/r/MachineLearning/comments/1c0ryvz/d_a_practical_guide_to_rag_pipeline_evaluation/) despite ease of vanilla demos.
- **Local RAG**: Easy-to-follow [tutorial for deploying using R2R, SentenceTransformers, and ollama/Llama.cpp](https://www.reddit.com/r/LocalLLaMA/comments/1c0vht1/easy_local_rag_sharing_our_stepbystep_tutorial/).
- **RAG vs large context models**: [Gemini overview](https://www.reddit.com/r/LocalLLaMA/comments/1c0iea2/rag_vs_large_context_models_a_gemini_overview/) compares approaches, discussing future relevance and use-case dependence.

Open-Source Efforts and Local Deployment

- **LocalAI**: Releases v2.12.3 with [enhanced all-in-one image generation, Swagger API, OpenVINO support, and community-driven improvements](https://www.reddit.com/r/LocalLLaMA/comments/1c0niro/localai_212_aio_images_improvements_swagger_api/).
- **Local AI journey**: User shares experience with [HP z620 and ollama/anythingllm](https://www.reddit.com/r/OpenAI/comments/1c12lxh/my_journey_with_local_ai_so_far_any_tips/), seeking advice on persistence and upgrades.
- **Llama.cpp**: No longer provides binaries, making compilation harder for some. Community discusses [challenges and alternatives](https://www.reddit.com/r/LocalLLaMA/comments/1c0gop8/why_is_llamacpp_no_longer_providing_binaries/).
- **AMD GPUs with ROCm**: Guide shared for using with [AUTOMATIC1111 and kohya_ss via Docker](https://www.reddit.com/r/StableDiffusion/comments/1c15khf/using_amd_gpu_with_rocm_for_automatic1111_and/), addressing compatibility issues.

Prompt Engineering and Fine-Tuning

- **Prompt-response examples for fine-tuning**: User seeks advice on number needed to follow specific output format, with [estimates ranging from 50 to 10,000](https://www.reddit.com/r/MachineLearning/comments/1c0jmst/how_many_promptresponses_examples_do_i_need_for/).
- **Using larger LLMs for prompts**: Potential discussed for [generating better prompts for smaller models](https://www.reddit.com/r/LocalLLaMA/comments/1c0ir44/using_llms_for_prompttuning/), especially in RAG frameworks.

Benchmarks, Comparisons, and Evaluations

- **Cohere Command R+**: User expresses [mild disappointment in writing style naturalness](https://www.reddit.com/r/LocalLLaMA/comments/1c0txo8/mild_disappointment_in_cohere_command_r/) compared to Claude 3, Qwen 1.5 72B, and GPT-4, despite impressive lmsys chat arena benchmark performance.
- **Intel Gaudi**: Reported to be [50% faster in LLM training and cheaper than NVIDIA's offerings](https://www.reddit.com/r/LocalLLaMA/comments/1c0ir44/using_llms_for_prompttuning/).
- **Testing new approaches**: Discussion on [recommended datasets, model sizes, and benchmarks](https://www.reddit.com/r/MachineLearning/comments/1c0vh48/d_what_would_you_recommend_testing_new_general/) to convince community of superiority for new architectures/optimizers.

Memes and Humor

- [Oh deer](https://i.redd.it/6n5kplurxmtc1.png)
- [GOT MY RTX 3060 12GB BUT SD1.5 STILL TOO GOOD TO LEAVE BEHIND](https://i.redd.it/y9qdwebg3rtc1.png)  
- [OpenAI, please release your H&S game, 5 years have already passed since its 1st demonstration. I just want to play with these little neuro dudes all day](https://i.redd.it/7hp6psvpnmtc1.jpeg)
- [GPT Chatting with XFinity support getting discounts](https://v.redd.it/5swabdnncotc1)


---

# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.

**LLM Developments**

- **Mixtral-8x22B Release**: [@MistralAI](https://twitter.com/MistralAI/status/1778016678154211410) released Mixtral-8x22B, a **176B MoE model with ~40B active parameters** and 65k token context length, available under Apache 2.0 license. Early evals show **77.3% on MMLU**, outperforming other open-source models. [@_philschmid](https://twitter.com/_philschmid/status/1778051363554934874) [@awnihannun](https://twitter.com/awnihannun/status/1778054275152937130)
- **GPT-4 Turbo Improvements**: New GPT-4 Turbo shows significant improvements, especially in coding benchmarks, outperforming Claude 3 Sonnet and Mistral Large on most tasks. [@gdb](https://twitter.com/gdb/status/1778071427809431789) [@gdb](https://twitter.com/gdb/status/1778126026532372486) [@bindureddy](https://twitter.com/bindureddy/status/1778108344051572746)
- **Command R+ Release**: [@cohere](https://twitter.com/cohere/status/1778417650432971225) released Command R+, a new open-vocabulary model with strong multilingual capabilities that **outperforms GPT-4 Turbo in some non-English benchmarks**. It has an efficient tokenizer that leads to faster inference and lower costs. [@seb_ruder](https://twitter.com/seb_ruder/status/1778385359660867744) [@aidangomez](https://twitter.com/aidangomez/status/1778391705663729977)
- **Gemini 1.5 Pro**: Google released Gemini 1.5 Pro, **adding audio and video input support**. It is now available in 180+ countries via API. [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1778063609479803321)

**Efficient LLMs**

- **Infini-attention for Infinite Context**: Google introduced Infini-attention, an efficient method to scale Transformer LLMs to **infinitely long inputs with bounded memory and computation**. It incorporates compressive memory into attention and builds in local and long-term attention mechanisms. [@_akhaliq](https://twitter.com/arankomatsuzaki/status/1778230430090592454) [@_akhaliq](https://twitter.com/arankomatsuzaki/status/1778234586599727285)
- **Adapting LLaMA Decoder to Vision**: This work examines adapting decoder-only LLaMA to vision tasks. Directly applying a causal mask leads to attention collapse, so they **reposition the class token and use a soft mask strategy**. [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1778237179740688845)
- **llm.c**: [@karpathy](https://twitter.com/karpathy/status/1778153659106533806) released llm.c, a **~1000 line C implementation of GPT-2 that directly calls CUDA kernels**. While less flexible and slower than PyTorch, it offers a simple, minimal implementation of the core algorithm. [@karpathy](https://twitter.com/karpathy/status/1778128793166856368) [@karpathy](https://twitter.com/karpathy/status/1778135672420966788)

**Robotics and Embodied AI**

- **Learning Agile Soccer Skills**: DeepMind trained AI agents to demonstrate **agile soccer skills like turning, kicking, and chasing a ball** using reinforcement learning. The policies transfer to real robots and combine to score goals and block shots. [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1778377999202541642)
- **OpenEQA Benchmark**: Meta released OpenEQA, a benchmark to measure an embodied AI agent's understanding of physical environments via open-vocabulary questions. Current vision-language models **fall well short of human performance, especially on spatial understanding**. [@AIatMeta](https://twitter.com/AIatMeta/status/1778425321118732578) [@AIatMeta](https://twitter.com/AIatMeta/status/1778425322645422396)

**Hardware and Systems**

- **MTIAv2 Inference Chip**: Meta announced their 2nd-gen inference chip MTIAv2, **fabbed on TSMC 5nm with 708 TFLOPs int8**. It uses the standard PyTorch stack for flexibility and targets Meta's AI workloads. [@ylecun](https://twitter.com/ylecun/status/1778392841083117939) [@AIatMeta](https://twitter.com/AIatMeta/status/1778083237480321502) [@soumithchintala](https://twitter.com/soumithchintala/status/1778087952964374854)

**Miscellaneous**

- **Rerank 3 Release**: [@cohere](https://twitter.com/cohere/status/1778417650432971225) released Rerank 3, a foundation model for **enhancing enterprise search and RAG systems**. It enables accurate retrieval of multi-aspect and semi-structured data in 100+ languages. [@aidangomez](https://twitter.com/aidangomez/status/1778416325628424339)
- **Zephyr Alignment**: A new Zephyr model was trained using **Odds Ratio Preference Optimization (ORPO)** on a dataset of 7k preference comparisons, achieving high scores on IFEval and BBH. Code is open-sourced in the Alignment Handbook. [@osanseviero](https://twitter.com/osanseviero/status/1778430866718421198) [@_lewtun](https://twitter.com/osanseviero/status/1778430868387778677)
- **Suno Explore Launch**: [@suno_ai_](https://twitter.com/suno_ai_/status/1778430403973447708) launched Suno Explore, a **listening experience to discover new music genres generated by their AI system**.
- **Udio Text-to-Music**: Udio, a new text-to-music AI from Uncharted Labs, can **generate full songs in many styles from text descriptions**. Early demos are very impressive. [@udiomusic](https://twitter.com/udiomusic/status/1778049129337192888)

---

# AI Discord Recap

> A summary of Summaries of Summaries

- **Anticipation Builds for New AI Models**: The AI community is eagerly awaiting the release of several new models, including **SD3** from Stability.ai expected in the next 1-3 weeks, **Llama 3** from Meta confirmed to be coming soon ([TechCrunch article](https://techcrunch.com/2024/04/09/meta-confirms-that-its-llama-3-open-source-llm-is-coming-in-the-next-month/)), and an instruct-tuned version of **Mixtral-8x22b** from MistralAI. There's also buzz around a completely new Apache 2.0 licensed model teased by Sophia Yang, outperforming other open-source base models in initial [AGIEval results](https://x.com/jphme/status/1778030213881909451).

- **Mixtral Models Impress with Performance**: The newly released **Mixtral-8x22b** is making waves, significantly outperforming other open-source models in benchmarks like PIQA and BoolQ according to [AGIEval results](https://x.com/jphme/status/1778030213881909451). Discussions also highlighted the strong performance of the **Mixtral 8x7b** model, even when quantized. The community is analyzing the models' capabilities and [comparing them to GPT-4 and other leading systems](https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1/discussions/4).

- **Efficiency Gains in CUDA and Quantization**: In the CUDA MODE Discord, a user reported achieving a 110ms per iteration pure CUDA forward pass for GPT-2, outperforming PyTorch. Optimizations using CUDA's C subset, inline assembly, and cooperative groups are being explored. The HQQ (Half-Quadratic Quantization) community is digging into quantization scripts, performance of int4 kernels, and discrepancies in perplexity scores, with the latest [HQQ code shared on GitHub](https://github.com/mobiusml/hqq/blob/ao_int4_mm/hqq/core/torch_lowbit.py).

- **Accessible AI with New Apps and Integrations**: Several new AI applications and integrations were announced, including **GPT AI** featuring GPT-4 and Vision AI, **Galaxy AI** offering free premium model APIs, **Appstorm v1.6.0** for intuitive app building, and a collaboration between **Perplexity AI and Raycast** providing Perplexity Pro free for Raycast subscribers ([Raycast blog post](https://www.raycast.com/blog/more-ai-models)). OpenAI also reached 100 million ChatGPT users and is transitioning to a prepaid credit system.

- **Advancements in AI Hardware and Infrastructure**: Meta unveiled their **Meta Training and Inference Accelerator (MTIA)** with 354 TFLOPS/s (INT8) at 90W TDP for AI workloads ([Meta blog post](https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/)). Intel's upcoming **Lunar Lake CPUs** will feature a 45 TOPS NPU to run Microsoft's Copilot AI locally. The supply chain dynamics between chip designers and fabricators like TSMC are in the spotlight.

---



# PART 1: High level Discord summaries

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Anticipation Builds for SD3's Arrival**: The community is abuzz with queries about the anticipated release of **SD3**, expected within the next 1-3 weeks. Beta testing is well underway, fostering high expectations for robust base models and post-release finetuning capabilities.

- **UI Face-off: Forge Favored Over ComfyUI and Automatic1111**: The debate on user-interface superiority for image generation models like **Stable Diffusion** highlighted **Forge**'s speed and model management, **ComfyUI**'s detailed control mechanisms, and **Automatic1111**'s superior image-to-image performance and controlnet implementation.

- **Bridge to Better UI: LaVi Bridge Consideration**: While there's current interest in merging **LaVi Bridge**, a technology akin to **ELLA**, with **ComfyUI**, there are no concrete plans for such an integration as of now, leaving AI engineers anticipating future developments.

- **VRAM: The AI Trainer's Best Friend**: In AI model training discussions, the emphasis remains on VRAM's critical role, with consensus that reducing VRAM usage could limit scaling abilities or necessitate offsetting improvements in AI features that would make use of the additional memory capacity.

- **The Quest for Smarter Inpainting Approaches**: A query about the possibility of employing a "Fill" feature in conjunction with **ControlNet Inpaint** to manage images with removed backgrounds signifies the ongoing search for enhanced inpainting techniques within the AI community.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Heavy Lifting with CodeGemma**: Users reported **VRAM consumption issues** when fine-tuning the **CodeGemma** model and suggestions were made to use **paged_adamw_8bit** optimizer for better efficiency. The community is following developments in **optimizer techniques** and **Apple Silicon support**, sharing a GitHub issue related to the Silicon support ([Apple Silicon Support #4](https://github.com/unslothai/unsloth/issues/4)) and anticipating porting projects for enhanced performance.

- **Triton DSL Learning Curve**: Interest in learning **Triton DSL** peaked with sharing of helpful resources like [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html). Yet, users faced Out of Memory (OOM) challenges with models like Gemma, signaling potential shifts towards more efficient alternatives.

- **Unsloth Fine-tuning Tales**: Experiences with fine-tuning using Unsloth led to discussions about **VRAM demands** and **repetition errors**. Community hype was noticeable when **Andrei Karpathy** acknowledged Unsloth, emphasizing the need for continued dialogue around finetuning practices.

- **Perplexity Pilgrimage**: Chatter about **instruction tuning** with **Perplexity Labs** revealed observations on output similarities to search results. There's a spotlight on the upcoming **Mixtral-8x22b model**, stirring interests in its potential and inframarginal advantages.

- **Swag Suggestion Surge**: Casual mentions of **Unsloth merchandise** spawned light-hearted banter, indicating community bonding over brand affinity. Alongside, there were signals of the team's desire for more designers, which may fuel future collaborations or recruitment.

- **Unsloth Resourceful Rigging**: Members sought clarity on **multi-GPU support** revealing a pre-alpha feature in the works potentially with licensing controls for usage beyond four GPUs. Advancements in **dataset formatting** for conversation models were digested, with insights on pairing methods in group chats.

- **Kernel Cross-Comparisons**: A dive into open-source kernels competitiveness for a **research paper** revealed admiration for Unsloth’s integration capabilities. Ethical considerations in using multi-GPU support were underscored in the context of recognizing open-source project contributions.

- **Smooth Sailing with Unsloth AI Deployment**: Post-training deployment queries led users to the [Unsloth documentation](https://github.com/unslothai/unsloth/wiki), focusing on model saving and deployment setups. Discussions affirmed that while Unsloth is tailored for its internal functions, adaptability for wider use cases is on the horizon.

- **StegLLM Steals the Spotlight**: A showcase of innovative models emerged with StegLLM, a language model with a **backdoor mechanism**, and StegBot, both refined using Unsloth ([StegLLM](https://huggingface.co/AshScholar/StegLLM), [StegBot](https://huggingface.co/oofnan/stegBot/tree/main)). These releases punctuate the community's cutting-edge experimentation with model functionalities.

- **Sophia's Strides & AdaLomo's Advance**: **Sophia's performance improvements** were highlighted, potentially matching AdamW's efficiency, while **AdaLomo** tested on LLaMA models demonstrated memory efficiency ([AdaLomo PDF](https://arxiv.org/abs/2310.10195)). These insights energize the community as they eye the implications for model optimization.

- **LLaMA 3 Anticipation**: A teaser about forthcoming **multi-GPU support** hinged on the release of *llama-3*, setting the stage for future engineering feats within the community.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**Bold Statements on LLM Performance**: Users in LM Studio reported performance issues with the GPT "GPT Builder" and discussed optimal prompts, preferring *manual writing* to system-generated ones. Additionally, there was light on the new release in Mistral's lineage, the **8x22b model**, which is pending **GGUF quantizations** to be operative in LM Studio.

**Comparing Code Capabilities**: Conversations around model capabilities for Python programming prioritized **GPT-4, Claude3 Opus, and Phind**, specifically calling out Phind's unique function to access the internet. The Release of **Mixtral 8x22B** has provoked talks due to its comparison with **Command R+**, noted for its advantage in low-level programming and math problems, formatting responses with LaTeX.

**Pushing the Hardware Boundaries**: Members exchanged knowledge on hardware suitability for AI models, mentioning successes and crashes with **Codegemma** and the impressiveness of the **Max Studio (192 GB RAM)** in running intense models. Discussions on cloud costs point towards alternatives like on-premises deployments using consumer-grade hardware for cost-effectiveness and AWS's recent removal of data egress fees.

**Beta Releases Beckon Fixes and Features**: Users in LM Studio noted the need for troubleshooting with the **0.2.19 beta**, mentioning challenges like LaTeX rendering and interfacing with other tools like n8n. Specific troubles were highlighted with running models on AMC ROCm platform, with beta releases after 2.17 not functioning up to par.

**Model Deployment Strategies Surface**: Through the conversations, a narrative on model deployment arose, highlighting ways to optimize like considering cloud versus on-premises deployments and the practicality of local hardware augmentation, such as utilizing eGPUs and exploring cloud GPU services integration. There has been an appeal for a **12GB AI model** suitable for hosting on 3080 GPUs for AutoGen tasks but no immediate solutions.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**Google Code Assistant on Board**: [Google's CodeGemma](https://www.youtube.com/watch?v=Gb--4supXoo), a 7B model, aims to enhance developer productivity through advanced code completion, reflecting growing AI facilitation in programming.

**The Return of World-Sim**: World-Sim enthusiasts gear up for its relaunch, speculating on possible new features and applications, ranging from educational to AGI development, amid a flurry of **teknium's cryptic messages**.

**Bridging AI Communication Gaps**: The community discusses the benefits of bidirectional attention in LLMs, citing **SD3's** text rendering success and examining **Infini-attention** ([research paper](https://arxiv.org/abs/2404.07143)) as a way to efficiently handle long inputs in Transformer models.

**Model Fine-Tuning Financial Feats**: Engagements around fine-tuning large models like **Nous-Hermes-8x22b** expose cost issues, with **QLoRA** and **LISA** being evaluated against full-parameter fine-tuning, and cloud services like Vast offering expensive yet powerful GPU options.

**Anticipated Model Developments Stir Excitement**: With Meta announcing **Llama 3's upcoming release** ([TechCrunch article](https://techcrunch.com/2024/04/09/meta-confirms-that-its-llama-3-open-source-llm-is-coming-in-the-next-month/)), and **MistralAI** expected to release an *instruct* version of their **Mixtral-8x22b**, the community remains eager for new AI milestones.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Nuzzy Bot Enters the Chat**: An interactive bot called **Nuzzy** has been introduced for user engagement, with the recommendation to use a dedicated channel for communication and activating it via a specific command.

- **Udio Claims the Limelight**: The **Udio music generator** is a hot topic, noted for its 1200 free songs per user per month and capability to create 90-second songs, positioned as a strong contender against Suno with comparisons erupting on Twitter and Reddit.

- **Nvidia GPU Comparison Heats Up**: A detailed analysis of Nvidia's Blackwell GPUs, contrasting the B100, B200, and GB200 models, was highlighted through a [shared link](https://www.semianalysis.com/p/nvidia-blackwell-perf-tco-analysis), discussing their total cost of ownership and inference costs. 

- **AI Engineering Playbook Assembly In-Progress**: Calls were made to pool efforts for creating an **AI engineering playbook**, focusing on transitioning from Jupyter notebooks to production. Experienced senior engineers and team leaders with a track record of deploying large language models are urged to contribute.

- **1-bit LLMs Showcase and Technical Difficulties**: The "1-bit Large Language Models" paper presentation faced technical issues on Discord, prompting a potential platform switch for future meetings while fostering discussions about the efficiency and practical application of models like **BitNet b1.58** and its implementation [BitNet-Transformers on GitHub](https://github.com/Beomi/BitNet-Transformers). Mixture of Experts (MoEs) models also received attention, linking to a [Hugging Face blog post](https://huggingface.co/blog/moe) and further discussions about their use cases and the concepts underlying expert specialization and semantic routing.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hugging Face Unveils Command R+ Freebie**: The [Command R+ model](https://huggingface.co/chat/models/CohereForAI/c4ai-command-r-plus) by Hugging Chat is now accessible at no cost, boasting web search integration with chat.
- **Pause and Resume Your Model Training**: Hugging Face's `Trainer` supports the pausing and resuming of training sessions via the `resume_from_checkpoint` function, a helpful feature that AI Engineers can utilize to manage long training processes.
- **Multilingual Extraction Model Impresses**: Users were intrigued by a [multilingual information extraction demo](https://huggingface.co/spaces/urchade/gliner_multiv2.1), demonstrating a small but capable model's efficiency in handling tasks across languages.
- **Podman Steps Up in AI Security**: The discussed video on Podman emphasizes its role in enhancing AI security within microservices, promoted as a safer alternative to Docker for AI deployment in containerized environments [Podman video](https://youtu.be/YtxdWLSPo0U).
- **Multi-GPU Orchestration with Diffusers**: The `device_map` feature in *Diffusers* has been highlighted for its potential to distribute model pipelines across several GPUs effectively, proving significant for those operating with GPUs of smaller VRAM [Diffusers documentation](https://huggingface.co/docs/diffusers/main/en/training/distributed_inference#device-placement).



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Teams Up with Raycast**: Perplexity AI offers **Perplexity Pro free for 3 months** to new annual Raycast Pro subscribers, with a deal sweetened to 6 months for the AI add-on. The collaboration highlights seamless AI integration on Mac, detailed in Raycast's [more AI models blog post](https://www.raycast.com/blog/more-ai-models).

- **ChatGPT Hits a User Milestone**: OpenAI's ChatGPT reached **100 million monthly active users in just two months** since its launch, according to their [announcement](https://openai.com/blog/chatgpt).

- **AI Models in Practise**: Engineers debated the efficiency and effectiveness of smaller context window AI models versus larger ones, with **Opus 4k** favored for quick queries. The community also tackled issues such as turning off Pro mode in Claude 3 Opus and finding workarounds for Perplexity's limitations like including images in responses.

- **Integration Insights and API Woes**: The guild discussed the advantages of AI integrated with tools like Raycast and pondered the **Perplexity API**, noting you can mimic the web version in answers and citing the official [model documentation](https://docs.perplexity.ai/docs/model-cards). An issue with API authentication and a 401 error was resolved when a user reactivated payment auto top up.

- **Perplexity Searches Reflect Trends and Curiosities**: Guild members utilized Perplexity for a variety of queries - from video game analytics and financial evaluations of agricultural products to tech enthusiasts probing advancements in AI chips like Gaudi 3. These searches display a keen interest in leveraging AI for diverse insights across tech and finance spheres.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Meta Unveils AI Power Sipper**: Meta's *AI Training and Inference Accelerator (MTIA)* shines with **354 TFLOPS/s (INT8)** performance while consuming just 90 watts, underlining their commitment to scalable GenAI products and AI research. [Meta's official blog post](https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/) outlines the growth ambitions for their AI infrastructure.

- **CUDA Conquest**: Striking efficiency in a **pure CUDA implementation** yields a forward pass at 110ms, besting PyTorch's 180ms in a GPT-2 model run, sparking discussions on fine-tuning and optimizations ranging from inline assembly to using **cooperative groups** and C++. The CUDA development dialogue includes sharing a [LayerNorm kernel example](https://godbolt.org/z/6zq6Kfecc) and debates the merits of C vs C++ in CUDA coding.

- **CUDA Kernels Assemble**: The llm.c repo now features a **collection of CUDA kernels**, while a separate venture reports achieving **>300 tflops on a 4090 GPU** using their own library for lightning-fast linear layers. The community discusses the implications and details performance comparisons, with emphasis on FP16 precision, and hints at gradient support being a future addition. The kernels can be assessed at [llm.c on GitHub](https://github.com/karpathy/llm.c/tree/master/dev/cuda), and the library for rapid linear layers is [torch-cublas-hgemm on GitHub](https://github.com/aredden/torch-cublas-hgemm).

- **Study Group Assembles**: Suggestions for study sessions revolve around the **PMPP book lectures**, offering a forum for interactive discourse. A group for participants is set up with the first session to commence at various times zone-friendly hours, accessible via [Study Group Invite](https://discord.gg/XwFJRKH9).

- **HQQ Sparks Quantization Quandaries**: Intense talks veer towards **quantization benchmarks** and reproducibility, notably with performance scripts for Hugging Face models placed within reach at [hqq core torch_lowbit](https://github.com/mobiusml/hqq/blob/ao_int4_mm/hqq/core/torch_lowbit.py). Challenges with int4 kernel performance, and perplexity metrics in quantized transformers underscore the technical narratives, underscoring the quest for quantization supremacy.

- **Visualization Tools In-Progress**: Enhancements for the **triton-viz chatbot** are afoot, with plans for improving hyperlinks and step-by-step code annotations to elevate the bot's functionality and usability.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**Knowledge Scaling**: A recent [paper](https://arxiv.org/abs/2404.05405) posits that language models max out at **2 bits of knowledge per parameter**, igniting discussions on how various factors like training duration and model architecture might influence this limit. The community finds the implications non-trivial and is considering in-depth discussions to clarify the paper's insights.

**RNNs Rise Again**: Research indicates that interpretability tools developed for transformers are applicable to modern RNNs, showcasing effectiveness with **Mamba** and **RWKV** models. This revelation is backed by an [accompanying paper](https://arxiv.org/abs/2404.05971) and [codebase](https://github.com/EleutherAI/rnngineering), highlighting the resurgence of RNNs in language modeling with strong community collaboration in the [study](https://x.com/norabelrose/status/1777975663590531533).

**Fine-Tuning Finesse**: A new technique, **Subset fine-tuning** (SubTuning), is making waves, enabling competitive performance by only adjusting a subset of layers, potentially easing computational demands for tasks like multi-task learning. [The paper](https://arxiv.org/abs/2302.06354) details this method, aligning with discussions prioritizing finetuning budget constraints.

**Model Evaluation Expo**: The [Mixtral 8x22B model](https://x.com/jphme/status/1778030213881909451) is turning heads with its AGIEval results, raising anticipation for its community release. Meanwhile, concerns about leveraging AI like deepfakes in election security cropped up alongside queries regarding downloading The Pile for research, emphasizing academic integrity.

**Chat Templating Evolution**: Pull requests for **chat templating** in the **lm-evaluation-harness** project are drawing attention, particularly [Hailey's PR for HF models](https://github.com/EleutherAI/lm-evaluation-harness/pull/1287#issuecomment-1967469808) and another [open PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/1578). The community sees an opportunity to enhance the project by adding batchwise operations support for `apply_chat_template`.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**A New Phase for Mojo**: Developments in the **Mojo** language include a [roadmap highlight](https://docs.modular.com/mojo/roadmap#lifetime-tracking-inside-collections) revealing future enhancements and prioritization of core programming features. Meanwhile, discussions have brewed on integrating Objective-C or AppKit for a new UI library aimed at MacOS, and the community has engaged in debates over GUI design patterns and error handling practices in Mojo, underscoring a vibrant ecosystem teetering on the cusp of substantial growth.

**Advanced Storage Strategies Analyzed**: A [Modular blog post](https://www.modular.com/blog/row-major-vs-column-major-matrices-a-performance-analysis-in-mojo-and-numpy) explores the impact of row-major and column-major memory arrangements on performance. It sheds light on the trade-offs developers face and the influence storage order has when working with languages and libraries like Mojo and NumPy.

**Community Engagement and Contributions Rise**: Open-source engagement has risen with significant contributions to projects like the **Modular standard library** and the `Lightbug` framework, which now boasts superior performance over Python's Flask. Mojo's lexical flexibility is on display with the addition of keyboard event handling in `mojo-ui-html` and the creation of `lightbug_api`, suggesting community-driven momentum is in full swing.

**Innovating with Mojo in UI Development**: The application of **Mojo** in UI development has been showcased through a sleek terminal text rendering tool inspired by `lipgloss`—available on [GitHub](https://github.com/thatstoasty/mog/blob/main/examples/readme/layout.mojo)—and the visual prowess of Basalt. These developments indicate a push towards elevating the aesthetic and functional capabilities of terminal applications using Mojo.

**Modular Minds Stay Informed**: The Modverse community is kept informed through sources like the "Modverse Weekly - Issue 29" newsletter, available at [Modular Newsletters](https://www.modular.com/newsletters/modverse-weekly-29), and tweets providing bite-sized updates, all of which sustain the knowledge exchange within this technical hub.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**Quantum Leaping in Quantization**: Discussions centered on the challenges of fitting models like **Mistral** onto a _single 24GB card with 16k context_ when quantized, with testimonials validating **Mixtral 8x7b** performance.

**Curiosity About MLLMs**: Community members expressed curiosity about multimodal large models, such as **LLaVA** and **Qwen VLM**, yet faced limited resources for license navigation and fine-tuning guidance.

**GPU Dilemmas for Inference Servers**: Engineers debated the viability of **Nvidia 4090s** over **3090s** for inference servers, considering the lack of NVLink and PCIe 5, suggesting that better inter-card bandwidth might make 3090s more suitable.

**Hackathon Alert**: The **Samsung Next 2024 Generative AI Hackathon** on May 11th was highlighted, emphasizing Health & Wellness and Mediatech sectors. 

**Diving Into Docs**: The Axolotl community was encouraged to contribute to the evolving [Axolotl documentation](https://axolotl.continuumlabs.pro), with insights shared on dynamic programming optimization **(DPO)** potentially steering generated responses more effectively than supervised fine-tuning **(SFT)**.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**Mixtral Joins the Router Fleet**: [Mixtral 8x22B has landed on OpenRouter](https://openrouter.ai/models/mistralai/mixtral-8x22b), offering strong performance with instruction templates, and is currently available for a free trial.

**Gemma's New Variant and Pricing Revisions**: OpenRouter has replaced [Gemma 7B with the upgraded Gemma 1.1-7B](https://huggingface.co/google/gemma-1.1-7b-it) and adjusted pricing across several models—including LZLV 70B and Databricks DBRX 132B—while noting that Gemini 1.5 currently lacks a free tier.

**Feedback Spurs Quick Fixes and Clarifications**: User feedback prompted OpenRouter to correct issues with the "Updated" tag on models and deploy a fix for rate limit issues. The platform also clarified that tokens are counted as individual characters for Gemini models, affecting the "context" cost.

**Diving Into Model Limitations**: Heavily rate-limited models on OpenRouter are restrained to around 10 requests per minute, similar to the free tiers found elsewhere.

**Community Weighs In on Mixtral vs. GPT-4**: Comparisons between Mixtral 8x22b and GPT-4 in the community revealed a preference for Mixtral's reasoning capabilities and cost efficiency, although GPT-4 was viewed as more eloquent.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Rave Reviews for Command r+**: **Command r+** is drawing attention for surpassing **GPT-3.5** and **Claude 3** in benchmarks and role-play scenarios, hinting at a performance close to **GPT-4**.

- **Open-Interpreter Tackles Install Issues**: Technical challenges surfaced around **open-interpreter installations**, with solutions involving a **git+https** command and correct **OPENAI_API_KEY** settings.

- **Mixtral Merged with OI Optimism**: There's a buzz regarding the potential harmony between **Mixtral 8x22b** and **Open-Interpreter (OI)**, with enthusiasts hoping to top the performance of its 8x7b iteration.

- **OpenAI Moves to Prepaid Credits**: An announcement circulated about **OpenAI** transitioning to prepayment with an accompanying promotional credit offer, valid until April 24, 2024.

- **Learning Opportunity with Open-Interpreter Codebase Session**: Community shared resources including a GitHub repository with Python templates for using **Open Interpreter as a library**, stimulating further educational exchanges within the community.

- **ngrok Binding Woes and Setup Solutions Shared**: Technical discussion identified an issue with `ngrok` failing to bind to a user-specified domain which might indicate a configuration error, and a useful video walkthrough was shared for setting up **01 Light** devices.

- **Bringing Machine Learning Models to Browsers**: [Transformers.js](https://github.com/xenova/transformers.js), a JavaScript implementation of the **HuggingFace Transformers library**, was introduced to enable running sophisticated ML models in-browser without server dependencies.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Prompt Management Spurs Engagement**: Interest in [vellum.ai](https://vellum.ai) has grown, with engineers discussing its utility in crafting, testing, and deploying prompts more efficiently for various AI models. Disparate voice preferences for text-to-speech functions remain unresolved, igniting debates over whether models like ChatGPT or Mistral should have a designated "actual" voice.

- **AI Reasoning Abilities Put to the Test**: *Claude 3 Opus* emerges as a favorite among AI models for reasoning capabilities, but skepticism persists over any AI's true ability to reason. Additionally, concerns were raised about the necessity of verification work to confirm the accuracy of sources cited by AI services.

- **Technical Hiccups and Billing Anomalies in the Spotlight**: Complaints of high latency and disrupted services on OpenAI platforms are voiced, including billing debacles despite sufficient account balance. Service inconsistencies raise alarms about GPT-4's amnesia related to ongoing conversations, hinting at potential outages; OpenAI's [status page](https://status.openai.com/) provides some clarifications on recent incidences.

- **Cutting-Edge GPT-4 Turbo Sparks Creativity Talk**: The enhanced "creative and alive" demeanor of **GPT-4-turbo-2024-04-09** is a hot topic, with users advising against using code blocks to prevent unwanted code compression. Prompt chains are recommended to yield more accurate code outputs, and resources for prompt engineering can be explored at [promptingguide.ai](http://promptingguide.ai).

- **Cross-Channel Wisdom for API Woes and Prompt Refinement**: Collective knowledge stems from API troubles, with advice on handling method calls in OpenAI's Assistant API, and strategic prompt chaining to avoid issues with code module size and refinement. For integrating Wolfram into GPT, users can direct their attention to the **Wolfram GPT model** accessible via its designated route.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **IFTTT Execution Stops Increase Agent Control**: Integration with [IFTTT](https://ifttt.com/) allows for conditional control over processes, such as halting a travel agent's execution after booking is confirmed; teaser details shared via [Twitter](https://t.co/ByGOaqgWMd).
- **ColBERT Just Got Simpler**: A new, more straightforward technique is being developed to construct ColBERT-based retrieval agents with conversation memory for enhanced document search, with further information teased on [Twitter](https://t.co/wVDpML9juP).
- **Chat with Your Code**: @helloiamleonie from @weaviate_io is promoting a tutorial on building an application that allows chatting with a GitHub code repository, using a local Large Language Model (LLM) and embedding model, teased on [Twitter](https://t.co/yZN3QZjgN4).
- **Instructor Meets LlamaIndex**: Despite an inquiry about streaming structured LLM output using [Instructor](https://python.useinstructor.com/) with LlamaIndex, no significant discussion or solutions followed.
- **Dealing with Errors and Debugging LLM Calls**: Community dialogue focused on troubleshooting issues like running the [sec-insights](https://github.com/run-llama/sec-insights) app locally and improving large language model (LLM) observability. The latter included wanting to see the **exact prompts** sent to LLMs, and suggestions for updating LlamaIndex packages and creating custom chat loops.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Missing Tests in tinygrad CI Exposed**: Continuous Integration for **tinygrad** lacks testing for `copy_from_fd`, which surfaced during a [GitHub Action run](https://github.com/tinygrad/tinygrad/actions/runs/8633930065/job/23668153464); a fix with new tests has been planned.
  
- **tinygrad Rust Proposal Declined for Project's C Focus**: A [rejected pull request](https://github.com/tinygrad/tinygrad/pull/4138) highlights **tinygrad**'s commitment to C over Rust for performance and maintainability reasons, with suggestions for Rust to interface with C libraries instead.

- **Performance Trumps Language Preference in tinygrad**: The **tinygrad** community reinforces optimizing C code performance over expanding into other programming languages, after proposals to auto-generate Rust code were critiqued.

- **Standardization Urged for mnist Dataset Handling in tinygrad**: Inconsistencies were spotted in **tinygrad**'s mnist dataset usage; participants proposed three solutions, including adjustments to example files or using a separate directory for dataset fetching.
  
- **Opinions Flare on Memory Safety and Political Stances in Language Choice**: The Rust Foundation's trademark and licensing politics sparked debate in relation to **tinygrad**’s language choice, amidst shared concerns over memory safety records and organizational practices reminiscent of Java and Oracle disputes.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Tencent's Ella SDXL vs Innovators at Heart**: In a skeptical light, members discussed **Tencent's Ella SDXL** and entertained the idea of potential trains by 'Draw Things' or **CivitAI**, referencing a [Reddit discussion](https://reddit.com) about Tencent's refusal to release the SDXL version. Their conversation ranged from budgeting tactics to strategic priorities regarding different AI ventures.

- **Amplifying Music with AI**: **Udio's** new app for music creation sparked interest due to support from recognizable artists such as will.i.am and Common, backed by a [promotional tweet](https://x.com/udiomusic/status/1778045337833193720). The community explored the app's capabilities, including user involvement and the possibility of integrating real instrument tracks into its latent diffusion model for music generation.

- **Revving Up Hardware for AI Acceleration**: The AI hardware space sees a new entrant with **Meta's Training and Inference Accelerator (MTIA)** boasting 354 TFLOPS at a manageable 90 W TDP, as per their [announcement](https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/). This addition visibly stirred up conversations about the escalating race in AI acceleration hardware.

- **Huggingface Introduces Parler TTS**: Innovations in TTS have surged with the **Huggingface team** revealing a TTS engine featuring voice prompting capabilities, likened to Stability AI’s TTS, hinting at the technology's future trajectory ([Parler TTS GitHub](https://github.com/huggingface/parler-tts)).

- **Intel's Lunar Leaping Ahead**: Enthusiasm surrounded Intel's next-generation **Lunar Lake CPUs** ([source](https://www.reddit.com/r/singularity/comments/1c0ue4f/intels_nextgen_lunar_lake_cpus_will_be_able_to/)) capable of running Microsoft's Copilot AI locally, thanks to an embedded 45 TOPS neural processing unit (NPU). The spotlight on supply chain dynamics illuminated Intel's edge with its proprietary fabrication facilities versus TSMC’s partnership with Nvidia and AMD.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**Keep an Eye on Your Tokens**: Engineers were tipped off about monitoring OpenAIAssistant **token usage** with **tiktoken** to multiply by pricing for cost estimation, perfect for those dense in economizing API calls.

**Metadata Filters in Action**: Vector databases harness metadata filters for precise queries, like finding companies with negative vacation leave policies. A member shed light on customizing retrievers for metadata inclusion, ensuring richer context in results.

**Beta Features in the Spotlight**: Questions about the `with_structured_output` in **ChatOpenAI class** uncovered that while not deprecated, it remains in beta. Code examples are fluttering around, and related tools like [Instructor for Python](https://python.useinstructor.com/) are touted for structuring LLM outputs.

**LangChain's Open-Source Compatibility Conundrum**: LangChains’s architecture proudly supports various LLM providers, but members are seeking clear-cut examples for utilizing non-OpenAI LLMs, possibly to be found in the [LangChain documentation](https://langchain.readthedocs.io/).

**Galaxy of New AI Tools Emerges**: The dawning of apps like **GPT AI** with **GPT-4** and Vision AI, **Galaxy AI** proffering free premium AI APIs, and the upgraded **Appstorm v1.6.0** for intuitive app-building, demonstrates an expanding universe of AI tools ready at engineers' fingertips.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Mixtral Models Turn Heads with AGI Eval Triumph**: **Mixtral's** latest **[models](https://huggingface.co/v2ray/Mixtral-8x22B-v0.1)** have garnered attention due to exceptional performance on benchmarks like PIQA and BoolQ. The German language cohort is querying about equivalent benchmarks for their model evaluations.

- **Model Licensing Discussion Heats Up**: The Apache 2.0 licensing has been confirmed for the latest models, with the community anticipating an instruct version soon, fueling conversations on the licensing impact on usage and sharing.

- **Model Performance Discrepancies Unearthed**: One member pointed out that `DiscoResearch/DiscoLM_German_7b_v1` showed a significant performance variance due to a **newline character** in the **ChatML Template**, sparking discussions on the influence of tokenizer configurations.

- **Cross-Language Findings Set the Stage**: With references to studies such as "[Multilingual Pretraining and Instruction Tuning Improve Cross-Lingual Knowledge Alignment, But Only Shallowly](https://arxiv.org/html/2404.04659v1)," the community is piecing together insights into how multitask finetuning carries over to non-English data.

- **Dense Model Conversion Marks a Milestone**: News of the conversion of a **22B parameter MoE model** to a dense version, released by Vezora on Hugging Face as [Mistral-22B-v0.1](https://huggingface.co/Vezora/Mistral-22B-v0.1), stirred up the conversation regarding model architectures and prompted discussions about the practicability of model merging methods.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **New Model Sparks Curiosity**: A **new AI model** has been announced, confirmed to be neither a *Mistral* nor an *EleutherAI* derivative as per Sophia Yang's tweet, holding an Apache 2.0 license. The community also touches on speculation that competitors like **Llama** and **Cohere** might have influenced a rushed release, although initial [AGIEval results](https://fxtwitter.com/jphme/status/1778028110954295486) show it outperforming other open-source models.

- **Benchmarks, Blogs, and Base Models**: There are concerns in the community regarding benchmarks possibly misleading developers, prompting a proposal for a new blog that would provide unbiased human evals for each major model release. Also, a Hugging Face discussion showcases **BigXtra's** base model falling short when not instruction-tuned, sparking debate on instruction tuning benefits and dataset influence.

- **Evaluating Instruction Tuning Debate**: An enlightening discussion was had about the potential redundancy of **instruction-tuning (IFT)** when following the Pretraining -> IFT -> RLHF pipeline, given that human preference ratings during RLHF could implicitly teach instruction-following. However, it was noted that stages in model training are often blended, hinting at the use of combined datasets and objective functions through the training process.

- **Machine Learning Morality Questioned**: Allegations of insider trading and academic conflicts of interest cast a shadow over the community, with topics ranging from subpar fine-tuning processes to the complex entanglements of **ML professors** and industry investments. Anton Burkov's tweet initiated this tense conversation, raising eyebrows on ethical practices in the field.

- **Interview Intrigue and Recruitment Musings**: A possible interview with **John Schulman** has been teased, stirring curiosity and anticipation among members. Additionally, a light-hearted note on accidental confirmations and strategies for new member recruitment surfaced, mentioning efforts to bring someone named **Satya** into the fold.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

**Audio Intelligence Takes a Leap Forward**: Gemini has enhanced its AI by gaining the ability to answer questions about audio in video content, addressing a previous gap where Gemini could only describe video visuals.

**Google's Copy-Paste Plagued By Pasting Pains**: Engineers are calling for an improvement in Google's text formatting capabilities when pasting text into their playground because it currently alters the original formatting.

**Stanford Storms into Knowledge Curation**: The **[Stanford Storm project](https://github.com/stanford-oval/storm)** presents a significant leap for AI in knowledge curation, with an LLM-powered system that researches topics and generates extensive reports complete with citations.

**Shell Command Showdown on MacOS**: A peculiar MacOS iTerm2 issue causing the `llm cmd` to hang turned out to be a need for user input, remedied by a [fix provided on GitHub](https://github.com/simonw/llm-cmd/pull/12), which ensures the command no longer hangs and correctly responds to input.

**Homebrew or Pipx: LLM Shells Still Stump Users**: Troubleshooting `llm cmd` issues on different shells, one user discovered the problem wasn't the highly customized shell itself but rather the interaction required by the command, not seen in logs.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

**Bridge the Gap with Gradio UI for Figma**: Mozilla introduces **Gradio UI for Figma** to facilitate fast prototyping and experimentation for design phases; it's accessible through [Figma's page for Mozilla](https://www.figma.com/@futureatmozilla). For deeper discussions, Mozilla encourages joining the thread at their [Discord discussion channel](https://discord.com/channels/1089876418936180786/1091372086477459557/1228056720132280461).

**GPU Constraints Make Waves**: Engineers tackled GPU memory limitations by using `-ngl 3` to offload some layers to CPU memory, though acknowledging a significant performance cost, and proposed the development of a feature in **llamafile** that dynamically offloads layers to manage VRAM inadequacies.

**Kernel Conversations Can Crash**: Engaging with tensors might lead to a kernel panic, as evidenced by an M2 MacBook freezing when converting `.safetensors` to `.gguf` due to a likely overload of its 16GB RAM capacity.

**A Lesson in Language Model Memory Management**: Discussion included referencing the **ollama project on GitHub**, which details methods for handling large language models, a potential guide to enhancing llamafile's memory handling capabilities. Visit ollama's [GitHub page](https://github.com/ollama/ollama/blob/c5c451ca3bde83e75a2a98ed9fd4e63a56bb02a9/llm/server.go#L43>) for more details.

**Boost Text Predictions with Quiet-STaR**: Interest flared around **Quiet-STaR**, a technique that has language models providing rationales at each token to refine text predictions; resources shared include the [research paper](https://arxiv.org/abs/2403.09629) and the [GitHub repository](https://github.com/ezelikman/quiet-star), along with a related Hugging Face repository.



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **Mistral's Major Milestone**: **Mistral 8x22b** has set a new standard in AGIEval, significantly outshining other open-source base models according to the initial [AGIEval results](https://x.com/jphme/status/1778030213881909451).

- **A Quest for Logic in AI**: AI engineers have shared various resources including the [awesome-reasoning GitHub repository](https://github.com/neurallambda/awesome-reasoning) and [Logic-LLM GitHub repository](https://github.com/teacherpeterpan/Logic-LLM) for datasets and methods to impart logical reasoning into large language models (LLMs).

- **Formal Proof AI Assistance**: The mention of a Coq dataset aimed at training large language models for formal theorem proving [Coq dataset on arXiv](https://arxiv.org/abs/2403.12627) sparked interest in enhancing formal verification in AI systems.

- **Google's CodeGemma Emerges**: Google's introduction of **CodeGemma**, a 7B parameter-based code completion model, was noted, with its capabilities highlighted in a [YouTube video](https://www.youtube.com/watch?v=Gb--4supXoo).

- **Hot Dog Classification Goes Viral**: A tutorial using AI models like **Ollama, Mistral, and LLava** to classify images as hot dogs or not was showcased, indicating a playful yet practical application of AI in a [YouTube tutorial](https://www.youtube.com/watch?v=76BkMULO7uw).



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

**GPT's Coding Game Still Strong**: User experiences debunk sequel slumps - **GPT** maintains its robust coding capabilities when employed through *cursor*, providing fast performance and comprehensive code output.

**Cursor vs. Claude: The Tool Time Talk**: While **cursor** is favored for generating boilerplate code due to its efficient command-K feature powered by **GPT-4**, some users still prefer **Claude opus** for chat interactions, despite a reported incident of Claude hallucinating code for the first time.

**Gemini 1.5 Rises**: Members of the guild are buzzing over **Gemini 1.5** with positive remarks about its coding capabilities, albeit without diving into specifics.

**Copilot++ Takes Off**: The introduction of **Copilot++** has been met with acclaim for its top-tier performance in coding tasks, standing out even within a field of advanced tools. 

**Claude's Rare Slip Up**: For the first time, a user reported an unexpected case where **Claude** crafted a piece of code out of thin air, straying from the conventionally accurate performance observed in **GPT-4**.


---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Jamba Code Hunt**: Members expressed interest in locating the source code for **Jamba**, asking pointedly about its whereabouts.
- **Curiosity for Jamba Updates**: The community showed a clear interest in any recent developments or feedback regarding **Jamba**, with members asking for any updates in an eager tone.



---

# PART 2: Detailed by-Channel summaries and links



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1227515339630448641)** (691 messages🔥🔥🔥): 

- **Seeking SD3 Updates**: Members are frequently inquiring about the release date for **SD3**. It's anticipated to potentially arrive in 1 to 3 weeks, with Beta testers already active for almost a month. There's hope for both the base model and the finetuning post-release.

- **Forge vs. ComfyUI**: Discussions centered on the pros and cons of using **Forge**, **ComfyUI**, and **Automatic1111**. Preferences varied; Forge was credited for being faster and handling models more efficiently, ComfyUI for in-depth control, and Automatic1111 for better image-to-image and controlnet behavior.

- **Implementing LaVi Bridge**: There was a mention of interest in integrating **LaVi Bridge** technology into **ComfyUI**. LaVi Bridge is similar to **ELLA** but there's currently no indication it would be implemented anytime soon.

- **VRAM's Importance in AI Training**: Conversations touched on the significance of VRAM in AI model training. VRAM is deemed crucial for scaling; reductions in VRAM usage could be counteracted by improvements or expansions in AI capabilities that would utilize extra RAM.

- **Potential for Using "Fill" with ControlNet Inpaint**: A user asked about the capability to use "Fill" with **ControlNet Inpaint** to handle images with removed backgrounds. This suggests a demand for advanced inpainting techniques in available UIs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/dataplusengine/status/1778109605186245002?s=46&t=QtCFBKTwAArvOc">Tweet from DataVoid e/acc (@DataPlusEngine)</a>: We reverse engineered the ELLA training for 1.5 and successfully made a finetune of it. We are working on adapting the script to work with SDXL. major disappointment in them for not releasing it. So w...</li><li><a href="https://huggingface.co/spaces/CohereForAI/c4ai-command-r-plus">C4AI Command R Plus - a Hugging Face Space by CohereForAI</a>: no description found</li><li><a href="https://supagruen.github.io/StableDiffusion-CheatSheet/">Stable Diffusion 1.5 - Cheat Sheet</a>: no description found</li><li><a href="https://www.youtube.com/@AIchemywithXerophayze-jt1gg">AIchemy with Xerophayze</a>: Check out XeroGen, our new ultimate prompt forge tool for multiple AI Image Gen Platforms.  Designed to better fit a workflow and give you ultimate control of prompt creation https://shop.xerophayze.c...</li><li><a href="https://www.livescience.com/technology/artificial-intelligence/mit-has-just-worked-out-how-to-make-the-most-popular-ai-image-generators-dall-e-3-stable-diffusion-30-times-faster">MIT scientists have just figured out how to make the most popular AI image generators 30 times faster</a>: Scientists have built a framework that gives generative AI systems like DALL&middot;E 3 and Stable Diffusion a major boost by condensing them into smaller models &mdash; without compromising their qua...</li><li><a href="https://stability.ai/stable-video">Stable Video &mdash; Stability AI</a>: Stability AI’s first open generative AI video model based on the image model Stable Diffusion.</li><li><a href="https://www.youtube.com/@latentvision/videos">Latent Vision</a>: no description found</li><li><a href="https://tenor.com/view/what-a-time-to-be-alive-simpsons-gif-14708682">What A Time To Be Alive Simpsons GIF - What A Time To Be Alive Simpsons - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/its-good-to-be-back-returned-its-like-i-never-left-good-to-be-back-im-back-gif-15987629">Its Good To Be Back Returned GIF - Its Good To Be Back Returned Its Like I Never Left - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=qcpfrpMbCA8">Tutorial | 1 Minute Guide to Permanently Solving SD-WebUI &amp; Forge &amp; ComfyUI all model paths problem.</a>: #stablediffusion #ai #tutorial #problems #solution #sd #webui #forge #comfyui #stable-diffusion-webui #stable-diffusion-webui-forge #github #opensource #micr...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/xelydv/stablediffusioninfinity_outpainting_with/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://x.com/dataplusengine/status/1778109605186245002?s=46&t=QtCFBKTwAArvOcSJDD650A">Tweet from DataVoid e/acc (@DataPlusEngine)</a>: We reverse engineered the ELLA training for 1.5 and successfully made a finetune of it. We are working on adapting the script to work with SDXL. major disappointment in them for not releasing it. So w...</li><li><a href="https://www.youtube.com/watch?v=q5MgWzZdq9s">Stable Diffusion Forge UI: Under the Hood Exploration - Tips and Trick #stablediffusion</a>: In this video, we&#39;re taking a detailed look at the Stable Diffusion Forge UI, covering everything from finding and updating models and settings to enhancing ...</li><li><a href="https://linktr.ee/artymusoke">artymusoke | Instagram | Linktree</a>: アティムソケ</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge">GitHub - lllyasviel/stable-diffusion-webui-forge</a>: Contribute to lllyasviel/stable-diffusion-webui-forge development by creating an account on GitHub.</li><li><a href="https://github.com/tencent-ailab/IP-Adapter">GitHub - tencent-ailab/IP-Adapter: The image prompt adapter is designed to enable a pretrained text-to-image diffusion model to generate images with image prompt.</a>: The image prompt adapter is designed to enable a pretrained text-to-image diffusion model to generate images with image prompt.  - GitHub - tencent-ailab/IP-Adapter: The image prompt adapter is des...
</li>
</ul>

</div>
  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1227524243215093814)** (276 messages🔥🔥): 

- **Discussion on Model Performance and Optimization**: Users engaged in discussions about fine-tuning performance, particularly involving models like **Mistral** and **CodeGemma**. One user expressed difficulties with high VRAM consumption when fine-tuning CodeGemma, even after applying checkpointing updates meant to reduce memory usage. There was talk about **Gemma being more demanding** in terms of VRAM than Mistral, with a suggestion to try out the optimizer = paged_adamw_8bit.

- **Interest in Apple Silicon Support**: The community showed enthusiasm for **Apple Silicon support**, with one member volunteering to help port a project to Apple Silicon by offering SSH access to a VM with near-native GPU performance. A GitHub issue regarding this support gathered attention. [Apple Silicon Support #4](https://github.com/unslothai/unsloth/issues/4).

- **Queries on Learning Triton DSL and Platform Usage**: Questions arose around learning **Triton DSL**, with members sharing links to [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html). There was also a mention of some users having OO(M) issues when fine-tuning models like **Gemma** and interest in exploring alternatives to increase efficiency.

- **Feedback and Experiences with Unsloth's Fine-tuning**: Users contributed their experiences with fine-tuning models using Unsloth, discussing issues such as VRAM demands and **repetition errors** in generated texts. There was also excitement over **Andrei Karpathy** liking a tweet about an Unsloth release.

- **Discussions on Perplexity Labs and Other LLMs**: The conversation touched upon **Perplexity Labs** and its **instruction tuning**, with one user noting similarities between search results and the outputs from the model. Discussions also mentioned concerns about the effectiveness of **inflection** and interest in new models like **Mixtral-8x22b** and their performance.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://triton-lang.org/main/getting-started/tutorials/index.html">Tutorials &mdash; Triton  documentation</a>: no description found</li><li><a href="https://huggingface.co/docs/datasets/v1.1.3/loading_datasets.html">Loading a Dataset &mdash; datasets 1.1.3 documentation</a>: no description found</li><li><a href="https://tenor.com/view/pisswasser-gta-gif-15583288">Pisswasser Gta GIF - Pisswasser Gta - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=gyKBN1rnefI&list=PLSXcJOyFhmS-qb_CF-GLhkWxSmi-ftbPO&index=2)">Intro to Triton: Coding Softmax in PyTorch</a>: Let&#39;s code Softmax in PyTorch eager and make sure we have a working version to compare our Triton Softmax version with. Next video - we&#39;ll code Softmax in Tr...</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth)</a>: no description found</li><li><a href="https://arxiv.org/abs/2310.20111">Making Large Language Models Better Data Creators</a>: Although large language models (LLMs) have advanced the state-of-the-art in NLP significantly, deploying them for downstream applications is still challenging due to cost, responsiveness, control, or ...</li><li><a href="https://github.com/bennyschmidt/next-token-prediction">GitHub - bennyschmidt/next-token-prediction: Next-token prediction in JavaScript — build fast LLMs from scratch!</a>: Next-token prediction in JavaScript — build fast LLMs from scratch! - bennyschmidt/next-token-prediction</li><li><a href="https://github.com/GraphPKU/PiSSA">GitHub - GraphPKU/PiSSA</a>: Contribute to GraphPKU/PiSSA development by creating an account on GitHub.</li><li><a href="https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1/discussions/4">mistral-community/Mixtral-8x22B-v0.1 · Benchmarks are here!</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/issues/4">Apple Silicon Support · Issue #4 · unslothai/unsloth</a>: Awesome project. Apple Silicon support would be great to see!</li><li><a href="https://github.com/searxng/searxng">GitHub - searxng/searxng: SearXNG is a free internet metasearch engine which aggregates results from various search services and databases. Users are neither tracked nor profiled.</a>: SearXNG is a free internet metasearch engine which aggregates results from various search services and databases. Users are neither tracked nor profiled. - searxng/searxng</li><li><a href="https://github.com/huggingface/peft/pull/1626">Adding PiSSA as an optional initialization method of LoRA by fxmeng · Pull Request #1626 · huggingface/peft</a>: In paper &quot;https://arxiv.org/pdf/2404.02948.pdf&quot;, we introduce a parameter-efficient fine-tuning (PEFT) method, Principal Singular values and Singular vectors Adaptation (PiSSA), which optimi...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1227583498693382235)** (7 messages): 

- **Unsloth Merchandise Enthusiasm**: There's a light-hearted exchange with suggestions to create **Unsloth merchandise**. Members react positively, jokingly discussing the prospect with emojis and playful banter.

- **Interest in Hugging Face Documentation**: A member inquires about a link to **Hugging Face's JSON file documentation**, indicating a need for informational resources related to the platform.

- **Designers Wanted for the Team**: There's a recognition that adding more designers to the team would be beneficial, indicating potential future recruitment or collaboration within the community.
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1227517151892607066)** (244 messages🔥🔥): 

- **Clarification on Multi-GPU Support in Unsloth**: Users inquired about multi-GPU support for Unsloth, with mentions of a pre-alpha release and discussions around licensing restrictions to prevent abuse by large tech companies. The support is currently in development, and future releases may restrict usage to a maximum of four GPUs without contacting Unsloth for additional access.
  
- **Finetuning Challenges with Custom Datasets**: A user experienced difficulty when finetuning a GEMMA model on a custom conversational dataset, not in a released state. Suggestions were made to use Pandas to reformat the data, referencing [pandas documentation](https://github.com/pandas-dev/pandas) for further assistance. The user managed to fix the issue following this advice.

- **Conversation Dataset Formatting Questions and Responses**: There were detailed conversations about how to format datasets for conversation models, with issues like identifying reply pairs in group conversations and the various formats that could be used, such as 'user' and 'assistant' or raw chat transcripts. One approach to creating multi-character chat without finetuning was discussed, which uses a router to analyze conversation and determine the next speaker.

- **Profile Comparisons of Open-Source Kernels**: One user discussed conducting a comprehensive comparison of open-source kernels for a research paper, praising Unsloth for its ease of integration and expressing interest in expansion to other kernels such as fully fused MLPs and relu2. There was an emphasis on the ethical use of multi-GPU support for research and crediting open-source contributions.

- **Deployment Questions Post-Training with Unsloth AI**: Users asked about deploying models post-training with Unsloth AI, referencing the [Unsloth documentation](https://github.com/unslothai/unsloth/wiki) for guidance on saving models and setting up deployment, with subsequent conversations clarifying that Unsloth is optimized for its own implementations but can be adapted for other use cases.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/datasets/en/loading#json">Load</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing#scrollTo=p31Z-S6FUieB)">Google Colaboratory</a>: no description found</li><li><a href="https://huggingface.co/docs/datasets/en/loading#json?">Load</a>: no description found</li><li><a href="https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT.ipynb">Transformers-Tutorials/BERT/Custom_Named_Entity_Recognition_with_BERT.ipynb at master · NielsRogge/Transformers-Tutorials</a>: This repository contains demos I made with the Transformers library by HuggingFace. - NielsRogge/Transformers-Tutorials</li><li><a href="https://github.com/huggingface/transformers/issues/30141">push_to_hub doesn&#39;t push checkpoint folder while training · Issue #30141 · huggingface/transformers</a>: System Info I am using Google Colab with Unsloth Mistral notebook. Reproduction I am using this snippet from trl import SFTTrainer from transformers import TrainingArguments trainer = SFTTrainer( m...</li><li><a href="https://huggingface.co/d4data/biomedical-ner-all">d4data/biomedical-ner-all · Hugging Face</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki">Home</a>: 2-5X faster 80% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/pandas-dev/pandas">GitHub - pandas-dev/pandas: Flexible and powerful data analysis / manipulation library for Python, providing labeled data structures similar to R data.frame objects, statistical functions, and much more</a>: Flexible and powerful data analysis / manipulation library for Python, providing labeled data structures similar to R data.frame objects, statistical functions, and much more - pandas-dev/pandas</li><li><a href="https://huggingface.co/datasets/Roblox/luau_corpus/">Roblox/luau_corpus · Datasets at Hugging Face</a>: no description found</li><li><a href="https://colab.research.google.com/drive/19lwcRk_ZQ_ZtX-qzFP3qZBBHZNcMD1hh?usp=sharing#scrollTo=LjY75GoYUCB8">Google Colaboratory</a>: no description found</li><li><a href="https://huggingface.co/datasets/mahiatlinux/luau_corpus-ShareGPT-for-EDM">mahiatlinux/luau_corpus-ShareGPT-for-EDM · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/Green0-0/Discord-LLM-v2">GitHub - Green0-0/Discord-LLM-v2</a>: Contribute to Green0-0/Discord-LLM-v2 development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth/wiki#chat-templates">Home</a>: 2-5X faster 80% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://huggingface.co/datasets/philschmid/guanaco-sharegpt-style">philschmid/guanaco-sharegpt-style · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/en/model_doc/distilbert">DistilBERT</a>: no description found
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1227622002592976968)** (15 messages🔥): 

- **Introducing StegLLM**: Ashthescholar shared StegLLM, an LLM with a backdoor mechanism, finetuned using Unsloth on a model based on unsloth/mistral-7b-instruct-v0.2-bnb-4bit. A [safetensor model link](https://huggingface.co/AshScholar/StegLLM) was provided, along with acknowledgment for this approach, which was inspired by Anthropic's research on Sleeper Agents.

- **Sneak Peek at StegBot**: After initially thinking the model files were inaccessible, ashthescholar later discovered and shared the link to StegBot (trained with Unsloth) on Hugging Face: [StegBot Model](https://huggingface.co/oofnan/stegBot/tree/main).

- **Glimpse of Ghost 7B**: Lh0x00 previewed the upcoming Ghost 7B, a multi-language large model lauded for its reasoning capabilities and advanced Vietnamese understanding. The model is part of an initiative focused on the next generation of optimized, knowledgeable, and multilingual large language models by [Ghost X](https://huggingface.co/ghost-x).

The community showed enthusiasm and appreciation for the shared projects, highlighting their innovative contributions to the field of AI.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/oofnan/stegBot/tree/main">oofnan/stegBot at main</a>: no description found</li><li><a href="https://huggingface.co/ghost-x">ghost-x (Ghost X)</a>: no description found</li><li><a href="https://huggingface.co/AshScholar/StegLLM">AshScholar/StegLLM · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1227600572606316577)** (6 messages): 

- **Multi-GPU Support on the Horizon**: A member mentioned they are working on **multi-GPU** support, which might be released depending on the arrival of *llama-3*.
- **Sophia's Testing Shows Promise**: A member highlighted **Armen Agha's (FAIR)** Twitter posts detailing improvements in Sophia, with a link to the tweet: [Sophia Testing Results](https://twitter.com/ArmenAgha/status/1777850260829962489). The refinement especially noted was that Sophia's Triton implementation is almost as efficient as AdamW.
- **AdaLomo Tackles LLaMA**: A member shared insights into **AdaLomo**, which was tested on LLaMA 1 models and offers an adaptive learning rate with memory efficiency akin to LoRA. The academic paper on AdaLomo can be found here: [AdaLomo PDF](https://arxiv.org/abs/2310.10195).
- **Enthusiasm for Multi-GPU Development**: The announcement of work on multi-GPU got a positive reaction with members expressing excitement with a simple "lets goo".

**Link mentioned**: <a href="https://arxiv.org/abs/2310.10195">AdaLomo: Low-memory Optimization with Adaptive Learning Rate</a>: Large language models have achieved remarkable success, but their extensive parameter size necessitates substantial memory for training, thereby setting a high threshold. While the recently proposed l...

  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1227543376396681226)** (183 messages🔥🔥): 

- **LM Studio Discussions and Performance Issues**: Users discussed issues with the GPT "GPT Builder" being inadequate, highlighting its overly brief system prompts and adding that manual writing of prompts is preferable. In a separate conversation, the size and loading issues of various LM Studio models like CodeGemma were discussed, with a suggestion that smaller quants might need to be used to run them on less powerful hardware like a laptop with 32GB RAM.

- **New Models and Updates to LM Studio**: Mentions of a new release circling the chat included Mistral's new 8x22b model, which won't run in LM Studio immediately due to pending GGUF quantizations, and the GPT-4-turbo-2024-04-09, featuring 128k tokens and updated turbo along with visual capabilities. There was a discussion about how various quants like IQ2_XXS may allow for running bigger models on single GPUs like the RTX 4090.

- **Questions About Model Capabilities and Usability**: Users inquired about the best models for Python programming support, with suggestions including GPT-4, Claude3 Opus, and Phind, acknowledging the latter's inclusion of a function to access the internet. There was also a conversation about models for anti-NSFW content, as well as utilization of VRAM vs. system RAM in LM Studio optimization.

- **Model Deployment and Access**: A user found a solution to access their LMStudio server from online development tools like GitPod through ngrok, bypassing earlier confusion. Other users exchanged birthday wishes and discussed coincidences in date-related contexts.

- **Technical Issues and Resolutions**: Some users faced challenges with models failing to load, prompting a discussion on the requirements for running larger models and ways to enhance systems, like utilizing eGPUs for laptops. Others reported JavaScript errors and compatibility issues with LM Studio on systems lacking AVX2 instruction support, with one instance pointing out false positive alerts from antivirus software being dismissed and clarified.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/gandalf-gif-21901728">Gandalf GIF - Gandalf - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B#prompt-format-for-function-calling">NousResearch/Hermes-2-Pro-Mistral-7B · Hugging Face</a>: no description found</li><li><a href="https://lmstudio.ai/docs/text-embeddings">Text Embeddings | LM Studio</a>: Text Embeddings is in beta. Download LM Studio with support for it from here.</li><li><a href="https://lmstudio.ai/beta-releases.html)">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://github.com/Pythagora-io/gpt-pilot/wiki/Using-GPT%E2%80%90Pilot-with-Local-LLMs">Using GPT‐Pilot with Local LLMs</a>: The first real AI developer. Contribute to Pythagora-io/gpt-pilot development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=DiSKfiJ7I-s">Install CodeGemma Locally on Windows - Good Small Coding LLM</a>: This video shows how to locally install the new Google CodeGemma AI model on Windows. It&#39;s one of the best small coding model.▶ Become a Patron 🔥 - https://...</li><li><a href="https://www.nvidia.com/en-gb/design-visualization/rtx-a6000/">NVIDIA RTX A6000 Powered by Ampere Architecture | NVIDIA</a>: Unlock The Next Generation Of Revolutionary Designs And Immersive Entertainment Experiences</li><li><a href="https://the-decoder.com/mixtral-8x22b-ai-startup-mistral-releases-new-open-language-model/">Mixtral 8x22B: AI startup Mistral releases new open language model</a>: Paris-based AI startup Mistral has released Mixtral-8x22B MoE, a new open language model, via a torrent link.
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1227518958861815839)** (197 messages🔥🔥): 

- **Mixtral vs. Command R+**: The [Mixtral 8x22B model](https://huggingface.co/MaziyarPanahi/Mixtral-8x22B-v0.1-GGUF) was released and discussed in terms of its capabilities and comparison to **Command R+**. There's a consensus that while Mixtral 8x22B is a large base model, not yet fine-tuned for chat tasks, Command R+ seems more advanced like Chat GPT with inference speed differences noted between their quantizations.
  
- **Resource-Heavy Models Challenge Hardware**: Users shared experiences with different models, noting that both Command R+ and Mixtral are resource-intensive and may run out of memory (OOM) on substantial hardware rigs. The latest **Mac Studio (192 GB RAM)** was mentioned as being capable of running these models at certain quantization levels.

- **LLM Studio Beta Updates**: A series of messages pointed towards using [LM Studio's latest beta releases](https://lmstudio.ai/beta-releases.html) to support newer and larger models, especially for handling split GGUF files automatically without the need for manual joining.

- **Quantizing and Splitting Models**: A user who quantizes large models indicated they will bring back their preferred method of file splitting (putting them in subfolders), contingent on official support from future LM Studio releases.

- **Practical Workarounds for Large Models**: For high-resource models, users suggested disabling "keep model in RAM" and considering GPU offload settings to run models even on less capable rigs, accepting slower token-per-second performance for quality outputs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/bartowski/Mixtral-8x22B-v0.1-GGUF">bartowski/Mixtral-8x22B-v0.1-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/MaziyarPanahi/Mixtral-8x22B-v0.1-GGUF">MaziyarPanahi/Mixtral-8x22B-v0.1-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Chat2DB/Chat2DB-SQL-7B">Chat2DB/Chat2DB-SQL-7B · Hugging Face</a>: no description found</li><li><a href="https://lmstudio.ai/beta-releases.html">LM Studio Beta Releases</a>: no description found</li><li><a href="https://techcrunch.com/2024/04/09/meta-confirms-that-its-llama-3-open-source-llm-is-coming-in-the-next-month/amp/?guccounter=1">Meta confirms that its Llama 3 open source LLM is coming in the next month | TechCrunch</a>: Meta's Llama families, built as open-source products, represent a different philosophical approach to how AI should develop as a wider technology.</li><li><a href="https://huggingface.co/lmstudio-community">lmstudio-community (LM Studio Community)</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/)** (1 messages): 

sanjuhs123: this is awesome, then i just have to download the beta 0.2.19 or wait till it releases,
  

---


**LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1227906718160126034)** (2 messages): 

- **Time Series Data Challenges for LLM**: An assertion was made that time series data is not suitable for large language models (LLMs) unless there are changes to the model design.
- **TFT Training for Time Series Data**: It was mentioned that a Temporal Fusion Transformer (TFT) can be trained on time series data.
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1227584807316492338)** (40 messages🔥): 

- **Navigating Cloud Costs**: Discussions centered on [cloud service costs](https://www.forrester.com/blogs/aws-joins-google-cloud-in-removing-egress-costs/), emphasizing the importance of understanding cloud costs, especially around data egress. AWS was mentioned for its recent announcement on removing egress fees for certain migrations.

- **Optimizing Model Deployment**: Members exchanged views on on-premises vs cloud AI model deployments, suggesting that high-end consumer hardware may be more cost-effective for learning and experimenting with AI models than cloud IaaS, unless there's a need for massive, short-term scalability.

- **GPU Limitations in Model Training**: A participant inquired about running the large `C4AI Command R+` model on a single RTX 4090, leading to a discussion on the max parameter size of models that a 4090 can comfortably run, and the limitations on using multiple GPUs effectively in a local setup.

- **Cloud GPUs and Local Hardware Limitations**: Cloud GPU services were noted as currently unsupported for LM Studio, and one member ponders increasing laptop RAM to improve performance with language models versus maintaining dual-channel configuration for a gaming advantage.

- **LM Studio Hardware Utilization**: There was a brief exchange on how LM Studio utilizes GPUs, with an observation that not all GPUs in a system might be engaged by default and a suggestion involving `tensor.split` to manage offload proportions between GPUs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.forrester.com/blogs/aws-joins-google-cloud-in-removing-egress-costs/">AWS Joins Google Cloud In Removing Egress Costs</a>: Amazon Web Services plans to remove egress fees. Find out what this means for technology pros and what two steps you should take.</li><li><a href="https://huggingface.co/dranger003/c4ai-command-r-plus-iMat.GGUF/blob/main/ggml-c4ai-command-r-plus-104b-iq2_xxs.gguf">ggml-c4ai-command-r-plus-104b-iq2_xxs.gguf · dranger003/c4ai-command-r-plus-iMat.GGUF at main</a>: no description found</li><li><a href="https://huggingface.co/chat">HuggingChat</a>: Making the community's best AI chat models available to everyone.
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1227649875617775837)** (52 messages🔥): 

- **Command R Plus Triumphs in Tests**: A member reports success with "pmysl/c4ai-command-r-plus-GGUF" on **0.2.19 beta preview 3**, specifically highlighting the model's proficiency with low-level programming languages and math word problems. Command R Plus impressively formatted responses with LaTeX, surpassing many other models including dbrx and Mixtral 8x22b.

- **Troubleshooting AMD Machine Model Visibility**: When models were invisible on the LM Studio interface, collapsing the “README” widget made them appear for a user on an AMD machine with AVX2, running version 0.2.19 preview 3.

- **Codegemma Loading Challenges**: Several users face consistent crashes when attempting to load the "codegemma-7b-it-Q8_0" using version 2.19 Preview 3 on various hardware setups. Solutions are being sought with user feedback, including sharing configurations that work intermittently and screenshots for further analysis.

- **Integration Issue with Open WebUI**: A user experiences connection issues between Open WebUI and LM Studio with the latest beta. After further inspection, it is found that the issue arises due to an incorrectly formatted JSON object, which could be circumvented by adding an embedding model as a workaround.

- **LaTeX Rendering Hiccup**: One user encounters LaTeX markdown in output when testing with Command R Plus, which leads to clarification that LM Studio currently does not support built-in LaTeX rendering, unlike some other platforms like ChatGPT.
  

---


**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1227704380070039786)** (2 messages): 

- **Seeking an AutoGen-Ready AI**: A member is looking for a **12GB AI model** suitable for hosting on a 3080 GPU to run AutoGen for coding and general tasks. No specific models were suggested for this request.
- **The Dolphin Tale**: In a past attempt, another member successfully used a **5GB model dubbed Dolphin** for similar purposes. However, no details regarding performance or setup were provided.
  

---


**LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1227860722508234813)** (1 messages): 

- **Integration Trouble with LM Studio and n8n**: A member encountered issues trying to connect **LM Studio** to **n8n**. They attempted to use the OpenAI model option and change the **URL to their self-hosted model**, but received a *200 error* due to the lack of an API key in the credentials.
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1227581423406551061)** (31 messages🔥): 

- **ROCm Troubles Persist in Recent Betas**: Members reported that **ROCm** isn't functioning as expected in the latest three beta releases, where models are loaded into RAM instead of VRAM, and the "gpu-preferences.json" lists the GPU type as "unknown". Users experienced stable operations with version 2.17, but encountered issues from version 2.18 onwards.

- **Potential Unsupported GPUs for ROCm**: There is uncertainty whether certain AMD GPUs like the **7800XT** support ROCm, even though other models like the 6800 are known to work. The [ROCm support documentation](https://rocm.docs.amd.com/en/docs-5.5.1/release/windows_support.html) was cited for clarification, with members advising to consult specific AMD resources.

- **Linux ROCm Tech Preview Inquiry**: A question arose about the availability of a Linux version for the **amd-rocm-tech-preview**, to which a member replied that while it may be planned, it is not expected soon.

- **GPU Quirks with ROCm**: Users pointed out distinctive sounds like coil whine when high-performance GPUs such as the 7900XTX are under heavy load, such as when running ROCm with LM Studio, indicating potential high resource utilization or hardware stress.

- **Model Loading Failures Detailed**: Errors like "Error loading model." with exit code 0 have been experienced by users trying to load models such as "Llama-2-7b-chat" and "Mistral instruct v0 1 7B" using the ROCm platform on Windows-based systems, with some success on different hardware or older versions.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://rocm.docs.amd.com/en/docs-5.7.1/release/windows_support.html">GPU and OS Support (Windows) — ROCm 5.7.1 Documentation Home</a>: no description found</li><li><a href="https://rocm.docs.amd.com/en/docs-5.5.1/release/windows_support.html">GPU and OS Support (Windows) — ROCm 5.5.1 Documentation Home</a>: no description found</li><li><a href="https://rocm.docs.amd.com/en/docs-5.5.1">AMD ROCm™ Documentation — ROCm 5.5.1 Documentation Home</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1227564205708939354)** (3 messages): 

- **DuckDuckGo as a solution?**: A member mentioned using DuckDuckGo for searches without an API, implying *Crewai* has restricted some functionality.
- **Curiosity about Model-Powered Searches**: Another member expressed enthusiasm about the concept of searching the internet through a model, following up on the DuckDuckGo topic.
  

---



**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1227622248752353331)** (8 messages🔥): 

- **Google's Code Completion AI, CodeGemma**: A member shared a [YouTube video](https://www.youtube.com/watch?v=Gb--4supXoo) introducing **Google's CodeGemma**, a code completion model that is available as a 7B pretrained variant offering powerful coding capabilities to developers.
- **Sleepless Inspiration from AI Research**: One member expressed gratitude for the inspiration provided by AI research and models, to the point of losing sleep, prompting another to forward the appreciation to a colleague, albeit jokingly noting the concern about the lack of sleep.
- **Technium Teasing More Models**: In response to a member's expression of appreciation for inspiring AI models, **teknium** hints that there are **"many more to come"**.
- **Is it a Hot Dog? AI Tutorial**: A new [YouTube video](https://www.youtube.com/watch?v=76BkMULO7uw) was introduced, showcasing a tutorial on distinguishing hot dogs from other images using **Ollama**, **Mistral**, and **LLava**.
- **Insightful North Korea Interview**: An [English-translated interview](https://www.youtube.com/watch?v=C84bzu9wXC0) where an expert talks about North Korea for 3 hours was shared, inviting members to explore the political and social dynamics of the country.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=Gb--4supXoo">Introducing CodeGemma Google&#39;s code completion model</a>: CodeGemma brings powerful yet lightweight coding capabilities to the community. CodeGemma models are available as a 7B pretrained variant that specializes in...</li><li><a href="https://www.youtube.com/watch?v=76BkMULO7uw">Hot dog or not with Ollama, Mistral and LLava</a>: In this tutorial we take a look at whether an image is hot dog or not using Ollama, mistral and lava#python #pythonprogramming #llm #ml #ai #aritificialintel...</li><li><a href="https://www.youtube.com/watch?v=C84bzu9wXC0">Стыдные вопросы про Северную Корею</a>: ERID: LjN8Jv34w Реклама. Рекламодатель ООО &quot;ФЛАУВАУ&quot; ИНН: 9702020445Радуем близких даже на расстоянии: https://flowwow.com/s/VDUD15Выберите подарок ко Дню ма...</li><li><a href="https://tenor.com/view/money-rain-erlich-bachman-tj-miller-silicon-valley-unicorn-gif-11481689">Money Rain Erlich Bachman Tj Miller Silicon Valley GIF - Money Rain Erlich Bachman Tj Miller - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1227555391500587038)** (8 messages🔥): 

- **Bidirectional Attention on the Starting Line**: A member discussed the potential need for bidirectional information flows in AI architecture, referencing SD3's success in rendering text due to bidirectional flows and improved training captions.

- **Speculating Bidirectional Attention in Mistral Models**: A direct *quote* was shared, speculating that **Mistral models** might use some form of bidirectional attention such as prefix language modeling, based on replicated results for various inputs and models.

- **Infini-attention Introduced for Scaling Transformers**: The community looked at a newly proposed method named **Infini-attention** from a paper ([Infini-attention](https://arxiv.org/abs/2404.07143)), which allows Transformer-based Large Language Models (LLMs) to process infinitely long inputs more efficiently.

- **The Renaissance of RNNs?**: A member linked a paper ([RNN Comeback](https://arxiv.org/abs/2402.19427)) indicating a resurgence of interest in RNNs or hybrid models, sparking discussion on the repetitive cycle where attempts to innovate often return to RNN-based architectures.

- **Google Unveils a Model Built on New RNN Architecture**: It was highlighted that Google released a 7B model leveraging the aforementioned RNN-based architecture very recently, which catches the attention of the AI community.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.07143">Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention</a>: This work introduces an efficient method to scale Transformer-based Large Language Models (LLMs) to infinitely long inputs with bounded memory and computation. A key component in our proposed approach...</li><li><a href="https://arxiv.org/abs/2402.19427">Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models</a>: Recurrent neural networks (RNNs) have fast inference and scale efficiently on long sequences, but they are difficult to train and hard to scale. We propose Hawk, an RNN with gated linear recurrences, ...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1227543960956702801)** (278 messages🔥🔥): 

- **All About Compute Constraints**: Multiple discussions highlighted the intense computational demands for training models like **Nous-Hermes-8x22b**. Costs were likened to $80k for necessary infrastructure, and there were significant hurdles with renting hardware, such as a lack of **infiniband interconnect** options among on-demand compute providers.

- **Comparing Finetunes**: The group discussed the performance of various finetuned models, noting that **Dolphin-2.2-70b** uses **QLoRA** and performs well, though not as good as full-parameter fine-tuning (**FFT**). Alternatives such as **LISA**, which unfreezes layers randomly each batch, also came up as a potentially superior method.

- **New Techniques and Hardware Options**: Discussion took place about potential ways to reduce the massive memory requirements for AI models, with various members pointing to **QLoRA** and **Unsloth** implementations. There was also anticipation for consumer-level GPUs with more RAM, given the high cost of server GPUs.

- **Hurdles with High Ambitions**: Users anticipate that **MistralAI** will eventually release an *instruct* version of their giant models, like their **Mixtral-8x22b**, to make them more manageable. The conversation suggested that while the raw models show promise, they are still *“wild monsters”* that require significant work to *tame* for specific applications.

- **Experiments and Outcomes**: Various experiments and observations were shared about using different models and benchmarks. It was mentioned that while **Mixtral-8x22b** seemed promising, it performed worse than expected on the **MT-Bench**, possibly due to a very small sample size and the expense of training.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/jphme/status/1778030213881909451">Tweet from Jan P. Harries (@jphme)</a>: @MistralAI first AGIEval results look great 👇 - thanks for releasing this beast, guys! 👏 https://x.com/jphme/status/1778028110954295486  ↘️ Quoting Jan P. Harries (@jphme)   First AGIEval results fo...</li><li><a href="https://foundershub.startups.microsoft.com/signup>">Microsoft for Startups FoundersHub</a>: no description found</li><li><a href="https://arxiv.org/abs/2404.05892">Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence</a>: We present Eagle (RWKV-5) and Finch (RWKV-6), sequence models improving upon the RWKV (RWKV-4) architecture. Our architectural design advancements include multi-headed matrix-valued states and a dynam...</li><li><a href="https://huggingface.co/lightblue/Karasu-Mixtral-8x22B-v0.1">lightblue/Karasu-Mixtral-8x22B-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/RWKV">RWKV (RWKV)</a>: no description found</li><li><a href="https://huggingface.co/v2ray/Mixtral-8x22B-v0.1">v2ray/Mixtral-8x22B-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://www.theregister.com/2024/04/09/intel_gaudi_ai_accelerator/">Intel Gaudi's third, final hurrah posited as H100 contender</a>: Goodbye dedicated AI hardware and hello to a GPU that fuses Xe graphics DNA with Habana chemistry</li><li><a href="https://techcrunch.com/2024/04/09/meta-confirms-that-its-llama-3-open-source-llm-is-coming-in-the-next-month/amp/">Meta confirms that its Llama 3 open source LLM is coming in the next month | TechCrunch</a>: Meta's Llama families, built as open-source products, represent a different philosophical approach to how AI should develop as a wider technology.</li><li><a href="https://en.wikipedia.org/wiki/List_of_logic_symbols">List of logic symbols - Wikipedia</a>: no description found</li><li><a href="https://openrouter.ai/models/mistralai/mixtral-8x22b">Mixtral 8x22B by mistralai | OpenRouter</a>: Mixtral 8x22B is a large-scale language model from Mistral AI. It consists of 8 experts, each 22 billion parameters, with each token using 2 experts at a time.  It was released via [X](https://twitter...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/fsdp_qlora.qmd">axolotl/docs/fsdp_qlora.qmd at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://huggingface.co/docs/transformers/v4.18.0/en/performance">Performance and Scalability: How To Fit a Bigger Model and Train It Faster</a>: no description found</li><li><a href="https://github.com/google/gemma.cpp">GitHub - google/gemma.cpp: lightweight, standalone C++ inference engine for Google&#39;s Gemma models.</a>: lightweight, standalone C++ inference engine for Google&#39;s Gemma models. - google/gemma.cpp</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 2-5X faster 80% less memory QLoRA &amp; LoRA finetuning</a>: 2-5X faster 80% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/Mihaiii/semantic-autocomplete">GitHub - Mihaiii/semantic-autocomplete: A blazing-fast semantic search React component. Match by meaning, not just by letters. Search as you type without waiting (no debounce needed). Rank by cosine similarity.</a>: A blazing-fast semantic search React component. Match by meaning, not just by letters. Search as you type without waiting (no debounce needed). Rank by cosine similarity. - Mihaiii/semantic-autocom...</li><li><a href="https://github.com/ContextualAI/gritlm">GitHub - ContextualAI/gritlm: Generative Representational Instruction Tuning</a>: Generative Representational Instruction Tuning. Contribute to ContextualAI/gritlm development by creating an account on GitHub.</li><li><a href="https://azure.microsoft.com/en-us/pricing/offers/ms-azr-0044p">Azure Free Trial | Microsoft Azure</a>: Start your free Microsoft Azure trial and receive $200 in Azure credits to use any way you want. Run virtual machines, store data, and develop apps.
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1227721101023379477)** (10 messages🔥): 

- **Llama 3 Model Coming Soon**: Meta has confirmed the upcoming release of its **Llama 3 models**, teasing some basic versions to be rolled out in the near future, as reported by [ZDNet](https://www.zdnet.com/article/meta-confirms-plans-to-start-rolling-out-llama-3-models-very-soon/) and [TechCrunch](https://techcrunch.com/2024/04/09/meta-confirms-that-its-llama-3-open-source-llm-is-coming-in-the-next-month/).
- **Fine-tuning Mistral at 7b on a Budget**: A member requested a step-by-step guide or tutorial for fine-tuning the 7b **Mistral** model, expressing interest in affordable cloud services for GPU use and achieving a gguf 8bit version for local run.
- **Alternatives to Full Fine-Tuning**: The Unsloth repository was suggested for cases where full fine-tuning may not be necessary, and **Qlora** was recommended to be used on Colab GPU, with the possibility of renting more powerful GPUs like the 3090 or 4090 from Vast.
- **Seeking Dataset for Logical Reasoning**: A member inquired about datasets geared toward reasoning with propositional and predicate logic over natural text, with a response linking to [Logic-LLM](https://github.com/teacherpeterpan/Logic-LLM)'s GitHub repository.
- **Genstruct Notebook for Scraped Data**: A member shared the discovery of a GitHub script [OllamaGenstruct/Paperstocsv.py](https://github.com/edmundman/OllamaGenstruct/blob/main/Paperstocsv.py) that closely matches the requirements for converting a webpage to markdown for genstruct input after initially contemplating writing a custom solution.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.zdnet.com/article/meta-confirms-plans-to-start-rolling-out-llama-3-models-very-soon/">Meta confirms plans to start rolling out Llama 3 models &apos;very soon&apos;</a>: Llama 3 is Meta&apos;s answer to OpenAI&apos;s GPT-4&#x200B;, Anthropic&apos;s Claude 3&#x200B;, Google&apos;s Gemini&#x200B;, and other LLMs.</li><li><a href="https://github.com/edmundman/OllamaGenstruct/blob/main/Paperstocsv.py">OllamaGenstruct/Paperstocsv.py at main · edmundman/OllamaGenstruct</a>: Contribute to edmundman/OllamaGenstruct development by creating an account on GitHub.</li><li><a href="https://github.com/teacherpeterpan/Logic-LLM/tree/main">GitHub - teacherpeterpan/Logic-LLM: The project page for &quot;LOGIC-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning&quot;</a>: The project page for &quot;LOGIC-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning&quot; - teacherpeterpan/Logic-LLM
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1227524315420033034)** (109 messages🔥🔥): 

- **Occult Wisdom for AI Guidance**: Archive.org's occult section is suggested as a valuable resource that **holds true** and may provide *guidance* for LLM development.
- **Enthusiasm for World-Sim's Return**: Members share excited anticipation for the **return of World-Sim**, discussing potential new features, with aspirations set for a comeback within the week.
- **World-Sim Potential Use Cases**: Discussion revolves around finding practical applications for **World-Sim**, with some members seeing it mainly as a fun tool and suggesting possible educational uses.
- **Local LLMs vs. Cloud Models**: The viability of running LLMs locally versus relying on **cloud-based models** prompted conversation, highlighting computational constraints and the trend towards mobile gaming.
- **Teasers and Tinkering with World-Sim's Return**: Speculation abounds as the World-Sim under-construction page changes, sparking theories on **teknium’s cryptic communication** and potential advancements like achieving AGI.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Fast_inverse_square_root">Fast inverse square root - Wikipedia</a>: no description found</li><li><a href="https://kbd.news/OGRE-cyberdeck-1835.html">OGRE cyberdeck</a>: OGRE is a doomsday or simply field cyberdeck, a knock-off of Jay Doscher's Recover Kit. Shared by rmw156.</li><li><a href="https://github.com/GitSquared/edex-ui/blob/master/media/screenshot_blade.png">edex-ui/media/screenshot_blade.png at master · GitSquared/edex-ui</a>: A cross-platform, customizable science fiction terminal emulator with advanced monitoring &amp; touchscreen support. - GitSquared/edex-ui
</li>
</ul>

</div>
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1227528061012938753)** (74 messages🔥🔥): 

- **Nuzzy Bot Engagement**: A bot named **Nuzzy** has been introduced in the chat for users to interact with. For best experience, a separate channel to talk to Nuzzy is recommended, and users can activate the bot by sending a specific activation command.
- **Unveiling of Udio Music Generator**: Users shared links to Twitter and Reddit discussing the new **Udio music generator**, comparing it to Suno, and highlighting its capabilities such as 90-second song cap with the free offering of 1200 songs per user per month. A detailed Reddit thread elaborated on Udio's superior music samples and potential release date.
- **Discussion on Nvidia’s Performance:** A shared [link](https://www.semianalysis.com/p/nvidia-blackwell-perf-tco-analysis) detailed an analysis of Nvidia's Blackwell GPU performance, including a comparison between their B100, B200, and GB200 models. The article discusses total cost of ownership and inference costs involved in using these GPUs.
- **Call for AI Engineering Playbook Contributions**: Users are discussing gathering material for an **AI engineering playbook**, which will address the transition from Jupyter notebooks to production. There’s an invitation for intros to senior engineers or team leaders who have experience in shipping features with large language models on mature engineering teams.
- **Various AI Discussions and Shares**: Members have shared numerous AI-related resources. Topics range from Meta’s AI hardware efforts, Jeremy Howard's **Practical Deep Learning for Coders** course relevance in 2024, the use of AI in creating a music video, and the potential impact of AI models like *Rerank 3* on search engines.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://stanford.zoom.us/j/99922151759?pwd=dW5CcUtVYkNybGZGY0hMWUZtVkZBZz09">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...</li><li><a href="https://x.com/AIatMeta/status/1778083237480321502">Tweet from AI at Meta (@AIatMeta)</a>: Introducing the next generation of the Meta Training and Inference Accelerator (MTIA), the next in our family of custom-made silicon, designed for Meta’s AI workloads.  Full details ➡️ https://go.fb.m...</li><li><a href="https://course.fast.ai/">Practical Deep Learning for Coders - Practical Deep Learning</a>: A free course designed for people with some coding experience, who want to learn how to apply deep learning and machine learning to practical problems.</li><li><a href="https://www.semianalysis.com/p/nvidia-blackwell-perf-tco-analysis">Nvidia Blackwell Perf TCO Analysis - B100 vs B200 vs GB200NVL72</a>: GPT-4 Profitability, Cost, Inference Simulator, Parallelism Explained, Performance TCO Modeling In Large &amp; Small Model Inference and Training</li><li><a href="https://x.com/TickerSymbolYOU/status/1778108179110387812">Tweet from Alex (@TickerSymbolYOU)</a>: My Google Cloud Next 2024 SUPERCUT saves you 91 minutes and skips all the marketing BS 00:00 - Google&#39;s New AI Hypercomputer 05:50 - Google Axion ARM CPU for Data Centers 07:25 - MASSIVE Gemini Up...</li><li><a href="https://www.honeycomb.io/blog/llms-demand-observability-driven-development">LLMs Demand Observability-Driven Development</a>: LLMs demand we modify our behavior and tooling in ways that will benefit even ordinary, deterministic software development. Find out why.</li><li><a href="https://x.com/itssandrakublik/status/1778422401648455694?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Sandra Kublik (@itsSandraKublik)</a>: Introducing our latest model, Rerank 3! 🚨  Further enhancing search and RAG systems.  What&#39;s inside? 🧑‍🍳 - 4k context length, - SOTA search accuracy on complex data  (e.g. emails, JSON document...</li><li><a href="https://x.com/daniel_eckler/status/1778421669201093057?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Eckler by Design ✦ (@daniel_eckler)</a>: C3PO x Childish Gambino 🤖 👑  100% AI (Official Music Video) @openAI + @runwayml + @suno_ai_ + @resembleai + @fable_motion + @midjourney + @topazlabs</li><li><a href="https://x.com/dylan522p/status/1777954675012305176?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Dylan Patel (@dylan522p)</a>: Nvidia Blackwell Perf TCO Analysis B100 vs B200 vs GB200NVL72 GPT-4 Profitability, Cost Inference Simulator Parallelism Explained Performance TCO Modeling In Large & Small Model Inference and Training...</li><li><a href="https://x.com/infobeautiful/status/1778059112250589561?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Information is Beautiful (@infobeautiful)</a>: Ask ChatGPT to pick a number between 1 and 100 - which does it pick?  (by @Leniolabs_)</li><li><a href="https://x.com/udiomusic/status/1778045322654003448?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from udio (@udiomusic)</a>: Introducing Udio, an app for music creation and sharing that allows you to generate amazing music in your favorite styles with intuitive and powerful text-prompting.  1/11</li><li><a href="https://github.com/GregorD1A1/TinderGPT">GitHub - GregorD1A1/TinderGPT</a>: Contribute to GregorD1A1/TinderGPT development by creating an account on GitHub.</li><li><a href="https://x.com/minchoi/status/1778074187778683253?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Min Choi (@minchoi)</a>: This is wild.  Udio just dropped and it&#39;s like Sora for music.  The music are insane quality, 100% AI. 🤯  1. &#34;Dune the Broadway Musical&#34;</li><li><a href="https://old.reddit.com/r/singularity/comments/1bzd4bo/its_been_confirmed_the_suno_killer_is_called_udio/">It's been confirmed - the &quot;Suno Killer&quot; is called Udio</a>: I've been investigating what some people have been calling the &quot;Suno killer&quot; - a music generation AI model that's supposedly 2 to 10 times better...
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1227692289808269392)** (6 messages): 

- **1-bit LLMs Paper Presentation Alert**: A presentation on the 1-bit Large Language Models paper is set to take place, promising insights into cost-effective and high-performance LLMs. Find the details and [join the event here](https://lu.ma/jcxntjox), and explore the paper [here](https://arxiv.org/abs/2402.17764).

- **Elicit Podcast Episode Released**: The latest podcast episode featuring Jungwon Byun and Andreas Stuhlmüller of Elicit is up. Listen and subscribe on [YouTube](https://www.youtube.com/watch?v=Dl66YqSIu5c).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2402.17764">The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits</a>: Recent research, such as BitNet, is paving the way for a new era of 1-bit Large Language Models (LLMs). In this work, we introduce a 1-bit LLM variant, namely BitNet b1.58, in which every single param...</li><li><a href="https://lu.ma/jcxntjox">LLM Paper Club (1-bit LLMs paper) · Luma</a>: This week @rj45 will be covering https://arxiv.org/abs/2402.17764 The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits Also submit and vote for our next paper:...</li><li><a href="https://www.youtube.com/watch?v=Dl66YqSIu5c&embeds_referring_euri=https%3A%2F%2Fwww.latent.space%2F&feature=emb_title">Supervise the Process of AI Research — with Jungwon Byun and Andreas Stuhlmüller of Elicit</a>: Timestamps:00:00:00 Introductions00:07:45 How Johan and Andreas Joined Forces to Create Elicit00:10:26 Why Products are better than Research00:15:49 The Evol...
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1227694310418546738)** (294 messages🔥🔥): 

- **Ternary Triumph or Trivial Trick?**: Members dived into discussions about the paper [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764), debating the efficiency of 1-bit large language models (LLMs). **BitNet b1.58** was praised for matching full-precision LLMs in performance while being cost-effective, yet skepticism arose about the true innovation behind ternary encoding and the reproducibility of results without access to detailed methodologies.
- **Paper Presentation Pandemonium**: Members encountered numerous technical difficulties with Discord's screen sharing feature during the paper presentation, leading to the exploration of alternative sharing platforms and lamenting the state of teleconferencing software.
- **From Papers to Practice**: Various links to GitHub were shared, providing resources like [BitNet-Transformers](https://github.com/Beomi/BitNet-Transformers), an implementation of 1-bit Transformers for LLMs, and a debate ensued about the practical implementations and hardware requirements for such models.
- **Musing Over MoEs**: Members discussed the concept and applications of Mixture of Experts (MoEs) models, linking resources like the [MoE blog post on Hugging Face](https://huggingface.co/blog/moe) and papers detailing expert specialization and load balancing. The conversation included reflections on the potential overlap and differences between MoEs and semantic routers at inference time.
- **Paper Club Picks and Pleasantries**: Participants wrapped up the session by selecting and suggesting new papers for future discussions, while thanking the presenters for their in-depth analysis and contributions. There was also talk of a potential move from Discord to Zoom for future meetings to avoid technical issues.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://matrix.to/#/#temporarylatentspace:matrix.org">You&apos;re invited to talk on Matrix</a>: no description found</li><li><a href="https://arxiv.org/abs/2402.17764">The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits</a>: Recent research, such as BitNet, is paving the way for a new era of 1-bit Large Language Models (LLMs). In this work, we introduce a 1-bit LLM variant, namely BitNet b1.58, in which every single param...</li><li><a href="https://llm-paper-club-asia-notes.vercel.app/papers/deekseek-moe">Nextra: the next docs builder</a>: Nextra: the next docs builder</li><li><a href="https://app.sli.do/event/bNV6mo3BFGhe8Bqzb1tonb/live/questions">Join Slido: Enter #code to vote and ask questions</a>: Participate in a live poll, quiz or Q&A. No login required.</li><li><a href="https://shapes.inc">Shapes, Inc.</a>: Shapes are AI friends that can talk to you on Discord</li><li><a href="https://arxiv.org/abs/2310.04793">FinGPT: Instruction Tuning Benchmark for Open-Source Large Language Models in Financial Datasets</a>: In the swiftly expanding domain of Natural Language Processing (NLP), the potential of GPT-based models for the financial sector is increasingly evident. However, the integration of these models with ...</li><li><a href="https://huggingface.co/blog/moe">Mixture of Experts Explained</a>: no description found</li><li><a href="https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1">mistral-community/Mixtral-8x22B-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://learning-exhaust.hashnode.dev/preview/6609ec4565bff73f1db1b51b">[Draft] 1.58 bits?</a>: no description found</li><li><a href="https://arxiv.org/abs/2402.18041">Datasets for Large Language Models: A Comprehensive Survey</a>: This paper embarks on an exploration into the Large Language Model (LLM) datasets, which play a crucial role in the remarkable advancements of LLMs. The datasets serve as the foundational infrastructu...</li><li><a href="https://www.youtube.com/watch?v=byCe7-c84d4">BloombergGPT - an LLM for Finance with David Rosenberg - 639</a>: Today we’re joined by David Rosenberg, head of the machine learning strategy team in the Office of the CTO at Bloomberg. In our conversation with David, we d...</li><li><a href="https://github.com/aurelio-labs/semantic-router">GitHub - aurelio-labs/semantic-router: Superfast AI decision making and intelligent processing of multi-modal data.</a>: Superfast AI decision making and intelligent processing of multi-modal data. - aurelio-labs/semantic-router</li><li><a href="https://github.com/AI4Finance-Foundation/FinGPT">GitHub - AI4Finance-Foundation/FinGPT: FinGPT: Open-Source Financial Large Language Models!  Revolutionize 🔥    We release the trained model on HuggingFace.</a>: FinGPT: Open-Source Financial Large Language Models!  Revolutionize 🔥    We release the trained model on HuggingFace. - AI4Finance-Foundation/FinGPT</li><li><a href="https://github.com/Beomi/BitNet-Transformers">GitHub - Beomi/BitNet-Transformers: 0️⃣1️⃣🤗 BitNet-Transformers: Huggingface Transformers Implementation of &quot;BitNet: Scaling 1-bit Transformers for Large Language Models&quot; in pytorch with Llama(2) Architecture</a>: 0️⃣1️⃣🤗 BitNet-Transformers: Huggingface Transformers Implementation of &amp;quot;BitNet: Scaling 1-bit Transformers for Large Language Models&amp;quot; in pytorch with Llama(2) Architecture - Beomi/...
</li>
</ul>

</div>
  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1227697534017732638)** (8 messages🔥): 

- **Hugging Chat Unleashes Command R+**: Hugging Chat has made the [CohereForAI/c4ai-command-r-plus](https://huggingface.co/chat/models/CohereForAI/c4ai-command-r-plus) model available for *free*, allowing integration of web search within the chat interface.

- **Community Highlight Reel**: Hugging Face showcases the community's contributions, such as the Portuguese tutorial on Hugging Face, fashion try-on AI, a repository on Deep Q Learning, tools for image augmentation, a RAG chatbot space, and an open-source alternative to character.ai.

- **Educational and Informative Spaces Spark Interest**: Members of the Hugging Face community have created valuable resources like a RAG chatbot using the wikipedia-small-3000-embedded dataset for generating responses without fine-tuning, and a step-by-step guide for building a neural network classifier, enhancing the collective knowledge.

- **No Training, Just Inference**: Clarifying a contribution, a member mentioned embedding the wikipedia-small-3000 dataset with the `mixedbread-ai/mxbai-embed-large-v1` model to retrieve information for use in a RAG chatbot space, emphasizing the use of RAG for inference rather than fine-tuning the model.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/datasets/not-lain/wikipedia-small-3000-embedded">not-lain/wikipedia-small-3000-embedded · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/chat/models/CohereForAI/c4ai-command-r-plus">CohereForAI/c4ai-command-r-plus - HuggingChat</a>: Use CohereForAI/c4ai-command-r-plus with HuggingChat</li><li><a href="https://www.youtube.com/watch?v=nK1hijr8Qng&t=74s">[IA a Z - 06] Apresentando o 🤗 Hugging Face</a>: 🤗🤗🤗🤗🤗🤗Se tem uma coisa que eu curto é ter um monte de opção de ferramenta para estudar!  Isso facilita demais o processo de aprender coisa nova, princi...</li><li><a href="https://huggingface.co/spaces/tonyassi/fashion-try-on">Fashion Try On - a Hugging Face Space by tonyassi</a>: no description found</li><li><a href="https://github.com/SuleymanEmreErdem/deep-q-learning-applications">GitHub - SuleymanEmreErdem/deep-q-learning-applications: My Deep Q-Learning projects</a>: My Deep Q-Learning projects. Contribute to SuleymanEmreErdem/deep-q-learning-applications development by creating an account on GitHub.</li><li><a href="https://huggingface.co/spaces/not-lain/RMBG1.4-with-imageslider">RMBG1.4 with imageslider - a Hugging Face Space by not-lain</a>: no description found</li><li><a href="https://github.com/RooTender/augmentator">GitHub - RooTender/augmentator: Ready-to-use tool for image augmentation</a>: Ready-to-use tool for image augmentation. Contribute to RooTender/augmentator development by creating an account on GitHub.</li><li><a href="https://not-lain-rag-chatbot.hf.space/"># RAG</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=76BkMULO7uw">Hot dog or not with Ollama, Mistral and LLava</a>: In this tutorial we take a look at whether an image is hot dog or not using Ollama, mistral and lava#python #pythonprogramming #llm #ml #ai #aritificialintel...</li><li><a href="https://github.com/EdoPedrocchi/RicercaMente">GitHub - EdoPedrocchi/RicercaMente: Open source project that aims to trace the history of data science through scientific research published over the years</a>: Open source project that aims to trace the history of data science through scientific research published over the years - EdoPedrocchi/RicercaMente</li><li><a href="https://ragdoll-studio.vercel.app/">no title found</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=oVJsJ0e6jWk">Where&#39;s My Pic Demo</a>: Hello everyone, I&#39;m Om Alve and in this video I&#39;m giving a demo of my project called &#39;Where&#39;s my pic?&#39;. This project solves the problem of searching through ...</li><li><a href="https://huggingface.co/blog/joey00072/mixture-of-depth-is-vibe">Mixture of Depth is Vibe</a>: no description found</li><li><a href="https://huggingface.co/blog/dcarpintero/building-a-neural-network-for-image-classification">Building a Neural Network Classifier from the Ground Up: A Step-by-Step Guide</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1227545273845153832)** (258 messages🔥🔥): 

- **Gradio Channel Queries**: Members were directed to specific Discord channels for questions related to Gradio, and one user was provided links to three different channels that discuss Gradio ([Channel 1](https://discord.com/channels/879548962464493619/1025174734427656283), [Channel 2](https://discord.com/channels/879548962464493619/1019296127847239751), and [Channel 3](https://discord.com/channels/879548962464493619/1014577787039924226)).

- **Challenging AI Interface Navigation**: Discussion focused on whether it's possible to train a model for near-perfect navigation of an OS GUI, exploring alternatives to direct pixel-based methods. Ideas included parsing applications as text and leveraging OS accessibility features.

- **Curiosity Over Model Size and Speed**: Questions arose regarding the difference between an 8x22B model and a 120B model, with insights shared about the usage of parameters during inference and the effectiveness of dense models compared to Mixture of Experts models.

- **Hugging Face Dataset Beginners' Resources**: A user inquired about resources for learning dataset basics, and was directed to Hugging Face's documentation, which offers guidance on creating datasets, builder scripts, metrics, and more ([Datasets documentation](https://huggingface.co/docs/datasets/index)).

- **In Search of a Diagram Generator**: One user was looking for a Hugging Face Space equivalent to DiagramGPT, which generates diagrams from text. Another member suggested using visual Q&A models that support inpainting with a strong prompt or checking out diagram creation tools within Hugging Face Spaces.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.eraser.io/diagramgpt">DiagramGPT – Powered by Eraser</a>: Generate technical diagrams from code or plain language prompts. Diagrams powered by Eraser.</li><li><a href="https://x.com/BigTechAlert">Tweet from undefined</a>: no description found</li><li><a href="https://huggingface.co/xai-org/grok-1">xai-org/grok-1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/nroggendorff/dolphin/blob/main/app.py">app.py · nroggendorff/dolphin at main</a>: no description found</li><li><a href="https://huggingface.co/docs/datasets/index">Datasets</a>: no description found</li><li><a href="https://huggingface.co/mlabonne/phixtral-2x2_8">mlabonne/phixtral-2x2_8 · Hugging Face</a>: no description found</li><li><a href="https://x.com/BigTechAlert/status/1778023873851871559>">Tweet from Big Tech Alert (@BigTechAlert)</a>: 🆕 @huggingface has started following @realmrfakename</li><li><a href="https://huggingface.co/spaces/nroggendorff/cascade/blob/main/app.py">app.py · nroggendorff/cascade at main</a>: no description found</li><li><a href="https://huggingface.co/spaces/macadeliccc/laser-dolphin-mixtral-chat/blob/main/app.py">app.py · macadeliccc/laser-dolphin-mixtral-chat at main</a>: no description found</li><li><a href="https://github.com/moritztng/fltr">GitHub - moritztng/fltr: Like grep but for natural language questions. Based on Mistral 7B or Mixtral 8x7B.</a>: Like grep but for natural language questions. Based on Mistral 7B or Mixtral 8x7B. - moritztng/fltr</li><li><a href="https://youtu.be/rSKMYc1CQHE?si=aEYaxyGwK7LdCLx6">Coding Adventure: Simulating Fluids</a>: Let&#39;s try to convince a bunch of particles to behave (at least somewhat) like water.Written in C# and HLSL, and running inside the Unity engine.Source code:h...</li><li><a href="https://youtu.be/Qz0KTGYJtUk?si=dq_Ptn1lpmwdNrt5">Coding Adventure: Ray Tracing</a>: I tried creating a custom ray/path tracing renderer. Featuring: maths, shaders, and cats!This project was written in C# and HLSL, and uses the Unity game eng...</li><li><a href="https://github.com/BrutPitt/glChAoS.P">GitHub - BrutPitt/glChAoS.P: 3D GPUs Strange Attractors and Hypercomplex Fractals explorer - up to 256 Million particles in RealTime</a>: 3D GPUs Strange Attractors and Hypercomplex Fractals explorer - up to 256 Million particles in RealTime - BrutPitt/glChAoS.P
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1227828614133973052)** (1 messages): 

- **Podman for AI Security**: A discussed video, titled ["The Root of AI Security is Rootless. On Podman: For GenAI Microservices"](https://youtu.be/YtxdWLSPo0U), provides an overview of installing **Podman** via terminal, highlighting its importance in AI security within containerized microservices environments. It is suggested as a more secure alternative to Docker for AI applications.

**Link mentioned**: <a href="https://youtu.be/YtxdWLSPo0U">The Root of AI Security is Rootless. On Podman: For GenAI Microservices</a>: An overview on installing @Podman from the terminal. #Podman #containers #AI #genAI #Docker #Linux #EdTech #deeplearning #microservices

  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1227558577523134504)** (7 messages): 

- **Tiny But Mighty Model Discovery**: A new [information extraction demo](https://huggingface.co/spaces/urchade/gliner_multiv2.1) featuring a tiny yet robust model for multilingual purposes was introduced, showing its efficacy through a user-shared image.
- **Quantum Leap with Quanto**: A GitHub repository containing a [notebook for using Quanto with Transformers](https://github.com/andysingal/llm-course/tree/main/Quantization) was added to the conversation, suggesting further exploration of quantization techniques.
- **HuggingFace Model Playground**: The [Marimo app](https://marimo.app/l/tmk0k2) was shared as a playground for HuggingFace models, encouraging users to save their code and experiment within this new tool.
- **RecurrentGemma Emerges**: A Medium article titled ["RecurrentGemma: A Leap Beyond Transformers with PyTorch Integration"](https://medium.com/technology-hits/recurrentgemma-a-leap-beyond-transformers-with-pytorch-integration-f6bb443766aa) was highlighted, signaling a potential paradigm shift in AI modeling.
- **Andrej Karpathy Simplifies LLM Implementation**: A GitHub project by Andrej Karpathy was mentioned, offering a stripped-down [LLM training implementation in raw C/CUDA](https://github.com/karpathy/llm.c).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/urchade/gliner_multiv2.1">GLiNER-Multiv2.1 - a Hugging Face Space by urchade</a>: no description found</li><li><a href="https://marimo.app/l/tmk0k2">marimo | a next-generation Python notebook</a>: Explore data and build apps seamlessly with marimo, a next-generation Python notebook.</li><li><a href="https://github.com/andysingal/llm-course/tree/main/Quantization">llm-course/Quantization at main · andysingal/llm-course</a>: Contribute to andysingal/llm-course development by creating an account on GitHub.</li><li><a href="https://github.com/karpathy/llm.c">GitHub - karpathy/llm.c: LLM training in simple, raw C/CUDA</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1227674375474516081)** (11 messages🔥): 

- **Color Confusion in Model Testing**: A member shared their test results, mentioning that while the model achieved *amazing results*, it sometimes mixed up colors, such as inverting the colors of a t-shirt and pants.
- **Pruning Strategies for LLMs**: A member found value in re-reading a paper on pruning, discussing the use of targeted datasets for specific use cases and considering the possibility of expert pruning in large models like *Mixtral 8x22B*.
- **Tutorial on Classifying Hot Dogs**: A [YouTube tutorial](https://www.youtube.com/watch?v=76BkMULO7uw) showcasing how to determine if an image depicts a hot dog or not through the use of models such as **Ollama**, **Mistral**, and **LLava** was linked.
- **Arduino-Based Incubator on GitHub**: A member shared their [project on GitHub](https://github.com/DHTScienceGuy/Incubator), an Arduino-based self-regulating chicken incubator, open for the community to explore.
- **Hugging Face Concepts in Portuguese**: An educational post and video introducing the basic concepts of Hugging Face were published, providing resources for Portuguese speakers in Brazil starting out with AI. Links can be found for the [post](https://iatalk.ing/hugging-face/) and the [video](https://www.youtube.com/watch?v=nK1hijr8Qng&t=74s).

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=76BkMULO7uw">Hot dog or not with Ollama, Mistral and LLava</a>: In this tutorial we take a look at whether an image is hot dog or not using Ollama, mistral and lava#python #pythonprogramming #llm #ml #ai #aritificialintel...</li><li><a href="https://vimeo.com/933289700">test</a>: This is &quot;test&quot; by Test Account on Vimeo, the home for high quality videos and the people who love them.</li><li><a href="https://github.com/DHTScienceGuy/Incubator">GitHub - DHTScienceGuy/Incubator: My Arduino Based Self Regulating Chicken Incubator</a>: My Arduino Based Self Regulating Chicken Incubator - DHTScienceGuy/Incubator</li><li><a href="https://iatalk.ing/hugging-face/">Apresentando o 🤗 Hugging Face</a>: Olá! Hoje quero te apresentar uma ferramenta obrigatória para quem está entrando, ou já faz parte, do mundo de Inteligência Artificial: O Hugging Face Hub , hf para os íntimos, ou apenas 🤗 O Huggin…</li><li><a href="https://www.youtube.com/watch?v=nK1hijr8Qng&t=74s">[IA a Z - 06] Apresentando o 🤗 Hugging Face</a>: 🤗🤗🤗🤗🤗🤗Se tem uma coisa que eu curto é ter um monte de opção de ferramenta para estudar!  Isso facilita demais o processo de aprender coisa nova, princi...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1227539324963393547)** (6 messages): 

- **Seeking Guidance for Intelligent Customer Service Chat System**: A user expressed interest in building a **multi-turn dialogue system** for intelligent customer service, and asked for research papers or works that could assist in this endeavor.
- **On the Hunt for Samplers and Schedulers Mathematics**: A user inquired about mathematical papers on **schedulers and samplers** after studying about ddpm and ddim, aiming to understand basic ones that followed the mentioned methods.
- **KD-Diffusion Paper and Blog Recommended**: Another member recommended the **k-diffusion** paper as a substantial resource for understanding schedulers and provided a [Medium blog link](https://medium.com/@isamu-website/understanding-k-diffusion-from-their-research-paper-and-source-code-55ae4aa802f) for a simplified explanation of the concepts.
  

---


**HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1227825823084712097)** (1 messages): 

- **Multi-GPU Support with Device Map**: *Diffusers* now supports experimental `device_map` functionality for balancing model pipeline distribution across multiple GPUs. This feature is especially beneficial for setups with multiple low-VRAM GPUs, and more strategies will be added depending on community interest. [Read the documentation](https://huggingface.co/docs/diffusers/main/en/training/distributed_inference#device-placement).

**Link mentioned**: <a href="https://huggingface.co/docs/diffusers/main/en/training/distributed_inference#device-placement">Distributed inference with multiple GPUs</a>: no description found

  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1227521406938644480)** (14 messages🔥): 

- **Eradicating Watermarks with Aladdin-Persson AI**: A GitHub repository for the Aladdin-Persson-AI-Watermark-Destroy project was shared. It's an older tool but is mentioned to still be effective ([GitHub Repo](https://github.com/Firdavs-coder/Aladdin-Persson-AI-Watermark-Destroy)).

- **NVidia GPU Process Monitoring Advice**: Using `nvidia-smi -l` to monitor GPU processes in a loop was suggested for continual tracking, while an external recommendation was to use nvitop, an interactive NVIDIA-GPU process viewer ([nvitop GitHub](https://github.com/XuehaiPan/nvitop)).

- **Navigating Video Correction Techniques**: A conversation about starting with image denoising and artifact removal as a foundation for video processing was offered, including two suggested papers with innovative approaches: one presenting the NAFNet model and another providing a large dataset for image restoration ([NAFNet ARXIV](https://arxiv.org/abs/2204.04676), [Image Restoration Dataset](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Li_LSDIR_A_Large_Scale_Dataset_for_Image_Restoration_CVPRW_2023_paper.pdf)).

- **Augmentation for Restoration Model Generalization**: The importance of augmentation in training restoration models was underlined, with references to two papers detailing their augmentation pipelines, BSRGAN and Real-ESRGAN, to inform strategies for handling diverse image degradations ([BSRGAN ARXIV](https://arxiv.org/abs/2103.14006), [Real-ESRGAN ARXIV](https://arxiv.org/abs/2107.10833)).

- **The Challenges of Specific Video Data Sets**: A user detailed difficulties with training models on limited video data captured from a vehicle, affected by noise and variable lighting. The discussion moved toward analyzing the video to create a balanced dataset after using other data sets for training was not effective.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2103.14006">Designing a Practical Degradation Model for Deep Blind Image Super-Resolution</a>: It is widely acknowledged that single image super-resolution (SISR) methods would not perform well if the assumed degradation model deviates from those in real images. Although several degradation mod...</li><li><a href="https://arxiv.org/abs/2107.10833">Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data</a>: Though many attempts have been made in blind super-resolution to restore low-resolution images with unknown and complex degradations, they are still far from addressing general real-world degraded ima...</li><li><a href="https://github.com/Firdavs-coder/Aladdin-Persson-AI-Watermark-Destroy">GitHub - Firdavs-coder/Aladdin-Persson-AI-Watermark-Destroy: Aladdin-Persson-AI-Watermark-Destroy Public</a>: Aladdin-Persson-AI-Watermark-Destroy Public. Contribute to Firdavs-coder/Aladdin-Persson-AI-Watermark-Destroy development by creating an account on GitHub.</li><li><a href="https://github.com/XuehaiPan/nvitop">GitHub - XuehaiPan/nvitop: An interactive NVIDIA-GPU process viewer and beyond, the one-stop solution for GPU process management.</a>: An interactive NVIDIA-GPU process viewer and beyond, the one-stop solution for GPU process management. - XuehaiPan/nvitop</li><li><a href="https://arxiv.org/abs/2204.04676">Simple Baselines for Image Restoration</a>: Although there have been significant advances in the field of image restoration recently, the system complexity of the state-of-the-art (SOTA) methods is increasing as well, which may hinder the conve...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1227534442701258764)** (10 messages🔥): 

- **Evaluating Language Models with Cosine Similarity**: A discussion centered around using cosine similarity to evaluate language models, considering the outputs generated against a markscheme (vector C) and tutoring principles (vector D). It's suggested to try *weighted pooling* over average pooling for vector B to potentially prioritize certain aspects, like the markscheme, in the evaluation protocol.

- **In Search of GPT-4 Alternatives**: A member inquired about alternatives to llama2/GPT-4 that allow for commercial use and are trainable on a 24-GB GPU, where even older versions could be considered viable options.

- **Seeking Long Context Models**: There's a request for encoder-decoder models capable of handling longer contexts, approximately 10-15k tokens. Suggestions included looking into **BigBird** and **Longformer** as potential options.

- **Hugging Face Trainer's Pause-Resume Feature**: A question about the ability to pause and resume training using Hugging Face's `Trainer` was answered affirmatively, pointing out the `resume_from_checkpoint` option in `trainer.train()` for this purpose.

- **Script Assistance for Model Training**: A request for help regarding a script to train a model using [transformers](https://huggingface.co/transformers/) and additional components like Bits and Bytes for quantization and Lora for low-rank adaptations. The script included the use of `accelerate launch` for execution and sought verification for the correct implementation and model saving process.
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1227555268133650492)** (8 messages🔥): 

- **Clarifications on Scheduler Math**: A member asked for recommendations on materials for understanding the math behind various AI schedulers and samplers post-DDPM and DDIM. It was suggested to study [fast.ai Part 2 classes](https://course.fast.ai/Lessons/part2.html), examine [code implementations on GitHub](https://github.com/huggingface/diffusers/issues?q=sort%3Acreated-asc+label%3A%22good+first+issue%22%2C%22Good+second+issue%22%2Chacktoberfest), read [Hugging Face's blog posts on diffusion](https://huggingface.co/blog?tag=diffusion&p=1), and follow [discussions on the topic](https://github.com/huggingface/diffusers/discussions?discussions_q=sort%3Atop).

- **Distributed Inference with MultiControlnet**: A member inquired about loading MultiControlnet on multiple GPUs for inference with each GPU having around 10GBs of VRAM, seeking a solution other than **Hugging Face's Accelerate** due to VRAM limitations. After a failed initial attempt with Accelerate, they were directed to [a more detailed guide on device placement](https://huggingface.co/docs/diffusers/main/en/training/distributed_inference#device-placement) to address their VRAM concerns.

- **Seeking Layer Decomposer Tools**: A user was on the lookout for a Layer Decomposer similar to one featured on a [webpage](https://cre8tiveai.com/ld) but couldn't find clear information on how to implement such a tool. They requested any leads on repositories or articles concerning the topic.

- **Performance Trade-offs with "Balanced" Device_map**: A community member questioned how using the **device_map** feature with a "balanced" strategy impacts the **inference time and resource efficiency**, particularly on GPUs with lower VRAM. They sought advice on configurations that optimize performance within memory constraints.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/diffusers/main/en/training/distributed_inference">Distributed inference with multiple GPUs</a>: no description found</li><li><a href="https://cre8tiveai.com/ld"> Layer Decomposer（Layer separation AI）｜Image and video editing AI tool: cre8tiveAI</a>: An AI-based SaaS that solves a variety of photo and illustration editing tasks in under 10 seconds, such as automatic painting and increasing the resolution of images and videos, as well as clipping, ...</li><li><a href="https://huggingface.co/docs/diffusers/main/en/training/distributed_inference#device-placement">Distributed inference with multiple GPUs</a>: no description found
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1227718650820169758)** (1 messages): 

- **Perplexity AI Partners with Raycast**: Perplexity AI has announced a collaboration with Raycast, providing **Perplexity Pro free for 3 months** to new annual Raycast Pro subscribers, with 6 months free if the advanced AI add-on is included. Find out more about this partnership and the integration of AI models on Mac in Raycast's [blog post](https://www.raycast.com/blog/more-ai-models).
- **Celebrating the Milestone of ChatGPT**: The [launch of ChatGPT](https://openai.com/blog/chatgpt) on November 30, 2022, achieved a record by reaching **100 million monthly active users in just two months**. The blog post highlights the surge in defining an AI strategy among all software businesses and discusses the potential of LLMs without the looming fear of an apocalypse.

**Link mentioned**: <a href="https://www.raycast.com/blog/more-ai-models">One interface, many LLMs - Raycast Blog</a>: Raycast AI gets more powerful with Anthropic Claude 3, Perplexity, and more models - making it the perfect UI for AI

  

---


**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1227522057873526795)** (266 messages🔥🔥): 

- **Model Preferences and Trade-offs**: Users discussed the trade-off between smaller context windows and smarter AI models with one advocating for Opus 4k due to its practicality. They pondered future choices such as Opus 200k or Haiku 200k, noting Opus 4k's likely popularity due to its efficiency for many short queries.

- **Claude 3 Opus**: There is a debate over whether Pro mode should be on or off when using Claude 3 Opus in writing mode. One user advises turning it off as Pro mode is said to leverage web searches which may not benefit writing mode.

- **Perplexity's Image Inclusion Limitations**: Members reported difficulties when attempting to include images directly in Perplexity's responses. A link was shared to demonstrate this, and one user successfully modified a prompt to work, suggesting there might be a workaround.

- **Integration with External Tools and Raycast Partnership**: Discussion about Perplexity integrated into external tools like Web browsers or Raycast, with a user sharing a [Perplexity AI announcement on Twitter](https://www.perplexity.ai/search/What-is-Raycast-.ItnfuzRRECM3I.88NKApg) regarding a Raycast collaboration. Users chatted about their experiences and benefits of using Raycast which integrates with Perplexity.

- **API Authentication Issues**: A user experienced a 401 error when attempting to use the Perplexity API, which was resolved upon figuring out that payment auto top up was inactive, affecting their authorization.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.tickertea.com/subscribe">Ticker Tea</a>: Delivering daily updates on the news that matters: financial markets. </li><li><a href="https://docs.perplexity.ai/">pplx-api</a>: no description found</li><li><a href="https://x.com/perplexity_ai/status/1778067977566294448">Tweet from Perplexity (@perplexity_ai)</a>: We teamed up with Raycast to make knowledge accessible anywhere, anytime on your Mac. New annual Raycast Pro subscribers get Perplexity Pro for free for 3 months, or 6 months if you include the advanc...</li><li><a href="https://www.youtube.com/watch?v=0O2yTG3n1Vc">I tested the Humane AI Pin - it&#39;s not good.</a>: I spend a LOT of time trying to make my videos as concise, polished and useful as possible for you - if you would like to support me on that mission then con...</li><li><a href="https://github.com/wallabag/wallabagger/blob/bc9bae830c2f51403b1679efdfab9a497365f05d/wallabagger/js/options.js#L109">wallabagger/wallabagger/js/options.js at bc9bae830c2f51403b1679efdfab9a497365f05d · wallabag/wallabagger</a>: Chrome / Firefox / Opera plugin for wallabag v2. Contribute to wallabag/wallabagger development by creating an account on GitHub.</li><li><a href="https://github.com/ollama/ollama">GitHub - ollama/ollama: Get up and running with Llama 2, Mistral, Gemma, and other large language models.</a>: Get up and running with Llama 2, Mistral, Gemma, and other large language models. - ollama/ollama</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1c0tdsb/comment/kyzsho1/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1227575877185962055)** (14 messages🔥): 

- **Exploring Video Game Queries**: Members searched Perplexity for insights on video games, possibly seeking analysis or comparisons between games.
- **Delving into Financials**: One user explored the net worth related to olives, indicating an interest in the economic aspects of agricultural products.
- **Pursuit of Meta Custom AI**: A search query on Perplexity mentioned "Meta custom AI," hinting at an exploration of Meta's proprietary artificial intelligence systems or custom AI solutions.
- **Tech Enthusiasts Probe Chip Developments**: The Gaudi 3 chip was a subject of interest, as one member sought information, potentially about its specs or performance in AI tasks.
- **Comparing Analytical Methods**: A user's search about Neowave versus Elliott Wave indicates a discussion on different analysis techniques, likely within the context of financial markets.

  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1227622224664596561)** (16 messages🔥): 

- **Paste to Prompt for Large Inputs**: Users can bypass the lack of file uploads on **labs.perplexity** by converting files to plain text and pasting up to 199k tokens directly into the prompt field at no cost.
- **Live Web Responses Available**: The **Perplexity API** does not offer direct live web responses, but users can obtain information from the web using the *sonar online models*.
- **Claude Opus Model Unsupported**: Inquiries about the availability of the Claude Opus model on **Perplexity API** revealed it is not supported. Users interested in available models were directed to the official [model documentation](https://docs.perplexity.ai/docs/model-cards).
- **Clarifying API Features**: While the feature roadmap for **Perplexity API** may change, users can apply for citation access to potentially include source page URLs in their requests.
- **API Mimics Web Version**: Users can obtain answers through the **Perplexity API** that are similar to those provided by the web version of Perplexity, leveraging appropriate request tactics and parameter settings.

**Link mentioned**: <a href="https://perplexity.typeform.com/to/j50rnNiB)">Discover Typeform, where forms = fun</a>: Create a beautiful, interactive form in minutes with no code. Get started for free.

  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1227702431606116482)** (1 messages): 

- **Meta's Impressive AI Training Infrastructure**: *Meta Training and Inference Accelerator (MTIA)* boasts a remarkable **354 TFLOPS/s (INT8) performance at just 90 watts**. The [official blog post](https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/) details how Meta's next-generation infrastructure is built to support GenAI products, recommendation systems, and AI research, with expectations for substantial growth to meet increasing compute demands.

**Link mentioned**: <a href="https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/">no title found</a>: no description found

  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/)** (1 messages): 

mobicham: https://github.com/BobMcDear/attorch
  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1227752239091945574)** (63 messages🔥🔥): 

- **CUDA Kernels Collection Available**: A new version of llm.c includes a collection of **Cuda kernels** which can be found at [karpathy/llm.c on GitHub](https://github.com/karpathy/llm.c/tree/master/dev/cuda).
- **Library for Lightning-Fast Linear Layers**: A library for super fast linear layers employing half precision accumulate was announced, claiming to achieve **>300 tflops on a 4090 GPU**, significantly outperforming PyTorch. It is designed for very fast inference and the repository is hosted at [torch-cublas-hgemm on GitHub](https://github.com/aredden/torch-cublas-hgemm).
- **Accuracy Maintained in Speedy Inference Tool**: The creator of the high-performance linear layers library mentioned that, although the library is much faster than nn.Linear layers for inference, it produces nearly identical results for all shape possibilities and does not yet support gradients.
- **Potential Issues and Future Updates for the Inference Library**: The author of the fast linear layers library admitted that it's a very new project and might have bugs, but they confirmed the results have been thoroughly tested. They also indicated a plan to add gradient support soon.
- **Discussion on Precision and Performance**: Clarification was provided that the high-speed inference library uses **FP16** with FP16 accumulate, offering significantly faster performance for consumer GPUs over FP32 with FP32 accumulate, which doesn't appear to be as advantageous on datacenter cards like the RTX 6000 ADA.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/tree/master/dev/cuda">llm.c/dev/cuda at master · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/aredden/torch-cublas-hgemm">GitHub - aredden/torch-cublas-hgemm: PyTorch half precision gemm lib w/ fused optional bias + optional relu/gelu</a>: PyTorch half precision gemm lib w/ fused optional bias + optional relu/gelu - aredden/torch-cublas-hgemm
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1227781176962584606)** (3 messages): 

- **Troubles Quantizing ViT Models**: A user encountered an issue while attempting to quantize the `google/vit-base-patch16-224-in21k` model from HuggingFace, receiving a `RuntimeError: empty_strided not supported on quantized tensors yet`. They referenced a [GitHub issue](https://github.com/pytorch/pytorch/issues/74540) for more details and are seeking a guide on quantization and pruning.

- **FlashAttention Woes with BERT**: Another user is working on adding support for `flashattention2` to a BERT model and has discovered discrepancies between the patched and unpatched models. They are looking for insights into the issue.

**Link mentioned**: <a href="https://github.com/pytorch/pytorch/issues/74540">No factory functions for strided quantized tensors · Issue #74540 · pytorch/pytorch</a>: 🐛 Describe the bug For non-quantized tensors, there is both empty and empty_strided. However, for quantized tensors there are only empty variants for functions. This means that it is difficult for .....

  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1227551275869143041)** (7 messages): 

- **Study Group for PMPP Book Lectures**: A member suggested setting up a viewing party for the University of Illinois lectures based on the PMPP book with a focus on interactive discussions. The proposed timing is early CET on weekdays, with interested participants being asked to reply.
- **Running Your Own Race**: A gentle reminder was given to the community to not feel left out with the pace of learning within the Discord, emphasizing the progress in individual learning with a comparison to learning languages.
- **Ease of Learning CUDA Over German**: A humorous comparison was struck between learning **CUDA** and German, portraying CUDA as much easier, with others in the channel attesting to this.
- **Scheduled Viewing Party Session and Discord Group**: A Discord group was created for those interested in the study group, with the first session scheduled for Saturday at 7:30 GMT/8:30 BST/9:30 CET, with an invitation link provided: [Study Group Invite](https://discord.gg/XwFJRKH9).
- **Suggestion to Utilize Existing Voice Channels**: In response to setting up a new group for study sessions, it was proposed to use one of the existing voice channels for potentially better engagement.

**Link mentioned**: <a href="https://discord.gg/XwFJRKH9">Join the PMPP UI lectures timezones Discord Server!</a>: Check out the PMPP UI lectures timezones community on Discord - hang out with 4 other members and enjoy free voice and text chat.

  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1227939087256649781)** (3 messages): 

- **Dataset Delivery**: A member whimsically announced the arrival of an "extra large dataset."
- **Task Organization Inquiry**: There was a suggestion about creating a list of next tasks for the team.
- **Testing Commitment**: A member mentioned they have testing tasks lined up with "mamba."
  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1227574193789341726)** (3 messages): 

- **Eureka! It's Goku**: The server picture has been identified as the character **Goku**.
- **Milestone Achieved**: The server has successfully surpassed **5000 members**.
- **Sage Advice on Knowledge Consumption**: A member suggested that **reading once a week** and letting the problems guide the depth of research can be more effective than excessive consumption of information, stating that being a "cons00mer" is always bad for the brain.
  

---


**CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1227528423958773860)** (76 messages🔥🔥): 

- **Quantization Scripts and Benchmarks Shared**: [Quantization benchmark scripts](https://github.com/mobiusml/hqq/blob/master/examples/llama2_benchmark/quant_llama2_hqq_demo.py) have been shared for Hugging Face models with pre-defined parameters to return specific performance tables. Members discussed using base models as opposed to chat models and the possibility of disabling cache for execution.
- **Up-to-date Efficient Code**: The [working quantization code](https://github.com/mobiusml/hqq/blob/ao_int4_mm/hqq/core/torch_lowbit.py) with proper perplexity scores and details on speed benchmarks on different GPUs like 3090, 4090, and A100 using HQQLinear and torchao int4 kernel has been provided.
- **Performance Discussions on int4 Kernels**: Concerns were raised about int4 kernel performance, specifically why the `Int4: AOInt4 backend` is significantly slower compared to PyTorch with `torch.compile` on the 3090 GPU. The discussion touched upon possible reasons and the need for testing on an A100 GPU.
- **Reproducibility and Conversion Issues Addressed**: Members encountered issues with reproducibility of results and confirmed perplexity discrepancies when quantizing along different axes. These challenges led to sharing conversion stages and clarified methods for converting between hqq quantization parameters and tinygemm int4wo, with a deep dive into how row altering can affect accuracy.
- **Discrepancies in Perplexity Metrics and Potential Causes**: There was a focus on perplexity metrics with discrepancies observed between HQQ and GPTQ results. Alongside some troubleshooting, the discussion pointed to potential issues like dropout, layernorm eps values, and differences in weight conversion processes being possible factors affecting perplexity.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/mobiusml/hqq/blob/ao_int4_mm/hqq/core/torch_lowbit.py">hqq/hqq/core/torch_lowbit.py at ao_int4_mm · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq</li><li><a href="https://github.com/mobiusml/hqq/blob/ao_int4_mm/hqq/core/torch_lowbit.py#L137-L139">hqq/hqq/core/torch_lowbit.py at ao_int4_mm · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq</li><li><a href="https://github.com/mobiusml/hqq/blob/master/examples/llama2_benchmark/quant_llama2_hqq_demo.py">hqq/examples/llama2_benchmark/quant_llama2_hqq_demo.py at master · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq</li><li><a href="https://github.com/pytorch-labs/gpt-fast/blob/main/scripts/convert_hf_checkpoint.py#L89">gpt-fast/scripts/convert_hf_checkpoint.py at main · pytorch-labs/gpt-fast</a>: Simple and efficient pytorch-native transformer text generation in &lt;1000 LOC of python. - pytorch-labs/gpt-fast</li><li><a href="https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py">transformers/src/transformers/models/llama/modeling_llama.py at main · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers</li><li><a href="https://github.com/mobiusml/hqq/blob/ao_int4_mm/hqq/core/torch_lowbit.py#L135">hqq/hqq/core/torch_lowbit.py at ao_int4_mm · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq</li><li><a href="https://github.com/zhxchen17/gpt-fast/blob/hqq_4bit_llama2_7b/scripts/convert_hf_checkpoint.py#L57">gpt-fast/scripts/convert_hf_checkpoint.py at hqq_4bit_llama2_7b · zhxchen17/gpt-fast</a>: Simple and efficient pytorch-native transformer text generation in &lt;1000 LOC of python. - zhxchen17/gpt-fast</li><li><a href="https://github.com/pytorch/pytorch/blob/8aa08b8b9d1fab2a13dc5fbda74c553cb2a08729/aten/src/ATen/native/cuda/int4mm.cu#L805-L860">pytorch/aten/src/ATen/native/cuda/int4mm.cu at 8aa08b8b9d1fab2a13dc5fbda74c553cb2a08729 · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch</li><li><a href="https://gist.github.com/mobicham/4b08fb0bdf4c3872e5bbf68ec9803137">hqq_eval_int4mm_noppl.py</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://github.com/zhxchen17/gpt-fast/commit/5cdb8bd61fcacccf018cd7a1c49417442e03644a">HQQ 4 bit llama 2 7b · zhxchen17/gpt-fast@5cdb8bd</a>: export MODEL_REPO=meta-llama/Llama-2-7b-hf scripts/prepare.sh $MODEL_REPO python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int4-hqq --groupsize 64 python generate.py --...</li><li><a href="https://github.com/mobiusml/hqq/blob/ao_int4_mm/hqq/core/torch_lowbit.py#L197-L221">hqq/hqq/core/torch_lowbit.py at ao_int4_mm · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton-viz](https://discord.com/channels/1189498204333543425/1225499141241573447/1227739060232261686)** (3 messages): 

- **Fine-tuning the triton-viz bot enhancements**: Members discussed potential improvements for the **triton-viz chatbot**, with suggestions including modifying **hyperlinks** and adding **code annotations** to make them step by step, indicating a phased approach to upgrades.
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1227548729788207104)** (67 messages🔥🔥): 

- **CUDA Forward Pass Efficiency Gains**: A user reported their **pure CUDA** forward pass running at 110ms per iteration compared to **PyTorch** at 180ms per iteration for a GPT-2 model with specific parameters (B=4, T=1024, GPT-2 124M). They plan to investigate the discrepancy further.
  
- **Potential Optimization with CUDA's C Subset**: Discussion centered on optimizing in pure CUDA's C subset, weighing the limitations and potential need for inline assembly to use tensor cores. **cuBLAS** is confirmed to use tensor cores, and there's a suggestion that `__restrict__` could optimize compiler handling of pointer arguments.

- **Implementing Warp-Wide Reductions and Kernel Fusion**: A member shared insights on using **cooperative groups** and **templating in CUDA**, including warp-wide reductions, kernel fusion at compile time, and use of macros in place of templates. They also provided a detailed **[LayerNorm kernel example](https://godbolt.org/z/6zq6Kfecc)** that uses cooperative groups and achieves substantial speed improvements on an A4000 GPU.

- **Evaluating Cooperative Groups for CUDA**: A member highlighted **[Cooperative Groups](https://developer.nvidia.com/blog/cooperative-groups/)**, a CUDA feature from 2017 that allows for flexible and dynamic grouping of threads, and expressed surprise that it's not covered more extensively in the CUDA book used in CUDA MODE. There's a discussion about its effectiveness and recent use headed by the same member.

- **Transitioning from C to C++ in CUDA Considered**: The discussion pivoted to whether moving from **C to C++** would offer concrete benefits in CUDA development, touching on nvcc's use of a C++ compiler, the convenience of C++ features like `constexpr` and templates, and the possibility of using **Cutlass library** with templates. There is consensus regarding the advantages of C++ in handling dynamic shared memory sizes and a variety of data types.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://developer.nvidia.com/blog/cooperative-groups/">Cooperative Groups: Flexible CUDA Thread Programming | NVIDIA Technical Blog</a>: In efficient parallel algorithms, threads cooperate and share data to perform collective computations. To share data, the threads must synchronize. The granularity of sharing varies from algorithm to&...</li><li><a href="https://github.com/karpathy/llm.c">GitHub - karpathy/llm.c: LLM training in simple, raw C/CUDA</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://godbolt.org/z/6zq6Kfecc">Compiler Explorer - CUDA C++ (NVCC 12.3.1)</a>: __global__ void crossentropy_forward_kernel1(float* losses,                             float* probs, int* targets,                             int B, int T, int V) {     int i = blockIdx.x * blockDim...
</li>
</ul>

</div>
  

---



**Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1227538082124009512)** (1 messages): 

- **Interpretability Tools Apply to Modern RNNs**: New research shows that popular interpretability tools for transformers can indeed be adapted for use with modern RNNs like **Mamba** and **RWKV**. Techniques such as vector arithmetic, the tuned lens, and eliciting latent knowledge from fine-tuned RNNs to produce incorrect answers were proven effective. [Read the Paper](https://arxiv.org/abs/2404.05971) | [Browse the Code](https://github.com/EleutherAI/rnngineering)

- **RNNs' Renaissance in Language Modeling**: **Mamba** and **RWKV**, the new kids on the block in RNN language models, might follow the path paved by transformers thanks to their comparable performance and successful application of interpretability tools. [Engage with the Twitter Thread](https://x.com/norabelrose/status/1777975663590531533)

- **Shoutout for Collaborative Efforts**: Active collaboration in the <#1186803874846212257> channel was acknowledged and appreciated in advancing the state of RNN language models. Thanks were extended to the contributors for their involvement in the study.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.05971">Does Transformer Interpretability Transfer to RNNs?</a>: Recent advances in recurrent neural network architectures, such as Mamba and RWKV, have enabled RNNs to match or exceed the performance of equal-size transformers in terms of language modeling perplex...</li><li><a href="https://github.com/EleutherAI/rnngineering">GitHub - EleutherAI/rnngineering: Engineering the state of RNN language models (Mamba, RWKV, etc.)</a>: Engineering the state of RNN language models (Mamba, RWKV, etc.) - EleutherAI/rnngineering</li><li><a href="https://x.com/norabelrose/status/1777975663590531533">Tweet from Nora Belrose (@norabelrose)</a>: RNN language models are making a comeback recently, with new architectures like Mamba and RWKV.  But do interpretability tools designed for transformers transfer to the new RNNs? We tested 3 popular i...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1227583941280796682)** (83 messages🔥🔥): 

- **Mixtral 8x22B Impresses with AGIEval**: The new [Mixtral 8x22B model](https://x.com/jphme/status/1778030213881909451) showcases impressive first AGIEval results, outperforming other open-source (base) models. There's enthusiasm in the community for its release and high performance.
- **Forecasting AI's Future**: A member highlighted a [timeline of AI predictions](https://theaidigest.org/timeline) based on median forecasts from Metaculus and Manifold, offering a rough sense of AI advancements, potential harms, and societal responses, with some questioning the projected timeline for models like llama3 becoming open-source.
- **Concerns Over Election Security with AI**: Members express worries about the potential negative impact of AI technologies like deepfakes on upcoming elections, with some contemplating avoiding social media during election times to focus on personal wellness.
- **Technical Discussion on Extending Encoder Models**: The channel witnessed a technical discussion about the challenges of extending context window size in encoder models such as BERT. [FlashAttention's incorporation](https://mosaicbert.github.io/) into encoder models and its absence from popular libraries like HuggingFace sparked curiosity.
- **Downloading The Pile for Research**: There was a conversation regarding the size discrepancies when downloading The Pile dataset, with users clarifying that the 886 GB refers to the uncompressed size, while the compressed files may be significantly smaller. Members affirmed that their usage of The Pile is strictly for academic and research-oriented purposes.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/jphme/status/1778030213881909451">Tweet from Jan P. Harries (@jphme)</a>: @MistralAI first AGIEval results look great 👇 - thanks for releasing this beast, guys! 👏 https://x.com/jphme/status/1778028110954295486  ↘️ Quoting Jan P. Harries (@jphme)   First AGIEval results fo...</li><li><a href="https://mosaicbert.github.io/">MosaicBERT: A Bidirectional Encoder Optimized for Fast Pretraining</a>: no description found</li><li><a href="https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2">Learning Agency Lab - Automated Essay Scoring 2.0 | Kaggle</a>: no description found</li><li><a href="https://theaidigest.org/timeline">Timeline of AI forecasts - AI Digest</a>: What to expect in AI capabilities, potential harms, and society&#x27;s response</li><li><a href="https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1/discussions/4">mistral-community/Mixtral-8x22B-v0.1 · Benchmarks are here!</a>: no description found</li><li><a href="https://github.com/huggingface/transformers/issues/26350">Community contribution: Adding Flash Attention 2 support for more architectures · Issue #26350 · huggingface/transformers</a>: Feature request Flash Attention 2 is a library that provides attention operation kernels for faster and more memory efficient inference and training: https://github.com/Dao-AILab/flash-attention Le...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1227519788532891729)** (132 messages🔥🔥): 

- **Adversarial Image Examples Not Just Noise**: Adversarial image examples can go beyond the appearance of unstructured noise; they might involve actual deformities like altering a dog's nose [Adversarial Image Research](https://arxiv.org/abs/1907.07174) shows machine learning models with shared weaknesses in real-world, unmodified examples.
- **Subset Fine-tuning Explored**: A new method called **Subset fine-tuning** (SubTuning) demonstrates that only tuning a subset of neural network layers can achieve performance comparable to tuning all layers, with potential benefits in multi-task learning and reduced computational resources needed [SubTuning Research](https://arxiv.org/abs/2302.06354).
- **Uncovering Model Training Budget Constraints**: A member emphasized how finetuning budget limitations are a valid reason to not train models like BERT from scratch; another member highlighted the technological relevance regardless of budget [No Paper Reference Provided].
- **Discovery of Mistral's Bidirectional Attention**: Recent findings suggest that Mistral models might leverage a form of bidirectional attention, leading to high cosine similarity across all layers and positions for the Mistral-7B with bidirectional attention enabled [No Paper Reference Provided].
- **Training Hybrid RWKV5 Transformer Model Queries**: Interest in if anyone has trained a hybrid RWKV5 transformer model, discussions pointed to the RWKV server for more details, and no public training of such models was noted [No Paper Reference Provided].

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.07143">Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention</a>: This work introduces an efficient method to scale Transformer-based Large Language Models (LLMs) to infinitely long inputs with bounded memory and computation. A key component in our proposed approach...</li><li><a href="https://arxiv.org/abs/2307.13912">Embedding Democratic Values into Social Media AIs via Societal Objective Functions</a>: Can we design artificial intelligence (AI) systems that rank our social media feeds to consider democratic values such as mitigating partisan animosity as part of their objective functions? We introdu...</li><li><a href="https://arxiv.org/abs/2404.06654">RULER: What&#39;s the Real Context Size of Your Long-Context Language Models?</a>: The needle-in-a-haystack (NIAH) test, which examines the ability to retrieve a piece of information (the &#34;needle&#34;) from long distractor texts (the &#34;haystack&#34;), has been widely adopted ...</li><li><a href="https://arxiv.org/abs/2310.17041">On Surgical Fine-tuning for Language Encoders</a>: Fine-tuning all the layers of a pre-trained neural language encoder (either using all the parameters or using parameter-efficient methods) is often the de-facto way of adapting it to a new task. We sh...</li><li><a href="https://arxiv.org/abs/2302.06354">Less is More: Selective Layer Finetuning with SubTuning</a>: Finetuning a pretrained model has become a standard approach for training neural networks on novel tasks, resulting in fast convergence and improved performance. In this work, we study an alternative ...</li><li><a href="https://tenor.com/view/avocado-bacon-salad-lunch-salad-gif-12338945">Avocado Bacon Salad Lunch GIF - Avocado Bacon Salad Lunch Salad - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/_akhaliq/status/1775740222120087847?t=55VlAx9tjP9PUgvRcnIfMQ&s=33">Tweet from AK (@_akhaliq)</a>: Google presents Mixture-of-Depths  Dynamically allocating compute in transformer-based language models  Transformer-based language models spread FLOPs uniformly across input sequences. In this work we...</li><li><a href="https://fxtwitter.com/JiaChenyan/status/1732898372359799159">Tweet from Chenyan Jia (@JiaChenyan)</a>: Can we design AI systems to consider democratic values as their objective functions? Our new #CSCW24 paper w/ @michelle123lam, Minh Chau Mai, @jeffhancock, @msbernst introduces a method for translatin...</li><li><a href="https://arxiv.org/abs/1907.07174">Natural Adversarial Examples</a>: We introduce two challenging datasets that reliably cause machine learning model performance to substantially degrade. The datasets are collected with a simple adversarial filtration technique to crea...</li><li><a href="https://arxiv.org/abs/2110.03111">Cut the CARP: Fishing for zero-shot story evaluation</a>: Recent advances in large-scale language models (Raffel et al., 2019; Brown et al., 2020) have brought significant qualitative and quantitative improvements in machine-driven text generation. Despite t...</li><li><a href="https://arxiv.org/abs/2210.07792">Robust Preference Learning for Storytelling via Contrastive Reinforcement Learning</a>: Controlled automated story generation seeks to generate natural language stories satisfying constraints from natural language critiques or preferences. Existing methods to control for story preference...</li><li><a href="https://github.com/hsiehjackson/RULER">GitHub - hsiehjackson/RULER: This repo contains the source code for RULER: What’s the Real Context Size of Your Long-Context Language Models?</a>: This repo contains the source code for RULER: What’s the Real Context Size of Your Long-Context Language Models? - hsiehjackson/RULER
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1227556417502969917)** (4 messages): 

- **Scaling Laws Meet Knowledge Storage**: A new paper released on [arXiv](https://arxiv.org/abs/2404.05405) argues that language models can store a maximum of **2 bits of knowledge per parameter**. The paper also explores how factors like training duration, model architecture, quantization, use of sparsity constraints such as MoE, and data signal-to-noise ratio impact a model's knowledge capacity.
  
- **Parsing Complexity in New Scaling Law Research**: Discussion in the Eleuther community indicates that the aforementioned paper on scaling laws and knowledge bits is challenging to interpret. Community members are considering creating a discussion space to unpack the paper's findings.

- **Seeking Benchmarks for OpenAI's Latest**: A member inquired about benchmarks for **OpenAI's new gpt-4-turbo** version and is looking for where these performance results may be published.

**Link mentioned**: <a href="https://arxiv.org/abs/2404.05405">Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws</a>: Scaling laws describe the relationship between the size of language models and their capabilities. Unlike prior studies that evaluate a model&#39;s capability via loss or benchmarks, we estimate the n...

  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1228056988588572672)** (2 messages): 

- **Brief Interaction with No Substance**: Chat in the channel included a member expressing dismay at an unspecified requirement, followed by agreement from another member. No context or topic of discussion was provided.
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1227633571619934228)** (2 messages): 

- **Progress on Chat Templating PRs**: Two pull requests (PRs) are currently being discussed—one by Hailey on adding [chat templating for HF models](https://github.com/EleutherAI/lm-evaluation-harness/pull/1287#issuecomment-1967469808), and another unspecified [PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/1578). Reading these PRs is recommended to get up to speed with the project’s progress.
- **Opportunity for Batchwise `apply_chat_template` Contributions**: A member noted that batchwise operations are not supported for `apply_chat_template` in the **transformers** library. Contributing this feature could greatly benefit this and other projects.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1287#issuecomment-1967469808)">[WIP] Add chat templating for HF models by haileyschoelkopf · Pull Request #1287 · EleutherAI/lm-evaluation-harness</a>: This is a WIP PR , carrying on the draft @daniel-furman in #1209 started of adding the specified oft-requested chat templating feature. Current TODOs are:   Check performance using e.g. OpenHermes ...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1578).">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.
</li>
</ul>

</div>
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1227611511426646117)** (103 messages🔥🔥): 

- **Seeking Projects Beyond Low-Level Code**: A developer with 20 years of experience expressed interest in contributing to **Modular projects** but wishes to avoid low-level C-style coding. Discussions pointed them toward **web development project Lightbug** and machine learning project **Basalt**, with a link provided to the Basalt repository on GitHub.

- **Ellipsis Debate in Trait Methods**: There has been a significant discussion about the use of ellipsis (`...`) in unimplemented trait methods within **Mojo**. A GitHub discussion was proposed to deprecate the use of ellipsis, with members debating its Pythonic nature and potential replacements, including `not_implemented` as an alias for `os.abort`.

- **Community Engagement in the Open Source Effort**: A member mentioned that **Mojo open-sourced its standard library**, which sparked a sarcastic rebuttal from another, yet the open contribution to the standard library by the community is evident from merged PRs in nightly builds.

- **Invitation for Collaboration with BackdropBuild**: A representative from **BackdropBuild** reached out to discuss future collaboration with Modular, aiming to support builders in AI, crypto, gaming, and dev tooling. The organization runs large cohort programs and works with well-known companies to foster development on various tech platforms.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://realpython.com/python-ellipsis/">When Do You Use an Ellipsis in Python? – Real Python</a>: You may have seen three dots in Python scripts. Although this syntax may look odd,  using an ellipsis is valid Python code. In this tutorial, you&#x27;ll learn when Python&#x27;s Ellipsis constant can...</li><li><a href="https://peps.python.org/pep-0544/">PEP 544 – Protocols: Structural subtyping (static duck typing) | peps.python.org</a>: no description found</li><li><a href="https://docs.rs/serde/latest/src/serde/de/mod.rs.html#908-1233">mod.rs - source</a>: no description found</li><li><a href="https://github.com/modularml/mojo/discussions/2259">[Proposal]: Deprecate Triple Dots (...) for Unimplemented Methods · modularml/mojo · Discussion #2259</a>: Motivation: Mojo aspires to be a seamless successor to Python++, adhering closely to Pythonic principles and fostering a positive experience for the Python community. The current practice of using ...</li><li><a href="https://github.com/basalt-org/basalt">GitHub - basalt-org/basalt: A Machine Learning framework from scratch in Pure Mojo 🔥</a>: A Machine Learning framework from scratch in Pure Mojo 🔥 - basalt-org/basalt</li><li><a href="https://github.com/mojicians/awesome-mojo">GitHub - mojicians/awesome-mojo: A curated list of awesome Mojo 🔥 frameworks, libraries, software and resources</a>: A curated list of awesome Mojo 🔥 frameworks, libraries, software and resources - mojicians/awesome-mojo</li><li><a href="https://peps.python.org/">PEP 0 – Index of Python Enhancement Proposals (PEPs) | peps.python.org</a>: no description found</li><li><a href="https://backdropbuild.com/">Backdrop Build</a>: Together we build - Bring that crazy idea to life alongside hundreds of other amazing builders in just 4 weeks.
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1227684149150613547)** (2 messages): 

- **Tweet Announcement from Modular**: A tweet from the official Modular Twitter account was shared in the Discord channel. It can be viewed directly on Twitter: [Modular Tweet](https://twitter.com/Modular/status/1778118673976402286).
- **Follow-up Tweet Shared**: Another tweet from Modular surfaced in the discussion. To see the content of this tweet, visit: [Modular's Latest Tweet](https://twitter.com/Modular/status/1778482233957101869).
  

---


**Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1228017366504312963)** (1 messages): 

- **Row-Major versus Column-Major in Memory**: A [blog post](https://www.modular.com/blog/row-major-vs-column-major-matrices-a-performance-analysis-in-mojo-and-numpy) examines the storage differences for matrices in memory, detailing how row-major order favors row-vectors in contiguous memory and column-major order favors column-vectors. The analysis aims to clarify the implications of these storage strategies for performance in Mojo🔥 and NumPy.
- **Performance Showdown Between Storage Orders**: The post intends to address why some programming languages and libraries opt for column-major order while others prefer row-major, and what performance consequences these choices entail. It emphasizes the speed advantage of reading contiguous memory locations and suggests that the chosen storage order could significantly impact performance.

**Link mentioned**: <a href="https://www.modular.com/blog/row-major-vs-column-major-matrices-a-performance-analysis-in-mojo-and-numpy">Modular: Row-major vs. column-major matrices: a performance analysis in Mojo and NumPy</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: Row-major vs. column-major matrices: a performance analysis in Mojo and NumPy

  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1227519673394790400)** (58 messages🔥🔥): 

- **Mojo Improvements on the Horizon**: A [roadmap document](https://docs.modular.com/mojo/roadmap#lifetime-tracking-inside-collections) provides insights into **Mojo's future plans** regarding core programming features and acknowledges major components to be added, with a focus on building Mojo's core language features first for long-term sustainability.
- **UI Library Development for Mojo in the Mac Ecosystem**: Discussion centered on a [new cross-platform UI library for Mojo](https://github.com/Moosems/Mojo-UI), tentative name "Mojective-C," and possible approaches for integrating Objective-C or AppKit with Mojo via C or C++ bindings were suggested.
- **Mojo’s Handling of Context Managers and `with` Blocks**: There's an ongoing discourse about the design of GUI frameworks in Mojo and possible alternatives to using `with` blocks, which are often viewed negatively due to their constraints on developers' control over widgets.
- **Segfaults in Mojo Posing Questions**: A segfault issue on GitHub was prematurely closed and then reopened, leading to clarifications on the policy that issues are marked closed internally once they’re fixed, but may not yet be reflected in the nightly/stable releases.
- **Bitwise Operations and Converting C Code to Mojo**: Exchanges regarding the translation of bitwise operations from C to Mojo resulted in shared code snippets. Some users offered corrections and advice on datatype usage in the context of implementing random number generation in Mojo.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/jeff-channing-tatum-22jump-street-disguise-gif-8025876">My Name Is Jeff Jeff GIF - Jeff Channing Tatum 22Jump Street - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://en.wikipedia.org/wiki/Xorshift#xorshift.2A">Xorshift - Wikipedia</a>: no description found</li><li><a href="https://docs.modular.com/mojo/roadmap#lifetime-tracking-inside-collections">Mojo🔥 roadmap &amp; sharp edges | Modular Docs</a>: A summary of our Mojo plans, including upcoming features and things we need to fix.</li><li><a href="https://github.com/Moosems/Mojo-UI">GitHub - Moosems/Mojo-UI: A cross-platform GUI library for Mojo</a>: A cross-platform GUI library for Mojo. Contribute to Moosems/Mojo-UI development by creating an account on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/issues/28">Why not Mojo? · Issue #28 · karpathy/llm.c</a>: Serious question here. If you&#39;re going down to the metal, Mojo provides a potentially large speedup and the language would significant benefit from this work. Either way - love the work. Thank you...</li><li><a href="https://github.com/modularml/mojo/issues/2208#issuecomment-2046614359">[BUG][Regression] Struct definition with recursive reference (no longer gives an error message and) crashes · Issue #2208 · modularml/mojo</a>: Bug description I believe the issue #74 has resurfaced. The following code will segfault: #crash.mojo struct Node: var rec: Node fn __init__(inout self): pass But I expected the error message: $🍔 m.....</li><li><a href="https://github.com/modularml/mojo/issues/2208#issuecomm">[BUG][Regression] Struct definition with recursive reference (no longer gives an error message and) crashes · Issue #2208 · modularml/mojo</a>: Bug description I believe the issue #74 has resurfaced. The following code will segfault: #crash.mojo struct Node: var rec: Node fn __init__(inout self): pass But I expected the error message: $🍔 m.....
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1227574677937983580)** (5 messages): 

- **Mojo Gets Iterative**: A code snippet for iterating over string characters was shared, potentially useful for others looking for a similar solution. The actual iterator code is available on the [Discord link provided](https://discord.com/channels/1087530497313357884/1227272073840431116/1227495619501752330).

- **Keyboard Events Come to mojo-ui-html**: An update to `mojo-ui-html` introduces keyboard events, window minimization capabilities, and per-element CSS styling improvements, favorable for game and custom widget developers, as well as those who use `neovim`. These new methods aim to enhance interactive experiences and can be seen in action with their [demo on GitHub](https://github.com/rd4com/mojo-ui-html/blob/main/demo_keyboard_and_css.mojo).

- **Lightbug Framework Celebrates Contributions**: The `Lightbug` HTTP framework has seen several community contributions including the addition of getting the remote address/peer name, a more efficient string-building method, a Mojo-based client implementation, and performance profiling which reveals Lightbug's superior request handling capabilities when compared to Python Flask. New developments also include a high-level API framework called `lightbug_api` that draws inspiration from Django, all of which can be explored on [GitHub](https://github.com/saviorand/lightbug_http).

- **Elevating Terminal Text Rendering with Mojo**: A preview of text rendered in a terminal using Mojo was showcased, demonstrating the tooling that can be built within this ecosystem, with inspiration drawn from Go packages such as `lipgloss`. The code behind this sleek terminal UI display is available for inspection on its [GitHub repository](https://github.com/thatstoasty/mog/blob/main/examples/readme/layout.mojo).

- **Basalt Illustrates Mojo's Visual Appeal**: Amidst the updates, a community member praised the use of Basalt (a preset for terminal styling) in the recently shared Mojo-rendered text example, highlighting the visual prowess of Mojo in enhancing terminal applications.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/thatstoasty/mog/blob/main/examples/readme/layout.mojo">mog/examples/readme/layout.mojo at main · thatstoasty/mog</a>: Contribute to thatstoasty/mog development by creating an account on GitHub.</li><li><a href="https://github.com/rd4com/mojo-ui-html/blob/main/demo_keyboard_and_css.mojo">mojo-ui-html/demo_keyboard_and_css.mojo at main · rd4com/mojo-ui-html</a>: Immediate mode GUI, HTML, CSS, Work in progress, Mojo language - rd4com/mojo-ui-html</li><li><a href="https://github.com/saviorand/lightbug_http/issues/6).">Issues · saviorand/lightbug_http</a>: Simple and fast HTTP framework for Mojo! 🔥. Contribute to saviorand/lightbug_http development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[📰︱newsletter](https://discord.com/channels/1087530497313357884/1157711273111339088/)** (1 messages): 

Zapier: Modverse Weekly - Issue 29
https://www.modular.com/newsletters/modverse-weekly-29
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1227523242445635664)** (131 messages🔥🔥): 

- **Quandaries of Quantization**: Members discussed the quantizing potential of models like **Mistral** to fit on a single 24GB card with 16k context; however, the performance of quantized **Mixtral 8x7b** was supported by some as being notably good.
- **Info Seekers on MLLM**: Queries concerning multimodal large models (MLLMs) surfaced with recommendations like **LLaVA** and **Qwen VLM**, albeit admittance of their restricting licenses and the limited guidance available for fine-tuning LMMs with axolotl.
- **Pondering on Inference Servers**: A debate unfolded about building an inference server with **Nvidia 4090s**, in the wake of dealing with constraints such as a lack of NVLink and PCIe 5. **Nvidia 3090s** were raised as a potentially more suitable alternative regarding inter-card bandwidth.
- **Hackathon Heads Up**: There was a plug for the **Samsung Next 2024 Generative AI Hackathon** happening on May 11th in New York, with focus areas including Health & Wellness and Mediatech.
- **Contributing and Sharing Insights on Axolotl**: A member expressed a desire to contribute to **Axolotl** with the response emphasizing the value of reproducing existing issues, helping with documentation, and the fact that programming experience could be less necessary than time. Another key discussion point involved the differentiation between supervised fine-tuning (SFT) and **dynamic programming optimization (DPO)**, highlighting that **DPO** can uniquely steer generated responses favorably over **SFT** in some contexts.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.05405">Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws</a>: Scaling laws describe the relationship between the size of language models and their capabilities. Unlike prior studies that evaluate a model&#39;s capability via loss or benchmarks, we estimate the n...</li><li><a href="https://lu.ma/nextgenainyc">Samsung Next 2024 Generative AI Hackathon · Luma</a>: 🚀 What&#x27;s Happening Apply to join the Samsung Next 2024 Generative AI Hackathon! We&#x27;ll explore two tracks: Health &amp; Wellness: Harness the power of AI in improving healthcare outcomes,...</li><li><a href="https://huggingface.co/MaziyarPanahi/Mixtral-8x22B-v0.1-GGUF/tree/main">MaziyarPanahi/Mixtral-8x22B-v0.1-GGUF at main</a>: no description found
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1227796708411441224)** (5 messages): 

- **Axolotl Documentation Unveiled**: A member shared their long-term project: the [Axolotl documentation](https://axolotl.continuumlabs.pro), inviting the community for feedback and acknowledging that there are still some gaps to be filled.
- **Feedback Loop Encouraged**: While appreciating the effort, a member suggested keeping document-related discussions to a single channel and pointed out that there are both valuable tips and mistakes in the current document draft.
- **Contribution Acknowledgment**: Another participant expressed gratitude for the initiative taken to compile the Axolotl documentation.

**Link mentioned**: <a href="https://axolotl.continuumlabs.pro">Introduction | Continuum Training Platform | Axolotl Training Platform</a>: no description found

  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1227675113735061636)** (3 messages): 

- **Empty Queue Handled**: To avoid errors from an empty queue, a member suggested checking the queue's state before iterating. The provided snippet corrects the iteration by wrapping the `for` loop with `if not streamer.empty()`.

- **Refactor Stop Token Checking**: A simple refactoring tip was shared to optimize the stop token function. The code `return input_ids[0][-1] in stop_ids` efficiently replaces a loop for checking stop conditions. 

- **Question on GPU Utilization for Model Merging**: A member inquired about the possibility of leveraging GPU resources while merging models to enhance performance.
  

---


**OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1227983231698337852)** (2 messages): 

- **Seeking Logic Reasoning Dataset**: A member inquired about a dataset that deals with **reasoning using propositional and predicate logic** over natural text but did not mention any they've found.
- **In Search of Colossal Training Data**: Another member requested recommendations for a dataset of approximately **200 billion tokens** suitable for experimenting with a new architecture; no datasets were suggested in the subsequent discussion.
  

---


**OpenAccess AI Collective (axolotl) ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/1227796935021432892)** (1 messages): 

- **Axolotl Documentation Released for Feedback**: Long-awaited [Axolotl documentation](https://axolotl.continuumlabs.pro) has been shared for community feedback. It's noted that the documentation might still have gaps, and the author encourages feedback for further development.

**Link mentioned**: <a href="https://axolotl.continuumlabs.pro">Introduction | Continuum Training Platform | Axolotl Training Platform</a>: no description found

  

---


**OpenAccess AI Collective (axolotl) ▷ #[minotaur](https://discord.com/channels/1104757954588196865/1116465236715786310/1227797012003557417)** (1 messages): 

- **Axolotl Documentation Released for Feedback**: Long-awaited **Axolotl documentation** has been shared with the community and is open for feedback. The documentation, which is a work in progress with some gaps, can be accessed at [Axolotl Documentation](https://axolotl.continuumlabs.pro) and feedback is encouraged to refine the material.

**Link mentioned**: <a href="https://axolotl.continuumlabs.pro">Introduction | Continuum Training Platform | Axolotl Training Platform</a>: no description found

  

---


**OpenAccess AI Collective (axolotl) ▷ #[bots](https://discord.com/channels/1104757954588196865/1117282691121954836/1227796957389651988)** (1 messages): 

- **Axolotl Documentation Released for Feedback**: New [Axolotl documentation](https://axolotl.continuumlabs.pro) has been shared with the community. The author has secured all necessary approvals and is open to feedback for further development.

**Link mentioned**: <a href="https://axolotl.continuumlabs.pro">Introduction | Continuum Training Platform | Axolotl Training Platform</a>: no description found

  

---


**OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1227953974930440212)** (3 messages): 

- **First Steps with Axolotl**: A new blog post detailing experiences with **fine-tuning** smaller encoder style LLMs for classification tasks and initial explorations into decoder style LLMs for text generation was shared. The resource is a helpful guide for those new to LLMs, accompanied by a [blog post](https://drchrislevy.github.io/posts/intro_fine_tune/intro_fine_tune.html) including references to models like GPT and Mistral.

- **A Community Member's Endorsement**: The aforementioned **Axolotl introductory guide** is recommended as a good starting point for those looking to understand the basics of using Axolotl.

- **Debugging Tips Shared**: For those working with data preprocessing, the use of a `--debug` flag during preprocess can help verify the correctness of the data record.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/cleavey1985/status/1778393547571384410">Tweet from Chris Levy (@cleavey1985)</a>: Finished a blog post on using @axolotl_ai for the first time. https://drchrislevy.github.io/posts/intro_fine_tune/intro_fine_tune.html Thanks @jeremyphoward  for recommending the tool, @HamelHusain  f...</li><li><a href="https://drchrislevy.github.io/posts/intro_fine_tune/intro_fine_tune.html">Chris Levy - Getting Started with Axolotl for Fine-Tuning LLMs</a>: no description found
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1227772768972701726)** (5 messages): 

- **QLoRA Training Mystery on Multi-GPU Setup**: A member queried why a **QLoRA training process** that consumes 98% VRAM on a single 4090 GPU could fail when scaled to 4x4090 GPUs despite using *DeepSpeed Zero Stage 3*. The possibility of additional memory overhead, the change in memory allocation patterns, and the added communication overhead might contribute to this failure.
- **Optimization Strategies for ZeRO-3**: To resolve the multi-GPU training issue, recommendations were made to experiment with **hyperparameter adjustments**, optimize **DeepSpeed configuration**, use *nvidia-smi* for monitoring, and explore **CPU offloading**. Adjusting batch sizes and gradient accumulation steps could help to balance the memory load across GPUs.
- **DeepSpeed Configuration Tuning**: It was suggested that carefully reviewing and adjusting the **DeepSpeed configuration** can significantly impact memory usage and training performance. The configuration can include enabling CPU offloading for optimizer and parameter states to alleviate GPU memory usage.

**Link mentioned**: <a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=eddbbf5d-0725-40a1-9baf-c02e328d4a61)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.

  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1227661649436217364)** (10 messages🔥): 

- **Mixtral Lands on OpenRouter**: The new base model, [Mixtral 8x22B](https://openrouter.ai/models/mistralai/mixtral-8x22b), is available on OpenRouter, boasting solid performance with instruct templates despite not being instruct tuned.

- **Gemma Gets an Upgrade**: OpenRouter updates its offering with [Google: Gemma 7B](https://openrouter.ai/models/google/gemma-7b-it) replaced by the newer [google/gemma-1.1-7b-it](https://huggingface.co/google/gemma-1.1-7b-it).

- **OpenRouter Cuts Prices Across the Board**: The platform announces reduced pricing for models including [LZLV 70B](https://openrouter.ai/models/lizpreciatior/lzlv-70b-fp16-hf), [Databricks: DBRX 132B Instruct](https://openrouter.ai/models/databricks/dbrx-instruct), and [Nous: Hermes 2 Mixtral 8x7B DPO](https://openrouter.ai/models/nousresearch/nous-hermes-2-mixtral-8x7b-dpo).

- **Limited Time Offer: Free Mixtral 8x22B**: Users are encouraged to try out [Mixtral 8x22B](https://openrouter.ai/models/mistralai/mixtral-8x22b) for free for a limited period.

- **User Feedback Shapes Gemma's Availability**: Following a user query on the availability of Gemma 2B, OpenRouter responds that there hasn't been much demand, noting the 7B version is already free for quick tasks, and suggesting the 2B might be more suited for running locally.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/mistralai/mixtral-8x22b>)">Mixtral 8x22B by mistralai | OpenRouter</a>: Mixtral 8x22B is a large-scale language model from Mistral AI. It consists of 8 experts, each 22 billion parameters, with each token using 2 experts at a time.  It was released via [X](https://twitter...</li><li><a href="https://openrouter.ai/models/mistralai/mixtral-8x22b)">Mixtral 8x22B by mistralai | OpenRouter</a>: Mixtral 8x22B is a large-scale language model from Mistral AI. It consists of 8 experts, each 22 billion parameters, with each token using 2 experts at a time.  It was released via [X](https://twitter...</li><li><a href="https://openrouter.ai/models/mistralai/mixtral-8x22b:free>).">Mixtral 8x22B by mistralai | OpenRouter</a>: Mixtral 8x22B is a large-scale language model from Mistral AI. It consists of 8 experts, each 22 billion parameters, with each token using 2 experts at a time.  It was released via [X](https://twitter...</li><li><a href="https://openrouter.ai/models/google/gemma-7b-it>)">Gemma 7B by google | OpenRouter</a>: Gemma by Google is an advanced, open-source language model family, leveraging the latest in decoder-only, text-to-text technology. It offers English language capabilities across text generation tasks ...</li><li><a href="https://openrouter.ai/models/lizpreciatior/lzlv-70b-fp16-hf>)">lzlv 70B by lizpreciatior | OpenRouter</a>: A Mythomax/MLewd_13B-style merge of selected 70B models. A multi-model merge of several LLaMA2 70B finetunes for roleplaying and creative work. The goal was to create a model that combines creativity ...</li><li><a href="https://openrouter.ai/models/databricks/dbrx-instruct>)">DBRX 132B Instruct by databricks | OpenRouter</a>: DBRX is a new open source large language model developed by Databricks. At 132B, it outperforms existing open source LLMs like Llama 2 70B and Mixtral-8x7B on standard industry benchmarks for language...</li><li><a href="https://openrouter.ai/models/nousresearch/nous-hermes-2-mixtral-8x7b-dpo>)">Hermes 2 Mixtral 8x7B DPO by nousresearch | OpenRouter</a>: Nous Hermes 2 Mixtral 8x7B DPO is the new flagship Nous Research model trained over the [Mixtral 8x7B MoE LLM](/models/mistralai/mixtral-8x7b).  The model was trained on over 1,000,000 entries of prim...
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1227538189950910546)** (136 messages🔥🔥): 

- **Model Rate Limit Clarifications**: A member clarified that "heavily rate limited" on OpenRouter typically means a model has rate limits similar to free ones: 10 requests per minute.

- **Gemini 1.5: Paid, not Free on OR**: It was pointed out that **Gemini 1.5** is paid on OpenRouter and does not have a free tier at the moment.

- **Gemma Rate Limit Concerns**: Users discussed issues around Gemma models not being free when expected, issues with the "Updated" tag not reflecting the latest changes, and confusion regarding rate limits and token counting, which led to a fix being deployed.

- **Gemini Token Pricing Explained**: Clarifications came through on the pricing for Gemini models, stating that OR counts Gemini tokens as individual characters, which is reflected in the higher "context" cost. This factor is estimated for use during sampling but billing counts each character as a token.

- **Mixtral 8x22b Discussions**: Users shared their experiences with **Mixtral 8x22b**, noting its good reasoning capabilities and considering its cost-effectiveness compared to GPT-4. There was a consensus that despite not being as eloquent as GPT-4, it provided coherent and surprisingly good outputs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/cohere/command-r-plus">Command R+ by cohere | OpenRouter</a>: Command R+ is a new, 104B-parameter LLM from Cohere. It&#x27;s useful for roleplay, general consumer usecases, and Retrieval Augmented Generation (RAG).  It offers multilingual support for ten key lan...</li><li><a href="https://openrouter.ai/models/google/gemma-7b-it:free">Gemma 7B by google | OpenRouter</a>: Gemma by Google is an advanced, open-source language model family, leveraging the latest in decoder-only, text-to-text technology. It offers English language capabilities across text generation tasks ...</li><li><a href="https://openrouter.ai/playground?models=openai/gpt-4-turbo,mistralai/mistral-large">OpenRouter</a>: A router for LLMs and other AI models</li><li><a href="https://docs.librechat.ai/install/index.html">Installation and Configuration</a>: 💻 In-depth guides about installation and configuration</li><li><a href="https://github.com/danny-avila/LibreChat">GitHub - danny-avila/LibreChat: Enhanced ChatGPT Clone: Features OpenAI, Assistants API, Azure, Groq, GPT-4 Vision, Mistral, Bing, Anthropic, OpenRouter, Google Gemini, AI model switching, message search, langchain, DALL-E-3, ChatGPT Plugins, OpenAI Functions, Secure Multi-User System, Presets, completely open-source for self-hosting. More features in development</a>: Enhanced ChatGPT Clone: Features OpenAI, Assistants API, Azure, Groq, GPT-4 Vision, Mistral, Bing, Anthropic, OpenRouter, Google Gemini, AI model switching, message search, langchain, DALL-E-3, Cha...</li><li><a href="https://discord.gg/uDyZ5Tzhct">Join the LibreChat Discord Server!</a>: Community for LibreChat, an Open-source, versatile AI Chat Web UI, with seamless self-hosting and active developments. | 3349 members
</li>
</ul>

</div>
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1227543783671861288)** (97 messages🔥🔥): 

- **Excitement Over Command r+**: Members have started experimenting with a model called **Command r+** which is noted for its impressive **instruction-following capabilities**. Users report that it outperforms other models, like **GPT-3.5** and **Claude 3**, in certain benchmarks and role-playing scenarios, even suggesting it's near the level of **GPT-4** in some aspects.

- **Technical Support on Open-Interpreter Setup**: Technical issues arose with installing **open-interpreter**, where it crashed after attempting to take screenshots and had dependency conflicts. A fix was suggested using a **git+https** installation command, and setting **OPENAI_API_KEY** as an environment variable for API access.

- **Anticipation for Mixtral and OI Integration**: Discussions indicate a member's hope for **Mixtral 8x22b** to work effectively with **OI**, mentioning that the 8x7b version wasn't quite meeting expectations when used with OI.

- **Open AI Prepaid Credits Promotion**: An update was shared about **OpenAI** transitioning to prepaid credits and discontinuing monthly billing. Members were informed about a promotional credit offer available until April 24, 2024.

- **Invitation to and Reflections on an Open-Interpreter Codebase Session**: A member shared information about a recent session on using **Open Interpreter as a library** in Python projects, including a link to a GitHub repository with starter templates. Another session is scheduled, and input is requested on which parts of the codebase the community would like to explore.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lu.ma/novus28">Novus #28 · Luma</a>: Novus is a community for startup founders, builders, and creatives to come together, co-work and demo. No bullshit. No sales. Just building. Agenda 12:00 PM - 12:15 PM - Updates &amp;...</li><li><a href="https://discord.gg/open-interpreter-1146610656779440188?event=1228084898993143920">Join the Open Interpreter Discord Server!</a>: A new way to use computers | 8227 members</li><li><a href="https://www.youtube.com/watch?v=rOW8OK7qXcM">oi house party 5 apr</a>: no description found</li><li><a href="https://github.com/MikeBirdTech/open-interpreter-python-templates">GitHub - MikeBirdTech/open-interpreter-python-templates</a>: Contribute to MikeBirdTech/open-interpreter-python-templates development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1227599141979553836)** (30 messages🔥): 

- **Nagging ngrok Nuisance**: A member encountered an issue where `ngrok` failed to bind to their specified domain, issuing an error with a different domain name. Despite resetting the token twice, the problem persisted, hinting at a potential configuration anomaly within the **ngrok setup**.
  
- **Order Status Updates via Email**: Mike.bird assured **@liamthememelord** that updates regarding order statuses, which are currently "still cooking," would be communicated through email notifications to customers.
  
- **Request for Extreme Transparency**: User **8i8__papillon__8i8d1tyr** made a tongue-in-cheek request for the full sourcing of parts and a short bio, including the spirit animal, of each employee involved in the production process.
  
- **Installation Questions and Poetry Puzzles**: Members discussed various installation hurdles, particularly dealing with the correct use of `poetry` and `pip` on different platforms like Windows PowerShell and MacOS, with one user being stuck on a command not found error for `poetry`.
  
- **Helpful Setup Video Shared**: A member, **tsmith.tx**, provided a video walkthrough for setting up the **01 Light** and connecting it to a local server which helps incase of encountering difficulties during the setup process.


**Link mentioned**: <a href="https://youtu.be/Y76zed8nEE8">01 Light Setup - Flash and Connect to Server</a>: Quick video to show how to flash the 01 Light and get it set up on a server running 01OS locally with OpenAI models.I&#39;m following the instructions at h ttps:...

  

---


**OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1227979796085084240)** (1 messages): 

- **Transformers.js Take Machine Learning to the Browser**: A new project called [transformers.js](https://github.com/xenova/transformers.js) was mentioned, which is a JavaScript port of the **HuggingFace Transformers library**. This allows state-of-the-art machine learning models to run directly in the browser without the need for a server.

**Link mentioned**: <a href="https://github.com/xenova/transformers.js">GitHub - xenova/transformers.js: State-of-the-art Machine Learning for the web. Run 🤗 Transformers directly in your browser, with no need for a server!</a>: State-of-the-art Machine Learning for the web. Run 🤗 Transformers directly in your browser, with no need for a server! - xenova/transformers.js

  

---



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1227525694892806186)** (67 messages🔥🔥): 

- **Prompt Management System Curiosity**: A member inquired about experiences using prompt management systems like [vellum.ai](https://vellum.ai) for the creation, testing, and deployment of prompts across different models.
- **Discussion on the Utility of AI Voice Options**: In the AI voice preference discussion, *Sky* and *Juniper* were mentioned as favored text-to-speech voices, with no consensus on an "actual" voice for chat models like ChatGPT or Mistral.
- **Choosing the Right AI for Reasoning**: Members debated the reasoning capabilities of various AI models, with some endorsing **Claude 3 Opus**, while others were skeptical about any AI model's ability to truly reason.
- **Perplexity's Payment Model and Verification Work**: Conversations on Perplexity involved a member expressing concerns about paid beta testing as a common practice, and the additional verification work users must do to ensure the sources cited by AI services are accurate.
- **Technical Assistance for OpenAI's Assistant API**: A user sought help with OpenAI's Assistant API code, where the 'Beta' object had no attribute 'messages', leading to a discussion on outdated documentation and solutions involving updated method calls for the beta version of client threads.



**Link mentioned**: <a href="https://status.openai.com/">OpenAI Status</a>: no description found

  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1227571243797184522)** (29 messages🔥): 

- **Rhyme Time Challenge for GPT-3.5/4**: A member inquired about how to get GPT-3.5 or GPT-4 to generate rhymes.
- **Billing Blues and API Dilemmas**: A user experienced an issue with billing as their API access was suspended due to payment failure, despite having a sufficient balance. They requested moderator assistance for the situation.
- **Is OpenAI a Ghost Town?**: Users reported experiencing high latency with OpenAI's GPT, with one mentioning that the issue is leading to slower response times.
- **Chat GPT-4 Amnesia or Outage?**: Several users faced issues with GPT-4 not finding existing conversations and speculated about possible service outages.
- **The Chatbot Rollercoaster**: Reports of Chat GPT downtimes and recoveries swept the discussions, spiked with concerns over reliability and calls for the service to remain free during such outages. A user also cited [information from OpenAI's status page](https://status.openai.com/) about recent incidents and resolutions.

**Link mentioned**: <a href="https://status.openai.com/">OpenAI Status</a>: no description found

  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1227772580451192892)** (14 messages🔥): 

- **GPT-4 Turbo's Creative Edge**: Members are observing improvements in **GPT-4-turbo-2024-04-09**, noting it appears more "creative and 'alive'" compared to previous versions, though it still has a tendency to condense code. 
  Specific strategies such as avoiding code blocks are mentioned to counteract this compression issue.
- **Strategies Against Code Compression**: Members shared tips to manage undesirable code compressing by GPT models, which involve offering [custom instructions](http://drinkoblog.weebly.com) or using prompt chains to progressively build up the desired code output.
- **Prompt Chains for More Precise Coding**: A member advises breaking down requests into smaller, manageable portions of about 200 lines of code to tackle issues with revision and to prevent the model from providing incomplete code.
- **Accessing Wolfram GPT Made Easy**: If you need to use Wolfram within GPT, it is suggested to use the **Wolfram GPT model** which can be accessed through a link shared in the chat. Once activated, the Wolfram GPT can be summoned in any conversation using the `@mention` feature.
- **Resources for Budding Prompt Engineers**: A member seeking resources on prompt engineering was directed to [promptingguide.ai](http://promptingguide.ai) for comprehensive information on the subject.
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1227772580451192892)** (14 messages🔥): 

- **GPT-4 Turbo Shows Improvement**: Members have noticed an improvement in **GPT-4-turbo-2024-04-09**, observing less refusal in prompts and a more "creative and alive" model, while acknowledging it still tends to compress code. They suggest avoiding code blocks to mitigate this issue.
- **Tackling Compression in Code Output**: Users express frustration with the model's tendency to compress code but share their workaround, including providing custom instructions to prevent such compression.
- **Strategic Prompt Chaining Suggested**: A user recommends a strategy for prompt engineering that involves low expectations and a chain of prompts to refine code outputs, along with advice to keep code modules under 200 lines to avoid issues.
- **Advice on Integrating Wolfram GPT**: A user inquires about using Wolfram with the GPT model, and another member points them to Wolfram GPT at [Wolfram GPT](https://chat.openai.com/g/g-0S5FXLyFN-wolfram) and suggests using the `@mention` feature after initial use.
- **Beginning with Prompt Engineering**: For those new to prompt engineering, a reference to an informative site is provided: [Prompting Guide](http://promptingguide.ai).
  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1227632149780234341)** (3 messages): 

- **Control Your Agent with IFTTT Execution Stops**: IFTTT integration can enhance control over agents, like terminating a travel agent's process after booking confirmation or ending an *agentic RAG pipeline* post-reply. Excitement around these tools is shared with a teaser link on [Twitter](https://t.co/ByGOaqgWMd).

- **Simplified ColBERT-Based Retrieval Agent**: A new method for building a ColBERT-based retrieval agent, capable of advanced document search with conversation memory, is highlighted. The approach promises simplicity and effectiveness, with further details hinted at through a [Twitter post](https://t.co/wVDpML9juP).

- **Create a Chat with Your Code App Tutorial**: An interactive tutorial by @helloiamleonie of @weaviate_io showcases how to create an app enabling users to chat with a GitHub code repository. The tutorial utilizes a local LLM and embedding model, with the step-wise guide teased on [Twitter](https://t.co/yZN3QZjgN4).
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1227560655058636801)** (89 messages🔥🔥): 

- **Integration Inquiry for Structured LLM Output**: A user inquired about integrating [Instructor](https://python.useinstructor.com/) with LlamaIndex to stream structured output from an API server to the frontend. There was no subsequent discussion or resolution provided in the channel.
- **Issue Running sec-insights App Locally**: A user experienced an error while trying to run the [sec-insights](https://github.com/run-llama/sec-insights) app locally, but could not complete the `make seed_db_local` command. They shared detailed error logs and several users discussed the issue without immediate resolution, though one suggested watching an [end-to-end guide video](https://youtu.be/2O52Tfj79T4?si=CYUcaBkc9P9g_m0P) for help.
- **Desire for Detailed LLM Call Debugging**: Users expressed a need for improving observability of LlamaIndex, specifically to see the **exact prompt** sent to Large Language Models (LLMs). It was acknowledged that LlamaIndex's native instrumentation might assist with this, and users discussed individually recreating chat loops to gain better control over system prompt behavior.
- **Troubleshooting LlamaIndex's MongoDB Integration**: A user faced a `TypeError` claiming an issue with `KVDocumentStore.__init__()`. Upon reporting the error and sharing relevant code, another user suggested updating LlamaIndex packages as a solution and provided commands for doing so.
- **Discussion on CPU-based LLM Server Frameworks**: A user asked about LLM server frameworks that can run on CPU to serve multiple inferences in parallel. It was noted that running LLM on CPU could be inefficient, but it was suggested that user-built chat loops or an autoscaling Kubernetes cluster with Ollama may achieve this goal, albeit without known batch-inferencing frameworks.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.llamaindex]">no title found</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1P0RiVeQQF5z09A4KxvWuYGzv2UoJUIsX?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://python.useinstructor.com/">Welcome To Instructor - Instructor</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/#custom-node-postprocessor">Node Postprocessor - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_memory/?h=query+pipeline">Query Pipeline Chat Engine - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/index_structs/knowledge_graph/NebulaGraphKGIndexDemo/#query-with-embeddings">Nebula Graph Store - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/sec-insights">GitHub - run-llama/sec-insights: A real world full-stack application using LlamaIndex</a>: A real world full-stack application using LlamaIndex - run-llama/sec-insights</li><li><a href="https://github.com/run-llama/llama_index/blob/2b77f89775840d6b796bcc693f7593d2aebc5fec/llama-index-integrations/llms/llama-index-llms-ollama/llama_index/llms/ollama/base.py#L56">llama_index/llama-index-integrations/llms/llama-index-llms-ollama/llama_index/llms/ollama/base.py at 2b77f89775840d6b796bcc693f7593d2aebc5fec · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://github.com/run-llama/rags">GitHub - run-llama/rags: Build ChatGPT over your data, all with natural language</a>: Build ChatGPT over your data, all with natural language - run-llama/rags</li><li><a href="https://youtu.be/2O52Tfj79T4?si=CYUcaBkc9P9g_m0P">Discover LlamaIndex: SEC Insights, End-to-End Guide</a>: secinsights.ai is a full-stack app that uses the Retrieval Augmented Generation (RAG) capabilities of LlamaIndex to answer questions about SEC 10-K &amp; 10-Q do...</li><li><a href="https://github.com/run-llama/llama_index/pull/12736">[BUGFIX] Update LlamaIndex-Predibase Integration by alexsherstinsky · Pull Request #12736 · run-llama/llama_index</a>: Description  The Predibase API has changed.  This contribution updates the implementation of connecting to and prompting the Predibase LLM service on the LlamaIndex side.  Once this pull request is...
</li>
</ul>

</div>
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1227642930043289681)** (68 messages🔥🔥): 

- **CI Tests Lacking for copy_from_fd**: It's noted that the Continuous Integration tests for **tinygrad** are missing for `copy_from_fd`; this was identified during a recent [GitHub Action job](https://github.com/tinygrad/tinygrad/actions/runs/8633930065/job/23668153464). A user has acknowledged the issue and plans to add tests in a separate pull request.
  
- **Rust Export Feature Proposal Rejected**: A pull request proposing Rust export capabilities for **tinygrad** was [declined](https://github.com/tinygrad/tinygrad/pull/4138) as it diverges from the project's focus. The leader emphasizes **tinygrad**’s commitment to C for performance and maintainability, suggesting Rust can call compiled C libraries if necessary.
  
- **Emphasis on Performance Over Language Preference**: In response to a discussion about generating Rust code from **tinygrad**, it was emphasized that neural networks are memory safe by design, and the focus should be on optimizing the performance of the generated C code rather than diversifying language backends.
  
- **Code Standardization for mnist Datasets**: A user observed an inconsistency in the way **tinygrad** handles the mnist dataset across different files, and three solutions were proposed, ranging from minor changes in the example files to maintaining current dataset fetching in an extra directory.
  
- **Discussions on Memory Safety and Political Implications of Language Use**: A user highlighted the memory safety record of Rust and shared their disapproval of the Rust Foundation's approach to trademarking and licensing, comparing the organizational practice to Java's situation with Oracle and voicing a personal stance against using licensed products.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lunduke.substack.com/p/the-rust-foundation-goes-to-war-against">The Rust Foundation goes to war against people using the word &quot;Rust&quot;</a>: Seriously. The title of this article violates the new Rust Trademark Policy. It&#x27;s insane.</li><li><a href="https://www.cvedetails.com/vulnerability-list/vendor_id-19029/product_id-48677/Rust-lang-Rust.html">Rust-lang Rust : Security vulnerabilities, CVEs </a>: Security vulnerabilities of Rust-lang Rust : List of vulnerabilities affecting any version of this product </li><li><a href="https://github.com/tinygrad/tinygrad/actions/runs/8633930065/job/23668153464">no more underlying diskbuffer, that&#39;s just the device (#4129) · tinygrad/tinygrad@ee457a4</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - no more underlying diskbuffer, that&#39;s just the device (#4129) · tinygrad/tinygrad@ee457a4</li><li><a href="https://github.com/tinygrad/tinygrad/commit/b0f14b4af886de8bc04a4cacc48880af24d69632?diff=unified&w=0)">move datasets into datasets · tinygrad/tinygrad@b0f14b4</a>: no description found</li><li><a href="https://github.com/tinygrad/tinygrad/pull/4138#">Rust backend by jaredeh · Pull Request #4138 · tinygrad/tinygrad</a>: Adding Rust backend.  LOC neutral (assuming can park rust.py in ./extra) Tests mostly pass Includes examples/export_code_from_onnx.py which creates a zero dependency pure Rust rlib crate Performanc...
</li>
</ul>

</div>
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1227515872260915291)** (23 messages🔥): 

- **Tensor Conversions Clarified**: A simple solution was provided for converting tensors to numpy arrays using `.numpy()` method, aligning with a user's requirements.
- **An Exercise in Device Implementation**: An exercise was suggested for learners to implement an "NPY" device to enable storing tensors, demonstrated by `Tensor(1).to("NPY")`.
- **George Hotz Goes Against NPY Device Idea**: Initially, George Hotz dismissed the exercise to make NPY a device, emphasizing that NPY should only be for copying out to GPU and not vice-versa.
- **Teaching Opportunity Recognized**: Despite initial dismissal, George Hotz later acknowledged that creating an NPY device could be a valuable learning exercise regarding allocators in the codebase.
- **Exercise Quality Debated**: There was a brief discussion about the usefulness of small exercises versus more complex bounties, with George Hotz suggesting that simpler tasks could be beneficial as they allow learners to check their work against a correct answer.
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1227576296012255253)** (78 messages🔥🔥): 

- **AI Art Enthusiasts Discuss Training and Outsourcing**: Community members expressed skepticism about **Tencent's Ella SDXL** and contemplated whether other entities like 'Draw Things' or **CivitAI** would attempt to train a variant of it. Budgetary constraints and strategic choices about different AI projects were hot topics, with some members sharing insights [from a reddit thread discussing Tencent's refusal to release the SDXL version](https://reddit.com).
  
- **Exploration of Music AI and Artist Involvement**: **Udio's music creation app** was introduced, collecting attention for its goal to be a tool for musicians, featuring endorsements from artists such as [will.i.am and Common](https://x.com/udiomusic/status/1778045337833193720). Queries about user involvement and the ability to upload real instrument tracks were discussed, along with the use of a latent diffusion model for music generation.

- **Diverse AI Hardware and Model Acceleration**: Community members discussed the proliferation of AI acceleration hardware by big players, pointing out the new [Meta Training and Inference Accelerator (MTIA) AI hardware revealed by Meta](https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/), emphasizing its 354 TFLOPS at a 90 W TDP as particularly impressive.

- **Text-to-Speech Technological Advancements** were explored, with a focus on a new TTS engine by the Huggingface team that uses a similar architecture to Stability's unreleased TTS, supporting voice prompting capabilities: [Parler TTS](https://github.com/huggingface/parler-tts).

- **Laion 5B Web Demo's Uncertain Future**: Users inquired about the status of **Laion 5B's web demo** with responses indicating no clear return date, caught between legal issues and administrative processes. However, alternatives like creating a personal dataset using [cc2dataset](https://github.com/rom1504/cc2dataset) were suggested for those seeking similar search engine capabilities.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.udio.com/songs/renWwtB7Zqk2mqZamEHHgJ>">no title found</a>: no description found</li><li><a href="https://tenor.com/mDErrG5aLdg.gif">Pinoquio GIF - Pinoquio - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/udiomusic/status/1778045337833193720">Tweet from udio (@udiomusic)</a>: Our goal is to make Udio a game-changing tool for both musicians and non-musicians alike, and we are excited to be backed by leading artists @iamwill and @common.   8/11</li><li><a href="https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/">no title found</a>: no description found</li><li><a href="https://x.com/udiomusic/status/1778045322654003448">Tweet from udio (@udiomusic)</a>: Introducing Udio, an app for music creation and sharing that allows you to generate amazing music in your favorite styles with intuitive and powerful text-prompting.  1/11</li><li><a href="https://news.ycombinator.com/item?id=39992817">Show HN: Sonauto – A more controllable AI music creator | Hacker News</a>: no description found
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1227717146528972931)** (7 messages): 

- **Intel Lunar Lake to Empower Copilot AI**: Intel's next generation [Lunar Lake CPUs](https://www.reddit.com/r/singularity/comments/1c0ue4f/intels_nextgen_lunar_lake_cpus_will_be_able_to/) are expected to run Microsoft's Copilot AI locally thanks to a powerful 45 TOPS neural processing unit (NPU).
- **Hardware Supply Concerns**: A member questioned if there's a bottleneck for chip fabrication in high-end AI hardware or if Nvidia might be restricting supply.
- **Semiconductor Fabrication Insights**: It was noted that **Intel has their own fabrication facilities** while AMD and Nvidia rely on TSMC for their production needs.
- **LRU Modifications Impact Benchmarks**: Modified least recently used (LRUs) algorithms are mentioned as effective according to the Long Range Arena (LRA) benchmark.
- **Real-World Performance of LRA Questioned**: A discussion emerged about the **effectiveness of LRA** in modeling **real-world long context performance**.

**Link mentioned**: <a href="https://www.reddit.com/r/singularity/comments/1c0ue4f/intels_nextgen_lunar_lake_cpus_will_be_able_to/">Reddit - Dive into anything</a>: no description found

  

---


**LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/1227931111926009886)** (1 messages): 

- **Request for HowTo100M dataset access**: A member is inquiring about access to the **HowTo100M dataset** and wonders if this is the correct channel to ask. The dataset is located at [di.ens.fr](https://www.di.ens.fr/willow/research/howto100m/).
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1227524493522767892)** (60 messages🔥🔥): 

- **Tracking OpenAIAssistant Token Usage**: A user inquired about how to track token usage for OpenAIAssistant. It was suggested to hook into the LLM callbacks, count tokens with **tiktoken**, and multiply by pricing for cost estimation.

- **Metadata Utilization in Vector Databases**: There was a discussion on utilizing metadata filters within vector databases of company policies to answer specific queries, such as finding companies that allow negative vacation leave balances. A member explained that metadata filters scope the search space and don't pass context to the LLM, but can be customized with a retriever to include metadata.

- **With_structured_output Method Inquiry**: A user questioned whether the `with_structured_output` method in the **ChatOpenAI class** is deprecated. It was clarified via shared documentation that the method is not deprecated but is in beta, with a code example given to demonstrate its use in JavaScript.

- **Exploring Structured Output Options**: One user brought up the tool [Instructor for Python](https://python.useinstructor.com/) and asked how it can be integrated with LangChain. Another user pointed to [LangChain Python documentation](https://python.langchain.com/docs/modules/model_io/chat/structured_output/) for information on structuring LLM outputs.

- **Personalized AI Model Access for Mobile**: An inquiry was made about personalizing AI access with the ability to test various models, especially for use on mobile devices. One user is considering building a script for **Pythonista** and leveraging **Bing's API** for web results due to its good latency.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://python.langchain.com/docs/get_started/introduction#api-reference>).">Introduction | 🦜️🔗 LangChain</a>: LangChain is a framework for developing applications powered by large language models (LLMs).</li><li><a href="https://python.useinstructor.com/">Welcome To Instructor - Instructor</a>: no description found</li><li><a href="https://python.langchain.com/docs/modules/model_io/chat/structured_output/">[beta] Structured Output | 🦜️🔗 LangChain</a>: It is often crucial to have LLMs return structured output. This is</li><li><a href="https://python.langchain.com/docs/guides/structured_output#openai>).">[beta] Structured Output | 🦜️🔗 LangChain</a>: It is often crucial to have LLMs return structured output. This is</li><li><a href="https://python.langchain.com/docs/guides/structured_output#openai>)">[beta] Structured Output | 🦜️🔗 LangChain</a>: It is often crucial to have LLMs return structured output. This is</li><li><a href="https://js.langchain.com/docs/integrations/chat/openai#withstructuredoutput-->).">ChatOpenAI | 🦜️🔗 Langchain</a>: You can use OpenAI&#x27;s chat models as follows:</li><li><a href="https://github.com/outlines-dev/outlines">GitHub - outlines-dev/outlines: Structured Text Generation</a>: Structured Text Generation. Contribute to outlines-dev/outlines development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1227860949105381407)** (5 messages): 

- **Request for Non-OpenAI LLM Chan**: A member requested instructions to create a **LangChain** with function calling for an open-source LLM that isn't OpenAI.
- **LangChain Versatility Acknowledged**: In response to the request, it was noted that LangChain's design is inherently independent of any specific LLM provider, suggesting adaptability with various LLMs.
- **Lack of Specific Instructions**: The follow-up response highlighted that the knowledge base lacks precise examples or directions for writing a chain with function calling for a non-OpenAI open-source LLM.
- **Guide to Find More Information**: For detailed guidance on integrating an open-source LLM with LangChain, the member was directed to consult the official [LangChain documentation](https://langchain.readthedocs.io/) or seek help from the community.
  

---


**LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/)** (1 messages): 

lhc1921: https://python.langchain.com/docs/integrations/llms/azure_openai/
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1227871914635038762)** (3 messages): 

- **GPT AI Debuts with GPT-4 & Vision AI**: A new app called **GPT AI** has been introduced, featuring **GPT-4** and a *Vision AI* technology for image recognition, offering a variety of functionalities like data analysis, language learning, and coding. It claims an aesthetically pleasing interface, no conversation limit, and instant mode switching, available for download at [Google Play Store](https://play.google.com/store/apps/details?id=com.projecthit.gptai&referrer=ph-aai).

- **Galaxy AI Unveils Free Premium AI APIs**: **Galaxy AI** promotes access to **free API** service for PREMIUM AI models, including the latest **Gemma**, **GPT-4**, **GPT-3.5-turbo**, and **Gemini-PRO API**, with **OpenAI format** for easy project integration. Interested users are invited to [try now](https://discord.com/invite/BSphj69773).

- **Appstorm Platform Elevates App-Building with v1.6.0**: **Appstorm v1.6.0** has been released, featuring mobile registration, the ability to generate music, integration of maps, enhanced data exploration and sharing functionalities, along with platform improvements to handle more concurrent apps and bug fixes for app resumption. It's currently accessible at [beta.appstorm.ai](https://beta.appstorm.ai/).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://galaxyapi.onrender.com">Galaxy AI - Swagger UI</a>: no description found</li><li><a href="https://beta.appstorm.ai/">Appstorm</a>: Build an AI app in seconds</li><li><a href="https://play.google.com/store/apps/details?id=com.projecthit.gptai&referrer=ph-aai">GPT AI - Chat GPT-4 &amp; Vision - Apps on Google Play</a>: no description found
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1227558541414629408)** (5 messages): 

- **Searching for Web Development Templates**: A member expressed their hope to find readily available web development templates, only to learn that they might need to delve into web development to achieve their goal.
- **Is it a Hot Dog? Image Classification Tutorial**: A tutorial video has been shared titled ["Hot dog or not with Ollama, Mistral and LLava"](https://www.youtube.com/watch?v=76BkMULO7uw), which teaches viewers how to determine if an image features a hot dog using various machine learning models.
- **New Tutorial on LCEL and Runnables**: A member has published a tutorial focused on LangChain's LCEL and composing chains from runnables, inviting feedback from the community. The tutorial is available at [Langchain Tutorial: LCEL and Composing Chains from Runnables](https://medium.com/@klcoder/langchain-tutorial-lcel-and-composing-chains-from-runnables-751090a0720c?sk=55c60f03fb95bdcc10eb24ce0f9a6ea7).

**Link mentioned**: <a href="https://www.youtube.com/watch?v=76BkMULO7uw">Hot dog or not with Ollama, Mistral and LLava</a>: In this tutorial we take a look at whether an image is hot dog or not using Ollama, mistral and lava#python #pythonprogramming #llm #ml #ai #aritificialintel...

  

---



**DiscoResearch ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/1227544935398506586)** (10 messages🔥): 

- **Mixtral Model Conversion Scripts Disclosed**: Members shared links for converting Mistral MoE weights to Hugging Face format: an [unofficial script](https://huggingface.co/DiscoResearch/mixtral-7b-8expert/blob/main/convert_mistral_moe_weights_to_hf.py) by a community member and the [official script](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/convert_mixtral_weights_to_hf.py) found in the transformers repository.

- **Launching of Mixtral-8x22B Model on HF**: A new **Mixtral-8x22B** model was introduced, available for download at Hugging Face. The [model card](https://huggingface.co/v2ray/Mixtral-8x22B-v0.1) includes conversion scripts and acknowledges Hugging Face staff for cloning to an official repository.

- **Mistral Community Shares 176B Parameter Model Specs**: A member cited from the discord that there's a 176B parameter model with performance between GPT-4 and Claude Sonnet, **using the same tokenizer as Mistral 7b** and having a massive sequence length of **65536**.

- **Confusion Over Model Performance Clarified**: In response to a previously shared post, another member clarified that the comparative performance between GPT-4 and Claude Sonnet was actually referring to the command-r+ model, not Mistral.

- **Experimental Merge of 22B MoE Model**: An experimental **22B parameter dense model**, crafted from a MOE model, was released by Vezora on Hugging Face. [Mistral-22B-v0.1](https://huggingface.co/Vezora/Mistral-22B-v0.1) is the first successful MOE to Dense model conversion announced.

- **Challenges with Mergekit and Fine-tuning**: Despite efforts to merge models using Mergekit and later fine-tune them, a member reported poor results, indicating similar experiences are common within the community, leading some to avoid custom merges for training.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Vezora/Mistral-22B-v0.1">Vezora/Mistral-22B-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1">HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/v2ray/Mixtral-8x22B-v0.1">v2ray/Mixtral-8x22B-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/DiscoResearch/mixtral-7b-8expert/blob/main/convert_mistral_moe_weights_to_hf.py">convert_mistral_moe_weights_to_hf.py · DiscoResearch/mixtral-7b-8expert at main</a>: no description found</li><li><a href="https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/convert_mixtral_weights_to_hf.py">transformers/src/transformers/models/mixtral/convert_mixtral_weights_to_hf.py at main · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1227517055771869214)** (22 messages🔥): 

- **Apache 2.0: The License of Choice**: Contributions to the discussion confirmed that the model in question is licensed under Apache 2.0, and the instruct version is expected to be released soon. There is a particular interest in the licensing terms of newly shared models.

- **Mixtral Dominates the Benchmarks**: Initial AGIEval results show **Mixtral** models outperforming other base models, with members expressing astonishment at their impressive performance. Specifics of the benchmarks include **Mixtral-8x22B-v0.1** achieving notable scores on tasks like **PIQA, BoolQ**, and others.

- **Benchmark Comparisons and Contextual Details**: Discussion provided comparisons of **Mixtral** model performance across various tasks alongside details on the benchmarking environment, revealing that evaluations are being conducted on **vLLM** using **4xH100s**.

- **Direct Model Acquisition and Discussions**: A member confirmed the use of a specific **Mixtral-8x22B-v0.1** model available on Hugging Face, and linked to a [discussion](https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1/discussions/3#6616af73203bf9d751696a84) where **5-shot MMLU** results were showcased, indicating new open-source SOTA performance.

- **Potential for German Benchmarking**: Interest in German language benchmarking sparked a question about suitable benchmarks, hinting at the potential use of **lm-evaluation-harness** but also questioning the current relevance and demand for German benchmark data.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/v2ray/Mixtral-8x22B-v0.1">v2ray/Mixtral-8x22B-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1/discussions/3#6616af73203bf9d751696a84">mistral-community/Mixtral-8x22B-v0.1 · MMLU - 77</a>: no description found
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1227528811751669843)** (41 messages🔥): 

- **Benchmarking Reveals Puzzling Model Behavior**: PhilipMay highlighted a curious performance discrepancy in `DiscoResearch/DiscoLM_German_7b_v1` based on the presence of a newline character in the ChatML Template. The model performed significantly better when a newline was present before `
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2211.01786">Crosslingual Generalization through Multitask Finetuning</a>: Multitask prompted finetuning (MTF) has been shown to help large language models generalize to new tasks in a zero-shot setting, but so far explorations of MTF have focused on English data and models....</li><li><a href="https://huggingface.co/occiglot/occiglot-7b-de-en-instruct">occiglot/occiglot-7b-de-en-instruct · Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/html/2404.04850v1">Lucky 52: How Many Languages Are Needed to Instruction Fine-Tune Large Language Models?</a>: no description found</li><li><a href="https://huggingface.co/DiscoResearch/DiscoLM_German_7b_v1/blob/main/tokenizer_config.json#L48">tokenizer_config.json · DiscoResearch/DiscoLM_German_7b_v1 at main</a>: no description found</li><li><a href="https://huggingface.co/DiscoResearch/DiscoLM-70b#dataset">DiscoResearch/DiscoLM-70b · Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/html/2404.04659v1">Multilingual Pretraining and Instruction Tuning Improve Cross-Lingual Knowledge Alignment, But Only Shallowly</a>: no description found</li><li><a href="https://huggingface.co/datasets/deutsche-telekom/Ger-RAG-eval">deutsche-telekom/Ger-RAG-eval · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/huggingface/lighteval/blob/main/community_tasks/german_rag_evals.py">lighteval/community_tasks/german_rag_evals.py at main · huggingface/lighteval</a>: LightEval is a lightweight LLM evaluation suite that Hugging Face has been using internally with the recently released LLM data processing library datatrove and LLM training library nanotron. - hug...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/5ed29393e34cf57b24a20ac1bafa3a94272ac3f5/src/axolotl/prompt_strategies/dpo/chatml.py#L86">axolotl/src/axolotl/prompt_strategies/dpo/chatml.py at 5ed29393e34cf57b24a20ac1bafa3a94272ac3f5 · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1227554085268619274)** (23 messages🔥): 

- **New Model on the Horizon**: A completely new model is confirmed by Sophia Yang with a tweet, explicitly stating that it's neither a *Mistral* nor an *EleutherAI* derivative. The model's license is Apache 2.0 as confirmed in the Discord.
  
- **Rushed Release Speculation**: Rumors of upcoming releases from **Llama** and **Cohere** might have pressured the release of the new model without full evaluations, indicated by a lack of checksum in the announcement tweet.

- **Benchmarks Dismiss Docs**: While **Mistral** traditionally delays complete documentation following a release, encouraging community hype, [AGIEval results suggest](https://fxtwitter.com/jphme/status/1778028110954295486) the new 8x22b model is outperforming other open-source base models.

- **Call for Unbiased Evals**: A member expressed interest in starting a blog to release public and unbiased human evals with each major model release, amidst frustration over benchmarks that may not align with developer needs for product building.

- **Mistral Community Goes Base Vs. R+**: An image from Hugging Face reveals **BigXtra**'s base model performance compared to its R+ version which is instruction-tuned; a discussion invites speculation on the datasets based on the eval scores. [View the discussion](https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1/discussions/4) and [evaluation scores image](https://cdn-uploads.huggingface.co/production/uploads/6382255fcae34727b9cc149e/ds0rDhvNWZl0dWsrfYb0K.jpeg).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/sophiamyang/status/1777978822199017728">Tweet from Sophia Yang, Ph.D. (@sophiamyang)</a>: @LiHongtu12138 neither. it&#39;s a completely new model.</li><li><a href="https://fxtwitter.com/jphme/status/1778028110954295486">Tweet from Jan P. Harries (@jphme)</a>: First AGIEval results for @MistralAI s new 8x22b model are in, destroying all other open source (base) models  - 🤯
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1227846875843919966)** (7 messages): 

- **Inquiring the Pretrain-to-RLHF Transition**: A member was curious about if a step like **instruction-tuning (IFT)** is necessary in the usual **Pretraining -> IFT -> RLHF** pipeline for language models. They pondered whether jumping directly to **RLHF** (Reinforcement Learning from Human Feedback) from a pretrained model is feasible, considering human preference ratings implicitly teach instruction following during RLHF. They hypothesized that instruction following might be simpler to teach with a supervised dataset given its straightforward nature, contrasting with the complexity of generating good output. The query was sparked by a student's question in a [Stanford CS224N lecture](https://www.youtube.com/watch?v=SXpJ9EmG3s4).

- **Blurred Lines in Training Phases**: In response, another member hinted that the boundaries between pretraining, instruction following tuning, and RLHF are not clear-cut in practice, implying that the training processes are **blended together**.

- **Seeking Clarity on Model Training Blending**: The member seeking information pondered over what "blending" means, asking if it involves combining the datasets for pretraining, SFT (Supervised Fine Tuning), and RLHF prompts in some way, and noted the variance in the objective functions used in PT/SFT versus RLHF. They requested further resources for clarity.

- **Anticipation for Upcoming Resources on Annealing**: The same member who discussed the blended practice mentioned upcoming, potentially undocumented methods in **model training**, including curriculums, schedulers, or annealing. They teased that there should be something on **annealing** available soon.

**Link mentioned**: <a href="https://www.youtube.com/watch?v=SXpJ9EmG3s4&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4&index=10">Stanford CS224N | 2023 | Lecture 10 - Prompting, Reinforcement Learning from Human Feedback</a>: For more information about Stanford&#39;s Artificial Intelligence professional and graduate programs visit: https://stanford.io/aiTo learn more about this course...

  

---


**Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1227737709792198797)** (5 messages): 

- **Machine Learning Insider Trading Accusation**: A member highlighted a scenario suggesting insider trading, attributable to the poor quality of a fine-tune. The implication is that the fine-tuning process did not meet the expected standards.
- **Academic Conflicts of Interest**: The same member raised concerns about possible conflicts of interest, pointing to **machine learning system professors** potentially being invested in companies like **Databricks**.
- **Burkov's Tweet Sparks Discussion**: A tweet by Anton Burkov was shared within the group as a starting point for a conversation. Specific content or context from the tweet was not discussed in the messages provided. [Burkov's Twitter Post](https://twitter.com/burkov/status/1778123473929282015)
- **Tense Atmosphere Over Allegations**: The term "SPICY" reflects the intense and possibly controversial nature of the allegations being discussed.
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1227750875901067274)** (14 messages🔥): 

- **Possible John Schulman Interview Tease**: The interview with **John Schulman** was mentioned with uncertainty, but there's hope to convert this from a "maybe" to a "yes."
- **The Accidental Yes**: A user humorously notes the potential accidental confirmation of an unspecified topic. It was suggested that it might be typed in as a meme, yet found amusing.
- **Mischief with Meeting Confirmations?**: A member points out that their system doesn't require acceptance for subscriptions, hinting at a reason for unexpected confirmations.
- **New Member Recruitment Strategy**: There was a brief exchange about recruiting new members to the server, including a notable reference to someone named **Satya**.
  

---



**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1228020142512804032)** (4 messages): 

- **Audio Awareness in AI Skyrockets**: Gemini's updated capability now includes answering questions about audio in videos, which marks a significant enhancement from its prior limitation of only generating descriptions of video content without audio.

- **Google's Formatting Frustrations**: A user expresses the need for Google to address the text formatting issues encountered when pasting text into their playground.

- **Curating Knowledge with AI**: [Stanford's Storm project](https://github.com/stanford-oval/storm) on GitHub is an exciting development; it's an LLM-powered knowledge curation system designed to research topics and generate comprehensive reports, complete with citations.

**Link mentioned**: <a href="https://github.com/stanford-oval/storm">GitHub - stanford-oval/storm: An LLM-powered knolwedge curation system that researches a topic and generates a full-length report with citations.</a>: An LLM-powered knolwedge curation system that researches a topic and generates a full-length report with citations. - stanford-oval/storm

  

---


**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1227674842107740280)** (12 messages🔥): 

- **LLM Command Tool Troubles**: A user encountered an issue where the `llm cmd` was hanging on macOS using iTerm2, with no errors being reported. Even when attempting to run the command via SSH on an Ubuntu server, the issue seemed local to the MacOS setup.

- **Shell Customization Not at Fault**: The user noted having a *highly customized omzsh shell* but confirmed this wasn't causing the issue since they used the same configuration on Ubuntu where `llm cmd` worked.

- **Testing Different Installations**: In troubleshooting, they switched from the homebrew to the pipx version without success, and confirmed the problematic behavior was not reflected in the `llm logs`.

- **Potentially an Interaction Prompt Issue**: They discovered the command was waiting for an interaction, returning output upon typing `'???'`, suggesting the process wasn't hanging, but awaiting input.

- **Pull Request Offers a Solution**: The user shared a [GitHub pull request](https://github.com/simonw/llm-cmd/pull/12) and confirmed it resolved the hanging issue with `llm cmd` on macOS zsh, indicating a potential fix for the problem being discussed.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/simonw/llm-cmd">GitHub - simonw/llm-cmd: Use LLM to generate and execute commands in your shell</a>: Use LLM to generate and execute commands in your shell - simonw/llm-cmd</li><li><a href="https://github.com/simonw/llm-cmd/pull/12">fix: macos zsh llm cmd hangs by nkkko · Pull Request #12 · simonw/llm-cmd</a>: Fix for #11, tested on M1 MacOs (14.3.) in Terminal and Alacritty (zsh), now works fine.
</li>
</ul>

</div>
  

---



**Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1228056986071990373)** (1 messages): 

- **Gradio meets Figma**: Mozilla Innovations introduces **Gradio UI for Figma**, a tool based on the low-code prototyping toolkit from Hugging Face, designed to streamline the design phase with quick wireframe creation to speed up experimentation. Access the Gradio UI library at [Figma's page for Mozilla](https://www.figma.com/@futureatmozilla).
- **Join the Conversation on Gradio UI**: For questions or to engage in further discussion about **Gradio UI for Figma**, join the thread with Thomas Lodato from Mozilla’s Innovation Studio at their [Discord discussion channel](https://discord.com/channels/1089876418936180786/1091372086477459557/1228056720132280461).

**Link mentioned**: <a href="https://www.figma.com/@futureatmozilla">Figma (@futureatmozilla) | Figma</a>: The latest files and plugins from Mozilla Innovation Projects (@futureatmozilla) — We&#39;re building products that focus on creating a more personal, private and open-source internet

  

---


**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1227593409972404265)** (11 messages🔥): 

- **GPU Memory Limitations Addressed**: A member successfully used `-ngl 3` to adjust for their GPU's limited memory, moving some layers to CPU memory instead, but reported that **performance suffered significantly**. They found that smaller models like 7B performed much better under these constrained conditions.

- **Feature Request for Dynamic Layer Offloading**: Discussing the limitation of VRAM on older GPU models like the 1050, a member queried if **llamafile** could intelligently offload layers to avoid crashing when VRAM is insufficient.

- **ollama's Handling of Large Language Models**: A link to the **ollama project on GitHub** was shared, highlighting how it manages operations with large language models, which could potentially inform improvements to llamafile's handling of memory.

- **Kernel Panic Triggered by Conversing with Tensors**: A member recounted causing a kernel panic on their M2 MacBook by attempting to convert a `.safetensors` file to `.gguf`, suggesting the panic was due to the device's limitation of 16GB RAM.

- **Introducing Quiet-STaR for Implicit Reasoning in Text**: A discussion on **Quiet-STaR**, a method for language models to generate rationales at each token to improve text predictions, was introduced with links to the [research paper](https://arxiv.org/abs/2403.09629) and the [GitHub repository](https://github.com/ezelikman/quiet-star). Hugging Face also featured a related repository for Quiet-STaR.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2403.09629">Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking</a>: When writing and talking, people sometimes pause to think. Although reasoning-focused works have often framed reasoning as a method of answering questions or completing agentic tasks, reasoning is imp...</li><li><a href="https://huggingface.co/ezelikman/quietstar-8-ahead/tree/main">ezelikman/quietstar-8-ahead at main</a>: no description found</li><li><a href="https://github.com/ezelikman/quiet-star">GitHub - ezelikman/quiet-star: Code for Quiet-STaR</a>: Code for Quiet-STaR. Contribute to ezelikman/quiet-star development by creating an account on GitHub.</li><li><a href="https://github.com/ollama/ollama/blob/c5c451ca3bde83e75a2a98ed9fd4e63a56bb02a9/llm/server.go#L43>">ollama/llm/server.go at c5c451ca3bde83e75a2a98ed9fd4e63a56bb02a9 · ollama/ollama</a>: Get up and running with Llama 2, Mistral, Gemma, and other large language models. - ollama/ollama
</li>
</ul>

</div>
  

---



**Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1227591389597007964)** (1 messages): 

- **Mistral 8x22b Crushes the Competition in AGIEval**: The initial [AGIEval results](https://x.com/jphme/status/1778030213881909451) for **Mistral 8x22b** have been shared, showing the model dramatically outperforming all other open-source base models. Kudos were given to the Mistral team for releasing the new AI model.

**Link mentioned**: <a href="https://x.com/jphme/status/1778030213881909451">Tweet from Jan P. Harries (@jphme)</a>: @MistralAI first AGIEval results look great 👇 - thanks for releasing this beast, guys! 👏 https://x.com/jphme/status/1778028110954295486  ↘️ Quoting Jan P. Harries (@jphme)   First AGIEval results fo...

  

---


**Skunkworks AI ▷ #[datasets](https://discord.com/channels/1131084849432768614/1131669182124138616/1227982498655633450)** (5 messages): 

- **Seeking Logic Datasets**: A member inquired about datasets for reasoning with propositional and predicate logic over natural text.
- **Curated List for Reasoning AI**: The [awesome-reasoning GitHub repository](https://github.com/neurallambda/awesome-reasoning), which provides a curated list of data for reasoning AI, was shared in response to a query on datasets for logical reasoning.
- **LOGIC-LM Project for Faithful Logical Reasoning**: Another member provided a link to the [Logic-LLM GitHub repository](https://github.com/teacherpeterpan/Logic-LLM), which focuses on empowering large language models with symbolic solvers for logical reasoning.
- **Towards COQ-Compatible LLMs**: The conversation included a mention of a [Coq dataset on arXiv](https://arxiv.org/abs/2403.12627), aimed at training large language models to handle the Coq proof language for formal theorem proving.
- **Clarifying Project Objectives**: A member sought clarification on a project's aim, wondering if it intended to enhance LLMs' reasoning abilities by converting human propositions into Lisp for execution and verification.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2403.12627">Enhancing Formal Theorem Proving: A Comprehensive Dataset for Training AI Models on Coq Code</a>: In the realm of formal theorem proving, the Coq proof assistant stands out for its rigorous approach to verifying mathematical assertions and software correctness. Despite the advances in artificial i...</li><li><a href="https://github.com/neurallambda/awesome-reasoning">GitHub - neurallambda/awesome-reasoning: a curated list of data for reasoning ai</a>: a curated list of data for reasoning ai. Contribute to neurallambda/awesome-reasoning development by creating an account on GitHub.</li><li><a href="https://github.com/teacherpeterpan/Logic-LLM">GitHub - teacherpeterpan/Logic-LLM: The project page for &quot;LOGIC-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning&quot;</a>: The project page for &quot;LOGIC-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning&quot; - teacherpeterpan/Logic-LLM
</li>
</ul>

</div>
  

---


**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1227622276351131799)** (2 messages): 

- **Google Unveils CodeGemma**: A new [YouTube video](https://www.youtube.com/watch?v=Gb--4supXoo) introduces **CodeGemma**, Google's code completion model, boasting a "powerful yet lightweight" 7B pretrained variant for coding capabilities.
- **Classifying Hot Dogs with AI**: Another [YouTube tutorial](https://www.youtube.com/watch?v=76BkMULO7uw) demonstrates how to determine if an image is a hot dog using AI models **Ollama, Mistral, and LLava** in a python programming context.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=76BkMULO7uw">Hot dog or not with Ollama, Mistral and LLava</a>: In this tutorial we take a look at whether an image is hot dog or not using Ollama, mistral and lava#python #pythonprogramming #llm #ml #ai #aritificialintel...</li><li><a href="https://www.youtube.com/watch?v=Gb--4supXoo">Introducing CodeGemma Google&#39;s code completion model</a>: CodeGemma brings powerful yet lightweight coding capabilities to the community. CodeGemma models are available as a 7B pretrained variant that specializes in...
</li>
</ul>

</div>
  

---



**LLM Perf Enthusiasts AI ▷ #[gpt4](https://discord.com/channels/1168579740391710851/1168582188950896641/1227622776542593074)** (6 messages): 

- **Debunking Slacking AI Rumors**: Despite claims on Twitter about GPT's coding prowess lapsing, a member reported no issues while using it within *cursor*, highlighting its faster performance and more complete code generation.
- **Gemini 1.5 Gains Praise**: While there are no detailed specifics, a member mentioned hearing positive feedback regarding **Gemini 1.5's** coding capabilities.
- **Cursor Gets a Nod**: A member expressed preference for using **cursor** for boilerplate code and appreciated the command-K feature with the new **GPT-4**, although for chats they still favor Claude opus.
- **Copilot++ Impressions**: The integration of **Copilot++** was praised by a member, suggesting it performs exceptionally well.
  

---


**LLM Perf Enthusiasts AI ▷ #[claude](https://discord.com/channels/1168579740391710851/1168582222194933860/1227875410054742017)** (2 messages): 

- **Claude's Hallucinating Code**: A user reported their first instance where **Claude** completely hallucinated some code, in a manner uncharacteristic of its predecessor, **GPT-4**.
  


---



**AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1228029803962302546)** (2 messages): 

- **In Search of Jamba's Code**: A member inquired about where to find the code for **Jamba**.
- **Anticipation for Updates**: Another member mirrored the quest for information with a simple query, "*Any update?*" indicating interest in recent developments or responses to previous inquiries.
  

