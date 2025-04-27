---
id: f0fcfec3-231c-497b-b30e-73388e597edb
title: not much happened today
date: '2025-01-08T04:01:51.252491Z'
original_slug: ainews-not-much-happened-today-7007
description: >-
  **NVIDIA** has launched **Cosmos**, an open-source video world model trained
  on **20 million hours of video**, aiming to transform **robotics** and
  **autonomous driving**. The release sparked debate over its open-source status
  and technical approach. Additionally, **NVIDIA** announced **Digits**, a
  $3,000 personal AI supercomputer designed for local AI fine-tuning,
  highlighting rapid AI hardware advancements. The AI community expresses mixed
  feelings about the pace of AI progress, with concerns about **AGI**, job
  displacement, and investment hype, while emphasizing the need to focus on
  present capabilities and gradual integration.
companies:
  - nvidia
  - openai
models:
  - cosmos
topics:
  - robotics
  - autonomous-driving
  - open-source
  - fine-tuning
  - memory-optimization
  - ai-hardware
  - model-training
  - model-performance
  - investment-strategy
  - community-reaction
people:
  - sama
---


<!-- buttondown-editor-mode: plaintext -->**GB10s may be all you need.**

> AI News for 1/6/2025-1/7/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **32** Discords (**218** channels, and **3342** messages) for you. Estimated reading time saved (at 200wpm): **365 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Happy 2hr Jensen keynote day.

https://www.youtube.com/watch?v=K4qQtPpSn-k

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

**Theme 1. NVIDIA Cosmos: Revolutionizing Robotics and Autonomous Systems**

- **[NVIDIA just unleashed Cosmos, a massive open-source video world model trained on 20 MILLION hours of video! This breakthrough in AI is set to revolutionize robotics, autonomous driving, and more.](https://v.redd.it/1j8k8iq7zibe1)** ([Score: 968, Comments: 141](https://reddit.com/r/OpenAI/comments/1hvmbcg/nvidia_just_unleashed_cosmos_a_massive_opensource/)): **NVIDIA** has released **Cosmos**, an open-source video world model trained on **20 million hours of video**. This model is expected to significantly impact fields like **robotics** and **autonomous driving**.
  - **Open Source Definition:** There is debate over whether **Cosmos** truly qualifies as open source, with some users noting it doesn't meet **OSI's** definition but is practically similar ([source](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)). Others question the authority of OSI to define open source standards.
  - **Technical Concerns and Impact:** Users are intrigued by the technical aspect of training a model on **20 million hours** of video to understand basic physics, questioning why existing physics models aren't used directly. The potential impact on industries like **manufacturing** and **autonomous driving** is noted, alongside concerns about job displacement.
  - **Community Reaction:** The release of **Cosmos** has sparked excitement and humor, with comments on the rapid pace of AI development and the symbolic significance of **NVIDIA's** CEO's attire upgrades. There's a general sense of anticipation and humor regarding the future implications of such advancements.


**Theme 2. Overwhelmed by AI Advancements: Navigating Uncertainty**

- **Anyone else feeling overwhelmed with recent AI news?** ([Score: 267, Comments: 193](https://reddit.com/r/OpenAI/comments/1hvjsm2/anyone_else_feeling_overwhelmed_with_recent_ai/)): The post expresses a sense of **overwhelm and anxiety** due to the frequent discussions about **AGI, ASI, and Singularity** from prominent figures like **Sama** and other **OpenAI members**. The author, a machine learning engineer, feels demotivated by the constant narrative of impending extreme changes and potential job loss, questioning how to plan for the future amidst such uncertainty.
  - Many commenters view the **hype around AGI/ASI** as a strategy to attract investment, with some expressing skepticism about the immediacy of such advancements. **Learninggamdev** and **FarTooLittleGravitas** argue it's about creating hype for funding, while **Houcemate** notes that the real audience for this hype is investors, not the general public.
  - **BrandonLang** and others suggest focusing on the present and controlling what you can, despite the overwhelming nature of the AI landscape. **Denvermuffcharmer** and **CGeorges89** recommend taking breaks from social media to gain clarity and emphasize that changes will integrate slowly, not overnight.
  - **Swagonflyyyy** highlights NVIDIA's upcoming release of a device for fine-tuning models at home, priced at **$3,000**, with related discussions on its potential impact on AI development. **ChymChymX** adds that NVIDIA is also working on a foundation model for AI robotics, showcasing the rapid advancements in AI technology.


---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. NVIDIA Digits: $3K AI Supercomputer Could Revolutionize Local AI**

- **[Nvidia announces $3,000 personal AI supercomputer called Digits](https://www.theverge.com/2025/1/6/24337530/nvidia-ces-digits-super-computer-ai)** ([Score: 1180, Comments: 298](https://reddit.com/r/LocalLLaMA/comments/1hvj4wn/nvidia_announces_3000_personal_ai_supercomputer/)): Nvidia introduced the **Digits**, a personal AI supercomputer priced at **$3,000**. This announcement highlights Nvidia's ongoing commitment to making advanced AI computing more accessible to individuals and smaller organizations.
  - **Specs and Performance Concerns**: Users are curious about the specifications, especially regarding **memory and bandwidth**. **LPDDR5X** is mentioned, with speculation about **memory controllers** and potential bottlenecks. Some users expect the device to be primarily for inference rather than training, comparing it to setups with multiple **3090/4090/5090 GPUs** in terms of cost and performance.
  - **Market Impact and Comparisons**: The **128GB unified RAM** is seen as a significant feature that could challenge **Apple's LLM market**. Comparisons are made with other hardware like the **5090**, with some users considering switching from cloud services like **Azure** to using this device locally due to potential cost savings and performance benefits.
  - **Availability and Pricing**: The device is priced starting at **$3,000**, with availability expected in **May**. Users discuss whether the pricing is competitive, with some suggesting that **Nvidia** could have priced it higher and still seen demand. There's also interest in how it compares to other options like **Strix Halo Solutions** and potential alternatives from **AMD**.


- **GB10 DIGITS will revolutionize local Llama** ([Score: 119, Comments: 66](https://reddit.com/r/LocalLLaMA/comments/1hvjjri/gb10_digits_will_revolutionize_local_llama/)): **GB10 DIGITS** is anticipated to significantly enhance local **Llama** applications, marking a pivotal development in local models over the past two years. The excitement is fueled by the potential accessibility of **NVIDIA's Grace Blackwell** technology, as outlined in the [NVIDIA news release](https://nvidianews.nvidia.com/news/nvidia-puts-grace-blackwell-on-every-desk-and-at-every-ai-developers-fingertips).
  - **Pricing and Specifications Concerns**: There are concerns about the **$3,000 starting price** and the potential cost scaling due to storage, not RAM, as each unit comes with **128GB of unified memory**. Some users believe the actual cost for the full specification could be higher, and there is skepticism about the **bandwidth capabilities** affecting performance, with comparisons to other GPUs like the **RTX5090**.
  - **Performance and Use Cases**: Discussions highlight that the **GB10 DIGITS** might be limited in performance due to **bandwidth constraints**, potentially affecting the **tokens per second** it can generate. While it can run large models, the **token generation speed** could be a bottleneck, making it less appealing for high-performance applications compared to cloud services or other GPUs.
  - **Market Position and Alternatives**: **NVIDIA's GB10** is seen as targeting the **prosumer market**, but there are debates about its value compared to alternatives like **AMD's AI Max** or potential future offerings from **Intel** and **Apple**. Users are considering the trade-offs between **price, performance, and memory bandwidth**, with some seeing it as a viable local AI solution while others question its practicality versus cloud solutions.


- **To understand the Project DIGITS desktop (128 GB for 3k), look at the existing Grace CPU systems** ([Score: 150, Comments: 73](https://reddit.com/r/LocalLLaMA/comments/1hvlbow/to_understand_the_project_digits_desktop_128_gb/)): Nvidia's **Project DIGITS desktop** is speculated to have **128 GB of VRAM** using LPDDR, which is cheaper and slower compared to GDDR and HBM typically used in GPUs. The **Grace-Hopper Superchip (GH200)** showcases a similar setup with 480 GB of LPDDR and 4.9 TB/s HBM bandwidth, while the **Grace CPU C1** configuration offers 120 GB of LPDDR RAM with 512 GB/s memory bandwidth. The Project DIGITS desktop is expected to achieve around **500 GB/s memory bandwidth**, potentially achieving ~7 tokens per second for **Llama-70B** at 8-bits.
  - Discussions highlighted the **potential use cases** of the Project DIGITS desktop, particularly for running local models like **Llama-70B**. Some commenters noted the device's limitations for large models due to its processing speed, while others found it suitable for inference tasks rather than training, with a focus on its **500 GB/s memory bandwidth**.
  - Commenters compared the Project DIGITS desktop with alternatives like **AMD EPYC Genoa** systems, highlighting the latter's higher RAM capacity and bandwidth but also noting the physical and noise constraints of larger setups. **EPYC Genoa** was suggested as a more cost-effective option for text inference, but some users valued the DIGITS desktop's compactness and potential for clustering with **ConnectX**.
  - The conversation touched on **low-bit arithmetic** and its impact on processing performance, with speculation that the DIGITS desktop could achieve **≥10 tokens per second** for 70B llama2 models at 4-bit quantization. The role of **ConnectX-8 interconnect** in enhancing connectivity and performance was noted, offering potential for home-based budget training setups.


**Theme 2. Fine-Tuning Success: 3B Model Excel in Math After Hugging Face Training**

- **Hugging Face continually pretrained Llama 3.2 3B to achieve 2-3x improvement on MATH** ([Score: 82, Comments: 20](https://reddit.com/r/LocalLLaMA/comments/1hv960u/hugging_face_continually_pretrained_llama_32_3b/)): **Hugging Face**'s SmolLM team achieved a **2-3x improvement** on MATH tasks by continually pre-training the **Llama 3.2 3B model** with **160B high-quality math tokens**. This enhancement resulted in a **2x higher score on GSM8K** and **3x higher on MATH**, with minimal performance drop on **MMLU-Pro** and no drop on **HellaSwag**. For more details, visit their [model](https://huggingface.co/HuggingFaceTB/FineMath-Llama-3B), [dataset](https://huggingface.co/datasets/HuggingFaceTB/finemath), and [training script](https://github.com/huggingface/smollm/tree/main/pre-training/continual-pretraining).
  - **Continual Pre-Training** involves extending the pre-training phase of a model with additional data, as explained by **mpasila**. This differs from fine-tuning by using a larger dataset, in this case, adding **160 billion tokens** to the existing **15 trillion** tokens for **Llama 3**.
  - The model's performance on **MMLU-Pro** did not improve, as noted by **Secure_Reflection409** and clarified by **r0kh0rd**, highlighting that the training was unsupervised without labels.
  - **EstarriolOfTheEast** raised concerns about the model's practical application beyond math tasks, questioning its effectiveness in instruction-following scenarios, which **DinoAmino** confirmed was not the focus of this training as the model was not instruction-tuned.


- **[Llama 3b - you can 2-3x the math capabilities just by continually training on high quality 160B tokens*](https://i.redd.it/t3kjugswufbe1.jpeg)** ([Score: 230, Comments: 31](https://reddit.com/r/LocalLLaMA/comments/1hv9w65/llama_3b_you_can_23x_the_math_capabilities_just/)): Pre-training **Llama 3.2-3B models** on high-quality **160 billion tokens** significantly enhances their math capabilities by 2-3 times without affecting other metrics. The performance improvements are quantified with specific increases: **+20.6%** on **GSM8K** and **+17.2%** on **MATH**, as depicted in a bar graph.
  - **Grokking in Machine Learning**: There is skepticism about the occurrence of grokking in this context, as it involves a neural network initially overfitting and then suddenly generalizing well after many epochs. It's noted that intentionally overfitting a well-performing model might not lead to better generalization, and continued pre-training on a large math dataset is expected to improve performance for small models.
  - **Training Data and Epochs**: Training on the same data for multiple epochs can yield good results, with 10x epochs being effective before degradation and 20-40x potentially burning the data. Concerns were raised about data leakage from **GSM8K** or **MATH** into the training dataset, with references to contamination reports and dataset sources on [Hugging Face](https://huggingface.co/datasets/HuggingFaceTB/finemath).
  - **Resource and Overfitting Concerns**: Some users argue that **160 billion tokens** might be excessive, with comments suggesting that overfitting is not a concern at this stage. Pretraining, as opposed to fine-tuning, requires significant VRAM, and the approach is defended as not compromising other metrics.


**Theme 3. Criticisms of RTX 5090 for AI Use: Balancing VRAM & Performance**

- **[RTX 5000 series official specs](https://i.redd.it/j0q0nd42rhbe1.png)** ([Score: 149, Comments: 62](https://reddit.com/r/LocalLLaMA/comments/1hvi9mi/rtx_5000_series_official_specs/)): The official specifications for the **RTX 5000 series** graphics cards, including the **RTX 5090, RTX 5080, RTX 5070 Ti, and RTX 5070**, are compared against the **RTX 4090** model. Key features highlighted include **NVIDIA Architecture, DLSS version, AI TOPS, Tensor Cores, Ray Tracing Cores**, and **Memory Configuration**.
  - Several commenters express dissatisfaction with the **VRAM capacity** of the new **RTX 5000 series**, noting that **32GB** is insufficient for running larger AI models. There is a call for increased VRAM to support more demanding tasks, with some suggesting that **24GB and 32GB** configurations would be more appropriate for the **RTX 5070** series.
  - **NVIDIA** is critiqued for its marketing strategy, with concerns about the lack of transparency regarding **core counts** and **AI TOPS** performance metrics. Some argue that the specifications are tailored for gamers rather than those interested in local AI model implementation, while others mention the difficulty of communicating comprehensive performance benchmarks.
  - Discussions highlight the perceived dominance of **NVIDIA's CUDA** in the AI industry, with **ROCm** cited as a less viable alternative, especially on Windows. There is mention of **Intel's AI playground** implementing **ComfyUI** and **Llama.cpp**, offering a potential alternative for Linux users.


- **[NVIDIA compares FP8 on 4090 to FP4 on 5090. Seems a little misleading](https://i.redd.it/aj6qbvpl4ibe1.jpeg)** ([Score: 340, Comments: 45](https://reddit.com/r/LocalLLaMA/comments/1hvjnar/nvidia_compares_fp8_on_4090_to_fp4_on_5090_seems/)): **NVIDIA** faces criticism for comparing **FP8** performance on the **RTX 4090** to **FP4** on the **RTX 5090**, which some find misleading. The comparison is visualized through a bar graph showing performance across several games, with metrics indicating potential discrepancies in the test settings and hardware used.
  - Discussions highlight the misleading nature of **NVIDIA's** performance comparisons, particularly the use of **FP8** on the **RTX 4090** versus **FP4** on the **RTX 5090**. Critics argue that the performance gains are largely due to software enhancements like **Multi-Frame Gen**, which artificially inflate performance metrics without significant hardware improvements.
  - Several commenters point out the questionable marketing tactics, noting that **FP4** sacrifices quality compared to **FP8**, and that **NVIDIA** has a history of exaggerating performance metrics. Additionally, **NVIDIA**'s marketing graphs are criticized for inconsistencies and potential oversights, such as font differences and lack of transparency regarding **AI TOPS** and **TFLOPS** figures.
  - There's skepticism about the actual compute improvements, with some suggesting that the **RTX 4090** may have intentionally limited cores to make room for a **Ti** version. Comparisons to past **NVIDIA** releases indicate that the performance jump might not be as substantial as advertised, with some users recommending waiting for price drops on current models.


**Theme 4. NVIDIA & AMD in THE AI Tech Race: Digits vs Strix Halo**

- **[HP Z2 Mini G1a is a workstation-class mini PC with AMD Strix Halo and up to 96GB graphics memory](https://liliputing.com/hp-z2-mini-g1a-is-a-workstation-class-mini-pc-with-amd-strix-halo-and-up-to-96gb-graphics-memory/)** ([Score: 83, Comments: 45](https://reddit.com/r/LocalLLaMA/comments/1hvqydy/hp_z2_mini_g1a_is_a_workstationclass_mini_pc_with/)): **HP** has introduced the **Z2 Mini G1a**, a workstation-class mini PC featuring the **AMD Strix Halo** with up to **96GB graphics memory**, positioning it as a competitor to new **NVIDIA** offerings.
  - The **HP Z2 Mini G1a** with **AMD Strix Halo** is notable for its **256GB/s memory bandwidth**, using **LPDDR5x-8000** with 4 memory channels. This configuration supports multiple smaller models or a single large model up to **70B parameters**. However, its **50 TOPS NPU** performance is limited compared to high-end GPUs like the **RTX 4090** with **1300 TOPS**.
  - Discussions highlight the **memory architecture** differences between AMD's traditional segmented model and Apple's unified memory architecture. Although AMD's **96GB graphics memory** allocation offers flexibility, it lacks the fully integrated access seen in Apple's systems, which could impact performance efficiency.
  - The **Z2 Mini G1a** is priced starting at **$1200** and presents a competitive option for local AI workstations. It is suitable for smaller quantized models and development, but it may not match the performance of high-end discrete GPUs for large model inference. The potential for **ROCm/DirectML** to support NPU acceleration could enhance its capabilities in the future.


- **[I made a CLI for improving prompts using a genetic algorithm](https://i.redd.it/p8q191zp2gbe1.gif)** ([Score: 97, Comments: 25](https://reddit.com/r/LocalLLaMA/comments/1hvayr2/i_made_a_cli_for_improving_prompts_using_a/)): The post introduces a **CLI tool** developed for enhancing prompts using a **genetic algorithm**. The accompanying GIF demonstrates the tool's operation on a **MacBook Pro** terminal, emphasizing its command-line interface functionality.
  - The **Promptimal** tool optimizes prompts without needing a dataset by using a **self-evaluation loop** or a custom evaluator. It employs a **genetic algorithm** to iteratively combine successful prompts and runs entirely in the terminal, making it user-friendly and accessible for experimentation.
  - The developer is considering improvements and is currently working on adding **ollama support** to enable the integration of local models. Users are encouraged to provide feedback as the tool remains experimental.
  - **FullstackSensei** suggests exploring alternatives like **Monte Carlo Tree Search (MCTS)** instead of a genetic algorithm, referencing tools like **optillm** as a potential option.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

Theme 1. NVIDIA Cosmos: Revolutionizing Robotics and Autonomous Systems**

- **[NVIDIA just unleashed Cosmos, a massive open-source video world model trained on 20 MILLION hours of video! This breakthrough in AI is set to revolutionize robotics, autonomous driving, and more.](https://v.redd.it/1j8k8iq7zibe1)** ([Score: 968, Comments: 141](https://reddit.com/r/OpenAI/comments/1hvmbcg/nvidia_just_unleashed_cosmos_a_massive_opensource/)): **NVIDIA** has released **Cosmos**, an open-source video world model trained on **20 million hours of video**. This model is expected to significantly impact fields like **robotics** and **autonomous driving**.
  - **Open Source Definition:** There is debate over whether **Cosmos** truly qualifies as open source, with some users noting it doesn't meet **OSI's** definition but is practically similar ([source](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)). Others question the authority of OSI to define open source standards.
  - **Technical Concerns and Impact:** Users are intrigued by the technical aspect of training a model on **20 million hours** of video to understand basic physics, questioning why existing physics models aren't used directly. The potential impact on industries like **manufacturing** and **autonomous driving** is noted, alongside concerns about job displacement.
  - **Community Reaction:** The release of **Cosmos** has sparked excitement and humor, with comments on the rapid pace of AI development and the symbolic significance of **NVIDIA's** CEO's attire upgrades. There's a general sense of anticipation and humor regarding the future implications of such advancements.


**Theme 2. Overwhelmed by AI Advancements: Navigating Uncertainty**

- **Anyone else feeling overwhelmed with recent AI news?** ([Score: 267, Comments: 193](https://reddit.com/r/OpenAI/comments/1hvjsm2/anyone_else_feeling_overwhelmed_with_recent_ai/)): The post expresses a sense of **overwhelm and anxiety** due to the frequent discussions about **AGI, ASI, and Singularity** from prominent figures like **Sama** and other **OpenAI members**. The author, a machine learning engineer, feels demotivated by the constant narrative of impending extreme changes and potential job loss, questioning how to plan for the future amidst such uncertainty.
  - Many commenters view the **hype around AGI/ASI** as a strategy to attract investment, with some expressing skepticism about the immediacy of such advancements. **Learninggamdev** and **FarTooLittleGravitas** argue it's about creating hype for funding, while **Houcemate** notes that the real audience for this hype is investors, not the general public.
  - **BrandonLang** and others suggest focusing on the present and controlling what you can, despite the overwhelming nature of the AI landscape. **Denvermuffcharmer** and **CGeorges89** recommend taking breaks from social media to gain clarity and emphasize that changes will integrate slowly, not overnight.
  - **Swagonflyyyy** highlights NVIDIA's upcoming release of a device for fine-tuning models at home, priced at **$3,000**, with related discussions on its potential impact on AI development. **ChymChymX** adds that NVIDIA is also working on a foundation model for AI robotics, showcasing the rapid advancements in AI technology.


---

# AI Discord Recap

> A summary of Summaries of Summaries by o1-2024-12-17

**Theme 1. GPU Hype and Infrastructure**

- [**NVIDIA’s ‘DIGITS’ Brings HPC to Your Desk**](https://www.nvidia.com/en-us/project-digits/): NVIDIA announced a $3,000 AI supercomputer with the new **Grace Blackwell** Superchip, claiming it can handle 200B-parameter models on a compact desktop box. Early adopters question real-world benchmarks, pointing to coverage like [The Verge article](https://www.theverge.com/2025/1/6/24337530/nvidia-ces-digits-super-computer-ai).  
- **AMD vs NVIDIA VRAM Duel**: Engineers debate AMD’s VRAM headroom vs. the RTX 4090’s ~95% GPU usage for big local LLMs. Some speculate on an RTX 5070 with “4090-level performance” at $549, doubting NVIDIA’s bold marketing.  
- **Speculative Decoding Races Ahead**: Recent updates to [llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/server.cpp) and others promise 25–60% faster LLM inference by drafting partial outputs. Early tests suggest minimal accuracy trade-offs, exciting devs to adopt the approach cross-platform.  

**Theme 2. Fine-Tuning and LoRA Adventures**

- [**LoRA Merging Wrangles Large Tokenizers**](https://github.com/unslothai/unsloth/issues/1518): Users saw bigger tokenizer files after fine-tuning with Unsloth’s LoRA, noting extra JSON files are needed for correct usage. Merging QLoRA back to a base model in FP16 is recommended to avoid performance drops.  
- **Deepspeed Zero-3 Disappoints**: Some found no memory gains when training 7B models with freezing, suspecting overhead from non-checkpointed gradients. Conversations stress that *“overlooked optimizer states”* hamper multi-GPU scaling.  
- [**Words or Concepts?**](https://arxiv.org/pdf/2412.08821): Heated debates push “ontological embeddings” over plain token fragments, claiming deeper semantic vector meaning. Advocates want to shift away from chunk-based embeddings to concept-based semantic representation.  

**Theme 3. Tools, Function Calling, and Agents**

- [**LM Studio 0.3.6 Releases Function Calling**](https://lmstudio.ai/download): The beta API supports local Qwen2VL and QVQ vision models plus in-app updates. Users praise the Windows installer’s new drive-selection feature and share a [Qwen2VL demo](https://cdn.discordapp.com/attachments/1111797717639901324/1325920413296885760/qwen2vl-demo-2.mp4).  
- [**Codeium vs DeepSeek for Enterprise**](https://marketplace.visualstudio.com/items?itemName=saoudrizwan.claude-dev): Some tout DeepSeek v3’s robust outputs if data issues get fixed, while Codeium remains popular for stable enterprise needs. Debates revolve around synergy vs. licensing headaches, with concerns about how each platform uses training data.  
- [**Multi-Agent Workflows Gain Steam**](https://t.co/eS4BhuAZVS): From NVIDIA’s multi-agent blueprint to community solutions using multiple LLMs, developers automate blog research and writing tasks. Early adopters applaud cross-agent synergy but demand more clarity on error handling and concurrency.  

**Theme 4. Payment and Privacy Dramas**

- [**AI21 Labs Token Sparks Scam Fears**](https://www.dextools.io/app/es/token/ai21labs?t=1736289130032&maker=81wGWkys3bHxktg8rXPGbW7xuJ8479RALZBBH55Jz54H): Community members label the “AI21 Labs Token” a rug-pull scam; AI21 publicly disowns it. Despite alleged audits, the project’s suspicious holder patterns spooked users into demanding an official Twitter statement.  
- **OpenRouter Payment Gateways Fizzle**: Virtual cards got declined repeatedly, forcing suggestions of crypto payments and alternative billing. [Issue #1157](https://github.com/cline/cline/issues/1157) documents related downtime, with some suspecting resource overload.  
- **Perplexity Brews Privacy Woes**: Targeted ads after health-related queries alarmed users about data sharing. They turned to the [Trust Center](https://trust.perplexity.ai/) for SOC 2 compliance details but still feel uneasy about potential user tracking.  

**Theme 5. MLOps, LLM Security, and What’s Next**

- [**MLOps & Feature Stores Webinar**](https://buff.ly/4j9oiVg): Ben Epstein and Simba Khadder will spotlight 2025’s MLOps trends on January 15 at 8 A.M. PT, covering best practices for data pipelines. They promise Q&A on real-world scaling, urging ML pros to keep pace with LLMOps advances.  
- [**GraySwanAI’s Harmful AI Assistant Challenge**](https://x.com/GraySwanAI/status/1872720375328411668): Launching January 4 at 1 PM EST with $40k in prizes for creative prompt injections. Multi-turn inputs are fair game, fueling competition to expose unsafe LLM behaviors.  
- [**Cerebras Calls for Bold AI Proposals**](https://cerebras.ai/blog/grantfrp): They invite research that pushes generative AI frontiers using their Wafer Scale Engine. Participants can leverage hardware grants to explore new training and inference techniques at scale.

---

# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth Troubleshooting & Tokenizer Woes**: After recent commits, users encountered GPU-specific errors with **Unsloth**, referencing [GitHub Issue #1518](https://github.com/unslothai/unsloth/issues/1518) and clarifying that larger tokenizer files from LoRA fine-tuning are expected.
   - Members suggested downgrading or updating specific library versions, reinforcing that the newly generated **added_tokens.json** must remain intact for proper usage.
- **LoRA Merging & Multi-Dataset Magic**: Community members emphasized merging LoRA with a base model in FP16 for Ollama, while pointing to [this Google Colab tutorial](https://colab.research.google.com/drive/1njCCbE1YVal9xC83hjdo2hiGItpY_D6t) on multi-dataset training.
   - They recommended consistent data formatting to avoid training mishaps, and warned that ignoring proper merging steps could compromise performance.
- **Hardware Hustle vs Cloud Convenience**: Engineers weighed using four 48GB DIMMs locally vs cloud-based solutions, citing [Unsloth AI’s tweet](https://x.com/UnslothAI/status/1876729710790815872) about 48GB RAM plus 250GB disk space for 2-bit quantization.
   - They acknowledged time spent on upload/download cycles in the cloud, but appreciated scalable options for running bigger models.
- **Gemini 1207’s Past-Prone Knowledge & Picotron Queries**: Some voiced frustration over **Gemini 1207**’s outdated knowledge cutoff, limiting help with modern libraries.
   - Others questioned the **Picotron codebase** for fine-tuning, seeking user experiences on its real-world efficacy.
- **Tokens vs Concepts: Ontological Embedding Push**: A heated exchange dissected the constraints of word-fragment embeddings and proposed **ontological ‘concepts’** for denser semantic vectors, referencing [this paper](https://arxiv.org/pdf/2412.08821).
   - Advocates claimed these conceptual embeddings could deliver deeper meaning, challenging the usual reliance on token-based approaches.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 0.3.6 Rolls Out Tools & Vision Models**: LM Studio released [version 0.3.6](https://lmstudio.ai/download) featuring a **Function Calling API** in beta and supporting **Qwen2VL** and **QVQ** for local inference, alongside a new Windows installer option.
   - The update adds **in-app updates** from 0.3.5 and showcases a [Qwen2VL demo](https://cdn.discordapp.com/attachments/1111797717639901324/1325920413296885760/qwen2vl-demo-2.mp4), drawing praise from early testers.
- **Speculative Decoding Accelerates LLMs**: A push for **Speculative Decoding** in [llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/server.cpp) suggests up to 60% faster parsing without hurting accuracy.
   - Contributors referenced [research](https://medium.com/ai-science/speculative-decoding-make-llm-inference-faster-c004501af120) explaining how draft models boost throughput, prompting enthusiasm for cross-platform rollouts.
- **NVIDIA Project DIGITS Targets 200B Model Loads**: **NVIDIA** revealed **Project DIGITS**, a compact AI system featuring 128GB of coherent memory, claiming the ability to handle 200B parameter models.
   - Developers admired the concept but noted that practical cost and benchmark data are still unknown, even though [NVIDIA’s site](https://www.nvidia.com/en-us/project-digits/) touts quicker development cycles.
- **AMD vs NVIDIA GPU Face-Off**: A heated comparison weighed AMD’s VRAM headroom against an **RTX 4090** pushing ~31 tokens/s for **Qwen2.5-Coder-32B-Instruct** at 95% GPU usage.
   - Participants speculated on a forthcoming GeForce **50** series, with some suggesting multi-GPU setups from both vendors to meet local LLM demands.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **DeepSeek vs Codeium in a Showdown**: Members weighed **DeepSeek v3** against **Codeium's** enterprise-friendly offerings, noting that **DeepSeek** could be a clear winner once data issues are resolved and if licensing questions are addressed. Some participants referenced **potential synergy** between these toolkits but expressed concerns about balancing model performance and **enterprise requirements**.
   - Several voices highlighted the robust **AI outputs** from *DeepSeek v3* and questioned how **Codeium** sources or manages its training data, sparking lively debate. Others argued that **Codeium** still stands out for its stable enterprise integration, while skeptics insisted that resolving *DeepSeek*'s data pipeline remains the key turning point.
- **Breezy Cline Extension Airlifts to VS Marketplace**: A new addition called **Cline (prev. Claude Dev)** surfaced on [Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=saoudrizwan.claude-dev), offering an autonomous coding agent integrated into the IDE. It garnered interest for enabling **file creation**, **editing**, and **command execution** all in one streamlined extension.
   - Users praised the convenience of this **all-in-one** approach, calling it *“a smooth ride for rapid prototyping.”* Meanwhile, some wanted more benchmarks around the **agent's** performance, noting that interest in advanced coding assistants continues to rise among AI-centric developers.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **NVIDIA's Nimble 'Digits' Debut**: NVIDIA introduced [Project DIGITS](https://www.nvidia.com/en-us/project-digits/) as a $3,000 personal AI supercomputer with the **GB10 Grace Blackwell** Superchip, capable of training models up to **200 billion parameters**.
   - It outperforms existing high-end GPUs and is aimed at local model prototyping, as described in [The Verge's coverage](https://www.theverge.com/2025/1/6/24337530/nvidia-ces-digits-super-computer-ai) with community feedback praising its practicality for advanced AI tasks.
- **Stable Diffusion's Slick Commercial Clause**: Stability AI allows commercial use of its **Stable Diffusion** models for annual revenues below **$1 million**, as outlined under the [Stability AI License](https://stability.ai/license).
   - Contributors noted confusion about license specifics, but the official [Stability AI Core Models](https://stability-ai.squarespace.com/core-models) documentation clarifies the terms for derivative works.
- **Speed vs. Sophistication in Image Generation**: Community members compared **Stable Diffusion 3.5** to **Flux** and found that 3.5 runs faster, but Flux yields more refined output.
   - Some recommended 3.5 for prototyping and then switching to Flux for final polishing, praising the synergy of these two approaches.
- **CFG Quirk Slows Flux**: Increasing **CFG scale** in Flux significantly ramps up processing times, raising inefficiency concerns during prompt tweaks.
   - Participants speculated Flux might be optimized for denoising rather than direct prompt expansions, emphasizing the trade-off between speed and quality.
- **NVIDIA's Cosmos for Physical AI**: The [NVIDIA Cosmos](https://github.com/NVIDIA/Cosmos) platform supports world foundation models, tokenizers, and a video pipeline for **Robotics** and **AV labs**.
   - It includes both diffusion and autoregressive modeling, and early adopters reported results on par with established systems.



---



## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **Bolt Exports Boost Workflows**: Members discovered how to **export Bolt projects** after each iteration, integrating them into other IDEs without friction.
   - They referenced a [Vite + React + TS example](https://stellular-beijinho-102b6b.netlify.app/) and suggested using `bolt.new/github.com/githubUsername/repoName` for **manual GitHub uploads**.
- **External LLMs Gobble Tokens**: Users reported **1.5 million tokens** consumed by a single prompt in smaller projects, driving concerns about runaway costs.
   - They suspected code inefficiencies and recommended offloading debugging to external LLMs to reduce overhead.
- **Supabase Chat Fails Real Time**: A few developers using **Supabase** for chat apps couldn't see new messages in real time.
   - They found passing the message in notifications might fix the UI shortfall, clarifying backend functionality wasn't at fault.
- **Bolt & GitHub Clash on Updates**: One user ran into deployment problems with GitHub to [Render.com](https://render.com/), forcing local fixes to Bolt-based projects.
   - They referenced [Issue #5108](https://github.com/stackblitz/bolt.new/issues/5108) for **backend server integration**, suggesting a forthcoming resolution.
- **Mobile Framework & Preview Snafus**: A soundboard project built with **NativeScript + Vue** triggered npm command errors, prompting alternative framework suggestions.
   - Another user struggled with blank screens in Bolt on a new laptop, hinting that direct GitHub usage versus project links might be the cause.



---



## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor’s Laggy Compositions**: Members reported that **Cursor IDE** slowed down and encountered frequent errors, particularly when the **Composer agent** attempted to handle larger codebases.
   - They described disappearing code, odd spacing, and unresponsive links, warning others to prepare backups while awaiting improvements.
- **Modular Musings with Code Chunks**: Some participants recommended splitting projects into 100-line files to help AI tools track changes more predictably.
   - Others countered that handling many small files complicates file discovery, creating confusion during multi-file edits.
- **A 'Project Brain' Extension Sparks Interest**: A user shared a [Reddit link](https://www.reddit.com/r/cursor/s/2BGt4BZ21e) about an extension that aims to give AI a better grasp of file relationships.
   - They hoped it would reduce confusion by offering a bird’s-eye view of dependencies, potentially improving AI-driven refactoring.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI Agents on Injection Edge**: Rumors suggest **OpenAI** delayed agent deployment over **prompt injection** worries, with talk of an enterprise plan near **$2K**.
   - Many in the community see this as a push for better support, hinting that **Agents** might still debut soon [more here](https://www.theinformation.com/articles/why-openai-is-taking-so-long-to-launch-agents).
- **01.AI Rumor Rebuttal Rallies**: **Kai-Fu Lee** from **01.AI** refuted gossip about the startup selling teams to Alibaba, citing strong 2024 revenue beyond **$14 million** [source](https://technode.com/2025/01/07/01-ai-refutes-rumors-of-selling-teams-to-alibaba/).
   - Yet the firm reportedly laid off key pre-training teams, leading many to question how they’ll balance future growth.
- **Anthropic’s Mega-Funding Maneuver**: **Anthropic** secured **$2B** at a hefty **$60B** valuation, with **$875 million** in expected ARR.
   - This bold move underscores fierce **B2B** rivalry as watchers gauge how quickly they can scale.
- **Nvidia’s Digits Debuts on Desktop**: **Nvidia** announced **Project Digits** at **$3,000**, featuring the **Grace Blackwell** Superchip for handling models up to **200 billion** parameters [link](https://www.theverge.com/2025/1/6/24337530/nvidia-ces-digits-super-computer-ai).
   - Engineers raised concerns about **ARM CPU** compatibility, given limited open-source support.
- **MeCo Method Springs Metadata Magic**: The **MeCo** approach, outlined in [this paper](https://arxiv.org/abs/2501.01), prepends source URLs to training docs for simpler **LM pre-training**.
   - Critics called it *ridiculous* initially, yet they acknowledged metadata can boost a model's contextual depth.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Deepspeed’s Dilemma: Memory Gains Gone Missing**: One user tried **Deepspeed Zero-3** to slash memory usage during 7B LLM training but found no major benefits, suspecting overhead from missing **gradient checkpointing**.
   - Community members concluded that *overlooked optimizer states* plus high-precision copies hamper memory usage, fueling more interest in **gradient checkpointing**.
- **Pythia’s Ethical Check: Does It Compute?**: The conversation soared around evaluating **Pythia** on the [Ethics dataset](https://huggingface.co/datasets/hendrycks/ethics), revealing a push for testing moral complexity.
   - Many expressed curiosity about **Pythia**'s performance and how these tasks might shape future model alignment efforts.
- **Cerebras Calls for Creative AI**: **Cerebras** issued a [Request for Proposals](https://cerebras.ai/blog/grantfrp) to turbocharge **Generative AI research** via their Wafer Scale Engine, seeking bold submissions.
   - They aim to highlight the performance advantage of their hardware and spur novel approaches to **inference and training**.
- **Chitchat Format Flops on MCQs**: Trials with **chat templates** saw multiple-choice scores dip, with **L3 8B base** doing better in a plain format.
   - Logprob analysis suggested chat framing deters precise letter-only answers, prompting calls for constrained output styles.
- **Llama2’s Fate in GPT-NeoX: Stuck at the Gate?**: **Llama2** checkpoint users asked if NeoX-trained weights convert smoothly to Hugging Face format but received no firm confirmation.
   - Differing optimizer setups (AdamW vs **Lion**) and BF16 scaling complications added to the uncertainty around direct checkpoint portability.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Payment Predicament**: Users reported repeated declines and issues with **OpenRouter**’s payment gateway, prompting speculation about virtual cards.
   - Some suggested transitioning to **crypto** transactions, particularly seeking user-friendly wallets for global convenience.
- **Hermes 405b Slips and Stalls**: Frequent crashes plagued **Lambda’s Hermes 405b**, even though the status indicator still glowed green.
   - High demand led participants to suspect resource pressure, with some pointing to **DeepSeek V3** as another lagging service.
- **DeepSeek V3 Doubles Down on Downtime**: Multiple users flagged **DeepSeek V3** reliability troubles, especially under large inputs.
   - They referenced [Issue #1157](https://github.com/cline/cline/issues/1157) as evidence of attempts to diagnose the indefinite loading glitch.
- **Crypto Conundrum Gains Traction**: Calls for a **crypto** alternative grew louder, with users noting better convenience in some regions such as the Philippines.
   - They mentioned **Trust Wallet** and similar platforms as possible solutions, citing fewer transaction failures.
- **LLM Game Dev Hits a Ceiling**: Users recognized **LLMs** like O3 and GPT-5 could handle simpler 2D games, but more complex designs remained elusive.
   - They agreed that advanced organizational logic hampers fully automated complex **game development**, especially for large-scale projects.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider’s Utility as a Pro-Level Coding Companion**: Multiple members applauded **Aider** for handling complex code tasks, referencing [images & web pages usage docs](https://aider.chat/docs/usage/images-urls.html#web-pages) for advanced project integration.
   - They likened it to a **coding mentor**, highlighting how strategic prompts and /ask commands refine results for more accurate outputs.
- **Continue.dev Co-Pilots with Aider**: Some members tested **Continue.dev** alongside **Aider**, finding them complementary for faster iteration and better task management.
   - They shared that combining both tools eases bigger coding workloads and keeps development more organized, with planned expansions to unify their workflows.
- **Custom LLM Magic with Aider**: Developers explored hooking up custom language models via 'custom/' name prefixes and [advanced model settings](https://aider.chat/docs/config/adv-model-settings.html), enabling specialized ML pipelines.
   - They reported smoother integration by properly registering model classes and adjusting API parameters to match their setups.
- **LLM Interviews for Structured Specs**: A shared approach uses an **LLM** to interview the user for specification creation before coding, as shown in a [YouTube video](https://www.youtube.com/watch?v=XWJGm3y207A).
   - This tactic ensures more organized planning, feeding directly into **Aider’s** coding prompts for better clarity.



---



## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **AI Sportscasting: A Slam Dunk for Recaps**: One user showcased how NotebookLM overlays sports recaps with highlights, referencing [this demonstration](https://youtu.be/D7qZ2VphetU) for the NBA and NFL.
   - They praised the approach’s **cost-effectiveness**, pointing out that real-time coverage and branded content can be automated at scale.
- **Citation Conundrum in Single-Source Debates**: Members debated the reliability of **Britannica** vs. **Wikipedia**, focusing on whether to reference multiple sources or rely on a single one.
   - They pursued a robust system prompt strategy to preserve factual accuracy and ensure precise quoting in AI-generated material.
- **Contract Review Gains AI Allies**: Users explored AI for **contract redlining**, emphasizing speed and cost reduction in tedious legal edits.
   - They highlighted a potential integration of virtual paralegals with avatar-based collaboration, better aligning stakeholder involvement in the negotiation process.
- **NotebookLM Slows Under Heavy Use**: Concerns surfaced about daily usage caps, with NotebookLM becoming slow after extended sessions, prompting references to the [support page](https://support.google.com/notebooklm/answer/15678219?hl=en).
   - Some users also struggled with **audio overview** length management and noted missing question-suggestion features, seeking clarity on current product updates.
- **NotebookLM Plus Features Shine Amid License Queries**: Subscribers praised **NotebookLM Plus** for supporting multiple PDFs and YouTube links, generating refined summaries and expanded usage quotas.
   - Google Workspace license requirements emerged as a hot topic, prompting users to consult the [Admin Help page](https://support.google.com/a/answer/6043385?hl=en&co=DASHER._Family%3DBusiness) for add-on details.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous Wraps Up Forge API Beta**: The beta for the [Nous Forge API](https://forge.nousresearch.com/) concluded recently, enabling advanced reasoning across multiple models like Hermes, Claude, Gemini, and OpenAI. Potential subscribers can still follow updates for new configurations that clarify usage and performance details.
   - Debates surfaced over user subscription models that might appear **profit-oriented**, intensifying scrutiny around how organizations treat user trust.
- **NVIDIA's Digits Gains Ground**: The new [NVIDIA Project DIGITS](https://nvidianews.nvidia.com/news/nvidia-puts-grace-blackwell-on-every-desk-and-at-every-ai-developers-fingertips) introduced Grace Blackwell Superchip for broader high-performance AI computing. Meanwhile, heated arguments erupted about the 5070’s rumored '4090-level performance' at $549.
   - Skeptics questioned whether NVIDIA’s marketing matched real benchmarks, pointing to [tweets citing inflated claims](https://x.com/meeix_/status/1876465349278994804). Others remain hopeful that DIGITS will reduce barriers to top-tier AI hardware.
- **Tweak That Talk: AI Behavior Boosts**: Some members shared system prompts to reduce anxious or uncertain model responses, suggesting more confident generative output. People joked about accidental confessions in AI logs, a side effect of incomplete tuning strategies.
   - USB-C took the spotlight as a cost-conscious networking link at 10-20Gbps, though the group warned about cable compatibility and potential limits in large-scale usage.
- **Privacy vs Profit Showdown**: A user pointed out that certain AI organizations lack a reputation for protecting **privacy**, feeding doubts about corporate intentions. This triggered discussions on whether **profit motives** inevitably overshadow user safeguards.
   - Others alleged that profit-first thinking fosters distrust, offering cautionary tales of security shortcuts to meet revenue goals.
- **MiniMind & Neural Embeddings Magic**: A blog post examined latent space geometry, referencing the [Manifold Hypothesis](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/) and hierarchical features in neural networks. Further reading included visualizations from [Colah's deep learning series](https://colah.github.io/posts/2015-01-Visualizing-Representations/) to clarify hidden representations.
   - The [MiniMind Project](https://github.com/jingyaogong/minimind/blob/master/README_en.md) presented a 26.88M-parameter LLM that can be pre-trained, SFT-ed, and DPO-ed within a few hours on 2x RTX3090s. Enthusiasts welcomed it for accessible code, quick training, and expansions into mixed-expert and multi-modal models.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pains and Model Mayhem**: Multiple users reported **Perplexity** slow response times and conflicting **Pro Searches** quotas, leading some to rely on copy-paste tricks for smoother queries.
   - They also debated a December 19 mail suggesting *Huge bummer if they just keep the online models!*, indicating fears over potential model exclusivity.
- **Privacy Perils and SOC 2 Pressures**: Users voiced alarm over targeted ads following health-related searches on **Perplexity**, questioning how user data might be shared and stored.
   - Some turned to the [Trust Center | Powered by Drata](https://trust.perplexity.ai/) for **SOC 2 compliance** info but remained uncertain about privacy protections.
- **NASA's Nimble Moon Micro-Mission**: Today, **NASA** showcased its **Moon Micro-Mission** aimed at refining lunar exploration, with details provided [here](https://www.perplexity.ai/search/the-amethyst-tablet-pdf-wSFKHIw3R9CFnbPRI3Vfxw#0).
   - Enthusiasts highlighted how these cutting-edge modules could reshape operational complexities for future manned missions.
- **AgiBot Advances Humanoid Dataset**: **AgiBot** revealed a new **humanoid robot training dataset**, outlined in [this video](https://www.youtube.com/embed/V8A6EdbPdGU), promising greater realism in robotic motion.
   - Community members anticipate better synergy between AI algorithms and physical controls, opening the door for more advanced task handling.
- **Microsoft's Mighty $100B AGI Bet**: **Microsoft** slapped down a bold **$100 billion** commitment to **AGI development**, as noted [here](https://www.perplexity.ai/search/why-is-it-called-donating-plas-8jLeogYsRZW4.OYo.eObhA).
   - Observers speculated this massive funding could reshape the AI landscape, with both excitement and concern over how it might challenge competing platforms.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **AI21 Token Turmoil**: Members suspected the **AI21 Labs Token** is a **scam**, citing questionable activities and urging others to *stay away*, referencing [DEXTools](https://www.dextools.io/app/es/token/ai21labs?t=1736289130032&maker=81wGWkys3bHxktg8rXPGbW7xuJ8479RALZBBH55Jz54H).
   - Users highlighted the token's suspicious holder distribution and alleged that it may have already **rugged**.
- **Community Craves Clarity**: Many demanded an official statement from **AI21 Labs** on **Twitter**, insisting a direct warning would help dismiss any perceived affiliation with the token.
   - Some expressed frustration, saying *it doesn't cost anything to tweet a warning*, emphasizing how strongly they wanted the company to intervene.
- **Security Team Steps In**: **AI21 Labs** staff declared the token **unaffiliated** with the company and warned of possible **bans** for prolonged crypto discourse.
   - They escalated the scam concerns to their **security team**, who questioned the token's audit claims and ties to **pumpfun**.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Mini O1 Throws Down with GPT-4**: In #gpt-4-discussions, participants debated whether **Mini O1** truly outsmarts **GPT-4**; one user claimed it surpassed the bigger model in select tasks.
   - Others argued it isn't a universal champion, with someone saying *'it excels in specialized domains but not across the board.'*
- **RTX 5000 Flaunts DLSS 4 Gains**: In #ai-discussions, members hyped **RTX 5000** featuring **DLSS 4** upgrades that promise triple-frame generation improvements.
   - They highlighted prospective boosts for gaming and graphics, calling it **a big leap** for GPU-based AI workloads.
- **Fine-Tuning LLaMA in the Wild**: In #ai-discussions, a user confirmed success fine-tuning **LLaMA** on personal text logs, calling it *'simpler than expected.'*
   - Others chimed in about structured data methods, describing clear performance gains once everything was properly arranged.
- **Schema Slip-Ups Frustrate Prompt Engineers**: In #prompt-engineering and #api-discussions, users reported the model returning the JSON schema itself **80%** of the time instead of valid data.
   - They tried multiple retries and adjustments, suspecting that vague instructions and large prompts fueled the persistent confusion.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Science Embraces Foundation Models**: A member shared the [Metagene 1 paper](https://metagene.ai/metagene-1-paper.pdf) to highlight the use of **foundation models** in scientific research, fueling curiosity about data sourcing and domain-specific performance.
   - Participants asked about potential expansions to related fields, sparking **hopes** for new collaborations between AI and specialized sciences.
- **NVIDIA's Cosmos Captivates AI Circles**: NVIDIA introduced [Cosmos](https://x.com/DrJimFan/status/1876516972512559170), an open-source video world model trained on **20M** hours of footage, featuring both diffusion and autoregressive generation.
   - Community members praised Cosmos for propelling **video-based synthetic data** forward, raising questions about scalability and broader enterprise applications.
- **Vercel's AI SDK Earns Mixed Feedback**: A user praised **Vercel's AI SDK** for quick setup but criticized its *too much abstraction* when layering multiple models.
   - Others debated the SDK’s trade-off between **user-friendly** scaffolding and developer control, spotlighting performance overhead concerns.
- **AI Powers Whale Tracking**: Collaborators at [Accenture and the University of Sydney](https://x.com/btibor91/status/1876630816199217208) used AI with **89.4%** accuracy to detect minke whales, compressing a two-week manual process into near real-time analysis.
   - Community members applauded the system's efficiency gains and drew parallels to other wildlife monitoring opportunities.
- **FP4 Format Fuels GPU Performance Debate**: NVIDIA’s emphasis on **FP4** metrics raised questions about fair comparisons to **FP8** and other floating-point formats.
   - Enthusiasts pushed for clearer **benchmark standards**, warning that insufficient definitions could mislead developers evaluating next-generation GPUs.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Thin Font Sparks Concern**: Community members criticized the **Modular docs** for having a font weight that is too slim, flagging potential readability issues.
   - They urged **Modular** to consider thicker or alternative font choices for a better user experience.
- **Mojo Debugger Taps LLDB**: Participants highlighted that **Mojo** uses an LLDB approach with upstream patches, referencing a [talk from the LLVM conference](https://www.youtube.com/watch?v=9jfukpjCPIg).
   - They praised **Modular** for not reinventing solutions, underlining how it accommodates multi-language debugging effectively.
- **Project Structures Under the Spotlight**: One user asked about managing imports and showed a [GitHub example](https://github.com/saviorand/lightbug_http/blob/main/tests/lightbug_http/test_client.mojo) for **Mojo** projects.
   - Another member shared the command `magic run mojo test -I . tests` and directed everyone to the [Mojo testing docs](https://docs.modular.com/mojo/tools/testing#writing-unit-tests).
- **Static Lists and Borrow Checker Dreams**: A user realized that **ListLiteral** can’t be indexed with runtime variables, opting for **InlineArray** instead.
   - Someone proposed outpacing Rust’s borrow checker through extended static analysis, though they prefer finalizing existing features first.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command R+ Conquers Complexity**: On the **Cohere** Discord, participants praised **Command R+08** for advanced reasoning in complex question tasks, surpassing others like **Sonnet 3.5**.
   - They noted that simpler inquiries reduced its effectiveness, emphasizing question complexity for peak performance.
- **Embed That Image with Cohere**: A snippet showcased [base64-encoded image input](https://docs.cohere.com/reference/embed) for **cohere.ClientV2** embedding calls, confirming returning embeddings in the same request order.
   - They focused on correct content-type headers alongside base64 transformations to ensure consistent embedding results.
- **JavaScript Brainchild: Neural Network Request**: One user asked for a **pure JavaScript** implementation of a neural network, entirely from scratch.
   - The conversation ended without specific code or further instruction, leaving the question open for future exploration.
- **AR & Cohere Combine for Plane Detection**: A user pursued an **AR** project aimed at detecting planes and classifying objects, seeking synergy with Cohere for real-time asset ranking.
   - Another contributor called it *'totally sick to see'*, reflecting the desire for more AR-based tooling in collaboration with Cohere's technology.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton's Terrific Turn on Expand_dims vs. Reshape**: Discussions highlight that **expand_dims** shows significantly different performance than **.reshape** in Triton, especially around dimension reorder capabilities. The community also weighed in on **autotuning** strategies like `CLOSEST_M` and usage of **wgmma** on **H100** for better MMA performance.
   - They debated kernel recompilation trade-offs for large sizes and how to ensure **PTX** uses **wgmma** instead of **mma.sync**. The conversation indicated potential config issues for maximizing HPC features.
- **CUDA's WMMA Wizardry Preserves Matrix Layout**: Participants confirmed that **WMMA** loading from matrix A and storing to matrix B retains the same register layout, with indices like **[0,1][0,2]** intact. Testing suggests that output fragments hold the input arrangement, effectively copying matrices as proven by multiple experiments.
   - They offered to share a runnable example, noting they've since moved on from deeper **WMMA** explorations. However, they remain open to showing how these hardware-level intrinsics handle data.
- **PyTorch Perplexities: Custom Autograd & Guard Logs**: **Modified in-place gradients** in a custom `autograd` function, despite PyTorch docs cautioning against it, matched the simpler reference models’ results. They linked the [PyTorch docs on extending autograd](https://pytorch.org/docs/main/notes/extending.html#how-to-use) for further context.
   - Another question arose about getting verbose logs on **guard failures**, with a user’s logs only yielding a cryptic **0/0** message. They used `TORCH_LOGS="+dynamo,guards,bytecode,recompiles,recompiles_verbose"` but found the output lacking in details.
- **Picotron & DeepSeek: A Double Dose of 4D Fun**: The [Picotron framework](https://github.com/huggingface/picotron) offers a **4D-parallelism** approach to distributed training for educational purposes, showcasing user-friendly exploration of advanced AI training tactics. Meanwhile, short videos covered *Pages 12-18 of the DeepSeek-v3 paper* ([arXiv link](https://arxiv.org/abs/2412.19437)) to clarify **LLM infrastructure** concepts.
   - A recommended [YouTube playlist](https://www.youtube.com/watch?v=76gulNlhiE4&list=PLO45-80-XKkT6BUKCYeBMTEqnlcpYavxq&index=1) further explained the paper’s complexities. This was aimed at AI enthusiasts seeking to ingest dense references more easily.
- **DIGITS & Discord: New Tools for GPU Greatness**: [Project DIGITS](https://www.nvidia.com/en-us/project-digits/) by **Nvidia** pairs the **Grace Blackwell Superchip** with up to an alleged **200B parameter** capacity and **128GB** unified memory in a compact, high-performance form factor. This hardware touts new **tensor cores** supporting fp4 and f8 modes for future training expansions.
   - Simultaneously, a newly announced **Discord-based GPU leaderboard** invites alpha testers to measure performance across specific kernels. A `gpu-glossary.zip` release also compiles references for GPU fundamentals in a single package.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex & MLflow: A Data-Driven Duo**: A step-by-step guide details how to combine **LlamaIndex**, **MLflow**, **Qdrant**, and **Ollama** for vector storage and model tracking, referencing [the full guide](https://t.co/lNDDgdOo86). The guide highlights using **Change Data Capture** to streamline real-time evaluations.
   - Community members praised the synergy for effectively bridging experiment tracking and embedded knowledge, noting simpler orchestration between **LlamaIndex** and backend services.
- **NVIDIA AI Supercharges Multi-Agent Blogging**: A fresh blueprint leverages **NVIDIA AI** to handle multi-agent tasks like blog research and writing, revealed at CES with an official announcement [here](https://t.co/eS4BhuAZVS). The approach aims to free teams from the **time sink** of content creation using LLM-powered research.
   - It synchronizes multiple agents to perform complex tasks in real-time, keeping workflow friction minimal for content generation.
- **Cohere's Crisp Integration with LlamaIndex**: Developers applauded **Cohere**'s embeddings and improved documentation for seamless usage with **LlamaIndex**. They highlighted installation instructions and prerequisites in the [documentation](https://t.co/dLKGgkqOe8), ensuring smooth collaboration.
   - This combined setup broadens the range of indexing and retrieval operations, giving engineers tighter control over their text-processing pipelines.
- **LlamParse's First-Run Mystery**: A user encountered an unexpected error parsing a PDF file with **LlamParse**, though every subsequent attempt worked without issue. Project contributors plan to inspect whether the glitch recurs consistently or was a one-time quirk.
   - They requested more details about the PDF in question, hoping to diagnose possible format or encoding conflicts behind the scenes.
- **Text-to-SQL Takes Center Stage**: **LlamaIndex** outlines structured data parsing and **text-to-SQL** capabilities for powering queries on unstructured sources, as described in [Structured Data docs](https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/structured_data) and [SQLIndexDemo](https://docs.llamaindex.ai/en/stable/examples/index_structs/struct_indices/SQLIndexDemo/). A working [notebook demo](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/index_structs/struct_indices/SQLIndexDemo.ipynb) addresses concerns about broken links in the official docs.
   - The guide intently warns against blindly executing arbitrary queries, urging best practices and security reviews for safe SQL usage.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter 1.0: The Code That Won't Run**: At [this GitHub commit](https://github.com/OpenInterpreter/open-interpreter/commit/21babb186f13e263a72cf525d15d79788edf4644), devs teased **Open Interpreter 1.0** but removed code-running capabilities, causing user confusion.
   - They offered no clear roadmap, leaving contributors unsure when or how these features might get restored.
- **Classic OI Drifts into Archives**: The older **Open Interpreter** was archived at [this commit](https://github.com/OpenInterpreter/open-interpreter/commit/275105771dfa985ee777e4a9c6e9c4d760c7b7b9), stashing outdated prompts in read-only folders.
   - PRs for the classic version are effectively locked, forcing developers to shift attention to the 1.0 branch.
- **Pip Installation Blues**: Folks reported that `pip install open-interpreter` fails to yield a stable build, hampering usage.
   - They encountered partial functionality and confusion about how to fix or enhance the current setup without breaking more components.
- **Tweaks That Trip Folks Up**: Community members hoped to refine prompts and add new features, but the shift to 1.0 complicated merging older modifications.
   - Contributors lament the backlog of unmerged PRs, as the upcoming version remains undecided on final structure.
- **Local Models: Use --no-tool-calling**: Users recommended the `--no-tool-calling` flag to improve performance on smaller local models and dodge overhead.
   - They fear new system prompt changes in 1.0 could reduce local model accuracy, prompting further discussion.



---



## [Axolotl AI](https://discord.com/channels/1104757954588196865) Discord

- **GH200 & Compilation Quirks**: A user recognized that **GH200** is in use and offered potential support, while others noted extended **compilation times** caused by layered dependencies, emphasizing the burdens of setting everything up from scratch.
   - They hoped that pooling experiences would reduce friction for new adopters, possibly speeding up GPU-based tinkering on advanced boards.
- **Discord Link Guy Strikes Again**: The notorious **Discord Link Guy** reappeared, posting suspicious links that prompted swift **warnings** and a subsequent ban.
   - A user confirmed the ban and removal of a bizarre welcome channel message that had caused confusion.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **MiPROv2 Trials One Instruction at a Time**: A suggestion was to feed instructions to **MiPROv2** in single steps, refining them with an LLM's output critiques.
   - This approach aims to yield real-time improvements in generated instructions, using a judge-like method for feedback.
- **dspy.COPRO! Sparks Curiosity**: Members saw parallels between MiPROv2's approach and **dspy.COPRO!**, prompting further exploration.
   - They suggested synergy in refining instructions via iterative trials, bridging MiPROv2 and dspy concepts.
- **dspy & LangChain Merge Hits Snags**: One user tried combining **dspy** with **LangChain** (version 2.6) to build LLM agents but faced difficulties.
   - A follow-up noted no easy path to unify these frameworks, highlighting friction in reconciling their designs.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Certificate Portal Pops Back Open**: The **Certificate Declaration form** was reopened for participants who completed assignments in December, and it must be submitted by the **end of January** for certification eligibility.
   - Organizers reemphasized the **one certificate** policy and warned that no past assignments would be reopened, urging everyone to finish all tasks on time.
- **Email Mismatch Mayhem**: Multiple users stressed that the **email address** in the declaration form must match the one used for course assignments to avoid errors.
   - One participant asked for confirmation after using a new email but listing their original in the form, highlighting the risk of delays in **certificate issuance** if details are mismatched.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Reasoner v1 Rolls Forward & Gains Traction**: A member praised **Reasoner v1** on [GPT4All](https://github.com/nomic-ai/gpt4all) and asked about other reasoning-ready models like Qwen 2.5 coder.
   - Another user confirmed that **OpenAI-compatible** remote models and several local models can run in reasoning mode, adding that more expansions are in progress.
- **LocalDocs Indexing Leaves Files Sitting Idle**: A user encountered subdirectory embedding issues with **LocalDocs**, noting timestamps might cause some files to remain unembedded.
   - They explained that once a document is indexed under one timestamp, subsequent additions could be skipped by the system.
- **Embedding Model Mashups Spark Curiosity**: Someone asked about swapping the default embedder with **text-embedding-inference** or **vLLM** to improve indexing tasks.
   - They highlighted the desire for flexible embeddings to handle custom data pipelines more efficiently.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **MLOps & Feature Stores Showdown**: On January 15th at 8 A.M. PT, Ben Epstein and Simba Khadder will host a [webinar](https://buff.ly/4j9oiVg) to spotlight **MLOps** and **Feature Stores** for 2025.
   - They will cover best methods and host a Q&A for **Data Engineers** and **ML** pros seeking deeper knowledge of future MLOps approaches.
- **2024 MLOps Trends Eye 2025**: Speakers plan to highlight major **MLOps** developments in 2024 and a forward look at 2025, placing emphasis on **LLMs** in real-world pipelines.
   - They anticipate synergy between standard MLOps and **LLMOps**, urging participants to consider more integrated model deployment and scaling strategies.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **GraySwanAI's $40k Gambit for LLM Security**: The **Harmful AI Assistant Challenge** kicks off on **January 4th at 1 PM EST**, offering **$40,000** in prizes for innovative **prompt injection** and **jailbreaking** methods, as shown in [this tweet](https://x.com/GraySwanAI/status/1872720375328411668).
   - Multi-turn inputs are allowed, and participants can register at [app.grayswan.ai](http://app.grayswan.ai/arena) or join via [Discord](http://discord.gg/WqHkWt99) to deepen **LLM security testing** skills.
- **OAI Pre-release Tests & Community Engagement**: Earlier GraySwanAI events spotlighted **o1 models** before they officially launched, referencing the **12/5 OAI paper** for context.
   - This track record of pre-release insights demonstrates energized momentum in **LLM security** and underscores community enthusiasm.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Common Voice AMA 2025 Gains Momentum**: Common Voice announced their **2025 AMA** in a [new Discord server](https://discord.gg/b4c83ppxdU), inviting participants to reflect on the past year's milestones and preview upcoming developments.
   - This session aims to tackle any questions regarding the project's direction, featuring direct insights from the core team and **expanded data collection** plans.
- **2024 Review & Q&A Bring Key Voices**: A **2024 review** event will feature the Product Director and a Frontend Engineer sharing top updates on Common Voice's progress and next steps.
   - Attendees can bring technical and strategic questions to this **live Q&A**, aiming to shape the project's near-future trajectory.
- **Accessibility Focus in Voice Tech**: Common Voice is dedicated to making **voice technology** more open and accessible, offering a dataset that can fuel speech recognition systems for multiple languages.
   - They emphasize lowering existing barriers by democratizing **voice data**, enabling developers to serve broader communities with locally relevant solutions.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Dolphin 3.0 rides BFCL curiosity**: A member asked if **Dolphin 3.0** from Cognitive Computations will appear on the BFCL leaderboard, pointing to [Dolphin 3.0 on Hugging Face](https://huggingface.co/collections/cognitivecomputations/dolphin-30-677ab47f73d7ff66743979a3).
   - They showed excitement over the model’s potential performance, speculating it could stand out among existing contenders.
- **Cognitive Computations' recent Dolphin 3.0 boost**: The **cognitivecomputations/Dolphin3.0-Llama3.2-1B** model update gained 34 stars on Hugging Face, and sparked 14 comments.
   - An attached image showcased the model’s build and drew interest in its technical details and real-world benchmarks.



---


The **tinygrad (George Hotz) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Torchtune Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **HuggingFace Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1325916950869835889)** (687 messages🔥🔥🔥): 

> `Unsloth updates and troubleshooting, Tokenization issues with trained LORA adapters, Fine-tuning Llama 3.2, Hardware and memory considerations for AI processing, Using cloud resources for large models` 


- **Unsloth updates and troubleshooting**: Users encountered issues with Unsloth after recent commits, particularly related to loading models on RTX3090 while being successful on T4 GPUs.
   - Several users shared troubleshooting steps, including rolling back and updating specific versions of the Unsloth libraries to resolve these errors.
- **Tokenization issues with trained LORA adapters**: A user inquired about the increased size of the `tokenizer.json` after saving a trained LORA adapter, noting the presence of an `added_tokens.json` file.
   - It was confirmed that the additional files are necessary and do not indicate a bug, and users were advised to retain them along with the original `tokenizer.json`.
- **Fine-tuning Llama 3.2**: A user sought clarity on the required dataset format for fine-tuning Llama 3.2, particularly regarding conversion to a format with ('role', 'content') keys.
   - Examples provided included issues with JSONL formatted files, highlighting the need for proper structure to avoid errors during training.
- **Hardware and memory considerations for AI processing**: Discussions among users centered on the challenges of maintaining stability with high memory configurations, such as using four 48GB DIMMs.
   - It was noted that having more RAM allows for efficiency in data processing and being able to run larger models without the hassle of cloud upload/download cycles.
- **Using cloud resources for large models**: Users shared their experiences balancing local resources versus cloud services for running larger models and data processing tasks.
   - While cloud access provides flexibility, concerns about upload/download times highlight the continued reliance on local hardware for efficient workflows.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/UnslothAI/status/1876729710790815872">Tweet from Unsloth AI (@UnslothAI)</a>: Deepseek V3, including GGUF + bf16 versions are now on @HuggingFace!Min. requirements to run: 48GB RAM + 250GB of disk space for 2-bit.Includes 2, 3, 4, 5, 6 and 8-bit quantized versions.See all versi...</li><li><a href="https://x.com/danielhanchen/status/1876465802951721011">Tweet from Daniel Han (@danielhanchen)</a>: NVIDIA RTX 5090 has 4000 AI TOPS - 3x more than RTX 4090 (1300 FP8 with sparsity)RTX 5090    $1,999 3,400 AI TOPSRTX 5080       $999 1,800 AI TOPSRTX 5070 Ti   $749 1,400 AI TOPSRTX 5070       $549 1,...</li><li><a href="https://research.google.com/colaboratory/local-runtimes.html">
      Google Colab
    </a>: no description found</li><li><a href="https://nvidianews.nvidia.com/news/nvidia-puts-grace-blackwell-on-every-desk-and-at-every-ai-developers-fingertips">NVIDIA Puts Grace Blackwell on Every Desk and at Every AI Developer’s Fingertips</a>: CES—NVIDIA today unveiled NVIDIA® Project DIGITS, a personal AI supercomputer that provides AI researchers, data scientists and students worldwide with access to the power of the NVIDIA Grace ...</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: Below is a list of all our notebooks:</li><li><a href="https://unsloth.ai/blog">Blog</a>: no description found</li><li><a href="https://unsloth.ai/introducing">Introducing Unsloth</a>: no description found</li><li><a href="https://discuss.huggingface.co/t/do-i-need-to-dequantization-before-merging-the-qlora/110175">Do I need to dequantization before merging the qlora</a>: In this DPO trainer link  It says   As suggested by [Benjamin Marie](https://medium.com/@bnjmn_marie/dont-merge-your-lora-adapter-into-a-4-bit-llm-65b6da287997), the best option for merging QLoRA adap...</li><li><a href="https://huggingface.co/docs/datasets/loading">Load</a>: no description found</li><li><a href="https://blog.gopenai.com/unsloth-unleashing-the-speed-of-large-language-model-fine-tuning-986ae7040711">Unsloth: Unleashing the Speed of Large Language Model Fine-Tuning</a>: Large language models (LLMs) have revolutionized the field of artificial intelligence, demonstrating remarkable capabilities in tasks like…</li><li><a href="https://blog.gopenai.com/unsloth-unleashing-the-speed-of-large-language-m">no title found</a>: no description found</li><li><a href="https://huggingface.co/unsloth/DeepSeek-V3-GGUF">unsloth/DeepSeek-V3-GGUF · Hugging Face</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements)">Unsloth Documentation</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1gwyuyg/beware_of_broken_tokenizers_learned_of_this_while/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/issues/1518">[BUG] Unsloth stopped working after todays commits · Issue #1518 · unslothai/unsloth</a>: Hi. I can&#39;t use Unsloth anymore on my RTX3090. It works only on Nvidia T4 on colab. When I try to download any model - I have this: ----------------------------------------------------------------...</li><li><a href="https://github.com/axolotl-ai-cloud/axolotl/issues/1263">[FR] Quantize + dequantize base model before merging LoRA · Issue #1263 · axolotl-ai-cloud/axolotl</a>: ⚠️ Please check that this feature request hasn&#39;t been suggested before. I searched previous Ideas in Discussions didn&#39;t find any similar feature requests. I searched previous Issues didn&#39;t...</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/issues/1089">Unsloth QLora merging into base-model: what is the best practice if you want to run trained model with vLLM or NVIDIA TensorRT-LLM ?  · Issue #1089 · unslothai/unsloth</a>: Just a quick (and important) question about LoRA vs QLoRA with Unsloth. I have read through a series of articles about DO NOT MERGED naievely QLoRA back to base model, it will give worse performanc...</li><li><a href="https://github.com/unslothai/notebooks">GitHub - unslothai/notebooks: Unsloth Fine-tuning Notebooks for Google Colab, Kaggle, Hugging Face and more.</a>: Unsloth Fine-tuning Notebooks for Google Colab, Kaggle, Hugging Face and more. - unslothai/notebooks</li><li><a href="https://github.com/unslothai/unsloth/issues/195">Does merge to 16 bit merge with a dequantized base model? · Issue #195 · unslothai/unsloth</a>: The colab notebooks show: # Merge to 16bit if False: model.save_pretrained_merged(&quot;model&quot;, tokenizer, save_method = &quot;merged_16bit&quot;,) if False: model.push_to_hub_merged(&quot;hf/mod...</li><li><a href="https://github.com/unslothai/unsloth/issues/195,">Issues · unslothai/unsloth</a>: Finetune Llama 3.3, Mistral, Phi, Qwen 2.5 &amp; Gemma LLMs 2-5x faster with 70% less memory - Issues · unslothai/unsloth</li><li><a href="https://github.com/jondurbin/qlora/blob/main/qmerge.py#L42">qlora/qmerge.py at main · jondurbin/qlora</a>: QLoRA: Efficient Finetuning of Quantized LLMs. Contribute to jondurbin/qlora development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: Finetune Llama 3.3, Mistral, Phi, Qwen 2.5 &amp; Gemma LLMs 2-5x faster with 70% less memory</a>: Finetune Llama 3.3, Mistral, Phi, Qwen 2.5 &amp; Gemma LLMs 2-5x faster with 70% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/pull/1520">Update __init__.py by sebaxakerhtc · Pull Request #1520 · unslothai/unsloth</a>: This PR is solving the issue with some GPUs
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1326109933347016814)** (2 messages): 

> `Gemini 1207 knowledge cutoff, Picotron codebase for fine tuning` 


- **Gemini 1207's outdated knowledge hampers coding**: A member pointed out that **Gemini 1207** has a very old knowledge cutoff, making it inadequate for providing support on the latest libraries.
   - This limitation is causing frustrations among users relying on up-to-date coding assistance.
- **Inquiry on Picotron codebase efficacy**: There was a question raised about how effective the **Picotron codebase** is when it comes to fine tuning.
   - Members are curious if anyone has insights or experiences regarding its capabilities for fine-tuning processes.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1326005131053105224)** (26 messages🔥): 

> `Merging LoRA adapters, Multiple datasets for finetuning, Deploying LLaMA models, Multi-GPU training support` 


- **Manual merging of base model and LoRA adapter for Ollama**: To merge a base model and LoRA adapter for use in Ollama, members advised saving the LoRA, merging it in FP16, and following regular conversion steps using provided wrappers or pipeline scripts.
   - This approach helps avoid relying on `model.save_pretrained_gguf()`, which may not be feasible in certain environments.
- **Finetuning with multiple datasets**: Yes, it's possible to finetune with multiple datasets, but it is recommended to combine them in a consistent format for better results.
   - A helpful [tutorial specifically for multiple datasets](https://colab.research.google.com/drive/1njCCbE1YVal9xC83hjdo2hiGItpY_D6t?usp=sharing) was shared for reference.
- **Deploying fine-tuned models in Flutter applications**: For deploying a fine-tuned LLaMA model in applications like Flutter, it's essential to have access to a powerful GPU, unless using cloud solutions.
   - Utilizing services like Hugging Face Spaces for public models, or hosting on platforms like together.ai, can provide options for running models easily.
- **API access solutions for free**: Members suggested using Ollama combined with OpenWebUI or Flowise for free API access to deployed models.
   - These tools can facilitate chat interfaces and integrations on websites effortlessly.
- **Current state of multi-GPU training with Unsloth**: Currently, Unsloth does not support multi-GPU training, and commercial support is expected in the future.
   - This limitation has been confirmed by community discussions and documentation browsing.



**Link mentioned**: <a href="https://colab.research.google.com/drive/1njCCbE1YVal9xC83hjdo2hiGItpY_D6t?usp=sharing">Google Colab</a>: no description found

  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1326044073840083025)** (1 messages): 

> `Token Embeddings, Ontological Concepts, Semantic Meaning` 


- **Debate on Token Embeddings Limitations**: Discussion centered around concerns regarding current **token embeddings** based on word fragments, with an emphasis on their limitations in capturing semantic richness.
   - One member asserted, *'current token embeddings are limited,'* advocating for a shift towards ontological 'concepts' for richer individual vector embeddings.
- **Push for Foundational Ontological Concepts**: A member has been advocating the exploration of **foundational ontological concepts** to enhance semantic embedding strategies, proposing they offer deeper meaning than mere word fragments.
   - They shared their perspective that these concepts could lead to significantly more informative embeddings, challenging the existing paradigms.
- **Review of Relevant Paper on Semantic Meaning**: A relevant paper, linked [here](https://arxiv.org/pdf/2412.08821), sparked discussions about the implications of alternative embedding techniques that address the limitations of current models.
   - Participants were encouraged to review the findings, as they align with the push for embeddings deriving from ontological perspectives.


  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1325920412890304654)** (1 messages): 

> `LM Studio 0.3.6 Release, Function Calling API, Qwen2VL and QVQ Support, New Installer Features, In-App Updates` 


- **LM Studio 0.3.6 launches with exciting features!**: The new version, [0.3.6](https://lmstudio.ai/download), introduces a **Function Calling / Tool Use API**, making it compatible with existing OpenAI tools while supporting local models.
   - The update is currently in beta, and users are encouraged to provide [bug reports](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues) and feedback.
- **Meet the new vision-input models: Qwen2VL and QVQ!**: Version **0.3.6** now supports the **Qwen2VL** family and **QVQ** models in both LM Studio's **MLX** and **llama.cpp** engines.
   - These models enhance capability with advanced vision and reasoning functionalities for more powerful applications.
- **Installer gains new features for Windows users!**: The new installer allows users to choose their install drive, a feature users have eagerly awaited.
   - The update process is now more efficient with smaller updates and a progress bar for user convenience.
- **In-app updates improve user experience!**: In-app updates from **0.3.5** stable will begin later this week, transitioning to a new updater system.
   - Users can update their **llama.cpp** and **MLX** engines without needing a full application update.
- **Demo showcasing Qwen2VL model impresses users!**: A demo featuring the **Qwen2VL** **2B** model highlights its capabilities, wowing the community.
   - Users can watch the demo linked in the announcement to see the model in action. [Watch the demo](https://cdn.discordapp.com/attachments/1111797717639901324/1325920413296885760/qwen2vl-demo-2.mp4?ex=677edc9c&is=677d8b1c&hm=aaf1d2dac50e73b9429009946a925816baaf6c8a49e440b49fd6610e57f7b79e&).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/download">Download LM Studio - Mac, Linux, Windows</a>: Discover, download, and run local LLMs</li><li><a href="https://lmstudio.ai/blog/lmstudio-v0.3.6">LM Studio 0.3.6</a>: Tool Calling API in beta, new installer / updater system, and support for `Qwen2VL` and `QVQ` (both GGUF and MLX)
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1325917009342500894)** (201 messages🔥🔥): 

> `LM Studio in AMD presentation, Function calling API updates, Model loading issues with Qwen-VL, Performance benchmarks with 4090 GPU, Feedback on new UI design` 


- **Surprise at LM Studio in AMD Presentation**: A user expressed surprise at seeing **LM Studio** featured during the **AMD presentation**.
   - The unexpected inclusion sparked interest in the capabilities of LM Studio.
- **Discussion on Function Calling API**: Users discussed the new **function calling API**, with one asking for clarification on changes compared to previous versions.
   - It was noted that this version expands functionality but required a special beta sign-up.
- **Challenges with Loading Qwen-VL Models**: Several users experienced problems loading the **Qwen-VL** model, particularly related to context length and exit codes.
   - The issue was found to be specific to Linux with certain functionality remaining broken during testing.
- **Performance Benchmarks on RTX 4090**: One user reported receiving ~**31 Tokens per second** with the **Qwen2.5-Coder-32B-Instruct** model on the **RTX 4090**.
   - The GPU utilization was noted to be about 95% at **385 Watts**, showcasing its effective performance.
- **Feedback Regarding New UI Design**: A user criticized the new UI, stating they preferred the old design, citing issues like light mode and button logic changes.
   - Responses included instructions on switching back to the classic UI and an acknowledgment of the feedback.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Barnacules/status/1873631114864292330">Tweet from Barnacules Nerdgasm (@Barnacules)</a>: The @NVIDIAGeForce RTX 4090 is such a beast of a GPU! 🏆 I&#39;m getting ~31 Tokens per seconds in LM Studio with Qwen2.5-Coder-32B-Instruct 18GB loaded into VRAM. It only draws about 350watt TDP at 1...</li><li><a href="https://tenor.com/view/surprised-pikachu-pokemon-shock-surprised-pikachu-gif-15357817">Surprised Pikachu GIF - Surprised Pikachu Pokemon - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=dRQUkq_HzEE">Facebook Made Creepy AI Profiles...</a>: Hello guys and gals, it&#39;s me Mutahar again! Facebook revealed it&#39;s plans to turn their site into a mixture of humans and artificial intelligence bots. With s...</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/285">(Exit code 133) Error when loading large LLM models · Issue #285 · lmstudio-ai/lmstudio-bug-tracker</a>: When loading large LLMs (for example, Meta-Llama-3.1-70B-Instruct-IQ2_S with context window 32768), I would encounter the error (Exit code: 133). Please check settings and try loading the model aga...</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge">GitHub - lllyasviel/stable-diffusion-webui-forge</a>: Contribute to lllyasviel/stable-diffusion-webui-forge development by creating an account on GitHub.</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/285.">Issues · lmstudio-ai/lmstudio-bug-tracker</a>: Bug tracking for the LM Studio desktop application - Issues · lmstudio-ai/lmstudio-bug-tracker</li><li><a href="https://lmstudio.ai/docs/advanced/tool-use">Tool Use - Advanced | LM Studio Docs</a>: Enable LLMs to interact with external functions and APIs.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1325917201638756475)** (227 messages🔥🔥): 

> `NVIDIA Project DIGITS, Speculative Decoding, AI model performance, AMD vs NVIDIA GPUs, Local LLM inference` 


- **NVIDIA Project DIGITS revolutionizes AI computing**: NVIDIA unveiled Project DIGITS, a compact AI supercomputer with the ability to run 200B parameter models, featuring 128GB of coherent memory and substantial performance improvement over traditional setups.
   - Developers can prototype and deploy large AI models efficiently, but the pricing and performance of this technology remain to be fully assessed in practical settings.
- **Speculative Decoding boosts LLM inference speed**: Recent discussions highlighted the introduction of Speculative Decoding in llama.cpp, promising 25-60% improvements in inference speed for large language models without losing accuracy.
   - While various channels await the integration of this feature in different platforms, it is noted that draft models can significantly enhance performance.
- **Comparative performance: AMD vs NVIDIA**: Comparisons between AMD and NVIDIA GPUs for AI applications brought up concerns about VRAM limitations, with insights on the upcoming GeForce 50 series, particularly the 5090 model.
   - Despite skepticism about NVIDIA’s cloud solutions, there's acknowledgment of the potential benefits of mixed hardware setups that leverage both companies' capabilities.
- **Innovative setup for AI inference tests**: Users explored setting up AI inference by combining higher VRAM models with smaller ones to boost efficiency and maintain quality, particularly on mid-range hardware like the 3090 and 3060 GPUs.
   - Notably, insights were shared on running experiments with alternative configurations, allowing for meaningful gains in terms of inference speed.
- **Local model performance and resource constraints**: Conversations highlighted the challenges of running large AI models on limited hardware resources, as some users reported successful setups using older machines with considerably less VRAM.
   - Despite limitations, users are finding innovative ways to manage performance requirements, demonstrating the ongoing adaptations in the landscape of AI development.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://medium.com/ai-science/speculative-decoding-make-llm-inference-faster-c004501af120">Speculative Decoding — Make LLM Inference Faster</a>: Improve LLM inference speed by 2–3X without degrading any accuracy</li><li><a href="https://www.nvidia.com/en-us/geforce/special-event/">NVIDIA GeForce Special Events at CES 2025 </a>: Tune in to the opening keynote by NVIDIA CEO Jensen Huang.</li><li><a href="https://www.gigabyte.com/Graphics-Card/GV-N5090AORUSX-WB-32GD/sp#sp">AORUS GeForce RTX™ 5090 XTREME WATERFORCE WB 32G Specification | Graphics Card - GIGABYTE Global</a>: no description found</li><li><a href="https://nvidianews.nvidia.com/news/nvidia-puts-grace-blackwell-on-every-desk-and-at-every-ai-developers-fingertips">NVIDIA Puts Grace Blackwell on Every Desk and at Every AI Developer’s Fingertips</a>: CES—NVIDIA today unveiled NVIDIA® Project DIGITS, a personal AI supercomputer that provides AI researchers, data scientists and students worldwide with access to the power of the NVIDIA Grace ...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1hqlug2/revisting_llamacpp_speculative_decoding_w/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1gzm93o/speculative_decoding_just_landed_in_llamacpps/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.nvidia.com/en-us/project-digits/">NVIDIA Project DIGITS: The World’s Smallest AI Supercomputer. </a>: Reserve yours today.</li><li><a href="https://www.youtube.com/watch?v=dQ8gSV_KyDw"> - YouTube</a>: no description found</li><li><a href="https://github.com/exo-explore/exo">GitHub - exo-explore/exo: Run your own AI cluster at home with everyday devices 📱💻 🖥️⌚</a>: Run your own AI cluster at home with everyday devices 📱💻 🖥️⌚ - exo-explore/exo</li><li><a href="https://github.com/ollama/ollama/pull/8134#issuecomment-2550018120">feat: Introduce speculative decoding by bfroemel · Pull Request #8134 · ollama/ollama</a>: This PR aims to replicate speculative decoding as implemented in https://github.com/ggerganov/llama.cpp/blob/master/examples/server/server.cpp.See hints in the documentation (docs/faq.md) for tryi...
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1325917307435876362)** (71 messages🔥🔥): 

> `DeepSeek vs Codeium Models, Codeium Subscription Issues, Codeium Chat Functionality, AI Model Support and Testing, User Concerns about Windsurf Performance` 


- **Debate on DeepSeek vs Codeium Models**: Members discussed whether to prefer **DeepSeek v3** over **Codeium's models**, indicating that if data issues were resolved, it would be a clear choice.
   - *DeepSeek v3* was noted to have roots in powerful AI outputs, raising concerns over Codeium's approach to enterprise licensing.
- **Subscription Issues with Codeium**: Users reported issues with Codeium subscriptions not reflecting upgrades, with inquiries about payment retries and support response times.
   - Members reassured one another that customer support might take longer due to recent holidays and high demand.
- **Functionality and Errors in Codeium Chat**: Several users expressed frustrations with Codeium Chat not connecting properly, citing error messages and needing to manually refresh.
   - Concerns were raised about the inconsistency of the **o1-preview**, with some users suggesting alternative methods to input data.
- **Support for Various AI Models**: A discussion on why other AI models aren't supported highlighted the thorough testing process Codeium undertakes before releasing new features.
   - Members noted there are numerous capable models available but emphasized the need for careful evaluation by Codeium's team.
- **User Frustrations with Windsurf Performance**: One user expressed frustration over Windsurf's performance after an upgrade, with it only analyzing a minimal number of code lines.
   - Concerns were raised about how the changes affected code stability and completeness, especially for premium users.



**Link mentioned**: <a href="https://marketplace.visualstudio.com/items?itemName=saoudrizwan.claude-dev">Cline&#32;(prev.&#32;Claude&#32;Dev)&#32;-&#32;Visual&#32;Studio&#32;Marketplace</a>: Extension&#32;for&#32;Visual&#32;Studio&#32;Code&#32;-&#32;Autonomous&#32;coding&#32;agent&#32;right&#32;in&#32;your&#32;IDE,&#32;capable&#32;of&#32;creating/editing&#32;files,&#32;running&#32;command...

  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1325926934206615616)** (242 messages🔥🔥): 

> `Windsurf Errors, Cascade Autocomplete Issues, User Experience Feedback, Internal Server Errors, Feature Requests and Suggestions` 


- **Windsurf Errors on Editing**: Numerous users reported encountering the error message 'ErrorCascade has encountered an internal error in this step' while editing files in Windsurf, indicating ongoing stability issues.
   - Darmitage noted that despite these issues, Windsurf cleverly attempts to work around them by creating .new files and replacing the originals afterward.
- **Slow Autocomplete Functionality**: Many users, including Sayokurisu and a_a_garc, experienced slow autocomplete performance, creating frustration among the community.
   - This slowdown was prevalent across the board, leading to discussions about potential underlying issues affecting the system.
- **Visibility of Errors During Workflow**: Several users reported facing repeated occurrences of the 'HTTP status 503 Service Temporarily Unavailable' error, impeding their workflow significantly.
   - This contributed to a shared sense of concern regarding service availability and reliability within the Windsurf application.
- **User Experience and Feedback**: Users expressed their feelings about Windsurf's current functionality, with some appreciating the clever handling of errors while others criticized the performance.
   - Conversations highlighted the need for better communication from the development team regarding ongoing issues and fixes.
- **Future Improvements and Support**: There was a call for additional support and transparency from the Windsurf team, with users wanting clarity on future updates and improvements.
   - Suggestions for enhancing user collaboration features and reducing the frequency of errors were also brought up during discussions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: Need help? Contact our support team for personalized assistance.</li><li><a href="https://tenor.com/view/multiversx-x-xportal-egld-crypto-gif-4249062898891695021">Multiversx Xportal GIF - Multiversx X Xportal - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.reddit.com/r/ChatGPTCoding">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1325927130558627883)** (268 messages🔥🔥): 

> `Project DIGITS, Stable Diffusion Licensing, Image Generation Quality, Flux Generation Times, NVIDIA Cosmos` 


- **NVIDIA's Project DIGITS Launch**: NVIDIA announced Project DIGITS, a personal AI supercomputer priced at $3,000, capable of handling AI models with up to **200 billion parameters** and powered by the new **GB10 Grace Blackwell** Superchip.
   - It's designed for developers to prototype large AI models locally and is expected to perform better than existing high-end GPUs.
- **Commercial Use of Stable Diffusion Models**: Models from Stability AI can be used commercially as long as annual revenue is below **$1 million**, allowing individuals to create and distribute derivatives without permission.
   - Discussion highlighted the confusion surrounding the terms, but it's noted that using AI content for commercial purposes requires adherence to licensing agreements.
- **Image Generation Quality Comparison**: Users compared the quality of image output between **Stable Diffusion 3.5** and **Flux**, noting that while 3.5 is speedier, Flux provides better image quality and refinement capabilities.
   - Some users suggested using 3.5 for initial prototypes followed by Flux for detailed refinement.
- **Impact of CFG Scale on Flux Generation Times**: A user noted that increasing the **CFG scale** in Flux generation resulted in significantly longer image processing times, indicating a potential inefficiency when the prompt is altered.
   - Concerns were raised about whether Flux was optimized for denoising rather than effective use of prompts.
- **NVIDIA Cosmos Overview**: The NVIDIA Cosmos platform has been introduced as a development tool for world models, offering various model types including **diffusion and autoregressive** models, targeted for physical AI applications.
   - Users praised its capabilities, noting performance levels comparable to other high-quality models in the market.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.runpod.io/runpod-slashes-gpu-prices-powering-your-ai-applications-for-less/">RunPod Slashes GPU Prices: Powering Your AI Applications for Less</a>: RunPod is dropping prices across our Serverless and Secure Cloud services. Why? Because we believe in giving you the firepower you need to build applications without breaking the bank.The Lowdown on O...</li><li><a href="https://www.theverge.com/2025/1/6/24337530/nvidia-ces-digits-super-computer-ai">Nvidia announces $3,000 personal AI supercomputer called Digits</a>: It’s the size of a desktop.</li><li><a href="https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/">NVIDIA Open Models License</a>: no description found</li><li><a href="https://stability-ai.squarespace.com/core-models">Stability AI Core Models &mdash; Stability AI</a>: The Core Models are available to Professional and Enterprise Members for commercial use under the terms of their Membership Agreement.</li><li><a href="https://stability.ai/license">Stability AI License &mdash; Stability AI</a>: Stability AI licenses offer flexibility for your generative AI needs by combining our range of state-of-the-art open models with self-hosting benefits.</li><li><a href="https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/">NVIDIA Blackwell Architecture</a>: Catapulting generative AI to trillion-parameter scale.</li><li><a href="https://github.com/NVIDIA/Cosmos">GitHub - NVIDIA/Cosmos: Cosmos is a world model development platform that consists of world foundation models, tokenizers and video processing pipeline to accelerate the development of Physical AI at Robotics &amp; AV labs.  Cosmos is purpose built for physical AI. The Cosmos repository will enable end users to run the Cosmos models, run inference scripts and generate videos.</a>: Cosmos is a world model development platform that consists of world foundation models, tokenizers and video processing pipeline to accelerate the development of Physical AI at Robotics &amp;amp; AV la...</li><li><a href="https://medium.com/@promptingpixels/can-stable-diffusion-models-be-used-for-commercial-use-it-depends-eedd89272245>">no title found</a>: no description found</li><li><a href="https://www.nvidia.com/en-us/project-digits/">NVIDIA Project DIGITS: The World’s Smallest AI Supercomputer. </a>: Reserve yours today.
</li>
</ul>

</div>
  

---


### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1325919641352011956)** (9 messages🔥): 

> `Exporting Bolt projects, Using external LLMs, Manually uploading projects` 


- **Efficiently Exporting Bolt Projects**: One member mentioned the ability to **export Bolt projects** after each iteration, seamlessly integrating into workflows.
   - *Using external LLMs* allows for debugging and tweaking without wasting tokens on minor adjustments.
- **IDE Compatibility for Bolt Projects**: Another member confirmed that while you can export projects to other IDEs, some **adjustments** are required to run them properly.
   - This indicates the need for users to familiarize themselves with settings specific to their chosen IDE.
- **Manual Project Uploads**: Inquiring about adding existing projects to Bolt, one member learned that they can upload their local project via a **public GitHub repo**.
   - To do this, they need to use the format: `bolt.new/github.com/githubUsername/repoName` to pull it into Bolt.



**Link mentioned**: <a href="https://stellular-beijinho-102b6b.netlify.app/">Vite + React + TS</a>: no description found

  

---


### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1325926075032670229)** (258 messages🔥🔥): 

> `Token Consumption Concerns, Chat App Development with Supabase, Bolt and GitHub Integration Issues, Framework Selection for Mobile Apps, Account Migration and Preview Issues` 


- **Concerns over Token Consumption**: Members expressed frustration regarding high token consumption, with reports of a single prompt using over **1.5 million tokens** even in smaller projects.
   - Some suggested that inefficiencies in code changes led to unexpected token usage, raising concerns about managing costs effectively.
- **Challenges in Chat App Development**: While developing a chat app using **Supabase**, one member encountered issues with one-on-one chats not displaying new messages in real time.
   - It was suggested that passing the message within notifications might resolve the problem, indicating a UI issue rather than a backend failure.
- **Issues with GitHub and Bolt Integration**: A member reported challenges with deploying updates through GitHub to **Render.com**, leading to manual interventions.
   - The community recommended submitting a GitHub issue for better tracking and potential future fixes.
- **Framework Selection for Mobile App Development**: One user discussed difficulties with **NativeScript + Vue**, receiving npm command errors while building a soundboard app.
   - Members suggested exploring other frameworks or troubleshooting existing setups to avoid encountering repeated errors.
- **Account and Preview Display Issues**: A user on a new laptop experienced issues with Bolt, unable to preview existing projects or new ones, resulting in blank screens.
   - Another member inquired about the importance of working off specific project links versus directly from GitHub directories as a potential solution.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://repocloud.io/boltdiy">RepoCloud | Bolt.diy: Choose Your AI Model</a>: Discover Bolt.diy, the ultimate fork for selecting your favorite AI model. Customize your coding experience with top LLMs like OpenAI and Anthropic!</li><li><a href="https://github.com/stackblitz/bolt.new/issues/2529">Bolt Outputs Application Logic in Chat · Issue #2529 · stackblitz/bolt.new</a>: Issue: Bolt outputs application logic in the chat. For example, when the user hits a rate limit, the code to offer a link to upgrade is sent as a response to the user in chat.</li><li><a href="https://github.com/stackblitz/bolt.new/issues/1809">Feature Request: Upload image files to Bolt · Issue #1809 · stackblitz/bolt.new</a>: Currently, image files cannot be uploaded to Bolt. A solution is in progress. In the interim, you can: Open your Bolt project in StackBlitz (click &quot;Open in StackBlitz&quot; at the top right of th...</li><li><a href="https://github.com/stackblitz/bolt.new/issues/5108">Backend Server integration · Issue #5108 · stackblitz/bolt.new</a>: Currently having to manually update my stackblitz connected to a github repo as bolt.new doesn&#39;t update that one. My render.com server is connected to my github repo and only updates when the gith...
</li>
</ul>

</div>
  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1325918688787824692)** (191 messages🔥🔥): 

> `Cursor IDE performance issues, Modularity in code structure, AI behavior in coding tasks, Cursor extension for project understanding, Issues with Composer agent` 


- **Cursor IDE experiences performance issues**: Users reported that **Cursor IDE** experiences lag and errors, particularly with the **Composer agent**, affecting their workflow and causing frustration.
   - Issues included files not being found, unwanted spaces being added in code, and challenges with saved context after multiple prompts.
- **Advocacy for modular code structure**: Several members advocated for smaller, modular files around 100 lines to help AI tools manage code better and avoid technical debt.
   - However, some expressed concerns that while modularity can prevent code loss, it complicates file management for AI, leading to difficulties in code discovery.
- **Persistent AI behavior challenges**: Users shared challenges with AI models making incorrect or destructive changes, leading to lost code and the need for constant recontextualization.
   - Discussions highlighted the necessity for careful instruction and memory management when working with AI in coding environments.
- **Interest in a Cursor extension for project awareness**: A user shared a **Reddit link** about a project extension designed to improve AI's understanding of a codebase, potentially enhancing its performance.
   - The extension aims to create a 'project brain' to help AI better track file relationships and understand coding patterns.
- **Encountering Composer agent problems**: Users noted frequent issues with the **Composer agent**, such as it being unresponsive after a limited number of exchanges and errors when clicking on file links.
   - It was suggested to use keyboard shortcuts or manual methods for file navigation to overcome the interface limitations.



**Link mentioned**: <a href="https://www.reddit.com/r/cursor/s/2BGt4BZ21e">Reddit - Dive into anything</a>: no description found

  

---


### **Interconnects (Nathan Lambert) ▷ #[events](https://discord.com/channels/1179127597926469703/1179127598442348729/1326020961568559205)** (3 messages): 

> `Embarcadero Meetup, Meeting Schedule, Shack15 Location` 


- **Embarcadero Meetup Planning**: A member mentioned they will be in **Embarcadero** and have some available time on **Wednesday** and **Thursday**.
   - Another member proposed to meet on **Thursday** morning at **Shack15**.
- **Thursday Meeting Confirmation**: The plan for the meetup is confirmed for **Thursday** at **Shack15**.
   - *Let's finalize the morning time* for the meeting details.


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1326221362032803851)** (38 messages🔥): 

> `OpenAI AI agents launch, Devin valuation and support, 01.AI startup updates, Anthropic funding, Competition in AI` 


- **OpenAI's Hesitation on AI Agents**: One reason cited for OpenAI delaying the launch of its AI agents is a concern over **prompt injection attacks**; however, reports suggest that the software could come out this month [more here](https://www.theinformation.com/articles/why-openai-is-taking-so-long-to-launch-agents).
   - Speculation about pricing hints at potential costs ranging from **$2K** for enterprises, raising concerns about necessary improvements in support.
- **Devin and Market Dynamics**: Despite a **$2B valuation**, there are doubts about Devin's product efficacy and support, with some users expressing frustrations leading to cancellations.
   - Members agree that competing companies will likely stabilize prices, preventing drastic increases even if products seem revolutionary.
- **AI Startup 01.AI's Claims**: Kai-Fu Lee from **01.AI** has refuted rumors of disbandment and selling teams to Alibaba, stating that their revenue for 2024 surpassed **RMB 100 million ($14 million)** and expects significant growth in 2025 [source](https://technode.com/2025/01/07/01-ai-refutes-rumors-of-selling-teams-to-alibaba/).
   - Despite the positive revenue outlook, uncertainties linger as the company reportedly laid off its pre-training algorithm and infrastructure teams.
- **Anthropic’s Funding Round**: More discussions about **Anthropic** revealed they have secured a 2B investment, valuing the company at **$60 billion**, with $875 million expected in ARR, primarily from B2B sources.
   - This bold funding has sparked intrigue about the competitive landscape as many startups jockey for position.
- **Market Hype and Reality Check**: The channel reflects on the rapid market hype surrounding AI, cautioning against believing that such technologies will lead to extensive job losses overnight.
   - Instead, participants highlighted the importance of established competition, such as Google, in keeping expectations—and prices—realistic.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/steph_palazzolo/status/1876646459698991573">Tweet from Stephanie Palazzolo (@steph_palazzolo)</a>: One reason why OpenAI has let rivals beat it to launching AI agents is a fear of so-called prompt injection attacks.Not to fear, though... we&#39;re hearing that OpenAI&#39;s computer-using agent soft...</li><li><a href="https://x.com/TheXeophon/status/1876529686349824163">Tweet from Xeophon (@TheXeophon)</a>: @adibvafa Context: We had some usages in our account which couldn’t be attributed to anything, the dashboard was contradictory and some usages were billed without any model being responsible. Got some...</li><li><a href="https://x.com/paulgauthier/status/1872423717969801320">Tweet from Paul Gauthier (@paulgauthier)</a>: Aider v0.70.0 is out:- Full support for o1.- Watch files:  - Now honors --subtree-only  - Improved prompting  - Show hints about AI! and AI? usage- Aider wrote 74% of the code in this release. Full ch...</li><li><a href="https://technode.com/2025/01/07/01-ai-refutes-rumors-of-selling-teams-to-alibaba/">01.AI refutes rumors of selling teams to Alibaba &#183; TechNode</a>: 01.AI, one of China’s leading AI unicorn startups, has been rumored to be disbanded, with its pre-training and card teams reportedly sold to Alibaba. In
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1325973756652552273)** (8 messages🔥): 

> `MosaicML researchers, ChatGPT transcription versions, Token usage in responses` 


- **MosaicML researchers facing challenges**: A member expressed concern for **MosaicML researchers** saying there are *sad goose noises* around them, hinting at undisclosed issues.
   - Another member emphasized their affection for the **Mosaic team**, appreciating their **awesome streaming dataset**.
- **Incognito rumors suggest difficulties**: A user mentioned having reliable secondhand information regarding MosaicML, stating they will not disclose more unless the community engages publicly.
   - This sparked speculation within the group, leaving many curious about the situation.
- **ChatGPT transcription humor**: A member humorously reflected on the number of variations produced by their Windows 10 transcription, listing names like *chatty gpt* and *chat ebt*.
   - This lighthearted comment drew laughter, showcasing the community's playful take on AI quirks.
- **Breaking news on ChatGPT's emoji usage**: A member announced that they instructed their *ChatGPT version* to stop using emojis because *it wastes tokens*.
   - This sentiment resonated with some, highlighting ongoing discussions about resource efficiency in AI interactions.


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1325937513214247015)** (39 messages🔥): 

> `Nvidia Project Digits Supercomputer, Challenges with Nvidia ARM CPUs, Community collaboration and funding for AI, Open-source software compatibility` 


- **Nvidia Project Digits packs AI power on desktop**: Nvidia announced [Project Digits](https://www.theverge.com/2025/1/6/24337530/nvidia-ces-digits-super-computer-ai) at CES, offering a personal AI supercomputer with the new GB10 Grace Blackwell Superchip capable of handling models with up to 200 billion parameters, priced at $3,000.
   - CEO Jensen Huang emphasized its potential for mainstream AI applications, which could give developers access to cutting-edge computational resources.
- **Concerns over Nvidia's ARM CPUs persist**: Discussions revealed that Nvidia's ARM CPU architecture poses challenges, with many open source packages lacking precompiled binaries, leading to difficulties in installation and compatibility.
   - Users shared experiences of their struggles working with Nvidia's Jetson devices, which had raw compute power but were challenging from a software perspective.
- **Funding discussions for new LLM machines**: Community members questioned whether to allocate a significant portion of their annual budget on new personal LLM machines, weighing the benefits of potential AI model experimentation.
   - Conversations highlighted the interest in utilizing unreleased AI models and the importance of cost-effectiveness for nonprofit organizations.
- **Collective learning through open source**: A member expressed willingness to assign a teammate to collaborate on tasks that promote open source and assist the community in learning and sharing knowledge.
   - The sentiment of using resources effectively while benefiting from mutual support and collaboration was emphasized.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.twitch.tv/badmephisto">badmephisto - Twitch</a>: The adventures of a Dwarf priest on Dreamscythe realm of WoW Classic</li><li><a href="https://www.theverge.com/2025/1/6/24337530/nvidia-ces-digits-super-computer-ai">Nvidia announces $3,000 personal AI supercomputer called Digits</a>: It’s the size of a desktop.</li><li><a href="https://www.bloomberg.com/news/articles/2025-01-06/us-adds-tencent-to-chinese-military-blacklist-shares-decline?utm_source=website&utm_medium=share&utm_campaign=copy">Bloomberg - Are you a robot?</a>: no description found</li><li><a href="https://www.bloomberg.com/news/articles/2025-01-06/us-adds-tencent-to-chinese-military-blacklist-sha">Bloomberg - Are you a robot?</a>: no description found</li><li><a href="https://x.com/HanchungLee/status/1876468876747341938">Tweet from Han (@HanchungLee)</a>: 10x scaling from leather jacket to alligator leather jacket.</li><li><a href="https://www.theregister.com/2025/01/07/nvidia_project_digits_mini_pc/">Nvidia unveils cut-down Grace-Blackwell Superchip</a>: Tuned for running chunky models on the desktop with 128GB of RAM, custom Ubuntu</li><li><a href="https://blackforestlabs.ai/flux-nvidia-blackwell/">Bringing Lightning&#x2d;Fast FLUX Performance to More Creators in Collaboration with NVIDIA</a>: Our new collaboration with NVIDIA marks a significant leap forward in making our FLUX models more universally accessible and efficient. Through reduced memory requirements, faster performance&#8230;</li><li><a href="https://huggingface.co/collections/nvidia/cosmos-6751e884dc10e013a0a0d8e6">Cosmos - a nvidia Collection</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1325981855614500874)** (4 messages): 

> `AI2 community involvement, Kling v1.6 and trolley problem, Nextcloud support challenges` 


- **AI2 communications evolving positively**: One member noted that **AI2** communications are becoming more **based**, suggesting optimism about their future direction in collaboration.
   - *If we are going to benefit from AI2's work, the community should find ways to give back.*
- **Kling v1.6 backs away from trolley problem**: A member attempted to test how **Kling v1.6** would respond to the **trolley problem**, but it merely *backed away slowly*.
   - This raises questions about **AI's ethical programming** and its responses to moral dilemmas.
- **Nextcloud's OSS community needs support**: Concerns were raised about **Nextcloud**, an OS platform that has limited community support despite its potential.
   - *Nextcloud GmbH* is busy enhancing features for institutional clients, and the community is encouraged to step up their contributions.
- **Support for Nextcloud's OSS community**: There is a collective concern for **Nextcloud** and its open-source community, with one member stating, *praying for Nextcloud and its OSS community*.
   - This highlights the need for more grassroots support initiatives to bolster its usage and community engagement.



**Link mentioned**: <a href="https://fxtwitter.com/fofrAI/status/1876638297134678173">Tweet from fofr (@fofrAI)</a>: I tried to see how Kling v1.6 would handle the trolley problem.But it just backed away slowly.

  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1325935902198857942)** (67 messages🔥🔥): 

> `Agents in RL Training, Function Calling and Tool Usage, Self-Correction Mechanisms, Reward Models and Gaming Behavior, Reasoning Traces Generation` 


- **Agents Coming with Diverse RL Training**: There is speculation on how agents will be trained using RL in complex environments, as one member noted, *'it can't be just Question => Forward CoTs => output and then receiving reward.'*
   - The conversation highlighted the need for diverse interactions with verifiable outcomes to effectively train these agents.
- **Function Calling and Tool Usage Clarified**: Members discussed the differences in function calling mechanisms, noting that o1 creates orchestration logic while 4o executes it using various tools.
   - This raised questions about how models allocate and utilize resources for task completion during reinforcement learning.
- **Emergence of Self-Correction in Models**: There's a debate on whether self-correcting behaviors in models are emergent or programmed, as noted by one participant regarding the *'thousands of tokens of CoT' generated by o1/r1/QwQ.*
   - This led to discussions around using processes like MCTS/PRMS for training effective reasoning traces for models.
- **Concerns About Reward Shaping and Gaming**: The topic of gaming behavior in reward models was raised, with one member questioning whether penalties would effectively address this issue.
   - There was consensus that reward shaping is complex and needs careful consideration to avoid unintended behaviors in models.
- **Exploration of Reasoning Trace Generation**: Members expressed uncertainty about generating effective reasoning traces, with one suggesting leveraging human data and clever prompting techniques.
   - The conversation concluded with acknowledgment that effective reasoning may require more than just typical instruction-tuning techniques.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/natolambert/status/1856777789313355929)">Tweet from Nathan Lambert (@natolambert)</a>: Ability unlocked.. &#34;wait, let&#39;s think step by step again carefully&#34;yes rl, no not o1</li><li><a href="https://cookbook.openai.com/examples/o1/using_reasoning_for_routine_generation">Using reasoning for routine generation | OpenAI Cookbook</a>: Open-source examples and guides for building with the OpenAI API. Browse a collection of snippets, advanced techniques and walkthroughs. Share your own examples and guides.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1325967460481896559)** (9 messages🔥): 

> `MeCo Method, Contextual Artifacts in LM Training, Danqi's Contributions, Physics of LLM Papers, Impact of Timestamps` 


- **MeCo Method simplifies LM pre-training**: The introduction of [MeCo](https://x.com/gaotianyu1350/status/1876303908899037642), which uses metadata conditioning followed by cooldown, offers a *remarkably simple method* for LM pre-training by prepending source URLs to documents.
   - One member initially found this concept *ridiculous*, but acknowledged that the URLs likely provide important context about the language used.
- **Contextual artifacts may enhance language models**: Discussion arose about what *contextual artifacts* could be added to improve models, with one user suggesting timestamps might influence the model's ability to understand time.
   - This led to comparisons with the WRAP technique, proposing that *globbing* artifacts could be a productive approach.
- **Danqi gains appreciation**: Members expressed admiration for *Danqi*, with one stating that they 'love crazy stuff that works', reflecting a positive reception of innovative ideas.
   - Another noted that Allen-Zhu had previously highlighted relevant points in the 'physics of LLM' papers, indicating deep engagement with the topic.
- **Part 3 video cited for additional context**: A member referenced the *Part 3 video*, indicating that it provides additional insights related to the discussed concepts.
   - This suggestion hints at a collaborative learning process as users seek to build on previous discussions with multimedia resources.



**Link mentioned**: <a href="https://x.com/gaotianyu1350/status/1876303908899037642">Tweet from Tianyu Gao (@gaotianyu1350)</a>: Introducing MeCo (metadata conditioning then cooldown), a remarkably simple method that accelerates LM pre-training by simply prepending source URLs to training documents.https://arxiv.org/abs/2501.01...

  

---


### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1325969803638210700)** (1 messages): 

> `Agents and Labor Policy, National Security, Model Shops and AI Proliferation` 


- **Exploring Agents' Role in Labor Policy**: A member expressed a need for articles on how to interact with **agents on the job site**, highlighting that discussions primarily focus on their **proliferation**.
   - *The sentiment gravitates towards skepticism,* but the reality is that capital will likely **deploy agents** for efficiency if they prove to perform at just **20% of human capability** at a fraction of the cost.
- **National Security Coverage is Saturated**: It was noted that the **national security angle** related to agents appears to be well-covered in existing discussions.
   - The member believes more emphasis should be placed on practical impacts and **working dynamics with agents** in labor contexts.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1325978343862177874)** (22 messages🔥): 

> `Training High-Parameter LLMs, Deepspeed Zero-3 Memory Issues, Gradient Checkpointing, Ethics Dataset Evaluation, Learning and Contribution in AI` 


- **Memory Concerns in Training 7B LLM**: A member reported that training their **7B LLM** consumes around **35GB** of memory despite freezing most weights and only having a few million trainable parameters.
   - They suspect that the high memory usage during training might be due to the absence of gradient checkpointing and unnecessary overhead from gradients and optimizer states.
- **Deepspeed Zero-3 Doesn't Reduce Memory**: Another member expressed frustration that their attempts at using **Deepspeed Zero-3** for model sharding did not yield any memory reductions.
   - *They claimed not to fully understand **Deepspeed** but tried it with hope but ended up seeing no benefits*.
- **Gradient Checkpointing Might Be Essential**: Community members discussed the potential need for **gradient checkpointing** to alleviate memory usage issues during training.
   - As stated, there’s significant memory overhead with gradients, optimizer states, and high-precision model copies during training.
- **Interest in Evaluating Pythia on Ethics Dataset**: A member inquired whether anyone has evaluated **Pythia** on the [Ethics dataset](https://huggingface.co/datasets/hendrycks/ethics).
   - This highlights a continued interest in evaluating model performance in ethical contexts.
- **Curiosity About Learning AI Techniques**: A member expressed curiosity about how others acquire knowledge in AI and conveyed a desire to contribute to the community.
   - This hints at an ongoing interest in community involvement and collaborative learning in AI.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1325979633614590023)** (9 messages🔥): 

> `Cerebras AI Grant Proposals, Inference-Aware Fine-Tuning for LLMs, In-Context Learning Representations, Tensor-GaLore for Neural Network Training, Cut Cross-Entropy Loss Method` 


- **Cerebras AI seeks research proposals**: Cerebras is inviting university faculty and researchers to respond to a [Request for Proposals](https://cerebras.ai/blog/grantfrp) aimed at advancing the field of Generative AI.
   - They emphasize their goal to support innovative research leveraging their **third generation Wafer Scale Engine**, providing substantial performance benefits.
- **Novel fine-tuning paradigm for LLMs**: A recent paper proposes an **inference-aware fine-tuning** paradigm that optimizes models for better performance during inference, evaluated through the **Best-of-N (BoN)** strategy [View PDF](https://arxiv.org/abs/2412.15287).
   - The authors demonstrate the effectiveness of this approach using imitation learning and reinforcement learning methods to interleave best and diverse responses.
- **LLMs create in-context representations**: A paper titled *In-Context Learning of Representations* discusses how LLMs form **'in-context representations'** to align with the given task structures [link](https://x.com/corefpark/status/1875929881856573905).
   - The findings suggest significant behavioral adaptations by LLMs in the **large context limit**, showing how they manage task alignment.
- **Efficient training with Tensor-GaLore**: Tensor-GaLore introduces a method for efficient **training of neural networks** with higher-order tensor weights, enhancing memory efficiency [View PDF](https://arxiv.org/abs/2501.02379).
   - It particularly focuses on optimization within the **high-order parameter space**, showcasing advantages in solving complex **partial differential equations**.
- **Memory savings with Cut Cross-Entropy**: A discussion highlighted the **Cut Cross-Entropy (CCE)** method as a solution to the increased memory footprint of LLMs by optimizing cross-entropy computation without materializing all logits [View PDF](https://arxiv.org/abs/2411.09009v1).
   - This novel approach reduces global memory consumption significantly during training, particularly important for large models with extensive vocabularies.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2412.15287">Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models</a>: Recent studies have indicated that effectively utilizing inference-time compute is crucial for attaining better performance from large language models (LLMs). In this work, we propose a novel inferenc...</li><li><a href="https://arxiv.org/abs/2501.02379">Tensor-GaLore: Memory-Efficient Training via Gradient Tensor Decomposition</a>: We present Tensor-GaLore, a novel method for efficient training of neural networks with higher-order tensor weights. Many models, particularly those used in scientific computing, employ tensor-paramet...</li><li><a href="https://arxiv.org/abs/2411.09009v1">Cut Your Losses in Large-Vocabulary Language Models</a>: As language models grow ever larger, so do their vocabularies. This has shifted the memory footprint of LLMs during training disproportionately to one single layer: the cross-entropy in the loss compu...</li><li><a href="https://x.com/corefpark/status/1875929881856573905">Tweet from Core Francisco Park @ NeurIPS2024 (@corefpark)</a>: New paper! “In-Context Learning of Representations”What happens to an LLM’s internal representations in the large context limit?We find that LLMs form “in-context representations” to match the structu...</li><li><a href="https://cerebras.ai/blog/grantfrp">Announcing Cerebras Inference Research Grant - Cerebras</a>: AIBI (AI Bot Interviewer) is the first end-to-end AI interview bot that delivers a seamless, real-time interview experience.
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1325993594900971551)** (112 messages🔥🔥): 

> `Evaluation of Chat Templates vs No Chat Templates, Logprob Analysis of Multiple Choice Questions, Instruct Model Performance, Arc Challenge and Generation Tasks, Chat Format Impact on Model Responses` 


- **Chat Templates may hinder Performance**: A member noted that while running evaluations with **chat templates**, the performance on multiple choice questions seemed lower compared to runs without templates, specifically mentioning **L3 8B base** scoring much higher without templates.
   - It was suspected that the chat format biases the model into conversational responses, reducing its ability to output precise letter answers.
- **Logprobs indicate potential Output Issues**: Discussion suggested that when using **logprobs** for multiple-choice tasks, the model may produce very low probabilities for all valid answers when queried in **chat format**.
   - It was recommended to analyze unrestricted logprobs to see if the model struggles to generate correct answers when under the constraints of a chat format.
- **Instruct Model may have Performance Discrepancies**: There was a debate on why the **instruct models** score lower on multiple-choice tasks, with suggestions that the restricted output space contributes to this issue.
   - Members considered a hypothesis that instructions in **chat format** led to better performances in conversational tasks but hindered precise outputs for structured responses.
- **Evaluation Harness for Generation Tasks**: One member highlighted the idea of creating a training data set that uses a system prompt to strictly dictate response formats for multiple-choice tasks and integrating this into the **evaluation harness**.
   - They noted that allowing the model some freedom to generate answers before structuring the output may yield better results in some contexts.
- **Challenges in Extracting Answers**: Concerns were raised about whether models trained on conversational formats could effectively output concise answers, suggesting that testing various configurations could reveal discrepancies.
   - There was interest in letting the model generate freely before calculating logprobs, potentially improving model effectiveness in structured tasks.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1326075339687854091)** (5 messages): 

> `Llama2 Checkpoints Conversion, Optimizer Support in NeoX, Scheduler Syntax in Configs, Mixed Precision Loss Scaling, Pythia Batch Size Calculation` 


- **Llama2 Checkpoints Compatibility**: A user inquired whether the saved checkpoints from training with Llama2 configs in NeoX are directly convertible to Hugging Face format.
   - *No direct response was provided, indicating uncertainty about the compatibility.*
- **AdamW vs Lion Optimizer Support**: Questions arose about the absence of AdamW and the inclusion of Lion in the NeoX training script, prompting a look at the [training.py file](https://github.com/EleutherAI/gpt-neox/blob/main/megatron/training.py#L1036).
   - *The user expressed surprise at this discrepancy, highlighting AdamW's popularity in similar contexts.*
- **Scheduler Syntax Inquiry**: A member presented a proposed syntax for passing the scheduler dictionary based on the optimizer dictionary, specifically for a WarmupCosineLR configuration.
   - *Feedback or confirmation on this proposed configuration was not provided in the discussion.*
- **Questions on BF16 Loss Scaling**: The user wondered if there's a benefit to including loss scale parameters for BF16 in NeoX and sought an example configuration that uses BF16 mixed precision.
   - *No examples or references were shared to clarify this aspect of the implementation.*
- **Pythia Global Batch Size Calculation**: Clarification was sought on the global batch size calculation for Pythia, considering the effective batch size of 16 across 128 GPUs, equating to 2048 globally.
   - *There was confusion about whether the total tokens processed contradict the stated documentation of 2M tokens.*



**Link mentioned**: <a href="https://github.com/EleutherAI/gpt-neox/blob/main/megatron/training.py#L1036">gpt-neox/megatron/training.py at main · EleutherAI/gpt-neox</a>: An implementation of model parallel autoregressive transformers on GPUs, based on the Megatron and DeepSpeed libraries - EleutherAI/gpt-neox

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1325932912582852742)** (138 messages🔥🔥): 

> `OpenRouter Payment Issues, Model Performance Concerns, DeepSeek V3 Reliability, Using Crypto for Payments, LLM Limitations in Game Development` 


- **OpenRouter Payment Issues Persist**: Users report ongoing payment problems with OpenRouter, including multiple declined transactions using virtual cards and difficulty with their payment gateway.
   - One user expressed frustration with their card not supporting OR purchases anymore, suggesting a shift to crypto payments.
- **Concerns Over Model Performance**: Users discussed the frequent crashes of Lambda's Hermes 405b, noting the status indicator still shows green despite issues.
   - There were also mentions of perceived slow performance from DeepSeek V3, which some users attributed to high demand.
- **DeepSeek V3 Reliability Issues**: DeepSeek V3 is experiencing reliability concerns, especially under high input conditions, affecting functionality across platforms.
   - A user pointed out that the issue seems to be prevalent on both DeepSeek and OpenRouter APIs.
- **Exploring Crypto Payments**: Several users discussed the feasibility of using crypto for payments in place of traditional methods, highlighting its advantages in certain regions.
   - Trust Wallet and other providers were suggested as potential options for users in the Philippines struggling with payment issues.
- **Limitations of LLMs in Game Creation**: Users explored the limitations of current LLMs, like O3 and GPT-5, in creating more complex game designs compared to simpler 2D games.
   - There was a consensus that while simpler games could potentially be generated, more complex designs remain challenging due to organizational difficulties.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/">Home</a>: aider is AI pair programming in your terminal</li><li><a href="https://github.com/cline/cline/issues/1157">API Request loading indefinitely, not completing. · Issue #1157 · cline/cline</a>: What happened? API Request starts loading indefinitely, never completing. I&#39;m using Deepseek v3. It was working totally fine for some 2 hours, then suddenly this started happening, in any chat win...</li><li><a href="https://github.com/googleapis/python-genai/pull/84">feat: Support API keys for VertexAI mode by copybara-service[bot] · Pull Request #84 · googleapis/python-genai</a>: feat: Support API keys for VertexAI mode
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1325948302973468842)** (73 messages🔥🔥): 

> `Aider's utility in coding, Issues with O1 Pro, Using Continue.dev alongside Aider, Tips for effective AI interactions, Challenges with command execution` 


- **Aider shines as a coding assistant**: Many users praised Aider for its ability to assist with complex coding tasks, noting that it's like a coding teacher rather than just a tool.
   - One user emphasized the importance of reading Aider's prompts and effectively utilizing /ask commands to refine outputs.
- **O1 Pro has inconsistencies**: A couple of users reported issues with O1 Pro, including the absence of the progress bar and unsatisfactory solution times.
   - Despite frustrations, some users prefer O1 Pro over alternatives, while others feel both options can be used in tandem for best results.
- **Integrating Continue.dev with Aider**: Users have started to explore Continue.dev as a complementary tool to Aider, particularly for quicker interactions and task management.
   - One user shared that this combination is not only productive but also helps in managing more significant coding tasks efficiently.
- **Effective strategies for AI-powered coding**: Members discussed strategies for leveraging AI in coding, suggesting the importance of crafting detailed prompts to enhance responses.
   - One user illustrated how using Sonnet and O1 Pro together yields excellent outcomes by iteratively refining queries.
- **Challenges in command execution**: Concerns were raised regarding Aider's execution of git commands, noting that while commands work outside the tool, issues have arisen within it.
   - Following troubleshooting, a user found that a system update resolved their command path visibility issue with Aider.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/usage/images-urls.html#web-pages">Images &amp; web pages</a>: Add images and web pages to the aider coding chat.</li><li><a href="https://jina.ai/reader/">Reader API</a>: Read URLs and search web for better grounding LLMs.</li><li><a href="https://github.com/ai-christianson/RA.Aid">GitHub - ai-christianson/RA.Aid: Aider in a ReAct loop</a>: Aider in a ReAct loop . Contribute to ai-christianson/RA.Aid development by creating an account on GitHub.</li><li><a href="https://github.com/gorilla-llm/gorilla-cli">GitHub - gorilla-llm/gorilla-cli: LLMs for your CLI</a>: LLMs for your CLI. Contribute to gorilla-llm/gorilla-cli development by creating an account on GitHub.</li><li><a href="https://r.jina.ai/https://docs.astro.build/en/guides/routing/">{title}</a>: no description found
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1325960078528090182)** (50 messages🔥): 

> `Aider prompt caching, Custom LLM usage with Aider, Terminal display issues, Color themes for terminal, Troubleshooting file updates in Aider` 


- **Aider Supports Prompt Caching for Faster Coding**: Aider supports prompt caching to enhance performance when running commands, as highlighted in the [documentation](https://aider.chat/docs/usage/caching.html). Users can enable caching and manage it with specific command-line options and settings.
- **Integrating Custom LLMs in Aider**: Users are discussing how to use custom language models with Aider, specifically by registering and instantiating custom classes. Some have succeeded by prefixing their model names with 'custom/' and configuring API settings correctly.
- **Addressing Terminal Display Issues**: A user reported issues with Aider's output not fitting within a small terminal window, leading to disorganized text. Suggestions were made to check terminal configurations and ensure that views can accommodate all output effectively.
- **Exploring Color Theme Options for Windows Terminal**: There were inquiries about improving the readability of color themes in Windows Terminal, particularly regarding dark modes making certain colors hard to see. Users are seeking advice to find a suitable color palette that enhances visibility.
- **Troubleshooting Aider File Update Issues**: A user faced a problem where Aider wouldn't update files despite providing context and instructions. It was suggested to check for errors, reproduce the issue, or try using the latest main branch via `aider --install-main-branch`.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://0.0.0.0:8000"">no title found</a>: no description found</li><li><a href="https://aider.chat/docs/usage/conventions.html">Specifying coding conventions</a>: Tell aider to follow your coding conventions when it works on your code.</li><li><a href="https://aider.chat/docs/usage/caching.html">Prompt caching</a>: Aider supports prompt caching for cost savings and faster coding.</li><li><a href="https://aider.chat/docs/config/adv-model-settings.html">Advanced model settings</a>: Configuring advanced settings for LLMs.</li><li><a href="https://aider.chat/docs/config/options.html">Options reference</a>: Details about all of aider’s settings.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1326270033378152530)** (2 messages): 

> `Aider workflow adaptation, LLM-guided interviews` 


- **Adapting a Good Workflow for Aider**: A user highlighted a [YouTube video](https://www.youtube.com/watch?v=XWJGm3y207A) showcasing a good workflow that could be adapted for Aider's functionalities.
   - This workflow focuses on enhancing communication and efficiency in developing coding tasks.
- **LLM Interviews Boost Coding Prompts**: A user expressed excitement about a prompt that guides an **LLM through interviewing** the user to create specifications.
   - This approach aims to feed back into coding prompts, providing **structured** and effective coding guidance.


  

---


### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1325933958579355649)** (14 messages🔥): 

> `NBA Game Recaps, AI in Virtual Sportscasting, Sources and Citation Practices, AI for Contract Review, NotebookLM's Capabilities` 


- **NBA Game Recaps Get a Tech Upgrade**: A member suggested using LMNotebook to overlay game recaps with highlights for the NBA and NFL, emphasizing its cost-effectiveness.
   - They shared a [YouTube video](https://youtu.be/D7qZ2VphetU) demonstrating the concept as a way to produce branded content at scale.
- **Discussion About Using Reliable Sources**: Members discussed sourcing content, with one using **Britannica** and another questioning if it came from **Wikipedia**.
   - A member confirmed using a **single source**, while another sought a system prompt for quoting relevant parts accurately.
- **AI and Digital Labor for Contract Reviews**: A member highlighted how AI and virtual paralegals could ease the burden of **contract 'redlining'**, which can be time-consuming and costly.
   - By engaging stakeholders with avatars, the process becomes more interactive, streamlining preparations and facilitating understanding.
- **NotebookLM Enhancing Collaborative Learning**: NotebookLM is transforming the training landscape by organizing content into topic-specific notebooks for better research and collaboration.
   - It enables group projects and serves as a continually updated resource, empowering participants to engage more effectively.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.akashq.com/post/4cc67748-6af9-4083-b63c-7a0d366dc01d">What happened on Jan 7?</a>: What happened on Jan 7? by This Day in History</li><li><a href="https://www.akashq.com/post/b1eeb736-890a-4306-bc56-2ac605e739d2">What happened on Jan 6?</a>: What happened on Jan 6? by This Day in History
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1325927740108701726)** (86 messages🔥🔥): 

> `NotebookLM Usage Limits, NotebookLM Plus Features, Audio Overview Length, Missing Features, Google Workspace Questions` 


- **NotebookLM Usage Limits Cause Slowdown**: Users expressed concerns about potential daily limits impacting NotebookLM's performance, noting that it became slow after prolonged use.
   - One member suggested checking the [official support page](https://support.google.com/notebooklm/answer/15678219?hl=en) for more information.
- **Explaining NotebookLM Plus Features**: NotebookLM Plus offers enhanced capabilities such as the ability to upload multiple sources, including PDFs and YouTube links, and generate summaries.
   - Additional premium features include greater limits on notebooks and queries, as highlighted in conversations about the value of the Plus subscription.
- **Challenges with Audio Overview Length**: Several users reported difficulties in controlling the length of generated audio overviews, voicing frustration when lengthy outputs were produced.
   - Workarounds suggested included removing unwanted sources to better manage the overview focus.
- **Missing Features and Functionality**: There were reports of lost capabilities in generating AI-generated question suggestions from selected text, impacting users' workflows.
   - Users sought resolutions and advice on expected features' status amid ongoing updates.
- **Google Workspace License Clarifications**: Questions arose regarding the proper licenses required for accessing NotebookLM Plus features within Google Workspace accounts.
   - Participants discussed the necessity of specific add-on licenses and how to activate them, along with support page references.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.testingcatalog.com/icymi-gemini-advanced-users-now-have-access-to-notebooklm-plus/">Gemini advanced users now have access to NotebookLM Plus</a>: Discover NotebookLM Plus by Google, an AI-powered research tool with expanded capabilities and customization options. Boost productivity with advanced features and higher usage limits.</li><li><a href="https://support.google.com/notebooklm/answer/15678219?hl=en">Upgrading to NotebookLM Plus - NotebookLM Help</a>: no description found</li><li><a href="https://support.google.com/notebooklm/answer/15678219?hl=en#:~:text=Where%20to%20get%20NotebookLM%20Plus%C2%A0">Upgrading to NotebookLM Plus - NotebookLM Help</a>: no description found</li><li><a href="https://support.google.com/notebooklm/answer/15678219?visit_id=638718328734806536-3512091001&p=plus&rd=1">Upgrading to NotebookLM Plus - NotebookLM Help</a>: no description found</li><li><a href="https://support.google.com/notebooklm/answer/15678219?sjid=11787293163246227446-AP&visit_id=638717415901875295-2373678412&p=plus&rd=1">Upgrading to NotebookLM Plus - NotebookLM Help</a>: no description found</li><li><a href="https://support.google.com/a/answer/6043385?hl=en&co=DASHER._Family%3DBusiness">Compare Google Workspace editions - Business - Google Workspace Admin Help</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1325937907034357761)** (78 messages🔥🔥): 

> `Nous Forge API updates, Performance comparisons of RTX GPUs, NVIDIA Project DIGITS, AI bot behavior tweaks, USB-C for networking` 


- **Nous Forge API phase ends**: The beta phase for [Nous Forge API](https://forge.nousresearch.com/) ended recently, where users can still subscribe for updates to learn about configurations in reasoning engines that utilize various models.
   - This API enables advanced reasoning and transparency for complex tasks involving Hermes, Claude, Gemini, and OpenAI models.
- **RTX GPUs performance debate**: Members discussed the pricing and performance of RTX 3090 and the new RTX 5070, with skepticism about NVIDIA's claims of delivering 4090 performance at a lower price.
   - Concerns were raised about comparing AI performance without substantial benchmarks, pointing to NVIDIA's tactics of claiming superior performance based on AI texture compression.
- **NVIDIA's new Project DIGITS**: [NVIDIA Project DIGITS](https://nvidianews.nvidia.com/news/nvidia-puts-grace-blackwell-on-every-desk-and-at-every-ai-developers-fingertips) was unveiled, featuring the Grace Blackwell Superchip designed to provide high-performance AI computing at an accessible level.
   - This development aims to democratize access to AI supercomputers for researchers and students, making model deployment and development more feasible.
- **Tuning AI character behavior**: The channel discussed techniques to adjust the behavior of AI models to prevent overly anxious character responses, with suggestions regarding system prompts.
   - Discussion included examples of prompts aimed at instilling confidence, as well as humor about the issues faced with AI logs.
- **USB-C as a networking solution**: USB-C connections can achieve high speeds (10-20Gbps) for networking between two PCs, making it a budget-friendly option.
   - Participants shared insights on selecting compatible cables for optimal performance and mentioned potential limitations in scaling.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://forge.nousresearch.com/">Forge Reasoning API by Nous Research</a>: Forge Reasoning API by Nous Research</li><li><a href="https://nvidianews.nvidia.com/news/nvidia-puts-grace-blackwell-on-every-desk-and-at-every-ai-developers-fingertips">NVIDIA Puts Grace Blackwell on Every Desk and at Every AI Developer’s Fingertips</a>: CES—NVIDIA today unveiled NVIDIA® Project DIGITS, a personal AI supercomputer that provides AI researchers, data scientists and students worldwide with access to the power of the NVIDIA Grace ...</li><li><a href="https://arxiv.org/abs/2411.05007">SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models</a>: Diffusion models have been proven highly effective at generating high-quality images. However, as these models grow larger, they require significantly more memory and suffer from higher latency, posin...</li><li><a href="https://x.com/meeix_/status/1876465349278994804">Tweet from Meeix (pronounced: “makes”) (@meeix_)</a>: rtx 5070 delivering &#34;the performance of a 4090&#34; for $549 ?????? bro nvdia just came out swinging</li><li><a href="https://x.com/drjimfan/status/1876516972512559170?s=46">Tweet from Jim Fan (@DrJimFan)</a>: Introducing NVIDIA Cosmos, an open-source, open-weight Video World Model. It&#39;s trained on 20M hours of videos and weighs from 4B to 14B. Cosmos offers two flavors: diffusion (continuous tokens) an...</li><li><a href="https://x.com/NousResearch/status/1848397863547515216">Tweet from Nous Research (@NousResearch)</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1326177295320219752)** (1 messages): 

> `Reputation concerns, Privacy issues, Profit-driven motivations` 


- **Concerns over Reputation and Privacy**: A member expressed that many organizations have been **operating for quite a while** but have not built a strong reputation concerning **privacy**.
   - They emphasized how this has contributed to various **doubts** about their intentions, particularly regarding **profit-driven motivations**.
- **Profit-Driven Motivations in Organizations**: Concerns were raised that organizations are primarily driven by **profit motivations**, which affects their commitment to user privacy and security.
   - The discussion highlighted the **distrust** that arises when profits take precedence over user protection.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1326334069180731457)** (3 messages): 

> `Structure of Neural Embeddings, MiniMind Training Pipeline, MiniMind Model Overview` 


- **Insights into Structure of Neural Embeddings**: A blog post explores the **structure of neural latent spaces** with several principles, such as the **Manifold Hypothesis**, stating that high-dimensional real-world data lies in low-dimensional manifolds. Additional explorations include concepts of **Hierarchical Organization** and the **Linear Hypothesis** related to neural network features.
   - For a deeper understanding, links to [Manifolds and Topology](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/), [Visualizing Representations](https://colah.github.io/posts/2015-01-Visualizing-Representations/), and [NLP Representations](https://colah.github.io/posts/2014-07-NLP-RNNs-Representations/) provide further reading.
- **Complete Training Pipeline for MiniMind**: The **MiniMind** project features a complete training pipeline for a small language model that includes pretraining, SFT, and DPO, suitable for 2x RTX3090 GPUs, as outlined in the [English README](https://github.com/jingyaogong/minimind/blob/master/README_en.md). The project allows anyone to train a model with just 26.88M parameters in approximately 3 hours.
   - This open-source initiative not only streamlines the training process but also serves as a tutorial for those starting with large language models, aiding in the understanding of model training and fine-tuning techniques.
- **MiniMind Model Capabilities**: The **MiniMind** model is praised for its lightweight structure, being about **1/7000 the size of GPT-3**, making quick inference and training accessible to users with average GPUs. The project includes code for a simplified model structure, dataset preprocessing, supervised pretraining, SFT, LoRA fine-tuning, and DPO.
   - Furthermore, it supports expanding the capabilities to **sparse models with mixed experts** and **multi-modal vision language models** as seen in the MiniMind-V project, enriching the resource for model explorations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://seanpedersen.github.io/posts/structure-of-neural-latent-space">Structure of Neural Embeddings</a>: no description found</li><li><a href="https://jingyaogong.github.io/minimind/">MiniMind Project</a>: no description found</li><li><a href="https://github.com/jingyaogong/minimind/blob/master/README_en.md">minimind/README_en.md at master · jingyaogong/minimind</a>: 「大模型」3小时完全从0训练26M的小参数GPT，个人显卡即可推理训练！. Contribute to jingyaogong/minimind development by creating an account on GitHub.</li><li><a href="https://github.com/jingyaogong/minimind">GitHub - jingyaogong/minimind: 「大模型」3小时完全从0训练26M的小参数GPT，个人显卡即可推理训练！</a>: 「大模型」3小时完全从0训练26M的小参数GPT，个人显卡即可推理训练！. Contribute to jingyaogong/minimind development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1325922647728586843)** (58 messages🔥🔥): 

> `Perplexity performance issues, Concerns about privacy and ads, User interface feedback, SOC 2 compliance inquiries, Subscription and usage questions` 


- **Perplexity faces widespread slowdowns**: Multiple users reported that **Perplexity** is experiencing significant slowdowns, with one mentioning it takes minutes for responses on both mobile and browser.
   - Users suggested using workarounds like typing in notepad and pasting responses as the app struggles to keep up.
- **Privacy concerns spike after personalized ads**: A user expressed concern over receiving targeted ads on **Instagram** after searching for specific health symptoms in Perplexity, fearing the app reads chat content.
   - This prompted a discussion about using the app without being logged in for better privacy.
- **Inquiries about SOC 2 compliance**: Users pushed for information regarding **SOC 2 compliance** for Perplexity, with one stating they received details on trust center info from support.
   - Another user claimed that many AIs are not compliant, countering with that they would be surprised about compliance in AI services.
- **Feedback on the Perplexity user interface**: Users criticized the app's **'shop now'** feature, claiming it undermines the product search experience by not allowing proper evaluation of sellers.
   - Some suggested improving functionalities like **toggle options** for Pro Searches and enabling clearer feedback on usage limitations.
- **Subscription and usage mismatch confusion**: A user received conflicting information about available **Pro Searches**, stating they only get three instead of five every four hours as previously expected.
   - Another mentioned that options should be better documented to avoid confusion, especially due to recent changes in subscription models.



**Link mentioned**: <a href="https://trust.perplexity.ai/">Trust Center | Powered by Drata</a>: Ready to turn trust into your competitive advantage? Sprint through security reviews and quickly share key security information with Trust Center.

  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1325939202671509535)** (10 messages🔥): 

> `NASA's Moon Micro-Mission, AgiBot's Humanoid Robot Training Dataset, Microsoft's AGI Development, Disney's New Projects, Gen Z Looksmaxxing Trend` 


- **NASA Launches Moon Micro-Mission**: Today, **NASA** revealed details about its **Moon Micro-Mission**, aiming to enhance lunar exploration technologies, which can be reviewed [here](https://www.perplexity.ai/search/the-amethyst-tablet-pdf-wSFKHIw3R9CFnbPRI3Vfxw#0).
   - The mission includes innovative tech that could aid future manned missions to the Moon.
- **AgiBot Debuts Humanoid Robot Training Dataset**: **AgiBot** introduced a new **humanoid robot training dataset**, which promises to advance the capabilities of AI in anthropomorphic robotics. More information can be found in the video [here](https://www.youtube.com/embed/V8A6EdbPdGU).
   - This dataset is expected to help improve interaction and task performance in humanoid robots.
- **Microsoft's Bold $100B AGI Investment**: **Microsoft** is making headlines with its **$100 billion** investment in **AGI development**, indicating a major push in this sector. Details can be found in the discussion [link](https://www.perplexity.ai/search/why-is-it-called-donating-plas-8jLeogYsRZW4.OYo.eObhA).
   - This move is seen as critical for enhancing their tech capabilities in the race towards advanced AI.
- **Disney's New Initiative Launched**: **Disney** is set to launch exciting new projects aimed at engaging younger audiences, with updates detailed [here](https://www.perplexity.ai/search/disney-zMu663WHTbirL8TOhHD0.g).
   - These initiatives are part of Disney's strategy to maintain relevance among Gen Z.
- **Exploring Gen Z's Looksmaxxing Trend**: A new trend known as **looksmaxxing** has emerged among **Gen Z**, which focuses on enhancing personal appearance, discussed further [here](https://www.perplexity.ai/page/gen-z-s-looksmaxxing-trend-R4cK0xe0R3.lhWVq5V.p6A).
   - This trend reflects a broader cultural focus on personal image and social media presence.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1326189835081351199)** (1 messages): 

> `Mail from December 19, Concerns about online models` 


- **Members question December 19 mail**: Users are inquiring about a mail they received on **December 19**, hoping for shared experiences regarding its content.
   - *One lamented*, 'Huge bummer if they just keep the online models!'
- **Worries over model availability**: A member expressed concerns that the decision may lead to the **exclusivity of online models**, rather than continuing support for diverse options.
   - The accompanying image linked in their message sparked further discussion on this issue.


  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1326213923061370992)** (66 messages🔥🔥): 

> `AI21 Labs Token, Scam Concerns, Social Media Communication, Token Audits` 


- **AI21 Labs Token raises scam concerns**: Members repeatedly questioned if the **AI21 Labs Token** was legitimate, with many asserting it appears to be a **scam**. One user highlighted the token's dubious activity, warning others to *stay away*.
- **Call for official communication**: Concerned members urged AI21 Labs to clarify their position on the token publicly, requesting a post on **Twitter** to distance the company from the alleged scam. One user expressed frustration, stating *it doesn't cost anything to tweet a warning*.
- **Platform's stance on cryptocurrency**: AI21 representatives confirmed that the token is **not affiliated** with the company and warned that further discussion about cryptocurrency could lead to **bans** from the channel. They communicated that the situation had been escalated to their **security team**.
- **Tokens being audited**: Despite claims of an audit, some users remained skeptical, stating the token is linked to **pumpfun**, raising further **scam** suspicions. Others suggested that the token's holder distribution looked questionable, indicating potential risks.
- **General sentiment about the token**: Overall, community sentiment leaned towards skepticism, with statements indicating that many believe the token has likely **rugged** or is tied to a **scam**. Users shared concerns regarding the legitimacy of the project, emphasizing the need for caution.



**Link mentioned**: <a href="https://www.dextools.io/app/es/token/ai21labs?t=1736289130032&maker=81wGWkys3bHxktg8rXPGbW7xuJ8479RALZBBH55Jz54H">DEXTools</a>: DEXTools, the gateway to DEFI, real-time charts, history and all token info from blockchain.

  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1325927612551528470)** (18 messages🔥): 

> `AGI and Innovation, AI as a Tool, Recent Advances in AI Technology, Fine-tuning AI Models, RTX 5000 DLSS 4` 


- **AGI's Impact on Innovation**: Members discussed whether **Artificial General Intelligence (AGI)** will empower or disrupt human innovation, with some believing it will enhance critical thinking rather than replace it.
   - *Right now is the best time for entrepreneurship,* noted concerns about competition with major corporations using AGI tools, leading to fears of dependency.
- **AI as a Collaborative Tool**: A member compared working with AI to having a car, emphasizing that using AI can enhance capabilities rather than diminish personal skills.
   - This perspective highlights that collaboration with AI can foster innovation as users build upon each other's strengths.
- **Exciting Advances with RTX 5000 DLSS 4**: **RTX 5000** showcasing **DLSS 4** upgrades received buzz with claims of *three times frame generation* improvements.
   - Members are excitedly discussing the potential advancements and implications for gaming and graphics performance.
- **AI Model Fine-tuning Experiences**: Questions arose about whether users have fine-tuned **LLaMA** models on their own texts or Discord conversations, sparking interest in personal experiences.
   - One member confirmed they had success in fine-tuning, remarking that it is relatively easy, especially with structured data.
- **Access to AI APIs**: A user mentioned gaining **unlimited access** to various AI systems, including those from **OpenAI**, **Google**, and **Anthropic**, as well as their API models.
   - Others chimed in, suggesting that similar resources can be found for free on platforms like GitHub.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1325922562940866720)** (9 messages🔥): 

> `Convo transfer from 4o to 1o, Mini O1 vs GPT-4, Ubuntu setup and GPU compatibility, O1 Pro upgrade discussion` 


- **Convo transfer from 4o to 1o grayed out**: A user asked if they could transfer their conversation from **GPT-4 (4o)** to **GPT-1 (1o)**, noting that the option was grayed out.
   - *A member responded* that this is not possible if they used features that **1o** does not support.
- **Mini O1 perceived as smarter than GPT-4**: A user inquired if **Mini O1** is smarter than **GPT-4**, to which another member affirmed its intelligence, mentioning it may not perform better universally.
   - This presents a nuanced view on the ongoing debate about model capabilities.
- **User shares Ubuntu specs for GPU tasks**: A user shared they are running **Ubuntu 24.04.1** with a **5800X CPU** and **6900XT GPU**, and inquired about resources for using **GPU 4o Mini**.
   - *They mentioned having ROCP 6.3.1* and previous experiences with **Ollama** versions.
- **Discussion on upgrading to O1 Pro**: A user posed a question regarding the worthiness of upgrading to **O1 Pro**, prompting discussions on its value.
   - This highlights the ongoing interest in features and improvements offered by the new model.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1325942047319920700)** (15 messages🔥): 

> `Midjourney SREF prompt in Dall-E, JSON schema responses, Retry implementations, Style naming for prompts` 


- **No SREF prompt available for Dall-E**: A member inquired about using the Midjourney —sref feature in Dall-E, but another member responded with a definitive **'No'**.
   - They noted that you can name the style in the prompt, but it often doesn't yield the expected results.
- **Issues with JSON schema returning itself**: A member reported that setting their model to JSON schema sometimes causes it to return the schema instead of a valid response **80%** of the time.
   - Despite implementing retries, they continue to encounter the same issue, suggesting potential vagueness in instructions.
- **Concerns over self-promotion regulations**: Another discussion arose regarding adherence to channel regulations against self-promotion, with a caution to be careful in such matters.
   - This led to a reflection on the appropriateness of some shared content in the conversation.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1325942047319920700)** (15 messages🔥): 

> `Midjourney prompt in Dall-E, JSON schema return issue, Retries not working, Prompt engineering concerns` 


- **Questions about Midjourney in Dall-E**: A member inquired whether there is a prompt to replicate the Midjourney —sref style in Dall-E.
   - Another member confirmed there isn't a direct prompt for this, suggesting to simply name the style and 'hope'.
- **Issues with JSON schema responses**: A user reported their model returning the JSON schema itself instead of a response 80% of the time.
   - Despite implementing retries, the same issue persisted, leading to frustration among users.
- **Input size and vagueness concerns**: A suggestion was made that the low success rate for completions could be due to input size and relative noise.
   - It was noted that vague instructions might also contribute to the problem, emphasizing the need for clarity.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1325921454591709297)** (47 messages🔥): 

> `Foundation Models in Science, NVIDIA's Cosmos, Vercel's AI SDK, AI in Whale Conservation, FP4 Wars` 


- **Exploring Foundations Models for Science**: A member shared a link to the [Metagene 1 paper](https://metagene.ai/metagene-1-paper.pdf), prompting inquiries about foundation models used in scientific applications.
   - This interest highlights the increasing relevance of AI in specialized fields.
- **NVIDIA Reveals Cosmos Model**: NVIDIA introduced [Cosmos](https://x.com/DrJimFan/status/1876516972512559170), an open-source video world model trained on **20M** hours of videos, which signifies a leap in synthetic data generation for robotics.
   - The model features both diffusion and autoregressive generation modes, showcasing NVIDIA’s ambition in the enterprise AI space.
- **Vercel's AI SDK Gains Attention**: A member discussed experiences with Vercel's AI SDK, noting its effectiveness for a simple setup but criticizing its **too much abstraction** when layered with other models.
   - This sparked discussions on the balance between usability and complexity in AI tool integrations.
- **AI Aids Whale Conservation**: A successful collaboration between [Accenture](https://x.com/btibor91/status/1876630816199217208?s=46) and the University of Sydney developed an AI system that detects minke whales with **89.4%** accuracy.
   - This innovation streamlines conservation efforts, transforming a two-week manual process into real-time monitoring.
- **Debate Over FP4 Metrics**: Members discussed the implications of NVIDIA's use of FP4 metrics amidst concerns over its comparative value against other formats like FP8.
   - The conversation highlighted the need for clear benchmarking standards in GPU performance claims.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/victortaelin/status/1876604185309048963?s=46">Tweet from Taelin (@VictorTaelin)</a>: So, let me get it right, RTX 5090 is ~50% more expensive, has ~50% more CUDA cores, and uses ~50% more energy than an RTX 4090, for ~50% increased performance. What exactly is the real gain here? Othe...</li><li><a href="https://x.com/DrJimFan/status/1876516972512559170.">Tweet from Jim Fan (@DrJimFan)</a>: Introducing NVIDIA Cosmos, an open-source, open-weight Video World Model. It&#39;s trained on 20M hours of videos and weighs from 4B to 14B. Cosmos offers two flavors: diffusion (continuous tokens) an...</li><li><a href="https://x.com/yuchenj_uw/status/1876680855630094725?s=46">Tweet from Yuchen Jin (@Yuchenj_UW)</a>: I love Nvidia and Jensen, but their presentation of numbers bothers me:- vague terms like &#34;AI TOPS&#34;- compare FP4 on 5090 with FP8 on 4090- show FP4 FLOPS and claim a $3,000 box runs a 200B mod...</li><li><a href="https://x.com/chipro/status/1876681640505901266">Tweet from Chip Huyen (@chipro)</a>: My 8000-word note on agents: https://huyenchip.com//2025/01/07/agents.htmlCovering:1. An overview of agents2. How the capability of an AI-powered agent is determined by the set of tools it has access ...</li><li><a href="https://x.com/kevinhou22/status/1876498858819424307">Tweet from Kevin Hou (@kevinhou22)</a>: holy cow Jensen talking about us @ CES 🤯&#34;Codeium. Every software engineer in the world, this is going to be the next giant AI application...Everybody is going to have a software assistant. If not...</li><li><a href="https://x.com/joaomdmoura/status/1876645077726216336">Tweet from João Moura (@joaomdmoura)</a>: ⚡️Major Announcement⚡️One of tech&#39;s biggest players just made their move in the AI Agent space.NVIDIA is collaborating with CrewAI to power their enterprise AI deployment.Here&#39;s why this is a ...</li><li><a href="https://x.com/btibor91/status/1876630816199217208?s=46">Tweet from Tibor Blaho (@btibor91)</a>: The real-world impact of AI - The University of Sydney and Accenture built a whale conservation system that uses Claude to analyze underwater microphone recordings and detect minke whales with 89.4% a...</li><li><a href="https://x.com/joshua_xu_/status/1876707348686995605">Tweet from Joshua Xu (@joshua_xu_)</a>: We have incorporated HeyGen&#39;s avatar model to work seamlessly with Sora, and the results are truly next-level. This is probably the most advanced talking avatar video to date—outperforming real ac...</li><li><a href="https://huyenchip.com//2025/01/07/agents.html">Agents</a>: Intelligent agents are considered by many to be the ultimate goal of AI. The classic book by Stuart Russell and Peter Norvig, Artificial Intelligence: A Modern Approach (Prentice Hall, 1995), defines ...</li><li><a href="https://mindy.com/">m@mindy.com</a>: no description found</li><li><a href="https://olly.bot">Olly | Personal AI Assistant</a>: Your personal AI assistant in iMessage. Available on your iPhone, Watch, Macbook, or CarPlay via Siri. Web-powered answers, image generations, chat with documents, reminders and more.</li><li><a href="https://x.com/abhagsain/status/1876362355870994538">Tweet from Anurag Bhagsain (@abhagsain)</a>: Last week, we asked Devin to make a change. It added an event on the banner component mount, which caused 6.6M @posthog events in one week, which will cost us $733 Devin cost $500 + $733 = $1273 😢👍L...</li><li><a href="https://github.com/mastra-ai/mastra">GitHub - mastra-ai/mastra: The TypeScript AI framework.</a>: The TypeScript AI framework. Contribute to mastra-ai/mastra development by creating an account on GitHub.</li><li><a href="https://x.com/calcsam/status/1859664608992297082">Tweet from Sam Bhagwat (@calcsam)</a>: Excited to share that @smthomas3, Abhi Aiyer and I are building Mastra, a Typescript AI framework for the next million AI developers:
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1326140999210106880)** (3 messages): 

> `Modular docs font weight, Font readability` 


- **Discussion on Modular Docs Font Weight**: A member expressed that the **Modular docs font weight** feels too thin, prompting others to weigh in on the issue.
   - Another member agreed, stating they have disliked the current font since it changed and suggested **Modular** should consider a different font or weight.
- **Concerns about Font Readability**: Multiple members highlighted concerns about the readability of the **Modular docs** font since its change.
   - This led to suggestions for **Modular** to explore alternative font weights to enhance user experience.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1326163405945438208)** (37 messages🔥): 

> `Mojo Debugger, Mojo Project Structure, Static Lists in Mojo, Indexing with Runtime Variables, Static Analysis Methods` 


- **Mojo Debugger Uses LLDB**: Members discussed that Mojo uses **LLDB** with upstream patches to enable it to work with multiple languages, and mentioned a [talk from the LLVM conference](https://www.youtube.com/watch?v=9jfukpjCPIg) covering the debugger.
   - One member appreciated the pragmatic approach Modular takes, avoiding efforts on problems already solved.
- **Organizing Mojo Projects**: A member inquired about organizing their Mojo project structure and how to import modules during testing, leading to a shared example from [GitHub](https://github.com/saviorand/lightbug_http/blob/main/tests/lightbug_http/test_client.mojo).
   - Another member explained how to run tests using `magic run mojo test -I . tests` and referred to the official [Mojo testing documentation](https://docs.modular.com/mojo/tools/testing#writing-unit-tests) for more details.
- **Indexing Static Lists in Mojo**: A user learned that **ListLiteral** cannot be indexed with runtime variables and should use **InlineArray** instead, which was successfully implemented in their case.
   - Further discussion clarified the differences between tuples and list literals, highlighting that tuples are fixed-length and can contain different types.
- **Borrow Checker Discussions**: A member proposed that Mojo should expand on static analysis methods to outshine Rust's borrow checker, suggesting a focus on resolving features first.
   - They expressed uncertainty about implementing a production-grade structure in Mojo, indicating a desire to explore the documentation further.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/tree/v0.3?tab=readme-ov-file#include-numojos-path-for-compiler-and-lsp">GitHub - Mojo-Numerics-and-Algorithms-group/NuMojo at v0.3</a>: NuMojo is a library for numerical computing in Mojo 🔥 similar to numpy in Python. - GitHub - Mojo-Numerics-and-Algorithms-group/NuMojo at v0.3</li><li><a href="https://docs.modular.com/mojo/tools/testing#writing-unit-tests">Testing | Modular Docs</a>: Testing Mojo programs.</li><li><a href="https://github.com/saviorand/lightbug_http/blob/main/tests/lightbug_http/test_client.mojo">lightbug_http/tests/lightbug_http/test_client.mojo at main · saviorand/lightbug_http</a>: Simple and fast HTTP framework for Mojo! 🔥. Contribute to saviorand/lightbug_http development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1325999017548713985)** (7 messages): 

> `AI-Plans Hackathon, Best AI Models, Command R+ Performance, AI Alignment Research` 


- **AI-Plans Hackathon Launches**: There's an upcoming hackathon hosted by **AI-Plans** focused on **AI Alignment Evals** scheduled for **January 25th**.
   - The event aims to engage participants in crucial **mechanistic interpretability** research along with a literature review.
- **Debating the Best Model for Tasks**: A discussion emerged around determining the **best AI model**, particularly in comparison to **OpenAI O1**.
   - *Competent* noted that the choice largely depends on the user's intended tasks and applications.
- **Command R+ Dominates Logical Reasoning**: Members concluded that the top-performing model for logical reasoning is **Command R+08**, excelling in complex question scenarios.
   - It was noted that while it handled simpler questions adequately, it significantly outperformed others like **Sonnet 3.5** in more challenging contexts.
- **Complex Question Handling**: The consensus among users showcases that **Command R+** handles complex inquiries more robustly compared to **Command R08** and others.
   - As shared by members, the effective performance drops when posed with simpler questions, highlighting the importance of question complexity.


  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1325999150696759417)** (2 messages): 

> `Evals, Mechanistic Interpretability, Object Detection in AR` 


- **Interest in Evals and Mechanistic Interpretability**: A member expressed interest in connecting with others about **Evals** or **Mechanistic Interpretability**.
   - *Hi! Any folk here interested in Evals or Mech Interp?*
- **Seeking AR Project Insights**: Another member is looking for research related to an **AR** project that can detect **planes** and classify **objects**.
   - *If there is anybody who knows, please tell me.*


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1326268661022851167)** (4 messages): 

> `Embed API Usage, Response Structure, Image Encoding` 


- **Using Embed API for Image Input**: A user shared a snippet to use the [embed API](https://docs.cohere.com/reference/embed) with image data, encoding an image fetched from a URL to base64 format.
   - The code demonstrates how to prepare and send image data for embedding using the **cohere.ClientV2**.
- **Clarification on Embedding Responses**: A user inquired whether the embedding response maintains the **same order** as the list of texts sent in a request.
   - Another user confirmed that the embeddings will indeed return in the **same order** as the request, alleviating concerns about matching text with its embedding.
- **Image Data Handling in Embedding**: Discussion included details about retrieving image content and transforming it into a **base64** encoded string for embedding.
   - The focus was on ensuring correct content type headers are handled to pass image data appropriately.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://]">no title found</a>: no description found</li><li><a href="https://docs.cohere.com/reference/embed">Embed — Cohere</a>: This endpoint returns text embeddings. An embedding is a list of floating point numbers that captures semantic information about the text that it represents.Embeddings can be used to create text class...
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1325999358600024065)** (16 messages🔥): 

> `Neural Network in JavaScript, Discord Restart Issues, Cohere Billing Policies` 


- **Build a Neural Network in Pure JavaScript**: A user requested the creation of a **pure JavaScript script that functions as a neural network from scratch**.
   - No further details or examples were provided regarding the implementation.
- **Discord Restarting on Ctrl + R**: A user inquired why Discord restarts when pressing **'ctrl + R'**.
   - The Cmd R Bot attempted to look up documentation but returned without an answer, indicating a lack of information.
- **Threshold for Charges in USD**: A user asked about the **threshold in USD for a single charge** but received no information directly related to the query.
   - The Cmd R Bot found details about billing policies, stating that a charge occurs once a self-serve customer accumulates **$250** in outstanding debts.
- **Cohere Billing Policy Details**: According to Cohere documentation, a **warning email** is sent when a user accrues **$150** of outstanding debts.
   - Once debts reach **$250**, a charge is automatically processed via **Stripe** for self-serve customers.
- **Source for Billing Policies**: The information regarding billing policies can be found in the [Release Notes for June 10th 2024](https://docs.cohere.com/v1/changelog/release-notes-for-june-10th-2024).
   - The release notes detail updates to tool use, SDKs, and billing practices.


  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1325919901046804501)** (4 messages): 

> `AR projects for object detection, Live AR asset implementation` 


- **Request for AR project assistance**: A member is seeking support for a project to detect planes and argue objects in augmented reality, calling for anyone with knowledge to contribute.
   - The enthusiasm for AR applications was echoed by another member who appreciated the concept, emphasizing how useful a reranker for live AR assets would be.
- **Interest in AR x Cohere collaboration**: Another member expressed excitement about the potential collaboration between AR and Cohere for live asset utilization, showcasing a strong wish for implementation.
   - They remarked that it would be 'totally sick to see' such innovations come to fruition, indicating a vibrant interest in applied AR technologies.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1326138230101250140)** (10 messages🔥): 

> `Array Operations in Triton, Config Management in Projects, Performance of MMAs with wgmma, Memory Layout and Data Movement, Kernel Compilation and Autotuning` 


- **Array Slicing Performance Mystery**: *Theoboyer* expressed confusion over the significant speed difference in **expand_dims** versus **.reshape** when manipulating arrays, especially regarding the `can_reorder` functionality.
   - The question centered around whether `can_reorder` allows faster computation by reordering data or dimensions and if it can control reordering.
- **Unique Dimension Handling with Powers of Two**: *Mobicham* described their approach of restricting dimensions to powers of two, implementing caching to skip autotuning when shapes match a cached configuration.
   - They mentioned that although compilation takes around **0.06s** per shape, it's manageable during the prefill phase.
- **Kernel Efficiency with Autotuning Strategies**: *Latkins* mentioned using `CLOSEST_M` in their kernel strategy, allowing recompilation on size changes while avoiding autotuning for better performance on large sizes.
   - They noted that autotuning might not be practical when performance drops outside expected parameters.
- **Using wgmma for MMAs on H100**: *Danielkoceja8071* asked how to ensure usage of **wgmma** for MMAs, noting their kernel's PTX only shows **mma.sync** despite using an **H100**.
   - The question implied a need for clarity on kernel configurations to utilize the latest features effectively.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1325937966304071733)** (1 messages): 

> `Output fragment register layout, WMMA loading and storing, Experimenting with matrix copying` 


- **Output fragment retains input fragment layout**: Output fragments reportedly maintain the same register layout as input fragments, with indices such as **[0,1][0,2]** and **[8,0][8,1]** consistent across operations.
   - A user mentioned testing this behavior with successful results when using WMMA to load from matrix A and store back to matrix B.
- **WMMA effectively copies matrices**: The loading and storing process in **WMMA** should copy matrix A to B while preserving the layout, according to user experimentation.
   - The user expressed willingness to provide a runnable example should others require further clarification.
- **User experimenting with WMMA**: One member shared insights on WMMA while clarifying that they had moved on from this exploration after their tests.
   - They expressed a light-hearted willingness to assist others with examples from their experiments.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1325959467950674062)** (2 messages): 

> `Custom Autograd Functions, Guard Failures in PyTorch` 


- **Custom Autograd function modifying gradients**: A member questioned whether it is acceptable for their custom **autograd function** to modify gradients in-place, despite the documentation warning against it. They observed that their model gradients matched closely with those from a simpler implementation that does not customize autograd.
   - For reference, they included a link to the [PyTorch documentation on extending autograd](https://pytorch.org/docs/main/notes/extending.html#how-to-use).
- **Seeking Verbose Logs for Guard Failures**: Another member faced challenges obtaining verbose output on **guard failures**, describing their logs as insufficiently informative. They speculated that the **0/0** error message could indicate a missing message related to the encountered failure.
   - They were running with the command `TORCH_LOGS="+dynamo,guards,bytecode,recompiles,recompiles_verbose"` to increase log detail.



**Link mentioned**: <a href="https://pytorch.org/docs/main/notes/extending.html#how-to-use>">Extending PyTorch &mdash; PyTorch main documentation</a>: no description found

  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1325975052901941259)** (3 messages): 

> `Picotron framework, DeepSeek-v3 paper, LLM infrastructure videos` 


- **Introducing Picotron for 4D Parallelism**: The [Picotron framework](https://github.com/huggingface/picotron) offers a **minimalistic 4D-parallelism** distributed training solution designed for educational purposes, enabling users to explore advanced training techniques.
   - Its GitHub repository highlights its utility and aim for facilitating learning in AI training methodologies.
- **DeepSeek-v3 Paper Insights**: *Ten short videos* were shared to enhance understanding of Pages **12-18 of the DeepSeek-v3 paper** ([arXiv link](https://arxiv.org/abs/2412.19437)).
   - These resources aim to clarify complex LLM infrastructure concepts presented in the paper for a broader audience.
- **Video Playlist on LLM Infrastructure**: A YouTube playlist was recommended featuring **short videos** that cover essential aspects of LLM infrastructure and are relevant for those engaging with advanced AI research topics.
   - The first video in the playlist can be found [here](https://www.youtube.com/watch?v=76gulNlhiE4&list=PLO45-80-XKkT6BUKCYeBMTEqnlcpYavxq&index=1&pp=gAQBiAQB).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/srush_nlp/status/1876640795765379531">Tweet from Sasha Rush (@srush_nlp)</a>: 10 short videos about LLM infrastructure to help you appreciate Pages 12-18 of the DeepSeek-v3 paper (https://arxiv.org/abs/2412.19437) 🧵https://www.youtube.com/watch?v=76gulNlhiE4&list=PLO45-80-XKkT...</li><li><a href="https://github.com/huggingface/picotron">GitHub - huggingface/picotron: Minimalistic 4D-parallelism distributed training framework for education purpose</a>: Minimalistic 4D-parallelism distributed training framework for education purpose - huggingface/picotron
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1325945667520167976)** (3 messages): 

> `Journey sharing, ONNX to TensorRT conversion issues` 


- **Excitement for Sharing Experiences**: A member expressed enthusiasm about sharing their journey and asked for more details regarding resources used.
   - Another member showed support with a positive reaction.
- **Troubles with ONNX to TensorRT Conversion**: A member reported having trouble with ONNX to TensorRT conversion, noting discrepancies in outputs.
   - They highlighted that the **TensorRT engine output** does not match the output from the **PyTorch model**.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1326054703154593874)** (4 messages): 

> `Nvidia's Project DIGITS, Grace Blackwell Superchip, Training Small Models` 


- **Nvidia's Project DIGITS brings AI power to your desk**: Nvidia has introduced [Project DIGITS](https://www.nvidia.com/en-us/project-digits/), featuring the **Grace Blackwell Superchip**, delivering petaflop AI performance in a compact design.
   - Developers can now prototype and run large AI models of up to **200B parameters** locally with **128GB of unified memory**.
- **Excitement over Project DIGITS capabilities**: A member expressed enthusiasm, noting that with the new **tensor cores**, fp4 and f8 will soon become standard for training models.
   - They pondered whether the capabilities of this system will be sufficient for training smaller models despite Nvidia's claim of supporting a **200B model**.



**Link mentioned**: <a href="https://www.nvidia.com/en-us/project-digits/">NVIDIA Project DIGITS: The World’s Smallest AI Supercomputer. </a>: Reserve yours today.

  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1326123623848742933)** (3 messages): 

> `hipDeviceAttributeMaxBlocksPerMultiProcessor, CUDA vs HIP attributes comparison, AMD hardware max occupancy, Thread block discussions` 


- **Clarification on hipDeviceAttributeMaxBlocksPerMultiProcessor**: A member raised a question on comparing `hipDeviceAttributeMaxBlocksPerMultiProcessor` with the CUDA equivalent from the [CUDA documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_17f337476973ea65db85c277f4646f0b3). They speculated that achieving **2048 threads/SM** required two thread blocks of 1024 threads, finding it perplexing.
- **Skepticism about the max blocks computation**: Another member shared a link to the [hip_device.cpp file](https://github.com/ROCm/clr/blob/b8ba4ccf9c53f6558a5e369e3c1c05de97a0c28f/hipamd/src/hip_device.cpp#L496) and expressed doubt regarding the reliability of the computation for max blocks per multiprocessor. They noted a sense of uncertainty in the results.
- **Max occupancy under AMD hardware varies**: A member confirmed that on AMD hardware, the max occupancy can vary between **8, 10, or 20** depending on the generation. They indicated uncertainty about how much the **workgroup size** influences these values.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_17f337476973ea65db85c277f4646f0b3">CUDA Runtime API :: CUDA Toolkit Documentation</a>: no description found</li><li><a href="https://github.com/ROCm/clr/blob/b8ba4ccf9c53f6558a5e369e3c1c05de97a0c28f/hipamd/src/hip_device.cpp#L496">clr/hipamd/src/hip_device.cpp at b8ba4ccf9c53f6558a5e369e3c1c05de97a0c28f · ROCm/clr</a>: Contribute to ROCm/clr development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1326304924220522617)** (5 messages): 

> `Discord based leaderboard, GPU Glossary resources` 


- **Alpha Users Needed for Discord Leaderboard!**: A member announced they are seeking **alpha users** for a new **Discord based leaderboard** that connects GPUs to facilitate competition on specific kernels.
   - *If this sounds remotely interesting*, they encouraged replies for a tutorial.
- **Release of GPU Glossary Materials**: Another user shared that `gpu-glossary.zip` contains all GPU Glossary materials formatted as **Markdown files**, with associated URLs in `contents.json`, acting as a ToC.
   - This zip file can be accessed directly through [this link](https://cdn.discordapp.com/attachments/1298372518293274644/1326364815421341696/gpu-glossary.zip?ex=677f28fe&is=677dd77e&hm=7bc8bcc2212219fa43b55a4dfe342017480bdf85ff2ffb599d30feb4ba302935&).


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1325957436208578641)** (4 messages): 

> `LlamaIndex and MLflow integration, Multi-agent systems with NVIDIA AI, Cohere models usage with LlamaIndex` 


- **Streamline with LlamaIndex and MLflow Integration**: A step-by-step guide outlines how to combine **LlamaIndex**, **MLflow**, **Qdrant**, and **Ollama** for enhanced vector storage and model tracking. Integrating these tools allows for efficient **real-time** operations and evaluations, as detailed in the [full guide](https://t.co/lNDDgdOo86).
   - This integration emphasizes utilizing **Change Data Capture** alongside these technologies for improved workflows.
- **NVIDIA AI Powers New Multi-Agent System**: A newly launched blueprint for a multi-agent system was unveiled at CES, leveraging **NVIDIA AI** to assist in researching and writing blog posts. This system aims to mitigate the **time sink** caused by content creation tasks, allowing LLMs to undertake complex research efficiently.
   - Check out the details in the official announcement [here](https://t.co/eS4BhuAZVS).
- **Cohere Models Get a Fresh Look**: The team applauds **Cohere's** models for their powerful embedding capabilities and recent documentation updates for integrating with **LlamaIndex**. Installation instructions and prerequisites were shared, ensuring users can fully utilize both **Cohere’s** SDK and LlamaIndex functionalities together.
   - For more information and installation steps, refer to the [documentation](https://t.co/dLKGgkqOe8).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://t.co/dLKGgkqOe8">LlamaIndex — Cohere</a>: Learn how to use Cohere and LlamaIndex together to generate responses based on data.</li><li><a href="https://t.co/eS4BhuAZVS">Document Research Assistant for Blog Creation Blueprint by Llamaindex | NVIDIA NIM</a>: Automate research, and generate blogs with AI Agents using LlamaIndex and Llama3.3-70B NIM LLM.
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1326091513616269354)** (9 messages🔥): 

> `LlamParse Error, LlamIndex Tutorial Notebook, Text-to-SQL Capabilities, Documentation Links` 


- **LlamParse encounters first-time parsing error**: A user reported receiving an error while parsing a PDF file with LlamParse for the first time, but it worked fine on subsequent attempts.
   - Another user asked for clarification on whether the error happens every time for the same file and sought to review the PDF file in question.
- **LlamaIndex tutorial notebook link is broken**: A user inquired about the notebook for a specific LlamaIndex tutorial, noting that the link provided in the documentation was dead.
   - Another user shared a working link to the correct notebook and mentioned that it might just be missing from the navigation.
- **Text-to-SQL explained in LlamaIndex**: The documentation covers LlamaIndex’s capabilities in creating structured data from unstructured sources, alongside text-to-SQL functionalities.
   - A safety note was included about the risks of executing arbitrary SQL queries, recommending sound practices.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/structured_data/">Structured Data - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/index_structs/struct_indices/SQLIndexDemo/">Text-to-SQL Guide (Query Engine + Retriever) - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/index_structs/struct_indices/SQLIndexDemo.ipynb">llama_index/docs/docs/examples/index_structs/struct_indices/SQLIndexDemo.ipynb at main · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1326187761366995039)** (10 messages🔥): 

> `Open Interpreter 1.0 Release, Archiving of Classic OI, Issues with pip installation, Modifications and PR submissions, Local Model Performance` 


- **Open Interpreter 1.0 Release Approaches**: The latest GitHub commit indicates that **Open Interpreter 1.0** is nearly here, but it currently cannot run code, causing confusion among users.
   - Documentation on the changes and roadmap for 1.0 hasn't been clearly outlined.
- **Classic Open Interpreter Archived**: The classic version of OI has been archived, with all previous prompts now stored in an outdated folder, limiting user contributions.
   - Users noted difficulties in submitting PRs due to the archived status of the classic version.
- **Pip Installation Issues Raised**: A user mentioned that installing the stable version using `pip install open-interpreter` hasn't been functioning as desired.
   - There's uncertainty about how to enhance the existing version, as modifications lead to further confusion.
- **Modification Limitations and Confusion**: Users expressed a desire to improve the tools and prompts for better functionality but are confused by the transition to version 1.0.
   - Some modifications are problematic since, while working on 1.0, older PR submissions are not an option.
- **Local Model Performance Concerns**: Advice was given to use `--no-tool-calling` for local models, suggesting some changes to the system prompt may have negative impacts.
   - Discussion highlighted difficulties that smaller models face due to the adjustments made in the new version.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/OpenInterpreter/open-interpreter/commit/21babb186f13e263a72cf525d15d79788edf4644">Open Interpreter 1.0 Preview · OpenInterpreter/open-interpreter@21babb1</a>: no description found</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/commit/275105771dfa985ee777e4a9c6e9c4d760c7b7b9">Archived Interpreter Classic · OpenInterpreter/open-interpreter@2751057</a>: no description found
</li>
</ul>

</div>
  

---


### **Axolotl AI ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1326024960493486134)** (8 messages🔥): 

> `GH200 Utilization, Compilation Challenges, Discord Link Issues` 


- **GH200 User Might Offer Assistance**: A user noticed that <@201777246367645696> is using **GH200** and suggested they might be able to provide help.
   - *It's hoped that collaboration on this might alleviate some of the challenges others face.*
- **Compilation Takes Time with Dependencies**: Another user expressed frustrations over **dependencies** causing delays in getting the setup to compile fully.
   - *They mentioned that while getting it to work is possible, it requires a considerable time investment.*
- **Discord Link Guy's Return Causes Stir**: The infamous **discord link guy** resurfaced, posting potentially unwanted links in various channels, prompting warnings.
   - A user reported this issue and later confirmed the **banning** and removal of a problematic welcome channel message.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1326162805392674949)** (7 messages): 

> `MiPROv2 Instructions Flow, Integration of dspy with Langchain` 


- **MiPROv2 could adapt instruction generation**: A member suggested that instead of writing a complete set of instructions upfront, **MiPROv2** could trial one instruction at a time and adapt based on outcomes.
   - This approach could leverage an LLM as a judge to critique outputs, providing valuable feedback on improving instructions.
- **dspy.COPRO! as a related concept**: Another member pointed out that the proposed MiPROv2 method would be closer to using **dspy.COPRO!** for instruction generation and trial.
   - This elicited interest, with members expressing intentions to explore the concept further.
- **Challenges in integrating dspy with Langchain**: A member asked about integrating **dspy** with **Langchain** using version 2.6, indicating interest in creating LLM agents.
   - A response detailed that there is no straightforward way to combine when working with these two frameworks.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1326261516261134521)** (1 messages): 

> `Certificate Declaration, Assignment Completion, Certificate Deadlines` 


- **Certificate Declaration Form Reopened**: The **Certificate Declaration form** has been reopened for those who completed all assignments back in December, allowing them to receive a certificate.
   - *Tentatively due* by the **end of January**, participants must fill out the form as no past assignments will be reopened.
- **One Certificate Limit**: Students are reminded that they can only earn **one certificate** for this course, ensuring clarity on certification eligibility.
   - This limitation emphasizes the necessity of completing all required assignments for certification.
- **Potential Delays in Certificate Issuance**: Certificates are intended to be sent out by the **end of January**, but there may be delays for those now filling out the declaration form.
   - Participants are urged to complete their forms promptly to avoid delays in receiving their certificates.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1325966328372265030)** (5 messages): 

> `Declaration Form Acknowledgment, Email Address Consistency for Submissions` 


- **Gratitude for Declaration Form Reopening**: <@854134294870884363> received appreciation from members for reopening the **declaration form**, highlighting its importance.
   - Members expressed their gratitude and acknowledged the positive impact of this action on the submission process.
- **Importance of Consistent Email Addresses**: <@tarande57> emphasized that the email for the declaration form must be the **same** as the one used for course assignments to receive a certificate.
   - This policy ensures accurate tracking of submissions, making it critical for participants to adhere to it.
- **Confirmation Request after Email Change**: @iamkrish10 inquired if their original email address used for assignments was acknowledged since they submitted the form with a different email.
   - They highlighted that they noted the original email in the submission text box and requested verification for their submission.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1325938386590109747)** (5 messages): 

> `Reasoner v1 capabilities, Local Docs indexing issue, Embedding model support` 


- **Reasoner v1 praised for its functionality**: A member praised the work on **Reasoner v1** and inquired about other models or templates that work with reasoning mode, besides the Qwen 2.5 coder variant.
   - Another member confirmed that **OpenAI compatible remote models** and several local models work, indicating they are adding more models that function out of the box.
- **Local Docs struggle with directory indexing**: One member reported issues with local docs not embedding files from subdirectories effectively, despite initial successful indexing.
   - They noted that the problem seemed related to how timestamps are used, potentially leaving some **LocalDocs** without any files embedded if they had already been included in another document.
- **Query on embedding model replacement**: Another member expressed interest in replacing an existing model with a different embedder model and checked for its compatibility with **Text-embedding-inference** or **vLLM embedder**.
   - This inquiry highlighted the ongoing considerations for model flexibility and support among members working with embeddings.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1326255868408496240)** (1 messages): 

> `MLOps and Feature Stores Webinar, Integration of LLMs in MLOps, 2024 MLOps Developments, Trends and Challenges in 2025` 


- **Free Webinar on MLOps and Feature Stores**: Join the upcoming webinar on **January 15th** at **8 A.M. PT** featuring Ben Epstein and Simba Khadder discussing **MLOps** and **Feature Stores** for 2025. [Sign up here](https://buff.ly/4j9oiVg) to secure your spot!
   - The webinar will dive into **best practices** and **cutting-edge architectures** while allowing time for a Q&A session at the end.
- **Key Developments in MLOps through 2024**: The discussion will reflect on the major **developments in MLOps** throughout **2024** and outlook for 2025, emphasizing the roles of **Large Language Models (LLMs)**. Key trends influencing both **MLOps** and **LLMOps** will be highlighted.
   - Expect insights into **areas of convergence** and the evolving landscape of MLOps as it adapts to advancements in machine learning technologies.
- **Who Should Attend the Webinar**: The event is aimed at **Data Engineers**, **Data Scientists**, **Machine Learning Engineers**, and **AI/ML Enthusiasts** interested in the latest MLOps trends. It's an opportunity for professionals to gather insights from industry leaders.
   - Participants will get to engage directly with the speakers during the Q&A, enhancing their understanding of future MLOps strategies.



**Link mentioned**: <a href="https://buff.ly/4j9oiVg">MLOps and Feature Stores in 2025 with Ben Epstein</a>: Join our 1-hr webinar where Simba Khadder of Featureform and Ben Epstein of MLOps Community will chat about upcoming MLOps trends in 2025!

  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1326271196614098944)** (1 messages): 

> `LLM security testing, Harmful AI Assistant Challenge, GraySwanAI Arena` 


- **GraySwanAI Launches Harmful AI Assistant Challenge**: The new **Harmful AI Assistant Challenge** will launch on **January 4th at 1 PM EST**, offering **$40,000** in prizes for innovative prompt injection and jailbreaking methods.
   - Participants must find unique ways to elicit harmful responses from AI assistants, and **multi-turn inputs** are allowed in this competitive event.
- **Last Event Featured Pre-release Testing**: Earlier events provided participants the opportunity to test **o1 models** before their official release, as referenced in the **12/5 OAI paper**.
   - This ongoing series of events showcases the latest advancements in LLM security testing and community engagement.
- **Join the GraySwanAI Community**: Interested participants can sign up and join the community at [app.grayswan.ai](http://app.grayswan.ai/arena) or connect through Discord at [discord.gg/WqHkWt99](http://discord.gg/WqHkWt99).
   - This event aims to foster collaboration and skill development among enthusiasts in the field of AI security testing.



**Link mentioned**: <a href="https://x.com/GraySwanAI/status/1872720375328411668">Tweet from Gray Swan AI (@GraySwanAI)</a>: 🚨 New Arena Launch Alert: Harmful AI Assistant Challenge 🚨💰 $40,000 in Prizes📅 Launch Date: January 4th, 1 PM EST🤖 5 Anonymous Models🔥 Prizes for speed & quantity.🎮 Multi-turn Inputs AllowedYou...

  

---


### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1325984835914694766)** (1 messages): 

> `Common Voice AMA, 2024 Review, Voice Technology Accessibility` 


- **Common Voice's New Year AMA Kickoff**: Common Voice is launching their **2025 AMA** in their [new Discord server](https://discord.gg/b4c83ppxdU) to review the past year and engage with the community.
   - This session aims to answer all questions about the project's journey and future developments.
- **2024 Review and Q&A Session**: Participants are invited to join the team for a **2024 review** with key guests including the Product Director and a Frontend Engineer.
   - The event will include an interactive Q&A session to foster community engagement and feedback.
- **Promoting Accessibility in Voice Technology**: Common Voice aims to make voice technology **open and accessible**, providing essential data to developers for creating voice recognition systems.
   - The project highlights the importance of democratizing voice data that has been traditionally inaccessible to lower barriers to innovation.


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1326074798165725196)** (1 messages): 

> `Dolphin 3.0 Model Series, BFCL Leaderboard` 


- **Will Dolphin 3.0 grace the BFCL leaderboard?**: A member inquired if the **Dolphin 3.0 model series** will appear on the BFCL leaderboard, expressing interest in its performance.
   - They included a link to [Dolphin 3.0 on Hugging Face](https://huggingface.co/collections/cognitivecomputations/dolphin-30-677ab47f73d7ff66743979a3) for further details.
- **Cognitive Computations' Latest Update**: The **cognitivecomputations/Dolphin3.0-Llama3.2-1B** model was updated recently, gaining **34** stars on Hugging Face.
   - The post included an image showcasing the model and attracted attention with **14** comments.



**Link mentioned**: <a href="https://huggingface.co/collections/cognitivecomputations/dolphin-30-677ab47f73d7ff66743979a3">Dolphin 3.0 - a cognitivecomputations Collection</a>: no description found

  

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
