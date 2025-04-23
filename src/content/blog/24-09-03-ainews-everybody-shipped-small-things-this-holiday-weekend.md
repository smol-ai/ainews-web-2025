---
id: a00a923a-5c1d-4a39-af65-baa5017358b8
title: Everybody shipped small things this holiday weekend
date: '2024-09-04T01:35:37.812399Z'
original_slug: ainews-everybody-shipped-small-things-this
description: >-
  xAI has brought its Colossus 100k H100 cluster online, capable of training a
  GPT-4 class FP8 model in just 4 days, marking a significant advancement in
  large-scale AI training infrastructure. Meanwhile, AI21 Labs released Jamba
  1.5 models featuring a massive 256K context window and faster long-context
  performance, pushing the boundaries of model capabilities.
tags:
  - xai
  - colossus
  - h100
  - gpt-4
  - ai21labs
  - jamba-1.5
  - long-context
---


<!-- buttondown-editor-mode: plaintext -->**smol updates are all you need.**

> AI News for 9/2/2024-9/3/2024. We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**214** channels, and **2424** messages) for you. Estimated reading time saved (at 200wpm): **281 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Let's see:

- From xAI: [Colossus 100k H100 cluster was online](https://x.com/elonmusk/status/1830650370336473253). [Per Semianalysis](https://www.semianalysis.com/p/100000-h100-clusters-power-network?triedRedirect=true), this cluster can train an FP8 GPT-4 class (2e25 FLOPs) model in 4 days.
- From Google: [Gemini got Structured Output](https://x.com/OfficialLoganK/status/1829678117054792160)
- From Anthropic: [Dario was on a podcast](
https://youtu.be/7xij6SoCClI?feature=shared)
  - A lot of people calling out that [Claude is getting worse](https://x.com/nearcyan/status/1829674215492161569?s=46) and it is perhaps from [modifying prompts in the API](https://x.com/_philschmid/status/1830559304241287611?s=46). No official response yet
- From OpenAI: [enhanced controls for File Search in Assistants API](https://x.com/openaidevs/status/1829259020437475771?s=46)
- From Cognition: [Scott Wu on a podcast](https://share.snipd.com/episode/faaed93f-9297-4926-aa03-78643ea68d65)
- [the Kwai-Kolors virtual try-on model went viral](https://x.com/basedjensen/status/1829763446763896903?s=46)
- [Mini-Omni]( https://x.com/osanseviero/status/1830875530209513587?s=46), an open source real-time audio conversational model, was released. Similar to GPT4o Voice.

Since it's a quiet day, you could think about the [broader trend of commoditization of intelligence](https://x.com/latentspacepod/status/1831020483967701260) from your friendly neighborhood AI Engineering podcast.

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

---

**AI Productivity Enhancement and Fine-Tuning**

- **Parameter-efficient fine-tuning**: [@fchollet](https://twitter.com/fchollet/status/1826674137089409377) shared a tutorial on parameter-efficient fine-tuning of LLMs with LoRA and QLoRA, highlighting how to enable QLoRA with a simple script. **"gemma_lm.quantize('int8')"**
- **Long-context embedding challenges**: [@JinaAI_](https://twitter.com/JinaAI_/status/1826649449919369726) discussed the **"Lost Context Problem"** in naive chunking-embedding pipelines of RAG systems and introduced the "Late Chunking" approach.
- **Claude enhancements**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1826667671364272301) announced the addition of **LaTeX rendering** in Claude's feature preview to improve the display of mathematical equations.

**High-Performance Model Releases**

- **Jamba 1.5 Models**: [@AI21Labs](https://twitter.com/AI21Labs/status/1826607933167469024) released Jamba 1.5 Mini & Large, featuring **256K context window**, **2.5x faster** long-context performance, and **JSON output** among other tools. **"The first mamba-hybrid being able to compete with top performers"** noted [@Yampeleg](https://twitter.com/Yampeleg/status/1826642273669544363).
- **Mistral-NeMo-Minitron-8B**: [@NVIDIA](https://twitter.com/clefourrier/status/1826672319970115887) debuted as the first Nvidia model on the Open LLM Leaderboard, outperforming other models significantly in various benchmarks.

**Enhanced Collaboration Tools and Frameworks**

- **LangSmith Workspace Organization**: [@LangChainAI](https://twitter.com/LangChainAI/status/1826643491130421476) introduced **resource tags** to manage projects, datasets, and prompts efficiently. **"Organize your workspace in LangSmith with resource tags."**
- **Low-Code Toolkit for AI Apps**: [@svpino](https://twitter.com/svpino/status/1826590311948452035) provided an open-source, self-hosted AI starter kit, including **n8n for workflow automation**, **Ollama for local model hosting**, and **Qdrant for vector storage**. **"Bootstrap a fully-featured low-code development environment to build AI applications."**

**AI in Legal and Financial Domains**

- **AI Legal Agents**: [@SpellbookLegal](https://twitter.com/scottastevenson/status/1826611628852609551) launched Spellbook Associate, an AI agent that **breaks down legal projects into plans**, executes tasks, and reviews work. **"An electric bicycle for lawyers."**
- **LangSmith Evaluations**: [@virattt](https://twitter.com/virattt/status/1826621769371021564) added evaluations to a Warren Buffett financial agent, using LangSmith to set up and visualize evaluations efficiently.

**Performance Optimization and Real-World Implementation**

- **Phi-3.5 Vision**: [@Microsoft](https://twitter.com/mervenoyann/status/1826640879995813925) introduced the Phi-3.5 vision models, surpassing existing benchmarks. **"4.2B model, 128k token context length"**
- **Neuralink Gaming**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1826619574651171148) shared progress on Neuralink trials, where participants control game elements with their minds, hinting at near-future applications in gaming and other sectors. **"Mind will be the ONLY constraint."**

**Memes/Humor**

- [@swyx](https://twitter.com/swyx/status/1826673468223688956): "RT [@latentspacepod](https://twitter.com/latentspacepod): Is finetuning GPT4o worth it?"
- [@rez0__](https://twitter.com/rez0__/status/1826671312330523118): "Okay, I give up. I'm a believer now. This is like the 'here's what my wife's scandal taught me about B2B sales' LinkedIn parody, but real."
- [@goodside](https://twitter.com/goodside/status/1826651729443827805): "It's a fun place to visit but you don't want to live there."

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Star Command R 32B v1: New Release from TheDrummer**

- **[Drummer's Coo- ... *ahem* Star Command R 32B v1! From the creators of Theia and Rocinante!](https://huggingface.co/TheDrummer/Star-Command-R-32B-v1)** ([Score: 47, Comments: 14](https://reddit.com//r/LocalLLaMA/comments/1f71b1j/drummers_coo_ahem_star_command_r_32b_v1_from_the/)): **Star Command R 32B v1**, a new AI model created by **TheDrummer**, the developer behind **Theia** and **Rocinante**, has been released. This model, described as a **32 billion parameter** AI, is positioned as a competitor to other large language models in the field, though specific performance metrics or comparisons were not provided in the announcement.
  - Users joked about **TheDrummer's** tamer model naming, with one comparing it to "*a porn star going mainstream, or a wrestler entering politics*". The developer responded with a humorous gif.
  - The **GGUF version** of the model is available on [Hugging Face](https://huggingface.co/TheDrummer/Star-Command-R-32B-v1-GGUF). Some users expressed interest in potential future models, including a hypothetical **104B Command-R-Sutra**.
  - Discussions touched on the model's potential for generating explicit content, with users speculating about its capabilities based on **TheDrummer's** reputation for creating models with such features.

**Theme 2. Community-Driven Free AI Server with Ollama**

- **I made my own local AI , u can use it for free ,** ([Score: 37, Comments: 52](https://reddit.com//r/LocalLLaMA/comments/1f711c3/i_made_my_own_local_ai_u_can_use_it_for_free/)): The user created a **local AI server** using **Ollama**, featuring **Llama 3.1** for current information, **Llama 3 (dolphin)** for unrestricted AI, and **LLava** for image recognition. The server is available for free public use at [evaai.ngrok.app](http://evaai.ngrok.app/), with the creator seeking assistance for fine-tuning, improving accessibility, and maintaining server operations through donations.
  - The creator expressed interest in adding **tools** like **image generation** to the server, potentially using **Stable Diffusion**. Users can find tools and functions in the **Workspace** panel of **open-webui**.
  - A suggestion was made to join **The Horde**, a **crowd-sourced computing network** for LLM/SD use without GPUs. The creator showed interest but expressed concerns about resource management and limitations.
  - Regarding **privacy**, the server doesn't verify emails, allows registration with fake emails, and offers options to delete chats and user data. The system runs on a **3070 GPU**, achieving **75 tokens/second**.

**Theme 3. Comparing Small Vision LLMs for OCR and Complex Layout Understanding**

- **Best small vision LLM for OCR?** ([Score: 31, Comments: 17](https://reddit.com//r/LocalLLaMA/comments/1f71k60/best_small_vision_llm_for_ocr/)): The post discusses the performance of **small vision Language Learning Models (LLMs)** for **Optical Character Recognition (OCR)**, particularly for **complex document structures** like resumes and invoices. The author found **InternVL 1.5** to be highly effective and relatively fast, while **Phi Vision** was more powerful but slower, and mentions using **PaddleOCR** for simpler cases. They also note that **Florence-2** excels at object detection and image description, and provide a [link to an open VLM leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) for reference.
  - **Surya OCR** is recommended for pure OCR tasks, with users reporting it outperforms **PaddleOCR** for handwritten text recognition. The [Surya GitHub repository](https://github.com/VikParuchuri/surya/tree/master) is available for implementation.
  - **Qwen2-vl** (especially the 7B model) is praised for OCR capabilities, even outperforming larger models like **internvl2-8b** in some tests. Users note that while OCR models extract text faster, **VLMs** can extract structured data more effectively.
  - **Kosmos-1.5** from Microsoft is highlighted for its OCR capabilities and ability to output in **markdown format**. However, some users prefer **Marker**, another open-source tool by **VikPachuri**, for markdown output and overall OCR performance.

## All AI Reddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI Model Development and Infrastructure**

- **xAI's Colossus training cluster**: xAI has brought online a **100,000 H100 GPU training cluster** called Colossus, which will [double to 200,000 GPUs in the coming months](https://x.com/elonmusk/status/1830650370336473253?s=46).

- **OpenAI's custom chip development**: OpenAI is [developing its first in-house chip with TSMC](https://wccftech.com/openai-developing-custom-chip-on-tsmc-a16-angstrom-process/) on the A16 Angstrom process, specifically for Sora video applications.

- **Google DeepMind's multimodal learning**: A [Google DeepMind paper](https://arxiv.org/html/2406.17711v1) demonstrates how data curation via joint example selection can accelerate multimodal learning.

- **Microsoft's MInference**: [Microsoft's MInference technique](https://arxiv.org/abs/2407.02490) enables inference of up to millions of tokens for long-context tasks while maintaining accuracy, dramatically speeding up supported models.

**AI Model Releases and Improvements**

- **Salesforce's xLAM-1b**: Salesforce released xLAM-1b, a 1 billion parameter model that [achieves 70% accuracy in function calling, surpassing GPT 3.5](https://www.reddit.com/r/LocalLLaMA/comments/1dz8g10/salesforce_tiny_giant_xlam1b_model_surpasses_gpt/).

- **Phi-3 Mini update**: Rubra AI released an updated Phi-3 Mini model [with function calling capabilities](https://www.reddit.com/r/LocalLLaMA/comments/1dzhe38/phi3_mini_june_with_function_calling/), competitive with Mistral-7b v3 and outperforming the base Phi-3 Mini.

**AI Research and Applications**

- **Synthetic data creation**: A [paper on scaling synthetic data creation](https://www.reddit.com/r/MachineLearning/comments/1dzergu/r_scaling_synthetic_data_creation_with_personas/) leverages diverse perspectives within a large language model to generate data from 1 billion personas curated from web data.

- **Anthropic's AI swarm intelligence**: Anthropic's CEO reports that [big models are now spawning smaller models](https://v.redd.it/oju63dvc0emd1) to complete tasks and report back, creating a swarm intelligence that decreases the need for human input.

**AI Industry and Community Discussions**

- **OpenAI subscription value**: OpenAI's Head of Applied Research [acknowledged disappointment](https://www.reddit.com/r/singularity/comments/1f7o9zg/head_of_applied_research_at_openai_im_sorry_we/) with their subscription offering, promising improvements to make it more valuable.

- **Stable Diffusion subreddit moderation**: The [Stable Diffusion subreddit is experiencing moderation issues](https://www.reddit.com/r/StableDiffusion/comments/1f7194f/we_need_to_talk_about_a_new_mod_who_has_a_history/), with concerns about a new moderator's behavior and changes to community rules.

**Memes and Humor**

- A post titled ["And then this happened"](https://i.redd.it/ob5wg9rxvemd1.jpeg) received significant attention in r/singularity.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Claude 3.5 Sonnet

**1\. LLM Advancements and Benchmarking**

- **Mistral-Nemo Pricing Shakeup**: The price of [Mistral-Nemo](https://openrouter.ai/models/mistralai/mistral-nemo) has dropped by **23%**, potentially signaling shifts in the competitive landscape for LLM providers.
  - This significant price change could indicate evolving market dynamics, with analysts keenly observing how competitors might respond to Mistral's aggressive pricing strategy.
- **GPT-4o Outperforms Turbo Variant**: **GPT-4o** is now **50% cheaper** than GPT-4 Turbo at **$5/M input and $15/M output tokens**, boasting **2x speed** and **5x higher rate limits** up to 10 million tokens per minute.
  - With a **128k context window** and enhanced **vision capabilities**, GPT-4o positions itself as a strong contender for users seeking efficiency and advanced features in language models.

**2\. Optimizing LLM Inference and Training**

- **Apple Silicon's Memory Bandwidth Conundrum**: While **Apple Silicon** boasts impressive memory bandwidth, its utility for CPU inference is limited compared to GPUs, with the **M1 Max's** advertised **400GB/s** raising questions about real-world effectiveness.
  - Discussions suggest that despite high theoretical bandwidth, practical performance for LLM inference on Apple Silicon may vary significantly, prompting further investigation into optimizing these architectures for AI workloads.
- **Triton Load Order Impacts Performance**: Users of **Triton** discovered that changing the order of loads can lead to significant speed differences, with one instance showing an improvement from **1.89506** to **2.440731**.
  - This observation raises questions about the compiler's handling of load stalls and instruction scheduling, suggesting potential optimizations for LLM training and inference pipelines.
- **Activation Checkpointing Triumph**: A member successfully implemented **activation checkpointing** with minimal code, demonstrating different memory requirements based on batch size using **124M BF16**.
  - The implementation showed memory usage of **1211 MiB** without reuse and **176 MiB** when recomputing 100% of layers, highlighting significant memory optimization potential for LLM training.

**3\. Open-Source AI Frameworks and Community Efforts**

- **Mini-Omni Voice Model Goes Open Source**: The [Mini-Omni](https://hf.co/gpt-omni/mini-omni) open-source model capable of generating text and audio simultaneously has been released for real-time audio conversations, with its [codebase](https://github.com/gpt-omni/mini-omni) and research paper detailing streaming audio output capabilities.
  - This release on Twitter sparked discussions about the model's potential applications and its impact on future AI interactions, showcasing the community's excitement for open-source advancements in multimodal AI.
- **Toolio 0.5.0 Enhances LLM Control**: **Toolio 0.5.0**, dubbed 'The triumph of text,' introduces improved documentation and better prompt construction for the Python toolkit designed for **Apple Silicon**, including structured LLM response generation conforming to [JSON schema](https://json-schema.org/).
  - This update aims to provide developers with fine-grained control over text generation, positioning Toolio as a critical tool for those requiring more than casual text generation, especially in tool-calling functionalities.
- **Mojo Standard Library Opens for Contributions**: The **Mojo Standard Library** is now partially open for contributions, although some sections remain closely tied to the compiler. A stable version is available, but robust stability guarantees are still being established.
  - Community members expressed excitement about the opportunity to contribute, while also noting the need for caution as the library's full potential and production-readiness are still being realized.

**4\. Hardware and Infrastructure for AI**

- **100k H100 Clusters Analysis Sparks Debate**: A comprehensive examination of **100,000 H100 clusters** discussed power efficiency, network topology, and trade-offs between Ethernet and InfiniBand options, highlighting how these clusters reflect a perceived slowdown in AI advancements post-**GPT-4**.
  - The analysis raised concerns about cluster reliability and fault recovery, indicating challenges in scaling current models effectively despite maintaining similar computational metrics to previous generations.
- **H200 and H100 Pricing Dynamics**: The **H200** GPU is currently priced at **180k** for the 8-unit variant, while a **huge increase** in **H100** prices was reported, potentially correlated with **Tesla's** activities in the market.
  - These pricing trends have sparked discussions about the impact of high demand from major tech companies on the AI hardware ecosystem, with the community closely watching how sustained demand might alter future pricing and availability strategies.

---

# PART 1: High level Discord summaries

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth Fine-tuning Sparks Debate**: Users reported obstacles while fine-tuning the **Gemma 2B** model, especially generating random outputs after adjustments to training parameters.
  - The discourse highlighted the need for consistent tuning templates to optimize token usage, cautioning against template changes.
- **Numpy vs. Cupy: Gemma 2 Implementation**: A member successfully implemented **Gemma 2** from scratch using [Numpy](https://github.com/githubpradeep/notebooks/blob/main/gemma2%20final.ipynb) and later transitioned to [Cupy](https://github.com/githubpradeep/notebooks/blob/main/gemma2%20final_cupy.ipynb).
  - The **Cupy** version requires a GPU with **24GB** of memory for effective computations, with an alternative [f16 version](https://github.com/githubpradeep/notebooks/blob/main/gemma2%20final_cupy_f16.ipynb) available for lower memory GPUs.
- **llama.cpp's RPC Memory Conundrum**: Members shared frustrations regarding **llama.cpp** integration with RPC servers, with one stating it failed to retain memory on server machines.
  - This frustration exemplifies the challenges associated with implementing complex AI models and infrastructure requirements.
- **Inquiry on Text-to-Speech Tuning**: A user sought assistance for tuning a Text-to-Speech model using Unsloth, but received clarification that it lacks this functionality.
  - The conversation led to mentions of a Whisper training guide that necessitates a larger dataset for effective training.
- **API Subscription Costs Under Scrutiny**: Concerns over costs prompted discussions on transitioning from subscription services to solely using the **API** due to underutilization of the full **$20** token allocation.
  - This trend reflects broader moves among users to better manage AI-related expenses and access.

 

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Phi-3.5-mini shines in-browser**: The **Phi-3.5-mini (3.8B)** model runs in-browser at ~90 tokens/second using WebGPU, ensuring fully local processing for enhanced **privacy**. Check out the demo and [source code here](https://x.com/xenovacom/status/1826992922509595068).
  - *Users reported significantly reduced latency while processing inputs locally compared to server-based models.*
- **Reinforcement Learning Repository Launches**: A member shared a GitHub repository for implementing **Reinforcement Learning Algorithms**, inspired by Sutton and Barto's book, aiming to cover various algorithms discussed. Visit the project [here](https://github.com/KhashayarRahimi/Reinforcement-Learning-Algorithms-From-Scratch).
  - *Community members showed interest in collaborative contributions to enhance algorithm implementations.*
- **Dynamic Game State Strategies for AOE2**: A member proposed a **CV project** for **Age of Empire II** to create AI assistants focusing on decision-making strategies by mapping game assets using computer vision tools like **SAM** and **YOLO**. Their approach involves detecting game elements efficiently.
  - *Discussion also sparked about the feasibility of local dynamic updates for meaningful insights during gameplay.*
- **Training Vision Language Models Needed**: Concerns were raised about the limitations of current LLMs, like ChatGPT-4, in effectively counting and localizing objects within images. Suggestion was made to consider training a **Vision Language Model (VLM)** to leverage advanced image processing techniques.
  - *The evolving intersection of vision and language models presents new challenges and opportunities for engineers in AI development.*
- **AI Tools for Health Insurance Appeals**: A new tool for appealing health insurance denials was introduced, leveraging **OCR** to scan letters and generate AI-driven appeals, accessible at [fighthealthinsurance.com](https://www.fighthealthinsurance.com/).
  - *Emphasis was placed on ensuring compliance with **HIPAA** laws in the tool’s operation and data management.*

 

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Tips for Loading Models in LM Studio**: **LM Studio** users learned that models saved in different folders cannot be loaded directly. To utilize models, they need to be organized in a specific directory structure within **LM Studio**.
  - Changing the model folder can be done from the 'My Models' view, streamlining the model management process.
- **GPU Troubleshooting in LM Studio**: A user reported issues with **LM Studio** not recognizing their GPU, leading to discussions on troubleshooting steps. Suggestions included checking the **Developer Tab** for LM Runtimes as a diagnostic measure.
  - This highlights the importance of compatible hardware in ensuring smooth operation within the software.
- **Temperature Settings for Quality Testing**: Users discussed the critical role of **temperature settings** in LM Studio to evaluate model outputs, particularly low settings for quality assessments. Beginners were urged to consult resources to understand temperature's effects in **LLMs**.
  - This emphasizes the need for careful parameter tuning to enhance model performance.
- **Apple Silicon's Memory Bandwidth Limitations**: While **Apple Silicon** offers exceptionally high memory bandwidth, its utility for CPU inference is limited compared to GPUs, raising performance concerns. The **M1 Max's** advertised **400GB/s** remains under scrutiny regarding effectiveness.
  - Discussions suggest that real-world performance varies significantly and merits further investigation.
- **RAM Caching Issues with OpenWebUI**: Report surfaced regarding **OpenWebUI** consuming excessive **RAM**, reportedly **150GB** out of **192GB**, due to preloading behaviors. Users speculated potential software bugs or misconfigurations in how the cache is managed.
  - This underlines the necessity for robust resource management strategies in web UI frameworks.

 

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Strategies to Combat Burnout in Tech**: Members discussed various methods to manage **burnout** in the demanding tech landscape, with expectations for further insights shared later.
  - *Maintaining motivation* was emphasized as a major hurdle for developers in the current environment.
- **CUDA Jobs Remain Elusive**: Concerns were raised regarding the **scarcity of CUDA jobs**, where companies often look for experience that many qualified candidates lack.
  - *This barrier to entry* has become a contentious point within the community, affecting newcomers.
- **Triton's Load Order Impacts Performance**: Changing the order of loads in **Triton** resulted in notable speed differences, with one user experiencing a speed-up from **1.89506** to **2.440731**.
  - This raises questions about the compiler's performance in handling load stalls and scheduling of instructions.
- **CUDA Kernel Needs for FP8**: For **FP8 support**, the kernel requires **SM_89** or higher, influencing compatibility with specific GPUs like the **A100**.
  - Testing on a **4090** showed a **1.3x performance improvement** over torch, indicating the benefits of newer architectures.
- **Efficient Use of Activation Checkpointing**: **Activation checkpointing** was successfully implemented using minimal code, affecting memory usage based on batch sizes processed.
  - Configurations displayed memory requirements of **1211 MiB** without reuse and **176 MiB** upon recomputing layers.

 

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Watch Out for Phishing!**: Participants raised concerns about a suspicious website, likely a phishing hub due to its **unsecured HTTP protocol** and *unencrypted data transmission*.
  - They urged users to *avoid sharing personal information* on such sites to mitigate security risks.
- **ComfyUI Faces Configuration Woes**: Users detailed issues with ComfyUI, particularly an error related to a missing configuration file and confusion over model installations.
  - It was suggested to utilize the *Save Text File* node for tracking prompts and workflows within ComfyUI.
- **Prompt Techniques for Better Results**: For Stable Diffusion, prompts structured as attributes separated by commas yield superior results, especially with older models like **SD 1.5**.
  - However, newer models benefit from *natural language prompts*, thanks to their enhanced text encoding capabilities.
- **Speculations on Stable Diffusion 3.1**: Participants speculated about the release of **Stable Diffusion 3.1**, noting limited information mostly from unofficial sources.
  - They called for *patience* as the community awaits official announcements from **Stable AI**.
- **Demand for Model Training Resources**: Users indicated a need for guidance on training **LoRA** models for specific characters and art styles, highlighting a gap in updated resources.
  - A [GitHub repository for Flux](https://github.com/black-forest-labs/flux) was shared, which may assist with insights on new model functionalities.

 

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Standard Library opens for contributions**: The **Mojo Standard Library** is partially open for contributions, although some sections remain closely tied to the compiler. Despite a stable version being available, concerns persist over its readiness for production, with **robust stability guarantees** still needing to be established.
  - Members indicated that updates and contributions are encouraged, yet the full potential of the library remains to be realized.
- **Modular CLI inches towards the final release**: Updates on the **Modular CLI** suggest it is nearing completion before the introduction of **Magic**, which will bring package management capabilities to the forefront. Current developments mainly focus on GPU support, signaling an end to further CPU-only releases.
  - Anticipation grows around a smoother package management experience similar to **Rust’s Cargo**, aimed at enhancing usability for developers.
- **MLIR points to language interoperability advancements**: **MLIR** integration discussions highlighted its potential to bridge communication across programming languages, though translation challenges remain. Notably, members commented on the simplicity MLIR may bring to some aspects, while also complicating others.
  - Concerns were raised relating to **backward compatibility** and adapting to existing C preprocessor dependencies.
- **OSDI '21 Keynote praises MAX**: The keynote from [OSDI '21](https://www.youtube.com/watch?v=36myc8wQhLo) emphasized that MAX can enhance computing capabilities beyond AI and HPC, citing its potential to optimize hardware interactions. The combination of **Mojo + MAX** could facilitate better utilization of diverse processors.
  - The expectation is that such integration would significantly boost computational power across various systems.
- **Memory Domains visualized as graph nodes**: Discussions proposed representing memory domains as graph nodes, enhancing the ability to understand relationships like latency and bandwidth between them. This method could allow hardware-aware compilers to make informed decisions about data movement.
  - Acknowledging existing channels as frictional, members expressed intent to develop a DPDK-based channel to ease these complexities while managing variable computation times.

 

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **AI's Content Quality Debate Escalates**: Participants believe that the rise of AI tools may lead to more **low-quality, clickbait content**, potentially degrading the overall quality of information online.
  - However, some assert that competition among AI-generated content will drive higher standards and **improve relevancy and accuracy**.
- **AI Assists Job Applications but Raises Concerns**: Discussion revealed that individuals are using AI to create tailored resumes for job applications, which AI tools then evaluate for efficiency.
  - This leads to worries about a potential **no human in the loop** scenario affecting hiring standards.
- **LAION Dataset Returns to Accessibility**: The LAION dataset is now accessible again after being previously removed over content concerns, with upcoming updates to integrate it with the **Clip retrieval API**.
  - Participants shared resources to access the dataset for enhanced AI training.
- **LLM-Based Agents Announce Insightful Paper**: The Manifold Research Group has released a position paper titled *Intelligent Digital Agents in the Era of Large Language Models*, highlighting advancements in **LLM-based AI agents**.
  - The paper addresses both breakthroughs and limitations, inviting further discussions on their [Discord](https://discord.gg/MfYZmYEGaa?ref=manifoldrg.com).
- **New MultiNet Evaluation Metrics Released**: Manifold defined new evaluation metrics for benchmarking several **Vision-Language Models (VLMs)** and applications, available in their [GitHub repository](https://github.com/ManifoldRG/MultiNet?ref=manifoldrg.com).
  - This initiative aims to provide detailed dataset coverage and improve quality assessments in AI metrics.

 

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Manifold Research Group releases position paper**: The Manifold Research Group shared their recent [position paper](https://www.manifoldrg.com/llm-agents/) on **LLM Based Autonomous Agents**, showcasing advancements in autonomous systems.
  - They invited interested individuals to join their [Discord community](https://discord.gg/MfYZmYEGaa?ref=manifoldrg.com) for more discussions.
- **Challenges with Compute Availability at Manifold**: Limited compute options from Manifold were confirmed, reliant on academic and industry partnerships, with specifics varying by project.
  - Inquiries for available compute resources were directed to **Harsh** or **Sidh** for tailored guidance.
- **ICLR conference holds prestige over NIPS workshops**: A discussion highlighted that publishing in the main **ICLR conference** is significantly more impactful for CVs than in a **NIPS workshop**, given the lower acceptance at workshops.
  - ICLR’s recognition as a **tier 1 conference** was underscored, lending weight to its papers.
- **Exploring LLMs and the Abstract-Crystallization Step**: A proposal surfaced suggesting LLMs could improve by incorporating an **abstraction-crystallization** step to evaluate multiple abstracted phrases, enhancing output creativity.
  - This could involve ranking phrases by vector similarity, steering outputs away from top-probability reliance.
- **Discussion on Diffusion Models learning Physics**: Concerns were raised about the efficacy of diffusion models in accurately learning physical laws versus simply overfitting on available datasets.
  - It was noted that enforcing physical structures might limit the expressivity of these models, warranting further investigation.

 

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Students Score Free Month of Perplexity Pro**: Students can grab a **free month of Perplexity Pro** by signing up with their .edu email before **September 15**. This service excels in delivering fast, precise answers for academic pursuits.
  - The features range from dissecting complex topics to crafting meal plans, making it a versatile tool for learners.
- **Whole School Wins Free Access at 500 Signups**: If a campus hits **500 signups**, the entire school will score **one year of Perplexity Pro** for free, promoting a competitive spirit.
  - The challenge runs until **September 15**, and users can monitor signups [here](https://www.perplexity.ai/backtoschool).
- **Perplexity API Usage Sparks Interest**: A member explored the potential of creating a Perplexity page using the API in combination with [Make.com](https://make.com), reflecting interest in integration.
  - Current documentation lacks clarity on this, prompting suggestions to consult the official [Perplexity documentation](https://docs.perplexity.ai) for further guidance.
- **File Upload Capabilities in Pro API**: Queries surfaced regarding the Pro API's ability to accept file uploads like .txt and .pdf during search queries through the CLI interface.
  - Users seek functionality similar to the web interface, indicating a desire for improved analytical capabilities.
- **Perplexity Xfinity Deal Creates Buzz**: A shared link regarding a [Perplexity Xfinity deal](https://www.perplexity.ai/search/perplexity-xfinity-deal-QCK.FX71SZCO6kSpE0YtYQ) suggests exciting offerings for users, potentially enhancing their experience.
  - Details remain vague, but anticipation builds around what this partnership may entail.

 

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Mistral-Nemo's Price Takes a Hit**: The price of [Mistral-Nemo](https://openrouter.ai/models/mistralai/mistral-nemo) has dropped by **23%**, reflecting changes in market dynamics.
  - This significant price change could indicate a shift in demand or supply for the **Mistral** models, prompting analysts to monitor competitor reactions.
- **Mume AI App Debuts with Excitement**: The **Mume AI** app, launched using OpenRouter as a provider, offers users access to over **100 models** for text and image generation.
  - The developer actively seeks community **feedback** to enhance the app as it enters its early stages, fostering user engagement.
- **Caching capabilities for Google and Claude models**: Discussions revealed that caching with **Google** and **Claude** models through OpenRouter might be close to being implemented.
  - Concerns about cache routing were expressed, particularly as the two endpoints do not share the same cache.
- **Clarification on Multi-Turn Conversations Support**: Inquiries about **multi-turn conversations** in OpenRouter clarified that users must resend the entire chat history to maintain continuity.
  - Responses noted that users need to manage this aspect since LLMs are inherently stateless.
- **Best Models for Character Consistency in AI**: A user sought recommendations for the best models to maintain character consistency, noting dissatisfaction with **Midjourney**.
  - Alternatives such as **Segmind** were suggested, as the conversation aimed at creating a reliable Instagram AI influencer.

 

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **NousCon Event Announced for September 18**: The **NousCon** event is set to take place in **San Francisco** on **September 18**, immediately after the **PyTorch Conference**.
  - Given the **limited space** available, eager participants are encouraged to check the [official announcement](https://x.com/NousResearch/status/1831032559477866754) and reserve their spot through the registration link [here](https://lu.ma/zlgp0ljd).
- **Hermes-3 trains at lightning speed**: The training process for **Hermes-3** can now be accomplished in just **4 minutes**, raising eyebrows about training techniques' efficiency.
  - This rapid training pace led to jokes about *speedrunning training* among the community members.
- **Questioning LLM Reasoning Frameworks**: Members noted a lack of notable **frameworks** addressing **LLM Reasoning and Planning**, highlighting a gap in effective solutions.
  - Discussions included skepticism towards the **LLM-Modulo concept**, with some members advocating for a focus on practical applications suggested by **Yann LeCun**.
- **Introducing Gemma 2: Numpy to CuPy Transition**: A member is working on implementing **Gemma 2** from scratch using **Numpy**, with plans to transfer it to **CuPy** for enhanced performance.
  - They shared links to the [Numpy Notebook](https://github.com/githubpradeep/notebooks/blob/main/gemma2%20final.ipynb) and [CuPy Notebook](https://github.com/githubpradeep/notebooks/blob/main/gemma2%20final_cupy.ipynb), along with GPU memory recommendations for effective execution.

 

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **SearchGPT release speculation heats up**: Users speculate about an imminent launch of **SearchGPT**, with some users briefly seeing a pop-up that read 'You're in' after joining the waitlist, though access was quickly lost.
  - Another user pointed out that **Perplexity** outperforms SearchGPT, especially since **Arc** integrates **Perplexity**, making it a more favorable option for now.
- **AI explores fun with gaming content**: A member initiated the idea of creating a video featuring **AI playing UNO**, sparking discussions about the potential for AI in engaging content creation.
  - This concept reflects a growing interest in leveraging AI for interactive experiences in gaming.
- **GPT-4o offers promising features over Turbo**: **GPT-4o** is touted as **50% cheaper** than **GPT-4 Turbo**, costing **$5/M input and $15/M output tokens**, while boasting **2x speed** and **5x higher rate limits** up to **10 million tokens per minute**.
  - With a **128k context window** and enhanced **vision capabilities**, GPT-4o positions itself as a strong contender for users seeking efficiency.
- **Community frustration with ChatGPT policies**: Concerns emerged over **ChatGPT**'s handling of sensitive topics, with users noting a shift in response patterns and increasing message deletions, potentially deterring users.
  - Users called for improved transparency and responsiveness from AI developers to address these ongoing issues.
- **Improving AI writing through clarity**: Members highlighted the need for clearer instructions to mitigate unwanted phrases in AI responses, advocating a shift towards providing positive examples of desired language.
  - By emphasizing what the model should do, rather than what to avoid, participants noted that this could lead to more effective outcomes consistent with behavioral techniques.

 

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Auto-Document Retrieval Boosts Efficiency**: A recent notebook illustrates combining **RAG (Retrieval-Augmented Generation)** with structured querying, enhancing document retrieval for large datasets, detailed in a [related post](https://t.co/nfXnpvM93D).
  - *How do you retrieve the right documents?* This method effectively targets that challenge.
- **LLMs Craft PowerPoint Decks Effortlessly**: An innovative TypeScript app transforms notes into **PowerPoint slides**, allowing users to ditch tedious tasks and focus on their creativity, demonstrated in this [demo link](https://t.co/ItJ3edWmXF).
  - The app not only summarizes notes but also generates extra content, showcasing the capabilities of **LLMs**.
- **Proposal for Jina AI's Late Embeddings Class**: A member proposed developing an embeddings class for **Jina** utilizing the new 'late embeddings' method, as found in the [HF code](https://github.com/jina-ai/late-chunking/tree/main/chunked_pooling).
  - Another member suggested most code might fit into a node parser package by using the BaseNodeParser class.
- **Gemini LLM Struggles with Initialization**: A user encountered an **AttributeError** with the **Gemini** LLM upon restarting their kernel, noting it worked before this change.
  - Updating dependencies was suggested to address issues stemming from a recent **pydantic** upgrade.
- **Chat Engine Message Filtering Inquires**: A member sought a way to filter answers from message history for LLM queries, aiming to send only questions to the chat engine.
  - Another proposed subclassing memory and overriding the `get()` method as a potential solution.

 

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **H200 Price Stays High at 180k**: Currently, the **H200** is priced at **180k** for the **8** variant, raising questions about high demand influencing market pricing.
  - Members are keeping an eye on how this price affects accessibility in the AI hardware ecosystem.
- **Surge in H100 Prices Linked to Tesla**: A recent **huge increase** in **H100** prices is suggested to be correlated with **Tesla's** activities.
  - The community is curious to see how sustained demand from such industries would alter future pricing strategies.
- **Chat Template PR Aids Setup**: The **chat template PR** has been highlighted as crucial for loading the tokenizer's template automatically, simplifying setup significantly.
  - This advancement is expected to streamline onboarding processes for new users working with AI chat interfaces.
- **Cross Entropy Loss in SFTT Explained**: A user questioned if **SFTT** computes **cross entropy loss**, with another pointing them to the modeling code for **LLaMA** on GitHub for checks.
  - This highlights the importance of clearly laying out the codebase reference to understand loss calculations.
- **Exploring Multi-User Dialogue for Fine-Tuning**: One member discussed fine-tuning a model on **dialogues from multiple people** without an agent, focusing on how to format such data.
  - Considerations were made on training models to better grasp conversation flow through chat history prompts.

 

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **New tools in Playground spark excitement**: Members confirmed that **tools are now enabled** for the new model in the playground, fostering exploration and creativity.
  - *Happy building!* was the enthusiastic encouragement from a team member following this announcement.
- **LLMs facilitating report generation?**: A query arose regarding the use of **LLMs** to generate reports based on previous writing styles and meeting notes for the Internal Audit team.
  - Members were invited to share their experiences on leveraging these models for effective report generation.
- **Model card discrepancy highlighted**: A member pointed out that the [model card](https://huggingface.co/CohereForAI/c4ai-command-r-08-2024) inaccurately states a model size of **35B**, instead of **32B**.
  - The team recognized the oversight and promised to correct it soon.
- **Cohere supports Server Side Events!**: Confirmation came that sending an `Accept: text/event-stream` header to the chat API will enable users to receive **SSE events**.
  - Documentation updates are underway to include this previously undocumented feature.
- **Feature request process clarified**: A member inquired about submitting a feature request for server side events, prompting conversation among team members.
  - Feedback was acknowledged, with plans for further discussion with the product team.

 

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Orchestrate your Multi-Agent Conversational Assistant**: A member sought help for setting up a **Multi-Agent Conversational Assistant**, particularly interested in the **Supervisor architecture** and its inherent complexities.
  - The discussion highlighted different architectural approaches with a call for shared experiences and insights.
- **Hybrid Retriever is the Future**: A user proposed the concept of a **hybrid retriever** that combines **two or more retrievers** to enhance search performance.
  - The idea sparked enthusiasm, with members expressing excitement about its potential applications.
- **Demystifying Hugging Face Embeddings**: A member discussed passing **encode_kwargs** to a **Hugging Face embedding endpoint**, sharing a code snippet for clarity.
  - They confirmed that the **TEI** handles embedding normalization automatically, simplifying their implementation.
- **Toolio 0.5.0 Brings Exciting Features**: The launch of **Toolio 0.5.0** introduces improved documentation and LLM response generation conforming to a [JSON schema](https://json-schema.org/).
  - Developers can expect more control over text generation through structured outputs tailored to their needs.
- **Generative AI Projects Demand Your Stars**: A member shared their **Generative AI projects** from this year on GitHub, encouraging others to [check out their work](https://www.linkedin.com/posts/isham-rashik-5a547711b_github-di37dubai-tech-talk-intelligent-chatbot-activity-7236606074173222912-Fp2U) and star the repositories.
  - The drive for project engagement emphasizes community feedback as pivotal for project visibility and collaboration.

 

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Python PATH Causes Confusion**: A member faced challenges getting their Python script for **Open Interpreter** to recognize the module after multiple installations using `pip install open-interpreter` in their virtual environment.
  - This has sparked an ongoing discussion in the community regarding *best practices for environment setup*.
- **House Party Event Announcement**: An exciting **House Party** event was announced, promising big news and demos that could be the most impactful yet.
  - The event will be **livestreamed** and recorded, but attendees are encouraged to come to avoid missing out on the experience.
- **Weekly Shill for Tool Use**: This week's episode of **Tool Use** features a guest, highlighting their insights and discussions. You can check out the episode [here](https://www.youtube.com/watch?v=UUUWt8GaS64).
  - *Thanks to the community for support*—the share of experiences continues to invigorate discussions around tool usage.
- **Excited Chat with Guest**: Members expressed happiness about chatting with a new guest during the Tool Use session.
  - *A member shared their joy* in the conversation, creating an inclusive environment for shared learning.

 

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Same Row Data Severs Outcomes**: A member confirmed that all data points from the same row affect the **final outcome** when sourced from the same **sample**.
  - They further inquired about a **specific dataset** being analyzed, emphasizing the need for clarity on data interactions.
- **LoRA Checkpoints Raise Questions**: Concerns emerged over using the full merged adapter weights in the checkpoint dictionary despite `adapter_weights_only` settings.
  - Clarification came that this process was *removed entirely* in the [Llama 405B PR](https://github.com/pytorch/torchtune/pull/1449), though updates are still pending in all recipes.
- **Room for More Adapter Weight Support**: A suggestion was put forward to enhance flexibility for supporting `adapter_weights_only` in fine-tuning configurations.
  - This aligns with the general consensus aiming to improve usability for current users in AI model training.
- **Max Sequence Length Solutions on the Horizon**: Excitement grew around new generation updates with potential fixes for **max_seq_len** issues being discussed.
  - Confidence in collaborative efforts to tackle these challenges suggests a proactive community approach moving forward.
- **Draft Max Sequence Length Refactor Under Review**: A draft for the **max_seq_len** implementation refactor was shared, indicating ongoing development on GitHub.
  - The member committed to updating documentation post-discussion set for tomorrow, showcasing a dedicated effort toward improvement.

 

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Missing Model Apology in Leaderboard**: The team acknowledged an oversight in missing a **model** during leaderboard results regeneration and vowed to correct this in the next update.
  - This commitment aims to enhance the accuracy of model representation on the leaderboard.
- **New Dataset Takes Priority for Hermes Model**: Focus has shifted to a new **dataset release**, causing delays in processing new model requests until later this week or next week.
  - Members are encouraged to submit PRs for their desired models while waiting for updates.
- **Chat Mode Adds Complexity to Decoding**: Models now operate in both **chat mode** and **FC mode**; the latter facilitates structured output, improving decoding efficiency.
  - The **DEFAULT_SYSTEM_PROMPT** in chat mode aims to guide responses more systematically.
- **Clarifying Leaderboard Data Sources**: The `leaderboard_live.html` uses the **BFCL V2-Live dataset**, while the main `leaderboard.html` aggregates all **BFCL V2 datasets**, both Live and non-Live.
  - Understanding this distinction is essential for accurate interpretation of leaderboard results.
- **Issue Raised on GitHub About Leaderboard Discrepancy**: A member reported opening an issue about the leaderboard discrepancy on GitHub, providing a [link to the issue](https://github.com/ShishirPatil/gorilla/issues/620).
  - They also offered to submit a PR if their solutions matched the outlined problems.

 

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Mini-Omni voice model goes open source**: The [Mini-Omni](https://hf.co/gpt-omni/mini-omni), an open-source model capable of generating text and audio simultaneously, has been released for real-time audio conversations. Its [codebase](https://github.com/gpt-omni/mini-omni) and accompanying research paper detail the model's impressive streaming audio output capabilities.
  - Discussion on Twitter highlighted the potential applications and excitement around this conversational model and its impact on future AI interactions.
- **Insightful analysis on 100k H100 clusters**: A comprehensive examination on the **100,000 H100 clusters** touched on power efficiency, network topology, and the trade-offs between Ethernet and InfiniBand options. It pointed out how these clusters reflect a slowdown in AI advancements post-**GPT-4**, despite maintaining similar computational metrics.
  - This detailed analysis raised concerns about cluster reliability and fault recovery, indicating challenges in scaling current models effectively, as illustrated in [this report](https://www.semianalysis.com/p/100000-h100-clusters-power-network?triedRedirect=true).
- **New Latent Space Podcast Launched**: A new podcast episode from [Latent Space](https://x.com/latentspacepod/status/1831020483967701260) was announced, focusing on the latest trends in AI engineering. This aims to address the evolving landscape and share insights from leading experts in the field.
  - Listeners can expect thought-provoking discussions that delve into essential AI topics and community-driven knowledge sharing.

 

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Exploration of WeaviateRM Integration**: A member showed interest in **WeaviateRM integration** and requested a forum issue about **text2vec-ollama**. They shared a link to the [Weaviate forum](https://forum.weaviate.io/latest) for further discussion.
  - Another member confirmed their willingness to assist by agreeing to open the forum issue, wrapping up the conversation with gratitude.
- **Exploring COPRO for Length Management**: A member inquired about using **COPRO** or similar models to optimize instruction length effectively, suggesting adjustments to **max_tokens**.
  - They proposed implementing a metric return system as a way to manage instruction lengths.
- **Zero-shot Instruction Optimizer Techniques**: Discussion revolved around employing a **zero-shot instruction optimizer** to control instruction lengths within models.
  - Members debated whether to set length constraints simply by limiting **max_tokens** or creating complex metrics for instructions and input length.

 

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **LLM Enhances Report Generation**: A member inquired about using **LLMs** to generate reports from previous writing styles and meeting notes, aimed at aiding the Internal Audit team with report creation.
  - This discussion emphasized the potential of automating report generation to improve efficiency.
- **Diverse Definitions of Meeting Notes**: Clarifications emerged around the term meeting notes, with suggestions they might include full transcripts with attendee names.
  - This led to a deeper conversation about varying interpretations of what constitutes comprehensive meeting documentation.
- **Synthetic Meetings Take Shape**: One user shared their work with the [persona-hub](https://github.com/tencent-ailab/persona-hub) to create synthetic meeting formats and facilitate simulated dialogues.
  - They noted the high token usage in these simulations but praised the rich variety it brings for training LLMs.
- **Text-to-Speech for Meeting Summaries Planning**: Plans unfolded to implement Text-to-Speech for generating audio from meeting summaries, utilizing LLMs for summarization.
  - Additionally, there was a focus on training a **whisper model** for speaker-diagram identification to enhance source attribution during meetings.

 

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **tinygrad Highlights**: George Hotz's project, **tinygrad**, showcases a minimalist approach to deep learning, providing an intriguing alternative to larger frameworks.
  - Although details were sparse in the chat, the excitement around **tinygrad** indicates a rising interest in lightweight solutions among AI engineers.
- **Community Engagement**: The channel had a brief interaction, with th.blitz greeting members enthusiastically, which highlights the community's active involvement.
  - This simple greeting shows that even small interactions can foster a sense of belonging in technical discussions.

 

---

The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **Interconnects (Nathan Lambert) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

# PART 2: Detailed by-Channel summaries and links

{% if medium == 'web' %}

 

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1280248522909683823)** (592 messages🔥🔥🔥):

> - `Unsloth fine-tuning`
> - `Gemma 2B model`
> - `Chat templates`
> - `Dataset quality`
> - `LLM training parameters`

- **Challenges with Unsloth Fine-Tuning**: Users discussed issues with fine-tuning the Gemma 2B model, particularly the challenge of it generating random content after training.
  - It was observed that changing training parameters or datasets may lead to unexpected results in the model's output.
- **Importance of Template Consistency**: The conversation emphasized that when tuning an instruct-tuned model, it's crucial to use the same template it was originally tuned with for the best outcome.
  - Users considered that altering the template could lead to less efficient token usage and inference challenges.
- **Quality Over Quantity in Datasets**: Participants concurred that it’s the quality of the dataset rather than the quantity that truly matters for effective fine-tuning.
  - To achieve optimal results, it was recommended to use high-quality datasets for tuning.
- **Experimentation in Fine-Tuning**: While maintaining traditional methods, participants expressed willingness to experiment with various tuning parameters such as rank and alpha.
  - There was a recognition that experimentation could yield valuable insights, even when breaking from conventions.
- **Collaboration and Learning**: Throughout the discussion, users shared insights and experiences, fostering a collaborative atmosphere for learning about LLM fine-tuning.
  - Members expressed appreciation for the community's help and the wealth of knowledge being exchanged.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct">unsloth/Meta-Llama-3.1-8B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1vIrqH5uYDQwsJ4-OO3DErvuv4pBgVwk4?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://tenor.com/view/jambajew-steve-brule-stare-gif-18457155">Jambajew Steve Brule GIF - Jambajew Steve Brule Stare - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://lu.ma/xd0zzk0h">Continued Pretraining and Fine-Tuning with Unsloth · Luma</a>: Continued pretraining, alongside Supervised Fine Tuning (SFT), is gaining in popularity alongside Small Language Models (SLMs) in the industry. Finding faster…</li><li><a href="https://ollama.com/library/llama3.1:8b-instruct-fp16">llama3.1:8b-instruct-fp16</a>: Llama 3.1 is a new state-of-the-art model from Meta available in 8B, 70B and 405B parameter sizes.</li><li><a href="https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966">🪐 SmolLM - a HuggingFaceTB Collection</a>: no description found</li><li><a href="https://tenor.com/view/jizz-adult-swim-john-reilly-blink-gif-14841420">Jizz Adult Swim GIF - Jizz Adult Swim John Reilly - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/unsloth/Meta-Llama-3.1-8B">unsloth/Meta-Llama-3.1-8B · Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1d05x6v/llamacpp_runs_18_times_faster_than_o">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1d05x6v/llamacpp_runs_18_times_faster_than_ollama/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/abetlen/llama-cpp-python">GitHub - abetlen/llama-cpp-python: Python bindings for llama.cpp</a>: Python bindings for llama.cpp. Contribute to abetlen/llama-cpp-python development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth/issues/975#issuecomment-2323009118">Single GPU training in Multi-GPU system doesn't work. · Issue #975 · unslothai/unsloth</a>: Single GPU training in Multi-GPU system doesn't work even if limited to 1 GPU with os.environ CUDA_VISIBLE_DEVICES before importing unsloth. Reason: check_nvidia function spawns new process to che...</li><li><a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/ggerganov/llama.cpp/blob/master/include/llama.h">llama.cpp/include/llama.h at master · ggerganov/llama.cpp</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.</li><li><a href="https://tenor.com/view/o-hearn-gif-3900414469346077199">O Hearn GIF - O hearn - Discover &amp; Share GIFs</a>: Click to view the GIF</li></ul></div>

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1280640529142382602)** (3 messages):

> - `llama.cpp integration with RPC`
> - `API subscription considerations`

- **Challenges with llama.cpp and RPC servers**: A member expressed difficulty using **llama.cpp** with RPC servers, stating that it wouldn't retain any memory on the server machines.
  - *I don't know why it wouldn't keep any memory* indicates frustration over the integration process.
- **Switching API usage due to cost**: Another member mentioned considering switching from subscription services to using only the **API** as they don't utilize the full **$20** worth of tokens each month.
  - This reflects a potential trend among users to optimize costs associated with AI access.

 

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1280281516232015944)** (19 messages🔥):

> - `DPO Notebook Inference`
> - `Unsloth Installation Issues`
> - `TypeError with Xformers`
> - `Text-to-Speech Model Tuning`
> - `Contact for Unsloth Purchase`

- **DPO Notebook lacks Inference Code**: A user referenced a DPO notebook for tuning a Llama model but noted the absence of inference code provided in it.
  - Another member suggested copying the inference code from an existing inference notebook as a solution.
- **Installation Problems with Unsloth**: A user faced issues while installing Unsloth in a Docker container and reported a strange error during the process.
  - Another member recommended creating a new Python environment with versions 3.9 or 3.10 as a potential fix.
- **TypeError Related to Xformers**: Members discussed encountering a TypeError when running a model generate command, specifically highlighting a 'Multiple dispatch failed' error.
  - One user found a solution though they were unsure about the steps they took to resolve it.
- **Resources for Text-to-Speech Model Tuning**: A beginner in AI tuning inquired if Unsloth could assist with tuning a Text-to-Speech model but was informed that it does not support that functionality.
  - They sought recommendations for resources, mentioning a Whisper training guide that might require a larger dataset for effective training.
- **Inquiry on Purchasing Unsloth**: A user expressed interest in buying Unsloth and asked for the appropriate contact person for this transaction.
  - Another member suggested reaching out to the project team or Unsloth Pro for assistance.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1aqlNQi7MMJbynFDyOQteD2t0yVfjb9Zh?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: See the list below for all our notebooks:</li><li><a href="https://download.pytorch.org/whl/cu121">no title found</a>: no description found</li><li><a href="https://huggingface.co/blog/fine-tune-whisper">Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers</a>: no description found</li></ul></div>

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1280490048659390617)** (1 messages):

> - `Gemma 2 implementation`
> - `Numpy vs Cupy`
> - `GPU requirements`

- **Implementing Gemma 2 from Scratch**: Over the last 3 days, a member successfully implemented **Gemma 2** from scratch using [Numpy](https://github.com/githubpradeep/notebooks/blob/main/gemma2%20final.ipynb) and later ported it to [Cupy](https://github.com/githubpradeep/notebooks/blob/main/gemma2%20final_cupy.ipynb).
  - The implementation showcases the ability to run Gemma 2 on both GPU and CPU, making it accessible for different hardware setups.
- **Cupy GPU Requirements**: For optimal performance, the **Cupy** version requires a GPU with **24GB** of memory, which is critical for handling the computations efficiently.
  - Alternatively, for GPUs with less than **16GB**, users can run the [Cupy f16 version](https://github.com/githubpradeep/notebooks/blob/main/gemma2%20final_cupy_f16.ipynb) to save memory while executing computations.
- **Running on CPU with Numpy**: Users can still run the implementation on CPU using the [Numpy notebook](https://github.com/githubpradeep/notebooks/blob/main/gemma2%20final.ipynb), providing a broader reach for those without access to powerful GPUs.
  - This option proves useful for testing and smaller scale computations that don't require extensive hardware resources.

 

---

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1280623804481142857)** (1 messages):

> - `Phi-3.5-mini`
> - `New Paper on Vision-Language Models`
> - `Building Your Own Robot`
> - `TRL v0.10.1 Release`
> - `Carbon Emissions Tracking`

- **Phi-3.5-mini operates in-browser**: The **Phi-3.5-mini (3.8B)** model is now running in-browser at ~90 tokens/second using WebGPU, Transformers.js, and ONNX Runtime Web, achieving fully local processing for enhanced **privacy**.
  - A demo and source code are available at [this link](https://x.com/xenovacom/status/1826992922509595068).
- **Insightful new paper released**: A new paper from Hugging Face provides insights into **state-of-the-art** vision-language models and their current limitations, catering to both beginners and experts.
  - It might be worth a read if you're looking for fresh perspectives in the field; check it out [here](https://x.com/HugoLaurencon/status/1827986085097402553).
- **Create your autonomous robot**: An in-depth tutorial was released on how to **build your own robot**, allowing users to teach it new skills with just a laptop.
  - This interactive approach lets your homemade robot act autonomously; the tutorial can be found [here](https://x.com/RemiCadene/status/1825455895561859185).
- **TRL v0.10.1 packed with new features**: The release of **TRL v0.10.1** includes enhancements like Online DPO by DeepMind to improve LLM alignment and integration with the Liger kernel for supercharged SFT.
  - Explore the various new capabilities, including DPO for vision-language models, on [GitHub](https://github.com/huggingface/trl/releases/tag/v0.10.1).
- **New carbon tracking feature on model cards**: A new feature has been introduced on the Hub that displays carbon emissions during model training directly on the model card.
  - This initiative aims to encourage model authors to share their **carbon emissions** data; more details are available [here](https://x.com/AymericRoucher/status/1830621688163127417).

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://x.com/xenovacom/status/1826992922509595068)">Tweet from Xenova (@xenovacom)</a>: I can't believe this... Phi-3.5-mini (3.8B) running in-browser at ~90 tokens/second on WebGPU w/ Transformers.js and ONNX Runtime Web! 🤯 Since everything runs 100% locally, no messages are sent ...</li><li><a href="https://x.com/HugoLaurencon/status/1827986085097402553)">Tweet from Hugo Laurençon (@HugoLaurencon)</a>: Whether you are: •A complete beginner looking to get a high-level overview of the SOTA VLM approaches and their limitations •An expert searching for new directions in the field Our new paper might b...</li><li><a href="https://x.com/RemiCadene/status/1825455895561859185)">Tweet from Remi Cadene (@RemiCadene)</a>: The wait is finally over!!! 😁 We just dropped an in-depth tutorial on how to build your own robot! Teach it new skills by showing it a few moves with just a laptop. Then watch your homemade robot ...</li><li><a href="https://x.com/NielsRogge/status/1828010283530424684)">Tweet from Niels Rogge (@NielsRogge)</a>: Alright finally able to dreambooth myself with Flux for free! Note that this is actually what @levelsio or services like @FAL or @replicate are monetizing. Here's how (small 🧵):</li><li><a href="https://x.com/_marcsun/status/1828824017593385276)">Tweet from Marc Sun (@_marcsun)</a>: `transformers` + `torchao` quantization + `torch.compile` for faster inference speed and less memory usage 🔥 Demo of "meta-llama/Meta-Llama-3.1-8B-Instruct" quantized in 4-bit weight-only :</li><li><a href="https://x.com/_lewtun/status/1829184370390777980)">Tweet from Lewis Tunstall (@_lewtun)</a>: TRL v0.10.1 is here and it's beefy 💪 🔁 Online DPO by @GoogleDeepMind for aligning better LLMs 🐯 Liger kernel integration from @LinkedIn to supercharge SFT 🖼️ DPO for VLMs: 🌋 LLaVa, ✨ PaliGem...</li><li><a href="https://x.com/abhi1thakur/status/1828049871967846897)">Tweet from abhishek (@abhi1thakur)</a>: 🚨 NEW COMPETITION ALERT 🚨 The Real-world Adversarial Attack from ROAM challenge addresses the critical issue of deploying deep learning systems in environments where images may be intentionally adve...</li><li><a href="https://x.com/AymericRoucher/status/1830621688163127417)!">Tweet from Aymeric (@AymericRoucher)</a>: New feature on the Hub! ☁️ Carbon emissions emitted during training now show up on the model card! (requires model authors to fill that info first) Hopes it will prompt more people to show the carbo...</li><li><a href="https://x.com/abhi1thakur/status/1830963506252067277)">Tweet from abhishek (@abhi1thakur)</a>: How to train your own Flux LoRA on Hugging Face: the easiest guide on lora training guide on twitter. In this thread, I'll show you how you can train your own flux lora on Hugging Face for all kin...</li><li><a href="https://x.com/_philschmid/status/1828441244558618944)">Tweet from Philipp Schmid (@_philschmid)</a>: Announcing “Cloud AI Tuesdays”. 🚀 Every Tuesday, we will share detailed examples of how to build AI with open models in the Cloud (@googlecloud, @awscloud, @microsoft Azure…) ☁️ Today, we are kicki...</li><li><a href="https://x.com/mervenoyann/status/1826005697924050966)">Tweet from merve (@mervenoyann)</a>: Microsoft dropped a series of Phi-3 models including a vision one! 🤏🏻 4.2B model, 128k token context length 🥹 43.0 on MMMU (very good for it's size) 🎥 accepts single/multiple image and video ...</li><li><a href="https://x.com/mervenoyann/status/1829144958101561681)">Tweet from merve (@mervenoyann)</a>: NVIDIA just dropped NVEagle 🦅 Super impressive vision language model that comes in 7B, 13B and 13B fine-tuned on chat, improved visual perception with MoE vision encoders 💬 Keep reading for detail...</li><li><a href="https://x.com/huggingface/status/1829549834652483983)">Tweet from Hugging Face (@huggingface)</a>: Several Hugging Face team members are coming to SF for the PyTorch Conference, and we'll celebrate in style Come join the🌟Hugging Face Party🌟at the @PyTorch Conference on the 19th of Sept! M...</li><li><a href="https://huggingface.co/organizations/HF-Party/share/LmPGIYKDiiYvPOUoAPXUjdIAXskWeRSKMk)">Hugging Face – The AI community building the future.</a>: no description found</li></ul></div>

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1280241580917264464)** (243 messages🔥🔥):

> - `Hugging Face API and Model Use`
> - `Model Performance and Training`
> - `Community Questions and Debugging`
> - `ChatGPT Developments and Updates`
> - `Content Creation and AI Tools`

- **Hugging Face API for Commercial Use**: A user inquired whether switching to a pro plan for Hugging Face would allow using inference APIs for a commercial app, provided the model permits it.
  - It was clarified that inference endpoints might be more efficient and cost-effective for their needs, with concerns about free usage and rate limits mentioned.
- **Model Performance on T4 vs L4 GPUs**: Discussion arose about running the FLUX model on a T4 GPU, with a user expressing concern about whether they might need to switch to an L4 for optimal performance.
  - Insights from community members indicated that the model's size (12B) could lead to high resource demands, hinting at potential cost implications.
- **Debugging Token Embeddings Issues**: A member reported fixing an issue with improperly sized token embeddings, which initially contributed to performance problems.
  - This reflects ongoing community engagement with debugging and improving model configurations.
- **User Interaction with AI Systems**: A light-hearted discussion occurred about creating an AI-supported video site that would filter based on content 'luridness', indicating a desire for better content management.
  - The idea sparked thoughts about leveraging AI technology for social media and content moderation improvements.
- **Updates and Observations on Hugging Face**: Users expressed curiosity about recent updates or changes in the Hugging Face ecosystem, highlighting a gap in news and development announcements.
  - Speculation included potential shifts in user engagement and community dynamics given the recent lack of communications.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="http://localhost:8080" )```"="">no title found</a>: no description found</li><li><a href="https://blog.adnansiddiqi.me/">Adnan's Random bytes</a>: Programming, Productivity, Entrepreneurship and Life Hacks</li><li><a href="https://distill.pub/2018/building-blocks/">The Building Blocks of Interpretability</a>: Interpretability techniques are normally studied in isolation. We explore the powerful interfaces that arise when you combine them -- and the rich structure of this combinatorial space.</li><li><a href="https://huggingface.co/docs/transformers/en/perf_infer_gpu_one">GPU inference</a>: no description found</li><li><a href="https://tenor.com/view/vine-so-no-head-angry-mad-throw-phone-gif-16162067">Vine So No Head GIF - Vine So No Head Angry - Discover &amp; Share GIFs</a>: Click to view the GIF</li></ul></div>

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1280263056957640755)** (5 messages):

> - `FP8 with Mixed Precision`
> - `AI Avatars using Meta Humans`
> - `Perplexity AI Pro for Students`
> - `Shipping RAG Chatbots`
> - `FST-NLP`

- **FP8 Baseline Achievements**: Successfully trained an improved baseline for **FP8 - Bfloat16** with **mixed precision**; addressing **losses** that still match after identifying an issue causing FP8 to go **NaN** during gradient accumulation.
  - This improvement enhances the **efficiency** of training models while managing complexity in computations.
- **Creating AI Avatars with Meta Humans**: Learned to build **AI Avatars** via **Meta Humans** from Epic Games by setting up **Unreal Engine 5.4** on x86_64, which is free with a linked GitHub account.
  - This resource opens avenues for further creative development in digital characters and immersive experiences.
- **Free Perplexity AI Pro for Students**: Discovered that students with a **.edu email** can sign up for a free month of **Perplexity AI Pro** by visiting [this link](https://www.perplexity.ai/backtoschool).
  - This offer is available for **two weeks only**, making it an excellent opportunity for students to explore advanced AI tools.
- **Preparing to Deploy a RAG Chatbot**: Discussed preparations to confidently say **'ship it!'** for deploying a RAG chatbot, considering **Docker/containerization** and possibly **Google Cloud Run**.
  - The focus is on balancing cost and innovative architecture in deployment strategies.
- **Exploration of FST-NLP**: Mentioned **FST-NLP**, signaling an interest in natural language processing advancements.
  - This reflects ongoing engagement with NLP technologies and their implications.

 

**Link mentioned**: [Perplexity - Race to Infinity](https://www.perplexity.ai/backtoschool): Welcome back to school! For just two weeks, redeem one free month of Perplexity Pro on us. Refer your friends, because if your school hits 500 signups we'll upgrade that free month to an entire free y...

 

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1280288160927977596)** (7 messages):

> - `Negative Probabilities`
> - `Hugging Face Blog Explorers`
> - `Firefox Tab Manager`
> - `GitHub Contributions`

- **Exploring Negative Probabilities**: A member shared a paper titled [Negative Probability](https://arxiv.org/abs/2405.03043) which discusses the use of negative probabilities in quantum theory and Bayesian modeling.
  - It was noted that *when the interest rate is negative*, there are correlations with negative values in certain distributions.
- **Joining the Hugging Face Blog Explorers**: A member requested help joining the [Hugging Face Blog Explorers](https://huggingface.co/blog-explorers) and shared their recent [GitHub PR](https://github.com/argilla-io/argilla/pull/5375) on a tutorial about #autotrain.
  - Another member, after reviewing requests, encouraged them to *feel free to request* again for acceptance.
- **Firefox Tab Manager Enhancements**: A member introduced a [Firefox add-on for tab management](https://addons.mozilla.org/en-US/firefox/addon/grasshopper-urls/) that supports vertical tabs and integrates History and Bookmarks.
  - The add-on requires permissions for Tabs, History, and Bookmarks, emphasizing that bookmarks won't be deleted within the extension itself.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://arxiv.org/abs/2405.03043">Negative Probability</a>: Negative probabilities arise primarily in quantum theory and computing. Bartlett provides a definition based on characteristic functions and extraordinary random variables. As Bartlett observes, negat...</li><li><a href="https://addons.mozilla.org/en-US/firefox/addon/grasshopper-urls/">Grasshopper – Get this Extension for 🦊 Firefox (en-US)</a>: Download Grasshopper for Firefox. Powerful Tab Manager</li><li><a href="https://github.com/argilla-io/argilla/pull/5375.">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li></ul></div>

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1280289020189999206)** (14 messages🔥):

> - `Reinforcement Learning Algorithms Repository`
> - `Health Insurance Appeal Bot`
> - `Basalt Project Launch`
> - `Data Transformation Tool`
> - `RAG System on Macbook`

- **Khashayar’s Reinforcement Learning Repository Shines**: A member shared their GitHub repository for implementing **Reinforcement Learning Algorithms** based on Sutton and Barto's classic book, hoping others find it useful. The repository is available [here](https://github.com/KhashayarRahimi/Reinforcement-Learning-Algorithms-From-Scratch).
  - The member expressed their enthusiasm about the project, which strives to cover various algorithms discussed in the book.
- **Health Insurance Appeal Bot Now Live!**: A member introduced a new tool that assists users in appealing health insurance denials, available at [fighthealthinsurance.com](https://www.fighthealthinsurance.com/). The tool uses **OCR** to scan denial letters and generates potential appeals through generative AI.
  - Feedback emphasized the importance of adhering to **HIPAA** laws and ensuring transparency regarding data usage.
- **Introducing Basalt: Next-Gen Feature Creation**: The **Basalt** project was launched, aimed at simplifying the creation and deployment of AI features for product managers. Interested users can access and try out the project through a linked [Typeform](https://www.linkedin.com/posts/marquis-guillaume_im-pleased-to-announce-the-launch-today-activity-7236642886371405825-Fm8x?utm_source=share&utm_medium=member_desktop).
  - The announcement encourages community feedback to improve engagement and refine the tool.
- **Transform Your Data with Cyyrus**: A member shared their project, **Cyyrus**, a tool for converting unstructured data into usable datasets for Hugging Face. They hope to assist users in building datasets for various applications, including evaluations and fine-tuning.
  - The tool is still in development, and feedback on its utility would be welcome.
- **Seeking Local RAG System Resources**: A member asked if anyone has created a **RAG system** locally on a Macbook using open-source models and resources. They received a useful link to [LlamaIndex](https://pyimagesearch.com/2024/09/02/llamaindex-building-a-smarter-rag-based-chatbot/#:~:text=LlamaIndex%20provides%20a%20very%20seamless%20way).
  - Discussion on CUDA compatibility on newer Macs followed, reflecting curiosity about optimizing performance in local setups.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://www.fighthealthinsurance.com/">Fight Your Health Insurance Denial -- Use AI to Generate Your Health Insurance Appeal</a>: no description found</li><li><a href="https://huggingface.co/blog/anakin87/spectrum">Selective fine-tuning of Language Models with Spectrum</a>: no description found</li><li><a href="https://pyimagesearch.com/2024/09/02/llamaindex-building-a-smarter-rag-based-chatbot/#:~:text=LlamaIndex%20provides%20a%20very%20seamless%20way">LlamaIndex: Building a Smarter RAG-Based Chatbot - PyImageSearch</a>: Discover how LlamaIndex enhances RAG-based chatbots with smarter indexing and retrieval techniques for more accurate and efficient responses.</li><li><a href="https://github.com/U-C4N/ImageWizard">GitHub - U-C4N/ImageWizard: ImageWizard is a modern web application that offers advanced image processing features like format conversion, compression, pixelation, ASCII art generation, and background removal. Built with Next.js, React, and TypeScript, it provides a user-friendly interface for various image manipulation tasks.</a>: ImageWizard is a modern web application that offers advanced image processing features like format conversion, compression, pixelation, ASCII art generation, and background removal. Built with Next...</li><li><a href="https://github.com/KhashayarRahimi/Reinforcement-Learning-Algorithms-From-Scratch">GitHub - KhashayarRahimi/Reinforcement-Learning-Algorithms-From-Scratch: Implementation of Reinforcement Learning Algorithms (From Reinforcement Learning An Introduction By Sutton &amp; Barto)</a>: Implementation of Reinforcement Learning Algorithms (From Reinforcement Learning An Introduction By Sutton &amp; Barto) - KhashayarRahimi/Reinforcement-Learning-Algorithms-From-Scratch</li><li><a href="https://github.com/mdabir1203/Modular-Rust-Learning">GitHub - mdabir1203/Modular-Rust-Learning: Learning Rust and OOP through Modular Projects</a>: Learning Rust and OOP through Modular Projects. Contribute to mdabir1203/Modular-Rust-Learning development by creating an account on GitHub.</li><li><a href="https://github.com/NotTheStallion/Re-shard_Safetensors">GitHub - NotTheStallion/Re-shard_Safetensors: This repo helps you understand how safetensors are structured to store different layers of an LLM and re-shard/re-chunk safetensors files even if they don't fit in the GPU.. ( No Autoclass )</a>: This repo helps you understand how safetensors are structured to store different layers of an LLM and re-shard/re-chunk safetensors files even if they don&amp;#39;t fit in the GPU.. ( No Autoclass ) -...</li><li><a href="https://github.com/wizenheimer/cyyrus">GitHub - wizenheimer/cyyrus: Transform Unstructured Data into Usable Datasets</a>: Transform Unstructured Data into Usable Datasets. Contribute to wizenheimer/cyyrus development by creating an account on GitHub.</li></ul></div>

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1280241746948919317)** (4 messages):

> - `CV project for Age of Empire II`
> - `Limitations of LLMs in Visual Tasks`
> - `Game Asset Mapping Strategy`
> - `Dynamic Game State Updates`

- **Innovative CV Project for AOE2**: A member proposed a **CV project** to create AI assistants for **Age of Empire II**, focusing on long-term and short-term decision-making strategies.
  - Their approach involves mapping game assets to a text matrix, using computer vision tools like **SAM** and **YOLO** to detect game elements.
- **LLMs Struggle with Visual Object Recognition**: Concerns were raised about the limitations of **state-of-the-art LLMs** like ChatGPT-4, which often fail at counting and localizing objects in images.
  - It was noted that these models mainly describe images rather than make precise observations at coordinate levels.
- **Mapping Game Assets to a Text Matrix**: The proposed strategy involves creating a **text_map** that downscales the game screen while representing key game assets and their movements.
  - The goal is to enhance counting and localization abilities by using a text-based input for the LLM.
- **Concerns on Single Snapshot Game Analysis**: A member expressed skepticism about how much strategy can be deduced from a single snapshot of the game, given the vastness of the map.
  - They suggested that capturing dynamic states could provide more meaningful insights.
- **Dynamic Updates or Game Injection Needed**: Suggestions were made for either maintaining a dynamic update of the text matrix while moving in-game or injecting information directly into the game.
  - This highlights the need for more comprehensive data capture rather than relying solely on computer vision.

 

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1280245637773922437)** (12 messages🔥):

> - `Multi-shot vs Many-shot learning`
> - `Training a custom model with nomic-embed-text-v1.5`
> - `Hugging Face inference endpoint errors`

- **Clarifying Multi-shot and Many-shot Learning Differences**: There was a discussion about the definitions of **few-shot**, **multi-shot**, and **many-shot** learning, with confusion around the latter two terms.
  - *One participant noted that typically terminology includes zero-shot, one-shot, and few-shot, and none involve updating weights during training.*
- **Seeking Guidance on Custom Model Training**: A user inquired about training a custom model using **nomic-embed-text-v1.5** as a base for specific use cases.
  - *They requested help on getting pointed in the right direction for the training process, particularly via direct messaging.*
- **Hugging Face Inference Endpoint Encountering Issues**: Another user reported a
  - *They expressed uncertainty regarding whether the issue originated from Hugging Face or AWS and sought assistance to resolve it.*

 

---

### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1280394652062715999)** (1 messages):

> - `Yolo Diffusion`
> - `Image Masking Techniques`
> - `Computer Vision`
> - `VLM Training`

- **Yolo Diffusion is outdated**: A member noted that **Yolo Diffusion** is an old technique primarily for masking and inpainting with masks, suggesting there are now better approaches available.
  - They recommended asking about this topic in **computer vision** for the most current methods.
- **Stock Level Measurement Misconception**: It was clarified that discussions regarding Yolo Diffusion are not relevant to measuring **stock levels**.
  - The member emphasized the need for a more specialized inquiry into **computer vision**.
- **Training a VLM becomes necessary**: To leverage improved techniques in image processing, one may need to consider **training a Vision Language Model (VLM)**.
  - This suggestion stems from the evolving landscape of image analysis and its applications.

 

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1280254461637300345)** (95 messages🔥🔥):

> - `LM Studio Model Management`
> - `Using Specific GPUs`
> - `Temperature Setting for Testing`
> - `Accessing Multi-Model Functionality`
> - `Text to Image Model Support`

- **LM Studio Model Management Tips**: Users are advised that models saved in different folders cannot be loaded directly but that the model folder can be changed in the 'My Models' view in LM Studio.
  - To load models from another folder, they need to be organized in a specific directory structure within LM Studio.
- **Using Specific GPUs with LM Studio**: A user is experiencing issues with LM Studio not recognizing their GPU, prompting queries about potential troubleshooting steps.
  - Another user suggested checking the Developer Tab for LM Runtimes as a diagnostic step.
- **Temperature Settings for Quality Testing**: Users are discussing the importance of temperature settings in LM Studio for evaluating model outputs, specifically highlighting low settings for quality assessments.
  - A user was recommended to consult beginner's guides on temperature in LLMs for further understanding.
- **Multi-Model Functionality in LM Studio**: There are discussions surrounding running multiple models on separate local server ports, with mixed responses about how to achieve that with current LM Studio functionality.
  - Most users affirmed that loading multiple models within a single instance is feasible, though autoupdating ports may complicate running separate instances.
- **Text to Image Model Support Query**: A user inquired about the availability of text to image generation within LM Studio, received confirmation that it is not currently supported.
  - Alternative suggestions included using external tools like Flux 1 supported in ComfyUI.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://lmstudio.ai">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://lmstudio.ai/docs/text-embeddings">Text Embeddings | LM Studio</a>: Text embeddings are a way to represent text as a vector of numbers.</li><li><a href="https://fastflux.ai/,">FastFLUX | Instant FLUX Image Creation for Free</a>: Create beautiful FLUX images in milliseconds with FastFLUX. Free, fast, and no sign-up required. Image generation powered by Runware.</li><li><a href="https://github.com/lmstudio-ai/lmstudio.js">GitHub - lmstudio-ai/lmstudio.js: LM Studio TypeScript SDK (pre-release public alpha)</a>: LM Studio TypeScript SDK (pre-release public alpha) - lmstudio-ai/lmstudio.js</li></ul></div>

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1280240839645200479)** (142 messages🔥🔥):

> - `Apple Silicon Memory Bandwidth`
> - `Needing Multiple GPUs for LLMs`
> - `Using Unsloth for Fine-tuning`
> - `Performance of Older GPUs for LLMs`
> - `Cache Issues with OpenWebUI`

- **Apple Silicon and Memory Bandwidth Limitations**: While Apple Silicon has impressive memory bandwidth, it remains limited for CPU inference due to restricted access compared to GPUs, with significant power differences in performance.
  - The M1 Max claims **400GB/s** memory bandwidth, but details on how effectively this bandwidth is utilized remain unclear.
- **GPU Resource Awareness for LLMs**: A member plans to use a **2015 Xeon server** with multiple **1070 GPUs** for LLMs, but concerns about performance limitations due to age and specifications were discussed.
  - Using older GPUs like the 1070 may scale memory but compromise speed, with expert opinions suggesting newer models for viable performance.
- **Fine-tuning with Unsloth**: The discussion turned to using the tool **Unsloth** for fine-tuning LLMs, with indications that it could work on current setups without needing a complete hardware overhaul.
  - Members noted advancements in fine-tuning methods could make it feasible without buying high-end rigs, pointing to examples from the community.
- **Performance Expectations of Older GPUs**: Members debated the effectiveness of **older GPUs** like the **GT 1030** and **1070** in inference tasks, with expectations set low for token speeds.
  - While GPUs offer advantages, the performance gain over CPU inference appears marginal and influenced by the model's architecture.
- **Cache Issues with OpenWebUI**: A user reported issues with **OpenWebUI** preloading too much data into cache, consuming an excessive amount of RAM, notably **150GB** out of **192GB**.
  - This unexpected behavior led to concerns and discussions about potential software bugs or misconfigurations in the RAM management strategy.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://www.anandtech.com/show/17024/apple-m1-max-performance-review/2">Apple's M1 Pro, M1 Max SoCs Investigated: New Performance and Efficiency Heights</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/qnguyen3/chat-with-mlx">GitHub - qnguyen3/chat-with-mlx: An all-in-one LLMs Chat UI for Apple Silicon Mac using MLX Framework.</a>: An all-in-one LLMs Chat UI for Apple Silicon Mac using MLX Framework. - qnguyen3/chat-with-mlx</li><li><a href="https://github.com/mlx-chat/mlx-chat-app">GitHub - mlx-chat/mlx-chat-app: Chat with MLX is a high-performance macOS application that connects your local documents to a personalized large language model (LLM).</a>: Chat with MLX is a high-performance macOS application that connects your local documents to a personalized large language model (LLM). - mlx-chat/mlx-chat-app</li><li><a href="https://github.com/preternatural-explore/mlx-swift-chat?tab=readme-ov-file">GitHub - preternatural-explore/mlx-swift-chat: A multi-platform SwiftUI frontend for running local LLMs with Apple's MLX framework.</a>: A multi-platform SwiftUI frontend for running local LLMs with Apple's MLX framework. - preternatural-explore/mlx-swift-chat</li></ul></div>

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1280508786024382498)** (14 messages🔥):

> - `LLM.int8() paper`
> - `Quantization techniques`
> - `Emergent outlier features`
> - `Dynamic vs Static quantization`
> - `Model performance on quantization`

- **Emergent Outlier Features in Quantization**: The conversation revolved around the **emergent outlier features** discussed in the [LLM.int8() paper](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/e4674531dd54874c0abbc786ad5635c92c34dc3e/bitsandbytes/autograd/_functions.py#L318-L321), leading to questions on their relevance for newer LLMs like Llama 2 and 3.
  - *There’s speculation that improvements in training and architectural changes may mitigate these outliers.*
- **Quantization Method Differences**: **Mobicham** pointed out that static quantization for activations can pose issues, while dynamic quantization tends to perform better, citing results from the SmoothQuant paper with large models.
  - They mentioned that while static quantization impacts accuracy, larger model weights are generally easier to quantize without significant loss.
- **Impacts of Outliers on Weight Quantization**: **Mobicham** conducted tests indicating that **outliers in activation** chemistry heavily influence `W8A8` performance, whereas weight-only quantization exhibits minimal impact.
  - They suggested that models like OPT/BLOOM might be more affected due to older training recipes and architecture.
- **Hopper Support and Limitations**: The user **theultrainstinct** noted that int8 in bitsandbytes isn't supported on **Hopper**, questioning the validity of certain claims.
  - They referenced [additional details](https://arxiv.org/pdf/2405.20835v1) about quantization capabilities and thresholds.
- **Threshold Options in Model Quantization**: **Theultrainstinct** mentioned the ability to set an outlier threshold in quantization, enabling skipping the decomposition step, but warned that some models, like OPT, are sensitive to this adjustment.
  - In contrast, models such as **Llama 2/3** and **Mistral** are noted to perform significantly better under these conditions.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://gist.github.com/mobicham/d08684728660f1cafbce94e4e69f7576">outliers_impact_W8A8.py</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://github.com/bitsandbyte">BitsAndByte - Overview</a>: GitHub is where BitsAndByte builds software.</li><li><a href="https://github.com/bitsandbytes-foundation/bitsandbytes/blob/e4674531dd54874c0abbc786ad5635c92c34dc3e/bitsandbytes/autograd/_functions.py#L318-L321">bitsandbytes/bitsandbytes/autograd/_functions.py at e4674531dd54874c0abbc786ad5635c92c34dc3e · bitsandbytes-foundation/bitsandbytes</a>: Accessible large language models via k-bit quantization for PyTorch. - bitsandbytes-foundation/bitsandbytes</li></ul></div>

---

### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1280517490597957694)** (34 messages🔥):

> - `Triton Load Ordering`
> - `Compiler Optimizations`
> - `Performance Tweaks in Triton`
> - `Dummy Conditions in Loops`
> - `Lecture References`

- **Triton Load Order Affects Speed**: Users noted that changing the order of loads in **Triton** can lead to varying speed-ups, with one experiencing a speed-up from **1.89506** to **2.440731** depending on the load order.
  - The notable speed variation raised questions about the compiler's handling of load stalls and instruction scheduling.
- **Compiler Limitations in Reordering Loads**: Discussion highlighted that while **Triton's** compiler can remove unnecessary loads, it lacks the capability for extensive instruction reordering.
  - This means developers may need to manually adjust load orders to optimize performance, counter to typical compiler expectations.
- **Dummy Conditions in Loops Bypass Errors**: It was observed that inserting a dummy condition like `it(k < bound)` in loops can circumvent certain **Triton** errors.
  - This prompted further inquiries into Triton's error handling behavior in loop constructs.
- **Interest in Triton Documentation**: One user referred to **Lecture 14** from the **CUDA Mode** series for additional context regarding Triton.
  - Despite the unclear guidance, users indicated that it remains a useful resource for understanding Triton's functionalities.
- **Investigating Load Order Tweaks**: Users encouraged manual experimentation with load orders in Triton, noting it is a quick way to determine performance variations.
  - This practical approach may help fine-tune future Triton kernels for better efficiency.

 

**Link mentioned**: [lectures/lecture_014/A_Practitioners_Guide_to_Triton.ipynb at main · cuda-mode/lectures](https://github.com/cuda-mode/lectures/blob/main/lecture_014/A_Practitioners_Guide_to_Triton.ipynb): Material for cuda-mode lectures. Contribute to cuda-mode/lectures development by creating an account on GitHub.

 

---

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1280291847360610335)** (1 messages):

> - `App Development Efficiency`
> - `Performance Optimization`
> - `Torch Scaling Techniques`

- **Developers prioritize building over running apps**: A member expressed that they generally spend more time on **building** and **debugging** apps than running them in production environments.
  - *I still want the models I test to run fast* to avoid waiting for results during code changes, suggesting this is a common priority among developers.
- **Direct use of torch._scaled_mm for speed**: To enhance testing efficiency, the member believes that using **torch._scaled_mm** is optimal for running models quickly during code changes.
  - They assume that others who code similarly would likely agree with this performance optimization strategy.

 

---

### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages):

iron_bound: [https://m.youtube.com/watch?v=RIkse0tJ0hE&t=1s](https://m.youtube.com/watch?v=RIkse0tJ0hE&t=1s)

---

### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1280622783013060701)** (6 messages):

> - `PMPP and Synchronization`
> - `Independent Thread Scheduling in Volta`
> - `Warp-Synchronous Programming Deprecation`

- **Understanding Synchronization in Thread Warps**: A user raised confusion over conflicting statements regarding barrier synchronization in PMPP, noting that *both can't be true* about the necessity of __syncthreads() for 32 threads per block.
  - Clarifications indicated that the **PMPP statement is accurate for newer NVIDIA hardware**, while the other reflects older architectures' practices.
- **Volta's Change to Thread Scheduling**: Discussion pointed to Robert_Crovella's answer explaining that Volta introduced **Independent Thread Scheduling**, which deprecated warp-synchronous programming.
  - This change allows developers to implement **fine-grained synchronization** without relying on the implicit behavior of earlier architectures.
- **Technological Shift from Warp-Synchronous Programming**: A user noted that prior methodologies reliant on warp-synchronous programming are now outdated due to improvements made in Volta.
  - The emphasis is shifted towards **explicit synchronization techniques** that leverage the capabilities of the newer architecture.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://forums.developer.nvidia.com/t/32-thread-block-doesnt-need-syncthreads/1490/18">32 thread block doesn't need _syncthreads()?</a>: Interesting topic, I tried on Ampere GPU with a reduced sum application, using 32 threads per block, and it seems that syncthreads() are needed __global__ void global_sum_reduce_kernel(float * arr, f...</li><li><a href="https://developer.nvidia.com/blog/inside-volta/">Inside Volta: The World’s Most Advanced Data Center GPU | NVIDIA Technical Blog</a>: Today at the 2017 GPU Technology Conference in San Jose, NVIDIA CEO Jen-Hsun Huang announced the new NVIDIA Tesla V100, the most advanced accelerator ever built. From recognizing speech to traini...</li></ul></div>

---

### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1280352896050925580)** (13 messages🔥):

> - `RuntimeError in TorchAO`
> - `AWQ w4a16 CUDA kernel porting`
> - `MXLinear Class Error Implementation`

- **RuntimeError in TorchAO with quant_llm_linear**: A user reported a `RuntimeError` indicating that the operator `torchao::quant_llm_linear` already has a fake implementation registered at a specific file location.
  - Another member suggested to *re-install torchao*, mentioning they faced a similar error earlier that morning.
- **Discussion on porting AWQ w4a16 CUDA kernel**: Questions arose about whether to port the AWQ w4a16 CUDA kernel, with a member unsure if it's already being handled by others.
  - One member suggested considering the use of the existing tinygemm kernel, but it was noted that *the tinygemm kernel uses floating point zeros*, which does not work with AWQ.
- **MXLinear Class Implementation Confusion**: A user seeking help with implementing a method in `MXLinear` noted potential confusion regarding type checks in the implementation, particularly around MXTensor types.
  - They later realized that *both weight and input tensors are converted to high precision* before the linear function call, resolving part of their confusion.

 

---

### **CUDA MODE ▷ #[sequence-parallel](https://discord.com/channels/1189498204333543425/1208496482005549086/1280670710221635585)** (1 messages):

> - `Tensor Model Parallelism`
> - `GPU Memory Utilization`

- **Tensor Model Parallelism for Production-Grade Work**: A discussion arose on whether **tensor model parallelism** is suitable for production-grade implementations, suggesting that using **8 GPUs** might be ideal.
  - This division could help achieve the appropriate **shared memory** (smem) requirements for optimal performance.
- **GPU Memory Division Insight**: The idea of dividing model computation across **8 GPUs** was highlighted as a means to achieve the right **shared memory** size.
  - This approach may offer benefits in terms of performance and resource allocation, ensuring that production models run efficiently.

 

---

### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1280549168363995206)** (8 messages🔥):

> - `Burnout management`
> - `CUDA job scarcity`
> - `Niche job dynamics`
> - `Triton and CUDA trends`
> - `OpenGL relevance`

- **Navigating the Burnout Trap**: Members discussed strategies for coping with **burnout**, highlighting the challenges of maintaining motivation in a persistently tough market.
  - *One member mentioned* they would share their insights later, hoping to spark a productive discussion.
- **CUDA Job Market Feels Sparse**: Frustration surfaced over the **rarity of CUDA jobs**, with comments on interviews that promise learning opportunities but exclude candidates lacking experience.
  - *Another member pointed out* that this provides an unfair barrier to entry for many qualified individuals.
- **Double-Edged Sword of Niche Jobs**: A member remarked that **niche jobs** offer fewer applicants but also significantly limit overall opportunities, creating a balancing act in the job market.
  - This sentiment resonated with others, sparking discussions on the implications of pursuing these specialized roles.
- **Triton and CUDA Lead the Charge**: The discussion turned to **Triton and CUDA**, which were noted as prominent in current technology trends, especially in machine learning applications.
  - *One member shared a link* to a [Reddit post](https://redd.it/1f7wumb) emphasizing their relevance in the industry.
- **OpenGL's Surprise Popularity**: The **OpenGL** framework emerged in conversation as surprisingly popular, raising questions about its applicability in current machine learning projects.
  - This comment prompted further inquiries into the reasons behind its sustained interest among developers.

 

**Link mentioned**: [Reddit - Dive into anything](https://redd.it/1f7wumb): no description found

 

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1280282039177842740)** (4 messages):

> - `Activation Checkpointing`
> - `Memory Optimization`
> - `GELU/Layernorm Backward Pass`
> - `Pipeline Parallelism`
> - `FP8 Implementation`

- **Activation Checkpointing Triumph**: A member successfully implemented **activation checkpointing** with surprisingly little code, showcasing different memory requirements based on batch size using **124M BF16**.
  - The memory usage for different configurations included **1211 MiB** with no memory reuse and **176 MiB** when recomputing 100% layers.
- **Memory Savings using GELU/Layernorm**: Recomputing **GELU/Layernorm** in the backward pass effectively reduces memory needs, performing this operation **3 times per layer**.
  - This approach leads to even **lower memory usage**, enhancing efficiency without significantly increasing complexity.
- **Residual Memory Management Suggestions**: The current implementation always saves **residual3** for every layer, but optimizing this could yield greater memory savings at the cost of added complexity.
  - *A member suggested* that combining careful residual management with **Pipeline Parallelism** could effectively leverage GPU storage more efficiently.
- **Pipeline Parallelism Feasibility**: The member expressed confidence that implementing **Pipeline Parallelism** may not be overly complex, albeit more demanding than checkpointing.
  - The intention is to prioritize the implementation of **FP8** after refining the existing features.

 

---

### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/)** (1 messages):

anthonix_tm: Yeah I tried that

---

### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/1280626274947039256)** (2 messages):

> - `Second Wave of Responses`
> - `Third Wave of Responses`

- **Second Wave of Responses Released**: The **second wave of responses** has now been released, indicating progress in gathering feedback from participants.
  - *Anticipation builds* as members await further details on attendance confirmations.
- **Potential Third Wave of Responses**: A third wave of responses will be issued depending on how many people confirm their attendance.
  - This approach aims to ensure that feedback remains relevant and representative of participant interest.

 

---

### **CUDA MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1280243121955012669)** (87 messages🔥🔥):

> - `CUDA kernel requirements`
> - `FP8 support`
> - `Model training issues`
> - `Liger-Kernel PR updates`
> - `CI/CD fixes`

- **CUDA Kernel requires SM_89 for FP8 support**: Discussions highlighted that the kernel requires **SM_89** or higher for native FP8 support, which affects compatibility with certain GPUs like **A100**.
  - Members noted that testing on **4090** achieved a peak improvement of **1.3x** over torch in performance.
- **Training Model Performance Concerns**: A query was raised regarding training a **Qwen2 72B** model with **DeepSpeed Zero3** using the Liger kernel, noting challenges with memory usage and training loss.
  - Suggestions included troubleshooting by disabling Liger features to identify performance issues.
- **Liger-Kernel PR Updates**: Recent PRs addressed conflicts and introduced updates, including a pull request for adding **pyproject.toml** to the repository.
  - There was a call to resolve CI conflicts, with collaborative efforts observed among members to ensure smooth merging.
- **CI/CD Fixes and Improvements**: Members discussed necessary changes to CI/CD configurations, including updates to contributing guidelines to reflect new build systems.
  - PRs aimed at fixing CI issues were shared, with encouragement to merge and validate the changes.
- **Experimental Features and Performance Testing**: Improvements made to the **conv2d** kernel and partial aggregation in **rms_norm** were shared, suggesting benefits in performance.
  - Participants noted intent to include additional benchmarks and optimize the functionalities further with focus on the fleeting **flux** model.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://github.com/linkedin/Liger-Kernel/blob/d338f4b9923e452baecff6d36775242a5319df4c/.github/workflows/publish-release.yml#L27">Liger-Kernel/.github/workflows/publish-release.yml at d338f4b9923e452baecff6d36775242a5319df4c · linkedin/Liger-Kernel</a>: Efficient Triton Kernels for LLM Training. Contribute to linkedin/Liger-Kernel development by creating an account on GitHub.</li><li><a href="https://github.com/linkedin/Liger-Kernel/blob/d338f4b9923e452baecff6d36775242a5319df4c/.github/workflows/publish-nightly.yml#L38">Liger-Kernel/.github/workflows/publish-nightly.yml at d338f4b9923e452baecff6d36775242a5319df4c · linkedin/Liger-Kernel</a>: Efficient Triton Kernels for LLM Training. Contribute to linkedin/Liger-Kernel development by creating an account on GitHub.</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/185">gemm fp8 e4m3 by AndreSlavescu · Pull Request #185 · linkedin/Liger-Kernel</a>: Summary Implemented FP8 gemm with E4M3 representation for FP8. Issue #65 Testing Done tested square matrices of varying sizes (64, 256, 512, 1024, 2048) + non-square matrices of varying sizes ...</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/202/files">ci fix by AndreSlavescu · Pull Request #202 · linkedin/Liger-Kernel</a>: Summary CI Fix Testing Done N/A Hardware Type: RTX 4090 run make test to ensure correctness run make checkstyle to ensure code style run make test-convergence to ensure convergence</li><li><a href="https://github.com/linkedin/Liger-Kernel/blob/e249eee723978bf8610ff1ea2297d048a2417e20/test/transformers/test_cross_entropy.py#L315">Liger-Kernel/test/transformers/test_cross_entropy.py at e249eee723978bf8610ff1ea2297d048a2417e20 · linkedin/Liger-Kernel</a>: Efficient Triton Kernels for LLM Training. Contribute to linkedin/Liger-Kernel development by creating an account on GitHub.</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/175">Monkeypatch for Qwen2-VL by tyler-romero · Pull Request #175 · linkedin/Liger-Kernel</a>: Summary Monkeypatch for the recently-published Qwen2-VL. HF transformers modeling code: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py F...</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/182/files">Feat/faster rms norm by S1ro1 · Pull Request #182 · linkedin/Liger-Kernel</a>: Summary Implements partial aggregation in rms_norm, similar to that in layer_norm, as described in #179 . Testing Done Hardware Type: run make test to ensure correctness run make checkstyle ...</li><li><a href="https://github.com/linkedin/Liger-Kernel/blob/d338f4b9923e452baecff6d36775242a5319df4c/test/convergence/test_mini_models.py#L337">Liger-Kernel/test/convergence/test_mini_models.py at d338f4b9923e452baecff6d36775242a5319df4c · linkedin/Liger-Kernel</a>: Efficient Triton Kernels for LLM Training. Contribute to linkedin/Liger-Kernel development by creating an account on GitHub.</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/150">[BUILD] Add pyproject.toml by AndreSlavescu · Pull Request #150 · linkedin/Liger-Kernel</a>: Summary added pyproject.toml Testing Done ran pip install -e . and it built successfully Hardware Type: RTX 3090 run make test to ensure correctness run make checkstyle to ensure code style ...</li></ul></div>

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1280244417885634680)** (145 messages🔥🔥):

> - `Phishing concerns about a website`
> - `Issues with ComfyUI and Stable Diffusion`
> - `Usage of prompts in Stable Diffusion`
> - `Stable Diffusion 3.1 updates`
> - `Resources for training models and workflows`

- **Phishing Website Warning**: Participants raised concerns about a suspicious website, noting it is likely a phishing hub due to its unsecured HTTP protocol and unencrypted data transmission.
  - *It looks wholly illegitimate* and users should avoid sharing personal information on such sites.
- **ComfyUI Errors and Model Confusion**: Users discussed issues with ComfyUI, particularly an error regarding a missing configuration file and misconceptions on whether certain models were installed.
  - Members suggested using the *Save Text File* node for tracking prompts and workflows within ComfyUI.
- **Prompt Structure for Stable Diffusion**: When using Stable Diffusion, it was noted that prompts structured as attributes separated by commas often yield better results, especially with older models like SD 1.5.
  - However, newer models benefit from using natural language prompts due to their improved text encoding capabilities.
- **Uncertainty Surrounding Stable Diffusion 3.1**: Participants speculated about the potential release of Stable Diffusion 3.1, noting that information was scarce and mostly derived from unofficial sources.
  - There were calls for patience as the community waits for official announcements from Stable AI.
- **Resources for Training Models**: Users expressed a need for guidance on training LoRA models for specific characters and art styles, indicating there's a demand for updated resources.
  - A GitHub repository for Flux was shared, which may assist in understanding updates and workflows related to new model functionalities.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://github.com/mcmonkeyprojects/SwarmUI">GitHub - mcmonkeyprojects/SwarmUI: SwarmUI (formerly StableSwarmUI), A Modular Stable Diffusion Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility.</a>: SwarmUI (formerly StableSwarmUI), A Modular Stable Diffusion Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility. - mcmonkeyprojects/Swa...</li><li><a href="https://github.com/black-forest-labs/flux">GitHub - black-forest-labs/flux: Official inference repo for FLUX.1 models</a>: Official inference repo for FLUX.1 models. Contribute to black-forest-labs/flux development by creating an account on GitHub.</li></ul></div>

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1280625383015710721)** (104 messages🔥🔥):

> - `Mojo Standard Library`
> - `Modular CLI Updates`
> - `Magic CLI Introduction`
> - `MLIR and LLVM Integration`
> - `C++ and Haskell Interop Challenges`

- **Mojo Standard Library is partially open for contributions**: Several members discussed the **Mojo Standard Library**, indicating that some parts are available for contributions while others remain tightly bound to the compiler.
  - However, the **production-ready version is not out yet**, with a stable version existing but lacking robust stability guarantees.
- **Modular CLI nearing final updates**: Updates on the **Modular CLI** suggest it is close to its last release before transitioning to **Magic**, a new tool that will integrate package management capabilities.
  - The team is currently focusing on GPU developments, implying that further CPU-only releases will soon come to an end.
- **Magic CLI's packaging approach similar to Rust's Cargo**: **Magic CLI** is proposed to utilize a conda wrapper, aiming for a more streamlined package management experience akin to **Rust’s Cargo**.
  - Members expressed excitement over avoiding the pitfalls of managing environments like in **pip**, while also ensuring C/C++ dependencies are more accessible.
- **MLIR as a bridge for better language interoperability**: Discussions focused on the potential of an **MLIR backend for Clang** to improve interoperability across programming languages, despite challenges in accurately translating constructs.
  - The consensus is that while it simplifies some aspects, it introduces complexity, particularly concerning backward compatibility and the C preprocessor.
- **Rust's benefits in performance and FFI**: Rust was highlighted as an effective kernel language for tasks requiring speed, especially where pure languages like Haskell might struggle.
  - The conversation noted that Haskell libraries could benefit from linking with Rust to obtain performance improvements, while acknowledging difficulties in establishing common ground between languages.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://github.com/llvm/clangir">GitHub - llvm/clangir: A new (MLIR based) high-level IR for clang.</a>: A new (MLIR based) high-level IR for clang. Contribute to llvm/clangir development by creating an account on GitHub.</li><li><a href="https://docs.google.com/document/d/1zzZC6Kl7Le3Pd124aRb9uXr48APaWKhKBZeISm7s-qs/edit#heading=h.jyh6j2yblt83)">Magic🪄 + Conda Alpha Release Documentation</a>: Magic🪄 + Conda Alpha Release Documentation Introduction We are excited to announce the alpha release of MAX on Conda along with our new package manager called Magic 🪄, which will supersede Modular C...</li></ul></div>

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1280241180394655876)** (24 messages🔥):

> - `Passing Environment Arguments to Mojo Scripts`
> - `Destructor Automatic Calls in Mojo`
> - `InlineFixedVector Usage and Lifecycle`
> - `Weak Reference for Arc`
> - `MaybeUninit Alternatives`

- **Passing Environment Arguments to Mojo Scripts**: To pass environment arguments during script execution, use `mojo run mojoScript.mojo '~/.config/' 2` according to the [Mojo CLI documentation](https://docs.modular.com/mojo/cli/run). Members discussed nuances of how `sys.argv` may cover this use case.
  - One member suggested trying different command formats to see how arguments are processed.
- **Understanding Destructor Calls in Mojo**: Mojo utilizes an **ASAP destruction policy**, destroying objects as soon as they are no longer needed and calling the `__del__()` destructor immediately. The [Mojo lifecycle](https://docs.modular.com/mojo/manual/lifecycle/death) documentation was referenced to clarify this behavior.
  - Members discussed whether certain functions like `!pop.array` require manual destruction or not, leading to varied opinions.
- **Concerns on InlineFixedVector's Design**: The design choice for **InlineFixedVector** to have inline methods instead of leaving that decision to the inliner was discussed; older programming practices were noted as a potential reason. One member speculated that with upcoming changes, **InlineFixedVector** may soon be phased out in favor of simpler data structures.
  - Another member mentioned that improvements could come once the blocking compiler work allows optimizations in Lists.
- **Query on Weak Reference for Arc**: A member inquired if adding a `Weak` reference for **Arc** would be beneficial or if it is currently on hold. This inquiry indicates an interest in enhancing **Arc's** functionality while managing its requirements.
  - There was also discussion regarding `kgen.variant` and whether it implies automatic behavior on initialization or destruction.
- **Exploring Alternatives to MaybeUninit**: One member questioned alternative representations for **MaybeUninit** without using unsafe methods like byte slice punning. Suggestions for maintaining safety while handling uninitialized data were explored.
  - The discussion reflected on avoiding overly broad requirements on types used in **Arc**.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://docs.modular.com/mojo/stdlib/sys/arg/argv">argv | Modular Docs</a>: argv() -&gt; VariadicList[StringRef]</li><li><a href="https://docs.modular.com/mojo/manual/lifecycle/death">Death of a value | Modular Docs</a>: An explanation of when and how Mojo destroys values.</li></ul></div>

---

### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1280522271597002864)** (9 messages🔥):

> - `OSDI '21 Keynote`
> - `Generality of MAX`
> - `Memory Domain Communication`
> - `Compiler Enhancements for Hardware`
> - `Heterogeneous Compute`

- **OSDI '21 Highlight on MAX's Potential**: An insightful keynote from [OSDI '21](https://www.youtube.com/watch?v=36myc8wQhLo) explained how MAX could enhance computing beyond AI and HPC, emphasizing its capability to optimize hardware interaction.
  - It suggests that **Mojo + MAX** may enable utilizing diverse processors effectively, thereby maximizing computational power across systems.
- **The Case for Generality in MAX**: A member affirmed the need for a unifying software to address the complexities of modern heterogeneous computing, expressing confidence in the potential of **Mojo + MAX**.
  - They emphasized the necessity to prevent vendor lock-in and achieve flexibility in languages to better utilize modern hardware advancements.
- **Exploring Advanced Communication Primitives**: There was a discussion about the potential for ambitious communication primitives between memory domains, suggesting improvements over traditional channels.
  - Concerns about existing channels being frictional were raised, questioning the efficiency of current mechanisms for work communication.
- **Memory Domains as Graph Nodes**: It was proposed that memory domains should be represented as graph nodes, detailing various links between them and their characteristics like latency and bandwidth.
  - This approach could empower a hardware-aware compiler to make informed data movement and computation decisions more effectively than manual efforts.
- **The Future of Channel Design**: A member indicated intent to develop a DPDK-based channel due to its performance reputation, acknowledging the friction channels introduce.
  - However, they still see channels as valuable for managing work in environments with variable computation times.

 

**Link mentioned**: [ASPLOS 2021 - Golden Age of Compilers](https://docs.google.com/presentation/d/1ZMtzT6nmfvNOlIaHRzdaXpFeaAklcT7DvfGjhgpzcxk/edit#slide=id.p): The Golden Age of Compilers in an era of Hardware/Software co-design Chris Lattner SiFive Inc April 19, 2021 International Conference on Architectural Support for Programming Languages and Operating S...

 

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1280250187327078430)** (108 messages🔥🔥):

> - `AI and Content Quality`
> - `Job Applications and AI`
> - `LAION Dataset Availability`
> - `AI as a Creativity Tool`
> - `Concerns about AI-generated Content`

- **Debate on AI's Impact on Content Quality**: There is a belief that the rise of AI tools may lead to an increase in low-quality, clickbait content, which some argue diminishes the internet's overall quality.
  - Conversely, others assert that competition in quality among AI-generated content will drive higher standards, leading to improved content relevancy and accuracy.
- **AI Usage in Job Applications**: A discussion highlighted how individuals are utilizing AI to craft customized resumes for job applications, which recruiters then evaluate using AI tools for efficiency.
  - This raises concerns regarding the potential for a 'no human in the loop' scenario and the implications for the quality of assessments in hiring processes.
- **LAION Dataset's Status**: The LAION dataset was discussed in terms of its availability after previously being removed due to concerns regarding its content.
  - Participants confirmed that the dataset is accessible again and mentioned that integration with tools like the Clip retrieval API will be updated shortly.
- **AI as a Creativity Enhancer**: A member proposed that AI could act as a 'creativity multiplier', where skilled users could enhance their productivity significantly through AI tools.
  - However, others worry about the misuse of AI, leading to a proliferation of low-value content in the creative space.
- **Concerns About AI and Disinformation**: The potential for large-scale AI-generated disinformation was noted, with worries about its impact on significant societal outcomes like elections.
  - Participants discussed the necessity for technical advancements to filter and assess the quality of AI-generated content to mitigate overwhelming misinformation.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://rom1504.github.io/clip-retrieval/">Clip front</a>: no description found</li><li><a href="https://aws.amazon.com/ec2/instance-types/inf2/">Compute – Amazon EC2 Inf2 instances – AWS</a>: no description found</li></ul></div>

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1280250504454209597)** (1 messages):

> - `LLM-Based Autonomous Agents`
> - `Manifold Research Group`
> - `Research Log Updates`
> - `MultiNet Evaluation Metrics`
> - `Research Opportunities`

- **Exploring LLM-Based Autonomous Agents**: Manifold Research Group released a position paper titled *Intelligent Digital Agents in the Era of Large Language Models*, providing insights into advancements in LLM-based AI agents and their human-like decision-making capabilities. Interested participants are encouraged to join the conversation on [Discord](https://discord.gg/MfYZmYEGaa?ref=manifoldrg.com) and explore further on their [website](https://www.manifoldrg.com/llm-agents/).
  - The paper discusses both breakthroughs and limitations in the research area, identifying future opportunities for collaboration.
- **Research Log #042 Highlights**: The latest [Research Log](https://www.manifoldrg.com/research-log-042/) from Manifold details their weekly progress on AI projects and notable breakthroughs in the AI community. This ongoing documentation reflects the group's commitment to transparency and innovation in open-source AI.
  - Participants can view shared highlights and join the ongoing discussions related to these important advancements.
- **MultiNet Evaluation Metrics Defined**: Manifold Team has successfully defined the evaluation metrics they plan to use for benchmarking several state-of-the-art Vision-Language Models (VLMs) and Vision-Language Applications (VLAs). The relevant details can be found on their [GitHub repository](https://github.com/ManifoldRG/MultiNet?ref=manifoldrg.com).
  - For detailed dataset coverage, the team has provided insights through this [link](https://github.com/ManifoldRG/MultiNet/issues/19?ref=manifoldrg.com).
- **Open Source Team Opportunities**: Manifold Research Group seeks individuals to contribute meaningfully through various research projects and operational roles, emphasizing their commitment to open-source collaboration. Interested candidates can find more on their [opportunities page](https://www.manifoldrg.com/opportunities/).
  - The OS Team is looking for passionate volunteers, and applicants are advised to review the [OS Research Team Expectations](https://docs.google.com/document/d/e/2PACX-1vQgq32ChlP_e26mRPgfC31lZJCcAHAgbJ_Tn1nfzq8pfysoPAUqAWnel87Qc26h2Q/pub) before applying.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://www.manifoldrg.com/llm-agents/">Intelligent Digital Agents in the Era of Large Language Models</a>: B Faught, H Lu, T Marshall, H Sikka, P Guruprasad, B Gauri (2024)</li><li><a href="https://www.manifoldrg.com/research-log-042/">Research Log #042</a>: Welcome to Research Log #042! We document weekly research progress across the various initiatives in the Manifold Research Group, and highlight breakthroughs from the broader research community we thi...</li><li><a href="https://www.manifoldrg.com/opportunities/">Manifold Research Group (Page 1)</a>: no description found</li></ul></div>

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1280250026802675796)** (12 messages🔥):

> - `Manifold Research Group's Position Paper`
> - `Compute Availability from Manifold`
> - `ICLR vs NIPS Workshop Publication Impact`
> - `Code Analogies to TinyStories`

- **Manifold Research Group shares recent paper**: Luke from the Manifold Research Group introduced their position paper on [LLM Based Autonomous Agents](https://www.manifoldrg.com/llm-agents/), highlighting key advancements in the field.
  - They encouraged interested individuals to join their [Discord community](https://discord.gg/MfYZmYEGaa?ref=manifoldrg.com) and check their [GitHub](https://github.com/ManifoldRG?ref=manifoldrg.com).
- **Limited compute offerings at Manifold**: Luke confirmed that Manifold offers limited compute as part of various academic and industry partnerships, but specifics depend on the project and team.
  - For detailed inquiries about available compute resources, contacting **Harsh** or **Sidh** directly was recommended.
- **ICLR has a higher CV impact than NIPS workshops**: A member mentioned that having a paper in the main **ICLR conference** is significantly better for a CV than having one in a **NIPS workshop** due to lower acceptance criteria at workshops.
  - ICLR is recognized as a **tier 1 conference**, making it more prestigious.
- **Linux Kernel Codebase as 'TinyStories' for Code**: In response to a question about code resources similar to **TinyStories**, a member humorously referenced the **Linux kernel codebase**.
  - Another member suggested **K&R** (Kernighan and Ritchie), likely referring to the classic computer science book that is also foundational programming knowledge.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://www.manifoldrg.com/llm-agents/">Intelligent Digital Agents in the Era of Large Language Models</a>: B Faught, H Lu, T Marshall, H Sikka, P Guruprasad, B Gauri (2024)</li><li><a href="https://www.manifoldrg.com/research-log-042/">Research Log #042</a>: Welcome to Research Log #042! We document weekly research progress across the various initiatives in the Manifold Research Group, and highlight breakthroughs from the broader research community we thi...</li><li><a href="https://www.manifoldrg.com/opportunities/">Manifold Research Group (Page 1)</a>: no description found</li></ul></div>

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1280372624442134619)** (34 messages🔥):

> - `Feedback on New Concepts`
> - `LLM Abstraction-Crystallization`
> - `Diffusion Models and Physics`
> - `Timestep Modifications in Diffusion Models`
> - `MoE Training with H100 GPUs`

- **Seeking Feedback on Novel Concepts**: A member expressed an interest in getting feedback on a new concept they've been developing, concerned about whether it would be annoying.
  - Another member encouraged them to share, reassuring that it can't hurt and they might learn about related existing work.
- **LLMs Lack an Abstraction-Crystallization Step**: A proposal was made that LLMs could benefit from a step that allows them to evaluate multiple abstracted phrases, enhancing their output potential.
  - The idea includes ranking relevant phrases by their vector similarity to prompts, which could yield more creative responses rather than relying solely on top probability outputs.
- **Concerns about Diffusion Models Understanding Physics**: A discussion emerged regarding whether diffusion models can truly learn physical laws or if they simply overfit to datasets.
  - One member highlighted that imposing physical structures could reduce model expressivity, raising concerns about learning such constraints.
- **Modifying Weights with Timesteps in Diffusion Models**: There was speculation about works that modify the weights of the diffusion U-Net using timesteps, rather than just adjusting the inputs.
  - One member noted that typical adaptive norms in diffusion models change their scales and biases based on the timestep.
- **MoE Training and H100 GPU Performance**: Questions arose regarding how to accurately assess the efficiency of MoE training on H100 GPUs, especially related to sparse operations.
  - One member clarified that the sparse tensor cores in H100 are distinct from MoE sparsity, suggesting that marketing claims may not align with practical benefits.

 

**Link mentioned**: [Can a machine learn mathematical structure?](https://pr4-kp.github.io/posts/machine-learning-sl2z/): A discussion of my research work last semester to use machine learning to answer questions in algebra

 

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1280240845853036585)** (31 messages🔥):

> - `Transformers and Token Embeddings`
> - `MLP Layers in Transformers`
> - `Interpretability Across Training Checkpoints`
> - `Transformers as Graph Neural Networks`

- **Understanding Token Embeddings in Transformers**: Members discussed how the transformer learns a vector that is **VocabSize x EmbeddingDimension**, asserting that each token has a corresponding embedding.
  - The **attention heads** are key in allowing each token to impact others by generating a **QK softmax** over the input and multiplying this by the token embeddings.
- **Role of MLP Layers in Combining Token Information**: The MLP in transformers expands and then reduces the embedding dimension, but notably, it does not mix token information across tokens.
  - Weights in the MLP are **shared across tokens**, allowing for neuron activations to be tracked per token effectively.
- **Interpretability of Neurons Over Training**: A question was raised about whether the interpretability of a model's neurons changes over training, particularly with different checkpoints in the **Pythia** model.
  - The hypothesis suggests interpretability may fluctuate, potentially starting low, increasing, and then decreasing due to superposition effects.
- **Transformers and Graph Neural Networks Connection**: A member compared the transformer block to a **graph neural network**, suggesting that it creates per-sentence **adjacency matrices** during operation.
  - The attention mechanism's similarity to node-edge graph connections was noted, particularly in how attention patterns might capture multi-hop relationships.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://github.com/PonceLab/circuit_toolkit">GitHub - PonceLab/circuit_toolkit: Common utility functions for insilico experiments for visual neuroscience</a>: Common utility functions for insilico experiments for visual neuroscience - PonceLab/circuit_toolkit</li><li><a href="https://transformer-circuits.pub/2021/framework/index.html">A Mathematical Framework for Transformer Circuits</a>: no description found</li></ul></div>

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1280442007663411215)** (2 messages):

> - `lm-evaluation-harness issue`
> - `Maintainer response`

- **Request for feedback on lm-evaluation-harness issue**: A member requested feedback from a maintainer on [this issue](https://github.com/EleutherAI/lm-evaluation-harness/issues/2268) to help move the project forward.
  - They also expressed willingness to contribute further if applicable.
- **Maintainer acknowledges issue request**: A maintainer responded, thanking the member for opening the issue and confirmed they will take a look.
  - This indicates a positive engagement and support from the maintenance team.

 

**Link mentioned**: [Issues · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/issues/2268),): A framework for few-shot evaluation of language models. - Issues · EleutherAI/lm-evaluation-harness

 

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1280612868722528399)** (22 messages🔥):

> - `PyTorch and CUDA compatibility`
> - `Deepspeed issues`
> - `Model codebases comparison`
> - `Training configurations`
> - `Testing and merging features`

- **Troubleshooting PyTorch and CUDA**: Members discussed resolving issues related to **PyTorch** version **2.4** and CUDA compatibility, particularly focusing on downgrading PyTorch to avoid installation problems with **flash attention**.
  - It was suggested that installing a **PyTorch wheel** compatible with the local CUDA version would fix installation issues, with specific links shared for reference.
- **Deepspeed bugs acknowledged**: A known bug related to **Deepspeed** was highlighted, including a GitHub link that provided a minor fix to resolve an **import error** caused by changes in **Torch**.
  - One member confirmed resolving previous import errors but anticipated further complications with settings, indicating that merging might introduce new issues.
- **Difficulties in pretraining implementations**: Concerns were raised regarding various pretraining codebases like **Nanotron** and **OLMO**, noting that they often lack compatibility with alternate transformers and parallelism schemes.
  - Members expressed that certain repos only support basic implementations, driving interest in **GPT-2 variants** with different positional encodings, highlighting **Neox** as a standout.
- **Seeking insights on training configurations**: Community members are now keen to utilize newly acquired **H100 GPUs** for enhancements instead of merely fine-tuning existing models.
  - There were discussions about the potential for breakthroughs with new configurations, and members were invited to share insights about their experiences.
- **Collaborative merging efforts**: One member offered to assist in merging and testing new features, emphasizing the need for collaborations in development.
  - Members encouraged sharing findings for potential promotion, demonstrating a supportive atmosphere for innovation and improvement in codebases.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://pytorch.org/get-started/previous-versions/">Previous PyTorch Versions</a>: Installing previous versions of PyTorch</li><li><a href="https://github.com/microsoft/">Microsoft</a>: Open source projects and samples from Microsoft. Microsoft has 6357 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/microsoft/DeepSpeed/pull/5346">logger update with torch master changes by rogerxfeng8 · Pull Request #5346 · microsoft/DeepSpeed</a>: minor fix to resolve the logger import issue caused by torch upstream cleanup pytorch/pytorch@b6201a6 log variable was renamed in the torch master. To create the logger using public API to avoid co...</li><li><a href="https://github.com/microsoft/DeepSpeed/pull/5346/files">logger update with torch master changes by rogerxfeng8 · Pull Request #5346 · microsoft/DeepSpeed</a>: minor fix to resolve the logger import issue caused by torch upstream cleanup pytorch/pytorch@b6201a6 log variable was renamed in the torch master. To create the logger using public API to avoid co...</li></ul></div>

---

### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1280625431774761010)** (2 messages):

> - `Free Perplexity Pro for Students`
> - `Campus Signup Challenge`
> - `Leaderboards and Incentives`

- **Students Score Free Month of Perplexity Pro**: Students can get a **free month of Perplexity Pro** by signing up with their .edu email before **September 15**. The service provides quick, accurate answers, making it perfect for tackling academic challenges.
  - Perplexity offers solutions ranging from explaining complex topics to making meal plans based on available ingredients.
- **Whole School Wins Free Access at 500 Signups**: If a campus reaches **500 signups**, the entire school will receive **one year of Perplexity Pro** for free. Participants are encouraged to spread the word and get their friends involved to achieve this goal.
  - This promotion is available until **September 15**, and details about current signups can be tracked [here](https://www.perplexity.ai/backtoschool).
- **Visuals Supporting Signup Campaign**: The announcements included several engaging visuals promoting the free month of service and the signup challenge. This creative approach aims to increase user interest and participation.
  - The visuals emphasize excitement and competition, aiming to motivate students to take advantage of this offer.

 

**Link mentioned**: [Perplexity - Race to Infinity](https://www.perplexity.ai/backtoschool): Welcome back to school! For just two weeks, redeem one free month of Perplexity Pro on us. Refer your friends, because if your school hits 500 signups we'll upgrade that free month to an entire free y...

 

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1280257905169272954)** (87 messages🔥🔥):

> - `Perplexity Pro Sharing Options`
> - `Copilot Rebranding`
> - `Xfinity Pro Subscription`
> - `Student Discounts`
> - `Usage Issues with Pro`

- **Perplexity Pro Sharing Options**: Members have inquired about sharing Perplexity Pro subscriptions with family, but currently, there are no sharing options available.
  - *Consider suggesting improvements in the community channel* as it's noted there are no existing family share options.
- **Copilot Transition**: A user was confused about enabling Copilot, which has been rebranded as 'Pro', leading to some misunderstandings.
  - Clarifications were made regarding the naming change, with no specific corresponding activation option.
- **Xfinity Pro Subscription Benefits**: It was mentioned that users who signed up for Pro through Xfinity can utilize a code for additional uses, hinting at promotional offers.
  - A user confirmed they were able to use the promo code multiple times, allowing for more flexibility.
- **Discrepancies in Student Discounts**: Various users expressed frustration with the limited availability of student discounts, questioning why it mostly applies to US schools.
  - Participants shared experiences of not receiving offers or being eligible due to regional email domains, advocating for inclusivity.
- **Usage Issues with Pro**: One member reported encountering a paywall after a limited number of searches, causing confusion over the differences between free and Pro access.
  - Others chimed in, sharing similar experiences and suggesting troubleshooting methods like rejoining channels.

 

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1280319517070725213)** (8 messages🔥):

> - `Perplexity Xfinity Deal`
> - `Morning Routine`
> - `DNA Development Leaders`
> - `Claude Powers Amazon's Alexa`
> - `Proxy Between Backend`

- **Perplexity Xfinity Deal Surfaces**: A link was shared regarding a [Perplexity Xfinity deal](https://www.perplexity.ai/search/perplexity-xfinity-deal-QCK.FX71SZCO6kSpE0YtYQ). The details might reveal exciting offerings or partnerships for users.
- **Unpacking Morning Routines**: An exploration into *what makes a good morning routine* was highlighted in a [new article](https://www.perplexity.ai/search/what-is-the-morning-routine-fo-W691GG9LQEuBaar1MqxNHw). This could provide insights into effective start-of-day practices.
- **Insights on DNA Development**: A query was raised about who leads development in *DNA computing*, with a link showing detailed insights [here](https://www.perplexity.ai/search/who-leads-development-in-dna-c-EKWUUF.BTUKLyifMrQcY9w). This could lead to greater understanding of this cutting-edge field.
- **Amazon's Alexa Powered by Claude**: An intriguing video surfaced about *Amazon's Alexa being powered by Claude*, exploring neuroscience and computing all in one framework [link](https://www.youtube.com/embed/oTWCM4aIA5g). This brings to light advancements at the intersection of AI and cognitive science.
- **Proxy Between Backend Usage**: A discussion shedding light on *why to use a proxy between the backend* was provided in a [shared link](https://www.perplexity.ai/search/why-use-proxy-between-backend-iUaKITv2QsCuUI4V7It.JA). This understanding is crucial for efficient backend architecture.

 

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1280500001511837759)** (3 messages):

> - `Perplexity API usage`
> - `File upload capabilities`
> - `Make.com integration`

- **Creating a Perplexity page via API**: A user inquired about the possibility of creating a Perplexity page using the API, specifically asking about integration with [Make.com](https://make.com).
  - Another member responded negatively, suggesting checking the official [Perplexity documentation](https://docs.perplexity.ai) for more information.
- **File upload support in pplx-api**: A user asked if the Pro API allows for file uploads (e.g., .txt, .pdf) in the search query payload when using the CLI interface.
  - The inquiry emphasized wanting the same file upload functionality as available in the web interface for improved analysis.

 

**Link mentioned**: [no title found](https://docs.perplexity.ai): no description found

 

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1280547914141401108)** (1 messages):

> - `Mistral price drop`

- **Mistral-Nemo's Price Takes a Hit**: The price of [Mistral-Nemo](https://openrouter.ai/models/mistralai/mistral-nemo) has dropped by **23%**, reflecting changes in market dynamics.
  - This significant price change might indicate a shift in demand or supply for the **Mistral** models.
- **Market Reactions to Mistral-Nemo's Price Drop**: Industry analysts are keenly observing the **23%** price drop of [Mistral-Nemo](https://openrouter.ai/models/mistralai/mistral-nemo) to understand its impact on competitors.
  - Some traders believe this could lead to an influx of users exploring alternative options.

 

**Link mentioned**: [Mistral Nemo - API, Providers, Stats](https://openrouter.ai/models/mistralai/mistral-nemo>)): A 12B parameter model with a 128k token context length built by Mistral in collaboration with NVIDIA. The model is multilingual, supporting English, French, German, Spanish, Italian, Portuguese, Chin...

 

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1280553288378355722)** (2 messages):

> - `Mume AI App Launch`
> - `Feedback Request`
> - `Free Tier Availability`

- **Mume AI App Debuts with Excitement**: The **Mume AI** app, short for **Muse Mesh**, has been launched using OpenRouter as a provider, marking an exciting milestone for the developer in this burgeoning space.
  - Users can explore over **100 models** that offer text and image generation capabilities and **vision-enabled models**.
- **Developer Encourages Community Feedback**: The developer expressed enthusiasm for receiving **feedback** from the community to improve Mume AI, emphasizing that it's just the beginning of many milestones.
  - It was highlighted that every bit of feedback would be valuable as the app is still in its **early stage** of development.
- **Free Tier Offers Daily Tokens**: Mume AI features a **free tier** that provides users with tokens every day, similar to the initial experience the developer had with OpenRouter’s free tier.
  - This feature encourages users to try out the app while making it accessible for a broader audience.
- **Cross-Platform Availability**: Mume AI is accessible on both the **App Store** and **Play Store**, allowing users to download and engage with the app seamlessly.
  - The app supports a range of features including **multimodal learning** and generating creative content through various model categories.
- **User-Friendly Interface Features**: The app boasts a sleek interface with **light and dark modes** tailored to the user's system theme, helping to maintain focus on tasks.
  - Its organized structure allows users to **explore** models by categories like **Marketing**, **Science**, and **Technology**.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://apps.apple.com/us/app/mume-ai/id6523427150">‎Mume AI</a>: ‎~ Access 100+ models with chat interface, brainstorm about ideas, get creative inspiration ~ Learn from images with wide range of multimodal models that recognise images ~ Generate beautiful images f...</li><li><a href="https://play.google.com/store/apps/details?id=ai.musemesh.mume">Mume AI - Apps on Google Play</a>: no description found</li></ul></div>

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1280260741106106489)** (83 messages🔥🔥):

> - `Caching with Google and Claude models`
> - `Multi-turn conversations in OpenRouter`
> - `Character consistency in AI models`
> - `Using OpenRouter with Cursor and ContinueDev`
> - `Refund request for accidental charge`

- **Caching capabilities for Google and Claude models**: Members discussed the potential for caching with **Google** and **Claude** models through OpenRouter, with indications that the feature is close to being implemented.
  - However, concerns were raised about cache routing due to the two endpoints not sharing the same cache.
- **Clarification on multi-turn conversations support**: A user inquired about the support for **multi-turn conversations** in OpenRouter, which prompted discussions on the necessity to resend the entire chat history for maintaining continuity.
  - Responses indicated that users need to handle this aspect on their end since LLMs are stateless.
- **Best models for character consistency in AI**: A user sought advice on the best models for maintaining character consistency, mentioning that **Midjourney** is not satisfactory, while another suggested **Segmind** as a potential solution.
  - The conversation highlighted the desire to create an Instagram AI influencer and ways to achieve more reliable outputs.
- **Challenges using OpenRouter with other providers**: A member expressed issues using OpenRouter with **Cursor**, indicating that Cursor requires all requests to go through them for privacy concerns.
  - Additional inquiries involved the difficulties faced when trying to utilize **ContinueDev** with OpenRouter, with documentation suggesting solutions.
- **Refund request for accidental charge**: A user requested a refund after accidentally charging themselves **$174**, expressing distress about the situation.
  - The request highlights the need for clear user support regarding billing issues.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://docs.continue.dev/reference/Model%20Providers/openrouter">OpenRouter | Continue</a>: OpenRouter is a unified interface for commercial and open-source models, giving you access to the best models at the best prices. You can sign up here, create your API key on the keys page, and then c...</li><li><a href="https://openrouter.ai/docs/frameworks#using-openai-sdk">Frameworks | OpenRouter</a>: Frameworks supporting model integration</li><li><a href="https://www.langchain.com/">LangChain</a>: LangChain’s suite of products supports developers along each step of their development journey.</li><li><a href="https://github.com/billmei/every-chatgpt-gui/blob/main/README.md">every-chatgpt-gui/README.md at main · billmei/every-chatgpt-gui</a>: Every front-end GUI client for ChatGPT. Contribute to billmei/every-chatgpt-gui development by creating an account on GitHub.</li></ul></div>

---

### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1280594847895523411)** (1 messages):

> - `NousCon Event`
> - `PyTorch Conference`
> - `San Francisco`

- **NousCon Event Announced for September 18**: We are hosting the **NousCon** event in **San Francisco** on **September 18** following the **PyTorch Conference**.
  - Limited space is available, and more details can be found in the [official announcement](https://x.com/NousResearch/status/1831032559477866754) and registration link [here](https://lu.ma/zlgp0ljd).
- **Limited Space for NousCon**: Participants are advised that the **NousCon** event has **limited space**, highlighting the need for early registration.
  - Attendees can secure their spot through the provided [registration link](https://lu.ma/zlgp0ljd).

 

**Link mentioned**: [Tweet from Nous Research (@NousResearch)](https://x.com/NousResearch/status/1831032559477866754): NousCon, September 18th, San Francisco, Limited Space. [https://lu.ma/zlgp0ljd](https://lu.ma/zlgp0ljd)

 

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1280245978263191572)** (56 messages🔥🔥):

> - `Hermes-3 Training Efficiency`
> - `Gender Ratio Among Creators`
> - `Scammer Engagement Strategies`
> - `Pronunciation of 'Nous'`
> - `Hermes Aesthetics`

- **Hermes-3 trains at lightning speed**: Hermes-3's training can be completed in just **4 minutes**, prompting remarks about the efficiency of current model training techniques.
  - Members joked about 'speedrunning training' due to this remarkable efficiency.
- **Curiosity over Hermes creators' gender dynamics**: A member humorously inquired about the **gender ratio** among Hermes' creators, showing interest in the diversity behind the model.
  - This sparked a light-hearted discussion about the significance of representation in AI development.
- **Innovative ways to combat scammers with Hermes**: A member proposed using Hermes to **waste scammers' time**, suggesting it could engage them without revealing the user's identity.
  - This led to a discussion on the potential for benchmarking how long Hermes could keep scammers occupied.
- **Insights on how to pronounce 'Nous'**: The community engaged in a discussion about the **pronunciation of 'Nous'**, revealing interesting nuggets about its linguistic roots.
  - Any confusion was cleared up, with some members jesting about the implications of silent letters.
- **Admiration for Hermes' aesthetics**: Members expressed their awe at the unmatched **aesthetics** of Hermes, attributing its visuals to a particular creator.
  - This prompted further praise and comments about the overall design and appeal of the Hermes brand.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://tenor.com/view/luh-calm-fit-hazbff-opium-bird-stoon-gif-3957809579245532765">Luh Calm Fit Hazbff GIF - LUH CALM FIT HAZBFF OPIUM BIRD - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://developer.nvidia.com/blog/nvidia-sets-new-generative-ai-performance-and-scale-records-in-mlperf-training-v4-0/">NVIDIA Sets New Generative AI Performance and Scale Records in MLPerf Training v4.0 | NVIDIA Technical Blog</a>: Generative AI models have a variety of uses, such as helping write computer code, crafting stories, composing music, generating images, producing videos, and more. And, as these models continue toR...</li><li><a href="https://developer.nvidia.com/blog/nvidia-blackwell-platform-sets-new-llm-inference-records-in-mlperf-inference-v4-1/">NVIDIA Blackwell Platform Sets New LLM Inference Records in MLPerf Inference v4.1 | NVIDIA Technical Blog</a>: Large language model (LLM) inference is a full-stack challenge. Powerful GPUs, high-bandwidth GPU-to-GPU interconnects, efficient acceleration libraries, and a highly optimized inf...</li></ul></div>

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1280440022104805457)** (1 messages):

> - `LLM Planning and Reasoning`
> - `Yann LeCun's concepts`
> - `LLM-Modulo architecture`

- **Seeking Insights on LLM Planning and Reasoning**: A member inquired about any updates regarding **LLM Planning and Reasoning**, expressing difficulty in finding remarkable frameworks that address this area at its core.
  - They noted that concepts like those proposed by **Yann LeCun** seem more realistic but still lack comprehensive solutions for fundamental LLM reasoning and planning challenges.
- **Concerns About LLM-Modulo Concept**: The same member commented that the **LLM-Modulo concept** does not seem impressive in addressing the critical aspects of LLM reasoning and planning.
  - They expressed a desire to connect with others who are actively discussing or working on architecture to fundamentally solve these issues.

 

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1280490116829417535)** (2 messages):

> - `Gemma 2 Implementation`
> - `Numpy and CuPy Notebooks`

- **Introducing Gemma 2: Numpy to CuPy Transition**: A member reported working on implementing **Gemma 2** from scratch using **Numpy** before porting it to **CuPy**.
  - They provided links to the notebooks for both implementations: [Numpy Notebook](https://github.com/githubpradeep/notebooks/blob/main/gemma2%20final.ipynb) and [CuPy Notebook](https://github.com/githubpradeep/notebooks/blob/main/gemma2%20final_cupy.ipynb).
- **Guidelines for Running CuPy Notebooks**: For the **CuPy notebook**, it is recommended to use a **GPU with 24GB** of memory for optimal performance.
  - Alternatively, for GPUs with less than **16GB**, users should utilize the [CuPy f16 notebook](https://github.com/githubpradeep/notebooks/blob/main/gemma2%20final_cupy_f16.ipynb) while the **Numpy notebook** is suitable for CPU runs.

 

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1280440022104805457)** (1 messages):

> - `LLM Reasoning Frameworks`
> - `Yann LeCun's Concepts`
> - `LLM-Modulo Approach`
> - `Architecture for LLM Planning`

- **Questioning LLM Reasoning Frameworks**: Members expressed a lack of **remarkable frameworks** that effectively address LLM **Reasoning and Planning** at their core.
  - *One member reflected on the need for concepts that genuinely solve reasoning issues*, citing Yann LeCun's views as potentially more practical.
- **Skepticism Towards LLM-Modulo**: There was skepticism regarding the **LLM-Modulo concept**, which did not impress some members.
  - *Concerns were raised about its efficacy*, prompting calls for discussions on fundamentally solving LLM reasoning and planning challenges.
- **Desire for Collaboration on LLM Solutions**: Members expressed a desire to connect with others, specifically mentioning collaboration opportunities regarding LLM frameworks.
  - *Interest was shown in engaging with key individuals* in the field to explore innovative solutions for reasoning and planning.

 

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1280285075535036598)** (31 messages🔥):

> - `SearchGPT release speculation`
> - `AI in gaming`
> - `Simulation and consciousness`
> - `AI model performance`
> - `Community feedback on ChatGPT`

- **SearchGPT release buzz**: A user speculated that SearchGPT might be released soon, noting some who signed up for the waitlist briefly saw a pop-up saying 'You're in'. However, access to the service was not achieved as the pop-up disappeared quickly.
  - Despite anticipation, another user argued that **Perplexity** currently outperforms SearchGPT, and that **Arc** has Perplexity integrated, making it a better choice.
- **AI playing UNO for video**: A member suggested creating a video with AI playing UNO and asked for insight on whether to proceed. Engagement in AI gaming sparked discussions about creative applications.
  - This showcases the ongoing interest in AI-led content creation and interactive experiences.
- **Redefinition of simulation**: A user proposed a redefinition of 'simulation', emphasizing the conscious role of the observer in interpreting experiences. This shifts the focus from external conditions to internal processes, especially in contexts like Virtual Reality.
  - Feedback was solicited from the community to evaluate the clarity and validity of this philosophical stance.
- **Frustration with ChatGPT policies**: A member expressed dissatisfaction with ChatGPT's handling of sensitive topics, noting a shift in response patterns and message deletions. They conveyed the sentiment that such behavior could drive users away from the platform if not addressed.
  - This discussion highlights ongoing concerns about user experiences in AI interactions, particularly around policy enforcement.
- **Community Suggestions for Improvement**: In light of frustrations, a user advised others to voice their concerns in dedicated feedback channels for effective change. The remarks highlighted the community's call for more transparent and responsive support from AI developers.
  - This points to a critical need for engagement between AI providers and users regarding policy and service improvements.

 

**Link mentioned**: [Tweet from Boris Power (@BorisMPower)](https://x.com/BorisMPower/status/1830714579116323004): @Dr_Singularity I’m sorry we failed you and thanks for the patience - hopefully we rectify this soon and make the subscription way more valuable

 

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1280571814765924394)** (4 messages):

> - `GPT-4o Features`
> - `ChatGPT File Saving Issues`

- **GPT-4o outperforms GPT-4 Turbo**: GPT-4o is **50% cheaper** than GPT-4 Turbo at **$5/M input and $15/M output tokens**, featuring **2x speed** and **5x higher rate limits** up to **10 million tokens per minute**.
  - Its context window is **128k**, and it has superior **vision capabilities** and **multilingual support** compared to GPT-4 Turbo, making it a compelling option for users.
- **File saving issues in ChatGPT**: A user reported encountering errors when trying to save files in ChatGPT, with the system indicating an issue with retrieving download links for updated text sections.
  - Despite using the **plain txt** format, the user faced obstacles that suggest potential disruptions or limitations in the current file service, expressing frustration over the functionality after previously saving larger texts successfully.

 

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1280265229623754832)** (4 messages):

> - `Instructions for Casual Writing`
> - `Positive vs Negative Examples`
> - `Behaviorism and Positive Reinforcement`
> - `Handling Taboos in Writing`

- **Instructions for Casual Writing**: A member expressed the desire to avoid overly complex or humorous sentences, urging a more casual style with simple words.
  - They pointed out issues with responses that still included unwanted phrases, highlighting the need for clarity in instructions.
- **Positive vs Negative Examples**: Another member suggested providing positive examples of language to use instead of negative examples of what to avoid, focusing on desired phrases.
  - This included a list of acceptable terms to guide the model away from undesirable phrasing.
- **Behaviorism and Positive Reinforcement**: A member supported the idea of emphasizing what the model should do rather than what it should not, likening it to behavioral techniques.
  - They explained that positive reinforcement could lead to better outcomes than negative reinforcement.
- **Handling Taboos in Writing**: A member remarked on the complexity of writing about taboo topics, comparing it to handling dangerous materials like radium.
  - They emphasized the need for careful caution and consideration in such instances to ensure appropriate handling.

 

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1280265229623754832)** (4 messages):

> - `Avoiding unwanted phrases`
> - `Positive reinforcement in instructions`
> - `Guiding model behavior`

- **Flaskie's feedback on instruction clarity**: A member expressed frustration with the model responding with unwanted phrases despite clear instructions to avoid them.
  - They argued for better model guidance toward positive examples rather than focusing on negatives.
- **Importance of positive instructions**: Another member emphasized that it's more effective to instruct the model on what to do rather than what to avoid.
  - They provided a behavioral perspective, suggesting that positive reinforcement encourages desired outcomes more effectively.
- **Concerns about model behavior patterns**: A member commented on the model's tendency to repeat contextually seen phrases, even when framed by instructions to avoid them.
  - They illustrated this with an example about the model's responses to commands, suggesting a fundamental challenge in how it interprets instructions.
- **Caution with sensitive topics**: A member drew an analogy between handling sensitive subjects and managing dangerous materials like radium, indicating the need for care.
  - They implied that navigating taboos requires a careful approach, acknowledging the complexities involved.

 

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1280324241421500431)** (2 messages):

> - `Auto-Document Retrieval`
> - `LLMs for Presentation Generation`

- **Auto-Document Retrieval Enhances RAG Efficiency**: A recent notebook demonstrates how to combine **RAG (Retrieval-Augmented Generation)** with structured querying for better document retrieval, especially when dealing with large datasets, as noted in a [related post](https://t.co/nfXnpvM93D).
  - *How do you retrieve the right documents?* This approach aims to address that question effectively.
- **LLMs Generate PowerPoint Decks from Notes**: An innovative TypeScript app allows users to convert notes into **PowerPoint slides**, freeing them from tedious structural tasks to focus on creativity, showcasing the power of **LLMs**.
  - The app not only summarizes speaking notes into slides but also generates additional content, as detailed in a [demo link](https://t.co/ItJ3edWmXF).

 

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1280282216076935188)** (37 messages🔥):

> - `Jina AI Late Embeddings`
> - `Gemini LLM Issues`
> - `Filtering Message History in ChatEngine`
> - `Q&A on VectorStoreIndex`
> - `Local Equivalent for Tavily Tool`

- **Jina AI Late Embeddings Class Proposal**: A member suggested creating an embeddings class for **Jina** to leverage the new 'late embeddings' approach via [HF](https://github.com/jina-ai/late-chunking/tree/main/chunked_pooling). Another member noted that most of the code could potentially be integrated into a node parser package by implementing the BaseNodeParser class.
- **Gemini LLM Facing Initialization Error**: A user reported an **AttributeError** related to the **Gemini** LLM after restarting their kernel, specifically mentioning that it worked fine before. It was suggested to update dependencies, particularly due to a recent **pydantic** upgrade that could lead to conflicts with lower versions.
- **Filtering Chat Message History for LLM Queries**: A member inquired about filtering out answers from message history before sending only questions to the chat engine. Another suggested that subclassing the memory and overriding the `get()` method could be a solution.
- **Fetching Node Text by ID in VectorStoreIndex**: A member asked how to obtain a node's embedded vector when they know the text and ID of a node in **VectorStoreIndex**. Suggestions included accessing the embedding data through the index's internal structure if embeddings were generated locally.
- **Local Equivalent of Tavily for RAG**: A user sought to find a local equivalent for the **Tavily** tool while following an example notebook for RAG workflows. It was clarified that Tavily is a web-search tool, and alternatives like Google or Bing would be necessary.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://llamahub.ai/l/readers/llama-index-readers-twitter?from=">no title found</a>: no description found</li><li><a href="https://llamahub.ai/l/readers/llama-index-readers-snscrape-twitter?from=">no title found</a>: no description found</li><li><a href="https://llamahub.ai/l/readers/]">no title found</a>: no description found</li><li><a href="https://github.com/run-llama/llamacloud-demo/blob/main/examples/advanced_rag/corrective_rag_workflow.ipynb">llamacloud-demo/examples/advanced_rag/corrective_rag_workflow.ipynb at main · run-llama/llamacloud-demo</a>: Contribute to run-llama/llamacloud-demo development by creating an account on GitHub.</li><li><a href="https://github.com/jina-ai/late-chunking/blob/main/chunked_pooling/chunking.py">late-chunking/chunked_pooling/chunking.py at main · jina-ai/late-chunking</a>: Code for explaining and evaluating late chunking (chunked pooling) - jina-ai/late-chunking</li><li><a href="https://github.com/jina-ai/late-chunking/tree/main/chunked_pooling">late-chunking/chunked_pooling at main · jina-ai/late-chunking</a>: Code for explaining and evaluating late chunking (chunked pooling) - jina-ai/late-chunking</li><li><a href="https://jina.ai/news/late-chunking-in-long-context-embedding-models/">Late Chunking in Long-Context Embedding Models</a>: Chunking long documents while preserving contextual information is challenging. We introduce the "Late Chunking" that leverages long-context embedding models to generate contextual chunk emb...</li></ul></div>

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1280461304083251201)** (14 messages🔥):

> - `H200 Pricing`
> - `H100 Demand Surge`
> - `Chat Template PR`
> - `GH200 Offer`
> - `KTO Performance`

- **H200 Price Stays High at 180k**: Currently, the **H200** is priced at **180k** for the **8** variant, as noted by a member.
  - This status raises questions about the high demand influencing pricing in the market.
- **Surge in H100 Prices Linked to Tesla**: A member reported a **huge increase** in the price of **H100** cards recently, suggesting a correlation with **Tesla's** activities.
  - The community anticipates how sustained demand from companies like Tesla will impact future market trends.
- **Chat Template PR Aids Setup**: A member highlighted the importance of the **chat template PR**, indicating it allows for loading the tokenizer’s template automatically.
  - Another member expressed that this functionality would simplify the setup process significantly.
- **GH200 Being Offered at 45k**: A member offered a deal to acquire the **GH200** for **45k**, prompting discussions about current pricing.
  - Interestingly, another member showed a preference for cards over deals, highlighting the ongoing demand for specific hardware.
- **KTO Performance Questions**: A member inquired about the performance of **KTO** with systems and multi-turn setups.
  - There seems to be a keen interest in understanding how KTO operates under these conditions, prompting community responses.

 

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/)** (1 messages):

caseus_: Create an issue for this enhancement pls

---

### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1280389179724927008)** (22 messages🔥):

> - `Cross Entropy Loss in SFTT`
> - `Fine-tuning Axolotl on Multi-User Dialogues`
> - `Custom Templates for Multi-User Interaction`

- **Cross Entropy Loss in SFTT Explained**: A user inquired about whether SFTT computes **cross entropy loss**, and another user directed them to check the modeling code for **LLaMA** on GitHub for verification.
  - This discussion emphasizes the importance of pinpointing the correct codebase as a reference for loss computation.
- **Exploring Multi-User Dialogue for Fine-Tuning**: A member expressed curiosity about fine-tuning a model using **dialogue from multiple people** without an agent, raising questions on how to format such data.
  - *They considered whether a model could be trained to understand conversation flow by using chat history as prompts.*
- **Custom Chat Templates Suggested**: Another user suggested customizing a **chat template** for multi-user simulations, rather than relying on traditional user-agent interactions.
  - *This approach highlights the potential for more tailored datasets, as current methods seem limited in handling multi-user scenarios.*
- **Challenges with Podcast Transcriptions**: One user noted difficulties in finding existing methods for handling **podcast transcriptions** or chats involving more than two participants.
  - *This reflects a broader sentiment that multi-user datasets are not heavily discussed in current AI training methodologies.*
- **Interest in Multi-User Datasets**: Several members expressed interest in developing **multi-user datasets**, recognizing their potential for enhancing conversational models.
  - *They acknowledged that while existing solutions are limited, exploring these datasets could yield valuable insights.*

 

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1280342359208624138)** (12 messages🔥):

> - `Tools in Playground`
> - `LLM for Report Generation`
> - `Model Card Accuracy`

- **Tools Enabled for New Model in Playground**: A member expressed eagerness to try tools for the new model in the playground, prompting a confirmation that **tools are now enabled**.
  - *Happy building!* was the response from a team member, encouraging further exploration.
- **Exploring LLM for Reports**: A query was raised about using LLMs to generate reports based on previous writing styles and meeting notes, aimed at assisting the Internal Audit team.
  - Members were soliciting experiences or insights related to leveraging LLMs for these purposes.
- **Incorrect Model Card Information**: A member pointed out that the [model card](https://huggingface.co/CohereForAI/c4ai-command-r-08-2024) incorrectly states a size of **35B** instead of the actual **32B**.
  - The team acknowledged the oversight and assured that it would be updated.

 

---

### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1280414887813582929)** (23 messages🔥):

> - `Server Side Events`
> - `Feature Request Submission`
> - `RAG JSON Output`
> - `Documentation Updates`

- **Cohere supports Server Side Events!**: It's confirmed that by sending an `Accept: text/event-stream` header to the chat API, users will receive **SSE events**.
  - *Billy* is updating the documentation to reflect this feature, which was previously undocumented.
- **Cohere Feature Request Process**: Frank inquired about submitting a feature request regarding the support for server side events.
  - Sssandra acknowledged the feedback and mentioned she would consult with the product team for further action.
- **RAG's JSON Output Limitation**: A member pointed out that **RAG** currently does not support JSON output through `response_format`, which might not be obvious.
  - Sssandra responded to this feedback by informing that the issue has been communicated to the team for consideration in future documentation.

 

**Link mentioned**: [Using server-sent events - Web APIs | MDN](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events): Developing a web application that uses server-sent events is straightforward. You'll need a bit of code on the server to stream events to the front-end, but the client side code works almost iden...

 

---

### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1280682299746422916)** (1 messages):

> - `Command-R-Plus 08-2024 Issues`
> - `Web-Search Connector Behavior`

- **Command-R-Plus 08-2024 exhibits instability**: The transition from **command-r-plus** (June version) to **command-r-plus-08-2024** resulted in erratic behavior like running at a **very high temperature**, leading to outputs filled with irrelevant content.
  - *This issue occurs only with the web-search connector enabled,* causing the app to hit max tokens quickly and disrupting the intended functionality.
- **Web-Search Connector exacerbates output issues**: The user noticed that the **web-search connector** is critical to the strange output behavior when utilizing the 08-2024 version.
  - In contrast, the June version functions reliably for fact-checking and online research purposes without these problems.

 

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1280303802242830438)** (12 messages🔥):

> - `Asistente Conversacional MultiAgente`
> - `Hybrid Retriever Implementation`
> - `Hugging Face Embedding`
> - `Normalization of Embeddings`
> - `Encode_kwargs Parameter`

- **Seeking guidance for Multi-Agent Conversational Assistant**: A member requested help in orchestrating a **Multi-Agent Conversational Assistant**, showing interest in advanced architectural approaches.
  - They inquired about experience concerning the **Supervisor architecture** and its complexities.
- **Hybrid Retriever Concept**: A user mentioned the possibility of creating a **hybrid retriever** that utilizes **two or more retrievers** in conjunction for better performance.
  - Another member expressed enthusiasm, simply responding with 'Cool'.
- **Passing Encode_kwargs in Hugging Face Embeddings**: A member discussed using a **Hugging Face embedding endpoint** and sought advice on how to pass **encode_kwargs** like normalization.
  - They provided a sample code snippet to illustrate their implementation attempt.
- **Normalization of Embeddings in TEI**: After a suggestion, a member confirmed that the **TEI** automatically normalizes embeddings, clarifying that they didn't need to specify **encode_kwargs**.
  - They noted that their check for embedding normalization returned **true**, confirming that the embeddings were already normalized.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="http://localhost:8080" ,"="">no title found</a>: no description found</li><li><a href="http://localhost:8080" )```"="">no title found</a>: no description found</li></ul></div>

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1280514382836535297)** (2 messages):

> - `Claude Sonnet 3.5 integration`
> - `Toolio 0.5.0 release`
> - `LLM structured response generation`
> - `Document chat application`
> - `OpenAI-like API`

- **Chatting with Documents using Claude Sonnet 3.5**: A developer introduced a tool that allows users to **chat with documents**, utilizing **Claude Sonnet 3.5** for seamless interactions, including file creation and editing capabilities.
  - They noted that the tool currently processes only **text files** and has limitations that can be optimized with a `.repoaiignore` file.
- **Toolio 0.5.0 Launches with Enhanced Features**: The **Toolio 0.5.0** release, dubbed 'The triumph of text,' brings improved documentation and better prompt construction for the Python toolkit designed for **Apple Silicon**.
  - Notable updates include structured LLM response generation that conforms to a [JSON schema](https://json-schema.org/) and support for easier tool integration.
- **Structured Control for Large Language Models**: Toolio aims to overcome the challenges posed by Large Language Models by offering developers **fine-grained control** over text generation with structured outputs.
  - It's positioned as a critical tool for developers needing more than casual text generation, with a focus on reliable tool calling functionalities.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://repoai.dev">RepoAI</a>: no description found</li><li><a href="https://OoriData.github.io/Toolio/">Toolio—Structured outputs, schema-controlled responses and tool-calling for LLMs on Mac</a>: no description found</li><li><a href="https://github.com/OoriData/Toolio/releases/tag/v0.5.0">Release 0.5.0 - Triumph of text (docs, better prompting, etc.) · OoriData/Toolio</a>: Added llm_helper.debug_model_manager—a way to extract raw prompt &amp; schema/tool-call info for debugging of underlying LLM behavior docs beyond the README (doc folder) test cases demo/algebra_tutor...</li><li><a href="https://pypi.org/project/Toolio/">Toolio</a>: OpenAI-like HTTP server API implementation which supports structured LLM response generation (e.g. make it conform to a JSON schema)</li></ul></div>

---

### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1280430357052784710)** (1 messages):

> - `Generative AI projects`
> - `Chatbot development`

- **Generative AI Projects to Check Out**: A member re-shared their **Generative AI projects** from this year, highlighting their work on GitHub with a [LinkedIn post](https://www.linkedin.com/posts/isham-rashik-5a547711b_github-di37dubai-tech-talk-intelligent-chatbot-activity-7236606074173222912-Fp2U) urging others to explore these projects.
  - They humorously asked for support by encouraging members to star their projects.
- **Push for Project Engagement**: In the community, there's an emphasis on **engaging** with shared projects, as members express interest in providing feedback and support.
  - This interaction not only fosters collaboration but also boosts visibility for innovators within the space.

 

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1280358622244638813)** (13 messages🔥):

> - `Python PATH issues`
> - `Open Interpreter installation struggles`
> - `Upcoming House Party event`

- **Python PATH Causes Confusion**: A member was having trouble getting their Python script for **Open Interpreter** to recognize the module after multiple installations using `pip install open-interpreter` in their virtual environment.
  
- **House Party Event Announcement**: An exciting **House Party** event was announced, promising big news and demos that could be the most impactful yet.
  - The event will be **livestreamed** and recorded, but attendees are encouraged to come to avoid missing out on the experience.

 

---

### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1280578355992989849)** (2 messages):

> - `Tool Use`
> - `Guest Appearance`

- **Weekly Shill for Tool Use**: This week's episode of **Tool Use** features a guest, highlighting their insights and discussions. You can check out the episode [here](https://www.youtube.com/watch?v=UUUWt8GaS64).
  - *Thanks to the community for support*—the share of experiences continues to invigorate discussions around tool usage.
- **Excited Chat with Guest**: Members expressed happiness about chatting with a new guest during the Tool Use session. Contributions and interactions with guests enrich the ongoing dialogue.
  - *A member shared their joy* in the conversation, creating an inclusive environment for shared learning.

 

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1280306446902493215)** (1 messages):

> - `Data Impact on Outcomes`
> - `Specific Dataset Inquiry`

- **Same Row Data Influences Outcome**: A member confirmed that all data points from the same row will affect the **final outcome** if they come from the same **sample**.
  - They inquired whether there was a specific **dataset** being analyzed, indicating interest in further details.
- **Request for Dataset Specifics**: The member asked if there was a **specific dataset** that others were looking at, suggesting a collaborative inquiry into data issues.
  - This inquiry highlights the importance of understanding how various datasets can interact within analytical contexts.

 

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1280494869751533639)** (6 messages):

> - `LoRA Fine-tuning Checkpoint Dictionary`
> - `Llama 405B PR Changes`
> - `Max Sequence Length Refactor`

- **Confusion over LoRA Checkpoint Dictionary**: A member raised a concern about constructing the checkpoint dict with the full merged adapter weights even when `adapter_weights_only` is set, questioning its necessity.
  - Another member clarified that this step was *removed entirely* in the Llama 405B PR, but it hasn't been updated in all recipes.
- **Support for Adapter Weights Only**: A member supported the idea that they should have flexibility in supporting `adapter_weights_only` as an option in general.
  - This suggests a consensus that improving options for fine-tuning configuration could enhance usability.
- **Looks like Max Sequence Length has Potential Solutions**: A member expressed excitement about the recent generation update and mentioned potential solutions for the `max_seq_len` issues.
  - They indicated confidence in finding workable solutions, suggesting a collaborative approach moving forward.
- **Draft Refactor for max_seq_len Discussed**: A draft refactor of the `max_seq_len` implementation was shared, indicating ongoing developments on GitHub.
  - The member committed to tidying up the documentation on the pull request after further discussions scheduled for tomorrow.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://github.com/pytorch/torchtune/pull/1449.">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://github.com/pytorch/torchtune/blob/70440446a4acf53e05cf7d74988fab21c8fd32e3/recipes/lora_finetune_single_device.py#L548).">torchtune/recipes/lora_finetune_single_device.py at 70440446a4acf53e05cf7d74988fab21c8fd32e3 · pytorch/torchtune</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.</li></ul></div>

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1280437614536691753)** (3 messages):

> - `Leaderboard Updates`
> - `New Hermes Model`
> - `Model Requests`

- **Apology for Missing Model in Leaderboard**: The team acknowledged an oversight in missing a **model** during the mass re-generation of results and promised to add it back in the next leaderboard update.
  - This update emphasizes their commitment to accurate representation of models on the leaderboard.
- **Focus Shift to New Dataset for Hermes Model**: Currently, attention is on a new **dataset release**, which has delayed processing requests for new models until later this week or early next week.
  - Listeners are encouraged to submit PRs for models they want included on the leaderboard meanwhile.
- **Appreciation for Clarifications**: One member expressed gratitude for the explanations provided regarding the recent updates and model management.
  - This reflects a positive community engagement and responsiveness to queries.

 

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1280435941617434634)** (4 messages):

> - `Chat Mode vs FC Mode`
> - `Leaderboard Differences`
> - `Issue Raising on GitHub`

- **Chat Mode complicates decoding**: Models have both **chat mode** and **FC mode**, with FC mode outputting in a structured way that eases decoding, while chat mode makes it challenging as it produces plain messages.
  - The **DEFAULT_SYSTEM_PROMPT** is implemented in chat mode to guide responses in a structured format, aiding in decoding.
- **Leaderboard variations clarified**: `leaderboard_live.html` specifically considers the **BFCL V2-Live dataset**, unlike the main `leaderboard.html` which incorporates all **BFCL V2 datasets**, both Live and non-Live.
  - This distinction is crucial for accurately interpreting leaderboard results and how datasets are evaluated.
- **Issue raised on GitHub**: A member confirmed they opened an issue regarding the leaderboard discrepancy on GitHub, providing a [link to the issue](https://github.com/ShishirPatil/gorilla/issues/620).
  - They also offered to submit a PR if aligned with the problems outlined, showing a proactive approach to collaborative problem-solving.

 

**Link mentioned**: [Issues · ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla/issues/620).): Gorilla: Training and Evaluating LLMs for Function Calls (Tool Calls) - Issues · ShishirPatil/gorilla

 

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1280429307453505599)** (5 messages):

> - `Mini-Omni Voice Model`
> - `100k H100 Clusters Analysis`

- **Mini-Omni voice model goes open source**: The [Mini-Omni](https://hf.co/gpt-omni/mini-omni), an open-source real-time audio conversational model, can generate text and audio simultaneously with streaming audio output.
  - The model was shared on Twitter, with links to its [codebase](https://github.com/gpt-omni/mini-omni) and research paper detailing its capabilities.
- **Insightful analysis on 100k H100 clusters**: A detailed explanation on the **100,000 H100 clusters** covers aspects like power, network topology, and the trade-offs of Ethernet vs InfiniBand.
  - The discussion highlights the perceived stagnation in AI capabilities since **GPT-4** due to a lack of significant compute increases for single models, despite other models having similar FLOP metrics.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://www.semianalysis.com/p/100000-h100-clusters-power-network?triedRedirect=true">100k H100 Clusters: Power, Network Topology, Ethernet vs InfiniBand, Reliability, Failures, Checkpointing</a>: Frontier Model Scaling Challenges and Requirements, Fault Recovery through Memory Reconstruction, Rack Layouts</li><li><a href="https://x.com/osanseviero/status/1830875530209513587?s=46">Tweet from Omar Sanseviero (@osanseviero)</a>: Mini-Omni, an open-source real-time audio conversational model ⚡️Real-time conversational speech-to-speech 🤯Can generate text and audio at the same time 🚀Streaming audio output Model: https://hf.c...</li><li><a href="https://www.latent.space/p/fb3dd9ec-ec10-4155-876f-4cf0c9faf67a?postPreview=paid&amp;updated=2024-09-03T07%3A28%3A24.287Z&amp;audience=everyone&amp;free_preview=false&amp;freemail=true">Latent Space</a>: The AI Engineer newsletter + Top 10 US Tech podcast. Exploring AI UX, Agents, Devtools, Infra, Open Source Models. See https://latent.space/about for highlights from Chris Lattner, Andrej Karpathy, Ge...</li></ul></div>

---

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 messages):

swyxio: new pod! [https://x.com/latentspacepod/status/1831020483967701260](https://x.com/latentspacepod/status/1831020483967701260)

---

### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1280357758667063339)** (3 messages):

> - `WeaviateRM Integration`
> - `text2vec-ollama Discussion`

- **Exploration of WeaviateRM Integration**: A member expressed interest in taking a closer look at the **WeaviateRM integration** and requested a forum issue to be opened about **text2vec-ollama**.
  - They shared a link to the [Weaviate forum](https://forum.weaviate.io/latest) for further discussion.
- **Acknowledgment of Collaboration**: Another member confirmed their willingness to assist by agreeing to open the forum issue.
  - The conversation was concluded with expressions of gratitude.

 

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1280292053560983552)** (1 messages):

> - `COPRO usage`
> - `Zero-shot instruction optimization`

- **Exploring COPRO for Length Management**: A member inquired about using **COPRO** or similar models to optimize instruction length effectively.
  - They suggested checking if adjusting **max_tokens** or implementing a metric return system could help manage instruction lengths.
- **Zero-shot Instruction Optimizer Techniques**: Discussion centered around using a zero-shot instruction optimizer to guide instruction lengths within models.
  - Members debated whether setting length constraints would involve simply limiting **max_tokens** or creating more complex metrics for instruction and input length.

 

---

### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1280382689991725147)** (2 messages):

> - `LLM Report Generation`
> - `Meeting Notes as Input`
> - `Synthetic Meeting Data`
> - `Text-to-Speech for Meeting Summaries`
> - `Speaker-Diarization Training`

- **Exploring LLM for Report Generation**: A member inquired if anyone has experimented with using **LLM** to generate reports based on previous writing styles and meeting notes from various stakeholders.
  - This approach aims to assist the Internal Audit team with report creation.
- **Clarification on Meeting Notes**: Another member sought clarification on the definition of meeting notes, suggesting it might refer to complete transcripts including attendees' names.
  - *What exactly do you mean by meeting notes?* prompted a discussion about different interpretations.
- **Synthetic Meeting Generation Insights**: A user discussed their work with the [persona-hub](https://github.com/tencent-ailab/persona-hub) to create synthetic meeting topics and simulate conversations.
  - They shared that generating these simulations involves significant token use but provides a diverse set of meetings for training purposes.
- **Audio and Summarization Techniques**: The conversation included plans to generate audio for each meeting attendee using a Text-to-Speech model and summarize meetings with an LLM.
  - It also touched on training a **whisper model** for speaker-diarization and developing a specific Text-to-Speech model related to those meetings.

 

**Link mentioned**: [GitHub - tencent-ailab/persona-hub: Official repo for the paper "Scaling Synthetic Data Creation with 1,000,000,000 Personas"](https://github.com/tencent-ailab/persona-hub): Official repo for the paper "Scaling Synthetic Data Creation with 1,000,000,000 Personas" - tencent-ailab/persona-hub

 

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/)** (1 messages):

th.blitz: Hello <a:LofiGirlWaveAnimated:927957453847556136>

---

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