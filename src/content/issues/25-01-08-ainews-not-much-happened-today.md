---
id: c953464a-5eac-4b6e-a1f6-ab1852dd08b1
title: not much happened today
date: '2025-01-09T03:45:48.690704Z'
original_slug: ainews-not-much-happened-today-1150
description: >-
  Sebastien Bubeck introduced REINFORCE++, a training method that speeds up
  reinforcement learning by 30% using PPO-inspired techniques, while AI21 Labs
  released the Phi-4 model under the MIT License, now available via Ollama.
  Additionally, François Chollet announced plans for ARC-AGI-2 and a
  next-generation AGI benchmark to better evaluate artificial general
  intelligence.
tags:
  - reinforce++
  - ppo
  - ai21
  - phi-4
  - ollama
  - francois-chollet
  - agi-benchmark
---


<!-- buttondown-editor-mode: plaintext -->**a quiet before the storm.**

> AI News for 1/7/2025-1/8/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **32** Discords (**218** channels, and **2346** messages) for you. Estimated reading time saved (at 200wpm): **278 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Traditionally, the industry wakes up on [the Ides of the month](https://buttondown.com/ainews/archive/ainews-multi-modal-multi-aspect-multi-form-factor/). We have a week to go.

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

**AI Research & Models**

- **Model Advancements and Releases**: [@SebastienBubeck](https://twitter.com/SebastienBubeck/status/1877010768689815696) introduced **REINFORCE++**, enhancing classical REINFORCE with **PPO-inspired techniques** for **30% faster training**. Additionally, [@AI21Labs](https://twitter.com/AI21Labs/status/1876931451804864689) announced the release of **Phi-4** under the **MIT License**, now accessible via [Ollama](https://twitter.com/ollama/status/1877058716069147092).

- **AGI Benchmarks and Foundations**: [@fchollet](https://twitter.com/fchollet/status/1877070012508180886) shared plans to release **ARC-AGI-2** and develop a **next-generation AGI benchmark**, moving beyond the 2019 ARC-AGI format to better evaluate **Artificial General Intelligence**.

**AI Development Tools & Frameworks**

- **Framework Enhancements and New Tools**: [@LangChainAI](https://twitter.com/LangChainAI/status/1877086957437632990) announced **10 new integration packages** for **LangChain**, facilitating enhanced **LLM application development**. Moreover, [@tom_doerr](https://twitter.com/tom_doerr/status/1877054737046000025) introduced **Ollama-OCR**, a **Python package** leveraging **Ollama's vision language models** for efficient **text extraction** from images.

- **Optimization Libraries**: [@arohan_](https://twitter.com/_arohan_/status/1877002595602235470) discussed optimizing **Shampoo** for **memory efficiency** in deep learning, reducing memory usage from **20 bytes per parameter** to **6 bytes** through innovative techniques.

**AI Applications & Use Cases**

- **AI in Software Development**: [@bindureddy](https://twitter.com/bindureddy/status/1876829984079380553) showcased **CodeLLM's v1 feature**, enabling **frontend code generation** from mocks, with future plans to integrate **backend context**. [@llama_index](https://twitter.com/llama_index/status/1877044767047168503) highlighted **LlamaIndex Workflows**, demonstrating **LLM-powered processes** for tasks like **academic paper summarization** and **PowerPoint slide generation**.

- **Property Management AI**: [@hwchase17](https://twitter.com/hwchase17/status/1877103276635848846) promoted collaboration with **@togethercompute** to enhance **WebDev Arena** with **complex coding agents** for superior **LLM coding evaluations**, aiming to assess **real-world coding capabilities**.

**AI Business & Industry**

- **Startup Growth and Investments**: [@bindureddy](https://twitter.com/bindureddy/status/1876829984079380553) detailed **CodeLLM's expansion**, driven by **customer feedback** and **sponsorships**. [@arohan_](https://twitter.com/_arohan_/status/1877003814282113033) emphasized the importance of **owning the tech stack** to **manage rapid changes** and recommended **distributed Shampoo** for **model layer optimizations**.

- **Compute Cost Reductions**: [@JonathanRoss321](https://twitter.com/JonathanRoss321/status/1877021439599243522) outlined **Groq's mission** to reduce **compute costs by 1000x**, anticipating a **100x spend increase** in **generative AI** due to **Jevons Paradox**.

**AI Policy & Ethics**

- **Ethical AI Deployment**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1877087202863190313) issued a **scam alert** regarding **malicious actors** falsely claiming associations with **AI21**, emphasizing the need for **vigilance** and **legal measures** against such scams.

- **AGI Concerns**: [@vikhyatk](https://twitter.com/vikhyatk/status/1877155735114563996) voiced concerns about the **lack of discourse on the dark side of AGI**, highlighting the necessity for discussions on **ethical implications** and potential **trade-offs** in **AI solutions**.

**Memes/Humor**

- **Humorous AI Insights**: [@mickeyxfriedman](https://twitter.com/mickeyxfriedman/status/1877045280757133372) shared a **creative prompt** for generating a **vivid winter scene** using AI, while [@teortaxesTex](https://twitter.com/teortaxesTex/status/1876851289864245386) humorously critiqued **LLM behaviors**, comparing **model philosophies** to human personalities.

- **Tech and AI Humor**: [@nearcyan](https://twitter.com/nearcyan/status/1877181322008002715) and [@qtnx_](https://twitter.com/qtnx_/status/1877096370797969851) posted **sarcastic remarks** and **jokes** about **AI models**, **compiler optimizations**, and **tech industry trends**, adding a lighthearted touch to the technical discourse.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. HP's Innovative AMD AI Machine with Unified RAM**

- **[HP announced a AMD based Generative AI machine with 128 GB Unified RAM (96GB VRAM) ahead of Nvidia Digits - We just missed it](https://aecmag.com/workstations/hp-amd-ryzen-ai-max-pro-hp-zbook-ultra-g1a-hp-z2-mini-g1a/)** ([Score: 423, Comments: 137](https://reddit.com/r/LocalLLaMA/comments/1hwe9mf/hp_announced_a_amd_based_generative_ai_machine/)): **HP** announced an **AMD-based Generative AI machine** with **128 GB Unified RAM**, where **96 GB can be allocated as VRAM**, allowing it to efficiently run **70B models q8**. The post speculates on whether this machine will utilize **RocM** or rely on **CPU inferencing** and anticipates that **Nvidia Digits** will likely use **CUDA** and **TensorRT** for inference optimization.
  - Discussions highlight the **limitations of ARM architecture** for AI workloads, emphasizing challenges with software compatibility and performance. The **x86 architecture** remains favored due to its broader support for AI frameworks and better performance with **NVIDIA GPUs**, despite ARM's potential in power efficiency and edge devices.
  - There is a detailed analysis of **memory types** and their performance implications, explaining the differences between **DDR (RAM)** and **GDDR (VRAM)**. The unified memory architecture offers benefits in shared access but can lead to bandwidth competition between processing units, impacting performance, especially in AI applications.
  - **RocM** is discussed as a viable alternative to **CUDA** for AMD-based systems, with users noting improvements and compatibility with various models. However, the performance may still lag behind CUDA, although it is seen as a cost-effective solution for certain applications.


**Theme 2. Phi-4 by Microsoft: Released and Analyzed**

- **[Phi-4 has been released](https://huggingface.co/microsoft/phi-4)** ([Score: 376, Comments: 108](https://reddit.com/r/LocalLLaMA/comments/1hwmy39/phi4_has_been_released/)): The post announces the release of **Phi-4**, a new model, but provides no additional details or evaluations in the body.
  - **Phi-4 Model Release and Performance**: **Phi-4**, released on **Hugging Face** after its initial availability on **Azure AI Foundry**, is noted for its impressive reasoning capability, outperforming other models like **Qwen2.5** in specific benchmarks despite its smaller size of **14B parameters**. Users praise its logical task performance but criticize its creative writing and factual tasks, with some noting its low **SimpleQA** score due to reduced hallucinations.
  - **Technical Benchmarks and Comparisons**: The model shows strong performance in benchmarks such as **MMLU** and **GPQA**, sometimes even surpassing larger models like **Llama 3.3 70B**. It excels in reasoning and logical tasks but falls short in code generation compared to **Qwen2.5**, with some users expressing doubts about the real-world applicability of these benchmarks.
  - **Licensing and Community Feedback**: The model's release under the **MIT license** is highlighted as significant, contrasting with previous releases under restrictive licenses. Community feedback is mixed, with some users skeptical of the benchmarks, while others appreciate the potential for small models to act as "smart tools" rather than comprehensive knowledge bases.


- **Phi 4 MIT licensed  - its show time folks** ([Score: 56, Comments: 4](https://reddit.com/r/LocalLLaMA/comments/1hwn90v/phi_4_mit_licensed_its_show_time_folks/)): **Microsoft** has released **Phi 4**, an **MIT licensed** model, now available on **Hugging Face**. This marks a significant move in open-source AI, providing broader access to advanced machine learning models.
  - **Phi 4's Coding Capabilities** are highlighted, with users noting its potential usefulness in synthetic textbook generation. However, it struggles with following instructions, which appears to be an intentional design choice.
  - There is curiosity about the model's performance in **coding and Retrieval-Augmented Generation (RAG)** scenarios, indicating interest in practical applications beyond standard benchmarks.


**Theme 3. DeepSeek V3 GGUF: 2-bit Quantization Success**

- **DeepSeek V3 GGUF 2-bit surprisingly works! + BF16, other quants** ([Score: 196, Comments: 104](https://reddit.com/r/LocalLLaMA/comments/1hw1nze/deepseek_v3_gguf_2bit_surprisingly_works_bf16/)): DeepSeek V3 has been released with **2 to 8-bit quantizations** and a **bf16 de-quantized version** available on [Hugging Face](https://huggingface.co/unsloth/DeepSeek-V3-bf16). The **2-bit version** requires a minimum of **48GB RAM** and **250GB disk space**, and detailed instructions for running the model using **K quantization** are provided, with specific examples such as using the **Q5_0 K quantized cache**.
  - **DeepSeek V3 Performance and Requirements**: **DeepSeek V3** is a **671B parameter mixture of experts model** that rivals state-of-the-art models like **GPT-4** and **Claude**. It requires significant resources, with a minimum of **48GB RAM** and **250GB disk space** for the **2-bit version**, and users have reported varied performance metrics, such as **2.57 tokens per second** using a **32-core CPU** with **192GB RAM**.
  - **Quantization Techniques and Challenges**: The model employs **2 to 8-bit quantizations** to optimize performance, with discussions on further reducing this to **1.08 bits** or even **0.6-bit quant** for extreme memory savings. Users have experimented with different quantization methods like **Q2_K** and **Q5_0 K**, noting that **2-bit quantization** can still maintain usability, though there are concerns about performance drops and the need for calibration.
  - **Hardware and Offloading Strategies**: Users have explored different hardware configurations, including **RTX 4090** and **AMD EPYC** processors, to run DeepSeek V3 efficiently. Discussions highlight the importance of **VRAM** and **CPU offloading**, with suggestions for using **NVME swap space** and **per layer GPU offloading** to manage memory constraints and improve token generation rates.


- **I Tested Aider vs Cline using DeepSeek 3: Codebase >20k LOC...** ([Score: 62, Comments: 44](https://reddit.com/r/LocalLLaMA/comments/1hwf5lv/i_tested_aider_vs_cline_using_deepseek_3_codebase/)): The post compares **Aider** and **Cline** in handling codebases larger than 10k LOC, with the author favoring **Aider** due to its flexibility, portability, and economic token usage. While **Qwen 2.5 Coder 32B** lags behind **DeepSeek 3** for medium-large codebases, **Claude 3.5 Sonnet** outperforms **DeepSeek 3** in larger codebases, suggesting a shift towards more complex organizational uses. [Test video](https://youtu.be/e1oDWeYvPbY) is provided for further insights.
  - **Aider** is favored for its tight **Git integration** and cost-effectiveness, with users noting that it's reliable and suitable for daily use. **DeepSeek 3** is preferred for day-to-day tasks, while **Cursor** is seen as less reliable but still valuable at $20/month. **Windsurf** has been criticized for losing focus, leading some users to cancel their subscriptions.
  - Concerns were raised about **Aider's use of ChatGPT/Claude subscriptions**, with clarification that Aider's `--copy-paste` mode involves manual steps to comply with terms of service. This mode requires users to manually copy and paste between Aider and LLM web chats, avoiding automated interactions prohibited by most LLM TOS.
  - **Qwen 2.5 Coder 32B** is noted to be less effective than **DeepSeek 3** for medium-large codebases, with a stark parameter size difference of **32B vs 671B**. Despite this, users find value in exploring both models to understand their strengths, and there's interest in comparing other open models like **Mistral Large** and **Llama 3.3**.


**Theme 4. NVIDIA Cosmos: Foundation Model for Virtual Worlds**

- **[NVIDIA Open Model License: NVIDIA Cosmos is a world foundation model trained on 20 million hours of video to build virtual worlds and generate photo-real, physically-based synthetic data for scientific and industrial testing.](https://v.redd.it/lfzohbxndqbe1)** ([Score: 121, Comments: 14](https://reddit.com/r/LocalLLaMA/comments/1hwfblu/nvidia_open_model_license_nvidia_cosmos_is_a/)): **NVIDIA** has introduced the **Cosmos model** under the **Open Model License**, designed to create virtual worlds and generate photo-realistic, physically-based synthetic data. The model is trained on **20 million hours of video** and aims to support scientific and industrial testing, as detailed on their [website](https://www.nvidia.com/en-in/ai/cosmos/).
  - **NVIDIA's Open Model License** allows for commercial use and the creation and distribution of derivative models without claiming ownership of outputs, as highlighted in the [Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). This permissive approach is intended to facilitate the development of AI technologies.
  - Some users express skepticism about NVIDIA's models being **state-of-the-art (SOTA)** for long, suggesting NVIDIA's ultimate goal is to sell GPUs rather than maintain leading-edge models.
  - There is curiosity about the implications of the license if **guardrails** are disabled, indicating concerns about the flexibility and limitations of the license terms.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**Theme 1. 25% of Google's Code Generated by AI**

- **[Google CEO says over 25% of new Google code is generated by AI](https://arstechnica.com/ai/2024/10/google-ceo-says-over-25-of-new-google-code-is-generated-by-ai/)** ([Score: 523, Comments: 89](https://reddit.com/r/OpenAI/comments/1hw610a/google_ceo_says_over_25_of_new_google_code_is/)): **Google CEO** reveals that **AI** is responsible for generating over **25%** of new code at Google. This highlights the increasing reliance on AI tools for software development within the company.
  - **AI's Role in Code Generation**: There is skepticism about the claim that **25% of Google's code is AI-generated**, with discussions on whether this includes autocompletion, function generation, or other forms of automated code. **Pichai** mentioned that these are suggestions accepted 25% of the time, indicating a more nuanced role of AI in code generation.
  - **Industry Impact and Skepticism**: The discussion highlights a disparity in AI usage across companies, with some engineers noting a significant shift towards AI in software development, while others remain skeptical about the exact figures and impact on the workforce. Concerns about job roles for junior engineers and the definition of "generated code" are prominent.
  - **Perception and Reality**: There is a mix of humor and criticism regarding the announcement, with some users mocking the claim as "old news" or suggesting it reflects poorly on Google's product quality. The conversation also touches on the evolving nature of AI tools and their integration into software engineering practices.


**Theme 2. Elon Musk's AI Launch Promises**

- **I just remembered that Elon Musk said that last december he would release an AI better than ChatGPT** ([Score: 221, Comments: 94](https://reddit.com/r/OpenAI/comments/1hwm5ol/i_just_remembered_that_elon_musk_said_that_last/)): **Elon Musk** announced plans in December to release an AI superior to **ChatGPT**, but there has been no follow-up or delivery on this promise.
  - Users express skepticism about **Elon Musk's** promises, comparing **Grok** to **ChatGPT** as inferior or non-existent, with comments highlighting a pattern of unfulfilled commitments, such as **FSD** and Teslas making money autonomously, which have been anticipated for years without fruition.
  - A sarcastic tone dominates the conversation, with references to Musk's **"Tesla measurement converter"** predicting Grok updates to take much longer than stated, and criticism of Musk's management style, implying that high-IQ individuals may be reluctant to work for him.
  - Concerns are raised about Musk's environmental impact, with a link provided to his **XAI facility** allegedly polluting areas in South Memphis, underscoring dissatisfaction with his broader business practices beyond AI promises.


---

# AI Discord Recap

> A summary of Summaries of Summaries by o1-mini-2024-09-12

**Theme 1. New AI Models Surge Forward**

- [**Phi-4 Dominates Multiple Platforms**](https://huggingface.co/microsoft/phi-4): The **Phi-4** model is extensively discussed across Discords for its performance enhancements and fine-tuning capabilities. Users highlight its compatibility issues with Unsloth and explore its simple **SFT** and **DPO** pipeline, sparking debates on multi-GPU support and overfitting concerns.
- [**MiniMind: TinyLLaMA in 3 Hours**](https://github.com/jingyaogong/minimind): The **MiniMind** project introduces a lightweight **26.88M-parameter** model trained in just **3 hours**, offering a guide for building personal-scale LLMs. Its rapid training process and minimal size make it a favorite for quick iterations and educational purposes.
- [**GPT4All Faces Quantization Quagmires**](https://github.com/GPT4All-Community/phi-4-GGUF/blob/main/phi-4-Q4_0.gguf): **GPT4All** users report that **low-bit quantization** significantly degrades model performance, especially for models below **7B** parameters. Community members share [GGUF builds](https://huggingface.co/SamPurkis/Microsoft_Phi-4-Q4_0-GGUF/tree/main) to mitigate these issues and enhance accessibility.

**Theme 2. AI Tools and API Integrations Expand**

- [**Unsloth API & Local Training UI Launched**](https://github.com/Leoleojames1/unslothAPI): A new **local Unsloth API** and training web UI enable fine-tuning **LoRA** adapters and merging models seamlessly. Users appreciate the [GitHub repo](https://github.com/Leoleojames1/unslothAPI) for its comprehensive features and seek feedback on its usability.
- [**OpenRouter Bridges Twitter with AI**](https://github.com/lord-dubious/x-mcp): The **x-mcp** project connects Twitter with the **Model Context Protocol**, allowing advanced interactions between tweets and AI models. Developers explore its potential to enhance Twitter functionalities and integrate with other AI frameworks.
- [**DSPy Integrates Vertex AI Models**](https://linkexample.com): Engineers discuss adding **Vertex AI** models for inference in **DSPy**, aiming to expand the framework's capabilities. They also consider dedicated approaches for **function calls**, simplifying integrations and enhancing performance.

**Theme 3. Community Support and Technical Hurdles**

- [**Authentication Woes and Billing Baffles**](https://codeium.canny.io/feature-requests/p/autonomous-iterative-visual-inspection-of-generated-code-in-browser): Multiple Discords report **authentication issues** and **billing frustrations**, particularly with platforms like **Codeium**. Users struggle with **Google-only registration** and unexpected credit purchases, urging clearer policies and better support.
- [**Multi-GPU Support Remains Elusive**](https://github.com/unslothai/unsloth/issues/1518): **Unsloth** users express disappointment over the lack of **multi-GPU training** support, which is anticipated to be a future commercial feature. This limitation affects training workflows and sparks discussions on potential workarounds.
- [**Token Usage and Export Challenges**](https://aider.chat/docs/config/adv-model-settings.html): **Cohere** and **Aider** communities face difficulties in **exporting token usage**, with members seeking solutions to track and manage their **token budgets** effectively. Suggestions include logging token usage per request as a temporary workaround.

**Theme 4. GPU Optimizations and Hardware Discussions**

- [**Speculative Decoding Boosts Inference Speed**](https://github.com/ggerganov/llama.cpp/pull/8134#issuecomment-2550018120): Implementing **Speculative Decoding** in **llama.cpp** results in a **25% to 60% speed increase**. Developers plan to integrate this feature into [**Ollama**](https://github.com/ollama/ollama/pull/8134), enhancing LLM workflow efficiencies.
- [**Cutlass and bfloat16 Performance Dip**](https://github.com/NVlabs/tiny-cuda-nn): In **Cutlass** kernels, using **bfloat16** is observed to be about **10% slower** than half precision. Members suggest using **diff tools** like **meld** to compare **PTX** and **SASS** changes for performance insights.
- [**Thunderkittens vs Flash Attention Showdown**](https://github.com/HazyResearch/ThunderKittens/blob/main/assets/attn.png): Users compare **Thunderkittens** with **Flash Attention 3**, sharing [plot images](https://github.com/HazyResearch/ThunderKittens/tree/main/tests/python) to analyze performance. Collaboration is encouraged to replicate and enhance these comparisons through shared scripts.

**Theme 5. AI Applications in Creative and Technical Domains**

- [**Stable Diffusion's Commercial Clarity**](https://stability.ai/license): Members discuss **commercial usage** guidelines for **Stable Diffusion**, noting that revenue up to **$1 million** typically requires no additional license. They emphasize adherence to the [**Stability AI License**](https://stability.ai/license) and explore tools like [**CivitAI**](https://civitai.green/models/626819/beauty-in-evil-by-hailoknight) for training **LoRA** models with minimal data.
- [**NotebookLM Enhances Content Repurposing**](https://youtu.be/4QJm_AptHF4): Users leverage **NotebookLM** to transform long-form content like videos and podcasts into micro-content for social media. Techniques such as **inner monologue** and **freeze frame** are employed to deepen engagement and streamline content creation.
- [**Omdena Tackles Real-World AI Challenges**](https://www.omdena.com/projects): **Omdena** coordinates large-scale collaborative AI projects, enabling up to **50 contributors** to develop solutions for community-specific challenges. Their emphasis on **local solutions** fosters impactful and sustainable AI applications.


---

# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Phi-4 & Unsloth: Fine-Tune Frenzy**: The new **Phi-4** model sparked discussions on bug fixes and training synergy with Unsloth, referencing [Phi-4 on Hugging Face](https://huggingface.co/collections/unsloth/phi-4-all-versions-677eecf93784e61afe762afa) for merges and **GGUF** conversions.
   - Users warned that Hugging Face updates might disturb fine-tuning workflows, overshadowing simpler tasks like single GPU setups.
- **Local Unsloth API & Web UI Appear**: A user introduced a **local Unsloth API** and training web UI, highlighting [their GitHub repo](https://github.com/Leoleojames1/unslothAPI) for fine-tuning **LoRA** adapters and merging models.
   - They also shared a new dataset on Hugging Face, seeking feedback on usability and performance in daily training tasks.
- **DeepSeek V3: GUFF Downloads Spark Nostalgia**: The latest **DeepSeek V3** release included multiple **GUFF** files, with fans comparing the slow downloads to old-school **Napster** days.
   - Participants clarified that downloading all files, placed together, is required for [DeepSeek-V3-GGUF](https://huggingface.co/unsloth/DeepSeek-V3-GGUF) to function properly.
- **Loss Spikes & Overfitting Worries**: Periodic **loss spikes** during training stumped some members, who saw values nearly double every few steps, fueling confusion about normal expectations.
   - Others debated dataset redundancy and *overfitting*, insisting it must be extremely repeated data to noticeably degrade performance.
- **Multi-GPU Dreams & Job Triumph**: Questions arose about **multiple GPU** support in Unsloth, concluding that it's not currently available and might become a commercial feature.
   - Meanwhile, a user’s **job search** ended successfully, bringing excitement about new opportunities and upcoming professional exploits.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Codeium Chat Glitches & Llama Lament**: Users reported frequent connectivity issues in **Codeium Chat** with the Llama model, repeatedly encountering *“E0108... i/o timeout”* errors that hamper real-time code generation.
   - They pointed out that the platform’s unstable performance overshadowed newly purchased **credits**, fueling worries over Codeium’s reliability.
- **Windsurf Woes with Heavy Code**: When dealing with over **600 lines** of code, **Windsurf** often becomes unresponsive, prompting frustration and hardware-blame from users on older machines.
   - Members demanded a more robust approach to large file handling, urging code-size optimizations to sustain development flow.
- **Python Linter Mystery in Windsurf**: Some developers observed that Python linters like pylint and mypy produce no visible output within **Windsurf**, despite functioning in other editors.
   - They proposed deeper **integration fixes** so that critical error and style checks can run smoothly in-browser.
- **Authentication & Billing Bafflement**: Multiple users faced **authentication** obstacles that locked them out, coupled with **billing** frustrations over canceled plans and foggy credit purchases.
   - People cited hurried over-buying of credits and reliance on **Google-only registration** as key pain points demanding clearer policies.
- **Debates on AI Model Capabilities**: Some compared **Claude** and **Sonnet** against Windsurf’s performance, noting differences in speed and advanced inspection features.
   - They referenced the [Autonomous iterative visual inspection request](https://codeium.canny.io/feature-requests/p/autonomous-iterative-visual-inspection-of-generated-code-in-browser) to underscore the demand for in-browser enhancements rivaling other AI tools.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Phi-4 Performance Sparks Curiosity**: Enthusiasts tested the **Phi-4 model** on **LM Studio v0.3.6**, with some reporting improved loading and others facing crashes.
   - Participants suggested version updates as a workaround, viewing **Phi-4** as an intriguing yet complicated choice for local LLM runs.
- **Speculative Decoding Speeds Inference**: Implementing **Speculative Decoding** in [llama.cpp](https://github.com/ggerganov/llama.cpp) led to a **25% to 60% speed boost** in processing rates.
   - Developers noted plans to integrate it into [Ollama](https://github.com/ollama/ollama/pull/8134), fueling ongoing excitement for faster LLM workflows.
- **Deepseek-V3 Adoption Soars on llama.cpp**: Community members reported running **Deepseek-V3** on llama.cpp with ample RAM needs for stable performance.
   - They posted resource links, emphasizing **Deepseek-V3** as an option for tasks requiring higher VRAM capacity.
- **Nvidia Digits & GPU Showdown**: A fresh **Nvidia Digits** lineup with unified memory stirred speculation on how it stacks up against the **RTX 5090**.
   - Discussions focused on bandwidth and memory speed, with [Reddit threads](https://www.reddit.com/r/LocalLLaMA/comments/1gzm93o/speculative_decoding_just_landed_in_llamacpps/) offering more insights into real-world performance.
- **LPDDR5X vs M2 Ultra & Rumored AI Box**: The **LPDDR5X** memory at about 500 GBps was contrasted with the **M2 Ultra**, highlighting differences in training frameworks.
   - Enthusiasts eyed an Nvidia AI computer pegged at **$3,000** and **250 TFLOPS**, though real performance checks remain uncertain.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Lightning-Fast 5090 Rumors**: Members speculated about the **NVIDIA 5090**, highlighting a possible performance jump that might overshadow the **4090** and potentially cut generation times down to **13 seconds**.
   - They compared it to the **30 seconds** on a 4090 and seemed excited about the impact on large-scale **Stable Diffusion** workflows.
- **Commercial Clarity for Stable Diffusion**: Participants shared that **commercial usage** of **Stable Diffusion** up to **$1 million** in revenue generally requires no extra license, referencing the official [Stability AI License](https://stability.ai/license).
   - Speakers emphasized the importance of following the **community license agreement**, suggesting a review of the [Stability AI Core Models](https://stability-ai.squarespace.com/core-models) and [NVIDIA Open Models License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) for domain-specific rules.
- **LoRA Training with Minimal Data**: Enthusiasts explained that just **30 images** can yield strong **LoRA** results, especially when combined with quality prompts and tools like [CivitAI](https://civitai.green/models/626819/beauty-in-evil-by-hailoknight).
   - They recommended watching **video tutorials** to refine workflows and use advanced training scripts for better outputs.
- **Monstrous Art Gains Traction**: Creators explored specialized models like **THRILLustrious** to produce realistic monster designs, pointing to resources on [CivitAI](https://civitai.green/models/626819/beauty-in-evil-by-hailoknight).
   - They showcased *Beauty in Evil* as an example LoRA set to tweak stylistic elements for monstrous imagery.
- **Image-to-Image & Video Surprises**: Contributors discussed advanced **image-to-image** workflows, including masking and **solid color frames** to style avatars with minimal overhead.
   - They also highlighted [HunyuanVideo support in ComfyUI](https://blog.comfy.org/p/hunyuanvideo-native-support-in-comfyui) for expanded motion-based content creation.



---



## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **Bolt’s Prompting Power & UI Flare**: Members emphasized that with skillful instructions, **Bolt** produces stronger outcomes, highlighting *it’s all about how you phrase your ideas* to guide the AI for better responses.
   - Others shared admiration for the **UI** and stressed specifying **colors** and placement details in prompts to shape the final result effectively.
- **Quest for Documentation & Hidden Features**: A user asked if there was **documentation** to navigate Bolt’s abilities, expressing interest in structured instructions to harness the tool completely.
   - They also wanted insight into the **process** of discovering Bolt’s capabilities, hoping for more transparency around advanced usage tips.
- **Token Tangles & Rate Limit Struggles**: Participants faced confusion over daily and monthly **token** quotas, with some running into rate limiting when usage exceeded shared limits.
   - They proposed adding clearer **user settings** to reduce confusion and help developers avoid abrupt stoppages mid-development.
- **Building Bigger Apps & Wrestling Deployments**: Contributors stressed that breaking larger codebases into smaller components keeps projects maintainable and logical, recommending an overview file for context.
   - They also noted **deployment** trouble, often caused by build errors, urging developers to run terminal checks rather than relying solely on Bolt for fixes.
- **Supabase Snags & Multi-Tool Mix**: Users encountered recurring **Supabase** setup issues, including repeated .env reconfigurations after disconnects.
   - They also compared experiences using **Bolt** alongside Cursor or Copilot, suggesting that each tool performs best in its own area.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Sonnet Storms with O1 Pro**: In #general, members noted that combining **Sonnet** with **O1 Pro** leads to better prompt crafting for complex tasks, referencing several user tests.
   - One user insisted *"Sonnet is as good as O1 Pro"* for their needs, fueling speculation that synergy might elevate performance further.
- **Aider Advice & File Flubs**: Users in #questions-and-tips shared **Aider** tactics like reading all generated comments and refining /ask prompts for clarity, linking to [advanced model settings](https://aider.chat/docs/config/adv-model-settings.html).
   - They also encountered file update mishaps and message format discrepancies, attributing them to Python errors and a 'prompt' vs 'messages' mix-up.
- **DeepSeek Dilemmas**: Some users experienced **DeepSeek v3** freezing and theorized it might overload with high-volume requests or large contexts.
   - Others claimed zero slowdown, suggesting resource constraints or usage variance could be the main cause.
- **Litellm & Ollama Ordeals**: A user struggled with **Litellm** custom models and prefixing, consulting the [options reference](https://aider.chat/docs/config/options.html) for proper configuration.
   - Another overcame **Ollama** local model issues by specifying model paths correctly, referencing a related [GitHub issue](https://github.com/ollama/ollama/issues/2203).
- **SynthLang Snags & Gemini 2.0 Gains**: Participants tested the [SynthLang platform](https://synthlang.fly.dev/) but encountered repeated selection errors, prompting bug reports.
   - Meanwhile, those using **Gemini 2.0 Flash Experimental** appreciated its voice-mode brainstorming, hoping for optional markdown outputs soon.



---



## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **NVIDIA's Project DIGITS Surfaces in Conversation**: Attendees highlighted **NVIDIA Project DIGITS**, promoted as *the world’s smallest AI supercomputer*, with references to [NVIDIA's official page](https://www.nvidia.com/en-us/project-digits/). They noted its reservation process and teased potential for on-device **LLM experimentation**.
   - No specific release date or performance metrics were shared, but participants viewed it as a compelling hardware option to handle **heavy AI workloads**.
- **No Additional Major AI Developments Found**: **Cursor IDE** bug reports included repeated linting errors, the Apply feature failing to manage code updates, and confusion from multiple trial accounts, with a [forum thread on stuck Composer sessions](https://forum.cursor.com/t/composer-stuck-at-generating-specific-composer-instance-not-global-issue/35479/4) also highlighting these issues. Participants noted **Flutter** dependency challenges as well, particularly with TensorFlow and Keras integrations.
   - They also stressed **smaller code files** to avoid technical debt and help new team members ramp up quickly. No new models, datasets, or next-gen tools emerged from these discussions.



---



## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **No-Fuss System Prompts & Language Tweaks**: Members explored code in the URL to force English replies, refined system prompts for **NotebookLM** to quote sources accurately, and stressed the impact of precise instructions for better responses.
   - They shared ideas about language parameter configuration, agreeing that **exact wording** significantly shapes **NotebookLM** output.
- **Repurposing Videos for Quick Social Posts**: A user shared a [YouTube tutorial on repurposing content](https://www.youtube.com/watch?v=4QJm_AptHF4), highlighting **NotebookLM**'s ability to transform long video material into micro-content, prioritizing speed for writers.
   - Another member suggested the same approach for podcast archives, calling it a fresh vantage point for older recordings.
- **AI Redlining & NotebookLM Plus Perks**: A proposal emerged to use **digital labor** for contract redlining and lighten paralegal tasks, alongside tips to enable [NotebookLM Plus](https://support.google.com/notebooklm/answer/15678219?hl=en#:~:text=Where%20to%20get%20NotebookLM%20Plus%C2%A0) under business units for extra features.
   - They provided a requirements list for user access, noting that a smooth setup fosters quick adoption among legal teams.
- **Podcast Scripts & Vanishing Quotes**: Creators struggled with inconsistent host monologues, plus **NotebookLM** only pulled quotes from the first 13 pages of a 250-page resource.
   - They requested better script control, flagged audiobook narration tone challenges, and joked about video imports failing without transcripts.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **x-mcp connects Twitter to AI**: A new [GitHub project called x-mcp](https://github.com/lord-dubious/x-mcp) aims to give users **full control** of bridging Twitter with the Model Context Protocol, providing advanced interactions with tweets and AI.
   - Developers see potential in **x-mcp** to expand Twitter functionality, referencing the repository's synergy with other AI frameworks in active discussions.
- **Agents Base automates marketing at scale**: The newly launched [Agents Base](https://www.producthunt.com/posts/agents-base) offers **50-500x better CPM** than standard ad platforms, as claimed in its Product Hunt listing.
   - It deploys **swarms of cloud marketing agents** to handle A/B testing across demographics and formats, sparking excitement about streamlined ad campaigns.
- **Community debates LLM game dev feasibility**: Participants noted **3D FPS** titles remain difficult due to a shortage of advanced world models, though simpler concepts are possible with iterative feedback and debugging.
   - Enthusiasts suggested carefully structured prompts and step-by-step user hints to push LLMs beyond typical pitfalls and produce workable prototypes.
- **Questions on using Azure GPT-4o with OpenRouter**: Some asked how to integrate a hosted GPT-4o on Azure with **OpenRouter**, pointing to [Azure's model listings](https://openrouter.ai/provider/azure) for more details.
   - They weighed differences between Azure-based GPT-4o and the official versions, specifically around feature stability for enterprise use.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo's Mind-Bending Moves in Static Indexing**: Several members discovered that **ListLiteral** cannot be indexed by a runtime variable in Mojo, and they recommended using **InlineArray** instead for dynamic needs, referencing [multiple issues in the `modularml/mojo` repo](https://github.com/modularml/mojo/issues/3403). They highlighted that after re-testing, **InlineArray** performed well for all indexing scenarios involving runtime data.
   - Confusion arose when a user claimed **InlineArray** initially failed, but they admitted their code was likely at fault. Others endorsed **InlineArray** as a more reliable approach than **ListLiteral**, noting its future potential for performance gains.
- **Trait Teases & Tantalizing Tinkering in Mojo**: Community members pushed for better **trait** capabilities like default functions, conditional traits, and parametric traits, hoping to mirror Rust’s flexibility in future releases. They cited [open issues in `modularml/mojo`](https://github.com/modularml/mojo/issues/3252) as grounds for broader **trait** improvements.
   - Discussions focused on how a refined trait system could reduce repetitive code and enforce stronger type checks. Enthusiasts want a more unified approach that ties traits effectively with static analysis and potential overload mechanics.
- **Overload Odyssey & Polymorphism Progress**: A user proposed **OOP-style overloads** and **polymorphic functions** in Mojo, suggesting a ranked approach to handle overlapping signatures. They noted that automatic type narrowing is vital for consistent **overload** selection, referencing [recent ideas in the `modularml/mojo` repo](https://github.com/modularml/mojo/issues/3630).
   - Some worried that mixing **TraitVariant** with complex overload rules could breed ambiguity, prompting calls for an ironclad syntax and better code organization. They argued that well-defined **where** clauses and careful resolution logic are essential for large codebases.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Quantization Quagmire Fells Model Performance**: Members highlighted how **low-bit quantization** can degrade performance, referencing [Low-Bit Quantization Favors Undertrained LLMs](https://arxiv.org/abs/2411.17691), especially in coding tasks.
   - They observed that once models drop below **7B** parameters, quantization inflicts a notably larger dip in accuracy.
- **GPU Glitches Gall Some Q4_0 Fans**: Several participants ran into **Q4_0** models crashing on GPU, yet [llama.cpp PR #10817](https://github.com/ggerganov/llama.cpp/pull/10817) suggested partial fixes.
   - They cited **CUDA** constraints and concluded that stable GPU acceleration can hinge on specific hardware setups.
- **Agent Development Hiring Hype**: A user announced spots for junior engineers working on **agent development**, offering payment after successful PR merges, plus a call for **UX designers** on Figma or AdobeXD.
   - They specifically sought US-based talent focused on practical tasks that integrate with GPT4All.
- **Q4_0 Model Mayhem Continues**: Community members noted multiple **Q4_0** models causing random crashes in GPT4All, but one user posted a [Q4_0 GGUF model](https://huggingface.co/GPT4All-Community/phi-4-GGUF/blob/main/phi-4-Q4_0.gguf) that worked better.
   - They speculated on a potential Q8_0 alternative but found no concrete evidence of progress.
- **Hugging Face Handoff for Models**: Contributors shared **GGUF** builds on Hugging Face, such as [SamPurkis/Microsoft_Phi-4-Q4_0-GGUF](https://huggingface.co/SamPurkis/Microsoft_Phi-4-Q4_0-GGUF/tree/main).
   - They confirmed some models hold **MIT** licenses, ensuring broader accessibility for the GPT4All community.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Phi-4's Surprising Simplicities**: The newly released [**Phi-4** model by Microsoft](https://huggingface.co/microsoft/phi-4) uses a straightforward pipeline of **SFT** and **DPO**, delivering advanced math and reasoning results.
   - Members noted the approach's simplicity and suggested open-source teams could match these strong outcomes with effective synthetic datasets.
- **MiniMind's 3-Hour Marathon**: The [**MiniMind** project](https://github.com/jingyaogong/minimind) offers a 26.88M-parameter language model fully trained in roughly **3 hours**, featuring complete code for data prep, supervised pretraining, instruction fine-tuning, and LoRA.
   - It's about 1/7000 the size of GPT-3, which allows fast iteration and serves as a guide for constructing personal-scale LLMs.
- **Networking on a Dime**: Participants explored **budget-friendly** HPC networking using 10GbE, USB-C, and older Mellanox cards to speed data transfers and manage costs.
   - They highlighted **USB**'s capability to mimic Ethernet, adding a do-it-yourself angle to cheaper lab deployments.
- **Placeholder Data for Zero-Trust MVPs**: Contributors debated the necessity of **zero trust** frameworks at project outset, proposing **placeholder data** in the cloud for early builds.
   - They emphasized that an **MVP** can skip final security requirements, enabling quick iteration without jeopardizing sensitive data.
- **Neural Embeddings' Hidden Layers**: A recent [**blog post**](https://seanpedersen.github.io/posts/structure-of-neural-latent-space) discussed the **manifold hypothesis**, suggesting high-dimensional data might reside within lower-dimensional spaces.
   - It also examined **hierarchical** feature organization and the **linear** representation across layers, prompting deeper analysis of embedding internals.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Pythia’s Ethical Enigma**: Members were looking for **Pythia** evaluations on the [Ethics Dataset](https://huggingface.co/datasets/hendrycks/ethics), but no results were shared, fueling curiosity about fine-tuning or direct testing.
   - A user championed a direct approach for learning AI by cloning [nanoGPT](https://github.com/karpathy/nanoGPT), highlighting that hands-on coding can surpass standard tutorials.
- **SFT Showdown with AdamW Insight**: Several recommended [AllenAI's open-instruct](https://github.com/allenai/open-instruct) and **GPT-NeoX** for both **SFT** and **RLHF**, with **NVIDIA NeMo** also considered for robust integration.
   - Clarification emerged that **AdamW** is simply the 'adam' optimizer plus weight decay, offering a more streamlined route to consistent regularization.
- **Cut Cross-Entropy Slices Memory Usage**: The [CCE paper](https://arxiv.org/abs/2411.09009v1) introduced computing logits only for the correct token, drastically reducing memory overhead in training large vocabulary models.
   - Parallel discussions touched on a **6.7B model** hitting **OOM** even with a batch size of 1, alongside a mysterious speed boost when **DeepSpeed pipe** was set to 0, hinting at hidden interplay with memory demands.
- **HunyuanProver Claims Theorem Win**: [HunyuanProver](https://arxiv.org/abs/2412.20735), built upon **Hunyuan 7B**, achieved a **68.4% pass** rate on miniF2F-test for **theorem proving** with LEAN4.
   - It also solved some IMO statements and will open-source a dataset of **30k** synthetic problems, signaling a leap forward for automated proof research.
- **SD3 Forward-or-Backward Crossfire**: A debate arose on whether the **SD3 paper** meant a forward process or if it was actually referencing a backward step, linked to the zero SNR discussion.
   - A possible oversight in the text has lingered for months, leaving the community curious about the paper’s intended meaning.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **01.AI’s Billionaire Buildup**: The Chinese AI startup **01.AI** locked in a **$1 billion** valuation within eight months, flatly refuting rumors of a team sale to Alibaba as *completely false*.
   - CEO Kai-Fu Lee noted their revenue surpassed **RMB 100 million** in 2024 and predicted bigger gains in 2025, according to [TechNode](https://technode.com/2025/01/07/01-ai-refutes-rumors-of-selling-teams-to-alibaba/).
- **Harvard’s Data Initiative Gains Momentum**: The **Institutional Data Initiative** at Harvard refines crucial datasets in collaboration with various knowledge institutions, promising open releases in early 2025.
   - They are hiring researchers for data stewardship roles, as mentioned on [their official site](https://institutionaldatainitiative.org/#get-involved).
- **Omdena Attacks Real-World AI**: **Omdena** coordinates collaborative AI projects featuring up to 50 contributors, focusing on local solutions for community-specific challenges.
   - They encourage global participation and highlight new challenges at [Omdena’s Project Page](https://www.omdena.com/projects).
- **Hugging Face’s Phi-4 Rolls Out**: A link from Sebastien Bubeck spotlighted the [**Phi-4** model](https://huggingface.co/microsoft/phi-4), capturing attention for its approach to AI tasks.
   - The post urged exploration of **Hugging Face** tools, underscoring an ongoing push for broader community involvement.
- **MoE Efficiency Sees Spotlight**: Participants challenged whether **MoE** models can keep experts fully loaded or must load/unload them per token to achieve optimal throughput.
   - References to **OlMoE** and **vLLM** surfaced, with some cautioning about increased VRAM demands and for-loop complexities in transformers.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **LLaMA Learns from Locals**: One user showed personal data fine-tuning on **LLaMA**, describing it as 'pretty easy' and sparking enthusiasm for custom model training approaches. They discussed incorporating structured personal texts, prompting questions about best practices.
   - Others weighed the practicality of expanded instructions and setups for **LLaMA**, hinting at broader community interest in refining user-driven fine-tuning strategies.
- **GPU 4o Mini Takes on Ubuntu 24.04.1**: A user running **Ubuntu 24.04.1** with a **6900XT** asked for setup guides on **GPU 4o Mini**, mentioning [Ollama 3.2 Vision](https://example.com/ollama-reference) and **ROCm 6.3.1** readiness. Early feedback highlighted improved inference speeds when configured correctly.
   - Community members pointed to potential pitfalls in installation and runtime, underscoring the importance of **GPU compatibility** for local model usage.
- **O1 Pro Upgrade Under the Microscope**: Debate surfaced about whether **O1 Pro** justifies the cost for heavier workloads, with some praising its benefits for intricate tasks. Others advised a usage-based assessment before committing resources to the upgrade.
   - They emphasized matching **O1 Pro** capabilities with the complexity of planned operations, advising caution to avoid unnecessary spending.
- **Prompt Style & the 80% Completion Conundrum**: Members noted that simply naming a **style** in the prompt rarely guarantees desired formatting, reporting an **80% completion** rate that they deemed suboptimal. Suggestions included tighter instructions and reduced ‘noise’ to improve success rates.
   - Some argued for more explicit guidelines and example-driven prompts, reinforcing the notion that clarity directly impacts output consistency.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **CSV Craze for Data Wrangling**: Perplexity introduced a **CSV download** capability for table responses, letting users quickly save and process data offline, with an example [image demonstration](https://discord.com/channels/1047197230748151888/1047204950763122820/1326655467577147412) posted to guide usage.
   - Community members welcomed the feature for **AI-driven data workflows**, praising the straightforward integration of a CSV button in the result interface.
- **Youzu AI Interiors Merge Style with Shopping**: A [Medium post](https://medium.com/design-bootcamp/youzu-ai-where-ai-interior-design-meets-real-world-shopping-76a066be3688) introduced **Youzu AI**—an AI interior design platform that links design concepts to actual purchasable items.
   - Early adopters pointed out that **dynamic room refits** could transform how typical e-commerce merges with design intelligence, praising the synergy between style suggestions and product listings.
- **Office Suite Synergy with Perplexity Tools**: Some members speculated about integrating **Perplexity** into services like **MS 365 Copilot**, citing better AI-based content generation than competing applications.
   - They argued that synergy with enterprise ecosystems would turbocharge daily tasks, giving a more robust drafting environment for business documentation.
- **Discord OAuth2 Flow for Devs**: A technical guide on **Discord's OAuth2** flow circulated, illustrating safe app authentication practices for bridging user logins with external platforms.
   - Contributors noted that the straightforward steps let devs seamlessly embed advanced AI features into Discord bots, with minimal overhead.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **NCU Nudges & Warmup Wisdom**: Comparing an **NCU profile** of a 32×32 vs 16×16 configuration reveals subtle performance distinctions, while **wgmma** usage demands tile sizes of at least **64** to effectively enlist 4 warps.
   - **Warmup** debates also surfaced, with some championing **25ms** over a meager **1ms** to keep the GPU clock from idle dips.
- **Fused MLP & On-Chip Curiosities**: **Triton** fans asked about a fused MLP akin to [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn), exploring the limited adoption of on-chip MLP solutions.
   - Community discussion hinted at the small scale of on-chip MLP tasks, fueling questions about broader real-world usage.
- **Cutlass & Comparisons with bfloat16**: In **Cutlass** kernels, using **bfloat16** is about **10%** slower than half precision, sparking speculation on whether any internal mechanics cause that dip.
   - One user suggested **meld** or diff tools to examine PTX and SASS changes, ignoring register names for clarity.
- **Softmax Showdown & Discord Leaderboard**: Alpha testers were invited to a new [Discord-based leaderboard](https://linkexample.com) that tracks the fastest softmax kernel in a GPU competition.
   - Participants can simply craft small kernels without major bot coding, while the channel pinned a separate server link to coordinate efforts.
- **Thunderkittens Tussles with Flash Attention**: Users compared **Thunderkittens** to Flash Attention 3 using a shared [plot image](https://github.com/HazyResearch/ThunderKittens/blob/main/assets/attn.png), requesting scripts to replicate the data in their setups.
   - They linked the [tests/python folder](https://github.com/HazyResearch/ThunderKittens/tree/main/tests/python) and invited collaboration for MoE or Deep Seek kernels, forging a code-sharing synergy.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Tricky Token Tally**: A user asked about exporting **token usage** to a file, but their repeated searches in Cohere's docs yielded no official export feature.
   - Some members proposed logging **token usage** per request as the best workaround, though the bot's attempts to find a direct CSV or JSON export solution were unsuccessful.
- **Recursive Repeats Rile**: A member reported that the **Cohere LLM** occasionally loops recursively, quickly depleting their **token budget** and prompting suggestions for bounding the response length.
   - They cited their use of the **command-r-plus-08-2024** model, noting potential **Persian** support but warning others to set maximum token limits to avoid runaway costs.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Fierce FP4 Feud**: NVIDIA's comparisons between **FP4** and **FP8** have fueled a heated debate, with some claiming the data is questionable, as noted in [Yuchen Jin's post](https://x.com/yuchenj_uw/status/1876680855630094725). **Jensen's** pitch of FP4 as a training metric is attracting attention, especially given FP8's possible effect on model quality at inference time.
   - Some people said they *love Nvidia and Jensen* but criticize *vague terms like 'AI TOPS'* and the mismatch in specs, while the [phi-4 weights release hype](https://x.com/sytelus/status/1877015492126220594) overlaps the entire discussion.
- **TTS Trials and Tribulations**: Open source text-to-speech models are under scrutiny for a *slightly robotic* tone and choppy cadence. Multiple attempts suggest that improved cloning still requires better voice samples for fidelity.
   - A [Deepseek V3 collection on Hugging Face](https://huggingface.co/collections/unsloth/deepseek-v3-all-versions-677cf5cfd7df8b7815fc723c) was used for testing, but the emphasis and rhythm remain off-key.
- **Omi's Odd Wearable**: A wearable named **Omi** promises to capture brain data, expecting a separate module in 2025, as teased in [Nik Shevchenko's post](https://x.com/kodjima33/status/1877017546697699363). Some see parallels with *Black Mirror* ideas of microchips and mind control.
   - With ordering at [omi.me](http://omi.me), users wonder if this ushers in next-level personal tech for real-time neural monitoring.
- **Salesforce Slams Hiring Door**: **Marc Benioff** declared that Salesforce won't hire new software engineers in 2025, citing productivity boosts from their **Agentforce** AI product, as shown in [SalesforceBen's write-up](https://www.salesforceben.com/salesforce-will-hire-no-more-software-engineers-in-2025-says-marc-benioff/).
   - While overall headcount may rise, the organization's workforce strategy is shifting toward AI-based efficiency.
- **LLM Ventures Gain Momentum**: Members emphasized that large organizations struggle to embrace advanced **LLM** strategies swiftly, leaving agile startups to capture the spotlight. Existing products with *bolt-on* LLM features lag, while from-scratch approaches show dramatic success.
   - They highlighted [Takeoff](https://www.jointakeoff.com/) as a case in point, anticipating more **LLM-first** product releases soon.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Cohere Cozy with LlamaIndex**: Cohere refreshed their [documentation](https://t.co/dLKGgkqOe8) to integrate with **LlamaIndex**, requiring the Cohere SDK and a trial API key for immediate usage.
   - Contributors noted it offers a straightforward way to run **Cohere** models on private text sources, highlighting quick package installation and seamless queries.
- **LlamaIndex Workflows Wow With ArXiv**: Lingzhen Chen showed how to use [LlamaIndex Workflows](https://t.co/OXU8DcUb5E) to systematically search and summarize academic papers from ArXiv in a repeatable pipeline.
   - They presented it as a controlled, step-by-step approach for refining AI-powered interactions and producing consistent analysis of technical documents.
- **GitHub Gathers AI Gurus**: On January 15th, GitHub HQ will host expert talks on debugging AI agents, creating fast inference systems, and harnessing LlamaIndex-based workflows ([event link](https://t.co/GnYXYqJfth)).
   - Organizers anticipate energetic sessions on optimizing large language models, encouraging early sign-ups for hands-on demos and networking.
- **Metadata Maneuvers in LlamaIndex**: A user questioned why `document.excluded_embed_metadata_keys = ['key']` did not remove fields from node storage, prompting a reminder to remove them prior to indexing.
   - They concluded that selective metadata trimming streamlines indexes, and participants urged proactive audits to keep them minimal.
- **FaithfulnessEvaluator’s First-Run Friction**: After switching to a larger **bge_onnx** model, the **FaithfulnessEvaluator** took over **25 seconds** on its first run, then stabilized at around **1 second**.
   - Discussions suggested model initialization overhead, with users proposing a warm-up pass or preloading to cut the initial delay.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **No Crypto Ties at AI21 Labs**: Members emphasized that **AI21 Labs** has no affiliation with any crypto tokens or related discussions, warning about bans for persistent mentions.
   - They clarified that this server is dedicated to **developer support and generative AI models**, and not a forum for promoting crypto ventures.
- **Jamba Jams with Dev Productivity**: A user spotlighted **Jamba** for coding support, explaining how its conversational RAG improved their Python app workflow.
   - They noted **increased efficiency** when pairing Jamba’s API with existing solutions like deepSeek and openAi.
- **Laughing at AI’s Coding Quirks**: One newcomer praised AI’s ability to **generate code** yet chuckled at occasional goofs while debugging.
   - They tested AI solutions in **HTML, Javascript, and PHP**, confirming that coding capabilities are still maturing.
- **Podcast Transcripts Powered by Jamba**: A developer described using **Jamba** for handling podcast episode transcripts in a Python application.
   - They found **conversational input** beneficial for script management, citing it as a more enjoyable experience than manual editing.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Form Frenzy for MOOC Certificates**: Multiple participants thanked the staff for opening the [declaration form](https://link.to.form) to submit details for certificate eligibility.
   - They stressed the importance of official submission, highlighting the need to fully complete the form to secure final credentials.
- **Email Emphasis for Proper Credential Tracking**: Several members noted the **same email** address must be used on the form and assignments to ensure certificates link up correctly.
   - A few switched back to their original email to avoid confusion and preserve course records.
- **Spring 2025 Continues F24 Momentum**: The community confirmed the Spring 2025 course will begin in **late January**, building on the **F24** materials.
   - Participants expect it to be a direct follow-up, keeping the curriculum consistent for returning learners.
- **Twitter Tangle Over Verification**: One member’s Twitter account got suspended, so they provided a Medium post instead for certificate validation.
   - They asked for alternative methods to confirm completion, given the suspension prevented standard profile checks.
- **Certificates Remain Under Wraps**: No one has received certificates yet, as confirmed by the course staff.
   - The team hinted that issuance may be delayed until **end of January**, stirring eagerness among learners.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Hide Demo Fields Tames Prompt Bloat**: Members tested **'hide_demo_fields'** to replace certain blocks with '*... omitted for brevity ...*', reducing **prompt bloat** while preserving clarity in demos.
   - They proposed that a built-in solution in **DSPy** would unify handling of large contexts, rather than relying on patchwork measures.
- **Vertex AI Embraces DSPy**: Engineers explored adding **Vertex AI** models for inference in **DSPy**, highlighting potential expansions of the framework's usage.
   - They also discussed a dedicated approach for **function calls** with Vertex AI, aiming for simpler integrations.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter Tuning Trials**: A user requested tips for **Open Interpreter** production workflows, including model choice and performance tweaks, as they're not finding widely shared successful setups yet.
   - They're hoping to see **community-tested** configurations for smoother deployments and better performance.
- **Prompting Tactics for Crisper Code**: Enthusiasts asked for direct advice on **effective prompting** to produce accurate code generation, suggesting structured instructions and carefully chosen tokens.
   - They stressed the importance of **concise prompts** to keep the model on track for coding tasks.
- **Custom Instructions Boost Output**: Discussions centered on using **custom instructions** to sharpen model responses and expand domain-specific accuracy.
   - Participants emphasized that tailoring these settings could lead to **consistent results** during intensive workloads.
- **NVIDIA Reveals Grace Blackwell**: [NVIDIA](https://www.nvidia.com/en-us/project-digits/) highlighted a compact AI machine delivering a petaflop of performance, enabling large-scale model training on a single box.
   - They claim users can handle up to **200B-parameter** models locally, with a helpful software stack included.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Double 3090s Strike a Note with LLM Fine-tuning**: A member with a **dual 3090 setup** expressed interest in **fine-tuning an LLM** for music notation, seeking help from the community.
   - They described their strong computational capacity for training, highlighting readiness to tackle heavier tasks and *inviting collaboration*.
- **Open Agent Tools: In Search of a Registry**: A participant in the research channel asked if there's a good open **tool registry** for building AI agents, signaling a need for structured resources.
   - No specific solution surfaced, and the question remains open for further insights from those with relevant repositories.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **ModernBERT gets a mention**: In [#general](https://discord.com/channels/1216353675241590815/1216353675744641096/) a user inquired about experiences finetuning **ModernBERT**, but no benchmarks or references were shared.
   - They asked for any known tips or performance tweaks, though no responses were available to confirm specific results.
- **No other broad discussions**: Beyond the single ModernBERT query, no further releases or advanced techniques were posted.
   - Community members did not engage with additional updates, leaving the discussion limited to that one question.



---


The **tinygrad (George Hotz) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Axolotl AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **HuggingFace Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1326282628164354049)** (407 messages🔥🔥🔥): 

> `Finetuning Phi-4, Unsloth API, CUDA on TPUs, Deepseek V3, Training Distinct LLMs` 


- **Fine-tuning and Compatibility of Phi-4**: Users discussed the current limitations and compatibility of the new Phi-4 model with Unsloth, particularly addressing bugs and operational issues encountered during training.
   - It was noted that Unsloth is undergoing updates to align with Hugging Face's changes, which may affect fine-tuning capabilities.
- **Local API and Training Web UI for Unsloth**: A user shared a new local Unsloth API and web UI for training models, emphasizing its capabilities to train LoRA adapters, merge them, and convert to GGUF.
   - They invited feedback on their project, along with the corresponding fine-tuned dataset hosted on Hugging Face.
- **CUDA Support on TPUs**: A discussion emerged about Google's implementation of CUDA on TPUs, with clarifications that there is no direct CUDA support, but some compatibility exists through PyTorch to JAX conversion.
   - This raised questions about the investment required to enable CUDA functionalities within the TPU architecture.
- **Deepseek V3 Fine-tuning Challenges**: Users inquired about the feasibility of fine-tuning Deepseek V3, with some expressing concerns about the model's size and current limitations for fine-tuning.
   - It was indicated that fine-tuning would typically require substantial GPU resources, suggesting multi-GPU setups may be necessary.
- **Navigating Loss Metrics During Training**: A user described observations regarding fluctuating loss metrics during model training, questioning whether substantial losses during training should prompt early stopping.
   - It was confirmed that such fluctuations are normal and do not necessitate premature termination of the training process.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: Below is a list of all our notebooks:</li><li><a href="https://x.com/UnslothAI/status/1876729710790815872">Tweet from Unsloth AI (@UnslothAI)</a>: Deepseek V3, including GGUF + bf16 versions are now on @HuggingFace!Min. requirements to run: 48GB RAM + 250GB of disk space for 2-bit.Includes 2, 3, 4, 5, 6 and 8-bit quantized versions.See all versi...</li><li><a href="https://huggingface.co/collections/unsloth/phi-4-all-versions-677eecf93784e61afe762afa">Phi-4 (All Versions) - a unsloth Collection</a>: no description found</li><li><a href="https://huggingface.co/microsoft/phi-4">microsoft/phi-4 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/unsloth/phi-4-GGUF">unsloth/phi-4-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/datasets/loading">Load</a>: no description found</li><li><a href="https://huggingface.co/unsloth/DeepSeek-V3-GGUF">unsloth/DeepSeek-V3-GGUF · Hugging Face</a>: no description found</li><li><a href="https://github.com/KaihuaTang/Qwen-Tokenizer-Pruner">GitHub - KaihuaTang/Qwen-Tokenizer-Pruner: Due to the huge vocaburary size (151,936) of Qwen models, the Embedding and LM Head weights are excessively heavy. Therefore, this project provides a Tokenizer vocabulary shearing solution for Qwen and Qwen-VL.</a>: Due to the huge vocaburary size (151,936) of Qwen models, the Embedding and LM Head weights are excessively heavy. Therefore, this project provides a Tokenizer vocabulary shearing solution for Qwen...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1gwyuyg/beware_of_broken_tokenizers_learned_of_this_while/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/tree/main#-installation-instructions">GitHub - unslothai/unsloth: Finetune Llama 3.3, Mistral, Phi, Qwen 2.5 &amp; Gemma LLMs 2-5x faster with 70% less memory</a>: Finetune Llama 3.3, Mistral, Phi, Qwen 2.5 &amp; Gemma LLMs 2-5x faster with 70% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/pull/1520">Update __init__.py by sebaxakerhtc · Pull Request #1520 · unslothai/unsloth</a>: This PR is solving the issue with some GPUs</li><li><a href="https://x.com/Dan50412374/">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter</li><li><a href="https://github.com/unslothai/unsloth/issues/1518">[BUG] Unsloth stopped working after todays commits · Issue #1518 · unslothai/unsloth</a>: Hi. I can&#39;t use Unsloth anymore on my RTX3090. It works only on Nvidia T4 on colab. When I try to download any model - I have this: ----------------------------------------------------------------...</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: Finetune Llama 3.3, Mistral, Phi, Qwen 2.5 &amp; Gemma LLMs 2-5x faster with 70% less memory</a>: Finetune Llama 3.3, Mistral, Phi, Qwen 2.5 &amp; Gemma LLMs 2-5x faster with 70% less memory - unslothai/unsloth</li><li><a href="https://docs.unsloth.ai/basics/saving-and-using-models/saving-to-gguf>">Unsloth Documentation</a>: no description found</li><li><a href="https://github.com/Leoleojames1/unslothAPI">GitHub - Leoleojames1/unslothAPI: local api for unsloth</a>: local api for unsloth. Contribute to Leoleojames1/unslothAPI development by creating an account on GitHub.</li><li><a href="https://huggingface.co/Borcherding/OARC_Commander_v002_alpha">Borcherding/OARC_Commander_v002_alpha · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/Borcherding/OARC_Commander_v001">Borcherding/OARC_Commander_v001 · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1326655511847763978)** (2 messages): 

> `Job Search` 


- **Job Search Successfully Completed**: A member announced that their job search is complete and they are now employed.
   - This marks a transition for them from job hunting to starting new professional endeavors.
- **Excitement for New Opportunities**: The member expressed their enthusiasm about securing a job and moving forward in their career path.
   - They highlighted the anticipation of new challenges and experiences ahead.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1326282783730962513)** (30 messages🔥): 

> `Unsloth multi-GPU support, Training loss iteration spikes, DeepSeek GUFF file concerns, Avoiding overfitting in datasets, RAG and fine-tuning discussions` 


- **Unsloth lacks multi-GPU training support**: Members expressed curiosity about whether Unsloth supports multi-GPU training, with indications that **it currently does not** and will be a commercial solution once supported.
   - *Edgarmartinez4430* noted that it feels like a single GPU setup based on their findings from **Reddit**.
- **Training loss spikes cause confusion**: A user shared experiencing spikes in **training loss** every 4 steps at roughly double the expected value, raising alarms about the training process.
   - This led to inquiries about the normalcy of such behavior, but no immediate solutions were provided.
- **DeepSeek GUFF files need all for functionality**: Concerns arose regarding the **DeepSeek release** having multiple GUFF files for the Q2_K_XS version, probing whether all need to be downloaded.
   - Members confirmed that **all GUFFs must be in the same folder** for proper operation, leading to feelings of nostalgia about slow downloads reminiscent of **Napster** days.
- **Clarifying dataset size and overfitting**: There was debate about the relationship between dataset size and **overfitting**, with conflicting opinions on the significance of data quality.
   - Some argued that *redundant data can cause overfitting*, while others emphasized that it must be excessively redundant for it to impact performance significantly.
- **RAG vs. fine-tuning confusion**: Discussion highlighted the differences between RAG (Retrieval-Augmented Generation) and fine-tuning methods, showing a need for clarity around the purposes of both.
   - *Fjefo* emphasized that they are entirely distinct processes and encouraged further **research** by others involved.


  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1326320761375428668)** (66 messages🔥🔥): 

> `Codeium Chat Issues, Windsurf Performance, Authentication Problems, Billing and Credits, Google Signup Only` 


- **Codeium Chat experiences frequent errors**: Users reported various issues with **Codeium Chat**, mentioning errors like inability to connect and service interruptions, especially regarding the **LLAMA** model.
   - *“E0108... i/o timeout”* errors were commonly noted, indicating connectivity problems within the service.
- **Windsurf struggles with large code files**: Members discussed **Windsurf** lagging when working with large code files, stating it becomes unresponsive with over **600 lines** of code.
   - One user humorously noted that their **8-year-old PC** contributes to the performance degradation.
- **Authentication issues plague users**: Several users expressed frustration with **authentication problems** on **codeium.com**, which prevented them from logging in and using **Windsurf**.
   - One user highlighted their recent purchase of credits as a reason for their urgency in addressing these login issues.
- **Confusion over billing, credits, and rebates**: The community debated inconsistencies in **billing** and **credits**, with several users questioning the billing system after inadvertently purchasing more than necessary.
   - One member humorously remarked on the financial repercussions of over-purchasing credits, describing their experience while navigating **Codeium's** credit system.
- **Concerns over Google-only registration**: Users expressed dissatisfaction with the **Google-only registration** requirement for **Codeium**, questioning the rationale behind such a limitation.
   - A user lamented this restriction, indicating that it was a recent change that hindered their ability to create an account without a Google account.


  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1326289704886210561)** (300 messages🔥🔥): 

> `Windsurf Performance Issues, User Support and Feedback, Integration with Python Linters, Account and Billing Problems, AI Model Capabilities` 


- **Windsurf experiences significant performance issues**: Numerous users reported issues with Windsurf, including lag, high RAM usage, and internal errors, leading to frustration with the platform's stability.
   - Users noted that the situation has persisted for weeks, impacting their ability to effectively use the tool for development.
- **Challenges with customer support and account issues**: Users are expressing frustration over unresolved account and billing discrepancies, with some reports indicating canceled plans and unrecognized transactions.
   - There are calls for better communication and support from the Codeium team regarding these issues.
- **Problems using Python linters in Windsurf**: A user reported that no output from Python linters like pylint and mypy was appearing in Windsurf, despite them working fine in VSCode and Cursor.
   - This raised concerns about the integration and functionality of these tools within the Windsurf environment.
- **Discussions on AI model capabilities**: There were discussions regarding the capabilities of AI models like Claude and Sonnet, with users comparing them to Windsurf’s performance.
   - Users suggested improvements, such as the autonomous visual inspection capabilities in competing tools, and proposed features for Windsurf.
- **User experiences with login issues**: Several users faced problems logging into the Windsurf platform, citing issues like browser redirection failures and token submission problems.
   - This led to discussions about potential fixes and the need for better support resources to resolve authentication failures.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/homer-simpson-hide-in-shrubs-hiding-in-bushes-disappear-into-hedges-embarrassed-gif-24183287">Homer Simpson Hide In Shrubs GIF - Homer Simpson Hide In Shrubs Hiding In Bushes - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/multiversx-x-xportal-egld-crypto-gif-4249062898891695021">Multiversx Xportal GIF - Multiversx X Xportal - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://codeium.canny.io/feature-requests/p/autonomous-iterative-visual-inspection-of-generated-code-in-browser">Autonomous iterative visual inspection of generated code in browser | Feature Requests | Codeium</a>: When developing web application, in some use case developer already have UI prototype of the page it want to build.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1326313494714519593)** (76 messages🔥🔥): 

> `Performance of Phi-4 model, Issues with LM Studio model loading, Deepseek-V3 compatibility, Qwen2 model functionality, LM Studio as a server and frontend connection` 


- **Phi-4 model performance discussion**: Users report mixed experiences with the new **Phi-4 model**, with some successfully loading it while others experience crashes or performance issues.
   - Updating LM Studio to version **0.3.6** is often necessary for optimal compatibility with this model.
- **Troubleshooting LM Studio model loading**: Several users are facing errors when loading models in LM Studio, prompting discussions on version updates and compatibility.
   - One user specifically noted that the Mistral 7B model does not support system prompts, suggesting a workaround by embedding directives in user messages.
- **Deepseek-V3 running on llama.cpp**: Discussions emerged regarding the successful implementation of **Deepseek-V3** on llama.cpp, with users sharing links to resources and discussions.
   - Hardware requirements for running Deepseek-V3 were highlighted, emphasizing the need for substantial RAM.
- **Qwen2 model capabilities**: The **Qwen2 model** was discussed in terms of its picture description features, with some users reporting issues resulting in gibberish outputs.
   - Support for the model is dependent on specific versions and naming conventions, indicating limited image processing for standard Qwen2 models.
- **Utilizing LM Studio as a server and frontend**: Users inquired about using LM Studio set up on a Mac Mini as a frontend from other devices, leading to suggestions to employ alternative client applications.
   - Alternatives like OpenWebUI and AnythingLLM were mentioned for better integration with local or remote endpoints.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/download">Download LM Studio - Mac, Linux, Windows</a>: Discover, download, and run local LLMs</li><li><a href="https://www.reddit.com/r/LocalLLa">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/microsoft/phi-4">microsoft/phi-4 · Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1hw1nze/deepseek_v3_gguf_2bit_surprisingly_works_bf16/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/OpenInterpreter/open-interpreter">GitHub - OpenInterpreter/open-interpreter: A natural language interface for computers</a>: A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1326279195156025496)** (113 messages🔥🔥): 

> `Speculative Decoding, Nvidia Digits Performance, 7900XT Comparison, LPDDR5X vs M2 Ultra, Recent Nvidia GPU Releases` 


- **Speculative Decoding Enhances Inference Speed**: Speculative Decoding implementation in llama.cpp shows a **25% to 60% increase in speed**, allowing for potentially faster LLM inference without sacrificing accuracy.
   - Discussions indicate this feature will soon be integrated into other models like [Ollama](https://github.com/ollama/ollama/pull/8134), adding to its appeal.
- **Comparing Nvidia Digits to Current GPUs**: The performance of the new Nvidia Digits architecture remains unclear, but Users noted it features **unified memory**, distinguishing it from the 5000 series.
   - Questions about the memory speed and potential bandwidth of the Digits systems are ongoing, with comparisons sought against models like the RTX 5090.
- **7900XT vs 4090 Performance Query**: A user inquired about the performance of the **7900XT** relative to other GPUs like the **4090**, **4080**, and **3090** in terms of TOPS.
   - Responses suggested looking into resources comparing these models, including a [Reddit link](https://www.reddit.com/r/LocalLLaMA/comments/1gzm93o/speculative_decoding_just_landed_in_llamacpps/) for further insights.
- **LPDDR5X Bandwidth vs M2 Ultra**: The **LPDDR5X** memory in new GPUs is estimated at **500 GBps**, reportedly less than the **M2 Ultra's** memory bandwidth.
   - However, some believe the availability of more frameworks for training makes these new GPUs appealing despite lower specs in certain areas.
- **Excitement Around New Nvidia Releases**: The upcoming Nvidia AI computer priced at **$3,000** is generating interest, with anticipated performance being evaluated against older models.
   - An inference estimate of **250 TFLOPS** was mentioned, although its practical reception is yet to be seen, especially regarding chaining possibilities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.gigabyte.com/Graphics-Card/GV-N5090AORUSX-WB-32GD/sp#sp">AORUS GeForce RTX™ 5090 XTREME WATERFORCE WB 32G Specification | Graphics Card - GIGABYTE Global</a>: no description found</li><li><a href="https://medium.com/ai-science/speculative-decoding-make-llm-inference-faster-c004501af120">Speculative Decoding — Make LLM Inference Faster</a>: Improve LLM inference speed by 2–3X without degrading any accuracy</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1gzm93o/speculative_decoding_just_landed_in_llamacpps/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1hqlug2/revisting_llamacpp_speculative_decoding_w/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=ORXoOKND1Tk">PC Gamers Learn the truth about RTX 5090&#39;s True Performance</a>: Nvidia has just unveiled their brand new Blackwell RTX 50 Series GPUs, their claim is that The RTX 5070 achieves the same performance as The RTX 5090 but the...</li><li><a href="https://github.com/ollama/ollama/pull/8134#issuecomment-2550018120">feat: Introduce speculative decoding by bfroemel · Pull Request #8134 · ollama/ollama</a>: This PR aims to replicate speculative decoding as implemented in https://github.com/ggerganov/llama.cpp/blob/master/examples/server/server.cpp.See hints in the documentation (docs/faq.md) for tryi...
</li>
</ul>

</div>
  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1326280093273821205)** (187 messages🔥🔥): 

> `NVIDIA 5090 Graphics Card, Commercial Use of Stable Diffusion Models, Lora Training Techniques, Creating Realistic Monsters with AI, Image-to-Image Generation Techniques` 


- **NVIDIA 5090 Graphics Card Speculation**: Discussions revolved around the **NVIDIA 5090**, with members pointing out that it might outperform the **4090** significantly in terms of speed and capabilities.
   - One member humorously noted that the **5090** could potentially reduce image generation times to **13 seconds**, compared to **30 seconds** for the **4090**.
- **Guidelines on Commercial Use of Stable Diffusion**: Members discussed the complexities surrounding the **commercial use of Stable Diffusion models**, clarifying that usage under **$1 million** in revenue is generally allowed without a special license.
   - Concerns were raised about compliance with **use licenses**, prompting cues for further exploration on documentation linked to the **community license agreement**.
- **Lora Training Techniques**: Community members shared insights on how to effectively train **Lora** models, emphasizing that as few as **30 quality images** can suffice for good results.
   - Recommendations included exploring video resources and using tools like **CivitAI** to enhance training processes for those new to the system.
- **Creating Realistic Monsters with AI**: Members sought recommendations for models that can generate realistic monsters, with suggestions leaning towards models hosted on **Civitai** and other specialized resources.
   - A proposed model named **THRILLustrious** was highlighted as capable of producing impressive monster-themed images.
- **Image-to-Image Generation Techniques**: Advice was given on using **image-to-image generation** to create various art styles, including using empty frames for avatar designs in games.
   - This includes employing masking techniques or starting with a solid color image, allowing flexibility in framing and design.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.comfy.org/p/hunyuanvideo-native-support-in-comfyui">HunyuanVideo Native Support in ComfyUI</a>: We’re excited to announce that HunyuanVideo, a groundbreaking 13-billion-parameter open-source video foundation model, is now natively supported in ComfyUI!</li><li><a href="https://stability.ai/license">Stability AI License &mdash; Stability AI</a>: Stability AI licenses offer flexibility for your generative AI needs by combining our range of state-of-the-art open models with self-hosting benefits.</li><li><a href="https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/">NVIDIA Open Models License</a>: no description found</li><li><a href="https://stability-ai.squarespace.com/core-models">Stability AI Core Models &mdash; Stability AI</a>: The Core Models are available to Professional and Enterprise Members for commercial use under the terms of their Membership Agreement.</li><li><a href="https://medium.com/@promptingpixels/can-stable-diffusion-models-be-used-for-commercial-use-it-depends-eedd89272245>">no title found</a>: no description found</li><li><a href="https://civitai.green/images/48983430">Image posted by pAInCREAT0R</a>: no description found</li><li><a href="https://civitai.green/models/626819/beauty-in-evil-by-hailoknight?modelVersionId=700704">Beauty In Evil - By HailoKnight - XL | Stable Diffusion XL LoRA | Civitai</a>: “When the people of the world all know beauty as beauty, there arises the recognition of ugliness. When they all know the good as good, there arise...</li><li><a href="https://civitai.green/models/626819/beauty-in-evil-by-hailoknight">Beauty In Evil - By HailoKnight - Flux | Flux LoRA | Civitai</a>: “When the people of the world all know beauty as beauty, there arises the recognition of ugliness. When they all know the good as good, there arise...</li><li><a href="https://github.com/typhon0130">Typhon0130 - Overview</a>: Imagine how you want to feel at the end of the day. Start working 
towards that now. - Typhon0130</li><li><a href="https://m.youtube.com/watch?v=WPKPO-2WFK8"> - YouTube</a>: no description found</li><li><a href="https://github.com/NVIDIA/Cosmos">GitHub - NVIDIA/Cosmos: Cosmos is a world model development platform that consists of world foundation models, tokenizers and video processing pipeline to accelerate the development of Physical AI at Robotics &amp; AV labs.  Cosmos is purpose built for physical AI. The Cosmos repository will enable end users to run the Cosmos models, run inference scripts and generate videos.</a>: Cosmos is a world model development platform that consists of world foundation models, tokenizers and video processing pipeline to accelerate the development of Physical AI at Robotics &amp;amp; AV la...
</li>
</ul>

</div>
  

---


### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1326391462618857543)** (6 messages): 

> `Bolt's capabilities, UI Design Prompts, Prompting Techniques` 


- **Bolt's Wonders through Effective Prompting**: A member highlighted that if you can prompt well, **Bolt can deliver impressive results**.
   - *Trust me*, it’s all about how you phrase your ideas for optimal output.
- **Request for Documentation on Bolt's Features**: A member inquired about whether there are **documentation** available for learning how to utilize Bolt effectively.
   - They expressed curiosity about the **process** behind discovering Bolt's abilities.
- **Admiring UI and Seeking Design Insights**: One member expressed appreciation for the UI and asked what was prompted to achieve that design.
   - In response, another member emphasized the importance of specifying **colors** and where to apply them in prompts.
- **Guidance on Prompting for Design**: A member advised that when prompting, you should convey your vision, but not in excessive detail; just an idea.
   - For instance, they cautioned against vague prompts like *'Make me a timer app. blue and white colors'*.


  

---


### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1326283183007989863)** (180 messages🔥🔥): 

> `Rate Limiting and Token Management, Complex Project Development Tips, Deployment Issues, Supabase Connection Challenges, Use of Different Tools with Bolt` 


- **Understanding Rate Limits and Token Usage**: Users reported confusion about their token limits, particularly with daily and monthly tokens shared across free accounts, leading to potential rate limiting.
   - Suggestions were made to clarify user settings to avoid confusion and improve understanding of token limits in the future.
- **Best Practices for Developing Larger Applications**: Discussion highlighted the importance of breaking larger applications into smaller, manageable components to maintain organization and improve code clarity.
   - Users were encouraged to document project details in an overview file for context when revisiting codebases, enhancing prompting for Bolt.
- **Common Deployment Issues Encountered**: Several users experienced failures while deploying projects, often related to build errors such as 'Failed building the project.'
   - It was recommended to run terminal commands to diagnose errors directly instead of relying solely on Bolt for fixes.
- **Challenges with Supabase Connections**: Issues connecting existing Supabase projects with Bolt were raised, with users needing to resubmit .env variable configurations upon disconnection.
   - Participants expressed the desire for more seamless connectivity solutions that persist across project changes without needing to recreate Supabase instances.
- **Integrating Multiple Tools with Bolt**: The effectiveness of using Bolt in combination with tools like Cursor and Copilot was discussed, suggesting that each tool serves a specific purpose.
   - Users shared their experiences and preferences, indicating that using multiple tools can enhance the development process.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://repocloud.io/boltdiy">RepoCloud | Bolt.diy: Choose Your AI Model</a>: Discover Bolt.diy, the ultimate fork for selecting your favorite AI model. Customize your coding experience with top LLMs like OpenAI and Anthropic!</li><li><a href="https://github.com/stackblitz/bolt.new/issues/2529">Bolt Outputs Application Logic in Chat · Issue #2529 · stackblitz/bolt.new</a>: Issue: Bolt outputs application logic in the chat. For example, when the user hits a rate limit, the code to offer a link to upgrade is sent as a response to the user in chat.</li><li><a href="https://github.com/stackblitz/bolt.new/issues/5149">Suggestion: Selector · Issue #5149 · stackblitz/bolt.new</a>: This is my suggestion to add a selector option for the sites. I will try to explain in more detail: When you highlight with your mouse and go to chat and say for example change the name or remove t...</li><li><a href="https://github.com/stackblitz/bolt.new/issues">Issues · stackblitz/bolt.new</a>: Prompt, run, edit, and deploy full-stack web applications - Issues · stackblitz/bolt.new
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1326294031038152729)** (101 messages🔥🔥): 

> `Sonnet vs O1 Pro performance, Aider usage tips, DeepSeek model performance, Sudoku solving discussion, Clickbait video frustration` 


- **Comparison between Sonnet and O1 Pro**: Users expressed varying opinions on the performance of **Sonnet** compared to **O1 Pro**, with some claiming Sonnet is *as good* for their needs.
   - A user noted that combining Sonnet with O1 Pro yields *astounding results*, particularly when crafting prompts.
- **Tips for Using Aider Effectively**: A user suggested reading all comments from Aider to maximize the value of the tokens spent, along with properly crafting /ask prompts for clearer outputs.
   - They emphasized the importance of constructing effective architect prompts by copying and pasting Aider's responses.
- **Issues with DeepSeek and Model Performance**: Several users reported experiencing freezes and delays with **DeepSeek v3**, questioning whether models could become *overloaded* from continuous use.
   - Despite some experiencing issues, others claimed never to have faced slowdowns while using DeepSeek or other models.
- **Sudoku Solving Success**: A Discord member shared a Sudoku puzzle link, with a subsequent discussion about the model's performance, proclaiming the original grid does not provide a unique solution.
   - Another user's output was celebrated for successfully solving the Sudoku with a singular completion.
- **Frustration with Clickbait Content**: A user expressed disdain for clickbait YouTube videos, wishing that disliked videos could impact the algorithm significantly.
   - This created a discussion around the presence of clickbait creators, with those involved sharing a common frustration.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://nometa.xyz/">nometa</a>: no description found</li><li><a href="https://tenor.com/view/patrick-stewart-cyborg-serious-scary-gif-1000347867466686561">Patrick Stewart Cyborg GIF - Patrick Stewart Cyborg Serious - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/vectara/hallucination-leaderboard?tab=readme-ov-file#hallucination-leaderboard">GitHub - vectara/hallucination-leaderboard: Leaderboard Comparing LLM Performance at Producing Hallucinations when Summarizing Short Documents</a>: Leaderboard Comparing LLM Performance at Producing Hallucinations when Summarizing Short Documents - vectara/hallucination-leaderboard
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1326300425070055538)** (49 messages🔥): 

> `Aider Usage Issues, Litellm Custom Model Setup, Ollama Model Interaction, Deepseek Configuration on OpenRouter, Message Formatting Errors` 


- **Troubleshooting Aider File Updates**: A user reported that Aider shows changes but does not update files in their repository. They are troubleshooting the issue by simplifying requests and considering potential Python errors.
- **Configuring Litellm for Custom Models**: Another user shared their experience with setting up a custom Litellm model and the confusion around API base settings. They discovered the need to correctly prefix model names and configure settings in the proper files to work with Aider.
- **Challenges with Ollama Models**: A user faced issues interacting with a local Ollama model due to API base configuration errors. They resolved the problem by correctly specifying model locations and ensuring the API base was set properly.
- **Deepseek Provider Issues**: Aider's users encountered a NotFoundError related to ignored providers in Deepseek on OpenRouter. Restarting Aider temporarily resolved the problem, but it was attributed to excessive context being sent.
- **Message Structure in Litellm Communication**: A user noted that Aider sends a 'prompt' list instead of a 'messages' list when communicating with Litellm, leading to errors on the server side. They are seeking clarification and potential solutions to modify this behavior.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://0.0.0.0:8000"">no title found</a>: no description found</li><li><a href="https://aider.chat/docs/config/adv-model-settings.html">Advanced model settings</a>: Configuring advanced settings for LLMs.</li><li><a href="https://aider.chat/docs/config/options.html">Options reference</a>: Details about all of aider’s settings.</li><li><a href="https://github.com/ollama/ollama/issues/2203#issuecomment-2005184119.">Model not found · Issue #2203 · ollama/ollama</a>: First of all, I must say, what a great piece of software Ollama is! THANK YOU for all your work everyone!!! I am trying to setup MemGPT to use CodeLlama via ollama serve I&#39;ve made sure that I&#39;...
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1326340555940036722)** (7 messages): 

> `LLM Interviewing, SynthLang, Gemini 2.0 Flash Experimental` 


- **LLM Guides User Through Specification Creation**: One user shared their experience of using an LLM to interview them and create specifications for coding prompts, enhancing their project development process.
   - They found it particularly helpful for generating criteria and tasks from their conversations.
- **Exploring SynthLang Platform**: Multiple users expressed interest in the [SynthLang platform](https://synthlang.fly.dev/), noting its intriguing features.
   - However, one user reported encountering a lot of errors when trying to select a proper model.
- **Feedback Mechanism for SynthLang**: Discussions highlighted issues with SynthLang, prompting one user to suggest filing a bug report for the errors they experienced.
   - The conversation emphasized the importance of addressing these technical woes for the user experience.
- **Casual Interaction with Gemini 2.0**: A user recounted their experience with **Gemini 2.0 Flash Experimental**, utilizing its voice mode to brainstorm app ideas during errands.
   - They appreciated the generated task bullet points that summarized their conversation, wishing for markdown outputs in the future.



**Link mentioned**: <a href="https://synthlang.fly.dev/">SynthLang - Prompt Generator & Tester</a>: no description found

  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1326280846696775762)** (153 messages🔥🔥): 

> `Cursor IDE Bugs, Composer Functionality, Flutter Development, Technical Debt in Coding, User Experience Issues in Cursor` 


- **Stability Issues with Composer**: Users report frustration with the Composer feature in Cursor, often getting stuck after only a few messages and unable to generate outputs, leading to a need for opening new sessions.
   - Repeated errors, especially related to linting, have been reported, prompting discussions about potential bugs introduced in recent updates.
- **Technical Debt and Code Organization**: Participants discussed the importance of maintaining small, manageable code files to avoid technical debt and make projects easier to maintain.
   - One user emphasized avoiding long files with multiple responsibilities, arguing that such practices can complicate future development and learning for new team members.
- **Difficulties in Flutter Development**: A new user detailed challenges encountered while installing dependencies for a Flutter mobile app project, specifically with TensorFlow and Keras.
   - Users shared experiences and insights on setting up Flutter projects in Cursor IDE, noting the necessity for proper dependencies to ensure smooth functionality.
- **Issues with AI Model and Apply Feature**: Discussion highlighted the frequent unreliability of the Apply feature in Cursor, causing users to lose trust in its capability to manage code updates effectively.
   - There were suggestions that the internal models may be insufficiently trained, contributing to inconsistent interactions when reading local files.
- **Account Confusion with Cursor**: Users encountered issues related to multiple trial accounts on a single machine, affecting their ability to utilize full credits and functionalities.
   - One user mistakenly logged in with GitHub, leading to accidental account creation and subsequent confusion, which required clarification and troubleshooting.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.cursor.com/get-started/usage#purchasing-additional-requests">Get Started / Usage – Cursor</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=Vy7dJKv1EpA"> - YouTube</a>: no description found</li><li><a href="https://www.nvidia.com/en-us/project-digits/">NVIDIA Project DIGITS: The World’s Smallest AI Supercomputer. </a>: Reserve yours today.</li><li><a href="https://forum.cursor.com/t/composer-stuck-at-generating-specific-composer-instance-not-global-issue/35479/4">Composer Stuck at &quot;Generating&quot; - Specific Composer Instance, Not Global Issue</a>: Hey… any luck on this yet? I’m still seeing stuck Composer sessions. I’ve just upgraded to 0.44.10; my current session which was stuck in 0.44.9 remains stuck in 0.44.10.  It sits on “generating” for ...
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1326286255209713727)** (23 messages🔥): 

> `System Prompt for Quoting, Language Settings in NotebookLM, Repurposing Content, Business Use Cases for AI, Video Content Analysis` 


- **Creating Effective System Prompts**: A member inquired about a system prompt for NotebookLM to quote relevant sources directly without additional commentary, while another shared their existing prompt focusing on text accuracy.
   - Another member pointed out that clarity in the instructions can help guide NotebookLM's performance in quoting effectively.
- **Changing Language Preferences**: There was a discussion about language settings in NotebookLM, including how to force responses in English and changing client language settings to avoid unintentional multilingual replies.
   - One member suggested adding language parameter codes to the URL to ensure responses are in the desired language.
- **Utilizing NotebookLM for Content Repurposing**: A member shared a video on using NotebookLM to convert long-form content into social media posts, highlighting its usefulness for writers and content creators.
   - Another member expressed interest in repurposing old podcast material, realizing its cultural significance and the value of new perspectives.
- **Business Use Cases for AI in Contract Management**: One user proposed the use of 'Digital Labor' for contract redlining, aiming to ease the workload on busy paralegals and stakeholders involved in contract reviews.
   - They emphasized the potential efficiency enhancements by using virtual paralegals to facilitate understanding among parties involved.
- **Challenges with Video Imports in NotebookLM**: Members discussed the challenges faced when trying to import videos into NotebookLM, with one noting that import attempts resulted in the video being incompatible due to a lack of transcripts.
   - There was humor exchanged about the feasibility of importing video content, considering alternatives like transcribing video audio for analysis.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=spj0n-bFKJo"> - YouTube</a>: no description found</li><li><a href="https://youtu.be/1G4W6XWl2WE"> - YouTube</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=4QJm_AptHF4">How To Repurpose Content - NotebookLM 📝 Effortless Social Media Posts</a>: Hi Friends, my name is Callum aka wanderloots. In this video, I walk through how to use NotebookLM to repurpose existing or new content into Social Media Pos...</li><li><a href="https://www.akashq.com/post/ad632a26-91b5-44b4-b8f4-5b5fd3f083e8">What happened on Jan 8?</a>: What happened on Jan 8? by This Day in History
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1326292068947202170)** (86 messages🔥🔥): 

> `NotebookLM Plus Access Issues, Using NotebookLM for Education, Podcast Features, Customization Challenges, General Usage Feedback` 


- **Understanding NotebookLM Plus Access**: Users discussed multiple requirements for accessing NotebookLM Plus, including needing a Business Starter license and activation of the NotebookLM Service under organizational units.
   - A detailed list of criteria was shared to assist users in troubleshooting their access issues.
- **Educational Applications of NotebookLM**: One user highlighted the challenges of sharing notebooks in education, particularly noting limited features for 'view only' versus 'chat only' modes.
   - Feedback was requested on the potential for more advanced functionality in 'view only' mode to enhance student engagement.
- **Challenges with Podcast Features**: There were discussions on the podcast features, with users noting that hosts sometimes deliver monologues or respond inconsistently, such as flipping scripts irregularly.
   - Suggestions included using the customization feature to enforce more consistent dialogue delivery among hosts.
- **Quote Extraction Limitations**: A user inquired why NotebookLM only extracts quotes from the first 13 pages of a lengthy 250-page document, highlighting potential issues with material processing.
   - This raised concerns about the efficiency of the source-uploading process within NoteBookLM.
- **Functionality of Audiobook Narration**: A user expressed frustration with getting hosts to deliver audiobook narration expressively and verbatim, experiencing issues with tone and delivery quality.
   - This led to requests for improvements in the voice modulation and adherence to scripts during narration.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://support.google.com/notebooklm/answer/15678219?hl=en#:~:text=Where%20to%20get%20NotebookLM%20Plus%C2%A0">Upgrading to NotebookLM Plus - NotebookLM Help</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=w7PA9kSJLlo"> - YouTube</a>: no description found</li><li><a href="https://youtu.be/spj0n-bFKJo"> - YouTube</a>: no description found</li><li><a href="https://support.google.com/a/answer/6043385?hl=en&co=DASHER._Family%3DBusiness">Compare Google Workspace editions - Business - Google Workspace Admin Help</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1326409538144047155)** (2 messages): 

> `Model Context Protocol, Agents Base launch, Marketing automation` 


- **Introducing Model Context Protocol for Twitter**: A new GitHub project, **x-mcp**, aims at bridging Twitter and AI with the Model Context Protocol, providing users with full control on the platform.
   - For more details, visit the [GitHub repository](https://github.com/lord-dubious/x-mcp) and explore how it can enhance Twitter functionality.
- **Agents Base Launches on Product Hunt**: A new product, **Agents Base**, has launched on Product Hunt, designed to automate marketing with swarms of cloud agents for incredible performance.
   - It claims to achieve **50-500x better CPM** than traditional ad platforms and offers features for automating video repurposing and SEO optimized content. Check it out [here](https://www.producthunt.com/posts/agents-base).
- **Marketing Agents for Brand Growth**: **Agents Base** enables brands to grow on autopilot using marketing agents that perform A/B testing across various demographics and formats.
   - This automation framework aims to streamline marketing efforts significantly, as highlighted by the positive reception and discussion on Product Hunt.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.producthunt.com/posts/agents-base"> Agents Base - Grow any brand on autopilot with swarms of marketing agents | Product Hunt</a>: Deploy swarms of cloud marketing agents that automate A/B testing across demographics, copywriting, and viral video styles to get 50-500x better CPM than Google, Instagram, or TikTok ads. Automate rep...</li><li><a href="https://github.com/lord-dubious/x-mcp">GitHub - lord-dubious/x-mcp: Bridging Twitter and AI with the Model Context Protocol</a>: Bridging Twitter and AI with the Model Context Protocol - lord-dubious/x-mcp
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1326292901071945821)** (60 messages🔥🔥): 

> `LLM Game Development, Azure Model Integration, AI Model Conversation Preferences, Bug Reports on Llama Models, API Call Timeout Issues` 


- **LLMs Struggle with Game Development**: Members discussed that current LLMs lack proper **world models**, making it difficult to create complex games like 3D FPS titles, while simpler games are feasible with careful prompting.
   - One suggestion pointed out that LLMs can produce simple games, but they require constant feedback and debugging from users to avoid getting stuck on bugs.
- **Integrating Azure Models with OpenRouter**: A user posed a question on how to use a hosted **gpt-4o model** on Azure with OpenRouter, receiving suggestions to look at the available models directly on the OpenRouter platform.
   - Further information was provided about checking differences between Azure-hosted models and those provided directly by OpenAI.
- **Preferences in LLM Conversations**: Participants shared their favorite models for casual chat, with recommendations for **Gemini 1206** and **Flash thinking** for non-coding discussions.
   - While preferences varied, some criticized the **Claude** model for its system prompts due to perceived limitations in conversational quality.
- **Bug Reports in Llama Models**: A user reported a potential bug in **Llama** models where the `usage` object returned all zero values for token counts, suggesting a persistent issue.
   - Another participant confirmed the **zero value issue** has been occurring for months, signaling ongoing trouble with the model's functionality.
- **Challenges with Vercel API Call Timeouts**: A discussion emerged about the **10 second timeout** issue occurring with Vercel when making API calls, leading users to seek solutions for overcoming this limitation.
   - One member indicated that issues with registration processes were also problematic, suggesting further complications around API interactions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://medium.com/@kellytgold/building-games-with-openais-o1-model-da3fc8d1a4e6">Building Games With OpenAI’s o1 Model.</a>: Let’s see what this new model can do!</li><li><a href="https://openrouter.ai/provider/azure">Azure | OpenRouter</a>: Browse models provided by Azure</li><li><a href="https://openrouter.ai/openai/gpt-4o-2024-11-20">GPT-4o (2024-11-20) - API, Providers, Stats</a>: The 2024-11-20 version of GPT-4o offers a leveled-up creative writing ability with more natural, engaging, and tailored writing to improve relevance &amp; readability. It’s also better at working with...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1326400120908156983)** (12 messages🔥): 

> `Feedback on Current State, Font Weight Adjustment, CPU/GPU Pairing, AMD vs Nvidia Performance` 


- **Feedback reflects current state**: Members confirmed that the feedback shared resonates with their **current state** of mind.
   - *Yes,* they expressed, indicating general agreement with this sentiment.
- **Discussion on font weight**: Members are currently using **TT Hoves Light** font but discussed increasing the **font-weight** for better visibility.
   - This indicates a desire for text that stands out more in their interface.
- **Poll on CPU/GPU combinations**: A user inquired whether an **AMD CPU** works better with **Nvidia** or **AMD** GPUs.
   - The consensus was that it doesn't matter much, with one member suggesting AMD for its support of **AVX512**.
- **Preference for AMD GPU with AMD CPU**: A member leaned towards using an **AMD CPU** with an **AMD GPU** for optimal performance.
   - This preference was echoed, emphasizing a potential advantage under specific circumstances.



**Link mentioned**: <a href="https://tenor.com/view/reaction-my-eyes-cant-unsee-burn-gif-7225082">Reaction My Eyes GIF - Reaction My Eyes Cant Unsee - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1326294567791759500)** (47 messages🔥): 

> `Indexing static lists in Mojo, Difference between ListLiteral and VariadickPack, Traits development in Mojo, Overloads and polymorphism proposals, Static analysis methods in Mojo` 


- **Runtime Variables and Static Lists**: The consensus is that you cannot index a `ListLiteral` using a runtime variable due to type restrictions, with a recommendation to use `InlineArray` instead.
   - One member expressed confusion about their previous attempts with `InlineArray`, noting that it worked successfully upon re-evaluation.
- **ListLiteral vs VariadickPack Explained**: A member highlighted that `ListLiteral` is not as useful due to its limitations, suggesting that `Tuple` could be a better alternative for fixed-length collections.
   - It was clarified that `VariadicPack` is primarily for function calls and cannot be easily instantiated outside of that context.
- **Mojo Traits Need Improvement**: The community discussed the need for better trait capabilities in Mojo, with desires for features like conditional traits, default functions, and parametric traits.
   - There is an ongoing conversation about potentially aligning trait features closer to Rust's model for greater expressiveness.
- **Proposals for Overloads and Polymorphic Functions**: One member put forward ideas for allowing OOP-style overloads and polymorphic functions, emphasizing the need for priority levels to manage overlapping signatures.
   - Concerns were raised regarding how type narrowing should automatically trigger overload selection to enhance soundness.
- **Concerns on Overload Resolution Complexity**: A member expressed apprehension about the complexity that could arise from combining `TraitVariant` with overload resolution, possibly leading to ambiguous implementations.
   - The need for clearer syntax and organization in large codebases was emphasized, as well as potential issues with `where` clauses causing duplicate trait implementations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/mo">mo - Overview</a>: mo has 49 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/3403)">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/3252).">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/3630)">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1326377051095175168)** (56 messages🔥🔥): 

> `Model Performance and Quantization, GPU Support Issues, Hiring Opportunities in AI, Q4_0 Model Issues, GPT4All Community Contributions` 


- **Model Performance Affected by Quantization**: Members discussed the implications of quantizing models, particularly highlighting that low-bit quantization can degrade performance, especially in coding tasks. A member noted that quantization diminishes the effectiveness of models trained on large datasets, referencing scholarly articles on quantization-induced degradation.
   - It's emphasized that low parameter-sized models (below 7b) might experience significant differences in performance between quantized versions.
- **Challenges in GPU Support for Models**: Several members expressed frustration over the inability to use GPU support with various Q4_0 models, citing issues with crashing on loading. One member confirmed success with GPU support using llama.cpp but noted significant differences in performance compared to GPT4All.
   - The conversation included insights on limitations of CUDA and the necessity of proper support for GPU acceleration depending on the user's hardware.
- **Opportunities for Hiring in AI Development**: A member announced an opportunity to hire junior engineers for work on agent development, emphasizing task-based work with payment upon successful PR merges. Additionally, they expressed a need for a UX designer skilled in Figma or AdobeXD.
   - The request was open to US-based candidates seeking to join in the development efforts.
- **Q4_0 Model Performance Issues**: Users reported performance issues with multiple Q4_0 models causing crashes in GPT4All, with some quantized versions reportedly better suited for testing. Members discussed the potential for a Q8_0 variant that they speculated might not have the same issues.
   - One user shared a Q4_0 GGUF model link that seemingly worked, along with the understanding that different versions impacted performance in coding scenarios.
- **Community Contributions and Model Sharing**: Users actively shared links to GGUF models on Hugging Face, with one describing the upload of a near-official Q4_0 model that successfully handled JavaScript tasks. Members highlighted licensing, confirming some models were under MIT licenses as indicated by their release details.
   - This collaboration reflects the community's efforts to improve model accessibility and performance in practical tasks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2411.17691">Low-Bit Quantization Favors Undertrained LLMs: Scaling Laws for Quantized LLMs with 100T Training Tokens</a>: We reveal that low-bit quantization favors undertrained large language models (LLMs) by observing that models with larger sizes or fewer training tokens experience less quantization-induced degradatio...</li><li><a href="https://huggingface.co/SamPurkis/Microsoft_Phi-4-Q4_0-GGUF/tree/main">SamPurkis/Microsoft_Phi-4-Q4_0-GGUF at main</a>: no description found</li><li><a href="https://huggingface.co/GPT4All-Community/phi-4-GGUF/blob/main/phi-4-Q4_0.gguf">phi-4-Q4_0.gguf · GPT4All-Community/phi-4-GGUF at main</a>: no description found</li><li><a href="https://huggingface.co/JackCloudman/Phi-4-jackterated-GGUF/tree/main">JackCloudman/Phi-4-jackterated-GGUF at main</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/10817">Add support for Microsoft Phi-4 model  by fairydreaming · Pull Request #10817 · ggerganov/llama.cpp</a>: This PR adds support for Microsoft Phi-4 model. Fixes #10814.Current solution is to:Use tokenizer_class value from tokenizer_config.json as a condition to use GPT2 vocab during model conversion....
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1326284957508702391)** (40 messages🔥): 

> `Networking Solutions for Budget, Phi-4 Model Technical Insights, USB Networking Capabilities, Job Opportunities in Web Development` 


- **Budget-Friendly Networking Options for PCs**: Members discussed alternative networking options for linking PCs, with suggestions for 10GbE and USB-C providing 10-20Gbps connections.
   - One member found traditional networking equipment to be unexpectedly expensive, exploring whether older Mellanox adapters might be a viable solution.
- **Insights on Phi-4 Model Release**: The release of the **Phi-4** model by Microsoft reveals a surprisingly simple fine-tuning pipeline that primarily uses **SFT** and **DPO** methods.
   - Despite its sophisticated results in math and reasoning, this simplicity suggests that open-source projects might leverage good synthetic datasets to achieve comparable outcomes.
- **Exploring USB as an Ethernet Substitute**: Members evaluated the possibility of using USB for full network stacks, particularly to connect Windows and Linux systems smoothly.
   - It's noted that USB can function as an Ethernet port, although practical implementation of this solution may vary.
- **Job Posting for Web Developers**: A member is seeking web developers and inquired about platforms for job postings.
   - **Salgadev** expressed interest in the opportunity and requested a private discussion about the necessary skill set.



**Link mentioned**: <a href="https://huggingface.co/microsoft/phi-4">microsoft/phi-4 · Hugging Face</a>: no description found

  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1326494820696326195)** (3 messages): 

> `Zero Trust in Development, Using Placeholder Data, MVP Development Environment, Solutions for Early Development` 


- **Discussion on Zero Trust Requirement**: *Footlooseboss* questioned whether a **zero trust** framework is necessary from the start or if one can use **placeholder data** to build applications in the cloud before committing to local hardware.
   - This approach allows developers to iterate without immediate requirements for full data security.
- **MVP Not Needing Final Environment**: *Regis369* affirmed that a minimum viable product (**MVP**) doesn't need to exist in the final security environment, suggesting flexibility during early stages of development.
   - This opens discussions on potential interim solutions that balance development speed and security.
- **Inquiry About Development Projects**: *Senor1854* inquired about the nature of the project being developed, inviting more details from the initial poster.
   - Understanding the specific project could help tailor discussions and recommendations for effective development solutions.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

craftycannon_98161: Any progress?
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1326334069180731457)** (3 messages): 

> `Structure of Neural Embeddings, MiniMind Lightweight Language Model, Training Pipeline for LLMs` 


- **Explore the Structure of Neural Latent Spaces**: A [blog post](https://seanpedersen.github.io/posts/structure-of-neural-latent-space) discusses insights on the structure of embeddings produced by deep neural networks, focusing on the **manifold hypothesis** that asserts high-dimensional data lies in low-dimensional manifolds.
   - It also highlights **hierarchical organization** of features across layers and the **linear hypothesis** regarding feature representation in activation space, linking to relevant articles for deeper understanding.
- **MiniMind: Training a Tiny LLM in 3 Hours**: The **MiniMind** project aims to train a small language model (26.88M) from scratch in just **3 hours**, suitable for personal GPUs, with a full training pipeline available on [GitHub](https://github.com/jingyaogong/minimind).
   - It includes comprehensive code for dataset preprocessing, supervised pretraining, instruction fine-tuning, and advanced features like **low-rank adaptation** and reinforcement learning techniques.
- **Lightweight LLMs for Everyone**: **MiniMind** exemplifies an extremely lightweight model, approximately **1/7000th** the size of GPT-3, allowing for rapid inference and training even on standard hardware.
   - The project not only serves as an implementation but also as a tutorial for beginners interested in developing large language models (LLM).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://jingyaogong.github.io/minimind/">MiniMind Project</a>: no description found</li><li><a href="https://seanpedersen.github.io/posts/structure-of-neural-latent-space">Structure of Neural Embeddings</a>: no description found</li><li><a href="https://github.com/jingyaogong/minimind/blob/master/README_en.md">minimind/README_en.md at master · jingyaogong/minimind</a>: 「大模型」3小时完全从0训练26M的小参数GPT，个人显卡即可推理训练！. Contribute to jingyaogong/minimind development by creating an account on GitHub.</li><li><a href="https://github.com/jingyaogong/minimind">GitHub - jingyaogong/minimind: 「大模型」3小时完全从0训练26M的小参数GPT，个人显卡即可推理训练！</a>: 「大模型」3小时完全从0训练26M的小参数GPT，个人显卡即可推理训练！. Contribute to jingyaogong/minimind development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

craftycannon_98161: Any progress?
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1326294366280482946)** (15 messages🔥): 

> `Pythia Evaluation, Learning AI Tools, Supervised Fine-Tuning Libraries` 


- **Inquiry on Pythia Evaluation on Ethics Dataset**: Members discussed if anyone had evaluated **Pythia** on the [Ethics Dataset](https://huggingface.co/datasets/hendrycks/ethics). No concrete evaluations were reported during the discussion.
   - This highlights a potential area for exploration within the community.
- **Navigating AI Learning Resources**: A member expressed frustration with tutorials, stating they often filter out relevance and understanding, advocating for practical approaches instead. They suggested cloning **nanoGPT** for hands-on learning, emphasizing the importance of directly engaging with implementations and papers.
   - This approach focuses on self-learning through exploration rather than relying solely on structured tutorials.
- **Open Source Libraries for Supervised Fine-Tuning**: Discussion was raised about which open-source libraries labs like **AllenAI** use for Supervised Fine-Tuning (SFT), with recommendations pointing towards [open-instruct](https://github.com/allenai/open-instruct) as a viable option. Members mentioned that **GPT-NeoX** supports both SFT and RLHF effectively.
   - Furthermore, it was noted that **NVIDIA NeMo** also offers megatron-based implementations for SFT and RLHF, pointing towards robust options available in the field.
- **Performance Trade-offs in Finetuning Libraries**: A member shared that **Open-Instruct** is based on **TRL** and **HF Trainer**, which are noted for their ease of use despite some performance drawbacks. In contrast, **GPT-NeoX** was favored for its superior implementation performance.
   - The conversation indicated varying preferences among practitioners regarding the choice of libraries for finetuning tasks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/allenai/open-instruct">GitHub - allenai/open-instruct</a>: Contribute to allenai/open-instruct development by creating an account on GitHub.</li><li><a href="https://github.com/huggingface/smollm">GitHub - huggingface/smollm: Everything about the SmolLM &amp; SmolLM2 family of models</a>: Everything about the SmolLM &amp; SmolLM2 family of models  - GitHub - huggingface/smollm: Everything about the SmolLM &amp; SmolLM2 family of models</li><li><a href="https://huggingface.co/datasets/hendrycks/ethics">hendrycks/ethics · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1326354916545138729)** (11 messages🔥): 

> `Cross-Entropy Memory Optimization, SD3 Paper Discussion, HunyuanProver for Theorem Proving` 


- **Cut Cross-Entropy Memory Innovation**: The paper introduces a method called Cut Cross-Entropy (CCE) that reduces global memory consumption during the loss computation process in language models by only computing logits for the correct token.
   - This method significantly optimizes memory usage by evaluating the log-sum-exp on-the-fly, alleviating the logit matrix's burden, as described in the [original paper](https://arxiv.org/abs/2411.09009v1).
- **Confusion Over Forward and Backward Processes in SD3**: A member questioned whether the reference to a forward process in the SD3 paper means it fails to remove all data from noise, speculating it could actually pertain to the backward process instead.
   - Another member noted that they are likely discussing the forward process given the references to the zero SNR paper, indicating potential oversight in the text that has persisted for months.
- **HunyuanProver Achieves Theorem Proving Milestone**: The HunyuanProver, a model fine-tuned from Hunyuan 7B for automatic theorem proving with LEAN4, shows state-of-the-art performance with a **68.4% pass** rate on the miniF2F-test.
   - This system has proven several IMO statements and intends to contribute to the community by open-sourcing a dataset of 30k synthesized instances, enhancing accessibility for researchers ([View Paper](https://arxiv.org/abs/2412.20735)).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2411.09009v1">Cut Your Losses in Large-Vocabulary Language Models</a>: As language models grow ever larger, so do their vocabularies. This has shifted the memory footprint of LLMs during training disproportionately to one single layer: the cross-entropy in the loss compu...</li><li><a href="https://arxiv.org/abs/2412.20735">HUNYUANPROVER: A Scalable Data Synthesis Framework and Guided Tree Search for Automated Theorem Proving</a>: We introduce HunyuanProver, an language model finetuned from the Hunyuan 7B for interactive automatic theorem proving with LEAN4. To alleviate the data sparsity issue, we design a scalable framework t...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/)** (1 messages): 

teknium: woops dno how that image got there
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1326471577872896085)** (21 messages🔥): 

> `OOM Issues with 6.7B model, DeepSpeed Pipe Module Performance, AdamW Optimizer Details, Batch Size Behavior in Training, BF16 Loss Scaling Discussion` 


- **OOM Issues with 6.7B Model**: A user recently transitioned to a **6.7B model** but encountered **Out of Memory (OOM)** errors even at a batch size of 1. They mentioned the need to conduct a thorough debug session to understand the cause.
   - *It was noted that weird OOM signals led to process being killed errors*, indicating potential issues in resource handling.
- **DeepSpeed Pipe Module Performance Concerns**: There was discussion regarding the **DeepSpeed pipe module**, where setting it to 0 yielded faster performance despite initial expectations. One member expressed uncertainty about why similar-sized models did not exhibit this behavior in the past.
   - *I suspect that something snuck into DeepSpeed* was mentioned, suggesting recent changes may have affected performance.
- **Clarification on AdamW Optimizer**: Clarification was provided that **AdamW** is simply the **'adam'** optimizer with weight decay. This points to a shared understanding that the benefits of AdamW stem from its improved handling of regularization.
   - The exchange indicated a **sensible** grasp on conflicting model training dynamics among the members involved.
- **Batch Size Behavior in Training**: There was confusion regarding **batch size behavior**, with a user noting inconsistencies found when shifting to the larger model. Discussions indicated that performance changes could be unexpected as model size increases.
   - *Hmm yeah these seem sensible*, but uncertainty remained about why such behavior was occurring.
- **BF16 Loss Scaling Discussion**: It was mentioned that there is *no need to use loss scaling for BF16*, as it has enough range to avoid overflows. If BF16 overflows occur, one member humorously noted that the run would be 'beyond saving'.
   - This pragmatic advice aimed to improve understanding and efficiency in BF16 model training processes.



**Link mentioned**: <a href="https://api.wandb.ai",">no title found</a>: no description found

  

---


### **Interconnects (Nathan Lambert) ▷ #[events](https://discord.com/channels/1179127597926469703/1179127598442348729/1326317946443796530)** (2 messages): 

> `Thursday Meeting, Shack15 Venue` 


- **Setting Up Thursday Meeting at Shack15**: A member proposed to meet on **Thursday morning** at **Shack15**.
   - Another member confirmed, suggesting **10:00 AM** as the meeting time.
- **Agreement on Meeting Time**: The initial message confirmed a **Thursday** meeting, with details finalized shortly after.
   - Both members are aligned on the time and location, facilitating an efficient coordination.


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1326285551485194321)** (13 messages🔥): 

> `01.AI Rumors and Valuation, Institutional Data Initiative, AI for Good: Omdena, Hugging Face and Phi-4` 


- **01.AI refutes selling rumors to Alibaba**: 01.AI, a leading AI startup in China, achieved a **$1 billion valuation** within eight months and addressed rumors of disbanding and selling teams to Alibaba as *completely false*, claiming revenues surpassed **RMB 100 million** in 2024.
   - CEO Kai-Fu Lee stated that despite layoffs in mid-December 2024, the company expects significant growth in 2025.
- **Harvard's Institutional Data Initiative**: The Institutional Data Initiative aims to refine datasets in collaboration with various knowledge institutions, with open releases planned for **early 2025** as part of its mission to enhance understanding of the data shaping AI.
   - They invite collaborations, contributions, and are hiring researchers to expand their role as data stewards in the AI age.
- **Omdena addresses real-world challenges**: A member highlighted **Omdena**, an organization that tackles real-world problems using AI through collaborative projects involving teams of up to 50 people for various challenges.
   - Among the project types, they emphasize *local solutions* that address specific community challenges while leveraging global talent.
- **Sebastien Bubeck shares Hugging Face link**: Sebastien Bubeck shared a link to [Hugging Face's Phi-4 model](https://huggingface.co/microsoft/phi-4) with an upbeat message, generating interest among members.
   - The tweet emphasizes engagement with leading AI tools, inviting exploration within the community.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://institutionaldatainitiative.org/#get-involved">The Institutional Data Initiative at Harvard Law School Library</a>: no description found</li><li><a href="https://www.omdena.com/projects">Projects | Omdena - Building Ethical AI Solutions for Real-World Problems</a>: Omdena AI projects are the best way to build sought-after data science and machine learning skills while solving real-world problems.</li><li><a href="https://x.com/sebastienbubeck/status/1877010995727470877?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">Tweet from Sebastien Bubeck (@SebastienBubeck)</a>: Enjoy!https://huggingface.co/microsoft/phi-4</li><li><a href="https://technode.com/2025/01/07/01-ai-refutes-rumors-of-selling-teams-to-alibaba/">01.AI refutes rumors of selling teams to Alibaba &#183; TechNode</a>: 01.AI, one of China’s leading AI unicorn startups, has been rumored to be disbanded, with its pre-training and card teams reportedly sold to Alibaba. In
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1326617035513528322)** (11 messages🔥): 

> `MoE models efficiency, Expert weight loading, OlMoE in vLLM, Transformer architectures, Peak performance in MoEs` 


- **Understanding MoE models efficiency claims**: A member questioned the efficiency of **MoE models**, pondering whether they need to load/unload expert weights per token or keep all experts loaded to achieve parallel utilization.
   - They wondered about potential throughput gains if multiple experts could be utilized efficiently, suggesting the complexity might be beneficial for large scale providers.
- **Concerns over MoE operational complexity**: Some members expressed skepticism regarding **MoE models**, stating that increased VRAM use cannot justify practical gains when batch sizes are small.
   - A member suggested exploring references in **OlMoE**, but others advised caution regarding reading **transformers** due to inefficiencies with expert looping.
- **Potential of SGLang and vLLM for MoE**: Members pointed to **SGLang** and **vLLM** as resources for understanding **OlMoE** implementations, although they noted a lack of minimal repos for reference.
   - One member commented on the architectural advantage that MoEs could offer maturing models, despite their implementation complexity.
- **Performance Debate on Transformer Architectures**: A member criticized the transformer architecture for its inefficient **for-loop** over experts, implying it hampers performance.
   - Another member suggested that this looping approach could serve as a basic understanding before diving deeper into more complex architectures.


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1326369096165752963)** (7 messages): 

> `ChatGPT Versions, Token Usage Concerns, OpenAI Executive Predictions, Community Dynamics` 


- **ChatGPT gets creative with names**: A user humorously noted how their Windows 10 transcription produces various quirky versions of ChatGPT, including names like **chab ptb** and **chatty gpt**.
   - *“Lmao,”* they added, highlighting the amusing confusion surrounding AI naming conventions.
- **Token Use: Emojis vs. Text**: A user humorously declared their command to an AI to stop using emojis in responses, arguing that they **waste tokens**.
   - *“Breaking News...”* they exclaimed, acknowledging the ongoing debate around efficient token usage.
- **OpenAI COO's Future in Question**: A post referenced a prediction from The Information about **OpenAI's COO Brad Lightcap** potentially leaving the company in **2025** due to a diminished role.
   - This change is seen as part of a trend of incorporating experienced public-company executives into leadership.
- **Chat about AI Community Interactions**: Comments teased about the dynamics within AI roles, with a user expressing enjoyment in engaging with **RL** (reinforcement learning) people.
   - They quoted Jonathan Frankle, who humorously emphasized the fun of hanging out with that crowd.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/rajammanabrolu/status/1877058858599825585">Tweet from Prithviraj (Raj) Ammanabrolu (@rajammanabrolu)</a>: my job here is doneQuoting Jonathan Frankle (e/🧱) (@jefrankle) @rajammanabrolu @DbrxMosaicAI Speak for yourself. I just think RL people are really fun to hang out with.</li><li><a href="https://x.com/btibor91/status/1877033734282817939">Tweet from Tibor Blaho (@btibor91)</a>: The Information predicts OpenAI&#39;s COO Brad Lightcap will leave in 2025, based on his reduced role after Sarah Friar and Giancarlo Lionetti took over his finance and sales teams in 2024, part of a ...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1326282089406005318)** (7 messages): 

> `NVIDIA's Performance, Orin in Robotics, Community Support in Open Source, Anthropic Research on AI Alignment` 


- **Concerns over NVIDIA's Performance**: A member expressed doubt about NVIDIA's performance, humorously suggesting it might be best seen at zero, questioning whether the operating system is indeed problematic.
   - This sentiment was followed by speculation that the member is aiming to protect others from potential mistakes in using NVIDIA products.
- **Praise for Orin in Robotics**: A member remarked that the **Orin** is a powerful tool for robotics, describing it as an 'absolute beast' that requires careful handling.
   - This comment underscores the challenges associated with harnessing high-performance technology in practical applications.
- **Nonprofit Support for Open Source Community**: A member offered to assign a teammate to a task, highlighting their nonprofit's mission to promote open source and community assistance.
   - This showcases their commitment to knowledge-sharing and collaboration, despite feeling exhausted from extensive interaction with AI.
- **AI Alignment Insights from Anthropic**: A timestamped [YouTube video](https://youtu.be/IPmt8b-qLgk?si=Cg2M9u4Rc5X7MHwb&t=964) from an Anthropic Research Salon discusses AI alignment, with a focus on how base models are shaped into agents.
   - A remark made during the discussion raised curiosity about the shaping process and what kind of data influences the model pretraining.
- **Debate on Interpretation of Josh Batson's Comment**: The community engaged in a discussion about a comment made by **Josh Batson** regarding shaping base models, with members pondering its implications.
   - One member insisted that the phrasing seemed too deliberate to be a simple misspeak, prompting deeper analysis of the topics discussed.



**Link mentioned**: <a href="https://youtu.be/IPmt8b-qLgk?si=Cg2M9u4Rc5X7MHwb&t=964"> - YouTube</a>: no description found

  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1326364932400218132)** (2 messages): 

> `Nextcloud Community Support, Ai2's Work, Open Source Contribution` 


- **Call to Action for Nextcloud's Community Support**: A member emphasized that to benefit from **Ai2's work**, the community should find ways to **give back**.
   - They expressed disappointment in the **limited community support** for **Nextcloud**, suggesting that users should contribute more to promote it.
- **Nextcloud's Challenge with Limited Support**: It was noted that **Nextcloud GmbH** is focused on improving the platform for institutional clients and lacks the bandwidth to promote it to the general public.
   - The member expected the community to **step up** and provide additional support, as they are fans of the platform.
- **Community Spirit for Nextcloud**: Another member showed solidarity by stating they are **praying** for the **Nextcloud** and its **OSS community**.
   - This sentiment highlights a shared hope for increased community involvement and support for Nextcloud.


  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/)** (1 messages): 

SnailBot News: <@&1216534966205284433>
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1326308028374323261)** (20 messages🔥): 

> `LLaMA Fine-Tuning, Censorship in AI Responses, Corporate Influence in Politics, Modern Guilds Concept, Custom GPT Model Behaviors` 


- **Users fine-tune LLaMA on personal data**: A member shared their experience of fine-tuning **LLaMA** using own structured data and stated, 'it's pretty easy.'
   - This raises questions about how many users are leveraging their personal texts for model training.
- **Censorship of political questions in AI**: Members discussed the limitations faced when asking **Gemini** about political questions, with one noting, 'most LLMS won't answer political questions.'
   - Another raised concerns about who defines what is considered 'politics' in AI interactions.
- **Rethinking corporate influence in politics**: A member referenced the **Federal Election Campaign Act (FECA)**, noting how it legitimized corporate influence in politics.
   - They humorously remarked that stopping corporatism is '50 years too late,' indicating a resigned perspective on this issue.
- **Proposal for modern guilds**: One member suggested a revival of **medieval guilds**, proposing it as a self-organizing alternative for creative professions.
   - They expressed a humorous intent to design a guild logo for AI and video game developers using tools like DALL-E and ChatGPT.
- **Insights on Custom GPT model outputs**: A user noted that their **Custom GPT** model produced dual responses, highlighting one was formatted as o1 with 'thinking.'
   - This sparked a discussion about the consistency of model responses, questioning how different models apply guidance within A/B testing.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1326350266416500747)** (7 messages): 

> `Ubuntu 24.04.1, ROCm 6.3.1, Ollama 3.2 Vision, O1 Pro upgrade, Concept clarifier GPT` 


- **Seeking Guides for GPU 4o Mini on Ubuntu 24.04.1**: A user with **Ubuntu 24.04.1** and a **6900XT** expressed interest in trying **GPU 4o Mini** and is looking for good guides.
   - *Previous experiences with **Ollama 3.2 Vision** and having **ROCm 6.3.1** installed* were cited.
- **Is O1 Pro Worth the Upgrade?**: A user questioned the value of upgrading to **O1 Pro**, prompting responses about its effectiveness for complex tasks.
   - Another member mentioned that *if you need it and use it for complex tasks, it’s likely worth the upgrade*.

  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1326315684157980754)** (7 messages): 

> `Prompt Instruction Vague, Style Naming in Prompts, Completion Quality Concerns` 


- **Vague Instructions Lead to Low Completions**: A member cited that **80% completion** is low and suggested it might depend on **input size** and **relative noise** in the instructions.
   - *It’s feasible your instructions are vague,* as noted by another member.
- **Naming Styles in Prompts**: There was a suggestion that you can just **name the style** in the prompt and hope for the best.
   - However, another member expressed disappointment that this method *doesn't work*, indicating a deeper issue.
- **Encouragement Amidst Challenges**: After discussing challenges in prompt style, one member wished another good luck in **finding a solution**.
   - This echoed a supportive tone within the chat despite the frustrations expressed.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1326315684157980754)** (7 messages): 

> `Prompt Engineering, Instruction Clarity, Completion Rates` 


- **Clarifying Prompt Styles Doesn't Guarantee Success**: A member noted that simply naming the **style** in the prompt does not assure desired outcomes, stating, 'Yeah, it doesn’t work, sadly.'
   - This highlights the **challenges** faced in obtaining consistently effective results through vague instructions.
- **80% Completion Rate Sparks Debate**: Another member observed that an **80% completion** rate is considered low, especially when factoring in **input size** and **relative noise**.
   - This sparked discussions about the influence of **instruction clarity** on overall performance and output quality.
- **Concern Over Self-Promotion Rules**: A user cautioned about potential violations of the channel's **self-promotion** policy, indicating a community focus on maintaining guidelines.
   - This reflects ongoing concerns regarding appropriate conduct within the discussion forum.
- **Troubleshooting Completion Issues**: A member received feedback suggesting that their vague instructions might be contributing to lower completion rates.
   - Supportive comments encouraged continued efforts to refine prompts in search of solutions.


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1326655467577147412)** (1 messages): 

> `CSV Download Feature` 


- **Download Tables as CSV Files Now Available**: Members can now conveniently download tables as **CSV files** by selecting the download option found in responses that include tables.
   - An example image was shared to illustrate this new feature, ensuring users are informed and can utilize it effectively.
- **CSV Download Illustration Provided**: An attached image labeled 'download_csv.jpg' demonstrates how to access the CSV download feature.
   - This visual aid is aimed at helping members locate the download option seamlessly.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1326351055109230713)** (23 messages🔥): 

> `Subscription Options, Performance Issues, Application Integration, File Upload Errors, Voice Functionality` 


- **Users Desire Permanent Subscription**: Several users expressed enthusiasm for a permanent subscription option for the Perplexity app, indicating they love the service.
   - One user stated, *'I wish there was a permanent subscription I love this app so much.'*
- **Lag and Input Delay Frustrations**: Multiple members reported experiencing significant **input lag** while typing, often affecting their productivity.
   - Complaints included one user noting, *'I write text and it takes 1 sec to show up in input field and is so slow.'*
- **Request for Improved Voice Features**: Users criticized the current voice functionality, mentioning that it is too slow and lacks options to speed up narration.
   - One user remarked, *'Perplexity voice is too slow... especially Alex,'* highlighting dissatisfaction with the feature.
- **File Upload and API Issues**: A user described encountering an error prompt regarding file uploads, despite not intending to upload anything.
   - They stated that, *'Whenever I copy text the 
- **Interest in Perplexity Integration**: Discussion arose around companies integrating Perplexity into office suites for content creation, praising its output over competitors.
   - One user concluded that integrating into workflows like **MS 365 Copilot** could enhance functionality and efficiency.



**Link mentioned**: <a href="https://medium.com/design-bootcamp/youzu-ai-where-ai-interior-design-meets-real-world-shopping-76a066be3688">Youzu.ai: Where AI Interior Design Meets Real-World Shopping</a>: Introducing the world’s first Design-to-Buy platform, powered by AI✨

  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1326295840981254298)** (15 messages🔥): 

> `AI Superintelligence, Nuclear Power Purchases, Nvidia's Personal AI, Healthiest Cooking Oils, React JS Learning Resources` 


- **Sam Altman discusses AI Superintelligence**: A video titled '[YouTube](https://www.youtube.com/embed/WF_gKD_MxSo)' features Sam Altman discussing **AI superintelligence** and its implications, alongside recent significant developments.
   - The conversation also touches upon the **U.S. buying record levels** of nuclear power, showcasing a mixed outlook on energy strategies.
- **Exploring Healthiest Cooking Oils**: Multiple links were shared regarding the **healthiest cooking oils**, providing insights into their nutritional value and culinary uses.
   - *One source denoted the importance of understanding different oils* and their impact on health, emphasizing data-driven choices.
- **Learning React JS with Useful Resources**: A link was shared on how to effectively **learn React JS**, presenting educational material that caters to various learning styles.
   - *The resource aims to help beginners grasp the fundamentals* of React and build upon them with practical exercises.
- **Updates on the Amethyst Tablet PDF**: Members discussed a link containing the **Amethyst Tablet PDF**, which may offer insights into its historical and cultural significance.
   - *A detailed exploration of the tablet's contents* might provide context around its production and discovery.
- **Using Discord's OAuth2 Flow**: A member shared a link about **using Discord's OAuth2 flow** to integrate applications, which can streamline user authentication.
   - *This resource is particularly useful for developers* aiming to enhance their apps with robust security features.



**Link mentioned**: <a href="https://www.youtube.com/embed/WF_gKD_MxSo">YouTube</a>: no description found

  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1326570802111582249)** (3 messages): 

> `NCU profile comparison, Community welcome` 


- **NCU profile comparison yields insights**: A member suggested that comparing an **NCU profile** of a **32x32** versus **16x16** configuration should provide clarity on performance differences.
   - This approach may help in understanding the impact of configuration changes on results.
- **New member expresses hope for inclusion**: A new member joined the channel, expressing hope to be welcomed by the community.
   - This highlights the supportive atmosphere and openness to newcomers.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1326299550662787103)** (9 messages🔥): 

> `Using wgmma for MMAs, GPU warmup importance, Benchmark timing, Fused MLP implementations, On-chip MLP usage` 


- **Ensure wgmma Usage in MMAs**: A member noted that using **wgmma** for **MMAs** requires kernel tiles to be at least **64** to split computations over **4 warps** with a minimum size of **16**.
   - This indicates that simply checking ptx for **mma.sync** instead of **wgmma.mma.async.sync** could lead to performance issues.
- **Warmup the GPU to Optimize Performance**: A discussion emerged about the necessity of **warming up** the GPU, as it does not always operate at maximum clock speed and throttles based on usage.
   - Members humorously debated the rationale behind using **25ms** for warmup time, with one suggesting that a value like **1** is clearly insufficient.
- **Default Benchmark Timing Review**: In reference to the default benchmark of **100ms**, it was pointed out that **25ms** accounts for only **25%** of that time, indicating a potential for optimization.
   - This has implications for how efficiently resources are being utilized during benchmarks and tests.
- **Inquiry about Fused MLP Implementations**: A member inquired about existing **Triton** implementations of the **fused MLP** found in [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).
   - They also questioned the limited use of on-chip MLPs in applications, speculating if the size is a barrier.
- **On-Chip MLP Applications Considered Small**: The discussion touched on why on-chip **MLP** applications may not see significant usage due to their seemingly **tiny** nature.
   - This led to questions about applicability in broader contexts beyond current implementations.



**Link mentioned**: <a href="https://github.com/NVlabs/tiny-cuda-nn">GitHub - NVlabs/tiny-cuda-nn: Lightning fast C++/CUDA neural network framework</a>: Lightning fast C++/CUDA neural network framework. Contribute to NVlabs/tiny-cuda-nn development by creating an account on GitHub.

  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1326407335857356943)** (2 messages): 

> `Cutlass Kernel Performance, Diffing Generated PTX and SASS` 


- **Cutlass kernel shows slower performance with bfloat16**: In discussing the **Cutlass kernel**, it was noted that **bfloat16** performance is approximately **10% slower** than that of **half** data type despite similar template parameters.
   - Members were encouraged to inspect aspects of Cutlass that might contribute to this performance difference, questioning if it was a reasonable outcome.
- **Using Diff Tools to Compare PTX and SASS**: One user suggested using **meld** with a filter to compare generated **PTX** or **SASS**, specifically to ignore register names for clarity.
   - This approach aims to streamline the comparison process and help identify meaningful differences without being distracted by irrelevant details.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

drisspg: https://hipscript.lights0123.com/
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1326539856494723195)** (3 messages): 

> `Compact PC Benefits, Gaming Laptop vs Desktop Size, Thermal Performance Concerns` 


- **Compact PCs make a bold statement**: A member noted that the **compact form** of certain PCs appears much more impactful compared to traditional gaming desktops.
   - *Joking aside*, this design shift could potentially attract more users looking for efficiency and aesthetics.
- **Gaming Laptops save space compared to desktops**: Another member highlighted the **significant size reduction** of gaming laptops in comparison to desktops, which allows for better space management.
   - This efficiency arises because laptops don't require the same modular and replaceable components as desktops.
- **Concerns about thermal performance**: There was speculation regarding the **thermal performance** of compact PCs due to their limited space for airflow and heat dissipation.
   - The discussion hinted at possible challenges in maintaining optimal performance while keeping components compact.


  

---


### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/)** (1 messages): 

iron_bound: https://hipscript.lights0123.com/
  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1326304924220522617)** (11 messages🔥): 

> `Discord based leaderboard, Alpha users recruitment, Fastest softmax kernel competition, GPU Glossary materials, Kernel coding` 


- **Seeking Alpha Users for Discord Leaderboard**: A member announced the search for **alpha users** for a new [Discord based leaderboard](https://linkexample.com) that integrates GPUs for competing on specific kernels.
   - They encouraged interest and promised to follow up with a tutorial for those who respond.
- **Competition with Fastest Softmax Kernel Launch**: An exciting opportunity arose as the alpha competition for the **fastest softmax kernel** is now live on the staging server, inviting participants to reach out directly for invites.
   - Members expressed eagerness to join, emphasizing the fun and relevance for beginner users.
- **GPU Glossary Resources Shared**: A user shared the **gpu-glossary.zip** containing all GPU Glossary materials formatted as Markdown files, with paths indicated in the attached [contents.json](https://link.to.contents.json).
   - This resource serves as a valuable reference for users involved in GPU discussions and projects.
- **Effortless Kernel Code Contributions**: Discussion highlighted that participating in the leaderboard mainly involves building and experimenting with simple kernel code, avoiding complex bot coding.
   - Members were invited to contribute, even with minor inputs, to support the initiative during its early phase.
- **Dedicated Server for Discussing Contributions**: There is a separate Discord server for organizing efforts about the leaderboard, making it easier to coordinate tasks and discuss contributions.
   - Mark has pinned this server link in the channel for ease of access and collaboration.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1326626156111466585)** (4 messages): 

> `Thunderkittens vs Flash Attention 3, Reproducing plots, Collaboration on kernels` 


- **Comparing Thunderkittens with Flash Attention 3**: A user inquired about the script used to compare **Thunderkittens** with **Flash Attention 3** and produce a plot, specifically referencing an [image of the plot](https://github.com/HazyResearch/ThunderKittens/blob/main/assets/attn.png).
   - They requested guidance on how to reproduce this comparison in their own work.
- **Script for Reproducing Plots Available**: Another user responded, providing a link to the [code repository](https://github.com/HazyResearch/ThunderKittens/tree/main/tests/python) that contains the necessary script to reproduce the results.
   - This indicates that users can easily access and utilize the existing resources for their analyses.
- **Call for Collaboration on Kernels**: The same user expressed interest in collaborating on a variety of kernels such as **MoE** and **Deep Seek Attention** and welcomed participants to contribute to the repository.
   - They encouraged others to reach out for collaboration or for anyone interested in learning about **Thunderkittens**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/HazyResearch/ThunderKittens/blob/main/assets/attn.png">ThunderKittens/assets/attn.png at main · HazyResearch/ThunderKittens</a>: Tile primitives for speedy kernels. Contribute to HazyResearch/ThunderKittens development by creating an account on GitHub.</li><li><a href="https://github.com/HazyResearch/ThunderKittens/tree/main/tests/python">ThunderKittens/tests/python at main · HazyResearch/ThunderKittens</a>: Tile primitives for speedy kernels. Contribute to HazyResearch/ThunderKittens development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1326413614940426322)** (1 messages): 

> `Shard counts adjustment, File generation process` 


- **Shard counts need increasing**: A member reported making good progress but identified the need to **increase shard counts** for several generated files that are already sharded.
   - This adjustment is crucial for improving the overall efficiency of file handling in their current process.
- **Generated files are sharded**: The same member mentioned that multiple **generated files** are currently sharded, necessitating a review of their configurations for optimal performance.
   - Ensuring proper sharding is essential for maintaining **data distribution** and quick access.


  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1326490613943894060)** (3 messages): 

> `Community Check-in` 


- **Community Check-in Signals Positivity**: Members greeted each other, initiating a friendly check-in about how everyone is doing.
   - *We are good! Hbu?* reflects a warm, engaging community vibe.
- **Initiating Friendly Conversations**: Axelbolston opened the discussion by asking how everyone is doing, promoting a positive atmosphere.
   - This engagement reinforces a sense of community among members.


  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1326466567071858688)** (2 messages): 

> `Token Usage Export` 


- **Token Usage Export Query**: A user inquired about the possibility of exporting **token usage** to a file.
   - Another member responded that **token usage** for each request can be logged, providing a way to track it.
- **Logging Token Usage Details**: It was mentioned that users can monitor their token usage for each individual request they make.
   - This approach could help in maintaining a record of **token usage** without needing a formal export feature.


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1326682963806519296)** (7 messages): 

> `Cohere LLM API, Token Budget Concerns, Model Specifications, Recursive Loop Issue, Max Token Configuration` 


- **Recursive Loop Issue with Cohere LLM API**: A member reported experiencing an issue with the **Cohere LLM API** where the model sometimes gets stuck in a recursive loop, which could quickly consume their token budget.
   - They inquired if anyone else encountered this issue and suggested implementing safeguards, like an upper bound on the response stream.
- **Need for Model Specifications Clarified**: Another member questioned which model was being used and asked for clarification on its generation specifications, specifically regarding language support.
   - The model mentioned was `command-r-plus-08-2024`, with potential support for **Persian**.
- **Discussion on Token Management**: Members discussed the need to manage the token budget effectively, with one suggesting the option to set a maximum token limit.
   - This approach aims to prevent runaway generation from affecting costs significantly.
- **Quality of Language Generation**: There was a comment regarding the **Cohere model's** capacity to generate in different languages, suggesting that not all languages are equally supported.
   - One member emphasized the importance of understanding the model's capabilities before extensive use.


  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1326466367360208970)** (23 messages🔥): 

> `Exporting Token Usage, Cohere Documentation Search` 


- **Bot's Struggle with Token Usage Export Details**: A user sought information on how to **export token usage to a file**, prompting the bot to initiate searches in Cohere's documentation.
   - Despite multiple attempts with variations like 'export token usage to CSV' and 'export token usage to JSON', the bot **did not find any relevant information**.
- **Bot's Repeated Queries Yield No Results**: The bot consistently searched for terms related to **exporting token usage** but was unable to provide a definitive answer.
   - After several queries, including **'export token usage to file'**, the bot reached its tool call limit without successful results.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1326288668872282123)** (30 messages🔥): 

> `FP4 Wars, State of the Art Open Source TTS, Omi Wearable Technology, Salesforce Hiring Freeze, New Directions in LLM Products` 


- **FP4 Wars Spark Controversy**: The FP4 wars continue to stir debate, with many questioning NVIDIA's benchmarks and comparisons between FP4 and FP8, suggesting discrepancies in the data presented.
   - *Jensen's pitch* of FP4 as a training metric is viewed as ahead of its time, especially with ongoing discussions about FP8's impact on model quality at inference time.
- **Quality Issues in Open Source TTS**: While exploring open source text-to-speech models, a user noted that the output quality is still *slightly robotic* and has noticeable cadence problems.
   - Despite attempting various examples, it appears that superior cloning requires more refined recording inputs to improve fidelity.
- **The Omi Wearable Raises Eyebrows**: A new wearable called *Omi* is introduced, designed to read brain data with a separate module expected in 2025, which has left many users intrigued and a bit skeptical.
   - Comments suggest that advancements such as *microchips* being integrated into wearables evoke concerns reminiscent of *Black Mirror* episodes.
- **Salesforce Freezes Software Engineer Hiring**: Salesforce's Marc Benioff announced that no new software engineers will be hired in 2025, attributing this to productivity boosts from their AI product, *Agentforce*.
   - *Benioff* believes overall company size may increase, but it's clear that AI is reshaping workforce dynamics in the organization.
- **Shift Towards LLM-Centric Products**: There’s a growing sentiment that large organizations will struggle to adapt to advanced paradigms, leading to more opportunities for agile startups.
   - Experts noted that existing products with *LLM integration* are underperforming, while those built from the ground up are experiencing unprecedented growth.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://sagipolaczek.github.io/NeuralSVG/">NeuralSVG: An Implicit Representation for Text-to-Vector Generation</a>: no description found</li><li><a href="https://x.com/yuchenj_uw/status/1876680855630094725?s=46">Tweet from Yuchen Jin (@Yuchenj_UW)</a>: I love Nvidia and Jensen, but their presentation of numbers bothers me:- vague terms like &#34;AI TOPS&#34;- compare FP4 on 5090 with FP8 on 4090- show FP4 FLOPS and claim a $3,000 box runs a 200B mod...</li><li><a href="https://huggingface.co/collections/unsloth/deepseek-v3-all-versions-677cf5cfd7df8b7815fc723c">Deepseek V3 (All Versions) - a unsloth Collection</a>: no description found</li><li><a href="https://www.salesforceben.com/salesforce-will-hire-no-more-software-engineers-in-2025-says-marc-benioff/">Salesforce Will Hire No More Software Engineers in 2025, Says Marc Benioff</a>: Salesforce CEO Marc Benioff announces no new software engineer hires – see how AI is shaping the company&#039;s future.</li><li><a href="https://x.com/sytelus/status/1877015492126220594">Tweet from Shital Shah (@sytelus)</a>: We have been completely amazed by the response to phi-4 release. A lot of folks had been asking us for weight release. Few even uploaded bootlegged phi-4 weights on HuggingFace😬.Well, wait no more. W...</li><li><a href="https://x.com/tsarnick/status/1877089046528217269">Tweet from Tsarathustra (@tsarnick)</a>: François Chollet says OpenAI&#39;s o1 model is running a search process in the space of possible chain of thought, generating a natural language program and adapting to novelty in a &#34;genuine break...</li><li><a href="https://www.salesforceben.com/salesforce-will-hire-no-more-software-engineers-in-2025-says-marc-beni">Salesforce Will Hire No More Software Engineers in 2025, Says Marc Benioff</a>: Salesforce CEO Marc Benioff announces no new software engineer hires – see how AI is shaping the company&#039;s future.</li><li><a href="https://x.com/kodjima33/status/1877017546697699363">Tweet from Nik Shevchenko (@kodjima33)</a>: introducing omi. thought to action.order now at http://omi.me</li><li><a href="https://www.jointakeoff.com/">Takeoff</a>: no description found
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1326346155029692498)** (3 messages): 

> `Cohere integration with LlamaIndex, LlamaIndex Workflows in AI, GitHub Event on AI Agents` 


- **Cohere models now easier with LlamaIndex**: Cohere refreshed their [documentation](https://t.co/dLKGgkqOe8) for using their models with LlamaIndex, including commands for installing necessary packages.
   - To utilize this, ensure you have the Cohere SDK and a trial API key available on your dashboard.
- **Deep Dive into LlamaIndex Workflows**: Lingzhen Chen demonstrates how to leverage LlamaIndex Workflows to search and summarize academic papers from ArXiv in a [recent deep dive](https://t.co/OXU8DcUb5E).
   - This approach crystallizes an LLM-powered process into a controllable, repeatable format that enhances AI interactions.
- **Exciting GitHub Event on AI**: An event at GitHub HQ on Jan 15th will feature expert talks on debugging AI agents, creating fast inference systems, and building workflows with LlamaIndex.
   - This gathering promises hands-on insights and networking opportunities for AI enthusiasts and professionals alike, check out the [event link](https://t.co/GnYXYqJfth).



**Link mentioned**: <a href="https://t.co/dLKGgkqOe8">LlamaIndex — Cohere</a>: Learn how to use Cohere and LlamaIndex together to generate responses based on data.

  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1326469930484502559)** (17 messages🔥): 

> `Metadata Management in LlamaIndex, Evaluation Times for FaithfulnessEvaluator, API Token Sharing, Python Dependency Conflicts` 


- **Managing Metadata in LlamaIndex**: A user inquired about excluding unnecessary keys in LlamaIndex nodes, referencing the setting `document.excluded_embed_metadata_keys = ['key']` but noting it does not remove keys from the node storage.
   - Another member clarified that to fully remove keys, you can iterate over the document metadata and configure it before indexing.
- **FaithfulnessEvaluator slowed after model updates**: After updating to a larger **bge_onnx** embedding model version, a user observed that the **FaithfulnessEvaluator's** first evaluation took over **25 seconds**, while subsequent evaluations dropped to around **1 second**.
   - They sought insights into the delay for the first evaluation and optimization strategies for the process.
- **OpenAI API Token Sharing**: A user shared their API token for a service, stating it supports **o1-mini** and **o1-preview** models for limited-time free use.
   - Concerns about token usage arose, with members discussing whether the token was stolen or legitimately theirs.
- **Daily Python Dependency Conflicts**: A user lamented about encountering another **dependency conflict** in Python, highlighting recurring frustration.
   - This led to a brief exchange about the ongoing challenge of managing dependencies in programming.


  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1326282411889131653)** (13 messages🔥): 

> `AI21 Labs and crypto, Using Jamba for coding assistance, AI's coding capabilities, Podcast app development, Exploring programming with Jamba` 


- **AI21 Labs IS not associated with crypto scams**: Members stressed that any tokens or discussions related to crypto are **not affiliated with AI21 Labs** and warned that continued discussions would lead to bans.
   - *This discord is for developer support and generative AI models, not for crypto discussions.*
- **Jamba's benefits for developers**: One member shared how they utilize **Jamba** to assist with coding, revealing they built a Python app to manage podcast episode transcripts.
   - *They found Jamba's Conversational RAG very helpful, enhancing their experience and productivity in development.*
- **Experiences with AI in coding**: A newcomer expressed their excitement about AI's ability to **generate code** but mentioned encountering humorous mistakes during coding sessions.
   - They highlighted having used AI for **HTML, Javascript, and PHP troubleshooting**, noting the ongoing evolution of AI capabilities.
- **Connecting with Jamba as a learning tool**: A user shared their journey in getting connected to **Jamba**, noting it has made programming tasks easier with its conversational arrays.
   - They compared Jamba's API functionality to tools like **deepSeek** and **openAi**, enhancing their local machine IRC bot coding efforts.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1326290070126329876)** (12 messages🔥): 

> `Certificate Declaration Form, Email Consistency for Certificates, Spring 2025 Course Details, Twitter Account Verification for Certificates, Certificate Availability Timeline` 


- **Thankful Responses for Certificate Declaration Form**: Multiple members expressed gratitude towards <@854134294870884363> for opening the [declaration form](https://link.to.form) to submit their information.
   - *
- **Email Must Match for Certificate Confirmation**: A member noted that you must use the **same email** for the declaration form as used for assignments to ensure proper tracking for certificate issuance.
   - Another user confirmed they listed their original email despite submitting with a different address to avoid issues.
- **Spring 2025 Course to Follow F24**: It was confirmed that the Spring 2025 course will kick off in **late January** and will continue from the **F24** course materials.
   - This signals continuity for participants familiar with prior content.
- **Verification of Twitter Account for Certificates**: A member reported their Twitter account got suspended but provided a link to their article on **Medium** for verification.
   - They inquired about how to secure their certificate despite their Twitter suspension at the time of submission.
- **Certificates Not Yet Released**: <@854134294870884363> was asked about the release status of certificates, to which it was stated that no one has received theirs yet.
   - It is likely that **certificates won't be ready until the end of January**.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1326461794029539419)** (6 messages): 

> `Long context issues, Hide demo fields parameter, Framework improvements` 


- **Experimenting with 'hide_demo_fields' parameter**: A member shared their experience experimenting with a 'hide_demo_fields' parameter to replace certain fields in demos with a placeholder like '... omitted for brevity ...'. This approach aims to mitigate the prompt bloat caused by the `codebase_context` field.
   - An attached image shows the results of this method in action, emphasizing the goal of maintaining clarity in demos.
- **Desire for a general pattern for context management**: Another member expressed interest in finding a general clean pattern for handling long context issues effectively.
   - They acknowledged that while the proposed solution makes sense, it may require simplification for broader applicability.
- **Call for framework-level solutions**: Discussion centered on the need for framework-level features to handle long context issues, rather than relying on workarounds.
   - One member suggested that this functionality might be more effective if it were integrated directly into the signature definition of the framework.


  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1326450857134526498)** (2 messages): 

> `Vertex AI models, DSPy integration, Inference processes` 


- **Integrating Vertex AI Models with DSPy**: A member sought guidance on how to add **Vertex AI models** and perform inference using **DSPy**.
   - This inquiry reflects ongoing interest in leveraging Vertex AI's capabilities within the DSPy framework.
- **Framework for Function Calls in DSPy**: Another member inquired if any framework has been created to handle **function calls** and basic inference processes while using **DSPy** with **Vertex AI**.
   - This highlights a need for streamlined processes and tools that can enhance the use of DSPy in conjunction with Vertex AI.


  

---


### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1326393158485151826)** (6 messages): 

> `Open Interpreter Production Setup, Prompting Techniques for Code Generation, Custom Instructions for Model Performance, NVIDIA Grace Blackwell AI Supercomputer` 


- **Seeking Optimal Production Setup for Open Interpreter**: A member is inquiring about the best configurations for Open Interpreter, including model selection, default settings, and performance tweaks for optimal use.
   - Despite extensive use, they noted a lack of shared configurations that others have found successful.
- **Effective Prompting Techniques Requested**: There is a request for insights on the most effective prompting techniques to achieve accurate code generation.
   - The user is looking for practical tips that can enhance the AI’s output in coding scenarios.
- **Custom Instructions to Enhance Model Performance**: A discussion is in place regarding the configuration of custom instructions to improve the model's performance for specific tasks.
   - Exploring optimal setups would provide better outcomes during model usage.
- **NVIDIA Grace Blackwell Supercomputer Announcement**: [NVIDIA](https://www.nvidia.com/en-us/project-digits/) introduced Project DIGITS, featuring the **Grace Blackwell Superchip**, designed to deliver a petaflop of AI performance in a compact format.
   - Developers can prototype and fine-tune large models of up to **200B parameters** locally, with a powerful software stack preinstalled.



**Link mentioned**: <a href="https://www.nvidia.com/en-us/project-digits/">NVIDIA Project DIGITS: The World’s Smallest AI Supercomputer. </a>: Reserve yours today.

  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/)** (1 messages): 

davidlandstarop1: God bless us all <@1075395291869614122>
  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/)** (1 messages): 

davidlandstarop1: Safety first <@1221270473355038720>
  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1326634070028980237)** (1 messages): 

> `Dual 3090 Setup, Fine-tuning LLM on Music Notation` 


- **Seeking Help for LLM Fine-Tuning**: A member with a **dual 3090 setup** expressed interest in **fine-tuning an LLM** specifically on music notation.
   - They reached out to see if anyone would be *willing to assist* with this project.
- **Dual GPU Discussion**: The discussion centered around having a **dual 3090 setup**, which indicates high computational power for machine learning tasks.
   - This setup is ideal for training models, especially when handling tasks such as music LLM notation.


  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/)** (1 messages): 

rom1504: Is there any good open  tool registry for building agents ?
  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/)** (1 messages): 

jovial_lynx_74856: Anyone here tried finetuning ModernBERT?
  

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
